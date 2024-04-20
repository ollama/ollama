// -*- mode:c++;indent-tabs-mode:nil;c-basic-offset:4;coding:utf-8 -*-
// vi: set et ft=c++ ts=4 sts=4 sw=4 fenc=utf-8 :vi
//
// Copyright 2024 Mozilla Foundation
//
// Permission is hereby granted, free of charge, to any person obtaining
// a copy of this software and associated documentation files (the
// "Software"), to deal in the Software without restriction, including
// without limitation the rights to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to whom the Software is furnished to do so, subject to
// the following conditions:
//
// The above copyright notice and this permission notice shall be
// included in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
// BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
// ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

//
//                   _   _          ___ _      _   ___
//                  | |_(_)_ _ _  _| _ ) |    /_\ / __|
//                  |  _| | ' \ || | _ \ |__ / _ \\__ \.
//                   \__|_|_||_\_, |___/____/_/ \_\___/
//                             |__/
//
//                    BASIC LINEAR ALGEBRA SUBPROGRAMS
//
//
// This file implements multithreaded CPU matrix multiplication for the
// common contiguous use case C = Aᵀ * B. These kernels are designed to
// have excellent performance[1] for matrices that fit in the CPU cache
// without imposing any overhead such as cache filling or malloc calls.
//
// This implementation does not guarantee any upper bound with rounding
// errors, which grow along with k. Our goal's to maximally exploit the
// hardware for performance, and then use whatever resources remain for
// improving numerical accuracy.
//
// [1] J. Tunney, ‘LLaMA Now Goes Faster on CPUs’, Mar. 2024. [Online].
//     Available: https://justine.lol/matmul/. [Accessed: 29-Mar-2024].

#pragma GCC diagnostic ignored "-Wpedantic"
#pragma GCC diagnostic ignored "-Wignored-attributes"

#include "sgemm.h"
#include "ggml-impl.h"
#include "ggml-quants.h"

#ifdef _MSC_VER
#define NOINLINE __declspec(noinline)
#else
#define NOINLINE __attribute__((__noinline__))
#endif

#if defined(__ARM_NEON) || defined(__AVX512F__)
#define VECTOR_REGISTERS 32
#else
#define VECTOR_REGISTERS 16
#endif

// there will be blocks
#define BEGIN_KERNEL(RM, RN) \
    int ytiles = (m - m0) / RM; \
    int xtiles = (n - n0) / RN; \
    int tiles = ytiles * xtiles; \
    int duty = (tiles + nth - 1) / nth; \
    int start = duty * ith; \
    int end = start + duty; \
    if (end > tiles) \
        end = tiles; \
    for (int job = start; job < end; ++job) { \
        int i = m0 + job / xtiles * RM; \
        int j = n0 + job % xtiles * RN;

#define END_KERNEL() }

#define MM256_SET_M128I(a, b) _mm256_insertf128_si256(_mm256_castsi128_si256(b), (a), 1)

namespace {

inline float unhalf(ggml_fp16_t d) {
    return GGML_FP16_TO_FP32(d);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// VECTORIZED ARITHMETIC OPERATIONS

#if defined(__SSE__) || defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
inline __m128 add(__m128 x, __m128 y) { return _mm_add_ps(x, y); }
inline __m128 sub(__m128 x, __m128 y) { return _mm_sub_ps(x, y); }
inline __m128 mul(__m128 x, __m128 y) { return _mm_mul_ps(x, y); }
#endif  // __SSE__

#if defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
inline __m256 add(__m256 x, __m256 y) { return _mm256_add_ps(x, y); }
inline __m256 sub(__m256 x, __m256 y) { return _mm256_sub_ps(x, y); }
inline __m256 mul(__m256 x, __m256 y) { return _mm256_mul_ps(x, y); }
#endif // __AVX__

#if defined(__AVX512F__)
inline __m512 add(__m512 x, __m512 y) { return _mm512_add_ps(x, y); }
inline __m512 sub(__m512 x, __m512 y) { return _mm512_sub_ps(x, y); }
inline __m512 mul(__m512 x, __m512 y) { return _mm512_mul_ps(x, y); }
#endif // __AVX512F__

#if defined(__ARM_NEON)
inline float32x4_t add(float32x4_t x, float32x4_t y) { return vaddq_f32(x, y); }
inline float32x4_t sub(float32x4_t x, float32x4_t y) { return vsubq_f32(x, y); }
inline float32x4_t mul(float32x4_t x, float32x4_t y) { return vmulq_f32(x, y); }
#endif // __ARM_NEON

#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
inline float16x8_t add(float16x8_t x, float16x8_t y) { return vaddq_f16(x, y); }
inline float16x8_t sub(float16x8_t x, float16x8_t y) { return vsubq_f16(x, y); }
inline float16x8_t mul(float16x8_t x, float16x8_t y) { return vmulq_f16(x, y); }
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

////////////////////////////////////////////////////////////////////////////////////////////////////
// VECTORIZED HORIZONTAL SUM

#if defined(__ARM_NEON)
inline float hsum(float32x4_t x) {
    return vaddvq_f32(x);
}
#endif // __ARM_NEON

#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) && !defined(_MSC_VER)
inline float hsum(float16x8_t x) {
    return vaddvq_f32(vaddq_f32(vcvt_f32_f16(vget_low_f16(x)),
                                vcvt_f32_f16(vget_high_f16(x))));
}
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

#if defined(__SSE__) || defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
inline float hsum(__m128 x) {
#if defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
    x = _mm_add_ps(x, _mm_movehl_ps(x, x));
    x = _mm_add_ss(x, _mm_movehdup_ps(x));
#else
    __m128 t;
    t = _mm_shuffle_ps(x, x, _MM_SHUFFLE(2, 3, 0, 1));
    x = _mm_add_ps(x, t);
    t = _mm_movehl_ps(t, x);
    x = _mm_add_ss(x, t);
#endif
    return _mm_cvtss_f32(x);
}
#endif

#if defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
inline float hsum(__m256 x) {
    return hsum(_mm_add_ps(_mm256_extractf128_ps(x, 1),
                           _mm256_castps256_ps128(x)));
}
#endif // __AVX__

#if defined(__AVX512F__)
inline float hsum(__m512 x) {
    return _mm512_reduce_add_ps(x);
}
#endif // __AVX512F__

////////////////////////////////////////////////////////////////////////////////////////////////////
// VECTORIZED MEMORY LOADING

template <typename T, typename U> T load(const U *);

#if defined(__ARM_NEON)
template <> inline float32x4_t load(const float *p) {
    return vld1q_f32(p);
}
#if !defined(_MSC_VER)
template <> inline float16x8_t load(const ggml_fp16_t *p) {
    return vld1q_f16((const float16_t *)p);
}
template <> inline float32x4_t load(const ggml_fp16_t *p) {
    return vcvt_f32_f16(vld1_f16((const float16_t *)p));
}
#endif // _MSC_VER
#endif // __ARM_NEON

#if defined(__SSE__) || defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
template <> inline __m128 load(const float *p) {
    return _mm_loadu_ps(p);
}
#endif  // __SSE__

#if defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
template <> inline __m256 load(const float *p) {
    return _mm256_loadu_ps(p);
}
#endif // __AVX__

#if defined(__F16C__)
template <> inline __m256 load(const ggml_fp16_t *p) {
    return _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)p));
}
#endif // __F16C__

#if defined(__AVX512F__)
template <> inline __m512 load(const float *p) {
    return _mm512_loadu_ps(p);
}
template <> inline __m512 load(const ggml_fp16_t *p) {
    return _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)p));
}
#endif // __AVX512F__

////////////////////////////////////////////////////////////////////////////////////////////////////
// ABSTRACTIONS

/**
 * Computes a * b + c.
 *
 * This operation will become fused into a single arithmetic instruction
 * if the hardware has support for this feature, e.g. Intel Haswell+ (c.
 * 2013), AMD Bulldozer+ (c. 2011), etc.
 */
template <typename T, typename U>
inline U madd(T a, T b, U c) {
    return add(mul(a, b), c);
}

/**
 * Computes a * b + c with error correction.
 *
 * @see W. Kahan, "Further remarks on reducing truncation errors,"
 *    Communications of the ACM, vol. 8, no. 1, p. 40, Jan. 1965,
 *    doi: 10.1145/363707.363723.
 */
template <typename T, typename U>
inline U madder(T a, T b, U c, U *e) {
    U y = sub(mul(a, b), *e);
    U t = add(c, y);
    *e = sub(sub(t, c), y);
    return t;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// FLOATING POINT MATRIX MULTIPLICATION

template <int KN, typename D, typename V, typename TA, typename TB, typename TC>
class tinyBLAS {
  public:
    tinyBLAS(int k,
             const TA *A, int lda,
             const TB *B, int ldb,
             TC *C, int ldc,
             int ith, int nth)
        : A(A), B(B), C(C), k(k), lda(lda), ldb(ldb), ldc(ldc), ith(ith), nth(nth) {
    }

    void matmul(int m, int n, int task) {
        if (task == GGML_TASK_TYPE_COMPUTE)
            mnpack(0, m, 0, n);
    }

  private:
    NOINLINE void mnpack(int m0, int m, int n0, int n) {
        int mc, nc, mp, np;
        if (m - m0 <= 0 || n - n0 <= 0)
            return;
        if (VECTOR_REGISTERS >= 32 && n - n0 >= 5 && m - m0 >= 5) {
            mc = 5;
            nc = 5;
            gemm5x5(m0, m, n0, n);
        } else if (n - n0 >= 4 && m - m0 >= 3) {
            mc = 3;
            nc = 4;
            gemm3x4(m0, m, n0, n);
        } else if (n - n0 >= 4) {
            mc = 1;
            nc = 4;
            gemm1x4(m0, m, n0, n);
        } else if (m - m0 >= 4) {
            mc = 4;
            nc = 1;
            gemm4x1(m0, m, n0, n);
        } else {
            mc = 1;
            nc = 1;
            gemm1x1(m0, m, n0, n);
        }
        mp = m0 + (m - m0) / mc * mc;
        np = n0 + (n - n0) / nc * nc;
        mnpack(mp, m, n0, np);
        mnpack(m0, mp, np, n);
        mnpack(mp, m, np, n);
    }

    NOINLINE void gemm5x5(int m0, int m, int n0, int n) {
        BEGIN_KERNEL(5, 5)
        D c00 = {0};
        D c01 = {0};
        D c02 = {0};
        D c03 = {0};
        D c04 = {0};
        D c10 = {0};
        D c11 = {0};
        D c12 = {0};
        D c13 = {0};
        D c14 = {0};
        D c20 = {0};
        D c21 = {0};
        D c22 = {0};
        D c23 = {0};
        D c24 = {0};
        D c30 = {0};
        D c31 = {0};
        D c32 = {0};
        D c33 = {0};
        D c34 = {0};
        D c40 = {0};
        D c41 = {0};
        D c42 = {0};
        D c43 = {0};
        D c44 = {0};
        for (int l = 0; l < k; l += KN) {
            V k0 = load<V>(B + ldb * (j + 0) + l);
            V k1 = load<V>(B + ldb * (j + 1) + l);
            V k2 = load<V>(B + ldb * (j + 2) + l);
            V k3 = load<V>(B + ldb * (j + 3) + l);
            V k4 = load<V>(B + ldb * (j + 4) + l);
            V a0 = load<V>(A + lda * (i + 0) + l);
            c00 = madd(a0, k0, c00);
            c01 = madd(a0, k1, c01);
            c02 = madd(a0, k2, c02);
            c03 = madd(a0, k3, c03);
            c04 = madd(a0, k4, c04);
            V a1 = load<V>(A + lda * (i + 1) + l);
            c10 = madd(a1, k0, c10);
            c11 = madd(a1, k1, c11);
            c12 = madd(a1, k2, c12);
            c13 = madd(a1, k3, c13);
            c14 = madd(a1, k4, c14);
            V a2 = load<V>(A + lda * (i + 2) + l);
            c20 = madd(a2, k0, c20);
            c21 = madd(a2, k1, c21);
            c22 = madd(a2, k2, c22);
            c23 = madd(a2, k3, c23);
            c24 = madd(a2, k4, c24);
            V a3 = load<V>(A + lda * (i + 3) + l);
            c30 = madd(a3, k0, c30);
            c31 = madd(a3, k1, c31);
            c32 = madd(a3, k2, c32);
            c33 = madd(a3, k3, c33);
            c34 = madd(a3, k4, c34);
            V a4 = load<V>(A + lda * (i + 4) + l);
            c40 = madd(a4, k0, c40);
            c41 = madd(a4, k1, c41);
            c42 = madd(a4, k2, c42);
            c43 = madd(a4, k3, c43);
            c44 = madd(a4, k4, c44);
        }
        C[ldc * (j + 0) + (i + 0)] = hsum(c00);
        C[ldc * (j + 0) + (i + 1)] = hsum(c10);
        C[ldc * (j + 0) + (i + 2)] = hsum(c20);
        C[ldc * (j + 0) + (i + 3)] = hsum(c30);
        C[ldc * (j + 0) + (i + 4)] = hsum(c40);
        C[ldc * (j + 1) + (i + 0)] = hsum(c01);
        C[ldc * (j + 1) + (i + 1)] = hsum(c11);
        C[ldc * (j + 1) + (i + 2)] = hsum(c21);
        C[ldc * (j + 1) + (i + 3)] = hsum(c31);
        C[ldc * (j + 1) + (i + 4)] = hsum(c41);
        C[ldc * (j + 2) + (i + 0)] = hsum(c02);
        C[ldc * (j + 2) + (i + 1)] = hsum(c12);
        C[ldc * (j + 2) + (i + 2)] = hsum(c22);
        C[ldc * (j + 2) + (i + 3)] = hsum(c32);
        C[ldc * (j + 2) + (i + 4)] = hsum(c42);
        C[ldc * (j + 3) + (i + 0)] = hsum(c03);
        C[ldc * (j + 3) + (i + 1)] = hsum(c13);
        C[ldc * (j + 3) + (i + 2)] = hsum(c23);
        C[ldc * (j + 3) + (i + 3)] = hsum(c33);
        C[ldc * (j + 3) + (i + 4)] = hsum(c43);
        C[ldc * (j + 4) + (i + 0)] = hsum(c04);
        C[ldc * (j + 4) + (i + 1)] = hsum(c14);
        C[ldc * (j + 4) + (i + 2)] = hsum(c24);
        C[ldc * (j + 4) + (i + 3)] = hsum(c34);
        C[ldc * (j + 4) + (i + 4)] = hsum(c44);
        END_KERNEL()
    }

    NOINLINE void gemm3x4(int m0, int m, int n0, int n) {
        BEGIN_KERNEL(3, 4)
        D c00 = {0};
        D c01 = {0};
        D c02 = {0};
        D c03 = {0};
        D c10 = {0};
        D c11 = {0};
        D c12 = {0};
        D c13 = {0};
        D c20 = {0};
        D c21 = {0};
        D c22 = {0};
        D c23 = {0};
        for (int l = 0; l < k; l += KN) {
            V k0 = load<V>(B + ldb * (j + 0) + l);
            V k1 = load<V>(B + ldb * (j + 1) + l);
            V k2 = load<V>(B + ldb * (j + 2) + l);
            V k3 = load<V>(B + ldb * (j + 3) + l);
            V a0 = load<V>(A + lda * (i + 0) + l);
            c00 = madd(a0, k0, c00);
            c01 = madd(a0, k1, c01);
            c02 = madd(a0, k2, c02);
            c03 = madd(a0, k3, c03);
            V a1 = load<V>(A + lda * (i + 1) + l);
            c10 = madd(a1, k0, c10);
            c11 = madd(a1, k1, c11);
            c12 = madd(a1, k2, c12);
            c13 = madd(a1, k3, c13);
            V a2 = load<V>(A + lda * (i + 2) + l);
            c20 = madd(a2, k0, c20);
            c21 = madd(a2, k1, c21);
            c22 = madd(a2, k2, c22);
            c23 = madd(a2, k3, c23);
        }
        C[ldc * (j + 0) + (i + 0)] = hsum(c00);
        C[ldc * (j + 0) + (i + 1)] = hsum(c10);
        C[ldc * (j + 0) + (i + 2)] = hsum(c20);
        C[ldc * (j + 1) + (i + 0)] = hsum(c01);
        C[ldc * (j + 1) + (i + 1)] = hsum(c11);
        C[ldc * (j + 1) + (i + 2)] = hsum(c21);
        C[ldc * (j + 2) + (i + 0)] = hsum(c02);
        C[ldc * (j + 2) + (i + 1)] = hsum(c12);
        C[ldc * (j + 2) + (i + 2)] = hsum(c22);
        C[ldc * (j + 3) + (i + 0)] = hsum(c03);
        C[ldc * (j + 3) + (i + 1)] = hsum(c13);
        C[ldc * (j + 3) + (i + 2)] = hsum(c23);
        END_KERNEL()
    }

    NOINLINE void gemm1x4(int m0, int m, int n0, int n) {
        BEGIN_KERNEL(1, 4)
        D c00 = {0}, e00 = {0};
        D c01 = {0}, e01 = {0};
        D c02 = {0}, e02 = {0};
        D c03 = {0}, e03 = {0};
        for (int l = 0; l < k; l += KN) {
            V a = load<V>(A + lda * (i + 0) + l);
            c00 = madder(a, load<V>(B + ldb * (j + 0) + l), c00, &e00);
            c01 = madder(a, load<V>(B + ldb * (j + 1) + l), c01, &e01);
            c02 = madder(a, load<V>(B + ldb * (j + 2) + l), c02, &e02);
            c03 = madder(a, load<V>(B + ldb * (j + 3) + l), c03, &e03);
        }
        C[ldc * (j + 0) + (i + 0)] = hsum(c00);
        C[ldc * (j + 1) + (i + 0)] = hsum(c01);
        C[ldc * (j + 2) + (i + 0)] = hsum(c02);
        C[ldc * (j + 3) + (i + 0)] = hsum(c03);
        END_KERNEL()
    }

    NOINLINE void gemm4x1(int m0, int m, int n0, int n) {
        BEGIN_KERNEL(4, 1)
        D c00 = {0}, e00 = {0};
        D c10 = {0}, e10 = {0};
        D c20 = {0}, e20 = {0};
        D c30 = {0}, e30 = {0};
        for (int l = 0; l < k; l += KN) {
            V b = load<V>(B + ldb * (j + 0) + l);
            c00 = madder(load<V>(A + lda * (i + 0) + l), b, c00, &e00);
            c10 = madder(load<V>(A + lda * (i + 1) + l), b, c10, &e10);
            c20 = madder(load<V>(A + lda * (i + 2) + l), b, c20, &e20);
            c30 = madder(load<V>(A + lda * (i + 3) + l), b, c30, &e30);
        }
        C[ldc * (j + 0) + (i + 0)] = hsum(c00);
        C[ldc * (j + 0) + (i + 1)] = hsum(c10);
        C[ldc * (j + 0) + (i + 2)] = hsum(c20);
        C[ldc * (j + 0) + (i + 3)] = hsum(c30);
        END_KERNEL()
    }

    NOINLINE void gemm1x1(int m0, int m, int n0, int n) {
        BEGIN_KERNEL(1, 1)
        D c = {0}, e = {0};
        for (int l = 0; l < k; l += KN)
            c = madder(load<V>(A + lda * i + l),
                       load<V>(B + ldb * j + l), c, &e);
        C[ldc * j + i] = hsum(c);
        END_KERNEL()
    }

    const TA *const A;
    const TB *const B;
    TC *const C;
    const int k;
    const int lda;
    const int ldb;
    const int ldc;
    const int ith;
    const int nth;
};

//////////////////////////////////////////////////////////////////////////////////////////
// QUANT ZERO MATRIX MULTIPLICATION

#if defined(__ARM_FEATURE_DOTPROD)
template <typename TA>
class tinyBLAS_Q0_ARM {
  public:
    tinyBLAS_Q0_ARM(int k,
                    const TA *A, int lda,
                    const block_q8_0 *B, int ldb,
                    float *C, int ldc,
                    int ith, int nth)
        : A(A), B(B), C(C), k(k), lda(lda), ldb(ldb), ldc(ldc), ith(ith), nth(nth) {
    }

    void matmul(int m, int n, int task) {
        if (task == GGML_TASK_TYPE_COMPUTE)
            mnpack(0, m, 0, n);
    }

  private:
    NOINLINE void mnpack(int m0, int m, int n0, int n) {
        int mc, nc, mp, np;
        if (m - m0 <= 0 || n - n0 <= 0)
            return;
        if (m - m0 >= 3 && n - n0 >= 3) {
            mc = 3;
            nc = 3;
            gemm3x3(m0, m, n0, n);
        } else {
            mc = 1;
            nc = 1;
            gemm1x1(m0, m, n0, n);
        }
        mp = m0 + (m - m0) / mc * mc;
        np = n0 + (n - n0) / nc * nc;
        mnpack(mp, m, n0, np);
        mnpack(m0, mp, np, n);
        mnpack(mp, m, np, n);
    }

    NOINLINE void gemm3x3(int m0, int m, int n0, int n) {
        BEGIN_KERNEL(3, 3)
        int32x4_t zero = vdupq_n_s32(0);
        float32x4_t c00 = vdupq_n_f32(0.f);
        float32x4_t c01 = vdupq_n_f32(0.f);
        float32x4_t c02 = vdupq_n_f32(0.f);
        float32x4_t c10 = vdupq_n_f32(0.f);
        float32x4_t c11 = vdupq_n_f32(0.f);
        float32x4_t c12 = vdupq_n_f32(0.f);
        float32x4_t c20 = vdupq_n_f32(0.f);
        float32x4_t c21 = vdupq_n_f32(0.f);
        float32x4_t c22 = vdupq_n_f32(0.f);
        const TA *Ap0 = A + lda * (i + 0);
        const TA *Ap1 = A + lda * (i + 1);
        const TA *Ap2 = A + lda * (i + 2);
        const block_q8_0 *Bp0 = B + ldb * (j + 0);
        const block_q8_0 *Bp1 = B + ldb * (j + 1);
        const block_q8_0 *Bp2 = B + ldb * (j + 2);
        for (int l = 0; l < k; ++l) {
            c00 = vmlaq_n_f32(
                c00,
                vcvtq_f32_s32(vdotq_s32(vdotq_s32(zero, load_lo(Ap0 + l), load_lo(Bp0 + l)),
                                        load_hi(Ap0 + l), load_hi(Bp0 + l))),
                unhalf(Ap0[l].d) * unhalf(Bp0[l].d));
            c01 = vmlaq_n_f32(
                c01,
                vcvtq_f32_s32(vdotq_s32(vdotq_s32(zero, load_lo(Ap0 + l), load_lo(Bp1 + l)),
                                        load_hi(Ap0 + l), load_hi(Bp1 + l))),
                unhalf(Ap0[l].d) * unhalf(Bp1[l].d));
            c02 = vmlaq_n_f32(
                c02,
                vcvtq_f32_s32(vdotq_s32(vdotq_s32(zero, load_lo(Ap0 + l), load_lo(Bp2 + l)),
                                        load_hi(Ap0 + l), load_hi(Bp2 + l))),
                unhalf(Ap0[l].d) * unhalf(Bp2[l].d));
            c10 = vmlaq_n_f32(
                c10,
                vcvtq_f32_s32(vdotq_s32(vdotq_s32(zero, load_lo(Ap1 + l), load_lo(Bp0 + l)),
                                        load_hi(Ap1 + l), load_hi(Bp0 + l))),
                unhalf(Ap1[l].d) * unhalf(Bp0[l].d));
            c11 = vmlaq_n_f32(
                c11,
                vcvtq_f32_s32(vdotq_s32(vdotq_s32(zero, load_lo(Ap1 + l), load_lo(Bp1 + l)),
                                        load_hi(Ap1 + l), load_hi(Bp1 + l))),
                unhalf(Ap1[l].d) * unhalf(Bp1[l].d));
            c12 = vmlaq_n_f32(
                c12,
                vcvtq_f32_s32(vdotq_s32(vdotq_s32(zero, load_lo(Ap1 + l), load_lo(Bp2 + l)),
                                        load_hi(Ap1 + l), load_hi(Bp2 + l))),
                unhalf(Ap1[l].d) * unhalf(Bp2[l].d));
            c20 = vmlaq_n_f32(
                c20,
                vcvtq_f32_s32(vdotq_s32(vdotq_s32(zero, load_lo(Ap2 + l), load_lo(Bp0 + l)),
                                        load_hi(Ap2 + l), load_hi(Bp0 + l))),
                unhalf(Ap2[l].d) * unhalf(Bp0[l].d));
            c21 = vmlaq_n_f32(
                c21,
                vcvtq_f32_s32(vdotq_s32(vdotq_s32(zero, load_lo(Ap2 + l), load_lo(Bp1 + l)),
                                        load_hi(Ap2 + l), load_hi(Bp1 + l))),
                unhalf(Ap2[l].d) * unhalf(Bp1[l].d));
            c22 = vmlaq_n_f32(
                c22,
                vcvtq_f32_s32(vdotq_s32(vdotq_s32(zero, load_lo(Ap2 + l), load_lo(Bp2 + l)),
                                        load_hi(Ap2 + l), load_hi(Bp2 + l))),
                unhalf(Ap2[l].d) * unhalf(Bp2[l].d));
        }
        C[ldc * (j + 0) + (i + 0)] = hsum(c00);
        C[ldc * (j + 0) + (i + 1)] = hsum(c10);
        C[ldc * (j + 0) + (i + 2)] = hsum(c20);
        C[ldc * (j + 1) + (i + 0)] = hsum(c01);
        C[ldc * (j + 1) + (i + 1)] = hsum(c11);
        C[ldc * (j + 1) + (i + 2)] = hsum(c21);
        C[ldc * (j + 2) + (i + 0)] = hsum(c02);
        C[ldc * (j + 2) + (i + 1)] = hsum(c12);
        C[ldc * (j + 2) + (i + 2)] = hsum(c22);
        END_KERNEL()
    }

    NOINLINE void gemm1x1(int m0, int m, int n0, int n) {
        BEGIN_KERNEL(1, 1)
        float32x4_t acc = vdupq_n_f32(0.f);
        const TA *Ap = A + lda * i;
        const block_q8_0 *Bp = B + ldb * j;
        for (int l = 0; l < k; ++l) {
            acc = vmlaq_n_f32(acc,
                              vcvtq_f32_s32(vdotq_s32(
                                  vdotq_s32(vdupq_n_s32(0), load_lo(Ap + l), load_lo(Bp + l)),
                                  load_hi(Ap + l), load_hi(Bp + l))),
                              unhalf(Ap[l].d) * unhalf(Bp[l].d));
        }
        C[ldc * j + i] = hsum(acc);
        END_KERNEL()
    }

    inline int8x16_t load_lo(const block_q8_0 *b) {
        return vld1q_s8(b->qs);
    }
    inline int8x16_t load_hi(const block_q8_0 *b) {
        return vld1q_s8(b->qs + 16);
    }

    inline int8x16_t load_lo(const block_q4_0 *b) {
        return vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vld1q_u8(b->qs),
                                                     vdupq_n_u8(0x0f))),
                        vdupq_n_s8(0x8));
    }
    inline int8x16_t load_hi(const block_q4_0 *b) {
        return vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(vld1q_u8(b->qs), 4)),
                        vdupq_n_s8(0x8));
    }

    const TA *const A;
    const block_q8_0 *const B;
    float *const C;
    const int k;
    const int lda;
    const int ldb;
    const int ldc;
    const int ith;
    const int nth;
};
#endif // __ARM_FEATURE_DOTPROD

#if defined(__AVX2__) || defined(__AVX512F__)
template <typename TA, typename TB, typename TC>
class tinyBLAS_Q0_AVX2 {
  public:
    tinyBLAS_Q0_AVX2(int k,
                     const TA *A, int lda,
                     const TB *B, int ldb,
                     TC *C, int ldc,
                     int ith, int nth)
        : A(A), B(B), C(C), k(k), lda(lda), ldb(ldb), ldc(ldc), ith(ith), nth(nth) {
    }

    void matmul(int m, int n, int task) {
        if (task == GGML_TASK_TYPE_COMPUTE)
            mnpack(0, m, 0, n);
    }

  private:
    NOINLINE void mnpack(int m0, int m, int n0, int n) {
        int mc, nc, mp, np;
        if (m - m0 <= 0 || n - n0 <= 0)
            return;
        if (m - m0 >= 4 && n - n0 >= 3) {
            mc = 4;
            nc = 3;
            gemm4x3(m0, m, n0, n);
        } else if (m - m0 >= 4 && n - n0 >= 1) {
            mc = 4;
            nc = 1;
            gemm4x1(m0, m, n0, n);
        } else if (m - m0 >= 1 && n - n0 >= 4) {
            mc = 1;
            nc = 4;
            gemm1x4(m0, m, n0, n);
        } else {
            mc = 1;
            nc = 1;
            gemm1x1(m0, m, n0, n);
        }
        mp = m0 + (m - m0) / mc * mc;
        np = n0 + (n - n0) / nc * nc;
        mnpack(mp, m, n0, np);
        mnpack(m0, mp, np, n);
        mnpack(mp, m, np, n);
    }

    NOINLINE void gemm4x3(int m0, int m, int n0, int n) {
        BEGIN_KERNEL(4, 3)
        __m256 c00 = _mm256_setzero_ps();
        __m256 c10 = _mm256_setzero_ps();
        __m256 c20 = _mm256_setzero_ps();
        __m256 c30 = _mm256_setzero_ps();
        __m256 c01 = _mm256_setzero_ps();
        __m256 c11 = _mm256_setzero_ps();
        __m256 c21 = _mm256_setzero_ps();
        __m256 c31 = _mm256_setzero_ps();
        __m256 c02 = _mm256_setzero_ps();
        __m256 c12 = _mm256_setzero_ps();
        __m256 c22 = _mm256_setzero_ps();
        __m256 c32 = _mm256_setzero_ps();
        const TA *Ap0 = A + lda * (i + 0);
        const TA *Ap1 = A + lda * (i + 1);
        const TA *Ap2 = A + lda * (i + 2);
        const TA *Ap3 = A + lda * (i + 3);
        const TB *Bp0 = B + ldb * (j + 0);
        const TB *Bp1 = B + ldb * (j + 1);
        const TB *Bp2 = B + ldb * (j + 2);
        for (int l = 0; l < k; ++l) {
            float da0 = unhalf(Ap0[l].d);
            float da1 = unhalf(Ap1[l].d);
            float da2 = unhalf(Ap2[l].d);
            float da3 = unhalf(Ap3[l].d);
            __m256i e0 = load(Ap0 + l);
            __m256i e1 = load(Ap1 + l);
            __m256i e2 = load(Ap2 + l);
            __m256i e3 = load(Ap3 + l);
            float db0 = unhalf(Bp0[l].d);
            __m256 d00 = _mm256_set1_ps(da0 * db0);
            __m256 d10 = _mm256_set1_ps(da1 * db0);
            __m256 d20 = _mm256_set1_ps(da2 * db0);
            __m256 d30 = _mm256_set1_ps(da3 * db0);
            __m256i f0 = load(Bp0 + l);
            __m256i u0 = _mm256_sign_epi8(f0, f0);
            __m256i s00 = _mm256_sign_epi8(e0, f0);
            __m256i s10 = _mm256_sign_epi8(e1, f0);
            __m256i s20 = _mm256_sign_epi8(e2, f0);
            __m256i s30 = _mm256_sign_epi8(e3, f0);
            c00 = madd(d00, updot(u0, s00), c00);
            c10 = madd(d10, updot(u0, s10), c10);
            c20 = madd(d20, updot(u0, s20), c20);
            c30 = madd(d30, updot(u0, s30), c30);
            float db1 = unhalf(Bp1[l].d);
            __m256 d01 = _mm256_set1_ps(da0 * db1);
            __m256 d11 = _mm256_set1_ps(da1 * db1);
            __m256 d21 = _mm256_set1_ps(da2 * db1);
            __m256 d31 = _mm256_set1_ps(da3 * db1);
            __m256i f1 = load(Bp1 + l);
            __m256i u1 = _mm256_sign_epi8(f1, f1);
            __m256i s01 = _mm256_sign_epi8(e0, f1);
            __m256i s11 = _mm256_sign_epi8(e1, f1);
            __m256i s21 = _mm256_sign_epi8(e2, f1);
            __m256i s31 = _mm256_sign_epi8(e3, f1);
            c01 = madd(d01, updot(u1, s01), c01);
            c11 = madd(d11, updot(u1, s11), c11);
            c21 = madd(d21, updot(u1, s21), c21);
            c31 = madd(d31, updot(u1, s31), c31);
            float db2 = unhalf(Bp2[l].d);
            __m256 d02 = _mm256_set1_ps(da0 * db2);
            __m256 d12 = _mm256_set1_ps(da1 * db2);
            __m256 d22 = _mm256_set1_ps(da2 * db2);
            __m256 d32 = _mm256_set1_ps(da3 * db2);
            __m256i f2 = load(Bp2 + l);
            __m256i u2 = _mm256_sign_epi8(f2, f2);
            __m256i s02 = _mm256_sign_epi8(e0, f2);
            __m256i s12 = _mm256_sign_epi8(e1, f2);
            __m256i s22 = _mm256_sign_epi8(e2, f2);
            __m256i s32 = _mm256_sign_epi8(e3, f2);
            c02 = madd(d02, updot(u2, s02), c02);
            c12 = madd(d12, updot(u2, s12), c12);
            c22 = madd(d22, updot(u2, s22), c22);
            c32 = madd(d32, updot(u2, s32), c32);
        }
        C[ldc * (j + 0) + (i + 0)] = hsum(c00);
        C[ldc * (j + 0) + (i + 1)] = hsum(c10);
        C[ldc * (j + 0) + (i + 2)] = hsum(c20);
        C[ldc * (j + 0) + (i + 3)] = hsum(c30);
        C[ldc * (j + 1) + (i + 0)] = hsum(c01);
        C[ldc * (j + 1) + (i + 1)] = hsum(c11);
        C[ldc * (j + 1) + (i + 2)] = hsum(c21);
        C[ldc * (j + 1) + (i + 3)] = hsum(c31);
        C[ldc * (j + 2) + (i + 0)] = hsum(c02);
        C[ldc * (j + 2) + (i + 1)] = hsum(c12);
        C[ldc * (j + 2) + (i + 2)] = hsum(c22);
        C[ldc * (j + 2) + (i + 3)] = hsum(c32);
        END_KERNEL()
    }

    NOINLINE void gemm4x1(int m0, int m, int n0, int n) {
        BEGIN_KERNEL(4, 1)
        __m256 c0 = _mm256_setzero_ps();
        __m256 c1 = _mm256_setzero_ps();
        __m256 c2 = _mm256_setzero_ps();
        __m256 c3 = _mm256_setzero_ps();
        const TA *Ap0 = A + lda * (i + 0);
        const TA *Ap1 = A + lda * (i + 1);
        const TA *Ap2 = A + lda * (i + 2);
        const TA *Ap3 = A + lda * (i + 3);
        const TB *Bp = B + ldb * j;
        for (int l = 0; l < k; ++l) {
            float db0 = unhalf(Bp[l].d);
            __m256i f = load(Bp + l);
            __m256i u = _mm256_sign_epi8(f, f);
            __m256 d0 = _mm256_set1_ps(unhalf(Ap0[l].d) * db0);
            __m256 d1 = _mm256_set1_ps(unhalf(Ap1[l].d) * db0);
            __m256 d2 = _mm256_set1_ps(unhalf(Ap2[l].d) * db0);
            __m256 d3 = _mm256_set1_ps(unhalf(Ap3[l].d) * db0);
            __m256i e0 = load(Ap0 + l);
            __m256i e1 = load(Ap1 + l);
            __m256i e2 = load(Ap2 + l);
            __m256i e3 = load(Ap3 + l);
            __m256i s0 = _mm256_sign_epi8(e0, f);
            __m256i s1 = _mm256_sign_epi8(e1, f);
            __m256i s2 = _mm256_sign_epi8(e2, f);
            __m256i s3 = _mm256_sign_epi8(e3, f);
            __m256 g0 = updot(u, s0);
            __m256 g1 = updot(u, s1);
            __m256 g2 = updot(u, s2);
            __m256 g3 = updot(u, s3);
            c0 = madd(d0, g0, c0);
            c1 = madd(d1, g1, c1);
            c2 = madd(d2, g2, c2);
            c3 = madd(d3, g3, c3);
        }
        C[ldc * j + (i + 0)] = hsum(c0);
        C[ldc * j + (i + 1)] = hsum(c1);
        C[ldc * j + (i + 2)] = hsum(c2);
        C[ldc * j + (i + 3)] = hsum(c3);
        END_KERNEL()
    }

    NOINLINE void gemm1x4(int m0, int m, int n0, int n) {
        BEGIN_KERNEL(1, 4)
        __m256 c0 = _mm256_setzero_ps();
        __m256 c1 = _mm256_setzero_ps();
        __m256 c2 = _mm256_setzero_ps();
        __m256 c3 = _mm256_setzero_ps();
        const TB *Bp0 = B + ldb * (j + 0);
        const TB *Bp1 = B + ldb * (j + 1);
        const TB *Bp2 = B + ldb * (j + 2);
        const TB *Bp3 = B + ldb * (j + 3);
        const TA *Ap = A + lda * i;
        for (int l = 0; l < k; ++l) {
            float da0 = unhalf(Ap[l].d);
            __m256i f = load(Ap + l);
            __m256i u = _mm256_sign_epi8(f, f);
            __m256 d0 = _mm256_set1_ps(unhalf(Bp0[l].d) * da0);
            __m256 d1 = _mm256_set1_ps(unhalf(Bp1[l].d) * da0);
            __m256 d2 = _mm256_set1_ps(unhalf(Bp2[l].d) * da0);
            __m256 d3 = _mm256_set1_ps(unhalf(Bp3[l].d) * da0);
            __m256 g0 = updot(u, _mm256_sign_epi8(load(Bp0 + l), f));
            __m256 g1 = updot(u, _mm256_sign_epi8(load(Bp1 + l), f));
            __m256 g2 = updot(u, _mm256_sign_epi8(load(Bp2 + l), f));
            __m256 g3 = updot(u, _mm256_sign_epi8(load(Bp3 + l), f));
            c0 = madd(d0, g0, c0);
            c1 = madd(d1, g1, c1);
            c2 = madd(d2, g2, c2);
            c3 = madd(d3, g3, c3);
        }
        C[ldc * (j + 0) + i] = hsum(c0);
        C[ldc * (j + 1) + i] = hsum(c1);
        C[ldc * (j + 2) + i] = hsum(c2);
        C[ldc * (j + 3) + i] = hsum(c3);
        END_KERNEL()
    }

    NOINLINE void gemm1x1(int m0, int m, int n0, int n) {
        BEGIN_KERNEL(1, 1)
        __m256 c = _mm256_setzero_ps();
        const TA *Ap = A + lda * i;
        const TB *Bp = B + ldb * j;
        for (int l = 0; l < k; ++l) {
            __m256 d = _mm256_set1_ps(unhalf(Ap[l].d) * unhalf(Bp[l].d));
            __m256i e = load(Ap + l);
            __m256i f = load(Bp + l);
            __m256 g = updot(_mm256_sign_epi8(e, e), _mm256_sign_epi8(f, e));
            c = madd(d, g, c);
        }
        C[ldc * j + i] = hsum(c);
        END_KERNEL()
    }

    inline __m256i load(const block_q8_0 *b) {
        return _mm256_loadu_si256((const __m256i *)b->qs);
    }

    inline __m256i load(const block_q4_0 *b) {
        return _mm256_sub_epi8(denibble(b->qs), _mm256_set1_epi8(8));
    }

    inline __m256 updot(__m256i u, __m256i s) {
        __m256i res;
#if defined(__AVXVNNI__) || (defined(__AVX512VNNI__) && defined(__AVX512VL__))
        res = _mm256_dpbusd_epi32(_mm256_setzero_si256(), u, s);
#else
        res = _mm256_madd_epi16(_mm256_set1_epi16(1), _mm256_maddubs_epi16(u, s));
#endif
        return _mm256_cvtepi32_ps(res);
    }

    static inline __m256i denibble(const uint8_t *p) {
        const __m128i tmp = _mm_loadu_si128((const __m128i *)p);
        const __m256i bytes = MM256_SET_M128I(_mm_srli_epi16(tmp, 4), tmp);
        const __m256i lowMask = _mm256_set1_epi8(15);
        return _mm256_and_si256(lowMask, bytes);
    }

    const TA *const A;
    const TB *const B;
    TC *const C;
    const int k;
    const int lda;
    const int ldb;
    const int ldc;
    const int ith;
    const int nth;
};
#endif // __AVX2__

} // namespace

/**
 * Performs optimized matrix multiplication on CPU.
 *
 * This subroutine may compute C = Aᵀ * B with column major ordering.
 * Despite its name, this isn't a generalized implementation. Work is
 * only performed when a handwritten kernel is written and available.
 * Otherwise the caller should fall back to a general matmul routine.
 *
 * For example, for single-threaded single-precision GEMM you can say
 *
 *     llamafile_sgemm(m, n, k, A, lda, B, ldb, C, ldc,
 *                     0, 1, GGML_TASK_TYPE_COMPUTE,
 *                     GGML_TYPE_F32, GGML_TYPE_F32, GGML_TYPE_F32);
 *
 * @param m is rows in `A` and `C`
 * @param n is cols in `B` and `C`
 * @param k is cols in `A` and rows in `B`
 * @param A is first input matrix (always transposed)
 * @param lda is row stride of `A`
 * @param B is second input matrix (never transposed)
 * @param ldb is row stride of `B`
 * @param C is input/output array of output matrices
 * @param ldc is row stride of `C`
 * @param ith is thread id (must be less than `nth`)
 * @param nth is number of threads (must be greater than zero)
 * @param task is GGML task type
 * @param Atype is GGML data type of `A`
 * @param Btype is GGML data type of `B`
 * @param Ctype is GGML data type of `C`
 * @return true if this function was able to service the matmul request
 */
bool llamafile_sgemm(int m, int n, int k, const void *A, int lda, const void *B, int ldb, void *C,
                     int ldc, int ith, int nth, int task, int Atype, int Btype, int Ctype) {

    assert(m >= 0);
    assert(n >= 0);
    assert(k >= 0);
    assert(lda >= k);
    assert(ldb >= k);
    assert(ldc >= m);
    assert(nth > 0);
    assert(ith < nth);
    assert(1ll * lda * m <= 0x7fffffff);
    assert(1ll * ldb * n <= 0x7fffffff);
    assert(1ll * ldc * n <= 0x7fffffff);

    if (Ctype != GGML_TYPE_F32)
        return false;

    switch (Atype) {

    case GGML_TYPE_F32: {
        if (Btype != GGML_TYPE_F32)
            return false;
#if defined(__AVX512F__)
        if (k % 16)
            return false;
        tinyBLAS<16, __m512, __m512, float, float, float> tb{
            k, (const float *)A, lda,
            (const float *)B, ldb,
            (float *)C, ldc,
            ith, nth};
        tb.matmul(m, n, task);
        return true;
#elif defined(__AVX__) || defined(__AVX2__)
        if (k % 8)
            return false;
        tinyBLAS<8, __m256, __m256, float, float, float> tb{
            k, (const float *)A, lda,
            (const float *)B, ldb,
            (float *)C, ldc,
            ith, nth};
        tb.matmul(m, n, task);
        return true;
#elif defined(__ARM_NEON)
        if (n < 4)
            return false;
        if (k % 4)
            return false;
        tinyBLAS<4, float32x4_t, float32x4_t, float, float, float> tb{
            k, (const float *)A, lda,
            (const float *)B, ldb,
            (float *)C, ldc,
            ith, nth};
        tb.matmul(m, n, task);
        return true;
#else
        return false;
#endif
    }

    case GGML_TYPE_F16: {
#if defined(__AVX512F__)
        if (k % 16)
            return false;
        if (Btype != GGML_TYPE_F32)
            return false;
        tinyBLAS<16, __m512, __m512, ggml_fp16_t, float, float> tb{
            k, (const ggml_fp16_t *)A, lda,
            (const float *)B, ldb,
            (float *)C, ldc,
            ith, nth};
        tb.matmul(m, n, task);
        return true;
#elif (defined(__AVX__) || defined(__AVX2__)) && defined(__F16C__)
        if (k % 8)
            return false;
        if (Btype != GGML_TYPE_F32)
            return false;
        tinyBLAS<8, __m256, __m256, ggml_fp16_t, float, float> tb{
            k, (const ggml_fp16_t *)A, lda,
            (const float *)B, ldb,
            (float *)C, ldc,
            ith, nth};
        tb.matmul(m, n, task);
        return true;
#elif defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) && !defined(_MSC_VER)
        if (n < 8)
            return false;
        if (k % 8)
            return false;
        if (Btype != GGML_TYPE_F16)
            return false;
        tinyBLAS<8, float16x8_t, float16x8_t, ggml_fp16_t, ggml_fp16_t, float> tb{
            k, (const ggml_fp16_t *)A, lda,
            (const ggml_fp16_t *)B, ldb,
            (float *)C, ldc,
            ith, nth};
        tb.matmul(m, n, task);
        return true;
#elif defined(__ARM_NEON) && !defined(_MSC_VER)
        if (k % 4)
            return false;
        if (Btype != GGML_TYPE_F32)
            return false;
        tinyBLAS<4, float32x4_t, float32x4_t, ggml_fp16_t, float, float> tb{
            k, (const ggml_fp16_t *)A, lda,
            (const float *)B, ldb,
            (float *)C, ldc,
            ith, nth};
        tb.matmul(m, n, task);
        return true;
#else
        return false;
#endif
    }

    case GGML_TYPE_Q8_0: {
        if (Btype != GGML_TYPE_Q8_0)
           return false;
#if defined(__AVX2__) || defined(__AVX512F__)
        tinyBLAS_Q0_AVX2<block_q8_0, block_q8_0, float> tb{
            k, (const block_q8_0 *)A, lda,
            (const block_q8_0 *)B, ldb,
            (float *)C, ldc,
            ith, nth};
        tb.matmul(m, n, task);
        return true;
#elif defined(__ARM_FEATURE_DOTPROD)
        tinyBLAS_Q0_ARM<block_q8_0> tb{
            k, (const block_q8_0 *)A, lda,
            (const block_q8_0 *)B, ldb,
            (float *)C, ldc,
            ith, nth};
        tb.matmul(m, n, task);
        return true;
#else
        return false;
#endif
    }

    case GGML_TYPE_Q4_0: {
        if (Btype != GGML_TYPE_Q8_0)
            return false;
#if defined(__AVX2__) || defined(__AVX512F__)
        tinyBLAS_Q0_AVX2<block_q4_0, block_q8_0, float> tb{
            k, (const block_q4_0 *)A, lda,
            (const block_q8_0 *)B, ldb,
            (float *)C, ldc,
            ith, nth};
        tb.matmul(m, n, task);
        return true;
#elif defined(__ARM_FEATURE_DOTPROD)
        tinyBLAS_Q0_ARM<block_q4_0> tb{
            k, (const block_q4_0 *)A, lda,
            (const block_q8_0 *)B, ldb,
            (float *)C, ldc,
            ith, nth};
        tb.matmul(m, n, task);
        return true;
#else
        return false;
#endif
    }

    default:
        return false;
    }

    (void)m;
    (void)n;
    (void)k;
    (void)A;
    (void)lda;
    (void)B;
    (void)ldb;
    (void)C;
    (void)ldc;
    (void)ith;
    (void)nth;
    (void)task;
    (void)Atype;
    (void)Btype;
    (void)Ctype;
}
