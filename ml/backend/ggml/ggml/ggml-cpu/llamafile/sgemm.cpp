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

#if defined(__GNUC__)
#pragma GCC diagnostic ignored "-Wpedantic"
#pragma GCC diagnostic ignored "-Wignored-attributes"
#endif

#include "sgemm.h"
#include "ggml-impl.h"
#include "ggml-cpu-impl.h"
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

#if defined(__MMA__)
typedef vector unsigned char vec_t;
typedef __vector_quad acc_t;
#endif
////////////////////////////////////////////////////////////////////////////////////////////////////
// VECTORIZED FUSED MULTIPLY ADD

/**
 * Computes a * b + c.
 */
template <typename T, typename U>
inline U madd(T a, T b, U c) {
    return add(mul(a, b), c);
}

#if defined(__FMA__)
#if defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
template <>
inline __m256 madd(__m256 a, __m256 b, __m256 c) {
    return _mm256_fmadd_ps(a, b, c);
}
#endif
#if defined(__AVX512F__)
template <>
inline __m512 madd(__m512 a, __m512 b, __m512 c) {
    return _mm512_fmadd_ps(a, b, c);
}
#endif
#endif

#if defined(__ARM_FEATURE_FMA)
template <>
inline float32x4_t madd(float32x4_t a, float32x4_t b, float32x4_t c) {
    return vfmaq_f32(c, b, a);
}
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) && !defined(_MSC_VER)
template <>
inline float16x8_t madd(float16x8_t a, float16x8_t b, float16x8_t c) {
    return vfmaq_f16(c, b, a);
}
#endif
#endif

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
// CONSTANTS

#if defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
static const int8_t kvalues_iq4nl[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
static const __m128i iq4nlt = _mm_loadu_si128((const __m128i *) kvalues_iq4nl);
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////
// FLOATING POINT MATRIX MULTIPLICATION

template <int KN, typename D, typename V, typename TA, typename TB, typename TC>
class tinyBLAS {
  public:
    tinyBLAS(int64_t k,
             const TA *A, int64_t lda,
             const TB *B, int64_t ldb,
             TC *C, int64_t ldc,
             int ith, int nth)
        : A(A), B(B), C(C), k(k), lda(lda), ldb(ldb), ldc(ldc), ith(ith), nth(nth) {
    }

    void matmul(int64_t m, int64_t n) {
        mnpack(0, m, 0, n);
    }

  private:
    NOINLINE void mnpack(int64_t m0, int64_t m, int64_t n0, int64_t n) {
        int64_t mc, nc, mp, np;
        switch ((MIN(m - m0, 5) << 4) | MIN(n - n0, 5)) {
#if VECTOR_REGISTERS == 32
        case 0x55:
            mc = 5;
            nc = 5;
            gemm<5, 5>(m0, m, n0, n);
            break;
        case 0x45:
            mc = 4;
            nc = 5;
            gemm<4, 5>(m0, m, n0, n);
            break;
        case 0x54:
            mc = 5;
            nc = 4;
            gemm<5, 4>(m0, m, n0, n);
            break;
        case 0x44:
            mc = 4;
            nc = 4;
            gemm<4, 4>(m0, m, n0, n);
            break;
        case 0x53:
            mc = 5;
            nc = 3;
            gemm<5, 3>(m0, m, n0, n);
            break;
        case 0x35:
            mc = 3;
            nc = 5;
            gemm<3, 5>(m0, m, n0, n);
            break;
        case 0x43:
            mc = 4;
            nc = 3;
            gemm<4, 3>(m0, m, n0, n);
            break;
#else
        case 0x55:
        case 0x54:
        case 0x53:
        case 0x45:
        case 0x44:
        case 0x43:
            mc = 4;
            nc = 3;
            gemm<4, 3>(m0, m, n0, n);
            break;
        case 0x35:
#endif
        case 0x34:
            mc = 3;
            nc = 4;
            gemm<3, 4>(m0, m, n0, n);
            break;
        case 0x52:
            mc = 5;
            nc = 2;
            gemm<5, 2>(m0, m, n0, n);
            break;
        case 0x33:
            mc = 3;
            nc = 3;
            gemm<3, 3>(m0, m, n0, n);
            break;
        case 0x25:
            mc = 2;
            nc = 5;
            gemm<2, 5>(m0, m, n0, n);
            break;
        case 0x42:
            mc = 4;
            nc = 2;
            gemm<4, 2>(m0, m, n0, n);
            break;
        case 0x24:
            mc = 2;
            nc = 4;
            gemm<2, 4>(m0, m, n0, n);
            break;
        case 0x32:
            mc = 3;
            nc = 2;
            gemm<3, 2>(m0, m, n0, n);
            break;
        case 0x23:
            mc = 2;
            nc = 3;
            gemm<2, 3>(m0, m, n0, n);
            break;
        case 0x51:
            mc = 5;
            nc = 1;
            gemm<5, 1>(m0, m, n0, n);
            break;
        case 0x41:
            mc = 4;
            nc = 1;
            gemm<4, 1>(m0, m, n0, n);
            break;
        case 0x22:
            mc = 2;
            nc = 2;
            gemm<2, 2>(m0, m, n0, n);
            break;
        case 0x15:
            mc = 1;
            nc = 5;
            gemm<1, 5>(m0, m, n0, n);
            break;
        case 0x14:
            mc = 1;
            nc = 4;
            gemm<1, 4>(m0, m, n0, n);
            break;
        case 0x31:
            mc = 3;
            nc = 1;
            gemm<3, 1>(m0, m, n0, n);
            break;
        case 0x13:
            mc = 1;
            nc = 3;
            gemm<1, 3>(m0, m, n0, n);
            break;
        case 0x21:
            mc = 2;
            nc = 1;
            gemm<2, 1>(m0, m, n0, n);
            break;
        case 0x12:
            mc = 1;
            nc = 2;
            gemm<1, 2>(m0, m, n0, n);
            break;
        case 0x11:
            mc = 1;
            nc = 1;
            gemm<1, 1>(m0, m, n0, n);
            break;
        default:
            return;
        }
        mp = m0 + (m - m0) / mc * mc;
        np = n0 + (n - n0) / nc * nc;
        mnpack(mp, m, n0, np);
        mnpack(m0, m, np, n);
    }

    template <int RM, int RN>
    NOINLINE void gemm(int64_t m0, int64_t m, int64_t n0, int64_t n) {
        int64_t ytiles = (m - m0) / RM;
        int64_t xtiles = (n - n0) / RN;
        int64_t tiles = xtiles * ytiles;
        int64_t duty = (tiles + nth - 1) / nth;
        int64_t start = duty * ith;
        int64_t end = start + duty;
        if (end > tiles)
            end = tiles;
        for (int64_t job = start; job < end; ++job) {
            int64_t ii = m0 + job / xtiles * RM;
            int64_t jj = n0 + job % xtiles * RN;
            D Cv[RN][RM] = {};
            for (int64_t l = 0; l < k; l += KN)
                for (int64_t j = 0; j < RN; ++j)
                    for (int64_t i = 0; i < RM; ++i)
                        Cv[j][i] = madd(load<V>(A + lda * (ii + i) + l),
                                        load<V>(B + ldb * (jj + j) + l),
                                        Cv[j][i]);
            for (int64_t j = 0; j < RN; ++j)
                for (int64_t i = 0; i < RM; ++i)
                    C[ldc * (jj + j) + (ii + i)] = hsum(Cv[j][i]);
        }
    }

    const TA *const A;
    const TB *const B;
    TC *const C;
    const int64_t k;
    const int64_t lda;
    const int64_t ldb;
    const int64_t ldc;
    const int ith;
    const int nth;
};

//////////////////////////////////////////////////////////////////////////////////////////
// QUANT ZERO MATRIX MULTIPLICATION

#if defined(__ARM_FEATURE_DOTPROD)
template <typename TA>
class tinyBLAS_Q0_ARM {
  public:
    tinyBLAS_Q0_ARM(int64_t k,
                    const TA *A, int64_t lda,
                    const block_q8_0 *B, int64_t ldb,
                    float *C, int64_t ldc,
                    int ith, int nth)
        : A(A), B(B), C(C), k(k), lda(lda), ldb(ldb), ldc(ldc), ith(ith), nth(nth) {
    }

    void matmul(int64_t m, int64_t n) {
        mnpack(0, m, 0, n);
    }

  private:
    NOINLINE void mnpack(int64_t m0, int64_t m, int64_t n0, int64_t n) {
        int64_t mc, nc, mp, np;
        switch ((MIN(m - m0, 3) << 4) | MIN(n - n0, 3ll)) {
        case 0x33:
            mc = 3;
            nc = 3;
            gemm<3, 3>(m0, m, n0, n);
            break;
        case 0x32:
            mc = 3;
            nc = 2;
            gemm<3, 2>(m0, m, n0, n);
            break;
        case 0x23:
            mc = 2;
            nc = 3;
            gemm<2, 3>(m0, m, n0, n);
            break;
        case 0x22:
            mc = 2;
            nc = 2;
            gemm<2, 2>(m0, m, n0, n);
            break;
        case 0x31:
            mc = 3;
            nc = 1;
            gemm<3, 1>(m0, m, n0, n);
            break;
        case 0x13:
            mc = 1;
            nc = 3;
            gemm<1, 3>(m0, m, n0, n);
            break;
        case 0x21:
            mc = 2;
            nc = 1;
            gemm<2, 1>(m0, m, n0, n);
            break;
        case 0x12:
            mc = 1;
            nc = 2;
            gemm<1, 2>(m0, m, n0, n);
            break;
        case 0x11:
            mc = 1;
            nc = 1;
            gemm<1, 1>(m0, m, n0, n);
            break;
        default:
            return;
        }
        mp = m0 + (m - m0) / mc * mc;
        np = n0 + (n - n0) / nc * nc;
        mnpack(mp, m, n0, np);
        mnpack(m0, m, np, n);
    }

    template <int RM, int RN>
    NOINLINE void gemm(int64_t m0, int64_t m, int64_t n0, int64_t n) {
        int64_t ytiles = (m - m0) / RM;
        int64_t xtiles = (n - n0) / RN;
        int64_t tiles = xtiles * ytiles;
        int64_t duty = (tiles + nth - 1) / nth;
        int64_t start = duty * ith;
        int64_t end = start + duty;
        if (end > tiles)
            end = tiles;
        for (int64_t job = start; job < end; ++job) {
            int64_t ii = m0 + job / xtiles * RM;
            int64_t jj = n0 + job % xtiles * RN;
            float32x4_t Cv[RN][RM] = {};
            for (int64_t l = 0; l < k; ++l)
                for (int64_t j = 0; j < RN; ++j)
                    for (int64_t i = 0; i < RM; ++i)
                        Cv[j][i] = vmlaq_n_f32(Cv[j][i],
                                               vcvtq_f32_s32(vdotq_s32(
                                                   vdotq_s32(vdupq_n_s32(0),
                                                             load_lo(A + lda * (ii + i) + l),
                                                             load_lo(B + ldb * (jj + j) + l)),
                                                   load_hi(A + lda * (ii + i) + l),
                                                   load_hi(B + ldb * (jj + j) + l))),
                                               unhalf(A[lda * (ii + i) + l].d) *
                                               unhalf(B[ldb * (jj + j) + l].d));
            for (int64_t j = 0; j < RN; ++j)
                for (int64_t i = 0; i < RM; ++i)
                    C[ldc * (jj + j) + (ii + i)] = hsum(Cv[j][i]);
        }
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
    const int64_t k;
    const int64_t lda;
    const int64_t ldb;
    const int64_t ldc;
    const int ith;
    const int nth;
};
#endif // __ARM_FEATURE_DOTPROD

#if defined(__AVX2__) || defined(__AVX512F__) || defined(__AVX__)
template <typename TA, typename TB, typename TC>
class tinyBLAS_Q0_AVX {
  public:
    tinyBLAS_Q0_AVX(int64_t k,
                    const TA *A, int64_t lda,
                    const TB *B, int64_t ldb,
                    TC *C, int64_t ldc,
                    int ith, int nth)
        : A(A), B(B), C(C), k(k), lda(lda), ldb(ldb), ldc(ldc), ith(ith), nth(nth) {
    }

    void matmul(int64_t m, int64_t n) {
        mnpack(0, m, 0, n);
    }

  private:
    void mnpack(int64_t m0, int64_t m, int64_t n0, int64_t n) {
        int64_t mc, nc, mp, np;
        switch ((MIN(m - m0, 4) << 4) | MIN(n - n0, 4)) {
#if VECTOR_REGISTERS == 32
        case 0x44:
            mc = 4;
            nc = 4;
#if defined(__AVX2__) && defined(__F16C__)
            gemm4xN<4>(m0, m, n0, n);
#else
            gemm<4, 4>(m0, m, n0, n);
#endif
            break;
        case 0x43:
            mc = 4;
            nc = 3;
#if defined(__AVX2__) && defined(__F16C__)
            gemm4xN<3>(m0, m, n0, n);
#else
            gemm<4, 3>(m0, m, n0, n);
#endif
            break;
        case 0x34:
            mc = 3;
            nc = 4;
#if defined(__AVX2__) && defined(__F16C__)
            gemmMx4<3>(m0, m, n0, n);
#else
            gemm<3, 4>(m0, m, n0, n);
#endif
            break;
        case 0x33:
            mc = 3;
            nc = 3;
            gemm<3, 3>(m0, m, n0, n);
            break;
        case 0x42:
            mc = 4;
            nc = 2;
#if defined(__AVX2__) && defined(__F16C__)
            gemm4xN<2>(m0, m, n0, n);
#else
            gemm<4, 2>(m0, m, n0, n);
#endif
            break;
        case 0x24:
            mc = 2;
            nc = 4;
#if defined(__AVX2__) && defined(__F16C__)
            gemmMx4<2>(m0, m, n0, n);
#else
            gemm<2, 4>(m0, m, n0, n);
#endif
            break;
#else
        case 0x44:
        case 0x43:
        case 0x42:
            mc = 4;
            nc = 2;
#if defined(__AVX2__) && defined(__F16C__)
            gemm4xN<2>(m0, m, n0, n);
#else
            gemm<4, 2>(m0, m, n0, n);
#endif
            break;
        case 0x34:
        case 0x24:
            mc = 2;
            nc = 4;
#if defined(__AVX2__) && defined(__F16C__)
            gemmMx4<2>(m0, m, n0, n);
#else
            gemm<2, 4>(m0, m, n0, n);
#endif
            break;
        case 0x33:
#endif
        case 0x32:
            mc = 3;
            nc = 2;
            gemm<3, 2>(m0, m, n0, n);
            break;
        case 0x23:
            mc = 2;
            nc = 3;
            gemm<2, 3>(m0, m, n0, n);
            break;
        case 0x41:
            mc = 4;
            nc = 1;
#if defined(__AVX2__) && defined(__F16C__)
            gemm4xN<1>(m0, m, n0, n);
#else
            gemm<4, 1>(m0, m, n0, n);
#endif
            break;
        case 0x22:
            mc = 2;
            nc = 2;
            gemm<2, 2>(m0, m, n0, n);
            break;
        case 0x14:
            mc = 1;
            nc = 4;
#if defined(__AVX2__) && defined(__F16C__)
            gemmMx4<1>(m0, m, n0, n);
#else
            gemm<1, 4>(m0, m, n0, n);
#endif
            break;
        case 0x31:
            mc = 3;
            nc = 1;
            gemm<3, 1>(m0, m, n0, n);
            break;
        case 0x13:
            mc = 1;
            nc = 3;
            gemm<1, 3>(m0, m, n0, n);
            break;
        case 0x21:
            mc = 2;
            nc = 1;
            gemm<2, 1>(m0, m, n0, n);
            break;
        case 0x12:
            mc = 1;
            nc = 2;
            gemm<1, 2>(m0, m, n0, n);
            break;
        case 0x11:
            mc = 1;
            nc = 1;
            gemm<1, 1>(m0, m, n0, n);
            break;
        default:
            return;
        }
        mp = m0 + (m - m0) / mc * mc;
        np = n0 + (n - n0) / nc * nc;
        mnpack(mp, m, n0, np);
        mnpack(m0, m, np, n);
    }

#if defined(__AVX2__) && defined(__F16C__)
// Templated functions for gemm of dimensions 4xN
    template <int RN>
    NOINLINE void gemm4xN(int64_t m0, int64_t m, int64_t n0, int64_t n) {
        int64_t ytiles = (m - m0) / 4;
        int64_t xtiles = (n - n0) / RN;
        int64_t tiles = xtiles * ytiles;
        int64_t duty = (tiles + nth - 1) / nth;
        int64_t start = duty * ith;
        int64_t end = start + duty;
        if (end > tiles)
            end = tiles;
        for (int64_t job = start; job < end; ++job) {
            int64_t ii = m0 + job / xtiles * 4;
            int64_t jj = n0 + job % xtiles * RN;
            __m256 Cv[RN][4] = {};
            for (int64_t l = 0; l < k; ++l) {
                uint64_t a_delta = ((uint64_t)A[lda * (ii + 3) + l].d << 48) | ((uint64_t)A[lda * (ii + 2) + l].d << 32) | ((uint64_t)A[lda * (ii + 1) + l].d << 16) | (A[lda * (ii + 0) + l].d);
                // Convert delta values for four blocks to float values
                __m128 da = _mm_cvtph_ps(_mm_set_epi64x(0, a_delta));
                __m256i avec0 = load(A + lda * (ii + 0) + l);
                __m256i avec1 = load(A + lda * (ii + 1) + l);
                __m256i avec2 = load(A + lda * (ii + 2) + l);
                __m256i avec3 = load(A + lda * (ii + 3) + l);
                for (int64_t j = 0; j < RN; ++j) {
                        __m128 db = _mm_set1_ps(unhalf(B[ldb * (jj + j) + l].d));
                        // Computation of product of delta values for four blocks and replicate it across 256 bit lane
                        __m256 dvec =  _mm256_castps128_ps256(_mm_mul_ps(da, db));
                        dvec = _mm256_permute2f128_ps(dvec ,dvec, 0);
                        // Computation of dot product and multiplication with appropriate delta value products
                        Cv[j][0] = madd(_mm256_shuffle_ps(dvec, dvec, 0),
                                    updot(_mm256_sign_epi8(avec0, avec0),
                                          _mm256_sign_epi8(load(B + ldb * (jj + j) + l), avec0)),
                                    Cv[j][0]);
                        Cv[j][1] = madd(_mm256_shuffle_ps(dvec, dvec, 85),
                                    updot(_mm256_sign_epi8(avec1, avec1),
                                            _mm256_sign_epi8(load(B + ldb * (jj + j) + l), avec1)),
                                    Cv[j][1]);
                        Cv[j][2] = madd(_mm256_shuffle_ps(dvec, dvec, 170),
                                    updot(_mm256_sign_epi8(avec2, avec2),
                                            _mm256_sign_epi8(load(B + ldb * (jj + j) + l), avec2)),
                                    Cv[j][2]);
                        Cv[j][3] = madd(_mm256_shuffle_ps(dvec, dvec, 255),
                                    updot(_mm256_sign_epi8(avec3, avec3),
                                            _mm256_sign_epi8(load(B + ldb * (jj + j) + l), avec3)),
                                    Cv[j][3]);
                }
            }

            for (int64_t j = 0; j < RN; ++j)
                for (int64_t i = 0; i < 4; ++i)
                    C[ldc * (jj + j) + (ii + i)] = hsum(Cv[j][i]);
        }
    }

    // Templated functions for gemm of dimensions Mx4
    template <int RM>
    NOINLINE void gemmMx4(int64_t m0, int64_t m, int64_t n0, int64_t n) {
        int64_t ytiles = (m - m0) / RM;
        int64_t xtiles = (n - n0) / 4;
        int64_t tiles = xtiles * ytiles;
        int64_t duty = (tiles + nth - 1) / nth;
        int64_t start = duty * ith;
        int64_t end = start + duty;
        if (end > tiles)
            end = tiles;
        for (int64_t job = start; job < end; ++job) {
            int64_t ii = m0 + job / xtiles * RM;
            int64_t jj = n0 + job % xtiles * 4;
            __m256 Cv[4][RM] = {};
            for (int64_t l = 0; l < k; ++l) {
                uint64_t b_delta = ((uint64_t)B[ldb * (jj + 3) + l].d << 48) | ((uint64_t)B[ldb * (jj + 2) + l].d << 32) | ((uint64_t)B[ldb * (jj + 1) + l].d << 16) | (B[ldb * (jj + 0) + l].d);
                // Convert delta values for four blocks to float values
                __m128 db = _mm_cvtph_ps(_mm_set_epi64x(0, b_delta));
                __m256i bvec0 = load(B + ldb * (jj + 0) + l);
                __m256i bvec1 = load(B + ldb * (jj + 1) + l);
                __m256i bvec2 = load(B + ldb * (jj + 2) + l);
                __m256i bvec3 = load(B + ldb * (jj + 3) + l);
                for (int64_t i = 0; i < RM; ++i) {
                    __m128 da = _mm_set1_ps(unhalf((A[lda * (ii + i) + l].d)));
                    // Computation of product of delta values for four blocks and replicate it across 256 bit lane
                    __m256 dvec =  _mm256_castps128_ps256(_mm_mul_ps(da, db));
                    dvec = _mm256_permute2f128_ps(dvec ,dvec, 0);
                    // Computation of dot product and multiplication with appropriate delta value products
                    Cv[0][i] = madd(_mm256_shuffle_ps(dvec, dvec, 0),
                                    updot(_mm256_sign_epi8(load(A + lda * (ii + i) + l),
                                                            load(A + lda * (ii + i) + l)),
                                            _mm256_sign_epi8(bvec0, load(A + lda * (ii + i) + l))),
                                    Cv[0][i]);
                    Cv[1][i] = madd(_mm256_shuffle_ps(dvec, dvec, 85),
                                    updot(_mm256_sign_epi8(load(A + lda * (ii + i) + l),
                                                            load(A + lda * (ii + i) + l)),
                                            _mm256_sign_epi8(bvec1, load(A + lda * (ii + i) + l))),
                                    Cv[1][i]);
                    Cv[2][i] = madd(_mm256_shuffle_ps(dvec, dvec, 170),
                                    updot(_mm256_sign_epi8(load(A + lda * (ii + i) + l),
                                                            load(A + lda * (ii + i) + l)),
                                            _mm256_sign_epi8(bvec2, load(A + lda * (ii + i) + l))),
                                    Cv[2][i]);
                    Cv[3][i] = madd(_mm256_shuffle_ps(dvec, dvec, 255),
                                    updot(_mm256_sign_epi8(load(A + lda * (ii + i) + l),
                                                            load(A + lda * (ii + i) + l)),
                                            _mm256_sign_epi8(bvec3, load(A + lda * (ii + i) + l))),
                                    Cv[3][i]);
                }
            }
            for (int64_t j = 0; j < 4; ++j)
                for (int64_t i = 0; i < RM; ++i)
                    C[ldc * (jj + j) + (ii + i)] = hsum(Cv[j][i]);
        }
    }
#endif

    template <int RM, int RN>
    NOINLINE void gemm(int64_t m0, int64_t m, int64_t n0, int64_t n) {
        int64_t ytiles = (m - m0) / RM;
        int64_t xtiles = (n - n0) / RN;
        int64_t tiles = xtiles * ytiles;
        int64_t duty = (tiles + nth - 1) / nth;
        int64_t start = duty * ith;
        int64_t end = start + duty;
        if (end > tiles)
            end = tiles;
        for (int64_t job = start; job < end; ++job) {
            int64_t ii = m0 + job / xtiles * RM;
            int64_t jj = n0 + job % xtiles * RN;
            __m256 Cv[RN][RM] = {};
            for (int64_t l = 0; l < k; ++l)
                for (int64_t j = 0; j < RN; ++j)
                    for (int64_t i = 0; i < RM; ++i) {
#if defined(__AVX2__)
                        __m256 udTmp = updot(_mm256_sign_epi8(load(A + lda * (ii + i) + l),
                                                              load(A + lda * (ii + i) + l)),
                                             _mm256_sign_epi8(load(B + ldb * (jj + j) + l),
                                                              load(A + lda * (ii + i) + l)));
#else
                        __m128i ali0 = load0(A + lda * (ii + i) + l);
                        __m128i ali1 = load1(A + lda * (ii + i) + l);
                        __m128i blj0 = load0(B + ldb * (jj + j) + l);
                        __m128i blj1 = load1(B + ldb * (jj + j) + l);

                        __m128i sepAA0 = _mm_sign_epi8(ali0, ali0);
                        __m128i sepAA1 = _mm_sign_epi8(ali1, ali1);
                        __m128i sepBA0 = _mm_sign_epi8(blj0, ali0);
                        __m128i sepBA1 = _mm_sign_epi8(blj1, ali1);

                        // updot
                        const __m128i oneFill = _mm_set1_epi16(1);
                        __m128i mad0 = _mm_maddubs_epi16(sepAA0, sepBA0);
                        __m128i mad1 = _mm_maddubs_epi16(sepAA1, sepBA1);
                        __m256 udTmp = _mm256_cvtepi32_ps(MM256_SET_M128I(_mm_madd_epi16(oneFill, mad1), _mm_madd_epi16(oneFill, mad0)));
#endif
                        Cv[j][i] = madd(_mm256_set1_ps(unhalf(A[lda * (ii + i) + l].d) *
                                                       unhalf(B[ldb * (jj + j) + l].d)),
                                                       udTmp,
                                                       Cv[j][i]);
                    }
            for (int64_t j = 0; j < RN; ++j)
                for (int64_t i = 0; i < RM; ++i)
                    C[ldc * (jj + j) + (ii + i)] = hsum(Cv[j][i]);
        }
    }

    inline __m256i load(const block_q8_0 *b) {
        return _mm256_loadu_si256((const __m256i *)b->qs);
    }

    inline __m128i load0(const block_q8_0 *b) {
        return _mm_loadu_si128((const __m128i *)b->qs);
    }

    inline __m128i load1(const block_q8_0 *b) {
        return _mm_loadu_si128(((const __m128i *)b->qs) + 1);
    }

    inline __m256i load(const block_q4_0 *b) {
        return _mm256_sub_epi8(denibble(b->qs), _mm256_set1_epi8(8));
    }

    inline __m128i load0(const block_q4_0 *b) {
        const __m128i x = _mm_loadu_si128((const __m128i *)(b->qs));
        return _mm_sub_epi8(_mm_and_si128(_mm_set1_epi8(15), x), _mm_set1_epi8(8));
    }

    inline __m128i load1(const block_q4_0 *b) {
        const __m128i x = _mm_loadu_si128((const __m128i *)(b->qs));
        return _mm_sub_epi8(_mm_and_si128(_mm_set1_epi8(15), _mm_srli_epi16(x, 4)), _mm_set1_epi8(8));
    }

    inline __m256i load(const block_q5_0 *b) {
        return _mm256_or_si256(denibble(b->qs), bittobyte(b->qh));
    }

    inline __m128i load0(const block_q5_0* b) {
        const __m128i x = _mm_loadu_si128((const __m128i *)(b->qs));
        uint32_t x32;
        memcpy(&x32, b->qh, sizeof(uint32_t));
        __m128i qxl = _mm_and_si128(_mm_set1_epi8(15), x);
        __m128i bytesl = _mm_cmpeq_epi8(_mm_set1_epi64x(-1),
                                        _mm_or_si128(_mm_set1_epi64x(0x7fbfdfeff7fbfdfe),
                                                     _mm_shuffle_epi8(_mm_set1_epi32(x32),
                                                                      _mm_set_epi64x(0x0101010101010101, 0x0000000000000000))));
        bytesl = _mm_andnot_si128(bytesl, _mm_set1_epi8((char)0xF0));
        return _mm_or_si128(qxl, bytesl);
    }

    inline __m128i load1(const block_q5_0* b) {
        const __m128i x = _mm_loadu_si128((const __m128i *)(b->qs));
        uint32_t x32;
        memcpy(&x32, b->qh, sizeof(uint32_t));
        __m128i qxh = _mm_and_si128(_mm_set1_epi8(15), _mm_srli_epi16(x, 4));
        __m128i bytesh = _mm_cmpeq_epi8(_mm_set1_epi64x(-1),
                                        _mm_or_si128(_mm_set1_epi64x(0x7fbfdfeff7fbfdfe),
                                                     _mm_shuffle_epi8(_mm_set1_epi32(x32),
                                                                      _mm_set_epi64x(0x0303030303030303, 0x0202020202020202))));
        bytesh = _mm_andnot_si128(bytesh, _mm_set1_epi8((char)0xF0));
        return _mm_or_si128(qxh, bytesh);
    }

    inline __m256i load(const block_iq4_nl *b) {
        return MM256_SET_M128I(load1(b), load0(b));
    }

    inline __m128i load0(const block_iq4_nl *b) {
        const __m128i x = _mm_loadu_si128((const __m128i *)(b->qs));
        return _mm_shuffle_epi8(iq4nlt, _mm_and_si128(_mm_set1_epi8(15), x));
    }

    inline __m128i load1(const block_iq4_nl *b) {
        const __m128i x = _mm_loadu_si128((const __m128i *)(b->qs));
        return _mm_shuffle_epi8(iq4nlt, _mm_and_si128(_mm_set1_epi8(15), _mm_srli_epi16(x, 4)));
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
        __m128i x = _mm_loadu_si128((const __m128i *)p);
        return _mm256_and_si256(_mm256_set1_epi8(15),
                                _mm256_insertf128_si256(_mm256_castsi128_si256(x),
                                                        _mm_srli_epi16(x, 4), 1));
    }

    static inline __m256i bittobyte(const uint8_t *p) {
        uint32_t x32;
        memcpy(&x32, p, sizeof(uint32_t));
        __m256i bytes = _mm256_cmpeq_epi8(_mm256_set1_epi64x(-1),
                                          _mm256_or_si256(_mm256_set1_epi64x(0x7fbfdfeff7fbfdfe),
                                                          _mm256_shuffle_epi8(_mm256_set1_epi32(x32),
                                                                              _mm256_set_epi64x(0x0303030303030303, 0x0202020202020202,
                                                                                                0x0101010101010101, 0x0000000000000000))));
        return _mm256_andnot_si256(bytes, _mm256_set1_epi8((char)0xF0));
    }

    const TA *const A;
    const TB *const B;
    TC *const C;
    const int64_t k;
    const int64_t lda;
    const int64_t ldb;
    const int64_t ldc;
    const int ith;
    const int nth;
};
#endif // __AVX__

//PPC Implementation
#if defined(__MMA__)

#define SAVE_ACC(ACC, ii, jj) \
   __builtin_mma_disassemble_acc(vec_C, ACC); \
   for (int I = 0; I < 4; I++) { \
      for (int J = 0; J < 4; J++) { \
         *((float*)(C+ii+((jj+J)*ldc)+I)) = *((float*)&vec_C[I]+J); \
      } \
   } \

template <typename TA, typename TB, typename TC>
class tinyBLAS_PPC {
  public:
    tinyBLAS_PPC(int64_t k,
                const TA *A, int64_t lda,
                const TB *B, int64_t ldb,
                TC *C, int64_t ldc,
                int ith, int nth)
        : A(A), B(B), C(C), k(k), lda(lda), ldb(ldb), ldc(ldc), ith(ith), nth(nth) {
    }

    void matmul(int64_t m, int64_t n) {
       mnpack(0, m, 0, n);
    }

  private:

    void (tinyBLAS_PPC::*kernel)(int64_t, int64_t);

    void READ_BLOCK(const float* a, int64_t lda, int rows, int cols, float* vec) {
        int64_t i, j;
        float *aoffset = NULL, *boffset = NULL;
        float *aoffset1 = NULL, *aoffset2 = NULL, *aoffset3 = NULL, *aoffset4 = NULL;
        float *aoffset5 = NULL, *aoffset6 = NULL, *aoffset7 = NULL, *aoffset8 = NULL;

        aoffset = const_cast<float*>(a);
        boffset = vec;
        j = (rows >> 3);
        if (j > 0) {
            do {
                aoffset1 = aoffset;
                aoffset2 = aoffset1 + lda;
                aoffset3 = aoffset2 + lda;
                aoffset4 = aoffset3 + lda;
                aoffset5 = aoffset4 + lda;
                aoffset6 = aoffset5 + lda;
                aoffset7 = aoffset6 + lda;
                aoffset8 = aoffset7 + lda;
                aoffset += 8 * lda;
                i = (cols >> 3);
                if (i > 0) {
                    __vector_pair C1, C2, C3, C4, C5, C6, C7, C8;
                    vector float c1[2], c2[2], c3[2], c4[2], c5[2], c6[2], c7[2], c8[2];
                    vector float t1, t2, t3, t4, t5, t6, t7, t8;
                    do {
                        C1 = __builtin_vsx_lxvp(0, (__vector_pair*)aoffset1);
                        C2 = __builtin_vsx_lxvp(0, (__vector_pair*)aoffset2);
                        C3 = __builtin_vsx_lxvp(0, (__vector_pair*)aoffset3);
                        C4 = __builtin_vsx_lxvp(0, (__vector_pair*)aoffset4);
                        C5 = __builtin_vsx_lxvp(0, (__vector_pair*)aoffset5);
                        C6 = __builtin_vsx_lxvp(0, (__vector_pair*)aoffset6);
                        C7 = __builtin_vsx_lxvp(0, (__vector_pair*)aoffset7);
                        C8 = __builtin_vsx_lxvp(0, (__vector_pair*)aoffset8);
                        __builtin_vsx_disassemble_pair(c1, &C1);
                        __builtin_vsx_disassemble_pair(c2, &C2);
                        __builtin_vsx_disassemble_pair(c3, &C3);
                        __builtin_vsx_disassemble_pair(c4, &C4);
                        __builtin_vsx_disassemble_pair(c5, &C5);
                        __builtin_vsx_disassemble_pair(c6, &C6);
                        __builtin_vsx_disassemble_pair(c7, &C7);
                        __builtin_vsx_disassemble_pair(c8, &C8);

                        t1 = vec_mergeh(c1[0], c2[0]);
                        t2 = vec_mergeh(c3[0], c4[0]);
                        t3 = vec_mergeh(c5[0], c6[0]);
                        t4 = vec_mergeh(c7[0], c8[0]);
                        t5 = vec_xxpermdi(t1, t2, 0);
                        t6 = vec_xxpermdi(t3, t4, 0);
                        t7 = vec_xxpermdi(t1, t2, 3);
                        t8 = vec_xxpermdi(t3, t4, 3);
                        vec_xst(t5, 0, boffset);
                        vec_xst(t6, 0, boffset+4);
                        vec_xst(t7, 0, boffset+8);
                        vec_xst(t8, 0, boffset+12);

                        t1 = vec_mergel(c1[0], c2[0]);
                        t2 = vec_mergel(c3[0], c4[0]);
                        t3 = vec_mergel(c5[0], c6[0]);
                        t4 = vec_mergel(c7[0], c8[0]);
                        t5 = vec_xxpermdi(t1, t2, 0);
                        t6 = vec_xxpermdi(t3, t4, 0);
                        t7 = vec_xxpermdi(t1, t2, 3);
                        t8 = vec_xxpermdi(t3, t4, 3);
                        vec_xst(t5, 0, boffset+16);
                        vec_xst(t6, 0, boffset+20);
                        vec_xst(t7, 0, boffset+24);
                        vec_xst(t8, 0, boffset+28);

                        t1 = vec_mergeh(c1[1], c2[1]);
                        t2 = vec_mergeh(c3[1], c4[1]);
                        t3 = vec_mergeh(c5[1], c6[1]);
                        t4 = vec_mergeh(c7[1], c8[1]);
                        t5 = vec_xxpermdi(t1, t2, 0);
                        t6 = vec_xxpermdi(t3, t4, 0);
                        t7 = vec_xxpermdi(t1, t2, 3);
                        t8 = vec_xxpermdi(t3, t4, 3);
                        vec_xst(t5, 0, boffset+32);
                        vec_xst(t6, 0, boffset+36);
                        vec_xst(t7, 0, boffset+40);
                        vec_xst(t8, 0, boffset+44);

                        t1 = vec_mergel(c1[1], c2[1]);
                        t2 = vec_mergel(c3[1], c4[1]);
                        t3 = vec_mergel(c5[1], c6[1]);
                        t4 = vec_mergel(c7[1], c8[1]);
                        t5 = vec_xxpermdi(t1, t2, 0);
                        t6 = vec_xxpermdi(t3, t4, 0);
                        t7 = vec_xxpermdi(t1, t2, 3);
                        t8 = vec_xxpermdi(t3, t4, 3);
                        vec_xst(t5, 0, boffset+48);
                        vec_xst(t6, 0, boffset+52);
                        vec_xst(t7, 0, boffset+56);
                        vec_xst(t8, 0, boffset+60);

                        aoffset1 += 8*lda;
                        aoffset2 += 8*lda;
                        aoffset3 += 8*lda;
                        aoffset4 += 8*lda;
                        boffset += 64;
                        i--;
                    } while(i > 0);
                }
                if (cols & 4) {
                    vector float c1, c2, c3, c4, c5, c6, c7, c8;
                    vector float t1, t2, t3, t4, t5, t6, t7, t8;
                    c1 = vec_xl(0, aoffset1);
                    c2 = vec_xl(0, aoffset2);
                    c3 = vec_xl(0, aoffset3);
                    c4 = vec_xl(0, aoffset4);
                    c5 = vec_xl(0, aoffset5);
                    c6 = vec_xl(0, aoffset6);
                    c7 = vec_xl(0, aoffset7);
                    c8 = vec_xl(0, aoffset8);

                    t1 = vec_mergeh(c1, c2);
                    t2 = vec_mergeh(c3, c4);
                    t3 = vec_mergeh(c5, c6);
                    t4 = vec_mergeh(c7, c8);
                    t5 = vec_xxpermdi(t1, t2, 0);
                    t6 = vec_xxpermdi(t3, t4, 0);
                    t7 = vec_xxpermdi(t1, t2, 3);
                    t8 = vec_xxpermdi(t3, t4, 3);
                    vec_xst(t5, 0, boffset);
                    vec_xst(t6, 0, boffset+4);
                    vec_xst(t7, 0, boffset+8);
                    vec_xst(t8, 0, boffset+12);

                    t1 = vec_mergel(c1, c2);
                    t2 = vec_mergel(c3, c4);
                    t3 = vec_mergel(c5, c6);
                    t4 = vec_mergel(c7, c8);
                    t5 = vec_xxpermdi(t1, t2, 0);
                    t6 = vec_xxpermdi(t3, t4, 0);
                    t7 = vec_xxpermdi(t1, t2, 3);
                    t8 = vec_xxpermdi(t3, t4, 3);
                    vec_xst(t5, 0, boffset+16);
                    vec_xst(t6, 0, boffset+20);
                    vec_xst(t7, 0, boffset+24);
                    vec_xst(t8, 0, boffset+28);
                }
            j--;
            } while(j > 0);
        }

        if (rows & 4) {
            aoffset1 = aoffset;
            aoffset2 = aoffset1 + lda;
            aoffset3 = aoffset2 + lda;
            aoffset4 = aoffset3 + lda;
            aoffset += 4 * lda;
            i = (cols >> 3);
            if (i > 0) {
                __vector_pair C1, C2, C3, C4;
                vector float c1[2], c2[2], c3[2], c4[2];
                vector float t1, t2, t3, t4, t5, t6, t7, t8;
                do {
                    C1 = __builtin_vsx_lxvp(0, (__vector_pair*)aoffset1);
                    C2 = __builtin_vsx_lxvp(0, (__vector_pair*)aoffset2);
                    C3 = __builtin_vsx_lxvp(0, (__vector_pair*)aoffset3);
                    C4 = __builtin_vsx_lxvp(0, (__vector_pair*)aoffset4);
                    __builtin_vsx_disassemble_pair(c1, &C1);
                    __builtin_vsx_disassemble_pair(c2, &C2);
                    __builtin_vsx_disassemble_pair(c3, &C3);
                    __builtin_vsx_disassemble_pair(c4, &C4);

                    t1 = vec_mergeh(c1[0], c2[0]);
                    t2 = vec_mergeh(c3[0], c4[0]);
                    t3 = vec_mergel(c1[0], c2[0]);
                    t4 = vec_mergel(c3[0], c4[0]);
                    t5 = vec_xxpermdi(t1, t2, 0);
                    t6 = vec_xxpermdi(t1, t2, 3);
                    t7 = vec_xxpermdi(t3, t4, 0);
                    t8 = vec_xxpermdi(t3, t4, 3);
                    vec_xst(t5, 0, boffset);
                    vec_xst(t6, 0, boffset+4);
                    vec_xst(t7, 0, boffset+8);
                    vec_xst(t8, 0, boffset+12);

                    t1 = vec_mergeh(c1[1], c2[1]);
                    t2 = vec_mergeh(c3[1], c4[1]);
                    t3 = vec_mergel(c1[1], c2[1]);
                    t4 = vec_mergel(c3[1], c4[1]);
                    t5 = vec_xxpermdi(t1, t2, 0);
                    t6 = vec_xxpermdi(t1, t2, 3);
                    t7 = vec_xxpermdi(t3, t4, 0);
                    t8 = vec_xxpermdi(t3, t4, 3);
                    vec_xst(t5, 0, boffset+16);
                    vec_xst(t6, 0, boffset+20);
                    vec_xst(t7, 0, boffset+24);
                    vec_xst(t8, 0, boffset+28);

                    aoffset1 += 8*lda;
                    aoffset2 += 8*lda;
                    aoffset3 += 8*lda;
                    aoffset4 += 8*lda;
                    boffset += 32;
                    i--;
                } while(i > 0);
            }

            if (cols & 4) {
                vector float c1, c2, c3, c4;
                vector float t1, t2, t3, t4;
                c1 = vec_xl(0, aoffset1);
                c2 = vec_xl(0, aoffset2);
                c3 = vec_xl(0, aoffset3);
                c4 = vec_xl(0, aoffset4);

                t1 = vec_mergeh(c1, c2);
                t2 = vec_mergeh(c3, c4);
                t3 = vec_xxpermdi(t1, t2, 0);
                t4 = vec_xxpermdi(t1, t2, 3);
                vec_xst(t3, 0, boffset);
                vec_xst(t4, 0, boffset+4);

                t1 = vec_mergel(c1, c2);
                t2 = vec_mergel(c3, c4);
                t3 = vec_xxpermdi(t1, t2, 0);
                t4 = vec_xxpermdi(t1, t2, 3);
                vec_xst(t3, 0, boffset+8);
                vec_xst(t4, 0, boffset+12);
            }
        }
        if (rows & 3) {
            aoffset1 = aoffset;
            aoffset2 = aoffset1 + lda;
            aoffset3 = aoffset2 + lda;
            if (cols & 4) {
                vector float c1, c2, c3, c4 = {0};
                vector float t1, t2, t3, t4;
                c1 = vec_xl(0, aoffset1);
                c2 = vec_xl(0, aoffset2);
                c3 = vec_xl(0, aoffset3);

                t1 = vec_mergeh(c1, c2);
                t2 = vec_mergeh(c3, c4);
                t3 = vec_xxpermdi(t1, t2, 0);
                t4 = vec_xxpermdi(t1, t2, 3);
                vec_xst(t3, 0, boffset);
                vec_xst(t4, 0, boffset+4);

                t1 = vec_mergel(c1, c2);
                t2 = vec_mergel(c3, c4);
                t3 = vec_xxpermdi(t1, t2, 0);
                t4 = vec_xxpermdi(t1, t2, 3);
                vec_xst(t3, 0, boffset+8);
                vec_xst(t4, 0, boffset+12);
            }
        }
    }

    void KERNEL_4x4(int64_t ii, int64_t jj) {
        vec_t vec_A[4], vec_B[4], vec_C[4];
        acc_t acc_0;
        __builtin_mma_xxsetaccz(&acc_0);
        for (int l = 0; l < k; l+=4) {
            READ_BLOCK(A+(ii*lda)+l, lda, 4, 4, (float*)vec_A);
            READ_BLOCK(B+(jj*ldb)+l, ldb, 4, 4, (float*)vec_B);
            __builtin_mma_xvf32gerpp(&acc_0, vec_A[0], vec_B[0]);
            __builtin_mma_xvf32gerpp(&acc_0, vec_A[1], vec_B[1]);
            __builtin_mma_xvf32gerpp(&acc_0, vec_A[2], vec_B[2]);
            __builtin_mma_xvf32gerpp(&acc_0, vec_A[3], vec_B[3]);
        }
        SAVE_ACC(&acc_0, ii, jj);
    }

    void KERNEL_4x8(int64_t ii, int64_t jj) {
        vec_t vec_A[4], vec_B[8], vec_C[4];
        acc_t acc_0, acc_1;
        __builtin_mma_xxsetaccz(&acc_0);
        __builtin_mma_xxsetaccz(&acc_1);
        for (int64_t l = 0; l < k; l+=4) {
            READ_BLOCK(A+(ii*lda)+l, lda, 4, 4, (float*)vec_A);
            READ_BLOCK(B+(jj*ldb)+l, ldb, 8, 4, (float*)vec_B);
            __builtin_mma_xvf32gerpp(&acc_0, vec_A[0], (vec_t)vec_B[0]);
            __builtin_mma_xvf32gerpp(&acc_1, vec_A[0], (vec_t)vec_B[1]);
            __builtin_mma_xvf32gerpp(&acc_0, vec_A[1], (vec_t)vec_B[2]);
            __builtin_mma_xvf32gerpp(&acc_1, vec_A[1], (vec_t)vec_B[3]);
            __builtin_mma_xvf32gerpp(&acc_0, vec_A[2], (vec_t)vec_B[4]);
            __builtin_mma_xvf32gerpp(&acc_1, vec_A[2], (vec_t)vec_B[5]);
            __builtin_mma_xvf32gerpp(&acc_0, vec_A[3], (vec_t)vec_B[6]);
            __builtin_mma_xvf32gerpp(&acc_1, vec_A[3], (vec_t)vec_B[7]);
        }
        SAVE_ACC(&acc_0, ii, jj);
        SAVE_ACC(&acc_1, ii, jj+4);
    }

    void KERNEL_8x4(int64_t ii, int64_t jj) {
        vec_t vec_A[8], vec_B[4], vec_C[4];
        acc_t acc_0, acc_1;
        __builtin_mma_xxsetaccz(&acc_0);
        __builtin_mma_xxsetaccz(&acc_1);
        for (int64_t l = 0; l < k; l+=4) {
            READ_BLOCK(A+(ii*lda)+l, lda, 8, 4, (float*)vec_A);
            READ_BLOCK(B+(jj*ldb)+l, ldb, 4, 4, (float*)vec_B);
            __builtin_mma_xvf32gerpp(&acc_0, (vec_t)vec_A[0], vec_B[0]);
            __builtin_mma_xvf32gerpp(&acc_1, (vec_t)vec_A[1], vec_B[0]);
            __builtin_mma_xvf32gerpp(&acc_0, (vec_t)vec_A[2], vec_B[1]);
            __builtin_mma_xvf32gerpp(&acc_1, (vec_t)vec_A[3], vec_B[1]);
            __builtin_mma_xvf32gerpp(&acc_0, (vec_t)vec_A[4], vec_B[2]);
            __builtin_mma_xvf32gerpp(&acc_1, (vec_t)vec_A[5], vec_B[2]);
            __builtin_mma_xvf32gerpp(&acc_0, (vec_t)vec_A[6], vec_B[3]);
            __builtin_mma_xvf32gerpp(&acc_1, (vec_t)vec_A[7], vec_B[3]);
        }
        SAVE_ACC(&acc_0, ii, jj);
        SAVE_ACC(&acc_1, ii+4, jj);
    }

    void KERNEL_8x8(int64_t ii, int64_t jj) {
        vec_t vec_A[16], vec_B[16], vec_C[4];
        acc_t acc_0, acc_1, acc_2, acc_3;
        __builtin_mma_xxsetaccz(&acc_0);
        __builtin_mma_xxsetaccz(&acc_1);
        __builtin_mma_xxsetaccz(&acc_2);
        __builtin_mma_xxsetaccz(&acc_3);
        for (int l = 0; l < k; l+=8) {
            READ_BLOCK(A+(ii*lda)+l, lda, 8, 8, (float*)vec_A);
            READ_BLOCK(B+(jj*ldb)+l, ldb, 8, 8, (float*)vec_B);
            for(int x = 0; x < 16; x+=2) {
                __builtin_mma_xvf32gerpp(&acc_0, (vec_t)vec_A[x], vec_B[x]);
                __builtin_mma_xvf32gerpp(&acc_1, (vec_t)vec_A[x], vec_B[x+1]);
                __builtin_mma_xvf32gerpp(&acc_2, (vec_t)vec_A[x+1], vec_B[x]);
                __builtin_mma_xvf32gerpp(&acc_3, (vec_t)vec_A[x+1], vec_B[x+1]);
            }
        }
        SAVE_ACC(&acc_0, ii, jj);
        SAVE_ACC(&acc_1, ii, jj+4);
        SAVE_ACC(&acc_2, ii+4, jj);
        SAVE_ACC(&acc_3, ii+4, jj+4);
    }

    void mnpack(int64_t m0, int64_t m, int64_t n0, int64_t n) {
        int64_t mc, nc, mp, np;
        int m_rem = MIN(m - m0, 16);
        int n_rem = MIN(n - n0, 16);
        if (m_rem >= 16 && n_rem >= 8) {
            mc = 8;
            nc = 8;
            gemm<8,8>(m0, m, n0, n);
        } else if(m_rem >= 8 && n_rem >= 16) {
            mc = 8;
            nc = 8;
            gemm<8,8>(m0, m, n0, n);
        } else if (m_rem >= 8 && n_rem >= 8) {
            mc = 8;
            nc = 8;
            gemm<8,8>(m0, m, n0, n);
        } else if (m_rem >= 4 && n_rem >= 8) {
            mc = 4;
            nc = 8;
            gemm<4,8>(m0, m, n0, n);
        } else if (m_rem >= 8 && n_rem >= 4) {
            mc = 8;
            nc = 4;
            gemm<8,4>(m0, m, n0, n);
        } else if (m_rem >= 4 && n_rem >= 4) {
            mc = 4;
            nc = 4;
            gemm<4,4>(m0, m, n0, n);
        } else if ((m_rem < 4) && (n_rem > 4)) {
            nc = 4;
            switch(m_rem) {
                case 1:
                    mc = 1;
                    gemm_small(m0, m, n0, n, mc, nc);
                    break;
                case 2:
                    mc = 2;
                    gemm_small(m0, m, n0, n, mc, nc);
                    break;
                case 3:
                    mc = 3;
                    gemm_small(m0, m, n0, n, mc, nc);
                    break;
                default:
                    return;
            }
        } else if ((m_rem > 4) && (n_rem < 4)) {
            mc = 4;
            switch(n_rem) {
                case 1:
                    nc = 1;
                    gemm_small(m0, m, n0, n, mc, nc);
                    break;
                case 2:
                    nc = 2;
                    gemm_small(m0, m, n0, n, mc, nc);
                    break;
                case 3:
                    nc = 3;
                    gemm_small(m0, m, n0, n, mc, nc);
                    break;
                default:
                    return;
            }
        } else {
            switch((m_rem << 4) | n_rem) {
                case 0x43:
                    mc = 4;
                    nc = 3;
                    gemm_small(m0, m, n0, n, mc, nc);
                    break;
                case 0x42:
                    mc = 4;
                    nc = 2;
                    gemm_small(m0, m, n0, n, mc, nc);
                    break;
                case 0x41:
                    mc = 4;
                    nc = 1;
                    gemm_small(m0, m, n0, n, mc, nc);
                    break;
                case 0x34:
                    mc = 3;
                    nc = 4;
                    gemm_small(m0, m, n0, n, mc, nc);
                    break;
                case 0x33:
                    mc = 3;
                    nc = 3;
                    gemm_small(m0, m, n0, n, mc, nc);
                    break;
                case 0x32:
                    mc = 3;
                    nc = 2;
                    gemm_small(m0, m, n0, n, mc, nc);
                    break;
                case 0x31:
                    mc = 3;
                    nc = 1;
                    gemm_small(m0, m, n0, n, mc, nc);
                    break;
                case 0x24:
                    mc = 2;
                    nc = 4;
                    gemm_small(m0, m, n0, n, mc, nc);
                    break;
                case 0x23:
                    mc = 2;
                    nc = 3;
                    gemm_small(m0, m, n0, n, mc, nc);
                    break;
                case 0x22:
                    mc = 2;
                    nc = 2;
                    gemm_small(m0, m, n0, n, mc, nc);
                    break;
                case 0x21:
                    mc = 2;
                    nc = 1;
                    gemm_small(m0, m, n0, n, mc, nc);
                    break;
                case 0x14:
                    mc = 1;
                    nc = 4;
                    gemm_small(m0, m, n0, n, mc, nc);
                    break;
                case 0x13:
                    mc = 1;
                    nc = 3;
                    gemm_small(m0, m, n0, n, mc, nc);
                    break;
                case 0x12:
                    mc = 1;
                    nc = 2;
                    gemm_small(m0, m, n0, n, mc, nc);
                    break;
                case 0x11:
                    mc = 1;
                    nc = 1;
                    gemm_small(m0, m, n0, n, mc, nc);
                    break;
                default:
                    return;
            }
        }
        mp = m0 + (m - m0) / mc * mc;
        np = n0 + (n - n0) / nc * nc;
        mnpack(mp, m, n0, np);
        mnpack(m0, m, np, n);
    }

     void gemm_small(int64_t m0, int64_t m, int64_t n0, int64_t n, int RM, int RN) {
        int64_t ytiles = (m - m0) / RM;
        int64_t xtiles = (n - n0) / RN;
        int64_t tiles = xtiles * ytiles;
        int64_t duty = (tiles + nth - 1) / nth;
        int64_t start = duty * ith;
        int64_t end = start + duty;
        if (end > tiles)
            end = tiles;
        for (int64_t job = start; job < end; ++job) {
            int64_t ii = m0 + job / xtiles * RM;
            int64_t jj = n0 + job % xtiles * RN;
            vec_t vec_C[4];
            acc_t acc_0;
            __builtin_mma_xxsetaccz(&acc_0);
            vec_t vec_A[4], vec_B[4];
            for (int l=0; l<k; l+=4) {
                if (RN >= 4 && RM == 1) {
                    float* a = const_cast<float*>(A+(ii)*lda+l);
                    READ_BLOCK(B+(jj*ldb)+l, ldb, 4, 4, (float*)vec_B);
                    vec_A[0] = (vec_t)vec_xl(0,a);
                    vec_A[1] = (vec_t)vec_splats(*((float*)&vec_A+1));
                    vec_A[2] = (vec_t)vec_splats(*((float*)&vec_A+2));
                    vec_A[3] = (vec_t)vec_splats(*((float*)&vec_A+3));
                } else {
                    READ_BLOCK(A+(ii*lda)+l, lda, RM, 4, (float*)vec_A);
                    READ_BLOCK(B+(jj*ldb)+l, ldb, RN, 4, (float*)vec_B);
                }
                __builtin_mma_xvf32gerpp(&acc_0, vec_A[0], vec_B[0]);
                __builtin_mma_xvf32gerpp(&acc_0, vec_A[1], vec_B[1]);
                __builtin_mma_xvf32gerpp(&acc_0, vec_A[2], vec_B[2]);
                __builtin_mma_xvf32gerpp(&acc_0, vec_A[3], vec_B[3]);
            }
            __builtin_mma_disassemble_acc(vec_C, &acc_0);
            for (int I = 0; I < RM; I++) {
                for (int J = 0; J < RN; J++) {
                    *((float*)(C+ii+((jj+J)*ldc)+I)) = *((float*)&vec_C[I]+J);
                }
            }
       }
    }

    template <int RM, int RN>
    NOINLINE void gemm(int64_t m0, int64_t m, int64_t n0, int64_t n) {
        int64_t ytiles = (m - m0) / RM;
        int64_t xtiles = (n - n0) / RN;
        int64_t tiles = xtiles * ytiles;
        int64_t duty = (tiles + nth - 1) / nth;
        int64_t start = duty * ith;
        int64_t end = start + duty;
        if (RM == 4 && RN == 4) {
            kernel = &tinyBLAS_PPC::KERNEL_4x4;
        } else if (RM == 4 && RN == 8) {
            kernel = &tinyBLAS_PPC::KERNEL_4x8;
        } else if (RM == 8 && RN == 4) {
            kernel = &tinyBLAS_PPC::KERNEL_8x4;
        } else if (RM == 8 && RN == 8) {
            kernel = &tinyBLAS_PPC::KERNEL_8x8;
        }
        if (end > tiles)
            end = tiles;
        for (int64_t job = start; job < end; ++job) {
            int64_t ii = m0 + job / xtiles * RM;
            int64_t jj = n0 + job % xtiles * RN;
            (this->*kernel)(ii, jj);
        }
    }

    const TA *const A;
    const TB *const B;
    TC *C;
    TA *At;
    TB *Bt;
    const int64_t k;
    const int64_t lda;
    const int64_t ldb;
    const int64_t ldc;
    const int ith;
    const int nth;
};
#endif
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
 *                     0, 1,
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
 * @param Atype is GGML data type of `A`
 * @param Btype is GGML data type of `B`
 * @param Ctype is GGML data type of `C`
 * @return true if this function was able to service the matmul request
 */
bool llamafile_sgemm(int64_t m, int64_t n, int64_t k, const void *A, int64_t lda, const void *B, int64_t ldb, void *C,
                     int64_t ldc, int ith, int nth, int Atype, int Btype, int Ctype) {

    assert(m >= 0);
    assert(n >= 0);
    assert(k >= 0);
    assert(lda >= k);
    assert(ldb >= k);
    assert(ldc >= m);
    assert(nth > 0);
    assert(ith < nth);

    // only enable sgemm for prompt processing
    if (n < 2)
        return false;

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
        tb.matmul(m, n);
        return true;
#elif defined(__AVX__) || defined(__AVX2__)
        if (k % 8)
            return false;
        tinyBLAS<8, __m256, __m256, float, float, float> tb{
            k, (const float *)A, lda,
            (const float *)B, ldb,
            (float *)C, ldc,
            ith, nth};
        tb.matmul(m, n);
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
        tb.matmul(m, n);
        return true;
#elif defined(__MMA__)
        if (k % 8)
            return false;
        tinyBLAS_PPC<float, float, float> tb{
            k, (const float *)A, lda,
            (const float *)B, ldb,
            (float *)C, ldc,
            ith, nth};
        tb.matmul(m, n);
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
        tb.matmul(m, n);
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
        tb.matmul(m, n);
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
        tb.matmul(m, n);
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
        tb.matmul(m, n);
        return true;
#else
        return false;
#endif
    }

    case GGML_TYPE_Q8_0: {
        if (Btype != GGML_TYPE_Q8_0)
           return false;
#if defined(__AVX2__) || defined(__AVX512F__) || defined(__AVX__)
        tinyBLAS_Q0_AVX<block_q8_0, block_q8_0, float> tb{
            k, (const block_q8_0 *)A, lda,
            (const block_q8_0 *)B, ldb,
            (float *)C, ldc,
            ith, nth};
        tb.matmul(m, n);
        return true;
#elif defined(__ARM_FEATURE_DOTPROD)
        tinyBLAS_Q0_ARM<block_q8_0> tb{
            k, (const block_q8_0 *)A, lda,
            (const block_q8_0 *)B, ldb,
            (float *)C, ldc,
            ith, nth};
        tb.matmul(m, n);
        return true;
#else
        return false;
#endif
    }

    case GGML_TYPE_Q4_0: {
        if (Btype != GGML_TYPE_Q8_0)
            return false;
#if defined(__AVX2__) || defined(__AVX512F__) || defined(__AVX__)
        tinyBLAS_Q0_AVX<block_q4_0, block_q8_0, float> tb{
            k, (const block_q4_0 *)A, lda,
            (const block_q8_0 *)B, ldb,
            (float *)C, ldc,
            ith, nth};
        tb.matmul(m, n);
        return true;
#elif defined(__ARM_FEATURE_DOTPROD)
        tinyBLAS_Q0_ARM<block_q4_0> tb{
            k, (const block_q4_0 *)A, lda,
            (const block_q8_0 *)B, ldb,
            (float *)C, ldc,
            ith, nth};
        tb.matmul(m, n);
        return true;
#else
        return false;
#endif
    }

    case GGML_TYPE_Q5_0: {
        if (Btype != GGML_TYPE_Q8_0)
            return false;
#if defined(__AVX2__) || defined(__AVX512F__) || defined(__AVX__)
        tinyBLAS_Q0_AVX<block_q5_0, block_q8_0, float> tb{
            k, (const block_q5_0 *)A, lda,
            (const block_q8_0 *)B, ldb,
            (float *)C, ldc,
            ith, nth};
        tb.matmul(m, n);
        return true;
#else
        return false;
#endif
    }

    case GGML_TYPE_IQ4_NL: {
        if (Btype != GGML_TYPE_Q8_0)
            return false;
#if defined(__AVX2__) || defined(__AVX512F__) || defined(__AVX__)
        tinyBLAS_Q0_AVX<block_iq4_nl, block_q8_0, float> tb{
            k, (const block_iq4_nl *)A, lda,
            (const block_q8_0 *)B, ldb,
            (float *)C, ldc,
            ith, nth};
        tb.matmul(m, n);
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
    (void)Atype;
    (void)Btype;
    (void)Ctype;
}
