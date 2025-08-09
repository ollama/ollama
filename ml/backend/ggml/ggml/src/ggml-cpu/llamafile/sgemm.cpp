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
#include "simd-mappings.h"

#include <array>
#include <type_traits>

#ifdef _MSC_VER
#define NOINLINE __declspec(noinline)
#else
#define NOINLINE __attribute__((__noinline__))
#endif

#if defined(__ARM_NEON) || defined(__AVX512F__) || defined(__VXE__) || defined(__VXE2__)
#define VECTOR_REGISTERS 32
#else
#define VECTOR_REGISTERS 16
#endif

#define MM256_SET_M128I(a, b) _mm256_insertf128_si256(_mm256_castsi128_si256(b), (a), 1)

namespace {

inline float unhalf(ggml_fp16_t d) {
    return GGML_CPU_FP16_TO_FP32(d);
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

#if defined(__VXE__) || defined(__VXE2__)
inline float32x4_t add(float32x4_t x, float32x4_t y) { return vec_add(x, y); }
inline float32x4_t sub(float32x4_t x, float32x4_t y) { return vec_sub(x, y); }
inline float32x4_t mul(float32x4_t x, float32x4_t y) { return vec_mul(x, y); }
#endif

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
#if defined(__AVX512BF16__)
template <>
inline __m512 madd(__m512bh a, __m512bh b, __m512 c) {
    return _mm512_dpbf16_ps(c, a, b);
}
template <>
inline __m256 madd(__m256bh a, __m256bh b, __m256 c) {
    return _mm256_dpbf16_ps(c, a, b);
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

#if defined(__VXE__) || defined(__VXE2__)
template <>
inline float32x4_t madd(float32x4_t a, float32x4_t b, float32x4_t c) {
    return vec_madd(a, b, c);
}
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

#if defined(__VXE__) || defined(__VXE2__)
inline float hsum(float32x4_t x) {
    float32x4_t tmp = x + vec_reve(x);
    return tmp[0] + tmp[1];
}
#endif

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
// FIXME: this should check for __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
template <> inline float16x8_t load(const ggml_fp16_t *p) {
    return vld1q_f16((const float16_t *)p);
}
template <> inline float32x4_t load(const ggml_fp16_t *p) {
    return vcvt_f32_f16(vld1_f16((const float16_t *)p));
}
#endif // _MSC_VER
#endif // __ARM_NEON

#if defined(__VXE__) || defined(__VXE2__)
template <> inline float32x4_t load(const ggml_fp16_t * p) {
    float tmp[4];

    for (int i = 0; i < 4; i++) {
        tmp[i] = GGML_CPU_FP16_TO_FP32(p[i]);
    }

    return vec_xl(0, (const float *)(tmp));
}
template <> inline float32x4_t load(const float * p) {
    return vec_xl(0, p);
}
#endif

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

#if defined(__AVX2__) || defined(__AVX512F__)
template <> inline __m256 load(const ggml_bf16_t *p) {
    return _mm256_castsi256_ps(
        _mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i *)p)), 16));
}
#endif // __AVX2__

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
template <> inline __m512 load(const ggml_bf16_t *p) {
    return _mm512_castsi512_ps(
        _mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((const __m256i *)p)), 16));
}
#endif // __AVX512F__

#if defined(__AVX512BF16__)
template <> inline __m512bh load(const ggml_bf16_t *p) {
    return (__m512bh)_mm512_loadu_ps((const float *)p);
}
template <> inline __m256bh load(const ggml_bf16_t *p) {
    return (__m256bh)_mm256_loadu_ps((const float *)p);
}
template <> inline __m512bh load(const float *p) {
    return _mm512_cvtne2ps_pbh(_mm512_loadu_ps(p + 16), _mm512_loadu_ps(p));
}
template <> inline __m256bh load(const float *p) {
    return _mm512_cvtneps_pbh(_mm512_loadu_ps(p));
}
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////
// FLOATING POINT MATRIX MULTIPLICATION

template <int M>
static inline int64_t BLOCK_SIZE(size_t m) {
    const int64_t NB_BLOC_M = (m + M - 1) / M;
    return (m % NB_BLOC_M == 0) ? m / NB_BLOC_M : (m / NB_BLOC_M) + 1;
}

static constexpr inline int64_t BLOC_POS(int64_t ib, int64_t ibN, int64_t bloc_size) {
    return ib < ibN ? ib * bloc_size : ibN * bloc_size + (ib - ibN) * (bloc_size - 1);
}

template <int KN, typename D, typename V, typename TA, typename TB, typename TC>
class tinyBLAS {
  public:
    tinyBLAS(const ggml_compute_params * params, int64_t k,
             const TA *A, int64_t lda,
             const TB *B, int64_t ldb,
             TC *C, int64_t ldc)
        : params(params), A(A), B(B), C(C), k(k), lda(lda), ldb(ldb), ldc(ldc) {
    }

    bool matmul(int64_t m, int64_t n) {
        if (k % KN != 0)
            return false;
        // compute RM for only need tile with size RM&RM-1
#if VECTOR_REGISTERS == 32
        if (m % 16 == 0 && (m/16 >= params->nth)) {
            const int64_t SIZE_N = BLOCK_SIZE<6>(n);
            mnpack<4, 6, 4>(m, n, SIZE_N, 12);
            return true;
        }
        if (m % 8 == 0 ) {
            const int64_t SIZE_N = BLOCK_SIZE<6>(n);
            mnpack<4, 6, 2>(m, n, SIZE_N, 12);
            return true;
        }
        if (m % 4 == 0) {
            const int64_t SIZE_N = BLOCK_SIZE<6>(n);
            mnpack<4, 6, 1>(m, n, SIZE_N, 12);
            return true;
        }
#else  // VECTOR_REGISTERS == 16
        if (m % 16 == 0 && (m/16 >= params->nth)) {
            const int64_t SIZE_N = BLOCK_SIZE<3>(n);
            mnpack<4, 3, 4>(m, n, SIZE_N, 24);
            return true;
        }
        if (m % 8 == 0 ) {
            const int64_t SIZE_N = BLOCK_SIZE<3>(n);
            mnpack<4, 3, 2>(m, n, SIZE_N, 24);
            return true;
        }
        if (m % 4 == 0) {
            const int64_t SIZE_N = BLOCK_SIZE<3>(n);
            mnpack<4, 3, 1>(m, n, SIZE_N, 24);
            return true;
        }
#endif
        return false;
    }

  private:
    template <int RM, int RN, int BM>
    inline void mnpack(int64_t m, int64_t n, int64_t SIZE_N, int64_t BN) {
        if (SIZE_N == RN) {
            return gemm<RM, RN, BM>(m, n, BN);
        }
        if constexpr (RN > 1) {
            return mnpack<RM, RN-1, BM>(m, n, SIZE_N, BN);
        } else {
            GGML_LOG_ERROR("mnpack<%d, %d> bloc size not supported\n", RM, (int)SIZE_N);
            GGML_ASSERT(false); // we have miss something.
        }
    }

    template <int RM, int RN>
    inline void gemm_bloc(int64_t ii, int64_t jj) {
        D Cv[RN][RM] = {};
        for (int64_t l = 0; l < k; l += KN) {
            // help compiler for op order.
            if constexpr (RM <= RN) {
                V Av[RM];
                for (int64_t i = 0; i < RM; ++i) {
                    Av[i] = load<V>(A + lda * (ii + i) + l);
                }
                for (int64_t j = 0; j < RN; ++j) {
                    V Bv = load<V>(B + ldb * (jj + j) + l);
                    for (int64_t i = 0; i < RM; ++i) {
                        Cv[j][i] = madd(Av[i], Bv, Cv[j][i]);
                    }
                }
            } else {
                V Bv[RN];
                for (int64_t j = 0; j < RN; ++j) {
                    Bv[j] = load<V>(B + ldb * (jj + j) + l);
                }
                for (int64_t i = 0; i < RM; ++i) {
                    V Av = load<V>(A + lda * (ii + i) + l);
                    for (int64_t j = 0; j < RN; ++j) {
                        Cv[j][i] = madd(Av, Bv[j], Cv[j][i]);
                    }
                }
            }
        }
        for (int64_t j = 0; j < RN; ++j)
            for (int64_t i = 0; i < RM; ++i)
                C[ldc * (jj + j) + (ii + i)] = hsum(Cv[j][i]);
    }

    template <int RM, int RN, int BM>
    NOINLINE void gemm(int64_t m, int64_t n, int64_t BN) {
        GGML_ASSERT(m % (RM * BM) == 0);
        const int64_t ytiles = m / (RM * BM);
        const int64_t xtiles = (n + RN -1) / RN;
        const int64_t jj_RN = (xtiles - (xtiles * RN - n));

        // "round" bloc_size to "nearest" BN
        const int64_t NB_BN = xtiles < BN ? 1 : (xtiles + BN / 2) / BN;
        const int64_t SIZE_BN = xtiles % NB_BN == 0 ? xtiles / NB_BN : xtiles / NB_BN + 1;
        const int64_t jj_BN = (NB_BN - (NB_BN * SIZE_BN - xtiles));
        const int64_t nb_job = ytiles * NB_BN;

        if (params->ith == 0) {
            GGML_ASSERT( jj_BN * SIZE_BN + (NB_BN - jj_BN) * (SIZE_BN - 1) == xtiles);
            // Every thread starts at ith, so the first unprocessed chunk is nth.  This save a bit of coordination right at the start.
            ggml_threadpool_chunk_set(params->threadpool, params->nth);
        }

        ggml_barrier(params->threadpool);

        int64_t job = params->ith;
        while (job < nb_job) {
            const int64_t ii = (job % ytiles) * RM * BM;
            const int64_t jb =  job / ytiles;
            const int64_t jr0 = BLOC_POS(jb  , jj_BN, SIZE_BN);
            const int64_t jrN = BLOC_POS(jb+1, jj_BN, SIZE_BN);

            const int64_t jj0 = BLOC_POS(jr0, jj_RN, RN);
            const int64_t jj2 = BLOC_POS(jrN, jj_RN, RN);
            const int64_t jj1 = jj2 < jj_RN * RN ? jj2 : jj_RN * RN;

            for (int64_t bi = 0; bi < BM * RM; bi += RM) {
                int64_t jj = jj0;
                for (; jj < jj1; jj += RN) {
                    gemm_bloc<RM, RN>(ii + bi, jj);
                }
                if constexpr (RN > 1) {
                    for (; jj < jj2; jj += RN - 1) {
                        gemm_bloc<RM, RN-1>(ii + bi, jj);
                    }
                }
                GGML_ASSERT(jj == jj2);
            }

            job = ggml_threadpool_chunk_add(params->threadpool, 1);
        }

        ggml_barrier(params->threadpool);
        return;
    }

    const ggml_compute_params * params;
    const TA *const A;
    const TB *const B;
    TC *const C;
    const int64_t k;
    const int64_t lda;
    const int64_t ldb;
    const int64_t ldc;
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
        const int8_t kvalues_iq4nl[16] = {
            -127, -104, -83, -65,
            -49,  -35,  -22, -10,
              1,   13,   25,  38,
             53,   69,   89, 113
        };

        iq4nlt = _mm_loadu_si128((const __m128i *)kvalues_iq4nl);
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
#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
        res = _mm256_dpbusd_epi32(_mm256_setzero_si256(), u, s);
#elif defined(__AVXVNNI__)
        res = _mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), u, s);
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
    __m128i iq4nlt;
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
class tinyBLAS_BF16_PPC {
  public:
    tinyBLAS_BF16_PPC(int64_t k,
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
    void vector_permute_store(vec_t *c, int numVec, unsigned char *vecOffset) {
        vec_t t[8], s[8];
        vec_t swiz1 = {0, 1, 2, 3, 16, 17, 18, 19, 4, 5, 6, 7, 20, 21, 22, 23};
        vec_t swiz2 = {8, 9, 10, 11, 24, 25, 26, 27, 12, 13, 14, 15, 28, 29, 30, 31};
        vec_t swiz3 = {0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23};
        vec_t swiz4 = {8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31};

        if (numVec == 2) {
            t[0] = vec_perm(c[0], c[1], swiz1);
            t[1] = vec_perm(c[2], c[3], swiz1);
            s[0] = vec_perm(t[0], t[1], swiz3);
            s[1] = vec_perm(t[0], t[1], swiz4);
            vec_xst(s[0], 0, (vec_t*)vecOffset);
            vec_xst(s[1], 0, (vec_t*)(vecOffset + 16));
        } else if (numVec == 4) {
            t[0] = vec_perm(c[0], c[1], swiz1);
            t[1] = vec_perm(c[0], c[1], swiz2);
            t[2] = vec_perm(c[2], c[3], swiz1);
            t[3] = vec_perm(c[2], c[3], swiz2);
            s[0] = vec_perm(t[0], t[2], swiz3);
            s[1] = vec_perm(t[0], t[2], swiz4);
            s[2] = vec_perm(t[1], t[3], swiz3);
            s[3] = vec_perm(t[1], t[3], swiz4);
            for (int i = 0; i < 4; ++i)
                vec_xst(s[i], 0, (vec_t*)(vecOffset + i * 16));
        } else if (numVec == 8) {
            for (int i = 0; i < 4; i += 2) {
                t[i+0] = vec_perm(c[i+0], c[i+1], swiz1);
                t[i+1] = vec_perm(c[i+0], c[i+1], swiz2);
            }
            for (int i = 4; i < 8; i += 2) {
                t[i+0] = vec_perm(c[i+0], c[i+1], swiz1);
                t[i+1] = vec_perm(c[i+0], c[i+1], swiz2);
            }
            s[0] = vec_perm(t[0], t[2], swiz3);
            s[1] = vec_perm(t[0], t[2], swiz4);
            s[2] = vec_perm(t[1], t[3], swiz3);
            s[3] = vec_perm(t[1], t[3], swiz4);
            s[4] = vec_perm(t[4], t[6], swiz3);
            s[5] = vec_perm(t[4], t[6], swiz4);
            s[6] = vec_perm(t[5], t[7], swiz3);
            s[7] = vec_perm(t[5], t[7], swiz4);
            for (int i = 0; i < 8; ++i)
                vec_xst(s[i], 0, (vec_t*)(vecOffset + i * 16));
        }
    }

    void packNormal(const TA* a, int64_t lda, int rows, int cols, unsigned char* vec) {
        int64_t i, j;
        TA *aoffset = NULL;
        unsigned char *vecOffset = NULL;
        TA * aoffsets[8];
        vector unsigned char c_arr[8];
        aoffset = const_cast<TA*>(a);
        vecOffset = vec;
        j = (rows >> 3);
        if (j > 0) {
            do {
                if (cols == 4) {
                    aoffsets[0] = aoffset;
                    for (int it = 1; it < 4; ++it)
                        aoffsets[it] = aoffsets[it-1] + lda;
                    aoffset += 4 * lda;
                    for (int i = 0; i < 4; ++i)
                        c_arr[i] = vec_xl(0, (vector unsigned char*)aoffsets[i]);
                    vector_permute_store(c_arr, 4, vecOffset);
                    for (int i = 0; i<4; i++)
                        aoffsets[i] = aoffsets[i]+lda;
                    vecOffset +=64;
                }
                i = (cols >> 3);
                if (i > 0) {
                    aoffsets[0] = aoffset;
                    for (int it = 1; it < 8; ++it) {
                        aoffsets[it] = aoffsets[it-1] + lda;
                    }
                    aoffset += 8 * lda;
                    do {
                        for (int it = 0; it < 8; ++it)
                            c_arr[it] = vec_xl(0, (vector unsigned char*)aoffsets[it]);
                        vector_permute_store(c_arr, 8, vecOffset);
                        for (int it = 0; it < 8; ++it)
                            aoffsets[it] = aoffsets[it] + 8*lda;
                        vecOffset += 128;
                        i--;
                    } while(i > 0);
                }
                j--;
            } while(j > 0);
        }
        if (rows & 4) {
            aoffsets[0] = aoffset;
            for (int it = 1; it < 4; ++it)
                aoffsets[it] = aoffsets[it-1] + lda;
            aoffset += 4 * lda;
            if (cols == 4) {
                for (int it = 0; it < 4; ++it)
                    c_arr[it] = vec_xl(0, (vector unsigned char*)aoffsets[it]);
                vector_permute_store(c_arr, 2, vecOffset);
                for (int it = 0; it< 4; it++)
                    aoffsets[it] = aoffsets[it] + lda;
                vecOffset += 32;
            }
            i = (cols >> 3);
            if (i > 0) {
                do {
                    for (int it = 0; it < 4; ++it)
                        c_arr[it] = vec_xl(0, (vector unsigned char*)aoffsets[it]);
                    vector_permute_store(c_arr, 4, vecOffset);
                    for (int it = 0; it< 4; it++)
                        aoffsets[it] = aoffsets[it] + 8*lda;
                    vecOffset += 64;
                    i--;
                } while(i > 0);
            }
        }
        if (rows & 3) {
            aoffsets[0] = aoffset;
            for (int it = 1; it < 4; ++it)
                aoffsets[it] = aoffsets[it-1] + lda;
            if (cols == 4) {
                switch(rows) {
                    case 3: c_arr[2] = vec_xl(0, (vector unsigned char*)aoffsets[2]);
                    case 2: c_arr[1] = vec_xl(0, (vector unsigned char*)aoffsets[1]);
                    case 1: c_arr[0] = vec_xl(0, (vector unsigned char*)aoffsets[0]);
                        break;
                }
                vector_permute_store(c_arr, 2, vecOffset);
                for (int it = 0; it< 4; it++)
                     aoffsets[it] = aoffsets[it] + lda;
                vecOffset += 32;
            }
            i = (cols >> 3);
            if (i > 0) {
                do {
                    switch(rows) {
                        case 3: c_arr[2] = vec_xl(0, (vector unsigned char*)aoffsets[2]);
                        case 2: c_arr[1] = vec_xl(0, (vector unsigned char*)aoffsets[1]);
                        case 1: c_arr[0] = vec_xl(0, (vector unsigned char*)aoffsets[0]);
                            break;
                    }
                    vector_permute_store(c_arr, 4, vecOffset);
                    for (int it = 0; it <4; it++)
                         aoffsets[it] = aoffsets[it] + 8* lda;
                    vecOffset += 64;
                    i--;
                } while(i > 0);
            }
        }
    }

    void mnpack(int64_t m0, int64_t m, int64_t n0, int64_t n) {
        int64_t mc, nc, mp, np;
        int m_rem = MIN(m - m0, 8);
        int n_rem = MIN(n - n0, 8);

        if (m_rem >= 8 && n_rem >= 8) {
            mc = 8;
            nc = 8;
            gemm<8,8>(m0, m, n0, n);
        } else if (m_rem >= 4 && n_rem >= 8) {
            mc = 4;
            nc = 8;
            gemm<4,8>(m0, m, n0, n);
        } else if (m_rem >=8 && n_rem >=4){
                mc = 8;
                nc = 4;
                gemm<8,4>(m0, m, n0, n);
        } else if ((m_rem < 4) && (n_rem >= 8)) {
            nc = 8;
            switch(m_rem) {
                case 1:
                    mc = 1;
                    gemm_Mx8<1>(m0, m, n0, n);
                    break;
                case 2:
                    mc = 2;
                    gemm_Mx8<2>(m0, m, n0, n);
                    break;
                case 3:
                    mc = 3;
                    gemm_Mx8<3>(m0, m, n0, n);
                    break;
                default:
                    return;
            }
        } else if (m_rem >= 4 && n_rem >= 4) {
            mc = 4;
            nc = 4;
            gemm_small<4, 4>(m0, m, n0, n);
        } else if ((m_rem > 4) && (n_rem < 4)) {
            mc = 4;
            switch(n_rem) {
                case 1:
                    nc = 1;
                    gemm_small<4, 1>(m0, m, n0, n);
                    break;
                case 2:
                    nc = 2;
                    gemm_small<4, 2>(m0, m, n0, n);
                    break;
                case 3:
                    nc = 3;
                    gemm_small<4, 3>(m0, m, n0, n);
                    break;

                default:
                    return;
            }
        } else {
            switch((m_rem << 4) | n_rem) {
                case 0x43:
                    mc = 4;
                    nc = 3;
                    gemm_small<4, 3>(m0, m, n0, n);
                    break;
                case 0x42:
                    mc = 4;
                    nc = 2;
                    gemm_small<4, 2>(m0, m, n0, n);
                    break;
                case 0x41:
                    mc = 4;
                    nc = 1;
                    gemm_small<4, 1>(m0, m, n0, n);
                    break;
                case 0x34:
                    mc = 3;
                    nc = 4;
                    gemm_small<3, 4>(m0, m, n0, n);
                    break;
                case 0x33:
                    mc = 3;
                    nc = 3;
                    gemm_small<3, 3>(m0, m, n0, n);
                    break;
                case 0x32:
                    mc = 3;
                    nc = 2;
                    gemm_small<3, 2>(m0, m, n0, n);
                    break;
                case 0x31:
                    mc = 3;
                    nc = 1;
                    gemm_small<3, 1>(m0, m, n0, n);
                    break;
                case 0x24:
                    mc = 2;
                    nc = 4;
                    gemm_small<2,4>(m0, m, n0, n);
                    break;
                case 0x23:
                    mc = 2;
                    nc = 3;
                    gemm_small<2, 3>(m0, m, n0, n);
                    break;
                case 0x22:
                    mc = 2;
                    nc = 2;
                    gemm_small<2, 2>(m0, m, n0, n);
                    break;
                case 0x21:
                    mc = 2;
                    nc = 1;
                    gemm_small<2, 1>(m0, m, n0, n);
                    break;
                case 0x14:
                    mc = 1;
                    nc = 4;
                    gemm_small<1, 4>(m0, m, n0, n);
                    break;
                case 0x13:
                    mc = 1;
                    nc = 3;
                    gemm_small<1, 3>(m0, m, n0, n);
                    break;
                case 0x12:
                    mc = 1;
                    nc = 2;
                    gemm_small<1, 2>(m0, m, n0, n);
                    break;
                case 0x11:
                    mc = 1;
                    nc = 1;
                    gemm_small<1, 1>(m0, m, n0, n);
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

    void KERNEL_4x8(int64_t ii, int64_t jj) {
        vec_t vec_A[4], vec_B[8] , vec_C[4];
        acc_t acc_0, acc_1;
        __builtin_mma_xxsetaccz(&acc_0);
        __builtin_mma_xxsetaccz(&acc_1);
        for (int l = 0; l < k; l+=8) {
            packNormal((A+(ii*lda)+l), lda, 4, 8, (uint8_t*)vec_A);
            packNormal((B+(jj*ldb)+l), ldb, 8, 8, (uint8_t*)vec_B);
            for (int x = 0; x < 4; x++) {
                __builtin_mma_xvbf16ger2pp(&acc_0, vec_A[x], vec_B[x]);
                __builtin_mma_xvbf16ger2pp(&acc_1, vec_A[x], vec_B[x+4]);
            }
        }
        SAVE_ACC(&acc_0, ii, jj);
        SAVE_ACC(&acc_1, ii, jj+4);
    }

    void KERNEL_8x4(int64_t ii, int64_t jj) {
        vec_t vec_A[8], vec_B[4] , vec_C[4];
        acc_t acc_0, acc_1;
        __builtin_mma_xxsetaccz(&acc_0);
        __builtin_mma_xxsetaccz(&acc_1);
        for (int l = 0; l < k; l+=8) {
            packNormal((A+(ii*lda)+l), lda, 8, 8, (uint8_t*)vec_A);
            packNormal((B+(jj*ldb)+l), ldb, 8, 4, (uint8_t*)vec_B);
            for (int x = 0; x < 4; x++) {
                __builtin_mma_xvbf16ger2pp(&acc_0, vec_A[x], vec_B[x]);
                __builtin_mma_xvbf16ger2pp(&acc_1, vec_A[x+4], vec_B[x]);
            }
        }
        SAVE_ACC(&acc_0, ii, jj);
        SAVE_ACC(&acc_1, ii+4, jj);
    }


    void KERNEL_8x8(int64_t ii, int64_t jj) {
        vec_t vec_A[8], vec_B[8], vec_C[4];
        acc_t acc_0, acc_1, acc_2, acc_3;
        __builtin_mma_xxsetaccz(&acc_0);
        __builtin_mma_xxsetaccz(&acc_1);
        __builtin_mma_xxsetaccz(&acc_2);
        __builtin_mma_xxsetaccz(&acc_3);
        for (int l = 0; l < k; l+=8) {
            packNormal(A+(ii*lda)+l, lda, 8, 8, (uint8_t*)vec_A);
            packNormal(B+(jj*ldb)+l, ldb, 8, 8, (uint8_t*)vec_B);
            for (int x = 0; x < 4; x++) {
                __builtin_mma_xvbf16ger2pp(&acc_0, vec_A[x], vec_B[x]);
                __builtin_mma_xvbf16ger2pp(&acc_1, (vec_t)vec_A[x], (vec_t)vec_B[x+4]);
                __builtin_mma_xvbf16ger2pp(&acc_2, (vec_t)vec_A[x+4], (vec_t)vec_B[x]);
                __builtin_mma_xvbf16ger2pp(&acc_3, (vec_t)vec_A[x+4], (vec_t)vec_B[x+4]);
            }
        }

        SAVE_ACC(&acc_0, ii, jj);
        SAVE_ACC(&acc_1, ii, jj+4);
        SAVE_ACC(&acc_2, ii+4, jj);
        SAVE_ACC(&acc_3, ii+4, jj+4);
    }

    template<int RM, int RN>
    void gemm_small(int64_t m0, int64_t m, int64_t n0, int64_t n) {
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
            vec_t vec_A[2], vec_B[2];
            for (int l=0; l<k; l+=4) {
                packNormal(A+(ii*lda)+l, lda, RM, 4, (uint8_t*)vec_A);
                packNormal(B+(jj*ldb)+l, ldb, RN, 4, (uint8_t*)vec_B);
                for (int x = 0; x<2; x++) {
                    __builtin_mma_xvbf16ger2pp(&acc_0, vec_A[x], vec_B[x]);
                }
            }
            __builtin_mma_disassemble_acc(vec_C, &acc_0);
            for (int I = 0; I < RM; I++) {
                for (int J = 0; J < RN; J++) {
                    *((TC*)(C+ii+((jj+J)*ldc)+I)) = *((TC*)&vec_C[I]+J);
                }
            }
        }
    }

    template<int RM>
    void gemm_Mx8(int64_t m0, int64_t m, int64_t n0, int64_t n) {
        int RN = 8;
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
            acc_t acc_0, acc_1;
            __builtin_mma_xxsetaccz(&acc_0);
            __builtin_mma_xxsetaccz(&acc_1);
            vec_t vec_A[4], vec_B[8];
            for (int l=0; l<k; l+=8) {
                packNormal(A+(ii*lda)+l, lda, RM, 8, (uint8_t*)vec_A);
                packNormal(B+(jj*ldb)+l, ldb, RN, 8, (uint8_t*)vec_B);
                for (int x = 0; x<4; x++) {
                    __builtin_mma_xvbf16ger2pp(&acc_0, vec_A[x], vec_B[x]);
                    __builtin_mma_xvbf16ger2pp(&acc_1, vec_A[x], vec_B[x+4]);
                }
            }
            __builtin_mma_disassemble_acc(vec_C, &acc_0);
            for (int I = 0; I < RM; I++) {
                for (int J = 0; J < 4; J++) {
                    *((TC*)(C+ii+((jj+J)*ldc)+I)) = *((TC*)&vec_C[I]+J);
                }
            }
            __builtin_mma_disassemble_acc(vec_C, &acc_1);
            for (int I = 0; I < RM; I++) {
                for (int J = 0; J < 4; J++) {
                    *((TC*)(C+ii+((jj+4+J)*ldc)+I)) = *((TC*)&vec_C[I]+J);
                }
            }
        }
    }

    template<int RM, int RN>
    inline void kernel(int64_t ii, int64_t jj) {
       if constexpr(RM == 4 && RN == 8) {
          KERNEL_4x8(ii,jj);
       } else if constexpr(RM == 8 && RN == 8) {
          KERNEL_8x8(ii,jj);
       } else if constexpr(RM == 8 && RN == 4) {
          KERNEL_8x4(ii,jj);
       } else {
          assert(false && "RN/RM values not supported");
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
        if (end > tiles)
            end = tiles;
        for (int64_t job = start; job < end; ++job) {
            int64_t ii = m0 + job / xtiles * RM;
            int64_t jj = n0 + job % xtiles * RN;
            kernel<RM, RN>(ii, jj);
        }
    }

    const TA *const A;
    const TB *const B;
    TC *C;
    const int64_t k;
    const int64_t lda;
    const int64_t ldb;
    const int64_t ldc;
    const int ith;
    const int nth;
};

template <typename TA>
class tinyBLAS_Q0_PPC {
  public:
    tinyBLAS_Q0_PPC(int64_t k,
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

    inline void save_res(int ii, int jj, int idx, vector float* fin_res, int RM=4, int RN=4) {
       for (int I = 0; I < RM; I++) {
          for (int J = 0; J < RN; J++) {
             *((float*)(C+ii+((jj+J)*ldc)+I)) = *((float*)&fin_res[idx+I]+J);
          }
       }
    }

    template<int size>
    inline void compute(acc_t* ACC, int c_idx, int s_idx, std::array<int, size>& comparray, vector float* vs, vector float* fin_res) {
       vector signed int vec_C[4];
       vector float CA[4] = {0};
       vector float res[4] = {0};
       __builtin_mma_disassemble_acc(vec_C, ACC);
       for (int i = 0; i < 4; i++) {
          CA[i] = vec_splats((float)(((double)comparray[c_idx+i]) * -128.0));
          res[i] = vec_add(vec_ctf(vec_C[i], 0), CA[i]);
          fin_res[s_idx+i] = vec_madd(res[i], vs[s_idx+i], fin_res[s_idx+i]);
       }
    }
    /* This function processes quantized data from block_q4_0 elements.
     * First the we try to extract the two int4 values stored in single int8_t into two signed int8.
     * And then we subtract each of the resultant element with 8, to convert signed int8 to unsigned int8.
     * Also compute the rowsum which is required to compensate the above conversion. */
    inline void process_q4_elements(vector signed char (&c)[2], int* ca) {
        const vector signed char lowMask = vec_splats((signed char)0xF);
        const vector unsigned char v4 = vec_splats((unsigned char)0x4);
        const vector signed char v8 = vec_splats((signed char)0x8);
        vector signed int vsum = {0};
        vector signed int vsum2 = {0};
        c[0] = vec_and(c[1], lowMask);
        c[1] = vec_sr(c[1], v4);
        c[0] = vec_sub(c[0], v8);
        c[1] = vec_sub(c[1], v8);
        vsum = vec_sum4s(c[0], vsum);
        vsum2 = vec_sum4s(c[1], vsum2);
        vsum = vec_add(vsum, vsum2);
        *(ca) = vsum[0] + vsum[1] + vsum[2] + vsum[3];
    }

    template <typename V1, typename V2>
    inline void vector_permute_store(V2 &s1, V2 &s2, V2 &s3, V2 &s4, V1 *vecOffset, bool flip) {
        vector unsigned char swiz1 = {0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23};
        vector unsigned char swiz2 = {8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31};
        vector unsigned char swiz3 = {0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27};
        vector unsigned char swiz4 = {4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31};
        V2 t1, t2, t3, t4, t5, t6, t7, t8;
        vector unsigned char xor_vector;
        uint8_t flip_vec = 0x80;
        xor_vector = vec_splats(flip_vec);
        t1 = vec_perm(s1, s2, swiz1);
        t2 = vec_perm(s1, s2, swiz2);
        t3 = vec_perm(s3, s4, swiz1);
        t4 = vec_perm(s3, s4, swiz2);
        t5 = vec_perm(t1, t3, swiz3);
        t6 = vec_perm(t1, t3, swiz4);
        t7 = vec_perm(t2, t4, swiz3);
        t8 = vec_perm(t2, t4, swiz4);
        if (flip == true) {
            t5 = vec_xor(t5, xor_vector);
            t6 = vec_xor(t6, xor_vector);
            t7 = vec_xor(t7, xor_vector);
            t8 = vec_xor(t8, xor_vector);
        }
        vec_xst(t5, 0, vecOffset);
        vec_xst(t6, 0, vecOffset+16);
        vec_xst(t7, 0, vecOffset+32);
        vec_xst(t8, 0, vecOffset+48);
    }

    template<int size>
    void packNormalInt4(const TA* a, int64_t lda, int rows, int cols, int8_t* vec, std::array<int, size>& comparray) {
        int64_t i, j;
        TA *aoffset = NULL;
        int8_t *vecOffset = NULL;
        TA *aoffset1 = NULL, *aoffset2 = NULL, *aoffset3 = NULL, *aoffset4 = NULL;
        TA *aoffset5 = NULL, *aoffset6 = NULL, *aoffset7 = NULL, *aoffset8 = NULL;
        vector signed char c1[2] = {0}, c2[2] = {0}, c3[2] = {0}, c4[2] = {0};
        vector signed char c5[2] = {0}, c6[2] = {0}, c7[2] = {0}, c8[2] = {0};
        aoffset = const_cast<TA*>(a);
        vecOffset = vec;
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
                i = (cols >> 2);
                if (i > 0) {
                    do {
                        c1[1] = reinterpret_cast<vector signed char>(vec_xl(0, aoffset1->qs));
                        c2[1] = reinterpret_cast<vector signed char>(vec_xl(0, aoffset2->qs));
                        c3[1] = reinterpret_cast<vector signed char>(vec_xl(0, aoffset3->qs));
                        c4[1] = reinterpret_cast<vector signed char>(vec_xl(0, aoffset4->qs));
                        c5[1] = reinterpret_cast<vector signed char>(vec_xl(0, aoffset5->qs));
                        c6[1] = reinterpret_cast<vector signed char>(vec_xl(0, aoffset6->qs));
                        c7[1] = reinterpret_cast<vector signed char>(vec_xl(0, aoffset7->qs));
                        c8[1] = reinterpret_cast<vector signed char>(vec_xl(0, aoffset8->qs));

                        process_q4_elements(c1, &comparray[0]);
                        process_q4_elements(c2, &comparray[1]);
                        process_q4_elements(c3, &comparray[2]);
                        process_q4_elements(c4, &comparray[3]);
                        process_q4_elements(c5, &comparray[4]);
                        process_q4_elements(c6, &comparray[5]);
                        process_q4_elements(c7, &comparray[6]);
                        process_q4_elements(c8, &comparray[7]);
                        vector_permute_store<int8_t, vector signed char>(c1[0], c2[0], c3[0], c4[0], vecOffset, false);
                        vector_permute_store<int8_t, vector signed char>(c1[1], c2[1], c3[1], c4[1], vecOffset+64, false);
                        vector_permute_store<int8_t, vector signed char>(c5[0], c6[0], c7[0], c8[0], vecOffset+128, false);
                        vector_permute_store<int8_t, vector signed char>(c5[1], c6[1], c7[1], c8[1], vecOffset+192, false);
                        aoffset1 += lda;
                        aoffset2 += lda;
                        aoffset3 += lda;
                        aoffset4 += lda;
                        aoffset5 += lda;
                        aoffset6 += lda;
                        aoffset7 += lda;
                        aoffset8 += lda;
                        vecOffset += 256;
                        i--;
                    } while (i > 0);
                }
                j--;
            } while (j > 0);
        }

        if (rows & 4) {
            aoffset1 = aoffset;
            aoffset2 = aoffset1 + lda;
            aoffset3 = aoffset2 + lda;
            aoffset4 = aoffset3 + lda;
            aoffset += 4 * lda;
            i = (cols >> 2);
            if (i > 0) {
                do {
                    c1[1] = reinterpret_cast<vector signed char>(vec_xl(0, aoffset1->qs));
                    c2[1] = reinterpret_cast<vector signed char>(vec_xl(0, aoffset2->qs));
                    c3[1] = reinterpret_cast<vector signed char>(vec_xl(0, aoffset3->qs));
                    c4[1] = reinterpret_cast<vector signed char>(vec_xl(0, aoffset4->qs));

                    process_q4_elements(c1, &comparray[0]);
                    process_q4_elements(c2, &comparray[1]);
                    process_q4_elements(c3, &comparray[2]);
                    process_q4_elements(c4, &comparray[3]);
                    vector_permute_store<int8_t, vector signed char>(c1[0], c2[0], c3[0], c4[0], vecOffset, false);
                    vector_permute_store<int8_t, vector signed char>(c1[1], c2[1], c3[1], c4[1], vecOffset+64, false);
                    aoffset1 += lda;
                    aoffset2 += lda;
                    aoffset3 += lda;
                    aoffset4 += lda;
                    vecOffset += 128;
                    i--;
                } while (i > 0);
            }
        }

        if (rows & 3) {
            aoffset1 = aoffset;
            aoffset2 = aoffset1 + lda;
            aoffset3 = aoffset2 + lda;
            i = (cols >> 2);
            if (i > 0) {
                do {
                    switch(rows) {
                        case 3: c3[1] = reinterpret_cast<vector signed char>(vec_xl(0, aoffset3->qs));
                        case 2: c2[1] = reinterpret_cast<vector signed char>(vec_xl(0, aoffset2->qs));
                        case 1: c1[1] = reinterpret_cast<vector signed char>(vec_xl(0, aoffset1->qs));
                            break;
                    }
                    process_q4_elements(c1, &comparray[0]);
                    process_q4_elements(c2, &comparray[1]);
                    process_q4_elements(c3, &comparray[2]);
                    process_q4_elements(c4, &comparray[3]);
                    vector_permute_store<int8_t, vector signed char>(c1[0], c2[0], c3[0], c4[0], vecOffset, false);
                    vector_permute_store<int8_t, vector signed char>(c1[1], c2[1], c3[1], c4[1], vecOffset+64, false);
                    aoffset1 += lda;
                    aoffset2 += lda;
                    aoffset3 += lda;
                    vecOffset += 128;
                    i--;
                } while(i > 0);
            }
        }
    }
    template<typename VA, typename VB>
    void packNormal(const block_q8_0* a, int64_t lda, int rows, int cols, VA* vec, bool flip) {
        int64_t i, j;
        block_q8_0 *aoffset = NULL;
        VA *vecOffset = NULL;
        block_q8_0* aoffsets[8];
        __vector_pair arr[8];
        VB c[8][2] = {0};
        VB c1[8] = {0}; VB c2[8] = {0};
        aoffset = const_cast<block_q8_0*>(a);
        vecOffset = vec;
        j = (rows >> 3);
        if (j > 0) {
            do {
                aoffsets[0] = aoffset;
                for (int it = 1; it < 8; it++)
                    aoffsets[it] = aoffsets[it-1] + lda;
                aoffset += 8 * lda;

                i = (cols >> 3);
                if (i > 0) {
                do {
                    for (int it = 0; it < 8; it++) {
                        arr[it] = __builtin_vsx_lxvp(0, (__vector_pair*)aoffsets[it]->qs);
                        __builtin_vsx_disassemble_pair(c[it], &arr[it]);
                        c1[it] = c[it][0];
                        c2[it] = c[it][1];
                    }
                    vector_permute_store<VA, VB>(c1[0], c1[1], c1[2], c1[3], vecOffset, flip);
                    vector_permute_store<VA, VB>(c2[0], c2[1], c2[2], c2[3], vecOffset+64, flip);
                    vector_permute_store<VA, VB>(c1[4], c1[5], c1[6], c1[7], vecOffset+128, flip);
                    vector_permute_store<VA, VB>(c2[4], c2[5], c2[6], c2[7], vecOffset+192, flip);
                    for (int it = 0; it < 8; it++)
                        aoffsets[it] += lda;
                    vecOffset += 256;
                    i--;
               } while(i > 0);
            }
            j--;
        } while(j > 0);
    }

    if (rows & 4) {
            aoffsets[0]  = aoffset;
            for (int it = 1; it < 4; it++ )
                aoffsets[it] = aoffsets[it-1] + lda;
            aoffset += 4 * lda;
        i = (cols >> 3);
            if (i > 0) {
               do {
                    for (int it = 0; it < 4; it++) {
                        arr[it] = __builtin_vsx_lxvp(0, (__vector_pair*)aoffsets[it]->qs);
                        __builtin_vsx_disassemble_pair(c[it], &arr[it]);
                        c1[it] = c[it][0];
                        c2[it] = c[it][1];
                    }
                    vector_permute_store<VA, VB>(c1[0], c1[1], c1[2], c1[3], vecOffset, flip);
                    vector_permute_store<VA, VB>(c2[0], c2[1], c2[2], c2[3], vecOffset+64, flip);
                    for (int it = 0; it < 4; it++) {
                        aoffsets[it] += lda;
                    }
                    vecOffset += 128;
                    i--;
               } while(i > 0);
            }
        }

        if (rows & 3) {
            aoffsets[0]  = aoffset;
            for (int it = 1; it < 3; it++ )
                aoffsets[it] = aoffsets[it-1] + lda;
            i = (cols >> 3);
            if (i > 0) {
                do {
                    switch(rows) {
                        case 3: arr[2] = __builtin_vsx_lxvp(0, (__vector_pair*)aoffsets[2]->qs);
                                __builtin_vsx_disassemble_pair(c[2], &arr[2]);
                                c1[2] = c[2][0]; c2[2] = c[2][1];
                        case 2: arr[1] = __builtin_vsx_lxvp(0, (__vector_pair*)aoffsets[1]->qs);
                                __builtin_vsx_disassemble_pair(c[1], &arr[1]);
                                c1[1] = c[1][0]; c2[1] = c[1][1];
                        case 1: arr[0] = __builtin_vsx_lxvp(0, (__vector_pair*)aoffsets[0]->qs);
                                __builtin_vsx_disassemble_pair(c[0], &arr[0]);
                                c1[0] = c[0][0]; c2[0] = c[0][1];
                                break;
                    }
                    vector_permute_store<VA, VB>(c1[0], c1[1], c1[2], c1[3], vecOffset, flip);
                    vector_permute_store<VA, VB>(c2[0], c2[1], c2[2], c2[3], vecOffset+64, flip);
                    for (int it = 0; it < 3; it++)
                         aoffsets[it] += lda;
                    vecOffset += 128;
                    i--;
               } while(i > 0);
            }
        }
    }

    void mnpack(int64_t m0, int64_t m, int64_t n0, int64_t n) {
        int m_rem = MIN(m - m0, 16);
        int n_rem = MIN(n - n0, 16);

        int mc = 0, nc = 0;

        if (m_rem >= 8 && n_rem >= 8) {
           mc = 8;
           nc = 8;
           gemm<8, 8>(m0, m, n0, n);
        } else if (m_rem >= 4 && n_rem >= 8) {
            mc = 4;
            nc = 8;
            gemm<4, 8>(m0, m, n0, n);
        } else if (m_rem >= 8 && n_rem >= 4) {
            mc = 8;
            nc = 4;
            gemm<8, 4>(m0, m, n0, n);
        } else if (m_rem >= 4 && n_rem >= 4) {
            mc = 4;
            nc = 4;
            gemm_small(m0, m, n0, n, mc, nc);
        } else {
            mc = (m_rem >= 4) ? 4 : m_rem;
            nc = (n_rem >= 4) ? 4 : n_rem;
            if (mc == 0 || nc == 0)
               return;
            gemm_small(m0, m, n0, n, mc, nc);
        }

        int64_t mp = m0 + ((m - m0) / mc) * mc;
        int64_t np = n0 + ((n - n0) / nc) * nc;
        mnpack(mp, m, n0, np);
        mnpack(m0, m, np, n);
    }


    void KERNEL_4x8(int64_t ii, int64_t jj) {
        vec_t vec_A[8], vec_B[16] = {0};
        acc_t acc_0, acc_1;
        std::array<int, 4> comparray {};
        vector float fin_res[8] = {0};
        vector float vs[8] = {0};
        bool isAblock_q4 = std::is_same_v<TA, block_q4_0>;
        for (int l = 0; l < k; l++) {
            __builtin_mma_xxsetaccz(&acc_0);
            __builtin_mma_xxsetaccz(&acc_1);
            if (std::is_same_v<TA, block_q4_0>) {
               packNormalInt4<4>((A+(ii*lda)+l), lda, 4, 4, (int8_t*)vec_A, comparray);
            } else {
               packNormal<int8_t, vector signed char>((const block_q8_0*)(A+(ii*lda)+l), lda, 4, 8, (int8_t*)vec_A, false);
            }
            packNormal<uint8_t, vector unsigned char>((B+(jj*ldb)+l), ldb, 8, 8, (uint8_t*)vec_B, true);
            for(int x = 0; x < 8; x++) {
                __builtin_mma_xvi8ger4pp(&acc_0, vec_A[x], vec_B[x]);
                __builtin_mma_xvi8ger4pp(&acc_1, vec_A[x], vec_B[x+8]);
            }
            for (int I = 0; I<4; I++) {
                for (int J = 0; J<4; J++) {
                    *((float*)&vs[I]+J) = (unhalf((A+((ii+I)*lda)+l)->d) * unhalf((B+((jj+J)*ldb)+l)->d));
                    *((float*)&vs[I+4]+J) = (unhalf((A+((ii+I)*lda)+l)->d) * unhalf((B+((jj+J+4)*ldb)+l)->d));
                }
            }
            if (!isAblock_q4) {
                auto aoffset = A+(ii*lda)+l;
                for (int i = 0; i < 4; i++) {
                    comparray[i] = 0;
                    int ca = 0;
                    auto *at = aoffset->qs;
                    for (int j = 0; j < 32; j++)
                        ca += (int)*at++;
                    comparray[i] = ca;
                    aoffset += lda;
                }
            }
            compute<4>(&acc_0, 0, 0, comparray, vs, fin_res);
            compute<4>(&acc_1, 0, 4, comparray, vs, fin_res);
        }
        save_res(ii, jj, 0, fin_res);
        save_res(ii, jj+4, 4, fin_res);
    }

    void KERNEL_8x4(int64_t ii, int64_t jj) {
        vec_t vec_A[16], vec_B[8] = {0};
        acc_t acc_0, acc_1;
        std::array<int, 8> comparray {};
        vector float fin_res[8] = {0};
        vector float vs[8] = {0};
        bool isAblock_q4 = std::is_same_v<TA, block_q4_0>;
        for (int l = 0; l < k; l++) {
            __builtin_mma_xxsetaccz(&acc_0);
            __builtin_mma_xxsetaccz(&acc_1);
            if (std::is_same_v<TA, block_q4_0>) {
               packNormalInt4<8>((A+(ii*lda)+l), lda, 8, 4, (int8_t*)vec_A, comparray);
            } else {
               packNormal<int8_t, vector signed char>((const block_q8_0*)(A+(ii*lda)+l), lda, 8, 8, (int8_t*)vec_A, false);
            }
            packNormal<uint8_t, vector unsigned char>((B+(jj*ldb)+l), ldb, 4, 8, (uint8_t*)vec_B, true);
            for(int x = 0; x < 8; x++) {
                __builtin_mma_xvi8ger4pp(&acc_0, vec_A[x], vec_B[x]);
                __builtin_mma_xvi8ger4pp(&acc_1, vec_A[x+8], vec_B[x]);
            }
            for (int I = 0; I<8; I++) {
                for (int J = 0; J<4; J++) {
                    *((float*)&vs[I]+J) = (unhalf((A+((ii+I)*lda)+l)->d) * unhalf((B+((jj+J)*ldb)+l)->d));
                }
            }
            if (!isAblock_q4) {
                auto aoffset = A+(ii*lda)+l;
                for (int i = 0; i < 8; i++) {
                    comparray[i] = 0;
                    int ca = 0;
                    auto *at = aoffset->qs;
                    for (int j = 0; j < 32; j++)
                        ca += (int)*at++;
                    comparray[i] = ca;
                    aoffset += lda;
                }
            }
            compute<8>(&acc_0, 0, 0, comparray, vs, fin_res);
            compute<8>(&acc_1, 4, 4, comparray, vs, fin_res);
        }
        save_res(ii, jj, 0, fin_res);
        save_res(ii+4, jj, 4, fin_res);
    }

    void KERNEL_8x8(int64_t ii, int64_t jj) {
        vec_t vec_A[16], vec_B[16] = {0};
        acc_t acc_0, acc_1, acc_2, acc_3;
        std::array<int, 8> comparray {};
        vector float fin_res[16] = {0};
        vector float vs[16] = {0};
        bool isAblock_q4 = std::is_same_v<TA, block_q4_0>;
        for (int l = 0; l < k; l++) {
            __builtin_mma_xxsetaccz(&acc_0);
            __builtin_mma_xxsetaccz(&acc_1);
            __builtin_mma_xxsetaccz(&acc_2);
            __builtin_mma_xxsetaccz(&acc_3);
            if (std::is_same_v<TA, block_q4_0>) {
               packNormalInt4<8>((A+(ii*lda)+l), lda, 8, 4, (int8_t*)vec_A, comparray);
            } else {
               packNormal<int8_t, vector signed char>((const block_q8_0*)(A+(ii*lda)+l), lda, 8, 8, (int8_t*)vec_A, false);
            }
            packNormal<uint8_t, vector unsigned char>((B+(jj*ldb)+l), ldb, 8, 8, (uint8_t*)vec_B, true);
            for(int x = 0; x < 8; x++) {
                __builtin_mma_xvi8ger4pp(&acc_0, vec_A[x], vec_B[x]);
                __builtin_mma_xvi8ger4pp(&acc_1, vec_A[x+8], vec_B[x]);
                __builtin_mma_xvi8ger4pp(&acc_2, vec_A[x], vec_B[x+8]);
                __builtin_mma_xvi8ger4pp(&acc_3, vec_A[x+8], vec_B[x+8]);
            }
            for (int I = 0; I<8; I++) {
                for (int J = 0; J<4; J++) {
                    *((float*)&vs[I]+J) = (unhalf((A+((ii+I)*lda)+l)->d) * unhalf((B+((jj+J)*ldb)+l)->d));
                    *((float*)&vs[I+8]+J) = (unhalf((A+((ii+I)*lda)+l)->d) * unhalf((B+((jj+J+4)*ldb)+l)->d));
                }
            }
            if (!isAblock_q4) {
                auto aoffset = A+(ii*lda)+l;
                for (int i = 0; i < 8; i++) {
                    comparray[i] = 0;
                    int ca = 0;
                    auto *at = aoffset->qs;
                    for (int j = 0; j < 32; j++)
                        ca += (int)*at++;
                    comparray[i] = ca;
                    aoffset += lda;
                }
            }
            compute<8>(&acc_0, 0, 0, comparray, vs, fin_res);
            compute<8>(&acc_1, 4, 4, comparray, vs, fin_res);
            compute<8>(&acc_2, 0, 8, comparray, vs, fin_res);
            compute<8>(&acc_3, 4, 12, comparray, vs, fin_res);
        }
        save_res(ii, jj, 0, fin_res);
        save_res(ii+4, jj, 4, fin_res);
        save_res(ii, jj+4, 8, fin_res);
        save_res(ii+4, jj+4, 12, fin_res);
    }

    void gemm_small(int64_t m0, int64_t m, int64_t n0, int64_t n, int RM, int RN) {
        int64_t ytiles = (m - m0) / RM;
        int64_t xtiles = (n - n0) / RN;
        int64_t tiles = xtiles * ytiles;
        int64_t duty = (tiles + nth - 1) / nth;
        int64_t start = duty * ith;
        int64_t end = start + duty;
        vec_t vec_A[8] = {0}, vec_B[8] = {0};
        vector signed int vec_C[4];
        acc_t acc_0;
        bool isAblock_q4 = std::is_same_v<TA, block_q4_0>;

        if (end > tiles)
            end = tiles;
        for (int64_t job = start; job < end; ++job) {
            int64_t ii = m0 + job / xtiles * RM;
            int64_t jj = n0 + job % xtiles * RN;
            std::array<int, 4> comparray{};
            vector float res[4] = {0};
            vector float fin_res[4] = {0};
            vector float vs[4] = {0};
            vector float CA[4] = {0};
            __builtin_prefetch((A+(ii*lda)+0)->qs, 0, 1); // prefetch first value
            __builtin_prefetch((B+(jj*ldb)+0)->qs, 0, 1); // prefetch first value
            for (int l = 0; l < k; l++) {
                __builtin_prefetch((A+(ii*lda)+(l+1))->qs, 0, 1); // prefetch one loop ahead
                __builtin_prefetch((B+(jj*ldb)+(l+1))->qs, 0, 1); // prefetch one loop ahead
                __builtin_mma_xxsetaccz(&acc_0);
                if (isAblock_q4) {
                   packNormalInt4<4>((A+(ii*lda)+l), lda, RM, 4, (int8_t*)vec_A, comparray);
                } else {
                   packNormal<int8_t, vector signed char>((const block_q8_0*)(A+(ii*lda)+l), lda, RM, 8, (int8_t*)vec_A, false);
                }
                packNormal<uint8_t, vector unsigned char>((B+(jj*ldb)+l), ldb, RN, 8, (uint8_t*)vec_B, true);
                for(int x = 0; x < 8; x+=4) {
                    __builtin_mma_xvi8ger4pp(&acc_0, vec_A[x], vec_B[x]);
                    __builtin_mma_xvi8ger4pp(&acc_0, vec_A[x+1], vec_B[x+1]);
                    __builtin_mma_xvi8ger4pp(&acc_0, vec_A[x+2], vec_B[x+2]);
                    __builtin_mma_xvi8ger4pp(&acc_0, vec_A[x+3], vec_B[x+3]);
                }
                for (int I = 0; I<RM; I++) {
                    for (int J = 0; J<RN; J++) {
                        *((float*)&vs[I]+J) = (unhalf((A+((ii+I)*lda)+l)->d) * unhalf((B+((jj+J)*ldb)+l)->d));
                    }
                }
                __builtin_mma_disassemble_acc(vec_C, &acc_0);
                if (!isAblock_q4) {
                    auto aoffset = A+(ii*lda)+l;
                    for (int i = 0; i < RM; i++) {
                        comparray[i] = 0;
                        int ca = 0;
                        auto *at = aoffset->qs;
                        for (int j = 0; j < 32; j++)
                            ca += (int)*at++;
                        comparray[i] = ca;
                        aoffset += lda;
                    }
                }
                for (int i = 0; i < RM; i++) {
                    CA[i] = vec_splats((float)(((double)comparray[i]) * -128.0));
                    res[i] = vec_add(vec_ctf(vec_C[i], 0), CA[i]);
                    fin_res[i] = vec_madd(res[i], vs[i], fin_res[i]);
                }
            }
            save_res(ii, jj, 0, fin_res, RM, RN);
        }
    }

    template<int RM, int RN>
    inline void kernel(int64_t ii, int64_t jj) {
       if constexpr(RM == 4 && RN == 8) {
          KERNEL_4x8(ii,jj);
       } else if constexpr(RM == 8 && RN == 4) {
          KERNEL_8x4(ii,jj);
       } else if constexpr(RM == 8 && RN == 8) {
          KERNEL_8x8(ii,jj);
       } else {
          assert(false && "RN/RM values not supported");
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
        if (end > tiles)
            end = tiles;
        for (int64_t job = start; job < end; ++job) {
            int64_t ii = m0 + job / xtiles * RM;
            int64_t jj = n0 + job % xtiles * RN;
            kernel<RM, RN>(ii, jj);
        }
    }

    const TA *const A;
    const block_q8_0 *const B;
    float *C;
    const int64_t k;
    const int64_t lda;
    const int64_t ldb;
    const int64_t ldc;
    const int ith;
    const int nth;
};

class tinyBLAS_PPC {
  public:
    tinyBLAS_PPC(int64_t k,
                const float *A, int64_t lda,
                const float *B, int64_t ldb,
                float *C, int64_t ldc,
                int ith, int nth)
        : A(A), B(B), C(C), k(k), lda(lda), ldb(ldb), ldc(ldc), ith(ith), nth(nth) {
    }

    void matmul(int64_t m, int64_t n) {
       mnpack(0, m, 0, n);
    }

  private:

    void (tinyBLAS_PPC::*kernel)(int64_t, int64_t);

    inline void vector_permute_store_4(vector float *src, float *vecOffset) {
       vector float t1, t2, t3, t4, t5, t6, t7, t8;
           t1 = vec_mergeh(src[0], src[1]);
           t2 = vec_mergeh(src[2], src[3]);
           t3 = vec_mergel(src[0], src[1]);
           t4 = vec_mergel(src[2], src[3]);

           t5 = vec_xxpermdi(t1, t2, 0);
           t6 = vec_xxpermdi(t1, t2, 3);
           t7 = vec_xxpermdi(t3, t4, 0);
           t8 = vec_xxpermdi(t3, t4, 3);

           vec_xst(t5, 0, vecOffset);
           vec_xst(t6, 0, vecOffset + 4);
           vec_xst(t7, 0, vecOffset + 8);
           vec_xst(t8, 0, vecOffset + 12);
       }

    inline void vector_permute_store_8(vector float *src, float *vecOffset) {
       vector float t1, t2, t3, t4, t5, t6, t7, t8;
          t1 = vec_mergeh(src[0], src[1]);
          t2 = vec_mergeh(src[2], src[3]);
          t3 = vec_mergeh(src[4], src[5]);
          t4 = vec_mergeh(src[6], src[7]);

          t5 = vec_xxpermdi(t1, t2, 0);
          t6 = vec_xxpermdi(t3, t4, 0);
          t7 = vec_xxpermdi(t1, t2, 3);
          t8 = vec_xxpermdi(t3, t4, 3);

          vec_xst(t5, 0, vecOffset);
          vec_xst(t6, 0, vecOffset + 4);
          vec_xst(t7, 0, vecOffset + 8);
          vec_xst(t8, 0, vecOffset + 12);

          t1 = vec_mergel(src[0], src[1]);
          t2 = vec_mergel(src[2], src[3]);
          t3 = vec_mergel(src[4], src[5]);
          t4 = vec_mergel(src[6], src[7]);

          t5 = vec_xxpermdi(t1, t2, 0);
          t6 = vec_xxpermdi(t3, t4, 0);
          t7 = vec_xxpermdi(t1, t2, 3);
          t8 = vec_xxpermdi(t3, t4, 3);

          vec_xst(t5, 0, vecOffset + 16);
          vec_xst(t6, 0, vecOffset + 20);
          vec_xst(t7, 0, vecOffset + 24);
          vec_xst(t8, 0, vecOffset + 28);
    }

 void packTranspose(const float* a, int64_t lda, int rows, int cols, float* vec) {
        int64_t i, j;
        float * aoffsets[8];
        float *aoffset = NULL, *boffset = NULL;
        __vector_pair arr[8];
        vector float c[8][2] = {0};
        vector float c1[8] = {0};
        vector float c2[8] = {0};
        aoffset = const_cast<float*>(a);
        boffset = vec;
        j = (rows >> 3);
        if (j > 0) {

            do {
                aoffsets[0] = aoffset;
                for (int it = 1; it< 8; it++)
                    aoffsets[it] = aoffsets[it-1] + lda;
                aoffset += 8 * lda;
                i = (cols >> 3);
                if (i > 0) {
                    do {
                        for (int it = 0; it< 8; it++) {
                            arr[it] = __builtin_vsx_lxvp(0, (__vector_pair*)aoffsets[it]);
                            __builtin_vsx_disassemble_pair(c[it], &arr[it]);
                            c1[it] = c[it][0];
                            c2[it] = c[it][1];
                        }

                        vector_permute_store_8(c1, boffset);
                        vector_permute_store_8(c2, boffset+32);
                        for (int it = 0; it < 4; it++)
                            aoffsets[it] = aoffsets[it] + 8*lda;
                        boffset += 64;
                        i--;
                    } while(i > 0);
                }
                if (cols & 4) {
                    for (int it = 0; it < 8 ; it++)
                        c1[it] = vec_xl(0, aoffsets[it]);
                    vector_permute_store_8(c1, boffset);
                }
            j--;
            } while(j > 0);
        }

        if (rows & 4) {
            aoffsets[0] = aoffset;
            for (int it = 1; it < 4; it++)
                aoffsets[it] = aoffsets[it-1] + lda;
            aoffset += 4 * lda;
            i = (cols >> 3);
            if (i > 0) {
                do {
                    for (int it = 0; it < 4; it++) {
                        arr[it] = __builtin_vsx_lxvp(0, (__vector_pair*)aoffsets[it]);
                        __builtin_vsx_disassemble_pair(c[it], &arr[it]);
                        c1[it] = c[it][0];
                        c2[it] = c[it][1];
                    }
                    vector_permute_store_4(c1, boffset);
                    vector_permute_store_4(c2, boffset+16);
                    for (int it = 0; it < 4; it++)
                        aoffsets[it] += 8*lda;
                    boffset += 32;
                    i--;
                } while(i > 0);
            }

            if (cols & 4) {
               for (int it = 0; it < 4; it++)
                   c1[it] = vec_xl(0, aoffsets[it]);
                vector_permute_store_4(c1, boffset);
            }
        }
        if (rows & 3) {
            aoffsets[0] = aoffset;
            for (int it = 1; it < 3; it++)
                aoffsets[it] = aoffsets[it-1] + lda;
            if (cols & 4) {
                for (int it = 0; it < 3; it++)
                    c1[it] = vec_xl(0, aoffsets[it]);
                vector_permute_store_4(c1, boffset);
            }
        }
    }

    void KERNEL_4x4(int64_t ii, int64_t jj) {
        vec_t vec_A[4], vec_B[4], vec_C[4];
        acc_t acc_0;
        __builtin_mma_xxsetaccz(&acc_0);
        for (int l = 0; l < k; l+=4) {
            packTranspose(A+(ii*lda)+l, lda, 4, 4, (float*)vec_A);
            packTranspose(B+(jj*ldb)+l, ldb, 4, 4, (float*)vec_B);
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
            packTranspose(A+(ii*lda)+l, lda, 4, 4, (float*)vec_A);
            packTranspose(B+(jj*ldb)+l, ldb, 8, 4, (float*)vec_B);
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
            packTranspose(A+(ii*lda)+l, lda, 8, 4, (float*)vec_A);
            packTranspose(B+(jj*ldb)+l, ldb, 4, 4, (float*)vec_B);
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
            packTranspose(A+(ii*lda)+l, lda, 8, 8, (float*)vec_A);
            packTranspose(B+(jj*ldb)+l, ldb, 8, 8, (float*)vec_B);
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
        int m_rem = MIN(m - m0, 8);
        int n_rem = MIN(n - n0, 8);
        int mc = 0, nc = 0;
        if (m_rem >= 8 && n_rem >= 8) {
           mc = 8;
           nc = 8;
           gemm<8, 8>(m0, m, n0, n);
        } else if (m_rem >= 4 && n_rem >= 8) {
                mc = 4;
                nc = 8;
                gemm<4, 8>(m0, m, n0, n);
        } else if (m_rem >= 8 && n_rem >= 4) {
                mc = 8;
                nc = 4;
                gemm<8, 4>(m0, m, n0, n);
        } else if (m_rem >= 4 && n_rem >= 4) {
                mc = 4;
                nc = 4;
                gemm<4, 4>(m0, m, n0, n);
        } else {
            mc = (m_rem >= 4) ? 4 : m_rem;
            nc = (n_rem >= 4) ? 4 : n_rem;
            if (mc == 0 || nc == 0)
               return;
            gemm_small(m0, m, n0, n, mc, nc);
        }
        int64_t mp = m0 + ((m - m0) / mc) * mc;
        int64_t np = n0 + ((n - n0) / nc) * nc;
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
            vec_t vec_A[4] {0}, vec_B[4] = {0};
            for (int l=0; l<k; l+=4) {
                /* 'GEMV Forwarding' concept is used in first two conditional loops.
                 * when one of the matrix has a single row/column, the elements are
                 * broadcasted, instead of using packing routine to prepack the
                 * matrix elements.
                 */
                if (RM == 1) {
                    float* a = const_cast<float*>(A+(ii)*lda+l);
                    packTranspose(B+(jj*ldb)+l, ldb, RN, 4, (float*)vec_B);
                    vec_A[0] = (vec_t)vec_xl(0,a);
                    vec_A[1] = (vec_t)vec_splats(*((float*)&vec_A+1));
                    vec_A[2] = (vec_t)vec_splats(*((float*)&vec_A+2));
                    vec_A[3] = (vec_t)vec_splats(*((float*)&vec_A+3));
                } else if (RN == 1) {
                    packTranspose(A+(ii*lda)+l, lda, RM, 4, (float*)vec_A);
                    float* b = const_cast<float*>(B+(jj)*ldb+l);
                    vec_B[0] = (vec_t)vec_xl(0,b);
                    vec_B[1] = (vec_t)vec_splats(*((float*)&vec_B+1));
                    vec_B[2] = (vec_t)vec_splats(*((float*)&vec_B+2));
                    vec_B[3] = (vec_t)vec_splats(*((float*)&vec_B+3));
                } else {
                    packTranspose(A+(ii*lda)+l, lda, RM, 4, (float*)vec_A);
                    packTranspose(B+(jj*ldb)+l, ldb, RN, 4, (float*)vec_B);
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

    const float *const A;
    const float *const B;
    float *C;
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
bool llamafile_sgemm(const struct ggml_compute_params * params, int64_t m, int64_t n, int64_t k,
                     const void *A, int64_t lda, const void *B, int64_t ldb, void *C,
                     int64_t ldc, int Atype, int Btype, int Ctype) {

    assert(m >= 0);
    assert(n >= 0);
    assert(k >= 0);
    assert(lda >= k);
    assert(ldb >= k);
    assert(ldc >= m);
    assert(params->nth > 0);
    assert(params->ith < params->nth);

    // only enable sgemm for prompt processing
#if !defined(__MMA__)
    if (n < 2)
        return false;
#endif

    if (Ctype != GGML_TYPE_F32)
        return false;

    switch (Atype) {

    case GGML_TYPE_F32: {
        if (Btype != GGML_TYPE_F32)
            return false;
#if defined(__AVX512F__)
        tinyBLAS<16, __m512, __m512, float, float, float> tb{ params,
            k, (const float *)A, lda,
            (const float *)B, ldb,
            (float *)C, ldc};
        return tb.matmul(m, n);
#elif defined(__AVX__) || defined(__AVX2__)
        tinyBLAS<8, __m256, __m256, float, float, float> tb{ params,
            k, (const float *)A, lda,
            (const float *)B, ldb,
            (float *)C, ldc};
        return tb.matmul(m, n);
#elif defined(__ARM_NEON)
        if (n < 4)
            return false;
        tinyBLAS<4, float32x4_t, float32x4_t, float, float, float> tb{ params,
            k, (const float *)A, lda,
            (const float *)B, ldb,
            (float *)C, ldc};
        return tb.matmul(m, n);
#elif defined(__VXE__) || defined(__VXE2__)
        if (n < 4)
            return false;
        tinyBLAS<4, float32x4_t, float32x4_t, float, float, float> tb{ params,
            k, (const float *)A, lda,
            (const float *)B, ldb,
            (float *)C, ldc};
        return tb.matmul(m, n);
#elif defined(__MMA__)
        if (k % 8)
            return false;
        tinyBLAS_PPC tb{
            k, (const float *)A, lda,
            (const float *)B, ldb,
            (float *)C, ldc,
            params->ith, params->nth};
        tb.matmul(m, n);
        return true;
#else
        return false;
#endif
    }

    case GGML_TYPE_BF16: {
#if defined(__AVX512BF16__)
        if (Btype == GGML_TYPE_BF16) {
            tinyBLAS<32, __m512, __m512bh, ggml_bf16_t, ggml_bf16_t, float> tb{ params, k,
                (const ggml_bf16_t *)A, lda,
                (const ggml_bf16_t *)B, ldb,
                (float *)C, ldc};
            return tb.matmul(m, n);
        }
#elif defined(__AVX512F__)
        if (Btype == GGML_TYPE_BF16) {
            tinyBLAS<16, __m512, __m512, ggml_bf16_t, ggml_bf16_t, float> tb{ params, k,
                (const ggml_bf16_t *)A, lda,
                (const ggml_bf16_t *)B, ldb,
                (float *)C, ldc};
            return tb.matmul(m, n);
        }
#elif defined(__AVX2__)
        if (Btype == GGML_TYPE_BF16) {
            tinyBLAS<8, __m256, __m256, ggml_bf16_t, ggml_bf16_t, float> tb{ params, k,
                (const ggml_bf16_t *)A, lda,
                (const ggml_bf16_t *)B, ldb,
                (float *)C, ldc};
            return tb.matmul(m, n);
        }
#elif defined(__MMA__)
        if ((k % 8))
                return false;
        if(Btype == GGML_TYPE_BF16) {
           tinyBLAS_BF16_PPC<ggml_bf16_t, ggml_bf16_t, float> tb{ k,
            (const ggml_bf16_t *)A, lda,
            (const ggml_bf16_t *)B, ldb,
            (float *)C, ldc,
            params->ith, params->nth};
        tb.matmul(m, n);
        return true;
        }
#endif
        return false;
    }

    case GGML_TYPE_F16: {
#if defined(__AVX512F__)
        if (Btype == GGML_TYPE_F16) {
            tinyBLAS<16, __m512, __m512, ggml_fp16_t, ggml_fp16_t, float> tb{ params, k,
                (const ggml_fp16_t *)A, lda,
                (const ggml_fp16_t *)B, ldb,
                (float *)C, ldc};
            return tb.matmul(m, n);
        }
#elif (defined(__AVX__) || defined(__AVX2__)) && defined(__F16C__)
        if (Btype == GGML_TYPE_F16) {
            tinyBLAS<8, __m256, __m256, ggml_fp16_t, ggml_fp16_t, float> tb{ params, k,
                (const ggml_fp16_t *)A, lda,
                (const ggml_fp16_t *)B, ldb,
                (float *)C, ldc};
            return tb.matmul(m, n);
        }
#elif defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) && !defined(_MSC_VER)
        if (n < 8)
            return false;
        if (Btype == GGML_TYPE_F16) {
            tinyBLAS<8, float16x8_t, float16x8_t, ggml_fp16_t, ggml_fp16_t, float> tb{ params,
                k, (const ggml_fp16_t *)A, lda,
                (const ggml_fp16_t *)B, ldb,
                (float *)C, ldc};
            return tb.matmul(m, n);
        }
#elif defined(__ARM_NEON) && !defined(_MSC_VER)
        if (Btype == GGML_TYPE_F32) {
            tinyBLAS<4, float32x4_t, float32x4_t, ggml_fp16_t, float, float> tb{ params,
                k, (const ggml_fp16_t *)A, lda,
                (const float *)B, ldb,
                (float *)C, ldc};
            return tb.matmul(m, n);
        }
#elif defined(__VXE__) || defined(__VXE2__)
        if (n < 4)
            return false;
        if (Btype == GGML_TYPE_F16) {
            tinyBLAS<4, float32x4_t, float32x4_t, ggml_fp16_t, ggml_fp16_t, float> tb{ params,
                k, (const ggml_fp16_t *)A, lda,
                (const ggml_fp16_t *)B, ldb,
                (float *)C, ldc};
            return tb.matmul(m, n);
        }
#endif
        return false;
    }

    case GGML_TYPE_Q8_0: {
        if (Btype != GGML_TYPE_Q8_0)
           return false;
#if defined(__AVX2__) || defined(__AVX512F__) || defined(__AVX__)
        tinyBLAS_Q0_AVX<block_q8_0, block_q8_0, float> tb{
            k, (const block_q8_0 *)A, lda,
            (const block_q8_0 *)B, ldb,
            (float *)C, ldc,
            params->ith, params->nth};
        tb.matmul(m, n);
        return true;
#elif defined(__ARM_FEATURE_DOTPROD)
        tinyBLAS_Q0_ARM<block_q8_0> tb{
            k, (const block_q8_0 *)A, lda,
            (const block_q8_0 *)B, ldb,
            (float *)C, ldc,
            params->ith, params->nth};
        tb.matmul(m, n);
        return true;
#elif defined(__MMA__)
    //TO-DO: Remove this condition once gemv forwarding is enabled.
        if (n < 8 && n != 4)
           return false;
        if (m < 8 && m != 4)
           return false;
        tinyBLAS_Q0_PPC<block_q8_0> tb{
            k, (const block_q8_0 *)A, lda,
            (const block_q8_0 *)B, ldb,
            (float *)C, ldc,
            params->ith, params->nth};
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
            params->ith, params->nth};
        tb.matmul(m, n);
        return true;
#elif defined(__ARM_FEATURE_DOTPROD)
        tinyBLAS_Q0_ARM<block_q4_0> tb{
            k, (const block_q4_0 *)A, lda,
            (const block_q8_0 *)B, ldb,
            (float *)C, ldc,
            params->ith, params->nth};
        tb.matmul(m, n);
        return true;
#elif defined(__MMA__)
    //TO-DO: Remove this condition once gemv forwarding is enabled.
        if (n < 8 && n != 4)
           return false;
        if (m < 8 && m != 4)
           return false;
        tinyBLAS_Q0_PPC<block_q4_0> tb{
            k, (const block_q4_0 *)A, lda,
            (const block_q8_0 *)B, ldb,
            (float *)C, ldc,
            params->ith, params->nth};
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
            params->ith, params->nth};
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
            params->ith, params->nth};
        tb.matmul(m, n);
        return true;
#else
        return false;
#endif
    }

    default:
        return false;
    }

    (void)params;
    (void)m;
    (void)n;
    (void)k;
    (void)A;
    (void)lda;
    (void)B;
    (void)ldb;
    (void)C;
    (void)ldc;
    (void)Atype;
    (void)Btype;
    (void)Ctype;
}
