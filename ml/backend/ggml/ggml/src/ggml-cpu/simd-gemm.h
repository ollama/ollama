#pragma once

// Computes C[M x N] += A[M x K] * B[K x N]

#include "ggml-cpu-impl.h"
#include "vec.h"
#include "common.h"
#include "simd-mappings.h"

// TODO: add support for sizeless vector types
#if defined(GGML_SIMD) && !defined(__ARM_FEATURE_SVE) && !defined(__riscv_v_intrinsic)

// TODO: untested on avx512
// These are in units of GGML_F32_EPR
#if defined(__AVX512F__) || defined (__ARM_NEON__)
    static constexpr int GEMM_RM = 4;
    static constexpr int GEMM_RN = 4; // 16+4+1 = 25/32
#elif defined(__AVX2__) || defined(__AVX__)
    static constexpr int GEMM_RM = 6;
    static constexpr int GEMM_RN = 2; // 12+2+1 = 15/16
#else
    static constexpr int GEMM_RM = 2;
    static constexpr int GEMM_RN = 2;
#endif

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Waggressive-loop-optimizations"
#endif

template <int RM, int RN>
static inline void simd_gemm_ukernel(
    float       * GGML_RESTRICT C,
    const float * GGML_RESTRICT A,
    const float * GGML_RESTRICT B,
    int64_t K, int64_t N,
    int64_t ii, int64_t jj)
{
    static constexpr int KN = GGML_F32_EPR;

    GGML_F32_VEC acc[RM][RN];
    for (int i = 0; i < RM; i++) {
        for (int r = 0; r < RN; r++) {
            acc[i][r] = GGML_F32_VEC_LOAD(C + (ii + i) * N + jj + r * KN);
        }
    }

    for (int64_t kk = 0; kk < K; kk++) {
        GGML_F32_VEC Bv[RN];
        for (int r = 0; r < RN; r++) {
            Bv[r] = GGML_F32_VEC_LOAD(B + kk * N + jj + r * KN);
        }
        for (int i = 0; i < RM; i++) {
            GGML_F32_VEC p = GGML_F32_VEC_SET1(A[(ii + i) * K + kk]);
            for (int r = 0; r < RN; r++) {
                acc[i][r] = GGML_F32_VEC_FMA(acc[i][r], Bv[r], p);
            }
        }
    }

    for (int i = 0; i < RM; i++) {
        for (int r = 0; r < RN; r++) {
            GGML_F32_VEC_STORE(C + (ii + i) * N + jj + r * KN, acc[i][r]);
        }
    }
}

// C[M x N] += A[M x K] * B[K x N]
static void simd_gemm(
    float       * GGML_RESTRICT C,
    const float * GGML_RESTRICT A,
    const float * GGML_RESTRICT B,
    int64_t M, int64_t K, int64_t N)
{
    static constexpr int KN = GGML_F32_EPR;

    int64_t ii = 0;
    for (; ii + GEMM_RM <= M; ii += GEMM_RM) {
        int64_t jj = 0;
        for (; jj + GEMM_RN * KN <= N; jj += GEMM_RN * KN) {
            simd_gemm_ukernel<GEMM_RM, GEMM_RN>(C, A, B, K, N, ii, jj);
        }
        for (; jj + KN <= N; jj += KN) {
            simd_gemm_ukernel<GEMM_RM, 1>(C, A, B, K, N, ii, jj);
        }
        for (; jj < N; jj++) {
            for (int i = 0; i < GEMM_RM; i++) {
                float a = C[(ii + i) * N + jj];
                for (int64_t kk = 0; kk < K; kk++) {
                    a += A[(ii + i) * K + kk] * B[kk * N + jj];
                }
                C[(ii + i) * N + jj] = a;
            }
        }
    }

    // Tail rows: one at a time
    for (; ii < M; ii++) {
        int64_t jj = 0;
        for (; jj + GEMM_RN * KN <= N; jj += GEMM_RN * KN) {
            simd_gemm_ukernel<1, GEMM_RN>(C, A, B, K, N, ii, jj);
        }
        for (; jj + KN <= N; jj += KN) {
            simd_gemm_ukernel<1, 1>(C, A, B, K, N, ii, jj);
        }
        for (; jj < N; jj++) {
            float a = C[ii * N + jj];
            for (int64_t kk = 0; kk < K; kk++) {
                a += A[ii * K + kk] * B[kk * N + jj];
            }
            C[ii * N + jj] = a;
        }
    }
}

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif

#else // scalar path

static void simd_gemm(
    float       * GGML_RESTRICT C,
    const float * GGML_RESTRICT A,
    const float * GGML_RESTRICT B,
    int64_t M, int64_t K, int64_t N)
{
    for (int64_t i = 0; i < M; i++) {
        for (int64_t j = 0; j < N; j++) {
            float sum = C[i * N + j];
            for (int64_t kk = 0; kk < K; kk++) {
                sum += A[i * K + kk] * B[kk * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

#endif // GGML_SIMD
