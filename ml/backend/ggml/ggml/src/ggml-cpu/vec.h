// Vectorized functions for fundamental operations

#pragma once

#include "ggml-impl.h"
#include "simd-mappings.h"
#include "ggml.h"
#include "ggml-cpu.h"

#if defined(GGML_USE_ACCELERATE)
#include <Accelerate/Accelerate.h>
#endif

// floating point type used to accumulate sums
typedef double ggml_float;

#define GGML_GELU_FP16
#define GGML_GELU_QUICK_FP16

#define GGML_SOFT_MAX_UNROLL 4
#define GGML_VEC_DOT_UNROLL  2
#define GGML_VEC_MAD_UNROLL  32

#ifdef __cplusplus
extern "C" {
#endif

//
// global data
//

// precomputed gelu table for f16 (128 KB)
extern ggml_fp16_t ggml_table_gelu_f16[1 << 16];

// precomputed quick gelu table for f16 (128 KB)
extern ggml_fp16_t ggml_table_gelu_quick_f16[1 << 16];

//
// fundamental operations
//

void ggml_vec_dot_f32(int n, float * GGML_RESTRICT s, size_t bs, const float * GGML_RESTRICT x, size_t bx, const float * GGML_RESTRICT y, size_t by, int nrc);
void ggml_vec_dot_bf16(int n, float * GGML_RESTRICT s, size_t bs, ggml_bf16_t * GGML_RESTRICT x, size_t bx, ggml_bf16_t * GGML_RESTRICT y, size_t by, int nrc);
void ggml_vec_dot_f16(int n, float * GGML_RESTRICT s, size_t bs, ggml_fp16_t * GGML_RESTRICT x, size_t bx, ggml_fp16_t * GGML_RESTRICT y, size_t by, int nrc);

void ggml_vec_silu_f32(const int n, float * y, const float * x);
ggml_float ggml_vec_cvar_f32(const int n, float * y, const float * x, const float mean); //it will also center y ( y = y - mean )
ggml_float ggml_vec_soft_max_f32(const int n, float * y, const float * x, float max);
ggml_float ggml_vec_log_soft_max_f32(const int n, float * y, const float * x, float max);

inline static void ggml_vec_set_i8(const int n, int8_t * x, const int8_t v) { for (int i = 0; i < n; ++i) x[i] = v; }
inline static void ggml_vec_set_i16(const int n, int16_t * x, const int16_t v) { for (int i = 0; i < n; ++i) x[i] = v; }

inline static void ggml_vec_set_i32(const int n, int32_t * x, const int32_t   v) { for (int i = 0; i < n; ++i) x[i] = v;    }
inline static void ggml_vec_cpy_i32(const int n, int32_t * y, const int32_t * x) { for (int i = 0; i < n; ++i) y[i] = x[i]; }

inline static void ggml_vec_set_f16(const int n, ggml_fp16_t * x, const ggml_fp16_t v) { for (int i = 0; i < n; ++i) x[i] = v; }
inline static void ggml_vec_set_bf16(const int n, ggml_bf16_t * x, const ggml_bf16_t v) { for (int i = 0; i < n; ++i) x[i] = v; }

inline static void ggml_vec_add_f32 (const int n, float * z, const float * x, const float * y) {
    int i = 0;
#if defined(__AVX2__)
    for (; i + 7 < n; i += 8) {
        __m256 vx = _mm256_loadu_ps(x + i);
        __m256 vy = _mm256_loadu_ps(y + i);
        __m256 vz = _mm256_add_ps(vx, vy);
        _mm256_storeu_ps(z + i, vz);
    }
#endif
    for (; i < n; ++i) {
        z[i] = x[i] + y[i];
    }
}

inline static void ggml_vec_add_f16 (const int n, ggml_fp16_t * z, const ggml_fp16_t * x, const ggml_fp16_t * y) {
    for (int i = 0; i < n; ++i) {
        z[i] = GGML_CPU_FP32_TO_FP16(GGML_CPU_FP16_TO_FP32(x[i]) + GGML_CPU_FP16_TO_FP32(y[i]));
    }
}
inline static void ggml_vec_add1_f32(const int n, float * z, const float * x, const float   v) { for (int i = 0; i < n; ++i) z[i]  = x[i] + v;    }
inline static void ggml_vec_acc_f32 (const int n, float * y, const float * x)                  { for (int i = 0; i < n; ++i) y[i] += x[i];        }
inline static void ggml_vec_acc1_f32(const int n, float * y, const float   v)                  { for (int i = 0; i < n; ++i) y[i] += v;           }
inline static void ggml_vec_sub_f32 (const int n, float * z, const float * x, const float * y) { for (int i = 0; i < n; ++i) z[i]  = x[i] - y[i]; }
inline static void ggml_vec_sub_f16 (const int n, ggml_fp16_t * z, const ggml_fp16_t * x, const ggml_fp16_t * y) {
    for (int i = 0; i < n; ++i) {
        z[i] = GGML_CPU_FP32_TO_FP16(GGML_CPU_FP16_TO_FP32(x[i]) - GGML_CPU_FP16_TO_FP32(y[i]));
    }
}
inline static void ggml_vec_set_f32 (const int n, float * x, const float   v)                  { for (int i = 0; i < n; ++i) x[i]  = v;           }
inline static void ggml_vec_cpy_f32 (const int n, float * y, const float * x)                  { for (int i = 0; i < n; ++i) y[i]  = x[i];        }
inline static void ggml_vec_neg_f32 (const int n, float * y, const float * x)                  { for (int i = 0; i < n; ++i) y[i]  = -x[i];       }
inline static void ggml_vec_neg_f16 (const int n, ggml_fp16_t * y, const ggml_fp16_t * x) {
    for (int i = 0; i < n; ++i) {
        y[i] = GGML_CPU_FP32_TO_FP16(-GGML_CPU_FP16_TO_FP32(x[i]));
    }
}

inline static void ggml_vec_mul_f32 (const int n, float * z, const float * x, const float * y) { for (int i = 0; i < n; ++i) z[i]  = x[i]*y[i];   }
inline static void ggml_vec_mul_f16 (const int n, ggml_fp16_t * z, const ggml_fp16_t * x, const ggml_fp16_t * y) {
    for (int i = 0; i < n; ++i) {
        z[i] = GGML_CPU_FP32_TO_FP16(GGML_CPU_FP16_TO_FP32(x[i]) * GGML_CPU_FP16_TO_FP32(y[i]));
    }
}
inline static void ggml_vec_div_f32 (const int n, float * z, const float * x, const float * y) { for (int i = 0; i < n; ++i) z[i]  = x[i]/y[i];   }
inline static void ggml_vec_div_f16 (const int n, ggml_fp16_t * z, const ggml_fp16_t * x, const ggml_fp16_t * y) {
    for (int i = 0; i < n; ++i) {
        z[i] = GGML_CPU_FP32_TO_FP16(GGML_CPU_FP16_TO_FP32(x[i]) / GGML_CPU_FP16_TO_FP32(y[i]));
    }
}

// compute GGML_VEC_DOT_UNROLL dot products at once
// xs - x row stride in bytes
inline static void ggml_vec_dot_f16_unroll(const int n, const int xs, float * GGML_RESTRICT s, void * GGML_RESTRICT xv, ggml_fp16_t * GGML_RESTRICT y) {
    ggml_float sumf[GGML_VEC_DOT_UNROLL] = { 0.0 };

    ggml_fp16_t * GGML_RESTRICT x[GGML_VEC_DOT_UNROLL];

    for (int i = 0; i < GGML_VEC_DOT_UNROLL; ++i) {
        x[i] = (ggml_fp16_t *) ((char *) xv + i*xs);
    }

#if defined(GGML_SIMD)
    #if defined(__ARM_FEATURE_SVE)

        const int sve_register_length = svcntb() * 8;
        const int ggml_f16_epr = sve_register_length / 16; // running when 16
        const int ggml_f16_step = 8 * ggml_f16_epr; // choose 8 SVE registers

        const int np = (n & ~(ggml_f16_step - 1));

        svfloat16_t sum_00 = svdup_n_f16(0.0f);
        svfloat16_t sum_01 = svdup_n_f16(0.0f);
        svfloat16_t sum_02 = svdup_n_f16(0.0f);
        svfloat16_t sum_03 = svdup_n_f16(0.0f);

        svfloat16_t sum_10 = svdup_n_f16(0.0f);
        svfloat16_t sum_11 = svdup_n_f16(0.0f);
        svfloat16_t sum_12 = svdup_n_f16(0.0f);
        svfloat16_t sum_13 = svdup_n_f16(0.0f);

        svfloat16_t ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8;
        svfloat16_t ay1, ay2, ay3, ay4, ay5, ay6, ay7, ay8;

        for (int i = 0; i < np; i += ggml_f16_step) {
            ay1 = GGML_F16x_VEC_LOAD(y + i + 0 * ggml_f16_epr, 0); // 8 elements

            ax1 = GGML_F16x_VEC_LOAD(x[0] + i + 0*ggml_f16_epr, 0); // 8 elements
            sum_00 = GGML_F16x_VEC_FMA(sum_00, ax1, ay1);     // sum_00 = sum_00+ax1*ay1
            ax1 = GGML_F16x_VEC_LOAD(x[1] + i + 0*ggml_f16_epr, 0); // 8 elements
            sum_10 = GGML_F16x_VEC_FMA(sum_10, ax1, ay1);

            ay2 = GGML_F16x_VEC_LOAD(y + i + 1 * ggml_f16_epr, 1); // next 8 elements

            ax2 = GGML_F16x_VEC_LOAD(x[0] + i + 1*ggml_f16_epr, 1); // next 8 elements
            sum_01 = GGML_F16x_VEC_FMA(sum_01, ax2, ay2);
            ax2 = GGML_F16x_VEC_LOAD(x[1] + i + 1*ggml_f16_epr, 1);
            sum_11 = GGML_F16x_VEC_FMA(sum_11, ax2, ay2);

            ay3 = GGML_F16x_VEC_LOAD(y + i + 2 * ggml_f16_epr, 2);

            ax3 = GGML_F16x_VEC_LOAD(x[0] + i + 2*ggml_f16_epr, 2);
            sum_02 = GGML_F16x_VEC_FMA(sum_02, ax3, ay3);
            ax3 = GGML_F16x_VEC_LOAD(x[1] + i + 2*ggml_f16_epr, 2);
            sum_12 = GGML_F16x_VEC_FMA(sum_12, ax3, ay3);

            ay4 = GGML_F16x_VEC_LOAD(y + i + 3 * ggml_f16_epr, 3);

            ax4 = GGML_F16x_VEC_LOAD(x[0] + i + 3*ggml_f16_epr, 3);
            sum_03 = GGML_F16x_VEC_FMA(sum_03, ax4, ay4);
            ax4 = GGML_F16x_VEC_LOAD(x[1] + i + 3*ggml_f16_epr, 3);
            sum_13 = GGML_F16x_VEC_FMA(sum_13, ax4, ay4);

            ay5 = GGML_F16x_VEC_LOAD(y + i + 4 * ggml_f16_epr, 4);

            ax5 = GGML_F16x_VEC_LOAD(x[0] + i + 4*ggml_f16_epr, 4);

            sum_00 = GGML_F16x_VEC_FMA(sum_00, ax5, ay5);
            ax5 = GGML_F16x_VEC_LOAD(x[1] + i + 4*ggml_f16_epr, 4);
            sum_10 = GGML_F16x_VEC_FMA(sum_10, ax5, ay5);

            ay6 = GGML_F16x_VEC_LOAD(y + i + 5 * ggml_f16_epr, 5);

            ax6 = GGML_F16x_VEC_LOAD(x[0] + i + 5*ggml_f16_epr, 5);

            sum_01 = GGML_F16x_VEC_FMA(sum_01, ax6, ay6);
            ax6 = GGML_F16x_VEC_LOAD(x[1] + i + 5*ggml_f16_epr, 5);
            sum_11 = GGML_F16x_VEC_FMA(sum_11, ax6, ay6);

            ay7 = GGML_F16x_VEC_LOAD(y + i + 6 * ggml_f16_epr, 6);

            ax7 = GGML_F16x_VEC_LOAD(x[0] + i + 6*ggml_f16_epr, 6);

            sum_02 = GGML_F16x_VEC_FMA(sum_02, ax7, ay7);
            ax7 = GGML_F16x_VEC_LOAD(x[1] + i + 6*ggml_f16_epr, 6);
            sum_12 = GGML_F16x_VEC_FMA(sum_12, ax7, ay7);

            ay8 = GGML_F16x_VEC_LOAD(y + i + 7 * ggml_f16_epr, 7);

            ax8 = GGML_F16x_VEC_LOAD(x[0] + i + 7*ggml_f16_epr, 7);

            sum_03 = GGML_F16x_VEC_FMA(sum_03, ax8, ay8);
            ax8 = GGML_F16x_VEC_LOAD(x[1] + i + 7*ggml_f16_epr, 7);
            sum_13 = GGML_F16x_VEC_FMA(sum_13, ax8, ay8);
        }

        const int np2 = (n & ~(ggml_f16_epr - 1));
        for (int k = np; k < np2; k += ggml_f16_epr) {
            svfloat16_t ry = GGML_F16x_VEC_LOAD(y + k, 0);

            svfloat16_t rx = GGML_F16x_VEC_LOAD(x[0] + k, 0);
            sum_00 = GGML_F16x_VEC_FMA(sum_00, rx, ry);
            rx = GGML_F16x_VEC_LOAD(x[1] + k, 0);
            sum_10 = GGML_F16x_VEC_FMA(sum_10, rx, ry);
        }

        if (np2 < n) {
            svbool_t pg = svwhilelt_b16(np2, n);
            svfloat16_t hx_0 = svld1_f16(pg, (const __fp16 *)(x[0] + np2));
            svfloat16_t hx_1 = svld1_f16(pg, (const __fp16 *)(x[1] + np2));
            svfloat16_t hy = svld1_f16(pg, (const __fp16 *)(y + np2));

            sum_00 = svmad_f16_x(pg, hx_0, hy, sum_00);
            sum_10 = svmad_f16_x(pg, hx_1, hy, sum_10);
        }
        GGML_F16x_VEC_REDUCE(sumf[0], sum_00, sum_01, sum_02, sum_03);
        GGML_F16x_VEC_REDUCE(sumf[1], sum_10, sum_11, sum_12, sum_13);
    #elif defined(__riscv_v_intrinsic)
      // todo: RVV impl
      for (int i = 0; i < n; ++i) {
          for (int j = 0; j < GGML_VEC_DOT_UNROLL; ++j) {
              sumf[j] += (ggml_float)(GGML_CPU_FP16_TO_FP32(x[j][i])*GGML_CPU_FP16_TO_FP32(y[i]));
          }
      }
    #else
        const int np = (n & ~(GGML_F16_STEP - 1));

        GGML_F16_VEC sum[GGML_VEC_DOT_UNROLL][GGML_F16_ARR] = { { GGML_F16_VEC_ZERO } };

        GGML_F16_VEC ax[GGML_F16_ARR];
        GGML_F16_VEC ay[GGML_F16_ARR];

        for (int i = 0; i < np; i += GGML_F16_STEP) {
            for (int j = 0; j < GGML_F16_ARR; j++) {
                ay[j] = GGML_F16_VEC_LOAD(y + i + j*GGML_F16_EPR, j);

                for (int k = 0; k < GGML_VEC_DOT_UNROLL; ++k) {
                    ax[j] = GGML_F16_VEC_LOAD(x[k] + i + j*GGML_F16_EPR, j);

                    sum[k][j] = GGML_F16_VEC_FMA(sum[k][j], ax[j], ay[j]);
                }
            }
        }

        // reduce sum0..sum3 to sum0
        for (int k = 0; k < GGML_VEC_DOT_UNROLL; ++k) {
            GGML_F16_VEC_REDUCE(sumf[k], sum[k]);
        }

        // leftovers
        for (int i = np; i < n; ++i) {
            for (int j = 0; j < GGML_VEC_DOT_UNROLL; ++j) {
                sumf[j] += (ggml_float)(GGML_CPU_FP16_TO_FP32(x[j][i])*GGML_CPU_FP16_TO_FP32(y[i]));
            }
        }
    #endif
#else
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < GGML_VEC_DOT_UNROLL; ++j) {
            sumf[j] += (ggml_float)(GGML_CPU_FP16_TO_FP32(x[j][i])*GGML_CPU_FP16_TO_FP32(y[i]));
        }
    }
#endif

    for (int i = 0; i < GGML_VEC_DOT_UNROLL; ++i) {
        s[i] = (float)sumf[i];
    }
}

inline static void ggml_vec_mad_f32(const int n, float * GGML_RESTRICT y, const float * GGML_RESTRICT x, const float v) {
#if defined(GGML_SIMD)
    #if defined(__ARM_FEATURE_SVE)

        const int sve_register_length = ggml_cpu_get_sve_cnt() * 8;
        const int ggml_f32_epr = sve_register_length / 32;//8;//svcntw(); // SVE128:4, SVE256:8, SVE512:16
        const int ggml_f32_step = 8 * ggml_f32_epr; // choose 8 SVE registers
        GGML_F32_VEC vx = GGML_F32_VEC_SET1(v);

        const int np = (n & ~(ggml_f32_step - 1));
        svfloat32_t ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8;
        svfloat32_t ay1, ay2, ay3, ay4, ay5, ay6, ay7, ay8;
        for (int i = 0; i < np; i += ggml_f32_step) {

            ax1 = GGML_F32_VEC_LOAD(x + i);
            ay1 = GGML_F32_VEC_LOAD(y + i);
            ay1 = GGML_F32_VEC_FMA(ay1, ax1, vx);

            GGML_F32_VEC_STORE(y + i, ay1);

            ax2 = GGML_F32_VEC_LOAD(x + i + 1*ggml_f32_epr);
            ay2 = GGML_F32_VEC_LOAD(y + i + 1*ggml_f32_epr);
            ay2 = GGML_F32_VEC_FMA(ay2, ax2, vx);

            GGML_F32_VEC_STORE(y + i + 1*ggml_f32_epr, ay2);

            ax3 = GGML_F32_VEC_LOAD(x + i + 2*ggml_f32_epr);
            ay3 = GGML_F32_VEC_LOAD(y + i + 2*ggml_f32_epr);
            ay3 = GGML_F32_VEC_FMA(ay3, ax3, vx);

            GGML_F32_VEC_STORE(y + i + 2*ggml_f32_epr, ay3);

            ax4 = GGML_F32_VEC_LOAD(x + i + 3*ggml_f32_epr);
            ay4 = GGML_F32_VEC_LOAD(y + i + 3*ggml_f32_epr);
            ay4 = GGML_F32_VEC_FMA(ay4, ax4, vx);

            GGML_F32_VEC_STORE(y + i + 3*ggml_f32_epr, ay4);

            ax5 = GGML_F32_VEC_LOAD(x + i + 4*ggml_f32_epr);
            ay5 = GGML_F32_VEC_LOAD(y + i + 4*ggml_f32_epr);
            ay5 = GGML_F32_VEC_FMA(ay5, ax5, vx);

            GGML_F32_VEC_STORE(y + i + 4*ggml_f32_epr, ay5);

            ax6 = GGML_F32_VEC_LOAD(x + i + 5*ggml_f32_epr);
            ay6 = GGML_F32_VEC_LOAD(y + i + 5*ggml_f32_epr);
            ay6 = GGML_F32_VEC_FMA(ay6, ax6, vx);

            GGML_F32_VEC_STORE(y + i + 5*ggml_f32_epr, ay6);

            ax7 = GGML_F32_VEC_LOAD(x + i + 6*ggml_f32_epr);
            ay7 = GGML_F32_VEC_LOAD(y + i + 6*ggml_f32_epr);
            ay7 = GGML_F32_VEC_FMA(ay7, ax7, vx);

            GGML_F32_VEC_STORE(y + i + 6*ggml_f32_epr, ay7);

            ax8 = GGML_F32_VEC_LOAD(x + i + 7*ggml_f32_epr);
            ay8 = GGML_F32_VEC_LOAD(y + i + 7*ggml_f32_epr);
            ay8 = GGML_F32_VEC_FMA(ay8, ax8, vx);

            GGML_F32_VEC_STORE(y + i + 7*ggml_f32_epr, ay8);
        }
        // leftovers
        // Since 8 unrolls are done in above loop, leftovers lie in range [0, ggml_f32_step] which is handled in below loop
        const int np2 = (n & ~(ggml_f32_epr - 1));
        for (int i = np; i < np2; i += ggml_f32_epr) {
            ax1 = GGML_F32_VEC_LOAD(x + i);
            ay1 = GGML_F32_VEC_LOAD(y + i);
            ay1 = GGML_F32_VEC_FMA(ay1, ax1, vx);

            GGML_F32_VEC_STORE(y + i, ay1);
        }
        // maximum number of leftover elements will be less that ggml_f32_epr. Apply predicated svmad on available elements only
        if (np2 < n) {
            svbool_t pg =svwhilelt_b32(np2, n);
            ax1 = svld1_f32(pg, x + np2);
            ay1 = svld1_f32(pg, y + np2);
            ay1 = svmad_f32_m(pg, ax1, vx, ay1);

            svst1_f32(pg, y + np2, ay1);
        }
    #elif defined(__riscv_v_intrinsic)
        for (int i = 0, avl; i < n; i += avl) {
            avl = __riscv_vsetvl_e32m8(n - i);
            vfloat32m8_t ax = __riscv_vle32_v_f32m8(&x[i], avl);
            vfloat32m8_t ay = __riscv_vle32_v_f32m8(&y[i], avl);
            vfloat32m8_t ny = __riscv_vfmadd_vf_f32m8(ax, v, ay, avl);
            __riscv_vse32_v_f32m8(&y[i], ny, avl);
        }
    #else
        const int np = (n & ~(GGML_F32_STEP - 1));

        GGML_F32_VEC vx = GGML_F32_VEC_SET1(v);

        GGML_F32_VEC ax[GGML_F32_ARR];
        GGML_F32_VEC ay[GGML_F32_ARR];

        for (int i = 0; i < np; i += GGML_F32_STEP) {
            for (int j = 0; j < GGML_F32_ARR; j++) {
                ax[j] = GGML_F32_VEC_LOAD(x + i + j*GGML_F32_EPR);
                ay[j] = GGML_F32_VEC_LOAD(y + i + j*GGML_F32_EPR);
                ay[j] = GGML_F32_VEC_FMA(ay[j], ax[j], vx);

                GGML_F32_VEC_STORE(y + i + j*GGML_F32_EPR, ay[j]);
            }
        }

        // leftovers
        for (int i = np; i < n; ++i) {
            y[i] += x[i]*v;
        }
    #endif
#else
    // scalar
    for (int i = 0; i < n; ++i) {
        y[i] += x[i]*v;
    }
#endif
}

inline static void ggml_vec_mad_f16(const int n, ggml_fp16_t * GGML_RESTRICT y, const ggml_fp16_t * GGML_RESTRICT x, const float v) {
#if defined(GGML_SIMD)
    #if defined(__ARM_FEATURE_SVE)
        const int sve_register_length = svcntb() * 8;
        const int ggml_f16_epr = sve_register_length / 16;
        const int ggml_f16_step = 8 * ggml_f16_epr;

        GGML_F16x_VEC vx = GGML_F16x_VEC_SET1(v);

        const int np= (n & ~(ggml_f16_step - 1));

        svfloat16_t ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8;
        svfloat16_t ay1, ay2, ay3, ay4, ay5, ay6, ay7, ay8;
        for (int i = 0; i < np; i += ggml_f16_step) {
            ax1 = GGML_F16x_VEC_LOAD(x + i + 0 * ggml_f16_epr, 0);
            ay1 = GGML_F16x_VEC_LOAD(y + i + 0 * ggml_f16_epr, 0);
            ay1 = GGML_F16x_VEC_FMA(ay1, ax1, vx);

            GGML_F16x_VEC_STORE(y + i + 0 * ggml_f16_epr, ay1, 0);

            ax2 = GGML_F16x_VEC_LOAD(x + i + 1 * ggml_f16_epr, 1);
            ay2 = GGML_F16x_VEC_LOAD(y + i + 1 * ggml_f16_epr, 1);
            ay2 = GGML_F16x_VEC_FMA(ay2, ax2, vx);

            GGML_F16x_VEC_STORE(y + i + 1 * ggml_f16_epr, ay2, 1);

            ax3 = GGML_F16x_VEC_LOAD(x + i + 2 * ggml_f16_epr, 2);
            ay3 = GGML_F16x_VEC_LOAD(y + i + 2 * ggml_f16_epr, 2);
            ay3 = GGML_F16x_VEC_FMA(ay3, ax3, vx);

            GGML_F16x_VEC_STORE(y + i + 2 * ggml_f16_epr, ay3, 2);

            ax4 = GGML_F16x_VEC_LOAD(x + i + 3 * ggml_f16_epr, 3);
            ay4 = GGML_F16x_VEC_LOAD(y + i + 3 * ggml_f16_epr, 3);
            ay4 = GGML_F16x_VEC_FMA(ay4, ax4, vx);

            GGML_F16x_VEC_STORE(y + i + 3 * ggml_f16_epr, ay4, 3);

            ax5 = GGML_F16x_VEC_LOAD(x + i + 4 * ggml_f16_epr, 4);
            ay5 = GGML_F16x_VEC_LOAD(y + i + 4 * ggml_f16_epr, 4);
            ay5 = GGML_F16x_VEC_FMA(ay5, ax5, vx);

            GGML_F16x_VEC_STORE(y + i + 4 * ggml_f16_epr, ay5, 4);

            ax6 = GGML_F16x_VEC_LOAD(x + i + 5 * ggml_f16_epr, 5);
            ay6 = GGML_F16x_VEC_LOAD(y + i + 5 * ggml_f16_epr, 5);
            ay6 = GGML_F16x_VEC_FMA(ay6, ax6, vx);

            GGML_F16x_VEC_STORE(y + i + 5 * ggml_f16_epr, ay6, 5);

            ax7 = GGML_F16x_VEC_LOAD(x + i + 6 * ggml_f16_epr, 6);
            ay7 = GGML_F16x_VEC_LOAD(y + i + 6 * ggml_f16_epr, 6);
            ay7 = GGML_F16x_VEC_FMA(ay7, ax7, vx);

            GGML_F16x_VEC_STORE(y + i + 6 * ggml_f16_epr, ay7, 6);

            ax8 = GGML_F16x_VEC_LOAD(x + i + 7 * ggml_f16_epr, 7);
            ay8 = GGML_F16x_VEC_LOAD(y + i + 7 * ggml_f16_epr, 7);
            ay8 = GGML_F16x_VEC_FMA(ay8, ax8, vx);

            GGML_F16x_VEC_STORE(y + i + 7 * ggml_f16_epr, ay8, 7);
        }
        const int np2 = (n & ~(ggml_f16_epr - 1));
        for (int k = np; k < np2; k += ggml_f16_epr) {
            svfloat16_t rx = GGML_F16x_VEC_LOAD(x + k, 0);
            svfloat16_t ry = GGML_F16x_VEC_LOAD(y + k, 0);
            ry = GGML_F16x_VEC_FMA(ry, rx, vx);

            GGML_F16x_VEC_STORE(y + k, ry, 0);
        }

        if (np2 < n) {
            svbool_t pg = svwhilelt_b16(np2, n);
            svfloat16_t hx = svld1_f16(pg, (const __fp16 *)(x + np2));
            svfloat16_t hy = svld1_f16(pg, (const __fp16 *)(y + np2));
            hy = svmad_f16_x(pg, hx, vx, hy);
            svst1_f16(pg, (__fp16 *)(y + np2), hy);
        }

    #elif defined(__riscv_v_intrinsic)
        // todo: RVV impl
        // scalar
        for (int i = 0; i < n; ++i) {
            y[i] = GGML_CPU_FP32_TO_FP16(GGML_CPU_FP16_TO_FP32(y[i]) + GGML_CPU_FP16_TO_FP32(x[i])*v);
        }
    #else
        const int np = (n & ~(GGML_F16_STEP - 1));

        GGML_F16_VEC vx = GGML_F16_VEC_SET1(v);

        GGML_F16_VEC ax[GGML_F16_ARR];
        GGML_F16_VEC ay[GGML_F16_ARR];

        for (int i = 0; i < np; i += GGML_F16_STEP) {
            for (int j = 0; j < GGML_F16_ARR; j++) {
                ax[j] = GGML_F16_VEC_LOAD(x + i + j*GGML_F16_EPR, j);
                ay[j] = GGML_F16_VEC_LOAD(y + i + j*GGML_F16_EPR, j);
                ay[j] = GGML_F16_VEC_FMA(ay[j], ax[j], vx);

                GGML_F16_VEC_STORE(y + i + j*GGML_F16_EPR, ay, j);
            }
        }

        // leftovers
        for (int i = np; i < n; ++i) {
            y[i] = GGML_CPU_FP32_TO_FP16(GGML_CPU_FP16_TO_FP32(y[i]) + GGML_CPU_FP16_TO_FP32(x[i])*v);
        }
    #endif
#else
    // scalar
    for (int i = 0; i < n; ++i) {
        y[i] = GGML_CPU_FP32_TO_FP16(GGML_CPU_FP16_TO_FP32(y[i]) + GGML_CPU_FP16_TO_FP32(x[i])*v);
    }
#endif
}

// xs and vs are byte strides of x and v
inline static void ggml_vec_mad_f32_unroll(const int n, const int xs, const int vs, float * GGML_RESTRICT y, const float * GGML_RESTRICT xv, const float * GGML_RESTRICT vv) {

    const float * GGML_RESTRICT x[GGML_VEC_MAD_UNROLL];
    const float * GGML_RESTRICT v[GGML_VEC_MAD_UNROLL];

    for (int i = 0; i < GGML_VEC_MAD_UNROLL; ++i) {
        x[i] = (const float *) ((const char *) xv + i*xs);
        v[i] = (const float *) ((const char *) vv + i*vs);
    }

#if defined(GGML_SIMD)
    #if defined(__ARM_FEATURE_SVE)
        // scalar Route to scalar implementation       //TODO: Write SVE code
        for (int k = 0; k < GGML_VEC_MAD_UNROLL; ++k) {
            for (int i = 0; i < n; ++i) {
                y[i] += x[k][i]*v[k][0];
            }
        }
    #elif defined(__riscv_v_intrinsic)
        for (int i = 0, avl; i < n; i += avl) {
            avl = __riscv_vsetvl_e32m8(n - i);
            vfloat32m8_t ay = __riscv_vle32_v_f32m8(&y[i], avl);
            for (int k = 0; k < GGML_VEC_MAD_UNROLL; k++) {
                vfloat32m8_t ax = __riscv_vle32_v_f32m8(&x[k][i], avl);
                ay = __riscv_vfmadd_vf_f32m8(ax, v[k][0], ay, avl);
            }
            __riscv_vse32_v_f32m8(&y[i], ay, avl);
        }
    #else
        const int np = (n & ~(GGML_F32_STEP - 1));

        GGML_F32_VEC vx[GGML_VEC_MAD_UNROLL];

        for (int k = 0; k < GGML_VEC_MAD_UNROLL; ++k) {
            vx[k] = GGML_F32_VEC_SET1(v[k][0]);
        }

        GGML_F32_VEC ax[GGML_VEC_MAD_UNROLL][GGML_F32_ARR];
        GGML_F32_VEC ay[GGML_F32_ARR];

        for (int i = 0; i < np; i += GGML_F32_STEP) {
            for (int j = 0; j < GGML_F32_ARR; j++) {
                ay[j] = GGML_F32_VEC_LOAD(y + i + j*GGML_F32_EPR);

                for (int k = 0; k < GGML_VEC_MAD_UNROLL; ++k) {
                    ax[k][j] = GGML_F32_VEC_LOAD(x[k] + i + j*GGML_F32_EPR);
                    ay[j] = GGML_F32_VEC_FMA(ay[j], ax[k][j], vx[k]);
                }

                GGML_F32_VEC_STORE(y + i + j*GGML_F32_EPR, ay[j]);
            }
        }

        // leftovers
        for (int k = 0; k < GGML_VEC_MAD_UNROLL; ++k) {
            for (int i = np; i < n; ++i) {
                y[i] += x[k][i]*v[k][0];
            }
        }
    #endif
#else
    // scalar
    for (int k = 0; k < GGML_VEC_MAD_UNROLL; ++k) {
        for (int i = 0; i < n; ++i) {
            y[i] += x[k][i]*v[k][0];
        }
    }
#endif
}

inline static void ggml_vec_mad1_f32(const int n, float * y, const float * x, const float s, const float b) {
#if defined(GGML_USE_ACCELERATE)
    vDSP_vsmsa(x, 1, &s, &b, y, 1, n);
#elif defined(GGML_SIMD)
    #if defined(__ARM_FEATURE_SVE)
        // scalar ; TODO: Write SVE code
        for (int i = 0; i < n; ++i) {
            y[i] = x[i]*s + b;
        }
    #elif defined(__riscv_v_intrinsic)
        for (int i = 0, avl; i < n; i += avl) {
            avl = __riscv_vsetvl_e32m8(n - i);
            vfloat32m8_t ax = __riscv_vle32_v_f32m8(&x[i], avl);
            vfloat32m8_t vb = __riscv_vfmv_v_f_f32m8(b, avl);
            vfloat32m8_t ny = __riscv_vfmadd_vf_f32m8(ax, s, vb, avl);
            __riscv_vse32_v_f32m8(&y[i], ny, avl);
        }
    #else
        const int np = (n & ~(GGML_F32_STEP - 1));

        GGML_F32_VEC vs = GGML_F32_VEC_SET1(s);
        GGML_F32_VEC vb = GGML_F32_VEC_SET1(b);

        GGML_F32_VEC ay[GGML_F32_ARR];

        for (int i = 0; i < np; i += GGML_F32_STEP) {
            for (int j = 0; j < GGML_F32_ARR; j++) {
                ay[j] = GGML_F32_VEC_LOAD(x + i + j*GGML_F32_EPR);
                ay[j] = GGML_F32_VEC_FMA(vb, ay[j], vs);

                GGML_F32_VEC_STORE(y + i + j*GGML_F32_EPR, ay[j]);
            }
        }

        // leftovers
        for (int i = np; i < n; ++i) {
            y[i] = x[i]*s + b;
        }
    #endif
#else
    // scalar
    for (int i = 0; i < n; ++i) {
        y[i] = x[i]*s + b;
    }
#endif
}

//inline static void ggml_vec_scale_f32(const int n, float * y, const float   v) { for (int i = 0; i < n; ++i) y[i] *= v;          }
inline static void ggml_vec_scale_f32(const int n, float * y, const float   v) {
#if defined(GGML_USE_ACCELERATE)
    vDSP_vsmul(y, 1, &v, y, 1, n);
#elif defined(GGML_SIMD)
    #if defined(__ARM_FEATURE_SVE)
        const int sve_register_length = ggml_cpu_get_sve_cnt() * 8;
        const int ggml_f32_epr = sve_register_length / 32;//8;//svcntw(); // SVE128:4, SVE256:8, SVE512:16
        const int ggml_f32_step = 2 * ggml_f32_epr;

        GGML_F32_VEC vx = GGML_F32_VEC_SET1(v);
        const int np = (n & ~(ggml_f32_step - 1));
        svfloat32_t ay1;
        svfloat32_t ay2;
        for (int i = 0; i < np; i += ggml_f32_step) {
            ay1 = GGML_F32_VEC_LOAD(y + i);
            ay1 = GGML_F32_VEC_MUL(ay1, vx);
            GGML_F32_VEC_STORE(y + i, ay1);

            ay2 = GGML_F32_VEC_LOAD(y + i + 1*ggml_f32_epr);
            ay2 = GGML_F32_VEC_MUL(ay2, vx);
            GGML_F32_VEC_STORE(y + i + 1*ggml_f32_epr, ay2);
        }
        // leftovers
        // maximum number of leftover elements will be less that ggml_f32_epr. Apply predicated svmad on available elements only
        for (int i = np; i < n; i += ggml_f32_epr) {
            svbool_t pg = svwhilelt_b32(i, n);
            ay1 = svld1_f32(pg, y + i);
            ay1 = svmul_f32_m(pg, ay1, vx);
            svst1_f32(pg, y + i, ay1);
        }
    #elif defined(__riscv_v_intrinsic)
        for (int i = 0, avl; i < n; i += avl) {
            avl = __riscv_vsetvl_e32m8(n - i);
            vfloat32m8_t ay = __riscv_vle32_v_f32m8(&y[i], avl);
            vfloat32m8_t ny = __riscv_vfmul_vf_f32m8(ay, v, avl);
            __riscv_vse32_v_f32m8(&y[i], ny, avl);
        }
    #else
        const int np = (n & ~(GGML_F32_STEP - 1));

        GGML_F32_VEC vx = GGML_F32_VEC_SET1(v);

        GGML_F32_VEC ay[GGML_F32_ARR];

        for (int i = 0; i < np; i += GGML_F32_STEP) {
            for (int j = 0; j < GGML_F32_ARR; j++) {
                ay[j] = GGML_F32_VEC_LOAD(y + i + j*GGML_F32_EPR);
                ay[j] = GGML_F32_VEC_MUL(ay[j], vx);

                GGML_F32_VEC_STORE(y + i + j*GGML_F32_EPR, ay[j]);
            }
        }

        // leftovers
        for (int i = np; i < n; ++i) {
            y[i] *= v;
        }
    #endif
#else
    // scalar
    for (int i = 0; i < n; ++i) {
        y[i] *= v;
    }
#endif
}

inline static void ggml_vec_scale_f16(const int n, ggml_fp16_t * y, const float v) {
#if defined(GGML_SIMD)
    #if defined(__ARM_FEATURE_SVE)
        const int sve_register_length = svcntb() * 8;
        const int ggml_f16_epr = sve_register_length / 16;
        const int ggml_f16_step = 2 * ggml_f16_epr;

        GGML_F16x_VEC vx =  GGML_F16x_VEC_SET1(v);
        const int np = (n & ~(ggml_f16_step - 1));
        svfloat16_t ay1, ay2;

        for (int i = 0; i < np; i += ggml_f16_step) {
            ay1 = GGML_F16x_VEC_LOAD(y + i + 0*ggml_f16_epr, 0);
            ay1 = GGML_F16x_VEC_MUL(ay1, vx);
            GGML_F16x_VEC_STORE(y + i + 0*ggml_f16_epr, ay1, 0);

            ay2 = GGML_F16x_VEC_LOAD(y + i + 1*ggml_f16_epr, 1);
            ay2 = GGML_F16x_VEC_MUL(ay2, vx);
            GGML_F16x_VEC_STORE(y + i + 1*ggml_f16_epr, ay2, 1);
        }
        // leftovers
        // maximum number of leftover elements will be less that ggmlF_16x_epr. Apply predicated svmad on available elements only
        if (np < n) {
            svbool_t pg = svwhilelt_b16(np, n);
            svfloat16_t hy = svld1_f16(pg, (__fp16 *)(y + np));
            svfloat16_t out = svmul_f16_m(pg, hy, vx);
            svst1_f16(pg, (__fp16 *)(y + np), out);
        }
    #elif defined(__riscv_v_intrinsic)
        // todo: RVV impl
        // scalar
        for (int i = 0; i < n; ++i) {
            y[i] = GGML_CPU_FP32_TO_FP16(GGML_CPU_FP16_TO_FP32(y[i])*v);
        }
    #else
        const int np = (n & ~(GGML_F16_STEP - 1));

        GGML_F16_VEC vx = GGML_F16_VEC_SET1(v);

        GGML_F16_VEC ay[GGML_F16_ARR];

        for (int i = 0; i < np; i += GGML_F16_STEP) {
            for (int j = 0; j < GGML_F16_ARR; j++) {
                ay[j] = GGML_F16_VEC_LOAD(y + i + j*GGML_F16_EPR, j);
                ay[j] = GGML_F16_VEC_MUL(ay[j], vx);

                GGML_F16_VEC_STORE(y + i + j*GGML_F16_EPR, ay, j);
            }
        }

        // leftovers
        for (int i = np; i < n; ++i) {
            y[i] = GGML_CPU_FP32_TO_FP16(GGML_CPU_FP16_TO_FP32(y[i])*v);
        }
    #endif
#else
    // scalar
    for (int i = 0; i < n; ++i) {
        y[i] = GGML_CPU_FP32_TO_FP16(GGML_CPU_FP16_TO_FP32(y[i])*v);
    }
#endif
}

inline static void ggml_vec_norm_f32 (const int n, float * s, const float * x) { ggml_vec_dot_f32(n, s, 0, x, 0, x, 0, 1); *s = sqrtf(*s);   }
inline static void ggml_vec_sqr_f32  (const int n, float * y, const float * x) { for (int i = 0; i < n; ++i) y[i] = x[i]*x[i];   }
inline static void ggml_vec_sqr_f16 (const int n, ggml_fp16_t * y, const ggml_fp16_t * x) {
    for (int i = 0; i < n; ++i) {
        float v = GGML_CPU_FP16_TO_FP32(x[i]);
        y[i] = GGML_CPU_FP32_TO_FP16(v*v);
    }
}
inline static void ggml_vec_sqrt_f32 (const int n, float * y, const float * x) { for (int i = 0; i < n; ++i) y[i] = sqrtf(x[i]); }
inline static void ggml_vec_sqrt_f16 (const int n, ggml_fp16_t * y, const ggml_fp16_t * x) {
    for (int i = 0; i < n; ++i) {
        y[i] = GGML_CPU_FP32_TO_FP16(sqrtf(GGML_CPU_FP16_TO_FP32(x[i])));
    }
}
inline static void ggml_vec_log_f32  (const int n, float * y, const float * x) { for (int i = 0; i < n; ++i) y[i] = logf(x[i]);  }
inline static void ggml_vec_log_f16 (const int n, ggml_fp16_t * y, const ggml_fp16_t * x) {
    for (int i = 0; i < n; ++i) {
        y[i] = GGML_CPU_FP32_TO_FP16(logf(GGML_CPU_FP16_TO_FP32(x[i])));
    }
}
inline static void ggml_vec_sin_f32  (const int n, float * y, const float * x) { for (int i = 0; i < n; ++i) y[i] = sinf(x[i]);  }
inline static void ggml_vec_sin_f16 (const int n, ggml_fp16_t * y, const ggml_fp16_t * x) {
    for (int i = 0; i < n; ++i) {
        y[i] = GGML_CPU_FP32_TO_FP16(sinf(GGML_CPU_FP16_TO_FP32(x[i])));
    }
}
inline static void ggml_vec_cos_f32  (const int n, float * y, const float * x) { for (int i = 0; i < n; ++i) y[i] = cosf(x[i]);  }
inline static void ggml_vec_cos_f16 (const int n, ggml_fp16_t * y, const ggml_fp16_t * x) {
    for (int i = 0; i < n; ++i) {
        y[i] = GGML_CPU_FP32_TO_FP16(cosf(GGML_CPU_FP16_TO_FP32(x[i])));
    }
}
inline static void ggml_vec_abs_f32  (const int n, float * y, const float * x) { for (int i = 0; i < n; ++i) y[i] = fabsf(x[i]); }
inline static void ggml_vec_abs_f16 (const int n, ggml_fp16_t * y, const ggml_fp16_t * x) {
    for (int i = 0; i < n; ++i) {
        y[i] = GGML_CPU_FP32_TO_FP16(fabsf(GGML_CPU_FP16_TO_FP32(x[i])));
    }
}
inline static void ggml_vec_sgn_f32  (const int n, float * y, const float * x) { for (int i = 0; i < n; ++i) y[i] = (x[i] > 0.f) ? 1.f : ((x[i] < 0.f) ? -1.f : 0.f); }
inline static void ggml_vec_sgn_f16 (const int n, ggml_fp16_t * y, const ggml_fp16_t * x) {
    for (int i = 0; i < n; ++i) {
        float v = GGML_CPU_FP16_TO_FP32(x[i]);
        y[i] = GGML_CPU_FP32_TO_FP16((v > 0.f) ? 1.f : ((v < 0.f) ? -1.f : 0.f));
    }
}
inline static void ggml_vec_step_f32 (const int n, float * y, const float * x) { for (int i = 0; i < n; ++i) y[i] = (x[i] > 0.f) ? 1.f : 0.f; }
inline static void ggml_vec_step_f16 (const int n, ggml_fp16_t * y, const ggml_fp16_t * x) {
    for (int i = 0; i < n; ++i) {
        y[i] = GGML_CPU_FP32_TO_FP16((GGML_CPU_FP16_TO_FP32(x[i]) > 0.f) ? 1.f : 0.f);
    }
}
inline static void ggml_vec_tanh_f32 (const int n, float * y, const float * x) { for (int i = 0; i < n; ++i) y[i] = tanhf(x[i]);  }
inline static void ggml_vec_tanh_f16 (const int n, ggml_fp16_t * y, const ggml_fp16_t * x) {
    for (int i = 0; i < n; ++i) {
        y[i] = GGML_CPU_FP32_TO_FP16(tanhf(GGML_CPU_FP16_TO_FP32(x[i])));
    }
}
inline static void ggml_vec_elu_f32  (const int n, float * y, const float * x) { for (int i = 0; i < n; ++i) y[i] = (x[i] > 0.f) ? x[i] : expm1f(x[i]); }
inline static void ggml_vec_elu_f16 (const int n, ggml_fp16_t * y, const ggml_fp16_t * x) {
    for (int i = 0; i < n; ++i) {
        const float v = GGML_CPU_FP16_TO_FP32(x[i]);
        y[i] = GGML_CPU_FP32_TO_FP16((v > 0.f) ? v : expm1f(v));
    }
}
inline static void ggml_vec_relu_f32 (const int n, float * y, const float * x) { for (int i = 0; i < n; ++i) y[i] = (x[i] > 0.f) ? x[i] : 0.f; }
inline static void ggml_vec_relu_f16 (const int n, ggml_fp16_t * y, const ggml_fp16_t * x) {
    for (int i = 0; i < n; ++i) {
        float v = GGML_CPU_FP16_TO_FP32(x[i]);
        y[i] = GGML_CPU_FP32_TO_FP16((v > 0.f) ? v : 0.f);
    }
}
inline static void ggml_vec_leaky_relu_f32 (const int n, float * y, const float * x, const float ns) { for (int i = 0; i < n; ++i) y[i] = ((x[i] > 0.f) ? x[i] : 0.f) + ns * ((x[i] < 0.0f) ? x[i] : 0.f); }
inline static void ggml_vec_leaky_relu_f16 (const int n, ggml_fp16_t * y, const ggml_fp16_t * x, const float ns) {
    for (int i = 0; i < n; ++i) {
        float v = GGML_CPU_FP16_TO_FP32(x[i]);
        y[i] = GGML_CPU_FP32_TO_FP16(((v > 0.f) ? v : 0.f) + ns * ((v < 0.0f) ? v : 0.f));
    }
}
inline static void ggml_vec_sigmoid_f32 (const int n, float * y, const float * x) { for (int i = 0; i < n; ++i) y[i] = 1.f / (1.f + expf(-x[i])); }
inline static void ggml_vec_sigmoid_f16 (const int n, ggml_fp16_t * y, const ggml_fp16_t * x) {
    for (int i = 0; i < n; ++i) {
        y[i] = GGML_CPU_FP32_TO_FP16(1.f / (1.f + expf(-GGML_CPU_FP16_TO_FP32(x[i]))));
    }
}
// TODO: optimize performance
inline static void ggml_vec_hardswish_f32 (const int n, float * y, const float * x) { for (int i = 0; i < n; ++i) y[i] = x[i] * fminf(1.0f, fmaxf(0.0f, (x[i] + 3.0f) / 6.0f)); }
inline static void ggml_vec_hardswish_f16 (const int n, ggml_fp16_t * y, const ggml_fp16_t * x) {
    for (int i = 0; i < n; ++i) {
        float v = GGML_CPU_FP16_TO_FP32(x[i]);
        y[i] = GGML_CPU_FP32_TO_FP16(v * fminf(1.0f, fmaxf(0.0f, (v + 3.0f) / 6.0f)));
    }
}
inline static void ggml_vec_hardsigmoid_f32 (const int n, float * y, const float * x) { for (int i = 0; i < n; ++i) y[i] = fminf(1.0f, fmaxf(0.0f, (x[i] + 3.0f) / 6.0f)); }
inline static void ggml_vec_hardsigmoid_f16 (const int n, ggml_fp16_t * y, const ggml_fp16_t * x) {
    for (int i = 0; i < n; ++i) {
        y[i] = GGML_CPU_FP32_TO_FP16(fminf(1.0f, fmaxf(0.0f, (GGML_CPU_FP16_TO_FP32(x[i]) + 3.0f) / 6.0f)));
    }
}
inline static void ggml_vec_exp_f32 (const int n, float * y, const float * x) { for (int i = 0; i < n; ++i) y[i] = expf(x[i]); }
inline static void ggml_vec_exp_f16 (const int n, ggml_fp16_t * y, const ggml_fp16_t * x) {
    for (int i = 0; i < n; ++i) {
        y[i] = GGML_CPU_FP32_TO_FP16(expf(GGML_CPU_FP16_TO_FP32(x[i])));
    }
}

static const float GELU_COEF_A     = 0.044715f;
static const float GELU_QUICK_COEF = -1.702f;
static const float SQRT_2_OVER_PI  = 0.79788456080286535587989211986876f;
static const float SQRT_2_INV      = 0.70710678118654752440084436210484f;

inline static float ggml_gelu_f32(float x) {
    return 0.5f*x*(1.0f + tanhf(SQRT_2_OVER_PI*x*(1.0f + GELU_COEF_A*x*x)));
}

inline static void ggml_vec_gelu_f16(const int n, ggml_fp16_t * y, const ggml_fp16_t * x) {
    const uint16_t * i16 = (const uint16_t *) x;
    for (int i = 0; i < n; ++i) {
        y[i] = ggml_table_gelu_f16[i16[i]];
    }
}

inline static void ggml_vec_gelu_erf_f16(const int n, ggml_fp16_t * y, const ggml_fp16_t * x) {
    for (int i = 0; i < n; ++i) {
        float xi = GGML_CPU_FP16_TO_FP32(x[i]);
        float res = 0.5f*xi*(1.0f + erff(xi*SQRT_2_INV));
        y[i] = GGML_CPU_FP32_TO_FP16(res);
    }
}

#ifdef GGML_GELU_FP16
inline static void ggml_vec_gelu_f32(const int n, float * y, const float * x) {
    uint16_t t;
    for (int i = 0; i < n; ++i) {
        if (x[i] <= -10.0f) {
            y[i] = 0.0f;
        } else if (x[i] >= 10.0f) {
            y[i] = x[i];
        } else {
            ggml_fp16_t fp16 = GGML_CPU_FP32_TO_FP16(x[i]);
            memcpy(&t, &fp16, sizeof(uint16_t));
            y[i] = GGML_CPU_FP16_TO_FP32(ggml_table_gelu_f16[t]);
        }
    }
}
#else
inline static void ggml_vec_gelu_f32(const int n, float * y, const float * x) {
    for (int i = 0; i < n; ++i) {
        y[i] = ggml_gelu_f32(x[i]);
    }
}
#endif

inline static void ggml_vec_gelu_erf_f32(const int n, float * y, const float * x) {
    for (int i = 0; i < n; ++i) {
        float xi = x[i];
        y[i] = 0.5f*xi*(1.0f + erff(xi*SQRT_2_INV));
    }
}

inline static float ggml_gelu_quick_f32(float x) {
    return x*(1.0f/(1.0f+expf(GELU_QUICK_COEF*x)));
}

//inline static void ggml_vec_gelu_quick_f16(const int n, ggml_fp16_t * y, const ggml_fp16_t * x) {
//    const uint16_t * i16 = (const uint16_t *) x;
//    for (int i = 0; i < n; ++i) {
//        y[i] = ggml_table_gelu_quick_f16[i16[i]];
//    }
//}

#ifdef GGML_GELU_QUICK_FP16
inline static void ggml_vec_gelu_quick_f32(const int n, float * y, const float * x) {
    uint16_t t;
    for (int i = 0; i < n; ++i) {
        ggml_fp16_t fp16 = GGML_CPU_FP32_TO_FP16(x[i]);
        memcpy(&t, &fp16, sizeof(uint16_t));
        y[i] = GGML_CPU_FP16_TO_FP32(ggml_table_gelu_quick_f16[t]);
    }
}
#else
inline static void ggml_vec_gelu_quick_f32(const int n, float * y, const float * x) {
    for (int i = 0; i < n; ++i) {
        y[i] = ggml_gelu_quick_f32(x[i]);
    }
}
#endif

inline static void ggml_vec_gelu_quick_f16(const int n, ggml_fp16_t * y, const ggml_fp16_t * x) {
    for (int i = 0; i < n; ++i) {
        float v = GGML_CPU_FP16_TO_FP32(x[i]);
        y[i] = GGML_CPU_FP32_TO_FP16(v*(1.0f/(1.0f+expf(GELU_QUICK_COEF*v))));
    }
}

// Sigmoid Linear Unit (SiLU) function
inline static float ggml_silu_f32(float x) {
    return x/(1.0f + expf(-x));
}
inline static ggml_fp16_t ggml_silu_f16(ggml_fp16_t x) {
    float v = GGML_CPU_FP16_TO_FP32(x);
    return GGML_CPU_FP32_TO_FP16(v/(1.0f + expf(-v)));
}

#if __FINITE_MATH_ONLY__
#error "some routines in ggml.c require non-finite math arithmetics -- pass -fno-finite-math-only to the compiler to fix"
#error "ref: https://github.com/ggml-org/llama.cpp/pull/7154#issuecomment-2143844461"
#endif

/* Below function was borrowed from the GitHub repository:
https://github.com/openvinotoolkit/openvino/blob/master/src/plugins/intel_cpu/src/nodes/kernels/scaled_attn/common.hpp */
#if defined(__ARM_FEATURE_SVE) && defined(__aarch64__)
    inline static svfloat32_t exp_ps_sve(svbool_t pg, svfloat32_t src) {
        // Constants
        const svfloat32_t log2_e = svdup_n_f32(1.4426950409f);
        const svfloat32_t ln2 = svdup_n_f32(0.6931473921f);
        const svfloat32_t half_ln2_sq = svdup_n_f32(0.2413862043f);
        const svuint32_t not_mask17 = svdup_n_u32(~((1u << 17) - 1));
        const svfloat32_t one = svdup_n_f32(1.0f);
        const svfloat32_t inactive1 = svdup_n_f32(0.0f);
        const svint32_t inactive2 = svdup_n_s32(0);

        // Algorithm starts here
        svfloat32_t t0 = svmul_f32_m(pg, src, log2_e);  // y = x * log2(e)
        svfloat32_t t1 = svrintm_f32_m(inactive1, pg, t0);         // rount to int (float)
        svint32_t t2 = svcvt_s32_f32_m(inactive2, pg, t1);         // n

        t1 = svsub_f32_m(pg, t0, t1);   // a = y - floor(y)
        t1 = svadd_f32_m(pg, t1, one);  // b = a + 1

        svuint32_t t3 = svlsr_n_u32_m(pg, svreinterpret_u32_f32(t1), 17);  // v = b >> 17 (u32)
        svfloat32_t t4 = svexpa_f32(t3);                                   // c = fexpa(v)
        t4 = svscale_f32_m(pg, t4, t2);                                    // fexpa(v) * 2^(n)

        // and_(t2.d, t1.d, not_mask17.d)
        svfloat32_t t5 = svreinterpret_f32_u32(svand_u32_m(pg, svreinterpret_u32_f32(t1), not_mask17));
        t5 = svsub_f32_m(pg, t1, t5);                // z
        t0 = svmla_f32_m(pg, ln2, t5, half_ln2_sq);  // ln2 + half_ln2_sq * z
        t0 = svmla_f32_m(pg, one, t5, t0);           // 1 + (ln2 * z) + (half_ln2_sq * z * z)
        t0 = svmul_f32_m(pg, t0, t4);                // Final result

        return t0;
    }
#endif

#if defined(__ARM_FEATURE_SVE) && defined(__aarch64__)

inline static svfloat32_t ggml_v_expf(svbool_t pg, svfloat32_t x) {
    const svfloat32_t r = svdup_n_f32_x(pg, 0x1.8p23f);
    const svfloat32_t z = svmla_n_f32_x(pg, r, x, 0x1.715476p+0f);
    const svfloat32_t n = svsub_f32_x(pg, z, r);
    const svfloat32_t b = svmls_n_f32_x(pg, svmls_n_f32_x(pg, x, n, 0x1.62e4p-1f), n, 0x1.7f7d1cp-20f);
    const svuint32_t e = svlsl_n_u32_x(pg, svreinterpret_u32_f32(z), 23);
    const svfloat32_t k = svreinterpret_f32_u32(svadd_u32_x(pg, e, svreinterpret_u32_f32(svdup_n_f32_x(pg, 1))));
    const svbool_t c = svacgt_n_f32(pg, n, 126);
    const svfloat32_t u = svmul_f32_x(pg, b, b);
    const svfloat32_t j = svmla_f32_x(pg,
        svmul_n_f32_x(pg, b, 0x1.ffffecp-1f),
        svmla_f32_x(pg, svmla_f32_x(pg, svdup_n_f32_x(pg, 0x1.fffdb6p-2f), svdup_n_f32_x(pg, 0x1.555e66p-3f), b),
                        svmla_f32_x(pg, svdup_n_f32_x(pg, 0x1.573e2ep-5f), svdup_n_f32_x(pg, 0x1.0e4020p-7f), b), u), u);
    const svuint32_t d = svdup_n_u32_z(svcmple_n_f32(pg, n, 0.0), 0x82000000);
    const svfloat32_t s1 = svreinterpret_f32_u32(svadd_n_u32_x(pg, d, 0x7f000000));
    const svfloat32_t s2 = svreinterpret_f32_u32(svsub_u32_x(pg, e, d));
    return svsel_f32(svacgt_f32(pg, n, svdup_n_f32_x(pg, 192)), svmul_f32_x(pg, s1, s1),
                     svsel_f32(c, svmul_f32_x(pg, svmla_f32_x(pg, s2, s2, j), s1), svmla_f32_x(pg, k, k, j)));
}

// computes silu x/(1+exp(-x)) in single precision vector
inline static svfloat32_t ggml_v_silu(svbool_t pg, svfloat32_t x) {
    const svfloat32_t one = svdup_n_f32_x(pg, 1.0f);
    const svfloat32_t zero = svdup_n_f32_x(pg, 0.0f);
    const svfloat32_t neg_x = svsub_f32_x(pg, zero, x);
    const svfloat32_t exp_neg_x = ggml_v_expf(pg, neg_x);
    const svfloat32_t one_plus_exp_neg_x = svadd_f32_x(pg, one, exp_neg_x);
    return svdiv_f32_x(pg, x, one_plus_exp_neg_x);
}

#elif defined(__ARM_NEON) && defined(__aarch64__)

// adapted from arm limited optimized routine
// the maximum error is 1.45358 plus 0.5 ulps
// numbers above 88.38 will flush to infinity
// numbers beneath -103.97 will flush to zero
inline static float32x4_t ggml_v_expf(float32x4_t x) {
    const float32x4_t r = vdupq_n_f32(0x1.8p23f);
    const float32x4_t z = vfmaq_f32(r, x, vdupq_n_f32(0x1.715476p+0f));
    const float32x4_t n = vsubq_f32(z, r);
    const float32x4_t b = vfmsq_f32(vfmsq_f32(x, n, vdupq_n_f32(0x1.62e4p-1f)), n,
                                    vdupq_n_f32(0x1.7f7d1cp-20f));
    const uint32x4_t e = vshlq_n_u32(vreinterpretq_u32_f32(z), 23);
    const float32x4_t k = vreinterpretq_f32_u32(vaddq_u32(e, vreinterpretq_u32_f32(vdupq_n_f32(1))));
    const uint32x4_t c = vcagtq_f32(n, vdupq_n_f32(126));
    const float32x4_t u = vmulq_f32(b, b);
    const float32x4_t j = vfmaq_f32(
        vmulq_f32(vdupq_n_f32(0x1.ffffecp-1f), b),
        vfmaq_f32(vfmaq_f32(vdupq_n_f32(0x1.fffdb6p-2f), vdupq_n_f32(0x1.555e66p-3f), b),
                  vfmaq_f32(vdupq_n_f32(0x1.573e2ep-5f), vdupq_n_f32(0x1.0e4020p-7f), b), u), u);
    if (!vpaddd_u64(vreinterpretq_u64_u32(c)))
        return vfmaq_f32(k, j, k);
    const uint32x4_t d = vandq_u32(vclezq_f32(n), vdupq_n_u32(0x82000000));
    const float32x4_t s1 = vreinterpretq_f32_u32(vaddq_u32(d, vdupq_n_u32(0x7f000000)));
    const float32x4_t s2 = vreinterpretq_f32_u32(vsubq_u32(e, d));
    return vbslq_f32(vcagtq_f32(n, vdupq_n_f32(192)), vmulq_f32(s1, s1),
                     vbslq_f32(c, vmulq_f32(vfmaq_f32(s2, s2, j), s1), vfmaq_f32(k, k, j)));
}

// computes silu x/(1+exp(-x)) in single precision vector
inline static float32x4_t ggml_v_silu(float32x4_t x) {
    const float32x4_t one = vdupq_n_f32(1.0f);
    const float32x4_t zero = vdupq_n_f32(0.0f);
    const float32x4_t neg_x = vsubq_f32(zero, x);
    const float32x4_t exp_neg_x = ggml_v_expf(neg_x);
    const float32x4_t one_plus_exp_neg_x = vaddq_f32(one, exp_neg_x);
    return vdivq_f32(x, one_plus_exp_neg_x);
}

#elif defined(__AVX512F__) && defined(__AVX512DQ__)

// adapted from arm limited optimized routine
// the maximum error is 1.45358 plus 0.5 ulps
// numbers above 88.38 will flush to infinity
// numbers beneath -103.97 will flush to zero
inline static __m512 ggml_v_expf(__m512 x) {
  const __m512 r = _mm512_set1_ps(0x1.8p23f);
  const __m512 z = _mm512_fmadd_ps(x, _mm512_set1_ps(0x1.715476p+0f), r);
  const __m512 n = _mm512_sub_ps(z, r);
  const __m512 b =
      _mm512_fnmadd_ps(n, _mm512_set1_ps(0x1.7f7d1cp-20f),
                       _mm512_fnmadd_ps(n, _mm512_set1_ps(0x1.62e4p-1f), x));
  const __mmask16 d =
      _mm512_cmp_ps_mask(_mm512_abs_ps(n), _mm512_set1_ps(192), _CMP_GT_OQ);
  const __m512 u = _mm512_mul_ps(b, b);
  const __m512 j = _mm512_fmadd_ps(
      _mm512_fmadd_ps(_mm512_fmadd_ps(_mm512_set1_ps(0x1.0e4020p-7f), b,
                                      _mm512_set1_ps(0x1.573e2ep-5f)),
                      u,
                      _mm512_fmadd_ps(_mm512_set1_ps(0x1.555e66p-3f), b,
                                      _mm512_set1_ps(0x1.fffdb6p-2f))),
      u,
      _mm512_fmadd_ps(_mm512_set1_ps(0x1.ffffecp-1f), b, _mm512_set1_ps(1.0F)));
  const __m512 res = _mm512_scalef_ps(j, n);
  if (_mm512_kortestz(d, d))
    return res;
  const __m512 zero = _mm512_setzero_ps();
  const __m512 alt = _mm512_mask_blend_ps(
      _mm512_cmp_ps_mask(n, zero, _CMP_LE_OQ), _mm512_set1_ps(INFINITY), zero);
  return _mm512_mask_blend_ps(d, res, alt);
}

// computes silu x/(1+exp(-x)) in single precision vector
inline static __m512 ggml_v_silu(__m512 x) {
    const __m512 one = _mm512_set1_ps(1);
    const __m512 zero = _mm512_setzero_ps();
    const __m512 neg_x = _mm512_sub_ps(zero, x);
    const __m512 exp_neg_x = ggml_v_expf(neg_x);
    const __m512 one_plus_exp_neg_x = _mm512_add_ps(one, exp_neg_x);
    return _mm512_div_ps(x, one_plus_exp_neg_x);
}

#elif defined(__AVX2__) && defined(__FMA__)

// adapted from arm limited optimized routine
// the maximum error is 1.45358 plus 0.5 ulps
// numbers above 88.38 will flush to infinity
// numbers beneath -103.97 will flush to zero
inline static __m256 ggml_v_expf(__m256 x) {
  const __m256 r = _mm256_set1_ps(0x1.8p23f);
  const __m256 z = _mm256_fmadd_ps(x, _mm256_set1_ps(0x1.715476p+0f), r);
  const __m256 n = _mm256_sub_ps(z, r);
  const __m256 b = _mm256_fnmadd_ps(n, _mm256_set1_ps(0x1.7f7d1cp-20f),
                                    _mm256_fnmadd_ps(n, _mm256_set1_ps(0x1.62e4p-1f), x));
  const __m256i e = _mm256_slli_epi32(_mm256_castps_si256(z), 23);
  const __m256 k = _mm256_castsi256_ps(
      _mm256_add_epi32(e, _mm256_castps_si256(_mm256_set1_ps(1))));
  const __m256i c = _mm256_castps_si256(
      _mm256_cmp_ps(_mm256_andnot_ps(_mm256_set1_ps(-0.f), n),
                    _mm256_set1_ps(126), _CMP_GT_OQ));
  const __m256 u = _mm256_mul_ps(b, b);
  const __m256 j = _mm256_fmadd_ps(_mm256_fmadd_ps(_mm256_fmadd_ps(_mm256_set1_ps(0x1.0e4020p-7f), b,
                                                                   _mm256_set1_ps(0x1.573e2ep-5f)), u,
                                                   _mm256_fmadd_ps(_mm256_set1_ps(0x1.555e66p-3f), b,
                                                                   _mm256_set1_ps(0x1.fffdb6p-2f))),
                                   u, _mm256_mul_ps(_mm256_set1_ps(0x1.ffffecp-1f), b));
  if (!_mm256_movemask_ps(_mm256_castsi256_ps(c)))
    return _mm256_fmadd_ps(j, k, k);
  const __m256i g = _mm256_and_si256(
      _mm256_castps_si256(_mm256_cmp_ps(n, _mm256_setzero_ps(), _CMP_LE_OQ)),
      _mm256_set1_epi32(0x82000000u));
  const __m256 s1 =
      _mm256_castsi256_ps(_mm256_add_epi32(g, _mm256_set1_epi32(0x7f000000u)));
  const __m256 s2 = _mm256_castsi256_ps(_mm256_sub_epi32(e, g));
  const __m256i d = _mm256_castps_si256(
      _mm256_cmp_ps(_mm256_andnot_ps(_mm256_set1_ps(-0.f), n),
                    _mm256_set1_ps(192), _CMP_GT_OQ));
  return _mm256_or_ps(
      _mm256_and_ps(_mm256_castsi256_ps(d), _mm256_mul_ps(s1, s1)),
      _mm256_andnot_ps(
          _mm256_castsi256_ps(d),
          _mm256_or_ps(
              _mm256_and_ps(_mm256_castsi256_ps(c),
                            _mm256_mul_ps(_mm256_fmadd_ps(s2, j, s2), s1)),
              _mm256_andnot_ps(_mm256_castsi256_ps(c), _mm256_fmadd_ps(k, j, k)))));
}

// computes silu x/(1+exp(-x)) in single precision vector
inline static __m256 ggml_v_silu(__m256 x) {
    const __m256 one = _mm256_set1_ps(1);
    const __m256 zero = _mm256_setzero_ps();
    const __m256 neg_x = _mm256_sub_ps(zero, x);
    const __m256 exp_neg_x = ggml_v_expf(neg_x);
    const __m256 one_plus_exp_neg_x = _mm256_add_ps(one, exp_neg_x);
    return _mm256_div_ps(x, one_plus_exp_neg_x);
}

#elif defined(__SSE2__) // __AVX2__ / __ARM_NEON

#if defined(__FMA__)
#define MADD128(x, y, z) _mm_fmadd_ps(x, y, z)
#define NMADD128(x, y, z) _mm_fnmadd_ps(x, y, z)
#else
#define MADD128(x, y, z) _mm_add_ps(_mm_mul_ps(x, y), z)
#define NMADD128(x, y, z) _mm_sub_ps(z, _mm_mul_ps(x, y))
#endif

// adapted from arm limited optimized routine
// the maximum error is 1.45358 plus 0.5 ulps
// numbers above 88.38 will flush to infinity
// numbers beneath -103.97 will flush to zero
inline static __m128 ggml_v_expf(__m128 x) {
    const __m128 r = _mm_set1_ps(0x1.8p23f);
    const __m128 z = MADD128(x, _mm_set1_ps(0x1.715476p+0f), r);
    const __m128 n = _mm_sub_ps(z, r);
    const __m128 b =
        NMADD128(n, _mm_set1_ps(0x1.7f7d1cp-20f), NMADD128(n, _mm_set1_ps(0x1.62e4p-1f), x));
    const __m128i e = _mm_slli_epi32(_mm_castps_si128(z), 23);
    const __m128 k = _mm_castsi128_ps(_mm_add_epi32(e, _mm_castps_si128(_mm_set1_ps(1))));
    const __m128i c =
        _mm_castps_si128(_mm_cmpgt_ps(_mm_andnot_ps(_mm_set1_ps(-0.f), n), _mm_set1_ps(126)));
    const __m128 u = _mm_mul_ps(b, b);
    const __m128 j =
        MADD128(MADD128(MADD128(_mm_set1_ps(0x1.0e4020p-7f), b, _mm_set1_ps(0x1.573e2ep-5f)), u,
                        MADD128(_mm_set1_ps(0x1.555e66p-3f), b, _mm_set1_ps(0x1.fffdb6p-2f))),
                u, _mm_mul_ps(_mm_set1_ps(0x1.ffffecp-1f), b));
    if (!_mm_movemask_epi8(c))
        return MADD128(j, k, k);
    const __m128i g = _mm_and_si128(_mm_castps_si128(_mm_cmple_ps(n, _mm_setzero_ps())),
                                    _mm_set1_epi32(0x82000000u));
    const __m128 s1 = _mm_castsi128_ps(_mm_add_epi32(g, _mm_set1_epi32(0x7f000000u)));
    const __m128 s2 = _mm_castsi128_ps(_mm_sub_epi32(e, g));
    const __m128i d =
        _mm_castps_si128(_mm_cmpgt_ps(_mm_andnot_ps(_mm_set1_ps(-0.f), n), _mm_set1_ps(192)));
    return _mm_or_ps(
        _mm_and_ps(_mm_castsi128_ps(d), _mm_mul_ps(s1, s1)),
        _mm_andnot_ps(_mm_castsi128_ps(d),
                      _mm_or_ps(_mm_and_ps(_mm_castsi128_ps(c), _mm_mul_ps(MADD128(s2, j, s2), s1)),
                                _mm_andnot_ps(_mm_castsi128_ps(c), MADD128(k, j, k)))));
}

// computes silu x/(1+exp(-x)) in single precision vector
inline static __m128 ggml_v_silu(__m128 x) {
    const __m128 one = _mm_set1_ps(1);
    const __m128 zero = _mm_setzero_ps();
    const __m128 neg_x = _mm_sub_ps(zero, x);
    const __m128 exp_neg_x = ggml_v_expf(neg_x);
    const __m128 one_plus_exp_neg_x = _mm_add_ps(one, exp_neg_x);
    return _mm_div_ps(x, one_plus_exp_neg_x);
}

#elif defined(__riscv_v_intrinsic)

// adapted from arm limited optimized routine
// the maximum error is 1.45358 plus 0.5 ulps
// numbers above 88.38 will flush to infinity
// numbers beneath -103.97 will flush to zero
inline static vfloat32m2_t ggml_v_expf_m2(vfloat32m2_t x, int vl) {
    const vfloat32m2_t r = __riscv_vfmv_v_f_f32m2(0x1.8p23f, vl);
#ifdef __riscv_xtheadvector
    // workaround for compiler bug (gcc 14.3.0: Error: unrecognized opcode `th.vmv1r.v v2,v4')
    vfloat32m2_t z = __riscv_vfadd_vf_f32m2(r, 0.0f, vl);
    z = __riscv_vfmacc_vf_f32m2(z, 0x1.715476p+0f, x, vl);
#else
    const vfloat32m2_t z = __riscv_vfmacc_vf_f32m2(r, 0x1.715476p+0f, x, vl);
#endif
    const vfloat32m2_t n = __riscv_vfsub_vv_f32m2(z, r, vl);
    const vfloat32m2_t b = __riscv_vfnmsac_vf_f32m2(__riscv_vfnmsac_vf_f32m2(x, 0x1.62e4p-1f, n, vl),
                                                    0x1.7f7d1cp-20f, n, vl);
    const vuint32m2_t e = __riscv_vsll_vx_u32m2(__riscv_vreinterpret_v_f32m2_u32m2(z), 23, vl);
    const vfloat32m2_t k = __riscv_vreinterpret_v_u32m2_f32m2(__riscv_vadd_vx_u32m2(e, 0x3f800000, vl)); // 1.0f
    const vbool16_t c = __riscv_vmfgt_vf_f32m2_b16(__riscv_vfabs_v_f32m2(n, vl), 126.0f, vl);
    const vfloat32m2_t u = __riscv_vfmul_vv_f32m2(b, b, vl);
    const vfloat32m2_t j = __riscv_vfmacc_vv_f32m2(
        __riscv_vfmul_vf_f32m2(b, 0x1.ffffecp-1f, vl),
        __riscv_vfmacc_vv_f32m2(
            __riscv_vfmacc_vf_f32m2(__riscv_vfmv_v_f_f32m2(0x1.fffdb6p-2f, vl), 0x1.555e66p-3f, b, vl),
            __riscv_vfmacc_vf_f32m2(__riscv_vfmv_v_f_f32m2(0x1.573e2ep-5f, vl), 0x1.0e4020p-7f, b, vl),
            u, vl), u, vl);
    if (!__riscv_vcpop_m_b16(c, vl))
        return __riscv_vfmacc_vv_f32m2(k, j, k, vl);
    const vbool16_t  dm = __riscv_vmfle_vf_f32m2_b16(n, 0.0f, vl);
    const vuint32m2_t d = __riscv_vmerge_vxm_u32m2(__riscv_vmv_v_x_u32m2(0, vl), 0x82000000, dm, vl);
    const vfloat32m2_t s1 = __riscv_vreinterpret_v_u32m2_f32m2(__riscv_vadd_vx_u32m2(d, 0x7f000000, vl));
    const vfloat32m2_t s2 = __riscv_vreinterpret_v_u32m2_f32m2(__riscv_vsub_vv_u32m2(e, d, vl));
    const vfloat32m2_t r1 = __riscv_vmerge_vvm_f32m2(
        __riscv_vfmacc_vv_f32m2(k, k, j, vl),
        __riscv_vfmul_vv_f32m2(__riscv_vfmacc_vv_f32m2(s2, s2, j, vl), s1, vl),
        c, vl);
    return __riscv_vmerge_vvm_f32m2(
        r1, __riscv_vfmul_vv_f32m2(s1, s1, vl),
        __riscv_vmfgt_vf_f32m2_b16(__riscv_vfabs_v_f32m2(n, vl), 192.0f, vl),
        vl);
}

// computes silu x/(1+exp(-x)) in single precision vector
inline static vfloat32m2_t ggml_v_silu_m2(vfloat32m2_t x, int vl) {
    const vfloat32m2_t neg_x = __riscv_vfneg_v_f32m2(x, vl);
    const vfloat32m2_t exp_neg_x = ggml_v_expf_m2(neg_x, vl);
    const vfloat32m2_t one_plus_exp_neg_x = __riscv_vfadd_vf_f32m2(exp_neg_x, 1.0f, vl);
    return __riscv_vfdiv_vv_f32m2(x, one_plus_exp_neg_x, vl);
}

#endif // __ARM_NEON / __AVX2__ / __SSE2__ / __riscv_v_intrinsic

inline static void ggml_vec_silu_f16(const int n, ggml_fp16_t * y, const ggml_fp16_t * x) {
    for (int i = 0; i < n; ++i) {
        y[i] = ggml_silu_f16(x[i]);
    }
}

inline static float ggml_silu_backward_f32(float x, float dy) {
    const float s = 1.0f/(1.0f + expf(-x));
    return dy*s*(1.0f + x*(1.0f - s));
}

inline static ggml_fp16_t ggml_silu_backward_f16(ggml_fp16_t x, ggml_fp16_t dy) {
    const float v = GGML_CPU_FP16_TO_FP32(x);
    const float s = 1.0f/(1.0f + expf(-v));
    return GGML_CPU_FP32_TO_FP16(GGML_CPU_FP16_TO_FP32(dy)*s*(1.0f + v*(1.0f - s)));
}

inline static void ggml_vec_silu_backward_f32(const int n, float * dx, const float * x, const float * dy) {
    for (int i = 0; i < n; ++i) {
        dx[i] = ggml_silu_backward_f32(x[i], dy[i]);
    }
}

inline static void ggml_vec_silu_backward_f16(const int n, ggml_fp16_t * dx, const ggml_fp16_t * x, const ggml_fp16_t * dy) {
    for (int i = 0; i < n; ++i) {
        dx[i] = ggml_silu_backward_f16(x[i], dy[i]);
    }
}

inline static void ggml_vec_reglu_f32 (const int n, float * y, const float * x, const float * g) {
    for (int i = 0; i < n; ++i) {
        y[i] = (x[i] > 0.f) ? x[i] * g[i] : 0.f;
    }
}

inline static void ggml_vec_reglu_f16 (const int n, ggml_fp16_t * y, const ggml_fp16_t * x, const ggml_fp16_t * g) {
    for (int i = 0; i < n; ++i) {
        float v = GGML_CPU_FP16_TO_FP32(x[i]);
        y[i] = GGML_CPU_FP32_TO_FP16((v > 0.f) ? v * GGML_CPU_FP16_TO_FP32(g[i]) : 0.f);
    }
}

#ifdef GGML_GELU_FP16
inline static void ggml_vec_geglu_f32(const int n, float * y, const float * x, const float * g) {
    uint16_t t;
    for (int i = 0; i < n; ++i) {
        if (x[i] <= -10.0f) {
            y[i] = 0.0f;
        } else if (x[i] >= 10.0f) {
            y[i] = x[i] * g[i];
        } else {
            ggml_fp16_t fp16 = GGML_CPU_FP32_TO_FP16(x[i]);
            memcpy(&t, &fp16, sizeof(uint16_t));
            y[i] = GGML_CPU_FP16_TO_FP32(ggml_table_gelu_f16[t]) * g[i];
        }
    }
}
#else
inline static void ggml_vec_geglu_f32(const int n, float * y, const float * x, const float * g) {
    for (int i = 0; i < n; ++i) {
        y[i] = ggml_gelu_f32(x[i]) * g[i];
    }
}
#endif

inline static void ggml_vec_geglu_f16(const int n, ggml_fp16_t * y, const ggml_fp16_t * x, const ggml_fp16_t * g) {
    const uint16_t * i16 = (const uint16_t *) x;
    for (int i = 0; i < n; ++i) {
        float v = GGML_CPU_FP16_TO_FP32(g[i]);
        y[i] = GGML_CPU_FP32_TO_FP16(GGML_CPU_FP16_TO_FP32(ggml_table_gelu_f16[i16[i]]) * v);
    }
}

void ggml_vec_swiglu_f32(const int n, float * y, const float * x, const float * g);

inline static void ggml_vec_swiglu_f16(const int n, ggml_fp16_t * y, const ggml_fp16_t * x, const ggml_fp16_t * g) {
    for (int i = 0; i < n; ++i) {
        float xi = GGML_CPU_FP16_TO_FP32(x[i]);
        float gi = GGML_CPU_FP16_TO_FP32(g[i]);
        y[i] = GGML_CPU_FP32_TO_FP16((xi/(1.0f + expf(-xi))) * gi);
    }
}

inline static void ggml_vec_geglu_erf_f32(const int n, float * y, const float * x, const float * g) {
    for (int i = 0; i < n; ++i) {
        float xi = x[i];
        y[i] = 0.5f * xi * (1.0f + erff(xi*SQRT_2_INV)) * g[i];
    }
}

inline static void ggml_vec_geglu_erf_f16(const int n, ggml_fp16_t * y, const ggml_fp16_t * x, const ggml_fp16_t * g) {
    for (int i = 0; i < n; ++i) {
        float xi = GGML_CPU_FP16_TO_FP32(x[i]);
        float gi = GGML_CPU_FP16_TO_FP32(g[i]);
        y[i] = GGML_CPU_FP32_TO_FP16(0.5f * xi * (1.0f + erff(xi*SQRT_2_INV)) * gi);
    }
}

#ifdef GGML_GELU_QUICK_FP16
inline static void ggml_vec_geglu_quick_f32(const int n, float * y, const float * x, const float * g) {
    uint16_t t;
    for (int i = 0; i < n; ++i) {
        ggml_fp16_t fp16 = GGML_CPU_FP32_TO_FP16(x[i]);
        memcpy(&t, &fp16, sizeof(uint16_t));
        y[i] = GGML_CPU_FP16_TO_FP32(ggml_table_gelu_quick_f16[t]) * g[i];
    }
}
#else
inline static void ggml_vec_geglu_quick_f32(const int n, float * y, const float * x, const float * g) {
    for (int i = 0; i < n; ++i) {
        y[i] = ggml_gelu_quick_f32(x[i]) * g[i];
    }
}
#endif

inline static void ggml_vec_geglu_quick_f16(const int n, ggml_fp16_t * y, const ggml_fp16_t * x, const ggml_fp16_t * g) {
    const uint16_t * i16 = (const uint16_t *) x;
    for (int i = 0; i < n; ++i) {
        float v = GGML_CPU_FP16_TO_FP32(g[i]);
        y[i] = GGML_CPU_FP32_TO_FP16(GGML_CPU_FP16_TO_FP32(ggml_table_gelu_quick_f16[i16[i]]) * v);
    }
}

inline static void ggml_vec_sum_f32(const int n, float * s, const float * x) {
#ifndef GGML_USE_ACCELERATE
    ggml_float sum = 0.0;
    for (int i = 0; i < n; ++i) {
        sum += (ggml_float)x[i];
    }
    *s = (float)sum;
#else
    vDSP_sve(x, 1, s, n);
#endif
}

inline static void ggml_vec_cumsum_f32(const int n, float * y, const float * x) {
    for (int i = 0; i < n; ++i) {
        if (i == 0) {
            y[i] = x[i];
        } else {
            y[i] = y[i - 1] + x[i];
        }
    }
}

inline static void ggml_vec_sum_f32_ggf(const int n, ggml_float * s, const float * x) {
    ggml_float sum = 0.0;
    for (int i = 0; i < n; ++i) {
        sum += (ggml_float)x[i];
    }
    *s = sum;
}

inline static void ggml_vec_sum_f16_ggf(const int n, float * s, const ggml_fp16_t * x) {
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        sum += GGML_CPU_FP16_TO_FP32(x[i]);
    }
    *s = sum;
}

inline static void ggml_vec_sum_bf16_ggf(const int n, float * s, const ggml_bf16_t * x) {
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        sum += GGML_BF16_TO_FP32(x[i]);
    }
    *s = sum;
}

inline static void ggml_vec_max_f32(const int n, float * s, const float * x) {
#ifndef GGML_USE_ACCELERATE
    float max = -INFINITY;
    for (int i = 0; i < n; ++i) {
        max = MAX(max, x[i]);
    }
    *s = max;
#else
    vDSP_maxv(x, 1, s, n);
#endif
}

inline static void ggml_vec_norm_inv_f32(const int n, float * s, const float * x) {
    ggml_vec_norm_f32(n, s, x);
    *s = 1.f/(*s);
}

inline static void ggml_vec_argmax_f32(const int n, int * s, const float * x) {
    float max = -INFINITY;
    int idx = 0;
    for (int i = 0; i < n; ++i) {
        max = MAX(max, x[i]);
        if (max == x[i]) { idx = i; }
    }
    *s = idx;
}

#ifdef __cplusplus
}
#endif
