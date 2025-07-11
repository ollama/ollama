#define GGML_COMMON_IMPL_C
#include "ggml-common.h"
#include "ggml-quants.h"
#include "ggml-impl.h"
#include "ggml-cpu.h"
#include "simd-mappings.h"

#include "../../quants.h"
#include "../../ggml-cpu-impl.h"

#include <math.h>
#include <string.h>
#include <assert.h>
#include <float.h>
#include <stdlib.h> // for qsort
#include <stdio.h>  // for GGML_ASSERT

#define GROUP_MAX_EPS 1e-15f
#define GROUP_MAX_EPS_IQ3_XXS 1e-8f
#define GROUP_MAX_EPS_IQ2_S 1e-8f
#define GROUP_MAX_EPS_IQ1_M 1e-7f
#define GROUP_MAX_EPS_IQ1_S 1e-12f

#define UNUSED GGML_UNUSED

void quantize_row_q8_0(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k) {
    assert(QK8_0 == 32);
    assert(k % QK8_0 == 0);
    const int nb = k / QK8_0;

    block_q8_0 * GGML_RESTRICT y = vy;

#if defined(__riscv_v)

    size_t vl = QK8_0;

    for (int i = 0; i < nb; i++) {
        // load elements
        vfloat32m8_t v_x   = __riscv_vle32_v_f32m8(x+i*QK8_0, vl);

        vfloat32m8_t vfabs = __riscv_vfabs_v_f32m8(v_x, vl);
        vfloat32m1_t tmp   = __riscv_vfmv_v_f_f32m1(0.0f, vl);
        vfloat32m1_t vmax  = __riscv_vfredmax_vs_f32m8_f32m1(vfabs, tmp, vl);
        float amax = __riscv_vfmv_f_s_f32m1_f32(vmax);

        const float d = amax / ((1 << 7) - 1);
        const float id = d ? 1.0f/d : 0.0f;

        y[i].d = GGML_CPU_FP32_TO_FP16(d);

        vfloat32m8_t x0 = __riscv_vfmul_vf_f32m8(v_x, id, vl);

        // convert to integer
        vint16m4_t   vi = __riscv_vfncvt_x_f_w_i16m4(x0, vl);
        vint8m2_t    vs = __riscv_vncvt_x_x_w_i8m2(vi, vl);

        // store result
        __riscv_vse8_v_i8m2(y[i].qs , vs, vl);
    }
#else
    GGML_UNUSED(nb);
    // scalar
    quantize_row_q8_0_ref(x, y, k);
#endif
}

void quantize_row_q8_1(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k) {
    assert(k % QK8_1 == 0);
    const int nb = k / QK8_1;

    block_q8_1 * GGML_RESTRICT y = vy;

#if defined(__riscv_v)

    size_t vl = QK8_1;

    for (int i = 0; i < nb; i++) {
        // load elements
        vfloat32m8_t v_x   = __riscv_vle32_v_f32m8(x+i*QK8_1, vl);

        vfloat32m8_t vfabs = __riscv_vfabs_v_f32m8(v_x, vl);
        vfloat32m1_t tmp   = __riscv_vfmv_v_f_f32m1(0.0, vl);
        vfloat32m1_t vmax  = __riscv_vfredmax_vs_f32m8_f32m1(vfabs, tmp, vl);
        float amax = __riscv_vfmv_f_s_f32m1_f32(vmax);

        const float d  = amax / ((1 << 7) - 1);
        const float id = d ? 1.0f/d : 0.0f;

        y[i].d = GGML_CPU_FP32_TO_FP16(d);

        vfloat32m8_t x0 = __riscv_vfmul_vf_f32m8(v_x, id, vl);

        // convert to integer
        vint16m4_t   vi = __riscv_vfncvt_x_f_w_i16m4(x0, vl);
        vint8m2_t    vs = __riscv_vncvt_x_x_w_i8m2(vi, vl);

        // store result
        __riscv_vse8_v_i8m2(y[i].qs , vs, vl);

        // compute sum for y[i].s
        vint16m1_t tmp2 = __riscv_vmv_v_x_i16m1(0, vl);
        vint16m1_t vwrs = __riscv_vwredsum_vs_i8m2_i16m1(vs, tmp2, vl);

        // set y[i].s
        int sum = __riscv_vmv_x_s_i16m1_i16(vwrs);
        y[i].s = GGML_CPU_FP32_TO_FP16(sum*d);
    }

#else
    GGML_UNUSED(nb);
    // scalar
    quantize_row_q8_1_ref(x, y, k);
#endif
}

//===================================== Dot products =================================

void ggml_vec_dot_q4_0_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    const int qk = QK8_0;
    const int nb = n / qk;

    assert(n % qk == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_q4_0 * GGML_RESTRICT x = vx;
    const block_q8_0 * GGML_RESTRICT y = vy;

    int ib = 0;
    float sumf = 0;

#if defined(__riscv_v)
    size_t vl = qk / 2;

    for (; ib < nb; ++ib) {
        // load elements
        vuint8m1_t tx = __riscv_vle8_v_u8m1(x[ib].qs, vl);

        vint8m1_t y0 = __riscv_vle8_v_i8m1(y[ib].qs, vl);
        vint8m1_t y1 = __riscv_vle8_v_i8m1(y[ib].qs+16, vl);

        // mask and store lower part of x, and then upper part
        vuint8m1_t x_a = __riscv_vand_vx_u8m1(tx, 0x0F, vl);
        vuint8m1_t x_l = __riscv_vsrl_vx_u8m1(tx, 0x04, vl);

        vint8m1_t x_ai = __riscv_vreinterpret_v_u8m1_i8m1(x_a);
        vint8m1_t x_li = __riscv_vreinterpret_v_u8m1_i8m1(x_l);

        // subtract offset
        vint8m1_t v0 = __riscv_vsub_vx_i8m1(x_ai, 8, vl);
        vint8m1_t v1 = __riscv_vsub_vx_i8m1(x_li, 8, vl);

        vint16m2_t vec_mul1 = __riscv_vwmul_vv_i16m2(v0, y0, vl);
        vint16m2_t vec_mul2 = __riscv_vwmacc_vv_i16m2(vec_mul1, v1, y1, vl);

        vint32m1_t vec_zero = __riscv_vmv_v_x_i32m1(0, vl);
        vint32m1_t vs2 = __riscv_vwredsum_vs_i16m2_i32m1(vec_mul2, vec_zero, vl);

        int sumi = __riscv_vmv_x_s_i32m1_i32(vs2);

        sumf += sumi*GGML_CPU_FP16_TO_FP32(x[ib].d)*GGML_CPU_FP16_TO_FP32(y[ib].d);
    }

#endif
    for (; ib < nb; ++ib) {
        int sumi0 = 0;
        int sumi1 = 0;

        for (int j = 0; j < qk/2; ++j) {
            const int v0 = (x[ib].qs[j] & 0x0F) - 8;
            const int v1 = (x[ib].qs[j] >>   4) - 8;

            sumi0 += (v0 * y[ib].qs[j]);
            sumi1 += (v1 * y[ib].qs[j + qk/2]);
        }

        int sumi = sumi0 + sumi1;
        sumf += sumi*GGML_CPU_FP16_TO_FP32(x[ib].d)*GGML_CPU_FP16_TO_FP32(y[ib].d);
    }

    *s = sumf;
}

void ggml_vec_dot_q4_1_q8_1(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    const int qk = QK8_1;
    const int nb = n / qk;

    assert(n % qk == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_q4_1 * GGML_RESTRICT x = vx;
    const block_q8_1 * GGML_RESTRICT y = vy;

    int ib = 0;
    float sumf = 0;

#if defined(__riscv_v)
    size_t vl = qk / 2;

    for (; ib < nb; ++ib) {
        // load elements
        vuint8m1_t tx = __riscv_vle8_v_u8m1(x[ib].qs, vl);

        vint8m1_t y0 = __riscv_vle8_v_i8m1(y[ib].qs, vl);
        vint8m1_t y1 = __riscv_vle8_v_i8m1(y[ib].qs+16, vl);

        // mask and store lower part of x, and then upper part
        vuint8m1_t x_a = __riscv_vand_vx_u8m1(tx, 0x0F, vl);
        vuint8m1_t x_l = __riscv_vsrl_vx_u8m1(tx, 0x04, vl);

        vint8m1_t v0 = __riscv_vreinterpret_v_u8m1_i8m1(x_a);
        vint8m1_t v1 = __riscv_vreinterpret_v_u8m1_i8m1(x_l);

        vint16m2_t vec_mul1 = __riscv_vwmul_vv_i16m2(v0, y0, vl);
        vint16m2_t vec_mul2 = __riscv_vwmacc_vv_i16m2(vec_mul1, v1, y1, vl);

        vint32m1_t vec_zero = __riscv_vmv_v_x_i32m1(0, vl);
        vint32m1_t vs2 = __riscv_vwredsum_vs_i16m2_i32m1(vec_mul2, vec_zero, vl);

        int sumi = __riscv_vmv_x_s_i32m1_i32(vs2);

        sumf += (GGML_CPU_FP16_TO_FP32(x[ib].d)*GGML_CPU_FP16_TO_FP32(y[ib].d))*sumi + GGML_CPU_FP16_TO_FP32(x[ib].m)*GGML_CPU_FP16_TO_FP32(y[ib].s);
    }

#endif
    for (; ib < nb; ++ib) {
        int sumi0 = 0;
        int sumi1 = 0;

        for (int j = 0; j < qk/2; ++j) {
            const int v0 = (x[ib].qs[j] & 0x0F);
            const int v1 = (x[ib].qs[j] >>   4);

            sumi0 += (v0 * y[ib].qs[j]);
            sumi1 += (v1 * y[ib].qs[j + qk/2]);
        }

        int sumi = sumi0 + sumi1;
        sumf += (GGML_CPU_FP16_TO_FP32(x[ib].d)*GGML_CPU_FP16_TO_FP32(y[ib].d))*sumi + GGML_CPU_FP16_TO_FP32(x[ib].m)*GGML_CPU_FP16_TO_FP32(y[ib].s);
    }

    *s = sumf;
}

void ggml_vec_dot_q5_0_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    const int qk = QK8_0;
    const int nb = n / qk;

    int ib = 0;
    float sumf = 0;

    assert(n % qk == 0);
    assert(qk == QK5_0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_q5_0 * GGML_RESTRICT x = vx;
    const block_q8_0 * GGML_RESTRICT y = vy;

#if defined(__riscv_v)
    size_t vl;
    size_t vlenb = __riscv_vlenb();

    for (; ib < nb; ++ib) {
        vl = qk / 2;
        vuint8m1_t v0 = __riscv_vle8_v_u8m1(x[ib].qs, vl);
        vint8m1_t v0l = __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vand_vx_u8m1(v0, 0x0F, vl));
        vint8m1_t v0h = __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vsrl_vx_u8m1(v0, 4, vl));
        vint8m2_t v0c;
        if (vlenb == 16) {
            v0c = __riscv_vcreate_v_i8m1_i8m2(v0l, v0h);
        } else {
            v0l = __riscv_vslideup_vx_i8m1(v0l, v0h, 16, 32);
            v0c = __riscv_vlmul_ext_v_i8m1_i8m2(v0l);
        }

        vl = qk;
        vbool4_t qh = __riscv_vlm_v_b4(x[ib].qh, vl);
        qh = __riscv_vmnand_mm_b4(qh, qh, vl);
        vint8m2_t v0f = __riscv_vsub_vx_i8m2_mu(qh, v0c, v0c, 0x10, vl);
        vint8m2_t v1 = __riscv_vle8_v_i8m2(y[ib].qs, vl);
        vint16m4_t mul = __riscv_vwmul_vv_i16m4(v0f, v1, vl);
        vint32m1_t zero = __riscv_vmv_v_x_i32m1(0, vl);
        vint32m1_t sum = __riscv_vwredsum_vs_i16m4_i32m1(mul, zero, vl);
        int32_t sumi = __riscv_vmv_x_s_i32m1_i32(sum);

        sumf += (GGML_CPU_FP16_TO_FP32(x[ib].d) * GGML_CPU_FP16_TO_FP32(y[ib].d)) * sumi;
    }

#endif
    for (; ib < nb; ++ib) {
        uint32_t qh;
        memcpy(&qh, x[ib].qh, sizeof(qh));

        int sumi0 = 0;
        int sumi1 = 0;

        for (int j = 0; j < qk/2; ++j) {
            const uint8_t xh_0 = ((qh & (1u << (j + 0 ))) >> (j + 0 )) << 4;
            const uint8_t xh_1 = ((qh & (1u << (j + 16))) >> (j + 12));

            const int32_t x0 = (int8_t)(((x[ib].qs[j] & 0x0F) | xh_0) - 16);
            const int32_t x1 = (int8_t)(((x[ib].qs[j] >>   4) | xh_1) - 16);

            sumi0 += (x0 * y[ib].qs[j]);
            sumi1 += (x1 * y[ib].qs[j + qk/2]);
        }

        int sumi = sumi0 + sumi1;
        sumf += (GGML_CPU_FP16_TO_FP32(x[ib].d)*GGML_CPU_FP16_TO_FP32(y[ib].d)) * sumi;
    }

    *s = sumf;
}

void ggml_vec_dot_q5_1_q8_1(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    const int qk = QK8_1;
    const int nb = n / qk;

    int ib = 0;
    float sumf = 0;

    assert(n % qk == 0);
    assert(qk == QK5_1);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_q5_1 * GGML_RESTRICT x = vx;
    const block_q8_1 * GGML_RESTRICT y = vy;

#if defined(__riscv_v)
    size_t vl;
    size_t vlenb = __riscv_vlenb();

    for (; ib < nb; ++ib) {
        vl = qk / 2;
        vuint8m1_t v0 = __riscv_vle8_v_u8m1(x[ib].qs, vl);
        vint8m1_t v0l = __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vand_vx_u8m1(v0, 0x0F, vl));
        vint8m1_t v0h = __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vsrl_vx_u8m1(v0, 4, vl));
        vint8m2_t v0c;
        if (vlenb == 16) {
            v0c = __riscv_vcreate_v_i8m1_i8m2(v0l, v0h);
        } else {
            v0l = __riscv_vslideup_vx_i8m1(v0l, v0h, 16, 32);
            v0c = __riscv_vlmul_ext_v_i8m1_i8m2(v0l);
        }

        vl = qk;
        vbool4_t qh = __riscv_vlm_v_b4(x[ib].qh, vl);
        vint8m2_t v0f = __riscv_vor_vx_i8m2_mu(qh, v0c, v0c, 0x10, vl);
        vint8m2_t v1 = __riscv_vle8_v_i8m2(y[ib].qs, vl);
        vint16m4_t mul = __riscv_vwmul_vv_i16m4(v0f, v1, vl);
        vint32m1_t zero = __riscv_vmv_v_x_i32m1(0, vl);
        vint32m1_t sum = __riscv_vwredsum_vs_i16m4_i32m1(mul, zero, vl);
        int32_t sumi = __riscv_vmv_x_s_i32m1_i32(sum);

        sumf += (GGML_CPU_FP16_TO_FP32(x[ib].d)*GGML_CPU_FP16_TO_FP32(y[ib].d))*sumi + GGML_CPU_FP16_TO_FP32(x[ib].m)*GGML_CPU_FP16_TO_FP32(y[ib].s);
    }

#endif
    for (; ib < nb; ++ib) {
        uint32_t qh;
        memcpy(&qh, x[ib].qh, sizeof(qh));

        int sumi0 = 0;
        int sumi1 = 0;

        for (int j = 0; j < qk/2; ++j) {
            const uint8_t xh_0 = ((qh >> (j +  0)) << 4) & 0x10;
            const uint8_t xh_1 = ((qh >> (j + 12))     ) & 0x10;

            const int32_t x0 = (x[ib].qs[j] & 0xF) | xh_0;
            const int32_t x1 = (x[ib].qs[j] >>  4) | xh_1;

            sumi0 += (x0 * y[ib].qs[j]);
            sumi1 += (x1 * y[ib].qs[j + qk/2]);
        }

        int sumi = sumi0 + sumi1;
        sumf += (GGML_CPU_FP16_TO_FP32(x[ib].d)*GGML_CPU_FP16_TO_FP32(y[ib].d))*sumi + GGML_CPU_FP16_TO_FP32(x[ib].m)*GGML_CPU_FP16_TO_FP32(y[ib].s);
    }

    *s = sumf;
}

void ggml_vec_dot_q8_0_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    const int qk = QK8_0;
    const int nb = n / qk;

    assert(n % qk == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_q8_0 * GGML_RESTRICT x = vx;
    const block_q8_0 * GGML_RESTRICT y = vy;

    int ib = 0;
    float sumf = 0;

#if defined(__riscv_v)
    size_t vl = qk;

    for (; ib < nb; ++ib) {
        // load elements
        vint8m2_t bx_0 = __riscv_vle8_v_i8m2(x[ib].qs, vl);
        vint8m2_t by_0 = __riscv_vle8_v_i8m2(y[ib].qs, vl);

        vint16m4_t vw_mul = __riscv_vwmul_vv_i16m4(bx_0, by_0, vl);

        vint32m1_t v_zero = __riscv_vmv_v_x_i32m1(0, vl);
        vint32m1_t v_sum = __riscv_vwredsum_vs_i16m4_i32m1(vw_mul, v_zero, vl);

        int sumi = __riscv_vmv_x_s_i32m1_i32(v_sum);

        sumf += sumi*(GGML_CPU_FP16_TO_FP32(x[ib].d)*GGML_CPU_FP16_TO_FP32(y[ib].d));
    }

#endif
    for (; ib < nb; ++ib) {
        int sumi = 0;

        for (int j = 0; j < qk; j++) {
            sumi += x[ib].qs[j]*y[ib].qs[j];
        }

        sumf += sumi*(GGML_CPU_FP16_TO_FP32(x[ib].d)*GGML_CPU_FP16_TO_FP32(y[ib].d));
    }

    *s = sumf;
}

void ggml_vec_dot_q2_K_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_q2_K * GGML_RESTRICT x = vx;
    const block_q8_K * GGML_RESTRICT y = vy;

    const int nb = n / QK_K;

#if defined __riscv_xtheadvector

    float sumf = 0;
    uint8_t atmp[16];

    for (int i = 0; i < nb; ++i) {
        const uint8_t * q2 = x[i].qs;
        const  int8_t * q8 = y[i].qs;
        const uint8_t * sc = x[i].scales;
        const float dall = y[i].d * GGML_CPU_FP16_TO_FP32(x[i].d);
        const float dmin = -y[i].d * GGML_CPU_FP16_TO_FP32(x[i].dmin);
        uint8_t *patmp = atmp;
        int vsums;
        int tmp;
        __asm__ __volatile__(
            "th.vsetvli zero, %[vl16], e8, m1\n\t"
            "th.vmv.v.x v8, zero\n\t"
            "th.vlb.v v1, (%[sc])\n\t"
            "th.vand.vi v0, v1, 0xF\n\t"
            "th.vsrl.vi v1, v1, 4\n\t"
            "th.vsb.v v0, (%[scale])\n\t"
            "th.vwaddu.vx v16, v1, zero\n\t"
            "th.vsetvli zero, %[vl16], e16, m2\n\t"
            "th.vlh.v v2, (%[bsums])\n\t"
            "th.vwmul.vv v4, v16, v2\n\t"
            "th.vsetvli zero, %[vl16], e32, m4\n\t"
            "th.vredsum.vs v8, v4, v8\n\t"
            "th.vmv.x.s %[vsums], v8"
            : [tmp] "=&r" (tmp), [vsums] "=&r" (vsums)
            : [sc] "r" (sc), [scale] "r" (atmp), [bsums] "r" (y[i].bsums)
            , [vl16] "r" (16)
            : "memory"
            , "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7"
            , "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15"
            , "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23"
            , "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31"
        );
        sumf += dmin * vsums;
        int isum = 0;

        for (int j = 0; j < QK_K/128; ++j) {
            __asm__ __volatile__(
                "th.vsetvli zero, %[vl32], e8, m2\n\t"
                "th.vlb.v v0, (%[q2])\n\t"
                "th.vsrl.vi v2, v0, 2\n\t"
                "th.vsrl.vi v4, v0, 4\n\t"
                "th.vsrl.vi v6, v0, 6\n\t"
                "th.vand.vi v0, v0, 0x3\n\t"
                "th.vand.vi v2, v2, 0x3\n\t"
                "th.vand.vi v4, v4, 0x3\n\t"
                "th.vsetvli zero, %[vl128], e8, m8\n\t"
                "th.vlb.v v8, (%[q8])\n\t"
                "th.vsetvli zero, %[vl64], e8, m4\n\t"
                "th.vwmul.vv v16, v0, v8\n\t"
                "th.vwmul.vv v24, v4, v12\n\t"
                "th.vsetvli zero, %[vl16], e16, m2\n\t"
                "th.vmv.v.x v0, zero\n\t"
                "th.vwredsum.vs v10, v16, v0\n\t"
                "th.vwredsum.vs v9, v18, v0\n\t"
                "th.vwredsum.vs v8, v20, v0\n\t"
                "th.vwredsum.vs v7, v22, v0\n\t"
                "th.vwredsum.vs v11, v24, v0\n\t"
                "th.vwredsum.vs v12, v26, v0\n\t"
                "th.vwredsum.vs v13, v28, v0\n\t"
                "th.vwredsum.vs v14, v30, v0\n\t"
                "li %[tmp], 4\n\t"
                "th.vsetvli zero, %[tmp], e32, m1\n\t"
                "th.vslideup.vi v10, v9, 1\n\t"
                "th.vslideup.vi v8, v7, 1\n\t"
                "th.vslideup.vi v11, v12, 1\n\t"
                "th.vslideup.vi v13, v14, 1\n\t"
                "th.vslideup.vi v10, v8, 2\n\t"
                "th.vslideup.vi v11, v13, 2\n\t"
                "li %[tmp], 8\n\t"
                "th.vsetvli zero, %[tmp], e32, m2\n\t"
                "th.vlbu.v v12, (%[scale])\n\t"
                "th.vmul.vv v10, v10, v12\n\t"
                "th.vredsum.vs v0, v10, v0\n\t"
                "th.vmv.x.s %[tmp], v0\n\t"
                "add %[isum], %[isum], %[tmp]"
                : [tmp] "=&r" (tmp), [isum] "+&r" (isum)
                : [q2] "r" (q2), [scale] "r" (patmp), [q8] "r" (q8)
                , [vl16] "r" (16), [vl32] "r" (32), [vl64] "r" (64), [vl128] "r" (128)
                : "memory"
                , "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7"
                , "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15"
                , "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23"
                , "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31"
            );
            q2 += 32; q8 += 128; patmp += 8;
        }

        sumf += dall * isum;
    }

    *s = sumf;

#elif defined __riscv_v

    float sumf = 0;
    uint8_t atmp[16];

    const int vector_length = __riscv_vlenb() * 8;
    uint8_t temp_01[32] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };

    switch (vector_length) {
    case 256:
        for (int i = 0; i < nb; ++i) {
            const uint8_t * q2 = x[i].qs;
            const int8_t *  q8 = y[i].qs;
            const uint8_t * sc = x[i].scales;

            const float dall = y[i].d * GGML_CPU_FP16_TO_FP32(x[i].d);
            const float dmin = -y[i].d * GGML_CPU_FP16_TO_FP32(x[i].dmin);

            size_t vl = 16;

            vuint8m1_t scales = __riscv_vle8_v_u8m1(sc, vl);
            vuint8m1_t aux    = __riscv_vand_vx_u8m1(scales, 0x0F, vl);

            vint16m1_t q8sums = __riscv_vle16_v_i16m1(y[i].bsums, vl);

            vuint8mf2_t scales_2 = __riscv_vle8_v_u8mf2(sc, vl);
            vuint8mf2_t mins8    = __riscv_vsrl_vx_u8mf2(scales_2, 0x4, vl);
            vint16m1_t  mins     = __riscv_vreinterpret_v_u16m1_i16m1(__riscv_vzext_vf2_u16m1(mins8, vl));
            vint32m2_t  prod     = __riscv_vwmul_vv_i32m2(q8sums, mins, vl);
            vint32m1_t  vsums    = __riscv_vredsum_vs_i32m2_i32m1(prod, __riscv_vmv_v_x_i32m1(0, 1), vl);

            sumf += dmin * __riscv_vmv_x_s_i32m1_i32(vsums);

            vl = 32;

            vint32m1_t vzero = __riscv_vmv_v_x_i32m1(0, 1);
            vuint8m1_t v_b   = __riscv_vle8_v_u8m1(temp_01, vl);

            uint8_t is   = 0;
            int     isum = 0;

            for (int j = 0; j < QK_K / 128; ++j) {
                // load Q2
                vuint8m1_t q2_x = __riscv_vle8_v_u8m1(q2, vl);

                vuint8m1_t q2_0 = __riscv_vand_vx_u8m1(q2_x, 0x03, vl);
                vuint8m1_t q2_1 = __riscv_vand_vx_u8m1(__riscv_vsrl_vx_u8m1(q2_x, 0x2, vl), 0x03, vl);
                vuint8m1_t q2_2 = __riscv_vand_vx_u8m1(__riscv_vsrl_vx_u8m1(q2_x, 0x4, vl), 0x03, vl);
                vuint8m1_t q2_3 = __riscv_vand_vx_u8m1(__riscv_vsrl_vx_u8m1(q2_x, 0x6, vl), 0x03, vl);

                // duplicate scale elements for product
                vuint8m1_t sc0 = __riscv_vrgather_vv_u8m1(aux, __riscv_vadd_vx_u8m1(v_b, 0 + is, vl), vl);
                vuint8m1_t sc1 = __riscv_vrgather_vv_u8m1(aux, __riscv_vadd_vx_u8m1(v_b, 2 + is, vl), vl);
                vuint8m1_t sc2 = __riscv_vrgather_vv_u8m1(aux, __riscv_vadd_vx_u8m1(v_b, 4 + is, vl), vl);
                vuint8m1_t sc3 = __riscv_vrgather_vv_u8m1(aux, __riscv_vadd_vx_u8m1(v_b, 6 + is, vl), vl);

                vint16m2_t p0 = __riscv_vreinterpret_v_u16m2_i16m2(__riscv_vwmulu_vv_u16m2(q2_0, sc0, vl));
                vint16m2_t p1 = __riscv_vreinterpret_v_u16m2_i16m2(__riscv_vwmulu_vv_u16m2(q2_1, sc1, vl));
                vint16m2_t p2 = __riscv_vreinterpret_v_u16m2_i16m2(__riscv_vwmulu_vv_u16m2(q2_2, sc2, vl));
                vint16m2_t p3 = __riscv_vreinterpret_v_u16m2_i16m2(__riscv_vwmulu_vv_u16m2(q2_3, sc3, vl));

                // load Q8
                vint8m1_t q8_0 = __riscv_vle8_v_i8m1(q8, vl);
                vint8m1_t q8_1 = __riscv_vle8_v_i8m1(q8 + 32, vl);
                vint8m1_t q8_2 = __riscv_vle8_v_i8m1(q8 + 64, vl);
                vint8m1_t q8_3 = __riscv_vle8_v_i8m1(q8 + 96, vl);

                vint32m4_t s0 = __riscv_vwmul_vv_i32m4(p0, __riscv_vwcvt_x_x_v_i16m2(q8_0, vl), vl);
                vint32m4_t s1 = __riscv_vwmul_vv_i32m4(p1, __riscv_vwcvt_x_x_v_i16m2(q8_1, vl), vl);
                vint32m4_t s2 = __riscv_vwmul_vv_i32m4(p2, __riscv_vwcvt_x_x_v_i16m2(q8_2, vl), vl);
                vint32m4_t s3 = __riscv_vwmul_vv_i32m4(p3, __riscv_vwcvt_x_x_v_i16m2(q8_3, vl), vl);

                vint32m1_t isum0 = __riscv_vredsum_vs_i32m4_i32m1(__riscv_vadd_vv_i32m4(s0, s1, vl), vzero, vl);
                vint32m1_t isum1 = __riscv_vredsum_vs_i32m4_i32m1(__riscv_vadd_vv_i32m4(s2, s3, vl), isum0, vl);

                isum += __riscv_vmv_x_s_i32m1_i32(isum1);

                q2 += 32;
                q8 += 128;
                is = 8;
            }

            sumf += dall * isum;
        }
        break;
    case 128:
        for (int i = 0; i < nb; ++i) {
            const uint8_t * q2 = x[i].qs;
            const  int8_t * q8 = y[i].qs;
            const uint8_t * sc = x[i].scales;
            const float dall = y[i].d * GGML_CPU_FP16_TO_FP32(x[i].d);
            const float dmin = -y[i].d * GGML_CPU_FP16_TO_FP32(x[i].dmin);
            uint8_t *patmp = atmp;
            int vsums;
            int tmp;
            __asm__ __volatile__(
                "vsetivli zero, 16, e8, m1\n\t"
                "vmv.v.x v8, zero\n\t"
                "vle8.v v1, (%[sc])\n\t"
                "vand.vi v0, v1, 0xF\n\t"
                "vsrl.vi v1, v1, 4\n\t"
                "vse8.v v0, (%[scale])\n\t"
                "vsetivli zero, 16, e16, m2\n\t"
                "vle16.v v2, (%[bsums])\n\t"
                "vzext.vf2 v0, v1\n\t"
                "vwmul.vv v4, v0, v2\n\t"
                "vsetivli zero, 16, e32, m4\n\t"
                "vredsum.vs v8, v4, v8\n\t"
                "vmv.x.s %[vsums], v8"
                : [tmp] "=&r" (tmp), [vsums] "=&r" (vsums)
                : [sc] "r" (sc), [scale] "r" (atmp), [bsums] "r" (y[i].bsums)
                : "memory"
                , "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7"
                , "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15"
                , "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23"
                , "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31"
            );
            sumf += dmin * vsums;
            int isum = 0;

            for (int j = 0; j < QK_K/128; ++j) {
                __asm__ __volatile__(
                    "vsetvli zero, %[vl32], e8, m2\n\t"
                    "vle8.v v0, (%[q2])\n\t"
                    "vsrl.vi v2, v0, 2\n\t"
                    "vsrl.vi v4, v0, 4\n\t"
                    "vsrl.vi v6, v0, 6\n\t"
                    "vand.vi v0, v0, 0x3\n\t"
                    "vand.vi v2, v2, 0x3\n\t"
                    "vand.vi v4, v4, 0x3\n\t"
                    "vsetvli zero, %[vl128], e8, m8\n\t"
                    "vle8.v v8, (%[q8])\n\t"
                    "vsetvli zero, %[vl64], e8, m4\n\t"
                    "vwmul.vv v16, v0, v8\n\t"
                    "vwmul.vv v24, v4, v12\n\t"
                    "vsetivli zero, 16, e16, m2\n\t"
                    "vmv.v.x v0, zero\n\t"
                    "vwredsum.vs v10, v16, v0\n\t"
                    "vwredsum.vs v9, v18, v0\n\t"
                    "vwredsum.vs v8, v20, v0\n\t"
                    "vwredsum.vs v7, v22, v0\n\t"
                    "vwredsum.vs v11, v24, v0\n\t"
                    "vwredsum.vs v12, v26, v0\n\t"
                    "vwredsum.vs v13, v28, v0\n\t"
                    "vwredsum.vs v14, v30, v0\n\t"
                    "vsetivli zero, 4, e32, m1\n\t"
                    "vslideup.vi v10, v9, 1\n\t"
                    "vslideup.vi v8, v7, 1\n\t"
                    "vslideup.vi v11, v12, 1\n\t"
                    "vslideup.vi v13, v14, 1\n\t"
                    "vslideup.vi v10, v8, 2\n\t"
                    "vslideup.vi v11, v13, 2\n\t"
                    "vsetivli zero, 8, e32, m2\n\t"
                    "vle8.v v15, (%[scale])\n\t"
                    "vzext.vf4 v12, v15\n\t"
                    "vmul.vv v10, v10, v12\n\t"
                    "vredsum.vs v0, v10, v0\n\t"
                    "vmv.x.s %[tmp], v0\n\t"
                    "add %[isum], %[isum], %[tmp]"
                    : [tmp] "=&r" (tmp), [isum] "+&r" (isum)
                    : [q2] "r" (q2), [scale] "r" (patmp), [q8] "r" (q8)
                    , [vl32] "r" (32), [vl64] "r" (64), [vl128] "r" (128)
                    : "memory"
                    , "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7"
                    , "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15"
                    , "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23"
                    , "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31"
                );
                q2 += 32; q8 += 128; patmp += 8;
            }

            sumf += dall * isum;
        }
        break;
    default:
        assert(false && "Unsupported vector length");
        break;
    }

    *s = sumf;

#else

    float sumf = 0;

    for (int i = 0; i < nb; ++i) {

        const uint8_t * q2 = x[i].qs;
        const  int8_t * q8 = y[i].qs;
        const uint8_t * sc = x[i].scales;

        int summs = 0;
        for (int j = 0; j < 16; ++j) {
            summs += y[i].bsums[j] * (sc[j] >> 4);
        }

        const float dall = y[i].d * GGML_CPU_FP16_TO_FP32(x[i].d);
        const float dmin = y[i].d * GGML_CPU_FP16_TO_FP32(x[i].dmin);

        int isum = 0;
        int is = 0;
        int d;
        for (int k = 0; k < QK_K/128; ++k) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {
                d = sc[is++] & 0xF;
                int isuml = 0;
                for (int l =  0; l < 16; ++l) isuml += q8[l] * ((q2[l] >> shift) & 3);
                isum += d * isuml;
                d = sc[is++] & 0xF;
                isuml = 0;
                for (int l = 16; l < 32; ++l) isuml += q8[l] * ((q2[l] >> shift) & 3);
                isum += d * isuml;
                shift += 2;
                q8 += 32;
            }
            q2 += 32;
        }
        sumf += dall * isum - dmin * summs;
    }
    *s = sumf;
#endif
}

void ggml_vec_dot_q3_K_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const uint32_t kmask1 = 0x03030303;
    const uint32_t kmask2 = 0x0f0f0f0f;

    const block_q3_K * GGML_RESTRICT x = vx;
    const block_q8_K * GGML_RESTRICT y = vy;

    const int nb = n / QK_K;

#if defined __riscv_xtheadvector

    uint32_t utmp[4];
    float sumf = 0;

    for (int i = 0; i < nb; ++i) {
        const uint8_t * restrict q3 = x[i].qs;
        const uint8_t * restrict qh = x[i].hmask;
        const  int8_t * restrict q8 = y[i].qs;

        int8_t * scale = (int8_t *)utmp;
        int tmp;
        __asm__ __volatile__(
            "li %[tmp], 12\n\t"
            "th.vsetvli zero, %[tmp], e8, m1\n\t"
            "th.vlb.v v0, (%[s6b])\n\t"
            "th.vmv.v.v v2, v0\n\t"
            "li %[tmp], 2\n\t"
            "th.vsetvli zero, %[tmp], e64, m1\n\t"
            "th.vmv.v.x v9, %[sh]\n\t"\
            "th.vslidedown.vi v1, v0, 1\n\t"
            "th.vslide1up.vx v8, v9, zero\n\t" // {0, 0, 4, 4}
            "th.vslideup.vi v0, v2, 1\n\t" // {aux[0], aux[1], aux[0], aux[1]}
            "li %[tmp], 4\n\t"
            "th.vsetvli zero, %[tmp], e32, m1\n\t"
            "th.vid.v v9\n\t"
            "th.vmv.x.s %[tmp], v1\n\t"
            "th.vsll.vi v9, v9, 1\n\t" // {0, 2, 4, 6}
            "th.vmv.v.x v1, %[tmp]\n\t" // {aux[2], aux[2], aux[2], aux[2]}
            "th.vsrl.vv v4, v1, v9\n\t"
            "th.vsrl.vv v2, v0, v8\n\t"
            "th.vand.vx v5, v4, %[kmask1]\n\t"
            "th.vand.vx v3, v2, %[kmask2]\n\t"
            "th.vsll.vi v6, v5, 4\n\t"
            "th.vor.vv v7, v6, v3\n\t"
            "li %[tmp], 16\n\t"
            "th.vsetvli zero, %[tmp], e8, m1\n\t"
            "th.vsub.vx v0, v7, %[c]\n\t"
            "th.vsb.v v0, (%[scale])"
            : [tmp] "=&r" (tmp)
            : [sh] "r" (0x0000000400000004), [s6b] "r" (x[i].scales), [c] "r" (32)
            , [scale] "r" (scale), [kmask1] "r" (kmask1), [kmask2] "r" (kmask2)
            : "memory"
            , "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7"
            , "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15"
            , "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23"
            , "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31"
        );

        uint8_t m = 1;
        int isum = 0;
        for (int j = 0; j < QK_K; j += 128) {
            __asm__ __volatile__(
                // fixme: use v0p7 mask layout directly
                "th.vsetvli zero, %[vl32], e8, m2\n\t"
                "th.vlb.v v8, (%[q3])\n\t"
                "th.vsrl.vi v10, v8, 2\n\t"
                "th.vsrl.vi v12, v8, 4\n\t"
                "th.vsrl.vi v14, v8, 6\n\t"
                "th.vand.vi v8, v8, 3\n\t"
                "th.vand.vi v10, v10, 3\n\t"
                "th.vand.vi v12, v12, 3\n\t"
                "th.vlb.v v2, (%[qh])\n\t"
                "th.vand.vx v4, v2, %[m]\n\t"
                "slli %[m], %[m], 1\n\t"
                "th.vmseq.vx v0, v4, zero\n\t"
                "th.vadd.vi v8, v8, -4, v0.t\n\t"
                "th.vand.vx v4, v2, %[m]\n\t"
                "slli %[m], %[m], 1\n\t"
                "th.vmseq.vx v0, v4, zero\n\t"
                "th.vadd.vi v10, v10, -4, v0.t\n\t"
                "th.vand.vx v4, v2, %[m]\n\t"
                "slli %[m], %[m], 1\n\t"
                "th.vmseq.vx v0, v4, zero\n\t"
                "th.vadd.vi v12, v12, -4, v0.t\n\t"
                "th.vand.vx v4, v2, %[m]\n\t"
                "slli %[m], %[m], 1\n\t"
                "th.vmseq.vx v0, v4, zero\n\t"
                "th.vadd.vi v14, v14, -4, v0.t\n\t"
                "th.vsetvli zero, %[vl128], e8, m8\n\t"
                "th.vlb.v v0, (%[q8])\n\t"
                "th.vsetvli zero, %[vl64], e8, m4\n\t"
                "th.vwmul.vv v16, v0, v8\n\t"
                "th.vwmul.vv v24, v4, v12\n\t"
                "li %[tmp], 16\n\t"
                "th.vsetvli zero, %[tmp], e16, m2\n\t"
                "th.vmv.v.x v0, zero\n\t"
                "th.vwredsum.vs v10, v16, v0\n\t"
                "th.vwredsum.vs v9, v18, v0\n\t"
                "th.vwredsum.vs v8, v20, v0\n\t"
                "th.vwredsum.vs v7, v22, v0\n\t"
                "th.vwredsum.vs v11, v24, v0\n\t"
                "th.vwredsum.vs v12, v26, v0\n\t"
                "th.vwredsum.vs v13, v28, v0\n\t"
                "th.vwredsum.vs v14, v30, v0\n\t"
                "li %[tmp], 4\n\t"
                "th.vsetvli zero, %[tmp], e32, m1\n\t"
                "th.vslideup.vi v10, v9, 1\n\t"
                "th.vslideup.vi v8, v7, 1\n\t"
                "th.vslideup.vi v11, v12, 1\n\t"
                "th.vslideup.vi v13, v14, 1\n\t"
                "th.vslideup.vi v10, v8, 2\n\t"
                "th.vslideup.vi v11, v13, 2\n\t"
                "li %[tmp], 8\n\t"
                "th.vsetvli zero, %[tmp], e32, m2\n\t"
                "th.vlb.v v12, (%[scale])\n\t"
                "th.vmul.vv v10, v10, v12\n\t"
                "th.vredsum.vs v0, v10, v0\n\t"
                "th.vmv.x.s %[tmp], v0\n\t"
                "add %[isum], %[isum], %[tmp]"
                : [tmp] "=&r" (tmp), [m] "+&r" (m), [isum] "+&r" (isum)
                : [vl128] "r" (128), [vl64] "r" (64), [vl32] "r" (32)
                , [q3] "r" (q3), [qh] "r" (qh), [scale] "r" (scale), [q8] "r" (q8)
                : "memory"
                , "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7"
                , "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15"
                , "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23"
                , "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31"
            );
            q3 += 32;    q8 += 128;   scale += 8;
        }

        const float d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;
        sumf += d * isum;
    }

    *s = sumf;

#elif defined __riscv_v

    uint32_t utmp[4];
    float sumf = 0;
    uint32_t aux[3];
    const int vector_length = __riscv_vlenb() * 8;

    switch (vector_length) {
    case 256:
        for (int i = 0; i < nb; ++i) {

            const uint8_t * GGML_RESTRICT q3 = x[i].qs;
            const uint8_t * GGML_RESTRICT qh = x[i].hmask;
            const  int8_t * GGML_RESTRICT q8 = y[i].qs;

            memcpy(aux, x[i].scales, 12);
            utmp[3] = ((aux[1] >> 4) & kmask2) | (((aux[2] >> 6) & kmask1) << 4);
            utmp[2] = ((aux[0] >> 4) & kmask2) | (((aux[2] >> 4) & kmask1) << 4);
            utmp[1] = (aux[1] & kmask2) | (((aux[2] >> 2) & kmask1) << 4);
            utmp[0] = (aux[0] & kmask2) | (((aux[2] >> 0) & kmask1) << 4);

            int8_t * scale = (int8_t *)utmp;
            for (int j = 0; j < 16; ++j) scale[j] -= 32;


            size_t vl = 32;
            uint8_t m =  1;

            vint32m1_t vzero = __riscv_vmv_v_x_i32m1(0, 1);
            vuint8m1_t vqh = __riscv_vle8_v_u8m1(qh, vl);

            int sum_t = 0;

            for (int j = 0; j < QK_K; j += 128) {

                vl = 32;

                // load Q3
                vuint8m1_t q3_x = __riscv_vle8_v_u8m1(q3, vl);

                vint8m1_t q3_0 = __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vand_vx_u8m1(q3_x, 0x03, vl));
                vint8m1_t q3_1 = __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vand_vx_u8m1(__riscv_vsrl_vx_u8m1(q3_x, 0x2, vl), 0x03 , vl));
                vint8m1_t q3_2 = __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vand_vx_u8m1(__riscv_vsrl_vx_u8m1(q3_x, 0x4, vl), 0x03 , vl));
                vint8m1_t q3_3 = __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vand_vx_u8m1(__riscv_vsrl_vx_u8m1(q3_x, 0x6, vl), 0x03 , vl));

                // compute mask for subtraction
                vuint8m1_t qh_m0 = __riscv_vand_vx_u8m1(vqh, m, vl);
                vbool8_t vmask_0 = __riscv_vmseq_vx_u8m1_b8(qh_m0, 0, vl);
                vint8m1_t q3_m0 = __riscv_vsub_vx_i8m1_mu(vmask_0, q3_0, q3_0, 0x4, vl);
                m <<= 1;

                vuint8m1_t qh_m1 = __riscv_vand_vx_u8m1(vqh, m, vl);
                vbool8_t vmask_1 = __riscv_vmseq_vx_u8m1_b8(qh_m1, 0, vl);
                vint8m1_t q3_m1 = __riscv_vsub_vx_i8m1_mu(vmask_1, q3_1, q3_1, 0x4, vl);
                m <<= 1;

                vuint8m1_t qh_m2 = __riscv_vand_vx_u8m1(vqh, m, vl);
                vbool8_t vmask_2 = __riscv_vmseq_vx_u8m1_b8(qh_m2, 0, vl);
                vint8m1_t q3_m2 = __riscv_vsub_vx_i8m1_mu(vmask_2, q3_2, q3_2, 0x4, vl);
                m <<= 1;

                vuint8m1_t qh_m3 = __riscv_vand_vx_u8m1(vqh, m, vl);
                vbool8_t vmask_3 = __riscv_vmseq_vx_u8m1_b8(qh_m3, 0, vl);
                vint8m1_t q3_m3 = __riscv_vsub_vx_i8m1_mu(vmask_3, q3_3, q3_3, 0x4, vl);
                m <<= 1;

                // load Q8 and take product with Q3
                vint16m2_t a0 = __riscv_vwmul_vv_i16m2(q3_m0, __riscv_vle8_v_i8m1(q8, vl), vl);
                vint16m2_t a1 = __riscv_vwmul_vv_i16m2(q3_m1, __riscv_vle8_v_i8m1(q8+32, vl), vl);
                vint16m2_t a2 = __riscv_vwmul_vv_i16m2(q3_m2, __riscv_vle8_v_i8m1(q8+64, vl), vl);
                vint16m2_t a3 = __riscv_vwmul_vv_i16m2(q3_m3, __riscv_vle8_v_i8m1(q8+96, vl), vl);

                vl = 16;

                // retrieve lane to multiply with scale
                vint32m2_t aux0_0 = __riscv_vwmul_vx_i32m2(__riscv_vget_v_i16m2_i16m1(a0, 0), (scale[0]), vl);
                vint32m2_t aux0_1 = __riscv_vwmul_vx_i32m2(__riscv_vget_v_i16m2_i16m1(a0, 1), (scale[1]), vl);
                vint32m2_t aux1_0 = __riscv_vwmul_vx_i32m2(__riscv_vget_v_i16m2_i16m1(a1, 0), (scale[2]), vl);
                vint32m2_t aux1_1 = __riscv_vwmul_vx_i32m2(__riscv_vget_v_i16m2_i16m1(a1, 1), (scale[3]), vl);
                vint32m2_t aux2_0 = __riscv_vwmul_vx_i32m2(__riscv_vget_v_i16m2_i16m1(a2, 0), (scale[4]), vl);
                vint32m2_t aux2_1 = __riscv_vwmul_vx_i32m2(__riscv_vget_v_i16m2_i16m1(a2, 1), (scale[5]), vl);
                vint32m2_t aux3_0 = __riscv_vwmul_vx_i32m2(__riscv_vget_v_i16m2_i16m1(a3, 0), (scale[6]), vl);
                vint32m2_t aux3_1 = __riscv_vwmul_vx_i32m2(__riscv_vget_v_i16m2_i16m1(a3, 1), (scale[7]), vl);

                vint32m1_t isum0 = __riscv_vredsum_vs_i32m2_i32m1(__riscv_vadd_vv_i32m2(aux0_0, aux0_1, vl), vzero, vl);
                vint32m1_t isum1 = __riscv_vredsum_vs_i32m2_i32m1(__riscv_vadd_vv_i32m2(aux1_0, aux1_1, vl), isum0, vl);
                vint32m1_t isum2 = __riscv_vredsum_vs_i32m2_i32m1(__riscv_vadd_vv_i32m2(aux2_0, aux2_1, vl), isum1, vl);
                vint32m1_t isum3 = __riscv_vredsum_vs_i32m2_i32m1(__riscv_vadd_vv_i32m2(aux3_0, aux3_1, vl), isum2, vl);

                sum_t +=  __riscv_vmv_x_s_i32m1_i32(isum3);

                q3 += 32;    q8 += 128;   scale += 8;

            }

            const float d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;

            sumf += d*sum_t;

        }
        break;
    case 128:
        for (int i = 0; i < nb; ++i) {
            const uint8_t * restrict q3 = x[i].qs;
            const uint8_t * restrict qh = x[i].hmask;
            const  int8_t * restrict q8 = y[i].qs;

            int8_t * scale = (int8_t *)utmp;
            int tmp;
            __asm__ __volatile__(
                "vsetivli zero, 12, e8, m1\n\t"
                "vle8.v v0, (%[s6b])\n\t"
                "vmv1r.v v2, v0\n\t"
                "vsetivli zero, 2, e64, m1\n\t"
                "vmv.v.x v9, %[sh]\n\t"\
                "vslidedown.vi v1, v0, 1\n\t"
                "vslide1up.vx v8, v9, zero\n\t" // {0, 0, 4, 4}
                "vslideup.vi v0, v2, 1\n\t" // {aux[0], aux[1], aux[0], aux[1]}
                "vsetivli zero, 4, e32, m1\n\t"
                "vid.v v9\n\t"
                "vmv.x.s %[tmp], v1\n\t"
                "vsll.vi v9, v9, 1\n\t" // {0, 2, 4, 6}
                "vmv.v.x v1, %[tmp]\n\t" // {aux[2], aux[2], aux[2], aux[2]}
                "vsrl.vv v4, v1, v9\n\t"
                "vsrl.vv v2, v0, v8\n\t"
                "vand.vx v5, v4, %[kmask1]\n\t"
                "vand.vx v3, v2, %[kmask2]\n\t"
                "vsll.vi v6, v5, 4\n\t"
                "vor.vv v7, v6, v3\n\t"
                "vsetivli zero, 16, e8, m1\n\t"
                "vsub.vx v0, v7, %[c]\n\t"
                "vse8.v v0, (%[scale])"
                : [tmp] "=&r" (tmp)
                : [sh] "r" (0x0000000400000004), [s6b] "r" (x[i].scales), [c] "r" (32)
                , [scale] "r" (scale), [kmask1] "r" (kmask1), [kmask2] "r" (kmask2)
                : "memory"
                , "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7"
                , "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15"
                , "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23"
                , "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31"
            );

            uint8_t m = 1;
            int isum = 0;
            for (int j = 0; j < QK_K; j += 128) {
                __asm__ __volatile__(
                    "vsetvli zero, %[vl32], e8, m2, ta, mu\n\t"
                    "vle8.v v8, (%[q3])\n\t"
                    "vsrl.vi v10, v8, 2\n\t"
                    "vsrl.vi v12, v8, 4\n\t"
                    "vsrl.vi v14, v8, 6\n\t"
                    "vand.vi v8, v8, 3\n\t"
                    "vand.vi v10, v10, 3\n\t"
                    "vand.vi v12, v12, 3\n\t"
                    "vle8.v v2, (%[qh])\n\t"
                    "vand.vx v4, v2, %[m]\n\t"
                    "slli %[m], %[m], 1\n\t"
                    "vmseq.vx v0, v4, zero\n\t"
                    "vadd.vi v8, v8, -4, v0.t\n\t"
                    "vand.vx v4, v2, %[m]\n\t"
                    "slli %[m], %[m], 1\n\t"
                    "vmseq.vx v0, v4, zero\n\t"
                    "vadd.vi v10, v10, -4, v0.t\n\t"
                    "vand.vx v4, v2, %[m]\n\t"
                    "slli %[m], %[m], 1\n\t"
                    "vmseq.vx v0, v4, zero\n\t"
                    "vadd.vi v12, v12, -4, v0.t\n\t"
                    "vand.vx v4, v2, %[m]\n\t"
                    "slli %[m], %[m], 1\n\t"
                    "vmseq.vx v0, v4, zero\n\t"
                    "vadd.vi v14, v14, -4, v0.t\n\t"
                    "vsetvli zero, %[vl128], e8, m8\n\t"
                    "vle8.v v0, (%[q8])\n\t"
                    "vsetvli zero, %[vl64], e8, m4\n\t"
                    "vwmul.vv v16, v0, v8\n\t"
                    "vwmul.vv v24, v4, v12\n\t"
                    "vsetivli zero, 16, e16, m2\n\t"
                    "vmv.v.x v0, zero\n\t"
                    "vwredsum.vs v10, v16, v0\n\t"
                    "vwredsum.vs v9, v18, v0\n\t"
                    "vwredsum.vs v8, v20, v0\n\t"
                    "vwredsum.vs v7, v22, v0\n\t"
                    "vwredsum.vs v11, v24, v0\n\t"
                    "vwredsum.vs v12, v26, v0\n\t"
                    "vwredsum.vs v13, v28, v0\n\t"
                    "vwredsum.vs v14, v30, v0\n\t"
                    "vsetivli zero, 4, e32, m1\n\t"
                    "vslideup.vi v10, v9, 1\n\t"
                    "vslideup.vi v8, v7, 1\n\t"
                    "vslideup.vi v11, v12, 1\n\t"
                    "vslideup.vi v13, v14, 1\n\t"
                    "vslideup.vi v10, v8, 2\n\t"
                    "vslideup.vi v11, v13, 2\n\t"
                    "vsetivli zero, 8, e32, m2\n\t"
                    "vle8.v v15, (%[scale])\n\t"
                    "vsext.vf4 v12, v15\n\t"
                    "vmul.vv v10, v10, v12\n\t"
                    "vredsum.vs v0, v10, v0\n\t"
                    "vmv.x.s %[tmp], v0\n\t"
                    "add %[isum], %[isum], %[tmp]"
                    : [tmp] "=&r" (tmp), [m] "+&r" (m), [isum] "+&r" (isum)
                    : [vl128] "r" (128), [vl64] "r" (64), [vl32] "r" (32)
                    , [q3] "r" (q3), [qh] "r" (qh), [scale] "r" (scale), [q8] "r" (q8)
                    : "memory"
                    , "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7"
                    , "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15"
                    , "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23"
                    , "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31"
                );
                q3 += 32;    q8 += 128;   scale += 8;
            }

            const float d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;
            sumf += d * isum;
        }
        break;
    default:
        assert(false && "Unsupported vector length");
        break;
    }

    *s = sumf;

#else
    // scalar version
    // This function is written like this so the compiler can manage to vectorize most of it
    // Using -Ofast, GCC and clang manage to produce code that is within a factor of 2 or so from the
    // manually vectorized version above. Every other version I tried would run at least 4 times slower.
    // The ideal situation would be if we could just write the code once, and the compiler would
    // automatically produce the best possible set of machine instructions, instead of us having to manually
    // write vectorized versions for AVX, ARM_NEON, etc.

    int8_t  aux8[QK_K];
    int16_t aux16[8];
    float   sums [8];
    int32_t aux32[8];
    memset(sums, 0, 8*sizeof(float));

    uint32_t auxs[4];
    const int8_t * scales = (const int8_t*)auxs;

    float sumf = 0;
    for (int i = 0; i < nb; ++i) {
        const uint8_t * GGML_RESTRICT q3 = x[i].qs;
        const uint8_t * GGML_RESTRICT hm = x[i].hmask;
        const  int8_t * GGML_RESTRICT q8 = y[i].qs;
        memset(aux32, 0, 8*sizeof(int32_t));
        int8_t * GGML_RESTRICT a = aux8;
        uint8_t m = 1;
        for (int j = 0; j < QK_K; j += 128) {
            for (int l = 0; l < 32; ++l) a[l] = q3[l] & 3;
            for (int l = 0; l < 32; ++l) a[l] -= (hm[l] & m ? 0 : 4);
            a += 32; m <<= 1;
            for (int l = 0; l < 32; ++l) a[l] = (q3[l] >> 2) & 3;
            for (int l = 0; l < 32; ++l) a[l] -= (hm[l] & m ? 0 : 4);
            a += 32; m <<= 1;
            for (int l = 0; l < 32; ++l) a[l] = (q3[l] >> 4) & 3;
            for (int l = 0; l < 32; ++l) a[l] -= (hm[l] & m ? 0 : 4);
            a += 32; m <<= 1;
            for (int l = 0; l < 32; ++l) a[l] = (q3[l] >> 6) & 3;
            for (int l = 0; l < 32; ++l) a[l] -= (hm[l] & m ? 0 : 4);
            a += 32; m <<= 1;
            q3 += 32;
        }
        a = aux8;

        memcpy(auxs, x[i].scales, 12);
        uint32_t tmp = auxs[2];
        auxs[2] = ((auxs[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        auxs[3] = ((auxs[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        auxs[0] = (auxs[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        auxs[1] = (auxs[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
        for (int j = 0; j < QK_K/16; ++j) {
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += (scales[j] - 32) * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += (scales[j] - 32) * aux16[l];
            q8 += 8; a += 8;
        }
        const float d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;
        for (int l = 0; l < 8; ++l) sums[l] += d * aux32[l];
    }
    for (int l = 0; l < 8; ++l) sumf += sums[l];
    *s = sumf;

#endif

}

void ggml_vec_dot_q4_K_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_q4_K * GGML_RESTRICT x = vx;
    const block_q8_K * GGML_RESTRICT y = vy;

    const int nb = n / QK_K;

    static const uint32_t kmask1 = 0x3f3f3f3f;
    static const uint32_t kmask2 = 0x0f0f0f0f;
    static const uint32_t kmask3 = 0x03030303;

    uint32_t utmp[4];

#if defined __riscv_xtheadvector

    const uint8_t * scales = (const uint8_t*)&utmp[0];
    const uint8_t * mins   = (const uint8_t*)&utmp[2];

    float sumf = 0;

    for (int i = 0; i < nb; ++i) {
        const float d = y[i].d * GGML_CPU_FP16_TO_FP32(x[i].d);
        const float dmin = y[i].d * GGML_CPU_FP16_TO_FP32(x[i].dmin);

        int tmp, tmp2, sumi;
        __asm__ __volatile__(
            "li %[t1], 12\n\t"
            "th.vsetvli zero, %[t1], e8, m1\n\t"
            "th.vlb.v v1, (%[s6b])\n\t" // {aux[0], aux[1], aux[2]}
            "li %[t1], 4\n\t"
            "th.vsetvli zero, %[t1], e32, m1\n\t"
            "th.vslidedown.vi v2, v1, 2\n\t"
            "th.vmv.v.v v3, v2\n\t"
            "th.vslideup.vi v2, v3, 1\n\t" // {aux[2], aux[2]}
            "li %[t1], 2\n\t"
            "th.vsetvli zero, %[t1], e32, m1\n\t"
            "th.vmv.v.i v4, 4\n\t"
            "th.vand.vx v8, v1, %[kmask1]\n\t"
            "th.vslide1up.vx v5, v4, zero\n\t" // {0, 4}
            "th.vsrl.vi v6, v1, 6\n\t"
            "th.vsrl.vv v7, v2, v5\n\t"
            "th.vand.vx v0, v6, %[kmask3]\n\t"
            "th.vand.vx v2, v7, %[kmask2]\n\t"
            "th.vsll.vi v6, v0, 4\n\t"
            "li %[t2], 8\n\t"
            "addi %[t1], %[utmp], 4\n\t"
            "th.vor.vv v1, v6, v2\n\t"
            "th.vssw.v v8, (%[utmp]), %[t2]\n\t"
            "th.vssw.v v1, (%[t1]), %[t2]\n\t"
            "th.vsetvli zero, zero, e32, m2\n\t" // vl == 8
            "th.vlw.v v2, (%[bsums])\n\t"
            "th.vsetvli zero, %[t2], e16, m1\n\t"
            "th.vnsrl.vi v0, v2, 0\n\t"
            "th.vnsrl.vi v1, v2, 16\n\t"
            "th.vadd.vv v2, v0, v1\n\t"
            "th.vlbu.v v4, (%[mins])\n\t"
            "th.vwmul.vv v6, v4, v2\n\t"
            "th.vmv.v.x v0, zero\n\t"
            "th.vsetvli zero, %[t2], e32, m2\n\t"
            "th.vredsum.vs v0, v6, v0\n\t"
            "th.vmv.x.s %[sumi], v0"
            : [t1] "=&r" (tmp), [t2] "=&r" (tmp2), [sumi] "=&r" (sumi)
            : [bsums] "r" (y[i].bsums), [mins] "r" (mins), [utmp] "r" (utmp)
            , [s6b] "r" (x[i].scales), [kmask1] "r" (kmask1)
            , [kmask2] "r" (kmask2), [kmask3] "r" (kmask3)
            : "memory"
            , "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7"
            , "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15"
            , "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23"
            , "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31"
        );
        sumf -= dmin * sumi;

        const uint8_t * restrict q4 = x[i].qs;
        const int8_t  * restrict q8 = y[i].qs;

        sumi = 0;
        const uint8_t * scale = scales;

        for (int j = 0; j < QK_K/128; ++j) {
            int vl128 = 128, vl64 = 64, vl32 = 32;
            __asm__ __volatile__(
                "th.vsetvli zero, %[vl128], e8, m8\n\t"
                "th.vlb.v v8, (%[q8])\n\t"
                "th.vsetvli zero, %[vl64], e8, m4\n\t"
                "th.vlb.v v0, (%[q4])\n\t"
                "th.vsrl.vi v4, v0, 4\n\t"
                "th.vand.vi v0, v0, 0xF\n\t"
                "th.vsetvli zero, %[vl32], e8, m2\n\t"
                "th.vwmul.vv v28, v6, v14\n\t"
                "th.vwmul.vv v20, v4, v10\n\t"
                "th.vwmul.vv v24, v2, v12\n\t"
                "th.vwmul.vv v16, v0, v8\n\t"
                "li %[tmp], 4\n\t"
                "th.vsetvli zero, %[tmp], e32, m1\n\t"
                "th.vlbu.v v1, (%[scale])\n\t"
                "th.vmv.v.x v0, zero\n\t"
                "th.vsetvli zero, %[vl32], e16, m4\n\t"
                "th.vwredsum.vs v6, v24, v0\n\t"
                "th.vwredsum.vs v7, v28, v0\n\t"
                "th.vwredsum.vs v4, v16, v0\n\t"
                "th.vwredsum.vs v5, v20, v0\n\t"
                "th.vsetvli zero, %[tmp], e32, m1\n\t"
                "th.vslideup.vi v6, v7, 1\n\t"
                "th.vslideup.vi v4, v5, 1\n\t"
                "th.vslideup.vi v4, v6, 2\n\t"
                "th.vmul.vv v8, v4, v1\n\t"
                "th.vredsum.vs v0, v8, v0\n\t"
                "th.vmv.x.s %[tmp], v0\n\t"
                "add %[sumi], %[sumi], %[tmp]"
                : [tmp] "=&r" (tmp), [sumi] "+&r" (sumi)
                : [vl128] "r" (vl128), [vl64] "r" (vl64), [vl32] "r" (vl32)
                , [q4] "r" (q4), [q8] "r" (q8), [scale] "r" (scale)
                : "memory"
                , "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7"
                , "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15"
                , "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23"
                , "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31"
            );

            q4 += 64;    q8 += 128;    scale += 4;
        }

        sumf += d * sumi;

    }

    *s = sumf;

#elif defined __riscv_v

    const uint8_t * scales = (const uint8_t*)&utmp[0];
    const uint8_t * mins   = (const uint8_t*)&utmp[2];

    float sumf = 0;
    const int vector_length = __riscv_vlenb() * 8;

    switch (vector_length) {
    case 256:
        for (int i = 0; i < nb; ++i) {

            size_t vl = 8;

            const float d = y[i].d * GGML_CPU_FP16_TO_FP32(x[i].d);
            const float dmin = y[i].d * GGML_CPU_FP16_TO_FP32(x[i].dmin);

            vint16mf2_t q8sums_0 = __riscv_vlse16_v_i16mf2(y[i].bsums, 4, vl);
            vint16mf2_t q8sums_1 = __riscv_vlse16_v_i16mf2(y[i].bsums+1, 4, vl);
            vint16mf2_t q8sums   = __riscv_vadd_vv_i16mf2(q8sums_0, q8sums_1, vl);

            memcpy(utmp, x[i].scales, 12);
            utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
            const uint32_t uaux = utmp[1] & kmask1;
            utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
            utmp[2] = uaux;
            utmp[0] &= kmask1;

            vuint8mf4_t mins8  = __riscv_vle8_v_u8mf4(mins, vl);
            vint16mf2_t v_mins = __riscv_vreinterpret_v_u16mf2_i16mf2(__riscv_vzext_vf2_u16mf2(mins8, vl));
            vint32m1_t  prod   = __riscv_vwmul_vv_i32m1(q8sums, v_mins, vl);

            vint32m1_t sumi = __riscv_vredsum_vs_i32m1_i32m1(prod, __riscv_vmv_v_x_i32m1(0, 1), vl);
            sumf -= dmin * __riscv_vmv_x_s_i32m1_i32(sumi);

            const uint8_t * GGML_RESTRICT q4 = x[i].qs;
            const int8_t  * GGML_RESTRICT q8 = y[i].qs;

            vl = 32;

            int32_t sum_1 = 0;
            int32_t sum_2 = 0;

            vint16m1_t vzero = __riscv_vmv_v_x_i16m1(0, 1);

            for (int j = 0; j < QK_K/64; ++j) {
                // load Q4
                vuint8m1_t q4_x = __riscv_vle8_v_u8m1(q4, vl);

                // load Q8 and multiply it with lower Q4 nibble
                vint8m1_t  q8_0 = __riscv_vle8_v_i8m1(q8, vl);
                vint8m1_t  q4_0 = __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vand_vx_u8m1(q4_x, 0x0F, vl));
                vint16m2_t qv_0 = __riscv_vwmul_vv_i16m2(q4_0, q8_0, vl);
                vint16m1_t vs_0 = __riscv_vredsum_vs_i16m2_i16m1(qv_0, vzero, vl);

                sum_1 += __riscv_vmv_x_s_i16m1_i16(vs_0) * scales[2*j+0];

                // load Q8 and multiply it with upper Q4 nibble
                vint8m1_t  q8_1 = __riscv_vle8_v_i8m1(q8+32, vl);
                vint8m1_t  q4_1 = __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vsrl_vx_u8m1(q4_x, 0x04, vl));
                vint16m2_t qv_1 = __riscv_vwmul_vv_i16m2(q4_1, q8_1, vl);
                vint16m1_t vs_1 = __riscv_vredsum_vs_i16m2_i16m1(qv_1, vzero, vl);

                sum_2 += __riscv_vmv_x_s_i16m1_i16(vs_1) * scales[2*j+1];

                q4 += 32;    q8 += 64;

            }

            sumf += d*(sum_1 + sum_2);

        }
        break;
    case 128:
        for (int i = 0; i < nb; ++i) {
            const float d = y[i].d * GGML_CPU_FP16_TO_FP32(x[i].d);
            const float dmin = y[i].d * GGML_CPU_FP16_TO_FP32(x[i].dmin);

            int tmp, tmp2, sumi;
            __asm__ __volatile__(
                "vsetivli zero, 12, e8, m1\n\t"
                "vle8.v v1, (%[s6b])\n\t" // {aux[0], aux[1], aux[2]}
                "vsetivli zero, 4, e32, m1\n\t"
                "vslidedown.vi v2, v1, 2\n\t"
                "vmv1r.v v3, v2\n\t"
                "vslideup.vi v2, v3, 1\n\t" // {aux[2], aux[2]}
                "vsetivli zero, 2, e32, m1\n\t"
                "vmv.v.i v4, 4\n\t"
                "vand.vx v8, v1, %[kmask1]\n\t"
                "vslide1up.vx v5, v4, zero\n\t" // {0, 4}
                "vsrl.vi v6, v1, 6\n\t"
                "vsrl.vv v7, v2, v5\n\t"
                "vand.vx v0, v6, %[kmask3]\n\t"
                "vand.vx v2, v7, %[kmask2]\n\t"
                "vsll.vi v6, v0, 4\n\t"
                "li %[t2], 8\n\t"
                "addi %[t1], %[utmp], 4\n\t"
                "vor.vv v1, v6, v2\n\t"
                "vsse32.v v8, (%[utmp]), %[t2]\n\t"
                "vsse32.v v1, (%[t1]), %[t2]\n\t"
                "vsetivli zero, 8, e16, m1\n\t"
                "vle32.v v2, (%[bsums])\n\t"
                "vnsrl.wi v0, v2, 0\n\t"
                "vnsrl.wi v1, v2, 16\n\t"
                "vadd.vv v2, v0, v1\n\t"
                "vle8.v v3, (%[mins])\n\t"
                "vzext.vf2 v4, v3\n\t"
                "vwmul.vv v6, v4, v2\n\t"
                "vmv.v.x v0, zero\n\t"
                "vsetivli zero, 8, e32, m2\n\t"
                "vredsum.vs v0, v6, v0\n\t"
                "vmv.x.s %[sumi], v0"
                : [t1] "=&r" (tmp), [t2] "=&r" (tmp2), [sumi] "=&r" (sumi)
                : [bsums] "r" (y[i].bsums), [mins] "r" (mins), [utmp] "r" (utmp)
                , [s6b] "r" (x[i].scales), [kmask1] "r" (kmask1)
                , [kmask2] "r" (kmask2), [kmask3] "r" (kmask3)
                : "memory"
                , "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7"
                , "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15"
                , "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23"
                , "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31"
            );
            sumf -= dmin * sumi;

            const uint8_t * restrict q4 = x[i].qs;
            const int8_t  * restrict q8 = y[i].qs;

            sumi = 0;
            const uint8_t * scale = scales;

            for (int j = 0; j < QK_K/128; ++j) {
                int vl128 = 128, vl64 = 64, vl32 = 32;
                __asm__ __volatile__(
                    "vsetvli zero, %[vl128], e8, m8\n\t"
                    "vle8.v v8, (%[q8])\n\t"
                    "vsetvli zero, %[vl64], e8, m4\n\t"
                    "vle8.v v0, (%[q4])\n\t"
                    "vsrl.vi v4, v0, 4\n\t"
                    "vand.vi v0, v0, 0xF\n\t"
                    "vsetvli zero, %[vl32], e8, m2\n\t"
                    "vwmul.vv v28, v6, v14\n\t"
                    "vwmul.vv v20, v4, v10\n\t"
                    "vwmul.vv v24, v2, v12\n\t"
                    "vwmul.vv v16, v0, v8\n\t"
                    "vsetivli zero, 4, e32, m1\n\t"
                    "vle8.v v2, (%[scale])\n\t"
                    "vmv.v.x v0, zero\n\t"
                    "vzext.vf4 v1, v2\n\t"
                    "vsetvli zero, %[vl32], e16, m4\n\t"
                    "vwredsum.vs v6, v24, v0\n\t"
                    "vwredsum.vs v7, v28, v0\n\t"
                    "vwredsum.vs v4, v16, v0\n\t"
                    "vwredsum.vs v5, v20, v0\n\t"
                    "vsetivli zero, 4, e32, m1\n\t"
                    "vslideup.vi v6, v7, 1\n\t"
                    "vslideup.vi v4, v5, 1\n\t"
                    "vslideup.vi v4, v6, 2\n\t"
                    "vmul.vv v8, v4, v1\n\t"
                    "vredsum.vs v0, v8, v0\n\t"
                    "vmv.x.s %[tmp], v0\n\t"
                    "add %[sumi], %[sumi], %[tmp]"
                    : [tmp] "=&r" (tmp), [sumi] "+&r" (sumi)
                    : [vl128] "r" (vl128), [vl64] "r" (vl64), [vl32] "r" (vl32)
                    , [q4] "r" (q4), [q8] "r" (q8), [scale] "r" (scale)
                    : "memory"
                    , "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7"
                    , "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15"
                    , "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23"
                    , "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31"
                );

                q4 += 64;    q8 += 128;    scale += 4;
            }

            sumf += d * sumi;
        }
        break;
    default:
        assert(false && "Unsupported vector length");
        break;
    }

    *s = sumf;

#else

    const uint8_t * scales = (const uint8_t*)&utmp[0];
    const uint8_t * mins   = (const uint8_t*)&utmp[2];

    int8_t  aux8[QK_K];
    int16_t aux16[8];
    float   sums [8];
    int32_t aux32[8];
    memset(sums, 0, 8*sizeof(float));

    float sumf = 0;
    for (int i = 0; i < nb; ++i) {
        const uint8_t * GGML_RESTRICT q4 = x[i].qs;
        const  int8_t * GGML_RESTRICT q8 = y[i].qs;
        memset(aux32, 0, 8*sizeof(int32_t));
        int8_t * GGML_RESTRICT a = aux8;
        for (int j = 0; j < QK_K/64; ++j) {
            for (int l = 0; l < 32; ++l) a[l] = (int8_t)(q4[l] & 0xF);
            a += 32;
            for (int l = 0; l < 32; ++l) a[l] = (int8_t)(q4[l]  >> 4);
            a += 32; q4 += 32;
        }
        memcpy(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        int sumi = 0;
        for (int j = 0; j < QK_K/16; ++j) sumi += y[i].bsums[j] * mins[j/2];
        a = aux8;
        int is = 0;
        for (int j = 0; j < QK_K/32; ++j) {
            int32_t scale = scales[is++];
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
        }
        const float d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;
        for (int l = 0; l < 8; ++l) sums[l] += d * aux32[l];
        const float dmin = GGML_CPU_FP16_TO_FP32(x[i].dmin) * y[i].d;
        sumf -= dmin * sumi;
    }
    for (int l = 0; l < 8; ++l) sumf += sums[l];
    *s = sumf;
#endif
}

void ggml_vec_dot_q5_K_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy,  size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_q5_K * GGML_RESTRICT x = vx;
    const block_q8_K * GGML_RESTRICT y = vy;

    const int nb = n / QK_K;

    static const uint32_t kmask1 = 0x3f3f3f3f;
    static const uint32_t kmask2 = 0x0f0f0f0f;
    static const uint32_t kmask3 = 0x03030303;

    uint32_t utmp[4];

#if defined __riscv_v

    const uint8_t * scales = (const uint8_t*)&utmp[0];
    const uint8_t * mins   = (const uint8_t*)&utmp[2];

    float sumf = 0;
    float sums = 0.0;

    size_t vl;

    for (int i = 0; i < nb; ++i) {

        vl = 8;

        const uint8_t * GGML_RESTRICT q5 = x[i].qs;
        const uint8_t * GGML_RESTRICT hm = x[i].qh;
        const  int8_t * GGML_RESTRICT q8 = y[i].qs;

        const float d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;
        const float dmin = GGML_CPU_FP16_TO_FP32(x[i].dmin) * y[i].d;

        vint16m1_t q8sums_0 = __riscv_vlse16_v_i16m1(y[i].bsums, 4, vl);
        vint16m1_t q8sums_1 = __riscv_vlse16_v_i16m1(y[i].bsums+1, 4, vl);
        vint16m1_t q8sums = __riscv_vadd_vv_i16m1(q8sums_0, q8sums_1, vl);

        memcpy(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        vuint8mf2_t mins8 = __riscv_vle8_v_u8mf2(mins, vl);
        vint16m1_t v_mins = __riscv_vreinterpret_v_u16m1_i16m1(__riscv_vzext_vf2_u16m1(mins8, vl));
        vint32m2_t prod = __riscv_vwmul_vv_i32m2(q8sums, v_mins, vl);

        vint32m1_t sumi = __riscv_vredsum_vs_i32m2_i32m1(prod, __riscv_vmv_v_x_i32m1(0, 1), vl);
        sumf -= dmin * __riscv_vmv_x_s_i32m1_i32(sumi);

        vl = 32;
        int32_t aux32 = 0;
        int is = 0;

        uint8_t m = 1;
        vint32m1_t vzero = __riscv_vmv_v_x_i32m1(0, 1);
        vuint8m2_t vqh = __riscv_vle8_v_u8m2(hm, vl);

        for (int j = 0; j < QK_K/64; ++j) {
            // load Q5 and Q8
            vuint8m2_t q5_x = __riscv_vle8_v_u8m2(q5, vl);
            vint8m2_t  q8_y1 = __riscv_vle8_v_i8m2(q8, vl);
            vint8m2_t  q8_y2 = __riscv_vle8_v_i8m2(q8+32, vl);

            // compute mask for addition
            vint8m2_t q5_a = __riscv_vreinterpret_v_u8m2_i8m2(__riscv_vand_vx_u8m2(q5_x, 0x0F, vl));
            vuint8m2_t qh_m1 = __riscv_vand_vx_u8m2(vqh, m, vl);
            vbool4_t vmask_1 = __riscv_vmsne_vx_u8m2_b4(qh_m1, 0, vl);
            vint8m2_t q5_m1 = __riscv_vadd_vx_i8m2_mu(vmask_1, q5_a, q5_a, 16, vl);
            m <<= 1;

            vint8m2_t q5_l = __riscv_vreinterpret_v_u8m2_i8m2(__riscv_vsrl_vx_u8m2(q5_x, 0x04, vl));
            vuint8m2_t qh_m2 = __riscv_vand_vx_u8m2(vqh, m, vl);
            vbool4_t vmask_2 = __riscv_vmsne_vx_u8m2_b4(qh_m2, 0, vl);
            vint8m2_t q5_m2 = __riscv_vadd_vx_i8m2_mu(vmask_2, q5_l, q5_l, 16, vl);
            m <<= 1;

            vint16m4_t v0 = __riscv_vwmul_vv_i16m4(q5_m1, q8_y1, vl);
            vint16m4_t v1 = __riscv_vwmul_vv_i16m4(q5_m2, q8_y2, vl);

            vint32m8_t vs1 = __riscv_vwmul_vx_i32m8(v0, scales[is++], vl);
            vint32m8_t vs2 = __riscv_vwmul_vx_i32m8(v1, scales[is++], vl);

            vint32m1_t vacc1 = __riscv_vredsum_vs_i32m8_i32m1(vs1, vzero, vl);
            vint32m1_t vacc2 = __riscv_vredsum_vs_i32m8_i32m1(vs2, vacc1, vl);

            aux32 += __riscv_vmv_x_s_i32m1_i32(vacc2);
            q5 += 32;    q8 += 64;

        }

        sums += aux32 * d;

    }

    *s = sumf+sums;

#else

    const uint8_t * scales = (const uint8_t*)&utmp[0];
    const uint8_t * mins   = (const uint8_t*)&utmp[2];

    int8_t  aux8[QK_K];
    int16_t aux16[8];
    float   sums [8];
    int32_t aux32[8];
    memset(sums, 0, 8*sizeof(float));

    float sumf = 0;
    for (int i = 0; i < nb; ++i) {
        const uint8_t * GGML_RESTRICT q4 = x[i].qs;
        const uint8_t * GGML_RESTRICT hm = x[i].qh;
        const  int8_t * GGML_RESTRICT q8 = y[i].qs;
        memset(aux32, 0, 8*sizeof(int32_t));
        int8_t * GGML_RESTRICT a = aux8;
        uint8_t m = 1;
        for (int j = 0; j < QK_K/64; ++j) {
            for (int l = 0; l < 32; ++l) a[l] = (int8_t)(q4[l] & 0xF);
            for (int l = 0; l < 32; ++l) a[l] += (hm[l] & m ? 16 : 0);
            a += 32; m <<= 1;
            for (int l = 0; l < 32; ++l) a[l] = (int8_t)(q4[l]  >> 4);
            for (int l = 0; l < 32; ++l) a[l] += (hm[l] & m ? 16 : 0);
            a += 32; m <<= 1;
            q4 += 32;
        }
        memcpy(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        int sumi = 0;
        for (int j = 0; j < QK_K/16; ++j) sumi += y[i].bsums[j] * mins[j/2];
        a = aux8;
        int is = 0;
        for (int j = 0; j < QK_K/32; ++j) {
            int32_t scale = scales[is++];
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
        }
        const float d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;
        for (int l = 0; l < 8; ++l) sums[l] += d * aux32[l];
        const float dmin = GGML_CPU_FP16_TO_FP32(x[i].dmin) * y[i].d;
        sumf -= dmin * sumi;
    }
    for (int l = 0; l < 8; ++l) sumf += sums[l];
    *s = sumf;
#endif
}

void ggml_vec_dot_q6_K_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_q6_K * GGML_RESTRICT x = vx;
    const block_q8_K * GGML_RESTRICT y = vy;

    const int nb = n / QK_K;

#if defined __riscv_xtheadvector

    float sumf = 0;

    for (int i = 0; i < nb; ++i) {

        const float d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;

        const uint8_t * restrict q6 = x[i].ql;
        const uint8_t * restrict qh = x[i].qh;
        const  int8_t * restrict q8 = y[i].qs;

        const int8_t * restrict scale = x[i].scales;

        int sum_t = 0;
        int t0;

        for (int j = 0; j < QK_K/128; ++j) {
            __asm__ __volatile__(
                "th.vsetvli zero, %[vl32], e8, m2\n\t" // vl == 32
                "th.vlb.v v4, (%[qh])\n\t"
                "th.vsll.vi v0, v4, 4\n\t"
                "th.vsll.vi v2, v4, 2\n\t"
                "th.vsrl.vi v6, v4, 2\n\t"
                "th.vsetvli zero, %[vl64], e8, m4\n\t" // vl == 64
                "th.vlb.v v8, (%[q6])\n\t"
                "th.vsrl.vi v12, v8, 4\n\t"
                "th.vand.vi v8, v8, 0xF\n\t"
                "th.vsetvli zero, %[vl128], e8, m8\n\t" // vl == 128
                "th.vand.vx v0, v0, %[mask]\n\t"
                "th.vor.vv v8, v8, v0\n\t"
                "th.vlb.v v0, (%[q8])\n\t"
                "th.vsub.vx v8, v8, %[vl32]\n\t"
                "th.vsetvli zero, %[vl64], e8, m4\n\t" // vl == 64
                "th.vwmul.vv v16, v0, v8\n\t"
                "th.vwmul.vv v24, v4, v12\n\t"
                "li %[t0], 16\n\t"
                "th.vsetvli zero, %[t0], e16, m2\n\t" // vl == 16
                "th.vmv.v.x v0, zero\n\t"
                "th.vwredsum.vs v10, v16, v0\n\t"
                "th.vwredsum.vs v9, v18, v0\n\t"
                "th.vwredsum.vs v8, v20, v0\n\t"
                "th.vwredsum.vs v7, v22, v0\n\t"
                "th.vwredsum.vs v11, v24, v0\n\t"
                "th.vwredsum.vs v12, v26, v0\n\t"
                "th.vwredsum.vs v13, v28, v0\n\t"
                "th.vwredsum.vs v14, v30, v0\n\t"
                "li %[t0], 4\n\t"
                "th.vsetvli zero, %[t0], e32, m1\n\t" // vl == 4
                "th.vslideup.vi v10, v9, 1\n\t"
                "th.vslideup.vi v8, v7, 1\n\t"
                "th.vslideup.vi v11, v12, 1\n\t"
                "th.vslideup.vi v13, v14, 1\n\t"
                "th.vslideup.vi v10, v8, 2\n\t"
                "th.vslideup.vi v11, v13, 2\n\t"
                "li %[t0], 8\n\t"
                "th.vsetvli zero, %[t0], e32, m2\n\t" // vl == 8
                "th.vlb.v v4, (%[scale])\n\t"
                "th.vmul.vv v2, v4, v10\n\t"
                "th.vredsum.vs v0, v2, v0\n\t"
                "th.vmv.x.s %[t0], v0\n\t"
                "add %[sumi], %[sumi], %[t0]"
                : [sumi] "+&r" (sum_t), [t0] "=&r" (t0)
                : [qh] "r" (qh), [q6] "r" (q6), [q8] "r" (q8), [scale] "r" (scale)
                , [vl32] "r" (32), [vl64] "r" (64), [vl128] "r" (128)
                , [mask] "r" (0x30)
                : "memory"
                , "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7"
                , "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15"
                , "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23"
                , "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31"
            );
            q6 += 64;   qh += 32;   q8 += 128;   scale += 8;
        }

        sumf += d * sum_t;

    }

    *s = sumf;

#elif defined __riscv_v

    float sumf = 0;
    const int vector_length = __riscv_vlenb() * 8;

    switch (vector_length) {
    case 256:
        for (int i = 0; i < nb; ++i) {

            const float d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;

            const uint8_t * GGML_RESTRICT q6 = x[i].ql;
            const uint8_t * GGML_RESTRICT qh = x[i].qh;
            const  int8_t * GGML_RESTRICT q8 = y[i].qs;

            const int8_t * GGML_RESTRICT scale = x[i].scales;

            size_t vl;

            vint32m1_t vzero = __riscv_vmv_v_x_i32m1(0, 1);

            int sum_t = 0;
            int is = 0;

            for (int j = 0; j < QK_K/128; ++j) {

                vl = 32;

                // load qh
                vuint8m1_t qh_x = __riscv_vle8_v_u8m1(qh, vl);

                // load Q6
                vuint8m1_t q6_0 = __riscv_vle8_v_u8m1(q6, vl);
                vuint8m1_t q6_1 = __riscv_vle8_v_u8m1(q6+32, vl);

                vuint8m1_t q6a_0 = __riscv_vand_vx_u8m1(q6_0, 0x0F, vl);
                vuint8m1_t q6a_1 = __riscv_vand_vx_u8m1(q6_1, 0x0F, vl);
                vuint8m1_t q6s_0 = __riscv_vsrl_vx_u8m1(q6_0, 0x04, vl);
                vuint8m1_t q6s_1 = __riscv_vsrl_vx_u8m1(q6_1, 0x04, vl);

                vuint8m1_t qh_0 = __riscv_vand_vx_u8m1(qh_x, 0x03, vl);
                vuint8m1_t qh_1 = __riscv_vand_vx_u8m1(__riscv_vsrl_vx_u8m1(qh_x, 0x2, vl), 0x03 , vl);
                vuint8m1_t qh_2 = __riscv_vand_vx_u8m1(__riscv_vsrl_vx_u8m1(qh_x, 0x4, vl), 0x03 , vl);
                vuint8m1_t qh_3 = __riscv_vand_vx_u8m1(__riscv_vsrl_vx_u8m1(qh_x, 0x6, vl), 0x03 , vl);

                vuint8m1_t qhi_0 = __riscv_vor_vv_u8m1(q6a_0, __riscv_vsll_vx_u8m1(qh_0, 0x04, vl), vl);
                vuint8m1_t qhi_1 = __riscv_vor_vv_u8m1(q6a_1, __riscv_vsll_vx_u8m1(qh_1, 0x04, vl), vl);
                vuint8m1_t qhi_2 = __riscv_vor_vv_u8m1(q6s_0, __riscv_vsll_vx_u8m1(qh_2, 0x04, vl), vl);
                vuint8m1_t qhi_3 = __riscv_vor_vv_u8m1(q6s_1, __riscv_vsll_vx_u8m1(qh_3, 0x04, vl), vl);

                vint8m1_t a_0 = __riscv_vsub_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(qhi_0), 32, vl);
                vint8m1_t a_1 = __riscv_vsub_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(qhi_1), 32, vl);
                vint8m1_t a_2 = __riscv_vsub_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(qhi_2), 32, vl);
                vint8m1_t a_3 = __riscv_vsub_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(qhi_3), 32, vl);

                // load Q8 and take product
                vint16m2_t va_q_0 = __riscv_vwmul_vv_i16m2(a_0, __riscv_vle8_v_i8m1(q8, vl), vl);
                vint16m2_t va_q_1 = __riscv_vwmul_vv_i16m2(a_1, __riscv_vle8_v_i8m1(q8+32, vl), vl);
                vint16m2_t va_q_2 = __riscv_vwmul_vv_i16m2(a_2, __riscv_vle8_v_i8m1(q8+64, vl), vl);
                vint16m2_t va_q_3 = __riscv_vwmul_vv_i16m2(a_3, __riscv_vle8_v_i8m1(q8+96, vl), vl);

                vl = 16;

                vint32m2_t vaux_0 = __riscv_vwmul_vx_i32m2(__riscv_vget_v_i16m2_i16m1(va_q_0, 0), scale[is+0], vl);
                vint32m2_t vaux_1 = __riscv_vwmul_vx_i32m2(__riscv_vget_v_i16m2_i16m1(va_q_0, 1), scale[is+1], vl);
                vint32m2_t vaux_2 = __riscv_vwmul_vx_i32m2(__riscv_vget_v_i16m2_i16m1(va_q_1, 0), scale[is+2], vl);
                vint32m2_t vaux_3 = __riscv_vwmul_vx_i32m2(__riscv_vget_v_i16m2_i16m1(va_q_1, 1), scale[is+3], vl);
                vint32m2_t vaux_4 = __riscv_vwmul_vx_i32m2(__riscv_vget_v_i16m2_i16m1(va_q_2, 0), scale[is+4], vl);
                vint32m2_t vaux_5 = __riscv_vwmul_vx_i32m2(__riscv_vget_v_i16m2_i16m1(va_q_2, 1), scale[is+5], vl);
                vint32m2_t vaux_6 = __riscv_vwmul_vx_i32m2(__riscv_vget_v_i16m2_i16m1(va_q_3, 0), scale[is+6], vl);
                vint32m2_t vaux_7 = __riscv_vwmul_vx_i32m2(__riscv_vget_v_i16m2_i16m1(va_q_3, 1), scale[is+7], vl);

                vint32m1_t isum0 = __riscv_vredsum_vs_i32m2_i32m1(__riscv_vadd_vv_i32m2(vaux_0, vaux_1, vl), vzero, vl);
                vint32m1_t isum1 = __riscv_vredsum_vs_i32m2_i32m1(__riscv_vadd_vv_i32m2(vaux_2, vaux_3, vl), isum0, vl);
                vint32m1_t isum2 = __riscv_vredsum_vs_i32m2_i32m1(__riscv_vadd_vv_i32m2(vaux_4, vaux_5, vl), isum1, vl);
                vint32m1_t isum3 = __riscv_vredsum_vs_i32m2_i32m1(__riscv_vadd_vv_i32m2(vaux_6, vaux_7, vl), isum2, vl);

                sum_t += __riscv_vmv_x_s_i32m1_i32(isum3);

                q6 += 64;   qh += 32;   q8 += 128;   is=8;

            }

            sumf += d * sum_t;

        }
        break;
    case 128:
        for (int i = 0; i < nb; ++i) {

            const float d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;

            const uint8_t * restrict q6 = x[i].ql;
            const uint8_t * restrict qh = x[i].qh;
            const  int8_t * restrict q8 = y[i].qs;

            const int8_t * restrict scale = x[i].scales;

            int sum_t = 0;
            int t0;

            for (int j = 0; j < QK_K/128; ++j) {
                __asm__ __volatile__(
                    "vsetvli zero, %[vl32], e8, m2\n\t"
                    "vle8.v v4, (%[qh])\n\t"
                    "vsll.vi v0, v4, 4\n\t"
                    "vsll.vi v2, v4, 2\n\t"
                    "vsrl.vi v6, v4, 2\n\t"
                    "vsetvli zero, %[vl64], e8, m4\n\t"
                    "vle8.v v8, (%[q6])\n\t"
                    "vsrl.vi v12, v8, 4\n\t"
                    "vand.vi v8, v8, 0xF\n\t"
                    "vsetvli zero, %[vl128], e8, m8\n\t"
                    "vand.vx v0, v0, %[mask]\n\t"
                    "vor.vv v8, v8, v0\n\t"
                    "vle8.v v0, (%[q8])\n\t"
                    "vsub.vx v8, v8, %[vl32]\n\t"
                    "vsetvli zero, %[vl64], e8, m4\n\t"
                    "vwmul.vv v16, v0, v8\n\t"
                    "vwmul.vv v24, v4, v12\n\t"
                    "vsetivli zero, 16, e16, m2\n\t"
                    "vmv.v.x v0, zero\n\t"
                    "vwredsum.vs v10, v16, v0\n\t"
                    "vwredsum.vs v9, v18, v0\n\t"
                    "vwredsum.vs v8, v20, v0\n\t"
                    "vwredsum.vs v7, v22, v0\n\t"
                    "vwredsum.vs v11, v24, v0\n\t"
                    "vwredsum.vs v12, v26, v0\n\t"
                    "vwredsum.vs v13, v28, v0\n\t"
                    "vwredsum.vs v14, v30, v0\n\t"
                    "vsetivli zero, 4, e32, m1\n\t"
                    "vslideup.vi v10, v9, 1\n\t"
                    "vslideup.vi v8, v7, 1\n\t"
                    "vslideup.vi v11, v12, 1\n\t"
                    "vslideup.vi v13, v14, 1\n\t"
                    "vslideup.vi v10, v8, 2\n\t"
                    "vslideup.vi v11, v13, 2\n\t"
                    "vsetivli zero, 8, e32, m2\n\t"
                    "vle8.v v2, (%[scale])\n\t"
                    "vsext.vf4 v4, v2\n\t"
                    "vmul.vv v2, v4, v10\n\t"
                    "vredsum.vs v0, v2, v0\n\t"
                    "vmv.x.s %[t0], v0\n\t"
                    "add %[sumi], %[sumi], %[t0]"
                    : [sumi] "+&r" (sum_t), [t0] "=&r" (t0)
                    : [qh] "r" (qh), [q6] "r" (q6), [q8] "r" (q8), [scale] "r" (scale)
                    , [vl32] "r" (32), [vl64] "r" (64), [vl128] "r" (128)
                    , [mask] "r" (0x30)
                    : "memory"
                    , "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7"
                    , "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15"
                    , "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23"
                    , "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31"
                );
                q6 += 64;   qh += 32;   q8 += 128;   scale += 8;
            }

            sumf += d * sum_t;

        }
        break;
    default:
        assert(false && "Unsupported vector length");
        break;
    }

    *s = sumf;

#else

    int8_t  aux8[QK_K];
    int16_t aux16[8];
    float   sums [8];
    int32_t aux32[8];
    memset(sums, 0, 8*sizeof(float));

    float sumf = 0;
    for (int i = 0; i < nb; ++i) {
        const uint8_t * GGML_RESTRICT q4 = x[i].ql;
        const uint8_t * GGML_RESTRICT qh = x[i].qh;
        const  int8_t * GGML_RESTRICT q8 = y[i].qs;
        memset(aux32, 0, 8*sizeof(int32_t));
        int8_t * GGML_RESTRICT a = aux8;
        for (int j = 0; j < QK_K; j += 128) {
            for (int l = 0; l < 32; ++l) {
                a[l +  0] = (int8_t)((q4[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                a[l + 32] = (int8_t)((q4[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                a[l + 64] = (int8_t)((q4[l +  0] >>  4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                a[l + 96] = (int8_t)((q4[l + 32] >>  4) | (((qh[l] >> 6) & 3) << 4)) - 32;
            }
            a  += 128;
            q4 += 64;
            qh += 32;
        }
        a = aux8;
        int is = 0;
        for (int j = 0; j < QK_K/16; ++j) {
            int scale = x[i].scales[is++];
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
        }
        const float d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;
        for (int l = 0; l < 8; ++l) sums[l] += d * aux32[l];
    }
    for (int l = 0; l < 8; ++l) sumf += sums[l];
    *s = sumf;
#endif
}

