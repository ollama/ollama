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

#if defined(__VXE__) || defined(__VXE2__)
    for (int i = 0; i < nb; i++) {
        __vector float srcv [8];
        __vector float asrcv[8];
        __vector float amaxv[8];

        for (int j = 0; j < 8; j++) srcv[j] = vec_xl(0, x + i*32 + 4*j);
        for (int j = 0; j < 8; j++) asrcv[j] = vec_abs(srcv[j]);
        for (int j = 0; j < 4; j++) amaxv[2*j] = vec_max(asrcv[2*j], asrcv[2*j+1]);
        for (int j = 0; j < 2; j++) amaxv[4*j] = vec_max(amaxv[4*j], amaxv[4*j+2]);
        for (int j = 0; j < 1; j++) amaxv[8*j] = vec_max(amaxv[8*j], amaxv[8*j+4]);

        const float amax = MAX(MAX(vec_extract(amaxv[0], 0),
                                   vec_extract(amaxv[0], 1)),
                               MAX(vec_extract(amaxv[0], 2),
                                   vec_extract(amaxv[0], 3)));

        const float d = amax / ((1 << 7) - 1);
        const float id = d ? 1.0f / d : 0.0f;

        y[i].d = GGML_CPU_FP32_TO_FP16(d);

        for (int j = 0; j < 8; j++) {
            const __vector float v = vec_mul(srcv[j], vec_splats(id));
            const __vector int32_t vi = vec_signed(v);

            y[i].qs[4*j + 0] = vec_extract(vi, 0);
            y[i].qs[4*j + 1] = vec_extract(vi, 1);
            y[i].qs[4*j + 2] = vec_extract(vi, 2);
            y[i].qs[4*j + 3] = vec_extract(vi, 3);
        }
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

#if defined(__VXE__) || defined(__VXE2__)
    for (int i = 0; i < nb; i++) {
        __vector float srcv [8];
        __vector float asrcv[8];
        __vector float amaxv[8];

        for (int j = 0; j < 8; j++) srcv[j] = vec_xl(0, x + i*32 + 4*j);
        for (int j = 0; j < 8; j++) asrcv[j] = vec_abs(srcv[j]);
        for (int j = 0; j < 4; j++) amaxv[2*j] = vec_max(asrcv[2*j], asrcv[2*j+1]);
        for (int j = 0; j < 2; j++) amaxv[4*j] = vec_max(amaxv[4*j], amaxv[4*j+2]);
        for (int j = 0; j < 1; j++) amaxv[8*j] = vec_max(amaxv[8*j], amaxv[8*j+4]);

        const float amax = MAX(MAX(vec_extract(amaxv[0], 0),
                                   vec_extract(amaxv[0], 1)),
                               MAX(vec_extract(amaxv[0], 2),
                                   vec_extract(amaxv[0], 3)));

        const float d = amax / ((1 << 7) - 1);
        const float id = d ? 1.0f / d : 0.0f;

        y[i].d = GGML_CPU_FP32_TO_FP16(d);

        __vector int32_t acc = vec_splats(0);

        for (int j = 0; j < 8; j++) {
            const __vector float v = vec_mul(srcv[j], vec_splats(id));
            const __vector int32_t vi = vec_signed(v);

            y[i].qs[4*j + 0] = vec_extract(vi, 0);
            y[i].qs[4*j + 1] = vec_extract(vi, 1);
            y[i].qs[4*j + 2] = vec_extract(vi, 2);
            y[i].qs[4*j + 3] = vec_extract(vi, 3);

            acc = vec_add(acc, vi);
        }

        y[i].s = GGML_CPU_FP32_TO_FP16(d * (acc[0] + acc[1] + acc[2] + acc[3]));
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

#if defined(__VXE__) || defined(__VXE2__)
    __vector float acc = vec_splats(0.0f);

    const __vector uint8_t v_m = vec_splats((const uint8_t)0x0F);
    const __vector int8_t  v_s = vec_splats( (const int8_t)0x08);

    for (; ib < nb; ++ib) {
        const __vector uint8_t v_x = vec_xl(0, x[ib].qs);
        const __vector int8_t v_xl = (const __vector int8_t)(v_x & v_m);
        const __vector int8_t v_xh = (const __vector int8_t)(v_x >> 4);

        const __vector int8_t v_xls = vec_sub(v_xl, v_s);
        const __vector int8_t v_xhs = vec_sub(v_xh, v_s);

        const __vector int8_t v_yl = vec_xl(0      , y[ib].qs);
        const __vector int8_t v_yh = vec_xl(QK8_0/2, y[ib].qs);

        const __vector int16_t v_xylso = vec_mulo(v_xls, v_yl);
        const __vector int16_t v_xylse = vec_mule(v_xls, v_yl);
        const __vector int16_t v_xyhso = vec_mulo(v_xhs, v_yh);
        const __vector int16_t v_xyhse = vec_mule(v_xhs, v_yh);

        __vector int16_t v_xy_ = v_xylso + v_xylse + v_xyhso + v_xyhse; v_xy_ += vec_reve(v_xy_);

        const __vector float v_xy = vec_float(vec_unpackh(v_xy_));
        const __vector float v_d = vec_splats(GGML_CPU_FP16_TO_FP32(x[ib].d) * GGML_CPU_FP16_TO_FP32(y[ib].d));

        acc = vec_madd(v_xy, v_d, acc);
    }

    sumf = acc[0] + acc[1] + acc[2] + acc[3];

    *s = sumf;
#else
    UNUSED(nb);
    UNUSED(x);
    UNUSED(y);
    UNUSED(ib);
    UNUSED(sumf);
    ggml_vec_dot_q4_0_q8_0_generic(n, s, bs, vx, bx, vy, by, nrc);
#endif
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

#if defined(__VXE__) || defined(__VXE2__)
    float summs = 0;
    float32x4_t acc = vec_splats(0.0f);

    const uint8x16_t v_m = vec_splat_u8(0x0F);

#pragma GCC unroll 4
    for (; ib < nb; ++ib) {
        __builtin_prefetch(x[ib].qs, 0, 1);
        __builtin_prefetch(y[ib].qs, 0, 1);

        summs += GGML_CPU_FP16_TO_FP32(x[ib].m) * GGML_CPU_FP16_TO_FP32(y[ib].s);

        const uint8x16_t v_x = vec_xl(0, x[ib].qs);
        const int8x16_t v_xl = (const int8x16_t)(v_x & v_m);
        const int8x16_t v_xh = (const int8x16_t)(v_x >> 4);

        const int8x16_t v_yl = vec_xl(0      , y[ib].qs);
        const int8x16_t v_yh = vec_xl(QK8_1/2, y[ib].qs);

        const int32x4_t v_xy_ = ggml_vec_dot(ggml_vec_dot(vec_splats(0), v_xl, v_yl), v_xh, v_yh);
        const float32x4_t v_xy = vec_float(v_xy_);

        const float32x4_t v_d = vec_splats(GGML_CPU_FP16_TO_FP32(x[ib].d) * GGML_CPU_FP16_TO_FP32(y[ib].d));

        acc = vec_madd(v_xy, v_d, acc);
    }

    sumf = acc[0] + acc[1] + acc[2] + acc[3] + summs;

    *s = sumf;
#else
    UNUSED(nb);
    UNUSED(x);
    UNUSED(y);
    UNUSED(ib);
    UNUSED(sumf);
    ggml_vec_dot_q4_1_q8_1_generic(n, s, bs, vx, bx, vy, by, nrc);
#endif
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

#if defined(__VXE__) || defined(__VXE2__)
    __vector float acc = vec_splats(0.0f);

#pragma GCC unroll 8
    for (; ib < nb; ++ib) {
        __builtin_prefetch(x[ib].qs, 0, 1);
        __builtin_prefetch(y[ib].qs, 0, 1);

        const int8x16_t v_xl = vec_xl(0      , x[ib].qs);
        const int8x16_t v_xh = vec_xl(QK8_0/2, x[ib].qs);
        const int8x16_t v_yl = vec_xl(0      , y[ib].qs);
        const int8x16_t v_yh = vec_xl(QK8_0/2, y[ib].qs);

        const int32x4_t v_xy_ = ggml_vec_dot(ggml_vec_dot(vec_splats(0), v_xl, v_yl), v_xh, v_yh);
        const float32x4_t v_xy = vec_float(v_xy_);
        const float32x4_t v_d = vec_splats(GGML_CPU_FP16_TO_FP32(x[ib].d) * GGML_CPU_FP16_TO_FP32(y[ib].d));

        acc = vec_madd(v_xy, v_d, acc);
    }

    sumf = acc[0] + acc[1] + acc[2] + acc[3];

    *s = sumf;
#else
    UNUSED(nb);
    UNUSED(x);
    UNUSED(y);
    UNUSED(ib);
    UNUSED(sumf);
    ggml_vec_dot_q8_0_q8_0_generic(n, s, bs, vx, bx, vy, by, nrc);
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

#if defined(__VXE__) || defined(__VXE2__)
    uint32_t aux[3];
    uint32_t utmp[4];

    const int32x4_t v_z = vec_splat_s32(0);
    const uint8x16_t v_3m = vec_splat_u8(0x03);

    const uint8x16_t v_0c = vec_splat_u8(1);
    const uint8x16_t v_1c = vec_sl(v_0c, 1);
    const uint8x16_t v_2c = vec_sl(v_0c, 2);
    const uint8x16_t v_3c = vec_sl(v_0c, 3);

    uint8x16_t q3h[4];
    uint8x16_t q3b[2];
    int8x16_t q3bytes[4];
    int8x16_t q8bytes[4];
    uint8x16_t qhbits[2];

    float sum = 0;

    for (int i = 0; i < nb; ++i) {
        const float d = y[i].d * GGML_CPU_FP16_TO_FP32(x[i].d);

        const uint8_t * restrict x0l = x[i].qs;
        const uint8_t * restrict x0h = x[i].hmask;
        const int8_t  * restrict y0  = y[i].qs;

        qhbits[0] = vec_xl(0 , x0h);
        qhbits[1] = vec_xl(16, x0h);

        int32_t isum = 0;

        memcpy(aux, x[i].scales, 12);
        utmp[3] = ((aux[1] >> 4) & kmask2) | (((aux[2] >> 6) & kmask1) << 4);
        utmp[2] = ((aux[0] >> 4) & kmask2) | (((aux[2] >> 4) & kmask1) << 4);
        utmp[1] = (aux[1] & kmask2) | (((aux[2] >> 2) & kmask1) << 4);
        utmp[0] = (aux[0] & kmask2) | (((aux[2] >> 0) & kmask1) << 4);

        int8_t * scale = (int8_t *)utmp;
        for (int j = 0; j < 16; ++j) scale[j] -= 32;

        for (int j = 0; j < QK_K/128; ++j) {
            int32x4_t isum0, isum1, isum2, isum3;

            q3b[0] = vec_xl(0 , x0l);
            q3b[1] = vec_xl(16, x0l);
            x0l += 32;

            q8bytes[0] = vec_xl(0  , y0);
            q8bytes[1] = vec_xl(16 , y0);
            q8bytes[2] = vec_xl(32 , y0);
            q8bytes[3] = vec_xl(48 , y0);
            q8bytes[4] = vec_xl(64 , y0);
            q8bytes[5] = vec_xl(80 , y0);
            q8bytes[6] = vec_xl(96 , y0);
            q8bytes[7] = vec_xl(112, y0);
            y0 += 128;

            q3h[0] = vec_sl(vec_andc(v_0c, qhbits[0]), 2);
            q3h[1] = vec_sl(vec_andc(v_0c, qhbits[1]), 2);
            q3h[2] = vec_sl(vec_andc(v_1c, qhbits[0]), 1);
            q3h[3] = vec_sl(vec_andc(v_1c, qhbits[1]), 1);

            q3bytes[0] = vec_sub((int8x16_t)vec_and(q3b[0], v_3m), (int8x16_t)q3h[0]);
            q3bytes[1] = vec_sub((int8x16_t)vec_and(q3b[1], v_3m), (int8x16_t)q3h[1]);
            q3bytes[2] = vec_sub((int8x16_t)vec_and(vec_sr(q3b[0], 2), v_3m), (int8x16_t)q3h[2]);
            q3bytes[3] = vec_sub((int8x16_t)vec_and(vec_sr(q3b[1], 2), v_3m), (int8x16_t)q3h[3]);

            isum0 = ggml_vec_dot(v_z, q3bytes[0], q8bytes[0]);
            isum1 = ggml_vec_dot(v_z, q3bytes[1], q8bytes[1]);
            isum2 = ggml_vec_dot(v_z, q3bytes[2], q8bytes[2]);
            isum3 = ggml_vec_dot(v_z, q3bytes[3], q8bytes[3]);

            isum += (isum0[0] + isum0[1] + isum0[2] + isum0[3]) * scale[0];
            isum += (isum1[0] + isum1[1] + isum1[2] + isum1[3]) * scale[1];
            isum += (isum2[0] + isum2[1] + isum2[2] + isum2[3]) * scale[2];
            isum += (isum3[0] + isum3[1] + isum3[2] + isum3[3]) * scale[3];

            scale += 4;

            q3h[0] = vec_andc(v_2c, qhbits[0]);
            q3h[1] = vec_andc(v_2c, qhbits[1]);
            q3h[2] = vec_sr(vec_andc(v_3c, qhbits[0]), 1);
            q3h[3] = vec_sr(vec_andc(v_3c, qhbits[1]), 1);

            q3bytes[0] = vec_sub((int8x16_t)vec_and(vec_sr(q3b[0], 4), v_3m), (int8x16_t)q3h[0]);
            q3bytes[1] = vec_sub((int8x16_t)vec_and(vec_sr(q3b[1], 4), v_3m), (int8x16_t)q3h[1]);
            q3bytes[2] = vec_sub((int8x16_t)vec_and(vec_sr(q3b[0], 6), v_3m), (int8x16_t)q3h[2]);
            q3bytes[3] = vec_sub((int8x16_t)vec_and(vec_sr(q3b[1], 6), v_3m), (int8x16_t)q3h[3]);

            isum0 = ggml_vec_dot(v_z, q3bytes[0], q8bytes[4]);
            isum1 = ggml_vec_dot(v_z, q3bytes[1], q8bytes[5]);
            isum2 = ggml_vec_dot(v_z, q3bytes[2], q8bytes[6]);
            isum3 = ggml_vec_dot(v_z, q3bytes[3], q8bytes[7]);

            isum += (isum0[0] + isum0[1] + isum0[2] + isum0[3]) * scale[0];
            isum += (isum1[0] + isum1[1] + isum1[2] + isum1[3]) * scale[1];
            isum += (isum2[0] + isum2[1] + isum2[2] + isum2[3]) * scale[2];
            isum += (isum3[0] + isum3[1] + isum3[2] + isum3[3]) * scale[3];

            scale += 4;

            if (j == 0) {
                qhbits[0] = vec_sr(qhbits[0], 4);
                qhbits[1] = vec_sr(qhbits[1], 4);
            }
        }

        sum += d * isum;
    }

    *s = sum;

#else
    UNUSED(kmask1);
    UNUSED(kmask2);
    UNUSED(x);
    UNUSED(y);
    UNUSED(nb);
    ggml_vec_dot_q3_K_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
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

#if defined(__VXE__) || defined(__VXE2__)
    const uint8x16_t v_lm = vec_splat_u8(0x0F);
    const int32x4_t v_z = vec_splat_s32(0);

    uint8x16_t v_x[2];
    int8x16_t  v_xl[2];
    int8x16_t  v_y[2];

    float sumf = 0;

    for (int i = 0; i < nb; ++i) {
        const float d = y[i].d * GGML_CPU_FP16_TO_FP32(x[i].d);
        const float dmin = y[i].d * GGML_CPU_FP16_TO_FP32(x[i].dmin);

        const int16x8_t v_ysumsl = vec_xl(0 , y[i].bsums);
        const int16x8_t v_ysumsh = vec_xl(16, y[i].bsums);
        const int16x8_t v_ysums = vec_padd_s16(v_ysumsl, v_ysumsh);

        memcpy(utmp, x[i].scales, 12);

        uint32x4_t v_mins8 = { 0 };
        v_mins8 = vec_insert(utmp[1] & kmask1, v_mins8, 0);
        v_mins8 = vec_insert(((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4), v_mins8, 1);

        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[0] &= kmask1;

        const int16x8_t v_minsh = (int16x8_t)vec_unpackh((uint8x16_t)v_mins8);

        const int32x4_t v_minso = vec_mulo(v_ysums, v_minsh);
        const int32x4_t v_minse = vec_mule(v_ysums, v_minsh);
        const int32x4_t v_mins = v_minso + v_minse;
        sumf -= dmin * (v_mins[0] + v_mins[1] + v_mins[2] + v_mins[3]);

        const uint8_t * scales = (const uint8_t *)utmp;
        const uint8_t * GGML_RESTRICT x0 = x[i].qs;
        const int8_t  * GGML_RESTRICT y0 = y[i].qs;

        int32_t sumi1 = 0;
        int32_t sumi2 = 0;

        for (int j = 0; j < QK_K/64; ++j) {
            v_x[0] = vec_xl(0 , x0);
            v_x[1] = vec_xl(16, x0);
            x0 += 32;

            v_y[0] = vec_xl(0 , y0);
            v_y[1] = vec_xl(16, y0);
            y0 += 32;

            v_xl[0] = (int8x16_t)vec_and(v_x[0], v_lm);
            v_xl[1] = (int8x16_t)vec_and(v_x[1], v_lm);

            const int32x4_t p1 = ggml_vec_dot(ggml_vec_dot(v_z, v_xl[0], v_y[0]), v_xl[1], v_y[1]);
            sumi1 += (p1[0] + p1[1] + p1[2] + p1[3]) * scales[2*j+0];

            v_y[0] = vec_xl(0 , y0);
            v_y[1] = vec_xl(16, y0);
            y0 += 32;

            v_xl[0] = (int8x16_t)vec_sr(v_x[0], 4);
            v_xl[1] = (int8x16_t)vec_sr(v_x[1], 4);

            const int32x4_t p2 = ggml_vec_dot(ggml_vec_dot(v_z, v_xl[0], v_y[0]), v_xl[1], v_y[1]);
            sumi2 += (p2[0] + p2[1] + p2[2] + p2[3]) * scales[2*j+1];
        }

        sumf += d * (sumi1 + sumi2);
    }

    *s = sumf;

#else
    UNUSED(x);
    UNUSED(y);
    UNUSED(nb);
    UNUSED(kmask1);
    UNUSED(kmask2);
    UNUSED(kmask3);
    UNUSED(utmp);
    ggml_vec_dot_q4_K_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
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

#if defined(__VXE__) || defined(__VXE2__)
    const uint8x16_t v_lm = vec_splat_u8(0x0F);
    const uint8x16_t v_1m = vec_splat_u8(0x01);
    const uint8x16_t v_2m = vec_splat_u8(0x02);

    const int32x4_t v_z = vec_splat_s32(0);

    const uchar8x16_t v_minsm = {
        0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F,
        0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF
    };

    int8x16_t  q5b[4];
    uint8x16_t q5h[4];

    uint8x16_t v_xl[2];
    uint8x16_t v_xh[2];
    int8x16_t  v_y[4];

    float sumf = 0;

    for (int i = 0; i < nb; ++i) {
        const float d = y[i].d * GGML_CPU_FP16_TO_FP32(x[i].d);
        const float dmin = y[i].d * GGML_CPU_FP16_TO_FP32(x[i].dmin);

        const int16x8_t v_ysumsl = vec_xl(0 , y[i].bsums);
        const int16x8_t v_ysumsh = vec_xl(16, y[i].bsums);
        const int16x8_t v_ysums = vec_padd_s16(v_ysumsl, v_ysumsh);

        memcpy(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        const uint8x16_t v_mins16 = vec_xl(0, (const uint8_t *)utmp);
        const uint8x16_t v_mins8 = vec_perm(v_mins16, v_mins16, v_minsm);
        const int16x8_t v_minsh = (int16x8_t)vec_unpackh(v_mins8);

        const int32x4_t v_minsho = vec_mulo(v_ysums, v_minsh);
        const int32x4_t v_minshe = vec_mule(v_ysums, v_minsh);
        const int32x4_t v_mins = vec_add(v_minsho, v_minshe);
        const int32_t mins = v_mins[0] + v_mins[1] + v_mins[2] + v_mins[3];

        const uint8_t * scales = (const uint8_t *)utmp;
        const uint8_t * GGML_RESTRICT x0l = x[i].qs;
        const uint8_t * GGML_RESTRICT x0h = x[i].qh;
        const int8_t  * GGML_RESTRICT y0 = y[i].qs;

        v_xh[0] = vec_xl(0 , x0h);
        v_xh[1] = vec_xl(16, x0h);

        int32_t sumi = 0;
        for (int j = 0; j < QK_K/64; ++j) {
            v_xl[0] = vec_xl(0 , x0l);
            v_xl[1] = vec_xl(16, x0l);
            x0l += 32;

            v_y[0] = vec_xl(0 , y0);
            v_y[1] = vec_xl(16, y0);
            v_y[2] = vec_xl(32, y0);
            v_y[3] = vec_xl(48, y0);
            y0 += 64;

            q5h[0] = vec_sl(vec_and(v_1m, v_xh[0]), 4);
            q5h[1] = vec_sl(vec_and(v_1m, v_xh[1]), 4);
            q5h[2] = vec_sl(vec_and(v_2m, v_xh[0]), 3);
            q5h[3] = vec_sl(vec_and(v_2m, v_xh[1]), 3);
            v_xh[0] = vec_sr(v_xh[0], 2);
            v_xh[1] = vec_sr(v_xh[1], 2);

            q5b[0] = (int8x16_t)vec_or(vec_and(v_xl[0], v_lm), q5h[0]);
            q5b[1] = (int8x16_t)vec_or(vec_and(v_xl[1], v_lm), q5h[1]);
            q5b[2] = (int8x16_t)vec_or(vec_sr(v_xl[0], 4), q5h[2]);
            q5b[3] = (int8x16_t)vec_or(vec_sr(v_xl[1], 4), q5h[3]);

            int32x4_t sumi0 = ggml_vec_dot(ggml_vec_dot(v_z, q5b[0], v_y[0]), q5b[1], v_y[1]);
            int32x4_t sumi1 = ggml_vec_dot(ggml_vec_dot(v_z, q5b[2], v_y[2]), q5b[3], v_y[3]);

            sumi += (sumi0[0] + sumi0[1] + sumi0[2] + sumi0[3]) * *scales++;
            sumi += (sumi1[0] + sumi1[1] + sumi1[2] + sumi1[3]) * *scales++;
        }

        sumf += d * sumi - dmin * mins;
    }

    *s = sumf;

#else
    UNUSED(x);
    UNUSED(y);
    UNUSED(nb);
    UNUSED(kmask1);
    UNUSED(kmask2);
    UNUSED(kmask3);
    UNUSED(utmp);
    ggml_vec_dot_q5_K_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
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

#if defined(__VXE__) || defined(__VXE2__)
    float sum = 0;

    // Lower 4-bit and upper 2-bit masks
    const uint8x16_t v_lm = vec_splat_u8(0x0F);
    const uint8x16_t v_um = vec_splat_u8(0x03);

    const int32x4_t v_z = vec_splat_s32(0);

    int8x16_t  q6b[4];
    uint8x16_t q6h[4];

    uint8x16_t v_xl[4];
    uint8x16_t v_xh[2];
    int8x16_t  v_y[4];

    for (int i = 0; i < nb; ++i) {
        const float d_all = GGML_CPU_FP16_TO_FP32(x[i].d);

        const uint8_t * GGML_RESTRICT x0l = x[i].ql;
        const uint8_t * GGML_RESTRICT x0h = x[i].qh;
        const int8_t  * GGML_RESTRICT y0 = y[i].qs;

        const int8_t  * GGML_RESTRICT scale = x[i].scales;

        const int16x8_t v_ysumsl = vec_xl(0 , y[i].bsums);
        const int16x8_t v_ysumsh = vec_xl(16, y[i].bsums);

        const int8x16_t v_scale  = vec_xl(0, scale);
        const int16x8_t v_scalel = vec_unpackh(v_scale);
        const int16x8_t v_scaleh = vec_unpackl(v_scale);

        const int32x4_t v_minslo = vec_mulo(v_ysumsl, v_scalel);
        const int32x4_t v_minsle = vec_mule(v_ysumsl, v_scalel);
        const int32x4_t v_minsho = vec_mulo(v_ysumsh, v_scaleh);
        const int32x4_t v_minshe = vec_mule(v_ysumsh, v_scaleh);
        const int32x4_t v_mins = v_minslo + v_minsle + v_minsho + v_minshe;

        const int32_t mins = v_mins[0] + v_mins[1] + v_mins[2] + v_mins[3];

        int32_t isum = 0;
        for (int j = 0; j < QK_K/128; ++j) {
            // Load model upper 2 bits
            v_xh[0] = vec_xl(0 , x0h);
            v_xh[1] = vec_xl(16, x0h);
            x0h += 32;

            // Load model lower 4 bits
            v_xl[0] = vec_xl(0 , x0l);
            v_xl[1] = vec_xl(16, x0l);
            v_xl[2] = vec_xl(32, x0l);
            v_xl[3] = vec_xl(48, x0l);
            x0l += 64;

            // Load activation quants
            v_y[0] = vec_xl(0 , y0);
            v_y[1] = vec_xl(16, y0);
            v_y[2] = vec_xl(32, y0);
            v_y[3] = vec_xl(48, y0);
            y0 += 64;

            q6h[0] = vec_sl(vec_and(v_um, v_xh[0]), 4);
            q6h[1] = vec_sl(vec_and(v_um, v_xh[1]), 4);
            uint8x16_t shifted = vec_sr(v_xh[0], 2);
            q6h[2] = vec_sl(vec_and(v_um, shifted), 4);
            shifted = vec_sr(v_xh[1], 2);
            q6h[3] = vec_sl(vec_and(v_um, shifted), 4);

            q6b[0] = (int8x16_t)(vec_or(vec_and(v_xl[0], v_lm), q6h[0]));
            q6b[1] = (int8x16_t)(vec_or(vec_and(v_xl[1], v_lm), q6h[1]));
            q6b[2] = (int8x16_t)(vec_or(vec_and(v_xl[2], v_lm), q6h[2]));
            q6b[3] = (int8x16_t)(vec_or(vec_and(v_xl[3], v_lm), q6h[3]));

            int32x4_t summs0 = ggml_vec_dot(v_z, q6b[0], v_y[0]);
            int32x4_t summs1 = ggml_vec_dot(v_z, q6b[1], v_y[1]);
            int32x4_t summs2 = ggml_vec_dot(v_z, q6b[2], v_y[2]);
            int32x4_t summs3 = ggml_vec_dot(v_z, q6b[3], v_y[3]);

            isum += (summs0[0] + summs0[1] + summs0[2] + summs0[3]) * scale[0] +
                    (summs1[0] + summs1[1] + summs1[2] + summs1[3]) * scale[1] +
                    (summs2[0] + summs2[1] + summs2[2] + summs2[3]) * scale[2] +
                    (summs3[0] + summs3[1] + summs3[2] + summs3[3]) * scale[3];

            scale += 4;


            // Load activation quants
            v_y[0] = vec_xl(0 , y0);
            v_y[1] = vec_xl(16, y0);
            v_y[2] = vec_xl(32, y0);
            v_y[3] = vec_xl(48, y0);
            y0 += 64;

            shifted = vec_sr(v_xh[0], 4);
            q6h[0] = vec_sl(vec_and(v_um, shifted), 4);
            shifted = vec_sr(v_xh[1], 4);
            q6h[1] = vec_sl(vec_and(v_um, shifted), 4);
            shifted = vec_sr(v_xh[0], 6);
            q6h[2] = vec_sl(vec_and(v_um, shifted), 4);
            shifted = vec_sr(v_xh[1], 6);
            q6h[3] = vec_sl(vec_and(v_um, shifted), 4);

            q6b[0] = (int8x16_t)(vec_or(vec_sr(v_xl[0], 4), q6h[0]));
            q6b[1] = (int8x16_t)(vec_or(vec_sr(v_xl[1], 4), q6h[1]));
            q6b[2] = (int8x16_t)(vec_or(vec_sr(v_xl[2], 4), q6h[2]));
            q6b[3] = (int8x16_t)(vec_or(vec_sr(v_xl[3], 4), q6h[3]));

            summs0 = ggml_vec_dot(v_z, q6b[0], v_y[0]);
            summs1 = ggml_vec_dot(v_z, q6b[1], v_y[1]);
            summs2 = ggml_vec_dot(v_z, q6b[2], v_y[2]);
            summs3 = ggml_vec_dot(v_z, q6b[3], v_y[3]);

            isum += (summs0[0] + summs0[1] + summs0[2] + summs0[3]) * scale[0] +
                    (summs1[0] + summs1[1] + summs1[2] + summs1[3]) * scale[1] +
                    (summs2[0] + summs2[1] + summs2[2] + summs2[3]) * scale[2] +
                    (summs3[0] + summs3[1] + summs3[2] + summs3[3]) * scale[3];

            scale += 4;
        }

        sum += d_all * y[i].d * (isum - 32 * mins);
    }

    *s = sum;

#else
    UNUSED(x);
    UNUSED(y);
    UNUSED(nb);
    ggml_vec_dot_q6_K_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#endif
}

// #if defined(__VXE__) || defined(__VXE2__)
// static const int8_t keven_signs_q2xs[1024] = {
//      1,  1,  1,  1,  1,  1,  1,  1, -1,  1,  1,  1,  1,  1,  1, -1,  1, -1,  1,  1,  1,  1,  1, -1, -1, -1,  1,  1,  1,  1,  1,  1,
//      1,  1, -1,  1,  1,  1,  1, -1, -1,  1, -1,  1,  1,  1,  1,  1,  1, -1, -1,  1,  1,  1,  1,  1, -1, -1, -1,  1,  1,  1,  1, -1,
//      1,  1,  1, -1,  1,  1,  1, -1, -1,  1,  1, -1,  1,  1,  1,  1,  1, -1,  1, -1,  1,  1,  1,  1, -1, -1,  1, -1,  1,  1,  1, -1,
//      1,  1, -1, -1,  1,  1,  1,  1, -1,  1, -1, -1,  1,  1,  1, -1,  1, -1, -1, -1,  1,  1,  1, -1, -1, -1, -1, -1,  1,  1,  1,  1,
//      1,  1,  1,  1, -1,  1,  1, -1, -1,  1,  1,  1, -1,  1,  1,  1,  1, -1,  1,  1, -1,  1,  1,  1, -1, -1,  1,  1, -1,  1,  1, -1,
//      1,  1, -1,  1, -1,  1,  1,  1, -1,  1, -1,  1, -1,  1,  1, -1,  1, -1, -1,  1, -1,  1,  1, -1, -1, -1, -1,  1, -1,  1,  1,  1,
//      1,  1,  1, -1, -1,  1,  1,  1, -1,  1,  1, -1, -1,  1,  1, -1,  1, -1,  1, -1, -1,  1,  1, -1, -1, -1,  1, -1, -1,  1,  1,  1,
//      1,  1, -1, -1, -1,  1,  1, -1, -1,  1, -1, -1, -1,  1,  1,  1,  1, -1, -1, -1, -1,  1,  1,  1, -1, -1, -1, -1, -1,  1,  1, -1,
//      1,  1,  1,  1,  1, -1,  1, -1, -1,  1,  1,  1,  1, -1,  1,  1,  1, -1,  1,  1,  1, -1,  1,  1, -1, -1,  1,  1,  1, -1,  1, -1,
//      1,  1, -1,  1,  1, -1,  1,  1, -1,  1, -1,  1,  1, -1,  1, -1,  1, -1, -1,  1,  1, -1,  1, -1, -1, -1, -1,  1,  1, -1,  1,  1,
//      1,  1,  1, -1,  1, -1,  1,  1, -1,  1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1, -1, -1,  1, -1,  1, -1,  1,  1,
//      1,  1, -1, -1,  1, -1,  1, -1, -1,  1, -1, -1,  1, -1,  1,  1,  1, -1, -1, -1,  1, -1,  1,  1, -1, -1, -1, -1,  1, -1,  1, -1,
//      1,  1,  1,  1, -1, -1,  1,  1, -1,  1,  1,  1, -1, -1,  1, -1,  1, -1,  1,  1, -1, -1,  1, -1, -1, -1,  1,  1, -1, -1,  1,  1,
//      1,  1, -1,  1, -1, -1,  1, -1, -1,  1, -1,  1, -1, -1,  1,  1,  1, -1, -1,  1, -1, -1,  1,  1, -1, -1, -1,  1, -1, -1,  1, -1,
//      1,  1,  1, -1, -1, -1,  1, -1, -1,  1,  1, -1, -1, -1,  1,  1,  1, -1,  1, -1, -1, -1,  1,  1, -1, -1,  1, -1, -1, -1,  1, -1,
//      1,  1, -1, -1, -1, -1,  1,  1, -1,  1, -1, -1, -1, -1,  1, -1,  1, -1, -1, -1, -1, -1,  1, -1, -1, -1, -1, -1, -1, -1,  1,  1,
//      1,  1,  1,  1,  1,  1, -1, -1, -1,  1,  1,  1,  1,  1, -1,  1,  1, -1,  1,  1,  1,  1, -1,  1, -1, -1,  1,  1,  1,  1, -1, -1,
//      1,  1, -1,  1,  1,  1, -1,  1, -1,  1, -1,  1,  1,  1, -1, -1,  1, -1, -1,  1,  1,  1, -1, -1, -1, -1, -1,  1,  1,  1, -1,  1,
//      1,  1,  1, -1,  1,  1, -1,  1, -1,  1,  1, -1,  1,  1, -1, -1,  1, -1,  1, -1,  1,  1, -1, -1, -1, -1,  1, -1,  1,  1, -1,  1,
//      1,  1, -1, -1,  1,  1, -1, -1, -1,  1, -1, -1,  1,  1, -1,  1,  1, -1, -1, -1,  1,  1, -1,  1, -1, -1, -1, -1,  1,  1, -1, -1,
//      1,  1,  1,  1, -1,  1, -1,  1, -1,  1,  1,  1, -1,  1, -1, -1,  1, -1,  1,  1, -1,  1, -1, -1, -1, -1,  1,  1, -1,  1, -1,  1,
//      1,  1, -1,  1, -1,  1, -1, -1, -1,  1, -1,  1, -1,  1, -1,  1,  1, -1, -1,  1, -1,  1, -1,  1, -1, -1, -1,  1, -1,  1, -1, -1,
//      1,  1,  1, -1, -1,  1, -1, -1, -1,  1,  1, -1, -1,  1, -1,  1,  1, -1,  1, -1, -1,  1, -1,  1, -1, -1,  1, -1, -1,  1, -1, -1,
//      1,  1, -1, -1, -1,  1, -1,  1, -1,  1, -1, -1, -1,  1, -1, -1,  1, -1, -1, -1, -1,  1, -1, -1, -1, -1, -1, -1, -1,  1, -1,  1,
//      1,  1,  1,  1,  1, -1, -1,  1, -1,  1,  1,  1,  1, -1, -1, -1,  1, -1,  1,  1,  1, -1, -1, -1, -1, -1,  1,  1,  1, -1, -1,  1,
//      1,  1, -1,  1,  1, -1, -1, -1, -1,  1, -1,  1,  1, -1, -1,  1,  1, -1, -1,  1,  1, -1, -1,  1, -1, -1, -1,  1,  1, -1, -1, -1,
//      1,  1,  1, -1,  1, -1, -1, -1, -1,  1,  1, -1,  1, -1, -1,  1,  1, -1,  1, -1,  1, -1, -1,  1, -1, -1,  1, -1,  1, -1, -1, -1,
//      1,  1, -1, -1,  1, -1, -1,  1, -1,  1, -1, -1,  1, -1, -1, -1,  1, -1, -1, -1,  1, -1, -1, -1, -1, -1, -1, -1,  1, -1, -1,  1,
//      1,  1,  1,  1, -1, -1, -1, -1, -1,  1,  1,  1, -1, -1, -1,  1,  1, -1,  1,  1, -1, -1, -1,  1, -1, -1,  1,  1, -1, -1, -1, -1,
//      1,  1, -1,  1, -1, -1, -1,  1, -1,  1, -1,  1, -1, -1, -1, -1,  1, -1, -1,  1, -1, -1, -1, -1, -1, -1, -1,  1, -1, -1, -1,  1,
//      1,  1,  1, -1, -1, -1, -1,  1, -1,  1,  1, -1, -1, -1, -1, -1,  1, -1,  1, -1, -1, -1, -1, -1, -1, -1,  1, -1, -1, -1, -1,  1,
//      1,  1, -1, -1, -1, -1, -1, -1, -1,  1, -1, -1, -1, -1, -1,  1,  1, -1, -1, -1, -1, -1, -1,  1, -1, -1, -1, -1, -1, -1, -1, -1,
// };
// #endif

// void ggml_vec_dot_iq2_xxs_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
//     assert(n % QK_K == 0);
//     assert(nrc == 1);
//     UNUSED(nrc);
//     UNUSED(bx);
//     UNUSED(by);
//     UNUSED(bs);

//     const block_iq2_xxs * GGML_RESTRICT x = vx;
//     const block_q8_K    * GGML_RESTRICT y = vy;

//     const int nb = n / QK_K;

// #if defined(__VXE__) || defined(__VXE2__)
//    const uint64_t * signs64 = (const uint64_t *)keven_signs_q2xs;

//    uint32_t aux32[4];
//    const uint8_t * aux8 = (const uint8_t *)aux32;

//    float sumf = 0;

//    for (int i = 0; i < nb; ++i) {
//        const float d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;
//        const uint16_t * GGML_RESTRICT q2 = x[i].qs;
//        const int8_t   * GGML_RESTRICT q8 = y[i].qs;

//        float sumf1 = 0, sumf2 = 0;

//        for (int ib32 = 0; ib32 < QK_K/32; ib += 2) {
//            int8x16_t q8b0 = vec_xl( 0, q8);
//            int8x16_t qb81 = vec_xl(16, q8);
//            int8x16_t q8b2 = vec_xl(32, q8);
//            int8x16_t q8b3 = vec_xl(48, q8);
//            q8 += 64;

//            memcpy(aux32, q2, 4 * sizeof(uint32_t));
//            q2 += 8;

//            int8x16_t q2u0 = { *(const int64_t *)(iq2xxs_grid + aux8[ 0]), *(const int64_t *)(iq2xxs_grid + aux8[ 1]) };
//            int8x16_t q2u1 = { *(const int64_t *)(iq2xxs_grid + aux8[ 2]), *(const int64_t *)(iq2xxs_grid + aux8[ 3]) };
//            int8x16_t q2u2 = { *(const int64_t *)(iq2xxs_grid + aux8[ 8]), *(const int64_t *)(iq2xxs_grid + aux8[ 9]) };
//            int8x16_t q2u3 = { *(const int64_t *)(iq2xxs_grid + aux8[10]), *(const int64_t *)(iq2xxs_grid + aux8[11]) };

//            int8x16_t q2s0 = { *(const int64_t *)(signs64 + ((aux32[1] >>  0) & 127)), *(const int64_t *)(signs64 + ((aux32[1] >>  7) & 127)) };
//            int8x16_t q2s1 = { *(const int64_t *)(signs64 + ((aux32[1] >> 14) & 127)), *(const int64_t *)(signs64 + ((aux32[1] >> 21) & 127)) };
//            int8x16_t q2s2 = { *(const int64_t *)(signs64 + ((aux32[3] >>  0) & 127)), *(const int64_t *)(signs64 + ((aux32[3] >>  7) & 127)) };
//            int8x16_t q2s3 = { *(const int64_t *)(signs64 + ((aux32[3] >> 14) & 127)), *(const int64_t *)(signs64 + ((aux32[3] >> 21) & 127)) };

//            q2u0 = vec_mul(q2u0, q2s0);
//            q2u1 = vec_mul(q2u1, q2s1);
//            q2u2 = vec_mul(q2u2, q2s2);
//            q2u3 = vec_mul(q2u3, q2s3);

//            const int32x4_t p1 = ggml_vec_dot(ggml_vec_dot(vec_splat_s32(0), q2u0, q8b0), q2u1, q8b1);
//            const int32x4_t p2 = ggml_vec_dot(ggml_vec_dot(vec_splat_s32(0), q2u2, q8b2), q2u3, q8b3);

//            sumf1 += (p1[0] + p1[1] + p1[2] + p1[3]) * (0.5f + (aux32[1] >> 28));
//            sumf2 += (p2[0] + p2[1] + p2[2] + p2[3]) * (0.5f + (aux32[3] >> 28));
//        }

//        sumf += d * (sumf1 + sumf2);
//    }

//    *s = 0.25f * sumf;

// #else

//     uint32_t aux32[2];
//     const uint8_t * aux8 = (const uint8_t *)aux32;

//     float sumf = 0.f;
//     for (int i = 0; i < nb; ++i) {
//         const float d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;
//         const uint16_t * GGML_RESTRICT q2 = x[i].qs;
//         const int8_t   * GGML_RESTRICT q8 = y[i].qs;
//         int32_t bsum = 0;
//         for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {
//             memcpy(aux32, q2, 2*sizeof(uint32_t));
//             q2 += 4;
//             const uint32_t ls = 2*(aux32[1] >> 28) + 1;
//             int32_t sumi = 0;
//             for (int l = 0; l < 4; ++l) {
//                 const uint8_t * grid = (const uint8_t *)(iq2xxs_grid + aux8[l]);
//                 const uint8_t  signs = ksigns_iq2xs[(aux32[1] >> 7*l) & 127];
//                 for (int j = 0; j < 8; ++j) {
//                     sumi += grid[j] * q8[j] * (signs & kmask_iq2xs[j] ? -1 : 1);
//                 }
//                 q8 += 8;
//             }
//             bsum += sumi * ls;
//         }
//         sumf += d * bsum;
//     }
//     *s = 0.125f * sumf;
// #endif
// }

void ggml_vec_dot_iq4_nl_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);
    assert(n % QK4_NL == 0);
    static_assert(QK4_NL == QK8_0, "QK4_NL and QK8_0 must be the same");

    const block_iq4_nl * GGML_RESTRICT x = vx;
    const block_q8_0   * GGML_RESTRICT y = vy;

    const int nb = n / QK4_NL;

    int ib = 0;
    float sumf = 0;

#if defined(__VXE__) || defined(__VXE2__)
    const int8x16_t v_k = vec_xl(0, kvalues_iq4nl);
    const uint8x16_t v_m = vec_splat_u8(0x0F);

    for (; ib < nb; ++ib) {
        const block_iq4_nl * GGML_RESTRICT x0 = &x[ib];
        const block_q8_0   * GGML_RESTRICT y0 = &y[ib];

        const uint8x16_t v_x = vec_xl(0, x0->qs);
        int8x16_t v_xl = (int8x16_t)vec_and(v_x, v_m);
        int8x16_t v_xh = (int8x16_t)vec_sr(v_x, 4);

        v_xl = vec_perm(v_k, v_k, (uchar8x16_t)v_xl);
        v_xh = vec_perm(v_k, v_k, (uchar8x16_t)v_xh);

        const int8x16_t v_yl = vec_xl(0      , y0->qs);
        const int8x16_t v_yh = vec_xl(QK8_0/2, y0->qs);
        const int32x4_t v_xy = ggml_vec_dot(ggml_vec_dot(vec_splats(0), v_xl, v_yl), v_xh, v_yh);

        sumf += GGML_CPU_FP16_TO_FP32(x0->d) * GGML_CPU_FP16_TO_FP32(y0->d) * (v_xy[0] + v_xy[1] + v_xy[2] + v_xy[3]);
    }

    *s = sumf;
#else
    UNUSED(x);
    UNUSED(y);
    UNUSED(nb);
    UNUSED(ib);
    UNUSED(sumf);
    ggml_vec_dot_iq4_nl_q8_0_generic(n, s, bs, vx, bx, vy, by, nrc);
#endif
}

void ggml_vec_dot_iq4_xs_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);
    assert(n % QK_K == 0);

    const block_iq4_xs * GGML_RESTRICT x = vx;
    const block_q8_K   * GGML_RESTRICT y = vy;

    const int nb = n / QK_K;

#if defined(__VXE__) || defined(__VXE2__)
    const int8x16_t v_k = vec_xl(0, kvalues_iq4nl);
    const uint8x16_t v_m = vec_splat_u8(0x0F);

    float sumf = 0;

    for (int ibl = 0; ibl < nb; ++ibl) {
        const uint8_t * GGML_RESTRICT q4 = x[ibl].qs;
        const int8_t  * GGML_RESTRICT q8 = y[ibl].qs;

        uint16_t h = x[ibl].scales_h;

        int sumi1 = 0, sumi2 = 0;
        for (int ib = 0; ib < QK_K/64; ++ib) {
            const uint8x16_t v_x0 = vec_xl(0       , q4);
            const uint8x16_t v_x1 = vec_xl(QK4_NL/2, q4);
            q4 += 32;

            int8x16_t v_x0l = (int8x16_t)vec_and(v_x0, v_m);
            int8x16_t v_x0h = (int8x16_t)vec_sr(v_x0, 4);
            int8x16_t v_x1l = (int8x16_t)vec_and(v_x1, v_m);
            int8x16_t v_x1h = (int8x16_t)vec_sr(v_x1, 4);

            v_x0l = vec_perm(v_k, v_k, (uchar8x16_t)v_x0l);
            v_x0h = vec_perm(v_k, v_k, (uchar8x16_t)v_x0h);
            v_x1l = vec_perm(v_k, v_k, (uchar8x16_t)v_x1l);
            v_x1h = vec_perm(v_k, v_k, (uchar8x16_t)v_x1h);

            const int8x16_t v_y0 = vec_xl( 0, q8);
            const int8x16_t v_y1 = vec_xl(16, q8);
            const int8x16_t v_y2 = vec_xl(32, q8);
            const int8x16_t v_y3 = vec_xl(48, q8);
            q8 += 64;

            int32x4_t vsumi0 = ggml_vec_dot(ggml_vec_dot(vec_splats(0), v_x0l, v_y0), v_x0h, v_y1);
            int32x4_t vsumi1 = ggml_vec_dot(ggml_vec_dot(vec_splats(0), v_x1l, v_y2), v_x1h, v_y3);

            int ls1 = ((x[ibl].scales_l[ib] & 0xF) | ((h << 4) & 0x30)) - 32;
            int ls2 = ((x[ibl].scales_l[ib] >>  4) | ((h << 2) & 0x30)) - 32;

            h >>= 4;

            sumi1 += (vsumi0[0] + vsumi0[1] + vsumi0[2] + vsumi0[3]) * ls1;
            sumi2 += (vsumi1[0] + vsumi1[1] + vsumi1[2] + vsumi1[3]) * ls2;
        }

        sumf += GGML_CPU_FP16_TO_FP32(x[ibl].d) * y[ibl].d * (sumi1 + sumi2);
    }

    *s = sumf;

#else
    UNUSED(x);
    UNUSED(y);
    UNUSED(nb);
    ggml_vec_dot_iq4_xs_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#endif
}

