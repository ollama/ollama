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

#if defined(__ARM_NEON)
#define B1(c,s,n)  0x ## n ## c ,  0x ## n ## s
#define B2(c,s,n) B1(c,s,n ## c), B1(c,s,n ## s)
#define B3(c,s,n) B2(c,s,n ## c), B2(c,s,n ## s)
#define B4(c,s,n) B3(c,s,n ## c), B3(c,s,n ## s)
#define B5(c,s,n) B4(c,s,n ## c), B4(c,s,n ## s)
#define B6(c,s,n) B5(c,s,n ## c), B5(c,s,n ## s)
#define B7(c,s,n) B6(c,s,n ## c), B6(c,s,n ## s)
#define B8(c,s  ) B7(c,s,     c), B7(c,s,     s)

// precomputed tables for expanding 8bits to 8 bytes:
static const uint64_t table_b2b_0[1 << 8] = { B8(00, 10) }; // ( b) << 4
static const uint64_t table_b2b_1[1 << 8] = { B8(10, 00) }; // (!b) << 4
#endif

void quantize_row_q8_0(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k) {
    assert(QK8_0 == 32);
    assert(k % QK8_0 == 0);
    const int nb = k / QK8_0;

    block_q8_0 * GGML_RESTRICT y = vy;

#if defined(__ARM_NEON)
    for (int i = 0; i < nb; i++) {
        float32x4_t srcv [8];
        float32x4_t asrcv[8];
        float32x4_t amaxv[8];

        for (int j = 0; j < 8; j++) srcv[j]  = vld1q_f32(x + i*32 + 4*j);
        for (int j = 0; j < 8; j++) asrcv[j] = vabsq_f32(srcv[j]);

        for (int j = 0; j < 4; j++) amaxv[2*j] = vmaxq_f32(asrcv[2*j], asrcv[2*j+1]);
        for (int j = 0; j < 2; j++) amaxv[4*j] = vmaxq_f32(amaxv[4*j], amaxv[4*j+2]);
        for (int j = 0; j < 1; j++) amaxv[8*j] = vmaxq_f32(amaxv[8*j], amaxv[8*j+4]);

        const float amax = vmaxvq_f32(amaxv[0]);

        const float d = amax / ((1 << 7) - 1);
        const float id = d ? 1.0f/d : 0.0f;

        y[i].d = GGML_CPU_FP32_TO_FP16(d);

        for (int j = 0; j < 8; j++) {
            const float32x4_t v  = vmulq_n_f32(srcv[j], id);
            const int32x4_t   vi = vcvtnq_s32_f32(v);

            y[i].qs[4*j + 0] = vgetq_lane_s32(vi, 0);
            y[i].qs[4*j + 1] = vgetq_lane_s32(vi, 1);
            y[i].qs[4*j + 2] = vgetq_lane_s32(vi, 2);
            y[i].qs[4*j + 3] = vgetq_lane_s32(vi, 3);
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
#if defined(__ARM_NEON)
    for (int i = 0; i < nb; i++) {
        float32x4_t srcv [8];
        float32x4_t asrcv[8];
        float32x4_t amaxv[8];

        for (int j = 0; j < 8; j++) srcv[j]  = vld1q_f32(x + i*32 + 4*j);
        for (int j = 0; j < 8; j++) asrcv[j] = vabsq_f32(srcv[j]);

        for (int j = 0; j < 4; j++) amaxv[2*j] = vmaxq_f32(asrcv[2*j], asrcv[2*j+1]);
        for (int j = 0; j < 2; j++) amaxv[4*j] = vmaxq_f32(amaxv[4*j], amaxv[4*j+2]);
        for (int j = 0; j < 1; j++) amaxv[8*j] = vmaxq_f32(amaxv[8*j], amaxv[8*j+4]);

        const float amax = vmaxvq_f32(amaxv[0]);

        const float d = amax / ((1 << 7) - 1);
        const float id = d ? 1.0f/d : 0.0f;

        y[i].d = GGML_CPU_FP32_TO_FP16(d);

        int32x4_t accv = vdupq_n_s32(0);

        for (int j = 0; j < 8; j++) {
            const float32x4_t v  = vmulq_n_f32(srcv[j], id);
            const int32x4_t   vi = vcvtnq_s32_f32(v);

            y[i].qs[4*j + 0] = vgetq_lane_s32(vi, 0);
            y[i].qs[4*j + 1] = vgetq_lane_s32(vi, 1);
            y[i].qs[4*j + 2] = vgetq_lane_s32(vi, 2);
            y[i].qs[4*j + 3] = vgetq_lane_s32(vi, 3);

            accv = vaddq_s32(accv, vi);
        }

        y[i].s = GGML_CPU_FP32_TO_FP16(d * vaddvq_s32(accv));
    }
#else
    GGML_UNUSED(nb);
    // scalar
    quantize_row_q8_1_ref(x, y, k);
#endif
}

// placeholder implementation for Apple targets
void quantize_row_q8_K(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
    quantize_row_q8_K_ref(x, y, k);
}

//===================================== Dot products =================================

void ggml_vec_dot_q4_0_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    const int qk = QK8_0;
    const int nb = n / qk;

    assert(n % qk == 0);
#if defined(__ARM_FEATURE_MATMUL_INT8)
    assert((nrc == 2) || (nrc == 1));
#else
    assert(nrc == 1);
#endif
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_q4_0 * GGML_RESTRICT x = vx;
    const block_q8_0 * GGML_RESTRICT y = vy;

#if defined(__ARM_FEATURE_MATMUL_INT8)
    if (nrc == 2) {
        const block_q4_0 * GGML_RESTRICT vx0 = vx;
        const block_q4_0 * GGML_RESTRICT vx1 = (const block_q4_0 *) ((const uint8_t*)vx + bx);
        const block_q8_0 * GGML_RESTRICT vy0 = vy;
        const block_q8_0 * GGML_RESTRICT vy1 = (const block_q8_0 *) ((const uint8_t*)vy + by);

        float32x4_t sumv0 = vdupq_n_f32(0.0f);

        for (int i = 0; i < nb; i++) {
            const block_q4_0 * GGML_RESTRICT b_x0 = &vx0[i];
            const block_q4_0 * GGML_RESTRICT b_x1 = &vx1[i];
            const block_q8_0 * GGML_RESTRICT b_y0 = &vy0[i];
            const block_q8_0 * GGML_RESTRICT b_y1 = &vy1[i];

            const uint8x16_t m4b = vdupq_n_u8(0x0F);
            const int8x16_t  s8b = vdupq_n_s8(0x8);

            const uint8x16_t v0_0 = vld1q_u8(b_x0->qs);
            const uint8x16_t v0_1 = vld1q_u8(b_x1->qs);

            // 4-bit -> 8-bit
            const int8x16_t v0_0l = vreinterpretq_s8_u8(vandq_u8  (v0_0, m4b));
            const int8x16_t v0_0h = vreinterpretq_s8_u8(vshrq_n_u8(v0_0, 4));
            const int8x16_t v0_1l = vreinterpretq_s8_u8(vandq_u8  (v0_1, m4b));
            const int8x16_t v0_1h = vreinterpretq_s8_u8(vshrq_n_u8(v0_1, 4));

            // sub 8
            const int8x16_t x0_l = vsubq_s8(v0_0l, s8b);
            const int8x16_t x0_h = vsubq_s8(v0_0h, s8b);
            const int8x16_t x1_l = vsubq_s8(v0_1l, s8b);
            const int8x16_t x1_h = vsubq_s8(v0_1h, s8b);

            // load y
            const int8x16_t y0_l = vld1q_s8(b_y0->qs);
            const int8x16_t y0_h = vld1q_s8(b_y0->qs + 16);
            const int8x16_t y1_l = vld1q_s8(b_y1->qs);
            const int8x16_t y1_h = vld1q_s8(b_y1->qs + 16);

            float32_t _scale[4] = {
                GGML_CPU_FP16_TO_FP32(b_x0->d)*GGML_CPU_FP16_TO_FP32(b_y0->d),
                GGML_CPU_FP16_TO_FP32(b_x0->d)*GGML_CPU_FP16_TO_FP32(b_y1->d),
                GGML_CPU_FP16_TO_FP32(b_x1->d)*GGML_CPU_FP16_TO_FP32(b_y0->d),
                GGML_CPU_FP16_TO_FP32(b_x1->d)*GGML_CPU_FP16_TO_FP32(b_y1->d)
            };
            float32x4_t scale = vld1q_f32(_scale);

            int8x16_t l0 = vreinterpretq_s8_s64(vzip1q_s64(vreinterpretq_s64_s8(x0_l), vreinterpretq_s64_s8(x1_l)));
            int8x16_t l1 = vreinterpretq_s8_s64(vzip2q_s64(vreinterpretq_s64_s8(x0_l), vreinterpretq_s64_s8(x1_l)));

            int8x16_t l2 = vreinterpretq_s8_s64(vzip1q_s64(vreinterpretq_s64_s8(x0_h), vreinterpretq_s64_s8(x1_h)));
            int8x16_t l3 = vreinterpretq_s8_s64(vzip2q_s64(vreinterpretq_s64_s8(x0_h), vreinterpretq_s64_s8(x1_h)));

            int8x16_t r0 = vreinterpretq_s8_s64(vzip1q_s64(vreinterpretq_s64_s8(y0_l), vreinterpretq_s64_s8(y1_l)));
            int8x16_t r1 = vreinterpretq_s8_s64(vzip2q_s64(vreinterpretq_s64_s8(y0_l), vreinterpretq_s64_s8(y1_l)));

            int8x16_t r2 = vreinterpretq_s8_s64(vzip1q_s64(vreinterpretq_s64_s8(y0_h), vreinterpretq_s64_s8(y1_h)));
            int8x16_t r3 = vreinterpretq_s8_s64(vzip2q_s64(vreinterpretq_s64_s8(y0_h), vreinterpretq_s64_s8(y1_h)));

            sumv0 = vmlaq_f32(sumv0,(vcvtq_f32_s32(vmmlaq_s32((vmmlaq_s32((vmmlaq_s32((vmmlaq_s32(vdupq_n_s32(0), l0, r0)),
                                                l1, r1)), l2, r2)), l3, r3))), scale);
        }

        float32x4_t sumv1 = vextq_f32 (sumv0, sumv0, 2);
        float32x4_t sumv2 = vzip1q_f32(sumv0, sumv1);

        vst1_f32(s,      vget_low_f32 (sumv2));
        vst1_f32(s + bs, vget_high_f32(sumv2));

        return;
    }
#endif

    int ib = 0;
    float sumf = 0;

#if defined(__ARM_FEATURE_SVE)
    svfloat32_t sumv0 = svdup_n_f32(0.0f);
    svfloat32_t sumv1 = svdup_n_f32(0.0f);

    const int vector_length = ggml_cpu_get_sve_cnt()*8;

    // VLA Implementation using switch case
    switch (vector_length) {
        case 128:
            {
                // predicate for activating higher lanes for 4 float32 elements
                const svbool_t ph4 = svptrue_pat_b32(SV_VL4);

                for (; ib + 1 < nb; ib += 2) {
                    const block_q4_0 * GGML_RESTRICT x0 = &x[ib + 0];
                    const block_q4_0 * GGML_RESTRICT x1 = &x[ib + 1];
                    const block_q8_0 * GGML_RESTRICT y0 = &y[ib + 0];
                    const block_q8_0 * GGML_RESTRICT y1 = &y[ib + 1];

                    // load x
                    const svuint8_t qx0r = svld1rq_u8(svptrue_b8(), x0->qs);
                    const svuint8_t qx1r = svld1rq_u8(svptrue_b8(), x1->qs);

                    // 4-bit -> 8-bit
                    const svint8_t qx0l = svreinterpret_s8_u8(svand_n_u8_m(svptrue_b8(), qx0r, 0x0F));
                    const svint8_t qx0h = svreinterpret_s8_u8(svlsr_n_u8_m(svptrue_b8(), qx0r, 0x04));
                    const svint8_t qx1l = svreinterpret_s8_u8(svand_n_u8_m(svptrue_b8(), qx1r, 0x0F));
                    const svint8_t qx1h = svreinterpret_s8_u8(svlsr_n_u8_m(svptrue_b8(), qx1r, 0x04));

                    // sub 8
                    const svint8_t qx0ls = svsub_n_s8_x(svptrue_b8(), qx0h, 8);
                    const svint8_t qx0hs = svsub_n_s8_x(svptrue_b8(), qx0l, 8);
                    const svint8_t qx1ls = svsub_n_s8_x(svptrue_b8(), qx1h, 8);
                    const svint8_t qx1hs = svsub_n_s8_x(svptrue_b8(), qx1l, 8);

                    // load y
                    const svint8_t qy0h = svld1_s8(svptrue_b8(), y0->qs);
                    const svint8_t qy0l = svld1_s8(svptrue_b8(), y0->qs + 16);
                    const svint8_t qy1h = svld1_s8(svptrue_b8(), y1->qs);
                    const svint8_t qy1l = svld1_s8(svptrue_b8(), y1->qs + 16);

                    // dot product
                    sumv0 = svmla_n_f32_x(ph4, sumv0, svcvt_f32_s32_x(ph4, svadd_x(ph4,
                                    svdot_s32(svdup_n_s32(0), qx0ls, qy0l),
                                    svdot_s32(svdup_n_s32(0), qx0hs, qy0h))), GGML_CPU_FP16_TO_FP32(x0->d)*GGML_CPU_FP16_TO_FP32(y0->d));
                    sumv1 = svmla_n_f32_x(ph4, sumv1, svcvt_f32_s32_x(ph4, svadd_x(ph4,
                                    svdot_s32(svdup_n_s32(0), qx1ls, qy1l),
                                    svdot_s32(svdup_n_s32(0), qx1hs, qy1h))), GGML_CPU_FP16_TO_FP32(x1->d)*GGML_CPU_FP16_TO_FP32(y1->d));
                }

                sumf = svaddv_f32(svptrue_b32(), svadd_f32_x(svptrue_b32(), sumv0, sumv1));
            } break;
        case 256:
            {
                // predicate for activating higher lanes for 16 int8 elements
                const svbool_t ph16 = svptrue_pat_b8(SV_VL16);
                // predicate for activating lower lanes for  16 int8 elements
                const svbool_t pl16 = svnot_b_z(svptrue_b8(), ph16);

                for (; ib + 1 < nb; ib += 2) {
                    const block_q4_0 * GGML_RESTRICT x0 = &x[ib + 0];
                    const block_q4_0 * GGML_RESTRICT x1 = &x[ib + 1];
                    const block_q8_0 * GGML_RESTRICT y0 = &y[ib + 0];
                    const block_q8_0 * GGML_RESTRICT y1 = &y[ib + 1];

                    // load x
                    const svuint8_t qx0r = svld1rq_u8(svptrue_b8(), x0->qs);
                    const svuint8_t qx1r = svld1rq_u8(svptrue_b8(), x1->qs);

                    // 4-bit -> 8-bit
                    const svint8_t qx0 = svreinterpret_s8_u8(svlsr_n_u8_m(pl16, svand_n_u8_m(ph16, qx0r, 0x0F), 0x04));
                    const svint8_t qx1 = svreinterpret_s8_u8(svlsr_n_u8_m(pl16, svand_n_u8_m(ph16, qx1r, 0x0F), 0x04));

                    // sub 8
                    const svint8_t qx0s = svsub_n_s8_x(svptrue_b8(), qx0, 8);
                    const svint8_t qx1s = svsub_n_s8_x(svptrue_b8(), qx1, 8);

                    // load y
                    const svint8_t qy0 = svld1_s8(svptrue_b8(), y0->qs);
                    const svint8_t qy1 = svld1_s8(svptrue_b8(), y1->qs);

                    // dot product
                    sumv0 = svmla_n_f32_x(svptrue_b32(), sumv0, svcvt_f32_s32_x(svptrue_b32(),
                                svdot_s32(svdup_n_s32(0), qx0s, qy0)), GGML_CPU_FP16_TO_FP32(x0->d)*GGML_CPU_FP16_TO_FP32(y0->d));
                    sumv1 = svmla_n_f32_x(svptrue_b32(), sumv1, svcvt_f32_s32_x(svptrue_b32(),
                                svdot_s32(svdup_n_s32(0), qx1s, qy1)), GGML_CPU_FP16_TO_FP32(x1->d)*GGML_CPU_FP16_TO_FP32(y1->d));
                }

                sumf = svaddv_f32(svptrue_b32(), svadd_f32_x(svptrue_b32(), sumv0, sumv1));
            } break;
        case 512:
            {
                // predicate for activating higher lanes for 32 int8 elements
                const svbool_t ph32 = svptrue_pat_b8(SV_VL32);

                // predicate for activating higher lanes for 16 int8 elements
                const svbool_t ph16 = svptrue_pat_b8(SV_VL16);
                // predicate for activating lower lanes for 16 int8 elements from first 32 int8 activated lanes
                const svbool_t pl16 = svnot_b_z(ph32, ph16);

                for (; ib + 1 < nb; ib += 2) {
                    const block_q4_0 * GGML_RESTRICT x0 = &x[ib + 0];
                    const block_q4_0 * GGML_RESTRICT x1 = &x[ib + 1];
                    const block_q8_0 * GGML_RESTRICT y0 = &y[ib + 0];
                    const block_q8_0 * GGML_RESTRICT y1 = &y[ib + 1];

                    // load x
                    const svuint8_t qx0r = svld1rq_u8(ph32, x0->qs);
                    const svuint8_t qx1r = svld1rq_u8(ph32, x1->qs);

                    // 4-bit -> 8-bit
                    const svint8_t qx0 = svreinterpret_s8_u8(svlsr_n_u8_m(pl16, svand_n_u8_m(ph16, qx0r, 0x0F), 0x04));
                    const svint8_t qx1 = svreinterpret_s8_u8(svlsr_n_u8_m(pl16, svand_n_u8_m(ph16, qx1r, 0x0F), 0x04));

                    // sub 8
                    const svint8_t qx0s = svsub_n_s8_x(ph32, qx0, 8);
                    const svint8_t qx1s = svsub_n_s8_x(ph32, qx1, 8);

                    // load y
                    const svint8_t qy0 = svld1_s8(ph32, y0->qs);
                    const svint8_t qy1 = svld1_s8(ph32, y1->qs);

                    // dot product
                    sumv0 = svmla_n_f32_x(ph32, sumv0, svcvt_f32_s32_x(ph32,
                                svdot_s32(svdup_n_s32(0), qx0s, qy0)), GGML_CPU_FP16_TO_FP32(x0->d)*GGML_CPU_FP16_TO_FP32(y0->d));
                    sumv1 = svmla_n_f32_x(ph32, sumv1, svcvt_f32_s32_x(ph32,
                                svdot_s32(svdup_n_s32(0), qx1s, qy1)), GGML_CPU_FP16_TO_FP32(x1->d)*GGML_CPU_FP16_TO_FP32(y1->d));
                }

                sumf = svaddv_f32(ph32, svadd_f32_x(ph32, sumv0, sumv1));
            } break;
        default:
            assert(false && "Unsupported vector length");
            break;
    }

#elif defined(__ARM_NEON)
    float32x4_t sumv0 = vdupq_n_f32(0.0f);
    float32x4_t sumv1 = vdupq_n_f32(0.0f);

    for (; ib + 1 < nb; ib += 2) {
        const block_q4_0 * GGML_RESTRICT x0 = &x[ib + 0];
        const block_q4_0 * GGML_RESTRICT x1 = &x[ib + 1];
        const block_q8_0 * GGML_RESTRICT y0 = &y[ib + 0];
        const block_q8_0 * GGML_RESTRICT y1 = &y[ib + 1];

        const uint8x16_t m4b = vdupq_n_u8(0x0F);
        const int8x16_t  s8b = vdupq_n_s8(0x8);

        const uint8x16_t v0_0 = vld1q_u8(x0->qs);
        const uint8x16_t v0_1 = vld1q_u8(x1->qs);

        // 4-bit -> 8-bit
        const int8x16_t v0_0l = vreinterpretq_s8_u8(vandq_u8  (v0_0, m4b));
        const int8x16_t v0_0h = vreinterpretq_s8_u8(vshrq_n_u8(v0_0, 4));
        const int8x16_t v0_1l = vreinterpretq_s8_u8(vandq_u8  (v0_1, m4b));
        const int8x16_t v0_1h = vreinterpretq_s8_u8(vshrq_n_u8(v0_1, 4));

        // sub 8
        const int8x16_t v0_0ls = vsubq_s8(v0_0l, s8b);
        const int8x16_t v0_0hs = vsubq_s8(v0_0h, s8b);
        const int8x16_t v0_1ls = vsubq_s8(v0_1l, s8b);
        const int8x16_t v0_1hs = vsubq_s8(v0_1h, s8b);

        // load y
        const int8x16_t v1_0l = vld1q_s8(y0->qs);
        const int8x16_t v1_0h = vld1q_s8(y0->qs + 16);
        const int8x16_t v1_1l = vld1q_s8(y1->qs);
        const int8x16_t v1_1h = vld1q_s8(y1->qs + 16);

        // dot product into int32x4_t
        const int32x4_t p_0 = ggml_vdotq_s32(ggml_vdotq_s32(vdupq_n_s32(0), v0_0ls, v1_0l), v0_0hs, v1_0h);
        const int32x4_t p_1 = ggml_vdotq_s32(ggml_vdotq_s32(vdupq_n_s32(0), v0_1ls, v1_1l), v0_1hs, v1_1h);

        sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(p_0), GGML_CPU_FP16_TO_FP32(x0->d)*GGML_CPU_FP16_TO_FP32(y0->d));
        sumv1 = vmlaq_n_f32(sumv1, vcvtq_f32_s32(p_1), GGML_CPU_FP16_TO_FP32(x1->d)*GGML_CPU_FP16_TO_FP32(y1->d));
    }

    sumf = vaddvq_f32(sumv0) + vaddvq_f32(sumv1);
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
#if defined(__ARM_FEATURE_MATMUL_INT8)
    assert((nrc == 2) || (nrc == 1));
#else
    assert(nrc == 1);
#endif
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_q4_1 * GGML_RESTRICT x = vx;
    const block_q8_1 * GGML_RESTRICT y = vy;

#if defined(__ARM_FEATURE_MATMUL_INT8)
    if (nrc == 2) {
        const block_q4_1 * GGML_RESTRICT vx0 = vx;
        const block_q4_1 * GGML_RESTRICT vx1 = (const block_q4_1 *) ((const uint8_t*)vx + bx);
        const block_q8_1 * GGML_RESTRICT vy0 = vy;
        const block_q8_1 * GGML_RESTRICT vy1 = (const block_q8_1 *) ((const uint8_t*)vy + by);

        float32x4_t sumv0 = vdupq_n_f32(0.0f);
        float32x4_t summs0 = vdupq_n_f32(0.0f);

        for (int i = 0; i < nb; i++) {
            const block_q4_1 * GGML_RESTRICT b_x0 = &vx0[i];
            const block_q4_1 * GGML_RESTRICT b_x1 = &vx1[i];
            const block_q8_1 * GGML_RESTRICT b_y0 = &vy0[i];
            const block_q8_1 * GGML_RESTRICT b_y1 = &vy1[i];

            float32_t summs_t[4] = {
                GGML_CPU_FP16_TO_FP32(b_x0->m) * GGML_CPU_FP16_TO_FP32(b_y0->s),
                GGML_CPU_FP16_TO_FP32(b_x1->m) * GGML_CPU_FP16_TO_FP32(b_y0->s),
                GGML_CPU_FP16_TO_FP32(b_x0->m) * GGML_CPU_FP16_TO_FP32(b_y1->s),
                GGML_CPU_FP16_TO_FP32(b_x1->m) * GGML_CPU_FP16_TO_FP32(b_y1->s)
            };
            summs0 = vaddq_f32(summs0, vld1q_f32(summs_t));

            const uint8x16_t m4b = vdupq_n_u8(0x0F);

            const uint8x16_t v0_0 = vld1q_u8(b_x0->qs);
            const uint8x16_t v0_1 = vld1q_u8(b_x1->qs);

            // 4-bit -> 8-bit
            const int8x16_t x0_l = vreinterpretq_s8_u8(vandq_u8  (v0_0, m4b));
            const int8x16_t x0_h = vreinterpretq_s8_u8(vshrq_n_u8(v0_0, 4));
            const int8x16_t x1_l = vreinterpretq_s8_u8(vandq_u8  (v0_1, m4b));
            const int8x16_t x1_h = vreinterpretq_s8_u8(vshrq_n_u8(v0_1, 4));

            // load y
            const int8x16_t y0_l = vld1q_s8(b_y0->qs);
            const int8x16_t y0_h = vld1q_s8(b_y0->qs + 16);
            const int8x16_t y1_l = vld1q_s8(b_y1->qs);
            const int8x16_t y1_h = vld1q_s8(b_y1->qs + 16);

            // mmla into int32x4_t
            float32_t _scale[4] = {
                GGML_CPU_FP16_TO_FP32(b_x0->d)*GGML_CPU_FP16_TO_FP32(b_y0->d),
                GGML_CPU_FP16_TO_FP32(b_x0->d)*GGML_CPU_FP16_TO_FP32(b_y1->d),
                GGML_CPU_FP16_TO_FP32(b_x1->d)*GGML_CPU_FP16_TO_FP32(b_y0->d),
                GGML_CPU_FP16_TO_FP32(b_x1->d)*GGML_CPU_FP16_TO_FP32(b_y1->d)
            };
            float32x4_t scale = vld1q_f32(_scale);

            int8x16_t l0 = vreinterpretq_s8_s64(vzip1q_s64(vreinterpretq_s64_s8(x0_l), vreinterpretq_s64_s8(x1_l)));
            int8x16_t l1 = vreinterpretq_s8_s64(vzip2q_s64(vreinterpretq_s64_s8(x0_l), vreinterpretq_s64_s8(x1_l)));

            int8x16_t l2 = vreinterpretq_s8_s64(vzip1q_s64(vreinterpretq_s64_s8(x0_h), vreinterpretq_s64_s8(x1_h)));
            int8x16_t l3 = vreinterpretq_s8_s64(vzip2q_s64(vreinterpretq_s64_s8(x0_h), vreinterpretq_s64_s8(x1_h)));

            int8x16_t r0 = vreinterpretq_s8_s64(vzip1q_s64(vreinterpretq_s64_s8(y0_l), vreinterpretq_s64_s8(y1_l)));
            int8x16_t r1 = vreinterpretq_s8_s64(vzip2q_s64(vreinterpretq_s64_s8(y0_l), vreinterpretq_s64_s8(y1_l)));

            int8x16_t r2 = vreinterpretq_s8_s64(vzip1q_s64(vreinterpretq_s64_s8(y0_h), vreinterpretq_s64_s8(y1_h)));
            int8x16_t r3 = vreinterpretq_s8_s64(vzip2q_s64(vreinterpretq_s64_s8(y0_h), vreinterpretq_s64_s8(y1_h)));
            sumv0 = vmlaq_f32(sumv0,(vcvtq_f32_s32(vmmlaq_s32((vmmlaq_s32((vmmlaq_s32((vmmlaq_s32(vdupq_n_s32(0), l0, r0)),
                                                l1, r1)), l2, r2)), l3, r3))), scale);
        }

        float32x4_t sumv1 = vextq_f32 (sumv0, sumv0, 2);
        float32x4_t sumv2 = vzip1q_f32(sumv0, sumv1);

        sumv2 = vaddq_f32(sumv2, summs0);

        vst1_f32(s,      vget_low_f32 (sumv2));
        vst1_f32(s + bs, vget_high_f32(sumv2));

        return;
    }
#endif

    int ib = 0;
    float sumf = 0;

#if defined(__ARM_NEON)
    float32x4_t sumv0 = vdupq_n_f32(0.0f);
    float32x4_t sumv1 = vdupq_n_f32(0.0f);

    float summs = 0;

    for (; ib + 1 < nb; ib += 2) {
        const block_q4_1 * GGML_RESTRICT x0 = &x[ib + 0];
        const block_q4_1 * GGML_RESTRICT x1 = &x[ib + 1];
        const block_q8_1 * GGML_RESTRICT y0 = &y[ib + 0];
        const block_q8_1 * GGML_RESTRICT y1 = &y[ib + 1];

        summs += GGML_CPU_FP16_TO_FP32(x0->m) * GGML_CPU_FP16_TO_FP32(y0->s) + GGML_CPU_FP16_TO_FP32(x1->m) * GGML_CPU_FP16_TO_FP32(y1->s);

        const uint8x16_t m4b = vdupq_n_u8(0x0F);

        const uint8x16_t v0_0 = vld1q_u8(x0->qs);
        const uint8x16_t v0_1 = vld1q_u8(x1->qs);

        // 4-bit -> 8-bit
        const int8x16_t v0_0l = vreinterpretq_s8_u8(vandq_u8  (v0_0, m4b));
        const int8x16_t v0_0h = vreinterpretq_s8_u8(vshrq_n_u8(v0_0, 4));
        const int8x16_t v0_1l = vreinterpretq_s8_u8(vandq_u8  (v0_1, m4b));
        const int8x16_t v0_1h = vreinterpretq_s8_u8(vshrq_n_u8(v0_1, 4));

        // load y
        const int8x16_t v1_0l = vld1q_s8(y0->qs);
        const int8x16_t v1_0h = vld1q_s8(y0->qs + 16);
        const int8x16_t v1_1l = vld1q_s8(y1->qs);
        const int8x16_t v1_1h = vld1q_s8(y1->qs + 16);

        // dot product into int32x4_t
        const int32x4_t p_0 = ggml_vdotq_s32(ggml_vdotq_s32(vdupq_n_s32(0), v0_0l, v1_0l), v0_0h, v1_0h);
        const int32x4_t p_1 = ggml_vdotq_s32(ggml_vdotq_s32(vdupq_n_s32(0), v0_1l, v1_1l), v0_1h, v1_1h);

        sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(p_0), GGML_CPU_FP16_TO_FP32(x0->d)*GGML_CPU_FP16_TO_FP32(y0->d));
        sumv1 = vmlaq_n_f32(sumv1, vcvtq_f32_s32(p_1), GGML_CPU_FP16_TO_FP32(x1->d)*GGML_CPU_FP16_TO_FP32(y1->d));
    }

    sumf = vaddvq_f32(sumv0) + vaddvq_f32(sumv1) + summs;

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

#if defined(__ARM_NEON)
    float32x4_t sumv0 = vdupq_n_f32(0.0f);
    float32x4_t sumv1 = vdupq_n_f32(0.0f);

    uint32_t qh0;
    uint32_t qh1;

    uint64_t tmp0[4];
    uint64_t tmp1[4];

    for (; ib + 1 < nb; ib += 2) {
        const block_q5_0 * GGML_RESTRICT x0 = &x[ib];
        const block_q5_0 * GGML_RESTRICT x1 = &x[ib + 1];
        const block_q8_0 * GGML_RESTRICT y0 = &y[ib];
        const block_q8_0 * GGML_RESTRICT y1 = &y[ib + 1];

        const uint8x16_t m4b = vdupq_n_u8(0x0F);

        // extract the 5th bit via lookup table ((!b) << 4)
        memcpy(&qh0, x0->qh, sizeof(qh0));
        memcpy(&qh1, x1->qh, sizeof(qh1));

        tmp0[0] = table_b2b_1[(qh0 >>  0) & 0xFF];
        tmp0[1] = table_b2b_1[(qh0 >>  8) & 0xFF];
        tmp0[2] = table_b2b_1[(qh0 >> 16) & 0xFF];
        tmp0[3] = table_b2b_1[(qh0 >> 24)       ];

        tmp1[0] = table_b2b_1[(qh1 >>  0) & 0xFF];
        tmp1[1] = table_b2b_1[(qh1 >>  8) & 0xFF];
        tmp1[2] = table_b2b_1[(qh1 >> 16) & 0xFF];
        tmp1[3] = table_b2b_1[(qh1 >> 24)       ];

        const int8x16_t qhl0 = vld1q_s8((const int8_t *)(tmp0 + 0));
        const int8x16_t qhh0 = vld1q_s8((const int8_t *)(tmp0 + 2));
        const int8x16_t qhl1 = vld1q_s8((const int8_t *)(tmp1 + 0));
        const int8x16_t qhh1 = vld1q_s8((const int8_t *)(tmp1 + 2));

        const uint8x16_t v0_0 = vld1q_u8(x0->qs);
        const uint8x16_t v0_1 = vld1q_u8(x1->qs);

        // 4-bit -> 8-bit
        int8x16_t v0_0l = vreinterpretq_s8_u8(vandq_u8  (v0_0, m4b));
        int8x16_t v0_0h = vreinterpretq_s8_u8(vshrq_n_u8(v0_0, 4));
        int8x16_t v0_1l = vreinterpretq_s8_u8(vandq_u8  (v0_1, m4b));
        int8x16_t v0_1h = vreinterpretq_s8_u8(vshrq_n_u8(v0_1, 4));

        // add high bit and sub 16 (equivalent to sub 0x10 when bit is zero)
        const int8x16_t v0_0lf = vsubq_s8(v0_0l, qhl0);
        const int8x16_t v0_0hf = vsubq_s8(v0_0h, qhh0);
        const int8x16_t v0_1lf = vsubq_s8(v0_1l, qhl1);
        const int8x16_t v0_1hf = vsubq_s8(v0_1h, qhh1);

        // load y
        const int8x16_t v1_0l = vld1q_s8(y0->qs);
        const int8x16_t v1_0h = vld1q_s8(y0->qs + 16);
        const int8x16_t v1_1l = vld1q_s8(y1->qs);
        const int8x16_t v1_1h = vld1q_s8(y1->qs + 16);

        sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(vaddq_s32(
                        ggml_vdotq_s32(vdupq_n_s32(0), v0_0lf, v1_0l),
                        ggml_vdotq_s32(vdupq_n_s32(0), v0_0hf, v1_0h))), GGML_CPU_FP16_TO_FP32(x0->d)*GGML_CPU_FP16_TO_FP32(y0->d));
        sumv1 = vmlaq_n_f32(sumv1, vcvtq_f32_s32(vaddq_s32(
                        ggml_vdotq_s32(vdupq_n_s32(0), v0_1lf, v1_1l),
                        ggml_vdotq_s32(vdupq_n_s32(0), v0_1hf, v1_1h))), GGML_CPU_FP16_TO_FP32(x1->d)*GGML_CPU_FP16_TO_FP32(y1->d));
    }

    sumf = vaddvq_f32(sumv0) + vaddvq_f32(sumv1);

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

#if defined(__ARM_NEON)
    float32x4_t sumv0 = vdupq_n_f32(0.0f);
    float32x4_t sumv1 = vdupq_n_f32(0.0f);

    float summs0 = 0.0f;
    float summs1 = 0.0f;

    uint32_t qh0;
    uint32_t qh1;

    uint64_t tmp0[4];
    uint64_t tmp1[4];

    for (; ib + 1 < nb; ib += 2) {
        const block_q5_1 * GGML_RESTRICT x0 = &x[ib];
        const block_q5_1 * GGML_RESTRICT x1 = &x[ib + 1];
        const block_q8_1 * GGML_RESTRICT y0 = &y[ib];
        const block_q8_1 * GGML_RESTRICT y1 = &y[ib + 1];

        const uint8x16_t m4b = vdupq_n_u8(0x0F);

        summs0 += GGML_CPU_FP16_TO_FP32(x0->m) * GGML_CPU_FP16_TO_FP32(y0->s);
        summs1 += GGML_CPU_FP16_TO_FP32(x1->m) * GGML_CPU_FP16_TO_FP32(y1->s);

        // extract the 5th bit via lookup table ((b) << 4)
        memcpy(&qh0, x0->qh, sizeof(qh0));
        memcpy(&qh1, x1->qh, sizeof(qh1));

        tmp0[0] = table_b2b_0[(qh0 >>  0) & 0xFF];
        tmp0[1] = table_b2b_0[(qh0 >>  8) & 0xFF];
        tmp0[2] = table_b2b_0[(qh0 >> 16) & 0xFF];
        tmp0[3] = table_b2b_0[(qh0 >> 24)       ];

        tmp1[0] = table_b2b_0[(qh1 >>  0) & 0xFF];
        tmp1[1] = table_b2b_0[(qh1 >>  8) & 0xFF];
        tmp1[2] = table_b2b_0[(qh1 >> 16) & 0xFF];
        tmp1[3] = table_b2b_0[(qh1 >> 24)       ];

        const int8x16_t qhl0 = vld1q_s8((const int8_t *)(tmp0 + 0));
        const int8x16_t qhh0 = vld1q_s8((const int8_t *)(tmp0 + 2));
        const int8x16_t qhl1 = vld1q_s8((const int8_t *)(tmp1 + 0));
        const int8x16_t qhh1 = vld1q_s8((const int8_t *)(tmp1 + 2));

        const uint8x16_t v0_0 = vld1q_u8(x0->qs);
        const uint8x16_t v0_1 = vld1q_u8(x1->qs);

        // 4-bit -> 8-bit
        const int8x16_t v0_0l = vreinterpretq_s8_u8(vandq_u8  (v0_0, m4b));
        const int8x16_t v0_0h = vreinterpretq_s8_u8(vshrq_n_u8(v0_0, 4));
        const int8x16_t v0_1l = vreinterpretq_s8_u8(vandq_u8  (v0_1, m4b));
        const int8x16_t v0_1h = vreinterpretq_s8_u8(vshrq_n_u8(v0_1, 4));

        // add high bit
        const int8x16_t v0_0lf = vorrq_s8(v0_0l, qhl0);
        const int8x16_t v0_0hf = vorrq_s8(v0_0h, qhh0);
        const int8x16_t v0_1lf = vorrq_s8(v0_1l, qhl1);
        const int8x16_t v0_1hf = vorrq_s8(v0_1h, qhh1);

        // load y
        const int8x16_t v1_0l = vld1q_s8(y0->qs);
        const int8x16_t v1_0h = vld1q_s8(y0->qs + 16);
        const int8x16_t v1_1l = vld1q_s8(y1->qs);
        const int8x16_t v1_1h = vld1q_s8(y1->qs + 16);

        sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(vaddq_s32(
                        ggml_vdotq_s32(vdupq_n_s32(0), v0_0lf, v1_0l),
                        ggml_vdotq_s32(vdupq_n_s32(0), v0_0hf, v1_0h))), GGML_CPU_FP16_TO_FP32(x0->d)*GGML_CPU_FP16_TO_FP32(y0->d));
        sumv1 = vmlaq_n_f32(sumv1, vcvtq_f32_s32(vaddq_s32(
                        ggml_vdotq_s32(vdupq_n_s32(0), v0_1lf, v1_1l),
                        ggml_vdotq_s32(vdupq_n_s32(0), v0_1hf, v1_1h))), GGML_CPU_FP16_TO_FP32(x1->d)*GGML_CPU_FP16_TO_FP32(y1->d));
    }

    sumf = vaddvq_f32(sumv0) + vaddvq_f32(sumv1) + summs0 + summs1;

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
#if defined(__ARM_FEATURE_MATMUL_INT8)
    assert((nrc == 2) || (nrc == 1));
#else
    assert(nrc == 1);
#endif
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_q8_0 * GGML_RESTRICT x = vx;
    const block_q8_0 * GGML_RESTRICT y = vy;

#if defined(__ARM_FEATURE_MATMUL_INT8)
    if (nrc == 2) {
        const block_q8_0 * GGML_RESTRICT vx0 = vx;
        const block_q8_0 * GGML_RESTRICT vx1 = (const block_q8_0 *) ((const uint8_t*)vx + bx);
        const block_q8_0 * GGML_RESTRICT vy0 = vy;
        const block_q8_0 * GGML_RESTRICT vy1 = (const block_q8_0 *) ((const uint8_t*)vy + by);

        float32x4_t sumv0 = vdupq_n_f32(0.0f);

        for (int i = 0; i < nb; i++) {
            const block_q8_0 * GGML_RESTRICT b_x0 = &vx0[i];
            const block_q8_0 * GGML_RESTRICT b_y0 = &vy0[i];

            const block_q8_0 * GGML_RESTRICT b_x1 = &vx1[i];
            const block_q8_0 * GGML_RESTRICT b_y1 = &vy1[i];

            const int8x16_t x0_l = vld1q_s8(b_x0->qs);
            const int8x16_t x0_h = vld1q_s8(b_x0->qs + 16);
            const int8x16_t x1_l = vld1q_s8(b_x1->qs);
            const int8x16_t x1_h = vld1q_s8(b_x1->qs + 16);

            // load y
            const int8x16_t y0_l = vld1q_s8(b_y0->qs);
            const int8x16_t y0_h = vld1q_s8(b_y0->qs + 16);
            const int8x16_t y1_l = vld1q_s8(b_y1->qs);
            const int8x16_t y1_h = vld1q_s8(b_y1->qs + 16);

            float32_t _scale[4] = {
                GGML_CPU_FP16_TO_FP32(b_x0->d)*GGML_CPU_FP16_TO_FP32(b_y0->d),
                GGML_CPU_FP16_TO_FP32(b_x0->d)*GGML_CPU_FP16_TO_FP32(b_y1->d),
                GGML_CPU_FP16_TO_FP32(b_x1->d)*GGML_CPU_FP16_TO_FP32(b_y0->d),
                GGML_CPU_FP16_TO_FP32(b_x1->d)*GGML_CPU_FP16_TO_FP32(b_y1->d)
            };
            float32x4_t scale = vld1q_f32(_scale);

            int8x16_t l0 = vreinterpretq_s8_s64(vzip1q_s64(vreinterpretq_s64_s8(x0_l), vreinterpretq_s64_s8(x1_l)));
            int8x16_t l1 = vreinterpretq_s8_s64(vzip2q_s64(vreinterpretq_s64_s8(x0_l), vreinterpretq_s64_s8(x1_l)));

            int8x16_t l2 = vreinterpretq_s8_s64(vzip1q_s64(vreinterpretq_s64_s8(x0_h), vreinterpretq_s64_s8(x1_h)));
            int8x16_t l3 = vreinterpretq_s8_s64(vzip2q_s64(vreinterpretq_s64_s8(x0_h), vreinterpretq_s64_s8(x1_h)));

            int8x16_t r0 = vreinterpretq_s8_s64(vzip1q_s64(vreinterpretq_s64_s8(y0_l), vreinterpretq_s64_s8(y1_l)));
            int8x16_t r1 = vreinterpretq_s8_s64(vzip2q_s64(vreinterpretq_s64_s8(y0_l), vreinterpretq_s64_s8(y1_l)));

            int8x16_t r2 = vreinterpretq_s8_s64(vzip1q_s64(vreinterpretq_s64_s8(y0_h), vreinterpretq_s64_s8(y1_h)));
            int8x16_t r3 = vreinterpretq_s8_s64(vzip2q_s64(vreinterpretq_s64_s8(y0_h), vreinterpretq_s64_s8(y1_h)));

            sumv0 = vmlaq_f32(sumv0,(vcvtq_f32_s32(vmmlaq_s32((vmmlaq_s32((vmmlaq_s32((vmmlaq_s32(vdupq_n_s32(0), l0, r0)),
                                                l1, r1)), l2, r2)), l3, r3))), scale);
        }

        float32x4_t sumv1 = vextq_f32 (sumv0, sumv0, 2);
        float32x4_t sumv2 = vzip1q_f32(sumv0, sumv1);

        vst1_f32(s,      vget_low_f32 (sumv2));
        vst1_f32(s + bs, vget_high_f32(sumv2));

        return;
    }
#endif

    int ib = 0;
    float sumf = 0;

#if defined(__ARM_FEATURE_SVE)
    svfloat32_t sumv0 = svdup_n_f32(0.0f);
    svfloat32_t sumv1 = svdup_n_f32(0.0f);

    const int vector_length = ggml_cpu_get_sve_cnt()*8;

    //VLA Implemenation for SVE
    switch (vector_length) {
        case 128:
            {
                // predicate for activating lanes for 16 Int8 elements
                const svbool_t ph16 = svptrue_pat_b8 (SV_VL16);
                const svbool_t pl16 = svptrue_pat_b32(SV_VL4);

                for (; ib + 1 < nb; ib += 2) {
                    const block_q8_0 * GGML_RESTRICT x0 = &x[ib + 0];
                    const block_q8_0 * GGML_RESTRICT x1 = &x[ib + 1];
                    const block_q8_0 * GGML_RESTRICT y0 = &y[ib + 0];
                    const block_q8_0 * GGML_RESTRICT y1 = &y[ib + 1];

                    // load x
                    const svint8_t qx0_0 = svld1_s8(ph16, x0->qs);
                    const svint8_t qx0_1 = svld1_s8(ph16, x0->qs+16);
                    const svint8_t qx1_0 = svld1_s8(ph16, x1->qs);
                    const svint8_t qx1_1 = svld1_s8(ph16, x1->qs+16);

                    // load y
                    const svint8_t qy0_0 = svld1_s8(ph16, y0->qs);
                    const svint8_t qy0_1 = svld1_s8(ph16, y0->qs+16);
                    const svint8_t qy1_0 = svld1_s8(ph16, y1->qs);
                    const svint8_t qy1_1 = svld1_s8(ph16, y1->qs+16);

                    sumv0 = svmla_n_f32_x(pl16, sumv0, svcvt_f32_s32_x(pl16, svadd_x(pl16,
                                    svdot_s32(svdup_n_s32(0), qx0_0, qy0_0),
                                    svdot_s32(svdup_n_s32(0), qx0_1, qy0_1))), GGML_CPU_FP16_TO_FP32(x0->d)*GGML_CPU_FP16_TO_FP32(y0->d));
                    sumv1 = svmla_n_f32_x(pl16, sumv1, svcvt_f32_s32_x(pl16, svadd_x(pl16,
                                    svdot_s32(svdup_n_s32(0), qx1_0, qy1_0),
                                    svdot_s32(svdup_n_s32(0), qx1_1, qy1_1))), GGML_CPU_FP16_TO_FP32(x1->d)*GGML_CPU_FP16_TO_FP32(y1->d));
                }

                sumf = svaddv_f32(pl16, svadd_f32_x(pl16, sumv0, sumv1));
            } break;
        case 256:
            {
                //printf("sve256");
                for (; ib + 1 < nb; ib += 2) {
                    const block_q8_0 * GGML_RESTRICT x0 = &x[ib + 0];
                    const block_q8_0 * GGML_RESTRICT x1 = &x[ib + 1];
                    const block_q8_0 * GGML_RESTRICT y0 = &y[ib + 0];
                    const block_q8_0 * GGML_RESTRICT y1 = &y[ib + 1];

                    // load x
                    const svint8_t qx0 = svld1_s8(svptrue_b8(), x0->qs);
                    const svint8_t qx1 = svld1_s8(svptrue_b8(), x1->qs);

                    // load y
                    const svint8_t qy0 = svld1_s8(svptrue_b8(), y0->qs);
                    const svint8_t qy1 = svld1_s8(svptrue_b8(), y1->qs);

                    sumv0 = svmla_n_f32_x(svptrue_b32(), sumv0, svcvt_f32_s32_x(svptrue_b32(),
                                svdot_s32(svdup_n_s32(0), qx0, qy0)), GGML_CPU_FP16_TO_FP32(x0->d)*GGML_CPU_FP16_TO_FP32(y0->d));
                    sumv1 = svmla_n_f32_x(svptrue_b32(), sumv1, svcvt_f32_s32_x(svptrue_b32(),
                                svdot_s32(svdup_n_s32(0), qx1, qy1)), GGML_CPU_FP16_TO_FP32(x1->d)*GGML_CPU_FP16_TO_FP32(y1->d));
                }

                sumf = svaddv_f32(svptrue_b32(), svadd_f32_x(svptrue_b32(), sumv0, sumv1));
            } break;
        case 512:
            {
                // predicate for activating high 256 bit
                const svbool_t ph32 = svptrue_pat_b8(SV_VL32);
                // predicate for activating low 256 bit
                const svbool_t pl32 = svnot_b_z(svptrue_b8(), ph32);

                // predicate for activating high lanes for 8 float32 elements
                const svbool_t ph8 = svptrue_pat_b32(SV_VL8);
                // predicate for activating low lanes for 8 float32 elements
                const svbool_t pl8 = svnot_b_z(svptrue_b32(), ph8);

                svfloat32_t sumv00 = svdup_n_f32(0.0f);

                for (; ib + 1 < nb; ib += 2) {
                    const block_q8_0 * GGML_RESTRICT x0 = &x[ib + 0];
                    const block_q8_0 * GGML_RESTRICT x1 = &x[ib + 1];
                    const block_q8_0 * GGML_RESTRICT y0 = &y[ib + 0];
                    const block_q8_0 * GGML_RESTRICT y1 = &y[ib + 1];

                    //load 32 int8_t in first half of vector and put another 32 int8_t in second vector lower bits
                    // and add them to make one 64 element vector
                    // load x
                    const svint8_t qx_32 = svld1_s8(ph32, x0->qs);
                          svint8_t qx_64 = svld1_s8(pl32, x0->qs + 2);

                    qx_64 = svadd_s8_x(svptrue_b8(), qx_32, qx_64);

                    // load y
                    const svint8_t qy_32 = svld1_s8(ph32, y0->qs);
                          svint8_t qy_64 = svld1_s8(pl32, y0->qs + 2);

                    qy_64 = svadd_s8_x(svptrue_b8(), qy_32, qy_64);

                    // scale creation
                    const float32_t deq1 = GGML_CPU_FP16_TO_FP32(x0->d)*GGML_CPU_FP16_TO_FP32(y0->d);
                    const float32_t deq2 = GGML_CPU_FP16_TO_FP32(x1->d)*GGML_CPU_FP16_TO_FP32(y1->d);

                    // duplicate deq1 in first half of vector and deq2 in second half of vector
                    const svfloat32_t temp = svdup_f32_m(svdup_f32_z(ph8, deq1), pl8, deq2);

                    const svfloat32_t sumvt = svcvt_f32_s32_x(svptrue_b32(), svdot_s32(svdup_n_s32(0), qx_64, qy_64));

                    sumv00 = svmla_f32_m(svptrue_b32(), sumv00, sumvt, temp);
                }

                sumf = svaddv_f32(svptrue_b32(), sumv00);
                break;
            }
        default:
            assert(false && "Unsupported vector length");
            break;
    }
#elif defined(__ARM_NEON)
    float32x4_t sumv0 = vdupq_n_f32(0.0f);
    float32x4_t sumv1 = vdupq_n_f32(0.0f);

    for (; ib + 1 < nb; ib += 2) {
        const block_q8_0 * GGML_RESTRICT x0 = &x[ib + 0];
        const block_q8_0 * GGML_RESTRICT x1 = &x[ib + 1];
        const block_q8_0 * GGML_RESTRICT y0 = &y[ib + 0];
        const block_q8_0 * GGML_RESTRICT y1 = &y[ib + 1];

        const int8x16_t x0_0 = vld1q_s8(x0->qs);
        const int8x16_t x0_1 = vld1q_s8(x0->qs + 16);
        const int8x16_t x1_0 = vld1q_s8(x1->qs);
        const int8x16_t x1_1 = vld1q_s8(x1->qs + 16);

        // load y
        const int8x16_t y0_0 = vld1q_s8(y0->qs);
        const int8x16_t y0_1 = vld1q_s8(y0->qs + 16);
        const int8x16_t y1_0 = vld1q_s8(y1->qs);
        const int8x16_t y1_1 = vld1q_s8(y1->qs + 16);

        sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(vaddq_s32(
                        ggml_vdotq_s32(vdupq_n_s32(0), x0_0, y0_0),
                        ggml_vdotq_s32(vdupq_n_s32(0), x0_1, y0_1))), GGML_CPU_FP16_TO_FP32(x0->d)*GGML_CPU_FP16_TO_FP32(y0->d));

        sumv1 = vmlaq_n_f32(sumv1, vcvtq_f32_s32(vaddq_s32(
                        ggml_vdotq_s32(vdupq_n_s32(0), x1_0, y1_0),
                        ggml_vdotq_s32(vdupq_n_s32(0), x1_1, y1_1))), GGML_CPU_FP16_TO_FP32(x1->d)*GGML_CPU_FP16_TO_FP32(y1->d));
    }

    sumf = vaddvq_f32(sumv0) + vaddvq_f32(sumv1);
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

void ggml_vec_dot_tq1_0_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_tq1_0 * GGML_RESTRICT x = vx;
    const block_q8_K  * GGML_RESTRICT y = vy;

    const int nb = n / QK_K;

#if defined(__ARM_NEON)
    float sumf = 0.0f;

    uint8_t k_shift[16] = {1, 1, 1, 1, 3, 3, 3, 3, 9, 9, 9, 9, 27, 27, 27, 27};

    const uint8x16_t shift = vld1q_u8(k_shift);

    for (int i = 0; i < nb; ++i) {
#if defined(__ARM_FEATURE_DOTPROD)
        int32x4_t sumi0 = vdupq_n_s32(0);
        int32x4_t sumi1 = vdupq_n_s32(0);
#else
        int16x8_t sumi0 = vdupq_n_s16(0);
        int16x8_t sumi1 = vdupq_n_s16(0);
#endif

        // first 32 bytes of 5 elements
        {
            uint8x16_t qx0 = vld1q_u8(x[i].qs + 0);
            uint8x16_t qx1 = vld1q_u8(x[i].qs + 16);
            uint8x16_t qx2 = vmulq_u8(qx0, vdupq_n_u8(3));
            uint8x16_t qx3 = vmulq_u8(qx1, vdupq_n_u8(3));
            uint8x16_t qx4 = vmulq_u8(qx0, vdupq_n_u8(9));
            uint8x16_t qx5 = vmulq_u8(qx1, vdupq_n_u8(9));
            uint8x16_t qx6 = vmulq_u8(qx0, vdupq_n_u8(27));
            uint8x16_t qx7 = vmulq_u8(qx1, vdupq_n_u8(27));
            uint8x16_t qx8 = vmulq_u8(qx0, vdupq_n_u8(81));
            uint8x16_t qx9 = vmulq_u8(qx1, vdupq_n_u8(81));

            // multiply by 3 and keep the 2 bits above 8 bits
            int8x16_t sqx0 = vreinterpretq_s8_u8(vshrq_n_u8(vhaddq_u8(qx0, vshrq_n_u8(qx0, 1)), 6));
            int8x16_t sqx1 = vreinterpretq_s8_u8(vshrq_n_u8(vhaddq_u8(qx1, vshrq_n_u8(qx1, 1)), 6));
            int8x16_t sqx2 = vreinterpretq_s8_u8(vshrq_n_u8(vhaddq_u8(qx2, vshrq_n_u8(qx2, 1)), 6));
            int8x16_t sqx3 = vreinterpretq_s8_u8(vshrq_n_u8(vhaddq_u8(qx3, vshrq_n_u8(qx3, 1)), 6));
            int8x16_t sqx4 = vreinterpretq_s8_u8(vshrq_n_u8(vhaddq_u8(qx4, vshrq_n_u8(qx4, 1)), 6));
            int8x16_t sqx5 = vreinterpretq_s8_u8(vshrq_n_u8(vhaddq_u8(qx5, vshrq_n_u8(qx5, 1)), 6));
            int8x16_t sqx6 = vreinterpretq_s8_u8(vshrq_n_u8(vhaddq_u8(qx6, vshrq_n_u8(qx6, 1)), 6));
            int8x16_t sqx7 = vreinterpretq_s8_u8(vshrq_n_u8(vhaddq_u8(qx7, vshrq_n_u8(qx7, 1)), 6));
            int8x16_t sqx8 = vreinterpretq_s8_u8(vshrq_n_u8(vhaddq_u8(qx8, vshrq_n_u8(qx8, 1)), 6));
            int8x16_t sqx9 = vreinterpretq_s8_u8(vshrq_n_u8(vhaddq_u8(qx9, vshrq_n_u8(qx9, 1)), 6));

            const int8x16_t qy0 = vld1q_s8(y[i].qs +   0);
            const int8x16_t qy1 = vld1q_s8(y[i].qs +  16);
            const int8x16_t qy2 = vld1q_s8(y[i].qs +  32);
            const int8x16_t qy3 = vld1q_s8(y[i].qs +  48);
            const int8x16_t qy4 = vld1q_s8(y[i].qs +  64);
            const int8x16_t qy5 = vld1q_s8(y[i].qs +  80);
            const int8x16_t qy6 = vld1q_s8(y[i].qs +  96);
            const int8x16_t qy7 = vld1q_s8(y[i].qs + 112);
            const int8x16_t qy8 = vld1q_s8(y[i].qs + 128);
            const int8x16_t qy9 = vld1q_s8(y[i].qs + 144);

#if defined(__ARM_FEATURE_DOTPROD)
            sumi0 = vdotq_s32(sumi0, sqx0, qy0);
            sumi1 = vdotq_s32(sumi1, sqx1, qy1);
            sumi0 = vdotq_s32(sumi0, sqx2, qy2);
            sumi1 = vdotq_s32(sumi1, sqx3, qy3);
            sumi0 = vdotq_s32(sumi0, sqx4, qy4);
            sumi1 = vdotq_s32(sumi1, sqx5, qy5);
            sumi0 = vdotq_s32(sumi0, sqx6, qy6);
            sumi1 = vdotq_s32(sumi1, sqx7, qy7);
            sumi0 = vdotq_s32(sumi0, sqx8, qy8);
            sumi1 = vdotq_s32(sumi1, sqx9, qy9);
#else
            sumi0 = vmlal_s8(sumi0, vget_low_s8(sqx0), vget_low_s8(qy0));
            sumi1 = vmlal_s8(sumi1, vget_high_s8(sqx0), vget_high_s8(qy0));
            sumi0 = vmlal_s8(sumi0, vget_low_s8(sqx1), vget_low_s8(qy1));
            sumi1 = vmlal_s8(sumi1, vget_high_s8(sqx1), vget_high_s8(qy1));
            sumi0 = vmlal_s8(sumi0, vget_low_s8(sqx2), vget_low_s8(qy2));
            sumi1 = vmlal_s8(sumi1, vget_high_s8(sqx2), vget_high_s8(qy2));
            sumi0 = vmlal_s8(sumi0, vget_low_s8(sqx3), vget_low_s8(qy3));
            sumi1 = vmlal_s8(sumi1, vget_high_s8(sqx3), vget_high_s8(qy3));
            sumi0 = vmlal_s8(sumi0, vget_low_s8(sqx4), vget_low_s8(qy4));
            sumi1 = vmlal_s8(sumi1, vget_high_s8(sqx4), vget_high_s8(qy4));
            sumi0 = vmlal_s8(sumi0, vget_low_s8(sqx5), vget_low_s8(qy5));
            sumi1 = vmlal_s8(sumi1, vget_high_s8(sqx5), vget_high_s8(qy5));
            sumi0 = vmlal_s8(sumi0, vget_low_s8(sqx6), vget_low_s8(qy6));
            sumi1 = vmlal_s8(sumi1, vget_high_s8(sqx6), vget_high_s8(qy6));
            sumi0 = vmlal_s8(sumi0, vget_low_s8(sqx7), vget_low_s8(qy7));
            sumi1 = vmlal_s8(sumi1, vget_high_s8(sqx7), vget_high_s8(qy7));
            sumi0 = vmlal_s8(sumi0, vget_low_s8(sqx8), vget_low_s8(qy8));
            sumi1 = vmlal_s8(sumi1, vget_high_s8(sqx8), vget_high_s8(qy8));
            sumi0 = vmlal_s8(sumi0, vget_low_s8(sqx9), vget_low_s8(qy9));
            sumi1 = vmlal_s8(sumi1, vget_high_s8(sqx9), vget_high_s8(qy9));
#endif
        }

        // last 16 bytes of 5-element, along with the 4 bytes of 4 elements
        {
            uint8x16_t qx0 = vld1q_u8(x[i].qs + 32);
            uint8x16_t qx1 = vmulq_u8(qx0, vdupq_n_u8(3));
            uint8x16_t qx2 = vmulq_u8(qx0, vdupq_n_u8(9));
            uint8x16_t qx3 = vmulq_u8(qx0, vdupq_n_u8(27));
            uint8x16_t qx4 = vmulq_u8(qx0, vdupq_n_u8(81));
            uint32_t qh;
            memcpy(&qh, x[i].qh, sizeof(qh)); // potentially unaligned
            uint8x16_t qx5 = vreinterpretq_u8_u32(vdupq_n_u32(qh));
            qx5 = vmulq_u8(qx5, shift);

            // multiply by 3 and keep the 2 bits above 8 bits
            int8x16_t sqx0 = vreinterpretq_s8_u8(vshrq_n_u8(vhaddq_u8(qx0, vshrq_n_u8(qx0, 1)), 6));
            int8x16_t sqx1 = vreinterpretq_s8_u8(vshrq_n_u8(vhaddq_u8(qx1, vshrq_n_u8(qx1, 1)), 6));
            int8x16_t sqx2 = vreinterpretq_s8_u8(vshrq_n_u8(vhaddq_u8(qx2, vshrq_n_u8(qx2, 1)), 6));
            int8x16_t sqx3 = vreinterpretq_s8_u8(vshrq_n_u8(vhaddq_u8(qx3, vshrq_n_u8(qx3, 1)), 6));
            int8x16_t sqx4 = vreinterpretq_s8_u8(vshrq_n_u8(vhaddq_u8(qx4, vshrq_n_u8(qx4, 1)), 6));
            int8x16_t sqx5 = vreinterpretq_s8_u8(vshrq_n_u8(vhaddq_u8(qx5, vshrq_n_u8(qx5, 1)), 6));

            const int8x16_t qy0 = vld1q_s8(y[i].qs + 160);
            const int8x16_t qy1 = vld1q_s8(y[i].qs + 176);
            const int8x16_t qy2 = vld1q_s8(y[i].qs + 192);
            const int8x16_t qy3 = vld1q_s8(y[i].qs + 208);
            const int8x16_t qy4 = vld1q_s8(y[i].qs + 224);
            const int8x16_t qy5 = vld1q_s8(y[i].qs + 240);

#if defined(__ARM_FEATURE_DOTPROD)
            sumi0 = vdotq_s32(sumi0, sqx0, qy0);
            sumi1 = vdotq_s32(sumi1, sqx1, qy1);
            sumi0 = vdotq_s32(sumi0, sqx2, qy2);
            sumi1 = vdotq_s32(sumi1, sqx3, qy3);
            sumi0 = vdotq_s32(sumi0, sqx4, qy4);
            sumi1 = vdotq_s32(sumi1, sqx5, qy5);
#else
            sumi0 = vmlal_s8(sumi0, vget_low_s8(sqx0), vget_low_s8(qy0));
            sumi1 = vmlal_s8(sumi1, vget_high_s8(sqx0), vget_high_s8(qy0));
            sumi0 = vmlal_s8(sumi0, vget_low_s8(sqx1), vget_low_s8(qy1));
            sumi1 = vmlal_s8(sumi1, vget_high_s8(sqx1), vget_high_s8(qy1));
            sumi0 = vmlal_s8(sumi0, vget_low_s8(sqx2), vget_low_s8(qy2));
            sumi1 = vmlal_s8(sumi1, vget_high_s8(sqx2), vget_high_s8(qy2));
            sumi0 = vmlal_s8(sumi0, vget_low_s8(sqx3), vget_low_s8(qy3));
            sumi1 = vmlal_s8(sumi1, vget_high_s8(sqx3), vget_high_s8(qy3));
            sumi0 = vmlal_s8(sumi0, vget_low_s8(sqx4), vget_low_s8(qy4));
            sumi1 = vmlal_s8(sumi1, vget_high_s8(sqx4), vget_high_s8(qy4));
            sumi0 = vmlal_s8(sumi0, vget_low_s8(sqx5), vget_low_s8(qy5));
            sumi1 = vmlal_s8(sumi1, vget_high_s8(sqx5), vget_high_s8(qy5));
#endif
        }

        const int16x8_t ysum0 = vld1q_s16(y[i].bsums);
        const int16x8_t ysum1 = vld1q_s16(y[i].bsums + 8);

        const float d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;

#if defined(__ARM_FEATURE_DOTPROD)
        sumi0 = vaddq_s32(sumi0, sumi1);
        sumi0 = vsubq_s32(sumi0, vpaddlq_s16(vaddq_s16(ysum0, ysum1)));

        sumf += d * (float) vaddvq_s32(sumi0);
#else
        sumi0 = vaddq_s16(sumi0, sumi1);
        sumi0 = vsubq_s16(sumi0, vaddq_s16(ysum0, ysum1));

        sumf += d * (float) vaddlvq_s16(sumi0);
#endif
    }

    *s = sumf;

#else
    UNUSED(x);
    UNUSED(y);
    UNUSED(nb);
    ggml_vec_dot_tq1_0_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#endif
}

void ggml_vec_dot_tq2_0_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_tq2_0 * GGML_RESTRICT x = vx;
    const block_q8_K  * GGML_RESTRICT y = vy;

    const int nb = n / QK_K;

#if defined(__ARM_NEON)
    float sumf = 0.0f;

    const uint8x16_t m3 = vdupq_n_u8(3);

    for (int i = 0; i < nb; ++i) {
#if defined(__ARM_FEATURE_DOTPROD)
        int32x4_t sumi0 = vdupq_n_s32(0);
        int32x4_t sumi1 = vdupq_n_s32(0);
#else
        int16x8_t sumi0 = vdupq_n_s16(0);
        int16x8_t sumi1 = vdupq_n_s16(0);
#endif

        for (size_t j = 0; j < sizeof(x->qs); j += 32) {
            uint8x16_t qx0 = vld1q_u8(x[i].qs + j);
            uint8x16_t qx1 = vld1q_u8(x[i].qs + j + 16);
            uint8x16_t qx2 = vshrq_n_u8(qx0, 2);
            uint8x16_t qx3 = vshrq_n_u8(qx1, 2);
            uint8x16_t qx4 = vshrq_n_u8(qx0, 4);
            uint8x16_t qx5 = vshrq_n_u8(qx1, 4);
            uint8x16_t qx6 = vshrq_n_u8(qx0, 6);
            uint8x16_t qx7 = vshrq_n_u8(qx1, 6);

            int8x16_t sqx0 = vreinterpretq_s8_u8(vandq_u8(qx0, m3));
            int8x16_t sqx1 = vreinterpretq_s8_u8(vandq_u8(qx1, m3));
            int8x16_t sqx2 = vreinterpretq_s8_u8(vandq_u8(qx2, m3));
            int8x16_t sqx3 = vreinterpretq_s8_u8(vandq_u8(qx3, m3));
            int8x16_t sqx4 = vreinterpretq_s8_u8(vandq_u8(qx4, m3));
            int8x16_t sqx5 = vreinterpretq_s8_u8(vandq_u8(qx5, m3));
            int8x16_t sqx6 = vreinterpretq_s8_u8(vandq_u8(qx6, m3));
            int8x16_t sqx7 = vreinterpretq_s8_u8(vandq_u8(qx7, m3));

            const int8x16_t qy0 = vld1q_s8(y[i].qs + j*4 +   0);
            const int8x16_t qy1 = vld1q_s8(y[i].qs + j*4 +  16);
            const int8x16_t qy2 = vld1q_s8(y[i].qs + j*4 +  32);
            const int8x16_t qy3 = vld1q_s8(y[i].qs + j*4 +  48);
            const int8x16_t qy4 = vld1q_s8(y[i].qs + j*4 +  64);
            const int8x16_t qy5 = vld1q_s8(y[i].qs + j*4 +  80);
            const int8x16_t qy6 = vld1q_s8(y[i].qs + j*4 +  96);
            const int8x16_t qy7 = vld1q_s8(y[i].qs + j*4 + 112);

#if defined(__ARM_FEATURE_DOTPROD)
            sumi0 = vdotq_s32(sumi0, sqx0, qy0);
            sumi1 = vdotq_s32(sumi1, sqx1, qy1);
            sumi0 = vdotq_s32(sumi0, sqx2, qy2);
            sumi1 = vdotq_s32(sumi1, sqx3, qy3);
            sumi0 = vdotq_s32(sumi0, sqx4, qy4);
            sumi1 = vdotq_s32(sumi1, sqx5, qy5);
            sumi0 = vdotq_s32(sumi0, sqx6, qy6);
            sumi1 = vdotq_s32(sumi1, sqx7, qy7);
#else
            sumi0 = vmlal_s8(sumi0, vget_low_s8(sqx0), vget_low_s8(qy0));
            sumi1 = vmlal_s8(sumi1, vget_high_s8(sqx0), vget_high_s8(qy0));
            sumi0 = vmlal_s8(sumi0, vget_low_s8(sqx1), vget_low_s8(qy1));
            sumi1 = vmlal_s8(sumi1, vget_high_s8(sqx1), vget_high_s8(qy1));
            sumi0 = vmlal_s8(sumi0, vget_low_s8(sqx2), vget_low_s8(qy2));
            sumi1 = vmlal_s8(sumi1, vget_high_s8(sqx2), vget_high_s8(qy2));
            sumi0 = vmlal_s8(sumi0, vget_low_s8(sqx3), vget_low_s8(qy3));
            sumi1 = vmlal_s8(sumi1, vget_high_s8(sqx3), vget_high_s8(qy3));
            sumi0 = vmlal_s8(sumi0, vget_low_s8(sqx4), vget_low_s8(qy4));
            sumi1 = vmlal_s8(sumi1, vget_high_s8(sqx4), vget_high_s8(qy4));
            sumi0 = vmlal_s8(sumi0, vget_low_s8(sqx5), vget_low_s8(qy5));
            sumi1 = vmlal_s8(sumi1, vget_high_s8(sqx5), vget_high_s8(qy5));
            sumi0 = vmlal_s8(sumi0, vget_low_s8(sqx6), vget_low_s8(qy6));
            sumi1 = vmlal_s8(sumi1, vget_high_s8(sqx6), vget_high_s8(qy6));
            sumi0 = vmlal_s8(sumi0, vget_low_s8(sqx7), vget_low_s8(qy7));
            sumi1 = vmlal_s8(sumi1, vget_high_s8(sqx7), vget_high_s8(qy7));
#endif
        }

        const int16x8_t ysum0 = vld1q_s16(y[i].bsums);
        const int16x8_t ysum1 = vld1q_s16(y[i].bsums + 8);

        const float d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;

#if defined(__ARM_FEATURE_DOTPROD)
        sumi0 = vaddq_s32(sumi0, sumi1);
        sumi0 = vsubq_s32(sumi0, vpaddlq_s16(vaddq_s16(ysum0, ysum1)));

        sumf += d * (float) vaddvq_s32(sumi0);
#else
        sumi0 = vaddq_s16(sumi0, sumi1);
        sumi0 = vsubq_s16(sumi0, vaddq_s16(ysum0, ysum1));

        sumf += d * (float) vaddlvq_s16(sumi0);
#endif
    }

    *s = sumf;

#else
    UNUSED(x);
    UNUSED(y);
    UNUSED(nb);
    ggml_vec_dot_tq2_0_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#endif
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

#ifdef __ARM_FEATURE_SVE
    const int vector_length = svcntb()*8;
    const svuint8_t m3s = svdup_n_u8(0x3);
    const svuint32_t m4s = svdup_n_u32(0xF);
    const svint32_t vzero_sv = svdup_n_s32(0);
    svfloat32_t acc_sum = svdup_n_f32(0);
    svbool_t pred_s32 = svptrue_pat_b32(SV_VL4);

    switch (vector_length) {
        case 128:
            for (int i = 0; i < nb; ++i) {
                const float d = y[i].d * GGML_CPU_FP16_TO_FP32(x[i].d);
                svfloat32_t d_broad = svdup_n_f32((float32_t)d);
                const float dmin = -y[i].d * GGML_CPU_FP16_TO_FP32(x[i].dmin);
                svfloat32_t dmin_broad = svdup_n_f32((float32_t)dmin);

                const uint8_t * GGML_RESTRICT q2 = x[i].qs;
                const int8_t  * GGML_RESTRICT q8_sv = y[i].qs;
                const uint8_t * GGML_RESTRICT sc = x[i].scales;

                svuint32_t mins_and_scales_sve = svld1ub_u32(svptrue_b32(), sc);
                const svint32_t mins_sv_1 = svreinterpret_s32_u32(svlsr_n_u32_x(svptrue_b32(), mins_and_scales_sve, 4));

                mins_and_scales_sve = svld1ub_u32(svptrue_b32(), sc+4);
                const svint32_t mins_sv_2 = svreinterpret_s32_u32(svlsr_n_u32_x(svptrue_b32(), mins_and_scales_sve, 4));

                svint32_t q8sums_sv_1 = svld1sh_s32(svptrue_b32(), y[i].bsums);
                svint32_t q8sums_sv_2 = svld1sh_s32(svptrue_b32(), y[i].bsums+4);

                const svint32_t s0 = svadd_s32_x(svptrue_b32(), svmul_s32_x(svptrue_b32(), mins_sv_1, q8sums_sv_1), svmul_s32_x(svptrue_b32(), mins_sv_2, q8sums_sv_2));

                mins_and_scales_sve = svld1ub_u32(svptrue_b32(), sc+8);
                const svint32_t mins_sv_3 = svreinterpret_s32_u32(svlsr_n_u32_x(svptrue_b32(), mins_and_scales_sve, 4));

                mins_and_scales_sve = svld1ub_u32(svptrue_b32(), sc+12);
                const svint32_t mins_sv_4 = svreinterpret_s32_u32(svlsr_n_u32_x(svptrue_b32(), mins_and_scales_sve, 4));

                q8sums_sv_1 = svld1sh_s32(svptrue_b32(), y[i].bsums+8);
                q8sums_sv_2 = svld1sh_s32(svptrue_b32(), y[i].bsums+12);

                svint32_t s1 = svadd_s32_x(svptrue_b32(), svmul_s32_x(svptrue_b32(), mins_sv_3, q8sums_sv_1), svmul_s32_x(svptrue_b32(), mins_sv_4, q8sums_sv_2));

                svfloat32_t temp = svcvt_f32_s32_x(svptrue_b32(), svadd_s32_x(svptrue_b32(), s0, s1));

                acc_sum = svmla_f32_m(svptrue_b32(), acc_sum, temp, dmin_broad);

                svint32_t sumi1 = svdup_n_s32(0);

                {
                    const svuint8_t q2bits_1 = svld1_u8(svptrue_b8(), q2);
                    svint8_t q2bytes_sv = svreinterpret_s8_u8(svand_u8_x(svptrue_b8(), q2bits_1, m3s));
                    svint8_t q8bytes_sv = svld1_s8(svptrue_b8(), q8_sv); q8_sv += 16;
                    const svint32_t scales_sv = svreinterpret_s32_u32(svand_u32_m(svptrue_b32(), svld1ub_u32(svptrue_b32(), sc), m4s));

                    sumi1 = svmla_s32_m(svptrue_b32(), sumi1, svdot_s32(vzero_sv, q2bytes_sv, q8bytes_sv), svdup_lane_s32(scales_sv, 0));

                    const svuint8_t q2bits_3 = svld1_u8(svptrue_b8(), q2+16);
                    q2bytes_sv = svreinterpret_s8_u8(svand_u8_x(svptrue_b8(), q2bits_3, m3s));
                    q8bytes_sv = svld1_s8(svptrue_b8(), q8_sv); q8_sv += 16;

                    sumi1 = svmla_s32_m(svptrue_b32(), sumi1, svdot_s32(vzero_sv, q2bytes_sv, q8bytes_sv), svdup_lane_s32(scales_sv, 1));

                    q2bytes_sv = svreinterpret_s8_u8(svand_u8_x(svptrue_b8(), svlsr_n_u8_x(svptrue_b8(), q2bits_1, 2), m3s));
                    q8bytes_sv = svld1_s8(svptrue_b8(), q8_sv); q8_sv += 16;

                    sumi1 = svmla_s32_m(svptrue_b32(), sumi1, svdot_s32(vzero_sv, q2bytes_sv, q8bytes_sv), svdup_lane_s32(scales_sv, 2));

                    q2bytes_sv = svreinterpret_s8_u8(svand_u8_x(svptrue_b8(), svlsr_n_u8_x(svptrue_b8(), q2bits_3, 2), m3s));
                    q8bytes_sv = svld1_s8(svptrue_b8(), q8_sv); q8_sv += 16;

                    sumi1 = svmla_s32_m(svptrue_b32(), sumi1, svdot_s32(vzero_sv, q2bytes_sv, q8bytes_sv), svdup_lane_s32(scales_sv, 3));


                    const svint32_t scales_sv_1 = svreinterpret_s32_u32(svand_u32_m(svptrue_b32(), svld1ub_u32(svptrue_b32(), sc+4), m4s));

                    q2bytes_sv = svreinterpret_s8_u8(svand_u8_x(svptrue_b8(), svlsr_n_u8_x(svptrue_b8(), q2bits_1, 4), m3s));
                    q8bytes_sv = svld1_s8(svptrue_b8(), q8_sv); q8_sv += 16;

                    sumi1 = svmla_s32_m(svptrue_b32(), sumi1, svdot_s32(vzero_sv, q2bytes_sv, q8bytes_sv), svdup_lane_s32(scales_sv_1, 0));

                    q2bytes_sv = svreinterpret_s8_u8(svand_u8_x(svptrue_b8(), svlsr_n_u8_x(svptrue_b8(), q2bits_3, 4), m3s));
                    q8bytes_sv = svld1_s8(svptrue_b8(), q8_sv); q8_sv += 16;

                    sumi1 = svmla_s32_m(svptrue_b32(), sumi1, svdot_s32(vzero_sv, q2bytes_sv, q8bytes_sv), svdup_lane_s32(scales_sv_1, 1));

                    q2bytes_sv = svreinterpret_s8_u8(svand_u8_x(svptrue_b8(), svlsr_n_u8_x(svptrue_b8(), q2bits_1, 6), m3s));
                    q8bytes_sv = svld1_s8(svptrue_b8(), q8_sv); q8_sv += 16;

                    sumi1 = svmla_s32_m(svptrue_b32(), sumi1, svdot_s32(vzero_sv, q2bytes_sv, q8bytes_sv), svdup_lane_s32(scales_sv_1, 2));

                    q2bytes_sv = svreinterpret_s8_u8(svand_u8_x(svptrue_b8(), svlsr_n_u8_x(svptrue_b8(), q2bits_3, 6), m3s));
                    q8bytes_sv = svld1_s8(svptrue_b8(), q8_sv); q8_sv += 16;

                    sumi1 = svmla_s32_m(svptrue_b32(), sumi1, svdot_s32(vzero_sv, q2bytes_sv, q8bytes_sv), svdup_lane_s32(scales_sv_1, 3));

                    //-------------------------------

                    q2 += 32;
                    const svint32_t scales_sv_2 = svreinterpret_s32_u32(svand_u32_m(svptrue_b32(), svld1ub_u32(svptrue_b32(), sc+8), m4s));
                    const svuint8_t q2bits_2 = svld1_u8(svptrue_b8(), q2);

                    q2bytes_sv = svreinterpret_s8_u8(svand_u8_x(svptrue_b8(), q2bits_2, m3s));
                    q8bytes_sv = svld1_s8(svptrue_b8(), q8_sv); q8_sv += 16;

                    sumi1 = svmla_s32_m(svptrue_b32(), sumi1, svdot_s32(vzero_sv, q2bytes_sv, q8bytes_sv), svdup_lane_s32(scales_sv_2, 0));

                    const svuint8_t q2bits_4 = svld1_u8(svptrue_b8(), q2+16);
                    q2bytes_sv = svreinterpret_s8_u8(svand_u8_x(svptrue_b8(), q2bits_4, m3s));
                    q8bytes_sv = svld1_s8(svptrue_b8(), q8_sv); q8_sv += 16;

                    sumi1 = svmla_s32_m(svptrue_b32(), sumi1, svdot_s32(vzero_sv, q2bytes_sv, q8bytes_sv), svdup_lane_s32(scales_sv_2, 1));


                    q2bytes_sv = svreinterpret_s8_u8(svand_u8_x(svptrue_b8(), svlsr_n_u8_x(svptrue_b8(), q2bits_2, 2), m3s));
                    q8bytes_sv = svld1_s8(svptrue_b8(), q8_sv); q8_sv += 16;

                    sumi1 = svmla_s32_m(svptrue_b32(), sumi1, svdot_s32(vzero_sv, q2bytes_sv, q8bytes_sv), svdup_lane_s32(scales_sv_2, 2));

                    q2bytes_sv = svreinterpret_s8_u8(svand_u8_x(svptrue_b8(), svlsr_n_u8_x(svptrue_b8(), q2bits_4, 2), m3s));
                    q8bytes_sv = svld1_s8(svptrue_b8(), q8_sv); q8_sv += 16;

                    sumi1 = svmla_s32_m(svptrue_b32(), sumi1, svdot_s32(vzero_sv, q2bytes_sv, q8bytes_sv), svdup_lane_s32(scales_sv_2, 3));


                    const svint32_t scales_sv_3 = svreinterpret_s32_u32(svand_u32_m(svptrue_b32(), svld1ub_u32(svptrue_b32(), sc+12), m4s));

                    q2bytes_sv = svreinterpret_s8_u8(svand_u8_x(svptrue_b8(), svlsr_n_u8_x(svptrue_b8(), q2bits_2, 4), m3s));
                    q8bytes_sv = svld1_s8(svptrue_b8(), q8_sv); q8_sv += 16;

                    sumi1 = svmla_s32_m(svptrue_b32(), sumi1, svdot_s32(vzero_sv, q2bytes_sv, q8bytes_sv), svdup_lane_s32(scales_sv_3, 0));

                    q2bytes_sv = svreinterpret_s8_u8(svand_u8_x(svptrue_b8(), svlsr_n_u8_x(svptrue_b8(), q2bits_4, 4), m3s));
                    q8bytes_sv = svld1_s8(svptrue_b8(), q8_sv); q8_sv += 16;

                    sumi1 = svmla_s32_m(svptrue_b32(), sumi1, svdot_s32(vzero_sv, q2bytes_sv, q8bytes_sv), svdup_lane_s32(scales_sv_3, 1));



                    q2bytes_sv = svreinterpret_s8_u8(svand_u8_x(svptrue_b8(), svlsr_n_u8_x(svptrue_b8(), q2bits_2, 6), m3s));
                    q8bytes_sv = svld1_s8(svptrue_b8(), q8_sv); q8_sv += 16;

                    sumi1 = svmla_s32_m(svptrue_b32(), sumi1, svdot_s32(vzero_sv, q2bytes_sv, q8bytes_sv), svdup_lane_s32(scales_sv_3, 2));

                    q2bytes_sv = svreinterpret_s8_u8(svand_u8_x(svptrue_b8(), svlsr_n_u8_x(svptrue_b8(), q2bits_4, 6), m3s));
                    q8bytes_sv = svld1_s8(svptrue_b8(), q8_sv); q8_sv += 16;

                    sumi1 = svmla_s32_m(svptrue_b32(), sumi1, svdot_s32(vzero_sv, q2bytes_sv, q8bytes_sv), svdup_lane_s32(scales_sv_3, 3));
                }
                acc_sum = svmla_f32_m(svptrue_b32(), acc_sum, svcvt_f32_s32_x(svptrue_b32(), sumi1), d_broad);
            }
            *s = svaddv_f32(svptrue_b32(), acc_sum);
            break;

        case 256:
        case 512:
            for (int i = 0; i < nb; ++i) {
                const float d = y[i].d * GGML_CPU_FP16_TO_FP32(x[i].d);
                svfloat32_t d_broad = svdup_n_f32((float32_t)d);
                const float dmin = -y[i].d * GGML_CPU_FP16_TO_FP32(x[i].dmin);
                svfloat32_t dmin_broad = svdup_n_f32((float32_t)dmin);

                const uint8_t * GGML_RESTRICT q2 = x[i].qs;
                const int8_t  * GGML_RESTRICT q8_sv = y[i].qs;
                const uint8_t * GGML_RESTRICT sc = x[i].scales;

                const svuint32_t mins_and_scales_sve = svld1ub_u32(svptrue_pat_b32(SV_VL8), sc); sc += 8;
                const svint32_t scales_sv = svreinterpret_s32_u32(svand_u32_m(svptrue_pat_b32(SV_VL8), mins_and_scales_sve, m4s));
                const svint32_t mins_sv_1 = svreinterpret_s32_u32(svlsr_n_u32_x(svptrue_pat_b32(SV_VL8), mins_and_scales_sve, 4));
                svint32_t q8sums_sv_1 = svld1sh_s32(svptrue_pat_b32(SV_VL8), y[i].bsums);

                const svuint32_t mins_and_scales_sve_1 = svld1ub_u32(svptrue_pat_b32(SV_VL8), sc);
                const svint32_t scales_sv_1 = svreinterpret_s32_u32(svand_u32_m(svptrue_pat_b32(SV_VL8), mins_and_scales_sve_1, m4s));
                const svint32_t mins_sv_2 = svreinterpret_s32_u32(svlsr_n_u32_x(svptrue_pat_b32(SV_VL8), mins_and_scales_sve_1, 4));

                svint32_t q8sums_sv_2 = svld1sh_s32(svptrue_pat_b32(SV_VL8), y[i].bsums+8);

                svfloat32_t temp = svcvt_f32_s32_x(svptrue_pat_b32(SV_VL8), svadd_s32_x(svptrue_pat_b32(SV_VL8), svmul_s32_x(svptrue_pat_b32(SV_VL8), mins_sv_1, q8sums_sv_1), svmul_s32_x(svptrue_pat_b32(SV_VL8), mins_sv_2, q8sums_sv_2)));

                acc_sum = svmla_f32_m(svptrue_pat_b32(SV_VL8), acc_sum, temp, dmin_broad);

                svint32_t sumi1 = svdup_n_s32(0);

                {
                    const svuint8_t q2bits_1 = svld1_u8(svptrue_pat_b8(SV_VL32), q2);
                    svint8_t q2bytes_sv = svreinterpret_s8_u8(svand_u8_m(svptrue_pat_b8(SV_VL32), q2bits_1, m3s));
                    svint8_t q8bytes_sv = svld1_s8(svptrue_pat_b8(SV_VL32), q8_sv); q8_sv += 32;

                    svint32_t scale_1 = svsel(pred_s32, svdup_lane_s32(scales_sv, 0), svdup_lane_s32(scales_sv, 1));
                    sumi1 = svmla_s32_m(svptrue_pat_b32(SV_VL8), sumi1, svdot_s32(vzero_sv, q2bytes_sv, q8bytes_sv), scale_1);

                    q2bytes_sv = svreinterpret_s8_u8(svand_u8_m(svptrue_pat_b8(SV_VL32), svlsr_n_u8_x(svptrue_pat_b8(SV_VL32), q2bits_1, 2), m3s));
                    q8bytes_sv = svld1_s8(svptrue_pat_b8(SV_VL32), q8_sv); q8_sv += 32;

                    svint32_t scale_2 = svsel(pred_s32, svdup_lane_s32(scales_sv, 2), svdup_lane_s32(scales_sv, 3));
                    sumi1 = svmla_s32_m(svptrue_pat_b32(SV_VL8), sumi1, svdot_s32(svdup_n_s32(0), q2bytes_sv, q8bytes_sv), scale_2);

                    q2bytes_sv = svreinterpret_s8_u8(svand_u8_m(svptrue_pat_b8(SV_VL32), svlsr_n_u8_x(svptrue_pat_b8(SV_VL32), q2bits_1, 4), m3s));
                    q8bytes_sv = svld1_s8(svptrue_pat_b8(SV_VL32), q8_sv); q8_sv += 32;

                    scale_1 = svsel(pred_s32, svdup_lane_s32(scales_sv, 4), svdup_lane_s32(scales_sv, 5));
                    sumi1 = svmla_s32_m(svptrue_pat_b32(SV_VL8), sumi1, svdot_s32(vzero_sv, q2bytes_sv, q8bytes_sv), scale_1);

                    q2bytes_sv = svreinterpret_s8_u8(svand_u8_m(svptrue_pat_b8(SV_VL32), svlsr_n_u8_x(svptrue_pat_b8(SV_VL32), q2bits_1, 6), m3s));
                    q8bytes_sv = svld1_s8(svptrue_pat_b8(SV_VL32), q8_sv); q8_sv += 32;

                    scale_2 = svsel(pred_s32, svdup_lane_s32(scales_sv, 6), svdup_lane_s32(scales_sv, 7));
                    sumi1 = svmla_s32_m(svptrue_pat_b32(SV_VL8), sumi1, svdot_s32(vzero_sv, q2bytes_sv, q8bytes_sv), scale_2);

                    q2 += 32;

                    const svuint8_t q2bits_2 = svld1_u8(svptrue_pat_b8(SV_VL32), q2);
                    q2bytes_sv = svreinterpret_s8_u8(svand_u8_m(svptrue_pat_b8(SV_VL32), q2bits_2, m3s));
                    q8bytes_sv = svld1_s8(svptrue_pat_b8(SV_VL32), q8_sv); q8_sv += 32;

                    scale_1 = svsel(pred_s32, svdup_lane_s32(scales_sv_1, 0), svdup_lane_s32(scales_sv_1, 1));
                    sumi1 = svmla_s32_m(svptrue_pat_b32(SV_VL8), sumi1, svdot_s32(vzero_sv, q2bytes_sv, q8bytes_sv), scale_1);

                    q2bytes_sv = svreinterpret_s8_u8(svand_u8_m(svptrue_pat_b8(SV_VL32), svlsr_n_u8_x(svptrue_pat_b8(SV_VL32), q2bits_2, 2), m3s));
                    q8bytes_sv = svld1_s8(svptrue_pat_b8(SV_VL32), q8_sv); q8_sv += 32;

                    scale_2 = svsel(pred_s32, svdup_lane_s32(scales_sv_1, 2), svdup_lane_s32(scales_sv_1, 3));
                    sumi1 = svmla_s32_m(svptrue_pat_b32(SV_VL8), sumi1, svdot_s32(vzero_sv, q2bytes_sv, q8bytes_sv), scale_2);

                    q2bytes_sv = svreinterpret_s8_u8(svand_u8_m(svptrue_pat_b8(SV_VL32), svlsr_n_u8_x(svptrue_pat_b8(SV_VL32), q2bits_2, 4), m3s));
                    q8bytes_sv = svld1_s8(svptrue_pat_b8(SV_VL32), q8_sv); q8_sv += 32;

                    scale_1 = svsel(pred_s32, svdup_lane_s32(scales_sv_1, 4), svdup_lane_s32(scales_sv_1, 5));
                    sumi1 = svmla_s32_m(svptrue_pat_b32(SV_VL8), sumi1, svdot_s32(vzero_sv, q2bytes_sv, q8bytes_sv), scale_1);

                    q2bytes_sv = svreinterpret_s8_u8(svand_u8_m(svptrue_pat_b8(SV_VL32), svlsr_n_u8_x(svptrue_pat_b8(SV_VL32), q2bits_2, 6), m3s));
                    q8bytes_sv = svld1_s8(svptrue_pat_b8(SV_VL32), q8_sv); q8_sv += 32;

                    scale_2 = svsel(pred_s32, svdup_lane_s32(scales_sv_1, 6), svdup_lane_s32(scales_sv_1, 7));
                    sumi1 = svmla_s32_m(svptrue_pat_b32(SV_VL8), sumi1, svdot_s32(vzero_sv, q2bytes_sv, q8bytes_sv), scale_2);
                }
                acc_sum = svmla_f32_m(svptrue_pat_b32(SV_VL8), acc_sum, svcvt_f32_s32_x(svptrue_pat_b32(SV_VL8), sumi1), d_broad);
            }
            *s = svaddv_f32(svptrue_pat_b32(SV_VL8), acc_sum);
            break;

        default:
            assert(false && "Unsupported vector length");
            break;
    }

#elif __ARM_NEON
    const uint8x16_t m3 = vdupq_n_u8(0x3);
    const uint8x16_t m4 = vdupq_n_u8(0xF);

    const int32x4_t vzero = vdupq_n_s32(0);

    ggml_int8x16x2_t q2bytes;
    uint8_t aux[16];

    float sum = 0;

    for (int i = 0; i < nb; ++i) {
        const float d = y[i].d * GGML_CPU_FP16_TO_FP32(x[i].d);
        const float dmin = -y[i].d * GGML_CPU_FP16_TO_FP32(x[i].dmin);

        const uint8_t * GGML_RESTRICT q2 = x[i].qs;
        const int8_t  * GGML_RESTRICT q8 = y[i].qs;
        const uint8_t * GGML_RESTRICT sc = x[i].scales;

        const uint8x16_t mins_and_scales = vld1q_u8(sc);
        const uint8x16_t scales = vandq_u8(mins_and_scales, m4);
        vst1q_u8(aux, scales);

        const uint8x16_t mins = vshrq_n_u8(mins_and_scales, 4);
        const ggml_int16x8x2_t q8sums = ggml_vld1q_s16_x2(y[i].bsums);
        const ggml_int16x8x2_t mins16 = {{vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(mins))), vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(mins)))}};
        const int32x4_t s0 = vaddq_s32(vmull_s16(vget_low_s16 (mins16.val[0]), vget_low_s16 (q8sums.val[0])),
                                       vmull_s16(vget_high_s16(mins16.val[0]), vget_high_s16(q8sums.val[0])));
        const int32x4_t s1 = vaddq_s32(vmull_s16(vget_low_s16 (mins16.val[1]), vget_low_s16 (q8sums.val[1])),
                                       vmull_s16(vget_high_s16(mins16.val[1]), vget_high_s16(q8sums.val[1])));
        sum += dmin * vaddvq_s32(vaddq_s32(s0, s1));

        int isum = 0;
        int is = 0;

// We use this macro instead of a function call because for some reason
// the code runs 2-3% slower, even if the function is declared inline
#define MULTIPLY_ACCUM_WITH_SCALE(index)\
        isum += vaddvq_s32(ggml_vdotq_s32(vzero, q2bytes.val[0], q8bytes.val[0])) * aux[is+(index)];\
        isum += vaddvq_s32(ggml_vdotq_s32(vzero, q2bytes.val[1], q8bytes.val[1])) * aux[is+1+(index)];

#define SHIFT_MULTIPLY_ACCUM_WITH_SCALE(shift, index)\
        q8bytes = ggml_vld1q_s8_x2(q8); q8 += 32;\
        q2bytes.val[0] = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q2bits.val[0], (shift)), m3));\
        q2bytes.val[1] = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q2bits.val[1], (shift)), m3));\
        MULTIPLY_ACCUM_WITH_SCALE((index));

        for (int j = 0; j < QK_K/128; ++j) {
            const ggml_uint8x16x2_t q2bits = ggml_vld1q_u8_x2(q2); q2 += 32;

            ggml_int8x16x2_t q8bytes = ggml_vld1q_s8_x2(q8); q8 += 32;
            q2bytes.val[0] = vreinterpretq_s8_u8(vandq_u8(q2bits.val[0], m3));
            q2bytes.val[1] = vreinterpretq_s8_u8(vandq_u8(q2bits.val[1], m3));

            MULTIPLY_ACCUM_WITH_SCALE(0);

            SHIFT_MULTIPLY_ACCUM_WITH_SCALE(2, 2);
            SHIFT_MULTIPLY_ACCUM_WITH_SCALE(4, 4);
            SHIFT_MULTIPLY_ACCUM_WITH_SCALE(6, 6);

            is += 8;
        }

        sum += d * isum;
    }

    *s = sum;

#else
    UNUSED(x);
    UNUSED(y);
    UNUSED(nb);
    ggml_vec_dot_q2_K_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
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

#if defined(__ARM_FEATURE_SVE)

    uint32_t aux[3];
    uint32_t utmp[4];

    const int8_t m32 = 32;
    const int vector_length = svcntb()*8;
    const svuint8_t m3b_sv = svdup_n_u8(0x3);
    const svint32_t vzero_sv = svdup_n_s32(0);

    const svuint8_t m0_sv = svdup_n_u8(1);
    const svuint8_t m1_sv = svlsl_n_u8_x(svptrue_b8(), m0_sv, 1);
    const svuint8_t m2_sv = svlsl_n_u8_x(svptrue_b8(), m0_sv, 2);
    const svuint8_t m3_sv = svlsl_n_u8_x(svptrue_b8(), m0_sv, 3);

    float sum = 0;

    for (int i = 0; i < nb; ++i) {

        const float d = y[i].d * GGML_CPU_FP16_TO_FP32(x[i].d);

        const uint8_t * GGML_RESTRICT q3_sv = x[i].qs;
        const uint8_t * GGML_RESTRICT qh_sv = x[i].hmask;
        const int8_t  * GGML_RESTRICT q8_sv = y[i].qs;

        // Set up scales
        memcpy(aux, x[i].scales, 12);
        utmp[3] = ((aux[1] >> 4) & kmask2) | (((aux[2] >> 6) & kmask1) << 4);
        utmp[2] = ((aux[0] >> 4) & kmask2) | (((aux[2] >> 4) & kmask1) << 4);
        utmp[1] = (aux[1] & kmask2) | (((aux[2] >> 2) & kmask1) << 4);
        utmp[0] = (aux[0] & kmask2) | (((aux[2] >> 0) & kmask1) << 4);

        int8_t * scale = (int8_t *)utmp;

        for (int j = 0; j < 16; ++j) scale[j] -= m32;

        switch (vector_length) {
            case 128:
                {
                    svuint8_t qhbits_sv_1 = svld1_u8(svptrue_b8(), qh_sv);
                    svuint8_t qhbits_sv_2 = svld1_u8(svptrue_b8(), qh_sv+16);
                    svuint8_t q3h_sv;

                    svint32_t sumi1_1 = svdup_n_s32(0);
                    svint8_t q3bytes_sv;

                    for (int j = 0; j < QK_K/128; ++j) {

                        const svuint8_t q3bits_sv = svld1_u8(svptrue_b8(), q3_sv); q3_sv += 16;
                        const svuint8_t q3bits_sv_1 = svld1_u8(svptrue_b8(), q3_sv); q3_sv += 16;
                        svint8_t q8bytes_1_sv_1 = svld1_s8(svptrue_b8(), q8_sv); q8_sv += 16;
                        svint8_t q8bytes_1_sv_2 = svld1_s8(svptrue_b8(), q8_sv); q8_sv += 16;

                        q3h_sv = svlsl_n_u8_x(svptrue_b8(), svbic_u8_x(svptrue_b8(), m0_sv, qhbits_sv_1), 2);
                        q3bytes_sv = svsub_s8_x(svptrue_b8(), svreinterpret_s8_u8(svand_u8_m(svptrue_b8(), q3bits_sv, m3b_sv)), svreinterpret_s8_u8(q3h_sv));

                        sumi1_1 = svmla_s32_m(svptrue_b32(), sumi1_1, svdot_s32(vzero_sv, q3bytes_sv, q8bytes_1_sv_1), svdup_n_s32((int32_t)scale[0]));

                        q3h_sv = svlsl_n_u8_x(svptrue_b8(), svbic_u8_x(svptrue_b8(), m0_sv, qhbits_sv_2), 2);
                        q3bytes_sv = svsub_s8_x(svptrue_b8(), svreinterpret_s8_u8(svand_u8_m(svptrue_b8(), q3bits_sv_1, m3b_sv)), svreinterpret_s8_u8(q3h_sv));

                        sumi1_1 = svmla_s32_m(svptrue_b32(), sumi1_1, svdot_s32(vzero_sv, q3bytes_sv, q8bytes_1_sv_2), svdup_n_s32((int32_t)scale[1]));

                        q8bytes_1_sv_1 = svld1_s8(svptrue_b8(), q8_sv); q8_sv += 16;
                        q8bytes_1_sv_2 = svld1_s8(svptrue_b8(), q8_sv); q8_sv += 16;

                        q3h_sv = svlsl_n_u8_x(svptrue_b8(), svbic_u8_x(svptrue_b8(), m1_sv, qhbits_sv_1), 1);
                        q3bytes_sv = svsub_s8_x(svptrue_b8(), svreinterpret_s8_u8(svand_u8_m(svptrue_b8(), svlsr_n_u8_x(svptrue_b8(), q3bits_sv, 2), m3b_sv)), svreinterpret_s8_u8(q3h_sv));

                        sumi1_1 = svmla_s32_m(svptrue_b32(), sumi1_1, svdot_s32(vzero_sv, q3bytes_sv, q8bytes_1_sv_1), svdup_n_s32((int32_t)scale[2]));

                        q3h_sv = svlsl_n_u8_x(svptrue_b8(), svbic_u8_x(svptrue_b8(), m1_sv, qhbits_sv_2), 1);
                        q3bytes_sv = svsub_s8_x(svptrue_b8(), svreinterpret_s8_u8(svand_u8_m(svptrue_b8(), svlsr_n_u8_x(svptrue_b8(), q3bits_sv_1, 2), m3b_sv)), svreinterpret_s8_u8(q3h_sv));

                        sumi1_1 = svmla_s32_m(svptrue_b32(), sumi1_1, svdot_s32(vzero_sv, q3bytes_sv, q8bytes_1_sv_2), svdup_n_s32((int32_t)scale[3]));


                        scale += 4;
                        q8bytes_1_sv_1 = svld1_s8(svptrue_b8(), q8_sv); q8_sv += 16;
                        q8bytes_1_sv_2 = svld1_s8(svptrue_b8(), q8_sv); q8_sv += 16;

                        q3h_sv = svbic_u8_x(svptrue_b8(), m2_sv, qhbits_sv_1);
                        q3bytes_sv = svsub_s8_x(svptrue_b8(), svreinterpret_s8_u8(svand_u8_m(svptrue_b8(), svlsr_n_u8_x(svptrue_b8(), q3bits_sv, 4), m3b_sv)), svreinterpret_s8_u8(q3h_sv));

                        sumi1_1 = svmla_s32_m(svptrue_b32(), sumi1_1, svdot_s32(vzero_sv, q3bytes_sv, q8bytes_1_sv_1), svdup_n_s32((int32_t)scale[0]));

                        q3h_sv = svbic_u8_x(svptrue_b8(), m2_sv, qhbits_sv_2);
                        q3bytes_sv = svsub_s8_x(svptrue_b8(), svreinterpret_s8_u8(svand_u8_m(svptrue_b8(), svlsr_n_u8_x(svptrue_b8(), q3bits_sv_1, 4), m3b_sv)), svreinterpret_s8_u8(q3h_sv));

                        sumi1_1 = svmla_s32_m(svptrue_b32(), sumi1_1, svdot_s32(vzero_sv, q3bytes_sv, q8bytes_1_sv_2), svdup_n_s32((int32_t)scale[1]));


                        q8bytes_1_sv_1 = svld1_s8(svptrue_b8(), q8_sv); q8_sv += 16;
                        q8bytes_1_sv_2 = svld1_s8(svptrue_b8(), q8_sv); q8_sv += 16;

                        q3h_sv = svlsr_n_u8_x(svptrue_b8(), svbic_u8_x(svptrue_b8(), m3_sv, qhbits_sv_1), 1);
                        q3bytes_sv = svsub_s8_x(svptrue_b8(), svreinterpret_s8_u8(svand_u8_m(svptrue_b8(), svlsr_n_u8_x(svptrue_b8(), q3bits_sv, 6), m3b_sv)), svreinterpret_s8_u8(q3h_sv));

                        sumi1_1 = svmla_s32_m(svptrue_b32(), sumi1_1, svdot_s32(vzero_sv, q3bytes_sv, q8bytes_1_sv_1), svdup_n_s32((int32_t)scale[2]));

                        q3h_sv = svlsr_n_u8_x(svptrue_b8(), svbic_u8_x(svptrue_b8(), m3_sv, qhbits_sv_2), 1);
                        q3bytes_sv = svsub_s8_x(svptrue_b8(), svreinterpret_s8_u8(svand_u8_m(svptrue_b8(), svlsr_n_u8_x(svptrue_b8(), q3bits_sv_1, 6), m3b_sv)), svreinterpret_s8_u8(q3h_sv));

                        sumi1_1 = svmla_s32_m(svptrue_b32(), sumi1_1, svdot_s32(vzero_sv, q3bytes_sv, q8bytes_1_sv_2), svdup_n_s32((int32_t)scale[3]));

                        if (j == 0) {
                            qhbits_sv_1 = svlsr_n_u8_x(svptrue_b8(), qhbits_sv_1, 4);
                            qhbits_sv_2 = svlsr_n_u8_x(svptrue_b8(), qhbits_sv_2, 4);
                        }

                        scale += 4;
                    }

                    sum += d * (svaddv_s32(svptrue_b32(), sumi1_1));
                } break;
            case 256:
            case 512:
                {
                    svuint8_t qhbits_sv = svld1_u8(svptrue_pat_b8(SV_VL32), qh_sv);
                    svuint8_t q3h_sv;

                    svint32_t sumi1_1 = svdup_n_s32(0);
                    svint8_t q3bytes_sv;

                    for (int j = 0; j < QK_K/128; ++j) {

                        const svuint8_t q3bits_sv = svld1_u8(svptrue_pat_b8(SV_VL32), q3_sv); q3_sv += 32;
                        svint8_t q8bytes_1_sv_1 = svld1_s8(svptrue_pat_b8(SV_VL32), q8_sv); q8_sv += 32;
                        svint8_t q8bytes_1_sv_2 = svld1_s8(svptrue_pat_b8(SV_VL32), q8_sv); q8_sv += 32;

                        q3h_sv = svlsl_n_u8_x(svptrue_pat_b8(SV_VL32), svbic_u8_x(svptrue_pat_b8(SV_VL32), m0_sv, qhbits_sv), 2);
                        q3bytes_sv = svsub_s8_x(svptrue_pat_b8(SV_VL32), svreinterpret_s8_u8(svand_u8_m(svptrue_pat_b8(SV_VL32), q3bits_sv, m3b_sv)), svreinterpret_s8_u8(q3h_sv));


                        svint32_t scale_1 = svsel_s32(svptrue_pat_b32(SV_VL4), svdup_n_s32((int32_t)scale[0]), svdup_n_s32((int32_t)scale[1]));
                        sumi1_1 = svmla_s32_m(svptrue_pat_b32(SV_VL8), sumi1_1, svdot_s32(vzero_sv, q3bytes_sv, q8bytes_1_sv_1), scale_1);

                        q3h_sv = svlsl_n_u8_x(svptrue_pat_b8(SV_VL32), svbic_u8_x(svptrue_pat_b8(SV_VL32), m1_sv, qhbits_sv), 1);
                        q3bytes_sv = svsub_s8_x(svptrue_pat_b8(SV_VL32), svreinterpret_s8_u8(svand_u8_m(svptrue_pat_b8(SV_VL32), svlsr_n_u8_x(svptrue_pat_b8(SV_VL32), q3bits_sv, 2), m3b_sv)), svreinterpret_s8_u8(q3h_sv));

                        scale_1 = svsel_s32(svptrue_pat_b32(SV_VL4), svdup_n_s32((int32_t)scale[2]), svdup_n_s32((int32_t)scale[3]));
                        sumi1_1 = svmla_s32_m(svptrue_pat_b32(SV_VL8), sumi1_1, svdot_s32(vzero_sv, q3bytes_sv, q8bytes_1_sv_2), scale_1);

                        scale += 4;
                        q8bytes_1_sv_1 = svld1_s8(svptrue_pat_b8(SV_VL32), q8_sv); q8_sv += 32;
                        q8bytes_1_sv_2 = svld1_s8(svptrue_pat_b8(SV_VL32), q8_sv); q8_sv += 32;

                        q3h_sv = svbic_u8_x(svptrue_pat_b8(SV_VL32), m2_sv, qhbits_sv);
                        q3bytes_sv = svsub_s8_x(svptrue_pat_b8(SV_VL32), svreinterpret_s8_u8(svand_u8_m(svptrue_pat_b8(SV_VL32), svlsr_n_u8_x(svptrue_pat_b8(SV_VL32), q3bits_sv, 4), m3b_sv)), svreinterpret_s8_u8(q3h_sv));

                        scale_1 = svsel_s32(svptrue_pat_b32(SV_VL4), svdup_n_s32((int32_t)scale[0]), svdup_n_s32((int32_t)scale[1]));
                        sumi1_1 = svmla_s32_m(svptrue_pat_b32(SV_VL8), sumi1_1, svdot_s32(vzero_sv, q3bytes_sv, q8bytes_1_sv_1), scale_1);

                        q3h_sv = svlsr_n_u8_x(svptrue_pat_b8(SV_VL32), svbic_u8_x(svptrue_pat_b8(SV_VL32), m3_sv, qhbits_sv), 1);
                        q3bytes_sv = svsub_s8_x(svptrue_pat_b8(SV_VL32), svreinterpret_s8_u8(svand_u8_m(svptrue_pat_b8(SV_VL32), svlsr_n_u8_x(svptrue_pat_b8(SV_VL32), q3bits_sv, 6), m3b_sv)), svreinterpret_s8_u8(q3h_sv));

                        scale_1 = svsel_s32(svptrue_pat_b32(SV_VL4), svdup_n_s32((int32_t)scale[2]), svdup_n_s32((int32_t)scale[3]));
                        sumi1_1 = svmla_s32_m(svptrue_pat_b32(SV_VL8), sumi1_1, svdot_s32(vzero_sv, q3bytes_sv, q8bytes_1_sv_2), scale_1);

                        if (j == 0) {
                            qhbits_sv = svlsr_n_u8_x(svptrue_pat_b8(SV_VL32), qhbits_sv, 4);
                        }

                        scale += 4;
                    }

                    sum += d * (svaddv_s32(svptrue_pat_b32(SV_VL8), sumi1_1));
                } break;
            default:
                assert(false && "Unsupported vector length");
                break;
        }
    }
    *s = sum;

#elif __ARM_NEON

    uint32_t aux[3];
    uint32_t utmp[4];

    const uint8x16_t m3b = vdupq_n_u8(0x3);
    const int32x4_t  vzero = vdupq_n_s32(0);

    const uint8x16_t m0 = vdupq_n_u8(1);
    const uint8x16_t m1 = vshlq_n_u8(m0, 1);
    const uint8x16_t m2 = vshlq_n_u8(m0, 2);
    const uint8x16_t m3 = vshlq_n_u8(m0, 3);
    const int8_t m32 = 32;

    ggml_int8x16x4_t q3bytes;

    float sum = 0;

    for (int i = 0; i < nb; ++i) {

        const float d = y[i].d * GGML_CPU_FP16_TO_FP32(x[i].d);

        const uint8_t * GGML_RESTRICT q3 = x[i].qs;
        const uint8_t * GGML_RESTRICT qh = x[i].hmask;
        const int8_t  * GGML_RESTRICT q8 = y[i].qs;

        ggml_uint8x16x2_t qhbits = ggml_vld1q_u8_x2(qh);

        ggml_uint8x16x4_t q3h;

        int32_t isum = 0;

        // Set up scales
        memcpy(aux, x[i].scales, 12);
        utmp[3] = ((aux[1] >> 4) & kmask2) | (((aux[2] >> 6) & kmask1) << 4);
        utmp[2] = ((aux[0] >> 4) & kmask2) | (((aux[2] >> 4) & kmask1) << 4);
        utmp[1] = (aux[1] & kmask2) | (((aux[2] >> 2) & kmask1) << 4);
        utmp[0] = (aux[0] & kmask2) | (((aux[2] >> 0) & kmask1) << 4);

        int8_t * scale = (int8_t *)utmp;
        for (int j = 0; j < 16; ++j) scale[j] -= m32;

        for (int j = 0; j < QK_K/128; ++j) {

            const ggml_uint8x16x2_t q3bits = ggml_vld1q_u8_x2(q3); q3 += 32;
            const ggml_int8x16x4_t q8bytes_1 = ggml_vld1q_s8_x4(q8); q8 += 64;
            const ggml_int8x16x4_t q8bytes_2 = ggml_vld1q_s8_x4(q8); q8 += 64;

            q3h.val[0] = vshlq_n_u8(vbicq_u8(m0, qhbits.val[0]), 2);
            q3h.val[1] = vshlq_n_u8(vbicq_u8(m0, qhbits.val[1]), 2);
            q3h.val[2] = vshlq_n_u8(vbicq_u8(m1, qhbits.val[0]), 1);
            q3h.val[3] = vshlq_n_u8(vbicq_u8(m1, qhbits.val[1]), 1);

            q3bytes.val[0] = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(q3bits.val[0], m3b)), vreinterpretq_s8_u8(q3h.val[0]));
            q3bytes.val[1] = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(q3bits.val[1], m3b)), vreinterpretq_s8_u8(q3h.val[1]));
            q3bytes.val[2] = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q3bits.val[0], 2), m3b)), vreinterpretq_s8_u8(q3h.val[2]));
            q3bytes.val[3] = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q3bits.val[1], 2), m3b)), vreinterpretq_s8_u8(q3h.val[3]));

            isum += vaddvq_s32(ggml_vdotq_s32(vzero, q3bytes.val[0], q8bytes_1.val[0])) * scale[0];
            isum += vaddvq_s32(ggml_vdotq_s32(vzero, q3bytes.val[1], q8bytes_1.val[1])) * scale[1];
            isum += vaddvq_s32(ggml_vdotq_s32(vzero, q3bytes.val[2], q8bytes_1.val[2])) * scale[2];
            isum += vaddvq_s32(ggml_vdotq_s32(vzero, q3bytes.val[3], q8bytes_1.val[3])) * scale[3];

            scale += 4;

            q3h.val[0] = vbicq_u8(m2, qhbits.val[0]);
            q3h.val[1] = vbicq_u8(m2, qhbits.val[1]);
            q3h.val[2] = vshrq_n_u8(vbicq_u8(m3, qhbits.val[0]), 1);
            q3h.val[3] = vshrq_n_u8(vbicq_u8(m3, qhbits.val[1]), 1);

            q3bytes.val[0] = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q3bits.val[0], 4), m3b)), vreinterpretq_s8_u8(q3h.val[0]));
            q3bytes.val[1] = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q3bits.val[1], 4), m3b)), vreinterpretq_s8_u8(q3h.val[1]));
            q3bytes.val[2] = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q3bits.val[0], 6), m3b)), vreinterpretq_s8_u8(q3h.val[2]));
            q3bytes.val[3] = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q3bits.val[1], 6), m3b)), vreinterpretq_s8_u8(q3h.val[3]));

            isum += vaddvq_s32(ggml_vdotq_s32(vzero, q3bytes.val[0], q8bytes_2.val[0])) * scale[0];
            isum += vaddvq_s32(ggml_vdotq_s32(vzero, q3bytes.val[1], q8bytes_2.val[1])) * scale[1];
            isum += vaddvq_s32(ggml_vdotq_s32(vzero, q3bytes.val[2], q8bytes_2.val[2])) * scale[2];
            isum += vaddvq_s32(ggml_vdotq_s32(vzero, q3bytes.val[3], q8bytes_2.val[3])) * scale[3];

            scale += 4;

            if (j == 0) {
                qhbits.val[0] = vshrq_n_u8(qhbits.val[0], 4);
                qhbits.val[1] = vshrq_n_u8(qhbits.val[1], 4);
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
#ifdef __ARM_FEATURE_MATMUL_INT8
    assert((nrc == 2) || (nrc == 1));
#else
    assert(nrc == 1);
#endif
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

#if defined(__ARM_FEATURE_MATMUL_INT8)
    if (nrc == 2) {
        const block_q4_K * GGML_RESTRICT x0 = x;
        const block_q4_K * GGML_RESTRICT x1 = (const block_q4_K *) ((const uint8_t *)vx + bx);
        const block_q8_K * GGML_RESTRICT y0 = y;
        const block_q8_K * GGML_RESTRICT y1 = (const block_q8_K *) ((const uint8_t *)vy + by);

        const uint8x16_t m4b = vdupq_n_u8(0x0f);

        float32x4_t vfsum = vdupq_n_f32(0.0f);

        for (int i = 0; i < nb; ++i, ++x0, ++x1, ++y0, ++y1) {
            const uint8_t * GGML_RESTRICT qx0 = x0->qs;
            const uint8_t * GGML_RESTRICT qx1 = x1->qs;
            const  int8_t * GGML_RESTRICT qy0 = y0->qs;
            const  int8_t * GGML_RESTRICT qy1 = y1->qs;

            // decode scales and mins
            int8_t x0_scales[8], x1_scales[8];
            int16x8_t x0_mins, x1_mins;
            {
                uint32_t scales_mins[3];
                memcpy(scales_mins, x0->scales, 12);
                const uint32_t mins_0_3 = scales_mins[1] & kmask1;
                const uint32_t mins_4_7 = ((scales_mins[2] >> 4) & kmask2) | (((scales_mins[1] >> 6) & kmask3) << 4);
                const uint32x2_t mins = {mins_0_3, mins_4_7};
                x0_mins = vreinterpretq_s16_u16(vmovl_u8(vreinterpret_u8_u32(mins)));
                uint32_t scales[2];
                scales[0] = scales_mins[0] & kmask1; // scales 0~3
                scales[1] = (scales_mins[2] & kmask2) | (((scales_mins[0] >> 6) & kmask3) << 4); // scales 4~7
                memcpy(x0_scales, scales, 8);
            }
            {
                uint32_t scales_mins[3];
                memcpy(scales_mins, x1->scales, 12);
                const uint32_t mins_0_3 = scales_mins[1] & kmask1;
                const uint32_t mins_4_7 = ((scales_mins[2] >> 4) & kmask2) | (((scales_mins[1] >> 6) & kmask3) << 4);
                const uint32x2_t mins = {mins_0_3, mins_4_7};
                x1_mins = vreinterpretq_s16_u16(vmovl_u8(vreinterpret_u8_u32(mins)));
                uint32_t scales[2];
                scales[0] = scales_mins[0] & kmask1; // scales 0~3
                scales[1] = (scales_mins[2] & kmask2) | (((scales_mins[0] >> 6) & kmask3) << 4); // scales 4~7
                memcpy(x1_scales, scales, 8);
            }

            int32x4_t visum = {0};

            // process 64 data points per iteration, totally 256 data points
            for (int j = 0; j < QK_K / 64; ++j, qx0 += 32, qx1 += 32, qy0 += 64, qy1 += 64) {
                const int8x16x4_t vy0 = vld1q_s8_x4(qy0);
                const int8x16x4_t vy1 = vld1q_s8_x4(qy1);

                int8x16_t vx0[4], vx1[4];
                {
                    const uint8x16x2_t vv = vld1q_u8_x2(qx0);
                    vx0[0] = vreinterpretq_s8_u8(vandq_u8(vv.val[0], m4b));
                    vx0[1] = vreinterpretq_s8_u8(vandq_u8(vv.val[1], m4b));
                    vx0[2] = vreinterpretq_s8_u8(vshrq_n_u8(vv.val[0], 4));
                    vx0[3] = vreinterpretq_s8_u8(vshrq_n_u8(vv.val[1], 4));
                }
                {
                    const uint8x16x2_t vv = vld1q_u8_x2(qx1);
                    vx1[0] = vreinterpretq_s8_u8(vandq_u8(vv.val[0], m4b));
                    vx1[1] = vreinterpretq_s8_u8(vandq_u8(vv.val[1], m4b));
                    vx1[2] = vreinterpretq_s8_u8(vshrq_n_u8(vv.val[0], 4));
                    vx1[3] = vreinterpretq_s8_u8(vshrq_n_u8(vv.val[1], 4));
                }

                // process 32 data points (share same block scale) per iteration
                for (int k = 0; k < 2; ++k) {
                    const int blk = j * 2 + k;
                    const int32x4_t block_scale = {
                        x0_scales[blk],
                        x0_scales[blk],
                        x1_scales[blk],
                        x1_scales[blk],
                    };

                    int32x4_t vr = {0};
                    for (int l = 0; l < 2; ++l) {
                        const int idx = k * 2 + l;
                        const int64x2_t vx0_s64 = vreinterpretq_s64_s8(vx0[idx]);
                        const int64x2_t vx1_s64 = vreinterpretq_s64_s8(vx1[idx]);
                        const int64x2_t vy0_s64 = vreinterpretq_s64_s8(vy0.val[idx]);
                        const int64x2_t vy1_s64 = vreinterpretq_s64_s8(vy1.val[idx]);
                        const int8x16_t vx_l = vreinterpretq_s8_s64(vzip1q_s64(vx0_s64, vx1_s64));
                        const int8x16_t vx_h = vreinterpretq_s8_s64(vzip2q_s64(vx0_s64, vx1_s64));
                        const int8x16_t vy_l = vreinterpretq_s8_s64(vzip1q_s64(vy0_s64, vy1_s64));
                        const int8x16_t vy_h = vreinterpretq_s8_s64(vzip2q_s64(vy0_s64, vy1_s64));
                        vr = vmmlaq_s32(vr, vx_l, vy_l);
                        vr = vmmlaq_s32(vr, vx_h, vy_h);
                    }
                    // apply block scale, will NOT overflow
                    // block_scale * sum_256(int4*int8) <= 2^(8+8+4+8) = 28 bits
                    visum = vmlaq_s32(visum, vr, block_scale);
                }
            }

            // adjust bias, apply superblock scale
            {
                int32_t bias[4];
                // no obvious uplift from sve sdot-16, just use neon mul add
                const int16x8_t y0_sums = vpaddq_s16(vld1q_s16(y0->bsums), vld1q_s16(y0->bsums+8));
                const int16x8_t y1_sums = vpaddq_s16(vld1q_s16(y1->bsums), vld1q_s16(y1->bsums+8));
                bias[0] = vaddvq_s32(vaddq_s32(vmull_s16(vget_low_s16(y0_sums), vget_low_s16(x0_mins)),
                                               vmull_s16(vget_high_s16(y0_sums), vget_high_s16(x0_mins))));
                bias[1] = vaddvq_s32(vaddq_s32(vmull_s16(vget_low_s16(y1_sums), vget_low_s16(x0_mins)),
                                               vmull_s16(vget_high_s16(y1_sums), vget_high_s16(x0_mins))));
                bias[2] = vaddvq_s32(vaddq_s32(vmull_s16(vget_low_s16(y0_sums), vget_low_s16(x1_mins)),
                                               vmull_s16(vget_high_s16(y0_sums), vget_high_s16(x1_mins))));
                bias[3] = vaddvq_s32(vaddq_s32(vmull_s16(vget_low_s16(y1_sums), vget_low_s16(x1_mins)),
                                               vmull_s16(vget_high_s16(y1_sums), vget_high_s16(x1_mins))));
                const float32x4_t dmins = {
                    GGML_CPU_FP16_TO_FP32(x0->dmin) * y0->d,
                    GGML_CPU_FP16_TO_FP32(x0->dmin) * y1->d,
                    GGML_CPU_FP16_TO_FP32(x1->dmin) * y0->d,
                    GGML_CPU_FP16_TO_FP32(x1->dmin) * y1->d,
                };
                vfsum = vmlsq_f32(vfsum, vcvtq_f32_s32(vld1q_s32(bias)), dmins);

                const float32x4_t superblock_scale = {
                    GGML_CPU_FP16_TO_FP32(x0->d) * y0->d,
                    GGML_CPU_FP16_TO_FP32(x0->d) * y1->d,
                    GGML_CPU_FP16_TO_FP32(x1->d) * y0->d,
                    GGML_CPU_FP16_TO_FP32(x1->d) * y1->d,
                };
                vfsum = vmlaq_f32(vfsum, vcvtq_f32_s32(visum), superblock_scale);
            }
        }

        // vfsum = ABCD -> ACBD
        // AC -> s, BD -> (s+bs)
        vfsum = vzip1q_f32(vfsum, vextq_f32(vfsum, vfsum, 2));
        vst1_f32(s,      vget_low_f32 (vfsum));
        vst1_f32(s + bs, vget_high_f32(vfsum));

        return;
    }
#endif

#ifdef __ARM_FEATURE_SVE
    float sumf = 0;
    for (int i = 0; i < nb; ++i) {

        const float d = y[i].d * GGML_CPU_FP16_TO_FP32(x[i].d);
        const float dmin = y[i].d * GGML_CPU_FP16_TO_FP32(x[i].dmin);

        const int16x8_t q8sums = vpaddq_s16(vld1q_s16(y[i].bsums), vld1q_s16(y[i].bsums + 8));

        memcpy(utmp, x[i].scales, K_SCALE_SIZE);

        uint32x2_t mins8 = { 0 };
        mins8 = vset_lane_u32(utmp[1] & kmask1, mins8, 0);
        mins8 = vset_lane_u32(((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4), mins8, 1);

        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[0] &= kmask1;

        const int16x8_t mins = vreinterpretq_s16_u16(vmovl_u8(vreinterpret_u8_u32(mins8)));
        const int32x4_t prod = vaddq_s32(vmull_s16(vget_low_s16 (q8sums), vget_low_s16 (mins)),
                                         vmull_s16(vget_high_s16(q8sums), vget_high_s16(mins)));
        sumf -= dmin * vaddvq_s32(prod);

        const uint8_t * scales = (const uint8_t *)utmp;

        const uint8_t * GGML_RESTRICT q4 = x[i].qs;
        const int8_t  * GGML_RESTRICT q8 = y[i].qs;

        const int vector_length = ggml_cpu_get_sve_cnt()*8;
        const svuint8_t m4b = svdup_n_u8(0xf);
        const svint32_t mzero = svdup_n_s32(0);
        svint32_t sumi1 = svdup_n_s32(0);
        svint32_t sumi1_1 = svdup_n_s32(0);
        svint32_t sumi1_2 = svdup_n_s32(0);
        svint32_t sumi2 = svdup_n_s32(0);
        svint32_t sumi2_1 = svdup_n_s32(0);
        svint32_t sumi2_2 = svdup_n_s32(0);
        switch (vector_length) {
            case 128:
                {
                    for (int j = 0; j < QK_K/64; ++j) {
                        svint8_t q4bytes = svreinterpret_s8_u8(svand_u8_x(svptrue_b8(), svld1_u8(svptrue_b8(), q4), m4b));
                        svint8_t q8bytes = svld1_s8(svptrue_b8(), q8); q8 += 16;
                        sumi1_1 = svmla_n_s32_x(svptrue_b32(), sumi1_1, svdot_s32(mzero, q4bytes, q8bytes), scales[2*j+0]);
                        q4bytes = svreinterpret_s8_u8(svand_u8_x(svptrue_b8(), svld1_u8(svptrue_b8(), q4+16), m4b));
                        q8bytes = svld1_s8(svptrue_b8(), q8); q8 += 16;
                        sumi1_2 = svmla_n_s32_x(svptrue_b32(), sumi1_2, svdot_s32(mzero, q4bytes, q8bytes), scales[2*j+0]);

                        q4bytes = svreinterpret_s8_u8(svlsr_n_u8_x(svptrue_b8(), svld1_u8(svptrue_b8(), q4), 4));
                        q8bytes = svld1_s8(svptrue_b8(), q8); q8 += 16;
                        sumi2_1 = svmla_n_s32_x(svptrue_b32(), sumi2_1, svdot_s32(mzero, q4bytes, q8bytes), scales[2*j+1]);
                        q4bytes = svreinterpret_s8_u8(svlsr_n_u8_x(svptrue_b8(), svld1_u8(svptrue_b8(), q4+16), 4));
                        q8bytes = svld1_s8(svptrue_b8(), q8); q8 += 16;
                        sumi2_2 = svmla_n_s32_x(svptrue_b32(), sumi2_2, svdot_s32(mzero, q4bytes, q8bytes), scales[2*j+1]);
                        q4 += 32;
                    }
                    sumi1 = svadd_s32_x(svptrue_b32(), sumi1_1, sumi1_2);
                    sumi2 = svadd_s32_x(svptrue_b32(), sumi2_1, sumi2_2);
                    sumf += d * (svaddv_s32(svptrue_b32(), svadd_s32_x(svptrue_b32(), sumi1, sumi2)));
                } break;
            case 256:
            case 512:
                {
                    for (int j = 0; j < QK_K/64; ++j) {
                        const svuint8_t q4bits  = svld1_u8(svptrue_pat_b8(SV_VL32), q4); q4 += 32;
                        svint8_t q4bytes = svreinterpret_s8_u8(svand_u8_x(svptrue_pat_b8(SV_VL32), q4bits, m4b));
                        svint8_t q8bytes = svld1_s8(svptrue_pat_b8(SV_VL32), q8); q8 += 32;
                        sumi1 = svmla_n_s32_x(svptrue_pat_b32(SV_VL8), sumi1, svdot_s32(mzero, q4bytes, q8bytes), scales[2*j+0]);

                        q4bytes = svreinterpret_s8_u8(svlsr_n_u8_x(svptrue_pat_b8(SV_VL32), q4bits, 4));
                        q8bytes = svld1_s8(svptrue_pat_b8(SV_VL32), q8); q8 += 32;
                        sumi2 = svmla_n_s32_x(svptrue_pat_b32(SV_VL8), sumi2, svdot_s32(mzero, q4bytes, q8bytes), scales[2*j+1]);
                    }
                    sumf += d * (svaddv_s32(svptrue_pat_b32(SV_VL8), svadd_s32_x(svptrue_pat_b32(SV_VL8), sumi1, sumi2)));
                } break;
            default:
                assert(false && "Unsupported vector length");
                break;
        }
    }
    *s = sumf;
#elif defined __ARM_NEON
    const uint8x16_t m4b = vdupq_n_u8(0xf);
    const int32x4_t mzero = vdupq_n_s32(0);

    ggml_int8x16x2_t q4bytes;
    ggml_int8x16x2_t q8bytes;

    float sumf = 0;

    for (int i = 0; i < nb; ++i) {

        const float d = y[i].d * GGML_CPU_FP16_TO_FP32(x[i].d);
        const float dmin = y[i].d * GGML_CPU_FP16_TO_FP32(x[i].dmin);

        const int16x8_t q8sums = vpaddq_s16(vld1q_s16(y[i].bsums), vld1q_s16(y[i].bsums + 8));

        memcpy(utmp, x[i].scales, 12);

        uint32x2_t mins8 = { 0 };
        mins8 = vset_lane_u32(utmp[1] & kmask1, mins8, 0);
        mins8 = vset_lane_u32(((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4), mins8, 1);

        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[0] &= kmask1;

        const int16x8_t mins = vreinterpretq_s16_u16(vmovl_u8(vreinterpret_u8_u32(mins8)));
        const int32x4_t prod = vaddq_s32(vmull_s16(vget_low_s16 (q8sums), vget_low_s16 (mins)),
                                         vmull_s16(vget_high_s16(q8sums), vget_high_s16(mins)));
        sumf -= dmin * vaddvq_s32(prod);

        const uint8_t * scales = (const uint8_t *)utmp;

        const uint8_t * GGML_RESTRICT q4 = x[i].qs;
        const int8_t  * GGML_RESTRICT q8 = y[i].qs;

        int32_t sumi1 = 0;
        int32_t sumi2 = 0;

        for (int j = 0; j < QK_K/64; ++j) {
            const ggml_uint8x16x2_t q4bits = ggml_vld1q_u8_x2(q4); q4 += 32;

            q8bytes = ggml_vld1q_s8_x2(q8); q8 += 32;
            q4bytes.val[0] = vreinterpretq_s8_u8(vandq_u8  (q4bits.val[0], m4b));
            q4bytes.val[1] = vreinterpretq_s8_u8(vandq_u8  (q4bits.val[1], m4b));

            const int32x4_t p1 = ggml_vdotq_s32(ggml_vdotq_s32(mzero, q4bytes.val[0], q8bytes.val[0]), q4bytes.val[1], q8bytes.val[1]);
            sumi1 += vaddvq_s32(p1) * scales[2*j+0];

            q8bytes = ggml_vld1q_s8_x2(q8); q8 += 32;
            q4bytes.val[0] = vreinterpretq_s8_u8(vshrq_n_u8(q4bits.val[0], 4));
            q4bytes.val[1] = vreinterpretq_s8_u8(vshrq_n_u8(q4bits.val[1], 4));

            const int32x4_t p2 = ggml_vdotq_s32(ggml_vdotq_s32(mzero, q4bytes.val[0], q8bytes.val[0]), q4bytes.val[1], q8bytes.val[1]);

            sumi2 += vaddvq_s32(p2) * scales[2*j+1];
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


#ifdef __ARM_NEON
    const uint8x16_t m4b = vdupq_n_u8(0xf);
    const uint8x16_t mone = vdupq_n_u8(1);
    const uint8x16_t mtwo = vdupq_n_u8(2);
    const int32x4_t mzero = vdupq_n_s32(0);

    ggml_int8x16x4_t q5bytes;

    float sumf = 0;

    for (int i = 0; i < nb; ++i) {

        const float d = y[i].d * GGML_CPU_FP16_TO_FP32(x[i].d);
        const float dmin = y[i].d * GGML_CPU_FP16_TO_FP32(x[i].dmin);

        const int16x8_t q8sums = vpaddq_s16(vld1q_s16(y[i].bsums), vld1q_s16(y[i].bsums + 8));

        memcpy(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        const uint8x8_t mins8 = vld1_u8((const uint8_t*)utmp + 8);
        const int16x8_t mins = vreinterpretq_s16_u16(vmovl_u8(mins8));
        const int32x4_t prod = vaddq_s32(vmull_s16(vget_low_s16 (q8sums), vget_low_s16 (mins)),
                                         vmull_s16(vget_high_s16(q8sums), vget_high_s16(mins)));
        int32_t sumi_mins = vaddvq_s32(prod);

        const uint8_t * scales = (const uint8_t *)utmp;

        const uint8_t * GGML_RESTRICT q5 = x[i].qs;
        const uint8_t * GGML_RESTRICT qh = x[i].qh;
        const int8_t  * GGML_RESTRICT q8 = y[i].qs;

        ggml_uint8x16x2_t qhbits = ggml_vld1q_u8_x2(qh);

        ggml_uint8x16x4_t q5h;

        int32_t sumi = 0;

        for (int j = 0; j < QK_K/64; ++j) {

            const ggml_uint8x16x2_t q5bits = ggml_vld1q_u8_x2(q5); q5 += 32;
            const ggml_int8x16x4_t q8bytes = ggml_vld1q_s8_x4(q8); q8 += 64;

            q5h.val[0] = vshlq_n_u8(vandq_u8(mone, qhbits.val[0]), 4);
            q5h.val[1] = vshlq_n_u8(vandq_u8(mone, qhbits.val[1]), 4);
            q5h.val[2] = vshlq_n_u8(vandq_u8(mtwo, qhbits.val[0]), 3);
            q5h.val[3] = vshlq_n_u8(vandq_u8(mtwo, qhbits.val[1]), 3);
            qhbits.val[0] = vshrq_n_u8(qhbits.val[0], 2);
            qhbits.val[1] = vshrq_n_u8(qhbits.val[1], 2);

            q5bytes.val[0] = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q5bits.val[0], m4b), q5h.val[0]));
            q5bytes.val[1] = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q5bits.val[1], m4b), q5h.val[1]));
            q5bytes.val[2] = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q5bits.val[0], 4), q5h.val[2]));
            q5bytes.val[3] = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q5bits.val[1], 4), q5h.val[3]));

            sumi += vaddvq_s32(ggml_vdotq_s32(ggml_vdotq_s32(mzero, q5bytes.val[0], q8bytes.val[0]), q5bytes.val[1], q8bytes.val[1])) * *scales++;
            sumi += vaddvq_s32(ggml_vdotq_s32(ggml_vdotq_s32(mzero, q5bytes.val[2], q8bytes.val[2]), q5bytes.val[3], q8bytes.val[3])) * *scales++;
        }

        sumf += d * sumi - dmin * sumi_mins;
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
#ifdef __ARM_FEATURE_MATMUL_INT8
    assert((nrc == 2) || (nrc == 1));
#else
    assert(nrc == 1);
#endif
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_q6_K * GGML_RESTRICT x = vx;
    const block_q8_K * GGML_RESTRICT y = vy;

    const int nb = n / QK_K;

#if defined(__ARM_FEATURE_MATMUL_INT8)
    if (nrc == 2) {
        const block_q6_K * GGML_RESTRICT x0 = x;
        const block_q6_K * GGML_RESTRICT x1 = (const block_q6_K *) ((const uint8_t *)vx + bx);
        const block_q8_K * GGML_RESTRICT y0 = y;
        const block_q8_K * GGML_RESTRICT y1 = (const block_q8_K *) ((const uint8_t *)vy + by);

        float32x4_t vfsum = vdupq_n_f32(0.0f);

        for (int i = 0; i < nb; ++i, ++x0, ++x1, ++y0, ++y1) {
            const uint8_t * GGML_RESTRICT ql0 = x0->ql;
            const uint8_t * GGML_RESTRICT ql1 = x1->ql;
            const uint8_t * GGML_RESTRICT qh0 = x0->qh;
            const uint8_t * GGML_RESTRICT qh1 = x1->qh;
            const  int8_t * GGML_RESTRICT qy0 = y0->qs;
            const  int8_t * GGML_RESTRICT qy1 = y1->qs;

            const uint8x16_t mone = vdupq_n_u8(0x30);
            const uint8x16_t  m4b = vdupq_n_u8(0x0f);

            int32x4_t visum = vdupq_n_s32(0);

            // process 8 blocks per iteration, totally 16 blocks
            for (int j = 0; j < 2; ++j, qh0 += 32, ql0 += 64, qh1 += 32, ql1 += 64) {
                int8x16_t vx0[8], vx1[8];

                // de-quantize vx0[8]
                {
                    const uint8x16x2_t qh_bits = vld1q_u8_x2(qh0);
                    const uint8x16x4_t ql_bits = vld1q_u8_x4(ql0);

                    uint8x16_t q6h_0 = vandq_u8(mone, vshlq_n_u8(qh_bits.val[0], 4));
                    uint8x16_t q6h_1 = vandq_u8(mone, vshlq_n_u8(qh_bits.val[1], 4));
                    uint8x16_t q6h_2 = vandq_u8(mone, vshlq_n_u8(qh_bits.val[0], 2));
                    uint8x16_t q6h_3 = vandq_u8(mone, vshlq_n_u8(qh_bits.val[1], 2));

                    vx0[0] = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(ql_bits.val[0], m4b), q6h_0));
                    vx0[1] = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(ql_bits.val[1], m4b), q6h_1));
                    vx0[2] = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(ql_bits.val[2], m4b), q6h_2));
                    vx0[3] = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(ql_bits.val[3], m4b), q6h_3));

                    q6h_0 = vandq_u8(mone, qh_bits.val[0]);
                    q6h_1 = vandq_u8(mone, qh_bits.val[1]);
                    q6h_2 = vandq_u8(mone, vshrq_n_u8(qh_bits.val[0], 2));
                    q6h_3 = vandq_u8(mone, vshrq_n_u8(qh_bits.val[1], 2));

                    vx0[4] = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(ql_bits.val[0], 4), q6h_0));
                    vx0[5] = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(ql_bits.val[1], 4), q6h_1));
                    vx0[6] = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(ql_bits.val[2], 4), q6h_2));
                    vx0[7] = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(ql_bits.val[3], 4), q6h_3));
                }

                // de-quantize vx1[8]
                {
                    const uint8x16x2_t qh_bits = vld1q_u8_x2(qh1);
                    const uint8x16x4_t ql_bits = vld1q_u8_x4(ql1);

                    uint8x16_t q6h_0 = vandq_u8(mone, vshlq_n_u8(qh_bits.val[0], 4));
                    uint8x16_t q6h_1 = vandq_u8(mone, vshlq_n_u8(qh_bits.val[1], 4));
                    uint8x16_t q6h_2 = vandq_u8(mone, vshlq_n_u8(qh_bits.val[0], 2));
                    uint8x16_t q6h_3 = vandq_u8(mone, vshlq_n_u8(qh_bits.val[1], 2));

                    vx1[0] = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(ql_bits.val[0], m4b), q6h_0));
                    vx1[1] = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(ql_bits.val[1], m4b), q6h_1));
                    vx1[2] = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(ql_bits.val[2], m4b), q6h_2));
                    vx1[3] = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(ql_bits.val[3], m4b), q6h_3));

                    q6h_0 = vandq_u8(mone, qh_bits.val[0]);
                    q6h_1 = vandq_u8(mone, qh_bits.val[1]);
                    q6h_2 = vandq_u8(mone, vshrq_n_u8(qh_bits.val[0], 2));
                    q6h_3 = vandq_u8(mone, vshrq_n_u8(qh_bits.val[1], 2));

                    vx1[4] = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(ql_bits.val[0], 4), q6h_0));
                    vx1[5] = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(ql_bits.val[1], 4), q6h_1));
                    vx1[6] = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(ql_bits.val[2], 4), q6h_2));
                    vx1[7] = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(ql_bits.val[3], 4), q6h_3));
                }

                // process 16 elements (one block with same scale) per iteration
                // - vx = concat(ql, qh) - 32
                // - r1,r2,r3,r4 = smmla(vx, vy)
                for (int k = 0; k < 8; ++k) {
                    const int blk = j * 8 + k;

                    const int8x16_t vy0 = vld1q_s8(qy0);
                    const int8x16_t vy1 = vld1q_s8(qy1);
                    qy0 += 16;
                    qy1 += 16;

                    const int32x4_t block_scale = {
                        x0->scales[blk],
                        x0->scales[blk],
                        x1->scales[blk],
                        x1->scales[blk],
                    };

                    // calculate four results at once with outer product
                    const int8x16_t vx_l = vreinterpretq_s8_s64(vzip1q_s64(vreinterpretq_s64_s8(vx0[k]), vreinterpretq_s64_s8(vx1[k])));
                    const int8x16_t vx_h = vreinterpretq_s8_s64(vzip2q_s64(vreinterpretq_s64_s8(vx0[k]), vreinterpretq_s64_s8(vx1[k])));
                    const int8x16_t vy_l = vreinterpretq_s8_s64(vzip1q_s64(vreinterpretq_s64_s8(vy0), vreinterpretq_s64_s8(vy1)));
                    const int8x16_t vy_h = vreinterpretq_s8_s64(vzip2q_s64(vreinterpretq_s64_s8(vy0), vreinterpretq_s64_s8(vy1)));
                    int32x4_t vr = vdupq_n_s32(0);
                    vr = vmmlaq_s32(vr, vx_l, vy_l);
                    vr = vmmlaq_s32(vr, vx_h, vy_h);

                    // apply block scale, will NOT overflow
                    // block_scale * sum_256(int6*int8) <= 2^(8+8+6+8) = 30 bits
                    visum = vmlaq_s32(visum, vr, block_scale);
                }
            }

            // adjust bias, apply superblock scale
            {
                int32_t bias[4];
#ifdef __ARM_FEATURE_SVE
                const svbool_t pg16_8 = svptrue_pat_b16(SV_VL8);
                const svbool_t pg8_8 = svptrue_pat_b8(SV_VL8);
                const svint16_t y0_q8sums_0 = svld1_s16(pg16_8, y0->bsums);
                const svint16_t y0_q8sums_1 = svld1_s16(pg16_8, y0->bsums + 8);
                const svint16_t y1_q8sums_0 = svld1_s16(pg16_8, y1->bsums);
                const svint16_t y1_q8sums_1 = svld1_s16(pg16_8, y1->bsums + 8);
                const svint16_t x0_q6scales_0 = svunpklo_s16(svld1_s8(pg8_8, x0->scales));
                const svint16_t x0_q6scales_1 = svunpklo_s16(svld1_s8(pg8_8, x0->scales + 8));
                const svint16_t x1_q6scales_0 = svunpklo_s16(svld1_s8(pg8_8, x1->scales));
                const svint16_t x1_q6scales_1 = svunpklo_s16(svld1_s8(pg8_8, x1->scales + 8));
                const svint64_t zero = svdup_n_s64(0);
                bias[0] = svaddv_s64(svptrue_b64(), svadd_s64_x(svptrue_b64(), svdot_s64(zero, y0_q8sums_0, x0_q6scales_0),
                                                                               svdot_s64(zero, y0_q8sums_1, x0_q6scales_1)));
                bias[1] = svaddv_s64(svptrue_b64(), svadd_s64_x(svptrue_b64(), svdot_s64(zero, y1_q8sums_0, x0_q6scales_0),
                                                                               svdot_s64(zero, y1_q8sums_1, x0_q6scales_1)));
                bias[2] = svaddv_s64(svptrue_b64(), svadd_s64_x(svptrue_b64(), svdot_s64(zero, y0_q8sums_0, x1_q6scales_0),
                                                                               svdot_s64(zero, y0_q8sums_1, x1_q6scales_1)));
                bias[3] = svaddv_s64(svptrue_b64(), svadd_s64_x(svptrue_b64(), svdot_s64(zero, y1_q8sums_0, x1_q6scales_0),
                                                                               svdot_s64(zero, y1_q8sums_1, x1_q6scales_1)));
#else
                // NEON doesn't support int16 dot product, fallback to separated mul and add
                const int16x8x2_t q8sums0 = vld1q_s16_x2(y0->bsums);
                const int16x8x2_t q8sums1 = vld1q_s16_x2(y1->bsums);

                int8x16_t scales_s8 = vld1q_s8(x0->scales);
                const int16x8x2_t q6scales0 = {{vmovl_s8(vget_low_s8(scales_s8)), vmovl_s8(vget_high_s8(scales_s8))}};
                scales_s8 = vld1q_s8(x1->scales);
                const int16x8x2_t q6scales1 = {{vmovl_s8(vget_low_s8(scales_s8)), vmovl_s8(vget_high_s8(scales_s8))}};

                int32x4_t prod;
                prod = vaddq_s32(vaddq_s32(vmull_s16(vget_low_s16 (q8sums0.val[0]), vget_low_s16 (q6scales0.val[0])),
                                           vmull_s16(vget_high_s16(q8sums0.val[0]), vget_high_s16(q6scales0.val[0]))),
                                 vaddq_s32(vmull_s16(vget_low_s16 (q8sums0.val[1]), vget_low_s16 (q6scales0.val[1])),
                                           vmull_s16(vget_high_s16(q8sums0.val[1]), vget_high_s16(q6scales0.val[1]))));
                bias[0] = vaddvq_s32(prod);
                prod = vaddq_s32(vaddq_s32(vmull_s16(vget_low_s16 (q8sums1.val[0]), vget_low_s16 (q6scales0.val[0])),
                                           vmull_s16(vget_high_s16(q8sums1.val[0]), vget_high_s16(q6scales0.val[0]))),
                                 vaddq_s32(vmull_s16(vget_low_s16 (q8sums1.val[1]), vget_low_s16 (q6scales0.val[1])),
                                           vmull_s16(vget_high_s16(q8sums1.val[1]), vget_high_s16(q6scales0.val[1]))));
                bias[1] = vaddvq_s32(prod);
                prod = vaddq_s32(vaddq_s32(vmull_s16(vget_low_s16 (q8sums0.val[0]), vget_low_s16 (q6scales1.val[0])),
                                           vmull_s16(vget_high_s16(q8sums0.val[0]), vget_high_s16(q6scales1.val[0]))),
                                 vaddq_s32(vmull_s16(vget_low_s16 (q8sums0.val[1]), vget_low_s16 (q6scales1.val[1])),
                                           vmull_s16(vget_high_s16(q8sums0.val[1]), vget_high_s16(q6scales1.val[1]))));
                bias[2] = vaddvq_s32(prod);
                prod = vaddq_s32(vaddq_s32(vmull_s16(vget_low_s16 (q8sums1.val[0]), vget_low_s16 (q6scales1.val[0])),
                                           vmull_s16(vget_high_s16(q8sums1.val[0]), vget_high_s16(q6scales1.val[0]))),
                                 vaddq_s32(vmull_s16(vget_low_s16 (q8sums1.val[1]), vget_low_s16 (q6scales1.val[1])),
                                           vmull_s16(vget_high_s16(q8sums1.val[1]), vget_high_s16(q6scales1.val[1]))));
                bias[3] = vaddvq_s32(prod);

#endif
                const int32x4_t vibias = vmulq_n_s32(vld1q_s32(bias), 32);

                const float32x4_t superblock_scale = {
                    GGML_CPU_FP16_TO_FP32(x0->d) * y0->d,
                    GGML_CPU_FP16_TO_FP32(x0->d) * y1->d,
                    GGML_CPU_FP16_TO_FP32(x1->d) * y0->d,
                    GGML_CPU_FP16_TO_FP32(x1->d) * y1->d,
                };

                visum = vsubq_s32(visum, vibias);
                vfsum = vmlaq_f32(vfsum, vcvtq_f32_s32(visum), superblock_scale);
            }
        }

        // vfsum = ABCD -> ACBD
        // AC -> s, BD -> (s+bs)
        vfsum = vzip1q_f32(vfsum, vextq_f32(vfsum, vfsum, 2));
        vst1_f32(s,      vget_low_f32 (vfsum));
        vst1_f32(s + bs, vget_high_f32(vfsum));

        return;
    }
#endif

#ifdef __ARM_FEATURE_SVE
    const int vector_length = ggml_cpu_get_sve_cnt()*8;
    float sum = 0;
    svuint8_t m4b = svdup_n_u8(0xf);
    svint32_t vzero = svdup_n_s32(0);
    svuint8_t mone = svdup_n_u8(0x30);
    svint8_t q6bytes_1, q6bytes_2, q6bytes_3, q6bytes_4;
    svuint8_t q6h_1, q6h_2, q6h_3, q6h_4;

    for (int i = 0; i < nb; ++i) {
        const float d_all = GGML_CPU_FP16_TO_FP32(x[i].d);

        const uint8_t * GGML_RESTRICT q6 = x[i].ql;
        const uint8_t * GGML_RESTRICT qh = x[i].qh;
        const int8_t  * GGML_RESTRICT q8 = y[i].qs;

        const int8_t * GGML_RESTRICT scale = x[i].scales;

        const svbool_t pg16_8 = svptrue_pat_b16(SV_VL8);
        const svint16_t q8sums_1 = svld1_s16(pg16_8, y[i].bsums);
        const svint16_t q8sums_2 = svld1_s16(pg16_8, y[i].bsums + 8);
        const svint16_t q6scales_1 = svunpklo_s16(svld1_s8(svptrue_pat_b8(SV_VL8), scale));
        const svint16_t q6scales_2 = svunpklo_s16(svld1_s8(svptrue_pat_b8(SV_VL8), scale + 8));
        const svint64_t prod = svdup_n_s64(0);
        int32_t isum_mins = svaddv_s64(svptrue_b64(), svadd_s64_x(svptrue_b64(), svdot_s64(prod, q8sums_1, q6scales_1),
                                                                                 svdot_s64(prod, q8sums_2, q6scales_2)));
        int32_t isum = 0;

        switch (vector_length) {
            case 128:
                {
                    const svbool_t pg32_4 = svptrue_pat_b32(SV_VL4);
                    const svbool_t pg8_16 = svptrue_pat_b8(SV_VL16);
                    svint32_t isum_tmp = svdup_n_s32(0);
                    for (int j = 0; j < QK_K/128; ++j) {
                        svuint8_t qhbits_1 = svld1_u8(pg8_16, qh);
                        svuint8_t qhbits_2 = svld1_u8(pg8_16, qh+16);
                        qh += 32;
                        svuint8_t q6bits_1 = svld1_u8(pg8_16, q6);
                        svuint8_t q6bits_2 = svld1_u8(pg8_16, q6+16);
                        svuint8_t q6bits_3 = svld1_u8(pg8_16, q6+32);
                        svuint8_t q6bits_4 = svld1_u8(pg8_16, q6+48);
                        q6 += 64;
                        svint8_t q8bytes_1 = svld1_s8(pg8_16, q8);
                        svint8_t q8bytes_2 = svld1_s8(pg8_16, q8+16);
                        svint8_t q8bytes_3 = svld1_s8(pg8_16, q8+32);
                        svint8_t q8bytes_4 = svld1_s8(pg8_16, q8+48);
                        q8 += 64;

                        q6h_1 = svand_u8_x(pg16_8, mone, svlsl_n_u8_x(pg16_8, qhbits_1, 4));
                        q6h_2 = svand_u8_x(pg16_8, mone, svlsl_n_u8_x(pg16_8, qhbits_2, 4));
                        q6h_3 = svand_u8_x(pg16_8, mone, svlsl_n_u8_x(pg16_8, qhbits_1, 2));
                        q6h_4 = svand_u8_x(pg16_8, mone, svlsl_n_u8_x(pg16_8, qhbits_2, 2));
                        q6bytes_1 = svreinterpret_s8_u8(svorr_u8_x(pg8_16, svand_u8_x(pg8_16, q6bits_1, m4b), q6h_1));
                        q6bytes_2 = svreinterpret_s8_u8(svorr_u8_x(pg8_16, svand_u8_x(pg8_16, q6bits_2, m4b), q6h_2));
                        q6bytes_3 = svreinterpret_s8_u8(svorr_u8_x(pg8_16, svand_u8_x(pg8_16, q6bits_3, m4b), q6h_3));
                        q6bytes_4 = svreinterpret_s8_u8(svorr_u8_x(pg8_16, svand_u8_x(pg8_16, q6bits_4, m4b), q6h_4));
                        isum_tmp = svmla_n_s32_x(pg32_4, isum_tmp, svdot_s32(vzero, q6bytes_1, q8bytes_1), scale[0]);
                        isum_tmp = svmla_n_s32_x(pg32_4, isum_tmp, svdot_s32(vzero, q6bytes_2, q8bytes_2), scale[1]);
                        isum_tmp = svmla_n_s32_x(pg32_4, isum_tmp, svdot_s32(vzero, q6bytes_3, q8bytes_3), scale[2]);
                        isum_tmp = svmla_n_s32_x(pg32_4, isum_tmp, svdot_s32(vzero, q6bytes_4, q8bytes_4), scale[3]);

                        scale += 4;
                        q8bytes_1 = svld1_s8(pg8_16, q8);
                        q8bytes_2 = svld1_s8(pg8_16, q8+16);
                        q8bytes_3 = svld1_s8(pg8_16, q8+32);
                        q8bytes_4 = svld1_s8(pg8_16, q8+48);
                        q8 += 64;

                        q6h_1 = svand_u8_x(pg16_8, mone, qhbits_1);
                        q6h_2 = svand_u8_x(pg16_8, mone, qhbits_2);
                        q6h_3 = svand_u8_x(pg16_8, mone, svlsr_n_u8_x(pg16_8, qhbits_1, 2));
                        q6h_4 = svand_u8_x(pg16_8, mone, svlsr_n_u8_x(pg16_8, qhbits_2, 2));
                        q6bytes_1 = svreinterpret_s8_u8(svorr_u8_x(pg8_16, svlsr_n_u8_x(pg8_16, q6bits_1, 4), q6h_1));
                        q6bytes_2 = svreinterpret_s8_u8(svorr_u8_x(pg8_16, svlsr_n_u8_x(pg8_16, q6bits_2, 4), q6h_2));
                        q6bytes_3 = svreinterpret_s8_u8(svorr_u8_x(pg8_16, svlsr_n_u8_x(pg8_16, q6bits_3, 4), q6h_3));
                        q6bytes_4 = svreinterpret_s8_u8(svorr_u8_x(pg8_16, svlsr_n_u8_x(pg8_16, q6bits_4, 4), q6h_4));
                        isum_tmp = svmla_n_s32_x(pg32_4, isum_tmp, svdot_s32(vzero, q6bytes_1, q8bytes_1), scale[0]);
                        isum_tmp = svmla_n_s32_x(pg32_4, isum_tmp, svdot_s32(vzero, q6bytes_2, q8bytes_2), scale[1]);
                        isum_tmp = svmla_n_s32_x(pg32_4, isum_tmp, svdot_s32(vzero, q6bytes_3, q8bytes_3), scale[2]);
                        isum_tmp = svmla_n_s32_x(pg32_4, isum_tmp, svdot_s32(vzero, q6bytes_4, q8bytes_4), scale[3]);
                        scale += 4;
                    }
                    isum += svaddv_s32(pg32_4, isum_tmp);
                    sum += d_all * y[i].d * (isum - 32 * isum_mins);
                }
                break;
            case 256:
            case 512:
                {
                    const svbool_t pg8_2 = svptrue_pat_b8(SV_VL2);
                    const svbool_t pg32_8 = svptrue_pat_b32(SV_VL8);
                    const svbool_t pg8_32 = svptrue_pat_b8(SV_VL32);
                    svint32_t isum_tmp = svdup_n_s32(0);
                    for (int j = 0; j < QK_K/128; j++) {
                        svuint8_t qhbits_1 = svld1_u8(pg8_32, qh);
                        qh += 32;
                        svuint8_t q6bits_1 = svld1_u8(pg8_32, q6);
                        svuint8_t q6bits_2 = svld1_u8(pg8_32, q6+32);
                        q6 += 64;
                        svint8_t q8bytes_1 = svld1_s8(pg8_32, q8);
                        svint8_t q8bytes_2 = svld1_s8(pg8_32, q8+32);
                        svint8_t q8bytes_3 = svld1_s8(pg8_32, q8+64);
                        svint8_t q8bytes_4 = svld1_s8(pg8_32, q8+96);
                        q8 += 128;
                        q6h_1 = svand_u8_x(pg8_32, mone, svlsl_n_u8_x(pg8_32, qhbits_1, 4));
                        q6h_2 = svand_u8_x(pg8_32, mone, svlsl_n_u8_x(pg8_32, qhbits_1, 2));
                        q6h_3 = svand_u8_x(pg8_32, mone, qhbits_1);
                        q6h_4 = svand_u8_x(pg8_32, mone, svlsr_n_u8_x(pg8_32, qhbits_1, 2));
                        q6bytes_1 = svreinterpret_s8_u8(svorr_u8_x(pg8_32, svand_u8_x(pg8_32, q6bits_1, m4b), q6h_1));
                        q6bytes_2 = svreinterpret_s8_u8(svorr_u8_x(pg8_32, svand_u8_x(pg8_32, q6bits_2, m4b), q6h_2));
                        q6bytes_3 = svreinterpret_s8_u8(svorr_u8_x(pg8_32, svlsr_n_u8_x(pg8_32, q6bits_1, 4), q6h_3));
                        q6bytes_4 = svreinterpret_s8_u8(svorr_u8_x(pg8_32, svlsr_n_u8_x(pg8_32, q6bits_2, 4), q6h_4));

                        svint8_t scale_lane_1_tmp = svld1_s8(pg8_2, scale);
                        scale_lane_1_tmp= svzip1_s8(scale_lane_1_tmp, scale_lane_1_tmp);
                        scale_lane_1_tmp= svzip1_s8(scale_lane_1_tmp, scale_lane_1_tmp);
                        svint8_t scale_lane_2_tmp = svld1_s8(pg8_2, scale+2);
                        scale_lane_2_tmp = svzip1_s8(scale_lane_2_tmp, scale_lane_2_tmp);
                        scale_lane_2_tmp = svzip1_s8(scale_lane_2_tmp, scale_lane_2_tmp);
                        svint8_t scale_lane_3_tmp = svld1_s8(pg8_2, scale+4);
                        scale_lane_3_tmp = svzip1_s8(scale_lane_3_tmp, scale_lane_3_tmp);
                        scale_lane_3_tmp = svzip1_s8(scale_lane_3_tmp, scale_lane_3_tmp);
                        svint8_t scale_lane_4_tmp = svld1_s8(pg8_2, scale+6);
                        scale_lane_4_tmp = svzip1_s8(scale_lane_4_tmp, scale_lane_4_tmp);
                        scale_lane_4_tmp = svzip1_s8(scale_lane_4_tmp, scale_lane_4_tmp);
                        svint32_t scale_lane_1 = svunpklo_s32(svunpklo_s16(scale_lane_1_tmp));
                        svint32_t scale_lane_2 = svunpklo_s32(svunpklo_s16(scale_lane_2_tmp));
                        svint32_t scale_lane_3 = svunpklo_s32(svunpklo_s16(scale_lane_3_tmp));
                        svint32_t scale_lane_4 = svunpklo_s32(svunpklo_s16(scale_lane_4_tmp));

                        isum_tmp = svmla_s32_x(pg32_8, isum_tmp, svdot_s32(vzero, q6bytes_1, q8bytes_1), scale_lane_1);
                        isum_tmp = svmla_s32_x(pg32_8, isum_tmp, svdot_s32(vzero, q6bytes_2, q8bytes_2), scale_lane_2);
                        isum_tmp = svmla_s32_x(pg32_8, isum_tmp, svdot_s32(vzero, q6bytes_3, q8bytes_3), scale_lane_3);
                        isum_tmp = svmla_s32_x(pg32_8, isum_tmp, svdot_s32(vzero, q6bytes_4, q8bytes_4), scale_lane_4);
                        scale += 8;
                    }
                    isum += svaddv_s32(pg32_8, isum_tmp);
                    sum += d_all * y[i].d * (isum - 32 * isum_mins);
                }
                break;
            default:
                assert(false && "Unsupported vector length");
                break;
        }
    }

    *s = sum;

#elif __ARM_NEON
    float sum = 0;

    const uint8x16_t m4b = vdupq_n_u8(0xF);
    const int32x4_t  vzero = vdupq_n_s32(0);
    //const int8x16_t  m32s = vdupq_n_s8(32);

    const uint8x16_t mone = vdupq_n_u8(3);

    ggml_int8x16x4_t q6bytes;
    ggml_uint8x16x4_t q6h;

    for (int i = 0; i < nb; ++i) {

        const float d_all = GGML_CPU_FP16_TO_FP32(x[i].d);

        const uint8_t * GGML_RESTRICT q6 = x[i].ql;
        const uint8_t * GGML_RESTRICT qh = x[i].qh;
        const int8_t  * GGML_RESTRICT q8 = y[i].qs;

        const int8_t * GGML_RESTRICT scale = x[i].scales;

        const ggml_int16x8x2_t q8sums = ggml_vld1q_s16_x2(y[i].bsums);
        const int8x16_t scales = vld1q_s8(scale);
        const ggml_int16x8x2_t q6scales = {{vmovl_s8(vget_low_s8(scales)), vmovl_s8(vget_high_s8(scales))}};

        const int32x4_t prod = vaddq_s32(vaddq_s32(vmull_s16(vget_low_s16 (q8sums.val[0]), vget_low_s16 (q6scales.val[0])),
                                                   vmull_s16(vget_high_s16(q8sums.val[0]), vget_high_s16(q6scales.val[0]))),
                                         vaddq_s32(vmull_s16(vget_low_s16 (q8sums.val[1]), vget_low_s16 (q6scales.val[1])),
                                                   vmull_s16(vget_high_s16(q8sums.val[1]), vget_high_s16(q6scales.val[1]))));
        int32_t isum_mins = vaddvq_s32(prod);

        int32_t isum = 0;

        for (int j = 0; j < QK_K/128; ++j) {

            ggml_uint8x16x2_t qhbits = ggml_vld1q_u8_x2(qh); qh += 32;
            ggml_uint8x16x4_t q6bits = ggml_vld1q_u8_x4(q6); q6 += 64;
            ggml_int8x16x4_t q8bytes = ggml_vld1q_s8_x4(q8); q8 += 64;

            q6h.val[0] = vshlq_n_u8(vandq_u8(mone, qhbits.val[0]), 4);
            q6h.val[1] = vshlq_n_u8(vandq_u8(mone, qhbits.val[1]), 4);
            uint8x16_t shifted = vshrq_n_u8(qhbits.val[0], 2);
            q6h.val[2] = vshlq_n_u8(vandq_u8(mone, shifted), 4);
            shifted = vshrq_n_u8(qhbits.val[1], 2);
            q6h.val[3] = vshlq_n_u8(vandq_u8(mone, shifted), 4);

            //q6bytes.val[0] = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q6bits.val[0], m4b), q6h.val[0])), m32s);
            //q6bytes.val[1] = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q6bits.val[1], m4b), q6h.val[1])), m32s);
            //q6bytes.val[2] = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q6bits.val[2], m4b), q6h.val[2])), m32s);
            //q6bytes.val[3] = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q6bits.val[3], m4b), q6h.val[3])), m32s);
            q6bytes.val[0] = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q6bits.val[0], m4b), q6h.val[0]));
            q6bytes.val[1] = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q6bits.val[1], m4b), q6h.val[1]));
            q6bytes.val[2] = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q6bits.val[2], m4b), q6h.val[2]));
            q6bytes.val[3] = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q6bits.val[3], m4b), q6h.val[3]));

            isum += vaddvq_s32(ggml_vdotq_s32(vzero, q6bytes.val[0], q8bytes.val[0])) * scale[0] +
                    vaddvq_s32(ggml_vdotq_s32(vzero, q6bytes.val[1], q8bytes.val[1])) * scale[1] +
                    vaddvq_s32(ggml_vdotq_s32(vzero, q6bytes.val[2], q8bytes.val[2])) * scale[2] +
                    vaddvq_s32(ggml_vdotq_s32(vzero, q6bytes.val[3], q8bytes.val[3])) * scale[3];

            scale += 4;

            q8bytes = ggml_vld1q_s8_x4(q8); q8 += 64;

            shifted = vshrq_n_u8(qhbits.val[0], 4);
            q6h.val[0] = vshlq_n_u8(vandq_u8(mone, shifted), 4);
            shifted = vshrq_n_u8(qhbits.val[1], 4);
            q6h.val[1] = vshlq_n_u8(vandq_u8(mone, shifted), 4);
            shifted = vshrq_n_u8(qhbits.val[0], 6);
            q6h.val[2] = vshlq_n_u8(vandq_u8(mone, shifted), 4);
            shifted = vshrq_n_u8(qhbits.val[1], 6);
            q6h.val[3] = vshlq_n_u8(vandq_u8(mone, shifted), 4);

            //q6bytes.val[0] = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6bits.val[0], 4), q6h.val[0])), m32s);
            //q6bytes.val[1] = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6bits.val[1], 4), q6h.val[1])), m32s);
            //q6bytes.val[2] = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6bits.val[2], 4), q6h.val[2])), m32s);
            //q6bytes.val[3] = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6bits.val[3], 4), q6h.val[3])), m32s);
            q6bytes.val[0] = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6bits.val[0], 4), q6h.val[0]));
            q6bytes.val[1] = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6bits.val[1], 4), q6h.val[1]));
            q6bytes.val[2] = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6bits.val[2], 4), q6h.val[2]));
            q6bytes.val[3] = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6bits.val[3], 4), q6h.val[3]));

            isum += vaddvq_s32(ggml_vdotq_s32(vzero, q6bytes.val[0], q8bytes.val[0])) * scale[0] +
                    vaddvq_s32(ggml_vdotq_s32(vzero, q6bytes.val[1], q8bytes.val[1])) * scale[1] +
                    vaddvq_s32(ggml_vdotq_s32(vzero, q6bytes.val[2], q8bytes.val[2])) * scale[2] +
                    vaddvq_s32(ggml_vdotq_s32(vzero, q6bytes.val[3], q8bytes.val[3])) * scale[3];
            scale += 4;
        }
        //sum += isum * d_all * y[i].d;
        sum += d_all * y[i].d * (isum - 32 * isum_mins);

    }
    *s = sum;
#else
    UNUSED(x);
    UNUSED(y);
    UNUSED(nb);
    ggml_vec_dot_q6_K_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#endif
}

#if defined (__ARM_NEON)
static const int8_t keven_signs_q2xs[1024] = {
     1,  1,  1,  1,  1,  1,  1,  1, -1,  1,  1,  1,  1,  1,  1, -1,  1, -1,  1,  1,  1,  1,  1, -1, -1, -1,  1,  1,  1,  1,  1,  1,
     1,  1, -1,  1,  1,  1,  1, -1, -1,  1, -1,  1,  1,  1,  1,  1,  1, -1, -1,  1,  1,  1,  1,  1, -1, -1, -1,  1,  1,  1,  1, -1,
     1,  1,  1, -1,  1,  1,  1, -1, -1,  1,  1, -1,  1,  1,  1,  1,  1, -1,  1, -1,  1,  1,  1,  1, -1, -1,  1, -1,  1,  1,  1, -1,
     1,  1, -1, -1,  1,  1,  1,  1, -1,  1, -1, -1,  1,  1,  1, -1,  1, -1, -1, -1,  1,  1,  1, -1, -1, -1, -1, -1,  1,  1,  1,  1,
     1,  1,  1,  1, -1,  1,  1, -1, -1,  1,  1,  1, -1,  1,  1,  1,  1, -1,  1,  1, -1,  1,  1,  1, -1, -1,  1,  1, -1,  1,  1, -1,
     1,  1, -1,  1, -1,  1,  1,  1, -1,  1, -1,  1, -1,  1,  1, -1,  1, -1, -1,  1, -1,  1,  1, -1, -1, -1, -1,  1, -1,  1,  1,  1,
     1,  1,  1, -1, -1,  1,  1,  1, -1,  1,  1, -1, -1,  1,  1, -1,  1, -1,  1, -1, -1,  1,  1, -1, -1, -1,  1, -1, -1,  1,  1,  1,
     1,  1, -1, -1, -1,  1,  1, -1, -1,  1, -1, -1, -1,  1,  1,  1,  1, -1, -1, -1, -1,  1,  1,  1, -1, -1, -1, -1, -1,  1,  1, -1,
     1,  1,  1,  1,  1, -1,  1, -1, -1,  1,  1,  1,  1, -1,  1,  1,  1, -1,  1,  1,  1, -1,  1,  1, -1, -1,  1,  1,  1, -1,  1, -1,
     1,  1, -1,  1,  1, -1,  1,  1, -1,  1, -1,  1,  1, -1,  1, -1,  1, -1, -1,  1,  1, -1,  1, -1, -1, -1, -1,  1,  1, -1,  1,  1,
     1,  1,  1, -1,  1, -1,  1,  1, -1,  1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1, -1, -1,  1, -1,  1, -1,  1,  1,
     1,  1, -1, -1,  1, -1,  1, -1, -1,  1, -1, -1,  1, -1,  1,  1,  1, -1, -1, -1,  1, -1,  1,  1, -1, -1, -1, -1,  1, -1,  1, -1,
     1,  1,  1,  1, -1, -1,  1,  1, -1,  1,  1,  1, -1, -1,  1, -1,  1, -1,  1,  1, -1, -1,  1, -1, -1, -1,  1,  1, -1, -1,  1,  1,
     1,  1, -1,  1, -1, -1,  1, -1, -1,  1, -1,  1, -1, -1,  1,  1,  1, -1, -1,  1, -1, -1,  1,  1, -1, -1, -1,  1, -1, -1,  1, -1,
     1,  1,  1, -1, -1, -1,  1, -1, -1,  1,  1, -1, -1, -1,  1,  1,  1, -1,  1, -1, -1, -1,  1,  1, -1, -1,  1, -1, -1, -1,  1, -1,
     1,  1, -1, -1, -1, -1,  1,  1, -1,  1, -1, -1, -1, -1,  1, -1,  1, -1, -1, -1, -1, -1,  1, -1, -1, -1, -1, -1, -1, -1,  1,  1,
     1,  1,  1,  1,  1,  1, -1, -1, -1,  1,  1,  1,  1,  1, -1,  1,  1, -1,  1,  1,  1,  1, -1,  1, -1, -1,  1,  1,  1,  1, -1, -1,
     1,  1, -1,  1,  1,  1, -1,  1, -1,  1, -1,  1,  1,  1, -1, -1,  1, -1, -1,  1,  1,  1, -1, -1, -1, -1, -1,  1,  1,  1, -1,  1,
     1,  1,  1, -1,  1,  1, -1,  1, -1,  1,  1, -1,  1,  1, -1, -1,  1, -1,  1, -1,  1,  1, -1, -1, -1, -1,  1, -1,  1,  1, -1,  1,
     1,  1, -1, -1,  1,  1, -1, -1, -1,  1, -1, -1,  1,  1, -1,  1,  1, -1, -1, -1,  1,  1, -1,  1, -1, -1, -1, -1,  1,  1, -1, -1,
     1,  1,  1,  1, -1,  1, -1,  1, -1,  1,  1,  1, -1,  1, -1, -1,  1, -1,  1,  1, -1,  1, -1, -1, -1, -1,  1,  1, -1,  1, -1,  1,
     1,  1, -1,  1, -1,  1, -1, -1, -1,  1, -1,  1, -1,  1, -1,  1,  1, -1, -1,  1, -1,  1, -1,  1, -1, -1, -1,  1, -1,  1, -1, -1,
     1,  1,  1, -1, -1,  1, -1, -1, -1,  1,  1, -1, -1,  1, -1,  1,  1, -1,  1, -1, -1,  1, -1,  1, -1, -1,  1, -1, -1,  1, -1, -1,
     1,  1, -1, -1, -1,  1, -1,  1, -1,  1, -1, -1, -1,  1, -1, -1,  1, -1, -1, -1, -1,  1, -1, -1, -1, -1, -1, -1, -1,  1, -1,  1,
     1,  1,  1,  1,  1, -1, -1,  1, -1,  1,  1,  1,  1, -1, -1, -1,  1, -1,  1,  1,  1, -1, -1, -1, -1, -1,  1,  1,  1, -1, -1,  1,
     1,  1, -1,  1,  1, -1, -1, -1, -1,  1, -1,  1,  1, -1, -1,  1,  1, -1, -1,  1,  1, -1, -1,  1, -1, -1, -1,  1,  1, -1, -1, -1,
     1,  1,  1, -1,  1, -1, -1, -1, -1,  1,  1, -1,  1, -1, -1,  1,  1, -1,  1, -1,  1, -1, -1,  1, -1, -1,  1, -1,  1, -1, -1, -1,
     1,  1, -1, -1,  1, -1, -1,  1, -1,  1, -1, -1,  1, -1, -1, -1,  1, -1, -1, -1,  1, -1, -1, -1, -1, -1, -1, -1,  1, -1, -1,  1,
     1,  1,  1,  1, -1, -1, -1, -1, -1,  1,  1,  1, -1, -1, -1,  1,  1, -1,  1,  1, -1, -1, -1,  1, -1, -1,  1,  1, -1, -1, -1, -1,
     1,  1, -1,  1, -1, -1, -1,  1, -1,  1, -1,  1, -1, -1, -1, -1,  1, -1, -1,  1, -1, -1, -1, -1, -1, -1, -1,  1, -1, -1, -1,  1,
     1,  1,  1, -1, -1, -1, -1,  1, -1,  1,  1, -1, -1, -1, -1, -1,  1, -1,  1, -1, -1, -1, -1, -1, -1, -1,  1, -1, -1, -1, -1,  1,
     1,  1, -1, -1, -1, -1, -1, -1, -1,  1, -1, -1, -1, -1, -1,  1,  1, -1, -1, -1, -1, -1, -1,  1, -1, -1, -1, -1, -1, -1, -1, -1,
};
#endif

void ggml_vec_dot_iq2_xxs_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_iq2_xxs * GGML_RESTRICT x = vx;
    const block_q8_K    * GGML_RESTRICT y = vy;

    const int nb = n / QK_K;

#if defined(__ARM_NEON)

    const uint64_t * signs64 = (const uint64_t *)keven_signs_q2xs;

    uint32_t aux32[4];
    const uint8_t * aux8 = (const uint8_t *)aux32;

    ggml_int8x16x4_t q2u;
    ggml_int8x16x4_t q2s;
    ggml_int8x16x4_t q8b;

    float sumf = 0;
    for (int i = 0; i < nb; ++i) {
        const float d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;
        const uint16_t * GGML_RESTRICT q2 = x[i].qs;
        const int8_t   * GGML_RESTRICT q8 = y[i].qs;
        float sumf1 = 0, sumf2 = 0;
        for (int ib32 = 0; ib32 < QK_K/32; ib32 += 2) {
            q8b = ggml_vld1q_s8_x4(q8); q8 += 64;
            memcpy(aux32, q2, 4*sizeof(uint32_t)); q2 += 8;
            q2u.val[0] = vcombine_s8(vld1_s8((const void *)(iq2xxs_grid + aux8[ 0])), vld1_s8((const void *)(iq2xxs_grid + aux8[ 1])));
            q2u.val[1] = vcombine_s8(vld1_s8((const void *)(iq2xxs_grid + aux8[ 2])), vld1_s8((const void *)(iq2xxs_grid + aux8[ 3])));
            q2u.val[2] = vcombine_s8(vld1_s8((const void *)(iq2xxs_grid + aux8[ 8])), vld1_s8((const void *)(iq2xxs_grid + aux8[ 9])));
            q2u.val[3] = vcombine_s8(vld1_s8((const void *)(iq2xxs_grid + aux8[10])), vld1_s8((const void *)(iq2xxs_grid + aux8[11])));
            q2s.val[0] = vcombine_s8(vld1_s8((const void *)(signs64 + ((aux32[1] >>  0) & 127))), vld1_s8((const void *)(signs64 + ((aux32[1] >>  7) & 127))));
            q2s.val[1] = vcombine_s8(vld1_s8((const void *)(signs64 + ((aux32[1] >> 14) & 127))), vld1_s8((const void *)(signs64 + ((aux32[1] >> 21) & 127))));
            q2s.val[2] = vcombine_s8(vld1_s8((const void *)(signs64 + ((aux32[3] >>  0) & 127))), vld1_s8((const void *)(signs64 + ((aux32[3] >>  7) & 127))));
            q2s.val[3] = vcombine_s8(vld1_s8((const void *)(signs64 + ((aux32[3] >> 14) & 127))), vld1_s8((const void *)(signs64 + ((aux32[3] >> 21) & 127))));
            q2u.val[0] = vmulq_s8(q2u.val[0], q2s.val[0]);
            q2u.val[1] = vmulq_s8(q2u.val[1], q2s.val[1]);
            q2u.val[2] = vmulq_s8(q2u.val[2], q2s.val[2]);
            q2u.val[3] = vmulq_s8(q2u.val[3], q2s.val[3]);
            const int32x4_t p1 = ggml_vdotq_s32(ggml_vdotq_s32(vdupq_n_s32(0), q2u.val[0], q8b.val[0]), q2u.val[1], q8b.val[1]);
            const int32x4_t p2 = ggml_vdotq_s32(ggml_vdotq_s32(vdupq_n_s32(0), q2u.val[2], q8b.val[2]), q2u.val[3], q8b.val[3]);
            sumf1 += vaddvq_s32(p1) * (0.5f + (aux32[1] >> 28));
            sumf2 += vaddvq_s32(p2) * (0.5f + (aux32[3] >> 28));
        }
        sumf += d*(sumf1 + sumf2);
    }
    *s = 0.25f * sumf;

#else
    UNUSED(x);
    UNUSED(y);
    UNUSED(nb);
    ggml_vec_dot_iq2_xxs_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#endif
}

void ggml_vec_dot_iq2_xs_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_iq2_xs * GGML_RESTRICT x = vx;
    const block_q8_K   * GGML_RESTRICT y = vy;

    const int nb = n / QK_K;

#if defined(__ARM_NEON)

    const uint64_t * signs64 = (const uint64_t *)keven_signs_q2xs;

    ggml_int8x16x4_t q2u;
    ggml_int8x16x4_t q2s;
    ggml_int8x16x4_t q8b;

    int32x4x4_t scales32;

    float sumf = 0;
    for (int i = 0; i < nb; ++i) {
        const float d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;
        const uint16_t * GGML_RESTRICT q2 = x[i].qs;
        const int8_t   * GGML_RESTRICT q8 = y[i].qs;
        const uint8x8_t scales8 = vld1_u8(x[i].scales);
        const uint8x8_t scales_l = vand_u8(scales8, vdup_n_u8(0xf));
        const uint8x8_t scales_h = vshr_n_u8(scales8, 4);
        uint8x16_t scales = vcombine_u8(vzip1_u8(scales_l, scales_h), vzip2_u8(scales_l, scales_h));
        scales = vaddq_u8(vshlq_n_u8(scales, 1), vdupq_n_u8(1));
        const uint16x8_t scales1 = vmovl_u8(vget_low_u8(scales));
        const uint16x8_t scales2 = vmovl_u8(vget_high_u8(scales));
        scales32.val[0] = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(scales1)));
        scales32.val[1] = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(scales1)));
        scales32.val[2] = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(scales2)));
        scales32.val[3] = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(scales2)));
        int32x4_t sumi = vdupq_n_s32(0);
        for (int ib64 = 0; ib64 < QK_K/64; ++ib64) {
            q8b = ggml_vld1q_s8_x4(q8); q8 += 64;
            q2u.val[0] = vcombine_s8(vld1_s8((const void *)(iq2xs_grid + (q2[0] & 511))), vld1_s8((const void *)(iq2xs_grid + (q2[1] & 511))));
            q2u.val[1] = vcombine_s8(vld1_s8((const void *)(iq2xs_grid + (q2[2] & 511))), vld1_s8((const void *)(iq2xs_grid + (q2[3] & 511))));
            q2u.val[2] = vcombine_s8(vld1_s8((const void *)(iq2xs_grid + (q2[4] & 511))), vld1_s8((const void *)(iq2xs_grid + (q2[5] & 511))));
            q2u.val[3] = vcombine_s8(vld1_s8((const void *)(iq2xs_grid + (q2[6] & 511))), vld1_s8((const void *)(iq2xs_grid + (q2[7] & 511))));
            q2s.val[0] = vcombine_s8(vld1_s8((const void *)(signs64 + (q2[0] >> 9))), vld1_s8((const void *)(signs64 + (q2[1] >> 9))));
            q2s.val[1] = vcombine_s8(vld1_s8((const void *)(signs64 + (q2[2] >> 9))), vld1_s8((const void *)(signs64 + (q2[3] >> 9))));
            q2s.val[2] = vcombine_s8(vld1_s8((const void *)(signs64 + (q2[4] >> 9))), vld1_s8((const void *)(signs64 + (q2[5] >> 9))));
            q2s.val[3] = vcombine_s8(vld1_s8((const void *)(signs64 + (q2[6] >> 9))), vld1_s8((const void *)(signs64 + (q2[7] >> 9))));
            q2u.val[0] = vmulq_s8(q2u.val[0], q2s.val[0]);
            q2u.val[1] = vmulq_s8(q2u.val[1], q2s.val[1]);
            q2u.val[2] = vmulq_s8(q2u.val[2], q2s.val[2]);
            q2u.val[3] = vmulq_s8(q2u.val[3], q2s.val[3]);
            const int32x4_t p1 = ggml_vdotq_s32(vdupq_n_s32(0), q2u.val[0], q8b.val[0]);
            const int32x4_t p2 = ggml_vdotq_s32(vdupq_n_s32(0), q2u.val[1], q8b.val[1]);
            const int32x4_t p3 = ggml_vdotq_s32(vdupq_n_s32(0), q2u.val[2], q8b.val[2]);
            const int32x4_t p4 = ggml_vdotq_s32(vdupq_n_s32(0), q2u.val[3], q8b.val[3]);
            const int32x4_t p = vpaddq_s32(vpaddq_s32(p1, p2), vpaddq_s32(p3, p4));
            sumi = vmlaq_s32(sumi, p, scales32.val[ib64]);
            q2 += 8;
        }
        sumf += d*vaddvq_s32(sumi);
    }
    *s = 0.125f * sumf;

#else
    UNUSED(x);
    UNUSED(y);
    UNUSED(nb);
    ggml_vec_dot_iq2_xs_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#endif
}

void ggml_vec_dot_iq2_s_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_iq2_s * GGML_RESTRICT x = vx;
    const block_q8_K  * GGML_RESTRICT y = vy;

    const int nb = n / QK_K;

#if defined(__ARM_NEON)

   static const uint8_t k_mask1[32] = {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
                                       0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03
   };

    static const uint8_t k_mask2[16] = {0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,};

    const ggml_uint8x16x2_t mask1 = ggml_vld1q_u8_x2(k_mask1);
    const uint8x16_t        mask2 = vld1q_u8(k_mask2);
    const uint8x16_t m1 = vdupq_n_u8(1);
    const int32x4_t vzero = vdupq_n_s32(0);

    uint8x16x2_t vs;
    ggml_int8x16x4_t q2s;
    ggml_int8x16x4_t q8b;

    float sumf = 0;
    for (int i = 0; i < nb; ++i) {

        const float d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;

        const uint8_t * GGML_RESTRICT qs = x[i].qs;
        const uint8_t * GGML_RESTRICT qh = x[i].qh;
        const uint16_t * GGML_RESTRICT signs = (const uint16_t *)(x[i].qs + QK_K/8);
        const int8_t  * GGML_RESTRICT q8 = y[i].qs;

        int sumi1 = 0, sumi2 = 0;
        for (int ib32 = 0; ib32 < QK_K/32; ib32 += 2) {
            q8b = ggml_vld1q_s8_x4(q8); q8 += 64;
            q2s.val[0] = vcombine_s8(vld1_s8((const int8_t *)(iq2s_grid + (qs[0] | ((qh[ib32+0] << 8) & 0x300)))),
                                     vld1_s8((const int8_t *)(iq2s_grid + (qs[1] | ((qh[ib32+0] << 6) & 0x300)))));
            q2s.val[1] = vcombine_s8(vld1_s8((const int8_t *)(iq2s_grid + (qs[2] | ((qh[ib32+0] << 4) & 0x300)))),
                                     vld1_s8((const int8_t *)(iq2s_grid + (qs[3] | ((qh[ib32+0] << 2) & 0x300)))));
            q2s.val[2] = vcombine_s8(vld1_s8((const int8_t *)(iq2s_grid + (qs[4] | ((qh[ib32+1] << 8) & 0x300)))),
                                     vld1_s8((const int8_t *)(iq2s_grid + (qs[5] | ((qh[ib32+1] << 6) & 0x300)))));
            q2s.val[3] = vcombine_s8(vld1_s8((const int8_t *)(iq2s_grid + (qs[6] | ((qh[ib32+1] << 4) & 0x300)))),
                                     vld1_s8((const int8_t *)(iq2s_grid + (qs[7] | ((qh[ib32+1] << 2) & 0x300)))));
            qs += 8;

            vs.val[0] = vreinterpretq_u8_u32(vdupq_n_u32(signs[0] | ((uint32_t) signs[1] << 16)));
            vs.val[1] = vandq_u8(ggml_vqtbl1q_u8(vs.val[0], mask1.val[1]), mask2);
            vs.val[0] = vandq_u8(ggml_vqtbl1q_u8(vs.val[0], mask1.val[0]), mask2);
            vs.val[0] = vceqq_u8(vs.val[0], mask2);
            vs.val[1] = vceqq_u8(vs.val[1], mask2);

            q2s.val[0] = vmulq_s8(vreinterpretq_s8_u8(vorrq_u8(vs.val[0], m1)), q2s.val[0]);
            q2s.val[1] = vmulq_s8(vreinterpretq_s8_u8(vorrq_u8(vs.val[1], m1)), q2s.val[1]);

            vs.val[0] = vreinterpretq_u8_u32(vdupq_n_u32(signs[2] | ((uint32_t) signs[3] << 16)));
            vs.val[1] = vandq_u8(ggml_vqtbl1q_u8(vs.val[0], mask1.val[1]), mask2);
            vs.val[0] = vandq_u8(ggml_vqtbl1q_u8(vs.val[0], mask1.val[0]), mask2);
            vs.val[0] = vceqq_u8(vs.val[0], mask2);
            vs.val[1] = vceqq_u8(vs.val[1], mask2);

            signs += 4;

            q2s.val[2] = vmulq_s8(vreinterpretq_s8_u8(vorrq_u8(vs.val[0], m1)), q2s.val[2]);
            q2s.val[3] = vmulq_s8(vreinterpretq_s8_u8(vorrq_u8(vs.val[1], m1)), q2s.val[3]);

            const int32x4_t p1 = ggml_vdotq_s32(vzero, q2s.val[0], q8b.val[0]);
            const int32x4_t p2 = ggml_vdotq_s32(vzero, q2s.val[1], q8b.val[1]);
            const int32x4_t p3 = ggml_vdotq_s32(vzero, q2s.val[2], q8b.val[2]);
            const int32x4_t p4 = ggml_vdotq_s32(vzero, q2s.val[3], q8b.val[3]);

            sumi1 += vaddvq_s32(p1) * (1 + 2*(x[i].scales[ib32+0] & 0xf));
            sumi2 += vaddvq_s32(p2) * (1 + 2*(x[i].scales[ib32+0] >>  4));
            sumi1 += vaddvq_s32(p3) * (1 + 2*(x[i].scales[ib32+1] & 0xf));
            sumi2 += vaddvq_s32(p4) * (1 + 2*(x[i].scales[ib32+1] >>  4));
        }
        sumf += d*(sumi1 + sumi2);
    }

    *s = 0.125f * sumf;

#else
    UNUSED(x);
    UNUSED(y);
    UNUSED(nb);
    ggml_vec_dot_iq2_s_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#endif

}

void ggml_vec_dot_iq3_xxs_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_iq3_xxs * GGML_RESTRICT x = vx;
    const block_q8_K    * GGML_RESTRICT y = vy;

    const int nb = n / QK_K;

#if defined(__ARM_NEON)

    const uint64_t * signs64 = (const uint64_t *)keven_signs_q2xs;

    uint32_t aux32[2];

    ggml_int8x16x4_t q3s;
    ggml_int8x16x4_t q8b;

    float sumf = 0;
    for (int i = 0; i < nb; ++i) {
        const float d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;
        const uint8_t * GGML_RESTRICT q3 = x[i].qs;
        const uint8_t * GGML_RESTRICT gas = x[i].qs + QK_K/4;
        const int8_t   * GGML_RESTRICT q8 = y[i].qs;
        float sumf1 = 0, sumf2 = 0;
        for (int ib32 = 0; ib32 < QK_K/32; ib32 += 2) {
            q8b = ggml_vld1q_s8_x4(q8); q8 += 64;
            memcpy(aux32, gas, 2*sizeof(uint32_t)); gas += 2*sizeof(uint32_t);
            const uint32x4_t aux32x4_0 = ggml_vld1q_u32(iq3xxs_grid[q3[ 0]], iq3xxs_grid[q3[ 1]], iq3xxs_grid[q3[ 2]], iq3xxs_grid[q3[ 3]]);
            const uint32x4_t aux32x4_1 = ggml_vld1q_u32(iq3xxs_grid[q3[ 4]], iq3xxs_grid[q3[ 5]], iq3xxs_grid[q3[ 6]], iq3xxs_grid[q3[ 7]]);
            const uint32x4_t aux32x4_2 = ggml_vld1q_u32(iq3xxs_grid[q3[ 8]], iq3xxs_grid[q3[ 9]], iq3xxs_grid[q3[10]], iq3xxs_grid[q3[11]]);
            const uint32x4_t aux32x4_3 = ggml_vld1q_u32(iq3xxs_grid[q3[12]], iq3xxs_grid[q3[13]], iq3xxs_grid[q3[14]], iq3xxs_grid[q3[15]]);
            q3 += 16;
            q3s.val[0] = vcombine_s8(vld1_s8((const void *)(signs64 + ((aux32[0] >>  0) & 127))), vld1_s8((const void *)(signs64 + ((aux32[0] >>  7) & 127))));
            q3s.val[1] = vcombine_s8(vld1_s8((const void *)(signs64 + ((aux32[0] >> 14) & 127))), vld1_s8((const void *)(signs64 + ((aux32[0] >> 21) & 127))));
            q3s.val[2] = vcombine_s8(vld1_s8((const void *)(signs64 + ((aux32[1] >>  0) & 127))), vld1_s8((const void *)(signs64 + ((aux32[1] >>  7) & 127))));
            q3s.val[3] = vcombine_s8(vld1_s8((const void *)(signs64 + ((aux32[1] >> 14) & 127))), vld1_s8((const void *)(signs64 + ((aux32[1] >> 21) & 127))));
            q3s.val[0] = vmulq_s8(q3s.val[0], vreinterpretq_s8_u32(aux32x4_0));
            q3s.val[1] = vmulq_s8(q3s.val[1], vreinterpretq_s8_u32(aux32x4_1));
            q3s.val[2] = vmulq_s8(q3s.val[2], vreinterpretq_s8_u32(aux32x4_2));
            q3s.val[3] = vmulq_s8(q3s.val[3], vreinterpretq_s8_u32(aux32x4_3));
            const int32x4_t p1 = ggml_vdotq_s32(ggml_vdotq_s32(vdupq_n_s32(0), q3s.val[0], q8b.val[0]), q3s.val[1], q8b.val[1]);
            const int32x4_t p2 = ggml_vdotq_s32(ggml_vdotq_s32(vdupq_n_s32(0), q3s.val[2], q8b.val[2]), q3s.val[3], q8b.val[3]);
            sumf1 += vaddvq_s32(p1) * (0.5f + (aux32[0] >> 28));
            sumf2 += vaddvq_s32(p2) * (0.5f + (aux32[1] >> 28));
        }
        sumf += d*(sumf1 + sumf2);
    }
    *s = 0.5f * sumf;

#else
    UNUSED(x);
    UNUSED(y);
    UNUSED(nb);
    ggml_vec_dot_iq3_xxs_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#endif
}

void ggml_vec_dot_iq3_s_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_iq3_s * GGML_RESTRICT x = vx;
    const block_q8_K  * GGML_RESTRICT y = vy;

    const int nb = n / QK_K;

#if defined(__ARM_NEON)

    typedef union {
        uint16x8_t vec_index;
        uint16_t   index[8];
    } vec_index_t;

   static const uint8_t k_mask1[32] = {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
                                       0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03
   };

    static const uint8_t k_mask2[16] = {0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,};

    static const int16_t k_shift[8] = {8, 7, 6, 5, 4, 3, 2, 1};

    const ggml_uint8x16x2_t mask1 = ggml_vld1q_u8_x2(k_mask1);
    const uint8x16_t        mask2 = vld1q_u8(k_mask2);

    const int16x8_t  hshift = vld1q_s16(k_shift);
    const uint16x8_t m256   = vdupq_n_u16(256);
    const uint8x16_t m1     = vdupq_n_u8(1);

    uint8x16x2_t vs;
    ggml_int8x16x4_t q3s;
    ggml_int8x16x4_t q8b;
    vec_index_t idx;

    uint32_t scales32[2];
    const uint8_t * scales8 = (const uint8_t *)scales32;

    float sumf = 0;
    for (int i = 0; i < nb; ++i) {
        const float d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;
        const uint8_t * GGML_RESTRICT qs = x[i].qs;
        const uint8_t * GGML_RESTRICT qh = x[i].qh;
        const uint16_t * GGML_RESTRICT signs = (const uint16_t *)x[i].signs;
        const int8_t   * GGML_RESTRICT q8 = y[i].qs;

        memcpy(scales32, x[i].scales, 4);
        scales32[1] = (((scales32[0] >> 4) & 0x0f0f0f0f) << 1) | 0x01010101;
        scales32[0] = ((scales32[0] & 0x0f0f0f0f) << 1) | 0x01010101;

        int sumi1 = 0, sumi2 = 0;
        for (int ib32 = 0; ib32 < QK_K/32; ib32 += 2) {
            q8b = ggml_vld1q_s8_x4(q8); q8 += 64;

            const uint8x16_t idx_l = vld1q_u8(qs); qs += 16;
            idx.vec_index = vorrq_u16(vmovl_u8(vget_low_u8 (idx_l)), vandq_u16(vshlq_u16(vdupq_n_u16(qh[ib32+0]), hshift), m256));
            const uint32x4_t aux32x4_0 = ggml_vld1q_u32(iq3s_grid[idx.index[0]], iq3s_grid[idx.index[1]],
                                                        iq3s_grid[idx.index[2]], iq3s_grid[idx.index[3]]);
            const uint32x4_t aux32x4_1 = ggml_vld1q_u32(iq3s_grid[idx.index[4]], iq3s_grid[idx.index[5]],
                                                        iq3s_grid[idx.index[6]], iq3s_grid[idx.index[7]]);
            idx.vec_index = vorrq_u16(vmovl_u8(vget_high_u8(idx_l)), vandq_u16(vshlq_u16(vdupq_n_u16(qh[ib32+1]), hshift), m256));
            const uint32x4_t aux32x4_2 = ggml_vld1q_u32(iq3s_grid[idx.index[0]], iq3s_grid[idx.index[1]],
                                                        iq3s_grid[idx.index[2]], iq3s_grid[idx.index[3]]);
            const uint32x4_t aux32x4_3 = ggml_vld1q_u32(iq3s_grid[idx.index[4]], iq3s_grid[idx.index[5]],
                                                        iq3s_grid[idx.index[6]], iq3s_grid[idx.index[7]]);


            vs.val[0] = vreinterpretq_u8_u32(vdupq_n_u32(signs[0] | ((uint32_t) signs[1] << 16)));
            vs.val[1] = vandq_u8(ggml_vqtbl1q_u8(vs.val[0], mask1.val[1]), mask2);
            vs.val[0] = vandq_u8(ggml_vqtbl1q_u8(vs.val[0], mask1.val[0]), mask2);
            vs.val[0] = vorrq_u8(vceqq_u8(vs.val[0], mask2), m1);
            vs.val[1] = vorrq_u8(vceqq_u8(vs.val[1], mask2), m1);

            q3s.val[0] = vmulq_s8(vreinterpretq_s8_u8(vs.val[0]), vreinterpretq_s8_u32(aux32x4_0));
            q3s.val[1] = vmulq_s8(vreinterpretq_s8_u8(vs.val[1]), vreinterpretq_s8_u32(aux32x4_1));

            vs.val[0] = vreinterpretq_u8_u32(vdupq_n_u32(signs[2] | ((uint32_t) signs[3] << 16)));
            vs.val[1] = vandq_u8(ggml_vqtbl1q_u8(vs.val[0], mask1.val[1]), mask2);
            vs.val[0] = vandq_u8(ggml_vqtbl1q_u8(vs.val[0], mask1.val[0]), mask2);
            vs.val[0] = vorrq_u8(vceqq_u8(vs.val[0], mask2), m1);
            vs.val[1] = vorrq_u8(vceqq_u8(vs.val[1], mask2), m1);

            signs += 4;

            q3s.val[2] = vmulq_s8(vreinterpretq_s8_u8(vs.val[0]), vreinterpretq_s8_u32(aux32x4_2));
            q3s.val[3] = vmulq_s8(vreinterpretq_s8_u8(vs.val[1]), vreinterpretq_s8_u32(aux32x4_3));

            const int32x4_t p1 = ggml_vdotq_s32(ggml_vdotq_s32(vdupq_n_s32(0), q3s.val[0], q8b.val[0]), q3s.val[1], q8b.val[1]);
            const int32x4_t p2 = ggml_vdotq_s32(ggml_vdotq_s32(vdupq_n_s32(0), q3s.val[2], q8b.val[2]), q3s.val[3], q8b.val[3]);

            sumi1 += vaddvq_s32(p1) * scales8[ib32/2+0];
            sumi2 += vaddvq_s32(p2) * scales8[ib32/2+4];
        }
        sumf += d*(sumi1 + sumi2);
    }
    *s = sumf;

#else
    UNUSED(x);
    UNUSED(y);
    UNUSED(nb);
    ggml_vec_dot_iq3_s_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#endif
}

void ggml_vec_dot_iq1_s_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_iq1_s * GGML_RESTRICT x = vx;
    const block_q8_K  * GGML_RESTRICT y = vy;

    const int nb = n / QK_K;

#if defined __ARM_NEON

    ggml_int8x16x4_t q1b;
    ggml_int8x16x4_t q8b;

    float sumf = 0;
    for (int i = 0; i < nb; ++i) {

        const int8_t   * q8 = y[i].qs;
        const uint8_t  * qs = x[i].qs;
        const uint16_t * qh = x[i].qh;

        int sumi1 = 0, sumi2 = 0, sumi3 = 0;

        for (int ib = 0; ib < QK_K/32; ib += 2) {

            q1b.val[0] = vcombine_s8(vld1_s8((const int8_t *)(iq1s_grid + (qs[0] | ((qh[ib+0] << 8) & 0x700)))),
                                     vld1_s8((const int8_t *)(iq1s_grid + (qs[1] | ((qh[ib+0] << 5) & 0x700)))));
            q1b.val[1] = vcombine_s8(vld1_s8((const int8_t *)(iq1s_grid + (qs[2] | ((qh[ib+0] << 2) & 0x700)))),
                                     vld1_s8((const int8_t *)(iq1s_grid + (qs[3] | ((qh[ib+0] >> 1) & 0x700)))));
            q1b.val[2] = vcombine_s8(vld1_s8((const int8_t *)(iq1s_grid + (qs[4] | ((qh[ib+1] << 8) & 0x700)))),
                                     vld1_s8((const int8_t *)(iq1s_grid + (qs[5] | ((qh[ib+1] << 5) & 0x700)))));
            q1b.val[3] = vcombine_s8(vld1_s8((const int8_t *)(iq1s_grid + (qs[6] | ((qh[ib+1] << 2) & 0x700)))),
                                     vld1_s8((const int8_t *)(iq1s_grid + (qs[7] | ((qh[ib+1] >> 1) & 0x700)))));
            qs += 8;

            q8b = ggml_vld1q_s8_x4(q8); q8 += 64;

            const int32x4_t p1 = ggml_vdotq_s32(ggml_vdotq_s32(vdupq_n_s32(0), q1b.val[0], q8b.val[0]), q1b.val[1], q8b.val[1]);
            const int32x4_t p2 = ggml_vdotq_s32(ggml_vdotq_s32(vdupq_n_s32(0), q1b.val[2], q8b.val[2]), q1b.val[3], q8b.val[3]);

            const int ls1 = 2*((qh[ib+0] >> 12) & 7) + 1;
            const int ls2 = 2*((qh[ib+1] >> 12) & 7) + 1;
            sumi1 += vaddvq_s32(p1) * ls1;
            sumi2 += vaddvq_s32(p2) * ls2;
            sumi3 += (y[i].bsums[2*ib+0] + y[i].bsums[2*ib+1]) * ls1 * (qh[ib+0] & 0x8000 ? -1 : 1)
                   + (y[i].bsums[2*ib+2] + y[i].bsums[2*ib+3]) * ls2 * (qh[ib+1] & 0x8000 ? -1 : 1);

        }

        sumf += y[i].d * GGML_CPU_FP16_TO_FP32(x[i].d) * (sumi1 + sumi2 + IQ1S_DELTA * sumi3);
    }

    *s = sumf;

#else
    UNUSED(x);
    UNUSED(y);
    UNUSED(nb);
    ggml_vec_dot_iq1_s_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#endif
}

void ggml_vec_dot_iq1_m_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_iq1_m * GGML_RESTRICT x = vx;
    const block_q8_K  * GGML_RESTRICT y = vy;

    const int nb = n / QK_K;

    iq1m_scale_t scale;

#if defined __ARM_NEON
    const int32x4_t mask  = vdupq_n_s32(0x7);
    const int32x4_t mone  = vdupq_n_s32(1);
    const int32x4_t mzero = vdupq_n_s32(0);

    ggml_int8x16x4_t deltas;
    deltas.val[0] = vcombine_s8(vdup_n_s8(+1), vdup_n_s8(+1));
    deltas.val[1] = vcombine_s8(vdup_n_s8(-1), vdup_n_s8(+1));
    deltas.val[2] = vcombine_s8(vdup_n_s8(+1), vdup_n_s8(-1));
    deltas.val[3] = vcombine_s8(vdup_n_s8(-1), vdup_n_s8(-1));

    ggml_int8x16x4_t q1b;
    ggml_int8x16x4_t q8b;

    uint32_t aux32;
    const uint8_t * aux8 = (const uint8_t *)&aux32;

    float sumf = 0;
    for (int i = 0; i < nb; ++i) {

        const int8_t   * q8 = y[i].qs;
        const uint8_t  * qs = x[i].qs;
        const uint8_t  * qh = x[i].qh;
        const uint16_t * sc = (const uint16_t *)x[i].scales;

        scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);

        int32x4_t sumi1 = mzero;
        int32x4_t sumi2 = mzero;

        for (int ib = 0; ib < QK_K/32; ib += 2) {

            q1b.val[0] = vcombine_s8(vld1_s8((const int8_t *)(iq1s_grid + (qs[0] | ((qh[0] << 8) & 0x700)))),
                                     vld1_s8((const int8_t *)(iq1s_grid + (qs[1] | ((qh[0] << 4) & 0x700)))));
            q1b.val[1] = vcombine_s8(vld1_s8((const int8_t *)(iq1s_grid + (qs[2] | ((qh[1] << 8) & 0x700)))),
                                     vld1_s8((const int8_t *)(iq1s_grid + (qs[3] | ((qh[1] << 4) & 0x700)))));
            q1b.val[2] = vcombine_s8(vld1_s8((const int8_t *)(iq1s_grid + (qs[4] | ((qh[2] << 8) & 0x700)))),
                                     vld1_s8((const int8_t *)(iq1s_grid + (qs[5] | ((qh[2] << 4) & 0x700)))));
            q1b.val[3] = vcombine_s8(vld1_s8((const int8_t *)(iq1s_grid + (qs[6] | ((qh[3] << 8) & 0x700)))),
                                     vld1_s8((const int8_t *)(iq1s_grid + (qs[7] | ((qh[3] << 4) & 0x700)))));

            q8b = ggml_vld1q_s8_x4(q8); q8 += 64;

            const int32x4_t p1 = vpaddq_s32(ggml_vdotq_s32(mzero, q1b.val[0], q8b.val[0]), ggml_vdotq_s32(mzero, q1b.val[1], q8b.val[1]));
            const int32x4_t p2 = vpaddq_s32(ggml_vdotq_s32(mzero, q1b.val[2], q8b.val[2]), ggml_vdotq_s32(mzero, q1b.val[3], q8b.val[3]));
            const int32x4_t p12 = vpaddq_s32(p1, p2);

            const uint32_t * qh32 = (const uint32_t *)qh; // we are 4-byte aligned, so we can do that
            aux32 = ((qh32[0] >> 3) & 0x01010101) | ((qh32[0] >> 6) & 0x02020202);

            const int32x4_t p3 = vpaddq_s32(ggml_vdotq_s32(mzero, deltas.val[aux8[0]], q8b.val[0]), ggml_vdotq_s32(mzero, deltas.val[aux8[1]], q8b.val[1]));
            const int32x4_t p4 = vpaddq_s32(ggml_vdotq_s32(mzero, deltas.val[aux8[2]], q8b.val[2]), ggml_vdotq_s32(mzero, deltas.val[aux8[3]], q8b.val[3]));
            const int32x4_t p34 = vpaddq_s32(p3, p4);

            int32x4_t scales_4 = ggml_vld1q_u32(sc[ib/2] >> 0, sc[ib/2] >> 3, sc[ib/2] >> 6, sc[ib/2] >> 9);

            scales_4 = vaddq_s32(vshlq_n_s32(vandq_s32(scales_4, mask), 1), mone);

            sumi1 = vmlaq_s32(sumi1, scales_4, p12);
            sumi2 = vmlaq_s32(sumi2, scales_4, p34);

            qs += 8; qh += 4;

        }

        sumf += y[i].d * GGML_CPU_FP16_TO_FP32(scale.f16) * (vaddvq_s32(sumi1) + IQ1M_DELTA * vaddvq_s32(sumi2));
    }

    *s = sumf;

#else
    UNUSED(x);
    UNUSED(y);
    UNUSED(nb);
    UNUSED(scale);
    ggml_vec_dot_iq1_m_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#endif
}

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

#if defined __ARM_NEON
    const int8x16_t values = vld1q_s8(kvalues_iq4nl);
    const uint8x16_t m4b = vdupq_n_u8(0x0f);
    uint8x16x2_t q4bits;
    int8x16x4_t q4b;
    int8x16x4_t q8b;
    int32x4_t prod_1, prod_2;

    for (; ib + 1 < nb; ib += 2) {

        q4bits.val[0] = vld1q_u8(x[ib + 0].qs);
        q4bits.val[1] = vld1q_u8(x[ib + 1].qs);
        q8b.val[0]    = vld1q_s8(y[ib + 0].qs);
        q8b.val[1]    = vld1q_s8(y[ib + 0].qs + 16);
        q8b.val[2]    = vld1q_s8(y[ib + 1].qs);
        q8b.val[3]    = vld1q_s8(y[ib + 1].qs + 16);

        q4b.val[0] = ggml_vqtbl1q_s8(values, vandq_u8  (q4bits.val[0], m4b));
        q4b.val[1] = ggml_vqtbl1q_s8(values, vshrq_n_u8(q4bits.val[0], 4));
        q4b.val[2] = ggml_vqtbl1q_s8(values, vandq_u8  (q4bits.val[1], m4b));
        q4b.val[3] = ggml_vqtbl1q_s8(values, vshrq_n_u8(q4bits.val[1], 4));

        prod_1 = ggml_vdotq_s32(ggml_vdotq_s32(vdupq_n_s32(0), q4b.val[0], q8b.val[0]), q4b.val[1], q8b.val[1]);
        prod_2 = ggml_vdotq_s32(ggml_vdotq_s32(vdupq_n_s32(0), q4b.val[2], q8b.val[2]), q4b.val[3], q8b.val[3]);

        sumf +=
            GGML_CPU_FP16_TO_FP32(x[ib+0].d) * GGML_CPU_FP16_TO_FP32(y[ib + 0].d) * vaddvq_s32(prod_1) +
            GGML_CPU_FP16_TO_FP32(x[ib+1].d) * GGML_CPU_FP16_TO_FP32(y[ib + 1].d) * vaddvq_s32(prod_2);
    }

#endif
    for (; ib < nb; ++ib) {
        const float d = GGML_CPU_FP16_TO_FP32(y[ib].d)*GGML_CPU_FP16_TO_FP32(x[ib].d);
        int sumi1 = 0, sumi2 = 0;
        for (int j = 0; j < QK4_NL/2; ++j) {
            sumi1 += y[ib].qs[j+       0] * kvalues_iq4nl[x[ib].qs[j] & 0xf];
            sumi2 += y[ib].qs[j+QK4_NL/2] * kvalues_iq4nl[x[ib].qs[j] >>  4];
        }
        sumf += d * (sumi1 + sumi2);
    }
    *s = sumf;
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

#if defined __ARM_NEON
    const int8x16_t values = vld1q_s8(kvalues_iq4nl);
    const uint8x16_t m4b = vdupq_n_u8(0x0f);
    ggml_uint8x16x2_t q4bits;
    ggml_int8x16x4_t q4b;
    ggml_int8x16x4_t q8b;
    int32x4_t prod_1, prod_2;

    float sumf = 0;

    for (int ibl = 0; ibl < nb; ++ibl) {

        const int8_t  * q8 = y[ibl].qs;
        const uint8_t * q4 = x[ibl].qs;
        uint16_t h = x[ibl].scales_h;

        int sumi1 = 0, sumi2 = 0;
        for (int ib = 0; ib < QK_K/64; ++ib) {

            q4bits = ggml_vld1q_u8_x2(q4); q4 += 32;
            q8b    = ggml_vld1q_s8_x4(q8); q8 += 64;

            q4b.val[0] = ggml_vqtbl1q_s8(values, vandq_u8  (q4bits.val[0], m4b));
            q4b.val[1] = ggml_vqtbl1q_s8(values, vshrq_n_u8(q4bits.val[0], 4));
            q4b.val[2] = ggml_vqtbl1q_s8(values, vandq_u8  (q4bits.val[1], m4b));
            q4b.val[3] = ggml_vqtbl1q_s8(values, vshrq_n_u8(q4bits.val[1], 4));

            prod_1 = ggml_vdotq_s32(ggml_vdotq_s32(vdupq_n_s32(0), q4b.val[0], q8b.val[0]), q4b.val[1], q8b.val[1]);
            prod_2 = ggml_vdotq_s32(ggml_vdotq_s32(vdupq_n_s32(0), q4b.val[2], q8b.val[2]), q4b.val[3], q8b.val[3]);

            int ls1 = ((x[ibl].scales_l[ib] & 0xf) | ((h << 4) & 0x30)) - 32;
            int ls2 = ((x[ibl].scales_l[ib] >>  4) | ((h << 2) & 0x30)) - 32;
            h >>= 4;
            sumi1 += vaddvq_s32(prod_1) * ls1;
            sumi2 += vaddvq_s32(prod_2) * ls2;

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

