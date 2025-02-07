#define GGML_COMMON_DECL_METAL
#define GGML_COMMON_IMPL_METAL
#if defined(GGML_METAL_EMBED_LIBRARY)
__embed_ggml-common.h__
#else
// TODO: this should not be a relative path, but can't figure out how to set Metal include paths in Package.swift
#include "../ggml-common.h"
#endif
#include "ggml-metal-impl.h"

#include <metal_stdlib>

using namespace metal;

#define MAX(x, y) ((x) > (y) ? (x) : (y))
#define MIN(x, y) ((x) < (y) ? (x) : (y))
#define SWAP(x, y) { auto tmp = (x); (x) = (y); (y) = tmp; }

#define N_SIMDWIDTH 32 // assuming SIMD group size is 32

// ref: https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf
//
// cmd:
//   .../usr/bin/metal -dM -E -c                             ggml/src/ggml-metal/ggml-metal.metal
//   .../usr/bin/metal -dM -E -c -target air64-apple-ios14.0 ggml/src/ggml-metal/ggml-metal.metal
//
#if __METAL_VERSION__ < 310 && defined(GGML_METAL_USE_BF16)
#undef GGML_METAL_USE_BF16
#endif

#if defined(GGML_METAL_USE_BF16)
typedef matrix<bfloat, 4, 4> bfloat4x4;
#endif

constexpr constant static float kvalues_iq4nl_f[16] = {
    -127.f, -104.f, -83.f, -65.f, -49.f, -35.f, -22.f, -10.f, 1.f, 13.f, 25.f, 38.f, 53.f, 69.f, 89.f, 113.f
};

// NOTE: this is not dequantizing - we are simply fitting the template
template <typename type4x4>
void dequantize_f32(device const float4x4 * src, short il, thread type4x4 & reg) {
    reg = (type4x4)(*src);
}

template <typename type4x4>
void dequantize_f16(device const half4x4 * src, short il, thread type4x4 & reg) {
    reg = (type4x4)(*src);
}

template <typename type4>
void dequantize_f16_t4(device const half4 * src, short il, thread type4 & reg) {
    reg = (type4)(*(src + il));
}

#if defined(GGML_METAL_USE_BF16)
template <typename type4x4>
void dequantize_bf16(device const bfloat4x4 * src, short il, thread type4x4 & reg) {
    reg = (type4x4)(*src);
}
#endif

template <typename type4x4>
void dequantize_q4_0(device const block_q4_0 * xb, short il, thread type4x4 & reg) {
    device const uint16_t * qs = ((device const uint16_t *)xb + 1);
    const float d1 = il ? (xb->d / 16.h) : xb->d;
    const float d2 = d1 / 256.f;
    const float md = -8.h * xb->d;
    const ushort mask0 = il ? 0x00F0 : 0x000F;
    const ushort mask1 = mask0 << 8;

    float4x4 reg_f;

    for (int i = 0; i < 8; i++) {
        reg_f[i/2][2*(i%2) + 0] = d1 * (qs[i] & mask0) + md;
        reg_f[i/2][2*(i%2) + 1] = d2 * (qs[i] & mask1) + md;
    }

    reg = (type4x4) reg_f;
}

template <typename type4>
void dequantize_q4_0_t4(device const block_q4_0 * xb, short il, thread type4 & reg) {
    device const uint16_t * qs = ((device const uint16_t *)xb + 1);
    const float d1 = (il/4) ? (xb->d / 16.h) : xb->d;
    const float d2 = d1 / 256.f;
    const float md = -8.h * xb->d;
    const ushort mask0 = (il/4) ? 0x00F0 : 0x000F;
    const ushort mask1 = mask0 << 8;

    for (int i = 0; i < 2; i++) {
        reg[2*i + 0] = d1 * (qs[2*(il%4) + i] & mask0) + md;
        reg[2*i + 1] = d2 * (qs[2*(il%4) + i] & mask1) + md;
    }
}

template <typename type4x4>
void dequantize_q4_1(device const block_q4_1 * xb, short il, thread type4x4 & reg) {
    device const uint16_t * qs = ((device const uint16_t *)xb + 2);
    const float d1 = il ? (xb->d / 16.h) : xb->d;
    const float d2 = d1 / 256.f;
    const float  m = xb->m;
    const ushort mask0 = il ? 0x00F0 : 0x000F;
    const ushort mask1 = mask0 << 8;

    float4x4 reg_f;

    for (int i = 0; i < 8; i++) {
        reg_f[i/2][2*(i%2) + 0] = ((qs[i] & mask0) * d1) + m;
        reg_f[i/2][2*(i%2) + 1] = ((qs[i] & mask1) * d2) + m;
    }

    reg = (type4x4) reg_f;
}

template <typename type4>
void dequantize_q4_1_t4(device const block_q4_1 * xb, short il, thread type4 & reg) {
    device const uint16_t * qs = ((device const uint16_t *)xb + 2);
    const float d1 = (il/4) ? (xb->d / 16.h) : xb->d;
    const float d2 = d1 / 256.f;
    const float  m = xb->m;
    const ushort mask0 = (il/4) ? 0x00F0 : 0x000F;
    const ushort mask1 = mask0 << 8;

    for (int i = 0; i < 2; i++) {
        reg[2*i + 0] = d1 * (qs[2*(il%4) + i] & mask0) + m;
        reg[2*i + 1] = d2 * (qs[2*(il%4) + i] & mask1) + m;
    }
}

template <typename type4x4>
void dequantize_q5_0(device const block_q5_0 * xb, short il, thread type4x4 & reg) {
    device const uint16_t * qs = ((device const uint16_t *)xb + 3);
    const float d = xb->d;
    const float md = -16.h * xb->d;
    const ushort mask = il ? 0x00F0 : 0x000F;

    const uint32_t qh = *((device const uint32_t *)xb->qh);

    const int x_mv = il ? 4 : 0;

    const int gh_mv = il ? 12 : 0;
    const int gh_bk = il ?  0 : 4;

    float4x4 reg_f;

    for (int i = 0; i < 8; i++) {
        // extract the 5-th bits for x0 and x1
        const uint8_t xh_0 = ((qh >> (gh_mv + 2*i  )) << gh_bk) & 0x10;
        const uint8_t xh_1 = ((qh >> (gh_mv + 2*i+1)) << gh_bk) & 0x10;

        // combine the 4-bits from qs with the 5th bit
        const int32_t x0 = ((((qs[i]     ) & mask) >> x_mv) | xh_0);
        const int32_t x1 = ((((qs[i] >> 8) & mask) >> x_mv) | xh_1);

        reg_f[i/2][2*(i%2) + 0] = d * x0 + md;
        reg_f[i/2][2*(i%2) + 1] = d * x1 + md;
    }

    reg = (type4x4) reg_f;
}

template <typename type4>
void dequantize_q5_0_t4(device const block_q5_0 * xb, short il, thread type4 & reg) {
    device const uint16_t * qs = ((device const uint16_t *)xb + 3);
    const float d = xb->d;
    const float md = -16.h * xb->d;
    const ushort mask = (il/4) ? 0x00F0 : 0x000F;

    const uint32_t qh = *((device const uint32_t *)xb->qh);

    const int x_mv = (il/4) ? 4 : 0;

    const int gh_mv = (il/4) ? 12 : 0;
    const int gh_bk = (il/4) ?  0 : 4;

    for (int ii = 0; ii < 2; ii++) {
        int i = 2*(il%4) + ii;

        // extract the 5-th bits for x0 and x1
        const uint8_t xh_0 = ((qh >> (gh_mv + 2*i  )) << gh_bk) & 0x10;
        const uint8_t xh_1 = ((qh >> (gh_mv + 2*i+1)) << gh_bk) & 0x10;

        // combine the 4-bits from qs with the 5th bit
        const int32_t x0 = ((((qs[i]     ) & mask) >> x_mv) | xh_0);
        const int32_t x1 = ((((qs[i] >> 8) & mask) >> x_mv) | xh_1);

        reg[2*ii + 0] = d * x0 + md;
        reg[2*ii + 1] = d * x1 + md;
    }
}

template <typename type4x4>
void dequantize_q5_1(device const block_q5_1 * xb, short il, thread type4x4 & reg) {
    device const uint16_t * qs = ((device const uint16_t *)xb + 4);
    const float d = xb->d;
    const float m = xb->m;
    const ushort mask = il ? 0x00F0 : 0x000F;

    const uint32_t qh = *((device const uint32_t *)xb->qh);

    const int x_mv = il ? 4 : 0;

    const int gh_mv = il ? 12 : 0;
    const int gh_bk = il ?  0 : 4;

    float4x4 reg_f;

    for (int i = 0; i < 8; i++) {
        // extract the 5-th bits for x0 and x1
        const uint8_t xh_0 = ((qh >> (gh_mv + 2*i  )) << gh_bk) & 0x10;
        const uint8_t xh_1 = ((qh >> (gh_mv + 2*i+1)) << gh_bk) & 0x10;

        // combine the 4-bits from qs with the 5th bit
        const int32_t x0 = ((((qs[i]     ) & mask) >> x_mv) | xh_0);
        const int32_t x1 = ((((qs[i] >> 8) & mask) >> x_mv) | xh_1);

        reg_f[i/2][2*(i%2) + 0] = d * x0 + m;
        reg_f[i/2][2*(i%2) + 1] = d * x1 + m;
    }

    reg = (type4x4) reg_f;
}

template <typename type4>
void dequantize_q5_1_t4(device const block_q5_1 * xb, short il, thread type4 & reg) {
    device const uint16_t * qs = ((device const uint16_t *)xb + 4);
    const float d = xb->d;
    const float m = xb->m;
    const ushort mask = (il/4) ? 0x00F0 : 0x000F;

    const uint32_t qh = *((device const uint32_t *)xb->qh);

    const int x_mv = (il/4) ? 4 : 0;

    const int gh_mv = (il/4) ? 12 : 0;
    const int gh_bk = (il/4) ?  0 : 4;

    for (int ii = 0; ii < 2; ii++) {
        int i = 2*(il%4) + ii;

        // extract the 5-th bits for x0 and x1
        const uint8_t xh_0 = ((qh >> (gh_mv + 2*i  )) << gh_bk) & 0x10;
        const uint8_t xh_1 = ((qh >> (gh_mv + 2*i+1)) << gh_bk) & 0x10;

        // combine the 4-bits from qs with the 5th bit
        const int32_t x0 = ((((qs[i]     ) & mask) >> x_mv) | xh_0);
        const int32_t x1 = ((((qs[i] >> 8) & mask) >> x_mv) | xh_1);

        reg[2*ii + 0] = d * x0 + m;
        reg[2*ii + 1] = d * x1 + m;
    }
}

template <typename type4x4>
void dequantize_q8_0(device const block_q8_0 *xb, short il, thread type4x4 & reg) {
    device const int8_t * qs = ((device const int8_t *)xb->qs);
    const float d = xb->d;

    float4x4 reg_f;

    for (int i = 0; i < 16; i++) {
        reg_f[i/4][i%4] = (qs[i + 16*il] * d);
    }

    reg = (type4x4) reg_f;
}

template <typename type4>
void dequantize_q8_0_t4(device const block_q8_0 *xb, short il, thread type4 & reg) {
    device const int8_t * qs = ((device const int8_t *)xb->qs);
    const float d = xb->d;

    for (int i = 0; i < 4; i++) {
        reg[i] = (qs[4*(il%4) + i + 16*(il/4)] * d);
    }
}

template <typename type4x4>
void dequantize_q2_K(device const block_q2_K *xb, short il, thread type4x4 & reg) {
    const float d = xb->d;
    const float min = xb->dmin;
    device const uint8_t * q = (device const uint8_t *)xb->qs;
    float dl, ml;
    uint8_t sc = xb->scales[il];

    q = q + 32*(il/8) + 16*(il&1);
    il = (il/2)%4;

    half  coef = il>1 ? (il>2 ? 1/64.h : 1/16.h) : (il>0 ? 1/4.h : 1.h);
    uchar mask = il>1 ? (il>2 ? 192    : 48)     : (il>0 ? 12    : 3);
    dl = d * (sc & 0xF) * coef, ml = min * (sc >> 4);
    for (int i = 0; i < 16; ++i) {
        reg[i/4][i%4] = dl * (q[i] & mask) - ml;
    }
}

template <typename type4x4>
void dequantize_q3_K(device const block_q3_K *xb, short il, thread type4x4 & reg) {
    const half d_all = xb->d;
    device const uint8_t * q = (device const uint8_t *)xb->qs;
    device const uint8_t * h = (device const uint8_t *)xb->hmask;
    device const int8_t * scales = (device const int8_t *)xb->scales;

    q = q + 32 * (il/8) + 16 * (il&1);
    h = h + 16 * (il&1);
    uint8_t m = 1 << (il/2);
    uint16_t kmask1 = (il/4)>1 ? ((il/4)>2 ? 192 : 48) : \
                                 ((il/4)>0 ? 12  : 3);
    uint16_t kmask2 = il/8 ? 0xF0 : 0x0F;
    uint16_t scale_2 = scales[il%8], scale_1 = scales[8 + il%4];
    int16_t  dl_int = (il/4)&1 ? (scale_2&kmask2) | ((scale_1&kmask1) << 2)
                               : (scale_2&kmask2) | ((scale_1&kmask1) << 4);
    float dl = il<8 ? d_all * (dl_int - 32.f) : d_all * (dl_int / 16.f - 32.f);
    const float ml = 4.f * dl;

    il = (il/2) & 3;
    const half    coef = il>1 ? (il>2 ? 1/64.h : 1/16.h) : (il>0 ? 1/4.h : 1.h);
    const uint8_t mask = il>1 ? (il>2 ? 192    : 48)     : (il>0 ? 12    : 3);
    dl *= coef;

    for (int i = 0; i < 16; ++i) {
        reg[i/4][i%4] = dl * (q[i] & mask) - (h[i] & m ? 0 : ml);
    }
}

static inline uchar2 get_scale_min_k4_just2(int j, int k, device const uchar * q) {
    return j < 4 ? uchar2{uchar(q[j+0+k] & 63), uchar(q[j+4+k] & 63)}
                 : uchar2{uchar((q[j+4+k] & 0xF) | ((q[j-4+k] & 0xc0) >> 2)), uchar((q[j+4+k] >> 4) | ((q[j-0+k] & 0xc0) >> 2))};
}

template <typename type4x4>
void dequantize_q4_K(device const block_q4_K * xb, short il, thread type4x4 & reg) {
    device const uchar * q = xb->qs;

    short is = (il/4) * 2;
    q = q + (il/4) * 32 + 16 * (il&1);
    il = il & 3;
    const uchar2 sc = get_scale_min_k4_just2(is, il/2, xb->scales);
    const float d   = il < 2 ? xb->d : xb->d / 16.h;
    const float min = xb->dmin;
    const float dl = d * sc[0];
    const float ml = min * sc[1];

    const ushort mask = il < 2 ? 0x0F : 0xF0;
    for (int i = 0; i < 16; ++i) {
        reg[i/4][i%4] = dl * (q[i] & mask) - ml;
    }
}

template <typename type4x4>
void dequantize_q5_K(device const block_q5_K *xb, short il, thread type4x4 & reg) {
    device const uint8_t * q  = xb->qs;
    device const uint8_t * qh = xb->qh;

    short is = (il/4) * 2;
    q  = q + 32 * (il/4) + 16 * (il&1);
    qh = qh + 16 * (il&1);
    uint8_t ul = 1 << (il/2);
    il = il & 3;
    const uchar2 sc = get_scale_min_k4_just2(is, il/2, xb->scales);
    const float d = il < 2 ? xb->d : xb->d / 16.f;
    const float min = xb->dmin;
    const float dl = d * sc[0];
    const float ml = min * sc[1];

    const ushort mask  = il<2 ? 0x0F : 0xF0;
    const float qh_val = il<2 ? 16.f : 256.f;
    for (int i = 0; i < 16; ++i) {
        reg[i/4][i%4] = dl * ((q[i] & mask) + (qh[i] & ul ? qh_val : 0)) - ml;
    }
}

template <typename type4x4>
void dequantize_q6_K(device const block_q6_K *xb, short il, thread type4x4 & reg) {
    const half d_all = xb->d;
    device const uint8_t * ql = (device const uint8_t *)xb->ql;
    device const uint8_t * qh = (device const uint8_t *)xb->qh;
    device const int8_t * scales = (device const int8_t *)xb->scales;

    ql = ql + 64*(il/8) + 32*((il/2)&1) + 16*(il&1);
    qh = qh + 32*(il/8) + 16*(il&1);
    float sc = scales[(il%2) + 2 * ((il/2))];
    il = (il/2) & 3;

    const uint16_t  kmask1 = il>1 ? (il>2 ? 192 : 48) : (il>0 ? 12 : 3);
    const uint16_t  kmask2 = il>1 ? 0xF0              : 0x0F;
    const float       coef = il>1 ? 1.f/16.f          : 1.f;
    const float ml = d_all * sc * 32.f;
    const float dl = d_all * sc * coef;
    for (int i = 0; i < 16; ++i) {
        const half q = il&1 ? ((ql[i] & kmask2) | ((qh[i] & kmask1) << 2))
                            : ((ql[i] & kmask2) | ((qh[i] & kmask1) << 4));
        reg[i/4][i%4] = dl * q - ml;
    }
}

template <typename type4x4>
void dequantize_iq2_xxs(device const block_iq2_xxs * xb, short il, thread type4x4 & reg) {
    // il is 0...15 for QK_K = 256 => index of block of 32 is il/2
    const float d = xb->d;
    const int ib32 = il/2;
    il = il%2;
    // il = 0 or 1. il = 0 processes the first 16 quants in a block of 32, il = 1 the second 16
    // each block of 32 needs 2 uint32_t's for the quants & scale, so 4 uint16_t's.
    device const uint16_t * q2 = xb->qs + 4*ib32;
    const uint32_t aux32_g = q2[0] | (q2[1] << 16);
    const uint32_t aux32_s = q2[2] | (q2[3] << 16);
    thread const uint8_t * aux8 = (thread const uint8_t *)&aux32_g;
    const float dl = d * (0.5f + (aux32_s >> 28)) * 0.25f;
    constant uint8_t * grid = (constant uint8_t *)(iq2xxs_grid + aux8[2*il+0]);
    uint8_t signs = ksigns_iq2xs[(aux32_s >> 14*il) & 127];
    for (int i = 0; i < 8; ++i) {
        reg[i/4][i%4] = dl * grid[i] * (signs & kmask_iq2xs[i] ? -1.f : 1.f);
    }
    grid = (constant uint8_t *)(iq2xxs_grid + aux8[2*il+1]);
    signs = ksigns_iq2xs[(aux32_s >> (14*il+7)) & 127];
    for (int i = 0; i < 8; ++i) {
        reg[2+i/4][i%4] = dl * grid[i] * (signs & kmask_iq2xs[i] ? -1.f : 1.f);
    }
}

template <typename type4x4>
void dequantize_iq2_xs(device const block_iq2_xs * xb, short il, thread type4x4 & reg) {
    // il is 0...15 for QK_K = 256 => index of block of 32 is il/2
    const float d = xb->d;
    const int ib32 = il/2;
    il = il%2;
    // il = 0 or 1. il = 0 processes the first 16 quants in a block of 32, il = 1 the second 16
    device const uint16_t * q2 = xb->qs + 4*ib32;
    const float dl = d * (0.5f + ((xb->scales[ib32] >> 4*il) & 0xf)) * 0.25f;
    constant uint8_t * grid = (constant uint8_t *)(iq2xs_grid + (q2[2*il+0] & 511));
    uint8_t signs = ksigns_iq2xs[q2[2*il+0] >> 9];
    for (int i = 0; i < 8; ++i) {
        reg[i/4][i%4] = dl * grid[i] * (signs & kmask_iq2xs[i] ? -1.f : 1.f);
    }
    grid = (constant uint8_t *)(iq2xs_grid + (q2[2*il+1] & 511));
    signs = ksigns_iq2xs[q2[2*il+1] >> 9];
    for (int i = 0; i < 8; ++i) {
        reg[2+i/4][i%4] = dl * grid[i] * (signs & kmask_iq2xs[i] ? -1.f : 1.f);
    }
}

template <typename type4x4>
void dequantize_iq3_xxs(device const block_iq3_xxs * xb, short il, thread type4x4 & reg) {
    // il is 0...15 for QK_K = 256 => index of block of 32 is il/2
    const float d = xb->d;
    const int ib32 = il/2;
    il = il%2;
    // il = 0 or 1. il = 0 processes the first 16 quants in a block of 32, il = 1 the second 16
    device const uint8_t * q3 = xb->qs + 8*ib32;
    device const uint16_t * gas = (device const uint16_t *)(xb->qs + QK_K/4) + 2*ib32;
    const uint32_t aux32 = gas[0] | (gas[1] << 16);
    const float dl = d * (0.5f + (aux32 >> 28)) * 0.5f;
    constant uint8_t * grid1 = (constant uint8_t *)(iq3xxs_grid + q3[4*il+0]);
    constant uint8_t * grid2 = (constant uint8_t *)(iq3xxs_grid + q3[4*il+1]);
    uint8_t signs = ksigns_iq2xs[(aux32 >> 14*il) & 127];
    for (int i = 0; i < 4; ++i) {
        reg[0][i] = dl * grid1[i] * (signs & kmask_iq2xs[i+0] ? -1.f : 1.f);
        reg[1][i] = dl * grid2[i] * (signs & kmask_iq2xs[i+4] ? -1.f : 1.f);
    }
    grid1 = (constant uint8_t *)(iq3xxs_grid + q3[4*il+2]);
    grid2 = (constant uint8_t *)(iq3xxs_grid + q3[4*il+3]);
    signs = ksigns_iq2xs[(aux32 >> (14*il+7)) & 127];
    for (int i = 0; i < 4; ++i) {
        reg[2][i] = dl * grid1[i] * (signs & kmask_iq2xs[i+0] ? -1.f : 1.f);
        reg[3][i] = dl * grid2[i] * (signs & kmask_iq2xs[i+4] ? -1.f : 1.f);
    }
}

template <typename type4x4>
void dequantize_iq3_s(device const block_iq3_s * xb, short il, thread type4x4 & reg) {
    // il is 0...15 for QK_K = 256 => index of block of 32 is il/2
    const float d = xb->d;
    const int ib32 = il/2;
    il = il%2;
    // il = 0 or 1. il = 0 processes the first 16 quants in a block of 32, il = 1 the second 16
    device const uint8_t * qs = xb->qs + 8*ib32;
    device const uint8_t * signs = xb->signs + 4*ib32 + 2*il;
    const uint8_t qh = xb->qh[ib32] >> 4*il;
    const float dl = d * (1 + 2*((xb->scales[ib32/2] >> 4*(ib32%2)) & 0xf));
    constant uint8_t * grid1 = (constant uint8_t *)(iq3s_grid + (qs[4*il+0] | ((qh << 8) & 256)));
    constant uint8_t * grid2 = (constant uint8_t *)(iq3s_grid + (qs[4*il+1] | ((qh << 7) & 256)));
    for (int i = 0; i < 4; ++i) {
        reg[0][i] = dl * grid1[i] * select(1, -1, signs[0] & kmask_iq2xs[i+0]);
        reg[1][i] = dl * grid2[i] * select(1, -1, signs[0] & kmask_iq2xs[i+4]);
    }
    grid1 = (constant uint8_t *)(iq3s_grid + (qs[4*il+2] | ((qh << 6) & 256)));
    grid2 = (constant uint8_t *)(iq3s_grid + (qs[4*il+3] | ((qh << 5) & 256)));
    for (int i = 0; i < 4; ++i) {
        reg[2][i] = dl * grid1[i] * select(1, -1, signs[1] & kmask_iq2xs[i+0]);
        reg[3][i] = dl * grid2[i] * select(1, -1, signs[1] & kmask_iq2xs[i+4]);
    }
}

template <typename type4x4>
void dequantize_iq2_s(device const block_iq2_s * xb, short il, thread type4x4 & reg) {
    // il is 0...15 for QK_K = 256 => index of block of 32 is il/2
    const float d = xb->d;
    const int ib32 = il/2;
    il = il%2;
    // il = 0 or 1. il = 0 processes the first 16 quants in a block of 32, il = 1 the second 16
    device const uint8_t * qs = xb->qs + 4*ib32 + 2*il;
    device const uint8_t * signs = qs + QK_K/8;
    const uint8_t qh = xb->qh[ib32] >> 4*il;
    const float dl = d * (0.5f + ((xb->scales[ib32] >> 4*il) & 0xf)) * 0.25f;
    constant uint8_t * grid1 = (constant uint8_t *)(iq2s_grid + (qs[0] | ((qh << 8) & 0x300)));
    constant uint8_t * grid2 = (constant uint8_t *)(iq2s_grid + (qs[1] | ((qh << 6) & 0x300)));
    for (int i = 0; i < 8; ++i) {
        reg[i/4+0][i%4] = dl * grid1[i] * select(1, -1, signs[0] & kmask_iq2xs[i]);
        reg[i/4+2][i%4] = dl * grid2[i] * select(1, -1, signs[1] & kmask_iq2xs[i]);
    }
}

template <typename type4x4>
void dequantize_iq1_s(device const block_iq1_s * xb, short il, thread type4x4 & reg) {
    // il is 0...15 for QK_K = 256 => index of block of 32 is il/2
    const int ib32 = il/2;
    il = il%2;
    const float d = xb->d;
    device const uint8_t  * qs = xb->qs + 4*ib32 + 2*il;
    device const uint16_t * qh = xb->qh;
    const float dl = d * (2*((qh[ib32] >> 12) & 7) + 1);
    const float ml = dl * (qh[ib32] & 0x8000 ? -1 - IQ1S_DELTA : -1 + IQ1S_DELTA);
    const uint16_t h = qh[ib32] >> 6*il;
    constant uint8_t * grid1 = (constant uint8_t *)(iq1s_grid_gpu + (qs[0] | ((h << 8) & 0x700)));
    constant uint8_t * grid2 = (constant uint8_t *)(iq1s_grid_gpu + (qs[1] | ((h << 5) & 0x700)));
    for (int i = 0; i < 4; ++i) {
        reg[0][i] = dl * (grid1[i] & 0xf) + ml;
        reg[1][i] = dl * (grid1[i] >>  4) + ml;
        reg[2][i] = dl * (grid2[i] & 0xf) + ml;
        reg[3][i] = dl * (grid2[i] >>  4) + ml;
    }
}

template <typename type4x4>
void dequantize_iq1_m(device const block_iq1_m * xb, short il, thread type4x4 & reg) {
    // il is 0...15 for QK_K = 256 => index of block of 32 is il/2
    const int ib32 = il/2;
    il = il%2;
    device const uint16_t * sc = (device const uint16_t *)xb->scales;

    iq1m_scale_t scale;
    scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
    const float d = scale.f16;

    device const uint8_t * qs = xb->qs + 4*ib32 + 2*il;
    device const uint8_t * qh = xb->qh + 2*ib32 + il;

    const float dl  = d * (2*((sc[ib32/2] >> (6*(ib32%2)+3*il)) & 7) + 1);
    const float ml1 = dl * (qh[0] & 0x08 ? -1 - IQ1M_DELTA : -1 + IQ1M_DELTA);
    const float ml2 = dl * (qh[0] & 0x80 ? -1 - IQ1M_DELTA : -1 + IQ1M_DELTA);
    constant uint8_t * grid1 = (constant uint8_t *)(iq1s_grid_gpu + (qs[0] | ((qh[0] << 8) & 0x700)));
    constant uint8_t * grid2 = (constant uint8_t *)(iq1s_grid_gpu + (qs[1] | ((qh[0] << 4) & 0x700)));
    for (int i = 0; i < 4; ++i) {
        reg[0][i] = dl * (grid1[i] & 0xf) + ml1;
        reg[1][i] = dl * (grid1[i] >>  4) + ml1;
        reg[2][i] = dl * (grid2[i] & 0xf) + ml2;
        reg[3][i] = dl * (grid2[i] >>  4) + ml2;
    }
}

template <typename type4x4>
void dequantize_iq4_nl(device const block_iq4_nl * xb, short il, thread type4x4 & reg) {
    device const uint16_t * q4 = (device const uint16_t *)xb->qs;
    const float d = xb->d;
    uint32_t aux32;
    thread const uint8_t * q8 = (thread const uint8_t *)&aux32;
    for (int i = 0; i < 4; ++i) {
        aux32 = ((q4[2*i] | (q4[2*i+1] << 16)) >> 4*il) & 0x0f0f0f0f;
        reg[i][0] = d * kvalues_iq4nl_f[q8[0]];
        reg[i][1] = d * kvalues_iq4nl_f[q8[1]];
        reg[i][2] = d * kvalues_iq4nl_f[q8[2]];
        reg[i][3] = d * kvalues_iq4nl_f[q8[3]];
    }
}

template <typename type4>
void dequantize_iq4_nl_t4(device const block_iq4_nl * xb, short il, thread type4 & reg) {
    device const uint16_t * q4 = (device const uint16_t *)xb->qs;
    const float d = xb->d;
    uint32_t aux32;
    thread const uint8_t * q8 = (thread const uint8_t *)&aux32;
    aux32 = ((q4[2*(il%4)] | (q4[2*(il%4)+1] << 16)) >> 4*(il/4)) & 0x0f0f0f0f;
    reg[0] = d * kvalues_iq4nl_f[q8[0]];
    reg[1] = d * kvalues_iq4nl_f[q8[1]];
    reg[2] = d * kvalues_iq4nl_f[q8[2]];
    reg[3] = d * kvalues_iq4nl_f[q8[3]];
}

template <typename type4x4>
void dequantize_iq4_xs(device const block_iq4_xs * xb, short il, thread type4x4 & reg) {
    // il is 0...15 for QK_K = 256 => index of block of 32 is il/2
    const int ib32 = il/2;
    il = il%2;
    // il = 0 or 1. il = 0 processes the first 16 quants in a block of 32, il = 1 the second 16
    device const uint32_t * q4 = (device const uint32_t *)xb->qs + 4*ib32;
    const int ls = ((xb->scales_l[ib32/2] >> 4*(ib32%2)) & 0xf) | (((xb->scales_h >> 2*ib32) & 3) << 4);
    const float d = (float)xb->d * (ls - 32);
    uint32_t aux32;
    thread const uint8_t * q8 = (thread const uint8_t *)&aux32;
    for (int i = 0; i < 4; ++i) {
        aux32 = (q4[i] >> 4*il) & 0x0f0f0f0f;
        reg[i][0] = d * kvalues_iq4nl_f[q8[0]];
        reg[i][1] = d * kvalues_iq4nl_f[q8[1]];
        reg[i][2] = d * kvalues_iq4nl_f[q8[2]];
        reg[i][3] = d * kvalues_iq4nl_f[q8[3]];
    }
}

enum ggml_sort_order {
    GGML_SORT_ORDER_ASC,
    GGML_SORT_ORDER_DESC,
};

// general-purpose kernel for addition, subtraction, multiplication and division of two tensors
// pros: works for non-contiguous tensors, supports broadcast across all dims
// cons: not very efficient
kernel void kernel_add(
        constant ggml_metal_kargs_bin & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    const int i03 = tgpig.z;
    const int i02 = tgpig.y;
    const int i01 = tgpig.x;

    const int i13 = i03%args.ne13;
    const int i12 = i02%args.ne12;
    const int i11 = i01%args.ne11;

    device const char * src0_ptr = src0 + i03*args.nb03 + i02*args.nb02 + i01*args.nb01 + args.offs;
    device const char * src1_ptr = src1 + i13*args.nb13 + i12*args.nb12 + i11*args.nb11;
    device       char * dst_ptr  = dst  + i03*args.nb3  + i02*args.nb2  + i01*args.nb1  + args.offs;

    for (int i0 = tpitg.x; i0 < args.ne0; i0 += ntg.x) {
        const int i10 = i0%args.ne10;
        *((device float *)(dst_ptr + i0*args.nb0)) = *((device float *)(src0_ptr + i0*args.nb00)) + *((device float *)(src1_ptr + i10*args.nb10));
    }
}

kernel void kernel_sub(
        constant ggml_metal_kargs_bin & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    const int i03 = tgpig.z;
    const int i02 = tgpig.y;
    const int i01 = tgpig.x;

    const int i13 = i03%args.ne13;
    const int i12 = i02%args.ne12;
    const int i11 = i01%args.ne11;

    device const char * src0_ptr = src0 + i03*args.nb03 + i02*args.nb02 + i01*args.nb01 + args.offs;
    device const char * src1_ptr = src1 + i13*args.nb13 + i12*args.nb12 + i11*args.nb11;
    device       char * dst_ptr  = dst  + i03*args.nb3  + i02*args.nb2  + i01*args.nb1  + args.offs;

    for (int i0 = tpitg.x; i0 < args.ne0; i0 += ntg.x) {
        const int i10 = i0%args.ne10;
        *((device float *)(dst_ptr + i0*args.nb0)) = *((device float *)(src0_ptr + i0*args.nb00)) - *((device float *)(src1_ptr + i10*args.nb10));
    }
}

kernel void kernel_mul(
        constant ggml_metal_kargs_bin & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    const int i03 = tgpig.z;
    const int i02 = tgpig.y;
    const int i01 = tgpig.x;

    const int i13 = i03%args.ne13;
    const int i12 = i02%args.ne12;
    const int i11 = i01%args.ne11;

    device const char * src0_ptr = src0 + i03*args.nb03 + i02*args.nb02 + i01*args.nb01;
    device const char * src1_ptr = src1 + i13*args.nb13 + i12*args.nb12 + i11*args.nb11;
    device       char * dst_ptr  = dst  + i03*args.nb3  + i02*args.nb2  + i01*args.nb1;

    for (int i0 = tpitg.x; i0 < args.ne0; i0 += ntg.x) {
        const int i10 = i0%args.ne10;
        *((device float *)(dst_ptr + i0*args.nb0)) = *((device float *)(src0_ptr + i0*args.nb00)) * *((device float *)(src1_ptr + i10*args.nb10));
    }
}

kernel void kernel_div(
        constant ggml_metal_kargs_bin & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    const int i03 = tgpig.z;
    const int i02 = tgpig.y;
    const int i01 = tgpig.x;

    const int i13 = i03%args.ne13;
    const int i12 = i02%args.ne12;
    const int i11 = i01%args.ne11;

    device const char * src0_ptr = src0 + i03*args.nb03 + i02*args.nb02 + i01*args.nb01;
    device const char * src1_ptr = src1 + i13*args.nb13 + i12*args.nb12 + i11*args.nb11;
    device       char * dst_ptr  = dst  + i03*args.nb3  + i02*args.nb2  + i01*args.nb1;

    for (int i0 = tpitg.x; i0 < args.ne0; i0 += ntg.x) {
        const int i10 = i0%args.ne10;
        *((device float *)(dst_ptr + i0*args.nb0)) = *((device float *)(src0_ptr + i0*args.nb00)) / *((device float *)(src1_ptr + i10*args.nb10));
    }
}

template<typename T>
kernel void kernel_repeat(
        constant ggml_metal_kargs_repeat & args,
        device const char * src0,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    const int i3 = tgpig.z;
    const int i2 = tgpig.y;
    const int i1 = tgpig.x;

    const int i03 = i3%args.ne03;
    const int i02 = i2%args.ne02;
    const int i01 = i1%args.ne01;

    device const char * src0_ptr = src0 + i03*args.nb03 + i02*args.nb02 + i01*args.nb01;
    device       char * dst_ptr  = dst  +  i3*args.nb3  +  i2*args.nb2  +  i1*args.nb1;

    for (int i0 = tpitg.x; i0 < args.ne0; i0 += ntg.x) {
        const int i00 = i0%args.ne00;
        *((device T *)(dst_ptr + i0*args.nb0)) = *((device T *)(src0_ptr + i00*args.nb00));
    }
}

typedef decltype(kernel_repeat<float>) kernel_repeat_t;

template [[host_name("kernel_repeat_f32")]] kernel kernel_repeat_t kernel_repeat<float>;
template [[host_name("kernel_repeat_f16")]] kernel kernel_repeat_t kernel_repeat<half>;
template [[host_name("kernel_repeat_i32")]] kernel kernel_repeat_t kernel_repeat<int>;
template [[host_name("kernel_repeat_i16")]] kernel kernel_repeat_t kernel_repeat<short>;

// assumption: src1 is a row
// broadcast src1 into src0
kernel void kernel_add_row(
        constant ggml_metal_kargs_bin & args,
        device const float4 * src0,
        device const float4 * src1,
        device       float4 * dst,
        uint tpig[[thread_position_in_grid]]) {
    const uint nb = args.ne00/4;
    dst[tpig] = src0[tpig] + src1[tpig % nb];
}

kernel void kernel_sub_row(
        constant ggml_metal_kargs_bin & args,
        device const float4 * src0,
        device const float4 * src1,
        device       float4 * dst,
        uint tpig[[thread_position_in_grid]]) {
    const uint nb = args.ne00/4;
    dst[tpig] = src0[tpig] - src1[tpig % nb];
}

kernel void kernel_mul_row(
        constant ggml_metal_kargs_bin & args,
        device const float4 * src0,
        device const float4 * src1,
        device       float4 * dst,
        uint tpig[[thread_position_in_grid]]) {
    const uint nb = args.ne00/4;
    dst[tpig] = src0[tpig] * src1[tpig % nb];
}

kernel void kernel_div_row(
        constant ggml_metal_kargs_bin & args,
        device const float4 * src0,
        device const float4 * src1,
        device       float4 * dst,
        uint tpig[[thread_position_in_grid]]) {
    const uint nb = args.ne00/4;
    dst[tpig] = src0[tpig] / src1[tpig % nb];
}

kernel void kernel_scale(
        device const float * src0,
        device       float * dst,
        constant     float & scale,
        uint tpig[[thread_position_in_grid]]) {
    dst[tpig] = src0[tpig] * scale;
}

kernel void kernel_scale_4(
        device const float4 * src0,
        device       float4 * dst,
        constant     float  & scale,
        uint tpig[[thread_position_in_grid]]) {
    dst[tpig] = src0[tpig] * scale;
}

kernel void kernel_clamp(
        device const float * src0,
        device       float * dst,
        constant     float & min,
        constant     float & max,
        uint tpig[[thread_position_in_grid]]) {
    dst[tpig] = src0[tpig] < min ? min : (src0[tpig] > max ? max : src0[tpig]);
}

kernel void kernel_relu(
        device const float * src0,
        device       float * dst,
        uint tpig[[thread_position_in_grid]]) {
    dst[tpig] = max(0.0f, src0[tpig]);
}

kernel void kernel_sigmoid(
        device const float * src0,
        device       float * dst,
        uint tpig[[thread_position_in_grid]]) {
    dst[tpig] = 1.0f / (1.0f + exp(-src0[tpig]));
}

kernel void kernel_tanh(
        device const float * src0,
        device       float * dst,
        uint tpig[[thread_position_in_grid]]) {
    device const float & x = src0[tpig];
    dst[tpig] = precise::tanh(x);
}

constant float GELU_COEF_A     = 0.044715f;
constant float GELU_QUICK_COEF = -1.702f;
constant float SQRT_2_OVER_PI  = 0.79788456080286535587989211986876f;

kernel void kernel_gelu(
    device const float * src0,
    device       float * dst,
    uint tpig[[thread_position_in_grid]]) {
    device const float & x = src0[tpig];

    dst[tpig] = 0.5f*x*(1.0f + precise::tanh(SQRT_2_OVER_PI*x*(1.0f + GELU_COEF_A*x*x)));
}

kernel void kernel_gelu_4(
    device const float4 * src0,
    device       float4 * dst,
    uint tpig[[thread_position_in_grid]]) {
    device const float4 & x = src0[tpig];

    // BEWARE !!!
    // Simply using "tanh" instead of "precise::tanh" will sometimes results in NaNs!
    // This was observed with Falcon 7B and 40B models
    //
    dst[tpig] = 0.5f*x*(1.0f + precise::tanh(SQRT_2_OVER_PI*x*(1.0f + GELU_COEF_A*x*x)));
}

kernel void kernel_gelu_quick(
    device const float * src0,
    device       float * dst,
    uint tpig[[thread_position_in_grid]]) {
    device const float & x = src0[tpig];

    dst[tpig] = x*(1.0f/(1.0f+exp(GELU_QUICK_COEF*x)));
}

kernel void kernel_gelu_quick_4(
    device const float4 * src0,
    device       float4 * dst,
    uint tpig[[thread_position_in_grid]]) {
    device const float4 & x = src0[tpig];

    dst[tpig] = x*(1.0f/(1.0f+exp(GELU_QUICK_COEF*x)));
}

kernel void kernel_silu(
        device const float * src0,
        device       float * dst,
        uint tpig[[thread_position_in_grid]]) {
    device const float & x = src0[tpig];
    dst[tpig] = x / (1.0f + exp(-x));
}

kernel void kernel_silu_4(
        device const float4 * src0,
        device       float4 * dst,
        uint tpig[[thread_position_in_grid]]) {
    device const float4 & x = src0[tpig];
    dst[tpig] = x / (1.0f + exp(-x));
}

kernel void kernel_elu(
        device const float * src0,
        device       float * dst,
        uint tpig[[thread_position_in_grid]]) {
    device const float & x = src0[tpig];
    dst[tpig] = (x > 0.0f) ? x : (exp(x) - 1.0f);
}

kernel void kernel_sqr(
        device const float * src0,
        device       float * dst,
        uint tpig[[thread_position_in_grid]]) {
    dst[tpig] = src0[tpig] * src0[tpig];
}

kernel void kernel_sqrt(
        device const float * src0,
        device       float * dst,
        uint tpig[[thread_position_in_grid]]) {
    dst[tpig] = sqrt(src0[tpig]);
}

kernel void kernel_sin(
        device const float * src0,
        device       float * dst,
        uint tpig[[thread_position_in_grid]]) {
    dst[tpig] = sin(src0[tpig]);
}

kernel void kernel_cos(
        device const float * src0,
        device       float * dst,
        uint tpig[[thread_position_in_grid]]) {
    dst[tpig] = cos(src0[tpig]);
}

kernel void kernel_sum_rows(
        device const float * src0,
        device       float * dst,
        constant  int64_t & ne00,
        constant  int64_t & ne01,
        constant  int64_t & ne02,
        constant  int64_t & ne03,
        constant uint64_t & nb00,
        constant uint64_t & nb01,
        constant uint64_t & nb02,
        constant uint64_t & nb03,
        constant  int64_t & ne10,
        constant  int64_t & ne11,
        constant  int64_t & ne12,
        constant  int64_t & ne13,
        constant uint64_t & nb10,
        constant uint64_t & nb11,
        constant uint64_t & nb12,
        constant uint64_t & nb13,
        constant  int64_t & ne0,
        constant  int64_t & ne1,
        constant  int64_t & ne2,
        constant  int64_t & ne3,
        constant uint64_t & nb0,
        constant uint64_t & nb1,
        constant uint64_t & nb2,
        constant uint64_t & nb3,
        uint3 tpig[[thread_position_in_grid]]) {
    int64_t i3 = tpig.z;
    int64_t i2 = tpig.y;
    int64_t i1 = tpig.x;

    if (i3 >= ne03 || i2 >= ne02 || i1 >= ne01) {
        return;
    }

    device const float * src_row = (device const float *) ((device const char *) src0 + i1*nb01 + i2*nb02 + i3*nb03);
    device       float * dst_row = (device       float *) ((device       char *) dst  + i1*nb1  + i2*nb2  + i3*nb3);

    float row_sum = 0;

    for (int64_t i0 = 0; i0 < ne00; i0++) {
        row_sum += src_row[i0];
    }

    dst_row[0] = row_sum;
}

template<typename T>
kernel void kernel_soft_max(
        device const  char * src0,
        device const  char * src1,
        device        char * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        constant     float & scale,
        constant     float & max_bias,
        constant     float & m0,
        constant     float & m1,
        constant  uint32_t & n_head_log2,
        threadgroup  float * buf [[threadgroup(0)]],
        uint  tgpig[[threadgroup_position_in_grid]],
        uint  tpitg[[thread_position_in_threadgroup]],
        uint  sgitg[[simdgroup_index_in_threadgroup]],
        uint  tiisg[[thread_index_in_simdgroup]],
        uint    ntg[[threads_per_threadgroup]]) {
    const int64_t i03 = (tgpig) / (ne02*ne01);
    const int64_t i02 = (tgpig - i03*ne02*ne01) / ne01;
    const int64_t i01 = (tgpig - i03*ne02*ne01 - i02*ne01);

    device const float * psrc0 = (device const float *) src0 + (i03*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00);
    device const     T * pmask = src1 != src0 ? (device const    T *) src1         + i01*ne00 : nullptr;
    device       float * pdst  = (device       float *) dst  + (i03*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00);

    float slope = 1.0f;

    // ALiBi
    if (max_bias > 0.0f) {
        const int64_t h = i02;

        const float base = h < n_head_log2 ? m0 : m1;
        const int   exp  = h < n_head_log2 ? h + 1 : 2*(h - n_head_log2) + 1;

        slope = pow(base, exp);
    }

    // parallel max
    float lmax = -INFINITY;

    for (int i00 = tpitg; i00 < ne00; i00 += ntg) {
        lmax = MAX(lmax, psrc0[i00]*scale + (pmask ? slope*pmask[i00] : 0.0f));
    }

    // find the max value in the block
    float max_val = simd_max(lmax);
    if (ntg > N_SIMDWIDTH) {
        if (sgitg == 0) {
            buf[tiisg] = -INFINITY;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tiisg == 0) {
            buf[sgitg] = max_val;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        max_val = buf[tiisg];
        max_val = simd_max(max_val);
    }

    // parallel sum
    float lsum = 0.0f;
    for (int i00 = tpitg; i00 < ne00; i00 += ntg) {
        const float exp_psrc0 = exp((psrc0[i00]*scale + (pmask ? slope*pmask[i00] : 0.0f)) - max_val);
        lsum += exp_psrc0;
        pdst[i00] = exp_psrc0;
    }

    // This barrier fixes a failing test
    // ref: https://github.com/ggerganov/ggml/pull/621#discussion_r1425156335
    threadgroup_barrier(mem_flags::mem_none);

    float sum = simd_sum(lsum);

    if (ntg > N_SIMDWIDTH) {
        if (sgitg == 0) {
            buf[tiisg] = 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tiisg == 0) {
            buf[sgitg] = sum;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        sum = buf[tiisg];
        sum = simd_sum(sum);
    }

    const float inv_sum = 1.0f/sum;

    for (int i00 = tpitg; i00 < ne00; i00 += ntg) {
        pdst[i00] *= inv_sum;
    }
}

template<typename T>
kernel void kernel_soft_max_4(
        device const  char * src0,
        device const  char * src1,
        device        char * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        constant     float & scale,
        constant     float & max_bias,
        constant     float & m0,
        constant     float & m1,
        constant  uint32_t & n_head_log2,
        threadgroup  float * buf [[threadgroup(0)]],
        uint  tgpig[[threadgroup_position_in_grid]],
        uint  tpitg[[thread_position_in_threadgroup]],
        uint  sgitg[[simdgroup_index_in_threadgroup]],
        uint  tiisg[[thread_index_in_simdgroup]],
        uint    ntg[[threads_per_threadgroup]]) {
    const int64_t i03 = (tgpig) / (ne02*ne01);
    const int64_t i02 = (tgpig - i03*ne02*ne01) / ne01;
    const int64_t i01 = (tgpig - i03*ne02*ne01 - i02*ne01);

    device const float4 * psrc4 = (device const float4 *) src0 + (i03*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00)/4;
    device const      T * pmask = src1 != src0 ? (device const     T *) src1         + i01*ne00/4 : nullptr;
    device       float4 * pdst4 = (device       float4 *) dst  + (i03*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00)/4;

    float slope = 1.0f;

    if (max_bias > 0.0f) {
        const int64_t h = i02;

        const float base = h < n_head_log2 ? m0 : m1;
        const int   exp  = h < n_head_log2 ? h + 1 : 2*(h - n_head_log2) + 1;

        slope = pow(base, exp);
    }

    // parallel max
    float4 lmax4 = -INFINITY;

    for (int i00 = tpitg; i00 < ne00/4; i00 += ntg) {
        lmax4 = fmax(lmax4, psrc4[i00]*scale + (float4)((pmask ? slope*pmask[i00] : 0.0f)));
    }

    const float lmax = MAX(MAX(lmax4[0], lmax4[1]), MAX(lmax4[2], lmax4[3]));

    float max_val = simd_max(lmax);
    if (ntg > N_SIMDWIDTH) {
        if (sgitg == 0) {
            buf[tiisg] = -INFINITY;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tiisg == 0) {
            buf[sgitg] = max_val;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        max_val = buf[tiisg];
        max_val = simd_max(max_val);
    }

    // parallel sum
    float4 lsum4 = 0.0f;
    for (int i00 = tpitg; i00 < ne00/4; i00 += ntg) {
        const float4 exp_psrc4 = exp((psrc4[i00]*scale + (float4)((pmask ? slope*pmask[i00] : 0.0f))) - max_val);
        lsum4 += exp_psrc4;
        pdst4[i00] = exp_psrc4;
    }

    const float lsum = lsum4[0] + lsum4[1] + lsum4[2] + lsum4[3];

    // This barrier fixes a failing test
    // ref: https://github.com/ggerganov/ggml/pull/621#discussion_r1425156335
    threadgroup_barrier(mem_flags::mem_none);

    float sum = simd_sum(lsum);

    if (ntg > N_SIMDWIDTH) {
        if (sgitg == 0) {
            buf[tiisg] = 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tiisg == 0) {
            buf[sgitg] = sum;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        sum = buf[tiisg];
        sum = simd_sum(sum);
    }

    const float inv_sum = 1.0f/sum;

    for (int i00 = tpitg; i00 < ne00/4; i00 += ntg) {
        pdst4[i00] *= inv_sum;
    }
}

typedef decltype(kernel_soft_max<float>)    kernel_soft_max_t;
typedef decltype(kernel_soft_max_4<float4>) kernel_soft_max_4_t;

template [[host_name("kernel_soft_max_f16")]]   kernel kernel_soft_max_t   kernel_soft_max<half>;
template [[host_name("kernel_soft_max_f32")]]   kernel kernel_soft_max_t   kernel_soft_max<float>;
template [[host_name("kernel_soft_max_f16_4")]] kernel kernel_soft_max_4_t kernel_soft_max_4<half4>;
template [[host_name("kernel_soft_max_f32_4")]] kernel kernel_soft_max_4_t kernel_soft_max_4<float4>;

kernel void kernel_diag_mask_inf(
        device const float * src0,
        device       float * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant       int & n_past,
        uint3 tpig[[thread_position_in_grid]]) {
    const int64_t i02 = tpig[2];
    const int64_t i01 = tpig[1];
    const int64_t i00 = tpig[0];

    if (i00 > n_past + i01) {
        dst[i02*ne01*ne00 + i01*ne00 + i00] = -INFINITY;
    } else {
        dst[i02*ne01*ne00 + i01*ne00 + i00] = src0[i02*ne01*ne00 + i01*ne00 + i00];
    }
}

kernel void kernel_diag_mask_inf_8(
        device const float4 * src0,
        device       float4 * dst,
        constant    int64_t & ne00,
        constant    int64_t & ne01,
        constant        int & n_past,
        uint3 tpig[[thread_position_in_grid]]) {

    const int64_t i = 2*tpig[0];

    dst[i+0] = src0[i+0];
    dst[i+1] = src0[i+1];
    int64_t i4 = 4*i;
    const int64_t i02 = i4/(ne00*ne01); i4 -= i02*ne00*ne01;
    const int64_t i01 = i4/(ne00);      i4 -= i01*ne00;
    const int64_t i00 = i4;
    for (int k = 3; k >= 0; --k) {
        if (i00 + 4 + k <= n_past + i01) {
            break;
        }
        dst[i+1][k] = -INFINITY;
        if (i00 + k > n_past + i01) {
            dst[i][k] = -INFINITY;
        }
    }
}

// ref: ggml.c:ggml_compute_forward_ssm_conv_f32
// TODO: optimize
kernel void kernel_ssm_conv_f32(
        device const  void * src0,
        device const  void * src1,
        device       float * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        constant  uint64_t & nb00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb02,
        constant   int64_t & ne10,
        constant   int64_t & ne11,
        constant  uint64_t & nb10,
        constant  uint64_t & nb11,
        constant   int64_t & ne0,
        constant   int64_t & ne1,
        constant   int64_t & ne2,
        constant  uint64_t & nb0,
        constant  uint64_t & nb1,
        constant  uint64_t & nb2,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint3   ntg[[threads_per_threadgroup]]) {
    const int64_t ir = tgpig.x;
    const int64_t i2 = tgpig.y;
    const int64_t i3 = tgpig.z;

    const int64_t nc  = ne10;
  //const int64_t ncs = ne00;
  //const int64_t nr  = ne01;
  //const int64_t n_t = ne1;
  //const int64_t n_s = ne2;

    device const float * s = (device const float *) ((device const char *) src0 + ir*nb01 + i2*nb00 + i3*nb02);
    device const float * c = (device const float *) ((device const char *) src1 + ir*nb11);
    device       float * x = (device       float *) ((device       char *) dst  + ir*nb0  + i2*nb1  + i3*nb2);

    float sumf = 0.0f;

    for (int64_t i0 = 0; i0 < nc; ++i0) {
        sumf += s[i0] * c[i0];
    }

    x[0] = sumf;
}

// ref: ggml.c:ggml_compute_forward_ssm_scan_f32
// TODO: optimize
kernel void kernel_ssm_scan_f32(
        device const void * src0,
        device const void * src1,
        device const void * src2,
        device const void * src3,
        device const void * src4,
        device const void * src5,
        device      float * dst,
        constant  int64_t & d_state,
        constant  int64_t & d_inner,
        constant  int64_t & n_seq_tokens,
        constant  int64_t & n_seqs,
        constant uint64_t & nb00,
        constant uint64_t & nb01,
        constant uint64_t & nb02,
        constant uint64_t & nb10,
        constant uint64_t & nb11,
        constant uint64_t & nb12,
        constant uint64_t & nb13,
        constant uint64_t & nb20,
        constant uint64_t & nb21,
        constant uint64_t & nb22,
        constant uint64_t & nb30,
        constant uint64_t & nb31,
        constant uint64_t & nb40,
        constant uint64_t & nb41,
        constant uint64_t & nb42,
        constant uint64_t & nb50,
        constant uint64_t & nb51,
        constant uint64_t & nb52,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint3   ntg[[threads_per_threadgroup]]) {
    const int64_t ir = tgpig.x;
    const int64_t i3 = tgpig.y;

    const int64_t nc  = d_state;
  //const int64_t nr  = d_inner;
    const int64_t n_t = n_seq_tokens;
  //const int64_t n_s = n_seqs;

    for (int64_t i2 = 0; i2 < n_t; ++i2) {
        device const float * s0 = (device const float *) ((device const char *) src0 + ir*nb01 + i3*nb02);
        device const float * x  = (device const float *) ((device const char *) src1 + ir*nb10 + i2*nb11 + i3*nb12);
        device const float * dt = (device const float *) ((device const char *) src2 + ir*nb20 + i2*nb21 + i3*nb22);
        device const float * A  = (device const float *) ((device const char *) src3 + ir*nb31);
        device const float * B  = (device const float *) ((device const char *) src4 + i2*nb41 + i3*nb42);
        device const float * C  = (device const float *) ((device const char *) src5 + i2*nb51 + i3*nb52);
        device       float * y  = (device       float *) ((device       char *) dst  + ir*nb10 + i2*nb11 + i3*nb12); // TODO: do not use src1 strides
        device       float * s  = (device       float *) ((device       char *) dst  + ir*nb01 + i3*nb02 +    nb13);

        if (i2 > 0) {
            s0 = s;
        }

        // i1 == 0
        float dt_soft_plus = dt[0] <= 20.0f ? log(1.0f + exp(dt[0])) : dt[0];
        float x_dt = x[0] * dt_soft_plus;
        float sumf = 0.0f;

        for (int64_t i0 = 0; i0 < nc; ++i0) {
            int64_t i = i0;
            float state = (s0[i] * exp(dt_soft_plus * A[i])) + (B[i0] * x_dt);
            sumf += state * C[i0];
            s[i] = state;
        }

        y[0] = sumf;
    }
}

kernel void kernel_argmax(
        device   const void * x,
        device      int32_t * dst,
        constant    int64_t & ncols,
        constant   uint64_t & nb01,
        threadgroup   float * shared_maxval [[threadgroup(0)]],
        threadgroup int32_t * shared_argmax [[threadgroup(1)]],
        uint  tgpig[[threadgroup_position_in_grid]],
        uint  tpitg[[thread_position_in_threadgroup]],
        uint  sgitg[[simdgroup_index_in_threadgroup]],
        uint  tiisg[[thread_index_in_simdgroup]],
        uint    ntg[[threads_per_threadgroup]]) {
    device const float * x_row = (device const float *) ((device const char *) x + tgpig * nb01);

    float   lmax = -INFINITY;
    int32_t larg = -1;

    for (int i00 = tpitg; i00 < ncols; i00 += ntg) {
        if (x_row[i00] > lmax) {
            lmax = x_row[i00];
            larg = i00;
        }
    }

    // find the argmax value in the block
    float max_val = simd_max(lmax);
    int32_t arg_val = simd_max(select(-1, larg, lmax == max_val));

    if (ntg > N_SIMDWIDTH) {
        if (sgitg == 0) {
            shared_maxval[tiisg] = -INFINITY;
            shared_argmax[tiisg] = -1;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tiisg == 0) {
            shared_maxval[sgitg] = max_val;
            shared_argmax[sgitg] = arg_val;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        max_val = shared_maxval[tiisg];
        arg_val = shared_argmax[tiisg];

        float max_val_reduced   = simd_max(max_val);
        int32_t arg_val_reduced = simd_max(select(-1, arg_val, max_val == max_val_reduced));

        dst[tgpig] = arg_val_reduced;

        return;
    }

    dst[tgpig] = arg_val;
}

kernel void kernel_norm(
        constant ggml_metal_kargs_norm & args,
        device const char * src0,
        device       char * dst,
        threadgroup float * shmem_f32 [[threadgroup(0)]],
        uint   tgpig[[threadgroup_position_in_grid]],
        ushort tpitg[[thread_position_in_threadgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort   ntg[[threads_per_threadgroup]]) {
    if (sgitg == 0) {
        shmem_f32[tiisg] = 0.0f;
    }

    device const float4 * x = (device const float4 *) (src0 + tgpig*args.nb01);

    float4 sumf4(0.0f);

    float sumf = 0.0f;

    for (int i00 = tpitg; i00 < args.ne00_4; i00 += ntg) {
        sumf4 += x[i00];
    }
    sumf = sumf4[0] + sumf4[1] + sumf4[2] + sumf4[3];
    sumf = simd_sum(sumf);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tiisg == 0) {
        shmem_f32[sgitg] = sumf;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    sumf = shmem_f32[tiisg];
    sumf = simd_sum(sumf);

    const float mean = sumf/args.ne00;

    device float4 * y = (device float4 *) dst + tgpig*args.ne00_4;

    sumf = 0.0f;
    for (int i00 = tpitg; i00 < args.ne00_4; i00 += ntg) {
        y[i00] = x[i00] - mean;
        sumf += dot(y[i00], y[i00]);
    }
    sumf = simd_sum(sumf);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tiisg == 0) {
        shmem_f32[sgitg] = sumf;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    sumf = shmem_f32[tiisg];
    sumf = simd_sum(sumf);

    const float variance = sumf/args.ne00;

    const float scale = 1.0f/sqrt(variance + args.eps);
    for (int i00 = tpitg; i00 < args.ne00_4; i00 += ntg) {
        y[i00] = y[i00] * scale;
    }
}

kernel void kernel_rms_norm(
        constant ggml_metal_kargs_rms_norm & args,
        device const char * src0,
        device       char * dst,
        threadgroup float * shmem_f32 [[threadgroup(0)]],
        uint   tgpig[[threadgroup_position_in_grid]],
        ushort tpitg[[thread_position_in_threadgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort   ntg[[threads_per_threadgroup]]) {
    if (sgitg == 0) {
        shmem_f32[tiisg] = 0.0f;
    }

    device const float4 * x = (device const float4 *) (src0 + tgpig*args.nb01);

    float sumf = 0.0f;

    // parallel sum
    for (int i00 = tpitg; i00 < args.ne00_4; i00 += ntg) {
        sumf += dot(x[i00], x[i00]);
    }
    sumf = simd_sum(sumf);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tiisg == 0) {
        shmem_f32[sgitg] = sumf;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    sumf = shmem_f32[tiisg];
    sumf = simd_sum(sumf);

    const float mean  = sumf/args.ne00;
    const float scale = 1.0f/sqrt(mean + args.eps);

    device float4 * y = (device float4 *) dst + tgpig*args.ne00_4;
    for (int i00 = tpitg; i00 < args.ne00_4; i00 += ntg) {
        y[i00] = x[i00] * scale;
    }
}

kernel void kernel_group_norm(
        device const float * src0,
        device       float * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        constant  uint64_t & nb00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb02,
        constant   int32_t & n_groups,
        constant     float & eps,
        threadgroup float  * buf [[threadgroup(0)]],
        uint tgpig[[threadgroup_position_in_grid]],
        uint tpitg[[thread_position_in_threadgroup]],
        uint sgitg[[simdgroup_index_in_threadgroup]],
        uint tiisg[[thread_index_in_simdgroup]],
        uint   ntg[[threads_per_threadgroup]]) {
    const int64_t ne = ne00*ne01*ne02;
    const int64_t gs = ne00*ne01*((ne02 + n_groups - 1) / n_groups);

    int start = tgpig * gs;
    int end   = start + gs;

    start += tpitg;

    if (end >= ne) {
        end = ne;
    }

    float tmp = 0.0f; // partial sum for thread in warp

    for (int j = start; j < end; j += ntg) {
        tmp += src0[j];
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    tmp = simd_sum(tmp);
    if (ntg > N_SIMDWIDTH) {
        if (sgitg == 0) {
            buf[tiisg] = 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tiisg == 0) {
            buf[sgitg] = tmp;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        tmp = buf[tiisg];
        tmp = simd_sum(tmp);
    }

    const float mean = tmp / gs;
    tmp = 0.0f;

    for (int j = start; j < end; j += ntg) {
        float xi = src0[j] - mean;
        dst[j] = xi;
        tmp += xi * xi;
    }

    tmp = simd_sum(tmp);
    if (ntg > N_SIMDWIDTH) {
        if (sgitg == 0) {
            buf[tiisg] = 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tiisg == 0) {
            buf[sgitg] = tmp;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        tmp = buf[tiisg];
        tmp = simd_sum(tmp);
    }

    const float variance = tmp / gs;
    const float scale = 1.0f/sqrt(variance + eps);
    for (int j = start; j < end; j += ntg) {
        dst[j] *= scale;
    }
}

// function for calculate inner product between half a q4_0 block and 16 floats (yl), sumy is SUM(yl[i])
// il indicates where the q4 quants begin (0 or QK4_0/4)
// we assume that the yl's have been multiplied with the appropriate scale factor
// that corresponds to the missing bit shifts (1, 1/16, 1/256, 1/4096)
inline float block_q_n_dot_y(device const block_q4_0 * qb_curr, float sumy, thread float * yl, int il) {
    float d = qb_curr->d;

    float acc[4] = { 0.0f, 0.0f, 0.0f, 0.0f };

    device const uint16_t * qs = ((device const uint16_t *) qb_curr + 1 + il/2);

    for (int i = 0; i < 8; i += 2) {
        acc[0] += yl[i + 0] * (qs[i / 2] & 0x000F);
        acc[1] += yl[i + 1] * (qs[i / 2] & 0x0F00);
        acc[2] += yl[i + 8] * (qs[i / 2] & 0x00F0);
        acc[3] += yl[i + 9] * (qs[i / 2] & 0xF000);
    }

    return d * (sumy * -8.f + acc[0] + acc[1] + acc[2] + acc[3]);
}

// function for calculate inner product between half a q4_1 block and 16 floats (yl), sumy is SUM(yl[i])
// il indicates where the q4 quants begin (0 or QK4_0/4)
// we assume that the yl's have been multiplied with the appropriate scale factor
// that corresponds to the missing bit shifts (1, 1/16, 1/256, 1/4096)
inline float block_q_n_dot_y(device const block_q4_1 * qb_curr, float sumy, thread float * yl, int il) {
    float d = qb_curr->d;
    float m = qb_curr->m;

    float acc[4] = { 0.0f, 0.0f, 0.0f, 0.0f };

    device const uint16_t * qs = ((device const uint16_t *) qb_curr + 2 + il/2);

    for (int i = 0; i < 8; i+=2) {
        acc[0] += yl[i + 0] * (qs[i / 2] & 0x000F);
        acc[1] += yl[i + 1] * (qs[i / 2] & 0x0F00);
        acc[2] += yl[i + 8] * (qs[i / 2] & 0x00F0);
        acc[3] += yl[i + 9] * (qs[i / 2] & 0xF000);
    }

    return d * (acc[0] + acc[1] + acc[2] + acc[3]) + sumy * m;
}

// function for calculate inner product between half a q5_0 block and 16 floats (yl), sumy is SUM(yl[i])
// il indicates where the q5 quants begin (0 or QK5_0/4)
// we assume that the yl's have been multiplied with the appropriate scale factor
// that corresponds to the missing bit shifts (1, 1/16, 1/256, 1/4096)
inline float block_q_n_dot_y(device const block_q5_0 * qb_curr, float sumy, thread float * yl, int il) {
    float d = qb_curr->d;

    float acc[4] = { 0.0f, 0.0f, 0.0f, 0.0f };

    device const uint16_t * qs =  ((device const uint16_t *)qb_curr + 3 + il/2);
           const uint32_t   qh = *((device const uint32_t *)qb_curr->qh);

    for (int i = 0; i < 8; i+=2) {
        acc[0] += yl[i + 0] * ((qs[i / 2] & 0x000F) | ((qh >> (i+0+il        ) << 4 ) & 0x00010));
        acc[1] += yl[i + 1] * ((qs[i / 2] & 0x0F00) | ((qh >> (i+1+il        ) << 12) & 0x01000));
        acc[2] += yl[i + 8] * ((qs[i / 2] & 0x00F0) | ((qh >> (i+0+il+QK5_0/2) << 8 ) & 0x00100));
        acc[3] += yl[i + 9] * ((qs[i / 2] & 0xF000) | ((qh >> (i+1+il+QK5_0/2) << 16) & 0x10000));
    }

    return d * (sumy * -16.f + acc[0] + acc[1] + acc[2] + acc[3]);
}

// function for calculate inner product between half a q5_1 block and 16 floats (yl), sumy is SUM(yl[i])
// il indicates where the q5 quants begin (0 or QK5_1/4)
// we assume that the yl's have been multiplied with the appropriate scale factor
// that corresponds to the missing bit shifts (1, 1/16, 1/256, 1/4096)
inline float block_q_n_dot_y(device const block_q5_1 * qb_curr, float sumy, thread float * yl, int il) {
    float d = qb_curr->d;
    float m = qb_curr->m;

    float acc[4] = { 0.0f, 0.0f, 0.0f, 0.0f };

    device const uint16_t * qs =  ((device const uint16_t *)qb_curr + 4 + il/2);
           const uint32_t   qh = *((device const uint32_t *)qb_curr->qh);

    for (int i = 0; i < 8; i+=2) {
        acc[0] += yl[i + 0] * ((qs[i / 2] & 0x000F) | ((qh >> (i+0+il        ) << 4 ) & 0x00010));
        acc[1] += yl[i + 1] * ((qs[i / 2] & 0x0F00) | ((qh >> (i+1+il        ) << 12) & 0x01000));
        acc[2] += yl[i + 8] * ((qs[i / 2] & 0x00F0) | ((qh >> (i+0+il+QK5_0/2) << 8 ) & 0x00100));
        acc[3] += yl[i + 9] * ((qs[i / 2] & 0xF000) | ((qh >> (i+1+il+QK5_0/2) << 16) & 0x10000));
    }

    return d * (acc[0] + acc[1] + acc[2] + acc[3]) + sumy * m;
}

// putting them in the kernel cause a significant performance penalty
#define N_DST 4        // each SIMD group works on 4 rows
#define N_SIMDGROUP 2  // number of SIMD groups in a thread group
//Note: This is a template, but strictly speaking it only applies to
//      quantizations where the block size is 32. It also does not
//      guard against the number of rows not being divisible by
//      N_DST, so this is another explicit assumption of the implementation.
template<typename block_q_type, int nr, int nsg, int nw, typename args_t>
void mul_vec_q_n_f32_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const int nb = args.ne00/QK4_0;

    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * nsg + sgitg) * nr;

    const uint i12 = im%args.ne12;
    const uint i13 = im/args.ne12;

  //const uint64_t offset0 = first_row*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;
    const uint64_t offset1 =        r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

  //device const block_q_type * x = (device const block_q_type *) (src0 + offset0);
    device const float        * y = (device const float        *) (src1 + offset1);

    // pointers to src0 rows
    device const block_q_type * ax[nr];
    for (int row = 0; row < nr; ++row) {
        const uint64_t offset0 = (first_row + row)*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;

        ax[row] = (device const block_q_type *) ((device char *) src0 + offset0);
    }

    float yl[16]; // src1 vector cache
    float sumf[nr] = {0.f};

    const short ix = (tiisg/2);
    const short il = (tiisg%2)*8;

    device const float * yb = y + ix*QK4_0 + il;

    // each thread in a SIMD group deals with half a block.
    for (int ib = ix; ib < nb; ib += nw/2) {
        float sumy[2] = { 0.f, 0.f };

#pragma unroll
        for (int i = 0; i < 8; i += 2) {
            sumy[0]  += yb[i +  0] + yb[i +  1];
            yl[i + 0] = yb[i +  0];
            yl[i + 1] = yb[i +  1]/256.f;

            sumy[1]  += yb[i + 16] + yb[i + 17];
            yl[i + 8] = yb[i + 16]/16.f;
            yl[i + 9] = yb[i + 17]/4096.f;
        }

#pragma unroll
        for (int row = 0; row < nr; row++) {
            sumf[row] += block_q_n_dot_y(ax[row] + ib, sumy[0] + sumy[1], yl, il);
        }

        yb += QK4_0 * 16;
    }

    device float * dst_f32 = (device float *) dst + im*args.ne0*args.ne1 + r1*args.ne0;

    for (int row = 0; row < nr; ++row) {
        const float tot = simd_sum(sumf[row]);

        if (tiisg == 0 && first_row + row < args.ne01) {
            dst_f32[first_row + row] = tot;
        }
    }
}

kernel void kernel_mul_mv_q4_0_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    mul_vec_q_n_f32_impl<block_q4_0, N_DST, N_SIMDGROUP, N_SIMDWIDTH, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, nullptr, tgpig, tiisg, sgitg);
}

kernel void kernel_mul_mv_q4_1_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
     mul_vec_q_n_f32_impl<block_q4_1, N_DST, N_SIMDGROUP, N_SIMDWIDTH, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, nullptr, tgpig, tiisg, sgitg);
}

kernel void kernel_mul_mv_q5_0_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    mul_vec_q_n_f32_impl<block_q5_0, N_DST, N_SIMDGROUP, N_SIMDWIDTH, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, nullptr, tgpig, tiisg, sgitg);
}

kernel void kernel_mul_mv_q5_1_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    mul_vec_q_n_f32_impl<block_q5_1, N_DST, N_SIMDGROUP, N_SIMDWIDTH, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, nullptr, tgpig, tiisg, sgitg);
}

#define NB_Q8_0 8

template<typename args_t>
void kernel_mul_mv_q8_0_f32_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const int nr  = N_DST;
    const int nsg = N_SIMDGROUP;
    const int nw  = N_SIMDWIDTH;

    const int nb = args.ne00/QK8_0;
    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0*nsg + sgitg)*nr;

    const uint i12 = im%args.ne12;
    const uint i13 = im/args.ne12;

  //const uint64_t offset0 = first_row*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;
    const uint64_t offset1 =        r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

  //device const block_q8_0 * x = (device const block_q8_0 *) (src0 + offset0);
    device const float      * y = (device const float      *) (src1 + offset1);

    // pointers to src0 rows
    device const block_q8_0 * ax[nr];
    for (int row = 0; row < nr; ++row) {
        const uint64_t offset0 = (first_row + row)*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;

        ax[row] = (device const block_q8_0 *) ((device char *) src0 + offset0);
    }

    float yl[NB_Q8_0];
    float sumf[nr] = { 0.f };

    const short ix = tiisg/4;
    const short il = tiisg%4;

    device const float * yb = y + ix*QK8_0 + il*NB_Q8_0;

    // each thread in a SIMD group deals with NB_Q8_0 quants at a time
    for (int ib = ix; ib < nb; ib += nw/4) {
        for (short i = 0; i < NB_Q8_0; ++i) {
            yl[i] = yb[i];
        }

        for (int row = 0; row < nr; row++) {
            device const int8_t * qs = ax[row][ib].qs + il*NB_Q8_0;
            float sumq = 0.f;
            for (short iq = 0; iq < NB_Q8_0; ++iq) {
                sumq += qs[iq] * yl[iq];
            }
            sumf[row] += sumq*ax[row][ib].d;
        }

        yb += nw*NB_Q8_0;
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    for (int row = 0; row < nr; ++row) {
        const float tot = simd_sum(sumf[row]);

        if (tiisg == 0 && first_row + row < args.ne01) {
            dst_f32[first_row + row] = tot;
        }
    }
}

[[host_name("kernel_mul_mv_q8_0_f32")]]
kernel void kernel_mul_mv_q8_0_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_q8_0_f32_impl<constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, nullptr, tgpig, tiisg, sgitg);
}

// mat-vec kernel processing in chunks of float4
// chpb - chunks per quantization block
template<short nxpsg, short r1ptg, typename q_t, short chpb, void (*deq_t4)(device const q_t *, short, thread float4 &) >
void kernel_mul_mv_ext_q4_f32_impl(
        constant ggml_metal_kargs_mul_mv_ext & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]]) {
    const short chpt = 4; // chunks per thread

  //const short nxpsg = (32);
    const short nypsg = (32/nxpsg);

    const short tx = tiisg%nxpsg;
    const short ty = tiisg/nxpsg;

    const int i01 = tgpig.x*(nypsg*args.nsg) + nypsg*sgitg + ty;
    const int i11 = tgpig.y*r1ptg;
    const int i1m = tgpig.z;

    const int i12 = i1m%args.ne12;
    const int i13 = i1m/args.ne12;

    const uint64_t offset0 = i01*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;
    const uint64_t offset1 = i11*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

    device const q_t * xq = (i01 < args.ne01) ? (device const q_t *) (src0 + offset0) + tx/chpb : (device const q_t *) src0;

    device const float4 * y4[r1ptg];

    for (int ir1 = 0; ir1 < r1ptg; ++ir1) {
        y4[ir1] = (i11 + ir1 < args.ne11) ? (device const float4 *) (src1 + offset1 + ir1*args.nb11) + tx : (device const float4 *) src1;
    }

    float sumf[r1ptg] = { [ 0 ... r1ptg - 1 ] = 0.0f };

    short cch = tx%chpb; // current chunk index

    for (int ich = tx; 4*ich < args.ne00; ich += chpt*nxpsg) {
        float4 lx[chpt];

#pragma unroll(chpt)
        for (short ch = 0; ch < chpt; ++ch) {
            deq_t4(xq, cch, lx[ch]);

            cch += nxpsg;
            if (cch >= chpb) {
                xq  += cch/chpb;
                cch %= chpb;
            }
        }

#pragma unroll(chpt)
        for (short ch = 0; ch < chpt; ++ch) {
#pragma unroll(r1ptg)
            for (short ir1 = 0; ir1 < r1ptg; ++ir1) {
                sumf[ir1] += dot(lx[ch], y4[ir1][ch*nxpsg]);

            }
        }

#pragma unroll(r1ptg)
        for (short ir1 = 0; ir1 < r1ptg; ++ir1) {
            y4[ir1] += chpt*nxpsg;
        }
    }

    // reduce only the threads in each row
    for (short ir1 = 0; ir1 < r1ptg; ++ir1) {
        if (nxpsg >= 32) {
            sumf[ir1] += simd_shuffle_down(sumf[ir1], 16);
        }
        if (nxpsg >= 16) {
            sumf[ir1] += simd_shuffle_down(sumf[ir1],  8);
        }
        if (nxpsg >= 8) {
            sumf[ir1] += simd_shuffle_down(sumf[ir1],  4);
        }
        if (nxpsg >= 4) {
            sumf[ir1] += simd_shuffle_down(sumf[ir1],  2);
        }
        if (nxpsg >= 2) {
            sumf[ir1] += simd_shuffle_down(sumf[ir1],  1);
        }

        //sumf[ir1] = simd_sum(sumf[ir1]);
    }

    if (tx == 0) {
        for (short ir1 = 0; ir1 < r1ptg && i11 + ir1 < args.ne11; ++ir1) {
            device float * dst_f32 = (device float *) dst + (uint64_t)i1m*args.ne0*args.ne1 + (uint64_t)(i11 + ir1)*args.ne0;

            if (i01 < args.ne01) {
                dst_f32[i01] = sumf[ir1];
            }
        }
    }
}

// mat-vec kernel processing in chunks of float4x4
template<short nxpsg, short r1ptg, typename q_t, short chpb, void (*deq_t4x4)(device const q_t *, short, thread float4x4 &) >
void kernel_mul_mv_ext_q4x4_f32_impl(
        constant ggml_metal_kargs_mul_mv_ext & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]]) {
    const short chpt = 1;

  //const short nxpsg = (32);
    const short nypsg = (32/nxpsg);

    const short tx = tiisg%nxpsg;
    const short ty = tiisg/nxpsg;

    const int i01 = tgpig.x*(nypsg*args.nsg) + nypsg*sgitg + ty;
    const int i11 = tgpig.y*r1ptg;
    const int i1m = tgpig.z;

    const int i12 = i1m%args.ne12;
    const int i13 = i1m/args.ne12;

    const uint64_t offset0 = i01*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;
    const uint64_t offset1 = i11*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

    device const q_t * xq = (i01 < args.ne01) ? (device const q_t *) (src0 + offset0) + tx/chpb : (device const q_t *) src0;

    device const float4x4 * y4x4[r1ptg];

    for (int ir1 = 0; ir1 < r1ptg; ++ir1) {
        y4x4[ir1] = (i11 + ir1 < args.ne11) ? (device const float4x4 *) (src1 + offset1 + ir1*args.nb11) + tx : (device const float4x4 *) src1;
    }

    float sumf[r1ptg] = { [ 0 ... r1ptg - 1 ] = 0.0f };

    short cch = tx%chpb;

    for (int ich = tx; 16*ich < args.ne00; ich += chpt*nxpsg) {
        float4x4 lx[chpt];

#pragma unroll(chpt)
        for (short ch = 0; ch < chpt; ++ch) {
            deq_t4x4(xq, cch, lx[ch]);

            cch += nxpsg;
            if (cch >= chpb) {
                xq  += cch/chpb;
                cch %= chpb;
            }
        }

#pragma unroll(chpt)
        for (short ch = 0; ch < chpt; ++ch) {
#pragma unroll(r1ptg)
            for (short ir1 = 0; ir1 < r1ptg; ++ir1) {
                sumf[ir1] +=
                    dot(lx[ch][0], y4x4[ir1][ch*nxpsg][0]) +
                    dot(lx[ch][1], y4x4[ir1][ch*nxpsg][1]) +
                    dot(lx[ch][2], y4x4[ir1][ch*nxpsg][2]) +
                    dot(lx[ch][3], y4x4[ir1][ch*nxpsg][3]);

            }
        }

#pragma unroll(r1ptg)
        for (short ir1 = 0; ir1 < r1ptg; ++ir1) {
            y4x4[ir1] += chpt*nxpsg;
        }
    }

    for (short ir1 = 0; ir1 < r1ptg; ++ir1) {
        if (nxpsg >= 32) {
            sumf[ir1] += simd_shuffle_down(sumf[ir1], 16);
        }
        if (nxpsg >= 16) {
            sumf[ir1] += simd_shuffle_down(sumf[ir1],  8);
        }
        if (nxpsg >= 8) {
            sumf[ir1] += simd_shuffle_down(sumf[ir1],  4);
        }
        if (nxpsg >= 4) {
            sumf[ir1] += simd_shuffle_down(sumf[ir1],  2);
        }
        if (nxpsg >= 2) {
            sumf[ir1] += simd_shuffle_down(sumf[ir1],  1);
        }

        //sumf[ir1] = simd_sum(sumf[ir1]);
    }

    if (tx == 0) {
        for (short ir1 = 0; ir1 < r1ptg && i11 + ir1 < args.ne11; ++ir1) {
            device float * dst_f32 = (device float *) dst + (uint64_t)i1m*args.ne0*args.ne1 + (uint64_t)(i11 + ir1)*args.ne0;

            if (i01 < args.ne01) {
                dst_f32[i01] = sumf[ir1];
            }
        }
    }
}

// dispatchers needed for compile-time nxpsg
// epb - elements per quantization block
template<short r1ptg, typename q_t, short epb, void (*deq_t4)(device const q_t *, short, thread float4 &)>
kernel void kernel_mul_mv_ext_q4_f32_disp(
        constant ggml_metal_kargs_mul_mv_ext & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]]) {
    switch (args.nxpsg) {
        case 4:  kernel_mul_mv_ext_q4_f32_impl<4,  r1ptg, q_t, epb/4, deq_t4>(args, src0, src1, dst, tgpig, tiisg, sgitg); break;
        case 8:  kernel_mul_mv_ext_q4_f32_impl<8,  r1ptg, q_t, epb/4, deq_t4>(args, src0, src1, dst, tgpig, tiisg, sgitg); break;
        case 16: kernel_mul_mv_ext_q4_f32_impl<16, r1ptg, q_t, epb/4, deq_t4>(args, src0, src1, dst, tgpig, tiisg, sgitg); break;
        case 32: kernel_mul_mv_ext_q4_f32_impl<32, r1ptg, q_t, epb/4, deq_t4>(args, src0, src1, dst, tgpig, tiisg, sgitg); break;
    }
}

template<short r1ptg, typename q_t, short epb, void (*deq_t4x4)(device const q_t *, short, thread float4x4 &)>
kernel void kernel_mul_mv_ext_q4x4_f32_disp(
        constant ggml_metal_kargs_mul_mv_ext & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]]) {
    switch (args.nxpsg) {
        case 4:  kernel_mul_mv_ext_q4x4_f32_impl<4,  r1ptg, q_t, epb/16, deq_t4x4>(args, src0, src1, dst, tgpig, tiisg, sgitg); break;
        case 8:  kernel_mul_mv_ext_q4x4_f32_impl<8,  r1ptg, q_t, epb/16, deq_t4x4>(args, src0, src1, dst, tgpig, tiisg, sgitg); break;
        case 16: kernel_mul_mv_ext_q4x4_f32_impl<16, r1ptg, q_t, epb/16, deq_t4x4>(args, src0, src1, dst, tgpig, tiisg, sgitg); break;
        case 32: kernel_mul_mv_ext_q4x4_f32_impl<32, r1ptg, q_t, epb/16, deq_t4x4>(args, src0, src1, dst, tgpig, tiisg, sgitg); break;
    }
}

typedef decltype(kernel_mul_mv_ext_q4_f32_disp  <2, block_q8_0, 32,  dequantize_q8_0_t4>) mul_mv_ext_q4_f32_t;
typedef decltype(kernel_mul_mv_ext_q4x4_f32_disp<2, block_q4_K, 256, dequantize_q4_K>)    mul_mv_ext_q4x4_f32_t;

template [[host_name("kernel_mul_mv_ext_f16_f32_r1_2")]]    kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<2, half4,        4,  dequantize_f16_t4>;
template [[host_name("kernel_mul_mv_ext_f16_f32_r1_3")]]    kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<3, half4,        4,  dequantize_f16_t4>;
template [[host_name("kernel_mul_mv_ext_f16_f32_r1_4")]]    kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<4, half4,        4,  dequantize_f16_t4>;
template [[host_name("kernel_mul_mv_ext_f16_f32_r1_5")]]    kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<5, half4,        4,  dequantize_f16_t4>;

template [[host_name("kernel_mul_mv_ext_q4_0_f32_r1_2")]]   kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<2, block_q4_0,   32, dequantize_q4_0_t4>;
template [[host_name("kernel_mul_mv_ext_q4_0_f32_r1_3")]]   kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<3, block_q4_0,   32, dequantize_q4_0_t4>;
template [[host_name("kernel_mul_mv_ext_q4_0_f32_r1_4")]]   kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<4, block_q4_0,   32, dequantize_q4_0_t4>;
template [[host_name("kernel_mul_mv_ext_q4_0_f32_r1_5")]]   kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<5, block_q4_0,   32, dequantize_q4_0_t4>;

template [[host_name("kernel_mul_mv_ext_q4_1_f32_r1_2")]]   kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<2, block_q4_1,   32, dequantize_q4_1_t4>;
template [[host_name("kernel_mul_mv_ext_q4_1_f32_r1_3")]]   kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<3, block_q4_1,   32, dequantize_q4_1_t4>;
template [[host_name("kernel_mul_mv_ext_q4_1_f32_r1_4")]]   kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<4, block_q4_1,   32, dequantize_q4_1_t4>;
template [[host_name("kernel_mul_mv_ext_q4_1_f32_r1_5")]]   kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<5, block_q4_1,   32, dequantize_q4_1_t4>;

template [[host_name("kernel_mul_mv_ext_q5_0_f32_r1_2")]]   kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<2, block_q5_0,   32, dequantize_q5_0_t4>;
template [[host_name("kernel_mul_mv_ext_q5_0_f32_r1_3")]]   kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<3, block_q5_0,   32, dequantize_q5_0_t4>;
template [[host_name("kernel_mul_mv_ext_q5_0_f32_r1_4")]]   kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<4, block_q5_0,   32, dequantize_q5_0_t4>;
template [[host_name("kernel_mul_mv_ext_q5_0_f32_r1_5")]]   kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<5, block_q5_0,   32, dequantize_q5_0_t4>;

template [[host_name("kernel_mul_mv_ext_q5_1_f32_r1_2")]]   kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<2, block_q5_1,   32, dequantize_q5_1_t4>;
template [[host_name("kernel_mul_mv_ext_q5_1_f32_r1_3")]]   kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<3, block_q5_1,   32, dequantize_q5_1_t4>;
template [[host_name("kernel_mul_mv_ext_q5_1_f32_r1_4")]]   kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<4, block_q5_1,   32, dequantize_q5_1_t4>;
template [[host_name("kernel_mul_mv_ext_q5_1_f32_r1_5")]]   kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<5, block_q5_1,   32, dequantize_q5_1_t4>;

template [[host_name("kernel_mul_mv_ext_q8_0_f32_r1_2")]]   kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<2, block_q8_0,   32, dequantize_q8_0_t4>;
template [[host_name("kernel_mul_mv_ext_q8_0_f32_r1_3")]]   kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<3, block_q8_0,   32, dequantize_q8_0_t4>;
template [[host_name("kernel_mul_mv_ext_q8_0_f32_r1_4")]]   kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<4, block_q8_0,   32, dequantize_q8_0_t4>;
template [[host_name("kernel_mul_mv_ext_q8_0_f32_r1_5")]]   kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<5, block_q8_0,   32, dequantize_q8_0_t4>;

template [[host_name("kernel_mul_mv_ext_iq4_nl_f32_r1_2")]] kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<2, block_iq4_nl, 32, dequantize_iq4_nl_t4>;
template [[host_name("kernel_mul_mv_ext_iq4_nl_f32_r1_3")]] kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<3, block_iq4_nl, 32, dequantize_iq4_nl_t4>;
template [[host_name("kernel_mul_mv_ext_iq4_nl_f32_r1_4")]] kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<4, block_iq4_nl, 32, dequantize_iq4_nl_t4>;
template [[host_name("kernel_mul_mv_ext_iq4_nl_f32_r1_5")]] kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<5, block_iq4_nl, 32, dequantize_iq4_nl_t4>;

template [[host_name("kernel_mul_mv_ext_q4_K_f32_r1_2")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4x4_f32_disp<2, block_q4_K, 256, dequantize_q4_K>;
template [[host_name("kernel_mul_mv_ext_q4_K_f32_r1_3")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4x4_f32_disp<3, block_q4_K, 256, dequantize_q4_K>;
template [[host_name("kernel_mul_mv_ext_q4_K_f32_r1_4")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4x4_f32_disp<4, block_q4_K, 256, dequantize_q4_K>;
template [[host_name("kernel_mul_mv_ext_q4_K_f32_r1_5")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4x4_f32_disp<5, block_q4_K, 256, dequantize_q4_K>;

template [[host_name("kernel_mul_mv_ext_q5_K_f32_r1_2")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4x4_f32_disp<2, block_q5_K, 256, dequantize_q5_K>;
template [[host_name("kernel_mul_mv_ext_q5_K_f32_r1_3")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4x4_f32_disp<3, block_q5_K, 256, dequantize_q5_K>;
template [[host_name("kernel_mul_mv_ext_q5_K_f32_r1_4")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4x4_f32_disp<4, block_q5_K, 256, dequantize_q5_K>;
template [[host_name("kernel_mul_mv_ext_q5_K_f32_r1_5")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4x4_f32_disp<5, block_q5_K, 256, dequantize_q5_K>;

template [[host_name("kernel_mul_mv_ext_q6_K_f32_r1_2")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4x4_f32_disp<2, block_q6_K, 256, dequantize_q6_K>;
template [[host_name("kernel_mul_mv_ext_q6_K_f32_r1_3")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4x4_f32_disp<3, block_q6_K, 256, dequantize_q6_K>;
template [[host_name("kernel_mul_mv_ext_q6_K_f32_r1_4")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4x4_f32_disp<4, block_q6_K, 256, dequantize_q6_K>;
template [[host_name("kernel_mul_mv_ext_q6_K_f32_r1_5")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4x4_f32_disp<5, block_q6_K, 256, dequantize_q6_K>;

#define N_MV_T_T 4

template<typename T0, typename T04, typename T1, typename T14, typename args_t>
void kernel_mul_mv_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig,
        ushort tiisg) {
    const int r0 = tgpig.x;
    const int rb = tgpig.y*N_MV_T_T;
    const int im = tgpig.z;

    const uint i12 = im%args.ne12;
    const uint i13 = im/args.ne12;

    const uint64_t offset0 = r0*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;

    device const T0 * x = (device const T0 *) (src0 + offset0);

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1;

    if (args.ne00 < 128) {
        for (int row = 0; row < N_MV_T_T; ++row) {
            int r1 = rb + row;
            if (r1 >= args.ne11) {
                break;
            }

            const uint64_t offset1 = r1*args.nb11 + (i12   )*args.nb12 + (i13   )*args.nb13;

            device const T1 * y = (device const T1 *) (src1 + offset1);

            float sumf = 0;
            for (int i = tiisg; i < args.ne00; i += 32) {
                sumf += (T0) x[i] * (T1) y[i];
            }

            float all_sum = simd_sum(sumf);
            if (tiisg == 0) {
                dst_f32[(uint64_t)r1*args.ne0 + r0] = all_sum;
            }
        }
    } else {
        device const T04 * x4 = (device const T04 *) x;
        for (int row = 0; row < N_MV_T_T; ++row) {
            int r1 = rb + row;
            if (r1 >= args.ne11) {
                break;
            }

            const uint64_t offset1 = r1*args.nb11 + (i12   )*args.nb12 + (i13   )*args.nb13;

            device const T1  * y  = (device const T1  *) (src1 + offset1);
            device const T14 * y4 = (device const T14 *) y;

            float sumf = 0;
            for (int i = tiisg; i < args.ne00/4; i += 32) {
                sumf += dot((float4) x4[i], (float4) y4[i]);
            }

            float all_sum = simd_sum(sumf);
            if (tiisg == 0) {
                for (int i = 4*(args.ne00/4); i < args.ne00; ++i) all_sum += (float) (x[i] * y[i]);
                dst_f32[(uint64_t)r1*args.ne0 + r0] = all_sum;
            }
        }
    }
}

template<typename T0, typename T04, typename T1, typename T14>
kernel void kernel_mul_mv(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]]) {
    kernel_mul_mv_impl<T0, T04, T1, T14, constant ggml_metal_kargs_mul_mv &>(
        args,
        src0,
        src1,
        dst,
        tgpig,
        tiisg);
}

typedef decltype(kernel_mul_mv<half, half4, half, half4>) mul_mv_t;

template [[host_name("kernel_mul_mv_f32_f32")]]   kernel mul_mv_t kernel_mul_mv<float,  float4,  float,  float4>;
template [[host_name("kernel_mul_mv_f16_f32")]]   kernel mul_mv_t kernel_mul_mv<half,   half4,   float,  float4>;
template [[host_name("kernel_mul_mv_f16_f16")]]   kernel mul_mv_t kernel_mul_mv<half,   half4,   half,   half4>;
#if defined(GGML_METAL_USE_BF16)
template [[host_name("kernel_mul_mv_bf16_f32")]]  kernel mul_mv_t kernel_mul_mv<bfloat, bfloat4, float,  float4>;
template [[host_name("kernel_mul_mv_bf16_bf16")]] kernel mul_mv_t kernel_mul_mv<bfloat, bfloat4, bfloat, bfloat4>;
#endif

template<typename T, typename T4>
kernel void kernel_mul_mv_1row(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]]) {

    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const uint i12 = im%args.ne12;
    const uint i13 = im/args.ne12;

    const uint64_t offset0 = r0*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;
    const uint64_t offset1 = r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

    device const T     * x = (device const T     *) (src0 + offset0);
    device const float * y = (device const float *) (src1 + offset1);

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    float sumf = 0;
    if (args.ne00 < 128) {
        for (int i = tiisg; i < args.ne00; i += 32) {
            sumf += (float) x[i] * (float) y[i];
        }
        float all_sum = simd_sum(sumf);
        if (tiisg == 0) {
            dst_f32[r0] = all_sum;
        }
    } else {
        device const T4     * x4 = (device const T4     *) x;
        device const float4 * y4 = (device const float4 *) y;

        for (int i = tiisg; i < args.ne00/4; i += 32) {
            sumf += dot((float4) x4[i], y4[i]);
        }

        float all_sum = simd_sum(sumf);

        if (tiisg == 0) {
            for (int i = 4*(args.ne00/4); i < args.ne00; ++i) all_sum += (float) (x[i] * y[i]);
            dst_f32[r0] = all_sum;
        }
    }
}

typedef decltype(kernel_mul_mv_1row<half, half4>) mul_mv_1row_t;

template [[host_name("kernel_mul_mv_f16_f32_1row")]]  kernel mul_mv_1row_t kernel_mul_mv_1row<half,   half4>;
#if defined(GGML_METAL_USE_BF16)
template [[host_name("kernel_mul_mv_bf16_f32_1row")]] kernel mul_mv_1row_t kernel_mul_mv_1row<bfloat, bfloat4>;
#endif

// Assumes row size (ne00) is a multiple of 4
template<typename T, typename T4>
kernel void kernel_mul_mv_l4(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]]) {

    const int nrows = args.ne11;
    const int r0 = tgpig.x;
    const int im = tgpig.z;

    const uint i12 = im%args.ne12;
    const uint i13 = im/args.ne12;

    const uint64_t offset0 = r0*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;

    device const T4 * x4 = (device const T4 *) (src0 + offset0);

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1;

    for (int r1 = 0; r1 < nrows; ++r1) {
        const uint64_t offset1 = r1*args.nb11 + (i12   )*args.nb12 + (i13   )*args.nb13;

        device const float4 * y4 = (device const float4 *) (src1 + offset1);

        float sumf = 0;
        for (int i = tiisg; i < args.ne00/4; i += 32) {
            sumf += dot((float4) x4[i], y4[i]);
        }

        float all_sum = simd_sum(sumf);
        if (tiisg == 0) {
            dst_f32[(uint64_t)r1*args.ne0 + r0] = all_sum;
        }
    }
}

typedef decltype(kernel_mul_mv_l4<half, half4>) mul_mv_l4_t;

template [[host_name("kernel_mul_mv_f16_f32_l4")]]  kernel mul_mv_l4_t kernel_mul_mv_l4<half, half4>;
#if defined(GGML_METAL_USE_BF16)
template [[host_name("kernel_mul_mv_bf16_f32_l4")]] kernel mul_mv_l4_t kernel_mul_mv_l4<bfloat, bfloat4>;
#endif

static float rope_yarn_ramp(const float low, const float high, const int i0) {
    const float y = (i0 / 2 - low) / max(0.001f, high - low);
    return 1.0f - min(1.0f, max(0.0f, y));
}

// YaRN algorithm based on LlamaYaRNScaledRotaryEmbedding.py from https://github.com/jquesnelle/yarn
// MIT licensed. Copyright (c) 2023 Jeffrey Quesnelle and Bowen Peng.
static void rope_yarn(
    float theta_extrap, float freq_scale, float corr_dims[2], int i0, float ext_factor, float mscale,
    thread float * cos_theta, thread float * sin_theta) {
    // Get n-d rotational scaling corrected for extrapolation
    float theta_interp = freq_scale * theta_extrap;
    float theta = theta_interp;
    if (ext_factor != 0.0f) {
        float ramp_mix = rope_yarn_ramp(corr_dims[0], corr_dims[1], i0) * ext_factor;
        theta = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix;

        // Get n-d magnitude scaling corrected for interpolation
        mscale *= 1.0f + 0.1f * log(1.0f / freq_scale);
    }
    *cos_theta = cos(theta) * mscale;
    *sin_theta = sin(theta) * mscale;
}

// Apparently solving `n_rot = 2pi * x * base^((2 * max_pos_emb) / n_dims)` for x, we get
// `corr_fac(n_rot) = n_dims * log(max_pos_emb / (n_rot * 2pi)) / (2 * log(base))`
static float rope_yarn_corr_factor(int n_dims, int n_ctx_orig, float n_rot, float base) {
    return n_dims * log(n_ctx_orig / (n_rot * 2 * M_PI_F)) / (2 * log(base));
}

static void rope_yarn_corr_dims(
    int n_dims, int n_ctx_orig, float freq_base, float beta_fast, float beta_slow, float dims[2]
) {
    // start and end correction dims
    dims[0] = max(0.0f,         floor(rope_yarn_corr_factor(n_dims, n_ctx_orig, beta_fast, freq_base)));
    dims[1] = min(n_dims - 1.0f, ceil(rope_yarn_corr_factor(n_dims, n_ctx_orig, beta_slow, freq_base)));
}

template<typename T>
kernel void kernel_rope_norm(
        constant ggml_metal_kargs_rope & args,
        device const char * src0,
        device const char * src1,
        device const char * src2,
        device       char * dst,
        ushort  tiitg[[thread_index_in_threadgroup]],
        ushort3 tptg [[threads_per_threadgroup]],
        uint3   tgpig[[threadgroup_position_in_grid]]) {
    const int i3 = tgpig[2];
    const int i2 = tgpig[1];
    const int i1 = tgpig[0];

    float corr_dims[2];
    rope_yarn_corr_dims(args.n_dims, args.n_ctx_orig, args.freq_base, args.beta_fast, args.beta_slow, corr_dims);

    device const int32_t * pos = (device const int32_t *) src1;

    const float theta_base = (float) pos[i2];
    const float inv_ndims = -1.f/args.n_dims;

    float cos_theta;
    float sin_theta;

    for (int i0 = 2*tiitg; i0 < args.ne0; i0 += 2*tptg.x) {
        if (i0 < args.n_dims) {
            const int ic = i0/2;

            const float theta = theta_base * pow(args.freq_base, inv_ndims*i0);

            const float freq_factor = src2 != src0 ? ((device const float *) src2)[ic] : 1.0f;

            rope_yarn(theta/freq_factor, args.freq_scale, corr_dims, i0, args.ext_factor, args.attn_factor, &cos_theta, &sin_theta);

            device const T * const src = (device T *)(src0 + i3*args.nb03 + i2*args.nb02 + i1*args.nb01 + i0*args.nb00);
            device       T * dst_data  = (device T *)( dst + i3*args.nb3  + i2*args.nb2  + i1*args.nb1  + i0*args.nb0);

            const float x0 = src[0];
            const float x1 = src[1];

            dst_data[0] = x0*cos_theta - x1*sin_theta;
            dst_data[1] = x0*sin_theta + x1*cos_theta;
        } else {
            device const T * const src = (device T *)(src0 + i3*args.nb03 + i2*args.nb02 + i1*args.nb01 + i0*args.nb00);
            device       T * dst_data  = (device T *)( dst + i3*args.nb3  + i2*args.nb2  + i1*args.nb1  + i0*args.nb0);

            dst_data[0] = src[0];
            dst_data[1] = src[1];
        }
    }
}

template<typename T>
kernel void kernel_rope_neox(
        constant ggml_metal_kargs_rope & args,
        device const char * src0,
        device const char * src1,
        device const char * src2,
        device       char * dst,
        ushort  tiitg[[thread_index_in_threadgroup]],
        ushort3 tptg [[threads_per_threadgroup]],
        uint3   tgpig[[threadgroup_position_in_grid]]) {
    const int i3 = tgpig[2];
    const int i2 = tgpig[1];
    const int i1 = tgpig[0];

    float corr_dims[2];
    rope_yarn_corr_dims(args.n_dims, args.n_ctx_orig, args.freq_base, args.beta_fast, args.beta_slow, corr_dims);

    device const int32_t * pos = (device const int32_t *) src1;

    const float theta_base = (float) pos[i2];
    const float inv_ndims = -1.f/args.n_dims;

    float cos_theta;
    float sin_theta;

    for (int i0 = 2*tiitg; i0 < args.ne0; i0 += 2*tptg.x) {
        if (i0 < args.n_dims) {
            const int ic = i0/2;

            const float theta = theta_base * pow(args.freq_base, inv_ndims*i0);

            const float freq_factor = src2 != src0 ? ((device const float *) src2)[ic] : 1.0f;

            rope_yarn(theta/freq_factor, args.freq_scale, corr_dims, i0, args.ext_factor, args.attn_factor, &cos_theta, &sin_theta);

            device const T * const src = (device T *)(src0 + i3*args.nb03 + i2*args.nb02 + i1*args.nb01 + ic*args.nb00);
            device       T * dst_data  = (device T *)( dst + i3*args.nb3  + i2*args.nb2  + i1*args.nb1  + ic*args.nb0);

            const float x0 = src[0];
            const float x1 = src[args.n_dims/2];

            dst_data[0]             = x0*cos_theta - x1*sin_theta;
            dst_data[args.n_dims/2] = x0*sin_theta + x1*cos_theta;
        } else {
            device const T * const src = (device T *)(src0 + i3*args.nb03 + i2*args.nb02 + i1*args.nb01 + i0*args.nb00);
            device       T * dst_data  = (device T *)( dst + i3*args.nb3  + i2*args.nb2  + i1*args.nb1  + i0*args.nb0);

            dst_data[0] = src[0];
            dst_data[1] = src[1];
        }
    }
}

typedef decltype(kernel_rope_norm<float>) kernel_rope_norm_t;
typedef decltype(kernel_rope_neox<float>) kernel_rope_neox_t;

template [[host_name("kernel_rope_norm_f32")]] kernel kernel_rope_norm_t kernel_rope_norm<float>;
template [[host_name("kernel_rope_norm_f16")]] kernel kernel_rope_norm_t kernel_rope_norm<half>;

template [[host_name("kernel_rope_neox_f32")]] kernel kernel_rope_neox_t kernel_rope_neox<float>;
template [[host_name("kernel_rope_neox_f16")]] kernel kernel_rope_neox_t kernel_rope_neox<half>;

typedef void (im2col_t)(
        device const float * x,
        device        char * dst,
        constant   int32_t & ofs0,
        constant   int32_t & ofs1,
        constant   int32_t & IW,
        constant   int32_t & IH,
        constant   int32_t & CHW,
        constant   int32_t & s0,
        constant   int32_t & s1,
        constant   int32_t & p0,
        constant   int32_t & p1,
        constant   int32_t & d0,
        constant   int32_t & d1,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3  tgpg[[threadgroups_per_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint3   ntg[[threads_per_threadgroup]]);

template <typename T>
kernel void kernel_im2col(
        device const float * x,
        device        char * dst,
        constant   int32_t & ofs0,
        constant   int32_t & ofs1,
        constant   int32_t & IW,
        constant   int32_t & IH,
        constant   int32_t & CHW,
        constant   int32_t & s0,
        constant   int32_t & s1,
        constant   int32_t & p0,
        constant   int32_t & p1,
        constant   int32_t & d0,
        constant   int32_t & d1,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3  tgpg[[threadgroups_per_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint3   ntg[[threads_per_threadgroup]]) {
//    const int64_t IC = tgpg[0];
    const int64_t OH = tgpg[1];
    const int64_t OW = tgpg[2];

//    const int64_t N  = ntg[0];
    const int64_t KH = ntg[1];
    const int64_t KW = ntg[2];

    const int64_t in  = tpitg[0];
    const int64_t ikh = tpitg[1];
    const int64_t ikw = tpitg[2];

    const int64_t iic = tgpig[0];
    const int64_t ioh = tgpig[1];
    const int64_t iow = tgpig[2];

    const int64_t iiw = iow*s0 + ikw*d0 - p0;
    const int64_t iih = ioh*s1 + ikh*d1 - p1;

    const int64_t offset_dst = (in*OH*OW + ioh*OW + iow)*CHW + (iic*(KH*KW) + ikh*KW + ikw);

    device T * pdst = (device T *) (dst);

    if (iih < 0 || iih >= IH || iiw < 0 || iiw >= IW) {
        pdst[offset_dst] = 0.0f;
    } else {
        const int64_t offset_src = in*ofs0 + iic*ofs1 + iih*IW + iiw;
        pdst[offset_dst] = x[offset_src];
    }
}

template [[host_name("kernel_im2col_f32")]] kernel im2col_t kernel_im2col<float>;
template [[host_name("kernel_im2col_f16")]] kernel im2col_t kernel_im2col<half>;

typedef void (im2col_ext_t)(
        device const float * x,
        device        char * dst,
        constant   int32_t & ofs0,
        constant   int32_t & ofs1,
        constant   int32_t & IW,
        constant   int32_t & IH,
        constant   int32_t & CHW,
        constant   int32_t & s0,
        constant   int32_t & s1,
        constant   int32_t & p0,
        constant   int32_t & p1,
        constant   int32_t & d0,
        constant   int32_t & d1,
        constant   int32_t & N,
        constant   int32_t & KH,
        constant   int32_t & KW,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3  tgpg[[threadgroups_per_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint3   ntg[[threads_per_threadgroup]]);

template <typename T>
kernel void kernel_im2col_ext(
        device const float * x,
        device        char * dst,
        constant   int32_t & ofs0,
        constant   int32_t & ofs1,
        constant   int32_t & IW,
        constant   int32_t & IH,
        constant   int32_t & CHW,
        constant   int32_t & s0,
        constant   int32_t & s1,
        constant   int32_t & p0,
        constant   int32_t & p1,
        constant   int32_t & d0,
        constant   int32_t & d1,
        constant   int32_t & N,
        constant   int32_t & KH,
        constant   int32_t & KW,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3  tgpg[[threadgroups_per_grid]],      // tgpg[0] = D x IC x KH x KW, CHW = IC x KH x KW
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint3   ntg[[threads_per_threadgroup]]) {  // [M, 1, 1]
    const int64_t KHW = KH * KW;             // KHW == ntg[1] * ntg[2], KW == ntg[2]

    const int64_t d = tgpig[0] / CHW;
    const int64_t chw = tgpig[0] % CHW;
    const int64_t tgpig_0 = chw / KHW;  // 0 ~ (IC - 1)
    const int64_t HW = tgpig[0] % KHW;

    const int64_t tpitg_0 = (d * ntg[0]) + tpitg[0];
    if (tpitg_0 >= N) {
        return;
    }

    const int64_t tpitg_1 = HW / KW;
    const int64_t tpitg_2 = HW % KW;

    const int64_t iiw = tgpig[2] * s0 + tpitg_2 * d0 - p0;
    const int64_t iih = tgpig[1] * s1 + tpitg_1 * d1 - p1;

    const int64_t offset_dst =
        (tpitg_0 * tgpg[1] * tgpg[2] + tgpig[1] * tgpg[2] + tgpig[2]) * CHW +
        (tgpig_0 * KHW + tpitg_1 * KW + tpitg_2);

    device T * pdst = (device T *) (dst);

    if (iih < 0 || iih >= IH || iiw < 0 || iiw >= IW) {
        pdst[offset_dst] = 0.0f;
    } else {
        const int64_t offset_src = tpitg_0 * ofs0 + tgpig_0 * ofs1;
        pdst[offset_dst] = x[offset_src + iih * IW + iiw];
    }
}

template [[host_name("kernel_im2col_ext_f32")]] kernel im2col_ext_t kernel_im2col_ext<float>;
template [[host_name("kernel_im2col_ext_f16")]] kernel im2col_ext_t kernel_im2col_ext<half>;

typedef void (conv_transpose_1d_t)(
        device const float * src0,
        device const float * src1,
        device        char * dst,
        constant   int32_t & IC,
        constant   int32_t & IL,
        constant   int32_t & K,
        constant   int32_t & s0,
        constant  uint64_t & nb0,
        constant  uint64_t & nb1,
        uint3   tgpig[[threadgroup_position_in_grid]],
        uint3    tgpg[[threadgroups_per_grid]]);

template <typename T>
kernel void kernel_conv_transpose_1d(
        device const     T * src0,
        device const float * src1,
        device        char * dst,
        constant   int32_t & IC,
        constant   int32_t & IL,
        constant   int32_t & K,
        constant   int32_t & s0,
        constant  uint64_t & nb0,
        constant  uint64_t & nb1,
        uint3   tgpig[[threadgroup_position_in_grid]],
        uint3   tgpg[[threadgroups_per_grid]]) {

    float v = 0.0f;

    for (int64_t c = 0; c < IC; c++) {
        const int32_t kernel_offset = c * tgpg[1] * K + K * tgpig[1];
        const int32_t input_offset = c * IL;

        for (int64_t i = 0; i < IL; i++) {
            if (tgpig[0] >= i * s0 && tgpig[0] < i * s0 + K) {
                v += src0[kernel_offset + tgpig[0] - i * s0] * src1[input_offset + i];
            }
        }
    }

    device float * dst_ptr = (device float *) (dst + tgpig[0] * nb0 + tgpig[1] * nb1);

    dst_ptr[0] = v;
}

template [[host_name("kernel_conv_transpose_1d_f32_f32")]]
kernel void kernel_conv_transpose_1d<float>(
    device const float * src0,
    device const float * src1,
    device        char * dst,
    constant   int32_t & IC,
    constant   int32_t & IL,
    constant   int32_t & K,
    constant   int32_t & s0,
    constant  uint64_t & nb0,
    constant  uint64_t & nb1,
    uint3   tgpig[[threadgroup_position_in_grid]],
    uint3    tgpg[[threadgroups_per_grid]]);

template [[host_name("kernel_conv_transpose_1d_f16_f32")]]
kernel void kernel_conv_transpose_1d<half>(
    device const half  * src0,
    device const float * src1,
    device        char * dst,
    constant   int32_t & IC,
    constant   int32_t & IL,
    constant   int32_t & K,
    constant   int32_t & s0,
    constant  uint64_t & nb0,
    constant  uint64_t & nb1,
    uint3   tgpig[[threadgroup_position_in_grid]],
    uint3    tgpg[[threadgroups_per_grid]]);

kernel void kernel_upscale_f32(
    device  const char * src0,
    device        char * dst,
    constant   int64_t & ne00,
    constant   int64_t & ne01,
    constant   int64_t & ne02,
    constant   int64_t & ne03,
    constant  uint64_t & nb00,
    constant  uint64_t & nb01,
    constant  uint64_t & nb02,
    constant  uint64_t & nb03,
    constant   int64_t & ne0,
    constant   int64_t & ne1,
    constant   int64_t & ne2,
    constant   int64_t & ne3,
    constant  uint64_t & nb0,
    constant  uint64_t & nb1,
    constant  uint64_t & nb2,
    constant  uint64_t & nb3,
    constant     float & sf0,
    constant     float & sf1,
    constant     float & sf2,
    constant     float & sf3,
    uint3 tgpig[[threadgroup_position_in_grid]],
    uint3 tpitg[[thread_position_in_threadgroup]],
    uint3   ntg[[threads_per_threadgroup]]) {

    const int64_t i3 = tgpig.z;
    const int64_t i2 = tgpig.y;
    const int64_t i1 = tgpig.x;

    const int64_t i03 = i3/sf3;
    const int64_t i02 = i2/sf2;
    const int64_t i01 = i1/sf1;

    for (int i0 = tpitg.x; i0 < ne0; i0 += ntg.x) {
        const int64_t i00 = i0/sf0;

        device const float * src0_ptr = (device const float *) (src0 + i03*nb03 + i02*nb02 + i01*nb01 + i00*nb00);
        device       float * dst_ptr  = (device       float *) (dst  +  i3*nb3  +  i2*nb2  +  i1*nb1  +  i0*nb0);

        dst_ptr[0] = src0_ptr[0];
    }
}

kernel void kernel_pad_f32(
    device  const char * src0,
    device        char * dst,
    constant   int64_t & ne00,
    constant   int64_t & ne01,
    constant   int64_t & ne02,
    constant   int64_t & ne03,
    constant  uint64_t & nb00,
    constant  uint64_t & nb01,
    constant  uint64_t & nb02,
    constant  uint64_t & nb03,
    constant   int64_t & ne0,
    constant   int64_t & ne1,
    constant   int64_t & ne2,
    constant   int64_t & ne3,
    constant  uint64_t & nb0,
    constant  uint64_t & nb1,
    constant  uint64_t & nb2,
    constant  uint64_t & nb3,
    uint3 tgpig[[threadgroup_position_in_grid]],
    uint3 tpitg[[thread_position_in_threadgroup]],
    uint3   ntg[[threads_per_threadgroup]]) {

    const int64_t i3 = tgpig.z;
    const int64_t i2 = tgpig.y;
    const int64_t i1 = tgpig.x;

    const int64_t i03 = i3;
    const int64_t i02 = i2;
    const int64_t i01 = i1;

    device const float * src0_ptr = (device const float *) (src0 + i03*nb03 + i02*nb02 + i01*nb01);
    device       float * dst_ptr  = (device       float *) (dst  +  i3*nb3  +  i2*nb2  +  i1*nb1);

    if (i1 < ne01 && i2 < ne02 && i3 < ne03) {
        for (int i0 = tpitg.x; i0 < ne0; i0 += ntg.x) {
            if (i0 < ne00) {
                dst_ptr[i0] = src0_ptr[i0];
            } else {
                dst_ptr[i0] = 0.0f;
            }
        }

        return;
    }

    for (int i0 = tpitg.x; i0 < ne0; i0 += ntg.x) {
        dst_ptr[i0] = 0.0f;
    }
}

kernel void kernel_pad_reflect_1d_f32(
    device  const char * src0,
    device        char * dst,
    constant   int64_t & ne00,
    constant   int64_t & ne01,
    constant   int64_t & ne02,
    constant   int64_t & ne03,
    constant   int64_t & ne0,
    constant  uint64_t & nb00,
    constant  uint64_t & nb01,
    constant  uint64_t & nb02,
    constant  uint64_t & nb03,
    constant  uint64_t & nb0,
    constant  uint64_t & nb1,
    constant  uint64_t & nb2,
    constant  uint64_t & nb3,
    constant   int32_t & p0,
    constant   int32_t & p1,
    uint3 tgpig[[threadgroup_position_in_grid]],
    uint3  tgpg[[threadgroups_per_grid]],
    uint3 tpitg[[thread_position_in_threadgroup]],
    uint3   ntg[[threads_per_threadgroup]]) {

    const int64_t i3 = tgpig.z;
    const int64_t i2 = tgpig.y;
    const int64_t i1 = tgpig.x;

    const int64_t i03 = i3;
    const int64_t i02 = i2;
    const int64_t i01 = i1;

    device const float * src0_ptr = (device const float *) (src0 + i03*nb03 + i02*nb02 + i01*nb01);
    device       float * dst_ptr  = (device       float *) (dst  +  i3*nb3  +  i2*nb2  +  i1*nb1);

    if (i1 < ne01 && i2 < ne02 && i3 < ne03) {
        for (int i0 = tpitg.x; i0 < ne0; i0 += ntg.x) {
            if (i0 < p0) {
                dst_ptr[i0] = src0_ptr[p0 - i0];
            } else if (i0 < ne0 - p1) {
                dst_ptr[i0] = src0_ptr[i0 - p0];
            } else {
                dst_ptr[i0] = src0_ptr[(ne0 - p1 - p0) - (p1 + 1 - (ne0 - i0)) - 1];
            }
        }
    }
}

kernel void kernel_unpad_f32(
    device  const char * src0,
    device        char * dst,
    constant   int64_t & ne00,
    constant   int64_t & ne01,
    constant   int64_t & ne02,
    constant   int64_t & ne03,
    constant  uint64_t & nb00,
    constant  uint64_t & nb01,
    constant  uint64_t & nb02,
    constant  uint64_t & nb03,
    constant   int64_t & ne0,
    constant   int64_t & ne1,
    constant   int64_t & ne2,
    constant   int64_t & ne3,
    constant  uint64_t & nb0,
    constant  uint64_t & nb1,
    constant  uint64_t & nb2,
    constant  uint64_t & nb3,
    uint3 tgpig[[threadgroup_position_in_grid]],
    uint3 tpitg[[thread_position_in_threadgroup]],
    uint3   ntg[[threads_per_threadgroup]]) {

    const int64_t i3 = tgpig.z;
    const int64_t i2 = tgpig.y;
    const int64_t i1 = tgpig.x;

    const int64_t i03 = i3;
    const int64_t i02 = i2;
    const int64_t i01 = i1;

    device const float * src0_ptr = (device const float *) (src0 + i03*nb03 + i02*nb02 + i01*nb01);
    device       float * dst_ptr  = (device       float *) (dst  +  i3*nb3  +  i2*nb2  +  i1*nb1);

    if (i1 < ne01 && i2 < ne02 && i3 < ne03) {
        for (int i0 = tpitg.x; i0 < ne0; i0 += ntg.x) {
            if (i0 < ne00) {
                dst_ptr[i0] = src0_ptr[i0];
            }
        }

        return;
    }
}

kernel void kernel_arange_f32(
    device        char * dst,
    constant   int64_t & ne0,
    constant   float   & start,
    constant   float   & step,
    uint3 tgpig[[threadgroup_position_in_grid]],
    uint3 tpitg[[thread_position_in_threadgroup]],
    uint3   ntg[[threads_per_threadgroup]]) {

    device float * dst_ptr = (device float *) dst;

    for (int i0 = tpitg.x; i0 < ne0; i0 += ntg.x) {
        dst_ptr[i0] = start + step * i0;
    }
}

kernel void kernel_timestep_embedding_f32(
    device  const char * src0,
    device        char * dst,
    constant  uint64_t & nb1,
    constant  int      & dim,
    constant  int      & max_period,
    uint3 tgpig[[threadgroup_position_in_grid]],
    uint3 tpitg[[thread_position_in_threadgroup]],
    uint3   ntg[[threads_per_threadgroup]]) {

    int i = tgpig.x;
    device float * embed_data = (device float *)(dst +  i*nb1);

    int half_ = dim / 2;
    for (int j = tpitg.x; j < half_; j += ntg.x) {
        float timestep = ((device float *)src0)[i];
        float freq = (float)exp(-log((float)max_period) * j / half_);
        float arg = timestep * freq;
        embed_data[j        ] = cos(arg);
        embed_data[j + half_] = sin(arg);
    }

    if (dim % 2 != 0 && tpitg.x == 0) {
        embed_data[dim] = 0.f;
    }
}

// bitonic sort implementation following the CUDA kernels as reference
typedef void (argsort_t)(
        device const float  * x,
        device     int32_t  * dst,
        constant   int64_t  & ncols,
        constant   int64_t  & ncols_pad,
        threadgroup int32_t * shared_values [[threadgroup(0)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]]);

template<ggml_sort_order order>
kernel void kernel_argsort_f32_i32(
        device const float   * x,
        device       int32_t * dst,
        constant     int64_t & ncols,
        constant     int64_t & ncols_pad,
        threadgroup int32_t  * shared_values [[threadgroup(0)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]]) {
    // bitonic sort
    int col = tpitg[0];
    int row = tgpig[1];

    if (col >= ncols_pad) return;

    device const float   * x_row   = x + row * ncols;
    threadgroup int32_t  * dst_row = shared_values;

    // initialize indices
    dst_row[col] = col;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int k = 2; k <= ncols_pad; k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
            int ixj = col ^ j;
            if (ixj > col) {
                if ((col & k) == 0) {
                    if (dst_row[col] >= ncols ||
                        (dst_row[ixj] < ncols && (order == GGML_SORT_ORDER_ASC ?
                            x_row[dst_row[col]] > x_row[dst_row[ixj]] :
                            x_row[dst_row[col]] < x_row[dst_row[ixj]]))
                    ) {
                        SWAP(dst_row[col], dst_row[ixj]);
                    }
                } else {
                    if (dst_row[ixj] >= ncols ||
                        (dst_row[col] < ncols && (order == GGML_SORT_ORDER_ASC ?
                            x_row[dst_row[col]] < x_row[dst_row[ixj]] :
                            x_row[dst_row[col]] > x_row[dst_row[ixj]]))
                    ) {
                        SWAP(dst_row[col], dst_row[ixj]);
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    // copy the result to dst without the padding
    if (col < ncols) {
        dst[row * ncols + col] = dst_row[col];
    }
}

template [[host_name("kernel_argsort_f32_i32_asc")]]  kernel argsort_t kernel_argsort_f32_i32<GGML_SORT_ORDER_ASC>;
template [[host_name("kernel_argsort_f32_i32_desc")]] kernel argsort_t kernel_argsort_f32_i32<GGML_SORT_ORDER_DESC>;

kernel void kernel_leaky_relu_f32(
        device const float * src0,
        device       float * dst,
        constant     float & slope,
        uint tpig[[thread_position_in_grid]]) {
    dst[tpig] = src0[tpig] > 0.0f ? src0[tpig] : src0[tpig] * slope;
}

// ref: https://arxiv.org/pdf/2307.08691.pdf
template<
    typename q_t,     // query types in shared memory
    typename q4_t,
    typename q8x8_t,
    typename k_t,     // key types in shared memory
    typename k4x4_t,
    typename k8x8_t,
    typename v_t,     // value types in shared memory
    typename v4x4_t,
    typename v8x8_t,
    typename qk_t,    // Q*K types
    typename qk8x8_t,
    typename s_t,     // soft-max types
    typename s8x8_t,
    typename o_t,     // attention accumulation types
    typename o4_t,
    typename o8x8_t,
    typename kd4x4_t, // key type in device memory
    short nl_k,
    void (*deq_k)(device const kd4x4_t *, short, thread k4x4_t &),
    typename vd4x4_t, // key type in device memory
    short nl_v,
    void (*deq_v)(device const vd4x4_t *, short, thread v4x4_t &),
    short D,         // head size
    short Q  = 8,    // queries per threadgroup
    short KV = 8,    // key/value processed per each simdgroup
    short C  = 32>   // cache items per threadgroup
kernel void kernel_flash_attn_ext(
        constant ggml_metal_kargs_flash_attn_ext & args,
        device const char * q,
        device const char * k,
        device const char * v,
        device const char * mask,
        device       char * dst,
        threadgroup  half * shmem_f16 [[threadgroup(0)]],
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3   ntg[[threads_per_threadgroup]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]]) {
    const short nsg = ntg.y; // number of simdgroups

    const int iq3 = tgpig[2];
    const int iq2 = tgpig[1];
    const int iq1 = tgpig[0]*Q;

    const short D4  = D/4;
    const short D8  = D/8;
    const short D16 = D/16;
    const short NW  = N_SIMDWIDTH;
    const short SH  = (2*C + Q); // shared memory per simdgroup (s_t == float)

    const short TS = nsg*SH;   // shared memory size per query in (s_t == float)
    const short T  = D + 2*TS; // shared memory size per query in (half)

    threadgroup q_t  * sq  = (threadgroup q_t  *) (shmem_f16 +              0*D); // holds the query data
    threadgroup q4_t * sq4 = (threadgroup q4_t *) (shmem_f16 +              0*D); // same as above but in q4_t
    threadgroup o_t  * so  = (threadgroup o_t  *) (shmem_f16 +              0*D); // reuse query data for accumulation
    threadgroup o4_t * so4 = (threadgroup o4_t *) (shmem_f16 +              0*D); // same as above but in o4_t
    threadgroup s_t  * ss  = (threadgroup s_t  *) (shmem_f16 + 2*sgitg*SH + Q*D); // scratch buffer for attention, mask and diagonal matrix

    threadgroup k_t    * sk    = (threadgroup k_t    *) (shmem_f16 + sgitg*(4*16*KV) + Q*T); // scratch buffer to load K in shared memory
    threadgroup k4x4_t * sk4x4 = (threadgroup k4x4_t *) (shmem_f16 + sgitg*(4*16*KV) + Q*T); // same as above but in k4x4_t

    threadgroup v_t    * sv    = (threadgroup v_t    *) (shmem_f16 + sgitg*(4*16*KV) + Q*T); // scratch buffer to load V in shared memory
    threadgroup v4x4_t * sv4x4 = (threadgroup v4x4_t *) (shmem_f16 + sgitg*(4*16*KV) + Q*T); // same as above but in v4x4_t

    // store the result for all queries in local memory in 8x8 matrices (the O matrix from the paper)
    o8x8_t lo[D8];

    // load heads from Q to shared memory
    for (short j = sgitg; j < Q; j += nsg) {
        device const float4 * q4 = (device const float4 *) ((device const char *) q + ((iq1 + j)*args.nb01 + iq2*args.nb02 + iq3*args.nb03));

        for (short i = tiisg; i < D4; i += NW) {
            if (iq1 + j < args.ne01) {
                sq4[j*D4 + i] = (q4_t) q4[i];
            } else {
                sq4[j*D4 + i] = (q4_t) 0.0f;
            }
        }
    }

    // zero out lo
    for (short i = 0; i < D8; ++i) {
        lo[i] = make_filled_simdgroup_matrix<o_t, 8>((o_t) 0.0f);
    }

    // zero out shared memory SH
    for (short j = 0; j < Q; ++j) {
        for (short i = tiisg; i < SH; i += NW) {
            ss[j*TS + i] = 0.0f;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    {
        half S[Q] = { [0 ... Q-1] = 0.0f };
        half M[Q] = { [0 ... Q-1] = -__FLT16_MAX__/2 };

        // thread indices inside the simdgroup
        // TODO: see if we can utilize quad-group functions for better performance
        //       https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf (6.9.3)
        const short tx = tiisg%4;
        const short ty = tiisg/4;

        // broadcast kv
        //const short rk2 = args.ne02/args.ne12;
        //const short rk3 = args.ne03/args.ne13;

        const short ikv2 = iq2/(args.ne02/args.ne_12_2);
        const short ikv3 = iq3/(args.ne03/args.ne_12_3);

        // load the queries from shared memory into local memory
        q8x8_t mq[D8];

        for (short i = 0; i < D8; ++i) {
            simdgroup_load(mq[i], sq + i*8, D);
        }

        const bool has_mask = mask != q;

        half slope = 1.0f;

        // ALiBi
        if (args.max_bias > 0.0f) {
            const short h = iq2;

            const half  base = h < args.n_head_log2 ? args.m0 : args.m1;
            const short exph = h < args.n_head_log2 ? h + 1 : 2*(h - args.n_head_log2) + 1;

            slope = pow(base, exph);
        }

        // loop over the KV cache
        // each simdgroup handles blocks of Q rows and C columns
        for (int ic0 = 0; ic0 < args.ne11; ic0 += C*nsg) {
            const int ic = ic0 + C*sgitg;
            if (ic >= args.ne11) {
                break;
            }

            if (has_mask) {
                // used to detect blocks full of -INF
                half smax = -INFINITY;

                // load the mask in shared memory
                #pragma unroll(Q)
                for (short j = 0; j < Q; ++j) {
                    device const half * pm = (device const half *) ((device const char *) mask + (iq1 + j)*args.nb31);

                    const half m = pm[ic + tiisg];

                    ss[j*TS + C + tiisg] = m;
                    smax = max(smax, m);
                }

                smax = simd_max(smax);

                if (smax == -INFINITY) {
                    continue;
                }
            }

            // Q*K^T
            {
                for (short cc = 0; cc < C/8; ++cc) {
                    qk8x8_t mqk = make_filled_simdgroup_matrix<qk_t, 8>((qk_t) 0.0f);

                    // this is compile-time check, so it does not have runtime overhead
                    if (is_same<kd4x4_t, k4x4_t>::value) {
                        // we can read directly from global memory
                        device const k_t * pk = (device const k_t *) ((device const char *) k + ((ic + 8*cc)*args.nb_12_1 + ikv2*args.nb_12_2 + ikv3*args.nb_12_3));

                        #pragma unroll(D8)
                        for (short i = 0; i < D8; ++i) {
                            k8x8_t mk;
                            simdgroup_load(mk, pk + i*8, args.nb_12_1/sizeof(k_t), 0, true); // transpose // TODO: use ne10

                            simdgroup_multiply_accumulate(mqk, mq[i], mk, mqk);
                        }
                    } else {
                        for (short ii = 0; ii < D16; ii += 4) {
                            device const kd4x4_t * pk4x4 = (device const kd4x4_t *) ((device const char *) k + ((ic + 8*cc + ty)*args.nb_12_1 + ikv2*args.nb_12_2 + ikv3*args.nb_12_3));

                            if (D16%4 == 0) {
                                // the head is evenly divisible by 4*16 = 64, so no need for bound checks
                                {
                                    k4x4_t tmp;
                                    deq_k(pk4x4 + (ii + tx)/nl_k, (ii + tx)%nl_k, tmp);
                                    sk4x4[4*ty + tx] = tmp;
                                }

                                simdgroup_barrier(mem_flags::mem_threadgroup);

                                #pragma unroll(4)
                                for (short k = 0; k < 4; ++k) {
                                    k8x8_t mk;

                                    simdgroup_load(mk, sk + 16*k + 0*8, 4*16, 0, true); // transpose
                                    simdgroup_multiply_accumulate(mqk, mq[2*(ii + k) + 0], mk, mqk);

                                    simdgroup_load(mk, sk + 16*k + 1*8, 4*16, 0, true); // transpose
                                    simdgroup_multiply_accumulate(mqk, mq[2*(ii + k) + 1], mk, mqk);
                                }
                            } else {
                                if (ii + tx < D16) {
                                    k4x4_t tmp;
                                    deq_k(pk4x4 + (ii + tx)/nl_k, (ii + tx)%nl_k, tmp);
                                    sk4x4[4*ty + tx] = tmp;
                                }

                                simdgroup_barrier(mem_flags::mem_threadgroup);

                                for (short k = 0; k < 4 && ii + k < D16; ++k) {
                                    k8x8_t mk;

                                    simdgroup_load(mk, sk + 16*k + 0*8, 4*16, 0, true); // transpose
                                    simdgroup_multiply_accumulate(mqk, mq[2*(ii + k) + 0], mk, mqk);

                                    simdgroup_load(mk, sk + 16*k + 1*8, 4*16, 0, true); // transpose
                                    simdgroup_multiply_accumulate(mqk, mq[2*(ii + k) + 1], mk, mqk);
                                }
                            }
                        }
                    }

                    // cast qk_t -> s_t
                    //s8x8_t mqks(1.0f);
                    //simdgroup_multiply(mqks, mqk, mqks);
                    //simdgroup_store(mqks, ss + 8*cc, TS, 0, false);

                    simdgroup_store(mqk, ss + 8*cc, TS, 0, false);
                }
            }

            // online softmax
            {
                for (ushort j = 0; j < Q; ++j) {
                    const half m = M[j];

                    // scale and apply the logitcap / mask
                    half s = ss[j*TS + tiisg]*args.scale;

                    if (args.logit_softcap != 0.0f) {
                        s = args.logit_softcap*precise::tanh(s);
                    }

                    // mqk = mqk + mask*slope
                    s += slope*ss[j*TS + C + tiisg];

                    M[j] = simd_max(max(M[j], s));

                    const half ms = exp(m - M[j]);
                    const half vs = exp(s - M[j]);

                    S[j] = S[j]*ms + simd_sum(vs);

                    // the P matrix from the paper (Q rows, C columns)
                    ss[j*TS + tiisg] = vs;

                    // create a QxQ diagonal matrix for rescaling the output
                    if (tiisg == j) {
                        ss[j*TS + 2*C + j] = ms;
                    }
                }
            }

            // O = diag(ms)*O
            {
                s8x8_t mm;
                simdgroup_load(mm, ss + 2*C, TS, 0, false);

                #pragma unroll(D8)
                for (short i = 0; i < D8; ++i) {
                    simdgroup_multiply(lo[i], mm, lo[i]);
                }
            }

            // O = O + (Q*K^T)*V
            {
                for (short cc = 0; cc < C/8; ++cc) {
                    s8x8_t ms;
                    simdgroup_load(ms, ss + 8*cc, TS, 0, false);

                    if (is_same<vd4x4_t, v4x4_t>::value) {
                        // we can read directly from global memory
                        device const v_t * pv = (device const v_t *) ((device const char *) v + ((ic + 8*cc)*args.nb_12_1 + ikv2*args.nb_12_2 + ikv3*args.nb_12_3));

                        #pragma unroll(D8)
                        for (short i = 0; i < D8; ++i) {
                            v8x8_t mv;
                            simdgroup_load(mv, pv + i*8, args.nb_12_1/sizeof(v_t), 0, false); // TODO: use ne20

                            simdgroup_multiply_accumulate(lo[i], ms, mv, lo[i]);
                        }
                    } else {
                        for (short ii = 0; ii < D16; ii += 4) {
                            device const vd4x4_t * pv4x4 = (device const vd4x4_t *) ((device const char *) v + ((ic + 8*cc + ty)*args.nb_12_1 + ikv2*args.nb_12_2 + ikv3*args.nb_12_3));

                            if (D16%4 == 0) {
                                // no need for bound checks
                                {
                                    v4x4_t tmp;
                                    deq_v(pv4x4 + (ii + tx)/nl_v, (ii + tx)%nl_v, tmp);
                                    sv4x4[4*ty + tx] = tmp;
                                }

                                simdgroup_barrier(mem_flags::mem_threadgroup);

                                #pragma unroll(4)
                                for (short k = 0; k < 4; ++k) {
                                    v8x8_t mv;

                                    simdgroup_load(mv, sv + 16*k + 0*8, 4*16, 0, false);
                                    simdgroup_multiply_accumulate(lo[2*(ii + k) + 0], ms, mv, lo[2*(ii + k) + 0]);

                                    simdgroup_load(mv, sv + 16*k + 1*8, 4*16, 0, false);
                                    simdgroup_multiply_accumulate(lo[2*(ii + k) + 1], ms, mv, lo[2*(ii + k) + 1]);
                                }
                            } else {
                                if (ii + tx < D16) {
                                    v4x4_t tmp;
                                    deq_v(pv4x4 + (ii + tx)/nl_v, (ii + tx)%nl_v, tmp);
                                    sv4x4[4*ty + tx] = tmp;
                                }

                                simdgroup_barrier(mem_flags::mem_threadgroup);

                                for (short k = 0; k < 4 && ii + k < D16; ++k) {
                                    v8x8_t mv;

                                    simdgroup_load(mv, sv + 16*k + 0*8, 4*16, 0, false);
                                    simdgroup_multiply_accumulate(lo[2*(ii + k) + 0], ms, mv, lo[2*(ii + k) + 0]);

                                    simdgroup_load(mv, sv + 16*k + 1*8, 4*16, 0, false);
                                    simdgroup_multiply_accumulate(lo[2*(ii + k) + 1], ms, mv, lo[2*(ii + k) + 1]);
                                }
                            }
                        }
                    }
                }
            }
        }

        // these are needed for reducing the results from the simdgroups (reuse the ss buffer)
        for (short j = 0; j < Q; ++j) {
            if (tiisg == 0) {
                ss[j*TS + 0] = S[j];
                ss[j*TS + 1] = M[j];
            }
        }
    }

    // reduce the warps sequentially
    for (ushort sg = 1; sg < nsg; ++sg) {
        half S = { 0.0f };
        half M = { -__FLT16_MAX__/2 };

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // each simdgroup stores its output to shared memory, reusing sq
        if (sgitg == sg) {
            for (short i = 0; i < D8; ++i) {
                simdgroup_store(lo[i], so + i*8, D, 0, false);
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // the first simdgroup accumulates the results from the other simdgroups
        if (sgitg == 0) {
            for (short j = 0; j < Q; ++j) {
                const half S0 = ss[j*TS +         0];
                const half S1 = ss[j*TS + sg*SH + 0];

                const half M0 = ss[j*TS +         1];
                const half M1 = ss[j*TS + sg*SH + 1];

                M = max(M0, M1);

                const half ms0 = exp(M0 - M);
                const half ms1 = exp(M1 - M);

                S = S0*ms0 + S1*ms1;

                if (tiisg == 0) {
                    ss[j*TS + 0] = S;
                    ss[j*TS + 1] = M;

                    ss[j*TS + 2*C + j        ] = ms0;
                    ss[j*TS + 2*C + j + sg*SH] = ms1;
                }
            }

            // O_0 = diag(ms0)*O_0 + diag(ms1)*O_1
            {
                s8x8_t ms0;
                s8x8_t ms1;

                simdgroup_load(ms0, ss + 2*C,         TS, 0, false);
                simdgroup_load(ms1, ss + 2*C + sg*SH, TS, 0, false);

                #pragma unroll(D8)
                for (short i = 0; i < D8; ++i) {
                    o8x8_t t;

                    simdgroup_load    (t, so + i*8, D, 0, false);
                    simdgroup_multiply(t, ms1, t);

                    simdgroup_multiply_accumulate(lo[i], ms0, lo[i], t);
                }
            }
        }
    }

    // store result to shared memory (reuse sq)
    if (sgitg == 0) {
        for (short i = 0; i < D8; ++i) {
            simdgroup_store(lo[i], so + i*8, D, 0, false);
        }
    }

    device float4 * dst4 = (device float4 *) dst;

    // final rescale with 1/S and store to global memory
    if (sgitg == 0) {
        for (short j = 0; j < Q && iq1 + j < args.ne01; ++j) {
            const float S = ss[j*TS + 0];

            for (short i = tiisg; i < D4; i += NW) {
                dst4[((uint64_t)iq3*args.ne2*args.ne1 + iq2 + (uint64_t)(iq1 + j)*args.ne1)*D4 + i] = (float4) so4[j*D4 + i]/S;
            }
        }
    }
}

// TODO: this is quite ugly. in the future these types will be hardcoded in the kernel, but for now keep them as
//       template to be able to explore different combinations
//
#define FA_TYPES \
    half,  half4,   simdgroup_half8x8,  \
    half,  half4x4, simdgroup_half8x8,  \
    half,  half4x4, simdgroup_half8x8,  \
    float,          simdgroup_float8x8, \
    float,          simdgroup_float8x8, \
    half,  half4,   simdgroup_half8x8

typedef decltype(kernel_flash_attn_ext<FA_TYPES, half4x4, 1, dequantize_f16, half4x4, 1, dequantize_f16, 64>) flash_attn_ext_t;

template [[host_name("kernel_flash_attn_ext_f16_h64" )]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, half4x4,    1, dequantize_f16,  half4x4,    1, dequantize_f16,  64>;
template [[host_name("kernel_flash_attn_ext_f16_h80" )]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, half4x4,    1, dequantize_f16,  half4x4,    1, dequantize_f16,  80>;
template [[host_name("kernel_flash_attn_ext_f16_h96" )]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, half4x4,    1, dequantize_f16,  half4x4,    1, dequantize_f16,  96>;
template [[host_name("kernel_flash_attn_ext_f16_h112")]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, half4x4,    1, dequantize_f16,  half4x4,    1, dequantize_f16,  112>;
template [[host_name("kernel_flash_attn_ext_f16_h128")]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, half4x4,    1, dequantize_f16,  half4x4,    1, dequantize_f16,  128>;
template [[host_name("kernel_flash_attn_ext_f16_h256")]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, half4x4,    1, dequantize_f16,  half4x4,    1, dequantize_f16,  256>;

#if defined(GGML_METAL_USE_BF16)
template [[host_name("kernel_flash_attn_ext_bf16_h64" )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, bfloat4x4,  1, dequantize_bf16, bfloat4x4,  1, dequantize_bf16, 64>;
template [[host_name("kernel_flash_attn_ext_bf16_h80" )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, bfloat4x4,  1, dequantize_bf16, bfloat4x4,  1, dequantize_bf16, 80>;
template [[host_name("kernel_flash_attn_ext_bf16_h96" )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, bfloat4x4,  1, dequantize_bf16, bfloat4x4,  1, dequantize_bf16, 96>;
template [[host_name("kernel_flash_attn_ext_bf16_h112")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, bfloat4x4,  1, dequantize_bf16, bfloat4x4,  1, dequantize_bf16, 112>;
template [[host_name("kernel_flash_attn_ext_bf16_h128")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, bfloat4x4,  1, dequantize_bf16, bfloat4x4,  1, dequantize_bf16, 128>;
template [[host_name("kernel_flash_attn_ext_bf16_h256")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, bfloat4x4,  1, dequantize_bf16, bfloat4x4,  1, dequantize_bf16, 256>;
#endif

template [[host_name("kernel_flash_attn_ext_q4_0_h64" )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_q4_0, 2, dequantize_q4_0, block_q4_0, 2, dequantize_q4_0, 64>;
template [[host_name("kernel_flash_attn_ext_q4_0_h80" )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_q4_0, 2, dequantize_q4_0, block_q4_0, 2, dequantize_q4_0, 80>;
template [[host_name("kernel_flash_attn_ext_q4_0_h96" )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_q4_0, 2, dequantize_q4_0, block_q4_0, 2, dequantize_q4_0, 96>;
template [[host_name("kernel_flash_attn_ext_q4_0_h112")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_q4_0, 2, dequantize_q4_0, block_q4_0, 2, dequantize_q4_0, 112>;
template [[host_name("kernel_flash_attn_ext_q4_0_h128")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_q4_0, 2, dequantize_q4_0, block_q4_0, 2, dequantize_q4_0, 128>;
template [[host_name("kernel_flash_attn_ext_q4_0_h256")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_q4_0, 2, dequantize_q4_0, block_q4_0, 2, dequantize_q4_0, 256>;

template [[host_name("kernel_flash_attn_ext_q4_1_h64" )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_q4_1, 2, dequantize_q4_1, block_q4_1, 2, dequantize_q4_1, 64>;
template [[host_name("kernel_flash_attn_ext_q4_1_h80" )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_q4_1, 2, dequantize_q4_1, block_q4_1, 2, dequantize_q4_1, 80>;
template [[host_name("kernel_flash_attn_ext_q4_1_h96" )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_q4_1, 2, dequantize_q4_1, block_q4_1, 2, dequantize_q4_1, 96>;
template [[host_name("kernel_flash_attn_ext_q4_1_h112")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_q4_1, 2, dequantize_q4_1, block_q4_1, 2, dequantize_q4_1, 112>;
template [[host_name("kernel_flash_attn_ext_q4_1_h128")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_q4_1, 2, dequantize_q4_1, block_q4_1, 2, dequantize_q4_1, 128>;
template [[host_name("kernel_flash_attn_ext_q4_1_h256")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_q4_1, 2, dequantize_q4_1, block_q4_1, 2, dequantize_q4_1, 256>;

template [[host_name("kernel_flash_attn_ext_q5_0_h64" )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_q5_0, 2, dequantize_q5_0, block_q5_0, 2, dequantize_q5_0, 64>;
template [[host_name("kernel_flash_attn_ext_q5_0_h80" )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_q5_0, 2, dequantize_q5_0, block_q5_0, 2, dequantize_q5_0, 80>;
template [[host_name("kernel_flash_attn_ext_q5_0_h96" )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_q5_0, 2, dequantize_q5_0, block_q5_0, 2, dequantize_q5_0, 96>;
template [[host_name("kernel_flash_attn_ext_q5_0_h112")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_q5_0, 2, dequantize_q5_0, block_q5_0, 2, dequantize_q5_0, 112>;
template [[host_name("kernel_flash_attn_ext_q5_0_h128")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_q5_0, 2, dequantize_q5_0, block_q5_0, 2, dequantize_q5_0, 128>;
template [[host_name("kernel_flash_attn_ext_q5_0_h256")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_q5_0, 2, dequantize_q5_0, block_q5_0, 2, dequantize_q5_0, 256>;

template [[host_name("kernel_flash_attn_ext_q5_1_h64" )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_q5_1, 2, dequantize_q5_1, block_q5_1, 2, dequantize_q5_1, 64>;
template [[host_name("kernel_flash_attn_ext_q5_1_h80" )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_q5_1, 2, dequantize_q5_1, block_q5_1, 2, dequantize_q5_1, 80>;
template [[host_name("kernel_flash_attn_ext_q5_1_h96" )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_q5_1, 2, dequantize_q5_1, block_q5_1, 2, dequantize_q5_1, 96>;
template [[host_name("kernel_flash_attn_ext_q5_1_h112")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_q5_1, 2, dequantize_q5_1, block_q5_1, 2, dequantize_q5_1, 112>;
template [[host_name("kernel_flash_attn_ext_q5_1_h128")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_q5_1, 2, dequantize_q5_1, block_q5_1, 2, dequantize_q5_1, 128>;
template [[host_name("kernel_flash_attn_ext_q5_1_h256")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_q5_1, 2, dequantize_q5_1, block_q5_1, 2, dequantize_q5_1, 256>;

template [[host_name("kernel_flash_attn_ext_q8_0_h64" )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_q8_0, 2, dequantize_q8_0, block_q8_0, 2, dequantize_q8_0, 64>;
template [[host_name("kernel_flash_attn_ext_q8_0_h80" )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_q8_0, 2, dequantize_q8_0, block_q8_0, 2, dequantize_q8_0, 80>;
template [[host_name("kernel_flash_attn_ext_q8_0_h96" )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_q8_0, 2, dequantize_q8_0, block_q8_0, 2, dequantize_q8_0, 96>;
template [[host_name("kernel_flash_attn_ext_q8_0_h112")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_q8_0, 2, dequantize_q8_0, block_q8_0, 2, dequantize_q8_0, 112>;
template [[host_name("kernel_flash_attn_ext_q8_0_h128")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_q8_0, 2, dequantize_q8_0, block_q8_0, 2, dequantize_q8_0, 128>;
template [[host_name("kernel_flash_attn_ext_q8_0_h256")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_q8_0, 2, dequantize_q8_0, block_q8_0, 2, dequantize_q8_0, 256>;

#undef FA_TYPES

template<
    typename q4_t,    // query types in shared memory
    typename q4x4_t,
    typename k4x4_t,  // key types in shared memory
    typename v4x4_t,  // value types in shared memory
    typename qk_t,    // Q*K types
    typename s_t,     // soft-max types
    typename s4_t,
    typename s4x4_t,
    typename o4x4_t,  // attention accumulation types
    typename kd4x4_t, // key type in device memory
    short nl_k,
    void (*deq_k)(device const kd4x4_t *, short, thread k4x4_t &),
    typename vd4x4_t, // key type in device memory
    short nl_v,
    void (*deq_v)(device const vd4x4_t *, short, thread v4x4_t &),
    short D,         // head size
    short Q  = 1,    // queries per threadgroup
    short C  = 32>   // cache items per threadgroup
kernel void kernel_flash_attn_ext_vec(
        constant ggml_metal_kargs_flash_attn_ext & args,
        device const char * q,
        device const char * k,
        device const char * v,
        device const char * mask,
        device       char * dst,
        threadgroup  half * shmem_f16 [[threadgroup(0)]],
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3   ntg[[threads_per_threadgroup]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]]) {
    const short nsg = ntg.y; // number of simdgroups

    const int iq3 = tgpig[2];
    const int iq2 = tgpig[1];
    const int iq1 = tgpig[0];

    const short D4  = D/4;
    const short D16 = D/16;
    const short NW  = N_SIMDWIDTH;
    const short NL  = NW/4; // note: this can be adjusted to support D%64 == 0 and D%32 == 0
    const short SH  = 2*C;  // shared memory per simdgroup

    const short T = D + nsg*SH; // shared memory size per query in (half)

  //threadgroup q_t    * sq    = (threadgroup q_t    *) (shmem_f16 +                0*D); // holds the query data
    threadgroup q4_t   * sq4   = (threadgroup q4_t   *) (shmem_f16 +                0*D); // same as above but in q4_t
    threadgroup q4x4_t * sq4x4 = (threadgroup q4x4_t *) (shmem_f16 +                0*D); // same as above but in q4x4_t
    threadgroup s_t    * ss    = (threadgroup s_t    *) (shmem_f16 + sgitg*SH     + Q*D); // scratch buffer for attention
    threadgroup s4_t   * ss4   = (threadgroup s4_t   *) (shmem_f16 + sgitg*SH     + Q*D); // same as above but in s4_t
    threadgroup half   * sm    = (threadgroup half   *) (shmem_f16 + sgitg*SH + C + Q*D); // scratch buffer for mask
    threadgroup o4x4_t * sr4x4 = (threadgroup o4x4_t *) (shmem_f16 + sgitg*D      + Q*T); // scratch buffer for the results

    // store the result for all queries in local memory in 8x8 matrices (the O matrix from the paper)
    o4x4_t lo[D16/NL];

    // load heads from Q to shared memory
    device const float4 * q4 = (device const float4 *) ((device const char *) q + (iq1*args.nb01 + iq2*args.nb02 + iq3*args.nb03));

    for (short i = tiisg; i < D4; i += NW) {
        if (iq1 < args.ne01) {
            sq4[i] = (q4_t) q4[i];
        } else {
            sq4[i] = (q4_t) 0.0f;
        }
    }

    // zero out lo
    for (short i = 0; i < D16/NL; ++i) {
        lo[i] = (o4x4_t) 0.0f;
    }

    // zero out shared memory SH
    for (short i = tiisg; i < SH/4; i += NW) {
        ss4[i] = (s4_t) 0.0f;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    {
        half S = 0.0f;
        half M = -__FLT16_MAX__/2;

        // thread indices inside the simdgroup
        const short tx = tiisg%NL;
        const short ty = tiisg/NL;

        // broadcast kv
        //const short rk2 = args.ne02/args.ne12;
        //const short rk3 = args.ne03/args.ne13;

        const short ikv2 = iq2/(args.ne02/args.ne_12_2);
        const short ikv3 = iq3/(args.ne03/args.ne_12_3);

        // load the queries from shared memory into local memory
        q4x4_t mq[D16/NL];

        #pragma unroll(D16/NL)
        for (short ii = 0; ii < D16; ii += NL) {
            mq[ii/NL] = sq4x4[ii + tx];
        }

        const bool has_mask = mask != q;

        // pointer to the mask
        device const half * pm = (device const half *) (mask + iq1*args.nb31);

        half slope = 1.0f;

        // ALiBi
        if (args.max_bias > 0.0f) {
            const short h = iq2;

            const half  base = h < args.n_head_log2 ? args.m0 : args.m1;
            const short exph = h < args.n_head_log2 ? h + 1 : 2*(h - args.n_head_log2) + 1;

            slope = pow(base, exph);
        }

        // loop over the KV cache
        // each simdgroup handles blocks of Q rows and C columns
        for (int ic0 = 0; ic0 < args.ne11; ic0 += C*nsg) {
            const int ic = ic0 + C*sgitg;
            if (ic >= args.ne11) {
                break;
            }

            if (has_mask) {
                sm[tiisg] = pm[ic + tiisg];
            }

            // Q*K^T
            {
                // each simdgroup processes 1 query and 4 (NW/NL) keys
                for (short cc = 0; cc < C/4; ++cc) {
                    qk_t mqka[4] = { 0.0, 0.0, 0.0, 0.0 };

                    device const kd4x4_t * pk = (device const kd4x4_t *) ((device const char *) k + ((ic + 4*cc + ty)*args.nb_12_1 + ikv2*args.nb_12_2 + ikv3*args.nb_12_3));

                    #pragma unroll(D16/NL)
                    for (short ii = 0; ii < D16; ii += NL) {
                        const short i = ii + tx;

                        k4x4_t mk;
                        deq_k(pk + i/nl_k, i%nl_k, mk);

                        // note: this is less precise than the version below
                        //mqka[0] += dot(mq[ii/NL][0], mk[0]);
                        //mqka[1] += dot(mq[ii/NL][1], mk[1]);
                        //mqka[2] += dot(mq[ii/NL][2], mk[2]);
                        //mqka[3] += dot(mq[ii/NL][3], mk[3]);

                        mqka[0] += dot((float4) mq[ii/NL][0], (float4) mk[0]);
                        mqka[1] += dot((float4) mq[ii/NL][1], (float4) mk[1]);
                        mqka[2] += dot((float4) mq[ii/NL][2], (float4) mk[2]);
                        mqka[3] += dot((float4) mq[ii/NL][3], (float4) mk[3]);
                    }

                    qk_t mqk = mqka[0] + mqka[1] + mqka[2] + mqka[3];

                    // simdgroup reduce
                    // [ 0 ..  7] -> [ 0]
                    // [ 8 .. 15] -> [ 8]
                    // [16 .. 23] -> [16]
                    // [24 .. 31] -> [24]
                  //mqk += simd_shuffle_down(mqk, 16);
                  //mqk += simd_shuffle_down(mqk,  8);
                    mqk += simd_shuffle_down(mqk,  4);
                    mqk += simd_shuffle_down(mqk,  2);
                    mqk += simd_shuffle_down(mqk,  1);

                    // mqk = mqk*scale + mask*slope
                    if (tx == 0) {
                        mqk *= args.scale;

                        if (args.logit_softcap != 0.0f) {
                            mqk = args.logit_softcap*precise::tanh(mqk);
                        }

                        mqk += sm[4*cc + ty]*slope;

                        ss[4*cc + ty] = mqk;
                    }
                }
            }

            simdgroup_barrier(mem_flags::mem_threadgroup);

            // online softmax
            {
                const half m = M;
                const half s = ss[tiisg];

                M = simd_max(max(M, s));

                const half ms = exp(m - M);
                const half vs = exp(s - M);

                S = S*ms + simd_sum(vs);

                // the P matrix from the paper (Q rows, C columns)
                ss[tiisg] = vs;

                // O = diag(ms)*O
                #pragma unroll(D16/NL)
                for (short ii = 0; ii < D16; ii += NL) {
                    lo[ii/NL] *= ms;
                }
            }

            simdgroup_barrier(mem_flags::mem_threadgroup);

            // O = O + (Q*K^T)*V
            {
                for (short cc = 0; cc < C/4; ++cc) {
                    device const vd4x4_t * pv4 = (device const vd4x4_t *) ((device const char *) v + ((ic + 4*cc + ty)*args.nb_12_1 + ikv2*args.nb_12_2 + ikv3*args.nb_12_3));

                    const s4x4_t ms(ss[4*cc + ty]);

                    #pragma unroll(D16/NL)
                    for (short ii = 0; ii < D16; ii += NL) {
                        const short i = ii + tx;

                        v4x4_t mv;
                        deq_v(pv4 + i/nl_v, i%nl_v, mv);

                        lo[ii/NL] += mv*ms;
                    }
                }
            }
        }

        // these are needed for reducing the results from the simdgroups (reuse the ss buffer)
        if (tiisg == 0) {
            ss[0] = (s_t) S;
            ss[1] = (s_t) M;
        }
    }

    // simdgroup reduce
    // [ 0,  8, 16, 24] -> [ 0]
    // [ 1,  9, 17, 25] -> [ 1]
    // [ 2, 10, 18, 26] -> [ 2]
    // [ 3, 11, 19, 27] -> [ 3]
    // [ 4, 12, 20, 28] -> [ 4]
    // [ 5, 13, 21, 29] -> [ 5]
    // [ 6, 14, 22, 30] -> [ 6]
    // [ 7, 15, 23, 31] -> [ 7]
    for (short ii = 0; ii < D16; ii += NL) {
        lo[ii/NL][0] += simd_shuffle_down(lo[ii/NL][0], 16);
        lo[ii/NL][0] += simd_shuffle_down(lo[ii/NL][0],  8);
      //lo[ii/NL][0] += simd_shuffle_down(lo[ii/NL][0],  4);
      //lo[ii/NL][0] += simd_shuffle_down(lo[ii/NL][0],  2);
      //lo[ii/NL][0] += simd_shuffle_down(lo[ii/NL][0],  1);

        lo[ii/NL][1] += simd_shuffle_down(lo[ii/NL][1], 16);
        lo[ii/NL][1] += simd_shuffle_down(lo[ii/NL][1],  8);
      //lo[ii/NL][1] += simd_shuffle_down(lo[ii/NL][1],  4);
      //lo[ii/NL][1] += simd_shuffle_down(lo[ii/NL][1],  2);
      //lo[ii/NL][1] += simd_shuffle_down(lo[ii/NL][1],  1);

        lo[ii/NL][2] += simd_shuffle_down(lo[ii/NL][2], 16);
        lo[ii/NL][2] += simd_shuffle_down(lo[ii/NL][2],  8);
      //lo[ii/NL][2] += simd_shuffle_down(lo[ii/NL][2],  4);
      //lo[ii/NL][2] += simd_shuffle_down(lo[ii/NL][2],  2);
      //lo[ii/NL][2] += simd_shuffle_down(lo[ii/NL][2],  1);

        lo[ii/NL][3] += simd_shuffle_down(lo[ii/NL][3], 16);
        lo[ii/NL][3] += simd_shuffle_down(lo[ii/NL][3],  8);
      //lo[ii/NL][3] += simd_shuffle_down(lo[ii/NL][3],  4);
      //lo[ii/NL][3] += simd_shuffle_down(lo[ii/NL][3],  2);
      //lo[ii/NL][3] += simd_shuffle_down(lo[ii/NL][3],  1);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // store results to shared memory
    for (short i = tiisg; i < D16; i += NL) {
        sr4x4[i] = lo[i/NL];
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // parallel reduce
    for (short r = nsg/2; r > 0; r >>= 1) {
        if (sgitg < r) {
            const half S0 = ss[       0];
            const half S1 = ss[r*SH + 0];

            const half M0 = ss[       1];
            const half M1 = ss[r*SH + 1];

            const half M = max(M0, M1);

            const half ms0 = exp(M0 - M);
            const half ms1 = exp(M1 - M);

            const half S = S0*ms0 + S1*ms1;

            if (tiisg == 0) {
                ss[0] = S;
                ss[1] = M;
            }

            // O_0 = diag(ms0)*O_0 + diag(ms1)*O_1
            for (short i = tiisg; i < D16; i += NW) {
                sr4x4[i] = sr4x4[i]*ms0 + sr4x4[i + r*D16]*ms1;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    device float4x4 * dst44 = (device float4x4 *) dst;

    // final rescale with 1/S and store to global memory
    if (sgitg == 0) {
        const float S = ss[0];

        for (short i = tiisg; i < D16; i += NW) {
            dst44[((uint64_t)iq3*args.ne2*args.ne1 + iq2 + (uint64_t)iq1*args.ne1)*D16 + i] = (float4x4) sr4x4[i]/S;
        }
    }
}

// note: I think the s_t can be half instead of float, because the Q*K scaling is done before storing to shared mem
//       in the other (non-vec) kernel, we need s_t to also be float because we scale during the soft_max
//
#define FA_TYPES \
           half4,  half4x4, \
                   half4x4, \
                   half4x4, \
    float,                  \
    half,  half4,  half4x4, \
                   half4x4

typedef decltype(kernel_flash_attn_ext_vec<FA_TYPES, half4x4, 1, dequantize_f16, half4x4, 1, dequantize_f16, 128>) flash_attn_ext_vec_t;

template [[host_name("kernel_flash_attn_ext_vec_f16_h128")]]  kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, half4x4,    1, dequantize_f16,  half4x4,     1, dequantize_f16,  128>;
#if defined(GGML_METAL_USE_BF16)
template [[host_name("kernel_flash_attn_ext_vec_bf16_h128")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, bfloat4x4,  1, dequantize_bf16, bfloat4x4,   1, dequantize_bf16, 128>;
#endif
template [[host_name("kernel_flash_attn_ext_vec_q4_0_h128")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_q4_0, 2, dequantize_q4_0, block_q4_0,  2, dequantize_q4_0, 128>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_h128")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_q4_1, 2, dequantize_q4_1, block_q4_1,  2, dequantize_q4_1, 128>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_h128")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_q5_0, 2, dequantize_q5_0, block_q5_0,  2, dequantize_q5_0, 128>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_h128")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_q5_1, 2, dequantize_q5_1, block_q5_1,  2, dequantize_q5_1, 128>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_h128")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_q8_0, 2, dequantize_q8_0, block_q8_0,  2, dequantize_q8_0, 128>;

template [[host_name("kernel_flash_attn_ext_vec_f16_h256")]]  kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, half4x4,    1, dequantize_f16,  half4x4,     1, dequantize_f16,  256>;
#if defined(GGML_METAL_USE_BF16)
template [[host_name("kernel_flash_attn_ext_vec_bf16_h256")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, bfloat4x4,  1, dequantize_bf16, bfloat4x4,   1, dequantize_bf16, 256>;
#endif
template [[host_name("kernel_flash_attn_ext_vec_q4_0_h256")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_q4_0, 2, dequantize_q4_0, block_q4_0,  2, dequantize_q4_0, 256>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_h256")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_q4_1, 2, dequantize_q4_1, block_q4_1,  2, dequantize_q4_1, 256>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_h256")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_q5_0, 2, dequantize_q5_0, block_q5_0,  2, dequantize_q5_0, 256>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_h256")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_q5_1, 2, dequantize_q5_1, block_q5_1,  2, dequantize_q5_1, 256>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_h256")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_q8_0, 2, dequantize_q8_0, block_q8_0,  2, dequantize_q8_0, 256>;

#undef FA_TYPES

template<typename T>
kernel void kernel_set(
    constant ggml_metal_kargs_set & args,
    device  const char * src0,
    device  const char * src1,
    device        char * dst,
    uint3   tgpig[[threadgroup_position_in_grid]],
    ushort3 tpitg[[thread_position_in_threadgroup]],
    ushort3   ntg[[threads_per_threadgroup]]) {
    const int i13 = tgpig[2];
    const int i12 = tgpig[1];
    const int i11 = tgpig[0];

    const int64_t n = i13*args.ne12*args.ne11*args.ne10 + i12*args.ne11*args.ne10 + i11*args.ne10;

    const int64_t i3 = n / (args.ne12*args.ne11*args.ne10);
    const int64_t i2 = (n - i3*args.ne12*args.ne11*args.ne10) / (args.ne11*args.ne10);
    const int64_t i1 = (n - i3*args.ne12*args.ne11*args.ne10 - i2*args.ne11*args.ne10) / args.ne10;

    device T * dst_data = (device T *) (dst + i3*args.nb3 + i2*args.nb2 + i1*args.nb1 + args.offs);

    for (int64_t i10 = tpitg.x; i10 < args.ne10; i10 += ntg.x) {
        device const T * src = (device T *) (src1 + i13*args.nb13 + i12*args.nb12 + i11*args.nb11 + i10*args.nb10);
        dst_data[i10] = (T) src[0];
    }
}

typedef decltype(kernel_set<float>) kernel_set_t;

template [[host_name("kernel_set_f32")]] kernel kernel_set_t kernel_set<float>;
template [[host_name("kernel_set_i32")]] kernel kernel_set_t kernel_set<int32_t>;

template<typename T0, typename T1>
kernel void kernel_cpy(
        constant ggml_metal_kargs_cpy & args,
        device  const char * src0,
        device        char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    const int i03 = tgpig[2];
    const int i02 = tgpig[1];
    const int i01 = tgpig[0];

    const int64_t n = i03*args.ne02*args.ne01*args.ne00 + i02*args.ne01*args.ne00 + i01*args.ne00;

    const int64_t i3 = n/(args.ne2*args.ne1*args.ne0);
    const int64_t i2 = (n - i3*args.ne2*args.ne1*args.ne0)/(args.ne1*args.ne0);
    const int64_t i1 = (n - i3*args.ne2*args.ne1*args.ne0 - i2*args.ne1*args.ne0)/args.ne0;
    const int64_t i0 = (n - i3*args.ne2*args.ne1*args.ne0 - i2*args.ne1*args.ne0 - i1*args.ne0);

    device T1 * dst_data = (device T1 *) (dst + i3*args.nb3 + i2*args.nb2 + i1*args.nb1 + i0*args.nb0);

    for (int64_t i00 = tpitg.x; i00 < args.ne00; i00 += ntg.x) {
        device const T0 * src = (device T0 *)(src0 + i03*args.nb03 + i02*args.nb02 + i01*args.nb01 + i00*args.nb00);
        dst_data[i00] = (T1) src[0];
    }
}

typedef decltype(kernel_cpy<float, float>) kernel_cpy_t;

template [[host_name("kernel_cpy_f32_f32")]]   kernel kernel_cpy_t kernel_cpy<float,  float>;
template [[host_name("kernel_cpy_f32_f16")]]   kernel kernel_cpy_t kernel_cpy<float,  half>;
#if defined(GGML_METAL_USE_BF16)
template [[host_name("kernel_cpy_f32_bf16")]]  kernel kernel_cpy_t kernel_cpy<float,  bfloat>;
#endif
template [[host_name("kernel_cpy_f16_f32")]]   kernel kernel_cpy_t kernel_cpy<half,   float>;
template [[host_name("kernel_cpy_f16_f16")]]   kernel kernel_cpy_t kernel_cpy<half,   half>;
#if defined(GGML_METAL_USE_BF16)
template [[host_name("kernel_cpy_bf16_f32")]]  kernel kernel_cpy_t kernel_cpy<bfloat, float>;
template [[host_name("kernel_cpy_bf16_bf16")]] kernel kernel_cpy_t kernel_cpy<bfloat, bfloat>;
#endif

kernel void kernel_cpy_f32_q8_0(
        constant ggml_metal_kargs_cpy & args,
        device const char * src0,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    const int i03 = tgpig[2];
    const int i02 = tgpig[1];
    const int i01 = tgpig[0];

    const int64_t n = i03*args.ne02*args.ne01*args.ne00 + i02*args.ne01*args.ne00 + i01*args.ne00;

    const int64_t i3 = n / (args.ne2*args.ne1*args.ne0);
    const int64_t i2 = (n - i3*args.ne2*args.ne1*args.ne0) / (args.ne1*args.ne0);
    const int64_t i1 = (n - i3*args.ne2*args.ne1*args.ne0 - i2*args.ne1*args.ne0) / args.ne0;
    const int64_t i0 = (n - i3*args.ne2*args.ne1*args.ne0 - i2*args.ne1*args.ne0 - i1*args.ne0)/QK8_0;

    device block_q8_0 * dst_data = (device block_q8_0 *) (dst + i3*args.nb3 + i2*args.nb2 + i1*args.nb1 + i0*args.nb0);

    for (int64_t i00 = tpitg.x*QK8_0; i00 < args.ne00; i00 += ntg.x*QK8_0) {
        device const float * src = (device float *)(src0 + i03*args.nb03 + i02*args.nb02 + i01*args.nb01 + i00*args.nb00);

        float amax = 0.0f; // absolute max

        for (int j = 0; j < QK8_0; j++) {
            const float v = src[j];
            amax = MAX(amax, fabs(v));
        }

        const float d = amax / ((1 << 7) - 1);
        const float id = d ? 1.0f/d : 0.0f;

        dst_data[i00/QK8_0].d = d;

        for (int j = 0; j < QK8_0; ++j) {
            const float x0 = src[j]*id;

            dst_data[i00/QK8_0].qs[j] = round(x0);
        }
    }
}

kernel void kernel_cpy_f32_q4_0(
        constant ggml_metal_kargs_cpy & args,
        device const char * src0,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    const int i03 = tgpig[2];
    const int i02 = tgpig[1];
    const int i01 = tgpig[0];

    const int64_t n = i03*args.ne02*args.ne01*args.ne00 + i02*args.ne01*args.ne00 + i01*args.ne00;

    const int64_t i3 = n / (args.ne2*args.ne1*args.ne0);
    const int64_t i2 = (n - i3*args.ne2*args.ne1*args.ne0) / (args.ne1*args.ne0);
    const int64_t i1 = (n - i3*args.ne2*args.ne1*args.ne0 - i2*args.ne1*args.ne0) / args.ne0;
    const int64_t i0 = (n - i3*args.ne2*args.ne1*args.ne0 - i2*args.ne1*args.ne0 - i1*args.ne0)/QK4_0;

    device block_q4_0 * dst_data = (device block_q4_0 *) (dst + i3*args.nb3 + i2*args.nb2 + i1*args.nb1 + i0*args.nb0);

    for (int64_t i00 = tpitg.x*QK4_0; i00 < args.ne00; i00 += ntg.x*QK4_0) {
        device const float * src = (device float *)(src0 + i03*args.nb03 + i02*args.nb02 + i01*args.nb01 + i00*args.nb00);

        float amax = 0.0f; // absolute max
        float max  = 0.0f;

        for (int j = 0; j < QK4_0; j++) {
            const float v = src[j];
            if (amax < fabs(v)) {
                amax = fabs(v);
                max  = v;
            }
        }

        const float d = max / -8;
        const float id = d ? 1.0f/d : 0.0f;

        dst_data[i00/QK4_0].d = d;

        for (int j = 0; j < QK4_0/2; ++j) {
            const float x0 = src[0       + j]*id;
            const float x1 = src[QK4_0/2 + j]*id;

            const uint8_t xi0 = MIN(15, (int8_t)(x0 + 8.5f));
            const uint8_t xi1 = MIN(15, (int8_t)(x1 + 8.5f));

            dst_data[i00/QK4_0].qs[j]  = xi0;
            dst_data[i00/QK4_0].qs[j] |= xi1 << 4;
        }
    }
}

kernel void kernel_cpy_f32_q4_1(
        constant ggml_metal_kargs_cpy & args,
        device const char * src0,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    const int i03 = tgpig[2];
    const int i02 = tgpig[1];
    const int i01 = tgpig[0];

    const int64_t n = i03*args.ne02*args.ne01*args.ne00 + i02*args.ne01*args.ne00 + i01*args.ne00;

    const int64_t i3 = n / (args.ne2*args.ne1*args.ne0);
    const int64_t i2 = (n - i3*args.ne2*args.ne1*args.ne0) / (args.ne1*args.ne0);
    const int64_t i1 = (n - i3*args.ne2*args.ne1*args.ne0 - i2*args.ne1*args.ne0) / args.ne0;
    const int64_t i0 = (n - i3*args.ne2*args.ne1*args.ne0 - i2*args.ne1*args.ne0 - i1*args.ne0)/QK4_1;

    device block_q4_1 * dst_data = (device block_q4_1 *) (dst + i3*args.nb3 + i2*args.nb2 + i1*args.nb1 + i0*args.nb0);

    for (int64_t i00 = tpitg.x*QK4_1; i00 < args.ne00; i00 += ntg.x*QK4_1) {
        device const float * src = (device float *)(src0 + i03*args.nb03 + i02*args.nb02 + i01*args.nb01 + i00*args.nb00);

        float min = FLT_MAX;
        float max = -FLT_MAX;

        for (int j = 0; j < QK4_1; j++) {
            const float v = src[j];
            if (min > v) min = v;
            if (max < v) max = v;
        }

        const float d = (max - min) / ((1 << 4) - 1);
        const float id = d ? 1.0f/d : 0.0f;

        dst_data[i00/QK4_1].d = d;
        dst_data[i00/QK4_1].m = min;

        for (int j = 0; j < QK4_1/2; ++j) {
            const float x0 = (src[0       + j] - min)*id;
            const float x1 = (src[QK4_1/2 + j] - min)*id;

            const uint8_t xi0 = MIN(15, (int8_t)(x0 + 0.5f));
            const uint8_t xi1 = MIN(15, (int8_t)(x1 + 0.5f));

            dst_data[i00/QK4_1].qs[j]  = xi0;
            dst_data[i00/QK4_1].qs[j] |= xi1 << 4;
        }
    }
}

kernel void kernel_cpy_f32_q5_0(
        constant ggml_metal_kargs_cpy & args,
        device const char * src0,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    const int i03 = tgpig[2];
    const int i02 = tgpig[1];
    const int i01 = tgpig[0];

    const int64_t n = i03*args.ne02*args.ne01*args.ne00 + i02*args.ne01*args.ne00 + i01*args.ne00;

    const int64_t i3 = n / (args.ne2*args.ne1*args.ne0);
    const int64_t i2 = (n - i3*args.ne2*args.ne1*args.ne0) / (args.ne1*args.ne0);
    const int64_t i1 = (n - i3*args.ne2*args.ne1*args.ne0 - i2*args.ne1*args.ne0) / args.ne0;
    const int64_t i0 = (n - i3*args.ne2*args.ne1*args.ne0 - i2*args.ne1*args.ne0 - i1*args.ne0)/QK5_0;

    device block_q5_0 * dst_data = (device block_q5_0 *) (dst + i3*args.nb3 + i2*args.nb2 + i1*args.nb1 + i0*args.nb0);

    for (int64_t i00 = tpitg.x*QK5_0; i00 < args.ne00; i00 += ntg.x*QK5_0) {
        device const float * src = (device float *)(src0 + i03*args.nb03 + i02*args.nb02 + i01*args.nb01 + i00*args.nb00);

        float amax = 0.0f; // absolute max
        float max  = 0.0f;

        for (int j = 0; j < QK5_0; j++) {
            const float v = src[j];
            if (amax < fabs(v)) {
                amax = fabs(v);
                max  = v;
            }
        }

        const float d = max / -16;
        const float id = d ? 1.0f/d : 0.0f;

        dst_data[i00/QK5_0].d = d;

        uint32_t qh = 0;
        for (int j = 0; j < QK5_0/2; ++j) {
            const float x0 = src[0       + j]*id;
            const float x1 = src[QK5_0/2 + j]*id;

            const uint8_t xi0 = MIN(31, (int8_t)(x0 + 16.5f));
            const uint8_t xi1 = MIN(31, (int8_t)(x1 + 16.5f));

            dst_data[i00/QK5_0].qs[j] = (xi0 & 0xf) | ((xi1 & 0xf) << 4);
            qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
            qh |= ((xi1 & 0x10u) >> 4) << (j + QK5_0/2);
        }
        thread const uint8_t * qh8 = (thread const uint8_t *)&qh;
        for (int j = 0; j < 4; ++j) {
            dst_data[i00/QK5_0].qh[j] = qh8[j];
        }
    }
}

kernel void kernel_cpy_f32_q5_1(
        constant ggml_metal_kargs_cpy & args,
        device const char * src0,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    const int i03 = tgpig[2];
    const int i02 = tgpig[1];
    const int i01 = tgpig[0];

    const int64_t n = i03*args.ne02*args.ne01*args.ne00 + i02*args.ne01*args.ne00 + i01*args.ne00;

    const int64_t i3 = n / (args.ne2*args.ne1*args.ne0);
    const int64_t i2 = (n - i3*args.ne2*args.ne1*args.ne0) / (args.ne1*args.ne0);
    const int64_t i1 = (n - i3*args.ne2*args.ne1*args.ne0 - i2*args.ne1*args.ne0) / args.ne0;
    const int64_t i0 = (n - i3*args.ne2*args.ne1*args.ne0 - i2*args.ne1*args.ne0 - i1*args.ne0)/QK5_1;

    device block_q5_1 * dst_data = (device block_q5_1 *) (dst + i3*args.nb3 + i2*args.nb2 + i1*args.nb1 + i0*args.nb0);

    for (int64_t i00 = tpitg.x*QK5_1; i00 < args.ne00; i00 += ntg.x*QK5_1) {
        device const float * src = (device float *)(src0 + i03*args.nb03 + i02*args.nb02 + i01*args.nb01 + i00*args.nb00);

        float max = src[0];
        float min = src[0];

        for (int j = 1; j < QK5_1; j++) {
            const float v = src[j];
            min = v < min ? v : min;
            max = v > max ? v : max;
        }

        const float d = (max - min) / 31;
        const float id = d ? 1.0f/d : 0.0f;

        dst_data[i00/QK5_1].d = d;
        dst_data[i00/QK5_1].m = min;

        uint32_t qh = 0;
        for (int j = 0; j < QK5_1/2; ++j) {
            const float x0 = (src[0       + j] - min)*id;
            const float x1 = (src[QK5_1/2 + j] - min)*id;

            const uint8_t xi0 = (uint8_t)(x0 + 0.5f);
            const uint8_t xi1 = (uint8_t)(x1 + 0.5f);

            dst_data[i00/QK5_1].qs[j] = (xi0 & 0xf) | ((xi1 & 0xf) << 4);
            qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
            qh |= ((xi1 & 0x10u) >> 4) << (j + QK5_1/2);
        }
        thread const uint8_t * qh8 = (thread const uint8_t *)&qh;
        for (int j = 0; j < 4; ++j) {
            dst_data[i00/QK5_1].qh[j] = qh8[j];
        }
    }
}

static inline int best_index_int8(int n, constant float * val, float x) {
    if (x <= val[0]) return 0;
    if (x >= val[n-1]) return n-1;
    int ml = 0, mu = n-1;
    while (mu-ml > 1) {
        int mav = (ml+mu)/2;
        if (x < val[mav]) mu = mav; else ml = mav;
    }
    return x - val[mu-1] < val[mu] - x ? mu-1 : mu;
}

kernel void kernel_cpy_f32_iq4_nl(
        constant ggml_metal_kargs_cpy & args,
        device const char * src0,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    const int i03 = tgpig[2];
    const int i02 = tgpig[1];
    const int i01 = tgpig[0];

    const int64_t n = i03*args.ne02*args.ne01*args.ne00 + i02*args.ne01*args.ne00 + i01*args.ne00;

    const int64_t i3 = n / (args.ne2*args.ne1*args.ne0);
    const int64_t i2 = (n - i3*args.ne2*args.ne1*args.ne0) / (args.ne1*args.ne0);
    const int64_t i1 = (n - i3*args.ne2*args.ne1*args.ne0 - i2*args.ne1*args.ne0) / args.ne0;
    const int64_t i0 = (n - i3*args.ne2*args.ne1*args.ne0 - i2*args.ne1*args.ne0 - i1*args.ne0)/QK4_NL;

    device block_iq4_nl * dst_data = (device block_iq4_nl *) (dst + i3*args.nb3 + i2*args.nb2 + i1*args.nb1 + i0*args.nb0);

    for (int64_t i00 = tpitg.x*QK4_NL; i00 < args.ne00; i00 += ntg.x*QK4_NL) {
        device const float * src = (device float *)(src0 + i03*args.nb03 + i02*args.nb02 + i01*args.nb01 + i00*args.nb00);

        float amax = 0.0f; // absolute max
        float max  = 0.0f;

        for (int j = 0; j < QK4_0; j++) {
            const float v = src[j];
            if (amax < fabs(v)) {
                amax = fabs(v);
                max  = v;
            }
        }

        const float d = max / kvalues_iq4nl_f[0];
        const float id = d ? 1.0f/d : 0.0f;

        float sumqx = 0, sumq2 = 0;
        for (int j = 0; j < QK4_NL/2; ++j) {
            const float x0 = src[0        + j]*id;
            const float x1 = src[QK4_NL/2 + j]*id;

            const uint8_t xi0 = best_index_int8(16, kvalues_iq4nl_f, x0);
            const uint8_t xi1 = best_index_int8(16, kvalues_iq4nl_f, x1);

            dst_data[i00/QK4_NL].qs[j] = xi0 | (xi1 << 4);

            const float v0 = kvalues_iq4nl_f[xi0];
            const float v1 = kvalues_iq4nl_f[xi1];
            const float w0 = src[0        + j]*src[0        + j];
            const float w1 = src[QK4_NL/2 + j]*src[QK4_NL/2 + j];
            sumqx += w0*v0*src[j] + w1*v1*src[QK4_NL/2 + j];
            sumq2 += w0*v0*v0 + w1*v1*v1;

        }

        dst_data[i00/QK4_NL].d = sumq2 > 0 ? sumqx/sumq2 : d;
    }
}

kernel void kernel_concat(
    constant ggml_metal_kargs_concat & args,
    device  const char * src0,
    device  const char * src1,
    device        char * dst,
    uint3   tgpig[[threadgroup_position_in_grid]],
    ushort3 tpitg[[thread_position_in_threadgroup]],
    ushort3   ntg[[threads_per_threadgroup]]) {

    const int i3 = tgpig.z;
    const int i2 = tgpig.y;
    const int i1 = tgpig.x;

    int o[4] = {0, 0, 0, 0};
    o[args.dim] = args.dim == 0 ? args.ne00 : (args.dim == 1 ? args.ne01 : (args.dim == 2 ? args.ne02 : args.ne03));

    device const float * x;

    for (int i0 = tpitg.x; i0 < args.ne0; i0 += ntg.x) {
        if (i0 < args.ne00 && i1 < args.ne01 && i2 < args.ne02 && i3 < args.ne03) {
            x = (device const float *)(src0 + (i3       )*args.nb03 + (i2       )*args.nb02 + (i1       )*args.nb01 + (i0       )*args.nb00);
        } else {
            x = (device const float *)(src1 + (i3 - o[3])*args.nb13 + (i2 - o[2])*args.nb12 + (i1 - o[1])*args.nb11 + (i0 - o[0])*args.nb10);
        }

        device float * y = (device float *)(dst + i3*args.nb3 + i2*args.nb2 + i1*args.nb1 + i0*args.nb0);

        *y = *x;
    }
}

template<typename args_t>
void kernel_mul_mv_q2_K_f32_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {

    const int nb = args.ne00/QK_K;
    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * N_SIMDGROUP + sgitg) * N_DST;

    const uint i12 = im%args.ne12;
    const uint i13 = im/args.ne12;

    const uint64_t offset0 = first_row*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;
    const uint64_t offset1 =        r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

    device const block_q2_K * x = (device const block_q2_K *) (src0 + offset0);
    device const float      * y = (device const float      *) (src1 + offset1);

    float yl[32];
    float sumf[N_DST]={0.f}, all_sum;

    const int ix = tiisg/8;  // 0...3
    const int it = tiisg%8;  // 0...7
    const int iq = it/4;     // 0 or 1
    const int ir = it%4;     // 0...3
    const int is = (8*ir)/16;// 0 or 1

    device const float * y4 = y + ix * QK_K + 128 * iq + 8 * ir;

    for (int ib = ix; ib < nb; ib += 4) {

        float4 sumy = {0.f, 0.f, 0.f, 0.f};
        for (int i = 0; i < 8; ++i) {
            yl[i+ 0] = y4[i+ 0]; sumy[0] += yl[i+ 0];
            yl[i+ 8] = y4[i+32]; sumy[1] += yl[i+ 8];
            yl[i+16] = y4[i+64]; sumy[2] += yl[i+16];
            yl[i+24] = y4[i+96]; sumy[3] += yl[i+24];
        }

        device const uint8_t  * sc = (device const uint8_t  *)x[ib].scales + 8*iq + is;
        device const uint16_t * qs = (device const uint16_t *)x[ib].qs + 16 * iq + 4 * ir;
        device const half     * dh = &x[ib].d;

        for (int row = 0; row < N_DST; row++) {

            float4 acc1 = {0.f, 0.f, 0.f, 0.f};
            float4 acc2 = {0.f, 0.f, 0.f, 0.f};
            for (int i = 0; i < 8; i += 2) {
                acc1[0] += yl[i+ 0] * (qs[i/2] & 0x0003);
                acc2[0] += yl[i+ 1] * (qs[i/2] & 0x0300);
                acc1[1] += yl[i+ 8] * (qs[i/2] & 0x000c);
                acc2[1] += yl[i+ 9] * (qs[i/2] & 0x0c00);
                acc1[2] += yl[i+16] * (qs[i/2] & 0x0030);
                acc2[2] += yl[i+17] * (qs[i/2] & 0x3000);
                acc1[3] += yl[i+24] * (qs[i/2] & 0x00c0);
                acc2[3] += yl[i+25] * (qs[i/2] & 0xc000);
            }
            float dall = dh[0];
            float dmin = dh[1] * 1.f/16.f;
            sumf[row] += dall * ((acc1[0] + 1.f/256.f * acc2[0]) * (sc[0] & 0xF) * 1.f/ 1.f +
                                 (acc1[1] + 1.f/256.f * acc2[1]) * (sc[2] & 0xF) * 1.f/ 4.f +
                                 (acc1[2] + 1.f/256.f * acc2[2]) * (sc[4] & 0xF) * 1.f/16.f +
                                 (acc1[3] + 1.f/256.f * acc2[3]) * (sc[6] & 0xF) * 1.f/64.f) -
                         dmin * (sumy[0] * (sc[0] & 0xF0) + sumy[1] * (sc[2] & 0xF0) + sumy[2] * (sc[4] & 0xF0) + sumy[3] * (sc[6] & 0xF0));

            qs += args.nb01/2;
            sc += args.nb01;
            dh += args.nb01/2;
        }

        y4 += 4 * QK_K;
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    for (int row = 0; row < N_DST; ++row) {
        all_sum = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst_f32[first_row + row] = all_sum;
        }
    }
}

[[host_name("kernel_mul_mv_q2_K_f32")]]
kernel void kernel_mul_mv_q2_K_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_q2_K_f32_impl<constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, nullptr, tgpig, tiisg, sgitg);
}

template<typename args_t>
void kernel_mul_mv_q3_K_f32_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {

    const int nb = args.ne00/QK_K;

    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * N_SIMDGROUP + sgitg) * 2;

    const uint i12 = im%args.ne12;
    const uint i13 = im/args.ne12;

    const uint64_t offset0 = first_row*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;
    const uint64_t offset1 =        r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

    device const block_q3_K * x = (device const block_q3_K *) (src0 + offset0);
    device const float     * yy = (device const float      *) (src1 + offset1);

    float yl[32];

    //const uint16_t kmask1 = 0x3030;
    //const uint16_t kmask2 = 0x0f0f;

    const int tid = tiisg/4;
    const int ix  = tiisg%4;
    const int ip  = tid/4;          // 0 or 1
    const int il  = 2*((tid%4)/2);  // 0 or 2
    const int ir  = tid%2;
    const int n   = 8;
    const int l0  = n*ir;

    // One would think that the Metal compiler would figure out that ip and il can only have
    // 4 possible states, and optimize accordingly. Well, no. It needs help, and we do it
    // with these two tales.
    //
    // Possible masks for the high bit
    const ushort4 mm[4] = {{0x0001, 0x0100, 0x0002, 0x0200},  // ip = 0, il = 0
                           {0x0004, 0x0400, 0x0008, 0x0800},  // ip = 0, il = 2
                           {0x0010, 0x1000, 0x0020, 0x2000},  // ip = 1, il = 0
                           {0x0040, 0x4000, 0x0080, 0x8000}}; // ip = 1, il = 2

    // Possible masks for the low 2 bits
    const int4 qm[2] = {{0x0003, 0x0300, 0x000c, 0x0c00}, {0x0030, 0x3000, 0x00c0, 0xc000}};

    const ushort4 hm = mm[2*ip + il/2];

    const short shift = 2*il;

    const float v1 = il == 0 ? 4.f : 64.f;
    const float v2 = 4.f * v1;

    const uint16_t s_shift1 = 4*ip;
    const uint16_t s_shift2 = s_shift1 + il;

    const int q_offset = 32*ip + l0;
    const int y_offset = 128*ip + 32*il + l0;

    device const float * y1 = yy + ix*QK_K + y_offset;

    uint32_t scales32, aux32;
    thread uint16_t * scales16 = (thread uint16_t *)&scales32;
    thread const int8_t * scales = (thread const int8_t *)&scales32;

    float sumf1[2] = {0.f};
    float sumf2[2] = {0.f};
    for (int i = ix; i < nb; i += 4) {
        for (int l = 0; l < 8; ++l) {
            yl[l+ 0] = y1[l+ 0];
            yl[l+ 8] = y1[l+16];
            yl[l+16] = y1[l+32];
            yl[l+24] = y1[l+48];
        }

        device const uint16_t * q = (device const uint16_t *)(x[i].qs + q_offset);
        device const uint16_t * h = (device const uint16_t *)(x[i].hmask + l0);
        device const uint16_t * a = (device const uint16_t *)(x[i].scales);
        device const half * dh = &x[i].d;

        for (int row = 0; row < 2; ++row) {
            const float d_all = (float)dh[0];

            scales16[0] = a[4];
            scales16[1] = a[5];
            aux32 = ((scales32 >> s_shift2) << 4) & 0x30303030;
            scales16[0] = a[il+0];
            scales16[1] = a[il+1];
            scales32 = ((scales32 >> s_shift1) & 0x0f0f0f0f) | aux32;

            float s1 = 0, s2 = 0, s3 = 0, s4 = 0, s5 = 0, s6 = 0;
            for (int l = 0; l < n; l += 2) {
                const int32_t qs = q[l/2];
                s1 += yl[l+0] * (qs & qm[il/2][0]);
                s2 += yl[l+1] * (qs & qm[il/2][1]);
                s3 += ((h[l/2] & hm[0]) ? 0.f : yl[l+0]) + ((h[l/2] & hm[1]) ? 0.f : yl[l+1]);
                s4 += yl[l+16] * (qs & qm[il/2][2]);
                s5 += yl[l+17] * (qs & qm[il/2][3]);
                s6 += ((h[l/2] & hm[2]) ? 0.f : yl[l+16]) + ((h[l/2] & hm[3]) ? 0.f : yl[l+17]);
            }
            float d1 = d_all * (s1 + 1.f/256.f * s2 - s3*v1);
            float d2 = d_all * (s4 + 1.f/256.f * s5 - s6*v2);
            sumf1[row] += d1 * (scales[0] - 32);
            sumf2[row] += d2 * (scales[2] - 32);

            s1 = s2 = s3 = s4 = s5 = s6 = 0;
            for (int l = 0; l < n; l += 2) {
                const int32_t qs = q[l/2+8];
                s1 += yl[l+8] * (qs & qm[il/2][0]);
                s2 += yl[l+9] * (qs & qm[il/2][1]);
                s3 += ((h[l/2+8] & hm[0]) ? 0.f : yl[l+8]) + ((h[l/2+8] & hm[1]) ? 0.f : yl[l+9]);
                s4 += yl[l+24] * (qs & qm[il/2][2]);
                s5 += yl[l+25] * (qs & qm[il/2][3]);
                s6 += ((h[l/2+8] & hm[2]) ? 0.f : yl[l+24]) + ((h[l/2+8] & hm[3]) ? 0.f : yl[l+25]);
            }
            d1 = d_all * (s1 + 1.f/256.f * s2 - s3*v1);
            d2 = d_all * (s4 + 1.f/256.f * s5 - s6*v2);
            sumf1[row] += d1 * (scales[1] - 32);
            sumf2[row] += d2 * (scales[3] - 32);

            q  += args.nb01/2;
            h  += args.nb01/2;
            a  += args.nb01/2;
            dh += args.nb01/2;
        }

        y1 += 4 * QK_K;
    }

    for (int row = 0; row < 2; ++row) {
        const float sumf = (sumf1[row] + 0.25f * sumf2[row]) / (1 << shift);
        sumf1[row] = simd_sum(sumf);
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    if (tiisg == 0) {
        for (int row = 0; row < 2; ++row) {
            dst_f32[first_row + row] = sumf1[row];
        }
    }
}

[[host_name("kernel_mul_mv_q3_K_f32")]]
kernel void kernel_mul_mv_q3_K_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_q3_K_f32_impl<constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, nullptr, tgpig, tiisg, sgitg);
}

template<typename args_t>
void kernel_mul_mv_q4_K_f32_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {

    const uint16_t kmask1 = 0x3f3f;
    const uint16_t kmask2 = 0x0f0f;
    const uint16_t kmask3 = 0xc0c0;

    const int ix = tiisg/8;  // 0...3
    const int it = tiisg%8;  // 0...7
    const int iq = it/4;     // 0 or 1
    const int ir = it%4;     // 0...3

    const int nb = args.ne00/QK_K;
    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;
    //const int first_row = (r0 * N_SIMDGROUP + sgitg) * N_DST;
    const int first_row = r0 * N_DST;

    const uint i12 = im%args.ne12;
    const uint i13 = im/args.ne12;

    const uint64_t offset0 = first_row*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;
    const uint64_t offset1 =        r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

    device const block_q4_K * x = (device const block_q4_K *) (src0 + offset0);
    device const float      * y = (device const float      *) (src1 + offset1);

    float yl[16];
    float yh[16];
    float sumf[N_DST]={0.f}, all_sum;

    device const float * y4 = y + ix * QK_K + 64 * iq + 8 * ir;

    uint16_t sc16[4];
    thread const uint8_t * sc8 = (thread const uint8_t *)sc16;

    for (int ib = ix; ib < nb; ib += 4) {
        float4 sumy = {0.f, 0.f, 0.f, 0.f};
        for (int i = 0; i < 8; ++i) {
            yl[i+0] = y4[i+  0]; sumy[0] += yl[i+0];
            yl[i+8] = y4[i+ 32]; sumy[1] += yl[i+8];
            yh[i+0] = y4[i+128]; sumy[2] += yh[i+0];
            yh[i+8] = y4[i+160]; sumy[3] += yh[i+8];
        }

        device const uint16_t * sc = (device const uint16_t *)x[ib].scales + iq;
        device const uint16_t * q1 = (device const uint16_t *)x[ib].qs + 16 * iq + 4 * ir;
        device const half     * dh = &x[ib].d;

        for (int row = 0; row < N_DST; row++) {
            sc16[0] = sc[0] & kmask1;
            sc16[1] = sc[2] & kmask1;
            sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
            sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);

            device const uint16_t * q2 = q1 + 32;

            float4 acc1 = {0.f, 0.f, 0.f, 0.f};
            float4 acc2 = {0.f, 0.f, 0.f, 0.f};
            for (int i = 0; i < 8; i += 2) {
                acc1[0] += yl[i+0] * (q1[i/2] & 0x000F);
                acc1[1] += yl[i+1] * (q1[i/2] & 0x0F00);
                acc1[2] += yl[i+8] * (q1[i/2] & 0x00F0);
                acc1[3] += yl[i+9] * (q1[i/2] & 0xF000);
                acc2[0] += yh[i+0] * (q2[i/2] & 0x000F);
                acc2[1] += yh[i+1] * (q2[i/2] & 0x0F00);
                acc2[2] += yh[i+8] * (q2[i/2] & 0x00F0);
                acc2[3] += yh[i+9] * (q2[i/2] & 0xF000);
            }

            float dall = dh[0];
            float dmin = dh[1];
            sumf[row] += dall * ((acc1[0] + 1.f/256.f * acc1[1]) * sc8[0] +
                                 (acc1[2] + 1.f/256.f * acc1[3]) * sc8[1] * 1.f/16.f +
                                 (acc2[0] + 1.f/256.f * acc2[1]) * sc8[4] +
                                 (acc2[2] + 1.f/256.f * acc2[3]) * sc8[5] * 1.f/16.f) -
                         dmin * (sumy[0] * sc8[2] + sumy[1] * sc8[3] + sumy[2] * sc8[6] + sumy[3] * sc8[7]);

            q1 += args.nb01/2;
            sc += args.nb01/2;
            dh += args.nb01/2;
        }

        y4 += 4 * QK_K;
    }

    device float * dst_f32 = (device float *) dst + (int64_t)im*args.ne0*args.ne1 + (int64_t)r1*args.ne0;

    for (int row = 0; row < N_DST; ++row) {
        all_sum = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst_f32[first_row + row] = all_sum;
        }
    }
}

[[host_name("kernel_mul_mv_q4_K_f32")]]
kernel void kernel_mul_mv_q4_K_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_q4_K_f32_impl<constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, nullptr, tgpig, tiisg, sgitg);
}

template<typename args_t>
void kernel_mul_mv_q5_K_f32_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {

    const int nb = args.ne00/QK_K;

    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * N_SIMDGROUP + sgitg) * 2;

    const uint i12 = im%args.ne12;
    const uint i13 = im/args.ne12;

    const uint64_t offset0 = first_row*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;
    const uint64_t offset1 =        r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

    device const block_q5_K * x = (device const block_q5_K *) (src0 + offset0);
    device const float     * yy = (device const float      *) (src1 + offset1);

    float sumf[2]={0.f};

    float yl[16], yh[16];

    const uint16_t kmask1 = 0x3f3f;
    const uint16_t kmask2 = 0x0f0f;
    const uint16_t kmask3 = 0xc0c0;

    const int tid = tiisg/4;
    const int ix  = tiisg%4;
    const int iq  = tid/4;
    const int ir  = tid%4;
    const int n   = 8;

    const int l0 = n*ir;
    const int q_offset = 32*iq + l0;
    const int y_offset = 64*iq + l0;

    const uint8_t hm1 = 1u << (2*iq);
    const uint8_t hm2 = hm1 << 1;
    const uint8_t hm3 = hm1 << 4;
    const uint8_t hm4 = hm2 << 4;

    uint16_t sc16[4];
    thread const uint8_t * sc8 = (thread const uint8_t *)sc16;

    device const float * y1 = yy + ix*QK_K + y_offset;

    for (int i = ix; i < nb; i += 4) {
        device const uint8_t * q1 = x[i].qs + q_offset;
        device const uint8_t * qh = x[i].qh + l0;
        device const half * dh = &x[i].d;
        device const uint16_t * a = (device const uint16_t *)x[i].scales + iq;

        device const float * y2 = y1 + 128;
        float4 sumy = {0.f, 0.f, 0.f, 0.f};
        for (int l = 0; l < 8; ++l) {
            yl[l+0] = y1[l+ 0]; sumy[0] += yl[l+0];
            yl[l+8] = y1[l+32]; sumy[1] += yl[l+8];
            yh[l+0] = y2[l+ 0]; sumy[2] += yh[l+0];
            yh[l+8] = y2[l+32]; sumy[3] += yh[l+8];
        }

        for (int row = 0; row < 2; ++row) {
            device const uint8_t * q2 = q1 + 64;

            sc16[0] = a[0] & kmask1;
            sc16[1] = a[2] & kmask1;
            sc16[2] = ((a[4] >> 0) & kmask2) | ((a[0] & kmask3) >> 2);
            sc16[3] = ((a[4] >> 4) & kmask2) | ((a[2] & kmask3) >> 2);

            float4 acc1 = {0.f};
            float4 acc2 = {0.f};
            for (int l = 0; l < n; ++l) {
                uint8_t h = qh[l];
                acc1[0] += yl[l+0] * (q1[l] & 0x0F);
                acc1[1] += yl[l+8] * (q1[l] & 0xF0);
                acc1[2] += yh[l+0] * (q2[l] & 0x0F);
                acc1[3] += yh[l+8] * (q2[l] & 0xF0);
                acc2[0] += h & hm1 ? yl[l+0] : 0.f;
                acc2[1] += h & hm2 ? yl[l+8] : 0.f;
                acc2[2] += h & hm3 ? yh[l+0] : 0.f;
                acc2[3] += h & hm4 ? yh[l+8] : 0.f;
            }
            const float dall = dh[0];
            const float dmin = dh[1];
            sumf[row] += dall * (sc8[0] * (acc1[0] +  16.f*acc2[0]) +
                                 sc8[1] * (acc1[1]/16.f + 16.f*acc2[1]) +
                                 sc8[4] * (acc1[2] +  16.f*acc2[2]) +
                                 sc8[5] * (acc1[3]/16.f + 16.f*acc2[3])) -
                         dmin * (sumy[0] * sc8[2] + sumy[1] * sc8[3] + sumy[2] * sc8[6] + sumy[3] * sc8[7]);

            q1 += args.nb01;
            qh += args.nb01;
            dh += args.nb01/2;
            a  += args.nb01/2;
        }

        y1 += 4 * QK_K;
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    for (int row = 0; row < 2; ++row) {
        const float tot = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst_f32[first_row + row] = tot;
        }
    }
}

[[host_name("kernel_mul_mv_q5_K_f32")]]
kernel void kernel_mul_mv_q5_K_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_q5_K_f32_impl<constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, nullptr, tgpig, tiisg, sgitg);
}

template <typename args_t>
void kernel_mul_mv_q6_K_f32_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {

    const uint8_t kmask1 = 0x03;
    const uint8_t kmask2 = 0x0C;
    const uint8_t kmask3 = 0x30;
    const uint8_t kmask4 = 0xC0;

    const int nb = args.ne00/QK_K;

    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int row = 2*r0 + sgitg;

    const uint i12 = im%args.ne12;
    const uint i13 = im/args.ne12;

    const uint64_t offset0 = row*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;
    const uint64_t offset1 =  r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

    device const block_q6_K * x = (device const block_q6_K *) (src0 + offset0);
    device const float     * yy = (device const float      *) (src1 + offset1);

    float sumf = 0;

    const int tid  = tiisg/2;
    const int ix   = tiisg%2;
    const int ip   = tid/8;         // 0 or 1
    const int il   = tid%8;
    const int n    = 4;
    const int l0   = n*il;
    const int is   = 8*ip + l0/16;

    const int y_offset = 128*ip + l0;
    const int q_offset_l = 64*ip + l0;
    const int q_offset_h = 32*ip + l0;

    for (int i = ix; i < nb; i += 2) {
        device const uint8_t * q1 = x[i].ql + q_offset_l;
        device const uint8_t * q2 = q1 + 32;
        device const uint8_t * qh = x[i].qh + q_offset_h;
        device const int8_t  * sc = x[i].scales + is;

        device const float * y = yy + i * QK_K + y_offset;

        const float dall = x[i].d;

        float4 sums = {0.f, 0.f, 0.f, 0.f};
        for (int l = 0; l < n; ++l) {
            sums[0] += y[l+ 0] * ((int8_t)((q1[l] & 0xF) | ((qh[l] & kmask1) << 4)) - 32);
            sums[1] += y[l+32] * ((int8_t)((q2[l] & 0xF) | ((qh[l] & kmask2) << 2)) - 32);
            sums[2] += y[l+64] * ((int8_t)((q1[l]  >> 4) | ((qh[l] & kmask3) << 0)) - 32);
            sums[3] += y[l+96] * ((int8_t)((q2[l]  >> 4) | ((qh[l] & kmask4) >> 2)) - 32);
        }

        sumf += dall * (sums[0] * sc[0] + sums[1] * sc[2] + sums[2] * sc[4] + sums[3] * sc[6]);

    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    const float tot = simd_sum(sumf);
    if (tiisg == 0) {
        dst_f32[row] = tot;
    }
}

[[host_name("kernel_mul_mv_q6_K_f32")]]
kernel void kernel_mul_mv_q6_K_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_q6_K_f32_impl<constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, nullptr, tgpig, tiisg, sgitg);
}

// ======================= "True" 2-bit

template<typename args_t>
void kernel_mul_mv_iq2_xxs_f32_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {

    const int nb = args.ne00/QK_K;
    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * N_SIMDGROUP + sgitg) * N_DST;

    const uint i12 = im%args.ne12;
    const uint i13 = im/args.ne12;

    const uint64_t offset0 = first_row*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;
    const uint64_t offset1 =        r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

    device const block_iq2_xxs * x = (device const block_iq2_xxs *) (src0 + offset0);
    device const float         * y = (device const float         *) (src1 + offset1);

    float yl[32];
    float sumf[N_DST]={0.f}, all_sum;

    const int nb32 = nb * (QK_K / 32);

    threadgroup uint64_t * svalues = (threadgroup uint64_t *)(shmem);
    threadgroup uint8_t  * ssigns  = (threadgroup uint8_t  *)(svalues + 256);
    {
        int nval = 4;
        int pos  = (32*sgitg + tiisg)*nval;
        for (int i = 0; i < nval; ++i) svalues[pos + i] = iq2xxs_grid[pos + i];
        nval = 2;
        pos  = (32*sgitg + tiisg)*nval;
        for (int i = 0; i < nval; ++i) ssigns[pos+i] = ksigns_iq2xs[pos+i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const int ix = tiisg;

    device const float * y4 = y + 32 * ix;

    for (int ib32 = ix; ib32 < nb32; ib32 += 32) {

        for (int i = 0; i < 32; ++i) {
            yl[i] = y4[i];
        }

        const int ibl = ib32 / (QK_K / 32);
        const int ib  = ib32 % (QK_K / 32);

        device const block_iq2_xxs * xr = x + ibl;
        device const uint16_t * q2 = xr->qs + 4 * ib;
        device const half * dh = &xr->d;

        for (int row = 0; row < N_DST; row++) {

            const float db = dh[0];
            device const uint8_t * aux8 = (device const uint8_t *)q2;
            const uint32_t aux32 = q2[2] | (q2[3] << 16);
            const float d = db * (0.5f + (aux32 >> 28));

            float sum = 0;
            for (int l = 0; l < 4; ++l) {
                const threadgroup uint8_t * grid = (const threadgroup uint8_t *)(svalues + aux8[l]);
                const uint8_t signs = ssigns[(aux32 >> 7*l) & 127];
                for (int j = 0; j < 8; ++j) {
                    sum += yl[8*l + j] * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
                }
            }
            sumf[row] += d * sum;

            dh += args.nb01/2;
            q2 += args.nb01/2;
        }

        y4 += 32 * 32;
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    for (int row = 0; row < N_DST; ++row) {
        all_sum = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst_f32[first_row + row] = all_sum * 0.25f;
        }
    }
}

[[host_name("kernel_mul_mv_iq2_xxs_f32")]]
kernel void kernel_mul_mv_iq2_xxs_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_iq2_xxs_f32_impl<constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

template<typename args_t>
void kernel_mul_mv_iq2_xs_f32_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {

    const int nb = args.ne00/QK_K;
    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * N_SIMDGROUP + sgitg) * N_DST;

    const uint i12 = im%args.ne12;
    const uint i13 = im/args.ne12;

    const uint64_t offset0 = first_row*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;
    const uint64_t offset1 =        r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

    device const block_iq2_xs * x = (device const block_iq2_xs *) (src0 + offset0);
    device const float        * y = (device const float        *) (src1 + offset1);

    float yl[32];
    float sumf[N_DST]={0.f}, all_sum;

    const int nb32 = nb * (QK_K / 32);

    threadgroup uint64_t * svalues = (threadgroup uint64_t *)(shmem);
    threadgroup uint8_t  * ssigns  = (threadgroup uint8_t  *)(svalues + 512);
    {
        int nval = 8;
        int pos  = (32*sgitg + tiisg)*nval;
        for (int i = 0; i < nval; ++i) svalues[pos + i] = iq2xs_grid[pos + i];
        nval = 2;
        pos  = (32*sgitg + tiisg)*nval;
        for (int i = 0; i < nval; ++i) ssigns[pos+i] = ksigns_iq2xs[pos+i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const int ix = tiisg;

    device const float * y4 = y + 32 * ix;

    for (int ib32 = ix; ib32 < nb32; ib32 += 32) {

        for (int i = 0; i < 32; ++i) {
            yl[i] = y4[i];
        }

        const int ibl = ib32 / (QK_K / 32);
        const int ib  = ib32 % (QK_K / 32);

        device const block_iq2_xs * xr = x + ibl;
        device const uint16_t * q2 = xr->qs + 4 * ib;
        device const uint8_t  * sc = xr->scales + ib;
        device const half * dh = &xr->d;

        for (int row = 0; row < N_DST; row++) {

            const float db = dh[0];
            const uint8_t ls1 = sc[0] & 0xf;
            const uint8_t ls2 = sc[0] >>  4;
            const float d1 = db * (0.5f + ls1);
            const float d2 = db * (0.5f + ls2);

            float sum1 = 0, sum2 = 0;
            for (int l = 0; l < 2; ++l) {
                const threadgroup uint8_t * grid = (const threadgroup uint8_t *)(svalues + (q2[l] & 511));
                const uint8_t signs = ssigns[(q2[l] >> 9)];
                for (int j = 0; j < 8; ++j) {
                    sum1 += yl[8*l + j] * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
                }
            }
            for (int l = 2; l < 4; ++l) {
                const threadgroup uint8_t * grid = (const threadgroup uint8_t *)(svalues + (q2[l] & 511));
                const uint8_t signs = ssigns[(q2[l] >> 9)];
                for (int j = 0; j < 8; ++j) {
                    sum2 += yl[8*l + j] * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
                }
            }
            sumf[row] += d1 * sum1 + d2 * sum2;

            dh += args.nb01/2;
            q2 += args.nb01/2;
            sc += args.nb01;
        }

        y4 += 32 * 32;
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    for (int row = 0; row < N_DST; ++row) {
        all_sum = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst_f32[first_row + row] = all_sum * 0.25f;
        }
    }
}

[[host_name("kernel_mul_mv_iq2_xs_f32")]]
kernel void kernel_mul_mv_iq2_xs_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_iq2_xs_f32_impl<constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

template <typename args_t>
void kernel_mul_mv_iq3_xxs_f32_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {

    const int nb = args.ne00/QK_K;
    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * N_SIMDGROUP + sgitg) * N_DST;

    const uint i12 = im%args.ne12;
    const uint i13 = im/args.ne12;

    const uint64_t offset0 = first_row*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;
    const uint64_t offset1 =        r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

    device const block_iq3_xxs * x = (device const block_iq3_xxs *) (src0 + offset0);
    device const float         * y = (device const float         *) (src1 + offset1);

    float yl[32];
    float sumf[N_DST]={0.f}, all_sum;

    const int nb32 = nb * (QK_K / 32);

    threadgroup uint32_t * svalues = (threadgroup uint32_t *)(shmem);
    threadgroup uint8_t  * ssigns  = (threadgroup uint8_t  *)(svalues + 256);
    {
        int nval = 4;
        int pos  = (32*sgitg + tiisg)*nval;
        for (int i = 0; i < nval; ++i) svalues[pos + i] = iq3xxs_grid[pos + i];
        nval = 2;
        pos  = (32*sgitg + tiisg)*nval;
        for (int i = 0; i < nval; ++i) ssigns[pos+i] = ksigns_iq2xs[pos+i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const int ix = tiisg;

    device const float * y4 = y + 32 * ix;

    for (int ib32 = ix; ib32 < nb32; ib32 += 32) {
        for (int i = 0; i < 32; ++i) {
            yl[i] = y4[i];
        }

        const int ibl = ib32 / (QK_K / 32);
        const int ib  = ib32 % (QK_K / 32);

        device const block_iq3_xxs * xr = x + ibl;
        device const uint8_t  * q3 = xr->qs + 8 * ib;
        device const uint16_t * gas = (device const uint16_t *)(xr->qs + QK_K/4) + 2 * ib;
        device const half * dh = &xr->d;

        for (int row = 0; row < N_DST; row++) {
            const float db = dh[0];
            const uint32_t aux32 = gas[0] | (gas[1] << 16);
            const float d = db * (0.5f + (aux32 >> 28));

            float2 sum = {0};
            for (int l = 0; l < 4; ++l) {
                const threadgroup uint8_t * grid1 = (const threadgroup uint8_t *)(svalues + q3[2*l+0]);
                const threadgroup uint8_t * grid2 = (const threadgroup uint8_t *)(svalues + q3[2*l+1]);
                const uint8_t signs = ssigns[(aux32 >> 7*l) & 127];
                for (int j = 0; j < 4; ++j) {
                    sum[0] += yl[8*l + j + 0] * grid1[j] * (signs & kmask_iq2xs[j+0] ? -1.f : 1.f);
                    sum[1] += yl[8*l + j + 4] * grid2[j] * (signs & kmask_iq2xs[j+4] ? -1.f : 1.f);
                }
            }
            sumf[row] += d * (sum[0] + sum[1]);

            dh  += args.nb01/2;
            q3  += args.nb01;
            gas += args.nb01/2;
        }

        y4 += 32 * 32;
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    for (int row = 0; row < N_DST; ++row) {
        all_sum = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst_f32[first_row + row] = all_sum * 0.5f;
        }
    }
}

[[host_name("kernel_mul_mv_iq3_xxs_f32")]]
kernel void kernel_mul_mv_iq3_xxs_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_iq3_xxs_f32_impl<constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

template<typename args_t>
void kernel_mul_mv_iq3_s_f32_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {

    const int nb = args.ne00/QK_K;
    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * N_SIMDGROUP + sgitg) * N_DST;

    const uint i12 = im%args.ne12;
    const uint i13 = im/args.ne12;

    const uint64_t offset0 = first_row*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;
    const uint64_t offset1 =        r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

    device const block_iq3_s * x = (device const block_iq3_s *) (src0 + offset0);
    device const float       * y = (device const float       *) (src1 + offset1);

    float yl[32];
    float sumf[N_DST]={0.f}, all_sum;

    const int nb32 = nb * (QK_K / 32);

    threadgroup uint32_t * svalues = (threadgroup uint32_t *) shmem;
    {
        int nval = 8;
        int pos  = (32*sgitg + tiisg)*nval;
        for (int i = 0; i < nval; ++i) svalues[pos + i] = iq3s_grid[pos + i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const int ix = tiisg;

    device const float * y4 = y + 32 * ix;

    for (int ib32 = ix; ib32 < nb32; ib32 += 32) {

        for (int i = 0; i < 32; ++i) {
            yl[i] = y4[i];
        }

        const int ibl = ib32 / (QK_K / 32);
        const int ib  = ib32 % (QK_K / 32);

        device const block_iq3_s * xr = x + ibl;
        device const uint8_t * qs = xr->qs + 8 * ib;
        device const uint8_t * qh = xr->qh + ib;
        device const uint8_t * sc = xr->scales + (ib/2);
        device const uint8_t * signs = xr->signs + 4 * ib;
        device const half * dh = &xr->d;

        for (int row = 0; row < N_DST; row++) {

            const float db = dh[0];
            const float d = db * (1 + 2*((sc[0] >> 4*(ib%2)) & 0xf));

            float2 sum = {0};
            for (int l = 0; l < 4; ++l) {
                const threadgroup uint32_t * table1 = qh[0] & kmask_iq2xs[2*l+0] ? svalues + 256 : svalues;
                const threadgroup uint32_t * table2 = qh[0] & kmask_iq2xs[2*l+1] ? svalues + 256 : svalues;
                const threadgroup uint8_t * grid1 = (const threadgroup uint8_t *)(table1 + qs[2*l+0]);
                const threadgroup uint8_t * grid2 = (const threadgroup uint8_t *)(table2 + qs[2*l+1]);
                for (int j = 0; j < 4; ++j) {
                    sum[0] += yl[8*l + j + 0] * grid1[j] * select(1, -1, signs[l] & kmask_iq2xs[j+0]);
                    sum[1] += yl[8*l + j + 4] * grid2[j] * select(1, -1, signs[l] & kmask_iq2xs[j+4]);
                }
            }
            sumf[row] += d * (sum[0] + sum[1]);

            dh    += args.nb01/2;
            qs    += args.nb01;
            qh    += args.nb01;
            sc    += args.nb01;
            signs += args.nb01;
        }

        y4 += 32 * 32;
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    for (int row = 0; row < N_DST; ++row) {
        all_sum = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst_f32[first_row + row] = all_sum;
        }
    }
}

[[host_name("kernel_mul_mv_iq3_s_f32")]]
kernel void kernel_mul_mv_iq3_s_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_iq3_s_f32_impl<constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

template <typename args_t>
void kernel_mul_mv_iq2_s_f32_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {

    const int nb = args.ne00/QK_K;
    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * N_SIMDGROUP + sgitg) * N_DST;

    const uint i12 = im%args.ne12;
    const uint i13 = im/args.ne12;

    const uint64_t offset0 = first_row*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;
    const uint64_t offset1 =        r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

    device const block_iq2_s * x = (device const block_iq2_s *) (src0 + offset0);
    device const float       * y = (device const float       *) (src1 + offset1);

    float yl[32];
    float sumf[N_DST]={0.f}, all_sum;

    const int nb32 = nb * (QK_K / 32);

    //threadgroup uint64_t * svalues = (threadgroup uint64_t *) shmem;
    //{
    //    int nval = 32;
    //    int pos  = (32*sgitg + tiisg)*nval;
    //    for (int i = 0; i < nval; ++i) svalues[pos + i] = iq2s_grid[pos + i];
    //    threadgroup_barrier(mem_flags::mem_threadgroup);
    //}

    const int ix = tiisg;

    device const float * y4 = y + 32 * ix;

    for (int ib32 = ix; ib32 < nb32; ib32 += 32) {

        for (int i = 0; i < 32; ++i) {
            yl[i] = y4[i];
        }

        const int ibl = ib32 / (QK_K / 32);
        const int ib  = ib32 % (QK_K / 32);

        device const block_iq2_s * xr = x + ibl;
        device const uint8_t * qs = xr->qs + 4 * ib;
        device const uint8_t * qh = xr->qh + ib;
        device const uint8_t * sc = xr->scales + ib;
        device const uint8_t * signs = qs + QK_K/8;
        device const half * dh = &xr->d;

        for (int row = 0; row < N_DST; row++) {

            const float db = dh[0];
            const float d1 = db * (0.5f + (sc[0] & 0xf));
            const float d2 = db * (0.5f + (sc[0] >>  4));

            float2 sum = {0};
            for (int l = 0; l < 2; ++l) {
                //const threadgroup uint8_t * grid1 = (const threadgroup uint8_t *)(svalues + (qs[l+0] | ((qh[0] << (8-2*l)) & 0x300)));
                //const threadgroup uint8_t * grid2 = (const threadgroup uint8_t *)(svalues + (qs[l+2] | ((qh[0] << (4-2*l)) & 0x300)));
                constant uint8_t * grid1 = (constant uint8_t *)(iq2s_grid + (qs[l+0] | ((qh[0] << (8-2*l)) & 0x300)));
                constant uint8_t * grid2 = (constant uint8_t *)(iq2s_grid + (qs[l+2] | ((qh[0] << (4-2*l)) & 0x300)));
                for (int j = 0; j < 8; ++j) {
                    sum[0] += yl[8*l + j +  0] * grid1[j] * select(1, -1, signs[l+0] & kmask_iq2xs[j]);
                    sum[1] += yl[8*l + j + 16] * grid2[j] * select(1, -1, signs[l+2] & kmask_iq2xs[j]);
                }
            }
            sumf[row] += d1 * sum[0] + d2 * sum[1];

            dh    += args.nb01/2;
            qs    += args.nb01;
            qh    += args.nb01;
            sc    += args.nb01;
            signs += args.nb01;
        }

        y4 += 32 * 32;
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    for (int row = 0; row < N_DST; ++row) {
        all_sum = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst_f32[first_row + row] = all_sum * 0.25f;
        }
    }
}

[[host_name("kernel_mul_mv_iq2_s_f32")]]
kernel void kernel_mul_mv_iq2_s_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_iq2_s_f32_impl<constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

template<typename args_t>
void kernel_mul_mv_iq1_s_f32_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {

    const int nb = args.ne00/QK_K;
    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * N_SIMDGROUP + sgitg) * N_DST;

    const uint i12 = im%args.ne12;
    const uint i13 = im/args.ne12;

    const uint64_t offset0 = first_row*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;
    const uint64_t offset1 =        r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

    device const block_iq1_s * x = (device const block_iq1_s *) (src0 + offset0);
    device const float       * y = (device const float       *) (src1 + offset1);

    float yl[32];
    float sumf[N_DST]={0.f}, all_sum;

    const int nb32 = nb * (QK_K / 32);

    const int ix = tiisg;

    device const float * y4 = y + 32 * ix;

    for (int ib32 = ix; ib32 < nb32; ib32 += 32) {

        float sumy = 0;
        for (int i = 0; i < 32; ++i) {
            yl[i] = y4[i];
            sumy += yl[i];
        }

        const int ibl = ib32 / (QK_K / 32);
        const int ib  = ib32 % (QK_K / 32);

        device const block_iq1_s * xr = x + ibl;
        device const uint8_t  * qs = xr->qs + 4 * ib;
        device const uint16_t * qh = xr->qh + ib;
        device const half     * dh = &xr->d;

        for (int row = 0; row < N_DST; row++) {

            constant uint8_t * grid1 = (constant uint8_t *)(iq1s_grid_gpu + (qs[0] | ((qh[0] << 8) & 0x700)));
            constant uint8_t * grid2 = (constant uint8_t *)(iq1s_grid_gpu + (qs[1] | ((qh[0] << 5) & 0x700)));
            constant uint8_t * grid3 = (constant uint8_t *)(iq1s_grid_gpu + (qs[2] | ((qh[0] << 2) & 0x700)));
            constant uint8_t * grid4 = (constant uint8_t *)(iq1s_grid_gpu + (qs[3] | ((qh[0] >> 1) & 0x700)));

            float sum = 0;
            for (int j = 0; j < 4; ++j) {
                sum += yl[j+ 0] * (grid1[j] & 0xf) + yl[j+ 4] * (grid1[j] >> 4)
                     + yl[j+ 8] * (grid2[j] & 0xf) + yl[j+12] * (grid2[j] >> 4)
                     + yl[j+16] * (grid3[j] & 0xf) + yl[j+20] * (grid3[j] >> 4)
                     + yl[j+24] * (grid4[j] & 0xf) + yl[j+28] * (grid4[j] >> 4);
            }
            sumf[row] += (float)dh[0] * (sum + sumy * (qh[0] & 0x8000 ? -1 - IQ1S_DELTA : -1 + IQ1S_DELTA)) * (2*((qh[0] >> 12) & 7) + 1);

            dh += args.nb01/2;
            qs += args.nb01;
            qh += args.nb01/2;
        }

        y4 += 32 * 32;
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    for (int row = 0; row < N_DST; ++row) {
        all_sum = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst_f32[first_row + row] = all_sum;
        }
    }
}

template <typename args_t>
void kernel_mul_mv_iq1_m_f32_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {

    const int nb = args.ne00/QK_K;
    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * N_SIMDGROUP + sgitg) * N_DST;

    const uint i12 = im%args.ne12;
    const uint i13 = im/args.ne12;

    const uint64_t offset0 = first_row*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;
    const uint64_t offset1 =        r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

    device const block_iq1_m * x = (device const block_iq1_m *) (src0 + offset0);
    device const float       * y = (device const float       *) (src1 + offset1);

    float yl[32];
    float sumf[N_DST]={0.f}, all_sum;

    const int nb32 = nb * (QK_K / 32);

    const int ix = tiisg;

    device const float * y4 = y + 32 * ix;

    iq1m_scale_t scale;

    for (int ib32 = ix; ib32 < nb32; ib32 += 32) {

        float4 sumy = {0.f};
        for (int i = 0; i < 8; ++i) {
            yl[i+ 0] = y4[i+ 0]; sumy[0] += yl[i+ 0];
            yl[i+ 8] = y4[i+ 8]; sumy[1] += yl[i+ 8];
            yl[i+16] = y4[i+16]; sumy[2] += yl[i+16];
            yl[i+24] = y4[i+24]; sumy[3] += yl[i+24];
        }

        const int ibl = ib32 / (QK_K / 32);
        const int ib  = ib32 % (QK_K / 32);

        device const block_iq1_m * xr = x + ibl;
        device const uint8_t  * qs = xr->qs + 4 * ib;
        device const uint8_t  * qh = xr->qh + 2 * ib;
        device const uint16_t * sc = (device const uint16_t *)xr->scales;

        for (int row = 0; row < N_DST; row++) {
            scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);

            constant uint8_t * grid1 = (constant uint8_t *)(iq1s_grid_gpu + (qs[0] | ((qh[0] << 8) & 0x700)));
            constant uint8_t * grid2 = (constant uint8_t *)(iq1s_grid_gpu + (qs[1] | ((qh[0] << 4) & 0x700)));
            constant uint8_t * grid3 = (constant uint8_t *)(iq1s_grid_gpu + (qs[2] | ((qh[1] << 8) & 0x700)));
            constant uint8_t * grid4 = (constant uint8_t *)(iq1s_grid_gpu + (qs[3] | ((qh[1] << 4) & 0x700)));

            float2 sum = {0.f};
            for (int j = 0; j < 4; ++j) {
                sum[0] += yl[j+ 0] * (grid1[j] & 0xf) + yl[j+ 4] * (grid1[j] >> 4)
                        + yl[j+ 8] * (grid2[j] & 0xf) + yl[j+12] * (grid2[j] >> 4);
                sum[1] += yl[j+16] * (grid3[j] & 0xf) + yl[j+20] * (grid3[j] >> 4)
                        + yl[j+24] * (grid4[j] & 0xf) + yl[j+28] * (grid4[j] >> 4);
            }
            const float delta1 = sumy[0] * (qh[0] & 0x08 ? -1 - IQ1M_DELTA : -1 + IQ1M_DELTA) + sumy[1] * (qh[0] & 0x80 ? -1 - IQ1M_DELTA : -1 + IQ1M_DELTA);
            const float delta2 = sumy[2] * (qh[1] & 0x08 ? -1 - IQ1M_DELTA : -1 + IQ1M_DELTA) + sumy[3] * (qh[1] & 0x80 ? -1 - IQ1M_DELTA : -1 + IQ1M_DELTA);

            sumf[row] += (float)scale.f16 * ((sum[0] + delta1) * (2*((sc[ib/2] >> (6*(ib%2)+0)) & 7) + 1) +
                                             (sum[1] + delta2) * (2*((sc[ib/2] >> (6*(ib%2)+3)) & 7) + 1));

            sc += args.nb01/2;
            qs += args.nb01;
            qh += args.nb01;
        }

        y4 += 32 * 32;
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    for (int row = 0; row < N_DST; ++row) {
        all_sum = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst_f32[first_row + row] = all_sum;
        }
    }
}

template<typename args_t>
void kernel_mul_mv_iq4_nl_f32_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {

    threadgroup float * shmem_f32 = (threadgroup float *) shmem;
    const int nb = args.ne00/QK4_NL;
    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;
    const int first_row = (r0 * 2 + sgitg) * 2;

    const uint i12 = im%args.ne12;
    const uint i13 = im/args.ne12;

    const uint64_t offset0 = first_row*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;
    const uint64_t offset1 =        r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

    device const block_iq4_nl * x = (device const block_iq4_nl *) (src0 + offset0);
    device const float        * y = (device const float        *) (src1 + offset1);

    const int ix = tiisg/2;  // 0...15
    const int it = tiisg%2;  // 0 or 1

    shmem_f32[tiisg] = kvalues_iq4nl_f[tiisg%16];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float4 yl[4];
    float sumf[2]={0.f}, all_sum;

    device const float * yb = y + ix * QK4_NL + it * 8;

    uint32_t aux32[2];
    thread const uint8_t * q8 = (thread const uint8_t *)aux32;

    float4 qf1, qf2;

    for (int ib = ix; ib < nb; ib += 16) {

        device const float4 * y4 = (device const float4 *)yb;
        yl[0] = y4[0]; yl[1] = y4[4]; yl[2] = y4[1]; yl[3] = y4[5];

        for (int row = 0; row < 2 && first_row + row < args.ne01; ++row) {

            device const block_iq4_nl & xb = x[row*nb + ib];
            device const uint16_t * q4 = (device const uint16_t *)(xb.qs + 8*it);

            float4 acc1 = {0.f}, acc2 = {0.f};

            aux32[0] = q4[0] | (q4[1] << 16);
            aux32[1] = (aux32[0] >> 4) & 0x0f0f0f0f;
            aux32[0] &= 0x0f0f0f0f;
            qf1 = {shmem_f32[q8[0]], shmem_f32[q8[1]], shmem_f32[q8[2]], shmem_f32[q8[3]]};
            qf2 = {shmem_f32[q8[4]], shmem_f32[q8[5]], shmem_f32[q8[6]], shmem_f32[q8[7]]};
            acc1 += yl[0] * qf1;
            acc2 += yl[1] * qf2;

            aux32[0] = q4[2] | (q4[3] << 16);
            aux32[1] = (aux32[0] >> 4) & 0x0f0f0f0f;
            aux32[0] &= 0x0f0f0f0f;
            qf1 = {shmem_f32[q8[0]], shmem_f32[q8[1]], shmem_f32[q8[2]], shmem_f32[q8[3]]};
            qf2 = {shmem_f32[q8[4]], shmem_f32[q8[5]], shmem_f32[q8[6]], shmem_f32[q8[7]]};
            acc1 += yl[2] * qf1;
            acc2 += yl[3] * qf2;

            acc1 += acc2;

            sumf[row] += (float)xb.d * (acc1[0] + acc1[1] + acc1[2] + acc1[3]);

        }

        yb += 16 * QK4_NL;
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    for (int row = 0; row < 2 && first_row + row < args.ne01; ++row) {
        all_sum = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst_f32[first_row + row] = all_sum;
        }
    }
}

template<typename args_t>
void kernel_mul_mv_iq4_xs_f32_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {

    threadgroup float * shmem_f32 = (threadgroup float *) shmem;
    const int nb = args.ne00/QK_K;
    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;
    const int first_row = (r0 * 2 + sgitg) * 2;

    const uint i12 = im%args.ne12;
    const uint i13 = im/args.ne12;

    const uint64_t offset0 = first_row*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;
    const uint64_t offset1 =        r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

    device const block_iq4_xs * x = (device const block_iq4_xs *) (src0 + offset0);
    device const float        * y = (device const float        *) (src1 + offset1);

    const int ix = tiisg/16;  // 0 or 1
    const int it = tiisg%16;  // 0...15
    const int ib = it/2;
    const int il = it%2;

    shmem_f32[tiisg] = kvalues_iq4nl_f[tiisg%16];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float4 yl[4];
    float sumf[2]={0.f}, all_sum;

    device const float * yb = y + ix * QK_K + ib * 32 + il * 8;

    uint32_t aux32[2];
    thread const uint8_t * q8 = (thread const uint8_t *)aux32;

    float4 qf1, qf2;

    for (int ibl = ix; ibl < nb; ibl += 2) {
        device const float4 * y4 = (device const float4 *)yb;
        yl[0] = y4[0]; yl[1] = y4[4]; yl[2] = y4[1]; yl[3] = y4[5];

        for (int row = 0; row < 2; ++row) {
            device const block_iq4_xs & xb = x[row*nb + ibl];
            device const uint32_t * q4 = (device const uint32_t *)(xb.qs + 16*ib + 8*il);

            float4 acc1 = {0.f}, acc2 = {0.f};

            aux32[0] = (q4[0]     ) & 0x0f0f0f0f;
            aux32[1] = (q4[0] >> 4) & 0x0f0f0f0f;
            qf1 = {shmem_f32[q8[0]], shmem_f32[q8[1]], shmem_f32[q8[2]], shmem_f32[q8[3]]};
            qf2 = {shmem_f32[q8[4]], shmem_f32[q8[5]], shmem_f32[q8[6]], shmem_f32[q8[7]]};
            acc1 += yl[0] * qf1;
            acc2 += yl[1] * qf2;

            aux32[0] = (q4[1]     ) & 0x0f0f0f0f;
            aux32[1] = (q4[1] >> 4) & 0x0f0f0f0f;
            qf1 = {shmem_f32[q8[0]], shmem_f32[q8[1]], shmem_f32[q8[2]], shmem_f32[q8[3]]};
            qf2 = {shmem_f32[q8[4]], shmem_f32[q8[5]], shmem_f32[q8[6]], shmem_f32[q8[7]]};
            acc1 += yl[2] * qf1;
            acc2 += yl[3] * qf2;

            acc1 += acc2;

            const int ls = (((xb.scales_l[ib/2] >> 4*(ib%2)) & 0xf) | (((xb.scales_h >> 2*ib) & 3) << 4)) - 32;
            sumf[row] += (float)xb.d * ls * (acc1[0] + acc1[1] + acc1[2] + acc1[3]);

        }

        yb += 2 * QK_K;
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    for (int row = 0; row < 2; ++row) {
        all_sum = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst_f32[first_row + row] = all_sum;
        }
    }
}

[[host_name("kernel_mul_mv_iq1_s_f32")]]
kernel void kernel_mul_mv_iq1_s_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_iq1_s_f32_impl<constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, nullptr, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_iq1_m_f32")]]
kernel void kernel_mul_mv_iq1_m_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_iq1_m_f32_impl<constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, nullptr, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_iq4_nl_f32")]]
kernel void kernel_mul_mv_iq4_nl_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_iq4_nl_f32_impl<constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_iq4_xs_f32")]]
kernel void kernel_mul_mv_iq4_xs_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_iq4_xs_f32_impl<constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

template<typename block_q, short nl, void (*dequantize_func)(device const block_q *, short, thread float4x4 &)>
kernel void kernel_get_rows_q(
        device const  void * src0,
        device const  void * src1,
        device       float * dst,
        constant   int64_t & ne00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb02,
        constant   int64_t & ne10,
        constant  uint64_t & nb10,
        constant  uint64_t & nb11,
        constant  uint64_t & nb1,
        constant  uint64_t & nb2,
        uint3                tgpig[[threadgroup_position_in_grid]],
        uint                 tiitg[[thread_index_in_threadgroup]],
        uint3                tptg [[threads_per_threadgroup]]) {
    const int64_t i10 = tgpig.x;
    const int64_t i11 = tgpig.y;

    const int64_t r = ((const device int32_t *) ((const device char *) src1 + i11*nb11 + i10*nb10))[0];

    const int64_t i02 = i11;

    for (int64_t ind = tiitg; ind < ne00/16; ind += tptg.x) {
        float4x4 temp;
        dequantize_func(((device const block_q *) ((const device char *) src0 + r*nb01 + i02*nb02)) + ind/nl, ind%nl, temp);
        *(((device float4x4 *) ((device char *) dst + i11*nb2 + i10*nb1)) + ind) = temp;
    }
}

template<typename T>
kernel void kernel_get_rows_f(
        device const  void * src0,
        device const  void * src1,
        device       float * dst,
        constant   int64_t & ne00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb02,
        constant   int64_t & ne10,
        constant  uint64_t & nb10,
        constant  uint64_t & nb11,
        constant  uint64_t & nb1,
        constant  uint64_t & nb2,
        uint3                tgpig[[threadgroup_position_in_grid]],
        uint                 tiitg[[thread_index_in_threadgroup]],
        uint3                tptg [[threads_per_threadgroup]]) {
    const int64_t i10 = tgpig.x;
    const int64_t i11 = tgpig.y;

    const int64_t r = ((const device int32_t *) ((const device char *) src1 + i11*nb11 + i10*nb10))[0];

    const int64_t i02 = i11;

    for (int ind = tiitg; ind < ne00; ind += tptg.x) {
        ((      device float *) ((      device char *)  dst + i11*nb2  + i10*nb1))[ind] =
        ((const device T     *) ((const device char *) src0 + i02*nb02 +  r*nb01))[ind];
    }
}

kernel void kernel_get_rows_i32(
        device const  void * src0,
        device const  void * src1,
        device     int32_t * dst,
        constant   int64_t & ne00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb02,
        constant   int64_t & ne10,
        constant  uint64_t & nb10,
        constant  uint64_t & nb11,
        constant  uint64_t & nb1,
        constant  uint64_t & nb2,
        uint3                tgpig[[threadgroup_position_in_grid]],
        uint                 tiitg[[thread_index_in_threadgroup]],
        uint3                tptg [[threads_per_threadgroup]]) {
    const int64_t i10 = tgpig.x;
    const int64_t i11 = tgpig.y;

    const int64_t r = ((const device int32_t *) ((const device char *) src1 + i11*nb11 + i10*nb10))[0];

    const int64_t i02 = i11;

    for (int ind = tiitg; ind < ne00; ind += tptg.x) {
        ((      device int32_t *) ((      device char *) dst  + i11*nb2 + i10*nb1))[ind] =
        ((const device int32_t *) ((const device char *) src0 + i02*nb02 + r*nb01))[ind];
    }
}


#define BLOCK_SIZE_M 64 // 8 simdgroup matrices from matrix A
#define BLOCK_SIZE_N 32 // 4 simdgroup matrices from matrix B
#define BLOCK_SIZE_K 32
#define THREAD_MAT_M 4 // each thread take 4 simdgroup matrices from matrix A
#define THREAD_MAT_N 2 // each thread take 2 simdgroup matrices from matrix B
#define THREAD_PER_BLOCK 128
#define THREAD_PER_ROW 2 // 2 thread for each row in matrix A to load numbers
#define THREAD_PER_COL 4 // 4 thread for each row in matrix B to load numbers
#define SG_MAT_SIZE 64 // simdgroup matrix is of shape 8x8
#define SG_MAT_ROW 8

// each block_q contains 16*nl weights
template<typename T, typename T4x4, typename simdgroup_T8x8, typename block_q, short nl, void (*dequantize_func)(device const block_q *, short, thread T4x4 &)>
kernel void kernel_mul_mm(
        constant ggml_metal_kargs_mul_mm & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    threadgroup T     * sa = (threadgroup T     *)(shmem);
    threadgroup float * sb = (threadgroup float *)(shmem + 4096);

    const int r0 = tgpig.y;
    const int r1 = tgpig.x;
    const int im = tgpig.z;

    // if this block is of 64x32 shape or smaller
    const short n_rows = (args.ne0 - r0*BLOCK_SIZE_M < BLOCK_SIZE_M) ? (args.ne0 - r0*BLOCK_SIZE_M) : BLOCK_SIZE_M;
    const short n_cols = (args.ne1 - r1*BLOCK_SIZE_N < BLOCK_SIZE_N) ? (args.ne1 - r1*BLOCK_SIZE_N) : BLOCK_SIZE_N;

    // a thread shouldn't load data outside of the matrix
    const short thread_row = ((short)tiitg/THREAD_PER_ROW) < n_rows ? ((short)tiitg/THREAD_PER_ROW) : n_rows - 1;
    const short thread_col = ((short)tiitg/THREAD_PER_COL) < n_cols ? ((short)tiitg/THREAD_PER_COL) : n_cols - 1;

    simdgroup_T8x8     ma[4];
    simdgroup_float8x8 mb[2];
    simdgroup_float8x8 mc[8];

    for (short i = 0; i < 8; i++){
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
    }

    short il = (tiitg % THREAD_PER_ROW);

    const int i12 = im%args.ne12;
    const int i13 = im/args.ne12;

    const uint64_t offset0 = (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;
    const short    offset1 = il/nl;

    device const block_q * x = (device const block_q *)(src0
        + args.nb01*(r0*BLOCK_SIZE_M + thread_row) + offset0) + offset1;

    device const float   * y = (device const float   *)(src1
        + args.nb13*i13
        + args.nb12*i12
        + args.nb11*(r1*BLOCK_SIZE_N + thread_col)
        + args.nb10*(BLOCK_SIZE_K / THREAD_PER_COL * (tiitg % THREAD_PER_COL)));

    for (int loop_k = 0; loop_k < args.ne00; loop_k += BLOCK_SIZE_K) {
        // load data and store to threadgroup memory
        T4x4 temp_a;
        dequantize_func(x, il, temp_a);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma unroll(16)
        for (short i = 0; i < 16; i++) {
            *(sa + SG_MAT_SIZE * ((tiitg/THREAD_PER_ROW/8) \
            +                     (tiitg%THREAD_PER_ROW)*16 + (i/8)*8) \
            +                     (tiitg/THREAD_PER_ROW)%8  + (i&7)*8) = temp_a[i/4][i%4];
        }

        *(threadgroup float2x4 *)(sb + 32*8*(tiitg%THREAD_PER_COL) + 8*(tiitg/THREAD_PER_COL)) = *((device float2x4 *) y);

        il = (il + 2 < nl) ? il + 2 : il % 2;
        x  = (il < 2) ? x + (2 + nl - 1)/nl : x;
        y += BLOCK_SIZE_K;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // load matrices from threadgroup memory and conduct outer products
        threadgroup const T     * lsma = (sa + THREAD_MAT_M*SG_MAT_SIZE*(sgitg%2));
        threadgroup const float * lsmb = (sb + THREAD_MAT_N*SG_MAT_SIZE*(sgitg/2));

        #pragma unroll(4)
        for (short ik = 0; ik < BLOCK_SIZE_K/8; ik++) {
            #pragma unroll(4)
            for (short i = 0; i < 4; i++) {
                simdgroup_load(ma[i], lsma + SG_MAT_SIZE * i);
            }

            simdgroup_barrier(mem_flags::mem_none);

            #pragma unroll(2)
            for (short i = 0; i < 2; i++) {
                simdgroup_load(mb[i], lsmb + SG_MAT_SIZE * i);
            }

            #pragma unroll(8)
            for (short i = 0; i < 8; i++){
                simdgroup_multiply_accumulate(mc[i], mb[i/4], ma[i%4], mc[i]);
            }

            lsma += (BLOCK_SIZE_M/SG_MAT_ROW)*SG_MAT_SIZE;
            lsmb += (BLOCK_SIZE_N/SG_MAT_ROW)*SG_MAT_SIZE;
        }
    }

    if ((r0 + 1) * BLOCK_SIZE_M <= args.ne0 && (r1 + 1) * BLOCK_SIZE_N <= args.ne1) {
        device float * C = (device float *) dst +
            (BLOCK_SIZE_M * r0 + 32*(sgitg &  1)) + \
            (BLOCK_SIZE_N * r1 + 16*(sgitg >> 1)) * args.ne0 + im*args.ne1*args.ne0;

        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], C + 8 * (i%4) + 8 * args.ne0 * (i/4), args.ne0);
        }
    } else {
        // block is smaller than 64x32, we should avoid writing data outside of the matrix
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup float * temp_str = ((threadgroup float *) shmem) \
                                     + 32*(sgitg&1) + (16*(sgitg >> 1))*BLOCK_SIZE_M;
        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], temp_str + 8*(i%4) + 8*BLOCK_SIZE_M*(i/4), BLOCK_SIZE_M);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (sgitg == 0) {
            for (int j = tiitg; j < n_cols; j += BLOCK_SIZE_N) {
                device float  * D  = (device float  *) dst + (r0*BLOCK_SIZE_M) + (r1*BLOCK_SIZE_N + j)*args.ne0 + im*args.ne1*args.ne0;
                device float4 * D4 = (device float4 *) D;

                threadgroup float  * C  = temp_str + (j*BLOCK_SIZE_M);
                threadgroup float4 * C4 = (threadgroup float4 *) C;

                int i = 0;
                for (; i < n_rows/4; i++) {
                    *(D4 + i) = *(C4 + i);
                }

                i *= 4;
                for (; i < n_rows; i++) {
                    *(D + i) = *(C + i);
                }
            }
        }
    }
}

// same as kernel_mul_mm_impl, but src1 and dst are accessed via indices stored in rowids
// TODO: this kernel needs to be reimplemented from scratch for better performance
template<typename block_q, short nl, void (*dequantize_func)(device const block_q *, short, thread half4x4 &)>
void kernel_mul_mm_id_impl(
        int32_t  ne00,
        int32_t  ne02,
        uint64_t nb01,
        uint64_t nb02,
        int32_t  ne11,
        int32_t  ne12,
        uint64_t nb10,
        uint64_t nb11,
        uint64_t nb12,
        int32_t  ne0,
        int32_t  ne1,
        int64_t  ne0ne1,
        device   const char * src0,
        device   const char * src1,
        threadgroup ushort2 * rowids,
        device         char * dst,
        threadgroup    char * shmem,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    threadgroup half  * sa = (threadgroup half  *)(shmem);
    threadgroup float * sb = (threadgroup float *)(shmem + 4096);

    const int r0 = tgpig.y;
    const int r1 = tgpig.x;

    if (r1*BLOCK_SIZE_N >= ne1) return;

    // if this block is of 64x32 shape or smaller
    short n_rows = (ne0 - r0 * BLOCK_SIZE_M < BLOCK_SIZE_M) ? (ne0 - r0 * BLOCK_SIZE_M) : BLOCK_SIZE_M;
    short n_cols = (ne1 - r1 * BLOCK_SIZE_N < BLOCK_SIZE_N) ? (ne1 - r1 * BLOCK_SIZE_N) : BLOCK_SIZE_N;

    // a thread shouldn't load data outside of the matrix
    short thread_row = ((short)tiitg/THREAD_PER_ROW) < n_rows ? ((short)tiitg/THREAD_PER_ROW) : n_rows - 1;
    short thread_col = ((short)tiitg/THREAD_PER_COL) < n_cols ? ((short)tiitg/THREAD_PER_COL) : n_cols - 1;

    simdgroup_half8x8  ma[4];
    simdgroup_float8x8 mb[2];
    simdgroup_float8x8 mc[8];
    for (int i = 0; i < 8; i++){
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
    }
    short il = (tiitg % THREAD_PER_ROW);

    ushort offset1 = il/nl;

    threadgroup const auto & id = rowids[r1 * BLOCK_SIZE_N + thread_col];

    device const block_q * x = (device const block_q *)(src0 + (r0 * BLOCK_SIZE_M + thread_row) * nb01) + offset1;
    device const float   * y = (device const float   *)(src1
        + nb12 * id[1]
        + nb11 * (id[0] % ne11)
        + nb10 * (BLOCK_SIZE_K / THREAD_PER_COL * (tiitg % THREAD_PER_COL)));

    for (int loop_k = 0; loop_k < ne00; loop_k += BLOCK_SIZE_K) {
        // load data and store to threadgroup memory
        half4x4 temp_a;
        dequantize_func(x, il, temp_a);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (int i = 0; i < 16; i++) {
            *(sa + SG_MAT_SIZE * ((tiitg / THREAD_PER_ROW / 8) \
            +                     (tiitg % THREAD_PER_ROW) * 16 + (i / 8) * 8) \
            +                     (tiitg / THREAD_PER_ROW) % 8  + (i & 7) * 8) = temp_a[i/4][i%4];
        }

        *(threadgroup float2x4 *)(sb + (tiitg % THREAD_PER_COL) * 8 * 32 + 8 * (tiitg / THREAD_PER_COL)) = *((device float2x4 *)y);

        il = (il + 2 < nl) ? il + 2 : il % 2;
        x  = (il < 2) ? x + (2+nl-1)/nl : x;
        y += BLOCK_SIZE_K;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // load matrices from threadgroup memory and conduct outer products
        threadgroup half  * lsma = (sa + THREAD_MAT_M * SG_MAT_SIZE * (sgitg % 2));
        threadgroup float * lsmb = (sb + THREAD_MAT_N * SG_MAT_SIZE * (sgitg / 2));

        #pragma unroll(BLOCK_SIZE_K/8)
        for (int ik = 0; ik < BLOCK_SIZE_K / 8; ik++) {
            #pragma unroll(4)
            for (int i = 0; i < 4; i++) {
                simdgroup_load(ma[i], lsma + SG_MAT_SIZE * i);
            }
            simdgroup_barrier(mem_flags::mem_none);
            #pragma unroll(2)
            for (int i = 0; i < 2; i++) {
                simdgroup_load(mb[i], lsmb + SG_MAT_SIZE * i);
            }

            lsma += BLOCK_SIZE_M / SG_MAT_ROW * SG_MAT_SIZE;
            lsmb += BLOCK_SIZE_N / SG_MAT_ROW * SG_MAT_SIZE;

            #pragma unroll(8)
            for (int i = 0; i < 8; i++){
                simdgroup_multiply_accumulate(mc[i], mb[i/4], ma[i%4], mc[i]);
            }
        }
    }

    {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup float * temp_str = ((threadgroup float *) shmem) \
                                      + 32 * (sgitg&1) + (16 * (sgitg>>1)) * BLOCK_SIZE_M;
        for (int i = 0; i < 8; i++) {
            simdgroup_store(mc[i], temp_str + 8 * (i%4) + 8 * BLOCK_SIZE_M * (i/4), BLOCK_SIZE_M);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (sgitg == 0) {
            for (int j = tiitg; j < n_cols; j += BLOCK_SIZE_N) {
                threadgroup const auto & jid = rowids[r1 * BLOCK_SIZE_N + j];
                int64_t joff = jid[0]*ne0 + jid[1]*ne0ne1;

                device float  * D  = (device float  *) dst + (r0*BLOCK_SIZE_M) + joff;
                device float4 * D4 = (device float4 *) D;

                threadgroup float  * C  = temp_str + (j*BLOCK_SIZE_M);
                threadgroup float4 * C4 = (threadgroup float4 *) C;

                int i = 0;
                for (; i < n_rows/4; i++) {
                    *(D4 + i) = *(C4 + i);
                }

                i *= 4;
                for (; i < n_rows; i++) {
                    *(D + i) = *(C + i);
                }
            }
        }
    }
}

template<typename block_q, short nl, void (*dequantize_func)(device const block_q *, short, thread half4x4 &)>
kernel void kernel_mul_mm_id(
        constant ggml_metal_kargs_mul_mm_id & args,
        device const char * src0s,
        device const char * src1,
        device       char * dst,
        device const char * ids,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    const int32_t i02 = tgpig.z;

    tgpig.z = 0;

    device const char * src0 = src0s + i02*args.nb02;

    // row indices
    threadgroup ushort2 * rowids = (threadgroup ushort2 *)(shmem + 8192);

    // TODO: parallelize this loop
    int32_t _ne1 = 0;
    for (ushort ii1 = 0; ii1 < args.nei1; ii1++) {
        for (ushort ii0 = 0; ii0 < args.nei0; ii0++) {
            int32_t id = ((device int32_t *) (ids + ii1*args.nbi1))[ii0];
            if (id == i02) {
                if (tiitg == 0) {
                    rowids[_ne1] = ushort2(ii0, ii1);
                }
                _ne1++;
            }
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    kernel_mul_mm_id_impl<block_q, nl, dequantize_func>(
        args.ne00,
        args.ne02,
        args.nb01,
        args.nb02,
        args.ne11,
        args.ne12,
        args.nb10,
        args.nb11,
        args.nb12,
        args.ne0,
        _ne1,
        (int64_t)args.ne0*args.ne1,
        src0,
        src1,
        rowids,
        dst,
        shmem,
        tgpig,
        tiitg,
        sgitg);
}

#define QK_NL 16

//
// get rows
//

typedef decltype(kernel_get_rows_f<float>) get_rows_f_t;

template [[host_name("kernel_get_rows_f32")]]  kernel get_rows_f_t kernel_get_rows_f<float>;
template [[host_name("kernel_get_rows_f16")]]  kernel get_rows_f_t kernel_get_rows_f<half>;
#if defined(GGML_METAL_USE_BF16)
template [[host_name("kernel_get_rows_bf16")]] kernel get_rows_f_t kernel_get_rows_f<bfloat>;
#endif

typedef decltype(kernel_get_rows_q<block_q4_0, 2, dequantize_q4_0>) get_rows_q_t;

template [[host_name("kernel_get_rows_q4_0")]]    kernel get_rows_q_t kernel_get_rows_q<block_q4_0,    2, dequantize_q4_0>;
template [[host_name("kernel_get_rows_q4_1")]]    kernel get_rows_q_t kernel_get_rows_q<block_q4_1,    2, dequantize_q4_1>;
template [[host_name("kernel_get_rows_q5_0")]]    kernel get_rows_q_t kernel_get_rows_q<block_q5_0,    2, dequantize_q5_0>;
template [[host_name("kernel_get_rows_q5_1")]]    kernel get_rows_q_t kernel_get_rows_q<block_q5_1,    2, dequantize_q5_1>;
template [[host_name("kernel_get_rows_q8_0")]]    kernel get_rows_q_t kernel_get_rows_q<block_q8_0,    2, dequantize_q8_0>;
template [[host_name("kernel_get_rows_q2_K")]]    kernel get_rows_q_t kernel_get_rows_q<block_q2_K,    QK_NL, dequantize_q2_K>;
template [[host_name("kernel_get_rows_q3_K")]]    kernel get_rows_q_t kernel_get_rows_q<block_q3_K,    QK_NL, dequantize_q3_K>;
template [[host_name("kernel_get_rows_q4_K")]]    kernel get_rows_q_t kernel_get_rows_q<block_q4_K,    QK_NL, dequantize_q4_K>;
template [[host_name("kernel_get_rows_q5_K")]]    kernel get_rows_q_t kernel_get_rows_q<block_q5_K,    QK_NL, dequantize_q5_K>;
template [[host_name("kernel_get_rows_q6_K")]]    kernel get_rows_q_t kernel_get_rows_q<block_q6_K,    QK_NL, dequantize_q6_K>;
template [[host_name("kernel_get_rows_iq2_xxs")]] kernel get_rows_q_t kernel_get_rows_q<block_iq2_xxs, QK_NL, dequantize_iq2_xxs>;
template [[host_name("kernel_get_rows_iq2_xs")]]  kernel get_rows_q_t kernel_get_rows_q<block_iq2_xs,  QK_NL, dequantize_iq2_xs>;
template [[host_name("kernel_get_rows_iq3_xxs")]] kernel get_rows_q_t kernel_get_rows_q<block_iq3_xxs, QK_NL, dequantize_iq3_xxs>;
template [[host_name("kernel_get_rows_iq3_s")]]   kernel get_rows_q_t kernel_get_rows_q<block_iq3_s,   QK_NL, dequantize_iq3_s>;
template [[host_name("kernel_get_rows_iq2_s")]]   kernel get_rows_q_t kernel_get_rows_q<block_iq2_s,   QK_NL, dequantize_iq2_s>;
template [[host_name("kernel_get_rows_iq1_s")]]   kernel get_rows_q_t kernel_get_rows_q<block_iq1_s,   QK_NL, dequantize_iq1_s>;
template [[host_name("kernel_get_rows_iq1_m")]]   kernel get_rows_q_t kernel_get_rows_q<block_iq1_m,   QK_NL, dequantize_iq1_m>;
template [[host_name("kernel_get_rows_iq4_nl")]]  kernel get_rows_q_t kernel_get_rows_q<block_iq4_nl,  2,     dequantize_iq4_nl>;
template [[host_name("kernel_get_rows_iq4_xs")]]  kernel get_rows_q_t kernel_get_rows_q<block_iq4_xs,  QK_NL, dequantize_iq4_xs>;

//
// matrix-matrix multiplication
//

typedef decltype(kernel_mul_mm<half, half4x4, simdgroup_half8x8, float4x4, 1, dequantize_f32>) mat_mm_t;

template [[host_name("kernel_mul_mm_f32_f32")]]     kernel mat_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   float4x4,      1,     dequantize_f32>;
template [[host_name("kernel_mul_mm_f16_f32")]]     kernel mat_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half4x4,       1,     dequantize_f16>;
#if defined(GGML_METAL_USE_BF16)
template [[host_name("kernel_mul_mm_bf16_f32")]]    kernel mat_mm_t kernel_mul_mm<bfloat, bfloat4x4, simdgroup_bfloat8x8, bfloat4x4,     1,     dequantize_bf16>;
#endif
template [[host_name("kernel_mul_mm_q4_0_f32")]]    kernel mat_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   block_q4_0,    2,     dequantize_q4_0>;
template [[host_name("kernel_mul_mm_q4_1_f32")]]    kernel mat_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   block_q4_1,    2,     dequantize_q4_1>;
template [[host_name("kernel_mul_mm_q5_0_f32")]]    kernel mat_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   block_q5_0,    2,     dequantize_q5_0>;
template [[host_name("kernel_mul_mm_q5_1_f32")]]    kernel mat_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   block_q5_1,    2,     dequantize_q5_1>;
template [[host_name("kernel_mul_mm_q8_0_f32")]]    kernel mat_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   block_q8_0,    2,     dequantize_q8_0>;
template [[host_name("kernel_mul_mm_q2_K_f32")]]    kernel mat_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   block_q2_K,    QK_NL, dequantize_q2_K>;
template [[host_name("kernel_mul_mm_q3_K_f32")]]    kernel mat_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   block_q3_K,    QK_NL, dequantize_q3_K>;
template [[host_name("kernel_mul_mm_q4_K_f32")]]    kernel mat_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   block_q4_K,    QK_NL, dequantize_q4_K>;
template [[host_name("kernel_mul_mm_q5_K_f32")]]    kernel mat_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   block_q5_K,    QK_NL, dequantize_q5_K>;
template [[host_name("kernel_mul_mm_q6_K_f32")]]    kernel mat_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   block_q6_K,    QK_NL, dequantize_q6_K>;
template [[host_name("kernel_mul_mm_iq2_xxs_f32")]] kernel mat_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   block_iq2_xxs, QK_NL, dequantize_iq2_xxs>;
template [[host_name("kernel_mul_mm_iq2_xs_f32")]]  kernel mat_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   block_iq2_xs,  QK_NL, dequantize_iq2_xs>;
template [[host_name("kernel_mul_mm_iq3_xxs_f32")]] kernel mat_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   block_iq3_xxs, QK_NL, dequantize_iq3_xxs>;
template [[host_name("kernel_mul_mm_iq3_s_f32")]]   kernel mat_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   block_iq3_s,   QK_NL, dequantize_iq3_s>;
template [[host_name("kernel_mul_mm_iq2_s_f32")]]   kernel mat_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   block_iq2_s,   QK_NL, dequantize_iq2_s>;
template [[host_name("kernel_mul_mm_iq1_s_f32")]]   kernel mat_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   block_iq1_s,   QK_NL, dequantize_iq1_s>;
template [[host_name("kernel_mul_mm_iq1_m_f32")]]   kernel mat_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   block_iq1_m,   QK_NL, dequantize_iq1_m>;
template [[host_name("kernel_mul_mm_iq4_nl_f32")]]  kernel mat_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   block_iq4_nl,  2,     dequantize_iq4_nl>;
template [[host_name("kernel_mul_mm_iq4_xs_f32")]]  kernel mat_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   block_iq4_xs,  QK_NL, dequantize_iq4_xs>;

//
// indirect matrix-matrix multiplication
//

typedef decltype(kernel_mul_mm_id<float4x4, 1, dequantize_f32>) mat_mm_id_t;

template [[host_name("kernel_mul_mm_id_f32_f32")]]     kernel mat_mm_id_t kernel_mul_mm_id<float4x4,      1,     dequantize_f32>;
template [[host_name("kernel_mul_mm_id_f16_f32")]]     kernel mat_mm_id_t kernel_mul_mm_id<half4x4,       1,     dequantize_f16>;
#if defined(GGML_METAL_USE_BF16)
template [[host_name("kernel_mul_mm_id_bf16_f32")]]    kernel mat_mm_id_t kernel_mul_mm_id<bfloat4x4,     1,     dequantize_bf16>;
#endif
template [[host_name("kernel_mul_mm_id_q4_0_f32")]]    kernel mat_mm_id_t kernel_mul_mm_id<block_q4_0,    2,     dequantize_q4_0>;
template [[host_name("kernel_mul_mm_id_q4_1_f32")]]    kernel mat_mm_id_t kernel_mul_mm_id<block_q4_1,    2,     dequantize_q4_1>;
template [[host_name("kernel_mul_mm_id_q5_0_f32")]]    kernel mat_mm_id_t kernel_mul_mm_id<block_q5_0,    2,     dequantize_q5_0>;
template [[host_name("kernel_mul_mm_id_q5_1_f32")]]    kernel mat_mm_id_t kernel_mul_mm_id<block_q5_1,    2,     dequantize_q5_1>;
template [[host_name("kernel_mul_mm_id_q8_0_f32")]]    kernel mat_mm_id_t kernel_mul_mm_id<block_q8_0,    2,     dequantize_q8_0>;
template [[host_name("kernel_mul_mm_id_q2_K_f32")]]    kernel mat_mm_id_t kernel_mul_mm_id<block_q2_K,    QK_NL, dequantize_q2_K>;
template [[host_name("kernel_mul_mm_id_q3_K_f32")]]    kernel mat_mm_id_t kernel_mul_mm_id<block_q3_K,    QK_NL, dequantize_q3_K>;
template [[host_name("kernel_mul_mm_id_q4_K_f32")]]    kernel mat_mm_id_t kernel_mul_mm_id<block_q4_K,    QK_NL, dequantize_q4_K>;
template [[host_name("kernel_mul_mm_id_q5_K_f32")]]    kernel mat_mm_id_t kernel_mul_mm_id<block_q5_K,    QK_NL, dequantize_q5_K>;
template [[host_name("kernel_mul_mm_id_q6_K_f32")]]    kernel mat_mm_id_t kernel_mul_mm_id<block_q6_K,    QK_NL, dequantize_q6_K>;
template [[host_name("kernel_mul_mm_id_iq2_xxs_f32")]] kernel mat_mm_id_t kernel_mul_mm_id<block_iq2_xxs, QK_NL, dequantize_iq2_xxs>;
template [[host_name("kernel_mul_mm_id_iq2_xs_f32")]]  kernel mat_mm_id_t kernel_mul_mm_id<block_iq2_xs,  QK_NL, dequantize_iq2_xs>;
template [[host_name("kernel_mul_mm_id_iq3_xxs_f32")]] kernel mat_mm_id_t kernel_mul_mm_id<block_iq3_xxs, QK_NL, dequantize_iq3_xxs>;
template [[host_name("kernel_mul_mm_id_iq3_s_f32")]]   kernel mat_mm_id_t kernel_mul_mm_id<block_iq3_s,   QK_NL, dequantize_iq3_s>;
template [[host_name("kernel_mul_mm_id_iq2_s_f32")]]   kernel mat_mm_id_t kernel_mul_mm_id<block_iq2_s,   QK_NL, dequantize_iq2_s>;
template [[host_name("kernel_mul_mm_id_iq1_s_f32")]]   kernel mat_mm_id_t kernel_mul_mm_id<block_iq1_s,   QK_NL, dequantize_iq1_s>;
template [[host_name("kernel_mul_mm_id_iq1_m_f32")]]   kernel mat_mm_id_t kernel_mul_mm_id<block_iq1_m,   QK_NL, dequantize_iq1_m>;
template [[host_name("kernel_mul_mm_id_iq4_nl_f32")]]  kernel mat_mm_id_t kernel_mul_mm_id<block_iq4_nl,  2,     dequantize_iq4_nl>;
template [[host_name("kernel_mul_mm_id_iq4_xs_f32")]]  kernel mat_mm_id_t kernel_mul_mm_id<block_iq4_xs,  QK_NL, dequantize_iq4_xs>;

//
// matrix-vector multiplication
//

typedef void (kernel_mul_mv_impl_t)(
        ggml_metal_kargs_mul_mv args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig,
        ushort tiisg);

typedef void (kernel_mul_mv2_impl_t)(
        ggml_metal_kargs_mul_mv args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg);

template<kernel_mul_mv_impl_t impl_fn>
void mmv_fn(
        ggml_metal_kargs_mul_mv args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiitg,
        ushort tiisg,
        ushort sgitg) {
    impl_fn(args, src0, src1, dst, tgpig, tiisg);
}

template<kernel_mul_mv2_impl_t impl_fn>
void mmv_fn(
        ggml_metal_kargs_mul_mv args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiitg,
        ushort tiisg,
        ushort sgitg) {
    impl_fn(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

typedef decltype(mmv_fn<kernel_mul_mv_impl<half, half4, half, half4, ggml_metal_kargs_mul_mv>>) mul_mv_impl_fn_t;

template<mul_mv_impl_fn_t impl_fn>
kernel void kernel_mul_mv_id(
        constant ggml_metal_kargs_mul_mv_id & args,
        device const char * src0s,
        device const char * src1,
        device       char * dst,
        device const char * ids,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const int iid1 = tgpig.z/args.nei0;
    const int idx  = tgpig.z%args.nei0;

    tgpig.z = 0;

    const int32_t i02 = ((device const int32_t *) (ids + iid1*args.nbi1))[idx];

    const int64_t i11 = idx % args.ne11;
    const int64_t i12 = iid1;

    const int64_t i1 = idx;
    const int64_t i2 = i12;

    device const char * src0_cur = src0s + i02*args.nb02;
    device const char * src1_cur = src1  + i11*args.nb11 + i12*args.nb12;

    device char * dst_cur = dst + (i1*args.ne0 + i2*args.ne1*args.ne0)*sizeof(float);

    ggml_metal_kargs_mul_mv args0 = {
        /*.ne00 =*/ args.ne00,
        /*.ne01 =*/ args.ne01,
        /*.ne02 =*/ 1, // args.ne02,
        /*.nb00 =*/ args.nb00,
        /*.nb01 =*/ args.nb01,
        /*.nb02 =*/ args.nb02,
        /*.nb03 =*/ args.nb02, // args.ne02 == 1
        /*.ne10 =*/ args.ne10,
        /*.ne11 =*/ 1, // args.ne11,
        /*.ne12 =*/ 1, // args.ne12,
        /*.nb10 =*/ args.nb10,
        /*.nb11 =*/ args.nb11,
        /*.nb12 =*/ args.nb12,
        /*.nb13 =*/ args.nb12, // ne12 == 1
        /*.ne0  =*/ args.ne0,
        /*.ne1  =*/ 1, // args.ne1,
        /*.r2   =*/ 1,
        /*.r3   =*/ 1,
    };

    impl_fn(
        args0,
        /* src0 */ src0_cur,
        /* src1 */ src1_cur,
        /* dst  */ dst_cur,
        shmem,
        tgpig,
        tiitg,
        tiisg,
        sgitg);
}

typedef decltype(kernel_mul_mv_id<mmv_fn<kernel_mul_mv_impl<float, float4, float, float4>>>) kernel_mul_mv_id_t;

template [[host_name("kernel_mul_mv_id_f32_f32")]]     kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_impl<float, float4, float, float4>>>;
template [[host_name("kernel_mul_mv_id_f16_f32")]]     kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_impl<half, half4, float, float4>>>;
#if defined(GGML_METAL_USE_BF16)
template [[host_name("kernel_mul_mv_id_bf16_f32")]]    kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_impl<bfloat, bfloat4, float, float4>>>;
#endif
template [[host_name("kernel_mul_mv_id_q8_0_f32")]]    kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_q8_0_f32_impl>>;
template [[host_name("kernel_mul_mv_id_q4_0_f32")]]    kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<mul_vec_q_n_f32_impl<block_q4_0, N_DST, N_SIMDGROUP, N_SIMDWIDTH>>>;
template [[host_name("kernel_mul_mv_id_q4_1_f32")]]    kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<mul_vec_q_n_f32_impl<block_q4_1, N_DST, N_SIMDGROUP, N_SIMDWIDTH>>>;
template [[host_name("kernel_mul_mv_id_q5_0_f32")]]    kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<mul_vec_q_n_f32_impl<block_q5_0, N_DST, N_SIMDGROUP, N_SIMDWIDTH>>>;
template [[host_name("kernel_mul_mv_id_q5_1_f32")]]    kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<mul_vec_q_n_f32_impl<block_q5_1, N_DST, N_SIMDGROUP, N_SIMDWIDTH>>>;
template [[host_name("kernel_mul_mv_id_q2_K_f32")]]    kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_q2_K_f32_impl>>;
template [[host_name("kernel_mul_mv_id_q3_K_f32")]]    kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_q3_K_f32_impl>>;
template [[host_name("kernel_mul_mv_id_q4_K_f32")]]    kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_q4_K_f32_impl>>;
template [[host_name("kernel_mul_mv_id_q5_K_f32")]]    kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_q5_K_f32_impl>>;
template [[host_name("kernel_mul_mv_id_q6_K_f32")]]    kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_q6_K_f32_impl>>;
template [[host_name("kernel_mul_mv_id_iq1_s_f32")]]   kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_iq1_s_f32_impl>>;
template [[host_name("kernel_mul_mv_id_iq1_m_f32")]]   kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_iq1_m_f32_impl>>;
template [[host_name("kernel_mul_mv_id_iq2_xxs_f32")]] kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_iq2_xxs_f32_impl>>;
template [[host_name("kernel_mul_mv_id_iq2_xs_f32")]]  kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_iq2_xs_f32_impl>>;
template [[host_name("kernel_mul_mv_id_iq3_xxs_f32")]] kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_iq3_xxs_f32_impl>>;
template [[host_name("kernel_mul_mv_id_iq3_s_f32")]]   kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_iq3_s_f32_impl>>;
template [[host_name("kernel_mul_mv_id_iq2_s_f32")]]   kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_iq2_s_f32_impl>>;
template [[host_name("kernel_mul_mv_id_iq4_nl_f32")]]  kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_iq4_nl_f32_impl>>;
template [[host_name("kernel_mul_mv_id_iq4_xs_f32")]]  kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_iq4_xs_f32_impl>>;

kernel void kernel_pool_2d_max_f32(
        device  const float * src0,
        device        float * dst,
        constant    int32_t & k0,
        constant    int32_t & k1,
        constant    int32_t & s0,
        constant    int32_t & s1,
        constant    int32_t & p0,
        constant    int32_t & p1,
        constant    int64_t & IH,
        constant    int64_t & IW,
        constant    int64_t & OH,
        constant    int64_t & OW,
        constant    int64_t & parallel_elements,
        uint        gid[[thread_position_in_grid]]) {

    if (gid >= parallel_elements) {
        return;
    }

    const int idx = gid;
    const int I_HW = IH * IW;
    const int O_HW = OH * OW;
    const int nc = idx / O_HW;
    const int cur_oh = idx % O_HW / OW;
    const int cur_ow = idx % O_HW % OW;

    device const float * i_ptr = src0 + nc * I_HW;
    device       float * o_ptr = dst  + nc * O_HW;

    const int start_h = cur_oh * s1 - p1;
    const int bh = MAX(0,  start_h);
    const int eh = MIN(IH, start_h + k1);
    const int start_w = cur_ow * s0 - p0;
    const int bw = MAX(0,  start_w);
    const int ew = MIN(IW, start_w + k0);

    float res = -INFINITY;

    for (int i = bh; i < eh; i += 1) {
        for (int j = bw; j < ew; j += 1) {
            res = MAX(res, i_ptr[i * IW + j]);
        }
    }

    o_ptr[cur_oh * OW + cur_ow] = res;
}

kernel void kernel_pool_2d_avg_f32(
        device  const float * src0,
        device        float * dst,
        constant    int32_t & k0,
        constant    int32_t & k1,
        constant    int32_t & s0,
        constant    int32_t & s1,
        constant    int32_t & p0,
        constant    int32_t & p1,
        constant    int64_t & IH,
        constant    int64_t & IW,
        constant    int64_t & OH,
        constant    int64_t & OW,
        constant    int64_t & parallel_elements,
        uint        gid[[thread_position_in_grid]]) {

    if (gid >= parallel_elements) {
        return;
    }

    const int idx = gid;
    const int I_HW = IH * IW;
    const int O_HW = OH * OW;
    const int nc = idx / O_HW;
    const int cur_oh = idx % O_HW / OW;
    const int cur_ow = idx % O_HW % OW;

    device const float * i_ptr = src0 + nc * I_HW;
    device       float * o_ptr = dst  + nc * O_HW;

    const int start_h = cur_oh * s1 - p1;
    const int bh = MAX(0,  start_h);
    const int eh = MIN(IH, start_h + k1);
    const int start_w = cur_ow * s0 - p0;
    const int bw = MAX(0,  start_w);
    const int ew = MIN(IW, start_w + k0);
    // const float scale = 1. / ((eh - bh) * (ew - bw));
    const float scale = 1. / (k0 * k1);

    float res = 0;

    for (int i = bh; i < eh; i += 1) {
        for (int j = bw; j < ew; j += 1) {
            float cur = i_ptr[i * IW + j];
            res += cur * scale;
        }
    }

    o_ptr[cur_oh * OW + cur_ow] = res;
}
