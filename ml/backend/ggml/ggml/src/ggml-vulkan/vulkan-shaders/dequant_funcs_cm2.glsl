
#include "types.glsl"

layout(buffer_reference, std430, buffer_reference_align = 16) buffer decodeBufF32 {
   vec4 block;
};

float16_t dequantFuncF32(const in decodeBufF32 bl, const in uint blockCoords[2], const in uint coordInBlock[2])
{
    const vec4 v = bl.block;
    const uint idx = coordInBlock[1];
    const f16vec4 vf16 = f16vec4(v);
    return vf16[idx];
}

layout(buffer_reference, std430, buffer_reference_align = 2) buffer decodeBufQ4_0 {
   block_q4_0_packed16 block;
};

float16_t dequantFuncQ4_0(const in decodeBufQ4_0 bl, const in uint blockCoords[2], const in uint coordInBlock[2])
{
    const float16_t d = bl.block.d;
    const uint idx = coordInBlock[1];
    const uint shift = (idx & 0x10) >> 2;
    uint32_t qs = uint32_t(bl.block.qs[(idx & 0xE) >> 1]);
    qs >>= shift;
    qs &= 0x0F0F;
    qs = unpack8(qs)[idx & 1];
    float16_t ret = (float16_t(qs) - float16_t(8)) * d;
    return ret;
}

layout(buffer_reference, std430, buffer_reference_align = 4) buffer decodeBufQ4_1 {
   block_q4_1 block;
};

float16_t dequantFuncQ4_1(const in decodeBufQ4_1 bl, const in uint blockCoords[2], const in uint coordInBlock[2])
{
    const float16_t d = bl.block.d;
    const float16_t m = bl.block.m;
    const uint idx = coordInBlock[1];
    const uint iqs = idx & 0xF;
    const uint shift = (idx & 0x10) >> 2;
    uint32_t qs = bl.block.qs[iqs];
    qs >>= shift;
    qs &= 0xF;
    float16_t ret = float16_t(qs) * d + m;
    return ret;
}

layout(buffer_reference, std430, buffer_reference_align = 2) buffer decodeBufQ5_0 {
   block_q5_0 block;
};

float16_t dequantFuncQ5_0(const in decodeBufQ5_0 bl, const in uint blockCoords[2], const in uint coordInBlock[2])
{
    const float16_t d = bl.block.d;
    const uint idx = coordInBlock[1];
    const uint iqs = idx & 0xF;

    const uint uint_qh = uint(bl.block.qh[1]) << 16 | bl.block.qh[0];
    const uint qh = ((uint_qh >> idx) << 4) & 0x10;

    const uint shift = (idx & 0x10) >> 2;
    uint32_t qs = bl.block.qs[iqs];
    qs >>= shift;
    qs &= 0xF;

    float16_t ret = (float16_t(qs | qh) - float16_t(16)) * d;
    return ret;
}

layout(buffer_reference, std430, buffer_reference_align = 8) buffer decodeBufQ5_1 {
   block_q5_1 block;
};

float16_t dequantFuncQ5_1(const in decodeBufQ5_1 bl, const in uint blockCoords[2], const in uint coordInBlock[2])
{
    const float16_t d = bl.block.d;
    const float16_t m = bl.block.m;
    const uint idx = coordInBlock[1];
    const uint iqs = idx & 0xF;

    const uint uint_qh = bl.block.qh;
    const uint qh = ((uint_qh >> idx) << 4) & 0x10;

    const uint shift = (idx & 0x10) >> 2;
    uint32_t qs = bl.block.qs[iqs];
    qs >>= shift;
    qs &= 0xF;

    float16_t ret = float16_t(qs | qh) * d + m;
    return ret;
}

layout(buffer_reference, std430, buffer_reference_align = 2) buffer decodeBufQ8_0 {
   block_q8_0_packed16 block;
};

float16_t dequantFuncQ8_0(const in decodeBufQ8_0 bl, const in uint blockCoords[2], const in uint coordInBlock[2])
{
    const float16_t d = bl.block.d;
    const uint idx = coordInBlock[1];
    const uint iqs = idx;

    // Load 16b and select the byte for this element
    int32_t qs = unpack8(bl.block.qs[(iqs & 0x1E) >> 1])[iqs & 1];
    float16_t ret = float16_t(qs) * d;
    return ret;
}

layout(buffer_reference, std430, buffer_reference_align = 4) buffer decodeBufQ2_K {
   block_q2_K block;
};

layout(buffer_reference, std430, buffer_reference_align = 16) buffer decodeBufQ2_K_packed16 {
   block_q2_K_packed16 block;
};

float16_t dequantFuncQ2_K(const in decodeBufQ2_K bl, const in uint blockCoords[2], const in uint coordInBlock[2])
{
    decodeBufQ2_K_packed16 bl16 = decodeBufQ2_K_packed16(bl);
    const f16vec2 dm = bl.block.dm;
    const uint idx = coordInBlock[1];

    const uint scalesi = (idx & 0xF0) >> 4;             // 0..15
    const uint qsshift = (idx & 0x60) >> 4;             // 0,2,4,6

    uint qs = uint32_t(bl16.block.qs[((idx & 0x80) >> 3) + ((idx & 0x1E) >> 1)]);
    qs = (qs >> qsshift) & 0x0303;
    qs = unpack8(qs)[idx & 1];

    const uint scales = bl.block.scales[scalesi];
    float16_t ret = dm.x * float16_t(scales & 0xF) * float16_t(qs) - dm.y * float16_t(scales >> 4);
    return ret;
}

layout(buffer_reference, std430, buffer_reference_align = 2) buffer decodeBufQ3_K {
   block_q3_K block;
};

float16_t dequantFuncQ3_K(const in decodeBufQ3_K bl, const in uint blockCoords[2], const in uint coordInBlock[2])
{
    const uint idx = coordInBlock[1];
    const uint iqs = idx;

    const uint n = iqs / 128;                    // 0,1
    const uint qsi = n * 32 + (iqs % 32);        // 0..63
    const uint hmi =          (iqs % 32);        // 0..31
    const uint j = (iqs % 128) / 8;              // 0..15
    const uint is = iqs / 16;                    // 0..15
    const uint halfsplit = ((iqs % 128) / 32);   // 0,1,2,3
    const uint qsshift = halfsplit * 2;          // 0,2,4,6
    const uint m = 1 << (4 * n + halfsplit);     // 1,2,4,8,16,32,64,128

    uint32_t scaleidx0 = (is < 8) ? is : (is-8);
    uint32_t scaleidx0shift = (is < 8) ? 0 : 4;
    uint32_t scaleidx1 = is + 8 - (is/4)*4;
    uint32_t scaleidx1shift = (is/4)*2;

    const int8_t us = int8_t(((bl.block.scales[scaleidx0] >> scaleidx0shift) & 0xF) | (((bl.block.scales[scaleidx1] >> scaleidx1shift) & 3) << 4));

    const float16_t dl = bl.block.d * float16_t(us - 32);

    float16_t ret = dl * float16_t(int8_t((bl.block.qs[qsi    ] >> qsshift) & 3) - (((bl.block.hmask[hmi    ] & m) != 0) ? 0 : 4));

    return ret;
}

layout(buffer_reference, std430, buffer_reference_align = 16) buffer decodeBufQ4_K {
   block_q4_K block;
};

layout(buffer_reference, std430, buffer_reference_align = 16) buffer decodeBufQ4_K_packed16 {
   block_q4_K_packed16 block;
};

layout(buffer_reference, std430, buffer_reference_align = 16) buffer decodeBufQ4_K_packed128 {
   block_q4_K_packed128 block;
};

#if defined(IS_MUL_MM2)

// For Q4_K and Q5_K in the mat-mul shader, we decode a tile's worth of scales
// into shared memory and then process the whole tile using those scales.
// There is a fetch function that loads into private variables and then a store
// function that stores into shared memory.
// Q4_K and Q5_K have the same encoding of scales, so everything is shared except
// the part that fetches from the structure (which has a different block layout).
#if defined(DATA_A_Q4_K) || defined(DATA_A_Q5_K)
const uint shAscales_stride = (BM + 2);
// 1 scale per 32 elements -> 8 scales per block, per row
shared vec2 shAscales[8 * shAscales_stride];
uvec4 row_v;
#endif

#if defined(DATA_A_Q4_K)
layout (binding = 0) readonly buffer A_Q4_K_128 {block_q4_K_packed128 data_a_q4_k_packed128[];};

void fetch_scalesQ4_K(uint ir_BM, uint pos_a, uint stride_a, uint block_k, uint tid, bool in_bounds)
{
    uint tids_per_row = BLOCK_SIZE / BM;
    uint is_per_tid = 8 / tids_per_row;
    uint is_start = is_per_tid * (tid % tids_per_row);
    uint tid_row = tid / tids_per_row;

    uint row = ir_BM + tid_row;
    uint block_index = pos_a + row * stride_a + (block_k / QUANT_K);
    if (in_bounds || row < p.M) {
        row_v = data_a_q4_k_packed128[block_index].q4k[0];
    }
}
#endif
#if defined(DATA_A_Q5_K)
layout (binding = 0) readonly buffer A_Q5_K_128 {block_q5_K_packed128 data_a_q5_k_packed128[];};

void fetch_scalesQ5_K(uint ir_BM, uint pos_a, uint stride_a, uint block_k, uint tid, bool in_bounds)
{
    uint tids_per_row = BLOCK_SIZE / BM;
    uint is_per_tid = 8 / tids_per_row;
    uint is_start = is_per_tid * (tid % tids_per_row);
    uint tid_row = tid / tids_per_row;

    uint row = ir_BM + tid_row;
    uint block_index = pos_a + row * stride_a + (block_k / QUANT_K);
    if (in_bounds || row < p.M) {
        row_v = data_a_q5_k_packed128[block_index].q5k[0];
    }
}
#endif

#if defined(DATA_A_Q4_K) || defined(DATA_A_Q5_K)
void store_scalesQ4_K(uint tid)
{
    barrier();

    uint tids_per_row = BLOCK_SIZE / BM;
    uint is_per_tid = 8 / tids_per_row;
    uint is_start = is_per_tid * (tid % tids_per_row);
    uint tid_row = tid / tids_per_row;

    [[unroll]] for (uint idx = 0; idx < is_per_tid; ++idx) {
        uint is = idx + is_start;
        uvec4 v = row_v;
        const vec2 loadd = vec2(unpackFloat2x16(v.x));

        uint32_t sc;
        uint32_t mbyte;

        uint32_t scale0 = v.y;
        uint32_t scale4 = v.z;
        uint32_t scale8 = v.w;

        uint32_t sc_lo = scale0;
        uint32_t mb_lo = scale4;
        uint32_t sc_hi = (scale8 & 0x0F0F0F0F) | ((scale0 & 0xC0C0C0C0) >> 2);
        uint32_t mb_hi = ((scale8 & 0xF0F0F0F0) >> 4) | ((scale4 & 0xC0C0C0C0) >> 2);

        sc = is < 4 ? sc_lo : sc_hi;
        mbyte = is < 4 ? mb_lo : mb_hi;
        sc = sc >> (8 * (is & 3));
        mbyte = mbyte >> (8 * (is & 3));
        sc &= 0x3F;
        mbyte &= 0x3F;

        const float d = loadd.x * float(sc);
        const float m = loadd.y * float(mbyte);
        shAscales[is * shAscales_stride + tid_row] = vec2(d,m);
    }

    barrier();
}
#endif

#endif

float16_t dequantFuncQ4_K(const in decodeBufQ4_K bl, const in uint blockCoords[2], const in uint coordInBlock[2])
{
    decodeBufQ4_K_packed16 bl16 = decodeBufQ4_K_packed16(bl);
    decodeBufQ4_K_packed128 bl128 = decodeBufQ4_K_packed128(bl);
    const uint idx = coordInBlock[1];

    const uint b = (idx & 0x20) >> 5;            // 0,1
    const uint is = (idx & 0xE0) >> 5;         // 0..7

#if defined(IS_MUL_MM2) && defined(DATA_A_Q4_K)
    vec2 v = shAscales[is * shAscales_stride + (blockCoords[0] % BM)];
    float d = v.x;
    float m = v.y;
#else
    uvec4 v = bl128.block.q4k[0];
    const vec2 loadd = vec2(unpackFloat2x16(v.x));

    uint32_t sc;
    uint32_t mbyte;

    uint32_t scale0 = v.y;
    uint32_t scale4 = v.z;
    uint32_t scale8 = v.w;

    uint32_t sc_lo = scale0;
    uint32_t mb_lo = scale4;
    uint32_t sc_hi = (scale8 & 0x0F0F0F0F) | ((scale0 & 0xC0C0C0C0) >> 2);
    uint32_t mb_hi = ((scale8 & 0xF0F0F0F0) >> 4) | ((scale4 & 0xC0C0C0C0) >> 2);

    sc = is < 4 ? sc_lo : sc_hi;
    mbyte = is < 4 ? mb_lo : mb_hi;
    sc = sc >> (8 * (is & 3));
    mbyte = mbyte >> (8 * (is & 3));
    sc &= 0x3F;
    mbyte &= 0x3F;

    const float d = loadd.x * float(sc);
    const float m = loadd.y * float(mbyte);
#endif

    uint qs = uint32_t(bl16.block.qs[((idx & 0xC0) >> 2) + ((idx & 0x1E) >> 1)]);
    qs = (qs >> (b * 4 + 8 * (idx & 1))) & 0xF;

    float ret = d * float(qs) - m;

    return float16_t(ret);
}

layout(buffer_reference, std430, buffer_reference_align = 16) buffer decodeBufQ5_K {
   block_q5_K block;
};

layout(buffer_reference, std430, buffer_reference_align = 16) buffer decodeBufQ5_K_packed16 {
   block_q5_K_packed16 block;
};

layout(buffer_reference, std430, buffer_reference_align = 16) buffer decodeBufQ5_K_packed128 {
   block_q5_K_packed128 block;
};

float16_t dequantFuncQ5_K(const in decodeBufQ5_K bl, const in uint blockCoords[2], const in uint coordInBlock[2])
{
    decodeBufQ5_K_packed16 bl16 = decodeBufQ5_K_packed16(bl);
    decodeBufQ5_K_packed128 bl128 = decodeBufQ5_K_packed128(bl);
    const uint idx = coordInBlock[1];

    const uint b = (idx & 0x20) >> 5;          // 0,1
    const uint is = (idx & 0xE0) >> 5;         // 0..7

#if defined(IS_MUL_MM2) && defined(DATA_A_Q5_K)
    vec2 v = shAscales[is * shAscales_stride + (blockCoords[0] % BM)];
    float d = v.x;
    float m = v.y;
#else
    uvec4 v = bl128.block.q5k[0];

    const f16vec2 loadd = unpackFloat2x16(v.x);

    uint32_t sc;
    uint32_t mbyte;

    uint32_t scale0 = v.y;
    uint32_t scale4 = v.z;
    uint32_t scale8 = v.w;

    uint32_t sc_lo = scale0;
    uint32_t mb_lo = scale4;
    uint32_t sc_hi = (scale8 & 0x0F0F0F0F) | ((scale0 & 0xC0C0C0C0) >> 2);
    uint32_t mb_hi = ((scale8 & 0xF0F0F0F0) >> 4) | ((scale4 & 0xC0C0C0C0) >> 2);

    sc = is < 4 ? sc_lo : sc_hi;
    mbyte = is < 4 ? mb_lo : mb_hi;
    sc = sc >> (8 * (is & 3));
    mbyte = mbyte >> (8 * (is & 3));
    sc &= 0x3F;
    mbyte &= 0x3F;

    const float16_t d = loadd.x * float16_t(sc);
    const float16_t m = loadd.y * float16_t(mbyte);
#endif

    uint qh = uint32_t(bl16.block.qh[(idx & 0x1E) >> 1]);
    qh = ((qh >> is) & 0x101) << 4;

    uint qs = uint32_t(bl16.block.qs[((idx & 0xC0) >> 2) + ((idx & 0x1E) >> 1)]);
    qs = (qs >> (b * 4)) & 0x0F0F;
    qs = unpack8(qs | qh)[idx & 1];

    float ret = d * float(qs) - m;

    return float16_t(ret);
}

layout(buffer_reference, std430, buffer_reference_align = 2) buffer decodeBufQ6_K {
   block_q6_K block;
};

layout(buffer_reference, std430, buffer_reference_align = 16) buffer decodeBufQ6_K_packed16 {
   block_q6_K_packed16 block;
};

float16_t dequantFuncQ6_K(const in decodeBufQ6_K bl, const in uint blockCoords[2], const in uint coordInBlock[2])
{
    decodeBufQ6_K_packed16 bl16 = decodeBufQ6_K_packed16(bl);
    const uint idx = coordInBlock[1];

    const uint b = (idx & 0x40) >> 6;           // 0,1
    const uint qhshift = (idx & 0x60) >> 4;    // 0,2,4,6
    const uint is = (idx & 0xF0) >> 4;          // 0..15

    const float16_t dscale = bl.block.d * float16_t(bl.block.scales[is]);

    uint ql = uint32_t(bl16.block.ql[((idx & 0x80) >> 2) + ((idx & 0x3E) >> 1)]);
    ql = (ql >> (b * 4)) & 0x0F0F;

    uint qh = uint32_t(bl16.block.qh[((idx & 0x80) >> 3) + ((idx & 0x1E) >> 1)]);
    qh = ((qh >> qhshift) & 0x0303) << 4;

    int q = unpack8(ql | qh)[idx & 1];

    float16_t ret = dscale * float16_t(q - 32);

    return ret;
}

#if defined(DATA_A_IQ1_S)
layout(buffer_reference, std430, buffer_reference_align = 2) buffer decodeBufIQ1_S {
   block_iq1_s block;
};

float16_t dequantFuncIQ1_S(const in decodeBufIQ1_S bl, const in uint blockCoords[2], const in uint coordInBlock[2])
{
    const float16_t d = bl.block.d;
    const uint idx = coordInBlock[1];

    const uint ib32 = (idx & 0xE0) >> 5;
    const uint ib8 = (idx & 0xF8) >> 3;

    const uint qh = bl.block.qh[ib32];
    const uint qs = bl.block.qs[ib8];
    const float dl = d * float(2 * bitfieldExtract(qh, 12, 3) + 1);
    const float delta = ((qh & 0x8000) != 0) ? -IQ1S_DELTA : IQ1S_DELTA;
    const uint grid = iq1s_grid[qs | (bitfieldExtract(qh, 3 * int(ib8 & 3), 3) << 8)];

    float16_t ret = float16_t(dl) * (float16_t(bitfieldExtract(int(grid), 2 * int(idx % 8), 2)) + float16_t(delta));
    return ret;
}
#endif

#if defined(DATA_A_IQ1_M)
layout(buffer_reference, std430, buffer_reference_align = 2) buffer decodeBufIQ1_M {
   block_iq1_m block;
};

layout(buffer_reference, std430, buffer_reference_align = 8) buffer decodeBufIQ1_M_packed64 {
   block_iq1_m_packed64 block;
};

float16_t dequantFuncIQ1_M(const in decodeBufIQ1_M bl, const in uint blockCoords[2], const in uint coordInBlock[2])
{
    decodeBufIQ1_M_packed64 bl64 = decodeBufIQ1_M_packed64(bl);
    const uint idx = coordInBlock[1];

    uvec2 scales = unpack32(bl64.block.scales);
    const float16_t d = uint16BitsToHalf(uint16_t(((scales.x & 0xF000) >> 12) | ((scales.x & 0xF0000000) >> 24) | ((scales.y & 0xF000) >> 4) | ((scales.y & 0xF0000000) >> 16)));

    const uint ib8 = (idx & 0xF8) >> 3;
    const uint ib16 = (idx & 0xF0) >> 4;
    const int i8 = int(idx % 8);
    const uint sc = bl.block.scales[ib8 / 8];
    const uint qs = bl.block.qs[ib8];
    const uint qh = bl.block.qh[ib16] >> (4 * (ib8 & 1));
    const float dl = 2 * bitfieldExtract(sc, 3 * int(ib16 & 3), 3) + 1;
    const float delta = ((qh & 8) != 0) ? -IQ1S_DELTA : IQ1S_DELTA;
    const uint grid = iq1s_grid[qs | ((qh & 7) << 8)];

    float16_t ret = d * float16_t(dl) * (float16_t(bitfieldExtract(int(grid), 2 * i8, 2)) + float16_t(delta));
    return ret;
}
#endif

#if defined(DATA_A_IQ2_XXS)
layout(buffer_reference, std430, buffer_reference_align = 2) buffer decodeBufIQ2_XXS {
   block_iq2_xxs block;
};

layout(buffer_reference, std430, buffer_reference_align = 2) buffer decodeBufIQ2_XXS_packed16 {
   block_iq2_xxs_packed16 block;
};

float16_t dequantFuncIQ2_XXS(const in decodeBufIQ2_XXS bl, const in uint blockCoords[2], const in uint coordInBlock[2])
{
    decodeBufIQ2_XXS_packed16 bl16 = decodeBufIQ2_XXS_packed16(bl);
    const float16_t d = bl.block.d;
    const uint idx = coordInBlock[1];

    const uint ib32 = (idx & 0xE0) >> 5; // 0..7
    const uint ib8 = (idx & 0x18) >> 3;  // 0..3
    const uint iqs = 8 * ib32 + ib8;

    const uint qs = bl.block.qs[iqs];
    const uint signscale = pack32(u16vec2(bl16.block.qs[4*ib32+2], bl16.block.qs[4*ib32+3]));

    const float dscale = float(bl.block.d) * 0.25 * (0.5 + float(signscale >> 28));
    uint sign = bitfieldExtract(signscale, 7 * int(ib8), 7);
    sign |= bitCount(sign) << 7;

    uint g2 = iq2xxs_grid[qs][(idx & 4) >> 2];
    g2 >>= (idx & 2) * 8;
    const vec2 g = vec2(unpack8(g2));

    vec2 ret = dscale * g * ((sign & (1 << (idx & 7))) != 0 ? -1.0hf : 1.0hf);
    return float16_t(ret[idx & 1]);
}
#endif

#if defined(DATA_A_IQ2_XS)
layout(buffer_reference, std430, buffer_reference_align = 2) buffer decodeBufIQ2_XS {
   block_iq2_xs block;
};

float16_t dequantFuncIQ2_XS(const in decodeBufIQ2_XS bl, const in uint blockCoords[2], const in uint coordInBlock[2])
{
    const float16_t d = bl.block.d;
    const uint idx = coordInBlock[1];

    const uint is = (idx & 0xE0) >> 5;     // 0..8
    const uint sshift = (idx & 0x10) >> 2; // 0,4
    const uint iqs = (idx & 0xF8) >> 3;    // 0..63

    const uint16_t qs = bl.block.qs[iqs];
    const float dscale = float(bl.block.d) * 0.25 * (0.5 + float((bl.block.scales[is] >> sshift) & 0xF));

    uint sign = uint(qs >> 9);
    sign |= bitCount(sign) << 7;
    uint g2 = iq2xs_grid[qs & 0x1FF][(idx & 4) >> 2];
    g2 >>= (idx & 2) * 8;
    const vec2 g = vec2(unpack8(g2));

    vec2 ret = dscale * g * ((sign & (1 << (idx & 7))) != 0 ? -1.0hf : 1.0hf);
    return float16_t(ret[idx & 1]);
}
#endif

#if defined(DATA_A_IQ2_S)
layout(buffer_reference, std430, buffer_reference_align = 2) buffer decodeBufIQ2_S {
   block_iq2_s block;
};

float16_t dequantFuncIQ2_S(const in decodeBufIQ2_S bl, const in uint blockCoords[2], const in uint coordInBlock[2])
{
    uint idx = coordInBlock[1];

    const uint ib32 = (idx & 0xE0) >> 5;        // 0..7
    const uint ib8 = (idx & 0xF8) >> 3;         // 0..31
    const uint qhshift = 2 * (ib8 % 4);

    const uint scale = (bl.block.scales[ib32] >> ((idx & 0x10) >> 2)) & 0xf;
    const uint qs = bl.block.qs[ib8];
    const uint qh = bl.block.qh[ib32];
    const uint sign = bl.block.qs[QUANT_K / 8 + ib8] >> (idx & 0x6);

    const float d = float(bl.block.d);
    const float db = d * 0.25 * (0.5 + scale);
    const ivec2 sign01 = 1 - (2 & ivec2(sign << 1, sign));
    uint g2 = iq2s_grid[qs | ((qh << (8 - qhshift)) & 0x300)][(idx & 4) >> 2];
    g2 >>= (idx & 2) * 8;
    const vec2 v = db * vec2(sign01) * vec2(unpack8(g2));
    return float16_t(v[idx & 1]);
}
#endif

#if defined(DATA_A_IQ3_XXS)
layout(buffer_reference, std430, buffer_reference_align = 2) buffer decodeBufIQ3_XXS {
   block_iq3_xxs block;
};

layout(buffer_reference, std430, buffer_reference_align = 2) buffer decodeBufIQ3_XXS_packed16 {
   block_iq3_xxs_packed16 block;
};

float16_t dequantFuncIQ3_XXS(const in decodeBufIQ3_XXS bl, const in uint blockCoords[2], const in uint coordInBlock[2])
{
    decodeBufIQ3_XXS_packed16 bl16 = decodeBufIQ3_XXS_packed16(bl);
    uint idx = coordInBlock[1];

    const uint iqs = (idx & 0xFC) >> 2;             // 0..63
    const uint is = QUANT_K / 4 + ((idx & 0xE0) >> 3);// 8 values

    const float d = float(bl.block.d);
    const uint qs = bl.block.qs[iqs];
    const uint signs = pack32(u16vec2(
        bl16.block.qs[is/2+0],
        bl16.block.qs[is/2+1]
    ));
    const float db = d * 0.5 * (0.5 + (signs >> 28));
    const uint32_t sign7 = bitfieldExtract(signs, 7 * (int(iqs / 2) % 4), 7);
    const uint sign = (sign7 | (bitCount(sign7) << 7)) >> (idx & 0x6);
    const ivec2 sign01 = ivec2(1 - (2 & ivec2(sign << 1, sign)));
    const uint grid = iq3xxs_grid[qs] >> (16 * ((idx & 2) >> 1));
    const vec2 v = db * vec2(sign01) * vec2(unpack8(grid).xy);
    return float16_t(v[idx & 1]);
}
#endif

#if defined(DATA_A_IQ3_S)
layout(buffer_reference, std430, buffer_reference_align = 2) buffer decodeBufIQ3_S {
   block_iq3_s block;
};

float16_t dequantFuncIQ3_S(const in decodeBufIQ3_S bl, const in uint blockCoords[2], const in uint coordInBlock[2])
{
    uint idx = coordInBlock[1];

    const uint iqs = (idx & 0xFC) >> 2;           // 0..63
    const uint iqh = (idx & 0xE0) >> 5;

    const float d = float(bl.block.d);
    const uint qs = bl.block.qs[iqs];
    const uint qh = bl.block.qh[iqh];
    const int8_t sign = int8_t(bl.block.signs[iqs / 2] >> (idx & 0x6));
    const uint scale = bl.block.scales[iqs / 16];
    const ivec2 sign01 = ivec2(1 - (2 & ivec2(sign << 1, sign)));
    const float db = d * (1 + 2 * ((scale >> (4 * (iqh & 1))) & 0xf));
    const uint32_t grid = iq3s_grid[qs | ((qh << (8 - (iqs % 8))) & 256)] >> ((idx & 2) << 3);
    const vec2 v = db * vec2(sign01) * vec2(unpack8(grid).xy);

    return float16_t(v[idx & 1]);
}
#endif

#if defined(DATA_A_IQ4_XS)
layout(buffer_reference, std430, buffer_reference_align = 2) buffer decodeBufIQ4_XS {
   block_iq4_xs block;
};

float16_t dequantFuncIQ4_XS(const in decodeBufIQ4_XS bl, const in uint blockCoords[2], const in uint coordInBlock[2])
{
    const float16_t d = bl.block.d;
    const uint idx = coordInBlock[1];

    const uint ib32 = (idx & 0xE0) >> 5; // 0..7

    const uint sl = (bl.block.scales_l[ib32/2] >> (4 * (ib32 & 1))) & 0xF;
    const uint sh = ((bl.block.scales_h) >> (2 * ib32)) & 3;
    const uint qshift = (idx & 16) >> 2;
    const uint q = (bl.block.qs[16 * ib32 + (idx % 16)] >> qshift) & 0xF;

    float16_t ret = d * float16_t(int(sl | (sh << 4)) - 32) * float16_t(kvalues_iq4nl[q]);
    return ret;
}
#endif

#if defined(DATA_A_IQ4_NL)
layout(buffer_reference, std430, buffer_reference_align = 2) buffer decodeBufIQ4_NL {
   block_iq4_nl block;
};

float16_t dequantFuncIQ4_NL(const in decodeBufIQ4_NL bl, const in uint blockCoords[2], const in uint coordInBlock[2])
{
    const float16_t d = bl.block.d;
    const uint idx = coordInBlock[1];
    const uint iqs = idx & 0xF;
    const uint shift = (idx & 0x10) >> 2;
    uint32_t qs = bl.block.qs[iqs];
    qs >>= shift;
    qs &= 0xF;
    float16_t ret = float16_t(kvalues_iq4nl[qs]) * d;
    return ret;
}
#endif

#if defined(DATA_A_MXFP4)
layout(buffer_reference, std430, buffer_reference_align = 2) buffer decodeBufMXFP4 {
   block_mxfp4 block;
};

float16_t dequantFuncMXFP4(const in decodeBufMXFP4 bl, const in uint blockCoords[2], const in uint coordInBlock[2])
{
    const float d = e8m0_to_fp32(bl.block.e);
    const uint idx = coordInBlock[1];
    const uint iqs = idx & 0xF;
    const uint shift = (idx & 0x10) >> 2;
    uint32_t qs = bl.block.qs[iqs];
    qs >>= shift;
    qs &= 0xF;
    float16_t ret = float16_t(kvalues_mxfp4[qs] * d * 0.5);
    return ret;
}
#endif

#if defined(DATA_A_Q4_0)
#define dequantFuncA dequantFuncQ4_0
#elif defined(DATA_A_Q4_1)
#define dequantFuncA dequantFuncQ4_1
#elif defined(DATA_A_Q5_0)
#define dequantFuncA dequantFuncQ5_0
#elif defined(DATA_A_Q5_1)
#define dequantFuncA dequantFuncQ5_1
#elif defined(DATA_A_Q8_0)
#define dequantFuncA dequantFuncQ8_0
#elif defined(DATA_A_Q2_K)
#define dequantFuncA dequantFuncQ2_K
#elif defined(DATA_A_Q3_K)
#define dequantFuncA dequantFuncQ3_K
#elif defined(DATA_A_Q4_K)
#define dequantFuncA dequantFuncQ4_K
#define fetch_scales fetch_scalesQ4_K
#define store_scales store_scalesQ4_K
#elif defined(DATA_A_Q5_K)
#define dequantFuncA dequantFuncQ5_K
#define fetch_scales fetch_scalesQ5_K
#define store_scales store_scalesQ4_K
#elif defined(DATA_A_Q6_K)
#define dequantFuncA dequantFuncQ6_K
#elif defined(DATA_A_IQ1_S)
#define dequantFuncA dequantFuncIQ1_S
#elif defined(DATA_A_IQ1_M)
#define dequantFuncA dequantFuncIQ1_M
#elif defined(DATA_A_IQ2_XXS)
#define dequantFuncA dequantFuncIQ2_XXS
#elif defined(DATA_A_IQ2_XS)
#define dequantFuncA dequantFuncIQ2_XS
#elif defined(DATA_A_IQ2_S)
#define dequantFuncA dequantFuncIQ2_S
#elif defined(DATA_A_IQ3_XXS)
#define dequantFuncA dequantFuncIQ3_XXS
#elif defined(DATA_A_IQ3_S)
#define dequantFuncA dequantFuncIQ3_S
#elif defined(DATA_A_IQ4_XS)
#define dequantFuncA dequantFuncIQ4_XS
#elif defined(DATA_A_IQ4_NL)
#define dequantFuncA dequantFuncIQ4_NL
#elif defined(DATA_A_MXFP4)
#define dequantFuncA dequantFuncMXFP4
#elif defined(DATA_A_F32)
#define dequantFuncA dequantFuncF32
#endif
