#if !defined(DATA_A_F32) && !defined(DATA_A_F16)
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
#endif

#include "types.glsl"

#if defined(DATA_A_F32)
vec2 dequantize(uint ib, uint iqs, uint a_offset) {
    return vec2(data_a[a_offset + ib], data_a[a_offset + ib + 1]);
}
#endif

#if defined(DATA_A_F16)
vec2 dequantize(uint ib, uint iqs, uint a_offset) {
    return vec2(data_a[a_offset + ib], data_a[a_offset + ib + 1]);
}
#endif

#if defined(DATA_A_BF16)
vec2 dequantize(uint ib, uint iqs, uint a_offset) {
    return vec2(bf16_to_fp32(data_a[a_offset + ib]), bf16_to_fp32(data_a[a_offset + ib + 1]));
}
#endif

#if defined(DATA_A_Q4_0)
vec2 dequantize(uint ib, uint iqs, uint a_offset) {
    const uint vui = uint(data_a[a_offset + ib].qs[iqs]);
    return (vec2(vui & 0xF, vui >> 4) - 8.0f);
}
vec4 dequantize4(uint ib, uint iqs, uint a_offset) {
    const uint vui = uint(data_a_packed16[a_offset + ib].qs[iqs/2]);
    return (vec4(vui & 0xF, (vui >> 4) & 0xF, (vui >> 8) & 0xF, vui >> 12) - 8.0f);
}
#endif

#if defined(DATA_A_Q4_1)
vec2 dequantize(uint ib, uint iqs, uint a_offset) {
    const uint vui = uint(data_a[a_offset + ib].qs[iqs]);
    return vec2(vui & 0xF, vui >> 4);
}
vec4 dequantize4(uint ib, uint iqs, uint a_offset) {
    const uint vui = uint(data_a_packed16[a_offset + ib].qs[iqs/2]);
    return vec4(vui & 0xF, (vui >> 4) & 0xF, (vui >> 8) & 0xF, vui >> 12);
}
#endif

#if defined(DATA_A_Q5_0)
vec2 dequantize(uint ib, uint iqs, uint a_offset) {
    const uint uint_qh = uint(data_a[a_offset + ib].qh[1]) << 16 | data_a[a_offset + ib].qh[0];
    const ivec2 qh = ivec2(((uint_qh >> iqs) << 4) & 0x10, (uint_qh >> (iqs + 12)) & 0x10);
    const uint vui = uint(data_a[a_offset + ib].qs[iqs]);
    return (vec2((vui & 0xF) | qh.x, (vui >> 4) | qh.y) - 16.0f);
}
vec4 dequantize4(uint ib, uint iqs, uint a_offset) {
    const uint uint_qh = uint(data_a_packed16[a_offset + ib].qh[1]) << 16 | data_a_packed16[a_offset + ib].qh[0];
    const ivec2 qh0 = ivec2(((uint_qh >> iqs) << 4) & 0x10, (uint_qh >> (iqs + 12)) & 0x10);
    const ivec2 qh1 = ivec2(((uint_qh >> (iqs + 1)) << 4) & 0x10, (uint_qh >> (iqs + 13)) & 0x10);
    const uint vui = uint(data_a_packed16[a_offset + ib].qs[iqs/2]);
    return (vec4((vui & 0xF) | qh0.x, ((vui >> 4) & 0xF) | qh0.y, ((vui >> 8) & 0xF) | qh1.x, (vui >> 12) | qh1.y) - 16.0f);
}
#endif

#if defined(DATA_A_Q5_1)
vec2 dequantize(uint ib, uint iqs, uint a_offset) {
    const uint uint_qh = data_a[a_offset + ib].qh;
    const ivec2 qh = ivec2(((uint_qh >> iqs) << 4) & 0x10, (uint_qh >> (iqs + 12)) & 0x10);
    const uint vui = uint(data_a[a_offset + ib].qs[iqs]);
    return vec2((vui & 0xF) | qh.x, (vui >> 4) | qh.y);
}
vec4 dequantize4(uint ib, uint iqs, uint a_offset) {
    const uint uint_qh = data_a_packed16[a_offset + ib].qh;
    const ivec2 qh0 = ivec2(((uint_qh >> iqs) << 4) & 0x10, (uint_qh >> (iqs + 12)) & 0x10);
    const ivec2 qh1 = ivec2(((uint_qh >> (iqs + 1)) << 4) & 0x10, (uint_qh >> (iqs + 13)) & 0x10);
    const uint vui = uint(data_a_packed16[a_offset + ib].qs[iqs/2]);
    return vec4((vui & 0xF) | qh0.x, ((vui >> 4) & 0xF) | qh0.y, ((vui >> 8) & 0xF) | qh1.x, (vui >> 12) | qh1.y);
}
#endif

#if defined(DATA_A_Q8_0)
vec2 dequantize(uint ib, uint iqs, uint a_offset) {
    return vec2(int(data_a[a_offset + ib].qs[iqs]), int(data_a[a_offset + ib].qs[iqs + 1]));
}
vec4 dequantize4(uint ib, uint iqs, uint a_offset) {
    const i8vec2 v0 = unpack8(int32_t(data_a_packed16[a_offset + ib].qs[iqs/2])).xy; // vec4 used due to #12147
    const i8vec2 v1 = unpack8(int32_t(data_a_packed16[a_offset + ib].qs[iqs/2 + 1])).xy;
    return vec4(v0.x, v0.y, v1.x, v1.y);
}
#endif

#if defined(DATA_A_IQ1_S)
vec2 dequantize(uint ib, uint iqs, uint a_offset) {
    const uint ib32 = iqs / 32;
    const uint ib8 = iqs / 8;
    const int i8 = int(iqs % 8);
    const uint qh = data_a[a_offset + ib].qh[ib32];
    const uint qs = data_a[a_offset + ib].qs[ib8];
    const float dl = float(2 * bitfieldExtract(qh, 12, 3) + 1);
    const float delta = ((qh & 0x8000) != 0) ? -IQ1S_DELTA : IQ1S_DELTA;
    const uint idxhi = bitfieldExtract(qh, 3 * int(ib8 & 3), 3);
    const int16_t grid = int16_t(iq1s_grid[qs | (idxhi << 8)]);
    // Signed bitfield extract.
    const ivec2 gvec = ivec2(
      bitfieldExtract(grid, 2 * (i8), 2),
      bitfieldExtract(grid, 2 * (i8 + 1), 2)
    );
    return dl * (vec2(gvec) + delta);
}
vec4 dequantize4(uint ib, uint iqs, uint a_offset) {
    const uint ib32 = iqs / 32;
    const uint ib8 = iqs / 8;
    const int i8 = int(iqs % 8);
    const uint qh = data_a[a_offset + ib].qh[ib32];
    const uint qs = data_a[a_offset + ib].qs[ib8];
    const float dl = 2 * bitfieldExtract(qh, 12, 3) + 1;
    const float delta = ((qh & 0x8000) != 0) ? -IQ1S_DELTA : IQ1S_DELTA;
    const int16_t grid = int16_t(iq1s_grid[qs | (bitfieldExtract(qh, 3 * int(ib8 & 3), 3) << 8)]);
    // Signed bitfield extract.
    const ivec4 gvec = ivec4(
      bitfieldExtract(grid, 2 * (i8), 2),
      bitfieldExtract(grid, 2 * (i8 + 1), 2),
      bitfieldExtract(grid, 2 * (i8 + 2), 2),
      bitfieldExtract(grid, 2 * (i8 + 3), 2)
    );
    return dl * (vec4(gvec) + delta);
}
#endif

#if defined(DATA_A_IQ1_M)
vec2 dequantize(uint ib, uint iqs, uint a_offset) {
    const uint ib8 = iqs / 8;
    const uint ib16 = iqs / 16;
    const int i8 = int(iqs % 8);
    const uint sc = data_a[a_offset + ib].scales[iqs / 64];
    const uint qs = data_a[a_offset + ib].qs[ib8];
    const uint qh = data_a[a_offset + ib].qh[ib16] >> (4 * (ib8 & 1));
    const float dl = 2 * bitfieldExtract(sc, 3 * int(ib16 & 3), 3) + 1;
    const float delta = ((qh & 8) != 0) ? -IQ1M_DELTA : IQ1M_DELTA;
    const int16_t grid = int16_t(iq1s_grid[qs | ((qh & 7) << 8)]);
    // Signed bitfield extract.
    const ivec2 gvec = ivec2(
      bitfieldExtract(grid, 2 * (i8), 2),
      bitfieldExtract(grid, 2 * (i8 + 1), 2)
    );
    return dl * (vec2(gvec) + delta);
}
vec4 dequantize4(uint ib, uint iqs, uint a_offset) {
    const uint ib8 = iqs / 8;
    const uint ib16 = iqs / 16;
    const int i8 = int(iqs % 8);
    const uint sc = data_a[a_offset + ib].scales[iqs / 64];
    const uint qs = data_a[a_offset + ib].qs[ib8];
    const uint qh = data_a[a_offset + ib].qh[ib16] >> (4 * (ib8 & 1));
    const float dl = 2 * bitfieldExtract(sc, 3 * int(ib16 & 3), 3) + 1;
    const float delta = ((qh & 8) != 0) ? -IQ1M_DELTA : IQ1M_DELTA;
    const int16_t grid = int16_t(iq1s_grid[qs | ((qh & 7) << 8)]);
    // Signed bitfield extract.
    const ivec4 gvec = ivec4(
      bitfieldExtract(grid, 2 * (i8), 2),
      bitfieldExtract(grid, 2 * (i8 + 1), 2),
      bitfieldExtract(grid, 2 * (i8 + 2), 2),
      bitfieldExtract(grid, 2 * (i8 + 3), 2)
    );
    return dl * (vec4(gvec) + delta);
}
#endif

#if defined(DATA_A_IQ2_XXS)
vec2 dequantize(uint ib, uint iqs, uint a_offset) {
    const uint ib32 = iqs / 32;
    const uint ib8 = (iqs / 8) % 4;
    const uint qs = data_a[a_offset + ib].qs[8 * ib32 + ib8];
    // Scales are stored as packed 7+7+7+7+4 bits (4 sign tuples and 1 int4 scale)
    const uint signs = pack32(u16vec2(data_a_packed16[a_offset + ib].qs[4 * ib32 + 2],
        data_a_packed16[a_offset + ib].qs[4 * ib32 + 3]));
    const float db = 0.25 * (0.5 + (signs >> 28));
    const uint sign7 = bitfieldExtract(signs, 7 * int(ib8), 7);
    // Add parity bit
    const uint sign8 = sign7 | (bitCount(sign7) << 7);
    const uint sign = sign8 >> (iqs % 8);
    const u8vec4 grid = unpack8(iq2xxs_grid[qs][(iqs % 8) / 4] >> (8 * (iqs % 4)));
    bool sign0 = (sign & 1) != 0;
    bool sign1 = (sign & 2) != 0;
    return db * vec2(
        grid.x * (sign0 ? -1.0 : 1.0),
        grid.y * (sign1 ? -1.0 : 1.0)
    );
}
vec4 dequantize4(uint ib, uint iqs, uint a_offset) {
    const uint ib32 = iqs / 32;
    const uint ib8 = (iqs / 8) % 4;
    const uint qs = data_a[a_offset + ib].qs[8 * ib32 + ib8];
    // Scales are stored as packed 7+7+7+7+4 bits (4 sign tuples and 1 int4 scale)
    const uint signs = pack32(u16vec2(data_a_packed16[a_offset + ib].qs[4 * ib32 + 2],
        data_a_packed16[a_offset + ib].qs[4 * ib32 + 3]));
    const float db = 0.25 * (0.5 + (signs >> 28));
    const uint sign7 = bitfieldExtract(signs, 7 * int(ib8), 7);
    // Add parity bit
    const uint sign8 = sign7 | (bitCount(sign7) << 7);
    const uint sign = sign8 >> (iqs % 8);
    const u8vec4 grid = unpack8(iq2xxs_grid[qs][(iqs % 8) / 4] >> (8 * (iqs % 4)));
    bool sign0 = (sign & 1) != 0;
    bool sign1 = (sign & 2) != 0;
    bool sign2 = (sign & 4) != 0;
    bool sign3 = (sign & 8) != 0;
    return db * vec4(
        grid.x * (sign0 ? -1.0 : 1.0),
        grid.y * (sign1 ? -1.0 : 1.0),
        grid.z * (sign2 ? -1.0 : 1.0),
        grid.w * (sign3 ? -1.0 : 1.0)
    );
}
#endif

#if defined(DATA_A_IQ2_XS)
vec2 dequantize(uint ib, uint iqs, uint a_offset) {
    const uint scale = (data_a[a_offset + ib].scales[iqs / 32] >> (4 * ((iqs / 16) & 1))) & 0xf;
    const uint qs = data_a[a_offset + ib].qs[iqs / 8];
    const float db = 0.25 * (0.5 + scale);
    const uint sign7 = qs >> 9;
    // Add parity bit
    const uint sign8 = sign7 | (bitCount(sign7) << 7);
    const uint sign = sign8 >> (iqs % 8);
    const u8vec4 grid = unpack8(iq2xs_grid[qs & 511][(iqs % 8) / 4] >> (8 * (iqs % 4)));
    bool sign0 = (sign & 1) != 0;
    bool sign1 = (sign & 2) != 0;
    return db * vec2(
        grid.x * (sign0 ? -1.0 : 1.0),
        grid.y * (sign1 ? -1.0 : 1.0)
    );
}
vec4 dequantize4(uint ib, uint iqs, uint a_offset) {
    const uint scale = (data_a[a_offset + ib].scales[iqs / 32] >> (4 * ((iqs / 16) & 1))) & 0xf;
    const uint qs = data_a[a_offset + ib].qs[iqs / 8];
    const float db = 0.25 * (0.5 + scale);
    const uint sign7 = qs >> 9;
    // Add parity bit
    const uint sign8 = sign7 | (bitCount(sign7) << 7);
    const uint sign = sign8 >> (iqs % 8);
    const u8vec4 grid = unpack8(iq2xs_grid[qs & 511][(iqs % 8) / 4] >> (8 * (iqs % 4)));
    bool sign0 = (sign & 1) != 0;
    bool sign1 = (sign & 2) != 0;
    bool sign2 = (sign & 4) != 0;
    bool sign3 = (sign & 8) != 0;
    return db * vec4(
        grid.x * (sign0 ? -1.0 : 1.0),
        grid.y * (sign1 ? -1.0 : 1.0),
        grid.z * (sign2 ? -1.0 : 1.0),
        grid.w * (sign3 ? -1.0 : 1.0)
    );
}
#endif

#if defined(DATA_A_IQ2_S)
vec2 dequantize(uint ib, uint iqs, uint a_offset) {
    const uint ib32 = iqs / 32;
    const uint ib8 = iqs / 8;

    const uint scale = (data_a[a_offset + ib].scales[ib32] >> (4 * ((iqs / 16) & 1))) & 0xf;
    const uint qs = data_a[a_offset + ib].qs[ib8];
    const uint qh = data_a[a_offset + ib].qh[ib32];
    const uint qhshift = 2 * (ib8 % 4);
    const uint sign = data_a[a_offset + ib].qs[QUANT_K / 8 + ib8] >> (iqs % 8);

    const float db = 0.25 * (0.5 + scale);
    const u8vec4 grid = unpack8(iq2s_grid[qs | ((qh << (8 - qhshift)) & 0x300)][(iqs % 8) / 4]);
    bool sign0 = (sign & 1) != 0;
    bool sign1 = (sign & 2) != 0;
    return db * vec2(
        grid[iqs % 4] * (sign0 ? -1.0 : 1.0),
        grid[(iqs % 4) + 1] * (sign1 ? -1.0 : 1.0)
    );
}
vec4 dequantize4(uint ib, uint iqs, uint a_offset) {
    const uint ib32 = iqs / 32;
    const uint ib8 = iqs / 8;

    const uint scale = (data_a[a_offset + ib].scales[ib32] >> (4 * ((iqs / 16) & 1))) & 0xf;
    const uint qs = data_a[a_offset + ib].qs[ib8];
    const uint qh = data_a[a_offset + ib].qh[ib32];
    const uint qhshift = 2 * (ib8 % 4);
    const uint sign = data_a[a_offset + ib].qs[QUANT_K / 8 + ib8] >> (iqs % 8);

    const float db = 0.25 * (0.5 + scale);
    const u8vec4 grid = unpack8(iq2s_grid[qs | ((qh << (8 - qhshift)) & 0x300)][(iqs % 8) / 4]);
    bool sign0 = (sign & 1) != 0;
    bool sign1 = (sign & 2) != 0;
    bool sign2 = (sign & 4) != 0;
    bool sign3 = (sign & 8) != 0;
    return db * vec4(
        grid.x * (sign0 ? -1.0 : 1.0),
        grid.y * (sign1 ? -1.0 : 1.0),
        grid.z * (sign2 ? -1.0 : 1.0),
        grid.w * (sign3 ? -1.0 : 1.0)
    );
}
#endif

#if defined(DATA_A_IQ3_XXS)
vec2 dequantize(uint ib, uint iqs, uint a_offset) {
    const uint ib4 = iqs / 4;
    const uint ib32 = iqs / 32;
    const uint is = QUANT_K / 4 + 4 * ib32;
    const uint qs = data_a[a_offset + ib].qs[ib4];
    // Scales are stored as packed 7+7+7+7+4 bits (4 sign tuples and 1 int4 scale)
    const uint signs = pack32(u16vec2(data_a_packed16[a_offset + ib].qs[is / 2],
        data_a_packed16[a_offset + ib].qs[is / 2 + 1]));
    const float db = 0.5 * (0.5 + (signs >> 28));
    const uint sign7 = bitfieldExtract(signs, 7 * (int(ib4 / 2) % 4), 7);
    // Add parity bit
    const uint sign8 = sign7 | (bitCount(sign7) << 7);
    const uint sign = sign8 >> (iqs % 8);
    const u8vec4 grid = unpack8(iq3xxs_grid[qs] >> (8 * (iqs % 4)));
    bool sign0 = (sign & 1) != 0;
    bool sign1 = (sign & 2) != 0;
    return db * vec2(
        grid.x * (sign0 ? -1.0 : 1.0),
        grid.y * (sign1 ? -1.0 : 1.0)
    );
}
vec4 dequantize4(uint ib, uint iqs, uint a_offset) {
    const uint ib4 = iqs / 4;
    const uint ib32 = iqs / 32;
    const uint is = QUANT_K / 4 + 4 * ib32;
    const uint qs = data_a[a_offset + ib].qs[ib4];
    const uint signs = pack32(u16vec2(data_a_packed16[a_offset + ib].qs[is / 2],
        data_a_packed16[a_offset + ib].qs[is / 2 + 1]));
    const float db = 0.5 * (0.5 + (signs >> 28));
    const uint sign7 = bitfieldExtract(signs, 7 * (int(ib4 / 2) % 4), 7);
    // Add parity bit
    const uint sign8 = sign7 | (bitCount(sign7) << 7);
    const uint sign = sign8 >> (iqs % 8);
    const u8vec4 grid = unpack8(iq3xxs_grid[qs]);
    bool sign0 = (sign & 1) != 0;
    bool sign1 = (sign & 2) != 0;
    bool sign2 = (sign & 4) != 0;
    bool sign3 = (sign & 8) != 0;
    return db * vec4(
        grid.x * (sign0 ? -1.0 : 1.0),
        grid.y * (sign1 ? -1.0 : 1.0),
        grid.z * (sign2 ? -1.0 : 1.0),
        grid.w * (sign3 ? -1.0 : 1.0)
    );
}
#endif

#if defined(DATA_A_IQ3_S)
vec2 dequantize(uint ib, uint iqs, uint a_offset) {
    const uint qs = data_a[a_offset + ib].qs[iqs / 4];
    const uint qh = data_a[a_offset + ib].qh[iqs / 32];
    const uint sign = data_a[a_offset + ib].signs[iqs / 8] >> (iqs % 8);
    const uint scale = data_a[a_offset + ib].scales[iqs / 64];
    bool sign0 = (sign & 1) != 0;
    bool sign1 = (sign & 2) != 0;
    const float db = 1 + 2 * ((scale >> (4 * ((iqs / 32) & 1))) & 0xf);
    const uint32_t grid = iq3s_grid[qs | ((qh << (8 - ((iqs / 4) % 8))) & 256)] >> (8 * (iqs % 4));
    return db * vec2(
        int(grid & 0xFF) * (sign0 ? -1.0 : 1.0),
        int((grid >> 8) & 0xFF) * (sign1 ? -1.0 : 1.0)
    );
}
vec4 dequantize4(uint ib, uint iqs, uint a_offset) {
    const uint ib4 = iqs / 4;
    const uint ib32 = iqs / 32;
    const uint qs = data_a[a_offset + ib].qs[ib4];
    const uint qh = data_a[a_offset + ib].qh[ib32];
    const uint sign = data_a[a_offset + ib].signs[iqs / 8] >> (iqs % 8);
    const uint scale = data_a[a_offset + ib].scales[ib32 / 2];
    bool sign0 = (sign & 1) != 0;
    bool sign1 = (sign & 2) != 0;
    bool sign2 = (sign & 4) != 0;
    bool sign3 = (sign & 8) != 0;
    const float db = 1 + 2 * ((scale >> (4 * (ib32 & 1))) & 0xf);
    const uint32_t grid = iq3s_grid[qs | ((qh << (8 - ib4 % 8)) & 256)] >> (8 * (iqs % 4));
    return db * vec4(
        int(grid & 0xFF) * (sign0 ? -1.0 : 1.0),
        int((grid >> 8) & 0xFF) * (sign1 ? -1.0 : 1.0),
        int((grid >> 16) & 0xFF) * (sign2 ? -1.0 : 1.0),
        int((grid >> 24) & 0xFF) * (sign3 ? -1.0 : 1.0)
    );
}
#endif

#if defined(DATA_A_IQ4_XS)
vec2 dequantize(uint ib, uint iqs, uint a_offset) {
    const uint ib32 = iqs / 32;
    const uint iq = 16 * ib32 + (iqs % 16);

    const uint sl = (data_a[a_offset + ib].scales_l[ib32/2] >> (4 * (ib32 & 1))) & 0xF;
    const uint sh = (data_a[a_offset + ib].scales_h >> (2 * ib32)) & 3;
    const uint qshift = (iqs & 16) >> 2;
    u8vec2 qs = u8vec2(data_a[a_offset + ib].qs[iq], data_a[a_offset + ib].qs[iq + 1]);
    qs = (qs >> qshift) & uint8_t(0xF);

    const float dl = float(int(sl | (sh << 4)) - 32);
    return dl * vec2(kvalues_iq4nl[qs.x], kvalues_iq4nl[qs.y]);
}
vec4 dequantize4(uint ib, uint iqs, uint a_offset) {
    const uint ib32 = iqs / 32;
    const uint iq = 16 * ib32 + (iqs % 16);

    const uint sl = (data_a[a_offset + ib].scales_l[ib32/2] >> (4 * (ib32 & 1))) & 0xF;
    const uint sh = (data_a[a_offset + ib].scales_h >> (2 * ib32)) & 3;
    const uint qshift = (iqs & 16) >> 2;
    u8vec4 qs = u8vec4(
        data_a[a_offset + ib].qs[iq + 0],
        data_a[a_offset + ib].qs[iq + 1],
        data_a[a_offset + ib].qs[iq + 2],
        data_a[a_offset + ib].qs[iq + 3]
    );
    qs = (qs >> qshift) & uint8_t(0xF);

    const float dl = float(int(sl | (sh << 4)) - 32);
    return dl * vec4(
        kvalues_iq4nl[qs.x], kvalues_iq4nl[qs.y],
        kvalues_iq4nl[qs.z], kvalues_iq4nl[qs.w]);
}
#endif

#if defined(DATA_A_IQ4_NL)
vec2 dequantize(uint ib, uint iqs, uint a_offset) {
    const uint vui = uint(data_a[a_offset + ib].qs[iqs]);
    return vec2(kvalues_iq4nl[vui & 0xF], kvalues_iq4nl[vui >> 4]);
}
vec4 dequantize4(uint ib, uint iqs, uint a_offset) {
    const uint vui = uint(data_a_packed16[a_offset + ib].qs[iqs/2]);
    return vec4(kvalues_iq4nl[vui & 0xF], kvalues_iq4nl[(vui >> 4) & 0xF], kvalues_iq4nl[(vui >> 8) & 0xF], kvalues_iq4nl[vui >> 12]);
}
#endif

#if defined(DATA_A_MXFP4)
vec2 dequantize(uint ib, uint iqs, uint a_offset) {
    const uint vui = uint(data_a[a_offset + ib].qs[iqs]);
    return vec2(kvalues_mxfp4[vui & 0xF], kvalues_mxfp4[vui >> 4]) * 0.5;
}
vec4 dequantize4(uint ib, uint iqs, uint a_offset) {
    vec2 v0 = dequantize(ib, iqs, a_offset);
    vec2 v1 = dequantize(ib, iqs + 1, a_offset);
    return vec4(v0.x, v0.y, v1.x, v1.y);
}
#endif

#if defined(DATA_A_F32) || defined(DATA_A_F16) || defined(DATA_A_BF16)
vec2 get_dm(uint ib, uint a_offset) {
    return vec2(0, 0);
}
#endif

#if defined(DATA_A_IQ1_M)
vec2 get_dm(uint ib, uint a_offset) {
    const uint16_t[4] scales = data_a[a_offset + ib].scales;
    const u16vec4 s = u16vec4(scales[0], scales[1], scales[2], scales[3]) >> 12;
    const float d = float(unpackHalf2x16(s.x | (s.y << 4) | (s.z << 8) | (s.w << 12)).x);
    return vec2(d, 0);
}
#endif

#if defined(DATA_A_Q4_0) || defined(DATA_A_Q5_0) || defined(DATA_A_Q8_0) || defined(DATA_A_IQ1_S) || defined(DATA_A_IQ2_XXS) || defined(DATA_A_IQ2_XS) || defined(DATA_A_IQ2_S) || defined(DATA_A_IQ3_XXS) || defined(DATA_A_IQ3_S) || defined(DATA_A_IQ4_XS) || defined(DATA_A_IQ4_NL)
vec2 get_dm(uint ib, uint a_offset) {
    return vec2(float(data_a[a_offset + ib].d), 0);
}
#endif

#if defined(DATA_A_MXFP4)
vec2 get_dm(uint ib, uint a_offset) {
    return vec2(e8m0_to_fp32(data_a[a_offset + ib].e), 0);
}
#endif

#if defined(DATA_A_Q4_1) || defined(DATA_A_Q5_1)
vec2 get_dm(uint ib, uint a_offset) {
    return vec2(float(data_a[a_offset + ib].d), float(data_a[a_offset + ib].m));
}
#endif

#if defined(DATA_A_Q2_K)
vec2 dequantize(uint ib, uint iqs, uint a_offset) {
    iqs /= 2;
    const uint qsi = (iqs / 64) * 32 + (iqs % 16) * 2; // 0,2,4..30
    const uint scalesi = iqs / 8;                      // 0..15
    const uint qsshift = ((iqs % 64) / 16) * 2;        // 0,2,4,6

    const uvec2 qs = uvec2(data_a[a_offset + ib].qs[qsi], data_a[a_offset + ib].qs[qsi + 1]);
    const uint scales = data_a[a_offset + ib].scales[scalesi];
    const vec2 dm = vec2(data_a[a_offset + ib].dm);

    return dm.x * float(scales & 0xF) * vec2((qs >> qsshift) & 3) - dm.y * float(scales >> 4);
}
vec2 get_dm(uint ib, uint a_offset) {
    return vec2(1, 0);
}
#endif

#if defined(DATA_A_Q3_K)
vec2 dequantize(uint ib, uint iqs, uint a_offset) {
    iqs /= 2;
    const uint n = iqs / 64;                     // 0,1
    const uint qsi = n * 32 + (iqs % 16) * 2;    // 0,2,4..62
    const uint hmi =          (iqs % 16) * 2;    // 0,2,4..30
    const uint j = (iqs % 64) / 4;               // 0..3
    const uint is = iqs / 8;                     // 0..15
    const uint halfsplit = ((iqs % 64) / 16);    // 0,1,2,3
    const uint qsshift = halfsplit * 2;          // 0,2,4,6
    const uint m = 1 << (4 * n + halfsplit);     // 1,2,4,8,16,32,64,128

    const int8_t us = int8_t(((data_a[a_offset + ib].scales[is % 8] >> (4 * int(is / 8))) & 0xF)
                          | (((data_a[a_offset + ib].scales[8 + (is % 4)] >> (2 * int(is / 4))) & 3) << 4));
    const float dl = float(data_a[a_offset + ib].d) * float(us - 32);

    return vec2(dl * float(int8_t((data_a[a_offset + ib].qs[qsi    ] >> qsshift) & 3) - (((data_a[a_offset + ib].hmask[hmi    ] & m) != 0) ? 0 : 4)),
                dl * float(int8_t((data_a[a_offset + ib].qs[qsi + 1] >> qsshift) & 3) - (((data_a[a_offset + ib].hmask[hmi + 1] & m) != 0) ? 0 : 4)));
}
vec2 get_dm(uint ib, uint a_offset) {
    return vec2(1, 0);
}
#endif

#if defined(DATA_A_Q4_K)
vec2 dequantize(uint ib, uint iqs, uint a_offset) {
    iqs /= 2;
    const uint n = iqs / 32;                   // 0,1,2,3
    const uint b = (iqs % 32) / 16;            // 0,1
    const uint is = 2 * n + b;                 // 0..7
    const uint qsi = n * 32 + (iqs % 16) * 2;  // 0,2,4..126

    const vec2 loadd = vec2(data_a[a_offset + ib].dm);

    const uint scidx0 = (is < 4) ? is : (is + 4);
    const uint scidx1 = (is < 4) ? is : (is - 4);
    const uint scidxmask1 = (is < 4) ? 0x30 : 0xC0;
    const uint scidxshift1 = (is < 4) ? 0 : 2;
    const uint mbidx0 = is + 4;
    const uint mbidx1 = (is < 4) ? is + 4 : is;
    const uint mbidxmask0 = (is < 4) ? 0xF : 0xF0;
    const uint mbidxshift0 = (is < 4) ? 0 : 4;
    const uint mbidxmask1 = (is < 4) ? 0x30 : 0xC0;
    const uint mbidxshift1 = (is < 4) ? 0 : 2;

    const uint8_t sc = uint8_t((data_a[a_offset + ib].scales[scidx0] & 0xF) | ((data_a[a_offset + ib].scales[scidx1] & scidxmask1) >> scidxshift1));
    const uint8_t mbyte = uint8_t((data_a[a_offset + ib].scales[mbidx0] & mbidxmask0) >> mbidxshift0 | ((data_a[a_offset + ib].scales[mbidx1] & mbidxmask1) >> mbidxshift1));

    const float d = loadd.x * sc;
    const float m = -loadd.y * mbyte;

    return vec2(fma(d, float((data_a[a_offset + ib].qs[qsi    ] >> (b * 4)) & 0xF), m),
                fma(d, float((data_a[a_offset + ib].qs[qsi + 1] >> (b * 4)) & 0xF), m));
}
vec2 get_dm(uint ib, uint a_offset) {
    return vec2(1, 0);
}
#endif

#if defined(DATA_A_Q5_K)
vec2 dequantize(uint ib, uint iqs, uint a_offset) {
    iqs /= 2;
    const uint n = iqs / 32;                   // 0,1,2,3
    const uint b = (iqs % 32) / 16;            // 0,1
    const uint is = 2 * n + b;                 // 0..7
    const uint qsi = n * 32 + (iqs % 16) * 2;  // 0,2,4..126
    const uint qhi = (iqs % 16) * 2;           // 0,2,4..30

    const uint8_t hm = uint8_t(1 << (iqs / 16));

    const vec2 loadd = vec2(data_a[a_offset + ib].dm);

    const uint scidx0 = (is < 4) ? is : (is + 4);
    const uint scidx1 = (is < 4) ? is : (is - 4);
    const uint scidxmask1 = (is < 4) ? 0x30 : 0xC0;
    const uint scidxshift1 = (is < 4) ? 0 : 2;
    const uint mbidx0 = is + 4;
    const uint mbidx1 = (is < 4) ? is + 4 : is;
    const uint mbidxmask0 = (is < 4) ? 0xF : 0xF0;
    const uint mbidxshift0 = (is < 4) ? 0 : 4;
    const uint mbidxmask1 = (is < 4) ? 0x30 : 0xC0;
    const uint mbidxshift1 = (is < 4) ? 0 : 2;

    const uint8_t sc    = uint8_t((data_a[a_offset + ib].scales[scidx0] & 0xF)                         | ((data_a[a_offset + ib].scales[scidx1] & scidxmask1) >> scidxshift1));
    const uint8_t mbyte = uint8_t(((data_a[a_offset + ib].scales[mbidx0] & mbidxmask0) >> mbidxshift0) | ((data_a[a_offset + ib].scales[mbidx1] & mbidxmask1) >> mbidxshift1));

    const float d = loadd.x * sc;
    const float m = -loadd.y * mbyte;

    return vec2(fma(d, float((data_a[a_offset + ib].qs[qsi    ] >> (b * 4)) & 0xF) + float((data_a[a_offset + ib].qh[qhi    ] & hm) != 0 ? 16 : 0), m),
                fma(d, float((data_a[a_offset + ib].qs[qsi + 1] >> (b * 4)) & 0xF) + float((data_a[a_offset + ib].qh[qhi + 1] & hm) != 0 ? 16 : 0), m));
}
vec2 get_dm(uint ib, uint a_offset) {
    return vec2(1, 0);
}
#endif

#if defined(DATA_A_Q6_K)
vec2 dequantize(uint ib, uint iqs, uint a_offset) {
    iqs /= 2;
    const uint n = iqs / 64;                    // 0,1
    const uint b = (iqs % 64) / 32;             // 0,1
    const uint is_b = (iqs % 16) / 8;           // 0,1
    const uint qhshift = ((iqs % 64) / 16) * 2; // 0,2,4,6
    const uint is = 8 * n + qhshift + is_b;     // 0..15
    const uint qsi = n * 64 + (iqs % 32) * 2;   // 0,2,4..126
    const uint qhi = n * 32 + (iqs % 16) * 2;   // 0,2,4..62

    const float dscale = float(data_a[a_offset + ib].d) * float(data_a[a_offset + ib].scales[is]);

    return vec2(dscale * float(int8_t(((data_a[a_offset + ib].ql[qsi    ] >> (b * 4)) & 0xF) | (((data_a[a_offset + ib].qh[qhi    ] >> qhshift) & 3) << 4)) - 32),
                dscale * float(int8_t(((data_a[a_offset + ib].ql[qsi + 1] >> (b * 4)) & 0xF) | (((data_a[a_offset + ib].qh[qhi + 1] >> qhshift) & 3) << 4)) - 32));
}
vec2 get_dm(uint ib, uint a_offset) {
    return vec2(1, 0);
}
#endif
