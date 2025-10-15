#extension GL_EXT_shader_explicit_arithmetic_types_int32 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require

#include "types.glsl"

// Each iqs value maps to a 32-bit integer

#if defined(DATA_A_Q4_0)
i32vec2 repack(uint ib, uint iqs) {
    // Use 2-byte loads since a q4_0 block (18 bytes) is not divisible by 4
    const u16vec2 quants = u16vec2(data_a[ib].qs[iqs * 2    ],
                                   data_a[ib].qs[iqs * 2 + 1]);
    const uint32_t vui = pack32(quants);
    return i32vec2( vui       & 0x0F0F0F0F,
                   (vui >> 4) & 0x0F0F0F0F);
}

ACC_TYPE mul_q8_1(const int32_t q_sum, const float da, const vec2 dsb, const int32_t sum_divisor) {
    return ACC_TYPE(da * (float(q_sum) * dsb.x - (8 / sum_divisor) * dsb.y));
}
#endif

#if defined(DATA_A_Q4_1)
i32vec2 repack(uint ib, uint iqs) {
    // Use 4-byte loads since a q4_1 block (20 bytes) is divisible by 4
    const uint32_t vui = data_a_packed32[ib].qs[iqs];
    return i32vec2( vui       & 0x0F0F0F0F,
                   (vui >> 4) & 0x0F0F0F0F);
}

ACC_TYPE mul_q8_1(const int32_t q_sum, const vec2 dma, const vec2 dsb, const int32_t sum_divisor) {
    return ACC_TYPE(float(q_sum) * dma.x * dsb.x + dma.y * dsb.y / sum_divisor);
}
#endif

#if defined(DATA_A_Q5_0)
i32vec2 repack(uint ib, uint iqs) {
    // Use 2-byte loads since a q5_0 block (22 bytes) is not divisible by 4
    const u16vec2 quants = u16vec2(data_a[ib].qs[iqs * 2    ],
                                   data_a[ib].qs[iqs * 2 + 1]);
    const uint32_t vui = pack32(quants);
    const int32_t qh = int32_t((uint32_t(data_a[ib].qh[1]) << 16 | data_a[ib].qh[0]) >> (4 * iqs));
    const int32_t v0 = int32_t(vui & 0x0F0F0F0F)
                     | ((qh & 0xF) * 0x02040810) & 0x10101010; // (0,1,2,3) -> (4,12,20,28)

    const int32_t v1 = int32_t((vui >> 4) & 0x0F0F0F0F)
                     | (((qh >> 16) & 0xF) * 0x02040810) & 0x10101010; // (16,17,18,19) -> (4,12,20,28)

    return i32vec2(v0, v1);
}

ACC_TYPE mul_q8_1(const int32_t q_sum, const float da, const vec2 dsb, const int32_t sum_divisor) {
    return ACC_TYPE(da * (float(q_sum) * dsb.x - (16 / sum_divisor) * dsb.y));
}
#endif

#if defined(DATA_A_Q5_1)
i32vec2 repack(uint ib, uint iqs) {
    // Use 4-byte loads since a q5_1 block (24 bytes) is divisible by 4
    const uint32_t vui = data_a_packed32[ib].qs[iqs];
    const int32_t qh = int32_t(data_a_packed32[ib].qh >> (4 * iqs));
    const int32_t v0 = int32_t(vui & 0x0F0F0F0F)
                     | ((qh & 0xF) * 0x02040810) & 0x10101010; // (0,1,2,3) -> (4,12,20,28)

    const int32_t v1 = int32_t((vui >> 4) & 0x0F0F0F0F)
                     | (((qh >> 16) & 0xF) * 0x02040810) & 0x10101010; // (16,17,18,19) -> (4,12,20,28)

    return i32vec2(v0, v1);
}

ACC_TYPE mul_q8_1(const int32_t q_sum, const vec2 dma, const vec2 dsb, const int32_t sum_divisor) {
    return ACC_TYPE(float(q_sum) * dma.x * dsb.x + dma.y * dsb.y / sum_divisor);
}
#endif

#if defined(DATA_A_Q8_0)
int32_t repack(uint ib, uint iqs) {
    // Use 2-byte loads since a q8_0 block (34 bytes) is not divisible by 4
    return pack32(i16vec2(data_a[ib].qs[iqs * 2    ],
                          data_a[ib].qs[iqs * 2 + 1]));
}

ACC_TYPE mul_q8_1(const int32_t q_sum, const float da, const vec2 dsb, const int32_t sum_divisor) {
    return ACC_TYPE(float(q_sum) * da * dsb.x);
}
#endif

#if defined(DATA_A_Q4_0) || defined(DATA_A_Q5_0) || defined(DATA_A_Q8_0) || defined(DATA_A_IQ1_S) || defined(DATA_A_IQ2_XXS) || defined(DATA_A_IQ2_XS) || defined(DATA_A_IQ2_S) || defined(DATA_A_IQ3_XXS) || defined(DATA_A_IQ3_S) || defined(DATA_A_IQ4_XS) || defined(DATA_A_IQ4_NL)
FLOAT_TYPE get_d(uint ib) {
    return FLOAT_TYPE(data_a[ib].d);
}
#endif

#if defined(DATA_A_MXFP4)
FLOAT_TYPE get_d(uint ib) {
    return FLOAT_TYPE(e8m0_to_fp32(data_a[ib].e));
}
#endif

#if defined(DATA_A_Q4_1) || defined(DATA_A_Q5_1)
FLOAT_TYPE_VEC2 get_dm(uint ib) {
    return FLOAT_TYPE_VEC2(data_a_packed32[ib].dm);
}
#endif
