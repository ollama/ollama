#extension GL_EXT_shader_explicit_arithmetic_types_int32 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require

#include "types.glsl"

// Each iqs value maps to a 32-bit integer

#if defined(DATA_A_Q4_0) || defined(DATA_A_Q4_1)
// 2-byte loads for Q4_0 blocks (18 bytes)
// 4-byte loads for Q4_1 blocks (20 bytes)
i32vec2 repack(uint ib, uint iqs) {
#ifdef DATA_A_Q4_0
    const u16vec2 quants = u16vec2(data_a_packed16[ib].qs[iqs * 2    ],
                                   data_a_packed16[ib].qs[iqs * 2 + 1]);
    const uint32_t vui = pack32(quants);
    return i32vec2( vui       & 0x0F0F0F0F,
                   (vui >> 4) & 0x0F0F0F0F);
#else // DATA_A_Q4_1
    const uint32_t vui = data_a_packed32[ib].qs[iqs];
    return i32vec2( vui       & 0x0F0F0F0F,
                   (vui >> 4) & 0x0F0F0F0F);
#endif
}

#ifdef DATA_A_Q4_0
ACC_TYPE mul_q8_1(const int32_t q_sum, const float da, const vec2 dsb, const int32_t sum_divisor) {
    return ACC_TYPE(da * (float(q_sum) * dsb.x - (8 / sum_divisor) * dsb.y));
}
#else // DATA_A_Q4_1
ACC_TYPE mul_q8_1(const int32_t q_sum, const vec2 dma, const vec2 dsb, const int32_t sum_divisor) {
    return ACC_TYPE(float(q_sum) * dma.x * dsb.x + dma.y * dsb.y / sum_divisor);
}
#endif

#ifdef MMQ_SHMEM
void block_a_to_shmem(const uint buf_ib, const uint ib, const uint iqs) {
#ifdef DATA_A_Q4_0
    buf_a[buf_ib].qs[iqs] = pack32(u16vec2(data_a_packed16[ib].qs[iqs * 2],
                                           data_a_packed16[ib].qs[iqs * 2 + 1]));

    if (iqs == 0) {
        buf_a[buf_ib].dm = FLOAT_TYPE(data_a_packed16[ib].d);
    }
#else // DATA_A_Q4_1
    buf_a[buf_ib].qs[iqs] = data_a_packed32[ib].qs[iqs];

    if (iqs == 0) {
        buf_a[buf_ib].dm = FLOAT_TYPE_VEC2(data_a_packed32[ib].dm);
    }
#endif
}

void block_a_to_registers(const uint reg_ib, const uint buf_ib) {
    cache_a[reg_ib].dm = buf_a[buf_ib].dm;

    [[unroll]] for (uint iqs = 0; iqs < 4; iqs++) {
        cache_a[reg_ib].qs[iqs] = buf_a[buf_ib].qs[iqs];
    }
}

ACC_TYPE mmq_dot_product(const uint ib_a) {
    int32_t q_sum = 0;
    [[unroll]] for (uint iqs = 0; iqs < 4; iqs++) {
        const uint32_t vui = cache_a[ib_a].qs[iqs];
        const i32vec2 qs_a = i32vec2( vui       & 0x0F0F0F0F,
                                     (vui >> 4) & 0x0F0F0F0F);

        const int32_t qs_b0 = cache_b.qs[iqs];
        const int32_t qs_b1 = cache_b.qs[iqs + 4];

        q_sum += dotPacked4x8EXT(qs_a.x, qs_b0);
        q_sum += dotPacked4x8EXT(qs_a.y, qs_b1);
    }

    return mul_q8_1(q_sum, cache_a[ib_a].dm, cache_b.ds, 1);
}
#endif // MMQ_SHMEM

#elif defined(DATA_A_Q5_0) || defined(DATA_A_Q5_1)
// 2-byte loads for Q5_0 blocks (22 bytes)
// 4-byte loads for Q5_1 blocks (24 bytes)
i32vec2 repack(uint ib, uint iqs) {
    const u16vec2 quants = u16vec2(data_a_packed16[ib].qs[iqs * 2    ],
                                   data_a_packed16[ib].qs[iqs * 2 + 1]);
    const uint32_t vui = pack32(quants);
#ifdef DATA_A_Q5_0
    const int32_t qh = int32_t((uint32_t(data_a_packed16[ib].qh[1]) << 16 | data_a_packed16[ib].qh[0]) >> (4 * iqs));
#else // DATA_A_Q5_1
    const int32_t qh = int32_t(data_a_packed32[ib].qh >> (4 * iqs));
#endif
    const int32_t v0 = int32_t(vui & 0x0F0F0F0F)
                     | ((qh & 0xF) * 0x02040810) & 0x10101010; // (0,1,2,3) -> (4,12,20,28)

    const int32_t v1 = int32_t((vui >> 4) & 0x0F0F0F0F)
                     | (((qh >> 16) & 0xF) * 0x02040810) & 0x10101010; // (16,17,18,19) -> (4,12,20,28)

    return i32vec2(v0, v1);
}

#ifdef DATA_A_Q5_0
ACC_TYPE mul_q8_1(const int32_t q_sum, const float da, const vec2 dsb, const int32_t sum_divisor) {
    return ACC_TYPE(da * (float(q_sum) * dsb.x - (16 / sum_divisor) * dsb.y));
}
#else // DATA_A_Q5_1
ACC_TYPE mul_q8_1(const int32_t q_sum, const vec2 dma, const vec2 dsb, const int32_t sum_divisor) {
    return ACC_TYPE(float(q_sum) * dma.x * dsb.x + dma.y * dsb.y / sum_divisor);
}
#endif

#ifdef MMQ_SHMEM
void block_a_to_shmem(const uint buf_ib, const uint ib, const uint iqs) {
#ifdef DATA_A_Q5_0
    buf_a[buf_ib].qs[iqs] = pack32(u16vec2(data_a_packed16[ib].qs[iqs * 2],
                                           data_a_packed16[ib].qs[iqs * 2 + 1]));

    if (iqs == 0) {
        buf_a[buf_ib].dm = FLOAT_TYPE(data_a_packed16[ib].d);
        buf_a[buf_ib].qh = pack32(u16vec2(data_a_packed16[ib].qh[0], data_a_packed16[ib].qh[1]));
    }
#else // DATA_A_Q5_1
    buf_a[buf_ib].qs[iqs] = data_a_packed32[ib].qs[iqs];

    if (iqs == 0) {
        buf_a[buf_ib].dm = FLOAT_TYPE_VEC2(data_a_packed32[ib].dm);
        buf_a[buf_ib].qh = data_a_packed32[ib].qh;
    }
#endif
}

void block_a_to_registers(const uint reg_ib, const uint buf_ib) {
    cache_a[reg_ib].dm = buf_a[buf_ib].dm;
    cache_a[reg_ib].qh = buf_a[buf_ib].qh;

    [[unroll]] for (uint iqs = 0; iqs < 4; iqs++) {
        cache_a[reg_ib].qs[iqs] = buf_a[buf_ib].qs[iqs];
    }
}

ACC_TYPE mmq_dot_product(const uint ib_a) {
    int32_t q_sum = 0;
    [[unroll]] for (uint iqs = 0; iqs < 4; iqs++) {
        const uint32_t vui = cache_a[ib_a].qs[iqs];
        const int32_t qh = int32_t(cache_a[ib_a].qh >> (4 * iqs));
        const int32_t qs_a0 = int32_t(vui & 0x0F0F0F0F)
                         | ((qh & 0xF) * 0x02040810) & 0x10101010; // (0,1,2,3) -> (4,12,20,28)
        const int32_t qs_a1 = int32_t((vui >> 4) & 0x0F0F0F0F)
                         | (((qh >> 16) & 0xF) * 0x02040810) & 0x10101010; // (16,17,18,19) -> (4,12,20,28)

        const int32_t qs_b0 = cache_b.qs[iqs];
        const int32_t qs_b1 = cache_b.qs[iqs + 4];

        q_sum += dotPacked4x8EXT(qs_a0, qs_b0);
        q_sum += dotPacked4x8EXT(qs_a1, qs_b1);
    }

    return mul_q8_1(q_sum, cache_a[ib_a].dm, cache_b.ds, 1);
}
#endif // MMQ_SHMEM
#endif

#if defined(DATA_A_Q8_0)
// 2-byte loads for Q8_0 blocks (34 bytes)
int32_t repack(uint ib, uint iqs) {
    return pack32(i16vec2(data_a_packed16[ib].qs[iqs * 2    ],
                          data_a_packed16[ib].qs[iqs * 2 + 1]));
}

ACC_TYPE mul_q8_1(const int32_t q_sum, const float da, const vec2 dsb, const int32_t sum_divisor) {
    return ACC_TYPE(float(q_sum) * da * dsb.x);
}

#ifdef MMQ_SHMEM
void block_a_to_shmem(const uint buf_ib, const uint ib, const uint iqs) {
    buf_a[buf_ib].qs[iqs] = pack32(i16vec2(data_a_packed16[ib].qs[iqs * 2],
                                           data_a_packed16[ib].qs[iqs * 2 + 1]));

    if (iqs == 0) {
        buf_a[buf_ib].dm = FLOAT_TYPE(data_a_packed16[ib].d);
    }
}

void block_a_to_registers(const uint reg_ib, const uint buf_ib) {
    cache_a[reg_ib].dm = buf_a[buf_ib].dm;

    [[unroll]] for (uint iqs = 0; iqs < 8; iqs++) {
        cache_a[reg_ib].qs[iqs] = buf_a[buf_ib].qs[iqs];
    }
}

ACC_TYPE mmq_dot_product(const uint ib_a) {
    int32_t q_sum = 0;
    [[unroll]] for (uint iqs = 0; iqs < 8; iqs++) {
        const int32_t qs_a = cache_a[ib_a].qs[iqs];
        const int32_t qs_b = cache_b.qs[iqs];

        q_sum += dotPacked4x8EXT(qs_a, qs_b);
    }

    return mul_q8_1(q_sum, cache_a[ib_a].dm, cache_b.ds, 1);
}
#endif // MMQ_SHMEM
#endif

#if defined(DATA_A_MXFP4)
// 1-byte loads for mxfp4 blocks (17 bytes)
i32vec2 repack(uint ib, uint iqs) {
    const uint32_t quants = pack32(u8vec4(data_a[ib].qs[iqs * 4    ],
                                          data_a[ib].qs[iqs * 4 + 1],
                                          data_a[ib].qs[iqs * 4 + 2],
                                          data_a[ib].qs[iqs * 4 + 3]));

    return i32vec2( quants       & 0x0F0F0F0F,
                   (quants >> 4) & 0x0F0F0F0F);
}

ACC_TYPE mul_q8_1(const int32_t q_sum, const float da, const vec2 dsb, const int32_t sum_divisor) {
    return ACC_TYPE(da * dsb.x * float(q_sum));
}

#ifdef MMQ_SHMEM
void block_a_to_shmem(const uint buf_ib, const uint ib, const uint iqs) {
    const uint32_t qs = pack32(u8vec4(data_a[ib].qs[iqs * 4    ],
                                      data_a[ib].qs[iqs * 4 + 1],
                                      data_a[ib].qs[iqs * 4 + 2],
                                      data_a[ib].qs[iqs * 4 + 3]));

    const u8vec4 i_a0 = unpack8( qs       & 0x0F0F0F0F);
    const u8vec4 i_a1 = unpack8((qs >> 4) & 0x0F0F0F0F);

    buf_a[buf_ib].qs[iqs    ] = pack32(i8vec4(kvalues_mxfp4[i_a0.x], kvalues_mxfp4[i_a0.y], kvalues_mxfp4[i_a0.z], kvalues_mxfp4[i_a0.w]));
    buf_a[buf_ib].qs[iqs + 4] = pack32(i8vec4(kvalues_mxfp4[i_a1.x], kvalues_mxfp4[i_a1.y], kvalues_mxfp4[i_a1.z], kvalues_mxfp4[i_a1.w]));

    if (iqs == 0) {
        buf_a[buf_ib].d = FLOAT_TYPE(e8m0_to_fp32(data_a[ib].e) * 0.5);
    }
}

void block_a_to_registers(const uint reg_ib, const uint buf_ib) {
    cache_a[reg_ib].d = buf_a[buf_ib].d;

    [[unroll]] for (uint iqs = 0; iqs < 8; iqs++) {
        cache_a[reg_ib].qs[iqs] = buf_a[buf_ib].qs[iqs];
    }
}

ACC_TYPE mmq_dot_product(const uint ib_a) {
    int32_t q_sum = 0;
    [[unroll]] for (uint iqs = 0; iqs < 8; iqs++) {
        const int32_t qs_a = cache_a[ib_a].qs[iqs];

        q_sum += dotPacked4x8EXT(qs_a, cache_b.qs[iqs]);
    }

    return mul_q8_1(q_sum, cache_a[ib_a].d, cache_b.ds, 1);
}
#endif // MMQ_SHMEM
#endif

// For k-quants, ib and iqs still assume 32-wide blocks, but k-quants are 256-wide
// iqs still refers to a 32-bit integer, meaning 0..7 for 32-wide quants
#if defined(DATA_A_Q2_K)
// 4-byte loads for Q2_K blocks (84 bytes)
int32_t repack(uint ib, uint iqs) {
    const uint ib_k = ib / 8;
    const uint iqs_k = (ib % 8) * 8 + iqs;

    const uint qs_idx = (iqs_k / 32) * 8 + (iqs_k % 8);
    const uint qs_shift = ((iqs_k % 32) / 8) * 2;

    return int32_t((data_a_packed32[ib_k].qs[qs_idx] >> qs_shift) & 0x03030303);
}

uint8_t get_scale(uint ib, uint iqs) {
    const uint ib_k = ib / 8;
    const uint iqs_k = (ib % 8) * 8 + iqs;

    return data_a[ib_k].scales[iqs_k / 4];
}

ACC_TYPE mul_q8_1(const int32_t sum_d, const int32_t sum_m, const vec2 dma, const vec2 dsb, const int32_t sum_divisor) {
    return ACC_TYPE(dsb.x * (dma.x * float(sum_d) - dma.y * float(sum_m)));
}

#ifdef MMQ_SHMEM
void block_a_to_shmem(const uint buf_ib, const uint ib, const uint iqs) {
    const uint ib_k = ib / 8;
    const uint iqs_k = (ib % 8) * 8 + iqs * QUANT_R_MMQ;

    const uint qs_idx = (iqs_k / 32) * 8 + (iqs_k % 8);
    const uint qs_shift = ((iqs_k % 32) / 8) * 2;

    // Repack 4x4 quants into one int
    const uint32_t vals0 = (data_a_packed32[ib_k].qs[qs_idx    ] >> qs_shift) & 0x03030303;
    const uint32_t vals1 = (data_a_packed32[ib_k].qs[qs_idx + 1] >> qs_shift) & 0x03030303;
    const uint32_t vals2 = (data_a_packed32[ib_k].qs[qs_idx + 2] >> qs_shift) & 0x03030303;
    const uint32_t vals3 = (data_a_packed32[ib_k].qs[qs_idx + 3] >> qs_shift) & 0x03030303;

    buf_a[buf_ib].qs[iqs] = vals0 | (vals1 << 2) | (vals2 << 4) | (vals3 << 6);

    if (iqs == 0) {
        buf_a[buf_ib].dm = FLOAT_TYPE_VEC2(data_a_packed32[ib_k].dm);
        buf_a[buf_ib].scales = unpack8(data_a_packed16[ib_k].scales[iqs_k / 8]);
    }
}

void block_a_to_registers(const uint reg_ib, const uint buf_ib) {
    cache_a[reg_ib].dm = buf_a[buf_ib].dm;
    cache_a[reg_ib].scales = buf_a[buf_ib].scales;

    [[unroll]] for (uint iqs = 0; iqs < 2; iqs++) {
        cache_a[reg_ib].qs[iqs] = buf_a[buf_ib].qs[iqs];
    }
}

ACC_TYPE mmq_dot_product(const uint ib_a) {
    int32_t sum_d = 0;
    int32_t sum_m = 0;

    [[unroll]] for (uint iqs = 0; iqs < 8; iqs++) {
        const uint8_t scale = cache_a[ib_a].scales[iqs / 4];
        const int32_t scale_m = int32_t(scale >> 4) * 0x01010101; // Duplicate 8-bit value across 32-bits.
        const int32_t qs_a = int32_t((cache_a[ib_a].qs[iqs / 4] >> ((iqs % 4) * 2)) & 0x03030303);

        sum_d += dotPacked4x8EXT(qs_a, cache_b.qs[iqs]) * (scale & 0xF);
        sum_m += dotPacked4x8EXT(scale_m, cache_b.qs[iqs]);
    }

    return mul_q8_1(sum_d, sum_m, cache_a[ib_a].dm, cache_b.ds, 1);
}
#endif // MMQ_SHMEM
#endif

#if defined(DATA_A_Q3_K)
// 2-byte loads for Q3_K blocks (110 bytes)
#ifdef MMQ_SHMEM
void block_a_to_shmem(const uint buf_ib, const uint ib, const uint iqs) {
    const uint ib_k = ib / 8;
    const uint hm_idx = iqs * QUANT_R_MMQ;
    const uint iqs_k = (ib % 8) * 8 + hm_idx;

    const uint qs_idx = (iqs_k / 32) * 8 + (iqs_k % 8);
    const uint qs_shift = ((iqs_k % 32) / 8) * 2;
    const uint hm_shift = iqs_k / 8;

    // Repack 2x4 quants into one int
    // Add the 3rd bit instead of subtracting it to allow packing the quants
    const i8vec2 vals00 = unpack8(int16_t((data_a_packed16[ib_k].qs[qs_idx * 2        ] >> qs_shift) & uint16_t(0x0303))) |
                          unpack8(int16_t(((data_a_packed16[ib_k].hmask[hm_idx * 2    ] >> hm_shift) & uint16_t(0x0101)) << 2));
    const i8vec2 vals01 = unpack8(int16_t((data_a_packed16[ib_k].qs[qs_idx * 2 + 1    ] >> qs_shift) & uint16_t(0x0303))) |
                          unpack8(int16_t(((data_a_packed16[ib_k].hmask[hm_idx * 2 + 1] >> hm_shift) & uint16_t(0x0101)) << 2));
    const i8vec2 vals10 = unpack8(int16_t((data_a_packed16[ib_k].qs[qs_idx * 2 + 2    ] >> qs_shift) & uint16_t(0x0303))) |
                          unpack8(int16_t(((data_a_packed16[ib_k].hmask[hm_idx * 2 + 2] >> hm_shift) & uint16_t(0x0101)) << 2));
    const i8vec2 vals11 = unpack8(int16_t((data_a_packed16[ib_k].qs[qs_idx * 2 + 3    ] >> qs_shift) & uint16_t(0x0303))) |
                          unpack8(int16_t(((data_a_packed16[ib_k].hmask[hm_idx * 2 + 3] >> hm_shift) & uint16_t(0x0101)) << 2));
    buf_a[buf_ib].qs[iqs] = pack32(u8vec4(vals00.x, vals00.y, vals01.x, vals01.y)) |
                           (pack32(u8vec4(vals10.x, vals10.y, vals11.x, vals11.y)) << 4);

    if (iqs == 0) {
        const uint is = iqs_k / 4;
        const i8vec2 scales = i8vec2(unpack8(((data_a_packed16[ib_k].scales[(is % 8      ) / 2] >> (4 * (is / 8))) & 0x0F0F) |
                                            (((data_a_packed16[ib_k].scales[(8 + (is % 4)) / 2] >> (2 * (is / 4))) & 0x0303) << 4)));

        buf_a[buf_ib].d_scales = FLOAT_TYPE(data_a_packed16[ib_k].d) * FLOAT_TYPE_VEC2(scales - 32);
    }
}

void block_a_to_registers(const uint reg_ib, const uint buf_ib) {
    cache_a[reg_ib].d_scales = buf_a[buf_ib].d_scales;

    [[unroll]] for (uint iqs = 0; iqs < 4; iqs++) {
        cache_a[reg_ib].qs[iqs] = buf_a[buf_ib].qs[iqs];
    }
}

ACC_TYPE mmq_dot_product(const uint ib_a) {
    float result = 0.0;
    int32_t q_sum = 0;

    [[unroll]] for (uint iqs = 0; iqs < 4; iqs++) {
        // Subtract 4 from the quants to correct the 3rd bit offset
        const int32_t qs_a = pack32(unpack8(int32_t((cache_a[ib_a].qs[iqs / 2] >> ((iqs % 2) * 4)) & 0x0F0F0F0F)) - int8_t(4));

        q_sum += dotPacked4x8EXT(qs_a, cache_b.qs[iqs]);
    }
    result += float(cache_a[ib_a].d_scales[0]) * float(q_sum);
    q_sum = 0;

    [[unroll]] for (uint iqs = 4; iqs < 8; iqs++) {
        const int32_t qs_a = pack32(unpack8(int32_t((cache_a[ib_a].qs[iqs / 2] >> ((iqs % 2) * 4)) & 0x0F0F0F0F)) - int8_t(4));

        q_sum += dotPacked4x8EXT(qs_a, cache_b.qs[iqs]);
    }
    result += float(cache_a[ib_a].d_scales[1]) * float(q_sum);

    return ACC_TYPE(cache_b.ds.x * result);
}
#endif // MMQ_SHMEM
#endif

#if defined(DATA_A_Q4_K) || defined(DATA_A_Q5_K)
// 4-byte loads for Q4_K blocks (144 bytes) and Q5_K blocks (176 bytes)
ACC_TYPE mul_q8_1(const int32_t q_sum, const vec2 dma, const vec2 dsb, const int32_t sum_divisor) {
    return ACC_TYPE(dsb.x * dma.x * float(q_sum) - dma.y * dsb.y);
}

#ifdef MMQ_SHMEM
void block_a_to_shmem(const uint buf_ib, const uint ib, const uint iqs) {
    const uint ib_k = ib / 8;
    const uint iqs_k = (ib % 8) * 8 + iqs * QUANT_R_MMQ;

    const uint qs_idx = (iqs_k / 16) * 8 + (iqs_k % 8);
    const uint qs_shift = ((iqs_k % 16) / 8) * 4;

    // Repack 2x4 quants into one int
#if defined(DATA_A_Q4_K)
    const uint32_t vals0 = (data_a_packed32[ib_k].qs[qs_idx    ] >> qs_shift) & 0x0F0F0F0F;
    const uint32_t vals1 = (data_a_packed32[ib_k].qs[qs_idx + 1] >> qs_shift) & 0x0F0F0F0F;

    buf_a[buf_ib].qs[iqs] = vals0 | (vals1 << 4);
#else // defined(DATA_A_Q5_K)
    const uint qh_idx = iqs * QUANT_R_MMQ;
    const uint qh_shift = iqs_k / 8;

    buf_a[buf_ib].qs[iqs] = int32_t(((data_a_packed32[ib_k].qs[qs_idx] >> qs_shift) & 0x0F0F0F0F) |
                                   (((data_a_packed32[ib_k].qh[qh_idx] >> qh_shift) & 0x01010101) << 4));
#endif


    if (iqs == 0) {
        // Scale index
        const uint is = iqs_k / 8;
        u8vec2 scale_dm;
        if (is < 4) {
            scale_dm = u8vec2(data_a[ib_k].scales[is] & 0x3F, data_a[ib_k].scales[is + 4] & 0x3F);
        } else {
            scale_dm = u8vec2((data_a[ib_k].scales[is+4] & 0xF) | ((data_a[ib_k].scales[is-4] & 0xC0) >> 2),
                              (data_a[ib_k].scales[is+4] >>  4) | ((data_a[ib_k].scales[is  ] & 0xC0) >> 2));
        }

        buf_a[buf_ib].dm = FLOAT_TYPE_VEC2(data_a_packed32[ib_k].dm) * FLOAT_TYPE_VEC2(scale_dm);
    }
}

void block_a_to_registers(const uint reg_ib, const uint buf_ib) {
    cache_a[reg_ib].dm = buf_a[buf_ib].dm;

    [[unroll]] for (uint iqs = 0; iqs < 8 / QUANT_R_MMQ; iqs++) {
        cache_a[reg_ib].qs[iqs] = buf_a[buf_ib].qs[iqs];
    }
}

ACC_TYPE mmq_dot_product(const uint ib_a) {
    int32_t q_sum = 0;

    [[unroll]] for (uint iqs = 0; iqs < 8; iqs++) {
#if defined(DATA_A_Q4_K)
        const int32_t qs_a = int32_t((cache_a[ib_a].qs[iqs / 2] >> ((iqs % 2) * 4)) & 0x0F0F0F0F);
#else // defined(DATA_A_Q5_K)
        const int32_t qs_a = cache_a[ib_a].qs[iqs];
#endif

        q_sum += dotPacked4x8EXT(qs_a, cache_b.qs[iqs]);
    }

    return mul_q8_1(q_sum, cache_a[ib_a].dm, cache_b.ds, 1);
}
#endif // MMQ_SHMEM
#endif

#ifdef MMQ_SHMEM
void block_b_to_shmem(const uint buf_ib, const uint ib, const uint iqs) {
    const uint ib_outer = ib / 4;
    const uint ib_inner = ib % 4;

    if (iqs == 0) {
        buf_b[buf_ib].ds = FLOAT_TYPE_VEC2(data_b[ib_outer].ds[ib_inner]);
    }

    const ivec4 values = data_b[ib_outer].qs[ib_inner * 2 + iqs];
    buf_b[buf_ib].qs[iqs * 4    ] = values.x;
    buf_b[buf_ib].qs[iqs * 4 + 1] = values.y;
    buf_b[buf_ib].qs[iqs * 4 + 2] = values.z;
    buf_b[buf_ib].qs[iqs * 4 + 3] = values.w;
}

void block_b_to_registers(const uint ib) {
    cache_b.ds = buf_b[ib].ds;
    [[unroll]] for (uint iqs = 0; iqs < BK / 4; iqs++) {
        cache_b.qs[iqs] = buf_b[ib].qs[iqs];
    }
}
#endif

#if defined(DATA_A_Q6_K)
// 2-byte loads for Q6_K blocks (210 bytes)
#ifdef MMQ_SHMEM
void block_a_to_shmem(const uint buf_ib, const uint ib, const uint iqs) {
    const uint ib_k = ib / 8;
    const uint iqs_k = (ib % 8) * 8 + iqs;

    const uint ql_idx = (iqs_k / 32) * 16 + iqs_k % 16;
    const uint ql_shift = ((iqs_k % 32) / 16) * 4;

    const uint qh_idx = (iqs_k / 32) * 8 + iqs;
    const uint qh_shift = ((iqs_k % 32) / 8) * 2;

    const i8vec2 vals00 = (unpack8(int16_t((data_a_packed16[ib_k].ql[ql_idx * 2    ] >> ql_shift) & uint16_t(0x0F0F))) |
                          unpack8(int16_t(((data_a_packed16[ib_k].qh[qh_idx * 2    ] >> qh_shift) & uint16_t(0x0303)) << 4))) - int8_t(32);
    const i8vec2 vals01 = (unpack8(int16_t((data_a_packed16[ib_k].ql[ql_idx * 2 + 1] >> ql_shift) & uint16_t(0x0F0F))) |
                          unpack8(int16_t(((data_a_packed16[ib_k].qh[qh_idx * 2 + 1] >> qh_shift) & uint16_t(0x0303)) << 4))) - int8_t(32);
    buf_a[buf_ib].qs[iqs] = pack32(i8vec4(vals00.x, vals00.y, vals01.x, vals01.y));

    if (iqs == 0) {
        const uint is = iqs_k / 4;
        const i8vec2 scales = unpack8(data_a_packed16[ib_k].scales[is / 2]);

        buf_a[buf_ib].d_scales = FLOAT_TYPE(data_a_packed16[ib_k].d) * FLOAT_TYPE_VEC2(scales);
    }
}

void block_a_to_registers(const uint reg_ib, const uint buf_ib) {
    cache_a[reg_ib].d_scales = buf_a[buf_ib].d_scales;

    [[unroll]] for (uint iqs = 0; iqs < 8; iqs++) {
        cache_a[reg_ib].qs[iqs] = buf_a[buf_ib].qs[iqs];
    }
}

ACC_TYPE mmq_dot_product(const uint ib_a) {
    float result = 0.0;
    int32_t q_sum = 0;

    [[unroll]] for (uint iqs = 0; iqs < 4; iqs++) {
        const int32_t qs_a = cache_a[ib_a].qs[iqs];

        q_sum += dotPacked4x8EXT(qs_a, cache_b.qs[iqs]);
    }
    result += float(cache_a[ib_a].d_scales[0]) * float(q_sum);
    q_sum = 0;

    [[unroll]] for (uint iqs = 4; iqs < 8; iqs++) {
        const int32_t qs_a = cache_a[ib_a].qs[iqs];

        q_sum += dotPacked4x8EXT(qs_a, cache_b.qs[iqs]);
    }
    result += float(cache_a[ib_a].d_scales[1]) * float(q_sum);

    return ACC_TYPE(cache_b.ds.x * result);
}
#endif // MMQ_SHMEM
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

#if defined(DATA_A_Q2_K)
FLOAT_TYPE_VEC2 get_dm(uint ib) {
    const uint ib_k = ib / 8;
    return FLOAT_TYPE_VEC2(data_a_packed32[ib_k].dm);
}
#endif
