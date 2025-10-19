void load_a_to_shmem(const uint pos_a, const uint row, const uint col, const uint idx_m, const uint block, const uint end_k) {
#if defined(DATA_A_F32) || defined(DATA_A_F16)
#if LOAD_VEC_A == 8
            const uint idx = pos_a + col * p.stride_a / LOAD_VEC_A + row;
            const uint buf_idx = col * SHMEM_STRIDE + row * LOAD_VEC_A / 2;
            FLOAT_TYPE_VEC8 aa = FLOAT_TYPE_VEC8(data_a[idx]);
            buf_a[buf_idx    ] = aa[0].xy;
            buf_a[buf_idx + 1] = aa[0].zw;
            buf_a[buf_idx + 2] = aa[1].xy;
            buf_a[buf_idx + 3] = aa[1].zw;
#elif LOAD_VEC_A == 4
            const uint idx = pos_a + col * p.stride_a / LOAD_VEC_A + row;
            const uint buf_idx = col * SHMEM_STRIDE + row * LOAD_VEC_A / 2;
            FLOAT_TYPE_VEC4 aa = FLOAT_TYPE_VEC4(data_a[idx]);
            buf_a[buf_idx    ] = aa.xy;
            buf_a[buf_idx + 1] = aa.zw;
#else // LOAD_VEC_BATCH_A == 2
            const uint idx = pos_a + col * p.stride_a + row * 2;
            const uint buf_idx = col * SHMEM_STRIDE + row;
            if (idx_m < p.M && block + row * 2 + 1 < end_k) {
                buf_a[buf_idx] = FLOAT_TYPE_VEC2(data_a[idx],
                                                 data_a[idx + 1]);
            } else if (idx_m < p.M && block + row * 2 < end_k) {
                buf_a[buf_idx] = FLOAT_TYPE_VEC2(data_a[idx], 0.0f);
            } else {
                buf_a[buf_idx] = FLOAT_TYPE_VEC2(0.0f);
            }
#endif
#elif defined(DATA_A_BF16)
#if LOAD_VEC_A == 4
            const uint idx = pos_a + col * p.stride_a / LOAD_VEC_A + row;
            const uint buf_idx = col * SHMEM_STRIDE + row * LOAD_VEC_A / 2;
            FLOAT_TYPE_VEC4 aa = FLOAT_TYPE_VEC4(TO_FLOAT_TYPE(data_a[idx]));
            buf_a[buf_idx    ] = aa.xy;
            buf_a[buf_idx + 1] = aa.zw;
#else // LOAD_VEC_BATCH_A == 2
            const uint idx = pos_a + col * p.stride_a + row * 2;
            const uint buf_idx = col * SHMEM_STRIDE + row;
            if (idx_m < p.M && block + row * 2 + 1 < end_k) {
                buf_a[buf_idx] = FLOAT_TYPE_VEC2(TO_FLOAT_TYPE(data_a[idx]),
                                                 TO_FLOAT_TYPE(data_a[idx + 1]));
            } else if (idx_m < p.M && block + row * 2 < end_k) {
                buf_a[buf_idx] = FLOAT_TYPE_VEC2(TO_FLOAT_TYPE(data_a[idx]), 0.0f);
            } else {
                buf_a[buf_idx] = FLOAT_TYPE_VEC2(0.0f);
            }
#endif
#elif defined(DATA_A_Q4_0)
            const uint idx = pos_a + col * p.stride_a / LOAD_VEC_A + row;
            const uint buf_idx = col * SHMEM_STRIDE + 2 * row;

            const uint ib = idx / 4;
            const uint iqs = idx & 0x03;

            const float d = float(data_a_packed16[ib].d);
            const uint vui = uint(data_a_packed16[ib].qs[2*iqs]) | (uint(data_a_packed16[ib].qs[2*iqs + 1]) << 16);
            const vec4 v0 = (vec4(unpack8(vui & 0x0F0F0F0F)) - 8.0f) * d;
            const vec4 v1 = (vec4(unpack8((vui >> 4) & 0x0F0F0F0F)) - 8.0f) * d;

            buf_a[buf_idx    ] = FLOAT_TYPE_VEC2(v0.xy);
            buf_a[buf_idx + 1] = FLOAT_TYPE_VEC2(v0.zw);
            buf_a[buf_idx + 8] = FLOAT_TYPE_VEC2(v1.xy);
            buf_a[buf_idx + 9] = FLOAT_TYPE_VEC2(v1.zw);
#elif defined(DATA_A_Q4_1)
            const uint idx = pos_a + col * p.stride_a / LOAD_VEC_A + row;
            const uint buf_idx = col * SHMEM_STRIDE + 2 * row;

            const uint ib = idx / 4;
            const uint iqs = idx & 0x03;

            const float d = float(data_a_packed16[ib].d);
            const float m = float(data_a_packed16[ib].m);
            const uint vui = uint(data_a_packed16[ib].qs[2*iqs]) | (uint(data_a_packed16[ib].qs[2*iqs + 1]) << 16);
            const vec4 v0 = vec4(unpack8(vui & 0x0F0F0F0F)) * d + m;
            const vec4 v1 = vec4(unpack8((vui >> 4) & 0x0F0F0F0F)) * d + m;

            buf_a[buf_idx     ] = FLOAT_TYPE_VEC2(v0.xy);
            buf_a[buf_idx + 1 ] = FLOAT_TYPE_VEC2(v0.zw);
            buf_a[buf_idx + 8 ] = FLOAT_TYPE_VEC2(v1.xy);
            buf_a[buf_idx + 9 ] = FLOAT_TYPE_VEC2(v1.zw);
#elif defined(DATA_A_Q5_0)
            const uint idx = pos_a + col * p.stride_a / LOAD_VEC_A + row;
            const uint buf_idx = col * SHMEM_STRIDE + row;

            const uint ib = idx / 8;
            const uint iqs = idx & 0x07;

            const float d = float(data_a_packed16[ib].d);
            const uint uint_qh = uint(data_a_packed16[ib].qh[1]) << 16 | uint(data_a_packed16[ib].qh[0]);
            const ivec2 qh0 = ivec2(((uint_qh >> 2*iqs) << 4) & 0x10, (uint_qh >> (2*iqs + 12)) & 0x10);
            const ivec2 qh1 = ivec2(((uint_qh >> (2*iqs + 1)) << 4) & 0x10, (uint_qh >> (2*iqs + 13)) & 0x10);

            const uint vui = uint(data_a_packed16[ib].qs[iqs]);
            const vec4 v = (vec4((vui & 0xF) | qh0.x, ((vui >> 4) & 0xF) | qh0.y, ((vui >> 8) & 0xF) | qh1.x, (vui >> 12) | qh1.y) - 16.0f) * d;

            buf_a[buf_idx    ] = FLOAT_TYPE_VEC2(v.xz);
            buf_a[buf_idx + 8] = FLOAT_TYPE_VEC2(v.yw);
#elif defined(DATA_A_Q5_1)
            const uint idx = pos_a + col * p.stride_a / LOAD_VEC_A + row;
            const uint buf_idx = col * SHMEM_STRIDE + row;

            const uint ib = idx / 8;
            const uint iqs = idx & 0x07;

            const float d = float(data_a_packed16[ib].d);
            const float m = float(data_a_packed16[ib].m);
            const uint uint_qh = data_a_packed16[ib].qh;
            const ivec2 qh0 = ivec2(((uint_qh >> 2*iqs) << 4) & 0x10, (uint_qh >> (2*iqs + 12)) & 0x10);
            const ivec2 qh1 = ivec2(((uint_qh >> (2*iqs + 1)) << 4) & 0x10, (uint_qh >> (2*iqs + 13)) & 0x10);

            const uint vui = uint(data_a_packed16[ib].qs[iqs]);
            const vec4 v = vec4((vui & 0xF) | qh0.x, ((vui >> 4) & 0xF) | qh0.y, ((vui >> 8) & 0xF) | qh1.x, (vui >> 12) | qh1.y) * d + m;

            buf_a[buf_idx    ] = FLOAT_TYPE_VEC2(v.xz);
            buf_a[buf_idx + 8] = FLOAT_TYPE_VEC2(v.yw);
#elif defined(DATA_A_Q8_0)
            const uint idx = pos_a + col * p.stride_a / LOAD_VEC_A + row;
            const uint buf_idx = col * SHMEM_STRIDE + row * LOAD_VEC_A / 2;

            const uint ib = idx / 8;
            const uint iqs = idx & 0x07;

            const float d = float(data_a_packed16[ib].d);
            const i8vec2 v0 = unpack8(int32_t(data_a_packed16[ib].qs[2*iqs])).xy; // vec4 used due to #12147
            const i8vec2 v1 = unpack8(int32_t(data_a_packed16[ib].qs[2*iqs + 1])).xy;
            const vec4 v = vec4(v0.x, v0.y, v1.x, v1.y) * d;

            buf_a[buf_idx    ] = FLOAT_TYPE_VEC2(v.xy);
            buf_a[buf_idx + 1] = FLOAT_TYPE_VEC2(v.zw);
#elif defined(DATA_A_Q2_K)
            const uint idx = pos_a + col * p.stride_a / LOAD_VEC_A + row;
            const uint buf_idx = col * SHMEM_STRIDE + row * LOAD_VEC_A / 2;

            const uint ib = idx / 128;                         // 2 values per idx
            const uint iqs = idx % 128;                        // 0..127

            const uint qsi = (iqs / 64) * 32 + (iqs % 16) * 2; // 0,2,4..30
            const uint scalesi = iqs / 8;                      // 0..15
            const uint qsshift = ((iqs % 64) / 16) * 2;        // 0,2,4,6

            const uvec2 qs = uvec2(data_a[ib].qs[qsi], data_a[ib].qs[qsi + 1]);
            const uint scales = data_a[ib].scales[scalesi];
            const vec2 d = vec2(data_a[ib].d);

            const vec2 v = d.x * float(scales & 0xF) * vec2((qs >> qsshift) & 3) - d.y * float(scales >> 4);

            buf_a[buf_idx] = FLOAT_TYPE_VEC2(v.xy);
#elif defined(DATA_A_Q3_K)
            const uint idx = pos_a + col * p.stride_a / LOAD_VEC_A + row;
            const uint buf_idx = col * SHMEM_STRIDE + row * LOAD_VEC_A / 2;

            const uint ib = idx / 128;                   // 2 values per idx
            const uint iqs = idx % 128;                  // 0..127

            const uint n = iqs / 64;                     // 0,1
            const uint qsi = n * 32 + (iqs % 16) * 2;    // 0,2,4..62
            const uint hmi =          (iqs % 16) * 2;    // 0,2,4..30
            const uint j = (iqs % 64) / 4;               // 0..3
            const uint is = iqs / 8;                     // 0..15
            const uint halfsplit = ((iqs % 64) / 16);    // 0,1,2,3
            const uint qsshift = halfsplit * 2;          // 0,2,4,6
            const uint m = 1 << (4 * n + halfsplit);     // 1,2,4,8,16,32,64,128

            const int8_t us = int8_t(((data_a[ib].scales[is % 8] >> (4 * int(is / 8))) & 0xF)
                                  | (((data_a[ib].scales[8 + (is % 4)] >> (2 * int(is / 4))) & 3) << 4));
            const float dl = float(data_a[ib].d) * float(us - 32);

            buf_a[buf_idx] = FLOAT_TYPE_VEC2(dl * float(int8_t((data_a[ib].qs[qsi    ] >> qsshift) & 3) - (((data_a[ib].hmask[hmi    ] & m) != 0) ? 0 : 4)),
                                             dl * float(int8_t((data_a[ib].qs[qsi + 1] >> qsshift) & 3) - (((data_a[ib].hmask[hmi + 1] & m) != 0) ? 0 : 4)));
#elif defined(DATA_A_Q4_K)
            const uint idx = pos_a + col * p.stride_a / LOAD_VEC_A + row;
            const uint buf_idx = col * SHMEM_STRIDE + row * LOAD_VEC_A / 2;

            const uint ib = idx / 128;                 // 2 values per idx
            const uint iqs = idx % 128;                // 0..127

            const uint n = iqs / 32;                   // 0,1,2,3
            const uint b = (iqs % 32) / 16;            // 0,1
            const uint is = 2 * n + b;                 // 0..7
            const uint qsi = n * 32 + (iqs % 16) * 2;  // 0,2,4..126

            const vec2 loadd = vec2(data_a[ib].d);

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

            const uint8_t sc = uint8_t((data_a[ib].scales[scidx0] & 0xF) | ((data_a[ib].scales[scidx1] & scidxmask1) >> scidxshift1));
            const uint8_t mbyte = uint8_t((data_a[ib].scales[mbidx0] & mbidxmask0) >> mbidxshift0 | ((data_a[ib].scales[mbidx1] & mbidxmask1) >> mbidxshift1));

            const float d = loadd.x * sc;
            const float m = -loadd.y * mbyte;

            buf_a[buf_idx] = FLOAT_TYPE_VEC2(fma(d, float((data_a[ib].qs[qsi    ] >> (b * 4)) & 0xF), m),
                                             fma(d, float((data_a[ib].qs[qsi + 1] >> (b * 4)) & 0xF), m));
#elif defined(DATA_A_Q5_K)
            const uint idx = pos_a + col * p.stride_a / LOAD_VEC_A + row;
            const uint buf_idx = col * SHMEM_STRIDE + row * LOAD_VEC_A / 2;

            const uint ib = idx / 128;                 // 2 values per idx
            const uint iqs = idx % 128;                // 0..127

            const uint n = iqs / 32;                   // 0,1,2,3
            const uint b = (iqs % 32) / 16;            // 0,1
            const uint is = 2 * n + b;                 // 0..7
            const uint qsi = n * 32 + (iqs % 16) * 2;  // 0,2,4..126
            const uint qhi = (iqs % 16) * 2;           // 0,2,4..30

            const uint8_t hm = uint8_t(1 << (iqs / 16));

            const vec2 loadd = vec2(data_a[ib].d);

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

            const uint8_t sc    = uint8_t((data_a[ib].scales[scidx0] & 0xF)                         | ((data_a[ib].scales[scidx1] & scidxmask1) >> scidxshift1));
            const uint8_t mbyte = uint8_t(((data_a[ib].scales[mbidx0] & mbidxmask0) >> mbidxshift0) | ((data_a[ib].scales[mbidx1] & mbidxmask1) >> mbidxshift1));

            const float d = loadd.x * sc;
            const float m = -loadd.y * mbyte;

            buf_a[buf_idx] = FLOAT_TYPE_VEC2(fma(d, float((data_a[ib].qs[qsi    ] >> (b * 4)) & 0xF) + float((data_a[ib].qh[qhi    ] & hm) != 0 ? 16 : 0), m),
                                             fma(d, float((data_a[ib].qs[qsi + 1] >> (b * 4)) & 0xF) + float((data_a[ib].qh[qhi + 1] & hm) != 0 ? 16 : 0), m));
#elif defined(DATA_A_Q6_K)
            const uint idx = pos_a + col * p.stride_a / LOAD_VEC_A + row;
            const uint buf_idx = col * SHMEM_STRIDE + row * LOAD_VEC_A / 2;

            const uint ib = idx / 128;                  // 2 values per idx
            const uint iqs = idx % 128;                 // 0..127

            const uint n = iqs / 64;                    // 0,1
            const uint b = (iqs % 64) / 32;             // 0,1
            const uint is_b = (iqs % 16) / 8;           // 0,1
            const uint qhshift = ((iqs % 64) / 16) * 2; // 0,2,4,6
            const uint is = 8 * n + qhshift + is_b;     // 0..15
            const uint qsi = n * 64 + (iqs % 32) * 2;   // 0,2,4..126
            const uint qhi = n * 32 + (iqs % 16) * 2;   // 0,2,4..62

            const float dscale = float(data_a[ib].d) * float(data_a[ib].scales[is]);

            buf_a[buf_idx] = FLOAT_TYPE_VEC2(dscale * float(int8_t(((data_a[ib].ql[qsi    ] >> (b * 4)) & 0xF) | (((data_a[ib].qh[qhi    ] >> qhshift) & 3) << 4)) - 32),
                                             dscale * float(int8_t(((data_a[ib].ql[qsi + 1] >> (b * 4)) & 0xF) | (((data_a[ib].qh[qhi + 1] >> qhshift) & 3) << 4)) - 32));
#elif defined(DATA_A_IQ1_S)
            const uint idx = pos_a + col * p.stride_a / LOAD_VEC_A + row;
            const uint buf_idx = col * SHMEM_STRIDE + row * LOAD_VEC_A / 2;

            const uint ib = idx / 32;                  // 8 values per idx
            const uint ib32 = (idx % 32) / 4;         // 0..7
            const uint ib8 = idx % 32;

            const float d = float(data_a[ib].d);
            const uint qh = data_a[ib].qh[ib32];
            const uint qs = data_a[ib].qs[ib8];
            const float dl = d * (2 * bitfieldExtract(qh, 12, 3) + 1);
            const float delta = ((qh & 0x8000) != 0) ? -IQ1S_DELTA : IQ1S_DELTA;
            const int16_t grid = int16_t(iq1s_grid[qs | (bitfieldExtract(qh, 3 * int(ib8 & 3), 3) << 8)]);

            [[unroll]] for (int k = 0; k < 4; ++k) {
                buf_a[buf_idx + k] = FLOAT_TYPE_VEC2(dl * (bitfieldExtract(grid, 4 * k    , 2) + delta),
                                                     dl * (bitfieldExtract(grid, 4 * k + 2, 2) + delta));
            }
#elif defined(DATA_A_IQ1_M)
            const uint idx = pos_a + col * p.stride_a / LOAD_VEC_A + row;
            const uint buf_idx = col * SHMEM_STRIDE + row * LOAD_VEC_A / 2;

            const uint ib = idx / 32;  // 8 values per idx
            const uint ib8 = idx % 32;
            const uint ib16 = ib8 / 2;

            const uint16_t[4] scales = data_a[ib].scales;
            const u16vec4 s = u16vec4(scales[0], scales[1], scales[2], scales[3]) >> 12;
            const float d = float(unpackHalf2x16(s.x | (s.y << 4) | (s.z << 8) | (s.w << 12)).x);
            const uint sc = scales[ib8 / 8];
            const uint qs = data_a[ib].qs[ib8];
            const uint qh = data_a[ib].qh[ib16] >> (4 * (ib8 & 1));
            const float dl = d * (2 * bitfieldExtract(sc, 3 * int(ib16 & 3), 3) + 1);
            const float delta = ((qh & 8) != 0) ? -IQ1M_DELTA : IQ1M_DELTA;
            const int16_t grid = int16_t(iq1s_grid[qs | ((qh & 7) << 8)]);

            [[unroll]] for (int k = 0; k < 4; ++k) {
                buf_a[buf_idx + k] = FLOAT_TYPE_VEC2(dl * (bitfieldExtract(grid, 4 * k    , 2) + delta),
                                                     dl * (bitfieldExtract(grid, 4 * k + 2, 2) + delta));
            }
#elif defined(DATA_A_IQ2_XXS)
            const uint idx = pos_a + col * p.stride_a / LOAD_VEC_A + row;
            const uint buf_idx = col * SHMEM_STRIDE + row * LOAD_VEC_A / 2;

            const uint ib = idx / 32;                 // 8 values per idx
            const uint ib32 = (idx % 32) / 4;         // 0..7
            const uint ib8 = idx % 4;

            const float d = float(data_a[ib].d);
            const uint qs = data_a[ib].qs[8 * ib32 + ib8];
            const uint signs = pack32(u8vec4(
                data_a[ib].qs[8*ib32 + 4],
                data_a[ib].qs[8*ib32 + 5],
                data_a[ib].qs[8*ib32 + 6],
                data_a[ib].qs[8*ib32 + 7]
            ));
            const FLOAT_TYPE db = FLOAT_TYPE(d * 0.25 * (0.5 + (signs >> 28)));
            const uint32_t sign7 = bitfieldExtract(signs, 7 * int(ib8), 7);
            const uint sign = sign7 | (bitCount(sign7) << 7);
            const uvec2 grid = iq2xxs_grid[qs];
            const vec4 grid0 = vec4(unpack8(grid.x));
            const vec4 grid1 = vec4(unpack8(grid.y));

            buf_a[buf_idx    ] = db * FLOAT_TYPE_VEC2((sign &   1) != 0 ? -grid0.x : grid0.x,
                                                      (sign &   2) != 0 ? -grid0.y : grid0.y);
            buf_a[buf_idx + 1] = db * FLOAT_TYPE_VEC2((sign &   4) != 0 ? -grid0.z : grid0.z,
                                                      (sign &   8) != 0 ? -grid0.w : grid0.w);
            buf_a[buf_idx + 2] = db * FLOAT_TYPE_VEC2((sign &  16) != 0 ? -grid1.x : grid1.x,
                                                      (sign &  32) != 0 ? -grid1.y : grid1.y);
            buf_a[buf_idx + 3] = db * FLOAT_TYPE_VEC2((sign &  64) != 0 ? -grid1.z : grid1.z,
                                                      (sign & 128) != 0 ? -grid1.w : grid1.w);
#elif defined(DATA_A_IQ2_XS)
            const uint idx = pos_a + col * p.stride_a / LOAD_VEC_A + row;
            const uint buf_idx = col * SHMEM_STRIDE + row * LOAD_VEC_A / 2;

            const uint ib = idx / 32;            // 8 values per idx
            const uint ib32 = (idx % 32) / 4;    // 0..7
            const uint ib8 = idx % 4;            // 0..3

            const float d = float(data_a[ib].d);
            const uint scale = (data_a[ib].scales[ib32] >> (2 * (ib8 & 2))) & 0xf;
            const FLOAT_TYPE db = FLOAT_TYPE(d * 0.25 * (0.5 + scale));
            const uint qs = data_a[ib].qs[4 * ib32 + ib8];
            const uint sign7 = qs >> 9;
            const uint sign = sign7 | (bitCount(sign7) << 7);
            const uvec2 grid = iq2xs_grid[qs & 511];
            const vec4 grid0 = vec4(unpack8(grid.x));
            const vec4 grid1 = vec4(unpack8(grid.y));

            buf_a[buf_idx    ] = db * FLOAT_TYPE_VEC2((sign &   1) != 0 ? -grid0.x : grid0.x,
                                                      (sign &   2) != 0 ? -grid0.y : grid0.y);
            buf_a[buf_idx + 1] = db * FLOAT_TYPE_VEC2((sign &   4) != 0 ? -grid0.z : grid0.z,
                                                      (sign &   8) != 0 ? -grid0.w : grid0.w);
            buf_a[buf_idx + 2] = db * FLOAT_TYPE_VEC2((sign &  16) != 0 ? -grid1.x : grid1.x,
                                                      (sign &  32) != 0 ? -grid1.y : grid1.y);
            buf_a[buf_idx + 3] = db * FLOAT_TYPE_VEC2((sign &  64) != 0 ? -grid1.z : grid1.z,
                                                      (sign & 128) != 0 ? -grid1.w : grid1.w);
#elif defined(DATA_A_IQ2_S)
            const uint idx = pos_a + col * p.stride_a / LOAD_VEC_A + row;
            const uint buf_idx = col * SHMEM_STRIDE + row * LOAD_VEC_A / 2;

            const uint ib = idx / 32;  // 8 values per idx
            const uint ib8 = idx % 32; // 0..31
            const uint ib32 = ib8 / 4; // 0..7

            const uint scale = (data_a[ib].scales[ib32] >> (2 * (ib8 & 2))) & 0xf;
            const uint qs = data_a[ib].qs[ib8];
            const uint qh = data_a[ib].qh[ib32];
            const uint qhshift = 2 * (ib8 % 4);
            const uint sign = data_a[ib].qs[QUANT_K / 8 + ib8];

            const float d = float(data_a[ib].d);
            const FLOAT_TYPE db = FLOAT_TYPE(d * 0.25 * (0.5 + scale));
            const uvec2 grid = iq2s_grid[qs | ((qh << (8 - qhshift)) & 0x300)];
            const vec4 grid0 = vec4(unpack8(grid.x));
            const vec4 grid1 = vec4(unpack8(grid.y));

            buf_a[buf_idx    ] = db * FLOAT_TYPE_VEC2((sign &   1) != 0 ? -grid0.x : grid0.x,
                                                      (sign &   2) != 0 ? -grid0.y : grid0.y);
            buf_a[buf_idx + 1] = db * FLOAT_TYPE_VEC2((sign &   4) != 0 ? -grid0.z : grid0.z,
                                                      (sign &   8) != 0 ? -grid0.w : grid0.w);
            buf_a[buf_idx + 2] = db * FLOAT_TYPE_VEC2((sign &  16) != 0 ? -grid1.x : grid1.x,
                                                      (sign &  32) != 0 ? -grid1.y : grid1.y);
            buf_a[buf_idx + 3] = db * FLOAT_TYPE_VEC2((sign &  64) != 0 ? -grid1.z : grid1.z,
                                                      (sign & 128) != 0 ? -grid1.w : grid1.w);
#elif defined(DATA_A_IQ3_XXS)
            const uint idx = pos_a + col * p.stride_a / LOAD_VEC_A + row;
            const uint buf_idx = col * SHMEM_STRIDE + row * LOAD_VEC_A / 2;

            const uint ib = idx / 64;            // 4 values per idx
            const uint iqs = idx % 64;           // 0..63
            const uint is = QUANT_K / 4 + 4 * (iqs / 8); // 8 values

            const float d = float(data_a[ib].d);
            const uint qs = data_a[ib].qs[iqs];
            const uint signs = pack32(u8vec4(
                data_a[ib].qs[is+0],
                data_a[ib].qs[is+1],
                data_a[ib].qs[is+2],
                data_a[ib].qs[is+3]
            ));
            const float db = d * 0.5 * (0.5 + (signs >> 28));
            const uint32_t sign7 = bitfieldExtract(signs, 7 * (int(iqs / 2) % 4), 7);
            const uint sign = (sign7 | (bitCount(sign7) << 7)) >> (4 * (idx % 2));
            const uint grid = iq3xxs_grid[qs];
            const vec4 v = db * vec4(unpack8(grid));

            buf_a[buf_idx    ] = FLOAT_TYPE_VEC2((sign &   1) != 0 ? -v.x : v.x,
                                                 (sign &   2) != 0 ? -v.y : v.y);
            buf_a[buf_idx + 1] = FLOAT_TYPE_VEC2((sign &   4) != 0 ? -v.z : v.z,
                                                 (sign &   8) != 0 ? -v.w : v.w);
#elif defined(DATA_A_IQ3_S)
            const uint idx = pos_a + col * p.stride_a / LOAD_VEC_A + row;
            const uint buf_idx = col * SHMEM_STRIDE + row * LOAD_VEC_A / 2;

            const uint ib = idx / 64;            // 4 values per idx
            const uint iqs = idx % 64;           // 0..63
            const uint iqh = iqs / 8;

            const float d = float(data_a[ib].d);
            const uint qs = data_a[ib].qs[iqs];
            const uint qh = data_a[ib].qh[iqh];
            const int8_t sign = int8_t(data_a[ib].signs[iqs / 2] >> (4 * (idx % 2)));
            const uint scale = data_a[ib].scales[iqs / 16];
            const i8vec2 sign01 = i8vec2(1 - (2 & i8vec2(sign << 1, sign)));
            const float db = d * (1 + 2 * ((scale >> (4 * (iqh & 1))) & 0xf));
            const uint32_t grid = iq3s_grid[qs | ((qh << (8 - (iqs % 8))) & 256)];
            const vec4 v = db * vec4(unpack8(grid));

            buf_a[buf_idx    ] = FLOAT_TYPE_VEC2((sign &   1) != 0 ? -v.x : v.x,
                                                 (sign &   2) != 0 ? -v.y : v.y);
            buf_a[buf_idx + 1] = FLOAT_TYPE_VEC2((sign &   4) != 0 ? -v.z : v.z,
                                                 (sign &   8) != 0 ? -v.w : v.w);
#elif defined(DATA_A_IQ4_XS)
            const uint idx = pos_a + col * p.stride_a / LOAD_VEC_A + row;
            const uint buf_idx = col * SHMEM_STRIDE + row * LOAD_VEC_A / 2;

            const uint ib = idx / 128;                  // 2 values per idx
            const uint ib32 = (idx % 128) / 16;         // 0..7
            const uint iq = 16 * ib32 + 2 * (idx % 8);

            const uint sl = (data_a[ib].scales_l[ib32/2] >> (4 * (ib32 & 1))) & 0xF;
            const uint sh = ((data_a[ib].scales_h) >> (2 * ib32)) & 3;
            const uint qshift = (idx & 8) >> 1;
            u8vec2 qs = u8vec2(data_a[ib].qs[iq], data_a[ib].qs[iq + 1]);
            qs = (qs >> qshift) & uint8_t(0xF);

            const float d = float(data_a[ib].d);
            const vec2 v = d * float(int(sl | (sh << 4)) - 32) * vec2(kvalues_iq4nl[qs.x], kvalues_iq4nl[qs.y]);

            buf_a[buf_idx    ] = FLOAT_TYPE_VEC2(v.xy);
#elif defined(DATA_A_IQ4_NL)
            const uint idx = pos_a + col * p.stride_a / LOAD_VEC_A + row;
            const uint buf_idx = col * SHMEM_STRIDE + row;

            const uint ib = idx / 8;
            const uint iqs = idx & 0x07;

            const FLOAT_TYPE d = FLOAT_TYPE(data_a_packed16[ib].d);
            const uint vui = uint(data_a_packed16[ib].qs[iqs]);

            buf_a[buf_idx    ] = d * FLOAT_TYPE_VEC2(kvalues_iq4nl[vui & 0xF],
                                                      kvalues_iq4nl[bitfieldExtract(vui, 8, 4)]);
            buf_a[buf_idx + 8] = d * FLOAT_TYPE_VEC2(kvalues_iq4nl[bitfieldExtract(vui, 4, 4)],
                                                     kvalues_iq4nl[vui >> 12]);
#elif defined(DATA_A_MXFP4)
            const uint idx = pos_a + col * p.stride_a / LOAD_VEC_A + row;
            const uint buf_idx = col * SHMEM_STRIDE + row;

            const uint ib = idx / 8;
            const uint iqs = (idx & 0x07) * 2;

            const float d = e8m0_to_fp32(data_a[ib].e);
            const uint vui = uint(data_a[ib].qs[iqs]);
            const uint vui2 = uint(data_a[ib].qs[iqs+1]);

            buf_a[buf_idx    ] = FLOAT_TYPE_VEC2(kvalues_mxfp4[vui  & 0xF] * d,
                                                 kvalues_mxfp4[vui2 & 0xF] * d);
            buf_a[buf_idx + 8] = FLOAT_TYPE_VEC2(kvalues_mxfp4[vui  >>  4] * d,
                                                 kvalues_mxfp4[vui2 >>  4] * d);
#endif
}

#if !defined(MUL_MAT_ID)
void load_b_to_shmem(const uint pos_b, const uint row, const uint col, const uint idx_n, const uint block, const uint end_k) {
#if LOAD_VEC_B == 8
            // Not supported for b_type bf16 because bf16mat2x4 does not exist
            const uint idx = pos_b + col * p.stride_b / LOAD_VEC_B + row;
            const uint buf_idx = col * SHMEM_STRIDE + row * LOAD_VEC_B / 2;
            FLOAT_TYPE_VEC8 bb = FLOAT_TYPE_VEC8(data_b[idx]);
            buf_b[buf_idx + 0] = bb[0].xy;
            buf_b[buf_idx + 1] = bb[0].zw;
            buf_b[buf_idx + 2] = bb[1].xy;
            buf_b[buf_idx + 3] = bb[1].zw;
#elif LOAD_VEC_B == 4
            const uint idx = pos_b + col * p.stride_b / LOAD_VEC_B + row;
            const uint buf_idx = col * SHMEM_STRIDE + row * LOAD_VEC_B / 2;
#if defined(DATA_B_BF16)
            FLOAT_TYPE_VEC4 bb = FLOAT_TYPE_VEC4(TO_FLOAT_TYPE(data_b[idx]));
#else
            FLOAT_TYPE_VEC4 bb = FLOAT_TYPE_VEC4(data_b[idx]);
#endif
            buf_b[buf_idx + 0] = bb.xy;
            buf_b[buf_idx + 1] = bb.zw;
#else // LOAD_VEC_BATCH_B == 2
            const uint idx = pos_b + col * p.stride_b + row * 2;
            const uint buf_idx = col * SHMEM_STRIDE + row;
            if (idx_n < p.N && block + row * 2 + 1 < end_k) {
                buf_b[buf_idx] = FLOAT_TYPE_VEC2(TO_FLOAT_TYPE(data_b[idx]),
                                                 TO_FLOAT_TYPE(data_b[idx + 1]));
            } else if (idx_n < p.N && block + row * 2 < end_k) {
                buf_b[buf_idx] = FLOAT_TYPE_VEC2(TO_FLOAT_TYPE(data_b[idx]), 0.0f);
            } else {
                buf_b[buf_idx] = FLOAT_TYPE_VEC2(0.0f);
            }
#endif
}
#else
void load_b_to_shmem(const uint pos_b, const uint row, const uint col, const uint ic, const uint _ne1, const uint block, const uint end_k) {
#if LOAD_VEC_B == 8
            // Not supported for b_type bf16 because bf16mat2x4 does not exist
            const u16vec2 row_idx = row_ids[col];
            const uint idx = pos_b + row_idx.y * p.batch_stride_b / LOAD_VEC_B + (row_idx.x % p.ne11) * p.stride_b / LOAD_VEC_B + row;
            const uint buf_idx = col * SHMEM_STRIDE + row * LOAD_VEC_B / 2;
            FLOAT_TYPE_VEC8 bb = FLOAT_TYPE_VEC8(data_b[idx]);
            buf_b[buf_idx + 0] = bb[0].xy;
            buf_b[buf_idx + 1] = bb[0].zw;
            buf_b[buf_idx + 2] = bb[1].xy;
            buf_b[buf_idx + 3] = bb[1].zw;
#elif LOAD_VEC_B == 4
            const u16vec2 row_idx = row_ids[col];
            const uint idx = pos_b + row_idx.y * p.batch_stride_b / LOAD_VEC_B + (row_idx.x % p.ne11) * p.stride_b / LOAD_VEC_B + row;
            const uint buf_idx = col * SHMEM_STRIDE + row * LOAD_VEC_B / 2;
#if defined(DATA_B_BF16)
            FLOAT_TYPE_VEC4 bb = FLOAT_TYPE_VEC4(TO_FLOAT_TYPE(data_b[idx]));
#else
            FLOAT_TYPE_VEC4 bb = FLOAT_TYPE_VEC4(data_b[idx]);
#endif
            buf_b[buf_idx + 0] = bb.xy;
            buf_b[buf_idx + 1] = bb.zw;
#else // LOAD_VEC_BATCH_B == 2
            const uint row_i = ic * BN + col;
            const uint buf_idx = col * SHMEM_STRIDE + row;
            if (row_i < _ne1 && block + row * 2 + 1 < end_k) {
                const u16vec2 row_idx = row_ids[col];
                const uint idx = pos_b + row_idx.y * p.batch_stride_b + (row_idx.x % p.ne11) * p.stride_b + row * 2;
                buf_b[buf_idx] = FLOAT_TYPE_VEC2(TO_FLOAT_TYPE(data_b[idx]),
                                                 TO_FLOAT_TYPE(data_b[idx + 1]));
            } else if (row_i < _ne1 && block + row * 2 < end_k) {
                const u16vec2 row_idx = row_ids[col];
                const uint idx = pos_b + row_idx.y * p.batch_stride_b + (row_idx.x % p.ne11) * p.stride_b + row * 2;
                buf_b[buf_idx] = FLOAT_TYPE_VEC2(TO_FLOAT_TYPE(data_b[idx]), 0.0f);
            } else {
                buf_b[buf_idx] = FLOAT_TYPE_VEC2(0.0f);
            }
#endif
}
#endif
