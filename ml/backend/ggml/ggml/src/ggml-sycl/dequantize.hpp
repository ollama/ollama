//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#ifndef GGML_SYCL_DEQUANTIZE_HPP
#define GGML_SYCL_DEQUANTIZE_HPP

#include "common.hpp"

typedef void (*dequantize_kernel_t)(const void * vx, const int64_t ib, const int iqs, dfloat2 & v);
typedef void (*dequantize_kernel_t_reorder)(const void *d, const int64_t ib, const void *qs,
                                            const int iqs, dfloat2 &v);

static __dpct_inline__ void dequantize_q4_0(const void *vx, const int64_t ib,
                                            const int iqs, dfloat2 &v) {
    const block_q4_0 * x = (const block_q4_0 *) vx;

    const dfloat d = x[ib].d;

    const int vui = x[ib].qs[iqs];

    v.x() = vui & 0xF;
    v.y() = vui >> 4;

#ifdef GGML_SYCL_F16
    // v = v - {8.0f, 8.0f};
    // v = v * {d, d};
    v.s0() = (v.s0() - 8.0f) * d;
    v.s1() = (v.s1() - 8.0f) * d;

#else
    v.x() = (v.x() - 8.0f) * d;
    v.y() = (v.y() - 8.0f) * d;
#endif // GGML_SYCL_F16
}

static __dpct_inline__ void dequantize_q4_0_reorder(const void *d_ptr, const int64_t ib, const void *qs,
                                            const int iqs, dfloat2 &v) {
    // const block_q4_0 * x = (const block_q4_0 *) vx;

    const dfloat d = (const dfloat)*((const sycl::half*)d_ptr+ib);

    const int vui = *((const uint8_t *)qs+iqs);

    v.x() = vui & 0xF;
    v.y() = vui >> 4;

#ifdef GGML_SYCL_F16
    // v = v - {8.0f, 8.0f};
    // v = v * {d, d};
    v.s0() = (v.s0() - 8.0f) * d;
    v.s1() = (v.s1() - 8.0f) * d;

#else
    v.x() = (v.x() - 8.0f) * d;
    v.y() = (v.y() - 8.0f) * d;
#endif // GGML_SYCL_F16
}

static __dpct_inline__ void dequantize_q4_1(const void *vx, const int64_t ib,
                                            const int iqs, dfloat2 &v) {
    const block_q4_1 * x = (const block_q4_1 *) vx;

    const dfloat d = x[ib].dm[0];
    const dfloat m = x[ib].dm[1];

    const int vui = x[ib].qs[iqs];

    v.x() = vui & 0xF;
    v.y() = vui >> 4;

#ifdef GGML_SYCL_F16
    // v = v * {d, d};
    // v = v + {m, m};
    v.s0() = sycl::fma(v.s0(), d, m);
    v.s1() = sycl::fma(v.s1(), d, m);

#else
    v.x() = sycl::fma(v.x(), d, m);
    v.y() = sycl::fma(v.y(), d, m);
#endif // GGML_SYCL_F16
}

static __dpct_inline__ void dequantize_q5_0(const void *vx, const int64_t ib,
                                            const int iqs, dfloat2 &v) {
    const block_q5_0 * x = (const block_q5_0 *) vx;

    const dfloat d = x[ib].d;

    uint32_t qh;
    memcpy(&qh, x[ib].qh, sizeof(qh));

    const int xh_0 = ((qh >> (iqs +  0)) << 4) & 0x10;
    const int xh_1 = ((qh >> (iqs + 12))     ) & 0x10;

    v.x() = ((x[ib].qs[iqs] & 0xf) | xh_0);
    v.y() = ((x[ib].qs[iqs] >> 4) | xh_1);

#ifdef GGML_SYCL_F16
    // v = v - {16.0f, 16.0f};
    // v = v * {d, d};
    v.s0() = (v.s0() - 16.0f) * d;
    v.s1() = (v.s1() - 16.0f) * d;

#else
    v.x() = (v.x() - 16.0f) * d;
    v.y() = (v.y() - 16.0f) * d;
#endif // GGML_SYCL_F16
}

static __dpct_inline__ void dequantize_q5_1(const void *vx, const int64_t ib,
                                            const int iqs, dfloat2 &v) {
    const block_q5_1 * x = (const block_q5_1 *) vx;

    const dfloat d = x[ib].dm[0];
    const dfloat m = x[ib].dm[1];

    uint32_t qh;
    memcpy(&qh, x[ib].qh, sizeof(qh));

    const int xh_0 = ((qh >> (iqs +  0)) << 4) & 0x10;
    const int xh_1 = ((qh >> (iqs + 12))     ) & 0x10;

    v.x() = ((x[ib].qs[iqs] & 0xf) | xh_0);
    v.y() = ((x[ib].qs[iqs] >> 4) | xh_1);

#ifdef GGML_SYCL_F16
    // v = v * {d, d};
    // v = v + {m, m};
    v.s0() = sycl::fma(v.s0(), d, m);
    v.s1() = sycl::fma(v.s1(), d, m);
#else
    v.x() = sycl::fma(v.x(), d, m);
    v.y() = sycl::fma(v.y(), d, m);
#endif // GGML_SYCL_F16
}

static __dpct_inline__ void dequantize_q8_0(const void *vx, const int64_t ib,
                                            const int iqs, dfloat2 &v) {
    const block_q8_0 * x = (const block_q8_0 *) vx;

    const dfloat d = x[ib].d;

    v.x() = x[ib].qs[iqs + 0];
    v.y() = x[ib].qs[iqs + 1];

#ifdef GGML_SYCL_F16
    // v = v * {d, d};
    v.s0() *= d;
    v.s1() *= d;
#else
    v.x() *= d;
    v.y() *= d;
#endif // GGML_SYCL_F16
}

template<typename dst_t>
static void dequantize_block_q4_0(const void * __restrict__ vx, dst_t * __restrict__ yy, int64_t nb32,
                                  const sycl::nd_item<3> &item_ct1) {

    const int64_t i = item_ct1.get_group(2);

    // assume 32 threads
    const int64_t tid = item_ct1.get_local_id(2);
    const int64_t il  = tid/8;
    const int64_t ir  = tid%8;
    const int64_t ib = 8*i + ir;
    if (ib >= nb32) {
        return;
    }

    dst_t * y = yy + 256*i + 32*ir + 4*il;

    const block_q4_0 * x = (const block_q4_0 *)vx + ib;
    const float d = sycl::vec<sycl::half, 1>(x->d)
                        .convert<float, sycl::rounding_mode::automatic>()[0];
    const float dm = -8*d;

    const uint8_t * q = x->qs + 4*il;

    for (int l = 0; l < 4; ++l) {
        y[l+ 0] = d * (q[l] & 0xF) + dm;
        y[l+16] = d * (q[l] >>  4) + dm;
    }
}

template<typename dst_t>
static void dequantize_block_q4_0_reorder(const void * __restrict__ vx, dst_t * __restrict__ yy, int64_t nb32,
                                  const sycl::nd_item<3> &item_ct1) {

    const int64_t i = item_ct1.get_group(2);
    auto k=nb32;
    // assume 32 threads
    const int64_t tid = item_ct1.get_local_id(2);
    const int lane_ib = i * WARP_SIZE + tid;

    if (lane_ib >= k / QK4_0) {
        return;
    }

    dst_t * y_ptr = yy + lane_ib * QK4_0;

    auto qs = (const uint8_t*)vx + lane_ib * QK4_0 / 2;
    auto s_ptr = (const sycl::half*)((const uint8_t*)vx + k / 2) + lane_ib;

    const float d = float(*s_ptr);

#pragma unroll
    for (int l = 0; l < QK4_0 / 2; ++l) {
        int vq = qs[l];
        y_ptr[l + 0] = d * ((vq & 0xF) - 8);
        y_ptr[l + 16] = d * ((vq >> 4) - 8);
    }

}

template<typename dst_t>
static void dequantize_block_q4_1(const void * __restrict__ vx, dst_t * __restrict__ yy, int64_t nb32,
                                  const sycl::nd_item<3> &item_ct1) {

    const int64_t i = item_ct1.get_group(2);

    // assume 32 threads
    const int64_t tid = item_ct1.get_local_id(2);
    const int64_t il  = tid/8;
    const int64_t ir  = tid%8;
    const int64_t ib = 8*i + ir;
    if (ib >= nb32) {
        return;
    }

    dst_t * y = yy + 256*i + 32*ir + 4*il;

    const block_q4_1 * x = (const block_q4_1 *)vx + ib;
    const sycl::float2 d =
        x->dm.convert<float, sycl::rounding_mode::automatic>();

    const uint8_t * q = x->qs + 4*il;

    for (int l = 0; l < 4; ++l) {
        y[l + 0] = d.x() * (q[l] & 0xF) + d.y();
        y[l + 16] = d.x() * (q[l] >> 4) + d.y();
    }
}


//================================== k-quants

template<typename dst_t>
static void dequantize_block_q2_K(const void * __restrict__ vx, dst_t * __restrict__ yy,
                                  const sycl::nd_item<3> &item_ct1) {

    const int64_t i = item_ct1.get_group(2);
    const block_q2_K * x = (const block_q2_K *) vx;

    const int64_t tid = item_ct1.get_local_id(2);
#if QK_K == 256
    const int64_t n   = tid/32;
    const int64_t l   = tid - 32*n;
    const int64_t is  = 8*n + l/16;

    const uint8_t q = x[i].qs[32*n + l];
    dst_t * y = yy + i*QK_K + 128*n;

    float dall = x[i].dm[0];
    float dmin = x[i].dm[1];
    y[l+ 0] = dall * (x[i].scales[is+0] & 0xF) * ((q >> 0) & 3) - dmin * (x[i].scales[is+0] >> 4);
    y[l+32] = dall * (x[i].scales[is+2] & 0xF) * ((q >> 2) & 3) - dmin * (x[i].scales[is+2] >> 4);
    y[l+64] = dall * (x[i].scales[is+4] & 0xF) * ((q >> 4) & 3) - dmin * (x[i].scales[is+4] >> 4);
    y[l+96] = dall * (x[i].scales[is+6] & 0xF) * ((q >> 6) & 3) - dmin * (x[i].scales[is+6] >> 4);
#else
    const int64_t is = tid/16;  // 0 or 1
    const int64_t il = tid%16;  // 0...15
    const uint8_t q = x[i].qs[il] >> (2*is);
    dst_t * y = yy + i*QK_K + 16*is + il;

    float dall = x[i].dm[0];
    float dmin = x[i].dm[1];
    y[ 0] = dall * (x[i].scales[is+0] & 0xF) * ((q >> 0) & 3) - dmin * (x[i].scales[is+0] >> 4);
    y[32] = dall * (x[i].scales[is+2] & 0xF) * ((q >> 4) & 3) - dmin * (x[i].scales[is+2] >> 4);
#endif

}

template<typename dst_t>
static void dequantize_block_q3_K(const void * __restrict__ vx, dst_t * __restrict__ yy,
                                  const sycl::nd_item<3> &item_ct1) {

    const int64_t i = item_ct1.get_group(2);
    const block_q3_K * x = (const block_q3_K *) vx;

#if QK_K == 256
    const int64_t r = item_ct1.get_local_id(2) / 4;
    const int64_t tid = r/2;
    const int64_t is0 = r%2;
    const int64_t l0 = 16 * is0 + 4 * (item_ct1.get_local_id(2) % 4);
    const int64_t n = tid / 4;
    const int64_t j = tid - 4*n;

    uint8_t m = 1 << (4*n + j);
    int64_t is = 8*n + 2*j + is0;
    int shift = 2*j;

    int8_t us = is <  4 ? (x[i].scales[is-0] & 0xF) | (((x[i].scales[is+8] >> 0) & 3) << 4) :
                is <  8 ? (x[i].scales[is-0] & 0xF) | (((x[i].scales[is+4] >> 2) & 3) << 4) :
                is < 12 ? (x[i].scales[is-8] >>  4) | (((x[i].scales[is+0] >> 4) & 3) << 4) :
                          (x[i].scales[is-8] >>  4) | (((x[i].scales[is-4] >> 6) & 3) << 4);
    float d_all = x[i].d;
    float dl = d_all * (us - 32);

    dst_t * y = yy + i*QK_K + 128*n + 32*j;
    const uint8_t * q = x[i].qs + 32*n;
    const uint8_t * hm = x[i].hmask;

    for (int l = l0; l < l0+4; ++l) y[l] = dl * ((int8_t)((q[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4));
#else
    const int64_t tid = item_ct1.get_local_id(2);
    const int64_t is  = tid/16;  // 0 or 1
    const int64_t il  = tid%16;  // 0...15
    const int64_t im  = il/8;    // 0...1
    const int64_t in  = il%8;    // 0...7

    dst_t * y = yy + i*QK_K + 16*is + il;

    const uint8_t q = x[i].qs[il] >> (2*is);
    const uint8_t h = x[i].hmask[in] >> (2*is + im);
    const float   d = (float)x[i].d;

    if (is == 0) {
        y[ 0] = d * ((x[i].scales[0] & 0xF) - 8) * ((int8_t)((q >> 0) & 3) - ((h >> 0) & 1 ? 0 : 4));
        y[32] = d * ((x[i].scales[1] & 0xF) - 8) * ((int8_t)((q >> 4) & 3) - ((h >> 4) & 1 ? 0 : 4));
    } else {
        y[ 0] = d * ((x[i].scales[0] >>  4) - 8) * ((int8_t)((q >> 0) & 3) - ((h >> 0) & 1 ? 0 : 4));
        y[32] = d * ((x[i].scales[1] >>  4) - 8) * ((int8_t)((q >> 4) & 3) - ((h >> 4) & 1 ? 0 : 4));
    }
#endif

}

#if QK_K == 256
static inline void get_scale_min_k4(int j, const uint8_t * q, uint8_t & d, uint8_t & m) {
    if (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
        m = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4);
    }
}
#endif

template <typename dst_t>
inline void dequantize_q4_K_common(dst_t * __restrict__ y, const uint8_t * __restrict__ qs_ptr, const float dall,
                                   const float dmin, uint8_t * __restrict__ scales_local, int il, int ir) {
    const int is = 2 * il;
    constexpr int n  = 4;

    uint8_t sc, m;
    get_scale_min_k4(is + 0, scales_local, sc, m);
    const float d1 = dall * sc;
    const float m1 = dmin * m;

    get_scale_min_k4(is + 1, scales_local, sc, m);
    const float d2 = dall * sc;
    const float m2 = dmin * m;

    sycl::vec<uint8_t, n> q_vec = vec_aligned_load<uint8_t, n>(qs_ptr + 32 * il + n * ir);
    for (int l = 0; l < n; ++l) {
        y[l + 0]  = d1 * (q_vec[l] & 0xF) - m1;
        y[l + 32] = d2 * (q_vec[l] >> 4) - m2;
    }
}

template<typename dst_t>
static void dequantize_block_q4_K(const void * __restrict__ vx, dst_t * __restrict__ yy,
                                  uint8_t* scales_local, const sycl::nd_item<3> &item_ct1) {
    const block_q4_K * x = (const block_q4_K *) vx;

    const int64_t i = item_ct1.get_group(2);

#if QK_K == 256
    const int64_t tid = item_ct1.get_local_id(2);
    const int64_t il  = tid / 8;
    const int64_t ir  = tid % 8;

    dst_t * y = yy + i * QK_K + 64 * il + 4 * ir;

    const sycl::half2 dm = x[i].dm;
    const float dall = dm[0];
    const float dmin = dm[1];

    if (tid < 12) {
        scales_local[tid] = x[i].scales[tid];
    }

    item_ct1.barrier(sycl::access::fence_space::local_space);
    dequantize_q4_K_common(y, x[i].qs, dall, dmin, scales_local, il, ir);
#else
    const int64_t tid = item_ct1.get_local_id(2);
    const uint8_t * q = x[i].qs;
    dst_t * y = yy + i*QK_K;
    const float d = (float)x[i].dm[0];
    const float m = (float)x[i].dm[1];
    y[tid+ 0] = d * (x[i].scales[0] & 0xF) * (q[tid] & 0xF) - m * (x[i].scales[0] >> 4);
    y[tid+32] = d * (x[i].scales[1] & 0xF) * (q[tid] >>  4) - m * (x[i].scales[1] >> 4);
#endif
}

template <typename dst_t>
static void dequantize_block_q4_K_reorder(const void * __restrict__ vx, dst_t * __restrict__ yy, uint8_t * scales_local,
                                          const sycl::nd_item<1> & item_ct1, int64_t nb) {
    const int64_t i   = item_ct1.get_group(0);     // block index
    const int64_t tid = item_ct1.get_local_id(0);  // thread index within block
    const int64_t il  = tid / 8;
    const int64_t ir  = tid % 8;

    dst_t * y = yy + i * QK_K + 64 * il + 4 * ir;

    const uint8_t * base          = static_cast<const uint8_t *>(vx);
    const size_t    qs_offset     = i * (QK_K / 2);
    const size_t    scales_offset = nb * (QK_K / 2) + i * K_SCALE_SIZE;
    const size_t    dm_offset     = nb * (QK_K / 2) + nb * K_SCALE_SIZE + i * sizeof(ggml_half2);

    const uint8_t *    qs_ptr     = base + qs_offset;
    const uint8_t *    scales_ptr = base + scales_offset;
    ggml_half2         dm_values  = *reinterpret_cast<const ggml_half2 *>(base + dm_offset);

    const float dall = dm_values.x();
    const float dmin = dm_values.y();

    if (tid < 12) {
        scales_local[tid] = scales_ptr[tid];
    }

    item_ct1.barrier(sycl::access::fence_space::local_space);
    dequantize_q4_K_common(y, qs_ptr, dall, dmin, scales_local, il, ir);
}

template<typename dst_t>
static void dequantize_block_q5_K(const void * __restrict__ vx, dst_t * __restrict__ yy,
                                  const sycl::nd_item<3> &item_ct1) {
    const block_q5_K * x = (const block_q5_K *) vx;

    const int64_t i = item_ct1.get_group(2);

#if QK_K == 256
    // assume 64 threads - this is very slightly better than the one below
    const int64_t tid = item_ct1.get_local_id(2);
    const int64_t il  = tid/16;   // il is in 0...3
    const int64_t ir  = tid%16;   // ir is in 0...15
    const int64_t is  = 2*il;     // is is in 0...6

    dst_t * y = yy + i*QK_K + 64*il + 2*ir;

    const float dall = x[i].dm[0];
    const float dmin = x[i].dm[1];

    const uint8_t * ql = x[i].qs + 32*il + 2*ir;
    const uint8_t * qh = x[i].qh + 2*ir;

    uint8_t sc, m;
    get_scale_min_k4(is + 0, x[i].scales, sc, m);
    const float d1 = dall * sc; const float m1 = dmin * m;
    get_scale_min_k4(is + 1, x[i].scales, sc, m);
    const float d2 = dall * sc; const float m2 = dmin * m;

    uint8_t   hm  = 1 << (2*il);
    y[ 0] = d1 * ((ql[ 0] & 0xF) + (qh[ 0] & hm ? 16 : 0)) - m1;
    y[ 1] = d1 * ((ql[ 1] & 0xF) + (qh[ 1] & hm ? 16 : 0)) - m1;
    hm <<= 1;
    y[32] = d2 * ((ql[ 0] >>  4) + (qh[ 0] & hm ? 16 : 0)) - m2;
    y[33] = d2 * ((ql[ 1] >>  4) + (qh[ 1] & hm ? 16 : 0)) - m2;
#else
    const int64_t tid = item_ct1.get_local_id(2);
    const uint8_t q = x[i].qs[tid];
    const int64_t im = tid/8;  // 0...3
    const int64_t in = tid%8;  // 0...7
    const int64_t is = tid/16; // 0 or 1
    const uint8_t h = x[i].qh[in] >> im;
    const float d = x[i].d;
    dst_t * y = yy + i*QK_K + tid;
    y[ 0] = d * x[i].scales[is+0] * ((q & 0xF) - ((h >> 0) & 1 ? 0 : 16));
    y[32] = d * x[i].scales[is+2] * ((q >>  4) - ((h >> 4) & 1 ? 0 : 16));
#endif
}

template<typename dst_t>
static void dequantize_block_q6_K(const void * __restrict__ vx, dst_t * __restrict__ yy,
                                  const sycl::nd_item<3> &item_ct1) {
    const block_q6_K * x = (const block_q6_K *) vx;

    const int64_t i = item_ct1.get_group(2);
#if QK_K == 256

    // assume 64 threads - this is very slightly better than the one below
    const int64_t tid = item_ct1.get_local_id(2);
    const int64_t ip  = tid/32;   // ip is 0 or 1
    const int64_t il  = tid - 32*ip; // 0...32
    const int64_t is  = 8*ip + il/16;

    dst_t * y = yy + i*QK_K + 128*ip + il;

    const float d = x[i].d;

    const uint8_t * ql = x[i].ql + 64*ip + il;
    const uint8_t   qh = x[i].qh[32*ip + il];
    const int8_t  * sc = x[i].scales + is;

    y[ 0] = d * sc[0] * ((int8_t)((ql[ 0] & 0xF) | (((qh >> 0) & 3) << 4)) - 32);
    y[32] = d * sc[2] * ((int8_t)((ql[32] & 0xF) | (((qh >> 2) & 3) << 4)) - 32);
    y[64] = d * sc[4] * ((int8_t)((ql[ 0]  >> 4) | (((qh >> 4) & 3) << 4)) - 32);
    y[96] = d * sc[6] * ((int8_t)((ql[32]  >> 4) | (((qh >> 6) & 3) << 4)) - 32);
#else

    // assume 32 threads
    const int64_t tid = item_ct1.get_local_id(2);
    const int64_t ip  = tid/16;         // 0 or 1
    const int64_t il  = tid - 16*ip;    // 0...15

    dst_t * y = yy + i*QK_K + 16*ip + il;

    const float d = x[i].d;

    const uint8_t   ql = x[i].ql[16*ip + il];
    const uint8_t   qh = x[i].qh[il] >> (2*ip);
    const int8_t  * sc = x[i].scales;

    y[ 0] = d * sc[ip+0] * ((int8_t)((ql & 0xF) | (((qh >> 0) & 3) << 4)) - 32);
    y[32] = d * sc[ip+2] * ((int8_t)((ql  >> 4) | (((qh >> 4) & 3) << 4)) - 32);
#endif
}

template<typename dst_t>
static void dequantize_block_iq2_xxs(const void * __restrict__ vx, dst_t * __restrict__ yy,
                                     const sycl::nd_item<3> &item_ct1,
                                     const uint64_t *iq2xxs_grid_ptr,
                                     const uint8_t *ksigns_iq2xs_ptr,
                                     const uint8_t *kmask_iq2xs_ptr) {

    const int64_t i = item_ct1.get_group(2);
    const block_iq2_xxs * x = (const block_iq2_xxs  *) vx;

    const int64_t tid = item_ct1.get_local_id(2);
#if QK_K == 256
    const int64_t il = tid/8; // 0...3
    const int64_t ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 8*il;
    const uint16_t * q2 = x[i].qs + 4*ib;
    const uint8_t  * aux8 = (const uint8_t *)q2;
    const uint8_t  * grid = (const uint8_t *)(iq2xxs_grid_ptr + aux8[il]);
    const uint32_t aux32 = q2[2] | (q2[3] << 16);
    const float d = (float)x[i].d * (0.5f + (aux32 >> 28)) * 0.25f;
    const uint8_t signs = ksigns_iq2xs_ptr[(aux32 >> 7*il) & 127];
    for (int j = 0; j < 8; ++j) y[j] = d * grid[j] * (signs & kmask_iq2xs_ptr[j] ? -1.f : 1.f);
#else
    assert(false);
#endif

}

template<typename dst_t>
static void dequantize_block_iq2_xs(const void * __restrict__ vx, dst_t * __restrict__ yy,
                                    const sycl::nd_item<3> &item_ct1,
                                    const uint64_t *iq2xs_grid,
                                    const uint8_t *ksigns_iq2xs,
                                    const uint8_t *kmask_iq2xs) {

    const int64_t i = item_ct1.get_group(2);
    const block_iq2_xs * x = (const block_iq2_xs *) vx;

    const int64_t tid = item_ct1.get_local_id(2);
#if QK_K == 256
    const int64_t il = tid/8; // 0...3
    const int64_t ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 8*il;
    const uint16_t * q2 = x[i].qs + 4*ib;
    const uint8_t  * grid = (const uint8_t *)(iq2xs_grid + (q2[il] & 511));
    const float d = (float)x[i].d * (0.5f + ((x[i].scales[ib] >> 4*(il/2)) & 0xf)) * 0.25f;
    const uint8_t signs = ksigns_iq2xs[q2[il] >> 9];
    for (int j = 0; j < 8; ++j) y[j] = d * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
#else
    assert(false);
#endif

}

template <typename dst_t>
__dpct_inline__ static void
dequantize_block_iq2_s(const void *__restrict__ vx, dst_t *__restrict__ yy,
                       const sycl::nd_item<3> &item_ct1) {

    const int64_t i = item_ct1.get_group(2);
    const block_iq2_s * x = (const block_iq2_s *) vx;

    const int64_t tid = item_ct1.get_local_id(2);
#if QK_K == 256
    const int64_t il = tid/8; // 0...3
    const int64_t ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 8*il;
    const uint8_t * grid = (const uint8_t *)(iq2s_grid + (x[i].qs[4*ib+il] | ((x[i].qh[ib] << (8-2*il)) & 0x300)));
    const float d = (float)x[i].d * (0.5f + ((x[i].scales[ib] >> 4*(il/2)) & 0xf)) * 0.25f;
    const uint8_t signs = x[i].qs[QK_K/8+4*ib+il];
#pragma unroll
    for (int j = 0; j < 8; ++j)
        y[j] = d * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
#else
    assert(false);

#endif

}

template<typename dst_t>
static void dequantize_block_iq3_xxs(const void * __restrict__ vx, dst_t * __restrict__ yy,
                                     const sycl::nd_item<3> &item_ct1,
                                     const uint32_t *iq3xxs_grid,
                                     const uint8_t *ksigns_iq2xs,
                                     const uint8_t *kmask_iq2xs) {

    const int64_t i = item_ct1.get_group(2);
    const block_iq3_xxs * x = (const block_iq3_xxs  *) vx;

    const int64_t tid = item_ct1.get_local_id(2);
#if QK_K == 256
    const int64_t il = tid/8; // 0...3
    const int64_t ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 8*il;
    const uint8_t  * q3 = x[i].qs + 8*ib;
    const uint16_t * gas = (const uint16_t *)(x[i].qs + QK_K/4) + 2*ib;
    const uint8_t  * grid1 = (const uint8_t *)(iq3xxs_grid + q3[2*il+0]);
    const uint8_t  * grid2 = (const uint8_t *)(iq3xxs_grid + q3[2*il+1]);
    const uint32_t aux32 = gas[0] | (gas[1] << 16);
    const float d = (float)x[i].d * (0.5f + (aux32 >> 28)) * 0.5f;
    const uint8_t signs = ksigns_iq2xs[(aux32 >> 7*il) & 127];
    for (int j = 0; j < 4; ++j) {
        y[j+0] = d * grid1[j] * (signs & kmask_iq2xs[j+0] ? -1.f : 1.f);
        y[j+4] = d * grid2[j] * (signs & kmask_iq2xs[j+4] ? -1.f : 1.f);
    }
#else
    assert(false);
#endif

}

template <typename dst_t>
__dpct_inline__ static void
dequantize_block_iq3_s(const void *__restrict__ vx, dst_t *__restrict__ yy,
                       const sycl::nd_item<3> &item_ct1,
                       const uint8_t *kmask_iq2xs, const uint32_t *iq3s_grid) {

    const int64_t i = item_ct1.get_group(2);
    const block_iq3_s * x = (const block_iq3_s *) vx;

    const int64_t tid = item_ct1.get_local_id(2);
#if QK_K == 256
    const int64_t il = tid/8; // 0...3
    const int64_t ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 8*il;
    const uint8_t * qs = x[i].qs + 8*ib;
    const uint8_t * grid1 = (const uint8_t *)(iq3s_grid + (qs[2*il+0] | ((x[i].qh[ib] << (8-2*il)) & 256)));
    const uint8_t * grid2 = (const uint8_t *)(iq3s_grid + (qs[2*il+1] | ((x[i].qh[ib] << (7-2*il)) & 256)));
    const float d = (float)x[i].d * (1 + 2*((x[i].scales[ib/2] >> 4*(ib%2)) & 0xf));
    const uint8_t signs = x[i].signs[4*ib + il];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        y[j+0] = d * grid1[j] * (signs & kmask_iq2xs[j+0] ? -1.f : 1.f);
        y[j+4] = d * grid2[j] * (signs & kmask_iq2xs[j+4] ? -1.f : 1.f);
    }
#else
    assert(false);
#endif

}

template <typename dst_t>
__dpct_inline__ static void
dequantize_block_iq1_s(const void *__restrict__ vx, dst_t *__restrict__ yy,
                       const sycl::nd_item<3> &item_ct1,
                       const uint32_t *iq1s_grid_gpu) {

    const int64_t i = item_ct1.get_group(2);
    const block_iq1_s * x = (const block_iq1_s  *) vx;

    const int64_t tid = item_ct1.get_local_id(2);
#if QK_K == 256
    const int64_t il = tid/8; // 0...3
    const int64_t ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 8*il;
    const float delta = x[i].qh[ib] & 0x8000 ? -1 - IQ1S_DELTA : -1 + IQ1S_DELTA;
    const float d = (float)x[i].d * (2*((x[i].qh[ib] >> 12) & 7) + 1);
    uint32_t grid32[2]; const int8_t * q = (const int8_t *)grid32;
    grid32[0] = iq1s_grid_gpu[x[i].qs[4*ib+il] | (((x[i].qh[ib] >> 3*il) & 7) << 8)];
    grid32[1] = (grid32[0] >> 4) & 0x0f0f0f0f;
    grid32[0] &= 0x0f0f0f0f;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        y[j] = d * (q[j] + delta);
    }
#else
    assert(false);
#endif

}

template <typename dst_t>
__dpct_inline__ static void
dequantize_block_iq1_m(const void *__restrict__ vx, dst_t *__restrict__ yy,
                       const sycl::nd_item<3> &item_ct1,
                       const uint32_t *iq1s_grid_gpu) {

    const int64_t i = item_ct1.get_group(2);
    const block_iq1_m * x = (const block_iq1_m  *) vx;

    const int64_t tid = item_ct1.get_local_id(2);
#if QK_K == 256
    const int64_t il = tid/8; // 0...3
    const int64_t ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 8*il;
    const uint16_t * sc = (const uint16_t *)x[i].scales;
    iq1m_scale_t scale;
    scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
    const int ib16 = 2*ib + il/2; // sc[ib16/4] >> 3*(ib16%4) -> sc[ib/2] >> 3*((2*ib+il/2)%4);
    const float d = (float)scale.f16 * (2*((sc[ib16/4] >> 3*(ib16%4)) & 0x7) + 1);
    const float delta = x[i].qh[2*ib+il/2] & (0x08 << 4*(il%2)) ? -1 - IQ1M_DELTA : -1 + IQ1M_DELTA;
    uint32_t grid32[2]; const int8_t * q = (const int8_t *)grid32;
    grid32[0] = iq1s_grid_gpu[x[i].qs[4*ib+il] | (((x[i].qh[2*ib+il/2] >> 4*(il%2)) & 7) << 8)];
    grid32[1] = (grid32[0] >> 4) & 0x0f0f0f0f;
    grid32[0] &= 0x0f0f0f0f;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        y[j] = d * (q[j] + delta);
    }
#else
    assert(false);
#endif

}

template <typename dst_t>
__dpct_inline__ static void
dequantize_block_iq4_nl(const void *__restrict__ vx, dst_t *__restrict__ yy,
                        const sycl::nd_item<3> &item_ct1) {

    const int64_t i = item_ct1.get_group(2);
    const block_iq4_nl * x = (const block_iq4_nl *) vx + i*(QK_K/QK4_NL);

    const int64_t tid = item_ct1.get_local_id(2);
    const int64_t il = tid/8; // 0...3
    const int64_t ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 4*il;
    const uint8_t  * q4 = x[ib].qs + 4*il;
    const float d = (float)x[ib].d;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        y[j+ 0] = d * kvalues_iq4nl[q4[j] & 0xf];
        y[j+16] = d * kvalues_iq4nl[q4[j] >>  4];
    }

}


template <typename dst_t>
__dpct_inline__ static void
dequantize_block_iq4_xs(const void *__restrict__ vx, dst_t *__restrict__ yy,
                        const sycl::nd_item<3> &item_ct1) {
    const int64_t i = item_ct1.get_group(2);
    const block_iq4_xs * x = (const block_iq4_xs *)vx;

    const int64_t tid = item_ct1.get_local_id(2);
    const int64_t il = tid/8; // 0...3
    const int64_t ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 4*il;
    const uint8_t  * q4 = x[i].qs + 16*ib + 4*il;
    const float d = (float)x[i].d * ((((x[i].scales_l[ib/2] >> 4*(ib%2)) & 0xf) | (((x[i].scales_h >> 2*ib) & 3) << 4)) - 32);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        y[j+ 0] = d * kvalues_iq4nl[q4[j] & 0xf];
        y[j+16] = d * kvalues_iq4nl[q4[j] >>  4];
    }
}


#endif // GGML_SYCL_DEQUANTIZE_HPP
