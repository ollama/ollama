//
// MIT license
// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: MIT
//

//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#ifndef GGML_SYCL_VECDOTQ_HPP
#define GGML_SYCL_VECDOTQ_HPP

#include "dpct/helper.hpp"
#include "ggml.h"
#include "quants.hpp"

typedef float (*vec_dot_q_sycl_t)(const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1,
                                  const int & iqs);

static __dpct_inline__ int get_int_from_int8(const int8_t* x8, const int& i32) {
  const uint16_t* x16 =
      (const uint16_t*)(x8 + sizeof(int) * i32); // assume at least 2 byte
                                                 // alignment

  int x32 = 0;
  x32 |= x16[0] << 0;
  x32 |= x16[1] << 16;

  return x32;
}

static __dpct_inline__ int get_int_from_uint8(
    const uint8_t* x8,
    const int& i32) {
  const uint16_t* x16 =
      (const uint16_t*)(x8 + sizeof(int) * i32); // assume at least 2 byte
                                                 // alignment

  int x32 = 0;
  x32 |= x16[0] << 0;
  x32 |= x16[1] << 16;

  return x32;
}

static __dpct_inline__ int get_int_from_int8_aligned(
    const int8_t* x8,
    const int& i32) {
  return *(
      (const int*)(x8 + sizeof(int) * i32)); // assume at least 4 byte alignment
}

static __dpct_inline__ int get_int_from_uint8_aligned(
    const uint8_t* x8,
    const int& i32) {
  return *(
      (const int*)(x8 + sizeof(int) * i32)); // assume at least 4 byte alignment
}

static __dpct_inline__ void get_int_from_table_16(const uint32_t &q4,
                                                  const uint8_t *values,
                                                  int &val1, int &val2) {

    uint32_t aux32; const uint8_t * q8 = (const uint8_t *)&aux32;
    aux32 = q4 & 0x0f0f0f0f;
    uint16_t v1 = values[q8[0]] | (values[q8[1]] << 8);
    uint16_t v2 = values[q8[2]] | (values[q8[3]] << 8);
    val1 = v1 | (v2 << 16);
    aux32 = (q4 >> 4) & 0x0f0f0f0f;
    v1 = values[q8[0]] | (values[q8[1]] << 8);
    v2 = values[q8[2]] | (values[q8[3]] << 8);
    val2 = v1 | (v2 << 16);
}

#define VDR_Q2_K_Q8_1_MMVQ 1

// contiguous v/x values
static __dpct_inline__ float vec_dot_q2_K_q8_1_impl_mmvq(
    const int &v, const int *__restrict__ u, const uint8_t *__restrict__ scales,
    const sycl::half2 &dm2, const float *__restrict__ d8) {

    float sumf_d = 0.0f;
    float sumf_m = 0.0f;

#pragma unroll
    for (int i = 0; i < QR2_K; ++i) {
        const int sc = scales[2*i];

        const int vi = (v >> (2*i)) & 0x03030303;

        sumf_d +=
            d8[i] * (dpct::dp4a(vi, u[i], 0) * (sc & 0xF)); // SIMD dot product

        // fill int with 4x m
        int m = sc >> 4;
        m |= m <<  8;
        m |= m << 16;
        sumf_m += d8[i] *
                  dpct::dp4a(
                      m, u[i],
                      0); // multiply constant q2_K part with sum of q8_1 values
    }

    const sycl::float2 dm2f =
        dm2.convert<float, sycl::rounding_mode::automatic>();

    return dm2f.x() * sumf_d - dm2f.y() * sumf_m;
}


#define VDR_Q3_K_Q8_1_MMVQ 1

// contiguous v/x values
static __dpct_inline__ float vec_dot_q3_K_q8_1_impl_mmvq(
    const int &vl, const int &vh, const int *__restrict__ u,
    const uint8_t *__restrict__ scales, const int &scale_offset,
    const float &d3, const float *__restrict__ d8) {

    float sumf = 0.0f;

#pragma unroll
    for (int i = 0; i < QR3_K; ++i) {
        const int isc = scale_offset + 2*i;

        const int isc_low = isc % (QK_K/32);
        const int sc_shift_low = 4 * (isc / (QK_K/32));
        const int sc_low  = (scales[isc_low] >> sc_shift_low) & 0xF;

        const int isc_high = isc % (QK_K/64);
        const int sc_shift_high = 2 * (isc / (QK_K/64));
        const int sc_high = ((scales[(QK_K/32) + isc_high] >> sc_shift_high) & 3) << 4;

        const int sc = (sc_low | sc_high) - 32;

        const int vil = (vl >> (2*i)) & 0x03030303;

        const int vih = ((vh >> i) << 2) & 0x04040404;

        const int vi =
            dpct::vectorized_binary<sycl::char4>(vil, vih, dpct::sub_sat());

        sumf += d8[i] * (dpct::dp4a(vi, u[i], 0) * sc); // SIMD dot product
    }

    return d3 * sumf;
}

#define VDR_Q4_K_Q8_1_MMVQ 2

// contiguous v/x values
static __dpct_inline__ float vec_dot_q4_K_q8_1_impl_vmmq(
    const int *__restrict__ v, const int *__restrict__ u,
    const uint8_t *__restrict__ sc, const uint8_t *__restrict__ m,
    const sycl::half2 &dm4, const float *__restrict__ d8) {

    float sumf_d = 0.0f;
    float sumf_m = 0.0f;

#pragma unroll
    for (int i = 0; i < QR4_K; ++i) {
        const int v0i = (v[0] >> (4*i)) & 0x0F0F0F0F;
        const int v1i = (v[1] >> (4*i)) & 0x0F0F0F0F;

        const int dot1 =
            dpct::dp4a(v1i, u[2 * i + 1],
                       dpct::dp4a(v0i, u[2 * i + 0], 0)); // SIMD dot product
        const int dot2 =
            dpct::dp4a(0x01010101, u[2 * i + 1],
                       dpct::dp4a(0x01010101, u[2 * i + 0], 0)); // sum of u

        sumf_d += d8[i] * (dot1 * sc[i]);
        sumf_m += d8[i] * (dot2 * m[i]);  // multiply constant part of q4_K with sum of q8_1 values
    }

    const sycl::float2 dm4f =
        dm4.convert<float, sycl::rounding_mode::automatic>();

    return dm4f.x() * sumf_d - dm4f.y() * sumf_m;
}


#define VDR_Q5_K_Q8_1_MMVQ 2

// contiguous v/x values
static __dpct_inline__ float vec_dot_q5_K_q8_1_impl_vmmq(
    const int *__restrict__ vl, const int *__restrict__ vh,
    const int *__restrict__ u, const uint8_t *__restrict__ sc,
    const uint8_t *__restrict__ m, const sycl::half2 &dm5,
    const float *__restrict__ d8) {

    float sumf_d = 0.0f;
    float sumf_m = 0.0f;

#pragma unroll
    for (int i = 0; i < QR5_K; ++i) {
        const int vl0i = (vl[0] >> (4*i)) & 0x0F0F0F0F;
        const int vl1i = (vl[1] >> (4*i)) & 0x0F0F0F0F;

        const int vh0i = ((vh[0] >> i) << 4) & 0x10101010;
        const int vh1i = ((vh[1] >> i) << 4) & 0x10101010;

        const int v0i = vl0i | vh0i;
        const int v1i = vl1i | vh1i;

        const int dot1 =
            dpct::dp4a(v0i, u[2 * i + 0],
                       dpct::dp4a(v1i, u[2 * i + 1], 0)); // SIMD dot product
        const int dot2 =
            dpct::dp4a(0x01010101, u[2 * i + 0],
                       dpct::dp4a(0x01010101, u[2 * i + 1], 0)); // sum of u

        sumf_d += d8[i] * (dot1 * sc[i]);
        sumf_m += d8[i] * (dot2 * m[i]);

    }

    const sycl::float2 dm5f =
        dm5.convert<float, sycl::rounding_mode::automatic>();

    return dm5f.x() * sumf_d - dm5f.y() * sumf_m;
}


#define VDR_Q6_K_Q8_1_MMVQ 1

// contiguous v/x values
static __dpct_inline__ float
vec_dot_q6_K_q8_1_impl_mmvq(const int &vl, const int &vh,
                            const int *__restrict__ u,
                            const int8_t *__restrict__ scales, const float &d,
                            const float *__restrict__ d8) {

    float sumf = 0.0f;

#pragma unroll
    for (int i = 0; i < QR6_K; ++i) {
        const int sc = scales[4*i];

        const int vil = (vl >> (4*i)) & 0x0F0F0F0F;

        const int vih = ((vh >> (4*i)) << 4) & 0x30303030;

        const int vi = dpct::vectorized_binary<sycl::char4>(
            (vil | vih), 0x20202020, dpct::sub_sat()); // vi = (vil | vih) - 32

        sumf += d8[i] * (dpct::dp4a(vi, u[i], 0) * sc); // SIMD dot product
    }

    return d*sumf;
}

// VDR = vec dot ratio, how many contiguous integers each thread processes when the vec dot kernel is called
// MMVQ = mul_mat_vec_q, MMQ = mul_mat_q

template <ggml_type T> struct reorder_vec_dot_q_sycl {
    static_assert(T != T, "ggml_type for reorder vecdot not implemented");
};

template <> struct reorder_vec_dot_q_sycl<GGML_TYPE_Q4_0> {
    static constexpr ggml_type gtype = GGML_TYPE_Q4_0;

    using q4_0_block  = ggml_sycl_reordered::block_q_t<GGML_TYPE_Q4_0>;
    using q4_0_traits = typename q4_0_block::traits;

    __dpct_inline__ float vec_dot_q4_0_q8_1_impl(const int * v, const int * u, const float & d4, const sycl::half2 & ds8) {
        int sumi = 0;

#pragma unroll
        for (size_t i = 0; i < q4_0_traits::vdr_mmvq; ++i) {
            const int vi0 = (v[i] >> 0) & 0x0F0F0F0F;
            const int vi1 = (v[i] >> 4) & 0x0F0F0F0F;

            // SIMD dot product of quantized values
            sumi = dpct::dp4a(vi0, u[2 * i + 0], sumi);
            sumi = dpct::dp4a(vi1, u[2 * i + 1], sumi);
        }

        const sycl::float2 ds8f = ds8.convert<float, sycl::rounding_mode::automatic>();

        // second part effectively subtracts 8 from each quant value
        return d4 * (sumi * ds8f.x() - (8 * q4_0_traits::vdr_mmvq / q4_0_traits::qi) * ds8f.y());
    }

    __dpct_inline__ float operator()(const void * __restrict__ vbq, const int ibx_offset, const int d_offset,
                     const block_q8_1 * __restrict__ bq8_1, const int & iqs, int /* nblocks */) {
        const uint8_t * bq4_0 = static_cast<const uint8_t *>(vbq) + ibx_offset;
        const ggml_half d     = *(reinterpret_cast<const ggml_half *>(static_cast<const uint8_t *>(vbq) + d_offset));
        int             v[q4_0_traits::vdr_mmvq];
        int             u[2 * q4_0_traits::vdr_mmvq];

#pragma unroll

        for (size_t i = 0; i < q4_0_traits::vdr_mmvq; ++i) {
            v[i]         = get_int_from_uint8(bq4_0, iqs + i);
            u[2 * i + 0] = get_int_from_int8_aligned(bq8_1->qs, iqs + i);
            u[2 * i + 1] = get_int_from_int8_aligned(bq8_1->qs, iqs + i + q4_0_traits::qi);
        }

        return vec_dot_q4_0_q8_1_impl(v, u, d, bq8_1->ds);
    };
};

static inline float vec_dot_q4_K_q8_1_common(const int * __restrict__ q4, const uint16_t * __restrict__ scales,
                                             const ggml_half2 & dm, const block_q8_1 * __restrict__ bq8_1,
                                             const int &        iqs) {
    int   v[2];
    int   u[2 * QR4_K];
    float d8[QR4_K];

    v[0] = q4[0];
    v[1] = q4[4];

    uint16_t  aux[2];
    const int j = (QR4_K * ((iqs / 2) / (QI8_1 / 2))) / 2;
    if (j < 2) {
        aux[0] = scales[j + 0] & 0x3f3f;
        aux[1] = scales[j + 2] & 0x3f3f;
    } else {
        aux[0] = ((scales[j + 2] >> 0) & 0x0f0f) | ((scales[j - 2] & 0xc0c0) >> 2);
        aux[1] = ((scales[j + 2] >> 4) & 0x0f0f) | ((scales[j - 0] & 0xc0c0) >> 2);
    }

    const uint8_t * sc = (const uint8_t *) aux;
    const uint8_t * m  = sc + 2;

    const int bq8_offset = QR4_K * ((iqs / 2) / (QI8_1 / 2));

    for (int i = 0; i < QR4_K; ++i) {
        const block_q8_1 * bq8i = bq8_1 + bq8_offset + i;
        d8[i]                   = bq8i->ds[0];

        const int * q8 = (const int *) bq8i->qs + ((iqs / 2) % 4);
        u[2 * i + 0]   = q8[0];
        u[2 * i + 1]   = q8[4];
    }

    return vec_dot_q4_K_q8_1_impl_vmmq(v, u, sc, m, dm, d8);
}

template <> struct reorder_vec_dot_q_sycl<GGML_TYPE_Q4_K> {
    static constexpr ggml_type gtype = GGML_TYPE_Q4_K;

    using q4_k_block  = ggml_sycl_reordered::block_q_t<GGML_TYPE_Q4_K>;
    using q4_k_traits = typename q4_k_block::traits;

    float operator()(const void * __restrict__ vbq, const int ibx_offset, const int d_offset,
                     const block_q8_1 * __restrict__ bq8_1, const int & iqs, int nblocks) {
        const int ib = ibx_offset / (QK_K / 2);

        const uint8_t *    base           = static_cast<const uint8_t *>(vbq);
        const uint8_t *    qs             = base + ibx_offset;
        const int          total_qs_bytes = nblocks * (QK_K / 2);
        const uint8_t *    scs            = base + total_qs_bytes + ib * K_SCALE_SIZE;
        const ggml_half2 * dms            = reinterpret_cast<const ggml_half2 *>(base + d_offset);

        const int        bq8_offset = QR4_K * ((iqs / 2) / (QI8_1 / 2));
        const int *      q4         = (const int *) (qs + 16 * bq8_offset + 4 * ((iqs / 2) % 4));
        const uint16_t * scales     = (const uint16_t *) scs;

        return vec_dot_q4_K_q8_1_common(q4, scales, *dms, bq8_1, iqs);
    }
};

#define VDR_Q4_0_Q8_1_MMVQ 2
#define VDR_Q4_0_Q8_1_MMQ  4

template <int vdr>
static __dpct_inline__ float vec_dot_q4_0_q8_1_impl(const int * v, const int * u, const float & d4,
                                                    const sycl::half2 & ds8) {
    int sumi = 0;
#pragma unroll
    for (int i = 0; i < vdr; ++i) {
        const int vi0 = (v[i] >> 0) & 0x0F0F0F0F;
        const int vi1 = (v[i] >> 4) & 0x0F0F0F0F;

        // SIMD dot product of quantized values
        sumi = dpct::dp4a(vi0, u[2 * i + 0], sumi);
        sumi = dpct::dp4a(vi1, u[2 * i + 1], sumi);
    }

    const sycl::float2 ds8f = ds8.convert<float, sycl::rounding_mode::automatic>();

    // second part effectively subtracts 8 from each quant value
    return d4 * (sumi * ds8f.x() - (8 * vdr / QI4_0) * ds8f.y());
}

#define VDR_Q4_1_Q8_1_MMVQ 2
#define VDR_Q4_1_Q8_1_MMQ  4

template <int vdr>
static __dpct_inline__ float vec_dot_q4_1_q8_1_impl(const int *v, const int *u,
                                                    const sycl::half2 &dm4,
                                                    const sycl::half2 &ds8) {

    int sumi = 0;

#pragma unroll
    for (int i = 0; i < vdr; ++i) {
        const int vi0 = (v[i] >> 0) & 0x0F0F0F0F;
        const int vi1 = (v[i] >> 4) & 0x0F0F0F0F;

        // SIMD dot product of quantized values
        sumi = dpct::dp4a(vi0, u[2 * i + 0], sumi);
        sumi = dpct::dp4a(vi1, u[2 * i + 1], sumi);
    }

#ifdef GGML_SYCL_F16
    const sycl::float2 tmp =
        (dm4 * ds8).convert<float, sycl::rounding_mode::automatic>();
    const float d4d8 = tmp.x();
    const float m4s8 = tmp.y();
#else
    const sycl::float2 dm4f =
        dm4.convert<float, sycl::rounding_mode::automatic>();
    const sycl::float2 ds8f =
        ds8.convert<float, sycl::rounding_mode::automatic>();
    const float d4d8 = dm4f.x() * ds8f.x();
    const float m4s8 = dm4f.y() * ds8f.y();
#endif // GGML_SYCL_F16

    // scale second part of sum by QI8_1/(vdr * QR4_1) to compensate for multiple threads adding it
    return sumi * d4d8 + m4s8 / (QI8_1 / (vdr * QR4_1));
}

#define VDR_Q5_0_Q8_1_MMVQ 2
#define VDR_Q5_0_Q8_1_MMQ  4

template <int vdr>
static __dpct_inline__ float
vec_dot_q5_0_q8_1_impl(const int *vl, const int *vh, const int *u,
                       const float &d5, const sycl::half2 &ds8) {
    int sumi = 0;

#pragma unroll
    for (int i = 0; i < vdr; ++i) {
        int vi0 = (vl[i] >>  0) & 0x0F0F0F0F; // lower 4 qs bits, still need qh as 5th bits
        vi0    |= (vh[i] <<  4) & 0x00000010; // 0 ->  4
        vi0    |= (vh[i] << 11) & 0x00001000; // 1 -> 12
        vi0    |= (vh[i] << 18) & 0x00100000; // 2 -> 20
        vi0    |= (vh[i] << 25) & 0x10000000; // 3 -> 28
        sumi = dpct::dp4a(vi0, u[2 * i + 0],
                          sumi); // SIMD dot product of quantized values

        int vi1 = (vl[i] >>  4) & 0x0F0F0F0F; // upper 4 qs bits, still need qh as 5th bits
        vi1    |= (vh[i] >> 12) & 0x00000010; // 16 ->  4
        vi1    |= (vh[i] >>  5) & 0x00001000; // 17 -> 12
        vi1    |= (vh[i] <<  2) & 0x00100000; // 18 -> 20
        vi1    |= (vh[i] <<  9) & 0x10000000; // 19 -> 28
        sumi = dpct::dp4a(vi1, u[2 * i + 1],
                          sumi); // SIMD dot product of quantized values
    }

    const sycl::float2 ds8f =
        ds8.convert<float, sycl::rounding_mode::automatic>();

    // second part effectively subtracts 16 from each quant value
    return d5 * (sumi * ds8f.x() - (16 * vdr / QI5_0) * ds8f.y());
}

#define VDR_Q5_1_Q8_1_MMVQ 2
#define VDR_Q5_1_Q8_1_MMQ  4

template <int vdr>
static __dpct_inline__ float
vec_dot_q5_1_q8_1_impl(const int *vl, const int *vh, const int *u,
                       const sycl::half2 &dm5, const sycl::half2 &ds8) {

    int sumi = 0;

#pragma unroll
    for (int i = 0; i < vdr; ++i) {
        int vi0 = (vl[i] >>  0) & 0x0F0F0F0F; // lower 4 qs bits, still need qh as 5th bits
        vi0    |= (vh[i] <<  4) & 0x00000010; // 0 ->  4
        vi0    |= (vh[i] << 11) & 0x00001000; // 1 -> 12
        vi0    |= (vh[i] << 18) & 0x00100000; // 2 -> 20
        vi0    |= (vh[i] << 25) & 0x10000000; // 3 -> 28
        sumi = dpct::dp4a(vi0, u[2 * i + 0],
                          sumi); // SIMD dot product of quantized values

        int vi1 = (vl[i] >>  4) & 0x0F0F0F0F; // upper 4 qs bits, still need qh as 5th bits
        vi1    |= (vh[i] >> 12) & 0x00000010; // 16 ->  4
        vi1    |= (vh[i] >>  5) & 0x00001000; // 17 -> 12
        vi1    |= (vh[i] <<  2) & 0x00100000; // 18 -> 20
        vi1    |= (vh[i] <<  9) & 0x10000000; // 19 -> 28
        sumi = dpct::dp4a(vi1, u[2 * i + 1],
                          sumi); // SIMD dot product of quantized values
    }

#ifdef GGML_SYCL_F16
     const sycl::float2 tmp =
        (dm5 * ds8).convert<float, sycl::rounding_mode::automatic>();
    const float d5d8 = tmp.x();
    const float m5s8 = tmp.y();


#else
    const sycl::float2 dm5f =
        dm5.convert<float, sycl::rounding_mode::automatic>();
    const sycl::float2 ds8f =
        ds8.convert<float, sycl::rounding_mode::automatic>();
    const float d5d8 = dm5f.x() * ds8f.x();
    const float m5s8 = dm5f.y() * ds8f.y();
#endif // GGML_SYCL_F16

    // scale second part of sum by QI5_1 / vdr to compensate for multiple threads adding it
    return sumi*d5d8 + m5s8 / (QI5_1 / vdr);
}

#define VDR_Q8_0_Q8_1_MMVQ 2
#define VDR_Q8_0_Q8_1_MMQ 8

template <int vdr>
static __dpct_inline__ float vec_dot_q8_0_q8_1_impl(const int *v, const int *u,
                                                    const float &d8_0,
                                                    const float &d8_1) {

    int sumi = 0;

#pragma unroll
    for (int i = 0; i < vdr; ++i) {
        // SIMD dot product of quantized values
        sumi = dpct::dp4a(v[i], u[i], sumi);
    }

    return d8_0*d8_1 * sumi;
}

template <int vdr>
static __dpct_inline__ float vec_dot_q8_1_q8_1_impl(const int *v, const int *u,
                                                    const sycl::half2 &dm8,
                                                    const sycl::half2 &ds8) {

    int sumi = 0;

#pragma unroll
    for (int i = 0; i < vdr; ++i) {
        // SIMD dot product of quantized values
        sumi = dpct::dp4a(v[i], u[i], sumi);
    }

#ifdef GGML_SYCL_F16
    const sycl::float2 tmp =
        (dm8 * ds8).convert<float, sycl::rounding_mode::automatic>();
    const float d8d8 = tmp.x();
    const float m8s8 = tmp.y();
#else
    const sycl::float2 dm8f =
        dm8.convert<float, sycl::rounding_mode::automatic>();
    const sycl::float2 ds8f =
        ds8.convert<float, sycl::rounding_mode::automatic>();
    const float d8d8 = dm8f.x() * ds8f.x();
    const float m8s8 = dm8f.y() * ds8f.y();
#endif // GGML_SYCL_F16

    // scale second part of sum by QI8_1/ vdr to compensate for multiple threads adding it
    return sumi*d8d8 + m8s8 / (QI8_1 / vdr);
}

static __dpct_inline__ float
vec_dot_q4_0_q8_1(const void *__restrict__ vbq,
                  const block_q8_1 *__restrict__ bq8_1, const int &iqs) {

    const block_q4_0 * bq4_0 = (const block_q4_0 *) vbq;

    int v[VDR_Q4_0_Q8_1_MMVQ];
    int u[2 * VDR_Q4_0_Q8_1_MMVQ];

#pragma unroll
    for (int i = 0; i < VDR_Q4_0_Q8_1_MMVQ; ++i) {
        v[i]         = get_int_from_uint8(bq4_0->qs, iqs + i);
        u[2 * i + 0] = get_int_from_int8_aligned(bq8_1->qs, iqs + i);
        u[2 * i + 1] = get_int_from_int8_aligned(bq8_1->qs, iqs + i + QI4_0);
    }

    return vec_dot_q4_0_q8_1_impl<VDR_Q4_0_Q8_1_MMVQ>(v, u, bq4_0->d, bq8_1->ds);
}

static __dpct_inline__ float
vec_dot_q4_1_q8_1(const void *__restrict__ vbq,
                  const block_q8_1 *__restrict__ bq8_1, const int &iqs) {

    const block_q4_1 * bq4_1 = (const block_q4_1 *) vbq;

    int v[VDR_Q4_1_Q8_1_MMVQ];
    int u[2*VDR_Q4_1_Q8_1_MMVQ];

#pragma unroll
    for (int i = 0; i < VDR_Q4_1_Q8_1_MMVQ; ++i) {
        v[i]    = get_int_from_uint8_aligned(bq4_1->qs, iqs + i);
        u[2*i+0] = get_int_from_int8_aligned(bq8_1->qs, iqs + i);
        u[2*i+1] = get_int_from_int8_aligned(bq8_1->qs, iqs + i + QI4_1);
    }

    return vec_dot_q4_1_q8_1_impl<VDR_Q4_1_Q8_1_MMVQ>(v, u, bq4_1->dm, bq8_1->ds);
}

static __dpct_inline__ float
vec_dot_q5_0_q8_1(const void *__restrict__ vbq,
                  const block_q8_1 *__restrict__ bq8_1, const int &iqs) {

    const block_q5_0 * bq5_0 = (const block_q5_0 *) vbq;

    int vl[VDR_Q5_0_Q8_1_MMVQ];
    int vh[VDR_Q5_0_Q8_1_MMVQ];
    int  u[2*VDR_Q5_0_Q8_1_MMVQ];

#pragma unroll
    for (int i = 0; i < VDR_Q5_0_Q8_1_MMVQ; ++i) {
        vl[i]    = get_int_from_uint8(bq5_0->qs, iqs + i);
        vh[i]    = get_int_from_uint8(bq5_0->qh, 0) >> (4 * (iqs + i));
        u[2*i+0] = get_int_from_int8_aligned(bq8_1->qs, iqs + i);
        u[2*i+1] = get_int_from_int8_aligned(bq8_1->qs, iqs + i + QI5_0);
    }

    return vec_dot_q5_0_q8_1_impl<VDR_Q5_0_Q8_1_MMVQ>(vl, vh, u, bq5_0->d, bq8_1->ds);
}

static __dpct_inline__ float
vec_dot_q5_1_q8_1(const void *__restrict__ vbq,
                  const block_q8_1 *__restrict__ bq8_1, const int &iqs) {

    const block_q5_1 * bq5_1 = (const block_q5_1 *) vbq;

    int vl[VDR_Q5_1_Q8_1_MMVQ];
    int vh[VDR_Q5_1_Q8_1_MMVQ];
    int  u[2*VDR_Q5_1_Q8_1_MMVQ];

#pragma unroll
    for (int i = 0; i < VDR_Q5_1_Q8_1_MMVQ; ++i) {
        vl[i]   = get_int_from_uint8_aligned(bq5_1->qs, iqs + i);
        vh[i]   = get_int_from_uint8_aligned(bq5_1->qh, 0) >> (4 * (iqs + i));
        u[2*i+0] = get_int_from_int8_aligned(bq8_1->qs, iqs + i);
        u[2*i+1] = get_int_from_int8_aligned(bq8_1->qs, iqs + i + QI5_1);
    }

    return vec_dot_q5_1_q8_1_impl<VDR_Q5_1_Q8_1_MMVQ>(vl, vh, u, bq5_1->dm, bq8_1->ds);
}

static __dpct_inline__ float
vec_dot_q8_0_q8_1(const void *__restrict__ vbq,
                  const block_q8_1 *__restrict__ bq8_1, const int &iqs) {

    const block_q8_0 * bq8_0 = (const block_q8_0 *) vbq;

    int v[VDR_Q8_0_Q8_1_MMVQ];
    int u[VDR_Q8_0_Q8_1_MMVQ];

#pragma unroll
    for (int i = 0; i < VDR_Q8_0_Q8_1_MMVQ; ++i) {
        v[i] = get_int_from_int8(bq8_0->qs, iqs + i);
        u[i] = get_int_from_int8_aligned(bq8_1->qs, iqs + i);
    }

    return vec_dot_q8_0_q8_1_impl<VDR_Q8_0_Q8_1_MMVQ>(v, u, bq8_0->d,
                                                      bq8_1->ds[0]);
}

static __dpct_inline__ float
vec_dot_q2_K_q8_1(const void *__restrict__ vbq,
                  const block_q8_1 *__restrict__ bq8_1, const int &iqs) {

    const block_q2_K * bq2_K = (const block_q2_K *) vbq;

    const int bq8_offset = QR2_K * (iqs / QI8_1);
    const int scale_offset = iqs - iqs % QI8_1 + (iqs % QI8_1) / (QI8_1/2);

    const uint8_t * scales = bq2_K->scales + scale_offset;

    const int v = get_int_from_uint8_aligned(bq2_K->qs, iqs);
    int    u[QR2_K];
    float d8[QR2_K];

#pragma unroll
    for (int i = 0; i < QR2_K; ++ i) {
        u[i]  = get_int_from_int8_aligned(bq8_1[bq8_offset + i].qs, iqs % QI8_1);
        d8[i] = bq8_1[bq8_offset + i].ds[0];
    }

    return vec_dot_q2_K_q8_1_impl_mmvq(v, u, scales, bq2_K->dm, d8);
}

static __dpct_inline__ float
vec_dot_q3_K_q8_1(const void *__restrict__ vbq,
                  const block_q8_1 *__restrict__ bq8_1, const int &iqs) {

    const block_q3_K * bq3_K = (const block_q3_K *) vbq;

    const int bq8_offset = QR3_K * (iqs / (QI3_K/2));
    const int scale_offset = iqs - iqs % QI8_1 + (iqs % QI8_1) / (QI8_1/2);

    const float d = bq3_K->d;

    const int vl = get_int_from_uint8(bq3_K->qs, iqs);

    // invert the mask with ~ so that a 0/1 results in 4/0 being subtracted
    const int vh = ~get_int_from_uint8(bq3_K->hmask, iqs % (QI3_K/2)) >> bq8_offset;

    int    u[QR3_K];
    float d8[QR3_K];

#pragma unroll
    for (int i = 0; i < QR3_K; ++i) {
        u[i]  = get_int_from_int8_aligned(bq8_1[bq8_offset + i].qs, iqs % QI8_1);
        d8[i] = bq8_1[bq8_offset + i].ds[0];
    }

    return vec_dot_q3_K_q8_1_impl_mmvq(vl, vh, u, bq3_K->scales, scale_offset, d, d8);
}

static __dpct_inline__ float vec_dot_q4_K_q8_1(const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1,
                                               const int & iqs) {
#ifndef GGML_QKK_64

    const block_q4_K * bq4_K = (const block_q4_K *) vbq;

    const int        bq8_offset = QR4_K * ((iqs / 2) / (QI8_1 / 2));
    const int *      q4         = (const int *) (bq4_K->qs + 16 * bq8_offset + 4 * ((iqs / 2) % 4));
    const uint16_t * scales     = (const uint16_t *) bq4_K->scales;

    return vec_dot_q4_K_q8_1_common(q4, scales, bq4_K->dm, bq8_1, iqs);

#else

#if __SYCL_ARCH__ >= VER_4VEC // lowest compute capability for integer intrinsics
    const block_q4_K * bq4_K = (const block_q4_K *) vbq;

    float sumf_d = 0.0f;
    float sumf_m = 0.0f;

    uint16_t aux16[2];
    const uint8_t * s = (const uint8_t *)aux16;

    const uint16_t * a = (const uint16_t *)bq4_K->scales;
    aux16[0] = a[0] & 0x0f0f;
    aux16[1] = (a[0] >> 4) & 0x0f0f;

    const float dall = bq4_K->dm[0];
    const float dmin = bq4_K->dm[1];

    const float d8_1 = bq8_1[0].ds[0];
    const float d8_2 = bq8_1[1].ds[1];

    const int ui1 = *((const int *)bq8_1[0].qs + (iqs/2));
    const int ui2 = *((const int *)bq8_1[0].qs + (iqs/2) + 4);
    const int ui3 = *((const int *)bq8_1[1].qs + (iqs/2));
    const int ui4 = *((const int *)bq8_1[1].qs + (iqs/2) + 4);

    const int * q4 = (const int *)bq4_K->qs + (iqs/2);
    const int v1 = q4[0];
    const int v2 = q4[4];

    const int dot1 = dpct::dp4a(ui2, v2 & 0x0f0f0f0f, dpct::dp4a(ui1, v1 & 0x0f0f0f0f, 0));
    const int dot2 = dpct::dp4a(ui4, (v2 >> 4) & 0x0f0f0f0f, dpct::dp4a(ui3, (v1 >> 4) & 0x0f0f0f0f, 0));
    const int dot3 = dpct::dp4a(0x01010101, ui2, dpct::dp4a(0x01010101, ui1, 0));
    const int dot4 = dpct::dp4a(0x01010101, ui4, dpct::dp4a(0x01010101, ui3, 0));

    sumf_d += d8_1 * (dot1 * s[0]) + d8_2 * (dot2 * s[1]);
    sumf_m += d8_1 * (dot3 * s[2]) + d8_2 * (dot4 * s[3]);

    return dall * sumf_d - dmin * sumf_m;

#else
    bad_arch();
#endif // __SYCL_ARCH__ >= VER_4VEC

#endif
}

static __dpct_inline__ float
vec_dot_q5_K_q8_1(const void *__restrict__ vbq,
                  const block_q8_1 *__restrict__ bq8_1, const int &iqs) {

#ifndef GGML_QKK_64
    const block_q5_K * bq5_K = (const block_q5_K *) vbq;

    int   vl[2];
    int   vh[2];
    int    u[2*QR5_K];
    float d8[QR5_K];

    const int bq8_offset = QR5_K * ((iqs/2) / (QI8_1/2));
    const int * ql = (const int *)(bq5_K->qs + 16 * bq8_offset + 4 * ((iqs/2)%4));
    const int * qh = (const int *)(bq5_K->qh + 4 * ((iqs/2)%4));

    vl[0] = ql[0];
    vl[1] = ql[4];

    vh[0] = qh[0] >> bq8_offset;
    vh[1] = qh[4] >> bq8_offset;

    const uint16_t * scales = (const uint16_t *)bq5_K->scales;
    uint16_t aux[2];
    const int j = bq8_offset/2;
    if (j < 2) {
        aux[0] = scales[j+0] & 0x3f3f;
        aux[1] = scales[j+2] & 0x3f3f;
    } else {
        aux[0] = ((scales[j+2] >> 0) & 0x0f0f) | ((scales[j-2] & 0xc0c0) >> 2);
        aux[1] = ((scales[j+2] >> 4) & 0x0f0f) | ((scales[j-0] & 0xc0c0) >> 2);
    }
    const uint8_t * sc = (const uint8_t *)aux;
    const uint8_t * m  = sc + 2;

#pragma unroll
    for (int i = 0; i < QR5_K; ++i) {
        const block_q8_1 * bq8i = bq8_1 + bq8_offset + i;
        d8[i] = bq8i->ds[0];

        const int * q8 = (const int *)bq8i->qs + ((iqs/2)%4);
        u[2*i+0] = q8[0];
        u[2*i+1] = q8[4];
    }

    return vec_dot_q5_K_q8_1_impl_vmmq(vl, vh, u, sc, m, bq5_K->dm, d8);

#else

#if __SYCL_ARCH__ >= VER_4VEC // lowest compute capability for integer intrinsics
    const block_q5_K * bq5_K = (const block_q5_K *) vbq;

    const int8_t * s = bq5_K->scales;

    const float d = bq5_K->d;

    const float d8_1 = bq8_1[0].ds[0];
    const float d8_2 = bq8_1[1].ds[1];

    const int ui1 = *((const int *)bq8_1[0].qs + (iqs/2));
    const int ui2 = *((const int *)bq8_1[0].qs + (iqs/2) + 4);
    const int ui3 = *((const int *)bq8_1[1].qs + (iqs/2));
    const int ui4 = *((const int *)bq8_1[1].qs + (iqs/2) + 4);

    const int * ql = (const int *)bq5_K->qs + (iqs/2);
    const int vl1 = ql[0];
    const int vl2 = ql[4];

    const int step = 4 * (iqs/2); // 0, 4, 8, 12
    const int im = step/8; // = 0 for iqs = 0, 2, = 1 for iqs = 4, 6
    const int in = step%8; // 0, 4, 0, 4
    const int vh = (*((const int *)(bq5_K->qh + in))) >> im;

    const int v1 = (((vh << 4) & 0x10101010) ^ 0x10101010) | ((vl1 >> 0) & 0x0f0f0f0f);
    const int v2 = (((vh << 2) & 0x10101010) ^ 0x10101010) | ((vl2 >> 0) & 0x0f0f0f0f);
    const int v3 = (((vh >> 0) & 0x10101010) ^ 0x10101010) | ((vl1 >> 4) & 0x0f0f0f0f);
    const int v4 = (((vh >> 2) & 0x10101010) ^ 0x10101010) | ((vl2 >> 4) & 0x0f0f0f0f);

    const float sumf_d = d8_1 * (dpct::dp4a(ui1, v1, 0) * s[0] + dpct::dp4a(ui2, v2, 0) * s[1])
                       + d8_2 * (dpct::dp4a(ui3, v3, 0) * s[2] + dpct::dp4a(ui4, v4, 0) * s[3]);

    return d * sumf_d;

#else
    bad_arch();
#endif // __SYCL_ARCH__ >= VER_4VEC

#endif
}

static __dpct_inline__ float
vec_dot_q6_K_q8_1(const void *__restrict__ vbq,
                  const block_q8_1 *__restrict__ bq8_1, const int &iqs) {

    const block_q6_K * bq6_K = (const block_q6_K *) vbq;

    const int bq8_offset = 2 * QR6_K * (iqs / (QI6_K/2)) + (iqs % (QI6_K/2)) / (QI6_K/4);
    const int scale_offset = (QI6_K/4) * (iqs / (QI6_K/2)) + (iqs % (QI6_K/2)) / (QI6_K/8);
    const int vh_shift = 2 * ((iqs % (QI6_K/2)) / (QI6_K/4));

    const int vl = get_int_from_uint8(bq6_K->ql, iqs);
    const int vh = get_int_from_uint8(bq6_K->qh, (QI6_K/4) * (iqs / (QI6_K/2)) + iqs % (QI6_K/4)) >> vh_shift;

    const int8_t * scales = bq6_K->scales + scale_offset;

    int    u[QR6_K];
    float d8[QR6_K];

#pragma unroll
    for (int i = 0; i < QR6_K; ++i) {
        u[i]  = get_int_from_int8_aligned(bq8_1[bq8_offset + 2*i].qs, iqs % QI8_1);
        d8[i] = bq8_1[bq8_offset + 2 * i].ds[0];
    }

    return vec_dot_q6_K_q8_1_impl_mmvq(vl, vh, u, scales, bq6_K->d, d8);
}


static __dpct_inline__ float
vec_dot_iq2_xxs_q8_1(const void *__restrict__ vbq,
                     const block_q8_1 *__restrict__ bq8_1, const int &iqs,
                     const uint64_t *iq2xxs_grid, const uint8_t *ksigns_iq2xs,
                     const uint8_t *kmask_iq2xs) {
#if QK_K == 256
    const block_iq2_xxs * bq2 = (const block_iq2_xxs *) vbq;

    const int ib32 = iqs;
    const uint16_t * q2 = bq2->qs + 4*ib32;
    const uint8_t  * aux8 = (const uint8_t *)q2;
    const int8_t   * q8 = bq8_1[ib32].qs;
    uint32_t aux32 = q2[2] | (q2[3] << 16);
    int sumi = 0;
    for (int l = 0; l < 4; ++l) {
        const uint8_t * grid = (const uint8_t *)(iq2xxs_grid + aux8[l]);
        const uint8_t  signs = ksigns_iq2xs[aux32 & 127];
        for (int j = 0; j < 8; ++j) {
            sumi += q8[j] * grid[j] * (signs & kmask_iq2xs[j] ? -1 : 1);
        }
        q8 += 8;
        aux32 >>= 7;
    }
    const float d = (float)bq2->d * (0.5f + aux32) * bq8_1[ib32].ds[0] * 0.25f;
    return d * sumi;
#else
    assert(false);
    return 0.f;
#endif
}

static __dpct_inline__ float
vec_dot_iq2_xs_q8_1(const void *__restrict__ vbq,
                    const block_q8_1 *__restrict__ bq8_1, const int &iqs,
                    const uint64_t *iq2xs_grid, const uint64_t *ksigns64) {
#if DPCT_COMPATIBILITY_TEMP >=                                                 \
    MIN_CC_DP4A // lowest compute capability for integer intrinsics
#if QK_K == 256
    const block_iq2_xs * bq2 = (const block_iq2_xs *) vbq;

    const int ib32 = iqs;
    const uint16_t * q2 = bq2->qs + 4*ib32;
    const int8_t   * q8 = bq8_1[ib32].qs;
    const uint8_t ls1 = bq2->scales[ib32] & 0xf;
    const uint8_t ls2 = bq2->scales[ib32] >>  4;
    int sumi1 = 0;
    for (int l = 0; l < 2; ++l) {
        const uint32_t * grid = (const uint32_t *)(iq2xs_grid + (q2[l] & 511));
        const uint32_t * signs = (const uint32_t *)(ksigns64 + (q2[l] >> 9));
        const int grid_l = dpct::vectorized_binary<sycl::uchar4>(
            grid[0] ^ signs[0], signs[0], std::minus<>());
        const int grid_h = dpct::vectorized_binary<sycl::uchar4>(
            grid[1] ^ signs[1], signs[1], std::minus<>());
        sumi1 = dpct::dp4a(grid_l, *((const int *)q8 + 0), sumi1);
        sumi1 = dpct::dp4a(grid_h, *((const int *)q8 + 1), sumi1);
        q8 += 8;
    }
    int sumi2 = 0;
    for (int l = 2; l < 4; ++l) {
        const uint32_t * grid = (const uint32_t *)(iq2xs_grid + (q2[l] & 511));
        const uint32_t * signs = (const uint32_t *)(ksigns64 + (q2[l] >> 9));
        const int grid_l = dpct::vectorized_binary<sycl::uchar4>(
            grid[0] ^ signs[0], signs[0], std::minus<>());
        const int grid_h = dpct::vectorized_binary<sycl::uchar4>(
            grid[1] ^ signs[1], signs[1], std::minus<>());
        sumi2 = dpct::dp4a(grid_l, *((const int *)q8 + 0), sumi2);
        sumi2 = dpct::dp4a(grid_h, *((const int *)q8 + 1), sumi2);
        q8 += 8;
    }
    const float d = (float)bq2->d * bq8_1[ib32].ds[0] * 0.25f;
    return d * ((0.5f + ls1) * sumi1 + (0.5f + ls2) * sumi2);
#else
    assert(false);
    return 0.f;
#endif
#else
    assert(false);
    return 0.f;
#endif
}

static __dpct_inline__ float
vec_dot_iq2_s_q8_1(const void *__restrict__ vbq,
                   const block_q8_1 *__restrict__ bq8_1, const int &iqs) {
#if QK_K == 256
    const block_iq2_s * bq2 = (const block_iq2_s *) vbq;

    const int ib32 = iqs;
    const int8_t  * q8 = bq8_1[ib32].qs;
    const uint8_t * signs = bq2->qs + QK_K/8 + 4*ib32;
    const uint8_t ls1 = bq2->scales[ib32] & 0xf;
    const uint8_t ls2 = bq2->scales[ib32] >>  4;
    int sumi1 = 0;
    for (int l = 0; l < 2; ++l) {
        const uint32_t * grid = (const uint32_t *)(iq2s_grid + (bq2->qs[4*ib32+l] | ((bq2->qh[ib32] << (8-2*l)) & 0x300)));
        const uint32_t signs0 = dpct::vectorized_binary<sycl::uchar4>(
            ((signs[l] & 0xf) * 0x01010101) & 0x08040201, 0x08040201,
            std::equal_to<>());
        const uint32_t signs1 = dpct::vectorized_binary<sycl::uchar4>(
            ((signs[l] >> 4) * 0x01010101) & 0x08040201, 0x08040201,
            std::equal_to<>());
        const int grid_l = dpct::vectorized_binary<sycl::uchar4>(
            grid[0] ^ signs0, signs0, std::minus<>());
        const int grid_h = dpct::vectorized_binary<sycl::uchar4>(
            grid[1] ^ signs1, signs1, std::minus<>());
        sumi1 = dpct::dp4a(grid_l, *((const int *)q8 + 0), sumi1);
        sumi1 = dpct::dp4a(grid_h, *((const int *)q8 + 1), sumi1);
        q8 += 8;
    }
    int sumi2 = 0;
    for (int l = 2; l < 4; ++l) {
        const uint32_t * grid = (const uint32_t *)(iq2s_grid + (bq2->qs[4*ib32+l] | ((bq2->qh[ib32] << (8-2*l)) & 0x300)));
        const uint32_t signs0 = dpct::vectorized_binary<sycl::uchar4>(
            ((signs[l] & 0xf) * 0x01010101) & 0x08040201, 0x08040201,
            std::equal_to<>());
        const uint32_t signs1 = dpct::vectorized_binary<sycl::uchar4>(
            ((signs[l] >> 4) * 0x01010101) & 0x08040201, 0x08040201,
            std::equal_to<>());
        const int grid_l = dpct::vectorized_binary<sycl::uchar4>(
            grid[0] ^ signs0, signs0, std::minus<>());
        const int grid_h = dpct::vectorized_binary<sycl::uchar4>(
            grid[1] ^ signs1, signs1, std::minus<>());
        sumi2 = dpct::dp4a(grid_l, *((const int *)q8 + 0), sumi2);
        sumi2 = dpct::dp4a(grid_h, *((const int *)q8 + 1), sumi2);
        q8 += 8;
    }
    const float d = (float)bq2->d * bq8_1[ib32].ds[0] * 0.25f;
    return d * ((0.5f + ls1) * sumi1 + (0.5f + ls2) * sumi2);
#else
    assert(false);
#endif
}

static __dpct_inline__ float
vec_dot_iq3_xxs_q8_1(const void *__restrict__ vbq,
                     const block_q8_1 *__restrict__ bq8_1, const int &iqs,
                     const uint32_t *iq3xxs_grid, const uint64_t *ksigns64) {
#if DPCT_COMPATIBILITY_TEMP >=                                                 \
    MIN_CC_DP4A // lowest compute capability for integer intrinsics
#if QK_K == 256
    const block_iq3_xxs * bq2 = (const block_iq3_xxs *) vbq;

    const int ib32 = iqs;
    const uint8_t  * q3 = bq2->qs + 8*ib32;
    const uint16_t * gas = (const uint16_t *)(bq2->qs + QK_K/4) + 2*ib32;
    const int8_t   * q8 = bq8_1[ib32].qs;
    uint32_t aux32 = gas[0] | (gas[1] << 16);
    int sumi = 0;
    for (int l = 0; l < 4; ++l) {
        const uint32_t * grid1 = iq3xxs_grid + q3[2*l+0];
        const uint32_t * grid2 = iq3xxs_grid + q3[2*l+1];
        const uint32_t * signs = (const uint32_t *)(ksigns64 + (aux32 & 127));
        const int grid_l = dpct::vectorized_binary<sycl::uchar4>(
            grid1[0] ^ signs[0], signs[0], std::minus<>());
        const int grid_h = dpct::vectorized_binary<sycl::uchar4>(
            grid2[0] ^ signs[1], signs[1], std::minus<>());
        sumi = dpct::dp4a(grid_l, *((const int *)q8 + 0), sumi);
        sumi = dpct::dp4a(grid_h, *((const int *)q8 + 1), sumi);
        q8 += 8;
        aux32 >>= 7;
    }
    const float d = (float)bq2->d * (0.5f + aux32) * bq8_1[ib32].ds[0] * 0.5f;
    return d * sumi;
#else
    assert(false);
    return 0.f;
#endif
#else
    assert(false);
    return 0.f;
#endif
}

static __dpct_inline__ float
vec_dot_iq3_s_q8_1(const void *__restrict__ vbq,
                   const block_q8_1 *__restrict__ bq8_1, const int &iqs,
                   const uint32_t *iq3s_grid) {
#if QK_K == 256
    const block_iq3_s * bq2 = (const block_iq3_s *) vbq;

    const int ib32 = iqs;
    const uint8_t  * qs = bq2->qs + 8*ib32;
    const int8_t   * q8 = bq8_1[ib32].qs;
    int sumi = 0;
    for (int l = 0; l < 4; ++l) {
        const uint32_t * grid1 = iq3s_grid + (qs[2*l+0] | ((bq2->qh[ib32] << (8 - 2*l)) & 256));
        const uint32_t * grid2 = iq3s_grid + (qs[2*l+1] | ((bq2->qh[ib32] << (7 - 2*l)) & 256));
        uint32_t signs0 = dpct::vectorized_binary<sycl::uchar4>(
            ((bq2->signs[4 * ib32 + l] & 0xf) * 0x01010101) & 0x08040201,
            0x08040201, std::equal_to<>());
        uint32_t signs1 = dpct::vectorized_binary<sycl::uchar4>(
            ((bq2->signs[4 * ib32 + l] >> 4) * 0x01010101) & 0x08040201,
            0x08040201, std::equal_to<>());
        const int grid_l = dpct::vectorized_binary<sycl::uchar4>(
            grid1[0] ^ signs0, signs0, std::minus<>());
        const int grid_h = dpct::vectorized_binary<sycl::uchar4>(
            grid2[0] ^ signs1, signs1, std::minus<>());
        sumi = dpct::dp4a(grid_l, *((const int *)q8 + 0), sumi);
        sumi = dpct::dp4a(grid_h, *((const int *)q8 + 1), sumi);
        q8 += 8;
    }
    const float d =
        (float)bq2->d *
        (1 + 2 * ((bq2->scales[ib32 / 2] >> 4 * (ib32 % 2)) & 0xf)) *
        bq8_1[ib32].ds[0];
    return d * sumi;
#else
    assert(false);
#endif
}

static __dpct_inline__ float
vec_dot_iq1_s_q8_1(const void *__restrict__ vbq,
                   const block_q8_1 *__restrict__ bq8_1, const int &iqs,
                   const uint32_t *iq1s_grid_gpu) {
#if QK_K == 256
    const block_iq1_s * bq1 = (const block_iq1_s *) vbq;

    const int ib32 = iqs;
    int sumi = 0;
    const int * q8 = (const int *)bq8_1[ib32].qs;
    for (int l = 0; l < 4; ++l) {
        const int * grid = (const int *)(iq1s_grid_gpu + (bq1->qs[4*ib32+l] | (((bq1->qh[ib32] >> 3*l) & 7) << 8)));
        int grid0 = grid[0] & 0x0f0f0f0f;
        int grid1 = (grid[0] >> 4) & 0x0f0f0f0f;
        sumi = dpct::dp4a(q8[2 * l + 1], grid1,
                          dpct::dp4a(q8[2 * l + 0], grid0, sumi));
    }

    const float delta = bq1->qh[ib32] & 0x8000 ? -1-IQ1S_DELTA : -1+IQ1S_DELTA;
    const float d1q = (float)bq1->d * (2*((bq1->qh[ib32] >> 12) & 7) + 1);
    const float d = d1q * bq8_1[ib32].ds[0];
    const float m = d1q * bq8_1[ib32].ds[1];
    return d * sumi + m * delta;
#else
    assert(false);
#endif
}

static __dpct_inline__ float
vec_dot_iq1_m_q8_1(const void *__restrict__ vbq,
                   const block_q8_1 *__restrict__ bq8_1, const int &iqs) {
#if QK_K == 256
    const block_iq1_m * bq1 = (const block_iq1_m *) vbq;

    const int ib32 = iqs;
    int   sumi[2] = {0, 0};
    float sumf[2] = {0.f, 0.f};

    const int * q8 = (const int *)bq8_1[ib32].qs;
    for (int l = 0; l < 4; ++l) {
        const int * grid = (const int *)(iq1s_grid_gpu + (bq1->qs[4*ib32+l] | (((bq1->qh[2*ib32+l/2] >> 4*(l%2)) & 7) << 8)));
        int grid0 = grid[0] & 0x0f0f0f0f;
        int grid1 = (grid[0] >> 4) & 0x0f0f0f0f;
        sumi[l / 2] = dpct::dp4a(q8[2 * l + 1], grid1,
                                 dpct::dp4a(q8[2 * l + 0], grid0, sumi[l / 2]));
        const float delta = (bq1->qh[2*ib32+l/2] >> 4*(l%2)) & 0x08 ? -1-IQ1M_DELTA : -1+IQ1M_DELTA;
        const int sumy = dpct::dp4a(q8[2 * l + 1], 0x01010101,
                                    dpct::dp4a(q8[2 * l + 0], 0x01010101, 0));
        sumf[l/2] += delta*sumy;
    }

    iq1m_scale_t scale;
    const uint16_t * sc = (const uint16_t *)bq1->scales;
    scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
    const float d = (float)scale.f16 * bq8_1[ib32].ds[0];
    return d * ((sumi[0] + sumf[0]) * (2*((sc[ib32/2] >> 6*(ib32%2)) & 0x7) + 1) + (sumi[1] + sumf[1]) * (2*((sc[ib32/2] >> (6*(ib32%2)+3)) & 0x7) + 1));
#else
    assert(false);
#endif
}


static __dpct_inline__ float
vec_dot_iq4_nl_q8_1(const void *__restrict__ vbq,
                    const block_q8_1 *__restrict__ bq8_1, const int &iqs) {

    const block_iq4_nl * bq = (const block_iq4_nl *) vbq;

    const uint16_t * q4 = (const uint16_t *)bq->qs + 2*iqs;
    const int32_t  * q8 = (const int32_t  *)bq8_1->qs + iqs;

    const uint8_t * values = (const uint8_t *)kvalues_iq4nl;

    int v1, v2;
    int sumi1 = 0, sumi2 = 0;
    for (int l = 0; l < VDR_Q4_0_Q8_1_MMVQ; ++l) {
        const uint32_t aux = q4[2*l] | (q4[2*l+1] << 16);
        get_int_from_table_16(aux, values, v1, v2);
        sumi1 = dpct::dp4a(v1, q8[l + 0], sumi1);
        sumi2 = dpct::dp4a(v2, q8[l + 4], sumi2);
    }

    const float d = (float)bq->d * bq8_1->ds[0];
    return d * (sumi1 + sumi2);
}


static __dpct_inline__ float
vec_dot_iq4_xs_q8_1(const void *__restrict__ vbq,
                    const block_q8_1 *__restrict__ bq8_1, const int &iqs) {

#if QK_K == 256
    const block_iq4_xs * bq4 = (const block_iq4_xs *) vbq;
    const uint8_t * values = (const uint8_t *)kvalues_iq4nl;

    // iqs is 0...7
    const int ib32 = iqs;
    const int32_t  * q8 = (const int *)bq8_1[ib32].qs;
    const uint32_t * q4 = (const uint32_t *)bq4->qs + 4*ib32;
    const int8_t ls = ((bq4->scales_l[ib32/2] >> 4*(ib32%2)) & 0xf) | (((bq4->scales_h >> 2*ib32) & 3) << 4);
    const float d = (float)bq4->d * (ls - 32) * bq8_1[ib32].ds[0];
    int v1, v2;
    int sumi1 = 0, sumi2 = 0;
    for (int j = 0; j < 4; ++j) {
        get_int_from_table_16(q4[j], values, v1, v2);
        sumi1 = dpct::dp4a(v1, q8[j + 0], sumi1);
        sumi2 = dpct::dp4a(v2, q8[j + 4], sumi2);
    }
    return d * (sumi1 + sumi2);
#else
    assert(false);
#endif
}

#endif // GGML_SYCL_VECDOTQ_HPP
