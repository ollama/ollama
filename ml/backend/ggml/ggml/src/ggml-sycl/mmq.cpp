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

#include "mmq.hpp"
#include "vecdotq.hpp"

typedef void (*allocate_tiles_sycl_t)(
    int** x_ql,
    sycl::half2** x_dm,
    int** x_qh,
    int** x_sc);
typedef void (*load_tiles_sycl_t)(
    const void* __restrict__ vx,
    int* __restrict__ x_ql,
    sycl::half2* __restrict__ x_dm,
    int* __restrict__ x_qh,
    int* __restrict__ x_sc,
    const int& i_offset,
    const int& i_max,
    const int& k,
    const int& blocks_per_row);
typedef float (*vec_dot_q_mul_mat_sycl_t)(
    const int* __restrict__ x_ql,
    const sycl::half2* __restrict__ x_dm,
    const int* __restrict__ x_qh,
    const int* __restrict__ x_sc,
    const int* __restrict__ y_qs,
    const sycl::half2* __restrict__ y_ms,
    const int& i,
    const int& j,
    const int& k);


template <int mmq_y>
static __dpct_inline__ void
allocate_tiles_q4_0(int **x_ql, sycl::half2 **x_dm, int **x_qh, int **x_sc,
                    int *tile_x_qs_q4_0, float *tile_x_d_q4_0) {
    (void)x_qh; (void)x_sc;

    *x_ql = tile_x_qs_q4_0;
    *x_dm = (sycl::half2 *)tile_x_d_q4_0;
}

template <int mmq_y, int nwarps, bool need_check>
static __dpct_inline__ void
load_tiles_q4_0(const void *__restrict__ vx, int *__restrict__ x_ql,
                sycl::half2 *__restrict__ x_dm, int *__restrict__ x_qh,
                int *__restrict__ x_sc, const int &i_offset, const int &i_max,
                const int &k, const int &blocks_per_row) {
    (void)x_qh; (void)x_sc;
    GGML_SYCL_ASSUME(i_offset >= 0);
    GGML_SYCL_ASSUME(i_offset <  nwarps);
    GGML_SYCL_ASSUME(k >= 0);
    GGML_SYCL_ASSUME(k <  WARP_SIZE);

    const int kbx  = k / QI4_0;
    const int kqsx = k % QI4_0;

    const block_q4_0 * bx0 = (const block_q4_0 *) vx;

    float * x_dmf = (float *) x_dm;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + i_offset;

        if (need_check) {
            i = sycl::min(i, i_max);
        }

        const block_q4_0 * bxi = bx0 + i*blocks_per_row + kbx;

        x_ql[i * (WARP_SIZE + 1) + k] = get_int_from_uint8(bxi->qs, kqsx);
        // x_dmf[i * (WARP_SIZE/QI4_0) + i / QI4_0 + kbx] = bxi->d;
    }

    const int blocks_per_tile_x_row = WARP_SIZE / QI4_0;
    const int kbxd = k % blocks_per_tile_x_row;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI4_0) {
        int i = i0 + i_offset * QI4_0 + k / blocks_per_tile_x_row;

        if (need_check) {
            i = sycl::min(i, i_max);
        }

        const block_q4_0 * bxi = bx0 + i*blocks_per_row + kbxd;

        x_dmf[i * (WARP_SIZE/QI4_0) + i / QI4_0 + kbxd] = bxi->d;
    }
}

static __dpct_inline__ float vec_dot_q4_0_q8_1_mul_mat(
    const int *__restrict__ x_ql, const sycl::half2 *__restrict__ x_dm,
    const int *__restrict__ x_qh, const int *__restrict__ x_sc,
    const int *__restrict__ y_qs, const sycl::half2 *__restrict__ y_ds,
    const int &i, const int &j, const int &k) {
    (void)x_qh; (void)x_sc;

    const int kyqs = k % (QI8_1/2) + QI8_1 * (k / (QI8_1/2));
    const float * x_dmf = (const float *) x_dm;

    int u[2*VDR_Q4_0_Q8_1_MMQ];

#pragma unroll
    for (int l = 0; l < VDR_Q4_0_Q8_1_MMQ; ++l) {
        u[2*l+0] = y_qs[j * WARP_SIZE + (kyqs + l)         % WARP_SIZE];
        u[2*l+1] = y_qs[j * WARP_SIZE + (kyqs + l + QI4_0) % WARP_SIZE];
    }

    return vec_dot_q4_0_q8_1_impl<VDR_Q4_0_Q8_1_MMQ>
        (&x_ql[i * (WARP_SIZE + 1) + k], u, x_dmf[i * (WARP_SIZE/QI4_0) + i/QI4_0 + k/QI4_0],
         y_ds[j * (WARP_SIZE/QI8_1) + (2*k/QI8_1) % (WARP_SIZE/QI8_1)]);
}

template <int mmq_y>
static __dpct_inline__ void
allocate_tiles_q4_1(int **x_ql, sycl::half2 **x_dm, int **x_qh, int **x_sc,
                    int *tile_x_qs_q4_1, sycl::half2 *tile_x_dm_q4_1) {
    (void)x_qh; (void)x_sc;

    *x_ql = tile_x_qs_q4_1;
    *x_dm = tile_x_dm_q4_1;
}


template <int mmq_y, int nwarps, bool need_check>
static __dpct_inline__ void
load_tiles_q4_1(const void *__restrict__ vx, int *__restrict__ x_ql,
                sycl::half2 *__restrict__ x_dm, int *__restrict__ x_qh,
                int *__restrict__ x_sc, const int &i_offset, const int &i_max,
                const int &k, const int &blocks_per_row) {
    (void)x_qh; (void)x_sc;

    GGML_SYCL_ASSUME(i_offset >= 0);
    GGML_SYCL_ASSUME(i_offset <  nwarps);
    GGML_SYCL_ASSUME(k >= 0);
    GGML_SYCL_ASSUME(k <  WARP_SIZE);

    const int kbx  = k / QI4_1;
    const int kqsx = k % QI4_1;

    const block_q4_1 * bx0 = (const block_q4_1 *) vx;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + i_offset;

        if (need_check) {
            i = sycl::min(i, i_max);
        }

        const block_q4_1 * bxi = bx0 + i*blocks_per_row + kbx;

        x_ql[i * (WARP_SIZE + 1) + k] = get_int_from_uint8_aligned(bxi->qs, kqsx);
    }

    const int blocks_per_tile_x_row = WARP_SIZE / QI4_1;
    const int kbxd = k % blocks_per_tile_x_row;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI4_1) {
        int i = i0 + i_offset * QI4_1 + k / blocks_per_tile_x_row;

        if (need_check) {
            i = sycl::min(i, i_max);
        }

        const block_q4_1 * bxi = bx0 + i*blocks_per_row + kbxd;

        x_dm[i * (WARP_SIZE/QI4_1) + i / QI4_1 + kbxd] = bxi->dm;
    }
}

static __dpct_inline__ float vec_dot_q4_1_q8_1_mul_mat(
    const int *__restrict__ x_ql, const sycl::half2 *__restrict__ x_dm,
    const int *__restrict__ x_qh, const int *__restrict__ x_sc,
    const int *__restrict__ y_qs, const sycl::half2 *__restrict__ y_ds,
    const int &i, const int &j, const int &k) {
    (void)x_qh; (void)x_sc;

    const int kyqs = k % (QI8_1/2) + QI8_1 * (k / (QI8_1/2));

    int u[2*VDR_Q4_1_Q8_1_MMQ];

#pragma unroll
    for (int l = 0; l < VDR_Q4_1_Q8_1_MMQ; ++l) {
        u[2*l+0] = y_qs[j * WARP_SIZE + (kyqs + l)         % WARP_SIZE];
        u[2*l+1] = y_qs[j * WARP_SIZE + (kyqs + l + QI4_1) % WARP_SIZE];
    }

    return vec_dot_q4_1_q8_1_impl<VDR_Q4_1_Q8_1_MMQ>
        (&x_ql[i * (WARP_SIZE + 1) + k], u, x_dm[i * (WARP_SIZE/QI4_1) + i/QI4_1 + k/QI4_1],
         y_ds[j * (WARP_SIZE/QI8_1) + (2*k/QI8_1) % (WARP_SIZE/QI8_1)]);
}

template <int mmq_y>
static __dpct_inline__ void
allocate_tiles_q5_0(int **x_ql, sycl::half2 **x_dm, int **x_qh, int **x_sc,
                    int *tile_x_ql_q5_0, float *tile_x_d_q5_0) {
    (void)x_qh; (void)x_sc;

    *x_ql = tile_x_ql_q5_0;
    *x_dm = (sycl::half2 *)tile_x_d_q5_0;
}

template <int mmq_y, int nwarps, bool need_check>
static __dpct_inline__ void
load_tiles_q5_0(const void *__restrict__ vx, int *__restrict__ x_ql,
                sycl::half2 *__restrict__ x_dm, int *__restrict__ x_qh,
                int *__restrict__ x_sc, const int &i_offset, const int &i_max,
                const int &k, const int &blocks_per_row) {
    (void)x_qh; (void)x_sc;

    GGML_SYCL_ASSUME(i_offset >= 0);
    GGML_SYCL_ASSUME(i_offset <  nwarps);
    GGML_SYCL_ASSUME(k >= 0);
    GGML_SYCL_ASSUME(k <  WARP_SIZE);

    const int kbx  = k / QI5_0;
    const int kqsx = k % QI5_0;

    const block_q5_0 * bx0 = (const block_q5_0 *) vx;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + i_offset;

        if (need_check) {
            i = sycl::min(i, i_max);
        }

        const block_q5_0 * bxi = bx0 + i*blocks_per_row + kbx;

        const int ql = get_int_from_uint8(bxi->qs, kqsx);
        const int qh = get_int_from_uint8(bxi->qh, 0) >> (4 * (k % QI5_0));

        int qs0 = (ql >>  0)   & 0x0F0F0F0F;
        qs0    |= (qh <<  4)   & 0x00000010;  // 0 ->  4
        qs0    |= (qh << 11)   & 0x00001000;  // 1 -> 12
        qs0    |= (qh << 18)   & 0x00100000;  // 2 -> 20
        qs0    |= (qh << 25)   & 0x10000000;  // 3 -> 28
        qs0 = dpct::vectorized_binary<sycl::char4>(
            qs0, 0x10101010, dpct::sub_sat()); // subtract 16

        x_ql[i * (2*WARP_SIZE + 1) + 2*k+0] = qs0;

        int qs1 = (ql >>  4)   & 0x0F0F0F0F;
        qs1    |= (qh >> 12)   & 0x00000010;  // 16 ->  4
        qs1    |= (qh >>  5)   & 0x00001000;  // 17 -> 12
        qs1    |= (qh <<  2)   & 0x00100000;  // 18 -> 20
        qs1    |= (qh <<  9)   & 0x10000000;  // 19 -> 28
        qs1 = dpct::vectorized_binary<sycl::char4>(
            qs1, 0x10101010, dpct::sub_sat()); // subtract 16

        x_ql[i * (2*WARP_SIZE + 1) + 2*k+1] = qs1;
    }

    const int blocks_per_tile_x_row = WARP_SIZE / QI5_0;
    const int kbxd = k % blocks_per_tile_x_row;
    float * x_dmf = (float *) x_dm;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI5_0) {
        int i = i0 + i_offset * QI5_0 + k / blocks_per_tile_x_row;

        if (need_check) {
            i = sycl::min(i, i_max);
        }

        const block_q5_0 * bxi = bx0 + i*blocks_per_row + kbxd;

        x_dmf[i * (WARP_SIZE/QI5_0) + i / QI5_0 + kbxd] = bxi->d;
    }
}

static __dpct_inline__ float vec_dot_q5_0_q8_1_mul_mat(
    const int *__restrict__ x_ql, const sycl::half2 *__restrict__ x_dm,
    const int *__restrict__ x_qh, const int *__restrict__ x_sc,
    const int *__restrict__ y_qs, const sycl::half2 *__restrict__ y_ds,
    const int &i, const int &j, const int &k) {
    (void)x_qh; (void)x_sc;

    const int kyqs = k % (QI8_1/2) + QI8_1 * (k / (QI8_1/2));
    const int index_bx = i * (WARP_SIZE/QI5_0) + i/QI5_0 + k/QI5_0;
    const float * x_dmf = (const float *) x_dm;
    const float * y_df  = (const float *) y_ds;

    int u[2*VDR_Q5_0_Q8_1_MMQ];

#pragma unroll
    for (int l = 0; l < VDR_Q5_0_Q8_1_MMQ; ++l) {
        u[2*l+0] = y_qs[j * WARP_SIZE + (kyqs + l)         % WARP_SIZE];
        u[2*l+1] = y_qs[j * WARP_SIZE + (kyqs + l + QI5_0) % WARP_SIZE];
    }

    return vec_dot_q8_0_q8_1_impl<QR5_0*VDR_Q5_0_Q8_1_MMQ>
        (&x_ql[i * (2*WARP_SIZE + 1) + 2 * k], u, x_dmf[index_bx], y_df[j * (WARP_SIZE/QI8_1) + (2*k/QI8_1) % (WARP_SIZE/QI8_1)]);
}

template <int mmq_y>
static __dpct_inline__ void
allocate_tiles_q5_1(int **x_ql, sycl::half2 **x_dm, int **x_qh, int **x_sc,
                    int *tile_x_ql_q5_1, sycl::half2 *tile_x_dm_q5_1) {
    (void)x_qh; (void)x_sc;

    *x_ql = tile_x_ql_q5_1;
    *x_dm = tile_x_dm_q5_1;
}

template <int mmq_y, int nwarps, bool need_check>
static __dpct_inline__ void
load_tiles_q5_1(const void *__restrict__ vx, int *__restrict__ x_ql,
                sycl::half2 *__restrict__ x_dm, int *__restrict__ x_qh,
                int *__restrict__ x_sc, const int &i_offset, const int &i_max,
                const int &k, const int &blocks_per_row) {
    (void)x_qh; (void)x_sc;

    GGML_SYCL_ASSUME(i_offset >= 0);
    GGML_SYCL_ASSUME(i_offset < nwarps);
    GGML_SYCL_ASSUME(k >= 0);
    GGML_SYCL_ASSUME(k <  WARP_SIZE);

    const int kbx  = k / QI5_1;
    const int kqsx = k % QI5_1;

    const block_q5_1 * bx0 = (const block_q5_1 *) vx;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + i_offset;

        if (need_check) {
            i = sycl::min(i, i_max);
        }

        const block_q5_1 * bxi = bx0 + i*blocks_per_row + kbx;

        const int ql = get_int_from_uint8_aligned(bxi->qs, kqsx);
        const int qh = get_int_from_uint8_aligned(bxi->qh, 0) >> (4 * (k % QI5_1));

        int qs0 = (ql >>  0) & 0x0F0F0F0F;
        qs0    |= (qh <<  4) & 0x00000010; // 0 ->  4
        qs0    |= (qh << 11) & 0x00001000; // 1 -> 12
        qs0    |= (qh << 18) & 0x00100000; // 2 -> 20
        qs0    |= (qh << 25) & 0x10000000; // 3 -> 28

        x_ql[i * (2*WARP_SIZE + 1) + 2*k+0] = qs0;

        int qs1 = (ql >>  4) & 0x0F0F0F0F;
        qs1    |= (qh >> 12) & 0x00000010; // 16 ->  4
        qs1    |= (qh >>  5) & 0x00001000; // 17 -> 12
        qs1    |= (qh <<  2) & 0x00100000; // 18 -> 20
        qs1    |= (qh <<  9) & 0x10000000; // 19 -> 28

        x_ql[i * (2*WARP_SIZE + 1) + 2*k+1] = qs1;
    }

    const int blocks_per_tile_x_row = WARP_SIZE / QI5_1;
    const int kbxd = k % blocks_per_tile_x_row;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI5_1) {
        int i = i0 + i_offset * QI5_1 + k / blocks_per_tile_x_row;

        if (need_check) {
            i = sycl::min(i, i_max);
        }

        const block_q5_1 * bxi = bx0 + i*blocks_per_row + kbxd;

        x_dm[i * (WARP_SIZE/QI5_1) + i / QI5_1 + kbxd] = bxi->dm;
    }
}

static __dpct_inline__ float vec_dot_q5_1_q8_1_mul_mat(
    const int *__restrict__ x_ql, const sycl::half2 *__restrict__ x_dm,
    const int *__restrict__ x_qh, const int *__restrict__ x_sc,
    const int *__restrict__ y_qs, const sycl::half2 *__restrict__ y_ds,
    const int &i, const int &j, const int &k) {
    (void)x_qh; (void)x_sc;

    const int kyqs = k % (QI8_1/2) + QI8_1 * (k / (QI8_1/2));
    const int index_bx = i * (WARP_SIZE/QI5_1) + + i/QI5_1 + k/QI5_1;

    int u[2*VDR_Q5_1_Q8_1_MMQ];

#pragma unroll
    for (int l = 0; l < VDR_Q5_1_Q8_1_MMQ; ++l) {
        u[2*l+0] = y_qs[j * WARP_SIZE + (kyqs + l)         % WARP_SIZE];
        u[2*l+1] = y_qs[j * WARP_SIZE + (kyqs + l + QI5_1) % WARP_SIZE];
    }

    return vec_dot_q8_1_q8_1_impl<QR5_1*VDR_Q5_1_Q8_1_MMQ>
        (&x_ql[i * (2*WARP_SIZE + 1) + 2 * k], u, x_dm[index_bx], y_ds[j * (WARP_SIZE/QI8_1) + (2*k/QI8_1) % (WARP_SIZE/QI8_1)]);
}

template <int mmq_y>
static __dpct_inline__ void
allocate_tiles_q8_0(int **x_ql, sycl::half2 **x_dm, int **x_qh, int **x_sc,
                    int *tile_x_qs_q8_0, float *tile_x_d_q8_0) {
    (void)x_qh; (void)x_sc;

    *x_ql = tile_x_qs_q8_0;
    *x_dm = (sycl::half2 *)tile_x_d_q8_0;
}

template <int mmq_y, int nwarps, bool need_check>
static __dpct_inline__ void
load_tiles_q8_0(const void *__restrict__ vx, int *__restrict__ x_ql,
                sycl::half2 *__restrict__ x_dm, int *__restrict__ x_qh,
                int *__restrict__ x_sc, const int &i_offset, const int &i_max,
                const int &k, const int &blocks_per_row) {
    (void)x_qh; (void)x_sc;

    GGML_SYCL_ASSUME(i_offset >= 0);
    GGML_SYCL_ASSUME(i_offset <  nwarps);
    GGML_SYCL_ASSUME(k >= 0);
    GGML_SYCL_ASSUME(k <  WARP_SIZE);

    const int kbx  = k / QI8_0;
    const int kqsx = k % QI8_0;
    float * x_dmf = (float *) x_dm;

    const block_q8_0 * bx0 = (const block_q8_0 *) vx;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + i_offset;

        if (need_check) {
            i = sycl::min(i, i_max);
        }

        const block_q8_0 * bxi = bx0 + i*blocks_per_row + kbx;

        x_ql[i * (WARP_SIZE + 1) + k] = get_int_from_int8(bxi->qs, kqsx);
    }

    const int blocks_per_tile_x_row = WARP_SIZE / QI8_0;
    const int kbxd = k % blocks_per_tile_x_row;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI8_0) {
        int i = i0 + i_offset * QI8_0 + k / blocks_per_tile_x_row;

        if (need_check) {
            i = sycl::min(i, i_max);
        }

        const block_q8_0 * bxi = bx0 + i*blocks_per_row + kbxd;

        x_dmf[i * (WARP_SIZE/QI8_0) + i / QI8_0 + kbxd] = bxi->d;
    }
}

static __dpct_inline__ float vec_dot_q8_0_q8_1_mul_mat(
    const int *__restrict__ x_ql, const sycl::half2 *__restrict__ x_dm,
    const int *__restrict__ x_qh, const int *__restrict__ x_sc,
    const int *__restrict__ y_qs, const sycl::half2 *__restrict__ y_ds,
    const int &i, const int &j, const int &k) {
    (void)x_qh; (void)x_sc;

    const float * x_dmf = (const float *) x_dm;
    const float * y_df  = (const float *) y_ds;

    return vec_dot_q8_0_q8_1_impl<VDR_Q8_0_Q8_1_MMQ>
        (&x_ql[i * (WARP_SIZE + 1) + k], &y_qs[j * WARP_SIZE + k], x_dmf[i * (WARP_SIZE/QI8_0) + i/QI8_0 + k/QI8_0],
         y_df[j * (WARP_SIZE/QI8_1) + k/QI8_1]);
}

template <int mmq_y>
static __dpct_inline__ void
allocate_tiles_q2_K(int **x_ql, sycl::half2 **x_dm, int **x_qh, int **x_sc,
                    int *tile_x_ql_q2_K, sycl::half2 *tile_x_dm_q2_K,
                    int *tile_x_sc_q2_K) {
    (void)x_qh;

    *x_ql = tile_x_ql_q2_K;
    *x_dm = tile_x_dm_q2_K;
    *x_sc = tile_x_sc_q2_K;
}

template <int mmq_y, int nwarps, bool need_check>
static __dpct_inline__ void
load_tiles_q2_K(const void *__restrict__ vx, int *__restrict__ x_ql,
                sycl::half2 *__restrict__ x_dm, int *__restrict__ x_qh,
                int *__restrict__ x_sc, const int &i_offset, const int &i_max,
                const int &k, const int &blocks_per_row) {
    (void)x_qh;

    GGML_SYCL_ASSUME(i_offset >= 0);
    GGML_SYCL_ASSUME(i_offset <  nwarps);
    GGML_SYCL_ASSUME(k >= 0);
    GGML_SYCL_ASSUME(k <  WARP_SIZE);

    const int kbx  = k / QI2_K;
    const int kqsx = k % QI2_K;

    const block_q2_K * bx0 = (const block_q2_K *) vx;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + i_offset;

        if (need_check) {
            i = sycl::min(i, i_max);
        }

        const block_q2_K * bxi = bx0 + i*blocks_per_row + kbx;

        x_ql[i * (WARP_SIZE + 1) + k] = get_int_from_uint8_aligned(bxi->qs, kqsx);
    }

    const int blocks_per_tile_x_row = WARP_SIZE / QI2_K;
    const int kbxd = k % blocks_per_tile_x_row;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI2_K) {
        int i = (i0 + i_offset * QI2_K + k / blocks_per_tile_x_row) % mmq_y;

        if (need_check) {
            i = sycl::min(i, i_max);
        }

        const block_q2_K * bxi = bx0 + i*blocks_per_row + kbxd;

        x_dm[i * (WARP_SIZE/QI2_K) + i / QI2_K + kbxd] = bxi->dm;
    }

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * 4) {
        int i = i0 + i_offset * 4 + k / (WARP_SIZE/4);

        if (need_check) {
            i = sycl::min(i, i_max);
        }

        const block_q2_K * bxi = bx0 + i*blocks_per_row + (k % (WARP_SIZE/4)) / (QI2_K/4);

        x_sc[i * (WARP_SIZE/4) + i / 4 + k % (WARP_SIZE/4)] = get_int_from_uint8_aligned(bxi->scales, k % (QI2_K/4));
    }
}

#define VDR_Q2_K_Q8_1_MMQ  2
// contiguous u/y values
static __dpct_inline__ float
vec_dot_q2_K_q8_1_impl_mmq(const int *__restrict__ v, const int *__restrict__ u,
                           const uint8_t *__restrict__ scales,
                           const sycl::half2 &dm2, const float &d8) {

    int sumi_d = 0;
    int sumi_m = 0;

#pragma unroll
    for (int i0 = 0; i0 < QI8_1; i0 += QI8_1/2) {
        int sumi_d_sc = 0;

        const int sc = scales[i0 / (QI8_1/2)];

        // fill int with 4x m
        int m = sc >> 4;
        m |= m <<  8;
        m |= m << 16;

#pragma unroll
        for (int i = i0; i < i0 + QI8_1/2; ++i) {
            sumi_d_sc = dpct::dp4a(v[i], u[i], sumi_d_sc); // SIMD dot product
            sumi_m = dpct::dp4a(m, u[i],
                                sumi_m); // multiply sum of q8_1 values with m
        }

        sumi_d += sumi_d_sc * (sc & 0xF);
    }

    const sycl::float2 dm2f =
        dm2.convert<float, sycl::rounding_mode::automatic>();

    return d8 * (dm2f.x() * sumi_d - dm2f.y() * sumi_m);
}

static __dpct_inline__ float vec_dot_q2_K_q8_1_mul_mat(
    const int *__restrict__ x_ql, const sycl::half2 *__restrict__ x_dm,
    const int *__restrict__ x_qh, const int *__restrict__ x_sc,
    const int *__restrict__ y_qs, const sycl::half2 *__restrict__ y_ds,
    const int &i, const int &j, const int &k) {
    (void)x_qh;

    const int kbx = k / QI2_K;
    const int ky  = (k % QI2_K) * QR2_K;
    const float * y_df = (const float *) y_ds;

    int v[QR2_K*VDR_Q2_K_Q8_1_MMQ];

    const int kqsx = i * (WARP_SIZE + 1) + kbx*QI2_K + (QI2_K/2) * (ky/(2*QI2_K)) + ky % (QI2_K/2);
    const int shift = 2 * ((ky % (2*QI2_K)) / (QI2_K/2));

#pragma unroll
    for (int l = 0; l < QR2_K*VDR_Q2_K_Q8_1_MMQ; ++l) {
        v[l] = (x_ql[kqsx + l] >> shift) & 0x03030303;
    }

    const uint8_t * scales = ((const uint8_t *) &x_sc[i * (WARP_SIZE/4) + i/4 + kbx*4]) + ky/4;

    const int index_y = j * WARP_SIZE + (QR2_K*k) % WARP_SIZE;
    return vec_dot_q2_K_q8_1_impl_mmq(v, &y_qs[index_y], scales, x_dm[i * (WARP_SIZE/QI2_K) + i/QI2_K + kbx], y_df[index_y/QI8_1]);
}

template <int mmq_y>
static __dpct_inline__ void
allocate_tiles_q3_K(int **x_ql, sycl::half2 **x_dm, int **x_qh, int **x_sc,
                    int *tile_x_ql_q3_K, sycl::half2 *tile_x_dm_q3_K,
                    int *tile_x_qh_q3_K, int *tile_x_sc_q3_K) {

    *x_ql = tile_x_ql_q3_K;
    *x_dm = tile_x_dm_q3_K;
    *x_qh = tile_x_qh_q3_K;
    *x_sc = tile_x_sc_q3_K;
}

template <int mmq_y, int nwarps, bool need_check>
static __dpct_inline__ void
load_tiles_q3_K(const void *__restrict__ vx, int *__restrict__ x_ql,
                sycl::half2 *__restrict__ x_dm, int *__restrict__ x_qh,
                int *__restrict__ x_sc, const int &i_offset, const int &i_max,
                const int &k, const int &blocks_per_row) {

    GGML_SYCL_ASSUME(i_offset >= 0);
    GGML_SYCL_ASSUME(i_offset <  nwarps);
    GGML_SYCL_ASSUME(k >= 0);
    GGML_SYCL_ASSUME(k <  WARP_SIZE);

    const int kbx  = k / QI3_K;
    const int kqsx = k % QI3_K;

    const block_q3_K * bx0 = (const block_q3_K *) vx;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + i_offset;

        if (need_check) {
            i = sycl::min(i, i_max);
        }

        const block_q3_K * bxi = bx0 + i*blocks_per_row + kbx;

        x_ql[i * (WARP_SIZE + 1) + k] = get_int_from_uint8(bxi->qs, kqsx);
    }

    const int blocks_per_tile_x_row = WARP_SIZE / QI3_K;
    const int kbxd = k % blocks_per_tile_x_row;
    float * x_dmf = (float *) x_dm;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI3_K) {
        int i = (i0 + i_offset * QI3_K + k / blocks_per_tile_x_row) % mmq_y;

        if (need_check) {
            i = sycl::min(i, i_max);
        }

        const block_q3_K * bxi = bx0 + i*blocks_per_row + kbxd;

        x_dmf[i * (WARP_SIZE/QI3_K) + i / QI3_K + kbxd] = bxi->d;
    }

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * 2) {
        int i = i0 + i_offset * 2 + k / (WARP_SIZE/2);

        if (need_check) {
            i = sycl::min(i, i_max);
        }

        const block_q3_K * bxi = bx0 + i*blocks_per_row + (k % (WARP_SIZE/2)) / (QI3_K/2);

        // invert the mask with ~ so that a 0/1 results in 4/0 being subtracted
        x_qh[i * (WARP_SIZE/2) + i / 2 + k % (WARP_SIZE/2)] = ~get_int_from_uint8(bxi->hmask, k % (QI3_K/2));
    }

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * 4) {
        int i = i0 + i_offset * 4 + k / (WARP_SIZE/4);

        if (need_check) {
            i = sycl::min(i, i_max);
        }

        const block_q3_K * bxi = bx0 + i*blocks_per_row + (k % (WARP_SIZE/4)) / (QI3_K/4);

        const int ksc = k % (QI3_K/4);

        const int ksc_low = ksc % (QI3_K/8);
        const int shift_low = 4 * (ksc / (QI3_K/8));
        const int sc_low = (get_int_from_uint8(bxi->scales, ksc_low) >> shift_low) & 0x0F0F0F0F;

        const int ksc_high = QI3_K/8;
        const int shift_high = 2 * ksc;
        const int sc_high = ((get_int_from_uint8(bxi->scales, ksc_high) >> shift_high) << 4) & 0x30303030;

        const int sc = dpct::vectorized_binary<sycl::char4>(
            sc_low | sc_high, 0x20202020, dpct::sub_sat());

        x_sc[i * (WARP_SIZE/4) + i / 4 + k % (WARP_SIZE/4)] = sc;
    }
}

#define VDR_Q3_K_Q8_1_MMQ  2
// contiguous u/y values
static __dpct_inline__ float
vec_dot_q3_K_q8_1_impl_mmq(const int *__restrict__ v, const int *__restrict__ u,
                           const int8_t *__restrict__ scales, const float &d3,
                           const float &d8) {

    int sumi = 0;

#pragma unroll
    for (int i0 = 0; i0 < QR3_K*VDR_Q3_K_Q8_1_MMQ; i0 += QI8_1/2) {
        int sumi_sc = 0;

        for (int i = i0; i < i0 + QI8_1/2; ++i) {
            sumi_sc = dpct::dp4a(v[i], u[i], sumi_sc); // SIMD dot product
        }

        sumi += sumi_sc * scales[i0 / (QI8_1/2)];
    }

    return d3*d8 * sumi;
}

static __dpct_inline__ float vec_dot_q3_K_q8_1_mul_mat(
    const int *__restrict__ x_ql, const sycl::half2 *__restrict__ x_dm,
    const int *__restrict__ x_qh, const int *__restrict__ x_sc,
    const int *__restrict__ y_qs, const sycl::half2 *__restrict__ y_ds,
    const int &i, const int &j, const int &k) {

    const int kbx  = k / QI3_K;
    const int ky  = (k % QI3_K) * QR3_K;
    const float * x_dmf = (const float *) x_dm;
    const float * y_df  = (const float *) y_ds;

    const int8_t * scales = ((const int8_t *) (x_sc + i * (WARP_SIZE/4) + i/4 + kbx*4)) + ky/4;

    int v[QR3_K*VDR_Q3_K_Q8_1_MMQ];

#pragma unroll
    for (int l = 0; l < QR3_K*VDR_Q3_K_Q8_1_MMQ; ++l) {
        const int kqsx = i * (WARP_SIZE + 1) + kbx*QI3_K + (QI3_K/2) * (ky/(2*QI3_K)) + ky % (QI3_K/2);
        const int shift = 2 * ((ky % 32) / 8);
        const int vll = (x_ql[kqsx + l] >> shift) & 0x03030303;

        const int vh = x_qh[i * (WARP_SIZE/2) + i/2 + kbx * (QI3_K/2) + (ky+l)%8] >> ((ky+l) / 8);
        const int vlh = (vh << 2) & 0x04040404;

        v[l] = dpct::vectorized_binary<sycl::char4>(vll, vlh, dpct::sub_sat());
    }

    const int index_y = j * WARP_SIZE + (k*QR3_K) % WARP_SIZE;
    return vec_dot_q3_K_q8_1_impl_mmq(v, &y_qs[index_y], scales, x_dmf[i * (WARP_SIZE/QI3_K) + i/QI3_K + kbx], y_df[index_y/QI8_1]);
}

template <int mmq_y>
static __dpct_inline__ void
allocate_tiles_q4_K(int **x_ql, sycl::half2 **x_dm, int **x_qh, int **x_sc,
                    int *tile_x_ql_q4_K, sycl::half2 *tile_x_dm_q4_K,
                    int *tile_x_sc_q4_K) {
    (void)x_qh;

    *x_ql = tile_x_ql_q4_K;
    *x_dm = tile_x_dm_q4_K;
    *x_sc = tile_x_sc_q4_K;
}

template <int mmq_y, int nwarps, bool need_check>
static __dpct_inline__ void
load_tiles_q4_K(const void *__restrict__ vx, int *__restrict__ x_ql,
                sycl::half2 *__restrict__ x_dm, int *__restrict__ x_qh,
                int *__restrict__ x_sc, const int &i_offset, const int &i_max,
                const int &k, const int &blocks_per_row) {
    (void)x_qh;

    GGML_SYCL_ASSUME(i_offset >= 0);
    GGML_SYCL_ASSUME(i_offset <  nwarps);
    GGML_SYCL_ASSUME(k >= 0);
    GGML_SYCL_ASSUME(k <  WARP_SIZE);

    const int kbx  = k / QI4_K; // == 0 if QK_K == 256
    const int kqsx = k % QI4_K; // == k if QK_K == 256

    const block_q4_K * bx0 = (const block_q4_K *) vx;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + i_offset;

        if (need_check) {
            i = sycl::min(i, i_max);
        }

        const block_q4_K * bxi = bx0 + i*blocks_per_row + kbx;

        x_ql[i * (WARP_SIZE + 1) + k] = get_int_from_uint8_aligned(bxi->qs, kqsx);
    }

    constexpr int blocks_per_tile_x_row = QI4_K > WARP_SIZE ? 1 : WARP_SIZE / QI4_K; // == 1 if QK_K == 256
    const int kbxd = k % blocks_per_tile_x_row;          // == 0 if QK_K == 256

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI4_K) {
        int i = (i0 + i_offset * QI4_K + k / blocks_per_tile_x_row) % mmq_y;

        if (need_check) {
            i = sycl::min(i, i_max);
        }

        const block_q4_K * bxi = bx0 + i*blocks_per_row + kbxd;

#if QK_K == 256
        x_dm[i * (WARP_SIZE/QI4_K) + i / QI4_K + kbxd] = bxi->dm;
#else
        x_dm[i * (WARP_SIZE/QI4_K) + i / QI4_K + kbxd] = {bxi->dm[0], bxi->dm[1]};
#endif
    }

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * 8) {
        int i = (i0 + i_offset * 8 + k / (WARP_SIZE/8)) % mmq_y;

        if (need_check) {
            i = sycl::min(i, i_max);
        }

        const block_q4_K * bxi = bx0 + i*blocks_per_row + (k % (WARP_SIZE/8)) / (QI4_K/8);

        const int * scales = (const int *) bxi->scales;

        const int ksc = k % (WARP_SIZE/8);

        // scale arrangement after the following two lines: sc0,...,sc3, sc4,...,sc7, m0,...,m3, m4,...,m8
        int scales8 = (scales[(ksc%2) + (ksc!=0)] >> (4 * (ksc & (ksc/2)))) & 0x0F0F0F0F; // lower 4 bits
        scales8    |= (scales[ksc/2]              >> (2 * (ksc % 2)))       & 0x30303030; // upper 2 bits

        x_sc[i * (WARP_SIZE/8) + i / 8 + ksc] = scales8;
    }
}


#define VDR_Q4_K_Q8_1_MMQ  8

// contiguous u/y values
static __dpct_inline__ float vec_dot_q4_K_q8_1_impl_mmq(
    const int *__restrict__ v, const int *__restrict__ u,
    const uint8_t *__restrict__ sc, const uint8_t *__restrict__ m,
    const sycl::half2 &dm4, const sycl::half2 *__restrict__ ds8) {

    float sumf_d = 0.0f;
    float sumf_m = 0.0f;

#pragma unroll
    for (int i = 0; i < QR4_K*VDR_Q4_K_Q8_1_MMQ/QI8_1; ++i) {
        int sumi_d = 0;

#pragma unroll
        for (int j = 0; j < QI8_1; ++j) {
            sumi_d = dpct::dp4a((v[j] >> (4 * i)) & 0x0F0F0F0F,
                                u[i * QI8_1 + j], sumi_d); // SIMD dot product
        }

        const sycl::float2 ds8f =
            ds8[i].convert<float, sycl::rounding_mode::automatic>();

        sumf_d += ds8f.x() * (sc[i] * sumi_d);
        sumf_m += ds8f.y() * m[i]; // sum of q8_1 block * q4_K min val
    }

    const sycl::float2 dm4f =
        dm4.convert<float, sycl::rounding_mode::automatic>();

    return dm4f.x() * sumf_d - dm4f.y() * sumf_m;
}


static __dpct_inline__ float vec_dot_q4_K_q8_1_mul_mat(
    const int *__restrict__ x_ql, const sycl::half2 *__restrict__ x_dm,
    const int *__restrict__ x_qh, const int *__restrict__ x_sc,
    const int *__restrict__ y_qs, const sycl::half2 *__restrict__ y_ds,
    const int &i, const int &j, const int &k) {
    (void)x_qh;

    const uint8_t * sc = ((const uint8_t *) &x_sc[i * (WARP_SIZE/8) + i/8 + k/16]) + 2*((k % 16) / 8);

    const int index_y = j * WARP_SIZE + (QR4_K*k) % WARP_SIZE;
    return vec_dot_q4_K_q8_1_impl_mmq(&x_ql[i * (WARP_SIZE + 1) + k], &y_qs[index_y], sc, sc+8,
                                      x_dm[i * (WARP_SIZE/QI4_K) + i/QI4_K], &y_ds[index_y/QI8_1]);
}

template <int mmq_y>
static __dpct_inline__ void
allocate_tiles_q5_K(int **x_ql, sycl::half2 **x_dm, int **x_qh, int **x_sc,
                    int *tile_x_ql_q5_K, sycl::half2 *tile_x_dm_q5_K,
                    int *tile_x_sc_q5_K) {
    (void)x_qh;

    *x_ql = tile_x_ql_q5_K;
    *x_dm = tile_x_dm_q5_K;
    *x_sc = tile_x_sc_q5_K;
}

template <int mmq_y, int nwarps, bool need_check>
static __dpct_inline__ void
load_tiles_q5_K(const void *__restrict__ vx, int *__restrict__ x_ql,
                sycl::half2 *__restrict__ x_dm, int *__restrict__ x_qh,
                int *__restrict__ x_sc, const int &i_offset, const int &i_max,
                const int &k, const int &blocks_per_row) {
    (void)x_qh;

    GGML_SYCL_ASSUME(i_offset >= 0);
    GGML_SYCL_ASSUME(i_offset <  nwarps);
    GGML_SYCL_ASSUME(k >= 0);
    GGML_SYCL_ASSUME(k <  WARP_SIZE);

    const int kbx  = k / QI5_K; // == 0 if QK_K == 256
    const int kqsx = k % QI5_K; // == k if QK_K == 256

    const block_q5_K * bx0 = (const block_q5_K *) vx;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + i_offset;

        if (need_check) {
            i = sycl::min(i, i_max);
        }

        const block_q5_K * bxi = bx0 + i*blocks_per_row + kbx;
        const int ky = QR5_K*kqsx;

        const int ql = get_int_from_uint8_aligned(bxi->qs, kqsx);
        const int ql0 = (ql >> 0) & 0x0F0F0F0F;
        const int ql1 = (ql >> 4) & 0x0F0F0F0F;

        const int qh = get_int_from_uint8_aligned(bxi->qh, kqsx % (QI5_K/4));
        const int qh0 = ((qh >> (2 * (kqsx / (QI5_K/4)) + 0)) << 4) & 0x10101010;
        const int qh1 = ((qh >> (2 * (kqsx / (QI5_K/4)) + 1)) << 4) & 0x10101010;

        const int kq0 = ky - ky % (QI5_K/2) + k % (QI5_K/4) + 0;
        const int kq1 = ky - ky % (QI5_K/2) + k % (QI5_K/4) + (QI5_K/4);

        x_ql[i * (2*WARP_SIZE + 1) + kq0] = ql0 | qh0;
        x_ql[i * (2*WARP_SIZE + 1) + kq1] = ql1 | qh1;
    }

    constexpr int blocks_per_tile_x_row = QI5_K > WARP_SIZE ? 1 : WARP_SIZE / QI5_K; // == 1 if QK_K == 256
    const int kbxd = k % blocks_per_tile_x_row;          // == 0 if QK_K == 256

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI5_K) {
        int i = (i0 + i_offset * QI5_K + k / blocks_per_tile_x_row) % mmq_y;

        if (need_check) {
            i = sycl::min(i, i_max);
        }

        const block_q5_K * bxi = bx0 + i*blocks_per_row + kbxd;

#if QK_K == 256
        x_dm[i * (WARP_SIZE/QI5_K) + i / QI5_K + kbxd] = bxi->dm;
#endif
    }

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * 8) {
        int i = (i0 + i_offset * 8 + k / (WARP_SIZE/8)) % mmq_y;

        if (need_check) {
            i = sycl::min(i, i_max);
        }

        const block_q5_K * bxi = bx0 + i*blocks_per_row + (k % (WARP_SIZE/8)) / (QI5_K/8);

        const int * scales = (const int *) bxi->scales;

        const int ksc = k % (WARP_SIZE/8);

        // scale arrangement after the following two lines: sc0,...,sc3, sc4,...,sc7, m0,...,m3, m4,...,m8
        int scales8 = (scales[(ksc%2) + (ksc!=0)] >> (4 * (ksc & (ksc/2)))) & 0x0F0F0F0F; // lower 4 bits
        scales8    |= (scales[ksc/2]              >> (2 * (ksc % 2)))       & 0x30303030; // upper 2 bits

        x_sc[i * (WARP_SIZE/8) + i / 8 + ksc] = scales8;
    }
}

#define VDR_Q5_K_Q8_1_MMQ  8

// contiguous u/y values
static __dpct_inline__ float vec_dot_q5_K_q8_1_impl_mmq(
    const int *__restrict__ v, const int *__restrict__ u,
    const uint8_t *__restrict__ sc, const uint8_t *__restrict__ m,
    const sycl::half2 &dm4, const sycl::half2 *__restrict__ ds8) {

    float sumf_d = 0.0f;
    float sumf_m = 0.0f;

#pragma unroll
    for (int i = 0; i < QR5_K*VDR_Q5_K_Q8_1_MMQ/QI8_1; ++i) {
        int sumi_d = 0;

#pragma unroll
        for (int j = 0; j < QI8_1; ++j) {
            sumi_d = dpct::dp4a(v[i * QI8_1 + j], u[i * QI8_1 + j],
                                sumi_d); // SIMD dot product
        }

        const sycl::float2 ds8f =
            ds8[i].convert<float, sycl::rounding_mode::automatic>();

        sumf_d += ds8f.x() * (sc[i] * sumi_d);
        sumf_m += ds8f.y() * m[i]; // sum of q8_1 block * q4_K min val
    }

    const sycl::float2 dm4f =
        dm4.convert<float, sycl::rounding_mode::automatic>();

    return dm4f.x() * sumf_d - dm4f.y() * sumf_m;
}

static __dpct_inline__ float vec_dot_q5_K_q8_1_mul_mat(
    const int *__restrict__ x_ql, const sycl::half2 *__restrict__ x_dm,
    const int *__restrict__ x_qh, const int *__restrict__ x_sc,
    const int *__restrict__ y_qs, const sycl::half2 *__restrict__ y_ds,
    const int &i, const int &j, const int &k) {
    (void)x_qh;

    const uint8_t * sc = ((const uint8_t *) &x_sc[i * (WARP_SIZE/8) + i/8 + k/16]) + 2 * ((k % 16) / 8);

    const int index_x = i * (QR5_K*WARP_SIZE + 1) +  QR5_K*k;
    const int index_y = j * WARP_SIZE             + (QR5_K*k) % WARP_SIZE;
    return vec_dot_q5_K_q8_1_impl_mmq(&x_ql[index_x], &y_qs[index_y], sc, sc+8,
                                      x_dm[i * (WARP_SIZE/QI5_K) + i/QI5_K], &y_ds[index_y/QI8_1]);
}

template <int mmq_y>
static __dpct_inline__ void
allocate_tiles_q6_K(int **x_ql, sycl::half2 **x_dm, int **x_qh, int **x_sc,
                    int *tile_x_ql, sycl::half2 *tile_x_dm, int *tile_x_sc) {
    (void)x_qh;

    *x_ql = tile_x_ql;
    *x_dm = tile_x_dm;
    *x_sc = tile_x_sc;
}

template <int mmq_y, int nwarps, bool need_check>
static __dpct_inline__ void
load_tiles_q6_K(const void *__restrict__ vx, int *__restrict__ x_ql,
                sycl::half2 *__restrict__ x_dm, int *__restrict__ x_qh,
                int *__restrict__ x_sc, const int &i_offset, const int &i_max,
                const int &k, const int &blocks_per_row) {
    (void)x_qh;

    GGML_SYCL_ASSUME(i_offset >= 0);
    GGML_SYCL_ASSUME(i_offset <  nwarps);
    GGML_SYCL_ASSUME(k >= 0);
    GGML_SYCL_ASSUME(k <  WARP_SIZE);

    const int kbx  = k / QI6_K; // == 0 if QK_K == 256
    const int kqsx = k % QI6_K; // == k if QK_K == 256

    const block_q6_K * bx0 = (const block_q6_K *) vx;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + i_offset;

        if (need_check) {
            i = sycl::min(i, i_max);
        }

        const block_q6_K * bxi = bx0 + i*blocks_per_row + kbx;
        const int ky = QR6_K*kqsx;

        const int ql = get_int_from_uint8(bxi->ql, kqsx);
        const int ql0 = (ql >> 0) & 0x0F0F0F0F;
        const int ql1 = (ql >> 4) & 0x0F0F0F0F;

        const int qh = get_int_from_uint8(bxi->qh, (QI6_K/4) * (kqsx / (QI6_K/2)) + kqsx % (QI6_K/4));
        const int qh0 = ((qh >> (2 * ((kqsx % (QI6_K/2)) / (QI6_K/4)))) << 4) & 0x30303030;
        const int qh1 =  (qh >> (2 * ((kqsx % (QI6_K/2)) / (QI6_K/4))))       & 0x30303030;

        const int kq0 = ky - ky % QI6_K + k % (QI6_K/2) + 0;
        const int kq1 = ky - ky % QI6_K + k % (QI6_K/2) + (QI6_K/2);

        x_ql[i * (2 * WARP_SIZE + 1) + kq0] =
            dpct::vectorized_binary<sycl::char4>(ql0 | qh0, 0x20202020,
                                                 dpct::sub_sat());
        x_ql[i * (2 * WARP_SIZE + 1) + kq1] =
            dpct::vectorized_binary<sycl::char4>(ql1 | qh1, 0x20202020,
                                                 dpct::sub_sat());
    }

    constexpr int blocks_per_tile_x_row = QI6_K > WARP_SIZE ? 1 : WARP_SIZE / QI6_K; // == 1 if QK_K == 256
    const int kbxd = k % blocks_per_tile_x_row;          // == 0 if QK_K == 256
    float * x_dmf = (float *) x_dm;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI6_K) {
        int i = (i0 + i_offset * QI6_K + k / blocks_per_tile_x_row) % mmq_y;

        if (need_check) {
            i = sycl::min(i, i_max);
        }

        const block_q6_K * bxi = bx0 + i*blocks_per_row + kbxd;

        x_dmf[i * (WARP_SIZE/QI6_K) + i / QI6_K + kbxd] = bxi->d;
    }

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * 8) {
        int i = (i0 + i_offset * 8 + k / (WARP_SIZE/8)) % mmq_y;

        if (need_check) {
            i = sycl::min(i, i_max);
        }

        const block_q6_K * bxi = bx0 + i*blocks_per_row + (k % (WARP_SIZE/8)) / 4;

        x_sc[i * (WARP_SIZE/8) + i / 8 + k % (WARP_SIZE/8)] = get_int_from_int8(bxi->scales, k % (QI6_K/8));
    }
}

#define VDR_Q6_K_Q8_1_MMQ  8

// contiguous u/y values
static __dpct_inline__ float
vec_dot_q6_K_q8_1_impl_mmq(const int *__restrict__ v, const int *__restrict__ u,
                           const int8_t *__restrict__ sc, const float &d6,
                           const float *__restrict__ d8) {

    float sumf_d = 0.0f;

#pragma unroll
    for (int i0 = 0; i0 < VDR_Q6_K_Q8_1_MMQ; i0 += 4) {
        sycl::int2 sumi_d = {0, 0}; // 2 q6_K scales per q8_1 scale

#pragma unroll
        for (int i = i0; i < i0 + 2; ++i) {
            sumi_d.x() = dpct::dp4a(v[2 * i + 0], u[2 * i + 0],
                                    sumi_d.x()); // SIMD dot product
            sumi_d.x() = dpct::dp4a(v[2 * i + 1], u[2 * i + 1],
                                    sumi_d.x()); // SIMD dot product

            sumi_d.y() = dpct::dp4a(v[2 * i + 4], u[2 * i + 4],
                                    sumi_d.y()); // SIMD dot product
            sumi_d.y() = dpct::dp4a(v[2 * i + 5], u[2 * i + 5],
                                    sumi_d.y()); // SIMD dot product
        }

        sumf_d += d8[i0 / 4] *
                  (sc[i0 / 2 + 0] * sumi_d.x() + sc[i0 / 2 + 1] * sumi_d.y());
    }

    return d6 * sumf_d;
}

static __dpct_inline__ float vec_dot_q6_K_q8_1_mul_mat(
    const int *__restrict__ x_ql, const sycl::half2 *__restrict__ x_dm,
    const int *__restrict__ x_qh, const int *__restrict__ x_sc,
    const int *__restrict__ y_qs, const sycl::half2 *__restrict__ y_ds,
    const int &i, const int &j, const int &k) {
    (void)x_qh;

    const float * x_dmf = (const float *) x_dm;
    const float * y_df  = (const float *) y_ds;

    const int8_t * sc = ((const int8_t *) &x_sc[i * (WARP_SIZE/8) + i/8 + k/8]);

    const int index_x = i * (QR6_K*WARP_SIZE + 1) +  QR6_K*k;
    const int index_y = j * WARP_SIZE             + (QR6_K*k) % WARP_SIZE;
    return vec_dot_q6_K_q8_1_impl_mmq(&x_ql[index_x], &y_qs[index_y], sc, x_dmf[i * (WARP_SIZE/QI6_K) + i/QI6_K], &y_df[index_y/QI8_1]);
}

template <int qk, int qr, int qi, bool need_sum, typename block_q_t, int mmq_x,
          int mmq_y, int nwarps, load_tiles_sycl_t load_tiles, int vdr,
          vec_dot_q_mul_mat_sycl_t vec_dot>
/*
DPCT1110:8: The total declared local variable size in device function mul_mat_q
exceeds 128 bytes and may cause high register pressure. Consult with your
hardware vendor to find the total register size available and adjust the code,
or use smaller sub-group size to avoid high register pressure.
*/
static __dpct_inline__ void
mul_mat_q(const void *__restrict__ vx, const void *__restrict__ vy,
          float *__restrict__ dst, const int ncols_x, const int nrows_x,
          const int ncols_y, const int nrows_y, const int nrows_dst,
          int *tile_x_ql, sycl::half2 *tile_x_dm, int *tile_x_qh,
          int *tile_x_sc, const sycl::nd_item<3> &item_ct1, int *tile_y_qs,
          sycl::half2 *tile_y_ds) {

    const block_q_t  * x = (const block_q_t  *) vx;
    const block_q8_1 * y = (const block_q8_1 *) vy;

    const int blocks_per_row_x = ncols_x / qk;
    const int blocks_per_col_y = nrows_y / QK8_1;
    const int blocks_per_warp = WARP_SIZE / qi;

    const int & ncols_dst = ncols_y;

    const int row_dst_0 = item_ct1.get_group(2) * mmq_y;
    const int & row_x_0 = row_dst_0;

    const int col_dst_0 = item_ct1.get_group(1) * mmq_x;
    const int & col_y_0 = col_dst_0;

    float sum[mmq_y/WARP_SIZE][mmq_x/nwarps] = {{0.0f}};

    for (int ib0 = 0; ib0 < blocks_per_row_x; ib0 += blocks_per_warp) {

        load_tiles(x + row_x_0 * blocks_per_row_x + ib0, tile_x_ql, tile_x_dm,
                   tile_x_qh, tile_x_sc, item_ct1.get_local_id(1),
                   nrows_x - row_x_0 - 1, item_ct1.get_local_id(2),
                   blocks_per_row_x);

#pragma unroll
        for (int ir = 0; ir < qr; ++ir) {
            const int kqs = ir * WARP_SIZE + item_ct1.get_local_id(2);
            const int kbxd = kqs / QI8_1;

#pragma unroll
            for (int i = 0; i < mmq_x; i += nwarps) {
                const int col_y_eff = dpct::min(
                    (unsigned int)(col_y_0 + item_ct1.get_local_id(1) + i),
                    ncols_y - 1); // to prevent out-of-bounds memory accesses

                const block_q8_1 * by0 = &y[col_y_eff*blocks_per_col_y + ib0 * (qk/QK8_1) + kbxd];

                const int index_y = (item_ct1.get_local_id(1) + i) * WARP_SIZE +
                                    kqs % WARP_SIZE;
                tile_y_qs[index_y] = get_int_from_int8_aligned(
                    by0->qs, item_ct1.get_local_id(2) % QI8_1);
            }

#pragma unroll
            for (int ids0 = 0; ids0 < mmq_x; ids0 += nwarps * QI8_1) {
                const int ids =
                    (ids0 + item_ct1.get_local_id(1) * QI8_1 +
                     item_ct1.get_local_id(2) / (WARP_SIZE / QI8_1)) %
                    mmq_x;
                const int kby = item_ct1.get_local_id(2) % (WARP_SIZE / QI8_1);
                const int col_y_eff = sycl::min(col_y_0 + ids, ncols_y - 1);

                // if the sum is not needed it's faster to transform the scale to f32 ahead of time
                const sycl::half2 *dsi_src =
                    &y[col_y_eff * blocks_per_col_y + ib0 * (qk / QK8_1) +
                       ir * (WARP_SIZE / QI8_1) + kby]
                         .ds;
                sycl::half2 *dsi_dst =
                    &tile_y_ds[ids * (WARP_SIZE / QI8_1) + kby];
                if (need_sum) {
                    *dsi_dst = *dsi_src;
                } else {
                    float * dfi_dst = (float *) dsi_dst;
                    *dfi_dst = (*dsi_src)[0];
                }
            }

            /*
            DPCT1118:9: SYCL group functions and algorithms must be encountered
            in converged control flow. You may need to adjust the code.
            */
            /*
            DPCT1065:56: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            item_ct1.barrier();

// #pragma unroll // unrolling this loop causes too much register pressure
            for (int k = ir*WARP_SIZE/qr; k < (ir+1)*WARP_SIZE/qr; k += vdr) {
#pragma unroll
                for (int j = 0; j < mmq_x; j += nwarps) {
#pragma unroll
                    for (int i = 0; i < mmq_y; i += WARP_SIZE) {
                        sum[i / WARP_SIZE][j / nwarps] += vec_dot(
                            tile_x_ql, tile_x_dm, tile_x_qh, tile_x_sc,
                            tile_y_qs, tile_y_ds, item_ct1.get_local_id(2) + i,
                            item_ct1.get_local_id(1) + j, k);
                    }
                }
            }

            /*
            DPCT1118:10: SYCL group functions and algorithms must be encountered
            in converged control flow. You may need to adjust the code.
            */
            /*
            DPCT1065:57: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            item_ct1.barrier();
        }
    }

#pragma unroll
    for (int j = 0; j < mmq_x; j += nwarps) {
        const int col_dst = col_dst_0 + j + item_ct1.get_local_id(1);

        if (col_dst >= ncols_dst) {
            return;
        }

#pragma unroll
        for (int i = 0; i < mmq_y; i += WARP_SIZE) {
            const int row_dst = row_dst_0 + item_ct1.get_local_id(2) + i;

            if (row_dst >= nrows_dst) {
                continue;
            }

            dst[col_dst*nrows_dst + row_dst] = sum[i/WARP_SIZE][j/nwarps];
        }
    }
}

#define  MMQ_X_Q4_0_RDNA2  64
#define  MMQ_Y_Q4_0_RDNA2  128
#define NWARPS_Q4_0_RDNA2  8
#define  MMQ_X_Q4_0_RDNA1  64
#define  MMQ_Y_Q4_0_RDNA1  64
#define NWARPS_Q4_0_RDNA1  8
#if defined(SYCL_USE_XMX)
#define  MMQ_X_Q4_0_AMPERE 4
#define  MMQ_Y_Q4_0_AMPERE 32
#define NWARPS_Q4_0_AMPERE 4
#else
#define  MMQ_X_Q4_0_AMPERE 64
#define  MMQ_Y_Q4_0_AMPERE 128
#define NWARPS_Q4_0_AMPERE 4
#endif
#define  MMQ_X_Q4_0_PASCAL 64
#define  MMQ_Y_Q4_0_PASCAL 64
#define NWARPS_Q4_0_PASCAL 8

template <bool need_check> static void
    mul_mat_q4_0(
    const void * __restrict__ vx, const void * __restrict__ vy, float * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int ncols_y, const int nrows_y, const int nrows_dst,
    const sycl::nd_item<3> &item_ct1, int *tile_x_qs_q4_0, float *tile_x_d_q4_0,
    int *tile_y_qs, sycl::half2 *tile_y_ds) {
    int   * tile_x_ql = nullptr;
    sycl::half2 *tile_x_dm = nullptr;
    int   * tile_x_qh = nullptr;
    int   * tile_x_sc = nullptr;

//sycl_todo: change according to hardware

    const int mmq_x  =  MMQ_X_Q4_0_AMPERE;
    const int mmq_y  =  MMQ_Y_Q4_0_AMPERE;
    const int nwarps = NWARPS_Q4_0_AMPERE;
    allocate_tiles_q4_0<mmq_y>(&tile_x_ql, &tile_x_dm, &tile_x_qh, &tile_x_sc,
                               tile_x_qs_q4_0, tile_x_d_q4_0);
    mul_mat_q<QK4_0, QR4_0, QI4_0, true, block_q4_0, mmq_x, mmq_y, nwarps,
              load_tiles_q4_0<mmq_y, nwarps, need_check>, VDR_Q4_0_Q8_1_MMQ,
              vec_dot_q4_0_q8_1_mul_mat>(
        vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, tile_x_ql,
        tile_x_dm, tile_x_qh, tile_x_sc, item_ct1, tile_y_qs, tile_y_ds);
}

#define  MMQ_X_Q4_1_RDNA2  64
#define  MMQ_Y_Q4_1_RDNA2  128
#define NWARPS_Q4_1_RDNA2  8
#define  MMQ_X_Q4_1_RDNA1  64
#define  MMQ_Y_Q4_1_RDNA1  64
#define NWARPS_Q4_1_RDNA1  8
#if defined(SYCL_USE_XMX)
#define  MMQ_X_Q4_1_AMPERE 4
#define  MMQ_Y_Q4_1_AMPERE 32
#define NWARPS_Q4_1_AMPERE 4
#else
#define  MMQ_X_Q4_1_AMPERE 64
#define  MMQ_Y_Q4_1_AMPERE 128
#define NWARPS_Q4_1_AMPERE 4
#endif
#define  MMQ_X_Q4_1_PASCAL 64
#define  MMQ_Y_Q4_1_PASCAL 64
#define NWARPS_Q4_1_PASCAL 8

template <bool need_check> static void
    mul_mat_q4_1(
    const void * __restrict__ vx, const void * __restrict__ vy, float * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int ncols_y, const int nrows_y, const int nrows_dst,
    const sycl::nd_item<3> &item_ct1, int *tile_x_qs_q4_1,
    sycl::half2 *tile_x_dm_q4_1, int *tile_y_qs, sycl::half2 *tile_y_ds) {
    int   * tile_x_ql = nullptr;
    sycl::half2 *tile_x_dm = nullptr;
    int   * tile_x_qh = nullptr;
    int   * tile_x_sc = nullptr;

//sycl_todo: change according to hardware
    const int mmq_x  =  MMQ_X_Q4_1_AMPERE;
    const int mmq_y  =  MMQ_Y_Q4_1_AMPERE;
    const int nwarps = NWARPS_Q4_1_AMPERE;
    allocate_tiles_q4_1<mmq_y>(&tile_x_ql, &tile_x_dm, &tile_x_qh, &tile_x_sc,
                               tile_x_qs_q4_1, tile_x_dm_q4_1);
    mul_mat_q<QK4_1, QR4_1, QI4_1, true, block_q4_1, mmq_x, mmq_y, nwarps,
              load_tiles_q4_1<mmq_y, nwarps, need_check>, VDR_Q4_1_Q8_1_MMQ,
              vec_dot_q4_1_q8_1_mul_mat>(
        vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, tile_x_ql,
        tile_x_dm, tile_x_qh, tile_x_sc, item_ct1, tile_y_qs, tile_y_ds);
}

#define  MMQ_X_Q5_0_RDNA2  64
#define  MMQ_Y_Q5_0_RDNA2  128
#define NWARPS_Q5_0_RDNA2  8
#define  MMQ_X_Q5_0_RDNA1  64
#define  MMQ_Y_Q5_0_RDNA1  64
#define NWARPS_Q5_0_RDNA1  8
#if defined(SYCL_USE_XMX)
#define  MMQ_X_Q5_0_AMPERE 4
#define  MMQ_Y_Q5_0_AMPERE 32
#define NWARPS_Q5_0_AMPERE 4
#else
#define  MMQ_X_Q5_0_AMPERE 128
#define  MMQ_Y_Q5_0_AMPERE 64
#define NWARPS_Q5_0_AMPERE 4
#endif
#define  MMQ_X_Q5_0_PASCAL 64
#define  MMQ_Y_Q5_0_PASCAL 64
#define NWARPS_Q5_0_PASCAL 8

template <bool need_check> static void
    mul_mat_q5_0(
    const void * __restrict__ vx, const void * __restrict__ vy, float * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int ncols_y, const int nrows_y, const int nrows_dst,
    const sycl::nd_item<3> &item_ct1, int *tile_x_ql_q5_0, float *tile_x_d_q5_0,
    int *tile_y_qs, sycl::half2 *tile_y_ds) {
    int   * tile_x_ql = nullptr;
    sycl::half2 *tile_x_dm = nullptr;
    int   * tile_x_qh = nullptr;
    int   * tile_x_sc = nullptr;

//sycl_todo: change according to hardware
    const int mmq_x  =  MMQ_X_Q5_0_AMPERE;
    const int mmq_y  =  MMQ_Y_Q5_0_AMPERE;
    const int nwarps = NWARPS_Q5_0_AMPERE;
    allocate_tiles_q5_0<mmq_y>(&tile_x_ql, &tile_x_dm, &tile_x_qh, &tile_x_sc,
                               tile_x_ql_q5_0, tile_x_d_q5_0);
    mul_mat_q<QK5_0, QR5_0, QI5_0, false, block_q5_0, mmq_x, mmq_y, nwarps,
              load_tiles_q5_0<mmq_y, nwarps, need_check>, VDR_Q5_0_Q8_1_MMQ,
              vec_dot_q5_0_q8_1_mul_mat>(
        vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, tile_x_ql,
        tile_x_dm, tile_x_qh, tile_x_sc, item_ct1, tile_y_qs, tile_y_ds);
}

#define  MMQ_X_Q5_1_RDNA2  64
#define  MMQ_Y_Q5_1_RDNA2  128
#define NWARPS_Q5_1_RDNA2  8
#define  MMQ_X_Q5_1_RDNA1  64
#define  MMQ_Y_Q5_1_RDNA1  64
#define NWARPS_Q5_1_RDNA1  8
#if defined(SYCL_USE_XMX)
#define  MMQ_X_Q5_1_AMPERE 4
#define  MMQ_Y_Q5_1_AMPERE 32
#define NWARPS_Q5_1_AMPERE 4
#else
#define  MMQ_X_Q5_1_AMPERE 128
#define  MMQ_Y_Q5_1_AMPERE 64
#define NWARPS_Q5_1_AMPERE 4
#endif
#define  MMQ_X_Q5_1_PASCAL 64
#define  MMQ_Y_Q5_1_PASCAL 64
#define NWARPS_Q5_1_PASCAL 8

template <bool need_check> static void
mul_mat_q5_1(
    const void * __restrict__ vx, const void * __restrict__ vy, float * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int ncols_y, const int nrows_y, const int nrows_dst,
    const sycl::nd_item<3> &item_ct1, int *tile_x_ql_q5_1,
    sycl::half2 *tile_x_dm_q5_1, int *tile_y_qs, sycl::half2 *tile_y_ds) {
    int   * tile_x_ql = nullptr;
    sycl::half2 *tile_x_dm = nullptr;
    int   * tile_x_qh = nullptr;
    int   * tile_x_sc = nullptr;

//sycl_todo: change according to hardware
    const int mmq_x  =  MMQ_X_Q5_1_AMPERE;
    const int mmq_y  =  MMQ_Y_Q5_1_AMPERE;
    const int nwarps = NWARPS_Q5_1_AMPERE;
    allocate_tiles_q5_1<mmq_y>(&tile_x_ql, &tile_x_dm, &tile_x_qh, &tile_x_sc,
                               tile_x_ql_q5_1, tile_x_dm_q5_1);
    mul_mat_q<QK5_1, QR5_1, QI5_1, true, block_q5_1, mmq_x, mmq_y, nwarps,
              load_tiles_q5_1<mmq_y, nwarps, need_check>, VDR_Q5_1_Q8_1_MMQ,
              vec_dot_q5_1_q8_1_mul_mat>(
        vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, tile_x_ql,
        tile_x_dm, tile_x_qh, tile_x_sc, item_ct1, tile_y_qs, tile_y_ds);
}

#define  MMQ_X_Q8_0_RDNA2  64
#define  MMQ_Y_Q8_0_RDNA2  128
#define NWARPS_Q8_0_RDNA2  8
#define  MMQ_X_Q8_0_RDNA1  64
#define  MMQ_Y_Q8_0_RDNA1  64
#define NWARPS_Q8_0_RDNA1  8
#if defined(SYCL_USE_XMX)
#define  MMQ_X_Q8_0_AMPERE 4
#define  MMQ_Y_Q8_0_AMPERE 32
#define NWARPS_Q8_0_AMPERE 4
#else
#define  MMQ_X_Q8_0_AMPERE 128
#define  MMQ_Y_Q8_0_AMPERE 64
#define NWARPS_Q8_0_AMPERE 4
#endif
#define  MMQ_X_Q8_0_PASCAL 64
#define  MMQ_Y_Q8_0_PASCAL 64
#define NWARPS_Q8_0_PASCAL 8

template <bool need_check> static void
    mul_mat_q8_0(
    const void * __restrict__ vx, const void * __restrict__ vy, float * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int ncols_y, const int nrows_y, const int nrows_dst,
    const sycl::nd_item<3> &item_ct1, int *tile_x_qs_q8_0, float *tile_x_d_q8_0,
    int *tile_y_qs, sycl::half2 *tile_y_ds) {
    int   * tile_x_ql = nullptr;
    sycl::half2 *tile_x_dm = nullptr;
    int   * tile_x_qh = nullptr;
    int   * tile_x_sc = nullptr;

//sycl_todo: change according to hardware
    const int mmq_x  =  MMQ_X_Q8_0_AMPERE;
    const int mmq_y  =  MMQ_Y_Q8_0_AMPERE;
    const int nwarps = NWARPS_Q8_0_AMPERE;
    allocate_tiles_q8_0<mmq_y>(&tile_x_ql, &tile_x_dm, &tile_x_qh, &tile_x_sc,
                               tile_x_qs_q8_0, tile_x_d_q8_0);
    mul_mat_q<QK8_0, QR8_0, QI8_0, false, block_q8_0, mmq_x, mmq_y, nwarps,
              load_tiles_q8_0<mmq_y, nwarps, need_check>, VDR_Q8_0_Q8_1_MMQ,
              vec_dot_q8_0_q8_1_mul_mat>(
        vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, tile_x_ql,
        tile_x_dm, tile_x_qh, tile_x_sc, item_ct1, tile_y_qs, tile_y_ds);
}

#define  MMQ_X_Q2_K_RDNA2  64
#define  MMQ_Y_Q2_K_RDNA2  128
#define NWARPS_Q2_K_RDNA2  8
#define  MMQ_X_Q2_K_RDNA1  128
#define  MMQ_Y_Q2_K_RDNA1  32
#define NWARPS_Q2_K_RDNA1  8
#if defined(SYCL_USE_XMX)
#define  MMQ_X_Q2_K_AMPERE 4
#define  MMQ_Y_Q2_K_AMPERE 32
#define NWARPS_Q2_K_AMPERE 4
#else
#define  MMQ_X_Q2_K_AMPERE 64
#define  MMQ_Y_Q2_K_AMPERE 128
#define NWARPS_Q2_K_AMPERE 4
#endif
#define  MMQ_X_Q2_K_PASCAL 64
#define  MMQ_Y_Q2_K_PASCAL 64
#define NWARPS_Q2_K_PASCAL 8

template <bool need_check> static void
mul_mat_q2_K(
    const void * __restrict__ vx, const void * __restrict__ vy, float * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int ncols_y, const int nrows_y, const int nrows_dst,
    const sycl::nd_item<3> &item_ct1, int *tile_x_ql_q2_K,
    sycl::half2 *tile_x_dm_q2_K, int *tile_x_sc_q2_K, int *tile_y_qs,
    sycl::half2 *tile_y_ds) {
    int   * tile_x_ql = nullptr;
    sycl::half2 *tile_x_dm = nullptr;
    int   * tile_x_qh = nullptr;
    int   * tile_x_sc = nullptr;

//sycl_todo: change according to hardware
    const int mmq_x  =  MMQ_X_Q2_K_AMPERE;
    const int mmq_y  =  MMQ_Y_Q2_K_AMPERE;
    const int nwarps = NWARPS_Q2_K_AMPERE;
    allocate_tiles_q2_K<mmq_y>(&tile_x_ql, &tile_x_dm, &tile_x_qh, &tile_x_sc,
                               tile_x_ql_q2_K, tile_x_dm_q2_K, tile_x_sc_q2_K);
    mul_mat_q<QK_K, QR2_K, QI2_K, false, block_q2_K, mmq_x, mmq_y, nwarps,
              load_tiles_q2_K<mmq_y, nwarps, need_check>, VDR_Q2_K_Q8_1_MMQ,
              vec_dot_q2_K_q8_1_mul_mat>(
        vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, tile_x_ql,
        tile_x_dm, tile_x_qh, tile_x_sc, item_ct1, tile_y_qs, tile_y_ds);
}

#define  MMQ_X_Q3_K_RDNA2  128
#define  MMQ_Y_Q3_K_RDNA2  64
#define NWARPS_Q3_K_RDNA2  8
#define  MMQ_X_Q3_K_RDNA1  32
#define  MMQ_Y_Q3_K_RDNA1  128
#define NWARPS_Q3_K_RDNA1  8
#if defined(SYCL_USE_XMX)
#define  MMQ_X_Q3_K_AMPERE 4
#define  MMQ_Y_Q3_K_AMPERE 32
#define NWARPS_Q3_K_AMPERE 4
#else
#define  MMQ_X_Q3_K_AMPERE 128
#define  MMQ_Y_Q3_K_AMPERE 128
#define NWARPS_Q3_K_AMPERE 4
#endif
#define  MMQ_X_Q3_K_PASCAL 64
#define  MMQ_Y_Q3_K_PASCAL 64
#define NWARPS_Q3_K_PASCAL 8

template <bool need_check> static void
mul_mat_q3_K(
    const void * __restrict__ vx, const void * __restrict__ vy, float * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int ncols_y, const int nrows_y, const int nrows_dst,
    const sycl::nd_item<3> &item_ct1, int *tile_x_ql_q3_K,
    sycl::half2 *tile_x_dm_q3_K, int *tile_x_qh_q3_K, int *tile_x_sc_q3_K,
    int *tile_y_qs, sycl::half2 *tile_y_ds) {
    int   * tile_x_ql = nullptr;
    sycl::half2 *tile_x_dm = nullptr;
    int   * tile_x_qh = nullptr;
    int   * tile_x_sc = nullptr;

//sycl_todo: change according to hardware
    const int mmq_x  =  MMQ_X_Q3_K_AMPERE;
    const int mmq_y  =  MMQ_Y_Q3_K_AMPERE;
    const int nwarps = NWARPS_Q3_K_AMPERE;
    allocate_tiles_q3_K<mmq_y>(&tile_x_ql, &tile_x_dm, &tile_x_qh, &tile_x_sc,
                               tile_x_ql_q3_K, tile_x_dm_q3_K, tile_x_qh_q3_K,
                               tile_x_sc_q3_K);
    mul_mat_q<QK_K, QR3_K, QI3_K, false, block_q3_K, mmq_x, mmq_y, nwarps,
              load_tiles_q3_K<mmq_y, nwarps, need_check>, VDR_Q3_K_Q8_1_MMQ,
              vec_dot_q3_K_q8_1_mul_mat>(
        vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, tile_x_ql,
        tile_x_dm, tile_x_qh, tile_x_sc, item_ct1, tile_y_qs, tile_y_ds);
}

#define  MMQ_X_Q4_K_RDNA2  64
#define  MMQ_Y_Q4_K_RDNA2  128
#define NWARPS_Q4_K_RDNA2  8
#define  MMQ_X_Q4_K_RDNA1  32
#define  MMQ_Y_Q4_K_RDNA1  64
#define NWARPS_Q4_K_RDNA1  8
#if defined(SYCL_USE_XMX)
#define  MMQ_X_Q4_K_AMPERE 4
#define  MMQ_Y_Q4_K_AMPERE 32
#define NWARPS_Q4_K_AMPERE 4
#else
#define  MMQ_X_Q4_K_AMPERE 64
#define  MMQ_Y_Q4_K_AMPERE 128
#define NWARPS_Q4_K_AMPERE 4
#endif
#define  MMQ_X_Q4_K_PASCAL 64
#define  MMQ_Y_Q4_K_PASCAL 64
#define NWARPS_Q4_K_PASCAL 8

template <bool need_check> static void
    mul_mat_q4_K(
    const void * __restrict__ vx, const void * __restrict__ vy, float * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int ncols_y, const int nrows_y, const int nrows_dst,
    const sycl::nd_item<3> &item_ct1, int *tile_x_ql_q4_K,
    sycl::half2 *tile_x_dm_q4_K, int *tile_x_sc_q4_K, int *tile_y_qs,
    sycl::half2 *tile_y_ds) {
    int   * tile_x_ql = nullptr;
    sycl::half2 *tile_x_dm = nullptr;
    int   * tile_x_qh = nullptr;
    int   * tile_x_sc = nullptr;

//sycl_todo: change according to hardware
    const int mmq_x  =  MMQ_X_Q4_K_AMPERE;
    const int mmq_y  =  MMQ_Y_Q4_K_AMPERE;
    const int nwarps = NWARPS_Q4_K_AMPERE;
    allocate_tiles_q4_K<mmq_y>(&tile_x_ql, &tile_x_dm, &tile_x_qh, &tile_x_sc,
                               tile_x_ql_q4_K, tile_x_dm_q4_K, tile_x_sc_q4_K);
    mul_mat_q<QK_K, QR4_K, QI4_K, true, block_q4_K, mmq_x, mmq_y, nwarps,
              load_tiles_q4_K<mmq_y, nwarps, need_check>, VDR_Q4_K_Q8_1_MMQ,
              vec_dot_q4_K_q8_1_mul_mat>(
        vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, tile_x_ql,
        tile_x_dm, tile_x_qh, tile_x_sc, item_ct1, tile_y_qs, tile_y_ds);
}

#define  MMQ_X_Q5_K_RDNA2  64
#define  MMQ_Y_Q5_K_RDNA2  128
#define NWARPS_Q5_K_RDNA2  8
#define  MMQ_X_Q5_K_RDNA1  32
#define  MMQ_Y_Q5_K_RDNA1  64
#define NWARPS_Q5_K_RDNA1  8
#if defined(SYCL_USE_XMX)
#define  MMQ_X_Q5_K_AMPERE 4
#define  MMQ_Y_Q5_K_AMPERE 32
#define NWARPS_Q5_K_AMPERE 4
#else
#define  MMQ_X_Q5_K_AMPERE 64
#define  MMQ_Y_Q5_K_AMPERE 128
#define NWARPS_Q5_K_AMPERE 4
#endif
#define  MMQ_X_Q5_K_PASCAL 64
#define  MMQ_Y_Q5_K_PASCAL 64
#define NWARPS_Q5_K_PASCAL 8

template <bool need_check> static void
mul_mat_q5_K(
    const void * __restrict__ vx, const void * __restrict__ vy, float * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int ncols_y, const int nrows_y, const int nrows_dst,
    const sycl::nd_item<3> &item_ct1, int *tile_x_ql_q5_K,
    sycl::half2 *tile_x_dm_q5_K, int *tile_x_sc_q5_K, int *tile_y_qs,
    sycl::half2 *tile_y_ds) {
    int   * tile_x_ql = nullptr;
    sycl::half2 *tile_x_dm = nullptr;
    int   * tile_x_qh = nullptr;
    int   * tile_x_sc = nullptr;

//sycl_todo: change according to hardware
    const int mmq_x  =  MMQ_X_Q5_K_AMPERE;
    const int mmq_y  =  MMQ_Y_Q5_K_AMPERE;
    const int nwarps = NWARPS_Q5_K_AMPERE;
    allocate_tiles_q5_K<mmq_y>(&tile_x_ql, &tile_x_dm, &tile_x_qh, &tile_x_sc,
                               tile_x_ql_q5_K, tile_x_dm_q5_K, tile_x_sc_q5_K);
    mul_mat_q<QK_K, QR5_K, QI5_K, true, block_q5_K, mmq_x, mmq_y, nwarps,
              load_tiles_q5_K<mmq_y, nwarps, need_check>, VDR_Q5_K_Q8_1_MMQ,
              vec_dot_q5_K_q8_1_mul_mat>(
        vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, tile_x_ql,
        tile_x_dm, tile_x_qh, tile_x_sc, item_ct1, tile_y_qs, tile_y_ds);
}

#define  MMQ_X_Q6_K_RDNA2  64
#define  MMQ_Y_Q6_K_RDNA2  128
#define NWARPS_Q6_K_RDNA2  8
#define  MMQ_X_Q6_K_RDNA1  32
#define  MMQ_Y_Q6_K_RDNA1  64
#define NWARPS_Q6_K_RDNA1  8
#if defined(SYCL_USE_XMX)
#define  MMQ_X_Q6_K_AMPERE 4
#define  MMQ_Y_Q6_K_AMPERE 32
#define NWARPS_Q6_K_AMPERE 4
#else
#define  MMQ_X_Q6_K_AMPERE 64
#define  MMQ_Y_Q6_K_AMPERE 64
#define NWARPS_Q6_K_AMPERE 4
#endif
#define  MMQ_X_Q6_K_PASCAL 64
#define  MMQ_Y_Q6_K_PASCAL 64
#define NWARPS_Q6_K_PASCAL 8

template <bool need_check> static void
    mul_mat_q6_K(
    const void * __restrict__ vx, const void * __restrict__ vy, float * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int ncols_y, const int nrows_y, const int nrows_dst,
    const sycl::nd_item<3> &item_ct1, int *tile_x_ql, sycl::half2 *tile_x_dm,
    int *tile_x_sc, int *tile_y_qs, sycl::half2 *tile_y_ds) {
    // int   * tile_x_ql = nullptr;
    // sycl::half2 *tile_x_dm = nullptr;
    int   * tile_x_qh = nullptr;
    // int   * tile_x_sc = nullptr;

//sycl_todo: change according to hardware
    const int mmq_x  =  MMQ_X_Q6_K_AMPERE;
    const int mmq_y  =  MMQ_Y_Q6_K_AMPERE;
    const int nwarps = NWARPS_Q6_K_AMPERE;
    allocate_tiles_q6_K<mmq_y>(&tile_x_ql, &tile_x_dm, &tile_x_qh, &tile_x_sc,
                               tile_x_ql, tile_x_dm, tile_x_sc);
    mul_mat_q<QK_K, QR6_K, QI6_K, false, block_q6_K, mmq_x, mmq_y, nwarps,
              load_tiles_q6_K<mmq_y, nwarps, need_check>, VDR_Q6_K_Q8_1_MMQ,
              vec_dot_q6_K_q8_1_mul_mat>(
        vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, tile_x_ql,
        tile_x_dm, tile_x_qh, tile_x_sc, item_ct1, tile_y_qs, tile_y_ds);
}

static void ggml_mul_mat_q4_0_q8_1_sycl(const void *vx, const void *vy,
                                        float *dst, const int ncols_x,
                                        const int nrows_x, const int ncols_y,
                                        const int nrows_y, const int nrows_dst,
                                        dpct::queue_ptr stream) try {

    int id;
    SYCL_CHECK(
        CHECK_TRY_ERROR(id = get_current_device_id()));
    const int compute_capability = ggml_sycl_info().devices[id].cc;

    int mmq_x, mmq_y, nwarps;
    if (compute_capability >= VER_GEN13) {
        mmq_x  =  MMQ_X_Q4_0_RDNA2;
        mmq_y  =  MMQ_Y_Q4_0_RDNA2;
        nwarps = NWARPS_Q4_0_RDNA2;
    } else if (compute_capability >= VER_GEN12) {
        mmq_x  =  MMQ_X_Q4_0_RDNA1;
        mmq_y  =  MMQ_Y_Q4_0_RDNA1;
        nwarps = NWARPS_Q4_0_RDNA1;
    } else if (compute_capability >= VER_GEN9) {
        mmq_x  =  MMQ_X_Q4_0_AMPERE;
        mmq_y  =  MMQ_Y_Q4_0_AMPERE;
        nwarps = NWARPS_Q4_0_AMPERE;
    } else if (compute_capability >= VER_4VEC) {
        mmq_x  =  MMQ_X_Q4_0_PASCAL;
        mmq_y  =  MMQ_Y_Q4_0_PASCAL;
        nwarps = NWARPS_Q4_0_PASCAL;
    } else {
        GGML_ABORT("fatal error");
    }

    const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
    const int block_num_y = (ncols_y + mmq_x - 1) / mmq_x;
    const sycl::range<3> block_nums(1, block_num_y, block_num_x);
    const sycl::range<3> block_dims(1, nwarps, WARP_SIZE);

    if (nrows_x % mmq_y == 0) {
        const bool need_check = false;
        /*
        DPCT1049:20: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        {
            dpct::has_capability_or_fail(stream->get_device(),
                                         {sycl::aspect::fp16});

            stream->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<int, 1> tile_x_qs_q4_0_acc_ct1(
                    sycl::range<1>(mmq_y * (WARP_SIZE) + mmq_y), cgh);
                sycl::local_accessor<float, 1> tile_x_d_q4_0_acc_ct1(
                    sycl::range<1>(mmq_y * (WARP_SIZE / QI4_0) + mmq_y / QI4_0),
                    cgh);
                sycl::local_accessor<int, 1> tile_y_qs_acc_ct1(
                    sycl::range<1>(mmq_x * WARP_SIZE), cgh);
                sycl::local_accessor<sycl::half2, 1> tile_y_ds_acc_ct1(
                    sycl::range<1>(mmq_x * WARP_SIZE / QI8_1), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(block_nums * block_dims, block_dims),
                    [=](sycl::nd_item<3> item_ct1) {
                        mul_mat_q4_0<need_check>(
                            vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y,
                            nrows_dst, item_ct1,
                            get_pointer(tile_x_qs_q4_0_acc_ct1),
                            get_pointer(tile_x_d_q4_0_acc_ct1),
                            get_pointer(tile_y_qs_acc_ct1),
                            get_pointer(tile_y_ds_acc_ct1));
                    });
            });
        }
    } else {
        const bool need_check = true;
        /*
        DPCT1049:21: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        {
            dpct::has_capability_or_fail(stream->get_device(),
                                         {sycl::aspect::fp16});

            stream->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<int, 1> tile_x_qs_q4_0_acc_ct1(
                    sycl::range<1>(mmq_y * (WARP_SIZE) + mmq_y), cgh);
                sycl::local_accessor<float, 1> tile_x_d_q4_0_acc_ct1(
                    sycl::range<1>(mmq_y * (WARP_SIZE / QI4_0) + mmq_y / QI4_0),
                    cgh);
                sycl::local_accessor<int, 1> tile_y_qs_acc_ct1(
                    sycl::range<1>(mmq_x * WARP_SIZE), cgh);
                sycl::local_accessor<sycl::half2, 1> tile_y_ds_acc_ct1(
                    sycl::range<1>(mmq_x * WARP_SIZE / QI8_1), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(block_nums * block_dims, block_dims),
                    [=](sycl::nd_item<3> item_ct1) {
                        mul_mat_q4_0<need_check>(
                            vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y,
                            nrows_dst, item_ct1,
                            get_pointer(tile_x_qs_q4_0_acc_ct1),
                            get_pointer(tile_x_d_q4_0_acc_ct1),
                            get_pointer(tile_y_qs_acc_ct1),
                            get_pointer(tile_y_ds_acc_ct1));
                    });
            });
        }
    }
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static void ggml_mul_mat_q4_1_q8_1_sycl(const void *vx, const void *vy,
                                        float *dst, const int ncols_x,
                                        const int nrows_x, const int ncols_y,
                                        const int nrows_y, const int nrows_dst,
                                        dpct::queue_ptr stream) try {

    int id;
    SYCL_CHECK(
        CHECK_TRY_ERROR(id = get_current_device_id()));
    const int compute_capability = ggml_sycl_info().devices[id].cc;

    int mmq_x, mmq_y, nwarps;
    if (compute_capability >= VER_GEN13) {
        mmq_x  =  MMQ_X_Q4_1_RDNA2;
        mmq_y  =  MMQ_Y_Q4_1_RDNA2;
        nwarps = NWARPS_Q4_1_RDNA2;
    } else if (compute_capability >= VER_GEN12) {
        mmq_x  =  MMQ_X_Q4_1_RDNA1;
        mmq_y  =  MMQ_Y_Q4_1_RDNA1;
        nwarps = NWARPS_Q4_1_RDNA1;
    } else if (compute_capability >= VER_GEN9) {
        mmq_x  =  MMQ_X_Q4_1_AMPERE;
        mmq_y  =  MMQ_Y_Q4_1_AMPERE;
        nwarps = NWARPS_Q4_1_AMPERE;
    } else if (compute_capability >= VER_4VEC) {
        mmq_x  =  MMQ_X_Q4_1_PASCAL;
        mmq_y  =  MMQ_Y_Q4_1_PASCAL;
        nwarps = NWARPS_Q4_1_PASCAL;
    } else {
        GGML_ABORT("fatal error");
    }

    const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
    const int block_num_y = (ncols_y + mmq_x - 1) / mmq_x;
    const sycl::range<3> block_nums(1, block_num_y, block_num_x);
    const sycl::range<3> block_dims(1, nwarps, WARP_SIZE);

    if (nrows_x % mmq_y == 0) {
        const bool need_check = false;
        /*
        DPCT1049:22: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        {
            dpct::has_capability_or_fail(stream->get_device(),
                                         {sycl::aspect::fp16});

            stream->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<int, 1> tile_x_qs_q4_1_acc_ct1(
                    sycl::range<1>(mmq_y * (WARP_SIZE) + +mmq_y), cgh);
                sycl::local_accessor<sycl::half2, 1> tile_x_dm_q4_1_acc_ct1(
                    sycl::range<1>(mmq_y * (WARP_SIZE / QI4_1) + mmq_y / QI4_1),
                    cgh);
                sycl::local_accessor<int, 1> tile_y_qs_acc_ct1(
                    sycl::range<1>(mmq_x * WARP_SIZE), cgh);
                sycl::local_accessor<sycl::half2, 1> tile_y_ds_acc_ct1(
                    sycl::range<1>(mmq_x * WARP_SIZE / QI8_1), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(block_nums * block_dims, block_dims),
                    [=](sycl::nd_item<3> item_ct1) {
                        mul_mat_q4_1<need_check>(
                            vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y,
                            nrows_dst, item_ct1,
                            get_pointer(tile_x_qs_q4_1_acc_ct1),
                            get_pointer(tile_x_dm_q4_1_acc_ct1),
                            get_pointer(tile_y_qs_acc_ct1),
                            get_pointer(tile_y_ds_acc_ct1));
                    });
            });
        }
    } else {
        const bool need_check = true;
        /*
        DPCT1049:23: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        {
            dpct::has_capability_or_fail(stream->get_device(),
                                         {sycl::aspect::fp16});

            stream->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<int, 1> tile_x_qs_q4_1_acc_ct1(
                    sycl::range<1>(mmq_y * (WARP_SIZE) + +mmq_y), cgh);
                sycl::local_accessor<sycl::half2, 1> tile_x_dm_q4_1_acc_ct1(
                    sycl::range<1>(mmq_y * (WARP_SIZE / QI4_1) + mmq_y / QI4_1),
                    cgh);
                sycl::local_accessor<int, 1> tile_y_qs_acc_ct1(
                    sycl::range<1>(mmq_x * WARP_SIZE), cgh);
                sycl::local_accessor<sycl::half2, 1> tile_y_ds_acc_ct1(
                    sycl::range<1>(mmq_x * WARP_SIZE / QI8_1), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(block_nums * block_dims, block_dims),
                    [=](sycl::nd_item<3> item_ct1) {
                        mul_mat_q4_1<need_check>(
                            vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y,
                            nrows_dst, item_ct1,
                            get_pointer(tile_x_qs_q4_1_acc_ct1),
                            get_pointer(tile_x_dm_q4_1_acc_ct1),
                            get_pointer(tile_y_qs_acc_ct1),
                            get_pointer(tile_y_ds_acc_ct1));
                    });
            });
        }
    }
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static void ggml_mul_mat_q5_0_q8_1_sycl(const void *vx, const void *vy,
                                        float *dst, const int ncols_x,
                                        const int nrows_x, const int ncols_y,
                                        const int nrows_y, const int nrows_dst,
                                        dpct::queue_ptr stream) try {

    int id;
    SYCL_CHECK(
        CHECK_TRY_ERROR(id = get_current_device_id()));
    const int compute_capability = ggml_sycl_info().devices[id].cc;

    int mmq_x, mmq_y, nwarps;
    if (compute_capability >= VER_GEN13) {
        mmq_x  =  MMQ_X_Q5_0_RDNA2;
        mmq_y  =  MMQ_Y_Q5_0_RDNA2;
        nwarps = NWARPS_Q5_0_RDNA2;
    } else if (compute_capability >= VER_GEN12) {
        mmq_x  =  MMQ_X_Q5_0_RDNA1;
        mmq_y  =  MMQ_Y_Q5_0_RDNA1;
        nwarps = NWARPS_Q5_0_RDNA1;
    } else if (compute_capability >= VER_GEN9) {
        mmq_x  =  MMQ_X_Q5_0_AMPERE;
        mmq_y  =  MMQ_Y_Q5_0_AMPERE;
        nwarps = NWARPS_Q5_0_AMPERE;
    } else if (compute_capability >= VER_4VEC) {
        mmq_x  =  MMQ_X_Q5_0_PASCAL;
        mmq_y  =  MMQ_Y_Q5_0_PASCAL;
        nwarps = NWARPS_Q5_0_PASCAL;
    } else {
        GGML_ABORT("fatal error");
    }

    const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
    const int block_num_y = (ncols_y + mmq_x - 1) / mmq_x;
    const sycl::range<3> block_nums(1, block_num_y, block_num_x);
    const sycl::range<3> block_dims(1, nwarps, WARP_SIZE);

    if (nrows_x % mmq_y == 0) {
        const bool need_check = false;
        /*
        DPCT1049:24: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        {
            dpct::has_capability_or_fail(stream->get_device(),
                                         {sycl::aspect::fp16});

            stream->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<int, 1> tile_x_ql_q5_0_acc_ct1(
                    sycl::range<1>(mmq_y * (2 * WARP_SIZE) + mmq_y), cgh);
                sycl::local_accessor<float, 1> tile_x_d_q5_0_acc_ct1(
                    sycl::range<1>(mmq_y * (WARP_SIZE / QI5_0) + mmq_y / QI5_0),
                    cgh);
                sycl::local_accessor<int, 1> tile_y_qs_acc_ct1(
                    sycl::range<1>(mmq_x * WARP_SIZE), cgh);
                sycl::local_accessor<sycl::half2, 1> tile_y_ds_acc_ct1(
                    sycl::range<1>(mmq_x * WARP_SIZE / QI8_1), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(block_nums * block_dims, block_dims),
                    [=](sycl::nd_item<3> item_ct1) {
                        mul_mat_q5_0<need_check>(
                            vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y,
                            nrows_dst, item_ct1,
                            get_pointer(tile_x_ql_q5_0_acc_ct1),
                            get_pointer(tile_x_d_q5_0_acc_ct1),
                            get_pointer(tile_y_qs_acc_ct1),
                            get_pointer(tile_y_ds_acc_ct1));
                    });
            });
        }
    } else {
        const bool need_check = true;
        /*
        DPCT1049:25: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        {
            dpct::has_capability_or_fail(stream->get_device(),
                                         {sycl::aspect::fp16});

            stream->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<int, 1> tile_x_ql_q5_0_acc_ct1(
                    sycl::range<1>(mmq_y * (2 * WARP_SIZE) + mmq_y), cgh);
                sycl::local_accessor<float, 1> tile_x_d_q5_0_acc_ct1(
                    sycl::range<1>(mmq_y * (WARP_SIZE / QI5_0) + mmq_y / QI5_0),
                    cgh);
                sycl::local_accessor<int, 1> tile_y_qs_acc_ct1(
                    sycl::range<1>(mmq_x * WARP_SIZE), cgh);
                sycl::local_accessor<sycl::half2, 1> tile_y_ds_acc_ct1(
                    sycl::range<1>(mmq_x * WARP_SIZE / QI8_1), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(block_nums * block_dims, block_dims),
                    [=](sycl::nd_item<3> item_ct1) {
                        mul_mat_q5_0<need_check>(
                            vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y,
                            nrows_dst, item_ct1,
                            get_pointer(tile_x_ql_q5_0_acc_ct1),
                            get_pointer(tile_x_d_q5_0_acc_ct1),
                            get_pointer(tile_y_qs_acc_ct1),
                            get_pointer(tile_y_ds_acc_ct1));
                    });
            });
        }
    }
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static void ggml_mul_mat_q5_1_q8_1_sycl(const void *vx, const void *vy,
                                        float *dst, const int ncols_x,
                                        const int nrows_x, const int ncols_y,
                                        const int nrows_y, const int nrows_dst,
                                        dpct::queue_ptr stream) try {

    int id;
    SYCL_CHECK(
        CHECK_TRY_ERROR(id = get_current_device_id()));
    const int compute_capability = ggml_sycl_info().devices[id].cc;

    int mmq_x, mmq_y, nwarps;
    if (compute_capability >= VER_GEN13) {
        mmq_x  =  MMQ_X_Q5_1_RDNA2;
        mmq_y  =  MMQ_Y_Q5_1_RDNA2;
        nwarps = NWARPS_Q5_1_RDNA2;
    } else if (compute_capability >= VER_GEN12) {
        mmq_x  =  MMQ_X_Q5_1_RDNA1;
        mmq_y  =  MMQ_Y_Q5_1_RDNA1;
        nwarps = NWARPS_Q5_1_RDNA1;
    } else if (compute_capability >= VER_GEN9) {
        mmq_x  =  MMQ_X_Q5_1_AMPERE;
        mmq_y  =  MMQ_Y_Q5_1_AMPERE;
        nwarps = NWARPS_Q5_1_AMPERE;
    } else if (compute_capability >= VER_4VEC) {
        mmq_x  =  MMQ_X_Q5_1_PASCAL;
        mmq_y  =  MMQ_Y_Q5_1_PASCAL;
        nwarps = NWARPS_Q5_1_PASCAL;
    } else {
        GGML_ABORT("fatal error");
    }

    const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
    const int block_num_y = (ncols_y + mmq_x - 1) / mmq_x;
    const sycl::range<3> block_nums(1, block_num_y, block_num_x);
    const sycl::range<3> block_dims(1, nwarps, WARP_SIZE);

    if (nrows_x % mmq_y == 0) {
        const bool need_check = false;
        /*
        DPCT1049:26: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        {
            dpct::has_capability_or_fail(stream->get_device(),
                                         {sycl::aspect::fp16});

            stream->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<int, 1> tile_x_ql_q5_1_acc_ct1(
                    sycl::range<1>(mmq_y * (2 * WARP_SIZE) + mmq_y), cgh);
                sycl::local_accessor<sycl::half2, 1> tile_x_dm_q5_1_acc_ct1(
                    sycl::range<1>(mmq_y * (WARP_SIZE / QI5_1) + mmq_y / QI5_1),
                    cgh);
                sycl::local_accessor<int, 1> tile_y_qs_acc_ct1(
                    sycl::range<1>(mmq_x * WARP_SIZE), cgh);
                sycl::local_accessor<sycl::half2, 1> tile_y_ds_acc_ct1(
                    sycl::range<1>(mmq_x * WARP_SIZE / QI8_1), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(block_nums * block_dims, block_dims),
                    [=](sycl::nd_item<3> item_ct1) {
                        mul_mat_q5_1<need_check>(
                            vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y,
                            nrows_dst, item_ct1,
                            get_pointer(tile_x_ql_q5_1_acc_ct1),
                            get_pointer(tile_x_dm_q5_1_acc_ct1),
                            get_pointer(tile_y_qs_acc_ct1),
                            get_pointer(tile_y_ds_acc_ct1));
                    });
            });
        }
    } else {
        const bool need_check = true;
        /*
        DPCT1049:27: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        {
            dpct::has_capability_or_fail(stream->get_device(),
                                         {sycl::aspect::fp16});

            stream->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<int, 1> tile_x_ql_q5_1_acc_ct1(
                    sycl::range<1>(mmq_y * (2 * WARP_SIZE) + mmq_y), cgh);
                sycl::local_accessor<sycl::half2, 1> tile_x_dm_q5_1_acc_ct1(
                    sycl::range<1>(mmq_y * (WARP_SIZE / QI5_1) + mmq_y / QI5_1),
                    cgh);
                sycl::local_accessor<int, 1> tile_y_qs_acc_ct1(
                    sycl::range<1>(mmq_x * WARP_SIZE), cgh);
                sycl::local_accessor<sycl::half2, 1> tile_y_ds_acc_ct1(
                    sycl::range<1>(mmq_x * WARP_SIZE / QI8_1), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(block_nums * block_dims, block_dims),
                    [=](sycl::nd_item<3> item_ct1) {
                        mul_mat_q5_1<need_check>(
                            vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y,
                            nrows_dst, item_ct1,
                            get_pointer(tile_x_ql_q5_1_acc_ct1),
                            get_pointer(tile_x_dm_q5_1_acc_ct1),
                            get_pointer(tile_y_qs_acc_ct1),
                            get_pointer(tile_y_ds_acc_ct1));
                    });
            });
        }
    }
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static void ggml_mul_mat_q8_0_q8_1_sycl(const void *vx, const void *vy,
                                        float *dst, const int ncols_x,
                                        const int nrows_x, const int ncols_y,
                                        const int nrows_y, const int nrows_dst,
                                        dpct::queue_ptr stream) try {

    int id;
    SYCL_CHECK(
        CHECK_TRY_ERROR(id = get_current_device_id()));
    const int compute_capability = ggml_sycl_info().devices[id].cc;

    int mmq_x, mmq_y, nwarps;
    if (compute_capability >= VER_GEN13) {
        mmq_x  =  MMQ_X_Q8_0_RDNA2;
        mmq_y  =  MMQ_Y_Q8_0_RDNA2;
        nwarps = NWARPS_Q8_0_RDNA2;
    } else if (compute_capability >= VER_GEN12) {
        mmq_x  =  MMQ_X_Q8_0_RDNA1;
        mmq_y  =  MMQ_Y_Q8_0_RDNA1;
        nwarps = NWARPS_Q8_0_RDNA1;
    } else if (compute_capability >= VER_GEN9) {
        mmq_x  =  MMQ_X_Q8_0_AMPERE;
        mmq_y  =  MMQ_Y_Q8_0_AMPERE;
        nwarps = NWARPS_Q8_0_AMPERE;
    } else if (compute_capability >= VER_4VEC) {
        mmq_x  =  MMQ_X_Q8_0_PASCAL;
        mmq_y  =  MMQ_Y_Q8_0_PASCAL;
        nwarps = NWARPS_Q8_0_PASCAL;
    } else {
        GGML_ABORT("fatal error");
    }

    const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
    const int block_num_y = (ncols_y + mmq_x - 1) / mmq_x;
    const sycl::range<3> block_nums(1, block_num_y, block_num_x);
    const sycl::range<3> block_dims(1, nwarps, WARP_SIZE);

    if (nrows_x % mmq_y == 0) {
        const bool need_check = false;
        /*
        DPCT1049:28: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        {
            dpct::has_capability_or_fail(stream->get_device(),
                                         {sycl::aspect::fp16});

            stream->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<int, 1> tile_x_qs_q8_0_acc_ct1(
                    sycl::range<1>(mmq_y * (WARP_SIZE) + mmq_y), cgh);
                sycl::local_accessor<float, 1> tile_x_d_q8_0_acc_ct1(
                    sycl::range<1>(mmq_y * (WARP_SIZE / QI8_0) + mmq_y / QI8_0),
                    cgh);
                sycl::local_accessor<int, 1> tile_y_qs_acc_ct1(
                    sycl::range<1>(mmq_x * WARP_SIZE), cgh);
                sycl::local_accessor<sycl::half2, 1> tile_y_ds_acc_ct1(
                    sycl::range<1>(mmq_x * WARP_SIZE / QI8_1), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(block_nums * block_dims, block_dims),
                    [=](sycl::nd_item<3> item_ct1) {
                        mul_mat_q8_0<need_check>(
                            vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y,
                            nrows_dst, item_ct1,
                            get_pointer(tile_x_qs_q8_0_acc_ct1),
                            get_pointer(tile_x_d_q8_0_acc_ct1),
                            get_pointer(tile_y_qs_acc_ct1),
                            get_pointer(tile_y_ds_acc_ct1));
                    });
            });
        }
    } else {
        const bool need_check = true;
        /*
        DPCT1049:29: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        {
            dpct::has_capability_or_fail(stream->get_device(),
                                         {sycl::aspect::fp16});

            stream->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<int, 1> tile_x_qs_q8_0_acc_ct1(
                    sycl::range<1>(mmq_y * (WARP_SIZE) + mmq_y), cgh);
                sycl::local_accessor<float, 1> tile_x_d_q8_0_acc_ct1(
                    sycl::range<1>(mmq_y * (WARP_SIZE / QI8_0) + mmq_y / QI8_0),
                    cgh);
                sycl::local_accessor<int, 1> tile_y_qs_acc_ct1(
                    sycl::range<1>(mmq_x * WARP_SIZE), cgh);
                sycl::local_accessor<sycl::half2, 1> tile_y_ds_acc_ct1(
                    sycl::range<1>(mmq_x * WARP_SIZE / QI8_1), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(block_nums * block_dims, block_dims),
                    [=](sycl::nd_item<3> item_ct1) {
                        mul_mat_q8_0<need_check>(
                            vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y,
                            nrows_dst, item_ct1,
                            get_pointer(tile_x_qs_q8_0_acc_ct1),
                            get_pointer(tile_x_d_q8_0_acc_ct1),
                            get_pointer(tile_y_qs_acc_ct1),
                            get_pointer(tile_y_ds_acc_ct1));
                    });
            });
        }
    }
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static void ggml_mul_mat_q2_K_q8_1_sycl(const void *vx, const void *vy,
                                        float *dst, const int ncols_x,
                                        const int nrows_x, const int ncols_y,
                                        const int nrows_y, const int nrows_dst,
                                        dpct::queue_ptr stream) try {

    int id;
    SYCL_CHECK(
        CHECK_TRY_ERROR(id = get_current_device_id()));
    const int compute_capability = ggml_sycl_info().devices[id].cc;

    int mmq_x, mmq_y, nwarps;
    if (compute_capability >= VER_GEN13) {
        mmq_x  =  MMQ_X_Q2_K_RDNA2;
        mmq_y  =  MMQ_Y_Q2_K_RDNA2;
        nwarps = NWARPS_Q2_K_RDNA2;
    } else if (compute_capability >= VER_GEN12) {
        mmq_x  =  MMQ_X_Q2_K_RDNA1;
        mmq_y  =  MMQ_Y_Q2_K_RDNA1;
        nwarps = NWARPS_Q2_K_RDNA1;
    } else if (compute_capability >= VER_GEN9) {
        mmq_x  =  MMQ_X_Q2_K_AMPERE;
        mmq_y  =  MMQ_Y_Q2_K_AMPERE;
        nwarps = NWARPS_Q2_K_AMPERE;
    } else if (compute_capability >= VER_4VEC) {
        mmq_x  =  MMQ_X_Q2_K_PASCAL;
        mmq_y  =  MMQ_Y_Q2_K_PASCAL;
        nwarps = NWARPS_Q2_K_PASCAL;
    } else {
        GGML_ABORT("fatal error");
    }

    const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
    const int block_num_y = (ncols_y + mmq_x - 1) / mmq_x;
    const sycl::range<3> block_nums(1, block_num_y, block_num_x);
    const sycl::range<3> block_dims(1, nwarps, WARP_SIZE);

    if (nrows_x % mmq_y == 0) {
        const bool need_check = false;
        /*
        DPCT1049:30: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        {
            dpct::has_capability_or_fail(stream->get_device(),
                                         {sycl::aspect::fp16});

            stream->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<int, 1> tile_x_ql_q2_K_acc_ct1(
                    sycl::range<1>(mmq_y * (WARP_SIZE) + mmq_y), cgh);
                sycl::local_accessor<sycl::half2, 1> tile_x_dm_q2_K_acc_ct1(
                    sycl::range<1>(mmq_y * (WARP_SIZE / QI2_K) + mmq_y / QI2_K),
                    cgh);
                sycl::local_accessor<int, 1> tile_x_sc_q2_K_acc_ct1(
                    sycl::range<1>(mmq_y * (WARP_SIZE / 4) + mmq_y / 4), cgh);
                sycl::local_accessor<int, 1> tile_y_qs_acc_ct1(
                    sycl::range<1>(mmq_x * WARP_SIZE), cgh);
                sycl::local_accessor<sycl::half2, 1> tile_y_ds_acc_ct1(
                    sycl::range<1>(mmq_x * WARP_SIZE / QI8_1), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(block_nums * block_dims, block_dims),
                    [=](sycl::nd_item<3> item_ct1) {
                        mul_mat_q2_K<need_check>(
                            vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y,
                            nrows_dst, item_ct1,
                            get_pointer(tile_x_ql_q2_K_acc_ct1),
                            get_pointer(tile_x_dm_q2_K_acc_ct1),
                            get_pointer(tile_x_sc_q2_K_acc_ct1),
                            get_pointer(tile_y_qs_acc_ct1),
                            get_pointer(tile_y_ds_acc_ct1));
                    });
            });
        }
    } else {
        const bool need_check = true;
        /*
        DPCT1049:31: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        {
            dpct::has_capability_or_fail(stream->get_device(),
                                         {sycl::aspect::fp16});

            stream->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<int, 1> tile_x_ql_q2_K_acc_ct1(
                    sycl::range<1>(mmq_y * (WARP_SIZE) + mmq_y), cgh);
                sycl::local_accessor<sycl::half2, 1> tile_x_dm_q2_K_acc_ct1(
                    sycl::range<1>(mmq_y * (WARP_SIZE / QI2_K) + mmq_y / QI2_K),
                    cgh);
                sycl::local_accessor<int, 1> tile_x_sc_q2_K_acc_ct1(
                    sycl::range<1>(mmq_y * (WARP_SIZE / 4) + mmq_y / 4), cgh);
                sycl::local_accessor<int, 1> tile_y_qs_acc_ct1(
                    sycl::range<1>(mmq_x * WARP_SIZE), cgh);
                sycl::local_accessor<sycl::half2, 1> tile_y_ds_acc_ct1(
                    sycl::range<1>(mmq_x * WARP_SIZE / QI8_1), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(block_nums * block_dims, block_dims),
                    [=](sycl::nd_item<3> item_ct1) {
                        mul_mat_q2_K<need_check>(
                            vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y,
                            nrows_dst, item_ct1,
                            get_pointer(tile_x_ql_q2_K_acc_ct1),
                            get_pointer(tile_x_dm_q2_K_acc_ct1),
                            get_pointer(tile_x_sc_q2_K_acc_ct1),
                            get_pointer(tile_y_qs_acc_ct1),
                            get_pointer(tile_y_ds_acc_ct1));
                    });
            });
        }
    }
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static void ggml_mul_mat_q3_K_q8_1_sycl(const void *vx, const void *vy,
                                        float *dst, const int ncols_x,
                                        const int nrows_x, const int ncols_y,
                                        const int nrows_y, const int nrows_dst,
                                        dpct::queue_ptr stream) try {

#if QK_K == 256

    int id;
    SYCL_CHECK(
        CHECK_TRY_ERROR(id = get_current_device_id()));
    const int compute_capability = ggml_sycl_info().devices[id].cc;

    int mmq_x, mmq_y, nwarps;
    if (compute_capability >= VER_GEN13) {
        mmq_x  =  MMQ_X_Q3_K_RDNA2;
        mmq_y  =  MMQ_Y_Q3_K_RDNA2;
        nwarps = NWARPS_Q3_K_RDNA2;
    } else if (compute_capability >= VER_GEN12) {
        mmq_x  =  MMQ_X_Q3_K_RDNA1;
        mmq_y  =  MMQ_Y_Q3_K_RDNA1;
        nwarps = NWARPS_Q3_K_RDNA1;
    } else if (compute_capability >= VER_GEN9) {
        mmq_x  =  MMQ_X_Q3_K_AMPERE;
        mmq_y  =  MMQ_Y_Q3_K_AMPERE;
        nwarps = NWARPS_Q3_K_AMPERE;
    } else if (compute_capability >= VER_4VEC) {
        mmq_x  =  MMQ_X_Q3_K_PASCAL;
        mmq_y  =  MMQ_Y_Q3_K_PASCAL;
        nwarps = NWARPS_Q3_K_PASCAL;
    } else {
        GGML_ABORT("fatal error");
    }

    const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
    const int block_num_y = (ncols_y + mmq_x - 1) / mmq_x;
    const sycl::range<3> block_nums(1, block_num_y, block_num_x);
    const sycl::range<3> block_dims(1, nwarps, WARP_SIZE);

    if (nrows_x % mmq_y == 0) {
        const bool need_check = false;
        /*
        DPCT1049:32: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        {
            dpct::has_capability_or_fail(stream->get_device(),
                                         {sycl::aspect::fp16});

            stream->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<int, 1> tile_x_ql_q3_K_acc_ct1(
                    sycl::range<1>(mmq_y * (WARP_SIZE) + mmq_y), cgh);
                sycl::local_accessor<sycl::half2, 1> tile_x_dm_q3_K_acc_ct1(
                    sycl::range<1>(mmq_y * (WARP_SIZE / QI3_K) + mmq_y / QI3_K),
                    cgh);
                sycl::local_accessor<int, 1> tile_x_qh_q3_K_acc_ct1(
                    sycl::range<1>(mmq_y * (WARP_SIZE / 2) + mmq_y / 2), cgh);
                sycl::local_accessor<int, 1> tile_x_sc_q3_K_acc_ct1(
                    sycl::range<1>(mmq_y * (WARP_SIZE / 4) + mmq_y / 4), cgh);
                sycl::local_accessor<int, 1> tile_y_qs_acc_ct1(
                    sycl::range<1>(mmq_x * WARP_SIZE), cgh);
                sycl::local_accessor<sycl::half2, 1> tile_y_ds_acc_ct1(
                    sycl::range<1>(mmq_x * WARP_SIZE / QI8_1), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(block_nums * block_dims, block_dims),
                    [=](sycl::nd_item<3> item_ct1) {
                        mul_mat_q3_K<need_check>(
                            vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y,
                            nrows_dst, item_ct1,
                            get_pointer(tile_x_ql_q3_K_acc_ct1),
                            get_pointer(tile_x_dm_q3_K_acc_ct1),
                            get_pointer(tile_x_qh_q3_K_acc_ct1),
                            get_pointer(tile_x_sc_q3_K_acc_ct1),
                            get_pointer(tile_y_qs_acc_ct1),
                            get_pointer(tile_y_ds_acc_ct1));
                    });
            });
        }
    } else {
        const bool need_check = true;
        /*
        DPCT1049:33: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        {
            dpct::has_capability_or_fail(stream->get_device(),
                                         {sycl::aspect::fp16});

            stream->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<int, 1> tile_x_ql_q3_K_acc_ct1(
                    sycl::range<1>(mmq_y * (WARP_SIZE) + mmq_y), cgh);
                sycl::local_accessor<sycl::half2, 1> tile_x_dm_q3_K_acc_ct1(
                    sycl::range<1>(mmq_y * (WARP_SIZE / QI3_K) + mmq_y / QI3_K),
                    cgh);
                sycl::local_accessor<int, 1> tile_x_qh_q3_K_acc_ct1(
                    sycl::range<1>(mmq_y * (WARP_SIZE / 2) + mmq_y / 2), cgh);
                sycl::local_accessor<int, 1> tile_x_sc_q3_K_acc_ct1(
                    sycl::range<1>(mmq_y * (WARP_SIZE / 4) + mmq_y / 4), cgh);
                sycl::local_accessor<int, 1> tile_y_qs_acc_ct1(
                    sycl::range<1>(mmq_x * WARP_SIZE), cgh);
                sycl::local_accessor<sycl::half2, 1> tile_y_ds_acc_ct1(
                    sycl::range<1>(mmq_x * WARP_SIZE / QI8_1), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(block_nums * block_dims, block_dims),
                    [=](sycl::nd_item<3> item_ct1) {
                        mul_mat_q3_K<need_check>(
                            vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y,
                            nrows_dst, item_ct1,
                            get_pointer(tile_x_ql_q3_K_acc_ct1),
                            get_pointer(tile_x_dm_q3_K_acc_ct1),
                            get_pointer(tile_x_qh_q3_K_acc_ct1),
                            get_pointer(tile_x_sc_q3_K_acc_ct1),
                            get_pointer(tile_y_qs_acc_ct1),
                            get_pointer(tile_y_ds_acc_ct1));
                    });
            });
        }
    }
#endif
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static void ggml_mul_mat_q4_K_q8_1_sycl(const void *vx, const void *vy,
                                        float *dst, const int ncols_x,
                                        const int nrows_x, const int ncols_y,
                                        const int nrows_y, const int nrows_dst,
                                        dpct::queue_ptr stream) try {

    int id;
    SYCL_CHECK(
        CHECK_TRY_ERROR(id = get_current_device_id()));
    const int compute_capability = ggml_sycl_info().devices[id].cc;

    int mmq_x, mmq_y, nwarps;
    if (compute_capability >= VER_GEN13) {
        mmq_x  =  MMQ_X_Q4_K_RDNA2;
        mmq_y  =  MMQ_Y_Q4_K_RDNA2;
        nwarps = NWARPS_Q4_K_RDNA2;
    } else if (compute_capability >= VER_GEN12) {
        mmq_x  =  MMQ_X_Q4_K_RDNA1;
        mmq_y  =  MMQ_Y_Q4_K_RDNA1;
        nwarps = NWARPS_Q4_K_RDNA1;
    } else if (compute_capability >= VER_GEN9) {
        mmq_x  =  MMQ_X_Q4_K_AMPERE;
        mmq_y  =  MMQ_Y_Q4_K_AMPERE;
        nwarps = NWARPS_Q4_K_AMPERE;
    } else if (compute_capability >= VER_4VEC) {
        mmq_x  =  MMQ_X_Q4_K_PASCAL;
        mmq_y  =  MMQ_Y_Q4_K_PASCAL;
        nwarps = NWARPS_Q4_K_PASCAL;
    } else {
        GGML_ABORT("fatal error");
    }

    const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
    const int block_num_y = (ncols_y + mmq_x - 1) / mmq_x;
    const sycl::range<3> block_nums(1, block_num_y, block_num_x);
    const sycl::range<3> block_dims(1, nwarps, WARP_SIZE);

    if (nrows_x % mmq_y == 0) {
        const bool need_check = false;
        /*
        DPCT1049:34: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        {
            dpct::has_capability_or_fail(stream->get_device(),
                                         {sycl::aspect::fp16});

            stream->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<int, 1> tile_x_ql_q4_K_acc_ct1(
                    sycl::range<1>(mmq_y * (WARP_SIZE) + mmq_y), cgh);
                sycl::local_accessor<sycl::half2, 1> tile_x_dm_q4_K_acc_ct1(
                    sycl::range<1>(mmq_y * (WARP_SIZE / QI4_K) + mmq_y / QI4_K),
                    cgh);
                sycl::local_accessor<int, 1> tile_x_sc_q4_K_acc_ct1(
                    sycl::range<1>(mmq_y * (WARP_SIZE / 8) + mmq_y / 8), cgh);
                sycl::local_accessor<int, 1> tile_y_qs_acc_ct1(
                    sycl::range<1>(mmq_x * WARP_SIZE), cgh);
                sycl::local_accessor<sycl::half2, 1> tile_y_ds_acc_ct1(
                    sycl::range<1>(mmq_x * WARP_SIZE / QI8_1), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(block_nums * block_dims, block_dims),
                    [=](sycl::nd_item<3> item_ct1) {
                        mul_mat_q4_K<need_check>(
                            vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y,
                            nrows_dst, item_ct1,
                            get_pointer(tile_x_ql_q4_K_acc_ct1),
                            get_pointer(tile_x_dm_q4_K_acc_ct1),
                            get_pointer(tile_x_sc_q4_K_acc_ct1),
                            get_pointer(tile_y_qs_acc_ct1),
                            get_pointer(tile_y_ds_acc_ct1));
                    });
            });
        }
    } else {
        const bool need_check = true;
        /*
        DPCT1049:35: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        {
            dpct::has_capability_or_fail(stream->get_device(),
                                         {sycl::aspect::fp16});

            stream->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<int, 1> tile_x_ql_q4_K_acc_ct1(
                    sycl::range<1>(mmq_y * (WARP_SIZE) + mmq_y), cgh);
                sycl::local_accessor<sycl::half2, 1> tile_x_dm_q4_K_acc_ct1(
                    sycl::range<1>(mmq_y * (WARP_SIZE / QI4_K) + mmq_y / QI4_K),
                    cgh);
                sycl::local_accessor<int, 1> tile_x_sc_q4_K_acc_ct1(
                    sycl::range<1>(mmq_y * (WARP_SIZE / 8) + mmq_y / 8), cgh);
                sycl::local_accessor<int, 1> tile_y_qs_acc_ct1(
                    sycl::range<1>(mmq_x * WARP_SIZE), cgh);
                sycl::local_accessor<sycl::half2, 1> tile_y_ds_acc_ct1(
                    sycl::range<1>(mmq_x * WARP_SIZE / QI8_1), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(block_nums * block_dims, block_dims),
                    [=](sycl::nd_item<3> item_ct1) {
                        mul_mat_q4_K<need_check>(
                            vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y,
                            nrows_dst, item_ct1,
                            get_pointer(tile_x_ql_q4_K_acc_ct1),
                            get_pointer(tile_x_dm_q4_K_acc_ct1),
                            get_pointer(tile_x_sc_q4_K_acc_ct1),
                            get_pointer(tile_y_qs_acc_ct1),
                            get_pointer(tile_y_ds_acc_ct1));
                    });
            });
        }
    }
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static void ggml_mul_mat_q5_K_q8_1_sycl(const void *vx, const void *vy,
                                        float *dst, const int ncols_x,
                                        const int nrows_x, const int ncols_y,
                                        const int nrows_y, const int nrows_dst,
                                        dpct::queue_ptr stream) try {

    int id;
    SYCL_CHECK(
        CHECK_TRY_ERROR(id = get_current_device_id()));
    const int compute_capability = ggml_sycl_info().devices[id].cc;

    int mmq_x, mmq_y, nwarps;
    if (compute_capability >= VER_GEN13) {
        mmq_x  =  MMQ_X_Q5_K_RDNA2;
        mmq_y  =  MMQ_Y_Q5_K_RDNA2;
        nwarps = NWARPS_Q5_K_RDNA2;
    } else if (compute_capability >= VER_GEN12) {
        mmq_x  =  MMQ_X_Q5_K_RDNA1;
        mmq_y  =  MMQ_Y_Q5_K_RDNA1;
        nwarps = NWARPS_Q5_K_RDNA1;
    } else if (compute_capability >= VER_GEN9) {
        mmq_x  =  MMQ_X_Q5_K_AMPERE;
        mmq_y  =  MMQ_Y_Q5_K_AMPERE;
        nwarps = NWARPS_Q5_K_AMPERE;
    } else if (compute_capability >= VER_4VEC) {
        mmq_x  =  MMQ_X_Q5_K_PASCAL;
        mmq_y  =  MMQ_Y_Q5_K_PASCAL;
        nwarps = NWARPS_Q5_K_PASCAL;
    } else {
        GGML_ABORT("fatal error");
    }

    const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
    const int block_num_y = (ncols_y + mmq_x - 1) / mmq_x;
    const sycl::range<3> block_nums(1, block_num_y, block_num_x);
    const sycl::range<3> block_dims(1, nwarps, WARP_SIZE);

    if (nrows_x % mmq_y == 0) {
        const bool need_check = false;
        /*
        DPCT1049:36: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        {
            dpct::has_capability_or_fail(stream->get_device(),
                                         {sycl::aspect::fp16});

            stream->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<int, 1> tile_x_ql_q5_K_acc_ct1(
                    sycl::range<1>(mmq_y * (2 * WARP_SIZE) + mmq_y), cgh);
                sycl::local_accessor<sycl::half2, 1> tile_x_dm_q5_K_acc_ct1(
                    sycl::range<1>(mmq_y * (WARP_SIZE / QI5_K) + mmq_y / QI5_K),
                    cgh);
                sycl::local_accessor<int, 1> tile_x_sc_q5_K_acc_ct1(
                    sycl::range<1>(mmq_y * (WARP_SIZE / 8) + mmq_y / 8), cgh);
                sycl::local_accessor<int, 1> tile_y_qs_acc_ct1(
                    sycl::range<1>(mmq_x * WARP_SIZE), cgh);
                sycl::local_accessor<sycl::half2, 1> tile_y_ds_acc_ct1(
                    sycl::range<1>(mmq_x * WARP_SIZE / QI8_1), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(block_nums * block_dims, block_dims),
                    [=](sycl::nd_item<3> item_ct1) {
                        mul_mat_q5_K<need_check>(
                            vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y,
                            nrows_dst, item_ct1,
                            get_pointer(tile_x_ql_q5_K_acc_ct1),
                            get_pointer(tile_x_dm_q5_K_acc_ct1),
                            get_pointer(tile_x_sc_q5_K_acc_ct1),
                            get_pointer(tile_y_qs_acc_ct1),
                            get_pointer(tile_y_ds_acc_ct1));
                    });
            });
        }
    } else {
        const bool need_check = true;
        /*
        DPCT1049:37: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        {
            dpct::has_capability_or_fail(stream->get_device(),
                                         {sycl::aspect::fp16});

            stream->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<int, 1> tile_x_ql_q5_K_acc_ct1(
                    sycl::range<1>(mmq_y * (2 * WARP_SIZE) + mmq_y), cgh);
                sycl::local_accessor<sycl::half2, 1> tile_x_dm_q5_K_acc_ct1(
                    sycl::range<1>(mmq_y * (WARP_SIZE / QI5_K) + mmq_y / QI5_K),
                    cgh);
                sycl::local_accessor<int, 1> tile_x_sc_q5_K_acc_ct1(
                    sycl::range<1>(mmq_y * (WARP_SIZE / 8) + mmq_y / 8), cgh);
                sycl::local_accessor<int, 1> tile_y_qs_acc_ct1(
                    sycl::range<1>(mmq_x * WARP_SIZE), cgh);
                sycl::local_accessor<sycl::half2, 1> tile_y_ds_acc_ct1(
                    sycl::range<1>(mmq_x * WARP_SIZE / QI8_1), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(block_nums * block_dims, block_dims),
                    [=](sycl::nd_item<3> item_ct1) {
                        mul_mat_q5_K<need_check>(
                            vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y,
                            nrows_dst, item_ct1,
                            get_pointer(tile_x_ql_q5_K_acc_ct1),
                            get_pointer(tile_x_dm_q5_K_acc_ct1),
                            get_pointer(tile_x_sc_q5_K_acc_ct1),
                            get_pointer(tile_y_qs_acc_ct1),
                            get_pointer(tile_y_ds_acc_ct1));
                    });
            });
        }
    }
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static void ggml_mul_mat_q6_K_q8_1_sycl(const void *vx, const void *vy,
                                        float *dst, const int ncols_x,
                                        const int nrows_x, const int ncols_y,
                                        const int nrows_y, const int nrows_dst,
                                        dpct::queue_ptr stream) try {

    int id;
    SYCL_CHECK(
        CHECK_TRY_ERROR(id = get_current_device_id()));
    const int compute_capability = ggml_sycl_info().devices[id].cc;

    int mmq_x, mmq_y, nwarps;
    if (compute_capability >= VER_GEN13) {
        mmq_x  =  MMQ_X_Q6_K_RDNA2;
        mmq_y  =  MMQ_Y_Q6_K_RDNA2;
        nwarps = NWARPS_Q6_K_RDNA2;
    } else if (compute_capability >= VER_GEN12) {
        mmq_x  =  MMQ_X_Q6_K_RDNA1;
        mmq_y  =  MMQ_Y_Q6_K_RDNA1;
        nwarps = NWARPS_Q6_K_RDNA1;
    } else if (compute_capability >= VER_GEN9) {
        mmq_x  =  MMQ_X_Q6_K_AMPERE;
        mmq_y  =  MMQ_Y_Q6_K_AMPERE;
        nwarps = NWARPS_Q6_K_AMPERE;
    } else if (compute_capability >= VER_4VEC) {
        mmq_x  =  MMQ_X_Q6_K_PASCAL;
        mmq_y  =  MMQ_Y_Q6_K_PASCAL;
        nwarps = NWARPS_Q6_K_PASCAL;
    } else {
        GGML_ABORT("fatal error");
    }

    const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
    const int block_num_y = (ncols_y + mmq_x - 1) / mmq_x;
    const sycl::range<3> block_nums(1, block_num_y, block_num_x);
    const sycl::range<3> block_dims(1, nwarps, WARP_SIZE);

    if (nrows_x % mmq_y == 0) {
        const bool need_check = false;
        /*
        DPCT1049:38: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        {
            dpct::has_capability_or_fail(stream->get_device(),
                                         {sycl::aspect::fp16});

            stream->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<int, 1> tile_x_ql_acc_ct1(
                    sycl::range<1>(mmq_y * (2 * WARP_SIZE) + mmq_y), cgh);
                sycl::local_accessor<sycl::half2, 1> tile_x_dm_acc_ct1(
                    sycl::range<1>(mmq_y * (WARP_SIZE / QI6_K) + mmq_y / QI6_K),
                    cgh);
                sycl::local_accessor<int, 1> tile_x_sc_acc_ct1(
                    sycl::range<1>(mmq_y * (WARP_SIZE / 8) + mmq_y / 8), cgh);
                sycl::local_accessor<int, 1> tile_y_qs_acc_ct1(
                    sycl::range<1>(mmq_x * WARP_SIZE), cgh);
                sycl::local_accessor<sycl::half2, 1> tile_y_ds_acc_ct1(
                    sycl::range<1>(mmq_x * WARP_SIZE / QI8_1), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(block_nums * block_dims, block_dims),
                    [=](sycl::nd_item<3> item_ct1) {
                        mul_mat_q6_K<need_check>(
                            vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y,
                            nrows_dst, item_ct1,
                            get_pointer(tile_x_ql_acc_ct1),
                            get_pointer(tile_x_dm_acc_ct1),
                            get_pointer(tile_x_sc_acc_ct1),
                            get_pointer(tile_y_qs_acc_ct1),
                            get_pointer(tile_y_ds_acc_ct1));
                    });
            });
        }
    } else {
        const bool need_check = true;
        /*
        DPCT1049:39: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        {
            dpct::has_capability_or_fail(stream->get_device(),
                                         {sycl::aspect::fp16});

            stream->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<int, 1> tile_x_ql_acc_ct1(
                    sycl::range<1>(mmq_y * (2 * WARP_SIZE) + mmq_y), cgh);
                sycl::local_accessor<sycl::half2, 1> tile_x_dm_acc_ct1(
                    sycl::range<1>(mmq_y * (WARP_SIZE / QI6_K) + mmq_y / QI6_K),
                    cgh);
                sycl::local_accessor<int, 1> tile_x_sc_acc_ct1(
                    sycl::range<1>(mmq_y * (WARP_SIZE / 8) + mmq_y / 8), cgh);
                sycl::local_accessor<int, 1> tile_y_qs_acc_ct1(
                    sycl::range<1>(mmq_x * WARP_SIZE), cgh);
                sycl::local_accessor<sycl::half2, 1> tile_y_ds_acc_ct1(
                    sycl::range<1>(mmq_x * WARP_SIZE / QI8_1), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(block_nums * block_dims, block_dims),
                    [=](sycl::nd_item<3> item_ct1) {
                        mul_mat_q6_K<need_check>(
                            vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y,
                            nrows_dst, item_ct1,
                            get_pointer(tile_x_ql_acc_ct1),
                            get_pointer(tile_x_dm_acc_ct1),
                            get_pointer(tile_x_sc_acc_ct1),
                            get_pointer(tile_y_qs_acc_ct1),
                            get_pointer(tile_y_ds_acc_ct1));
                    });
            });
        }
    }
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

void ggml_sycl_op_mul_mat_q(
    ggml_backend_sycl_context & ctx,
    const ggml_tensor *src0, const ggml_tensor *src1, ggml_tensor *dst,
    const char *src0_dd_i, const float *src1_ddf_i, const char *src1_ddq_i,
    float *dst_dd_i, const int64_t row_low, const int64_t row_high,
    const int64_t src1_ncols, const int64_t src1_padded_row_size,
    const dpct::queue_ptr &stream) try {

    const int64_t ne00 = src0->ne[0];

    const int64_t ne10 = src1->ne[0];
    GGML_ASSERT(ne10 % QK8_1 == 0);

    const int64_t ne0 = dst->ne[0];

    const int64_t row_diff = row_high - row_low;

    int device_id;
    SYCL_CHECK(
        CHECK_TRY_ERROR(device_id = get_current_device_id()));

    // the main device has a larger memory buffer to hold the results from all GPUs
    // nrows_dst == nrows of the matrix that the dequantize_mul_mat kernel writes into
    const int64_t nrows_dst = device_id == ctx.device ? ne0 : row_diff;

    switch (src0->type) {
        case GGML_TYPE_Q4_0:
            ggml_mul_mat_q4_0_q8_1_sycl(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_ncols, src1_padded_row_size, nrows_dst, stream);
            break;
        case GGML_TYPE_Q4_1:
            ggml_mul_mat_q4_1_q8_1_sycl(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_ncols, src1_padded_row_size, nrows_dst, stream);
            break;
        case GGML_TYPE_Q5_0:
            ggml_mul_mat_q5_0_q8_1_sycl(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_ncols, src1_padded_row_size, nrows_dst, stream);
            break;
        case GGML_TYPE_Q5_1:
            ggml_mul_mat_q5_1_q8_1_sycl(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_ncols, src1_padded_row_size, nrows_dst, stream);
            break;
        case GGML_TYPE_Q8_0:
            ggml_mul_mat_q8_0_q8_1_sycl(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_ncols, src1_padded_row_size, nrows_dst, stream);
            break;
        case GGML_TYPE_Q2_K:
            ggml_mul_mat_q2_K_q8_1_sycl(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_ncols, src1_padded_row_size, nrows_dst, stream);
            break;
        case GGML_TYPE_Q3_K:
            ggml_mul_mat_q3_K_q8_1_sycl(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_ncols, src1_padded_row_size, nrows_dst, stream);
            break;
        case GGML_TYPE_Q4_K:
            ggml_mul_mat_q4_K_q8_1_sycl(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_ncols, src1_padded_row_size, nrows_dst, stream);
            break;
        case GGML_TYPE_Q5_K:
            ggml_mul_mat_q5_K_q8_1_sycl(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_ncols, src1_padded_row_size, nrows_dst, stream);
            break;
        case GGML_TYPE_Q6_K:
            ggml_mul_mat_q6_K_q8_1_sycl(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_ncols, src1_padded_row_size, nrows_dst, stream);
            break;
        default:
            GGML_ABORT("fatal error");
    }

    GGML_UNUSED(src1);
    GGML_UNUSED(dst);
    GGML_UNUSED(src1_ddf_i);
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}
