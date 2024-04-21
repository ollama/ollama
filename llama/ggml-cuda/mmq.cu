#include "mmq.cuh"
#include "vecdotq.cuh"

typedef void (*allocate_tiles_cuda_t)(int ** x_ql, half2 ** x_dm, int ** x_qh, int ** x_sc);
typedef void (*load_tiles_cuda_t)(
    const void * __restrict__ vx, int * __restrict__ x_ql, half2 * __restrict__ x_dm, int * __restrict__ x_qh,
    int * __restrict__ x_sc, const int & i_offset, const int & i_max, const int & k, const int & blocks_per_row);
typedef float (*vec_dot_q_mul_mat_cuda_t)(
    const int * __restrict__ x_ql, const half2 * __restrict__ x_dm, const int * __restrict__ x_qh, const int * __restrict__ x_sc,
    const int * __restrict__ y_qs, const half2 * __restrict__ y_ms, const int & i, const int & j, const int & k);
typedef void (*dot_kernel_k_t)(const void * __restrict__ vx, const int ib, const int iqs, const float * __restrict__ y, float & v);

template <int mmq_y> static __device__ __forceinline__ void allocate_tiles_q4_0(int ** x_ql, half2 ** x_dm, int ** x_qh, int ** x_sc) {
    GGML_UNUSED(x_qh);
    GGML_UNUSED(x_sc);

    __shared__ int  tile_x_qs[mmq_y * (WARP_SIZE)       + mmq_y];
    __shared__ float tile_x_d[mmq_y * (WARP_SIZE/QI4_0) + mmq_y/QI4_0];

    *x_ql = tile_x_qs;
    *x_dm = (half2 *) tile_x_d;
}

template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_q4_0(
    const void * __restrict__ vx, int * __restrict__ x_ql, half2 * __restrict__ x_dm, int * __restrict__ x_qh,
    int * __restrict__ x_sc, const int & i_offset, const int & i_max, const int & k, const int & blocks_per_row) {
    GGML_UNUSED(x_qh); GGML_UNUSED(x_sc);
    GGML_CUDA_ASSUME(i_offset >= 0);
    GGML_CUDA_ASSUME(i_offset <  nwarps);
    GGML_CUDA_ASSUME(k >= 0);
    GGML_CUDA_ASSUME(k <  WARP_SIZE);

    const int kbx  = k / QI4_0;
    const int kqsx = k % QI4_0;

    const block_q4_0 * bx0 = (const block_q4_0 *) vx;

    float * x_dmf = (float *) x_dm;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + i_offset;

        if (need_check) {
            i = min(i, i_max);
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
            i = min(i, i_max);
        }

        const block_q4_0 * bxi = bx0 + i*blocks_per_row + kbxd;

        x_dmf[i * (WARP_SIZE/QI4_0) + i / QI4_0 + kbxd] = bxi->d;
    }
}

static __device__ __forceinline__ float vec_dot_q4_0_q8_1_mul_mat(
    const int * __restrict__ x_ql, const half2 * __restrict__ x_dm, const int * __restrict__ x_qh, const int * __restrict__ x_sc,
    const int * __restrict__ y_qs, const half2 * __restrict__ y_ds, const int & i, const int & j, const int & k) {
    GGML_UNUSED(x_qh); GGML_UNUSED(x_sc);

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

template <int mmq_y> static __device__ __forceinline__ void allocate_tiles_q4_1(int ** x_ql, half2 ** x_dm, int ** x_qh, int ** x_sc) {
    GGML_UNUSED(x_qh); GGML_UNUSED(x_sc);

    __shared__ int   tile_x_qs[mmq_y * (WARP_SIZE) +     + mmq_y];
    __shared__ half2 tile_x_dm[mmq_y * (WARP_SIZE/QI4_1) + mmq_y/QI4_1];

    *x_ql = tile_x_qs;
    *x_dm = tile_x_dm;
}

template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_q4_1(
    const void * __restrict__ vx, int * __restrict__ x_ql, half2 * __restrict__ x_dm, int * __restrict__ x_qh,
    int * __restrict__ x_sc, const int & i_offset, const int & i_max, const int & k, const int & blocks_per_row) {
    GGML_UNUSED(x_qh); GGML_UNUSED(x_sc);

    GGML_CUDA_ASSUME(i_offset >= 0);
    GGML_CUDA_ASSUME(i_offset <  nwarps);
    GGML_CUDA_ASSUME(k >= 0);
    GGML_CUDA_ASSUME(k <  WARP_SIZE);

    const int kbx  = k / QI4_1;
    const int kqsx = k % QI4_1;

    const block_q4_1 * bx0 = (const block_q4_1 *) vx;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + i_offset;

        if (need_check) {
            i = min(i, i_max);
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
            i = min(i, i_max);
        }

        const block_q4_1 * bxi = bx0 + i*blocks_per_row + kbxd;

        x_dm[i * (WARP_SIZE/QI4_1) + i / QI4_1 + kbxd] = bxi->dm;
    }
}

static __device__ __forceinline__ float vec_dot_q4_1_q8_1_mul_mat(
    const int * __restrict__ x_ql, const half2 * __restrict__ x_dm, const int * __restrict__ x_qh, const int * __restrict__ x_sc,
    const int * __restrict__ y_qs, const half2 * __restrict__ y_ds, const int & i, const int & j, const int & k) {
    GGML_UNUSED(x_qh); GGML_UNUSED(x_sc);

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

template <int mmq_y> static __device__ __forceinline__ void allocate_tiles_q5_0(int ** x_ql, half2 ** x_dm, int ** x_qh, int ** x_sc) {
    GGML_UNUSED(x_qh); GGML_UNUSED(x_sc);

    __shared__ int  tile_x_ql[mmq_y * (2*WARP_SIZE)     + mmq_y];
    __shared__ float tile_x_d[mmq_y * (WARP_SIZE/QI5_0) + mmq_y/QI5_0];

    *x_ql = tile_x_ql;
    *x_dm = (half2 *) tile_x_d;
}

template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_q5_0(
    const void * __restrict__ vx, int * __restrict__ x_ql, half2 * __restrict__ x_dm, int * __restrict__ x_qh,
    int * __restrict__ x_sc, const int & i_offset, const int & i_max, const int & k, const int & blocks_per_row) {
    GGML_UNUSED(x_qh); GGML_UNUSED(x_sc);

    GGML_CUDA_ASSUME(i_offset >= 0);
    GGML_CUDA_ASSUME(i_offset <  nwarps);
    GGML_CUDA_ASSUME(k >= 0);
    GGML_CUDA_ASSUME(k <  WARP_SIZE);

    const int kbx  = k / QI5_0;
    const int kqsx = k % QI5_0;

    const block_q5_0 * bx0 = (const block_q5_0 *) vx;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + i_offset;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q5_0 * bxi = bx0 + i*blocks_per_row + kbx;

        const int ql = get_int_from_uint8(bxi->qs, kqsx);
        const int qh = get_int_from_uint8(bxi->qh, 0) >> (4 * (k % QI5_0));

        int qs0 = (ql >>  0)   & 0x0F0F0F0F;
        qs0    |= (qh <<  4)   & 0x00000010;  // 0 ->  4
        qs0    |= (qh << 11)   & 0x00001000;  // 1 -> 12
        qs0    |= (qh << 18)   & 0x00100000;  // 2 -> 20
        qs0    |= (qh << 25)   & 0x10000000;  // 3 -> 28
        qs0     = __vsubss4(qs0, 0x10101010); // subtract 16

        x_ql[i * (2*WARP_SIZE + 1) + 2*k+0] = qs0;

        int qs1 = (ql >>  4)   & 0x0F0F0F0F;
        qs1    |= (qh >> 12)   & 0x00000010;  // 16 ->  4
        qs1    |= (qh >>  5)   & 0x00001000;  // 17 -> 12
        qs1    |= (qh <<  2)   & 0x00100000;  // 18 -> 20
        qs1    |= (qh <<  9)   & 0x10000000;  // 19 -> 28
        qs1     = __vsubss4(qs1, 0x10101010); // subtract 16

        x_ql[i * (2*WARP_SIZE + 1) + 2*k+1] = qs1;
    }

    const int blocks_per_tile_x_row = WARP_SIZE / QI5_0;
    const int kbxd = k % blocks_per_tile_x_row;
    float * x_dmf = (float *) x_dm;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI5_0) {
        int i = i0 + i_offset * QI5_0 + k / blocks_per_tile_x_row;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q5_0 * bxi = bx0 + i*blocks_per_row + kbxd;

        x_dmf[i * (WARP_SIZE/QI5_0) + i / QI5_0 + kbxd] = bxi->d;
    }
}

static __device__ __forceinline__ float vec_dot_q5_0_q8_1_mul_mat(
    const int * __restrict__ x_ql, const half2 * __restrict__ x_dm, const int * __restrict__ x_qh, const int * __restrict__ x_sc,
    const int * __restrict__ y_qs, const half2 * __restrict__ y_ds, const int & i, const int & j, const int & k) {
    GGML_UNUSED(x_qh); GGML_UNUSED(x_sc);

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


template <int mmq_y> static __device__ __forceinline__ void allocate_tiles_q5_1(int ** x_ql, half2 ** x_dm, int ** x_qh, int ** x_sc) {
    GGML_UNUSED(x_qh); GGML_UNUSED(x_sc);

    __shared__ int   tile_x_ql[mmq_y * (2*WARP_SIZE)     + mmq_y];
    __shared__ half2 tile_x_dm[mmq_y * (WARP_SIZE/QI5_1) + mmq_y/QI5_1];

    *x_ql = tile_x_ql;
    *x_dm = tile_x_dm;
}

template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_q5_1(
    const void * __restrict__ vx, int * __restrict__ x_ql, half2 * __restrict__ x_dm, int * __restrict__ x_qh,
    int * __restrict__ x_sc, const int & i_offset, const int & i_max, const int & k, const int & blocks_per_row) {
    GGML_UNUSED(x_qh); GGML_UNUSED(x_sc);

    GGML_CUDA_ASSUME(i_offset >= 0);
    GGML_CUDA_ASSUME(i_offset < nwarps);
    GGML_CUDA_ASSUME(k >= 0);
    GGML_CUDA_ASSUME(k <  WARP_SIZE);

    const int kbx  = k / QI5_1;
    const int kqsx = k % QI5_1;

    const block_q5_1 * bx0 = (const block_q5_1 *) vx;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + i_offset;

        if (need_check) {
            i = min(i, i_max);
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
            i = min(i, i_max);
        }

        const block_q5_1 * bxi = bx0 + i*blocks_per_row + kbxd;

        x_dm[i * (WARP_SIZE/QI5_1) + i / QI5_1 + kbxd] = bxi->dm;
    }
}

static __device__ __forceinline__ float vec_dot_q5_1_q8_1_mul_mat(
    const int * __restrict__ x_ql, const half2 * __restrict__ x_dm, const int * __restrict__ x_qh, const int * __restrict__ x_sc,
    const int * __restrict__ y_qs, const half2 * __restrict__ y_ds, const int & i, const int & j, const int & k) {
    GGML_UNUSED(x_qh); GGML_UNUSED(x_sc);

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

template <int mmq_y> static __device__ __forceinline__ void allocate_tiles_q8_0(int ** x_ql, half2 ** x_dm, int ** x_qh, int ** x_sc) {
    GGML_UNUSED(x_qh); GGML_UNUSED(x_sc);

    __shared__ int  tile_x_qs[mmq_y * (WARP_SIZE)       + mmq_y];
    __shared__ float tile_x_d[mmq_y * (WARP_SIZE/QI8_0) + mmq_y/QI8_0];

    *x_ql = tile_x_qs;
    *x_dm = (half2 *) tile_x_d;
}

template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_q8_0(
    const void * __restrict__ vx, int * __restrict__ x_ql, half2 * __restrict__ x_dm, int * __restrict__ x_qh,
    int * __restrict__ x_sc, const int & i_offset, const int & i_max, const int & k, const int & blocks_per_row) {
    GGML_UNUSED(x_qh); GGML_UNUSED(x_sc);

    GGML_CUDA_ASSUME(i_offset >= 0);
    GGML_CUDA_ASSUME(i_offset <  nwarps);
    GGML_CUDA_ASSUME(k >= 0);
    GGML_CUDA_ASSUME(k <  WARP_SIZE);

    const int kbx  = k / QI8_0;
    const int kqsx = k % QI8_0;
    float * x_dmf = (float *) x_dm;

    const block_q8_0 * bx0 = (const block_q8_0 *) vx;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + i_offset;

        if (need_check) {
            i = min(i, i_max);
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
            i = min(i, i_max);
        }

        const block_q8_0 * bxi = bx0 + i*blocks_per_row + kbxd;

        x_dmf[i * (WARP_SIZE/QI8_0) + i / QI8_0 + kbxd] = bxi->d;
    }
}

static __device__ __forceinline__ float vec_dot_q8_0_q8_1_mul_mat(
    const int * __restrict__ x_ql, const half2 * __restrict__ x_dm, const int * __restrict__ x_qh, const int * __restrict__ x_sc,
    const int * __restrict__ y_qs, const half2 * __restrict__ y_ds, const int & i, const int & j, const int & k) {
    GGML_UNUSED(x_qh); GGML_UNUSED(x_sc);

    const float * x_dmf = (const float *) x_dm;
    const float * y_df  = (const float *) y_ds;

    return vec_dot_q8_0_q8_1_impl<VDR_Q8_0_Q8_1_MMQ>
        (&x_ql[i * (WARP_SIZE + 1) + k], &y_qs[j * WARP_SIZE + k], x_dmf[i * (WARP_SIZE/QI8_0) + i/QI8_0 + k/QI8_0],
         y_df[j * (WARP_SIZE/QI8_1) + k/QI8_1]);
}

template <int mmq_y> static __device__ __forceinline__ void allocate_tiles_q2_K(int ** x_ql, half2 ** x_dm, int ** x_qh, int ** x_sc) {
    GGML_UNUSED(x_qh);

    __shared__ int   tile_x_ql[mmq_y * (WARP_SIZE)       + mmq_y];
    __shared__ half2 tile_x_dm[mmq_y * (WARP_SIZE/QI2_K) + mmq_y/QI2_K];
    __shared__ int   tile_x_sc[mmq_y * (WARP_SIZE/4)     + mmq_y/4];

    *x_ql = tile_x_ql;
    *x_dm = tile_x_dm;
    *x_sc = tile_x_sc;
}

template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_q2_K(
    const void * __restrict__ vx, int * __restrict__ x_ql, half2 * __restrict__ x_dm, int * __restrict__ x_qh,
    int * __restrict__ x_sc, const int & i_offset, const int & i_max, const int & k, const int & blocks_per_row) {
    GGML_UNUSED(x_qh);

    GGML_CUDA_ASSUME(i_offset >= 0);
    GGML_CUDA_ASSUME(i_offset <  nwarps);
    GGML_CUDA_ASSUME(k >= 0);
    GGML_CUDA_ASSUME(k <  WARP_SIZE);

    const int kbx  = k / QI2_K;
    const int kqsx = k % QI2_K;

    const block_q2_K * bx0 = (const block_q2_K *) vx;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + i_offset;

        if (need_check) {
            i = min(i, i_max);
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
            i = min(i, i_max);
        }

        const block_q2_K * bxi = bx0 + i*blocks_per_row + kbxd;

        x_dm[i * (WARP_SIZE/QI2_K) + i / QI2_K + kbxd] = bxi->dm;
    }

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * 4) {
        int i = i0 + i_offset * 4 + k / (WARP_SIZE/4);

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q2_K * bxi = bx0 + i*blocks_per_row + (k % (WARP_SIZE/4)) / (QI2_K/4);

        x_sc[i * (WARP_SIZE/4) + i / 4 + k % (WARP_SIZE/4)] = get_int_from_uint8_aligned(bxi->scales, k % (QI2_K/4));
    }
}

static __device__ __forceinline__ float vec_dot_q2_K_q8_1_mul_mat(
    const int * __restrict__ x_ql, const half2 * __restrict__ x_dm, const int * __restrict__ x_qh, const int * __restrict__ x_sc,
    const int * __restrict__ y_qs, const half2 * __restrict__ y_ds, const int & i, const int & j, const int & k) {
    GGML_UNUSED(x_qh);

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

template <int mmq_y> static __device__ __forceinline__ void allocate_tiles_q3_K(int ** x_ql, half2 ** x_dm, int ** x_qh, int ** x_sc) {

    __shared__ int   tile_x_ql[mmq_y * (WARP_SIZE)       + mmq_y];
    __shared__ half2 tile_x_dm[mmq_y * (WARP_SIZE/QI3_K) + mmq_y/QI3_K];
    __shared__ int   tile_x_qh[mmq_y * (WARP_SIZE/2)     + mmq_y/2];
    __shared__ int   tile_x_sc[mmq_y * (WARP_SIZE/4)     + mmq_y/4];

    *x_ql = tile_x_ql;
    *x_dm = tile_x_dm;
    *x_qh = tile_x_qh;
    *x_sc = tile_x_sc;
}

template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_q3_K(
    const void * __restrict__ vx, int * __restrict__ x_ql, half2 * __restrict__ x_dm, int * __restrict__ x_qh,
    int * __restrict__ x_sc, const int & i_offset, const int & i_max, const int & k, const int & blocks_per_row) {

    GGML_CUDA_ASSUME(i_offset >= 0);
    GGML_CUDA_ASSUME(i_offset <  nwarps);
    GGML_CUDA_ASSUME(k >= 0);
    GGML_CUDA_ASSUME(k <  WARP_SIZE);

    const int kbx  = k / QI3_K;
    const int kqsx = k % QI3_K;

    const block_q3_K * bx0 = (const block_q3_K *) vx;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + i_offset;

        if (need_check) {
            i = min(i, i_max);
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
            i = min(i, i_max);
        }

        const block_q3_K * bxi = bx0 + i*blocks_per_row + kbxd;

        x_dmf[i * (WARP_SIZE/QI3_K) + i / QI3_K + kbxd] = bxi->d;
    }

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * 2) {
        int i = i0 + i_offset * 2 + k / (WARP_SIZE/2);

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q3_K * bxi = bx0 + i*blocks_per_row + (k % (WARP_SIZE/2)) / (QI3_K/2);

        // invert the mask with ~ so that a 0/1 results in 4/0 being subtracted
        x_qh[i * (WARP_SIZE/2) + i / 2 + k % (WARP_SIZE/2)] = ~get_int_from_uint8(bxi->hmask, k % (QI3_K/2));
    }

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * 4) {
        int i = i0 + i_offset * 4 + k / (WARP_SIZE/4);

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q3_K * bxi = bx0 + i*blocks_per_row + (k % (WARP_SIZE/4)) / (QI3_K/4);

        const int ksc = k % (QI3_K/4);

        const int ksc_low = ksc % (QI3_K/8);
        const int shift_low = 4 * (ksc / (QI3_K/8));
        const int sc_low = (get_int_from_uint8(bxi->scales, ksc_low) >> shift_low) & 0x0F0F0F0F;

        const int ksc_high = QI3_K/8;
        const int shift_high = 2 * ksc;
        const int sc_high = ((get_int_from_uint8(bxi->scales, ksc_high) >> shift_high) << 4) & 0x30303030;

        const int sc = __vsubss4(sc_low | sc_high, 0x20202020);

        x_sc[i * (WARP_SIZE/4) + i / 4 + k % (WARP_SIZE/4)] = sc;
    }
}

static __device__ __forceinline__ float vec_dot_q3_K_q8_1_mul_mat(
    const int * __restrict__ x_ql, const half2 * __restrict__ x_dm, const int * __restrict__ x_qh, const int * __restrict__ x_sc,
    const int * __restrict__ y_qs, const half2 * __restrict__ y_ds, const int & i, const int & j, const int & k) {

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

        v[l] = __vsubss4(vll, vlh);
    }

    const int index_y = j * WARP_SIZE + (k*QR3_K) % WARP_SIZE;
    return vec_dot_q3_K_q8_1_impl_mmq(v, &y_qs[index_y], scales, x_dmf[i * (WARP_SIZE/QI3_K) + i/QI3_K + kbx], y_df[index_y/QI8_1]);
}

template <int mmq_y> static __device__ __forceinline__ void allocate_tiles_q4_K(int ** x_ql, half2 ** x_dm, int ** x_qh, int ** x_sc) {
    GGML_UNUSED(x_qh);

    __shared__ int   tile_x_ql[mmq_y * (WARP_SIZE)       + mmq_y];
    __shared__ half2 tile_x_dm[mmq_y * (WARP_SIZE/QI4_K) + mmq_y/QI4_K];
    __shared__ int   tile_x_sc[mmq_y * (WARP_SIZE/8)     + mmq_y/8];

    *x_ql = tile_x_ql;
    *x_dm = tile_x_dm;
    *x_sc = tile_x_sc;
}

template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_q4_K(
    const void * __restrict__ vx, int * __restrict__ x_ql, half2 * __restrict__ x_dm, int * __restrict__ x_qh,
    int * __restrict__ x_sc, const int & i_offset, const int & i_max, const int & k, const int & blocks_per_row) {
    GGML_UNUSED(x_qh);

    GGML_CUDA_ASSUME(i_offset >= 0);
    GGML_CUDA_ASSUME(i_offset <  nwarps);
    GGML_CUDA_ASSUME(k >= 0);
    GGML_CUDA_ASSUME(k <  WARP_SIZE);

    const int kbx  = k / QI4_K; // == 0 if QK_K == 256
    const int kqsx = k % QI4_K; // == k if QK_K == 256

    const block_q4_K * bx0 = (const block_q4_K *) vx;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + i_offset;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q4_K * bxi = bx0 + i*blocks_per_row + kbx;

        x_ql[i * (WARP_SIZE + 1) + k] = get_int_from_uint8_aligned(bxi->qs, kqsx);
    }

    const int blocks_per_tile_x_row = WARP_SIZE / QI4_K; // == 1 if QK_K == 256
    const int kbxd = k % blocks_per_tile_x_row;          // == 0 if QK_K == 256

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI4_K) {
        int i = (i0 + i_offset * QI4_K + k / blocks_per_tile_x_row) % mmq_y;

        if (need_check) {
            i = min(i, i_max);
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
            i = min(i, i_max);
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

static __device__ __forceinline__ float vec_dot_q4_K_q8_1_mul_mat(
    const int * __restrict__ x_ql, const half2 * __restrict__ x_dm, const int * __restrict__ x_qh, const int * __restrict__ x_sc,
    const int * __restrict__ y_qs, const half2 * __restrict__ y_ds, const int & i, const int & j, const int & k) {
    GGML_UNUSED(x_qh);

    const uint8_t * sc = ((const uint8_t *) &x_sc[i * (WARP_SIZE/8) + i/8 + k/16]) + 2*((k % 16) / 8);

    const int index_y = j * WARP_SIZE + (QR4_K*k) % WARP_SIZE;
    return vec_dot_q4_K_q8_1_impl_mmq(&x_ql[i * (WARP_SIZE + 1) + k], &y_qs[index_y], sc, sc+8,
                                      x_dm[i * (WARP_SIZE/QI4_K) + i/QI4_K], &y_ds[index_y/QI8_1]);
}

template <int mmq_y> static __device__ __forceinline__ void allocate_tiles_q5_K(int ** x_ql, half2 ** x_dm, int ** x_qh, int ** x_sc) {
    GGML_UNUSED(x_qh);

    __shared__ int   tile_x_ql[mmq_y * (2*WARP_SIZE)     + mmq_y];
    __shared__ half2 tile_x_dm[mmq_y * (WARP_SIZE/QI5_K) + mmq_y/QI5_K];
    __shared__ int   tile_x_sc[mmq_y * (WARP_SIZE/8)     + mmq_y/8];

    *x_ql = tile_x_ql;
    *x_dm = tile_x_dm;
    *x_sc = tile_x_sc;
}

template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_q5_K(
    const void * __restrict__ vx, int * __restrict__ x_ql, half2 * __restrict__ x_dm, int * __restrict__ x_qh,
    int * __restrict__ x_sc, const int & i_offset, const int & i_max, const int & k, const int & blocks_per_row) {
    GGML_UNUSED(x_qh);

    GGML_CUDA_ASSUME(i_offset >= 0);
    GGML_CUDA_ASSUME(i_offset <  nwarps);
    GGML_CUDA_ASSUME(k >= 0);
    GGML_CUDA_ASSUME(k <  WARP_SIZE);

    const int kbx  = k / QI5_K; // == 0 if QK_K == 256
    const int kqsx = k % QI5_K; // == k if QK_K == 256

    const block_q5_K * bx0 = (const block_q5_K *) vx;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + i_offset;

        if (need_check) {
            i = min(i, i_max);
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

    const int blocks_per_tile_x_row = WARP_SIZE / QI5_K; // == 1 if QK_K == 256
    const int kbxd = k % blocks_per_tile_x_row;          // == 0 if QK_K == 256

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI5_K) {
        int i = (i0 + i_offset * QI5_K + k / blocks_per_tile_x_row) % mmq_y;

        if (need_check) {
            i = min(i, i_max);
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
            i = min(i, i_max);
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

static __device__ __forceinline__ float vec_dot_q5_K_q8_1_mul_mat(
    const int * __restrict__ x_ql, const half2 * __restrict__ x_dm, const int * __restrict__ x_qh, const int * __restrict__ x_sc,
    const int * __restrict__ y_qs, const half2 * __restrict__ y_ds, const int & i, const int & j, const int & k) {
    GGML_UNUSED(x_qh);

    const uint8_t * sc = ((const uint8_t *) &x_sc[i * (WARP_SIZE/8) + i/8 + k/16]) + 2 * ((k % 16) / 8);

    const int index_x = i * (QR5_K*WARP_SIZE + 1) +  QR5_K*k;
    const int index_y = j * WARP_SIZE             + (QR5_K*k) % WARP_SIZE;
    return vec_dot_q5_K_q8_1_impl_mmq(&x_ql[index_x], &y_qs[index_y], sc, sc+8,
                                      x_dm[i * (WARP_SIZE/QI5_K) + i/QI5_K], &y_ds[index_y/QI8_1]);
}

template <int mmq_y> static __device__ __forceinline__ void allocate_tiles_q6_K(int ** x_ql, half2 ** x_dm, int ** x_qh, int ** x_sc) {
    GGML_UNUSED(x_qh);

    __shared__ int   tile_x_ql[mmq_y * (2*WARP_SIZE)     + mmq_y];
    __shared__ half2 tile_x_dm[mmq_y * (WARP_SIZE/QI6_K) + mmq_y/QI6_K];
    __shared__ int   tile_x_sc[mmq_y * (WARP_SIZE/8)     + mmq_y/8];

    *x_ql = tile_x_ql;
    *x_dm = tile_x_dm;
    *x_sc = tile_x_sc;
}

template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_q6_K(
    const void * __restrict__ vx, int * __restrict__ x_ql, half2 * __restrict__ x_dm, int * __restrict__ x_qh,
    int * __restrict__ x_sc, const int & i_offset, const int & i_max, const int & k, const int & blocks_per_row) {
    GGML_UNUSED(x_qh);

    GGML_CUDA_ASSUME(i_offset >= 0);
    GGML_CUDA_ASSUME(i_offset <  nwarps);
    GGML_CUDA_ASSUME(k >= 0);
    GGML_CUDA_ASSUME(k <  WARP_SIZE);

    const int kbx  = k / QI6_K; // == 0 if QK_K == 256
    const int kqsx = k % QI6_K; // == k if QK_K == 256

    const block_q6_K * bx0 = (const block_q6_K *) vx;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + i_offset;

        if (need_check) {
            i = min(i, i_max);
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

        x_ql[i * (2*WARP_SIZE + 1) + kq0] = __vsubss4(ql0 | qh0, 0x20202020);
        x_ql[i * (2*WARP_SIZE + 1) + kq1] = __vsubss4(ql1 | qh1, 0x20202020);
    }

    const int blocks_per_tile_x_row = WARP_SIZE / QI6_K; // == 1 if QK_K == 256
    const int kbxd = k % blocks_per_tile_x_row;          // == 0 if QK_K == 256
    float * x_dmf = (float *) x_dm;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI6_K) {
        int i = (i0 + i_offset * QI6_K + k / blocks_per_tile_x_row) % mmq_y;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q6_K * bxi = bx0 + i*blocks_per_row + kbxd;

        x_dmf[i * (WARP_SIZE/QI6_K) + i / QI6_K + kbxd] = bxi->d;
    }

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * 8) {
        int i = (i0 + i_offset * 8 + k / (WARP_SIZE/8)) % mmq_y;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q6_K * bxi = bx0 + i*blocks_per_row + (k % (WARP_SIZE/8)) / 4;

        x_sc[i * (WARP_SIZE/8) + i / 8 + k % (WARP_SIZE/8)] = get_int_from_int8(bxi->scales, k % (QI6_K/8));
    }
}

static __device__ __forceinline__ float vec_dot_q6_K_q8_1_mul_mat(
    const int * __restrict__ x_ql, const half2 * __restrict__ x_dm, const int * __restrict__ x_qh, const int * __restrict__ x_sc,
    const int * __restrict__ y_qs, const half2 * __restrict__ y_ds, const int & i, const int & j, const int & k) {
    GGML_UNUSED(x_qh);

    const float * x_dmf = (const float *) x_dm;
    const float * y_df  = (const float *) y_ds;

    const int8_t * sc = ((const int8_t *) &x_sc[i * (WARP_SIZE/8) + i/8 + k/8]);

    const int index_x = i * (QR6_K*WARP_SIZE + 1) +  QR6_K*k;
    const int index_y = j * WARP_SIZE             + (QR6_K*k) % WARP_SIZE;
    return vec_dot_q6_K_q8_1_impl_mmq(&x_ql[index_x], &y_qs[index_y], sc, x_dmf[i * (WARP_SIZE/QI6_K) + i/QI6_K], &y_df[index_y/QI8_1]);
}

#define  MMQ_X_Q4_0_RDNA2  64
#define  MMQ_Y_Q4_0_RDNA2  128
#define NWARPS_Q4_0_RDNA2  8
#define  MMQ_X_Q4_0_RDNA1  64
#define  MMQ_Y_Q4_0_RDNA1  64
#define NWARPS_Q4_0_RDNA1  8
#if defined(CUDA_USE_TENSOR_CORES)
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

template <int qk, int qr, int qi, bool need_sum, typename block_q_t, int mmq_x, int mmq_y, int nwarps,
              allocate_tiles_cuda_t allocate_tiles, load_tiles_cuda_t load_tiles, int vdr, vec_dot_q_mul_mat_cuda_t vec_dot>
static __device__ __forceinline__ void mul_mat_q(
    const void * __restrict__ vx, const void * __restrict__ vy, float * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int ncols_y, const int nrows_y, const int nrows_dst) {

    const block_q_t  * x = (const block_q_t  *) vx;
    const block_q8_1 * y = (const block_q8_1 *) vy;

    const int blocks_per_row_x = ncols_x / qk;
    const int blocks_per_col_y = nrows_y / QK8_1;
    const int blocks_per_warp = WARP_SIZE / qi;

    const int & ncols_dst = ncols_y;

    const int row_dst_0 = blockIdx.x*mmq_y;
    const int & row_x_0 = row_dst_0;

    const int col_dst_0 = blockIdx.y*mmq_x;
    const int & col_y_0 = col_dst_0;

    int   * tile_x_ql = nullptr;
    half2 * tile_x_dm = nullptr;
    int   * tile_x_qh = nullptr;
    int   * tile_x_sc = nullptr;

    allocate_tiles(&tile_x_ql, &tile_x_dm, &tile_x_qh, &tile_x_sc);

    __shared__ int    tile_y_qs[mmq_x * WARP_SIZE];
    __shared__ half2  tile_y_ds[mmq_x * WARP_SIZE/QI8_1];

    float sum[mmq_y/WARP_SIZE][mmq_x/nwarps] = {{0.0f}};

    for (int ib0 = 0; ib0 < blocks_per_row_x; ib0 += blocks_per_warp) {

        load_tiles(x + row_x_0*blocks_per_row_x + ib0, tile_x_ql, tile_x_dm, tile_x_qh, tile_x_sc,
                   threadIdx.y, nrows_x-row_x_0-1, threadIdx.x, blocks_per_row_x);

#pragma unroll
        for (int ir = 0; ir < qr; ++ir) {
            const int kqs = ir*WARP_SIZE + threadIdx.x;
            const int kbxd = kqs / QI8_1;

#pragma unroll
            for (int i = 0; i < mmq_x; i += nwarps) {
                const int col_y_eff = min(col_y_0 + threadIdx.y + i, ncols_y-1); // to prevent out-of-bounds memory accesses

                const block_q8_1 * by0 = &y[col_y_eff*blocks_per_col_y + ib0 * (qk/QK8_1) + kbxd];

                const int index_y = (threadIdx.y + i) * WARP_SIZE + kqs % WARP_SIZE;
                tile_y_qs[index_y] = get_int_from_int8_aligned(by0->qs, threadIdx.x % QI8_1);
            }

#pragma unroll
            for (int ids0 = 0; ids0 < mmq_x; ids0 += nwarps * QI8_1) {
                const int ids = (ids0 + threadIdx.y * QI8_1 + threadIdx.x / (WARP_SIZE/QI8_1)) % mmq_x;
                const int kby = threadIdx.x % (WARP_SIZE/QI8_1);
                const int col_y_eff = min(col_y_0 + ids, ncols_y-1);

                // if the sum is not needed it's faster to transform the scale to f32 ahead of time
                const half2 * dsi_src = &y[col_y_eff*blocks_per_col_y + ib0 * (qk/QK8_1) + ir*(WARP_SIZE/QI8_1) + kby].ds;
                half2       * dsi_dst = &tile_y_ds[ids * (WARP_SIZE/QI8_1) + kby];
                if (need_sum) {
                    *dsi_dst = *dsi_src;
                } else {
                    float * dfi_dst = (float *) dsi_dst;
                    *dfi_dst = __low2float(*dsi_src);
                }
            }

            __syncthreads();

// #pragma unroll // unrolling this loop causes too much register pressure
            for (int k = ir*WARP_SIZE/qr; k < (ir+1)*WARP_SIZE/qr; k += vdr) {
#pragma unroll
                for (int j = 0; j < mmq_x; j += nwarps) {
#pragma unroll
                    for (int i = 0; i < mmq_y; i += WARP_SIZE) {
                        sum[i/WARP_SIZE][j/nwarps] += vec_dot(
                            tile_x_ql, tile_x_dm, tile_x_qh, tile_x_sc, tile_y_qs, tile_y_ds,
                            threadIdx.x + i, threadIdx.y + j, k);
                    }
                }
            }

            __syncthreads();
        }
    }

#pragma unroll
    for (int j = 0; j < mmq_x; j += nwarps) {
        const int col_dst = col_dst_0 + j + threadIdx.y;

        if (col_dst >= ncols_dst) {
            return;
        }

#pragma unroll
        for (int i = 0; i < mmq_y; i += WARP_SIZE) {
            const int row_dst = row_dst_0 + threadIdx.x + i;

            if (row_dst >= nrows_dst) {
                continue;
            }

            dst[col_dst*nrows_dst + row_dst] = sum[i/WARP_SIZE][j/nwarps];
        }
    }
}

template <bool need_check> static __global__ void
#if defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)
#if defined(RDNA3) || defined(RDNA2)
    __launch_bounds__(WARP_SIZE*NWARPS_Q4_0_RDNA2, 2)
#endif // defined(RDNA3) || defined(RDNA2)
#endif // defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)
    mul_mat_q4_0(
    const void * __restrict__ vx, const void * __restrict__ vy, float * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int ncols_y, const int nrows_y, const int nrows_dst) {

#if defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)
#if defined(RDNA3) || defined(RDNA2)
    const int mmq_x  =  MMQ_X_Q4_0_RDNA2;
    const int mmq_y  =  MMQ_Y_Q4_0_RDNA2;
    const int nwarps = NWARPS_Q4_0_RDNA2;
#else
    const int mmq_x  =  MMQ_X_Q4_0_RDNA1;
    const int mmq_y  =  MMQ_Y_Q4_0_RDNA1;
    const int nwarps = NWARPS_Q4_0_RDNA1;
#endif // defined(RDNA3) || defined(RDNA2)

    mul_mat_q<QK4_0, QR4_0, QI4_0, true, block_q4_0, mmq_x, mmq_y, nwarps, allocate_tiles_q4_0<mmq_y>,
        load_tiles_q4_0<mmq_y, nwarps, need_check>, VDR_Q4_0_Q8_1_MMQ, vec_dot_q4_0_q8_1_mul_mat>
        (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);

#elif __CUDA_ARCH__ >= CC_VOLTA
    const int mmq_x  =  MMQ_X_Q4_0_AMPERE;
    const int mmq_y  =  MMQ_Y_Q4_0_AMPERE;
    const int nwarps = NWARPS_Q4_0_AMPERE;

    mul_mat_q<QK4_0, QR4_0, QI4_0, true, block_q4_0, mmq_x, mmq_y, nwarps, allocate_tiles_q4_0<mmq_y>,
        load_tiles_q4_0<mmq_y, nwarps, need_check>, VDR_Q4_0_Q8_1_MMQ, vec_dot_q4_0_q8_1_mul_mat>
        (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);

#elif __CUDA_ARCH__ >= MIN_CC_DP4A
    const int mmq_x  =  MMQ_X_Q4_0_PASCAL;
    const int mmq_y  =  MMQ_Y_Q4_0_PASCAL;
    const int nwarps = NWARPS_Q4_0_PASCAL;

    mul_mat_q<QK4_0, QR4_0, QI4_0, true, block_q4_0, mmq_x, mmq_y, nwarps, allocate_tiles_q4_0<mmq_y>,
        load_tiles_q4_0<mmq_y, nwarps, need_check>, VDR_Q4_0_Q8_1_MMQ, vec_dot_q4_0_q8_1_mul_mat>
        (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
#else
    GGML_UNUSED(vec_dot_q4_0_q8_1_mul_mat);
    NO_DEVICE_CODE;
#endif // __CUDA_ARCH__ >= CC_VOLTA
}

#define  MMQ_X_Q4_1_RDNA2  64
#define  MMQ_Y_Q4_1_RDNA2  128
#define NWARPS_Q4_1_RDNA2  8
#define  MMQ_X_Q4_1_RDNA1  64
#define  MMQ_Y_Q4_1_RDNA1  64
#define NWARPS_Q4_1_RDNA1  8
#if defined(CUDA_USE_TENSOR_CORES)
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

template <bool need_check> static __global__ void
#if defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)
#if defined(RDNA3) || defined(RDNA2)
    __launch_bounds__(WARP_SIZE*NWARPS_Q4_1_RDNA2, 2)
#endif // defined(RDNA3) || defined(RDNA2)
#elif __CUDA_ARCH__ < CC_VOLTA
    __launch_bounds__(WARP_SIZE*NWARPS_Q4_1_PASCAL, 2)
#endif // __CUDA_ARCH__ < CC_VOLTA
    mul_mat_q4_1(
    const void * __restrict__ vx, const void * __restrict__ vy, float * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int ncols_y, const int nrows_y, const int nrows_dst) {

#if defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)
#if defined(RDNA3) || defined(RDNA2)
    const int mmq_x  =  MMQ_X_Q4_1_RDNA2;
    const int mmq_y  =  MMQ_Y_Q4_1_RDNA2;
    const int nwarps = NWARPS_Q4_1_RDNA2;
#else
    const int mmq_x  =  MMQ_X_Q4_1_RDNA1;
    const int mmq_y  =  MMQ_Y_Q4_1_RDNA1;
    const int nwarps = NWARPS_Q4_1_RDNA1;
#endif // defined(RDNA3) || defined(RDNA2)

    mul_mat_q<QK4_1, QR4_1, QI4_1, true, block_q4_1, mmq_x, mmq_y, nwarps, allocate_tiles_q4_1<mmq_y>,
        load_tiles_q4_1<mmq_y, nwarps, need_check>, VDR_Q4_1_Q8_1_MMQ, vec_dot_q4_1_q8_1_mul_mat>
        (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);

#elif __CUDA_ARCH__ >= CC_VOLTA
    const int mmq_x  =  MMQ_X_Q4_1_AMPERE;
    const int mmq_y  =  MMQ_Y_Q4_1_AMPERE;
    const int nwarps = NWARPS_Q4_1_AMPERE;

    mul_mat_q<QK4_1, QR4_1, QI4_1, true, block_q4_1, mmq_x, mmq_y, nwarps, allocate_tiles_q4_1<mmq_y>,
        load_tiles_q4_1<mmq_y, nwarps, need_check>, VDR_Q4_1_Q8_1_MMQ, vec_dot_q4_1_q8_1_mul_mat>
        (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);

#elif __CUDA_ARCH__ >= MIN_CC_DP4A
    const int mmq_x  =  MMQ_X_Q4_1_PASCAL;
    const int mmq_y  =  MMQ_Y_Q4_1_PASCAL;
    const int nwarps = NWARPS_Q4_1_PASCAL;

    mul_mat_q<QK4_1, QR4_1, QI4_1, true, block_q4_1, mmq_x, mmq_y, nwarps, allocate_tiles_q4_1<mmq_y>,
        load_tiles_q4_1<mmq_y, nwarps, need_check>, VDR_Q4_1_Q8_1_MMQ, vec_dot_q4_1_q8_1_mul_mat>
        (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
#else
    GGML_UNUSED(vec_dot_q4_1_q8_1_mul_mat);
    NO_DEVICE_CODE;
#endif // __CUDA_ARCH__ >= CC_VOLTA
}

#define  MMQ_X_Q5_0_RDNA2  64
#define  MMQ_Y_Q5_0_RDNA2  128
#define NWARPS_Q5_0_RDNA2  8
#define  MMQ_X_Q5_0_RDNA1  64
#define  MMQ_Y_Q5_0_RDNA1  64
#define NWARPS_Q5_0_RDNA1  8
#if defined(CUDA_USE_TENSOR_CORES)
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

template <bool need_check> static __global__ void
#if defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)
#if defined(RDNA3) || defined(RDNA2)
    __launch_bounds__(WARP_SIZE*NWARPS_Q5_0_RDNA2, 2)
#endif // defined(RDNA3) || defined(RDNA2)
#endif // defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)
    mul_mat_q5_0(
    const void * __restrict__ vx, const void * __restrict__ vy, float * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int ncols_y, const int nrows_y, const int nrows_dst) {

#if defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)
#if defined(RDNA3) || defined(RDNA2)
    const int mmq_x  =  MMQ_X_Q5_0_RDNA2;
    const int mmq_y  =  MMQ_Y_Q5_0_RDNA2;
    const int nwarps = NWARPS_Q5_0_RDNA2;
#else
    const int mmq_x  =  MMQ_X_Q5_0_RDNA1;
    const int mmq_y  =  MMQ_Y_Q5_0_RDNA1;
    const int nwarps = NWARPS_Q5_0_RDNA1;
#endif // defined(RDNA3) || defined(RDNA2)

    mul_mat_q<QK5_0, QR5_0, QI5_0, false, block_q5_0, mmq_x, mmq_y, nwarps, allocate_tiles_q5_0<mmq_y>,
        load_tiles_q5_0<mmq_y, nwarps, need_check>, VDR_Q5_0_Q8_1_MMQ, vec_dot_q5_0_q8_1_mul_mat>
        (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);

#elif __CUDA_ARCH__ >= CC_VOLTA
    const int mmq_x  =  MMQ_X_Q5_0_AMPERE;
    const int mmq_y  =  MMQ_Y_Q5_0_AMPERE;
    const int nwarps = NWARPS_Q5_0_AMPERE;

    mul_mat_q<QK5_0, QR5_0, QI5_0, false, block_q5_0, mmq_x, mmq_y, nwarps, allocate_tiles_q5_0<mmq_y>,
        load_tiles_q5_0<mmq_y, nwarps, need_check>, VDR_Q5_0_Q8_1_MMQ, vec_dot_q5_0_q8_1_mul_mat>
        (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);

#elif __CUDA_ARCH__ >= MIN_CC_DP4A
    const int mmq_x  =  MMQ_X_Q5_0_PASCAL;
    const int mmq_y  =  MMQ_Y_Q5_0_PASCAL;
    const int nwarps = NWARPS_Q5_0_PASCAL;

    mul_mat_q<QK5_0, QR5_0, QI5_0, false, block_q5_0, mmq_x, mmq_y, nwarps, allocate_tiles_q5_0<mmq_y>,
        load_tiles_q5_0<mmq_y, nwarps, need_check>, VDR_Q5_0_Q8_1_MMQ, vec_dot_q5_0_q8_1_mul_mat>
        (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
#else
    GGML_UNUSED(vec_dot_q5_0_q8_1_mul_mat);
    NO_DEVICE_CODE;
#endif // __CUDA_ARCH__ >= CC_VOLTA
}

#define  MMQ_X_Q5_1_RDNA2  64
#define  MMQ_Y_Q5_1_RDNA2  128
#define NWARPS_Q5_1_RDNA2  8
#define  MMQ_X_Q5_1_RDNA1  64
#define  MMQ_Y_Q5_1_RDNA1  64
#define NWARPS_Q5_1_RDNA1  8
#if defined(CUDA_USE_TENSOR_CORES)
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

template <bool need_check> static __global__ void
#if defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)
#if defined(RDNA3) || defined(RDNA2)
    __launch_bounds__(WARP_SIZE*NWARPS_Q5_1_RDNA2, 2)
#endif // defined(RDNA3) || defined(RDNA2)
#endif // defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)
mul_mat_q5_1(
    const void * __restrict__ vx, const void * __restrict__ vy, float * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int ncols_y, const int nrows_y, const int nrows_dst) {

#if defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)
#if defined(RDNA3) || defined(RDNA2)
    const int mmq_x  =  MMQ_X_Q5_1_RDNA2;
    const int mmq_y  =  MMQ_Y_Q5_1_RDNA2;
    const int nwarps = NWARPS_Q5_1_RDNA2;
#else
    const int mmq_x  =  MMQ_X_Q5_1_RDNA1;
    const int mmq_y  =  MMQ_Y_Q5_1_RDNA1;
    const int nwarps = NWARPS_Q5_1_RDNA1;
#endif // defined(RDNA3) || defined(RDNA2)

    mul_mat_q<QK5_1, QR5_1, QI5_1, true, block_q5_1, mmq_x, mmq_y, nwarps, allocate_tiles_q5_1<mmq_y>,
        load_tiles_q5_1<mmq_y, nwarps, need_check>, VDR_Q5_1_Q8_1_MMQ, vec_dot_q5_1_q8_1_mul_mat>
        (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);

#elif __CUDA_ARCH__ >= CC_VOLTA
    const int mmq_x  =  MMQ_X_Q5_1_AMPERE;
    const int mmq_y  =  MMQ_Y_Q5_1_AMPERE;
    const int nwarps = NWARPS_Q5_1_AMPERE;

    mul_mat_q<QK5_1, QR5_1, QI5_1, true, block_q5_1, mmq_x, mmq_y, nwarps, allocate_tiles_q5_1<mmq_y>,
        load_tiles_q5_1<mmq_y, nwarps, need_check>, VDR_Q5_1_Q8_1_MMQ, vec_dot_q5_1_q8_1_mul_mat>
        (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);

#elif __CUDA_ARCH__ >= MIN_CC_DP4A
    const int mmq_x  =  MMQ_X_Q5_1_PASCAL;
    const int mmq_y  =  MMQ_Y_Q5_1_PASCAL;
    const int nwarps = NWARPS_Q5_1_PASCAL;

    mul_mat_q<QK5_1, QR5_1, QI5_1, true, block_q5_1, mmq_x, mmq_y, nwarps, allocate_tiles_q5_1<mmq_y>,
        load_tiles_q5_1<mmq_y, nwarps, need_check>, VDR_Q5_1_Q8_1_MMQ, vec_dot_q5_1_q8_1_mul_mat>
        (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
#else
    GGML_UNUSED(vec_dot_q5_1_q8_1_mul_mat);
    NO_DEVICE_CODE;
#endif // __CUDA_ARCH__ >= CC_VOLTA
}

#define  MMQ_X_Q8_0_RDNA2  64
#define  MMQ_Y_Q8_0_RDNA2  128
#define NWARPS_Q8_0_RDNA2  8
#define  MMQ_X_Q8_0_RDNA1  64
#define  MMQ_Y_Q8_0_RDNA1  64
#define NWARPS_Q8_0_RDNA1  8
#if defined(CUDA_USE_TENSOR_CORES)
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

template <bool need_check> static __global__ void
#if defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)
#if defined(RDNA3) || defined(RDNA2)
    __launch_bounds__(WARP_SIZE*NWARPS_Q8_0_RDNA2, 2)
#endif // defined(RDNA3) || defined(RDNA2)
#endif // defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)
    mul_mat_q8_0(
    const void * __restrict__ vx, const void * __restrict__ vy, float * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int ncols_y, const int nrows_y, const int nrows_dst) {

#if defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)
#if defined(RDNA3) || defined(RDNA2)
    const int mmq_x  =  MMQ_X_Q8_0_RDNA2;
    const int mmq_y  =  MMQ_Y_Q8_0_RDNA2;
    const int nwarps = NWARPS_Q8_0_RDNA2;
#else
    const int mmq_x  =  MMQ_X_Q8_0_RDNA1;
    const int mmq_y  =  MMQ_Y_Q8_0_RDNA1;
    const int nwarps = NWARPS_Q8_0_RDNA1;
#endif // defined(RDNA3) || defined(RDNA2)

    mul_mat_q<QK8_0, QR8_0, QI8_0, false, block_q8_0, mmq_x, mmq_y, nwarps, allocate_tiles_q8_0<mmq_y>,
        load_tiles_q8_0<mmq_y, nwarps, need_check>, VDR_Q8_0_Q8_1_MMQ, vec_dot_q8_0_q8_1_mul_mat>
        (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);

#elif __CUDA_ARCH__ >= CC_VOLTA
    const int mmq_x  =  MMQ_X_Q8_0_AMPERE;
    const int mmq_y  =  MMQ_Y_Q8_0_AMPERE;
    const int nwarps = NWARPS_Q8_0_AMPERE;

    mul_mat_q<QK8_0, QR8_0, QI8_0, false, block_q8_0, mmq_x, mmq_y, nwarps, allocate_tiles_q8_0<mmq_y>,
        load_tiles_q8_0<mmq_y, nwarps, need_check>, VDR_Q8_0_Q8_1_MMQ, vec_dot_q8_0_q8_1_mul_mat>
        (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);

#elif __CUDA_ARCH__ >= MIN_CC_DP4A
    const int mmq_x  =  MMQ_X_Q8_0_PASCAL;
    const int mmq_y  =  MMQ_Y_Q8_0_PASCAL;
    const int nwarps = NWARPS_Q8_0_PASCAL;

    mul_mat_q<QK8_0, QR8_0, QI8_0, false, block_q8_0, mmq_x, mmq_y, nwarps, allocate_tiles_q8_0<mmq_y>,
        load_tiles_q8_0<mmq_y, nwarps, need_check>, VDR_Q8_0_Q8_1_MMQ, vec_dot_q8_0_q8_1_mul_mat>
        (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
#else
    GGML_UNUSED(vec_dot_q8_0_q8_1_mul_mat);
    NO_DEVICE_CODE;
#endif // __CUDA_ARCH__ >= CC_VOLTA
}

#define  MMQ_X_Q2_K_RDNA2  64
#define  MMQ_Y_Q2_K_RDNA2  128
#define NWARPS_Q2_K_RDNA2  8
#define  MMQ_X_Q2_K_RDNA1  128
#define  MMQ_Y_Q2_K_RDNA1  32
#define NWARPS_Q2_K_RDNA1  8
#if defined(CUDA_USE_TENSOR_CORES)
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

template <bool need_check> static __global__ void
#if defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)
#if defined(RDNA3) || defined(RDNA2)
    __launch_bounds__(WARP_SIZE*NWARPS_Q2_K_RDNA2, 2)
#endif // defined(RDNA3) || defined(RDNA2)
#endif // defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)
mul_mat_q2_K(
    const void * __restrict__ vx, const void * __restrict__ vy, float * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int ncols_y, const int nrows_y, const int nrows_dst) {

#if defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)
#if defined(RDNA3) || defined(RDNA2)
    const int mmq_x  =  MMQ_X_Q2_K_RDNA2;
    const int mmq_y  =  MMQ_Y_Q2_K_RDNA2;
    const int nwarps = NWARPS_Q2_K_RDNA2;
#else
    const int mmq_x  =  MMQ_X_Q2_K_RDNA1;
    const int mmq_y  =  MMQ_Y_Q2_K_RDNA1;
    const int nwarps = NWARPS_Q2_K_RDNA1;
#endif // defined(RDNA3) || defined(RDNA2)

    mul_mat_q<QK_K, QR2_K, QI2_K, false, block_q2_K, mmq_x, mmq_y, nwarps, allocate_tiles_q2_K<mmq_y>,
        load_tiles_q2_K<mmq_y, nwarps, need_check>, VDR_Q2_K_Q8_1_MMQ, vec_dot_q2_K_q8_1_mul_mat>
        (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);

#elif __CUDA_ARCH__ >= CC_VOLTA
    const int mmq_x  =  MMQ_X_Q2_K_AMPERE;
    const int mmq_y  =  MMQ_Y_Q2_K_AMPERE;
    const int nwarps = NWARPS_Q2_K_AMPERE;

    mul_mat_q<QK_K, QR2_K, QI2_K, false, block_q2_K, mmq_x, mmq_y, nwarps, allocate_tiles_q2_K<mmq_y>,
        load_tiles_q2_K<mmq_y, nwarps, need_check>, VDR_Q2_K_Q8_1_MMQ, vec_dot_q2_K_q8_1_mul_mat>
        (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);

#elif __CUDA_ARCH__ >= MIN_CC_DP4A
    const int mmq_x  =  MMQ_X_Q2_K_PASCAL;
    const int mmq_y  =  MMQ_Y_Q2_K_PASCAL;
    const int nwarps = NWARPS_Q2_K_PASCAL;

    mul_mat_q<QK_K, QR2_K, QI2_K, false, block_q2_K, mmq_x, mmq_y, nwarps, allocate_tiles_q2_K<mmq_y>,
        load_tiles_q2_K<mmq_y, nwarps, need_check>, VDR_Q2_K_Q8_1_MMQ, vec_dot_q2_K_q8_1_mul_mat>
        (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
#else
    GGML_UNUSED(vec_dot_q2_K_q8_1_mul_mat);
    NO_DEVICE_CODE;
#endif // __CUDA_ARCH__ >= CC_VOLTA
}

#define  MMQ_X_Q3_K_RDNA2  128
#define  MMQ_Y_Q3_K_RDNA2  64
#define NWARPS_Q3_K_RDNA2  8
#define  MMQ_X_Q3_K_RDNA1  32
#define  MMQ_Y_Q3_K_RDNA1  128
#define NWARPS_Q3_K_RDNA1  8
#if defined(CUDA_USE_TENSOR_CORES)
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

template <bool need_check> static __global__ void
#if defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)
#if defined(RDNA3) || defined(RDNA2)
    __launch_bounds__(WARP_SIZE*NWARPS_Q3_K_RDNA2, 2)
#endif // defined(RDNA3) || defined(RDNA2)
#elif __CUDA_ARCH__ < CC_VOLTA
    __launch_bounds__(WARP_SIZE*NWARPS_Q3_K_PASCAL, 2)
#endif // __CUDA_ARCH__ < CC_VOLTA
    mul_mat_q3_K(
    const void * __restrict__ vx, const void * __restrict__ vy, float * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int ncols_y, const int nrows_y, const int nrows_dst) {

#if defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)
#if defined(RDNA3) || defined(RDNA2)
    const int mmq_x  =  MMQ_X_Q3_K_RDNA2;
    const int mmq_y  =  MMQ_Y_Q3_K_RDNA2;
    const int nwarps = NWARPS_Q3_K_RDNA2;
#else
    const int mmq_x  =  MMQ_X_Q3_K_RDNA1;
    const int mmq_y  =  MMQ_Y_Q3_K_RDNA1;
    const int nwarps = NWARPS_Q3_K_RDNA1;
#endif // defined(RDNA3) || defined(RDNA2)

    mul_mat_q<QK_K, QR3_K, QI3_K, false, block_q3_K, mmq_x, mmq_y, nwarps, allocate_tiles_q3_K<mmq_y>,
        load_tiles_q3_K<mmq_y, nwarps, need_check>, VDR_Q3_K_Q8_1_MMQ, vec_dot_q3_K_q8_1_mul_mat>
        (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);

#elif __CUDA_ARCH__ >= CC_VOLTA
    const int mmq_x  =  MMQ_X_Q3_K_AMPERE;
    const int mmq_y  =  MMQ_Y_Q3_K_AMPERE;
    const int nwarps = NWARPS_Q3_K_AMPERE;

    mul_mat_q<QK_K, QR3_K, QI3_K, false, block_q3_K, mmq_x, mmq_y, nwarps, allocate_tiles_q3_K<mmq_y>,
        load_tiles_q3_K<mmq_y, nwarps, need_check>, VDR_Q3_K_Q8_1_MMQ, vec_dot_q3_K_q8_1_mul_mat>
        (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);

#elif __CUDA_ARCH__ >= MIN_CC_DP4A
    const int mmq_x  =  MMQ_X_Q3_K_PASCAL;
    const int mmq_y  =  MMQ_Y_Q3_K_PASCAL;
    const int nwarps = NWARPS_Q3_K_PASCAL;

    mul_mat_q<QK_K, QR3_K, QI3_K, false, block_q3_K, mmq_x, mmq_y, nwarps, allocate_tiles_q3_K<mmq_y>,
        load_tiles_q3_K<mmq_y, nwarps, need_check>, VDR_Q3_K_Q8_1_MMQ, vec_dot_q3_K_q8_1_mul_mat>
        (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
#else
    GGML_UNUSED(vec_dot_q3_K_q8_1_mul_mat);
    NO_DEVICE_CODE;
#endif // __CUDA_ARCH__ >= CC_VOLTA
}

#define  MMQ_X_Q4_K_RDNA2  64
#define  MMQ_Y_Q4_K_RDNA2  128
#define NWARPS_Q4_K_RDNA2  8
#define  MMQ_X_Q4_K_RDNA1  32
#define  MMQ_Y_Q4_K_RDNA1  64
#define NWARPS_Q4_K_RDNA1  8
#if defined(CUDA_USE_TENSOR_CORES)
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

template <bool need_check> static __global__ void
#if defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)
#if defined(RDNA3) || defined(RDNA2)
    __launch_bounds__(WARP_SIZE*NWARPS_Q4_K_RDNA2, 2)
#endif // defined(RDNA3) || defined(RDNA2)
#elif __CUDA_ARCH__ < CC_VOLTA
    __launch_bounds__(WARP_SIZE*NWARPS_Q4_K_PASCAL, 2)
#endif // __CUDA_ARCH__ < CC_VOLTA
    mul_mat_q4_K(
    const void * __restrict__ vx, const void * __restrict__ vy, float * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int ncols_y, const int nrows_y, const int nrows_dst) {

#if defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)
#if defined(RDNA3) || defined(RDNA2)
    const int mmq_x  =  MMQ_X_Q4_K_RDNA2;
    const int mmq_y  =  MMQ_Y_Q4_K_RDNA2;
    const int nwarps = NWARPS_Q4_K_RDNA2;
#else
    const int mmq_x  =  MMQ_X_Q4_K_RDNA1;
    const int mmq_y  =  MMQ_Y_Q4_K_RDNA1;
    const int nwarps = NWARPS_Q4_K_RDNA1;
#endif // defined(RDNA3) || defined(RDNA2)

    mul_mat_q<QK_K, QR4_K, QI4_K, true, block_q4_K, mmq_x, mmq_y, nwarps, allocate_tiles_q4_K<mmq_y>,
        load_tiles_q4_K<mmq_y, nwarps, need_check>, VDR_Q4_K_Q8_1_MMQ, vec_dot_q4_K_q8_1_mul_mat>
        (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);

#elif __CUDA_ARCH__ >= CC_VOLTA
    const int mmq_x  =  MMQ_X_Q4_K_AMPERE;
    const int mmq_y  =  MMQ_Y_Q4_K_AMPERE;
    const int nwarps = NWARPS_Q4_K_AMPERE;

    mul_mat_q<QK_K, QR4_K, QI4_K, true, block_q4_K, mmq_x, mmq_y, nwarps, allocate_tiles_q4_K<mmq_y>,
        load_tiles_q4_K<mmq_y, nwarps, need_check>, VDR_Q4_K_Q8_1_MMQ, vec_dot_q4_K_q8_1_mul_mat>
        (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);

#elif __CUDA_ARCH__ >= MIN_CC_DP4A
    const int mmq_x  =  MMQ_X_Q4_K_PASCAL;
    const int mmq_y  =  MMQ_Y_Q4_K_PASCAL;
    const int nwarps = NWARPS_Q4_K_PASCAL;

    mul_mat_q<QK_K, QR4_K, QI4_K, true, block_q4_K, mmq_x, mmq_y, nwarps, allocate_tiles_q4_K<mmq_y>,
        load_tiles_q4_K<mmq_y, nwarps, need_check>, VDR_Q4_K_Q8_1_MMQ, vec_dot_q4_K_q8_1_mul_mat>
        (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
#else
    GGML_UNUSED(vec_dot_q4_K_q8_1_mul_mat);
    NO_DEVICE_CODE;
#endif // __CUDA_ARCH__ >= CC_VOLTA
}

#define  MMQ_X_Q5_K_RDNA2  64
#define  MMQ_Y_Q5_K_RDNA2  128
#define NWARPS_Q5_K_RDNA2  8
#define  MMQ_X_Q5_K_RDNA1  32
#define  MMQ_Y_Q5_K_RDNA1  64
#define NWARPS_Q5_K_RDNA1  8
#if defined(CUDA_USE_TENSOR_CORES)
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

template <bool need_check> static __global__ void
#if defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)
#if defined(RDNA3) || defined(RDNA2)
    __launch_bounds__(WARP_SIZE*NWARPS_Q5_K_RDNA2, 2)
#endif // defined(RDNA3) || defined(RDNA2)
#endif // defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)
mul_mat_q5_K(
    const void * __restrict__ vx, const void * __restrict__ vy, float * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int ncols_y, const int nrows_y, const int nrows_dst) {

#if defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)
#if defined(RDNA3) || defined(RDNA2)
    const int mmq_x  =  MMQ_X_Q5_K_RDNA2;
    const int mmq_y  =  MMQ_Y_Q5_K_RDNA2;
    const int nwarps = NWARPS_Q5_K_RDNA2;
#else
    const int mmq_x  =  MMQ_X_Q5_K_RDNA1;
    const int mmq_y  =  MMQ_Y_Q5_K_RDNA1;
    const int nwarps = NWARPS_Q5_K_RDNA1;
#endif // defined(RDNA3) || defined(RDNA2)

    mul_mat_q<QK_K, QR5_K, QI5_K, true, block_q5_K, mmq_x, mmq_y, nwarps, allocate_tiles_q5_K<mmq_y>,
        load_tiles_q5_K<mmq_y, nwarps, need_check>, VDR_Q5_K_Q8_1_MMQ, vec_dot_q5_K_q8_1_mul_mat>
        (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);

#elif __CUDA_ARCH__ >= CC_VOLTA
    const int mmq_x  =  MMQ_X_Q5_K_AMPERE;
    const int mmq_y  =  MMQ_Y_Q5_K_AMPERE;
    const int nwarps = NWARPS_Q5_K_AMPERE;

    mul_mat_q<QK_K, QR5_K, QI5_K, true, block_q5_K, mmq_x, mmq_y, nwarps, allocate_tiles_q5_K<mmq_y>,
        load_tiles_q5_K<mmq_y, nwarps, need_check>, VDR_Q5_K_Q8_1_MMQ, vec_dot_q5_K_q8_1_mul_mat>
        (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);

#elif __CUDA_ARCH__ >= MIN_CC_DP4A
    const int mmq_x  =  MMQ_X_Q5_K_PASCAL;
    const int mmq_y  =  MMQ_Y_Q5_K_PASCAL;
    const int nwarps = NWARPS_Q5_K_PASCAL;

    mul_mat_q<QK_K, QR5_K, QI5_K, true, block_q5_K, mmq_x, mmq_y, nwarps, allocate_tiles_q5_K<mmq_y>,
        load_tiles_q5_K<mmq_y, nwarps, need_check>, VDR_Q5_K_Q8_1_MMQ, vec_dot_q5_K_q8_1_mul_mat>
        (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
#else
    GGML_UNUSED(vec_dot_q5_K_q8_1_mul_mat);
    NO_DEVICE_CODE;
#endif // __CUDA_ARCH__ >= CC_VOLTA
}

#define  MMQ_X_Q6_K_RDNA2  64
#define  MMQ_Y_Q6_K_RDNA2  128
#define NWARPS_Q6_K_RDNA2  8
#define  MMQ_X_Q6_K_RDNA1  32
#define  MMQ_Y_Q6_K_RDNA1  64
#define NWARPS_Q6_K_RDNA1  8
#if defined(CUDA_USE_TENSOR_CORES)
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

template <bool need_check> static __global__ void
#if defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)
#if defined(RDNA3) || defined(RDNA2)
    __launch_bounds__(WARP_SIZE*NWARPS_Q6_K_RDNA2, 2)
#endif // defined(RDNA3) || defined(RDNA2)
#elif __CUDA_ARCH__ < CC_VOLTA
    __launch_bounds__(WARP_SIZE*NWARPS_Q6_K_PASCAL, 2)
#endif // __CUDA_ARCH__ < CC_VOLTA
    mul_mat_q6_K(
    const void * __restrict__ vx, const void * __restrict__ vy, float * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int ncols_y, const int nrows_y, const int nrows_dst) {

#if defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)
#if defined(RDNA3) || defined(RDNA2)
    const int mmq_x  =  MMQ_X_Q6_K_RDNA2;
    const int mmq_y  =  MMQ_Y_Q6_K_RDNA2;
    const int nwarps = NWARPS_Q6_K_RDNA2;
#else
    const int mmq_x  =  MMQ_X_Q6_K_RDNA1;
    const int mmq_y  =  MMQ_Y_Q6_K_RDNA1;
    const int nwarps = NWARPS_Q6_K_RDNA1;
#endif // defined(RDNA3) || defined(RDNA2)

    mul_mat_q<QK_K, QR6_K, QI6_K, false, block_q6_K, mmq_x, mmq_y, nwarps, allocate_tiles_q6_K<mmq_y>,
        load_tiles_q6_K<mmq_y, nwarps, need_check>, VDR_Q6_K_Q8_1_MMQ, vec_dot_q6_K_q8_1_mul_mat>
        (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);

#elif __CUDA_ARCH__ >= CC_VOLTA
    const int mmq_x  =  MMQ_X_Q6_K_AMPERE;
    const int mmq_y  =  MMQ_Y_Q6_K_AMPERE;
    const int nwarps = NWARPS_Q6_K_AMPERE;

    mul_mat_q<QK_K, QR6_K, QI6_K, false, block_q6_K, mmq_x, mmq_y, nwarps, allocate_tiles_q6_K<mmq_y>,
        load_tiles_q6_K<mmq_y, nwarps, need_check>, VDR_Q6_K_Q8_1_MMQ, vec_dot_q6_K_q8_1_mul_mat>
        (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);

#elif __CUDA_ARCH__ >= MIN_CC_DP4A
    const int mmq_x  =  MMQ_X_Q6_K_PASCAL;
    const int mmq_y  =  MMQ_Y_Q6_K_PASCAL;
    const int nwarps = NWARPS_Q6_K_PASCAL;

    mul_mat_q<QK_K, QR6_K, QI6_K, false, block_q6_K, mmq_x, mmq_y, nwarps, allocate_tiles_q6_K<mmq_y>,
        load_tiles_q6_K<mmq_y, nwarps, need_check>, VDR_Q6_K_Q8_1_MMQ, vec_dot_q6_K_q8_1_mul_mat>
        (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
#else
    GGML_UNUSED(vec_dot_q6_K_q8_1_mul_mat);
    NO_DEVICE_CODE;
#endif // __CUDA_ARCH__ >= CC_VOLTA
}

static void ggml_mul_mat_q4_0_q8_1_cuda(
    const void * vx, const void * vy, float * dst, const int ncols_x, const int nrows_x,
    const int ncols_y, const int nrows_y, const int nrows_dst, cudaStream_t stream) {

    int id = ggml_cuda_get_device();
    const int compute_capability = ggml_cuda_info().devices[id].cc;

    int mmq_x, mmq_y, nwarps;
    if (compute_capability >= CC_RDNA2) {
        mmq_x  =  MMQ_X_Q4_0_RDNA2;
        mmq_y  =  MMQ_Y_Q4_0_RDNA2;
        nwarps = NWARPS_Q4_0_RDNA2;
    } else if (compute_capability >= CC_OFFSET_AMD) {
        mmq_x  =  MMQ_X_Q4_0_RDNA1;
        mmq_y  =  MMQ_Y_Q4_0_RDNA1;
        nwarps = NWARPS_Q4_0_RDNA1;
    } else if (compute_capability >= CC_VOLTA) {
        mmq_x  =  MMQ_X_Q4_0_AMPERE;
        mmq_y  =  MMQ_Y_Q4_0_AMPERE;
        nwarps = NWARPS_Q4_0_AMPERE;
    } else if (compute_capability >= MIN_CC_DP4A) {
        mmq_x  =  MMQ_X_Q4_0_PASCAL;
        mmq_y  =  MMQ_Y_Q4_0_PASCAL;
        nwarps = NWARPS_Q4_0_PASCAL;
    } else {
        GGML_ASSERT(false);
    }

    const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
    const int block_num_y = (ncols_y + mmq_x - 1) / mmq_x;
    const dim3 block_nums(block_num_x, block_num_y, 1);
    const dim3 block_dims(WARP_SIZE, nwarps, 1);

    if (nrows_x % mmq_y == 0) {
        const bool need_check = false;
        mul_mat_q4_0<need_check><<<block_nums, block_dims, 0, stream>>>
            (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
    } else {
        const bool need_check = true;
        mul_mat_q4_0<need_check><<<block_nums, block_dims, 0, stream>>>
            (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
    }
}

static void ggml_mul_mat_q4_1_q8_1_cuda(
    const void * vx, const void * vy, float * dst, const int ncols_x, const int nrows_x,
    const int ncols_y, const int nrows_y, const int nrows_dst, cudaStream_t stream) {

    int id = ggml_cuda_get_device();
    const int compute_capability = ggml_cuda_info().devices[id].cc;

    int mmq_x, mmq_y, nwarps;
    if (compute_capability >= CC_RDNA2) {
        mmq_x  =  MMQ_X_Q4_1_RDNA2;
        mmq_y  =  MMQ_Y_Q4_1_RDNA2;
        nwarps = NWARPS_Q4_1_RDNA2;
    } else if (compute_capability >= CC_OFFSET_AMD) {
        mmq_x  =  MMQ_X_Q4_1_RDNA1;
        mmq_y  =  MMQ_Y_Q4_1_RDNA1;
        nwarps = NWARPS_Q4_1_RDNA1;
    } else if (compute_capability >= CC_VOLTA) {
        mmq_x  =  MMQ_X_Q4_1_AMPERE;
        mmq_y  =  MMQ_Y_Q4_1_AMPERE;
        nwarps = NWARPS_Q4_1_AMPERE;
    } else if (compute_capability >= MIN_CC_DP4A) {
        mmq_x  =  MMQ_X_Q4_1_PASCAL;
        mmq_y  =  MMQ_Y_Q4_1_PASCAL;
        nwarps = NWARPS_Q4_1_PASCAL;
    } else {
        GGML_ASSERT(false);
    }

    const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
    const int block_num_y = (ncols_y + mmq_x - 1) / mmq_x;
    const dim3 block_nums(block_num_x, block_num_y, 1);
    const dim3 block_dims(WARP_SIZE, nwarps, 1);

    if (nrows_x % mmq_y == 0) {
        const bool need_check = false;
        mul_mat_q4_1<need_check><<<block_nums, block_dims, 0, stream>>>
            (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
    } else {
        const bool need_check = true;
        mul_mat_q4_1<need_check><<<block_nums, block_dims, 0, stream>>>
            (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
    }
}

static void ggml_mul_mat_q5_0_q8_1_cuda(
    const void * vx, const void * vy, float * dst, const int ncols_x, const int nrows_x,
    const int ncols_y, const int nrows_y, const int nrows_dst, cudaStream_t stream) {

    int id = ggml_cuda_get_device();
    const int compute_capability = ggml_cuda_info().devices[id].cc;

    int mmq_x, mmq_y, nwarps;
    if (compute_capability >= CC_RDNA2) {
        mmq_x  =  MMQ_X_Q5_0_RDNA2;
        mmq_y  =  MMQ_Y_Q5_0_RDNA2;
        nwarps = NWARPS_Q5_0_RDNA2;
    } else if (compute_capability >= CC_OFFSET_AMD) {
        mmq_x  =  MMQ_X_Q5_0_RDNA1;
        mmq_y  =  MMQ_Y_Q5_0_RDNA1;
        nwarps = NWARPS_Q5_0_RDNA1;
    } else if (compute_capability >= CC_VOLTA) {
        mmq_x  =  MMQ_X_Q5_0_AMPERE;
        mmq_y  =  MMQ_Y_Q5_0_AMPERE;
        nwarps = NWARPS_Q5_0_AMPERE;
    } else if (compute_capability >= MIN_CC_DP4A) {
        mmq_x  =  MMQ_X_Q5_0_PASCAL;
        mmq_y  =  MMQ_Y_Q5_0_PASCAL;
        nwarps = NWARPS_Q5_0_PASCAL;
    } else {
        GGML_ASSERT(false);
    }

    const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
    const int block_num_y = (ncols_y + mmq_x - 1) / mmq_x;
    const dim3 block_nums(block_num_x, block_num_y, 1);
    const dim3 block_dims(WARP_SIZE, nwarps, 1);

    if (nrows_x % mmq_y == 0) {
        const bool need_check = false;
        mul_mat_q5_0<need_check><<<block_nums, block_dims, 0, stream>>>
            (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
    } else {
        const bool need_check = true;
        mul_mat_q5_0<need_check><<<block_nums, block_dims, 0, stream>>>
            (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
    }
}

static void ggml_mul_mat_q5_1_q8_1_cuda(
    const void * vx, const void * vy, float * dst, const int ncols_x, const int nrows_x,
    const int ncols_y, const int nrows_y, const int nrows_dst, cudaStream_t stream) {

    int id = ggml_cuda_get_device();
    const int compute_capability = ggml_cuda_info().devices[id].cc;

    int mmq_x, mmq_y, nwarps;
    if (compute_capability >= CC_RDNA2) {
        mmq_x  =  MMQ_X_Q5_1_RDNA2;
        mmq_y  =  MMQ_Y_Q5_1_RDNA2;
        nwarps = NWARPS_Q5_1_RDNA2;
    } else if (compute_capability >= CC_OFFSET_AMD) {
        mmq_x  =  MMQ_X_Q5_1_RDNA1;
        mmq_y  =  MMQ_Y_Q5_1_RDNA1;
        nwarps = NWARPS_Q5_1_RDNA1;
    } else if (compute_capability >= CC_VOLTA) {
        mmq_x  =  MMQ_X_Q5_1_AMPERE;
        mmq_y  =  MMQ_Y_Q5_1_AMPERE;
        nwarps = NWARPS_Q5_1_AMPERE;
    } else if (compute_capability >= MIN_CC_DP4A) {
        mmq_x  =  MMQ_X_Q5_1_PASCAL;
        mmq_y  =  MMQ_Y_Q5_1_PASCAL;
        nwarps = NWARPS_Q5_1_PASCAL;
    } else {
        GGML_ASSERT(false);
    }

    const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
    const int block_num_y = (ncols_y + mmq_x - 1) / mmq_x;
    const dim3 block_nums(block_num_x, block_num_y, 1);
    const dim3 block_dims(WARP_SIZE, nwarps, 1);

    if (nrows_x % mmq_y == 0) {
        const bool need_check = false;
        mul_mat_q5_1<need_check><<<block_nums, block_dims, 0, stream>>>
            (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
    } else {
        const bool need_check = true;
        mul_mat_q5_1<need_check><<<block_nums, block_dims, 0, stream>>>
            (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
    }
}

static void ggml_mul_mat_q8_0_q8_1_cuda(
    const void * vx, const void * vy, float * dst, const int ncols_x, const int nrows_x,
    const int ncols_y, const int nrows_y, const int nrows_dst, cudaStream_t stream) {

    int id = ggml_cuda_get_device();
    const int compute_capability = ggml_cuda_info().devices[id].cc;

    int mmq_x, mmq_y, nwarps;
    if (compute_capability >= CC_RDNA2) {
        mmq_x  =  MMQ_X_Q8_0_RDNA2;
        mmq_y  =  MMQ_Y_Q8_0_RDNA2;
        nwarps = NWARPS_Q8_0_RDNA2;
    } else if (compute_capability >= CC_OFFSET_AMD) {
        mmq_x  =  MMQ_X_Q8_0_RDNA1;
        mmq_y  =  MMQ_Y_Q8_0_RDNA1;
        nwarps = NWARPS_Q8_0_RDNA1;
    } else if (compute_capability >= CC_VOLTA) {
        mmq_x  =  MMQ_X_Q8_0_AMPERE;
        mmq_y  =  MMQ_Y_Q8_0_AMPERE;
        nwarps = NWARPS_Q8_0_AMPERE;
    } else if (compute_capability >= MIN_CC_DP4A) {
        mmq_x  =  MMQ_X_Q8_0_PASCAL;
        mmq_y  =  MMQ_Y_Q8_0_PASCAL;
        nwarps = NWARPS_Q8_0_PASCAL;
    } else {
        GGML_ASSERT(false);
    }

    const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
    const int block_num_y = (ncols_y + mmq_x - 1) / mmq_x;
    const dim3 block_nums(block_num_x, block_num_y, 1);
    const dim3 block_dims(WARP_SIZE, nwarps, 1);

    if (nrows_x % mmq_y == 0) {
        const bool need_check = false;
        mul_mat_q8_0<need_check><<<block_nums, block_dims, 0, stream>>>
            (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
    } else {
        const bool need_check = true;
        mul_mat_q8_0<need_check><<<block_nums, block_dims, 0, stream>>>
            (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
    }
}

static void ggml_mul_mat_q2_K_q8_1_cuda(
    const void * vx, const void * vy, float * dst, const int ncols_x, const int nrows_x,
    const int ncols_y, const int nrows_y, const int nrows_dst, cudaStream_t stream) {

    int id = ggml_cuda_get_device();
    const int compute_capability = ggml_cuda_info().devices[id].cc;

    int mmq_x, mmq_y, nwarps;
    if (compute_capability >= CC_RDNA2) {
        mmq_x  =  MMQ_X_Q2_K_RDNA2;
        mmq_y  =  MMQ_Y_Q2_K_RDNA2;
        nwarps = NWARPS_Q2_K_RDNA2;
    } else if (compute_capability >= CC_OFFSET_AMD) {
        mmq_x  =  MMQ_X_Q2_K_RDNA1;
        mmq_y  =  MMQ_Y_Q2_K_RDNA1;
        nwarps = NWARPS_Q2_K_RDNA1;
    } else if (compute_capability >= CC_VOLTA) {
        mmq_x  =  MMQ_X_Q2_K_AMPERE;
        mmq_y  =  MMQ_Y_Q2_K_AMPERE;
        nwarps = NWARPS_Q2_K_AMPERE;
    } else if (compute_capability >= MIN_CC_DP4A) {
        mmq_x  =  MMQ_X_Q2_K_PASCAL;
        mmq_y  =  MMQ_Y_Q2_K_PASCAL;
        nwarps = NWARPS_Q2_K_PASCAL;
    } else {
        GGML_ASSERT(false);
    }

    const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
    const int block_num_y = (ncols_y + mmq_x - 1) / mmq_x;
    const dim3 block_nums(block_num_x, block_num_y, 1);
    const dim3 block_dims(WARP_SIZE, nwarps, 1);

    if (nrows_x % mmq_y == 0) {
        const bool need_check = false;
        mul_mat_q2_K<need_check><<<block_nums, block_dims, 0, stream>>>
            (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
    } else {
        const bool need_check = true;
        mul_mat_q2_K<need_check><<<block_nums, block_dims, 0, stream>>>
            (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
    }
}

static void ggml_mul_mat_q3_K_q8_1_cuda(
    const void * vx, const void * vy, float * dst, const int ncols_x, const int nrows_x,
    const int ncols_y, const int nrows_y, const int nrows_dst, cudaStream_t stream) {

#if QK_K == 256

    int id = ggml_cuda_get_device();
    const int compute_capability = ggml_cuda_info().devices[id].cc;

    int mmq_x, mmq_y, nwarps;
    if (compute_capability >= CC_RDNA2) {
        mmq_x  =  MMQ_X_Q3_K_RDNA2;
        mmq_y  =  MMQ_Y_Q3_K_RDNA2;
        nwarps = NWARPS_Q3_K_RDNA2;
    } else if (compute_capability >= CC_OFFSET_AMD) {
        mmq_x  =  MMQ_X_Q3_K_RDNA1;
        mmq_y  =  MMQ_Y_Q3_K_RDNA1;
        nwarps = NWARPS_Q3_K_RDNA1;
    } else if (compute_capability >= CC_VOLTA) {
        mmq_x  =  MMQ_X_Q3_K_AMPERE;
        mmq_y  =  MMQ_Y_Q3_K_AMPERE;
        nwarps = NWARPS_Q3_K_AMPERE;
    } else if (compute_capability >= MIN_CC_DP4A) {
        mmq_x  =  MMQ_X_Q3_K_PASCAL;
        mmq_y  =  MMQ_Y_Q3_K_PASCAL;
        nwarps = NWARPS_Q3_K_PASCAL;
    } else {
        GGML_ASSERT(false);
    }

    const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
    const int block_num_y = (ncols_y + mmq_x - 1) / mmq_x;
    const dim3 block_nums(block_num_x, block_num_y, 1);
    const dim3 block_dims(WARP_SIZE, nwarps, 1);

    if (nrows_x % mmq_y == 0) {
        const bool need_check = false;
        mul_mat_q3_K<need_check><<<block_nums, block_dims, 0, stream>>>
            (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
    } else {
        const bool need_check = true;
        mul_mat_q3_K<need_check><<<block_nums, block_dims, 0, stream>>>
            (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
    }
#endif
}

static void ggml_mul_mat_q4_K_q8_1_cuda(
    const void * vx, const void * vy, float * dst, const int ncols_x, const int nrows_x,
    const int ncols_y, const int nrows_y, const int nrows_dst, cudaStream_t stream) {

    int id = ggml_cuda_get_device();
    const int compute_capability = ggml_cuda_info().devices[id].cc;

    int mmq_x, mmq_y, nwarps;
    if (compute_capability >= CC_RDNA2) {
        mmq_x  =  MMQ_X_Q4_K_RDNA2;
        mmq_y  =  MMQ_Y_Q4_K_RDNA2;
        nwarps = NWARPS_Q4_K_RDNA2;
    } else if (compute_capability >= CC_OFFSET_AMD) {
        mmq_x  =  MMQ_X_Q4_K_RDNA1;
        mmq_y  =  MMQ_Y_Q4_K_RDNA1;
        nwarps = NWARPS_Q4_K_RDNA1;
    } else if (compute_capability >= CC_VOLTA) {
        mmq_x  =  MMQ_X_Q4_K_AMPERE;
        mmq_y  =  MMQ_Y_Q4_K_AMPERE;
        nwarps = NWARPS_Q4_K_AMPERE;
    } else if (compute_capability >= MIN_CC_DP4A) {
        mmq_x  =  MMQ_X_Q4_K_PASCAL;
        mmq_y  =  MMQ_Y_Q4_K_PASCAL;
        nwarps = NWARPS_Q4_K_PASCAL;
    } else {
        GGML_ASSERT(false);
    }

    const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
    const int block_num_y = (ncols_y + mmq_x - 1) / mmq_x;
    const dim3 block_nums(block_num_x, block_num_y, 1);
    const dim3 block_dims(WARP_SIZE, nwarps, 1);

    if (nrows_x % mmq_y == 0) {
        const bool need_check = false;
        mul_mat_q4_K<need_check><<<block_nums, block_dims, 0, stream>>>
            (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
    } else {
        const bool need_check = true;
        mul_mat_q4_K<need_check><<<block_nums, block_dims, 0, stream>>>
            (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
    }
}

static void ggml_mul_mat_q5_K_q8_1_cuda(
    const void * vx, const void * vy, float * dst, const int ncols_x, const int nrows_x,
    const int ncols_y, const int nrows_y, const int nrows_dst, cudaStream_t stream) {

    int id = ggml_cuda_get_device();
    const int compute_capability = ggml_cuda_info().devices[id].cc;

    int mmq_x, mmq_y, nwarps;
    if (compute_capability >= CC_RDNA2) {
        mmq_x  =  MMQ_X_Q5_K_RDNA2;
        mmq_y  =  MMQ_Y_Q5_K_RDNA2;
        nwarps = NWARPS_Q5_K_RDNA2;
    } else if (compute_capability >= CC_OFFSET_AMD) {
        mmq_x  =  MMQ_X_Q5_K_RDNA1;
        mmq_y  =  MMQ_Y_Q5_K_RDNA1;
        nwarps = NWARPS_Q5_K_RDNA1;
    } else if (compute_capability >= CC_VOLTA) {
        mmq_x  =  MMQ_X_Q5_K_AMPERE;
        mmq_y  =  MMQ_Y_Q5_K_AMPERE;
        nwarps = NWARPS_Q5_K_AMPERE;
    } else if (compute_capability >= MIN_CC_DP4A) {
        mmq_x  =  MMQ_X_Q5_K_PASCAL;
        mmq_y  =  MMQ_Y_Q5_K_PASCAL;
        nwarps = NWARPS_Q5_K_PASCAL;
    } else {
        GGML_ASSERT(false);
    }

    const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
    const int block_num_y = (ncols_y + mmq_x - 1) / mmq_x;
    const dim3 block_nums(block_num_x, block_num_y, 1);
    const dim3 block_dims(WARP_SIZE, nwarps, 1);

    if (nrows_x % mmq_y == 0) {
        const bool need_check = false;
        mul_mat_q5_K<need_check><<<block_nums, block_dims, 0, stream>>>
            (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
    } else {
        const bool need_check = true;
        mul_mat_q5_K<need_check><<<block_nums, block_dims, 0, stream>>>
            (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
    }
}

static void ggml_mul_mat_q6_K_q8_1_cuda(
    const void * vx, const void * vy, float * dst, const int ncols_x, const int nrows_x,
    const int ncols_y, const int nrows_y, const int nrows_dst, cudaStream_t stream) {

    int id = ggml_cuda_get_device();
    const int compute_capability = ggml_cuda_info().devices[id].cc;

    int mmq_x, mmq_y, nwarps;
    if (compute_capability >= CC_RDNA2) {
        mmq_x  =  MMQ_X_Q6_K_RDNA2;
        mmq_y  =  MMQ_Y_Q6_K_RDNA2;
        nwarps = NWARPS_Q6_K_RDNA2;
    } else if (compute_capability >= CC_OFFSET_AMD) {
        mmq_x  =  MMQ_X_Q6_K_RDNA1;
        mmq_y  =  MMQ_Y_Q6_K_RDNA1;
        nwarps = NWARPS_Q6_K_RDNA1;
    } else if (compute_capability >= CC_VOLTA) {
        mmq_x  =  MMQ_X_Q6_K_AMPERE;
        mmq_y  =  MMQ_Y_Q6_K_AMPERE;
        nwarps = NWARPS_Q6_K_AMPERE;
    } else if (compute_capability >= MIN_CC_DP4A) {
        mmq_x  =  MMQ_X_Q6_K_PASCAL;
        mmq_y  =  MMQ_Y_Q6_K_PASCAL;
        nwarps = NWARPS_Q6_K_PASCAL;
    } else {
        GGML_ASSERT(false);
    }

    const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
    const int block_num_y = (ncols_y + mmq_x - 1) / mmq_x;
    const dim3 block_nums(block_num_x, block_num_y, 1);
    const dim3 block_dims(WARP_SIZE, nwarps, 1);

    if (nrows_x % mmq_y == 0) {
        const bool need_check = false;
        mul_mat_q6_K<need_check><<<block_nums, block_dims, 0, stream>>>
            (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
    } else {
        const bool need_check = true;
        mul_mat_q6_K<need_check><<<block_nums, block_dims, 0, stream>>>
            (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
    }
}

void ggml_cuda_op_mul_mat_q(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream) {

    const int64_t ne00 = src0->ne[0];

    const int64_t ne10 = src1->ne[0];
    GGML_ASSERT(ne10 % QK8_1 == 0);

    const int64_t ne0 = dst->ne[0];

    const int64_t row_diff = row_high - row_low;

    int id = ggml_cuda_get_device();

    // the main device has a larger memory buffer to hold the results from all GPUs
    // nrows_dst == nrows of the matrix that the kernel writes into
    const int64_t nrows_dst = id == ctx.device ? ne0 : row_diff;

    switch (src0->type) {
        case GGML_TYPE_Q4_0:
            ggml_mul_mat_q4_0_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_ncols, src1_padded_row_size, nrows_dst, stream);
            break;
        case GGML_TYPE_Q4_1:
            ggml_mul_mat_q4_1_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_ncols, src1_padded_row_size, nrows_dst, stream);
            break;
        case GGML_TYPE_Q5_0:
            ggml_mul_mat_q5_0_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_ncols, src1_padded_row_size, nrows_dst, stream);
            break;
        case GGML_TYPE_Q5_1:
            ggml_mul_mat_q5_1_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_ncols, src1_padded_row_size, nrows_dst, stream);
            break;
        case GGML_TYPE_Q8_0:
            ggml_mul_mat_q8_0_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_ncols, src1_padded_row_size, nrows_dst, stream);
            break;
        case GGML_TYPE_Q2_K:
            ggml_mul_mat_q2_K_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_ncols, src1_padded_row_size, nrows_dst, stream);
            break;
        case GGML_TYPE_Q3_K:
            ggml_mul_mat_q3_K_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_ncols, src1_padded_row_size, nrows_dst, stream);
            break;
        case GGML_TYPE_Q4_K:
            ggml_mul_mat_q4_K_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_ncols, src1_padded_row_size, nrows_dst, stream);
            break;
        case GGML_TYPE_Q5_K:
            ggml_mul_mat_q5_K_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_ncols, src1_padded_row_size, nrows_dst, stream);
            break;
        case GGML_TYPE_Q6_K:
            ggml_mul_mat_q6_K_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_ncols, src1_padded_row_size, nrows_dst, stream);
            break;
        default:
            GGML_ASSERT(false);
            break;
    }

    GGML_UNUSED(src1);
    GGML_UNUSED(dst);
    GGML_UNUSED(src1_ddf_i);
}

bool ggml_cuda_supports_mmq(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
            return true;
        default:
            return false;
    }
}
