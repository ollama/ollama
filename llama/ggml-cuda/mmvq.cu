#include "mmvq.cuh"
#include "vecdotq.cuh"

typedef float (*vec_dot_q_cuda_t)(const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & iqs);

template <int ncols_y, int qk, int qi, typename block_q_t, int vdr, vec_dot_q_cuda_t vec_dot_q_cuda>
#if !(defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__))
// tell the compiler to use as many registers as it wants, see nwarps definition below
__launch_bounds__((ncols_y <= 4 ? 4 : 2)*WARP_SIZE, 1)
#endif // !(defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__))
static __global__ void mul_mat_vec_q(
    const void * __restrict__ vx, const void * __restrict__ vy, float * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

#if defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__) && (defined(RDNA2) || defined(RDNA3))
    constexpr int nwarps              = 1;
    constexpr int rows_per_cuda_block = 1;
#else
    constexpr int nwarps              = ncols_y <= 4 ? 4 : 2;
    constexpr int rows_per_cuda_block = ncols_y == 1 ? 1 : 2;
#endif // defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__) && !defined(RDNA2) && !defined(RDNA3)

    const     int tid = WARP_SIZE*threadIdx.y + threadIdx.x;
    const     int row0 = rows_per_cuda_block*blockIdx.x;
    const     int blocks_per_row_x = ncols_x / qk;
    const     int blocks_per_col_y = nrows_y / QK8_1;
    constexpr int blocks_per_iter = vdr * nwarps*WARP_SIZE / qi;

// partial sum for each thread
    float tmp[ncols_y][rows_per_cuda_block] = {0.0f};

    const block_q_t  * x = (const block_q_t  *) vx;
    const block_q8_1 * y = (const block_q8_1 *) vy;

    for (int kbx = tid / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk/QK8_1); // y block index that aligns with kbx

        // x block quant index when casting the quants to int
        const int kqs = vdr * (tid % (qi/vdr));

#pragma unroll
        for (int j = 0; j < ncols_y; ++j) {
#pragma unroll
            for (int i = 0; i < rows_per_cuda_block; ++i) {
                tmp[j][i] += vec_dot_q_cuda(
                    &x[kbx + (row0 + i)*blocks_per_row_x], &y[j*blocks_per_col_y + kby], kqs);
            }
        }
    }

    __shared__ float tmp_shared[nwarps-1 > 0 ? nwarps-1 : 1][ncols_y][rows_per_cuda_block][WARP_SIZE];
    if (threadIdx.y > 0) {
#pragma unroll
        for (int j = 0; j < ncols_y; ++j) {
#pragma unroll
            for (int i = 0; i < rows_per_cuda_block; ++i) {
                tmp_shared[threadIdx.y-1][j][i][threadIdx.x] = tmp[j][i];
            }
        }
    }
    __syncthreads();
    if (threadIdx.y > 0) {
        return;
    }

    // sum up partial sums and write back result
#pragma unroll
    for (int j = 0; j < ncols_y; ++j) {
#pragma unroll
        for (int i = 0; i < rows_per_cuda_block; ++i) {
#pragma unroll
            for (int l = 0; l < nwarps-1; ++l) {
                tmp[j][i] += tmp_shared[l][j][i][threadIdx.x];
            }
            tmp[j][i] = warp_reduce_sum(tmp[j][i]);
        }

        if (threadIdx.x < rows_per_cuda_block) {
            dst[j*nrows_dst + row0 + threadIdx.x] = tmp[j][threadIdx.x];
        }
    }
}

template <int qk, int qi, typename block_q_t, int vdr, vec_dot_q_cuda_t vec_dot>
static void mul_mat_vec_q_cuda(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int ncols_y, const int nrows_dst, cudaStream_t stream) {

    GGML_ASSERT(ncols_x % qk == 0);
    GGML_ASSERT(ncols_y <= MMVQ_MAX_BATCH_SIZE);

    int id = ggml_cuda_get_device();

    int64_t nwarps = 1;
    int64_t rows_per_cuda_block = 1;

    if (ggml_cuda_info().devices[id].cc < CC_RDNA2) { // NVIDIA and AMD older than RDNA2
        switch(ncols_y) {
            case 1:
                nwarps = 4;
                rows_per_cuda_block = 1;
                break;
            case 2:
            case 3:
            case 4:
                nwarps = 4;
                rows_per_cuda_block = 2;
                break;
            case 5:
            case 6:
            case 7:
            case 8:
                nwarps = 2;
                rows_per_cuda_block = 2;
                break;
            default:
                GGML_ASSERT(false);
                break;
        }
    }
    const int64_t nblocks = (nrows_x + rows_per_cuda_block - 1) / rows_per_cuda_block;
    const dim3 block_nums(nblocks, 1, 1);
    const dim3 block_dims(WARP_SIZE, nwarps, 1);

    switch (ncols_y) {
        case 1:
            mul_mat_vec_q<1, qk, qi, block_q_t, vdr, vec_dot>
                <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
            break;
        case 2:
            mul_mat_vec_q<2, qk, qi, block_q_t, vdr, vec_dot>
                <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
            break;
        case 3:
            mul_mat_vec_q<3, qk, qi, block_q_t, vdr, vec_dot>
                <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
            break;
        case 4:
            mul_mat_vec_q<4, qk, qi, block_q_t, vdr, vec_dot>
                <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
            break;
        case 5:
            mul_mat_vec_q<5, qk, qi, block_q_t, vdr, vec_dot>
                <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
            break;
        case 6:
            mul_mat_vec_q<6, qk, qi, block_q_t, vdr, vec_dot>
                <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
            break;
        case 7:
            mul_mat_vec_q<7, qk, qi, block_q_t, vdr, vec_dot>
                <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
            break;
        case 8:
            mul_mat_vec_q<8, qk, qi, block_q_t, vdr, vec_dot>
                <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
            break;
        default:
            GGML_ASSERT(false);
            break;
    }
}

static void mul_mat_vec_q4_0_q8_1_cuda(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int ncols_y, const int nrows_dst, cudaStream_t stream) {

    mul_mat_vec_q_cuda<QK4_0, QI4_0, block_q4_0, VDR_Q4_0_Q8_1_MMVQ, vec_dot_q4_0_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, ncols_y, nrows_dst, stream);
}

static void mul_mat_vec_q4_1_q8_1_cuda(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int ncols_y, const int nrows_dst, cudaStream_t stream) {

    mul_mat_vec_q_cuda<QK4_1, QI4_1, block_q4_1, VDR_Q4_1_Q8_1_MMVQ, vec_dot_q4_1_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, ncols_y, nrows_dst, stream);
}

static void mul_mat_vec_q5_0_q8_1_cuda(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int ncols_y, const int nrows_dst, cudaStream_t stream) {

    mul_mat_vec_q_cuda<QK5_0, QI5_0, block_q5_0, VDR_Q5_0_Q8_1_MMVQ, vec_dot_q5_0_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, ncols_y, nrows_dst, stream);
}

static void mul_mat_vec_q5_1_q8_1_cuda(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int ncols_y, const int nrows_dst, cudaStream_t stream) {

    mul_mat_vec_q_cuda<QK5_1, QI5_1, block_q5_1, VDR_Q5_1_Q8_1_MMVQ, vec_dot_q5_1_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, ncols_y, nrows_dst, stream);
}

static void mul_mat_vec_q8_0_q8_1_cuda(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int ncols_y, const int nrows_dst, cudaStream_t stream) {

    mul_mat_vec_q_cuda<QK8_0, QI8_0, block_q8_0, VDR_Q8_0_Q8_1_MMVQ, vec_dot_q8_0_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, ncols_y, nrows_dst, stream);
}

static void mul_mat_vec_q2_K_q8_1_cuda(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int ncols_y, const int nrows_dst, cudaStream_t stream) {

    mul_mat_vec_q_cuda<QK_K, QI2_K, block_q2_K, VDR_Q2_K_Q8_1_MMVQ, vec_dot_q2_K_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, ncols_y, nrows_dst, stream);
}

static void mul_mat_vec_q3_K_q8_1_cuda(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int ncols_y, const int nrows_dst, cudaStream_t stream) {

    mul_mat_vec_q_cuda<QK_K, QI3_K, block_q3_K, VDR_Q3_K_Q8_1_MMVQ, vec_dot_q3_K_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, ncols_y, nrows_dst, stream);
}

static void mul_mat_vec_q4_K_q8_1_cuda(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int ncols_y, const int nrows_dst, cudaStream_t stream) {

    mul_mat_vec_q_cuda<QK_K, QI4_K, block_q4_K, VDR_Q4_K_Q8_1_MMVQ, vec_dot_q4_K_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, ncols_y, nrows_dst, stream);
}

static void mul_mat_vec_q5_K_q8_1_cuda(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int ncols_y, const int nrows_dst, cudaStream_t stream) {

    mul_mat_vec_q_cuda<QK_K, QI5_K, block_q5_K, VDR_Q5_K_Q8_1_MMVQ, vec_dot_q5_K_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, ncols_y, nrows_dst, stream);
}

static void mul_mat_vec_q6_K_q8_1_cuda(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int ncols_y, const int nrows_dst, cudaStream_t stream) {

    mul_mat_vec_q_cuda<QK_K, QI6_K, block_q6_K, VDR_Q6_K_Q8_1_MMVQ, vec_dot_q6_K_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, ncols_y, nrows_dst, stream);
}

static void mul_mat_vec_iq2_xxs_q8_1_cuda(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int ncols_y, const int nrows_dst, cudaStream_t stream) {

    mul_mat_vec_q_cuda<QK_K, QI2_XXS, block_iq2_xxs, 1, vec_dot_iq2_xxs_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, ncols_y, nrows_dst, stream);
}

static void mul_mat_vec_iq2_xs_q8_1_cuda(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int ncols_y, const int nrows_dst, cudaStream_t stream) {

    mul_mat_vec_q_cuda<QK_K, QI2_XS, block_iq2_xs, 1, vec_dot_iq2_xs_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, ncols_y, nrows_dst, stream);
}

static void mul_mat_vec_iq2_s_q8_1_cuda(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int ncols_y, const int nrows_dst, cudaStream_t stream) {

    mul_mat_vec_q_cuda<QK_K, QI2_S, block_iq2_s, 1, vec_dot_iq2_s_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, ncols_y, nrows_dst, stream);
}

static void mul_mat_vec_iq3_xxs_q8_1_cuda(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int ncols_y, const int nrows_dst, cudaStream_t stream) {

    mul_mat_vec_q_cuda<QK_K, QI3_XXS, block_iq3_xxs, 1, vec_dot_iq3_xxs_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, ncols_y, nrows_dst, stream);
}

static void mul_mat_vec_iq1_s_q8_1_cuda(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int ncols_y, const int nrows_dst, cudaStream_t stream) {

    mul_mat_vec_q_cuda<QK_K, QI1_S, block_iq1_s, 1, vec_dot_iq1_s_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, ncols_y, nrows_dst, stream);
}

static void mul_mat_vec_iq1_m_q8_1_cuda(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int ncols_y, const int nrows_dst, cudaStream_t stream) {

    mul_mat_vec_q_cuda<QK_K, QI1_S, block_iq1_m, 1, vec_dot_iq1_m_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, ncols_y, nrows_dst, stream);
}

static void mul_mat_vec_iq4_nl_q8_1_cuda(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int ncols_y, const int nrows_dst, cudaStream_t stream) {

    mul_mat_vec_q_cuda<QK4_NL, QI4_NL, block_iq4_nl, VDR_Q4_0_Q8_1_MMVQ, vec_dot_iq4_nl_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, ncols_y, nrows_dst, stream);
}

static void mul_mat_vec_iq4_xs_q8_1_cuda(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int ncols_y, const int nrows_dst, cudaStream_t stream) {

    mul_mat_vec_q_cuda<QK_K, QI4_XS, block_iq4_xs, 1, vec_dot_iq4_xs_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, ncols_y, nrows_dst, stream);
}

static void mul_mat_vec_iq3_s_q8_1_cuda(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int ncols_y, const int nrows_dst, cudaStream_t stream) {

    mul_mat_vec_q_cuda<QK_K, QI3_XS, block_iq3_s, 1, vec_dot_iq3_s_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, ncols_y, nrows_dst, stream);
}

void ggml_cuda_op_mul_mat_vec_q(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream) {

    const int64_t ne00 = src0->ne[0];
    const int64_t row_diff = row_high - row_low;

    const int64_t ne10 = src1->ne[0];
    GGML_ASSERT(ne10 % QK8_1 == 0);

    const int64_t ne0 = dst->ne[0];

    int id = ggml_cuda_get_device();

    // the main device has a larger memory buffer to hold the results from all GPUs
    // nrows_dst == nrows of the matrix that the kernel writes into
    const int64_t nrows_dst = id == ctx.device ? ne0 : row_diff;

    switch (src0->type) {
        case GGML_TYPE_Q4_0:
            mul_mat_vec_q4_0_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst, stream);
            break;
        case GGML_TYPE_Q4_1:
            mul_mat_vec_q4_1_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst, stream);
            break;
        case GGML_TYPE_Q5_0:
            mul_mat_vec_q5_0_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst, stream);
            break;
        case GGML_TYPE_Q5_1:
            mul_mat_vec_q5_1_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst, stream);
            break;
        case GGML_TYPE_Q8_0:
            mul_mat_vec_q8_0_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst, stream);
            break;
        case GGML_TYPE_Q2_K:
            mul_mat_vec_q2_K_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst, stream);
            break;
        case GGML_TYPE_Q3_K:
            mul_mat_vec_q3_K_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst, stream);
            break;
        case GGML_TYPE_Q4_K:
            mul_mat_vec_q4_K_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst, stream);
            break;
        case GGML_TYPE_Q5_K:
            mul_mat_vec_q5_K_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst, stream);
            break;
        case GGML_TYPE_Q6_K:
            mul_mat_vec_q6_K_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst, stream);
            break;
        case GGML_TYPE_IQ2_XXS:
            mul_mat_vec_iq2_xxs_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst, stream);
            break;
        case GGML_TYPE_IQ2_XS:
            mul_mat_vec_iq2_xs_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst, stream);
            break;
        case GGML_TYPE_IQ2_S:
            mul_mat_vec_iq2_s_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst, stream);
            break;
        case GGML_TYPE_IQ3_XXS:
            mul_mat_vec_iq3_xxs_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst, stream);
            break;
        case GGML_TYPE_IQ1_S:
            mul_mat_vec_iq1_s_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst, stream);
            break;
        case GGML_TYPE_IQ1_M:
            mul_mat_vec_iq1_m_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst, stream);
            break;
        case GGML_TYPE_IQ4_NL:
            mul_mat_vec_iq4_nl_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst, stream);
            break;
        case GGML_TYPE_IQ4_XS:
            mul_mat_vec_iq4_xs_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst, stream);
            break;
        case GGML_TYPE_IQ3_S:
            mul_mat_vec_iq3_s_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst, stream);
            break;
        default:
            GGML_ASSERT(false);
            break;
    }

    GGML_UNUSED(src1);
    GGML_UNUSED(dst);
    GGML_UNUSED(src1_ddf_i);
    GGML_UNUSED(src1_ncols);
    GGML_UNUSED(src1_padded_row_size);
}
