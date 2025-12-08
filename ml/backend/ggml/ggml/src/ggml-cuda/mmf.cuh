#pragma once

#include "mma.cuh"
#include "common.cuh"
#include "convert.cuh"

using namespace ggml_cuda_mma;

#define MMF_ROWS_PER_BLOCK 32

struct mmf_ids_data {
    const int32_t * ids_src_compact = nullptr;
    const int32_t * ids_dst_compact = nullptr;
    const int32_t * expert_bounds_dev = nullptr;
    int n_experts = 0;
    int sis1 = 0;
};

void ggml_cuda_mul_mat_f(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst);

bool ggml_cuda_should_use_mmf(enum ggml_type type, int cc, int warp_size, const int64_t * scr0_ne, const size_t * src0_nb, const int src1_ncols, bool mul_mat_id);

template <typename T, int rows_per_block, int cols_per_block, int nwarps, bool has_ids>
__launch_bounds__(ggml_cuda_get_physical_warp_size()*nwarps, 1)
static __global__ void mul_mat_f(
        const T * __restrict__ x, const float * __restrict__ y, const int32_t * __restrict__ ids, float * __restrict__ dst,
        const int ncols, const int ncols_dst_total, const int nchannels_dst, const int stride_row, const int stride_col_y, const int stride_col_dst,
        const int stride_col_id, const int stride_row_id,
        const int channel_ratio, const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst,
        const int sample_ratio, const int stride_sample_x, const int stride_sample_y, const int stride_sample_dst) {
// TODO: handle this in a consistent and simpler way after AMD MFMA support has been added
#if (!defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)) || defined(AMD_WMMA_AVAILABLE)
#if defined(AMD_WMMA_AVAILABLE)
    // Special case for tf32, just dummy mma layout as wmma doesn't support it.
    constexpr int tile_B_I = std::is_same_v<T, float> ? 8 : 16;
    constexpr int tile_C_J = std::is_same_v<T, float> ? 8 : 16;
    typedef tile<16,       8, T>     tile_A;
    typedef tile<tile_B_I, 8, T>     tile_B;
    typedef tile<16,       tile_C_J, float> tile_C;

    constexpr bool a_supported = tile_A::supported();
    constexpr bool b_supported = tile_B::supported();
    constexpr bool c_supported = tile_C::supported();
    constexpr bool supported = a_supported && b_supported && c_supported;
#else
    constexpr bool I_16_supported = tile<16, 8, T>::supported() && tile<16, 8, float>::supported();
    constexpr bool I_32_supported = tile<32, 8, T>::supported() && tile<32, 8, float>::supported();
    constexpr bool supported = I_16_supported || I_32_supported;

    constexpr int I_preferred = I_16_supported ? 16 : 32; // For Turing MMA both work but 16 is ~1% faster.

    typedef tile<I_preferred, 8, T>     tile_A;
    typedef tile<8,           8, T>     tile_B;
    typedef tile<I_preferred, 8, float> tile_C;
#endif // defined(AMD_WMMA_AVAILABLE)
    if constexpr (!supported) {
        NO_DEVICE_CODE;
        return;
    }

    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int tile_k_padded = warp_size + 4;
    constexpr int ntA = rows_per_block / tile_A::I;
    constexpr int ntB = (cols_per_block + tile_B::I - 1) / tile_B::I;

    const int row0        = blockIdx.x * rows_per_block;

    int expert_idx = 0;
    int col_base = 0;

    const int channel_dst = has_ids ? 0 : blockIdx.y;

    if constexpr (has_ids) {
        // experts + tiles of ncols_dst are packed in the y dimension
        int col_tiles = (ncols_dst_total + cols_per_block - 1) / cols_per_block;
        const int nchannels_x = gridDim.y / col_tiles;
        const int tile_idx = blockIdx.y / nchannels_x;
        expert_idx = blockIdx.y - tile_idx * nchannels_x;
        col_base = tile_idx * cols_per_block;
    }

    const int channel_x   = has_ids ? expert_idx : (channel_dst / channel_ratio);
    const int channel_y   = channel_dst;
    const int sample_dst  = blockIdx.z;
    const int sample_x    = sample_dst / sample_ratio;
    const int sample_y    = sample_dst;

    x   += int64_t(sample_x)  *stride_sample_x   + channel_x  *stride_channel_x  + row0*stride_row ;
    y   += int64_t(sample_y)  *stride_sample_y   + (has_ids ? 0 : channel_y  *stride_channel_y);
    dst += int64_t(sample_dst)*stride_sample_dst + (has_ids ? 0 : channel_dst*stride_channel_dst);

    if constexpr (has_ids) {
        constexpr int y_stride_scale = std::is_same_v<T, float> ? 1 : 2;
        const int64_t col_offset = col_base;
        y   += col_offset * stride_col_y * y_stride_scale;
        dst += col_offset * stride_col_dst;
        ids += col_offset * stride_row_id;
    }

    const float2 * y2 = (const float2 *) y;

    extern __shared__ char data_mmv[];

    char * shmem_base = data_mmv;
    int  * slot_map   = (int *) shmem_base;
    char * compute_base = has_ids ? (shmem_base + GGML_PAD(cols_per_block, 16) * sizeof(int)) : shmem_base;

    tile_C C[ntA][ntB];

    T * tile_xy = (T *) compute_base + threadIdx.y*(tile_A::I * tile_k_padded);

    if constexpr (has_ids) {
        int found = 0;

        for (int j0 = 0; j0 < cols_per_block; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

            if (threadIdx.x == 0) {
                slot_map[j] = -1;
            }

            if (col_base + j >= ncols_dst_total) {
                continue;
            }

            const int32_t * __restrict__ id_row = ids + j*stride_row_id;

            for (int k = threadIdx.x; k < nchannels_dst; k += warp_size) {
                int match = id_row[k*stride_col_id] == expert_idx;

                if (match) {
                    slot_map[j] = k;
                    found = 1;
                    break;
                }
            }
        }

        if (!__syncthreads_or(found)) {
            return;
        }
    }


    for (int col = threadIdx.y*warp_size + threadIdx.x; col < ncols; col += nwarps*warp_size) {
        tile_A A[ntA][warp_size / tile_A::J];
#pragma unroll
        for (int itA = 0; itA < ntA; ++itA) {
#pragma unroll
            for (int i = 0; i < tile_A::I; ++i) {
                tile_xy[i*tile_k_padded + threadIdx.x] = x[(itA*tile_A::I + i)*stride_row  + col];
            }
#pragma unroll
            for (int k0 = 0; k0 < warp_size; k0 += tile_A::J) {
                load_ldmatrix(A[itA][k0/tile_A::J], tile_xy + k0, tile_k_padded);
            }
        }

#pragma unroll
        for (int itB = 0; itB < ntB; ++itB) {
            if constexpr (std::is_same_v<T, float>) {
#pragma unroll
                for (int j0 = 0; j0 < tile_B::I; ++j0) {
                    const int j = j0 + itB*tile_B::I;

                    if constexpr (!has_ids) {
                        tile_xy[j0*tile_k_padded + threadIdx.x] = j < cols_per_block ? y[j*stride_col_y + col] : 0.0f;
                    } else {
                        const bool valid = j < cols_per_block && (col_base + j) < ncols_dst_total && slot_map[j] >= 0;
                        tile_xy[j0*tile_k_padded + threadIdx.x] = valid ? y[slot_map[j]*stride_channel_y + j*stride_col_y + col] : 0.0f;
                    }
                }
            } else if constexpr (std::is_same_v<T, half2> || std::is_same_v<T, nv_bfloat162>) {
#pragma unroll
                for (int j0 = 0; j0 < tile_B::I; ++j0) {
                    const int j = j0 + itB*tile_B::I;

                    if constexpr (!has_ids) {
                        const float2 tmp = j < cols_per_block ? y2[j*stride_col_y + col] : make_float2(0.0f, 0.0f);
                        tile_xy[j0*tile_k_padded + threadIdx.x] = ggml_cuda_cast<T>(tmp);
                    } else {
                        const bool valid = j < cols_per_block && (col_base + j) < ncols_dst_total && slot_map[j] >= 0;
                        float2 tmp = valid ? *(const float2*) &y[slot_map[j]*stride_channel_y + 2*(j*stride_col_y + col)] : make_float2(0.0f, 0.0f);
                        tile_xy[j0*tile_k_padded + threadIdx.x] = ggml_cuda_cast<T>(tmp);
                    }
                }
            } else {
                static_assert(std::is_same_v<T, void>, "unsupported type");
            }
#pragma unroll
            for (int k0 = 0; k0 < warp_size; k0 += tile_B::J) {
                tile_B B;
                load_ldmatrix(B, tile_xy + k0, tile_k_padded);
#pragma unroll
                for (int itA = 0; itA < ntA; ++itA) {
                    mma(C[itA][itB], A[itA][k0/tile_B::J], B);
                }
            }
        }
    }

    float * buf_iw = (float *) compute_base;
    constexpr int kiw = nwarps*rows_per_block + 4;

    if (nwarps > 1) {
        __syncthreads();
    }
#pragma unroll
    for (int itB = 0; itB < ntB; ++itB) {
#pragma unroll
        for (int itA = 0; itA < ntA; ++itA) {
#pragma unroll
            for (int l = 0; l < tile_C::ne; ++l) {
                const int i = threadIdx.y*rows_per_block + itA*tile_C::I + tile_C::get_i(l);
                const int j = itB*tile_C::J + tile_C::get_j(l);
                buf_iw[j*kiw + i] = C[itA][itB].x[l];
            }
        }
    }

    if (nwarps > 1) {
        __syncthreads();
    }

#pragma unroll
    for (int j0 = 0; j0 < cols_per_block; j0 += nwarps) {
        const int j = j0 + threadIdx.y;

        if (j0 + nwarps > cols_per_block && j >= cols_per_block) {
            return;
        }

        float sum = 0.0f;
        static_assert(rows_per_block == warp_size, "need loop/check");
#pragma unroll
        for (int i0 = 0; i0 < nwarps*rows_per_block; i0 += rows_per_block) {
            const int i = i0 + threadIdx.x;

            sum += buf_iw[j*kiw + i];
        }

        if constexpr (!has_ids) {
            dst[j*stride_col_dst + row0 + threadIdx.x] = sum;
        } else {
            const int slot = (j < cols_per_block) ? slot_map[j] : -1;
            if (slot >= 0 && (col_base + j) < ncols_dst_total) {
                dst[slot*stride_channel_dst + j*stride_col_dst + row0 + threadIdx.x] = sum;
            }
        }
    }
#else
    GGML_UNUSED_VARS(x, y, ids, dst,
        ncols, ncols_dst_total, nchannels_dst, stride_row, stride_col_y, stride_col_dst,
        stride_col_id, stride_row_id,
        channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
        sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst);
    NO_DEVICE_CODE;
#endif // (!defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)) || defined(AMD_WMMA_AVAILABLE)
}

//This kernel is for larger batch sizes of mul_mat_id
template <typename T, int rows_per_block, int cols_per_block, int nwarps>
__launch_bounds__(ggml_cuda_get_physical_warp_size()*nwarps, 1)
static __global__ void mul_mat_f_ids(
        const T * __restrict__ x, const float * __restrict__ y,
        const int32_t * __restrict__ ids_src_compact, const int32_t * __restrict__ ids_dst_compact,
        const int32_t * __restrict__ expert_bounds, float * __restrict__ dst,
        const int ncols, const int ncols_dst_total, const int nchannels_dst, const int stride_row, const int stride_col_y, const int stride_col_dst,
        const int channel_ratio, const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst,
        const int sample_ratio, const int stride_sample_x, const int stride_sample_y, const int stride_sample_dst,
        const uint3 sis1_fd, const uint3 nch_fd) {
// TODO: handle this in a consistent and simpler way after AMD MFMA support has been added
#if (!defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)) || defined(AMD_WMMA_AVAILABLE)
#if defined(AMD_WMMA_AVAILABLE)
    // Special case for tf32, just dummy mma layout as wmma doesn't support it.
    constexpr int tile_B_I = std::is_same_v<T, float> ? 8 : 16;
    constexpr int tile_C_J = std::is_same_v<T, float> ? 8 : 16;
    typedef tile<16,       8, T>     tile_A;
    typedef tile<tile_B_I, 8, T>     tile_B;
    typedef tile<16,       tile_C_J, float> tile_C;

    constexpr bool a_supported = tile_A::supported();
    constexpr bool b_supported = tile_B::supported();
    constexpr bool c_supported = tile_C::supported();
    constexpr bool supported = a_supported && b_supported && c_supported;
#else
    constexpr bool I_16_supported = tile<16, 8, T>::supported() && tile<16, 8, float>::supported();
    constexpr bool I_32_supported = tile<32, 8, T>::supported() && tile<32, 8, float>::supported();
    constexpr bool supported = I_16_supported || I_32_supported;

    constexpr int I_preferred = I_16_supported ? 16 : 32; // For Turing MMA both work but 16 is ~1% faster.

    typedef tile<I_preferred, 8, T>     tile_A;
    typedef tile<8,           8, T>     tile_B;
    typedef tile<I_preferred, 8, float> tile_C;
#endif // defined(AMD_WMMA_AVAILABLE)
    if constexpr (!supported) {
        NO_DEVICE_CODE;
        return;
    }

    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int tile_k_padded = warp_size + 4;
    constexpr int ntA = rows_per_block / tile_A::I;
    constexpr int ntB = (cols_per_block + tile_B::I - 1) / tile_B::I;

    const int row0        = blockIdx.x * rows_per_block;

    const int expert_idx = blockIdx.y;
    const int expert_start = expert_bounds[expert_idx];
    const int expert_end   = expert_bounds[expert_idx + 1];
    const int ncols_expert = expert_end - expert_start;

    const int tiles_for_expert = (ncols_expert + cols_per_block - 1) / cols_per_block;
    const int tile_idx = blockIdx.z;
    if (tile_idx >= tiles_for_expert) {
        return;
    }

    const int col_base = tile_idx * cols_per_block;

    GGML_UNUSED(channel_ratio);

    const int channel_x   = expert_idx;
    const int sample_dst  = 0;
    const int sample_x    = sample_dst / sample_ratio;
    const int sample_y    = sample_dst;

    x   += int64_t(sample_x)  *stride_sample_x   + channel_x  *stride_channel_x  + row0*stride_row;
    y   += int64_t(sample_y)  *stride_sample_y;
    dst += int64_t(sample_dst)*stride_sample_dst;

    const int32_t * ids_src_expert = ids_src_compact + expert_start;
    const int32_t * ids_dst_expert = ids_dst_compact + expert_start;

    extern __shared__ char data_mmv[];
    char * compute_base = data_mmv;

    //const float2 * y2 = (const float2 *) y;

    tile_C C[ntA][ntB];

    T * tile_xy = (T *) compute_base + threadIdx.y*(tile_A::I * tile_k_padded);

    for (int col = threadIdx.y*warp_size + threadIdx.x; col < ncols; col += nwarps*warp_size) {
        tile_A A[ntA][warp_size / tile_A::J];
#pragma unroll
        for (int itA = 0; itA < ntA; ++itA) {
#pragma unroll
            for (int i = 0; i < tile_A::I; ++i) {
                tile_xy[i*tile_k_padded + threadIdx.x] = x[(itA*tile_A::I + i)*stride_row  + col];
            }
#pragma unroll
            for (int k0 = 0; k0 < warp_size; k0 += tile_A::J) {
                load_ldmatrix(A[itA][k0/tile_A::J], tile_xy + k0, tile_k_padded);
            }
        }

        if constexpr (std::is_same_v<T, float>) {
            float vals_buf[2][tile_B::I];
            auto gather_tile = [&](int tile_idx_local, float *vals) {
#pragma unroll
                for (int j0 = 0; j0 < tile_B::I; ++j0) {
                    const int j = j0 + tile_idx_local*tile_B::I;
                    const int global_j = col_base + j;
                    float val = 0.0f;
                    if (j < cols_per_block && global_j < ncols_expert) {
                        const int src_entry = ids_src_expert[global_j];
                        const uint2 qrm = fast_div_modulo((uint32_t) src_entry, sis1_fd);
                        const int token   = (int) qrm.x;
                        const int channel = (int) qrm.y;
                        if (token < ncols_dst_total) {
                            val = y[channel*stride_channel_y + token*stride_col_y + col];
                        }
                    }
                    vals[j0] = val;
                }
            };

            gather_tile(0, vals_buf[0]);

            int curr_buf = 0;
            int next_buf = 1;
#pragma unroll
            for (int itB = 0; itB < ntB; ++itB) {
#pragma unroll
                for (int j0 = 0; j0 < tile_B::I; ++j0) {
                    tile_xy[j0*tile_k_padded + threadIdx.x] = vals_buf[curr_buf][j0];
                }

                if (itB + 1 < ntB) {
                    gather_tile(itB + 1, vals_buf[next_buf]);
                }

#pragma unroll
                for (int k0 = 0; k0 < warp_size; k0 += tile_B::J) {
                    tile_B B;
                    load_ldmatrix(B, tile_xy + k0, tile_k_padded);
#pragma unroll
                    for (int itA = 0; itA < ntA; ++itA) {
                        mma(C[itA][itB], A[itA][k0/tile_B::J], B);
                    }
                }

                if (itB + 1 < ntB) {
                    curr_buf ^= 1;
                    next_buf ^= 1;
                }
            }
        } else if constexpr (std::is_same_v<T, half2> || std::is_same_v<T, nv_bfloat162>) {
            float2 vals_buf[2][tile_B::I];
            auto gather_tile = [&](int tile_idx_local, float2 *vals) {
#pragma unroll
                for (int j0 = 0; j0 < tile_B::I; ++j0) {
                    const int j = j0 + tile_idx_local*tile_B::I;
                    const int global_j = col_base + j;
                    float2 tmp = make_float2(0.0f, 0.0f);
                    if (j < cols_per_block && global_j < ncols_expert) {
                        const int src_entry = ids_src_expert[global_j];
                        const uint2 qrm = fast_div_modulo((uint32_t) src_entry, sis1_fd);
                        const int token   = (int) qrm.x;
                        const int channel = (int) qrm.y;
                        if (token < ncols_dst_total) {
                            tmp = *(const float2*) &y[channel*stride_channel_y + 2*(token*stride_col_y + col)];
                        }
                    }
                    vals[j0] = tmp;
                }
            };

            if (ntB > 0) {
                gather_tile(0, vals_buf[0]);
            }

            int curr_buf = 0;
            int next_buf = 1;
#pragma unroll
            for (int itB = 0; itB < ntB; ++itB) {
#pragma unroll
                for (int j0 = 0; j0 < tile_B::I; ++j0) {
                    const float2 tmp = vals_buf[curr_buf][j0];
                    tile_xy[j0*tile_k_padded + threadIdx.x] = ggml_cuda_cast<T>(tmp);
                }

                if (itB + 1 < ntB) {
                    gather_tile(itB + 1, vals_buf[next_buf]);
                }

#pragma unroll
                for (int k0 = 0; k0 < warp_size; k0 += tile_B::J) {
                    tile_B B;
                    load_ldmatrix(B, tile_xy + k0, tile_k_padded);
#pragma unroll
                    for (int itA = 0; itA < ntA; ++itA) {
                        mma(C[itA][itB], A[itA][k0/tile_B::J], B);
                    }
                }

                if (itB + 1 < ntB) {
                    curr_buf ^= 1;
                    next_buf ^= 1;
                }
            }
        } else {
            static_assert(std::is_same_v<T, void>, "unsupported type");
        }
    }

    float * buf_iw = (float *) compute_base;
    constexpr int kiw = nwarps*rows_per_block + 4;

    if (nwarps > 1) {
        __syncthreads();
    }
#pragma unroll
    for (int itB = 0; itB < ntB; ++itB) {
#pragma unroll
        for (int itA = 0; itA < ntA; ++itA) {
#pragma unroll
            for (int l = 0; l < tile_C::ne; ++l) {
                const int i = threadIdx.y*rows_per_block + itA*tile_C::I + tile_C::get_i(l);
                const int j = itB*tile_C::J + tile_C::get_j(l);
                buf_iw[j*kiw + i] = C[itA][itB].x[l];
            }
        }
    }

    if (nwarps > 1) {
        __syncthreads();
    }

#pragma unroll
    for (int j0 = 0; j0 < cols_per_block; j0 += nwarps) {
        const int j = j0 + threadIdx.y;

        if (j0 + nwarps > cols_per_block && j >= cols_per_block) {
            return;
        }

        float sum = 0.0f;
        static_assert(rows_per_block == warp_size, "need loop/check");
#pragma unroll
        for (int i0 = 0; i0 < nwarps*rows_per_block; i0 += rows_per_block) {
            const int i = i0 + threadIdx.x;

            sum += buf_iw[j*kiw + i];
        }

        const int global_j = col_base + j;
        if (j < cols_per_block && global_j < ncols_expert && nchannels_dst > 0) {
            const int dst_entry = ids_dst_expert[global_j];
            const uint2 qrm = fast_div_modulo((uint32_t) dst_entry, nch_fd);
            const int token = (int) qrm.x;
            if (token < ncols_dst_total) {
                const int slot = (int) qrm.y;
                dst[slot*stride_channel_dst + token*stride_col_dst + row0 + threadIdx.x] = sum;
            }
        }
    }
#else
    GGML_UNUSED_VARS(x, y, ids_src_compact, ids_dst_compact, expert_bounds, dst,
        ncols, ncols_dst_total, nchannels_dst, stride_row, stride_col_y, stride_col_dst,
        channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
        sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst, sis1_fd, nch_fd);
    NO_DEVICE_CODE;
#endif // (!defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)) || defined(AMD_WMMA_AVAILABLE)
}

template<typename T, int cols_per_block, int nwarps>
static inline void mul_mat_f_switch_ids(
        const T * x, const float * y, const int32_t * ids, float * dst,
        const int64_t ncols_x, const int64_t ncols_dst, const int64_t nchannels_dst,
        const int64_t stride_row, const int64_t stride_col_y, const int64_t stride_col_dst,
        const int64_t stride_col_id, const int64_t stride_row_id,
        const int64_t channel_ratio, const int64_t stride_channel_x, const int64_t stride_channel_y, const int64_t stride_channel_dst,
        const int64_t sample_ratio, const int64_t stride_sample_x, const int64_t stride_sample_y, const int64_t stride_sample_dst,
        const dim3 & block_nums, const dim3 & block_dims, const int nbytes_shared_total, cudaStream_t stream,
        const mmf_ids_data * ids_data) {
    const bool has_ids_data = ids_data && ids_data->ids_src_compact;

    // Use the compact-ids kernel only for larger tiles; for small ncols_dst (< 16)
    // we prefer the normal mul_mat_f path with has_ids=true.
    if (has_ids_data && ncols_dst > 16) {
        const int max_tiles = (int) ((ncols_dst + cols_per_block - 1) / cols_per_block);
        if (max_tiles == 0) {
            return;
        }
        dim3 block_nums_ids(block_nums.x, ids_data->n_experts, max_tiles);

        const uint3 sis1_fd = ids_data->sis1 > 0 ? init_fastdiv_values((uint32_t) ids_data->sis1) : make_uint3(0, 0, 1);
        const uint3 nch_fd  = init_fastdiv_values((uint32_t) nchannels_dst);

        mul_mat_f_ids<T, MMF_ROWS_PER_BLOCK, cols_per_block, nwarps><<<block_nums_ids, block_dims, nbytes_shared_total, stream>>>
            (x, y, ids_data->ids_src_compact, ids_data->ids_dst_compact, ids_data->expert_bounds_dev, dst,
            ncols_x, ncols_dst, nchannels_dst, stride_row, stride_col_y, stride_col_dst,
            channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
            sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst,
            sis1_fd, nch_fd);
    } else if (ids) {
        const int64_t col_tiles = (ncols_dst + cols_per_block - 1) / cols_per_block;
        dim3 block_nums_ids = block_nums;
        block_nums_ids.y *= col_tiles;

        mul_mat_f<T, MMF_ROWS_PER_BLOCK, cols_per_block, nwarps, true><<<block_nums_ids, block_dims, nbytes_shared_total, stream>>>
            (x, y, ids, dst, ncols_x, ncols_dst, nchannels_dst, stride_row, stride_col_y, stride_col_dst,
             stride_col_id, stride_row_id, channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
             sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst);
    } else {
        mul_mat_f<T, MMF_ROWS_PER_BLOCK, cols_per_block, nwarps, false><<<block_nums, block_dims, nbytes_shared_total, stream>>>
            (x, y, ids, dst, ncols_x, cols_per_block, nchannels_dst, stride_row, stride_col_y, stride_col_dst,
             stride_col_id, stride_row_id, channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
             sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst);
    }
}

template <typename T, int cols_per_block>
void mul_mat_f_cuda(
        const T * x, const float * y, const int32_t * ids, float * dst,
        const int64_t ncols_x, const int64_t nrows_x, const int64_t ncols_dst,
        const int64_t stride_row, const int64_t stride_col_y, const int64_t stride_col_dst,
        const int64_t stride_col_id, const int64_t stride_row_id,
        const int64_t nchannels_x, const int64_t nchannels_y, const int64_t nchannels_dst,
        const int64_t stride_channel_x, const int64_t stride_channel_y, const int64_t stride_channel_dst, const int64_t nsamples_x,
        const int64_t nsamples_dst, const int64_t stride_sample_x, const int64_t stride_sample_y, const int64_t stride_sample_dst,
        cudaStream_t stream, const mmf_ids_data * ids_data) {
    typedef tile<16, 8, T>     tile_A_16;
    typedef tile<32, 8, T>     tile_A_32;
    typedef tile<16, 8, T>     tile_B_16;
    typedef tile< 8, 8, T>     tile_B_8;

    GGML_ASSERT(ncols_x      % 2 == 0);
    GGML_ASSERT(stride_row   % 2 == 0);
    GGML_ASSERT(stride_col_y % 2 == 0);
    GGML_ASSERT(ids || nchannels_dst % nchannels_x == 0);
    GGML_ASSERT(       nsamples_dst  % nsamples_x  == 0);
    const int64_t channel_ratio = nchannels_dst / nchannels_x;
    const int64_t sample_ratio  = nsamples_dst  / nsamples_x;

    const int device    = ggml_cuda_get_device();
    const int cc        = ggml_cuda_info().devices[device].cc;
    const int warp_size = ggml_cuda_info().devices[device].warp_size;

    int64_t nwarps_best     = 1;
    int64_t niter_best      = (ncols_x + warp_size*2 - 1) / (warp_size*2);
    int64_t max_block_size  = 256;
    for (int64_t nwarps = 2; nwarps <= max_block_size/warp_size; nwarps++) {
        const int64_t niter = (ncols_x + nwarps*warp_size*2 - 1) / (nwarps*warp_size*2);
        if (niter < niter_best) {
            niter_best  = niter;
            nwarps_best = nwarps;
        }
    }

    constexpr int rows_per_block = MMF_ROWS_PER_BLOCK;
    const int nbytes_shared_iter = nwarps_best * (volta_mma_available(cc) ? tile_A_32::I : tile_A_16::I) * (warp_size + 4) * 4;
    const int nbytes_cols_per_block_pad = amd_wmma_available(cc) ? tile_B_16::I : tile_B_8::I;
    const int nbytes_shared_combine = GGML_PAD(cols_per_block, nbytes_cols_per_block_pad) * (nwarps_best*rows_per_block + 4) * 4;
    const int nbytes_shared = std::max(nbytes_shared_iter, nbytes_shared_combine);
    const int nbytes_slotmap = ids ? GGML_PAD(cols_per_block, 16) * sizeof(int) : 0;
    const int nbytes_shared_total = nbytes_shared + nbytes_slotmap;
    const int64_t grid_y = ids ? nchannels_x : nchannels_dst;

    const dim3 block_nums(nrows_x/rows_per_block, grid_y, nsamples_dst);
    const dim3 block_dims(warp_size, nwarps_best, 1);

    switch (nwarps_best) {
        case 1: {
            mul_mat_f_switch_ids<T, cols_per_block, 1>(
                x, y, ids, dst, ncols_x, ncols_dst, nchannels_dst, stride_row, stride_col_y, stride_col_dst,
                stride_col_id, stride_row_id, channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
                sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst, block_nums, block_dims, nbytes_shared_total, stream,
                ids_data);
        } break;
        case 2: {
            mul_mat_f_switch_ids<T, cols_per_block, 2>(
                x, y, ids, dst, ncols_x, ncols_dst, nchannels_dst, stride_row, stride_col_y, stride_col_dst,
                stride_col_id, stride_row_id, channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
                sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst, block_nums, block_dims, nbytes_shared_total, stream,
                ids_data);
        } break;
        case 3: {
            mul_mat_f_switch_ids<T, cols_per_block, 3>(
                x, y, ids, dst, ncols_x, ncols_dst, nchannels_dst, stride_row, stride_col_y, stride_col_dst,
                stride_col_id, stride_row_id, channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
                sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst, block_nums, block_dims, nbytes_shared_total, stream,
                ids_data);
        } break;
        case 4: {
            mul_mat_f_switch_ids<T, cols_per_block, 4>(
                x, y, ids, dst, ncols_x, ncols_dst, nchannels_dst, stride_row, stride_col_y, stride_col_dst,
                stride_col_id, stride_row_id, channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
                sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst, block_nums, block_dims, nbytes_shared_total, stream,
                ids_data);
        } break;
        case 5: {
            mul_mat_f_switch_ids<T, cols_per_block, 5>(
                x, y, ids, dst, ncols_x, ncols_dst, nchannels_dst, stride_row, stride_col_y, stride_col_dst,
                stride_col_id, stride_row_id, channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
                sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst, block_nums, block_dims, nbytes_shared_total, stream,
                ids_data);
        } break;
        case 6: {
            mul_mat_f_switch_ids<T, cols_per_block, 6>(
                x, y, ids, dst, ncols_x, ncols_dst, nchannels_dst, stride_row, stride_col_y, stride_col_dst,
                stride_col_id, stride_row_id, channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
                sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst, block_nums, block_dims, nbytes_shared_total, stream,
                ids_data);
        } break;
        case 7: {
            mul_mat_f_switch_ids<T, cols_per_block, 7>(
                x, y, ids, dst, ncols_x, ncols_dst, nchannels_dst, stride_row, stride_col_y, stride_col_dst,
                stride_col_id, stride_row_id, channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
                sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst, block_nums, block_dims, nbytes_shared_total, stream,
                ids_data);
        } break;
        case 8: {
            mul_mat_f_switch_ids<T, cols_per_block, 8>(
                x, y, ids, dst, ncols_x, ncols_dst, nchannels_dst, stride_row, stride_col_y, stride_col_dst,
                stride_col_id, stride_row_id, channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
                sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst, block_nums, block_dims, nbytes_shared_total, stream,
                ids_data);
        } break;
        default: {
            GGML_ABORT("fatal error");
        } break;
    }

    GGML_UNUSED_VARS(nchannels_y);
}

template <typename T>
static void mul_mat_f_switch_cols_per_block(
        const T * x, const float * y, const int32_t * ids, float * dst,
        const int64_t ncols_x, const int64_t nrows_x, const int64_t ncols_dst,
        const int64_t stride_row, const int64_t stride_col_y, const int64_t stride_col_dst,
        const int64_t stride_col_id, const int stride_row_id,
        const int64_t nchannels_x, const int64_t nchannels_y, const int64_t nchannels_dst,
        const int64_t stride_channel_x, const int64_t stride_channel_y, const int64_t stride_channel_dst, const int64_t nsamples_x,
        const int64_t nsamples_dst, const int64_t stride_sample_x, const int64_t stride_sample_y, const int64_t stride_sample_dst,
        cudaStream_t stream, const mmf_ids_data * ids_data) {

    const int ncols_case = (ids && ncols_dst > 16) ? 16 : ncols_dst;

    GGML_ASSERT(ids || ncols_dst <= 16);

    switch (ncols_case) {
        case  1: {
            mul_mat_f_cuda<T,  1>(x, y, ids, dst, ncols_x, nrows_x, ncols_dst, stride_row, stride_col_y, stride_col_dst,
                stride_col_id, stride_row_id, nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, stream, ids_data);
        } break;
        case  2: {
            mul_mat_f_cuda<T,  2>(x, y, ids, dst, ncols_x, nrows_x, ncols_dst, stride_row, stride_col_y, stride_col_dst,
                stride_col_id, stride_row_id, nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, stream, ids_data);
        } break;
        case  3: {
            mul_mat_f_cuda<T,  3>(x, y, ids, dst, ncols_x, nrows_x, ncols_dst, stride_row, stride_col_y, stride_col_dst,
                stride_col_id, stride_row_id, nchannels_x, nchannels_y,  nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, stream, ids_data);
        } break;
        case  4: {
            mul_mat_f_cuda<T,  4>(x, y, ids, dst, ncols_x, nrows_x, ncols_dst, stride_row, stride_col_y, stride_col_dst,
                stride_col_id, stride_row_id, nchannels_x, nchannels_y,  nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, stream, ids_data);
        } break;
        case  5: {
            mul_mat_f_cuda<T,  5>(x, y, ids, dst, ncols_x, nrows_x, ncols_dst, stride_row, stride_col_y, stride_col_dst,
                stride_col_id, stride_row_id, nchannels_x, nchannels_y,  nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y,  stride_sample_dst, stream, ids_data);
        } break;
        case  6: {
            mul_mat_f_cuda<T,  6>(x, y, ids, dst, ncols_x, nrows_x, ncols_dst, stride_row, stride_col_y, stride_col_dst,
                stride_col_id, stride_row_id, nchannels_x, nchannels_y,  nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, stream, ids_data);
        } break;
        case  7: {
            mul_mat_f_cuda<T,  7>(x, y, ids, dst, ncols_x, nrows_x, ncols_dst, stride_row, stride_col_y, stride_col_dst,
                stride_col_id, stride_row_id, nchannels_x, nchannels_y,  nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, stream, ids_data);
        } break;
        case  8: {
            mul_mat_f_cuda<T,  8>(x, y, ids, dst, ncols_x, nrows_x, ncols_dst, stride_row, stride_col_y, stride_col_dst,
                stride_col_id, stride_row_id, nchannels_x, nchannels_y,  nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, stream, ids_data);
        } break;
        case  9: {
            mul_mat_f_cuda<T,  9>(x, y, ids, dst, ncols_x, nrows_x, ncols_dst, stride_row, stride_col_y, stride_col_dst,
                stride_col_id, stride_row_id, nchannels_x, nchannels_y,  nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, stream, ids_data);
        } break;
        case 10: {
            mul_mat_f_cuda<T, 10>(x, y, ids, dst, ncols_x, nrows_x, ncols_dst, stride_row, stride_col_y, stride_col_dst,
                stride_col_id, stride_row_id, nchannels_x, nchannels_y,  nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, stream, ids_data);
        } break;
        case 11: {
            mul_mat_f_cuda<T, 11>(x, y, ids, dst, ncols_x, nrows_x, ncols_dst, stride_row, stride_col_y, stride_col_dst,
                stride_col_id, stride_row_id, nchannels_x, nchannels_y,  nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, stream, ids_data);
        } break;
        case 12: {
            mul_mat_f_cuda<T, 12>(x, y, ids, dst, ncols_x, nrows_x, ncols_dst, stride_row, stride_col_y, stride_col_dst,
                stride_col_id, stride_row_id, nchannels_x, nchannels_y,  nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, stream, ids_data);
        } break;
        case 13: {
            mul_mat_f_cuda<T, 13>(x, y, ids, dst, ncols_x, nrows_x, ncols_dst, stride_row, stride_col_y, stride_col_dst,
                stride_col_id, stride_row_id, nchannels_x, nchannels_y,  nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, stream, ids_data);
        } break;
        case 14: {
            mul_mat_f_cuda<T, 14>(x, y, ids, dst, ncols_x, nrows_x, ncols_dst, stride_row, stride_col_y, stride_col_dst,
                stride_col_id, stride_row_id, nchannels_x, nchannels_y,  nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, stream, ids_data);
        } break;
        case 15: {
            mul_mat_f_cuda<T, 15>(x, y, ids, dst, ncols_x, nrows_x, ncols_dst, stride_row, stride_col_y, stride_col_dst,
                stride_col_id, stride_row_id, nchannels_x, nchannels_y,  nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, stream, ids_data);
        } break;
        case 16: {
            mul_mat_f_cuda<T, 16>(x, y, ids, dst, ncols_x, nrows_x, ncols_dst, stride_row, stride_col_y, stride_col_dst,
                stride_col_id, stride_row_id, nchannels_x, nchannels_y,  nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, stream, ids_data);
        } break;
        default: {
            GGML_ABORT("fatal error");
        } break;
    }
}

#define DECL_MMF_CASE_HELPER(T, ncols_dst) \
    template void mul_mat_f_cuda<T, ncols_dst>( \
        const T * x, const float * y, const int32_t * ids, float * dst, \
        const int64_t ncols_x, const int64_t nrows_x, int64_t ncols_dst_total, const int64_t stride_row, const int64_t stride_col_y, const int64_t stride_col_dst, \
        const int64_t stride_col_id, const int64_t stride_row_id, \
        const int64_t nchannels_x, const int64_t nchannels_y, const int64_t nchannels_dst, \
        const int64_t stride_channel_x, const int64_t stride_channel_y, const int64_t stride_channel_dst, const int64_t nsamples_x,\
        const int64_t nsamples_dst, const int64_t stride_sample_x, const int64_t stride_sample_y, const int64_t stride_sample_dst, \
        cudaStream_t stream, const mmf_ids_data * ids_data);

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
#define DECL_MMF_CASE_EXTERN(ncols_dst) \
    extern DECL_MMF_CASE_HELPER(float, ncols_dst) \
    extern DECL_MMF_CASE_HELPER(half2, ncols_dst) \
    extern DECL_MMF_CASE_HELPER(nv_bfloat162, ncols_dst)

#define DECL_MMF_CASE(ncols_dst) \
    DECL_MMF_CASE_HELPER(float, ncols_dst) \
    DECL_MMF_CASE_HELPER(half2, ncols_dst) \
    DECL_MMF_CASE_HELPER(nv_bfloat162, ncols_dst)

DECL_MMF_CASE_EXTERN(1);
DECL_MMF_CASE_EXTERN(2);
DECL_MMF_CASE_EXTERN(3);
DECL_MMF_CASE_EXTERN(4);
DECL_MMF_CASE_EXTERN(5);
DECL_MMF_CASE_EXTERN(6);
DECL_MMF_CASE_EXTERN(7);
DECL_MMF_CASE_EXTERN(8);
DECL_MMF_CASE_EXTERN(9);
DECL_MMF_CASE_EXTERN(10);
DECL_MMF_CASE_EXTERN(11);
DECL_MMF_CASE_EXTERN(12);
DECL_MMF_CASE_EXTERN(13);
DECL_MMF_CASE_EXTERN(14);
DECL_MMF_CASE_EXTERN(15);
DECL_MMF_CASE_EXTERN(16);
#else
#define DECL_MMF_CASE(ncols_dst)
#endif
