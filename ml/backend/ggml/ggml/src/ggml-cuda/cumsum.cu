#include <algorithm>
#include "cumsum.cuh"
#include "convert.cuh"
#include "ggml-cuda/common.cuh"
#include "ggml.h"

#ifdef GGML_CUDA_USE_CUB
#   include <cub/device/device_scan.cuh>
#endif // GGML_CUDA_USE_CUB

template<typename T, int BLOCK_SIZE>
static __global__ void cumsum_cub_kernel(
        const T * __restrict__ src,
        T * __restrict__ dst,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t  s01, const int64_t  s02, const int64_t  s03,
        const int64_t   s1,  const int64_t   s2,  const int64_t   s3) {
#ifdef GGML_CUDA_USE_CUB
    using BlockScan = cub::BlockScan<T, BLOCK_SIZE>;

    __shared__ typename BlockScan::TempStorage temp_storage;
    __shared__ T block_carry;      // carry from previous tile

    const int tid = threadIdx.x;

    const int64_t i1 = blockIdx.x;
    const int64_t i2 = blockIdx.y;
    const int64_t i3 = blockIdx.z;

    if (i1 >= ne01 || i2 >= ne02 || i3 >= ne03) {
        return;
    }

    const T * src_row = src + i1 * s01 + i2 * s02 + i3 * s03;
    T *       dst_row = dst + i1 * s1  + i2 * s2  + i3 * s3;

    if (tid == 0) {
        block_carry = 0;
    }
    __syncthreads();

    for (int64_t start = 0; start < ne00; start += BLOCK_SIZE) {
        int64_t idx = start + tid;
        T x = (idx < ne00) ? src_row[idx] : T(0);

        T inclusive;
        T block_total;
        BlockScan(temp_storage).InclusiveSum(x, inclusive, block_total);

        __syncthreads();

        T final_val = inclusive + block_carry;

        // store result
        if (idx < ne00) {
            dst_row[idx] = final_val;
        }

        __syncthreads();

        if (tid == 0) {
            block_carry += block_total;
        }

        __syncthreads();
    }
#else
    NO_DEVICE_CODE;
#endif // GGML_CUDA_USE_CUB
}

// Fallback kernel implementation (original)
template<typename T>
static __global__ void cumsum_kernel(
        const T * src, T * dst,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t  s00, const int64_t  s01, const int64_t  s02, const int64_t  s03,
        const int64_t   s0, const int64_t   s1, const int64_t   s2, const int64_t   s3) {

    GGML_UNUSED_VARS(s00, s0);

    const int tid = threadIdx.x;
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    const int lane = tid % warp_size;
    const int warp = tid / warp_size;
    const int warps_per_block = blockDim.x / warp_size;

    extern __shared__ float smem[];
    float * s_vals = smem;
    float * s_warp_sums = smem + blockDim.x;
    float * s_carry = smem + blockDim.x + warps_per_block;
    float * s_chunk_total = s_carry + 1;

    // Initialize carry
    if (tid == 0) {
        *s_carry = 0.0f;
    }
    __syncthreads();

    const int64_t i3 = blockIdx.z;
    const int64_t i2 = blockIdx.y;
    const int64_t i1 = blockIdx.x;
    if (i3 >= ne03 || i2 >= ne02 || i1 >= ne01) {
        return;
    }

    const T * src_row = src + i1 * s01 + i2 * s02 + i3 * s03;
    T       * dst_row = dst + i1 * s1  + i2 * s2  + i3 * s3;

    for (int64_t start = 0; start < ne00; start += blockDim.x) {
        int64_t idx = start + tid;
        float val = (idx < ne00) ? ggml_cuda_cast<float, T>(src_row[idx]) : 0.0f;

        // 1. Warp inclusive scan
        val = warp_prefix_inclusive_sum<T, warp_size>(val);
        s_vals[tid] = val;

        // Store warp total
        if (lane == warp_size - 1) {
            s_warp_sums[warp] = val;
        }
        __syncthreads();

        // 2. Exclusive scan of warp sums (warp 0 only)
        if (warp == 0) {
            float w = (tid < warps_per_block) ? s_warp_sums[tid] : 0.0f;
            float inc = warp_prefix_inclusive_sum<T, warp_size>(w);
            if (tid < warps_per_block) {
                s_warp_sums[tid] = inc - w;   // exclusive sum
            }
            if (tid == warps_per_block - 1) {
                *s_chunk_total = inc;          // total sum of this chunk
            }
        }
        __syncthreads();

        float carry = *s_carry;
        float final_val = s_vals[tid] + s_warp_sums[warp] + carry;
        if (idx < ne00) {
            dst_row[idx] = ggml_cuda_cast<T, float>(final_val);
        }
        __syncthreads();

        // Update carry for next chunk
        if (tid == 0) {
            *s_carry += *s_chunk_total;
        }
        __syncthreads();
    }
}

template<typename T>
static void cumsum_cuda(
        const T * src, T * dst,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t nb00, const int64_t nb01, const int64_t nb02, const int64_t nb03,
        const int64_t  nb0,  const int64_t nb1, const int64_t  nb2, const int64_t  nb3,
        cudaStream_t stream) {

    const size_t type_size = sizeof(T);
    bool use_cub = false;
#ifdef GGML_CUDA_USE_CUB
    // Check if we can use CUB (data must be contiguous along innermost dimension)
    const bool is_contiguous = (nb00 == type_size) && (nb0 == type_size);

    if (is_contiguous) {
        use_cub = true;
    }
#endif // GGML_CUDA_USE_CUB
    dim3 grid_dims(ne01, ne02, ne03);
    const auto &info = ggml_cuda_info().devices[ggml_cuda_get_device()];
    const int warp_size = info.warp_size;
    const int num_warps = (ne00 + warp_size - 1) / warp_size;
    int block_size = num_warps * warp_size;
    block_size = std::min(block_size, CUDA_CUMSUM_BLOCK_SIZE);
    dim3 block_dims(block_size, 1, 1);
    const int warps_per_block = block_size / warp_size;
    const size_t shmem_size = (block_size + warps_per_block + 2) * sizeof(float);

    if (use_cub) {
        cumsum_cub_kernel<T, CUDA_CUMSUM_BLOCK_SIZE><<<grid_dims, CUDA_CUMSUM_BLOCK_SIZE, 0, stream>>>(
            src, dst,
            ne00, ne01, ne02, ne03,
            nb01 / type_size, nb02 / type_size, nb03 / type_size,
            nb1 / type_size,  nb2 / type_size,  nb3 / type_size
        );
    } else {
        cumsum_kernel<<<grid_dims, block_dims, shmem_size, stream>>>(
            src, dst,
            ne00, ne01, ne02, ne03,
            nb00 / type_size, nb01 / type_size, nb02 / type_size, nb03 / type_size,
            nb0 / type_size, nb1 / type_size, nb2 / type_size, nb3 / type_size
        );
    }
}

void ggml_cuda_op_cumsum(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == dst->type);
    switch(src0->type) {
        case GGML_TYPE_F32:
            {
                cumsum_cuda(
                    (const float *)src0->data, (float *)dst->data,
                    src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
                    src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
                    dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3],
                    stream
                );
            } break;
        // We do not support those on CPU for now anyway, so comment them out because they cause errors on some CI platforms
        /*case GGML_TYPE_F16:
            {
                cumsum_cuda(
                    (const half *)src0->data, (half *)dst->data,
                    src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
                    src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
                    dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3],
                    stream
                );
            } break;
        case GGML_TYPE_BF16:
            {
                cumsum_cuda(
                    (const nv_bfloat16 *)src0->data, (nv_bfloat16 *)dst->data,
                    src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
                    src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
                    dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3],
                    stream
                );
            } break;*/
        default:
            GGML_ABORT("fatal error");
    }
}
