#include "common.cuh"
#include "ggml.h"
#include "softmax.cuh"

#ifdef GGML_USE_HIP
#include <hip/hip_cooperative_groups.h>
#else
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#endif // GGML_USE_HIP

#include <cstdint>
#include <utility>

template <typename T>
static __device__ __forceinline__ float t2f32(T val) {
    return (float) val;
}

template <>
__device__ float __forceinline__ t2f32<half>(half val) {
    return __half2float(val);
}

struct soft_max_params {

    int64_t nheads;
    uint32_t n_head_log2;
    int64_t ncols;
    int64_t nrows_x;
    int64_t nrows_y;
    int64_t ne00;
    int64_t ne01;
    int64_t ne02;
    int64_t ne03;
    int64_t nb11;
    int64_t nb12;
    int64_t nb13;

    int64_t ne12;
    int64_t ne13;
    float scale;
    float max_bias;
    float m0;
    float m1;
};

// When ncols_template == 0 the bounds for the loops in this function are not known and can't be unrolled.
// As we want to keep pragma unroll for all other cases we supress the clang transformation warning here.
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpass-failed"
#endif // __clang__
template <bool use_shared, int ncols_template, int block_size_template, typename T>
static __global__ void soft_max_f32(
        const float * x, const T * mask, const float * sinks, float * dst, const soft_max_params p) {
    const int ncols = ncols_template == 0 ? p.ncols : ncols_template;

    const int tid  = threadIdx.x;

    const int64_t i03 = blockIdx.z;
    const int64_t i02 = blockIdx.y;
    const int64_t i01 = blockIdx.x;

    //TODO: noncontigous inputs/outputs
    const int rowx = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;

    const int64_t i11 = i01;
    const int64_t i12 = i02 % p.ne12;
    const int64_t i13 = i03 % p.ne13;

    x    += int64_t(rowx)*ncols;
    mask += (i11*p.nb11 + i12*p.nb12 + i13*p.nb13) / sizeof(T) * (mask != nullptr);
    dst  += int64_t(rowx)*ncols;

    const int block_size = block_size_template == 0 ? blockDim.x : block_size_template;

    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;

    const float slope = get_alibi_slope(p.max_bias, i02, p.n_head_log2, p.m0, p.m1);

    extern __shared__ float data_soft_max_f32[];
    float * buf_iw = data_soft_max_f32; // shared memory buffer for inter-warp communication
    // shared memory buffer to cache values between iterations:
    float * vals = use_shared ? buf_iw + WARP_SIZE : dst;

    float max_val = sinks ? sinks[i02] : -INFINITY;

#pragma unroll
    for (int col0 = 0; col0 < ncols; col0 += block_size) {
        const int col = col0 + tid;

        if (ncols_template == 0 && col >= ncols) {
            break;
        }

        const float val = x[col]*p.scale + (mask ? slope*t2f32(mask[col]) : 0.0f);

        vals[col] = val;
        max_val = max(max_val, val);
    }

    // find the max value in the block
    max_val = warp_reduce_max(max_val);
    if (block_size > WARP_SIZE) {
        if (warp_id == 0) {
            buf_iw[lane_id] = -INFINITY;
        }
        __syncthreads();

        if (lane_id == 0) {
            buf_iw[warp_id] = max_val;
        }
        __syncthreads();

        max_val = buf_iw[lane_id];
        max_val = warp_reduce_max(max_val);
    }

    float tmp = 0.0f; // partial sum

#pragma unroll
    for (int col0 = 0; col0 < ncols; col0 += block_size) {
        const int col = col0 + tid;

        if (ncols_template == 0 && col >= ncols) {
            break;
        }

        const float val = expf(vals[col] - max_val);
        tmp += val;
        vals[col] = val;
    }

    // find the sum of exps in the block
    tmp = warp_reduce_sum(tmp);
    if (block_size > WARP_SIZE) {
        __syncthreads();
        if (warp_id == 0) {
            buf_iw[lane_id] = 0.0f;
        }
        __syncthreads();

        if (lane_id == 0) {
            buf_iw[warp_id] = tmp;
        }
        __syncthreads();

        tmp = buf_iw[lane_id];
        tmp = warp_reduce_sum(tmp);
    }

    if (sinks) {
        tmp += expf(sinks[i02] - max_val);
    }

    const float inv_sum = 1.0f / tmp;

#pragma unroll
    for (int col0 = 0; col0 < ncols; col0 += block_size) {
        const int col = col0 + tid;

        if (ncols_template == 0 && col >= ncols) {
            return;
        }

        dst[col] = vals[col] * inv_sum;
    }
}


// TODO: This is a common pattern used across kernels that could be moved to common.cuh + templated
static __device__ float two_stage_warp_reduce_max(float val) {
    val = warp_reduce_max(val);
    if (blockDim.x > WARP_SIZE) {
        assert((blockDim.x <= 1024) && (blockDim.x % WARP_SIZE) == 0);
        __shared__ float local_vals[32];
        const int        warp_id = threadIdx.x / WARP_SIZE;
        const int        lane_id = threadIdx.x % WARP_SIZE;
        if (lane_id == 0) {
            local_vals[warp_id] = val;
        }
        __syncthreads();
        val = -INFINITY;
        if (lane_id < (static_cast<int>(blockDim.x) / WARP_SIZE)) {
            val = local_vals[lane_id];
        }
        return warp_reduce_max(val);
    } else {
        return val;
    }
}

static __device__ float two_stage_warp_reduce_sum(float val) {
    val = warp_reduce_sum(val);
    if (blockDim.x > WARP_SIZE) {
        assert((blockDim.x <= 1024) && (blockDim.x % WARP_SIZE) == 0);
        __shared__ float local_vals[32];
        const int        warp_id = threadIdx.x / WARP_SIZE;
        const int        lane_id = threadIdx.x % WARP_SIZE;
        if (lane_id == 0) {
            local_vals[warp_id] = val;
        }
        __syncthreads();
        val = 0.0f;
        if (lane_id < (static_cast<int>(blockDim.x) / WARP_SIZE)) {
            val = local_vals[lane_id];
        }
        return warp_reduce_sum(val);
    } else {
        return val;
    }
}

// TODO: Template to allow keeping ncols in registers if they fit
static __device__ void soft_max_f32_parallelize_cols_single_row(const float * __restrict__ x,
                                                                float * __restrict__ dst,
                                                                float * __restrict__ tmp_maxs,
                                                                float * __restrict__ tmp_sums,
                                                                const soft_max_params p) {
    namespace cg = cooperative_groups;

    const cg::grid_group g = cg::this_grid();

    const int tid               = threadIdx.x;
    const int col_start         = blockIdx.x * blockDim.x + tid;
    const int n_elem_per_thread = 4;

    float     local_vals[n_elem_per_thread] = { -INFINITY, -INFINITY, -INFINITY, -INFINITY };
    float     local_max                     = -INFINITY;
    const int step_size                     = gridDim.x * blockDim.x;

    // Compute thread-local max
    for (int col = col_start; col < p.ncols;) {
#pragma unroll
        for (int i = 0; i < n_elem_per_thread; i++) {
            const int idx = col + i * step_size;
            local_vals[i] = idx < p.ncols ? x[idx] : -INFINITY;
        }
#pragma unroll
        for (int i = 0; i < n_elem_per_thread; i++) {
            local_max = fmaxf(local_max, local_vals[i]);
        }
        col += step_size * n_elem_per_thread;
    }

    // Compute CTA-level max
    local_max = two_stage_warp_reduce_max(local_max);

    // Store CTA-level max to GMEM
    if (tid == 0) {
        tmp_maxs[blockIdx.x] = local_max;
    }
    g.sync();

    // Compute compute global max from CTA-level maxs
    assert(gridDim.x < blockDim.x);  // currently we only support this case
    if (tid < gridDim.x) {
        local_max = tmp_maxs[tid];
    } else {
        local_max = -INFINITY;
    }
    local_max = two_stage_warp_reduce_max(local_max);

    // Compute softmax dividends, accumulate divisor
    float tmp_expf = 0.0f;
    for (int col = col_start; col < p.ncols;) {
#pragma unroll
        for (int i = 0; i < n_elem_per_thread; i++) {
            const int idx = col + i * step_size;
            local_vals[i] = idx < p.ncols ? x[idx] : -INFINITY;
        }
#pragma unroll
        for (int i = 0; i < n_elem_per_thread; i++) {
            const int idx = col + i * step_size;
            if (idx < p.ncols) {
                const float tmp = expf(local_vals[i] - local_max);
                tmp_expf += tmp;
                dst[idx] = tmp;
            }
        }
        col += step_size * n_elem_per_thread;
    }

    // Reduce divisor within CTA
    tmp_expf = two_stage_warp_reduce_sum(tmp_expf);

    // Store CTA-level sum to GMEM
    if (tid == 0) {
        tmp_sums[blockIdx.x] = tmp_expf;
    }
    g.sync();

    // Compute global sum from CTA-level sums
    if (tid < gridDim.x) {
        tmp_expf = tmp_sums[tid];
    } else {
        tmp_expf = 0.0f;
    }
    tmp_expf = two_stage_warp_reduce_sum(tmp_expf);

    // Divide dividend by global sum + store data
    for (int col = col_start; col < p.ncols;) {
#pragma unroll
        for (int i = 0; i < n_elem_per_thread; i++) {
            const int idx = col + i * step_size;
            local_vals[i] = idx < p.ncols ? dst[idx] : -INFINITY;
        }
#pragma unroll
        for (int i = 0; i < n_elem_per_thread; i++) {
            const int idx = col + i * step_size;
            if (idx < p.ncols) {
                dst[idx] = local_vals[i] / tmp_expf;
            }
        }
        col += step_size * n_elem_per_thread;
    }
}

#ifdef __clang__
#pragma clang diagnostic pop
#endif // __clang__

static __global__ void soft_max_back_f32(
        const float * grad, const float * dstf, float * dst, const int ncols, const float scale) {
    const int tid  = threadIdx.x;
    const int rowx = blockIdx.x;

    grad += int64_t(rowx)*ncols;
    dstf += int64_t(rowx)*ncols;
    dst  += int64_t(rowx)*ncols;

    float dgf_dot = 0.0f; // dot product of dst from forward pass and gradients

    for (int col = tid; col < ncols; col += WARP_SIZE) {
        dgf_dot += dstf[col]*grad[col];
    }

    dgf_dot = warp_reduce_sum(dgf_dot);

    for (int col = tid; col < ncols; col += WARP_SIZE) {
        dst[col] = scale * (grad[col] - dgf_dot) * dstf[col];
    }
}

template<int... Ns, typename T>
static void launch_soft_max_kernels(const float * x, const T * mask, const float * sinks, float * dst,
                             const soft_max_params & p, cudaStream_t stream, dim3 block_dims, dim3 block_nums, size_t nbytes_shared)
{
    const int id       = ggml_cuda_get_device();
    const size_t smpbo = ggml_cuda_info().devices[id].smpbo;

    auto launch_kernel = [=](auto I) -> bool {
        constexpr int ncols = decltype(I)::value;
        constexpr int block = (ncols > 1024 ? 1024 : ncols);

        if (p.ncols == ncols) {
            CUDA_SET_SHARED_MEMORY_LIMIT((soft_max_f32<true, ncols, block, T>), smpbo);
            soft_max_f32<true, ncols, block><<<block_nums, block_dims, nbytes_shared, stream>>>
                (x, mask, sinks, dst, p);
            return true;
        }
        return false;
    };

    // unary fold over launch_kernel
    if ((launch_kernel(std::integral_constant<int, Ns>{}) || ...)) {
        return;
    }

    //default case
    CUDA_SET_SHARED_MEMORY_LIMIT((soft_max_f32<true, 0, 0, T>), smpbo);
    soft_max_f32<true, 0, 0><<<block_nums, block_dims, nbytes_shared, stream>>>(x, mask, sinks, dst, p);
}

__launch_bounds__(8*WARP_SIZE, 1) static __global__ void soft_max_f32_parallelize_cols(const float * __restrict__ x,
                                                     float * __restrict__ dst,
                                                     float * __restrict__ tmp_maxs,
                                                     float * __restrict__ tmp_sums,
                                                     const soft_max_params p)
// We loop over all instead of parallelizing across gridDim.y as cooperative groups
// currently only support synchronizing the complete grid if not launched as a cluster group
// (which requires CC > 9.0)
// https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/device-callable-apis.html#grid-synchronization
// https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/device-callable-apis.html#class-cluster-group
{
    for (int rowx = 0; rowx < p.ne01 * p.ne02 * p.ne03; rowx++) {
        soft_max_f32_parallelize_cols_single_row(x + int64_t(rowx) * p.ncols, dst + int64_t(rowx) * p.ncols, tmp_maxs,
                                                 tmp_sums, p);
    }
}

template <typename T>
static void soft_max_f32_cuda(const float *                                x,
                              const T *                                    mask,
                              const float *                                sinks,
                              float *                                      dst,
                              const soft_max_params &                      params,
                              cudaStream_t                                 stream,
                              [[maybe_unused]] ggml_backend_cuda_context & ctx) {
    int nth = WARP_SIZE;
    const int64_t ncols_x = params.ncols;

    while (nth < ncols_x && nth < CUDA_SOFT_MAX_BLOCK_SIZE) nth *= 2;
    const dim3 block_dims(nth,     1, 1);
    const dim3 block_nums(params.ne01, params.ne02, params.ne03);
    const size_t nbytes_shared = (GGML_PAD(ncols_x, WARP_SIZE) + WARP_SIZE)*sizeof(float);
    static_assert(CUDA_SOFT_MAX_BLOCK_SIZE == 1024, "These values need to be adjusted.");


    const int id       = ggml_cuda_get_device();
    const size_t smpbo = ggml_cuda_info().devices[id].smpbo;


    if (nbytes_shared <= smpbo) {
        launch_soft_max_kernels<32, 64, 128, 256, 512, 1024, 2048, 4096>(x, mask, sinks, dst, params, stream, block_dims, block_nums, nbytes_shared);
    } else {
        // Parallelize across SMs for top-p/dist-sampling
        // The heuristic for parallelizing rows across SMs vs parallelizing single row & looping over all rows was done on the basis of a B6000 GPU and
        // Can be adapted further for lower-SM-count GPUs, though keeping data in registers should be implemented first as that is the optimal solution.
        if (ggml_cuda_info().devices[id].supports_cooperative_launch &&
            ncols_x / (params.ne01 * params.ne02 * params.ne03) > 8192 && mask == nullptr && sinks == nullptr &&
            params.scale == 1.0f && params.max_bias == 0.0f) {
            ggml_cuda_pool_alloc<float> tmp_maxs_alloc(ctx.pool(), ggml_cuda_info().devices[id].nsm * sizeof(float));
            ggml_cuda_pool_alloc<float> tmp_sums_alloc(ctx.pool(), ggml_cuda_info().devices[id].nsm * sizeof(float));

            void * kernel_args[] = { (void *) &x, (void *) &dst, (void *) &tmp_maxs_alloc.ptr,
                                     (void *) &tmp_sums_alloc.ptr, (void *) const_cast<soft_max_params *>(&params) };
            CUDA_CHECK(cudaLaunchCooperativeKernel((void *) soft_max_f32_parallelize_cols,
                                                   dim3(ggml_cuda_info().devices[id].nsm, 1, 1),
                                                   dim3(WARP_SIZE * 8, 1, 1), kernel_args, 0, stream));
        } else {
            const size_t nbytes_shared_low = WARP_SIZE * sizeof(float);
            soft_max_f32<false, 0, 0>
                <<<block_nums, block_dims, nbytes_shared_low, stream>>>(x, mask, sinks, dst, params);
        }
    }
}

static void soft_max_back_f32_cuda(
        const float * grad, const float * dstf, float * dst,
        const int ncols, const int nrows, const float scale, cudaStream_t stream) {
    const dim3 block_dims(WARP_SIZE, 1, 1);
    const dim3 block_nums(nrows,     1, 1);

    soft_max_back_f32<<<block_nums, block_dims, 0, stream>>>(grad, dstf, dst, ncols, scale);
}

void ggml_cuda_op_soft_max(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * src2 = dst->src[2];

    const float * src0_d = (const float *) src0->data;
    const void  * src1_d = src1 ? (const void *) src1->data : nullptr;
    const void  * src2_d = src2 ? (const void *) src2->data : nullptr;
    float       *  dst_d = (float *) dst->data;

    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_ASSERT(!src1 || src1->type == GGML_TYPE_F16 || src1->type == GGML_TYPE_F32); // src1 contains mask and it is optional

    const int64_t nrows_x = ggml_nrows(src0);
    const int64_t nrows_y = src0->ne[1];

    const int64_t ne00 = src0->ne[0];

    float scale    = 1.0f;
    float max_bias = 0.0f;

    memcpy(&scale,    (const float *) dst->op_params + 0, sizeof(float));
    memcpy(&max_bias, (const float *) dst->op_params + 1, sizeof(float));

    const bool use_f16 = (src1 && src1->type == GGML_TYPE_F16);

    const int64_t nb11 = src1 ? src1->nb[1] : 1;
    const int64_t nb12 = src1 ? src1->nb[2] : 1;
    const int64_t nb13 = src1 ? src1->nb[3] : 1;

    const int64_t ne12 = src1 ? src1->ne[2] : 1;
    const int64_t ne13 = src1 ? src1->ne[3] : 1;

    const uint32_t n_head      = src0->ne[2];
    const uint32_t n_head_log2 = 1u << (uint32_t) floorf(log2f((float) n_head));

    const float m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);


    soft_max_params params = {};
    params.nheads = src0->ne[2];
    params.n_head_log2 = n_head_log2;
    params.ncols = ne00;
    params.nrows_x = nrows_x;
    params.nrows_y = nrows_y;
    params.ne00 = src0->ne[0];
    params.ne01 = src0->ne[1];
    params.ne02 = src0->ne[2];
    params.ne03 = src0->ne[3];
    params.nb11 = nb11;
    params.nb12 = nb12;
    params.nb13 = nb13;
    params.ne12 = ne12;
    params.ne13 = ne13;
    params.scale = scale;
    params.max_bias = max_bias;
    params.m0 = m0;
    params.m1 = m1;

    if (use_f16) {
        soft_max_f32_cuda(src0_d, (const half *) src1_d, (const float *) src2_d, dst_d, params, stream, ctx);
    } else {
        soft_max_f32_cuda(src0_d, (const float *) src1_d, (const float *) src2_d, dst_d, params, stream, ctx);
    }
}

void ggml_cuda_op_soft_max_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0]; // grad
    const ggml_tensor * src1 = dst->src[1]; // forward pass output

    const float * src0_d = (const float *) src0->data;
    const float * src1_d = (const float *) src1->data;
    float       * dst_d  = (float       *) dst->data;

    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    const int64_t ncols = src0->ne[0];
    const int64_t nrows = ggml_nrows(src0);

    float scale    = 1.0f;
    float max_bias = 0.0f;

    memcpy(&scale,    (const float *) dst->op_params + 0, sizeof(float));
    memcpy(&max_bias, (const float *) dst->op_params + 1, sizeof(float));

    GGML_ASSERT(max_bias == 0.0f);

    soft_max_back_f32_cuda(src0_d, src1_d, dst_d, ncols, nrows, scale, stream);
}
