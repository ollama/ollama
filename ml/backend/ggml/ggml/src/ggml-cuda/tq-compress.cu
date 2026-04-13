#include "tq-compress.cuh"

// Fused TurboQuant compress: L2 norm + normalize + FWHT + Lloyd-Max quantize + concat norm.
// Input: F32 tensor [dim, ...] (one head vector per row)
// Output: F32 tensor [packed_d+1, ...] where packed_d I32 bit patterns + 1 F32 norm.
// Replaces 8 separate GGML graph nodes with one kernel, eliminating kernel launch overhead.

// Lloyd-Max boundaries for N(0,1) — used for quantization (binary search).
// After FWHT without 1/sqrt(d) scaling, data ~ N(0,1), so compare against raw boundaries.
__device__ static const float tqc_boundaries_2[] = {
    -0.9815988243f, 0.0f, 0.9815988243f
};
__device__ static const float tqc_boundaries_3[] = {
    -1.7479286962f, -1.0499572551f, -0.5005631008f, 0.0f,
     0.5005631008f,  1.0499572551f,  1.7479286962f
};
__device__ static const float tqc_boundaries_4[] = {
    -2.4008579413f, -1.8435512249f, -1.4371162600f, -1.0992995649f,
    -0.7995711954f, -0.5224363515f, -0.2582504565f, 0.0f,
     0.2582504565f,  0.5224363515f,  0.7995711954f,  1.0992995649f,
     1.4371162600f,  1.8435512249f,  2.4008579413f
};

// Position-independent sign hash: splitmix64 finalizer — must match fwht.cu and tq-decompress.cu.
static __device__ __forceinline__ float tqc_sign(uint64_t seed, int pos) {
    uint64_t x = seed + (uint64_t)pos * 0x9E3779B97F4A7C15ULL;
    x ^= x >> 30;
    x *= 0xBF58476D1CE4E5B9ULL;
    x ^= x >> 27;
    x *= 0x94D049BB133111EBULL;
    x ^= x >> 31;
    return (x & 1) ? 1.0f : -1.0f;
}

// Warp-level reduction for L2 norm computation
static __device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Fused kernel: one block per row (one head-position vector)
template <int MSE_BITS>
static __global__ void tq_compress_kernel(
        const float * __restrict__ src,   // F32 input
        float *       __restrict__ dst,   // F32 output (I32 bitpatterns + norm)
        const int       dim,
        const int       packed_dim,       // = dim * MSE_BITS / 32
        const float     fwht_scale,       // 1/sqrt(dim) for FWHT
        const uint64_t  seed,
        const int64_t   src_stride,       // stride in floats
        const int64_t   dst_stride,       // stride in floats
        const int64_t   n_rows) {

    const int64_t row = blockIdx.x;
    if (row >= n_rows) return;

    extern __shared__ float smem[];
    // smem layout: [0..dim-1] = work buffer, [dim] = norm reduction scratch

    const float * src_row = src + row * src_stride;
    int32_t * dst_row = reinterpret_cast<int32_t *>(dst + row * dst_stride);
    float   * dst_row_f = dst + row * dst_stride;

    const int tid = threadIdx.x;
    const int n_boundaries = (1 << MSE_BITS) - 1;

    const float * boundaries;
    if constexpr (MSE_BITS == 2) {
        boundaries = tqc_boundaries_2;
    } else if constexpr (MSE_BITS == 3) {
        boundaries = tqc_boundaries_3;
    } else {
        boundaries = tqc_boundaries_4;
    }

    // === Phase 1: Load input into shared memory + compute partial L2 norm ===
    float partial_norm_sq = 0.0f;
    for (int i = tid; i < dim; i += blockDim.x) {
        float v = src_row[i];
        smem[i] = v;
        partial_norm_sq += v * v;
    }

    // Reduce norm_sq across warp
    partial_norm_sq = warp_reduce_sum(partial_norm_sq);

    // Cross-warp reduction via shared memory
    // Use smem[dim..dim+31] as scratch for warp results
    float * norm_scratch = smem + dim;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int n_warps = (blockDim.x + 31) / 32;

    if (lane_id == 0) {
        norm_scratch[warp_id] = partial_norm_sq;
    }
    __syncthreads();

    // Thread 0 reduces across warps
    float norm_sq = 0.0f;
    if (tid == 0) {
        for (int w = 0; w < n_warps; w++) {
            norm_sq += norm_scratch[w];
        }
        norm_scratch[0] = norm_sq; // broadcast result
    }
    __syncthreads();
    norm_sq = norm_scratch[0];

    float norm_val = sqrtf(norm_sq);
    float norm_inv = (norm_val > 1e-12f) ? (1.0f / norm_val) : 0.0f;

    // === Phase 2: Normalize + sign flip ===
    for (int i = tid; i < dim; i += blockDim.x) {
        smem[i] = smem[i] * norm_inv * tqc_sign(seed, i);
    }
    __syncthreads();

    // === Phase 3: FWHT butterfly stages ===
    for (int64_t len = 1; len < dim; len <<= 1) {
        for (int64_t j = tid; j < dim / 2; j += blockDim.x) {
            int64_t block_idx = j / len;
            int64_t k = j % len;
            int64_t idx0 = block_idx * (len << 1) + k;
            int64_t idx1 = idx0 + len;

            float a = smem[idx0];
            float b = smem[idx1];
            smem[idx0] = a + b;
            smem[idx1] = a - b;
        }
        __syncthreads();
    }

    // === Phase 4: Zero output packed words ===
    for (int i = tid; i < packed_dim; i += blockDim.x) {
        dst_row[i] = 0;
    }
    __syncthreads();

    // === Phase 5: Scale + Lloyd-Max quantize + pack into bitstream ===
    // After butterfly, data ~ N(0,1) since input was unit-normalized.
    // The decompress kernel uses centroids * (1/sqrt(d)) = centroids * fwht_scale.
    // Here we scale data by fwht_scale first, then compare against boundaries * fwht_scale.
    // Equivalently: compare unscaled data against unscaled boundaries (same result).
    for (int i = tid; i < dim; i += blockDim.x) {
        float val = smem[i] * fwht_scale;

        // Binary search for quantization index
        int left = 0, right = n_boundaries;
        while (left < right) {
            int mid = left + (right - left) / 2;
            if (val <= boundaries[mid] * fwht_scale) {
                right = mid;
            } else {
                left = mid + 1;
            }
        }
        int idx = left;

        // Pack into bitstream using atomicOr
        int64_t bit_offset = (int64_t)i * MSE_BITS;
        int64_t word_idx   = bit_offset / 32;
        int     bit_pos    = (int)(bit_offset % 32);

        atomicOr(&dst_row[word_idx], (int32_t)((uint32_t)idx << bit_pos));
        if (bit_pos + MSE_BITS > 32) {
            int overflow = bit_pos + MSE_BITS - 32;
            atomicOr(&dst_row[word_idx + 1], (int32_t)((uint32_t)idx >> (MSE_BITS - overflow)));
        }
    }

    // === Phase 6: Write L2 norm as last element ===
    if (tid == 0) {
        dst_row_f[packed_dim] = norm_val;
    }
}

// F16 input variant
template <int MSE_BITS>
static __global__ void tq_compress_kernel_f16(
        const half  * __restrict__ src,   // F16 input
        float *       __restrict__ dst,   // F32 output (I32 bitpatterns + norm)
        const int       dim,
        const int       packed_dim,
        const float     fwht_scale,
        const uint64_t  seed,
        const int64_t   src_stride,       // stride in halfs
        const int64_t   dst_stride,       // stride in floats
        const int64_t   n_rows) {

    const int64_t row = blockIdx.x;
    if (row >= n_rows) return;

    extern __shared__ float smem[];

    const half * src_row = src + row * src_stride;
    int32_t * dst_row = reinterpret_cast<int32_t *>(dst + row * dst_stride);
    float   * dst_row_f = dst + row * dst_stride;

    const int tid = threadIdx.x;
    const int n_boundaries = (1 << MSE_BITS) - 1;

    const float * boundaries;
    if constexpr (MSE_BITS == 2) {
        boundaries = tqc_boundaries_2;
    } else if constexpr (MSE_BITS == 3) {
        boundaries = tqc_boundaries_3;
    } else {
        boundaries = tqc_boundaries_4;
    }

    // === Phase 1: Load F16 input → F32 shared memory + partial L2 norm ===
    float partial_norm_sq = 0.0f;
    for (int i = tid; i < dim; i += blockDim.x) {
        float v = __half2float(src_row[i]);
        smem[i] = v;
        partial_norm_sq += v * v;
    }

    // Warp + cross-warp reduction
    partial_norm_sq = warp_reduce_sum(partial_norm_sq);
    float * norm_scratch = smem + dim;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int n_warps = (blockDim.x + 31) / 32;

    if (lane_id == 0) {
        norm_scratch[warp_id] = partial_norm_sq;
    }
    __syncthreads();

    float norm_sq = 0.0f;
    if (tid == 0) {
        for (int w = 0; w < n_warps; w++) {
            norm_sq += norm_scratch[w];
        }
        norm_scratch[0] = norm_sq;
    }
    __syncthreads();
    norm_sq = norm_scratch[0];

    float norm_val = sqrtf(norm_sq);
    float norm_inv = (norm_val > 1e-12f) ? (1.0f / norm_val) : 0.0f;

    // === Phase 2: Normalize + sign flip ===
    for (int i = tid; i < dim; i += blockDim.x) {
        smem[i] = smem[i] * norm_inv * tqc_sign(seed, i);
    }
    __syncthreads();

    // === Phase 3: FWHT butterfly stages ===
    for (int64_t len = 1; len < dim; len <<= 1) {
        for (int64_t j = tid; j < dim / 2; j += blockDim.x) {
            int64_t block_idx = j / len;
            int64_t k = j % len;
            int64_t idx0 = block_idx * (len << 1) + k;
            int64_t idx1 = idx0 + len;

            float a = smem[idx0];
            float b = smem[idx1];
            smem[idx0] = a + b;
            smem[idx1] = a - b;
        }
        __syncthreads();
    }

    // === Phase 4: Zero output + quantize + pack ===
    for (int i = tid; i < packed_dim; i += blockDim.x) {
        dst_row[i] = 0;
    }
    __syncthreads();

    for (int i = tid; i < dim; i += blockDim.x) {
        float val = smem[i] * fwht_scale;

        int left = 0, right = n_boundaries;
        while (left < right) {
            int mid = left + (right - left) / 2;
            if (val <= boundaries[mid] * fwht_scale) {
                right = mid;
            } else {
                left = mid + 1;
            }
        }
        int idx = left;

        int64_t bit_offset = (int64_t)i * MSE_BITS;
        int64_t word_idx   = bit_offset / 32;
        int     bit_pos    = (int)(bit_offset % 32);

        atomicOr(&dst_row[word_idx], (int32_t)((uint32_t)idx << bit_pos));
        if (bit_pos + MSE_BITS > 32) {
            int overflow = bit_pos + MSE_BITS - 32;
            atomicOr(&dst_row[word_idx + 1], (int32_t)((uint32_t)idx >> (MSE_BITS - overflow)));
        }
    }

    // === Phase 5: Write L2 norm ===
    if (tid == 0) {
        dst_row_f[packed_dim] = norm_val;
    }
}

void ggml_cuda_op_tq_compress(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    const int mse_bits = ggml_get_op_params_i32(dst, 0);
    const int dim      = ggml_get_op_params_i32(dst, 1);
    const uint32_t seed_hi = (uint32_t) ggml_get_op_params_i32(dst, 2);
    const uint32_t seed_lo = (uint32_t) ggml_get_op_params_i32(dst, 3);
    const uint64_t seed = ((uint64_t)seed_hi << 32) | (uint64_t)seed_lo;

    const int packed_dim   = dim * mse_bits / 32;
    const float fwht_scale = 1.0f / sqrtf((float)dim);

    const int64_t n_rows = ggml_nrows(src0);

    // One block per row, threads = min(dim/2, BLOCK_SIZE)
    const int threads = (int)(dim / 2 < CUDA_TQ_COMPRESS_BLOCK_SIZE ? dim / 2 : CUDA_TQ_COMPRESS_BLOCK_SIZE);
    // Shared memory: dim floats (work buffer) + 32 floats (warp reduction scratch)
    const size_t shared_mem = (dim + 32) * sizeof(float);

    cudaStream_t stream = ctx.stream();

    if (src0->type == GGML_TYPE_F16) {
        const half * src_d = (const half *) src0->data;
        float * dst_d = (float *) dst->data;
        const int64_t src_stride = src0->nb[1] / sizeof(half);
        const int64_t dst_stride = dst->nb[1] / sizeof(float);

        if (mse_bits == 2) {
            tq_compress_kernel_f16<2><<<n_rows, threads, shared_mem, stream>>>(
                src_d, dst_d, dim, packed_dim, fwht_scale, seed,
                src_stride, dst_stride, n_rows);
        } else if (mse_bits == 3) {
            tq_compress_kernel_f16<3><<<n_rows, threads, shared_mem, stream>>>(
                src_d, dst_d, dim, packed_dim, fwht_scale, seed,
                src_stride, dst_stride, n_rows);
        } else {
            tq_compress_kernel_f16<4><<<n_rows, threads, shared_mem, stream>>>(
                src_d, dst_d, dim, packed_dim, fwht_scale, seed,
                src_stride, dst_stride, n_rows);
        }
    } else {
        const float * src_d = (const float *) src0->data;
        float * dst_d = (float *) dst->data;
        const int64_t src_stride = src0->nb[1] / sizeof(float);
        const int64_t dst_stride = dst->nb[1] / sizeof(float);

        if (mse_bits == 2) {
            tq_compress_kernel<2><<<n_rows, threads, shared_mem, stream>>>(
                src_d, dst_d, dim, packed_dim, fwht_scale, seed,
                src_stride, dst_stride, n_rows);
        } else if (mse_bits == 3) {
            tq_compress_kernel<3><<<n_rows, threads, shared_mem, stream>>>(
                src_d, dst_d, dim, packed_dim, fwht_scale, seed,
                src_stride, dst_stride, n_rows);
        } else {
            tq_compress_kernel<4><<<n_rows, threads, shared_mem, stream>>>(
                src_d, dst_d, dim, packed_dim, fwht_scale, seed,
                src_stride, dst_stride, n_rows);
        }
    }
}
