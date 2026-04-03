#include "tq-decompress.cuh"

// Fused TurboQuant decompress: LloydMaxDQ + inverse FWHT + norm multiply in one kernel.
// Input: F32 tensor [packed_d+1, ...] where last element is L2 norm.
// Output: F16 tensor [dim, ...].
// Zero intermediate tensors in the GGML graph.

// Lloyd-Max centroids for N(0,1) — same as lloyd-max.cu
__device__ static const float tq_centroids_2[] = {
    -1.5104176088f, -0.4527800398f, 0.4527800398f, 1.5104176088f
};
__device__ static const float tq_centroids_3[] = {
    -2.1519481310f, -1.3439092613f, -0.7560052489f, -0.2451209526f,
     0.2451209526f,  0.7560052489f,  1.3439092613f,  2.1519481310f
};

// Xorshift64 matching CPU/FWHT implementation
static __device__ __forceinline__ uint64_t tq_xorshift64(uint64_t x) {
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    return x;
}

// Fused kernel: one block per row (one head-position vector)
// Shared memory holds the full dim-sized vector for FWHT butterfly.
template <int MSE_BITS>
static __global__ void tq_decompress_kernel(
        const float * __restrict__ src,  // packed I32 data + norm (F32 tensor)
        half *        __restrict__ dst,  // F16 output
        const int       dim,
        const int       packed_dim,      // = dim * MSE_BITS / 32
        const float     dq_scale,        // 1/sqrt(dim) for Lloyd-Max
        const float     fwht_scale,      // 1/sqrt(dim) for FWHT
        const uint64_t  seed,
        const int64_t   src_stride,      // stride in floats (src is F32)
        const int64_t   dst_stride,      // stride in halfs
        const int64_t   n_rows) {

    const int64_t row = blockIdx.x;
    if (row >= n_rows) return;

    extern __shared__ float smem[];

    const int32_t * src_row = reinterpret_cast<const int32_t *>(src + row * src_stride);
    // Norm is the last F32 element in the row (at index packed_dim)
    const float norm_val = src[row * src_stride + packed_dim];
    half * dst_row = dst + row * dst_stride;

    const int tid = threadIdx.x;
    const int mask = (1 << MSE_BITS) - 1;

    const float * centroids;
    if constexpr (MSE_BITS == 2) {
        centroids = tq_centroids_2;
    } else {
        centroids = tq_centroids_3;
    }

    // === Phase 1: Lloyd-Max dequantize into shared memory ===
    for (int i = tid; i < dim; i += blockDim.x) {
        int64_t bit_offset = (int64_t)i * MSE_BITS;
        int64_t word_idx   = bit_offset / 32;
        int     bit_pos    = (int)(bit_offset % 32);

        int idx = (src_row[word_idx] >> bit_pos) & mask;

        if (bit_pos + MSE_BITS > 32) {
            int overflow = bit_pos + MSE_BITS - 32;
            idx |= ((src_row[word_idx + 1] & ((1 << overflow) - 1)) << (MSE_BITS - overflow));
        }

        smem[i] = centroids[idx] * dq_scale;
    }
    __syncthreads();

    // === Phase 2: Inverse FWHT (butterfly first, then sign flip) ===

    // Step 2a: Hadamard butterfly stages
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

    // Step 2b: Apply diagonal D (sign flips) + scale + norm multiply + write F16
    const float combined_scale = fwht_scale * norm_val;
    for (int i = tid; i < dim; i += blockDim.x) {
        uint64_t rng = seed;
        for (int j = 0; j <= i; j++) {
            rng = rng ^ (rng << 13);
            rng = rng ^ (rng >> 7);
            rng = rng ^ (rng << 17);
        }
        float sign = (rng & 1) ? 1.0f : -1.0f;
        dst_row[i] = __float2half(smem[i] * sign * combined_scale);
    }
}

void ggml_cuda_op_tq_decompress(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F16);

    const int mse_bits = ggml_get_op_params_i32(dst, 0);
    const int dim      = ggml_get_op_params_i32(dst, 1);
    const uint32_t seed_hi = (uint32_t) ggml_get_op_params_i32(dst, 2);
    const uint32_t seed_lo = (uint32_t) ggml_get_op_params_i32(dst, 3);
    const uint64_t seed = ((uint64_t)seed_hi << 32) | (uint64_t)seed_lo;

    const int packed_dim = dim * mse_bits / 32;
    const float dq_scale   = 1.0f / sqrtf((float)dim);
    const float fwht_scale = 1.0f / sqrtf((float)dim);

    const float * src_d = (const float *) src0->data;
    half *        dst_d = (half *) dst->data;

    const int64_t n_rows     = ggml_nrows(dst);
    const int64_t src_stride = src0->nb[1] / sizeof(float);
    const int64_t dst_stride = dst->nb[1] / sizeof(half);

    // One block per row, threads = min(dim/2, BLOCK_SIZE)
    const int threads = (int)(dim / 2 < CUDA_TQ_DECOMPRESS_BLOCK_SIZE ? dim / 2 : CUDA_TQ_DECOMPRESS_BLOCK_SIZE);
    const size_t shared_mem = dim * sizeof(float);

    cudaStream_t stream = ctx.stream();

    if (mse_bits == 2) {
        tq_decompress_kernel<2><<<n_rows, threads, shared_mem, stream>>>(
            src_d, dst_d, dim, packed_dim, dq_scale, fwht_scale, seed,
            src_stride, dst_stride, n_rows);
    } else {
        tq_decompress_kernel<3><<<n_rows, threads, shared_mem, stream>>>(
            src_d, dst_d, dim, packed_dim, dq_scale, fwht_scale, seed,
            src_stride, dst_stride, n_rows);
    }
}
