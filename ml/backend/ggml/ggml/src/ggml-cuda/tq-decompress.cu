#include "tq-decompress.cuh"

// Fused TurboQuant decompress: LloydMaxDQ + inverse FWHT + norm multiply + Q8_0 quantize.
// Input: F32 tensor [packed_d+1, ...] where last element is L2 norm.
// Output: Q8_0 tensor [dim, ...].
// Zero intermediate tensors in the GGML graph — FA reads Q8_0 directly.

// Lloyd-Max centroids for N(0,1) — same as lloyd-max.cu
__device__ static const float tq_centroids_2[] = {
    -1.5104176088f, -0.4527800398f, 0.4527800398f, 1.5104176088f
};
__device__ static const float tq_centroids_3[] = {
    -2.1519481310f, -1.3439092613f, -0.7560052489f, -0.2451209526f,
     0.2451209526f,  0.7560052489f,  1.3439092613f,  2.1519481310f
};
__device__ static const float tq_centroids_4[] = {
    -2.7326368500f, -2.0690790327f, -1.6180234170f, -1.2562091030f,
    -0.9423520268f, -0.6567903640f, -0.3880823390f, -0.1284185740f,
     0.1284185740f,  0.3880823390f,  0.6567903640f,  0.9423520268f,
     1.2562091030f,  1.6180234170f,  2.0690790327f,  2.7326368500f
};

// Position-independent sign hash: O(1) per element, fully parallel.
// Uses splitmix64 finalizer — must match fwht.cu and CPU ops.cpp.
static __device__ __forceinline__ float tq_sign(uint64_t seed, int pos) {
    uint64_t x = seed + (uint64_t)pos * 0x9E3779B97F4A7C15ULL;
    x ^= x >> 30;
    x *= 0xBF58476D1CE4E5B9ULL;
    x ^= x >> 27;
    x *= 0x94D049BB133111EBULL;
    x ^= x >> 31;
    return (x & 1) ? 1.0f : -1.0f;
}

// Fused kernel: one block per row (one head-position vector)
// Outputs Q8_0 blocks directly — no F16 intermediate.
template <int MSE_BITS>
static __global__ void tq_decompress_kernel(
        const float * __restrict__ src,  // packed I32 data + norm (F32 tensor)
        char *        __restrict__ dst,  // Q8_0 output (byte-addressed)
        const int       dim,
        const int       packed_dim,      // = dim * MSE_BITS / 32
        const float     dq_scale,        // 1/sqrt(dim) for Lloyd-Max
        const float     fwht_scale,      // 1/sqrt(dim) for FWHT
        const uint64_t  seed,
        const int64_t   src_stride,      // stride in floats (src is F32)
        const int64_t   dst_stride,      // stride in bytes (Q8_0 row)
        const int64_t   n_rows) {

    const int64_t row = blockIdx.x;
    if (row >= n_rows) return;

    extern __shared__ float smem[];

    const int32_t * src_row = reinterpret_cast<const int32_t *>(src + row * src_stride);
    // Norm is the last F32 element in the row (at index packed_dim)
    const float norm_val = src[row * src_stride + packed_dim];

    const int tid = threadIdx.x;
    const int mask = (1 << MSE_BITS) - 1;

    const float * centroids;
    if constexpr (MSE_BITS == 2) {
        centroids = tq_centroids_2;
    } else if constexpr (MSE_BITS == 3) {
        centroids = tq_centroids_3;
    } else {
        centroids = tq_centroids_4;
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

    // === Phase 3: Sign flip + scale + norm multiply (into shared memory) ===
    const float combined_scale = fwht_scale * norm_val;
    for (int i = tid; i < dim; i += blockDim.x) {
        smem[i] = smem[i] * tq_sign(seed, i) * combined_scale;
    }
    __syncthreads();

    // === Phase 4: Quantize to Q8_0 blocks and write output ===
    char * dst_row = dst + row * dst_stride;
    const int n_blocks = dim / QK8_0;

    for (int b = tid; b < n_blocks; b += blockDim.x) {
        const float * blk_data = smem + b * QK8_0;

        // Find max absolute value in block
        float amax = 0.0f;
        for (int j = 0; j < QK8_0; j++) {
            amax = fmaxf(amax, fabsf(blk_data[j]));
        }

        const float d = amax / 127.0f;
        const float id = d > 0.0f ? 1.0f / d : 0.0f;

        block_q8_0 * blk = (block_q8_0 *)(dst_row + b * sizeof(block_q8_0));
        blk->d = __float2half(d);

        for (int j = 0; j < QK8_0; j++) {
            blk->qs[j] = (int8_t)roundf(blk_data[j] * id);
        }
    }
}

void ggml_cuda_op_tq_decompress(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_Q8_0);

    const int mse_bits = ggml_get_op_params_i32(dst, 0);
    const int dim      = ggml_get_op_params_i32(dst, 1);
    const uint32_t seed_hi = (uint32_t) ggml_get_op_params_i32(dst, 2);
    const uint32_t seed_lo = (uint32_t) ggml_get_op_params_i32(dst, 3);
    const uint64_t seed = ((uint64_t)seed_hi << 32) | (uint64_t)seed_lo;

    const int packed_dim = dim * mse_bits / 32;
    const float dq_scale   = 1.0f / sqrtf((float)dim);
    const float fwht_scale = 1.0f / sqrtf((float)dim);

    const float * src_d = (const float *) src0->data;
    char *        dst_d = (char *) dst->data;

    const int64_t n_rows     = ggml_nrows(dst);
    const int64_t src_stride = src0->nb[1] / sizeof(float);
    const int64_t dst_stride = dst->nb[1];  // bytes (Q8_0 row stride)

    // One block per row, threads = min(dim/2, BLOCK_SIZE)
    const int threads = (int)(dim / 2 < CUDA_TQ_DECOMPRESS_BLOCK_SIZE ? dim / 2 : CUDA_TQ_DECOMPRESS_BLOCK_SIZE);
    const size_t shared_mem = dim * sizeof(float); // single buffer for FWHT + quantize

    cudaStream_t stream = ctx.stream();

    if (mse_bits == 2) {
        tq_decompress_kernel<2><<<n_rows, threads, shared_mem, stream>>>(
            src_d, dst_d, dim, packed_dim, dq_scale, fwht_scale, seed,
            src_stride, dst_stride, n_rows);
    } else if (mse_bits == 3) {
        tq_decompress_kernel<3><<<n_rows, threads, shared_mem, stream>>>(
            src_d, dst_d, dim, packed_dim, dq_scale, fwht_scale, seed,
            src_stride, dst_stride, n_rows);
    } else {
        tq_decompress_kernel<4><<<n_rows, threads, shared_mem, stream>>>(
            src_d, dst_d, dim, packed_dim, dq_scale, fwht_scale, seed,
            src_stride, dst_stride, n_rows);
    }
}
