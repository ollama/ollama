#include "fwht.cuh"

// Fast Walsh-Hadamard Transform (TurboQuant rotation)
// Each thread block processes one vector of dimension d.
// The butterfly stages are synchronized via __syncthreads().
// Supports both F32 and F16 input/output (computation in F32 shared memory).

// Position-independent sign hash: O(1) per element, fully parallel.
// Replaces sequential xorshift which required O(i) iterations for element i.
// Uses splitmix64 finalizer for high-quality random bits.
static __device__ __forceinline__ float tq_sign(uint64_t seed, int pos) {
    uint64_t x = seed + (uint64_t)pos * 0x9E3779B97F4A7C15ULL;
    x ^= x >> 30;
    x *= 0xBF58476D1CE4E5B9ULL;
    x ^= x >> 27;
    x *= 0x94D049BB133111EBULL;
    x ^= x >> 31;
    return (x & 1) ? 1.0f : -1.0f;
}

// Templated kernel supporting F32 and F16 I/O with F32 shared memory computation
template <typename T>
static __global__ void fwht_kernel(
        const T     * __restrict__ src,
        T           * __restrict__ dst,
        const int64_t d,
        const int64_t stride,       // number of elements between consecutive vectors
        const int64_t nr,           // total number of vectors
        const uint64_t seed,
        const int inverse,
        const float scale) {

    const int64_t vec_idx = blockIdx.x;
    if (vec_idx >= nr) return;

    extern __shared__ float smem[];

    const T * src_vec = src + vec_idx * stride;
    T       * dst_vec = dst + vec_idx * stride;

    const int tid = threadIdx.x;

    // Load vector into shared memory (converting to F32 for computation)
    for (int i = tid; i < d; i += blockDim.x) {
        smem[i] = (float)src_vec[i];
    }
    __syncthreads();

    if (!inverse) {
        // Forward: apply D first (sign flip), then Hadamard butterfly
        for (int i = tid; i < d; i += blockDim.x) {
            smem[i] *= tq_sign(seed, i);
        }
        __syncthreads();

        // Hadamard butterfly stages
        for (int64_t len = 1; len < d; len <<= 1) {
            for (int64_t j = tid; j < d / 2; j += blockDim.x) {
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
    } else {
        // Inverse: Hadamard butterfly first, then apply D
        for (int64_t len = 1; len < d; len <<= 1) {
            for (int64_t j = tid; j < d / 2; j += blockDim.x) {
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

        for (int i = tid; i < d; i += blockDim.x) {
            smem[i] *= tq_sign(seed, i);
        }
        __syncthreads();
    }

    // Scale by 1/sqrt(d) and write to output (converting back to T)
    for (int i = tid; i < d; i += blockDim.x) {
        dst_vec[i] = (T)(smem[i] * scale);
    }
}

void ggml_cuda_op_fwht(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == dst->type);
    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);

    const int64_t d  = src0->ne[0];
    const int64_t nr = ggml_nrows(src0);
    const int64_t stride = d; // contiguous along dim 0

    GGML_ASSERT(d > 0 && (d & (d - 1)) == 0); // power of 2
    GGML_ASSERT(d <= 1024); // shared memory limit

    const uint32_t seed_hi = (uint32_t) dst->op_params[0];
    const uint32_t seed_lo = (uint32_t) dst->op_params[1];
    const int inverse      = dst->op_params[2];
    const uint64_t seed    = ((uint64_t) seed_hi << 32) | (uint64_t) seed_lo;

    const float scale = 1.0f / sqrtf((float) d);

    // One block per vector, threads per block = min(d/2, 256)
    const int threads_per_block = (int) (d / 2 < CUDA_FWHT_BLOCK_SIZE ? d / 2 : CUDA_FWHT_BLOCK_SIZE);
    const int num_blocks = (int) nr;
    const size_t shared_mem = d * sizeof(float); // single buffer for FWHT data

    if (src0->type == GGML_TYPE_F16) {
        const ggml_fp16_t * src0_d = (const ggml_fp16_t *) src0->data;
        ggml_fp16_t * dst_d = (ggml_fp16_t *) dst->data;
        fwht_kernel<__half><<<num_blocks, threads_per_block, shared_mem, stream>>>(
            (const __half *)src0_d, (__half *)dst_d, d, stride, nr, seed, inverse, scale);
    } else {
        const float * src0_d = (const float *) src0->data;
        float * dst_d = (float *) dst->data;
        fwht_kernel<float><<<num_blocks, threads_per_block, shared_mem, stream>>>(
            src0_d, dst_d, d, stride, nr, seed, inverse, scale);
    }
}
