#include "lloyd-max.cuh"

// Lloyd-Max optimal centroids for N(0,1) distribution (same as codebook.go)
// Device constant memory for fast access from all threads

// mse_bits=2: 4 centroids, 3 boundaries
__device__ static const float d_centroids_2[] = {
    -1.5104176088f, -0.4527800398f, 0.4527800398f, 1.5104176088f
};
__device__ static const float d_boundaries_2[] = {
    -0.9815988243f, 0.0f, 0.9815988243f
};

// mse_bits=3: 8 centroids, 7 boundaries
__device__ static const float d_centroids_3[] = {
    -2.1519481310f, -1.3439092613f, -0.7560052489f, -0.2451209526f,
     0.2451209526f,  0.7560052489f,  1.3439092613f,  2.1519481310f
};
__device__ static const float d_boundaries_3[] = {
    -1.7479286962f, -1.0499572551f, -0.5005631008f, 0.0f,
     0.5005631008f,  1.0499572551f,  1.7479286962f
};

// mse_bits=4: 16 centroids, 15 boundaries
__device__ static const float d_centroids_4[] = {
    -2.7326368500f, -2.0690790327f, -1.6180234170f, -1.2562091030f,
    -0.9423520268f, -0.6567903640f, -0.3880823390f, -0.1284185740f,
     0.1284185740f,  0.3880823390f,  0.6567903640f,  0.9423520268f,
     1.2562091030f,  1.6180234170f,  2.0690790327f,  2.7326368500f
};
__device__ static const float d_boundaries_4[] = {
    -2.4008579413f, -1.8435512249f, -1.4371162600f, -1.0992995649f,
    -0.7995711954f, -0.5224363515f, -0.2582504565f, 0.0f,
     0.2582504565f,  0.5224363515f,  0.7995711954f,  1.0992995649f,
     1.4371162600f,  1.8435512249f,  2.4008579413f
};

// Quantize kernel: one thread per element, one block per row
// Output is F32 tensor but we store I32 bit patterns via reinterpret_cast.
template <int MSE_BITS>
__global__ void lloyd_max_quantize_kernel(
        const float * __restrict__ src,
        float *       __restrict__ dst,  // F32 tensor, stores I32 bit patterns
        const int     dim,
        const float   scale,
        const int64_t src_stride,  // stride between rows in elements (floats)
        const int64_t dst_stride,  // stride between rows in elements (floats)
        const int64_t n_rows) {

    const int64_t row = blockIdx.x;
    if (row >= n_rows) return;

    const float *   src_row = src + row * src_stride;
    int32_t *       dst_row = reinterpret_cast<int32_t *>(dst + row * dst_stride);
    const int       packed_d = (dim * MSE_BITS + 31) / 32;

    const float * boundaries;
    int n_boundaries;
    if constexpr (MSE_BITS == 2) {
        boundaries   = d_boundaries_2;
        n_boundaries = 3;
    } else if constexpr (MSE_BITS == 3) {
        boundaries   = d_boundaries_3;
        n_boundaries = 7;
    } else {
        boundaries   = d_boundaries_4;
        n_boundaries = 15;
    }

    // Zero output (collaborative across threads)
    for (int i = threadIdx.x; i < packed_d; i += blockDim.x) {
        dst_row[i] = 0;
    }
    __syncthreads();

    // Each thread quantizes one element at a time
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float val = src_row[i];

        // Binary search for nearest centroid
        int left = 0, right = n_boundaries;
        while (left < right) {
            int mid = left + (right - left) / 2;
            if (val <= boundaries[mid] * scale) {
                right = mid;
            } else {
                left = mid + 1;
            }
        }
        int idx = left;

        // Pack into bit-stream using atomicOr
        int64_t bit_offset = (int64_t)i * MSE_BITS;
        int64_t word_idx   = bit_offset / 32;
        int     bit_pos    = (int)(bit_offset % 32);

        atomicOr(&dst_row[word_idx], idx << bit_pos);

        if (bit_pos + MSE_BITS > 32) {
            int overflow = bit_pos + MSE_BITS - 32;
            atomicOr(&dst_row[word_idx + 1], idx >> (MSE_BITS - overflow));
        }
    }
}

// Dequantize kernel: one thread per element, one block per row
// Input is F32 tensor but contains I32 bit patterns via reinterpret_cast.
// Output can be F32 or F16 (T parameter).
template <int MSE_BITS, typename T>
__global__ void lloyd_max_dequantize_kernel(
        const float * __restrict__ src,  // F32 tensor, contains I32 bit patterns
        T *           __restrict__ dst,
        const int       dim,
        const float     scale,
        const int64_t   src_stride,  // stride in floats (src is always F32)
        const int64_t   dst_stride,  // stride in elements of type T
        const int64_t   n_rows) {

    const int64_t row = blockIdx.x;
    if (row >= n_rows) return;

    const int32_t * src_row = reinterpret_cast<const int32_t *>(src + row * src_stride);
    T *             dst_row = dst + row * dst_stride;

    const int mask = (1 << MSE_BITS) - 1;

    const float * centroids;
    if constexpr (MSE_BITS == 2) {
        centroids = d_centroids_2;
    } else if constexpr (MSE_BITS == 3) {
        centroids = d_centroids_3;
    } else {
        centroids = d_centroids_4;
    }

    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        int64_t bit_offset = (int64_t)i * MSE_BITS;
        int64_t word_idx   = bit_offset / 32;
        int     bit_pos    = (int)(bit_offset % 32);

        int idx = (src_row[word_idx] >> bit_pos) & mask;

        if (bit_pos + MSE_BITS > 32) {
            int overflow = bit_pos + MSE_BITS - 32;
            idx |= ((src_row[word_idx + 1] & ((1 << overflow) - 1)) << (MSE_BITS - overflow));
        }

        dst_row[i] = (T)(centroids[idx] * scale);
    }
}

void ggml_cuda_op_lloyd_max_q(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    const int mse_bits = ggml_get_op_params_i32(dst, 0);
    const int dim      = ggml_get_op_params_i32(dst, 1);
    const float scale  = 1.0f / sqrtf((float)dim);

    const float *   src_d = (const float *)   src0->data;
    float *         dst_d = (float *)         dst->data;

    const int64_t n_rows     = ggml_nrows(src0);
    const int64_t src_stride = src0->nb[1] / sizeof(float);
    const int64_t dst_stride = dst->nb[1] / sizeof(float);

    const int threads = min(dim, CUDA_LLOYD_MAX_BLOCK_SIZE);

    cudaStream_t stream = ctx.stream();

    if (mse_bits == 2) {
        lloyd_max_quantize_kernel<2><<<n_rows, threads, 0, stream>>>(
            src_d, dst_d, dim, scale, src_stride, dst_stride, n_rows);
    } else if (mse_bits == 3) {
        lloyd_max_quantize_kernel<3><<<n_rows, threads, 0, stream>>>(
            src_d, dst_d, dim, scale, src_stride, dst_stride, n_rows);
    } else {
        lloyd_max_quantize_kernel<4><<<n_rows, threads, 0, stream>>>(
            src_d, dst_d, dim, scale, src_stride, dst_stride, n_rows);
    }
}

void ggml_cuda_op_lloyd_max_dq(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16);

    const int mse_bits = ggml_get_op_params_i32(dst, 0);
    const int dim      = ggml_get_op_params_i32(dst, 1);
    const float scale  = 1.0f / sqrtf((float)dim);

    const float *   src_d      = (const float *) src0->data;
    const int64_t   n_rows     = ggml_nrows(dst);
    const int64_t   src_stride = src0->nb[1] / sizeof(float);
    const int       threads    = min(dim, CUDA_LLOYD_MAX_BLOCK_SIZE);
    cudaStream_t    stream     = ctx.stream();

    if (dst->type == GGML_TYPE_F16) {
        half * dst_d = (half *) dst->data;
        const int64_t dst_stride = dst->nb[1] / sizeof(half);
        if (mse_bits == 2) {
            lloyd_max_dequantize_kernel<2, half><<<n_rows, threads, 0, stream>>>(
                src_d, dst_d, dim, scale, src_stride, dst_stride, n_rows);
        } else if (mse_bits == 3) {
            lloyd_max_dequantize_kernel<3, half><<<n_rows, threads, 0, stream>>>(
                src_d, dst_d, dim, scale, src_stride, dst_stride, n_rows);
        } else {
            lloyd_max_dequantize_kernel<4, half><<<n_rows, threads, 0, stream>>>(
                src_d, dst_d, dim, scale, src_stride, dst_stride, n_rows);
        }
    } else {
        float * dst_d = (float *) dst->data;
        const int64_t dst_stride = dst->nb[1] / sizeof(float);
        if (mse_bits == 2) {
            lloyd_max_dequantize_kernel<2, float><<<n_rows, threads, 0, stream>>>(
                src_d, dst_d, dim, scale, src_stride, dst_stride, n_rows);
        } else if (mse_bits == 3) {
            lloyd_max_dequantize_kernel<3, float><<<n_rows, threads, 0, stream>>>(
                src_d, dst_d, dim, scale, src_stride, dst_stride, n_rows);
        } else {
            lloyd_max_dequantize_kernel<4, float><<<n_rows, threads, 0, stream>>>(
                src_d, dst_d, dim, scale, src_stride, dst_stride, n_rows);
        }
    }
}
