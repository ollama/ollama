#include "alibi.cuh"

static __global__ void alibi_f32(const float * x, float * dst, const int ncols, const int k_rows,
                                 const int n_heads_log2_floor, const float m0, const float m1) {
    const int col = blockDim.x*blockIdx.x + threadIdx.x;

    if (col >= ncols) {
        return;
    }

    const int row = blockDim.y*blockIdx.y + threadIdx.y;
    const int i = row*ncols + col;

    const int k = row/k_rows;

    float m_k;
    if (k < n_heads_log2_floor) {
        m_k = powf(m0, k + 1);
    } else {
        m_k = powf(m1, 2 * (k - n_heads_log2_floor) + 1);
    }

    dst[i] = col * m_k + x[i];
}

static void alibi_f32_cuda(const float * x, float * dst, const int ncols, const int nrows,
                           const int k_rows, const int n_heads_log2_floor, const float m0,
                           const float m1, cudaStream_t stream) {
    const dim3 block_dims(CUDA_ALIBI_BLOCK_SIZE, 1, 1);
    const int num_blocks_x = (ncols + CUDA_ALIBI_BLOCK_SIZE - 1) / (CUDA_ALIBI_BLOCK_SIZE);
    const dim3 block_nums(num_blocks_x, nrows, 1);
    alibi_f32<<<block_nums, block_dims, 0, stream>>>(x, dst, ncols, k_rows, n_heads_log2_floor, m0, m1);
}

void ggml_cuda_op_alibi(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne02 = src0->ne[2];
    const int64_t nrows = ggml_nrows(src0);

    //const int n_past = ((int32_t *) dst->op_params)[0];
    const int n_head = ((int32_t *) dst->op_params)[1];
    float max_bias;
    memcpy(&max_bias, (int32_t *) dst->op_params + 2, sizeof(float));

    //GGML_ASSERT(ne01 + n_past == ne00);
    GGML_ASSERT(n_head == ne02);

    const int n_heads_log2_floor = 1 << (int) floor(log2(n_head));

    const float m0 = powf(2.0f, -(max_bias) / n_heads_log2_floor);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_heads_log2_floor);

    alibi_f32_cuda(src0_d, dst_d, ne00, nrows, ne01, n_heads_log2_floor, m0, m1, stream);
}
