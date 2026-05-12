#include "ggml-cuda/common.cuh"
#include "roll.cuh"

static __forceinline__ __device__ int64_t wrap_index(const int64_t idx, const int64_t ne) {
    if (idx < 0) {
        return idx + ne;
    }
    if (idx >= ne) {
        return idx - ne;
    }
    return idx;
}

static __global__ void roll_f32_cuda(const float * __restrict__ src,
                                     float * __restrict__ dst,
                                     const int64_t ne00,
                                     const int64_t ne01,
                                     const int64_t ne02,
                                     const int64_t ne03,
                                     const int     s0,
                                     const int     s1,
                                     const int     s2,
                                     const int     s3) {
    const int64_t idx        = int64_t(blockDim.x) * blockIdx.x + threadIdx.x;
    const int64_t n_elements = ne00 * ne01 * ne02 * ne03;

    if (idx >= n_elements) {
        return;
    }

    const int64_t i0 = idx % ne00;
    const int64_t i1 = (idx / ne00) % ne01;
    const int64_t i2 = (idx / (ne00 * ne01)) % ne02;
    const int64_t i3 = (idx / (ne00 * ne01 * ne02)) % ne03;

    const int64_t d0 = wrap_index(i0 - s0, ne00);
    const int64_t d1 = wrap_index(i1 - s1, ne01);
    const int64_t d2 = wrap_index(i2 - s2, ne02);
    const int64_t d3 = wrap_index(i3 - s3, ne03);

    dst[i3 * (ne00 * ne01 * ne02) + i2 * (ne01 * ne00) + i1 * ne00 + i0] =
        src[d3 * (ne00 * ne01 * ne02) + d2 * (ne01 * ne00) + d1 * ne00 + d0];
}

void ggml_cuda_op_roll(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    int s0 = dst->op_params[0];
    int s1 = dst->op_params[1];
    int s2 = dst->op_params[2];
    int s3 = dst->op_params[3];

    const ggml_tensor * src0   = dst->src[0];
    const float *       src0_d = (const float *) dst->src[0]->data;
    float *             dst_d  = (float *) dst->data;

    GGML_TENSOR_UNARY_OP_LOCALS;

    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_are_same_shape(dst->src[0], dst));

    cudaStream_t stream = ctx.stream();

    int64_t sz         = (ne00 * ne01 * ne02 * ne03);
    int64_t num_blocks = (sz + CUDA_ROLL_BLOCK_SIZE - 1) / CUDA_ROLL_BLOCK_SIZE;

    roll_f32_cuda<<<num_blocks, CUDA_ROLL_BLOCK_SIZE, 0, stream>>>(
        src0_d, dst_d, ne00, ne01, ne02, ne03, s0, s1, s2, s3);
}
