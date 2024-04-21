#include "clamp.cuh"

static __global__ void clamp_f32(const float * x, float * dst, const float min, const float max, const int k) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }

    dst[i] = x[i] < min ? min : (x[i] > max ? max : x[i]);
}

static void clamp_f32_cuda(const float * x, float * dst, const float min, const float max, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_CLAMP_BLOCK_SIZE - 1) / CUDA_CLAMP_BLOCK_SIZE;
    clamp_f32<<<num_blocks, CUDA_CLAMP_BLOCK_SIZE, 0, stream>>>(x, dst, min, max, k);
}


void ggml_cuda_op_clamp(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    float min;
    float max;
    memcpy(&min, dst->op_params, sizeof(float));
    memcpy(&max, (float *) dst->op_params + 1, sizeof(float));

    clamp_f32_cuda(src0_d, dst_d, min, max, ggml_nelements(src0), stream);
}
