#include "upscale.cuh"

static __global__ void upscale_f32(const float * x, float * dst,
        const int nb00, const int nb01, const int nb02, const int nb03,
        const int ne10, const int ne11, const int ne12, const int ne13,
        const float sf0, const float sf1, const float sf2, const float sf3) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= ne10 * ne11 * ne12 * ne13) {
        return;
    }

    int i10 = index % ne10;
    int i11 = (index / ne10) % ne11;
    int i12 = (index / (ne10 * ne11)) % ne12;
    int i13 = (index / (ne10 * ne11 * ne12)) % ne13;

    int i00 = i10 / sf0;
    int i01 = i11 / sf1;
    int i02 = i12 / sf2;
    int i03 = i13 / sf3;

    dst[index] = *( (const float *)((const char *)x + i03 * nb03 + i02 * nb02 + i01 * nb01 + i00 * nb00) );
}

static void upscale_f32_cuda(const float * x, float * dst,
        const int nb00, const int nb01, const int nb02, const int nb03,
        const int ne10, const int ne11, const int ne12, const int ne13,
        const float sf0, const float sf1, const float sf2, const float sf3,
        cudaStream_t stream) {
    int dst_size = ne10 * ne11 * ne12 * ne13;
    int num_blocks = (dst_size + CUDA_UPSCALE_BLOCK_SIZE - 1) / CUDA_UPSCALE_BLOCK_SIZE;

    upscale_f32<<<num_blocks, CUDA_UPSCALE_BLOCK_SIZE,0,stream>>>(x, dst, nb00, nb01, nb02, nb03, ne10, ne11, ne12, ne13, sf0, sf1, sf2, sf3);
}

void ggml_cuda_op_upscale(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    const float sf0 = (float)dst->ne[0]/src0->ne[0];
    const float sf1 = (float)dst->ne[1]/src0->ne[1];
    const float sf2 = (float)dst->ne[2]/src0->ne[2];
    const float sf3 = (float)dst->ne[3]/src0->ne[3];

    upscale_f32_cuda(src0_d, dst_d, src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3], dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], sf0, sf1, sf2, sf3, stream);
}
