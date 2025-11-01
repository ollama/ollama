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

static __global__ void upscale_f32_bilinear(const float * x, float * dst,
        const int nb00, const int nb01, const int nb02, const int nb03,
        const int ne00_src, const int ne01_src,
        const int ne10_dst, const int ne11_dst, const int ne12_dst, const int ne13_dst,
        const float sf0, const float sf1, const float sf2, const float sf3,
        const float pixel_offset) {
    const int64_t index              = threadIdx.x + blockIdx.x * blockDim.x;
    const int64_t dst_total_elements = ne10_dst * ne11_dst * ne12_dst * ne13_dst;

    if (index >= dst_total_elements) {
        return;
    }

    const int i10_dst = index % ne10_dst;
    const int i11_dst = (index / ne10_dst) % ne11_dst;
    const int i12_dst = (index / (ne10_dst * ne11_dst)) % ne12_dst;
    const int i13_dst = index / (ne10_dst * ne11_dst * ne12_dst);

    const int i02_src = (int)(i12_dst / sf2);
    const int i03_src = (int)(i13_dst / sf3);

    const float y_src_f = ((float)i11_dst + pixel_offset) / sf1 - pixel_offset;
    int y0_src    = (int)floorf(y_src_f);
    int y1_src    = y0_src + 1;

    y0_src = max(0, min(y0_src, ne01_src - 1));
    y1_src = max(0, min(y1_src, ne01_src - 1));

    float dy = y_src_f - (float)y0_src;
    dy       = max(0.0f, min(dy, 1.0f));

    float x_src_f = ((float)i10_dst + pixel_offset) / sf0 - pixel_offset;
    int x0_src    = (int)floorf(x_src_f);
    int x1_src    = x0_src + 1;

    x0_src = max(0, min(x0_src, ne00_src - 1));
    x1_src = max(0, min(x1_src, ne00_src - 1));

    float dx = x_src_f - (float)x0_src;
    dx = max(0.0f, min(dx, 1.0f));

    const float * p_a = (const float *)((const char *)x + (int64_t)x0_src * nb00 + (int64_t)y0_src * nb01 + (int64_t)i02_src * nb02 + (int64_t)i03_src * nb03);
    const float * p_b = (const float *)((const char *)x + (int64_t)x1_src * nb00 + (int64_t)y0_src * nb01 + (int64_t)i02_src * nb02 + (int64_t)i03_src * nb03);
    const float * p_c = (const float *)((const char *)x + (int64_t)x0_src * nb00 + (int64_t)y1_src * nb01 + (int64_t)i02_src * nb02 + (int64_t)i03_src * nb03);
    const float * p_d = (const float *)((const char *)x + (int64_t)x1_src * nb00 + (int64_t)y1_src * nb01 + (int64_t)i02_src * nb02 + (int64_t)i03_src * nb03);

    const float val_a = *p_a;
    const float val_b = *p_b;
    const float val_c = *p_c;
    const float val_d = *p_d;

    float result = val_a * (1.0f - dx) * (1.0f - dy) +
                   val_b * dx * (1.0f - dy) +
                   val_c * (1.0f - dx) * dy +
                   val_d * dx * dy;

    dst[index] = result;
}

static void upscale_f32_cuda(const float * x, float * dst,
        const int nb00, const int nb01, const int nb02, const int nb03,
        const int ne10, const int ne11, const int ne12, const int ne13,
        const float sf0, const float sf1, const float sf2, const float sf3,
        cudaStream_t stream) {
    const int64_t dst_size   = ne10 * ne11 * ne12 * ne13;
    const int64_t num_blocks = (dst_size + CUDA_UPSCALE_BLOCK_SIZE - 1) / CUDA_UPSCALE_BLOCK_SIZE;

    upscale_f32<<<num_blocks, CUDA_UPSCALE_BLOCK_SIZE,0,stream>>>(x, dst, nb00, nb01, nb02, nb03, ne10, ne11, ne12, ne13, sf0, sf1, sf2, sf3);
}

static void upscale_f32_bilinear_cuda(const float * x, float * dst,
        const int nb00, const int nb01, const int nb02, const int nb03,
        const int ne00_src, const int ne01_src,
        const int ne10_dst, const int ne11_dst, const int ne12_dst, const int ne13_dst,
        const float sf0, const float sf1, const float sf2, const float sf3,
        const float pixel_offset, cudaStream_t stream) {
    const int64_t dst_size   = ne10_dst * ne11_dst * ne12_dst * ne13_dst;
    const int64_t num_blocks = (dst_size + CUDA_UPSCALE_BLOCK_SIZE - 1) / CUDA_UPSCALE_BLOCK_SIZE;

    upscale_f32_bilinear<<<num_blocks, CUDA_UPSCALE_BLOCK_SIZE,0,stream>>>(x, dst, nb00, nb01, nb02, nb03, ne00_src, ne01_src, ne10_dst, ne11_dst, ne12_dst, ne13_dst, sf0, sf1, sf2, sf3, pixel_offset);
}

void ggml_cuda_op_upscale(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    const int mode_flags = dst->op_params[0];
    const ggml_scale_mode mode = (ggml_scale_mode)(mode_flags & 0xFF);

    float sf0 = (float)dst->ne[0]/src0->ne[0];
    float sf1 = (float)dst->ne[1]/src0->ne[1];
    float sf2 = (float)dst->ne[2]/src0->ne[2];
    const float sf3 = (float)dst->ne[3]/src0->ne[3];

    if (mode == GGML_SCALE_MODE_NEAREST) {
        upscale_f32_cuda(src0_d, dst_d, src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3], dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], sf0, sf1, sf2, sf3, stream);
    } else if (mode == GGML_SCALE_MODE_BILINEAR) {
        float pixel_offset = 0.5f;
        if (mode_flags & GGML_SCALE_FLAG_ALIGN_CORNERS) {
            sf0          = dst->ne[0] > 1 && src0->ne[0] > 1 ? (float)(dst->ne[0] - 1) / (src0->ne[0] - 1) : sf0;
            sf1          = dst->ne[1] > 1 && src0->ne[1] > 1 ? (float)(dst->ne[1] - 1) / (src0->ne[1] - 1) : sf1;
            pixel_offset = 0.0f;
        }
        upscale_f32_bilinear_cuda(src0_d, dst_d, src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
                                 src0->ne[0], src0->ne[1], dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
                                 sf0, sf1, sf2, sf3, pixel_offset, stream);
    }
}
