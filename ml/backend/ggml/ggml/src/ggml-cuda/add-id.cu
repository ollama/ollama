#include "add-id.cuh"

static __global__ void add_id_kernel(
        const float * src0, const float * src1, const int32_t * src2, float * dst,
        int64_t ne0, int64_t ne1,
        size_t nb01, size_t nb02,
        size_t nb11,
        size_t nb21
    ) {

    const int64_t i1 = blockIdx.x;
    const int64_t i2 = blockIdx.y;

    const int i11 = *(const int32_t *) ((const char *) src2 + i1*sizeof(int32_t) + i2*nb21);

    const size_t nb1 = ne0 * sizeof(float);
    const size_t nb2 = ne1 * nb1;

    float * dst_row = (float *)((char *)dst + i1*nb1 + i2*nb2);
    const float * src0_row = (const float *)((const char *)src0 +  i1*nb01 + i2*nb02);
    const float * src1_row = (const float *)((const char *)src1 + i11*nb11);

    for (int64_t i0 = threadIdx.x; i0 < ne0; i0 += blockDim.x) {
        dst_row[i0] = src0_row[i0] + src1_row[i0];
    }
}

void ggml_cuda_op_add_id(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * src2 = dst->src[2];

    GGML_TENSOR_TERNARY_OP_LOCALS

    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(src2->type == GGML_TYPE_I32);

    GGML_ASSERT(nb00 == sizeof(float));
    GGML_ASSERT(nb10 == sizeof(float));
    GGML_ASSERT(nb20 == sizeof(int32_t));

    const float * src0_d = (const float *)src0->data;
    const float * src1_d = (const float *)src1->data;
    const int32_t * src2_d = (const int32_t *)src2->data;
    float * dst_d = (float *)dst->data;

    int threads = std::min((int)ne00, 768); // cols
    dim3 blocks(ne01, ne02); // n_experts_used, n_tokens
    add_id_kernel<<<blocks, threads, 0, ctx.stream()>>>(
        src0_d, src1_d, src2_d, dst_d,
        ne0, ne1,
        nb01, nb02,
        nb11,
        nb21
    );
}
