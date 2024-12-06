#include "out-prod.cuh"

#include <cstdint>

void ggml_cuda_out_prod(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_BINARY_OP_LOCALS

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(dst));

    GGML_ASSERT(ne01 == ne11);
    GGML_ASSERT(ne0 == ne00);
    GGML_ASSERT(ne1 == ne10);

    GGML_ASSERT(ne2 == src0->ne[2]);
    GGML_ASSERT(ne2 == src1->ne[2]);
    GGML_ASSERT(ne3 == src0->ne[3]);
    GGML_ASSERT(ne3 == src1->ne[3]);

    const float * src0_d = (const float *) src0->data;
    const float * src1_d = (const float *) src1->data;
    float       *  dst_d = (float       *)  dst->data;

    cudaStream_t   stream = ctx.stream();
    cublasHandle_t handle = ctx.cublas_handle();

    const float alpha = 1.0f;
    const float beta = 0.0f;

    GGML_ASSERT(ne2 == 1);
    GGML_ASSERT(ne3 == 1);
    CUBLAS_CHECK(cublasSetStream(handle, stream));

    const bool src1_T = ggml_is_transposed(src1);
    const cublasOperation_t src1_cublas_op =  src1_T ? CUBLAS_OP_N : CUBLAS_OP_T;
    const int64_t           ldb            = (src1_T ?        nb10 :        nb11) /  sizeof(float);
    GGML_ASSERT(                             (src1_T ?        nb11 :        nb10) == sizeof(float));

    CUBLAS_CHECK(
        cublasSgemm(handle, CUBLAS_OP_N, src1_cublas_op,
                ne0, ne1, ne01,
                &alpha, src0_d, ne00,
                        src1_d, ldb,
                &beta,  dst_d,  ne0));
}
