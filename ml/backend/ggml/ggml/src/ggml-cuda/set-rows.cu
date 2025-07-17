#include "set-rows.cuh"

typedef void (*set_rows_kernel_t)(const char * src, char * dst);

template<typename src_t, typename dst_t>
__device__ void set_rows_1(const src_t * src_f, dst_t * dst_f) {
    GGML_UNUSED(src_f);
    GGML_UNUSED(dst_f);
}

template<>
__device__ __forceinline__ void set_rows_1<float, half>(const float * src_f, half * dst_h) {
    *dst_h = __float2half(*src_f);
}

template<>
__device__ __forceinline__ void set_rows_1<float, nv_bfloat16>(const float * src_f, nv_bfloat16 * dst_b) {
    *dst_b = *src_f;
}

template<>
__device__ __forceinline__ void set_rows_1<float, float>(const float * src_f, float * dst_f) {
    *dst_f = *src_f;
}

template<typename src_t, typename dst_t>
static __global__ void k_set_rows(
        const src_t * __restrict__ src0, const int64_t * __restrict__ src1, dst_t * __restrict__ dst,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t ne13,
        const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t s10, const int64_t s11, const int64_t s12,
        const int64_t s1, const int64_t s2, const int64_t s3) {

    const int64_t i = int64_t(blockDim.x) * blockIdx.x + threadIdx.x;
    const int64_t ne_total = ne00 * ne01 * ne02 * ne03;

    if (i >= ne_total) {
        return;
    }

    const int64_t i03 = i / (ne00 * ne01 * ne02);
    const int64_t i02 = (i - i03 * ne00 * ne01 * ne02) / (ne00 * ne01);
    const int64_t i01 = (i - i03 * ne00 * ne01 * ne02 - i02 * ne00 * ne01) / ne00;
    const int64_t i00 = i - i03 * ne00 * ne01 * ne02 - i02 * ne00 * ne01 - i01 * ne00;

    const int64_t i12 = i03 % ne12;
    const int64_t i11 = i02 % ne11;
    const int64_t i10 = i01;

    const int64_t dst_row = *(src1 + i10*s10 + i11*s11 + i12*s12);

    const src_t * src0_row = src0 + i01*s01 + i02*s02 + i03*s03;
    dst_t * dst_row_ptr    = dst + dst_row*s1 + i02*s2 + i03*s3;

    const src_t* src_elem = src0_row + i00;
    dst_t* dst_elem = dst_row_ptr + i00;
    set_rows_1(src_elem, dst_elem);

    GGML_UNUSED(ne10);
    GGML_UNUSED(ne13);
}

template<typename src_t, typename dst_t>
static void set_rows_cuda(
        const src_t * src0_d, const int64_t * src1_d, dst_t * dst_d,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t ne13,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        cudaStream_t stream) {

    const int64_t ne_total = ne00 * ne01 * ne02 * ne03;
    const int num_blocks = (ne_total + CUDA_SET_ROWS_BLOCK_SIZE - 1) / CUDA_SET_ROWS_BLOCK_SIZE;
    const dim3 block_size(CUDA_SET_ROWS_BLOCK_SIZE);
    const dim3 grid_size(num_blocks);


    const int64_t s01 = nb01/sizeof(src_t);
    const int64_t s02 = nb02/sizeof(src_t);
    const int64_t s03 = nb03/sizeof(src_t);
    const int64_t s10 = nb10/sizeof(int64_t);
    const int64_t s11 = nb11/sizeof(int64_t);
    const int64_t s12 = nb12/sizeof(int64_t);
    const int64_t s1  = nb1/sizeof(dst_t);
    const int64_t s2  = nb2/sizeof(dst_t);
    const int64_t s3  = nb3/sizeof(dst_t);

    if (ne_total > 0) {
        k_set_rows<<<grid_size, block_size, 0, stream>>>(
            src0_d, src1_d, dst_d,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            s01, s02, s03,
            s10, s11, s12,
            s1, s2, s3);
    }
}


void ggml_cuda_op_set_rows(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_I64);

    GGML_TENSOR_BINARY_OP_LOCALS

    const float * src0_d   = (const float *)src0->data;
    const int64_t * src1_d = (const int64_t *)src1->data;

    cudaStream_t stream = ctx.stream();



    if (dst->type == GGML_TYPE_F32) {
        set_rows_cuda(
            src0_d, src1_d, (float*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_F16) {
        set_rows_cuda(
            src0_d, src1_d, (half*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_BF16) {
        set_rows_cuda(
            src0_d, src1_d, (nv_bfloat16*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else {
        GGML_ABORT("unsupported type");
    }
}
