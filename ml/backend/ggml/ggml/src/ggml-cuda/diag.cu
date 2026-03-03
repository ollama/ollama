#include "convert.cuh"
#include "diag.cuh"
#include "ggml.h"

template <typename T>
static __global__ void diag_kernel(T * __restrict__ dst,
                                   const T * __restrict__ src,
                                   const int64_t ne0,
                                   const int64_t ne1,
                                   const int64_t ne2,
                                   const int64_t ne3,
                                   const int64_t total_elements) {
    const int64_t global_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (global_idx >= total_elements) {
        return;
    }

    const int64_t i0 = global_idx % ne0;
    const int64_t i1 = (global_idx / ne0) % ne1;
    const int64_t i2 = (global_idx / (ne0 * ne1)) % ne2;
    const int64_t i3 = global_idx / (ne0 * ne1 * ne2);

    const int64_t dst_idx = ((i3 * ne2 + i2) * ne1 + i1) * ne0 + i0;

    if (i0 == i1) {
        const int64_t batch_idx = i3 * ne2 + i2;
        const int64_t src_idx   = batch_idx * ne0 + i0;
        dst[dst_idx]            = src[src_idx];
    } else {
        dst[dst_idx] = ggml_cuda_cast<T>(0);
    }
    GGML_UNUSED_VARS(ne3);
}

void ggml_cuda_op_diag(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];

    void *       dst_d  = dst->data;
    const void * src0_d = src0->data;

    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_is_contiguous(src0));

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne02 = src0->ne[2];
    const int64_t ne03 = src0->ne[3];

    const int64_t ne0 = dst->ne[0];
    const int64_t ne1 = dst->ne[1];
    const int64_t ne2 = dst->ne[2];
    const int64_t ne3 = dst->ne[3];

    GGML_ASSERT(ne00 == ne0);
    GGML_ASSERT(ne01 == 1);
    GGML_ASSERT(ne02 == ne2);
    GGML_ASSERT(ne03 == ne3);

    const int64_t n_elems    = ggml_nelements(dst);
    const int64_t num_blocks = (n_elems + CUDA_DIAG_BLOCK_SIZE - 1) / CUDA_DIAG_BLOCK_SIZE;

    switch (dst->type) {
        case GGML_TYPE_F32:
            diag_kernel<<<num_blocks, CUDA_DIAG_BLOCK_SIZE, 0, stream>>>((float *) dst_d, (const float *) src0_d, ne0,
                                                                         ne1, ne2, ne3, n_elems);
            break;
        case GGML_TYPE_F16:
            diag_kernel<<<num_blocks, CUDA_DIAG_BLOCK_SIZE, 0, stream>>>((half *) dst_d, (const half *) src0_d, ne0,
                                                                         ne1, ne2, ne3, n_elems);
            break;
        default:
            GGML_ABORT("unsupported type");
    }
}
