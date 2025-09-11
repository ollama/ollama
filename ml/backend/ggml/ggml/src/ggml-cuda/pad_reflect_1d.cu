#include "pad_reflect_1d.cuh"

static __global__ void pad_reflect_1d_kernel_f32(
    const void * __restrict__ src0,
    void * __restrict__ dst,
    const int64_t ne0,
    const int64_t ne00,
    const int64_t ne01,
    const int64_t ne02,
    const int64_t ne03,
    const int64_t nb00,
    const int64_t nb01,
    const int64_t nb02,
    const int64_t nb03,
    const int64_t nb0,
    const int64_t nb1,
    const int64_t nb2,
    const int64_t nb3,
    const int p0,
    const int p1) {

    const int64_t i3 = blockIdx.z;
    const int64_t i2 = blockIdx.y;
    const int64_t i1 = blockIdx.x;

    if (i1 >= ne01 || i2 >= ne02 || i3 >= ne03) {
        return;
    }

    const char * src0_ptr = (const char *)src0 + i3*nb03 + i2*nb02 + i1*nb01;
    char * dst_ptr = (char *)dst + i3*nb3 + i2*nb2 + i1*nb1;

    for (int64_t i0 = threadIdx.x; i0 < ne0; i0 += blockDim.x) {
        float value;

        if (i0 < p0) {
            // Left padding - reflect
            value = *(const float *)(src0_ptr + (p0 - i0) * nb00);
        } else if (i0 < ne0 - p1) {
            // Middle - copy
            value = *(const float *)(src0_ptr + (i0 - p0) * nb00);
        } else {
            // Right padding - reflect
            int64_t src_idx = (ne0 - p1 - p0) - (p1 + 1 - (ne0 - i0)) - 1;
            value = *(const float *)(src0_ptr + src_idx * nb00);
        }

        *(float *)(dst_ptr + i0 * nb0) = value;
    }
}

void ggml_cuda_op_pad_reflect_1d(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    const int32_t * opts = (const int32_t *) dst->op_params;
    const int p0 = opts[0];
    const int p1 = opts[1];

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne02 = src0->ne[2];
    const int64_t ne03 = src0->ne[3];

    const int64_t ne0 = dst->ne[0];

    GGML_ASSERT(ne0 == ne00 + p0 + p1);

    const dim3 block_dims(CUDA_PAD_REFLECT_1D_BLOCK_SIZE, 1, 1);
    const dim3 grid_dims(ne01, ne02, ne03);

    pad_reflect_1d_kernel_f32<<<grid_dims, block_dims, 0, stream>>>(
        src0->data, dst->data,
        ne0, ne00, ne01, ne02, ne03,
        src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
        dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3],
        p0, p1
    );
}
