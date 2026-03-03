#include "pad.cuh"

#include <stdint.h>

__device__ __forceinline__ int64_t wrap_around(int64_t coord, int64_t size) {
    // + size ensures negatives are handled properly
    return (coord + size) % size;
}

static __global__ void pad_f32(const float * src, float * dst,
                               const int lp0, const int rp0, const int lp1, const int rp1,
                               const int lp2, const int rp2, const int lp3, const int rp3,
                               const int ne0, const int ne1, const int ne2, const int ne3,
                               const bool circular) {
    // blockIdx.z: i3*ne2+i2
    // blockIdx.y: i1
    // blockIDx.x: i0 / CUDA_PAD_BLOCK_SIZE
    // gridDim.y:  ne1
    int i0 = threadIdx.x + blockIdx.x * blockDim.x;
    int i1 = blockIdx.y;
    int i2 = blockIdx.z % ne2;
    int i3 = blockIdx.z / ne2;

    if (i0 >= ne0 || i1 >= ne1 || i2 >= ne2 || i3 >= ne3) {
        return;
    }

    const int64_t dst_idx = i3 * (ne0 * ne1 * ne2) + i2 * (ne0 * ne1) + i1 * ne0 + i0;

    if (!circular) {
        if ((i0 >= lp0 && i0 < ne0 - rp0) && (i1 >= lp1 && i1 < ne1 - rp1) && (i2 >= lp2 && i2 < ne2 - rp2) &&
            (i3 >= lp3 && i3 < ne3 - rp3)) {
            const int64_t i00  = i0 - lp0;
            const int64_t i01  = i1 - lp1;
            const int64_t i02  = i2 - lp2;
            const int64_t i03  = i3 - lp3;
            const int64_t ne02 = ne2 - lp2 - rp2;
            const int64_t ne01 = ne1 - lp1 - rp1;
            const int64_t ne00 = ne0 - lp0 - rp0;

            const int64_t src_idx = i03 * (ne00 * ne01 * ne02) + i02 * (ne00 * ne01) + i01 * ne00 + i00;

            dst[dst_idx] = src[src_idx];
        } else {
            dst[dst_idx] = 0.0f;
        }
    }
    // circular means on a torus, so x and y wrap around
    else {
        const int64_t ne00 = ne0 - lp0 - rp0;
        const int64_t ne01 = ne1 - lp1 - rp1;
        const int64_t ne02 = ne2 - lp2 - rp2;
        const int64_t ne03 = ne3 - lp3 - rp3;

        const int64_t i00 = wrap_around(i0 - lp0, ne00);
        const int64_t i01 = wrap_around(i1 - lp1, ne01);
        const int64_t i02 = wrap_around(i2 - lp2, ne02);
        const int64_t i03 = wrap_around(i3 - lp3, ne03);

        const int64_t src_idx = i03 * (ne00 * ne01 * ne02) + i02 * (ne00 * ne01) + i01 * ne00 + i00;

        dst[dst_idx] = src[src_idx];
    }
}


static void pad_f32_cuda(const float * src, float * dst,
    const int lp0, const int rp0, const int lp1, const int rp1,
    const int lp2, const int rp2, const int lp3, const int rp3,
    const int ne0, const int ne1, const int ne2, const int ne3,
    const bool circular, cudaStream_t stream) {
    int  num_blocks = (ne0 + CUDA_PAD_BLOCK_SIZE - 1) / CUDA_PAD_BLOCK_SIZE;
    dim3 gridDim(num_blocks, ne1, ne2 * ne3);
    pad_f32<<<gridDim, CUDA_PAD_BLOCK_SIZE, 0, stream>>>(src, dst,
                                                         lp0, rp0, lp1, rp1, lp2, rp2, lp3, rp3,
                                                         ne0, ne1, ne2, ne3, circular);
}

void ggml_cuda_op_pad(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0   = dst->src[0];
    const float *       src0_d = (const float *) src0->data;
    float *             dst_d  = (float *) dst->data;
    cudaStream_t        stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src0));

    const int32_t lp0      = ((const int32_t *) (dst->op_params))[0];
    const int32_t rp0      = ((const int32_t *) (dst->op_params))[1];
    const int32_t lp1      = ((const int32_t *) (dst->op_params))[2];
    const int32_t rp1      = ((const int32_t *) (dst->op_params))[3];
    const int32_t lp2      = ((const int32_t *) (dst->op_params))[4];
    const int32_t rp2      = ((const int32_t *) (dst->op_params))[5];
    const int32_t lp3      = ((const int32_t *) (dst->op_params))[6];
    const int32_t rp3      = ((const int32_t *) (dst->op_params))[7];
    const int32_t circular = ((const int32_t *) (dst->op_params))[8];

    pad_f32_cuda(src0_d, dst_d,
                 lp0, rp0, lp1, rp1, lp2, rp2, lp3, rp3,
                 dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
                 (bool) circular, stream);
}
