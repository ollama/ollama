// GGML_OP_TQ_WHT — applies the symmetric Walsh-Hadamard transform
// F(x) = S·H·S·x/√headDim to every [headDim] vector in the tensor.
// F is self-inverse, so this op serves as both apply and undo.
//
// src[0]: input tensor (any shape; headDim = ne[0], must be power-of-2)
//         type: GGML_TYPE_F32 or GGML_TYPE_F16
// src[1]: signs tensor [headDim] f32 ±1
// dst:    same shape/layout as src[0], written in-place via src→dst copy + WHT

#include "tq-wht.cuh"
#include "common.cuh"

static __global__ void tq_wht_kernel(
    const float * __restrict__ src,
    const float * __restrict__ signs,
    float       * __restrict__ dst,
    int headDim,
    int64_t stride1,   // nb[1] / sizeof(float)
    int64_t stride2,   // nb[2] / sizeof(float)
    int64_t stride3,   // nb[3] / sizeof(float)
    int ne1, int ne2, int ne3)
{
    // One block per [headDim] vector.  blockIdx.x encodes the flat index over
    // (ne1, ne2, ne3).  Strides are in elements (bytes / sizeof(float)).
    int flat = blockIdx.x;
    int i1 = flat % ne1;
    int i2 = (flat / ne1) % ne2;
    int i3 = (flat / ne1 / ne2) % ne3;

    // src may be non-contiguous (e.g. after Permute); use its strides for reading.
    int64_t src_offset = (int64_t)i1 * stride1 + (int64_t)i2 * stride2 + (int64_t)i3 * stride3;
    // dst is always contiguous (ggml_dup_tensor allocates fresh memory).
    int64_t dst_offset = (int64_t)flat * headDim;

    extern __shared__ float s_x[];

    // Load [headDim] vector into shared memory.
    for (int i = threadIdx.x; i < headDim; i += blockDim.x) {
        s_x[i] = src[src_offset + i];
    }
    __syncthreads();

    // Apply F(x) = S·H·S·x/√headDim in-place.
    // kTrailingSync=false: write-back reads only thread-local s_x positions.
    apply_shs_wht<false>(s_x, signs, headDim, threadIdx.x, blockDim.x);

    // Write back to contiguous dst.
    for (int i = threadIdx.x; i < headDim; i += blockDim.x) {
        dst[dst_offset + i] = s_x[i];
    }
}

// f16 variant: reads half precision, computes in f32 shared memory, writes half.
static __global__ void tq_wht_kernel_f16(
    const half  * __restrict__ src,
    const float * __restrict__ signs,
    half        * __restrict__ dst,
    int headDim,
    int64_t stride1,   // nb[1] / sizeof(half)
    int64_t stride2,   // nb[2] / sizeof(half)
    int64_t stride3,   // nb[3] / sizeof(half)
    int ne1, int ne2, int ne3)
{
    int flat = blockIdx.x;
    int i1 = flat % ne1;
    int i2 = (flat / ne1) % ne2;
    int i3 = (flat / ne1 / ne2) % ne3;

    int64_t src_offset = (int64_t)i1 * stride1 + (int64_t)i2 * stride2 + (int64_t)i3 * stride3;
    int64_t dst_offset = (int64_t)flat * headDim;

    extern __shared__ float s_x[];

    for (int i = threadIdx.x; i < headDim; i += blockDim.x) {
        s_x[i] = __half2float(src[src_offset + i]);
    }
    __syncthreads();

    apply_shs_wht<false>(s_x, signs, headDim, threadIdx.x, blockDim.x);

    for (int i = threadIdx.x; i < headDim; i += blockDim.x) {
        dst[dst_offset + i] = __float2half(s_x[i]);
    }
}

void ggml_cuda_tq_wht(ggml_backend_cuda_context & ctx, struct ggml_tensor * dst) {
    const struct ggml_tensor * src   = dst->src[0];
    const struct ggml_tensor * signs = dst->src[1];

    const int headDim = (int)src->ne[0];
    const int ne1     = (int)src->ne[1];
    const int ne2     = (int)src->ne[2];
    const int ne3     = (int)src->ne[3];

    // dst must be contiguous (GGML scheduler guarantees this).
    GGML_ASSERT(ggml_is_contiguous(dst));

    int block_size = headDim;
    if (block_size > 128) block_size = 128;

    int n_vecs = ne1 * ne2 * ne3;
    dim3 grid(n_vecs);
    size_t smem = (size_t)headDim * sizeof(float);

    cudaStream_t stream = ctx.stream();

    if (src->type == GGML_TYPE_F16) {
        GGML_ASSERT(src->nb[0] == sizeof(ggml_fp16_t));
        const int64_t stride1 = (int64_t)(src->nb[1] / sizeof(ggml_fp16_t));
        const int64_t stride2 = (int64_t)(src->nb[2] / sizeof(ggml_fp16_t));
        const int64_t stride3 = (int64_t)(src->nb[3] / sizeof(ggml_fp16_t));
        tq_wht_kernel_f16<<<grid, block_size, smem, stream>>>(
            (const half *)src->data,
            (const float *)signs->data,
            (half *)dst->data,
            headDim, stride1, stride2, stride3, ne1, ne2, ne3);
    } else {
        // F32 path (used for Q rotation and attnOut undo).
        GGML_ASSERT(src->nb[0] == sizeof(float));
        const int64_t stride1 = (int64_t)(src->nb[1] / sizeof(float));
        const int64_t stride2 = (int64_t)(src->nb[2] / sizeof(float));
        const int64_t stride3 = (int64_t)(src->nb[3] / sizeof(float));
        tq_wht_kernel<<<grid, block_size, smem, stream>>>(
            (const float *)src->data,
            (const float *)signs->data,
            (float *)dst->data,
            headDim, stride1, stride2, stride3, ne1, ne2, ne3);
    }
}
