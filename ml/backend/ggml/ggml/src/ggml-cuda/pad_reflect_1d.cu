#include "pad_reflect_1d.cuh"

static __global__ __launch_bounds__(CUDA_PAD_REFLECT_1D_BLOCK_SIZE, 1) void
    pad_reflect_1d_kernel_f32(
        const void * __restrict__ src0,
        void * __restrict__       dst,
        const int64_t             ne0,
        const int64_t             ne00,
        const uint3               ne01,
        const int64_t             ne02,
        const int64_t             ne03,
        const int64_t             nb00,
        const int64_t             nb01,
        const int64_t             nb02,
        const int64_t             nb03,
        const int64_t             nb0,
        const int64_t             nb1,
        const int64_t             nb2,
        const int64_t             nb3,
        const int                 p0,
        const int                 p1) {
    const int64_t i3 = blockIdx.z;
    const int64_t i2 = blockIdx.y;

    const uint2   div_mod_packed = fast_div_modulo(blockIdx.x, ne01);
    const int64_t tile1          = div_mod_packed.y;  // i1
    const int64_t tile0          = div_mod_packed.x;  // nth i0 tile
    const int64_t i1             = tile1;
    const int64_t i0             = threadIdx.x + tile0 * blockDim.x;

    // ne01.z is original value of unpacked ne01 (see init_fastdiv_values in common.cuh)
    if (i0 >= ne0 || i1 >= ne01.z || i2 >= ne02 || i3 >= ne03) {
        return;
    }

    const char * src0_ptr = (const char *) src0 + i3 * nb03 + i2 * nb02 + i1 * nb01;
    char *       dst_ptr  = (char *) dst + i3 * nb3 + i2 * nb2 + i1 * nb1;

    const int64_t rel_i0 = i0 - p0;  // relative i0 in src0
    int64_t src_idx;

    if (rel_i0 < 0) {
        // Left padding - reflect
        src_idx = -rel_i0;
    } else if (rel_i0 < ne00) {
        // Middle - copy
        src_idx = rel_i0;
    } else {
        // Right padding - reflect
        src_idx = 2 * ne00 - 2 - rel_i0;
    }
    const float value               = *(const float *) (src0_ptr + src_idx * nb00);
    *(float *) (dst_ptr + i0 * nb0) = value;
}

void ggml_cuda_op_pad_reflect_1d(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0   = dst->src[0];
    cudaStream_t        stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    const int32_t * opts = (const int32_t *) dst->op_params;
    const int       p0   = opts[0];
    const int       p1   = opts[1];

    const int64_t ne00        = src0->ne[0];
    const int64_t ne01        = src0->ne[1];
    const uint3   ne01_packed = init_fastdiv_values(ne01);
    const int64_t ne02        = src0->ne[2];
    const int64_t ne03        = src0->ne[3];

    const int64_t ne0 = dst->ne[0];

    // sanity: padded length matches
    GGML_ASSERT(ne0 == ne00 + p0 + p1);

    constexpr int64_t bx     = CUDA_PAD_REFLECT_1D_BLOCK_SIZE;  // threads per block (x)
    const int64_t     tiles0 = (ne0 + bx - 1) / bx;             // number of tiles along i0
    // grid.x covers i1 and all tiles of i0: [ne01 * tiles0]
    // grid.y covers i2: [ne02]
    // grid.z covers i3: [ne03]
    const dim3        grid_dims((unsigned) (ne01 * tiles0), (unsigned) ne02, (unsigned) ne03);
    const dim3        block_dims((unsigned) bx, 1, 1);

    pad_reflect_1d_kernel_f32<<<grid_dims, block_dims, 0, stream>>>(
        src0->data, dst->data, ne0, ne00, ne01_packed, ne02, ne03, src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
        dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3], p0, p1);
}
