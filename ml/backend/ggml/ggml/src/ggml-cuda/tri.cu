#include "common.cuh"
#include "convert.cuh"
#include "tri.cuh"
#include "ggml.h"

template<typename T, bool prefix_keep, int add_to_split>
static __global__ void tri_kernel(
        const T * src, T * dst,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t nb00, const int64_t nb01, const int64_t nb02, const int64_t nb03,
        const int64_t nb0,  const int64_t nb1,  const int64_t nb2,  const int64_t nb3) {
    const int64_t i3 = blockIdx.z;
    const int64_t i2 = blockIdx.y;
    const int64_t i1 = blockIdx.x;
    const int64_t split_point = i1 + add_to_split;

    GGML_UNUSED_VARS(nb00, nb0);

    if (i3 >= ne03 || i2 >= ne02 || i1 >= ne01) {
        return;
    }

    const T * src_row = src + i1*nb01 + i2*nb02 + i3*nb03;
    T       * dst_row = dst + i1*nb1  + i2*nb2  + i3*nb3;

    if constexpr (prefix_keep) {
        for (int64_t i0 = threadIdx.x; i0 < split_point; i0 += blockDim.x) {
            dst_row[i0] = src_row[i0];
        }
        for (int64_t i0 = threadIdx.x + split_point; i0 < ne00; i0 += blockDim.x) {
            dst_row[i0] = ggml_cuda_cast<T, float>(0.0f);
        }
    } else {
        for (int64_t i0 = threadIdx.x; i0 < split_point; i0 += blockDim.x) {
            dst_row[i0] = ggml_cuda_cast<T, float>(0.0f);
        }
        for (int64_t i0 = threadIdx.x + split_point; i0 < ne00; i0 += blockDim.x) {
            dst_row[i0] = src_row[i0];
        }
    }
}

template<typename T>
static void tri_cuda(
        const T * src, T * dst,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t nb00, const int64_t nb01, const int64_t nb02, const int64_t nb03,
        const int64_t nb0,  const int64_t nb1,  const int64_t nb2,  const int64_t nb3,
        const ggml_tri_type ttype,
        cudaStream_t stream) {

    dim3 block_dims(CUDA_TRI_BLOCK_SIZE, 1, 1);
    dim3 grid_dims(ne01, ne02, ne03);
    const size_t type_size = sizeof(T);

    const int add_to_split = (ttype == GGML_TRI_TYPE_LOWER_DIAG || ttype == GGML_TRI_TYPE_UPPER) ? 1 : 0;
    const bool prefix_keep = (ttype == GGML_TRI_TYPE_LOWER || ttype == GGML_TRI_TYPE_LOWER_DIAG);

    if (prefix_keep) {
        if (add_to_split == 0) {
            tri_kernel<T, true, 0><<<grid_dims, block_dims, 0, stream>>>(
                src, dst,
                ne00, ne01, ne02, ne03,
                nb00 / type_size, nb01 / type_size, nb02 / type_size, nb03 / type_size,
                nb0 / type_size, nb1 / type_size, nb2 / type_size, nb3 / type_size
            );
        } else { // only 0 and 1 supported
            tri_kernel<T, true, 1><<<grid_dims, block_dims, 0, stream>>>(
                src, dst,
                ne00, ne01, ne02, ne03,
                nb00 / type_size, nb01 / type_size, nb02 / type_size, nb03 / type_size,
                nb0 / type_size, nb1 / type_size, nb2 / type_size, nb3 / type_size
            );
        }
    } else {
        if (add_to_split == 0) {
            tri_kernel<T, false, 0><<<grid_dims, block_dims, 0, stream>>>(
                src, dst,
                ne00, ne01, ne02, ne03,
                nb00 / type_size, nb01 / type_size, nb02 / type_size, nb03 / type_size,
                nb0 / type_size, nb1 / type_size, nb2 / type_size, nb3 / type_size
            );
        } else {
            tri_kernel<T, false, 1><<<grid_dims, block_dims, 0, stream>>>(
                src, dst,
                ne00, ne01, ne02, ne03,
                nb00 / type_size, nb01 / type_size, nb02 / type_size, nb03 / type_size,
                nb0 / type_size, nb1 / type_size, nb2 / type_size, nb3 / type_size
            );
        }
    }
}

void ggml_cuda_op_tri(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    cudaStream_t stream = ctx.stream();

    const ggml_tri_type ttype = static_cast<ggml_tri_type>(ggml_get_op_params_i32(dst, 0));

    GGML_ASSERT(src0->type == dst->type);

    switch(src0->type) {
        case GGML_TYPE_F32:
            {
                tri_cuda(
                    (const float *)src0->data, (float *)dst->data,
                    src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
                    src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
                    dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3],
                    ttype, stream
                );
            } break;
        case GGML_TYPE_F16:
            {
                tri_cuda(
                    (const half *)src0->data, (half *)dst->data,
                    src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
                    src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
                    dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3],
                    ttype, stream
                );
            } break;
        case GGML_TYPE_BF16:
            {
                tri_cuda(
                    (const nv_bfloat16 *)src0->data, (nv_bfloat16 *)dst->data,
                    src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
                    src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
                    dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3],
                    ttype, stream
                );
            } break;
        default:
            GGML_ABORT("fatal error");
    }
}
