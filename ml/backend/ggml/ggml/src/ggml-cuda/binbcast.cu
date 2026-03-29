#include "binbcast.cuh"
#include <cstdint>
#include <utility>

static __device__ __forceinline__ float op_repeat(const float a, const float b) {
    return b;
    GGML_UNUSED(a);
}

static __device__ __forceinline__ float op_add(const float a, const float b) {
    return a + b;
}

static __device__ __forceinline__ float op_sub(const float a, const float b) {
    return a - b;
}

static __device__ __forceinline__ float op_mul(const float a, const float b) {
    return a * b;
}

static __device__ __forceinline__ float op_div(const float a, const float b) {
    return a / b;
}

template <float (*bin_op)(const float, const float),
          typename src0_t,
          typename src1_t,
          typename dst_t,
          typename... src1_ptrs>
static __global__ void k_bin_bcast(const src0_t *         src0,
                                   const src1_t *         src1,
                                   dst_t *                dst,
                                   const int              ne0,
                                   const int              ne1,
                                   const int              ne2,
                                   const uint3            ne3,
                                   const uint3            ne10,
                                   const uint3            ne11,
                                   const uint3            ne12,
                                   const uint3            ne13,
                                 /*const int              s0,*/
                                   const int              s1,
                                   const int              s2,
                                   const int              s3,
                                   const int              s00,
                                   const int              s01,
                                   const int              s02,
                                   const int              s03,
                                   const int              s10,
                                   const int              s11,
                                   const int              s12,
                                   const int              s13,
                                   src1_ptrs... src1s) {
    const uint32_t i0s = blockDim.x * blockIdx.x + threadIdx.x;
    const uint32_t i1  = (blockDim.y * blockIdx.y + threadIdx.y);
    const uint32_t i2  = fastdiv((blockDim.z * blockIdx.z + threadIdx.z), ne3);
    const uint32_t i3  = (blockDim.z * blockIdx.z + threadIdx.z) - (i2 * ne3.z);

    if (i0s >= (uint32_t)ne0 || i1 >= (uint32_t)ne1 || i2 >= (uint32_t)ne2 || i3 >= ne3.z) {
        return;
    }

    const uint32_t i11 = fastmodulo(i1, ne11);
    const uint32_t i12 = fastmodulo(i2, ne12);
    const uint32_t i13 = fastmodulo(i3, ne13);

    const size_t i_src0 =  i3*s03 +  i2*s02 +  i1*s01;
    const size_t i_src1 = i13*s13 + i12*s12 + i11*s11;
    const size_t i_dst  =  i3*s3  +  i2*s2  +  i1*s1;

    const src0_t * src0_row = src0 ? (src0 + i_src0) : nullptr;
    dst_t * dst_row = dst + i_dst;

    for (int i0 = i0s; i0 < ne0; i0 += blockDim.x * gridDim.x) {
        const uint32_t i10 = fastmodulo(i0, ne10);

        float result = src0_row ? (float) src0_row[i0*s00] : 0.0f;
        if constexpr (sizeof...(src1_ptrs) > 0) {
            result = (..., (result = bin_op(result, (float)src1s[i_src1 + i10*s10])));
        } else {
            result = bin_op(result, (float)src1[i_src1 + i10*s10]);
        }

        dst_row[i0] = (dst_t) result;
    }
}

template <float (*bin_op)(const float, const float),
          typename src0_t,
          typename src1_t,
          typename dst_t,
          typename... src1_ptrs>
static __global__ void k_bin_bcast_unravel(const src0_t *         src0,
                                           const src1_t *         src1,
                                           dst_t *                dst,
                                           const uint3            ne0,
                                           const uint3            ne1,
                                           const uint3            ne2,
                                           const uint32_t         ne3,
                                           const uint3            prod_012,
                                           const uint3            prod_01,
                                           const uint3            ne10,
                                           const uint3            ne11,
                                           const uint3            ne12,
                                           const uint3            ne13,
                                         /*const int              s0,*/
                                           const int              s1,
                                           const int              s2,
                                           const int              s3,
                                           const int              s00,
                                           const int              s01,
                                           const int              s02,
                                           const int              s03,
                                           const int              s10,
                                           const int              s11,
                                           const int              s12,
                                           const int              s13,
                                           src1_ptrs... src1s) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    const uint32_t i3 = fastdiv(i, prod_012);
    const uint32_t i2 = fastdiv(i - i3 * prod_012.z, prod_01);
    const uint32_t i1 = fastdiv(i - i3 * prod_012.z - i2 * prod_01.z, ne0);
    const uint32_t i0 = i - i3 * prod_012.z - i2 * prod_01.z - i1 * ne0.z;

    if (i0 >= ne0.z || i1 >= ne1.z || i2 >= ne2.z || i3 >= ne3) {
        return;
    }

    const int i11 = fastmodulo(i1, ne11);
    const int i12 = fastmodulo(i2, ne12);
    const int i13 = fastmodulo(i3, ne13);

    const size_t i_src0 =  i3*s03 +  i2*s02 +  i1*s01;
    const size_t i_src1 = i13*s13 + i12*s12 + i11*s11;
    const size_t i_dst  =  i3*s3  +  i2*s2  +  i1*s1;

    const src0_t * src0_row = src0 ? (src0 + i_src0) : nullptr;
    dst_t * dst_row = dst + i_dst;

    const int i10 = fastmodulo(i0, ne10);

    float result = src0_row ? (float) src0_row[i0*s00] : 0.0f;
    if constexpr (sizeof...(src1_ptrs) > 0) {
        result = (..., (result = bin_op(result, (float)src1s[i_src1 + i10*s10])));
    } else {
        result = bin_op(result, (float)src1[i_src1 + i10*s10]);
    }

    dst_row[i0] = (dst_t) result;
}

template <float (*bin_op)(const float, const float), typename src0_t, typename src1_t, typename dst_t, size_t... I>
static void launch_bin_bcast_pack(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
                                  const src0_t * src0_dd, const src1_t * src1_dd, dst_t * dst_dd,
                                  cudaStream_t stream, std::index_sequence<I...>) {
    GGML_TENSOR_BINARY_OP_LOCALS

    int nr0 = ne10 / ne0;
    int nr1 = ne11 / ne1;
    int nr2 = ne12 / ne2;
    int nr3 = ne13 / ne3;

    int nr[4] = { nr0, nr1, nr2, nr3 };

    int64_t cne[]  = { ne0, ne1, ne2, ne3 };
    int64_t cne0[] = { ne00, ne01, ne02, ne03 };
    int64_t cne1[] = { ne10, ne11, ne12, ne13 };

    size_t cnb[]  = { nb0, nb1, nb2, nb3 };
    size_t cnb0[] = { nb00, nb01, nb02, nb03 };
    size_t cnb1[] = { nb10, nb11, nb12, nb13 };

    auto collapse = [](int64_t cne[]) {
        cne[0] *= cne[1];
        cne[1] = cne[2];
        cne[2] = cne[3];
        cne[3] = 1;
    };

    auto collapse_nb = [](size_t cnb[], const int64_t cne[]) {
        cnb[1] *= cne[1];
        cnb[2] *= cne[2];
        cnb[3] *= cne[3];
    };

    if (ggml_is_contiguous(src0) && ggml_is_contiguous(src1) && !ggml_is_permuted(src0) && !ggml_is_permuted(src1)) {
        for (int i = 0; i < 4; i++) {
            if (nr[i] != 1) {
                break;
            }
            if (i > 0) {
                collapse_nb(cnb, cne);
                collapse_nb(cnb0, cne0);
                collapse_nb(cnb1, cne1);
                collapse(cne);
                collapse(cne0);
                collapse(cne1);
            }
        }
    }

    {
        int64_t ne0 = cne[0];
        int64_t ne1 = cne[1];
        int64_t ne2 = cne[2];
        int64_t ne3 = cne[3];

        //int64_t ne00 = cne0[0]; GGML_UNUSED(ne00);
        //int64_t ne01 = cne0[1]; GGML_UNUSED(ne01);
        //int64_t ne02 = cne0[2]; GGML_UNUSED(ne02);
        //int64_t ne03 = cne0[3]; GGML_UNUSED(ne03);

        size_t nb0 = cnb[0];
        size_t nb1 = cnb[1];
        size_t nb2 = cnb[2];
        size_t nb3 = cnb[3];

        size_t nb00 = cnb0[0];
        size_t nb01 = cnb0[1];
        size_t nb02 = cnb0[2];
        size_t nb03 = cnb0[3];

        size_t nb10 = cnb1[0];
        size_t nb11 = cnb1[1];
        size_t nb12 = cnb1[2];
        size_t nb13 = cnb1[3];

      //size_t s0 = nb0 / sizeof(dst_t);
        size_t s1 = nb1 / sizeof(dst_t);
        size_t s2 = nb2 / sizeof(dst_t);
        size_t s3 = nb3 / sizeof(dst_t);

        size_t s10 = nb10 / sizeof(src1_t);
        size_t s11 = nb11 / sizeof(src1_t);
        size_t s12 = nb12 / sizeof(src1_t);
        size_t s13 = nb13 / sizeof(src1_t);

        size_t s00 = nb00 / sizeof(src0_t);
        size_t s01 = nb01 / sizeof(src0_t);
        size_t s02 = nb02 / sizeof(src0_t);
        size_t s03 = nb03 / sizeof(src0_t);

        GGML_ASSERT(nb0 % sizeof(dst_t) == 0);
        GGML_ASSERT(nb1 % sizeof(dst_t) == 0);
        GGML_ASSERT(nb2 % sizeof(dst_t) == 0);
        GGML_ASSERT(nb3 % sizeof(dst_t) == 0);

        GGML_ASSERT(nb00 % sizeof(src0_t) == 0);
        GGML_ASSERT(nb01 % sizeof(src0_t) == 0);
        GGML_ASSERT(nb02 % sizeof(src0_t) == 0);
        GGML_ASSERT(nb03 % sizeof(src0_t) == 0);

        GGML_ASSERT(nb10 % sizeof(src1_t) == 0);
        GGML_ASSERT(nb11 % sizeof(src1_t) == 0);
        GGML_ASSERT(nb12 % sizeof(src1_t) == 0);
        GGML_ASSERT(nb13 % sizeof(src1_t) == 0);

        const int block_size = 128;

        int64_t hne0 = std::max(ne0 / 2LL, 1LL);

        dim3 block_dims;
        block_dims.x = std::min<unsigned int>(hne0, block_size);
        block_dims.y = std::min<unsigned int>(ne1, block_size / block_dims.x);
        block_dims.z = std::min(std::min<unsigned int>(ne2 * ne3, block_size / block_dims.x / block_dims.y), 64U);

        dim3 block_nums((hne0 + block_dims.x - 1) / block_dims.x, (ne1 + block_dims.y - 1) / block_dims.y,
                        (ne2 * ne3 + block_dims.z - 1) / block_dims.z);

        const uint3 ne10 = init_fastdiv_values((uint32_t) cne1[0]);
        const uint3 ne11 = init_fastdiv_values((uint32_t) cne1[1]);
        const uint3 ne12 = init_fastdiv_values((uint32_t) cne1[2]);
        const uint3 ne13 = init_fastdiv_values((uint32_t) cne1[3]);

        if (block_nums.z > 65535 || block_nums.y > 65535) {
            int         block_num  = (ne0 * ne1 * ne2 * ne3 + block_size - 1) / block_size;
            const uint3 prod_012    = init_fastdiv_values((uint32_t) (ne0 * ne1 * ne2));
            const uint3 prod_01     = init_fastdiv_values((uint32_t) (ne0 * ne1));
            const uint3 ne0_fastdiv = init_fastdiv_values((uint32_t) ne0);
            const uint3 ne1_fastdiv = init_fastdiv_values((uint32_t) ne1);
            const uint3 ne2_fastdiv = init_fastdiv_values((uint32_t) ne2);

            if constexpr (sizeof...(I) > 0) {
                k_bin_bcast_unravel<bin_op, src0_t, src1_t, dst_t><<<block_num, block_size, 0, stream>>>(
                    src0_dd, src1_dd, dst_dd, ne0_fastdiv, ne1_fastdiv, ne2_fastdiv, ne3, prod_012, prod_01, ne10, ne11,
                    ne12, ne13,
                  /*s0,*/ s1,  s2,  s3,
                    s00, s01, s02, s03,
                    s10, s11, s12, s13, (const src1_t *) dst->src[I + 1]->data...);
            } else {
                k_bin_bcast_unravel<bin_op, src0_t, src1_t, dst_t>
                    <<<block_num, block_size, 0, stream>>>(src0_dd, src1_dd, dst_dd, ne0_fastdiv, ne1_fastdiv,
                                                           ne2_fastdiv, ne3, prod_012, prod_01, ne10, ne11, ne12, ne13,
                                                         /*s0,*/ s1,  s2,  s3,
                                                           s00, s01, s02, s03,
                                                           s10, s11, s12, s13);
            }
        } else {
            const uint3 ne3_fastdiv = init_fastdiv_values((uint32_t) ne3);
            if constexpr (sizeof...(I) > 0) {
                k_bin_bcast<bin_op, src0_t, src1_t, dst_t><<<block_nums, block_dims, 0, stream>>>(
                    src0_dd, src1_dd, dst_dd, ne0, ne1, ne2, ne3_fastdiv, ne10, ne11, ne12, ne13,
                  /*s0,*/ s1, s2,  s3,
                    s00 ,s01, s02, s03,
                    s10, s11, s12, s13, (const src1_t *) dst->src[I + 1]->data...);
            } else {
                k_bin_bcast<bin_op, src0_t, src1_t, dst_t><<<block_nums, block_dims, 0, stream>>>(
                    src0_dd, src1_dd, dst_dd, ne0, ne1, ne2, ne3_fastdiv, ne10, ne11, ne12, ne13,
                  /*s0,*/ s1,  s2,  s3,
                    s00, s01, s02, s03,
                    s10, s11, s12, s13);
            }
        }
    }
}

template <typename T>
static __global__ void k_repeat_back(
    const T * __restrict__ src, T * __restrict__ dst, const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
    const size_t s00, const size_t s01, const size_t s02, const size_t s03,
    const int64_t ne0, const int64_t ne1, const int64_t ne2, const int64_t ne3) {

    const int64_t tid0  = int64_t(blockIdx.x)*blockDim.x + threadIdx.x;
    const int64_t tid1  = int64_t(blockIdx.y)*blockDim.y + threadIdx.y;
    const int64_t tid23 = int64_t(blockIdx.z)*blockDim.z + threadIdx.z;
    const int64_t tid2  = tid23 % ne2;
    const int64_t tid3  = tid23 / ne2;

    if (tid0 >= ne0) {
        return;
    }

    T sum = 0;
    for (int64_t i3 = tid3; i3 < ne03; i3 += ne3) {
        for (int64_t i2 = tid2; i2 < ne02; i2 += ne2) {
            for (int64_t i1 = tid1; i1 < ne01; i1 += ne1) {
                for (int64_t i0 = tid0; i0 < ne00; i0 += ne0) {
                    sum += src[i3*s03 + i2*s02 + i1*s01 + i0*s00];
                }
            }
        }
    }
    dst[tid3*ne2*ne1*ne0 + tid2*ne1*ne0 + tid1*ne0 + tid0] = sum;
}

template <float (*bin_op)(const float, const float), int n_fuse = 1>
struct bin_bcast_cuda {
    template<typename src0_t, typename src1_t, typename dst_t>
    void operator()(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst,
            const src0_t * src0_dd, const src1_t * src1_dd, dst_t * dst_dd,
            cudaStream_t stream) {
        launch_bin_bcast_pack<bin_op, src0_t, src1_t, dst_t>(
            src0, src1, dst, src0_dd, src1_dd, dst_dd, stream, std::make_index_sequence<n_fuse>{});
    }
};

template <typename T>
static void repeat_back_cuda(
    const T * src, T * dst, const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
    const size_t s00, const size_t s01, const size_t s02, const size_t s03,
    const int64_t ne0, const int64_t ne1, const int64_t ne2, const int64_t ne3, cudaStream_t stream) {

    const dim3 block_dims(WARP_SIZE, 1, 1);
    const dim3 block_nums((ne0 + WARP_SIZE - 1) / WARP_SIZE, ne1, ne2*ne3);
    k_repeat_back<T><<<block_nums, block_dims, 0, stream>>>
        (src, dst, ne00, ne01, ne02, ne03, s00, s01, s02, s03, ne0, ne1, ne2, ne3);
}

template<class op>
static void ggml_cuda_op_bin_bcast(
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
    const void * src0_dd, const void * src1_dd, void * dst_dd, cudaStream_t stream) {

    GGML_ASSERT(src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16);

    if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
        op()(src0, src1, dst, (const float *)src0_dd, (const float *)src1_dd, (float *)dst_dd, stream);
    } else if (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F16) {
        op()(src0, src1, dst, (const half *) src0_dd, (const half *)src1_dd, (half *) dst_dd, stream);
    } else if (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16) {
        op()(src0, src1, dst, (const half *) src0_dd, (const float *)src1_dd, (half *) dst_dd, stream);
    } else if (src0->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F32) {
        op()(src0, src1, dst, (const half *) src0_dd, (const float *)src1_dd, (float *)dst_dd, stream);
    } else {
        fprintf(stderr, "%s: unsupported types: dst: %s, src0: %s, src1: %s\n", __func__,
            ggml_type_name(dst->type), ggml_type_name(src0->type), ggml_type_name(src1->type));
        GGML_ABORT("fatal error");
    }
}

void ggml_cuda_op_repeat(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_bin_bcast<bin_bcast_cuda<op_repeat, 0>>(dst, dst->src[0], dst, nullptr, dst->src[0]->data, dst->data, ctx.stream());
}

void ggml_cuda_op_add(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_bin_bcast<bin_bcast_cuda<op_add>>(dst->src[0], dst->src[1], dst, dst->src[0]->data, dst->src[1]->data, dst->data, ctx.stream());
}

void ggml_cuda_op_sub(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_bin_bcast<bin_bcast_cuda<op_sub>>(dst->src[0], dst->src[1], dst, dst->src[0]->data, dst->src[1]->data, dst->data, ctx.stream());
}

void ggml_cuda_op_mul(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_bin_bcast<bin_bcast_cuda<op_mul>>(dst->src[0], dst->src[1], dst, dst->src[0]->data, dst->src[1]->data, dst->data, ctx.stream());
}

void ggml_cuda_op_div(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_bin_bcast<bin_bcast_cuda<op_div>>(dst->src[0], dst->src[1], dst, dst->src[0]->data, dst->src[1]->data, dst->data, ctx.stream());
}

template <float (*op)(const float, const float), int n_fuse>
static void ggml_cuda_op_fused_binbcast_impl(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    cudaStream_t stream = ctx.stream();

    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
        launch_bin_bcast_pack<op, float, float, float>(src0, src1, dst,
            (const float *) src0->data, (const float *) src1->data, (float *) dst->data,
            stream, std::make_index_sequence<n_fuse>{});
    } else if (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F16) {
        launch_bin_bcast_pack<op, half, half, half>(src0, src1, dst,
            (const half *) src0->data, (const half *) src1->data, (half *) dst->data,
            stream, std::make_index_sequence<n_fuse>{});
    } else if (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16) {
        launch_bin_bcast_pack<op, half, float, half>(src0, src1, dst,
            (const half *) src0->data, (const float *) src1->data, (half *) dst->data,
            stream, std::make_index_sequence<n_fuse>{});
    } else if (src0->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F32) {
        launch_bin_bcast_pack<op, half, float, float>(src0, src1, dst,
            (const half *) src0->data, (const float *) src1->data, (float *) dst->data,
            stream, std::make_index_sequence<n_fuse>{});
    } else {
        fprintf(stderr,
                "%s: unsupported types for fusion: dst: %s, src0: %s, src1: %s\n",
                __func__, ggml_type_name(dst->type), ggml_type_name(src0->type), ggml_type_name(src1->type));
        GGML_ABORT("fatal error");
    }
}


void ggml_cuda_op_fused_add(ggml_backend_cuda_context & ctx, ggml_tensor * dst, int n_fuse) {
    GGML_ASSERT(2 <= n_fuse && n_fuse <= 8);

    switch (n_fuse) {
        case 2:
            ggml_cuda_op_fused_binbcast_impl<op_add, 2>(ctx, dst);
            break;
        case 3:
            ggml_cuda_op_fused_binbcast_impl<op_add, 3>(ctx, dst);
            break;
        case 4:
            ggml_cuda_op_fused_binbcast_impl<op_add, 4>(ctx, dst);
            break;
        case 5:
            ggml_cuda_op_fused_binbcast_impl<op_add, 5>(ctx, dst);
            break;
        case 6:
            ggml_cuda_op_fused_binbcast_impl<op_add, 6>(ctx, dst);
            break;
        case 7:
            ggml_cuda_op_fused_binbcast_impl<op_add, 7>(ctx, dst);
            break;
        case 8:
            ggml_cuda_op_fused_binbcast_impl<op_add, 8>(ctx, dst);
            break;
        default:
            GGML_ASSERT(false && "Unsupported n_fuse value");
    }
}

void ggml_cuda_op_repeat_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0->type == dst->type);
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_can_repeat(dst, src0));

    cudaStream_t stream = ctx.stream();

    GGML_TENSOR_UNARY_OP_LOCALS;

    GGML_ASSERT(ne2*ne3 <= (1 << 15));

    const size_t ts = ggml_type_size(src0->type);
    const size_t s00 = nb00 / ts;
    const size_t s01 = nb01 / ts;
    const size_t s02 = nb02 / ts;
    const size_t s03 = nb03 / ts;

    switch (dst->type) {
        case GGML_TYPE_F32: {
            const float * src0_d = (const float *) src0->data;
            float       * dst_d  = (float       *) dst->data;
            repeat_back_cuda(src0_d, dst_d, ne00, ne01, ne02, ne03, s00, s01, s02, s03, ne0, ne1, ne2, ne3, stream);
        } break;
        default: {
            GGML_ASSERT(false);
        } break;
    }
}
