#include "binbcast.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <sycl/sycl.hpp>

#include "dpct/helper.hpp"
#include "ggml.h"

template <float (*bin_op)(const float, const float), typename src0_t, typename src1_t, typename dst_t>
static __dpct_inline__ void k_bin_bcast_contiguous(const src0_t * __restrict__ src0, const src1_t * __restrict__ src1,
                                                   dst_t * dst, std::size_t num_elements, const sycl::nd_item<1> & it) {
    auto element_id   = it.get_global_id(0);
    auto global_range = it.get_global_range(0);
    for (; element_id < num_elements; element_id += global_range) {
        auto  src0_float_val = sycl::vec(src0[element_id]).template convert<float, sycl::rounding_mode::rte>();
        auto  src1_float_val = sycl::vec(src1[element_id]).template convert<float, sycl::rounding_mode::rte>();
        float dst_val        = bin_op(src0_float_val[0], src1_float_val[0]);
        auto  val_to_store   = sycl::vec(dst_val).template convert<dst_t, sycl::rounding_mode::rte>();
        dst[element_id]      = val_to_store;
    }
}

template <float (*bin_op)(const float, const float), typename src0_t, typename src1_t, typename dst_t>
static __dpct_inline__ void k_bin_bcast(const src0_t * __restrict__ src0, const src1_t * __restrict__ src1, dst_t * dst,
                                        int ne0, int ne1, int ne2, int ne3, int ne10, int ne11, int ne12, int ne13,
                                        int s0, int s1, int s2, int s3, int s00, int s01, int s02, int s03, int s10,
                                        int s11, int s12, int s13, std::size_t num_dst_elements,
                                        const sycl::nd_item<1> & item_ct1) {
    auto calculate_logical_index =
        [](const std::array<int, 4> & dims, std::size_t element_id) __attribute__((always_inline))->std::array<int, 4> {
        std::array<int, 4> logical_index;
#pragma unroll(4)
        for (int i = 3; i >= 0; i--) {
            logical_index[i] = element_id % dims[i];
            element_id /= dims[i];
        }
        return logical_index;
    };

    auto calculate_index = [](const std::array<int, 4> & dims, const std::array<int, 4> & strides,
                              const std::array<int, 4> & indices) __attribute__((always_inline))
                               ->std::size_t {
        std::size_t index = 0;
#pragma unroll(4)
        for (int i = 0; i < 4; i++) {
            auto index_i = indices[i];
            if (indices[i] >= dims[i]) {
                index_i = indices[i] % dims[i];
            }
            index += strides[i] * index_i;
        }
        return index;
    };

    auto element_id = item_ct1.get_global_id(0);
    for (; element_id < num_dst_elements; element_id += item_ct1.get_global_range(0)) {
        auto  logical_index  = calculate_logical_index({ ne3, ne2, ne1, ne0 }, element_id);
        auto  src_0_index    = calculate_index({ ne3, ne2, ne1, ne0 }, { s03, s02, s01, s00 }, logical_index);
        auto  src_1_index    = calculate_index({ ne13, ne12, ne11, ne10 }, { s13, s12, s11, s10 }, logical_index);
        auto  dst_index      = calculate_index({ ne3, ne2, ne1, ne0 }, { s3, s2, s1, s0 }, logical_index);
        auto  src0_float_val = sycl::vec(src0[src_0_index]).template convert<float, sycl::rounding_mode::rte>();
        auto  src1_float_val = sycl::vec(src1[src_1_index]).template convert<float, sycl::rounding_mode::rte>();
        float dst_val        = bin_op(src0_float_val[0], src1_float_val[0]);
        auto  val_to_store   = sycl::vec(dst_val).template convert<dst_t, sycl::rounding_mode::rte>();
        dst[dst_index]       = val_to_store;
    }
}

template <float (*bin_op)(const float, const float)> struct bin_bcast_sycl {
    template <typename src0_t, typename src1_t, typename dst_t>
    void operator()(const src0_t * src0_dd, const src1_t * src1_dd, dst_t * dst_dd, const int64_t ne00,
                    const int64_t ne01, const int64_t ne02, const int64_t ne03, const int64_t ne10, const int64_t ne11,
                    const int64_t ne12, const int64_t ne13, const int64_t ne0, const int64_t ne1, const int64_t ne2,
                    const int64_t ne3, const size_t nb00, const size_t nb01, const size_t nb02, const size_t nb03,
                    const size_t nb10, const size_t nb11, const size_t nb12, const size_t nb13, const size_t nb0,
                    const size_t nb1, const size_t nb2, const size_t nb3, const bool src0_is_contiguous,
                    const bool src1_is_contiguous, const bool dst_is_contiguous, queue_ptr stream) {
        auto check_bcast_required = [](const std::array<int64_t, 4> & src_dims,
                                       const std::array<int64_t, 4> & dst_dims) -> bool {
            for (int i = 0; i < 4; i++) {
                if (dst_dims[i] > src_dims[i]) {
                    return true;
                }
            }
            return false;
        };

        dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });

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

        // dst strides in number of elements
        size_t s0 = nb0 / sizeof(dst_t);
        size_t s1 = nb1 / sizeof(dst_t);
        size_t s2 = nb2 / sizeof(dst_t);
        size_t s3 = nb3 / sizeof(dst_t);

        // src1 strides in number of elements
        size_t s10 = nb10 / sizeof(src0_t);
        size_t s11 = nb11 / sizeof(src1_t);
        size_t s12 = nb12 / sizeof(src1_t);
        size_t s13 = nb13 / sizeof(src1_t);

        // src0 strides in number of elements
        size_t s00 = nb00 / sizeof(src0_t);
        size_t s01 = nb01 / sizeof(src0_t);
        size_t s02 = nb02 / sizeof(src0_t);
        size_t s03 = nb03 / sizeof(src0_t);

        std::size_t num_dst_elements = static_cast<std::size_t>(ne0) * static_cast<std::size_t>(ne1) *
                                       static_cast<std::size_t>(ne2) * static_cast<std::size_t>(ne3);
        std::size_t local_range  = 256;
        std::size_t global_range = ceil_div(num_dst_elements, local_range) * local_range;

        bool needs_broadcasting = check_bcast_required({ ne00, ne01, ne02, ne03 }, { ne0, ne1, ne2, ne3 }) ||
                                  check_bcast_required({ ne10, ne11, ne12, ne13 }, { ne0, ne1, ne2, ne3 });
        bool all_contiguous = src0_is_contiguous && src1_is_contiguous && dst_is_contiguous;

        if (! needs_broadcasting && all_contiguous) {
            stream->submit([&](sycl::handler & cgh) {
                cgh.parallel_for(sycl::nd_range<1>({ global_range }, { local_range }), [=](sycl::nd_item<1> it) {
                    k_bin_bcast_contiguous<bin_op>(src0_dd, src1_dd, dst_dd, num_dst_elements, it);
                });
            });
        } else {
            stream->submit([&](sycl::handler & cgh) {
                cgh.parallel_for(sycl::nd_range<1>({ global_range }, { local_range }), [=](sycl::nd_item<1> it) {
                    k_bin_bcast<bin_op>(src0_dd, src1_dd, dst_dd, ne0, ne1, ne2, ne3, ne10, ne11, ne12, ne13, s0, s1,
                                        s2, s3, s00, s01, s02, s03, s10, s11, s12, s13, num_dst_elements, it);
                });
            });
        }
    }
};

template <class op>
inline void ggml_sycl_op_bin_bcast(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
                                   ggml_tensor * dst) {
    dpct::queue_ptr main_stream = ctx.stream();
    GGML_TENSOR_BINARY_OP_LOCALS

    if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
        op()((const float *) src0->data, (const float *) src1->data, (float *) dst->data, ne00, ne01, ne02, ne03, ne10,
             ne11, ne12, ne13, ne0, ne1, ne2, ne3, nb00, nb01, nb02, nb03, nb10, nb11, nb12, nb13, nb0, nb1, nb2, nb3,
             ggml_is_contiguous(src0), ggml_is_contiguous(src1), ggml_is_contiguous(dst), main_stream);
    } else if (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F16) {
        op()((const sycl::half *) src0->data, (const sycl::half *) src1->data, (sycl::half *) dst->data, ne00, ne01,
             ne02, ne03, ne10, ne11, ne12, ne13, ne0, ne1, ne2, ne3, nb00, nb01, nb02, nb03, nb10, nb11, nb12, nb13,
             nb0, nb1, nb2, nb3, ggml_is_contiguous(src0), ggml_is_contiguous(src1), ggml_is_contiguous(dst),
             main_stream);
    } else if (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16) {
        op()((const sycl::half *) src0->data, (const float *) src1->data, (sycl::half *) dst->data, ne00, ne01, ne02,
             ne03, ne10, ne11, ne12, ne13, ne0, ne1, ne2, ne3, nb00, nb01, nb02, nb03, nb10, nb11, nb12, nb13, nb0, nb1,
             nb2, nb3, ggml_is_contiguous(src0), ggml_is_contiguous(src1), ggml_is_contiguous(dst), main_stream);
    } else if (src0->type == GGML_TYPE_I32 && src1->type == GGML_TYPE_I32 && dst->type == GGML_TYPE_I32) {
        op()((const int32_t *) src0->data, (const int32_t *) src1->data, (int32_t *) dst->data, ne00, ne01, ne02, ne03,
             ne10, ne11, ne12, ne13, ne0, ne1, ne2, ne3, nb00, nb01, nb02, nb03, nb10, nb11, nb12, nb13, nb0, nb1, nb2,
             nb3, ggml_is_contiguous(src0), ggml_is_contiguous(src1), ggml_is_contiguous(dst), main_stream);
    } else if (src0->type == GGML_TYPE_I16 && src1->type == GGML_TYPE_I16 && dst->type == GGML_TYPE_I16) {
        op()((const int16_t *) src0->data, (const int16_t *) src1->data, (int16_t *) dst->data, ne00, ne01, ne02, ne03,
             ne10, ne11, ne12, ne13, ne0, ne1, ne2, ne3, nb00, nb01, nb02, nb03, nb10, nb11, nb12, nb13, nb0, nb1, nb2,
             nb3, ggml_is_contiguous(src0), ggml_is_contiguous(src1), ggml_is_contiguous(dst), main_stream);
    } else {
        fprintf(stderr, "%s: unsupported types: dst: %s, src0: %s, src1: %s\n", __func__, ggml_type_name(dst->type),
                ggml_type_name(src0->type), ggml_type_name(src1->type));
        GGML_ABORT("fatal error");
    }
}

inline void ggml_sycl_op_add(ggml_backend_sycl_context & ctx, ggml_tensor *dst) {

    ggml_sycl_op_bin_bcast<bin_bcast_sycl<op_add>>(ctx, dst->src[0], dst->src[1], dst);
}

inline void ggml_sycl_op_sub(ggml_backend_sycl_context & ctx, ggml_tensor *dst) {

    ggml_sycl_op_bin_bcast<bin_bcast_sycl<op_sub>>(ctx, dst->src[0], dst->src[1], dst);
}

inline void ggml_sycl_op_mul(ggml_backend_sycl_context & ctx, ggml_tensor *dst) {

    ggml_sycl_op_bin_bcast<bin_bcast_sycl<op_mul>>(ctx, dst->src[0], dst->src[1], dst);
}

inline void ggml_sycl_op_div(ggml_backend_sycl_context & ctx, ggml_tensor *dst) {

    ggml_sycl_op_bin_bcast<bin_bcast_sycl<op_div>>(ctx, dst->src[0], dst->src[1], dst);
}

inline void ggml_sycl_op_repeat(ggml_backend_sycl_context & ctx, ggml_tensor *dst) {
    ggml_sycl_op_bin_bcast<bin_bcast_sycl<op_repeat>>(ctx, dst, dst->src[0], dst);
}


void ggml_sycl_add(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_SYCL_DEBUG("call %s\n", __func__);
    ggml_sycl_op_add(ctx, dst);
    GGML_SYCL_DEBUG("call %s done\n", __func__);
}

void ggml_sycl_sub(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_SYCL_DEBUG("call %s\n", __func__);
    ggml_sycl_op_sub(ctx, dst);
    GGML_SYCL_DEBUG("call %s done\n", __func__);
}

void ggml_sycl_mul(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_SYCL_DEBUG("call %s\n", __func__);
    ggml_sycl_op_mul(ctx, dst);
    GGML_SYCL_DEBUG("call %s done\n", __func__);
}

void ggml_sycl_div(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_SYCL_DEBUG("call %s\n", __func__);
    ggml_sycl_op_div(ctx, dst);
    GGML_SYCL_DEBUG("call %s done\n", __func__);
}

void ggml_sycl_repeat(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_SYCL_DEBUG("call %s\n", __func__);
    ggml_sycl_op_repeat(ctx, dst);
    GGML_SYCL_DEBUG("call %s done\n", __func__);
}

