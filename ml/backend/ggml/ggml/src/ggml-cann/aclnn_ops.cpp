/*
 * Copyright (c) 2023-2024 The ggml authors
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#include "aclnn_ops.h"

#include <aclnnop/aclnn_addcdiv.h>
#include <aclnnop/aclnn_avgpool2d.h>
#include <aclnnop/aclnn_batch_matmul.h>
#include <aclnnop/aclnn_cast.h>
#include <aclnnop/aclnn_constant_pad_nd.h>
#include <aclnnop/aclnn_copy.h>
#include <aclnnop/aclnn_div.h>
#include <aclnnop/aclnn_embedding.h>
#include <aclnnop/aclnn_exp.h>
#include <aclnnop/aclnn_fill_scalar.h>
#include <aclnnop/aclnn_group_norm.h>
#include <aclnnop/aclnn_index_fill_tensor.h>
#include <aclnnop/aclnn_layer_norm.h>
#include <aclnnop/aclnn_matmul.h>
#include <aclnnop/aclnn_max_pool.h>
#include <aclnnop/aclnn_mm.h>
#include <aclnnop/aclnn_permute.h>
#include <aclnnop/aclnn_pow_tensor_tensor.h>
#include <aclnnop/aclnn_reduce_sum.h>
#include <aclnnop/aclnn_repeat.h>
#include <aclnnop/aclnn_repeat_interleave.h>
#include <aclnnop/aclnn_roll.h>
#include <aclnnop/aclnn_softmax.h>
#include <aclnnop/aclnn_tril.h>
#include <aclnnop/aclnn_triu.h>
#include <aclnnop/aclnn_upsample_nearest_2d.h>
#include <aclnnop/aclnn_weight_quant_batch_matmul_v2.h>
#include <aclnnop/aclnn_argmax.h>
#include <aclnnop/aclnn_sum.h>
#include <aclnnop/aclnn_rms_norm.h>
#include <aclnnop/aclnn_im2col.h>
#include <aclnnop/aclnn_add.h>
#include <aclnnop/aclnn_sub.h>
#include <aclnnop/aclnn_mul.h>
#include <aclnnop/aclnn_div.h>
#include <aclnnop/aclnn_convolution.h>
#include <aclnnop/aclnn_elu.h>
#include <aclnnop/aclnn_log.h>
#include <aclnnop/aclnn_mean.h>
#include <aclnnop/aclnn_reflection_pad1d.h>
#include <aclnnop/aclnn_eq_tensor.h>
#include <aclnnop/aclnn_gt_scalar.h>
#include <aclnnop/aclnn_pow.h>
#include <aclnnop/aclnn_grouped_matmul_v2.h>
#include <aclnnop/aclnn_fused_infer_attention_score_v2.h>
#include <float.h>

#include <cmath>
#include <cstring>
#include <exception>
#include <vector>

#include "ggml-impl.h"
#include "ggml.h"

#define GGML_COMMON_DECL_C

#include "../ggml-common.h"


void bcast_shape(ggml_tensor * src0, ggml_tensor * src1, ggml_tensor * dst, aclTensor ** acl_src0,
                 aclTensor ** acl_src1, aclTensor ** acl_dst) {
    GGML_ASSERT(ggml_are_same_shape(src0, dst) && ggml_can_repeat(src1, src0));
    // Need bcast
    if (!ggml_are_same_shape(src0, src1) && ggml_cann_need_bcast(src0, src1)) {
        BCAST_SHAPE(src0, src1)
        *acl_src0 = ggml_cann_create_tensor(src0, BCAST_PARAM(src0));
        *acl_src1 = ggml_cann_create_tensor(src1, BCAST_PARAM(src1));
        *acl_dst  = ggml_cann_create_tensor(dst, BCAST_PARAM(src0));
    } else {
        *acl_src0 = ggml_cann_create_tensor(src0);
        *acl_src1 = ggml_cann_create_tensor(src1);
        *acl_dst  = ggml_cann_create_tensor(dst);
    }
}

void ggml_cann_unary_op(
    std::function<void(ggml_backend_cann_context&, aclTensor*, aclTensor*)> unary_op,
    ggml_backend_cann_context& ctx, ggml_tensor* dst) {
    ggml_tensor* src = dst->src[0];

    aclTensor* acl_src = ggml_cann_create_tensor(src);
    aclTensor* acl_dst = ggml_cann_create_tensor(dst);

    unary_op(ctx, acl_src, acl_dst);
    ggml_cann_release_resources(ctx, acl_src, acl_dst);
}

/**
 * @brief Repeats elements of a tensor along each dimension according to the
 * specified repeat array.
 *
 * @param ctx The context for the CANN backend operations.
 * @param acl_src The source tensor to be repeated.
 * @param acl_dst The destination tensor after repeating.
 * @param repeat_array The array specifying the number of repetitions along each
 * dimension.
 */
static void aclnn_repeat(ggml_backend_cann_context& ctx, aclTensor* acl_src,
                         aclTensor* acl_dst, int64_t* repeat_array) {
    // repeat tensor along each dim with repeat_array
    aclIntArray* repeats = aclCreateIntArray(repeat_array, GGML_MAX_DIMS);

    GGML_CANN_CALL_ACLNN_OP(ctx, Repeat, acl_src, repeats, acl_dst);
    ggml_cann_release_resources(ctx, repeats);
}

/**
 * @brief Casts the data type of a source tensor to a destination tensor.
 *
 * This function casts the data type of the source tensor `acl_src` to the
 * specified data type `cast_data_type` and stores the result in the destination
 * tensor `acl_dst`.
 *
 * @param ctx The context for the CANN backend operations.
 * @param acl_src The source tensor whose data type will be casted.
 * @param acl_dst The destination tensor where the casted result will be stored.
 * @param cast_data_type The target data type to which the source tensor will be
 * casted.
 */
static void aclnn_cast(ggml_backend_cann_context& ctx, aclTensor* acl_src,
    aclTensor* acl_dst, aclDataType cast_data_type) {
    GGML_CANN_CALL_ACLNN_OP(ctx, Cast, acl_src, cast_data_type, acl_dst);
}

void ggml_cann_repeat(ggml_backend_cann_context& ctx, ggml_tensor* dst) {
    ggml_tensor* src = dst->src[0];
    GGML_ASSERT(ggml_can_repeat(src, dst));

    aclTensor* acl_src = ggml_cann_create_tensor(src);
    aclTensor* acl_dst = ggml_cann_create_tensor(dst);

    int64_t repeatsArray[] = {dst->ne[3] / src->ne[3], dst->ne[2] / src->ne[2],
                              dst->ne[1] / src->ne[1], dst->ne[0] / src->ne[0]};

    aclnn_repeat(ctx, acl_src, acl_dst, repeatsArray);
    ggml_cann_release_resources(ctx, acl_src, acl_dst);
}

void aclnn_add(ggml_backend_cann_context& ctx, aclTensor* acl_src0,
                      aclTensor* acl_src1, aclTensor* acl_dst) {
    float alphaValue = 1.0f;
    aclScalar* alpha = aclCreateScalar(&alphaValue, aclDataType::ACL_FLOAT);
    if (acl_dst != nullptr)
        GGML_CANN_CALL_ACLNN_OP(ctx, Add, acl_src0, acl_src1, alpha, acl_dst);
    else
        GGML_CANN_CALL_ACLNN_OP(ctx, InplaceAdd, acl_src0, acl_src1, alpha);
    ggml_cann_release_resources(ctx, alpha);
}

void aclnn_sub(ggml_backend_cann_context& ctx, aclTensor* acl_src0,
    aclTensor* acl_src1, aclTensor* acl_dst) {
    float alphaValue = 1.0f;
    aclScalar* alpha = aclCreateScalar(&alphaValue, aclDataType::ACL_FLOAT);
    if (acl_dst != nullptr)
        GGML_CANN_CALL_ACLNN_OP(ctx, Sub, acl_src0, acl_src1, alpha, acl_dst);
    else
        GGML_CANN_CALL_ACLNN_OP(ctx, InplaceSub, acl_src0, acl_src1, alpha);
    ggml_cann_release_resources(ctx, alpha);
}

void aclnn_mul(ggml_backend_cann_context& ctx, aclTensor* acl_src,
    aclTensor* acl_other, aclTensor* acl_dst) {
    if (acl_dst != nullptr)
        GGML_CANN_CALL_ACLNN_OP(ctx, Mul, acl_src, acl_other, acl_dst);
    else
        GGML_CANN_CALL_ACLNN_OP(ctx, InplaceMul, acl_src, acl_other);
}

void aclnn_div(ggml_backend_cann_context& ctx, aclTensor* acl_src,
    aclTensor* acl_other, aclTensor* acl_dst) {
    if (acl_dst != nullptr)
        GGML_CANN_CALL_ACLNN_OP(ctx, Div, acl_src, acl_other, acl_dst);
    else
        GGML_CANN_CALL_ACLNN_OP(ctx, InplaceDiv, acl_src, acl_other);
}

/**
 * @brief Multiplies elements of a tensor by a scalar value, optionally
 * in-place.
 *
 * This function multiplies each element of the source tensor `acl_src` by the
 * scalar `scale` and stores the result in the destination tensor `acl_dst`. If
 * `inplace` is true, `acl_dst` will not be used and the operation is performed
 *  in-place on `acl_src`.
 * The operation is defined as:
 * \f[
 *     \text {acl_dst }_i=\text {acl_src }_i \times \text {scale}
 * \f]
 *
 * @param ctx The context for the CANN backend operations.
 * @param acl_src The source tensor whose elements will be multiplied.
 * @param scale The scalar value by which each element of `acl_src` will be
 *  multiplied.
 * @param acl_dst The destination tensor where the result will be stored if
 * `inplace` is false.
 * @param inplace Flag indicating whether to perform the operation in-place on
 * `acl_src`.
 */
static void aclnn_muls(ggml_backend_cann_context& ctx, aclTensor* acl_src,
    float scale, aclTensor* acl_dst, bool inplace) {
    aclScalar* acl_scale = aclCreateScalar(&scale, aclDataType::ACL_FLOAT);
    if (inplace) {
        GGML_CANN_CALL_ACLNN_OP(ctx, InplaceMuls, acl_src, acl_scale);
    } else {
        GGML_CANN_CALL_ACLNN_OP(ctx, Muls, acl_src, acl_scale, acl_dst);
    }
    ggml_cann_release_resources(ctx, acl_scale);
}

void ggml_cann_leaky_relu(ggml_backend_cann_context& ctx, ggml_tensor* dst) {
    ggml_tensor* src = dst->src[0];

    GGML_ASSERT(src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    aclTensor* acl_src = ggml_cann_create_tensor(src);
    aclTensor* acl_dst = ggml_cann_create_tensor(dst);

    float negative_slope;
    memcpy(&negative_slope, dst->op_params, sizeof(float));
    aclScalar* acl_negative_slope =
        aclCreateScalar(&negative_slope, aclDataType::ACL_FLOAT);

    GGML_CANN_CALL_ACLNN_OP(ctx, LeakyRelu, acl_src, acl_negative_slope, acl_dst);
    ggml_cann_release_resources(ctx, acl_negative_slope, acl_src, acl_dst);
}

/**
 * @brief Concatenates a list of tensors along a specified dimension and stores
 * the result in a destination tensor.
 *
 * @param ctx The context for the CANN backend operations.
 * @param tensorList The list of tensors to be concatenated.
 * @param acl_dst The destination tensor where the concatenated result will be
 * stored.
 * @param concat_dim The dimension along which the tensors will be concatenated.
 */
static void aclnn_concat(ggml_backend_cann_context& ctx,
                         aclTensorList* tensorList, aclTensor* acl_dst,
                         int64_t concat_dim) {
    GGML_CANN_CALL_ACLNN_OP(ctx, Cat, tensorList, concat_dim, acl_dst);
}

void ggml_cann_concat(ggml_backend_cann_context& ctx, ggml_tensor* dst) {
    ggml_tensor* src0 = dst->src[0];
    ggml_tensor* src1 = dst->src[1];
    aclTensor* acl_src0 = ggml_cann_create_tensor(src0);
    aclTensor* acl_src1 = ggml_cann_create_tensor(src1);
    aclTensor* acl_dst = ggml_cann_create_tensor(dst);

    const int32_t dim = ggml_get_op_params_i32(dst, 0);

    GGML_ASSERT(dim >= 0 && dim < 4);
    int32_t acl_dim = 3 - dim;

    aclTensor* tensors[] = {acl_src0, acl_src1};
    aclTensorList* tensor_list = aclCreateTensorList(tensors, 2);
    aclnn_concat(ctx, tensor_list, acl_dst, acl_dim);

    ggml_cann_release_resources(ctx, tensor_list, acl_dst);
}

/**
 * @brief Creates a tensor with values starting from `start`, incremented by
 * `step`, and ending before `stop`.
 *
 * This function performs the operation:
 * \f[
 *    \text {out }_{i+1}=\text {out }_i+\text {step}
 * \f]
 * the range is [start, stop).
 *
 * @param ctx The context for the CANN backend operations.
 * @param acl_dst The destination tensor where the values will be stored.
 * @param start The starting value of the range.
 * @param stop The ending value of the range (exclusive).
 * @param step The step size between consecutive values.
 * @param n_elements The number of elements in the destination tensor.
 */
static void aclnn_arange(ggml_backend_cann_context& ctx, aclTensor* acl_dst,
                         float start, float stop, float step,
                         int64_t n_elements) {
    int64_t steps = (int64_t)std::ceil((stop - start) / step);
    GGML_ASSERT(n_elements == steps);

    aclScalar* acl_start = aclCreateScalar(&start, aclDataType::ACL_FLOAT);
    aclScalar* acl_end = aclCreateScalar(&stop, aclDataType::ACL_FLOAT);
    aclScalar* acl_step = aclCreateScalar(&step, aclDataType::ACL_FLOAT);

    GGML_CANN_CALL_ACLNN_OP(ctx, Arange, acl_start, acl_end, acl_step, acl_dst);
    ggml_cann_release_resources(ctx, acl_start, acl_end, acl_step);
}

void ggml_cann_arange(ggml_backend_cann_context& ctx, ggml_tensor* dst) {
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    aclTensor* acl_dst = ggml_cann_create_tensor(dst);

    int64_t n_elements = ggml_nelements(dst);
    float start;
    float stop;
    float step;
    memcpy(&start, (float*)dst->op_params + 0, sizeof(float));
    memcpy(&stop, (float*)dst->op_params + 1, sizeof(float));
    memcpy(&step, (float*)dst->op_params + 2, sizeof(float));

    aclnn_arange(ctx, acl_dst, start, stop, step, n_elements);
    ggml_cann_release_resources(ctx, acl_dst);
}

void ggml_cann_clamp(ggml_backend_cann_context& ctx, ggml_tensor* dst) {
    ggml_tensor* src = dst->src[0];

    float min;
    float max;
    memcpy(&min, dst->op_params, sizeof(float));
    memcpy(&max, (float*)dst->op_params + 1, sizeof(float));

    aclTensor* acl_src = ggml_cann_create_tensor(src);
    aclTensor* acl_dst = ggml_cann_create_tensor(dst);

    aclScalar* acl_min = aclCreateScalar(&min, aclDataType::ACL_FLOAT);
    aclScalar* acl_max = aclCreateScalar(&max, aclDataType::ACL_FLOAT);

    GGML_CANN_CALL_ACLNN_OP(ctx, Clamp, acl_src, acl_min, acl_max, acl_dst);
    ggml_cann_release_resources(ctx, acl_min, acl_max, acl_src, acl_dst);
}

void ggml_cann_scale(ggml_backend_cann_context& ctx, ggml_tensor* dst) {
    ggml_tensor* src = dst->src[0];

    // scale factor
    float v;
    memcpy(&v, dst->op_params, sizeof(float));

    aclScalar* scale = aclCreateScalar(&v, aclDataType::ACL_FLOAT);
    aclTensor* acl_src = ggml_cann_create_tensor(src);
    aclTensor* acl_dst = ggml_cann_create_tensor(dst);

    GGML_CANN_CALL_ACLNN_OP(ctx, Muls, acl_src, scale, acl_dst);
    ggml_cann_release_resources(ctx, scale, acl_src, acl_dst);
}

void ggml_cann_argsort(ggml_backend_cann_context& ctx, ggml_tensor* dst) {
    ggml_tensor* src = dst->src[0];
    enum ggml_sort_order order = (enum ggml_sort_order)dst->op_params[0];

    aclTensor* acl_src = ggml_cann_create_tensor(src);
    aclTensor* acl_dst = ggml_cann_create_tensor(dst);
    ggml_cann_pool_alloc temp_buffer_allocator(
        ctx.pool(), ggml_nelements(dst) * sizeof(int64_t));
    void* buffer = temp_buffer_allocator.get();
    aclTensor* tmp_tensor =
        ggml_cann_create_tensor(buffer, ACL_INT64, ggml_type_size(dst->type),
                                dst->ne, dst->nb, GGML_MAX_DIMS);
    GGML_CANN_CALL_ACLNN_OP(ctx, Argsort, acl_src, -1, (order == GGML_SORT_ORDER_DESC ? true : false),
                      tmp_tensor);
    GGML_CANN_CALL_ACLNN_OP(ctx, Cast, tmp_tensor, ggml_cann_type_mapping(dst->type), acl_dst);
    ggml_cann_release_resources(ctx, acl_src, tmp_tensor, acl_dst);
}

void ggml_cann_norm(ggml_backend_cann_context& ctx, ggml_tensor* dst) {
    ggml_tensor* src = dst->src[0];

    aclTensor* acl_src = ggml_cann_create_tensor(src);
    aclTensor* acl_dst = ggml_cann_create_tensor(dst);

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));

    std::vector<int64_t> normData = {dst->ne[0]};
    aclIntArray* norm = aclCreateIntArray(normData.data(), normData.size());
    GGML_CANN_CALL_ACLNN_OP(ctx, LayerNorm, acl_src, norm, nullptr, nullptr,
                    eps, acl_dst, nullptr, nullptr);
    ggml_cann_release_resources(ctx, norm, acl_src, acl_dst);
}

void ggml_cann_group_norm(ggml_backend_cann_context& ctx, ggml_tensor* dst) {
    ggml_tensor* src = dst->src[0];

    aclTensor* acl_src = ggml_cann_create_tensor(src);
    aclTensor* acl_dst = ggml_cann_create_tensor(dst);

    int n_groups = dst->op_params[0];

    float eps;
    memcpy(&eps, dst->op_params + 1, sizeof(float));

    int64_t N = src->ne[3];
    int64_t C = src->ne[2];
    int64_t HxW = src->ne[1] * src->ne[0];

    size_t type_size = ggml_type_size(src->type);
    int64_t ne[] = {n_groups, N};
    size_t nb[] = {type_size, type_size * n_groups};
    size_t n_bytes = N * n_groups;

    ggml_cann_pool_alloc temp_buffer_allocator(ctx.pool(), n_bytes * 2);
    void* buffer = temp_buffer_allocator.get();
    aclTensor* acl_mean_out = ggml_cann_create_tensor(
        buffer, ACL_FLOAT, type_size, ne, nb, ACL_FORMAT_ND);
    aclTensor* acl_rstd_out = ggml_cann_create_tensor(
        (char*)buffer + n_bytes, ACL_FLOAT, type_size, ne, nb, ACL_FORMAT_ND);

    GGML_CANN_CALL_ACLNN_OP(ctx, GroupNorm, acl_src, nullptr, nullptr, N, C, HxW, n_groups, eps,
        acl_dst, acl_mean_out, acl_rstd_out);
    ggml_cann_release_resources(ctx, acl_src, acl_dst, acl_mean_out, acl_rstd_out);
}

void ggml_cann_acc(ggml_backend_cann_context& ctx, ggml_tensor* dst) {
    ggml_tensor* src0 = dst->src[0];
    ggml_tensor* src1 = dst->src[1];

    size_t nb1 = ((int32_t*)dst->op_params)[0];
    size_t nb2 = ((int32_t*)dst->op_params)[1];
    size_t nb3 = ((int32_t*)dst->op_params)[2];
    size_t offset = ((int32_t*)dst->op_params)[3];
    bool inplace = (bool)((int32_t*)dst->op_params)[4];

    size_t param_nb[] = {ggml_element_size(src0), nb1, nb2, nb3};

    aclTensor* acl_dst = ggml_cann_create_tensor(
        dst, src1->ne, param_nb, GGML_MAX_DIMS, ACL_FORMAT_ND, offset);
    aclTensor* acl_src1 = ggml_cann_create_tensor(src1);

    aclScalar* alpha = nullptr;
    float alphaValue = 1.0f;
    alpha = aclCreateScalar(&alphaValue, aclDataType::ACL_FLOAT);

    if (!inplace) {
        size_t cpy_size = ggml_nbytes(dst);
        ggml_cann_async_memcpy(ctx, dst->data, src0->data, cpy_size,
            ACL_MEMCPY_DEVICE_TO_DEVICE);
        aclTensor* acl_src0 = ggml_cann_create_tensor(
            src0, src1->ne, src0->nb, GGML_MAX_DIMS, ACL_FORMAT_ND, offset);

        GGML_CANN_CALL_ACLNN_OP(ctx, Add, acl_src0, acl_src1, alpha, acl_dst);
        ggml_cann_release_resources(ctx, acl_src0);
    } else {
        GGML_CANN_CALL_ACLNN_OP(ctx, InplaceAdd, acl_dst, acl_src1, alpha);
    }
    ggml_cann_release_resources(ctx, acl_src1, acl_dst);
}

/**
 * @brief Performs sum reduction on a given tensor along specified dimensions.
 *
 * This function reduces the input tensor by summing along the specified dimensions.
 *
 * @param ctx The context for the CANN backend operations.
 * @param dst The destination tensor where the reduced result will be stored.
 * @param dim An array of dimension indices.
 * @param dim_size The number of dimensions.
 */
static void aclnn_reduce_sum(ggml_backend_cann_context& ctx, ggml_tensor* dst,
                             int64_t* dim, size_t dim_size) {
    GGML_ASSERT(dst->ne[0] == 1);
    ggml_tensor* src = dst->src[0];
    aclTensor* acl_src = ggml_cann_create_tensor(src);
    aclTensor* acl_dst = ggml_cann_create_tensor(dst);
    aclIntArray* reduce_dims = aclCreateIntArray(dim, dim_size);

    GGML_CANN_CALL_ACLNN_OP(ctx, ReduceSum, acl_src, reduce_dims, true,
                      ggml_cann_type_mapping(dst->type), acl_dst);
    ggml_cann_release_resources(ctx, acl_src, acl_dst, reduce_dims);
}

void ggml_cann_sum_rows(ggml_backend_cann_context& ctx, ggml_tensor* dst) {
    int64_t reduce_dims[] = {3};
    aclnn_reduce_sum(ctx, dst, reduce_dims, 1);
}

void ggml_cann_sum(ggml_backend_cann_context& ctx, ggml_tensor* dst) {
    int64_t reduce_dims[] = {0, 1, 2, 3};
    aclnn_reduce_sum(ctx, dst, reduce_dims, 4);
}

void ggml_cann_upsample_nearest2d(ggml_backend_cann_context& ctx,
                                  ggml_tensor* dst) {
    ggml_tensor* src = dst->src[0];
    aclTensor* acl_src =
        ggml_cann_create_tensor(src, nullptr, nullptr, 0, ACL_FORMAT_NCHW);
    aclTensor* acl_dst =
        ggml_cann_create_tensor(dst, nullptr, nullptr, 0, ACL_FORMAT_NCHW);

    std::vector<int64_t> output_size{dst->ne[1], dst->ne[0]};
    auto output_size_array = aclCreateIntArray(output_size.data(), 2);

    GGML_CANN_CALL_ACLNN_OP(ctx, UpsampleNearest2d, acl_src, output_size_array, acl_dst);
    ggml_cann_release_resources(ctx, acl_src, acl_dst, output_size_array);
}

/**
 * @brief Pads a tensor with a specified value along each dimension.
 *
 * This function performs padding of the source tensor `acl_src` and stores the
 * result in the destination tensor `acl_dst`. The padding values for each
 * dimension are specified in the `paddings` array.
 *
 * @param ctx The context for the CANN backend operations.
 * @param acl_src The source tensor to be padded.
 * @param acl_dst The destination tensor where the padded result will be stored.
 * @param paddings An array specifying the padding values for each dimension.
 * The size of the array should be twice the number of dimensions of the tensor.
 * @param value The value to be used for padding. The default value is 0.0.
 */
static void aclnn_pad(ggml_backend_cann_context& ctx, aclTensor* acl_src,
                      aclTensor* acl_dst, int64_t* paddings,
                      float value = 0.0f) {
    aclIntArray* acl_pad = aclCreateIntArray(paddings, GGML_MAX_DIMS * 2);
    aclScalar* acl_value = aclCreateScalar(&value, aclDataType::ACL_FLOAT);

    GGML_CANN_CALL_ACLNN_OP(ctx, ConstantPadNd, acl_src, acl_pad, acl_value, acl_dst);
    ggml_cann_release_resources(ctx, acl_pad, acl_value);
}

void ggml_cann_pad(ggml_backend_cann_context& ctx, ggml_tensor* dst) {
    ggml_tensor* src = dst->src[0];
    aclTensor* acl_src = ggml_cann_create_tensor(src);
    aclTensor* acl_dst = ggml_cann_create_tensor(dst);

    // padding: value in the array means how much distance will be padding.
    // the position of elements in the array means which dirction to padding,
    // each position means: [dim0.front, dim0.behind, dim1.front, dim1.behind,
    //                       dim2.front, dim2.behind, dim3.front, dim3.behind]
    int64_t paddings[] = {
        0, dst->ne[0] - src->ne[0], 0, dst->ne[1] - src->ne[1],
        0, dst->ne[2] - src->ne[2], 0, dst->ne[3] - src->ne[3]};
    aclnn_pad(ctx, acl_src, acl_dst, paddings);
    ggml_cann_release_resources(ctx, acl_src, acl_dst);
}

/**
 * @brief Performs 2D average pooling on the input tensor and stores the result
 * in the destination tensor.
 *
 * This function performs average pooling on the source tensor and stores the
 * result in the destination tensor. The pooling parameters (kernel size,
 * strides, padding) are specified in the `op_params` of the destination tensor.
 *
 * @param ctx The context for the CANN backend operations.
 * @param dst The destination tensor where the result will be stored. The source
 * tensor is referenced by `dst->src[0]`.
 */
static void ggml_cann_avg_pool2d(ggml_backend_cann_context& ctx,
                                 ggml_tensor* dst) {
    ggml_tensor* src = dst->src[0];
    GGML_ASSERT(src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    aclTensor* acl_src =
        ggml_cann_create_tensor(src, nullptr, nullptr, 0, ACL_FORMAT_NCHW);
    aclTensor* acl_dst =
        ggml_cann_create_tensor(dst, nullptr, nullptr, 0, ACL_FORMAT_NCHW);

    const int32_t* opts = (const int32_t*)dst->op_params;
    const int k0 = opts[1];
    const int k1 = opts[2];
    const int s0 = opts[3];
    const int s1 = opts[4];
    const int p0 = opts[5];
    const int p1 = opts[6];

    std::vector<int64_t> kernel_dims = {k1, k0};
    std::vector<int64_t> stride_dims = {s1, s0};
    std::vector<int64_t> padding_avg_dims = {p1, p0};  // (padH, padW)

    auto* kernel_size = aclCreateIntArray(kernel_dims.data(), 2);
    auto* strides = aclCreateIntArray(stride_dims.data(), 2);
    auto* paddings_avg = aclCreateIntArray(padding_avg_dims.data(), 2);

    bool ceil_mode = false;
    bool count_include_pad = true;
    int64_t divisor_override = 0;
    int8_t cube_math_type = 0;
#ifdef ASCEND_310P
    cube_math_type = 1;
#endif

    GGML_CANN_CALL_ACLNN_OP(ctx, AvgPool2d, acl_src, kernel_size, strides, paddings_avg,
                    ceil_mode, count_include_pad, divisor_override,
                    cube_math_type, acl_dst);
    ggml_cann_release_resources(ctx, acl_src, acl_dst, kernel_size, strides,
                                paddings_avg);
}

/**
 * @brief Performs 2D max pooling on the input tensor and stores the result in
 * the destination tensor.
 *
 * This function performs max pooling on the source tensor and stores the result
 * in the destination tensor. The pooling parameters (kernel size, strides,
 * padding) are specified in the `op_params` of the destination tensor.
 *
 * @param ctx The context for the CANN backend operations.
 * @param dst The destination tensor where the result will be stored. The source
 * tensor is referenced by `dst->src[0]`.
 */
static void ggml_cann_max_pool2d(ggml_backend_cann_context& ctx,
                                 ggml_tensor* dst) {
    ggml_tensor* src = dst->src[0];
    GGML_ASSERT(src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    aclTensor* acl_src =
        ggml_cann_create_tensor(src, nullptr, nullptr, 0, ACL_FORMAT_NCHW);
    aclTensor* acl_dst =
        ggml_cann_create_tensor(dst, nullptr, nullptr, 0, ACL_FORMAT_NCHW);

    const int32_t* opts = (const int32_t*)dst->op_params;
    const int k0 = opts[1];
    const int k1 = opts[2];
    const int s0 = opts[3];
    const int s1 = opts[4];
    const int p0 = opts[5];
    const int p1 = opts[6];

    int64_t temp_ne[] = {src->ne[0] + p0 * 2, src->ne[1] + p1 * 2, src->ne[2],
                         src->ne[3]};
    size_t temp_nb[GGML_MAX_DIMS];

    temp_nb[0] = ggml_element_size(src);
    for (int i = 1; i < GGML_MAX_DIMS; i++) {
        temp_nb[i] = temp_nb[i - 1] * temp_ne[i - 1];
    }

    ggml_cann_pool_alloc temp_buffer_allocator(
        ctx.pool(), ggml_nbytes(src) + p0 * 2 + p1 * 2 * src->nb[1]);
    void* buffer = temp_buffer_allocator.get();
    aclTensor* tmp_tensor = ggml_cann_create_tensor(
        buffer, ACL_FLOAT, ggml_element_size(src), temp_ne, temp_nb,
        GGML_MAX_DIMS, ACL_FORMAT_NCHW);

    // pad: see padding in ggml_cann_pad()
    int64_t paddings[] = {p0, p0, p1, p1, 0, 0, 0, 0};
    float value = -FLT_MAX;
    aclnn_pad(ctx, acl_src, tmp_tensor, paddings, value);

    // max_pool
    std::vector<int64_t> kernel_dims = {k1, k0};
    std::vector<int64_t> stride_dims = {s1, s0};
    // padding_max_dims: [dim0_start, dim0_end, dim1_start, dim1_end]
    std::vector<int64_t> padding_max_dims = {0, 0, 0, 0};
    std::vector<int64_t> dilation_size = {1, 1};
    auto* kernel_size = aclCreateIntArray(kernel_dims.data(), 2);
    auto* strides = aclCreateIntArray(stride_dims.data(), 2);
    auto* paddings_max = aclCreateIntArray(padding_max_dims.data(), 4);
    auto* dilations = aclCreateIntArray(dilation_size.data(), 2);

    bool ceil_mode = false;
    int64_t auto_pads = 0;
    GGML_CANN_CALL_ACLNN_OP(ctx, MaxPool, tmp_tensor, kernel_size, strides, auto_pads,
                    paddings_max, dilations, ceil_mode, acl_dst);
    ggml_cann_release_resources(ctx, acl_src, acl_dst, tmp_tensor, kernel_size,
                                strides, paddings_max, dilations);
}

void ggml_cann_pool2d(ggml_backend_cann_context& ctx, ggml_tensor* dst) {
    const int32_t* opts = (const int32_t*)dst->op_params;
    enum ggml_op_pool op = static_cast<ggml_op_pool>(opts[0]);
    switch (op) {
        case GGML_OP_POOL_AVG:
            ggml_cann_avg_pool2d(ctx, dst);
            break;
        case GGML_OP_POOL_MAX:
            ggml_cann_max_pool2d(ctx, dst);
            break;
        case GGML_OP_POOL_COUNT:
            GGML_ABORT("fatal error");
            break;
    }
}

/**
 * @brief Copies data from the source tensor to the destination tensor.
 *
 * This function copies data from the source tensor `acl_src` to the destination
 * tensor `acl_dst`.
 *
 * @param ctx The context for the CANN backend operations.
 * @param acl_src The source tensor from which data will be copied.
 * @param acl_dst The destination tensor where the data will be copied to.
 */
static void cann_copy(ggml_backend_cann_context& ctx, aclTensor* acl_src,
                      aclTensor* acl_dst) {
    GGML_CANN_CALL_ACLNN_OP(ctx, InplaceCopy, acl_dst, acl_src);
}

void ggml_cann_dup(ggml_backend_cann_context& ctx, ggml_tensor* dst) {
    ggml_tensor* src0 = dst->src[0];

    aclTensor* acl_src = ggml_cann_create_tensor(src0);
    aclTensor* acl_dst = ggml_cann_create_tensor(dst);
    if (ggml_are_same_shape(src0, dst)) {
        if (dst->type == src0->type) {
            cann_copy(ctx, acl_src, acl_dst);
        } else {
            aclnn_cast(ctx, acl_src, acl_dst, ggml_cann_type_mapping(dst->type));
        }
    } else {
        if (ggml_is_contiguous(src0) && ggml_is_contiguous(dst)) {
            if (dst->type == src0->type) {
                size_t cpy_size = ggml_nbytes(dst);
                ggml_cann_async_memcpy(ctx, dst->data, src0->data, cpy_size,
                    ACL_MEMCPY_DEVICE_TO_DEVICE);
                return;
            } else {
                ggml_cann_pool_alloc src_buffer_allocator(
                    ctx.pool(),
                    ggml_nelements(dst) * ggml_type_size(dst->type));
                void* src_trans_buffer = src_buffer_allocator.get();
                size_t src_trans_nb[GGML_MAX_DIMS];
                src_trans_nb[0] = ggml_type_size(dst->type);
                for (int i = 1; i < GGML_MAX_DIMS; i++) {
                    src_trans_nb[i] = src_trans_nb[i - 1] * src0->ne[i - 1];
                }
                aclTensor* src_trans_tensor = ggml_cann_create_tensor(
                    src_trans_buffer, ggml_cann_type_mapping(dst->type),
                    ggml_type_size(dst->type), src0->ne, src_trans_nb,
                    GGML_MAX_DIMS);

                aclnn_cast(ctx, acl_src, src_trans_tensor, ggml_cann_type_mapping(dst->type));
                size_t cpy_size = ggml_nbytes(dst);
                ggml_cann_async_memcpy(ctx, dst->data, src_trans_buffer, cpy_size,
                    ACL_MEMCPY_DEVICE_TO_DEVICE);
                ggml_cann_release_resources(ctx, src_trans_tensor);
                return;
            }
        } else if (ggml_is_contiguous(dst)) {
            ggml_cann_pool_alloc src_buffer_allocator(
                ctx.pool(), ggml_nelements(dst) * ggml_type_size(dst->type));
            void* src_trans_buffer = src_buffer_allocator.get();
            size_t src_trans_nb[GGML_MAX_DIMS];
            src_trans_nb[0] = ggml_type_size(dst->type);
            for (int i = 1; i < GGML_MAX_DIMS; i++) {
                src_trans_nb[i] = src_trans_nb[i - 1] * src0->ne[i - 1];
            }
            aclTensor* src_trans_tensor = ggml_cann_create_tensor(
                src_trans_buffer, ggml_cann_type_mapping(dst->type),
                ggml_type_size(dst->type), src0->ne, src_trans_nb,
                GGML_MAX_DIMS);

            aclnn_cast(ctx, acl_src, src_trans_tensor, ggml_cann_type_mapping(dst->type));

            size_t cpy_size = ggml_nbytes(dst);
            ggml_cann_async_memcpy(ctx, dst->data, src_trans_buffer, cpy_size,
                ACL_MEMCPY_DEVICE_TO_DEVICE);
            ggml_cann_release_resources(ctx, src_trans_tensor);
            return;
        } else {
            GGML_ABORT("Unsupport dst is not tontiguous.");
        }
    }
    ggml_cann_release_resources(ctx, acl_src, acl_dst);
}
