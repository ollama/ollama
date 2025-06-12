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

#include "acl_tensor.h"

#include <algorithm>
#include <cstring>

aclDataType ggml_cann_type_mapping(ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:
            return ACL_FLOAT;
        case GGML_TYPE_F16:
            return ACL_FLOAT16;
        case GGML_TYPE_BF16:
            return ACL_BF16;
        case GGML_TYPE_I8:
            return ACL_INT8;
        case GGML_TYPE_I16:
            return ACL_INT16;
        case GGML_TYPE_I32:
            return ACL_INT32;
        case GGML_TYPE_Q4_0:
            return ACL_INT4;
        case GGML_TYPE_Q8_0:
            return ACL_INT8;
        case GGML_TYPE_I64:
            return ACL_INT64;
        default:
            return ACL_DT_UNDEFINED;
    }
    return ACL_DT_UNDEFINED;
}

aclTensor* ggml_cann_create_tensor(const ggml_tensor* tensor, int64_t* ne,
                                   size_t* nb, int64_t dims, aclFormat format,
                                   size_t offset) {
    // If tensor is bcasted, Up to GGML_MAX_DIMS additional dimensions will be
    // added.
    int64_t acl_ne[GGML_MAX_DIMS * 2], acl_stride[GGML_MAX_DIMS * 2];

    if (ne == nullptr) {
        for (int i = 0; i < GGML_MAX_DIMS; i++) {
            acl_ne[i] = tensor->ne[i];
            // The step size of acl is in elements.
            acl_stride[i] = tensor->nb[i] / ggml_element_size(tensor);
        }
    } else {
        // With bcast
        for (int i = 0; i < dims; i++) {
            acl_ne[i] = ne[i];
            acl_stride[i] = nb[i] / ggml_element_size(tensor);
        }
    }

    int64_t final_dims = (dims == 0 ? GGML_MAX_DIMS : dims);
    int64_t acl_storage_len = 1;
    for (int i = 0; i < final_dims; i++) {
        acl_storage_len += (acl_ne[i] - 1) * acl_stride[i];
    }

    // Reverse ne and stride.
    std::reverse(acl_ne, acl_ne + final_dims);
    std::reverse(acl_stride, acl_stride + final_dims);

    aclTensor* acl_tensor = aclCreateTensor(
        acl_ne, final_dims, ggml_cann_type_mapping(tensor->type), acl_stride,
        offset / ggml_element_size(tensor), format, &acl_storage_len, 1,
        tensor->data);

    return acl_tensor;
}

bool ggml_cann_need_bcast(const ggml_tensor* t0, const ggml_tensor* t1) {
    for (int i = 0; i < GGML_MAX_DIMS; i++) {
        if (t1->ne[i] != t0->ne[i] && t1->ne[i] != 1) {
            return true;
        }
    }
    return false;
}

int64_t ggml_cann_get_bcast_shape(const ggml_tensor* src0,
                                  const ggml_tensor* src1,
                                  int64_t* bcast_src0_ne,
                                  int64_t* bcast_src1_ne, size_t* bcast_src0_nb,
                                  size_t* bcast_src1_nb) {
    GGML_ASSERT(ggml_can_repeat(src1, src0));
    int bcast_dim_cnt = 0;
    for (int i = 0; i < GGML_MAX_DIMS; i++) {
        int64_t nr = src0->ne[i] / src1->ne[i];
        bcast_src0_ne[bcast_dim_cnt] = src0->ne[i] / nr;
        bcast_src1_ne[bcast_dim_cnt] = src1->ne[i];
        bcast_src0_nb[bcast_dim_cnt] = src0->nb[i];
        bcast_src1_nb[bcast_dim_cnt] = src1->nb[i];
        bcast_dim_cnt++;
        if (nr != 1) {
            // Need to add an extra dim.
            bcast_src0_ne[bcast_dim_cnt] = nr;
            bcast_src1_ne[bcast_dim_cnt] = 1;
            bcast_src0_nb[bcast_dim_cnt] = bcast_src0_nb[bcast_dim_cnt - 1] *
                                           bcast_src0_ne[bcast_dim_cnt - 1];
            bcast_src1_nb[bcast_dim_cnt] = bcast_src1_nb[bcast_dim_cnt - 1] *
                                           bcast_src1_ne[bcast_dim_cnt - 1];
            bcast_dim_cnt++;
        }
    }
    return bcast_dim_cnt;
}

int64_t ggml_cann_get_mulmat_bcast_shape(
    const int64_t* input_ne, const int64_t* weight_ne, const int64_t* dst_ne,
    const size_t* input_nb, const size_t* weight_nb, const size_t* dst_nb,
    int64_t* bcast_input_ne, int64_t* bcast_weight_ne, int64_t* bcast_dst_ne,
    size_t* bcast_input_nb, size_t* bcast_weight_nb, size_t* bcast_dst_nb) {
    // input and dst shoule in same shape, except first two dims.
    GGML_ASSERT(input_ne[2] == dst_ne[2]);
    GGML_ASSERT(input_ne[3] == dst_ne[3]);

    int bcast_dim_cnt = 0;

    // For mul_mat, a dimension needs to be added before the dimension that
    // weight needs to be expanded to satisfy the bcast rule of matrix
    // multiplication.
    for (int i = 0; i < GGML_MAX_DIMS; i++) {
        int64_t nr = input_ne[i] / weight_ne[i];
        // Do not use bcast in the first two dimensions because we only support
        // the bcast batch dimension. Just copy them.
        if (i < 2 || nr == 1) {
            bcast_input_ne[bcast_dim_cnt] = input_ne[i];
            bcast_weight_ne[bcast_dim_cnt] = weight_ne[i];
            bcast_dst_ne[bcast_dim_cnt] = dst_ne[i];

            bcast_input_nb[bcast_dim_cnt] = input_nb[i];
            bcast_weight_nb[bcast_dim_cnt] = weight_nb[i];
            bcast_dst_nb[bcast_dim_cnt] = dst_nb[i];
            bcast_dim_cnt++;
        } else {
            // Need to add an extra dim.
            bcast_input_ne[bcast_dim_cnt] = nr;
            bcast_dst_ne[bcast_dim_cnt] = nr;
            bcast_weight_ne[bcast_dim_cnt] = 1;
            bcast_input_nb[bcast_dim_cnt] = input_nb[i];
            bcast_dst_nb[bcast_dim_cnt] = dst_nb[i];
            bcast_weight_nb[bcast_dim_cnt] = weight_nb[i];
            bcast_dim_cnt++;

            bcast_input_ne[bcast_dim_cnt] = input_ne[i] / nr;
            bcast_dst_ne[bcast_dim_cnt] = dst_ne[i] / nr;
            bcast_weight_ne[bcast_dim_cnt] = weight_ne[i];
            bcast_input_nb[bcast_dim_cnt] = bcast_input_nb[bcast_dim_cnt - 1] *
                                            bcast_input_ne[bcast_dim_cnt - 1];
            bcast_dst_nb[bcast_dim_cnt] = bcast_dst_nb[bcast_dim_cnt - 1] *
                                          bcast_dst_ne[bcast_dim_cnt - 1];
            bcast_weight_nb[bcast_dim_cnt] =
                bcast_weight_nb[bcast_dim_cnt - 1] *
                bcast_weight_ne[bcast_dim_cnt - 1];
            bcast_dim_cnt++;
        }
    }
    return bcast_dim_cnt;
}
