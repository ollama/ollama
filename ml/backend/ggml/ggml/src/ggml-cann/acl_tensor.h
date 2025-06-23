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

#ifndef CANN_ACL_TENSOR_H
#define CANN_ACL_TENSOR_H

#include <algorithm>
#include <cstring>

#include <aclnn/aclnn_base.h>
#include "common.h"

/**
 * @brief	Maps a ggml_type to its corresponding aclDataType.
 *
 * @details	This function takes a ggml_type as input and returns the corresponding
 *			aclDataType. It supports mapping for various ggml_types. If the input type
 *			does not match any of the predefined ggml_types, the function returns
 *          ACL_DT_UNDEFINED.
 *
 * @param	type    The ggml_type to be mapped.
 * @return	The corresponding aclDataType. If the input type is not recognized,
 *			ACL_DT_UNDEFINED is returned.
 */
aclDataType ggml_cann_type_mapping(ggml_type type);

/**
 * @brief   Creates an ACL tensor from a ggml_tensor with optional shape.
 *
 * @details This function creates an ACL tensor based on the properties of the
 *          provided ggml_tensor. It supports customer shape by adjusting dimensions
 *          and strides accordingly. If customer shape is applied, additional
 *          dimensions and strides are calculated based on the provided parameters.
 *
 * @param   tensor      Pointer to the ggml_tensor to be converted to ACL tensor.
 * @param   ne          Pointer to an array containing dimensions. Defaults to nullptr
 *                      if no customer shape is applied.
 * @param   nb          Pointer to an array containing strides. Defaults to nullptr
 *                      if no customer shape is applied.
 * @param   dims        Number of dimensions in the tensor. Defaults to 0 if no customer
 *                      shape is applied.
 * @param   format      ACL tensor format. Defaults to ACL_FORMAT_ND.
 * @param   offset      Offset in bytes for the ACL tensor data. Defaults to 0.
 * @return  Pointer to the created ACL tensor.
 */
aclTensor* ggml_cann_create_tensor(const ggml_tensor* tensor, int64_t* ne = nullptr,
                             size_t* nb = nullptr, int64_t dims = 0,
                             aclFormat format = ACL_FORMAT_ND,
                             size_t offset = 0);

/**
 * @brief   Template for creating an ACL tensor from provided parameters. typename TYPE
 *          should be size_t or float.
 *
 * @details This function creates an ACL tensor using the provided data pointer,
 *          data type, dimensions, strides, format, offset, and additional parameters.
 *          It calculates necessary dimensions and strides based on the provided ne and nb
 *          arrays, adjusting them for the ACL tensor creation. The ACL storage length
 *          is also calculated based on the provided dimensions and strides.
 *
 * @param   data_ptr    Pointer to the data buffer for the ACL tensor.
 * @param   dtype       ACL data type of the tensor.
 * @param   type_size   Size of each element in the tensor data buffer.
 * @param   ne          Pointer to an array containing tensor dimensions.
 * @param   nb          Pointer to an array containing tensor strides.
 * @param   dims        Number of dimensions of the tensor.
 * @param   format      ACL tensor format. Defaults to ACL_FORMAT_ND.
 * @param   offset      Offset in bytes for the ACL tensor data. Defaults to 0.
 * @return  Pointer to the created ACL tensor.
 */
template<typename TYPE>
aclTensor* ggml_cann_create_tensor(void* data_ptr, aclDataType dtype,
                                   TYPE type_size, int64_t* ne, TYPE* nb,
                                   int64_t dims,
                                   aclFormat format = ACL_FORMAT_ND,
                                   size_t offset = 0) {
    int64_t tmp_ne[GGML_MAX_DIMS * 2];
    int64_t tmp_stride[GGML_MAX_DIMS * 2];

    memcpy(tmp_ne, ne, dims * sizeof(int64_t));
    for (int i = 0; i < dims; i++) {
        tmp_stride[i] = nb[i] / type_size;
    }

    int64_t acl_storage_len = 1;
    for (int i = 0; i < dims; i++) {
        acl_storage_len += (tmp_ne[i] - 1) * tmp_stride[i];
    }

    std::reverse(tmp_ne, tmp_ne + dims);
    std::reverse(tmp_stride, tmp_stride + dims);

    aclTensor* acl_tensor =
        aclCreateTensor(tmp_ne, dims, dtype, tmp_stride, offset / type_size,
                        format, &acl_storage_len, 1, data_ptr);

    return acl_tensor;
}

/**
 * @brief   Checks if tensors require broadcasting based on their shapes.
 *
 * @details This function determines if two ggml_tensors need to be broadcasted for
 *          element-wise operations. Broadcasting is necessary if the shapes of the
 *          tensors are not identical and no dimension in either tensor equals 1.
 *
 * @param   t0      Pointer to the first ggml_tensor.
 * @param   t1      Pointer to the second ggml_tensor.
 * @return  True if broadcasting is needed, False otherwise.
 *
 * @remarks This function iterates over the dimensions of t0 and t1. It checks if each
 *          dimension in t1 differs from t0's corresponding dimension and is not equal
 *          to 1. If such a dimension is found, broadcasting is required to align t1
 *          with t0 for element-wise operations.
 */
bool ggml_cann_need_bcast(const ggml_tensor* t0, const ggml_tensor* t1);

/**
 * @brief   Computes broadcast shapes and strides for two ggml_tensors.
 *
 * @details This function calculates the broadcast shapes and strides for two ggml_tensors,
 *          following the broadcasting rules similar to numpy. It adjusts dimensions and
 *          strides to ensure compatibility for element-wise operations where one tensor
 *          can be broadcasted to match the shape of another tensor.
 *
 * @param   src0                Pointer to the first ggml_tensor.
 * @param   src1                Pointer to the second ggml_tensor.
 * @param   bcast_ne_src0       Output array to store broadcasted dimensions for src0.
 * @param   bcast_ne_src1       Output array to store broadcasted dimensions for src1.
 * @param   bcast_nb_src0       Output array to store broadcasted strides for src0.
 * @param   bcast_nb_src1       Output array to store broadcasted strides for src1.
 * @return  Number of dimensions in the broadcasted shape.
 *
 * @pre     ggml_can_repeat(src1, src0) must return true, indicating src1 can be broadcasted
 *          to match src0.
 *
 * @remarks This function iterates over the dimensions of src0 and src1, calculating the
 *          necessary broadcast dimensions and strides. If a dimension requires broadcasting
 *          (i.e., its size in src1 is smaller than in src0), an additional dimension is
 *          added with size calculated to match src0's dimension. This adjustment ensures
 *          that src1 can be element-wise broadcasted to src0's shape.
 *
 *  How it works:
 *
 *  if dim0 has padding.
 *  a -> (2, 2) padding = 2
 *   a: [[1, 2, *, *]
 *       [2, 3, *, *]]
 *  nb = (8, 4, 2)
 *
 *  if a should bcast with b -> (2, 4)
 *  b' -> (2, 2, 2)
 *  b : [[1, 2, 3, 4, *, *]
 *       [5, 6, 7, 8, *, *]]
 *  nb = (12, 6, 1)
 *
 *  after bcast:
 *  a' -> (2, 1, 2)
 *  a': [[[1, 2], *, *]
 *       [[2, 3], *, *]]
 *  nb = (8, 4, 2, 1)
 *
 *  b' : [[[1, 2], [3, 4], *, *]
 *        [[5, 6], [7, 8], *, *]]
 *  nb = (12, 6, 2, 1)
 *  \endcode
 *
 *  dim1 in a inserted dim, should add nb for dim1,
 *  and all other nb moves to next in order.
 */
int64_t ggml_cann_get_bcast_shape(const ggml_tensor* src0, const ggml_tensor* src1,
                        int64_t* bcast_ne_src0, int64_t* bcast_ne_src1,
                        size_t* bcast_nb_src0, size_t* bcast_nb_src1);

// Bcast macro to avoid duplicate code.
#define BCAST_SHAPE(src0, src1)                                              \
    int64_t bcast_##src0##_ne[GGML_MAX_DIMS * 2];                            \
    int64_t bcast_##src1##_ne[GGML_MAX_DIMS * 2];                            \
    size_t bcast_##src0##_nb[GGML_MAX_DIMS * 2];                             \
    size_t bcast_##src1##_nb[GGML_MAX_DIMS * 2];                             \
    int64_t bcast_dims = ggml_cann_get_bcast_shape(                          \
        src0, src1, bcast_##src0##_ne, bcast_##src1##_ne, bcast_##src0##_nb, \
        bcast_##src1##_nb);

#define BCAST_PARAM(tensor) bcast_##tensor##_ne, bcast_##tensor##_nb, bcast_dims

/**
 * @brief Calculates broadcast shapes for matrix multiplication.
 *
 * @details This function computes the broadcast shapes required for matrix multiplication
 *          based on the input, weight, and destination tensor shapes. It ensures that the
 *          dimensions of weight tensors are expanded appropriately to satisfy matrix
 *          multiplication broadcast rules.
 *
 * @param input_ne      Array containing the dimensions of the input tensor.
 * @param weight_ne     Array containing the dimensions of the weight tensor.
 * @param dst_ne        Array containing the dimensions of the destination tensor.
 * @param input_nb      Array containing the strides of the input tensor.
 * @param weight_nb     Array containing the strides of the weight tensor.
 * @param dst_nb        Array containing the strides of the destination tensor.
 * @param bcast_input_ne    Output array for broadcasted input tensor dimensions.
 * @param bcast_weight_ne   Output array for broadcasted weight tensor dimensions.
 * @param bcast_dst_ne      Output array for broadcasted destination tensor dimensions.
 * @param bcast_input_nb    Output array for broadcasted input tensor strides.
 * @param bcast_weight_nb   Output array for broadcasted weight tensor strides.
 * @param bcast_dst_nb      Output array for broadcasted destination tensor strides.
 * @return The number of dimensions in the broadcasted tensors.
 *
 * @remarks This function iterates over the tensor dimensions and calculates the broadcast
 *          shapes needed for matrix multiplication. It ensures that dimensions where
 *          weight tensor requires expansion are appropriately handled to conform with
 *          broadcasting rules.
 * @note compare with ggml_cann_get_bcast_shape, mul_mat broadcast need add this new dim
 *       before cast dim.
 * @sa ggml_cann_get_bcast_shape
 */
int64_t ggml_cann_get_mulmat_bcast_shape(
    const int64_t* input_ne, const int64_t* weight_ne, const int64_t* dst_ne,
    const size_t* input_nb, const size_t* weight_nb, const size_t* dst_nb,
    int64_t* bcast_input_ne, int64_t* bcast_weight_ne, int64_t* bcast_dst_ne,
    size_t* bcast_input_nb, size_t* bcast_weight_nb, size_t* bcast_dst_nb);

// Bcast macro to avoid duplicate code.
#define BCAST_MUL_MAT_SHAPE(input, weight, dst)                         \
    int64_t bcast_##input##_ne[GGML_MAX_DIMS * 2];                      \
    int64_t bcast_##weight##_ne[GGML_MAX_DIMS * 2];                     \
    int64_t bcast_##dst##_ne[GGML_MAX_DIMS * 2];                        \
    size_t bcast_##input##_nb[GGML_MAX_DIMS * 2];                       \
    size_t bcast_##weight##_nb[GGML_MAX_DIMS * 2];                      \
    size_t bcast_##dst##_nb[GGML_MAX_DIMS * 2];                         \
    int64_t bcast_dims = ggml_cann_get_mulmat_bcast_shape(              \
        input->ne, weight->ne, dst->ne, input->nb, weight->nb, dst->nb, \
        bcast_##input##_ne, bcast_##weight##_ne, bcast_##dst##_ne,      \
        bcast_##input##_nb, bcast_##weight##_nb, bcast_##dst##_nb);

#define BCAST_MUL_MAT_PARAM(tensor) \
    bcast_##tensor##_ne, bcast_##tensor##_nb, bcast_dims

#endif  // CANN_ACL_TENSOR_H
