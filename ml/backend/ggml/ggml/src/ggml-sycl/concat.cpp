//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#include "concat.hpp"
#include "common.hpp"

static void concat_f32_dim0(const float *x, const float *y, float *dst,
                            const int ne0, const int ne00,
                            const sycl::nd_item<3> &item_ct1) {
  int nidx = item_ct1.get_local_id(2) +
             item_ct1.get_group(2) * item_ct1.get_local_range(2);
  if (nidx >= ne0) {
    return;
  }
  // operation
  int offset_dst = nidx + item_ct1.get_group(1) * ne0 +
                   item_ct1.get_group(0) * ne0 * item_ct1.get_group_range(1);
  if (nidx < ne00) { // src0
    int offset_src = nidx + item_ct1.get_group(1) * ne00 +
                     item_ct1.get_group(0) * ne00 * item_ct1.get_group_range(1);
    dst[offset_dst] = x[offset_src];
  } else {
    int offset_src =
        nidx - ne00 + item_ct1.get_group(1) * (ne0 - ne00) +
        item_ct1.get_group(0) * (ne0 - ne00) * item_ct1.get_group_range(1);
    dst[offset_dst] = y[offset_src];
  }
}

static void concat_f32_dim1(const float *x, const float *y, float *dst,
                            const int ne0, const int ne01,
                            const sycl::nd_item<3> &item_ct1) {
  int nidx = item_ct1.get_local_id(2) +
             item_ct1.get_group(2) * item_ct1.get_local_range(2);
  if (nidx >= ne0) {
    return;
  }
  // operation
  int offset_dst = nidx + item_ct1.get_group(1) * ne0 +
                   item_ct1.get_group(0) * ne0 * item_ct1.get_group_range(1);
  if (item_ct1.get_group(1) < (size_t) ne01) { // src0
    int offset_src =
        nidx + item_ct1.get_group(1) * ne0 + item_ct1.get_group(0) * ne0 * ne01;
    dst[offset_dst] = x[offset_src];
  } else {
    int offset_src =
        nidx + (item_ct1.get_group(1) - ne01) * ne0 +
        item_ct1.get_group(0) * ne0 * (item_ct1.get_group_range(1) - ne01);
    dst[offset_dst] = y[offset_src];
  }
}

static void concat_f32_dim2(const float *x, const float *y, float *dst,
                            const int ne0, const int ne02,
                            const sycl::nd_item<3> &item_ct1) {
  int nidx = item_ct1.get_local_id(2) +
             item_ct1.get_group(2) * item_ct1.get_local_range(2);
  if (nidx >= ne0) {
    return;
  }
  // operation
  int offset_dst = nidx + item_ct1.get_group(1) * ne0 +
                   item_ct1.get_group(0) * ne0 * item_ct1.get_group_range(1);
  if (item_ct1.get_group(0) < (size_t) ne02) { // src0
    int offset_src = nidx + item_ct1.get_group(1) * ne0 +
                     item_ct1.get_group(0) * ne0 * item_ct1.get_group_range(1);
    dst[offset_dst] = x[offset_src];
  } else {
    int offset_src =
        nidx + item_ct1.get_group(1) * ne0 +
        (item_ct1.get_group(0) - ne02) * ne0 * item_ct1.get_group_range(1);
    dst[offset_dst] = y[offset_src];
  }
}

static void concat_f32_sycl(const float *x, const float *y, float *dst,
                            int ne00, int ne01, int ne02, int ne0, int ne1,
                            int ne2, int dim, queue_ptr stream) {
  int num_blocks = (ne0 + SYCL_CONCAT_BLOCK_SIZE - 1) / SYCL_CONCAT_BLOCK_SIZE;
  sycl::range<3> gridDim(ne2, ne1, num_blocks);
  switch (dim) {
  case 0:
    stream->parallel_for(
        sycl::nd_range<3>(gridDim *
                              sycl::range<3>(1, 1, SYCL_CONCAT_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_CONCAT_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) {
          concat_f32_dim0(x, y, dst, ne0, ne00, item_ct1);
        });
    break;
  case 1:
    stream->parallel_for(
        sycl::nd_range<3>(gridDim *
                              sycl::range<3>(1, 1, SYCL_CONCAT_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_CONCAT_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) {
          concat_f32_dim1(x, y, dst, ne0, ne01, item_ct1);
        });
    break;
  // dim >=2 will be dispatched to the default path
  default:
    stream->parallel_for(
        sycl::nd_range<3>(gridDim *
                              sycl::range<3>(1, 1, SYCL_CONCAT_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_CONCAT_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) {
          concat_f32_dim2(x, y, dst, ne0, ne02, item_ct1);
        });
    break;
  }
}

// non-contiguous kernel (slow)
static void concat_f32_sycl_non_cont(
    queue_ptr stream, const char *src0, const char *src1, char *dst,
    int64_t ne00, int64_t ne01, int64_t ne02, int64_t ne03, uint64_t nb00,
    uint64_t nb01, uint64_t nb02, uint64_t nb03, int64_t /*ne10*/,
    int64_t /*ne11*/, int64_t /*ne12*/, int64_t /*ne13*/, uint64_t nb10,
    uint64_t nb11, uint64_t nb12, uint64_t nb13, int64_t ne0, int64_t ne1,
    int64_t ne2, int64_t ne3, uint64_t nb0, uint64_t nb1, uint64_t nb2,
    uint64_t nb3, int32_t dim) {
  sycl::range<3> gridDim(ne3, ne2, ne1);
  stream->parallel_for(
      sycl::nd_range<3>(gridDim, sycl::range<3>(1, 1, 1)),
      [=](sycl::nd_item<3> item_ct1) {
        int64_t i3 = item_ct1.get_group(0);
        int64_t i2 = item_ct1.get_group(1);
        int64_t i1 = item_ct1.get_group(2);

        int64_t o[4] = {0, 0, 0, 0};
        o[dim] = dim == 0 ? ne00 : (dim == 1 ? ne01 : (dim == 2 ? ne02 : ne03));

        const float *x;

        for (int i0 = item_ct1.get_local_id(2); i0 < ne0;
             i0 += item_ct1.get_local_range(2)) {
          if (i0 < ne00 && i1 < ne01 && i2 < ne02 && i3 < ne03) {
            x = (const float *)(src0 + (i3)*nb03 + (i2)*nb02 + (i1)*nb01 +
                                (i0)*nb00);
          } else {
            x = (const float *)(src1 + (i3 - o[3]) * nb13 + (i2 - o[2]) * nb12 +
                                (i1 - o[1]) * nb11 + (i0 - o[0]) * nb10);
          }

          float *y = (float *)(dst + i3 * nb3 + i2 * nb2 + i1 * nb1 + i0 * nb0);

          *y = *x;
        }
      });
}

void ggml_sycl_op_concat(ggml_backend_sycl_context & ctx, ggml_tensor *dst) {
  const ggml_tensor *src0 = dst->src[0];
  const ggml_tensor *src1 = dst->src[1];
  queue_ptr stream = ctx.stream();

  const int32_t dim = ((int32_t *)dst->op_params)[0];

  if (ggml_is_contiguous(src0) && ggml_is_contiguous(src1)) {
    const float *src0_d = (const float *)src0->data;
    const float *src1_d = (const float *)src1->data;

    float *dst_d = (float *)dst->data;

    if (dim != 3) {
      for (int i3 = 0; i3 < dst->ne[3]; i3++) {
        concat_f32_sycl(
            src0_d + i3 * (src0->nb[3] / 4), src1_d + i3 * (src1->nb[3] / 4),
            dst_d + i3 * (dst->nb[3] / 4), src0->ne[0], src0->ne[1],
            src0->ne[2], dst->ne[0], dst->ne[1], dst->ne[2], dim, stream);
      }
    } else {
      const size_t size0 = ggml_nbytes(src0);
      const size_t size1 = ggml_nbytes(src1);

      SYCL_CHECK(CHECK_TRY_ERROR(stream->memcpy(dst_d, src0_d, size0).wait()));
      SYCL_CHECK(CHECK_TRY_ERROR(
          stream->memcpy(dst_d + size0 / 4, src1_d, size1).wait()));
    }
  } else
    concat_f32_sycl_non_cont(
        stream, (const char *)src0->data, (const char *)src1->data,
        (char *)dst->data, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
        src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3], src1->ne[0],
        src1->ne[1], src1->ne[2], src1->ne[3], src1->nb[0], src1->nb[1],
        src1->nb[2], src1->nb[3], dst->ne[0], dst->ne[1], dst->ne[2],
        dst->ne[3], dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3], dim);
}
