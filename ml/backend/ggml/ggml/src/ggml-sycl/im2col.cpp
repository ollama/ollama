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

#include "im2col.hpp"

#include <sycl/sycl.hpp>
#include <type_traits>  // For std::is_same_v

#include "ggml.h"

template <typename T>
static void im2col_kernel(const float * x, T * dst, int64_t batch_offset, int64_t offset_delta, int64_t IC, int64_t IW,
                          int64_t IH, int64_t OH, int64_t OW, int64_t KW, int64_t KH, int64_t pelements, int64_t CHW,
                          int s0, int s1, int p0, int p1, int d0, int d1, const sycl::nd_item<3> & item_ct1) {
    const int64_t work_group_size = item_ct1.get_local_range(2);
    const int64_t global_id       = item_ct1.get_local_id(2) + (work_group_size * item_ct1.get_group(2));

    // make each work-item deal with more elements since sycl global range can not exceed max int
    for (int64_t i = global_id; i < pelements; i += (work_group_size * item_ct1.get_group_range(2))) {
        const int64_t ksize = OW * (KH > 1 ? KW : 1);
        const int64_t kx    = i / ksize;
        const int64_t kd    = kx * ksize;
        const int64_t ky    = (i - kd) / OW;
        const int64_t ix    = i % OW;

        const int64_t oh    = item_ct1.get_group(1);
        const int64_t batch = item_ct1.get_group(0) / IC;
        const int64_t ic    = item_ct1.get_group(0) % IC;

        const int64_t iiw = (ix * s0) + (kx * d0) - p0;
        const int64_t iih = (oh * s1) + (ky * d1) - p1;

        const int64_t offset_dst = (((batch * OH + oh) * OW + ix) * CHW) + (ic * (KW * KH) + ky * KW + kx);

        const int64_t offset_src_base = (ic * offset_delta) + (batch * batch_offset);
        const int64_t offset_src      = offset_src_base + (iih * IW) + iiw;

        const bool  out_of_bounds = (iih < 0 || iih >= IH || iiw < 0 || iiw >= IW);
        const float src_val       = out_of_bounds ? 0.0f : x[offset_src];

        if constexpr (std::is_same_v<T, sycl::half>) {
            dst[offset_dst] = sycl::half(src_val);
        } else if constexpr (std::is_same_v<T, float>) {
            dst[offset_dst] = src_val;
        }
    }
}

template <typename T>
static void im2col_sycl_internal(const float * x, T * dst, int64_t IW, int64_t IH, int64_t OW, int64_t OH, int64_t KW,
                                 int64_t KH, int64_t IC, int64_t batch, int64_t batch_offset, int64_t offset_delta,
                                 int s0, int s1, int p0, int p1, int d0, int d1, queue_ptr stream) {
    const int64_t parallel_elements = OW * KW * KH;
    const int64_t num_blocks        = (parallel_elements + SYCL_IM2COL_BLOCK_SIZE - 1) / SYCL_IM2COL_BLOCK_SIZE;

    // decrease global range when it exceeds the max int
    int64_t local_size = downsample_sycl_global_range(batch * IC * OH * num_blocks, SYCL_IM2COL_BLOCK_SIZE);

    sycl::range<3> block_nums(batch * IC, OH, num_blocks);
    sycl::range<3> local_range(1, 1, local_size);

    const int64_t CHW = IC * KH * KW;

    stream->parallel_for(sycl::nd_range<3>(block_nums * local_range, local_range), [=](sycl::nd_item<3> item_ct1) {
        im2col_kernel<T>(x, dst, batch_offset, offset_delta, IC, IW, IH, OH, OW, KW, KH, parallel_elements, CHW, s0, s1,
                         p0, p1, d0, d1, item_ct1);
    });
}

static void im2col_sycl_f16(const float * x, sycl::half * dst, int64_t IW, int64_t IH, int64_t OW, int64_t OH,
                            int64_t KW, int64_t KH, int64_t IC, int64_t batch, int64_t batch_offset,
                            int64_t offset_delta, int s0, int s1, int p0, int p1, int d0, int d1, queue_ptr stream) {
    if (!stream->get_device().has(sycl::aspect::fp16)) {
        throw sycl::exception(sycl::make_error_code(sycl::errc::kernel_not_supported),
                              "Device does not support half precision (fp16) operations!");
    }
    im2col_sycl_internal<sycl::half>(x, dst, IW, IH, OW, OH, KW, KH, IC, batch, batch_offset, offset_delta, s0, s1, p0,
                                     p1, d0, d1, stream);
}

static void im2col_sycl_f32(const float * x, float * dst, int64_t IW, int64_t IH, int64_t OW, int64_t OH, int64_t KW,
                            int64_t KH, int64_t IC, int64_t batch, int64_t batch_offset, int64_t offset_delta, int s0,
                            int s1, int p0, int p1, int d0, int d1, queue_ptr stream) {
    im2col_sycl_internal<float>(x, dst, IW, IH, OW, OH, KW, KH, IC, batch, batch_offset, offset_delta, s0, s1, p0, p1,
                                d0, d1, stream);
}

void ggml_sycl_op_im2col(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F16 || dst->type == GGML_TYPE_F32);

    const int32_t s0 = ((const int32_t *) (dst->op_params))[0];
    const int32_t s1 = ((const int32_t *) (dst->op_params))[1];
    const int32_t p0 = ((const int32_t *) (dst->op_params))[2];
    const int32_t p1 = ((const int32_t *) (dst->op_params))[3];
    const int32_t d0 = ((const int32_t *) (dst->op_params))[4];
    const int32_t d1 = ((const int32_t *) (dst->op_params))[5];

    const bool is_2D = ((const int32_t *) (dst->op_params))[6] == 1;

    const int64_t IC = src1->ne[is_2D ? 2 : 1];
    const int64_t IH = is_2D ? src1->ne[1] : 1;
    const int64_t IW = src1->ne[0];

    const int64_t KH = is_2D ? src0->ne[1] : 1;
    const int64_t KW = src0->ne[0];

    const int64_t OH = is_2D ? dst->ne[2] : 1;
    const int64_t OW = dst->ne[1];

    const size_t  delta_offset = src1->nb[is_2D ? 2 : 1] / sizeof(float);
    const int64_t batch        = src1->ne[is_2D ? 3 : 2];
    const size_t  batch_offset = src1->nb[is_2D ? 3 : 2] / sizeof(float);

    queue_ptr stream = ctx.stream();

    if (dst->type == GGML_TYPE_F16) {
        im2col_sycl_f16((const float *) src1->data, (sycl::half *) dst->data, IW, IH, OW, OH, KW, KH, IC, batch,
                        batch_offset, delta_offset, s0, s1, p0, p1, d0, d1, stream);
    } else {
        im2col_sycl_f32((const float *) src1->data, (float *) dst->data, IW, IH, OW, OH, KW, KH, IC, batch,
                        batch_offset, delta_offset, s0, s1, p0, p1, d0, d1, stream);
    }
}
