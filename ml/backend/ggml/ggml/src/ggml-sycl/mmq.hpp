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

#ifndef GGML_SYCL_MMQ_HPP
#define GGML_SYCL_MMQ_HPP

#include "common.hpp"

void ggml_sycl_op_mul_mat_q(
    ggml_backend_sycl_context & ctx,
    const ggml_tensor* src0,
    const ggml_tensor* src1,
    ggml_tensor* dst,
    const char* src0_dd_i,
    const float* src1_ddf_i,
    const char* src1_ddq_i,
    float* dst_dd_i,
    const int64_t row_low,
    const int64_t row_high,
    const int64_t src1_ncols,
    const int64_t src1_padded_row_size,
    const dpct::queue_ptr& stream);

#endif // GGML_SYCL_MMQ_HPP
