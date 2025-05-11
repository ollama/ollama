//
// MIT license
// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: MIT
//

//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#ifndef GGML_SYCL_CONVERT_HPP
#define GGML_SYCL_CONVERT_HPP

#include "common.hpp"

template <typename T>
using to_t_sycl_t = void (*)(const void * __restrict__ x, T * __restrict__ y, int64_t k, dpct::queue_ptr stream);
typedef to_t_sycl_t<float>      to_fp32_sycl_t;
typedef to_t_sycl_t<sycl::half> to_fp16_sycl_t;

to_fp16_sycl_t ggml_get_to_fp16_sycl(ggml_type type, ggml_tensor * dst);
to_fp32_sycl_t ggml_get_to_fp32_sycl(ggml_type type, ggml_tensor * dst);

// Nc = Non-contiguous
template <typename T>
using to_t_nc_sycl_t = void (*)(const void * x, T * y, int64_t ne00, int64_t ne01, int64_t ne02, int64_t ne03,
                                   int64_t s01, int64_t s02, int64_t s03, dpct::queue_ptr queue);

typedef to_t_nc_sycl_t<sycl::half> to_fp16_nc_sycl_t;
to_fp16_nc_sycl_t get_to_fp16_nc_sycl(ggml_type type);

#endif  // GGML_SYCL_CONVERT_HPP
