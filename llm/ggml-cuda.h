/**
 * llama.cpp - git 8183159cf3def112f6d1fe94815fce70e1bffa12
 *
 * MIT License
 *
 * Copyright (c) 2023 Georgi Gerganov
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#pragma once

#include "ggml.h"

#ifdef  __cplusplus
extern "C" {
#endif

#define GGML_CUDA_MAX_DEVICES       16

void   ggml_init_cublas(void);
void   ggml_cuda_set_tensor_split(const float * tensor_split);

void   ggml_cuda_mul(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst);
bool   ggml_cuda_can_mul_mat(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst);
size_t ggml_cuda_mul_mat_get_wsize(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst);
void   ggml_cuda_mul_mat(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst, void * wdata, size_t wsize);

// TODO: export these with GGML_API
void * ggml_cuda_host_malloc(size_t size);
void   ggml_cuda_host_free(void * ptr);

void   ggml_cuda_transform_tensor(void * data, struct ggml_tensor * tensor);

void   ggml_cuda_free_data(struct ggml_tensor * tensor);
void   ggml_cuda_assign_buffers(struct ggml_tensor * tensor);
void   ggml_cuda_assign_buffers_no_scratch(struct ggml_tensor * tensor);
void   ggml_cuda_assign_buffers_force_inplace(struct ggml_tensor * tensor);
void   ggml_cuda_set_main_device(int main_device);
void   ggml_cuda_set_mul_mat_q(bool mul_mat_q);
void   ggml_cuda_set_scratch_size(size_t scratch_size);
void   ggml_cuda_free_scratch(void);
bool   ggml_cuda_compute_forward(struct ggml_compute_params * params, struct ggml_tensor * tensor);

#ifdef  __cplusplus
}
#endif
