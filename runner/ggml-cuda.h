/**
 * llama.cpp - git 465219b9143ac01db0990bbcb0a081ef72ec2008
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
#include "ggml-backend.h"

#ifdef GGML_USE_HIPBLAS
#define GGML_CUDA_NAME "ROCm"
#define GGML_CUBLAS_NAME "hipBLAS"
#else
#define GGML_CUDA_NAME "CUDA"
#define GGML_CUBLAS_NAME "cuBLAS"
#endif

#ifdef  __cplusplus
extern "C" {
#endif

#define GGML_CUDA_MAX_DEVICES       16

GGML_API void   ggml_init_cublas(void);
GGML_API void * ggml_cuda_host_malloc(size_t size);
GGML_API void   ggml_cuda_host_free(void * ptr);

GGML_API bool   ggml_cuda_can_mul_mat(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst);
GGML_API void   ggml_cuda_set_tensor_split(const float * tensor_split);
GGML_API void   ggml_cuda_transform_tensor(void * data, struct ggml_tensor * tensor);
GGML_API void   ggml_cuda_free_data(struct ggml_tensor * tensor);

GGML_API void   ggml_cuda_assign_buffers(struct ggml_tensor * tensor);
GGML_API void   ggml_cuda_assign_buffers_no_scratch(struct ggml_tensor * tensor);
GGML_API void   ggml_cuda_assign_buffers_force_inplace(struct ggml_tensor * tensor);

GGML_API void   ggml_cuda_assign_buffers_no_alloc(struct ggml_tensor * tensor);
GGML_API void   ggml_cuda_assign_scratch_offset(struct ggml_tensor * tensor, size_t offset);
GGML_API void   ggml_cuda_copy_to_device(struct ggml_tensor * tensor);

GGML_API void   ggml_cuda_set_main_device(int main_device);
GGML_API void   ggml_cuda_set_mul_mat_q(bool mul_mat_q);
GGML_API void   ggml_cuda_set_scratch_size(size_t scratch_size);
GGML_API void   ggml_cuda_free_scratch(void);
GGML_API bool   ggml_cuda_compute_forward(struct ggml_compute_params * params, struct ggml_tensor * tensor);

GGML_API int    ggml_cuda_get_device_count(void);
GGML_API void   ggml_cuda_get_device_description(int device, char * description, size_t description_size);

// backend API
GGML_API ggml_backend_t ggml_backend_cuda_init(void); // TODO: take a list of devices to use

#ifdef  __cplusplus
}
#endif
