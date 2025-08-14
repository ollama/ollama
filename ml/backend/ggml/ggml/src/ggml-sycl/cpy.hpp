#ifndef GGML_SYCL_CPY_HPP
#define GGML_SYCL_CPY_HPP

#include "common.hpp"

typedef void (*cpy_kernel_t)(const char * cx, char * cdst);

void ggml_sycl_cpy(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1);
void ggml_sycl_dup(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

#endif // GGML_SYCL_CPY_HPP
