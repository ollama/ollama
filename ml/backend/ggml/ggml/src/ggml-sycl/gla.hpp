#ifndef GGML_SYCL_GLA_HPP
#define GGML_SYCL_GLA_HPP

#include "common.hpp"

void ggml_sycl_op_gated_linear_attn(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

#endif  // GGML_SYCL_GLA_HPP
