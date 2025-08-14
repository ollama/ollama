#ifndef GGML_SYCL_BINBCAST_HPP
#define GGML_SYCL_BINBCAST_HPP
#include "common.hpp"


static __dpct_inline__ float op_repeat(const float a, const float b) {
    return b;
    GGML_UNUSED(a);
}

static __dpct_inline__ float op_add(const float a, const float b) {
    return a + b;
}

static __dpct_inline__ float op_sub(const float a, const float b) {
    return a - b;
}

static __dpct_inline__ float op_mul(const float a, const float b) {
    return a * b;
}

static __dpct_inline__ float op_div(const float a, const float b) {
    return a / b;
}

void ggml_sycl_add(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

void ggml_sycl_sub(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

void ggml_sycl_mul(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

void ggml_sycl_div(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

void ggml_sycl_repeat(ggml_backend_sycl_context & ctx, ggml_tensor * dst);


#endif //GGML_SYCL_BINBCAST_HPP

