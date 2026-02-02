#ifndef GGML_SYCL_ELEMENTWISE_HPP
#define GGML_SYCL_ELEMENTWISE_HPP

#include "common.hpp"
#include "ggml.h"
#include <limits> // For std::numeric_limits

#define SYCL_GLU_BLOCK_SIZE 256

template <typename T>
T neg_infinity() {
    return -std::numeric_limits<T>::infinity();
}

template<typename T_Dst, typename T_Src = T_Dst>
struct typed_data {
    const T_Src * src;
    T_Dst * dst;
};

template<typename T_Dst, typename T_Src = T_Dst>
typed_data<T_Dst, T_Src> cast_data(ggml_tensor * dst) {
    return {
        /* .src = */ static_cast<const T_Src *>(dst->src[0]->data),
        /* .dst = */ static_cast<T_Dst *>(dst->data)
    };
}

const float GELU_QUICK_COEF = -1.702f;


void ggml_sycl_sqrt(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

void ggml_sycl_sin(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

void ggml_sycl_cos(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

void ggml_sycl_acc(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

void ggml_sycl_gelu(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

void ggml_sycl_silu(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

void ggml_sycl_gelu_quick(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

void ggml_sycl_swiglu_oai(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

void ggml_sycl_gelu_erf(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

void ggml_sycl_tanh(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

void ggml_sycl_relu(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

void ggml_sycl_sigmoid(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

void ggml_sycl_hardsigmoid(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

void ggml_sycl_hardswish(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

void ggml_sycl_exp(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

void ggml_sycl_log(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

void ggml_sycl_neg(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

void ggml_sycl_step(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

void ggml_sycl_leaky_relu(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

void ggml_sycl_sqr(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

void ggml_sycl_upscale(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

void ggml_sycl_clamp(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

void ggml_sycl_sgn(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

void ggml_sycl_abs(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

void ggml_sycl_elu(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

void ggml_sycl_geglu(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
void ggml_sycl_reglu(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
void ggml_sycl_swiglu(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
void ggml_sycl_geglu_erf(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
void ggml_sycl_geglu_quick(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
void ggml_sycl_floor(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
void ggml_sycl_ceil(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
void ggml_sycl_round(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
void ggml_sycl_trunc(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

void ggml_sycl_arange(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

#endif // GGML_SYCL_ELEMENTWISE_HPP
