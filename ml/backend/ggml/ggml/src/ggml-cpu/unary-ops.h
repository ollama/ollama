#pragma once

#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

void ggml_compute_forward_abs(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_sgn(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_neg(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_step(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_tanh(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_elu(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_relu(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_sigmoid(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_hardsigmoid(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_exp(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_hardswish(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_sqr(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_sqrt(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_sin(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_cos(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_log(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_floor(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_ceil(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_round(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_trunc(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_xielu(const struct ggml_compute_params * params, struct ggml_tensor * dst);

#ifdef __cplusplus
}
#endif
