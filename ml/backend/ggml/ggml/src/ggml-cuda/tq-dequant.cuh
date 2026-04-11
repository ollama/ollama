#pragma once

#include "common.cuh"

void ggml_cuda_tq_dequant(ggml_backend_cuda_context & ctx, struct ggml_tensor * dst);
void ggml_cuda_tq_dequant_kv(ggml_backend_cuda_context & ctx, struct ggml_tensor * dst);
