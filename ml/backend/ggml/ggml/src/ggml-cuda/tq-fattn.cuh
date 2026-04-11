#pragma once
#include "ggml-cuda/common.cuh"

void ggml_cuda_tq_flash_attn_ext(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
