#pragma once

#include "common.cuh"

void ggml_cuda_tq_encode_v(ggml_backend_cuda_context & ctx, struct ggml_tensor * dst);
