#pragma once

#include "common.cuh"

#define CUDA_SET_BLOCK_SIZE 256

void ggml_cuda_op_set(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
