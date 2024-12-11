#include "common.cuh"

#define CUDA_ARANGE_BLOCK_SIZE 256

void ggml_cuda_op_arange(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
