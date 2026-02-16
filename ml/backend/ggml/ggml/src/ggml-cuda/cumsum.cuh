#include "common.cuh"

#define CUDA_CUMSUM_BLOCK_SIZE 256

void ggml_cuda_op_cumsum(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
