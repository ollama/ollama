#include "common.cuh"

#define CUDA_POOL2D_BLOCK_SIZE 256

void ggml_cuda_op_pool2d(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
