#include "common.cuh"

#define CUDA_UPSCALE_BLOCK_SIZE 256

void ggml_cuda_op_upscale(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
