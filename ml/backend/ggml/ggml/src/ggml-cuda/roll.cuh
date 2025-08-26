#include "common.cuh"

#define CUDA_ROLL_BLOCK_SIZE 256

void ggml_cuda_op_roll(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
