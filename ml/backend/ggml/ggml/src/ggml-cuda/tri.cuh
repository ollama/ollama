#include "common.cuh"

#define CUDA_TRI_BLOCK_SIZE 256

void ggml_cuda_op_tri(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
