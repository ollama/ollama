#include "common.cuh"

#define CUDA_ALIBI_BLOCK_SIZE 32

void ggml_cuda_op_alibi(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
