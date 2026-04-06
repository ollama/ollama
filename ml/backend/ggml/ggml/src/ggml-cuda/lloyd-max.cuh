#include "common.cuh"

#define CUDA_LLOYD_MAX_BLOCK_SIZE 256

void ggml_cuda_op_lloyd_max_q(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_lloyd_max_dq(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
