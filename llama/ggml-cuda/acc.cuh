#include "common.cuh"

#define CUDA_ACC_BLOCK_SIZE 256

void ggml_cuda_op_acc(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
