#include "common.cuh"

#define CUDA_CROSS_ENTROPY_LOSS_BLOCK_SIZE 256

void ggml_cuda_cross_entropy_loss(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_cross_entropy_loss_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
