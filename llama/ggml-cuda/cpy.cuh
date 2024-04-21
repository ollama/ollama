#include "common.cuh"

#define CUDA_CPY_BLOCK_SIZE 32

void ggml_cuda_cpy(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, ggml_tensor * src1);

void ggml_cuda_dup(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void* ggml_cuda_cpy_fn(const ggml_tensor * src0, ggml_tensor * src1);
