#include "common.cuh"

#define CUDA_FWHT_BLOCK_SIZE 256

void ggml_cuda_op_fwht(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
