#include "common.cuh"

#define CUDA_DIAG_MASK_INF_BLOCK_SIZE 32

void ggml_cuda_op_diag_mask_inf(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
