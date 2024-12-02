#include "common.cuh"

#define CUDA_GET_ROWS_BLOCK_SIZE 256

void ggml_cuda_op_get_rows(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
