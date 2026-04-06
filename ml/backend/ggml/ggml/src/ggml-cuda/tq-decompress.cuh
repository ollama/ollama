#include "common.cuh"

#define CUDA_TQ_DECOMPRESS_BLOCK_SIZE 256

void ggml_cuda_op_tq_decompress(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
