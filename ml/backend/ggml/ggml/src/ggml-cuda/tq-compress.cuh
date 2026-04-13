#include "common.cuh"

#define CUDA_TQ_COMPRESS_BLOCK_SIZE 256

void ggml_cuda_op_tq_compress(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
