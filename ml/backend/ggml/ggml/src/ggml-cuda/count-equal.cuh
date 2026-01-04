#include "common.cuh"

#define CUDA_COUNT_EQUAL_CHUNK_SIZE 128

void ggml_cuda_count_equal(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
