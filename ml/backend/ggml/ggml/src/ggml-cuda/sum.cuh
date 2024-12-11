#include "common.cuh"

void sum_f32_cuda(ggml_cuda_pool & pool, const float * x, float * dst, const int64_t ne, cudaStream_t stream);

void ggml_cuda_op_sum(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
