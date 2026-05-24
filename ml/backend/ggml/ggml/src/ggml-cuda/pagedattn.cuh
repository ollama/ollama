#include "common.cuh"

void ggml_cuda_op_paged_attention(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
bool ggml_cuda_paged_attention_supported(const ggml_tensor * dst);
