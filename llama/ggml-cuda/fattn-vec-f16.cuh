#include "common.cuh"

void ggml_cuda_flash_attn_ext_vec_f16(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_flash_attn_ext_vec_f16_no_mma(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
