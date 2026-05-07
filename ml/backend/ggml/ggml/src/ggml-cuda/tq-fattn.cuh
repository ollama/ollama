#pragma once
#include "ggml-cuda/common.cuh"

void ggml_cuda_tq_flash_attn_ext(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
// K-only outlier fused path (V_PACKED=false, HAS_OUTLIERS=true); split into its
// own TU to stay under the gas single-object size limit for 10-arch fatbinaries.
void ggml_cuda_tq_flash_attn_ext_konly_outlier(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
