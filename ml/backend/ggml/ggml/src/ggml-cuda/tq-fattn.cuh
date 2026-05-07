#pragma once
#include "ggml-cuda/common.cuh"

void ggml_cuda_tq_flash_attn_ext(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
// Per-dim dispatch TUs — each holds 18 template instantiations × 10 archs to
// stay under the gas single-object size limit (~2 GiB).
void ggml_cuda_tq_flash_attn_ext_d64 (ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_tq_flash_attn_ext_d128(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_tq_flash_attn_ext_d256(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_tq_flash_attn_ext_d512(ggml_backend_cuda_context & ctx, ggml_tensor * dst); // 16 instantiations (ncols capped at 2)
// K-only outlier fused path (V_PACKED=false, HAS_OUTLIERS=true); split across
// two TUs to stay under the gas single-object size limit for 10-arch fatbinaries.
void ggml_cuda_tq_flash_attn_ext_konly_outlier(ggml_backend_cuda_context & ctx, ggml_tensor * dst);      // D=64/128
void ggml_cuda_tq_flash_attn_ext_konly_outlier_wide(ggml_backend_cuda_context & ctx, ggml_tensor * dst); // D=256
