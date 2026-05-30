#pragma once

#include "common.cuh"

void ggml_cuda_tq_encode(ggml_backend_cuda_context & ctx, struct ggml_tensor * dst);
void ggml_cuda_tq_encode_kv(ggml_backend_cuda_context & ctx, struct ggml_tensor * dst);
void ggml_cuda_tq_wht(ggml_backend_cuda_context & ctx, struct ggml_tensor * dst);

// Shared helper: Path B (EDEN refinement) is on by default; OLLAMA_TQ_DISABLE_EDEN=1
// forces RMS-only by nulling the codebook pointer in every encode dispatcher.
bool tq_encode_eden_disabled();
