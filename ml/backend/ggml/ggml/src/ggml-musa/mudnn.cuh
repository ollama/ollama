#pragma once

#include "ggml-cuda/common.cuh"
#include "ggml.h"

// Asynchronously copies data from src tensor to dst tensor using the provided context.
// Returns a musaError_t indicating success or failure.
musaError_t mudnnMemcpyAsync(
    ggml_backend_cuda_context &ctx,
    const ggml_tensor *dst,
    const ggml_tensor *src
);
