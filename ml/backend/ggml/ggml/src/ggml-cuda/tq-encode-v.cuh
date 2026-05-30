#pragma once

#include "common.cuh"

void ggml_cuda_tq_encode_v(ggml_backend_cuda_context & ctx, struct ggml_tensor * dst);

// Dispatch wrapper: selects the correct D template instantiation at runtime.
void tq_encode_v_dispatch(
    int headDim,
    dim3 grid, int block_size, size_t smem, cudaStream_t stream,
    const void *v, const float *rotation, uint8_t *packed_out, float *scales_out,
    int firstCell, const float *boundaries,
    int headDimArg, int numKVHeads, int bits, int numBoundaries, int vIsF32,
    const float *codebook,
    const int32_t *locs);
