#pragma once

#include "common.cuh"
#include "mmq.cuh"

#include <cstdint>

#define CUDA_QUANTIZE_BLOCK_SIZE     256
#define CUDA_QUANTIZE_BLOCK_SIZE_MMQ 128

static_assert(MATRIX_ROW_PADDING %    CUDA_QUANTIZE_BLOCK_SIZE      == 0, "Risk of out-of-bounds access.");
static_assert(MATRIX_ROW_PADDING % (4*CUDA_QUANTIZE_BLOCK_SIZE_MMQ) == 0, "Risk of out-of-bounds access.");

typedef void (*quantize_cuda_t)(
    const float * x, void * vy, const int64_t kx0, const int64_t kx1, const int64_t channels, const int64_t kx0_padded,
    const ggml_type type_x, cudaStream_t stream);

void quantize_row_q8_1_cuda(
    const float * x, void * vy, const int64_t kx0, const int64_t kx1, const int64_t channels, const int64_t kx0_padded,
    const ggml_type type_x, cudaStream_t stream);

void quantize_mmq_q8_1_cuda(
    const float * x, void * vy, const int64_t kx0, const int64_t kx1, const int64_t channels, const int64_t kx0_padded,
    const ggml_type type_x, cudaStream_t stream);
