#include "common.cuh"

#define CUDA_QUANTIZE_BLOCK_SIZE 256

void quantize_row_q8_1_cuda(const float * x, void * vy, const int64_t kx, const int64_t ky, const int64_t kx_padded, cudaStream_t stream);
