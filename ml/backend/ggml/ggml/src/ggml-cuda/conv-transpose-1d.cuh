#include "common.cuh"

#define CUDA_CONV_TRANPOSE_1D_BLOCK_SIZE 256

void ggml_cuda_op_conv_transpose_1d(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
