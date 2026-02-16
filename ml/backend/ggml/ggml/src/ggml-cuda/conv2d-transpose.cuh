#include "common.cuh"

#define CUDA_CONV2D_TRANSPOSE_BLOCK_SIZE 256
void ggml_cuda_conv_2d_transpose_p0(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
