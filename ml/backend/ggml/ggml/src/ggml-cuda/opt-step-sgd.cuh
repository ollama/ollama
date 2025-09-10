#include "common.cuh"

#define CUDA_OPT_STEP_SGD_BLOCK_SIZE 256

void ggml_cuda_opt_step_sgd(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
