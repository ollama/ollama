#include "common.cuh"

void ggml_cuda_mul_mat_f(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst);

bool ggml_cuda_should_use_mmf(enum ggml_type type, int cc, int warp_size, const int64_t * scr0_ne, int64_t ne11);
