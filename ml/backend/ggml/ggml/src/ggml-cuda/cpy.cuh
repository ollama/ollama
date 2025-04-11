#include "common.cuh"

#define CUDA_CPY_BLOCK_SIZE 64

void ggml_cuda_cpy(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, ggml_tensor * src1);

void ggml_cuda_dup(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void* ggml_cuda_cpy_fn(const ggml_tensor * src0, ggml_tensor * src1);

void ggml_cuda_cpy_dest_ptrs_copy(ggml_cuda_graph * cuda_graph, char ** host_dest_ptrs, const int host_dest_ptrs_size, cudaStream_t stream);
