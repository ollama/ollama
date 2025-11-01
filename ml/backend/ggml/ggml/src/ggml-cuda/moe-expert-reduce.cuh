#include "common.cuh"
#include "ggml.h"

#include <initializer_list>

void ggml_cuda_op_moe_expert_reduce(ggml_backend_cuda_context & ctx,
                                    const ggml_tensor *         experts,
                                    const ggml_tensor *         weights,
                                    ggml_tensor *               dst);

bool ggml_cuda_should_use_moe_expert_reduce(const ggml_cgraph * cgraph, int start_index, int end_index);
