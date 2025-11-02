#include "common.cuh"
#include "ggml.h"

#include <initializer_list>

void ggml_cuda_op_topk_moe(ggml_backend_cuda_context & ctx,
                           const ggml_tensor *         logits,
                           ggml_tensor *               weights,
                           ggml_tensor *               top_k,
                           const bool                  with_norm);

bool ggml_cuda_should_use_topk_moe(const ggml_tensor * softmax, const ggml_tensor * weights);

std::initializer_list<enum ggml_op> ggml_cuda_topk_moe_ops(bool with_norm);
