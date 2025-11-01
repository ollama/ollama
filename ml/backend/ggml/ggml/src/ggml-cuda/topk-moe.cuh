#include "common.cuh"
#include "ggml.h"

#include <initializer_list>

void ggml_cuda_op_topk_moe(ggml_backend_cuda_context & ctx,
                           const ggml_tensor *         logits,
                           ggml_tensor *               weights,
                           ggml_tensor *               ids,
                           const bool                  with_norm,
                           const bool                  delayed_softmax = false,
                           ggml_tensor *               weight_clamp    = nullptr);

bool ggml_cuda_should_use_topk_moe(const ggml_tensor * softmax, const ggml_tensor * weights, const ggml_tensor * clamp = nullptr);

std::initializer_list<enum ggml_op> ggml_cuda_topk_moe_ops(bool with_norm, bool delayed_softmax = false);
