#pragma once

#include "llama-impl.h"
#include "llama-hparams.h"

#include "ggml-cpp.h"

#include <unordered_map>
#include <vector>

//
// llama_adapter_cvec
//

// TODO: rename to llama_adapter_cvec
struct llama_control_vector {
    std::vector<ggml_context_ptr> ctxs;
    std::vector<ggml_backend_buffer_ptr> bufs;

    std::vector<struct ggml_tensor *> tensors; // per layer

    int32_t layer_start = -1;
    int32_t layer_end   = -1;

    struct ggml_tensor * tensor_for(int il) const;

    struct ggml_tensor * apply_to(struct ggml_context * ctx, struct ggml_tensor * cur, int  il) const;
};

int32_t llama_control_vector_apply(
        struct llama_control_vector & cvec,
        const llama_model & model,
        const float * data,
        size_t len,
        int32_t n_embd,
        int32_t il_start,
        int32_t il_end);

//
// llama_adapter_lora
//

// TODO: rename to llama_adapter_lora_weight
struct llama_lora_weight {
    struct ggml_tensor * a = nullptr;
    struct ggml_tensor * b = nullptr;

    llama_lora_weight() = default;
    llama_lora_weight(struct ggml_tensor * a, struct ggml_tensor * b) : a(a), b(b) {}
};

// TODO: rename to llama_adapter_lora
struct llama_lora_adapter {
    // map tensor name to lora_a_b
    std::unordered_map<std::string, struct llama_lora_weight> ab_map;

    std::vector<ggml_context_ptr> ctxs;
    std::vector<ggml_backend_buffer_ptr> bufs;

    float alpha;

    llama_lora_adapter() = default;
    ~llama_lora_adapter() = default;

    llama_lora_weight * get_weight(struct ggml_tensor * w);
};
