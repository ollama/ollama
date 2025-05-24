#pragma once

#include "llama.h"
#include "llama-arch.h"

#include <vector>

struct llama_model_saver {
    struct gguf_context * gguf_ctx = nullptr;
    const struct llama_model & model;
    const struct LLM_KV llm_kv;

    llama_model_saver(const struct llama_model & model);
    ~llama_model_saver();

    void add_kv(enum llm_kv key, uint32_t     value);
    void add_kv(enum llm_kv key, int32_t      value);
    void add_kv(enum llm_kv key, float        value);
    void add_kv(enum llm_kv key, bool         value);
    void add_kv(enum llm_kv key, const char * value);

    [[noreturn]]
    void add_kv(enum llm_kv key, char value); // needed to make the template below compile

    template <typename Container>
    void add_kv(enum llm_kv key, const Container & value, bool per_layer = false);

    void add_kv(enum llm_kv key, const std::vector<std::string> & value);

    void add_tensor(const struct ggml_tensor * tensor);

    void add_kv_from_model();

    void add_tensors_from_model();

    void save(const std::string & path_model);
};
