#pragma once

#ifndef __cplusplus
#error "This header is for C++ only"
#endif

#include <memory>

#include "llama.h"

struct llama_model_deleter {
    void operator()(llama_model * model) { llama_model_free(model); }
};

struct llama_context_deleter {
    void operator()(llama_context * context) { llama_free(context); }
};

struct llama_sampler_deleter {
    void operator()(llama_sampler * sampler) { llama_sampler_free(sampler); }
};

struct llama_adapter_lora_deleter {
    void operator()(llama_adapter_lora *) {
        // llama_adapter_lora_free is deprecated
    }
};

typedef std::unique_ptr<llama_model, llama_model_deleter> llama_model_ptr;
typedef std::unique_ptr<llama_context, llama_context_deleter> llama_context_ptr;
typedef std::unique_ptr<llama_sampler, llama_sampler_deleter> llama_sampler_ptr;
typedef std::unique_ptr<llama_adapter_lora, llama_adapter_lora_deleter> llama_adapter_lora_ptr;
