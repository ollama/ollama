#include <stdlib.h>
#include <string>
#include <vector>
#include <cstdint>
#include <stdexcept>

#include "llama-sampling.h"
#include "llama-grammar.h"
#include "llama-vocab.h"
#include "grammar_ext.h"

extern "C" {

struct llama_grammar* grammar_create_from_string(const struct llama_vocab* vocab, const char* grammar_str, const char* grammar_root) {
    try {
        // Initialize grammar sampler directly with the model
        struct llama_sampler* sampler = llama_sampler_init_grammar(vocab, grammar_str, grammar_root);
        if (!sampler) {
            return nullptr;
        }
        
        // Cast the sampler to a grammar and return it
        return (struct llama_grammar*)sampler;
    } catch (const std::exception &err) {
        return nullptr;
    }
}

void grammar_apply_to_logits(struct llama_grammar* grammar, float* logits, int n_logits) {
    if (!grammar || !logits || n_logits <= 0) {
        return;
    }

    // Create token data array for the grammar application
    llama_token_data* token_data = (llama_token_data*)malloc(n_logits * sizeof(llama_token_data));
    if (!token_data) {
        return;
    }

    // Initialize token data from logits
    for (int i = 0; i < n_logits; i++) {
        token_data[i].id = i;
        token_data[i].logit = logits[i];
        token_data[i].p = 0.0f;
    }

    // Create token data array structure
    llama_token_data_array arr = {
        .data = token_data,
        .size = (size_t)n_logits,
        .sorted = false,
        .selected = -1
    };

    // Apply grammar constraints to the token data array
    llama_grammar_apply_impl(*grammar, &arr);

    // Copy back the modified logits
    for (int i = 0; i < n_logits; i++) {
        logits[i] = token_data[i].logit;
    }

    free(token_data);
}

void grammar_free(struct llama_grammar* grammar) {
    if (grammar) {
        // Free the grammar as a sampler
        llama_sampler_free((struct llama_sampler*)grammar);
    }
}

struct llama_vocab* vocab_bridge_from_tokens(const char** tokens, int n_tokens) {
    // Call the C++ function from llama-vocab.cpp
    return llama_vocab_from_tokens(tokens, n_tokens);
}

void vocab_bridge_free(struct llama_vocab* vocab) {
    // Call the C++ function from llama-vocab.cpp
    llama_vocab_free(vocab);
}

} // extern "C"