#ifndef GRAMMAR_EXT_H
#define GRAMMAR_EXT_H

#include "llama.h"

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations
struct llama_grammar;
struct llama_vocab;

// Create a new grammar from a string (returns a grammar implemented as a sampler)
struct llama_grammar* grammar_create_from_string(const struct llama_vocab* vocab, const char* grammar_str, const char* grammar_root);

// Apply grammar constraints to logits
void grammar_apply_to_logits(struct llama_grammar* grammar, float* logits, int n_logits);

// Free grammar resources (frees the underlying sampler)
void grammar_free(struct llama_grammar* grammar);

// C wrapper for llama_vocab_from_tokens
struct llama_vocab* vocab_bridge_from_tokens(const char** tokens, int n_tokens);

// C wrapper for llama_vocab_free
void vocab_bridge_free(struct llama_vocab* vocab);

#ifdef __cplusplus
}
#endif

#endif // GRAMMAR_EXT_H