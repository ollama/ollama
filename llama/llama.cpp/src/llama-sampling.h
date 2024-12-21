#pragma once

// TODO: rename llama-sampling.h/.cpp to llama-sampler.h/.cpp ?

#include "llama-grammar.h"

struct llama_vocab;
struct llama_grammar;

// sampler chain

struct llama_sampler_chain {
    llama_sampler_chain_params params;

    std::vector<struct llama_sampler *> samplers;

    // timing

    mutable int64_t t_sample_us;

    mutable int32_t n_sample;
};

struct llama_sampler * llama_sampler_init_grammar_impl(
        const struct llama_vocab & vocab,
                      const char * grammar_str,
                      const char * grammar_root);

struct llama_sampler * llama_sampler_init_infill_impl(
        const struct llama_vocab & vocab);

struct llama_sampler * llama_sampler_init_dry_impl(
        const struct llama_vocab &  vocab,
                         int32_t    context_size,
                           float    dry_multiplier,
                           float    dry_base,
                         int32_t    dry_allowed_length,
                         int32_t    dry_penalty_last_n,
                      const char ** seq_breakers,
                          size_t    num_breakers);

struct llama_sampler * llama_sampler_init_dry_testing(
                         int32_t   context_size,
                           float   dry_multiplier,
                           float   dry_base,
                         int32_t   dry_allowed_length,
                         int32_t   dry_penalty_last_n,
  const std::vector<std::vector<llama_token>>& seq_breakers);
