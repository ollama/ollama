#pragma once

#include "llama.h"

#include <cstdint>
#include <vector>

enum common_reasoning_budget_state {
    REASONING_BUDGET_IDLE,         // waiting for start sequence
    REASONING_BUDGET_COUNTING,     // counting down tokens
    REASONING_BUDGET_FORCING,      // forcing budget message + end sequence
    REASONING_BUDGET_WAITING_UTF8, // budget exhausted, waiting for UTF-8 completion
    REASONING_BUDGET_DONE,         // passthrough forever
};

// Creates a reasoning budget sampler that limits token generation inside a
// reasoning block (e.g. between <think> and </think>).
//
// State machine: IDLE -> COUNTING -> WAITING_UTF8 -> FORCING -> DONE
//   IDLE:         passthrough, watching for start_tokens sequence
//   COUNTING:     counting down remaining tokens, watching for natural end_tokens
//   WAITING_UTF8: budget exhausted, allowing tokens to complete a UTF-8 sequence
//   FORCING:      forces forced_tokens token-by-token (all other logits -> -inf)
//   DONE:         passthrough forever
//
// Parameters:
//   vocab         - vocabulary (used for UTF-8 boundary detection; can be nullptr)
//   start_tokens  - token sequence that activates counting
//   end_tokens    - token sequence for natural deactivation
//   forced_tokens - token sequence forced when budget expires
//   budget        - max tokens allowed in the reasoning block
//   initial_state - initial state of the sampler (e.g. IDLE or COUNTING)
//                   note: COUNTING with budget <= 0 is promoted to FORCING
//
struct llama_sampler * common_reasoning_budget_init(
        const struct llama_vocab       * vocab,
        const std::vector<llama_token> & start_tokens,
        const std::vector<llama_token> & end_tokens,
        const std::vector<llama_token> & forced_tokens,
        int32_t                          budget,
        common_reasoning_budget_state    initial_state);
