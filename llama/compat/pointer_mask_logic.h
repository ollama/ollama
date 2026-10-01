#pragma once

#include <cstdint>

// Pointer-head attention allow-rule. This is the mask llama.cpp writes into the
// KQ tensor before the Qwen3 attention graph runs. It is not a causal mask and
// not a letter-token score.
//
// seg[i] == 0 is the state. Those tokens attend to every state token, both
// directions, and never to a branch. seg[i] > 0 is question i. A branch token
// attends to the state and to earlier tokens of its own question only.
// seg[i] < 0 is padding. The diagonal is always allowed so a row is not empty.
//
// Token order is the sequence index, not the RoPE position. Branch positions
// restart after the state; the mask must not use those positions.
inline bool llama_pointer_allow(const int32_t * seg, int32_t n, int32_t i, int32_t j, bool state_bidir) {
    if (i == j) {
        return true;
    }
    if (i < 0 || j < 0 || i >= n || j >= n || seg[j] < 0) {
        return false;
    }
    const bool causal = j <= i;
    const bool same = seg[j] == 0 || seg[j] == seg[i];
    if (causal && same) {
        return true;
    }
    return state_bidir && seg[i] == 0 && seg[j] == 0;
}
