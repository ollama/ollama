#pragma once

#include <cstdint>

struct llama_context;
struct ggml_tensor;
struct llama_ubatch;

// Bind the context whose segments apply to KQ masks filled on this thread.
// llama_decode calls these around the graph build. A null bind is a no-op.
void llama_pointer_bind(struct llama_context * ctx);
void llama_pointer_unbind(void);

// Overwrite a host KQ mask with the pointer-head pattern when the bound
// context has segments and the batch is one stream of that length.
// Otherwise leave the causal fill untouched.
void llama_pointer_apply_kq_mask(struct ggml_tensor * dst, const struct llama_ubatch * ubatch);

extern "C" {

// Install segment ids for this context. n == 0 clears them.
// The next llama_decode of exactly n tokens, on an empty cache and one
// sequence, writes this mask into the attention graph. Other decodes are
// unchanged. This does not load weights and does not letter-score.
void llama_set_pointer_segments(struct llama_context * ctx, const int32_t * seg, int32_t n, bool state_bidir);

// 0 if the last apply wrote the pointer mask, 1 if it was not applied.
int32_t llama_pointer_mask_status(void);

}
