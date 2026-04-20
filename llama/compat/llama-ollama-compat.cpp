#include "llama-ollama-compat.h"
#include "llama-ollama-compat-util.h"

#include "llama-impl.h"

#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace llama_ollama_compat {

using namespace llama_ollama_compat::detail; // pull detail:: helpers into scope

namespace {

// Per-loader file path registry — set by translate_metadata, read by
// maybe_load_text_tensor so it can pass the path to load ops without a
// separate patch insertion in the model loader's load_all_data path.
std::mutex g_loader_path_mutex;
std::unordered_map<const llama_model_loader *, std::string> g_loader_paths;

// =========================================================================
// gemma3 (text side)
// =========================================================================

// An Ollama-format gemma3 file declares arch="gemma3" AND exhibits at
// least one converter quirk. Different converter versions produced
// different quirks (4B/12B/27B have embedded vision + mm KVs; 1B uses
// non-standard rope key names; all of them omit layer_norm_rms_epsilon).
bool detect_ollama_gemma3(const gguf_context * meta, const ggml_context * ctx) {
    const int64_t arch_kid = gguf_find_key(meta, "general.architecture");
    if (arch_kid < 0) return false;
    if (std::strcmp(gguf_get_val_str(meta, arch_kid), "gemma3") != 0) return false;

    return has_key(meta, "gemma3.mm.tokens_per_image")
        || any_tensor_with_prefix(ctx, "v.")
        || any_tensor_with_prefix(ctx, "mm.")
        || has_key(meta, "gemma3.rope.global.freq_base")
        || has_key(meta, "gemma3.rope.local.freq_base")
        || has_key(meta, "tokenizer.ggml.add_padding_token")
        || has_key(meta, "tokenizer.ggml.add_unknown_token")
        || !has_key(meta, "gemma3.attention.layer_norm_rms_epsilon");
}

void handle_gemma3(const llama_model_loader * ml, gguf_context * meta, ggml_context * ctx) {
    if (!detect_ollama_gemma3(meta, ctx)) return;

    LLAMA_LOG_INFO("%s: detected Ollama-format gemma3 GGUF; applying compatibility fixes\n", __func__);

    // Old Ollama converters sometimes used nested rope key names. Copy
    // them to the flat names upstream expects BEFORE injecting defaults.
    copy_f32_kv(meta, "gemma3.rope.global.freq_base", "gemma3.rope.freq_base");
    copy_f32_kv(meta, "gemma3.rope.local.freq_base",  "gemma3.rope.freq_base_swa");

    // Inject required KVs with their standard gemma3 defaults.
    inject_f32_if_missing(meta, "gemma3.attention.layer_norm_rms_epsilon", 1e-6f);
    inject_f32_if_missing(meta, "gemma3.rope.freq_base",                   1000000.0f);
    inject_f32_if_missing(meta, "gemma3.rope.freq_base_swa",               10000.0f);

    // Gemma3 4B/12B/27B ship with {type: "linear", factor: 8.0} rope scaling
    // in their HF config to extend the 16k trained context to 131072. Ollama's
    // old converter didn't write these. The 1B has no scaling — detect by
    // context length.
    const int64_t ctx_key = gguf_find_key(meta, "gemma3.context_length");
    if (ctx_key >= 0 && gguf_get_val_u32(meta, ctx_key) >= 131072) {
        inject_str_if_missing(meta, "gemma3.rope.scaling.type",   "linear");
        inject_f32_if_missing(meta, "gemma3.rope.scaling.factor", 8.0f);
    }

    // Tokenizer vocab size vs embedding rows mismatch: Ollama leaves extra
    // multimodal tokens (e.g. <image_soft_token>) in the tokenizer arrays.
    // Truncate to match token_embd rows so llama.cpp's dim check passes.
    for (ggml_tensor * t = ggml_get_first_tensor(ctx); t; t = ggml_get_next_tensor(ctx, t)) {
        if (std::strcmp(ggml_get_name(t), "token_embd.weight") == 0) {
            const size_t rows = t->ne[1]; // shape is [n_embd, n_vocab]
            truncate_str_arr (meta, "tokenizer.ggml.tokens",     rows);
            truncate_data_arr(meta, "tokenizer.ggml.scores",     GGUF_TYPE_FLOAT32, sizeof(float),   rows);
            truncate_data_arr(meta, "tokenizer.ggml.token_type", GGUF_TYPE_INT32,   sizeof(int32_t), rows);
            break;
        }
    }

    // Hide embedded vision tensors from the text loader. Ollama's Go side
    // re-passes the same blob as --mmproj so the clip loader picks them up.
    add_skip_prefix(ml, "v.");
    add_skip_prefix(ml, "mm.");

    // Note: no RMSNorm weight shift needed. Ollama's published gemma3 blobs
    // already have the +1 shift baked in, same as upstream's convert_hf.
}

// =========================================================================
// embeddinggemma (text side — sentence-transformer dense projection)
// =========================================================================
//
// Ollama publishes embeddinggemma:300m with general.architecture=gemma3 and
// two extra dense layers stored as `dense.0.weight` / `dense.1.weight`
// (the sentence-transformers post-pooling projection that maps the 768-dim
// pooled embedding through 768→3072→768 for the matryoshka head).
//
// Upstream loads this model under arch=gemma-embedding, which:
//   * disables causal attention (embeddings are bidirectional)
//   * loads `dense_2.weight` and `dense_3.weight` by name (with shapes
//     derived from gemma-embedding.dense_2_feat_in/out etc.)
//
// Without that arch, the gemma3 loader leaves dense.0/dense.1 unrequested
// and `done_getting_tensors` raises "wrong number of tensors" (2 unused).
//
// Detection: arch=gemma3 AND has dense.0.weight tensor (only embeddinggemma
// ships these — regular gemma3 chat models do not).
// Translation: switch arch_name to gemma-embedding, copy the gemma3.* KV
// prefix to gemma-embedding.*, derive dense_*_feat_* from the actual tensor
// shapes, and rename dense.0/dense.1 → dense_2/dense_3.

bool detect_ollama_embeddinggemma(const gguf_context * meta, const ggml_context * ctx) {
    const int64_t arch_kid = gguf_find_key(meta, "general.architecture");
    if (arch_kid < 0) return false;
    if (std::strcmp(gguf_get_val_str(meta, arch_kid), "gemma3") != 0) return false;
    return ggml_get_tensor(const_cast<ggml_context *>(ctx), "dense.0.weight") != nullptr;
}

void handle_embeddinggemma(const llama_model_loader * ml, gguf_context * meta,
                           ggml_context * ctx, std::string & arch_name) {
    (void) ml;
    if (!detect_ollama_embeddinggemma(meta, ctx)) return;

    LLAMA_LOG_INFO("%s: detected Ollama-format embeddinggemma; translating to gemma-embedding\n", __func__);

    // Switch architecture so upstream loads the embedding-specific code path
    // (no causal attention, dense_2/dense_3 loaded by name).
    arch_name = "gemma-embedding";
    gguf_set_val_str(meta, "general.architecture", "gemma-embedding");

    // Mirror gemma3.* hparams under the new arch prefix. rename_kv_prefix
    // copies (does not remove); the leftover gemma3.* keys are unused.
    rename_kv_prefix(meta, "gemma3.", "gemma-embedding.");

    // Derive dense feat dims from the actual tensor shapes.
    //   dense.0.weight: [n_embd, dense_2_feat_out]
    //   dense.1.weight: [dense_3_feat_in, n_embd]
    ggml_tensor * d0 = ggml_get_tensor(ctx, "dense.0.weight");
    ggml_tensor * d1 = ggml_get_tensor(ctx, "dense.1.weight");
    if (d0 && d1) {
        gguf_set_val_u32(meta, "gemma-embedding.dense_2_feat_in",  (uint32_t) d0->ne[0]);
        gguf_set_val_u32(meta, "gemma-embedding.dense_2_feat_out", (uint32_t) d0->ne[1]);
        gguf_set_val_u32(meta, "gemma-embedding.dense_3_feat_in",  (uint32_t) d1->ne[0]);
        gguf_set_val_u32(meta, "gemma-embedding.dense_3_feat_out", (uint32_t) d1->ne[1]);
    }

    rename_tensor(meta, ctx, "dense.0.weight", "dense_2.weight");
    rename_tensor(meta, ctx, "dense.1.weight", "dense_3.weight");
}

// =========================================================================
// qwen35moe (text side)
// =========================================================================

// Shared text-side fixes for Ollama-format qwen35 / qwen35moe GGUFs.
// Both arches use the same SSM-hybrid + M-RoPE + MTP+vision-monolithic
// converter quirks; only the arch name (and KV prefix) differs.
void apply_qwen35_text_fixes(const llama_model_loader * ml, gguf_context * meta,
                             ggml_context * ctx, const char * arch_prefix) {
    auto kv = [arch_prefix](const char * suffix) {
        return std::string(arch_prefix) + suffix;
    };

    // 1. attention.head_count_kv — upstream expects UINT32; Ollama wrote
    //    an array (one entry per layer, 0 for SSM layers, 2/4 for attention).
    //    Collapse to the max non-zero value.
    {
        const std::string key = kv(".attention.head_count_kv");
        const int64_t kid = gguf_find_key(meta, key.c_str());
        if (kid >= 0 && gguf_get_kv_type(meta, kid) == GGUF_TYPE_ARRAY) {
            const size_t n = gguf_get_arr_n(meta, kid);
            const auto * arr = static_cast<const uint32_t *>(gguf_get_arr_data(meta, kid));
            uint32_t max_kv = 0;
            for (size_t i = 0; i < n; ++i) if (arr[i] > max_kv) max_kv = arr[i];
            if (max_kv == 0) max_kv = 2; // safety fallback
            gguf_remove_key  (meta, key.c_str());
            gguf_set_val_u32 (meta, key.c_str(), max_kv);
        }
    }

    // 2. rope.dimension_sections — upstream expects a 4-element array
    //    (M-RoPE convention); Ollama wrote 3 elements. Pad with a trailing 0.
    {
        const std::string key = kv(".rope.dimension_sections");
        const int64_t kid = gguf_find_key(meta, key.c_str());
        if (kid >= 0 && gguf_get_arr_n(meta, kid) == 3) {
            const auto * src = static_cast<const int32_t *>(gguf_get_arr_data(meta, kid));
            const int32_t padded[4] = { src[0], src[1], src[2], 0 };
            gguf_set_arr_data(meta, key.c_str(), GGUF_TYPE_INT32, padded, 4);
        }
    }

    // 3. Tensor rename: Ollama's `blk.N.ssm_dt` is upstream's
    //    `blk.N.ssm_dt.bias` (same shape).
    {
        std::vector<std::string> targets;
        const int64_t n = gguf_get_n_tensors(meta);
        static const char suffix[] = ".ssm_dt";
        const size_t slen = sizeof(suffix) - 1;
        for (int64_t i = 0; i < n; ++i) {
            std::string name(gguf_get_tensor_name(meta, i));
            if (name.size() >= slen
                    && name.compare(name.size() - slen, slen, suffix) == 0) {
                targets.push_back(std::move(name));
            }
        }
        for (const auto & from : targets) {
            rename_tensor(meta, ctx, from.c_str(), (from + ".bias").c_str());
        }
    }

    // 4. Drop embedded vision + MTP + projector tensors from the text loader.
    add_skip_prefix(ml, "v.");
    add_skip_prefix(ml, "mm.");
    add_skip_prefix(ml, "mtp.");
}

bool detect_ollama_qwen35moe(const gguf_context * meta, const ggml_context * ctx) {
    const int64_t arch_kid = gguf_find_key(meta, "general.architecture");
    if (arch_kid < 0) return false;
    if (std::strcmp(gguf_get_val_str(meta, arch_kid), "qwen35moe") != 0) return false;

    // Any Ollama-ism. Upstream qwen35moe files have none of these — the
    // vision KVs live in a separate mmproj, MTP tensors are dropped,
    // head_count_kv is a scalar, and the extra rope / ssm / feed_forward
    // KVs are either absent or stored differently.
    return has_key(meta, "qwen35moe.vision.block_count")
        || has_key(meta, "qwen35moe.image_token_id")
        || has_key(meta, "qwen35moe.ssm.v_head_reordered")
        || has_key(meta, "qwen35moe.feed_forward_length")
        || has_key(meta, "qwen35moe.rope.mrope_interleaved")
        || any_tensor_with_prefix(ctx, "mtp.")
        || any_tensor_with_prefix(ctx, "v.");
}

void handle_qwen35moe(const llama_model_loader * ml, gguf_context * meta, ggml_context * ctx) {
    if (!detect_ollama_qwen35moe(meta, ctx)) return;
    LLAMA_LOG_INFO("%s: detected Ollama-format qwen35moe GGUF; applying compatibility fixes\n", __func__);
    apply_qwen35_text_fixes(ml, meta, ctx, "qwen35moe");
}

// =========================================================================
// qwen35 (text side — non-MoE, e.g. qwen3.5:9b)
// =========================================================================
//
// Same converter quirks as qwen35moe but the arch name has no "moe" suffix.
// All the SSM-hybrid / M-RoPE / MTP / monolithic-vision fix-ups apply.

bool detect_ollama_qwen35(const gguf_context * meta, const ggml_context * ctx) {
    const int64_t arch_kid = gguf_find_key(meta, "general.architecture");
    if (arch_kid < 0) return false;
    if (std::strcmp(gguf_get_val_str(meta, arch_kid), "qwen35") != 0) return false;
    return has_key(meta, "qwen35.vision.block_count")
        || has_key(meta, "qwen35.image_token_id")
        || has_key(meta, "qwen35.ssm.v_head_reordered")
        || has_key(meta, "qwen35.rope.mrope_interleaved")
        || any_tensor_with_prefix(ctx, "mtp.")
        || any_tensor_with_prefix(ctx, "v.");
}

void handle_qwen35(const llama_model_loader * ml, gguf_context * meta, ggml_context * ctx) {
    if (!detect_ollama_qwen35(meta, ctx)) return;
    LLAMA_LOG_INFO("%s: detected Ollama-format qwen35 GGUF; applying compatibility fixes\n", __func__);
    apply_qwen35_text_fixes(ml, meta, ctx, "qwen35");
}

// =========================================================================
// gemma4 (text side)
// =========================================================================
//
// Same arch name on both sides. Ollama publishes a monolithic GGUF that
// embeds the vision encoder + audio encoder + projector inline. Text-side
// KVs/tensor names match upstream verbatim — only fix is to hide the
// `a.*` / `v.*` / `mm.*` tensors from the text loader so n_tensors lines up.

bool detect_ollama_gemma4(const gguf_context * meta, const ggml_context * ctx) {
    const int64_t arch_kid = gguf_find_key(meta, "general.architecture");
    if (arch_kid < 0) return false;
    if (std::strcmp(gguf_get_val_str(meta, arch_kid), "gemma4") != 0) return false;
    return any_tensor_with_prefix(ctx, "a.")
        || any_tensor_with_prefix(ctx, "v.")
        || any_tensor_with_prefix(ctx, "mm.");
}

void handle_gemma4(const llama_model_loader * ml, gguf_context * meta, ggml_context * ctx) {
    if (!detect_ollama_gemma4(meta, ctx)) return;
    (void) ctx;

    LLAMA_LOG_INFO("%s: detected Ollama-format gemma4 GGUF; applying compatibility fixes\n", __func__);

    // Tokenizer fix: Ollama writes `tokenizer.ggml.model = 'llama'` (SPM) on
    // gemma4 GGUFs, but gemma4 actually uses BPE — upstream-converted GGUFs
    // use `'gemma4'` which selects LLAMA_VOCAB_TYPE_BPE in src/llama-vocab.cpp.
    // With the wrong tokenizer type, gemma4's special tokens (e.g.
    // `<|thought|>`, `<|turn>`, `<|channel>`) get split into multiple SPM
    // subword pieces, so when the model emits them they come out as raw
    // text instead of being recognized as control tokens.
    //
    // Ollama already supplies `tokenizer.ggml.merges` (needed for BPE) and
    // `tokenizer.ggml.pre = 'gemma4'`, so flipping the model name is enough.
    {
        const int64_t kid = gguf_find_key(meta, "tokenizer.ggml.model");
        if (kid >= 0) {
            const char * cur = gguf_get_val_str(meta, kid);
            if (cur && std::strcmp(cur, "llama") == 0) {
                gguf_set_val_str(meta, "tokenizer.ggml.model", "gemma4");
            }
        }
    }

    // Hide embedded audio + vision + projector tensors from the text loader.
    add_skip_prefix(ml, "a.");
    add_skip_prefix(ml, "v.");
    add_skip_prefix(ml, "mm.");
}

// =========================================================================
// deepseek-ocr (text side)
// =========================================================================
//
// Ollama uses arch name "deepseekocr" / KV prefix "deepseekocr.*".
// Upstream uses "deepseek2-ocr" (with hyphen) / "deepseek2-ocr.*".
//
// Aside from the prefix rename:
//   * Inject `expert_feed_forward_length` from the per-expert ffn_down_exps
//     shape (Ollama omits it; the value is the inner FFN dim of one expert,
//     896 for the 3B model).
//   * Inject `expert_shared_count` from the ffn_down_shexp shape (Ollama
//     omits it; the shared experts share their FFN dim with regular experts,
//     so count = shexp_dim / expert_feed_forward_length).
//   * Skip embedded vision (`v.*`), projector (`mm.*`), and the SAM encoder
//     (`s.*`) tensors from the text loader.

bool detect_ollama_deepseekocr(const gguf_context * meta) {
    const int64_t arch_kid = gguf_find_key(meta, "general.architecture");
    if (arch_kid < 0) return false;
    return std::strcmp(gguf_get_val_str(meta, arch_kid), "deepseekocr") == 0;
}

void handle_deepseekocr(const llama_model_loader * ml, gguf_context * meta,
                        ggml_context * ctx, std::string & arch_name) {
    if (!detect_ollama_deepseekocr(meta)) return;

    LLAMA_LOG_INFO("%s: detected Ollama-format deepseekocr GGUF; applying compatibility fixes\n", __func__);

    gguf_set_val_str(meta, "general.architecture", "deepseek2-ocr");
    rename_kv_prefix(meta, "deepseekocr.", "deepseek2-ocr.");
    arch_name = "deepseek2-ocr";

    // Inject defaults Ollama omitted entirely.
    inject_f32_if_missing(meta, "deepseek2-ocr.attention.layer_norm_rms_epsilon",
                          1e-6f);

    // Recover expert_feed_forward_length from blk.1 (first MoE block; blk.0
    // is dense). ne[0] of ffn_down_exps is the per-expert inner dim.
    if (!has_key(meta, "deepseek2-ocr.expert_feed_forward_length")) {
        if (ggml_tensor * t = ggml_get_tensor(ctx, "blk.1.ffn_down_exps.weight")) {
            gguf_set_val_u32(meta, "deepseek2-ocr.expert_feed_forward_length",
                             (uint32_t) t->ne[0]);
        }
    }

    // Recover expert_shared_count from blk.1.ffn_down_shexp shape.
    // shape ne[0] = expert_shared_count * expert_feed_forward_length
    if (!has_key(meta, "deepseek2-ocr.expert_shared_count")) {
        ggml_tensor * shexp = ggml_get_tensor(ctx, "blk.1.ffn_down_shexp.weight");
        const int64_t fflen_kid = gguf_find_key(meta, "deepseek2-ocr.expert_feed_forward_length");
        if (shexp && fflen_kid >= 0) {
            const uint32_t fflen = gguf_get_val_u32(meta, fflen_kid);
            if (fflen > 0) {
                gguf_set_val_u32(meta, "deepseek2-ocr.expert_shared_count",
                                 (uint32_t)(shexp->ne[0] / fflen));
            }
        }
    }

    // Hide embedded SAM (`s.*`), vision (`v.*`), and projector (`mm.*`)
    // tensors from the text loader.
    add_skip_prefix(ml, "s.");
    add_skip_prefix(ml, "v.");
    add_skip_prefix(ml, "mm.");
}

// =========================================================================
// nemotron_h_moe (text only)
// =========================================================================
//
// Same arch name on both sides. Most variants (e.g. nemotron-cascade-2)
// load as-is. The latent-FFN variants (e.g. nemotron-3-super 120B-A12B)
// rename `ffn_latent_in` / `ffn_latent_out` to `ffn_latent_down` /
// `ffn_latent_up`, and need `moe_latent_size` injected (derived from
// the latent tensor shape).

bool detect_ollama_nemotron_h_moe(const gguf_context * meta, const ggml_context * ctx) {
    const int64_t arch_kid = gguf_find_key(meta, "general.architecture");
    if (arch_kid < 0) return false;
    if (std::strcmp(gguf_get_val_str(meta, arch_kid), "nemotron_h_moe") != 0) return false;
    return any_tensor_with_prefix(ctx, "blk.1.ffn_latent_in")
        || any_tensor_with_prefix(ctx, "blk.0.ffn_latent_in")
        || any_tensor_with_prefix(ctx, "mtp.");
}

void handle_nemotron_h_moe(const llama_model_loader * ml, gguf_context * meta, ggml_context * ctx) {
    if (!detect_ollama_nemotron_h_moe(meta, ctx)) return;

    LLAMA_LOG_INFO("%s: detected Ollama-format nemotron_h_moe GGUF; applying compatibility fixes\n", __func__);

    // Inject moe_latent_size for latent-FFN variants (e.g. super 120B-A12B).
    // Standard variants (e.g. cascade-2 30B-A3B) have no latent tensors and
    // use n_embd as the MoE inner dim — leave the key absent.
    if (!has_key(meta, "nemotron_h_moe.moe_latent_size")) {
        for (uint32_t b = 0; b < 1024; ++b) {
            char name[64];
            std::snprintf(name, sizeof(name), "blk.%u.ffn_latent_in.weight", b);
            if (ggml_tensor * t = ggml_get_tensor(ctx, name)) {
                gguf_set_val_u32(meta, "nemotron_h_moe.moe_latent_size",
                                 (uint32_t) t->ne[1]);
                break;
            }
        }
    }

    // Rename the latent projection tensors to upstream's naming (no-op when
    // the file has no latent tensors).
    rename_tensors_containing(meta, ctx, ".ffn_latent_in",  ".ffn_latent_down");
    rename_tensors_containing(meta, ctx, ".ffn_latent_out", ".ffn_latent_up");

    // Drop MTP (Multi-Token Prediction) tensors — Ollama's converter emits
    // them as one-tensor-per-expert (`mtp.layers.X.mixer.experts.Y.{up,down}_proj`)
    // which upstream's nemotron_h_moe loader doesn't claim. Total: ~1040 extra
    // tensors on super 120B.
    add_skip_prefix(ml, "mtp.");
}

// =========================================================================
// llama4 (text side)
// =========================================================================
//
// Same arch name on both sides. Ollama publishes a monolithic GGUF that
// embeds the vision encoder + projector inline. Text-side KVs/tensor
// names match upstream verbatim — only fix is to hide `v.*`/`mm.*` from
// the text loader so n_tensors lines up.

bool detect_ollama_llama4(const gguf_context * meta, const ggml_context * ctx) {
    const int64_t arch_kid = gguf_find_key(meta, "general.architecture");
    if (arch_kid < 0) return false;
    if (std::strcmp(gguf_get_val_str(meta, arch_kid), "llama4") != 0) return false;
    return any_tensor_with_prefix(ctx, "v.")
        || any_tensor_with_prefix(ctx, "mm.");
}

void handle_llama4(const llama_model_loader * ml, gguf_context * meta, ggml_context * ctx) {
    if (!detect_ollama_llama4(meta, ctx)) return;
    (void) meta;
    (void) ctx;

    LLAMA_LOG_INFO("%s: detected Ollama-format llama4 GGUF; applying compatibility fixes\n", __func__);

    add_skip_prefix(ml, "v.");
    add_skip_prefix(ml, "mm.");
}

// =========================================================================
// glm-ocr (text side)
// =========================================================================
//
// Ollama uses arch name "glmocr" / KV prefix "glmocr.*" with 16 blocks.
// Upstream uses "glm4" / "glm4.*" — the GLM-OCR variant of LLM_ARCH_GLM4
// is identified by `n_layer = 17` (16 main + 1 nextn predict layer).
// Ollama drops the nextn layer entirely, so we report n_layer = 16 and
// leave `nextn_predict_layers` absent (defaults to 0 = no nextn path).
//
// Bigger surgery: GLM4 expects fused gate+up MLP weights stored at
// `blk.X.ffn_up.weight` with shape `[n_embd, n_ff*2]`. Ollama writes
// the gate and up halves as separate `ffn_gate.weight` / `ffn_up.weight`
// tensors (each `[n_embd, n_ff]`). We register a concat load op that
// reads gate+up bytes and stitches them into the fused upstream slot.

// Per-block: register a concat load that fuses Ollama's separate
// ffn_gate + ffn_up into upstream's single `blk.X.ffn_up.weight`
// tensor with doubled out dim. Capture source file offsets BEFORE any
// renames invalidate them (same pattern as qwen35moe QKV merge).
void register_glm4_ffn_concat(gguf_context * meta, ggml_context * ctx, int block_idx) {
    char gate_n[64], up_n[64];
    std::snprintf(gate_n, sizeof(gate_n), "blk.%d.ffn_gate.weight", block_idx);
    std::snprintf(up_n,   sizeof(up_n),   "blk.%d.ffn_up.weight",   block_idx);

    if (!ggml_get_tensor(ctx, gate_n) || !ggml_get_tensor(ctx, up_n)) return;

    // GLM4's fused ffn_up has gate as first half, up as second half
    // (so ggml_swiglu's silu(first_half) * second_half gives silu(gate) * up).
    register_concat_load(meta, up_n, {gate_n, up_n});

    if (ggml_tensor * t = ggml_get_tensor(ctx, up_n)) {
        set_tensor_shape(t, {t->ne[0], t->ne[1] * 2});
    }
}

bool detect_ollama_glmocr(const gguf_context * meta) {
    const int64_t arch_kid = gguf_find_key(meta, "general.architecture");
    if (arch_kid < 0) return false;
    return std::strcmp(gguf_get_val_str(meta, arch_kid), "glmocr") == 0;
}

void handle_glmocr(const llama_model_loader * ml, gguf_context * meta,
                   ggml_context * ctx, std::string & arch_name) {
    if (!detect_ollama_glmocr(meta)) return;

    LLAMA_LOG_INFO("%s: detected Ollama-format glmocr GGUF; applying compatibility fixes\n", __func__);

    gguf_set_val_str(meta, "general.architecture", "glm4");
    rename_kv_prefix(meta, "glmocr.", "glm4.");
    arch_name = "glm4";

    // M-RoPE: Ollama writes a 3-element `rope.mrope_section`, upstream expects
    // a 4-element `rope.dimension_sections` (pad trailing 0).
    {
        const int64_t kid = gguf_find_key(meta, "glm4.rope.mrope_section");
        if (kid >= 0 && gguf_get_arr_n(meta, kid) == 3) {
            const auto * src = static_cast<const int32_t *>(gguf_get_arr_data(meta, kid));
            const int32_t padded[4] = { src[0], src[1], src[2], 0 };
            gguf_set_arr_data(meta, "glm4.rope.dimension_sections",
                              GGUF_TYPE_INT32, padded, 4);
        }
    }
    // Inject `rope.dimension_count` from key_length (used as the rope dim).
    if (!has_key(meta, "glm4.rope.dimension_count")) {
        const int64_t kid = gguf_find_key(meta, "glm4.attention.key_length");
        if (kid >= 0) {
            gguf_set_val_u32(meta, "glm4.rope.dimension_count",
                             gguf_get_val_u32(meta, kid));
        }
    }

    // Tokenizer pre-tokenizer: Ollama wrote `llama-bpe`, but glm-ocr uses
    // `chatglm-bpe` (different regex split — wrong pre-tokenization can
    // fragment GLM's special tokens).
    {
        const int64_t kid = gguf_find_key(meta, "tokenizer.ggml.pre");
        if (kid >= 0) {
            const char * cur = gguf_get_val_str(meta, kid);
            if (cur && std::strcmp(cur, "chatglm-bpe") != 0) {
                gguf_set_val_str(meta, "tokenizer.ggml.pre", "chatglm-bpe");
            }
        }
    }

    // Tensor renames (substring): each leaf appears once per block and
    // doesn't overlap the others.
    rename_tensors_containing(meta, ctx, ".attn_out",       ".attn_output");
    rename_tensors_containing(meta, ctx, ".post_attn_norm", ".post_attention_norm");
    rename_tensors_containing(meta, ctx, ".post_ffn_norm",  ".post_ffw_norm");

    // Fuse ffn_gate + ffn_up → ffn_up[:, 2*n_ff] for every block, then mark
    // the orphan ffn_gate tensors as skip so n_tensors lines up.
    {
        const int64_t n_blk_kid = gguf_find_key(meta, "glm4.block_count");
        const uint32_t n_blocks = n_blk_kid >= 0 ? gguf_get_val_u32(meta, n_blk_kid) : 16;
        for (uint32_t b = 0; b < n_blocks; ++b) {
            register_glm4_ffn_concat(meta, ctx, (int) b);
            char skip_pref[64];
            std::snprintf(skip_pref, sizeof(skip_pref), "blk.%u.ffn_gate.", b);
            add_skip_prefix(ml, skip_pref);
        }
    }

    // Hide embedded vision + projector tensors from the text loader.
    add_skip_prefix(ml, "v.");
    add_skip_prefix(ml, "mm.");
}

// =========================================================================
// gpt-oss (text only)
// =========================================================================
//
// Ollama uses arch name "gptoss" (no hyphen) and KV prefix "gptoss.*".
// Upstream uses "gpt-oss" / "gpt-oss.*". Same tensor layout otherwise,
// except:
//   * `blk.X.attn_sinks` -> `blk.X.attn_sinks.weight` (missing suffix)
//   * `blk.X.ffn_norm.weight` -> `blk.X.post_attention_norm.weight`
//     (the second-norm-per-block names differ between converters)

bool detect_ollama_gptoss(const gguf_context * meta) {
    const int64_t arch_kid = gguf_find_key(meta, "general.architecture");
    if (arch_kid < 0) return false;
    return std::strcmp(gguf_get_val_str(meta, arch_kid), "gptoss") == 0;
}

// `arch_name` is mutated to "gpt-oss" so the caller's subsequent
// LLM_KV lookups query the renamed prefix.
void handle_gptoss(const llama_model_loader * ml, gguf_context * meta,
                   ggml_context * ctx, std::string & arch_name) {
    if (!detect_ollama_gptoss(meta)) return;
    (void) ml;

    LLAMA_LOG_INFO("%s: detected Ollama-format gpt-oss GGUF; applying compatibility fixes\n", __func__);

    gguf_set_val_str(meta, "general.architecture", "gpt-oss");
    rename_kv_prefix(meta, "gptoss.", "gpt-oss.");
    arch_name = "gpt-oss";

    // Upstream's gpt-oss loader requires `gpt-oss.expert_feed_forward_length`
    // (n_ff_exp). Ollama omitted it; recover from the ffn_gate_exps tensor
    // shape — for gpt-oss the tensor is created as {n_embd, n_ff_exp, n_expert}
    // so ne[1] is the per-expert FFN dim.
    if (!has_key(meta, "gpt-oss.expert_feed_forward_length")) {
        if (ggml_tensor * t = ggml_get_tensor(ctx, "blk.0.ffn_gate_exps.weight")) {
            gguf_set_val_u32(meta, "gpt-oss.expert_feed_forward_length", (uint32_t) t->ne[1]);
        }
    }

    // Tensor renames. `rename_tensors_containing` does a substring replace
    // on first occurrence — each needle below appears exactly once per
    // tensor name and the needles don't overlap each other.
    rename_tensors_containing(meta, ctx, ".attn_out",
                              ".attn_output");      // wo: out -> output
    rename_tensors_containing(meta, ctx, ".attn_sinks",
                              ".attn_sinks.weight"); // add missing suffix
    rename_tensors_containing(meta, ctx, ".ffn_norm",
                              ".post_attention_norm");
}

// =========================================================================
// lfm2 (text only)
// =========================================================================
//
// Same arch name ("lfm2") on both sides. Only difference is the
// pre-output-projection norm: Ollama writes `output_norm.weight`,
// upstream writes `token_embd_norm.weight` (with the LFM2-specific
// LLM_TENSOR_OUTPUT_NORM_LFM2 mapping). One tensor rename.

bool detect_ollama_lfm2(const gguf_context * meta, const ggml_context * ctx) {
    const int64_t arch_kid = gguf_find_key(meta, "general.architecture");
    if (arch_kid < 0) return false;
    if (std::strcmp(gguf_get_val_str(meta, arch_kid), "lfm2") != 0) return false;
    // Marker: Ollama-converted lfm2 has output_norm.weight, upstream has
    // token_embd_norm.weight instead.
    return ggml_get_tensor(const_cast<ggml_context *>(ctx), "output_norm.weight") != nullptr
        && ggml_get_tensor(const_cast<ggml_context *>(ctx), "token_embd_norm.weight") == nullptr;
}

void handle_lfm2(const llama_model_loader * ml, gguf_context * meta, ggml_context * ctx) {
    if (!detect_ollama_lfm2(meta, ctx)) return;
    (void) ml;

    LLAMA_LOG_INFO("%s: detected Ollama-format lfm2 GGUF; applying compatibility fixes\n", __func__);

    rename_tensor(meta, ctx, "output_norm.weight", "token_embd_norm.weight");

    // Older Ollama converters wrote a stale `lfm2.feed_forward_length` that
    // didn't match the actual ffn_gate tensor shape (e.g. claimed 12288 on
    // a model whose ffn_gate is [2048, 8192]). Fix from the tensor shape.
    if (ggml_tensor * t = ggml_get_tensor(ctx, "blk.0.ffn_gate.weight")) {
        const uint32_t real_n_ff = (uint32_t) t->ne[1];
        const int64_t kid = gguf_find_key(meta, "lfm2.feed_forward_length");
        if (kid < 0 || gguf_get_val_u32(meta, kid) != real_n_ff) {
            gguf_set_val_u32(meta, "lfm2.feed_forward_length", real_n_ff);
        }
    }
}

// =========================================================================
// mistral3 (text only — for now)
// =========================================================================
//
// Same arch name on both sides. Ollama publishes a monolithic GGUF that
// embeds the vision encoder + projector inline, similar to gemma3 and
// qwen35moe. Differences this handler addresses:
//
//   * Embedded `v.*` / `mm.*` tensors must be hidden from the text
//     loader (otherwise n_tensors mismatch).
//   * RoPE YaRN parameters use unprefixed names: Ollama writes
//     `rope.scaling.beta_fast`/`beta_slow`, upstream wants
//     `rope.scaling.yarn_beta_fast`/`yarn_beta_slow`.
//   * Attention temperature scale: Ollama writes `rope.scaling_beta`,
//     upstream reads `attention.temperature_scale`. Same numeric value.
//
// Vision/clip translation is not implemented yet — the user has to skip
// `--mmproj` until a clip handler lands.

bool detect_ollama_mistral3(const gguf_context * meta, const ggml_context * ctx) {
    const int64_t arch_kid = gguf_find_key(meta, "general.architecture");
    if (arch_kid < 0) return false;
    if (std::strcmp(gguf_get_val_str(meta, arch_kid), "mistral3") != 0) return false;
    // Marker: Ollama-style monolithic file embeds v.*/mm.* tensors;
    // upstream HF mistral3 ships these in a separate mmproj.
    return any_tensor_with_prefix(ctx, "v.")
        || any_tensor_with_prefix(ctx, "mm.")
        || has_key(meta, "mistral3.rope.scaling.beta_fast")
        || has_key(meta, "mistral3.rope.scaling_beta");
}

void handle_mistral3(const llama_model_loader * ml, gguf_context * meta, ggml_context * ctx) {
    if (!detect_ollama_mistral3(meta, ctx)) return;
    (void) ctx;

    LLAMA_LOG_INFO("%s: detected Ollama-format mistral3 GGUF; applying compatibility fixes\n", __func__);

    // RoPE YaRN parameter renames.
    copy_kv(meta, "mistral3.rope.scaling.beta_fast",
                  "mistral3.rope.scaling.yarn_beta_fast");
    copy_kv(meta, "mistral3.rope.scaling.beta_slow",
                  "mistral3.rope.scaling.yarn_beta_slow");
    // Attention temperature scale: same value, different home.
    copy_kv(meta, "mistral3.rope.scaling_beta",
                  "mistral3.attention.temperature_scale");

    // Hide embedded vision + projector tensors from the text loader.
    add_skip_prefix(ml, "v.");
    add_skip_prefix(ml, "mm.");
}

// =========================================================================
// gemma3 (clip side)
// =========================================================================

constexpr std::pair<const char *, const char *> kGemma3ClipRenames[] = {
    {"v.patch_embedding",       "v.patch_embd"},
    {"v.position_embedding",    "v.position_embd"},
    {"v.post_layernorm",        "v.post_ln"},
    {".layer_norm1",            ".ln1"},
    {".layer_norm2",            ".ln2"},
    {".attn_output",            ".attn_out"},
    {".mlp.fc1",                ".ffn_down"},
    {".mlp.fc2",                ".ffn_up"},
    {"mm.mm_input_projection",  "mm.input_projection"},
    {"mm.mm_soft_emb_norm",     "mm.soft_emb_norm"},
};

void handle_gemma3_clip(gguf_context * meta, ggml_context * ctx) {
    copy_u32_kv(meta, "gemma3.vision.block_count",                   "clip.vision.block_count");
    copy_u32_kv(meta, "gemma3.vision.embedding_length",              "clip.vision.embedding_length");
    copy_u32_kv(meta, "gemma3.vision.feed_forward_length",           "clip.vision.feed_forward_length");
    copy_u32_kv(meta, "gemma3.vision.image_size",                    "clip.vision.image_size");
    copy_u32_kv(meta, "gemma3.vision.patch_size",                    "clip.vision.patch_size");
    copy_u32_kv(meta, "gemma3.vision.attention.head_count",          "clip.vision.attention.head_count");
    copy_f32_kv(meta, "gemma3.vision.attention.layer_norm_epsilon",  "clip.vision.attention.layer_norm_epsilon");
    // projection_dim = text model's embedding_length (mmproj out == LM in).
    copy_u32_kv(meta, "gemma3.embedding_length",                     "clip.vision.projection_dim");

    static const float kHalfHalfHalf[3] = {0.5f, 0.5f, 0.5f};
    inject_f32_arr_if_missing(meta, "clip.vision.image_mean", kHalfHalfHalf, 3);
    inject_f32_arr_if_missing(meta, "clip.vision.image_std",  kHalfHalfHalf, 3);

    inject_bool_if_missing(meta, "clip.has_vision_encoder", true);
    inject_bool_if_missing(meta, "clip.use_gelu",           true);
    gguf_set_val_str(meta, "clip.projector_type",  "gemma3");
    gguf_set_val_str(meta, "general.architecture", "clip");

    for (const auto & [from, to] : kGemma3ClipRenames) {
        rename_tensors_containing(meta, ctx, from, to);
    }

    // Upstream stores patch_embd/position_embd as F32 (Gemma3VisionModel
    // tensor_force_quant); Ollama stored F16. Metal's IM2COL convolution
    // requires F32, so promote both at load time.
    promote_tensor_to_f32(meta, ctx, "v.patch_embd.weight");
    promote_tensor_to_f32(meta, ctx, "v.position_embd.weight");
}

// =========================================================================
// qwen35moe (clip side)
// =========================================================================

constexpr std::pair<const char *, const char *> kQwen35moeClipRenames[] = {
    {"v.pos_embed",          "v.position_embd"},
    {"v.patch_embed",        "v.patch_embd"},
    {"v.merger.norm",        "v.post_ln"},
    {"v.merger.linear_fc1",  "mm.0"},
    {"v.merger.linear_fc2",  "mm.2"},
    {".mlp.linear_fc1",      ".ffn_up"},
    {".mlp.linear_fc2",      ".ffn_down"},
    {".norm1",               ".ln1"},
    {".norm2",               ".ln2"},
};

// Register a QKV merge for a single vision block: Ollama has separate
// attn_q, attn_k, attn_v tensors; upstream wants them concatenated along
// their slow axis. Capture source file offsets BEFORE renaming attn_q.
void register_qwen35moe_qkv_merge(gguf_context * meta, ggml_context * ctx, int block_idx) {
    char q[64], k[64], v[64], qbias[64], kbias[64], vbias[64], qkv_w[64], qkv_b[64];
    std::snprintf(q,     sizeof(q),     "v.blk.%d.attn_q.weight",   block_idx);
    std::snprintf(k,     sizeof(k),     "v.blk.%d.attn_k.weight",   block_idx);
    std::snprintf(v,     sizeof(v),     "v.blk.%d.attn_v.weight",   block_idx);
    std::snprintf(qbias, sizeof(qbias), "v.blk.%d.attn_q.bias",     block_idx);
    std::snprintf(kbias, sizeof(kbias), "v.blk.%d.attn_k.bias",     block_idx);
    std::snprintf(vbias, sizeof(vbias), "v.blk.%d.attn_v.bias",     block_idx);
    std::snprintf(qkv_w, sizeof(qkv_w), "v.blk.%d.attn_qkv.weight", block_idx);
    std::snprintf(qkv_b, sizeof(qkv_b), "v.blk.%d.attn_qkv.bias",   block_idx);

    if (!ggml_get_tensor(ctx, q)) return; // no vision block at this index

    // Capture source offsets for the concat BEFORE renaming.
    register_concat_load(meta, qkv_w, {q, k, v});
    register_concat_load(meta, qkv_b, {qbias, kbias, vbias});

    // Rename attn_q -> attn_qkv and widen from [hidden, hidden] to [hidden, 3*hidden].
    rename_tensor(meta, ctx, q, qkv_w);
    if (ggml_tensor * t = ggml_get_tensor(ctx, qkv_w)) set_tensor_shape(t, {t->ne[0], t->ne[1] * 3});
    rename_tensor(meta, ctx, qbias, qkv_b);
    if (ggml_tensor * t = ggml_get_tensor(ctx, qkv_b)) set_tensor_shape(t, {t->ne[0] * 3});
}

// Register the patch_embed reshape + split + F16->F32.
//
// Source: one Ollama tensor `v.patch_embed.weight`, ggml shape
//   [h=16, w=16, t=2, packed=3456] F16
// where `packed` is the PyTorch row-major flattening of HF's
// [out_c=1152, in_c=3, ...] dim pair, so packed_c = c_out*3 + c_in.
//
// Destination: two upstream tensors with ggml shape
//   [h=16, w=16, c_in=3, c_out=1152] F32 each, one per temporal slice.
//
// For each output element (h, w, c_in, c_out):
//   src_idx = h + w*W + t*W*H + (c_out*C_in + c_in)*W*H*T
//   dst_idx = h + w*W + c_in*W*H + c_out*W*H*C_in
void register_qwen35moe_patch_embed_split(gguf_context * meta, ggml_context * ctx) {
    const char * src_name = "v.patch_embed.weight";
    if (gguf_find_tensor(meta, src_name) < 0) return;
    const ggml_tensor * src_t = ggml_get_tensor(ctx, src_name);
    if (!src_t) return;

    const size_t src_offset = tensor_file_offset(meta, src_name);
    const size_t src_size   = ggml_nelements(src_t) * sizeof(uint16_t);

    constexpr int H = 16, W = 16, T = 2, CIN = 3, COUT = 1152;
    constexpr size_t HW = (size_t) H * W;

    auto make_slice_op = [=](int slice_idx) {
        return LoadOp{
            [=](const char * path, void * dst, size_t dst_size) {
                if (dst_size != (size_t) H * W * CIN * COUT * sizeof(float)) return false;
                std::vector<uint8_t> src(src_size);
                if (!read_at(path, src_offset, src.data(), src_size)) return false;
                const uint16_t * sp = reinterpret_cast<const uint16_t *>(src.data());
                float          * dp = reinterpret_cast<float *>(dst);
                for (int c_out = 0; c_out < COUT; ++c_out) {
                    for (int c_in = 0; c_in < CIN; ++c_in) {
                        const size_t packed = (size_t) c_out * CIN + c_in;
                        const uint16_t * in_base  = sp + HW * (slice_idx + T * packed);
                        float          * out_base = dp + HW * (c_in + CIN * c_out);
                        for (size_t i = 0; i < HW; ++i) out_base[i] = ggml_fp16_to_fp32(in_base[i]);
                    }
                }
                return true;
            },
            slice_idx == 0 ? "patch_embed slice 0 (permute+F16->F32)"
                           : "patch_embed slice 1 (permute+F16->F32)",
        };
    };

    // Rename src -> `v.patch_embd.weight`, reshape to dest layout, register
    // the slice-0 load op.
    rename_tensor(meta, ctx, src_name, "v.patch_embd.weight");
    if (ggml_tensor * dest0 = ggml_get_tensor(ctx, "v.patch_embd.weight")) {
        set_tensor_shape(dest0, {H, W, CIN, COUT});
        set_tensor_type (dest0, GGML_TYPE_F32);
    }
    register_load_op("v.patch_embd.weight", make_slice_op(0));

    // Reclaim the `v.blk.0.attn_k.weight` slot (orphaned by the QKV merge)
    // as the sibling `v.patch_embd.weight.1`.
    reclaim_slot_as(meta, ctx,
                    "v.blk.0.attn_k.weight", "v.patch_embd.weight.1",
                    {H, W, CIN, COUT}, GGML_TYPE_F32);
    register_load_op("v.patch_embd.weight.1", make_slice_op(1));
}

void handle_qwen35moe_clip(gguf_context * meta, ggml_context * ctx) {
    LLAMA_LOG_INFO("%s: detected Ollama-format qwen35moe GGUF used as mmproj; translating\n", __func__);

    copy_u32_kv(meta, "qwen35moe.vision.block_count",                   "clip.vision.block_count");
    copy_u32_kv(meta, "qwen35moe.vision.embedding_length",              "clip.vision.embedding_length");
    copy_u32_kv(meta, "qwen35moe.vision.attention.head_count",          "clip.vision.attention.head_count");
    copy_u32_kv(meta, "qwen35moe.vision.patch_size",                    "clip.vision.patch_size");
    copy_u32_kv(meta, "qwen35moe.vision.spatial_merge_size",            "clip.vision.spatial_merge_size");
    copy_u32_kv(meta, "qwen35moe.vision.num_channels",                  "clip.vision.num_channels");
    // projection_dim = text model's embedding_length.
    copy_u32_kv(meta, "qwen35moe.embedding_length",                     "clip.vision.projection_dim");

    // Defaults for KVs Ollama omitted (match the Qwen3.5-35B-A3B reference mmproj).
    inject_u32_if_missing(meta, "clip.vision.feed_forward_length",          4304);
    inject_u32_if_missing(meta, "clip.vision.image_size",                   768);
    inject_f32_if_missing(meta, "clip.vision.attention.layer_norm_epsilon", 1e-6f);

    static const float kHalfHalfHalf[3] = {0.5f, 0.5f, 0.5f};
    inject_f32_arr_if_missing(meta, "clip.vision.image_mean", kHalfHalfHalf, 3);
    inject_f32_arr_if_missing(meta, "clip.vision.image_std",  kHalfHalfHalf, 3);

    // is_deepstack_layers: qwen3.5 35B has no deepstack layers. Set 27 False.
    if (!has_key(meta, "clip.vision.is_deepstack_layers")) {
        uint8_t bools[27] = {};
        gguf_set_arr_data(meta, "clip.vision.is_deepstack_layers", GGUF_TYPE_BOOL, bools, 27);
    }

    inject_bool_if_missing(meta, "clip.has_vision_encoder", true);
    inject_bool_if_missing(meta, "clip.use_gelu",           true);
    gguf_set_val_str(meta, "clip.projector_type",  "qwen3vl_merger");
    gguf_set_val_str(meta, "general.architecture", "clip");

    // QKV merge runs BEFORE substring renames so it can find attn_q/k/v by name.
    const int64_t n_blocks_key = gguf_find_key(meta, "clip.vision.block_count");
    const uint32_t n_blocks = n_blocks_key >= 0 ? gguf_get_val_u32(meta, n_blocks_key) : 27;
    for (uint32_t b = 0; b < n_blocks; ++b) register_qwen35moe_qkv_merge(meta, ctx, (int) b);

    // Also before renames: patch_embed references the source by name.
    register_qwen35moe_patch_embed_split(meta, ctx);

    // Simple substring renames.
    for (const auto & [from, to] : kQwen35moeClipRenames) {
        rename_tensors_containing(meta, ctx, from, to);
    }

    promote_tensor_to_f32(meta, ctx, "v.position_embd.weight");
}

// =========================================================================
// deepseek-ocr (clip side — SAM + CLIP + projector)
// =========================================================================
//
// Ollama's monolithic deepseek-ocr GGUF embeds three vision components:
//   * SAM encoder under the `s.*` prefix (12 blocks)
//   * CLIP encoder under the `v.*` prefix (24 blocks)
//   * MLP projector under `mm.*`
// Upstream's PROJECTOR_TYPE_DEEPSEEKOCR loader expects:
//   * SAM under `v.sam.*`
//   * CLIP under `v.*` (different leaf names than Ollama)
//   * Projector as `mm.model.fc.*` plus `v.image_newline` / `v.view_seperator`

constexpr std::pair<const char *, const char *> kDeepseekocrClipRenames[] = {
    // CLIP block leaf renames (also affects v.sam.* but those names don't overlap).
    {".self_attn.out_proj", ".attn_out"},
    {".self_attn.qkv_proj", ".attn_qkv"},
    {".layer_norm1",        ".ln1"},
    {".layer_norm2",        ".ln2"},
    {".mlp.fc1",            ".ffn_up"},
    {".mlp.fc2",            ".ffn_down"},
    {"v.pre_layrnorm",      "v.pre_ln"}, // Ollama typo

    // SAM block leaf renames (after `s.*` -> `v.sam.*` is applied).
    {".attn.proj",          ".attn.out"},
    {".attn.rel_pos_h",     ".attn.pos_h.weight"},
    {".attn.rel_pos_w",     ".attn.pos_w.weight"},
    {".norm1",              ".pre_ln"},
    {".norm2",              ".post_ln"},

    // Projector renames.
    {"mm.layers",           "mm.model.fc"},
    {"mm.image_newline",    "v.image_newline"},
    {"mm.view_seperator",   "v.view_seperator"},
};

void handle_deepseekocr_clip(gguf_context * meta, ggml_context * ctx) {
    LLAMA_LOG_INFO("%s: detected Ollama-format deepseekocr GGUF used as mmproj; translating\n", __func__);

    // CLIP encoder hparams.
    copy_u32_kv(meta, "deepseekocr.vision.block_count",      "clip.vision.block_count");
    copy_u32_kv(meta, "deepseekocr.vision.embedding_length", "clip.vision.embedding_length");
    copy_u32_kv(meta, "deepseekocr.vision.head_count",       "clip.vision.attention.head_count");
    copy_u32_kv(meta, "deepseekocr.vision.image_size",       "clip.vision.image_size");
    copy_u32_kv(meta, "deepseekocr.vision.patch_size",       "clip.vision.patch_size");

    // SAM encoder hparams.
    copy_u32_kv(meta, "deepseekocr.sam.block_count",         "clip.vision.sam.block_count");
    copy_u32_kv(meta, "deepseekocr.sam.embedding_length",    "clip.vision.sam.embedding_length");
    copy_u32_kv(meta, "deepseekocr.sam.head_count",          "clip.vision.sam.head_count");

    // Defaults pulled from the upstream-converted reference mmproj.
    inject_u32_if_missing(meta, "clip.vision.feed_forward_length",          64);
    inject_u32_if_missing(meta, "clip.vision.projection_dim",               1280);
    inject_u32_if_missing(meta, "clip.vision.projector.scale_factor",       1);
    inject_u32_if_missing(meta, "clip.vision.window_size",                  14);
    inject_f32_if_missing(meta, "clip.vision.attention.layer_norm_epsilon", 1e-6f);

    static const float kHalfHalfHalf[3] = {0.5f, 0.5f, 0.5f};
    inject_f32_arr_if_missing(meta, "clip.vision.image_mean", kHalfHalfHalf, 3);
    inject_f32_arr_if_missing(meta, "clip.vision.image_std",  kHalfHalfHalf, 3);

    inject_bool_if_missing(meta, "clip.has_vision_encoder", true);
    inject_bool_if_missing(meta, "clip.use_gelu",           true);
    gguf_set_val_str(meta, "clip.projector_type",  "deepseekocr");
    gguf_set_val_str(meta, "general.architecture", "clip");

    // Step 1: rename SAM prefix `s.` -> `v.sam.` only at the start of names
    // (substring rename would corrupt e.g. `mm.layers.weight` -> `mm.layerv.sam.weight`).
    {
        std::vector<std::string> sam_names;
        const int64_t n = gguf_get_n_tensors(meta);
        for (int64_t i = 0; i < n; ++i) {
            std::string name(gguf_get_tensor_name(meta, i));
            if (name.size() >= 2 && name[0] == 's' && name[1] == '.') {
                sam_names.push_back(std::move(name));
            }
        }
        for (const auto & old_name : sam_names) {
            rename_tensor(meta, ctx, old_name.c_str(),
                          ("v.sam." + old_name.substr(2)).c_str());
        }
    }

    // Step 2: SAM `s.position_embd` (no `.weight` suffix) — handle exactly,
    // since after the `s.` rename it lives at `v.sam.position_embd`.
    rename_tensor(meta, ctx, "v.sam.position_embd", "v.sam.pos_embd.weight");

    // Step 3: substring renames for CLIP, SAM block leaves, and projector.
    for (const auto & [from, to] : kDeepseekocrClipRenames) {
        rename_tensors_containing(meta, ctx, from, to);
    }

    // Metal IM2COL needs F32 patch_embd (same issue as gemma3 / mistral3).
    promote_tensor_to_f32(meta, ctx, "v.patch_embd.weight");
    promote_tensor_to_f32(meta, ctx, "v.sam.patch_embd.weight");
    // CLIP position embedding too — Ollama stores F16, upstream stores F32.
    promote_tensor_to_f32(meta, ctx, "v.position_embd.weight");
}

// =========================================================================
// gemma4 (clip side — gemma4v projector)
// =========================================================================
//
// Ollama's monolithic gemma4 GGUF embeds a SigLIP-style ViT plus the
// gemma4v projector (a single `mm.input_projection`). All v.* / mm.*
// tensor names already match upstream's PROJECTOR_TYPE_GEMMA4V — this
// handler only needs KV translation and an F32 promote of the patch
// embedding (Metal IM2COL).
//
// gemma4 vision uses image normalization mean=[0,0,0] / std=[1,1,1]
// (the LM does its own per-image normalization via v.std_bias /
// v.std_scale tensors) — different from the [0.5,0.5,0.5] used by
// most other arches.

void handle_gemma4_clip(gguf_context * meta, ggml_context * ctx) {
    LLAMA_LOG_INFO("%s: detected Ollama-format gemma4 GGUF used as mmproj; translating\n", __func__);

    gguf_set_val_str(meta, "general.architecture", "clip");

    const bool has_vision = any_tensor_with_prefix(ctx, "v.");
    const bool has_audio  = any_tensor_with_prefix(ctx, "a.");

    if (has_vision) {
        copy_u32_kv(meta, "gemma4.vision.block_count",                   "clip.vision.block_count");
        copy_u32_kv(meta, "gemma4.vision.embedding_length",              "clip.vision.embedding_length");
        copy_u32_kv(meta, "gemma4.vision.feed_forward_length",           "clip.vision.feed_forward_length");
        copy_u32_kv(meta, "gemma4.vision.attention.head_count",          "clip.vision.attention.head_count");
        copy_f32_kv(meta, "gemma4.vision.attention.layer_norm_epsilon",  "clip.vision.attention.layer_norm_epsilon");
        copy_u32_kv(meta, "gemma4.vision.patch_size",                    "clip.vision.patch_size");
        // gemma4 vision is fixed at 224x224 patches.
        inject_u32_if_missing(meta, "clip.vision.image_size", 224);
        // projection_dim = LM embedding length.
        copy_u32_kv(meta, "gemma4.embedding_length",                     "clip.vision.projection_dim");

        static const float kZeros[3] = {0.0f, 0.0f, 0.0f};
        static const float kOnes [3] = {1.0f, 1.0f, 1.0f};
        inject_f32_arr_if_missing(meta, "clip.vision.image_mean", kZeros, 3);
        inject_f32_arr_if_missing(meta, "clip.vision.image_std",  kOnes,  3);

        inject_bool_if_missing(meta, "clip.has_vision_encoder", true);
        gguf_set_val_str(meta, "clip.vision.projector_type", "gemma4v");

        // Metal IM2COL needs F32 patch_embd weights (same as other arches).
        promote_tensor_to_f32(meta, ctx, "v.patch_embd.weight");
    }

    if (has_audio) {
        // Audio (gemma4a — conformer encoder + audio multimodal embedder).
        copy_u32_kv(meta, "gemma4.audio.block_count",                   "clip.audio.block_count");
        copy_u32_kv(meta, "gemma4.audio.embedding_length",              "clip.audio.embedding_length");
        copy_u32_kv(meta, "gemma4.audio.feed_forward_length",           "clip.audio.feed_forward_length");
        copy_u32_kv(meta, "gemma4.audio.attention.head_count",          "clip.audio.attention.head_count");
        copy_f32_kv(meta, "gemma4.audio.attention.layer_norm_epsilon",  "clip.audio.attention.layer_norm_epsilon");
        // Defaults from the upstream-converted reference E2B mmproj.
        inject_u32_if_missing(meta, "clip.audio.num_mel_bins",   128);
        inject_u32_if_missing(meta, "clip.audio.projection_dim", 1536);

        inject_bool_if_missing(meta, "clip.has_audio_encoder", true);
        gguf_set_val_str(meta, "clip.audio.projector_type", "gemma4a");

        // Top-level tensor renames. Ollama uses different leaf names for the
        // SSCP input projection and the encoder output projection:
        //   a.pre_encode.out.weight   →  a.input_projection.weight (SSCP proj)
        //   mm.a.fc.{weight,bias}     →  a.pre_encode.out.{weight,bias}
        //   mm.a.input_projection.weight already matches.
        rename_tensor(meta, ctx, "a.pre_encode.out.weight", "a.input_projection.weight");
        rename_tensor(meta, ctx, "mm.a.fc.weight",          "a.pre_encode.out.weight");
        rename_tensor(meta, ctx, "mm.a.fc.bias",            "a.pre_encode.out.bias");

        // Per-block renames. Scoped to a.blk.* (NOT vision blocks, which also
        // have ln1/ln2). Order matters: ln2 → attn_post_norm must run before
        // layer_pre_norm → ln2 (otherwise the second rename collides).
        //
        // Semantic mapping (from Ollama's model_audio.go vs upstream gemma4a.cpp):
        //   ln1            → attn_pre_norm    (pre-attention norm)
        //   ln2            → attn_post_norm   (post-attention norm; NOT block out)
        //   layer_pre_norm → ln2              (final block output norm)
        //   linear_pos     → attn_k_rel       (relative-position K projection)
        const int kid = gguf_find_key(meta, "gemma4.audio.block_count");
        const uint32_t n_audio = (kid >= 0) ? gguf_get_val_u32(meta, kid) : 12;
        for (uint32_t il = 0; il < n_audio; ++il) {
            char from[GGML_MAX_NAME], to[GGML_MAX_NAME];
            auto rn = [&](const char * a, const char * b) {
                std::snprintf(from, sizeof(from), "a.blk.%u.%s.weight", il, a);
                std::snprintf(to,   sizeof(to),   "a.blk.%u.%s.weight", il, b);
                rename_tensor(meta, ctx, from, to);
            };
            rn("ln1",            "attn_pre_norm");
            rn("ln2",            "attn_post_norm");
            rn("layer_pre_norm", "ln2");
            rn("linear_pos",     "attn_k_rel");
        }
    }
}

// =========================================================================
// glm-ocr (clip side — glm4v projector)
// =========================================================================
//
// Ollama stores the GLM4V vision tower with v.blk.X.* tensor names that
// already match upstream's expectations (`attn_qkv`, `attn_out`,
// `attn_q_norm`, `attn_k_norm`, `ln1`/`ln2`, `ffn_{gate,up,down}`).
// Most of mm.* (mm.model.fc, mm.up/gate/down, mm.post_norm,
// mm.patch_merger) is also already named correctly. The two diffs:
//   * `v.patch_embd_0.weight` / `v.patch_embd_1.weight` → upstream's
//     pixel-shuffle patch-embed pair `v.patch_embd.weight` /
//     `v.patch_embd.weight.1`.
//   * F32 promote of patch_embd weights (Metal IM2COL).

void handle_glmocr_clip(gguf_context * meta, ggml_context * ctx) {
    LLAMA_LOG_INFO("%s: detected Ollama-format glm-ocr GGUF used as mmproj; translating\n", __func__);

    copy_u32_kv(meta, "glmocr.vision.block_count",                   "clip.vision.block_count");
    copy_u32_kv(meta, "glmocr.vision.embedding_length",              "clip.vision.embedding_length");
    copy_u32_kv(meta, "glmocr.vision.intermediate_size",             "clip.vision.feed_forward_length");
    copy_u32_kv(meta, "glmocr.vision.attention.head_count",          "clip.vision.attention.head_count");
    copy_f32_kv(meta, "glmocr.vision.attention.layer_norm_rms_epsilon", "clip.vision.attention.layer_norm_epsilon");
    copy_u32_kv(meta, "glmocr.vision.image_size",                    "clip.vision.image_size");
    copy_u32_kv(meta, "glmocr.vision.patch_size",                    "clip.vision.patch_size");
    copy_u32_kv(meta, "glmocr.vision.spatial_merge_size",            "clip.vision.spatial_merge_size");
    copy_u32_kv(meta, "glmocr.vision.out_hidden_size",               "clip.vision.projection_dim");

    // Ollama already shipped image_mean / image_std under glmocr.vision.*;
    // copy them through.
    {
        const int64_t kid = gguf_find_key(meta, "glmocr.vision.image_mean");
        if (kid >= 0 && !has_key(meta, "clip.vision.image_mean")) {
            const size_t n = gguf_get_arr_n(meta, kid);
            gguf_set_arr_data(meta, "clip.vision.image_mean", GGUF_TYPE_FLOAT32,
                              gguf_get_arr_data(meta, kid), n);
        }
    }
    {
        const int64_t kid = gguf_find_key(meta, "glmocr.vision.image_std");
        if (kid >= 0 && !has_key(meta, "clip.vision.image_std")) {
            const size_t n = gguf_get_arr_n(meta, kid);
            gguf_set_arr_data(meta, "clip.vision.image_std", GGUF_TYPE_FLOAT32,
                              gguf_get_arr_data(meta, kid), n);
        }
    }

    inject_bool_if_missing(meta, "clip.has_vision_encoder", true);
    inject_bool_if_missing(meta, "clip.use_silu",           true);
    gguf_set_val_str(meta, "clip.projector_type",  "glm4v");
    gguf_set_val_str(meta, "general.architecture", "clip");

    // Patch-embed temporal pair: Ollama uses _0/_1 suffixes, upstream uses
    // unsuffixed/.1.
    rename_tensor(meta, ctx, "v.patch_embd_0.weight", "v.patch_embd.weight");
    rename_tensor(meta, ctx, "v.patch_embd_1.weight", "v.patch_embd.weight.1");

    // F32 promote for IM2COL on Metal (same fix as gemma3 / mistral3).
    promote_tensor_to_f32(meta, ctx, "v.patch_embd.weight");
    promote_tensor_to_f32(meta, ctx, "v.patch_embd.weight.1");
}

// =========================================================================
// llama4 (clip side)
// =========================================================================
//
// Ollama's monolithic llama4 GGUF embeds the CLIP-style ViT and a 3-layer
// projector (`mm.linear_1` + `v.vision_adapter.mlp.fc1/fc2`). Upstream's
// PROJECTOR_TYPE_LLAMA4 expects the projector under `mm.model.fc` /
// `mm.model.mlp.{1,2}` and standard CLIP block leaf names.

constexpr std::pair<const char *, const char *> kLlama4ClipRenames[] = {
    // Vision-adapter MLP -> upstream's MM-MLP slots. Run BEFORE the generic
    // `.mlp.fc{1,2}` -> `.ffn_{up,down}` rename so the substring match stays
    // pinned to the adapter prefix.
    {"v.vision_adapter.mlp.fc1", "mm.model.mlp.1"},
    {"v.vision_adapter.mlp.fc2", "mm.model.mlp.2"},

    // Main projector.
    {"mm.linear_1",              "mm.model.fc"},

    // Vision tower non-blk.
    {"v.class_embedding",        "v.class_embd"},
    {"v.layernorm_post",         "v.post_ln"},
    {"v.layernorm_pre",          "v.pre_ln"},
    {"v.patch_embedding",        "v.patch_embd"},

    // Vision-tower block leaves.
    {".attn_output",             ".attn_out"},
    {".attn_norm",               ".ln1"},
    {".ffn_norm",                ".ln2"},
    {".mlp.fc1",                 ".ffn_up"},
    {".mlp.fc2",                 ".ffn_down"},
};

void handle_llama4_clip(gguf_context * meta, ggml_context * ctx) {
    LLAMA_LOG_INFO("%s: detected Ollama-format llama4 GGUF used as mmproj; translating\n", __func__);

    copy_u32_kv(meta, "llama4.vision.block_count",                    "clip.vision.block_count");
    copy_u32_kv(meta, "llama4.vision.embedding_length",               "clip.vision.embedding_length");
    copy_u32_kv(meta, "llama4.vision.feed_forward_length",            "clip.vision.feed_forward_length");
    copy_u32_kv(meta, "llama4.vision.attention.head_count",           "clip.vision.attention.head_count");
    copy_u32_kv(meta, "llama4.vision.image_size",                     "clip.vision.image_size");
    copy_u32_kv(meta, "llama4.vision.patch_size",                     "clip.vision.patch_size");
    copy_f32_kv(meta, "llama4.vision.layer_norm_epsilon",             "clip.vision.attention.layer_norm_epsilon");
    // projection_dim = LM embedding length (= mm.model.fc output dim).
    copy_u32_kv(meta, "llama4.embedding_length",                      "clip.vision.projection_dim");

    // Defaults (match the upstream-converted reference mmproj).
    inject_u32_if_missing(meta, "clip.vision.projector.scale_factor", 2);

    static const float kHalfHalfHalf[3] = {0.5f, 0.5f, 0.5f};
    inject_f32_arr_if_missing(meta, "clip.vision.image_mean", kHalfHalfHalf, 3);
    inject_f32_arr_if_missing(meta, "clip.vision.image_std",  kHalfHalfHalf, 3);

    inject_bool_if_missing(meta, "clip.has_vision_encoder", true);
    inject_bool_if_missing(meta, "clip.use_gelu",           true);
    gguf_set_val_str(meta, "clip.projector_type",  "llama4");
    gguf_set_val_str(meta, "general.architecture", "clip");

    // Position embedding has no `.weight` suffix in Ollama; rename exactly.
    rename_tensor(meta, ctx, "v.positional_embedding_vlm", "v.position_embd.weight");

    for (const auto & [from, to] : kLlama4ClipRenames) {
        rename_tensors_containing(meta, ctx, from, to);
    }
}

// =========================================================================
// mistral3 (clip side — pixtral projector)
// =========================================================================
//
// Tensor renames Ollama → upstream pixtral:
//   v.patch_conv                       -> v.patch_embd
//   v.encoder_norm                     -> v.pre_ln
//   v.blk.X.attn_output                -> v.blk.X.attn_out
//   v.blk.X.attn_norm                  -> v.blk.X.ln1
//   v.blk.X.ffn_norm                   -> v.blk.X.ln2
//   mm.linear_1                        -> mm.1
//   mm.linear_2                        -> mm.2
//   mm.norm                            -> mm.input_norm
//   mm.patch_merger.merging_layer      -> mm.patch_merger
//
// img_break: pixtral's loader requires `v.token_embd.img_break` (the
// embedding row for the [IMG_BREAK] token, used as a row separator).
// Ollama's monolithic blob doesn't ship it as a separate tensor; the
// "ideal" value is row 12 of token_embd.weight, but token_embd is
// quantized (Q4_K) and per-row dequant is heavyweight. Reclaim the
// orphan output_norm.weight slot (already [n_embd] F32) and zero-fill
// it — pixtral.cpp adds img_break to row separator embeddings, so a
// zero embedding makes [IMG_BREAK] insertion a no-op without breaking
// the rest of the vision graph.
constexpr std::pair<const char *, const char *> kMistral3ClipRenames[] = {
    {"v.patch_conv",                  "v.patch_embd"},
    {"v.encoder_norm",                "v.pre_ln"},
    {".attn_output",                  ".attn_out"},
    {".attn_norm",                    ".ln1"},
    {".ffn_norm",                     ".ln2"},
    {"mm.linear_1",                   "mm.1"},
    {"mm.linear_2",                   "mm.2"},
    {"mm.patch_merger.merging_layer", "mm.patch_merger"},
    {"mm.norm",                       "mm.input_norm"},
};

// Apply the LLaMA-style RoPE permutation to Ollama's vision Q/K weight.
//
// Ollama's mistral3 converter (convert/convert_mistral.go) only applies
// its repack to TEXT-side attn_q/attn_k (the `if !HasPrefix(name, "v.")`
// guard skips vision tensors). So vision Q/K leave the converter in raw
// HF/PyTorch order. Upstream's HF→GGUF flow (convert_hf_to_gguf.py
// Mistral3 path) DOES permute vision Q/K with the vision head count,
// because pixtral's clip graph uses `ggml_rope_ext` in mode 0 which
// expects the [n_head, head_dim/2, 2, ...] layout.
//
// To bridge the two: apply LlamaModel.permute equivalently — reshape
// to [n_head, 2, head_dim/2, in], swap axes 1↔2, reshape back. The
// permutation acts only on the output dim, which is ne[1] for ggml
// weights stored as [in_dim, out_dim], so we shuffle whole rows.
//
// Permutation formula: oa = h*head_dim + dp*2 + half  (post-permute idx)
//                      ob = h*head_dim + half*(head_dim/2) + dp  (HF idx)
//                      copy row ob in src → row oa in dst.
//
// Only F16 Q/K rows handled (V is not RoPE'd; quantized rows would need
// block-aware shuffling — Ollama keeps Q/K F16 for mistral3 8B).
void register_mistral3_vision_qk_permute(gguf_context * meta, ggml_context * ctx,
                                         const char * tensor_name, int n_head) {
    ggml_tensor * t = ggml_get_tensor(ctx, tensor_name);
    if (!t || t->type != GGML_TYPE_F16) return;

    const int total_out = (int) t->ne[1];
    if (total_out % n_head != 0) return;
    const size_t row_bytes = ggml_row_size(t->type, t->ne[0]);
    const size_t total_bytes = ggml_nbytes(t);
    const size_t src_offset = tensor_file_offset(meta, tensor_name);

    const int head_dim  = total_out / n_head;
    const int head_dim2 = head_dim / 2;

    register_load_op(tensor_name, LoadOp{
        [=](const char * path, void * dst, size_t dst_size) {
            if (dst_size != total_bytes) return false;
            std::vector<uint8_t> src(total_bytes);
            if (!read_at(path, src_offset, src.data(), total_bytes)) return false;
            uint8_t * dp = static_cast<uint8_t *>(dst);
            for (int oa = 0; oa < total_out; ++oa) {
                const int h    = oa / head_dim;
                const int dp_  = (oa % head_dim) / 2;
                const int hf   = oa % 2;
                const int ob   = h * head_dim + hf * head_dim2 + dp_;
                std::memcpy(dp + (size_t) oa * row_bytes,
                            src.data() + (size_t) ob * row_bytes, row_bytes);
            }
            return true;
        },
        "vision Q/K LLaMA permute",
    });
}

void handle_mistral3_clip(gguf_context * meta, ggml_context * ctx) {
    LLAMA_LOG_INFO("%s: detected Ollama-format mistral3 GGUF used as mmproj; translating\n", __func__);

    copy_u32_kv(meta, "mistral3.vision.block_count",            "clip.vision.block_count");
    copy_u32_kv(meta, "mistral3.vision.embedding_length",       "clip.vision.embedding_length");
    copy_u32_kv(meta, "mistral3.vision.feed_forward_length",    "clip.vision.feed_forward_length");
    copy_u32_kv(meta, "mistral3.vision.attention.head_count",   "clip.vision.attention.head_count");
    copy_u32_kv(meta, "mistral3.vision.image_size",             "clip.vision.image_size");
    copy_u32_kv(meta, "mistral3.vision.patch_size",             "clip.vision.patch_size");
    copy_u32_kv(meta, "mistral3.vision.num_channels",           "clip.vision.num_channels");
    copy_u32_kv(meta, "mistral3.spatial_merge_size",            "clip.vision.spatial_merge_size");
    copy_f32_kv(meta, "mistral3.vision.rope.freq_base",         "clip.rope.freq_base");
    // projection_dim is required by the loader but pixtral derives the
    // actual output dim from mm_2_w shape — any non-zero value works.
    // Mirror the LM embedding length for diagnostics-friendliness.
    copy_u32_kv(meta, "mistral3.embedding_length",              "clip.vision.projection_dim");

    inject_f32_if_missing(meta, "clip.vision.attention.layer_norm_epsilon", 1e-5f);

    // Pixtral image stats (CLIP-style means).
    static const float kPixtralMean[3] = {0.48145467f, 0.45782750f, 0.40821072f};
    static const float kPixtralStd [3] = {0.26862955f, 0.26130259f, 0.27577710f};
    inject_f32_arr_if_missing(meta, "clip.vision.image_mean", kPixtralMean, 3);
    inject_f32_arr_if_missing(meta, "clip.vision.image_std",  kPixtralStd,  3);

    inject_bool_if_missing(meta, "clip.has_vision_encoder", true);
    inject_bool_if_missing(meta, "clip.use_silu",           true);
    gguf_set_val_str(meta, "clip.projector_type",  "pixtral");
    gguf_set_val_str(meta, "general.architecture", "clip");

    // Reclaim output_norm.weight as v.token_embd.img_break (zero-filled).
    const int64_t lm_embd_kid = gguf_find_key(meta, "mistral3.embedding_length");
    const uint32_t lm_embd = lm_embd_kid >= 0 ? gguf_get_val_u32(meta, lm_embd_kid) : 0;
    if (lm_embd > 0 && reclaim_slot_as(meta, ctx,
                                       "output_norm.weight", "v.token_embd.img_break",
                                       {(int64_t) lm_embd}, GGML_TYPE_F32)) {
        register_load_op("v.token_embd.img_break", LoadOp{
            [](const char *, void * dst, size_t dst_size) {
                std::memset(dst, 0, dst_size);
                return true;
            },
            "img_break zero-fill",
        });
    }

    // Apply LLaMA-style RoPE permutation to vision Q/K BEFORE renames
    // (we capture offsets by current name). Ollama's converter only
    // repacks TEXT-side q/k (skipping `v.*`), but pixtral's clip graph
    // expects HF→GGUF's permuted layout for vision Q/K.
    {
        const int64_t v_hk    = gguf_find_key(meta, "mistral3.vision.attention.head_count");
        const int64_t n_blk_k = gguf_find_key(meta, "mistral3.vision.block_count");
        if (v_hk >= 0 && n_blk_k >= 0) {
            const int n_head = (int) gguf_get_val_u32(meta, v_hk);
            const uint32_t n_blocks = gguf_get_val_u32(meta, n_blk_k);
            for (uint32_t b = 0; b < n_blocks; ++b) {
                char qn[64], kn[64];
                std::snprintf(qn, sizeof(qn), "v.blk.%u.attn_q.weight", b);
                std::snprintf(kn, sizeof(kn), "v.blk.%u.attn_k.weight", b);
                register_mistral3_vision_qk_permute(meta, ctx, qn, n_head);
                register_mistral3_vision_qk_permute(meta, ctx, kn, n_head);
            }
        }
    }

    for (const auto & [from, to] : kMistral3ClipRenames) {
        rename_tensors_containing(meta, ctx, from, to);
    }

    // Upstream stores patch_embd as F32; Ollama stored F16. Metal's
    // IM2COL convolution silently produces garbage with F16 weights
    // (same issue as gemma3 — see handle_gemma3_clip). Promote to F32.
    promote_tensor_to_f32(meta, ctx, "v.patch_embd.weight");
}

} // anonymous namespace

// =========================================================================
// Public entry points
// =========================================================================

void translate_metadata(const llama_model_loader * ml,
                        gguf_context * meta,
                        ggml_context * ctx,
                        std::string & arch_name,
                        const char * fname) {
    if (!meta) return;
    {
        std::lock_guard<std::mutex> lk(g_loader_path_mutex);
        g_loader_paths[ml] = fname ? fname : "";
    }
    // embeddinggemma must run before gemma3: it switches arch_name to
    // "gemma-embedding", which is what later checks (and the loader's KV
    // prefix) need to see.
    if (arch_name == "gemma3")    handle_embeddinggemma(ml, meta, ctx, arch_name);
    if (arch_name == "gemma3")    handle_gemma3   (ml, meta, ctx);
    if (arch_name == "gemma4")    handle_gemma4   (ml, meta, ctx);
    if (arch_name == "qwen35moe") handle_qwen35moe(ml, meta, ctx);
    if (arch_name == "qwen35")    handle_qwen35   (ml, meta, ctx);
    if (arch_name == "gptoss")        handle_gptoss        (ml, meta, ctx, arch_name);
    if (arch_name == "lfm2")          handle_lfm2          (ml, meta, ctx);
    if (arch_name == "mistral3")      handle_mistral3      (ml, meta, ctx);
    if (arch_name == "deepseekocr")   handle_deepseekocr   (ml, meta, ctx, arch_name);
    if (arch_name == "nemotron_h_moe") handle_nemotron_h_moe(ml, meta, ctx);
    if (arch_name == "llama4")        handle_llama4        (ml, meta, ctx);
    if (arch_name == "glmocr")        handle_glmocr        (ml, meta, ctx, arch_name);
    // Dispatch. Add more arches as they are wired up.
}

void translate_clip_metadata(gguf_context * meta, ggml_context * ctx) {
    if (!meta) return;
    if (!any_tensor_with_prefix(ctx, "v.")) return; // nothing to translate

    if (detect_ollama_gemma3(meta, ctx)) {
        LLAMA_LOG_INFO("%s: detected Ollama-format gemma3 GGUF used as mmproj; translating\n", __func__);
        handle_gemma3_clip(meta, ctx);
        return;
    }
    if (detect_ollama_qwen35moe(meta, ctx)) {
        handle_qwen35moe_clip(meta, ctx);
        return;
    }
    if (detect_ollama_mistral3(meta, ctx)) {
        handle_mistral3_clip(meta, ctx);
        return;
    }
    if (detect_ollama_deepseekocr(meta)) {
        handle_deepseekocr_clip(meta, ctx);
        return;
    }
    if (detect_ollama_llama4(meta, ctx)) {
        handle_llama4_clip(meta, ctx);
        return;
    }
    if (detect_ollama_gemma4(meta, ctx)) {
        handle_gemma4_clip(meta, ctx);
        return;
    }
    if (detect_ollama_glmocr(meta)) {
        handle_glmocr_clip(meta, ctx);
        return;
    }
}

bool should_skip_tensor(const llama_model_loader * ml, const char * tensor_name) {
    return should_skip_tensor_prefix(ml, tensor_name);
}

bool maybe_load_tensor(ggml_tensor * cur,
                       const char * source_file,
                       size_t file_offset,
                       ggml_backend_buffer_type_t buft) {
    (void) file_offset; // registered ops capture their own offsets

    LoadOp op;
    if (!take_load_op(ggml_get_name(cur), op)) return false;

    const size_t dst_size = ggml_nbytes(cur);
    std::vector<uint8_t> dst(dst_size);
    if (!op.apply(source_file, dst.data(), dst_size)) {
        LLAMA_LOG_ERROR("%s: %s failed for %s\n", __func__, op.description, ggml_get_name(cur));
        return false;
    }

    if (ggml_backend_buft_is_host(buft)) std::memcpy(cur->data, dst.data(), dst_size);
    else                                 ggml_backend_tensor_set(cur, dst.data(), 0, dst_size);

    LLAMA_LOG_INFO("%s: %s for %s (%zu bytes)\n", __func__, op.description, ggml_get_name(cur), dst_size);
    return true;
}

bool maybe_load_text_tensor(const llama_model_loader * ml,
                            ggml_tensor * cur,
                            size_t file_offset) {
    std::string path;
    {
        std::lock_guard<std::mutex> lk(g_loader_path_mutex);
        auto it = g_loader_paths.find(ml);
        if (it == g_loader_paths.end() || it->second.empty()) return false;
        path = it->second;
    }
    ggml_backend_buffer_type_t buft = cur->buffer
        ? ggml_backend_buffer_get_type(cur->buffer)
        : nullptr;
    return maybe_load_tensor(cur, path.c_str(), file_offset, buft);
}

} // namespace llama_ollama_compat
