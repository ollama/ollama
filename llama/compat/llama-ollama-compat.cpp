#include "llama-ollama-compat.h"
#include "llama-ollama-compat-util.h"

#include "llama-impl.h"

// Temporary compatibility layer for existing published GGUFs whose metadata
// or tensor layout does not match llama.cpp's current loaders. The goal is to
// minimize user disruption during a transition phase to llama.cpp-compatible
// GGUFs with manifest-list support. Ultimately this patch should be removed.

#include <cmath>
#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstdarg>
#include <cstdint>
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

#ifdef OLLAMA_COMPAT_MTMD_BUILD
void ollama_compat_log(const char * format, ...) {
    std::va_list args;
    va_start(args, format);
    std::vfprintf(stderr, format, args);
    va_end(args);
}

#define OLLAMA_COMPAT_LOG_INFO(...)  do { ollama_compat_log(__VA_ARGS__); } while (0)
#define OLLAMA_COMPAT_LOG_ERROR(...) ollama_compat_log(__VA_ARGS__)
#else
#define OLLAMA_COMPAT_LOG_INFO(...)  do { LLAMA_LOG_INFO(__VA_ARGS__); } while (0)
#define OLLAMA_COMPAT_LOG_ERROR(...) LLAMA_LOG_ERROR(__VA_ARGS__)
#endif

double elapsed_ms(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
}

struct TransformTiming {
    uint64_t count;
    size_t bytes;
    double ms;
};

std::mutex g_transform_timing_mutex;
TransformTiming g_transform_timing = {};

TransformTiming record_transform_timing(size_t bytes, double ms) {
    std::lock_guard<std::mutex> lk(g_transform_timing_mutex);
    g_transform_timing.count++;
    g_transform_timing.bytes += bytes;
    g_transform_timing.ms += ms;
    return g_transform_timing;
}

bool compat_disabled() {
    const char * value = std::getenv("OLLAMA_LLAMA_CPP_COMPAT");
    return value && std::strcmp(value, "0") == 0;
}

bool clip_mmproj_embd_uses_projection_dim(const char * projector_type) {
    static constexpr const char * kProjectorTypes[] = {
        "gemma4a",
    };

    if (!projector_type) return false;
    for (const char * compat_type : kProjectorTypes) {
        if (std::strcmp(projector_type, compat_type) == 0) return true;
    }
    return false;
}

// Per-loader file path registry — set by translate_metadata, read by
// maybe_load_text_tensor so it can pass the path to load ops without a
// separate patch insertion in the model loader's load_all_data path.
std::mutex g_loader_path_mutex;
std::unordered_map<const llama_model_loader *, std::string> g_loader_paths;

void fix_glm4moelite_eog_token_ids(gguf_context * meta) {
    const bool need_eot = !has_key(meta, "tokenizer.ggml.eot_token_id");
    const bool need_eom = !has_key(meta, "tokenizer.ggml.eom_token_id");
    if (!need_eot && !need_eom) return;

    const int64_t kid = gguf_find_key(meta, "tokenizer.ggml.eos_token_ids");
    if (kid < 0 || gguf_get_kv_type(meta, kid) != GGUF_TYPE_ARRAY) return;

    std::vector<uint32_t> ids;
    const size_t n = gguf_get_arr_n(meta, kid);
    ids.reserve(n);

    if (gguf_get_arr_type(meta, kid) == GGUF_TYPE_INT32) {
        const auto * src = static_cast<const int32_t *>(gguf_get_arr_data(meta, kid));
        for (size_t i = 0; i < n; ++i) {
            if (src[i] >= 0) ids.push_back((uint32_t) src[i]);
        }
    } else if (gguf_get_arr_type(meta, kid) == GGUF_TYPE_UINT32) {
        const auto * src = static_cast<const uint32_t *>(gguf_get_arr_data(meta, kid));
        ids.assign(src, src + n);
    }

    if (need_eot && ids.size() >= 2) {
        gguf_set_val_u32(meta, "tokenizer.ggml.eot_token_id", ids[1]);
    }
    if (need_eom && ids.size() >= 3) {
        gguf_set_val_u32(meta, "tokenizer.ggml.eom_token_id", ids[2]);
    }
}

bool get_u32_kv(const gguf_context * meta, const char * key, uint32_t & out) {
    const int64_t kid = gguf_find_key(meta, key);
    if (kid < 0) return false;

    switch (gguf_get_kv_type(meta, kid)) {
        case GGUF_TYPE_UINT32:
            out = gguf_get_val_u32(meta, kid);
            return true;
        case GGUF_TYPE_INT32: {
            const int32_t v = gguf_get_val_i32(meta, kid);
            if (v < 0) return false;
            out = (uint32_t) v;
            return true;
        }
        default:
            return false;
    }
}

bool token_at_equals(const gguf_context * meta, size_t idx, const char * want) {
    const int64_t kid = gguf_find_key(meta, "tokenizer.ggml.tokens");
    if (kid < 0 || gguf_get_kv_type(meta, kid) != GGUF_TYPE_ARRAY) return false;
    if (gguf_get_arr_type(meta, kid) != GGUF_TYPE_STRING) return false;
    if (idx >= gguf_get_arr_n(meta, kid)) return false;

    const char * tok = gguf_get_arr_str(meta, kid, idx);
    return tok && std::strcmp(tok, want) == 0;
}

bool string_kv_equals(const gguf_context * meta, const char * key, const char * want) {
    const int64_t kid = gguf_find_key(meta, key);
    if (kid < 0 || gguf_get_kv_type(meta, kid) != GGUF_TYPE_STRING) return false;

    const char * value = gguf_get_val_str(meta, kid);
    return value && std::strcmp(value, want) == 0;
}

bool string_kv_contains(const gguf_context * meta, const char * key, const char * needle) {
    const int64_t kid = gguf_find_key(meta, key);
    if (kid < 0 || gguf_get_kv_type(meta, kid) != GGUF_TYPE_STRING) return false;

    const char * value = gguf_get_val_str(meta, kid);
    return value && std::strstr(value, needle) != nullptr;
}

bool string_kv_missing_or_default(const gguf_context * meta, const char * key) {
    const int64_t kid = gguf_find_key(meta, key);
    if (kid < 0) return true;
    if (gguf_get_kv_type(meta, kid) != GGUF_TYPE_STRING) return false;

    const char * value = gguf_get_val_str(meta, kid);
    return value && (std::strcmp(value, "") == 0 || std::strcmp(value, "default") == 0);
}

// =========================================================================
// gemma3 (text side)
// =========================================================================

constexpr const char * kGemma3ChatTemplate = R"jinja({{ bos_token }}{% if messages[0]['role'] == 'system' %}{{ raise_exception('System role not supported') }}{% endif %}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if (message['role'] == 'assistant') %}{% set role = 'model' %}{% else %}{% set role = message['role'] %}{% endif %}{{ '<start_of_turn>' + role + '\n' + message['content'] | trim + '<end_of_turn>\n' }}{% endfor %}{% if add_generation_prompt %}{{'<start_of_turn>model\n'}}{% endif %})jinja";

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

    OLLAMA_COMPAT_LOG_INFO("%s: detected Ollama-format gemma3 GGUF; applying compatibility fixes\n", __func__);

    // Some published files use nested rope key names. Copy them to the flat
    // names llama.cpp expects before injecting defaults.
    copy_f32_kv(meta, "gemma3.rope.global.freq_base", "gemma3.rope.freq_base");
    copy_f32_kv(meta, "gemma3.rope.local.freq_base",  "gemma3.rope.freq_base_swa");

    // Inject required KVs with their standard gemma3 defaults.
    inject_f32_if_missing(meta, "gemma3.attention.layer_norm_rms_epsilon", 1e-6f);
    inject_f32_if_missing(meta, "gemma3.rope.freq_base",                   1000000.0f);
    inject_f32_if_missing(meta, "gemma3.rope.freq_base_swa",               10000.0f);
    inject_str_if_missing(meta, "tokenizer.chat_template",                 kGemma3ChatTemplate);

    // Gemma3 4B/12B/27B ship with {type: "linear", factor: 8.0} rope scaling
    // in their HF config to extend the 16k trained context to 131072. Some
    // published files do not have these KVs. The 1B has no scaling; detect it
    // by context length.
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

    // Note: no RMSNorm weight shift needed. Published Gemma 3 blobs already
    // have the +1 shift baked in, matching llama.cpp's converter.
}

// =========================================================================
// gemma3n (text side — vocab mismatch between token_embd and per_layer_token_embd)
// =========================================================================
//
// Existing Gemma 3n files can store the multimodal vocab (262400 tokens) for
// the main token_embd and tokenizer arrays, while per-layer token embeddings
// only have the text vocab (262144 tokens). The loader expects both embedding
// tensors to have the same n_vocab, and reads n_vocab from the tokenizer.tokens
// array length, so the larger value wins and per_layer fails the dim check:
//
//   tensor 'per_layer_token_embd.weight' has wrong shape;
//     expected 8960, 262400, got 8960, 262144
//
// Fix (mirroring handle_gemma3): truncate the tokenizer arrays AND the
// token_embd tensor's vocab dim down to the per_layer count. The dropped
// 256 entries are multimodal special tokens (image/audio markers); the
// llama.cpp Gemma 3n text path does not use them.
//
// Note: tensor data isn't read at this point — the loader reads ggml_nbytes
// from the (newly-shrunk) tensor shape, so it just reads fewer rows from
// the same file offset. No load_op needed.

bool detect_ollama_gemma3n(const gguf_context * meta, const ggml_context * ctx) {
    const int64_t arch_kid = gguf_find_key(meta, "general.architecture");
    if (arch_kid < 0) return false;
    if (std::strcmp(gguf_get_val_str(meta, arch_kid), "gemma3n") != 0) return false;
    ggml_tensor * te = ggml_get_tensor(const_cast<ggml_context *>(ctx), "token_embd.weight");
    ggml_tensor * pe = ggml_get_tensor(const_cast<ggml_context *>(ctx), "per_layer_token_embd.weight");
    return te && pe && te->ne[1] != pe->ne[1];
}

void handle_gemma3n(const llama_model_loader * ml, gguf_context * meta, ggml_context * ctx) {
    (void) ml;
    if (!detect_ollama_gemma3n(meta, ctx)) return;

    OLLAMA_COMPAT_LOG_INFO("%s: detected Ollama-format gemma3n GGUF; normalizing tokenizer and truncating vocab to per_layer_token_embd size\n", __func__);

    ggml_tensor * pe = ggml_get_tensor(ctx, "per_layer_token_embd.weight");
    if (!pe) return;
    const uint32_t target_vocab = (uint32_t) pe->ne[1];

    gguf_set_val_str(meta, "tokenizer.ggml.model", "llama");

    if (ggml_tensor * t = ggml_get_tensor(ctx, "token_embd.weight")) {
        set_tensor_shape(t, {t->ne[0], target_vocab});
    }
    truncate_str_arr (meta, "tokenizer.ggml.tokens",     target_vocab);
    truncate_data_arr(meta, "tokenizer.ggml.scores",     GGUF_TYPE_FLOAT32, sizeof(float),   target_vocab);
    truncate_data_arr(meta, "tokenizer.ggml.token_type", GGUF_TYPE_INT32,   sizeof(int32_t), target_vocab);

    const int64_t tok_kid = gguf_find_key(meta, "tokenizer.ggml.tokens");
    const int64_t typ_kid = gguf_find_key(meta, "tokenizer.ggml.token_type");
    if (tok_kid >= 0 && typ_kid >= 0 &&
        gguf_get_kv_type(meta, typ_kid) == GGUF_TYPE_ARRAY &&
        gguf_get_arr_type(meta, typ_kid) == GGUF_TYPE_INT32) {
        const size_t n = std::min<size_t>(gguf_get_arr_n(meta, typ_kid), target_vocab);
        std::vector<int32_t> types(static_cast<const int32_t *>(gguf_get_arr_data(meta, typ_kid)),
                                   static_cast<const int32_t *>(gguf_get_arr_data(meta, typ_kid)) + n);
        for (size_t i = 0; i < n; ++i) {
            const char * tok = gguf_get_arr_str(meta, tok_kid, i);
            if (tok && std::strlen(tok) == 6 && tok[0] == '<' && tok[1] == '0' && tok[2] == 'x' && tok[5] == '>') {
                types[i] = 6; // LLAMA_TOKEN_TYPE_BYTE
            } else if (tok && std::strncmp(tok, "<unused", 7) == 0) {
                types[i] = 1; // LLAMA_TOKEN_TYPE_NORMAL
            }
        }
        gguf_set_arr_data(meta, "tokenizer.ggml.token_type", GGUF_TYPE_INT32, types.data(), types.size());
    }

    const int64_t eos_kid = gguf_find_key(meta, "tokenizer.ggml.eos_token_ids");
    if (eos_kid >= 0 && gguf_get_kv_type(meta, eos_kid) == GGUF_TYPE_ARRAY) {
        if (gguf_get_arr_type(meta, eos_kid) == GGUF_TYPE_INT32) {
            std::vector<int32_t> ids;
            const auto * src = static_cast<const int32_t *>(gguf_get_arr_data(meta, eos_kid));
            ids.assign(src, src + gguf_get_arr_n(meta, eos_kid));
            if (std::find(ids.begin(), ids.end(), 106) == ids.end()) ids.push_back(106);
            gguf_set_arr_data(meta, "tokenizer.ggml.eos_token_ids", GGUF_TYPE_INT32, ids.data(), ids.size());
        } else if (gguf_get_arr_type(meta, eos_kid) == GGUF_TYPE_UINT32) {
            std::vector<uint32_t> ids;
            const auto * src = static_cast<const uint32_t *>(gguf_get_arr_data(meta, eos_kid));
            ids.assign(src, src + gguf_get_arr_n(meta, eos_kid));
            if (std::find(ids.begin(), ids.end(), 106u) == ids.end()) ids.push_back(106u);
            gguf_set_arr_data(meta, "tokenizer.ggml.eos_token_ids", GGUF_TYPE_UINT32, ids.data(), ids.size());
        }
    } else {
        const int32_t ids[2] = { 1, 106 };
        gguf_set_arr_data(meta, "tokenizer.ggml.eos_token_ids", GGUF_TYPE_INT32, ids, 2);
    }
}

// =========================================================================
// embeddinggemma (text side — sentence-transformer dense projection)
// =========================================================================
//
// Existing embeddinggemma:300m files use general.architecture=gemma3 and
// two extra dense layers stored as `dense.0.weight` / `dense.1.weight`
// (the sentence-transformers post-pooling projection that maps the 768-dim
// pooled embedding through 768→3072→768 for the matryoshka head).
//
// llama.cpp loads this model under arch=gemma-embedding, which:
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

    OLLAMA_COMPAT_LOG_INFO("%s: detected Ollama-format embeddinggemma; translating to gemma-embedding\n", __func__);

    // Switch architecture so llama.cpp loads the embedding-specific code path
    // (no causal attention, dense_2/dense_3 loaded by name).
    arch_name = "gemma-embedding";
    gguf_set_val_str(meta, "general.architecture", "gemma-embedding");

    // Mirror gemma3.* hparams under the new arch prefix. rename_kv_prefix
    // copies (does not remove); the leftover gemma3.* keys are unused.
    rename_kv_prefix(meta, "gemma3.", "gemma-embedding.");
    gguf_remove_key(meta, "gemma-embedding.attention.causal");
    gguf_remove_key(meta, "gemma-embedding.attention.sliding_window_pattern");
    inject_f32_if_missing(meta, "gemma-embedding.rope.freq_base_swa", 10000.0f);

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
    rename_tensor(meta, ctx, "norm.weight", "output_norm.weight");
}

// =========================================================================
// snowflake-arctic-embed2 (text side)
// =========================================================================
//
// Some published GGUFs store tokenizer.ggml.precompiled_charsmap as an
// array of single-character base64 strings. llama.cpp now expects this KV as a
// raw int8/uint8 byte array and aborts while loading the vocab otherwise.

int base64_value(char c) {
    if (c >= 'A' && c <= 'Z') return c - 'A';
    if (c >= 'a' && c <= 'z') return c - 'a' + 26;
    if (c >= '0' && c <= '9') return c - '0' + 52;
    if (c == '+') return 62;
    if (c == '/') return 63;
    return -1;
}

bool decode_base64(const std::string & encoded, std::vector<uint8_t> & decoded) {
    int value = 0;
    int bits  = -8;
    decoded.clear();
    decoded.reserve(encoded.size() * 3 / 4);

    for (const char c : encoded) {
        if (c == '=') break;
        const int d = base64_value(c);
        if (d < 0) return false;

        value = (value << 6) | d;
        bits += 6;
        if (bits >= 0) {
            decoded.push_back((uint8_t) ((value >> bits) & 0xff));
            bits -= 8;
        }
    }

    return true;
}

void handle_snowflake_arctic_embed2(gguf_context * meta) {
    const int64_t arch_kid = gguf_find_key(meta, "general.architecture");
    if (arch_kid < 0 || std::strcmp(gguf_get_val_str(meta, arch_kid), "bert") != 0) return;

    const int64_t tok_kid = gguf_find_key(meta, "tokenizer.ggml.model");
    if (tok_kid < 0 || std::strcmp(gguf_get_val_str(meta, tok_kid), "t5") != 0) return;

    const char * key = "tokenizer.ggml.precompiled_charsmap";
    const int64_t kid = gguf_find_key(meta, key);
    if (kid < 0 || gguf_get_kv_type(meta, kid) != GGUF_TYPE_ARRAY) return;
    if (gguf_get_arr_type(meta, kid) != GGUF_TYPE_STRING) return;

    const size_t n = gguf_get_arr_n(meta, kid);
    std::string encoded;
    encoded.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        const char * s = gguf_get_arr_str(meta, kid, i);
        if (!s || std::strlen(s) != 1) {
            OLLAMA_COMPAT_LOG_ERROR("%s: unexpected precompiled charsmap entry length at index %zu\n",
                                    __func__, i);
            return;
        }
        encoded.push_back(s[0]);
    }

    std::vector<uint8_t> decoded;
    if (!decode_base64(encoded, decoded) || decoded.empty()) {
        OLLAMA_COMPAT_LOG_ERROR("%s: failed to decode precompiled charsmap\n", __func__);
        return;
    }

    gguf_set_arr_data(meta, key, GGUF_TYPE_UINT8, decoded.data(), decoded.size());
    OLLAMA_COMPAT_LOG_INFO("%s: converted tokenizer precompiled charsmap to byte array\n", __func__);
}

// =========================================================================
// qwen35moe (text side)
// =========================================================================

std::vector<std::string> qwen_ssm_dt_tensors(const gguf_context * meta) {
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
    return targets;
}

void rename_qwen_ssm_dt_bias_tensors(gguf_context * meta, ggml_context * ctx) {
    for (const auto & from : qwen_ssm_dt_tensors(meta)) {
        rename_tensor(meta, ctx, from.c_str(), (from + ".bias").c_str());
    }
}

void collapse_u32_array_to_max(gguf_context * meta, const std::string & key, uint32_t fallback) {
    const int64_t kid = gguf_find_key(meta, key.c_str());
    if (kid < 0 || gguf_get_kv_type(meta, kid) != GGUF_TYPE_ARRAY) return;

    const size_t n = gguf_get_arr_n(meta, kid);
    const auto * arr = static_cast<const uint32_t *>(gguf_get_arr_data(meta, kid));
    uint32_t max_value = 0;
    for (size_t i = 0; i < n; ++i) if (arr[i] > max_value) max_value = arr[i];
    if (max_value == 0) max_value = fallback;
    if (max_value == 0) return;

    gguf_remove_key (meta, key.c_str());
    gguf_set_val_u32(meta, key.c_str(), max_value);
}

void set_u32_kv(gguf_context * meta, const std::string & key, uint32_t value) {
    gguf_remove_key(meta, key.c_str());
    gguf_set_val_u32(meta, key.c_str(), value);
}

bool parse_tensor_index(const std::string & name, const char * prefix, uint32_t & index, std::string * suffix) {
    const size_t prefix_len = std::strlen(prefix);
    if (name.compare(0, prefix_len, prefix) != 0) return false;

    const char * start = name.c_str() + prefix_len;
    char * end = nullptr;
    errno = 0;
    const unsigned long value = std::strtoul(start, &end, 10);
    if (errno != 0 || end == start || *end != '.' || value > UINT32_MAX) return false;

    index = static_cast<uint32_t>(value);
    if (suffix) *suffix = end + 1;
    return true;
}

uint32_t qwen35_text_block_count(const gguf_context * meta, const char * arch_prefix) {
    uint32_t max_block = 0;
    bool found = false;
    const int64_t n = gguf_get_n_tensors(meta);
    for (int64_t i = 0; i < n; ++i) {
        const std::string name(gguf_get_tensor_name(meta, i));
        uint32_t index = 0;
        if (!parse_tensor_index(name, "blk.", index, nullptr)) continue;
        if (!found || index > max_block) max_block = index;
        found = true;
    }
    if (found) return max_block + 1;

    const std::string key = std::string(arch_prefix) + ".block_count";
    const int64_t kid = gguf_find_key(meta, key.c_str());
    if (kid >= 0 && gguf_get_kv_type(meta, kid) == GGUF_TYPE_UINT32) {
        return gguf_get_val_u32(meta, kid);
    }
    return 0;
}

uint32_t qwen35_mtp_layer_count(const gguf_context * meta) {
    uint32_t max_mtp_layer = 0;
    bool found_layer = false;
    bool found_mtp = false;

    const int64_t n = gguf_get_n_tensors(meta);
    for (int64_t i = 0; i < n; ++i) {
        const std::string name(gguf_get_tensor_name(meta, i));
        if (name.compare(0, 4, "mtp.") != 0) continue;
        found_mtp = true;

        uint32_t index = 0;
        std::string suffix;
        if (!parse_tensor_index(name, "mtp.layers.", index, &suffix) || suffix.empty()) continue;
        if (!found_layer || index > max_mtp_layer) max_mtp_layer = index;
        found_layer = true;
    }

    if (found_layer) return max_mtp_layer + 1;
    return found_mtp ? 1 : 0;
}

bool qwen35_has_native_mtp_tensors(const gguf_context * meta) {
    const int64_t n = gguf_get_n_tensors(meta);
    for (int64_t i = 0; i < n; ++i) {
        const std::string name(gguf_get_tensor_name(meta, i));
        if (name.compare(0, 4, "blk.") == 0 && name.find(".nextn.") != std::string::npos) return true;
    }
    return false;
}

bool string_ends_with(const std::string & s, const char * suffix) {
    const size_t n = std::strlen(suffix);
    return s.size() >= n && s.compare(s.size() - n, n, suffix) == 0;
}

bool qwen35_should_shift_norm_after_rename(const std::string & name) {
    if (string_ends_with(name, ".ssm_norm.weight")) return false;
    return string_ends_with(name, "_norm.weight") ||
        string_ends_with(name, ".nextn.enorm.weight") ||
        string_ends_with(name, ".nextn.hnorm.weight");
}

bool register_qwen35_norm_shift_load(gguf_context * meta, ggml_context * ctx,
                                     const char * from, const std::string & to) {
    const int64_t tid = gguf_find_tensor(meta, from);
    if (tid < 0) return false;

    ggml_tensor * src = ggml_get_tensor(ctx, from);
    if (!src) return false;

    const size_t src_offset = tensor_file_offset(meta, from);
    const size_t src_size   = gguf_get_tensor_size(meta, tid);
    const ggml_type src_type = src->type;
    const size_t n_elem = (size_t) ggml_nelements(src);

    rename_tensor(meta, ctx, from, to.c_str());
    ggml_tensor * dst = ggml_get_tensor(ctx, to.c_str());
    if (!dst) return false;

    set_tensor_type(dst, GGML_TYPE_F32);
    register_load_op(to, LoadOp{
        [src_offset, src_size, src_type, n_elem](const char * path, void * out, size_t out_size) {
            if (out_size != n_elem * sizeof(float)) return false;

            float * dst = static_cast<float *>(out);
            if (src_type == GGML_TYPE_F32) {
                if (src_size != n_elem * sizeof(float)) return false;
                std::vector<uint8_t> src(src_size);
                if (!read_at(path, src_offset, src.data(), src.size())) return false;
                const float * fp = reinterpret_cast<const float *>(src.data());
                for (size_t i = 0; i < n_elem; ++i) dst[i] = fp[i] + 1.0f;
                return true;
            }

            std::vector<uint8_t> src(src_size);
            if (!read_at(path, src_offset, src.data(), src.size())) return false;
            const auto * traits = ggml_get_type_traits(src_type);
            if (!traits || !traits->to_float) return false;
            traits->to_float(src.data(), dst, (int64_t) n_elem);
            for (size_t i = 0; i < n_elem; ++i) dst[i] += 1.0f;
            return true;
        },
        "F32 add-one norm shift",
    });
    return true;
}

bool rename_qwen35_mtp_tensor(gguf_context * meta, ggml_context * ctx, const char * from, const std::string & to) {
    if (gguf_find_tensor(meta, from) < 0 || gguf_find_tensor(meta, to.c_str()) >= 0) return false;
    if (qwen35_should_shift_norm_after_rename(to)) {
        return register_qwen35_norm_shift_load(meta, ctx, from, to);
    }
    rename_tensor(meta, ctx, from, to.c_str());
    return false;
}

bool qwen35moe_mtp_expert_source(const std::string & name, uint32_t mtp_index,
                                 const char * suffix, uint32_t & expert) {
    char prefix[128];
    std::snprintf(prefix, sizeof(prefix), "mtp.layers.%u.mlp.experts.", mtp_index);
    const size_t prefix_len = std::strlen(prefix);
    if (name.compare(0, prefix_len, prefix) != 0) return false;

    const char * start = name.c_str() + prefix_len;
    char * end = nullptr;
    errno = 0;
    const unsigned long value = std::strtoul(start, &end, 10);
    if (errno != 0 || end == start || *end != '.' || value > UINT32_MAX) return false;
    if (std::strcmp(end + 1, suffix) != 0) return false;

    expert = static_cast<uint32_t>(value);
    return true;
}

std::vector<std::pair<uint32_t, std::string>> qwen35moe_mtp_expert_sources(
        const gguf_context * meta, uint32_t mtp_index, const char * suffix) {
    std::vector<std::pair<uint32_t, std::string>> sources;
    const int64_t n = gguf_get_n_tensors(meta);
    for (int64_t i = 0; i < n; ++i) {
        const std::string name(gguf_get_tensor_name(meta, i));
        uint32_t expert = 0;
        if (!qwen35moe_mtp_expert_source(name, mtp_index, suffix, expert)) continue;
        sources.emplace_back(expert, name);
    }

    std::sort(sources.begin(), sources.end(),
              [](const auto & a, const auto & b) { return a.first < b.first; });
    return sources;
}

bool register_qwen35moe_mtp_expert_merge(gguf_context * meta, ggml_context * ctx,
                                         uint32_t mtp_index, uint32_t block,
                                         const char * src_suffix,
                                         const char * dst_suffix) {
    const auto sources = qwen35moe_mtp_expert_sources(meta, mtp_index, src_suffix);
    if (sources.empty()) return false;

    std::vector<std::string> names;
    names.reserve(sources.size());
    for (size_t i = 0; i < sources.size(); ++i) {
        if (sources[i].first != static_cast<uint32_t>(i)) {
            OLLAMA_COMPAT_LOG_ERROR("%s: non-contiguous qwen35moe MTP experts for layer %u suffix %s\n",
                                    __func__, mtp_index, src_suffix);
            return false;
        }
        names.push_back(sources[i].second);
    }

    ggml_tensor * first = ggml_get_tensor(ctx, names[0].c_str());
    if (!first || first->ne[2] != 1 || first->ne[3] != 1) return false;
    const int64_t ne0 = first->ne[0];
    const int64_t ne1 = first->ne[1];
    const ggml_type type = first->type;

    for (const auto & name : names) {
        ggml_tensor * t = ggml_get_tensor(ctx, name.c_str());
        if (!t || t->type != type || t->ne[0] != ne0 || t->ne[1] != ne1 ||
                t->ne[2] != 1 || t->ne[3] != 1) {
            OLLAMA_COMPAT_LOG_ERROR("%s: inconsistent qwen35moe MTP expert tensor shape/type for %s\n",
                                    __func__, name.c_str());
            return false;
        }
    }

    char dest[GGML_MAX_NAME];
    std::snprintf(dest, sizeof(dest), "blk.%u.%s", block, dst_suffix);

    register_concat_load(meta, dest, names);
    rename_tensor(meta, ctx, names[0].c_str(), dest);
    if (ggml_tensor * t = ggml_get_tensor(ctx, dest)) {
        set_tensor_shape(t, {ne0, ne1, (int64_t) names.size()});
    }
    return true;
}

bool merge_qwen35moe_mtp_expert_tensors(gguf_context * meta, ggml_context * ctx,
                                        uint32_t base_block, uint32_t nextn) {
    bool merged_any = false;
    for (uint32_t i = 0; i < nextn; ++i) {
        const uint32_t block = base_block + i;
        const bool gate = register_qwen35moe_mtp_expert_merge(meta, ctx, i, block,
                                                              "gate_proj.weight", "ffn_gate_exps.weight");
        const bool up = register_qwen35moe_mtp_expert_merge(meta, ctx, i, block,
                                                            "up_proj.weight", "ffn_up_exps.weight");
        const bool down = register_qwen35moe_mtp_expert_merge(meta, ctx, i, block,
                                                              "down_proj.weight", "ffn_down_exps.weight");
        if (gate || up || down) {
            merged_any = true;
            if (!gate || !up || !down) {
                OLLAMA_COMPAT_LOG_ERROR("%s: incomplete qwen35moe MTP expert merge for layer %u\n", __func__, i);
            }
        }
    }
    return merged_any;
}

bool rename_qwen35_mtp_tensors(gguf_context * meta, ggml_context * ctx,
                               const char * arch_prefix,
                               uint32_t base_block, uint32_t nextn) {
    const bool is_moe = std::strcmp(arch_prefix, "qwen35moe") == 0;
    bool registered_load_transform = false;
    if (is_moe) {
        registered_load_transform = merge_qwen35moe_mtp_expert_tensors(meta, ctx, base_block, nextn);
    }

    std::vector<std::pair<std::string, std::string>> layer_renames;
    const int64_t n = gguf_get_n_tensors(meta);
    for (int64_t i = 0; i < n; ++i) {
        const std::string name(gguf_get_tensor_name(meta, i));
        uint32_t mtp_index = 0;
        std::string suffix;
        if (!parse_tensor_index(name, "mtp.layers.", mtp_index, &suffix) || suffix.empty()) continue;
        if (mtp_index >= nextn) continue;
        if (is_moe && suffix.compare(0, std::strlen("mlp.experts."), "mlp.experts.") == 0) continue;

        char dest[GGML_MAX_NAME];
        std::snprintf(dest, sizeof(dest), "blk.%u.%s", base_block + mtp_index, suffix.c_str());
        layer_renames.emplace_back(name, dest);
    }
    for (const auto & [from, to] : layer_renames) {
        registered_load_transform |= rename_qwen35_mtp_tensor(meta, ctx, from.c_str(), to);
    }

    if (nextn != 1) {
        if (gguf_find_tensor(meta, "mtp.fc.weight") >= 0 ||
            gguf_find_tensor(meta, "mtp.pre_fc_norm_embedding.weight") >= 0 ||
            gguf_find_tensor(meta, "mtp.pre_fc_norm_hidden.weight") >= 0 ||
            gguf_find_tensor(meta, "mtp.embed_tokens.weight") >= 0 ||
            gguf_find_tensor(meta, "mtp.shared_head.head.weight") >= 0 ||
            gguf_find_tensor(meta, "mtp.shared_head.norm.weight") >= 0 ||
            gguf_find_tensor(meta, "mtp.norm.weight") >= 0) {
            OLLAMA_COMPAT_LOG_ERROR("%s: cannot duplicate shared MTP tensors across %u layers\n", __func__, nextn);
        }
        return registered_load_transform;
    }

    auto nextn_name = [base_block](const char * suffix) {
        char dest[GGML_MAX_NAME];
        std::snprintf(dest, sizeof(dest), "blk.%u.%s", base_block, suffix);
        return std::string(dest);
    };

    registered_load_transform |= rename_qwen35_mtp_tensor(meta, ctx, "mtp.fc.weight", nextn_name("nextn.eh_proj.weight"));
    registered_load_transform |= rename_qwen35_mtp_tensor(meta, ctx, "mtp.pre_fc_norm_embedding.weight", nextn_name("nextn.enorm.weight"));
    registered_load_transform |= rename_qwen35_mtp_tensor(meta, ctx, "mtp.pre_fc_norm_hidden.weight", nextn_name("nextn.hnorm.weight"));
    registered_load_transform |= rename_qwen35_mtp_tensor(meta, ctx, "mtp.embed_tokens.weight", nextn_name("nextn.embed_tokens.weight"));
    registered_load_transform |= rename_qwen35_mtp_tensor(meta, ctx, "mtp.shared_head.head.weight", nextn_name("nextn.shared_head_head.weight"));
    registered_load_transform |= rename_qwen35_mtp_tensor(meta, ctx, "mtp.shared_head.norm.weight", nextn_name("nextn.shared_head_norm.weight"));
    registered_load_transform |= rename_qwen35_mtp_tensor(meta, ctx, "mtp.norm.weight", nextn_name("nextn.shared_head_norm.weight"));
    return registered_load_transform;
}

bool apply_qwen35_mtp_fixes(gguf_context * meta, ggml_context * ctx, const char * arch_prefix) {
    const uint32_t nextn = qwen35_mtp_layer_count(meta);
    if (nextn == 0) return false;
    if (qwen35_has_native_mtp_tensors(meta)) return false;

    const uint32_t base_block = qwen35_text_block_count(meta, arch_prefix);
    if (base_block == 0) {
        OLLAMA_COMPAT_LOG_ERROR("%s: cannot infer qwen3.5 text block count for MTP translation\n", __func__);
        return false;
    }

    set_u32_kv(meta, std::string(arch_prefix) + ".nextn_predict_layers", nextn);
    set_u32_kv(meta, std::string(arch_prefix) + ".block_count", base_block + nextn);
    return rename_qwen35_mtp_tensors(meta, ctx, arch_prefix, base_block, nextn);
}

// Shared text-side fixes for qwen35 / qwen35moe published-model layouts.
// Both arches use the same SSM-hybrid + M-RoPE + MTP+vision-monolithic
// layout differences; only the arch name and KV prefix differ.
void apply_qwen35_text_fixes(const llama_model_loader * ml, gguf_context * meta,
                             ggml_context * ctx, const char * arch_prefix) {
    auto kv = [arch_prefix](const char * suffix) {
        return std::string(arch_prefix) + suffix;
    };

    // 1. attention.head_count_kv — llama.cpp expects UINT32; published files
    //    can store an array (one entry per layer, 0 for SSM layers, 2/4 for attention).
    //    Collapse to the max non-zero value.
    {
        const std::string key = kv(".attention.head_count_kv");
        collapse_u32_array_to_max(meta, key, 2);
    }

    // 2. rope.dimension_sections — llama.cpp expects a 4-element array
    //    (M-RoPE convention); published files can store 3 elements. Pad with a trailing 0.
    {
        const std::string key = kv(".rope.dimension_sections");
        const int64_t kid = gguf_find_key(meta, key.c_str());
        if (kid >= 0 && gguf_get_arr_n(meta, kid) == 3) {
            const auto * src = static_cast<const int32_t *>(gguf_get_arr_data(meta, kid));
            const int32_t padded[4] = { src[0], src[1], src[2], 0 };
            gguf_set_arr_data(meta, key.c_str(), GGUF_TYPE_INT32, padded, 4);
        }
    }

    // 3. Tensor rename: `blk.N.ssm_dt` maps to llama.cpp's
    //    `blk.N.ssm_dt.bias` (same shape).
    rename_qwen_ssm_dt_bias_tensors(meta, ctx);

    // 4. Translate legacy embedded MTP tensors to llama.cpp's qwen3.5 layout.
    if (apply_qwen35_mtp_fixes(meta, ctx, arch_prefix)) {
        disable_mmap_for(ml);
    }

    // 5. Drop embedded vision/projector tensors, plus any untranslated legacy
    //    MTP tensors, from the text loader.
    add_skip_prefix(ml, "v.");
    add_skip_prefix(ml, "mm.");
    add_skip_prefix(ml, "mtp.");
}

bool detect_ollama_qwen35moe(const gguf_context * meta, const ggml_context * ctx) {
    const int64_t arch_kid = gguf_find_key(meta, "general.architecture");
    if (arch_kid < 0) return false;
    if (std::strcmp(gguf_get_val_str(meta, arch_kid), "qwen35moe") != 0) return false;

    // Published-model markers. llama.cpp-converted qwen35moe files have none
    // of these: vision KVs live in a separate mmproj, MTP tensors are dropped,
    // head_count_kv is a scalar, and the extra rope / ssm / feed_forward KVs
    // are either absent or stored differently.
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
    OLLAMA_COMPAT_LOG_INFO("%s: detected Ollama-format qwen35moe GGUF; applying compatibility fixes\n", __func__);
    apply_qwen35_text_fixes(ml, meta, ctx, "qwen35moe");
}

// =========================================================================
// qwen35 (text side — non-MoE, e.g. qwen3.5:9b)
// =========================================================================
//
// Same layout differences as qwen35moe but the arch name has no "moe" suffix.
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
    OLLAMA_COMPAT_LOG_INFO("%s: detected Ollama-format qwen35 GGUF; applying compatibility fixes\n", __func__);
    apply_qwen35_text_fixes(ml, meta, ctx, "qwen35");
}

// =========================================================================
// qwen3next (text side)
// =========================================================================

bool detect_ollama_qwen3next(const gguf_context * meta) {
    const int64_t arch_kid = gguf_find_key(meta, "general.architecture");
    if (arch_kid < 0) return false;
    if (std::strcmp(gguf_get_val_str(meta, arch_kid), "qwen3next") != 0) return false;

    return !qwen_ssm_dt_tensors(meta).empty();
}

void handle_qwen3next(gguf_context * meta, ggml_context * ctx) {
    if (!detect_ollama_qwen3next(meta)) return;
    OLLAMA_COMPAT_LOG_INFO("%s: detected qwen3next GGUF with ssm_dt tensors; applying compatibility fixes\n", __func__);
    collapse_u32_array_to_max(meta, "qwen3next.attention.head_count_kv", 0);
    rename_qwen_ssm_dt_bias_tensors(meta, ctx);
}

// =========================================================================
// gemma4 (text side)
// =========================================================================
//
// Same arch name on both sides. Existing published models can use a monolithic
// GGUF that embeds the vision encoder + audio encoder + projector inline.
// Split expert gate/up tensors are valid in llama.cpp GGUFs, so they are not
// sufficient to identify an existing published Ollama model.

bool detect_ollama_gemma4(const gguf_context * meta, const ggml_context * ctx) {
    const int64_t arch_kid = gguf_find_key(meta, "general.architecture");
    if (arch_kid < 0) return false;
    if (std::strcmp(gguf_get_val_str(meta, arch_kid), "gemma4") != 0) return false;
    return any_tensor_with_prefix(ctx, "a.")
        || any_tensor_with_prefix(ctx, "v.")
        || any_tensor_with_prefix(ctx, "mm.")
        || any_tensor_with_prefix(ctx, "model.vision_tower.")
        || string_kv_equals(meta, "tokenizer.ggml.model", "llama");
}

bool has_gemma4_split_expert_sidecars(ggml_context * ctx, uint32_t b) {
    char gate_scale[64], gate_input_scale[64], up_scale[64], up_input_scale[64];
    std::snprintf(gate_scale,       sizeof(gate_scale),       "blk.%u.ffn_gate_exps.scale",       b);
    std::snprintf(gate_input_scale, sizeof(gate_input_scale), "blk.%u.ffn_gate_exps.input_scale", b);
    std::snprintf(up_scale,         sizeof(up_scale),         "blk.%u.ffn_up_exps.scale",         b);
    std::snprintf(up_input_scale,   sizeof(up_input_scale),   "blk.%u.ffn_up_exps.input_scale",   b);

    return ggml_get_tensor(ctx, gate_scale)
        || ggml_get_tensor(ctx, gate_input_scale)
        || ggml_get_tensor(ctx, up_scale)
        || ggml_get_tensor(ctx, up_input_scale);
}

bool register_gemma4_moe_gate_up_load(gguf_context * meta,
                                      ggml_context * ctx,
                                      const char * gate_n,
                                      const char * up_n,
                                      const char * gate_up_n) {
    ggml_tensor * gate = ggml_get_tensor(ctx, gate_n);
    ggml_tensor * up   = ggml_get_tensor(ctx, up_n);
    if (!gate || !up) return false;
    if (ggml_get_tensor(ctx, gate_up_n)) return false;
    if (gate->type != up->type) return false;
    if (gate->ne[0] != up->ne[0] || gate->ne[1] != up->ne[1] || gate->ne[2] != up->ne[2]) return false;
    if (gate->ne[2] <= 0) return false;

    const int64_t n_expert = gate->ne[2];
    const int64_t n_embd   = gate->ne[0];
    const int64_t n_ff     = gate->ne[1];

    const int64_t gate_id = gguf_find_tensor(meta, gate_n);
    const int64_t up_id   = gguf_find_tensor(meta, up_n);
    if (gate_id < 0 || up_id < 0) return false;

    const size_t gate_offset = tensor_file_offset(meta, gate_n);
    const size_t up_offset   = tensor_file_offset(meta, up_n);
    const size_t gate_size   = gguf_get_tensor_size(meta, gate_id);
    const size_t up_size     = gguf_get_tensor_size(meta, up_id);
    if (gate_size != up_size || gate_size % (size_t) n_expert != 0) return false;

    const size_t expert_size = gate_size / (size_t) n_expert;
    register_load_op(gate_up_n, LoadOp{
        [gate_offset, up_offset, expert_size, n_expert](const char * path, void * dst, size_t dst_size) {
            if (dst_size != expert_size * (size_t) n_expert * 2) return false;

            uint8_t * p = static_cast<uint8_t *>(dst);
            for (int64_t e = 0; e < n_expert; ++e) {
                const size_t off = (size_t) e * expert_size;
                if (!read_at(path, gate_offset + off, p, expert_size)) return false;
                p += expert_size;
                if (!read_at(path, up_offset + off, p, expert_size)) return false;
                p += expert_size;
            }
            return true;
        },
        "gemma4 MoE gate/up interleave",
    });

    rename_tensor(meta, ctx, gate_n, gate_up_n);
    if (ggml_tensor * t = ggml_get_tensor(ctx, gate_up_n)) {
        set_tensor_shape(t, {n_embd, n_ff * 2, n_expert});
    }
    return true;
}

void handle_gemma4(const llama_model_loader * ml, gguf_context * meta, ggml_context * ctx) {
    if (!detect_ollama_gemma4(meta, ctx)) return;

    OLLAMA_COMPAT_LOG_INFO("%s: detected Ollama-format gemma4 GGUF; applying compatibility fixes\n", __func__);

    // Tokenizer fix: published Gemma 4 GGUFs can write
    // `tokenizer.ggml.model = 'llama'` (SPM), but Gemma 4 uses BPE. GGUFs
    // produced by the llama.cpp converter
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
    add_skip_prefix(ml, "model.vision_tower.");

    const int64_t n_blocks_key = gguf_find_key(meta, "gemma4.block_count");
    const uint32_t n_blocks = n_blocks_key >= 0 ? gguf_get_val_u32(meta, n_blocks_key) : 0;
    bool transformed_moe = false;
    for (uint32_t b = 0; b < n_blocks; ++b) {
        char gate[64], up[64], gate_up[64], scale[64], down_scale[64];
        std::snprintf(gate,    sizeof(gate),    "blk.%u.ffn_gate_exps.weight",    b);
        std::snprintf(up,      sizeof(up),      "blk.%u.ffn_up_exps.weight",      b);
        std::snprintf(gate_up, sizeof(gate_up), "blk.%u.ffn_gate_up_exps.weight", b);
        std::snprintf(scale,      sizeof(scale),      "blk.%u.ffn_gate_inp.per_expert_scale", b);
        std::snprintf(down_scale, sizeof(down_scale), "blk.%u.ffn_down_exps.scale",           b);

        const bool can_fuse_gate_up = !has_gemma4_split_expert_sidecars(ctx, b);
        if (can_fuse_gate_up && register_gemma4_moe_gate_up_load(meta, ctx, gate, up, gate_up)) {
            add_skip_prefix(ml, up);
            transformed_moe = true;
        }
        rename_tensor(meta, ctx, scale, down_scale);
    }
    if (transformed_moe) {
        disable_mmap_for(ml);
    }
}

// =========================================================================
// deepseek-ocr (text side)
// =========================================================================
//
// Existing files use arch name "deepseekocr" / KV prefix "deepseekocr.*".
// llama.cpp uses "deepseek2-ocr" (with hyphen) / "deepseek2-ocr.*".
//
// Aside from the prefix rename:
//   * Inject `expert_feed_forward_length` from the per-expert ffn_down_exps
//     shape (the value is the inner FFN dim of one expert, 896 for the 3B model).
//   * Inject `expert_shared_count` from the ffn_down_shexp shape. The shared
//     experts use the same FFN dim as regular experts, so count =
//     shexp_dim / expert_feed_forward_length.
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

    OLLAMA_COMPAT_LOG_INFO("%s: detected Ollama-format deepseekocr GGUF; applying compatibility fixes\n", __func__);

    gguf_set_val_str(meta, "general.architecture", "deepseek2-ocr");
    rename_kv_prefix(meta, "deepseekocr.", "deepseek2-ocr.");
    arch_name = "deepseek2-ocr";

    // Inject defaults needed by the llama.cpp loader.
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

    OLLAMA_COMPAT_LOG_INFO("%s: detected Ollama-format nemotron_h_moe GGUF; applying compatibility fixes\n", __func__);

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

    // Rename the latent projection tensors to llama.cpp's naming (no-op when
    // the file has no latent tensors).
    rename_tensors_containing(meta, ctx, ".ffn_latent_in",  ".ffn_latent_down");
    rename_tensors_containing(meta, ctx, ".ffn_latent_out", ".ffn_latent_up");

    // Drop MTP (Multi-Token Prediction) tensors. Existing files can include
    // one tensor per expert (`mtp.layers.X.mixer.experts.Y.{up,down}_proj`);
    // the nemotron_h_moe loader does not claim these tensors.
    add_skip_prefix(ml, "mtp.");
}

// =========================================================================
// nemotron_h_omni (text side)
// =========================================================================
//
// The published nemotron3:33b GGUF is a unified text+vision+audio
// `nemotron_h_omni` blob. llama.cpp currently loads the runnable pieces as
// `nemotron_h_moe` text plus a `clip` / `nemotron_v2_vl` projector. Audio is
// intentionally hidden until llama.cpp can skip or load it safely.

bool detect_ollama_nemotron_h_omni(const gguf_context * meta, const ggml_context * ctx) {
    const int64_t arch_kid = gguf_find_key(meta, "general.architecture");
    if (arch_kid < 0) return false;
    if (std::strcmp(gguf_get_val_str(meta, arch_kid), "nemotron_h_omni") != 0) return false;
    return any_tensor_with_prefix(ctx, "v.")
        || any_tensor_with_prefix(ctx, "a.")
        || any_tensor_with_prefix(ctx, "mm.");
}

const char * nemotron_h_omni_text_arch(const gguf_context * meta) {
    const int64_t expert_count_kid = gguf_find_key(meta, "nemotron_h_omni.expert_count");
    const int64_t expert_used_kid  = gguf_find_key(meta, "nemotron_h_omni.expert_used_count");
    const bool has_experts =
        (expert_count_kid >= 0 && gguf_get_val_u32(meta, expert_count_kid) > 0) ||
        (expert_used_kid  >= 0 && gguf_get_val_u32(meta, expert_used_kid)  > 0);
    return has_experts ? "nemotron_h_moe" : "nemotron_h";
}

void copy_nemotron_h_omni_text_kvs(gguf_context * meta, const char * text_arch) {
    const char * old_prefix = "nemotron_h_omni.";
    const size_t old_len = std::strlen(old_prefix);

    std::vector<std::string> matches;
    const int64_t n = gguf_get_n_kv(meta);
    for (int64_t i = 0; i < n; ++i) {
        const char * k = gguf_get_key(meta, i);
        if (std::strncmp(k, old_prefix, old_len) != 0) continue;
        const char * suffix = k + old_len;
        if (std::strncmp(suffix, "vision.", 7) == 0) continue;
        if (std::strncmp(suffix, "audio.", 6) == 0) continue;
        matches.emplace_back(k);
    }

    const std::string new_prefix = std::string(text_arch) + ".";
    for (const auto & old_key : matches) {
        copy_kv(meta, old_key.c_str(),
                (new_prefix + old_key.substr(old_len)).c_str());
    }
}

void handle_nemotron_h_omni(const llama_model_loader * ml,
                            gguf_context * meta,
                            ggml_context * ctx,
                            std::string & arch_name) {
    if (!detect_ollama_nemotron_h_omni(meta, ctx)) return;

    const char * text_arch = nemotron_h_omni_text_arch(meta);
    OLLAMA_COMPAT_LOG_INFO("%s: detected Ollama-format nemotron_h_omni GGUF; translating text side to %s\n",
                           __func__, text_arch);

    gguf_set_val_str(meta, "general.architecture", text_arch);
    arch_name = text_arch;
    copy_nemotron_h_omni_text_kvs(meta, text_arch);

    // Hide embedded audio + vision + projector tensors from the text loader.
    add_skip_prefix(ml, "a.");
    add_skip_prefix(ml, "v.");
    add_skip_prefix(ml, "mm.");

    // Reuse the existing nemotron_h_moe fixes if this unified model also has
    // latent FFN or MTP tensors in future variants.
    if (arch_name == "nemotron_h_moe") handle_nemotron_h_moe(ml, meta, ctx);
}

// =========================================================================
// llama3-family metadata (text side)
// =========================================================================
//
// Some published Llama 3-family GGUFs omit the pre-tokenizer and leave the
// scalar EOS/EOT pointed at <|end_of_text|>. Instruct variants terminate turns
// with <|eot_id|>; with that metadata, llama-server can miss the stop token
// and run until the request timeout.

bool detect_ollama_llama3_metadata_gap(const gguf_context * meta) {
    if (!string_kv_equals(meta, "general.architecture", "llama")) return false;

    if (!token_at_equals(meta, 128009, "<|eot_id|>")) return false;

    const bool has_llama3_chat_markers =
        token_at_equals(meta, 128006, "<|start_header_id|>") ||
        string_kv_contains(meta, "tokenizer.chat_template", "<|start_header_id|>") ||
        string_kv_contains(meta, "tokenizer.chat_template", "<|eot_id|>");

    const bool missing_or_default_pre = string_kv_missing_or_default(meta, "tokenizer.ggml.pre");

    uint32_t eos = 0;
    const bool end_of_text_eos = get_u32_kv(meta, "tokenizer.ggml.eos_token_id", eos) && eos == 128001;

    return has_llama3_chat_markers && (missing_or_default_pre || end_of_text_eos);
}

void handle_llama3_metadata(gguf_context * meta) {
    if (!detect_ollama_llama3_metadata_gap(meta)) return;

    OLLAMA_COMPAT_LOG_INFO("%s: detected Llama 3 tokenizer metadata gap; applying compatibility fixes\n", __func__);

    if (string_kv_missing_or_default(meta, "tokenizer.ggml.pre")) {
        gguf_set_val_str(meta, "tokenizer.ggml.pre", "llama-bpe");
    }

    uint32_t eos = 0;
    if (!get_u32_kv(meta, "tokenizer.ggml.eos_token_id", eos) || eos == 128001) {
        gguf_set_val_u32(meta, "tokenizer.ggml.eos_token_id", 128009);
    }

    uint32_t eot = 0;
    if (!get_u32_kv(meta, "tokenizer.ggml.eot_token_id", eot) || eot == 128001) {
        gguf_set_val_u32(meta, "tokenizer.ggml.eot_token_id", 128009);
    }

    const int32_t eos_ids[1] = {128009};
    gguf_set_arr_data(meta, "tokenizer.ggml.eos_token_ids", GGUF_TYPE_INT32, eos_ids, 1);
}

// =========================================================================
// llama4 (text side)
// =========================================================================
//
// Same arch name on both sides. Existing published models use a monolithic GGUF that
// embeds the vision encoder + projector inline. Text-side KVs/tensor
// names already match llama.cpp; only fix is to hide `v.*`/`mm.*` from
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

    OLLAMA_COMPAT_LOG_INFO("%s: detected Ollama-format llama4 GGUF; applying compatibility fixes\n", __func__);

    add_skip_prefix(ml, "v.");
    add_skip_prefix(ml, "mm.");
}

// =========================================================================
// glm-ocr (text side)
// =========================================================================
//
// Existing files use arch name "glmocr" / KV prefix "glmocr.*" with 16
// blocks. llama.cpp uses "glm4" / "glm4.*"; the GLM-OCR variant of
// LLM_ARCH_GLM4 is identified by `n_layer = 17` for models with 16 main
// layers plus one nextn prediction layer. Some existing files omit nextn and
// report 16 layers; newer created files can include it.
//
// Bigger surgery: GLM4 expects fused gate+up MLP weights stored at
// `blk.X.ffn_up.weight` with shape `[n_embd, n_ff*2]`. Existing files store
// the gate and up halves as separate `ffn_gate.weight` / `ffn_up.weight`
// tensors (each `[n_embd, n_ff]`). We register a concat load op that
// reads gate+up bytes and stitches them into the fused llama.cpp slot.

// Per-block: register a concat load that fuses separate ffn_gate + ffn_up
// tensors into llama.cpp's single `blk.X.ffn_up.weight`
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

    OLLAMA_COMPAT_LOG_INFO("%s: detected Ollama-format glmocr GGUF; applying compatibility fixes\n", __func__);

    gguf_set_val_str(meta, "general.architecture", "glm4");
    rename_kv_prefix(meta, "glmocr.", "glm4.");
    arch_name = "glm4";

    // M-RoPE: existing files can store a 3-element `rope.mrope_section`;
    // llama.cpp expects a 4-element `rope.dimension_sections`.
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

    // Tokenizer pre-tokenizer: some files wrote `llama-bpe`, but glm-ocr uses
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
    rename_tensors_containing(meta, ctx, ".attn_out.",       ".attn_output.");
    rename_tensors_containing(meta, ctx, ".post_attn_norm.", ".post_attention_norm.");
    rename_tensors_containing(meta, ctx, ".post_ffn_norm.",  ".post_ffw_norm.");

    // Fuse ffn_gate + ffn_up → ffn_up[:, 2*n_ff] for every block, then mark
    // the orphan ffn_gate tensors as skip so n_tensors lines up.
    //
    // The concat reshape grows ne[1] of ffn_up from N to 2N, so the file's
    // mmap region for the original tensor is too small to back it. Force
    // the loader off the mmap path so it pre-allocates real backend buffers
    // that our register_concat_load can fill at load_all_data time.
    disable_mmap_for(ml);
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
// Existing files use arch name "gptoss" (no hyphen) and KV prefix "gptoss.*".
// llama.cpp uses "gpt-oss" / "gpt-oss.*". Same tensor layout otherwise,
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

    OLLAMA_COMPAT_LOG_INFO("%s: detected Ollama-format gpt-oss GGUF; applying compatibility fixes\n", __func__);

    gguf_set_val_str(meta, "general.architecture", "gpt-oss");
    rename_kv_prefix(meta, "gptoss.", "gpt-oss.");
    arch_name = "gpt-oss";

    // The gpt-oss loader requires `gpt-oss.expert_feed_forward_length`
    // (n_ff_exp). Recover it from the ffn_gate_exps tensor
    // shape — for gpt-oss the tensor is created as {n_embd, n_ff_exp, n_expert}
    // so ne[1] is the per-expert FFN dim.
    if (!has_key(meta, "gpt-oss.expert_feed_forward_length")) {
        if (ggml_tensor * t = ggml_get_tensor(ctx, "blk.0.ffn_gate_exps.weight")) {
            gguf_set_val_u32(meta, "gpt-oss.expert_feed_forward_length", (uint32_t) t->ne[1]);
        }
    }
    inject_str_if_missing(meta, "gpt-oss.rope.scaling.type", "yarn");

    // Existing published GGUFs use the generic/default GPT-2 pre-tokenizer
    // marker. gpt-oss needs the gpt-4o pre-tokenizer; without it, prompts
    // tokenize differently
    // and the model produces malformed Harmony headers.
    gguf_set_val_str(meta, "tokenizer.ggml.pre", "gpt-4o");

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
// pre-output-projection norm: existing files use `output_norm.weight`,
// while llama.cpp reads `token_embd_norm.weight` (with the LFM2-specific
// LLM_TENSOR_OUTPUT_NORM_LFM2 mapping). One tensor rename.

bool detect_ollama_lfm2(const gguf_context * meta, const ggml_context * ctx) {
    const int64_t arch_kid = gguf_find_key(meta, "general.architecture");
    if (arch_kid < 0) return false;
    if (std::strcmp(gguf_get_val_str(meta, arch_kid), "lfm2") != 0) return false;
    // Marker: existing files have output_norm.weight; llama.cpp-compatible
    // files have token_embd_norm.weight instead.
    return ggml_get_tensor(const_cast<ggml_context *>(ctx), "output_norm.weight") != nullptr
        && ggml_get_tensor(const_cast<ggml_context *>(ctx), "token_embd_norm.weight") == nullptr;
}

void handle_lfm2(const llama_model_loader * ml, gguf_context * meta, ggml_context * ctx) {
    if (!detect_ollama_lfm2(meta, ctx)) return;
    (void) ml;

    OLLAMA_COMPAT_LOG_INFO("%s: detected Ollama-format lfm2 GGUF; applying compatibility fixes\n", __func__);

    rename_tensor(meta, ctx, "output_norm.weight", "token_embd_norm.weight");
    gguf_set_val_str(meta, "tokenizer.ggml.pre", "lfm2");

    // Some files have a `lfm2.feed_forward_length` value that
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
// olmo3 (text only)
// =========================================================================

void handle_olmo3(gguf_context * meta, std::string & arch_name) {
    OLLAMA_COMPAT_LOG_INFO("%s: detected Ollama-format olmo3 GGUF; applying compatibility fixes\n", __func__);

    // llama.cpp does not currently expose an "olmo3" architecture string, but
    // its olmo2 loader covers the same tensor layout and contains the OLMo3
    // sliding-window RoPE behavior. Preserve the installed GGUF by translating
    // only the architecture/KV prefix at load time.
    gguf_set_val_str(meta, "general.architecture", "olmo2");
    rename_kv_prefix(meta, "olmo3.", "olmo2.");
    arch_name = "olmo2";
}

// =========================================================================
// mistral3 (text only — for now)
// =========================================================================
//
// Same arch name on both sides. Existing published models use a monolithic
// GGUF that embeds the vision encoder + projector inline, similar to gemma3
// and qwen35moe. Differences this handler addresses:
//
//   * Embedded `v.*` / `mm.*` tensors must be hidden from the text
//     loader (otherwise n_tensors mismatch).
//   * RoPE YaRN parameters use different names:
//     `rope.scaling.beta_fast`/`beta_slow` maps to
//     `rope.scaling.yarn_beta_fast`/`yarn_beta_slow`; mscale_all_dim maps
//     to `rope.scaling.yarn_log_multiplier`.
//   * Attention temperature scale: `rope.scaling_beta` maps to
//     `attention.temperature_scale`. Same numeric value.

bool detect_ollama_mistral3(const gguf_context * meta, const ggml_context * ctx) {
    const int64_t arch_kid = gguf_find_key(meta, "general.architecture");
    if (arch_kid < 0) return false;
    if (std::strcmp(gguf_get_val_str(meta, arch_kid), "mistral3") != 0) return false;
    // Marker: published monolithic files embed v.*/mm.* tensors; files
    // produced by the llama.cpp HF converter keep them in a separate mmproj.
    return any_tensor_with_prefix(ctx, "v.")
        || any_tensor_with_prefix(ctx, "mm.")
        || has_key(meta, "mistral3.rope.scaling.beta_fast")
        || has_key(meta, "mistral3.rope.scaling.mscale_all_dim")
        || has_key(meta, "mistral3.rope.scaling_beta");
}

void handle_mistral3(const llama_model_loader * ml, gguf_context * meta, ggml_context * ctx) {
    if (!detect_ollama_mistral3(meta, ctx)) return;
    (void) ctx;

    OLLAMA_COMPAT_LOG_INFO("%s: detected Ollama-format mistral3 GGUF; applying compatibility fixes\n", __func__);

    // RoPE YaRN parameter renames.
    copy_kv(meta, "mistral3.rope.scaling.beta_fast",
                  "mistral3.rope.scaling.yarn_beta_fast");
    copy_kv(meta, "mistral3.rope.scaling.beta_slow",
                  "mistral3.rope.scaling.yarn_beta_slow");
    copy_kv(meta, "mistral3.rope.scaling.mscale_all_dim",
                  "mistral3.rope.scaling.yarn_log_multiplier");
    // Attention temperature scale: same value, different home.
    copy_kv(meta, "mistral3.rope.scaling_beta",
                  "mistral3.attention.temperature_scale");

    if (string_kv_missing_or_default(meta, "tokenizer.ggml.pre")) {
        gguf_set_val_str(meta, "tokenizer.ggml.pre", "tekken");
    }

    // Hide embedded vision + projector tensors from the text loader.
    add_skip_prefix(ml, "v.");
    add_skip_prefix(ml, "mm.");
}

// =========================================================================
// glm4moelite (text side — GLM-4.x-Flash, arch translation to deepseek2)
// =========================================================================
//
// Existing GLM-4.7-Flash (and similar Flash variants) files use
// general.architecture=glm4moelite using DeepSeek-V2 style MLA attention,
// but with the older convention of writing PER-HEAD key/value dims.
// llama.cpp loads these through the deepseek2 arch with the MLA-absorbed
// convention (head_count_kv=1, key/value dims = the kv_lora_rank-relative
// absorbed sizes).
//
// Tensor structure is identical to deepseek2 (844 tensors, exact name
// match including attn_kv_a_mqa, attn_k_b, attn_v_b, attn_q_a/b, etc.) —
// only KV semantics differ.
//
// Translation:
//   * arch_name: glm4moelite → deepseek2
//   * KV prefix: glm4moelite.* → deepseek2.*
//   * head_count_kv: original num_kv_heads (e.g. 20) → 1 (MLA absorbed)
//   * key_length:   head_dim → kv_lora_rank + rope.dimension_count
//                              (e.g. 256 → 512+64=576)
//   * value_length: head_dim → kv_lora_rank
//                              (e.g. 256 → 512)
//   * key_length_mla / value_length_mla: were the absorbed dims (576/512);
//                              the _mla variants are per-head dims
//                              (256/256). Swap to head_dim.
//   * expert_group_count / expert_group_used_count: required by deepseek2
//                              loader; default to 1 (no group routing).

bool detect_ollama_glm4moelite(const gguf_context * meta) {
    const int64_t arch_kid = gguf_find_key(meta, "general.architecture");
    if (arch_kid < 0) return false;
    return std::strcmp(gguf_get_val_str(meta, arch_kid), "glm4moelite") == 0;
}

void handle_glm4moelite(const llama_model_loader * ml, gguf_context * meta,
                        ggml_context * ctx, std::string & arch_name) {
    (void) ml;
    (void) ctx;
    if (!detect_ollama_glm4moelite(meta)) return;

    OLLAMA_COMPAT_LOG_INFO("%s: detected Ollama-format glm4moelite GGUF; translating to deepseek2 (MLA conventions)\n", __func__);

    arch_name = "deepseek2";
    gguf_set_val_str(meta, "general.architecture", "deepseek2");

    // Mirror glm4moelite.* hparams under deepseek2.* (rename copies; the
    // original glm4moelite.* keys remain but are unread — only used as the
    // "original head_dim" source below).
    rename_kv_prefix(meta, "glm4moelite.", "deepseek2.");

    // MLA absorbs all KV heads; llama.cpp uses 1.
    gguf_set_val_u32(meta, "deepseek2.attention.head_count_kv", 1);

    // key/value lengths to MLA-absorbed dims, derived from kv_lora_rank
    // and rope.dimension_count (both already mirrored under deepseek2.*).
    {
        const int64_t kv_lora_kid = gguf_find_key(meta, "deepseek2.attention.kv_lora_rank");
        const int64_t rope_kid    = gguf_find_key(meta, "deepseek2.rope.dimension_count");
        if (kv_lora_kid >= 0 && rope_kid >= 0) {
            const uint32_t kv_lora = gguf_get_val_u32(meta, kv_lora_kid);
            const uint32_t rope_d  = gguf_get_val_u32(meta, rope_kid);
            gguf_set_val_u32(meta, "deepseek2.attention.key_length",   kv_lora + rope_d);
            gguf_set_val_u32(meta, "deepseek2.attention.value_length", kv_lora);
        }
    }

    // *_mla variants: llama.cpp wants per-head dims (head_dim). Read original
    // head_dim from the un-renamed glm4moelite.attention.key_length (which
    // held the per-head dim in the published-file convention).
    {
        const int64_t hd_kid = gguf_find_key(meta, "glm4moelite.attention.key_length");
        if (hd_kid >= 0) {
            const uint32_t head_dim = gguf_get_val_u32(meta, hd_kid);
            gguf_set_val_u32(meta, "deepseek2.attention.key_length_mla",   head_dim);
            gguf_set_val_u32(meta, "deepseek2.attention.value_length_mla", head_dim);
        }
    }

    // DeepSeek-V3 expert grouping; GLM-4-MoE-Lite doesn't use it but the
    // loader expects the keys to be present. Default to 1.
    inject_u32_if_missing(meta, "deepseek2.expert_group_count",      1);
    inject_u32_if_missing(meta, "deepseek2.expert_group_used_count", 1);
    gguf_set_val_str(meta, "tokenizer.ggml.pre", "glm4");
    fix_glm4moelite_eog_token_ids(meta);
}

// =========================================================================
// qwen25vl (text side — Qwen2.5-VL, arch translation to qwen2vl)
// =========================================================================
//
// Existing Qwen2.5-VL files use general.architecture=qwen25vl, but
// llama.cpp loads both Qwen2-VL and Qwen2.5-VL under arch=qwen2vl
// (which reads the rope.dimension_sections KV for M-RoPE).
//
// Translation:
//   * arch_name: qwen25vl → qwen2vl (loader uses arch as KV prefix)
//   * KV prefix: qwen25vl.* → qwen2vl.*
//   * rope.mrope_section (3 elements) → rope.dimension_sections (4, padded with 0)
//   * Hide vision+projector tensors from the text loader.

bool detect_ollama_qwen25vl(const gguf_context * meta) {
    const int64_t arch_kid = gguf_find_key(meta, "general.architecture");
    if (arch_kid < 0) return false;
    return std::strcmp(gguf_get_val_str(meta, arch_kid), "qwen25vl") == 0;
}

void handle_qwen25vl(const llama_model_loader * ml, gguf_context * meta,
                     ggml_context * ctx, std::string & arch_name) {
    (void) ctx;
    if (!detect_ollama_qwen25vl(meta)) return;

    OLLAMA_COMPAT_LOG_INFO("%s: detected Ollama-format qwen25vl GGUF; translating to qwen2vl\n", __func__);

    // Switch architecture so the loader reads qwen2vl.* keys (and uses the
    // qwen2vl model build path, which handles M-RoPE).
    arch_name = "qwen2vl";
    gguf_set_val_str(meta, "general.architecture", "qwen2vl");

    // Mirror the qwen25vl.* KVs under qwen2vl.* (rename_kv_prefix copies;
    // the original qwen25vl.* keys remain but are unread).
    rename_kv_prefix(meta, "qwen25vl.", "qwen2vl.");

    // Translate mrope_section (3 elems) → dimension_sections (4 elems, padded).
    const int64_t kid = gguf_find_key(meta, "qwen2vl.rope.mrope_section");
    if (kid >= 0 && gguf_get_arr_n(meta, kid) >= 3) {
        const auto * src = static_cast<const int32_t *>(gguf_get_arr_data(meta, kid));
        const int32_t padded[4] = { src[0], src[1], src[2], 0 };
        gguf_set_arr_data(meta, "qwen2vl.rope.dimension_sections",
                          GGUF_TYPE_INT32, padded, 4);
    }

    add_skip_prefix(ml, "v.");
    add_skip_prefix(ml, "mm.");
}

// =========================================================================
// qwen3vl/qwen3vlmoe (text side — Qwen3-VL)
// =========================================================================
//
// Existing Qwen3-VL files use general.architecture=qwen3vl or qwen3vlmoe.
// Two missing KVs are required by the llama.cpp loader:
//
//   * <arch>.rope.dimension_sections — M-RoPE section sizes. Derived from
//     the HF config (rope_scaling.mrope_section). Hardcoded here as
//     [24, 20, 20, 0] which matches Qwen3-VL-8B (head_dim=128, sum=64).
//     If new Qwen3-VL variants ship with a different mrope, derive from
//     the head_dim or read from a published KV.
//
//   * <arch>.n_deepstack_layers — count of deepstack adapters. Length of
//     <arch>.vision.deepstack_visual_indexes (3 for current Qwen3-VL).

const char * qwen3vl_arch(const gguf_context * meta) {
    const int64_t arch_kid = gguf_find_key(meta, "general.architecture");
    if (arch_kid < 0) return nullptr;

    const char * arch = gguf_get_val_str(meta, arch_kid);
    if (std::strcmp(arch, "qwen3vl") == 0 || std::strcmp(arch, "qwen3vlmoe") == 0) {
        return arch;
    }

    return nullptr;
}

std::string qwen3vl_key(const char * arch, const char * suffix) {
    return std::string(arch) + suffix;
}

bool detect_ollama_qwen3vl(const gguf_context * meta, const ggml_context * ctx) {
    (void) ctx;
    const char * arch = qwen3vl_arch(meta);
    if (!arch) return false;
    // Marker: llama.cpp-compatible qwen3vl/qwen3vlmoe files have
    // rope.dimension_sections; affected published files do not.
    return !has_key(meta, qwen3vl_key(arch, ".rope.dimension_sections").c_str());
}

void handle_qwen3vl(const llama_model_loader * ml, gguf_context * meta, ggml_context * ctx) {
    (void) ctx;
    const char * arch = qwen3vl_arch(meta);
    if (!arch || !detect_ollama_qwen3vl(meta, ctx)) return;

    OLLAMA_COMPAT_LOG_INFO("%s: detected Ollama-format %s GGUF; applying compatibility fixes\n", __func__, arch);

    // Inject required M-RoPE sections (current Qwen3-VL family default).
    const int32_t mrope[4] = { 24, 20, 20, 0 };
    gguf_set_arr_data(meta, qwen3vl_key(arch, ".rope.dimension_sections").c_str(),
                      GGUF_TYPE_INT32, mrope, 4);

    // Derive n_deepstack_layers from the deepstack indexes array length.
    const int64_t ds_kid = gguf_find_key(meta, qwen3vl_key(arch, ".vision.deepstack_visual_indexes").c_str());
    const uint32_t n_ds = (ds_kid >= 0) ? (uint32_t) gguf_get_arr_n(meta, ds_kid) : 0;
    inject_u32_if_missing(meta, qwen3vl_key(arch, ".n_deepstack_layers").c_str(), n_ds);

    // Hide embedded vision tensors from the text loader. Ollama's Go side
    // re-passes the same blob as --mmproj so the clip loader picks them up.
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

    // llama.cpp-compatible Gemma 3 projectors store patch_embd/position_embd
    // as F32 (Gemma3VisionModel tensor_force_quant). Metal's IM2COL
    // convolution requires F32, so promote both at load time.
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

// Register a QKV merge for a single vision block. Existing files have
// separate attn_q, attn_k, and attn_v tensors; llama.cpp expects them
// concatenated along their slow axis. Capture source file offsets before
// renaming attn_q.
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
// Source: one tensor `v.patch_embed.weight`, ggml shape
//   [h, w, t=2, packed=out_c*in_c] F16
// where `packed` is the PyTorch row-major flattening of HF's
// [out_c, in_c, ...] dim pair, so packed_c = c_out*in_c + c_in.
//
// Destination: two llama.cpp tensors with ggml shape
//   [h, w, c_in, c_out] F32 each, one per temporal slice.
//
// For each output element (h, w, c_in, c_out):
//   src_idx = h + w*W + t*W*H + (c_out*C_in + c_in)*W*H*T
//   dst_idx = h + w*W + c_in*W*H + c_out*W*H*C_in
void register_qwen35moe_patch_embed_split(gguf_context * meta, ggml_context * ctx) {
    const char * src_name = "v.patch_embed.weight";
    if (gguf_find_tensor(meta, src_name) < 0) return;
    const ggml_tensor * src_t = ggml_get_tensor(ctx, src_name);
    if (!src_t) return;
    if (src_t->type != GGML_TYPE_F16) {
        OLLAMA_COMPAT_LOG_ERROR("%s: unsupported %s type %d; expected F16\n", __func__, src_name, src_t->type);
        return;
    }

    const int64_t cin_kid = gguf_find_key(meta, "clip.vision.num_channels");
    const int64_t cin     = cin_kid >= 0 ? gguf_get_val_u32(meta, cin_kid) : 3;
    const int64_t width   = src_t->ne[0];
    const int64_t height  = src_t->ne[1];
    const int64_t frames  = src_t->ne[2];
    const int64_t packed  = src_t->ne[3];
    if (cin <= 0 || width <= 0 || height <= 0 || frames != 2 || packed % cin != 0) {
        OLLAMA_COMPAT_LOG_ERROR("%s: unsupported %s shape [%lld %lld %lld %lld] with channels %lld\n",
                                __func__, src_name,
                                (long long) width, (long long) height, (long long) frames,
                                (long long) packed, (long long) cin);
        return;
    }

    const size_t src_offset = tensor_file_offset(meta, src_name);
    const size_t src_size   = (size_t) ggml_nelements(src_t) * sizeof(uint16_t);
    const size_t hw         = (size_t) width * (size_t) height;
    const int64_t cout      = packed / cin;

    auto make_slice_op = [=](int slice_idx) {
        return LoadOp{
            [=](const char * path, void * dst, size_t dst_size) {
                const size_t expected = hw * (size_t) cin * (size_t) cout * sizeof(float);
                if (dst_size != expected) return false;
                std::vector<uint8_t> src(src_size);
                if (!read_at(path, src_offset, src.data(), src_size)) return false;
                const uint16_t * sp = reinterpret_cast<const uint16_t *>(src.data());
                float          * dp = reinterpret_cast<float *>(dst);
                for (int64_t c_out = 0; c_out < cout; ++c_out) {
                    for (int64_t c_in = 0; c_in < cin; ++c_in) {
                        const size_t packed_idx = (size_t) c_out * (size_t) cin + (size_t) c_in;
                        const uint16_t * in_base  = sp + hw * ((size_t) slice_idx + (size_t) frames * packed_idx);
                        float          * out_base = dp + hw * ((size_t) c_in + (size_t) cin * (size_t) c_out);
                        for (size_t i = 0; i < hw; ++i) out_base[i] = ggml_fp16_to_fp32(in_base[i]);
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
        set_tensor_shape(dest0, {width, height, cin, cout});
        set_tensor_type (dest0, GGML_TYPE_F32);
    }
    register_load_op("v.patch_embd.weight", make_slice_op(0));

    // Reclaim the `v.blk.0.attn_k.weight` slot (orphaned by the QKV merge)
    // as the sibling `v.patch_embd.weight.1`.
    reclaim_slot_as(meta, ctx,
                    "v.blk.0.attn_k.weight", "v.patch_embd.weight.1",
                    {width, height, cin, cout}, GGML_TYPE_F32);
    register_load_op("v.patch_embd.weight.1", make_slice_op(1));
}

void register_qwen3vl_patch_embed_split(gguf_context * meta, ggml_context * ctx, const char * arch) {
    const char * src_name = "v.patch_embed.weight";
    if (gguf_find_tensor(meta, src_name) < 0) return;
    const ggml_tensor * src_t = ggml_get_tensor(ctx, src_name);
    if (!src_t) return;
    if (src_t->type != GGML_TYPE_F16) {
        OLLAMA_COMPAT_LOG_ERROR("%s: unsupported %s type %d; expected F16\n", __func__, src_name, src_t->type);
        return;
    }

    const int64_t cin_kid = gguf_find_key(meta, qwen3vl_key(arch, ".vision.num_channels").c_str());
    const int64_t cin     = cin_kid >= 0 ? gguf_get_val_u32(meta, cin_kid) : 3;
    const int64_t width   = src_t->ne[0];
    const int64_t height  = src_t->ne[1];
    const int64_t frames  = src_t->ne[2];
    const int64_t packed  = src_t->ne[3];
    if (cin <= 0 || width <= 0 || height <= 0 || frames != 2 || packed % cin != 0) {
        OLLAMA_COMPAT_LOG_ERROR("%s: unsupported %s shape [%lld %lld %lld %lld] with channels %lld\n",
                                __func__, src_name,
                                (long long) width, (long long) height, (long long) frames,
                                (long long) packed, (long long) cin);
        return;
    }

    const int64_t cout = packed / cin;
    const size_t src_offset = tensor_file_offset(meta, src_name);
    const size_t src_size   = (size_t) ggml_nelements(src_t) * sizeof(uint16_t);
    const size_t hw         = (size_t) width * (size_t) height;

    auto make_slice_op = [=](int slice_idx) {
        return LoadOp{
            [=](const char * path, void * dst, size_t dst_size) {
                const size_t expected = hw * (size_t) cin * (size_t) cout * sizeof(float);
                if (dst_size != expected) return false;
                std::vector<uint8_t> src(src_size);
                if (!read_at(path, src_offset, src.data(), src_size)) return false;
                const uint16_t * sp = reinterpret_cast<const uint16_t *>(src.data());
                float          * dp = reinterpret_cast<float *>(dst);
                for (int64_t c_out = 0; c_out < cout; ++c_out) {
                    for (int64_t c_in = 0; c_in < cin; ++c_in) {
                        const size_t packed_idx = (size_t) c_out * (size_t) cin + (size_t) c_in;
                        const uint16_t * in_base  = sp + hw * ((size_t) slice_idx + (size_t) frames * packed_idx);
                        float          * out_base = dp + hw * ((size_t) c_in + (size_t) cin * (size_t) c_out);
                        for (size_t i = 0; i < hw; ++i) out_base[i] = ggml_fp16_to_fp32(in_base[i]);
                    }
                }
                return true;
            },
            slice_idx == 0 ? "qwen3vl patch_embed slice 0 (permute+F16->F32)"
                           : "qwen3vl patch_embed slice 1 (permute+F16->F32)",
        };
    };

    rename_tensor(meta, ctx, src_name, "v.patch_embd.weight");
    if (ggml_tensor * dest0 = ggml_get_tensor(ctx, "v.patch_embd.weight")) {
        set_tensor_shape(dest0, {width, height, cin, cout});
        set_tensor_type (dest0, GGML_TYPE_F32);
    }
    register_load_op("v.patch_embd.weight", make_slice_op(0));

    reclaim_slot_as(meta, ctx,
                    "v.blk.0.attn_k.weight", "v.patch_embd.weight.1",
                    {width, height, cin, cout}, GGML_TYPE_F32);
    register_load_op("v.patch_embd.weight.1", make_slice_op(1));
}

void handle_qwen35_like_clip(gguf_context * meta, ggml_context * ctx, const char * arch) {
    OLLAMA_COMPAT_LOG_INFO("%s: detected Ollama-format %s GGUF used as mmproj; translating\n", __func__, arch);

    auto kv = [arch](const char * suffix) {
        return std::string(arch) + suffix;
    };

    copy_u32_kv(meta, kv(".vision.block_count").c_str(),                   "clip.vision.block_count");
    copy_u32_kv(meta, kv(".vision.embedding_length").c_str(),              "clip.vision.embedding_length");
    copy_u32_kv(meta, kv(".vision.attention.head_count").c_str(),          "clip.vision.attention.head_count");
    copy_u32_kv(meta, kv(".vision.patch_size").c_str(),                    "clip.vision.patch_size");
    copy_u32_kv(meta, kv(".vision.spatial_merge_size").c_str(),            "clip.vision.spatial_merge_size");
    copy_u32_kv(meta, kv(".vision.num_channels").c_str(),                  "clip.vision.num_channels");
    // projection_dim = text model's embedding_length.
    copy_u32_kv(meta, kv(".embedding_length").c_str(),                     "clip.vision.projection_dim");

    // Defaults for KVs absent from existing files. Values match the
    // Qwen3.5-35B-A3B reference mmproj.
    inject_u32_if_missing(meta, "clip.vision.feed_forward_length",          4304);
    inject_u32_if_missing(meta, "clip.vision.image_size",                   768);
    inject_f32_if_missing(meta, "clip.vision.attention.layer_norm_epsilon", 1e-6f);

    static const float kHalfHalfHalf[3] = {0.5f, 0.5f, 0.5f};
    inject_f32_arr_if_missing(meta, "clip.vision.image_mean", kHalfHalfHalf, 3);
    inject_f32_arr_if_missing(meta, "clip.vision.image_std",  kHalfHalfHalf, 3);

    // qwen3.5 has no deepstack layers, but the mask length must match the
    // vision tower block count for all sizes.
    if (!has_key(meta, "clip.vision.is_deepstack_layers")) {
        const int64_t block_kid = gguf_find_key(meta, "clip.vision.block_count");
        const uint32_t n_blocks = block_kid >= 0 ? gguf_get_val_u32(meta, block_kid) : 0;
        std::vector<uint8_t> bools(n_blocks);
        if (!bools.empty()) {
            gguf_set_arr_data(meta, "clip.vision.is_deepstack_layers", GGUF_TYPE_BOOL, bools.data(), bools.size());
        }
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

void handle_qwen35moe_clip(gguf_context * meta, ggml_context * ctx) {
    handle_qwen35_like_clip(meta, ctx, "qwen35moe");
}

void handle_qwen35_clip(gguf_context * meta, ggml_context * ctx) {
    handle_qwen35_like_clip(meta, ctx, "qwen35");
}

// =========================================================================
// deepseek-ocr (clip side — SAM + CLIP + projector)
// =========================================================================
//
// The monolithic DeepSeek OCR GGUF embeds three vision components:
//   * SAM encoder under the `s.*` prefix (12 blocks)
//   * CLIP encoder under the `v.*` prefix (24 blocks)
//   * MLP projector under `mm.*`
// The PROJECTOR_TYPE_DEEPSEEKOCR loader expects:
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
    {"v.pre_layrnorm",      "v.pre_ln"}, // published tensor spelling

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
    OLLAMA_COMPAT_LOG_INFO("%s: detected Ollama-format deepseekocr GGUF used as mmproj; translating\n", __func__);

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

    // Defaults pulled from a llama.cpp-compatible reference mmproj.
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
    // CLIP position embedding too; llama.cpp-compatible files store F32.
    promote_tensor_to_f32(meta, ctx, "v.position_embd.weight");
}

// =========================================================================
// nemotron_h_omni (clip side — nemotron_v2_vl projector)
// =========================================================================

uint32_t read_u32_kv(const gguf_context * meta, const char * key, uint32_t fallback) {
    const int64_t kid = gguf_find_key(meta, key);
    return kid >= 0 ? gguf_get_val_u32(meta, kid) : fallback;
}

std::vector<int64_t> tensor_shape(const ggml_tensor * t) {
    std::vector<int64_t> shape;
    const int n_dims = ggml_n_dims(t);
    shape.reserve(n_dims);
    for (int i = 0; i < n_dims; ++i) shape.push_back(t->ne[i]);
    return shape;
}

bool read_tensor_as_f32(const char * path,
                        size_t offset,
                        ggml_type type,
                        size_t src_size,
                        size_t n_elem,
                        std::vector<float> & out) {
    out.resize(n_elem);
    if (type == GGML_TYPE_F32) {
        return read_at(path, offset, out.data(), n_elem * sizeof(float));
    }

    std::vector<uint8_t> src(src_size);
    if (!read_at(path, offset, src.data(), src.size())) return false;
    const auto * traits = ggml_get_type_traits(type);
    if (!traits || !traits->to_float) return false;
    traits->to_float(src.data(), out.data(), (int64_t) n_elem);
    return true;
}

uint64_t nemotron3_square_side(uint64_t n) {
    const uint64_t side = (uint64_t) std::sqrt((double) n);
    if (side * side == n) return side;
    if ((side + 1) * (side + 1) == n) return side + 1;
    return 0;
}

double nemotron3_align_corners(uint64_t target, uint64_t target_side, uint64_t source_side) {
    if (target_side <= 1) return 0.0;
    return (double) target * (double) (source_side - 1) / (double) (target_side - 1);
}

bool nemotron3_position_layout(const std::vector<int64_t> & shape,
                               uint64_t & embedding,
                               uint64_t & positions) {
    if (shape.size() == 2) {
        if (nemotron3_square_side((uint64_t) shape[1]) > 0) {
            embedding = (uint64_t) shape[0];
            positions = (uint64_t) shape[1];
            return true;
        }
        if (nemotron3_square_side((uint64_t) shape[0]) > 0) {
            embedding = (uint64_t) shape[1];
            positions = (uint64_t) shape[0];
            return true;
        }
    } else if (shape.size() == 3) {
        if (shape[0] == 1 && nemotron3_square_side((uint64_t) shape[1]) > 0) {
            embedding = (uint64_t) shape[2];
            positions = (uint64_t) shape[1];
            return true;
        }
        if (shape[2] == 1 && nemotron3_square_side((uint64_t) shape[1]) > 0) {
            embedding = (uint64_t) shape[0];
            positions = (uint64_t) shape[1];
            return true;
        }
    }
    return false;
}

void register_nemotron3_position_embedding(gguf_context * meta,
                                           ggml_context * ctx,
                                           uint64_t embedding,
                                           uint64_t target_side) {
    const char * src_name = ggml_get_tensor(ctx, "v.position_embd")
        ? "v.position_embd"
        : "v.position_embd.weight";
    ggml_tensor * t = ggml_get_tensor(ctx, src_name);
    if (!t || target_side == 0) return;

    const size_t src_offset = tensor_file_offset(meta, src_name);
    const size_t src_size   = ggml_nbytes(t);
    const size_t n_elem     = ggml_nelements(t);
    const ggml_type src_type = t->type;
    const std::vector<int64_t> src_shape = tensor_shape(t);

    uint64_t detected_embedding = 0;
    uint64_t positions = 0;
    if (!nemotron3_position_layout(src_shape, detected_embedding, positions)) return;
    if (embedding == 0) embedding = detected_embedding;
    const uint64_t source_side = nemotron3_square_side(positions);
    if (embedding == 0 || source_side == 0) return;

    if (std::strcmp(src_name, "v.position_embd.weight") != 0) {
        rename_tensor(meta, ctx, src_name, "v.position_embd.weight");
        t = ggml_get_tensor(ctx, "v.position_embd.weight");
        if (!t) return;
    }
    set_tensor_shape(t, {(int64_t) embedding, (int64_t) (target_side * target_side)});
    set_tensor_type(t, GGML_TYPE_F32);

    register_load_op("v.position_embd.weight", LoadOp{
        [=](const char * path, void * dst, size_t dst_size) {
            std::vector<float> src;
            if (!read_tensor_as_f32(path, src_offset, src_type, src_size, n_elem, src)) return false;
            if (src.size() != embedding * positions) return false;

            auto source_at = [&](uint64_t pos, uint64_t emb) -> float {
                if (src_shape.size() == 2) {
                    if ((uint64_t) src_shape[0] == embedding) {
                        return src[emb + embedding * pos];
                    }
                    return src[pos + positions * emb];
                }
                if (src_shape.size() == 3 && src_shape[0] == 1) {
                    return src[pos * embedding + emb];
                }
                return src[emb * positions + pos];
            };

            const size_t out_elems = embedding * target_side * target_side;
            if (dst_size != out_elems * sizeof(float)) return false;
            float * out = static_cast<float *>(dst);
            for (uint64_t y = 0; y < target_side; ++y) {
                const double source_y = nemotron3_align_corners(y, target_side, source_side);
                const uint64_t y0 = (uint64_t) std::floor(source_y);
                const uint64_t y1 = std::min(y0 + 1, source_side - 1);
                const float wy = (float) (source_y - (double) y0);
                for (uint64_t x = 0; x < target_side; ++x) {
                    const double source_x = nemotron3_align_corners(x, target_side, source_side);
                    const uint64_t x0 = (uint64_t) std::floor(source_x);
                    const uint64_t x1 = std::min(x0 + 1, source_side - 1);
                    const float wx = (float) (source_x - (double) x0);
                    for (uint64_t emb = 0; emb < embedding; ++emb) {
                        const float v00 = source_at(y0 * source_side + x0, emb);
                        const float v01 = source_at(y0 * source_side + x1, emb);
                        const float v10 = source_at(y1 * source_side + x0, emb);
                        const float v11 = source_at(y1 * source_side + x1, emb);
                        const float top = v00 * (1.0f - wx) + v01 * wx;
                        const float bottom = v10 * (1.0f - wx) + v11 * wx;
                        out[emb + embedding * (y * target_side + x)] = top * (1.0f - wy) + bottom * wy;
                    }
                }
            }
            return true;
        },
        "Nemotron3 position embedding downsample",
    });
}

void register_nemotron3_patch_embedding(gguf_context * meta,
                                        ggml_context * ctx,
                                        uint64_t patch_size,
                                        uint64_t channels,
                                        uint64_t embedding) {
    ggml_tensor * t = ggml_get_tensor(ctx, "v.patch_embd.weight");
    if (!t || patch_size == 0 || channels == 0 || embedding == 0) return;

    const size_t src_offset = tensor_file_offset(meta, "v.patch_embd.weight");
    const size_t src_size   = ggml_nbytes(t);
    const size_t n_elem     = ggml_nelements(t);
    const ggml_type src_type = t->type;
    const std::vector<int64_t> src_shape = tensor_shape(t);
    const uint64_t flat = patch_size * patch_size * channels;

    if (src_shape.size() != 2 || (uint64_t) src_shape[0] != flat || (uint64_t) src_shape[1] != embedding) return;

    set_tensor_shape(t, {(int64_t) patch_size, (int64_t) patch_size, (int64_t) channels, (int64_t) embedding});
    set_tensor_type(t, GGML_TYPE_F32);

    register_load_op("v.patch_embd.weight", LoadOp{
        [=](const char * path, void * dst, size_t dst_size) {
            std::vector<float> src;
            if (!read_tensor_as_f32(path, src_offset, src_type, src_size, n_elem, src)) return false;
            if (src.size() != flat * embedding || dst_size != src.size() * sizeof(float)) return false;

            float * out = static_cast<float *>(dst);
            for (uint64_t emb = 0; emb < embedding; ++emb) {
                for (uint64_t channel = 0; channel < channels; ++channel) {
                    for (uint64_t y = 0; y < patch_size; ++y) {
                        for (uint64_t x = 0; x < patch_size; ++x) {
                            // llama.cpp's Nemotron HF converter reshapes the source kernel as
                            // [embedding, channels, patch, patch], so the flattened source is
                            // channel-major within each embedding row.
                            const uint64_t flat_index = x + patch_size * (y + patch_size * channel);
                            const uint64_t src_index = flat_index + flat * emb;
                            const uint64_t dst_index = x + patch_size * (y + patch_size * (channel + channels * emb));
                            out[dst_index] = src[src_index];
                        }
                    }
                }
            }
            return true;
        },
        "Nemotron3 patch embedding reshape",
    });
}

void handle_nemotron_h_omni_clip(gguf_context * meta, ggml_context * ctx) {
    OLLAMA_COMPAT_LOG_INFO("%s: detected Ollama-format nemotron_h_omni GGUF used as mmproj; translating\n", __func__);

    gguf_set_val_str(meta, "general.architecture", "clip");

    copy_u32_kv(meta, "nemotron_h_omni.vision.block_count",                  "clip.vision.block_count");
    copy_u32_kv(meta, "nemotron_h_omni.vision.embedding_length",             "clip.vision.embedding_length");
    copy_u32_kv(meta, "nemotron_h_omni.vision.feed_forward_length",          "clip.vision.feed_forward_length");
    copy_u32_kv(meta, "nemotron_h_omni.vision.attention.head_count",         "clip.vision.attention.head_count");
    copy_f32_kv(meta, "nemotron_h_omni.vision.attention.layer_norm_epsilon", "clip.vision.attention.layer_norm_epsilon");
    copy_u32_kv(meta, "nemotron_h_omni.vision.patch_size",                   "clip.vision.patch_size");
    copy_u32_kv(meta, "nemotron_h_omni.vision.image_size",                   "clip.vision.image_size");
    copy_u32_kv(meta, "nemotron_h_omni.vision.num_channels",                 "clip.vision.num_channels");
    copy_u32_kv(meta, "nemotron_h_omni.vision.projector.scale_factor",       "clip.vision.projector.scale_factor");
    copy_u32_kv(meta, "nemotron_h_omni.embedding_length",                    "clip.vision.projection_dim");
    copy_kv    (meta, "nemotron_h_omni.vision.image_mean",                  "clip.vision.image_mean");
    copy_kv    (meta, "nemotron_h_omni.vision.image_std",                   "clip.vision.image_std");

    for (const char * key : {
        "image_token_id",
        "image_start_token_id",
        "image_end_token_id",
        "max_tiles",
        "min_num_patches",
        "max_num_patches",
    }) {
        const std::string src = std::string("nemotron_h_omni.vision.") + key;
        const std::string dst = std::string("clip.vision.") + key;
        copy_kv(meta, src.c_str(), dst.c_str());
    }

    const uint32_t image_size = read_u32_kv(meta, "clip.vision.image_size", 512);
    const uint32_t patch_size = read_u32_kv(meta, "clip.vision.patch_size", 16);
    const uint32_t embedding  = read_u32_kv(meta, "clip.vision.embedding_length", 1280);
    const uint32_t channels   = read_u32_kv(meta, "clip.vision.num_channels", 3);
    const uint32_t target_side = patch_size > 0 ? image_size / patch_size : 0;

    inject_u32_if_missing(meta, "clip.vision.block_count", 32);
    inject_u32_if_missing(meta, "clip.vision.embedding_length", embedding);
    inject_u32_if_missing(meta, "clip.vision.feed_forward_length", embedding * 4);
    inject_u32_if_missing(meta, "clip.vision.attention.head_count", 16);
    inject_f32_if_missing(meta, "clip.vision.attention.layer_norm_epsilon", 1e-6f);
    inject_u32_if_missing(meta, "clip.vision.patch_size", patch_size);
    inject_u32_if_missing(meta, "clip.vision.image_size", image_size);
    inject_u32_if_missing(meta, "clip.vision.num_channels", channels);
    inject_u32_if_missing(meta, "clip.vision.projector.scale_factor", 2);
    copy_u32_kv(meta, "nemotron_h_omni.embedding_length", "clip.vision.projection_dim");
    static const float kNemotronMean[3] = {0.48145466f, 0.45782750f, 0.40821073f};
    static const float kNemotronStd [3] = {0.26862954f, 0.26130258f, 0.27577711f};
    inject_f32_arr_if_missing(meta, "clip.vision.image_mean", kNemotronMean, 3);
    inject_f32_arr_if_missing(meta, "clip.vision.image_std",  kNemotronStd,  3);

    inject_bool_if_missing(meta, "clip.has_vision_encoder", true);
    gguf_set_val_str(meta, "clip.vision.projector_type", "nemotron_v2_vl");
    inject_bool_if_missing(meta, "clip.use_gelu", true);

    // TODO: expose Nemotron3 audio once llama.cpp can skip or load the
    // Nemotron audio projector safely. For now, do not create an audio
    // context and do not load a.* / mm.a.* tensors into backend memory.
    gguf_set_val_bool(meta, "clip.has_audio_encoder", false);

    rename_tensor(meta, ctx, "v.cls_embd", "v.class_embd");
    promote_tensor_to_f32(meta, ctx, "v.class_embd");
    register_nemotron3_position_embedding(meta, ctx, embedding, target_side);
    register_nemotron3_patch_embedding(meta, ctx, patch_size, channels, embedding);

    rename_tensor(meta, ctx, "mm.norm.weight", "mm.model.mlp.0.weight");
    rename_tensor(meta, ctx, "mm.1.weight",    "mm.model.mlp.1.weight");
    rename_tensor(meta, ctx, "mm.2.weight",    "mm.model.mlp.3.weight");
}

// =========================================================================
// gemma4 (clip side — gemma4v projector)
// =========================================================================
//
// The monolithic Gemma 4 GGUF embeds a SigLIP-style ViT plus the
// gemma4v projector (a single `mm.input_projection`). All v.* / mm.*
// tensor names already match PROJECTOR_TYPE_GEMMA4V; this
// handler only needs KV translation and an F32 promote of the patch
// embedding (Metal IM2COL).
//
// gemma4 vision uses image normalization mean=[0,0,0] / std=[1,1,1]
// (the LM does its own per-image normalization via v.std_bias /
// v.std_scale tensors) — different from the [0.5,0.5,0.5] used by
// most other arches.

void handle_gemma4_clip(gguf_context * meta, ggml_context * ctx) {
    OLLAMA_COMPAT_LOG_INFO("%s: detected Ollama-format gemma4 GGUF used as mmproj; translating\n", __func__);

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
        // Defaults from Gemma 4 audio preprocessing.
        inject_u32_if_missing(meta, "clip.audio.num_mel_bins",   128);
        copy_u32_kv(meta, "gemma4.embedding_length", "clip.audio.projection_dim");

        inject_bool_if_missing(meta, "clip.has_audio_encoder", true);
        gguf_set_val_str(meta, "clip.audio.projector_type", "gemma4a");

        // Top-level tensor renames. Existing files use different leaf names for the
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
        // Semantic mapping (from Ollama's model_audio.go and gemma4a.cpp):
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
// The GLM4V vision tower uses v.blk.X.* tensor names that already match
// llama.cpp expectations (`attn_qkv`, `attn_out`,
// `attn_q_norm`, `attn_k_norm`, `ln1`/`ln2`, `ffn_{gate,up,down}`).
// Most of mm.* (mm.model.fc, mm.up/gate/down, mm.post_norm,
// mm.patch_merger) is also already named correctly. The two diffs:
//   * `v.patch_embd_0.weight` / `v.patch_embd_1.weight` →
//     pixel-shuffle patch-embed pair `v.patch_embd.weight` /
//     `v.patch_embd.weight.1`.
//   * F32 promote of patch_embd weights (Metal IM2COL).

void handle_glmocr_clip(gguf_context * meta, ggml_context * ctx) {
    OLLAMA_COMPAT_LOG_INFO("%s: detected Ollama-format glm-ocr GGUF used as mmproj; translating\n", __func__);

    copy_u32_kv(meta, "glmocr.vision.block_count",                   "clip.vision.block_count");
    copy_u32_kv(meta, "glmocr.vision.embedding_length",              "clip.vision.embedding_length");
    copy_u32_kv(meta, "glmocr.vision.intermediate_size",             "clip.vision.feed_forward_length");
    copy_u32_kv(meta, "glmocr.vision.attention.head_count",          "clip.vision.attention.head_count");
    copy_f32_kv(meta, "glmocr.vision.attention.layer_norm_rms_epsilon", "clip.vision.attention.layer_norm_epsilon");
    copy_u32_kv(meta, "glmocr.vision.image_size",                    "clip.vision.image_size");
    copy_u32_kv(meta, "glmocr.vision.patch_size",                    "clip.vision.patch_size");
    copy_u32_kv(meta, "glmocr.vision.spatial_merge_size",            "clip.vision.spatial_merge_size");
    copy_u32_kv(meta, "glmocr.vision.out_hidden_size",               "clip.vision.projection_dim");

    // Existing files already have image_mean / image_std under glmocr.vision.*;
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

    // Patch-embed temporal pair: existing files use _0/_1 suffixes; llama.cpp uses
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
// The monolithic Llama 4 GGUF embeds the CLIP-style ViT and a 3-layer
// projector (`mm.linear_1` + `v.vision_adapter.mlp.fc1/fc2`). The
// PROJECTOR_TYPE_LLAMA4 expects the projector under `mm.model.fc` /
// `mm.model.mlp.{1,2}` and standard CLIP block leaf names.

constexpr std::pair<const char *, const char *> kLlama4ClipRenames[] = {
    // Vision-adapter MLP -> MM-MLP slots. Run before the generic
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
    OLLAMA_COMPAT_LOG_INFO("%s: detected Ollama-format llama4 GGUF used as mmproj; translating\n", __func__);

    copy_u32_kv(meta, "llama4.vision.block_count",                    "clip.vision.block_count");
    copy_u32_kv(meta, "llama4.vision.embedding_length",               "clip.vision.embedding_length");
    copy_u32_kv(meta, "llama4.vision.feed_forward_length",            "clip.vision.feed_forward_length");
    copy_u32_kv(meta, "llama4.vision.attention.head_count",           "clip.vision.attention.head_count");
    copy_u32_kv(meta, "llama4.vision.image_size",                     "clip.vision.image_size");
    copy_u32_kv(meta, "llama4.vision.patch_size",                     "clip.vision.patch_size");
    copy_f32_kv(meta, "llama4.vision.layer_norm_epsilon",             "clip.vision.attention.layer_norm_epsilon");
    // projection_dim = LM embedding length (= mm.model.fc output dim).
    copy_u32_kv(meta, "llama4.embedding_length",                      "clip.vision.projection_dim");

    // Defaults match a llama.cpp-compatible reference mmproj.
    uint32_t projector_scale = 2;
    const int64_t ratio_kid = gguf_find_key(meta, "llama4.vision.pixel_shuffle_ratio");
    if (ratio_kid >= 0) {
        const float ratio = gguf_get_val_f32(meta, ratio_kid);
        if (ratio > 0.0f) {
            projector_scale = (uint32_t) std::round(1.0f / ratio);
            if (projector_scale == 0) {
                projector_scale = 2;
            }
        }
    }
    inject_u32_if_missing(meta, "clip.vision.projector.scale_factor", projector_scale);

    static const float kHalfHalfHalf[3] = {0.5f, 0.5f, 0.5f};
    inject_f32_arr_if_missing(meta, "clip.vision.image_mean", kHalfHalfHalf, 3);
    inject_f32_arr_if_missing(meta, "clip.vision.image_std",  kHalfHalfHalf, 3);

    inject_bool_if_missing(meta, "clip.has_vision_encoder", true);
    inject_bool_if_missing(meta, "clip.use_gelu",           true);
    gguf_set_val_str(meta, "clip.projector_type",  "llama4");
    gguf_set_val_str(meta, "clip.vision.projector_type", "llama4");
    gguf_set_val_str(meta, "general.architecture", "clip");

    // Position embedding has no `.weight` suffix in existing files; rename exactly.
    rename_tensor(meta, ctx, "v.positional_embedding_vlm", "v.position_embd.weight");

    for (const auto & [from, to] : kLlama4ClipRenames) {
        rename_tensors_containing(meta, ctx, from, to);
    }
}

// =========================================================================
// mistral3 (clip side — pixtral projector)
// =========================================================================
//
// Tensor renames for Pixtral projector compatibility:
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
// Existing monolithic files do not ship it as a separate tensor; the
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

// Apply the LLaMA-style RoPE permutation to the vision Q/K weight.
//
// The existing Mistral 3 conversion path only applies
// its repack to TEXT-side attn_q/attn_k (the `if !HasPrefix(name, "v.")`
// guard skips vision tensors). So vision Q/K leave the converter in raw
// HF/PyTorch order. The llama.cpp HF-to-GGUF flow permutes vision Q/K with
// the vision head count,
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
// block-aware shuffling; published Mistral 3 8B files keep Q/K F16).
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
    OLLAMA_COMPAT_LOG_INFO("%s: detected Ollama-format mistral3 GGUF used as mmproj; translating\n", __func__);

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

    // Apply LLaMA-style RoPE permutation to vision Q/K before renames
    // (we capture offsets by current name). The published-file conversion
    // path only repacks text-side q/k (skipping `v.*`), but pixtral's clip
    // graph expects the HF-to-GGUF permuted layout for vision Q/K.
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

    // llama.cpp-compatible files store patch_embd as F32. Metal's IM2COL
    // convolution requires F32 weights, same as gemma3.
    promote_tensor_to_f32(meta, ctx, "v.patch_embd.weight");
}

// =========================================================================
// qwen25vl (clip side — Qwen2.5-VL vision tower + merger)
// =========================================================================
//
// Qwen2.5-VL published files have a vision tower with mostly
// llama.cpp-compatible tensor names. Five tensor renames + KV translation:
//
//   v.merger.ln_q.weight         → v.post_ln.weight       (post-tower norm)
//   v.merger.mlp.0.{weight,bias} → mm.0.{weight,bias}     (LLaVA proj 0)
//   v.merger.mlp.2.{weight,bias} → mm.2.{weight,bias}     (LLaVA proj 2)
//   v.patch_embd_0.weight        → v.patch_embd.weight    (slice 0)
//   v.patch_embd_1.weight        → v.patch_embd.weight.1  (slice 1)
//
// The KV side maps qwen25vl.vision.* → clip.vision.*, sets the projector
// type and use_silu, derives n_wa_pattern from fullatt_block_indexes[0]+1
// (matching llama.cpp's Qwen2.5-VL converter), and supplies image_size=560 and
// projection_dim (= text embedding_length, qwen25vl.embedding_length).

void handle_qwen25vl_clip(gguf_context * meta, ggml_context * ctx) {
    OLLAMA_COMPAT_LOG_INFO("%s: detected Ollama-format qwen25vl GGUF used as mmproj; translating\n", __func__);

    copy_u32_kv(meta, "qwen25vl.vision.attention.head_count",          "clip.vision.attention.head_count");
    copy_f32_kv(meta, "qwen25vl.vision.attention.layer_norm_epsilon",  "clip.vision.attention.layer_norm_epsilon");
    copy_u32_kv(meta, "qwen25vl.vision.block_count",                   "clip.vision.block_count");
    copy_u32_kv(meta, "qwen25vl.vision.embedding_length",              "clip.vision.embedding_length");
    copy_u32_kv(meta, "qwen25vl.vision.num_channels",                  "clip.vision.num_channels");
    copy_u32_kv(meta, "qwen25vl.vision.patch_size",                    "clip.vision.patch_size");
    copy_u32_kv(meta, "qwen25vl.vision.spatial_merge_size",            "clip.vision.spatial_merge_size");
    copy_u32_kv(meta, "qwen25vl.vision.window_size",                   "clip.vision.window_size");
    copy_u32_kv(meta, "qwen25vl.embedding_length",                     "clip.vision.projection_dim");

    // Derive feed_forward_length from the actual ffn_up shape if missing.
    if (!has_key(meta, "clip.vision.feed_forward_length")) {
        if (ggml_tensor * t = ggml_get_tensor(ctx, "v.blk.0.ffn_up.weight")) {
            gguf_set_val_u32(meta, "clip.vision.feed_forward_length", (uint32_t) t->ne[1]);
        }
    }

    // Derive n_wa_pattern from fullatt_block_indexes[0]+1.
    {
        const int64_t kid = gguf_find_key(meta, "qwen25vl.vision.fullatt_block_indexes");
        if (kid >= 0 && gguf_get_arr_n(meta, kid) >= 1) {
            const auto * arr = static_cast<const int32_t *>(gguf_get_arr_data(meta, kid));
            gguf_set_val_u32(meta, "clip.vision.n_wa_pattern", (uint32_t)(arr[0] + 1));
        }
    }

    // Default image_size = 560 (Qwen2VLVisionModel default, no image_size in HF config).
    inject_u32_if_missing(meta, "clip.vision.image_size", 560);

    // Standard preprocessor mean/std for Qwen2.5-VL (CLIP convention).
    static const float kMean[3] = {0.48145466f, 0.4578275f,  0.40821073f};
    static const float kStd [3] = {0.26862954f, 0.26130258f, 0.27577711f};
    inject_f32_arr_if_missing(meta, "clip.vision.image_mean", kMean, 3);
    inject_f32_arr_if_missing(meta, "clip.vision.image_std",  kStd,  3);

    inject_bool_if_missing(meta, "clip.has_vision_encoder", true);
    inject_bool_if_missing(meta, "clip.use_silu",           true);
    gguf_set_val_str(meta, "clip.projector_type",  "qwen2.5vl_merger");
    gguf_set_val_str(meta, "general.architecture", "clip");

    // Tensor renames.
    rename_tensor(meta, ctx, "v.merger.ln_q.weight",   "v.post_ln.weight");
    rename_tensor(meta, ctx, "v.merger.mlp.0.weight",  "mm.0.weight");
    rename_tensor(meta, ctx, "v.merger.mlp.0.bias",    "mm.0.bias");
    rename_tensor(meta, ctx, "v.merger.mlp.2.weight",  "mm.2.weight");
    rename_tensor(meta, ctx, "v.merger.mlp.2.bias",    "mm.2.bias");
    rename_tensor(meta, ctx, "v.patch_embd_0.weight",  "v.patch_embd.weight");
    rename_tensor(meta, ctx, "v.patch_embd_1.weight",  "v.patch_embd.weight.1");

    // Metal IM2COL needs F32 patch_embd (same issue as gemma3 / glmocr).
    promote_tensor_to_f32(meta, ctx, "v.patch_embd.weight");
    promote_tensor_to_f32(meta, ctx, "v.patch_embd.weight.1");
}

// =========================================================================
// qwen3vl/qwen3vlmoe (clip side — Qwen3-VL vision tower + deepstack adapters)
// =========================================================================
//
// Qwen3-VL monolithic GGUFs embed the vision tower (27 blocks),
// deepstack merger adapters (3 of them, indexed 0/1/2), and the merger
// MLP. Compared to qwen3vl_merger expectations:
//
//   * Per-block leaf renames: norm1→ln1, norm2→ln2, mlp.linear_fc1→ffn_up,
//     mlp.linear_fc2→ffn_down.
//   * Merger renames: v.merger.norm→v.post_ln, v.merger.linear_fc1→mm.0,
//     v.merger.linear_fc2→mm.2 (LLaVA proj).
//   * Deepstack remap: v.deepstack_merger.X.* → v.deepstack.{indexes[X]}.*
//     where indexes is <arch>.vision.deepstack_visual_indexes (e.g.
//     [8, 16, 24] for Qwen3-VL-8B). The leaf names also rename:
//     linear_fc1→fc1, linear_fc2→fc2.
//   * Per-block QKV merge: qwen3vl reads a single attn_qkv tensor
//     (shape [hidden, 3*hidden]); existing files store separate
//     Q/K/V. Same merge as qwen35moe — reuse that helper.
//   * Patch embed: split the merged Conv3D weight [W,H,T,OUT*IN] into two
//     Conv2D weights [W,H,IN,OUT], one per temporal slice. Same logic and
//     donor (orphaned attn_k from QKV merge) as qwen35moe; reuse that helper.

void handle_qwen3vl_clip(gguf_context * meta, ggml_context * ctx) {
    const char * arch_cstr = qwen3vl_arch(meta);
    if (!arch_cstr) return;
    const std::string arch(arch_cstr);

    OLLAMA_COMPAT_LOG_INFO("%s: detected Ollama-format %s GGUF used as mmproj; translating\n", __func__, arch.c_str());

    copy_u32_kv(meta, qwen3vl_key(arch.c_str(), ".vision.attention.head_count").c_str(),         "clip.vision.attention.head_count");
    copy_f32_kv(meta, qwen3vl_key(arch.c_str(), ".vision.attention.layer_norm_epsilon").c_str(), "clip.vision.attention.layer_norm_epsilon");
    copy_u32_kv(meta, qwen3vl_key(arch.c_str(), ".vision.block_count").c_str(),                  "clip.vision.block_count");
    copy_u32_kv(meta, qwen3vl_key(arch.c_str(), ".vision.embedding_length").c_str(),             "clip.vision.embedding_length");
    copy_u32_kv(meta, qwen3vl_key(arch.c_str(), ".vision.num_channels").c_str(),                 "clip.vision.num_channels");
    copy_u32_kv(meta, qwen3vl_key(arch.c_str(), ".vision.patch_size").c_str(),                   "clip.vision.patch_size");
    copy_u32_kv(meta, qwen3vl_key(arch.c_str(), ".vision.spatial_merge_size").c_str(),           "clip.vision.spatial_merge_size");
    copy_u32_kv(meta, qwen3vl_key(arch.c_str(), ".vision.temporal_patch_size").c_str(),          "clip.vision.temporal_patch_size");
    copy_f32_kv(meta, qwen3vl_key(arch.c_str(), ".vision.rope.freq_base").c_str(),               "clip.vision.rope.freq_base");
    copy_u32_kv(meta, qwen3vl_key(arch.c_str(), ".embedding_length").c_str(),                    "clip.vision.projection_dim");

    inject_u32_if_missing(meta, "clip.vision.num_channels",        3);
    inject_u32_if_missing(meta, "clip.vision.temporal_patch_size", 2);
    inject_f32_if_missing(meta, "clip.vision.rope.freq_base",      10000.0f);

    // Derive feed_forward_length from ffn_up / mlp.linear_fc1 shape.
    if (!has_key(meta, "clip.vision.feed_forward_length")) {
        if (ggml_tensor * t = ggml_get_tensor(ctx, "v.blk.0.mlp.linear_fc1.weight")) {
            gguf_set_val_u32(meta, "clip.vision.feed_forward_length", (uint32_t) t->ne[1]);
        }
    }

    // image_size = sqrt(num_position_embeddings) * patch_size. v.pos_embed
    // shape is [n_embd, num_positions], so num_positions = ne[1].
    if (!has_key(meta, "clip.vision.image_size")) {
        ggml_tensor * pe = ggml_get_tensor(ctx, "v.pos_embed.weight");
        const int64_t patch_kid = gguf_find_key(meta, qwen3vl_key(arch.c_str(), ".vision.patch_size").c_str());
        if (pe && patch_kid >= 0) {
            const uint32_t patch = gguf_get_val_u32(meta, patch_kid);
            const uint32_t side  = (uint32_t) std::sqrt((double) pe->ne[1]);
            gguf_set_val_u32(meta, "clip.vision.image_size", side * patch);
        }
    }

    copy_kv(meta, qwen3vl_key(arch.c_str(), ".vision.image_mean").c_str(), "clip.vision.image_mean");
    copy_kv(meta, qwen3vl_key(arch.c_str(), ".vision.image_std").c_str(),  "clip.vision.image_std");
    copy_u32_kv(meta, qwen3vl_key(arch.c_str(), ".vision.shortest_edge").c_str(), "clip.vision.image_min_pixels");
    copy_u32_kv(meta, qwen3vl_key(arch.c_str(), ".vision.longest_edge").c_str(),  "clip.vision.image_max_pixels");
    inject_u32_if_missing(meta, "clip.vision.image_min_pixels", 65536);
    inject_u32_if_missing(meta, "clip.vision.image_max_pixels", 16777216);

    // Image mean/std (Qwen3-VL uses [0.5, 0.5, 0.5] for both, per HF config).
    static const float kHalfHalfHalf[3] = {0.5f, 0.5f, 0.5f};
    inject_f32_arr_if_missing(meta, "clip.vision.image_mean", kHalfHalfHalf, 3);
    inject_f32_arr_if_missing(meta, "clip.vision.image_std",  kHalfHalfHalf, 3);

    if (!has_key(meta, "clip.vision.is_deepstack_layers")) {
        const int64_t block_kid = gguf_find_key(meta, "clip.vision.block_count");
        const uint32_t n_blocks = block_kid >= 0 ? gguf_get_val_u32(meta, block_kid) : 0;
        std::vector<uint8_t> mask(n_blocks);
        const int64_t ds_kid = gguf_find_key(meta, qwen3vl_key(arch.c_str(), ".vision.deepstack_visual_indexes").c_str());
        if (ds_kid >= 0) {
            const size_t n = gguf_get_arr_n(meta, ds_kid);
            const auto * idx = static_cast<const int32_t *>(gguf_get_arr_data(meta, ds_kid));
            for (size_t i = 0; i < n; ++i) {
                if (idx[i] >= 0 && (uint32_t) idx[i] < n_blocks) mask[idx[i]] = 1;
            }
        }
        if (!mask.empty()) {
            gguf_set_arr_data(meta, "clip.vision.is_deepstack_layers", GGUF_TYPE_BOOL, mask.data(), mask.size());
        }
    }

    inject_bool_if_missing(meta, "clip.has_vision_encoder", true);
    inject_bool_if_missing(meta, "clip.use_gelu",           true);
    gguf_set_val_str(meta, "clip.projector_type",  "qwen3vl_merger");
    gguf_set_val_str(meta, "general.architecture", "clip");

    // Per-block QKV merge: qwen3vl_merger reads a single
    // `v.blk.X.attn_qkv.weight` (shape [hidden, 3*hidden]). Existing files
    // store separate Q/K/V and can mix F16 (Q/K) with Q8_0 (V), so a raw byte
    // concat is not valid. Dequantize all three to F32 and concat in F32.
    // After the merge, attn_k/attn_v become orphaned in the clip ctx, which
    // the patch_embed split then reclaims for `v.patch_embd.weight.1`.
    const int64_t n_blocks_key = gguf_find_key(meta, "clip.vision.block_count");
    const uint32_t n_blocks = n_blocks_key >= 0 ? gguf_get_val_u32(meta, n_blocks_key) : 27;
    for (uint32_t b = 0; b < n_blocks; ++b) {
        char q[64], k[64], v[64], qb[64], kb[64], vb[64], qkv_w[64], qkv_b[64];
        std::snprintf(q,     sizeof(q),     "v.blk.%u.attn_q.weight",   b);
        std::snprintf(k,     sizeof(k),     "v.blk.%u.attn_k.weight",   b);
        std::snprintf(v,     sizeof(v),     "v.blk.%u.attn_v.weight",   b);
        std::snprintf(qb,    sizeof(qb),    "v.blk.%u.attn_q.bias",     b);
        std::snprintf(kb,    sizeof(kb),    "v.blk.%u.attn_k.bias",     b);
        std::snprintf(vb,    sizeof(vb),    "v.blk.%u.attn_v.bias",     b);
        std::snprintf(qkv_w, sizeof(qkv_w), "v.blk.%u.attn_qkv.weight", b);
        std::snprintf(qkv_b, sizeof(qkv_b), "v.blk.%u.attn_qkv.bias",   b);
        if (!ggml_get_tensor(ctx, q)) continue;

        register_concat_load_to_f32(meta, ctx, qkv_w, {q, k, v});
        register_concat_load_to_f32(meta, ctx, qkv_b, {qb, kb, vb});

        rename_tensor(meta, ctx, q, qkv_w);
        if (ggml_tensor * t = ggml_get_tensor(ctx, qkv_w)) {
            set_tensor_shape(t, {t->ne[0], t->ne[1] * 3});
            set_tensor_type (t, GGML_TYPE_F32);
        }
        rename_tensor(meta, ctx, qb, qkv_b);
        if (ggml_tensor * t = ggml_get_tensor(ctx, qkv_b)) {
            set_tensor_shape(t, {t->ne[0] * 3});
            set_tensor_type (t, GGML_TYPE_F32);
        }
    }

    // Patch embed split runs BEFORE per-block substring renames so it can
    // find the source by name `v.patch_embed.weight`. Qwen3-VL variants have
    // different vision widths, so derive the split shape from the source.
    register_qwen3vl_patch_embed_split(meta, ctx, arch.c_str());

    // Top-level renames (full names) — must run before substring per-block
    // renames so .linear_fc1 substring matches only inside .mlp.linear_fc1.
    rename_tensor(meta, ctx, "v.merger.norm.weight",         "v.post_ln.weight");
    rename_tensor(meta, ctx, "v.merger.norm.bias",           "v.post_ln.bias");
    rename_tensor(meta, ctx, "v.merger.linear_fc1.weight",   "mm.0.weight");
    rename_tensor(meta, ctx, "v.merger.linear_fc1.bias",     "mm.0.bias");
    rename_tensor(meta, ctx, "v.merger.linear_fc2.weight",   "mm.2.weight");
    rename_tensor(meta, ctx, "v.merger.linear_fc2.bias",     "mm.2.bias");
    rename_tensor(meta, ctx, "v.patch_embed.bias",           "v.patch_embd.bias");
    rename_tensor(meta, ctx, "v.pos_embed.weight",           "v.position_embd.weight");

    // Deepstack remap: v.deepstack_merger.X.{norm,linear_fc1,linear_fc2}.{weight,bias}
    // → v.deepstack.{deepstack_visual_indexes[X]}.{norm,fc1,fc2}.{weight,bias}.
    // Deepstack tensors use the absolute clip layer index
    // (e.g. v.deepstack.8.* for the adapter that fires after layer 8).
    {
        const int64_t ds_kid = gguf_find_key(meta, qwen3vl_key(arch.c_str(), ".vision.deepstack_visual_indexes").c_str());
        if (ds_kid >= 0) {
            const size_t n = gguf_get_arr_n(meta, ds_kid);
            const auto * idx = static_cast<const int32_t *>(gguf_get_arr_data(meta, ds_kid));
            for (size_t i = 0; i < n; ++i) {
                char from[GGML_MAX_NAME], to[GGML_MAX_NAME];
                auto rn = [&](const char * leaf_from, const char * leaf_to, const char * suffix) {
                    std::snprintf(from, sizeof(from), "v.deepstack_merger.%zu.%s.%s", i, leaf_from, suffix);
                    std::snprintf(to,   sizeof(to),   "v.deepstack.%d.%s.%s", idx[i], leaf_to, suffix);
                    rename_tensor(meta, ctx, from, to);
                };
                rn("norm",        "norm", "weight");
                rn("norm",        "norm", "bias");
                rn("linear_fc1",  "fc1",  "weight");
                rn("linear_fc1",  "fc1",  "bias");
                rn("linear_fc2",  "fc2",  "weight");
                rn("linear_fc2",  "fc2",  "bias");
            }
        }
    }

    // Per-block substring renames (safe — these substrings now only appear
    // in v.blk.X.* paths after the top-level/deepstack renames above).
    rename_tensors_containing(meta, ctx, ".norm1",          ".ln1");
    rename_tensors_containing(meta, ctx, ".norm2",          ".ln2");
    rename_tensors_containing(meta, ctx, ".mlp.linear_fc1", ".ffn_up");
    rename_tensors_containing(meta, ctx, ".mlp.linear_fc2", ".ffn_down");

    // Position embed should be F32 (precision matters for resize_position_embeddings).
    promote_tensor_to_f32(meta, ctx, "v.position_embd.weight");
}

bool needs_default_llava_projector_type(const gguf_context * meta) {
    const int64_t arch_kid = gguf_find_key(meta, "general.architecture");
    if (arch_kid < 0) return false;
    if (std::strcmp(gguf_get_val_str(meta, arch_kid), "clip") != 0) return false;

    const int64_t vision_kid = gguf_find_key(meta, "clip.has_vision_encoder");
    if (vision_kid < 0 || !gguf_get_val_bool(meta, vision_kid)) return false;

    return !has_key(meta, "clip.projector_type")
        && !has_key(meta, "clip.vision.projector_type");
}

void handle_missing_llava_projector_type(gguf_context * meta) {
    if (!needs_default_llava_projector_type(meta)) return;

    OLLAMA_COMPAT_LOG_INFO("%s: detected LLaVA/BakLLaVA projector without projector type; defaulting to mlp\n", __func__);
    gguf_set_val_str(meta, "clip.projector_type", "mlp");
}

} // anonymous namespace

// =========================================================================
// Public entry points
// =========================================================================

bool translate_metadata(const llama_model_loader * ml,
                        gguf_context * meta,
                        ggml_context * ctx,
                        std::string & arch_name,
                        const char * fname) {
    if (!meta) return false;
    if (compat_disabled()) return false;
    {
        std::lock_guard<std::mutex> lk(g_loader_path_mutex);
        g_loader_paths[ml] = fname ? fname : "";
    }
    // embeddinggemma must run before gemma3: it switches arch_name to
    // "gemma-embedding", which is what later checks (and the loader's KV
    // prefix) need to see.
    if (arch_name == "gemma3")    handle_embeddinggemma(ml, meta, ctx, arch_name);
    if (arch_name == "gemma3")    handle_gemma3   (ml, meta, ctx);
    if (arch_name == "bert")      handle_snowflake_arctic_embed2(meta);
    if (arch_name == "gemma3n")   handle_gemma3n  (ml, meta, ctx);
    if (arch_name == "gemma4")    handle_gemma4   (ml, meta, ctx);
    if (arch_name == "qwen35moe") handle_qwen35moe(ml, meta, ctx);
    if (arch_name == "qwen35")    handle_qwen35   (ml, meta, ctx);
    if (arch_name == "qwen3next") handle_qwen3next(meta, ctx);
    if (arch_name == "gptoss")        handle_gptoss        (ml, meta, ctx, arch_name);
    if (arch_name == "lfm2")          handle_lfm2          (ml, meta, ctx);
    if (arch_name == "olmo3")         handle_olmo3         (meta, arch_name);
    if (arch_name == "mistral3")      handle_mistral3      (ml, meta, ctx);
    // qwen25vl must run before any qwen2vl-targeted handler — it switches
    // arch_name to "qwen2vl" so the loader uses qwen2vl.* keys.
    if (arch_name == "qwen25vl")      handle_qwen25vl      (ml, meta, ctx, arch_name);
    if (arch_name == "qwen3vl" || arch_name == "qwen3vlmoe")
                                     handle_qwen3vl       (ml, meta, ctx);
    // glm4moelite switches arch_name to "deepseek2" — same pattern.
    if (arch_name == "glm4moelite")   handle_glm4moelite   (ml, meta, ctx, arch_name);
    if (arch_name == "deepseekocr")   handle_deepseekocr   (ml, meta, ctx, arch_name);
    if (arch_name == "nemotron_h_omni") handle_nemotron_h_omni(ml, meta, ctx, arch_name);
    if (arch_name == "nemotron_h_moe")  handle_nemotron_h_moe (ml, meta, ctx);
    if (arch_name == "llama")         handle_llama3_metadata(meta);
    if (arch_name == "llama4")        handle_llama4        (ml, meta, ctx);
    if (arch_name == "glmocr")        handle_glmocr        (ml, meta, ctx, arch_name);
    // Dispatch. Add more arches as they are wired up.

    const bool no_mmap = is_mmap_disabled_for(ml);
    if (no_mmap) {
        OLLAMA_COMPAT_LOG_INFO("compat patch disabled mmap for transformed text tensors\n");
    }
    return no_mmap;
}

void translate_clip_metadata(gguf_context * meta, ggml_context * ctx) {
    if (!meta) return;
    if (compat_disabled()) return;

    handle_missing_llava_projector_type(meta);

    if (!any_tensor_with_prefix(ctx, "v.")) return; // nothing to translate

    if (detect_ollama_gemma3(meta, ctx)) {
        OLLAMA_COMPAT_LOG_INFO("%s: detected Ollama-format gemma3 GGUF used as mmproj; translating\n", __func__);
        handle_gemma3_clip(meta, ctx);
        return;
    }
    if (detect_ollama_qwen35moe(meta, ctx)) {
        handle_qwen35moe_clip(meta, ctx);
        return;
    }
    if (detect_ollama_qwen35(meta, ctx)) {
        handle_qwen35_clip(meta, ctx);
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
    if (detect_ollama_nemotron_h_omni(meta, ctx)) {
        handle_nemotron_h_omni_clip(meta, ctx);
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
    if (detect_ollama_qwen25vl(meta)) {
        handle_qwen25vl_clip(meta, ctx);
        return;
    }
    if (detect_ollama_qwen3vl(meta, ctx)) {
        handle_qwen3vl_clip(meta, ctx);
        return;
    }
}

bool should_skip_tensor(const llama_model_loader * ml, const char * tensor_name) {
    if (compat_disabled()) return false;
    return should_skip_tensor_prefix(ml, tensor_name);
}

static bool write_tensor_data(ggml_tensor * cur,
                              ggml_backend_buffer_type_t buft,
                              const void * data,
                              size_t size) {
    // buft can be null for tensors not yet bound to a backend buffer (e.g.
    // tied output reusing token_embd's storage). In that case the tensor
    // already has a host-side data pointer — write to it directly.
    const bool is_host = !buft || ggml_backend_buft_is_host(buft);
    if (is_host) {
        if (!cur->data) {
            OLLAMA_COMPAT_LOG_ERROR("%s: no destination for %s (no buffer, no data)\n", __func__, ggml_get_name(cur));
            return false;
        }
        std::memcpy(cur->data, data, size);
    } else {
        ggml_backend_tensor_set(cur, data, 0, size);
    }
    return true;
}

static bool load_tensor_with_op(ggml_tensor * cur,
                                const char * source_file,
                                ggml_backend_buffer_type_t buft,
                                const LoadOp & op) {

    const auto start = std::chrono::steady_clock::now();
    const size_t dst_size = ggml_nbytes(cur);
    std::vector<uint8_t> dst(dst_size);
    if (!op.apply(source_file, dst.data(), dst_size)) {
        OLLAMA_COMPAT_LOG_ERROR("%s: %s failed for %s after %.3f ms\n",
                                __func__, op.description, ggml_get_name(cur), elapsed_ms(start));
        return false;
    }

    if (!write_tensor_data(cur, buft, dst.data(), dst_size)) return false;

    const double ms = elapsed_ms(start);
    const TransformTiming total = record_transform_timing(dst_size, ms);
    OLLAMA_COMPAT_LOG_INFO("compat tensor transform: op=%s tensor=%s bytes=%zu duration_ms=%.3f total_ops=%llu total_bytes=%zu total_ms=%.3f\n",
                           op.description, ggml_get_name(cur), dst_size, ms,
                           (unsigned long long) total.count, total.bytes, total.ms);
    return true;
}

bool maybe_load_tensor(ggml_tensor * cur,
                       const char * source_file,
                       size_t file_offset,
                       ggml_backend_buffer_type_t buft) {
    if (compat_disabled()) return false;

    LoadOp op;
    if (take_load_op(ggml_get_name(cur), op)) {
        return load_tensor_with_op(cur, source_file, buft, op);
    }

#if defined(_WIN32)
    // Avoid Windows iostream seek failures in clip tensor loading.
    const size_t dst_size = ggml_nbytes(cur);
    std::vector<uint8_t> dst(dst_size);
    if (!read_at(source_file, file_offset, dst.data(), dst_size)) {
        OLLAMA_COMPAT_LOG_ERROR("%s: read failed for %s\n", __func__, ggml_get_name(cur));
        return false;
    }
    return write_tensor_data(cur, buft, dst.data(), dst_size);
#else
    (void) source_file;
    (void) file_offset;
    (void) buft;
    return false;
#endif
}

bool maybe_load_text_tensor(const llama_model_loader * ml,
                            ggml_tensor * cur,
                            size_t file_offset) {
    if (compat_disabled()) return false;
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
    (void) file_offset; // registered text ops capture their own offsets

    LoadOp op;
    if (!take_load_op(ggml_get_name(cur), op)) return false;
    return load_tensor_with_op(cur, path.c_str(), buft, op);
}

int maybe_clip_mmproj_embd(const char * projector_type, int projection_dim) {
    if (compat_disabled() || projection_dim <= 0) return 0;
    if (!clip_mmproj_embd_uses_projection_dim(projector_type)) return 0;
    return projection_dim;
}

} // namespace llama_ollama_compat
