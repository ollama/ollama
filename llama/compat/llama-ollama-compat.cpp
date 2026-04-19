#include "llama-ollama-compat.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"
#include "llama-impl.h"
#include "llama-model-loader.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace llama_ollama_compat {
namespace {

// -------------------------------------------------------------------------
// tiny gguf_context helpers
// -------------------------------------------------------------------------

bool has_key(const gguf_context * meta, const char * key) {
    return gguf_find_key(meta, key) >= 0;
}

bool any_tensor_with_prefix(const ggml_context * ctx, const char * prefix) {
    const size_t plen = std::strlen(prefix);
    for (ggml_tensor * t = ggml_get_first_tensor(ctx); t; t = ggml_get_next_tensor(ctx, t)) {
        if (std::strncmp(ggml_get_name(t), prefix, plen) == 0) return true;
    }
    return false;
}

// Copy a uint32 KV from src to dst if src exists and dst doesn't.
void copy_u32_kv(gguf_context * meta, const char * src, const char * dst) {
    if (has_key(meta, dst)) return;
    const int64_t k = gguf_find_key(meta, src);
    if (k < 0) return;
    gguf_set_val_u32(meta, dst, gguf_get_val_u32(meta, k));
}

// Copy a float32 KV from src to dst if src exists and dst doesn't.
void copy_f32_kv(gguf_context * meta, const char * src, const char * dst) {
    if (has_key(meta, dst)) return;
    const int64_t k = gguf_find_key(meta, src);
    if (k < 0) return;
    gguf_set_val_f32(meta, dst, gguf_get_val_f32(meta, k));
}

// Truncate a string-typed KV array to `new_n` entries.
void truncate_str_arr(gguf_context * meta, const char * key, size_t new_n) {
    const int64_t kid = gguf_find_key(meta, key);
    if (kid < 0 || new_n >= gguf_get_arr_n(meta, kid)) return;

    std::vector<std::string> owned;
    owned.reserve(new_n);
    std::vector<const char *> ptrs;
    ptrs.reserve(new_n);
    for (size_t i = 0; i < new_n; ++i) owned.emplace_back(gguf_get_arr_str(meta, kid, i));
    for (const auto & s : owned) ptrs.push_back(s.c_str());
    gguf_set_arr_str(meta, key, ptrs.data(), new_n);
}

// Truncate a primitive-typed KV array to `new_n` entries.
void truncate_data_arr(gguf_context * meta, const char * key,
                       gguf_type elem_type, size_t elem_size, size_t new_n) {
    const int64_t kid = gguf_find_key(meta, key);
    if (kid < 0 || new_n >= gguf_get_arr_n(meta, kid)) return;

    std::vector<uint8_t> copy(elem_size * new_n);
    std::memcpy(copy.data(), gguf_get_arr_data(meta, kid), elem_size * new_n);
    gguf_set_arr_data(meta, key, elem_type, copy.data(), new_n);
}

// Rename a tensor in BOTH the gguf_context and the ggml_context so that all
// name-based lookups agree. gguf_get_tensor_name returns a pointer into a
// mutable `char[GGML_MAX_NAME]` inside a std::vector element; the const on
// the return type is API courtesy, so writing through const_cast is defined.
void rename_tensor(gguf_context * meta, ggml_context * ctx,
                   const char * old_name, const char * new_name) {
    const int64_t id = gguf_find_tensor(meta, old_name);
    if (id < 0) return;
    if (char * p = const_cast<char *>(gguf_get_tensor_name(meta, id))) {
        std::strncpy(p, new_name, GGML_MAX_NAME - 1);
        p[GGML_MAX_NAME - 1] = '\0';
    }
    if (ggml_tensor * t = ggml_get_tensor(ctx, old_name)) ggml_set_name(t, new_name);
}

// Rename every tensor whose name contains `needle` (covers `.weight` + `.bias`).
void rename_tensors_containing(gguf_context * meta, ggml_context * ctx,
                               const char * needle, const char * replacement) {
    std::vector<std::pair<std::string, std::string>> renames;
    const int64_t n = gguf_get_n_tensors(meta);
    const size_t needle_len = std::strlen(needle);
    for (int64_t i = 0; i < n; ++i) {
        std::string s(gguf_get_tensor_name(meta, i));
        const size_t pos = s.find(needle);
        if (pos == std::string::npos) continue;
        std::string ns = s;
        ns.replace(pos, needle_len, replacement);
        renames.emplace_back(std::move(s), std::move(ns));
    }
    for (const auto & [from, to] : renames) rename_tensor(meta, ctx, from.c_str(), to.c_str());
}

// -------------------------------------------------------------------------
// per-loader state (currently just the "drop these tensor prefixes" list)
// -------------------------------------------------------------------------

std::mutex g_registry_mutex;
std::unordered_map<const llama_model_loader *, std::vector<std::string>> g_skip_prefixes;

void add_skip_prefix(const llama_model_loader * ml, std::string prefix) {
    std::lock_guard<std::mutex> lk(g_registry_mutex);
    g_skip_prefixes[ml].push_back(std::move(prefix));
}

// -------------------------------------------------------------------------
// F16 -> F32 tensor promotion (needed for Metal IM2COL on gemma3 conv weights)
// -------------------------------------------------------------------------

std::mutex g_promote_mutex;
std::unordered_set<std::string> g_promote_f16_to_f32;

// Set a tensor's type + strides in a ggml_context. The companion to this is
// the `maybe_load_tensor` read hook, which converts F16 bytes from disk into
// the newly-wider F32 buffer at load time.
void promote_tensor_to_f32(ggml_context * ctx, const char * name) {
    ggml_tensor * t = ggml_get_tensor(ctx, name);
    if (!t) return;
    t->type = GGML_TYPE_F32;
    t->nb[0] = ggml_type_size(GGML_TYPE_F32);
    t->nb[1] = t->nb[0] * (t->ne[0] / ggml_blck_size(GGML_TYPE_F32));
    for (int i = 2; i < GGML_MAX_DIMS; ++i) t->nb[i] = t->nb[i - 1] * t->ne[i - 1];

    std::lock_guard<std::mutex> lk(g_promote_mutex);
    g_promote_f16_to_f32.insert(name);
}

// -------------------------------------------------------------------------
// gemma3 (text side)
// -------------------------------------------------------------------------

// Returns true if this looks like an Ollama-format gemma3 blob. Requires
// the file to declare itself gemma3 (either via general.architecture or
// by having at least one gemma3.* KV), AND to exhibit at least one Ollama
// quirk. Different Ollama converter versions produced different quirks
// (4B/12B/27B have embedded vision + mm KVs; 1B uses non-standard rope
// key names; all of them omit layer_norm_rms_epsilon).
bool detect_ollama_gemma3(const gguf_context * meta, const ggml_context * ctx) {
    // Claim #1: the file is gemma3.
    const int64_t arch_kid = gguf_find_key(meta, "general.architecture");
    if (arch_kid < 0) return false;
    if (std::strcmp(gguf_get_val_str(meta, arch_kid), "gemma3") != 0) return false;

    // Claim #2: at least one Ollama-ism. An upstream-converted gemma3 would
    // have none of these (except possibly the v./mm. prefixes, which upstream
    // never ships in the text file — they live in a separate mmproj).
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
    // them to the flat names upstream expects. Copy-if-missing order
    // matters: we want real values to take priority over injected defaults.
    copy_f32_kv(meta, "gemma3.rope.global.freq_base", "gemma3.rope.freq_base");
    copy_f32_kv(meta, "gemma3.rope.local.freq_base",  "gemma3.rope.freq_base_swa");

    // Inject required KVs with their standard gemma3 defaults (no-op if
    // already present).
    if (!has_key(meta, "gemma3.attention.layer_norm_rms_epsilon"))
        gguf_set_val_f32(meta, "gemma3.attention.layer_norm_rms_epsilon", 1e-6f);
    if (!has_key(meta, "gemma3.rope.freq_base"))
        gguf_set_val_f32(meta, "gemma3.rope.freq_base", 1000000.0f);
    if (!has_key(meta, "gemma3.rope.freq_base_swa"))
        gguf_set_val_f32(meta, "gemma3.rope.freq_base_swa", 10000.0f);

    // Gemma3 4B/12B/27B ship with {type: "linear", factor: 8.0} rope scaling
    // in their HF config to extend the 16k trained context to 131072. Ollama's
    // old converter didn't write these. The 1B has no scaling — detect by
    // context length.
    int64_t ctx_key = gguf_find_key(meta, "gemma3.context_length");
    if (ctx_key >= 0 && gguf_get_val_u32(meta, ctx_key) >= 131072
            && !has_key(meta, "gemma3.rope.scaling.factor")) {
        gguf_set_val_str(meta, "gemma3.rope.scaling.type", "linear");
        gguf_set_val_f32(meta, "gemma3.rope.scaling.factor", 8.0f);
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

// -------------------------------------------------------------------------
// qwen35moe (text side)
// -------------------------------------------------------------------------

bool detect_ollama_qwen35moe(const gguf_context * meta, const ggml_context * ctx) {
    // Strongest markers: vision KVs live in-file (upstream splits to mmproj)
    // or MTP tensors are present (upstream strips them).
    if (has_key(meta, "qwen35moe.vision.block_count"))     return true;
    if (has_key(meta, "qwen35moe.image_token_id"))         return true;
    if (has_key(meta, "qwen35moe.ssm.v_head_reordered"))   return true;
    if (has_key(meta, "qwen35moe.feed_forward_length"))    return true; // upstream omits (=0 stored)
    if (has_key(meta, "qwen35moe.rope.mrope_interleaved")) return true;
    if (any_tensor_with_prefix(ctx, "mtp."))               return true;
    if (any_tensor_with_prefix(ctx, "v."))                 return true;

    // Scalar-vs-array: upstream writes head_count_kv as UINT32; Ollama wrote
    // it as a per-layer array. has_key alone can't tell us that, but a mismatch
    // shows up as a type-mismatch crash downstream, which is worse than over-
    // detecting. If any of the above markers fire we'll normalize it below.
    return false;
}

void handle_qwen35moe(const llama_model_loader * ml, gguf_context * meta, ggml_context * ctx) {
    if (!detect_ollama_qwen35moe(meta, ctx)) return;

    LLAMA_LOG_INFO("%s: detected Ollama-format qwen35moe GGUF; applying compatibility fixes\n", __func__);

    // 1. attention.head_count_kv — upstream expects UINT32; Ollama wrote
    //    an array (one entry per layer, 0 for SSM layers, 2 for attention
    //    layers). Collapse to the max non-zero value.
    {
        const int64_t kid = gguf_find_key(meta, "qwen35moe.attention.head_count_kv");
        if (kid >= 0 && gguf_get_kv_type(meta, kid) == GGUF_TYPE_ARRAY) {
            const size_t n = gguf_get_arr_n(meta, kid);
            const auto * arr = static_cast<const uint32_t *>(gguf_get_arr_data(meta, kid));
            uint32_t max_kv = 0;
            for (size_t i = 0; i < n; ++i) if (arr[i] > max_kv) max_kv = arr[i];
            if (max_kv == 0) max_kv = 2; // safety fallback
            gguf_remove_key(meta, "qwen35moe.attention.head_count_kv");
            gguf_set_val_u32(meta, "qwen35moe.attention.head_count_kv", max_kv);
        }
    }

    // 2. rope.dimension_sections — upstream expects a 4-element array
    //    (M-RoPE convention); Ollama wrote 3 elements. Pad with a trailing 0.
    {
        const int64_t kid = gguf_find_key(meta, "qwen35moe.rope.dimension_sections");
        if (kid >= 0 && gguf_get_arr_n(meta, kid) == 3) {
            const auto * src = static_cast<const int32_t *>(gguf_get_arr_data(meta, kid));
            const int32_t padded[4] = { src[0], src[1], src[2], 0 };
            gguf_set_arr_data(meta, "qwen35moe.rope.dimension_sections",
                              GGUF_TYPE_INT32, padded, 4);
        }
    }

    // 3. Tensor rename: Ollama's `blk.N.ssm_dt` corresponds to upstream's
    //    `blk.N.ssm_dt.bias` (same shape, F32 [32]). 40 layers.
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
    //    (vision goes to clip via --mmproj; MTP isn't used by upstream.)
    add_skip_prefix(ml, "v.");
    add_skip_prefix(ml, "mm.");
    add_skip_prefix(ml, "mtp.");
}

// -------------------------------------------------------------------------
// gemma3 (clip side)
// -------------------------------------------------------------------------

// Ollama -> upstream tensor-name renames. Applied via substring match, so
// both `.weight` and `.bias` variants are covered with one entry each.
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
    // Synthesize clip.vision.* from gemma3.vision.* (same values, different key).
    copy_u32_kv(meta, "gemma3.vision.block_count",                   "clip.vision.block_count");
    copy_u32_kv(meta, "gemma3.vision.embedding_length",              "clip.vision.embedding_length");
    copy_u32_kv(meta, "gemma3.vision.feed_forward_length",           "clip.vision.feed_forward_length");
    copy_u32_kv(meta, "gemma3.vision.image_size",                    "clip.vision.image_size");
    copy_u32_kv(meta, "gemma3.vision.patch_size",                    "clip.vision.patch_size");
    copy_u32_kv(meta, "gemma3.vision.attention.head_count",          "clip.vision.attention.head_count");
    copy_f32_kv(meta, "gemma3.vision.attention.layer_norm_epsilon",  "clip.vision.attention.layer_norm_epsilon");
    // projection_dim = text model's embedding_length (mmproj out == LM in).
    copy_u32_kv(meta, "gemma3.embedding_length",                     "clip.vision.projection_dim");

    // image_mean / image_std are constants for gemma3 vision.
    if (!has_key(meta, "clip.vision.image_mean")) {
        const float mean[3] = {0.5f, 0.5f, 0.5f};
        gguf_set_arr_data(meta, "clip.vision.image_mean", GGUF_TYPE_FLOAT32, mean, 3);
    }
    if (!has_key(meta, "clip.vision.image_std")) {
        const float std_[3] = {0.5f, 0.5f, 0.5f};
        gguf_set_arr_data(meta, "clip.vision.image_std", GGUF_TYPE_FLOAT32, std_, 3);
    }

    if (!has_key(meta, "clip.has_vision_encoder")) gguf_set_val_bool(meta, "clip.has_vision_encoder", true);
    if (!has_key(meta, "clip.use_gelu"))           gguf_set_val_bool(meta, "clip.use_gelu",           true);
    gguf_set_val_str(meta, "clip.projector_type",  "gemma3");
    gguf_set_val_str(meta, "general.architecture", "clip");

    for (const auto & [from, to] : kGemma3ClipRenames) {
        rename_tensors_containing(meta, ctx, from, to);
    }

    // Upstream stores patch_embd/position_embd as F32 (Gemma3VisionModel
    // tensor_force_quant); Ollama stored F16. Metal's IM2COL convolution
    // requires F32, so promote both at load time.
    promote_tensor_to_f32(ctx, "v.patch_embd.weight");
    promote_tensor_to_f32(ctx, "v.position_embd.weight");
}

} // anonymous namespace

// -------------------------------------------------------------------------
// public entry points
// -------------------------------------------------------------------------

void translate_metadata(const llama_model_loader * ml,
                        gguf_context * meta,
                        ggml_context * ctx,
                        std::string & arch_name) {
    if (!meta) return;
    if (arch_name == "gemma3")    handle_gemma3(ml, meta, ctx);
    if (arch_name == "qwen35moe") handle_qwen35moe(ml, meta, ctx);
    // Dispatch. Add more arches as they are wired up.
}

void translate_clip_metadata(gguf_context * meta, ggml_context * ctx) {
    if (!meta) return;
    // Require both the gemma3 markers AND embedded vision tensors to fire.
    if (detect_ollama_gemma3(meta, ctx) && any_tensor_with_prefix(ctx, "v.")) {
        LLAMA_LOG_INFO("%s: detected Ollama-format gemma3 GGUF used as mmproj; translating\n", __func__);
        handle_gemma3_clip(meta, ctx);
    }
}

bool should_skip_tensor(const llama_model_loader * ml, const char * tensor_name) {
    std::lock_guard<std::mutex> lk(g_registry_mutex);
    auto it = g_skip_prefixes.find(ml);
    if (it == g_skip_prefixes.end()) return false;
    for (const auto & prefix : it->second) {
        if (std::strncmp(tensor_name, prefix.c_str(), prefix.size()) == 0) return true;
    }
    return false;
}

bool maybe_load_tensor(ggml_tensor * cur,
                       const char * source_file,
                       size_t file_offset,
                       ggml_backend_buffer_type_t buft) {
    {
        std::lock_guard<std::mutex> lk(g_promote_mutex);
        if (g_promote_f16_to_f32.find(ggml_get_name(cur)) == g_promote_f16_to_f32.end()) return false;
    }
    if (cur->type != GGML_TYPE_F32) return false;

    const size_t n_elem   = ggml_nelements(cur);
    const size_t src_size = n_elem * sizeof(uint16_t);
    const size_t dst_size = n_elem * sizeof(float);

    std::vector<uint8_t> src(src_size);
    FILE * f = std::fopen(source_file, "rb");
    if (!f || std::fseek(f, (long) file_offset, SEEK_SET) != 0
           || std::fread(src.data(), 1, src_size, f) != src_size) {
        if (f) std::fclose(f);
        LLAMA_LOG_ERROR("%s: failed to read F16 bytes for '%s'\n", __func__, ggml_get_name(cur));
        return false;
    }
    std::fclose(f);

    std::vector<uint8_t> dst(dst_size);
    const uint16_t * sp = reinterpret_cast<const uint16_t *>(src.data());
    float          * dp = reinterpret_cast<float *>(dst.data());
    for (size_t i = 0; i < n_elem; ++i) dp[i] = ggml_fp16_to_fp32(sp[i]);

    if (ggml_backend_buft_is_host(buft)) std::memcpy(cur->data, dst.data(), dst_size);
    else                                 ggml_backend_tensor_set(cur, dst.data(), 0, dst_size);

    LLAMA_LOG_INFO("%s: promoted F16->F32 for %s (%zu elems)\n", __func__, ggml_get_name(cur), n_elem);
    return true;
}

} // namespace llama_ollama_compat
