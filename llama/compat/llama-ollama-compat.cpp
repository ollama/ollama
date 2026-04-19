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
// Load-time tensor transforms (registry consumed by maybe_load_tensor)
//
// Each registered op produces the final bytes for a single destination
// tensor by reading + transforming bytes from the source GGUF file.
// Used for F16->F32 promotion, QKV merging, and patch-embed splitting.
// -------------------------------------------------------------------------

struct LoadOp {
    // apply() reads what it needs from `src_file` and fills `dst` (dst_size
    // bytes). Returns false on failure.
    std::function<bool(const char * src_file, void * dst, size_t dst_size)> apply;
    const char * description;
};

std::mutex g_loadop_mutex;
std::unordered_map<std::string, LoadOp> g_loadops;

void register_load_op(std::string dest_name, LoadOp op) {
    std::lock_guard<std::mutex> lk(g_loadop_mutex);
    g_loadops[std::move(dest_name)] = std::move(op);
}

// Helper: read `size` bytes at `offset` from `path` into `dst`.
bool read_at(const char * path, size_t offset, void * dst, size_t size) {
    FILE * f = std::fopen(path, "rb");
    if (!f) return false;
    bool ok = (std::fseek(f, (long) offset, SEEK_SET) == 0
               && std::fread(dst, 1, size, f) == size);
    std::fclose(f);
    return ok;
}

// Capture a tensor's absolute file offset BEFORE any rename or reshape.
size_t tensor_file_offset(const gguf_context * meta, const char * name) {
    const int64_t id = gguf_find_tensor(meta, name);
    if (id < 0) return 0;
    return gguf_get_data_offset(meta) + gguf_get_tensor_offset(meta, id);
}

// Set a tensor's type and recompute strides in a ggml_context.
void set_tensor_type(ggml_tensor * t, ggml_type type) {
    t->type  = type;
    t->nb[0] = ggml_type_size(type);
    t->nb[1] = t->nb[0] * (t->ne[0] / ggml_blck_size(type));
    for (int i = 2; i < GGML_MAX_DIMS; ++i) t->nb[i] = t->nb[i - 1] * t->ne[i - 1];
}

// Set a tensor's shape and recompute strides in a ggml_context.
void set_tensor_shape(ggml_tensor * t, std::initializer_list<int64_t> shape) {
    int i = 0;
    for (auto v : shape) t->ne[i++] = v;
    for (; i < GGML_MAX_DIMS; ++i) t->ne[i] = 1;
    set_tensor_type(t, t->type);
}

// Promote a tensor F16 -> F32. The disk bytes stay F16; we register a
// load op that converts on read.
void promote_tensor_to_f32(gguf_context * meta, ggml_context * ctx, const char * name) {
    const int64_t tid = gguf_find_tensor(meta, name);
    if (tid < 0) return;
    ggml_tensor * t = ggml_get_tensor(ctx, name);
    if (!t || t->type != GGML_TYPE_F16) return;

    const size_t src_offset = tensor_file_offset(meta, name);
    const size_t n_elem     = ggml_nelements(t);
    const size_t src_size   = n_elem * sizeof(uint16_t);

    set_tensor_type(t, GGML_TYPE_F32);

    register_load_op(name, LoadOp{
        [src_offset, src_size, n_elem](const char * path, void * dst, size_t dst_size) {
            (void) dst_size;
            std::vector<uint8_t> src(src_size);
            if (!read_at(path, src_offset, src.data(), src_size)) return false;
            const uint16_t * sp = reinterpret_cast<const uint16_t *>(src.data());
            float          * dp = reinterpret_cast<float *>(dst);
            for (size_t i = 0; i < n_elem; ++i) dp[i] = ggml_fp16_to_fp32(sp[i]);
            return true;
        },
        "F16->F32 promote",
    });
}

// Concatenate N source tensors into one destination tensor. Captures
// source file offsets and sizes at registration time so later renames or
// reshapes don't affect the read. Layout assumption: the source tensors
// concatenate cleanly along their slowest dim, which in C/ggml order
// means the destination's bytes are just src[0] || src[1] || ... .
void register_concat_load(const gguf_context * meta, std::string dest_name,
                          const std::vector<std::string> & src_names) {
    std::vector<std::pair<size_t, size_t>> regions; // (offset, size)
    regions.reserve(src_names.size());
    for (const auto & n : src_names) {
        const int64_t id = gguf_find_tensor(meta, n.c_str());
        if (id < 0) return; // bail; downstream will fail loudly
        regions.emplace_back(
            gguf_get_data_offset(meta) + gguf_get_tensor_offset(meta, id),
            gguf_get_tensor_size(meta, id));
    }
    register_load_op(std::move(dest_name), LoadOp{
        [regions](const char * path, void * dst, size_t dst_size) {
            size_t total = 0;
            for (auto & [_, sz] : regions) total += sz;
            if (total != dst_size) return false;
            uint8_t * p = static_cast<uint8_t *>(dst);
            for (auto & [off, sz] : regions) {
                if (!read_at(path, off, p, sz)) return false;
                p += sz;
            }
            return true;
        },
        "concat sources",
    });
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
    // Require the file to declare itself qwen35moe first.
    const int64_t arch_kid = gguf_find_key(meta, "general.architecture");
    if (arch_kid < 0) return false;
    if (std::strcmp(gguf_get_val_str(meta, arch_kid), "qwen35moe") != 0) return false;

    // Then: at least one Ollama-ism. Upstream qwen35moe text files have none
    // of these — the vision KVs move to mmproj, MTP tensors are dropped,
    // head_count_kv is a scalar not an array, and the various extra rope /
    // ssm KVs below are either absent or stored differently.
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
    promote_tensor_to_f32(meta, ctx, "v.patch_embd.weight");
    promote_tensor_to_f32(meta, ctx, "v.position_embd.weight");
}

// -------------------------------------------------------------------------
// qwen35moe (clip side)
// -------------------------------------------------------------------------

// Substring renames. One entry handles both `.weight` and `.bias` variants.
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

// Register a QKV merge for a single block: Ollama has separate attn_q,
// attn_k, attn_v tensors; upstream wants them concatenated along their
// slow axis. Capture source file offsets BEFORE renaming.
void register_qwen35moe_qkv_merge(gguf_context * meta, ggml_context * ctx, int block_idx) {
    char qname[64], kname[64], vname[64];
    std::snprintf(qname, sizeof(qname), "v.blk.%d.attn_q.weight",   block_idx);
    std::snprintf(kname, sizeof(kname), "v.blk.%d.attn_k.weight",   block_idx);
    std::snprintf(vname, sizeof(vname), "v.blk.%d.attn_v.weight",   block_idx);

    const ggml_tensor * q = ggml_get_tensor(ctx, qname);
    if (!q) return; // not a qwen35moe vision block

    // Set up the destination tensor. We rename attn_q -> attn_qkv and
    // widen its slow axis from [1152, 1152] to [1152, 3456] (3 * hidden).
    char qkv_w[64], qkv_b[64], qbias[64], kbias[64], vbias[64];
    std::snprintf(qkv_w, sizeof(qkv_w), "v.blk.%d.attn_qkv.weight", block_idx);
    std::snprintf(qkv_b, sizeof(qkv_b), "v.blk.%d.attn_qkv.bias",   block_idx);
    std::snprintf(qbias, sizeof(qbias), "v.blk.%d.attn_q.bias",     block_idx);
    std::snprintf(kbias, sizeof(kbias), "v.blk.%d.attn_k.bias",     block_idx);
    std::snprintf(vbias, sizeof(vbias), "v.blk.%d.attn_v.bias",     block_idx);

    // Capture source offsets for the concat BEFORE renaming.
    register_concat_load(meta, qkv_w, {qname, kname, vname});
    register_concat_load(meta, qkv_b, {qbias, kbias, vbias});

    // Rename attn_q -> attn_qkv and widen shape.
    rename_tensor(meta, ctx, qname, qkv_w);
    if (ggml_tensor * t = ggml_get_tensor(ctx, qkv_w)) {
        set_tensor_shape(t, {t->ne[0], t->ne[1] * 3});
    }
    // Rename attn_q.bias -> attn_qkv.bias and widen from [1152] to [3456].
    rename_tensor(meta, ctx, qbias, qkv_b);
    if (ggml_tensor * t = ggml_get_tensor(ctx, qkv_b)) {
        set_tensor_shape(t, {t->ne[0] * 3});
    }
}

// Register the patch_embed reshape + split + F16->F32.
//
// Source: one Ollama tensor `v.patch_embed.weight`, ggml shape
//   [h=16, w=16, t=2, packed=3456] F16
// where `packed` is the PyTorch row-major flattening of HF's
// [out_c=1152, in_c=3, ...] dim pair, so packed_c = c_out*3 + c_in.
//
// Destination: two upstream tensors with ggml shape
//   [h=16, w=16, c_in=3, c_out=1152] F32 each,
// one per temporal slice. Matches upstream's
//   yield data_torch[:, :, 0, ...]   # PyTorch [1152, 3, 16, 16]
//   yield data_torch[:, :, 1, ...]
// which reverses to ggml ne=[16, 16, 3, 1152] per slice.
//
// For each output element (h, w, c_in, c_out):
//   src_idx = h + w*W + t*W*H + (c_out*C_in + c_in)*W*H*T
//   dst_idx = h + w*W + c_in*W*H + c_out*W*H*C_in
void register_qwen35moe_patch_embed_split(gguf_context * meta, ggml_context * ctx) {
    const char * src_name = "v.patch_embed.weight";
    const int64_t tid = gguf_find_tensor(meta, src_name);
    if (tid < 0) return;

    ggml_tensor * src_t = ggml_get_tensor(ctx, src_name);
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
                        const uint16_t * in_base = sp + HW * (slice_idx + T * packed);
                        float * out_base = dp + HW * (c_in + CIN * c_out);
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
    // the slice-0 load op against its new name.
    rename_tensor(meta, ctx, src_name, "v.patch_embd.weight");
    ggml_tensor * dest0 = ggml_get_tensor(ctx, "v.patch_embd.weight");
    if (!dest0) return;
    set_tensor_shape(dest0, {16, 16, 3, 1152});
    set_tensor_type (dest0, GGML_TYPE_F32);
    register_load_op("v.patch_embd.weight", make_slice_op(0));

    // We need a sibling tensor `v.patch_embd.weight.1` in ctx_meta so clip's
    // get_tensor() can find it. ggml_new_tensor() would blow ctx_meta's
    // fixed memory pool (sized exactly for the original tensor count).
    // Instead, steal an unused slot: after the QKV merge, `v.blk.0.attn_k`
    // is orphaned in ctx_meta — clip never looks it up because it asks for
    // the merged `attn_qkv`. Rename it to our sibling and reshape.
    rename_tensor(meta, ctx, "v.blk.0.attn_k.weight", "v.patch_embd.weight.1");
    ggml_tensor * dest1 = ggml_get_tensor(ctx, "v.patch_embd.weight.1");
    if (!dest1) return;
    set_tensor_shape(dest1, {16, 16, 3, 1152});
    set_tensor_type (dest1, GGML_TYPE_F32);
    register_load_op("v.patch_embd.weight.1", make_slice_op(1));
}

void handle_qwen35moe_clip(gguf_context * meta, ggml_context * ctx) {
    LLAMA_LOG_INFO("%s: detected Ollama-format qwen35moe GGUF used as mmproj; translating\n", __func__);

    // KV synthesis: clip.vision.* from qwen35moe.vision.* (plus defaults).
    copy_u32_kv(meta, "qwen35moe.vision.block_count",                   "clip.vision.block_count");
    copy_u32_kv(meta, "qwen35moe.vision.embedding_length",              "clip.vision.embedding_length");
    copy_u32_kv(meta, "qwen35moe.vision.attention.head_count",          "clip.vision.attention.head_count");
    copy_u32_kv(meta, "qwen35moe.vision.patch_size",                    "clip.vision.patch_size");
    copy_u32_kv(meta, "qwen35moe.vision.spatial_merge_size",            "clip.vision.spatial_merge_size");
    copy_u32_kv(meta, "qwen35moe.vision.num_channels",                  "clip.vision.num_channels");
    // projection_dim is the text model's embedding_length (merger out dim).
    copy_u32_kv(meta, "qwen35moe.embedding_length",                     "clip.vision.projection_dim");

    // Ollama omitted these; defaults match reference (ref_Q3.5-35B-A3B mmproj).
    if (!has_key(meta, "clip.vision.feed_forward_length"))
        gguf_set_val_u32(meta, "clip.vision.feed_forward_length", 4304);
    if (!has_key(meta, "clip.vision.image_size"))
        gguf_set_val_u32(meta, "clip.vision.image_size", 768);
    if (!has_key(meta, "clip.vision.attention.layer_norm_epsilon"))
        gguf_set_val_f32(meta, "clip.vision.attention.layer_norm_epsilon", 1e-6f);

    // image_mean / image_std — constants for qwen3.5 vision.
    if (!has_key(meta, "clip.vision.image_mean")) {
        const float v[3] = {0.5f, 0.5f, 0.5f};
        gguf_set_arr_data(meta, "clip.vision.image_mean", GGUF_TYPE_FLOAT32, v, 3);
    }
    if (!has_key(meta, "clip.vision.image_std")) {
        const float v[3] = {0.5f, 0.5f, 0.5f};
        gguf_set_arr_data(meta, "clip.vision.image_std", GGUF_TYPE_FLOAT32, v, 3);
    }

    // is_deepstack_layers: qwen3.5 35B has no deepstack layers. Set a
    // 27-element array of False matching clip.vision.block_count.
    if (!has_key(meta, "clip.vision.is_deepstack_layers")) {
        uint8_t bools[27] = {};
        gguf_set_arr_data(meta, "clip.vision.is_deepstack_layers", GGUF_TYPE_BOOL, bools, 27);
    }

    if (!has_key(meta, "clip.has_vision_encoder")) gguf_set_val_bool(meta, "clip.has_vision_encoder", true);
    if (!has_key(meta, "clip.use_gelu"))           gguf_set_val_bool(meta, "clip.use_gelu",           true);
    gguf_set_val_str(meta, "clip.projector_type",  "qwen3vl_merger");
    gguf_set_val_str(meta, "general.architecture", "clip");

    // QKV merge per block. Runs BEFORE the substring renames so we can
    // reliably find attn_q / attn_k / attn_v by name.
    const int64_t n_blocks_key = gguf_find_key(meta, "clip.vision.block_count");
    const uint32_t n_blocks = n_blocks_key >= 0 ? gguf_get_val_u32(meta, n_blocks_key) : 27;
    for (uint32_t b = 0; b < n_blocks; ++b) {
        register_qwen35moe_qkv_merge(meta, ctx, (int) b);
    }

    // patch_embed: reshape + temporal split + F16->F32. Also BEFORE renames
    // because it references `v.patch_embed.weight` by name.
    register_qwen35moe_patch_embed_split(meta, ctx);

    // Substring renames (last). These handle the simple pos_embed, merger.*,
    // linear_fc1/2, norm1/2 conversions.
    for (const auto & [from, to] : kQwen35moeClipRenames) {
        rename_tensors_containing(meta, ctx, from, to);
    }

    // F16 -> F32 on position_embd after rename.
    promote_tensor_to_f32(meta, ctx, "v.position_embd.weight");
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
    (void) file_offset; // registered ops capture their own offsets

    LoadOp op;
    {
        std::lock_guard<std::mutex> lk(g_loadop_mutex);
        auto it = g_loadops.find(ggml_get_name(cur));
        if (it == g_loadops.end()) return false;
        op = it->second;
    }

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

} // namespace llama_ollama_compat
