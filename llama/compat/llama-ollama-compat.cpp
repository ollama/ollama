#include "llama-ollama-compat.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"
#include "llama-impl.h"
#include "llama-model-loader.h"

#include <cstdint>
#include <cstring>
#include <cstdio>
#include <functional>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace llama_ollama_compat {

namespace {

// ---- helpers -------------------------------------------------------------

bool has_key(const gguf_context * meta, const char * key) {
    return gguf_find_key(meta, key) >= 0;
}

void set_f32_if_missing(gguf_context * meta, const char * key, float value) {
    if (!has_key(meta, key)) {
        gguf_set_val_f32(meta, key, value);
    }
}

bool any_tensor_with_prefix(const ggml_context * ctx, const char * prefix) {
    const size_t plen = std::strlen(prefix);
    for (ggml_tensor * t = ggml_get_first_tensor(ctx); t; t = ggml_get_next_tensor(ctx, t)) {
        if (std::strncmp(ggml_get_name(t), prefix, plen) == 0) {
            return true;
        }
    }
    return false;
}

const ggml_tensor * find_tensor(const ggml_context * ctx, const char * name) {
    for (ggml_tensor * t = ggml_get_first_tensor(ctx); t; t = ggml_get_next_tensor(ctx, t)) {
        if (std::strcmp(ggml_get_name(t), name) == 0) return t;
    }
    return nullptr;
}

// Truncate a string-typed KV array to `new_n` entries. No-op if absent or
// already that size or smaller.
void truncate_str_arr(gguf_context * meta, const char * key, size_t new_n) {
    const int64_t kid = gguf_find_key(meta, key);
    if (kid < 0) return;
    const size_t cur_n = gguf_get_arr_n(meta, kid);
    if (new_n >= cur_n) return;

    std::vector<std::string> owned;
    owned.reserve(new_n);
    std::vector<const char *> ptrs;
    ptrs.reserve(new_n);
    for (size_t i = 0; i < new_n; ++i) {
        owned.emplace_back(gguf_get_arr_str(meta, kid, i));
    }
    for (const auto & s : owned) ptrs.push_back(s.c_str());
    gguf_set_arr_str(meta, key, ptrs.data(), new_n);
}

// Truncate a primitive-typed KV array to `new_n` entries.
void truncate_data_arr(gguf_context * meta, const char * key, gguf_type elem_type, size_t elem_size, size_t new_n) {
    const int64_t kid = gguf_find_key(meta, key);
    if (kid < 0) return;
    const size_t cur_n = gguf_get_arr_n(meta, kid);
    if (new_n >= cur_n) return;

    const void * data = gguf_get_arr_data(meta, kid);
    std::vector<uint8_t> copy(elem_size * new_n);
    std::memcpy(copy.data(), data, elem_size * new_n);
    gguf_set_arr_data(meta, key, elem_type, copy.data(), new_n);
}

// ---- per-loader state (skip lists + tensor transforms) -------------------

struct TransformSpec {
    std::function<bool(const std::string &)> matches;
    std::function<void(void *, size_t, ggml_type)> apply;
    const char * description;
};

struct LoaderState {
    std::vector<TransformSpec> transforms;
    std::vector<std::string>   skip_prefixes;
};

std::mutex g_registry_mutex;
std::unordered_map<const llama_model_loader *, LoaderState> g_registry;

void add_skip_prefix(const llama_model_loader * ml, std::string prefix) {
    std::lock_guard<std::mutex> lk(g_registry_mutex);
    g_registry[ml].skip_prefixes.push_back(std::move(prefix));
}

// ---- gemma3 --------------------------------------------------------------

// Returns true if this looks like an Ollama-format gemma3 blob. We collect
// several independent markers because different Ollama converter versions
// produced different quirks (the 4B has embedded vision, the 1B has
// non-standard rope key names, etc.) — any one marker flips detection on.
bool detect_ollama_gemma3(const gguf_context * meta, const ggml_context * ctx) {
    // Vision-capable gemma3 (4B/12B/27B): Ollama writes this key.
    if (has_key(meta, "gemma3.mm.tokens_per_image")) return true;

    // Embedded vision tensors in the main file. Upstream stores vision in
    // a separate mmproj file.
    if (any_tensor_with_prefix(ctx, "v.") ||
        any_tensor_with_prefix(ctx, "mm.")) return true;

    // Non-standard rope key names. Ollama's 1B converter used
    // `gemma3.rope.{global,local}.freq_base` instead of upstream's flat
    // `gemma3.rope.freq_base` / `gemma3.rope.freq_base_swa`.
    if (has_key(meta, "gemma3.rope.global.freq_base")) return true;
    if (has_key(meta, "gemma3.rope.local.freq_base"))  return true;

    // Tokenizer KVs Ollama writes but upstream doesn't.
    if (has_key(meta, "tokenizer.ggml.add_padding_token")) return true;
    if (has_key(meta, "tokenizer.ggml.add_unknown_token")) return true;

    // Required KV upstream always writes — its absence is a strong marker.
    if (!has_key(meta, "gemma3.attention.layer_norm_rms_epsilon")) return true;

    return false;
}

void handle_gemma3(const llama_model_loader * ml, gguf_context * meta, ggml_context * ctx) {
    if (!detect_ollama_gemma3(meta, ctx)) return;

    LLAMA_LOG_INFO("%s: detected Ollama-format gemma3 GGUF; applying compatibility fixes\n", __func__);

    // 1. Inject required KVs that Ollama's old converter omitted. Defaults
    //    are the gemma3 standard values; only injected if missing, so explicit
    //    values in a file take precedence.
    //
    //    Some older Ollama converters also used the non-standard keys
    //    `gemma3.rope.global.freq_base` and `gemma3.rope.local.freq_base`.
    //    llama.cpp reads only the flat names, so copy those over first so
    //    the has_key checks below don't trample real values.
    if (!has_key(meta, "gemma3.rope.freq_base")) {
        const int64_t k = gguf_find_key(meta, "gemma3.rope.global.freq_base");
        if (k >= 0) {
            gguf_set_val_f32(meta, "gemma3.rope.freq_base", gguf_get_val_f32(meta, k));
        }
    }
    if (!has_key(meta, "gemma3.rope.freq_base_swa")) {
        const int64_t k = gguf_find_key(meta, "gemma3.rope.local.freq_base");
        if (k >= 0) {
            gguf_set_val_f32(meta, "gemma3.rope.freq_base_swa", gguf_get_val_f32(meta, k));
        }
    }

    set_f32_if_missing(meta, "gemma3.attention.layer_norm_rms_epsilon", 1e-6f);
    set_f32_if_missing(meta, "gemma3.rope.freq_base", 1000000.0f);
    set_f32_if_missing(meta, "gemma3.rope.freq_base_swa", 10000.0f);

    // RoPE linear scaling: gemma3 4B/12B/27B ship with
    //   rope_scaling = { type: "linear", factor: 8.0 }
    // in their HF config. This extends the native ~16k trained context to
    // the declared 131072 token context. Ollama's old converter didn't
    // write these KVs; without them llama.cpp uses factor=1.0 which makes
    // all positional embeddings subtly wrong (coherent but off-distribution
    // output). The 1B variant has no rope_scaling — detect by context
    // length.
    {
        const int64_t ctx_key = gguf_find_key(meta, "gemma3.context_length");
        const uint32_t ctx_len = ctx_key >= 0 ? gguf_get_val_u32(meta, ctx_key) : 0;
        if (ctx_len >= 131072 && !has_key(meta, "gemma3.rope.scaling.factor")) {
            gguf_set_val_str(meta, "gemma3.rope.scaling.type", "linear");
            gguf_set_val_f32(meta, "gemma3.rope.scaling.factor", 8.0f);
        }
    }

    // 2. Tokenizer vocab size vs. embedding dim mismatch. Ollama's old
    //    converter leaves special/multimodal tokens (e.g. <image_soft_token>)
    //    in the tokenizer arrays even though the embedding matrix doesn't
    //    cover them. Truncate the tokenizer to match the embedding rows.
    if (const ggml_tensor * tok = find_tensor(ctx, "token_embd.weight")) {
        const size_t embd_rows = tok->ne[1]; // shape is [n_embd, n_vocab]
        truncate_str_arr (meta, "tokenizer.ggml.tokens",     embd_rows);
        truncate_data_arr(meta, "tokenizer.ggml.scores",     GGUF_TYPE_FLOAT32, sizeof(float),   embd_rows);
        truncate_data_arr(meta, "tokenizer.ggml.token_type", GGUF_TYPE_INT32,   sizeof(int32_t), embd_rows);
    }

    // 3. Drop embedded vision/projector tensors from the text loader.
    //    Ollama's Go wrapper extracts them to a sidecar mmproj file before
    //    passing --mmproj to llama-server.
    add_skip_prefix(ml, "v.");
    add_skip_prefix(ml, "mm.");

    // Note: no RMSNorm weight shift is required. Ollama's published gemma3
    // blobs already have the +1 shift baked in at conversion time — same as
    // upstream llama.cpp's convert_hf_to_gguf.py.
}

} // anonymous namespace

void translate_metadata(const llama_model_loader * ml,
                        gguf_context * meta,
                        ggml_context * ctx,
                        std::string & arch_name) {
    if (!meta) return;

    if (arch_name == "gemma3") {
        handle_gemma3(ml, meta, ctx);
    }
    // Dispatch. Add more arches as they are wired up.
}

// -------------------------------------------------------------------------
// Clip-side (mmproj) translation
// -------------------------------------------------------------------------

namespace {

// Rename a tensor in BOTH the gguf_context and the ggml_context so that all
// name-based lookups — offset map, ggml_get_tensor, tensor.name — agree.
//
// The gguf_context side is a bit sneaky: gguf_get_tensor_name returns a
// pointer into the embedded ggml_tensor's `name[GGML_MAX_NAME]` buffer.
// That buffer is non-const storage inside a std::vector element; the const
// on the return type is just API hygiene. Casting it away and strncpy'ing
// a new name is well-defined and avoids needing to patch gguf's internals.
void rename_tensor(gguf_context * meta, ggml_context * ctx,
                   const char * old_name, const char * new_name) {
    const int64_t id = gguf_find_tensor(meta, old_name);
    if (id < 0) return;

    // Update the gguf-side name (what gguf_get_tensor_name returns later).
    if (char * name_ptr = const_cast<char *>(gguf_get_tensor_name(meta, id))) {
        std::strncpy(name_ptr, new_name, GGML_MAX_NAME - 1);
        name_ptr[GGML_MAX_NAME - 1] = '\0';
    }

    // Update the ggml-side name (what ggml_get_tensor looks up by).
    if (ggml_tensor * t = ggml_get_tensor(ctx, old_name)) {
        ggml_set_name(t, new_name);
    }
}

// Rename every tensor whose name contains `needle` by replacing that
// substring with `replacement`. Applies to both `.weight` and `.bias`.
void rename_tensors_containing(gguf_context * meta, ggml_context * ctx,
                               const char * needle, const char * replacement) {
    // Collect names first — renaming while iterating would shift indices.
    std::vector<std::string> renames; // old -> new
    const int64_t n = gguf_get_n_tensors(meta);
    for (int64_t i = 0; i < n; ++i) {
        const char * name = gguf_get_tensor_name(meta, i);
        std::string s(name);
        size_t pos = s.find(needle);
        if (pos == std::string::npos) continue;
        std::string new_s = s;
        new_s.replace(pos, std::strlen(needle), replacement);
        renames.push_back(s);
        renames.push_back(std::move(new_s));
    }
    for (size_t i = 0; i + 1 < renames.size(); i += 2) {
        rename_tensor(meta, ctx, renames[i].c_str(), renames[i + 1].c_str());
    }
}

// Copy a KV from src_key to dst_key if src_key exists and dst_key doesn't.
template <typename Getter, typename Setter>
bool copy_kv(gguf_context * meta, const char * src_key, const char * dst_key,
             Getter get, Setter set) {
    if (has_key(meta, dst_key)) return true; // already set, keep explicit values
    const int64_t kid = gguf_find_key(meta, src_key);
    if (kid < 0) return false;
    set(meta, dst_key, get(meta, kid));
    return true;
}

void copy_u32_kv(gguf_context * meta, const char * src_key, const char * dst_key) {
    copy_kv(meta, src_key, dst_key,
            gguf_get_val_u32,
            [](gguf_context * m, const char * k, uint32_t v){ gguf_set_val_u32(m, k, v); });
}

void copy_f32_kv(gguf_context * meta, const char * src_key, const char * dst_key) {
    copy_kv(meta, src_key, dst_key,
            gguf_get_val_f32,
            [](gguf_context * m, const char * k, float v){ gguf_set_val_f32(m, k, v); });
}

void set_str(gguf_context * meta, const char * key, const char * value) {
    gguf_set_val_str(meta, key, value);
}

// Tensors marked for F16→F32 promotion. Looked up by tensor name.
// Populated by handle_gemma3_clip; consumed by supply_promoted_tensor_data.
std::mutex g_promote_mutex;
std::unordered_set<std::string> g_promote_f16_to_f32;

void mark_promote_f16_to_f32(const std::string & name) {
    std::lock_guard<std::mutex> lk(g_promote_mutex);
    g_promote_f16_to_f32.insert(name);
}

// Change a tensor's type in the ggml_context. Updates type and strides so
// that ggml_nbytes(t) returns the new-type size, and ggml_dup_tensor
// propagates the new type to any copies.
void set_tensor_type_in_ctx(ggml_context * ctx, const char * name, ggml_type new_type) {
    ggml_tensor * t = ggml_get_tensor(ctx, name);
    if (!t) return;
    t->type = new_type;
    t->nb[0] = ggml_type_size(new_type);
    t->nb[1] = t->nb[0] * (t->ne[0] / ggml_blck_size(new_type));
    for (int i = 2; i < GGML_MAX_DIMS; ++i) {
        t->nb[i] = t->nb[i - 1] * t->ne[i - 1];
    }
}

// Promote a tensor's type in both gguf_context and ggml_context. Used for
// F16→F32 conversion of conv weights that Metal requires as F32.
void promote_tensor_to_f32(gguf_context * meta, ggml_context * ctx, const char * name) {
    // Update ggml_context (clip.cpp reads type from here via ggml_dup_tensor).
    set_tensor_type_in_ctx(ctx, name, GGML_TYPE_F32);
    // Note: we do NOT call gguf_set_tensor_type on `meta`, because that
    // recomputes tensor data offsets based on the new type — but we still
    // have F16 bytes at the original offset. clip.cpp reads the offset from
    // its own tensor_offset map (populated from gguf_context BEFORE this
    // promotion), so leaving meta's offset alone preserves the correct
    // source location. We also don't use meta's type for sizing.
    mark_promote_f16_to_f32(name);
}

// Convert F16 → F32 in place.
void convert_f16_to_f32(const uint16_t * src, float * dst, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        dst[i] = ggml_fp16_to_fp32(src[i]);
    }
}

void handle_gemma3_clip(gguf_context * meta, ggml_context * ctx) {
    // Build clip.* KVs from the gemma3.vision.* KVs already in the file.
    copy_u32_kv(meta, "gemma3.vision.block_count",                   "clip.vision.block_count");
    copy_u32_kv(meta, "gemma3.vision.embedding_length",              "clip.vision.embedding_length");
    copy_u32_kv(meta, "gemma3.vision.feed_forward_length",           "clip.vision.feed_forward_length");
    copy_u32_kv(meta, "gemma3.vision.image_size",                    "clip.vision.image_size");
    copy_u32_kv(meta, "gemma3.vision.patch_size",                    "clip.vision.patch_size");
    copy_u32_kv(meta, "gemma3.vision.attention.head_count",          "clip.vision.attention.head_count");
    copy_f32_kv(meta, "gemma3.vision.attention.layer_norm_epsilon",  "clip.vision.attention.layer_norm_epsilon");
    // projection_dim is the TEXT model's embedding_length (the mmproj
    // output dim == language model input dim).
    copy_u32_kv(meta, "gemma3.embedding_length",                     "clip.vision.projection_dim");

    // image_mean / image_std — constant defaults for gemma3 vision.
    if (!has_key(meta, "clip.vision.image_mean")) {
        const float mean[3] = {0.5f, 0.5f, 0.5f};
        gguf_set_arr_data(meta, "clip.vision.image_mean", GGUF_TYPE_FLOAT32, mean, 3);
    }
    if (!has_key(meta, "clip.vision.image_std")) {
        const float std_[3] = {0.5f, 0.5f, 0.5f};
        gguf_set_arr_data(meta, "clip.vision.image_std", GGUF_TYPE_FLOAT32, std_, 3);
    }

    // Top-level clip flags.
    if (!has_key(meta, "clip.has_vision_encoder")) {
        gguf_set_val_bool(meta, "clip.has_vision_encoder", true);
    }
    if (!has_key(meta, "clip.use_gelu")) {
        gguf_set_val_bool(meta, "clip.use_gelu", true);
    }
    set_str(meta, "clip.projector_type", "gemma3");
    set_str(meta, "general.architecture", "clip");

    // Tensor name translation (Ollama -> upstream mtmd convention).
    rename_tensors_containing(meta, ctx, "v.patch_embedding",      "v.patch_embd");
    rename_tensors_containing(meta, ctx, "v.position_embedding",   "v.position_embd");
    rename_tensors_containing(meta, ctx, "v.post_layernorm",       "v.post_ln");
    rename_tensors_containing(meta, ctx, ".layer_norm1",           ".ln1");
    rename_tensors_containing(meta, ctx, ".layer_norm2",           ".ln2");
    rename_tensors_containing(meta, ctx, ".attn_output",           ".attn_out");
    rename_tensors_containing(meta, ctx, ".mlp.fc1",               ".ffn_down");
    rename_tensors_containing(meta, ctx, ".mlp.fc2",               ".ffn_up");
    rename_tensors_containing(meta, ctx, "mm.mm_input_projection", "mm.input_projection");
    rename_tensors_containing(meta, ctx, "mm.mm_soft_emb_norm",    "mm.soft_emb_norm");

    // Promote F16 patch-embed / position-embed to F32. Upstream stores these
    // as F32 (see Gemma3VisionModel.tensor_force_quant in convert_hf_to_gguf.py).
    // Metal's IM2COL op requires F32 for these convolution inputs.
    promote_tensor_to_f32(meta, ctx, "v.patch_embd.weight");
    promote_tensor_to_f32(meta, ctx, "v.position_embd.weight");
}

} // anonymous namespace

void translate_clip_metadata(gguf_context * meta, ggml_context * ctx) {
    if (!meta) return;

    // Detection: Ollama-format gemma3 blob has `gemma3.mm.tokens_per_image`
    // plus embedded `v.*` tensors. Upstream mmproj files use `general.architecture=clip`
    // and don't have gemma3.* KVs.
    if (has_key(meta, "gemma3.mm.tokens_per_image") &&
        any_tensor_with_prefix(ctx, "v.")) {
        LLAMA_LOG_INFO("%s: detected Ollama-format gemma3 GGUF used as mmproj; translating\n", __func__);
        handle_gemma3_clip(meta, ctx);
    }
}

bool maybe_load_tensor(ggml_tensor * cur,
                       const char * source_file,
                       size_t file_offset,
                       ggml_backend_buffer_type_t buft) {
    // Check registry: is this tensor marked for F16→F32 promotion?
    {
        std::lock_guard<std::mutex> lk(g_promote_mutex);
        if (g_promote_f16_to_f32.find(ggml_get_name(cur)) == g_promote_f16_to_f32.end()) {
            return false;
        }
    }
    // Destination was promoted to F32 by translate_clip_metadata. Source
    // bytes on disk are still F16 at file_offset.
    if (cur->type != GGML_TYPE_F32) return false;

    const size_t n_elem   = ggml_nelements(cur);
    const size_t src_size = n_elem * sizeof(uint16_t);
    const size_t dst_size = n_elem * sizeof(float);

    std::vector<uint8_t> src(src_size);

    FILE * f = std::fopen(source_file, "rb");
    if (!f) {
        LLAMA_LOG_ERROR("%s: failed to open '%s'\n", __func__, source_file);
        return false;
    }
    if (std::fseek(f, (long) file_offset, SEEK_SET) != 0 ||
        std::fread(src.data(), 1, src_size, f) != src_size) {
        std::fclose(f);
        LLAMA_LOG_ERROR("%s: failed to read %zu bytes for '%s'\n",
                        __func__, src_size, ggml_get_name(cur));
        return false;
    }
    std::fclose(f);

    std::vector<uint8_t> dst(dst_size);
    convert_f16_to_f32(reinterpret_cast<const uint16_t *>(src.data()),
                       reinterpret_cast<float *>(dst.data()),
                       n_elem);

    // Deliver the converted bytes to the tensor's final backend buffer.
    if (ggml_backend_buft_is_host(buft)) {
        std::memcpy(cur->data, dst.data(), dst_size);
    } else {
        ggml_backend_tensor_set(cur, dst.data(), 0, dst_size);
    }

    LLAMA_LOG_INFO("%s: promoted F16->F32 for %s (%zu elems)\n",
                   __func__, ggml_get_name(cur), n_elem);
    return true;
}

bool should_skip_tensor(const llama_model_loader * ml, const char * tensor_name) {
    std::lock_guard<std::mutex> lk(g_registry_mutex);
    auto it = g_registry.find(ml);
    if (it == g_registry.end()) return false;
    for (const auto & prefix : it->second.skip_prefixes) {
        if (std::strncmp(tensor_name, prefix.c_str(), prefix.size()) == 0) {
            return true;
        }
    }
    return false;
}

void apply_tensor_transforms(const llama_model_loader * ml, ggml_context * ctx) {
    std::vector<TransformSpec> specs;
    {
        std::lock_guard<std::mutex> lk(g_registry_mutex);
        auto it = g_registry.find(ml);
        if (it == g_registry.end()) return;
        specs = it->second.transforms;
    }
    if (specs.empty()) return;

    std::vector<uint8_t> buf;
    for (ggml_tensor * t = ggml_get_first_tensor(ctx); t; t = ggml_get_next_tensor(ctx, t)) {
        if (!t->buffer) continue;
        const std::string name = ggml_get_name(t);
        for (const auto & spec : specs) {
            if (!spec.matches(name)) continue;

            const size_t nbytes = ggml_nbytes(t);
            const size_t n_elem = ggml_nelements(t);

            buf.resize(nbytes);
            ggml_backend_tensor_get(t, buf.data(), 0, nbytes);
            spec.apply(buf.data(), n_elem, t->type);
            ggml_backend_tensor_set(t, buf.data(), 0, nbytes);
        }
    }
}

} // namespace llama_ollama_compat
