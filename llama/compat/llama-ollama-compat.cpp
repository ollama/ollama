#include "llama-ollama-compat.h"
#include "llama-ollama-compat-util.h"

#include "llama-impl.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

namespace llama_ollama_compat {

using namespace llama_ollama_compat::detail; // pull detail:: helpers into scope

namespace {

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
// qwen35moe (text side)
// =========================================================================

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

    // 1. attention.head_count_kv — upstream expects UINT32; Ollama wrote
    //    an array (one entry per layer, 0 for SSM layers, 2 for attention).
    //    Collapse to the max non-zero value.
    {
        const int64_t kid = gguf_find_key(meta, "qwen35moe.attention.head_count_kv");
        if (kid >= 0 && gguf_get_kv_type(meta, kid) == GGUF_TYPE_ARRAY) {
            const size_t n = gguf_get_arr_n(meta, kid);
            const auto * arr = static_cast<const uint32_t *>(gguf_get_arr_data(meta, kid));
            uint32_t max_kv = 0;
            for (size_t i = 0; i < n; ++i) if (arr[i] > max_kv) max_kv = arr[i];
            if (max_kv == 0) max_kv = 2; // safety fallback
            gguf_remove_key  (meta, "qwen35moe.attention.head_count_kv");
            gguf_set_val_u32 (meta, "qwen35moe.attention.head_count_kv", max_kv);
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

    // 3. Tensor rename: Ollama's `blk.N.ssm_dt` is upstream's
    //    `blk.N.ssm_dt.bias` (same shape). 40 layers.
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

} // anonymous namespace

// =========================================================================
// Public entry points
// =========================================================================

void translate_metadata(const llama_model_loader * ml,
                        gguf_context * meta,
                        ggml_context * ctx,
                        std::string & arch_name) {
    if (!meta) return;
    if (arch_name == "gemma3")    handle_gemma3   (ml, meta, ctx);
    if (arch_name == "qwen35moe") handle_qwen35moe(ml, meta, ctx);
    if (arch_name == "gptoss")    handle_gptoss   (ml, meta, ctx, arch_name);
    if (arch_name == "lfm2")      handle_lfm2     (ml, meta, ctx);
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

} // namespace llama_ollama_compat
