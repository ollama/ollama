// Regression test for ollama#19457: a context shift must not rescale the KV
// cache keys.
//
// A context shift re-ropes keys that are already roped, moving them to new
// relative positions. Only the rotational delta belongs on that path. llama.cpp
// forwards the forward-pass rope params into the shift graph, including the
// YaRN amplitude, so ggml's rope_yarn() multiplies every cached key by
// `attn_factor * (1 + 0.1 * log(1 / freq_scale))` once per shift.
//
// This test does not reproduce that formula. It loads a synthetic model, forces
// YaRN scaling on it, populates the KV cache through the public API, calls
// llama_memory_seq_add() to trigger the shift graph, and measures the actual
// key magnitudes. Without the fix in llama/compat/002-kv-shift-yarn-attn-factor.patch
// the ratios grow by ~1.1386x per shift; with it they stay at 1.0.
//
// The F32 cache exercises the in-place shift path. F16 and Q8_0 exercise the
// conversion-based path (dequantize -> rope -> requantize), which is what real
// servers use, with tolerances that account for the round trip.
//
// A second geometry uses gemma2, which alternates sliding-window and dense
// layers. Sliding-window layers take their rope parameters from the model
// (`rope_freq_scale_train_swa`) instead of the context (`cparams.rope_freq_scale`),
// so that scenario pins down the per-layer part of the fix: the SWA half of the
// cache must not be rescaled even when the context applies YaRN.
//
// The models are written from scratch with the gguf API so the test needs no
// network access and no multi-gigabyte download.

#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"
#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

constexpr int N_EMBD     = 64;
constexpr int N_HEAD     = 2;
constexpr int N_HEAD_KV  = 2;
constexpr int N_FF       = 128;
constexpr int N_VOCAB    = 32;
constexpr int N_LAYER_LLAMA  = 1;
constexpr int N_LAYER_GEMMA2 = 2;   // gemma2 alternates SWA/dense layers
constexpr int N_CTX      = 128;
constexpr int N_ROT      = N_EMBD / N_HEAD;              // 32
constexpr int N_EMBD_K   = N_EMBD / N_HEAD;              // 32 per head
constexpr int N_EMBD_K_GQA = N_EMBD_K * N_HEAD_KV;       // 64
constexpr int N_PROMPT   = 8;                            // cells populated up front
constexpr int N_SHIFTS   = 6;                            // shifts to accumulate
constexpr int N_KEEP     = 4;                            // "keep prefix" cells

// ggml rope applies `attn_factor * (1 + 0.1 * log(1/freq_scale))` when
// ext_factor != 0. With freq_scale = 1/4 that is 1 + 0.1*log(4) = 1.138629.
constexpr float ROPE_FREQ_SCALE = 0.25f;
constexpr float EXPECTED_UNPATCHED_SCALE = 1.138629f;

// The bug is purely multiplicative growth, ~1.1386x per shift, so `grow` stays
// much tighter than the signal. `shrink` is the budget for lossy cache types:
// a dequantize -> rope -> requantize round trip perturbs the magnitude in both
// directions, mostly downward, by a fraction of a percent per shift.
struct bounds {
    double grow;    // largest tolerated relative increase
    double shrink;  // largest tolerated relative decrease
};

// Lossless F32 cache: the patched path should be exact to within a couple of
// ULPs, so 1e-3 leaves room for backend reassociation and nothing else.
constexpr bounds BOUNDS_F32 { 1e-3, 1e-3 };

// F16 keeps ~3 decimal digits. Observed drift is <1e-3 over six shifts.
constexpr bounds BOUNDS_F16 { 5e-3, 5e-3 };

// Q8_0 rounds every 32-value block to an 8-bit scale; observed drift is under
// 4% over six shifts. Unpatched code grows 15% on the first shift alone.
constexpr bounds BOUNDS_Q8_0 { 3e-2, 8e-2 };

int g_failures = 0;

void check(bool ok, const char * what) {
    std::printf("  %s %s\n", ok ? "[ OK ]" : "[FAIL]", what);
    if (!ok) {
        ++g_failures;
    }
}

struct ggml_context * g_tensor_ctx = nullptr;

// Deterministic filler so the test is reproducible.
void fill_tensor(ggml_tensor * t) {
    const int64_t n = ggml_nelements(t);
    float * data = static_cast<float *>(t->data);
    for (int64_t i = 0; i < n; ++i) {
        data[i] = 0.02f * std::sin(0.7f * static_cast<float>(i) + 0.3f);
    }
}

void add_tensor(gguf_context * fctx, const char * name, int64_t ne0, int64_t ne1 = 1) {
    ggml_tensor * t = ne1 == 1
        ? ggml_new_tensor_1d(g_tensor_ctx, GGML_TYPE_F32, ne0)
        : ggml_new_tensor_2d(g_tensor_ctx, GGML_TYPE_F32, ne0, ne1);
    ggml_set_name(t, name);
    fill_tensor(t);
    gguf_add_tensor(fctx, t);
}

std::string write_test_model(bool swa) {
    const char * path = swa ? "kv-shift-rope-test-swa.gguf" : "kv-shift-rope-test-model.gguf";
    const char * arch = swa ? "gemma2" : "llama";
    const std::string ap = std::string(arch) + ".";
    const int n_layer = swa ? N_LAYER_GEMMA2 : N_LAYER_LLAMA;

    gguf_context * fctx = gguf_init_empty();

    gguf_set_val_str(fctx, "general.architecture", arch);
    gguf_set_val_str(fctx, "general.name", "kv-shift-rope-test");

    gguf_set_val_u32(fctx, (ap + "block_count").c_str(), n_layer);
    gguf_set_val_u32(fctx, (ap + "context_length").c_str(), N_CTX);
    gguf_set_val_u32(fctx, (ap + "embedding_length").c_str(), N_EMBD);
    gguf_set_val_u32(fctx, (ap + "feed_forward_length").c_str(), N_FF);
    gguf_set_val_u32(fctx, (ap + "attention.head_count").c_str(), N_HEAD);
    gguf_set_val_u32(fctx, (ap + "attention.head_count_kv").c_str(), N_HEAD_KV);
    gguf_set_val_f32(fctx, (ap + "attention.layer_norm_rms_epsilon").c_str(), 1e-5f);
    gguf_set_val_u32(fctx, (ap + "rope.dimension_count").c_str(), N_ROT);
    gguf_set_val_f32(fctx, (ap + "rope.freq_base").c_str(), 10000.0f);
    gguf_set_val_u32(fctx, (ap + "vocab_size").c_str(), N_VOCAB);

    if (swa) {
        // load_swa_pattern(ml, 2) marks even layers SWA and odd layers dense by
        // default, but writing the key explicitly keeps the fixture independent
        // of that default. gemma2 then reports rope_freq_scale_train_swa = 1.0.
        gguf_set_val_u32(fctx, "gemma2.attention.sliding_window", 4096);
        gguf_set_val_u32(fctx, "gemma2.attention.sliding_window_pattern", 2);
    }

    // Minimal llama-style vocabulary: the test never needs real tokenization,
    // it only needs llama_model_load_from_file() to build a vocab.
    std::vector<const char *> tokens;
    std::vector<std::string>   storage;
    storage.reserve(N_VOCAB);
    for (int i = 0; i < N_VOCAB; ++i) {
        storage.push_back("<t" + std::to_string(i) + ">");
    }
    for (auto & s : storage) {
        tokens.push_back(s.c_str());
    }

    std::vector<float>   scores(N_VOCAB, 0.0f);
    std::vector<int32_t> types(N_VOCAB, 1); // LLAMA_TOKEN_TYPE_NORMAL
    types[0] = 2;                           // unknown
    types[1] = 3;                           // control (BOS)
    types[2] = 3;                           // control (EOS)

    gguf_set_val_str(fctx, "tokenizer.ggml.model", "llama");
    gguf_set_arr_str(fctx, "tokenizer.ggml.tokens", tokens.data(), tokens.size());
    gguf_set_arr_data(fctx, "tokenizer.ggml.scores", GGUF_TYPE_FLOAT32, scores.data(), scores.size());
    gguf_set_arr_data(fctx, "tokenizer.ggml.token_type", GGUF_TYPE_INT32, types.data(), types.size());
    gguf_set_val_u32(fctx, "tokenizer.ggml.bos_token_id", 1);
    gguf_set_val_u32(fctx, "tokenizer.ggml.eos_token_id", 2);

    add_tensor(fctx, "token_embd.weight", N_EMBD, N_VOCAB);
    add_tensor(fctx, "output_norm.weight", N_EMBD);
    if (!swa) {
        // gemma2 reuses token_embd.weight as the output matrix (TENSOR_DUPLICATED)
        // and therefore has no output.weight tensor of its own.
        add_tensor(fctx, "output.weight", N_EMBD, N_VOCAB);
    }

    for (int il = 0; il < n_layer; ++il) {
        const std::string p = "blk." + std::to_string(il) + ".";

        auto add = [&](const char * suffix, int64_t ne0, int64_t ne1 = 1) {
            add_tensor(fctx, (p + suffix).c_str(), ne0, ne1);
        };

        add("attn_norm.weight", N_EMBD);
        add("attn_q.weight", N_EMBD, N_EMBD_K * N_HEAD);
        add("attn_k.weight", N_EMBD, N_EMBD_K_GQA);
        add("attn_v.weight", N_EMBD, N_EMBD_K_GQA);
        add("attn_output.weight", N_EMBD_K * N_HEAD, N_EMBD);
        add("ffn_norm.weight", N_EMBD);
        add("ffn_gate.weight", N_EMBD, N_FF);
        add("ffn_down.weight", N_FF, N_EMBD);
        add("ffn_up.weight", N_EMBD, N_FF);

        if (swa) {
            // gemma2 applies an extra RMS norm after attention and after the FFN.
            add("post_attention_norm.weight", N_EMBD);
            add("post_ffw_norm.weight", N_EMBD);
        }
    }

    if (!gguf_write_to_file(fctx, path, false)) {
        std::fprintf(stderr, "failed to write %s\n", path);
        std::exit(2);
    }

    gguf_free(fctx);
    return path;
}

// Reading the keys back needs the serialized layout from
// llama_context::state_seq_get_data() and llama_kv_cache::state_write*():
//
//   uint32 io_magic, int32 seq_id
//   one or more memory sections (llama_kv_cache_iswa writes the dense/base
//   state and then the SWA state; a plain cache writes a single section):
//     uint32 n_stream
//     per stream: uint32 cell_count,
//                 per cell: int32 pos, uint32 n_seq_id, n_seq_id * int32 seq ids
//                 uint32 v_trans, uint32 n_layer, uint32 n_rot_k, uint32 n_rot_v
//                 per layer: int32 k_type, uint64 k_size_row, cell_count rows
//                 per layer: int32 v_type, uint64 v_size_row, cell_count rows
//
// No cell carries an extension and every layer has a value cache, so the keys
// start at a fixed offset and cell i lands at k_offset + i * k_size_row. The
// expected total size is recomputed from the parsed values and compared against
// the real one, so a layout change fails loudly instead of silently measuring
// the wrong bytes.
struct layer_keys {
    enum ggml_type      type      = GGML_TYPE_F32;
    uint64_t            row_bytes = 0;
    std::vector<float>  k;   // dequantized keys, cell-major
};

// One llama_kv_cache section. gemma2 alternates SWA and dense layers, so each
// section holds exactly one layer: the dense cache is written first, then the
// SWA cache.
struct kv_section {
    int64_t                  n_cell = 0;
    std::vector<layer_keys>  layers;
};

struct seq_state {
    std::vector<kv_section> sections;
};

// Inverse of the type-trait dequantizer: F32 is a plain copy, F16/Q8_0 go
// through ggml's own conversion so the test measures exactly the bytes the
// kernels would read.
size_t row_bytes_for(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:
        case GGML_TYPE_F16:
        case GGML_TYPE_Q8_0:
            return ggml_row_size(type, N_EMBD_K_GQA);
        default:
            return 0;
    }
}

bool dequantize_row(enum ggml_type type, const uint8_t * src, int64_t n, float * dst) {
    if (type == GGML_TYPE_F32) {
        std::memcpy(dst, src, static_cast<size_t>(n) * sizeof(float));
        return true;
    }
    const ggml_type_traits * traits = ggml_get_type_traits(type);
    if (traits == nullptr || traits->to_float == nullptr) {
        return false;
    }
    traits->to_float(src, dst, n);
    return true;
}

bool read_section(const std::vector<uint8_t> & buf, size_t & off, kv_section & out) {
    auto need = [&](size_t n) { return off + n <= buf.size(); };

    uint32_t n_stream = 0;
    if (!need(4)) {
        return false;
    }
    std::memcpy(&n_stream, buf.data() + off, 4);
    off += 4;

    if (n_stream != 1) {
        std::fprintf(stderr, "unsupported state layout (n_stream=%u)\n", n_stream);
        return false;
    }

    // Stream metadata, then cell metadata.
    uint32_t cell_count = 0;
    if (!need(4)) {
        return false;
    }
    std::memcpy(&cell_count, buf.data() + off, 4);
    off += 4;

    if (cell_count == 0) {
        std::fprintf(stderr, "unexpected empty state section\n");
        return false;
    }

    for (uint32_t i = 0; i < cell_count; ++i) {
        if (!need(8)) {
            return false;
        }
        uint32_t n_seq_id = 0;
        std::memcpy(&n_seq_id, buf.data() + off + 4, 4);
        if (!need(8 + static_cast<size_t>(n_seq_id) * 4)) {
            return false;
        }
        off += 8 + static_cast<size_t>(n_seq_id) * 4;
    }

    if (!need(16)) {
        return false;
    }
    uint32_t n_layer = 0;
    std::memcpy(&n_layer, buf.data() + off + 4, 4); // v_trans, n_layer, n_rot_k, n_rot_v
    off += 16;

    if (n_layer == 0) {
        std::fprintf(stderr, "unexpected state shape (n_layer=%u)\n", n_layer);
        return false;
    }

    out.n_cell = cell_count;
    out.layers.resize(n_layer);

    // Keys: (type, row_size) followed by cell_count rows, per layer.
    for (uint32_t il = 0; il < n_layer; ++il) {
        if (!need(12)) {
            return false;
        }
        int32_t  type_i  = 0;
        uint64_t size_row = 0;
        std::memcpy(&type_i,   buf.data() + off,     4);
        std::memcpy(&size_row, buf.data() + off + 4, 8);
        off += 12;

        layer_keys & lk = out.layers[il];
        lk.type      = static_cast<enum ggml_type>(type_i);
        lk.row_bytes = size_row;

        if (row_bytes_for(lk.type) != size_row || size_row == 0) {
            std::fprintf(stderr, "unexpected k row size (type %d, %llu bytes, expected %zu)\n",
                         static_cast<int>(lk.type),
                         static_cast<unsigned long long>(size_row),
                         row_bytes_for(lk.type));
            return false;
        }

        const size_t bytes = static_cast<size_t>(cell_count) * static_cast<size_t>(size_row);
        if (!need(bytes)) {
            return false;
        }

        lk.k.resize(static_cast<size_t>(cell_count) * static_cast<size_t>(N_EMBD_K_GQA));

        std::vector<uint8_t> row(static_cast<size_t>(size_row));
        for (uint32_t i = 0; i < cell_count; ++i) {
            std::memcpy(row.data(), buf.data() + off + static_cast<size_t>(i) * size_row, row.size());
            if (!dequantize_row(lk.type, row.data(), N_EMBD_K_GQA,
                                lk.k.data() + static_cast<size_t>(i) * N_EMBD_K_GQA)) {
                std::fprintf(stderr, "cannot dequantize k cache type %d\n", static_cast<int>(lk.type));
                return false;
            }
        }
        off += bytes;
    }

    // Values: same shape, but only their size matters here.
    for (uint32_t il = 0; il < n_layer; ++il) {
        if (!need(12)) {
            return false;
        }
        uint64_t size_row = 0;
        std::memcpy(&size_row, buf.data() + off + 4, 8);
        off += 12;

        const size_t bytes = static_cast<size_t>(cell_count) * static_cast<size_t>(size_row);
        if (!need(bytes)) {
            return false;
        }
        off += bytes;
    }

    return true;
}

bool read_seq_state(llama_context * ctx, seq_state & out) {
    const size_t size = llama_state_seq_get_size(ctx, 0);
    if (size == 0) {
        return false;
    }

    std::vector<uint8_t> buf(size);
    if (llama_state_seq_get_data(ctx, buf.data(), buf.size(), 0) != size) {
        return false;
    }

    size_t off = 8; // io_magic, seq_id

    out.sections.clear();
    while (off < buf.size()) {
        kv_section sec;
        if (!read_section(buf, off, sec)) {
            return false;
        }
        out.sections.push_back(std::move(sec));
    }

    // The sections have to account for the whole buffer; a layout change fails
    // loudly instead of silently measuring the wrong bytes.
    if (off != buf.size() || out.sections.empty()) {
        std::fprintf(stderr, "state size mismatch: parsed %zu, declared %zu\n", off, buf.size());
        return false;
    }
    return true;
}

double cell_norm(const kv_section & sec, size_t il, int64_t cell) {
    const layer_keys & lk = sec.layers.at(il);
    const float * p = lk.k.data() + cell * N_EMBD_K_GQA;
    double acc = 0.0;
    for (int64_t i = 0; i < N_EMBD_K_GQA; ++i) {
        acc += static_cast<double>(p[i]) * static_cast<double>(p[i]);
    }
    return std::sqrt(acc);
}

struct scenario {
    const char    * label;
    bool            yarn;         // force YaRN rope scaling at runtime
    float           attn_factor;  // explicit yarn_attn_factor, < 0 = library default
    enum ggml_type  k_type;       // KV cache key type
    bounds          tol;
};

struct fixture {
    llama_model   * model = nullptr;
    llama_context * ctx   = nullptr;
    std::string     path;

    bool init(bool swa, bool yarn, float attn_factor = -1.0f, enum ggml_type k_type = GGML_TYPE_F32) {
        path = write_test_model(swa);

        llama_model_params mparams = llama_model_default_params();
        mparams.n_gpu_layers = 0;

        model = llama_model_load_from_file(path.c_str(), mparams);
        if (model == nullptr) {
            std::fprintf(stderr, "failed to load synthetic model\n");
            return false;
        }

        llama_context_params cparams = llama_context_default_params();
        cparams.n_ctx           = N_CTX;
        cparams.n_batch         = N_CTX;
        cparams.n_ubatch        = N_CTX;
        cparams.n_threads       = 2;
        cparams.n_threads_batch = 2;
        cparams.type_k          = k_type;
        cparams.type_v          = GGML_TYPE_F32;

        if (swa) {
            // llama_kv_cache_iswa::get_can_shift() requires both caches to have
            // the same number of cells, and the cached SWA window is otherwise
            // padded to a smaller ring. A full-size SWA cache keeps them equal
            // without changing the rope parameters.
            cparams.swa_full = true;
        }

        if (yarn) {
            // What a factor-4 YaRN model resolves to: llama_context() recomputes
            // yarn_attn_factor to exactly 1.0 on the forward path, which is why
            // the forward pass is correct and only the shift path is wrong.
            cparams.rope_freq_scale  = ROPE_FREQ_SCALE;
            cparams.yarn_ext_factor  = 1.0f;
            cparams.yarn_orig_ctx    = 4096;
            cparams.rope_scaling_type = LLAMA_ROPE_SCALING_TYPE_YARN;
        }

        if (attn_factor >= 0.0f) {
            // An amplitude that is not 1 on the forward pass, and no extrapolation
            // mixing. The shift path must still be amplitude-neutral.
            cparams.yarn_attn_factor = attn_factor;
        }

        ctx = llama_init_from_model(model, cparams);
        if (ctx == nullptr) {
            std::fprintf(stderr, "failed to create context\n");
            return false;
        }

        batch = llama_batch_init(N_CTX, 0, 1);
        return true;
    }

    ~fixture() {
        if (batch.token) {
            llama_batch_free(batch);
        }
        if (ctx) {
            llama_free(ctx);
        }
        if (model) {
            llama_model_free(model);
        }
        if (!path.empty()) {
            std::remove(path.c_str());
        }
    }

    llama_batch batch{};

    bool decode(llama_pos pos, int n) {
        batch.n_tokens = n;
        for (int i = 0; i < n; ++i) {
            batch.token[i]     = (i * 7 + 3) % N_VOCAB;
            batch.pos[i]       = pos + i;
            batch.n_seq_id[i]  = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i]    = (i == n - 1) ? 1 : 0;
        }
        return llama_decode(ctx, batch) == 0;
    }

    // One context shift over the cells whose position is >= p0, followed by a
    // decode so the pending K-shift graph is actually executed.
    bool shift_and_apply(llama_pos p0) {
        llama_memory_seq_add(llama_get_memory(ctx), 0, p0, -1, 1);

        const llama_pos next = llama_memory_seq_pos_max(llama_get_memory(ctx), 0) + 1;
        return decode(next, 1);
    }
};

// Scenario A: every cached cell is shifted. Mirrors the common "shift the whole
// window" case.
void test_full_shift(const scenario & sc) {
    std::printf("\n[ scenario ] full-window shift, %s\n", sc.label);

    fixture f;
    if (!f.init(false, sc.yarn, sc.attn_factor, sc.k_type)) {
        check(false, "fixture setup");
        return;
    }

    if (!f.decode(0, N_PROMPT)) {
        check(false, "prompt decode");
        return;
    }

    seq_state base;
    if (!read_seq_state(f.ctx, base) || base.sections.size() != 1 ||
        base.sections[0].n_cell < N_PROMPT || base.sections[0].layers.size() != 1) {
        check(false, "baseline KV read");
        return;
    }

    std::vector<double> norms(N_PROMPT);
    for (int i = 0; i < N_PROMPT; ++i) {
        norms[i] = cell_norm(base.sections[0], 0, i);
    }

    std::printf("  per-shift worst key magnitude ratio (cells 0..%d):", N_PROMPT - 1);
    bool in_bounds = true;
    double worst_grow   = 1.0;
    double worst_shrink = 1.0;

    for (int s = 1; s <= N_SHIFTS; ++s) {
        if (!f.shift_and_apply(0)) {
            check(false, "shift + decode");
            return;
        }

        seq_state now;
        if (!read_seq_state(f.ctx, now) || now.sections.size() != 1 ||
            now.sections[0].n_cell < N_PROMPT || now.sections[0].layers.size() != 1) {
            check(false, "KV read after shift");
            return;
        }

        double worst_here = 1.0;
        for (int i = 0; i < N_PROMPT; ++i) {
            const double ratio = cell_norm(now.sections[0], 0, i) / norms[i];

            if (ratio > worst_grow) {
                worst_grow = ratio;
            }
            if (ratio < worst_shrink) {
                worst_shrink = ratio;
            }
            if (std::fabs(ratio - 1.0) > std::fabs(worst_here - 1.0)) {
                worst_here = ratio;
            }
            if (ratio - 1.0 > sc.tol.grow || 1.0 - ratio > sc.tol.shrink) {
                in_bounds = false;
            }
        }

        std::printf(" %.6f", worst_here);
    }
    std::printf("\n");

    std::printf("  measured band over %d shifts: %.6f .. %.6f (allowed +%.0e / -%.0e)\n",
                N_SHIFTS, worst_shrink, worst_grow, sc.tol.grow, sc.tol.shrink);
    check(in_bounds, "key magnitudes stay 1.0x across repeated shifts");

    if (sc.yarn) {
        // Sanity: the unpatched growth must be big enough for this test to
        // actually discriminate. If it is not, the test is not testing anything.
        const double expected_unpatched = std::pow(EXPECTED_UNPATCHED_SCALE, N_SHIFTS);
        std::printf("  (unpatched code would grow to %.3fx after %d shifts)\n",
                    expected_unpatched, N_SHIFTS);
        check(EXPECTED_UNPATCHED_SCALE > 1.0f && expected_unpatched > 1.5,
              "amplitude scale is large enough for the test to discriminate");
    }
}

// Scenario B: the "--keep" prefix. llama_kv_cache::set_input_k_shift() fills a
// shift value for every cell and build_graph_shift() ropes the whole buffer, so
// cells outside the shifted range take delta 0 and are *still* multiplied by the
// amplitude. That is why a reused keep prefix degrades as well.
void test_keep_prefix() {
    std::printf("\n[ scenario ] keep-prefix (%d cells) left in place, YaRN factor 4.0, F32 cache\n", N_KEEP);

    fixture f;
    if (!f.init(false, true)) {
        check(false, "fixture setup");
        return;
    }

    if (!f.decode(0, N_PROMPT)) {
        check(false, "prompt decode");
        return;
    }

    seq_state base;
    if (!read_seq_state(f.ctx, base) || base.sections.size() != 1 ||
        base.sections[0].n_cell < N_PROMPT || base.sections[0].layers.size() != 1) {
        check(false, "baseline KV read");
        return;
    }

    std::vector<double> head(N_KEEP), tail(N_PROMPT - N_KEEP);
    for (int i = 0; i < N_KEEP; ++i) {
        head[i] = cell_norm(base.sections[0], 0, i);
    }
    for (int i = N_KEEP; i < N_PROMPT; ++i) {
        tail[i - N_KEEP] = cell_norm(base.sections[0], 0, i);
    }

    bool head_ok = true;
    bool tail_ok = true;
    double head_worst = 0.0;

    for (int s = 1; s <= N_SHIFTS; ++s) {
        if (!f.shift_and_apply(N_KEEP)) {
            check(false, "shift + decode");
            return;
        }

        seq_state now;
        if (!read_seq_state(f.ctx, now) || now.sections.size() != 1 ||
            now.sections[0].n_cell < N_PROMPT || now.sections[0].layers.size() != 1) {
            check(false, "KV read after shift");
            return;
        }

        for (int i = 0; i < N_KEEP; ++i) {
            const double r = cell_norm(now.sections[0], 0, i) / head[i];
            const double err = std::fabs(r - 1.0);
            if (err > head_worst) {
                head_worst = err;
            }
            if (err > BOUNDS_F32.grow) {
                head_ok = false;
            }
        }
        for (int i = N_KEEP; i < N_PROMPT; ++i) {
            const double r = cell_norm(now.sections[0], 0, i) / tail[i - N_KEEP];
            if (std::fabs(r - 1.0) > BOUNDS_F32.grow) {
                tail_ok = false;
            }
        }
    }

    std::printf("  max relative deviation of the untouched prefix over %d shifts: %.3e\n",
                N_SHIFTS, head_worst);
    check(head_ok, "keep-prefix keys (shift 0) retain their magnitude");
    check(tail_ok, "shifted keys retain their magnitude");
}

// Scenario C: a model that mixes sliding-window and dense layers. gemma2
// alternates them (layer 0 = SWA, layer 1 = dense) and gives the SWA layers
// their own train-time rope parameters. llama_model::get_rope_freq_scale()
// returns `rope_freq_scale_train_swa` (1.0 here) for those layers instead of
// `cparams.rope_freq_scale` (0.25 under YaRN), so the correction has to cancel
// with the per-layer freq_scale, not the context-wide one.
//
// The unpatched code is already amplitude-neutral for the SWA half (the ggml
// factor is 1 + 0.1*log(1/1) = 1), so this scenario does not reproduce a bug on
// its own; the dense half of the same fixture does. Its job is to pin the fix
// down: a future "simplification" that used cparams.rope_freq_scale for every
// layer would leave the dense layer at 1.0 but shrink the SWA layer by ~1.14x
// per shift. llama_kv_cache_iswa::state_write() emits the dense cache first and
// the SWA cache second, so section 0 is dense and section 1 is SWA.
void test_swa_layers() {
    std::printf("\n[ scenario ] sliding-window (layer 0) + dense (layer 1), YaRN factor 4.0, F32 cache\n");

    fixture f;
    if (!f.init(true, true)) {
        check(false, "fixture setup");
        return;
    }

    if (!f.decode(0, N_PROMPT)) {
        check(false, "prompt decode");
        return;
    }

    seq_state base;
    if (!read_seq_state(f.ctx, base) || base.sections.size() != 2) {
        check(false, "baseline KV read (expected [dense][swa])");
        return;
    }
    for (const auto & sec : base.sections) {
        if (sec.n_cell < N_PROMPT || sec.layers.size() != 1) {
            check(false, "unexpected state section shape");
            return;
        }
    }

    // The sections are per-cache, not per-model-layer, so both have one layer.
    const kv_section & dense0 = base.sections[0];
    const kv_section & swa0   = base.sections[1];

    std::vector<double> dense_norm(N_PROMPT), swa_norm(N_PROMPT);
    for (int i = 0; i < N_PROMPT; ++i) {
        dense_norm[i] = cell_norm(dense0, 0, i);
        swa_norm[i]   = cell_norm(swa0,   0, i);
    }

    bool dense_ok = true;
    bool swa_ok   = true;
    double dense_grow = 1.0, dense_shrink = 1.0;
    double swa_grow   = 1.0, swa_shrink   = 1.0;

    std::printf("  per-shift worst key magnitude ratio (dense / SWA):");
    for (int s = 1; s <= N_SHIFTS; ++s) {
        if (!f.shift_and_apply(0)) {
            check(false, "shift + decode");
            return;
        }

        seq_state now;
        if (!read_seq_state(f.ctx, now) || now.sections.size() != 2 ||
            now.sections[0].layers.size() != 1 || now.sections[1].layers.size() != 1) {
            check(false, "KV read after shift");
            return;
        }

        double dense_here = 1.0;
        double swa_here   = 1.0;

        for (int i = 0; i < N_PROMPT; ++i) {
            const double dr = cell_norm(now.sections[0], 0, i) / dense_norm[i];
            const double sr = cell_norm(now.sections[1], 0, i) / swa_norm[i];

            dense_grow   = std::max(dense_grow, dr);
            dense_shrink = std::min(dense_shrink, dr);
            swa_grow     = std::max(swa_grow, sr);
            swa_shrink   = std::min(swa_shrink, sr);

            if (std::fabs(dr - 1.0) > std::fabs(dense_here - 1.0)) {
                dense_here = dr;
            }
            if (std::fabs(sr - 1.0) > std::fabs(swa_here - 1.0)) {
                swa_here = sr;
            }

            if (dr - 1.0 > BOUNDS_F32.grow || 1.0 - dr > BOUNDS_F32.shrink) {
                dense_ok = false;
            }
            if (sr - 1.0 > BOUNDS_F32.grow || 1.0 - sr > BOUNDS_F32.shrink) {
                swa_ok = false;
            }
        }

        std::printf(" %.6f/%.6f", dense_here, swa_here);
    }
    std::printf("\n");

    std::printf("  dense layer band over %d shifts: %.6f .. %.6f (allowed +%.0e / -%.0e)\n",
                N_SHIFTS, dense_shrink, dense_grow, BOUNDS_F32.grow, BOUNDS_F32.shrink);
    std::printf("  SWA layer band over %d shifts:   %.6f .. %.6f (allowed +%.0e / -%.0e)\n",
                N_SHIFTS, swa_shrink, swa_grow, BOUNDS_F32.grow, BOUNDS_F32.shrink);

    // The dense layer is the actual regression: without the patch it grows by
    // EXPECTED_UNPATCHED_SCALE per shift.
    check(dense_ok, "dense layer key magnitudes stay 1.0x across repeated shifts");

    // The SWA layer is the guard for the per-layer part of the fix. It passes
    // both before and after this patch, and fails if the correction is ever
    // derived from cparams.rope_freq_scale instead of the layer's own scale.
    check(swa_ok, "SWA layer key magnitudes stay 1.0x across repeated shifts");
}

} // namespace

int main() {
    // With GGML_BACKEND_DL the CPU backend is a loadable module, so the registry
    // has to be populated before a model can be loaded. The test executable is
    // placed next to the modules by CMake.
    ggml_backend_load_all();

    llama_backend_init();

    {
        const ggml_init_params params = {
            /*.mem_size   =*/ 64ull * 1024ull * 1024ull,
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ false,
        };
        g_tensor_ctx = ggml_init(params);
        if (g_tensor_ctx == nullptr) {
            std::fprintf(stderr, "failed to create ggml context\n");
            return 2;
        }
    }

    test_full_shift({ "baseline, no rope scaling, F32 cache",
                      false, -1.0f, GGML_TYPE_F32,  BOUNDS_F32 });
    test_full_shift({ "YaRN factor 4.0, F32 cache",
                      true,  -1.0f, GGML_TYPE_F32,  BOUNDS_F32 });
    test_full_shift({ "explicit amplitude 1.3, no extrapolation, F32 cache",
                      false,  1.3f, GGML_TYPE_F32,  BOUNDS_F32 });
    test_full_shift({ "YaRN factor 4.0, F16 cache (conversion path)",
                      true,  -1.0f, GGML_TYPE_F16,  BOUNDS_F16 });
    test_full_shift({ "YaRN factor 4.0, Q8_0 cache (conversion path)",
                      true,  -1.0f, GGML_TYPE_Q8_0, BOUNDS_Q8_0 });
    test_keep_prefix();
    test_swa_layers();

    ggml_free(g_tensor_ctx);
    llama_backend_free();

    std::printf("\n%s\n", g_failures == 0 ? "PASS" : "FAIL");
    return g_failures == 0 ? 0 : 1;
}
