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
// The model is written from scratch with the gguf API so the test needs no
// network access and no multi-gigabyte download.

#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"
#include "llama.h"

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
constexpr int N_LAYER    = 1;
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

std::string write_test_model() {
    const char * path = "kv-shift-rope-test-model.gguf";

    gguf_context * fctx = gguf_init_empty();

    gguf_set_val_str(fctx, "general.architecture", "llama");
    gguf_set_val_str(fctx, "general.name", "kv-shift-rope-test");

    gguf_set_val_u32(fctx, "llama.block_count", N_LAYER);
    gguf_set_val_u32(fctx, "llama.context_length", N_CTX);
    gguf_set_val_u32(fctx, "llama.embedding_length", N_EMBD);
    gguf_set_val_u32(fctx, "llama.feed_forward_length", N_FF);
    gguf_set_val_u32(fctx, "llama.attention.head_count", N_HEAD);
    gguf_set_val_u32(fctx, "llama.attention.head_count_kv", N_HEAD_KV);
    gguf_set_val_f32(fctx, "llama.attention.layer_norm_rms_epsilon", 1e-5f);
    gguf_set_val_u32(fctx, "llama.rope.dimension_count", N_ROT);
    gguf_set_val_f32(fctx, "llama.rope.freq_base", 10000.0f);
    gguf_set_val_u32(fctx, "llama.vocab_size", N_VOCAB);

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
    add_tensor(fctx, "output.weight", N_EMBD, N_VOCAB);

    add_tensor(fctx, "blk.0.attn_norm.weight", N_EMBD);
    add_tensor(fctx, "blk.0.attn_q.weight", N_EMBD, N_EMBD_K * N_HEAD);
    add_tensor(fctx, "blk.0.attn_k.weight", N_EMBD, N_EMBD_K_GQA);
    add_tensor(fctx, "blk.0.attn_v.weight", N_EMBD, N_EMBD_K_GQA);
    add_tensor(fctx, "blk.0.attn_output.weight", N_EMBD_K * N_HEAD, N_EMBD);
    add_tensor(fctx, "blk.0.ffn_norm.weight", N_EMBD);
    add_tensor(fctx, "blk.0.ffn_gate.weight", N_EMBD, N_FF);
    add_tensor(fctx, "blk.0.ffn_down.weight", N_FF, N_EMBD);
    add_tensor(fctx, "blk.0.ffn_up.weight", N_EMBD, N_FF);

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
//   uint32 io_magic, int32 seq_id, uint32 n_stream, uint32 cell_count
//   per cell: int32 pos, uint32 n_seq_id, then n_seq_id * int32 seq ids
//   uint32 v_trans, uint32 n_layer, uint32 n_rot_k, uint32 n_rot_v
//   per layer: int32 k_type, uint64 k_size_row, then the key rows
//   (values follow with the same shape)
//
// n_layer is 1 for this model and no cell carries an extension, so the keys
// start at a fixed offset and cell i lands at k_offset + i * k_size_row. The
// expected total size is recomputed from the parsed values and compared against
// the real one, so a layout change fails loudly instead of silently measuring
// the wrong bytes.
struct seq_state {
    std::vector<float>  k;   // dequantized keys, cell-major
    int64_t             n_cell     = 0;
    int64_t             cell_elems = 0;   // floats per cell row
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

bool read_seq_state(llama_context * ctx, seq_state & out) {
    const size_t size = llama_state_seq_get_size(ctx, 0);
    if (size == 0) {
        return false;
    }

    std::vector<uint8_t> buf(size);
    if (llama_state_seq_get_data(ctx, buf.data(), buf.size(), 0) != size) {
        return false;
    }

    size_t   off = 0;
    uint32_t n_stream = 0;
    uint32_t cell_count = 0;
    auto need = [&](size_t n) { return off + n <= buf.size(); };

    if (!need(16)) {
        return false;
    }
    off += 4 + 4; // io_magic, seq_id
    std::memcpy(&n_stream,   buf.data() + off, 4); off += 4;
    std::memcpy(&cell_count, buf.data() + off, 4); off += 4;

    if (n_stream != 1 || cell_count == 0) {
        std::fprintf(stderr, "unsupported state layout (n_stream=%u, cells=%u)\n", n_stream, cell_count);
        return false;
    }

    for (uint32_t i = 0; i < cell_count; ++i) {
        if (!need(8)) {
            return false;
        }
        uint32_t n_seq_id = 0;
        std::memcpy(&n_seq_id, buf.data() + off + 4, 4);
        off += 8 + static_cast<size_t>(n_seq_id) * 4;
    }

    if (!need(16)) {
        return false;
    }
    uint32_t n_layer = 0;
    std::memcpy(&n_layer, buf.data() + off + 4, 4);
    off += 16;

    if (n_layer != 1) {
        std::fprintf(stderr, "unexpected state shape (n_layer=%u)\n", n_layer);
        return false;
    }

    // Keys, then values, each as (type, row_size) followed by cell_count rows.
    auto read_section = [&](uint64_t & row_bytes, enum ggml_type & type) {
        if (!need(12)) {
            return false;
        }
        int32_t  type_i = 0;
        uint64_t size_row = 0;
        std::memcpy(&type_i,   buf.data() + off,     4);
        std::memcpy(&size_row, buf.data() + off + 4, 8);
        type     = static_cast<enum ggml_type>(type_i);
        row_bytes = size_row;
        off += 12;
        return true;
    };

    uint64_t k_row_bytes = 0;
    enum ggml_type k_type = GGML_TYPE_F32;
    if (!read_section(k_row_bytes, k_type)) {
        return false;
    }
    if (row_bytes_for(k_type) != k_row_bytes || k_row_bytes == 0) {
        std::fprintf(stderr, "unexpected k row size (type %d, %llu bytes, expected %zu)\n",
                     static_cast<int>(k_type),
                     static_cast<unsigned long long>(k_row_bytes),
                     row_bytes_for(k_type));
        return false;
    }

    const size_t k_bytes = static_cast<size_t>(cell_count) * static_cast<size_t>(k_row_bytes);
    if (!need(k_bytes)) {
        return false;
    }
    const size_t k_off = off;
    off += k_bytes;

    uint64_t v_row_bytes = 0;
    enum ggml_type v_type = GGML_TYPE_F32;
    if (!read_section(v_row_bytes, v_type)) {
        return false;
    }
    off += static_cast<size_t>(cell_count) * static_cast<size_t>(v_row_bytes);

    // The sections have to account for the whole buffer; a layout change fails
    // loudly instead of silently measuring the wrong bytes.
    if (off != buf.size()) {
        std::fprintf(stderr, "state size mismatch: parsed %zu, declared %zu\n", off, buf.size());
        return false;
    }

    out.n_cell     = cell_count;
    out.cell_elems = N_EMBD_K_GQA;
    out.k.resize(static_cast<size_t>(cell_count) * static_cast<size_t>(N_EMBD_K_GQA));

    std::vector<uint8_t> row(static_cast<size_t>(k_row_bytes));
    for (uint32_t i = 0; i < cell_count; ++i) {
        std::memcpy(row.data(), buf.data() + k_off + static_cast<size_t>(i) * k_row_bytes, row.size());
        if (!dequantize_row(k_type, row.data(), N_EMBD_K_GQA,
                            out.k.data() + static_cast<size_t>(i) * N_EMBD_K_GQA)) {
            std::fprintf(stderr, "cannot dequantize k cache type %d\n", static_cast<int>(k_type));
            return false;
        }
    }
    return true;
}

double cell_norm(const seq_state & st, int64_t cell) {
    const float * p = st.k.data() + cell * st.cell_elems;
    double acc = 0.0;
    for (int64_t i = 0; i < st.cell_elems; ++i) {
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

    bool init(bool yarn, float attn_factor = -1.0f, enum ggml_type k_type = GGML_TYPE_F32) {
        path = write_test_model();

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
    if (!f.init(sc.yarn, sc.attn_factor, sc.k_type)) {
        check(false, "fixture setup");
        return;
    }

    if (!f.decode(0, N_PROMPT)) {
        check(false, "prompt decode");
        return;
    }

    seq_state base;
    if (!read_seq_state(f.ctx, base) || base.n_cell < N_PROMPT) {
        check(false, "baseline KV read");
        return;
    }

    std::vector<double> norms(N_PROMPT);
    for (int i = 0; i < N_PROMPT; ++i) {
        norms[i] = cell_norm(base, i);
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
        if (!read_seq_state(f.ctx, now) || now.n_cell < N_PROMPT) {
            check(false, "KV read after shift");
            return;
        }

        double worst_here = 1.0;
        for (int i = 0; i < N_PROMPT; ++i) {
            const double ratio = cell_norm(now, i) / norms[i];

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
    if (!f.init(true)) {
        check(false, "fixture setup");
        return;
    }

    if (!f.decode(0, N_PROMPT)) {
        check(false, "prompt decode");
        return;
    }

    seq_state base;
    if (!read_seq_state(f.ctx, base) || base.n_cell < N_PROMPT) {
        check(false, "baseline KV read");
        return;
    }

    std::vector<double> head(N_KEEP), tail(N_PROMPT - N_KEEP);
    for (int i = 0; i < N_KEEP; ++i) {
        head[i] = cell_norm(base, i);
    }
    for (int i = N_KEEP; i < N_PROMPT; ++i) {
        tail[i - N_KEEP] = cell_norm(base, i);
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
        if (!read_seq_state(f.ctx, now) || now.n_cell < N_PROMPT) {
            check(false, "KV read after shift");
            return;
        }

        for (int i = 0; i < N_KEEP; ++i) {
            const double r = cell_norm(now, i) / head[i];
            const double err = std::fabs(r - 1.0);
            if (err > head_worst) {
                head_worst = err;
            }
            if (err > BOUNDS_F32.grow) {
                head_ok = false;
            }
        }
        for (int i = N_KEEP; i < N_PROMPT; ++i) {
            const double r = cell_norm(now, i) / tail[i - N_KEEP];
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

    ggml_free(g_tensor_ctx);
    llama_backend_free();

    std::printf("\n%s\n", g_failures == 0 ? "PASS" : "FAIL");
    return g_failures == 0 ? 0 : 1;
}
