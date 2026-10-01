#include "pointer_mask.h"
#include "pointer_mask_logic.h"

#include "llama-impl.h"
#include "llama-batch.h"
#include "ggml.h"

#include <mutex>
#include <unordered_map>
#include <vector>

namespace {

struct pointer_state {
    std::vector<int32_t> seg;
    bool state_bidir = true;
};

std::mutex g_mu;
std::unordered_map<llama_context *, pointer_state> g_states;
thread_local llama_context * g_bound = nullptr;
thread_local int32_t g_status = 1;

void write_mask(float * data, int64_t n_kv, int32_t n, const int32_t * seg, bool state_bidir) {
    for (int32_t i = 0; i < n; ++i) {
        float * row = data + static_cast<int64_t>(i) * n_kv;
        for (int32_t j = 0; j < n; ++j) {
            row[j] = llama_pointer_allow(seg, n, i, j, state_bidir) ? 0.0f : -INFINITY;
        }
        for (int64_t j = n; j < n_kv; ++j) {
            row[j] = -INFINITY;
        }
    }
}

void write_mask(ggml_fp16_t * data, int64_t n_kv, int32_t n, const int32_t * seg, bool state_bidir) {
    for (int32_t i = 0; i < n; ++i) {
        ggml_fp16_t * row = data + static_cast<int64_t>(i) * n_kv;
        for (int32_t j = 0; j < n; ++j) {
            const float v = llama_pointer_allow(seg, n, i, j, state_bidir) ? 0.0f : -INFINITY;
            row[j] = ggml_fp32_to_fp16(v);
        }
        const ggml_fp16_t drop = ggml_fp32_to_fp16(-INFINITY);
        for (int64_t j = n; j < n_kv; ++j) {
            row[j] = drop;
        }
    }
}

} // namespace

void llama_pointer_bind(llama_context * ctx) {
    g_bound = ctx;
    g_status = 1;
}

void llama_pointer_unbind() {
    g_bound = nullptr;
}

extern "C" void llama_set_pointer_segments(llama_context * ctx, const int32_t * seg, int32_t n, bool state_bidir) {
    std::lock_guard<std::mutex> lock(g_mu);
    if (ctx == nullptr || n <= 0 || seg == nullptr) {
        g_states.erase(ctx);
        return;
    }
    pointer_state state;
    state.seg.assign(seg, seg + n);
    state.state_bidir = state_bidir;
    g_states[ctx] = std::move(state);
}

extern "C" int32_t llama_pointer_mask_status(void) {
    return g_status;
}

void llama_pointer_apply_kq_mask(ggml_tensor * dst, const llama_ubatch * ubatch) {
    g_status = 1;
    if (g_bound == nullptr || dst == nullptr || ubatch == nullptr || dst->data == nullptr) {
        return;
    }
    pointer_state state;
    {
        std::lock_guard<std::mutex> lock(g_mu);
        const auto it = g_states.find(g_bound);
        if (it == g_states.end()) {
            return;
        }
        state = it->second;
    }
    const int32_t n = static_cast<int32_t>(state.seg.size());
    const int64_t n_kv = dst->ne[0];
    const int64_t n_stream = dst->ne[3];
    // One full record, one stream, cache cells 0..n-1 holding those tokens.
    // A split ubatch cannot build this mask: state attention is bidirectional.
    if (n_stream != 1 || ubatch->n_tokens != n || n_kv < n) {
        return;
    }
    if (dst->type == GGML_TYPE_F32) {
        write_mask(static_cast<float *>(dst->data), n_kv, n, state.seg.data(), state.state_bidir);
    } else if (dst->type == GGML_TYPE_F16) {
        write_mask(static_cast<ggml_fp16_t *>(dst->data), n_kv, n, state.seg.data(), state.state_bidir);
    } else {
        return;
    }
    g_status = 0;
}
