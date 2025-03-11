#pragma once

#include "llama.h"
#include "llama-batch.h"
#include "llama-cparams.h"
#include "llama-model.h"
#include "llama-kv-cache.h"
#include "llama-adapter.h"

#include "ggml-cpp.h"

#include <map>
#include <unordered_map>
#include <vector>
#include <set>

struct llama_context {
    llama_context(const llama_model & model)
        : model(model)
        , t_start_us(model.t_start_us)
        , t_load_us(model.t_load_us) {}

    const struct llama_model & model;

    struct llama_cparams      cparams;
    struct llama_sbatch       sbatch;  // TODO: revisit if needed
    struct llama_kv_cache     kv_self;
    struct llama_adapter_cvec cvec;

    std::unordered_map<struct llama_adapter_lora *, float> lora;

    std::vector<ggml_backend_ptr> backends;
    std::vector<std::pair<ggml_backend_t, ggml_backend_set_n_threads_t>> set_n_threads_fns;

    ggml_backend_t backend_cpu = nullptr;

    ggml_threadpool_t threadpool       = nullptr;
    ggml_threadpool_t threadpool_batch = nullptr;

    bool has_evaluated_once = false;

    mutable int64_t t_start_us;
    mutable int64_t t_load_us;
    mutable int64_t t_p_eval_us = 0;
    mutable int64_t t_eval_us   = 0;

    mutable int64_t t_compute_start_us = 0;
    mutable int64_t n_queued_tokens = 0;

    mutable int32_t n_p_eval = 0; // number of tokens in eval calls for the prompt (with batch size > 1)
    mutable int32_t n_eval   = 0; // number of eval calls

    // host buffer for the model output (logits and embeddings)
    ggml_backend_buffer_ptr buf_output;

    // decode output (2-dimensional array: [n_outputs][n_vocab])
    size_t  logits_size = 0; // capacity (of floats) for logits
    float * logits      = nullptr;

    std::vector<int32_t> output_ids; // map batch token positions to ids of the logits and embd buffers
    size_t  output_size = 0; // capacity (of tokens positions) for the output buffers
    int32_t n_outputs   = 0; // number of actually-used outputs in the current ubatch or last logical batch

    bool logits_all = false;

    // embeddings output (2-dimensional array: [n_outputs][n_embd])
    // populated only when pooling_type == LLAMA_POOLING_TYPE_NONE
    size_t  embd_size = 0; // capacity (of floats) for embeddings
    float * embd      = nullptr;

    // sequence embeddings output (map of [n_embd] vectors)
    // populated only when pooling_type != LLAMA_POOLING_TYPE_NONE
    std::map<llama_seq_id, std::vector<float>> embd_seq;

    // whether we are computing encoder output or decoder output
    bool is_encoding = false;

    // TODO: find a better way to accommodate mutli-dimension position encoding methods
    // number of position id each token get, 1 for each token in most cases.
    // when using m-rope, it will be 3 position ids per token to representing 3 dimension coordinate.
    int n_pos_per_token = 1;

    // output of the encoder part of the encoder-decoder models
    std::vector<float> embd_enc;
    std::vector<std::set<llama_seq_id>> seq_ids_enc;

    // memory buffers used to evaluate the model
    std::vector<uint8_t> buf_compute_meta;
    ggml_backend_sched_ptr sched;

    ggml_abort_callback abort_callback      = nullptr;
    void *              abort_callback_data = nullptr;

    // input tensors
    struct ggml_tensor * inp_tokens;        // I32 [n_batch]
    struct ggml_tensor * inp_embd;          // F32 [n_embd, n_batch]
    struct ggml_tensor * inp_pos;           // I32 [n_batch]
    struct ggml_tensor * inp_out_ids;       // I32 [n_outputs]
    struct ggml_tensor * inp_KQ_mask;       // F32 [kv_size, n_batch]
    struct ggml_tensor * inp_KQ_mask_swa;   // F32 [kv_size, n_batch]
    struct ggml_tensor * inp_K_shift;       // I32 [kv_size]
    struct ggml_tensor * inp_mean;          // F32 [n_batch, n_batch]
    struct ggml_tensor * inp_cls;           // I32 [n_batch]
    struct ggml_tensor * inp_s_copy;        // I32 [kv_size]
    struct ggml_tensor * inp_s_mask;        // F32 [1, n_kv]
    struct ggml_tensor * inp_s_seq;         // I32 [n_kv, n_batch]
    struct ggml_tensor * inp_pos_bucket;    // I32 [n_batch|n_kv, n_batch]
    struct ggml_tensor * inp_embd_enc;      // F32 [n_embd, n_outputs_enc]
    struct ggml_tensor * inp_KQ_mask_cross; // F32 [n_outputs_enc, n_batch]

    struct ggml_tensor * inp_cross_attn_state; // F32 [4, n_embd, 1061]
};

// TODO: make these methods of llama_context
void llama_set_k_shift(struct llama_context & lctx);

void llama_set_s_copy(struct llama_context & lctx);

void llama_set_inputs(llama_context & lctx, const llama_ubatch & ubatch);

// Make sure enough space is available for outputs.
// Returns max number of outputs for which space was reserved.
size_t llama_output_reserve(struct llama_context & lctx, size_t n_outputs);

// make the outputs have the same order they had in the user-provided batch
void llama_output_reorder(struct llama_context & ctx);

// For internal test use
// TODO: remove
const std::vector<std::pair<std::string, struct ggml_tensor *>> & llama_internal_get_tensor_map(struct llama_context * ctx);
