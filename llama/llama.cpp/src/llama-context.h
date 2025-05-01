#pragma once

#include "llama.h"
#include "llama-batch.h"
#include "llama-cparams.h"
#include "llama-graph.h"
#include "llama-adapter.h"
#include "llama-kv-cache.h"

#include "ggml-cpp.h"

#include <map>
#include <vector>

struct llama_model;
struct llama_kv_cache;

class llama_io_read_i;
class llama_io_write_i;

struct llama_context {
    // init scheduler and compute buffers, reserve worst-case graphs
    llama_context(
            const llama_model & model,
                  llama_context_params params);

    ~llama_context();

    void synchronize();

    const llama_model & get_model() const;

    uint32_t n_ctx()         const;
    uint32_t n_ctx_per_seq() const;
    uint32_t n_batch()       const;
    uint32_t n_ubatch()      const;
    uint32_t n_seq_max()     const;

    uint32_t n_threads()       const;
    uint32_t n_threads_batch() const;

          llama_kv_cache * get_kv_self();
    const llama_kv_cache * get_kv_self() const;

    void kv_self_update();

    enum llama_pooling_type pooling_type() const;

    float * get_logits();
    float * get_logits_ith(int32_t i);

    float * get_embeddings();
    float * get_embeddings_ith(int32_t i);
    float * get_embeddings_seq(llama_seq_id seq_id);

    void attach_threadpool(
            ggml_threadpool_t threadpool,
            ggml_threadpool_t threadpool_batch);

    void detach_threadpool();

    void set_n_threads(int32_t n_threads, int32_t n_threads_batch);

    void set_abort_callback(bool (*abort_callback)(void * data), void * abort_callback_data);

    void set_embeddings (bool value);
    void set_causal_attn(bool value);
    void set_warmup(bool value);
    void set_cross_attn(bool value);

    void set_adapter_lora(
            llama_adapter_lora * adapter,
            float scale);

    bool rm_adapter_lora(
            llama_adapter_lora * adapter);

    void clear_adapter_lora();

    bool apply_adapter_cvec(
            const float * data,
                 size_t   len,
                int32_t   n_embd,
                int32_t   il_start,
                int32_t   il_end);

    int encode(llama_batch & inp_batch);
    int decode(llama_batch & inp_batch);

    //
    // state save/load
    //

    size_t state_get_size();
    size_t state_get_data(      uint8_t * dst, size_t size);
    size_t state_set_data(const uint8_t * src, size_t size);

    size_t state_seq_get_size(llama_seq_id seq_id);
    size_t state_seq_get_data(llama_seq_id seq_id,       uint8_t * dst, size_t size);
    size_t state_seq_set_data(llama_seq_id seq_id, const uint8_t * src, size_t size);

    bool state_load_file(
            const char * filepath,
           llama_token * tokens_out,
                size_t   n_token_capacity,
                size_t * n_token_count_out);

    bool state_save_file(
            const char * filepath,
     const llama_token * tokens,
                size_t   n_token_count);

    size_t state_seq_load_file(
          llama_seq_id   seq_id,
            const char * filepath,
           llama_token * tokens_out,
                size_t   n_token_capacity,
                size_t * n_token_count_out);

    size_t state_seq_save_file(
          llama_seq_id   seq_id,
            const char * filepath,
     const llama_token * tokens,
                size_t   n_token_count);

    //
    // perf
    //

    llama_perf_context_data perf_get_data() const;
    void perf_reset();

private:
    //
    // output
    //

    // Make sure enough space is available for outputs.
    // Returns max number of outputs for which space was reserved.
    int32_t output_reserve(int32_t n_outputs);

    // make the outputs have the same order they had in the user-provided batch
    // TODO: maybe remove this
    void output_reorder();

    //
    // graph
    //

    int32_t graph_max_nodes() const;

    // zero-out inputs and create the ctx_compute for the compute graph
    ggml_cgraph * graph_init();

    llm_graph_result_ptr graph_build(
            ggml_context * ctx,
             ggml_cgraph * gf,
      const llama_ubatch & ubatch,
          llm_graph_type   gtype);

    // returns the result of ggml_backend_sched_graph_compute_async execution
    ggml_status graph_compute(
            ggml_cgraph * gf,
                   bool   batched);

    llm_graph_cb graph_get_cb() const;

    // used by kv_self_update()
    ggml_tensor * build_rope_shift(
        ggml_context * ctx0,
        ggml_tensor * cur,
        ggml_tensor * shift,
        ggml_tensor * factors,
              float   freq_base,
              float   freq_scale,
        ggml_backend_buffer * bbuf) const;

    llm_graph_result_ptr build_kv_self_shift(
            ggml_context * ctx0,
            ggml_cgraph * gf) const;

    llm_graph_result_ptr build_kv_self_defrag(
            ggml_context * ctx0,
            ggml_cgraph * gf,
            const std::vector<struct llama_kv_defrag_move> & moves) const;

    // TODO: read/write lora adapters and cvec
    size_t state_write_data(llama_io_write_i & io);
    size_t state_read_data (llama_io_read_i  & io);

    size_t state_seq_write_data(llama_io_write_i & io, llama_seq_id seq_id);
    size_t state_seq_read_data (llama_io_read_i  & io, llama_seq_id seq_id);

    //
    // members
    //

    const llama_model & model;

    llama_cparams       cparams;
    llama_adapter_cvec  cvec;
    llama_adapter_loras loras;
    llama_sbatch        sbatch;

    llama_cross cross; // TODO: tmp for handling cross-attention - need something better probably

    std::unique_ptr<llama_kv_cache_unified> kv_self;

    // TODO: remove
    bool logits_all = false;

    // decode output (2-dimensional array: [n_outputs][n_vocab])
    size_t  logits_size = 0; // capacity (of floats) for logits
    float * logits      = nullptr;

    // embeddings output (2-dimensional array: [n_outputs][n_embd])
    // populated only when pooling_type == LLAMA_POOLING_TYPE_NONE
    size_t  embd_size = 0; // capacity (of floats) for embeddings
    float * embd      = nullptr;

    // sequence embeddings output (map of [n_embd] vectors)
    // populated only when pooling_type != LLAMA_POOLING_TYPE_NONE
    std::map<llama_seq_id, std::vector<float>> embd_seq;

    int32_t n_outputs     = 0; // number of actually-used outputs in the current ubatch or last logical batch
    int32_t n_outputs_max = 0; // capacity (of tokens positions) for the output buffers

    std::vector<int32_t> output_ids; // map batch token positions to ids of the logits and embd buffers

    ggml_backend_sched_ptr sched;

    ggml_backend_t backend_cpu = nullptr;
    std::vector<ggml_backend_ptr> backends;

    ggml_context_ptr ctx_compute;

    ggml_threadpool_t threadpool       = nullptr;
    ggml_threadpool_t threadpool_batch = nullptr;

    ggml_abort_callback abort_callback      = nullptr;
    void *              abort_callback_data = nullptr;

    std::vector<std::pair<ggml_backend_t, ggml_backend_set_n_threads_t>> set_n_threads_fns;

    // buffer types used for the compute buffer of each backend
    std::vector<ggml_backend_t>             backend_ptrs;
    std::vector<ggml_backend_buffer_type_t> backend_buft;

    // memory buffers used to evaluate the model
    std::vector<uint8_t> buf_compute_meta;

    // host buffer for the model output (logits and embeddings)
    ggml_backend_buffer_ptr buf_output;

    bool has_evaluated_once = false;

    // perf
    mutable int64_t t_start_us  = 0;
    mutable int64_t t_load_us   = 0;
    mutable int64_t t_p_eval_us = 0;
    mutable int64_t t_eval_us   = 0;

    mutable int64_t t_compute_start_us = 0;
    mutable int64_t n_queued_tokens    = 0;

    mutable int32_t n_p_eval = 0; // number of tokens in eval calls for the prompt (with batch size > 1)
    mutable int32_t n_eval   = 0; // number of eval calls
};
