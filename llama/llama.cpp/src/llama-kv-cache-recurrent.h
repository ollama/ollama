#pragma once

#include "llama-batch.h"
#include "llama-graph.h"
#include "llama-memory.h"

#include <set>
#include <vector>

//
// llama_kv_cache_recurrent
//

// TODO: extract the KV cache state used for graph computation into llama_kv_cache_recurrent_state_i
//       see the implementation of llama_kv_cache_unified_state_i for an example how to do it
class llama_kv_cache_recurrent : public llama_memory_i {
public:
    llama_kv_cache_recurrent(
            const llama_model & model,
                    ggml_type   type_k,
                    ggml_type   type_v,
                         bool   offload,
                     uint32_t   kv_size,
                     uint32_t   n_seq_max);

    ~llama_kv_cache_recurrent() = default;

    //
    // llama_memory_i
    //

    llama_memory_state_ptr init_batch(
            const llama_batch & batch,
            uint32_t n_ubatch,
            bool embd_pooled,
            bool logits_all) override;

    llama_memory_state_ptr init_full() override;

    llama_memory_state_ptr init_update(llama_context * lctx, bool optimize) override;

    void clear() override;

    bool seq_rm  (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1) override;
    void seq_cp  (llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) override;
    void seq_keep(llama_seq_id seq_id)                                                          override;
    void seq_add (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, llama_pos shift) override;
    void seq_div (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, int d) override;

    llama_pos seq_pos_min(llama_seq_id seq_id) const override;
    llama_pos seq_pos_max(llama_seq_id seq_id) const override;

    bool prepare(const std::vector<llama_ubatch> & ubatches);

    // find a contiguous slot of kv cells and emplace the ubatch there
    bool find_slot(const llama_ubatch & ubatch);

    bool get_can_shift() const override;

    // TODO: temporary methods - they are not really const as they do const_cast<>, fix this
    int32_t s_copy(int i) const;
    float   s_mask(int i) const;

    // state write/load

    void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1) const override;
    void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1) override;

    uint32_t head = 0; // the location where the batch will be placed in the cache (see find_slot())
    uint32_t size = 0; // total number of cells, shared across all sequences
    uint32_t used = 0; // used cells (i.e. at least one seq_id)

    // computed before each graph build
    uint32_t n = 0;

    // TODO: optimize for recurrent state needs
    struct kv_cell {
        llama_pos pos  = -1;
        int32_t   src  = -1; // used to copy states
        int32_t   tail = -1;

        std::set<llama_seq_id> seq_id;

        bool has_seq_id(const llama_seq_id & id) const {
            return seq_id.find(id) != seq_id.end();
        }

        bool is_empty() const {
            return seq_id.empty();
        }

        bool is_same_seq(const kv_cell & other) const {
            return seq_id == other.seq_id;
        }
    };

    std::vector<kv_cell> cells;

    std::vector<ggml_tensor *> k_l; // per layer
    std::vector<ggml_tensor *> v_l;

private:
    //const llama_model & model;
    const llama_hparams & hparams;

    const uint32_t n_seq_max = 1;

    std::vector<ggml_context_ptr>        ctxs;
    std::vector<ggml_backend_buffer_ptr> bufs;

    size_t total_size() const;

    size_t size_k_bytes() const;
    size_t size_v_bytes() const;

    void state_write_meta(llama_io_write_i & io, const std::vector<std::pair<uint32_t, uint32_t>> & cell_ranges, llama_seq_id seq_id = -1) const;
    void state_write_data(llama_io_write_i & io, const std::vector<std::pair<uint32_t, uint32_t>> & cell_ranges) const;

    bool state_read_meta(llama_io_read_i & io, uint32_t cell_count, llama_seq_id dest_seq_id = -1);
    bool state_read_data(llama_io_read_i & io, uint32_t cell_count);
};

class llama_kv_cache_recurrent_state : public llama_memory_state_i {
public:
    // used for errors
    llama_kv_cache_recurrent_state(llama_memory_status status);

    // used to create a full-cache state
    llama_kv_cache_recurrent_state(
            llama_memory_status status,
            llama_kv_cache_recurrent * kv);

    // used to create a state from a batch
    llama_kv_cache_recurrent_state(
            llama_memory_status status,
            llama_kv_cache_recurrent * kv,
            llama_sbatch sbatch,
            std::vector<llama_ubatch> ubatches);

    virtual ~llama_kv_cache_recurrent_state();

    //
    // llama_memory_state_i
    //

    bool next()  override;
    bool apply() override;

    std::vector<int64_t> & out_ids() override;

    llama_memory_status  get_status() const override;
    const llama_ubatch & get_ubatch() const override;

    //
    // llama_kv_cache_recurrent_state specific API
    //

    uint32_t get_n_kv() const;
    uint32_t get_head() const;
    uint32_t get_size() const;

    ggml_tensor * get_k_l(int32_t il) const;
    ggml_tensor * get_v_l(int32_t il) const;

    int32_t s_copy(int i) const;
    float   s_mask(int i) const;

private:
    const llama_memory_status status;

    llama_kv_cache_recurrent * kv;

    llama_sbatch sbatch;

    size_t i_next = 0;

    std::vector<llama_ubatch> ubatches;

    //
    // data needed for building the compute graph for the current ubatch:
    // TODO: extract all the state like `head` and `n` here
    //

    const bool is_full = false;
};
