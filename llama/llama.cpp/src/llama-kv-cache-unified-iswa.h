#pragma once

#include "llama-kv-cache-unified.h"

#include <vector>

//
// llama_kv_cache_unified_iswa
//

// utilizes two instances of llama_kv_cache_unified
//   the first instance is for the non-SWA layers of the model and the second instance is for the SWA layers

class llama_kv_cache_unified_iswa : public llama_memory_i {
public:
    llama_kv_cache_unified_iswa(
            const llama_model & model,
                    ggml_type   type_k,
                    ggml_type   type_v,
                         bool   v_trans,
                         bool   offload,
                         bool   swa_full,
                     uint32_t   kv_size,
                     uint32_t   n_seq_max,
                     uint32_t   n_ubatch,
                     uint32_t   n_pad);

    ~llama_kv_cache_unified_iswa() = default;

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

    bool get_can_shift() const override;

    void clear() override;

    bool seq_rm  (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1) override;
    void seq_cp  (llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) override;
    void seq_keep(llama_seq_id seq_id)                                                          override;
    void seq_add (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, llama_pos shift) override;
    void seq_div (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, int d) override;

    llama_pos seq_pos_min(llama_seq_id seq_id) const override;
    llama_pos seq_pos_max(llama_seq_id seq_id) const override;

    // state write/load

    void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1) const override;
    void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1)       override;

    //
    // llama_kv_cache_unified_iswa specific API
    //

    llama_kv_cache_unified * get_base() const;
    llama_kv_cache_unified * get_swa () const;

private:
    const llama_hparams & hparams;

    std::unique_ptr<llama_kv_cache_unified> kv_base;
    std::unique_ptr<llama_kv_cache_unified> kv_swa;
};

class llama_kv_cache_unified_iswa_state : public llama_memory_state_i {
public:
    // used for errors
    llama_kv_cache_unified_iswa_state(llama_memory_status status);

    // used to create a full-cache state
    llama_kv_cache_unified_iswa_state(
            llama_kv_cache_unified_iswa * kv);

    // used to create an update state
    llama_kv_cache_unified_iswa_state(
            llama_kv_cache_unified_iswa * kv,
            llama_context * lctx,
            bool optimize);

    // used to create a state from a batch
    llama_kv_cache_unified_iswa_state(
            llama_kv_cache_unified_iswa * kv,
            llama_sbatch sbatch,
            std::vector<uint32_t> heads_base,
            std::vector<uint32_t> heads_swa,
            std::vector<llama_ubatch> ubatches);

    virtual ~llama_kv_cache_unified_iswa_state();

    //
    // llama_memory_state_i
    //

    bool next()  override;
    bool apply() override;

    std::vector<int64_t> & out_ids() override;

    llama_memory_status  get_status() const override;
    const llama_ubatch & get_ubatch() const override;

    //
    // llama_kv_cache_unified_iswa_state specific API
    //

    const llama_kv_cache_unified_state * get_base() const;
    const llama_kv_cache_unified_state * get_swa()  const;

private:
    llama_memory_status status;

    //llama_kv_cache_unified_iswa * kv;

    llama_sbatch sbatch;

    // the index of the next ubatch to process
    size_t i_next = 0;

    std::vector<llama_ubatch> ubatches;

    llama_memory_state_ptr state_base;
    llama_memory_state_ptr state_swa;
};
