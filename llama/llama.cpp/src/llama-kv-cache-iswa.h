#pragma once

#include "llama-kv-cache.h"

#include <vector>

//
// llama_kv_cache_iswa
//

// utilizes two instances of llama_kv_cache
//   the first instance is for the non-SWA layers of the model and the second instance is for the SWA layers

class llama_kv_cache_iswa : public llama_memory_i {
public:
    llama_kv_cache_iswa(
            const llama_model & model,
                    ggml_type   type_k,
                    ggml_type   type_v,
                         bool   v_trans,
                         bool   offload,
                         bool   swa_full,
                         bool   unified,
                     uint32_t   kv_size,
                     uint32_t   n_seq_max,
                     uint32_t   n_ubatch,
                     uint32_t   n_pad,
        const layer_filter_cb & filter,
        const  layer_reuse_cb & reuse);

    ~llama_kv_cache_iswa() = default;

    //
    // llama_memory_i
    //

    llama_memory_context_ptr init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) override;

    llama_memory_context_ptr init_full() override;

    llama_memory_context_ptr init_update(llama_context * lctx, bool optimize) override;

    bool get_can_shift() const override;

    void clear(bool data) override;

    bool seq_rm  (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1) override;
    void seq_cp  (llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) override;
    void seq_keep(llama_seq_id seq_id)                                                          override;
    void seq_add (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, llama_pos shift) override;
    void seq_div (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, int d) override;

    llama_pos seq_pos_min(llama_seq_id seq_id) const override;
    llama_pos seq_pos_max(llama_seq_id seq_id) const override;

    // state write/load

    void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) const override;
    void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) override;

    //
    // llama_kv_cache_iswa specific API
    //

    llama_kv_cache * get_base() const;
    llama_kv_cache * get_swa () const;

private:
    const llama_hparams & hparams;

    const bool unified;

    std::unique_ptr<llama_kv_cache> kv_base;
    std::unique_ptr<llama_kv_cache> kv_swa;
};

class llama_kv_cache_iswa_context : public llama_memory_context_i {
public:
    using slot_info_vec_t = llama_kv_cache::slot_info_vec_t;

    // used for errors
    llama_kv_cache_iswa_context(llama_memory_status status);

    // used to create a full-cache context
    llama_kv_cache_iswa_context(
            llama_kv_cache_iswa * kv);

    // used to create an update context
    llama_kv_cache_iswa_context(
            llama_kv_cache_iswa * kv,
            llama_context * lctx,
            bool optimize);

    // used to create a batch processing context from a batch
    llama_kv_cache_iswa_context(
            llama_kv_cache_iswa * kv,
            slot_info_vec_t sinfos_base,
            slot_info_vec_t sinfos_swa,
            std::vector<llama_ubatch> ubatches);

    virtual ~llama_kv_cache_iswa_context();

    //
    // llama_memory_context_i
    //

    bool next()  override;
    bool apply() override;

    llama_memory_status  get_status() const override;
    const llama_ubatch & get_ubatch() const override;

    //
    // llama_kv_cache_iswa_context specific API
    //

    const llama_kv_cache_context * get_base() const;
    const llama_kv_cache_context * get_swa()  const;

private:
    //llama_kv_cache_iswa * kv;

    // the index of the next ubatch to process
    size_t i_next = 0;

    std::vector<llama_ubatch> ubatches;

    const llama_memory_context_ptr ctx_base;
    const llama_memory_context_ptr ctx_swa;

    const llama_memory_status status;
};
