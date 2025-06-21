#pragma once

#include "llama.h"

#include <memory>
#include <vector>

struct llama_ubatch;

class llama_io_write_i;
class llama_io_read_i;

struct llama_memory_params {
    // kv cache
    ggml_type type_k;
    ggml_type type_v;

    // use full-size SWA cache
    bool swa_full;
};

enum llama_memory_status {
    LLAMA_MEMORY_STATUS_SUCCESS = 0,
    LLAMA_MEMORY_STATUS_NO_UPDATE,
    LLAMA_MEMORY_STATUS_FAILED_PREPARE,
    LLAMA_MEMORY_STATUS_FAILED_COMPUTE,
};

// helper function for combining the status of two memory states
// useful for implementing hybrid memory types (e.g. iSWA)
llama_memory_status llama_memory_status_combine(llama_memory_status s0, llama_memory_status s1);

// the interface for managing the memory state during batch processing
// this interface is implemented per memory type. see:
//   - llama_kv_cache_unified_state
//   - llama_kv_cache_unified_iswa_state
//   ...
//
// the only method that can mutate the memory and the memory state is llama_memory_i::apply()
//
// TODO: rename to llama_memory_context_i ?
struct llama_memory_state_i {
    virtual ~llama_memory_state_i() = default;

    // consume the current ubatch from the state and proceed to the next one
    // return false if we are done
    virtual bool next() = 0;

    // apply the memory state for the current ubatch to the memory object
    // return false on failure
    virtual bool apply() = 0;

    // TODO: this might get reworked in the future when refactoring llama_batch
    virtual std::vector<int64_t> & out_ids() = 0;

    // get the current ubatch
    virtual const llama_ubatch & get_ubatch() const = 0;

    // get the status of the memory state - used for error handling and checking if any updates would be applied
    virtual llama_memory_status get_status() const = 0;
};

using llama_memory_state_ptr = std::unique_ptr<llama_memory_state_i>;

// general concept of LLM memory
// the KV cache is a type of LLM memory, but there can be other types
struct llama_memory_i {
    virtual ~llama_memory_i() = default;

    // split the input batch into a set of ubatches and verify that they can fit into the cache
    // return a state object containing the ubatches and KV cache state required to process them
    // check the llama_memory_state_i::get_status() for the result
    virtual llama_memory_state_ptr init_batch(
            const llama_batch & batch,
            uint32_t n_ubatch,
            bool embd_pooled,
            bool logits_all) = 0;

    // simulate full cache, used for allocating worst-case compute buffers
    virtual llama_memory_state_ptr init_full() = 0;

    // prepare for any pending memory updates, such as shifts, defrags, etc.
    // status == LLAMA_MEMORY_STATUS_NO_UPDATE if there is nothing to update
    virtual llama_memory_state_ptr init_update(llama_context * lctx, bool optimize) = 0;

    // getters
    virtual bool get_can_shift() const = 0;

    //
    // ops
    //

    virtual void clear() = 0;

    virtual bool seq_rm  (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1) = 0;
    virtual void seq_cp  (llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) = 0;
    virtual void seq_keep(llama_seq_id seq_id) = 0;
    virtual void seq_add (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, llama_pos shift) = 0;
    virtual void seq_div (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, int d) = 0;

    virtual llama_pos seq_pos_min(llama_seq_id seq_id) const = 0;
    virtual llama_pos seq_pos_max(llama_seq_id seq_id) const = 0;

    //
    // state write/read
    //

    virtual void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1) const = 0;
    virtual void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1) = 0;
};

using llama_memory_ptr = std::unique_ptr<llama_memory_i>;

// TODO: temporary until the llama_kv_cache is removed from the public API
struct llama_kv_cache : public llama_memory_i {
    virtual ~llama_kv_cache() = default;
};
