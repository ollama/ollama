#pragma once

#include "llama.h"

#include "ggml-cpp.h"

#include <set>
#include <vector>

struct llama_kv_cell {
    llama_pos pos   = -1;
    llama_pos delta = 0;
    int32_t   src   = -1; // used by recurrent state models to copy states
    int32_t   tail  = -1;

    std::set<llama_seq_id> seq_id;

    bool has_seq_id(const llama_seq_id & id) const {
        return seq_id.find(id) != seq_id.end();
    }

    bool is_empty() const {
        return seq_id.empty();
    }

    bool is_same_seq(const llama_kv_cell & other) const {
        return seq_id == other.seq_id;
    }
};

// ring-buffer of cached KV data
struct llama_kv_cache {
    bool has_shift = false;
    bool do_defrag = false;
    bool recurrent = false; // with recurrent state models, a cell can hold the state for more than one past token
    bool v_trans   = true;  // the value tensor is transposed
    bool can_shift = false;

    // Note: The value of head isn't only used to optimize searching
    // for a free KV slot. llama_decode_impl also uses it, so it
    // cannot be freely changed after a slot has been allocated.
    uint32_t head = 0;
    uint32_t size = 0;
    uint32_t used = 0; // used cells (i.e. at least one seq_id)

    // computed before each graph build
    uint32_t n = 0;

    ggml_type type_k = GGML_TYPE_F16;
    ggml_type type_v = GGML_TYPE_F16;

    std::vector<llama_kv_cell> cells;

    std::vector<struct ggml_tensor *> k_l; // per layer
    std::vector<struct ggml_tensor *> v_l;

    std::vector<ggml_context_ptr> ctxs;
    std::vector<ggml_backend_buffer_ptr> bufs;

    size_t total_size() const {
        size_t size = 0;
        for (const auto & buf : bufs) {
            size += ggml_backend_buffer_get_size(buf.get());
        }

        return size;
    }

    // TODO: better data structures to reduce the cost of this operation
    llama_pos max_pos() const {
        llama_pos max_pos = -1;
        for (const auto & cell : cells) {
            max_pos = std::max(max_pos, cell.pos);
        }

        return max_pos;
    }
};

// a structure holds information about the slot found in llama_kv_cache_find_slot
struct llama_kv_cache_slot_info {
    std::pair<uint32_t, uint32_t> boundaries; // slot boundaries [begin, end)
    bool found = false;                       // the slot was found

    explicit llama_kv_cache_slot_info(bool found_) : found{found_} {}
    llama_kv_cache_slot_info(uint32_t begin, uint32_t end) : boundaries{begin, end}, found{true} {}

    operator bool() const { return found; }
};

// TODO: maybe not needed
uint32_t llama_kv_cache_get_padding(const struct llama_cparams & cparams);

bool llama_kv_cache_init(
        struct llama_kv_cache & cache,
            const llama_model & model,
          const llama_cparams & cparams,
                    ggml_type   type_k,
                    ggml_type   type_v,
                     uint32_t   kv_size,
                         bool   offload);

// find an empty slot of size "n_tokens" in the cache
// updates the cache head
// returns a structure holding information about the slot found
// Note: On success, it's important that cache.head points
// to the first cell of the slot.
struct llama_kv_cache_slot_info llama_kv_cache_find_slot(
           struct llama_kv_cache & cache,
       const struct llama_ubatch & batch);

// find how many cells are currently in use
uint32_t llama_kv_cache_cell_max(const struct llama_kv_cache & cache);

void llama_kv_cache_clear(struct llama_kv_cache & cache);

bool llama_kv_cache_seq_rm(
        struct llama_kv_cache & cache,
                 llama_seq_id   seq_id,
                    llama_pos   p0,
                    llama_pos   p1);

void llama_kv_cache_seq_cp(
        struct llama_kv_cache & cache,
                 llama_seq_id   seq_id_src,
                 llama_seq_id   seq_id_dst,
                    llama_pos   p0,
                    llama_pos   p1);

void llama_kv_cache_seq_keep(
        struct llama_kv_cache & cache,
                 llama_seq_id   seq_id);

void llama_kv_cache_seq_add(
        struct llama_kv_cache & cache,
                 llama_seq_id   seq_id,
                    llama_pos   p0,
                    llama_pos   p1,
                    llama_pos   delta);

void llama_kv_cache_seq_div(
        struct llama_kv_cache & cache,
                 llama_seq_id   seq_id,
                    llama_pos   p0,
                    llama_pos   p1,
                          int   d);

llama_pos llama_kv_cache_seq_pos_max(
        struct llama_kv_cache & cache,
                 llama_seq_id   seq_id);

void llama_kv_cache_defrag(struct llama_kv_cache & cache);

int32_t llama_get_kv_cache_token_count(const struct llama_kv_cache & kv);

int32_t llama_get_kv_cache_used_cells(const struct llama_kv_cache & kv);

bool llama_kv_cache_can_shift(const struct llama_kv_cache & kv);

//
// kv cache view
//

struct llama_kv_cache_view llama_kv_cache_view_init(const struct llama_kv_cache & kv, int32_t n_seq_max);

void llama_kv_cache_view_update(struct llama_kv_cache_view * view, const struct llama_kv_cache & kv);

//
// kv cache restore
//

// saves the kv_cache state for future recovery.
// used to rollback llama_kv_cache_find_slot changes.
struct llama_kv_slot_restorer {
    struct llama_kv_cache_state {
        uint32_t head = 0;
        uint32_t n    = 0;
    } old_state;

    // for non-recurrent models only
    // list of slots to restore
    std::vector<std::pair<uint32_t, uint32_t>> slot_boundaries;

    bool do_restore = false;

    explicit llama_kv_slot_restorer(const struct llama_kv_cache & cache) {
        old_state.head = cache.head;
        old_state.n    = cache.n;
    }

    // saves a slot information for future restoration
    void save(const struct llama_kv_cache_slot_info & slot) {
        if (slot) {
            do_restore = true;
            if (slot.boundaries.first != slot.boundaries.second) {
                slot_boundaries.push_back(slot.boundaries);
            }
        }
    }

    // must be explicitly called to restore the kv_cache state
    // and rollback changes from all llama_kv_cache_find_slot calls
    void restore(struct llama_kv_cache & cache) {
        if (do_restore) {
            cache.head = old_state.head;
            cache.n    = old_state.n;

            if (cache.recurrent) { // recurrent models like Mamba or RWKV can't have a state partially erased
                llama_kv_cache_seq_rm(cache, -1, -1, -1);
            } else {
                for (auto & slot : slot_boundaries) {
                    llama_kv_cache_seq_rm(cache, -1, slot.first, slot.second);
                }
            }
        }
    }
};

