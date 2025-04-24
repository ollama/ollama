#include "llama-kv-cache.h"

#include "llama-impl.h"
#include "llama-batch.h"
#include "llama-cparams.h"
#include "llama-model.h"

#include <algorithm>
#include <cassert>
#include <limits>
#include <map>
#include <stdexcept>

llama_kv_cache_unified::llama_kv_cache_unified(const llama_hparams & hparams, callbacks cbs) : hparams(hparams), cbs(std::move(cbs)) {
}

bool llama_kv_cache_unified::init(
        const llama_model & model,
      const llama_cparams & cparams,
                ggml_type   type_k,
                ggml_type   type_v,
                 uint32_t   kv_size,
                     bool   offload) {
    const int32_t n_layer = hparams.n_layer;

    has_shift = false;

    recurrent = llama_model_is_recurrent(&model);
    v_trans   = !recurrent && !cparams.flash_attn;
    can_shift = !recurrent && model.arch != LLM_ARCH_DEEPSEEK2; // not supported due to MLA

    LLAMA_LOG_INFO("%s: kv_size = %d, offload = %d, type_k = '%s', type_v = '%s', n_layer = %d, can_shift = %d\n",
            __func__, kv_size, offload, ggml_type_name(type_k), ggml_type_name(type_v), n_layer, can_shift);

    head = 0;
    size = kv_size;
    used = 0;

    this->type_k = type_k;
    this->type_v = type_v;

    cells.clear();
    cells.resize(kv_size);

    // create a context for each buffer type
    std::map<ggml_backend_buffer_type_t, ggml_context *> ctx_map;
    auto ctx_for_buft = [&](ggml_backend_buffer_type_t buft) -> ggml_context * {
        auto it = ctx_map.find(buft);
        if (it == ctx_map.end()) {
            ggml_init_params params = {
                /*.mem_size   =*/ size_t(2u*n_layer*ggml_tensor_overhead()),
                /*.mem_buffer =*/ NULL,
                /*.no_alloc   =*/ true,
            };

            ggml_context * ctx = ggml_init(params);
            if (!ctx) {
                return nullptr;
            }

            ctx_map[buft] = ctx;
            ctxs.emplace_back(ctx);

            return ctx;
        }

        return it->second;
    };

    k_l.reserve(n_layer);
    v_l.reserve(n_layer);

    for (int i = 0; i < n_layer; i++) {
        const uint32_t n_embd_k_gqa = hparams.n_embd_k_gqa(i) + hparams.n_embd_k_s();
        const uint32_t n_embd_v_gqa = hparams.n_embd_v_gqa(i) + hparams.n_embd_v_s();

        const char * dev_name = "CPU";

        ggml_backend_buffer_type_t buft;
        if (offload) {
            auto * dev = model.dev_layer(i);
            buft = ggml_backend_dev_buffer_type(dev);

            dev_name = ggml_backend_dev_name(dev);
        } else {
            buft = ggml_backend_cpu_buffer_type();
        }

        LLAMA_LOG_DEBUG("%s: layer %3d: n_embd_k_gqa = %d, n_embd_v_gqa = %d, dev = %s\n", __func__,
                i, n_embd_k_gqa, n_embd_v_gqa, dev_name);

        ggml_context * ctx = ctx_for_buft(buft);
        if (!ctx) {
            LLAMA_LOG_ERROR("%s: failed to create ggml context for kv cache\n", __func__);
            return false;
        }

        ggml_tensor * k, *v;

        // for cross attention layers
        if (model.arch == LLM_ARCH_MLLAMA && hparams.cross_attention_layers(i)) {
            k = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hparams.n_embd_head_k, 6404, hparams.n_head_kv(i));
            v = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hparams.n_embd_head_v, 6404, hparams.n_head_kv(i));
        } else {
            k = ggml_new_tensor_1d(ctx, type_k, n_embd_k_gqa*kv_size);
            v = ggml_new_tensor_1d(ctx, type_v, n_embd_v_gqa*kv_size);
        }
        ggml_format_name(k, "cache_k_l%d", i);
        ggml_format_name(v, "cache_v_l%d", i);
        k_l.push_back(k);
        v_l.push_back(v);
    }

    // allocate tensors and initialize the buffers to avoid NaNs in the padding
    for (auto it : ctx_map) {
        auto * buft = it.first;
        auto * ctx  = it.second;

        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
        if (!buf) {
            LLAMA_LOG_ERROR("%s: failed to allocate buffer for kv cache\n", __func__);
            return false;
        }
        ggml_backend_buffer_clear(buf, 0);
        LLAMA_LOG_INFO("%s: %10s KV buffer size = %8.2f MiB\n", __func__, ggml_backend_buffer_name(buf), ggml_backend_buffer_get_size(buf)/1024.0/1024.0);
        bufs.emplace_back(buf);
    }

    return true;
}

int32_t llama_kv_cache_unified::get_n_tokens() const {
    int32_t result = 0;

    for (uint32_t i = 0; i < size; i++) {
        result += cells[i].seq_id.size();
    }

    return result;
}

int32_t llama_kv_cache_unified::get_used_cells() const {
    return used;
}

size_t llama_kv_cache_unified::total_size() const {
    size_t size = 0;
    for (const auto & buf : bufs) {
        size += ggml_backend_buffer_get_size(buf.get());
    }

    return size;
}

llama_pos llama_kv_cache_unified::pos_max() const {
    llama_pos pos_max = -1;
    for (const auto & cell : cells) {
        pos_max = std::max(pos_max, cell.pos);
    }

    return pos_max;
}

void llama_kv_cache_unified::clear() {
    for (int32_t i = 0; i < (int32_t) size; ++i) {
        cells[i].pos = -1;
        cells[i].seq_id.clear();
        cells[i].src = -1;
        cells[i].tail = -1;
    }
    head = 0;
    used = 0;

    for (auto & buf : bufs) {
        ggml_backend_buffer_clear(buf.get(), 0);
    }
}

bool llama_kv_cache_unified::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    uint32_t new_head = size;

    if (p0 < 0) {
        p0 = 0;
    }

    if (p1 < 0) {
        p1 = std::numeric_limits<llama_pos>::max();
    }

    // models like Mamba or RWKV can't have a state partially erased
    if (recurrent) {
        if (seq_id >= (int64_t) size) {
            // could be fatal
            return false;
        }
        if (0 <= seq_id) {
            int32_t & tail_id = cells[seq_id].tail;
            if (tail_id >= 0) {
                const llama_kv_cell & cell = cells[tail_id];
                // partial intersection is invalid
                if ((0 < p0 && p0 <= cell.pos) || (0 < p1 && p1 <= cell.pos)) {
                    return false;
                }
                // invalidate tails which will be cleared
                if (p0 <= cell.pos && cell.pos < p1) {
                    tail_id = -1;
                }
            }
        } else {
            // seq_id is negative, then the range should include everything or nothing
            if (p0 != p1 && (p0 != 0 || p1 != std::numeric_limits<llama_pos>::max())) {
                return false;
            }
        }

        return true;
    }

    for (uint32_t i = 0; i < size; ++i) {
        if (cells[i].pos >= p0 && cells[i].pos < p1) {
            if (seq_id < 0) {
                cells[i].seq_id.clear();
            } else if (cells[i].has_seq_id(seq_id)) {
                cells[i].seq_id.erase(seq_id);
            } else {
                continue;
            }
            if (cells[i].is_empty()) {
                // keep count of the number of used cells
                if (cells[i].pos >= 0) {
                    used--;
                }

                cells[i].pos = -1;
                cells[i].src = -1;

                if (new_head == size) {
                    new_head = i;
                }
            }
        }
    }

    // If we freed up a slot, set head to it so searching can start there.
    if (new_head != size && new_head < head) {
        head = new_head;
    }

    return true;
}

void llama_kv_cache_unified::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    if (seq_id_src == seq_id_dst) {
        return;
    }

    if (p0 < 0) {
        p0 = 0;
    }

    if (p1 < 0) {
        p1 = std::numeric_limits<llama_pos>::max();
    }

    if (recurrent) {
        if ((uint32_t) seq_id_dst < size && (uint32_t) seq_id_src < size) {
            llama_kv_cell & tail_src = cells[seq_id_src];
            llama_kv_cell & tail_dst = cells[seq_id_dst];
            if (tail_dst.tail >= 0) {
                // clear destination seq_id if it wasn't empty
                llama_kv_cell & cell_dst = cells[tail_dst.tail];

                cell_dst.seq_id.erase(seq_id_dst);
                tail_dst.tail = -1;
                if (cell_dst.seq_id.empty()) {
                    cell_dst.pos = -1;
                    cell_dst.delta = -1;
                    cell_dst.src = -1;
                    used -= 1;
                }
            }
            if (tail_src.tail >= 0) {
                llama_kv_cell & cell_src = cells[tail_src.tail];

                cell_src.seq_id.insert(seq_id_dst);
                tail_dst.tail = tail_src.tail;
            }
        }

        return;
    }

    // otherwise, this is the KV of a Transformer-like model
    head = 0;

    for (uint32_t i = 0; i < size; ++i) {
        if (cells[i].has_seq_id(seq_id_src) && cells[i].pos >= p0 && cells[i].pos < p1) {
            cells[i].seq_id.insert(seq_id_dst);
        }
    }
}

void llama_kv_cache_unified::seq_keep(llama_seq_id seq_id) {
    uint32_t new_head = size;

    for (uint32_t i = 0; i < size; ++i) {
        if (recurrent && (llama_seq_id) i != seq_id) {
            cells[i].tail = -1;
        }

        if (!cells[i].has_seq_id(seq_id)) {
            if (cells[i].pos >= 0) {
                used--;
            }

            cells[i].pos = -1;
            cells[i].src = -1;
            cells[i].seq_id.clear();

            if (new_head == size){
                new_head = i;
            }
        } else {
            cells[i].seq_id.clear();
            cells[i].seq_id.insert(seq_id);
        }
    }

    // If we freed up a slot, set head to it so searching can start there.
    if (new_head != size && new_head < head) {
        head = new_head;
    }
}

void llama_kv_cache_unified::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos delta) {
    if (delta == 0) {
        return;
    }

    uint32_t new_head = size;

    if (p0 < 0) {
        p0 = 0;
    }

    if (p1 < 0) {
        p1 = std::numeric_limits<llama_pos>::max();
    }

    // If there is no range then return early to avoid looping over the
    if (p0 == p1) {
        return;
    }

    if (recurrent) {
        // for Mamba-like or RWKV models, only the pos needs to be shifted
        if (0 <= seq_id && seq_id < (int64_t) size) {
            const int32_t tail_id = cells[seq_id].tail;
            if (tail_id >= 0) {
                llama_kv_cell & cell = cells[tail_id];
                if (cell.has_seq_id(seq_id) && p0 <= cell.pos && cell.pos < p1) {
                    cell.pos += delta;
                }
            }
        }
        return;
    }

    for (uint32_t i = 0; i < size; ++i) {
        if (cells[i].has_seq_id(seq_id) && cells[i].pos >= p0 && cells[i].pos < p1) {
            has_shift = true;
            cells[i].pos   += delta;
            cells[i].delta += delta;

            if (cells[i].pos < 0) {
                if (!cells[i].is_empty()) {
                    used--;
                }
                cells[i].pos = -1;
                cells[i].seq_id.clear();
                if (new_head == size) {
                    new_head = i;
                }
            }
        }
    }

    // If we freed up a slot, set head to it so searching can start there.
    // Otherwise we just start the next search from the beginning.
    head = new_head != size ? new_head : 0;
}

void llama_kv_cache_unified::seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
    if (d == 1) {
        return;
    }

    if (p0 < 0) {
        p0 = 0;
    }

    if (p1 < 0) {
        p1 = std::numeric_limits<llama_pos>::max();
    }

    // If there is no range then return early to avoid looping over the cache.
    if (p0 == p1) {
        return;
    }

    if (recurrent) {
        // for Mamba-like or RWKV models, only the pos needs to be changed
        if (0 <= seq_id && seq_id < (int64_t) size) {
            const int32_t tail_id = cells[seq_id].tail;
            if (tail_id >= 0) {
                llama_kv_cell & cell = cells[tail_id];
                if (cell.has_seq_id(seq_id) && p0 <= cell.pos && cell.pos < p1) {
                    cell.pos /= d;
                }
            }
        }

        return;
    }

    for (uint32_t i = 0; i < size; ++i) {
        if (cells[i].has_seq_id(seq_id) && cells[i].pos >= p0 && cells[i].pos < p1) {
            has_shift = true;

            {
                llama_pos p_old = cells[i].pos;
                cells[i].pos   /= d;
                cells[i].delta += cells[i].pos - p_old;
            }
        }
    }
}

llama_pos llama_kv_cache_unified::seq_pos_max(llama_seq_id seq_id) const {
    llama_pos result = 0;

    for (uint32_t i = 0; i < size; ++i) {
        if (cells[i].has_seq_id(seq_id)) {
            result = std::max(result, cells[i].pos);
        }
    }

    return result;
}

void llama_kv_cache_unified::defrag() {
    if (!recurrent) {
        do_defrag = true;
    }
}

void llama_kv_cache_unified::restore() {
    if (pending.ranges.empty()) {
        return;
    }

    // TODO: tmp - move to llama_kv_cache_recurrent
    if (recurrent) {
        seq_rm(-1, -1, -1);
        return;
    }

    uint32_t new_head = size;

    for (auto & range : pending.ranges) {
        for (uint32_t i = range.c0; i < range.c1; ++i) {
            cells[i].seq_id.clear();

            // keep count of the number of used cells
            if (cells[i].pos >= 0) {
                used--;
            }

            cells[i].pos = -1;
            cells[i].src = -1;
        }

        new_head = std::min(new_head, range.c0);
    }

    if (new_head != size && new_head < head) {
        head = new_head;
    }
}

void llama_kv_cache_unified::commit() {
    // TODO: tmp - move to llama_kv_cache_recurrent
    if (recurrent) {
        return;
    }

    if (pending.ranges.empty()) {
        LLAMA_LOG_WARN("%s: no pending KV cache updates to commit - might indicate a bug (ref: %s)\n",
                __func__, "https://github.com/ggml-org/llama.cpp/pull/12695");
        return;
    }

    pending.ranges.clear();
}

bool llama_kv_cache_unified::get_can_shift() const {
    return can_shift;
}

bool llama_kv_cache_unified::find_slot(
       const llama_ubatch & ubatch) {
    const uint32_t n_tokens = ubatch.n_tokens;
    const uint32_t n_seqs   = ubatch.n_seqs;
    const uint32_t n_seq_tokens = ubatch.n_seq_tokens;

    // if we have enough unused cells before the current head ->
    //   better to start searching from the beginning of the cache, hoping to fill it
    if (head > used + 2*ubatch.n_tokens) {
        head = 0;
    }

    if (recurrent) {
        // For recurrent state architectures (like Mamba or RWKV),
        // each cache cell can store the state for a whole sequence.
        // A slot should be always be contiguous.

        // can only process batches with an equal number of new tokens in each sequence
        GGML_ASSERT(ubatch.equal_seqs);

        int32_t min = size - 1;
        int32_t max = 0;

        // everything should fit if all seq_ids are smaller than the max
        for (uint32_t s = 0; s < n_seqs; ++s) {
            const uint32_t n_seq_id = ubatch.n_seq_id[s];
            for (uint32_t j = 0; j < n_seq_id; ++j) {
                const llama_seq_id seq_id = ubatch.seq_id[s][j];

                if (seq_id < 0 || (uint32_t) seq_id >= size) {
                    // too big seq_id
                    // TODO: would it be possible to resize the cache instead?
                    LLAMA_LOG_ERROR("%s: seq_id=%d >= n_seq_max=%d Try using a bigger --parallel value\n", __func__, seq_id, size);
                    return false;
                }
                if (j > 0) {
                    llama_kv_cell & seq = cells[seq_id];
                    if (seq.tail >= 0) {
                        llama_kv_cell & cell = cells[seq.tail];
                        // clear cells from seq_ids that become shared
                        // (should not normally happen, but let's handle it anyway)
                        cell.seq_id.erase(seq_id);
                        seq.tail = -1;
                        if (cell.seq_id.empty()) {
                            cell.pos = -1;
                            cell.src = -1;
                            used -= 1;
                        }
                    }
                }
            }
        }

#ifndef NDEBUG
        {
            std::vector<int32_t> tails_verif;
            tails_verif.assign(size, -1);
            for (uint32_t i = 0; i < size; ++i) {
                llama_kv_cell & cell = cells[i];
                for (llama_seq_id seq_id : cell.seq_id) {
                    if (tails_verif[seq_id] != -1) {
                        LLAMA_LOG_ERROR("%s: duplicate tail for seq_id %d in cell %d and %d\n", __func__, seq_id, i, tails_verif[seq_id]);
                    }
                    tails_verif[seq_id] = i;
                }
            }
            for (uint32_t i = 0; i < size; ++i) {
                if (tails_verif[i] != cells[i].tail) {
                    LLAMA_LOG_ERROR("%s: wrong tail for seq_id %d, (%d instead of %d)\n", __func__, i, cells[i].tail, tails_verif[i]);
                }
            }
        }
#endif

        // find next empty cell
        uint32_t next_empty_cell = head;

        for (uint32_t i = 0; i < size; ++i) {
            if (next_empty_cell >= size) { next_empty_cell -= size; }
            llama_kv_cell & cell = cells[next_empty_cell];
            if (cell.is_empty()) { break; }
            next_empty_cell += 1;
        }

        // find usable cell range
        for (uint32_t s = 0; s < n_seqs; ++s) {
            const llama_seq_id seq_id = ubatch.seq_id[s][0];
            llama_kv_cell & seq_meta = cells[seq_id];
            bool has_cell = false;
            if (seq_meta.tail >= 0) {
                llama_kv_cell & cell = cells[seq_meta.tail];
                GGML_ASSERT(cell.has_seq_id(seq_id));
                // does this seq_id "own" the cell?
                if (cell.seq_id.size() == 1) { has_cell = true; }
            }
            if (!has_cell) {
                llama_kv_cell & empty_cell = cells[next_empty_cell];
                GGML_ASSERT(empty_cell.is_empty());
                // copy old tail into the empty cell
                if (seq_meta.tail >= 0) {
                    llama_kv_cell & orig_cell = cells[seq_meta.tail];
                    empty_cell.pos = orig_cell.pos;
                    empty_cell.src = orig_cell.src;
                    orig_cell.seq_id.erase(seq_id);
                    empty_cell.seq_id.insert(seq_id); // will be overwritten
                }
                seq_meta.tail = next_empty_cell;
                // find next empty cell
                if (s + 1 < n_seqs) {
                    next_empty_cell += 1;
                    for (uint32_t i = 0; i < size; ++i) {
                        if (next_empty_cell >= size) { next_empty_cell -= size; }
                        llama_kv_cell & cell = cells[next_empty_cell];
                        if (cell.is_empty()) { break; }
                        next_empty_cell += 1;
                    }
                }
            }
            if (min > seq_meta.tail) { min = seq_meta.tail; }
            if (max < seq_meta.tail) { max = seq_meta.tail; }
        }

        // gather and re-order
        for (uint32_t s = 0; s < n_seqs; ++s) {
            int32_t dst_id = s + min;
            int32_t src_id = cells[ubatch.seq_id[s][0]].tail;
            if (dst_id != src_id) {
                llama_kv_cell & dst_cell = cells[dst_id];
                llama_kv_cell & src_cell = cells[src_id];

                std::swap(dst_cell.pos, src_cell.pos);
                std::swap(dst_cell.src, src_cell.src);
                std::swap(dst_cell.seq_id, src_cell.seq_id);

                // swap tails (assuming they NEVER overlap)
                for (const llama_seq_id seq_id : src_cell.seq_id) {
                    cells[seq_id].tail = src_id;
                }
                for (const llama_seq_id seq_id : dst_cell.seq_id) {
                    cells[seq_id].tail = dst_id;
                }
            }
        }

        // update the pos of the used seqs
        for (uint32_t s = 0; s < n_seqs; ++s) {
            const llama_pos last_pos = ubatch.pos[n_seq_tokens * s + n_seq_tokens - 1];
            int32_t cell_id = s + min;
            llama_kv_cell & cell = cells[cell_id];

            if (cell.pos >= 0 && last_pos != cell.pos + (llama_pos) n_seq_tokens) {
                // What should happen when the pos backtracks or skips a value?
                // Clearing the state mid-batch would require special-casing which isn't done.
                LLAMA_LOG_WARN("%s: non-consecutive token position %d after %d for sequence %d with %u new tokens\n",
                    __func__, last_pos, cell.pos, ubatch.seq_id[s][0], n_seq_tokens);
            }
            cell.pos = last_pos;
            cell.seq_id.clear();
            for (int32_t j = 0; j < ubatch.n_seq_id[s]; ++j) {
                const llama_seq_id seq_id = ubatch.seq_id[s][j];
                cell.seq_id.insert(seq_id);
                cells[seq_id].tail = cell_id;
            }
        }

        // allow getting the range of used cells, from head to head + n
        head = min;
        n    = max - min + 1;
        used = std::count_if(cells.begin(), cells.end(),
            [](const llama_kv_cell& cell){ return !cell.is_empty(); });

        // sanity check
        return n >= n_seqs;
    }

    // otherwise, one cell per token.

    if (n_tokens > size) {
        LLAMA_LOG_ERROR("%s: n_tokens = %d > size = %d\n", __func__, n_tokens, size);
        return false;
    }

    uint32_t n_tested = 0;

    while (true) {
        if (head + n_tokens > size) {
            n_tested += size - head;
            head = 0;
            continue;
        }

        bool found = true;
        for (uint32_t i = 0; i < n_tokens; i++) {
            if (cells[head + i].pos >= 0) {
                found = false;
                head     += i + 1;
                n_tested += i + 1;
                break;
            }
        }

        if (found) {
            break;
        }

        if (n_tested >= size) {
            //LLAMA_LOG_ERROR("%s: failed to find a slot for %d tokens\n", __func__, n_tokens);
            return false;
        }
    }

    for (uint32_t s = 0; s < n_seqs; s++) {
        for (uint32_t i = 0; i < n_seq_tokens; ++i) {
            uint32_t k = s*n_seq_tokens + i;
            cells[head + k].pos = ubatch.pos[k];

            for (int32_t j = 0; j < ubatch.n_seq_id[s]; j++) {
                cells[head + k].seq_id.insert(ubatch.seq_id[s][j]);
            }
        }
    }

    used += n_tokens;

    pending.ranges.push_back({head, head + n_tokens});

    return true;
}

uint32_t llama_kv_cache_unified::get_padding(const llama_cparams & cparams) const {
    // the FA kernels require padding to avoid extra runtime boundary checks
    return cparams.flash_attn ? 256u : 32u;
}

uint32_t llama_kv_cache_unified::cell_max() const {
    for (uint32_t i = size; i > 0; --i) {
        const llama_kv_cell & cell = cells[i - 1];

        if (cell.pos >= 0 && !cell.is_empty()) {
            return i;
        }
    }

    return 0;
}

size_t llama_kv_cache_unified::size_k_bytes() const {
    size_t size_k_bytes = 0;

    for (const auto & k : k_l) {
        size_k_bytes += ggml_nbytes(k);
    }

    return size_k_bytes;
}

size_t llama_kv_cache_unified::size_v_bytes() const {
    size_t size_v_bytes = 0;

    for (const auto & v : v_l) {
        size_v_bytes += ggml_nbytes(v);
    }

    return size_v_bytes;
}

bool llama_kv_cache_unified::defrag_prepare(int32_t n_max_nodes) {
    const uint32_t n_layer = hparams.n_layer;

    const uint32_t n_kv   = cell_max();
    const uint32_t n_used = used;

    assert(n_used <= n_kv);

    defrag_info.moves.clear();

    // determine which KV cells to move where
    //
    //  cell i moves to ids[i]
    //
    //  if ids[i] == i || ids[i] == n_kv, then cell i is not moved
    //
    std::vector<uint32_t> ids(n_kv, n_kv);

    for (uint32_t i0 = 0; i0 < n_used; ++i0) {
        const auto & cell0 = cells[i0];

        if (!cell0.is_empty()) {
            ids[i0] = i0;

            continue;
        }

        // found a hole - fill it with data from the end of the cache

        uint32_t nh = 1;

        // determine the size of the hole
        while (i0 + nh < n_used && cells[i0 + nh].is_empty()) {
            nh++;
        }

        uint32_t nf = 0;
        uint32_t is = n_kv - 1;

        // starting from the end, find nh non-empty cells
        for (; is > i0; --is) {
            const auto & cell1 = cells[is];

            if (cell1.is_empty() || ids[is] != n_kv) {
                continue;
            }

            // non-empty cell which is not yet moved
            nf++;

            if (nf == nh) {
                break;
            }
        }

        // this can only happen if `n_used` is not accurate, which would be a bug
        GGML_ASSERT(nf == nh && "KV defrag bug: nf != nh");

        nf = 0;

        uint32_t i1 = is;

        // are we moving a continuous block of memory?
        bool cont = false;

        // go back and move the nf cells to the hole
        for (; i1 < n_kv; ++i1) {
            auto & cell1 = cells[i1];

            if (cell1.is_empty() || ids[i1] != n_kv) {
                cont = false;
                continue;
            }

            // this cell goes to (i0 + nf)
            ids[i1] = i0 + nf;

            // move the cell meta data
            cells[i0 + nf] = cell1;

            // clear the old cell and move the head there
            cell1 = llama_kv_cell();
            head = n_used;

            if (!cont) {
                defrag_info.moves.push_back({i1, i0 + nf, 1});
                cont = true;
            } else {
                defrag_info.moves.back().len++;
            }

            nf++;

            if (nf == nh) {
                break;
            }
        }

        //LLAMA_LOG_INFO("(tmp log) KV defrag: move [%u, %u) to [%u, %u)\n", is, i1 + 1, i0, i0 + nh);

        i0 += nh - 1;
    }

    if (defrag_info.moves.size() == 0) {
        return false;
    }

    // LLAMA_LOG_DEBUG("(tmp log) KV defrag cell moves: %u\n", n_moves);

    return true;
}

void llama_kv_cache_unified::state_write(llama_io_write_i & io, llama_seq_id seq_id) const {
    std::vector<std::pair<uint32_t, uint32_t>> cell_ranges; // ranges, from inclusive, to exclusive
    uint32_t cell_count = 0;

    // Count the number of cells with the specified seq_id
    // Find all the ranges of cells with this seq id (or all, when -1)
    uint32_t cell_range_begin = size;
    for (uint32_t i = 0; i < size; ++i) {
        const auto & cell = cells[i];
        if ((seq_id == -1 && !cell.is_empty()) || cell.has_seq_id(seq_id)) {
            ++cell_count;
            if (cell_range_begin == size) {
                cell_range_begin = i;
            }
        } else {
            if (cell_range_begin != size) {
                cell_ranges.emplace_back(cell_range_begin, i);
                cell_range_begin = size;
            }
        }
    }
    if (cell_range_begin != size) {
        cell_ranges.emplace_back(cell_range_begin, size);
    }

    // DEBUG CHECK: Sum of cell counts in ranges should equal the total cell count
    uint32_t cell_count_check = 0;
    for (const auto & range : cell_ranges) {
        cell_count_check += range.second - range.first;
    }
    GGML_ASSERT(cell_count == cell_count_check);

    io.write(&cell_count, sizeof(cell_count));

    state_write_meta(io, cell_ranges, seq_id);
    state_write_data(io, cell_ranges);
}

void llama_kv_cache_unified::state_read(llama_io_read_i & io, llama_seq_id seq_id) {
    uint32_t cell_count;
    io.read_to(&cell_count, sizeof(cell_count));

    bool res = true;
    res = res && state_read_meta(io, cell_count, seq_id);
    res = res && state_read_data(io, cell_count);

    if (!res) {
        if (seq_id == -1) {
            clear();
        } else {
            seq_rm(seq_id, -1, -1);
        }
        throw std::runtime_error("failed to restore kv cache");
    }
}

void llama_kv_cache_unified::state_write_meta(llama_io_write_i & io, const std::vector<std::pair<uint32_t, uint32_t>> & cell_ranges, llama_seq_id seq_id) const {
    for (const auto & range : cell_ranges) {
        for (uint32_t i = range.first; i < range.second; ++i) {
            const auto & cell = cells[i];
            const llama_pos pos      = cell.pos;
            const uint32_t  n_seq_id = seq_id == -1 ? cell.seq_id.size() : 0;

            io.write(&pos,      sizeof(pos));
            io.write(&n_seq_id, sizeof(n_seq_id));

            if (n_seq_id) {
                for (auto seq_id : cell.seq_id) {
                    io.write(&seq_id, sizeof(seq_id));
                }
            }
        }
    }
}

void llama_kv_cache_unified::state_write_data(llama_io_write_i & io, const std::vector<std::pair<uint32_t, uint32_t>> & cell_ranges) const {
    const uint32_t v_trans = this->v_trans ? 1 : 0;
    const uint32_t n_layer = hparams.n_layer;

    io.write(&v_trans, sizeof(v_trans));
    io.write(&n_layer, sizeof(n_layer));

    std::vector<uint8_t> tmp_buf;

    // Iterate and write all the keys first, each row is a cell
    // Get whole range at a time
    for (uint32_t il = 0; il < n_layer; ++il) {
        const uint32_t n_embd_k_gqa = hparams.n_embd_k_gqa(il) + hparams.n_embd_k_s();

        // Write key type
        const int32_t k_type_i = (int32_t)k_l[il]->type;
        io.write(&k_type_i, sizeof(k_type_i));

        // Write row size of key
        const uint64_t k_size_row = ggml_row_size(k_l[il]->type, n_embd_k_gqa);
        io.write(&k_size_row, sizeof(k_size_row));

        // Read each range of cells of k_size length each into tmp_buf and write out
        for (const auto & range : cell_ranges) {
            const size_t range_size = range.second - range.first;
            const size_t buf_size = range_size * k_size_row;
            io.write_tensor(k_l[il], range.first * k_size_row, buf_size);
        }
    }

    if (!v_trans) {
        for (uint32_t il = 0; il < n_layer; ++il) {
            const uint32_t n_embd_v_gqa = hparams.n_embd_v_gqa(il) + hparams.n_embd_v_s();

            // Write value type
            const int32_t v_type_i = (int32_t)v_l[il]->type;
            io.write(&v_type_i, sizeof(v_type_i));

            // Write row size of value
            const uint64_t v_size_row = ggml_row_size(v_l[il]->type, n_embd_v_gqa);
            io.write(&v_size_row, sizeof(v_size_row));

            // Read each range of cells of v_size length each into tmp_buf and write out
            for (const auto & range : cell_ranges) {
                const size_t range_size = range.second - range.first;
                const size_t buf_size = range_size * v_size_row;
                io.write_tensor(v_l[il], range.first * v_size_row, buf_size);
            }
        }
    } else {
        // When v is transposed, we also need the element size and get the element ranges from each row
        const uint32_t kv_size = size;
        for (uint32_t il = 0; il < n_layer; ++il) {
            const uint32_t n_embd_v_gqa = hparams.n_embd_v_gqa(il) + hparams.n_embd_v_s();

            // Write value type
            const int32_t v_type_i = (int32_t)v_l[il]->type;
            io.write(&v_type_i, sizeof(v_type_i));

            // Write element size
            const uint32_t v_size_el = ggml_type_size(v_l[il]->type);
            io.write(&v_size_el, sizeof(v_size_el));

            // Write GQA embedding size
            io.write(&n_embd_v_gqa, sizeof(n_embd_v_gqa));

            // For each row, we get the element values of each cell
            for (uint32_t j = 0; j < n_embd_v_gqa; ++j) {
                // Read each range of cells of v_size_el length each into tmp_buf and write out
                for (const auto & range : cell_ranges) {
                    const size_t range_size = range.second - range.first;
                    const size_t src_offset = (range.first + j * kv_size) * v_size_el;
                    const size_t buf_size = range_size * v_size_el;
                    io.write_tensor(v_l[il], src_offset, buf_size);
                }
            }
        }
    }
}

bool llama_kv_cache_unified::state_read_meta(llama_io_read_i & io, uint32_t cell_count, llama_seq_id dest_seq_id) {
    if (dest_seq_id != -1) {
        // single sequence

        seq_rm(dest_seq_id, -1, -1);

        llama_sbatch sbatch;
        llama_ubatch batch = sbatch.reserve_ubatch(cell_count, /* has_embd */ false);

        batch.n_tokens = cell_count;
        batch.n_seq_tokens = cell_count;
        batch.n_seqs = 1;

        for (uint32_t i = 0; i < cell_count; ++i) {
            llama_pos pos;
            uint32_t n_seq_id;

            io.read_to(&pos,      sizeof(pos));
            io.read_to(&n_seq_id, sizeof(n_seq_id));

            if (n_seq_id != 0) {
                LLAMA_LOG_ERROR("%s: invalid seq_id-agnostic kv cell\n", __func__);
                return false;
            }

            batch.pos[i] = pos;
        }
        batch.n_seq_id[0] = 1;
        batch.seq_id[0] = &dest_seq_id;
        if (!find_slot(batch)) {
            LLAMA_LOG_ERROR("%s: failed to find available cells in kv cache\n", __func__);
            return false;
        }
        commit();

        // DEBUG CHECK: kv.head should be our first cell, kv.head + cell_count - 1 should be our last cell (verify seq_id and pos values)
        // Assume that this is one contiguous block of cells
        GGML_ASSERT(head + cell_count <= size);
        GGML_ASSERT(cells[head].pos == batch.pos[0]);
        GGML_ASSERT(cells[head + cell_count - 1].pos == batch.pos[cell_count - 1]);
        GGML_ASSERT(cells[head].has_seq_id(dest_seq_id));
        GGML_ASSERT(cells[head + cell_count - 1].has_seq_id(dest_seq_id));
    } else {
        // whole KV cache restore

        if (cell_count > size) {
            LLAMA_LOG_ERROR("%s: not enough cells in kv cache\n", __func__);
            return false;
        }

        clear();

        for (uint32_t i = 0; i < cell_count; ++i) {
            llama_kv_cell & cell = cells[i];

            llama_pos pos;
            uint32_t  n_seq_id;

            io.read_to(&pos,      sizeof(pos));
            io.read_to(&n_seq_id, sizeof(n_seq_id));

            cell.pos = pos;

            for (uint32_t j = 0; j < n_seq_id; ++j) {
                llama_seq_id seq_id;
                io.read_to(&seq_id, sizeof(seq_id));

                // TODO: llama_kv_cache_unified should have a notion of max sequences
                //if (seq_id < 0 || (uint32_t) seq_id >= llama_n_seq_max(ctx)) {
                if (seq_id < 0) {
                    //LLAMA_LOG_ERROR("%s: invalid seq_id, %d is out of range [0, %u)\n", __func__, seq_id, llama_n_seq_max(ctx));
                    LLAMA_LOG_ERROR("%s: invalid seq_id, %d is out of range [0, inf)\n", __func__, seq_id);
                    return false;
                }

                cell.seq_id.insert(seq_id);

                if (recurrent) {
                    int32_t & tail = cells[seq_id].tail;
                    if (tail != -1) {
                        LLAMA_LOG_ERROR("%s: duplicate tail for seq_id %d in cell %d and %d\n", __func__, seq_id, i, tail);
                        return false;
                    }
                    tail = i;
                }
            }
        }

        head = 0;
        used = cell_count;
    }

    if (recurrent) {
        for (uint32_t i = 0; i < cell_count; ++i) {
            uint32_t cell_id = head + i;
            // make sure the recurrent states will keep their restored state
            cells[cell_id].src = cell_id;
        }
    }

    return true;
}

bool llama_kv_cache_unified::state_read_data(llama_io_read_i & io, uint32_t cell_count) {
    uint32_t v_trans;
    uint32_t n_layer;
    io.read_to(&v_trans, sizeof(v_trans));
    io.read_to(&n_layer, sizeof(n_layer));

    if (n_layer != hparams.n_layer) {
        LLAMA_LOG_ERROR("%s: mismatched layer count (%u instead of %u)\n", __func__, n_layer, hparams.n_layer);
        return false;
    }
    if (cell_count > size) {
        LLAMA_LOG_ERROR("%s: not enough cells in kv cache to restore state (%u > %u)\n", __func__, cell_count, size);
        return false;
    }
    if (v_trans != (bool) v_trans) {
        LLAMA_LOG_ERROR("%s: incompatible V transposition\n", __func__);
        return false;
    }

    // For each layer, read the keys for each cell, one row is one cell, read as one contiguous block
    for (uint32_t il = 0; il < n_layer; ++il) {
        const uint32_t n_embd_k_gqa = hparams.n_embd_k_gqa(il) + hparams.n_embd_k_s();

        // Read type of key
        int32_t k_type_i_ref;
        io.read_to(&k_type_i_ref, sizeof(k_type_i_ref));
        const int32_t k_type_i = (int32_t) k_l[il]->type;
        if (k_type_i != k_type_i_ref) {
            LLAMA_LOG_ERROR("%s: mismatched key type (%d != %d, layer %d)\n", __func__, k_type_i, k_type_i_ref, il);
            return false;
        }

        // Read row size of key
        uint64_t k_size_row_ref;
        io.read_to(&k_size_row_ref, sizeof(k_size_row_ref));
        const size_t k_size_row = ggml_row_size(k_l[il]->type, n_embd_k_gqa);
        if (k_size_row != k_size_row_ref) {
            LLAMA_LOG_ERROR("%s: mismatched key row size (%zu != %zu, layer %d)\n", __func__, k_size_row, (size_t) k_size_row_ref, il);
            return false;
        }

        if (cell_count) {
            // Read and set the keys for the whole cell range
            ggml_backend_tensor_set(k_l[il], io.read(cell_count * k_size_row), head * k_size_row, cell_count * k_size_row);
        }
    }

    if (!v_trans) {
        for (uint32_t il = 0; il < n_layer; ++il) {
            const uint32_t n_embd_v_gqa = hparams.n_embd_v_gqa(il) + hparams.n_embd_v_s();

            // Read type of value
            int32_t v_type_i_ref;
            io.read_to(&v_type_i_ref, sizeof(v_type_i_ref));
            const int32_t v_type_i = (int32_t)v_l[il]->type;
            if (v_type_i != v_type_i_ref) {
                LLAMA_LOG_ERROR("%s: mismatched value type (%d != %d, layer %d)\n", __func__, v_type_i, v_type_i_ref, il);
                return false;
            }

            // Read row size of value
            uint64_t v_size_row_ref;
            io.read_to(&v_size_row_ref, sizeof(v_size_row_ref));
            const size_t v_size_row = ggml_row_size(v_l[il]->type, n_embd_v_gqa);
            if (v_size_row != v_size_row_ref) {
                LLAMA_LOG_ERROR("%s: mismatched value row size (%zu != %zu, layer %d)\n", __func__, v_size_row, (size_t) v_size_row_ref, il);
                return false;
            }

            if (cell_count) {
                // Read and set the values for the whole cell range
                ggml_backend_tensor_set(v_l[il], io.read(cell_count * v_size_row), head * v_size_row, cell_count * v_size_row);
            }
        }
    } else {
        // For each layer, read the values for each cell (transposed)
        for (uint32_t il = 0; il < n_layer; ++il) {
            const uint32_t n_embd_v_gqa = hparams.n_embd_v_gqa(il) + hparams.n_embd_v_s();

            // Read type of value
            int32_t v_type_i_ref;
            io.read_to(&v_type_i_ref, sizeof(v_type_i_ref));
            const int32_t v_type_i = (int32_t)v_l[il]->type;
            if (v_type_i != v_type_i_ref) {
                LLAMA_LOG_ERROR("%s: mismatched value type (%d != %d, layer %d)\n", __func__, v_type_i, v_type_i_ref, il);
                return false;
            }

            // Read element size of value
            uint32_t v_size_el_ref;
            io.read_to(&v_size_el_ref, sizeof(v_size_el_ref));
            const size_t v_size_el = ggml_type_size(v_l[il]->type);
            if (v_size_el != v_size_el_ref) {
                LLAMA_LOG_ERROR("%s: mismatched value element size (%zu != %zu, layer %d)\n", __func__, v_size_el, (size_t) v_size_el_ref, il);
                return false;
            }

            // Read GQA embedding size
            uint32_t n_embd_v_gqa_ref;
            io.read_to(&n_embd_v_gqa_ref, sizeof(n_embd_v_gqa_ref));
            if (n_embd_v_gqa != n_embd_v_gqa_ref) {
                LLAMA_LOG_ERROR("%s: mismatched GQA embedding size (%u != %u, layer %d)\n", __func__, n_embd_v_gqa, n_embd_v_gqa_ref, il);
                return false;
            }

            if (cell_count) {
                // For each row in the transposed matrix, read the values for the whole cell range
                for (uint32_t j = 0; j < n_embd_v_gqa; ++j) {
                    const size_t dst_offset = (head + j * size) * v_size_el;
                    ggml_backend_tensor_set(v_l[il], io.read(cell_count * v_size_el), dst_offset, cell_count * v_size_el);
                }
            }
        }
    }

    return true;
}

//
// kv cache view
//

llama_kv_cache_view llama_kv_cache_view_init(const llama_kv_cache & kv, int32_t n_seq_max) {
    llama_kv_cache_view result = {
        /*.n_cells            = */ 0,
        /*.n_seq_max          = */ n_seq_max,
        /*.token_count        = */ 0,
        /*.used_cells         = */ kv.get_used_cells(),
        /*.max_contiguous     = */ 0,
        /*.max_contiguous_idx = */ -1,
        /*.cells              = */ nullptr,
        /*.cells_sequences    = */ nullptr,
    };

    return result;
}

void llama_kv_cache_view_free(llama_kv_cache_view * view) {
    if (view->cells != nullptr) {
        free(view->cells);
        view->cells = nullptr;
    }
    if (view->cells_sequences != nullptr) {
        free(view->cells_sequences);
        view->cells_sequences = nullptr;
    }
}

void llama_kv_cache_view_update(llama_kv_cache_view * view, const llama_kv_cache * kv) {
    // TODO: rework this in the future, for now quick hack
    const llama_kv_cache_unified * kvu = dynamic_cast<const llama_kv_cache_unified *>(kv);
    if (kvu == nullptr) {
        LLAMA_LOG_ERROR("%s: the kv_cache_view currently works only with llama_kv_cache_unified\n", __func__);
        return;
    }

    if (uint32_t(view->n_cells) < kvu->size || view->cells == nullptr) {
        view->n_cells = int32_t(kvu->size);
        void * p = realloc(view->cells, sizeof(llama_kv_cache_view_cell) * view->n_cells);
        GGML_ASSERT(p != nullptr && "Failed to alloc kv_cache_view cells");
        view->cells = (llama_kv_cache_view_cell *)p;
        p = realloc(view->cells_sequences, sizeof(llama_seq_id) * view->n_seq_max * view->n_cells);
        GGML_ASSERT(p != nullptr && "Failed to alloc kv_cache_view cells sequences");
        view->cells_sequences = (llama_seq_id *)p;
    }

    const std::vector<llama_kv_cell> & kv_cells = kvu->cells;
    llama_kv_cache_view_cell * c_curr = view->cells;
    llama_seq_id * cs_curr = view->cells_sequences;
    int32_t used_cells = 0;
    int32_t token_count = 0;
    int32_t curr_contig_idx = -1;
    uint32_t max_contig = 0;
    int32_t max_contig_idx = -1;

    for (int32_t i = 0; i < int32_t(kvu->size); i++, c_curr++, cs_curr += view->n_seq_max) {
        const size_t curr_size = kv_cells[i].seq_id.size();
        token_count += curr_size;
        c_curr->pos = kv_cells[i].pos + kv_cells[i].delta;

        if (curr_size > 0) {
            if (curr_contig_idx >= 0 && uint32_t(i - curr_contig_idx) > max_contig) {
                max_contig = i - curr_contig_idx;
                max_contig_idx = curr_contig_idx;
            }
            curr_contig_idx = -1;
        } else if (curr_contig_idx < 0) {
            curr_contig_idx = i;
        }

        int seq_idx = 0;
        for (const llama_seq_id it : kv_cells[i].seq_id) {
            if (seq_idx >= view->n_seq_max) {
                break;
            }
            cs_curr[seq_idx] = it;
            seq_idx++;
        }
        if (seq_idx != 0) {
            used_cells++;
        }
        for (; seq_idx < view->n_seq_max; seq_idx++) {
            cs_curr[seq_idx] = -1;
        }
    }
    if (curr_contig_idx >= 0 && kv_cells.size() - curr_contig_idx > max_contig) {
        max_contig_idx = curr_contig_idx;
        max_contig = kv_cells.size() - curr_contig_idx;
    }
    view->max_contiguous = max_contig;
    view->max_contiguous_idx = max_contig_idx;
    view->token_count = token_count;
    view->used_cells = used_cells;
    if (uint32_t(used_cells) != kvu->used) {
        LLAMA_LOG_ERROR("%s: used cells mismatch. kv_cache says %d but we calculated %d\n",
            __func__, kvu->used, used_cells);
    }
}
