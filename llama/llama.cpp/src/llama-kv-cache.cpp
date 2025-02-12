#include "llama-kv-cache.h"

#include "llama-impl.h"
#include "llama-batch.h"
#include "llama-cparams.h"
#include "llama-model.h"

#include <algorithm>
#include <limits>
#include <map>

static const llama_kv_cache_slot_info llama_kv_cache_slot_info_failed{false};

uint32_t llama_kv_cache_get_padding(const struct llama_cparams & cparams) {
    // the FA kernels require padding to avoid extra runtime boundary checks
    return cparams.flash_attn ? 256u : 32u;
}

bool llama_kv_cache_init(
             struct llama_kv_cache & cache,
                 const llama_model & model,
               const llama_cparams & cparams,
                         ggml_type   type_k,
                         ggml_type   type_v,
                          uint32_t   kv_size,
                              bool   offload) {
    const struct llama_hparams & hparams = model.hparams;

    const int32_t n_layer = hparams.n_layer;

    cache.has_shift = false;

    cache.recurrent = llama_model_is_recurrent(&model);
    cache.v_trans   = !cache.recurrent && !cparams.flash_attn;
    cache.can_shift = !cache.recurrent && model.arch != LLM_ARCH_DEEPSEEK2; // not supported due to MLA

    LLAMA_LOG_INFO("%s: kv_size = %d, offload = %d, type_k = '%s', type_v = '%s', n_layer = %d, can_shift = %d\n",
            __func__, kv_size, offload, ggml_type_name(type_k), ggml_type_name(type_v), n_layer, cache.can_shift);

    cache.head = 0;
    cache.size = kv_size;
    cache.used = 0;

    cache.type_k = type_k;
    cache.type_v = type_v;

    cache.cells.clear();
    cache.cells.resize(kv_size);

    // create a context for each buffer type
    std::map<ggml_backend_buffer_type_t, ggml_context *> ctx_map;
    auto ctx_for_buft = [&](ggml_backend_buffer_type_t buft) -> ggml_context * {
        auto it = ctx_map.find(buft);
        if (it == ctx_map.end()) {
            struct ggml_init_params params = {
                /*.mem_size   =*/ size_t(2u*n_layer*ggml_tensor_overhead()),
                /*.mem_buffer =*/ NULL,
                /*.no_alloc   =*/ true,
            };
            ggml_context * ctx = ggml_init(params);
            if (!ctx) {
                return nullptr;
            }
            ctx_map[buft] = ctx;
            cache.ctxs.emplace_back(ctx);
            return ctx;
        }
        return it->second;
    };

    cache.k_l.reserve(n_layer);
    cache.v_l.reserve(n_layer);

    for (int i = 0; i < n_layer; i++) {
        // for cross attention layers
        if (model.arch == LLM_ARCH_MLLAMA && hparams.cross_attention_layers(i)) {
            const uint32_t n_embd_k_gqa = hparams.n_embd_k_gqa(i) + hparams.n_embd_k_s();
            const llama_model::buft_list_t * buft_list;
            if (offload) {
                buft_list = model.dev_layer.at(i).buft_list;
            } else {
                buft_list = &model.cpu_buft_list;
            }
            ggml_backend_buffer_type_t buft = select_buft(*buft_list,
                [&](ggml_context * ctx) {
                    ggml_tensor * k = ggml_new_tensor_1d(ctx, type_k, n_embd_k_gqa*kv_size);
                    if (hparams.rope_type == LLAMA_ROPE_TYPE_NONE) {
                        return k;
                    }
                    ggml_tensor * p = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
                    return ggml_rope(ctx, k, p, hparams.n_rot, hparams.rope_type);
                });
            ggml_context * ctx = ctx_for_buft(buft);

            if (!ctx) {
                LLAMA_LOG_ERROR("%s: failed to create ggml context for kv cache\n", __func__);
                return false;
            }
            ggml_tensor * k = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hparams.n_embd_head_k, 6404, hparams.n_head_kv(i));
            ggml_tensor * v = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hparams.n_embd_head_v, 6404, hparams.n_head_kv(i));
            ggml_format_name(k, "cache_k_l%d", i);
            ggml_format_name(v, "cache_v_l%d", i);
            cache.k_l.push_back(k);
            cache.v_l.push_back(v);
            continue;
        }

        const uint32_t n_embd_k_gqa = hparams.n_embd_k_gqa(i) + hparams.n_embd_k_s();
        const uint32_t n_embd_v_gqa = hparams.n_embd_v_gqa(i) + hparams.n_embd_v_s();

        LLAMA_LOG_DEBUG("%s: layer %d: n_embd_k_gqa = %d, n_embd_v_gqa = %d\n", __func__, i, n_embd_k_gqa, n_embd_v_gqa);

        ggml_backend_buffer_type_t buft;
        if (offload) {
            auto * dev = model.dev_layer.at(i).dev;
            buft = ggml_backend_dev_buffer_type(dev);
        } else {
            buft = ggml_backend_cpu_buffer_type();
        }
        ggml_context * ctx = ctx_for_buft(buft);

        if (!ctx) {
            LLAMA_LOG_ERROR("%s: failed to create ggml context for kv cache\n", __func__);
            return false;
        }

        ggml_tensor * k = ggml_new_tensor_1d(ctx, type_k, n_embd_k_gqa*kv_size);
        ggml_tensor * v = ggml_new_tensor_1d(ctx, type_v, n_embd_v_gqa*kv_size);
        ggml_format_name(k, "cache_k_l%d", i);
        ggml_format_name(v, "cache_v_l%d", i);
        cache.k_l.push_back(k);
        cache.v_l.push_back(v);
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
        cache.bufs.emplace_back(buf);
    }

    return true;
}

struct llama_kv_cache_slot_info llama_kv_cache_find_slot(
           struct llama_kv_cache & cache,
       const struct llama_ubatch & batch) {
    const uint32_t n_tokens = batch.n_tokens;
    const uint32_t n_seqs   = batch.n_seqs;
    const uint32_t n_seq_tokens = batch.n_seq_tokens;

    if (cache.recurrent) {
        // For recurrent state architectures (like Mamba or RWKV),
        // each cache cell can store the state for a whole sequence.
        // A slot should be always be contiguous.

        // can only process batches with an equal number of new tokens in each sequence
        GGML_ASSERT(batch.equal_seqs);

        int32_t min = cache.size - 1;
        int32_t max = 0;

        // everything should fit if all seq_ids are smaller than the max
        for (uint32_t s = 0; s < n_seqs; ++s) {
            const uint32_t n_seq_id = batch.n_seq_id[s];
            for (uint32_t j = 0; j < n_seq_id; ++j) {
                const llama_seq_id seq_id = batch.seq_id[s][j];

                if (seq_id < 0 || (uint32_t) seq_id >= cache.size) {
                    // too big seq_id
                    // TODO: would it be possible to resize the cache instead?
                    LLAMA_LOG_ERROR("%s: seq_id=%d >= n_seq_max=%d Try using a bigger --parallel value\n", __func__, seq_id, cache.size);
                    return llama_kv_cache_slot_info_failed;
                }
                if (j > 0) {
                    llama_kv_cell & seq = cache.cells[seq_id];
                    if (seq.tail >= 0) {
                        llama_kv_cell & cell = cache.cells[seq.tail];
                        // clear cells from seq_ids that become shared
                        // (should not normally happen, but let's handle it anyway)
                        cell.seq_id.erase(seq_id);
                        seq.tail = -1;
                        if (cell.seq_id.empty()) {
                            cell.pos = -1;
                            cell.src = -1;
                            cache.used -= 1;
                        }
                    }
                }
            }
        }

#ifndef NDEBUG
        {
            std::vector<int32_t> tails_verif;
            tails_verif.assign(cache.size, -1);
            for (uint32_t i = 0; i < cache.size; ++i) {
                llama_kv_cell & cell = cache.cells[i];
                for (llama_seq_id seq_id : cell.seq_id) {
                    if (tails_verif[seq_id] != -1) {
                        LLAMA_LOG_ERROR("%s: duplicate tail for seq_id %d in cell %d and %d\n", __func__, seq_id, i, tails_verif[seq_id]);
                    }
                    tails_verif[seq_id] = i;
                }
            }
            for (uint32_t i = 0; i < cache.size; ++i) {
                if (tails_verif[i] != cache.cells[i].tail) {
                    LLAMA_LOG_ERROR("%s: wrong tail for seq_id %d, (%d instead of %d)\n", __func__, i, cache.cells[i].tail, tails_verif[i]);
                }
            }
        }
#endif

        // find next empty cell
        uint32_t next_empty_cell = cache.head;

        for (uint32_t i = 0; i < cache.size; ++i) {
            if (next_empty_cell >= cache.size) { next_empty_cell -= cache.size; }
            llama_kv_cell & cell = cache.cells[next_empty_cell];
            if (cell.is_empty()) { break; }
            next_empty_cell += 1;
        }

        // find usable cell range
        for (uint32_t s = 0; s < n_seqs; ++s) {
            const llama_seq_id seq_id = batch.seq_id[s][0];
            llama_kv_cell & seq_meta = cache.cells[seq_id];
            bool has_cell = false;
            if (seq_meta.tail >= 0) {
                llama_kv_cell & cell = cache.cells[seq_meta.tail];
                GGML_ASSERT(cell.has_seq_id(seq_id));
                // does this seq_id "own" the cell?
                if (cell.seq_id.size() == 1) { has_cell = true; }
            }
            if (!has_cell) {
                llama_kv_cell & empty_cell = cache.cells[next_empty_cell];
                GGML_ASSERT(empty_cell.is_empty());
                // copy old tail into the empty cell
                if (seq_meta.tail >= 0) {
                    llama_kv_cell & orig_cell = cache.cells[seq_meta.tail];
                    empty_cell.pos = orig_cell.pos;
                    empty_cell.src = orig_cell.src;
                    orig_cell.seq_id.erase(seq_id);
                    empty_cell.seq_id.insert(seq_id); // will be overwritten
                }
                seq_meta.tail = next_empty_cell;
                // find next empty cell
                if (s + 1 < n_seqs) {
                    next_empty_cell += 1;
                    for (uint32_t i = 0; i < cache.size; ++i) {
                        if (next_empty_cell >= cache.size) { next_empty_cell -= cache.size; }
                        llama_kv_cell & cell = cache.cells[next_empty_cell];
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
            int32_t src_id = cache.cells[batch.seq_id[s][0]].tail;
            if (dst_id != src_id) {
                llama_kv_cell & dst_cell = cache.cells[dst_id];
                llama_kv_cell & src_cell = cache.cells[src_id];

                std::swap(dst_cell.pos, src_cell.pos);
                std::swap(dst_cell.src, src_cell.src);
                std::swap(dst_cell.seq_id, src_cell.seq_id);

                // swap tails (assuming they NEVER overlap)
                for (const llama_seq_id seq_id : src_cell.seq_id) {
                    cache.cells[seq_id].tail = src_id;
                }
                for (const llama_seq_id seq_id : dst_cell.seq_id) {
                    cache.cells[seq_id].tail = dst_id;
                }
            }
        }

        // update the pos of the used seqs
        for (uint32_t s = 0; s < n_seqs; ++s) {
            const llama_pos last_pos = batch.pos[n_seq_tokens * s + n_seq_tokens - 1];
            int32_t cell_id = s + min;
            llama_kv_cell & cell = cache.cells[cell_id];

            if (cell.pos >= 0 && last_pos != cell.pos + (llama_pos) n_seq_tokens) {
                // What should happen when the pos backtracks or skips a value?
                // Clearing the state mid-batch would require special-casing which isn't done.
                LLAMA_LOG_WARN("%s: non-consecutive token position %d after %d for sequence %d with %u new tokens\n",
                    __func__, last_pos, cell.pos, batch.seq_id[s][0], n_seq_tokens);
            }
            cell.pos = last_pos;
            cell.seq_id.clear();
            for (int32_t j = 0; j < batch.n_seq_id[s]; ++j) {
                const llama_seq_id seq_id = batch.seq_id[s][j];
                cell.seq_id.insert(seq_id);
                cache.cells[seq_id].tail = cell_id;
            }
        }

        // allow getting the range of used cells, from head to head + n
        cache.head = min;
        cache.n    = max - min + 1;
        cache.used = std::count_if(cache.cells.begin(), cache.cells.end(),
            [](const llama_kv_cell& cell){ return !cell.is_empty(); });

        // sanity check
        return llama_kv_cache_slot_info(cache.n >= n_seqs);
    }
    // otherwise, one cell per token.

    if (n_tokens > cache.size) {
        LLAMA_LOG_ERROR("%s: n_tokens=%d > cache.size=%d\n", __func__, n_tokens, cache.size);
        return llama_kv_cache_slot_info_failed;
    }

    uint32_t n_tested = 0;

    while (true) {
        if (cache.head + n_tokens > cache.size) {
            n_tested += cache.size - cache.head;
            cache.head = 0;
            continue;
        }

        bool found = true;
        for (uint32_t i = 0; i < n_tokens; i++) {
            if (cache.cells[cache.head + i].pos >= 0) {
                found = false;
                cache.head += i + 1;
                n_tested   += i + 1;
                break;
            }
        }

        if (found) {
            break;
        }

        if (n_tested >= cache.size) {
            //LLAMA_LOG_ERROR("%s: failed to find a slot for %d tokens\n", __func__, n_tokens);
            return llama_kv_cache_slot_info_failed;
        }
    }

    for (uint32_t s = 0; s < n_seqs; s++) {
        for (uint32_t i = 0; i < n_seq_tokens; ++i) {
            uint32_t k = s*n_seq_tokens + i;
            cache.cells[cache.head + k].pos = batch.pos[k];

            for (int32_t j = 0; j < batch.n_seq_id[s]; j++) {
                cache.cells[cache.head + k].seq_id.insert(batch.seq_id[s][j]);
            }
        }
    }

    cache.used += n_tokens;

    return llama_kv_cache_slot_info(cache.head, cache.head + n_tokens);
}

uint32_t llama_kv_cache_cell_max(const struct llama_kv_cache & cache) {
    for (uint32_t i = cache.size; i > 0; --i) {
        const llama_kv_cell & cell = cache.cells[i - 1];

        if (cell.pos >= 0 && !cell.is_empty()) {
            return i;
        }
    }

    return 0;
}

void llama_kv_cache_clear(struct llama_kv_cache & cache) {
    for (int32_t i = 0; i < (int32_t) cache.size; ++i) {
        cache.cells[i].pos = -1;
        cache.cells[i].seq_id.clear();
        cache.cells[i].src = -1;
        cache.cells[i].tail = -1;
    }
    cache.head = 0;
    cache.used = 0;

    for (auto & buf : cache.bufs) {
        ggml_backend_buffer_clear(buf.get(), 0);
    }
}

bool llama_kv_cache_seq_rm(
        struct llama_kv_cache & cache,
                 llama_seq_id   seq_id,
                    llama_pos   p0,
                    llama_pos   p1) {
    uint32_t new_head = cache.size;

    if (p0 < 0) p0 = 0;
    if (p1 < 0) p1 = std::numeric_limits<llama_pos>::max();

    // models like Mamba or RWKV can't have a state partially erased
    if (cache.recurrent) {
        if (seq_id >= (int64_t) cache.size) {
            // could be fatal
            return false;
        }
        if (0 <= seq_id) {
            int32_t & tail_id = cache.cells[seq_id].tail;
            if (tail_id >= 0) {
                const llama_kv_cell & cell = cache.cells[tail_id];
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
    }

    for (uint32_t i = 0; i < cache.size; ++i) {
        if (cache.cells[i].pos >= p0 && cache.cells[i].pos < p1) {
            if (seq_id < 0) {
                cache.cells[i].seq_id.clear();
            } else if (cache.cells[i].has_seq_id(seq_id)) {
                cache.cells[i].seq_id.erase(seq_id);
            } else {
                continue;
            }
            if (cache.cells[i].is_empty()) {
                // keep count of the number of used cells
                if (cache.cells[i].pos >= 0) cache.used--;

                cache.cells[i].pos = -1;
                cache.cells[i].src = -1;
                if (new_head == cache.size) new_head = i;
            }
        }
    }

    // If we freed up a slot, set head to it so searching can start there.
    if (new_head != cache.size && new_head < cache.head) cache.head = new_head;

    return true;
}

void llama_kv_cache_seq_cp(
        struct llama_kv_cache & cache,
                 llama_seq_id   seq_id_src,
                 llama_seq_id   seq_id_dst,
                    llama_pos   p0,
                    llama_pos   p1) {
    if (p0 < 0) p0 = 0;
    if (p1 < 0) p1 = std::numeric_limits<llama_pos>::max();

    if (cache.recurrent) {
        if ((uint32_t) seq_id_dst < cache.size && (uint32_t) seq_id_src < cache.size) {
            llama_kv_cell & tail_src = cache.cells[seq_id_src];
            llama_kv_cell & tail_dst = cache.cells[seq_id_dst];
            if (tail_dst.tail >= 0) {
                // clear destination seq_id if it wasn't empty
                llama_kv_cell & cell_dst = cache.cells[tail_dst.tail];

                cell_dst.seq_id.erase(seq_id_dst);
                tail_dst.tail = -1;
                if (cell_dst.seq_id.empty()) {
                    cell_dst.pos = -1;
                    cell_dst.delta = -1;
                    cell_dst.src = -1;
                    cache.used -= 1;
                }
            }
            if (tail_src.tail >= 0) {
                llama_kv_cell & cell_src = cache.cells[tail_src.tail];

                cell_src.seq_id.insert(seq_id_dst);
                tail_dst.tail = tail_src.tail;
            }
        }

        return;
    }
    // otherwise, this is the KV cache of a Transformer-like model

    cache.head = 0;

    for (uint32_t i = 0; i < cache.size; ++i) {
        if (cache.cells[i].has_seq_id(seq_id_src) && cache.cells[i].pos >= p0 && cache.cells[i].pos < p1) {
            cache.cells[i].seq_id.insert(seq_id_dst);
        }
    }
}

void llama_kv_cache_seq_keep(struct llama_kv_cache & cache, llama_seq_id seq_id) {
    uint32_t new_head = cache.size;

    for (uint32_t i = 0; i < cache.size; ++i) {
        if (cache.recurrent && (llama_seq_id) i != seq_id) {
            cache.cells[i].tail = -1;
        }
        if (!cache.cells[i].has_seq_id(seq_id)) {
            if (cache.cells[i].pos >= 0) cache.used--;
            cache.cells[i].pos = -1;
            cache.cells[i].src = -1;
            cache.cells[i].seq_id.clear();
            if (new_head == cache.size) new_head = i;
        } else {
            cache.cells[i].seq_id.clear();
            cache.cells[i].seq_id.insert(seq_id);
        }
    }

    // If we freed up a slot, set head to it so searching can start there.
    if (new_head != cache.size && new_head < cache.head) cache.head = new_head;
}

void llama_kv_cache_seq_add(
        struct llama_kv_cache & cache,
                 llama_seq_id   seq_id,
                    llama_pos   p0,
                    llama_pos   p1,
                    llama_pos   delta) {
    uint32_t new_head = cache.size;

    if (p0 < 0) p0 = 0;
    if (p1 < 0) p1 = std::numeric_limits<llama_pos>::max();
    // If there is no range then return early to avoid looping over the cache.
    if (p0 == p1) return;

    if (cache.recurrent) {
        // for Mamba-like or RWKV models, only the pos needs to be shifted
        if (0 <= seq_id && seq_id < (int64_t) cache.size) {
            const int32_t tail_id = cache.cells[seq_id].tail;
            if (tail_id >= 0) {
                llama_kv_cell & cell = cache.cells[tail_id];
                if (cell.has_seq_id(seq_id) && p0 <= cell.pos && cell.pos < p1) {
                    cell.pos += delta;
                }
            }
        }
        return;
    }

    for (uint32_t i = 0; i < cache.size; ++i) {
        if (cache.cells[i].has_seq_id(seq_id) && cache.cells[i].pos >= p0 && cache.cells[i].pos < p1) {
            cache.has_shift = true;
            cache.cells[i].pos   += delta;
            cache.cells[i].delta += delta;

            if (cache.cells[i].pos < 0) {
                if (!cache.cells[i].is_empty()) {
                    cache.used--;
                }
                cache.cells[i].pos = -1;
                cache.cells[i].seq_id.clear();
                if (new_head == cache.size) {
                    new_head = i;
                }
            }
        }
    }

    // If we freed up a slot, set head to it so searching can start there.
    // Otherwise we just start the next search from the beginning.
    cache.head = new_head != cache.size ? new_head : 0;
}

void llama_kv_cache_seq_div(
        struct llama_kv_cache & cache,
                 llama_seq_id   seq_id,
                    llama_pos   p0,
                    llama_pos   p1,
                          int   d) {
    if (p0 < 0) p0 = 0;
    if (p1 < 0) p1 = std::numeric_limits<llama_pos>::max();
    // If there is no range then return early to avoid looping over the cache.
    if (p0 == p1) return;

    if (cache.recurrent) {
        // for Mamba-like or RWKV models, only the pos needs to be changed
        if (0 <= seq_id && seq_id < (int64_t) cache.size) {
            const int32_t tail_id = cache.cells[seq_id].tail;
            if (tail_id >= 0) {
                llama_kv_cell & cell = cache.cells[tail_id];
                if (cell.has_seq_id(seq_id) && p0 <= cell.pos && cell.pos < p1) {
                    cell.pos /= d;
                }
            }
        }
        return;
    }

    for (uint32_t i = 0; i < cache.size; ++i) {
        if (cache.cells[i].has_seq_id(seq_id) && cache.cells[i].pos >= p0 && cache.cells[i].pos < p1) {
            cache.has_shift = true;

            {
                llama_pos p_old = cache.cells[i].pos;
                cache.cells[i].pos   /= d;
                cache.cells[i].delta += cache.cells[i].pos - p_old;
            }
        }
    }
}

llama_pos llama_kv_cache_seq_pos_max(struct llama_kv_cache & cache, llama_seq_id seq_id) {
    llama_pos result = 0;

    for (uint32_t i = 0; i < cache.size; ++i) {
        if (cache.cells[i].has_seq_id(seq_id)) {
            result = std::max(result, cache.cells[i].pos);
        }
    }

    return result;
}

void llama_kv_cache_defrag(struct llama_kv_cache & cache) {
    if (!cache.recurrent) {
        cache.do_defrag = true;
    }
}

int32_t llama_get_kv_cache_token_count(const struct llama_kv_cache & kv) {
    int result = 0;

    for (uint32_t i = 0; i < kv.size; i++) {
        result += kv.cells[i].seq_id.size();
    }

    return result;
}

int32_t llama_get_kv_cache_used_cells(const struct llama_kv_cache & kv) {
    return kv.used;
}

bool llama_kv_cache_can_shift(const struct llama_kv_cache & kv) {
    return kv.can_shift;
}

//
// kv cache view
//

struct llama_kv_cache_view llama_kv_cache_view_init(const struct llama_kv_cache & kv, int32_t n_seq_max) {
    struct llama_kv_cache_view result = {
        /*.n_cells            = */ 0,
        /*.n_seq_max          = */ n_seq_max,
        /*.token_count        = */ 0,
        /*.used_cells         = */ llama_get_kv_cache_used_cells(kv),
        /*.max_contiguous     = */ 0,
        /*.max_contiguous_idx = */ -1,
        /*.cells              = */ nullptr,
        /*.cells_sequences    = */ nullptr,
    };

    return result;
}

void llama_kv_cache_view_free(struct llama_kv_cache_view * view) {
    if (view->cells != nullptr) {
        free(view->cells);
        view->cells = nullptr;
    }
    if (view->cells_sequences != nullptr) {
        free(view->cells_sequences);
        view->cells_sequences = nullptr;
    }
}

void llama_kv_cache_view_update(struct llama_kv_cache_view * view, const struct llama_kv_cache & kv) {
    if (uint32_t(view->n_cells) < kv.size || view->cells == nullptr) {
        view->n_cells = int32_t(kv.size);
        void * p = realloc(view->cells, sizeof(struct llama_kv_cache_view_cell) * view->n_cells);
        GGML_ASSERT(p != nullptr && "Failed to alloc kv_cache_view cells");
        view->cells = (struct llama_kv_cache_view_cell *)p;
        p = realloc(view->cells_sequences, sizeof(llama_seq_id) * view->n_seq_max * view->n_cells);
        GGML_ASSERT(p != nullptr && "Failed to alloc kv_cache_view cells sequences");
        view->cells_sequences = (llama_seq_id *)p;
    }

    const std::vector<llama_kv_cell> & kv_cells = kv.cells;
    llama_kv_cache_view_cell * c_curr = view->cells;
    llama_seq_id * cs_curr = view->cells_sequences;
    int32_t used_cells = 0;
    int32_t token_count = 0;
    int32_t curr_contig_idx = -1;
    uint32_t max_contig = 0;
    int32_t max_contig_idx = -1;

    for (int32_t i = 0; i < int32_t(kv.size); i++, c_curr++, cs_curr += view->n_seq_max) {
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
    if (uint32_t(used_cells) != kv.used) {
        LLAMA_LOG_ERROR("%s: used cells mismatch. kv_cache says %d but we calculated %d\n",
            __func__, kv.used, used_cells);
    }
}
