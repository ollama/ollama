#include "llama-kv-cache-unified.h"

#include "llama-impl.h"
#include "llama-io.h"
#include "llama-model.h"
#include "llama-context.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>

//
// llama_kv_cache_unified
//

llama_kv_cache_unified::llama_kv_cache_unified(
        const llama_model &  model,
          layer_filter_cb && filter,
                ggml_type    type_k,
                ggml_type    type_v,
                     bool    v_trans,
                     bool    offload,
                 uint32_t    kv_size,
                 uint32_t    n_seq_max,
                 uint32_t    n_pad,
                 uint32_t    n_swa,
           llama_swa_type    swa_type) :
    model(model), hparams(model.hparams), v_trans(v_trans),
    n_seq_max(n_seq_max), n_pad(n_pad), n_swa(n_swa), swa_type(swa_type) {

    GGML_ASSERT(kv_size % n_pad == 0);

    // TODO: this is temporary until we support passing reuse layer filters [KV_REUSE]
    auto n_layer_cache = hparams.n_layer;
    if (model.arch == LLM_ARCH_GEMMA3N) {
        n_layer_cache = 20;
    }

    // create a context for each buffer type
    std::map<ggml_backend_buffer_type_t, ggml_context *> ctx_map;
    auto ctx_for_buft = [&](ggml_backend_buffer_type_t buft) -> ggml_context * {
        auto it = ctx_map.find(buft);
        if (it == ctx_map.end()) {
            ggml_init_params params = {
                /*.mem_size   =*/ size_t(2u*n_layer_cache*ggml_tensor_overhead()),
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

    head = 0;

    cells.resize(kv_size);

    for (uint32_t il = 0; il < n_layer_cache; il++) {
        if (filter && !filter(il)) {
            LLAMA_LOG_DEBUG("%s: layer %3d: skipped\n", __func__, il);
            continue;
        }

        const uint32_t n_embd_k_gqa = hparams.n_embd_k_gqa(il);
        const uint32_t n_embd_v_gqa = hparams.n_embd_v_gqa(il);

        const char * dev_name = "CPU";

        ggml_backend_buffer_type_t buft = ggml_backend_cpu_buffer_type();

        if (offload) {
            auto * dev = model.dev_layer(il);
            buft = ggml_backend_dev_buffer_type(dev);

            dev_name = ggml_backend_dev_name(dev);
        }

        LLAMA_LOG_DEBUG("%s: layer %3d: dev = %s\n", __func__, il, dev_name);

        ggml_context * ctx = ctx_for_buft(buft);
        if (!ctx) {
            throw std::runtime_error("failed to create ggml context for kv cache");
        }

        ggml_tensor * k;
        ggml_tensor * v;

        k = ggml_new_tensor_2d(ctx, type_k, n_embd_k_gqa, kv_size);
        v = ggml_new_tensor_2d(ctx, type_v, n_embd_v_gqa, kv_size);

        ggml_format_name(k, "cache_k_l%d", il);
        ggml_format_name(v, "cache_v_l%d", il);

        map_layer_ids[il] = layers.size();
        layers.push_back({ il, k, v });
    }

    // TODO: this is temporary until we support passing reuse layer filters [KV_REUSE]
    if (model.arch == LLM_ARCH_GEMMA3N) {
        LLAMA_LOG_DEBUG("%s: GEMMA3N: reuse layers [%d, %d]\n", __func__, n_layer_cache, hparams.n_layer - 1);

        for (uint32_t il = n_layer_cache; il < hparams.n_layer; il++) {
            if (filter && !filter(il)) {
                LLAMA_LOG_DEBUG("%s: layer %3d: skipped\n", __func__, il);
                continue;
            }

            const bool     is_swa   = hparams.is_swa(il);
            const uint32_t il_reuse = n_layer_cache - (is_swa ? 2 : 1);

            GGML_ASSERT(map_layer_ids.find(il_reuse) != map_layer_ids.end());
            map_layer_ids[il] = map_layer_ids[il_reuse];

            LLAMA_LOG_DEBUG("%s: layer %3d: reuse layer %d, isw = %d\n", __func__, il, il_reuse, is_swa);
        }
    }

    // allocate tensors and initialize the buffers to avoid NaNs in the padding
    for (auto it : ctx_map) {
        auto * buft = it.first;
        auto * ctx  = it.second;

        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
        if (!buf) {
            throw std::runtime_error("failed to allocate buffer for kv cache");
        }

        LLAMA_LOG_INFO("%s: %10s KV buffer size = %8.2f MiB\n", __func__, ggml_backend_buffer_name(buf), ggml_backend_buffer_get_size(buf)/1024.0/1024.0);

        ggml_backend_buffer_clear(buf, 0);
        bufs.emplace_back(buf);
    }

    {
        const size_t memory_size_k = size_k_bytes();
        const size_t memory_size_v = size_v_bytes();

        LLAMA_LOG_INFO("%s: size = %7.2f MiB (%6u cells, %3d layers, %2u seqs), K (%s): %7.2f MiB, V (%s): %7.2f MiB\n", __func__,
                (float)(memory_size_k + memory_size_v) / (1024.0f * 1024.0f), kv_size, (int) layers.size(), n_seq_max,
                ggml_type_name(type_k), (float)memory_size_k / (1024.0f * 1024.0f),
                ggml_type_name(type_v), (float)memory_size_v / (1024.0f * 1024.0f));
    }

    const char * LLAMA_KV_CACHE_DEBUG = getenv("LLAMA_KV_CACHE_DEBUG");
    debug = LLAMA_KV_CACHE_DEBUG ? atoi(LLAMA_KV_CACHE_DEBUG) : 0;

    const char * LLAMA_SET_ROWS = getenv("LLAMA_SET_ROWS");
    supports_set_rows = LLAMA_SET_ROWS ? atoi(LLAMA_SET_ROWS) : 0;

    if (!supports_set_rows) {
        LLAMA_LOG_WARN("%s: LLAMA_SET_ROWS=0, using old ggml_cpy() method for backwards compatibility\n", __func__);
    }
}

void llama_kv_cache_unified::clear(bool data) {
    cells.reset();

    head = 0;

    if (data) {
        for (auto & buf : bufs) {
            ggml_backend_buffer_clear(buf.get(), 0);
        }
    }
}

bool llama_kv_cache_unified::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    uint32_t new_head = cells.size();

    if (p0 < 0) {
        p0 = 0;
    }

    if (p1 < 0) {
        p1 = std::numeric_limits<llama_pos>::max();
    }

    if (seq_id >= 0) {
        for (uint32_t i = 0; i < cells.size(); ++i) {
            if (!cells.pos_in(i, p0, p1)) {
                continue;
            }

            if (cells.seq_has(i, seq_id) && cells.seq_rm(i, seq_id)) {
                if (new_head == cells.size()) {
                    new_head = i;
                }
            }
        }
    } else {
        // match any sequence
        for (uint32_t i = 0; i < cells.size(); ++i) {
            if (!cells.pos_in(i, p0, p1)) {
                continue;
            }

            cells.rm(i);

            if (new_head == cells.size()) {
                new_head = i;
            }
        }
    }

    // If we freed up a slot, set head to it so searching can start there.
    if (new_head != cells.size() && new_head < head) {
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

    for (uint32_t i = 0; i < cells.size(); ++i) {
        if (!cells.pos_in(i, p0, p1)) {
            continue;
        }

        if (cells.seq_has(i, seq_id_src)) {
            cells.seq_add(i, seq_id_dst);
        }
    }
}

void llama_kv_cache_unified::seq_keep(llama_seq_id seq_id) {
    uint32_t new_head = cells.size();

    for (uint32_t i = 0; i < cells.size(); ++i) {
        if (cells.seq_keep(i, seq_id)) {
            if (new_head == cells.size()) {
                new_head = i;
            }
        }
    }

    // If we freed up a slot, set head to it so searching can start there.
    if (new_head != cells.size() && new_head < head) {
        head = new_head;
    }
}

void llama_kv_cache_unified::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) {
    if (shift == 0) {
        return;
    }

    uint32_t new_head = cells.size();

    if (p0 < 0) {
        p0 = 0;
    }

    if (p1 < 0) {
        p1 = std::numeric_limits<llama_pos>::max();
    }

    // If there is no range then return early to avoid looping over all cells.
    if (p0 == p1) {
        return;
    }

    for (uint32_t i = 0; i < cells.size(); ++i) {
        if (!cells.pos_in(i, p0, p1)) {
            continue;
        }

        if (cells.seq_has(i, seq_id)) {
            if (cells.pos_add(i, shift)) {
                if (new_head == cells.size()) {
                    new_head = i;
                }
            }
        }
    }

    // If we freed up a slot, set head to it so searching can start there.
    // Otherwise we just start the next search from the beginning.
    head = new_head != cells.size() ? new_head : 0;
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

    for (uint32_t i = 0; i < cells.size(); ++i) {
        if (!cells.pos_in(i, p0, p1)) {
            continue;
        }

        if (cells.seq_has(i, seq_id)) {
            cells.pos_div(i, d);
        }
    }
}

llama_pos llama_kv_cache_unified::seq_pos_min(llama_seq_id seq_id) const {
    return cells.seq_pos_min(seq_id);
}

llama_pos llama_kv_cache_unified::seq_pos_max(llama_seq_id seq_id) const {
    return cells.seq_pos_max(seq_id);
}

llama_memory_context_ptr llama_kv_cache_unified::init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) {
    GGML_UNUSED(embd_all);

    do {
        balloc.split_reset();

        std::vector<llama_ubatch> ubatches;
        while (true) {
            auto ubatch = balloc.split_simple(n_ubatch);

            if (ubatch.n_tokens == 0) {
                break;
            }

            ubatches.push_back(std::move(ubatch)); // NOLINT
        }

        if (balloc.get_n_used() < balloc.get_n_tokens()) {
            // failed to find a suitable split
            break;
        }

        auto sinfos = prepare(ubatches);
        if (sinfos.empty()) {
            break;
        }

        return std::make_unique<llama_kv_cache_unified_context>(
                this, std::move(sinfos), std::move(ubatches));
    } while (false);

    return std::make_unique<llama_kv_cache_unified_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
}

llama_memory_context_ptr llama_kv_cache_unified::init_full() {
    return std::make_unique<llama_kv_cache_unified_context>(this);
}

llama_memory_context_ptr llama_kv_cache_unified::init_update(llama_context * lctx, bool optimize) {
    bool do_shift = get_has_shift();

    defrag_info dinfo;

    // see if we need to defrag
    {
        bool do_defrag = optimize;

        const auto thold = lctx->get_cparams().defrag_thold;

        if (!do_defrag && thold > 0.0f) {
            const auto n_kv = cells.used_max_p1();

            // - do not defrag small contexts (i.e. < 2048 tokens)
            // - count the padding towards the number of used tokens
            const float fragmentation = n_kv >= 2048 ? std::max(0.0f, 1.0f - (float(cells.get_used() + n_pad)/n_kv)) : 0.0f;

            if (fragmentation > thold) {
                LLAMA_LOG_DEBUG("%s: fragmentation: %.2f - requesting defrag\n", __func__, fragmentation);

                do_defrag = true;
            }
        }

        if (do_defrag) {
            dinfo = defrag_prepare(lctx->graph_max_nodes());
        }
    }

    return std::make_unique<llama_kv_cache_unified_context>(this, lctx, do_shift, std::move(dinfo));
}

llama_kv_cache_unified::slot_info_vec_t llama_kv_cache_unified::prepare(const std::vector<llama_ubatch> & ubatches) {
    llama_kv_cache_unified::slot_info_vec_t res;

    struct state {
        uint32_t head_old; // old position of the head, before placing the ubatch

        slot_info sinfo; // slot info for the ubatch

        llama_kv_cells_unified cells; // copy of the old cells, before placing the ubatch
    };

    // remember the old state of the cells so we can restore it in the end
    std::vector<state> states;

    bool success = true;

    for (const auto & ubatch : ubatches) {
        // non-continuous slots require support for ggml_set_rows()
        const bool cont = supports_set_rows ? false : true;

        // only find a suitable slot for the ubatch. don't modify the cells yet
        const auto sinfo_new = find_slot(ubatch, cont);
        if (sinfo_new.empty()) {
            success = false;
            break;
        }

        // remeber the position that we found
        res.push_back(sinfo_new);

        // store the old state of the cells in the recovery stack
        states.push_back({head, sinfo_new, cells.cp(sinfo_new.idxs)});

        // now emplace the ubatch
        apply_ubatch(sinfo_new, ubatch);
    }

    // iterate backwards and restore the cells to their original state
    for (auto it = states.rbegin(); it != states.rend(); ++it) {
        cells.set(it->sinfo.idxs, it->cells);
        head = it->head_old;
    }

    if (!success) {
        return {};
    }

    return res;
}

bool llama_kv_cache_unified::update(llama_context * lctx, bool do_shift, const defrag_info & dinfo) {
    bool updated = false;

    auto * sched = lctx->get_sched();

    if (do_shift) {
        if (!get_can_shift()) {
            GGML_ABORT("The current KV cache / model configuration does not support K-shift");
        }

        LLAMA_LOG_DEBUG("%s: applying K-shift\n", __func__);

        // apply K-shift if needed
        if (hparams.rope_type != LLAMA_ROPE_TYPE_NONE) {
            ggml_backend_sched_reset(sched);

            auto * gf = lctx->graph_init();

            auto res = build_graph_shift(lctx->get_cparams(), lctx->get_ctx_compute(), gf);
            if (!res) {
                LLAMA_LOG_ERROR("%s: failed to build graph for K-shift\n", __func__);
                return updated;
            }

            if (!ggml_backend_sched_alloc_graph(sched, gf)) {
                LLAMA_LOG_ERROR("%s: failed to allocate compute graph for K-shift\n", __func__);
                return updated;
            }

            res->set_inputs(nullptr);

            if (lctx->graph_compute(gf, false) != GGML_STATUS_SUCCESS) {
                LLAMA_LOG_ERROR("%s: failed to compute K-shift\n", __func__);
                return updated;
            }

            updated = true;
        }

        cells.reset_shift();
    }

    if (!dinfo.empty()) {
        LLAMA_LOG_DEBUG("%s: defragmenting KV cache\n", __func__);

        // apply moves:
        {
            const auto n_kv = dinfo.ids.size();

            for (uint32_t i = 0; i < n_kv; ++i) {
                assert(dinfo.ids[i] <= n_kv);

                if (dinfo.ids[i] == n_kv || dinfo.ids[i] == i) {
                    continue;
                }

                cells.mv(i, dinfo.ids[i]);
            }

            // reset the head so we can find the first free slot during the next ubatch
            head = 0;
        }

        ggml_backend_sched_reset(sched);

        auto * gf = lctx->graph_init();

        auto res = build_graph_defrag(lctx->get_cparams(), lctx->get_ctx_compute(), gf, dinfo);
        if (!res) {
            LLAMA_LOG_ERROR("%s: failed to build graph for defrag\n", __func__);
            return updated;
        }

        if (!ggml_backend_sched_alloc_graph(sched, gf)) {
            LLAMA_LOG_ERROR("%s: failed to allocate compute graph for defrag\n", __func__);
            return updated;
        }

        res->set_inputs(nullptr);

        if (lctx->graph_compute(gf, false) != GGML_STATUS_SUCCESS) {
            LLAMA_LOG_ERROR("%s: failed to compute defrag\n", __func__);
            return updated;
        }

        updated = true;
    }

    return updated;
}

llama_kv_cache_unified::slot_info llama_kv_cache_unified::find_slot(const llama_ubatch & ubatch, bool cont) const {
    const uint32_t n_tokens = ubatch.n_tokens;

    uint32_t head_cur = this->head;

    // if we have enough unused cells before the current head ->
    //   better to start searching from the beginning of the cache, hoping to fill it
    if (head_cur > cells.get_used() + 2*ubatch.n_tokens) {
        head_cur = 0;
    }

    if (n_tokens > cells.size()) {
        LLAMA_LOG_ERROR("%s: n_tokens = %d > size = %u\n", __func__, n_tokens, cells.size());
        return { };
    }

    if (debug > 0) {
        LLAMA_LOG_DEBUG("%s: n = %5d, used = %5d, head = %5d, size = %5d, n_swa = %5d\n", __func__, cells.used_max_p1(), cells.get_used(), head, get_size(), n_swa);

        if ((debug == 2 && n_swa > 0) || debug > 2) {
            std::string ss;
            for (uint32_t i = 0; i < cells.size(); ++i) {
                if (cells.is_empty(i)) {
                    ss += '.';
                } else {
                    assert(cells.seq_count(i) >= 1);

                    if (cells.seq_count(i) == 1) {
                        ss += std::to_string(cells.seq_get(i));
                    } else {
                        ss += 'M';
                    }
                }
                if (i%256 == 255) {
                    ss += " *";
                    ss += '\n';
                }
            }
            LLAMA_LOG_DEBUG("\n%s\n", ss.c_str());
        }

        if ((debug == 2 && n_swa > 0) || debug > 2) {
            std::string ss;
            for (uint32_t i = 0; i < cells.size(); ++i) {
                std::string cur;
                if (cells.is_empty(i)) {
                    cur = '.';
                } else {
                    cur = std::to_string(cells.pos_get(i));
                }
                const int n = cur.size();
                for (int j = 0; j < 5 - n; ++j) {
                    cur += ' ';
                }
                ss += cur;
                if (i%256 == 255) {
                    ss += " *";
                }
                if (i%64 == 63) {
                    ss += '\n';
                }
            }
            LLAMA_LOG_DEBUG("\n%s\n", ss.c_str());
        }

        for (int s = 0; s < LLAMA_MAX_SEQ; ++s) {
            if (cells.seq_pos_min(s) < 0) {
                continue;
            }

            LLAMA_LOG_DEBUG("%s: min[%d] = %5d, max[%d] = %5d\n", __func__, s, cells.seq_pos_min(s), s, cells.seq_pos_max(s));
        }
    }

    uint32_t n_tested = 0;

    // for continuous slots, we test that all tokens in the ubatch fit, starting from the current head
    // for non-continuous slots, we test the tokens one by one
    const uint32_t n_test = cont ? n_tokens : 1;

    slot_info res;

    auto & idxs = res.idxs;

    idxs.reserve(n_tokens);

    while (true) {
        if (head_cur + n_test > cells.size()) {
            n_tested += cells.size() - head_cur;
            head_cur = 0;
            continue;
        }

        for (uint32_t i = 0; i < n_test; i++) {
            const auto idx = head_cur;

            //const llama_pos    pos    = ubatch.pos[i];
            //const llama_seq_id seq_id = ubatch.seq_id[i][0];

            // can we use this cell? either:
            //  - the cell is empty
            //  - the cell is occupied only by one sequence:
            //    - (disabled) mask causally, if the sequence is the same as the one we are inserting
            //    - mask SWA, using current max pos for that sequence in the cache
            //                always insert in the cell with minimum pos
            bool can_use = cells.is_empty(idx);

            if (!can_use && cells.seq_count(idx) == 1) {
                const llama_pos pos_cell = cells.pos_get(idx);

                // (disabled) causal mask
                // note: it's better to purge any "future" tokens beforehand
                //if (cells.seq_has(idx, seq_id)) {
                //    can_use = pos_cell >= pos;
                //}

                if (!can_use) {
                    const llama_seq_id seq_id_cell = cells.seq_get(idx);

                    // SWA mask
                    if (is_masked_swa(pos_cell, cells.seq_pos_max(seq_id_cell) + 1)) {
                        can_use = true;
                    }
                }
            }

            head_cur++;
            n_tested++;

            if (can_use) {
                idxs.push_back(idx);
            } else {
                break;
            }
        }

        if (idxs.size() == n_tokens) {
            break;
        }

        if (cont) {
            idxs.clear();
        }

        if (n_tested >= cells.size()) {
            //LLAMA_LOG_ERROR("%s: failed to find a slot for %d tokens\n", __func__, n_tokens);
            return { };
        }
    }

    // we didn't find a suitable slot - return empty result
    if (idxs.size() < n_tokens) {
        res.clear();
    }

    return res;
}

void llama_kv_cache_unified::apply_ubatch(const slot_info & sinfo, const llama_ubatch & ubatch) {
    // keep track of the max sequence position that we would overwrite with this ubatch
    // for non-SWA cache, this would be always empty
    llama_seq_id seq_pos_max_rm[LLAMA_MAX_SEQ];
    for (int s = 0; s < LLAMA_MAX_SEQ; ++s) {
        seq_pos_max_rm[s] = -1;
    }

    assert(ubatch.n_tokens == sinfo.idxs.size());

    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        const auto idx = sinfo.idxs.at(i);

        if (!cells.is_empty(idx)) {
            assert(cells.seq_count(idx) == 1);

            const llama_seq_id seq_id = cells.seq_get(idx);
            const llama_pos    pos    = cells.pos_get(idx);

            seq_pos_max_rm[seq_id] = std::max(seq_pos_max_rm[seq_id], pos);

            cells.rm(idx);
        }

        cells.pos_set(idx, ubatch.pos[i]);

        for (int32_t s = 0; s < ubatch.n_seq_id[i]; s++) {
            cells.seq_add(idx, ubatch.seq_id[i][s]);
        }
    }

    // note: we want to preserve the invariant that all positions between [pos_min, pos_max] for each sequence
    //       will be present in the cache. so we have to purge any position which is less than those we would overwrite
    //       ref: https://github.com/ggml-org/llama.cpp/pull/13746#issuecomment-2916057092
    for (int s = 0; s < LLAMA_MAX_SEQ; ++s) {
        if (seq_pos_max_rm[s] == -1) {
            continue;
        }

        if (cells.seq_pos_min(s) <= seq_pos_max_rm[s]) {
            LLAMA_LOG_DEBUG("%s: purging positions [%d, %d] of sequence %d from KV cache\n",
                    __func__, cells.seq_pos_min(s), seq_pos_max_rm[s], s);

            seq_rm(s, cells.seq_pos_min(s), seq_pos_max_rm[s] + 1);
        }
    }

    // move the head at the end of the slot
    head = sinfo.idxs.back() + 1;
}

bool llama_kv_cache_unified::get_can_shift() const {
    return true;
}

uint32_t llama_kv_cache_unified::get_size() const {
    return cells.size();
}

bool llama_kv_cache_unified::get_has_shift() const {
    return cells.get_has_shift();
}

uint32_t llama_kv_cache_unified::get_n_kv() const {
    return std::min(cells.size(), std::max(n_pad, GGML_PAD(cells.used_max_p1(), n_pad)));
}

ggml_tensor * llama_kv_cache_unified::get_k(ggml_context * ctx, int32_t il, uint32_t n_kv) const {
    const int32_t ikv = map_layer_ids.at(il);

    auto * k = layers[ikv].k;

    return ggml_view_3d(ctx, k,
            hparams.n_embd_head_k, hparams.n_head_kv(il), n_kv,
            ggml_row_size(k->type, hparams.n_embd_head_k),
            ggml_row_size(k->type, hparams.n_embd_k_gqa(il)),
            0);
}

ggml_tensor * llama_kv_cache_unified::get_v(ggml_context * ctx, int32_t il, uint32_t n_kv) const {
    const int32_t ikv = map_layer_ids.at(il);

    auto * v = layers[ikv].v;

    if (!v_trans) {
        // note: v->nb[1] <= v->nb[2]
        return ggml_view_3d(ctx, v,
                hparams.n_embd_head_v, hparams.n_head_kv(il), n_kv,
                ggml_row_size(v->type, hparams.n_embd_head_v),    // v->nb[1]
                ggml_row_size(v->type, hparams.n_embd_v_gqa(il)), // v->nb[2]
                0);
    }

    // note: v->nb[1] > v->nb[2]
    return ggml_view_3d(ctx, v,
            n_kv, hparams.n_head_kv(il), hparams.n_embd_head_v,
            ggml_row_size(v->type, v->ne[1]*hparams.n_embd_head_v), // v->nb[1]
            ggml_row_size(v->type, v->ne[1]),                       // v->nb[2]
            0);
}

ggml_tensor * llama_kv_cache_unified::cpy_k(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * k_idxs, int32_t il, const slot_info & sinfo) const {
    const int32_t ikv = map_layer_ids.at(il);

    auto * k = layers[ikv].k;

    const int64_t n_embd_k_gqa = k->ne[0];
    const int64_t n_tokens = k_cur->ne[2];

    k_cur = ggml_reshape_2d(ctx, k_cur, k->ne[0], n_tokens);

    if (k_idxs && supports_set_rows) {
        return ggml_set_rows(ctx, k, k_cur, k_idxs);
    }

    // TODO: fallback to old ggml_cpy() method for backwards compatibility
    //       will be removed when ggml_set_rows() is adopted by all backends

    ggml_tensor * k_view = ggml_view_1d(ctx, k,
            n_tokens*n_embd_k_gqa,
            ggml_row_size(k->type, n_embd_k_gqa)*sinfo.head());

    return ggml_cpy(ctx, k_cur, k_view);
}

ggml_tensor * llama_kv_cache_unified::cpy_v(ggml_context * ctx, ggml_tensor * v_cur, ggml_tensor * v_idxs, int32_t il, const slot_info & sinfo) const {
    const int32_t ikv = map_layer_ids.at(il);

    auto * v = layers[ikv].v;

    const int64_t n_embd_v_gqa = v->ne[0];
    const int64_t n_tokens = v_cur->ne[2];

    v_cur = ggml_reshape_2d(ctx, v_cur, n_embd_v_gqa, n_tokens);

    if (v_idxs && supports_set_rows) {
        if (!v_trans) {
            return ggml_set_rows(ctx, v, v_cur, v_idxs);
        }

        // the row becomes a single element
        ggml_tensor * v_view = ggml_reshape_3d(ctx, v, 1, v->ne[1], v->ne[0]);

        // note: the V cache is transposed when not using flash attention
        v_cur = ggml_permute(ctx, ggml_reshape_3d(ctx, v_cur, v_cur->ne[0], 1, v_cur->ne[1]), 2, 0, 1, 3);

        // note: we can be more explicit here at the cost of extra cont
        //       however, above we take advantage that a row of single element is always continuous regardless of the row stride
        //v_cur = ggml_transpose(ctx, v_cur);
        //v_cur = ggml_cont_3d(ctx, v_cur, 1, v_cur->ne[0], v_cur->ne[1]);

        // we broadcast the KV indices n_embd_v_gqa times
        // v      [1,        n_kv,     n_embd_v_gqa]
        // v_cur  [1,        n_tokens, n_embd_v_gqa]
        // v_idxs [n_tokens, 1,        1]
        return ggml_set_rows(ctx, v_view, v_cur, v_idxs);
    }

    // TODO: fallback to old ggml_cpy() method for backwards compatibility
    //       will be removed when ggml_set_rows() is adopted by all backends

    ggml_tensor * v_view = nullptr;

    if (!v_trans) {
        v_view = ggml_view_1d(ctx, v,
                n_tokens*n_embd_v_gqa,
                ggml_row_size(v->type, n_embd_v_gqa)*sinfo.head());
    } else {
        v_cur = ggml_transpose(ctx, v_cur);

        v_view = ggml_view_2d(ctx, v, n_tokens, n_embd_v_gqa,
                (v->ne[1]    )*ggml_element_size(v),
                (sinfo.head())*ggml_element_size(v));
    }

    return ggml_cpy(ctx, v_cur, v_view);
}

ggml_tensor * llama_kv_cache_unified::build_input_k_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const {
    const uint32_t n_tokens = ubatch.n_tokens;

    ggml_tensor * k_idxs = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n_tokens);

    ggml_set_input(k_idxs);

    return k_idxs;
}

ggml_tensor * llama_kv_cache_unified::build_input_v_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const {
    const uint32_t n_tokens = ubatch.n_tokens;

    ggml_tensor * v_idxs = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n_tokens);

    ggml_set_input(v_idxs);

    return v_idxs;
}

void llama_kv_cache_unified::set_input_k_idxs(ggml_tensor * dst, const llama_ubatch * ubatch, const slot_info & sinfo) const {
    if (!supports_set_rows) {
        return;
    }

    const uint32_t n_tokens = ubatch->n_tokens;

    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));
    int64_t * data = (int64_t *) dst->data;

    for (int64_t i = 0; i < n_tokens; ++i) {
        data[i] = sinfo.idxs.at(i);
    }
}

void llama_kv_cache_unified::set_input_v_idxs(ggml_tensor * dst, const llama_ubatch * ubatch, const slot_info & sinfo) const {
    if (!supports_set_rows) {
        return;
    }

    const uint32_t n_tokens = ubatch->n_tokens;

    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));
    int64_t * data = (int64_t *) dst->data;

    for (int64_t i = 0; i < n_tokens; ++i) {
        data[i] = sinfo.idxs.at(i);
    }
}

void llama_kv_cache_unified::set_input_kq_mask(ggml_tensor * dst, const llama_ubatch * ubatch, bool causal_attn) const {
    const uint32_t n_tokens = ubatch->n_tokens;

    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));
    float * data = (float *) dst->data;

    const int64_t n_kv = dst->ne[0];

    // Use only the previous KV cells of the correct sequence for each token of the ubatch.
    // It's assumed that if a token in the batch has multiple sequences, they are equivalent.
    // Example with a cache of 10 tokens, 2 tokens populated in cache and 3 tokens in batch:
    //   Causal mask:
    //      xxx-------
    //      xxxx------
    //      xxxxx-----
    //   Non-causal mask:
    //      xxxxx-----
    //      xxxxx-----
    //      xxxxx-----
    // To visualize the mask, see https://github.com/ggml-org/llama.cpp/pull/12615
    for (uint32_t h = 0; h < 1; ++h) {
        for (uint32_t i = 0; i < n_tokens; ++i) {
            const llama_seq_id seq_id = ubatch->seq_id[i][0];

            const llama_pos p1 = ubatch->pos[i];

            for (uint32_t j = 0; j < n_kv; ++j) {
                float f = 0.0f;

                bool masked = false;

                if (cells.is_empty(j)) {
                    masked = true;
                } else {
                    const llama_pos p0 = cells.pos_get(j);

                    // mask the token if not the same sequence
                    masked = masked || (!cells.seq_has(j, seq_id));

                    // mask future tokens
                    masked = masked || (causal_attn && p0 > p1);

                    // apply SWA if any
                    masked = masked || (is_masked_swa(p0, p1));

                    if (!masked && hparams.use_alibi) {
                        f = -std::abs(p0 - p1);
                    }
                }

                if (masked) {
                    f = -INFINITY;
                }

                data[h*(n_kv*n_tokens) + i*n_kv + j] = f;
            }
        }

        // mask padded tokens
        if (data) {
            for (uint32_t i = n_tokens; i < GGML_PAD(n_tokens, GGML_KQ_MASK_PAD); ++i) {
                for (uint32_t j = 0; j < n_kv; ++j) {
                    data[h*(n_kv*n_tokens) + i*n_kv + j] = -INFINITY;
                }
            }
        }
    }
}

void llama_kv_cache_unified::set_input_k_shift(ggml_tensor * dst) const {
    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));

    int32_t * data = (int32_t *) dst->data;

    for (uint32_t i = 0; i < cells.size(); ++i) {
        data[i] = cells.is_empty(i) ? 0 : cells.get_shift(i);
    }
}

void llama_kv_cache_unified::set_input_pos_bucket(ggml_tensor * dst, const llama_ubatch * ubatch) const {
    const int64_t n_tokens = ubatch->n_tokens;

    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));
    GGML_ASSERT(!ubatch->equal_seqs); // TODO: use ubatch->n_seqs instead of failing

    int32_t * data = (int32_t *) dst->data;

    const int32_t n_kv = dst->ne[0];

    for (int h = 0; h < 1; ++h) {
        for (int i = 0; i < n_tokens; ++i) {
            for (int j = 0; j < n_kv; ++j) {
                // the position when the cells is empty is irrelevant - it will be masked out later in the attention
                const llama_pos p0 = cells.is_empty(j) ? -1 : cells.pos_get(j);

                data[h*(n_kv*n_tokens) + i*n_kv + j] = llama_relative_position_bucket(p0, ubatch->pos[i], hparams.n_rel_attn_bkts, false);
            }
        }
    }
}

size_t llama_kv_cache_unified::total_size() const {
    size_t size = 0;

    for (const auto & buf : bufs) {
        size += ggml_backend_buffer_get_size(buf.get());
    }

    return size;
}

size_t llama_kv_cache_unified::size_k_bytes() const {
    size_t size_k_bytes = 0;

    for (const auto & layer : layers) {
        size_k_bytes += ggml_nbytes(layer.k);
    }

    return size_k_bytes;
}

size_t llama_kv_cache_unified::size_v_bytes() const {
    size_t size_v_bytes = 0;

    for (const auto & layer : layers) {
        size_v_bytes += ggml_nbytes(layer.v);
    }

    return size_v_bytes;
}

ggml_tensor * llama_kv_cache_unified::build_rope_shift(
        const llama_cparams & cparams,
               ggml_context * ctx,
                ggml_tensor * cur,
                ggml_tensor * shift,
                ggml_tensor * factors,
                      float   freq_base,
                      float   freq_scale) const {
    const auto & n_ctx_orig = cparams.n_ctx_orig_yarn;

    const auto & yarn_ext_factor = cparams.yarn_ext_factor;
    const auto & yarn_beta_fast  = cparams.yarn_beta_fast;
    const auto & yarn_beta_slow  = cparams.yarn_beta_slow;

    const auto & n_rot     = hparams.n_rot;
    const auto & rope_type = hparams.rope_type == LLAMA_ROPE_TYPE_MROPE
                                // @ngxson : this is a workaround
                                // for M-RoPE, we want to rotate the whole vector when doing KV shift
                                // a normal RoPE should work, we just need to use the correct ordering
                                // ref: https://github.com/ggml-org/llama.cpp/pull/13870
                                ? LLAMA_ROPE_TYPE_NEOX
                                : hparams.rope_type;

    // See llm_build_deepseek2() for why attn_factor has to be scaled for YaRN RoPE to work correctly.
    // See https://github.com/ggerganov/llama.cpp/discussions/7416 for detailed explanation.
    const float yarn_attn_factor = model.arch == LLM_ARCH_DEEPSEEK2
                                    ? 1.0f / (1.0f + 0.1f * logf(1.0f / freq_scale))
                                    : cparams.yarn_attn_factor;

    ggml_tensor * tmp;

    if (ggml_is_quantized(cur->type)) {
        // dequantize to f32 -> RoPE -> quantize back
        tmp = ggml_cast(ctx, cur, GGML_TYPE_F32);

        tmp = ggml_rope_ext(ctx, tmp,
                shift, factors, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                yarn_ext_factor, yarn_attn_factor, yarn_beta_fast, yarn_beta_slow);

        tmp = ggml_cpy(ctx, tmp, cur);
    } else {
        // we rotate only the first n_rot dimensions
        tmp = ggml_rope_ext_inplace(ctx, cur,
                shift, factors, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                yarn_ext_factor, yarn_attn_factor, yarn_beta_fast, yarn_beta_slow);
    }

    return tmp;
}

class llm_graph_input_k_shift : public llm_graph_input_i {
public:
    llm_graph_input_k_shift(const llama_kv_cache_unified * kv_self) : kv_self(kv_self) {}
    virtual ~llm_graph_input_k_shift() = default;

    void set_input(const llama_ubatch * ubatch) override;

    ggml_tensor * k_shift; // I32 [kv_size]

    const llama_kv_cache_unified * kv_self;
};

void llm_graph_input_k_shift::set_input(const llama_ubatch * ubatch) {
    GGML_UNUSED(ubatch);

    if (k_shift) {
        kv_self->set_input_k_shift(k_shift);
    }
}

llm_graph_result_ptr llama_kv_cache_unified::build_graph_shift(
        const llama_cparams & cparams,
               ggml_context * ctx,
                ggml_cgraph * gf) const {
    auto res = std::make_unique<llm_graph_result>();

    const auto & n_embd_head_k = hparams.n_embd_head_k;
  //const auto & n_embd_head_v = hparams.n_embd_head_v;

    auto inp = std::make_unique<llm_graph_input_k_shift>(this);

    inp->k_shift = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, cells.size());
    ggml_set_input(inp->k_shift);

    for (const auto & layer : layers) {
        const uint32_t il = layer.il;

        const int64_t n_head_kv    = hparams.n_head_kv(il);
        const int64_t n_embd_k_gqa = hparams.n_embd_k_gqa(il);

        const float freq_base_l  = model.get_rope_freq_base (cparams, il);
        const float freq_scale_l = model.get_rope_freq_scale(cparams, il);

        ggml_tensor * rope_factors = model.get_rope_factors(cparams, il);

        ggml_tensor * k =
            ggml_view_3d(ctx, layer.k,
                n_embd_head_k, n_head_kv, cells.size(),
                ggml_row_size(layer.k->type, n_embd_head_k),
                ggml_row_size(layer.k->type, n_embd_k_gqa),
                0);

        ggml_tensor * cur = build_rope_shift(cparams, ctx, k, inp->k_shift, rope_factors, freq_base_l, freq_scale_l);

        ggml_build_forward_expand(gf, cur);
    }

    res->add_input(std::move(inp));

    return res;
}

llm_graph_result_ptr llama_kv_cache_unified::build_graph_defrag(
                const llama_cparams & cparams,
                       ggml_context * ctx,
                        ggml_cgraph * gf,
                  const defrag_info & dinfo) const {
    auto res = std::make_unique<llm_graph_result>();

    const auto & ids = dinfo.ids;

#if 0
    // CPU defrag
    //
    // TODO: optimizations are possible:
    //       - multiple threads
    //       - avoid copying to the host memory when already there
    //
    // likely not worth the effort, as we have ggml_graph based defrag
    //

    const uint32_t n_embd_k_gqa = hparams.n_embd_k_gqa();
    const uint32_t n_embd_v_gqa = hparams.n_embd_v_gqa();

    const uint32_t kv_size = size;

    std::vector<uint8_t> buf_k;
    std::vector<uint8_t> buf_v;

    for (uint32_t il = 0; il < n_layer; ++il) {
        const size_t k_size_row = ggml_row_size(k_l[il]->type, n_embd_k_gqa);
        const size_t k_size     = ggml_row_size(k_l[il]->type, n_embd_k_gqa*kv_size);

        const size_t v_size_el = ggml_type_size(v_l[il]->type);
        const size_t v_size    = ggml_row_size (v_l[il]->type, n_embd_v_gqa*kv_size);

        buf_k.resize(k_size);
        buf_v.resize(v_size);

        ggml_backend_tensor_get(k_l[il], buf_k.data(), 0, buf_k.size());
        ggml_backend_tensor_get(v_l[il], buf_v.data(), 0, buf_v.size());

        // batch move [i, i+nm) to [id, id+nm)
        // note: cells can move only to a lower index
        for (uint32_t i = 0; i < n_kv; ++i) {
            const uint32_t id = ids[i];

            if (i == id || id == n_kv) {
                continue;
            }

            uint32_t nm = 1;

            while (i + nm < n_kv && ids[i + nm] == id + nm) {
                nm++;
            }

            // move keys
            {
                const int64_t os =  i*k_size_row;
                const int64_t od = id*k_size_row;

                memcpy(buf_k.data() + od, buf_k.data() + os, nm*k_size_row);
            }

            // move values (note: they are transposed)
            {
                const int64_t os =  i;
                const int64_t od = id;

                for (uint32_t j = 0; j < n_embd_v_gqa; ++j) {
                    memcpy(buf_v.data() + (od + j*kv_size)*v_size_el, buf_v.data() + (os + j*kv_size)*v_size_el, nm*v_size_el);
                }
            }

            i += nm - 1;
        }

        ggml_backend_tensor_set(k_l[il], buf_k.data(), 0, buf_k.size());
        ggml_backend_tensor_set(v_l[il], buf_v.data(), 0, buf_v.size());
    }
#else
    for (uint32_t i = 0; i < ids.size(); ++i) {
        const uint32_t id = ids[i];

        if (i == id || id == ids.size()) {
            continue;
        }

        uint32_t nm = 1;

        while (i + nm < ids.size() && ids[i + nm] == id + nm) {
            nm++;
        }

        for (const auto & layer : layers) {
            const uint32_t il = layer.il;

            const int64_t n_embd_k_gqa = hparams.n_embd_k_gqa(il);
            const int64_t n_embd_v_gqa = hparams.n_embd_v_gqa(il);

            ggml_tensor * view_k_src = ggml_view_2d(ctx, layer.k,
                    n_embd_k_gqa, nm,
                    ggml_row_size(layer.k->type, n_embd_k_gqa),
                    ggml_row_size(layer.k->type, n_embd_k_gqa*i));

            ggml_tensor * view_k_dst = ggml_view_2d(ctx, layer.k,
                    n_embd_k_gqa, nm,
                    ggml_row_size(layer.k->type, n_embd_k_gqa),
                    ggml_row_size(layer.k->type, n_embd_k_gqa*id));

            ggml_tensor * view_v_src;
            ggml_tensor * view_v_dst;

            if (cparams.flash_attn) {
                // NOTE: the V cache is not transposed when using flash attention
                view_v_src = ggml_view_2d(ctx, layer.v,
                        n_embd_v_gqa, nm,
                        ggml_row_size(layer.v->type, n_embd_v_gqa),
                        ggml_row_size(layer.v->type, n_embd_v_gqa*i));

                view_v_dst = ggml_view_2d(ctx, layer.v,
                        n_embd_v_gqa, nm,
                        ggml_row_size(layer.v->type, n_embd_v_gqa),
                        ggml_row_size(layer.v->type, n_embd_v_gqa*id));
            } else {
                view_v_src = ggml_view_2d(ctx, layer.v,
                        nm, n_embd_v_gqa,
                        ggml_row_size(layer.v->type, cells.size()),
                        ggml_row_size(layer.v->type, i));

                view_v_dst = ggml_view_2d(ctx, layer.v,
                        nm, n_embd_v_gqa,
                        ggml_row_size(layer.v->type, cells.size()),
                        ggml_row_size(layer.v->type, id));
            }

            ggml_build_forward_expand(gf, ggml_cpy(ctx, view_k_src, view_k_dst));
            ggml_build_forward_expand(gf, ggml_cpy(ctx, view_v_src, view_v_dst));
        }

        i += nm - 1;
    }

    //LLAMA_LOG_INFO("gf->n_nodes = %d\n", gf->n_nodes);
#endif

    return res;
}

llama_kv_cache_unified::defrag_info llama_kv_cache_unified::defrag_prepare(int32_t n_max_nodes) const {
    const uint32_t n_layer = layers.size();

    const uint32_t n_kv   = cells.used_max_p1();
    const uint32_t n_used = cells.get_used();

    assert(n_used <= n_kv);

    //const int64_t t_start = ggml_time_us();

    // number of cells moved
    uint32_t n_moves = 0;

    // each move requires 6*n_layer tensors (see graph_build_kv_self_defrag)
    //   - source view, destination view, copy operation
    //   - x2 for keys and values
    //const uint32_t max_moves = max_nodes()/(6*n_layer);
    // TODO: tmp fix https://github.com/ggerganov/llama.cpp/issues/6685#issuecomment-2057579516
    const uint32_t max_moves = (n_max_nodes - 2*n_layer)/(6*n_layer);

    // determine which KV cells to move where
    defrag_info res;
    auto & ids = res.ids;

    ids.resize(n_kv, n_kv);

    for (uint32_t i0 = 0; i0 < n_used; ++i0) {
        if (!cells.is_empty(i0)) {
            ids[i0] = i0;

            continue;
        }

        // found a hole - fill it with data from the end of the cache

        uint32_t nh = 1;

        // determine the size of the hole
        while (i0 + nh < n_used && cells.is_empty(i0 + nh)) {
            nh++;
        }

        uint32_t nf = 0;
        uint32_t is = n_kv - 1;

        // starting from the end, find nh non-empty cells
        for (; is > i0; --is) {
            if (cells.is_empty(is) || ids[is] != n_kv) {
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

        // should we stop searching for the next move?
        bool stop = false;

        // go back and move the nf cells to the hole
        for (; i1 < n_kv; ++i1) {
            if (cells.is_empty(i1) || ids[i1] != n_kv) {
                if (n_moves == max_moves) {
                    stop = true;
                    break;
                }

                cont = false;
                continue;
            }

            // this cell goes to (i0 + nf)
            ids[i1] = i0 + nf;

            if (!cont) {
                n_moves++;
                cont = true;
            }

            nf++;

            if (nf == nh) {
                break;
            }
        }

        if (stop || n_moves == max_moves) {
            break;
        }

        //LLAMA_LOG_INFO("(tmp log) KV defrag: move [%u, %u) to [%u, %u)\n", is, i1 + 1, i0, i0 + nh);

        i0 += nh - 1;
    }

    if (n_moves == 0) {
        return {};
    }

    LLAMA_LOG_DEBUG("%s: (tmp log) KV defrag cell moves: %u\n", __func__, n_moves);

    LLAMA_LOG_DEBUG("%s: expected gf nodes: %u\n", __func__, 6*n_moves*n_layer);

    return res;
}

bool llama_kv_cache_unified::is_masked_swa(llama_pos p0, llama_pos p1) const {
    assert(p0 >= 0 && p1 >= 0);

    switch (swa_type) {
        case LLAMA_SWA_TYPE_NONE:
            {
            } break;
        case LLAMA_SWA_TYPE_STANDARD:
            {
                if (p1 - p0 >= (int32_t) n_swa) {
                    return true;
                }
            } break;
        case LLAMA_SWA_TYPE_CHUNKED:
            {
                const llama_pos pos_chunk_start = (p1 / n_swa) * n_swa;

                if (p0 < pos_chunk_start) {
                    return true;
                }
            } break;
    }

    return false;
}

void llama_kv_cache_unified::state_write(llama_io_write_i & io, llama_seq_id seq_id) const {
    std::vector<std::pair<uint32_t, uint32_t>> cell_ranges; // ranges, from inclusive, to exclusive
    uint32_t cell_count = 0;

    // Count the number of cells with the specified seq_id
    // Find all the ranges of cells with this seq id (or all, when -1)
    uint32_t cell_range_begin = cells.size();

    for (uint32_t i = 0; i < cells.size(); ++i) {
        if (!cells.is_empty(i) && (seq_id == -1 || cells.seq_has(i, seq_id))) {
            ++cell_count;
            if (cell_range_begin == cells.size()) {
                cell_range_begin = i;
            }
        } else {
            if (cell_range_begin != cells.size()) {
                cell_ranges.emplace_back(cell_range_begin, i);
                cell_range_begin = cells.size();
            }
        }
    }

    if (cell_range_begin != cells.size()) {
        cell_ranges.emplace_back(cell_range_begin, cells.size());
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
            clear(true);
        } else {
            seq_rm(seq_id, -1, -1);
        }
        throw std::runtime_error("failed to restore kv cache");
    }
}

void llama_kv_cache_unified::state_write_meta(llama_io_write_i & io, const std::vector<std::pair<uint32_t, uint32_t>> & cell_ranges, llama_seq_id seq_id) const {
    for (const auto & range : cell_ranges) {
        for (uint32_t i = range.first; i < range.second; ++i) {
            std::vector<llama_seq_id> seq_ids;

            for (llama_seq_id cur = 0; cur < (int) n_seq_max; ++cur) {
                if (cur == seq_id || seq_id == -1) {
                    if (cells.seq_has(i, cur)) {
                        seq_ids.push_back(cur);
                    }
                }
            }

            const llama_pos pos     = cells.pos_get(i);
            const uint32_t n_seq_id = seq_ids.size();

            io.write(&pos,      sizeof(pos));
            io.write(&n_seq_id, sizeof(n_seq_id));

            for (const auto & seq_id : seq_ids) {
                io.write(&seq_id, sizeof(seq_id));
            }
        }
    }
}

void llama_kv_cache_unified::state_write_data(llama_io_write_i & io, const std::vector<std::pair<uint32_t, uint32_t>> & cell_ranges) const {
    const uint32_t v_trans = this->v_trans ? 1 : 0;
    const uint32_t n_layer = layers.size();

    io.write(&v_trans, sizeof(v_trans));
    io.write(&n_layer, sizeof(n_layer));

    std::vector<uint8_t> tmp_buf;

    // Iterate and write all the keys first, each row is a cell
    // Get whole range at a time
    for (const auto & layer : layers) {
        const uint32_t il = layer.il;

        const uint32_t n_embd_k_gqa = hparams.n_embd_k_gqa(il);

        // Write key type
        const int32_t k_type_i = (int32_t)layer.k->type;
        io.write(&k_type_i, sizeof(k_type_i));

        // Write row size of key
        const uint64_t k_size_row = ggml_row_size(layer.k->type, n_embd_k_gqa);
        io.write(&k_size_row, sizeof(k_size_row));

        // Read each range of cells of k_size length each into tmp_buf and write out
        for (const auto & range : cell_ranges) {
            const size_t range_size = range.second - range.first;
            const size_t buf_size = range_size * k_size_row;
            io.write_tensor(layer.k, range.first * k_size_row, buf_size);
        }
    }

    if (!v_trans) {
        for (const auto & layer : layers) {
            const uint32_t il = layer.il;

            const uint32_t n_embd_v_gqa = hparams.n_embd_v_gqa(il);

            // Write value type
            const int32_t v_type_i = (int32_t)layer.v->type;
            io.write(&v_type_i, sizeof(v_type_i));

            // Write row size of value
            const uint64_t v_size_row = ggml_row_size(layer.v->type, n_embd_v_gqa);
            io.write(&v_size_row, sizeof(v_size_row));

            // Read each range of cells of v_size length each into tmp_buf and write out
            for (const auto & range : cell_ranges) {
                const size_t range_size = range.second - range.first;
                const size_t buf_size = range_size * v_size_row;
                io.write_tensor(layer.v, range.first * v_size_row, buf_size);
            }
        }
    } else {
        // When v is transposed, we also need the element size and get the element ranges from each row
        const uint32_t kv_size = cells.size();

        for (const auto & layer : layers) {
            const uint32_t il = layer.il;

            const uint32_t n_embd_v_gqa = hparams.n_embd_v_gqa(il);

            // Write value type
            const int32_t v_type_i = (int32_t)layer.v->type;
            io.write(&v_type_i, sizeof(v_type_i));

            // Write element size
            const uint32_t v_size_el = ggml_type_size(layer.v->type);
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
                    io.write_tensor(layer.v, src_offset, buf_size);
                }
            }
        }
    }
}

bool llama_kv_cache_unified::state_read_meta(llama_io_read_i & io, uint32_t cell_count, llama_seq_id dest_seq_id) {
    if (dest_seq_id != -1) {
        // single sequence

        seq_rm(dest_seq_id, -1, -1);

        llama_batch_allocr balloc(hparams.n_pos_per_embd());

        llama_ubatch ubatch = balloc.ubatch_reserve(cell_count, 1);

        for (uint32_t i = 0; i < cell_count; ++i) {
            llama_pos pos;
            uint32_t n_seq_id;

            io.read_to(&pos,      sizeof(pos));
            io.read_to(&n_seq_id, sizeof(n_seq_id));

            if (n_seq_id != 1) {
                LLAMA_LOG_ERROR("%s: invalid seq_id-agnostic kv cell\n", __func__);
                return false;
            }

            // read the sequence id, but directly discard it - we will use dest_seq_id instead
            {
                llama_seq_id seq_id;
                io.read_to(&seq_id, sizeof(seq_id));
            }

            ubatch.pos[i]      = pos;
            ubatch.n_seq_id[i] = n_seq_id;
            ubatch.seq_id[i]   = &dest_seq_id;
        }

        const auto sinfo = find_slot(ubatch, true);
        if (sinfo.empty()) {
            LLAMA_LOG_ERROR("%s: failed to find available cells in kv cache\n", __func__);
            return false;
        }

        apply_ubatch(sinfo, ubatch);

        const auto head_cur = sinfo.head();

        // keep the head at the old position because we will read the KV data into it in state_read_data()
        head = head_cur;

        // DEBUG CHECK: head_cur should be our first cell, head_cur + cell_count - 1 should be our last cell (verify seq_id and pos values)
        // Assume that this is one contiguous block of cells
        GGML_ASSERT(head_cur + cell_count <= cells.size());
        GGML_ASSERT(cells.pos_get(head_cur)                  == ubatch.pos[0]);
        GGML_ASSERT(cells.pos_get(head_cur + cell_count - 1) == ubatch.pos[cell_count - 1]);
        GGML_ASSERT(cells.seq_has(head_cur,                  dest_seq_id));
        GGML_ASSERT(cells.seq_has(head_cur + cell_count - 1, dest_seq_id));
    } else {
        // whole KV cache restore

        if (cell_count > cells.size()) {
            LLAMA_LOG_ERROR("%s: not enough cells in kv cache\n", __func__);
            return false;
        }

        clear(true);

        for (uint32_t i = 0; i < cell_count; ++i) {
            llama_pos pos;
            uint32_t  n_seq_id;

            io.read_to(&pos,      sizeof(pos));
            io.read_to(&n_seq_id, sizeof(n_seq_id));

            cells.pos_set(i, pos);

            for (uint32_t j = 0; j < n_seq_id; ++j) {
                llama_seq_id seq_id;
                io.read_to(&seq_id, sizeof(seq_id));

                if (seq_id < 0 || (uint32_t) seq_id >= n_seq_max) {
                    LLAMA_LOG_ERROR("%s: invalid seq_id, %d is out of range [0, %u)\n", __func__, seq_id, n_seq_max);
                    return false;
                }

                cells.seq_add(i, seq_id);
            }
        }

        head = 0;
    }

    return true;
}

bool llama_kv_cache_unified::state_read_data(llama_io_read_i & io, uint32_t cell_count) {
    uint32_t v_trans;
    uint32_t n_layer;

    io.read_to(&v_trans, sizeof(v_trans));
    io.read_to(&n_layer, sizeof(n_layer));

    if (n_layer != layers.size()) {
        LLAMA_LOG_ERROR("%s: mismatched layer count (%u instead of %u)\n", __func__, n_layer, (uint32_t) layers.size());
        return false;
    }

    if (cell_count > cells.size()) {
        LLAMA_LOG_ERROR("%s: not enough cells in kv cache to restore state (%u > %u)\n", __func__, cell_count, cells.size());
        return false;
    }

    if (this->v_trans != (bool) v_trans) {
        LLAMA_LOG_ERROR("%s: incompatible V transposition\n", __func__);
        return false;
    }

    // For each layer, read the keys for each cell, one row is one cell, read as one contiguous block
    for (const auto & layer : layers) {
        const uint32_t il = layer.il;

        const uint32_t n_embd_k_gqa = hparams.n_embd_k_gqa(il);

        // Read type of key
        int32_t k_type_i_ref;
        io.read_to(&k_type_i_ref, sizeof(k_type_i_ref));
        const int32_t k_type_i = (int32_t) layer.k->type;
        if (k_type_i != k_type_i_ref) {
            LLAMA_LOG_ERROR("%s: mismatched key type (%d != %d, layer %d)\n", __func__, k_type_i, k_type_i_ref, il);
            return false;
        }

        // Read row size of key
        uint64_t k_size_row_ref;
        io.read_to(&k_size_row_ref, sizeof(k_size_row_ref));
        const size_t k_size_row = ggml_row_size(layer.k->type, n_embd_k_gqa);
        if (k_size_row != k_size_row_ref) {
            LLAMA_LOG_ERROR("%s: mismatched key row size (%zu != %zu, layer %d)\n", __func__, k_size_row, (size_t) k_size_row_ref, il);
            return false;
        }

        if (cell_count) {
            // Read and set the keys for the whole cell range
            ggml_backend_tensor_set(layer.k, io.read(cell_count * k_size_row), head * k_size_row, cell_count * k_size_row);
        }
    }

    if (!this->v_trans) {
        for (const auto & layer : layers) {
            const uint32_t il = layer.il;

            const uint32_t n_embd_v_gqa = hparams.n_embd_v_gqa(il);

            // Read type of value
            int32_t v_type_i_ref;
            io.read_to(&v_type_i_ref, sizeof(v_type_i_ref));
            const int32_t v_type_i = (int32_t)layer.v->type;
            if (v_type_i != v_type_i_ref) {
                LLAMA_LOG_ERROR("%s: mismatched value type (%d != %d, layer %d)\n", __func__, v_type_i, v_type_i_ref, il);
                return false;
            }

            // Read row size of value
            uint64_t v_size_row_ref;
            io.read_to(&v_size_row_ref, sizeof(v_size_row_ref));
            const size_t v_size_row = ggml_row_size(layer.v->type, n_embd_v_gqa);
            if (v_size_row != v_size_row_ref) {
                LLAMA_LOG_ERROR("%s: mismatched value row size (%zu != %zu, layer %d)\n", __func__, v_size_row, (size_t) v_size_row_ref, il);
                return false;
            }

            if (cell_count) {
                // Read and set the values for the whole cell range
                ggml_backend_tensor_set(layer.v, io.read(cell_count * v_size_row), head * v_size_row, cell_count * v_size_row);
            }
        }
    } else {
        // For each layer, read the values for each cell (transposed)
        for (const auto & layer : layers) {
            const uint32_t il = layer.il;

            const uint32_t n_embd_v_gqa = hparams.n_embd_v_gqa(il);

            // Read type of value
            int32_t v_type_i_ref;
            io.read_to(&v_type_i_ref, sizeof(v_type_i_ref));
            const int32_t v_type_i = (int32_t)layer.v->type;
            if (v_type_i != v_type_i_ref) {
                LLAMA_LOG_ERROR("%s: mismatched value type (%d != %d, layer %d)\n", __func__, v_type_i, v_type_i_ref, il);
                return false;
            }

            // Read element size of value
            uint32_t v_size_el_ref;
            io.read_to(&v_size_el_ref, sizeof(v_size_el_ref));
            const size_t v_size_el = ggml_type_size(layer.v->type);
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
                    const size_t dst_offset = (head + j * cells.size()) * v_size_el;
                    ggml_backend_tensor_set(layer.v, io.read(cell_count * v_size_el), dst_offset, cell_count * v_size_el);
                }
            }
        }
    }

    return true;
}

//
// llama_kv_cache_unified_context
//

llama_kv_cache_unified_context::llama_kv_cache_unified_context(llama_memory_status status) : status(status) {}

llama_kv_cache_unified_context::llama_kv_cache_unified_context(
        llama_kv_cache_unified * kv) : status(LLAMA_MEMORY_STATUS_SUCCESS), kv(kv) {
    n_kv = kv->get_size();

    // create a dummy slot info - the actual data is irrelevant. we just need to build the graph
    sinfos.resize(1);
    sinfos[0].idxs.resize(1);
    sinfos[0].idxs[0] = 0;
}

llama_kv_cache_unified_context::llama_kv_cache_unified_context(
        llama_kv_cache_unified * kv,
        llama_context * lctx,
        bool do_shift,
        defrag_info dinfo) : status(LLAMA_MEMORY_STATUS_SUCCESS), kv(kv), lctx(lctx), do_shift(do_shift), dinfo(std::move(dinfo)) {
    if (!do_shift && this->dinfo.empty()) {
        status = LLAMA_MEMORY_STATUS_NO_UPDATE;
    }
}

llama_kv_cache_unified_context::llama_kv_cache_unified_context(
        llama_kv_cache_unified * kv,
        llama_kv_cache_unified::slot_info_vec_t sinfos,
        std::vector<llama_ubatch> ubatches) : status(LLAMA_MEMORY_STATUS_SUCCESS), kv(kv), sinfos(std::move(sinfos)), ubatches(std::move(ubatches)) {
}

llama_kv_cache_unified_context::~llama_kv_cache_unified_context() = default;

bool llama_kv_cache_unified_context::next() {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    if (++i_cur >= ubatches.size()) {
        return false;
    }

    return true;
}

bool llama_kv_cache_unified_context::apply() {
    assert(!llama_memory_status_is_fail(status));

    // no ubatches -> this is a KV cache update
    if (ubatches.empty()) {
        kv->update(lctx, do_shift, dinfo);

        return true;
    }

    kv->apply_ubatch(sinfos[i_cur], ubatches[i_cur]);

    n_kv = kv->get_n_kv();

    return true;
}

llama_memory_status llama_kv_cache_unified_context::get_status() const {
    return status;
}

const llama_ubatch & llama_kv_cache_unified_context::get_ubatch() const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    return ubatches[i_cur];
}

uint32_t llama_kv_cache_unified_context::get_n_kv() const {
    return n_kv;
}

ggml_tensor * llama_kv_cache_unified_context::get_k(ggml_context * ctx, int32_t il) const {
    return kv->get_k(ctx, il, n_kv);
}

ggml_tensor * llama_kv_cache_unified_context::get_v(ggml_context * ctx, int32_t il) const {
    return kv->get_v(ctx, il, n_kv);
}

ggml_tensor * llama_kv_cache_unified_context::cpy_k(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * k_idxs, int32_t il) const {
    return kv->cpy_k(ctx, k_cur, k_idxs, il, sinfos[i_cur]);
}

ggml_tensor * llama_kv_cache_unified_context::cpy_v(ggml_context * ctx, ggml_tensor * v_cur, ggml_tensor * v_idxs, int32_t il) const {
    return kv->cpy_v(ctx, v_cur, v_idxs, il, sinfos[i_cur]);
}

ggml_tensor * llama_kv_cache_unified_context::build_input_k_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const {
    return kv->build_input_k_idxs(ctx, ubatch);
}

ggml_tensor * llama_kv_cache_unified_context::build_input_v_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const {
    return kv->build_input_v_idxs(ctx, ubatch);
}

void llama_kv_cache_unified_context::set_input_k_shift(ggml_tensor * dst) const {
    kv->set_input_k_shift(dst);
}

void llama_kv_cache_unified_context::set_input_k_idxs(ggml_tensor * dst, const llama_ubatch * ubatch) const {
    kv->set_input_k_idxs(dst, ubatch, sinfos[i_cur]);
}

void llama_kv_cache_unified_context::set_input_v_idxs(ggml_tensor * dst, const llama_ubatch * ubatch) const {
    kv->set_input_v_idxs(dst, ubatch, sinfos[i_cur]);
}

void llama_kv_cache_unified_context::set_input_kq_mask(ggml_tensor * dst, const llama_ubatch * ubatch, bool causal_attn) const {
    kv->set_input_kq_mask(dst, ubatch, causal_attn);
}

void llama_kv_cache_unified_context::set_input_pos_bucket(ggml_tensor * dst, const llama_ubatch * ubatch) const {
    kv->set_input_pos_bucket(dst, ubatch);
}

uint32_t llama_kv_cache_unified::get_padding(const llama_cparams & cparams) {
    // the FA kernels require padding to avoid extra runtime boundary checks
    return cparams.flash_attn ? 256u : 32u;
}
