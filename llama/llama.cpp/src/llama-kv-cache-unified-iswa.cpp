#include "llama-kv-cache-unified-iswa.h"

#include "llama-impl.h"
#include "llama-batch.h"
#include "llama-model.h"

#include <algorithm>
#include <cassert>

//
// llama_kv_cache_unified_iswa
//

llama_kv_cache_unified_iswa::llama_kv_cache_unified_iswa(
        const llama_model & model,
                ggml_type   type_k,
                ggml_type   type_v,
                     bool   v_trans,
                     bool   offload,
                     bool   swa_full,
                 uint32_t   kv_size,
                 uint32_t   n_seq_max,
                 uint32_t   n_ubatch,
                 uint32_t   n_pad) : hparams(model.hparams) {
    llama_kv_cache_unified::layer_filter_cb filter_base = [&](int32_t il) { return !model.hparams.is_swa(il); };
    llama_kv_cache_unified::layer_filter_cb filter_swa  = [&](int32_t il) { return  model.hparams.is_swa(il); };

    const uint32_t size_base = kv_size;

    uint32_t size_swa = std::min(size_base, GGML_PAD(hparams.n_swa*n_seq_max + n_ubatch, n_pad));

    // when using full-size SWA cache, we set the SWA cache size to be equal to the base cache size
    if (swa_full) {
        LLAMA_LOG_WARN("%s: using full-size SWA cache (ref: %s)\n",
                __func__, "https://github.com/ggml-org/llama.cpp/pull/13194#issuecomment-2868343055");

        size_swa = size_base;
    }

    LLAMA_LOG_INFO("%s: creating non-SWA KV cache, size = %u cells\n", __func__, size_base);

    kv_base = std::make_unique<llama_kv_cache_unified>(
            model, std::move(filter_base), type_k, type_v,
            v_trans, offload, size_base, n_seq_max, n_pad,
            0, LLAMA_SWA_TYPE_NONE);

    LLAMA_LOG_INFO("%s: creating     SWA KV cache, size = %u cells\n", __func__, size_swa);

    kv_swa = std::make_unique<llama_kv_cache_unified>(
            model, std::move(filter_swa), type_k, type_v,
            v_trans, offload, size_swa, n_seq_max, n_pad,
            hparams.n_swa, hparams.swa_type);
}

void llama_kv_cache_unified_iswa::clear() {
    kv_base->clear();
    kv_swa ->clear();
}

bool llama_kv_cache_unified_iswa::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    bool res = true;

    res = res & kv_base->seq_rm(seq_id, p0, p1);
    res = res & kv_swa ->seq_rm(seq_id, p0, p1);

    return res;
}

void llama_kv_cache_unified_iswa::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    kv_base->seq_cp(seq_id_src, seq_id_dst, p0, p1);
    kv_swa ->seq_cp(seq_id_src, seq_id_dst, p0, p1);
}

void llama_kv_cache_unified_iswa::seq_keep(llama_seq_id seq_id) {
    kv_base->seq_keep(seq_id);
    kv_swa ->seq_keep(seq_id);
}

void llama_kv_cache_unified_iswa::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) {
    kv_base->seq_add(seq_id, p0, p1, shift);
    kv_swa ->seq_add(seq_id, p0, p1, shift);
}

void llama_kv_cache_unified_iswa::seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
    kv_base->seq_div(seq_id, p0, p1, d);
    kv_swa ->seq_div(seq_id, p0, p1, d);
}

llama_pos llama_kv_cache_unified_iswa::seq_pos_min(llama_seq_id seq_id) const {
    // the base cache is a superset of the SWA cache, so we can just check the SWA cache
    return kv_swa->seq_pos_min(seq_id);
}

llama_pos llama_kv_cache_unified_iswa::seq_pos_max(llama_seq_id seq_id) const {
    return kv_swa->seq_pos_max(seq_id);
}

llama_memory_state_ptr llama_kv_cache_unified_iswa::init_batch(const llama_batch & batch, uint32_t n_ubatch, bool embd_pooled, bool logits_all) {
    GGML_UNUSED(embd_pooled);

    // TODO: if we fail with split_simple, we should attempt different splitting strategies
    //       but to do that properly, we first have to refactor the batches to be more flexible

    auto sbatch = llama_sbatch(batch, hparams.n_embd, true, logits_all);

    std::vector<llama_ubatch> ubatches;

    while (sbatch.n_tokens > 0) {
        auto ubatch = sbatch.split_simple(n_ubatch);

        ubatches.push_back(ubatch);
    }

    auto heads_base = kv_base->prepare(ubatches);
    if (heads_base.empty()) {
        return std::make_unique<llama_kv_cache_unified_iswa_state>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
    }

    auto heads_swa = kv_swa->prepare(ubatches);
    if (heads_swa.empty()) {
        return std::make_unique<llama_kv_cache_unified_iswa_state>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
    }

    assert(heads_base.size() == heads_swa.size());

    return std::make_unique<llama_kv_cache_unified_iswa_state>(
            this, std::move(sbatch), std::move(heads_base), std::move(heads_swa), std::move(ubatches));
}

llama_memory_state_ptr llama_kv_cache_unified_iswa::init_full() {
    return std::make_unique<llama_kv_cache_unified_iswa_state>(this);
}

llama_memory_state_ptr llama_kv_cache_unified_iswa::init_update(llama_context * lctx, bool optimize) {
    return std::make_unique<llama_kv_cache_unified_iswa_state>(this, lctx, optimize);
}

bool llama_kv_cache_unified_iswa::get_can_shift() const {
    return kv_base->get_size() == kv_swa->get_size();
}

void llama_kv_cache_unified_iswa::state_write(llama_io_write_i & io, llama_seq_id seq_id) const {
    kv_base->state_write(io, seq_id);
    kv_swa ->state_write(io, seq_id);
}

void llama_kv_cache_unified_iswa::state_read(llama_io_read_i & io, llama_seq_id seq_id) {
    kv_base->state_read(io, seq_id);
    kv_swa ->state_read(io, seq_id);
}

llama_kv_cache_unified * llama_kv_cache_unified_iswa::get_base() const {
    return kv_base.get();
}

llama_kv_cache_unified * llama_kv_cache_unified_iswa::get_swa() const {
    return kv_swa.get();
}

//
// llama_kv_cache_unified_iswa_state
//

llama_kv_cache_unified_iswa_state::llama_kv_cache_unified_iswa_state(llama_memory_status status) : status(status) {}

llama_kv_cache_unified_iswa_state::llama_kv_cache_unified_iswa_state(
        llama_kv_cache_unified_iswa * kv) : status(LLAMA_MEMORY_STATUS_SUCCESS) {
    state_base = kv->get_base()->init_full();
    state_swa  = kv->get_swa ()->init_full();

    status = llama_memory_status_combine(state_base->get_status(), state_swa->get_status());
}

llama_kv_cache_unified_iswa_state::llama_kv_cache_unified_iswa_state(
        llama_kv_cache_unified_iswa * kv,
        llama_context * lctx,
        bool optimize) : status(LLAMA_MEMORY_STATUS_SUCCESS) {
    state_base = kv->get_base()->init_update(lctx, optimize);
    state_swa  = kv->get_swa ()->init_update(lctx, optimize);

    status = llama_memory_status_combine(state_base->get_status(), state_swa->get_status());
}

llama_kv_cache_unified_iswa_state::llama_kv_cache_unified_iswa_state(
        llama_kv_cache_unified_iswa * kv,
        llama_sbatch sbatch,
        std::vector<uint32_t> heads_base,
        std::vector<uint32_t> heads_swa,
        std::vector<llama_ubatch> ubatches)
        : status(LLAMA_MEMORY_STATUS_SUCCESS),
        sbatch(std::move(sbatch)),
        ubatches(std::move(ubatches)) {
    // note: here we copy the ubatches. not sure if this is ideal
    state_base.reset(new llama_kv_cache_unified_state(kv->get_base(), {}, std::move(heads_base), this->ubatches));
    state_swa .reset(new llama_kv_cache_unified_state(kv->get_swa (), {}, std::move(heads_swa),  this->ubatches));

    status = llama_memory_status_combine(state_base->get_status(), state_swa->get_status());
}

llama_kv_cache_unified_iswa_state:: ~llama_kv_cache_unified_iswa_state() = default;

bool llama_kv_cache_unified_iswa_state::next() {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    state_base->next();
    state_swa ->next();

    if (++i_next >= ubatches.size()) {
        return false;
    }

    return true;
}

bool llama_kv_cache_unified_iswa_state::apply() {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    bool res = true;

    res = res & state_base->apply();
    res = res & state_swa ->apply();

    return res;
}

std::vector<int64_t> & llama_kv_cache_unified_iswa_state::out_ids() {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    return sbatch.out_ids;
}

llama_memory_status llama_kv_cache_unified_iswa_state::get_status() const {
    return status;
}

const llama_ubatch & llama_kv_cache_unified_iswa_state::get_ubatch() const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    return ubatches[i_next];
}

const llama_kv_cache_unified_state * llama_kv_cache_unified_iswa_state::get_base() const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    return static_cast<const llama_kv_cache_unified_state *>(state_base.get());
}

const llama_kv_cache_unified_state * llama_kv_cache_unified_iswa_state::get_swa()  const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    return static_cast<const llama_kv_cache_unified_state *>(state_swa.get());
}
