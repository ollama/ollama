#include "llama-batch.h"

#include "llama-impl.h"
#include "llama-vocab.h"
#include "llama-memory.h"

#include <cassert>
#include <cstring>
#include <algorithm>
#include <sstream>

llama_batch_allocr::llama_batch_allocr(uint32_t n_pos_per_embd) : n_pos_per_embd(n_pos_per_embd) {
    const char * LLAMA_BATCH_DEBUG = getenv("LLAMA_BATCH_DEBUG");
    debug = LLAMA_BATCH_DEBUG ? atoi(LLAMA_BATCH_DEBUG) : 0;

    seq_pos.resize(LLAMA_MAX_SEQ);
    seq_cpl.resize(LLAMA_MAX_SEQ);
    for (auto & cur : seq_cpl) {
        cur.resize(LLAMA_MAX_SEQ);
    }

    seq_idx.resize(LLAMA_MAX_SEQ, -1);
}

bool llama_batch_allocr::init(
        const llama_batch & batch_inp,
        const llama_vocab & vocab,
        const llama_memory_i * memory,
        uint32_t n_embd,
        bool output_all) {
    clear();

    batch = batch_inp;

    this->vocab = &vocab;

    GGML_ASSERT(batch.n_tokens > 0);

    //
    // validate input batch
    //

    if (batch.token) {
        for (int32_t i = 0; i < batch.n_tokens; ++i) {
            if (batch.token[i] < 0 || (uint32_t) batch.token[i] >= vocab.n_tokens()) {
                LLAMA_LOG_ERROR("%s: invalid token[%d] = %d\n", __func__, i, batch.token[i]);
                return false;
            }
        }
    }

    if (batch.seq_id) {
        for (int32_t i = 0; i < batch.n_tokens; ++i) {
            for (int32_t s = 0; s < batch.n_seq_id[i]; ++s) {
                if (batch.seq_id && (batch.seq_id[i][s] < 0 || batch.seq_id[i][s] >= LLAMA_MAX_SEQ)) {
                    LLAMA_LOG_ERROR("%s: invalid seq_id[%d][%d] = %d > %d\n", __func__, i, s, batch.seq_id[i][s], LLAMA_MAX_SEQ);
                    return false;
                }
            }
        }
    }

    //
    // auto-generate missing fields
    //

    if (!batch.n_seq_id) {
        n_seq_id.resize(batch.n_tokens);
        for (int32_t i = 0; i < batch.n_tokens; i++) {
            n_seq_id[i] = seq_id_0.size();
        }
        batch.n_seq_id = n_seq_id.data();
    }

    if (!batch.seq_id) {
        seq_id.resize(batch.n_tokens + 1);
        seq_id[batch.n_tokens] = NULL;
        for (int32_t i = 0; i < batch.n_tokens; i++) {
            seq_id[i] = seq_id_0.data();
        }
        batch.seq_id = seq_id.data();
    }

    if (!batch.pos) {
        pos.resize(batch.n_tokens);

        // initialize the starting position for each sequence based on the positions in the memory
        llama_pos p0[LLAMA_MAX_SEQ];
        for (int32_t s = 0; s < LLAMA_MAX_SEQ; ++s) {
            if (!memory) {
                // if no memory -> start from 0
                p0[s] = 0;
            } else {
                p0[s] = memory->seq_pos_max(s) + 1;
            }
        }

        for (int32_t i = 0; i < batch.n_tokens; i++) {
            const llama_seq_id seq_id = batch.seq_id[i][0];

            pos[i] = p0[seq_id];

            // update the starting position for all sequences that are assigned to the this token
            for (int32_t s = 0; s < batch.n_seq_id[i]; ++s) {
                const llama_seq_id seq_id = batch.seq_id[i][s];

                p0[seq_id] = pos[i] + 1;
            }
        }

        batch.pos = pos.data();
    }

    if (!batch.logits) {
        if (output_all) {
            // return the output for all tokens
            output.resize(batch.n_tokens, true);
        } else {
            // return the output only for the last token
            output.resize(batch.n_tokens, false);
            output[output.size() - 1] = true;
        }

        batch.logits = output.data();
    } else if (output_all) {
        bool warn = false;

        for (int32_t i = 0; i < batch.n_tokens; ++i) {
            if (batch.logits[i] == 0) {
                warn = true;
            }
        }

        if (warn) {
            LLAMA_LOG_WARN("%s: embeddings required but some input tokens were not marked as outputs -> overriding\n", __func__);

            output.resize(batch.n_tokens, true);
            batch.logits = output.data();
        }
    }

    //
    // compute stats
    //

    this->n_embd = n_embd;

    // count the outputs in this batch
    for (int32_t i = 0; i < batch.n_tokens; ++i) {
        n_outputs += batch.logits[i] != 0;
    }

    // determine coupled sequences
    // these are pairs of sequences that have at least one token in the input batch that is assigned to both of them
    for (int32_t i = 0; i < batch.n_tokens; ++i) {
        const llama_seq_id s0 = batch.seq_id[i][0];

        for (int32_t s = 0; s < batch.n_seq_id[i]; ++s) {
            const llama_seq_id s1 = batch.seq_id[i][s];

            seq_pos[s1].insert(batch.pos[i]);

            if (s > 0) {
                // mark that sequence s1 is coupled to s0
                seq_cpl[s1][s0] = true;

                // note: tracking the other way around is not necessary for now
                //seq_cpl[s0][s1] = true;
            }
        }
    }

    // precompute the sequence sets for each token and determine the unique sequence ids that participate in the batch
    {
        seq_set_t seq_set_unq;

        for (int32_t i = 0; i < batch.n_tokens; ++i) {
            seq_set_t cur;
            for (int32_t s = 0; s < batch.n_seq_id[i]; ++s) {
                const llama_seq_id seq_id = batch.seq_id[i][s];

                cur        .set(seq_id);
                seq_set_unq.set(seq_id);
            }

            seq_set.push_back(cur);
            seq_set_map[cur].push_back(i);
        }

        for (int32_t s = 0; s < LLAMA_MAX_SEQ; ++s) {
            if (seq_set_unq.test(s)) {
                seq_idx[s] = seq_id_unq.size();
                seq_id_unq.push_back(s);
            }
        }
    }

    if (debug > 0) {
        LLAMA_LOG_DEBUG("%s: input batch info:\n", __func__);

        llama_ubatch ubatch {
            /*.equal_seqs   =*/ false,
            /*.n_tokens     =*/ (uint32_t) batch.n_tokens,
            /*.n_seq_tokens =*/ (uint32_t) 1,
            /*.n_seqs       =*/ (uint32_t) batch.n_tokens,
            /*.n_seqs_unq   =*/ (uint32_t) this->seq_id_unq.size(),
            /*.token        =*/ batch.token,
            /*.embd         =*/ batch.embd,
            /*.pos          =*/ batch.pos,
            /*.n_seq_id     =*/ batch.n_seq_id,
            /*.seq_id       =*/ batch.seq_id,
            /*.seq_id_unq   =*/ this->seq_id_unq.data(),
            /*.seq_idx      =*/ this->seq_idx.data(),
            /*.output       =*/ batch.logits,
        };

        ubatch_print(ubatch, debug);

        LLAMA_LOG_DEBUG("%s:   seq       = [\n", __func__);
        for (int s0 = 0; s0 < (int) seq_pos.size(); ++s0) {
            if (seq_pos[s0].empty()) {
                continue;
            }

            std::stringstream ss;
            for (int s1 = 0; s1 < (int) seq_cpl[s0].size(); ++s1) {
                if (seq_cpl[s0][s1]) {
                    ss << s1 << " ";
                }
            }

            LLAMA_LOG_DEBUG("%s:  %4d: pos = [%4d, %4d], cpl = %s\n",
                    __func__, s0, seq_pos_min(s0), seq_pos_max(s0), ss.str().empty() ? "-" : ss.str().c_str());
        }
        LLAMA_LOG_DEBUG("%s:   ]\n", __func__);
    }

    //
    // consistency checks
    //

    for (int32_t s = 0; s < LLAMA_MAX_SEQ; ++s) {
        if (seq_pos[s].empty()) {
            continue;
        }

        const llama_pos p0 = memory ? memory->seq_pos_max(s) : -1;

        if (p0 >= 0) {
            bool ok = true;

            if (batch.token) {
                if (seq_pos_min(s) != p0 + 1) {
                    ok = false;
                }
            } else {
                assert(batch.embd);

                // for embeddings (typically used as vision input), we allow them to have repeating positions
                // ref: https://github.com/ggml-org/llama.cpp/issues/13694#issuecomment-2983871762
                if (seq_pos_min(s) != p0 && seq_pos_min(s) != p0 + 1) {
                    ok = false;
                }
            }

            if (!ok) {
                LLAMA_LOG_ERROR(
                        "%s: the tokens of sequence %d in the input batch have inconsistent sequence positions:\n"
                        " - the last position stored in the memory module of the context (i.e. the KV cache) for sequence %d is X = %d\n"
                        " - the tokens for sequence %d in the input batch have a starting position of Y = %d\n"
                        " it is required that the sequence positions remain consecutive: Y = X + 1\n",
                        __func__, s, s, p0, s, seq_pos_min(s));

                return false;
            }
        }

        if (seq_pos_max(s) - seq_pos_min(s) + 1 > (int) seq_pos[s].size()) {
            LLAMA_LOG_ERROR("%s: sequence %d positions are not continuous\n", __func__, s);
            return false;
        }
    }

    if (memory) {
        for (int32_t s0 = 0; s0 < LLAMA_MAX_SEQ; ++s0) {
            for (int32_t s1 = 0; s1 < LLAMA_MAX_SEQ; ++s1) {
                if (seq_cpl[s0][s1]) {
                    if (memory->seq_pos_min(s0) != memory->seq_pos_min(s1) ||
                        memory->seq_pos_max(s0) != memory->seq_pos_max(s1)) {
                        LLAMA_LOG_ERROR("%s: sequence %d is coupled to %d in the input batch, but have divereged\n", __func__, s0, s1);
                        return false;
                    }
                }
            }
        }
    }

    // disallow partial sequence sub-sets:
    //
    // invalid:          x
    //            i: 0 1 2 ...
    // ---------------------------------------
    // seq_id[i][0]: 0 0 1
    // seq_id[i][1]: 1 1 2
    // seq_id[i][2]: 2
    //
    // disallow decreasing sequence positions:
    //
    // invalid:                  x
    //            i: 0 1 2 3 4 5 6 ...
    // ---------------------------------------
    //       pos[i]: 4 5 0 1 6 2 3
    // seq_id[i][0]: 0 0 1 1 0 1 0
    //
    {
        seq_set_t cur_seq_set[LLAMA_MAX_SEQ];
        for (int32_t s = 0; s < LLAMA_MAX_SEQ; ++s) {
            cur_seq_set[s].set();
        }

        llama_pos cur_seq_pos[LLAMA_MAX_SEQ];
        for (int32_t s = 0; s < LLAMA_MAX_SEQ; ++s) {
            cur_seq_pos[s] = -1;
        }

        for (int32_t i = 0; i < batch.n_tokens; ++i) {
            const llama_pos pos = batch.pos[i];

            for (int32_t s = 0; s < batch.n_seq_id[i]; ++s) {
                const llama_seq_id seq_id = batch.seq_id[i][s];

                cur_seq_set[seq_id] &= seq_set[i];

                if (cur_seq_set[seq_id].none()) {
                    LLAMA_LOG_ERROR("%s: sequence %d belongs to incompatible sequence sets (not allowed)\n", __func__, seq_id);
                    return false;
                }

                if (pos < cur_seq_pos[seq_id]) {
                    LLAMA_LOG_ERROR("%s: sequence %d positions are decreasing (not allowed)\n", __func__, seq_id);
                    return false;
                }
            }
        }
    }

    split_reset();

    return true;
}

llama_ubatch llama_batch_allocr::ubatch_reserve(uint32_t n_seq_tokens, uint32_t n_seqs) {
    const uint32_t n_tokens = n_seq_tokens*n_seqs;

    clear();
    split_reset();

    ubatches.emplace_back();

    auto & ubatch = ubatches.back();

    ubatch.token     .resize(n_tokens);
    ubatch.embd      .clear();
    ubatch.pos       .resize(n_tokens);
    ubatch.n_seq_id  .resize(n_tokens);
    ubatch.seq_id    .resize(n_tokens);
    ubatch.seq_id_unq.resize(0);
    ubatch.seq_idx   .resize(LLAMA_MAX_SEQ, -1);
    ubatch.output    .resize(n_tokens);

    for (uint32_t s = 0; s < n_seqs; ++s) {
        ubatch.seq_idx[s] = s;
        ubatch.seq_id_unq.push_back(s);
    }

    llama_ubatch res {
        /*.equal_seqs   =*/ true,
        /*.n_tokens     =*/ n_tokens,
        /*.n_seq_tokens =*/ n_seq_tokens,
        /*.n_seqs       =*/ n_seqs,
        /*.n_seqs_unq   =*/ n_seqs,

        /*.token        =*/ ubatch.token.data(),
        /*.embd         =*/ nullptr,
        /*.pos          =*/ ubatch.pos.data(),
        /*.n_seq_id     =*/ ubatch.n_seq_id.data(),
        /*.seq_id       =*/ ubatch.seq_id.data(),
        /*.seq_id_unq   =*/ ubatch.seq_id_unq.data(),
        /*.seq_idx      =*/ ubatch.seq_idx.data(),
        /*.output       =*/ ubatch.output.data(),
    };

    return res;
}

const llama_batch & llama_batch_allocr::get_batch() const {
    return batch;
}

uint32_t llama_batch_allocr::get_n_tokens() const {
    return batch.n_tokens;
}

uint32_t llama_batch_allocr::get_n_outputs() const {
    return n_outputs;
}

std::vector<int32_t> & llama_batch_allocr::get_out_ids() {
    return out_ids;
}

llama_pos llama_batch_allocr::seq_pos_min(llama_seq_id seq_id) const {
    return seq_pos[seq_id].empty() ? -1 : *seq_pos[seq_id].begin();
}

llama_pos llama_batch_allocr::seq_pos_max(llama_seq_id seq_id) const {
    return seq_pos[seq_id].empty() ? -1 : *seq_pos[seq_id].rbegin();
}

void llama_batch_allocr::split_reset() {
    out_ids.clear();

    used.clear();
    used.resize(get_n_tokens(), false);

    ubatches.clear();
}

llama_ubatch llama_batch_allocr::split_simple(uint32_t n_ubatch) {
    // find the first unused token
    uint32_t cur_idx = 0;
    while (cur_idx < used.size() && used[cur_idx]) {
        ++cur_idx;
    }

    // we are done
    if (cur_idx >= used.size()) {
        return {};
    }

    std::vector<int32_t> idxs;

    while (true) {
        idxs.push_back(cur_idx);

        used[cur_idx] = true;

        ++cur_idx;

        if (cur_idx >= used.size()) {
            break;
        }

        if (idxs.size() >= n_ubatch) {
            break;
        }
    }

    return ubatch_add(idxs, idxs.size(), false);
}

llama_ubatch llama_batch_allocr::split_equal(uint32_t n_ubatch) {
    std::vector<seq_set_t> cur_seq_set;

    // determine the non-overlapping sequence sets participating in this ubatch
    for (int32_t i = 0; i < batch.n_tokens; ++i) {
        if (used[i]) {
            continue;
        }

        bool add = true;

        for (uint32_t s = 0; s < cur_seq_set.size(); ++s) {
            // no overlap with existing sequence sets:
            if (!(cur_seq_set[s] & seq_set[i]).none()) {
                add = false;
                break;
            }
        }

        if (add) {
            cur_seq_set.push_back(seq_set[i]);

            if (cur_seq_set.size() > n_ubatch) {
                break;
            }
        }
    }

    const uint32_t n_seqs = cur_seq_set.size();

    // we are done
    if (n_seqs == 0) {
        return {};
    }

    // the current batch index of each sequence set
    std::vector<int32_t> cur_idx(n_seqs, 0);

    for (uint32_t s = 0; s < n_seqs; ++s) {
        while (used[seq_set_map[cur_seq_set[s]][cur_idx[s]]]) {
            ++cur_idx[s];
        }
    }

    // the list of batch indices for each sequence set
    // at the end we will concat these to get the final ubatch
    std::vector<idx_vec_t> idxs_per_seq(n_seqs);

    while (true) {
        // we can only add new n_seq_tokens tokens if all the sequence sets have at least one more unused token and
        //   if we haven't reached n_ubatch
        bool can_expand = true;

        for (uint32_t s = 0; s < n_seqs; ++s) {
            if (cur_idx[s] >= (int32_t) seq_set_map[cur_seq_set[s]].size()) {
                can_expand = false;
                break;
            }
        }

        if (!can_expand) {
            break;
        }

        for (uint32_t s = 0; s < n_seqs; ++s) {
            const int32_t idx = seq_set_map[cur_seq_set[s]][cur_idx[s]];

            idxs_per_seq[s].push_back(idx);

            used[idx] = true;

            ++cur_idx[s];
        }

        if  ((idxs_per_seq[0].size() + 1)*n_seqs > n_ubatch) {
            break;
        }
    }

    // concat the per-sequence-set lists
    std::vector<int32_t> idxs;

    for (uint32_t s = 0; s < n_seqs; ++s) {
        idxs.insert(idxs.end(), idxs_per_seq[s].begin(), idxs_per_seq[s].end());
    }

    return ubatch_add(idxs, n_seqs, true);
}

llama_ubatch llama_batch_allocr::split_seq(uint32_t n_ubatch) {
    // find the first unused token
    uint32_t cur_idx = 0;
    while (cur_idx < used.size() && used[cur_idx]) {
        ++cur_idx;
    }

    // we are done
    if (cur_idx >= used.size()) {
        return {};
    }

    // this is the starting sequence set
    // we allow adding tokens only if their sequence set is a subset of the current sequence set
    auto cur_seq_set = seq_set[cur_idx];

    std::vector<int32_t> idxs;

    while (true) {
        idxs.push_back(cur_idx);

        used[cur_idx] = true;

        if (idxs.size() >= n_ubatch) {
            break;
        }

        do {
            ++cur_idx;
        } while (cur_idx < get_n_tokens() && (used[cur_idx] || ((cur_seq_set & seq_set[cur_idx]) != seq_set[cur_idx])));

        if (cur_idx == get_n_tokens()) {
            break;
        }

        cur_seq_set = seq_set[cur_idx];
    }

    return ubatch_add(idxs, 1, true);
}

void llama_batch_allocr::clear() {
    n_outputs = 0;

    batch = {};

    pos       .clear();
    n_seq_id  .clear();
    seq_id    .clear();
    seq_id_unq.clear();
    output    .clear();

    for (auto & cur : seq_pos) {
        cur.clear();
    }

    for (auto & cur : seq_cpl) {
        std::fill(cur.begin(), cur.end(), false);
    }

    seq_set.clear();

    seq_set_map.clear();

    std::fill(seq_idx.begin(), seq_idx.end(), -1);
}

llama_ubatch llama_batch_allocr::ubatch_add(const std::vector<int32_t> & idxs, uint32_t n_seqs, bool equal_seqs) {
    const uint32_t n_tokens = idxs.size();

    assert(n_tokens%n_seqs == 0);

    ubatches.emplace_back();

    auto & ubatch = ubatches.back();

    const int32_t n_pos_cur = batch.embd ? n_pos_per_embd : 1;

    const int64_t n_embd_all = batch.embd ? (int64_t) n_tokens*n_embd : 0;
    const int64_t n_pos_all  =              (int64_t) n_tokens*n_pos_cur;

    ubatch.token     .resize(n_tokens);
    ubatch.embd      .resize(n_embd_all);
    ubatch.pos       .resize(n_pos_all);
    ubatch.n_seq_id  .resize(n_tokens);
    ubatch.seq_id    .resize(n_tokens);
    ubatch.seq_id_unq.resize(0);
    ubatch.seq_idx   .resize(LLAMA_MAX_SEQ, -1);
    ubatch.output    .resize(n_tokens);

    seq_set_t seq_set_unq;

    for (size_t i = 0; i < idxs.size(); ++i) {
        if (batch.token) {
            ubatch.token[i] = batch.token[idxs[i]];
        }

        if (batch.embd) {
            memcpy(ubatch.embd.data() + i*n_embd, batch.embd + (int64_t) idxs[i]*n_embd, n_embd*sizeof(float));
        }

        for (int j = 0; j < n_pos_cur; ++j) {
            ubatch.pos[j*n_tokens + i] = batch.pos[j*batch.n_tokens + idxs[i]];
        }

        ubatch.n_seq_id[i] = batch.n_seq_id[idxs[i]];
        ubatch.seq_id[i]   = batch.seq_id[idxs[i]];
        ubatch.output[i]   = batch.logits[idxs[i]];

        for (int s = 0; s < ubatch.n_seq_id[i]; ++s) {
            seq_set_unq.set(ubatch.seq_id[i][s]);
        }

        if (ubatch.output[i]) {
            out_ids.push_back(idxs[i]);
        }
    }

    for (int32_t s = 0; s < LLAMA_MAX_SEQ; ++s) {
        if (seq_set_unq.test(s)) {
            ubatch.seq_idx[s] = ubatch.seq_id_unq.size();
            ubatch.seq_id_unq.push_back(s);
        }
    }

    llama_ubatch res {
        /*.equal_seqs   =*/ equal_seqs,
        /*.n_tokens     =*/ n_tokens,
        /*.n_seq_tokens =*/ n_tokens/n_seqs,
        /*.n_seqs       =*/ n_seqs,
        /*.n_seqs_unq   =*/ (uint32_t) ubatch.seq_id_unq.size(),

        /*.token        =*/ batch.token ? ubatch.token.data() : nullptr,
        /*.embd         =*/ batch.embd ? ubatch.embd.data() : nullptr,
        /*.pos          =*/ ubatch.pos.data(),
        /*.n_seq_id     =*/ ubatch.n_seq_id.data(),
        /*.seq_id       =*/ ubatch.seq_id.data(),
        /*.seq_id_unq   =*/ ubatch.seq_id_unq.data(),
        /*.seq_idx      =*/ ubatch.seq_idx.data(),
        /*.output       =*/ ubatch.output.data(),
    };

    if (debug > 0) {
        LLAMA_LOG_DEBUG("%s: added ubatch %d to split:\n", __func__, (int) ubatches.size() - 1);

        ubatch_print(res, debug);
    }

    return res;
}

void llama_batch_allocr::ubatch_print(const llama_ubatch & ubatch, int debug) {
    if (debug > 0) {
        LLAMA_LOG_DEBUG("%s:   equal_seqs   = %d\n", __func__, ubatch.equal_seqs);
        LLAMA_LOG_DEBUG("%s:   n_tokens     = %d\n", __func__, ubatch.n_tokens);
        LLAMA_LOG_DEBUG("%s:   n_seq_tokens = %d\n", __func__, ubatch.n_seq_tokens);
        LLAMA_LOG_DEBUG("%s:   n_seqs       = %d\n", __func__, ubatch.n_seqs);
        LLAMA_LOG_DEBUG("%s:   n_seqs_unq   = %d\n", __func__, ubatch.n_seqs_unq);

        std::stringstream ss_seq_id_unq;
        std::stringstream ss_seq_idx;

        ss_seq_id_unq << "[ ";
        ss_seq_idx << "[";

        for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
            ss_seq_id_unq << ubatch.seq_id_unq[s] << " ";
        }

        for (uint32_t s = 0; s < LLAMA_MAX_SEQ; ++s) {
            if (ubatch.seq_idx[s] >= 0) {
                ss_seq_idx << ubatch.seq_idx[s]%10;
            } else {
                ss_seq_idx << ".";
            }
        }

        ss_seq_id_unq << "]";
        ss_seq_idx    << "]";

        LLAMA_LOG_DEBUG("%s:   token      = %p\n", __func__, (void *) ubatch.token);
        LLAMA_LOG_DEBUG("%s:   embd       = %p\n", __func__, (void *) ubatch.embd);
        LLAMA_LOG_DEBUG("%s:   pos        = %p\n", __func__, (void *) ubatch.pos);
        LLAMA_LOG_DEBUG("%s:   n_seq_id   = %p\n", __func__, (void *) ubatch.n_seq_id);
        LLAMA_LOG_DEBUG("%s:   seq_id     = %p\n", __func__, (void *) ubatch.seq_id);
        LLAMA_LOG_DEBUG("%s:   seq_id_unq = %s\n", __func__, ss_seq_id_unq.str().c_str());
        LLAMA_LOG_DEBUG("%s:   seq_idx    = %s\n", __func__, ss_seq_idx.str().c_str());
        LLAMA_LOG_DEBUG("%s:   output     = %p\n", __func__, (void *) ubatch.output);
        LLAMA_LOG_DEBUG("%s:   n_outputs  = %d\n", __func__, n_outputs);

        if (debug > 1) {
            int seq_id_max = 0;
            for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
                for (int s = 0; s < ubatch.n_seq_id[i]; ++s) {
                    for (int s = 0; s < ubatch.n_seq_id[i]; ++s) {
                        seq_id_max = std::max(seq_id_max, ubatch.seq_id[i][s]);
                    }
                }
            }
            ++seq_id_max;

            LLAMA_LOG_DEBUG("%s:   token     = [\n", __func__);
            for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
                std::vector<int8_t> seq_id(seq_id_max);

                for (int s = 0; s < ubatch.n_seq_id[i]; ++s) {
                    seq_id[ubatch.seq_id[i][s]] = 1;
                }

                std::stringstream ss;
                for (int s = 0; s < seq_id_max; ++s) {
                    if (seq_id[s]) {
                        ss << s%10;
                    } else {
                        ss << ".";
                    }
                }

                if (ubatch.token) {
                    LLAMA_LOG_DEBUG("%s:  %4d: id = %6d (%16s), pos = %4d, n_seq_id = %2d, seq_id = [%s], output = %d\n",
                            __func__, i, ubatch.token[i], vocab->token_to_piece(ubatch.token[i]).c_str(),
                            ubatch.pos[i], ubatch.n_seq_id[i], ss.str().c_str(), ubatch.output[i]);
                } else {
                    LLAMA_LOG_DEBUG("%s:  %4d: [embd], pos = %4d, n_seq_id = %2d, seq_id = [%s], output = %d\n",
                            __func__, i, ubatch.pos[i], ubatch.n_seq_id[i], ss.str().c_str(), ubatch.output[i]);
                }
            }
            LLAMA_LOG_DEBUG("%s:   ]\n", __func__);
        }
    }
}

//
// interface implementation
//

struct llama_batch llama_batch_get_one(
             llama_token * tokens,
                 int32_t   n_tokens) {
    return {
        /*n_tokens =*/ n_tokens,
        /*tokens   =*/ tokens,
        /*embd     =*/ nullptr,
        /*pos      =*/ nullptr,
        /*n_seq_id =*/ nullptr,
        /*seq_id   =*/ nullptr,
        /*logits   =*/ nullptr,
    };
}

struct llama_batch llama_batch_init(int32_t n_tokens_alloc, int32_t embd, int32_t n_seq_max) {
    llama_batch batch = {
        /*n_tokens =*/ 0,
        /*tokens   =*/ nullptr,
        /*embd     =*/ nullptr,
        /*pos      =*/ nullptr,
        /*n_seq_id =*/ nullptr,
        /*seq_id   =*/ nullptr,
        /*logits   =*/ nullptr,
    };

    if (embd) {
        batch.embd = (float *) malloc(sizeof(float) * n_tokens_alloc * embd);
    } else {
        batch.token = (llama_token *) malloc(sizeof(llama_token) * n_tokens_alloc);
    }

    batch.pos      = (llama_pos *)     malloc(sizeof(llama_pos)      * n_tokens_alloc);
    batch.n_seq_id = (int32_t *)       malloc(sizeof(int32_t)        * n_tokens_alloc);
    batch.seq_id   = (llama_seq_id **) malloc(sizeof(llama_seq_id *) * (n_tokens_alloc + 1));
    for (int i = 0; i < n_tokens_alloc; ++i) {
        batch.seq_id[i] = (llama_seq_id *) malloc(sizeof(llama_seq_id) * n_seq_max);
    }
    batch.seq_id[n_tokens_alloc] = nullptr;

    batch.logits   = (int8_t *)        malloc(sizeof(int8_t)         * n_tokens_alloc);

    return batch;
}

void llama_batch_free(struct llama_batch batch) {
    if (batch.token)    free(batch.token);
    if (batch.embd)     free(batch.embd);
    if (batch.pos)      free(batch.pos);
    if (batch.n_seq_id) free(batch.n_seq_id);
    if (batch.seq_id) {
        for (int i = 0; batch.seq_id[i] != nullptr; ++i) {
            free(batch.seq_id[i]);
        }
        free(batch.seq_id);
    }
    if (batch.logits)   free(batch.logits);
}
