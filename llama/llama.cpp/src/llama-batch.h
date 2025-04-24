#pragma once

#include "llama.h"

#include <array>
#include <vector>

// very similar to llama_batch,
// but has more metadata about sequences
struct llama_ubatch {
    bool equal_seqs;
    // TODO: whole_seqs for embeddings?

    uint32_t n_tokens; // total tokens (n_seq_tokens * n_seqs)
    uint32_t n_seq_tokens; // tokens per sequence
    uint32_t n_seqs;

    llama_token  *  token;    // [n_tokens]
    float        *  embd;     // [n_embd, n_tokens]
    llama_pos    *  pos;      // [n_tokens]
    int32_t      *  n_seq_id; // [n_seqs]
    llama_seq_id ** seq_id;   // [n_seqs]
    int8_t       *  output;   // [n_tokens]
};

struct llama_sbatch_seq {
    int32_t n_seq_id;

    llama_seq_id * seq_id;

    size_t offset;
    size_t length;
};

// sequence-length-aware batch splitting
struct llama_sbatch {
    // tokens left in this batch
    size_t n_tokens;

    size_t n_embd;

    bool logits_all; // TODO: remove once lctx.logits_all is removed too

    // sorted indices into the batch
    std::vector<int64_t> ids;
    // batch indices of the output
    std::vector<int64_t> out_ids;
    std::vector<llama_sbatch_seq> seq;

    const llama_batch * batch = nullptr;

    // buffers for the ubatch
    std::vector<llama_token>    ubatch_token;
    std::vector<float>          ubatch_embd;
    std::vector<llama_pos>      ubatch_pos;
    std::vector<int32_t>        ubatch_n_seq_id;
    std::vector<llama_seq_id *> ubatch_seq_id;
    std::vector<int8_t>         ubatch_output;

    llama_ubatch reserve_ubatch(size_t n_ubatch, bool has_embd = false);

    void add_seq_to_ubatch(llama_ubatch & ubatch, llama_sbatch_seq & seq, size_t length);

    // simple split, unknown number of sequences of unequal lengths
    llama_ubatch split_simple(size_t n_ubatch);

    // make batches of equal-length sequences
    llama_ubatch split_equal(size_t n_ubatch);

    // sequence-wise split
    llama_ubatch split_seq(size_t n_ubatch);

    void from_batch(const llama_batch & batch, size_t n_embd, bool simple_split = false, bool logits_all = false);
};

// temporary allocate memory for the input batch if needed
struct llama_batch_allocr {
    struct llama_batch batch;

    std::array<llama_seq_id, 1> seq_id_0 = { 0 }; // default sequence id
    std::vector<llama_pos>      pos;
    std::vector<int32_t>        n_seq_id;
    std::vector<llama_seq_id *> seq_id;
    std::vector<int8_t>         logits;

    // optionally fulfill the batch returned by llama_batch_get_one
    llama_batch_allocr(struct llama_batch in_batch, llama_pos p0);
};
