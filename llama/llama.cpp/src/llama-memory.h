#pragma once

#include "llama.h"

struct llama_memory_params {
    // kv cache
    ggml_type type_k;
    ggml_type type_v;

    // parameters for other types of memory
    // ...
};

// general concept of LLM memory
// the KV cache is a type of LLM memory, but there can be other types
class llama_memory_i {
public:
    virtual ~llama_memory_i() = default;

    virtual void clear() = 0;

    virtual bool seq_rm  (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1) = 0;
    virtual void seq_cp  (llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) = 0;
    virtual void seq_keep(llama_seq_id seq_id) = 0;
    virtual void seq_add (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, llama_pos delta) = 0;
    virtual void seq_div (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, int d) = 0;

    virtual llama_pos seq_pos_max(llama_seq_id seq_id) const = 0;

    virtual bool get_can_edit() const = 0;
};
