#pragma once

#include "llama-arch.h"
#include "llama-hparams.h"
#include "llama-adapter.h"

#include <cstdint>
#include <vector>
#include <memory>
#include <set>
#include <functional>

struct ggml_cgraph;
struct ggml_context;
struct ggml_tensor;

struct llama_ubatch;
struct llama_cparams;

class llama_memory_i;
class llama_kv_cache_unified;

// certain models (typically multi-modal) can produce different types of graphs
enum llm_graph_type {
    LLM_GRAPH_TYPE_DEFAULT,
    LLM_GRAPH_TYPE_ENCODER,
    LLM_GRAPH_TYPE_DECODER,
};

enum llm_ffn_op_type {
    LLM_FFN_SILU,
    LLM_FFN_GELU,
    LLM_FFN_RELU,
    LLM_FFN_RELU_SQR,
    LLM_FFN_SWIGLU,
};

enum llm_ffn_gate_type {
    LLM_FFN_SEQ,
    LLM_FFN_PAR, // ffn_gate is parallel to ffn_up
};

enum llm_norm_type {
    LLM_NORM,
    LLM_NORM_RMS,
    LLM_NORM_GROUP,
};

// TODO: tmp - need something better to pass the data from the encoder to the decoder
struct llama_cross {
    // the output embeddings from the encoder as a ggml tensor
    // TODO: this needs more work to be correct, for now copy the embeddings data to host memory
    //       ref: https://github.com/ggml-org/llama.cpp/pull/11213#discussion_r1969892524
    //ggml_tensor * t_embd = nullptr;

    int64_t n_embd = 0;
    int64_t n_enc  = 0;

    // embeddings data copied to host memory (tmp)
    std::vector<float> v_embd;

    // needed to construct the cross-attention mask in the decoder
    std::vector<std::set<llama_seq_id>> seq_ids_enc;
};

//
// llm_graph_input
//

class llm_graph_input_i {
public:
    virtual ~llm_graph_input_i() = default;

    virtual void set_input(const llama_ubatch * ubatch) = 0;
};

using llm_graph_input_ptr = std::unique_ptr<llm_graph_input_i>;


class llm_graph_input_embd : public llm_graph_input_i {
public:
    llm_graph_input_embd()          = default;
    virtual ~llm_graph_input_embd() = default;

    void set_input(const llama_ubatch * ubatch) override;

    ggml_tensor * tokens = nullptr; // I32 [n_batch]
    ggml_tensor * embd   = nullptr; // F32 [n_embd, n_batch]
};

class llm_graph_input_pos : public llm_graph_input_i {
public:
    llm_graph_input_pos(int64_t n_pos_per_embd) : n_pos_per_embd(n_pos_per_embd) {}
    virtual ~llm_graph_input_pos() = default;

    void set_input(const llama_ubatch * ubatch) override;

    ggml_tensor * pos = nullptr; // I32 [n_batch]

    const int64_t n_pos_per_embd = 1;
};

// temperature tuning, used by llama4
class llm_graph_input_attn_temp : public llm_graph_input_i {
public:
    llm_graph_input_attn_temp(uint32_t n_attn_temp_floor_scale, float f_attn_temp_scale)
        : n_attn_temp_floor_scale(n_attn_temp_floor_scale), f_attn_temp_scale(f_attn_temp_scale) {}
    virtual ~llm_graph_input_attn_temp() = default;

    void set_input(const llama_ubatch * ubatch) override;

    ggml_tensor * attn_scale = nullptr; // F32 [n_batch]

    const uint32_t n_attn_temp_floor_scale;
    const float    f_attn_temp_scale;
};

class llm_graph_input_pos_bucket : public llm_graph_input_i {
public:
    llm_graph_input_pos_bucket(const llama_hparams & hparams) : hparams(hparams) {}
    virtual ~llm_graph_input_pos_bucket() = default;

    void set_input(const llama_ubatch * ubatch) override;

    ggml_tensor * pos_bucket = nullptr; // I32 [n_batch, n_batch]

    const llama_hparams & hparams;
};

class llm_graph_input_pos_bucket_kv : public llm_graph_input_i {
public:
    llm_graph_input_pos_bucket_kv(
            const llama_hparams & hparams,
            const llama_kv_cache_unified * kv_self) : hparams(hparams), kv_self(kv_self) {}
    virtual ~llm_graph_input_pos_bucket_kv() = default;

    void set_input(const llama_ubatch * ubatch) override;

    ggml_tensor * pos_bucket = nullptr; // I32 [n_kv, n_batch]

    const llama_hparams & hparams;
    const llama_kv_cache_unified * kv_self;
};

class llm_graph_input_out_ids : public llm_graph_input_i {
public:
    llm_graph_input_out_ids(
            const llama_hparams & hparams,
            const llama_cparams & cparams,
            int32_t n_outputs) : hparams(hparams), cparams(cparams), n_outputs(n_outputs) {}
    virtual ~llm_graph_input_out_ids() = default;

    void set_input(const llama_ubatch * ubatch) override;

    ggml_tensor * out_ids; // I32 [n_outputs]

    const llama_hparams & hparams;
    const llama_cparams & cparams;

    const int32_t n_outputs;
};

class llm_graph_input_mean : public llm_graph_input_i {
public:
    llm_graph_input_mean(const llama_cparams & cparams) : cparams(cparams) {}
    virtual ~llm_graph_input_mean() = default;

    void set_input(const llama_ubatch * ubatch) override;

    ggml_tensor * mean; // F32 [n_batch, n_batch]

    const llama_cparams & cparams;
};

class llm_graph_input_cls : public llm_graph_input_i {
public:
    llm_graph_input_cls(const llama_cparams & cparams) : cparams(cparams) {}
    virtual ~llm_graph_input_cls() = default;

    void set_input(const llama_ubatch * ubatch) override;

    ggml_tensor * cls; // I32 [n_batch]

    const llama_cparams & cparams;
};

class llm_graph_input_s_copy : public llm_graph_input_i {
public:
    llm_graph_input_s_copy(const llama_kv_cache_unified * kv_self) : kv_self(kv_self) {}
    virtual ~llm_graph_input_s_copy() = default;

    void set_input(const llama_ubatch * ubatch) override;

    ggml_tensor * s_copy; // I32 [kv_size]

    const llama_kv_cache_unified * kv_self;
};

class llm_graph_input_s_mask : public llm_graph_input_i {
public:
    llm_graph_input_s_mask(const llama_kv_cache_unified * kv_self) : kv_self(kv_self) {}
    virtual ~llm_graph_input_s_mask() = default;

    void set_input(const llama_ubatch * ubatch) override;

    ggml_tensor * s_mask; // F32 [1, n_kv]

    const llama_kv_cache_unified * kv_self;
};

class llm_graph_input_cross_embd : public llm_graph_input_i {
public:
    llm_graph_input_cross_embd(
            const llama_cross * cross) : cross(cross) {}
    virtual ~llm_graph_input_cross_embd() = default;

    void set_input(const llama_ubatch * ubatch) override;

    ggml_tensor * cross_embd; // F32 [n_embd, n_outputs_enc]

    const llama_cross * cross;
};

class llm_graph_input_attn_no_cache : public llm_graph_input_i {
public:
    llm_graph_input_attn_no_cache(const llama_hparams & hparams, const llama_cparams & cparams) :
        hparams(hparams),
        cparams(cparams) {
    }
    ~llm_graph_input_attn_no_cache() = default;

    void set_input(const llama_ubatch * ubatch) override;

    ggml_tensor * get_kq_mask() const { return kq_mask_cnv; }

    ggml_tensor * kq_mask     = nullptr; // F32 [n_tokens, n_batch]
    ggml_tensor * kq_mask_cnv = nullptr; //     [n_tokens, n_batch]

    const llama_hparams & hparams;
    const llama_cparams & cparams;
};

class llm_graph_input_attn_kv_unified : public llm_graph_input_i {
public:
    llm_graph_input_attn_kv_unified(
            const llama_hparams & hparams,
            const llama_cparams & cparams,
            const llama_kv_cache_unified * kv_self) :
        hparams(hparams),
        cparams(cparams),
        kv_self(kv_self) {
    }
    ~llm_graph_input_attn_kv_unified() = default;

    void set_input(const llama_ubatch * ubatch) override;

    ggml_tensor * get_kq_mask()     const { return self_kq_mask_cnv; }
    ggml_tensor * get_kq_mask_swa() const { return self_kq_mask_swa_cnv; }

    ggml_tensor * self_kq_mask         = nullptr; // F32 [n_kv, n_batch]
    ggml_tensor * self_kq_mask_cnv     = nullptr; //     [n_kv, n_batch]
    ggml_tensor * self_kq_mask_swa     = nullptr; // F32 [n_kv, n_batch]
    ggml_tensor * self_kq_mask_swa_cnv = nullptr; //     [n_kv, n_batch]

    const llama_hparams & hparams;
    const llama_cparams & cparams;

    const llama_kv_cache_unified * kv_self;
};

class llm_graph_input_attn_cross : public llm_graph_input_i {
public:
    llm_graph_input_attn_cross(const llama_cross * cross) : cross(cross) {}
    ~llm_graph_input_attn_cross() = default;

    void set_input(const llama_ubatch * ubatch) override;

    ggml_tensor * get_kq_mask_cross() const { return cross_kq_mask_cnv; }

    ggml_tensor * cross_kq_mask     = nullptr; // F32 [n_outputs_enc, n_batch]
    ggml_tensor * cross_kq_mask_cnv = nullptr; // F32 [n_outputs_enc, n_batch]

    const llama_cross * cross = nullptr;
};

//
// llm_graph_result
//

// these objects deliver the result from the graph build process back to the llama_context
// note that the input tensors created for the graph are referenced here - the goal is to be able to populate their
//   specific data, by calling the set_inputs() method
// along with the input tensors, the object also provides commonly used outputs tensors, such as logits, embeddings, etc.
//   these are used by the llama_context to extact the relevant data, based on the compute parameters

class llm_graph_result_i {
public:
    virtual ~llm_graph_result_i() = default;

    virtual ggml_tensor * get_logits()      = 0;
    virtual ggml_tensor * get_embd()        = 0;
    virtual ggml_tensor * get_embd_pooled() = 0;

    virtual void set_inputs(const llama_ubatch * ubatch) = 0;
};

using llm_graph_result_ptr = std::unique_ptr<llm_graph_result_i>;


class llm_graph_result : public llm_graph_result_i {
public:
    virtual ~llm_graph_result() = default;

    ggml_tensor * get_logits()      override { return t_logits; }
    ggml_tensor * get_embd()        override { return t_embd; }
    ggml_tensor * get_embd_pooled() override { return t_embd_pooled; }

    void set_inputs(const llama_ubatch * ubatch) override {
        for (auto & input : inputs) {
            input->set_input(ubatch);
        }
    }

    llm_graph_input_i * add_input(llm_graph_input_ptr input) {
        inputs.emplace_back(std::move(input));
        return inputs.back().get();
    }

    // important graph nodes
    ggml_tensor * t_logits      = nullptr;
    ggml_tensor * t_embd        = nullptr;
    ggml_tensor * t_embd_pooled = nullptr;

    std::vector<llm_graph_input_ptr> inputs;
};

//
// llm_graph_context
//

// callback that allows us to apply custom logic to each tensor (e.g. ggml-alloc, offloading, etc.)
using llm_graph_cb = std::function<void(const llama_ubatch & ubatch, ggml_tensor * cur, const char * name, int il)>;

struct llm_graph_params {
    ggml_context * ctx;

    const llm_arch arch;

    const llama_hparams & hparams;
    const llama_cparams & cparams;
    const llama_ubatch  & ubatch;

    ggml_backend_sched * sched;
    ggml_backend * backend_cpu;

    const llama_adapter_cvec  * cvec;
    const llama_adapter_loras * loras;
    const llama_memory_i      * memory;
    const llama_cross         * cross;

    int32_t n_outputs;

    const llm_graph_cb & cb;
};

struct llm_graph_context {
    const llm_arch arch;

    const llama_hparams & hparams;
    const llama_cparams & cparams;
    const llama_ubatch  & ubatch;

    const int64_t n_embd;
    const int64_t n_layer;
    const int64_t n_rot;
    const int64_t n_ctx;       // user-specified context size (can be different from n_ctx_train)
    const int64_t n_ctx_per_seq;
    const int64_t n_head;
    const int64_t n_head_kv;
    const int64_t n_embd_head_k;
    const int64_t n_embd_k_gqa;
    const int64_t n_embd_head_v;
    const int64_t n_embd_v_gqa;
    const int64_t n_expert;
    const int64_t n_expert_used;

    const float freq_base;
    const float freq_scale;
    const float ext_factor;
    const float attn_factor;
    const float beta_fast;
    const float beta_slow;
    const float norm_eps;
    const float norm_rms_eps;

    const int32_t n_tokens;
    const int32_t n_outputs;
    const int32_t n_ctx_orig; // yarn

    const enum llama_pooling_type pooling_type;
    const enum llama_rope_type    rope_type;

    ggml_context * ctx0 = nullptr;

    ggml_backend_sched * sched;

    ggml_backend * backend_cpu; // TODO: needed by build_attn_mha, figure out a way to remove?

    const llama_adapter_cvec  * cvec;
    const llama_adapter_loras * loras;
    const llama_memory_i      * memory;
    const llama_cross         * cross;

    const llm_graph_cb & cb_func;

    std::unique_ptr<llm_graph_result> res;

    llm_graph_context(const llm_graph_params & params);

    int64_t n_pos_per_embd() const;

    void cb(ggml_tensor * cur, const char * name, int il) const;

    //
    // common
    //

    ggml_tensor * build_cvec(
             ggml_tensor * cur,
                     int   il) const;

    // do mat_mul, while optionally apply lora
    ggml_tensor * build_lora_mm(
              ggml_tensor * w,
              ggml_tensor * cur) const;

    // do mat_mul_id, while optionally apply lora
    ggml_tensor * build_lora_mm_id(
              ggml_tensor * w,   // ggml_tensor * as
              ggml_tensor * cur, // ggml_tensor * b
              ggml_tensor * ids) const;

    ggml_tensor * build_norm(
             ggml_tensor * cur,
             ggml_tensor * mw,
             ggml_tensor * mb,
           llm_norm_type   type,
                     int   il) const;

    ggml_tensor * build_ffn(
             ggml_tensor * cur,
             ggml_tensor * up,
             ggml_tensor * up_b,
             ggml_tensor * up_s,
             ggml_tensor * gate,
             ggml_tensor * gate_b,
             ggml_tensor * gate_s,
             ggml_tensor * down,
             ggml_tensor * down_b,
             ggml_tensor * down_s,
             ggml_tensor * act_scales,
         llm_ffn_op_type   type_op,
       llm_ffn_gate_type   type_gate,
                     int   il) const;

    ggml_tensor * build_moe_ffn(
             ggml_tensor * cur,
             ggml_tensor * gate_inp,
             ggml_tensor * up_exps,
             ggml_tensor * gate_exps,
             ggml_tensor * down_exps,
             ggml_tensor * exp_probs_b,
                 int64_t   n_expert,
                 int64_t   n_expert_used,
         llm_ffn_op_type   type_op,
                    bool   norm_w,
                    bool   scale_w,
                   float   w_scale,
            llama_expert_gating_func_type gating_op,
                     int   il) const;

    //
    // inputs
    //

    ggml_tensor * build_inp_embd(ggml_tensor * tok_embd) const;
    ggml_tensor * build_inp_pos() const;
    ggml_tensor * build_inp_attn_scale() const;
    ggml_tensor * build_inp_out_ids() const;
    ggml_tensor * build_inp_mean() const;
    ggml_tensor * build_inp_cls() const;
    ggml_tensor * build_inp_s_copy() const;
    ggml_tensor * build_inp_s_mask() const;

    ggml_tensor * build_inp_cross_embd() const;
    ggml_tensor * build_inp_pos_bucket_enc() const;
    ggml_tensor * build_inp_pos_bucket_dec() const;
    ggml_tensor * build_pos_bias(ggml_tensor * pos_bucket, ggml_tensor * attn_rel_b) const;

    //
    // attention
    //

    ggml_tensor * build_attn_mha(
             ggml_cgraph * gf,
             ggml_tensor * q,     // [n_embd_head_q, n_tokens, n_head_q]
             ggml_tensor * k,     // [n_embd_head_k, n_tokens, n_head_k]
             ggml_tensor * v,     // [n_embd_head_v, n_tokens, n_head_v] (v_trans == false)
             ggml_tensor * kq_b,
             ggml_tensor * kq_mask,
             ggml_tensor * v_mla, // [n_embd_head_v_mla, n_embd_head_v, n_head_v]
                    bool   v_trans,
                   float   kq_scale) const;

    llm_graph_input_attn_no_cache * build_attn_inp_no_cache() const;

    ggml_tensor * build_attn(
            llm_graph_input_attn_no_cache * inp,
            ggml_cgraph * gf,
            ggml_tensor * wo,
            ggml_tensor * wo_b,
            ggml_tensor * q_cur, // [n_embd_head_q, n_head_q, n_tokens]
            ggml_tensor * k_cur, // [n_embd_head_k, n_head_k, n_tokens]
            ggml_tensor * v_cur, // [n_embd_head_v, n_head_v, n_tokens]
            ggml_tensor * kq_b,
            ggml_tensor * v_mla, // [n_embd_head_v_mla, n_embd_head_v, n_head_v]
                  float   kq_scale,
                    int   il) const;

    llm_graph_input_attn_kv_unified * build_attn_inp_kv_unified() const;

    ggml_tensor * build_attn(
            llm_graph_input_attn_kv_unified * inp,
            ggml_cgraph * gf,
            ggml_tensor * wo,
            ggml_tensor * wo_b,
            ggml_tensor * q_cur, // [n_embd_head_q, n_head_q, n_tokens]
            ggml_tensor * k_cur, // [n_embd_head_k, n_head_k, n_tokens]
            ggml_tensor * v_cur, // [n_embd_head_v, n_head_v, n_tokens]
            ggml_tensor * kq_b,
            ggml_tensor * v_mla, // [n_embd_head_v_mla, n_embd_head_v, n_head_v]
                  float   kq_scale,
                    int   il) const;

    llm_graph_input_attn_cross * build_attn_inp_cross() const;

    ggml_tensor * build_attn(
            llm_graph_input_attn_cross * inp,
            ggml_cgraph * gf,
            ggml_tensor * wo,
            ggml_tensor * wo_b,
            ggml_tensor * q_cur, // [n_embd_head_q, n_head_q, n_tokens]
            ggml_tensor * k_cur, // [n_embd_head_k, n_head_k, n_tokens]
            ggml_tensor * v_cur, // [n_embd_head_v, n_head_v, n_tokens]
            ggml_tensor * kq_b,
            ggml_tensor * v_mla, // [n_embd_head_v_mla, n_embd_head_v, n_head_v]
                  float   kq_scale,
                    int   il) const;

    //
    // recurrent
    //

    ggml_tensor * build_copy_mask_state(
             ggml_cgraph * gf,
             ggml_tensor * s,
             ggml_tensor * state_copy,
             ggml_tensor * state_mask,
                 int32_t   n_state,
                 int32_t   n_seqs) const;

    ggml_tensor * build_rwkv_token_shift_load(
             ggml_cgraph * gf,
             ggml_tensor * state_copy,
             ggml_tensor * state_mask,
      const llama_ubatch & ubatch,
                     int   il) const;

    ggml_tensor * build_rwkv_token_shift_store(
             ggml_tensor * token_shift,
      const llama_ubatch & ubatch,
                     int   il) const;

    //
    // pooling
    //

    void build_pooling(
            ggml_cgraph * gf,
            ggml_tensor * cls,
            ggml_tensor * cls_b,
            ggml_tensor * cls_out,
            ggml_tensor * cls_out_b) const;
};
