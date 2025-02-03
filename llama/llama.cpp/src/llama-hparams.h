#pragma once

#include "llama.h"

#include <array>

// bump if necessary
#define LLAMA_MAX_LAYERS  512
#define LLAMA_MAX_EXPERTS 256  // DeepSeekV3

enum llama_expert_gating_func_type {
    LLAMA_EXPERT_GATING_FUNC_TYPE_NONE    = 0,
    LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX = 1,
    LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID = 2,
};

struct llama_hparams_posnet {
    uint32_t n_embd;
    uint32_t n_layer;
};

struct llama_hparams_convnext {
    uint32_t n_embd;
    uint32_t n_layer;
};

struct llama_hparams {
    bool vocab_only;
    bool rope_finetuned;
    bool use_par_res;
    bool swin_norm;

    uint32_t n_vocab = 0;
    uint32_t n_ctx_train; // context size the model was trained on
    uint32_t n_embd;
    uint32_t n_embd_features = 0;
    uint32_t n_layer;
    uint32_t n_rot;
    uint32_t n_swa = 0; // sliding window attention (SWA)
    uint32_t n_embd_head_k; // dimension of keys (d_k). d_q is assumed to be the same, but there are n_head q heads, and only n_head_kv k-v heads
    uint32_t n_embd_head_v; // dimension of values (d_v) aka n_embd_head
    uint32_t n_expert = 0;
    uint32_t n_expert_used = 0;
    uint32_t n_vocab_type = 0; // for BERT-style token types
    uint32_t n_rel_attn_bkts = 0;

    // for WavTokenizer
    struct llama_hparams_posnet   posnet;
    struct llama_hparams_convnext convnext;

    std::array<uint32_t, LLAMA_MAX_LAYERS> n_head_arr;
    std::array<uint32_t, LLAMA_MAX_LAYERS> n_head_kv_arr;
    std::array<uint32_t, LLAMA_MAX_LAYERS> n_ff_arr;

    std::array<std::array<uint32_t, LLAMA_MAX_LAYERS>, 4> n_bskcn_arr = {};
    std::array<uint32_t, LLAMA_MAX_LAYERS> cross_attn_layers;

    uint32_t n_layer_dense_lead = 0;
    uint32_t n_lora_q           = 0;
    uint32_t n_lora_kv          = 0;
    uint32_t n_ff_exp           = 0;
    uint32_t n_ff_shexp         = 0;
    uint32_t n_expert_shared    = 0;
    uint32_t n_norm_groups      = 0;

    float    expert_weights_scale = 0.0;
    bool     expert_weights_norm  = false;
    uint32_t expert_gating_func   = LLAMA_EXPERT_GATING_FUNC_TYPE_NONE;

    float f_norm_eps;
    float f_norm_rms_eps;
    float f_norm_group_eps;

    float f_attn_logit_softcapping  = 50.0f;
    float f_final_logit_softcapping = 30.0f;

    // for RWKV
    uint32_t rescale_every_n_layers = 0;
    uint32_t time_mix_extra_dim     = 0;
    uint32_t time_decay_extra_dim   = 0;
    uint32_t wkv_head_size          = 0;

    float    rope_attn_factor = 1.0f;
    float    rope_freq_base_train;
    float    rope_freq_scale_train;
    uint32_t n_ctx_orig_yarn;
    float    rope_yarn_log_mul;

    std::array<int, 4> rope_sections;

    // for State Space Models
    uint32_t ssm_d_conv  = 0;
    uint32_t ssm_d_inner = 0;
    uint32_t ssm_d_state = 0;
    uint32_t ssm_dt_rank = 0;

    bool ssm_dt_b_c_rms = false;

    float f_clamp_kqv      = 0.0f;
    float f_max_alibi_bias = 0.0f;
    float f_logit_scale    = 0.0f;

    // Additional scale factors (Granite/Granite MoE)
    float f_residual_scale  = 0.0f;
    float f_embedding_scale = 0.0f;
    float f_attention_scale = 0.0f;

    bool causal_attn   = true;
    bool use_alibi     = false;
    bool attn_soft_cap = false;

    // needed by encoder-decoder models (e.g. T5, FLAN-T5)
    // ref: https://github.com/ggerganov/llama.cpp/pull/8141
    llama_token dec_start_token_id = LLAMA_TOKEN_NULL;

    enum llama_pooling_type      pooling_type            = LLAMA_POOLING_TYPE_NONE;
    enum llama_rope_type         rope_type               = LLAMA_ROPE_TYPE_NONE;
    enum llama_rope_scaling_type rope_scaling_type_train = LLAMA_ROPE_SCALING_TYPE_NONE;

    uint32_t n_head(uint32_t il = 0) const;

    uint32_t n_head_kv(uint32_t il = 0) const;

    uint32_t n_ff(uint32_t il = 0) const;

    uint32_t n_gqa(uint32_t il = 0) const;

    // dimension of key embeddings across all k-v heads
    uint32_t n_embd_k_gqa(uint32_t il = 0) const;

    // dimension of value embeddings across all k-v heads
    uint32_t n_embd_v_gqa(uint32_t il = 0) const;

    // dimension of the rolling state embeddings
    // corresponds to Mamba's conv_states size or RWKV's token_shift states size
    uint32_t n_embd_k_s() const;

    // dimension of the recurrent state embeddings
    uint32_t n_embd_v_s() const;

    // Block skip connection
    bool n_bskcn(uint32_t n, uint32_t il) const;

    // cross attention layers   
    bool cross_attention_layers(uint32_t il) const;
};

static_assert(std::is_trivially_copyable<llama_hparams>::value, "llama_hparams must be trivially copyable");

