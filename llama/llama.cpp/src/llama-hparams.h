#pragma once

#include "llama.h"

#include <array>

// bump if necessary
#define LLAMA_MAX_LAYERS  512
#define LLAMA_MAX_EXPERTS 512 // Qwen3 Next

enum llama_expert_gating_func_type {
    LLAMA_EXPERT_GATING_FUNC_TYPE_NONE           = 0,
    LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX        = 1,
    LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID        = 2,
    LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX_WEIGHT = 3, // applied to the router weights instead of the logits
};

enum llama_swa_type {
    LLAMA_SWA_TYPE_NONE      = 0,
    LLAMA_SWA_TYPE_STANDARD  = 1,
    LLAMA_SWA_TYPE_CHUNKED   = 2,
    LLAMA_SWA_TYPE_SYMMETRIC = 3,
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
    bool no_alloc;
    bool rope_finetuned;
    bool use_par_res;
    bool swin_norm;

    uint32_t n_ctx_train; // context size the model was trained on
    uint32_t n_embd;
    uint32_t n_embd_features = 0;
    uint32_t n_layer;
    int32_t n_layer_kv_from_start = -1; // if non-negative, the first n_layer_kv_from_start layers have KV cache
    uint32_t n_rot;
    uint32_t n_embd_head_k; // dimension of keys (d_k). d_q is assumed to be the same, but there are n_head q heads, and only n_head_kv k-v heads
    uint32_t n_embd_head_v; // dimension of values (d_v) aka n_embd_head
    uint32_t n_expert = 0;
    uint32_t n_expert_used = 0;
    uint32_t n_rel_attn_bkts = 0;

    // note: deepseek2 using MLA converts into MQA with larger heads, then decompresses to MHA
    uint32_t n_embd_head_k_mla = 0;
    uint32_t n_embd_head_v_mla = 0;

    // for WavTokenizer
    struct llama_hparams_posnet   posnet;
    struct llama_hparams_convnext convnext;

    uint32_t n_shortconv_l_cache  = 0;

    std::array<uint32_t, LLAMA_MAX_LAYERS> n_head_arr;
    std::array<uint32_t, LLAMA_MAX_LAYERS> n_head_kv_arr;
    std::array<uint32_t, LLAMA_MAX_LAYERS> n_ff_arr;

    std::array<std::array<uint32_t, LLAMA_MAX_LAYERS>, 4> n_bskcn_arr = {};

    uint32_t n_layer_dense_lead = 0;
    uint32_t n_lora_q           = 0;
    uint32_t n_lora_kv          = 0;
    uint32_t n_ff_exp           = 0;
    uint32_t n_ff_shexp         = 0;
    uint32_t n_ff_chexp         = 0;
    uint32_t n_expert_shared    = 0;
    uint32_t n_norm_groups      = 0;
    uint32_t n_expert_groups    = 0;
    uint32_t n_group_used       = 0;
    uint32_t n_group_experts    = 0;

    float    expert_group_scale   = 0.05f;
    float    expert_weights_scale = 0.0f;
    bool     expert_weights_norm  = false;
    uint32_t expert_gating_func   = LLAMA_EXPERT_GATING_FUNC_TYPE_NONE;
    uint32_t moe_every_n_layers   = 0;
    uint32_t nextn_predict_layers = 0;

    float f_norm_eps;
    float f_norm_rms_eps;
    float f_norm_group_eps;

    float f_attn_logit_softcapping   = 50.0f;
    float f_router_logit_softcapping = 30.0f;
    float f_final_logit_softcapping  = 30.0f;

    // for RWKV
    uint32_t rescale_every_n_layers = 0;
    uint32_t time_mix_extra_dim     = 0;
    uint32_t time_decay_extra_dim   = 0;
    uint32_t wkv_head_size          = 0;
    uint32_t token_shift_count      = 2;
    uint32_t n_lora_decay           = 0;
    uint32_t n_lora_iclr            = 0;
    uint32_t n_lora_value_res_mix   = 0;
    uint32_t n_lora_gate            = 0;

    float    rope_attn_factor = 1.0f;
    float    rope_freq_base_train;
    float    rope_freq_base_train_swa;
    float    rope_freq_scale_train;
    float    rope_freq_scale_train_swa;

    uint32_t n_ctx_orig_yarn;
    float    rope_yarn_log_mul = 0.0f;

    float    yarn_ext_factor  = -1.0f;
    float    yarn_attn_factor =  1.0f;
    float    yarn_beta_fast   = 32.0f;
    float    yarn_beta_slow   =  1.0f;

    std::array<int, 4> rope_sections;

    // Sliding Window Attention (SWA)
    llama_swa_type swa_type = LLAMA_SWA_TYPE_NONE;
    // the size of the sliding window (0 - no SWA)
    uint32_t n_swa = 0;
    // if swa_layers[il] == true, then layer il is SWA
    // if swa_layers[il] == false, then layer il is dense (i.e. non-SWA)
    // by default, all layers are dense
    std::array<bool, LLAMA_MAX_LAYERS> swa_layers;

    // for State Space Models
    uint32_t ssm_d_conv  = 0;
    uint32_t ssm_d_inner = 0;
    uint32_t ssm_d_state = 0;
    uint32_t ssm_dt_rank = 0;
    uint32_t ssm_n_group = 0;

    // for hybrid state space models
    std::array<bool, LLAMA_MAX_LAYERS> recurrent_layer_arr;

    bool ssm_dt_b_c_rms = false;

    float f_clamp_kqv      = 0.0f;
    float f_max_alibi_bias = 0.0f;
    float f_logit_scale    = 0.0f;

    // Additional scale factors (Granite/Granite MoE)
    float f_residual_scale  = 0.0f;
    float f_embedding_scale = 0.0f;
    float f_attention_scale = 0.0f;

    // grok-2
    float    f_attn_out_scale = 0.0f;
    uint32_t attn_temp_length = 0;

    bool causal_attn   = true;
    bool use_alibi     = false;
    bool attn_soft_cap = false;
    bool use_kq_norm   = false;

    // for Classifiers
    uint32_t n_cls_out = 1;

    // llama4 smallthinker
    uint32_t n_moe_layer_step        = 0;
    uint32_t n_no_rope_layer_step    = 4;
    uint32_t n_attn_temp_floor_scale = 0;
    float    f_attn_temp_scale       = 0.0f;
    float    f_attn_temp_offset      = 0.0f; // offset position index

    // gemma3n altup
    uint32_t n_altup      = 4; // altup_num_inputs
    uint32_t i_altup_act  = 0; // altup_active_idx
    uint32_t laurel_rank  = 64;
    uint32_t n_embd_altup = 256;

    // needed for sentence-transformers dense layers
    uint32_t dense_2_feat_in  = 0;  // in_features of the 2_Dense
    uint32_t dense_2_feat_out = 0;  // out_features of the 2_Dense
    uint32_t dense_3_feat_in  = 0;  // in_features of the 3_Dense
    uint32_t dense_3_feat_out = 0;  // out_features of the 3_Dense

    // xIELU
    std::array<float, LLAMA_MAX_LAYERS> xielu_alpha_n;
    std::array<float, LLAMA_MAX_LAYERS> xielu_alpha_p;
    std::array<float, LLAMA_MAX_LAYERS> xielu_beta;
    std::array<float, LLAMA_MAX_LAYERS> xielu_eps;

    // qwen3vl deepstack
    uint32_t n_deepstack_layers = 0;

    // needed by encoder-decoder models (e.g. T5, FLAN-T5)
    // ref: https://github.com/ggerganov/llama.cpp/pull/8141
    llama_token dec_start_token_id = LLAMA_TOKEN_NULL;
    uint32_t    dec_n_layer        = 0;

    enum llama_pooling_type      pooling_type            = LLAMA_POOLING_TYPE_NONE;
    enum llama_rope_type         rope_type               = LLAMA_ROPE_TYPE_NONE;
    enum llama_rope_scaling_type rope_scaling_type_train = LLAMA_ROPE_SCALING_TYPE_NONE;

    // this value n_pattern means that every nth layer is dense (i.e. non-SWA)
    // dense_first means whether the pattern is start with a dense layer
    // note that if n_pattern == 0, all layers are SWA
    //           if n_pattern == 1, all layers are dense
    // example 1: n_pattern = 3, dense_first = false
    //   il == 0: swa
    //   il == 1: swa
    //   il == 2: dense
    //   il == 3: swa
    //   il == 4: swa
    //   il == 5: dense
    //   il == 6: swa
    //   etc ...
    // example 2: n_pattern = 2, dense_first = true
    //   il == 0: dense
    //   il == 1: swa
    //   il == 2: dense
    //   il == 3: swa
    //   etc ...
    void set_swa_pattern(uint32_t n_pattern, bool dense_first = false);

    // return true if one of the layers is SWA
    bool is_swa_any() const;

    uint32_t n_head(uint32_t il = 0) const;

    uint32_t n_head_kv(uint32_t il = 0) const;

    uint32_t n_ff(uint32_t il = 0) const;

    uint32_t n_gqa(uint32_t il = 0) const;

    // dimension of main + auxiliary input embeddings
    uint32_t n_embd_inp() const;

    // dimension of key embeddings across all k-v heads
    uint32_t n_embd_k_gqa(uint32_t il = 0) const;

    // dimension of value embeddings across all k-v heads
    uint32_t n_embd_v_gqa(uint32_t il = 0) const;

    // true if any layer has a different n_embd_k_gqa/n_embd_v_gqa
    bool is_n_embd_k_gqa_variable() const;
    bool is_n_embd_v_gqa_variable() const;

    // return the maximum n_embd_k_gqa/n_embd_v_gqa across all layers
    uint32_t n_embd_k_gqa_max() const;
    uint32_t n_embd_v_gqa_max() const;

    // dimension of the rolling state embeddings
    // corresponds to Mamba's conv_states size or RWKV's token_shift states size
    uint32_t n_embd_r() const;

    // dimension of the recurrent state embeddings
    uint32_t n_embd_s() const;

    // whether or not the given layer is recurrent (for hybrid models)
    bool is_recurrent(uint32_t il) const;

    uint32_t n_pos_per_embd() const;

    // Block skip connection
    bool n_bskcn(uint32_t n, uint32_t il) const;

    bool is_swa(uint32_t il) const;

    bool has_kv(uint32_t il) const;

    // number of layers for which has_kv() returns true
    uint32_t n_layer_kv() const;

    // note that this function uses different SWA parameters from those in the hparams
    // TODO: think of a better place for this function
    // TODO: pack the SWA params in a struct?
    static bool is_masked_swa(uint32_t n_swa, llama_swa_type swa_type, llama_pos p0, llama_pos p1);

    bool use_mrope() const;
};

static_assert(std::is_trivially_copyable<llama_hparams>::value, "llama_hparams must be trivially copyable");
