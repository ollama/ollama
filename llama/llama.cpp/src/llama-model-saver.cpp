#include "llama-model-saver.h"

#include "gguf.h"

#include "llama.h"
#include "llama-hparams.h"
#include "llama-model.h"
#include "llama-vocab.h"

#include <string>

llama_model_saver::llama_model_saver(const struct llama_model & model) : model(model), llm_kv(model.arch) {
    gguf_ctx = gguf_init_empty();
}

llama_model_saver::~llama_model_saver() {
    gguf_free(gguf_ctx);
}

void llama_model_saver::add_kv(const enum llm_kv key, const uint32_t value) {
    gguf_set_val_u32(gguf_ctx, llm_kv(key).c_str(), value);
}

void llama_model_saver::add_kv(const enum llm_kv key, const int32_t value) {
    gguf_set_val_i32(gguf_ctx, llm_kv(key).c_str(), value);
}

void llama_model_saver::add_kv(const enum llm_kv key, const float value) {
    gguf_set_val_f32(gguf_ctx, llm_kv(key).c_str(), value);
}

void llama_model_saver::add_kv(const enum llm_kv key, const bool value) {
    gguf_set_val_bool(gguf_ctx, llm_kv(key).c_str(), value);
}

void llama_model_saver::add_kv(const enum llm_kv key, const char * value) {
    gguf_set_val_str(gguf_ctx, llm_kv(key).c_str(), value);
}

[[noreturn]]
void llama_model_saver::add_kv(const enum llm_kv key, const char value) {
    GGML_UNUSED(key);
    GGML_UNUSED(value);
    GGML_ABORT("fatal error"); // this should never be called, only needed to make the template below compile
}

template <typename Container>
void llama_model_saver::add_kv(const enum llm_kv key, const Container & value, const bool per_layer) {
    const size_t n_values = per_layer ? size_t(model.hparams.n_layer) : value.size();
    GGML_ASSERT(n_values <= value.size());

    if (n_values == 0) {
        return;
    }

    if (per_layer) {
        bool all_values_the_same = true;
        for (size_t i = 1; i < n_values; ++i) {
            if (value[i] != value[0]) {
                all_values_the_same = false;
                break;
            }
        }
        if (all_values_the_same) {
            add_kv(key, value[0]);
            return;
        }
    }

    if (std::is_same<typename Container::value_type, uint8_t>::value) {
        gguf_set_arr_data(gguf_ctx, llm_kv(key).c_str(), GGUF_TYPE_UINT8, value.data(), n_values);
    } else if (std::is_same<typename Container::value_type, int8_t>::value) {
        gguf_set_arr_data(gguf_ctx, llm_kv(key).c_str(), GGUF_TYPE_INT8, value.data(), n_values);
    } else if (std::is_same<typename Container::value_type, uint32_t>::value) {
        gguf_set_arr_data(gguf_ctx, llm_kv(key).c_str(), GGUF_TYPE_UINT32, value.data(), n_values);
    } else if (std::is_same<typename Container::value_type, int32_t>::value) {
        gguf_set_arr_data(gguf_ctx, llm_kv(key).c_str(), GGUF_TYPE_INT32, value.data(), n_values);
    } else if (std::is_same<typename Container::value_type, float>::value) {
        gguf_set_arr_data(gguf_ctx, llm_kv(key).c_str(), GGUF_TYPE_FLOAT32, value.data(), n_values);
    } else if (std::is_same<Container, std::string>::value) {
        gguf_set_val_str(gguf_ctx, llm_kv(key).c_str(), reinterpret_cast<const char *>(value.data()));
    } else {
        GGML_ABORT("fatal error");
    }
}

void llama_model_saver::add_kv(const enum llm_kv key, const std::vector<std::string> & value) {
    std::vector<const char *> tmp(value.size());
    for (size_t i = 0; i < value.size(); ++i) {
        tmp[i] = value[i].c_str();
    }
    gguf_set_arr_str(gguf_ctx, llm_kv(key).c_str(), tmp.data(), tmp.size());
}

void llama_model_saver::add_tensor(const struct ggml_tensor * tensor) {
    if (!tensor) {
        return;
    }
    if (gguf_find_tensor(gguf_ctx, tensor->name) >= 0) {
        GGML_ASSERT(std::string(tensor->name) == "rope_freqs.weight"); // FIXME
        return;
    }
    gguf_add_tensor(gguf_ctx, tensor);
}

void llama_model_saver::add_kv_from_model() {
    const llama_hparams & hparams = model.hparams;
    const llama_vocab   & vocab   = model.vocab;

    const int32_t n_vocab = vocab.n_tokens();
    std::vector<std::string> tokens(n_vocab);
    std::vector<float>       scores(n_vocab);
    std::vector<int32_t>     token_types(n_vocab);

    for (int32_t id = 0; id < n_vocab; ++id) {
        const llama_vocab::token_data & token_data = vocab.get_token_data(id);

        tokens[id] = token_data.text;
        scores[id] = token_data.score;

        switch(token_data.attr) {
            case LLAMA_TOKEN_ATTR_UNKNOWN:      token_types[id] = LLAMA_TOKEN_TYPE_UNKNOWN;      break;
            case LLAMA_TOKEN_ATTR_UNUSED:       token_types[id] = LLAMA_TOKEN_TYPE_UNUSED;       break;
            case LLAMA_TOKEN_ATTR_NORMAL:       token_types[id] = LLAMA_TOKEN_TYPE_NORMAL;       break;
            case LLAMA_TOKEN_ATTR_CONTROL:      token_types[id] = LLAMA_TOKEN_TYPE_CONTROL;      break;
            case LLAMA_TOKEN_ATTR_USER_DEFINED: token_types[id] = LLAMA_TOKEN_TYPE_USER_DEFINED; break;
            case LLAMA_TOKEN_ATTR_BYTE:         token_types[id] = LLAMA_TOKEN_TYPE_BYTE;         break;
            case LLAMA_TOKEN_ATTR_UNDEFINED:
            default:                            token_types[id] = LLAMA_TOKEN_TYPE_UNDEFINED;    break;
        }
    }

    // add_kv(LLM_KV_GENERAL_TYPE,                      ???);
    add_kv(LLM_KV_GENERAL_ARCHITECTURE,              model.arch_name());
    // add_kv(LLM_KV_GENERAL_QUANTIZATION_VERSION,      ???);
    // add_kv(LLM_KV_GENERAL_ALIGNMENT,                 ???);
    add_kv(LLM_KV_GENERAL_NAME,                      model.name);
    // add_kv(LLM_KV_GENERAL_AUTHOR,                    ???);
    // add_kv(LLM_KV_GENERAL_VERSION,                   ???);
    // add_kv(LLM_KV_GENERAL_URL,                       ???);
    // add_kv(LLM_KV_GENERAL_DESCRIPTION,               ???);
    // add_kv(LLM_KV_GENERAL_LICENSE,                   ???);
    // add_kv(LLM_KV_GENERAL_SOURCE_URL,                ???);
    // add_kv(LLM_KV_GENERAL_SOURCE_HF_REPO,            ???);

    add_kv(LLM_KV_VOCAB_SIZE,                        vocab.n_tokens());
    add_kv(LLM_KV_CONTEXT_LENGTH,                    hparams.n_ctx_train);
    add_kv(LLM_KV_EMBEDDING_LENGTH,                  hparams.n_embd);
    if (hparams.n_embd_out > 0) {
        add_kv(LLM_KV_EMBEDDING_LENGTH_OUT,          hparams.n_embd_out);
    }
    add_kv(LLM_KV_BLOCK_COUNT,                       hparams.n_layer);
    add_kv(LLM_KV_LEADING_DENSE_BLOCK_COUNT,         hparams.n_layer_dense_lead);
    add_kv(LLM_KV_FEED_FORWARD_LENGTH,               hparams.n_ff_arr, true);
    add_kv(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,        hparams.n_ff_exp);
    add_kv(LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH, hparams.n_ff_exp);
    add_kv(LLM_KV_USE_PARALLEL_RESIDUAL,             hparams.use_par_res);
    // add_kv(LLM_KV_TENSOR_DATA_LAYOUT,                ???);
    add_kv(LLM_KV_EXPERT_COUNT,                      hparams.n_expert);
    add_kv(LLM_KV_EXPERT_USED_COUNT,                 hparams.n_expert_used);
    add_kv(LLM_KV_EXPERT_SHARED_COUNT,               hparams.n_expert_shared);
    add_kv(LLM_KV_EXPERT_WEIGHTS_SCALE,              hparams.expert_weights_scale);
    add_kv(LLM_KV_POOLING_TYPE,                      uint32_t(hparams.pooling_type));
    add_kv(LLM_KV_LOGIT_SCALE,                       hparams.f_logit_scale);
    add_kv(LLM_KV_DECODER_START_TOKEN_ID,            hparams.dec_start_token_id);
    add_kv(LLM_KV_ATTN_LOGIT_SOFTCAPPING,            hparams.f_attn_logit_softcapping);
    add_kv(LLM_KV_FINAL_LOGIT_SOFTCAPPING,           hparams.f_final_logit_softcapping);
    add_kv(LLM_KV_SWIN_NORM,                         hparams.swin_norm);
    add_kv(LLM_KV_RESCALE_EVERY_N_LAYERS,            hparams.rescale_every_n_layers);
    add_kv(LLM_KV_TIME_MIX_EXTRA_DIM,                hparams.time_mix_extra_dim);
    add_kv(LLM_KV_TIME_DECAY_EXTRA_DIM,              hparams.time_decay_extra_dim);
    add_kv(LLM_KV_RESIDUAL_SCALE,                    hparams.f_residual_scale);
    add_kv(LLM_KV_EMBEDDING_SCALE,                   hparams.f_embedding_scale);

    add_kv(LLM_KV_ATTENTION_HEAD_COUNT,              hparams.n_head_arr, true);
    add_kv(LLM_KV_ATTENTION_HEAD_COUNT_KV,           hparams.n_head_kv_arr, true);
    add_kv(LLM_KV_ATTENTION_MAX_ALIBI_BIAS,          hparams.f_max_alibi_bias);
    add_kv(LLM_KV_ATTENTION_CLAMP_KQV,               hparams.f_clamp_kqv);
    add_kv(LLM_KV_ATTENTION_KEY_LENGTH,              hparams.n_embd_head_k);
    add_kv(LLM_KV_ATTENTION_VALUE_LENGTH,            hparams.n_embd_head_v);
    add_kv(LLM_KV_ATTENTION_LAYERNORM_EPS,           hparams.f_norm_eps);
    add_kv(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,       hparams.f_norm_rms_eps);
    add_kv(LLM_KV_ATTENTION_CAUSAL,                  hparams.causal_attn);
    add_kv(LLM_KV_ATTENTION_Q_LORA_RANK,             hparams.n_lora_q);
    add_kv(LLM_KV_ATTENTION_KV_LORA_RANK,            hparams.n_lora_kv);
    add_kv(LLM_KV_ATTENTION_RELATIVE_BUCKETS_COUNT,  hparams.n_rel_attn_bkts);
    add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW,          hparams.n_swa);
    add_kv(LLM_KV_ATTENTION_SCALE,                   hparams.f_attention_scale);

    const float rope_scaling_factor = hparams.rope_freq_scale_train == 1.0f ? 0.0f : 1.0f/hparams.rope_freq_scale_train;

    add_kv(LLM_KV_ROPE_DIMENSION_COUNT,              hparams.n_rot);
    add_kv(LLM_KV_ROPE_FREQ_BASE,                    hparams.rope_freq_base_train);
    // add_kv(LLM_KV_ROPE_SCALE_LINEAR,                 rope_scaling_factor); // old name
    add_kv(LLM_KV_ROPE_SCALING_TYPE,                 llama_rope_scaling_type_name(hparams.rope_scaling_type_train));
    add_kv(LLM_KV_ROPE_SCALING_FACTOR,               rope_scaling_factor);
    add_kv(LLM_KV_ROPE_SCALING_ATTN_FACTOR,          hparams.rope_attn_factor);
    add_kv(LLM_KV_ROPE_SCALING_ORIG_CTX_LEN,         hparams.n_ctx_orig_yarn);
    add_kv(LLM_KV_ROPE_SCALING_FINETUNED,            hparams.rope_finetuned);
    add_kv(LLM_KV_ROPE_SCALING_YARN_LOG_MUL,         hparams.rope_yarn_log_mul);

    // TODO: implement split file support
    // add_kv(LLM_KV_SPLIT_NO,                          ???);
    // add_kv(LLM_KV_SPLIT_COUNT,                       ???);
    // add_kv(LLM_KV_SPLIT_TENSORS_COUNT,               ???);

    add_kv(LLM_KV_SSM_INNER_SIZE,                    hparams.ssm_d_inner);
    add_kv(LLM_KV_SSM_CONV_KERNEL,                   hparams.ssm_d_conv);
    add_kv(LLM_KV_SSM_STATE_SIZE,                    hparams.ssm_d_state);
    add_kv(LLM_KV_SSM_TIME_STEP_RANK,                hparams.ssm_dt_rank);
    add_kv(LLM_KV_SSM_DT_B_C_RMS,                    hparams.ssm_dt_b_c_rms);

    add_kv(LLM_KV_WKV_HEAD_SIZE,                     hparams.wkv_head_size);

    add_kv(LLM_KV_TOKENIZER_MODEL,                   vocab.get_tokenizer_model());
    add_kv(LLM_KV_TOKENIZER_PRE,                     vocab.get_tokenizer_pre());
    add_kv(LLM_KV_TOKENIZER_LIST,                    tokens);
    add_kv(LLM_KV_TOKENIZER_TOKEN_TYPE,              token_types);
    add_kv(LLM_KV_TOKENIZER_TOKEN_TYPE_COUNT,        vocab.n_token_types());
    add_kv(LLM_KV_TOKENIZER_SCORES,                  scores);
    add_kv(LLM_KV_TOKENIZER_MERGES,                  vocab.get_bpe_merges());
    // FIXME llama_token is type i32 but when reading in a GGUF file u32 is expected, not an issue for writing though
    add_kv(LLM_KV_TOKENIZER_BOS_ID,                  uint32_t(vocab.token_bos()));
    add_kv(LLM_KV_TOKENIZER_EOS_ID,                  uint32_t(vocab.token_eos()));
    add_kv(LLM_KV_TOKENIZER_EOT_ID,                  uint32_t(vocab.token_eot()));
    add_kv(LLM_KV_TOKENIZER_EOM_ID,                  uint32_t(vocab.token_eom()));
    add_kv(LLM_KV_TOKENIZER_UNK_ID,                  uint32_t(vocab.token_unk()));
    add_kv(LLM_KV_TOKENIZER_SEP_ID,                  uint32_t(vocab.token_sep()));
    add_kv(LLM_KV_TOKENIZER_PAD_ID,                  uint32_t(vocab.token_pad()));
    // add_kv(LLM_KV_TOKENIZER_CLS_ID,                  uint32_t(vocab.token_bos())); // deprecated
    // add_kv(LLM_KV_TOKENIZER_MASK_ID,                 ???);
    add_kv(LLM_KV_TOKENIZER_ADD_BOS,                 vocab.get_add_bos());
    add_kv(LLM_KV_TOKENIZER_ADD_EOS,                 vocab.get_add_eos());
    add_kv(LLM_KV_TOKENIZER_ADD_SEP,                 vocab.get_add_sep());
    add_kv(LLM_KV_TOKENIZER_ADD_PREFIX,              vocab.get_add_space_prefix());
    add_kv(LLM_KV_TOKENIZER_REMOVE_EXTRA_WS,         vocab.get_remove_extra_whitespaces());
    add_kv(LLM_KV_TOKENIZER_PRECOMPILED_CHARSMAP,    vocab.get_precompiled_charsmap());
    // add_kv(LLM_KV_TOKENIZER_HF_JSON,                 ???);
    // add_kv(LLM_KV_TOKENIZER_RWKV,                    ???);
    add_kv(LLM_KV_TOKENIZER_FIM_PRE_ID,              uint32_t(vocab.token_fim_pre()));
    add_kv(LLM_KV_TOKENIZER_FIM_SUF_ID,              uint32_t(vocab.token_fim_suf()));
    add_kv(LLM_KV_TOKENIZER_FIM_MID_ID,              uint32_t(vocab.token_fim_mid()));
    add_kv(LLM_KV_TOKENIZER_FIM_PAD_ID,              uint32_t(vocab.token_fim_pad()));
    add_kv(LLM_KV_TOKENIZER_FIM_REP_ID,              uint32_t(vocab.token_fim_rep()));
    add_kv(LLM_KV_TOKENIZER_FIM_SEP_ID,              uint32_t(vocab.token_fim_sep()));

    // TODO: implement LoRA support
    // add_kv(LLM_KV_ADAPTER_TYPE,                      ???);
    // add_kv(LLM_KV_ADAPTER_LORA_ALPHA,                ???);

    // deprecated
    // add_kv(LLM_KV_TOKENIZER_PREFIX_ID,               ???);
    // add_kv(LLM_KV_TOKENIZER_SUFFIX_ID,               ???);
    // add_kv(LLM_KV_TOKENIZER_MIDDLE_ID,               ???);
}

void llama_model_saver::add_tensors_from_model() {
    if (std::string(model.output->name) != std::string(model.tok_embd->name)) {
        add_tensor(model.tok_embd); // some models use the same tensor for tok_embd and output
    }
    add_tensor(model.type_embd);
    add_tensor(model.pos_embd);
    add_tensor(model.tok_norm);
    add_tensor(model.tok_norm_b);
    add_tensor(model.output_norm);
    add_tensor(model.output_norm_b);
    add_tensor(model.output);
    add_tensor(model.output_b);
    add_tensor(model.output_norm_enc);
    add_tensor(model.cls);
    add_tensor(model.cls_b);
    add_tensor(model.cls_out);
    add_tensor(model.cls_out_b);

    for (const struct llama_layer & layer : model.layers) {
        for (size_t i = 0; i < sizeof(layer)/sizeof(struct ggml_tensor *); ++i) {
            add_tensor(reinterpret_cast<const struct ggml_tensor * const *>(&layer)[i]);
        }
    }
}

void llama_model_saver::save(const std::string & path_model) {
    gguf_write_to_file(gguf_ctx, path_model.c_str(), false);
}

