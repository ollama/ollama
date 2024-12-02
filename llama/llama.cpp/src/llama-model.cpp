#include "llama-model.h"

#include "llama-impl.h"
#include "llama-model-loader.h"

#include "unicode.h" // TODO: remove

#include <algorithm>
#include <cassert>
#include <functional>
#include <sstream>
#include <stdexcept>

static const size_t kiB = 1024;
static const size_t MiB = 1024*kiB;
static const size_t GiB = 1024*MiB;

const char * llm_type_name(llm_type type) {
    switch (type) {
        case MODEL_14M:           return "14M";
        case MODEL_17M:           return "17M";
        case MODEL_22M:           return "22M";
        case MODEL_33M:           return "33M";
        case MODEL_60M:           return "60M";
        case MODEL_70M:           return "70M";
        case MODEL_80M:           return "80M";
        case MODEL_109M:          return "109M";
        case MODEL_137M:          return "137M";
        case MODEL_160M:          return "160M";
        case MODEL_220M:          return "220M";
        case MODEL_250M:          return "250M";
        case MODEL_270M:          return "270M";
        case MODEL_335M:          return "335M";
        case MODEL_410M:          return "410M";
        case MODEL_450M:          return "450M";
        case MODEL_770M:          return "770M";
        case MODEL_780M:          return "780M";
        case MODEL_0_5B:          return "0.5B";
        case MODEL_1B:            return "1B";
        case MODEL_1_3B:          return "1.3B";
        case MODEL_1_4B:          return "1.4B";
        case MODEL_1_5B:          return "1.5B";
        case MODEL_1_6B:          return "1.6B";
        case MODEL_2B:            return "2B";
        case MODEL_2_8B:          return "2.8B";
        case MODEL_3B:            return "3B";
        case MODEL_4B:            return "4B";
        case MODEL_6B:            return "6B";
        case MODEL_6_9B:          return "6.9B";
        case MODEL_7B:            return "7B";
        case MODEL_8B:            return "8B";
        case MODEL_9B:            return "9B";
        case MODEL_11B:           return "11B";
        case MODEL_12B:           return "12B";
        case MODEL_13B:           return "13B";
        case MODEL_14B:           return "14B";
        case MODEL_15B:           return "15B";
        case MODEL_16B:           return "16B";
        case MODEL_20B:           return "20B";
        case MODEL_30B:           return "30B";
        case MODEL_32B:           return "32B";
        case MODEL_34B:           return "34B";
        case MODEL_35B:           return "35B";
        case MODEL_40B:           return "40B";
        case MODEL_65B:           return "65B";
        case MODEL_70B:           return "70B";
        case MODEL_236B:          return "236B";
        case MODEL_314B:          return "314B";
        case MODEL_671B:          return "671B";
        case MODEL_SMALL:         return "0.1B";
        case MODEL_MEDIUM:        return "0.4B";
        case MODEL_LARGE:         return "0.8B";
        case MODEL_XL:            return "1.5B";
        case MODEL_A1_7B:         return "A1.7B";
        case MODEL_A2_7B:         return "A2.7B";
        case MODEL_8x7B:          return "8x7B";
        case MODEL_8x22B:         return "8x22B";
        case MODEL_16x12B:        return "16x12B";
        case MODEL_10B_128x3_66B: return "10B+128x3.66B";
        case MODEL_57B_A14B:      return "57B.A14B";
        case MODEL_27B:           return "27B";
        default:                  return "?B";
    }
}

static std::string llama_model_ftype_name(llama_ftype ftype) {
    if (ftype & LLAMA_FTYPE_GUESSED) {
        return llama_model_ftype_name((enum llama_ftype) (ftype & ~LLAMA_FTYPE_GUESSED)) + " (guessed)";
    }

    switch (ftype) {
        case LLAMA_FTYPE_ALL_F32:         return "all F32";
        case LLAMA_FTYPE_MOSTLY_F16:      return "F16";
        case LLAMA_FTYPE_MOSTLY_BF16:     return "BF16";
        case LLAMA_FTYPE_MOSTLY_Q4_0:     return "Q4_0";
        case LLAMA_FTYPE_MOSTLY_Q4_1:     return "Q4_1";
        case LLAMA_FTYPE_MOSTLY_Q5_0:     return "Q5_0";
        case LLAMA_FTYPE_MOSTLY_Q5_1:     return "Q5_1";
        case LLAMA_FTYPE_MOSTLY_Q8_0:     return "Q8_0";
        case LLAMA_FTYPE_MOSTLY_Q2_K:     return "Q2_K - Medium";
        case LLAMA_FTYPE_MOSTLY_Q2_K_S:   return "Q2_K - Small";
        case LLAMA_FTYPE_MOSTLY_Q3_K_S:   return "Q3_K - Small";
        case LLAMA_FTYPE_MOSTLY_Q3_K_M:   return "Q3_K - Medium";
        case LLAMA_FTYPE_MOSTLY_Q3_K_L:   return "Q3_K - Large";
        case LLAMA_FTYPE_MOSTLY_Q4_K_S:   return "Q4_K - Small";
        case LLAMA_FTYPE_MOSTLY_Q4_K_M:   return "Q4_K - Medium";
        case LLAMA_FTYPE_MOSTLY_Q5_K_S:   return "Q5_K - Small";
        case LLAMA_FTYPE_MOSTLY_Q5_K_M:   return "Q5_K - Medium";
        case LLAMA_FTYPE_MOSTLY_Q6_K:     return "Q6_K";
        case LLAMA_FTYPE_MOSTLY_TQ1_0:    return "TQ1_0 - 1.69 bpw ternary";
        case LLAMA_FTYPE_MOSTLY_TQ2_0:    return "TQ2_0 - 2.06 bpw ternary";
        case LLAMA_FTYPE_MOSTLY_IQ2_XXS:  return "IQ2_XXS - 2.0625 bpw";
        case LLAMA_FTYPE_MOSTLY_IQ2_XS:   return "IQ2_XS - 2.3125 bpw";
        case LLAMA_FTYPE_MOSTLY_IQ2_S:    return "IQ2_S - 2.5 bpw";
        case LLAMA_FTYPE_MOSTLY_IQ2_M:    return "IQ2_M - 2.7 bpw";
        case LLAMA_FTYPE_MOSTLY_IQ3_XS:   return "IQ3_XS - 3.3 bpw";
        case LLAMA_FTYPE_MOSTLY_IQ3_XXS:  return "IQ3_XXS - 3.0625 bpw";
        case LLAMA_FTYPE_MOSTLY_IQ1_S:    return "IQ1_S - 1.5625 bpw";
        case LLAMA_FTYPE_MOSTLY_IQ1_M:    return "IQ1_M - 1.75 bpw";
        case LLAMA_FTYPE_MOSTLY_IQ4_NL:   return "IQ4_NL - 4.5 bpw";
        case LLAMA_FTYPE_MOSTLY_IQ4_XS:   return "IQ4_XS - 4.25 bpw";
        case LLAMA_FTYPE_MOSTLY_IQ3_S:    return "IQ3_S - 3.4375 bpw";
        case LLAMA_FTYPE_MOSTLY_IQ3_M:    return "IQ3_S mix - 3.66 bpw";

        default: return "unknown, may not work";
    }
}

static const char * llama_expert_gating_func_name(llama_expert_gating_func_type type) {
    switch (type) {
        case LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX: return "softmax";
        case LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID: return "sigmoid";
        default:                                    return "unknown";
    }
}

std::string llama_model_arch_name (const llama_model & model) {
    return llm_arch_name(model.arch);
}

std::string llama_model_type_name (const llama_model & model) {
    return llm_type_name(model.type);
}

std::string llama_model_ftype_name(const llama_model & model) {
    return llama_model_ftype_name(model.ftype);
}

ggml_backend_buffer_type_t llama_model_select_buft(const llama_model & model, int il) {
    return select_buft(
            *model.dev_layer.at(il).buft_list,
            [&](ggml_context * ctx) {
                ggml_tensor * cur = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, model.hparams.n_embd);
                ggml_tensor * layer_dir = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, model.hparams.n_embd);
                return ggml_add(ctx, cur, layer_dir);
            });
}

struct ggml_tensor * llama_model_get_tensor(const struct llama_model & model, const char * name) {
    auto it = std::find_if(model.tensors_by_name.begin(), model.tensors_by_name.end(),
            [name](const std::pair<std::string, struct ggml_tensor *> & it) {
                return it.first == name;
            });
    if (it == model.tensors_by_name.end()) {
        return nullptr;
    }

    return it->second;
}

size_t llama_model_max_nodes(const llama_model & model) {
    return std::max<size_t>(8192, model.tensors_by_name.size()*5);
}

static const std::map<llama_rope_scaling_type, const char *> LLAMA_ROPE_SCALING_TYPES = {
    { LLAMA_ROPE_SCALING_TYPE_NONE,       "none"       },
    { LLAMA_ROPE_SCALING_TYPE_LINEAR,     "linear"     },
    { LLAMA_ROPE_SCALING_TYPE_YARN,       "yarn"       },
    { LLAMA_ROPE_SCALING_TYPE_LONGROPE,   "longrope"   },
};

static llama_rope_scaling_type llama_rope_scaling_type_from_string(const std::string & name) {
    for (const auto & kv : LLAMA_ROPE_SCALING_TYPES) {
        if (kv.second == name) {
            return (llama_rope_scaling_type) kv.first;
        }
    }

    return LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED;
}

// NOTE: avoid ever using this except for building the token_to_piece caches
static std::string llama_token_to_piece(const struct llama_model * model, llama_token token, bool special) {
    std::string piece;
    piece.resize(piece.capacity());  // using string internal cache
    const int n_chars = llama_token_to_piece(model, token, &piece[0], piece.size(), 0, special);
    if (n_chars < 0) {
        piece.resize(-n_chars);
        int check = llama_token_to_piece(model, token, &piece[0], piece.size(), 0, special);
        GGML_ASSERT(check == -n_chars);
    }
    else {
        piece.resize(n_chars);
    }

    return piece;
}

void llm_load_stats(llama_model_loader & ml, llama_model & model) {
    model.n_elements = ml.n_elements;
    model.n_bytes = ml.n_bytes;
}

void llm_load_arch(llama_model_loader & ml, llama_model & model) {
    model.arch = ml.get_arch();
    if (model.arch == LLM_ARCH_UNKNOWN) {
        throw std::runtime_error("unknown model architecture: '" + ml.get_arch_name() + "'");
    }
}

void llm_load_hparams(llama_model_loader & ml, llama_model & model) {
    auto & hparams = model.hparams;
    const gguf_context * ctx = ml.meta.get();

    // get metadata as string
    for (int i = 0; i < gguf_get_n_kv(ctx); i++) {
        enum gguf_type type = gguf_get_kv_type(ctx, i);
        if (type == GGUF_TYPE_ARRAY) {
            continue;
        }
        const char * name = gguf_get_key(ctx, i);
        const std::string value = gguf_kv_to_str(ctx, i);
        model.gguf_kv.emplace(name, value);
    }

    // get general kv
    ml.get_key(LLM_KV_GENERAL_NAME, model.name, false);

    // get hparams kv
    ml.get_key(LLM_KV_VOCAB_SIZE, hparams.n_vocab, false) || ml.get_arr_n(LLM_KV_TOKENIZER_LIST, hparams.n_vocab, false);

    // everything past this point is not vocab-related
    if (hparams.vocab_only) {
        return;
    }

    ml.get_key(LLM_KV_CONTEXT_LENGTH,    hparams.n_ctx_train);
    ml.get_key(LLM_KV_EMBEDDING_LENGTH,  hparams.n_embd);
    ml.get_key(LLM_KV_BLOCK_COUNT,       hparams.n_layer);
    ml.get_key(LLM_KV_EXPERT_COUNT,      hparams.n_expert,      false);
    ml.get_key(LLM_KV_EXPERT_USED_COUNT, hparams.n_expert_used, false);

    if (model.arch == LLM_ARCH_WAVTOKENIZER_DEC) {
        ml.get_key(LLM_KV_FEATURES_LENGTH, hparams.n_embd_features);

        ml.get_key(LLM_KV_POSNET_EMBEDDING_LENGTH, hparams.posnet.n_embd);
        ml.get_key(LLM_KV_POSNET_BLOCK_COUNT,      hparams.posnet.n_layer);

        ml.get_key(LLM_KV_CONVNEXT_EMBEDDING_LENGTH, hparams.convnext.n_embd);
        ml.get_key(LLM_KV_CONVNEXT_BLOCK_COUNT,      hparams.convnext.n_layer);
    }

    GGML_ASSERT(hparams.n_expert <= LLAMA_MAX_EXPERTS);
    GGML_ASSERT(hparams.n_expert_used <= hparams.n_expert);
    if (hparams.n_expert > 0) {
        GGML_ASSERT(hparams.n_expert_used > 0);
    } else {
        GGML_ASSERT(hparams.n_expert_used == 0);
    }

    // zero-out the array hparams
    std::fill(hparams.n_head_arr.begin(),    hparams.n_head_arr.end(),    0);
    std::fill(hparams.n_head_kv_arr.begin(), hparams.n_head_kv_arr.end(), 0);
    std::fill(hparams.n_ff_arr.begin(),      hparams.n_ff_arr.end(),      0);
    std::fill(hparams.cross_attn_layers.begin(), hparams.cross_attn_layers.end(), -1);

    ml.get_key_or_arr(LLM_KV_FEED_FORWARD_LENGTH,       hparams.n_ff_arr,   hparams.n_layer, false);
    ml.get_key_or_arr(LLM_KV_ATTENTION_HEAD_COUNT,      hparams.n_head_arr, hparams.n_layer, false);
    ml.get_arr(LLM_KV_ATTENTION_CROSS_ATTENTION_LAYERS, hparams.cross_attn_layers, false);

    // n_head_kv is optional, default to n_head
    hparams.n_head_kv_arr = hparams.n_head_arr;

    ml.get_key_or_arr(LLM_KV_ATTENTION_HEAD_COUNT_KV, hparams.n_head_kv_arr, hparams.n_layer, false);

    bool rope_finetuned = false;
    ml.get_key(LLM_KV_ROPE_SCALING_FINETUNED, rope_finetuned, false);
    hparams.rope_finetuned = rope_finetuned;

    hparams.n_ctx_orig_yarn = hparams.n_ctx_train;
    ml.get_key(LLM_KV_ROPE_SCALING_ORIG_CTX_LEN, hparams.n_ctx_orig_yarn, false);

    // rope_freq_base (optional)
    hparams.rope_freq_base_train = 10000.0f;
    ml.get_key(LLM_KV_ROPE_FREQ_BASE, hparams.rope_freq_base_train, false);

    std::string rope_scaling("linear");
    ml.get_key(LLM_KV_ROPE_SCALING_TYPE, rope_scaling, false);
    hparams.rope_scaling_type_train = llama_rope_scaling_type_from_string(rope_scaling);
    GGML_ASSERT(hparams.rope_scaling_type_train != LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED);

    // rope_freq_scale (inverse of the kv) is optional
    float ropescale = 0.0f;
    if (!ml.get_key(LLM_KV_ROPE_SCALING_FACTOR, ropescale, false)) {
        // try the old key name
        ml.get_key(LLM_KV_ROPE_SCALE_LINEAR, ropescale, false);
    }
    hparams.rope_freq_scale_train = ropescale == 0.0f ? 1.0f : 1.0f/ropescale;

    ml.get_key(LLM_KV_ROPE_SCALING_ATTN_FACTOR, hparams.rope_attn_factor, false);

    // non-transformer models do not have attention heads
    if (hparams.n_head() > 0) {
        // gpt-neox n_rot = rotary_pct * (n_embd / n_head)
        // gpt-j n_rot = rotary_dim

        hparams.n_embd_head_k = hparams.n_embd / hparams.n_head();
        ml.get_key(LLM_KV_ATTENTION_KEY_LENGTH, hparams.n_embd_head_k, false);

        hparams.n_embd_head_v = hparams.n_embd / hparams.n_head();
        ml.get_key(LLM_KV_ATTENTION_VALUE_LENGTH, hparams.n_embd_head_v, false);

        // sanity check for n_rot (optional)
        hparams.n_rot = hparams.n_embd_head_k;

        ml.get_key(LLM_KV_ROPE_DIMENSION_COUNT, hparams.n_rot, false);

        if (model.arch == LLM_ARCH_LLAMA || model.arch == LLM_ARCH_MLLAMA || model.arch == LLM_ARCH_DECI || model.arch == LLM_ARCH_FALCON) {
            if (hparams.n_rot != hparams.n_embd_head_k) {
                throw std::runtime_error(format("invalid n_rot: %u, expected %u", hparams.n_rot, hparams.n_embd_head_k));
            }
        }
    } else {
        hparams.n_rot = 0;
        hparams.n_embd_head_k = 0;
        hparams.n_embd_head_v = 0;
    }

    using e_model = llm_type; // TMP

    // arch-specific KVs
    switch (model.arch) {
        case LLM_ARCH_LLAMA:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

                if (hparams.n_expert == 8) {
                    switch (hparams.n_layer) {
                        case 32: model.type = e_model::MODEL_8x7B; break;
                        case 56: model.type = e_model::MODEL_8x22B; break;
                        default: model.type = e_model::MODEL_UNKNOWN;
                    }
                } else {
                    switch (hparams.n_layer) {
                        case 16: model.type = e_model::MODEL_1B; break; // Llama 3.2 1B
                        case 22: model.type = e_model::MODEL_1B; break;
                        case 26: model.type = e_model::MODEL_3B; break;
                        case 28: model.type = e_model::MODEL_3B; break; // Llama 3.2 3B
                        // granite uses a vocab with len 49152
                        case 32: model.type = hparams.n_vocab == 49152 ? e_model::MODEL_3B : (hparams.n_vocab < 40000 ? e_model::MODEL_7B : e_model::MODEL_8B); break;
                        case 36: model.type = e_model::MODEL_8B; break; // granite
                        case 40: model.type = e_model::MODEL_13B; break;
                        case 48: model.type = e_model::MODEL_34B; break;
                        case 60: model.type = e_model::MODEL_30B; break;
                        case 80: model.type = hparams.n_head() == hparams.n_head_kv() ? e_model::MODEL_65B : e_model::MODEL_70B; break;
                        default: model.type = e_model::MODEL_UNKNOWN;
                    }
                }
            } break;
        case LLM_ARCH_MLLAMA:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

                switch (hparams.n_layer) {
                    case 40: model.type = e_model::MODEL_11B; break;
                    case 100: model.type = e_model::MODEL_90B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_DECI:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                switch (hparams.n_layer) {
                    case 32: model.type = e_model::MODEL_7B; break;
                    case 80: model.type = e_model::MODEL_70B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_MINICPM:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                ml.get_key(LLM_KV_EMBEDDING_SCALE, hparams.f_embedding_scale);
                ml.get_key(LLM_KV_RESIDUAL_SCALE, hparams.f_residual_scale);
                ml.get_key(LLM_KV_LOGIT_SCALE, hparams.f_logit_scale);

                switch (hparams.n_layer) {
                    case 52: model.type = e_model::MODEL_1B; break;
                    case 40: model.type = e_model::MODEL_2B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_MINICPM3:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                ml.get_key(LLM_KV_ATTENTION_Q_LORA_RANK, hparams.n_lora_q);
                ml.get_key(LLM_KV_ATTENTION_KV_LORA_RANK, hparams.n_lora_kv);

                switch (hparams.n_layer) {
                    case 62: model.type = e_model::MODEL_4B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_GROK:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

                switch (hparams.n_layer) {
                    case 64: model.type = e_model::MODEL_314B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_FALCON:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps);

                switch (hparams.n_layer) {
                    case 32: model.type = e_model::MODEL_7B; break;
                    case 60: model.type = e_model::MODEL_40B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_BAICHUAN:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                switch (hparams.n_layer) {
                    case 32: model.type = e_model::MODEL_7B; break;
                    case 40: model.type = e_model::MODEL_13B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }

                if (model.type == e_model::MODEL_13B) {
                    // TODO: become GGUF KV parameter
                    hparams.f_max_alibi_bias = 8.0f;
                }
            } break;
        case LLM_ARCH_STARCODER:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps);
                switch (hparams.n_layer) {
                    case 24: model.type = e_model::MODEL_1B; break;
                    case 36: model.type = e_model::MODEL_3B; break;
                    case 42: model.type = e_model::MODEL_7B; break;
                    case 40: model.type = e_model::MODEL_15B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_REFACT:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                switch (hparams.n_layer) {
                    case 32: model.type = e_model::MODEL_1B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }

                // TODO: become GGUF KV parameter
                hparams.f_max_alibi_bias = 8.0f;
            } break;
        case LLM_ARCH_BERT:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS,    hparams.f_norm_eps);
                ml.get_key(LLM_KV_ATTENTION_CAUSAL,           hparams.causal_attn);
                ml.get_key(LLM_KV_TOKENIZER_TOKEN_TYPE_COUNT, hparams.n_vocab_type);
                ml.get_key(LLM_KV_POOLING_TYPE,               hparams.pooling_type, false);

                switch (hparams.n_layer) {
                    case 3:
                        model.type = e_model::MODEL_17M; break; // bge-micro
                    case 6:
                        model.type = e_model::MODEL_22M; break; // MiniLM-L6
                    case 12:
                        switch (hparams.n_embd) {
                            case 384: model.type = e_model::MODEL_33M; break; // MiniLM-L12, bge-small
                            case 768: model.type = e_model::MODEL_109M; break; // bge-base
                            default: model.type = e_model::MODEL_UNKNOWN;
                        } break;
                    case 24:
                        model.type = e_model::MODEL_335M; break; // bge-large
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_JINA_BERT_V2:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS,    hparams.f_norm_eps);
                ml.get_key(LLM_KV_ATTENTION_CAUSAL,           hparams.causal_attn);
                ml.get_key(LLM_KV_TOKENIZER_TOKEN_TYPE_COUNT, hparams.n_vocab_type);
                ml.get_key(LLM_KV_POOLING_TYPE,               hparams.pooling_type, false);
                hparams.f_max_alibi_bias = 8.0f;

                switch (hparams.n_layer) {
                    case 4:  model.type = e_model::MODEL_33M;  break; // jina-embeddings-small
                    case 12: model.type = e_model::MODEL_137M; break; // jina-embeddings-base
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_NOMIC_BERT:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS,    hparams.f_norm_eps);
                ml.get_key(LLM_KV_ATTENTION_CAUSAL,           hparams.causal_attn);
                ml.get_key(LLM_KV_TOKENIZER_TOKEN_TYPE_COUNT, hparams.n_vocab_type);
                ml.get_key(LLM_KV_POOLING_TYPE,               hparams.pooling_type);

                if (hparams.n_layer == 12 && hparams.n_embd == 768) {
                    model.type = e_model::MODEL_137M;
                }
            } break;
        case LLM_ARCH_BLOOM:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps);

                switch (hparams.n_layer) {
                    case 24: model.type = e_model::MODEL_1B; break;
                    case 30:
                        switch (hparams.n_embd) {
                            case 2560: model.type = e_model::MODEL_3B; break;
                            case 4096: model.type = e_model::MODEL_7B; break;
                            default: model.type = e_model::MODEL_UNKNOWN;
                        } break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }

                // TODO: become GGUF KV parameter
                hparams.f_max_alibi_bias = 8.0f;
            } break;
        case LLM_ARCH_MPT:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS,  hparams.f_norm_eps);
                ml.get_key(LLM_KV_ATTENTION_CLAMP_KQV,      hparams.f_clamp_kqv, false);
                ml.get_key(LLM_KV_ATTENTION_MAX_ALIBI_BIAS, hparams.f_max_alibi_bias);

                switch (hparams.n_layer) {
                    case 32: model.type = e_model::MODEL_7B; break;
                    case 48: model.type = e_model::MODEL_30B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_STABLELM:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps);

                switch (hparams.n_layer) {
                    case 24: model.type = e_model::MODEL_1B; break;
                    case 32: model.type = e_model::MODEL_3B; break;
                    case 40: model.type = e_model::MODEL_12B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
               }
            } break;
        case LLM_ARCH_QWEN:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

                switch (hparams.n_layer) {
                    case 32: model.type = e_model::MODEL_7B; break;
                    case 40: model.type = e_model::MODEL_13B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_QWEN2VL:
            {
                ml.get_key_or_arr(LLM_KV_ROPE_DIMENSION_SECTIONS, hparams.rope_sections, 4, true);
            }
            // fall through
        case LLM_ARCH_QWEN2:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                switch (hparams.n_layer) {
                    case 24: model.type = hparams.n_embd == 1024 ? e_model::MODEL_0_5B : e_model::MODEL_1B; break;
                    case 28: model.type = hparams.n_embd == 1536 ? e_model::MODEL_1_5B : e_model::MODEL_7B; break;
                    case 32: model.type = e_model::MODEL_7B; break;
                    case 36: model.type = e_model::MODEL_3B; break;
                    case 40: model.type = hparams.n_head() == 20 ? e_model::MODEL_4B : e_model::MODEL_13B; break;
                    case 48: model.type = e_model::MODEL_14B; break;
                    case 64: model.type = e_model::MODEL_32B; break;
                    case 80: model.type = e_model::MODEL_70B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_QWEN2MOE:
            {
                ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp, false);
                ml.get_key(LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH, hparams.n_ff_shexp, false);

                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                switch (hparams.n_layer) {
                    case 24: model.type = e_model::MODEL_A2_7B; break;
                    case 28: model.type = e_model::MODEL_57B_A14B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_PHI2:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps);

                switch (hparams.n_layer) {
                    case 24: model.type = e_model::MODEL_1B; break;
                    case 32: model.type = e_model::MODEL_3B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_PHI3:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

                switch (hparams.n_layer) {
                    case 24: model.type = e_model::MODEL_1B; break;
                    case 32: model.type = e_model::MODEL_3B; break;
                    case 40: model.type = e_model::MODEL_14B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }

                // for backward compatibility ; see: https://github.com/ggerganov/llama.cpp/pull/8931
                if ((hparams.n_layer == 32 || hparams.n_layer == 40) && hparams.n_ctx_train == 4096) {
                    // default value for Phi-3-mini-4k-instruct and Phi-3-medium-4k-instruct
                    hparams.n_swa = 2047;
                } else if (hparams.n_layer == 32 && hparams.n_head_kv(0) == 32 && hparams.n_ctx_train == 131072) {
                    // default value for Phi-3-mini-128k-instruct
                    hparams.n_swa = 262144;
                } else if (hparams.n_layer == 40 && hparams.n_ctx_train == 131072) {
                    // default value for Phi-3-medium-128k-instruct
                    hparams.n_swa = 131072;
                }
                bool found_swa = ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW, hparams.n_swa, false);
                if (!found_swa && hparams.n_swa == 0) {
                    throw std::runtime_error("invalid value for sliding_window");
                }
            } break;
        case LLM_ARCH_PLAMO:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

                switch (hparams.n_layer) {
                    case 40: model.type = e_model::MODEL_13B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
               }
            } break;
        case LLM_ARCH_GPT2:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps);
                switch (hparams.n_layer) {
                    case 12: model.type = e_model::MODEL_SMALL; break;
                    case 24: model.type = e_model::MODEL_MEDIUM; break;
                    case 36: model.type = e_model::MODEL_LARGE; break;
                    case 48: model.type = e_model::MODEL_XL; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_CODESHELL:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps);
                switch (hparams.n_layer) {
                    case 42: model.type = e_model::MODEL_7B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_ORION:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps);

                switch (hparams.n_layer) {
                    case 40: model.type = e_model::MODEL_14B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_INTERNLM2:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                switch (hparams.n_layer) {
                    case 32: model.type = e_model::MODEL_7B; break;
                    case 48: model.type = e_model::MODEL_20B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_GEMMA:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

                switch (hparams.n_layer) {
                    case 18: model.type = e_model::MODEL_2B; break;
                    case 28: model.type = e_model::MODEL_7B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
               }
            } break;
        case LLM_ARCH_GEMMA2:
            {
                hparams.n_swa = 4096; // default value of gemma 2
                ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW, hparams.n_swa, false);
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                ml.get_key(LLM_KV_ATTN_LOGIT_SOFTCAPPING, hparams.f_attn_logit_softcapping, false);
                ml.get_key(LLM_KV_FINAL_LOGIT_SOFTCAPPING, hparams.f_final_logit_softcapping, false);
                hparams.attn_soft_cap = true;

                switch (hparams.n_layer) {
                    case 26: model.type = e_model::MODEL_2B; break;
                    case 42: model.type = e_model::MODEL_9B; break;
                    case 46: model.type = e_model::MODEL_27B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
               }
            } break;
        case LLM_ARCH_STARCODER2:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps);
                switch (hparams.n_layer) {
                    case 30: model.type = e_model::MODEL_3B; break;
                    case 32: model.type = e_model::MODEL_7B; break;
                    case 40: model.type = e_model::MODEL_15B; break;
                    case 52: model.type = e_model::MODEL_20B; break; // granite
                    case 88: model.type = e_model::MODEL_34B; break; // granite
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_MAMBA:
            {
                ml.get_key(LLM_KV_SSM_CONV_KERNEL,    hparams.ssm_d_conv);
                ml.get_key(LLM_KV_SSM_INNER_SIZE,     hparams.ssm_d_inner);
                ml.get_key(LLM_KV_SSM_STATE_SIZE,     hparams.ssm_d_state);
                ml.get_key(LLM_KV_SSM_TIME_STEP_RANK, hparams.ssm_dt_rank);
                ml.get_key(LLM_KV_SSM_DT_B_C_RMS,     hparams.ssm_dt_b_c_rms, false);

                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

                switch (hparams.n_layer) {
                    case 24:
                        switch (hparams.n_embd) {
                            case 768: model.type = e_model::MODEL_SMALL; break;
                            default: model.type = e_model::MODEL_UNKNOWN;
                        } break;
                    case 48:
                        switch (hparams.n_embd) {
                            case 1024: model.type = e_model::MODEL_MEDIUM; break;
                            case 1536: model.type = e_model::MODEL_LARGE; break;
                            case 2048: model.type = e_model::MODEL_XL; break;
                            default:   model.type = e_model::MODEL_UNKNOWN;
                        } break;
                    case 64:
                        switch (hparams.n_embd) {
                            case 2560: model.type = e_model::MODEL_3B; break;
                            default: model.type = e_model::MODEL_UNKNOWN;
                        } break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_XVERSE:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                switch (hparams.n_layer) {
                    case 32: model.type = e_model::MODEL_7B; break;
                    case 40: model.type = e_model::MODEL_13B; break;
                    case 80: model.type = e_model::MODEL_65B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_COMMAND_R:
            {
                ml.get_key(LLM_KV_LOGIT_SCALE, hparams.f_logit_scale);
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps);
                switch (hparams.n_layer) {
                    case 40: model.type = e_model::MODEL_35B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_COHERE2:
            {
                ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW, hparams.n_swa);
                ml.get_key(LLM_KV_LOGIT_SCALE, hparams.f_logit_scale);
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps);
                switch (hparams.n_layer) {
                    case 32: model.type = e_model::MODEL_8B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_DBRX:
        {
            ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps);
            ml.get_key(LLM_KV_ATTENTION_CLAMP_KQV,     hparams.f_clamp_kqv);

            switch (hparams.n_layer) {
                case 40: model.type = e_model::MODEL_16x12B; break;
                default: model.type = e_model::MODEL_UNKNOWN;
            }
        } break;
        case LLM_ARCH_OLMO:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps);
                ml.get_key(LLM_KV_ATTENTION_CLAMP_KQV,     hparams.f_clamp_kqv, false);

                switch (hparams.n_layer) {
                    case 22: model.type = e_model::MODEL_1B; break;
                    case 32: model.type = e_model::MODEL_7B; break;
                    case 80: model.type = e_model::MODEL_70B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_OLMO2:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

                switch (hparams.n_layer) {
                    case 16: model.type = e_model::MODEL_1B; break;
                    case 32: model.type = e_model::MODEL_7B; break;
                    case 40: model.type = e_model::MODEL_13B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_OLMOE:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                switch (hparams.n_layer) {
                    case 16: model.type = e_model::MODEL_A1_7B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_OPENELM:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

                switch (hparams.n_layer) {
                case 16: model.type = e_model::MODEL_270M; break;
                case 20: model.type = e_model::MODEL_450M; break;
                case 28: model.type = e_model::MODEL_1B; break;
                case 36: model.type = e_model::MODEL_3B; break;
                default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_GPTNEOX:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps);
                ml.get_key(LLM_KV_USE_PARALLEL_RESIDUAL, hparams.use_par_res);
                switch (hparams.n_layer) {
                    case 6:
                        switch (hparams.n_ff()) {
                            case 512: model.type = e_model::MODEL_14M; break;
                            case 2048: model.type = e_model::MODEL_70M; break;
                            default: model.type = e_model::MODEL_UNKNOWN;
                        } break;
                    case 12:
                        switch (hparams.n_ff()) {
                            case 3072: model.type = e_model::MODEL_160M; break;
                            default: model.type = e_model::MODEL_UNKNOWN;
                        } break;
                    case 16:
                        switch (hparams.n_ff()) {
                            case 8192: model.type = e_model::MODEL_1B; break;
                            default: model.type = e_model::MODEL_UNKNOWN;
                        } break;
                    case 24:
                        switch (hparams.n_ff()) {
                            case 4096: model.type = e_model::MODEL_410M; break;
                            case 8192: model.type = e_model::MODEL_1_4B; break;
                            default: model.type = e_model::MODEL_UNKNOWN;
                        } break;
                    case 32:
                        switch (hparams.n_ff()) {
                            case 10240: model.type = e_model::MODEL_2_8B; break;
                            case 16384: model.type = e_model::MODEL_6_9B; break;
                            default: model.type = e_model::MODEL_UNKNOWN;
                        } break;
                    case 36:
                        switch (hparams.n_ff()) {
                            case 20480: model.type = e_model::MODEL_12B; break;
                            default: model.type = e_model::MODEL_UNKNOWN;
                        } break;
                    case 44:
                        switch (hparams.n_ff()) {
                            case 24576: model.type = e_model::MODEL_20B; break;
                            default: model.type = e_model::MODEL_UNKNOWN;
                        } break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_ARCTIC:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

                if (hparams.n_expert == 128) {
                    switch (hparams.n_layer) {
                        case 35: model.type = e_model::MODEL_10B_128x3_66B; break;
                        default: model.type = e_model::MODEL_UNKNOWN;
                    }
                } else {
                    model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_DEEPSEEK:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                ml.get_key(LLM_KV_LEADING_DENSE_BLOCK_COUNT, hparams.n_layer_dense_lead);
                ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp);
                ml.get_key(LLM_KV_EXPERT_SHARED_COUNT, hparams.n_expert_shared);
                ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE, hparams.expert_weights_scale);

                switch (hparams.n_layer) {
                    case 28: model.type = e_model::MODEL_20B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_DEEPSEEK2:
            {
                bool is_lite = (hparams.n_layer == 27);
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                ml.get_key(LLM_KV_LEADING_DENSE_BLOCK_COUNT, hparams.n_layer_dense_lead);
                if (!is_lite) {
                    ml.get_key(LLM_KV_ATTENTION_Q_LORA_RANK, hparams.n_lora_q);
                }
                ml.get_key(LLM_KV_ATTENTION_KV_LORA_RANK, hparams.n_lora_kv);
                ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp);
                ml.get_key(LLM_KV_EXPERT_SHARED_COUNT, hparams.n_expert_shared);
                ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE, hparams.expert_weights_scale);
                ml.get_key(LLM_KV_EXPERT_WEIGHTS_NORM, hparams.expert_weights_norm, false);
                ml.get_key(LLM_KV_EXPERT_GATING_FUNC, hparams.expert_gating_func, false);
                if (hparams.expert_gating_func == LLAMA_EXPERT_GATING_FUNC_TYPE_NONE) {
                    // for compatibility with existing DeepSeek V2 and V2.5 GGUFs
                    // that have no expert_gating_func model parameter set
                    hparams.expert_gating_func = LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX;
                }
                ml.get_key(LLM_KV_ROPE_SCALING_YARN_LOG_MUL, hparams.rope_yarn_log_mul);

                switch (hparams.n_layer) {
                    case 27: model.type = e_model::MODEL_16B; break;
                    case 60: model.type = e_model::MODEL_236B; break;
                    case 61: model.type = e_model::MODEL_671B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_CHATGLM:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                switch (hparams.n_layer) {
                    case 28: model.type = e_model::MODEL_6B; break;
                    case 40: model.type = e_model::MODEL_9B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_BITNET:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

                switch (hparams.n_layer) {
                    case 26: model.type = e_model::MODEL_3B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_T5:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                ml.get_key(LLM_KV_ATTENTION_RELATIVE_BUCKETS_COUNT, hparams.n_rel_attn_bkts);

                uint32_t dec_start_token_id;
                if (ml.get_key(LLM_KV_DECODER_START_TOKEN_ID, dec_start_token_id, false)) {
                    hparams.dec_start_token_id = dec_start_token_id;
                }

                switch (hparams.n_layer) {
                    case 6:  model.type = e_model::MODEL_60M;  break; // t5-small
                    case 8:  model.type = e_model::MODEL_80M;  break; // flan-t5-small
                    case 12:
                        switch (hparams.n_ff()) {
                            case 3072: model.type = e_model::MODEL_220M; break; // t5-base
                            case 2048: model.type = e_model::MODEL_250M; break; // flan-t5-base
                            default: model.type = e_model::MODEL_UNKNOWN;
                        } break;
                    case 24:
                        switch (hparams.n_ff()) {
                            case 4096:  model.type = e_model::MODEL_770M; break; // t5-large
                            case 2816:  model.type = e_model::MODEL_780M; break; // flan-t5-large
                            case 16384: model.type = e_model::MODEL_3B;   break; // t5-3b
                            case 5120:  model.type = e_model::MODEL_3B;   break; // flan-t5-xl
                            case 65536: model.type = e_model::MODEL_11B;  break; // t5-11b
                            case 10240: model.type = e_model::MODEL_11B;  break; // flan-t5-xxl
                            default: model.type = e_model::MODEL_UNKNOWN;
                        } break;
                    default: model.type = e_model::MODEL_UNKNOWN;
               }
            } break;
        case LLM_ARCH_T5ENCODER:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                ml.get_key(LLM_KV_ATTENTION_RELATIVE_BUCKETS_COUNT, hparams.n_rel_attn_bkts);
                model.type = e_model::MODEL_UNKNOWN;
            } break;
        case LLM_ARCH_JAIS:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps);
                ml.get_key(LLM_KV_ATTENTION_MAX_ALIBI_BIAS, hparams.f_max_alibi_bias);

                switch (hparams.n_layer) {
                    case 24: model.type = e_model::MODEL_1_3B; break;
                    case 40: model.type = e_model::MODEL_13B; break;
                    /* TODO: add variants */
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_NEMOTRON:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps);
                switch (hparams.n_layer) {
                    case 32: model.type = e_model::MODEL_4B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_EXAONE:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

                switch (hparams.n_layer) {
                    case 32: model.type = e_model::MODEL_8B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_RWKV6:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps);
                ml.get_key(LLM_KV_WKV_HEAD_SIZE, hparams.wkv_head_size);
                ml.get_key(LLM_KV_TIME_MIX_EXTRA_DIM, hparams.time_mix_extra_dim);
                ml.get_key(LLM_KV_TIME_DECAY_EXTRA_DIM, hparams.time_decay_extra_dim);
                ml.get_key(LLM_KV_RESCALE_EVERY_N_LAYERS, hparams.rescale_every_n_layers, false);

                switch (hparams.n_layer) {
                    case 24: model.type = e_model::MODEL_1_6B; break;
                    case 32:
                        switch (hparams.n_embd) {
                            case 2560: model.type = e_model::MODEL_3B; break;
                            case 4096: model.type = e_model::MODEL_7B; break;
                            default: model.type = e_model::MODEL_UNKNOWN;
                        } break;
                    case 61: model.type = e_model::MODEL_14B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_GRANITE:
        case LLM_ARCH_GRANITE_MOE:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                ml.get_key(LLM_KV_LOGIT_SCALE, hparams.f_logit_scale);
                ml.get_key(LLM_KV_RESIDUAL_SCALE, hparams.f_residual_scale);
                ml.get_key(LLM_KV_EMBEDDING_SCALE, hparams.f_embedding_scale);
                ml.get_key(LLM_KV_ATTENTION_SCALE, hparams.f_attention_scale);

                switch (hparams.n_layer) {
                    case 32: model.type = e_model::MODEL_3B; break;
                    case 40: model.type = e_model::MODEL_3B; break;
                    // Add additional layer/vocab/etc checks here for other model sizes
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_CHAMELEON:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                hparams.f_norm_eps = 1e-5;  // eps for qk-norm, torch default
                ml.get_key(LLM_KV_SWIN_NORM, hparams.swin_norm);

                switch (hparams.n_layer) {
                    case 32: model.type = e_model::MODEL_7B; break;
                    case 48: model.type = e_model::MODEL_34B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
               }
            } break;
        case LLM_ARCH_SOLAR:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                for (size_t i = 0; i < hparams.n_bskcn_arr.max_size(); ++i) {
                    auto & bskcn = hparams.n_bskcn_arr[i];
                    bskcn.fill(0);
                    auto kv = LLM_KV(model.arch);
                    ml.get_key_or_arr(format((kv(LLM_KV_ATTENTION_BLOCK_SKIP_CONNECTION) + ".%d").c_str(), i), bskcn, hparams.n_layer, false);
                }

                switch (hparams.n_layer) {
                    case 64: model.type = e_model::MODEL_22B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_WAVTOKENIZER_DEC:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS,    hparams.f_norm_eps);
                ml.get_key(LLM_KV_ATTENTION_GROUPNORM_EPS,    hparams.f_norm_group_eps);
                ml.get_key(LLM_KV_ATTENTION_GROUPNORM_GROUPS, hparams.n_norm_groups);
                ml.get_key(LLM_KV_ATTENTION_CAUSAL,           hparams.causal_attn);
            } break;
        default: throw std::runtime_error("unsupported model architecture");
    }

    model.ftype = ml.ftype;

    if (hparams.f_max_alibi_bias > 0.0f) {
        hparams.use_alibi = true;
    }

    hparams.rope_type = llama_rope_type(&model);
}

void llm_load_vocab(llama_model_loader & ml, llama_model & model) {
    auto & vocab = model.vocab;

    struct gguf_context * ctx = ml.meta.get();

    const auto kv = LLM_KV(model.arch);

    // determine vocab type
    {
        std::string tokenizer_model;
        std::string tokenizer_pre;

        ml.get_key(LLM_KV_TOKENIZER_MODEL, tokenizer_model);
        ml.get_key(LLM_KV_TOKENIZER_PRE,   tokenizer_pre, false);

        if (tokenizer_model == "no_vocab" || tokenizer_model == "none") {
            vocab.type = LLAMA_VOCAB_TYPE_NONE;

            // default special tokens
            vocab.special_bos_id  = LLAMA_TOKEN_NULL;
            vocab.special_eos_id  = LLAMA_TOKEN_NULL;
            vocab.special_unk_id  = LLAMA_TOKEN_NULL;
            vocab.special_sep_id  = LLAMA_TOKEN_NULL;
            vocab.special_pad_id  = LLAMA_TOKEN_NULL;
            vocab.special_cls_id  = LLAMA_TOKEN_NULL;
            vocab.special_mask_id = LLAMA_TOKEN_NULL;
            vocab.linefeed_id     = LLAMA_TOKEN_NULL;

            // read vocab size from metadata
            if (!ml.get_key(LLM_KV_VOCAB_SIZE, vocab.n_vocab, false)) {
                vocab.n_vocab = 0;
                LLAMA_LOG_WARN("%s: there is no vocab_size in metadata, vocab.n_vocab will be set to %u\n", __func__, vocab.n_vocab);
            }
            return;
        }

        if (tokenizer_model == "llama") {
            vocab.type = LLAMA_VOCAB_TYPE_SPM;

            // default special tokens
            vocab.special_bos_id  = 1;
            vocab.special_eos_id  = 2;
            vocab.special_unk_id  = 0;
            vocab.special_sep_id  = LLAMA_TOKEN_NULL;
            vocab.special_pad_id  = LLAMA_TOKEN_NULL;
            vocab.special_cls_id  = LLAMA_TOKEN_NULL;
            vocab.special_mask_id = LLAMA_TOKEN_NULL;
        } else if (tokenizer_model == "bert") {
            vocab.type = LLAMA_VOCAB_TYPE_WPM;

            // default special tokens
            vocab.special_bos_id  = LLAMA_TOKEN_NULL;
            vocab.special_eos_id  = LLAMA_TOKEN_NULL;
            vocab.special_unk_id  = 100;
            vocab.special_sep_id  = 102;
            vocab.special_pad_id  = 0;
            vocab.special_cls_id  = 101;
            vocab.special_mask_id = 103;
        } else if (tokenizer_model == "gpt2") {
            vocab.type = LLAMA_VOCAB_TYPE_BPE;

            // read bpe merges and populate bpe ranks
            const int merges_keyidx = gguf_find_key(ctx, kv(LLM_KV_TOKENIZER_MERGES).c_str());
            if (merges_keyidx == -1) {
                throw std::runtime_error("cannot find tokenizer merges in model file\n");
            }

            const int n_merges = gguf_get_arr_n(ctx, merges_keyidx);
            for (int i = 0; i < n_merges; i++) {
                const std::string word = gguf_get_arr_str(ctx, merges_keyidx, i);
                GGML_ASSERT(unicode_cpts_from_utf8(word).size() > 0);

                std::string first;
                std::string second;

                const size_t pos = word.find(' ', 1);

                if (pos != std::string::npos) {
                    first  = word.substr(0, pos);
                    second = word.substr(pos + 1);
                }

                vocab.bpe_ranks.emplace(std::make_pair(first, second), i);
            }

            // default special tokens
            vocab.special_bos_id  = 11;
            vocab.special_eos_id  = 11;
            vocab.special_unk_id  = LLAMA_TOKEN_NULL;
            vocab.special_sep_id  = LLAMA_TOKEN_NULL;
            vocab.special_pad_id  = LLAMA_TOKEN_NULL;
            vocab.special_cls_id  = LLAMA_TOKEN_NULL;
            vocab.special_mask_id = LLAMA_TOKEN_NULL;
        } else if (tokenizer_model == "t5") {
            vocab.type = LLAMA_VOCAB_TYPE_UGM;

            // default special tokens
            vocab.special_bos_id  = LLAMA_TOKEN_NULL;
            vocab.special_eos_id  = 1;
            vocab.special_unk_id  = 2;
            vocab.special_sep_id  = LLAMA_TOKEN_NULL;
            vocab.special_pad_id  = 0;
            vocab.special_cls_id  = LLAMA_TOKEN_NULL;
            vocab.special_mask_id = LLAMA_TOKEN_NULL;

            const int precompiled_charsmap_keyidx = gguf_find_key(ctx, kv(LLM_KV_TOKENIZER_PRECOMPILED_CHARSMAP).c_str());
            if (precompiled_charsmap_keyidx != -1) {
                size_t n_precompiled_charsmap = gguf_get_arr_n(ctx, precompiled_charsmap_keyidx);
                const char * precompiled_charsmap = (const char *) gguf_get_arr_data(ctx, precompiled_charsmap_keyidx);
                vocab.precompiled_charsmap.assign(precompiled_charsmap, precompiled_charsmap + n_precompiled_charsmap);
#ifdef IS_BIG_ENDIAN
                // correct endiannes of data in precompiled_charsmap binary blob
                uint32_t * xcda_blob_size = (uint32_t *) &vocab.precompiled_charsmap[0];
                *xcda_blob_size = __builtin_bswap32(*xcda_blob_size);
                assert(*xcda_blob_size + sizeof(uint32_t) < n_precompiled_charsmap);
                size_t xcda_array_size = *xcda_blob_size / sizeof(uint32_t);
                uint32_t * xcda_array = (uint32_t *) &vocab.precompiled_charsmap[sizeof(uint32_t)];
                for (size_t i = 0; i < xcda_array_size; ++i) {
                    xcda_array[i] = __builtin_bswap32(xcda_array[i]);
                }
#endif
            }
        } else if (tokenizer_model == "rwkv") {
            vocab.type = LLAMA_VOCAB_TYPE_RWKV;

            // default special tokens
            vocab.special_bos_id = LLAMA_TOKEN_NULL;
            vocab.special_eos_id = LLAMA_TOKEN_NULL;
            vocab.special_unk_id = LLAMA_TOKEN_NULL;
            vocab.special_sep_id = LLAMA_TOKEN_NULL;
            vocab.special_pad_id = LLAMA_TOKEN_NULL;
        } else {
            throw std::runtime_error(format("unknown tokenizer: '%s'", tokenizer_model.c_str()));
        }

        // for now, only BPE models have pre-tokenizers
        if (vocab.type == LLAMA_VOCAB_TYPE_BPE) {
            vocab.tokenizer_add_space_prefix = false;
            vocab.tokenizer_clean_spaces = true;
            if (tokenizer_pre == "default") {
                vocab.type_pre = LLAMA_VOCAB_PRE_TYPE_DEFAULT;
            } else if (
                    tokenizer_pre == "llama3"   ||
                    tokenizer_pre == "llama-v3" ||
                    tokenizer_pre == "llama-bpe"||
                    tokenizer_pre == "falcon3") {
                vocab.type_pre = LLAMA_VOCAB_PRE_TYPE_LLAMA3;
                vocab.tokenizer_ignore_merges = true;
                vocab.tokenizer_add_bos = true;
            } else if (
                    tokenizer_pre == "deepseek-llm") {
                vocab.type_pre = LLAMA_VOCAB_PRE_TYPE_DEEPSEEK_LLM;
                vocab.tokenizer_clean_spaces = false;
            } else if (
                    tokenizer_pre == "deepseek-coder") {
                vocab.type_pre = LLAMA_VOCAB_PRE_TYPE_DEEPSEEK_CODER;
                vocab.tokenizer_clean_spaces = false;
            } else if (
                    tokenizer_pre == "deepseek-v3") {
                vocab.type_pre = LLAMA_VOCAB_PRE_TYPE_DEEPSEEK3_LLM;
                vocab.tokenizer_clean_spaces = false;
            } else if (
                    tokenizer_pre == "falcon") {
                vocab.type_pre = LLAMA_VOCAB_PRE_TYPE_FALCON;
            } else if (
                    tokenizer_pre == "mpt") {
                vocab.type_pre = LLAMA_VOCAB_PRE_TYPE_MPT;
            } else if (
                    tokenizer_pre == "starcoder") {
                vocab.type_pre = LLAMA_VOCAB_PRE_TYPE_STARCODER;
            } else if (
                    tokenizer_pre == "gpt-2"   ||
                    tokenizer_pre == "phi-2"   ||
                    tokenizer_pre == "jina-es" ||
                    tokenizer_pre == "jina-de" ||
                    tokenizer_pre == "gigachat"   ||
                    tokenizer_pre == "jina-v1-en" ||
                    tokenizer_pre == "jina-v2-es" ||
                    tokenizer_pre == "jina-v2-de" ||
                    tokenizer_pre == "jina-v2-code" ||
                    tokenizer_pre == "roberta-bpe") {
                vocab.type_pre = LLAMA_VOCAB_PRE_TYPE_GPT2;
            } else if (
                    tokenizer_pre == "refact") {
                vocab.type_pre = LLAMA_VOCAB_PRE_TYPE_REFACT;
            } else if (
                tokenizer_pre == "command-r") {
                vocab.type_pre = LLAMA_VOCAB_PRE_TYPE_COMMAND_R;
                vocab.tokenizer_clean_spaces = false;
            } else if (
                tokenizer_pre == "qwen2") {
                vocab.type_pre = LLAMA_VOCAB_PRE_TYPE_QWEN2;
                vocab.tokenizer_clean_spaces = false;
            } else if (
                tokenizer_pre == "stablelm2") {
                vocab.type_pre = LLAMA_VOCAB_PRE_TYPE_STABLELM2;
            } else if (
                tokenizer_pre == "olmo") {
                vocab.type_pre = LLAMA_VOCAB_PRE_TYPE_OLMO;
            } else if (
                tokenizer_pre == "dbrx") {
                vocab.type_pre = LLAMA_VOCAB_PRE_TYPE_DBRX;
            } else if (
                tokenizer_pre == "smaug-bpe") {
                vocab.type_pre = LLAMA_VOCAB_PRE_TYPE_SMAUG;
            } else if (
                tokenizer_pre == "poro-chat") {
                vocab.type_pre = LLAMA_VOCAB_PRE_TYPE_PORO;
                vocab.tokenizer_clean_spaces = false;
            } else if (
                tokenizer_pre == "chatglm-bpe") {
                vocab.type_pre = LLAMA_VOCAB_PRE_TYPE_CHATGLM4;
                vocab.special_bos_id = LLAMA_TOKEN_NULL;
            } else if (
                tokenizer_pre == "viking") {
                vocab.type_pre = LLAMA_VOCAB_PRE_TYPE_VIKING;
                vocab.tokenizer_clean_spaces = false;
            } else if (
                tokenizer_pre == "jais") {
                vocab.type_pre = LLAMA_VOCAB_PRE_TYPE_JAIS;
            } else if (
                tokenizer_pre == "tekken") {
                vocab.type_pre = LLAMA_VOCAB_PRE_TYPE_TEKKEN;
                vocab.tokenizer_clean_spaces = false;
                vocab.tokenizer_ignore_merges = true;
                vocab.tokenizer_add_bos = true;
            } else if (
                tokenizer_pre == "smollm") {
                vocab.type_pre = LLAMA_VOCAB_PRE_TYPE_SMOLLM;
                vocab.tokenizer_clean_spaces = false;
            } else if (
                tokenizer_pre == "codeshell") {
                vocab.type_pre = LLAMA_VOCAB_PRE_TYPE_CODESHELL;
            } else if (
                tokenizer_pre == "bloom") {
                vocab.type_pre = LLAMA_VOCAB_PRE_TYPE_BLOOM;
            } else if (
                tokenizer_pre == "gpt3-finnish") {
                vocab.type_pre = LLAMA_VOCAB_PRE_TYPE_GPT3_FINNISH;
            } else if (
                tokenizer_pre == "exaone") {
                vocab.type_pre = LLAMA_VOCAB_PRE_TYPE_EXAONE;
            } else if (
                tokenizer_pre == "chameleon") {
                vocab.type_pre = LLAMA_VOCAB_PRE_TYPE_CHAMELEON;
                vocab.tokenizer_add_bos = true;
                vocab.tokenizer_clean_spaces = false;
            } else if (
                tokenizer_pre == "minerva-7b") {
                vocab.type_pre = LLAMA_VOCAB_PRE_TYPE_MINERVA;
            } else if (
                tokenizer_pre == "megrez") {
                vocab.type_pre = LLAMA_VOCAB_PRE_TYPE_QWEN2;
            } else {
                LLAMA_LOG_WARN("%s: missing or unrecognized pre-tokenizer type, using: 'default'\n", __func__);
                vocab.type_pre = LLAMA_VOCAB_PRE_TYPE_DEFAULT;
            }
        } else if (vocab.type == LLAMA_VOCAB_TYPE_SPM) {
            vocab.type_pre = LLAMA_VOCAB_PRE_TYPE_DEFAULT;
            vocab.tokenizer_add_space_prefix = true;
            vocab.tokenizer_clean_spaces = false;
            vocab.tokenizer_add_bos = true;
            vocab.tokenizer_add_eos = false;
        } else if (vocab.type == LLAMA_VOCAB_TYPE_WPM) {
            vocab.type_pre = LLAMA_VOCAB_PRE_TYPE_DEFAULT;
            vocab.tokenizer_add_space_prefix = false;
            vocab.tokenizer_clean_spaces = true;
            vocab.tokenizer_add_bos = true;
            vocab.tokenizer_add_eos = false;
        } else if (vocab.type == LLAMA_VOCAB_TYPE_UGM) {
            vocab.type_pre = LLAMA_VOCAB_PRE_TYPE_DEFAULT;
            vocab.tokenizer_add_bos = false;
            vocab.tokenizer_add_eos = true;
        } else if (vocab.type == LLAMA_VOCAB_TYPE_RWKV) {
            vocab.type_pre = LLAMA_VOCAB_PRE_TYPE_DEFAULT;
            vocab.tokenizer_add_space_prefix = false;
            vocab.tokenizer_clean_spaces = false;
            vocab.tokenizer_add_bos = false;
            vocab.tokenizer_add_eos = false;
        } else {
            vocab.type_pre = LLAMA_VOCAB_PRE_TYPE_DEFAULT;
        }

        ml.get_key(LLM_KV_TOKENIZER_ADD_PREFIX,      vocab.tokenizer_add_space_prefix,         false);
        ml.get_key(LLM_KV_TOKENIZER_REMOVE_EXTRA_WS, vocab.tokenizer_remove_extra_whitespaces, false);
    }

    const int token_idx = gguf_find_key(ctx, kv(LLM_KV_TOKENIZER_LIST).c_str());
    if (token_idx == -1) {
        throw std::runtime_error("cannot find tokenizer vocab in model file\n");
    }

    const float * scores = nullptr;
    const int score_idx = gguf_find_key(ctx, kv(LLM_KV_TOKENIZER_SCORES).c_str());
    if (score_idx != -1) {
        scores = (const float * ) gguf_get_arr_data(ctx, score_idx);
    }

    const int * toktypes = nullptr;
    const int toktype_idx = gguf_find_key(ctx, kv(LLM_KV_TOKENIZER_TOKEN_TYPE).c_str());
    if (toktype_idx != -1) {
        toktypes = (const int * ) gguf_get_arr_data(ctx, toktype_idx);
    }

    const uint32_t n_vocab = gguf_get_arr_n(ctx, token_idx);

    vocab.n_vocab = n_vocab;
    vocab.id_to_token.resize(n_vocab);

    for (uint32_t i = 0; i < n_vocab; i++) {
        std::string word = gguf_get_arr_str(ctx, token_idx, i);
        if (word.empty()) {
            LLAMA_LOG_WARN("%s: empty token at index %u\n", __func__, i);
            word = "[EMPTY_" + std::to_string(i) + "]";
        }

        vocab.token_to_id[word] = i;
        vocab.max_token_len = std::max(vocab.max_token_len, (int) word.size());

        auto & token_data = vocab.id_to_token[i];
        token_data.text  = std::move(word);
        token_data.score = scores ? scores[i] : 0.0f;
        token_data.attr  = LLAMA_TOKEN_ATTR_NORMAL;

        if (toktypes) {  //TODO: remove, required until per token attributes are available from GGUF file
            switch(toktypes[i]) {
                case LLAMA_TOKEN_TYPE_UNKNOWN:      token_data.attr = LLAMA_TOKEN_ATTR_UNKNOWN;      break;
                case LLAMA_TOKEN_TYPE_UNUSED:       token_data.attr = LLAMA_TOKEN_ATTR_UNUSED;       break;
                case LLAMA_TOKEN_TYPE_NORMAL:       token_data.attr = LLAMA_TOKEN_ATTR_NORMAL;       break;
                case LLAMA_TOKEN_TYPE_CONTROL:      token_data.attr = LLAMA_TOKEN_ATTR_CONTROL;      break;
                case LLAMA_TOKEN_TYPE_USER_DEFINED: token_data.attr = LLAMA_TOKEN_ATTR_USER_DEFINED; break;
                case LLAMA_TOKEN_TYPE_BYTE:         token_data.attr = LLAMA_TOKEN_ATTR_BYTE;         break;
                case LLAMA_TOKEN_TYPE_UNDEFINED:    token_data.attr = LLAMA_TOKEN_ATTR_UNDEFINED;    break;
                default:                            token_data.attr = LLAMA_TOKEN_ATTR_UNDEFINED;    break;
            }
        }
    }
    GGML_ASSERT(vocab.id_to_token.size() == vocab.token_to_id.size());

    vocab.init_tokenizer();

    // determine the newline token: LLaMA "<0x0A>" == 10 == '\n', Falcon 193 == '\n'
    if (vocab.type == LLAMA_VOCAB_TYPE_SPM) {
        try {
            vocab.linefeed_id = llama_byte_to_token_impl(vocab, '\n');
        } catch (const std::exception & e) {
            LLAMA_LOG_WARN("%s: SPM vocabulary, but newline token not found: %s! Using special_pad_id instead.", __func__, e.what());
            vocab.linefeed_id = vocab.special_pad_id;
        }
    } else if (vocab.type == LLAMA_VOCAB_TYPE_WPM) {
        vocab.linefeed_id = vocab.special_pad_id;
    } else if (vocab.type == LLAMA_VOCAB_TYPE_RWKV) {
        const std::vector<int> ids = llama_tokenize_internal(vocab, "\n", false);
        GGML_ASSERT(!ids.empty() && "model vocab missing newline token");
        vocab.linefeed_id = ids[0];
    } else {
        const std::vector<int> ids = llama_tokenize_internal(vocab, "\xC4\x8A", false); // U+010A

        //GGML_ASSERT(!ids.empty() && "model vocab missing newline token");
        if (ids.empty()) {
            LLAMA_LOG_WARN("%s: model vocab missing newline token, using special_pad_id instead\n", __func__);
            vocab.linefeed_id = vocab.special_pad_id;
        } else {
            vocab.linefeed_id = ids[0];
        }
    }

    // special tokens
    {
        const std::vector<std::pair<enum llm_kv, int32_t &>> special_token_types = {
            { LLM_KV_TOKENIZER_BOS_ID,     vocab.special_bos_id     },
            { LLM_KV_TOKENIZER_EOS_ID,     vocab.special_eos_id     },
            { LLM_KV_TOKENIZER_EOT_ID,     vocab.special_eot_id     },
            { LLM_KV_TOKENIZER_EOM_ID,     vocab.special_eom_id     },
            { LLM_KV_TOKENIZER_UNK_ID,     vocab.special_unk_id     },
            { LLM_KV_TOKENIZER_SEP_ID,     vocab.special_sep_id     },
            { LLM_KV_TOKENIZER_PAD_ID,     vocab.special_pad_id     },
            { LLM_KV_TOKENIZER_CLS_ID,     vocab.special_cls_id     },
            { LLM_KV_TOKENIZER_MASK_ID,    vocab.special_mask_id    },
            { LLM_KV_TOKENIZER_FIM_PRE_ID, vocab.special_fim_pre_id },
            { LLM_KV_TOKENIZER_FIM_SUF_ID, vocab.special_fim_suf_id },
            { LLM_KV_TOKENIZER_FIM_MID_ID, vocab.special_fim_mid_id },
            { LLM_KV_TOKENIZER_FIM_PAD_ID, vocab.special_fim_pad_id },
            { LLM_KV_TOKENIZER_FIM_REP_ID, vocab.special_fim_rep_id },
            { LLM_KV_TOKENIZER_FIM_SEP_ID, vocab.special_fim_sep_id },

            // deprecated
            { LLM_KV_TOKENIZER_PREFIX_ID, vocab.special_fim_pre_id },
            { LLM_KV_TOKENIZER_SUFFIX_ID, vocab.special_fim_suf_id },
            { LLM_KV_TOKENIZER_MIDDLE_ID, vocab.special_fim_mid_id },
        };

        for (const auto & it : special_token_types) {
            const std::string & key = kv(std::get<0>(it));
            int32_t & id = std::get<1>(it);

            uint32_t new_id;
            if (!ml.get_key(std::get<0>(it), new_id, false)) {
                continue;
            }
            if (new_id >= vocab.id_to_token.size()) {
                LLAMA_LOG_WARN("%s: bad special token: '%s' = %ud, using default id %d\n",
                    __func__, key.c_str(), new_id, id);
            } else {
                id = new_id;
            }
        }

        // Handle add_bos_token and add_eos_token
        {
            bool temp = true;

            if (ml.get_key(LLM_KV_TOKENIZER_ADD_BOS, temp, false)) {
                vocab.tokenizer_add_bos = temp;
            }
            if (ml.get_key(LLM_KV_TOKENIZER_ADD_EOS, temp, false)) {
                vocab.tokenizer_add_eos = temp;
            }
        }

        // auto-detect special tokens by text
        // TODO: convert scripts should provide these tokens through the KV metadata LLM_KV_TOKENIZER_...
        //       for now, we apply this workaround to find the tokens based on their text

        for (const auto & t : vocab.token_to_id) {
            // find EOT token: "<|eot_id|>", "<|im_end|>", "<end_of_turn>", etc.
            if (vocab.special_eot_id == LLAMA_TOKEN_NULL) {
                if (false
                        || t.first == "<|eot_id|>"
                        || t.first == "<|im_end|>"
                        || t.first == "<|end|>"
                        || t.first == "<end_of_turn>"
                        || t.first == "<|endoftext|>"
                        || t.first == "<EOT>"
                        || t.first == "<｜end▁of▁sentence｜>" // DeepSeek
                   ) {
                    vocab.special_eot_id = t.second;
                    if ((vocab.id_to_token[t.second].attr & LLAMA_TOKEN_ATTR_CONTROL) == 0) {
                        LLAMA_LOG_WARN("%s: control-looking token: %6d '%s' was not control-type; this is probably a bug in the model. its type will be overridden\n",
                                __func__, t.second, t.first.c_str());
                        vocab.id_to_token[t.second].attr = LLAMA_TOKEN_ATTR_CONTROL;
                    }
                }
            }

            // find EOM token: "<|eom_id|>"
            if (vocab.special_eom_id == LLAMA_TOKEN_NULL) {
                if (false
                        || t.first == "<|eom_id|>"
                        ) {
                    vocab.special_eom_id = t.second;
                    if ((vocab.id_to_token[t.second].attr & LLAMA_TOKEN_ATTR_CONTROL) == 0) {
                        LLAMA_LOG_WARN("%s: control-looking token: %6d '%s' was not control-type; this is probably a bug in the model. its type will be overridden\n",
                                __func__, t.second, t.first.c_str());
                        vocab.id_to_token[t.second].attr = LLAMA_TOKEN_ATTR_CONTROL;
                    }
                }
            }

            // find FIM_PRE token: "<|fim_prefix|>", "<fim-prefix>", "<PRE>", etc.
            if (vocab.special_fim_pre_id == LLAMA_TOKEN_NULL) {
                if (false
                        || t.first == "<|fim_prefix|>"  // Qwen
                        || t.first == "<fim-prefix>"
                        || t.first == "<｜fim▁begin｜>" // DeepSeek
                        || t.first == "<PRE>"
                        ) {
                    vocab.special_fim_pre_id = t.second;
                    if ((vocab.id_to_token[t.second].attr & LLAMA_TOKEN_ATTR_CONTROL) == 0) {
                        LLAMA_LOG_WARN("%s: control-looking token: %6d '%s' was not control-type; this is probably a bug in the model. its type will be overridden\n",
                                __func__, t.second, t.first.c_str());
                        vocab.id_to_token[t.second].attr = LLAMA_TOKEN_ATTR_CONTROL;
                    }
                }
            }

            // find FIM_SUF token: "<|fim_suffix|>", "<fim-suffix>", "<SUF>", etc.
            if (vocab.special_fim_suf_id == LLAMA_TOKEN_NULL) {
                if (false
                        || t.first == "<|fim_suffix|>" // Qwen
                        || t.first == "<fim-suffix>"
                        || t.first == "<｜fim▁hole｜>" // DeepSeek
                        || t.first == "<SUF>"
                        ) {
                    vocab.special_fim_suf_id = t.second;
                    if ((vocab.id_to_token[t.second].attr & LLAMA_TOKEN_ATTR_CONTROL) == 0) {
                        LLAMA_LOG_WARN("%s: control-looking token: %6d '%s' was not control-type; this is probably a bug in the model. its type will be overridden\n",
                                __func__, t.second, t.first.c_str());
                        vocab.id_to_token[t.second].attr = LLAMA_TOKEN_ATTR_CONTROL;
                    }
                }
            }

            // find FIM_MID token: "<|fim_middle|>", "<fim-middle>", "<MID>", etc.
            if (vocab.special_fim_mid_id == LLAMA_TOKEN_NULL) {
                if (false
                        || t.first == "<|fim_middle|>" // Qwen
                        || t.first == "<fim-middle>"
                        || t.first == "<｜fim▁end｜>"  // DeepSeek
                        || t.first == "<MID>"
                        ) {
                    vocab.special_fim_mid_id = t.second;
                    if ((vocab.id_to_token[t.second].attr & LLAMA_TOKEN_ATTR_CONTROL) == 0) {
                        LLAMA_LOG_WARN("%s: control-looking token: %6d '%s' was not control-type; this is probably a bug in the model. its type will be overridden\n",
                                __func__, t.second, t.first.c_str());
                        vocab.id_to_token[t.second].attr = LLAMA_TOKEN_ATTR_CONTROL;
                    }
                }
            }

            // find FIM_PAD token: "<|fim_pad|>", "<fim-pad>", "<PAD>", etc.
            if (vocab.special_fim_pad_id == LLAMA_TOKEN_NULL) {
                if (false
                        || t.first == "<|fim_pad|>" // Qwen
                        || t.first == "<fim-pad>"
                        || t.first == "<PAD>"
                        ) {
                    vocab.special_fim_pad_id = t.second;
                    if ((vocab.id_to_token[t.second].attr & LLAMA_TOKEN_ATTR_CONTROL) == 0) {
                        LLAMA_LOG_WARN("%s: control-looking token: %6d '%s' was not control-type; this is probably a bug in the model. its type will be overridden\n",
                                __func__, t.second, t.first.c_str());
                        vocab.id_to_token[t.second].attr = LLAMA_TOKEN_ATTR_CONTROL;
                    }
                }
            }

            // find FIM_REP token: "<|fim_repo|>", "<fim-repo>", "<REP>", etc.
            if (vocab.special_fim_rep_id == LLAMA_TOKEN_NULL) {
                if (false
                        || t.first == "<|fim_repo|>"  // Qwen
                        || t.first == "<|repo_name|>"
                        || t.first == "<fim-repo>"
                        || t.first == "<REPO>"
                        ) {
                    vocab.special_fim_rep_id = t.second;
                    if ((vocab.id_to_token[t.second].attr & LLAMA_TOKEN_ATTR_CONTROL) == 0) {
                        LLAMA_LOG_WARN("%s: control-looking token: %6d '%s' was not control-type; this is probably a bug in the model. its type will be overridden\n",
                                __func__, t.second, t.first.c_str());
                        vocab.id_to_token[t.second].attr = LLAMA_TOKEN_ATTR_CONTROL;
                    }
                }
            }

            // find FIM_SEP token: "<|file_sep|>"
            if (vocab.special_fim_sep_id == LLAMA_TOKEN_NULL) {
                if (false
                        || t.first == "<|file_sep|>" // Qwen
                        ) {
                    vocab.special_fim_sep_id = t.second;
                    if ((vocab.id_to_token[t.second].attr & LLAMA_TOKEN_ATTR_CONTROL) == 0) {
                        LLAMA_LOG_WARN("%s: control-looking token: %6d '%s' was not control-type; this is probably a bug in the model. its type will be overridden\n",
                                __func__, t.second, t.first.c_str());
                        vocab.id_to_token[t.second].attr = LLAMA_TOKEN_ATTR_CONTROL;
                    }
                }
            }
        }

        // maintain a list of tokens that cause end-of-generation
        // this is currently determined based on the token text, which is obviously not ideal
        // ref: https://github.com/ggerganov/llama.cpp/issues/9606
        vocab.special_eog_ids.clear();

        if (vocab.special_fim_pad_id != LLAMA_TOKEN_NULL && vocab.special_eog_ids.count(vocab.special_fim_pad_id) == 0) {
            vocab.special_eog_ids.insert(vocab.special_fim_pad_id);
        }

        if (vocab.special_fim_rep_id != LLAMA_TOKEN_NULL && vocab.special_eog_ids.count(vocab.special_fim_rep_id) == 0) {
            vocab.special_eog_ids.insert(vocab.special_fim_rep_id);
        }

        if (vocab.special_fim_sep_id != LLAMA_TOKEN_NULL && vocab.special_eog_ids.count(vocab.special_fim_sep_id) == 0) {
            vocab.special_eog_ids.insert(vocab.special_fim_sep_id);
        }

        for (const auto & t : vocab.token_to_id) {
            if (false
                    || t.first == "<|eot_id|>"
                    || t.first == "<|im_end|>"
                    || t.first == "<|end|>"
                    || t.first == "<end_of_turn>"
                    || t.first == "<|endoftext|>"
                    || t.first == "<|eom_id|>"
                    || t.first == "<EOT>"
               ) {
                vocab.special_eog_ids.insert(t.second);
                if ((vocab.id_to_token[t.second].attr & LLAMA_TOKEN_ATTR_CONTROL) == 0) {
                    LLAMA_LOG_WARN("%s: control-looking token: %6d '%s' was not control-type; this is probably a bug in the model. its type will be overridden\n",
                            __func__, t.second, t.first.c_str());
                    vocab.id_to_token[t.second].attr = LLAMA_TOKEN_ATTR_CONTROL;
                }
            } else {
                // token is control, but not marked as EOG -> print a debug log
                if (vocab.id_to_token[t.second].attr & LLAMA_TOKEN_ATTR_CONTROL && vocab.special_eog_ids.count(t.second) == 0) {
                    LLAMA_LOG_DEBUG("%s: control token: %6d '%s' is not marked as EOG\n",
                            __func__, t.second, t.first.c_str());
                }
            }
        }

        // sanity checks
        if (vocab.special_eos_id != LLAMA_TOKEN_NULL && vocab.special_eog_ids.count(vocab.special_eos_id) == 0) {
            vocab.special_eog_ids.insert(vocab.special_eos_id);
            LLAMA_LOG_WARN("%s: special_eos_id is not in special_eog_ids - the tokenizer config may be incorrect\n", __func__);
        }

        if (vocab.special_eot_id != LLAMA_TOKEN_NULL && vocab.special_eog_ids.count(vocab.special_eot_id) == 0) {
            vocab.special_eog_ids.insert(vocab.special_eot_id);
            LLAMA_LOG_WARN("%s: special_eot_id is not in special_eog_ids - the tokenizer config may be incorrect\n", __func__);
        }

        if (vocab.special_eom_id != LLAMA_TOKEN_NULL && vocab.special_eog_ids.count(vocab.special_eom_id) == 0) {
            vocab.special_eog_ids.insert(vocab.special_eom_id);
            LLAMA_LOG_WARN("%s: special_eom_id is not in special_eog_ids - the tokenizer config may be incorrect\n", __func__);
        }
    }

    // build special tokens cache
    {
        for (llama_vocab::id id = 0; id < (llama_vocab::id)n_vocab; ++id) {
            if (vocab.id_to_token[id].attr & (LLAMA_TOKEN_ATTR_CONTROL | LLAMA_TOKEN_ATTR_USER_DEFINED | LLAMA_TOKEN_ATTR_UNKNOWN)) {
                vocab.cache_special_tokens.push_back(id);
            }
        }

        std::sort(vocab.cache_special_tokens.begin(), vocab.cache_special_tokens.end(),
            [&] (const llama_vocab::id a, const llama_vocab::id b) {
                return vocab.id_to_token[a].text.size() > vocab.id_to_token[b].text.size();
            }
        );

        LLAMA_LOG_INFO("%s: special tokens cache size = %u\n", __func__, (uint32_t)vocab.cache_special_tokens.size());
    }

    // build token to piece cache
    {
        size_t size_cache = 0;

        std::vector<llama_vocab::token> cache_token_to_piece(n_vocab);

        for (uint32_t id = 0; id < n_vocab; ++id) {
            cache_token_to_piece[id] = llama_token_to_piece(&model, id, true);

            size_cache += cache_token_to_piece[id].size();
        }

        std::swap(vocab.cache_token_to_piece, cache_token_to_piece);

        LLAMA_LOG_INFO("%s: token to piece cache size = %.4f MB\n", __func__, size_cache / 1024.0 / 1024.0);
    }

    // Handle per token attributes
    //NOTE: Each model customizes per token attributes.
    //NOTE: Per token attributes are missing from the GGUF file.
    //TODO: Extract attributes from GGUF file.
    {
        auto _contains_any = [] (const std::string &str, const std::vector<std::string> &substrs) -> bool {
            for (auto substr : substrs) {
                if (str.find(substr) < std::string::npos) {
                    return true;
                }
            }
            return false;
        };

        auto _set_tokenid_attr = [&] (const llama_vocab::id id, llama_token_attr attr, bool value) {
            uint32_t current = vocab.id_to_token.at(id).attr;
            current = value ? (current | attr) : (current & ~attr);
            vocab.id_to_token[id].attr = (llama_token_attr) current;
        };

        auto _set_token_attr = [&] (const std::string & token, llama_token_attr attr, bool value) {
            _set_tokenid_attr(vocab.token_to_id.at(token), attr, value);
        };

        std::string model_name;
        std::string tokenizer_pre;

        ml.get_key(LLM_KV_GENERAL_NAME, model_name, false);
        ml.get_key(LLM_KV_TOKENIZER_PRE, tokenizer_pre, false);

        // model name to lowercase
        std::transform(model_name.begin(), model_name.end(), model_name.begin(),
            [] (const std::string::value_type x) {
                return std::tolower(x);
            }
        );

        // set attributes by model/tokenizer name
        if (_contains_any(tokenizer_pre, {"jina-v2-de", "jina-v2-es", "jina-v2-code"})) {
            _set_token_attr("<mask>", LLAMA_TOKEN_ATTR_LSTRIP, true);
        } else if (_contains_any(model_name, {"phi-3", "phi3"})) {
            for (auto id : vocab.cache_special_tokens) {
                _set_tokenid_attr(id, LLAMA_TOKEN_ATTR_RSTRIP, true);
            }
            for (auto token : {"</s>"}) {
                _set_token_attr(token, LLAMA_TOKEN_ATTR_RSTRIP, true);
            }
            for (auto token : {"<unk>", "<s>", "<|endoftext|>"}) {
                _set_token_attr(token, LLAMA_TOKEN_ATTR_RSTRIP, false);
            }
        }
    }
}

void llm_load_print_meta(llama_model_loader & ml, llama_model & model) {
    const auto & hparams = model.hparams;
    const auto & vocab   = model.vocab;

    const char * rope_scaling_type = LLAMA_ROPE_SCALING_TYPES.at(hparams.rope_scaling_type_train);

    auto print_f = [](const std::function<uint32_t(uint32_t)> & f, uint32_t n) {
        bool is_var = false;

        std::vector<uint32_t> v;
        for (uint32_t i = 0; i < n; ++i) {
            v.push_back(f(i));
            if (v[i] != v[0]) {
                is_var = true;
            }
        }

        std::stringstream ss;

        if (is_var) {
            ss << "[";
            for (uint32_t i = 0; i < n; ++i) {
                ss << v[i];
                if (i < n - 1) {
                    ss << ", ";
                }
            }
            ss << "]";
        } else {
            ss << v[0];
        }

        return ss.str();
    };

    // hparams
    LLAMA_LOG_INFO("%s: format           = %s\n",     __func__, llama_file_version_name(ml.fver));
    LLAMA_LOG_INFO("%s: arch             = %s\n",     __func__, llm_arch_name(model.arch));
    LLAMA_LOG_INFO("%s: vocab type       = %s\n",     __func__, llama_model_vocab_type_name(vocab.type));
    LLAMA_LOG_INFO("%s: n_vocab          = %u\n",     __func__, hparams.n_vocab);
    LLAMA_LOG_INFO("%s: n_merges         = %u\n",     __func__, (int) vocab.bpe_ranks.size());
    LLAMA_LOG_INFO("%s: vocab_only       = %d\n",     __func__, hparams.vocab_only);

    if (!hparams.vocab_only) {
        LLAMA_LOG_INFO("%s: n_ctx_train      = %u\n",     __func__, hparams.n_ctx_train);
        LLAMA_LOG_INFO("%s: n_embd           = %u\n",     __func__, hparams.n_embd);
        LLAMA_LOG_INFO("%s: n_layer          = %u\n",     __func__, hparams.n_layer);
        LLAMA_LOG_INFO("%s: n_head           = %s\n",     __func__, print_f([&](uint32_t il) { return hparams.n_head(il);    }, hparams.n_layer).c_str());
        LLAMA_LOG_INFO("%s: n_head_kv        = %s\n",     __func__, print_f([&](uint32_t il) { return hparams.n_head_kv(il); }, hparams.n_layer).c_str());
        LLAMA_LOG_INFO("%s: n_rot            = %u\n",     __func__, hparams.n_rot);
        LLAMA_LOG_INFO("%s: n_swa            = %u\n",     __func__, hparams.n_swa);
        LLAMA_LOG_INFO("%s: n_embd_head_k    = %u\n",     __func__, hparams.n_embd_head_k);
        LLAMA_LOG_INFO("%s: n_embd_head_v    = %u\n",     __func__, hparams.n_embd_head_v);
        LLAMA_LOG_INFO("%s: n_gqa            = %s\n",     __func__, print_f([&](uint32_t il) { return hparams.n_gqa(il);        }, hparams.n_layer).c_str());
        LLAMA_LOG_INFO("%s: n_embd_k_gqa     = %s\n",     __func__, print_f([&](uint32_t il) { return hparams.n_embd_k_gqa(il); }, hparams.n_layer).c_str());
        LLAMA_LOG_INFO("%s: n_embd_v_gqa     = %s\n",     __func__, print_f([&](uint32_t il) { return hparams.n_embd_v_gqa(il); }, hparams.n_layer).c_str());
        LLAMA_LOG_INFO("%s: f_norm_eps       = %.1e\n",   __func__, hparams.f_norm_eps);
        LLAMA_LOG_INFO("%s: f_norm_rms_eps   = %.1e\n",   __func__, hparams.f_norm_rms_eps);
        LLAMA_LOG_INFO("%s: f_clamp_kqv      = %.1e\n",   __func__, hparams.f_clamp_kqv);
        LLAMA_LOG_INFO("%s: f_max_alibi_bias = %.1e\n",   __func__, hparams.f_max_alibi_bias);
        LLAMA_LOG_INFO("%s: f_logit_scale    = %.1e\n",   __func__, hparams.f_logit_scale);
        LLAMA_LOG_INFO("%s: n_ff             = %s\n",     __func__, print_f([&](uint32_t il) { return hparams.n_ff(il); }, hparams.n_layer).c_str());
        LLAMA_LOG_INFO("%s: n_expert         = %u\n",     __func__, hparams.n_expert);
        LLAMA_LOG_INFO("%s: n_expert_used    = %u\n",     __func__, hparams.n_expert_used);
        LLAMA_LOG_INFO("%s: causal attn      = %d\n",     __func__, hparams.causal_attn);
        LLAMA_LOG_INFO("%s: pooling type     = %d\n",     __func__, hparams.pooling_type);
        LLAMA_LOG_INFO("%s: rope type        = %d\n",     __func__, hparams.rope_type);
        LLAMA_LOG_INFO("%s: rope scaling     = %s\n",     __func__, rope_scaling_type);
        LLAMA_LOG_INFO("%s: freq_base_train  = %.1f\n",   __func__, hparams.rope_freq_base_train);
        LLAMA_LOG_INFO("%s: freq_scale_train = %g\n",     __func__, hparams.rope_freq_scale_train);
        LLAMA_LOG_INFO("%s: n_ctx_orig_yarn  = %u\n",     __func__, hparams.n_ctx_orig_yarn);
        LLAMA_LOG_INFO("%s: rope_finetuned   = %s\n",     __func__, hparams.rope_finetuned ? "yes" : "unknown");
        LLAMA_LOG_INFO("%s: ssm_d_conv       = %u\n",     __func__, hparams.ssm_d_conv);
        LLAMA_LOG_INFO("%s: ssm_d_inner      = %u\n",     __func__, hparams.ssm_d_inner);
        LLAMA_LOG_INFO("%s: ssm_d_state      = %u\n",     __func__, hparams.ssm_d_state);
        LLAMA_LOG_INFO("%s: ssm_dt_rank      = %u\n",     __func__, hparams.ssm_dt_rank);
        LLAMA_LOG_INFO("%s: ssm_dt_b_c_rms   = %d\n",     __func__, hparams.ssm_dt_b_c_rms);
    }

    LLAMA_LOG_INFO("%s: model type       = %s\n",     __func__, llama_model_type_name(model).c_str());
    LLAMA_LOG_INFO("%s: model ftype      = %s\n",     __func__, llama_model_ftype_name(model).c_str());
    if (ml.n_elements >= 1e12) {
        LLAMA_LOG_INFO("%s: model params     = %.2f T\n", __func__, ml.n_elements*1e-12);
    } else if (ml.n_elements >= 1e9) {
        LLAMA_LOG_INFO("%s: model params     = %.2f B\n", __func__, ml.n_elements*1e-9);
    } else if (ml.n_elements >= 1e6) {
        LLAMA_LOG_INFO("%s: model params     = %.2f M\n", __func__, ml.n_elements*1e-6);
    } else {
        LLAMA_LOG_INFO("%s: model params     = %.2f K\n", __func__, ml.n_elements*1e-3);
    }
    if (ml.n_bytes < GiB) {
        LLAMA_LOG_INFO("%s: model size       = %.2f MiB (%.2f BPW) \n", __func__, ml.n_bytes/1024.0/1024.0,        ml.n_bytes*8.0/ml.n_elements);
    } else {
        LLAMA_LOG_INFO("%s: model size       = %.2f GiB (%.2f BPW) \n", __func__, ml.n_bytes/1024.0/1024.0/1024.0, ml.n_bytes*8.0/ml.n_elements);
    }

    // general kv
    LLAMA_LOG_INFO("%s: general.name     = %s\n",    __func__, model.name.c_str());

    // special tokens
    if (vocab.special_bos_id  != -1)    { LLAMA_LOG_INFO( "%s: BOS token        = %d '%s'\n", __func__, vocab.special_bos_id,     vocab.id_to_token[vocab.special_bos_id].text.c_str() );  }
    if (vocab.special_eos_id  != -1)    { LLAMA_LOG_INFO( "%s: EOS token        = %d '%s'\n", __func__, vocab.special_eos_id,     vocab.id_to_token[vocab.special_eos_id].text.c_str() );  }
    if (vocab.special_eot_id  != -1)    { LLAMA_LOG_INFO( "%s: EOT token        = %d '%s'\n", __func__, vocab.special_eot_id,     vocab.id_to_token[vocab.special_eot_id].text.c_str() );  }
    if (vocab.special_eom_id  != -1)    { LLAMA_LOG_INFO( "%s: EOM token        = %d '%s'\n", __func__, vocab.special_eom_id,     vocab.id_to_token[vocab.special_eom_id].text.c_str() );  }
    if (vocab.special_unk_id  != -1)    { LLAMA_LOG_INFO( "%s: UNK token        = %d '%s'\n", __func__, vocab.special_unk_id,     vocab.id_to_token[vocab.special_unk_id].text.c_str() );  }
    if (vocab.special_sep_id  != -1)    { LLAMA_LOG_INFO( "%s: SEP token        = %d '%s'\n", __func__, vocab.special_sep_id,     vocab.id_to_token[vocab.special_sep_id].text.c_str() );  }
    if (vocab.special_pad_id  != -1)    { LLAMA_LOG_INFO( "%s: PAD token        = %d '%s'\n", __func__, vocab.special_pad_id,     vocab.id_to_token[vocab.special_pad_id].text.c_str() );  }
    if (vocab.special_cls_id  != -1)    { LLAMA_LOG_INFO( "%s: CLS token        = %d '%s'\n", __func__, vocab.special_cls_id,     vocab.id_to_token[vocab.special_cls_id].text.c_str() );  }
    if (vocab.special_mask_id != -1)    { LLAMA_LOG_INFO( "%s: MASK token       = %d '%s'\n", __func__, vocab.special_mask_id,    vocab.id_to_token[vocab.special_mask_id].text.c_str() ); }

    if (vocab.linefeed_id != -1)        { LLAMA_LOG_INFO( "%s: LF token         = %d '%s'\n", __func__, vocab.linefeed_id,        vocab.id_to_token[vocab.linefeed_id].text.c_str() ); }

    if (vocab.special_fim_pre_id != -1) { LLAMA_LOG_INFO( "%s: FIM PRE token    = %d '%s'\n", __func__, vocab.special_fim_pre_id, vocab.id_to_token[vocab.special_fim_pre_id].text.c_str() ); }
    if (vocab.special_fim_suf_id != -1) { LLAMA_LOG_INFO( "%s: FIM SUF token    = %d '%s'\n", __func__, vocab.special_fim_suf_id, vocab.id_to_token[vocab.special_fim_suf_id].text.c_str() ); }
    if (vocab.special_fim_mid_id != -1) { LLAMA_LOG_INFO( "%s: FIM MID token    = %d '%s'\n", __func__, vocab.special_fim_mid_id, vocab.id_to_token[vocab.special_fim_mid_id].text.c_str() ); }
    if (vocab.special_fim_pad_id != -1) { LLAMA_LOG_INFO( "%s: FIM PAD token    = %d '%s'\n", __func__, vocab.special_fim_pad_id, vocab.id_to_token[vocab.special_fim_pad_id].text.c_str() ); }
    if (vocab.special_fim_rep_id != -1) { LLAMA_LOG_INFO( "%s: FIM REP token    = %d '%s'\n", __func__, vocab.special_fim_rep_id, vocab.id_to_token[vocab.special_fim_rep_id].text.c_str() ); }
    if (vocab.special_fim_sep_id != -1) { LLAMA_LOG_INFO( "%s: FIM SEP token    = %d '%s'\n", __func__, vocab.special_fim_sep_id, vocab.id_to_token[vocab.special_fim_sep_id].text.c_str() ); }

    for (const auto & id : vocab.special_eog_ids) {
        LLAMA_LOG_INFO( "%s: EOG token        = %d '%s'\n", __func__, id, vocab.id_to_token[id].text.c_str() );
    }

    LLAMA_LOG_INFO("%s: max token length = %d\n", __func__, vocab.max_token_len);

    if (model.arch == LLM_ARCH_DEEPSEEK) {
        LLAMA_LOG_INFO("%s: n_layer_dense_lead   = %d\n",     __func__, hparams.n_layer_dense_lead);
        LLAMA_LOG_INFO("%s: n_ff_exp             = %d\n",     __func__, hparams.n_ff_exp);
        LLAMA_LOG_INFO("%s: n_expert_shared      = %d\n",     __func__, hparams.n_expert_shared);
        LLAMA_LOG_INFO("%s: expert_weights_scale = %.1f\n",   __func__, hparams.expert_weights_scale);
    }

    if (model.arch == LLM_ARCH_DEEPSEEK2) {
        LLAMA_LOG_INFO("%s: n_layer_dense_lead   = %d\n",     __func__, hparams.n_layer_dense_lead);
        LLAMA_LOG_INFO("%s: n_lora_q             = %d\n",     __func__, hparams.n_lora_q);
        LLAMA_LOG_INFO("%s: n_lora_kv            = %d\n",     __func__, hparams.n_lora_kv);
        LLAMA_LOG_INFO("%s: n_ff_exp             = %d\n",     __func__, hparams.n_ff_exp);
        LLAMA_LOG_INFO("%s: n_expert_shared      = %d\n",     __func__, hparams.n_expert_shared);
        LLAMA_LOG_INFO("%s: expert_weights_scale = %.1f\n",   __func__, hparams.expert_weights_scale);
        LLAMA_LOG_INFO("%s: expert_weights_norm  = %d\n",     __func__, hparams.expert_weights_norm);
        LLAMA_LOG_INFO("%s: expert_gating_func   = %s\n",     __func__, llama_expert_gating_func_name((enum llama_expert_gating_func_type) hparams.expert_gating_func));
        LLAMA_LOG_INFO("%s: rope_yarn_log_mul    = %.4f\n",   __func__, hparams.rope_yarn_log_mul);
    }

    if (model.arch == LLM_ARCH_QWEN2MOE) {
        LLAMA_LOG_INFO("%s: n_ff_exp         = %d\n",     __func__, hparams.n_ff_exp);
        LLAMA_LOG_INFO("%s: n_ff_shexp       = %d\n",     __func__, hparams.n_ff_shexp);
    }

    if (model.arch == LLM_ARCH_MINICPM || model.arch == LLM_ARCH_GRANITE || model.arch == LLM_ARCH_GRANITE_MOE) {
        LLAMA_LOG_INFO("%s: f_embedding_scale = %f\n", __func__, hparams.f_embedding_scale);
        LLAMA_LOG_INFO("%s: f_residual_scale  = %f\n", __func__, hparams.f_residual_scale);
        LLAMA_LOG_INFO("%s: f_attention_scale = %f\n", __func__, hparams.f_attention_scale);
    }
}

//
// interface implementation
//

struct llama_model_params llama_model_default_params() {
    struct llama_model_params result = {
        /*.devices                     =*/ nullptr,
        /*.n_gpu_layers                =*/ 0,
        /*.split_mode                  =*/ LLAMA_SPLIT_MODE_LAYER,
        /*.main_gpu                    =*/ 0,
        /*.tensor_split                =*/ nullptr,
        /*.rpc_servers                 =*/ nullptr,
        /*.progress_callback           =*/ nullptr,
        /*.progress_callback_user_data =*/ nullptr,
        /*.kv_overrides                =*/ nullptr,
        /*.vocab_only                  =*/ false,
        /*.use_mmap                    =*/ true,
        /*.use_mlock                   =*/ false,
        /*.check_tensors               =*/ false,
    };

#ifdef GGML_USE_METAL
    // note: we usually have plenty of VRAM, so by default offload all layers to the GPU
    result.n_gpu_layers = 999;
#endif

    return result;
}

void llama_free_model(struct llama_model * model) {
    delete model;
}

enum llama_vocab_type llama_vocab_type(const struct llama_model * model) {
    return model->vocab.type;
}

int32_t llama_n_vocab(const struct llama_model * model) {
    return model->hparams.n_vocab;
}

int32_t llama_n_ctx_train(const struct llama_model * model) {
    return model->hparams.n_ctx_train;
}

int32_t llama_n_embd(const struct llama_model * model) {
    return model->hparams.n_embd;
}

int32_t llama_n_layer(const struct llama_model * model) {
    return model->hparams.n_layer;
}

int32_t llama_n_head(const struct llama_model * model) {
    return model->hparams.n_head();
}

enum llama_rope_type llama_rope_type(const struct llama_model * model) {
    switch (model->arch) {
        // these models do not use RoPE
        case LLM_ARCH_GPT2:
        case LLM_ARCH_GPTJ:
        case LLM_ARCH_MPT:
        case LLM_ARCH_REFACT:
        case LLM_ARCH_BLOOM:
        case LLM_ARCH_MAMBA:
        case LLM_ARCH_JINA_BERT_V2:
        case LLM_ARCH_T5:
        case LLM_ARCH_T5ENCODER:
        case LLM_ARCH_JAIS:
        case LLM_ARCH_RWKV6:
        case LLM_ARCH_WAVTOKENIZER_DEC:
            return LLAMA_ROPE_TYPE_NONE;

        // use what we call a normal RoPE, operating on pairs of consecutive head values
        case LLM_ARCH_LLAMA:
        case LLM_ARCH_MLLAMA:
        case LLM_ARCH_DECI:
        case LLM_ARCH_BAICHUAN:
        case LLM_ARCH_STARCODER:
        case LLM_ARCH_PLAMO:
        case LLM_ARCH_ORION:
        case LLM_ARCH_INTERNLM2:
        case LLM_ARCH_MINICPM:
        case LLM_ARCH_XVERSE:
        case LLM_ARCH_COMMAND_R:
        case LLM_ARCH_COHERE2:
        case LLM_ARCH_OLMO:
        case LLM_ARCH_ARCTIC:
        case LLM_ARCH_DEEPSEEK:
        case LLM_ARCH_DEEPSEEK2:
        case LLM_ARCH_CHATGLM:
        case LLM_ARCH_GRANITE:
        case LLM_ARCH_GRANITE_MOE:
        case LLM_ARCH_CHAMELEON:
        case LLM_ARCH_SOLAR:
            return LLAMA_ROPE_TYPE_NORM;

        // the pairs of head values are offset by n_rot/2
        case LLM_ARCH_FALCON:
        case LLM_ARCH_GROK:
        case LLM_ARCH_DBRX:
        case LLM_ARCH_BERT:
        case LLM_ARCH_NOMIC_BERT:
        case LLM_ARCH_STABLELM:
        case LLM_ARCH_BITNET:
        case LLM_ARCH_QWEN:
        case LLM_ARCH_QWEN2:
        case LLM_ARCH_QWEN2MOE:
        case LLM_ARCH_OLMO2:
        case LLM_ARCH_OLMOE:
        case LLM_ARCH_PHI2:
        case LLM_ARCH_PHI3:
        case LLM_ARCH_GEMMA:
        case LLM_ARCH_GEMMA2:
        case LLM_ARCH_STARCODER2:
        case LLM_ARCH_OPENELM:
        case LLM_ARCH_GPTNEOX:
        case LLM_ARCH_CODESHELL:
        case LLM_ARCH_NEMOTRON:
        case LLM_ARCH_EXAONE:
        case LLM_ARCH_MINICPM3:
            return LLAMA_ROPE_TYPE_NEOX;

        case LLM_ARCH_QWEN2VL:
            return LLAMA_ROPE_TYPE_MROPE;

        // all model arches should be listed explicitly here
        case LLM_ARCH_UNKNOWN:
            GGML_ABORT("unknown architecture");
    }

    return LLAMA_ROPE_TYPE_NONE;
}

float llama_rope_freq_scale_train(const struct llama_model * model) {
    return model->hparams.rope_freq_scale_train;
}

int32_t llama_model_meta_val_str(const struct llama_model * model, const char * key, char * buf, size_t buf_size) {
    const auto & it = model->gguf_kv.find(key);
    if (it == model->gguf_kv.end()) {
        if (buf_size > 0) {
            buf[0] = '\0';
        }
        return -1;
    }
    return snprintf(buf, buf_size, "%s", it->second.c_str());
}

int32_t llama_model_meta_count(const struct llama_model * model) {
    return (int)model->gguf_kv.size();
}

int32_t llama_model_meta_key_by_index(const struct llama_model * model, int i, char * buf, size_t buf_size) {
    if (i < 0 || i >= (int)model->gguf_kv.size()) {
        if (buf_size > 0) {
            buf[0] = '\0';
        }
        return -1;
    }
    auto it = model->gguf_kv.begin();
    std::advance(it, i);
    return snprintf(buf, buf_size, "%s", it->first.c_str());
}

int32_t llama_model_meta_val_str_by_index(const struct llama_model * model, int32_t i, char * buf, size_t buf_size) {
    if (i < 0 || i >= (int)model->gguf_kv.size()) {
        if (buf_size > 0) {
            buf[0] = '\0';
        }
        return -1;
    }
    auto it = model->gguf_kv.begin();
    std::advance(it, i);
    return snprintf(buf, buf_size, "%s", it->second.c_str());
}

int32_t llama_model_desc(const struct llama_model * model, char * buf, size_t buf_size) {
    return snprintf(buf, buf_size, "%s %s %s",
            llama_model_arch_name (*model).c_str(),
            llama_model_type_name (*model).c_str(),
            llama_model_ftype_name(*model).c_str());
}

uint64_t llama_model_size(const struct llama_model * model) {
    return model->n_bytes;
}

uint64_t llama_model_n_params(const struct llama_model * model) {
    return model->n_elements;
}

bool llama_model_has_encoder(const struct llama_model * model) {
    switch (model->arch) {
        case LLM_ARCH_T5:        return true;
        case LLM_ARCH_T5ENCODER: return true;
        default:                 return false;
    }
}

bool llama_model_has_decoder(const struct llama_model * model) {
    switch (model->arch) {
        case LLM_ARCH_T5ENCODER: return false;
        default:                 return true;
    }
}

llama_token llama_model_decoder_start_token(const struct llama_model * model) {
    return model->hparams.dec_start_token_id;
}

bool llama_model_is_recurrent(const struct llama_model * model) {
    switch (model->arch) {
        case LLM_ARCH_MAMBA:  return true;
        case LLM_ARCH_RWKV6:  return true;
        default:              return false;
    }
}
