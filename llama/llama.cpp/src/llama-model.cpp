#include "llama-model.h"

#include "llama-impl.h"
#include "llama-mmap.h"
#include "llama-batch.h"
#include "llama-cparams.h"
#include "llama-model-loader.h"

#include "llama-kv-cache-unified.h"
#include "llama-kv-cache-unified-iswa.h"
#include "llama-memory-hybrid.h"
#include "llama-memory-recurrent.h"

#include "ggml-cpp.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cfloat>
#include <cstring>
#include <cmath>
#include <functional>
#include <map>
#include <regex>
#include <sstream>
#include <stdexcept>

const char * llm_type_name(llm_type type) {
    switch (type) {
        case LLM_TYPE_14M:           return "14M";
        case LLM_TYPE_17M:           return "17M";
        case LLM_TYPE_22M:           return "22M";
        case LLM_TYPE_33M:           return "33M";
        case LLM_TYPE_60M:           return "60M";
        case LLM_TYPE_70M:           return "70M";
        case LLM_TYPE_80M:           return "80M";
        case LLM_TYPE_109M:          return "109M";
        case LLM_TYPE_137M:          return "137M";
        case LLM_TYPE_160M:          return "160M";
        case LLM_TYPE_190M:          return "190M";
        case LLM_TYPE_220M:          return "220M";
        case LLM_TYPE_250M:          return "250M";
        case LLM_TYPE_256M:          return "256M";
        case LLM_TYPE_270M:          return "270M";
        case LLM_TYPE_335M:          return "335M";
        case LLM_TYPE_350M:          return "350M";
        case LLM_TYPE_410M:          return "410M";
        case LLM_TYPE_450M:          return "450M";
        case LLM_TYPE_475M:          return "475M";
        case LLM_TYPE_700M:          return "700M";
        case LLM_TYPE_770M:          return "770M";
        case LLM_TYPE_780M:          return "780M";
        case LLM_TYPE_0_3B:          return "0.3B";
        case LLM_TYPE_0_5B:          return "0.5B";
        case LLM_TYPE_0_6B:          return "0.6B";
        case LLM_TYPE_1B:            return "1B";
        case LLM_TYPE_1_2B:          return "1.2B";
        case LLM_TYPE_1_3B:          return "1.3B";
        case LLM_TYPE_1_4B:          return "1.4B";
        case LLM_TYPE_1_5B:          return "1.5B";
        case LLM_TYPE_1_6B:          return "1.6B";
        case LLM_TYPE_1_7B:          return "1.7B";
        case LLM_TYPE_1_8B:          return "1.8B";
        case LLM_TYPE_2B:            return "2B";
        case LLM_TYPE_2_8B:          return "2.8B";
        case LLM_TYPE_2_9B:          return "2.9B";
        case LLM_TYPE_3B:            return "3B";
        case LLM_TYPE_4B:            return "4B";
        case LLM_TYPE_6B:            return "6B";
        case LLM_TYPE_6_9B:          return "6.9B";
        case LLM_TYPE_7B:            return "7B";
        case LLM_TYPE_8B:            return "8B";
        case LLM_TYPE_9B:            return "9B";
        case LLM_TYPE_11B:           return "11B";
        case LLM_TYPE_12B:           return "12B";
        case LLM_TYPE_13B:           return "13B";
        case LLM_TYPE_14B:           return "14B";
        case LLM_TYPE_15B:           return "15B";
        case LLM_TYPE_16B:           return "16B";
        case LLM_TYPE_20B:           return "20B";
        case LLM_TYPE_27B:           return "27B";
        case LLM_TYPE_30B:           return "30B";
        case LLM_TYPE_32B:           return "32B";
        case LLM_TYPE_34B:           return "34B";
        case LLM_TYPE_35B:           return "35B";
        case LLM_TYPE_40B:           return "40B";
        case LLM_TYPE_65B:           return "65B";
        case LLM_TYPE_70B:           return "70B";
        case LLM_TYPE_142B:          return "142B";
        case LLM_TYPE_236B:          return "236B";
        case LLM_TYPE_290B:          return "290B";
        case LLM_TYPE_314B:          return "314B";
        case LLM_TYPE_405B:          return "405B";
        case LLM_TYPE_671B:          return "671B";
        case LLM_TYPE_SMALL:         return "0.1B";
        case LLM_TYPE_MEDIUM:        return "0.4B";
        case LLM_TYPE_LARGE:         return "0.8B";
        case LLM_TYPE_XL:            return "1.5B";
        case LLM_TYPE_A1_7B:         return "A1.7B";
        case LLM_TYPE_A2_7B:         return "A2.7B";
        case LLM_TYPE_8x7B:          return "8x7B";
        case LLM_TYPE_8x22B:         return "8x22B";
        case LLM_TYPE_16x12B:        return "16x12B";
        case LLM_TYPE_16x3_8B:       return "16x3.8B";
        case LLM_TYPE_10B_128x3_66B: return "10B+128x3.66B";
        case LLM_TYPE_57B_A14B:      return "57B.A14B";
        case LLM_TYPE_17B_16E:       return "17Bx16E (Scout)";
        case LLM_TYPE_17B_128E:      return "17Bx128E (Maverick)";
        case LLM_TYPE_A13B:          return "A13B";
        case LLM_TYPE_21B_A3B:       return "21B.A3B";
        case LLM_TYPE_30B_A3B:       return "30B.A3B";
        case LLM_TYPE_106B_A12B:     return "106B.A12B";
        case LLM_TYPE_235B_A22B:     return "235B.A22B";
        case LLM_TYPE_300B_A47B:     return "300B.A47B";
        case LLM_TYPE_355B_A32B:     return "355B.A32B";
        case LLM_TYPE_E2B:           return "E2B";
        case LLM_TYPE_E4B:           return "E4B";
        default:                     return "?B";
    }
}

static const char * llama_expert_gating_func_name(llama_expert_gating_func_type type) {
    switch (type) {
        case LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX: return "softmax";
        case LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID: return "sigmoid";
        default:                                    return "unknown";
    }
}

static const std::map<llama_rope_scaling_type, const char *> LLAMA_ROPE_SCALING_TYPES = {
    { LLAMA_ROPE_SCALING_TYPE_NONE,       "none"       },
    { LLAMA_ROPE_SCALING_TYPE_LINEAR,     "linear"     },
    { LLAMA_ROPE_SCALING_TYPE_YARN,       "yarn"       },
    { LLAMA_ROPE_SCALING_TYPE_LONGROPE,   "longrope"   },
};

std::string llama_rope_scaling_type_name(llama_rope_scaling_type rope_scaling_type) {
    return LLAMA_ROPE_SCALING_TYPES.at(rope_scaling_type);
}

static llama_rope_scaling_type llama_rope_scaling_type_from_string(const std::string & name) {
    for (const auto & kv : LLAMA_ROPE_SCALING_TYPES) {
        if (kv.second == name) {
            return (llama_rope_scaling_type) kv.first;
        }
    }

    return LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED;
}

// checks if the weight tensor can be used with the specified buffer type and device
static bool weight_buft_supported(const llama_hparams & hparams, ggml_tensor * w, ggml_op op, ggml_backend_buffer_type_t buft, ggml_backend_dev_t dev) {
    GGML_ASSERT(w != nullptr);

    if (op == GGML_OP_NONE) {
        return true;
    }

    ggml_init_params params = {
        /*.mem_size   =*/ ggml_tensor_overhead()*8,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    if (!ctx_ptr) {
        throw std::runtime_error(format("failed to create ggml context"));
    }
    ggml_context * ctx = ctx_ptr.get();

    ggml_tensor * op_tensor = nullptr;

    switch (op) {
        case GGML_OP_GET_ROWS:
            {
                ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 512);
                op_tensor = ggml_get_rows(ctx, w, b);
            } break;
        case GGML_OP_MUL_MAT:
            {
                ggml_tensor * b = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, w->ne[0], 512, w->ne[2], w->ne[3]);
                op_tensor = ggml_mul_mat(ctx, w, b);
            } break;
        case GGML_OP_MUL_MAT_ID:
            {
                int n_expert_used = hparams.n_expert_used;
                ggml_tensor * b = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, w->ne[0], n_expert_used, 512);
                ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_expert_used, 512);
                op_tensor = ggml_mul_mat_id(ctx, w, b, ids);
            } break;
        case GGML_OP_ADD:
            {
                ggml_tensor * a = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, w->ne[0], w->ne[1], w->ne[2], w->ne[3]);
                op_tensor = ggml_add(ctx, a, w);
            } break;
        case GGML_OP_ADD_ID:
            {
                int n_expert_used = hparams.n_expert_used;
                ggml_tensor * a = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, w->ne[0], n_expert_used, 512);
                ggml_tensor * c = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_expert_used, 512);
                op_tensor = ggml_add_id(ctx, a, w, c);
            } break;
        case GGML_OP_MUL:
            {
                ggml_tensor * a = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, w->ne[0], w->ne[1], w->ne[2], w->ne[3]);
                op_tensor = ggml_mul(ctx, a, w);
            } break;
        case GGML_OP_DIV:
            {
                ggml_tensor * a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, w->ne[0]);
                op_tensor = ggml_div(ctx, a, w);
            } break;
        case GGML_OP_ROPE:
            {
                int n_embd_head = hparams.n_embd_head_v;
                int n_head = hparams.n_head();
                ggml_tensor * a = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd_head, n_head, 512);
                ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 512);
                op_tensor = ggml_rope_ext(
                    ctx, a, b, w,
                    0, 0, 0, 0, 0,
                    0, 0, 0, 0
                );

            } break;
        case GGML_OP_SSM_CONV:
            {
                const int64_t n_seq_tokens = 512;
                const int64_t n_seqs       = 3;
                ggml_tensor * conv_x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, w->ne[0] - 1 + n_seq_tokens, w->ne[1], n_seqs);
                op_tensor = ggml_ssm_conv(ctx, conv_x, w);
            } break;
        case GGML_OP_SSM_SCAN:
            {
                // w is ssm_a, which is used to distinguish Mamba-1 and Mamba-2
                const int64_t d_state      = w->ne[0] == 1 ? hparams.ssm_d_state : w->ne[0];
                const int64_t n_head       = w->ne[1];
                const int64_t head_dim     = hparams.ssm_d_inner / n_head;
                const int64_t n_group      = hparams.ssm_n_group ? hparams.ssm_n_group : 1;
                const int64_t n_seq_tokens = 512;
                const int64_t n_seqs       = 3;
                ggml_tensor * s   = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d_state, head_dim, n_head, n_seqs);
                ggml_tensor * x   = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, head_dim, n_head, n_seq_tokens, n_seqs);
                ggml_tensor * dt  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_head, n_seq_tokens, n_seqs);
                ggml_tensor * B   = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d_state, n_group, n_seq_tokens, n_seqs);
                ggml_tensor * C   = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d_state, n_group, n_seq_tokens, n_seqs);
                ggml_tensor * ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_seqs);
                op_tensor = ggml_ssm_scan(ctx, s, x, dt, w, B, C, ids);
            } break;
        case GGML_OP_RWKV_WKV6:
            {
                // FIXME
                const int64_t S = 123;
                const int64_t H = 123;
                const int64_t n_tokens = 123;
                const int64_t n_seqs = 123;
                ggml_tensor  * k = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, S, H, n_tokens);
                ggml_tensor  * v = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, S, H, n_tokens);
                ggml_tensor  * r = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, S, H, n_tokens);
                ggml_tensor  * tf = w;
                ggml_tensor  * td = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, S, H, n_tokens);
                ggml_tensor  * state = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, n_seqs, S, H);
                op_tensor = ggml_rwkv_wkv6(ctx, k, v, r, tf, td, state);
            } break;
        case GGML_OP_IM2COL:
            {
                const int n_embd = hparams.n_embd;
                ggml_tensor * b = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, n_embd, w->ne[1], 1, 1);
                op_tensor = ggml_im2col(ctx, w, b, 1, 0, 0, 0, 1, 0, false, GGML_TYPE_F16);
            } break;
        case GGML_OP_SCALE:
            {
                op_tensor = ggml_scale(ctx, w, 1.0f);
            } break;
        default:
            GGML_ABORT("%s: missing test for op %s for tensor %s", __func__, ggml_op_name(op), w->name);
    }

    // create a temporary dummy buffer for the weight so that supports_op can check the buffer type
    GGML_ASSERT(w->buffer == nullptr);
    w->buffer = ggml_backend_buft_alloc_buffer(buft, 0);
    bool op_supported = ggml_backend_dev_supports_op(dev, op_tensor);
    ggml_backend_buffer_free(w->buffer);
    w->buffer = nullptr;

    return op_supported;
}

// lists of buffer types used for each layer
using buft_list_t = std::vector<std::pair<ggml_backend_dev_t, ggml_backend_buffer_type_t>>;

// find the first buffer type in the list that can use the tensor
static ggml_backend_buffer_type_t select_weight_buft(const llama_hparams & hparams, ggml_tensor * tensor, ggml_op op, const buft_list_t & buft_list) {
    GGML_ASSERT(!buft_list.empty());
    for (const auto & cur : buft_list) {
        ggml_backend_dev_t cur_dev = cur.first;
        ggml_backend_buffer_type_t cur_buft = cur.second;
        if (weight_buft_supported(hparams, tensor, op, cur_buft, cur_dev)) {
            return cur_buft;
        }
    }

    return nullptr;
}

// CPU: ACCEL -> GPU host -> CPU extra -> CPU
static buft_list_t make_cpu_buft_list(const std::vector<ggml_backend_dev_t> & devices, bool use_extra_bufts) {
    buft_list_t buft_list;

    // add ACCEL buffer types
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_ACCEL) {
            auto * buft = ggml_backend_dev_buffer_type(dev);
            // skip
            if (buft != ggml_backend_cpu_buffer_type()) {
                buft_list.emplace_back(dev, buft);
            }
        }
    }

    // add a host buffer type
    // storing the tensors in a host buffer is useful when the processing of large batches
    // is offloaded to a GPU device, since it reduces the time spent on data transfers
    // generally, this will be done using the first device in the list
    // a better approach would be to handle this on a weight-by-weight basis using the offload_op
    // function of the device to determine if it would benefit from being stored in a host buffer
    for (auto * dev : devices) {
        ggml_backend_buffer_type_t buft = ggml_backend_dev_host_buffer_type(dev);
        if (buft) {
            buft_list.emplace_back(dev, buft);
            break;
        }
    }

    // add extra buffer types
    if (use_extra_bufts) {
        auto * cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        if (cpu_dev == nullptr) {
            throw std::runtime_error(format("%s: no CPU backend found", __func__));
        }

        auto * cpu_reg = ggml_backend_dev_backend_reg(cpu_dev);
        auto ggml_backend_dev_get_extra_bufts_fn = (ggml_backend_dev_get_extra_bufts_t)
            ggml_backend_reg_get_proc_address(cpu_reg, "ggml_backend_dev_get_extra_bufts");
        if (ggml_backend_dev_get_extra_bufts_fn) {
            ggml_backend_buffer_type_t * extra_bufts = ggml_backend_dev_get_extra_bufts_fn(cpu_dev);
            while (extra_bufts && *extra_bufts) {
                buft_list.emplace_back(cpu_dev, *extra_bufts);
                ++extra_bufts;
            }
        }
    }

    // add the CPU buffer type
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
            buft_list.emplace_back(dev, ggml_backend_dev_buffer_type(dev));
        }
    }

    return buft_list;
}

// GPU: split if LLAMA_SPLIT_MODE_ROW -> GPU
static buft_list_t make_gpu_buft_list(ggml_backend_dev_t dev, llama_split_mode split_mode, const float * tensor_split) {
    buft_list_t buft_list;

    // add the device split buffer type if requested and available
    if (split_mode == LLAMA_SPLIT_MODE_ROW) {
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
        auto ggml_backend_split_buffer_type_fn = (ggml_backend_split_buffer_type_t)
            ggml_backend_reg_get_proc_address(reg, "ggml_backend_split_buffer_type");
        if (ggml_backend_split_buffer_type_fn) {
            size_t dev_index = [&]() {
                auto * reg = ggml_backend_dev_backend_reg(dev);
                for (size_t i = 0; i < ggml_backend_reg_dev_count(reg); ++i) {
                    if (ggml_backend_reg_dev_get(reg, i) == dev) {
                        return i;
                    }
                }
                throw std::runtime_error(format("device %s not found in its backend reg", ggml_backend_dev_name(dev)));
            }();
            auto * buft = ggml_backend_split_buffer_type_fn(dev_index, tensor_split);
            if (buft != nullptr) {
                buft_list.emplace_back(dev, buft);
            }
        }
    }

    // add the device default buffer type
    buft_list.emplace_back(dev, ggml_backend_dev_buffer_type(dev));

    return buft_list;
}

struct llama_model::impl {
    impl() {}
    ~impl() {}

    uint64_t n_elements = 0;

    size_t n_bytes = 0;

    std::string desc_str;

    // model memory mapped files
    llama_mmaps mappings;

    // objects representing data potentially being locked in memory
    llama_mlocks mlock_bufs;
    llama_mlocks mlock_mmaps;

    // contexts where the model tensors metadata is stored
    std::vector<ggml_context_ptr> ctxs;

    // the model memory buffers for the tensor data
    std::vector<ggml_backend_buffer_ptr> bufs;

    buft_list_t cpu_buft_list;
    std::map<ggml_backend_dev_t, buft_list_t> gpu_buft_list;

    struct layer_dev {
        ggml_backend_dev_t dev;
        buft_list_t * buft_list;
    };

    layer_dev dev_input = {};
    layer_dev dev_output = {};
    std::vector<layer_dev> dev_layer;

    bool has_tensor_overrides;
};

llama_model::llama_model(const llama_model_params & params) : params(params), pimpl(std::make_unique<impl>()) {
    pimpl->has_tensor_overrides = params.tensor_buft_overrides && params.tensor_buft_overrides[0].pattern;
}

llama_model::~llama_model() {}

void llama_model::load_stats(llama_model_loader & ml) {
    pimpl->n_elements = ml.n_elements;
    pimpl->n_bytes = ml.n_bytes;
}

void llama_model::load_arch(llama_model_loader & ml) {
    arch = ml.get_arch();
    if (arch == LLM_ARCH_UNKNOWN) {
        throw std::runtime_error("unknown model architecture: '" + ml.get_arch_name() + "'");
    }
}

void llama_model::load_hparams(llama_model_loader & ml) {
    const gguf_context * ctx = ml.meta.get();

    // get metadata as string
    for (int i = 0; i < gguf_get_n_kv(ctx); i++) {
        gguf_type type = gguf_get_kv_type(ctx, i);
        if (type == GGUF_TYPE_ARRAY) {
            continue;
        }
        const char * name = gguf_get_key(ctx, i);
        const std::string value = gguf_kv_to_str(ctx, i);
        gguf_kv.emplace(name, value);
    }

    // get general kv
    ml.get_key(LLM_KV_GENERAL_NAME, name, false);

    // everything past this point is not vocab-related
    if (hparams.vocab_only) {
        return;
    }

    ml.get_key(LLM_KV_CONTEXT_LENGTH,    hparams.n_ctx_train);
    ml.get_key(LLM_KV_EMBEDDING_LENGTH,  hparams.n_embd);
    ml.get_key(LLM_KV_BLOCK_COUNT,       hparams.n_layer);
    ml.get_key(LLM_KV_EXPERT_COUNT,      hparams.n_expert,      false);
    ml.get_key(LLM_KV_EXPERT_USED_COUNT, hparams.n_expert_used, false);

    if (arch == LLM_ARCH_WAVTOKENIZER_DEC) {
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

    std::fill(hparams.n_head_arr.begin(),    hparams.n_head_arr.end(),    0);
    std::fill(hparams.n_head_kv_arr.begin(), hparams.n_head_kv_arr.end(), 0);
    std::fill(hparams.n_ff_arr.begin(),      hparams.n_ff_arr.end(),      0);
    std::fill(
        hparams.recurrent_layer_arr.begin(),
        hparams.recurrent_layer_arr.end(),
        llm_arch_is_recurrent(ml.get_arch()));

    std::fill(hparams.rope_sections.begin(), hparams.rope_sections.end(), 0);

    std::fill(hparams.swa_layers.begin(), hparams.swa_layers.end(), 0);

    ml.get_key_or_arr(LLM_KV_FEED_FORWARD_LENGTH,  hparams.n_ff_arr,   hparams.n_layer, false);
    ml.get_key_or_arr(LLM_KV_ATTENTION_HEAD_COUNT, hparams.n_head_arr, hparams.n_layer, false);

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

    // by default assume that the sliding-window layers use the same scaling type as the non-sliding-window layers
    hparams.rope_freq_base_train_swa  = hparams.rope_freq_base_train;
    hparams.rope_freq_scale_train_swa = hparams.rope_freq_scale_train;

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

        if (arch == LLM_ARCH_LLAMA || arch == LLM_ARCH_DECI || arch == LLM_ARCH_FALCON) {
            if (hparams.n_rot != hparams.n_embd_head_k) {
                throw std::runtime_error(format("invalid n_rot: %u, expected %u", hparams.n_rot, hparams.n_embd_head_k));
            }
        }
    } else {
        hparams.n_rot = 0;
        hparams.n_embd_head_k = 0;
        hparams.n_embd_head_v = 0;
    }

    // for differentiating model types
    uint32_t n_vocab = 0;
    ml.get_key(LLM_KV_VOCAB_SIZE, n_vocab, false) || ml.get_arr_n(LLM_KV_TOKENIZER_LIST, n_vocab, false);

    // for classifier models
    ml.get_arr(LLM_KV_CLASSIFIER_OUTPUT_LABELS, classifier_labels, false);
    if (!classifier_labels.empty()) {
        hparams.n_cls_out = classifier_labels.size();
    }

    // arch-specific KVs
    switch (arch) {
        case LLM_ARCH_LLAMA:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

                if (hparams.n_expert == 8) {
                    switch (hparams.n_layer) {
                        case 32: type = LLM_TYPE_8x7B; break;
                        case 56: type = LLM_TYPE_8x22B; break;
                        default: type = LLM_TYPE_UNKNOWN;
                    }
                } else {
                    switch (hparams.n_layer) {
                        case 16: type = LLM_TYPE_1B; break; // Llama 3.2 1B
                        case 22: type = LLM_TYPE_1B; break;
                        case 26: type = LLM_TYPE_3B; break;
                        case 28: type = LLM_TYPE_3B; break; // Llama 3.2 3B
                        case 30: type = LLM_TYPE_256M; break; // smoldocling 256M
                        // granite uses a vocab with len 49152
                        case 32: type = n_vocab == 49152 ? LLM_TYPE_3B : (n_vocab < 40000 ? LLM_TYPE_7B : LLM_TYPE_8B); break;
                        case 36: type = LLM_TYPE_8B; break; // granite
                        case 40: type = LLM_TYPE_13B; break;
                        case 48: type = LLM_TYPE_34B; break;
                        case 60: type = LLM_TYPE_30B; break;
                        case 80: type = hparams.n_head() == hparams.n_head_kv() ? LLM_TYPE_65B : LLM_TYPE_70B; break;
                        default: type = LLM_TYPE_UNKNOWN;
                    }
                }
            } break;
        case LLM_ARCH_LLAMA4:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,  hparams.n_ff_exp);
                ml.get_key(LLM_KV_INTERLEAVE_MOE_LAYER_STEP,   hparams.n_moe_layer_step);

                hparams.swa_type      = LLAMA_SWA_TYPE_CHUNKED;
                hparams.n_swa         = 8192; // should this be a gguf kv? currently it's the same for Scout and Maverick
                hparams.set_swa_pattern(4);   // pattern: 3 chunked - 1 full

                switch (hparams.n_expert) {
                    case 16:  type = LLM_TYPE_17B_16E; break;
                    case 128: type = LLM_TYPE_17B_128E; break;
                    default:  type = LLM_TYPE_UNKNOWN;
                }

                if (type == LLM_TYPE_17B_128E) {
                    hparams.use_kq_norm = false;
                }
            } break;
        case LLM_ARCH_ARCEE:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

                // Arcee uses the same structure as Llama
                switch (hparams.n_layer) {
                    case 36: type = LLM_TYPE_4B; break;
                    default: type = LLM_TYPE_UNKNOWN;
                }
            } break;
        case LLM_ARCH_DECI:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                switch (hparams.n_layer) {
                    case 32: type = LLM_TYPE_7B; break;
                    case 80: type = LLM_TYPE_70B; break;
                    case 162: type = LLM_TYPE_405B; break;
                    default: type = LLM_TYPE_UNKNOWN;
                }
            } break;
        case LLM_ARCH_MINICPM:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                ml.get_key(LLM_KV_EMBEDDING_SCALE,             hparams.f_embedding_scale);
                ml.get_key(LLM_KV_RESIDUAL_SCALE,              hparams.f_residual_scale);
                ml.get_key(LLM_KV_LOGIT_SCALE,                 hparams.f_logit_scale);

                // MiniCPM uses rope by default, unlike Granite which uses it as a switch
                hparams.rope_finetuned = true;

                switch (hparams.n_layer) {
                    case 52: type = LLM_TYPE_1B; break;
                    case 40: type = LLM_TYPE_2B; break;
                    default: type = LLM_TYPE_UNKNOWN;
                }
            } break;
        case LLM_ARCH_MINICPM3:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                ml.get_key(LLM_KV_ATTENTION_Q_LORA_RANK,       hparams.n_lora_q);
                ml.get_key(LLM_KV_ATTENTION_KV_LORA_RANK,      hparams.n_lora_kv);

                switch (hparams.n_layer) {
                    case 62: type = LLM_TYPE_4B; break;
                    default: type = LLM_TYPE_UNKNOWN;
                }
            } break;
        case LLM_ARCH_GROK:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

                switch (hparams.n_layer) {
                    case 64: type = LLM_TYPE_314B; break;
                    default: type = LLM_TYPE_UNKNOWN;
                }
            } break;
        case LLM_ARCH_FALCON:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps);

                switch (hparams.n_layer) {
                    case 32: type = LLM_TYPE_7B; break;
                    case 60: type = LLM_TYPE_40B; break;
                    default: type = LLM_TYPE_UNKNOWN;
                }
            } break;
        case LLM_ARCH_BAICHUAN:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                switch (hparams.n_layer) {
                    case 32: type = LLM_TYPE_7B; break;
                    case 40: type = LLM_TYPE_13B; break;
                    default: type = LLM_TYPE_UNKNOWN;
                }

                if (type == LLM_TYPE_13B) {
                    // TODO: become GGUF KV parameter
                    hparams.f_max_alibi_bias = 8.0f;
                }
            } break;
        case LLM_ARCH_STARCODER:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps);
                switch (hparams.n_layer) {
                    case 24: type = LLM_TYPE_1B; break;
                    case 36: type = LLM_TYPE_3B; break;
                    case 42: type = LLM_TYPE_7B; break;
                    case 40: type = LLM_TYPE_15B; break;
                    default: type = LLM_TYPE_UNKNOWN;
                }
            } break;
        case LLM_ARCH_REFACT:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                switch (hparams.n_layer) {
                    case 32: type = LLM_TYPE_1B; break;
                    default: type = LLM_TYPE_UNKNOWN;
                }

                // TODO: become GGUF KV parameter
                hparams.f_max_alibi_bias = 8.0f;
            } break;
        case LLM_ARCH_BERT:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS,    hparams.f_norm_eps);
                ml.get_key(LLM_KV_ATTENTION_CAUSAL,           hparams.causal_attn);
                ml.get_key(LLM_KV_POOLING_TYPE,               hparams.pooling_type, false);

                switch (hparams.n_layer) {
                    case 3:
                        type = LLM_TYPE_17M; break; // bge-micro
                    case 6:
                        type = LLM_TYPE_22M; break; // MiniLM-L6
                    case 12:
                        switch (hparams.n_embd) {
                            case 384: type = LLM_TYPE_33M; break; // MiniLM-L12, bge-small
                            case 768: type = LLM_TYPE_109M; break; // bge-base
                            default: type = LLM_TYPE_UNKNOWN;
                        } break;
                    case 24:
                        type = LLM_TYPE_335M; break; // bge-large
                    default: type = LLM_TYPE_UNKNOWN;
                }
            } break;
        case LLM_ARCH_JINA_BERT_V2:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS,    hparams.f_norm_eps);
                ml.get_key(LLM_KV_ATTENTION_CAUSAL,           hparams.causal_attn);
                ml.get_key(LLM_KV_POOLING_TYPE,               hparams.pooling_type, false);
                hparams.f_max_alibi_bias = 8.0f;

                switch (hparams.n_layer) {
                    case 4:  type = LLM_TYPE_33M;  break; // jina-embeddings-small
                    case 12: type = LLM_TYPE_137M; break; // jina-embeddings-base
                    default: type = LLM_TYPE_UNKNOWN;
                }
            } break;
        case LLM_ARCH_NOMIC_BERT:
        case LLM_ARCH_NOMIC_BERT_MOE:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS,    hparams.f_norm_eps);
                ml.get_key(LLM_KV_ATTENTION_CAUSAL,           hparams.causal_attn);
                ml.get_key(LLM_KV_POOLING_TYPE,               hparams.pooling_type);
                ml.get_key(LLM_KV_MOE_EVERY_N_LAYERS,         hparams.moe_every_n_layers, 0);

                if (hparams.n_layer == 12 && hparams.n_embd == 768) {
                    if (arch == LLM_ARCH_NOMIC_BERT) {
                        type = LLM_TYPE_137M;
                    } else if (arch == LLM_ARCH_NOMIC_BERT_MOE && hparams.moe_every_n_layers == 2) {
                        type = LLM_TYPE_475M;
                    }
                }
            } break;
        case LLM_ARCH_NEO_BERT:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                ml.get_key(LLM_KV_ATTENTION_CAUSAL,            hparams.causal_attn);
                ml.get_key(LLM_KV_POOLING_TYPE,                hparams.pooling_type);

                if (hparams.n_layer == 28) {
                    type = LLM_TYPE_250M;
                }
            } break;
        case LLM_ARCH_BLOOM:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps);

                switch (hparams.n_layer) {
                    case 24: type = LLM_TYPE_1B; break;
                    case 30:
                        switch (hparams.n_embd) {
                            case 2560: type = LLM_TYPE_3B; break;
                            case 4096: type = LLM_TYPE_7B; break;
                            default: type = LLM_TYPE_UNKNOWN;
                        } break;
                    default: type = LLM_TYPE_UNKNOWN;
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
                    case 32: type = LLM_TYPE_7B; break;
                    case 48: type = LLM_TYPE_30B; break;
                    default: type = LLM_TYPE_UNKNOWN;
                }
            } break;
        case LLM_ARCH_STABLELM:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps);

                switch (hparams.n_layer) {
                    case 24: type = LLM_TYPE_1B; break;
                    case 32: type = LLM_TYPE_3B; break;
                    case 40: type = LLM_TYPE_12B; break;
                    default: type = LLM_TYPE_UNKNOWN;
               }
            } break;
        case LLM_ARCH_QWEN:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

                switch (hparams.n_layer) {
                    case 32: type = LLM_TYPE_7B; break;
                    case 40: type = LLM_TYPE_13B; break;
                    default: type = LLM_TYPE_UNKNOWN;
                }
            } break;
        case LLM_ARCH_QWEN2VL:
            {
                ml.get_key_or_arr(LLM_KV_ROPE_DIMENSION_SECTIONS, hparams.rope_sections, 4, true);
            }
            // fall through
        case LLM_ARCH_QWEN2:
            {
                ml.get_key(LLM_KV_POOLING_TYPE, hparams.pooling_type, false);
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                switch (hparams.n_layer) {
                    case 24: type = hparams.n_embd == 1024 ? LLM_TYPE_0_5B : LLM_TYPE_1B; break;
                    case 28: type = hparams.n_embd == 1536 ? LLM_TYPE_1_5B : LLM_TYPE_7B; break;
                    case 32: type = LLM_TYPE_7B; break;
                    case 36: type = LLM_TYPE_3B; break;
                    case 40: type = hparams.n_head() == 20 ? LLM_TYPE_4B : LLM_TYPE_13B; break;
                    case 48: type = LLM_TYPE_14B; break;
                    case 64: type = LLM_TYPE_32B; break;
                    case 80: type = LLM_TYPE_70B; break;
                    default: type = LLM_TYPE_UNKNOWN;
                }
            } break;
        case LLM_ARCH_DREAM:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                // Dream models are primarily 7B with 28 layers
                switch (hparams.n_layer) {
                    case 28:
                        type = LLM_TYPE_7B;
                        break;
                    default:
                        type = LLM_TYPE_UNKNOWN;
                }
                // Set non-causal attention for diffusion models
                hparams.causal_attn = false;
            }
            break;
        case LLM_ARCH_LLADA:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                // LLaDA-8B has 32 layers, similar to LLaMA but for diffusion
                switch (hparams.n_layer) {
                    case 32:
                        type = LLM_TYPE_8B;
                        break;
                    default:
                        type = LLM_TYPE_UNKNOWN;
                }
                // Set non-causal attention for diffusion models
                hparams.causal_attn = false;
            }
            break;
        case LLM_ARCH_QWEN2MOE:
            {
                ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,        hparams.n_ff_exp, false);
                ml.get_key(LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH, hparams.n_ff_shexp, false);

                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                switch (hparams.n_layer) {
                    case 24: type = LLM_TYPE_A2_7B; break;
                    case 28: type = LLM_TYPE_57B_A14B; break;
                    default: type = LLM_TYPE_UNKNOWN;
                }
            } break;
        case LLM_ARCH_QWEN3:
            {
                ml.get_key(LLM_KV_POOLING_TYPE, hparams.pooling_type, false);
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                switch (hparams.n_layer) {
                    case 28: type = hparams.n_embd == 1024 ? LLM_TYPE_0_6B : LLM_TYPE_1_7B; break;
                    case 36: type = hparams.n_embd == 2560 ? LLM_TYPE_4B : LLM_TYPE_8B; break;
                    case 40: type = LLM_TYPE_14B; break;
                    case 64: type = LLM_TYPE_32B; break;
                    default: type = LLM_TYPE_UNKNOWN;
                }
            } break;
        case LLM_ARCH_QWEN3MOE:
            {
                ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,        hparams.n_ff_exp, false);

                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                switch (hparams.n_layer) {
                    case 48: type = LLM_TYPE_30B_A3B; break;
                    case 94: type = LLM_TYPE_235B_A22B; break;
                    default: type = LLM_TYPE_UNKNOWN;
                }
            } break;
        case LLM_ARCH_PHI2:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps);

                switch (hparams.n_layer) {
                    case 24: type = LLM_TYPE_1B; break;
                    case 32: type = LLM_TYPE_3B; break;
                    default: type = LLM_TYPE_UNKNOWN;
                }
            } break;
        case LLM_ARCH_PHI3:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

                switch (hparams.n_layer) {
                    case 24: type = LLM_TYPE_1B; break;
                    case 32: type = LLM_TYPE_3B; break;
                    case 40: type = LLM_TYPE_14B; break;
                    default: type = LLM_TYPE_UNKNOWN;
                }

                const bool found_swa = ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW, hparams.n_swa, false);

                if (found_swa && hparams.n_swa > 0) {
                    LLAMA_LOG_WARN("%s: Phi SWA is currently disabled - results might be suboptimal for some models (see %s)\n",
                            __func__, "https://github.com/ggml-org/llama.cpp/pull/13676");

                    // TODO: fix conversion scripts to correctly populate `n_swa` and `n_swa_pattern`
                    hparams.swa_type = LLAMA_SWA_TYPE_NONE;

                    hparams.n_swa         = 0;
                    hparams.set_swa_pattern(1);
                }
            } break;
        case LLM_ARCH_PHIMOE:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

                switch (hparams.n_layer) {
                    case 32: type = LLM_TYPE_16x3_8B; break;
                    default: type = LLM_TYPE_UNKNOWN;
                }
            } break;
        case LLM_ARCH_PLAMO:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

                switch (hparams.n_layer) {
                    case 40: type = LLM_TYPE_13B; break;
                    default: type = LLM_TYPE_UNKNOWN;
               }
            } break;
        case LLM_ARCH_PLAMO2:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

                // Load Mamba SSM parameters
                ml.get_key(LLM_KV_SSM_CONV_KERNEL,    hparams.ssm_d_conv);
                ml.get_key(LLM_KV_SSM_INNER_SIZE,     hparams.ssm_d_inner);
                ml.get_key(LLM_KV_SSM_STATE_SIZE,     hparams.ssm_d_state);
                ml.get_key(LLM_KV_SSM_TIME_STEP_RANK, hparams.ssm_dt_rank);
                ml.get_key(LLM_KV_SSM_GROUP_COUNT,    hparams.ssm_n_group);

                for (uint32_t i = 0; i < hparams.n_layer; ++i) {
                    hparams.recurrent_layer_arr[i] = hparams.n_head_kv(i) == 0;
                }

                switch (hparams.n_layer) {
                    case 16: type = LLM_TYPE_1B; break;
                    case 32:
                        if (hparams.n_embd == 2048) {
                            type = LLM_TYPE_2B;
                        } else if (hparams.n_embd == 4096) {
                            type = LLM_TYPE_8B;
                        }
                        break;
                    default: type = LLM_TYPE_UNKNOWN;
               }
            } break;
        case LLM_ARCH_GPT2:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps);
                switch (hparams.n_layer) {
                    case 12: type = LLM_TYPE_SMALL; break;
                    case 24: type = LLM_TYPE_MEDIUM; break;
                    case 36: type = LLM_TYPE_LARGE; break;
                    case 48: type = LLM_TYPE_XL; break;
                    default: type = LLM_TYPE_UNKNOWN;
                }
            } break;
        case LLM_ARCH_CODESHELL:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps);
                switch (hparams.n_layer) {
                    case 42: type = LLM_TYPE_7B; break;
                    default: type = LLM_TYPE_UNKNOWN;
                }
            } break;
        case LLM_ARCH_ORION:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps);

                switch (hparams.n_layer) {
                    case 40: type = LLM_TYPE_14B; break;
                    default: type = LLM_TYPE_UNKNOWN;
                }
            } break;
        case LLM_ARCH_INTERNLM2:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                switch (hparams.n_layer) {
                    case 32: type = LLM_TYPE_7B; break;
                    case 48: type = LLM_TYPE_20B; break;
                    default: type = LLM_TYPE_UNKNOWN;
                }
            } break;
        case LLM_ARCH_GEMMA:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

                switch (hparams.n_layer) {
                    case 18: type = LLM_TYPE_2B; break;
                    case 28: type = LLM_TYPE_7B; break;
                    default: type = LLM_TYPE_UNKNOWN;
               }
            } break;
        case LLM_ARCH_GEMMA2:
            {
                hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;
                hparams.n_swa = 4096; // default value of gemma 2
                hparams.set_swa_pattern(2);
                hparams.attn_soft_cap = true;

                ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW,    hparams.n_swa, false);
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                ml.get_key(LLM_KV_ATTN_LOGIT_SOFTCAPPING,      hparams.f_attn_logit_softcapping, false);
                ml.get_key(LLM_KV_FINAL_LOGIT_SOFTCAPPING,     hparams.f_final_logit_softcapping, false);

                switch (hparams.n_layer) {
                    case 26: type = LLM_TYPE_2B; break;
                    case 42: type = LLM_TYPE_9B; break;
                    case 46: type = LLM_TYPE_27B; break;
                    default: type = LLM_TYPE_UNKNOWN;
               }

                // ref: https://github.com/google/gemma_pytorch/blob/014acb7ac4563a5f77c76d7ff98f31b568c16508/gemma/config.py#L173
                hparams.f_attention_scale = type == LLM_TYPE_27B
                    ? 1.0f / std::sqrt(float(hparams.n_embd / hparams.n_head(0)))
                    : 1.0f / std::sqrt(float(hparams.n_embd_head_k));
            } break;
        case LLM_ARCH_GEMMA3:
            {
                hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;
                hparams.set_swa_pattern(6);

                hparams.rope_freq_base_train_swa  = 10000.0f;
                hparams.rope_freq_scale_train_swa = 1.0f;

                ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW,    hparams.n_swa);
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

                switch (hparams.n_layer) {
                    case 26: type = LLM_TYPE_1B; break;
                    case 34: type = LLM_TYPE_4B; break;
                    case 48: type = LLM_TYPE_12B; break;
                    case 62: type = LLM_TYPE_27B; break;
                    default: type = LLM_TYPE_UNKNOWN;
                }

                // ref: https://github.com/google/gemma_pytorch/blob/014acb7ac4563a5f77c76d7ff98f31b568c16508/gemma/config.py#L289
                hparams.f_attention_scale = type == LLM_TYPE_27B
                    ? 1.0f / std::sqrt(float(hparams.n_embd / hparams.n_head(0)))
                    : 1.0f / std::sqrt(float(hparams.n_embd_head_k));
            } break;
        case LLM_ARCH_GEMMA3N:
            {
                hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;
                hparams.set_swa_pattern(5);

                hparams.rope_freq_base_train_swa  = 10000.0f;
                hparams.rope_freq_scale_train_swa = 1.0f;
                hparams.f_attention_scale         = 1.0f;

                ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW,    hparams.n_swa);
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

                switch (hparams.n_layer) {
                    case 30: type = LLM_TYPE_E2B; break;
                    case 35: type = LLM_TYPE_E4B; break;
                    default: type = LLM_TYPE_UNKNOWN;
                }
            } break;
        case LLM_ARCH_STARCODER2:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps);
                switch (hparams.n_layer) {
                    case 30: type = LLM_TYPE_3B; break;
                    case 32: type = LLM_TYPE_7B; break;
                    case 40: type = LLM_TYPE_15B; break;
                    case 52: type = LLM_TYPE_20B; break; // granite
                    case 88: type = LLM_TYPE_34B; break; // granite
                    default: type = LLM_TYPE_UNKNOWN;
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
                            case 768: type = LLM_TYPE_SMALL; break;
                            default: type = LLM_TYPE_UNKNOWN;
                        } break;
                    case 48:
                        switch (hparams.n_embd) {
                            case 1024: type = LLM_TYPE_MEDIUM; break;
                            case 1536: type = LLM_TYPE_LARGE; break;
                            case 2048: type = LLM_TYPE_XL; break;
                            default:   type = LLM_TYPE_UNKNOWN;
                        } break;
                    case 64:
                        switch (hparams.n_embd) {
                            case 2560: type = LLM_TYPE_3B; break;
                            default: type = LLM_TYPE_UNKNOWN;
                        } break;
                    default: type = LLM_TYPE_UNKNOWN;
                }
            } break;
        case LLM_ARCH_MAMBA2:
            {
                ml.get_key(LLM_KV_SSM_CONV_KERNEL,    hparams.ssm_d_conv);
                ml.get_key(LLM_KV_SSM_INNER_SIZE,     hparams.ssm_d_inner);
                ml.get_key(LLM_KV_SSM_STATE_SIZE,     hparams.ssm_d_state);
                ml.get_key(LLM_KV_SSM_TIME_STEP_RANK, hparams.ssm_dt_rank);
                ml.get_key(LLM_KV_SSM_GROUP_COUNT,    hparams.ssm_n_group);

                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

                switch (hparams.n_layer) {
                    case 24:
                        switch (hparams.n_embd) {
                            case 768: type = LLM_TYPE_SMALL; break;
                            default: type = LLM_TYPE_UNKNOWN;
                        } break;
                    case 48:
                        switch (hparams.n_embd) {
                            case 1024: type = LLM_TYPE_MEDIUM; break;
                            case 1536: type = LLM_TYPE_LARGE; break;
                            case 2048: type = LLM_TYPE_XL; break;
                            default: type = LLM_TYPE_UNKNOWN;
                        } break;
                    case 64:
                        switch (hparams.n_embd) {
                            case 2560: type = LLM_TYPE_3B; break;
                            case 4096: type = LLM_TYPE_7B; break;
                            default: type = LLM_TYPE_UNKNOWN;
                        } break;
                    default: type = LLM_TYPE_UNKNOWN;
                }
            } break;
        case LLM_ARCH_JAMBA:
            {
                ml.get_key(LLM_KV_SSM_CONV_KERNEL,    hparams.ssm_d_conv);
                ml.get_key(LLM_KV_SSM_INNER_SIZE,     hparams.ssm_d_inner);
                ml.get_key(LLM_KV_SSM_STATE_SIZE,     hparams.ssm_d_state);
                ml.get_key(LLM_KV_SSM_TIME_STEP_RANK, hparams.ssm_dt_rank);

                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

                for (uint32_t i = 0; i < hparams.n_layer; ++i) {
                    hparams.recurrent_layer_arr[i] = hparams.n_head_kv(i) == 0;
                }

                switch (hparams.n_layer) {
                    // TODO: Jamba layers are a bit heterogenous, so naming this is hard.
                    case 12: // 900M  8x???M
                    case 32: // 51B  16x?B
                    default: type = LLM_TYPE_UNKNOWN;
                }
            } break;
        case LLM_ARCH_XVERSE:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                switch (hparams.n_layer) {
                    case 32: type = LLM_TYPE_7B; break;
                    case 40: type = LLM_TYPE_13B; break;
                    case 80: type = LLM_TYPE_65B; break;
                    default: type = LLM_TYPE_UNKNOWN;
                }
            } break;
        case LLM_ARCH_COMMAND_R:
            {
                ml.get_key(LLM_KV_LOGIT_SCALE,             hparams.f_logit_scale);
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps);
                switch (hparams.n_layer) {
                    case 40: type = LLM_TYPE_35B; break;
                    default: type = LLM_TYPE_UNKNOWN;
                }
            } break;
        case LLM_ARCH_COHERE2:
            {
                hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;
                hparams.set_swa_pattern(4);

                ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW, hparams.n_swa);
                ml.get_key(LLM_KV_LOGIT_SCALE,              hparams.f_logit_scale);
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS,  hparams.f_norm_eps);
                switch (hparams.n_layer) {
                    case 32: type = LLM_TYPE_8B; break;
                    default: type = LLM_TYPE_UNKNOWN;
                }
            } break;
        case LLM_ARCH_DBRX:
        {
            ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps);
            ml.get_key(LLM_KV_ATTENTION_CLAMP_KQV,     hparams.f_clamp_kqv);

            switch (hparams.n_layer) {
                case 40: type = LLM_TYPE_16x12B; break;
                default: type = LLM_TYPE_UNKNOWN;
            }
        } break;
        case LLM_ARCH_OLMO:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps);
                ml.get_key(LLM_KV_ATTENTION_CLAMP_KQV,     hparams.f_clamp_kqv, false);

                switch (hparams.n_layer) {
                    case 22: type = LLM_TYPE_1B; break;
                    case 32: type = LLM_TYPE_7B; break;
                    case 80: type = LLM_TYPE_70B; break;
                    default: type = LLM_TYPE_UNKNOWN;
                }
            } break;
        case LLM_ARCH_OLMO2:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

                switch (hparams.n_layer) {
                    case 16: type = LLM_TYPE_1B; break;
                    case 32: type = LLM_TYPE_7B; break;
                    case 40: type = LLM_TYPE_13B; break;
                    case 64: type = LLM_TYPE_32B; break;
                    default: type = LLM_TYPE_UNKNOWN;
                }
            } break;
        case LLM_ARCH_OLMOE:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                switch (hparams.n_layer) {
                    case 16: type = LLM_TYPE_A1_7B; break;
                    default: type = LLM_TYPE_UNKNOWN;
                }
            } break;
        case LLM_ARCH_OPENELM:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

                switch (hparams.n_layer) {
                case 16: type = LLM_TYPE_270M; break;
                case 20: type = LLM_TYPE_450M; break;
                case 28: type = LLM_TYPE_1B; break;
                case 36: type = LLM_TYPE_3B; break;
                default: type = LLM_TYPE_UNKNOWN;
                }
            } break;
        case LLM_ARCH_GPTNEOX:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps);
                ml.get_key(LLM_KV_USE_PARALLEL_RESIDUAL,   hparams.use_par_res);
                switch (hparams.n_layer) {
                    case 6:
                        switch (hparams.n_ff()) {
                            case 512:  type = LLM_TYPE_14M; break;
                            case 2048: type = LLM_TYPE_70M; break;
                            default:   type = LLM_TYPE_UNKNOWN;
                        } break;
                    case 12:
                        switch (hparams.n_ff()) {
                            case 3072: type = LLM_TYPE_160M; break;
                            default: type = LLM_TYPE_UNKNOWN;
                        } break;
                    case 16:
                        switch (hparams.n_ff()) {
                            case 8192: type = LLM_TYPE_1B; break;
                            default: type = LLM_TYPE_UNKNOWN;
                        } break;
                    case 24:
                        switch (hparams.n_ff()) {
                            case 4096: type = LLM_TYPE_410M; break;
                            case 8192: type = LLM_TYPE_1_4B; break;
                            default: type = LLM_TYPE_UNKNOWN;
                        } break;
                    case 32:
                        switch (hparams.n_ff()) {
                            case 10240: type = LLM_TYPE_2_8B; break;
                            case 16384: type = LLM_TYPE_6_9B; break;
                            default: type = LLM_TYPE_UNKNOWN;
                        } break;
                    case 36:
                        switch (hparams.n_ff()) {
                            case 20480: type = LLM_TYPE_12B; break;
                            default: type = LLM_TYPE_UNKNOWN;
                        } break;
                    case 44:
                        switch (hparams.n_ff()) {
                            case 24576: type = LLM_TYPE_20B; break;
                            default: type = LLM_TYPE_UNKNOWN;
                        } break;
                    default: type = LLM_TYPE_UNKNOWN;
                }
            } break;
        case LLM_ARCH_ARCTIC:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

                if (hparams.n_expert == 128) {
                    switch (hparams.n_layer) {
                        case 35: type = LLM_TYPE_10B_128x3_66B; break;
                        default: type = LLM_TYPE_UNKNOWN;
                    }
                } else {
                    type = LLM_TYPE_UNKNOWN;
                }
            } break;
        case LLM_ARCH_DEEPSEEK:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                ml.get_key(LLM_KV_LEADING_DENSE_BLOCK_COUNT,   hparams.n_layer_dense_lead);
                ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,  hparams.n_ff_exp);
                ml.get_key(LLM_KV_EXPERT_SHARED_COUNT,         hparams.n_expert_shared);
                ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE,        hparams.expert_weights_scale);

                switch (hparams.n_layer) {
                    case 28: type = LLM_TYPE_20B; break;
                    default: type = LLM_TYPE_UNKNOWN;
                }
            } break;
        case LLM_ARCH_DEEPSEEK2:
            {
                bool is_lite = (hparams.n_layer == 27);
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                ml.get_key(LLM_KV_LEADING_DENSE_BLOCK_COUNT,   hparams.n_layer_dense_lead);
                if (!is_lite) {
                    ml.get_key(LLM_KV_ATTENTION_Q_LORA_RANK, hparams.n_lora_q);
                }
                ml.get_key(LLM_KV_ATTENTION_KV_LORA_RANK,     hparams.n_lora_kv);
                ml.get_key(LLM_KV_ATTENTION_KEY_LENGTH_MLA,   hparams.n_embd_head_k_mla, false);
                ml.get_key(LLM_KV_ATTENTION_VALUE_LENGTH_MLA, hparams.n_embd_head_v_mla, false);
                ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp);
                ml.get_key(LLM_KV_EXPERT_SHARED_COUNT,        hparams.n_expert_shared);
                ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE,       hparams.expert_weights_scale);
                ml.get_key(LLM_KV_EXPERT_WEIGHTS_NORM,        hparams.expert_weights_norm, false);
                ml.get_key(LLM_KV_EXPERT_GATING_FUNC,         hparams.expert_gating_func, false);
                if (hparams.expert_gating_func == LLAMA_EXPERT_GATING_FUNC_TYPE_NONE) {
                    // for compatibility with existing DeepSeek V2 and V2.5 GGUFs
                    // that have no expert_gating_func model parameter set
                    hparams.expert_gating_func = LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX;
                }
                ml.get_key(LLM_KV_ROPE_SCALING_YARN_LOG_MUL, hparams.rope_yarn_log_mul, false);

                switch (hparams.n_layer) {
                    case 27: type = LLM_TYPE_16B; break;
                    case 60: type = LLM_TYPE_236B; break;
                    case 61: type = LLM_TYPE_671B; break;
                    default: type = LLM_TYPE_UNKNOWN;
                }
            } break;
        case LLM_ARCH_PLM:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                ml.get_key(LLM_KV_ATTENTION_KV_LORA_RANK, hparams.n_lora_kv);
                switch (hparams.n_layer) {
                    case 32: type = LLM_TYPE_1_8B; break;
                    default: type = LLM_TYPE_UNKNOWN;
                }
            } break;
        case LLM_ARCH_CHATGLM:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                switch (hparams.n_layer) {
                    case 28: {
                        if (hparams.n_head(0) == 16) {
                            type = LLM_TYPE_1_5B;
                        } else {
                            type = LLM_TYPE_6B;
                        }
                    } break;
                    case 40: {
                        if (hparams.n_head(0) == 24) {
                            type = LLM_TYPE_4B;
                        } else {
                            type = LLM_TYPE_9B;
                        }
                    } break;
                    default: type = LLM_TYPE_UNKNOWN;
                }
            } break;
        case LLM_ARCH_GLM4:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                switch (hparams.n_layer) {
                    case 40: type = LLM_TYPE_9B; break;
                    case 61: type = LLM_TYPE_32B; break;
                    default: type = LLM_TYPE_UNKNOWN;
                }
            } break;
        case LLM_ARCH_GLM4_MOE:
            {
                ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,  hparams.n_ff_exp);
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

                // MoE parameters
                ml.get_key(LLM_KV_EXPERT_COUNT,                hparams.n_expert);
                ml.get_key(LLM_KV_EXPERT_USED_COUNT,           hparams.n_expert_used);
                ml.get_key(LLM_KV_EXPERT_SHARED_COUNT,         hparams.n_expert_shared);
                ml.get_key(LLM_KV_LEADING_DENSE_BLOCK_COUNT,   hparams.n_layer_dense_lead, false);
                ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE,        hparams.expert_weights_scale);
                ml.get_key(LLM_KV_EXPERT_WEIGHTS_NORM,         hparams.expert_weights_norm, false);

                // Expert gating function (GLM-4.5 uses sigmoid)
                ml.get_key(LLM_KV_EXPERT_GATING_FUNC,          hparams.expert_gating_func, false);
                if (hparams.expert_gating_func == LLAMA_EXPERT_GATING_FUNC_TYPE_NONE) {
                    hparams.expert_gating_func = LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID;
                }

                // NextN/MTP parameters
                ml.get_key(LLM_KV_NEXTN_PREDICT_LAYERS,        hparams.nextn_predict_layers, false);

                switch (hparams.n_layer) {
                    case 47: type = LLM_TYPE_106B_A12B; break; // GLM-4.5-Air (46 layers + 1 NextN layer)
                    case 93: type = LLM_TYPE_355B_A32B; break; // GLM-4.5 (92 layers + 1 NextN layer)
                    default: type = LLM_TYPE_UNKNOWN;
                }
            } break;
        case LLM_ARCH_BITNET:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

                switch (hparams.n_layer) {
                    case 26: type = LLM_TYPE_3B; break;
                    default: type = LLM_TYPE_UNKNOWN;
                }
            } break;
        case LLM_ARCH_T5:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,      hparams.f_norm_rms_eps);
                ml.get_key(LLM_KV_ATTENTION_RELATIVE_BUCKETS_COUNT, hparams.n_rel_attn_bkts);

                uint32_t dec_start_token_id;
                if (ml.get_key(LLM_KV_DECODER_START_TOKEN_ID, dec_start_token_id, false)) {
                    hparams.dec_start_token_id = dec_start_token_id;
                }

                switch (hparams.n_layer) {
                    case 6:  type = LLM_TYPE_60M;  break; // t5-small
                    case 8:  type = LLM_TYPE_80M;  break; // flan-t5-small
                    case 12:
                        switch (hparams.n_ff()) {
                            case 3072: type = LLM_TYPE_220M; break; // t5-base
                            case 2048: type = LLM_TYPE_250M; break; // flan-t5-base
                            default: type = LLM_TYPE_UNKNOWN;
                        } break;
                    case 24:
                        switch (hparams.n_ff()) {
                            case 4096:  type = LLM_TYPE_770M; break; // t5-large
                            case 2816:  type = LLM_TYPE_780M; break; // flan-t5-large
                            case 16384: type = LLM_TYPE_3B;   break; // t5-3b
                            case 5120:  type = LLM_TYPE_3B;   break; // flan-t5-xl
                            case 65536: type = LLM_TYPE_11B;  break; // t5-11b
                            case 10240: type = LLM_TYPE_11B;  break; // flan-t5-xxl
                            default: type = LLM_TYPE_UNKNOWN;
                        } break;
                    default: type = LLM_TYPE_UNKNOWN;
               }
            } break;
        case LLM_ARCH_T5ENCODER:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                ml.get_key(LLM_KV_ATTENTION_RELATIVE_BUCKETS_COUNT, hparams.n_rel_attn_bkts);
                type = LLM_TYPE_UNKNOWN;
            } break;
        case LLM_ARCH_JAIS:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps);
                ml.get_key(LLM_KV_ATTENTION_MAX_ALIBI_BIAS, hparams.f_max_alibi_bias);

                switch (hparams.n_layer) {
                    case 24: type = LLM_TYPE_1_3B; break;
                    case 40: type = LLM_TYPE_13B; break;
                    /* TODO: add variants */
                    default: type = LLM_TYPE_UNKNOWN;
                }
            } break;
        case LLM_ARCH_NEMOTRON:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps);
                switch (hparams.n_layer) {
                    case 32: type = LLM_TYPE_4B; break;
                    default: type = LLM_TYPE_UNKNOWN;
                }
            } break;
        case LLM_ARCH_EXAONE:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

                switch (hparams.n_layer) {
                    case 32: type = LLM_TYPE_8B; break;
                    default: type = LLM_TYPE_UNKNOWN;
                }
            } break;
        case LLM_ARCH_EXAONE4:
            {
                if (hparams.n_layer == 64) {    // 32B
                    hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;
                    hparams.n_swa = 4096;
                    hparams.set_swa_pattern(4);
                }

                ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW,    hparams.n_swa, false);
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

                switch (hparams.n_layer) {
                    case 30: type = LLM_TYPE_1_2B; break;
                    case 64: type = LLM_TYPE_32B; break;
                    default: type = LLM_TYPE_UNKNOWN;
                }
            } break;
        case LLM_ARCH_RWKV6:
        case LLM_ARCH_RWKV6QWEN2:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS,     hparams.f_norm_eps, false);
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps, false);
                ml.get_key(LLM_KV_WKV_HEAD_SIZE,               hparams.wkv_head_size);
                ml.get_key(LLM_KV_TIME_MIX_EXTRA_DIM,          hparams.time_mix_extra_dim);
                ml.get_key(LLM_KV_TIME_DECAY_EXTRA_DIM,        hparams.time_decay_extra_dim);
                ml.get_key(LLM_KV_RESCALE_EVERY_N_LAYERS,      hparams.rescale_every_n_layers, false);
                ml.get_key(LLM_KV_TOKEN_SHIFT_COUNT,           hparams.token_shift_count, false);

                switch (hparams.n_layer) {
                    case 24: type = LLM_TYPE_1_6B; break;
                    case 32:
                        switch (hparams.n_embd) {
                            case 2560: type = LLM_TYPE_3B; break;
                            case 4096: type = LLM_TYPE_7B; break;
                            default: type = LLM_TYPE_UNKNOWN;
                        } break;
                    case 61: type = LLM_TYPE_14B; break;
                    case 64: type = LLM_TYPE_32B; break;
                    default: type = LLM_TYPE_UNKNOWN;
                }
            } break;
        case LLM_ARCH_RWKV7:
        case LLM_ARCH_ARWKV7:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS,                hparams.f_norm_eps, false);
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,            hparams.f_norm_rms_eps, false);
                ml.get_key(LLM_KV_WKV_HEAD_SIZE,                          hparams.wkv_head_size);
                ml.get_key(LLM_KV_ATTENTION_DECAY_LORA_RANK,              hparams.n_lora_decay);
                ml.get_key(LLM_KV_ATTENTION_ICLR_LORA_RANK,               hparams.n_lora_iclr);
                ml.get_key(LLM_KV_ATTENTION_VALUE_RESIDUAL_MIX_LORA_RANK, hparams.n_lora_value_res_mix);
                ml.get_key(LLM_KV_ATTENTION_GATE_LORA_RANK,               hparams.n_lora_gate, false);
                ml.get_key(LLM_KV_TOKEN_SHIFT_COUNT,                      hparams.token_shift_count, false);

                switch (hparams.n_layer) {
                    case 12:
                        switch (hparams.n_embd) {
                            case 768: type = LLM_TYPE_190M; break;
                            default: type = LLM_TYPE_UNKNOWN;
                        } break;
                    case 24:
                        switch (hparams.n_embd) {
                            case 1024: type = LLM_TYPE_450M; break;
                            case 2048: type = LLM_TYPE_1_5B; break;
                            default: type = LLM_TYPE_UNKNOWN;
                        } break;
                    case 28:
                        switch (hparams.n_embd) {
                            case 1536: type = LLM_TYPE_1_5B; break;
                            case 3584: type = LLM_TYPE_7B; break;
                            default: type = LLM_TYPE_UNKNOWN;
                        } break;
                    case 32:
                        switch (hparams.n_embd) {
                            case 2560: type = LLM_TYPE_2_9B; break;
                            case 4096: type = LLM_TYPE_7B; break;
                            default: type = LLM_TYPE_UNKNOWN;
                        } break;
                    case 61:
                        switch (hparams.n_embd) {
                            case 4096: type = LLM_TYPE_14B; break;
                            default: type = LLM_TYPE_UNKNOWN;
                        } break;
                    default: type = LLM_TYPE_UNKNOWN;
                }
            } break;
        case LLM_ARCH_GRANITE:
        case LLM_ARCH_GRANITE_MOE:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                ml.get_key(LLM_KV_LOGIT_SCALE,                 hparams.f_logit_scale);
                ml.get_key(LLM_KV_RESIDUAL_SCALE,              hparams.f_residual_scale);
                ml.get_key(LLM_KV_EMBEDDING_SCALE,             hparams.f_embedding_scale);
                ml.get_key(LLM_KV_ATTENTION_SCALE,             hparams.f_attention_scale);

                // Granite uses rope_finetuned as a switch for rope, so default to true
                bool rope_finetuned = true;
                ml.get_key(LLM_KV_ROPE_SCALING_FINETUNED, rope_finetuned, false);
                hparams.rope_finetuned = rope_finetuned;

                switch (hparams.n_layer) {
                    case 32: type = LLM_TYPE_3B; break;
                    case 40: type = LLM_TYPE_3B; break;
                    // Add additional layer/vocab/etc checks here for other model sizes
                    default: type = LLM_TYPE_UNKNOWN;
                }

                // For Granite MoE Shared
                ml.get_key(LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH, hparams.n_ff_shexp, /* required */ false);
            } break;
        case LLM_ARCH_GRANITE_HYBRID:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                ml.get_key(LLM_KV_LOGIT_SCALE,                 hparams.f_logit_scale, /* required */ false);
                ml.get_key(LLM_KV_RESIDUAL_SCALE,              hparams.f_residual_scale, /* required */ false);
                ml.get_key(LLM_KV_EMBEDDING_SCALE,             hparams.f_embedding_scale, /* required */ false);
                ml.get_key(LLM_KV_ATTENTION_SCALE,             hparams.f_attention_scale, /* required */ false);

                ml.get_key(LLM_KV_SSM_CONV_KERNEL,    hparams.ssm_d_conv);
                ml.get_key(LLM_KV_SSM_INNER_SIZE,     hparams.ssm_d_inner);
                ml.get_key(LLM_KV_SSM_STATE_SIZE,     hparams.ssm_d_state);
                ml.get_key(LLM_KV_SSM_TIME_STEP_RANK, hparams.ssm_dt_rank);
                ml.get_key(LLM_KV_SSM_GROUP_COUNT,    hparams.ssm_n_group);

                // Granite uses rope_finetuned as a switch for rope, so default to true
                bool rope_finetuned = true;
                ml.get_key(LLM_KV_ROPE_SCALING_FINETUNED, rope_finetuned, false);
                hparams.rope_finetuned = rope_finetuned;

                // A layer is recurrent IFF the n_head_kv value is set to 0
                for (uint32_t i = 0; i < hparams.n_layer; ++i) {
                    hparams.recurrent_layer_arr[i] = hparams.n_head_kv(i) == 0;
                }

                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

                switch (hparams.n_layer) {
                    // TODO: Add llm type label (not sure this is useful)
                    default: type = LLM_TYPE_UNKNOWN;
                }

                // For Granite MoE Shared
                ml.get_key(LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH, hparams.n_ff_shexp, /* required */ false);
            } break;
        case LLM_ARCH_CHAMELEON:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                hparams.f_norm_eps = 1e-5;  // eps for qk-norm, torch default
                ml.get_key(LLM_KV_SWIN_NORM, hparams.swin_norm);

                switch (hparams.n_layer) {
                    case 32: type = LLM_TYPE_7B; break;
                    case 48: type = LLM_TYPE_34B; break;
                    default: type = LLM_TYPE_UNKNOWN;
               }
            } break;
        case LLM_ARCH_SOLAR:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                for (size_t i = 0; i < hparams.n_bskcn_arr.max_size(); ++i) {
                    auto & bskcn = hparams.n_bskcn_arr[i];
                    bskcn.fill(0);
                    auto kv = LLM_KV(arch);
                    ml.get_key_or_arr(format((kv(LLM_KV_ATTENTION_BLOCK_SKIP_CONNECTION) + ".%d").c_str(), i), bskcn, hparams.n_layer, false);
                }

                switch (hparams.n_layer) {
                    case 64: type = LLM_TYPE_22B; break;
                    default: type = LLM_TYPE_UNKNOWN;
                }
            } break;
        case LLM_ARCH_WAVTOKENIZER_DEC:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS,    hparams.f_norm_eps);
                ml.get_key(LLM_KV_ATTENTION_GROUPNORM_EPS,    hparams.f_norm_group_eps);
                ml.get_key(LLM_KV_ATTENTION_GROUPNORM_GROUPS, hparams.n_norm_groups);
                ml.get_key(LLM_KV_ATTENTION_CAUSAL,           hparams.causal_attn);
            } break;
        case LLM_ARCH_BAILINGMOE:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                ml.get_key(LLM_KV_LEADING_DENSE_BLOCK_COUNT,   hparams.n_layer_dense_lead);
                ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,  hparams.n_ff_exp);
                ml.get_key(LLM_KV_EXPERT_SHARED_COUNT,         hparams.n_expert_shared);
                ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE,        hparams.expert_weights_scale);
                ml.get_key(LLM_KV_EXPERT_WEIGHTS_NORM,         hparams.expert_weights_norm, false);

                switch (hparams.n_layer) {
                    case 28: type = LLM_TYPE_16B; break;
                    case 88: type = LLM_TYPE_290B; break;
                    default: type = LLM_TYPE_UNKNOWN;
                }
            } break;
        case LLM_ARCH_DOTS1:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                ml.get_key(LLM_KV_LEADING_DENSE_BLOCK_COUNT,   hparams.n_layer_dense_lead);
                ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,  hparams.n_ff_exp);
                ml.get_key(LLM_KV_EXPERT_SHARED_COUNT,         hparams.n_expert_shared);
                ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE,        hparams.expert_weights_scale);
                ml.get_key(LLM_KV_EXPERT_WEIGHTS_NORM,         hparams.expert_weights_norm, false);
                ml.get_key(LLM_KV_EXPERT_GATING_FUNC,          hparams.expert_gating_func, false);
                switch (hparams.n_layer) {
                    case 62: type = LLM_TYPE_142B; break;
                    default: type = LLM_TYPE_UNKNOWN;
                }
            } break;
        case LLM_ARCH_ERNIE4_5:
        case LLM_ARCH_ERNIE4_5_MOE:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                if (arch == LLM_ARCH_ERNIE4_5_MOE) {
                    ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,        hparams.n_ff_exp);
                    ml.get_key(LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH, hparams.n_ff_shexp, false);
                    ml.get_key(LLM_KV_INTERLEAVE_MOE_LAYER_STEP,         hparams.n_moe_layer_step);
                    ml.get_key(LLM_KV_LEADING_DENSE_BLOCK_COUNT,         hparams.n_layer_dense_lead);
                }

                switch (hparams.n_layer) {
                    case 18: type = LLM_TYPE_0_3B; break;
                    case 28: type = LLM_TYPE_21B_A3B; break;
                    case 54: type = LLM_TYPE_300B_A47B; break;
                    default: type = LLM_TYPE_UNKNOWN;
                }
            } break;
        case LLM_ARCH_FALCON_H1:
            {
                // Common parameters
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

                // SSM parameters
                ml.get_key(LLM_KV_SSM_CONV_KERNEL,    hparams.ssm_d_conv);
                ml.get_key(LLM_KV_SSM_INNER_SIZE,     hparams.ssm_d_inner);
                ml.get_key(LLM_KV_SSM_STATE_SIZE,     hparams.ssm_d_state);
                ml.get_key(LLM_KV_SSM_TIME_STEP_RANK, hparams.ssm_dt_rank);
                ml.get_key(LLM_KV_SSM_GROUP_COUNT,    hparams.ssm_n_group);

                std::fill(hparams.recurrent_layer_arr.begin(), hparams.recurrent_layer_arr.end(), true);

                switch (hparams.n_layer) {
                    case 36:
                        type = LLM_TYPE_0_5B; break;
                    case 24:
                        type = LLM_TYPE_1_5B; break;
                    case 66:
                        type = LLM_TYPE_1B; break;
                    case 32:
                        type = LLM_TYPE_3B; break;
                    case 44:
                        type = LLM_TYPE_7B; break;
                    case 72:
                        type = LLM_TYPE_34B; break;
                    default:
                        type = LLM_TYPE_UNKNOWN;
                }
            } break;
        case LLM_ARCH_HUNYUAN_MOE:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,       hparams.f_norm_rms_eps);
                ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,        hparams.n_ff_exp);
                ml.get_key(LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH, hparams.n_ff_shexp);

                switch (hparams.n_layer) {
                    case 32: type = LLM_TYPE_A13B; break;
                    default: type = LLM_TYPE_UNKNOWN;
                }
            } break;
        case LLM_ARCH_HUNYUAN_DENSE:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

                switch (hparams.n_embd) {
                    case 1024: type = LLM_TYPE_0_5B; break;
                    case 2048: type = LLM_TYPE_1_8B; break;
                    case 3072: type = LLM_TYPE_4B; break;
                    case 4096: type = LLM_TYPE_7B; break;
                    default: type = LLM_TYPE_UNKNOWN;
                }
            } break;
        case LLM_ARCH_SMOLLM3:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                hparams.n_no_rope_layer_step = 4;

                switch (hparams.n_layer) {
                    case 36: type = LLM_TYPE_3B; break;
                    default: type = LLM_TYPE_UNKNOWN;
                }
            } break;
        case LLM_ARCH_OPENAI_MOE:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,  hparams.n_ff_exp);
                ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW,    hparams.n_swa);

                hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;
                hparams.set_swa_pattern(2);

                // TODO: switch (hparams.n_layer)
            } break;
        case LLM_ARCH_LFM2:
            {
                ml.get_key(LLM_KV_SHORTCONV_L_CACHE,           hparams.n_shortconv_l_cache);
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                for (uint32_t il = 0; il < hparams.n_layer; ++il) {
                    hparams.recurrent_layer_arr[il] = hparams.n_head_kv(il) == 0;
                }
                switch (hparams.n_embd) {
                    case 1024: type = LLM_TYPE_350M; break;
                    case 1536: type = LLM_TYPE_700M; break;
                    case 2048: type = LLM_TYPE_1_2B; break;
                    default:   type = LLM_TYPE_UNKNOWN;
                }
            } break;
        case LLM_ARCH_SMALLTHINKER:
            {
                const bool found_swa = ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW, hparams.n_swa, false);

                if (found_swa && hparams.n_swa > 0) {
                    hparams.swa_type      = LLAMA_SWA_TYPE_STANDARD;
                    hparams.n_swa         = 4096;
                    hparams.set_swa_pattern(4, true);
                } else {
                    hparams.swa_type             = LLAMA_SWA_TYPE_NONE;
                    hparams.n_no_rope_layer_step = hparams.n_layer;
                }

                ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,  hparams.n_ff_exp, false);
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                ml.get_key(LLM_KV_EXPERT_GATING_FUNC,          hparams.expert_gating_func, false);

                switch (hparams.n_layer) {
                    case 32: type = LLM_TYPE_4B;  break;
                    case 52: type = LLM_TYPE_20B; break;
                    default: type = LLM_TYPE_UNKNOWN;
                }
            } break;
        default: throw std::runtime_error("unsupported model architecture");
    }

    pimpl->n_bytes = ml.n_bytes;

    pimpl->desc_str = arch_name() + " " + type_name() + " " + ml.ftype_name();

    if (hparams.f_max_alibi_bias > 0.0f) {
        hparams.use_alibi = true;
    }

    hparams.rope_type = llama_model_rope_type(this);
}

void llama_model::load_vocab(llama_model_loader & ml) {
    const auto kv = LLM_KV(arch);

    vocab.load(ml, kv);
}

bool llama_model::load_tensors(llama_model_loader & ml) {
    const auto & split_mode   = params.split_mode;
    const auto & n_gpu_layers = params.n_gpu_layers;
    const auto & use_mlock    = params.use_mlock;
    const auto & tensor_split = params.tensor_split;

    const int n_layer = hparams.n_layer;

    const bool use_mmap_buffer = true;

    LLAMA_LOG_INFO("%s: loading model tensors, this can take a while... (mmap = %s)\n", __func__, ml.use_mmap ? "true" : "false");

    // build a list of buffer types for the CPU and GPU devices
    pimpl->cpu_buft_list = make_cpu_buft_list(devices, params.use_extra_bufts);
    for (auto * dev : devices) {
        buft_list_t buft_list = make_gpu_buft_list(dev, split_mode, tensor_split);
        // add CPU buffer types as a fallback
        buft_list.insert(buft_list.end(), pimpl->cpu_buft_list.begin(), pimpl->cpu_buft_list.end());
        pimpl->gpu_buft_list.emplace(dev, std::move(buft_list));
    }

    // calculate the split points
    bool all_zero = tensor_split == nullptr || std::all_of(tensor_split, tensor_split + n_devices(), [](float x) { return x == 0.0f; });
    std::vector<float> splits(n_devices());
    if (all_zero) {
        // default split, by free memory
        for (size_t i = 0; i < n_devices(); ++i) {
            ggml_backend_dev_t dev = devices[i];
            size_t total;
            size_t free;
            ggml_backend_dev_memory(dev, &free, &total);
            splits[i] = free;
        }
    } else {
        std::copy(tensor_split, tensor_split + n_devices(), splits.begin());
    }

    // sum and normalize the splits to get the split points
    float split_sum = 0.0f;
    for (size_t i = 0; i < n_devices(); ++i) {
        split_sum += splits[i];
        splits[i] = split_sum;
    }
    for (size_t i = 0; i < n_devices(); ++i) {
        splits[i] /= split_sum;
    }

    ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (cpu_dev == nullptr) {
        throw std::runtime_error(format("%s: no CPU backend found", __func__));
    }
    const int i_gpu_start = std::max((int) hparams.n_layer - n_gpu_layers, (int) 0);
    const int act_gpu_layers = devices.empty() ? 0 : std::min(n_gpu_layers, (int)n_layer + 1);
    auto get_layer_buft_list = [&](int il) -> llama_model::impl::layer_dev {
        const bool is_swa = il < (int) hparams.n_layer && hparams.is_swa(il);
        if (il < i_gpu_start || (il - i_gpu_start) >= act_gpu_layers) {
            LLAMA_LOG_DEBUG("load_tensors: layer %3d assigned to device %s, is_swa = %d\n", il, ggml_backend_dev_name(cpu_dev), is_swa);
            return {cpu_dev, &pimpl->cpu_buft_list};
        }
        const int layer_gpu = std::upper_bound(splits.begin(), splits.begin() + n_devices(), float(il - i_gpu_start)/act_gpu_layers) - splits.begin();
        auto * dev = devices.at(layer_gpu);
        LLAMA_LOG_DEBUG("load_tensors: layer %3d assigned to device %s, is_swa = %d\n", il, ggml_backend_dev_name(dev), is_swa);
        return {dev, &pimpl->gpu_buft_list.at(dev)};
    };

    // assign the input layer
    // there is very little benefit to offloading the input layer, so always keep it on the CPU
    pimpl->dev_input = { cpu_dev, &pimpl->cpu_buft_list };

    // assign the repeating layers to the devices according to the splits
    pimpl->dev_layer.resize(n_layer);
    for (int il = 0; il < n_layer; ++il) {
        pimpl->dev_layer[il] = get_layer_buft_list(il);
    }

    // assign the output layer
    pimpl->dev_output = get_layer_buft_list(n_layer);

    // one ggml context per buffer type
    int max_n_tensors = ml.n_tensors;
    max_n_tensors += 1;         // duplicated output tensor
    max_n_tensors += n_layer*2; // duplicated rope freq tensors
    const size_t ctx_size = ggml_tensor_overhead()*max_n_tensors;

    std::map<ggml_backend_buffer_type_t, ggml_context *> ctx_map;
    auto ctx_for_buft = [&](ggml_backend_buffer_type_t buft) -> ggml_context * {
        auto it = ctx_map.find(buft);
        if (it == ctx_map.end()) {
            ggml_init_params params = {
                /*.mem_size   =*/ ctx_size,
                /*.mem_buffer =*/ NULL,
                /*.no_alloc   =*/ true,
            };

            ggml_context * ctx = ggml_init(params);
            if (!ctx) {
                throw std::runtime_error(format("failed to create ggml context"));
            }

            ctx_map[buft] = ctx;
            pimpl->ctxs.emplace_back(ctx);

            return ctx;
        }
        return it->second;
    };

    const auto TENSOR_DUPLICATED   = llama_model_loader::TENSOR_DUPLICATED;
    const auto TENSOR_NOT_REQUIRED = llama_model_loader::TENSOR_NOT_REQUIRED;
    const auto TENSOR_SKIP         = llama_model_loader::TENSOR_SKIP;

    // create tensors for the weights
    {
        // note: cast to int64_t since we will use these for the tensor dimensions
        const int64_t n_head        = hparams.n_head();
        const int64_t n_head_kv     = hparams.n_head_kv();
        const int64_t n_embd        = hparams.n_embd;
        const int64_t n_embd_k_gqa  = hparams.n_embd_k_gqa();
        const int64_t n_embd_v_gqa  = hparams.n_embd_v_gqa();
        const int64_t n_embd_head_k = hparams.n_embd_head_k;
        const int64_t n_embd_head_v = hparams.n_embd_head_v;
        const int64_t n_ff          = hparams.n_ff();
        const int64_t n_embd_gqa    = n_embd_v_gqa;
        const int64_t n_vocab       = vocab.n_tokens();
        const int64_t n_token_types = vocab.n_token_types();
        const int64_t n_rot         = hparams.n_rot;
        const int64_t n_expert      = hparams.n_expert;
        const int64_t n_expert_used = hparams.n_expert_used;
        const int64_t n_ctx_train   = hparams.n_ctx_train;

        if (n_expert > 0 && hparams.n_expert_used == 0) {
            throw std::runtime_error("model has expert layers but no expert layers are used");
        }

        int n_moved_tensors = 0;
        ggml_tensor * first_moved_tensor = nullptr;
        ggml_backend_buffer_type_t first_moved_from_buft = nullptr;
        ggml_backend_buffer_type_t first_moved_to_buft = nullptr;

        auto create_tensor = [&](const LLM_TN_IMPL & tn, const std::initializer_list<int64_t> & ne, int flags) -> ggml_tensor * {
            ggml_tensor * t_meta = ml.get_tensor_meta(tn.str().c_str());

            if (!t_meta) {
                if (flags & TENSOR_NOT_REQUIRED) {
                    return nullptr;
                }
                throw std::runtime_error(format("missing tensor '%s'", tn.str().c_str()));
            }

            // some models use the token embedding tensor as the output, but since these are used in different layers and with different ops
            // the tensor is duplicated
            // to handle this, we check if the tensor is duplicated, and if so, we assume that it is being loaded as the output tensor
            llm_tensor tn_tensor = tn.tensor;
            if (tn.tensor == LLM_TENSOR_TOKEN_EMBD && flags & TENSOR_DUPLICATED) {
                tn_tensor = LLM_TENSOR_OUTPUT;
            }

            llm_tensor_info info;
            try {
                info = llm_tensor_info_for(tn_tensor);
            } catch (const std::out_of_range & e) {
                throw std::runtime_error(format("missing tensor info mapping for %s", tn.str().c_str()));
            }

            // skip unused tensors
            if (info.op == GGML_OP_NONE || flags & TENSOR_SKIP) {
                const size_t nbytes = ggml_nbytes(t_meta);
                LLAMA_LOG_WARN("model has unused tensor %s (size = %zu bytes) -- ignoring\n", tn.str().c_str(), nbytes);

                ml.size_data -= nbytes;
                ml.n_created++;

                return nullptr;
            }

            // tensors with "bias" suffix are always used with GGML_OP_ADD or GGML_OP_ADD_ID
            ggml_op op;
            bool bias = tn.suffix != nullptr && strcmp(tn.suffix, "bias") == 0;
            if (bias) {
                if (info.op == GGML_OP_MUL_MAT_ID) {
                    op = GGML_OP_ADD_ID;
                } else {
                    op = GGML_OP_ADD;
                }
            } else {
                op = info.op;
            }

            // sanity checks
            if (info.layer == LLM_TENSOR_LAYER_INPUT || info.layer == LLM_TENSOR_LAYER_OUTPUT) {
                if (tn.bid != -1) {
                    GGML_ABORT("input/output layer tensor %s used with a layer number", tn.str().c_str());
                }
            } else {
                if (tn.bid == -1) {
                    GGML_ABORT("repeating layer tensor %s used without a layer number", tn.str().c_str());
                }
            }

            // select the buffer type for this tensor
            buft_list_t * buft_list;
            switch (info.layer) {
                case LLM_TENSOR_LAYER_INPUT:
                    buft_list = pimpl->dev_input.buft_list;
                    break;
                case LLM_TENSOR_LAYER_OUTPUT:
                    buft_list = pimpl->dev_output.buft_list;
                    break;
                case LLM_TENSOR_LAYER_REPEATING:
                    buft_list = pimpl->dev_layer.at(tn.bid).buft_list;
                    break;
                default:
                    GGML_ABORT("invalid layer %d for tensor %s", info.layer, tn.str().c_str());
            }

            ggml_backend_buffer_type_t buft = nullptr;

            // check overrides
            if (ml.tensor_buft_overrides) {
                std::string tensor_name = tn.str();
                for (const auto * overrides = ml.tensor_buft_overrides; overrides->pattern != nullptr; ++overrides) {
                    std::regex pattern(overrides->pattern);
                    if (std::regex_search(tensor_name, pattern)) {
                        if (overrides->buft == ggml_backend_cpu_buffer_type()) {
                            // when overriding to a CPU buffer, consider the extra buffer types
                            buft = select_weight_buft(hparams, t_meta, op, pimpl->cpu_buft_list);
                        } else {
                            buft = overrides->buft;
                        }

                        LLAMA_LOG_DEBUG("tensor %s (%zu MiB %s) buffer type overridden to %s\n",
                                tensor_name.c_str(),
                                ggml_nbytes(t_meta) / 1024 / 1024, ggml_type_name(t_meta->type),
                                ggml_backend_buft_name(buft));
                        break;
                    }
                }
            }

            if (!buft) {
                buft = select_weight_buft(hparams, t_meta, op, *buft_list);
                if (!buft) {
                    throw std::runtime_error(format("failed to find a compatible buffer type for tensor %s", tn.str().c_str()));
                }
            }

            // avoid using a host buffer when using mmap
            auto * buft_dev = ggml_backend_buft_get_device(buft);
            if (ml.use_mmap && buft_dev && buft == ggml_backend_dev_host_buffer_type(buft_dev)) {
                auto * cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
                if (!cpu_dev) {
                    throw std::runtime_error("no CPU backend found");
                }
                buft = ggml_backend_dev_buffer_type(cpu_dev);
            }

            if (buft != buft_list->front().second) {
                n_moved_tensors++;
                if (!first_moved_tensor) {
                    first_moved_tensor = t_meta;
                    first_moved_from_buft = buft_list->front().second;
                    first_moved_to_buft   = buft;
                }
            }

            ggml_context * ctx = ctx_for_buft(buft);

            // if duplicated, check if the original tensor was allocated in the same buffer type context and avoid creating a new one
            if (flags & TENSOR_DUPLICATED) {
                ggml_tensor * t = ggml_get_tensor(ctx, tn.str().c_str());
                if (t) {
                    return t;
                }
            }
            return ml.create_tensor(ctx, tn, ne, flags);
        };

        layers.resize(n_layer);

        // TODO: move to a separate function
        const auto tn = LLM_TN(arch);
        switch (arch) {
            case LLM_ARCH_LLAMA:
            case LLM_ARCH_REFACT:
            case LLM_ARCH_MINICPM:
            case LLM_ARCH_GRANITE:
            case LLM_ARCH_GRANITE_MOE:
                {
                    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

                    // output
                    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
                    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);

                    // if output is NULL, init from the input tok embed
                    if (output == NULL) {
                        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
                    }

                    for (int i = 0; i < n_layer; ++i) {
                        auto & layer = layers[i];

                        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

                        layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_embd_head_k * n_head}, 0);
                        layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_embd_k_gqa}, 0);
                        layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_v_gqa}, 0);
                        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head_k * n_head, n_embd}, 0);

                        // optional bias tensors
                        layer.bq = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "bias", i), {n_embd},     TENSOR_NOT_REQUIRED);
                        layer.bk = create_tensor(tn(LLM_TENSOR_ATTN_K,   "bias", i), {n_embd_gqa}, TENSOR_NOT_REQUIRED);
                        layer.bv = create_tensor(tn(LLM_TENSOR_ATTN_V,   "bias", i), {n_embd_gqa}, TENSOR_NOT_REQUIRED);
                        layer.bo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "bias", i), {n_embd},     TENSOR_NOT_REQUIRED);

                        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);

                        if (hparams.rope_scaling_type_train == LLAMA_ROPE_SCALING_TYPE_LONGROPE) {
                            layer.rope_long  = create_tensor(tn(LLM_TENSOR_ROPE_FACTORS_LONG,  "weight", i), {n_rot/2}, TENSOR_NOT_REQUIRED | (i != 0 ? TENSOR_DUPLICATED : 0));
                            layer.rope_short = create_tensor(tn(LLM_TENSOR_ROPE_FACTORS_SHORT, "weight", i), {n_rot/2}, TENSOR_NOT_REQUIRED | (i != 0 ? TENSOR_DUPLICATED : 0));
                        }
                        else {
                            layer.rope_freqs = create_tensor(tn(LLM_TENSOR_ROPE_FREQS, "weight", i), {n_rot/2}, TENSOR_NOT_REQUIRED | (i != 0 ? TENSOR_DUPLICATED : 0));
                        }

                        if (n_expert == 0) {
                            layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff}, 0);
                            layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd}, 0);
                            layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff}, 0);

                            // optional MLP bias
                            layer.ffn_gate_b = create_tensor(tn(LLM_TENSOR_FFN_GATE, "bias", i), {n_ff}, TENSOR_NOT_REQUIRED);
                            layer.ffn_down_b = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "bias", i), {n_embd}, TENSOR_NOT_REQUIRED);
                            layer.ffn_up_b   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "bias", i), {n_ff}, TENSOR_NOT_REQUIRED);
                        } else {
                            layer.ffn_gate_inp  = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,  "weight", i), {n_embd, n_expert}, 0);
                            layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {n_embd,   n_ff, n_expert}, TENSOR_NOT_REQUIRED);
                            layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {  n_ff, n_embd, n_expert}, 0);
                            layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {n_embd,   n_ff, n_expert}, 0);

                            // For Granite MoE Shared
                            if (hparams.n_ff_shexp > 0) {
                                layer.ffn_gate_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", i), {n_embd, hparams.n_ff_shexp}, 0);
                                layer.ffn_up_shexp   = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,   "weight", i), {n_embd, hparams.n_ff_shexp}, 0);
                                layer.ffn_down_shexp = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i), {hparams.n_ff_shexp, n_embd}, 0);
                            }
                        }
                    }
                } break;
            case LLM_ARCH_LLADA:
                {
                    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, 0);

                    // output
                    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), { n_embd }, 0);
                    output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), { n_embd, n_vocab }, TENSOR_NOT_REQUIRED);

                    // if output is NULL, init from the input tok embed
                    if (output == NULL) {
                        output =
                            create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, TENSOR_DUPLICATED);
                    }

                    for (int i = 0; i < n_layer; ++i) {
                        auto & layer = layers[i];

                        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), { n_embd }, 0);

                        // Use separate Q, K, V projections without bias, matching LLaDALlamaBlock
                        layer.wq =
                            create_tensor(tn(LLM_TENSOR_ATTN_Q, "weight", i), { n_embd, n_embd_head_k * n_head }, 0);
                        layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K, "weight", i), { n_embd, n_embd_k_gqa }, 0);
                        layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V, "weight", i), { n_embd, n_embd_v_gqa }, 0);
                        // No bias for QKV projections as per config: include_bias=false, include_qkv_bias=false
                        layer.wo =
                            create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), { n_embd_head_k * n_head, n_embd }, 0);
                        layer.bo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "bias", i), { n_embd }, TENSOR_NOT_REQUIRED);

                        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), { n_embd }, 0);

                        layer.rope_freqs = create_tensor(tn(LLM_TENSOR_ROPE_FREQS, "weight", i), { n_rot / 2 },
                                                         TENSOR_NOT_REQUIRED | (i != 0 ? TENSOR_DUPLICATED : 0));

                        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), { n_embd, n_ff }, 0);
                        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), { n_ff, n_embd }, 0);
                        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP, "weight", i), { n_embd, n_ff }, 0);

                        // optional MLP bias
                        layer.ffn_gate_b =
                            create_tensor(tn(LLM_TENSOR_FFN_GATE, "bias", i), { n_ff }, TENSOR_NOT_REQUIRED);
                        layer.ffn_down_b =
                            create_tensor(tn(LLM_TENSOR_FFN_DOWN, "bias", i), { n_embd }, TENSOR_NOT_REQUIRED);
                        layer.ffn_up_b = create_tensor(tn(LLM_TENSOR_FFN_UP, "bias", i), { n_ff }, TENSOR_NOT_REQUIRED);
                    }
                }
                break;
            case LLM_ARCH_LLAMA4:
                {
                    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

                    // output
                    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
                    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);

                    // if output is NULL, init from the input tok embed
                    if (output == NULL) {
                        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
                    }

                    GGML_ASSERT(hparams.n_moe_layer_step > 0 && "Llama 4 requires n_moe_layer_step > 0");
                    for (int i = 0; i < n_layer; ++i) {
                        bool is_moe_layer = (i + 1) % hparams.n_moe_layer_step == 0;

                        auto & layer = layers[i];

                        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

                        layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_embd_head_k * n_head}, 0);
                        layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_embd_k_gqa}, 0);
                        layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_v_gqa}, 0);
                        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head_k * n_head, n_embd}, 0);

                        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);

                        layer.rope_freqs = create_tensor(tn(LLM_TENSOR_ROPE_FREQS, "weight", i), {n_rot/2}, TENSOR_NOT_REQUIRED | (i != 0 ? TENSOR_DUPLICATED : 0));

                        if (is_moe_layer) {
                            int n_ff_exp = hparams.n_ff_exp;

                            layer.ffn_gate_inp  = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,  "weight", i), {n_embd, n_expert}, 0);
                            layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {n_embd,   n_ff_exp, n_expert}, 0);
                            layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {  n_ff_exp, n_embd, n_expert}, 0);
                            layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {n_embd,   n_ff_exp, n_expert}, 0);

                            // Shared expert
                            const int64_t n_ff_shexp = n_ff_exp;
                            layer.ffn_gate_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", i), {    n_embd, n_ff_shexp}, 0);
                            layer.ffn_down_shexp = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i), {n_ff_shexp, n_embd    }, 0);
                            layer.ffn_up_shexp   = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,   "weight", i), {    n_embd, n_ff_shexp}, 0);
                        } else {
                            layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff}, 0);
                            layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd}, 0);
                            layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff}, 0);
                        }
                    }
                } break;
            case LLM_ARCH_DECI:
                {
                    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

                    // output
                    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
                    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);

                    // if output is NULL, init from the input tok embed
                    if (output == NULL) {
                        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
                    }

                    for (int i = 0; i < n_layer; ++i) {
                        auto & layer = layers[i];
                        const int64_t n_embd_k_gqa  = hparams.n_embd_k_gqa(i);
                        const int64_t n_embd_v_gqa  = hparams.n_embd_v_gqa(i);
                        const int64_t n_embd_gqa    = hparams.n_embd_v_gqa(i);
                        const int64_t n_ff          = hparams.n_ff(i);
                        const int64_t n_head        = hparams.n_head(i);
                        const int64_t n_head_kv     = hparams.n_head_kv(i);

                        if (n_head_kv == 0 && n_head > 0) {
                            // linear attention for DeciLMCausalModel
                            layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);
                            layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd}, 0);
                        }
                        else if (n_head_kv > 0) {
                            layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

                            layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_embd_head_k * n_head}, 0);
                            layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_embd_k_gqa}, 0);
                            layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_v_gqa}, 0);
                            layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head_k * n_head, n_embd}, 0);
                        }

                        // optional bias tensors
                        layer.bq = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "bias", i), {n_embd},     TENSOR_NOT_REQUIRED);
                        layer.bk = create_tensor(tn(LLM_TENSOR_ATTN_K,   "bias", i), {n_embd_gqa}, TENSOR_NOT_REQUIRED);
                        layer.bv = create_tensor(tn(LLM_TENSOR_ATTN_V,   "bias", i), {n_embd_gqa}, TENSOR_NOT_REQUIRED);
                        layer.bo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "bias", i), {n_embd},     TENSOR_NOT_REQUIRED);

                        if (n_ff > 0) {
                            layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);
                        }

                        if (hparams.rope_scaling_type_train == LLAMA_ROPE_SCALING_TYPE_LONGROPE) {
                            layer.rope_long  = create_tensor(tn(LLM_TENSOR_ROPE_FACTORS_LONG,  "weight", i), {n_rot/2}, TENSOR_NOT_REQUIRED | (i != 0 ? TENSOR_DUPLICATED : 0));
                            layer.rope_short = create_tensor(tn(LLM_TENSOR_ROPE_FACTORS_SHORT, "weight", i), {n_rot/2}, TENSOR_NOT_REQUIRED | (i != 0 ? TENSOR_DUPLICATED : 0));
                        }
                        else {
                            layer.rope_freqs = create_tensor(tn(LLM_TENSOR_ROPE_FREQS, "weight", i), {n_rot/2}, TENSOR_NOT_REQUIRED | (i != 0 ? TENSOR_DUPLICATED : 0));
                        }

                        if (n_ff > 0) {
                            layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff}, 0);
                            layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd}, 0);
                            layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff}, 0);
                        }

                        // optional MLP bias
                        layer.ffn_gate_b = create_tensor(tn(LLM_TENSOR_FFN_GATE, "bias", i), {n_ff}, TENSOR_NOT_REQUIRED);
                        layer.ffn_down_b = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "bias", i), {n_embd}, TENSOR_NOT_REQUIRED);
                        layer.ffn_up_b   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "bias", i), {n_ff}, TENSOR_NOT_REQUIRED);
                    }
                } break;
            case LLM_ARCH_MINICPM3:
                {
                    const int64_t n_embd_head_qk_rope = hparams.n_rot;
                    const int64_t n_embd_head_qk_nope = hparams.n_embd_head_k - hparams.n_rot;

                    const int64_t q_lora_rank  = hparams.n_lora_q;
                    const int64_t kv_lora_rank = hparams.n_lora_kv;
                    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

                    // output
                    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
                    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);

                    // if output is NULL, init from the input tok embed
                    if (output == NULL) {
                        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
                    }

                    for (int i = 0; i < n_layer; ++i) {
                        auto & layer = layers[i];

                        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);
                        layer.attn_q_a_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_A_NORM, "weight", i), {q_lora_rank}, 0);

                        layer.attn_kv_a_norm = create_tensor(tn(LLM_TENSOR_ATTN_KV_A_NORM, "weight", i), {kv_lora_rank}, 0);

                        layer.wq_a = create_tensor(tn(LLM_TENSOR_ATTN_Q_A, "weight", i), {n_embd, q_lora_rank}, 0);
                        layer.wq_b = create_tensor(tn(LLM_TENSOR_ATTN_Q_B, "weight", i), {q_lora_rank, n_head * n_embd_head_k}, 0);

                        layer.wkv_a_mqa = create_tensor(tn(LLM_TENSOR_ATTN_KV_A_MQA, "weight", i), {n_embd, kv_lora_rank + (n_embd_head_qk_rope)}, 0);
                        layer.wkv_b     = create_tensor(tn(LLM_TENSOR_ATTN_KV_B,     "weight", i), {kv_lora_rank, n_head * (n_embd_head_qk_nope + n_embd_head_v)}, 0);
                        layer.wo        = create_tensor(tn(LLM_TENSOR_ATTN_OUT,      "weight", i), {              n_head * (                      n_embd_head_v), n_embd}, 0);

                        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);

                        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff}, 0);
                        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd}, 0);
                        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff}, 0);

                        layer.rope_long  = create_tensor(tn(LLM_TENSOR_ROPE_FACTORS_LONG,  "weight", i), { n_embd_head_qk_rope/2 }, TENSOR_NOT_REQUIRED | (i != 0 ? TENSOR_DUPLICATED : 0));
                        layer.rope_short = create_tensor(tn(LLM_TENSOR_ROPE_FACTORS_SHORT, "weight", i), { n_embd_head_qk_rope/2 }, TENSOR_NOT_REQUIRED | (i != 0 ? TENSOR_DUPLICATED : 0));
                    }
                } break;
            case LLM_ARCH_GROK:
                {
                    if (n_expert == 0) {
                        throw std::runtime_error("Grok model cannot have zero experts");
                    }

                    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

                    // output
                    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
                    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);

                    // if output is NULL, init from the input tok embed
                    if (output == NULL) {
                        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
                    }

                    for (int i = 0; i < n_layer; ++i) {
                        auto & layer = layers[i];

                        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

                        layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_embd}, 0);
                        layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_embd_gqa}, 0);
                        layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_gqa}, 0);
                        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd}, 0);

                        layer.attn_out_norm   = create_tensor(tn(LLM_TENSOR_ATTN_OUT_NORM, "weight", i), {n_embd}, 0);

                        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);

                        layer.ffn_gate_inp  = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,  "weight", i), {n_embd, n_expert}, 0);
                        layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {n_embd, n_ff, n_expert}, TENSOR_NOT_REQUIRED);
                        layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {  n_ff, n_embd, n_expert}, 0);
                        layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {n_embd,   n_ff, n_expert}, 0);

                        layer.layer_out_norm   = create_tensor(tn(LLM_TENSOR_LAYER_OUT_NORM, "weight", i), {n_embd}, 0);
                    }
                } break;
            case LLM_ARCH_DBRX:
                {
                    if (n_expert == 0) {
                        throw std::runtime_error("DBRX model cannot have zero experts");
                    }

                    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

                    // output
                    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
                    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, 0);

                    for (int i = 0; i < n_layer; ++i) {
                        auto & layer = layers[i];

                        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

                        layer.wqkv = create_tensor(tn(LLM_TENSOR_ATTN_QKV, "weight", i), {n_embd, n_embd + 2*n_embd_gqa}, 0);
                        layer.wo   = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd}, 0);

                        layer.attn_out_norm = create_tensor(tn(LLM_TENSOR_ATTN_OUT_NORM, "weight", i), {n_embd}, 0);

                        layer.ffn_gate_inp  = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,  "weight", i), {n_embd, n_expert}, 0);
                        layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {n_embd, n_ff,   n_expert}, 0);
                        layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {n_ff,   n_embd, n_expert}, 0);
                        layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {n_embd, n_ff,   n_expert}, 0);
                    }
                } break;
            case LLM_ARCH_BAICHUAN:
                {
                    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);
                    {
                        output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
                        output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, 0);
                    }

                    for (int i = 0; i < n_layer; ++i) {
                        auto & layer = layers[i];

                        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

                        layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_embd}, 0);
                        layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_embd_gqa}, 0);
                        layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_gqa}, 0);
                        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd}, 0);

                        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);

                        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff}, 0);
                        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd}, 0);
                        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff}, 0);
                    }
                } break;
            case LLM_ARCH_FALCON:
                {
                    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

                    // output
                    {
                        output_norm   = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
                        output_norm_b = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "bias"),   {n_embd}, 0);

                        output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);
                        if (!output) {
                            output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED); // needs to be on GPU
                        }
                    }

                    for (int i = 0; i < n_layer; ++i) {
                        auto & layer = layers[i];

                        layer.attn_norm   = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);
                        layer.attn_norm_b = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "bias", i),   {n_embd}, 0);

                        layer.attn_norm_2   = create_tensor(tn(LLM_TENSOR_ATTN_NORM_2, "weight", i), {n_embd}, TENSOR_NOT_REQUIRED);
                        layer.attn_norm_2_b = create_tensor(tn(LLM_TENSOR_ATTN_NORM_2, "bias", i),   {n_embd}, TENSOR_NOT_REQUIRED);

                        layer.wqkv = create_tensor(tn(LLM_TENSOR_ATTN_QKV, "weight", i), {n_embd, n_embd + 2*n_embd_gqa}, 0);
                        layer.wo   = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd}, 0);

                        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd}, 0);
                        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff}, 0);
                    }
                } break;
            case LLM_ARCH_STARCODER:
                {
                    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);
                    pos_embd = create_tensor(tn(LLM_TENSOR_POS_EMBD,   "weight"), {n_embd, n_ctx_train}, 0);

                    // output
                    {
                        output_norm   = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
                        output_norm_b = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "bias"),   {n_embd}, 0);
                        output        = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);
                        if (!output) {
                            // needs to be on GPU
                            output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
                        }

                    }

                    for (int i = 0; i < n_layer; ++i) {
                        auto & layer = layers[i];

                        layer.attn_norm   = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);
                        layer.attn_norm_b = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "bias", i),   {n_embd}, 0);

                        layer.wqkv = create_tensor(tn(LLM_TENSOR_ATTN_QKV, "weight", i), {n_embd, n_embd + 2*n_embd_gqa}, 0);
                        layer.bqkv = create_tensor(tn(LLM_TENSOR_ATTN_QKV, "bias", i),   {n_embd + 2*n_embd_gqa}, 0);

                        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd}, 0);
                        layer.bo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "bias", i),   {n_embd}, 0);

                        layer.ffn_norm   = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);
                        layer.ffn_norm_b = create_tensor(tn(LLM_TENSOR_FFN_NORM, "bias", i),   {n_embd}, 0);

                        layer.ffn_down   = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff, n_embd}, 0);
                        layer.ffn_down_b = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "bias", i),   {n_embd}, 0);

                        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP, "weight", i),   {n_embd, n_ff}, 0);
                        layer.ffn_up_b = create_tensor(tn(LLM_TENSOR_FFN_UP, "bias", i),     {n_ff}, 0);
                    }
                } break;
            case LLM_ARCH_BERT:
            case LLM_ARCH_NOMIC_BERT:
            case LLM_ARCH_NOMIC_BERT_MOE:
                {
                    tok_embd     = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD,  "weight"), {n_embd, n_vocab}, 0);
                    type_embd    = create_tensor(tn(LLM_TENSOR_TOKEN_TYPES, "weight"), {n_embd, n_token_types}, TENSOR_NOT_REQUIRED);

                    if (arch == LLM_ARCH_BERT) {
                        pos_embd = create_tensor(tn(LLM_TENSOR_POS_EMBD,    "weight"), {n_embd, n_ctx_train}, 0);

                        cls   = create_tensor(tn(LLM_TENSOR_CLS, "weight"), {n_embd, n_embd}, TENSOR_NOT_REQUIRED);
                        cls_b = create_tensor(tn(LLM_TENSOR_CLS, "bias"),   {n_embd},         TENSOR_NOT_REQUIRED);

                        cls_out   = create_tensor(tn(LLM_TENSOR_CLS_OUT, "weight"), {n_embd, hparams.n_cls_out}, TENSOR_NOT_REQUIRED);
                        cls_out_b = create_tensor(tn(LLM_TENSOR_CLS_OUT, "bias"),   {hparams.n_cls_out},         TENSOR_NOT_REQUIRED);
                    }

                    tok_norm   = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD_NORM, "weight"), {n_embd}, 0);
                    tok_norm_b = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD_NORM, "bias"),   {n_embd}, 0);

                    for (int i = 0; i < n_layer; ++i) {
                        auto & layer = layers[i];

                        layer.wqkv = create_tensor(tn(LLM_TENSOR_ATTN_QKV, "weight", i), {n_embd, n_embd + 2*n_embd_gqa}, TENSOR_NOT_REQUIRED);
                        layer.bqkv = create_tensor(tn(LLM_TENSOR_ATTN_QKV, "bias", i), {n_embd + 2*n_embd_gqa}, TENSOR_NOT_REQUIRED);

                        if (!layer.wqkv) {
                            layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_embd}, 0);
                            layer.bq = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "bias", i),   {n_embd}, 0);

                            layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_embd_gqa}, 0);
                            layer.bk = create_tensor(tn(LLM_TENSOR_ATTN_K,   "bias", i),   {n_embd_gqa}, 0);

                            layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_gqa}, 0);
                            layer.bv = create_tensor(tn(LLM_TENSOR_ATTN_V,   "bias", i),   {n_embd_gqa}, 0);
                        }

                        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT,      "weight", i), {n_embd, n_embd}, 0);

                        layer.attn_out_norm   = create_tensor(tn(LLM_TENSOR_ATTN_OUT_NORM, "weight", i), {n_embd}, 0);
                        layer.attn_out_norm_b = create_tensor(tn(LLM_TENSOR_ATTN_OUT_NORM, "bias", i),   {n_embd}, 0);

                        if (hparams.moe_every_n_layers > 0 && i % hparams.moe_every_n_layers == 1) {
                            layer.bo         = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "bias", i), {n_embd}, 0);
                            layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {  n_embd, n_ff,   n_expert}, 0);
                            layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {  n_ff,   n_embd, n_expert}, 0);
                            layer.ffn_gate_inp = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,   "weight", i), {n_embd, n_expert}, 0);
                        } else {
                            layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,        "weight", i), {n_embd, n_ff}, 0);
                            layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN,      "weight", i), {n_ff, n_embd}, 0);

                            if (arch == LLM_ARCH_BERT || arch == LLM_ARCH_NOMIC_BERT_MOE) {
                                layer.bo         = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "bias", i), {n_embd}, 0);
                                layer.ffn_up_b   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "bias", i), {n_ff}, 0);
                                layer.ffn_down_b = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "bias", i), {n_embd}, 0);
                            } else {
                                layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd, n_ff}, 0);
                            }
                        }

                        layer.layer_out_norm   = create_tensor(tn(LLM_TENSOR_LAYER_OUT_NORM, "weight", i), {n_embd}, 0);
                        layer.layer_out_norm_b = create_tensor(tn(LLM_TENSOR_LAYER_OUT_NORM, "bias", i),   {n_embd}, 0);
                    }
                } break;
            case LLM_ARCH_NEO_BERT:
                {
                    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD,  "weight"), {n_embd, n_vocab}, 0);

                    cls   = create_tensor(tn(LLM_TENSOR_CLS, "weight"), {n_embd, n_embd}, TENSOR_NOT_REQUIRED);
                    cls_b = create_tensor(tn(LLM_TENSOR_CLS, "bias"),   {n_embd},         TENSOR_NOT_REQUIRED);

                    cls_out   = create_tensor(tn(LLM_TENSOR_CLS_OUT, "weight"), {n_embd, hparams.n_cls_out}, TENSOR_NOT_REQUIRED);
                    cls_out_b = create_tensor(tn(LLM_TENSOR_CLS_OUT, "bias"),   {hparams.n_cls_out},         TENSOR_NOT_REQUIRED);

                    output_norm_enc = create_tensor(tn(LLM_TENSOR_ENC_OUTPUT_NORM, "weight"), {n_embd}, 0);

                    for (int i = 0; i < n_layer; ++i) {
                        auto & layer = layers[i];

                        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

                        layer.wqkv = create_tensor(tn(LLM_TENSOR_ATTN_QKV, "weight", i), {n_embd, n_embd + 2*n_embd_gqa}, 0);
                        layer.wo   = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd}, 0);

                        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);

                        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd, n_ff*2}, 0);
                        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff, n_embd}, 0);
                    }
                } break;
            case LLM_ARCH_JINA_BERT_V2:
                {
                    tok_embd  = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD,  "weight"), {n_embd, n_vocab}, 0); // word_embeddings
                    type_embd = create_tensor(tn(LLM_TENSOR_TOKEN_TYPES, "weight"), {n_embd, n_token_types}, 0); // token_type_embeddings

                    tok_norm   = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD_NORM, "weight"), {n_embd}, 0); // LayerNorm
                    tok_norm_b = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD_NORM, "bias"),   {n_embd}, 0); //LayerNorm bias

                    cls   = create_tensor(tn(LLM_TENSOR_CLS, "weight"), {n_embd, 1}, TENSOR_NOT_REQUIRED);
                    cls_b = create_tensor(tn(LLM_TENSOR_CLS, "bias"),   {1},         TENSOR_NOT_REQUIRED);
                    for (int i = 0; i < n_layer; ++i) {
                        auto & layer = layers[i]; // JinaBertLayer

                        layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q, "weight", i), {n_embd, n_embd}, 0);
                        layer.bq = create_tensor(tn(LLM_TENSOR_ATTN_Q, "bias", i),   {n_embd}, 0);

                        layer.attn_q_norm   = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), {n_embd}, TENSOR_NOT_REQUIRED);
                        layer.attn_q_norm_b = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "bias",   i), {n_embd}, TENSOR_NOT_REQUIRED);

                        layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K, "weight", i), {n_embd, n_embd_gqa}, 0);
                        layer.bk = create_tensor(tn(LLM_TENSOR_ATTN_K, "bias",   i), {n_embd_gqa}, 0);

                        layer.attn_k_norm   = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), {n_embd}, TENSOR_NOT_REQUIRED);
                        layer.attn_k_norm_b = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "bias",   i), {n_embd}, TENSOR_NOT_REQUIRED);

                        layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V, "weight", i), {n_embd, n_embd_gqa}, 0);
                        layer.bv = create_tensor(tn(LLM_TENSOR_ATTN_V, "bias",   i), {n_embd_gqa}, 0);

                        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd}, 0); //output_dens
                        layer.bo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "bias",   i), {n_embd}, 0); //output_dens

                        layer.attn_out_norm   = create_tensor(tn(LLM_TENSOR_ATTN_OUT_NORM, "weight", i), {n_embd}, 0); //output_norm
                        layer.attn_out_norm_b = create_tensor(tn(LLM_TENSOR_ATTN_OUT_NORM, "bias",   i), {n_embd}, 0);

                        layer.attn_norm_2   = create_tensor(tn(LLM_TENSOR_ATTN_NORM_2, "weight", i), {n_embd}, TENSOR_NOT_REQUIRED);
                        layer.attn_norm_2_b = create_tensor(tn(LLM_TENSOR_ATTN_NORM_2, "bias",   i), {n_embd}, TENSOR_NOT_REQUIRED);

                        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd, n_ff}, TENSOR_NOT_REQUIRED);
                        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd, layer.ffn_gate ? n_ff : n_ff * 2}, 0);

                        layer.ffn_down   = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff, n_embd}, 0);
                        layer.ffn_down_b = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "bias",   i), {n_embd}, 0);

                        layer.layer_out_norm   = create_tensor(tn(LLM_TENSOR_LAYER_OUT_NORM, "weight", i), {n_embd}, 0);
                        layer.layer_out_norm_b = create_tensor(tn(LLM_TENSOR_LAYER_OUT_NORM, "bias",   i), {n_embd}, 0);
                    }
                } break;
            case LLM_ARCH_BLOOM:
                {
                    tok_embd   = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD,      "weight"), {n_embd, n_vocab}, 0);
                    tok_norm   = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD_NORM, "weight"), {n_embd}, 0);
                    tok_norm_b = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD_NORM, "bias"),   {n_embd}, 0);

                    // output
                    output_norm   = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
                    output_norm_b = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "bias"),   {n_embd}, 0);
                    output        = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);

                    // if output is NULL, init from the input tok embed
                    if (output == NULL) {
                        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
                    }

                    for (int i = 0; i < n_layer; ++i) {
                        auto & layer = layers[i];

                        layer.attn_norm   = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);
                        layer.attn_norm_b = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "bias",   i), {n_embd}, 0);

                        layer.wqkv = create_tensor(tn(LLM_TENSOR_ATTN_QKV, "weight", i), {n_embd, n_embd + 2*n_embd_gqa}, 0);
                        layer.bqkv = create_tensor(tn(LLM_TENSOR_ATTN_QKV, "bias",   i), {n_embd + 2*n_embd_gqa}, 0);

                        layer.wo   = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd}, 0);
                        layer.bo   = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "bias",   i), {n_embd}, 0);

                        layer.ffn_norm   = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);
                        layer.ffn_norm_b = create_tensor(tn(LLM_TENSOR_FFN_NORM, "bias",   i), {n_embd}, 0);

                        layer.ffn_down   = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff, n_embd}, 0);
                        layer.ffn_down_b = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "bias",   i), {n_embd}, 0);

                        layer.ffn_up     = create_tensor(tn(LLM_TENSOR_FFN_UP, "weight", i), {n_embd, n_ff}, 0);
                        layer.ffn_up_b   = create_tensor(tn(LLM_TENSOR_FFN_UP, "bias",   i), {n_ff}, 0);
                    }
                } break;
            case LLM_ARCH_MPT:
                {
                    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);
                    pos_embd = create_tensor(tn(LLM_TENSOR_POS_EMBD,   "weight"), {n_embd, n_ctx_train}, TENSOR_NOT_REQUIRED);

                    // output
                    output_norm   = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
                    output_norm_b = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "bias"),   {n_embd}, TENSOR_NOT_REQUIRED);

                    output        = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);
                    if (!output) {
                        output    = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED); // needs to be on GPU
                    }

                    for (int i = 0; i < n_layer; ++i) {
                        auto & layer = layers[i];

                        layer.attn_norm   = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);
                        layer.attn_norm_b = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "bias", i),   {n_embd}, TENSOR_NOT_REQUIRED);

                        layer.wqkv = create_tensor(tn(LLM_TENSOR_ATTN_QKV, "weight", i), {n_embd, n_embd + 2*n_embd_gqa}, 0);
                        layer.bqkv = create_tensor(tn(LLM_TENSOR_ATTN_QKV, "bias", i),   {n_embd + 2*n_embd_gqa}, TENSOR_NOT_REQUIRED);

                        layer.wo   = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd}, 0);
                        layer.bo   = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "bias", i),   {n_embd}, TENSOR_NOT_REQUIRED);

                        layer.ffn_norm   = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);
                        layer.ffn_norm_b = create_tensor(tn(LLM_TENSOR_FFN_NORM, "bias", i),   {n_embd}, TENSOR_NOT_REQUIRED);

                        layer.ffn_down   = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff, n_embd}, 0);
                        layer.ffn_down_b = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "bias", i),   {n_embd}, TENSOR_NOT_REQUIRED);

                        layer.ffn_up     = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff}, 0);
                        layer.ffn_up_b   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "bias", i),   {n_ff}, TENSOR_NOT_REQUIRED);

                        layer.attn_q_norm   = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), {n_embd}, TENSOR_NOT_REQUIRED);
                        layer.attn_q_norm_b = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "bias",   i), {n_embd}, TENSOR_NOT_REQUIRED);

                        layer.attn_k_norm   = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), {n_embd}, TENSOR_NOT_REQUIRED);
                        layer.attn_k_norm_b = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "bias",   i), {n_embd}, TENSOR_NOT_REQUIRED);

                        // AWQ ScaleActivation layer
                        layer.ffn_act = create_tensor(tn(LLM_TENSOR_FFN_ACT, "scales", i), {n_ff}, TENSOR_NOT_REQUIRED);
                    }
                } break;
            case LLM_ARCH_STABLELM:
                {
                    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

                    // output
                    output_norm_b = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "bias"),   {n_embd}, 0);
                    output_norm   = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
                    output        = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, 0);

                    for (int i = 0; i < n_layer; ++i) {
                        auto & layer = layers[i];

                        layer.attn_norm =   create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);
                        layer.attn_norm_b = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "bias", i), {n_embd}, 0);

                        layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_embd}, 0);
                        layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_embd_gqa}, 0);
                        layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_gqa}, 0);
                        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd}, 0);

                        // optional bias tensors, present in Stable LM 2 1.6B
                        layer.bq = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "bias", i), {n_embd},     TENSOR_NOT_REQUIRED);
                        layer.bk = create_tensor(tn(LLM_TENSOR_ATTN_K,   "bias", i), {n_embd_gqa}, TENSOR_NOT_REQUIRED);
                        layer.bv = create_tensor(tn(LLM_TENSOR_ATTN_V,   "bias", i), {n_embd_gqa}, TENSOR_NOT_REQUIRED);

                        // optional q and k layernorms, present in StableLM 2 12B
                        layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), {n_embd_head_k, n_head},    TENSOR_NOT_REQUIRED);
                        layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), {n_embd_head_k, n_head_kv}, TENSOR_NOT_REQUIRED);

                        // optional FFN norm, not present in StableLM 2 12B which uses parallel residual
                        layer.ffn_norm   = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, TENSOR_NOT_REQUIRED);
                        layer.ffn_norm_b = create_tensor(tn(LLM_TENSOR_FFN_NORM, "bias", i),   {n_embd}, TENSOR_NOT_REQUIRED);

                        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff}, 0);
                        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd}, 0);
                        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff}, 0);
                    }
                } break;
            case LLM_ARCH_QWEN:
                {
                    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

                    // output
                    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
                    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, 0);

                    for (int i = 0; i < n_layer; ++i) {
                        auto & layer = layers[i];

                        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

                        layer.wqkv = create_tensor(tn(LLM_TENSOR_ATTN_QKV, "weight", i), {n_embd, n_embd*3}, 0);
                        layer.bqkv = create_tensor(tn(LLM_TENSOR_ATTN_QKV, "bias", i),   {n_embd*3}, 0);
                        layer.wo   = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd}, 0);

                        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);

                        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd, n_ff/2}, 0);
                        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff/2, n_embd}, 0);
                        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd, n_ff/2}, 0);
                    }
                } break;
            case LLM_ARCH_QWEN2:
            case LLM_ARCH_QWEN2VL:
            case LLM_ARCH_DREAM:
                {
                    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

                    // output
                    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
                    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);
                    output_b    = create_tensor(tn(LLM_TENSOR_OUTPUT,      "bias"),   {n_vocab}, TENSOR_NOT_REQUIRED);
                    // if output is NULL, init from the input tok embed
                    if (output == NULL) {
                        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
                    }

                    for (int i = 0; i < n_layer; ++i) {
                        auto & layer = layers[i];

                        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

                        layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_embd}, 0);
                        layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_embd_gqa}, 0);
                        layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_gqa}, 0);
                        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd}, 0);

                        // optional bias tensors
                        layer.bq = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "bias", i), {n_embd}, 0);
                        layer.bk = create_tensor(tn(LLM_TENSOR_ATTN_K,   "bias", i), {n_embd_gqa}, 0);
                        layer.bv = create_tensor(tn(LLM_TENSOR_ATTN_V,   "bias", i), {n_embd_gqa}, 0);

                        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);

                        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff}, 0);
                        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd}, 0);
                        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff}, 0);
                    }
                } break;
            case LLM_ARCH_QWEN2MOE:
                {
                    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

                    // output
                    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
                    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, 0);

                    for (int i = 0; i < n_layer; ++i) {
                        auto & layer = layers[i];

                        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

                        layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_embd}, 0);
                        layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_embd_gqa}, 0);
                        layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_gqa}, 0);
                        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd}, 0);

                        // optional bias tensors
                        layer.bq = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "bias", i), {n_embd}, TENSOR_NOT_REQUIRED);
                        layer.bk = create_tensor(tn(LLM_TENSOR_ATTN_K,   "bias", i), {n_embd_gqa}, TENSOR_NOT_REQUIRED);
                        layer.bv = create_tensor(tn(LLM_TENSOR_ATTN_V,   "bias", i), {n_embd_gqa}, TENSOR_NOT_REQUIRED);

                        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);

                        layer.ffn_gate_inp = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP, "weight", i), {n_embd, n_expert}, 0);

                        if (n_expert == 0) {
                            throw std::runtime_error("n_expert must be > 0 for QWEN2MOE");
                        }
                        if (n_expert_used == 0) {
                            throw std::runtime_error("n_expert_used must be > 0 for QWEN2MOE");
                        }

                        // MoE branch
                        const int64_t n_ff_exp = hparams.n_ff_exp ? hparams.n_ff_exp : n_ff / n_expert_used;

                        layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {  n_embd, n_ff_exp, n_expert}, 0);
                        layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {n_ff_exp,   n_embd, n_expert}, 0);
                        layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {  n_embd, n_ff_exp, n_expert}, 0);

                        // Shared expert branch
                        const int64_t n_ff_shexp = hparams.n_ff_shexp ? hparams.n_ff_shexp : n_ff;

                        layer.ffn_gate_inp_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP_SHEXP, "weight", i), {n_embd}, 0);
                        layer.ffn_gate_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", i), {    n_embd, n_ff_shexp}, 0);
                        layer.ffn_down_shexp = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i), {n_ff_shexp,     n_embd}, 0);
                        layer.ffn_up_shexp   = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,   "weight", i), {    n_embd, n_ff_shexp}, 0);
                    }
                } break;
            case LLM_ARCH_QWEN3:
                {
                    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

                    // output
                    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
                    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);
                    // if output is NULL, init from the input tok embed
                    if (output == NULL) {
                        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
                    }

                    for (int i = 0; i < n_layer; ++i) {
                        auto & layer = layers[i];

                        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

                        layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_embd_head_k * n_head}, 0);
                        layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_embd_gqa}, 0);
                        layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_gqa}, 0);
                        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head_k * n_head, n_embd}, 0);

                        layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), {n_embd_head_k}, 0);
                        layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), {n_embd_head_k}, 0);

                        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);
                        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff}, 0);
                        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd}, 0);
                        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff}, 0);
                    }
                } break;
            case LLM_ARCH_QWEN3MOE:
                {
                    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

                    // output
                    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
                    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);
                    // if output is NULL, init from the input tok embed
                    if (output == NULL) {
                        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
                    }

                    for (int i = 0; i < n_layer; ++i) {
                        auto & layer = layers[i];

                        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

                        layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_embd_head_k * n_head}, 0);
                        layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_embd_gqa}, 0);
                        layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_gqa}, 0);
                        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head_k * n_head, n_embd}, 0);

                        layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), {n_embd_head_k}, 0);
                        layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), {n_embd_head_k}, 0);

                        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);

                        layer.ffn_gate_inp = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP, "weight", i), {n_embd, n_expert}, 0);

                        if (n_expert == 0) {
                            throw std::runtime_error("n_expert must be > 0 for QWEN3MOE");
                        }
                        if (n_expert_used == 0) {
                            throw std::runtime_error("n_expert_used must be > 0 for QWEN3MOE");
                        }

                        // MoE branch
                        const int64_t n_ff_exp = hparams.n_ff_exp ? hparams.n_ff_exp : n_ff / n_expert_used;

                        layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {  n_embd, n_ff_exp, n_expert}, 0);
                        layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {n_ff_exp,   n_embd, n_expert}, 0);
                        layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {  n_embd, n_ff_exp, n_expert}, 0);
                    }
                } break;
            case LLM_ARCH_PHI2:
                {
                    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

                    // output
                    output_norm   = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
                    output_norm_b = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "bias"),   {n_embd}, 0);
                    output        = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, 0);
                    output_b      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "bias"),   {n_vocab}, 0);

                    for (int i = 0; i < n_layer; ++i) {
                        auto & layer = layers[i];

                        layer.attn_norm   = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);
                        layer.attn_norm_b = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "bias", i),   {n_embd}, 0);

                        layer.wqkv = create_tensor(tn(LLM_TENSOR_ATTN_QKV, "weight", i), {n_embd, n_embd + 2*n_embd_gqa}, TENSOR_NOT_REQUIRED);
                        layer.bqkv = create_tensor(tn(LLM_TENSOR_ATTN_QKV, "bias", i),   {n_embd + 2*n_embd_gqa}, TENSOR_NOT_REQUIRED);

                        if (layer.wqkv == nullptr) {
                            layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q, "weight", i), {n_embd, n_embd}, 0);
                            layer.bq = create_tensor(tn(LLM_TENSOR_ATTN_Q, "bias", i),   {n_embd}, 0);

                            layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K, "weight", i), {n_embd, n_embd_gqa}, 0);
                            layer.bk = create_tensor(tn(LLM_TENSOR_ATTN_K, "bias", i),   {n_embd_gqa}, 0);

                            layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V, "weight", i), {n_embd, n_embd_gqa}, 0);
                            layer.bv = create_tensor(tn(LLM_TENSOR_ATTN_V, "bias", i),   {n_embd_gqa}, 0);
                        }

                        layer.wo   = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd}, 0);
                        layer.bo   = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "bias", i),   {n_embd}, 0);

                        layer.ffn_down   = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff, n_embd}, 0);
                        layer.ffn_down_b = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "bias", i),   {n_embd}, 0);

                        layer.ffn_up     = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd, n_ff}, 0);
                        layer.ffn_up_b   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "bias", i),   {n_ff}, 0);
                    }
                } break;
            case LLM_ARCH_PHI3:
                {
                    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, 0);

                    // output
                    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), { n_embd }, 0);
                    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);

                    // if output is NULL, init from the input tok embed
                    if (output == NULL) {
                        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
                    }

                    for (int i = 0; i < n_layer; ++i) {
                        auto & layer = layers[i];

                        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), { n_embd }, 0);

                        layer.wqkv = create_tensor(tn(LLM_TENSOR_ATTN_QKV, "weight", i), { n_embd, n_embd + 2 * n_embd_gqa }, TENSOR_NOT_REQUIRED);
                        layer.wo   = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), { n_embd, n_embd }, 0);

                        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), { n_embd }, 0);

                        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), { n_ff, n_embd }, 0);
                        layer.ffn_up = create_tensor(tn(LLM_TENSOR_FFN_UP, "weight", i), { n_embd, 2 * n_ff }, 0);

                        layer.rope_long  = create_tensor(tn(LLM_TENSOR_ROPE_FACTORS_LONG,  "weight", i), { n_rot/2 }, TENSOR_NOT_REQUIRED | (i != 0 ? TENSOR_DUPLICATED : 0));
                        layer.rope_short = create_tensor(tn(LLM_TENSOR_ROPE_FACTORS_SHORT, "weight", i), { n_rot/2 }, TENSOR_NOT_REQUIRED | (i != 0 ? TENSOR_DUPLICATED : 0));
                    }
                } break;
            case LLM_ARCH_PHIMOE:
                {
                    const int64_t n_embd_head = n_embd / n_head;

                    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, 0);

                    // output
                    output_norm   = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), { n_embd }, 0);
                    output_norm_b = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "bias"),   {n_embd}, 0);
                    output        = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), { n_embd, n_vocab }, 0);
                    output_b      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "bias"),   { n_vocab }, 0);

                    for (int i = 0; i < n_layer; ++i) {
                        auto & layer = layers[i];

                        layer.attn_norm   = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), { n_embd }, 0);
                        layer.attn_norm_b = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "bias",   i), { n_embd }, 0);

                        layer.wqkv = create_tensor(tn(LLM_TENSOR_ATTN_QKV, "weight", i), { n_embd, n_embd + 2 * n_embd_gqa }, TENSOR_NOT_REQUIRED);
                        if (layer.wqkv == nullptr) {
                            layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q, "weight", i), {n_embd, n_embd}, 0);
                            layer.bq = create_tensor(tn(LLM_TENSOR_ATTN_Q, "bias",   i), {n_embd}, 0);

                            layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K, "weight", i), {n_embd, n_embd_gqa}, 0);
                            layer.bk = create_tensor(tn(LLM_TENSOR_ATTN_K, "bias",   i), {n_embd_gqa}, 0);

                            layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V, "weight", i), {n_embd, n_embd_gqa}, 0);
                            layer.bv = create_tensor(tn(LLM_TENSOR_ATTN_V, "bias",   i), {n_embd_gqa}, 0);
                        }
                        layer.wo   = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), { n_embd, n_embd }, 0);
                        layer.bo   = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "bias",   i), { n_embd }, 0);

                        layer.ffn_norm   = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), { n_embd }, 0);
                        layer.ffn_norm_b = create_tensor(tn(LLM_TENSOR_FFN_NORM, "bias",   i), { n_embd }, 0);

                        layer.ffn_gate_inp  = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,  "weight", i), {n_embd, n_expert},         0);
                        layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {n_embd, n_ff,   n_expert}, 0);
                        layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {n_ff,   n_embd, n_expert}, 0);
                        layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {n_embd, n_ff,   n_expert}, 0);

                        layer.rope_long  = create_tensor(tn(LLM_TENSOR_ROPE_FACTORS_LONG,  "weight", i), { n_embd_head/2 }, TENSOR_NOT_REQUIRED | (i != 0 ? TENSOR_DUPLICATED : 0));
                        layer.rope_short = create_tensor(tn(LLM_TENSOR_ROPE_FACTORS_SHORT, "weight", i), { n_embd_head/2 }, TENSOR_NOT_REQUIRED | (i != 0 ? TENSOR_DUPLICATED : 0));
                     }
                } break;
            case LLM_ARCH_PLAMO:
                {
                    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

                    // output
                    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
                    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, 0);

                    for (int i = 0; i < n_layer; ++i) {
                        auto & layer = layers[i];

                        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

                        layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_embd}, 0);
                        layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_embd_gqa}, 0);
                        layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_gqa}, 0);
                        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd}, 0);

                        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff}, 0);
                        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd}, 0);
                        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff}, 0);
                    }
                } break;
            case LLM_ARCH_PLAMO2:
                {
                    const uint32_t d_conv             = hparams.ssm_d_conv;
                    const uint32_t d_state            = hparams.ssm_d_state;
                    const uint32_t num_heads          = hparams.ssm_dt_rank;
                    const uint32_t intermediate_size  = hparams.ssm_d_inner;
                    const uint32_t head_dim           = intermediate_size / num_heads;
                    const uint32_t qk_dim             = head_dim;
                    const uint32_t v_dim              = head_dim;
                    const int64_t num_attention_heads = hparams.n_head();
                    const int64_t q_num_heads         = num_attention_heads;
                    const int64_t dt_dim              = std::max(64, int(hparams.n_embd / 16));

                    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

                    // output
                    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
                    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);
                    // if output is NULL, init from the input tok embed
                    if (output == NULL) {
                        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
                    }

                    for (int i = 0; i < n_layer; ++i) {
                        auto & layer = layers[i];
                        bool is_mamba_layer = hparams.is_recurrent(i);

                        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

                        if (is_mamba_layer) {
                            layer.ssm_in       = create_tensor(tn(LLM_TENSOR_SSM_IN,     "weight", i), {n_embd, 2 * intermediate_size}, 0);
                            layer.ssm_conv1d   = create_tensor(tn(LLM_TENSOR_SSM_CONV1D, "weight", i), {d_conv, intermediate_size}, 0);

                            layer.ssm_x    = create_tensor(tn(LLM_TENSOR_SSM_X,  "weight", i), {intermediate_size, dt_dim + 2*d_state}, 0);
                            layer.ssm_dt   = create_tensor(tn(LLM_TENSOR_SSM_DT, "weight", i), {dt_dim, num_heads}, 0);
                            layer.ssm_dt_b = create_tensor(tn(LLM_TENSOR_SSM_DT, "bias", i), {num_heads}, 0);

                            layer.ssm_a = create_tensor(tn(LLM_TENSOR_SSM_A, i), {num_heads}, 0);
                            layer.ssm_d = create_tensor(tn(LLM_TENSOR_SSM_D, i), {num_heads}, 0);

                            layer.ssm_out = create_tensor(tn(LLM_TENSOR_SSM_OUT, "weight", i), {intermediate_size, n_embd}, 0);

                            layer.ssm_dt_norm = create_tensor(tn(LLM_TENSOR_SSM_DT_NORM, i), {dt_dim}, 0);
                            layer.ssm_b_norm = create_tensor(tn(LLM_TENSOR_SSM_B_NORM, i), {d_state}, 0);
                            layer.ssm_c_norm = create_tensor(tn(LLM_TENSOR_SSM_C_NORM, i), {d_state}, 0);
                        } else {
                            const int64_t num_key_value_heads = hparams.n_head_kv(i);
                            const int64_t k_num_heads         = num_key_value_heads;
                            const int64_t v_num_heads         = num_key_value_heads;
                            const int64_t q_proj_dim          = q_num_heads * qk_dim;
                            const int64_t k_proj_dim          = k_num_heads * qk_dim;
                            const int64_t v_proj_dim          = v_num_heads * v_dim;

                            layer.wqkv = create_tensor(tn(LLM_TENSOR_ATTN_QKV, "weight", i), {n_embd, q_proj_dim + k_proj_dim + v_proj_dim}, 0);
                            layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), {head_dim, num_attention_heads}, 0);
                            layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), {head_dim, k_num_heads}, 0);
                            layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {q_num_heads * v_dim, n_embd}, 0);
                        }

                        // All layers have post-attention norm, FFN norm, and FFN tensors
                        layer.attn_post_norm = create_tensor(tn(LLM_TENSOR_ATTN_POST_NORM, i), {n_embd}, 0);
                        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);
                        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff, n_embd}, 0);
                        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd, n_ff * 2}, 0);
                        layer.ffn_post_norm = create_tensor(tn(LLM_TENSOR_FFN_POST_NORM, i), {n_embd}, 0);
                    }
                } break;
            case LLM_ARCH_GPT2:
                {
                    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);
                    pos_embd = create_tensor(tn(LLM_TENSOR_POS_EMBD,   "weight"), {n_embd, n_ctx_train}, 0);

                    // output
                    output_norm   = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
                    output_norm_b = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "bias"),   {n_embd}, 0);
                    output        = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);

                    // if output is NULL, init from the input tok embed
                    if (output == NULL) {
                        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
                    }

                    for (int i = 0; i < n_layer; ++i) {
                        auto & layer = layers[i];

                        layer.attn_norm   = create_tensor(tn(LLM_TENSOR_ATTN_NORM,   "weight", i), {n_embd}, 0);
                        layer.attn_norm_b = create_tensor(tn(LLM_TENSOR_ATTN_NORM,   "bias", i),   {n_embd}, 0);

                        layer.wqkv = create_tensor(tn(LLM_TENSOR_ATTN_QKV, "weight", i), {n_embd, n_embd + 2*n_embd_gqa}, 0);
                        layer.bqkv = create_tensor(tn(LLM_TENSOR_ATTN_QKV, "bias", i),   {n_embd + 2*n_embd_gqa}, 0);

                        layer.wo   = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd}, 0);
                        layer.bo   = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "bias", i),   {n_embd}, 0);

                        layer.ffn_norm   = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);
                        layer.ffn_norm_b = create_tensor(tn(LLM_TENSOR_FFN_NORM, "bias", i),   {n_embd}, 0);

                        layer.ffn_down   = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff, n_embd}, 0);
                        layer.ffn_down_b = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "bias", i),   {n_embd}, 0);

                        layer.ffn_up     = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd, n_ff}, 0);
                        layer.ffn_up_b   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "bias", i),   {n_ff}, 0);
                    }
                } break;
            case LLM_ARCH_CODESHELL:
                {
                    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);

                    // if tok embd is NULL, init from output
                    if (tok_embd == NULL) {
                        tok_embd = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
                    }

                    // output
                    output_norm   = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
                    output_norm_b = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "bias"),   {n_embd}, 0);
                    output        = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, 0);

                    for (int i = 0; i < n_layer; ++i) {
                        auto & layer = layers[i];

                        layer.attn_norm   = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);
                        layer.attn_norm_b = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "bias", i),   {n_embd}, 0);

                        layer.wqkv = create_tensor(tn(LLM_TENSOR_ATTN_QKV, "weight", i), {n_embd, n_embd + 2*n_embd_gqa}, 0);
                        layer.bqkv = create_tensor(tn(LLM_TENSOR_ATTN_QKV, "bias", i),   {n_embd + 2*n_embd_gqa}, 0);

                        layer.wo   = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd}, 0);
                        layer.bo   = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "bias", i),   {n_embd}, 0);

                        layer.ffn_norm   = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);
                        layer.ffn_norm_b = create_tensor(tn(LLM_TENSOR_FFN_NORM, "bias", i),   {n_embd}, 0);

                        layer.ffn_down   = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff, n_embd}, 0);
                        layer.ffn_down_b = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "bias", i),   {n_embd}, 0);

                        layer.ffn_up     = create_tensor(tn(LLM_TENSOR_FFN_UP, "weight", i),   {n_embd, n_ff}, 0);
                        layer.ffn_up_b   = create_tensor(tn(LLM_TENSOR_FFN_UP, "bias", i),     {n_ff}, 0);
                    }
                } break;
            case LLM_ARCH_ORION:
                {
                    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

                    output_norm   = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
                    output_norm_b = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "bias"),   {n_embd}, 0);
                    output        = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, 0);

                    for (int i = 0; i < n_layer; ++i) {
                        auto & layer = layers[i];

                        layer.attn_norm   = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);
                        layer.attn_norm_b = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "bias", i),   {n_embd}, 0);

                        layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_embd}, 0);
                        layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_embd_gqa}, 0);
                        layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_gqa}, 0);
                        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd}, 0);

                        layer.ffn_norm   = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);
                        layer.ffn_norm_b = create_tensor(tn(LLM_TENSOR_FFN_NORM, "bias", i),   {n_embd}, 0);

                        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff}, 0);
                        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd}, 0);
                        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff}, 0);
                    }
                } break;
            case LLM_ARCH_INTERNLM2:
                {
                    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

                    // output
                    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
                    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, 0);

                    for (int i = 0; i < n_layer; ++i) {
                        auto & layer = layers[i];

                        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);
                        // layer.wqkv = create_tensor(tn(LLM_TENSOR_ATTN_QKV, "weight", i), {n_embd, n_embd + 2*n_embd_gqa}, 0);
                        layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_embd}, 0);
                        layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_embd_gqa}, 0);
                        layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_gqa}, 0);

                        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd}, 0);
                        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);
                        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff}, 0);
                        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd}, 0);
                        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff}, 0);
                    }
                } break;
            case LLM_ARCH_GEMMA:
                {
                    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

                    // output
                    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
                    output      = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD,  "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED); // same as tok_embd, duplicated to allow offloading

                    for (int i = 0; i < n_layer; ++i) {
                        auto & layer = layers[i];

                        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

                        layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_embd_head_k * n_head}, 0);
                        layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_embd_k_gqa}, 0);
                        layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_v_gqa}, 0);
                        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head_k * n_head, n_embd}, 0);

                        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);
                        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff}, 0);
                        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff}, 0);
                        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd}, 0);
                    }
                } break;
            case LLM_ARCH_GEMMA2:
                {
                    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

                    // output
                    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
                    output      = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD,  "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED); // same as tok_embd, duplicated to allow offloading

                    for (int i = 0; i < n_layer; ++i) {
                        auto & layer = layers[i];

                        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

                        layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_embd_head_k * n_head}, 0);
                        layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_embd_k_gqa}, 0);
                        layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_v_gqa}, 0);
                        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head_k * n_head, n_embd}, 0);
                        layer.attn_post_norm = create_tensor(tn(LLM_TENSOR_ATTN_POST_NORM, "weight", i), {n_embd}, 0);

                        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);
                        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff}, 0);
                        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff}, 0);
                        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd}, 0);
                        layer.ffn_post_norm = create_tensor(tn(LLM_TENSOR_FFN_POST_NORM, "weight", i), {n_embd}, 0);
                    }
                } break;
            case LLM_ARCH_GEMMA3:
                {
                    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

                    // output
                    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
                    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);

                    // if output is NULL, init from the input tok embed
                    if (output == NULL) {
                        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD,   "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
                    }

                    for (int i = 0; i < n_layer; ++i) {
                        auto & layer = layers[i];

                        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

                        layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_embd_head_k * n_head}, 0);
                        layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_embd_k_gqa}, 0);
                        layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_v_gqa}, 0);
                        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head_k * n_head, n_embd}, 0);

                        layer.attn_post_norm = create_tensor(tn(LLM_TENSOR_ATTN_POST_NORM, "weight", i), {n_embd}, 0);
                        layer.attn_k_norm    = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM,    "weight", i), {n_embd_head_k}, 0);
                        layer.attn_q_norm    = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM,    "weight", i), {n_embd_head_k}, 0);

                        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);
                        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff}, 0);
                        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff}, 0);
                        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd}, 0);
                        layer.ffn_post_norm = create_tensor(tn(LLM_TENSOR_FFN_POST_NORM, "weight", i), {n_embd}, 0);
                    }
                } break;
            case LLM_ARCH_GEMMA3N:
                {
                    const int64_t n_altup      = hparams.n_altup;
                    const int64_t laurel_rank  = hparams.laurel_rank;
                    const int64_t n_embd_altup = hparams.n_embd_altup;

                    output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);
                    // if output is NULL, init from the input tok embed
                    if (output == NULL) {
                        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
                    }

                    tok_embd           = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD,           "weight"), {n_embd, n_vocab}, 0);
                    tok_embd_per_layer = create_tensor(tn(LLM_TENSOR_PER_LAYER_TOKEN_EMBD, "weight"), {n_embd_altup * n_layer, n_vocab}, 0);

                    altup_proj           = create_tensor(tn(LLM_TENSOR_ALTUP_PROJ,           "weight"), {n_embd, n_embd, n_altup - 1}, 0);
                    altup_unembd_proj    = create_tensor(tn(LLM_TENSOR_ALTUP_UNEMBD_PROJ,    "weight"), {n_embd, n_embd, n_altup - 1}, 0);
                    per_layer_model_proj = create_tensor(tn(LLM_TENSOR_PER_LAYER_MODEL_PROJ, "weight"), {n_embd, n_embd_altup * n_layer}, 0);
                    per_layer_proj_norm  = create_tensor(tn(LLM_TENSOR_PER_LAYER_PROJ_NORM,  "weight"), {n_embd_altup}, 0);

                    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);

                    for (int i = 0; i < n_layer; ++i) {
                        auto & layer = layers[i];

                        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

                        layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_embd_head_k * n_head}, 0);
                        layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_embd_k_gqa}, 0);
                        layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_v_gqa}, 0);
                        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head_k * n_head, n_embd}, 0);

                        layer.attn_q_norm    = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM,    "weight", i), {n_embd_head_k}, 0);
                        layer.attn_k_norm    = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM,    "weight", i), {n_embd_head_k}, 0);
                        layer.attn_post_norm = create_tensor(tn(LLM_TENSOR_ATTN_POST_NORM, "weight", i), {n_embd}, 0);

                        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);
                        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff}, 0);
                        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff}, 0);
                        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd}, 0);
                        layer.ffn_post_norm = create_tensor(tn(LLM_TENSOR_FFN_POST_NORM, "weight", i), {n_embd}, 0);

                        // altup & laurel
                        layer.per_layer_inp_gate   = create_tensor(tn(LLM_TENSOR_PER_LAYER_INP_GATE,  "weight", i), {n_embd, n_embd_altup}, 0);
                        layer.per_layer_proj       = create_tensor(tn(LLM_TENSOR_PER_LAYER_PROJ,      "weight", i), {n_embd_altup, n_embd}, 0);
                        layer.per_layer_post_norm  = create_tensor(tn(LLM_TENSOR_PER_LAYER_POST_NORM, "weight", i), {n_embd}, 0);
                        layer.altup_correct_coef   = create_tensor(tn(LLM_TENSOR_ALTUP_CORRECT_COEF,  "weight", i), {n_altup, n_altup}, 0);
                        layer.altup_correct_scale  = create_tensor(tn(LLM_TENSOR_ALTUP_CORRECT_SCALE, "weight", i), {n_embd}, 0);
                        layer.altup_predict_coef   = create_tensor(tn(LLM_TENSOR_ALTUP_PREDICT_COEF,  "weight", i), {n_altup, n_altup * n_altup}, 0);
                        layer.altup_router         = create_tensor(tn(LLM_TENSOR_ALTUP_ROUTER,        "weight", i), {n_embd, n_altup}, 0);
                        layer.altup_router_norm    = create_tensor(tn(LLM_TENSOR_ALTUP_ROUTER_NORM,   "weight", i), {n_embd}, 0);
                        layer.laurel_l             = create_tensor(tn(LLM_TENSOR_LAUREL_L,            "weight", i), {n_embd, laurel_rank}, 0);
                        layer.laurel_r             = create_tensor(tn(LLM_TENSOR_LAUREL_R,            "weight", i), {laurel_rank, n_embd}, 0);
                        layer.laurel_post_norm     = create_tensor(tn(LLM_TENSOR_LAUREL_POST_NORM,    "weight", i), {n_embd}, 0);
                    }
                } break;
            case LLM_ARCH_STARCODER2:
                {
                    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

                    // output
                    output_norm   = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
                    output_norm_b = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "bias"),   {n_embd}, 0);

                    output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);
                    // if output is NULL, init from the input tok embed
                    if (output == NULL) {
                        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
                    }

                    for (int i = 0; i < n_layer; ++i) {
                        auto & layer = layers[i];

                        layer.attn_norm   = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);
                        layer.attn_norm_b = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "bias", i),   {n_embd}, 0);

                        layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_embd}, 0);
                        layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_embd_gqa}, 0);
                        layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_gqa}, 0);
                        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd}, 0);

                        // optional bias tensors
                        layer.bq = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "bias", i), {n_embd}, 0);
                        layer.bk = create_tensor(tn(LLM_TENSOR_ATTN_K,   "bias", i), {n_embd_gqa}, 0);
                        layer.bv = create_tensor(tn(LLM_TENSOR_ATTN_V,   "bias", i), {n_embd_gqa}, 0);
                        layer.bo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "bias", i), {n_embd}, 0);

                        layer.ffn_norm   = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);
                        layer.ffn_norm_b = create_tensor(tn(LLM_TENSOR_FFN_NORM, "bias", i),   {n_embd}, 0);

                        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd}, 0);
                        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff}, 0);

                        // optional bias tensors
                        layer.ffn_down_b = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "bias", i), {n_embd}, 0);
                        layer.ffn_up_b   = create_tensor(tn(LLM_TENSOR_FFN_UP ,  "bias", i), {  n_ff}, 0);
                    }
                } break;
            case LLM_ARCH_MAMBA:
                {
                    const int64_t d_conv  = hparams.ssm_d_conv;
                    const int64_t d_inner = hparams.ssm_d_inner;
                    const int64_t d_state = hparams.ssm_d_state;
                    const int64_t dt_rank = hparams.ssm_dt_rank;

                    // only an expansion factor of 2 is supported for now
                    if (2 * n_embd != d_inner) {
                        throw std::runtime_error("only an expansion factor of 2 is supported for now");
                    }

                    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

                    // output
                    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);

                    output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);
                    // if output is NULL, init from the input tok embed, duplicated to allow offloading
                    if (output == NULL) {
                        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
                    }

                    for (int i = 0; i < n_layer; ++i) {
                        auto & layer = layers[i];

                        // norm
                        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

                        layer.ssm_in = create_tensor(tn(LLM_TENSOR_SSM_IN, "weight", i), {n_embd, 2*d_inner}, 0);

                        layer.ssm_conv1d = create_tensor(tn(LLM_TENSOR_SSM_CONV1D, "weight", i), {d_conv, d_inner}, 0);
                        layer.ssm_conv1d_b = create_tensor(tn(LLM_TENSOR_SSM_CONV1D, "bias", i), {d_inner}, 0);

                        layer.ssm_x = create_tensor(tn(LLM_TENSOR_SSM_X, "weight", i), {d_inner, dt_rank + 2*d_state}, 0);

                        layer.ssm_dt = create_tensor(tn(LLM_TENSOR_SSM_DT, "weight", i), {dt_rank, d_inner}, 0);
                        layer.ssm_dt_b = create_tensor(tn(LLM_TENSOR_SSM_DT, "bias", i), {d_inner}, 0);

                        // no "weight" suffix for these
                        layer.ssm_a = create_tensor(tn(LLM_TENSOR_SSM_A, i), {d_state, d_inner}, 0);
                        layer.ssm_d = create_tensor(tn(LLM_TENSOR_SSM_D, i), {d_inner}, 0);

                        // out_proj
                        layer.ssm_out = create_tensor(tn(LLM_TENSOR_SSM_OUT, "weight", i), {d_inner, n_embd}, 0);
                    }
                } break;
            case LLM_ARCH_MAMBA2:
                {
                    const int64_t d_conv  = hparams.ssm_d_conv;
                    const int64_t d_inner = hparams.ssm_d_inner;
                    const int64_t d_state = hparams.ssm_d_state;
                    const int64_t n_head  = hparams.ssm_dt_rank;
                    const int64_t n_group = hparams.ssm_n_group;
                    const int64_t d_in_proj = 2*d_inner + 2*n_group*d_state + n_head;

                    // only an expansion factor of 2 is supported for now
                    GGML_ASSERT(2 * n_embd == d_inner);

                    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

                    // output
                    {
                        output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);

                        output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);
                        // if output is NULL, init from the input tok embed, duplicated to allow offloading
                        if (output == NULL) {
                            output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
                        }
                    }

                    for (int i = 0; i < n_layer; ++i) {
                        auto & layer = layers[i];

                        // norm
                        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

                        layer.ssm_in = create_tensor(tn(LLM_TENSOR_SSM_IN, "weight", i), {n_embd, d_in_proj}, 0);

                        layer.ssm_conv1d = create_tensor(tn(LLM_TENSOR_SSM_CONV1D, "weight", i), {d_conv, d_inner + 2*n_group*d_state}, 0);
                        layer.ssm_conv1d_b = create_tensor(tn(LLM_TENSOR_SSM_CONV1D, "bias", i), {d_inner + 2*n_group*d_state}, 0);

                        layer.ssm_dt_b = create_tensor(tn(LLM_TENSOR_SSM_DT, "bias", i), {n_head}, 0);

                        // no "weight" suffix for these
                        layer.ssm_a = create_tensor(tn(LLM_TENSOR_SSM_A, i), {1, n_head}, 0);
                        layer.ssm_d = create_tensor(tn(LLM_TENSOR_SSM_D, i), {1, n_head}, 0);

                        layer.ssm_norm = create_tensor(tn(LLM_TENSOR_SSM_NORM, "weight", i), {d_inner / n_group, n_group}, 0);

                        // out_proj
                        layer.ssm_out = create_tensor(tn(LLM_TENSOR_SSM_OUT, "weight", i), {d_inner, n_embd}, 0);
                    }
                } break;
            case LLM_ARCH_JAMBA:
                {
                    const int64_t d_conv  = hparams.ssm_d_conv;
                    const int64_t d_inner = hparams.ssm_d_inner;
                    const int64_t d_state = hparams.ssm_d_state;
                    const int64_t dt_rank = hparams.ssm_dt_rank;

                    // only an expansion factor of 2 is supported for now
                    GGML_ASSERT(2 * n_embd == d_inner);

                    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

                    // output
                    {
                        output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);

                        output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);
                        // if output is NULL, init from the input tok embed, duplicated to allow offloading
                        if (output == NULL) {
                            output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
                        }
                    }

                    for (int i = 0; i < n_layer; ++i) {
                        const int64_t n_head_kv = hparams.n_head_kv(i);
                        const int64_t n_embd_gqa = hparams.n_embd_v_gqa(i);

                        auto & layer = layers[i];

                        // norm
                        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

                        if (n_head_kv == 0) {
                            // Mamba layer
                            layer.ssm_in = create_tensor(tn(LLM_TENSOR_SSM_IN, "weight", i), {n_embd, 2*d_inner}, 0);

                            layer.ssm_conv1d = create_tensor(tn(LLM_TENSOR_SSM_CONV1D, "weight", i), {d_conv, d_inner}, 0);
                            layer.ssm_conv1d_b = create_tensor(tn(LLM_TENSOR_SSM_CONV1D, "bias", i), {d_inner}, 0);

                            layer.ssm_x = create_tensor(tn(LLM_TENSOR_SSM_X, "weight", i), {d_inner, dt_rank + 2*d_state}, 0);

                            layer.ssm_dt_norm = create_tensor(tn(LLM_TENSOR_SSM_DT_NORM, "weight", i), {dt_rank}, 0);

                            layer.ssm_dt = create_tensor(tn(LLM_TENSOR_SSM_DT, "weight", i), {dt_rank, d_inner}, 0);
                            layer.ssm_dt_b = create_tensor(tn(LLM_TENSOR_SSM_DT, "bias", i), {d_inner}, 0);

                            layer.ssm_b_norm = create_tensor(tn(LLM_TENSOR_SSM_B_NORM, "weight", i), {d_state}, 0);
                            layer.ssm_c_norm = create_tensor(tn(LLM_TENSOR_SSM_C_NORM, "weight", i), {d_state}, 0);

                            // no "weight" suffix for these
                            layer.ssm_a = create_tensor(tn(LLM_TENSOR_SSM_A, i), {d_state, d_inner}, 0);
                            layer.ssm_d = create_tensor(tn(LLM_TENSOR_SSM_D, i), {d_inner}, 0);

                            // out_proj
                            layer.ssm_out = create_tensor(tn(LLM_TENSOR_SSM_OUT, "weight", i), {d_inner, n_embd}, 0);
                        } else {
                            // Attention layers

                            layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_embd}, 0);
                            layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_embd_gqa}, 0);
                            layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_gqa}, 0);
                            layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd}, 0);
                        }

                        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);

                        layer.ffn_gate_inp = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP, "weight", i), {n_embd, n_expert}, TENSOR_NOT_REQUIRED);

                        if (layer.ffn_gate_inp) {
                            // MoE
                            layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {n_embd, n_ff, n_expert}, 0);
                            layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {n_ff, n_embd, n_expert}, 0);
                            layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {n_embd, n_ff, n_expert}, 0);
                        } else {
                            // FFN (no MoE)
                            layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd, n_ff}, 0);
                            layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff, n_embd}, 0);
                            layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd, n_ff}, 0);
                        }
                    }
                } break;
            case LLM_ARCH_GRANITE_HYBRID:
                {
                    // mamba2 Mixer SSM params
                    // NOTE: int64_t for tensor dimensions
                    const int64_t d_conv     = hparams.ssm_d_conv;
                    const int64_t d_inner    = hparams.ssm_d_inner;
                    const int64_t d_state    = hparams.ssm_d_state;
                    const int64_t n_ssm_head = hparams.ssm_dt_rank;
                    const int64_t n_group    = hparams.ssm_n_group;
                    const int64_t d_in_proj  = 2*d_inner + 2*n_group*d_state + n_ssm_head;

                    // only an expansion factor of 2 is supported for now
                    GGML_ASSERT(2 * n_embd == d_inner);

                    // embeddings
                    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

                    // output
                    {
                        output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
                        output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);
                        // if output is NULL, init from the input tok embed, duplicated to allow offloading
                        if (output == NULL) {
                            output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
                        }
                    }

                    for (int i = 0; i < n_layer; ++i) {
                        auto & layer = layers[i];

                        // norm
                        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

                        if (hparams.is_recurrent(i)) {
                            // ssm layers
                            layer.ssm_in = create_tensor(tn(LLM_TENSOR_SSM_IN, "weight", i), {n_embd, d_in_proj}, 0);

                            layer.ssm_conv1d = create_tensor(tn(LLM_TENSOR_SSM_CONV1D, "weight", i), {d_conv, d_inner + 2*n_group*d_state}, 0);
                            layer.ssm_conv1d_b = create_tensor(tn(LLM_TENSOR_SSM_CONV1D, "bias", i), {d_inner + 2*n_group*d_state}, TENSOR_NOT_REQUIRED);

                            layer.ssm_dt_b = create_tensor(tn(LLM_TENSOR_SSM_DT, "bias", i), {n_ssm_head}, 0);

                            // no "weight" suffix for these
                            layer.ssm_a = create_tensor(tn(LLM_TENSOR_SSM_A, i), {1, n_ssm_head}, 0);
                            layer.ssm_d = create_tensor(tn(LLM_TENSOR_SSM_D, i), {1, n_ssm_head}, 0);

                            layer.ssm_norm = create_tensor(tn(LLM_TENSOR_SSM_NORM, "weight", i), {d_inner / n_group, n_group}, 0);

                            // out_proj
                            layer.ssm_out = create_tensor(tn(LLM_TENSOR_SSM_OUT, "weight", i), {d_inner, n_embd}, 0);
                        } else {
                            // attention layers (with optional bias)
                            const int64_t n_head_i = hparams.n_head(i);
                            const int64_t n_embd_k_gqa_i = hparams.n_embd_k_gqa(i);
                            const int64_t n_embd_v_gqa_i = hparams.n_embd_v_gqa(i);
                            layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_embd_head_k * n_head_i}, 0);
                            layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_embd_k_gqa_i}, 0);
                            layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_v_gqa_i}, 0);
                            layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head_k * n_head_i, n_embd}, 0);
                            layer.bq = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "bias", i), {n_embd},         TENSOR_NOT_REQUIRED);
                            layer.bk = create_tensor(tn(LLM_TENSOR_ATTN_K,   "bias", i), {n_embd_k_gqa_i}, TENSOR_NOT_REQUIRED);
                            layer.bv = create_tensor(tn(LLM_TENSOR_ATTN_V,   "bias", i), {n_embd_v_gqa_i}, TENSOR_NOT_REQUIRED);
                            layer.bo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "bias", i), {n_embd},         TENSOR_NOT_REQUIRED);
                        }

                        // feed forward (w/ optional biases)
                        if (n_expert > 0) {
                            // MoE FFN
                            layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);
                            layer.rope_freqs = create_tensor(tn(LLM_TENSOR_ROPE_FREQS, "weight", i), {n_rot/2}, TENSOR_NOT_REQUIRED | (i != 0 ? TENSOR_DUPLICATED : 0));
                            layer.ffn_gate_inp  = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,  "weight", i), {n_embd, n_expert}, 0);
                            layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {n_embd,   n_ff, n_expert}, TENSOR_NOT_REQUIRED);
                            layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {  n_ff, n_embd, n_expert}, 0);
                            layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {n_embd,   n_ff, n_expert}, 0);

                            // For Granite MoE Shared
                            if (hparams.n_ff_shexp > 0) {
                                layer.ffn_gate_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", i), {n_embd, hparams.n_ff_shexp}, 0);
                                layer.ffn_up_shexp   = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,   "weight", i), {n_embd, hparams.n_ff_shexp}, 0);
                                layer.ffn_down_shexp = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i), {hparams.n_ff_shexp, n_embd}, 0);
                            }
                        } else {
                            layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);
                            layer.rope_freqs = create_tensor(tn(LLM_TENSOR_ROPE_FREQS, "weight", i), {n_rot/2}, TENSOR_NOT_REQUIRED | (i != 0 ? TENSOR_DUPLICATED : 0));
                            layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff}, 0);
                            layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd}, 0);
                            layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff}, 0);
                            layer.ffn_gate_b = create_tensor(tn(LLM_TENSOR_FFN_GATE, "bias", i), {n_ff}, TENSOR_NOT_REQUIRED);
                            layer.ffn_down_b = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "bias", i), {n_embd}, TENSOR_NOT_REQUIRED);
                            layer.ffn_up_b   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "bias", i), {n_ff}, TENSOR_NOT_REQUIRED);
                        }
                    }
                } break;
            case LLM_ARCH_XVERSE:
                {
                    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

                    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
                    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, 0);

                    for (int i = 0; i < n_layer; ++i) {
                        auto & layer = layers[i];

                        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

                        layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_embd}, 0);
                        layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_embd_gqa}, 0);
                        layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_gqa}, 0);
                        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd}, 0);

                        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);
                        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff}, 0);
                        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd}, 0);
                        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff}, 0);
                    }
                } break;
            case LLM_ARCH_COMMAND_R:
                {
                    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

                    // output
                    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
                    // init output from the input tok embed
                    output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);

                    for (int i = 0; i < n_layer; ++i) {
                        auto & layer = layers[i];

                        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

                        if (n_layer >= 64){
                            layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), {n_embd_head_k, n_head}, 0);
                            layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), {n_embd_head_k, n_head_kv}, 0);
                        }

                        layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_embd}, 0);
                        layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_embd_gqa}, 0);
                        layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_gqa}, 0);
                        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd}, 0);

                        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff}, 0);
                        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd}, 0);
                        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff}, 0);
                    }
                } break;
            case LLM_ARCH_COHERE2:
                {
                    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, 0);

                    // output
                    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), { n_embd }, 0);
                    // init output from the input tok embed
                    output      = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab },
                                                      TENSOR_DUPLICATED);

                    for (int i = 0; i < n_layer; ++i) {
                        auto & layer = layers[i];

                        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), { n_embd }, 0);

                        layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q, "weight", i), { n_embd, n_embd }, 0);
                        layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K, "weight", i), { n_embd, n_embd_gqa }, 0);
                        layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V, "weight", i), { n_embd, n_embd_gqa }, 0);
                        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), { n_embd, n_embd }, 0);

                        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), { n_embd, n_ff }, 0);
                        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), { n_ff, n_embd }, 0);
                        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP, "weight", i), { n_embd, n_ff }, 0);
                    }
                }
                break;
            case LLM_ARCH_OLMO:  // adapted from LLM_ARCH_LLAMA with norm params removed
                {
                    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

                    // output
                    output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);
                    // if output is NULL, init from the input tok embed
                    if (output == NULL) {
                        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
                    }

                    for (int i = 0; i < n_layer; ++i) {
                        auto & layer = layers[i];

                        layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_embd}, 0);
                        layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_embd_gqa}, 0);
                        layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_gqa}, 0);
                        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd}, 0);

                        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff}, 0);
                        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd}, 0);
                        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff}, 0);
                    }
                } break;
            case LLM_ARCH_OLMO2:
                {
                    const int64_t n_embd_head = n_embd / n_head;

                    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

                    // output
                    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
                    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, 0);

                    for (int i = 0; i < n_layer; ++i) {
                        auto & layer = layers[i];

                        layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_embd}, 0);
                        layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_embd_gqa}, 0);
                        layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_gqa}, 0);
                        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd}, 0);
                        layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), {n_embd}, 0);
                        layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), {n_head_kv * n_embd_head}, 0);
                        layer.attn_post_norm = create_tensor(tn(LLM_TENSOR_ATTN_POST_NORM, "weight", i), {n_embd}, 0);

                        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff}, 0);
                        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff}, 0);
                        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd}, 0);
                        layer.ffn_post_norm = create_tensor(tn(LLM_TENSOR_FFN_POST_NORM, "weight", i), {n_embd}, 0);
                    }
                } break;
            case LLM_ARCH_OLMOE:
                {
                    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

                    // output
                    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
                    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, 0);

                    for (int i = 0; i < n_layer; ++i) {
                        auto & layer = layers[i];

                        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

                        layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_embd}, 0);
                        layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_embd_gqa}, 0);
                        layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_gqa}, 0);
                        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd}, 0);
                        layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), {n_embd}, 0);
                        layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), {n_embd}, 0);

                        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);

                        layer.ffn_gate_inp = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP, "weight", i), {n_embd, n_expert}, 0);

                        if (n_expert == 0) {
                            throw std::runtime_error("n_expert must be > 0");
                        }
                        if (n_expert_used == 0) {
                            throw std::runtime_error("n_expert_used must be > 0");
                        }

                        // MoE branch
                        layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {n_embd, n_ff,   n_expert}, 0);
                        layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {n_ff,   n_embd, n_expert}, 0);
                        layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {n_embd, n_ff,   n_expert}, 0);
                    }
                } break;
            case LLM_ARCH_OPENELM:
                {
                    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

                    // output
                    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
                    // init output from the input tok embed
                    output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);

                    for (int i = 0; i < n_layer; ++i) {
                        const int64_t n_head      =   hparams.n_head(i);
                        const int64_t n_head_qkv  = 2*hparams.n_head_kv(i) + n_head;
                        const int64_t n_ff        =   hparams.n_ff(i);

                        auto & layer = layers[i];

                        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

                        layer.wqkv = create_tensor(tn(LLM_TENSOR_ATTN_QKV, "weight", i), {n_embd, n_head_qkv*n_embd_head_k}, 0);
                        layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), {n_embd_head_k}, 0);
                        layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), {n_embd_head_k}, 0);
                        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_head*n_embd_head_k, n_embd}, 0);

                        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);
                        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd, n_ff}, 0);
                        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff, n_embd}, 0);
                        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd, n_ff}, 0);
                    }
                } break;
            case LLM_ARCH_GPTNEOX:
                {
                    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

                    // output
                    output_norm   = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
                    output_norm_b = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "bias"),   {n_embd}, 0);
                    output        = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, 0);

                    for (int i = 0; i < n_layer; ++i) {
                        auto & layer = layers[i];

                        layer.attn_norm   = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);
                        layer.attn_norm_b = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "bias", i),   {n_embd}, 0);

                        layer.wqkv = create_tensor(tn(LLM_TENSOR_ATTN_QKV, "weight", i), {n_embd, n_embd + 2*n_embd_gqa}, 0);
                        layer.bqkv = create_tensor(tn(LLM_TENSOR_ATTN_QKV, "bias", i),   {n_embd + 2*n_embd_gqa}, 0);

                        layer.wo   = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd}, 0);
                        layer.bo   = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "bias", i),   {n_embd}, 0);

                        layer.ffn_norm   = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);
                        layer.ffn_norm_b = create_tensor(tn(LLM_TENSOR_FFN_NORM, "bias", i),   {n_embd}, 0);

                        layer.ffn_down   = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff, n_embd}, 0);
                        layer.ffn_down_b = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "bias", i),   {n_embd}, 0);

                        layer.ffn_up     = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd, n_ff}, 0);
                        layer.ffn_up_b   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "bias", i),   {n_ff}, 0);
                    }
                } break;
            case LLM_ARCH_ARCTIC:
                {
                    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

                    // output
                    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
                    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);

                    // if output is NULL, init from the input tok embed
                    if (output == NULL) {
                        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
                    }

                    for (int i = 0; i < n_layer; ++i) {
                        auto & layer = layers[i];

                        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

                        layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_embd}, 0);
                        layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_embd_gqa}, 0);
                        layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_gqa}, 0);
                        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd}, 0);

                        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);

                        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd, n_embd}, 0);
                        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_embd, n_embd}, 0);
                        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd, n_embd}, 0);

                        layer.ffn_gate_inp = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP, "weight", i), {n_embd, n_expert}, 0);
                        layer.ffn_norm_exps = create_tensor(tn(LLM_TENSOR_FFN_NORM_EXPS, "weight", i), {n_embd}, 0);
                        layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {n_embd,   n_ff, n_expert}, false);
                        layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {  n_ff, n_embd, n_expert}, 0);
                        layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {n_embd,   n_ff, n_expert}, 0);
                    }
                } break;
            case LLM_ARCH_DEEPSEEK:
                {

                    const int64_t n_ff_exp        = hparams.n_ff_exp;
                    const int64_t n_expert_shared = hparams.n_expert_shared;

                    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

                    // output
                    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
                    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, 0);

                    for (int i = 0; i < n_layer; ++i) {
                        auto & layer = layers[i];

                        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

                        layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_embd}, 0);
                        layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_embd_gqa}, 0);
                        layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_gqa}, 0);
                        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd}, 0);
                        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);

                        if (i < (int) hparams.n_layer_dense_lead) {
                            layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff}, 0);
                            layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd}, 0);
                            layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff}, 0);
                        } else {
                            layer.ffn_gate_inp = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP, "weight", i), {n_embd, n_expert}, 0);

                            if (n_expert == 0) {
                                throw std::runtime_error("n_expert must be > 0");
                            }
                            if (n_expert_used == 0) {
                                throw std::runtime_error("n_expert_used must be > 0");
                            }

                            // MoE branch
                            layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {  n_embd, n_ff_exp, n_expert}, 0);
                            layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {n_ff_exp,   n_embd, n_expert}, 0);
                            layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {  n_embd, n_ff_exp, n_expert}, 0);

                            // Shared expert branch
                            layer.ffn_gate_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", i), {n_embd, n_ff_exp * n_expert_shared}, 0);
                            layer.ffn_down_shexp = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i), {        n_ff_exp * n_expert_shared, n_embd}, 0);
                            layer.ffn_up_shexp   = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,   "weight", i), {n_embd, n_ff_exp * n_expert_shared}, 0);
                        }
                    }
                } break;
            case LLM_ARCH_DEEPSEEK2:
                {
                    const bool is_lite = (hparams.n_layer == 27);

                    const bool is_mla = (hparams.n_embd_head_k_mla != 0 && hparams.n_embd_head_v_mla != 0);

                    // note: these are the actual head sizes you get when treating as MHA or after "decompression" using wv_b for MLA
                    const int64_t n_embd_head_k_mla = is_mla ? hparams.n_embd_head_k_mla : hparams.n_embd_head_k;
                    const int64_t n_embd_head_v_mla = is_mla ? hparams.n_embd_head_v_mla : hparams.n_embd_head_v;

                    const int64_t n_embd_head_qk_rope = hparams.n_rot;
                    const int64_t n_embd_head_qk_nope = n_embd_head_k_mla - n_embd_head_qk_rope;

                    const int64_t q_lora_rank  = hparams.n_lora_q;
                    const int64_t kv_lora_rank = hparams.n_lora_kv;

                    const int64_t n_ff_exp        = hparams.n_ff_exp;
                    const int64_t n_expert_shared = hparams.n_expert_shared;

                    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

                    // output
                    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
                    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, 0);

                    for (int i = 0; i < n_layer; ++i) {
                        auto & layer = layers[i];

                        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);
                        if (!is_lite) {
                            layer.attn_q_a_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_A_NORM, "weight", i), {q_lora_rank}, 0);
                        }

                        layer.attn_kv_a_norm = create_tensor(tn(LLM_TENSOR_ATTN_KV_A_NORM, "weight", i), {kv_lora_rank}, 0);

                        if (!is_lite) {
                            layer.wq_a = create_tensor(tn(LLM_TENSOR_ATTN_Q_A, "weight", i), {n_embd, q_lora_rank}, 0);
                            layer.wq_b = create_tensor(tn(LLM_TENSOR_ATTN_Q_B, "weight", i), {q_lora_rank, n_head * n_embd_head_k_mla}, 0);
                        } else {
                            layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q, "weight", i), {n_embd, n_head * n_embd_head_k_mla}, 0);
                        }

                        layer.wkv_a_mqa = create_tensor(tn(LLM_TENSOR_ATTN_KV_A_MQA, "weight", i), {n_embd, kv_lora_rank + n_embd_head_qk_rope}, 0);

                        // note: only old legacy GGUF files will have the unsplit wkv_b tensor in
                        if (is_mla) {
                            layer.wk_b = create_tensor(tn(LLM_TENSOR_ATTN_K_B, "weight", i), {n_embd_head_qk_nope, kv_lora_rank, n_head}, 0);
                            layer.wv_b = create_tensor(tn(LLM_TENSOR_ATTN_V_B, "weight", i), {kv_lora_rank, n_embd_head_v_mla, n_head}, 0);
                        } else {
                            layer.wkv_b = create_tensor(tn(LLM_TENSOR_ATTN_KV_B, "weight", i), {kv_lora_rank, n_head * (n_embd_head_qk_nope + n_embd_head_v_mla)}, 0);
                        }

                        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_head * n_embd_head_v_mla, n_embd}, 0);

                        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);

                        if (i < (int) hparams.n_layer_dense_lead) {
                            layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff}, 0);
                            layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd}, 0);
                            layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff}, 0);
                        } else {
                            layer.ffn_gate_inp = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP, "weight", i), {n_embd, n_expert}, 0);
                            layer.ffn_exp_probs_b = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias", i), {n_expert}, TENSOR_NOT_REQUIRED);

                            if (n_expert == 0) {
                                throw std::runtime_error("n_expert must be > 0");
                            }
                            if (n_expert_used == 0) {
                                throw std::runtime_error("n_expert_used must be > 0");
                            }

                            // MoE branch
                            layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {  n_embd, n_ff_exp, n_expert}, 0);
                            layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {n_ff_exp,   n_embd, n_expert}, 0);
                            layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {  n_embd, n_ff_exp, n_expert}, 0);

                            // Shared expert branch
                            layer.ffn_gate_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", i), {n_embd, n_ff_exp * n_expert_shared}, 0);
                            layer.ffn_down_shexp = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i), {        n_ff_exp * n_expert_shared, n_embd}, 0);
                            layer.ffn_up_shexp   = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,   "weight", i), {n_embd, n_ff_exp * n_expert_shared}, 0);
                        }
                    }
                } break;
            case LLM_ARCH_PLM:
                {
                    const int64_t n_embd_head_qk_rope = hparams.n_rot;
                    const int64_t n_embd_head_qk_nope = hparams.n_embd_head_k - hparams.n_rot;
                    const int64_t kv_lora_rank = hparams.n_lora_kv;

                    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

                    // output
                    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
                    // output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, 0);
                    output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);

                    for (int i = 0; i < n_layer; ++i) {
                        auto & layer = layers[i];

                        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

                        layer.wq        = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_embd_head_k * n_head}, 0);
                        layer.wkv_a_mqa = create_tensor(tn(LLM_TENSOR_ATTN_KV_A_MQA, "weight", i), {n_embd, kv_lora_rank + (n_embd_head_qk_rope)}, 0);
                        layer.attn_kv_a_norm = create_tensor(tn(LLM_TENSOR_ATTN_KV_A_NORM, "weight", i), {kv_lora_rank}, 0);
                        layer.wkv_b     = create_tensor(tn(LLM_TENSOR_ATTN_KV_B,     "weight", i), {kv_lora_rank, n_head * (n_embd_head_qk_nope + n_embd_head_v)}, 0);
                        layer.wo        = create_tensor(tn(LLM_TENSOR_ATTN_OUT,      "weight", i), {              n_head * (                      n_embd_head_v), n_embd}, 0);

                        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);
                        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd}, 0);
                        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff}, 0);
                    }
                } break;
            case LLM_ARCH_BITNET:
                {
                    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

                    // output
                    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);

                    for (int i = 0; i < n_layer; ++i) {
                        auto & layer = layers[i];

                        layer.attn_norm     = create_tensor(tn(LLM_TENSOR_ATTN_NORM,     "weight", i), {n_embd}, 0);
                        layer.attn_sub_norm = create_tensor(tn(LLM_TENSOR_ATTN_SUB_NORM, "weight", i), {n_embd}, 0);

                        layer.wq       = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_embd}, 0);
                        layer.wq_scale = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "scale",  i), {1}, TENSOR_NOT_REQUIRED);
                        layer.wk       = create_tensor(tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_embd_gqa}, 0);
                        layer.wk_scale = create_tensor(tn(LLM_TENSOR_ATTN_K,   "scale",  i), {1}, TENSOR_NOT_REQUIRED);
                        layer.wv       = create_tensor(tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_gqa}, 0);
                        layer.wv_scale = create_tensor(tn(LLM_TENSOR_ATTN_V,   "scale",  i), {1}, TENSOR_NOT_REQUIRED);
                        layer.wo       = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd}, 0);
                        layer.wo_scale = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "scale",  i), {1}, TENSOR_NOT_REQUIRED);

                        layer.ffn_norm     = create_tensor(tn(LLM_TENSOR_FFN_NORM,     "weight", i), {n_embd}, 0);
                        layer.ffn_sub_norm = create_tensor(tn(LLM_TENSOR_FFN_SUB_NORM, "weight", i), {n_ff}, 0);

                        layer.ffn_gate       = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd, n_ff}, 0);
                        layer.ffn_gate_scale = create_tensor(tn(LLM_TENSOR_FFN_GATE, "scale",  i), {1}, TENSOR_NOT_REQUIRED);
                        layer.ffn_down       = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff, n_embd}, 0);
                        layer.ffn_down_scale = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "scale",  i), {1}, TENSOR_NOT_REQUIRED);
                        layer.ffn_up         = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd, n_ff}, 0);
                        layer.ffn_up_scale   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "scale",  i), {1}, TENSOR_NOT_REQUIRED);
                    }
                } break;
            case LLM_ARCH_T5:
                {
                    const auto n_rel_attn_bkts = hparams.n_rel_attn_bkts;

                    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

                    // output
                    output_norm_enc = create_tensor(tn(LLM_TENSOR_ENC_OUTPUT_NORM, "weight"), {n_embd}, 0);
                    output_norm     = create_tensor(tn(LLM_TENSOR_DEC_OUTPUT_NORM, "weight"), {n_embd}, 0);

                    output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);
                    // if output is NULL, init from the input tok embed
                    if (output == NULL) {
                        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
                    }

                    for (int i = 0; i < n_layer; ++i) {
                        auto & layer = layers[i];

                        layer.attn_norm_enc  = create_tensor(tn(LLM_TENSOR_ENC_ATTN_NORM,  "weight", i), {n_embd}, 0);
                        layer.attn_rel_b_enc = create_tensor(tn(LLM_TENSOR_ENC_ATTN_REL_B, "weight", i), {n_head, n_rel_attn_bkts}, TENSOR_NOT_REQUIRED);

                        layer.wq_enc = create_tensor(tn(LLM_TENSOR_ENC_ATTN_Q,   "weight", i), {n_embd, n_embd_k_gqa}, 0);
                        layer.wk_enc = create_tensor(tn(LLM_TENSOR_ENC_ATTN_K,   "weight", i), {n_embd, n_embd_k_gqa}, 0);
                        layer.wv_enc = create_tensor(tn(LLM_TENSOR_ENC_ATTN_V,   "weight", i), {n_embd, n_embd_v_gqa}, 0);
                        layer.wo_enc = create_tensor(tn(LLM_TENSOR_ENC_ATTN_OUT, "weight", i), {n_embd_v_gqa, n_embd}, 0);

                        layer.ffn_norm_enc = create_tensor(tn(LLM_TENSOR_ENC_FFN_NORM, "weight", i), {n_embd}, 0);
                        layer.ffn_gate_enc = create_tensor(tn(LLM_TENSOR_ENC_FFN_GATE, "weight", i), {n_embd,   n_ff}, TENSOR_NOT_REQUIRED);
                        layer.ffn_down_enc = create_tensor(tn(LLM_TENSOR_ENC_FFN_DOWN, "weight", i), {  n_ff, n_embd}, 0);
                        layer.ffn_up_enc   = create_tensor(tn(LLM_TENSOR_ENC_FFN_UP,   "weight", i), {n_embd,   n_ff}, 0);

                        layer.attn_norm  = create_tensor(tn(LLM_TENSOR_DEC_ATTN_NORM,  "weight", i), {n_embd}, 0);
                        layer.attn_rel_b = create_tensor(tn(LLM_TENSOR_DEC_ATTN_REL_B, "weight", i), {n_head, n_rel_attn_bkts}, TENSOR_NOT_REQUIRED);

                        layer.wq = create_tensor(tn(LLM_TENSOR_DEC_ATTN_Q,   "weight", i), {n_embd, n_embd_k_gqa}, 0);
                        layer.wk = create_tensor(tn(LLM_TENSOR_DEC_ATTN_K,   "weight", i), {n_embd, n_embd_k_gqa}, 0);
                        layer.wv = create_tensor(tn(LLM_TENSOR_DEC_ATTN_V,   "weight", i), {n_embd, n_embd_v_gqa}, 0);
                        layer.wo = create_tensor(tn(LLM_TENSOR_DEC_ATTN_OUT, "weight", i), {n_embd_v_gqa, n_embd}, 0);

                        layer.attn_norm_cross  = create_tensor(tn(LLM_TENSOR_DEC_CROSS_ATTN_NORM,  "weight", i), {n_embd}, 0);
                        // this tensor seems to be unused in HF transformers implementation
                        layer.attn_rel_b_cross = create_tensor(tn(LLM_TENSOR_DEC_CROSS_ATTN_REL_B, "weight", i), {n_head, n_rel_attn_bkts}, TENSOR_NOT_REQUIRED);

                        layer.wq_cross = create_tensor(tn(LLM_TENSOR_DEC_CROSS_ATTN_Q,   "weight", i), {n_embd, n_embd_k_gqa}, 0);
                        layer.wk_cross = create_tensor(tn(LLM_TENSOR_DEC_CROSS_ATTN_K,   "weight", i), {n_embd, n_embd_k_gqa}, 0);
                        layer.wv_cross = create_tensor(tn(LLM_TENSOR_DEC_CROSS_ATTN_V,   "weight", i), {n_embd, n_embd_v_gqa}, 0);
                        layer.wo_cross = create_tensor(tn(LLM_TENSOR_DEC_CROSS_ATTN_OUT, "weight", i), {n_embd_v_gqa, n_embd}, 0);

                        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_DEC_FFN_NORM, "weight", i), {n_embd}, 0);
                        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_DEC_FFN_GATE, "weight", i), {n_embd,   n_ff}, TENSOR_NOT_REQUIRED);
                        layer.ffn_down = create_tensor(tn(LLM_TENSOR_DEC_FFN_DOWN, "weight", i), {  n_ff, n_embd}, 0);
                        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_DEC_FFN_UP,   "weight", i), {n_embd,   n_ff}, 0);
                    }
                } break;
            case LLM_ARCH_T5ENCODER:
                {
                    const auto n_rel_attn_bkts = hparams.n_rel_attn_bkts;

                    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

                    // output
                    output_norm_enc = create_tensor(tn(LLM_TENSOR_ENC_OUTPUT_NORM, "weight"), {n_embd}, 0);
                    output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);
                    // if output is NULL, init from the input tok embed
                    if (output == NULL) {
                        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
                    }

                    for (int i = 0; i < n_layer; ++i) {
                        auto & layer = layers[i];

                        layer.attn_norm_enc  = create_tensor(tn(LLM_TENSOR_ENC_ATTN_NORM,  "weight", i), {n_embd}, 0);
                        layer.attn_rel_b_enc = create_tensor(tn(LLM_TENSOR_ENC_ATTN_REL_B, "weight", i), {n_head, n_rel_attn_bkts}, TENSOR_NOT_REQUIRED);

                        layer.wq_enc = create_tensor(tn(LLM_TENSOR_ENC_ATTN_Q,   "weight", i), {n_embd, n_embd_k_gqa}, 0);
                        layer.wk_enc = create_tensor(tn(LLM_TENSOR_ENC_ATTN_K,   "weight", i), {n_embd, n_embd_k_gqa}, 0);
                        layer.wv_enc = create_tensor(tn(LLM_TENSOR_ENC_ATTN_V,   "weight", i), {n_embd, n_embd_v_gqa}, 0);
                        layer.wo_enc = create_tensor(tn(LLM_TENSOR_ENC_ATTN_OUT, "weight", i), {n_embd_v_gqa, n_embd}, 0);

                        layer.ffn_norm_enc = create_tensor(tn(LLM_TENSOR_ENC_FFN_NORM, "weight", i), {n_embd}, 0);
                        layer.ffn_gate_enc = create_tensor(tn(LLM_TENSOR_ENC_FFN_GATE, "weight", i), {n_embd,   n_ff}, TENSOR_NOT_REQUIRED);
                        layer.ffn_down_enc = create_tensor(tn(LLM_TENSOR_ENC_FFN_DOWN, "weight", i), {  n_ff, n_embd}, 0);
                        layer.ffn_up_enc   = create_tensor(tn(LLM_TENSOR_ENC_FFN_UP,   "weight", i), {n_embd,   n_ff}, 0);
                    }
                } break;
            case LLM_ARCH_JAIS:
                {
                    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

                    // output
                    output_norm   = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
                    output_norm_b = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "bias"),   {n_embd}, 0);
                    output        = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, 0);

                    for (int i = 0; i < n_layer; ++i) {
                        auto & layer = layers[i];

                        layer.attn_norm   = create_tensor(tn(LLM_TENSOR_ATTN_NORM,   "weight", i), {n_embd}, 0);
                        layer.attn_norm_b = create_tensor(tn(LLM_TENSOR_ATTN_NORM,   "bias", i),   {n_embd}, 0);

                        layer.wqkv = create_tensor(tn(LLM_TENSOR_ATTN_QKV, "weight", i), {n_embd, n_embd + 2*n_embd_gqa}, 0);
                        layer.bqkv = create_tensor(tn(LLM_TENSOR_ATTN_QKV, "bias", i),   {n_embd + 2*n_embd_gqa}, 0);

                        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd}, 0);
                        layer.bo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "bias", i),   {n_embd}, 0);

                        layer.ffn_norm   = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);
                        layer.ffn_norm_b = create_tensor(tn(LLM_TENSOR_FFN_NORM, "bias", i),   {n_embd}, 0);

                        layer.ffn_down   = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff, n_embd}, 0);
                        layer.ffn_down_b = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "bias", i),   {n_embd}, 0);

                        layer.ffn_gate   = create_tensor(tn(LLM_TENSOR_FFN_GATE,   "weight", i), {n_embd, n_ff}, 0);
                        layer.ffn_gate_b = create_tensor(tn(LLM_TENSOR_FFN_GATE,   "bias", i),   {n_ff}, 0);

                        layer.ffn_up     = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd, n_ff}, 0);
                        layer.ffn_up_b   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "bias", i),   {n_ff}, 0);
                    }
                } break;
            case LLM_ARCH_CHATGLM:
                {
                    tok_embd   = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD,      "weight"), {n_embd, n_vocab}, 0);

                    // output
                    output_norm   = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
                    output        = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);
                    // if output is NULL, init from the input tok embed
                    if (output == NULL) {
                        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
                    }

                    for (int i = 0; i < n_layer; ++i) {
                        auto & layer = layers[i];

                        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);
                        layer.wqkv = create_tensor(tn(LLM_TENSOR_ATTN_QKV, "weight", i), {n_embd, n_embd + 2*n_embd_gqa}, TENSOR_NOT_REQUIRED);
                        layer.bqkv = create_tensor(tn(LLM_TENSOR_ATTN_QKV, "bias", i),   {n_embd + 2*n_embd_gqa}, TENSOR_NOT_REQUIRED);

                        if (layer.wqkv == nullptr) {
                            layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_embd_head_k * n_head}, 0);
                            layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_embd_k_gqa}, 0);
                            layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_v_gqa}, 0);
                            layer.bq = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "bias", i), {n_embd}, TENSOR_NOT_REQUIRED);
                            layer.bk = create_tensor(tn(LLM_TENSOR_ATTN_K,   "bias", i), {n_embd_gqa}, TENSOR_NOT_REQUIRED);
                            layer.bv = create_tensor(tn(LLM_TENSOR_ATTN_V,   "bias", i), {n_embd_gqa}, TENSOR_NOT_REQUIRED);
                        }

                        layer.wo   = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd}, 0);

                        layer.ffn_norm   = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);

                        layer.ffn_up     = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd, n_ff * 2}, 0);

                        layer.ffn_down   = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff, n_embd}, 0);
                    }
                } break;
            case LLM_ARCH_GLM4:
                {
                    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

                    // output
                    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
                    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);
                    // if output is NULL, init from the input tok embed
                    if (output == NULL) {
                        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
                    }

                    for (int i = 0; i < n_layer; ++i) {
                        auto & layer = layers[i];

                        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);
                        layer.wqkv = create_tensor(tn(LLM_TENSOR_ATTN_QKV, "weight", i), {n_embd, n_embd + 2*n_embd_gqa}, TENSOR_NOT_REQUIRED);
                        layer.bqkv = create_tensor(tn(LLM_TENSOR_ATTN_QKV, "bias", i),   {n_embd + 2*n_embd_gqa}, TENSOR_NOT_REQUIRED);

                        if (layer.wqkv == nullptr) {
                            layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_embd_head_k * n_head}, 0);
                            layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_embd_k_gqa}, 0);
                            layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_v_gqa}, 0);
                            layer.bq = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "bias", i), {n_embd}, TENSOR_NOT_REQUIRED);
                            layer.bk = create_tensor(tn(LLM_TENSOR_ATTN_K,   "bias", i), {n_embd_gqa}, TENSOR_NOT_REQUIRED);
                            layer.bv = create_tensor(tn(LLM_TENSOR_ATTN_V,   "bias", i), {n_embd_gqa}, TENSOR_NOT_REQUIRED);
                        }

                        layer.wo   = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd}, 0);

                        layer.attn_post_norm = create_tensor(tn(LLM_TENSOR_ATTN_POST_NORM, "weight", i), {n_embd}, 0);

                        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);
                        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd}, 0);
                        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd, n_ff * 2}, 0);

                        layer.ffn_post_norm  = create_tensor(tn(LLM_TENSOR_FFN_POST_NORM, "weight", i), {n_embd}, 0);
                    }
                } break;
            case LLM_ARCH_GLM4_MOE:
                {
                    const int64_t n_expert        = hparams.n_expert;
                    const int64_t n_expert_used   = hparams.n_expert_used;
                    const int64_t n_expert_shared = hparams.n_expert_shared;

                    GGML_ASSERT(hparams.n_expert > 0 && "n_expert must be > 0 for GLM4_MOE MoE layers");
                    GGML_ASSERT(hparams.n_expert_used > 0 && "n_expert_used must be > 0 for GLM4_MOE MoE layers");

                    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, 0);

                    // output
                    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), { n_embd }, 0);
                    output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), { n_embd, n_vocab }, TENSOR_NOT_REQUIRED);
                    // if output is NULL, init from the input tok embed
                    if (output == NULL) {
                        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, TENSOR_DUPLICATED);
                    }

                    // Load ALL tensors including NextN layer to satisfy total tensor count
                    // but only PROCESS up to last layer (skipping final NextN layer) in forward pass
                    for (int i = 0; i < n_layer; ++i) {
                        int flags = 0;
                        if (hparams.nextn_predict_layers > 0 && static_cast<uint32_t>(i) >= n_layer - hparams.nextn_predict_layers) {
                            // skip all tensors in the NextN layers
                            flags |= TENSOR_SKIP;
                        }

                        auto & layer = layers[i];

                        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), { n_embd }, flags);

                        // GLM-style attention with bias terms
                        layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q, "weight", i), { n_embd, n_embd_head_k * n_head }, flags);
                        layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K, "weight", i), { n_embd, n_embd_k_gqa }, flags);
                        layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V, "weight", i), { n_embd, n_embd_v_gqa }, flags);
                        layer.bq = create_tensor(tn(LLM_TENSOR_ATTN_Q, "bias", i), { n_embd_head_k * n_head }, flags);
                        layer.bk = create_tensor(tn(LLM_TENSOR_ATTN_K, "bias", i), { n_embd_k_gqa }, flags);
                        layer.bv = create_tensor(tn(LLM_TENSOR_ATTN_V, "bias", i), { n_embd_v_gqa }, flags);

                        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), { n_embd_head_k * n_head, n_embd }, flags);

                        // K/Q norm tensors (optional for GLM-4.5 355B variant)
                        layer.attn_q_norm = create_tensor(
                            tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), { n_embd_head_k }, TENSOR_NOT_REQUIRED | flags);
                        layer.attn_k_norm = create_tensor(
                            tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), { n_embd_head_k }, TENSOR_NOT_REQUIRED | flags);

                        layer.attn_post_norm = create_tensor(tn(LLM_TENSOR_ATTN_POST_NORM, "weight", i), { n_embd }, flags);

                        // Check if this layer uses MoE or dense FFN based on n_layer_dense_lead
                        // GLM 4.5 uses hybrid architecture: layer 0 is dense, layers 1+ are MoE
                        const bool use_moe = (static_cast<uint32_t>(i) >= hparams.n_layer_dense_lead);

                        if (use_moe) {
                            // MoE layers
                            layer.ffn_gate_inp =
                                create_tensor(tn(LLM_TENSOR_FFN_GATE_INP, "weight", i), { n_embd, n_expert }, flags);
                            layer.ffn_exp_probs_b = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias", i), { n_expert }, flags);

                            // MoE branch
                            const int64_t n_ff_exp = hparams.n_ff_exp ? hparams.n_ff_exp : n_ff / n_expert_used;

                            layer.ffn_gate_exps = create_tensor(
                                tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), { n_embd, n_ff_exp, n_expert }, flags);
                            layer.ffn_down_exps = create_tensor(
                                tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), { n_ff_exp, n_embd, n_expert }, flags);
                            layer.ffn_up_exps = create_tensor(
                                tn(LLM_TENSOR_FFN_UP_EXPS, "weight", i), { n_embd, n_ff_exp, n_expert }, flags);

                            // Shared expert
                            if (n_expert_shared > 0) {
                                const int64_t n_ff_shexp = n_ff_exp * n_expert_shared;
                                layer.ffn_gate_shexp = create_tensor(
                                    tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", i), { n_embd, n_ff_shexp }, flags);
                                layer.ffn_down_shexp = create_tensor(
                                    tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i), { n_ff_shexp, n_embd }, flags);
                                layer.ffn_up_shexp = create_tensor(
                                    tn(LLM_TENSOR_FFN_UP_SHEXP, "weight", i), { n_embd, n_ff_shexp }, flags);
                            }
                        } else {
                            // Dense layers (first k layers) - GLM uses separate gate/up projections
                            layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), { n_embd, n_ff }, flags);
                            layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), { n_ff, n_embd }, flags);
                            layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), { n_embd, n_ff }, flags);
                        }

                        // NextN/MTP tensors (preserved but unused) - conditionally load for last nextn_predict_layers
                        if (hparams.nextn_predict_layers > 0 && static_cast<uint32_t>(i) >= n_layer - hparams.nextn_predict_layers) {
                            layer.nextn.eh_proj          = create_tensor(tn(LLM_TENSOR_NEXTN_EH_PROJ, "weight", i), { 2 * n_embd, n_embd }, flags);
                            layer.nextn.embed_tokens     = create_tensor(tn(LLM_TENSOR_NEXTN_EMBED_TOKENS, "weight", i), { n_embd, n_vocab }, flags);
                            layer.nextn.enorm            = create_tensor(tn(LLM_TENSOR_NEXTN_ENORM, "weight", i), { n_embd }, flags);
                            layer.nextn.hnorm            = create_tensor(tn(LLM_TENSOR_NEXTN_HNORM, "weight", i), { n_embd }, flags);
                            layer.nextn.shared_head_head = create_tensor(tn(LLM_TENSOR_NEXTN_SHARED_HEAD_HEAD, "weight", i), { n_embd, n_vocab }, flags);
                            layer.nextn.shared_head_norm = create_tensor(tn(LLM_TENSOR_NEXTN_SHARED_HEAD_NORM, "weight", i), { n_embd }, flags);
                        }
                    }
                }
                break;
            case LLM_ARCH_NEMOTRON:
                {
                    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

                    // output
                    output_norm   = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
                    output_norm_b = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "bias"), {n_embd}, 0);
                    output        = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), {n_embd, n_vocab}, 0);

                    for (int i = 0; i < n_layer; ++i) {
                        auto & layer = layers[i];

                        layer.attn_norm   = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);
                        layer.attn_norm_b = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "bias", i), {n_embd}, 0);

                        layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_embd}, 0);
                        layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_embd_gqa}, 0);
                        layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_gqa}, 0);
                        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd}, 0);

                        // optional bias tensors
                        layer.bq = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "bias", i), {n_embd},     TENSOR_NOT_REQUIRED);
                        layer.bk = create_tensor(tn(LLM_TENSOR_ATTN_K,   "bias", i), {n_embd_gqa}, TENSOR_NOT_REQUIRED);
                        layer.bv = create_tensor(tn(LLM_TENSOR_ATTN_V,   "bias", i), {n_embd_gqa}, TENSOR_NOT_REQUIRED);
                        layer.bo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "bias", i), {n_embd},     TENSOR_NOT_REQUIRED);

                        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);
                        layer.ffn_norm_b = create_tensor(tn(LLM_TENSOR_FFN_NORM, "bias", i), {n_embd}, 0);

                        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd}, 0);
                        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff}, 0);

                        // optional MLP bias
                        layer.ffn_down_b = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "bias", i), {n_embd}, TENSOR_NOT_REQUIRED);
                        layer.ffn_up_b   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "bias", i), {n_ff}, TENSOR_NOT_REQUIRED);
                    }
                } break;
            case LLM_ARCH_EXAONE:
                {
                    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

                    // output
                    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
                    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);

                    // if output is NULL, init from the input tok embed
                    if (output == NULL) {
                        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
                    }

                    for (int i = 0; i < n_layer; ++i) {
                        auto & layer = layers[i];

                        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

                        layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_embd_head_k * n_head}, 0);
                        layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_embd_k_gqa}, 0);
                        layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_v_gqa}, 0);
                        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head_k * n_head, n_embd}, 0);

                        layer.ffn_norm   = create_tensor(tn(LLM_TENSOR_FFN_NORM,   "weight", i), {n_embd}, 0);
                        layer.rope_freqs = create_tensor(tn(LLM_TENSOR_ROPE_FREQS, "weight", i), {n_rot/2}, TENSOR_NOT_REQUIRED | (i != 0 ? TENSOR_DUPLICATED : 0));
                        layer.ffn_gate   = create_tensor(tn(LLM_TENSOR_FFN_GATE,   "weight", i), {n_embd,   n_ff}, 0);
                        layer.ffn_down   = create_tensor(tn(LLM_TENSOR_FFN_DOWN,   "weight", i), {  n_ff, n_embd}, 0);
                        layer.ffn_up     = create_tensor(tn(LLM_TENSOR_FFN_UP,     "weight", i), {n_embd,   n_ff}, 0);
                    }
                } break;
            case LLM_ARCH_EXAONE4:
                {
                    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

                    // output
                    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
                    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);

                    // if output is NULL, init from the input tok embed
                    if (output == NULL) {
                        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
                    }

                    for (int i = 0; i < n_layer; ++i) {
                        auto & layer = layers[i];

                        layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_embd_head_k * n_head}, 0);
                        layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_embd_k_gqa}, 0);
                        layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_v_gqa}, 0);
                        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd}, 0);

                        layer.rope_freqs = create_tensor(tn(LLM_TENSOR_ROPE_FREQS, "weight", i), {n_rot/2}, TENSOR_NOT_REQUIRED | (i != 0 ? TENSOR_DUPLICATED : 0));

                        layer.attn_post_norm = create_tensor(tn(LLM_TENSOR_ATTN_POST_NORM, "weight", i), {n_embd}, 0);
                        layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), {n_embd_head_k}, 0);
                        layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), {n_embd_head_k}, 0);

                        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd, n_ff}, 0);
                        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd}, 0);
                        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff}, 0);
                        layer.ffn_post_norm  = create_tensor(tn(LLM_TENSOR_FFN_POST_NORM, "weight", i), {n_embd}, 0);
                    }
                } break;
            case LLM_ARCH_RWKV6:
                {
                    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

                    // Block 0, LN0
                    tok_norm = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD_NORM, "weight"), {n_embd}, 0);
                    tok_norm_b = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD_NORM, "bias"), {n_embd}, 0);

                    // output
                    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
                    output_norm_b = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "bias"), {n_embd}, 0);
                    output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), {n_embd, n_vocab}, 0);

                    const int time_mix_extra_dim = hparams.time_mix_extra_dim;
                    const int time_decay_extra_dim = hparams.time_decay_extra_dim;
                    const int head_size = hparams.wkv_head_size;
                    const int attn_hidden_size = n_embd;
                    const int ffn_size = hparams.n_ff_arr[0];

                    for (int i = 0; i < n_layer; ++i) {
                        auto & layer = layers[i];

                        layer.attn_norm   = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);
                        layer.attn_norm_b = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "bias", i),   {n_embd}, 0);

                        layer.attn_norm_2   = create_tensor(tn(LLM_TENSOR_ATTN_NORM_2, "weight", i), {n_embd}, 0);
                        layer.attn_norm_2_b = create_tensor(tn(LLM_TENSOR_ATTN_NORM_2, "bias", i),   {n_embd}, 0);

                        layer.time_mix_w1 = create_tensor(tn(LLM_TENSOR_TIME_MIX_W1, "weight", i), {n_embd, time_mix_extra_dim * 5}, 0);
                        layer.time_mix_w2 = create_tensor(tn(LLM_TENSOR_TIME_MIX_W2, "weight", i), {time_mix_extra_dim, n_embd, 5}, 0);

                        layer.time_mix_lerp_x = create_tensor(tn(LLM_TENSOR_TIME_MIX_LERP_X, "weight", i), {n_embd, 1, 1}, 0);
                        layer.time_mix_lerp_w = create_tensor(tn(LLM_TENSOR_TIME_MIX_LERP_W, "weight", i), {n_embd, 1, 1}, TENSOR_NOT_REQUIRED);
                        layer.time_mix_lerp_k = create_tensor(tn(LLM_TENSOR_TIME_MIX_LERP_K, "weight", i), {n_embd, 1, 1}, TENSOR_NOT_REQUIRED);
                        layer.time_mix_lerp_v = create_tensor(tn(LLM_TENSOR_TIME_MIX_LERP_V, "weight", i), {n_embd, 1, 1}, TENSOR_NOT_REQUIRED);
                        layer.time_mix_lerp_r = create_tensor(tn(LLM_TENSOR_TIME_MIX_LERP_R, "weight", i), {n_embd, 1, 1}, TENSOR_NOT_REQUIRED);
                        layer.time_mix_lerp_g = create_tensor(tn(LLM_TENSOR_TIME_MIX_LERP_G, "weight", i), {n_embd, 1, 1}, TENSOR_NOT_REQUIRED);
                        layer.time_mix_lerp_fused = create_tensor(tn(LLM_TENSOR_TIME_MIX_LERP_FUSED, "weight", i), {n_embd, 1, 1, 5}, TENSOR_NOT_REQUIRED);
                        GGML_ASSERT(!(layer.time_mix_lerp_fused == NULL && layer.time_mix_lerp_w == NULL));

                        layer.time_mix_first = create_tensor(tn(LLM_TENSOR_TIME_MIX_FIRST, "weight", i), {head_size, n_embd / head_size}, 0);
                        layer.time_mix_decay = create_tensor(tn(LLM_TENSOR_TIME_MIX_DECAY, "weight", i), {n_embd}, 0);
                        layer.time_mix_decay_w1 = create_tensor(tn(LLM_TENSOR_TIME_MIX_DECAY_W1, "weight", i), {n_embd, time_decay_extra_dim}, 0);
                        layer.time_mix_decay_w2 = create_tensor(tn(LLM_TENSOR_TIME_MIX_DECAY_W2, "weight", i), {time_decay_extra_dim, attn_hidden_size}, 0);
                        layer.time_mix_key = create_tensor(tn(LLM_TENSOR_TIME_MIX_KEY, "weight", i), {attn_hidden_size, n_embd}, 0);
                        layer.time_mix_value = create_tensor(tn(LLM_TENSOR_TIME_MIX_VALUE, "weight", i), {attn_hidden_size, n_embd}, 0);
                        layer.time_mix_receptance = create_tensor(tn(LLM_TENSOR_TIME_MIX_RECEPTANCE, "weight", i), {attn_hidden_size, n_embd}, 0);
                        layer.time_mix_gate = create_tensor(tn(LLM_TENSOR_TIME_MIX_GATE, "weight", i), {attn_hidden_size, n_embd}, 0);

                        layer.time_mix_ln = create_tensor(tn(LLM_TENSOR_TIME_MIX_LN, "weight", i), {n_embd}, 0);
                        layer.time_mix_ln_b = create_tensor(tn(LLM_TENSOR_TIME_MIX_LN, "bias", i), {n_embd}, 0);
                        layer.time_mix_output = create_tensor(tn(LLM_TENSOR_TIME_MIX_OUTPUT, "weight", i), {n_embd, attn_hidden_size}, 0);

                        layer.channel_mix_lerp_k = create_tensor(tn(LLM_TENSOR_CHANNEL_MIX_LERP_K, "weight", i), {n_embd, 1, 1}, 0);
                        layer.channel_mix_lerp_r = create_tensor(tn(LLM_TENSOR_CHANNEL_MIX_LERP_R, "weight", i), {n_embd, 1, 1}, 0);

                        layer.channel_mix_key = create_tensor(tn(LLM_TENSOR_CHANNEL_MIX_KEY, "weight", i), {n_embd, ffn_size}, 0);
                        layer.channel_mix_value = create_tensor(tn(LLM_TENSOR_CHANNEL_MIX_VALUE, "weight", i), {ffn_size, n_embd}, 0);
                        layer.channel_mix_receptance = create_tensor(tn(LLM_TENSOR_CHANNEL_MIX_RECEPTANCE, "weight", i), {n_embd, n_embd}, 0);
                    }

                } break;
            case LLM_ARCH_RWKV6QWEN2:
                {
                    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

                    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
                    output_norm_b = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "bias"), {n_embd}, TENSOR_NOT_REQUIRED);
                    output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), {n_embd, n_vocab}, 0);

                    const int time_mix_extra_dim = hparams.time_mix_extra_dim;
                    const int time_decay_extra_dim = hparams.time_decay_extra_dim;
                    const int head_size = hparams.wkv_head_size;
                    const int attn_hidden_size = n_embd;
                    const int n_head_kv = hparams.n_head_kv();
                    int attn_key_value_size;
                    if (n_head_kv == 0 || attn_hidden_size / head_size == n_head_kv) {
                        attn_key_value_size = attn_hidden_size;
                    } else {
                        attn_key_value_size = n_head_kv * head_size;
                    }

                    for (int i = 0; i < n_layer; ++i) {
                        auto & layer = layers[i];

                        layer.attn_norm   = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

                        layer.time_mix_w1 = create_tensor(tn(LLM_TENSOR_TIME_MIX_W1, "weight", i), {n_embd, time_mix_extra_dim * 5}, 0);
                        layer.time_mix_w2 = create_tensor(tn(LLM_TENSOR_TIME_MIX_W2, "weight", i), {time_mix_extra_dim, n_embd, 5}, 0);

                        layer.time_mix_lerp_x = create_tensor(tn(LLM_TENSOR_TIME_MIX_LERP_X, "weight", i), {n_embd, 1, 1}, 0);
                        layer.time_mix_lerp_fused = create_tensor(tn(LLM_TENSOR_TIME_MIX_LERP_FUSED, "weight", i), {n_embd, 1, 1, 5}, 0);

                        layer.time_mix_first = create_tensor(tn(LLM_TENSOR_TIME_MIX_FIRST, "weight", i), {head_size, n_embd / head_size}, TENSOR_NOT_REQUIRED);
                        layer.time_mix_decay = create_tensor(tn(LLM_TENSOR_TIME_MIX_DECAY, "weight", i), {n_embd}, 0);
                        layer.time_mix_decay_w1 = create_tensor(tn(LLM_TENSOR_TIME_MIX_DECAY_W1, "weight", i), {n_embd, time_decay_extra_dim}, 0);
                        layer.time_mix_decay_w2 = create_tensor(tn(LLM_TENSOR_TIME_MIX_DECAY_W2, "weight", i), {time_decay_extra_dim, attn_hidden_size}, 0);
                        layer.time_mix_key = create_tensor(tn(LLM_TENSOR_TIME_MIX_KEY, "weight", i), {n_embd, attn_key_value_size}, 0);
                        layer.time_mix_value = create_tensor(tn(LLM_TENSOR_TIME_MIX_VALUE, "weight", i), {n_embd, attn_key_value_size}, 0);
                        layer.time_mix_receptance = create_tensor(tn(LLM_TENSOR_TIME_MIX_RECEPTANCE, "weight", i), {attn_hidden_size, n_embd}, 0);
                        layer.time_mix_gate = create_tensor(tn(LLM_TENSOR_TIME_MIX_GATE, "weight", i), {attn_hidden_size, n_embd}, 0);
                        // optional bias tensors
                        layer.time_mix_key_b = create_tensor(tn(LLM_TENSOR_TIME_MIX_KEY, "bias", i), {attn_key_value_size}, TENSOR_NOT_REQUIRED);
                        layer.time_mix_value_b = create_tensor(tn(LLM_TENSOR_TIME_MIX_VALUE, "bias", i), {attn_key_value_size}, TENSOR_NOT_REQUIRED);
                        layer.time_mix_receptance_b = create_tensor(tn(LLM_TENSOR_TIME_MIX_RECEPTANCE, "bias", i), {attn_hidden_size}, TENSOR_NOT_REQUIRED);

                        layer.time_mix_output = create_tensor(tn(LLM_TENSOR_TIME_MIX_OUTPUT, "weight", i), {n_embd, attn_hidden_size}, 0);

                        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);

                        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff}, 0);
                        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd}, 0);
                        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff}, 0);
                    }
                } break;
            case LLM_ARCH_SOLAR:
                {
                    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

                    // output
                    {
                        output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
                        output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, llama_model_loader::TENSOR_NOT_REQUIRED);
                    }

                    for (int i = 0; i < n_layer; ++i) {
                        auto & layer = layers[i];

                        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

                        layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_embd_head_k * n_head}, 0);
                        layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_embd_k_gqa}, 0);
                        layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_v_gqa}, 0);
                        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head_k * n_head, n_embd}, 0);

                        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);

                        layer.bskcn_tv = create_tensor(tn(LLM_TENSOR_BSKCN_TV, "weight", i), {2}, llama_model_loader::TENSOR_NOT_REQUIRED | (i != 0 ? llama_model_loader::TENSOR_DUPLICATED : 0));
                        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff}, 0);
                        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd}, 0);
                        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff}, 0);
                    }
                } break;
            case LLM_ARCH_RWKV7:
                {
                    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

                    // Block 0, LN0
                    tok_norm = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD_NORM, "weight"), {n_embd}, 0);
                    tok_norm_b = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD_NORM, "bias"), {n_embd}, 0);

                    // output
                    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
                    output_norm_b = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "bias"), {n_embd}, 0);
                    output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), {n_embd, n_vocab}, 0);

                    const int n_lora_decay = hparams.n_lora_decay;
                    const int n_lora_iclr = hparams.n_lora_iclr;
                    const int n_lora_value_res_mix = hparams.n_lora_value_res_mix;
                    const int n_lora_gate = hparams.n_lora_gate;
                    const int attn_hidden_size = n_embd;
                    const int ffn_size = hparams.n_ff_arr[0];

                    for (int i = 0; i < n_layer; ++i) {
                        auto & layer = layers[i];

                        layer.attn_norm   = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);
                        layer.attn_norm_b = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "bias", i),   {n_embd}, 0);

                        layer.attn_norm_2   = create_tensor(tn(LLM_TENSOR_ATTN_NORM_2, "weight", i), {n_embd}, 0);
                        layer.attn_norm_2_b = create_tensor(tn(LLM_TENSOR_ATTN_NORM_2, "bias", i),   {n_embd}, 0);

                        layer.time_mix_w0 = create_tensor(tn(LLM_TENSOR_TIME_MIX_W0, "weight", i), {n_embd}, 0);
                        layer.time_mix_w1 = create_tensor(tn(LLM_TENSOR_TIME_MIX_W1, "weight", i), {n_embd, n_lora_decay}, 0);
                        layer.time_mix_w2 = create_tensor(tn(LLM_TENSOR_TIME_MIX_W2, "weight", i), {n_lora_decay, n_embd}, 0);

                        layer.time_mix_a0 = create_tensor(tn(LLM_TENSOR_TIME_MIX_A0, "weight", i), {n_embd}, 0);
                        layer.time_mix_a1 = create_tensor(tn(LLM_TENSOR_TIME_MIX_A1, "weight", i), {n_embd, n_lora_iclr}, 0);
                        layer.time_mix_a2 = create_tensor(tn(LLM_TENSOR_TIME_MIX_A2, "weight", i), {n_lora_iclr, n_embd}, 0);

                        if (i == 0) {
                            // actually not used
                            layer.time_mix_v0 = create_tensor(tn(LLM_TENSOR_TIME_MIX_V0, "weight", i), {n_embd}, 0);
                            layer.time_mix_v1 = create_tensor(tn(LLM_TENSOR_TIME_MIX_V1, "weight", i), {n_embd, n_lora_iclr}, 0);
                            layer.time_mix_v2 = create_tensor(tn(LLM_TENSOR_TIME_MIX_V2, "weight", i), {n_lora_iclr, n_embd}, 0);
                        } else {
                            layer.time_mix_v0 = create_tensor(tn(LLM_TENSOR_TIME_MIX_V0, "weight", i), {n_embd}, 0);
                            layer.time_mix_v1 = create_tensor(tn(LLM_TENSOR_TIME_MIX_V1, "weight", i), {n_embd, n_lora_value_res_mix}, 0);
                            layer.time_mix_v2 = create_tensor(tn(LLM_TENSOR_TIME_MIX_V2, "weight", i), {n_lora_value_res_mix, n_embd}, 0);
                        }

                        layer.time_mix_g1 = create_tensor(tn(LLM_TENSOR_TIME_MIX_G1, "weight", i), {n_embd, n_lora_gate}, 0);
                        layer.time_mix_g2 = create_tensor(tn(LLM_TENSOR_TIME_MIX_G2, "weight", i), {n_lora_gate, n_embd}, 0);

                        layer.time_mix_lerp_fused = create_tensor(tn(LLM_TENSOR_TIME_MIX_LERP_FUSED, "weight", i), {n_embd, 1, 1, 6}, 0);

                        layer.time_mix_k_k = create_tensor(tn(LLM_TENSOR_TIME_MIX_K_K, "weight", i), {attn_hidden_size}, 0);
                        layer.time_mix_k_a = create_tensor(tn(LLM_TENSOR_TIME_MIX_K_A, "weight", i), {attn_hidden_size}, 0);
                        layer.time_mix_r_k = create_tensor(tn(LLM_TENSOR_TIME_MIX_R_K, "weight", i), {attn_hidden_size}, 0);

                        layer.time_mix_key = create_tensor(tn(LLM_TENSOR_TIME_MIX_KEY, "weight", i), {attn_hidden_size, n_embd}, 0);
                        layer.time_mix_value = create_tensor(tn(LLM_TENSOR_TIME_MIX_VALUE, "weight", i), {attn_hidden_size, n_embd}, 0);
                        layer.time_mix_receptance = create_tensor(tn(LLM_TENSOR_TIME_MIX_RECEPTANCE, "weight", i), {attn_hidden_size, n_embd}, 0);

                        layer.time_mix_ln = create_tensor(tn(LLM_TENSOR_TIME_MIX_LN, "weight", i), {n_embd}, 0);
                        layer.time_mix_ln_b = create_tensor(tn(LLM_TENSOR_TIME_MIX_LN, "bias", i), {n_embd}, 0);
                        layer.time_mix_output = create_tensor(tn(LLM_TENSOR_TIME_MIX_OUTPUT, "weight", i), {n_embd, attn_hidden_size}, 0);

                        layer.channel_mix_lerp_k = create_tensor(tn(LLM_TENSOR_CHANNEL_MIX_LERP_K, "weight", i), {n_embd, 1, 1}, 0);

                        layer.channel_mix_key = create_tensor(tn(LLM_TENSOR_CHANNEL_MIX_KEY, "weight", i), {n_embd, ffn_size}, 0);
                        layer.channel_mix_value = create_tensor(tn(LLM_TENSOR_CHANNEL_MIX_VALUE, "weight", i), {ffn_size, n_embd}, 0);
                    }

                } break;
            case LLM_ARCH_ARWKV7:
                {
                    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

                    // output
                    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
                    output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), {n_embd, n_vocab}, 0);

                    const int n_lora_decay = hparams.n_lora_decay;
                    const int n_lora_iclr = hparams.n_lora_iclr;
                    const int n_lora_value_res_mix = hparams.n_lora_value_res_mix;
                    const int n_lora_gate = hparams.n_lora_gate;
                    const int attn_hidden_size = n_embd;

                    for (int i = 0; i < n_layer; ++i) {
                        auto & layer = layers[i];

                        layer.attn_norm   = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

                        layer.time_mix_w0 = create_tensor(tn(LLM_TENSOR_TIME_MIX_W0, "weight", i), {n_embd}, 0);
                        layer.time_mix_w1 = create_tensor(tn(LLM_TENSOR_TIME_MIX_W1, "weight", i), {n_embd, n_lora_decay}, 0);
                        layer.time_mix_w2 = create_tensor(tn(LLM_TENSOR_TIME_MIX_W2, "weight", i), {n_lora_decay, n_embd}, 0);

                        layer.time_mix_a0 = create_tensor(tn(LLM_TENSOR_TIME_MIX_A0, "weight", i), {n_embd}, 0);
                        layer.time_mix_a1 = create_tensor(tn(LLM_TENSOR_TIME_MIX_A1, "weight", i), {n_embd, n_lora_iclr}, 0);
                        layer.time_mix_a2 = create_tensor(tn(LLM_TENSOR_TIME_MIX_A2, "weight", i), {n_lora_iclr, n_embd}, 0);

                        if (i == 0) {
                            // actually not used
                            layer.time_mix_v0 = create_tensor(tn(LLM_TENSOR_TIME_MIX_V0, "weight", i), {n_embd}, 0);
                            layer.time_mix_v1 = create_tensor(tn(LLM_TENSOR_TIME_MIX_V1, "weight", i), {n_embd, n_lora_iclr}, 0);
                            layer.time_mix_v2 = create_tensor(tn(LLM_TENSOR_TIME_MIX_V2, "weight", i), {n_lora_iclr, n_embd}, 0);
                        } else {
                            layer.time_mix_v0 = create_tensor(tn(LLM_TENSOR_TIME_MIX_V0, "weight", i), {n_embd}, 0);
                            layer.time_mix_v1 = create_tensor(tn(LLM_TENSOR_TIME_MIX_V1, "weight", i), {n_embd, n_lora_value_res_mix}, 0);
                            layer.time_mix_v2 = create_tensor(tn(LLM_TENSOR_TIME_MIX_V2, "weight", i), {n_lora_value_res_mix, n_embd}, 0);
                        }

                        layer.time_mix_g1 = create_tensor(tn(LLM_TENSOR_TIME_MIX_G1, "weight", i), {n_embd, n_lora_gate}, TENSOR_NOT_REQUIRED);
                        layer.time_mix_g2 = create_tensor(tn(LLM_TENSOR_TIME_MIX_G2, "weight", i), {n_lora_gate, n_embd}, TENSOR_NOT_REQUIRED);

                        try {
                            layer.time_mix_lerp_fused = create_tensor(tn(LLM_TENSOR_TIME_MIX_LERP_FUSED, "weight", i), {n_embd, 1, 1, 6}, 0);
                        } catch(std::runtime_error & e) {
                            // ARWKV models may not have gate tensors
                            layer.time_mix_lerp_fused = create_tensor(tn(LLM_TENSOR_TIME_MIX_LERP_FUSED, "weight", i), {n_embd, 1, 1, 5}, 0);
                        }

                        layer.time_mix_k_k = create_tensor(tn(LLM_TENSOR_TIME_MIX_K_K, "weight", i), {attn_hidden_size}, 0);
                        layer.time_mix_k_a = create_tensor(tn(LLM_TENSOR_TIME_MIX_K_A, "weight", i), {attn_hidden_size}, 0);
                        layer.time_mix_r_k = create_tensor(tn(LLM_TENSOR_TIME_MIX_R_K, "weight", i), {attn_hidden_size}, 0);

                        layer.time_mix_key = create_tensor(tn(LLM_TENSOR_TIME_MIX_KEY, "weight", i), {attn_hidden_size, n_embd}, 0);
                        layer.time_mix_value = create_tensor(tn(LLM_TENSOR_TIME_MIX_VALUE, "weight", i), {attn_hidden_size, n_embd}, 0);
                        layer.time_mix_receptance = create_tensor(tn(LLM_TENSOR_TIME_MIX_RECEPTANCE, "weight", i), {attn_hidden_size, n_embd}, 0);

                        layer.time_mix_ln = create_tensor(tn(LLM_TENSOR_TIME_MIX_LN, "weight", i), {n_embd}, TENSOR_NOT_REQUIRED);
                        layer.time_mix_ln_b = create_tensor(tn(LLM_TENSOR_TIME_MIX_LN, "bias", i), {n_embd}, TENSOR_NOT_REQUIRED);
                        layer.time_mix_output = create_tensor(tn(LLM_TENSOR_TIME_MIX_OUTPUT, "weight", i), {n_embd, attn_hidden_size}, 0);

                        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);

                        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff}, 0);
                        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd}, 0);
                        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff}, 0);
                    }

                } break;
            case LLM_ARCH_CHAMELEON:
                {
                    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

                    // output
                    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
                    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);
                    // if output is NULL, init from the input tok embed
                    if (output == NULL) {
                        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
                    }

                    for (int i = 0; i < n_layer; ++i) {
                        auto & layer = layers[i];

                        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);
                        layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), {n_embd_head_k, n_head}, 0);
                        layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), {n_embd_head_k, n_head_kv}, 0);
                        layer.attn_q_norm_b = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "bias", i),  {n_embd_head_k, n_head}, TENSOR_NOT_REQUIRED);
                        layer.attn_k_norm_b = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "bias", i),  {n_embd_head_k, n_head_kv}, TENSOR_NOT_REQUIRED);

                        layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_embd}, 0);
                        layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_embd_gqa}, 0);
                        layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_gqa}, 0);
                        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd}, 0);

                        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);

                        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff}, 0);
                        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd}, 0);
                        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff}, 0);
                    }
                } break;
            case LLM_ARCH_WAVTOKENIZER_DEC:
                {
                    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {hparams.n_embd_features, n_vocab}, 0);

                    conv1d   = create_tensor(tn(LLM_TENSOR_CONV1D, "weight"), {7, hparams.n_embd_features, hparams.posnet.n_embd}, 0);
                    conv1d_b = create_tensor(tn(LLM_TENSOR_CONV1D, "bias"),   {1, hparams.posnet.n_embd}, 0);

                    // posnet
                    {
                        const int64_t n_embd = hparams.posnet.n_embd;

                        for (uint32_t i = 0; i < hparams.posnet.n_layer; ++i) {
                            auto & layer = layers[i].posnet;

                            // posnet:
                            //
                            //  - resnet
                            //  - resnet
                            //  - attn
                            //  - resnet
                            //  - resnet
                            //  - norm
                            //
                            switch (i) {
                                case 0:
                                case 1:
                                case 3:
                                case 4:
                                    {
                                        layer.norm1   = create_tensor(tn(LLM_TENSOR_POS_NET_NORM1, "weight", i), {1, n_embd}, 0);
                                        layer.norm1_b = create_tensor(tn(LLM_TENSOR_POS_NET_NORM1, "bias",   i), {1, n_embd}, 0);

                                        layer.conv1   = create_tensor(tn(LLM_TENSOR_POS_NET_CONV1, "weight", i), {3, n_embd, n_embd}, 0);
                                        layer.conv1_b = create_tensor(tn(LLM_TENSOR_POS_NET_CONV1, "bias",   i), {1, n_embd}, 0);

                                        layer.norm2   = create_tensor(tn(LLM_TENSOR_POS_NET_NORM2, "weight", i), {1, n_embd}, 0);
                                        layer.norm2_b = create_tensor(tn(LLM_TENSOR_POS_NET_NORM2, "bias",   i), {1, n_embd}, 0);

                                        layer.conv2   = create_tensor(tn(LLM_TENSOR_POS_NET_CONV2, "weight", i), {3, n_embd, n_embd}, 0);
                                        layer.conv2_b = create_tensor(tn(LLM_TENSOR_POS_NET_CONV2, "bias",   i), {1, n_embd}, 0);
                                    } break;
                                case 2:
                                    {
                                        layer.attn_norm   = create_tensor(tn(LLM_TENSOR_POS_NET_ATTN_NORM, "weight", i), {1, n_embd}, 0);
                                        layer.attn_norm_b = create_tensor(tn(LLM_TENSOR_POS_NET_ATTN_NORM, "bias",   i), {1, n_embd}, 0);

                                        layer.attn_q      = create_tensor(tn(LLM_TENSOR_POS_NET_ATTN_Q,    "weight", i), {1, n_embd, n_embd}, 0);
                                        layer.attn_q_b    = create_tensor(tn(LLM_TENSOR_POS_NET_ATTN_Q,    "bias",   i), {1, n_embd}, 0);

                                        layer.attn_k      = create_tensor(tn(LLM_TENSOR_POS_NET_ATTN_K,    "weight", i), {1, n_embd, n_embd}, 0);
                                        layer.attn_k_b    = create_tensor(tn(LLM_TENSOR_POS_NET_ATTN_K,    "bias",   i), {1, n_embd}, 0);

                                        layer.attn_v      = create_tensor(tn(LLM_TENSOR_POS_NET_ATTN_V,    "weight", i), {1, n_embd, n_embd}, 0);
                                        layer.attn_v_b    = create_tensor(tn(LLM_TENSOR_POS_NET_ATTN_V,    "bias",   i), {1, n_embd}, 0);

                                        layer.attn_o      = create_tensor(tn(LLM_TENSOR_POS_NET_ATTN_OUT,  "weight", i), {1, n_embd, n_embd}, 0);
                                        layer.attn_o_b    = create_tensor(tn(LLM_TENSOR_POS_NET_ATTN_OUT,  "bias",   i), {1, n_embd}, 0);
                                    } break;
                                case 5:
                                    {
                                        layer.norm   = create_tensor(tn(LLM_TENSOR_POS_NET_ATTN_NORM, "weight", i), {1, n_embd}, 0);
                                        layer.norm_b = create_tensor(tn(LLM_TENSOR_POS_NET_ATTN_NORM, "bias",   i), {1, n_embd}, 0);
                                    } break;
                                default: GGML_ABORT("unknown posnet layer");
                            };
                        }
                    }

                    GGML_ASSERT(hparams.posnet.n_embd == hparams.convnext.n_embd);

                    tok_norm   = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD_NORM, "weight"), {hparams.posnet.n_embd}, 0);
                    tok_norm_b = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD_NORM, "bias"),   {hparams.posnet.n_embd}, 0);

                    // convnext
                    {
                        const int64_t n_embd = hparams.convnext.n_embd;

                        for (uint32_t i = 0; i < hparams.convnext.n_layer; ++i) {
                            auto & layer = layers[i].convnext;

                            layer.dw     = create_tensor(tn(LLM_TENSOR_CONVNEXT_DW,    "weight", i), {7, 1, n_embd}, 0);
                            layer.dw_b   = create_tensor(tn(LLM_TENSOR_CONVNEXT_DW,    "bias",   i), {1, n_embd}, 0);

                            layer.norm   = create_tensor(tn(LLM_TENSOR_CONVNEXT_NORM,  "weight", i), {n_embd}, 0);
                            layer.norm_b = create_tensor(tn(LLM_TENSOR_CONVNEXT_NORM,  "bias",   i), {n_embd}, 0);

                            layer.pw1    = create_tensor(tn(LLM_TENSOR_CONVNEXT_PW1,   "weight", i), {n_embd, n_ff}, 0);
                            layer.pw1_b  = create_tensor(tn(LLM_TENSOR_CONVNEXT_PW1,   "bias",   i), {n_ff}, 0);

                            layer.pw2    = create_tensor(tn(LLM_TENSOR_CONVNEXT_PW2,   "weight", i), {n_ff, n_embd}, 0);
                            layer.pw2_b  = create_tensor(tn(LLM_TENSOR_CONVNEXT_PW2,   "bias",   i), {n_embd}, 0);

                            layer.gamma  = create_tensor(tn(LLM_TENSOR_CONVNEXT_GAMMA, "weight", i), {n_embd}, 0);
                        }

                        // output
                        output_norm   = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
                        output_norm_b = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "bias"),   {n_embd}, 0);
                    }

                    output   = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), {hparams.convnext.n_embd, n_embd}, 0);
                    output_b = create_tensor(tn(LLM_TENSOR_OUTPUT, "bias"),   {n_embd}, 0);
                } break;
            case LLM_ARCH_BAILINGMOE:
                {
                    const int64_t n_ff_exp            = hparams.n_ff_exp;
                    const int64_t n_expert_shared     = hparams.n_expert_shared;

                    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

                    // output
                    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
                    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, 0);

                    for (int i = 0; i < n_layer; ++i) {
                        auto & layer = layers[i];

                        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

                        layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_head * n_rot}, 0);
                        layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_head_kv * n_rot}, 0);
                        layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_head_kv * n_rot}, 0);
                        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_head * n_rot, n_embd}, 0);
                        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);

                        layer.ffn_gate_inp = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP, "weight", i), {n_embd, n_expert}, 0);

                        if (n_expert == 0) {
                            throw std::runtime_error("n_expert must be > 0");
                        }
                        if (n_expert_used == 0) {
                            throw std::runtime_error("n_expert_used must be > 0");
                        }

                        layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {  n_embd, n_ff_exp, n_expert}, 0);
                        layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {n_ff_exp,   n_embd, n_expert}, 0);
                        layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {  n_embd, n_ff_exp, n_expert}, 0);

                        layer.ffn_gate_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", i), {n_embd, n_ff_exp * n_expert_shared}, 0);
                        layer.ffn_down_shexp = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i), {        n_ff_exp * n_expert_shared, n_embd}, 0);
                        layer.ffn_up_shexp   = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,   "weight", i), {n_embd, n_ff_exp * n_expert_shared}, 0);
                    }
                } break;
            case LLM_ARCH_DOTS1:
                {
                    const int64_t n_ff_exp        = hparams.n_ff_exp;
                    const int64_t n_expert_shared = hparams.n_expert_shared;

                    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

                    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
                    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, 0);

                    for (int i = 0; i < n_layer; ++i) {
                        auto & layer = layers[i];

                        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

                        layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_embd_head_k * n_head}, 0);
                        layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_embd_head_k * n_head}, 0);
                        layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_head_k * n_head}, 0);
                        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head_k * n_head, n_embd}, 0);

                        layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), {n_embd_head_k}, 0);
                        layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), {n_embd_head_k}, 0);

                        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);

                        if (i < (int) hparams.n_layer_dense_lead) {
                            layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff}, 0);
                            layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd}, 0);
                            layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff}, 0);
                        } else {
                            layer.ffn_gate_inp = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP, "weight", i), {n_embd, n_expert}, 0);
                            layer.ffn_exp_probs_b = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias", i), {n_expert}, TENSOR_NOT_REQUIRED);

                            if (n_expert == 0) {
                                throw std::runtime_error("n_expert must be > 0");
                            }
                            if (n_expert_used == 0) {
                                throw std::runtime_error("n_expert_used must be > 0");
                            }

                            // MoE branch
                            layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {  n_embd, n_ff_exp, n_expert}, 0);
                            layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {n_ff_exp,   n_embd, n_expert}, 0);
                            layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {  n_embd, n_ff_exp, n_expert}, 0);

                            // Shared expert branch
                            layer.ffn_gate_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", i), {n_embd, n_ff_exp * n_expert_shared}, 0);
                            layer.ffn_down_shexp = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i), {        n_ff_exp * n_expert_shared, n_embd}, 0);
                            layer.ffn_up_shexp   = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,   "weight", i), {n_embd, n_ff_exp * n_expert_shared}, 0);
                        }
                    }
                } break;
            case LLM_ARCH_ARCEE:
                {
                    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

                    // output
                    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
                    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);

                    // if output is NULL, init from the input tok embed
                    if (output == NULL) {
                        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
                    }

                    for (int i = 0; i < n_layer; ++i) {
                        auto & layer = layers[i];

                        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

                        layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_embd_head_k * n_head}, 0);
                        layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_embd_k_gqa}, 0);
                        layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_v_gqa}, 0);
                        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head_k * n_head, n_embd}, 0);

                        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);

                        layer.rope_freqs = create_tensor(tn(LLM_TENSOR_ROPE_FREQS, "weight", i), {n_rot/2}, TENSOR_NOT_REQUIRED | (i != 0 ? TENSOR_DUPLICATED : 0));

                        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd}, 0);
                        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff}, 0);
                    }
                } break;
            case LLM_ARCH_ERNIE4_5:
            case LLM_ARCH_ERNIE4_5_MOE:
                {
                    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

                    // output
                    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
                    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);
                    // if output is NULL, init from the input tok embed
                    if (output == NULL) {
                        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
                    }

                    for (int i = 0; i < n_layer; ++i) {
                        auto & layer = layers[i];

                        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

                        layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_embd_head_k * n_head}, 0);
                        layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_embd_gqa}, 0);
                        layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_gqa}, 0);
                        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head_k * n_head, n_embd}, 0);

                        // optional bias tensors
                        layer.bq = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "bias", i), {n_embd},     TENSOR_NOT_REQUIRED);
                        layer.bk = create_tensor(tn(LLM_TENSOR_ATTN_K,   "bias", i), {n_embd_gqa}, TENSOR_NOT_REQUIRED);
                        layer.bv = create_tensor(tn(LLM_TENSOR_ATTN_V,   "bias", i), {n_embd_gqa}, TENSOR_NOT_REQUIRED);
                        layer.bo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "bias", i), {n_embd},     TENSOR_NOT_REQUIRED);

                        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);

                        if (arch == LLM_ARCH_ERNIE4_5_MOE && static_cast<uint32_t>(i) >= hparams.n_layer_dense_lead) { // MoE layers
                            int n_ff_exp = hparams.n_ff_exp;

                            layer.ffn_gate_inp  = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,  "weight", i), {n_embd, n_expert}, 0);
                            layer.ffn_exp_probs_b = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias", i), {n_expert}, TENSOR_NOT_REQUIRED);
                            layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {n_embd,   n_ff_exp, n_expert}, TENSOR_NOT_REQUIRED);
                            layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {  n_ff_exp, n_embd, n_expert}, 0);
                            layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {n_embd,   n_ff_exp, n_expert}, 0);

                            // Shared expert (if present)
                            if (hparams.n_ff_shexp > 0) {
                                layer.ffn_gate_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", i), {    n_embd, hparams.n_ff_shexp}, 0);
                                layer.ffn_down_shexp = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i), {hparams.n_ff_shexp, n_embd    }, 0);
                                layer.ffn_up_shexp   = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,   "weight", i), {    n_embd, hparams.n_ff_shexp}, 0);
                            }
                        } else { // Dense layers
                            layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff}, 0);
                            layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd}, 0);
                            layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff}, 0);
                        }
                    }
                } break;
            case LLM_ARCH_FALCON_H1:
                {
                    // Common
                    const int64_t hidden_size = hparams.n_embd; // hidden_size

                    // mamba2 Mixer SSM params
                    const int64_t ssm_conv_kernel_size  = hparams.ssm_d_conv; // ssm_conv_kernel_size
                    const int64_t ssm_n_groups          = hparams.ssm_n_group; // ssm_n_groups
                    const int64_t ssm_state_size        = hparams.ssm_d_state; // ssm_state_size
                    const int64_t ssm_intermediate_size = hparams.ssm_d_inner; // TODO expand
                    const int64_t ssm_num_heads         = hparams.ssm_dt_rank; // ssm_num_heads
                    const int64_t ssm_conv_dim          = ssm_intermediate_size + 2 * ssm_n_groups * ssm_state_size;
                    const int64_t ssm_projection_size   = ssm_intermediate_size + ssm_conv_dim + ssm_num_heads;

                    // attn params
                    const int64_t attn_num_attention_head = hparams.n_head(0); // rename to: attn_num_attention_head
                    const int64_t attn_num_key_value_head = hparams.n_head_kv(0);

                    // ffn params
                    const int64_t ffn_intermediate_size = hparams.n_ff(0);

                    // embeddings
                    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {hidden_size, n_vocab}, 0);

                    // output
                    output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), {hidden_size, n_vocab}, TENSOR_NOT_REQUIRED);
                    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {hidden_size}, 0);

                    // if output is NULL, init from the input tok embed
                    if (output == NULL) {
                        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {hidden_size, n_vocab}, TENSOR_DUPLICATED);
                    }

                    for (int i = 0; i < n_layer; ++i) {
                        auto & layer = layers[i];

                        /*SSM LAYERS*/
                        // ssm in
                        layer.ssm_in = create_tensor(tn(LLM_TENSOR_SSM_IN, "weight", i), {hidden_size, ssm_projection_size}, 0);
                        // ssm 1d conv
                        layer.ssm_conv1d = create_tensor(tn(LLM_TENSOR_SSM_CONV1D, "weight", i), {ssm_conv_kernel_size, ssm_conv_dim}, 0);
                        layer.ssm_conv1d_b = create_tensor(tn(LLM_TENSOR_SSM_CONV1D, "bias", i), {ssm_conv_dim}, TENSOR_NOT_REQUIRED);
                        // ssm_dt
                        layer.ssm_dt_b = create_tensor(tn(LLM_TENSOR_SSM_DT, "bias", i), {ssm_num_heads}, 0);
                        // no "weight" suffix for these
                        layer.ssm_a = create_tensor(tn(LLM_TENSOR_SSM_A, i), {1, ssm_num_heads}, 0);
                        layer.ssm_d = create_tensor(tn(LLM_TENSOR_SSM_D, i), {1, ssm_num_heads}, 0);
                        // ssm_norm
                        layer.ssm_norm = create_tensor(tn(LLM_TENSOR_SSM_NORM, "weight", i), {ssm_intermediate_size / ssm_n_groups, ssm_n_groups}, TENSOR_NOT_REQUIRED);
                        // out_proj
                        layer.ssm_out = create_tensor(tn(LLM_TENSOR_SSM_OUT, "weight", i), {ssm_intermediate_size, hidden_size}, 0);

                        /*ATTENTION LAYERS*/
                        // attention layers (with optional bias)
                        layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "weight", i), {hidden_size, n_embd_head_k * attn_num_attention_head}, 0);
                        layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K,   "weight", i), {hidden_size, attn_num_key_value_head * n_embd_head_k}, 0);
                        layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V,   "weight", i), {hidden_size, attn_num_key_value_head * n_embd_head_v}, 0);
                        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head_k * attn_num_attention_head, hidden_size}, 0);
                        layer.bq = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "bias", i), {hidden_size}, TENSOR_NOT_REQUIRED);
                        layer.bk = create_tensor(tn(LLM_TENSOR_ATTN_K,   "bias", i), {attn_num_key_value_head * n_embd_head_k}, TENSOR_NOT_REQUIRED);
                        layer.bv = create_tensor(tn(LLM_TENSOR_ATTN_V,   "bias", i), {attn_num_key_value_head * n_embd_head_v}, TENSOR_NOT_REQUIRED);
                        layer.bo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "bias", i), {hidden_size}, TENSOR_NOT_REQUIRED);
                        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {hidden_size}, 0);


                        // feed forward (w/ optional biases)
                        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, i), {hidden_size}, 0);
                        layer.rope_freqs = create_tensor(tn(LLM_TENSOR_ROPE_FREQS, "weight", i), {n_rot/2}, TENSOR_NOT_REQUIRED | (i != 0 ? TENSOR_DUPLICATED : 0));
                        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {hidden_size,   ffn_intermediate_size}, 0);
                        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  ffn_intermediate_size, hidden_size}, 0);
                        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {hidden_size,   ffn_intermediate_size}, 0);

                        layer.ffn_gate_b = create_tensor(tn(LLM_TENSOR_FFN_GATE, "bias", i), {ffn_intermediate_size}, TENSOR_NOT_REQUIRED);
                        layer.ffn_down_b = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "bias", i), {hidden_size}, TENSOR_NOT_REQUIRED);
                        layer.ffn_up_b   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "bias", i), {ffn_intermediate_size}, TENSOR_NOT_REQUIRED);
                    }
                } break;
            case LLM_ARCH_HUNYUAN_MOE:
                {
                    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

                    // output
                    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
                    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);
                    // if output is NULL, init from the input tok embed
                    if (output == NULL) {
                        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
                    }

                    for (int i = 0; i < n_layer; ++i) {
                        auto & layer = layers[i];

                        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

                        layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_embd_head_k * n_head}, 0);
                        layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_embd_k_gqa}, 0);
                        layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_v_gqa}, 0);
                        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head_k * n_head, n_embd}, 0);

                        layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), {n_embd_head_k}, 0);
                        layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), {n_embd_head_k}, 0);

                        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);

                        layer.ffn_gate_inp  = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,  "weight", i), {n_embd, n_expert}, 0);
                        layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {n_embd,   n_ff, n_expert}, 0);
                        layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {  n_ff, n_embd, n_expert}, 0);
                        layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {n_embd,   n_ff, n_expert}, 0);

                        layer.ffn_gate_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", i), {n_embd, hparams.n_ff_shexp}, 0);
                        layer.ffn_up_shexp   = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,   "weight", i), {n_embd, hparams.n_ff_shexp}, 0);
                        layer.ffn_down_shexp = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i), {hparams.n_ff_shexp, n_embd}, 0);
                    }
                } break;
            case LLM_ARCH_HUNYUAN_DENSE:
                {
                    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

                    // output
                    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
                    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);
                    // if output is NULL, init from the input tok embed
                    if (output == NULL) {
                        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
                    }

                    for (int i = 0; i < n_layer; ++i) {
                        auto & layer = layers[i];

                        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

                        layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_embd_head_k * n_head}, 0);
                        layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_embd_k_gqa}, 0);
                        layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_v_gqa}, 0);
                        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head_k * n_head, n_embd}, 0);

                        layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), {n_embd_head_k}, 0);
                        layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), {n_embd_head_k}, 0);

                        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);

                        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff}, 0);
                        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd}, 0);
                        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff}, 0);

                    }
                } break;
            case LLM_ARCH_SMOLLM3:
                {
                    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

                    // output
                    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
                    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);

                    // if output is NULL, init from the input tok embed
                    if (output == NULL) {
                        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
                    }

                    for (int i = 0; i < n_layer; ++i) {
                        auto & layer = layers[i];

                        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

                        layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_embd_head_k * n_head}, 0);
                        layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_embd_k_gqa}, 0);
                        layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_v_gqa}, 0);
                        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head_k * n_head, n_embd}, 0);

                        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);
                        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff}, 0);
                        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd}, 0);
                        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff}, 0);
                    }
                } break;
            case LLM_ARCH_OPENAI_MOE:
                {
                    const int64_t n_ff_exp = hparams.n_ff_exp;

                    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

                    // output
                    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
                    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, 0);

                    for (int i = 0; i < n_layer; ++i) {
                        auto & layer = layers[i];

                        layer.attn_norm      = create_tensor(tn(LLM_TENSOR_ATTN_NORM,      "weight", i), {n_embd}, 0);
                        layer.attn_post_norm = create_tensor(tn(LLM_TENSOR_ATTN_POST_NORM, "weight", i), {n_embd}, 0);

                        layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_head * n_rot}, 0);
                        layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_head_kv * n_rot}, 0);
                        layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_head_kv * n_rot}, 0);
                        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_head * n_rot, n_embd}, 0);

                        layer.attn_sinks = create_tensor(tn(LLM_TENSOR_ATTN_SINKS, "weight", i), {n_head}, 0);

                        layer.ffn_gate_inp  = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,  "weight", i), {  n_embd, n_expert}, 0);
                        layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {  n_embd, n_ff_exp, n_expert}, 0);
                        layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {n_ff_exp,   n_embd, n_expert}, 0);
                        layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {  n_embd, n_ff_exp, n_expert}, 0);

                        // bias
                        layer.bq = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "bias", i), {n_head * n_rot}, 0);
                        layer.bk = create_tensor(tn(LLM_TENSOR_ATTN_K,   "bias", i), {n_head_kv * n_rot}, 0);
                        layer.bv = create_tensor(tn(LLM_TENSOR_ATTN_V,   "bias", i), {n_head_kv * n_rot}, 0);
                        layer.bo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "bias", i), {n_embd}, 0);

                        layer.ffn_gate_inp_b  = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,  "bias", i), {n_expert}, 0);
                        layer.ffn_gate_exps_b = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "bias", i), {n_ff_exp, n_expert}, 0);
                        layer.ffn_down_exps_b = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "bias", i), {  n_embd, n_expert}, 0);
                        layer.ffn_up_exps_b   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "bias", i), {n_ff_exp, n_expert}, 0);
                    }
                } break;
            case LLM_ARCH_LFM2:
                {
                    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);
                    tok_norm = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD_NORM, "weight"), {n_embd}, 0);

                    for (int i = 0; i < n_layer; ++i) {
                        auto & layer = layers[i];
                        // ffn is same for transformer and conv layers
                        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);
                        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff}, 0);
                        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd}, 0);
                        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff}, 0);

                        // for operator_norm
                        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

                        if (!hparams.is_recurrent(i)) {
                            layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), {n_embd_head_k}, 0);
                            layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), {n_embd_head_k}, 0);
                            GGML_ASSERT(n_embd_v_gqa == n_embd_k_gqa);

                            layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q, "weight", i), {n_embd, n_embd}, 0);
                            layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K, "weight", i), {n_embd, hparams.n_embd_k_gqa(i)}, 0);
                            layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V, "weight", i), {n_embd, hparams.n_embd_v_gqa(i)}, 0);

                            layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd}, 0);
                        } else {
                            layer.shortconv.conv     = create_tensor(tn(LLM_TENSOR_SHORTCONV_CONV,    "weight", i), {hparams.n_shortconv_l_cache, n_embd}, 0);
                            layer.shortconv.in_proj  = create_tensor(tn(LLM_TENSOR_SHORTCONV_INPROJ,  "weight", i), {n_embd, 3 * n_embd}, 0);
                            layer.shortconv.out_proj = create_tensor(tn(LLM_TENSOR_SHORTCONV_OUTPROJ, "weight", i), {n_embd, n_embd}, 0);
                        }
                    }
                } break;
            case LLM_ARCH_SMALLTHINKER:
                {
                    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, 0);

                    // output
                    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), { n_embd }, 0);
                    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);

                    // if output is NULL, init from the input tok embed
                    if (output == NULL) {
                        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
                    }

                    for (int i = 0; i < n_layer; ++i) {
                        auto & layer = layers[i];

                        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), { n_embd }, 0);

                        layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q, "weight", i), { n_embd, n_embd_head_k * n_head }, 0);
                        layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K, "weight", i), { n_embd, n_embd_gqa }, 0);
                        layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V, "weight", i), { n_embd, n_embd_gqa }, 0);
                        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), { n_embd_head_k * n_head, n_embd }, 0);

                        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), { n_embd }, 0);

                        GGML_ASSERT(n_expert > 0 && "n_expert must be > 0 for SMALLTHINKER");
                        GGML_ASSERT(n_expert_used > 0 && "n_expert_used must be > 0 for SMALLTHINKER");

                        // MoE branch
                        const int64_t n_ff_exp = hparams.n_ff_exp;
                        layer.ffn_gate_inp  = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP, "weight", i), { n_embd, n_expert }, 0);
                        layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), { n_embd, n_ff_exp, n_expert }, 0);
                        layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), { n_ff_exp, n_embd, n_expert }, 0);
                        layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS, "weight", i), { n_embd, n_ff_exp, n_expert }, 0);
                    }
                } break;
            default:
                throw std::runtime_error("unknown architecture");
        }

        if (n_moved_tensors > 0) {
            LLAMA_LOG_DEBUG("%s: tensor '%s' (%s) (and %d others) cannot be used with preferred buffer type %s, using %s instead\n",
                __func__, first_moved_tensor->name, ggml_type_name(first_moved_tensor->type), n_moved_tensors - 1,
                ggml_backend_buft_name(first_moved_from_buft), ggml_backend_buft_name(first_moved_to_buft));
        }
    }

    ml.done_getting_tensors();

    ml.init_mappings(true, use_mlock ? &pimpl->mlock_mmaps : nullptr);
    pimpl->mappings.reserve(ml.mappings.size());

    // create the backend buffers
    std::vector<std::pair<ggml_context *, llama_buf_map>> ctx_bufs;
    ctx_bufs.reserve(ctx_map.size());

    // Ensure we have enough capacity for the maximum backend buffer we will potentially create
    const size_t n_max_backend_buffer = ctx_map.size() * ml.files.size();
    pimpl->bufs.reserve(n_max_backend_buffer);

    for (auto & it : ctx_map) {
        ggml_backend_buffer_type_t buft = it.first;
        ggml_context * ctx              = it.second;

        // skip contexts without tensors
        if (ggml_get_first_tensor(ctx) == nullptr) {
            continue;
        }

        llama_buf_map buf_map;
        buf_map.reserve(n_max_backend_buffer);

        // check if it is possible to use buffer_from_host_ptr with this buffer type
        ggml_backend_dev_t dev = ggml_backend_buft_get_device(buft);
        if (!dev) {
            // FIXME: workaround for CPU backend buft having a NULL device
            dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
            if (!dev) {
                throw std::runtime_error(format("%s: no CPU backend found", __func__));
            }
        }
        ggml_backend_dev_props props;
        ggml_backend_dev_get_props(dev, &props);
        bool buffer_from_host_ptr_supported = props.caps.buffer_from_host_ptr;
        bool is_default_buft = buft == ggml_backend_dev_buffer_type(dev);

        if (ml.use_mmap && use_mmap_buffer && buffer_from_host_ptr_supported && is_default_buft) {
            for (uint32_t idx = 0; idx < ml.files.size(); idx++) {
                // only the mmap region containing the tensors in the model is mapped to the backend buffer
                // this is important for metal with apple silicon: if the entire model could be mapped to a metal buffer, then we could just use metal for all layers
                // this allows using partial offloading when the model size exceeds the metal buffer size, but not the RAM size
                void * addr = nullptr;
                size_t first, last; // NOLINT
                ml.get_mapping_range(&first, &last, &addr, idx, ctx);
                if (first >= last) {
                    continue;
                }
                const size_t max_size = ggml_get_max_tensor_size(ctx);
                ggml_backend_buffer_t buf = ggml_backend_dev_buffer_from_host_ptr(dev, (char *) addr + first, last - first, max_size);
                if (buf == nullptr) {
                    throw std::runtime_error(format("unable to allocate %s buffer", ggml_backend_buft_name(buft)));
                }
                pimpl->bufs.emplace_back(buf);
                buf_map.emplace(idx, buf);
            }
        }
        else {
            ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
            if (buf == nullptr) {
                throw std::runtime_error(format("unable to allocate %s buffer", ggml_backend_buft_name(buft)));
            }
            pimpl->bufs.emplace_back(buf);
            if (use_mlock && ggml_backend_buffer_is_host(buf)) {
                pimpl->mlock_bufs.emplace_back(new llama_mlock);
                auto & mlock_buf = pimpl->mlock_bufs.back();
                mlock_buf->init   (ggml_backend_buffer_get_base(buf));
                mlock_buf->grow_to(ggml_backend_buffer_get_size(buf));
            }
            for (uint32_t idx = 0; idx < ml.files.size(); idx++) {
                buf_map.emplace(idx, buf);
            }
        }

        if (pimpl->bufs.empty()) {
            throw std::runtime_error("failed to allocate buffer");
        }

        for (auto & buf : buf_map) {
            // indicate that this buffer contains weights
            // this is used by ggml_backend_sched to improve op scheduling: ops that use a weight are preferably scheduled to the backend that contains the weight
            ggml_backend_buffer_set_usage(buf.second, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        }

        ctx_bufs.emplace_back(ctx, buf_map);
    }

    if (llama_supports_gpu_offload()) {
        const int n_gpu = std::min(n_gpu_layers, int(hparams.n_layer));

        LLAMA_LOG_INFO("%s: offloading %d repeating layers to GPU\n", __func__, n_gpu);
        if (n_gpu_layers > (int) hparams.n_layer) {
            LLAMA_LOG_INFO("%s: offloading output layer to GPU\n", __func__);
        }

        const int max_backend_supported_layers = hparams.n_layer + 1;
        const int max_offloadable_layers       = hparams.n_layer + 1;

        LLAMA_LOG_INFO("%s: offloaded %d/%d layers to GPU\n", __func__, std::min(n_gpu_layers, max_offloadable_layers), max_backend_supported_layers);
    }

    // print memory requirements per buffer type
    for (auto & buf : pimpl->bufs) {
        LLAMA_LOG_INFO("%s: %12s model buffer size = %8.2f MiB\n", __func__, ggml_backend_buffer_name(buf.get()), ggml_backend_buffer_get_size(buf.get()) / 1024.0 / 1024.0);
    }

    // populate tensors_by_name
    for (auto & ctx : pimpl->ctxs) {
        for (auto * cur = ggml_get_first_tensor(ctx.get()); cur != NULL; cur = ggml_get_next_tensor(ctx.get(), cur)) {
            tensors_by_name.emplace_back(ggml_get_name(cur), cur);
        }
    }

    // load tensor data
    for (auto & it : ctx_bufs) {
        ggml_context * ctx = it.first;
        auto & bufs = it.second;
        if (!ml.load_all_data(ctx, bufs, use_mlock ? &pimpl->mlock_mmaps : NULL, params.progress_callback, params.progress_callback_user_data)) {
            return false;
        }
    }

    if (use_mmap_buffer) {
        for (auto & mapping : ml.mappings) {
            pimpl->mappings.emplace_back(std::move(mapping));
        }
    }

    return true;
}

std::string llama_model::arch_name() const {
    return llm_arch_name(arch);
}

std::string llama_model::type_name() const {
    return llm_type_name(type);
}

std::string llama_model::desc() const {
    return pimpl->desc_str;
}

size_t llama_model::size() const {
    return pimpl->n_bytes;
}

size_t llama_model::n_tensors() const {
    return tensors_by_name.size();
}

size_t llama_model::n_devices() const {
    return devices.size();
}

uint64_t llama_model::n_elements() const {
    return pimpl->n_elements;
}

void llama_model::print_info() const {
    const std::string rope_scaling_type = llama_rope_scaling_type_name(hparams.rope_scaling_type_train);

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
    LLAMA_LOG_INFO("%s: arch             = %s\n",     __func__, arch_name().c_str());
    LLAMA_LOG_INFO("%s: vocab_only       = %d\n",     __func__, hparams.vocab_only);

    if (!hparams.vocab_only) {
        LLAMA_LOG_INFO("%s: n_ctx_train      = %u\n",     __func__, hparams.n_ctx_train);
        LLAMA_LOG_INFO("%s: n_embd           = %u\n",     __func__, hparams.n_embd);
        LLAMA_LOG_INFO("%s: n_layer          = %u\n",     __func__, hparams.n_layer);
        LLAMA_LOG_INFO("%s: n_head           = %s\n",     __func__, print_f([&](uint32_t il) { return hparams.n_head(il);    }, hparams.n_layer).c_str());
        LLAMA_LOG_INFO("%s: n_head_kv        = %s\n",     __func__, print_f([&](uint32_t il) { return hparams.n_head_kv(il); }, hparams.n_layer).c_str());
        LLAMA_LOG_INFO("%s: n_rot            = %u\n",     __func__, hparams.n_rot);
        LLAMA_LOG_INFO("%s: n_swa            = %u\n",     __func__, hparams.n_swa);
        LLAMA_LOG_INFO("%s: is_swa_any       = %u\n",     __func__, hparams.is_swa_any());
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
        LLAMA_LOG_INFO("%s: f_attn_scale     = %.1e\n",   __func__, hparams.f_attention_scale);
        LLAMA_LOG_INFO("%s: n_ff             = %s\n",     __func__, print_f([&](uint32_t il) { return hparams.n_ff(il); }, hparams.n_layer).c_str());
        LLAMA_LOG_INFO("%s: n_expert         = %u\n",     __func__, hparams.n_expert);
        LLAMA_LOG_INFO("%s: n_expert_used    = %u\n",     __func__, hparams.n_expert_used);
        LLAMA_LOG_INFO("%s: causal attn      = %d\n",     __func__, hparams.causal_attn);
        LLAMA_LOG_INFO("%s: pooling type     = %d\n",     __func__, hparams.pooling_type);
        LLAMA_LOG_INFO("%s: rope type        = %d\n",     __func__, hparams.rope_type);
        LLAMA_LOG_INFO("%s: rope scaling     = %s\n",     __func__, rope_scaling_type.c_str());
        LLAMA_LOG_INFO("%s: freq_base_train  = %.1f\n",   __func__, hparams.rope_freq_base_train);
        LLAMA_LOG_INFO("%s: freq_scale_train = %g\n",     __func__, hparams.rope_freq_scale_train);
        LLAMA_LOG_INFO("%s: n_ctx_orig_yarn  = %u\n",     __func__, hparams.n_ctx_orig_yarn);
        LLAMA_LOG_INFO("%s: rope_finetuned   = %s\n",     __func__, hparams.rope_finetuned ? "yes" : "unknown");
        if (!classifier_labels.empty()) {
            LLAMA_LOG_INFO("%s: n_cls_out        = %u\n", __func__, hparams.n_cls_out);

            size_t i = 0;
            for (auto label : classifier_labels) {
                LLAMA_LOG_INFO("%s: cls_label[%2zu]    = %s\n", __func__, i++, label.c_str());
            }
        }
    }

    if (arch == LLM_ARCH_MAMBA ||
        arch == LLM_ARCH_MAMBA2 ||
        arch == LLM_ARCH_JAMBA ||
        arch == LLM_ARCH_FALCON_H1 ||
        arch == LLM_ARCH_PLAMO2 ||
        arch == LLM_ARCH_GRANITE_HYBRID) {
        LLAMA_LOG_INFO("%s: ssm_d_conv       = %u\n",     __func__, hparams.ssm_d_conv);
        LLAMA_LOG_INFO("%s: ssm_d_inner      = %u\n",     __func__, hparams.ssm_d_inner);
        LLAMA_LOG_INFO("%s: ssm_d_state      = %u\n",     __func__, hparams.ssm_d_state);
        LLAMA_LOG_INFO("%s: ssm_dt_rank      = %u\n",     __func__, hparams.ssm_dt_rank);
        LLAMA_LOG_INFO("%s: ssm_n_group      = %u\n",     __func__, hparams.ssm_n_group);
        LLAMA_LOG_INFO("%s: ssm_dt_b_c_rms   = %d\n",     __func__, hparams.ssm_dt_b_c_rms);
    }

    LLAMA_LOG_INFO("%s: model type       = %s\n",     __func__, type_name().c_str());
    if (pimpl->n_elements >= 1e12) {
        LLAMA_LOG_INFO("%s: model params     = %.2f T\n", __func__, pimpl->n_elements*1e-12);
    } else if (pimpl->n_elements >= 1e9) {
        LLAMA_LOG_INFO("%s: model params     = %.2f B\n", __func__, pimpl->n_elements*1e-9);
    } else if (pimpl->n_elements >= 1e6) {
        LLAMA_LOG_INFO("%s: model params     = %.2f M\n", __func__, pimpl->n_elements*1e-6);
    } else {
        LLAMA_LOG_INFO("%s: model params     = %.2f K\n", __func__, pimpl->n_elements*1e-3);
    }

    // general kv
    LLAMA_LOG_INFO("%s: general.name     = %s\n",    __func__, name.c_str());

    if (arch == LLM_ARCH_DEEPSEEK) {
        LLAMA_LOG_INFO("%s: n_layer_dense_lead   = %d\n",     __func__, hparams.n_layer_dense_lead);
        LLAMA_LOG_INFO("%s: n_ff_exp             = %d\n",     __func__, hparams.n_ff_exp);
        LLAMA_LOG_INFO("%s: n_expert_shared      = %d\n",     __func__, hparams.n_expert_shared);
        LLAMA_LOG_INFO("%s: expert_weights_scale = %.1f\n",   __func__, hparams.expert_weights_scale);
    }

    if (arch == LLM_ARCH_DEEPSEEK2) {
        LLAMA_LOG_INFO("%s: n_layer_dense_lead   = %d\n",     __func__, hparams.n_layer_dense_lead);
        LLAMA_LOG_INFO("%s: n_lora_q             = %d\n",     __func__, hparams.n_lora_q);
        LLAMA_LOG_INFO("%s: n_lora_kv            = %d\n",     __func__, hparams.n_lora_kv);
        LLAMA_LOG_INFO("%s: n_embd_head_k_mla    = %d\n",     __func__, hparams.n_embd_head_k_mla);
        LLAMA_LOG_INFO("%s: n_embd_head_v_mla    = %d\n",     __func__, hparams.n_embd_head_v_mla);
        LLAMA_LOG_INFO("%s: n_ff_exp             = %d\n",     __func__, hparams.n_ff_exp);
        LLAMA_LOG_INFO("%s: n_expert_shared      = %d\n",     __func__, hparams.n_expert_shared);
        LLAMA_LOG_INFO("%s: expert_weights_scale = %.1f\n",   __func__, hparams.expert_weights_scale);
        LLAMA_LOG_INFO("%s: expert_weights_norm  = %d\n",     __func__, hparams.expert_weights_norm);
        LLAMA_LOG_INFO("%s: expert_gating_func   = %s\n",     __func__, llama_expert_gating_func_name((llama_expert_gating_func_type) hparams.expert_gating_func));
        LLAMA_LOG_INFO("%s: rope_yarn_log_mul    = %.4f\n",   __func__, hparams.rope_yarn_log_mul);
    }

    if (arch == LLM_ARCH_QWEN2MOE) {
        LLAMA_LOG_INFO("%s: n_ff_exp         = %d\n",     __func__, hparams.n_ff_exp);
        LLAMA_LOG_INFO("%s: n_ff_shexp       = %d\n",     __func__, hparams.n_ff_shexp);
    }

    if (arch == LLM_ARCH_QWEN3MOE || arch == LLM_ARCH_OPENAI_MOE) {
        LLAMA_LOG_INFO("%s: n_ff_exp         = %d\n",     __func__, hparams.n_ff_exp);
    }

    if (arch == LLM_ARCH_MINICPM ||
        arch == LLM_ARCH_GRANITE ||
        arch == LLM_ARCH_GRANITE_MOE ||
        arch == LLM_ARCH_GRANITE_HYBRID) {
        LLAMA_LOG_INFO("%s: f_embedding_scale = %f\n", __func__, hparams.f_embedding_scale);
        LLAMA_LOG_INFO("%s: f_residual_scale  = %f\n", __func__, hparams.f_residual_scale);
        LLAMA_LOG_INFO("%s: f_attention_scale = %f\n", __func__, hparams.f_attention_scale);
        LLAMA_LOG_INFO("%s: n_ff_shexp        = %d\n", __func__, hparams.n_ff_shexp);
    }

    if (arch == LLM_ARCH_BAILINGMOE) {
        LLAMA_LOG_INFO("%s: n_layer_dense_lead   = %d\n",     __func__, hparams.n_layer_dense_lead);
        LLAMA_LOG_INFO("%s: n_ff_exp             = %d\n",     __func__, hparams.n_ff_exp);
        LLAMA_LOG_INFO("%s: n_expert_shared      = %d\n",     __func__, hparams.n_expert_shared);
        LLAMA_LOG_INFO("%s: expert_weights_scale = %.1f\n",   __func__, hparams.expert_weights_scale);
        LLAMA_LOG_INFO("%s: expert_weights_norm  = %d\n",     __func__, hparams.expert_weights_norm);
    }

    if (arch == LLM_ARCH_SMALLTHINKER) {
        LLAMA_LOG_INFO("%s: n_ff_exp             = %d\n",     __func__, hparams.n_ff_exp);
        LLAMA_LOG_INFO("%s: expert_gating_func   = %s\n",     __func__, llama_expert_gating_func_name((llama_expert_gating_func_type) hparams.expert_gating_func));
    }

    vocab.print_info();
}

ggml_backend_dev_t llama_model::dev_layer(int il) const {
    return pimpl->dev_layer.at(il).dev;
}

ggml_backend_dev_t llama_model::dev_output() const {
    return pimpl->dev_output.dev;
}

template<typename F>
static bool buft_supported(ggml_backend_buffer_type_t buft, ggml_backend_dev_t dev, F & fn) {
    ggml_init_params params = {
        /*.mem_size   =*/ ggml_tensor_overhead()*8,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };

    ggml_context_ptr ctx { ggml_init(params) };
    if (!ctx) {
        throw std::runtime_error(format("failed to create ggml context"));
    }

    ggml_backend_buffer_ptr buf { ggml_backend_buft_alloc_buffer(buft, 0) };
    ggml_tensor * op_tensor = fn(ctx.get());
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        if (op_tensor->src[i] != nullptr) {
            assert(op_tensor->src[i]->buffer == nullptr);
            op_tensor->src[i]->buffer = buf.get();
        }
    }

    bool op_supported = ggml_backend_dev_supports_op(dev, op_tensor);

    return op_supported;
}

template<typename F>
static ggml_backend_buffer_type_t select_buft(const buft_list_t & buft_list, const F & fn) {
    for (const auto & cur : buft_list) {
        ggml_backend_dev_t cur_dev = cur.first;
        ggml_backend_buffer_type_t cur_buft = cur.second;
        if (buft_supported(cur_buft, cur_dev, fn)) {
            return cur_buft;
        }
    }

    throw std::runtime_error(format("no suitable buffer type found"));
}

ggml_backend_buffer_type_t llama_model::select_buft(int il) const {
    return ::select_buft(
            *pimpl->dev_layer.at(il).buft_list,
            [&](ggml_context * ctx) {
                ggml_tensor * cur = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hparams.n_embd);
                ggml_tensor * layer_dir = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hparams.n_embd);
                return ggml_add(ctx, cur, layer_dir);
            });
}

bool llama_model::has_tensor_overrides() const {
    return pimpl->has_tensor_overrides;
}

const ggml_tensor * llama_model::get_tensor(const char * name) const {
    auto it = std::find_if(tensors_by_name.begin(), tensors_by_name.end(),
            [name](const std::pair<std::string, ggml_tensor *> & it) {
                return it.first == name;
            });
    if (it == tensors_by_name.end()) {
        return nullptr;
    }

    return it->second;
}

float llama_model::get_rope_freq_base (const llama_cparams & cparams, int il) const {
    return hparams.is_swa(il) ? hparams.rope_freq_base_train_swa : cparams.rope_freq_base;
}

float llama_model::get_rope_freq_scale(const llama_cparams & cparams, int il) const {
    return hparams.is_swa(il) ? hparams.rope_freq_scale_train_swa : cparams.rope_freq_scale;
}

ggml_tensor * llama_model::get_rope_factors(const llama_cparams & cparams, int il) const {
    const uint32_t n_ctx_per_seq = cparams.n_ctx / cparams.n_seq_max;

    // choose long/short freq factors based on the context size
    if (layers[il].rope_freqs != nullptr) {
        return layers[il].rope_freqs;
    }

    if (n_ctx_per_seq > hparams.n_ctx_orig_yarn) {
        return layers[il].rope_long;
    }

    return layers[il].rope_short;
}

struct llm_build_llama : public llm_graph_context {
    llm_build_llama(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
        const int64_t n_embd_head = hparams.n_embd_head_v;

        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
        GGML_ASSERT(n_embd_head == hparams.n_rot);

        ggml_tensor * cur;
        ggml_tensor * inpL;

        inpL = build_inp_embd(model.tok_embd);

        // inp_pos - contains the positions
        ggml_tensor * inp_pos = build_inp_pos();

        auto * inp_attn = build_attn_inp_kv_unified();

        const float kq_scale = hparams.f_attention_scale == 0.0f ? 1.0f/sqrtf(float(n_embd_head)) : hparams.f_attention_scale;

        ggml_tensor * inp_out_ids = build_inp_out_ids();

        for (int il = 0; il < n_layer; ++il) {
            ggml_tensor * inpSA = inpL;

            // norm
            cur = build_norm(inpL,
                    model.layers[il].attn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "attn_norm", il);

            // self-attention
            {
                // rope freq factors for llama3; may return nullptr for llama2 and other models
                ggml_tensor * rope_factors = model.get_rope_factors(cparams, il);

                // compute Q and K and RoPE them
                ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
                cb(Qcur, "Qcur", il);
                if (model.layers[il].bq) {
                    Qcur = ggml_add(ctx0, Qcur, model.layers[il].bq);
                    cb(Qcur, "Qcur", il);
                }

                ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
                cb(Kcur, "Kcur", il);
                if (model.layers[il].bk) {
                    Kcur = ggml_add(ctx0, Kcur, model.layers[il].bk);
                    cb(Kcur, "Kcur", il);
                }

                ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
                cb(Vcur, "Vcur", il);
                if (model.layers[il].bv) {
                    Vcur = ggml_add(ctx0, Vcur, model.layers[il].bv);
                    cb(Vcur, "Vcur", il);
                }

                Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
                Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
                Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

                Qcur = ggml_rope_ext(
                        ctx0, Qcur, inp_pos, rope_factors,
                        n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow
                        );

                Kcur = ggml_rope_ext(
                        ctx0, Kcur, inp_pos, rope_factors,
                        n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow
                        );

                cb(Qcur, "Qcur", il);
                cb(Kcur, "Kcur", il);
                cb(Vcur, "Vcur", il);

                cur = build_attn(inp_attn,
                        model.layers[il].wo, model.layers[il].bo,
                        Qcur, Kcur, Vcur, nullptr, nullptr, kq_scale, il);
                cb(cur, "attn_out", il);
            }

            if (il == n_layer - 1 && inp_out_ids) {
                cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
                inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
            }

            ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
            cb(ffn_inp, "ffn_inp", il);

            // feed-forward network (non-MoE)
            if (model.layers[il].ffn_gate_inp == nullptr) {

                cur = build_norm(ffn_inp,
                        model.layers[il].ffn_norm, NULL,
                        LLM_NORM_RMS, il);
                cb(cur, "ffn_norm", il);

                cur = build_ffn(cur,
                        model.layers[il].ffn_up,   model.layers[il].ffn_up_b,   NULL,
                        model.layers[il].ffn_gate, model.layers[il].ffn_gate_b, NULL,
                        model.layers[il].ffn_down, model.layers[il].ffn_down_b, NULL,
                        NULL,
                        LLM_FFN_SILU, LLM_FFN_PAR, il);
                cb(cur, "ffn_out", il);
            } else {
                // MoE branch
                cur = build_norm(ffn_inp,
                        model.layers[il].ffn_norm, NULL,
                        LLM_NORM_RMS, il);
                cb(cur, "ffn_norm", il);

                cur = build_moe_ffn(cur,
                        model.layers[il].ffn_gate_inp,
                        model.layers[il].ffn_up_exps,
                        model.layers[il].ffn_gate_exps,
                        model.layers[il].ffn_down_exps,
                        nullptr,
                        n_expert, n_expert_used,
                        LLM_FFN_SILU, true,
                        false, 0.0,
                        LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX,
                        il);
                cb(cur, "ffn_moe_out", il);
            }

            cur = ggml_add(ctx0, cur, ffn_inp);
            cb(cur, "ffn_out", il);

            cur = build_cvec(cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        cur = build_norm(cur,
                model.output_norm, NULL,
                LLM_NORM_RMS, -1);

        cb(cur, "result_norm", -1);
        res->t_embd = cur;

        // lm_head
        cur = build_lora_mm(model.output, cur);

        cb(cur, "result_output", -1);
        res->t_logits = cur;

        ggml_build_forward_expand(gf, cur);
    }
};

struct llm_build_llama_iswa : public llm_graph_context {
    llm_build_llama_iswa(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
        const int64_t n_embd_head = hparams.n_embd_head_v;

        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
        GGML_ASSERT(n_embd_head == hparams.n_rot);

        ggml_tensor * cur;
        ggml_tensor * inpL;

        inpL = build_inp_embd(model.tok_embd);

        // inp_pos - contains the positions
        ggml_tensor * inp_pos = build_inp_pos();

        // temperature tuning
        ggml_tensor * inp_attn_scale = nullptr;
        inp_attn_scale = build_inp_attn_scale();

        auto * inp_attn = build_attn_inp_kv_unified_iswa();

        const float kq_scale = hparams.f_attention_scale == 0.0f ? 1.0f/sqrtf(float(n_embd_head)) : hparams.f_attention_scale;

        ggml_tensor * inp_out_ids = build_inp_out_ids();

        for (int il = 0; il < n_layer; ++il) {
            ggml_tensor * inpSA = inpL;

            const bool use_rope = (il + 1) % hparams.n_no_rope_layer_step != 0;

            // norm
            cur = build_norm(inpL,
                    model.layers[il].attn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "attn_norm", il);

            // self-attention
            {
                // rope freq factors for llama3; may return nullptr for llama2 and other models
                ggml_tensor * rope_factors = model.get_rope_factors(cparams, il);

                // compute Q and K and RoPE them
                ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
                cb(Qcur, "Qcur", il);
                if (model.layers[il].bq) {
                    Qcur = ggml_add(ctx0, Qcur, model.layers[il].bq);
                    cb(Qcur, "Qcur", il);
                }

                ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
                cb(Kcur, "Kcur", il);
                if (model.layers[il].bk) {
                    Kcur = ggml_add(ctx0, Kcur, model.layers[il].bk);
                    cb(Kcur, "Kcur", il);
                }

                ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
                cb(Vcur, "Vcur", il);
                if (model.layers[il].bv) {
                    Vcur = ggml_add(ctx0, Vcur, model.layers[il].bv);
                    cb(Vcur, "Vcur", il);
                }

                Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
                Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
                Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

                if (use_rope) {
                    Qcur = ggml_rope_ext(
                            ctx0, Qcur, inp_pos, rope_factors,
                            n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                            ext_factor, attn_factor, beta_fast, beta_slow
                            );

                    Kcur = ggml_rope_ext(
                            ctx0, Kcur, inp_pos, rope_factors,
                            n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                            ext_factor, attn_factor, beta_fast, beta_slow
                            );
                } else if (inp_attn_scale) {
                    Qcur = ggml_mul(ctx0, Qcur, inp_attn_scale);
                }

                cb(Qcur, "Qcur", il);
                cb(Kcur, "Kcur", il);
                cb(Vcur, "Vcur", il);

                if (use_rope && hparams.use_kq_norm) {
                    // Llama4TextL2Norm
                    Qcur = ggml_rms_norm(ctx0, Qcur, hparams.f_norm_rms_eps);
                    Kcur = ggml_rms_norm(ctx0, Kcur, hparams.f_norm_rms_eps);
                    cb(Qcur, "Qcur_normed", il);
                    cb(Kcur, "Kcur_normed", il);
                }

                cur = build_attn(inp_attn,
                        model.layers[il].wo, model.layers[il].bo,
                        Qcur, Kcur, Vcur, nullptr, nullptr, kq_scale, il);
                cb(cur, "attn_out", il);
            }

            if (il == n_layer - 1 && inp_out_ids) {
                cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
                inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
            }

            ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
            cb(ffn_inp, "ffn_inp", il);

            // feed-forward network (non-MoE)
            if (model.layers[il].ffn_gate_inp == nullptr) {
                cur = build_norm(ffn_inp,
                        model.layers[il].ffn_norm, NULL,
                        LLM_NORM_RMS, il);
                cb(cur, "ffn_norm", il);

                cur = build_ffn(cur,
                        model.layers[il].ffn_up,   model.layers[il].ffn_up_b,   NULL,
                        model.layers[il].ffn_gate, model.layers[il].ffn_gate_b, NULL,
                        model.layers[il].ffn_down, model.layers[il].ffn_down_b, NULL,
                        NULL,
                        LLM_FFN_SILU, LLM_FFN_PAR, il);
                cb(cur, "ffn_out", il);
            } else {
                ggml_tensor * ffn_inp_normed = build_norm(ffn_inp,
                        model.layers[il].ffn_norm, NULL,
                        LLM_NORM_RMS, il);
                cb(cur, "ffn_norm", il);

                ggml_tensor * moe_out = build_moe_ffn(ffn_inp_normed,
                        model.layers[il].ffn_gate_inp,
                        model.layers[il].ffn_up_exps,
                        model.layers[il].ffn_gate_exps,
                        model.layers[il].ffn_down_exps,
                        nullptr,
                        n_expert, n_expert_used,
                        LLM_FFN_SILU, false,
                        false, 0.0,
                        LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID,
                        il);

                // Shared experts
                ggml_tensor * shexp_out = build_ffn(ffn_inp_normed,
                    model.layers[il].ffn_up_shexp,   NULL, NULL,
                    model.layers[il].ffn_gate_shexp, NULL, NULL,
                    model.layers[il].ffn_down_shexp, NULL, NULL,
                    NULL,
                    LLM_FFN_SILU, LLM_FFN_PAR, il);
                cb(shexp_out, "ffn_moe_shexp", il);

                cur = ggml_add(ctx0, moe_out, shexp_out);
                cb(cur, "ffn_moe_out_merged", il);
            }

            cur = ggml_add(ctx0, cur, ffn_inp);
            cb(cur, "ffn_out", il);

            cur = build_cvec(cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        cur = build_norm(cur,
                model.output_norm, NULL,
                LLM_NORM_RMS, -1);

        cb(cur, "result_norm", -1);
        res->t_embd = cur;

        // lm_head
        cur = build_lora_mm(model.output, cur);

        cb(cur, "result_output", -1);
        res->t_logits = cur;

        ggml_build_forward_expand(gf, cur);
    }
};

struct llm_build_deci : public llm_graph_context {
    llm_build_deci(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
        const int64_t n_embd_head = hparams.n_embd_head_v;

        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
        GGML_ASSERT(n_embd_head == hparams.n_rot);

        ggml_tensor * cur;
        ggml_tensor * inpL;

        inpL = build_inp_embd(model.tok_embd);

        // inp_pos - contains the positions
        ggml_tensor * inp_pos = build_inp_pos();

        auto * inp_attn = build_attn_inp_kv_unified();

        const float kq_scale = hparams.f_attention_scale == 0.0f ? 1.0f/sqrtf(float(n_embd_head)) : hparams.f_attention_scale;

        ggml_tensor * inp_out_ids = build_inp_out_ids();

        for (int il = 0; il < n_layer; ++il) {
            ggml_tensor * inpSA = inpL;
            const int64_t n_head_kv = hparams.n_head_kv(il);
            const int64_t n_head    = hparams.n_head(il);
            const int64_t n_ff      = hparams.n_ff(il);

            if (n_head == 0) {
                // attention-free layer of Llama-3_1-Nemotron-51B
                cur = inpL;
            } else {
                // norm
                cur = build_norm(inpL,
                        model.layers[il].attn_norm, NULL,
                        LLM_NORM_RMS, il);
                cb(cur, "attn_norm", il);
            }

            if (n_head > 0 && n_head_kv == 0) {
                // "linear attention" of Llama-3_1-Nemotron-51B
                cur = build_lora_mm(model.layers[il].wo, cur);
                cb(cur, "wo", il);
            } else if (n_head > 0) {
                // self-attention
                // rope freq factors for llama3; may return nullptr for llama2 and other models
                ggml_tensor * rope_factors = model.get_rope_factors(cparams, il);

                // compute Q and K and RoPE them
                ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
                cb(Qcur, "Qcur", il);
                if (model.layers[il].bq) {
                    Qcur = ggml_add(ctx0, Qcur, model.layers[il].bq);
                    cb(Qcur, "Qcur", il);
                }

                ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
                cb(Kcur, "Kcur", il);
                if (model.layers[il].bk) {
                    Kcur = ggml_add(ctx0, Kcur, model.layers[il].bk);
                    cb(Kcur, "Kcur", il);
                }

                ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
                cb(Vcur, "Vcur", il);
                if (model.layers[il].bv) {
                    Vcur = ggml_add(ctx0, Vcur, model.layers[il].bv);
                    cb(Vcur, "Vcur", il);
                }

                Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
                Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
                Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

                Qcur = ggml_rope_ext(
                        ctx0, Qcur, inp_pos, rope_factors,
                        n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow
                        );

                Kcur = ggml_rope_ext(
                        ctx0, Kcur, inp_pos, rope_factors,
                        n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow
                        );

                cb(Qcur, "Qcur", il);
                cb(Kcur, "Kcur", il);
                cb(Vcur, "Vcur", il);

                cur = build_attn(inp_attn,
                        model.layers[il].wo, model.layers[il].bo,
                        Qcur, Kcur, Vcur, nullptr, nullptr, kq_scale, il);
            }

            if (il == n_layer - 1 && inp_out_ids) {
                cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
                inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
            }

            // FFN-free layer of Llama-3_1-Nemotron-Ultra-253B
            if (n_ff == 0) {
                continue;
            }

            // modified to support attention-free layer of Llama-3_1-Nemotron-51B
            ggml_tensor * ffn_inp = cur;
            if (n_head > 0) {
                ffn_inp = ggml_add(ctx0, cur, inpSA);
                cb(ffn_inp, "ffn_inp", il);
            }

            // feed-forward network
            if (model.layers[il].ffn_gate_inp == nullptr) {
                cur = build_norm(ffn_inp,
                        model.layers[il].ffn_norm, NULL,
                        LLM_NORM_RMS, il);
                cb(cur, "ffn_norm", il);

                cur = build_ffn(cur,
                        model.layers[il].ffn_up,   model.layers[il].ffn_up_b,   NULL,
                        model.layers[il].ffn_gate, model.layers[il].ffn_gate_b, NULL,
                        model.layers[il].ffn_down, model.layers[il].ffn_down_b, NULL,
                        NULL,
                        LLM_FFN_SILU, LLM_FFN_PAR, il);
                cb(cur, "ffn_out", il);
            }

            cur = ggml_add(ctx0, cur, ffn_inp);
            cb(cur, "ffn_out", il);

            cur = build_cvec(cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        cur = build_norm(cur,
                model.output_norm, NULL,
                LLM_NORM_RMS, -1);

        cb(cur, "result_norm", -1);
        res->t_embd = cur;

        // lm_head
        cur = build_lora_mm(model.output, cur);

        cb(cur, "result_output", -1);
        res->t_logits = cur;

        ggml_build_forward_expand(gf, cur);
    }
};

struct llm_build_baichuan : public llm_graph_context {
    llm_build_baichuan(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
        const int64_t n_embd_head = hparams.n_embd_head_v;

        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
        GGML_ASSERT(n_embd_head == hparams.n_rot);

        ggml_tensor * cur;
        ggml_tensor * inpL;

        inpL = build_inp_embd(model.tok_embd);

        // inp_pos - contains the positions
        ggml_tensor * inp_pos = model.type == LLM_TYPE_7B ? build_inp_pos() : nullptr;

        auto * inp_attn = build_attn_inp_kv_unified();

        ggml_tensor * inp_out_ids = build_inp_out_ids();

        for (int il = 0; il < n_layer; ++il) {
            ggml_tensor * inpSA = inpL;

            cur = build_norm(inpL,
                    model.layers[il].attn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "attn_norm", il);

            // self-attention
            {
                ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
                cb(Qcur, "Qcur", il);

                ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
                cb(Kcur, "Kcur", il);

                ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
                cb(Vcur, "Vcur", il);

                Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
                Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
                Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

                switch (model.type) {
                    case LLM_TYPE_7B:
                        Qcur = ggml_rope_ext(
                                ctx0, Qcur, inp_pos, nullptr,
                                n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                                ext_factor, attn_factor, beta_fast, beta_slow
                                );
                        Kcur = ggml_rope_ext(
                                ctx0, Kcur, inp_pos, nullptr,
                                n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                                ext_factor, attn_factor, beta_fast, beta_slow
                                );
                        break;
                    case LLM_TYPE_13B:
                        break;
                    default:
                        GGML_ABORT("fatal error");
                }

                cb(Qcur, "Qcur", il);
                cb(Kcur, "Kcur", il);
                cb(Vcur, "Vcur", il);

                cur = build_attn(inp_attn,
                        model.layers[il].wo, NULL,
                        Qcur, Kcur, Vcur, nullptr, nullptr, 1.0f/sqrtf(float(n_embd_head)), il);
            }

            if (il == n_layer - 1 && inp_out_ids) {
                cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
                inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
            }

            ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
            cb(ffn_inp, "ffn_inp", il);

            // feed-forward network
            {
                cur = build_norm(ffn_inp,
                        model.layers[il].ffn_norm, NULL,
                        LLM_NORM_RMS, il);
                cb(cur, "ffn_norm", il);

                cur = build_ffn(cur,
                        model.layers[il].ffn_up,   NULL, NULL,
                        model.layers[il].ffn_gate, NULL, NULL,
                        model.layers[il].ffn_down, NULL, NULL,
                        NULL,
                        LLM_FFN_SILU, LLM_FFN_PAR, il);
                cb(cur, "ffn_out", il);
            }

            cur = ggml_add(ctx0, cur, ffn_inp);

            cur = build_cvec(cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        cur = build_norm(cur,
                model.output_norm, NULL,
                LLM_NORM_RMS, -1);

        cb(cur, "result_norm", -1);
        res->t_embd = cur;

        // lm_head
        cur = build_lora_mm(model.output, cur);

        cb(cur, "result_output", -1);
        res->t_logits = cur;

        ggml_build_forward_expand(gf, cur);
    }
};

struct llm_build_xverse : public llm_graph_context {
    llm_build_xverse(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
        const int64_t n_embd_head = hparams.n_embd_head_v;

        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
        GGML_ASSERT(n_embd_head == hparams.n_rot);

        ggml_tensor * cur;
        ggml_tensor * inpL;

        inpL = build_inp_embd(model.tok_embd);

        // inp_pos - contains the positions
        ggml_tensor * inp_pos = build_inp_pos();

        auto * inp_attn = build_attn_inp_kv_unified();

        ggml_tensor * inp_out_ids = build_inp_out_ids();

        for (int il = 0; il < n_layer; ++il) {
            ggml_tensor * inpSA = inpL;

            cur = build_norm(inpL,
                    model.layers[il].attn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "attn_norm", il);

            // self-attention
            {
                ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
                cb(Qcur, "Qcur", il);

                ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
                cb(Kcur, "Kcur", il);

                ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
                cb(Vcur, "Vcur", il);

                Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
                Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
                Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

                Qcur = ggml_rope_ext(
                        ctx0, Qcur, inp_pos, nullptr,
                        n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow
                        );

                Kcur = ggml_rope_ext(
                        ctx0, Kcur, inp_pos, nullptr,
                        n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow
                        );

                cb(Qcur, "Qcur", il);
                cb(Kcur, "Kcur", il);
                cb(Vcur, "Vcur", il);

                cur = build_attn(inp_attn,
                        model.layers[il].wo, NULL,
                        Qcur, Kcur, Vcur, nullptr, nullptr, 1.0f/sqrtf(float(n_embd_head)), il);
            }

            if (il == n_layer - 1 && inp_out_ids) {
                cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
                inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
            }

            ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
            cb(ffn_inp, "ffn_inp", il);

            // feed-forward network
            {
                cur = build_norm(ffn_inp,
                        model.layers[il].ffn_norm, NULL,
                        LLM_NORM_RMS, il);
                cb(cur, "ffn_norm", il);

                cur = build_ffn(cur,
                        model.layers[il].ffn_up,   NULL, NULL,
                        model.layers[il].ffn_gate, NULL, NULL,
                        model.layers[il].ffn_down, NULL, NULL,
                        NULL,
                        LLM_FFN_SILU, LLM_FFN_PAR, il);
                cb(cur, "ffn_out", il);
            }

            cur = ggml_add(ctx0, cur, ffn_inp);

            cur = build_cvec(cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        cur = build_norm(cur, model.output_norm, NULL, LLM_NORM_RMS, -1);

        cb(cur, "result_norm", -1);
        res->t_embd = cur;

        // lm_head
        cur = build_lora_mm(model.output, cur);

        cb(cur, "result_output", -1);
        res->t_logits = cur;

        ggml_build_forward_expand(gf, cur);
    }
};

struct llm_build_falcon : public llm_graph_context {
    llm_build_falcon(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
        const int64_t n_embd_head = hparams.n_embd_head_v;
        const int64_t n_embd_gqa  = hparams.n_embd_v_gqa();

        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
        GGML_ASSERT(n_embd_head == hparams.n_rot);

        ggml_tensor * cur;
        ggml_tensor * inpL;

        inpL = build_inp_embd(model.tok_embd);

        // inp_pos - contains the positions
        ggml_tensor * inp_pos = build_inp_pos();

        auto * inp_attn = build_attn_inp_kv_unified();

        ggml_tensor * inp_out_ids = build_inp_out_ids();

        for (int il = 0; il < n_layer; ++il) {
            ggml_tensor * attn_norm;

            attn_norm = build_norm(inpL,
                    model.layers[il].attn_norm,
                    model.layers[il].attn_norm_b,
                    LLM_NORM, il);
            cb(attn_norm, "attn_norm", il);

            // self-attention
            {
                if (model.layers[il].attn_norm_2) {
                    // Falcon-40B
                    cur = build_norm(inpL,
                            model.layers[il].attn_norm_2,
                            model.layers[il].attn_norm_2_b,
                            LLM_NORM, il);
                    cb(cur, "attn_norm_2", il);
                } else {
                    cur = attn_norm;
                }

                cur = build_lora_mm(model.layers[il].wqkv, cur);
                cb(cur, "wqkv", il);

                ggml_tensor * Qcur = ggml_view_3d(ctx0, cur, n_embd_head, n_head,    n_tokens, n_embd_head*sizeof(float), cur->nb[1], 0*sizeof(float)*(n_embd));
                ggml_tensor * Kcur = ggml_view_3d(ctx0, cur, n_embd_head, n_head_kv, n_tokens, n_embd_head*sizeof(float), cur->nb[1], 1*sizeof(float)*(n_embd));
                ggml_tensor * Vcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd_gqa, n_tokens, cur->nb[1], 1*sizeof(float)*(n_embd + n_embd_gqa)));

                Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

                // using mode = 2 for neox mode
                Qcur = ggml_rope_ext(
                        ctx0, Qcur, inp_pos, nullptr,
                        n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow
                        );

                Kcur = ggml_rope_ext(
                        ctx0, Kcur, inp_pos, nullptr,
                        n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow
                        );

                cb(Qcur, "Qcur", il);
                cb(Kcur, "Kcur", il);
                cb(Vcur, "Vcur", il);

                cur = build_attn(inp_attn,
                        model.layers[il].wo, NULL,
                        Qcur, Kcur, Vcur, nullptr, nullptr, 1.0f/sqrtf(float(n_embd_head)), il);
            }

            if (il == n_layer - 1 && inp_out_ids) {
                cur       = ggml_get_rows(ctx0,       cur, inp_out_ids);
                inpL      = ggml_get_rows(ctx0,      inpL, inp_out_ids);
                attn_norm = ggml_get_rows(ctx0, attn_norm, inp_out_ids);
            }

            ggml_tensor * ffn_inp = cur;

            // feed forward
            {
                cur = build_ffn(attn_norm, // !! use the attn norm, not the result
                        model.layers[il].ffn_up,   NULL, NULL,
                        NULL,                      NULL, NULL,
                        model.layers[il].ffn_down, NULL, NULL,
                        NULL,
                        LLM_FFN_GELU, LLM_FFN_SEQ, il);
                cb(cur, "ffn_out", il);
            }

            cur = ggml_add(ctx0, cur, ffn_inp);
            cur = ggml_add(ctx0, cur, inpL);

            cur = build_cvec(cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        // norm
        cur = build_norm(cur,
                model.output_norm,
                model.output_norm_b,
                LLM_NORM, -1);

        cb(cur, "result_norm", -1);
        res->t_embd = cur;

        cur = build_lora_mm(model.output, cur);

        cb(cur, "result_output", -1);
        res->t_logits = cur;

        ggml_build_forward_expand(gf, cur);
    }
};

struct llm_build_grok : public llm_graph_context {
    llm_build_grok(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
        const int64_t n_embd_head = hparams.n_embd_head_v;

        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
        GGML_ASSERT(n_embd_head == hparams.n_rot);

        ggml_tensor * cur;
        ggml_tensor * inpL;

        inpL = build_inp_embd(model.tok_embd);

        // multiply by embedding_multiplier_scale of 78.38367176906169
        inpL = ggml_scale(ctx0, inpL, 78.38367176906169f);

        // inp_pos - contains the positions
        ggml_tensor * inp_pos = build_inp_pos();

        auto * inp_attn = build_attn_inp_kv_unified();

        ggml_tensor * inp_out_ids = build_inp_out_ids();

        for (int il = 0; il < n_layer; ++il) {
            ggml_tensor * inpSA = inpL;

            // norm
            cur = build_norm(inpL,
                    model.layers[il].attn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "attn_norm", il);


            // self-attention
            {
                // compute Q and K and RoPE them
                ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
                cb(Qcur, "Qcur", il);
                if (model.layers[il].bq) {
                    Qcur = ggml_add(ctx0, Qcur, model.layers[il].bq);
                    cb(Qcur, "Qcur", il);
                }

                ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
                cb(Kcur, "Kcur", il);
                if (model.layers[il].bk) {
                    Kcur = ggml_add(ctx0, Kcur, model.layers[il].bk);
                    cb(Kcur, "Kcur", il);
                }

                ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
                cb(Vcur, "Vcur", il);
                if (model.layers[il].bv) {
                    Vcur = ggml_add(ctx0, Vcur, model.layers[il].bv);
                    cb(Vcur, "Vcur", il);
                }

                Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
                Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
                Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

                Qcur = ggml_rope_ext(
                        ctx0, Qcur, inp_pos, nullptr,
                        n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow
                        );

                Kcur = ggml_rope_ext(
                        ctx0, Kcur, inp_pos, nullptr,
                        n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow
                        );

                cb(Qcur, "Qcur", il);
                cb(Kcur, "Kcur", il);
                cb(Vcur, "Vcur", il);

                cur = build_attn(inp_attn,
                        model.layers[il].wo, model.layers[il].bo,
                        Qcur, Kcur, Vcur, nullptr, nullptr, 1.0f, il);
            }

            if (il == n_layer - 1 && inp_out_ids) {
                cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
                inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
            }

            // Grok
            // if attn_out_norm is present then apply it before adding the input
            if (model.layers[il].attn_out_norm) {
                cur = build_norm(cur,
                        model.layers[il].attn_out_norm, NULL,
                        LLM_NORM_RMS, il);
                cb(cur, "attn_out_norm", il);
            }

            ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
            cb(ffn_inp, "ffn_inp", il);

            // feed-forward network
            // MoE branch
            cur = build_norm(ffn_inp,
                    model.layers[il].ffn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "ffn_norm", il);

            cur = build_moe_ffn(cur,
                    model.layers[il].ffn_gate_inp,
                    model.layers[il].ffn_up_exps,
                    model.layers[il].ffn_gate_exps,
                    model.layers[il].ffn_down_exps,
                    nullptr,
                    n_expert, n_expert_used,
                    LLM_FFN_GELU, true,
                    false, 0.0,
                    LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX,
                    il);
            cb(cur, "ffn_moe_out", il);

            // Grok
            // if layer_out_norm is present then apply it before adding the input
            // Idea: maybe ffn_out_norm is a better name
            if (model.layers[il].layer_out_norm) {
                cur = build_norm(cur,
                        model.layers[il].layer_out_norm, NULL,
                        LLM_NORM_RMS, il);
                cb(cur, "layer_out_norm", il);
            }

            cur = ggml_add(ctx0, cur, ffn_inp);
            cb(cur, "ffn_out", il);

            cur = build_cvec(cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        cur = build_norm(cur,
                model.output_norm, NULL,
                LLM_NORM_RMS, -1);

        cb(cur, "result_norm", -1);
        res->t_embd = cur;

        // lm_head
        cur = build_lora_mm(model.output, cur);

        // Grok
        // multiply logits by output_multiplier_scale of 0.5773502691896257

        cur = ggml_scale(ctx0, cur, 0.5773502691896257f);

        cb(cur, "result_output", -1);
        res->t_logits = cur;

        ggml_build_forward_expand(gf, cur);
    }
};

struct llm_build_dbrx : public llm_graph_context {
    llm_build_dbrx(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
        const int64_t n_embd_head = hparams.n_embd_head_v;
        const int64_t n_embd_gqa  = hparams.n_embd_v_gqa();

        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
        GGML_ASSERT(n_embd_head == hparams.n_rot);

        ggml_tensor * cur;
        ggml_tensor * inpL;

        inpL = build_inp_embd(model.tok_embd);

        // inp_pos - contains the positions
        ggml_tensor * inp_pos = build_inp_pos();

        auto * inp_attn = build_attn_inp_kv_unified();

        ggml_tensor * inp_out_ids = build_inp_out_ids();

        for (int il = 0; il < n_layer; ++il) {
            ggml_tensor * inpSA = inpL;

            // norm
            cur = build_norm(inpL,
                    model.layers[il].attn_norm, NULL,
                    LLM_NORM, il);
            cb(cur, "attn_norm", il);

            // self-attention
            {
                ggml_tensor * Qcur = nullptr;
                ggml_tensor * Kcur = nullptr;
                ggml_tensor * Vcur = nullptr;

                cur = build_lora_mm(model.layers[il].wqkv, cur);
                cb(cur, "wqkv", il);

                cur = ggml_clamp(ctx0, cur, -hparams.f_clamp_kqv, hparams.f_clamp_kqv);
                cb(cur, "wqkv_clamped", il);

                Qcur = ggml_view_3d(ctx0, cur, n_embd_head, n_head,    n_tokens, n_embd_head*sizeof(float), cur->nb[1], 0*sizeof(float)*(n_embd));
                Kcur = ggml_view_3d(ctx0, cur, n_embd_head, n_head_kv, n_tokens, n_embd_head*sizeof(float), cur->nb[1], 1*sizeof(float)*(n_embd));
                Vcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd_gqa, n_tokens, cur->nb[1], 1*sizeof(float)*(n_embd + n_embd_gqa)));

                Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

                Qcur = ggml_rope_ext(
                        ctx0, Qcur, inp_pos, nullptr,
                        n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow
                        );

                Kcur = ggml_rope_ext(
                        ctx0, Kcur, inp_pos, nullptr,
                        n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow
                        );

                cb(Qcur, "Qcur", il);
                cb(Kcur, "Kcur", il);
                cb(Vcur, "Vcur", il);

                cur = build_attn(inp_attn,
                        model.layers[il].wo, NULL,
                        Qcur, Kcur, Vcur, nullptr, nullptr, 1.0f/sqrtf(float(n_embd_head)), il);
            }

            if (il == n_layer - 1 && inp_out_ids) {
                cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
                inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
            }

            ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
            cb(ffn_inp, "ffn_inp", il);

            // feed-forward network
            // MoE branch
            cur = build_norm(ffn_inp,
                    model.layers[il].attn_out_norm, NULL,
                    LLM_NORM, il);
            cb(cur, "attn_out_norm", il);

            cur = build_moe_ffn(cur,
                    model.layers[il].ffn_gate_inp,
                    model.layers[il].ffn_up_exps,
                    model.layers[il].ffn_gate_exps,
                    model.layers[il].ffn_down_exps,
                    nullptr,
                    n_expert, n_expert_used,
                    LLM_FFN_SILU, true,
                    false, 0.0,
                    LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX,
                    il);
            cb(cur, "ffn_moe_out", il);

            cur = ggml_add(ctx0, cur, ffn_inp);
            cb(cur, "ffn_out", il);

            cur = build_cvec(cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        cur = build_norm(cur,
                model.output_norm, NULL,
                LLM_NORM, -1);

        cb(cur, "result_norm", -1);
        res->t_embd = cur;

        // lm_head
        cur = build_lora_mm(model.output, cur);

        cb(cur, "result_output", -1);
        res->t_logits = cur;

        ggml_build_forward_expand(gf, cur);
    }
};

struct llm_build_starcoder : public llm_graph_context {
    llm_build_starcoder(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
        const int64_t n_embd_head = hparams.n_embd_head_v;
        const int64_t n_embd_gqa  = hparams.n_embd_v_gqa();

        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);

        ggml_tensor * cur;
        ggml_tensor * inpL;

        inpL = build_inp_embd(model.tok_embd);

        // inp_pos - contains the positions
        ggml_tensor * inp_pos = build_inp_pos();

        auto * inp_attn = build_attn_inp_kv_unified();

        ggml_tensor * pos = ggml_get_rows(ctx0, model.pos_embd, inp_pos);
        cb(pos, "pos_embd", -1);

        inpL = ggml_add(ctx0, inpL, pos);
        cb(inpL, "inpL", -1);

        ggml_tensor * inp_out_ids = build_inp_out_ids();

        for (int il = 0; il < n_layer; ++il) {
            cur = build_norm(inpL,
                    model.layers[il].attn_norm,
                    model.layers[il].attn_norm_b,
                    LLM_NORM, il);
            cb(cur, "attn_norm", il);

            // self-attention
            {
                cur = build_lora_mm(model.layers[il].wqkv, cur);
                cb(cur, "wqkv", il);

                cur = ggml_add(ctx0, cur, model.layers[il].bqkv);
                cb(cur, "bqkv", il);

                ggml_tensor * Qcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd,     n_tokens, cur->nb[1], 0*sizeof(float)*(n_embd)));
                ggml_tensor * Kcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd_gqa, n_tokens, cur->nb[1], 1*sizeof(float)*(n_embd)));
                ggml_tensor * Vcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd_gqa, n_tokens, cur->nb[1], 1*sizeof(float)*(n_embd + n_embd_gqa)));

                Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
                Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
                Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

                cb(Qcur, "Qcur", il);
                cb(Kcur, "Kcur", il);
                cb(Vcur, "Vcur", il);

                cur = build_attn(inp_attn,
                        model.layers[il].wo, model.layers[il].bo,
                        Qcur, Kcur, Vcur, nullptr, nullptr, 1.0f/sqrtf(float(n_embd_head)), il);
            }

            if (il == n_layer - 1 && inp_out_ids) {
                cur  = ggml_get_rows(ctx0,  cur, inp_out_ids);
                inpL = ggml_get_rows(ctx0, inpL, inp_out_ids);
            }

            // add the input
            ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpL);
            cb(ffn_inp, "ffn_inp", il);

            // FF
            {
                cur = build_norm(ffn_inp,
                        model.layers[il].ffn_norm,
                        model.layers[il].ffn_norm_b,
                        LLM_NORM, il);
                cb(cur, "ffn_norm", il);

                cur = build_ffn(cur,
                        model.layers[il].ffn_up,   model.layers[il].ffn_up_b,   NULL,
                        NULL,                      NULL,                        NULL,
                        model.layers[il].ffn_down, model.layers[il].ffn_down_b, NULL,
                        NULL,
                        LLM_FFN_GELU, LLM_FFN_SEQ, il);
                cb(cur, "ffn_out", il);
            }

            cur = ggml_add(ctx0, cur, ffn_inp);

            cur = build_cvec(cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = build_norm(inpL,
                model.output_norm,
                model.output_norm_b,
                LLM_NORM, -1);

        cb(cur, "result_norm", -1);
        res->t_embd = cur;

        cur = build_lora_mm(model.output, cur);

        cb(cur, "result_output", -1);
        res->t_logits = cur;

        ggml_build_forward_expand(gf, cur);
    }
};

struct llm_build_refact : public llm_graph_context {
    llm_build_refact(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
        const int64_t n_embd_head = hparams.n_embd_head_v;

        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);

        ggml_tensor * cur;
        ggml_tensor * inpL;

        inpL = build_inp_embd(model.tok_embd);

        auto * inp_attn = build_attn_inp_kv_unified();

        ggml_tensor * inp_out_ids = build_inp_out_ids();

        for (int il = 0; il < n_layer; ++il) {
            ggml_tensor * inpSA = inpL;

            cur = build_norm(inpL,
                    model.layers[il].attn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "attn_norm", il);

            // self-attention
            {
                ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
                cb(Qcur, "Qcur", il);

                ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
                cb(Kcur, "Kcur", il);

                ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
                cb(Vcur, "Vcur", il);

                Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
                Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
                Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

                cb(Qcur, "Qcur", il);
                cb(Kcur, "Kcur", il);
                cb(Vcur, "Vcur", il);

                cur = build_attn(inp_attn,
                        model.layers[il].wo, NULL,
                        Qcur, Kcur, Vcur, nullptr, nullptr, 1.0f/sqrtf(float(n_embd_head)), il);
            }

            if (il == n_layer - 1 && inp_out_ids) {
                cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
                inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
            }

            ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
            cb(ffn_inp, "ffn_inp", il);

            // feed-forward network
            {
                cur = build_norm(ffn_inp,
                        model.layers[il].ffn_norm, NULL,
                        LLM_NORM_RMS, il);
                cb(cur, "ffn_norm", il);

                cur = build_ffn(cur,
                        model.layers[il].ffn_up,   NULL, NULL,
                        model.layers[il].ffn_gate, NULL, NULL,
                        model.layers[il].ffn_down, NULL, NULL,
                        NULL,
                        LLM_FFN_SILU, LLM_FFN_PAR, il);
                cb(cur, "ffn_out", il);
            }

            cur = ggml_add(ctx0, cur, ffn_inp);

            cur = build_cvec(cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        cur = build_norm(cur,
                model.output_norm, NULL,
                LLM_NORM_RMS, -1);

        cb(cur, "result_norm", -1);
        res->t_embd = cur;

        // lm_head
        cur = build_lora_mm(model.output, cur);

        cb(cur, "result_output", -1);
        res->t_logits = cur;

        ggml_build_forward_expand(gf, cur);
    }
};

struct llm_build_bert : public llm_graph_context {
    llm_build_bert(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
        const int64_t n_embd_head = hparams.n_embd_head_v;
        const int64_t n_embd_gqa  = hparams.n_embd_v_gqa();

        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);

        ggml_tensor * cur;
        ggml_tensor * inpL;
        ggml_tensor * inp_pos = nullptr;

        if (model.arch != LLM_ARCH_JINA_BERT_V2) {
            inp_pos = build_inp_pos();
        }

        // construct input embeddings (token, type, position)
        inpL = build_inp_embd(model.tok_embd);

        // token types are hardcoded to zero ("Sentence A")
        if (model.type_embd) {
            ggml_tensor * type_row0 = ggml_view_1d(ctx0, model.type_embd, n_embd, 0);
            inpL = ggml_add(ctx0, inpL, type_row0);
        }
        if (model.arch == LLM_ARCH_BERT) {
            inpL = ggml_add(ctx0, ggml_get_rows(ctx0, model.pos_embd, inp_pos), inpL);
        }
        cb(inpL, "inp_embd", -1);

        // embed layer norm
        inpL = build_norm(inpL, model.tok_norm, model.tok_norm_b, LLM_NORM, -1);
        cb(inpL, "inp_norm", -1);

        auto * inp_attn = build_attn_inp_no_cache();

        ggml_tensor * inp_out_ids = build_inp_out_ids();

        for (int il = 0; il < n_layer; ++il) {
            ggml_tensor * cur = inpL;

            {
                ggml_tensor * Qcur;
                ggml_tensor * Kcur;
                ggml_tensor * Vcur;

                // self-attention
                if (model.layers[il].wqkv) {
                    cur = build_lora_mm(model.layers[il].wqkv, cur);
                    cb(cur, "wqkv", il);

                    if (model.layers[il].bqkv) {
                        cur = ggml_add(ctx0, cur, model.layers[il].bqkv);
                        cb(cur, "bqkv", il);
                    }

                    Qcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd,     n_tokens, cur->nb[1], 0*sizeof(float)*(n_embd)));
                    Kcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd_gqa, n_tokens, cur->nb[1], 1*sizeof(float)*(n_embd)));
                    Vcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd_gqa, n_tokens, cur->nb[1], 1*sizeof(float)*(n_embd + n_embd_gqa)));
                } else {
                    Qcur = ggml_add(ctx0, build_lora_mm(model.layers[il].wq, cur), model.layers[il].bq);
                    Kcur = ggml_add(ctx0, build_lora_mm(model.layers[il].wk, cur), model.layers[il].bk);
                    Vcur = ggml_add(ctx0, build_lora_mm(model.layers[il].wv, cur), model.layers[il].bv);
                }

                if (model.layers[il].attn_q_norm) {
                    Qcur = build_norm(Qcur,
                            model.layers[il].attn_q_norm,
                            model.layers[il].attn_q_norm_b,
                            LLM_NORM, il);
                }

                if (model.layers[il].attn_k_norm) {
                    Kcur = build_norm(Kcur,
                            model.layers[il].attn_k_norm,
                            model.layers[il].attn_k_norm_b,
                            LLM_NORM, il);
                }

                Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
                Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
                Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

                // RoPE
                if (model.arch == LLM_ARCH_NOMIC_BERT || model.arch == LLM_ARCH_NOMIC_BERT_MOE) {
                    Qcur = ggml_rope_ext(
                            ctx0, Qcur, inp_pos, nullptr,
                            n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                            ext_factor, attn_factor, beta_fast, beta_slow
                            );

                    Kcur = ggml_rope_ext(
                            ctx0, Kcur, inp_pos, nullptr,
                            n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                            ext_factor, attn_factor, beta_fast, beta_slow
                            );
                }

                cb(Qcur, "Qcur", il);
                cb(Kcur, "Kcur", il);
                cb(Vcur, "Vcur", il);

                cur = build_attn(inp_attn,
                        model.layers[il].wo, model.layers[il].bo,
                        Qcur, Kcur, Vcur, nullptr, nullptr, 1.0f/sqrtf(float(n_embd_head)), il);
                cb(cur, "kqv_out", il);
            }

            if (il == n_layer - 1 && inp_out_ids) {
                cur  = ggml_get_rows(ctx0,  cur, inp_out_ids);
                inpL = ggml_get_rows(ctx0, inpL, inp_out_ids);
            }

            // re-add the layer input
            cur = ggml_add(ctx0, cur, inpL);

            // attention layer norm
            cur = build_norm(cur, model.layers[il].attn_out_norm, model.layers[il].attn_out_norm_b, LLM_NORM, il);

            if (model.layers[il].attn_norm_2 != nullptr) {
                cur = ggml_add(ctx0, cur, inpL); // re-add the layer input
                cur = build_norm(cur, model.layers[il].attn_norm_2, model.layers[il].attn_norm_2_b, LLM_NORM, il);
            }

            ggml_tensor * ffn_inp = cur;
            cb(ffn_inp, "ffn_inp", il);

            // feed-forward network
            if (hparams.moe_every_n_layers > 0 && il % hparams.moe_every_n_layers == 1) {
                // MoE branch
                cur = build_moe_ffn(cur,
                        model.layers[il].ffn_gate_inp,
                        model.layers[il].ffn_up_exps,
                        nullptr,
                        model.layers[il].ffn_down_exps,
                        nullptr,
                        hparams.n_expert,
                        hparams.n_expert_used,
                        LLM_FFN_GELU,
                        false, false,
                        0.0f,
                        LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX, il);
                cb(cur, "ffn_moe_out", il);
            } else if (model.arch == LLM_ARCH_BERT || model.arch == LLM_ARCH_NOMIC_BERT_MOE) {
                cur = build_ffn(cur,
                        model.layers[il].ffn_up,   model.layers[il].ffn_up_b,   NULL,
                        NULL,                      NULL,                        NULL,
                        model.layers[il].ffn_down, model.layers[il].ffn_down_b, NULL,
                        NULL,
                        LLM_FFN_GELU, LLM_FFN_SEQ, il);
                cb(cur, "ffn_out", il);
            } else if (model.arch == LLM_ARCH_JINA_BERT_V2) {
                cur = build_ffn(cur,
                        model.layers[il].ffn_up,   NULL,                        NULL,
                        model.layers[il].ffn_gate, NULL,                        NULL,
                        model.layers[il].ffn_down, model.layers[il].ffn_down_b, NULL,
                        NULL,
                        model.layers[il].ffn_gate ? LLM_FFN_GELU : LLM_FFN_GEGLU, LLM_FFN_PAR, il);
                cb(cur, "ffn_out", il);
            } else {
                cur = build_ffn(cur,
                        model.layers[il].ffn_up,   NULL, NULL,
                        model.layers[il].ffn_gate, NULL, NULL,
                        model.layers[il].ffn_down, NULL, NULL,
                        NULL,
                        LLM_FFN_SILU, LLM_FFN_PAR, il);
                cb(cur, "ffn_out", il);
            }

            // attentions bypass the intermediate layer
            cur = ggml_add(ctx0, cur, ffn_inp);

            // output layer norm
            cur = build_norm(cur, model.layers[il].layer_out_norm, model.layers[il].layer_out_norm_b, LLM_NORM, il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        cb(cur, "result_embd", -1);
        res->t_embd = cur;

        ggml_build_forward_expand(gf, cur);
    }
};

struct llm_build_neo_bert : public llm_graph_context {
    llm_build_neo_bert(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
        const int64_t n_embd_head = hparams.n_embd_head_v;
        const int64_t n_embd_gqa  = hparams.n_embd_v_gqa();

        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);

        ggml_tensor * cur;
        ggml_tensor * inpL;
        ggml_tensor * inp_pos = build_inp_pos();

        // construct input embeddings (token, type, position)
        inpL = build_inp_embd(model.tok_embd);
        cb(inpL, "inp_embd", -1);

        auto * inp_attn = build_attn_inp_no_cache();

        ggml_tensor * inp_out_ids = build_inp_out_ids();

        for (int il = 0; il < n_layer; ++il) {
            ggml_tensor * cur = inpL;

            // pre-norm
            cur = build_norm(inpL,
                    model.layers[il].attn_norm, NULL,
                    LLM_NORM_RMS, il);

            {
                ggml_tensor * Qcur;
                ggml_tensor * Kcur;
                ggml_tensor * Vcur;

                // self-attention
                cur = build_lora_mm(model.layers[il].wqkv, cur);
                cb(cur, "wqkv", il);

                Qcur = ggml_view_3d(ctx0, cur, n_embd_head, n_head,    n_tokens, n_embd_head*sizeof(float), cur->nb[1], 0*sizeof(float)*(n_embd));
                Kcur = ggml_view_3d(ctx0, cur, n_embd_head, n_head_kv, n_tokens, n_embd_head*sizeof(float), cur->nb[1], 1*sizeof(float)*(n_embd));
                Vcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd_gqa, n_tokens, cur->nb[1], 1*sizeof(float)*(n_embd + n_embd_gqa)));

                Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

                // RoPE
                Qcur = ggml_rope_ext(
                        ctx0, Qcur, inp_pos, nullptr,
                        n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow
                        );

                Kcur = ggml_rope_ext(
                        ctx0, Kcur, inp_pos, nullptr,
                        n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow
                        );

                cb(Qcur, "Qcur", il);
                cb(Kcur, "Kcur", il);
                cb(Vcur, "Vcur", il);

                cur = build_attn(inp_attn,
                        model.layers[il].wo, nullptr,
                        Qcur, Kcur, Vcur, nullptr, nullptr, 1.0f/sqrtf(float(n_embd_head)), il);
                cb(cur, "kqv_out", il);
            }

            if (il == n_layer - 1 && inp_out_ids) {
                cur  = ggml_get_rows(ctx0,  cur, inp_out_ids);
                inpL = ggml_get_rows(ctx0, inpL, inp_out_ids);
            }

            // re-add the layer input
            cur = ggml_add(ctx0, cur, inpL);

            ggml_tensor * ffn_inp = cur;
            cb(ffn_inp, "ffn_inp", il);

            // pre-norm
            cur = build_norm(ffn_inp,
                    model.layers[il].ffn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "ffn_norm", il);

            // feed-forward network
            cur = build_ffn(cur,
                    model.layers[il].ffn_up,
                    NULL, NULL, NULL, NULL, NULL,
                    model.layers[il].ffn_down,
                    NULL, NULL, NULL,
                    LLM_FFN_SWIGLU, LLM_FFN_SEQ, il);

            // attentions bypass the intermediate layer
            cur = ggml_add(ctx0, cur, ffn_inp);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        cur = build_norm(cur,
                model.output_norm_enc, NULL,
                LLM_NORM_RMS, -1);

        cb(cur, "result_embd", -1);
        res->t_embd = cur;

        ggml_build_forward_expand(gf, cur);
    }
};

struct llm_build_bloom : public llm_graph_context {
    llm_build_bloom(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
        const int64_t n_embd_head = hparams.n_embd_head_v;
        const int64_t n_embd_gqa  = hparams.n_embd_v_gqa();

        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);

        ggml_tensor * cur;
        ggml_tensor * inpL;

        inpL = build_inp_embd(model.tok_embd);

        auto * inp_attn = build_attn_inp_kv_unified();

        inpL = build_norm(inpL,
                model.tok_norm,
                model.tok_norm_b,
                LLM_NORM, -1);
        cb(inpL, "inp_norm", -1);

        ggml_tensor * inp_out_ids = build_inp_out_ids();

        for (int il = 0; il < n_layer; ++il) {
            cur = build_norm(inpL,
                    model.layers[il].attn_norm,
                    model.layers[il].attn_norm_b,
                    LLM_NORM, il);
            cb(cur, "attn_norm", il);

            // self-attention
            {
                cur = build_lora_mm(model.layers[il].wqkv, cur);
                cb(cur, "wqkv", il);

                cur = ggml_add(ctx0, cur, model.layers[il].bqkv);
                cb(cur, "bqkv", il);

                ggml_tensor * Qcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd,     n_tokens, cur->nb[1], 0*sizeof(float)*(n_embd)));
                ggml_tensor * Kcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd_gqa, n_tokens, cur->nb[1], 1*sizeof(float)*(n_embd)));
                ggml_tensor * Vcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd_gqa, n_tokens, cur->nb[1], 1*sizeof(float)*(n_embd + n_embd_gqa)));

                Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
                Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
                Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

                cb(Qcur, "Qcur", il);
                cb(Kcur, "Kcur", il);
                cb(Vcur, "Vcur", il);

                cur = build_attn(inp_attn,
                        model.layers[il].wo, model.layers[il].bo,
                        Qcur, Kcur, Vcur, nullptr, nullptr, 1.0f/sqrtf(float(n_embd_head)), il);
            }

            if (il == n_layer - 1 && inp_out_ids) {
                cur  = ggml_get_rows(ctx0,  cur, inp_out_ids);
                inpL = ggml_get_rows(ctx0, inpL, inp_out_ids);
            }

            // Add the input
            ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpL);
            cb(ffn_inp, "ffn_inp", il);

            // FF
            {
                cur = build_norm(ffn_inp,
                        model.layers[il].ffn_norm,
                        model.layers[il].ffn_norm_b,
                        LLM_NORM, il);
                cb(cur, "ffn_norm", il);

                cur = build_ffn(cur,
                        model.layers[il].ffn_up,   model.layers[il].ffn_up_b,   NULL,
                        NULL,                      NULL,                        NULL,
                        model.layers[il].ffn_down, model.layers[il].ffn_down_b, NULL,
                        NULL,
                        LLM_FFN_GELU, LLM_FFN_SEQ, il);
                cb(cur, "ffn_out", il);
            }

            cur = ggml_add(ctx0, cur, ffn_inp);

            cur = build_cvec(cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = build_norm(inpL,
                model.output_norm,
                model.output_norm_b,
                LLM_NORM, -1);

        cb(cur, "result_norm", -1);
        res->t_embd = cur;

        cur = build_lora_mm(model.output, cur);

        cb(cur, "result_output", -1);
        res->t_logits = cur;

        ggml_build_forward_expand(gf, cur);
    }
};

struct llm_build_mpt : public llm_graph_context {
    llm_build_mpt(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
        const int64_t n_embd_head = hparams.n_embd_head_v;
        const int64_t n_embd_gqa  = hparams.n_embd_v_gqa();

        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);

        ggml_tensor * cur;
        ggml_tensor * pos;
        ggml_tensor * inpL;

        inpL = build_inp_embd(model.tok_embd);

        auto * inp_attn = build_attn_inp_kv_unified();

        if (model.pos_embd) {
            // inp_pos - contains the positions
            ggml_tensor * inp_pos = build_inp_pos();
            pos = ggml_get_rows(ctx0, model.pos_embd, inp_pos);
            cb(pos, "pos_embd", -1);

            inpL = ggml_add(ctx0, inpL, pos);
            cb(inpL, "inpL", -1);
        }

        ggml_tensor * inp_out_ids = build_inp_out_ids();

        for (int il = 0; il < n_layer; ++il) {
            ggml_tensor * attn_norm;

            attn_norm = build_norm(inpL,
                    model.layers[il].attn_norm,
                    model.layers[il].attn_norm_b,
                    LLM_NORM, il);
            cb(attn_norm, "attn_norm", il);

            // self-attention
            {
                cur = attn_norm;

                cur = build_lora_mm(model.layers[il].wqkv, cur);
                cb(cur, "wqkv", il);

                if (model.layers[il].bqkv){
                    cur = ggml_add(ctx0, cur, model.layers[il].bqkv);
                    cb(cur, "bqkv", il);
                }

                if (hparams.f_clamp_kqv > 0.0f) {
                    cur = ggml_clamp(ctx0, cur, -hparams.f_clamp_kqv, hparams.f_clamp_kqv);
                    cb(cur, "wqkv_clamped", il);
                }

                ggml_tensor * Qcur = ggml_view_2d(ctx0, cur, n_embd,     n_tokens, cur->nb[1], 0*sizeof(float)*(n_embd));
                ggml_tensor * Kcur = ggml_view_2d(ctx0, cur, n_embd_gqa, n_tokens, cur->nb[1], 1*sizeof(float)*(n_embd));
                ggml_tensor * Vcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd_gqa, n_tokens, cur->nb[1], 1*sizeof(float)*(n_embd + n_embd_gqa)));

                cb(Qcur, "Qcur", il);
                cb(Kcur, "Kcur", il);
                cb(Vcur, "Vcur", il);

                // Q/K Layernorm
                if (model.layers[il].attn_q_norm) {
                    Qcur = build_norm(Qcur,
                            model.layers[il].attn_q_norm,
                            model.layers[il].attn_q_norm_b,
                            LLM_NORM, il);
                    cb(Qcur, "Qcur", il);

                    Kcur = build_norm(Kcur,
                            model.layers[il].attn_k_norm,
                            model.layers[il].attn_k_norm_b,
                            LLM_NORM, il);
                    cb(Kcur, "Kcur", il);
                } else {
                    Qcur = ggml_cont(ctx0, Qcur);
                    cb(Qcur, "Qcur", il);

                    Kcur = ggml_cont(ctx0, Kcur);
                    cb(Kcur, "Kcur", il);
                }

                Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
                Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
                Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

                cb(Qcur, "Qcur", il);
                cb(Kcur, "Kcur", il);
                cb(Vcur, "Vcur", il);

                cur = build_attn(inp_attn,
                        model.layers[il].wo, model.layers[il].bo,
                        Qcur, Kcur, Vcur, nullptr, nullptr, 1.0f/sqrtf(float(n_embd_head)), il);
            }

            if (il == n_layer - 1 && inp_out_ids) {
                cur  = ggml_get_rows(ctx0,  cur, inp_out_ids);
                inpL = ggml_get_rows(ctx0, inpL, inp_out_ids);
            }

            // Add the input
            ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpL);
            cb(ffn_inp, "ffn_inp", il);

            // feed forward
            {
                cur = build_norm(ffn_inp,
                        model.layers[il].ffn_norm,
                        model.layers[il].ffn_norm_b,
                        LLM_NORM, il);
                cb(cur, "ffn_norm", il);
                cur = build_ffn(cur,
                        model.layers[il].ffn_up,   model.layers[il].ffn_up_b,   NULL,
                        NULL,                      NULL,                        NULL,
                        model.layers[il].ffn_down, model.layers[il].ffn_down_b, NULL,
                        model.layers[il].ffn_act,
                        LLM_FFN_GELU, LLM_FFN_SEQ, il);
                cb(cur, "ffn_out", il);
            }

            cur = ggml_add(ctx0, cur, ffn_inp);

            cur = build_cvec(cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        cur = build_norm(cur,
                model.output_norm,
                model.output_norm_b,
                LLM_NORM, -1);

        cb(cur, "result_norm", -1);
        res->t_embd = cur;

        cur = build_lora_mm(model.output, cur);

        cb(cur, "result_output", -1);
        res->t_logits = cur;

        ggml_build_forward_expand(gf, cur);
    }
};

struct llm_build_stablelm : public llm_graph_context {
    llm_build_stablelm(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
        const int64_t n_embd_head = hparams.n_embd_head_v;

        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);

        ggml_tensor * cur;
        ggml_tensor * inpL;

        inpL = build_inp_embd(model.tok_embd);

        // inp_pos - contains the positions
        ggml_tensor * inp_pos = build_inp_pos();

        auto * inp_attn = build_attn_inp_kv_unified();

        ggml_tensor * inp_out_ids = build_inp_out_ids();

        for (int il = 0; il < n_layer; ++il) {
            // norm
            cur = build_norm(inpL,
                    model.layers[il].attn_norm,
                    model.layers[il].attn_norm_b,
                    LLM_NORM, il);
            cb(cur, "attn_norm", il);

            ggml_tensor * inpSA = cur;

            // self-attention
            {
                // compute Q and K and RoPE them
                ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
                cb(Qcur, "Qcur", il);
                if (model.layers[il].bq) {
                    Qcur = ggml_add(ctx0, Qcur, model.layers[il].bq);
                    cb(Qcur, "Qcur", il);
                }

                ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
                cb(Kcur, "Kcur", il);
                if (model.layers[il].bk) {
                    Kcur = ggml_add(ctx0, Kcur, model.layers[il].bk);
                    cb(Kcur, "Kcur", il);
                }

                ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
                cb(Vcur, "Vcur", il);
                if (model.layers[il].bv) {
                    Vcur = ggml_add(ctx0, Vcur, model.layers[il].bv);
                    cb(Vcur, "Vcur", il);
                }

                Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
                Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
                Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

                if (model.layers[il].attn_q_norm) {
                    Qcur = build_norm(Qcur,
                            model.layers[il].attn_q_norm,
                            NULL,
                            LLM_NORM, il);
                    cb(Qcur, "Qcur", il);
                }

                if (model.layers[il].attn_k_norm) {
                    Kcur = build_norm(Kcur,
                            model.layers[il].attn_k_norm,
                            NULL,
                            LLM_NORM, il);
                    cb(Kcur, "Kcur", il);
                }

                Qcur = ggml_rope_ext(
                        ctx0, Qcur, inp_pos, nullptr,
                        n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow
                        );

                Kcur = ggml_rope_ext(
                        ctx0, Kcur, inp_pos, nullptr,
                        n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow
                        );

                cb(Qcur, "Qcur", il);
                cb(Kcur, "Kcur", il);
                cb(Vcur, "Vcur", il);

                cur = build_attn(inp_attn,
                        model.layers[il].wo, NULL,
                        Qcur, Kcur, Vcur, nullptr, nullptr, 1.0f/sqrtf(float(n_embd_head)), il);
            }

            if (il == n_layer - 1 && inp_out_ids) {
                cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
                inpL  = ggml_get_rows(ctx0,  inpL, inp_out_ids);
                inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
            }

            ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpL);
            cb(ffn_inp, "ffn_inp", il);

            // feed-forward network
            {
                if (model.layers[il].ffn_norm) {
                    cur = build_norm(ffn_inp,
                            model.layers[il].ffn_norm,
                            model.layers[il].ffn_norm_b,
                            LLM_NORM, il);
                    cb(cur, "ffn_norm", il);
                } else {
                    // parallel residual
                    cur = inpSA;
                }
                cur = build_ffn(cur,
                        model.layers[il].ffn_up,   NULL, NULL,
                        model.layers[il].ffn_gate, NULL, NULL,
                        model.layers[il].ffn_down, NULL, NULL,
                        NULL,
                        LLM_FFN_SILU, LLM_FFN_PAR, il);
                cb(cur, "ffn_out", il);
            }

            cur = ggml_add(ctx0, cur, ffn_inp);

            cur = build_cvec(cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        cur = build_norm(cur,
                model.output_norm,
                model.output_norm_b,
                LLM_NORM, -1);

        cb(cur, "result_norm", -1);
        res->t_embd = cur;

        // lm_head
        cur = build_lora_mm(model.output, cur);

        cb(cur, "result_output", -1);
        res->t_logits = cur;

        ggml_build_forward_expand(gf, cur);
    }
};

struct llm_build_qwen : public llm_graph_context {
    llm_build_qwen(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
        const int64_t n_embd_head = hparams.n_embd_head_v;

        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);

        ggml_tensor * cur;
        ggml_tensor * inpL;

        inpL = build_inp_embd(model.tok_embd);

        // inp_pos - contains the positions
        ggml_tensor * inp_pos = build_inp_pos();

        auto * inp_attn = build_attn_inp_kv_unified();

        ggml_tensor * inp_out_ids = build_inp_out_ids();

        for (int il = 0; il < n_layer; ++il) {
            ggml_tensor * inpSA = inpL;

            cur = build_norm(inpL,
                    model.layers[il].attn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "attn_norm", il);

            // self-attention
            {
                cur = build_lora_mm(model.layers[il].wqkv, cur);
                cb(cur, "wqkv", il);

                cur = ggml_add(ctx0, cur, model.layers[il].bqkv);
                cb(cur, "bqkv", il);

                ggml_tensor * Qcur = ggml_view_3d(ctx0, cur, n_embd_head, n_head,   n_tokens, n_embd_head*sizeof(float), cur->nb[1], 0*sizeof(float)*(n_embd));
                ggml_tensor * Kcur = ggml_view_3d(ctx0, cur, n_embd_head, n_head_kv, n_tokens, n_embd_head*sizeof(float), cur->nb[1], 1*sizeof(float)*(n_embd));
                ggml_tensor * Vcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd, n_tokens, cur->nb[1], 2*sizeof(float)*(n_embd)));

                Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

                // using mode = 2 for neox mode
                Qcur = ggml_rope_ext(
                        ctx0, Qcur, inp_pos, nullptr,
                        n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow
                        );

                Kcur = ggml_rope_ext(
                        ctx0, Kcur, inp_pos, nullptr,
                        n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow
                        );

                cb(Qcur, "Qcur", il);
                cb(Kcur, "Kcur", il);
                cb(Vcur, "Vcur", il);

                cur = build_attn(inp_attn,
                        model.layers[il].wo, NULL,
                        Qcur, Kcur, Vcur, nullptr, nullptr, 1.0f/sqrtf(float(n_embd_head)), il);
            }

            if (il == n_layer - 1 && inp_out_ids) {
                cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
                inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
            }

            ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
            cb(ffn_inp, "ffn_inp", il);

            // feed-forward forward
            {
                cur = build_norm(ffn_inp,
                        model.layers[il].ffn_norm, NULL,
                        LLM_NORM_RMS, il);
                cb(cur, "ffn_norm", il);

                cur = build_ffn(cur,
                        model.layers[il].ffn_up,   NULL, NULL,
                        model.layers[il].ffn_gate, NULL, NULL,
                        model.layers[il].ffn_down, NULL, NULL,
                        NULL,
                        LLM_FFN_SILU, LLM_FFN_PAR, il);
                cb(cur, "ffn_out", il);
            }

            cur = ggml_add(ctx0, cur, ffn_inp);

            cur = build_cvec(cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        cur = build_norm(cur,
                model.output_norm, NULL,
                LLM_NORM_RMS, -1);

        cb(cur, "result_norm", -1);
        res->t_embd = cur;

        // lm_head
        cur = build_lora_mm(model.output, cur);

        cb(cur, "result_output", -1);
        res->t_logits = cur;

        ggml_build_forward_expand(gf, cur);
    }
};

struct llm_build_qwen2 : public llm_graph_context {
    llm_build_qwen2(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
        const int64_t n_embd_head = hparams.n_embd_head_v;

        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
        GGML_ASSERT(n_embd_head == hparams.n_rot);

        ggml_tensor * cur;
        ggml_tensor * inpL;

        inpL = build_inp_embd(model.tok_embd);

        // inp_pos - contains the positions
        ggml_tensor * inp_pos = build_inp_pos();

        auto * inp_attn = build_attn_inp_kv_unified();

        ggml_tensor * inp_out_ids = build_inp_out_ids();

        for (int il = 0; il < n_layer; ++il) {
            ggml_tensor * inpSA = inpL;

            // norm
            cur = build_norm(inpL,
                    model.layers[il].attn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "attn_norm", il);

            // self-attention
            {
                // compute Q and K and RoPE them
                ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
                Qcur = ggml_add(ctx0, Qcur, model.layers[il].bq);
                cb(Qcur, "Qcur", il);

                ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
                Kcur = ggml_add(ctx0, Kcur, model.layers[il].bk);
                cb(Kcur, "Kcur", il);

                ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
                Vcur = ggml_add(ctx0, Vcur, model.layers[il].bv);
                cb(Vcur, "Vcur", il);

                Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
                Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
                Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

                Qcur = ggml_rope_ext(
                        ctx0, Qcur, inp_pos, nullptr,
                        n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow
                        );

                Kcur = ggml_rope_ext(
                        ctx0, Kcur, inp_pos, nullptr,
                        n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow
                        );

                cb(Qcur, "Qcur", il);
                cb(Kcur, "Kcur", il);
                cb(Vcur, "Vcur", il);

                cur = build_attn(inp_attn,
                        model.layers[il].wo, model.layers[il].bo,
                        Qcur, Kcur, Vcur, nullptr, nullptr, 1.0f/sqrtf(float(n_embd_head)), il);
            }

            if (il == n_layer - 1 && inp_out_ids) {
                cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
                inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
            }

            ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
            cb(ffn_inp, "ffn_inp", il);

            // feed-forward network
            cur = build_norm(ffn_inp,
                    model.layers[il].ffn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "ffn_norm", il);

            cur = build_ffn(cur,
                    model.layers[il].ffn_up,   NULL, NULL,
                    model.layers[il].ffn_gate, NULL, NULL,
                    model.layers[il].ffn_down, NULL, NULL,
                    NULL,
                    LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(cur, "ffn_out", il);

            cur = ggml_add(ctx0, cur, ffn_inp);

            cur = build_cvec(cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        cur = build_norm(cur,
                model.output_norm, NULL,
                LLM_NORM_RMS, -1);

        cb(cur, "result_norm", -1);
        res->t_embd = cur;

        // lm_head
        cur = build_lora_mm(model.output, cur);

        if (model.output_b != nullptr) {
            cur = ggml_add(ctx0, cur, model.output_b);
        }

        cb(cur, "result_output", -1);
        res->t_logits = cur;

        ggml_build_forward_expand(gf, cur);
    }
};

struct llm_build_dream : public llm_graph_context {
    llm_build_dream(const llama_model & model, const llm_graph_params & params) :
        llm_graph_context(params) {
        //copied from qwen2
        const int64_t n_embd_head = hparams.n_embd_head_v;

        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
        GGML_ASSERT(n_embd_head == hparams.n_rot);

        ggml_tensor * cur;
        ggml_tensor * inpL;

        inpL = build_inp_embd(model.tok_embd);

        // inp_pos - contains the positions
        ggml_tensor * inp_pos = build_inp_pos();

        auto * inp_attn = build_attn_inp_no_cache();

        ggml_tensor * inp_out_ids = build_inp_out_ids();

        for (int il = 0; il < n_layer; ++il) {
            ggml_tensor * inpSA = inpL;

            // norm
            cur = build_norm(inpL, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, il);
            cb(cur, "attn_norm", il);

            // self-attention
            {
                // compute Q and K and RoPE them
                ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
                Qcur               = ggml_add(ctx0, Qcur, model.layers[il].bq);
                cb(Qcur, "Qcur", il);

                ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
                Kcur               = ggml_add(ctx0, Kcur, model.layers[il].bk);
                cb(Kcur, "Kcur", il);

                ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
                Vcur               = ggml_add(ctx0, Vcur, model.layers[il].bv);
                cb(Vcur, "Vcur", il);

                Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens);
                Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
                Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

                Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                                     ext_factor, attn_factor, beta_fast, beta_slow);

                Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                                     ext_factor, attn_factor, beta_fast, beta_slow);

                cb(Qcur, "Qcur", il);
                cb(Kcur, "Kcur", il);
                cb(Vcur, "Vcur", il);

                cur = build_attn(inp_attn, model.layers[il].wo, model.layers[il].bo, Qcur, Kcur, Vcur, nullptr,
                                 nullptr, 1.0f / sqrtf(float(n_embd_head)), il);
            }

            if (il == n_layer - 1 && inp_out_ids) {
                cur   = ggml_get_rows(ctx0, cur, inp_out_ids);
                inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
            }

            ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
            cb(ffn_inp, "ffn_inp", il);

            // feed-forward network
            cur = build_norm(ffn_inp, model.layers[il].ffn_norm, NULL, LLM_NORM_RMS, il);
            cb(cur, "ffn_norm", il);

            cur = build_ffn(cur, model.layers[il].ffn_up, NULL, NULL, model.layers[il].ffn_gate, NULL, NULL,
                            model.layers[il].ffn_down, NULL, NULL, NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(cur, "ffn_out", il);

            cur = ggml_add(ctx0, cur, ffn_inp);

            cur = build_cvec(cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        cur = build_norm(cur, model.output_norm, NULL, LLM_NORM_RMS, -1);

        cb(cur, "result_norm", -1);
        res->t_embd = cur;

        // lm_head
        cur = build_lora_mm(model.output, cur);

        cb(cur, "result_output", -1);
        res->t_logits = cur;

        ggml_build_forward_expand(gf, cur);
    }
};

struct llm_build_llada : public llm_graph_context {
    llm_build_llada(const llama_model & model, const llm_graph_params & params) :
        llm_graph_context(params) {
        // LLaDA is similar to LLaMA but uses non-causal attention for diffusion
        const int64_t n_embd_head = hparams.n_embd_head_v;

        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
        GGML_ASSERT(n_embd_head == hparams.n_rot);

        ggml_tensor * cur;
        ggml_tensor * inpL;

        inpL = build_inp_embd(model.tok_embd);

        // inp_pos - contains the positions
        ggml_tensor * inp_pos = build_inp_pos();

        // Non-causal attention for diffusion
        auto * inp_attn = build_attn_inp_no_cache();

        ggml_tensor * inp_out_ids = build_inp_out_ids();

        for (int il = 0; il < n_layer; ++il) {
            ggml_tensor * inpSA = inpL;

            // norm
            cur = build_norm(inpL, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, il);
            cb(cur, "attn_norm", il);

            // self-attention
            {
                // compute separate Q, K, V projections without bias, matching LLaDALlamaBlock
                ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
                ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
                ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);

                cb(Qcur, "Qcur", il);
                cb(Kcur, "Kcur", il);
                cb(Vcur, "Vcur", il);

                Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens);
                Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
                Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

                Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                                     ext_factor, attn_factor, beta_fast, beta_slow);

                Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                                     ext_factor, attn_factor, beta_fast, beta_slow);

                cb(Qcur, "Qcur", il);
                cb(Kcur, "Kcur", il);
                cb(Vcur, "Vcur", il);

                cur = build_attn(inp_attn, model.layers[il].wo, NULL, Qcur, Kcur, Vcur, nullptr, nullptr,
                                 1.0f / sqrtf(float(n_embd_head)), il);
            }

            if (il == n_layer - 1 && inp_out_ids) {
                cur   = ggml_get_rows(ctx0, cur, inp_out_ids);
                inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
            }

            ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
            cb(ffn_inp, "ffn_inp", il);

            // feed-forward network
            cur = build_norm(ffn_inp, model.layers[il].ffn_norm, NULL, LLM_NORM_RMS, il);
            cb(cur, "ffn_norm", il);

            cur = build_ffn(cur, model.layers[il].ffn_up, NULL, NULL, model.layers[il].ffn_gate, NULL, NULL,
                            model.layers[il].ffn_down, NULL, NULL, NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(cur, "ffn_out", il);

            cur = ggml_add(ctx0, cur, ffn_inp);

            cur = build_cvec(cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        cur = build_norm(cur, model.output_norm, NULL, LLM_NORM_RMS, -1);

        cb(cur, "result_norm", -1);
        res->t_embd = cur;

        // lm_head
        cur = build_lora_mm(model.output, cur);

        cb(cur, "result_output", -1);
        res->t_logits = cur;

        ggml_build_forward_expand(gf, cur);
    }
};

struct llm_build_qwen2vl : public llm_graph_context {
    llm_build_qwen2vl(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
        const int64_t n_embd_head = hparams.n_embd_head_v;

        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
        GGML_ASSERT(n_embd_head == hparams.n_rot);

        ggml_tensor * cur;
        ggml_tensor * inpL;

        inpL = build_inp_embd(model.tok_embd);

        // inp_pos - contains the positions
        ggml_tensor * inp_pos = build_inp_pos();

        auto * inp_attn = build_attn_inp_kv_unified();

        int sections[4];
        std::copy(std::begin(hparams.rope_sections), std::begin(hparams.rope_sections) + 4, sections);

        ggml_tensor * inp_out_ids = build_inp_out_ids();

        for (int il = 0; il < n_layer; ++il) {
            ggml_tensor * inpSA = inpL;

            // norm
            cur = build_norm(inpL,
                    model.layers[il].attn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "attn_norm", il);

            // self-attention
            {
                // compute Q and K and RoPE them
                ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
                Qcur = ggml_add(ctx0, Qcur, model.layers[il].bq);
                cb(Qcur, "Qcur", il);

                ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
                Kcur = ggml_add(ctx0, Kcur, model.layers[il].bk);
                cb(Kcur, "Kcur", il);

                ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
                Vcur = ggml_add(ctx0, Vcur, model.layers[il].bv);
                cb(Vcur, "Vcur", il);

                Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
                Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
                Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

                Qcur = ggml_rope_multi(
                        ctx0, Qcur, inp_pos, nullptr,
                        n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow
                        );

                Kcur = ggml_rope_multi(
                        ctx0, Kcur, inp_pos, nullptr,
                        n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow
                        );

                cb(Qcur, "Qcur", il);
                cb(Kcur, "Kcur", il);
                cb(Vcur, "Vcur", il);

                cur = build_attn(inp_attn,
                        model.layers[il].wo, model.layers[il].bo,
                        Qcur, Kcur, Vcur, nullptr, nullptr, 1.0f/sqrtf(float(n_embd_head)), il);
            }

            if (il == n_layer - 1 && inp_out_ids) {
                cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
                inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
            }

            ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
            cb(ffn_inp, "ffn_inp", il);

            // feed-forward network
            cur = build_norm(ffn_inp,
                    model.layers[il].ffn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "ffn_norm", il);

            cur = build_ffn(cur,
                    model.layers[il].ffn_up,   NULL, NULL,
                    model.layers[il].ffn_gate, NULL, NULL,
                    model.layers[il].ffn_down, NULL, NULL,
                    NULL,
                    LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(cur, "ffn_out", il);

            cur = ggml_add(ctx0, cur, ffn_inp);

            cur = build_cvec(cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        cur = build_norm(cur,
                model.output_norm, NULL,
                LLM_NORM_RMS, -1);

        cb(cur, "result_norm", -1);
        res->t_embd = cur;

        // lm_head
        cur = build_lora_mm(model.output, cur);

        cb(cur, "result_output", -1);
        res->t_logits = cur;

        ggml_build_forward_expand(gf, cur);
    }
};

struct llm_build_qwen2moe : public llm_graph_context {
    llm_build_qwen2moe(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
        const int64_t n_embd_head = hparams.n_embd_head_v;

        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
        GGML_ASSERT(n_embd_head == hparams.n_rot);

        ggml_tensor * cur;
        ggml_tensor * inpL;

        inpL = build_inp_embd(model.tok_embd);

        // inp_pos - contains the positions
        ggml_tensor * inp_pos = build_inp_pos();

        auto * inp_attn = build_attn_inp_kv_unified();

        ggml_tensor * inp_out_ids = build_inp_out_ids();

        for (int il = 0; il < n_layer; ++il) {
            ggml_tensor * inpSA = inpL;

            // norm
            cur = build_norm(inpL,
                    model.layers[il].attn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "attn_norm", il);

            // self_attention
            {
                // compute Q and K and RoPE them
                ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
                cb(Qcur, "Qcur", il);
                if (model.layers[il].bq) {
                    Qcur = ggml_add(ctx0, Qcur, model.layers[il].bq);
                    cb(Qcur, "Qcur", il);
                }

                ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
                cb(Kcur, "Kcur", il);
                if (model.layers[il].bk) {
                    Kcur = ggml_add(ctx0, Kcur, model.layers[il].bk);
                    cb(Kcur, "Kcur", il);
                }

                ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
                cb(Vcur, "Vcur", il);
                if (model.layers[il].bv) {
                    Vcur = ggml_add(ctx0, Vcur, model.layers[il].bv);
                    cb(Vcur, "Vcur", il);
                }

                Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
                Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
                Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

                Qcur = ggml_rope_ext(
                        ctx0, Qcur, inp_pos, nullptr,
                        n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow
                        );

                Kcur = ggml_rope_ext(
                        ctx0, Kcur, inp_pos, nullptr,
                        n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow
                        );

                cb(Qcur, "Qcur", il);
                cb(Kcur, "Kcur", il);
                cb(Vcur, "Vcur", il);

                cur = build_attn(inp_attn,
                        model.layers[il].wo, model.layers[il].bo,
                        Qcur, Kcur, Vcur, nullptr, nullptr, 1.0f/sqrtf(float(n_embd_head)), il);
            }

            if (il == n_layer - 1 && inp_out_ids) {
                cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
                inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
            }

            ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
            cb(ffn_inp, "ffn_inp", il);

            // MoE branch
            cur = build_norm(ffn_inp,
                    model.layers[il].ffn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "ffn_norm", il);

            ggml_tensor * moe_out =
                build_moe_ffn(cur,
                        model.layers[il].ffn_gate_inp,
                        model.layers[il].ffn_up_exps,
                        model.layers[il].ffn_gate_exps,
                        model.layers[il].ffn_down_exps,
                        nullptr,
                        n_expert, n_expert_used,
                        LLM_FFN_SILU, false,
                        false, 0.0,
                        LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX,
                        il);
            cb(moe_out, "ffn_moe_out", il);

            // FFN shared expert
            {
                ggml_tensor * cur_gate_inp = build_lora_mm(model.layers[il].ffn_gate_inp_shexp, cur);
                cb(cur_gate_inp, "ffn_shexp_gate_inp", il);

                // sigmoid
                ggml_tensor * cur_gate = ggml_div(ctx0, ggml_silu(ctx0, cur_gate_inp), cur_gate_inp);
                cb(cur_gate, "ffn_shexp_gate", il);

                ggml_tensor * cur_ffn = build_ffn(cur,
                        model.layers[il].ffn_up_shexp,   NULL, NULL,
                        model.layers[il].ffn_gate_shexp, NULL, NULL,
                        model.layers[il].ffn_down_shexp, NULL, NULL,
                        NULL,
                        LLM_FFN_SILU, LLM_FFN_PAR, il);
                cb(cur_ffn, "ffn_shexp", il);

                ggml_tensor * ffn_shexp_out = ggml_mul(ctx0, cur_ffn, cur_gate);
                cb(ffn_shexp_out, "ffn_shexp_out", il);

                moe_out = ggml_add(ctx0, moe_out, ffn_shexp_out);
                cb(moe_out, "ffn_out", il);

                cur = moe_out;
            }

            cur = ggml_add(ctx0, cur, ffn_inp);

            cur = build_cvec(cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        cur = build_norm(cur,
                model.output_norm, NULL,
                LLM_NORM_RMS, -1);

        cb(cur, "result_norm", -1);
        res->t_embd = cur;

        // lm_head
        cur = build_lora_mm(model.output, cur);

        cb(cur, "result_output", -1);
        res->t_logits = cur;

        ggml_build_forward_expand(gf, cur);
    }
};

struct llm_build_qwen3 : public llm_graph_context {
    llm_build_qwen3(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
        const int64_t n_embd_head = hparams.n_embd_head_v;

        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
        GGML_ASSERT(n_embd_head == hparams.n_rot);

        ggml_tensor * cur;
        ggml_tensor * inpL;

        inpL = build_inp_embd(model.tok_embd);

        // inp_pos - contains the positions
        ggml_tensor * inp_pos = build_inp_pos();

        auto * inp_attn = build_attn_inp_kv_unified();

        ggml_tensor * inp_out_ids = build_inp_out_ids();

        for (int il = 0; il < n_layer; ++il) {
            ggml_tensor * inpSA = inpL;

            // norm
            cur = build_norm(inpL,
                    model.layers[il].attn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "attn_norm", il);

            // self-attention
            {
                // compute Q and K and RoPE them
                ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
                cb(Qcur, "Qcur", il);

                ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
                cb(Kcur, "Kcur", il);

                ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
                cb(Vcur, "Vcur", il);

                Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
                Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
                Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

                Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, NULL, LLM_NORM_RMS, il);
                cb(Qcur, "Qcur_normed", il);

                Qcur = ggml_rope_ext(
                        ctx0, Qcur, inp_pos, nullptr,
                        n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow
                        );

                Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, NULL, LLM_NORM_RMS, il);
                cb(Kcur, "Kcur_normed", il);

                Kcur = ggml_rope_ext(
                        ctx0, Kcur, inp_pos, nullptr,
                        n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow
                        );

                cb(Qcur, "Qcur", il);
                cb(Kcur, "Kcur", il);
                cb(Vcur, "Vcur", il);

                cur = build_attn(inp_attn,
                        model.layers[il].wo, model.layers[il].bo,
                        Qcur, Kcur, Vcur, nullptr, nullptr, 1.0f/sqrtf(float(n_embd_head)), il);
            }

            if (il == n_layer - 1 && inp_out_ids) {
                cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
                inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
            }

            ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
            cb(ffn_inp, "ffn_inp", il);

            // feed-forward network
            cur = build_norm(ffn_inp,
                    model.layers[il].ffn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "ffn_norm", il);

            cur = build_ffn(cur,
                    model.layers[il].ffn_up,   NULL, NULL,
                    model.layers[il].ffn_gate, NULL, NULL,
                    model.layers[il].ffn_down, NULL, NULL,
                    NULL,
                    LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(cur, "ffn_out", il);

            cur = ggml_add(ctx0, cur, ffn_inp);

            cur = build_cvec(cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        cur = build_norm(cur,
                model.output_norm, NULL,
                LLM_NORM_RMS, -1);

        cb(cur, "result_norm", -1);
        res->t_embd = cur;

        // lm_head
        cur = build_lora_mm(model.output, cur);

        cb(cur, "result_output", -1);
        res->t_logits = cur;

        ggml_build_forward_expand(gf, cur);
    }
};

struct llm_build_qwen3moe : public llm_graph_context {
    llm_build_qwen3moe(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
        const int64_t n_embd_head = hparams.n_embd_head_v;

        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
        GGML_ASSERT(n_embd_head == hparams.n_rot);

        ggml_tensor * cur;
        ggml_tensor * inpL;

        inpL = build_inp_embd(model.tok_embd);

        // inp_pos - contains the positions
        ggml_tensor * inp_pos = build_inp_pos();

        auto * inp_attn = build_attn_inp_kv_unified();

        ggml_tensor * inp_out_ids = build_inp_out_ids();

        for (int il = 0; il < n_layer; ++il) {
            ggml_tensor * inpSA = inpL;

            // norm
            cur = build_norm(inpL,
                    model.layers[il].attn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "attn_norm", il);

            // self_attention
            {
                // compute Q and K and RoPE them
                ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
                cb(Qcur, "Qcur", il);

                ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
                cb(Kcur, "Kcur", il);

                ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
                cb(Vcur, "Vcur", il);

                Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
                Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
                Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

                Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, NULL, LLM_NORM_RMS, il);
                cb(Qcur, "Qcur_normed", il);

                Qcur = ggml_rope_ext(
                        ctx0, Qcur, inp_pos, nullptr,
                        n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow
                        );

                Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, NULL, LLM_NORM_RMS, il);
                cb(Kcur, "Kcur_normed", il);

                Kcur = ggml_rope_ext(
                        ctx0, Kcur, inp_pos, nullptr,
                        n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow
                        );

                cb(Qcur, "Qcur", il);
                cb(Kcur, "Kcur", il);
                cb(Vcur, "Vcur", il);

                cur = build_attn(inp_attn,
                        model.layers[il].wo, model.layers[il].bo,
                        Qcur, Kcur, Vcur, nullptr, nullptr, 1.0f/sqrtf(float(n_embd_head)), il);
            }

            if (il == n_layer - 1 && inp_out_ids) {
                cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
                inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
            }

            ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
            cb(ffn_inp, "ffn_inp", il);

            // MoE branch
            cur = build_norm(ffn_inp,
                    model.layers[il].ffn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "ffn_norm", il);

            ggml_tensor * moe_out =
                build_moe_ffn(cur,
                        model.layers[il].ffn_gate_inp,
                        model.layers[il].ffn_up_exps,
                        model.layers[il].ffn_gate_exps,
                        model.layers[il].ffn_down_exps,
                        nullptr,
                        n_expert, n_expert_used,
                        LLM_FFN_SILU, true,
                        false, 0.0,
                        LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX,
                        il);
            cb(moe_out, "ffn_moe_out", il);
            cur = moe_out;

            cur = ggml_add(ctx0, cur, ffn_inp);

            cur = build_cvec(cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        cur = build_norm(cur,
                model.output_norm, NULL,
                LLM_NORM_RMS, -1);

        cb(cur, "result_norm", -1);
        res->t_embd = cur;

        // lm_head
        cur = build_lora_mm(model.output, cur);

        cb(cur, "result_output", -1);
        res->t_logits = cur;

        ggml_build_forward_expand(gf, cur);
    }
};

struct llm_build_phi2 : public llm_graph_context {
    llm_build_phi2(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
        const int64_t n_embd_head = hparams.n_embd_head_v;
        const int64_t n_embd_gqa  = hparams.n_embd_v_gqa();

        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);

        ggml_tensor * cur;
        ggml_tensor * attn_norm_output;
        ggml_tensor * ffn_output;
        ggml_tensor * inpL;

        inpL = build_inp_embd(model.tok_embd);

        // inp_pos - contains the positions
        ggml_tensor * inp_pos = build_inp_pos();

        auto * inp_attn = build_attn_inp_kv_unified();

        ggml_tensor * inp_out_ids = build_inp_out_ids();

        for (int il = 0; il < n_layer; ++il) {
            attn_norm_output = build_norm(inpL,
                    model.layers[il].attn_norm,
                    model.layers[il].attn_norm_b,
                    LLM_NORM, il);
            cb(attn_norm_output, "attn_norm", il);

            // self-attention
            {
                ggml_tensor * Qcur = nullptr;
                ggml_tensor * Kcur = nullptr;
                ggml_tensor * Vcur = nullptr;

                if (model.layers[il].wqkv) {
                    cur = build_lora_mm(model.layers[il].wqkv, attn_norm_output);
                    cb(cur, "wqkv", il);

                    cur = ggml_add(ctx0, cur, model.layers[il].bqkv);
                    cb(cur, "bqkv", il);

                    Qcur = ggml_view_3d(ctx0, cur, n_embd_head, n_head,    n_tokens, n_embd_head*sizeof(float), cur->nb[1], 0*sizeof(float)*(n_embd));
                    Kcur = ggml_view_3d(ctx0, cur, n_embd_head, n_head_kv, n_tokens, n_embd_head*sizeof(float), cur->nb[1], 1*sizeof(float)*(n_embd));
                    Vcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd_gqa, n_tokens, cur->nb[1], 1*sizeof(float)*(n_embd + n_embd_gqa)));
                } else {
                    Qcur = ggml_add(ctx0, build_lora_mm(model.layers[il].wq, attn_norm_output), model.layers[il].bq);
                    Kcur = ggml_add(ctx0, build_lora_mm(model.layers[il].wk, attn_norm_output), model.layers[il].bk);
                    Vcur = ggml_add(ctx0, build_lora_mm(model.layers[il].wv, attn_norm_output), model.layers[il].bv);
                    Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
                    Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
                }

                cb(Qcur, "Qcur", il);
                cb(Kcur, "Kcur", il);
                cb(Vcur, "Vcur", il);

                Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

                Qcur = ggml_rope_ext(
                        ctx0, Qcur, inp_pos, nullptr,
                        n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow
                        );

                Kcur = ggml_rope_ext(
                        ctx0, Kcur, inp_pos, nullptr,
                        n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow
                        );

                cb(Qcur, "Qcur", il);
                cb(Kcur, "Kcur", il);
                cb(Vcur, "Vcur", il);

                // with phi2, we scale the Q to avoid precision issues
                // ref: https://github.com/ml-explore/mlx-examples/blob/08e862336ade809bc37d1035f94b359e7d1a5152/phi2/phi2.py#L64-L66
                Qcur = ggml_scale(ctx0, Qcur, 1.0f/sqrtf(float(n_embd_head)));

                cur = build_attn(inp_attn,
                        model.layers[il].wo, model.layers[il].bo,
                        Qcur, Kcur, Vcur, nullptr, nullptr, 1.0f, il);
            }

            if (il == n_layer - 1 && inp_out_ids) {
                cur              = ggml_get_rows(ctx0,              cur, inp_out_ids);
                inpL             = ggml_get_rows(ctx0,             inpL, inp_out_ids);
                attn_norm_output = ggml_get_rows(ctx0, attn_norm_output, inp_out_ids);
            }

            // FF
            {
                ffn_output = build_ffn(attn_norm_output,
                        model.layers[il].ffn_up,   model.layers[il].ffn_up_b,   NULL,
                        NULL,                      NULL,                        NULL,
                        model.layers[il].ffn_down, model.layers[il].ffn_down_b, NULL,
                        NULL,
                        LLM_FFN_GELU, LLM_FFN_SEQ, il);
                cb(ffn_output, "ffn_out", il);
            }

            cur = ggml_add(ctx0, cur, ffn_output);
            cur = ggml_add(ctx0, cur, inpL);

            cur = build_cvec(cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = build_norm(inpL,
                model.output_norm,
                model.output_norm_b,
                LLM_NORM, -1);

        cb(cur, "result_norm", -1);
        res->t_embd = cur;

        cur = build_lora_mm(model.output, cur);
        cb(cur, "result_output_no_bias", -1);

        cur = ggml_add(ctx0, cur, model.output_b);

        cb(cur, "result_output", -1);
        res->t_logits = cur;

        ggml_build_forward_expand(gf, cur);
    }
};

template<bool iswa>
struct llm_build_phi3 : public llm_graph_context {
    llm_build_phi3(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
        const int64_t n_embd_head = hparams.n_embd_head_v;
        const int64_t n_embd_gqa = hparams.n_embd_v_gqa();

        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);

        ggml_tensor * cur;
        ggml_tensor * inpL;

        inpL = build_inp_embd(model.tok_embd);

        // inp_pos - contains the positions
        ggml_tensor * inp_pos = build_inp_pos();

        using inp_attn_type = std::conditional_t<iswa, llm_graph_input_attn_kv_unified_iswa, llm_graph_input_attn_kv_unified>;
        inp_attn_type * inp_attn = nullptr;

        if constexpr (iswa) {
            inp_attn = build_attn_inp_kv_unified_iswa();
        } else {
            inp_attn = build_attn_inp_kv_unified();
        }

        ggml_tensor * inp_out_ids = build_inp_out_ids();

        for (int il = 0; il < n_layer; ++il) {
            auto * residual = inpL;

            // self-attention
            {
                // rope freq factors for 128k context
                ggml_tensor * rope_factors = model.get_rope_factors(cparams, il);

                ggml_tensor* attn_norm_output = build_norm(inpL,
                        model.layers[il].attn_norm,
                        model.layers[il].attn_norm_b,
                        LLM_NORM_RMS, il);
                cb(attn_norm_output, "attn_norm", il);

                ggml_tensor * Qcur = nullptr;
                ggml_tensor * Kcur = nullptr;
                ggml_tensor * Vcur = nullptr;

                if (model.layers[il].wqkv) {
                    cur = build_lora_mm(model.layers[il].wqkv, attn_norm_output);
                    cb(cur, "wqkv", il);

                    Qcur = ggml_view_3d(ctx0, cur, n_embd_head, n_head,    n_tokens, n_embd_head * sizeof(float), cur->nb[1], 0 * sizeof(float) * (n_embd));
                    Kcur = ggml_view_3d(ctx0, cur, n_embd_head, n_head_kv, n_tokens, n_embd_head * sizeof(float), cur->nb[1], 1 * sizeof(float) * (n_embd));
                    Vcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd_gqa, n_tokens, cur->nb[1], 1 * sizeof(float) * (n_embd + n_embd_gqa)));
                } else {
                    Qcur = ggml_add(ctx0, build_lora_mm(model.layers[il].wq, attn_norm_output), model.layers[il].bq);
                    Kcur = ggml_add(ctx0, build_lora_mm(model.layers[il].wk, attn_norm_output), model.layers[il].bk);
                    Vcur = ggml_add(ctx0, build_lora_mm(model.layers[il].wv, attn_norm_output), model.layers[il].bv);
                    Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
                    Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
                }

                cb(Qcur, "Qcur", il);
                cb(Kcur, "Kcur", il);
                cb(Vcur, "Vcur", il);

                Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

                Qcur = ggml_rope_ext(
                        ctx0, Qcur, inp_pos, rope_factors,
                        n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow
                        );

                Kcur = ggml_rope_ext(
                        ctx0, Kcur, inp_pos, rope_factors,
                        n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow
                        );

                cb(Qcur, "Qcur", il);
                cb(Kcur, "Kcur", il);
                cb(Vcur, "Vcur", il);

                Qcur = ggml_scale(ctx0, Qcur, 1.0f / sqrtf(float(n_embd_head)));
                cb(Qcur, "Qcur", il);

                cur = build_attn(inp_attn,
                        model.layers[il].wo, model.layers[il].bo,
                        Qcur, Kcur, Vcur, nullptr, nullptr, 1.0f, il);
            }

            if (il == n_layer - 1 && inp_out_ids) {
                cur      = ggml_get_rows(ctx0, cur,      inp_out_ids);
                residual = ggml_get_rows(ctx0, residual, inp_out_ids);
            }

            cur = ggml_add(ctx0, cur, residual);
            residual = cur;

            cur = build_norm(cur,
                    model.layers[il].ffn_norm, model.layers[il].ffn_norm_b,
                    LLM_NORM_RMS, il);
            cb(cur, "ffn_norm", il);

            // feed-forward network
            if (model.layers[il].ffn_gate_inp == nullptr) {
                cur = build_ffn(cur,
                        model.layers[il].ffn_up,   NULL, NULL,
                        NULL,                      NULL, NULL,
                        model.layers[il].ffn_down, NULL, NULL,
                        NULL,
                        LLM_FFN_SWIGLU, LLM_FFN_SEQ, il);
                cb(cur, "ffn_out", il);
            } else {
                // MoE branch
                cur = build_moe_ffn(cur,
                        model.layers[il].ffn_gate_inp,
                        model.layers[il].ffn_up_exps,
                        model.layers[il].ffn_gate_exps,
                        model.layers[il].ffn_down_exps,
                        nullptr,
                        n_expert, n_expert_used,
                        LLM_FFN_SILU, true,
                        false, 0.0,
                        LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX,
                        il);
                cb(cur, "ffn_moe_out", il);
            }

            cur = ggml_add(ctx0, residual, cur);

            cur = build_cvec(cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = build_norm(inpL,
                model.output_norm,
                model.output_norm_b,
                LLM_NORM_RMS, -1);

        cb(cur, "result_norm", -1);
        res->t_embd = cur;

        cur = build_lora_mm(model.output, cur);

        if (model.output_b != nullptr) {
            cb(cur, "result_output_no_bias", -1);
            cur = ggml_add(ctx0, cur, model.output_b);
        }

        cb(cur, "result_output", -1);
        res->t_logits = cur;

        ggml_build_forward_expand(gf, cur);
    }
};

struct llm_build_plamo : public llm_graph_context {
    llm_build_plamo(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
        const int64_t n_embd_head = hparams.n_embd_head_v;

        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
        GGML_ASSERT(n_embd_head == hparams.n_rot);

        ggml_tensor * cur;
        ggml_tensor * inpL;

        inpL = build_inp_embd(model.tok_embd);

        // inp_pos - contains the positions
        ggml_tensor * inp_pos = build_inp_pos();

        auto * inp_attn = build_attn_inp_kv_unified();

        ggml_tensor * inp_out_ids = build_inp_out_ids();

        for (int il = 0; il < n_layer; ++il) {
            // norm
            cur = build_norm(inpL,
                    model.layers[il].attn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "attn_norm", il);

            ggml_tensor * sa_inp = cur;

            // self-attention
            {
                // compute Q and K and RoPE them
                ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
                cb(Qcur, "Qcur", il);

                ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
                cb(Kcur, "Kcur", il);

                ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
                cb(Vcur, "Vcur", il);

                Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
                Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
                Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

                Qcur = ggml_rope_ext(
                        ctx0, Qcur, inp_pos, nullptr,
                        n_embd_head, rope_type, n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow
                        );

                Kcur = ggml_rope_ext(
                        ctx0, Kcur, inp_pos, nullptr,
                        n_embd_head, rope_type, n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow
                        );

                cb(Qcur, "Qcur", il);
                cb(Kcur, "Kcur", il);
                cb(Vcur, "Vcur", il);

                cur = build_attn(inp_attn,
                        model.layers[il].wo, NULL,
                        Qcur, Kcur, Vcur, nullptr, nullptr, 1.0f/sqrtf(float(n_embd_head)), il);
            }

            if (il == n_layer - 1 && inp_out_ids) {
                cur    = ggml_get_rows(ctx0,    cur, inp_out_ids);
                sa_inp = ggml_get_rows(ctx0, sa_inp, inp_out_ids);
                inpL   = ggml_get_rows(ctx0,   inpL, inp_out_ids);
            }

            ggml_tensor * sa_out = cur;

            cur = sa_inp;

            // feed-forward network
            {
                cur = build_ffn(cur,
                        model.layers[il].ffn_up,   NULL, NULL,
                        model.layers[il].ffn_gate, NULL, NULL,
                        model.layers[il].ffn_down, NULL, NULL,
                        NULL,
                        LLM_FFN_SILU, LLM_FFN_PAR, il);
                cb(cur, "ffn_out", il);
            }

            cur = ggml_add(ctx0, cur, sa_out);
            cur = ggml_add(ctx0, cur, inpL);

            cur = build_cvec(cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        cur = build_norm(cur,
                model.output_norm, NULL,
                LLM_NORM_RMS, -1);

        cb(cur, "result_norm", -1);
        res->t_embd = cur;

        // lm_head
        cur = build_lora_mm(model.output, cur);

        cb(cur, "result_output", -1);
        res->t_logits = cur;

        ggml_build_forward_expand(gf, cur);
    }
};

struct llm_build_gpt2 : public llm_graph_context {
    llm_build_gpt2(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
        const int64_t n_embd_head = hparams.n_embd_head_v;
        const int64_t n_embd_gqa  = hparams.n_embd_v_gqa();

        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);

        ggml_tensor * cur;
        ggml_tensor * pos;
        ggml_tensor * inpL;

        inpL = build_inp_embd(model.tok_embd);

        // inp_pos - contains the positions
        ggml_tensor * inp_pos = build_inp_pos();

        auto * inp_attn = build_attn_inp_kv_unified();

        pos = ggml_get_rows(ctx0, model.pos_embd, inp_pos);
        cb(pos, "pos_embd", -1);

        inpL = ggml_add(ctx0, inpL, pos);
        cb(inpL, "inpL", -1);

        ggml_tensor * inp_out_ids = build_inp_out_ids();

        for (int il = 0; il < n_layer; ++il) {
            cur = build_norm(inpL,
                    model.layers[il].attn_norm,
                    model.layers[il].attn_norm_b,
                    LLM_NORM, il);
            cb(cur, "attn_norm", il);

            // self-attention
            {
                cur = build_lora_mm(model.layers[il].wqkv, cur);
                cb(cur, "wqkv", il);

                cur = ggml_add(ctx0, cur, model.layers[il].bqkv);
                cb(cur, "bqkv", il);

                ggml_tensor * Qcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd,     n_tokens, cur->nb[1], 0*sizeof(float)*(n_embd)));
                ggml_tensor * Kcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd_gqa, n_tokens, cur->nb[1], 1*sizeof(float)*(n_embd)));
                ggml_tensor * Vcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd_gqa, n_tokens, cur->nb[1], 1*sizeof(float)*(n_embd + n_embd_gqa)));

                cb(Qcur, "Qcur", il);
                cb(Kcur, "Kcur", il);
                cb(Vcur, "Vcur", il);

                Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
                Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
                Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

                cur = build_attn(inp_attn,
                        model.layers[il].wo, model.layers[il].bo,
                        Qcur, Kcur, Vcur, nullptr, nullptr, 1.0f/sqrtf(float(n_embd_head)), il);
            }

            if (il == n_layer - 1 && inp_out_ids) {
                cur  = ggml_get_rows(ctx0,  cur, inp_out_ids);
                inpL = ggml_get_rows(ctx0, inpL, inp_out_ids);
            }

            // add the input
            ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpL);
            cb(ffn_inp, "ffn_inp", il);

            // FF
            {
                cur = build_norm(ffn_inp,
                        model.layers[il].ffn_norm,
                        model.layers[il].ffn_norm_b,
                        LLM_NORM, il);
                cb(cur, "ffn_norm", il);

                cur = build_ffn(cur,
                        model.layers[il].ffn_up,   model.layers[il].ffn_up_b,   NULL,
                        NULL,                      NULL,                        NULL,
                        model.layers[il].ffn_down, model.layers[il].ffn_down_b, NULL,
                        NULL,
                        LLM_FFN_GELU, LLM_FFN_SEQ, il);
                cb(cur, "ffn_out", il);
            }

            cur = ggml_add(ctx0, cur, ffn_inp);

            cur = build_cvec(cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = build_norm(inpL,
                model.output_norm,
                model.output_norm_b,
                LLM_NORM, -1);

        cb(cur, "result_norm", -1);
        res->t_embd = cur;

        cur = build_lora_mm(model.output, cur);

        cb(cur, "result_output", -1);
        res->t_logits = cur;

        ggml_build_forward_expand(gf, cur);
    }
};

struct llm_build_codeshell : public llm_graph_context {
    llm_build_codeshell(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
        const int64_t n_embd_head = hparams.n_embd_head_v;
        const int64_t n_embd_gqa  = hparams.n_embd_v_gqa();

        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
        GGML_ASSERT(n_embd_head == hparams.n_rot);

        ggml_tensor * cur;
        ggml_tensor * inpL;

        inpL = build_inp_embd(model.tok_embd);

        // inp_pos - contains the positions
        ggml_tensor * inp_pos = build_inp_pos();

        auto * inp_attn = build_attn_inp_kv_unified();

        ggml_tensor * inp_out_ids = build_inp_out_ids();

        for (int il = 0; il < n_layer; ++il) {
            cur = build_norm(inpL,
                    model.layers[il].attn_norm,
                    model.layers[il].attn_norm_b,
                    LLM_NORM, il);
            cb(cur, "attn_norm", il);

            // self-attention
            {
                cur = build_lora_mm(model.layers[il].wqkv, cur);
                cb(cur, "wqkv", il);

                cur = ggml_add(ctx0, cur, model.layers[il].bqkv);
                cb(cur, "bqkv", il);

                ggml_tensor * Qcur = ggml_view_3d(ctx0, cur, n_embd_head, n_head,    n_tokens, n_embd_head*sizeof(float), cur->nb[1], 0*sizeof(float)*(n_embd));
                ggml_tensor * Kcur = ggml_view_3d(ctx0, cur, n_embd_head, n_head_kv, n_tokens, n_embd_head*sizeof(float), cur->nb[1], 1*sizeof(float)*(n_embd));
                ggml_tensor * Vcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd_gqa, n_tokens, cur->nb[1], 1*sizeof(float)*(n_embd + n_embd_gqa)));

                Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

                Qcur = ggml_rope_ext(
                        ctx0, Qcur, inp_pos, nullptr,
                        n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow
                        );

                Kcur = ggml_rope_ext(
                        ctx0, Kcur, inp_pos, nullptr,
                        n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow
                        );

                cb(Qcur, "Qcur", il);
                cb(Kcur, "Kcur", il);
                cb(Vcur, "Vcur", il);

                cur = build_attn(inp_attn,
                        model.layers[il].wo, model.layers[il].bo,
                        Qcur, Kcur, Vcur, nullptr, nullptr, 1.0f/sqrtf(float(n_embd_head)), il);
            }

            if (il == n_layer - 1 && inp_out_ids) {
                cur  = ggml_get_rows(ctx0,  cur, inp_out_ids);
                inpL = ggml_get_rows(ctx0, inpL, inp_out_ids);
            }

            // add the input
            ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpL);
            cb(ffn_inp, "ffn_inp", il);

            // FF
            {
                cur = build_norm(ffn_inp,
                        model.layers[il].ffn_norm,
                        model.layers[il].ffn_norm_b,
                        LLM_NORM, il);
                cb(cur, "ffn_norm", il);

                cur = build_ffn(cur,
                        model.layers[il].ffn_up,   model.layers[il].ffn_up_b,   NULL,
                        NULL,                      NULL,                        NULL,
                        model.layers[il].ffn_down, model.layers[il].ffn_down_b, NULL,
                        NULL,
                        LLM_FFN_GELU, LLM_FFN_SEQ, il);
                cb(cur, "ffn_out", il);
            }

            cur = ggml_add(ctx0, cur, ffn_inp);

            cur = build_cvec(cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = build_norm(inpL,
                model.output_norm,
                model.output_norm_b,
                LLM_NORM, -1);

        cb(cur, "result_norm", -1);
        res->t_embd = cur;

        cur = build_lora_mm(model.output, cur);

        cb(cur, "result_output", -1);
        res->t_logits = cur;

        ggml_build_forward_expand(gf, cur);
    }
};

struct llm_build_orion : public llm_graph_context {
    llm_build_orion(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
        const int64_t n_embd_head = hparams.n_embd_head_v;

        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
        GGML_ASSERT(n_embd_head == hparams.n_rot);

        ggml_tensor * cur;
        ggml_tensor * inpL;

        inpL = build_inp_embd(model.tok_embd);

        // inp_pos - contains the positions
        ggml_tensor * inp_pos = build_inp_pos();

        auto * inp_attn = build_attn_inp_kv_unified();

        ggml_tensor * inp_out_ids = build_inp_out_ids();

        for (int il = 0; il < n_layer; ++il) {
            ggml_tensor * inpSA = inpL;

            // norm
            cur = build_norm(inpL,
                    model.layers[il].attn_norm, model.layers[il].attn_norm_b,
                    LLM_NORM, il);
            cb(cur, "attn_norm", il);

            // self-attention
            {
                // compute Q and K and RoPE them
                ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
                cb(Qcur, "Qcur", il);
                // if (model.layers[il].bq) {
                //     Qcur = ggml_add(ctx0, Qcur, model.layers[il].bq);
                //     cb(Qcur, "Qcur", il);
                // }

                ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
                cb(Kcur, "Kcur", il);
                // if (model.layers[il].bk) {
                //     Kcur = ggml_add(ctx0, Kcur, model.layers[il].bk);
                //     cb(Kcur, "Kcur", il);
                // }

                ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
                cb(Vcur, "Vcur", il);
                // if (model.layers[il].bv) {
                //     Vcur = ggml_add(ctx0, Vcur, model.layers[il].bv);
                //     cb(Vcur, "Vcur", il);
                // }

                Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
                Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
                Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

                Qcur = ggml_rope_ext(
                        ctx0, Qcur, inp_pos, nullptr,
                        n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow
                        );

                Kcur = ggml_rope_ext(
                        ctx0, Kcur, inp_pos, nullptr,
                        n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow
                        );

                cb(Qcur, "Qcur", il);
                cb(Kcur, "Kcur", il);
                cb(Vcur, "Vcur", il);

                cur = build_attn(inp_attn,
                        model.layers[il].wo, NULL,
                        Qcur, Kcur, Vcur, nullptr, nullptr, 1.0f/sqrtf(float(n_embd_head)), il);
            }

            if (il == n_layer - 1 && inp_out_ids) {
                cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
                inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
            }

            ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
            cb(ffn_inp, "ffn_inp", il);

            // feed-forward network
            cur = build_norm(ffn_inp,
                    model.layers[il].ffn_norm, model.layers[il].ffn_norm_b,
                    LLM_NORM, il);
            cb(cur, "ffn_norm", il);

            cur = build_ffn(cur,
                    model.layers[il].ffn_up,   NULL, NULL,
                    model.layers[il].ffn_gate, NULL, NULL,
                    model.layers[il].ffn_down, NULL, NULL,
                    NULL,
                    LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(cur, "ffn_out", il);

            cur = ggml_add(ctx0, cur, ffn_inp);

            cur = build_cvec(cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        cur = build_norm(cur,
                model.output_norm, model.output_norm_b,
                LLM_NORM, -1);

        cb(cur, "result_norm", -1);
        res->t_embd = cur;

        // lm_head
        cur = build_lora_mm(model.output, cur);

        cb(cur, "result_output", -1);
        res->t_logits = cur;

        ggml_build_forward_expand(gf, cur);
    }
};

struct llm_build_internlm2 : public llm_graph_context {
    llm_build_internlm2(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
        const int64_t n_embd_head = hparams.n_embd_head_v;

        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
        GGML_ASSERT(n_embd_head == hparams.n_rot);

        ggml_tensor * cur;
        ggml_tensor * inpL;

        inpL = build_inp_embd(model.tok_embd);

        // inp_pos - contains the positions
        ggml_tensor * inp_pos = build_inp_pos();

        auto * inp_attn = build_attn_inp_kv_unified();

        ggml_tensor * inp_out_ids = build_inp_out_ids();

        for (int il = 0; il < n_layer; ++il) {
            ggml_tensor * inpSA = inpL;

            // norm
            cur = build_norm(inpL,
                    model.layers[il].attn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "attn_norm", il);

            // self-attention
            {
                // compute Q and K and RoPE them
                ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
                cb(Qcur, "Qcur", il);
                if (model.layers[il].bq) {
                    Qcur = ggml_add(ctx0, Qcur, model.layers[il].bq);
                    cb(Qcur, "Qcur", il);
                }

                ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
                cb(Kcur, "Kcur", il);
                if (model.layers[il].bk) {
                    Kcur = ggml_add(ctx0, Kcur, model.layers[il].bk);
                    cb(Kcur, "Kcur", il);
                }

                ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
                cb(Vcur, "Vcur", il);
                if (model.layers[il].bv) {
                    Vcur = ggml_add(ctx0, Vcur, model.layers[il].bv);
                    cb(Vcur, "Vcur", il);
                }

                Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
                Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
                Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

                Qcur = ggml_rope_ext(
                        ctx0, Qcur, inp_pos, nullptr,
                        n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow
                        );

                Kcur = ggml_rope_ext(
                        ctx0, Kcur, inp_pos, nullptr,
                        n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow
                        );

                cb(Qcur, "Qcur", il);
                cb(Kcur, "Kcur", il);
                cb(Vcur, "Vcur", il);

                cur = build_attn(inp_attn,
                        model.layers[il].wo, model.layers[il].bo,
                        Qcur, Kcur, Vcur, nullptr, nullptr, 1.0f/sqrtf(float(n_embd_head)), il);
            }

            if (il == n_layer - 1 && inp_out_ids) {
                cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
                inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
            }

            ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
            cb(ffn_inp, "ffn_inp", il);

            // feed-forward network
            cur = build_norm(ffn_inp,
                    model.layers[il].ffn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "ffn_norm", il);

            cur = build_ffn(cur,
                    model.layers[il].ffn_up,   NULL, NULL,
                    model.layers[il].ffn_gate, NULL, NULL,
                    model.layers[il].ffn_down, NULL, NULL,
                    NULL,
                    LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(cur, "ffn_out", il);

            cur = ggml_add(ctx0, cur, ffn_inp);

            cur = build_cvec(cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        cur = build_norm(cur,
                model.output_norm, NULL,
                LLM_NORM_RMS, -1);

        cb(cur, "result_norm", -1);
        res->t_embd = cur;

        // lm_head
        cur = build_lora_mm(model.output, cur);

        cb(cur, "result_output", -1);
        res->t_logits = cur;

        ggml_build_forward_expand(gf, cur);
    }
};

struct llm_build_minicpm3 : public llm_graph_context {
    llm_build_minicpm3(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
        //TODO: if the model varies, these parameters need to be read from the model
        const int64_t n_embd_base = 256;
        const float scale_embd  = 12.0f;
        const float scale_depth = 1.4f;
        const float kq_scale = 1.0f / sqrtf(float(hparams.n_embd_head_k));

        const uint32_t n_embd_head_qk_rope = hparams.n_rot;
        const uint32_t n_embd_head_qk_nope = hparams.n_embd_head_k - hparams.n_rot;
        const uint32_t kv_lora_rank = hparams.n_lora_kv;

        ggml_tensor * cur;
        ggml_tensor * inpL;

        inpL = build_inp_embd(model.tok_embd);

        // scale the input embeddings
        inpL = ggml_scale(ctx0, inpL, scale_embd);
        cb(inpL, "inp_scaled", -1);

        // inp_pos - contains the positions
        ggml_tensor * inp_pos = build_inp_pos();

        auto * inp_attn = build_attn_inp_kv_unified();

        ggml_tensor * inp_out_ids = build_inp_out_ids();

        for (int il = 0; il < n_layer; ++il) {
            ggml_tensor * inpSA = inpL;

            ggml_tensor * rope_factors = model.get_rope_factors(cparams, il);

            // norm
            cur = build_norm(inpL,
                    model.layers[il].attn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "attn_norm", il);

            // self_attention
            {
                ggml_tensor * q = NULL;
                // {n_embd, q_lora_rank} * {n_embd, n_tokens} -> {q_lora_rank, n_tokens}
                q = ggml_mul_mat(ctx0, model.layers[il].wq_a, cur);
                cb(q, "q", il);

                q = build_norm(q,
                        model.layers[il].attn_q_a_norm, NULL,
                        LLM_NORM_RMS, il);
                cb(q, "q", il);

                // {q_lora_rank, n_head * hparams.n_embd_head_k} * {q_lora_rank, n_tokens} -> {n_head * hparams.n_embd_head_k, n_tokens}
                q = ggml_mul_mat(ctx0, model.layers[il].wq_b, q);
                cb(q, "q", il);

                // split into {n_head * n_embd_head_qk_nope, n_tokens}
                ggml_tensor * q_nope = ggml_view_3d(ctx0, q, n_embd_head_qk_nope, n_head, n_tokens,
                        ggml_row_size(q->type, hparams.n_embd_head_k),
                        ggml_row_size(q->type, hparams.n_embd_head_k * n_head),
                        0);
                cb(q_nope, "q_nope", il);

                // and {n_head * n_embd_head_qk_rope, n_tokens}
                ggml_tensor * q_pe = ggml_view_3d(ctx0, q, n_embd_head_qk_rope, n_head, n_tokens,
                        ggml_row_size(q->type, hparams.n_embd_head_k),
                        ggml_row_size(q->type, hparams.n_embd_head_k * n_head),
                        ggml_row_size(q->type, n_embd_head_qk_nope));
                cb(q_pe, "q_pe", il);

                // {n_embd, kv_lora_rank + n_embd_head_qk_rope} * {n_embd, n_tokens} -> {kv_lora_rank + n_embd_head_qk_rope, n_tokens}
                ggml_tensor * kv_pe_compresseed = ggml_mul_mat(ctx0, model.layers[il].wkv_a_mqa, cur);
                cb(kv_pe_compresseed, "kv_pe_compresseed", il);

                // split into {kv_lora_rank, n_tokens}
                ggml_tensor * kv_compressed = ggml_view_2d(ctx0, kv_pe_compresseed, kv_lora_rank, n_tokens,
                        kv_pe_compresseed->nb[1],
                        0);
                cb(kv_compressed, "kv_compressed", il);

                // and {n_embd_head_qk_rope, n_tokens}
                ggml_tensor * k_pe = ggml_view_3d(ctx0, kv_pe_compresseed, n_embd_head_qk_rope, 1, n_tokens,
                        kv_pe_compresseed->nb[1],
                        kv_pe_compresseed->nb[1],
                        ggml_row_size(kv_pe_compresseed->type, kv_lora_rank));
                cb(k_pe, "k_pe", il);

                kv_compressed = build_norm(kv_compressed,
                        model.layers[il].attn_kv_a_norm, NULL,
                        LLM_NORM_RMS, il);
                cb(kv_compressed, "kv_compressed", il);

                // {kv_lora_rank, n_head * (n_embd_head_qk_nope + n_embd_head_v)} * {kv_lora_rank, n_tokens} -> {n_head * (n_embd_head_qk_nope + n_embd_head_v), n_tokens}
                ggml_tensor * kv = ggml_mul_mat(ctx0, model.layers[il].wkv_b, kv_compressed);
                cb(kv, "kv", il);

                // split into {n_head * n_embd_head_qk_nope, n_tokens}
                ggml_tensor * k_nope = ggml_view_3d(ctx0, kv, n_embd_head_qk_nope, n_head, n_tokens,
                        ggml_row_size(kv->type, n_embd_head_qk_nope + hparams.n_embd_head_v),
                        ggml_row_size(kv->type, n_head * (n_embd_head_qk_nope + hparams.n_embd_head_v)),
                        0);
                cb(k_nope, "k_nope", il);

                // and {n_head * n_embd_head_v, n_tokens}
                ggml_tensor * v_states = ggml_view_3d(ctx0, kv, hparams.n_embd_head_v, n_head, n_tokens,
                        ggml_row_size(kv->type, (n_embd_head_qk_nope + hparams.n_embd_head_v)),
                        ggml_row_size(kv->type, (n_embd_head_qk_nope + hparams.n_embd_head_v)*n_head),
                        ggml_row_size(kv->type, (n_embd_head_qk_nope)));
                cb(v_states, "v_states", il);

                v_states = ggml_cont(ctx0, v_states);
                cb(v_states, "v_states", il);

                q_pe = ggml_rope_ext(
                        ctx0, q_pe, inp_pos, rope_factors,
                        n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow
                        );
                cb(q_pe, "q_pe", il);

                // shared RoPE key
                k_pe = ggml_rope_ext(
                        ctx0, k_pe, inp_pos, rope_factors,
                        n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow
                        );
                cb(k_pe, "k_pe", il);

                ggml_tensor * q_states = ggml_concat(ctx0, q_nope, q_pe, 0);
                cb(q_states, "q_states", il);

                ggml_tensor * k_states = ggml_concat(ctx0, k_nope, ggml_repeat(ctx0, k_pe, q_pe), 0);
                cb(k_states, "k_states", il);

                cur = build_attn(inp_attn,
                        model.layers[il].wo, NULL,
                        q_states, k_states, v_states, nullptr, nullptr, kq_scale, il);
            }

            if (il == n_layer - 1 && inp_out_ids) {
                cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
                inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
            }

            // scale_res - scale the hidden states for residual connection
            const float scale_res = scale_depth/sqrtf(float(n_layer)); // TODO: is this correct?
            cur = ggml_scale(ctx0, cur, scale_res);
            cb(cur, "hidden_scaled", il);

            ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
            cb(ffn_inp, "ffn_inp", il);

            // feed-forward network
            {
                cur = build_norm(ffn_inp,
                        model.layers[il].ffn_norm, NULL,
                        LLM_NORM_RMS, il);
                cb(cur, "ffn_norm", il);

                cur = build_ffn(cur,
                        model.layers[il].ffn_up,   NULL, NULL,
                        model.layers[il].ffn_gate, NULL, NULL,
                        model.layers[il].ffn_down, NULL, NULL,
                        NULL,
                        LLM_FFN_SILU, LLM_FFN_PAR, il);
                cb(cur, "ffn_out", il);
            }

            // scale the hidden states for residual connection
            cur = ggml_scale(ctx0, cur, scale_res);
            cb(cur, "hidden_scaled_ffn", il);

            cur = ggml_add(ctx0, cur, ffn_inp);

            cur = build_cvec(cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        cur = build_norm(cur,
                model.output_norm, NULL,
                LLM_NORM_RMS, -1);

        cb(cur, "result_norm", -1);
        res->t_embd = cur;

        // lm_head scaling
        const float scale_lmhead = float(n_embd_base)/float(n_embd);
        cur = ggml_scale(ctx0, cur, scale_lmhead);
        cb(cur, "lmhead_scaling", -1);

        // lm_head
        cur = build_lora_mm(model.output, cur);

        cb(cur, "result_output", -1);
        res->t_logits = cur;

        ggml_build_forward_expand(gf, cur);
    }
};

struct llm_build_gemma : public llm_graph_context {
    llm_build_gemma(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
        const int64_t n_embd_head = hparams.n_embd_head_v;

        ggml_tensor * cur;
        ggml_tensor * inpL;

        inpL = build_inp_embd(model.tok_embd);

        inpL = ggml_scale(ctx0, inpL, sqrtf(n_embd));
        cb(inpL, "inp_scaled", -1);

        // inp_pos - contains the positions
        ggml_tensor * inp_pos = build_inp_pos();

        auto * inp_attn = build_attn_inp_kv_unified();

        ggml_tensor * inp_out_ids = build_inp_out_ids();

        for (int il = 0; il < n_layer; ++il) {
            // norm
            cur = build_norm(inpL,
                    model.layers[il].attn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "attn_norm", il);

            // self-attention
            {
                // compute Q and K and RoPE them
                ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
                cb(Qcur, "Qcur", il);

                ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
                cb(Kcur, "Kcur", il);

                ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
                cb(Vcur, "Vcur", il);

                Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
                Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
                Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

                Qcur = ggml_rope_ext(
                        ctx0, Qcur, inp_pos, nullptr,
                        n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow);

                Kcur = ggml_rope_ext(
                        ctx0, Kcur, inp_pos, nullptr,
                        n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow);

                cb(Qcur, "Qcur", il);
                cb(Kcur, "Kcur", il);
                cb(Vcur, "Vcur", il);

                Qcur = ggml_scale(ctx0, Qcur, 1.0f / sqrtf(float(n_embd_head)));
                cb(Qcur, "Qcur_scaled", il);

                cur = build_attn(inp_attn,
                        model.layers[il].wo, NULL,
                        Qcur, Kcur, Vcur, nullptr, nullptr, 1.0f, il);
            }

            if (il == n_layer - 1 && inp_out_ids) {
                cur  = ggml_get_rows(ctx0,  cur, inp_out_ids);
                inpL = ggml_get_rows(ctx0, inpL, inp_out_ids);
            }

            ggml_tensor * sa_out = ggml_add(ctx0, cur, inpL);
            cb(sa_out, "sa_out", il);

            cur = build_norm(sa_out,
                    model.layers[il].ffn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "ffn_norm", il);

            // feed-forward network
            {
                cur = build_ffn(cur,
                        model.layers[il].ffn_up,   NULL, NULL,
                        model.layers[il].ffn_gate, NULL, NULL,
                        model.layers[il].ffn_down, NULL, NULL,
                        NULL,
                        LLM_FFN_GELU, LLM_FFN_PAR, il);
                cb(cur, "ffn_out", il);
            }

            cur = ggml_add(ctx0, cur, sa_out);

            cur = build_cvec(cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        cur = build_norm(cur,
                model.output_norm, NULL,
                LLM_NORM_RMS, -1);

        cb(cur, "result_norm", -1);
        res->t_embd = cur;

        // lm_head
        cur = build_lora_mm(model.output, cur);

        cb(cur, "result_output", -1);
        res->t_logits = cur;

        ggml_build_forward_expand(gf, cur);
    }
};

struct llm_build_gemma2_iswa : public llm_graph_context {
    llm_build_gemma2_iswa(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
        const int64_t n_embd_head = hparams.n_embd_head_k;

        ggml_tensor * cur;
        ggml_tensor * inpL;

        inpL = build_inp_embd(model.tok_embd);

        inpL = ggml_scale(ctx0, inpL, sqrtf(n_embd));
        cb(inpL, "inp_scaled", -1);

        // inp_pos - contains the positions
        ggml_tensor * inp_pos = build_inp_pos();

        auto * inp_attn = build_attn_inp_kv_unified_iswa();

        ggml_tensor * inp_out_ids = build_inp_out_ids();

        for (int il = 0; il < n_layer; ++il) {
            // norm
            cur = build_norm(inpL,
                    model.layers[il].attn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "attn_norm", il);

            // self-attention
            {
                // compute Q and K and RoPE them
                ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
                cb(Qcur, "Qcur", il);

                ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
                cb(Kcur, "Kcur", il);

                ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
                cb(Vcur, "Vcur", il);

                Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
                Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
                Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

                Qcur = ggml_rope_ext(
                        ctx0, Qcur, inp_pos, nullptr,
                        n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow);

                Kcur = ggml_rope_ext(
                        ctx0, Kcur, inp_pos, nullptr,
                        n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow);

                cb(Qcur, "Qcur", il);
                cb(Kcur, "Kcur", il);
                cb(Vcur, "Vcur", il);

                Qcur = ggml_scale(ctx0, Qcur, hparams.f_attention_scale);

                cur = build_attn(inp_attn,
                        model.layers[il].wo, NULL,
                        Qcur, Kcur, Vcur, nullptr, nullptr, 1.0f, il);
            }

            if (il == n_layer - 1 && inp_out_ids) {
                cur  = ggml_get_rows(ctx0,  cur, inp_out_ids);
                inpL = ggml_get_rows(ctx0, inpL, inp_out_ids);
            }

            cur = build_norm(cur,
                    model.layers[il].attn_post_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "attn_post_norm", il);

            ggml_tensor * sa_out = ggml_add(ctx0, cur, inpL);
            cb(sa_out, "sa_out", il);

            cur = build_norm(sa_out,
                    model.layers[il].ffn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "ffn_norm", il);

            // feed-forward network
            {
                cur = build_ffn(cur,
                        model.layers[il].ffn_up,   NULL, NULL,
                        model.layers[il].ffn_gate, NULL, NULL,
                        model.layers[il].ffn_down, NULL, NULL,
                        NULL,
                        LLM_FFN_GELU, LLM_FFN_PAR, il);
                cb(cur, "ffn_out", il);
            }

            cur = build_norm(cur,
                    model.layers[il].ffn_post_norm, NULL,
                    LLM_NORM_RMS, -1);
            cb(cur, "ffn_post_norm", -1);

            cur = ggml_add(ctx0, cur, sa_out);

            cur = build_cvec(cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        cur = build_norm(cur,
                model.output_norm, NULL,
                LLM_NORM_RMS, -1);

        cb(cur, "result_norm", -1);
        res->t_embd = cur;

        // lm_head
        cur = build_lora_mm(model.output, cur);

        // final logit soft-capping
        cur = ggml_scale(ctx0, cur, 1.0f / hparams.f_final_logit_softcapping);
        cur = ggml_tanh(ctx0, cur);
        cur = ggml_scale(ctx0, cur, hparams.f_final_logit_softcapping);

        cb(cur, "result_output", -1);
        res->t_logits = cur;

        ggml_build_forward_expand(gf, cur);
    }
};

struct llm_build_gemma3_iswa : public llm_graph_context {
    llm_build_gemma3_iswa(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
        const int64_t n_embd_head = hparams.n_embd_head_k;

        ggml_tensor * cur;
        ggml_tensor * inpL;

        inpL = build_inp_embd(model.tok_embd);

        // important: do not normalize weights for raw embeddings input (i.e. encoded image emdeddings)
        if (ubatch.token) {
            inpL = ggml_scale(ctx0, inpL, sqrtf(n_embd));
            cb(inpL, "inp_scaled", -1);
        }

        // inp_pos - contains the positions
        ggml_tensor * inp_pos = build_inp_pos();

        // TODO: is causal == true correct? might need some changes
        auto * inp_attn = build_attn_inp_kv_unified_iswa();

        ggml_tensor * inp_out_ids = build_inp_out_ids();

        for (int il = 0; il < n_layer; ++il) {
            const float freq_base_l  = model.get_rope_freq_base (cparams, il);
            const float freq_scale_l = model.get_rope_freq_scale(cparams, il);

            // norm
            cur = build_norm(inpL, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, il);
            cb(cur, "attn_norm", il);

            // self-attention
            {
                // compute Q and K and RoPE them
                ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
                cb(Qcur, "Qcur", il);

                ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
                cb(Kcur, "Kcur", il);

                ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
                cb(Vcur, "Vcur", il);

                Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
                Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
                Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

                Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, NULL, LLM_NORM_RMS, il);
                cb(Qcur, "Qcur_normed", il);

                Qcur = ggml_rope_ext(
                        ctx0, Qcur, inp_pos, nullptr,
                        n_rot, rope_type, n_ctx_orig, freq_base_l, freq_scale_l,
                        ext_factor, attn_factor, beta_fast, beta_slow);

                Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, NULL, LLM_NORM_RMS, il);
                cb(Kcur, "Kcur_normed", il);

                Kcur = ggml_rope_ext(
                        ctx0, Kcur, inp_pos, nullptr,
                        n_rot, rope_type, n_ctx_orig, freq_base_l, freq_scale_l,
                        ext_factor, attn_factor, beta_fast, beta_slow);

                cb(Qcur, "Qcur", il);
                cb(Kcur, "Kcur", il);
                cb(Vcur, "Vcur", il);

                // ref: https://github.com/google/gemma_pytorch/blob/014acb7ac4563a5f77c76d7ff98f31b568c16508/gemma/model.py#L315
                Qcur = ggml_scale(ctx0, Qcur, hparams.f_attention_scale);

                cur = build_attn(inp_attn,
                        model.layers[il].wo, NULL,
                        Qcur, Kcur, Vcur, nullptr, nullptr, 1.0f, il);
            }

            if (il == n_layer - 1 && inp_out_ids) {
                cur  = ggml_get_rows(ctx0,  cur, inp_out_ids);
                inpL = ggml_get_rows(ctx0, inpL, inp_out_ids);
            }

            cur = build_norm(cur,
                    model.layers[il].attn_post_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "attn_post_norm", il);

            ggml_tensor * sa_out = ggml_add(ctx0, cur, inpL);
            cb(sa_out, "sa_out", il);

            cur = build_norm(sa_out,
                    model.layers[il].ffn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "ffn_norm", il);

            // feed-forward network
            {
                cur = build_ffn(cur,
                        model.layers[il].ffn_up,   NULL, NULL,
                        model.layers[il].ffn_gate, NULL, NULL,
                        model.layers[il].ffn_down, NULL, NULL,
                        NULL,
                        LLM_FFN_GELU, LLM_FFN_PAR, il);
                cb(cur, "ffn_out", il);
            }

            cur = build_norm(cur,
                    model.layers[il].ffn_post_norm, NULL,
                    LLM_NORM_RMS, -1);
            cb(cur, "ffn_post_norm", -1);

            cur = ggml_add(ctx0, cur, sa_out);

            cur = build_cvec(cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        cur = build_norm(cur,
                model.output_norm, NULL,
                LLM_NORM_RMS, -1);

        cb(cur, "result_norm", -1);
        res->t_embd = cur;

        // lm_head
        cur = build_lora_mm(model.output, cur);

        cb(cur, "result_output", -1);
        res->t_logits = cur;

        ggml_build_forward_expand(gf, cur);
    }
};

struct llm_build_gemma3n_iswa : public llm_graph_context {
    const llama_model & model;

    const int64_t n_embd_head;
    const int64_t n_embd_altup;
    const int64_t n_altup;
    const int     i_altup_act;
    const int     n_layer_kv = 20; // number of layers having KV [KV_REUSE]
    const int     n_layer_sparsity = 10; // number of layers using activation sparsity
    const float   f_sparsity_std_mul = 1.6448533535003662f; // std_multiplier = normal_dist.icdf(0.95)

    llm_build_gemma3n_iswa(const llama_model & model, const llm_graph_params & params)
            : llm_graph_context(params),
              model(model),
              n_embd_head(model.hparams.n_embd_head_k),
              n_embd_altup(model.hparams.n_embd_altup),
              n_altup(model.hparams.n_altup),
              i_altup_act(model.hparams.i_altup_act) {
        ggml_tensor * cur;
        ggml_tensor * inpL;

        inpL = build_inp_embd(model.tok_embd);

        // important: do not normalize weights for raw embeddings input (i.e. encoded image emdeddings)
        if (ubatch.token) {
            inpL = ggml_scale(ctx0, inpL, sqrtf(n_embd));
            cb(inpL, "inp_scaled", -1);
        }

        // inp_pos - contains the positions
        ggml_tensor * inp_pos = build_inp_pos();

        // TODO: is causal == true correct? might need some changes
        auto * inp_attn = build_attn_inp_kv_unified_iswa();

        // inp_per_layer shape: [n_embd_altup, n_tokens, n_layer]
        ggml_tensor * inp_per_layer = project_per_layer_inputs(inpL, get_per_layer_inputs());

        // inpL now has only 1 altup, project it to the rest of the altups
        // these "added" altups will be concat to the last dim of inpL
        {
            ggml_tensor * target_magnitude = calc_magnitude(inpL);
            ggml_tensor * inp_repeated = ggml_repeat_4d(ctx0, inpL, n_embd, n_tokens, n_altup - 1, 1);
            ggml_tensor * altup_added = ggml_mul_mat(ctx0, model.altup_proj, inp_repeated); // shape: [n_embd, n_tokens, n_altup - 1]
            ggml_tensor * new_magnitude = calc_magnitude(altup_added);
            altup_added = ggml_div(ctx0,
                                ggml_mul(ctx0, altup_added, target_magnitude),
                                new_magnitude);
            inpL = ggml_concat(ctx0, inpL, altup_added, 2); // shape: [n_embd, n_tokens, n_altup]
            cb(inpL, "inp_stacked", -1);
        }

        // inpL now has shape:          [n_embd,       n_tokens, n_altup]
        // inp_per_layer now has shape: [n_embd_altup, n_tokens, n_layer]

        for (int il = 0; il < n_layer; ++il) {
            // this block is made to be closely resemble Gemma3p5DecoderLayer on python code
            const bool has_kv = (il < n_layer_kv);

            const float freq_base_l  = model.get_rope_freq_base (cparams, il);
            const float freq_scale_l = model.get_rope_freq_scale(cparams, il);

            ggml_tensor * cur = inpL; // [n_embd, n_tokens, n_altup]
            ggml_tensor * predictions = altup_predict(cur, il); // [n_embd, n_tokens, n_altup]

            // predicted value will go through self-attention and laurel
            ggml_tensor * active_prediction = view_2d_slice(predictions, i_altup_act); // [n_embd, n_tokens]
            cur = active_prediction;
            cb(cur, "active_prediction", il);

            // norm
            cur = build_norm(cur, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, il);
            cb(cur, "attn_norm", il);

            // laurel
            ggml_tensor * laurel_out = laurel(cur, il); // [n_embd, n_tokens]

            // self-attention
            if (has_kv) {
                // compute Q and K and RoPE them
                ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
                cb(Qcur, "Qcur", il);

                ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
                cb(Kcur, "Kcur", il);

                ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
                cb(Vcur, "Vcur", il);

                Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
                Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
                Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

                Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, NULL, LLM_NORM_RMS, il);
                Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, NULL, LLM_NORM_RMS, il);
                Vcur = ggml_rms_norm(ctx0, Vcur, hparams.f_norm_rms_eps);

                cb(Qcur, "Qcur_normed", il);
                cb(Kcur, "Kcur_normed", il);
                cb(Vcur, "Vcur_normed", il);

                Qcur = ggml_rope_ext(
                        ctx0, Qcur, inp_pos, nullptr,
                        n_rot, rope_type, n_ctx_orig, freq_base_l, freq_scale_l,
                        ext_factor, attn_factor, beta_fast, beta_slow);

                Kcur = ggml_rope_ext(
                        ctx0, Kcur, inp_pos, nullptr,
                        n_rot, rope_type, n_ctx_orig, freq_base_l, freq_scale_l,
                        ext_factor, attn_factor, beta_fast, beta_slow);

                cb(Qcur, "Qcur_pos", il);
                cb(Kcur, "Kcur_pos", il);

                cur = build_attn(inp_attn,
                        model.layers[il].wo, NULL,
                        Qcur, Kcur, Vcur, nullptr, nullptr, hparams.f_attention_scale, il);
            } else {
                // no KV layers
                ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
                cb(Qcur, "Qcur", il);
                Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens);

                Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, NULL, LLM_NORM_RMS, il);
                cb(Qcur, "Qcur_normed", il);

                Qcur = ggml_rope_ext(
                        ctx0, Qcur, inp_pos, nullptr,
                        n_rot, rope_type, n_ctx_orig, freq_base_l, freq_scale_l,
                        ext_factor, attn_factor, beta_fast, beta_slow);
                cb(Qcur, "Qcur_pos", il);

                cur = build_attn(inp_attn,
                    model.layers[il].wo, NULL,
                    Qcur, nullptr, nullptr, nullptr, nullptr, hparams.f_attention_scale, il);
            }

            cur = build_norm(cur,
                    model.layers[il].attn_post_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "attn_post_norm", il);

            cur = ggml_add(ctx0, cur, active_prediction); // [n_embd, n_tokens]
            cb(cur, "attn_gated", il);

            ggml_tensor * attn_laurel = ggml_scale(ctx0,
                                            ggml_add(ctx0, cur, laurel_out),
                                            1.0f / sqrtf(2.0f)); // [n_embd, n_tokens]
            cb(attn_laurel, "attn_laurel", il);

            cur = build_norm(attn_laurel,
                    model.layers[il].ffn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "ffn_norm", il);

            // feed-forward network
            {
                ggml_tensor * up_proj   = build_lora_mm(model.layers[il].ffn_up,   cur);
                ggml_tensor * gate_proj = build_lora_mm(model.layers[il].ffn_gate, cur);

                if (il < n_layer_sparsity) {
                    // apply activation sparsity
                    gate_proj = gaussian_topk(gate_proj);
                }
                gate_proj = ggml_gelu(ctx0, gate_proj);

                cur = ggml_mul(ctx0, up_proj, gate_proj);
                cur = build_lora_mm(model.layers[il].ffn_down, cur);
                cb(cur, "ffn_out", il);
            }

            cur = build_norm(cur,
                    model.layers[il].ffn_post_norm, NULL,
                    LLM_NORM_RMS, -1);
            cb(cur, "ffn_post_norm", il);

            ggml_tensor * attn_ffw_laurel_gated = ggml_add(ctx0, cur, attn_laurel); // [n_embd, n_tokens]
            cb(attn_ffw_laurel_gated, "attn_ffw_laurel_gated", il);

            ggml_tensor * corrected = altup_correct(predictions, attn_ffw_laurel_gated, il); // [n_embd, n_tokens, n_altup]

            ggml_tensor * first_prediction; // [n_embd, n_tokens]
            {
                first_prediction = view_2d_slice(corrected, i_altup_act); // [n_embd, n_tokens]
                first_prediction = ggml_mul(ctx0, first_prediction, model.layers[il].altup_correct_scale);
                first_prediction = build_lora_mm(model.layers[il].per_layer_inp_gate, first_prediction);
                first_prediction = ggml_gelu(ctx0, first_prediction); // [n_embd_altup, n_tokens]
                cb(first_prediction, "first_prediction_gated", il);
                ggml_tensor * inp_this_layer = view_2d_slice(inp_per_layer, il); // [n_embd_altup, n_tokens]
                first_prediction = ggml_mul(ctx0, first_prediction, inp_this_layer); // [n_embd_altup, n_tokens]
                cb(first_prediction, "first_prediction_scaled", il);

                first_prediction = build_lora_mm(model.layers[il].per_layer_proj, first_prediction); // [n_embd, n_tokens]
                first_prediction = build_norm(first_prediction,
                        model.layers[il].per_layer_post_norm, NULL,
                        LLM_NORM_RMS, il);
                cb(first_prediction, "first_prediction_out", il);
            }

            // equivalent to python code: corrected_predictions[1:] += first_prediction
            {
                ggml_tensor * slice_first = view_2d_slice(corrected, 0);
                ggml_tensor * slice_rest  = ggml_view_3d(ctx0, corrected, n_embd, n_tokens, n_altup - 1,
                                                    ggml_row_size(corrected->type, n_embd),
                                                    ggml_row_size(corrected->type, n_embd*n_tokens),
                                                    n_embd*n_tokens*ggml_element_size(corrected));
                ggml_tensor * tmp = ggml_add(ctx0, slice_rest, first_prediction); // [n_embd, n_tokens, n_altup - 1]
                corrected = ggml_concat(ctx0, slice_first, tmp, 2); // [n_embd, n_tokens, n_altup]
            }

            cur = corrected; // [n_embd, n_tokens, n_altup]
            cur = build_cvec(cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL; // [n_embd, n_tokens, n_altup]

        // cur now has multiple altup(s), we want to merge them back to 1 altup
        {
            ggml_tensor * target_magnitude = calc_magnitude(view_2d_slice(cur, i_altup_act)); // [n_embd, n_tokens]
            // do a view to skip the first slice (active altup)
            ggml_tensor * alt_slice = ggml_view_3d(ctx0, cur, n_embd, n_tokens, n_altup - 1,
                                                    ggml_row_size(cur->type, n_embd),
                                                    ggml_row_size(cur->type, n_embd*n_tokens),
                                                    n_embd*n_tokens*ggml_element_size(cur));
            ggml_tensor * altup_unembd = ggml_mul_mat(ctx0, model.altup_unembd_proj, alt_slice); // shape: [n_embd, n_tokens, n_altup - 1]
            ggml_tensor * new_magnitude = calc_magnitude(altup_unembd);
            altup_unembd = ggml_div(ctx0,
                                ggml_mul(ctx0, altup_unembd, target_magnitude),
                                new_magnitude);
            cb(altup_unembd, "altup_unembd", -1);

            // equivalent to torch.mean(hidden_states, dim=0)
            cur = view_2d_slice(cur, 0); // [n_embd, n_tokens]
            for (int i = 0; i < n_altup - 1; ++i) {
                cur = ggml_add(ctx0, cur, view_2d_slice(altup_unembd, i));
            }
            cur = ggml_scale(ctx0, cur, 1.0f / float(n_altup)); // [n_embd, n_tokens]
            cb(cur, "unembd_merged", -1);
        }

        // cur now has shape: [n_embd, n_tokens]

        // TODO: move this to right after the last KV layer
        {
            // skip computing output for unused tokens
            ggml_tensor * inp_out_ids = build_inp_out_ids();
            cur  = ggml_get_rows(ctx0,  cur, inp_out_ids);
        }

        cur = build_norm(cur,
                model.output_norm, NULL,
                LLM_NORM_RMS, -1);

        cb(cur, "result_norm", -1);
        res->t_embd = cur;

        cur = build_lora_mm(model.output, cur);

        {
            // final logit soft-capping
            cur = ggml_scale(ctx0, cur, 1.0f / hparams.f_final_logit_softcapping);
            cur = ggml_tanh(ctx0, cur);
            cur = ggml_scale(ctx0, cur, hparams.f_final_logit_softcapping);
        }

        cb(cur, "result_output", -1);
        res->t_logits = cur;

        ggml_build_forward_expand(gf, cur);
    }

    ggml_tensor * calc_magnitude(ggml_tensor * x) {
        return ggml_sqrt(ctx0, ggml_sum_rows(ctx0, ggml_sqr(ctx0, x)));
    }

    // get 2D slice view from a 3D tensor, the idx corresponds to the 3rd dim
    ggml_tensor * view_2d_slice(ggml_tensor * x, int idx) {
        GGML_ASSERT(idx < (int)x->ne[2]);
        return ggml_view_2d(ctx0, x, x->ne[0], x->ne[1],
                            ggml_row_size(x->type, x->ne[0]),
                            idx * x->ne[0] * x->ne[1] * ggml_element_size(x));
    }

    // equivalent to get_per_layer_inputs() in python code
    // output shape: [n_embd_altup, n_layer, n_tokens]
    ggml_tensor * get_per_layer_inputs() {
        auto inp = std::make_unique<llm_graph_input_embd>();
        ggml_tensor * inp_per_layer;
        if (ubatch.token) {
            inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, ubatch.n_tokens);
            ggml_set_input(inp->tokens);
            res->t_tokens = inp->tokens;
            inp_per_layer = ggml_get_rows(ctx0, model.tok_embd_per_layer, inp->tokens);
            inp_per_layer = ggml_reshape_3d(ctx0, inp_per_layer, n_embd_altup, n_layer, n_tokens);
            inp_per_layer = ggml_scale(ctx0, inp_per_layer, sqrtf((float)n_embd_altup));
            cb(inp_per_layer, "inp_per_layer_selected", -1);
        } else {
            GGML_ABORT("TODO: support embd input");
        }
        res->add_input(std::move(inp));
        return inp_per_layer;
    }

    // equivalent to project_per_layer_inputs() in python code
    // this calculates the per-layer inputs, so the final tensor shape will have n_layer as the last dim
    // output shape: [n_embd_altup, n_tokens, n_layer]
    ggml_tensor * project_per_layer_inputs(ggml_tensor * inputs_embeds, ggml_tensor * inp_per_layer) {
        const float per_layer_projection_scale = 1.0f / sqrtf((float)n_embd);
        const float per_layer_input_scale      = 1.0f / sqrtf(2.0f);

        ggml_tensor * per_layer_proj = ggml_mul_mat(ctx0, model.per_layer_model_proj, inputs_embeds);
        per_layer_proj = ggml_scale(ctx0, per_layer_proj, per_layer_projection_scale);
        per_layer_proj = ggml_reshape_3d(ctx0, per_layer_proj, n_embd_altup, n_layer, n_tokens);
        per_layer_proj = build_norm(per_layer_proj,
                                    model.per_layer_proj_norm, NULL,
                                    LLM_NORM_RMS, -1); // [n_embd_altup, n_layer, n_tokens]
        cb(per_layer_proj, "per_layer_proj", -1);

        inp_per_layer = ggml_add(ctx0, inp_per_layer, per_layer_proj);
        inp_per_layer = ggml_scale(ctx0, inp_per_layer, per_layer_input_scale);
        cb(inp_per_layer, "inp_per_layer", -1);

        // permute to shape: [n_embd_altup, n_tokens, n_layer]
        inp_per_layer = ggml_cont(ctx0, ggml_permute(ctx0, inp_per_layer, 0, 2, 1, 3));
        return inp_per_layer;
    }

    // input cur shape: [n_altup, n_tokens]
    // output    shape: [n_altup, n_tokens]
    ggml_tensor * laurel(ggml_tensor * cur, int il) {
        ggml_tensor * tmp = cur;
        tmp = build_lora_mm(model.layers[il].laurel_l, tmp);
        tmp = build_lora_mm(model.layers[il].laurel_r, tmp);
        tmp = build_norm(tmp, model.layers[il].laurel_post_norm, NULL, LLM_NORM_RMS, il);
        tmp = ggml_add(ctx0, tmp, cur);
        cb(tmp, "laurel_out", il);
        return tmp;
    }

    // input x shape: [n_embd, n_tokens]
    // output  shape: [n_embd, n_tokens]
    ggml_tensor * gaussian_topk(ggml_tensor * x) {
        ggml_tensor * mean = ggml_mean(ctx0, x);
        ggml_tensor * std  = ggml_sqrt(ctx0, ggml_scale(ctx0,
            ggml_sum_rows(ctx0, ggml_sqr(ctx0, ggml_sub(ctx0, x, mean))),
            1.0f / (float)(x->ne[0] - 1)
        ));
        ggml_tensor * cutoff_x = ggml_add(ctx0, mean, ggml_scale(ctx0, std, f_sparsity_std_mul));
        return ggml_relu(ctx0, ggml_sub(ctx0, x, cutoff_x));
    }

    //
    // altup functions
    //

    // equivalent to compute_router_modalities() in python code
    // input x shape: [n_embd,  n_tokens]
    // output  shape: [n_altup, n_tokens]
    ggml_tensor * altup_compute_router_modalities(ggml_tensor * x, int il) {
        ggml_tensor * router_inputs = build_norm(x,
            model.layers[il].altup_router_norm, NULL,
            LLM_NORM_RMS, il);

        // router_input_scale
        router_inputs = ggml_scale(ctx0, router_inputs, 1.0f / (float)n_embd);

        ggml_tensor * output = ggml_mul_mat(ctx0, model.layers[il].altup_router, router_inputs);
        return ggml_tanh(ctx0, output); // [n_altup, n_tokens]
    }

    // input cur shape: [n_embd, n_tokens, n_altup]
    // output    shape: [n_embd, n_tokens, n_altup]
    ggml_tensor * altup_predict(ggml_tensor * cur, int il) {
        ggml_tensor * activated = view_2d_slice(cur, i_altup_act); // [n_embd, n_tokens]
        ggml_tensor * modalities = altup_compute_router_modalities(activated, il); // [n_altup, n_tokens]
        cb(modalities, "modalities", il);

        ggml_tensor * all_coefs = build_lora_mm(model.layers[il].altup_predict_coef, modalities);
        cb(all_coefs, "all_coefs", il);
        // first dim now having n_altup^2 elements, we reshape it to 2D (so we end up with 3D tensor)
        all_coefs = ggml_reshape_3d(ctx0, all_coefs, n_altup, n_altup, n_tokens);

        // permute to [n_altup, n_embd, n_tokens]
        ggml_tensor * cur_permuted = ggml_cont(ctx0, ggml_permute(ctx0, cur, 1, 2, 0, 3));
        ggml_tensor * predictions = ggml_mul_mat(ctx0, cur_permuted, all_coefs); // [n_altup, n_embd, n_tokens]

        // final shape must be the same as cur: [n_embd, n_tokens, n_altup]
        predictions = ggml_cont(ctx0, ggml_permute(ctx0, predictions, 0, 2, 1, 3));
        predictions = ggml_add(ctx0, predictions, cur);
        cb(predictions, "predictions", il);

        return predictions;
    }

    // input predictions       shape: [n_embd, n_tokens, n_altup]
    // input activated         shape: [n_embd, n_tokens]
    // output                  shape: [n_embd, n_tokens, n_altup]
    ggml_tensor * altup_correct(ggml_tensor * predictions, ggml_tensor * activated, int il) {
        ggml_tensor * modalities = altup_compute_router_modalities(activated, il); // [n_altup, n_tokens]
        cb(modalities, "modalities", il);

        ggml_tensor * active_prediction = view_2d_slice(predictions, i_altup_act);
        ggml_tensor * innovation = ggml_sub(ctx0, activated, active_prediction); // [n_embd, n_tokens]
        cb(innovation, "innovation", il);

        ggml_tensor * all_coefs = build_lora_mm(model.layers[il].altup_correct_coef, modalities); // [n_altup, n_tokens]
        all_coefs = ggml_scale_bias(ctx0, all_coefs, 1.0f, 1.0f); // + 1.0
        cb(all_coefs, "all_coefs", il);
        all_coefs = ggml_cont(ctx0, ggml_transpose(ctx0, all_coefs)); // [n_tokens, n_altup]
        all_coefs = ggml_reshape_3d(ctx0, all_coefs, 1, n_tokens, n_altup); // [1, n_tokens, n_altup]

        innovation = ggml_repeat_4d(ctx0, innovation, n_embd, n_tokens, n_altup, 1);
        ggml_tensor * corrected = ggml_mul(ctx0, innovation, all_coefs); // [n_embd, n_tokens, n_altup]
        corrected = ggml_add(ctx0, corrected, predictions); // [n_embd, n_tokens, n_altup]
        cb(corrected, "corrected", il);

        return corrected;
    }
};

// TODO: move up next to build_starcoder
struct llm_build_starcoder2 : public llm_graph_context {
    llm_build_starcoder2(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
        const int64_t n_embd_head = hparams.n_embd_head_v;

        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
        GGML_ASSERT(n_embd_head == hparams.n_rot);

        ggml_tensor * cur;
        ggml_tensor * inpL;

        inpL = build_inp_embd(model.tok_embd);

        // inp_pos - contains the positions
        ggml_tensor * inp_pos = build_inp_pos();

        auto * inp_attn = build_attn_inp_kv_unified();

        ggml_tensor * inp_out_ids = build_inp_out_ids();

        for (int il = 0; il < n_layer; ++il) {
            ggml_tensor * inpSA = inpL;

            // norm
            cur = build_norm(inpL,
                    model.layers[il].attn_norm, model.layers[il].attn_norm_b,
                    LLM_NORM, il);
            cb(cur, "attn_norm", il);

            // self-attention
            {
                // compute Q and K and RoPE them
                ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
                cb(Qcur, "Qcur", il);
                if (model.layers[il].bq) {
                    Qcur = ggml_add(ctx0, Qcur, model.layers[il].bq);
                    cb(Qcur, "Qcur", il);
                }

                ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
                cb(Kcur, "Kcur", il);
                if (model.layers[il].bk) {
                    Kcur = ggml_add(ctx0, Kcur, model.layers[il].bk);
                    cb(Kcur, "Kcur", il);
                }

                ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
                cb(Vcur, "Vcur", il);
                if (model.layers[il].bv) {
                    Vcur = ggml_add(ctx0, Vcur, model.layers[il].bv);
                    cb(Vcur, "Vcur", il);
                }

                Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
                Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
                Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

                Qcur = ggml_rope_ext(
                        ctx0, Qcur, inp_pos, nullptr,
                        n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow
                        );

                Kcur = ggml_rope_ext(
                        ctx0, Kcur, inp_pos, nullptr,
                        n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow
                        );

                cb(Qcur, "Qcur", il);
                cb(Kcur, "Kcur", il);
                cb(Vcur, "Vcur", il);

                cur = build_attn(inp_attn,
                        model.layers[il].wo, model.layers[il].bo,
                        Qcur, Kcur, Vcur, nullptr, nullptr, 1.0f/sqrtf(float(n_embd_head)), il);
            }

            if (il == n_layer - 1 && inp_out_ids) {
                cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
                inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
            }

            ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
            cb(ffn_inp, "ffn_inp", il);

            // feed-forward network

            cur = build_norm(ffn_inp,
                    model.layers[il].ffn_norm, model.layers[il].ffn_norm_b,
                    LLM_NORM, il);
            cb(cur, "ffn_norm", il);

            cur = build_ffn(cur,
                    model.layers[il].ffn_up,   model.layers[il].ffn_up_b,   NULL,
                    NULL,                      NULL,                        NULL,
                    model.layers[il].ffn_down, model.layers[il].ffn_down_b, NULL,
                    NULL,
                    LLM_FFN_GELU, LLM_FFN_SEQ, il);
            cb(cur, "ffn_out", il);

            cur = ggml_add(ctx0, cur, ffn_inp);

            cur = build_cvec(cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        cur = build_norm(cur,
                model.output_norm, model.output_norm_b,
                LLM_NORM, -1);

        cb(cur, "result_norm", -1);
        res->t_embd = cur;

        // lm_head
        cur = build_lora_mm(model.output, cur);

        cb(cur, "result_output", -1);
        res->t_logits = cur;

        ggml_build_forward_expand(gf, cur);
    }
};

struct llm_graph_context_mamba : public llm_graph_context {
    llm_graph_context_mamba(const llm_graph_params & params) : llm_graph_context(params) {}

    ggml_tensor * build_mamba_layer(
        llm_graph_input_rs * inp,
               ggml_tensor * cur,
         const llama_model & model,
        const llama_ubatch & ubatch,
                       int   il) {

        const auto * mctx_cur = inp->mctx;

        const auto kv_head = mctx_cur->get_head();

        const auto & layer = model.layers[il];

        const int64_t d_conv  = hparams.ssm_d_conv;
        const int64_t d_inner = hparams.ssm_d_inner;
        const int64_t d_state = hparams.ssm_d_state;
        const int64_t dt_rank = hparams.ssm_dt_rank;
        const int64_t n_head  = d_inner;
        const int64_t head_dim = 1;
        const int64_t n_seqs  = ubatch.n_seqs;
        // Some variants of Mamba arch (e.g. FalconMamba do apply layer norm on B and Dt layers)
        const bool ssm_dt_b_c_rms = hparams.ssm_dt_b_c_rms;

        const int64_t n_seq_tokens = ubatch.n_seq_tokens;

        GGML_ASSERT(n_seqs != 0);
        GGML_ASSERT(ubatch.equal_seqs());
        GGML_ASSERT(ubatch.n_tokens == n_seq_tokens * n_seqs);

        ggml_tensor * conv_states_all = mctx_cur->get_r_l(il);
        ggml_tensor * ssm_states_all  = mctx_cur->get_s_l(il);

        ggml_tensor * conv = build_rs(inp, conv_states_all, hparams.n_embd_r(), n_seqs);
        conv = ggml_reshape_3d(ctx0, conv, d_conv - 1, d_inner, n_seqs);

        // {n_embd, n_tokens} => {n_embd, n_seq_tokens, n_seqs}
        cur = ggml_reshape_3d(ctx0, cur, cur->ne[0], n_seq_tokens, n_seqs);

        // {n_embd, 2*d_inner} @ {n_embd, n_seq_tokens, n_seqs} => {2*d_inner, n_seq_tokens, n_seqs}
        ggml_tensor * xz = build_lora_mm(layer.ssm_in, cur);
        // split the above in two
        // => {d_inner, n_seq_tokens, n_seqs}
        ggml_tensor * x = ggml_view_3d(ctx0, xz, d_inner, xz->ne[1], xz->ne[2], xz->nb[1], xz->nb[2], 0);
        ggml_tensor * z = ggml_view_3d(ctx0, xz, d_inner, xz->ne[1], xz->ne[2], xz->nb[1], xz->nb[2], d_inner*ggml_element_size(xz));

        // conv
        {
            // => {d_conv - 1 + n_seq_tokens, d_inner, n_seqs}
            ggml_tensor * conv_x = ggml_concat(ctx0, conv, ggml_transpose(ctx0, x), 0);

            // copy last (d_conv - 1) columns back into the state cache
            ggml_tensor * last_conv = ggml_view_3d(ctx0, conv_x, d_conv - 1, d_inner, n_seqs, conv_x->nb[1], conv_x->nb[2], n_seq_tokens*(conv_x->nb[0]));

            ggml_build_forward_expand(gf,
                ggml_cpy(ctx0, last_conv,
                    ggml_view_1d(ctx0, conv_states_all,
                        (d_conv - 1)*(d_inner)*(n_seqs),
                        kv_head*(d_conv - 1)*(d_inner)*ggml_element_size(conv_states_all))));

            // 1D convolution
            // The equivalent is to make a self-overlapping view of conv_x
            // over d_conv columns at each stride in the 3rd dimension,
            // then element-wise multiply that with the conv1d weight,
            // then sum the elements of each row,
            // (the last two steps are a dot product over rows (also doable with mul_mat))
            // then permute away the ne[0] dimension,
            // and then you're left with the resulting x tensor.
            // For simultaneous sequences, all sequences need to have the same length.
            x = ggml_ssm_conv(ctx0, conv_x, layer.ssm_conv1d);

            // bias
            x = ggml_add(ctx0, x, layer.ssm_conv1d_b);

            x = ggml_silu(ctx0, x);
        }

        // ssm
        {
            // {d_inner, dt_rank + 2*d_state} @ {d_inner, n_seq_tokens, n_seqs} => {dt_rank + 2*d_state, n_seq_tokens, n_seqs}
            ggml_tensor * x_db = build_lora_mm(layer.ssm_x, x);
            // split
            ggml_tensor * dt = ggml_view_3d(ctx0, x_db, dt_rank, n_seq_tokens, n_seqs, x_db->nb[1], x_db->nb[2], 0);
            ggml_tensor * B  = ggml_view_4d(ctx0, x_db, d_state, /* n_group */ 1, n_seq_tokens, n_seqs, d_state*x_db->nb[0], x_db->nb[1], x_db->nb[2], ggml_element_size(x_db)*dt_rank);
            ggml_tensor * C  = ggml_view_4d(ctx0, x_db, d_state, /* n_group */ 1, n_seq_tokens, n_seqs, d_state*x_db->nb[0], x_db->nb[1], x_db->nb[2], ggml_element_size(x_db)*(dt_rank+d_state));

            // Some Mamba variants (e.g. FalconMamba, Jamba) apply RMS norm in B, C & Dt layers
            if (ssm_dt_b_c_rms || (layer.ssm_dt_norm && layer.ssm_b_norm && layer.ssm_c_norm)) {
                dt = build_norm(dt, layer.ssm_dt_norm, NULL, LLM_NORM_RMS, il);
                B  = build_norm(B,  layer.ssm_b_norm,  NULL, LLM_NORM_RMS, il);
                C  = build_norm(C,  layer.ssm_c_norm,  NULL, LLM_NORM_RMS, il);
            }

            // {dt_rank, d_inner} @ {dt_rank, n_seq_tokens, n_seqs} => {d_inner, n_seq_tokens, n_seqs}
            dt = build_lora_mm(layer.ssm_dt, dt);
            dt = ggml_add(ctx0, dt, layer.ssm_dt_b);

            cur = x;
            x = ggml_reshape_4d(ctx0, x, head_dim, n_head, n_seq_tokens, n_seqs);

            ggml_tensor * A = layer.ssm_a;

            // use the states and the indices provided by build_recurrent_state
            // (this is necessary in order to properly use the states before they are overwritten,
            //  while avoiding to make unnecessary copies of the states)
            auto get_ssm_rows = [&](ggml_context * ctx, ggml_tensor * states, ggml_tensor * ids) {
                ggml_tensor * ssm = ggml_reshape_4d(ctx, states, d_state, head_dim, n_head, mctx_cur->get_size());

                // Custom operator to optimize the parallel associative scan
                // as described in the Annex D of the Mamba paper.
                // => {d_inner, n_seq_tokens, n_seqs} and {d_state, d_inner, n_seqs}
                return ggml_ssm_scan(ctx, ssm, x, dt, A, B, C, ids);
            };

            ggml_tensor * y_ssm = build_rs(inp, ssm_states_all, hparams.n_embd_s(), ubatch.n_seqs, get_ssm_rows);

            // store last states
            ggml_build_forward_expand(gf,
                ggml_cpy(ctx0,
                    ggml_view_1d(ctx0, y_ssm, d_state*d_inner*n_seqs, x->nb[3]*x->ne[3]),
                    ggml_view_1d(ctx0, ssm_states_all, d_state*d_inner*n_seqs, kv_head*d_state*d_inner*ggml_element_size(ssm_states_all))));

            ggml_tensor * y = ggml_view_3d(ctx0, y_ssm, d_inner, n_seq_tokens, n_seqs, x->nb[2], x->nb[3], 0);

            // TODO: skip computing output earlier for unused tokens

            y = ggml_add(ctx0, y, ggml_mul(ctx0, cur, layer.ssm_d));
            y = ggml_swiglu_split(ctx0, ggml_cont(ctx0, z), y);

            // {d_inner, n_embd} @ {d_inner, n_seq_tokens, n_seqs} => {n_embd, n_seq_tokens, n_seqs}
            cur = build_lora_mm(layer.ssm_out, y);
        }

        // {n_embd, n_seq_tokens, n_seqs} => {n_embd, n_tokens}
        cur = ggml_reshape_2d(ctx0, cur, cur->ne[0], n_seq_tokens * n_seqs);

        return cur;
    }

    ggml_tensor * build_mamba2_layer(
        llm_graph_input_rs * inp,
               ggml_tensor * cur,
         const llama_model & model,
        const llama_ubatch & ubatch,
                       int   il) const {

        const auto * mctx_cur = inp->mctx;

        const auto kv_head = mctx_cur->get_head();

        const int64_t d_conv  = hparams.ssm_d_conv;
        const int64_t d_inner = hparams.ssm_d_inner;
        const int64_t d_state = hparams.ssm_d_state;
        const int64_t n_head  = hparams.ssm_dt_rank;
        const int64_t head_dim = d_inner / n_head;
        const int64_t n_group = hparams.ssm_n_group;
        const int64_t n_seqs  = ubatch.n_seqs;

        const int64_t n_seq_tokens = ubatch.n_seq_tokens;

        GGML_ASSERT(n_seqs != 0);
        GGML_ASSERT(ubatch.equal_seqs());
        GGML_ASSERT(ubatch.n_tokens == n_seq_tokens * n_seqs);

        ggml_tensor * conv_states_all = mctx_cur->get_r_l(il);
        ggml_tensor * ssm_states_all  = mctx_cur->get_s_l(il);

        ggml_tensor * conv = build_rs(inp, conv_states_all, hparams.n_embd_r(), n_seqs);
        conv = ggml_reshape_3d(ctx0, conv, d_conv - 1, d_inner + 2*n_group*d_state, n_seqs);

        // {n_embd, n_tokens} => {n_embd, n_seq_tokens, n_seqs}
        cur = ggml_reshape_3d(ctx0, cur, cur->ne[0], n_seq_tokens, n_seqs);

        // d_in_proj = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads

        // {n_embd, d_in_proj} @ {n_embd, n_seq_tokens, n_seqs} => {d_in_proj, n_seq_tokens, n_seqs}
        ggml_tensor * zxBCdt = build_lora_mm(model.layers[il].ssm_in, cur);

        // split the above in three
        ggml_tensor * z = ggml_view_4d(ctx0, zxBCdt, head_dim, n_head, n_seq_tokens, n_seqs, head_dim*zxBCdt->nb[0], zxBCdt->nb[1], zxBCdt->nb[2], 0);
        ggml_tensor * xBC = ggml_view_3d(ctx0, zxBCdt, d_inner + 2*n_group*d_state, n_seq_tokens, n_seqs, zxBCdt->nb[1], zxBCdt->nb[2], d_inner*ggml_element_size(zxBCdt));
        ggml_tensor * dt = ggml_view_3d(ctx0, zxBCdt, n_head, n_seq_tokens, n_seqs, zxBCdt->nb[1], zxBCdt->nb[2], (2*d_inner + 2*n_group*d_state)*ggml_element_size(zxBCdt));

        // conv
        {
            // => {d_conv - 1 + n_seq_tokens, d_inner + 2*n_group*d_state, n_seqs}
            ggml_tensor * conv_x = ggml_concat(ctx0, conv, ggml_transpose(ctx0, xBC), 0);

            // copy last (d_conv - 1) columns back into the state cache
            ggml_tensor * last_conv = ggml_view_3d(ctx0, conv_x, d_conv - 1, d_inner + 2*n_group*d_state, n_seqs, conv_x->nb[1], conv_x->nb[2], n_seq_tokens*(conv_x->nb[0]));

            ggml_build_forward_expand(gf,
                ggml_cpy(ctx0, last_conv,
                    ggml_view_1d(ctx0, conv_states_all,
                        (d_conv - 1)*(d_inner + 2*n_group*d_state)*(n_seqs),
                        kv_head*(d_conv - 1)*(d_inner + 2*n_group*d_state)*ggml_element_size(conv_states_all))));

            // 1D convolution
            // The equivalent is to make a self-overlapping view of conv_x
            // over d_conv columns at each stride in the 3rd dimension,
            // then element-wise multiply that with the conv1d weight,
            // then sum the elements of each row,
            // (the last two steps are a dot product over rows (also doable with mul_mat))
            // then permute away the ne[0] dimension,
            // and then you're left with the resulting x tensor.
            // For simultaneous sequences, all sequences need to have the same length.
            xBC = ggml_ssm_conv(ctx0, conv_x, model.layers[il].ssm_conv1d);

            // bias
            xBC = ggml_add(ctx0, xBC, model.layers[il].ssm_conv1d_b);

            xBC = ggml_silu(ctx0, xBC);
        }

        // ssm
        {
            // These correspond to V K Q in SSM/attention duality
            ggml_tensor * x = ggml_view_4d(ctx0, xBC, head_dim, n_head, n_seq_tokens, n_seqs, head_dim*xBC->nb[0], xBC->nb[1], xBC->nb[2], 0);
            ggml_tensor * B = ggml_view_4d(ctx0, xBC, d_state, n_group, n_seq_tokens, n_seqs, d_state*xBC->nb[0], xBC->nb[1], xBC->nb[2], d_inner*ggml_element_size(xBC));
            ggml_tensor * C = ggml_view_4d(ctx0, xBC, d_state, n_group, n_seq_tokens, n_seqs, d_state*xBC->nb[0], xBC->nb[1], xBC->nb[2], (d_inner + n_group*d_state)*ggml_element_size(xBC));

            // {n_head, n_seq_tokens, n_seqs}
            dt = ggml_add(ctx0, ggml_cont(ctx0, dt), model.layers[il].ssm_dt_b);

            ggml_tensor * A = model.layers[il].ssm_a;

            // use the states and the indices provided by build_recurrent_state
            // (this is necessary in order to properly use the states before they are overwritten,
            //  while avoiding to make unnecessary copies of the states)
            auto get_ssm_rows = [&](ggml_context * ctx, ggml_tensor * states, ggml_tensor * ids) {
                ggml_tensor * ssm = ggml_reshape_4d(ctx, states, d_state, head_dim, n_head, mctx_cur->get_size());

                // TODO: use semistructured matrices to implement state-space duality
                // => {d_inner, n_seq_tokens, n_seqs} and {d_state, d_inner, n_seqs}
                return ggml_ssm_scan(ctx, ssm, x, dt, A, B, C, ids);
            };

            ggml_tensor * y_ssm = build_rs(inp, ssm_states_all, hparams.n_embd_s(), ubatch.n_seqs, get_ssm_rows);

            // store last states
            ggml_build_forward_expand(gf,
                ggml_cpy(ctx0,
                    ggml_view_1d(ctx0, y_ssm, d_state*d_inner*n_seqs, ggml_nelements(x)*x->nb[0]),
                    ggml_view_1d(ctx0, ssm_states_all, d_state*d_inner*n_seqs, kv_head*d_state*d_inner*ggml_element_size(ssm_states_all))));

            ggml_tensor * y = ggml_view_4d(ctx0, y_ssm, head_dim, n_head, n_seq_tokens, n_seqs, x->nb[1], n_head*x->nb[1], n_seq_tokens*n_head*x->nb[1], 0);

            // TODO: skip computing output earlier for unused tokens

            y = ggml_add(ctx0, y, ggml_mul(ctx0, x, model.layers[il].ssm_d));
            y = ggml_swiglu_split(ctx0, ggml_cont(ctx0, z), y);

            // grouped RMS norm
            if (model.layers[il].ssm_norm) {
                y = ggml_reshape_4d(ctx0, y, d_inner / n_group, n_group, n_seq_tokens, n_seqs);
                y = build_norm(y, model.layers[il].ssm_norm, NULL, LLM_NORM_RMS, il);
            }

            y = ggml_reshape_3d(ctx0, y, d_inner, n_seq_tokens, n_seqs);

            // {d_inner, n_embd} @ {d_inner, n_seq_tokens, n_seqs} => {n_embd, n_seq_tokens, n_seqs}
            cur = build_lora_mm(model.layers[il].ssm_out, y);
        }

        // {n_embd, n_seq_tokens, n_seqs} => {n_embd, n_tokens}
        cur = ggml_reshape_2d(ctx0, cur, cur->ne[0], n_seq_tokens * n_seqs);
        cb(cur, "mamba_out", il);

        return cur;
    }
};

struct llm_build_mamba : public llm_graph_context_mamba {
    llm_build_mamba(const llama_model & model, const llm_graph_params & params) : llm_graph_context_mamba(params) {
        ggml_tensor * cur;
        ggml_tensor * inpL;

        // {n_embd, n_tokens}
        inpL = build_inp_embd(model.tok_embd);

        auto * rs_inp = build_rs_inp();

        ggml_tensor * inp_out_ids = build_inp_out_ids();

        for (int il = 0; il < n_layer; ++il) {
            // norm
            cur = build_norm(inpL,
                    model.layers[il].attn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "attn_norm", il);

            if (model.arch == LLM_ARCH_MAMBA2) {
                cur = build_mamba2_layer(rs_inp, cur, model, ubatch, il);
            } else {
                cur = build_mamba_layer(rs_inp, cur, model, ubatch, il);
            }

            if (il == n_layer - 1 && inp_out_ids) {
                cur  = ggml_get_rows(ctx0,  cur, inp_out_ids);
                inpL = ggml_get_rows(ctx0, inpL, inp_out_ids);
            }

            // residual
            cur = ggml_add(ctx0, cur, inpL);

            cur = build_cvec(cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        // final rmsnorm
        cur = build_norm(inpL, model.output_norm, NULL, LLM_NORM_RMS, -1);

        cb(cur, "result_norm", -1);
        res->t_embd = cur;

        // lm_head
        cur = build_lora_mm(model.output, cur);

        cb(cur, "result_output", -1);
        res->t_logits = cur;

        ggml_build_forward_expand(gf, cur);
    }

};

struct llm_build_jamba : public llm_graph_context_mamba {
    llm_build_jamba(const llama_model & model, const llm_graph_params & params) : llm_graph_context_mamba(params) {
        const int64_t n_embd_head = hparams.n_embd_head_v;

        ggml_tensor * cur;
        ggml_tensor * inpL;

        // {n_embd, n_tokens}
        inpL = build_inp_embd(model.tok_embd);

        auto * inp_hybrid = build_inp_mem_hybrid();

        ggml_tensor * inp_out_ids = build_inp_out_ids();

        for (int il = 0; il < n_layer; ++il) {
            const int64_t n_head_kv = hparams.n_head_kv(il);

            cur = build_norm(inpL, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, il);
            cb(cur, "attn_norm", il);

            if (n_head_kv == 0) {
                cur = build_mamba_layer(inp_hybrid->get_recr(), cur, model, ubatch, il);
            } else {
                // Attention

                struct ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
                struct ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
                struct ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);

                cb(Qcur, "Qcur", il);
                cb(Kcur, "Kcur", il);
                cb(Vcur, "Vcur", il);

                Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
                Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
                Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

                cb(Qcur, "Qcur", il);
                cb(Kcur, "Kcur", il);
                cb(Vcur, "Vcur", il);

                // No RoPE :)
                cur = build_attn(inp_hybrid->get_attn(), model.layers[il].wo, NULL, Qcur, Kcur, Vcur, NULL, NULL, 1.0f/sqrtf(float(n_embd_head)), il);
            }

            if (il == n_layer - 1 && inp_out_ids) {
                cur  = ggml_get_rows(ctx0,  cur, inp_out_ids);
                inpL = ggml_get_rows(ctx0, inpL, inp_out_ids);
            }

            // residual
            struct ggml_tensor * ffn_inp = ggml_add(ctx0, inpL, cur);
            cb(cur, "ffn_inp", il);

            cur = build_norm(ffn_inp, model.layers[il].ffn_norm, NULL, LLM_NORM_RMS, il);
            cb(cur, "ffn_norm", il);

            // feed-forward network
            if (model.layers[il].ffn_gate_inp == nullptr) {
                // FFN
                cur = build_ffn(cur,
                        model.layers[il].ffn_up,   NULL, NULL,
                        model.layers[il].ffn_gate, NULL, NULL,
                        model.layers[il].ffn_down, NULL, NULL,
                        NULL,
                        LLM_FFN_SILU, LLM_FFN_PAR, il);
                cb(cur, "ffn_out", il);
            } else {
                // MoE branch
                cur = build_moe_ffn(cur,
                        model.layers[il].ffn_gate_inp,
                        model.layers[il].ffn_up_exps,
                        model.layers[il].ffn_gate_exps,
                        model.layers[il].ffn_down_exps,
                        nullptr,
                        n_expert, n_expert_used,
                        LLM_FFN_SILU, false,
                        false, 0.0,
                        LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX,
                        il);
                cb(cur, "ffn_moe_out", il);
            }

            // residual
            cur = ggml_add(ctx0, ffn_inp, cur);

            cur = build_cvec(cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        // final rmsnorm
        cur = build_norm(inpL, model.output_norm, NULL, LLM_NORM_RMS, -1);

        cb(cur, "result_norm", -1);
        res->t_embd = cur;

        // lm_head
        cur = build_lora_mm(model.output, cur);

        cb(cur, "result_output", -1);
        res->t_logits = cur;

        ggml_build_forward_expand(gf, cur);
    }
};

struct llm_build_command_r : public llm_graph_context {
    llm_build_command_r(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
        const int64_t n_embd_head = hparams.n_embd_head_v;

        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);

        const float f_logit_scale = hparams.f_logit_scale;

        ggml_tensor * cur;
        ggml_tensor * inpL;

        inpL = build_inp_embd(model.tok_embd);

        // inp_pos - contains the positions
        ggml_tensor * inp_pos = build_inp_pos();

        auto * inp_attn = build_attn_inp_kv_unified();

        ggml_tensor * inp_out_ids = build_inp_out_ids();

        for (int il = 0; il < n_layer; ++il) {
            // norm
            cur = build_norm(inpL,
                    model.layers[il].attn_norm, NULL,
                    LLM_NORM, il);
            cb(cur, "attn_norm", il);

            ggml_tensor * ffn_inp = cur;

            // self-attention
            {
                // compute Q and K and RoPE them
                ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
                cb(Qcur, "Qcur", il);
                if (model.layers[il].bq) {
                    Qcur = ggml_add(ctx0, Qcur, model.layers[il].bq);
                    cb(Qcur, "Qcur", il);
                }

                ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
                cb(Kcur, "Kcur", il);
                if (model.layers[il].bk) {
                    Kcur = ggml_add(ctx0, Kcur, model.layers[il].bk);
                    cb(Kcur, "Kcur", il);
                }

                ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
                cb(Vcur, "Vcur", il);
                if (model.layers[il].bv) {
                    Vcur = ggml_add(ctx0, Vcur, model.layers[il].bv);
                    cb(Vcur, "Vcur", il);
                }

                Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
                Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
                Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

                if (model.layers[il].attn_q_norm) {
                    Qcur = build_norm(Qcur,
                            model.layers[il].attn_q_norm,
                            NULL,
                            LLM_NORM, il);
                    cb(Qcur, "Qcur", il);
                }

                Qcur = ggml_rope_ext(
                        ctx0, Qcur, inp_pos, nullptr,
                        n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow
                        );

                if (model.layers[il].attn_k_norm) {
                    Kcur = build_norm(Kcur,
                            model.layers[il].attn_k_norm,
                            NULL,
                            LLM_NORM, il);
                    cb(Kcur, "Kcur", il);
                }

                Kcur = ggml_rope_ext(
                        ctx0, Kcur, inp_pos, nullptr,
                        n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow
                        );

                cb(Qcur, "Qcur", il);
                cb(Kcur, "Kcur", il);
                cb(Vcur, "Vcur", il);

                cur = build_attn(inp_attn,
                        model.layers[il].wo, model.layers[il].bo,
                        Qcur, Kcur, Vcur, nullptr, nullptr, 1.0f/sqrtf(float(n_embd_head)), il);
            }

            if (il == n_layer - 1 && inp_out_ids) {
                cur     = ggml_get_rows(ctx0,     cur, inp_out_ids);
                inpL    = ggml_get_rows(ctx0,    inpL, inp_out_ids);
                ffn_inp = ggml_get_rows(ctx0, ffn_inp, inp_out_ids);
            }

            ggml_tensor * attn_out = cur;

            // feed-forward network
            {
                cur = build_ffn(ffn_inp,
                        model.layers[il].ffn_up,   NULL, NULL,
                        model.layers[il].ffn_gate, NULL, NULL,
                        model.layers[il].ffn_down, NULL, NULL,
                        NULL,
                        LLM_FFN_SILU, LLM_FFN_PAR, il);
                cb(cur, "ffn_out", il);
            }

            // add together residual + FFN + self-attention
            cur = ggml_add(ctx0, cur, inpL);
            cur = ggml_add(ctx0, cur, attn_out);

            cur = build_cvec(cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        cur = build_norm(cur,
                model.output_norm, NULL,
                LLM_NORM, -1);

        cb(cur, "result_norm", -1);
        res->t_embd = cur;

        // lm_head
        cur = build_lora_mm(model.output, cur);

        if (f_logit_scale) {
            cur = ggml_scale(ctx0, cur, f_logit_scale);
        }

        cb(cur, "result_output", -1);
        res->t_logits = cur;

        ggml_build_forward_expand(gf, cur);
    }
};

struct llm_build_cohere2_iswa : public llm_graph_context {
    llm_build_cohere2_iswa(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
        const int64_t n_embd_head = hparams.n_embd_head_v;

        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);

        const float f_logit_scale = hparams.f_logit_scale;

        ggml_tensor * cur;
        ggml_tensor * inpL;

        inpL = build_inp_embd(model.tok_embd);

        // inp_pos - contains the positions
        ggml_tensor * inp_pos = build_inp_pos();

        auto * inp_attn = build_attn_inp_kv_unified_iswa();

        ggml_tensor * inp_out_ids = build_inp_out_ids();

        for (int il = 0; il < n_layer; ++il) {
            const bool is_swa = hparams.is_swa(il);

            // norm
            cur = build_norm(inpL, model.layers[il].attn_norm, NULL, LLM_NORM, il);
            cb(cur, "attn_norm", il);
            ggml_tensor * ffn_inp = cur;

            // self-attention
            {
                // rope freq factors for 128k context
                ggml_tensor * rope_factors = model.get_rope_factors(cparams, il);

                // compute Q and K and RoPE them
                ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
                cb(Qcur, "Qcur", il);
                if (model.layers[il].bq) {
                    Qcur = ggml_add(ctx0, Qcur, model.layers[il].bq);
                    cb(Qcur, "Qcur", il);
                }

                ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
                cb(Kcur, "Kcur", il);
                if (model.layers[il].bk) {
                    Kcur = ggml_add(ctx0, Kcur, model.layers[il].bk);
                    cb(Kcur, "Kcur", il);
                }

                ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
                cb(Vcur, "Vcur", il);
                if (model.layers[il].bv) {
                    Vcur = ggml_add(ctx0, Vcur, model.layers[il].bv);
                    cb(Vcur, "Vcur", il);
                }

                Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
                Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
                Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

                if (is_swa) {
                    Qcur = ggml_rope_ext(
                            ctx0, Qcur, inp_pos, rope_factors,
                            n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                            ext_factor, attn_factor, beta_fast, beta_slow
                            );

                    Kcur = ggml_rope_ext(
                            ctx0, Kcur, inp_pos, rope_factors,
                            n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                            ext_factor, attn_factor, beta_fast, beta_slow
                            );
                }

                cb(Qcur, "Qcur", il);
                cb(Kcur, "Kcur", il);
                cb(Vcur, "Vcur", il);

                cur = build_attn(inp_attn,
                        model.layers[il].wo, model.layers[il].bo,
                        Qcur, Kcur, Vcur, nullptr, nullptr, 1.0f/sqrtf(float(n_embd_head)), il);
            }

            if (il == n_layer - 1 && inp_out_ids) {
                cur     = ggml_get_rows(ctx0, cur, inp_out_ids);
                inpL    = ggml_get_rows(ctx0, inpL, inp_out_ids);
                ffn_inp = ggml_get_rows(ctx0, ffn_inp, inp_out_ids);
            }

            ggml_tensor * attn_out = cur;

            // feed-forward network
            {
                cur = build_ffn(ffn_inp, model.layers[il].ffn_up, NULL, NULL, model.layers[il].ffn_gate,
                        NULL, NULL, model.layers[il].ffn_down, NULL, NULL, NULL, LLM_FFN_SILU, LLM_FFN_PAR,
                        il);
                cb(cur, "ffn_out", il);
            }

            // add together residual + FFN + self-attention
            cur = ggml_add(ctx0, cur, inpL);
            cur = ggml_add(ctx0, cur, attn_out);

            cur = build_cvec(cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        cur = build_norm(cur, model.output_norm, NULL, LLM_NORM, -1);

        cb(cur, "result_norm", -1);
        res->t_embd = cur;

        // lm_head
        cur = build_lora_mm(model.output, cur);

        if (f_logit_scale) {
            cur = ggml_scale(ctx0, cur, f_logit_scale);
        }

        cb(cur, "result_output", -1);
        res->t_logits = cur;

        ggml_build_forward_expand(gf, cur);
    }
};

// ref: https://allenai.org/olmo
// based on the original build_llama() function, changes:
//   * non-parametric layer norm
//   * clamp qkv
//   * removed bias
//   * removed MoE
struct llm_build_olmo : public llm_graph_context {
    llm_build_olmo(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
        const int64_t n_embd_head = hparams.n_embd_head_v;

        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
        GGML_ASSERT(n_embd_head == hparams.n_rot);

        ggml_tensor * cur;
        ggml_tensor * inpL;

        inpL = build_inp_embd(model.tok_embd);

        // inp_pos - contains the positions
        ggml_tensor * inp_pos = build_inp_pos();

        auto * inp_attn = build_attn_inp_kv_unified();

        ggml_tensor * inp_out_ids = build_inp_out_ids();

        for (int il = 0; il < n_layer; ++il) {
            ggml_tensor * inpSA = inpL;

            // norm
            cur = build_norm(inpL,
                    NULL, NULL,
                    LLM_NORM, il);
            cb(cur, "attn_norm", il);

            // self-attention
            {
                // compute Q and K and RoPE them
                ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
                cb(Qcur, "Qcur", il);
                if (hparams.f_clamp_kqv > 0.0f) {
                    Qcur = ggml_clamp(ctx0, Qcur, -hparams.f_clamp_kqv, hparams.f_clamp_kqv);
                    cb(Qcur, "Qcur", il);
                }

                ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
                cb(Kcur, "Kcur", il);
                if (hparams.f_clamp_kqv > 0.0f) {
                    Kcur = ggml_clamp(ctx0, Kcur, -hparams.f_clamp_kqv, hparams.f_clamp_kqv);
                    cb(Kcur, "Kcur", il);
                }

                ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
                cb(Vcur, "Vcur", il);
                if (hparams.f_clamp_kqv > 0.0f) {
                    Vcur = ggml_clamp(ctx0, Vcur, -hparams.f_clamp_kqv, hparams.f_clamp_kqv);
                    cb(Vcur, "Vcur", il);
                }

                Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
                Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
                Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

                Qcur = ggml_rope_ext(
                        ctx0, Qcur, inp_pos, nullptr,
                        n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow
                        );

                Kcur = ggml_rope_ext(
                        ctx0, Kcur, inp_pos, nullptr,
                        n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow
                        );

                cb(Qcur, "Qcur", il);
                cb(Kcur, "Kcur", il);
                cb(Vcur, "Vcur", il);

                cur = build_attn(inp_attn,
                        model.layers[il].wo, nullptr,
                        Qcur, Kcur, Vcur, nullptr, nullptr, 1.0f/sqrtf(float(n_embd_head)), il);
            }

            if (il == n_layer - 1 && inp_out_ids) {
                cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
                inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
            }

            ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
            cb(ffn_inp, "ffn_inp", il);

            // feed-forward network
            cur = build_norm(ffn_inp,
                    NULL, NULL,
                    LLM_NORM, il);
            cb(cur, "ffn_norm", il);

            cur = build_ffn(cur,
                    model.layers[il].ffn_up,   NULL, NULL,
                    model.layers[il].ffn_gate, NULL, NULL,
                    model.layers[il].ffn_down, NULL, NULL,
                    NULL,
                    LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(cur, "ffn_out", il);

            cur = ggml_add(ctx0, cur, ffn_inp);
            cb(cur, "ffn_out", il);

            cur = build_cvec(cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        cur = build_norm(cur,
                NULL, NULL,
                LLM_NORM, -1);

        cb(cur, "result_norm", -1);
        res->t_embd = cur;

        // lm_head
        cur = build_lora_mm(model.output, cur);

        cb(cur, "result_output", -1);
        res->t_logits = cur;

        ggml_build_forward_expand(gf, cur);
    }
};

struct llm_build_olmo2 : public llm_graph_context {
    llm_build_olmo2(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
        const int64_t n_embd_head = hparams.n_embd_head_v;

        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
        GGML_ASSERT(n_embd_head == hparams.n_rot);

        ggml_tensor * cur;
        ggml_tensor * inpL;

        inpL = build_inp_embd(model.tok_embd);

        // inp_pos - contains the positions
        ggml_tensor * inp_pos = build_inp_pos();

        auto * inp_attn = build_attn_inp_kv_unified();

        ggml_tensor * inp_out_ids = build_inp_out_ids();

        for (int il = 0; il < n_layer; ++il) {
            ggml_tensor * inpSA = inpL;

            cur = inpL;

            // self_attention
            {
                // compute Q and K and RoPE them
                ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
                cb(Qcur, "Qcur", il);

                ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
                cb(Kcur, "Kcur", il);

                ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
                cb(Vcur, "Vcur", il);

                Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, NULL,
                        LLM_NORM_RMS, il);
                cb(Qcur, "Qcur_normed", il);

                Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, NULL,
                        LLM_NORM_RMS, il);
                cb(Kcur, "Kcur_normed", il);

                Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
                Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
                Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

                Qcur = ggml_rope_ext(
                        ctx0, Qcur, inp_pos, nullptr,
                        n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow
                        );

                Kcur = ggml_rope_ext(
                        ctx0, Kcur, inp_pos, nullptr,
                        n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow
                        );

                cb(Qcur, "Qcur", il);
                cb(Kcur, "Kcur", il);
                cb(Vcur, "Vcur", il);

                cur = build_attn(inp_attn,
                        model.layers[il].wo, NULL,
                        Qcur, Kcur, Vcur, nullptr, nullptr, 1.0f/sqrtf(float(n_embd_head)), il);
            }

            if (il == n_layer - 1 && inp_out_ids) {
                cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
                inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
            }

            cur = build_norm(cur,
                    model.layers[il].attn_post_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "attn_post_norm", il);

            ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
            cb(ffn_inp, "ffn_inp", il);

            // feed-forward network
            cur = build_ffn(ffn_inp,
                    model.layers[il].ffn_up,   NULL, NULL,
                    model.layers[il].ffn_gate, NULL, NULL,
                    model.layers[il].ffn_down, NULL, NULL,
                    NULL,
                    LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(cur, "ffn_out", il);

            cur = build_norm(cur,
                    model.layers[il].ffn_post_norm, NULL,
                    LLM_NORM_RMS, -1);
            cb(cur, "ffn_post_norm", -1);

            cur = ggml_add(ctx0, cur, ffn_inp);
            cb(cur, "ffn_out", il);

            cur = build_cvec(cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        cur = build_norm(cur,
                model.output_norm, NULL,
                LLM_NORM_RMS, -1);

        cb(cur, "result_norm", -1);
        res->t_embd = cur;

        // lm_head
        cur = build_lora_mm(model.output, cur);

        cb(cur, "result_output", -1);
        res->t_logits = cur;

        ggml_build_forward_expand(gf, cur);
    }
};

// based on the build_qwen2moe() function, changes:
//   * removed shared experts
//   * removed bias
//   * added q, k norm
struct llm_build_olmoe : public llm_graph_context {
    llm_build_olmoe(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
        const int64_t n_embd_head = hparams.n_embd_head_v;

        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
        GGML_ASSERT(n_embd_head == hparams.n_rot);

        ggml_tensor * cur;
        ggml_tensor * inpL;

        inpL = build_inp_embd(model.tok_embd);

        // inp_pos - contains the positions
        ggml_tensor * inp_pos = build_inp_pos();

        auto * inp_attn = build_attn_inp_kv_unified();

        ggml_tensor * inp_out_ids = build_inp_out_ids();

        for (int il = 0; il < n_layer; ++il) {
            ggml_tensor * inpSA = inpL;

            // norm
            cur = build_norm(inpL,
                    model.layers[il].attn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "attn_norm", il);

            // self_attention
            {
                // compute Q and K and RoPE them
                ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
                cb(Qcur, "Qcur", il);

                ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
                cb(Kcur, "Kcur", il);

                ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
                cb(Vcur, "Vcur", il);

                Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, NULL,
                        LLM_NORM_RMS, il);
                cb(Qcur, "Qcur_normed", il);

                Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, NULL,
                        LLM_NORM_RMS, il);
                cb(Kcur, "Kcur_normed", il);

                Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
                Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
                Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

                Qcur = ggml_rope_ext(
                        ctx0, Qcur, inp_pos, nullptr,
                        n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow
                        );

                Kcur = ggml_rope_ext(
                        ctx0, Kcur, inp_pos, nullptr,
                        n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow
                        );

                cb(Qcur, "Qcur", il);
                cb(Kcur, "Kcur", il);
                cb(Vcur, "Vcur", il);

                cur = build_attn(inp_attn,
                        model.layers[il].wo, NULL,
                        Qcur, Kcur, Vcur, nullptr, nullptr, 1.0f/sqrtf(float(n_embd_head)), il);
            }

            if (il == n_layer - 1 && inp_out_ids) {
                cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
                inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
            }

            ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
            cb(ffn_inp, "ffn_inp", il);

            // MoE branch
            cur = build_norm(ffn_inp,
                    model.layers[il].ffn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "ffn_norm", il);

            cur = build_moe_ffn(cur,
                    model.layers[il].ffn_gate_inp,
                    model.layers[il].ffn_up_exps,
                    model.layers[il].ffn_gate_exps,
                    model.layers[il].ffn_down_exps,
                    nullptr,
                    n_expert, n_expert_used,
                    LLM_FFN_SILU, false,
                    false, 0.0,
                    LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX,
                    il);
            cb(cur, "ffn_moe_out", il);

            cur = ggml_add(ctx0, cur, ffn_inp);

            cur = build_cvec(cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        cur = build_norm(cur,
                model.output_norm, NULL,
                LLM_NORM_RMS, -1);

        cb(cur, "result_norm", -1);
        res->t_embd = cur;

        // lm_head
        cur = build_lora_mm(model.output, cur);

        cb(cur, "result_output", -1);
        res->t_logits = cur;

        ggml_build_forward_expand(gf, cur);
    }
};

struct llm_build_openelm : public llm_graph_context {
    llm_build_openelm(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
        const int64_t n_embd_head = hparams.n_embd_head_v;

        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);

        ggml_tensor * cur;
        ggml_tensor * inpL;
        inpL = build_inp_embd(model.tok_embd);

        // inp_pos - contains the positions
        ggml_tensor * inp_pos = build_inp_pos();

        auto * inp_attn = build_attn_inp_kv_unified();

        ggml_tensor * inp_out_ids = build_inp_out_ids();

        for (int il = 0; il < n_layer; ++il) {
            const int64_t n_head    = hparams.n_head(il);
            const int64_t n_head_kv = hparams.n_head_kv(il);
            const int64_t n_head_qkv = 2*n_head_kv + n_head;

            cur = inpL;
            ggml_tensor * residual = cur;

            // norm
            cur = build_norm(inpL,
                    model.layers[il].attn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "attn_norm", il);

            // self-attention
            {
                cur = build_lora_mm(model.layers[il].wqkv, cur);
                cb(cur, "wqkv", il);

                cur = ggml_reshape_3d(ctx0, cur, n_embd_head_k, n_head_qkv, n_tokens);

                ggml_tensor * Qcur = ggml_view_3d(ctx0, cur, n_embd_head, n_head,    n_tokens, cur->nb[1], cur->nb[2], 0);
                cb(Qcur, "Qcur", il);

                ggml_tensor * Kcur = ggml_view_3d(ctx0, cur, n_embd_head, n_head_kv, n_tokens, cur->nb[1], cur->nb[2], cur->nb[1]*n_head);
                cb(Kcur, "Kcur", il);

                ggml_tensor * Vcur = ggml_cont(ctx0, ggml_view_3d(ctx0, cur, n_embd_head, n_head_kv, n_tokens, cur->nb[1], cur->nb[2], cur->nb[1]*(n_head+n_head_kv)));
                cb(Vcur, "Vcur", il);

                Qcur = build_norm(Qcur,
                        model.layers[il].attn_q_norm, NULL,
                        LLM_NORM_RMS, il);
                cb(Qcur, "Qcur", il);

                Kcur = build_norm(Kcur,
                        model.layers[il].attn_k_norm, NULL,
                        LLM_NORM_RMS, il);
                cb(Kcur, "Kcur", il);

                Qcur = ggml_rope_ext(
                        ctx0, Qcur, inp_pos, NULL,
                        n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow
                        );

                Kcur = ggml_rope_ext(
                        ctx0, Kcur, inp_pos, NULL,
                        n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow
                        );

                cb(Qcur, "Qcur", il);
                cb(Kcur, "Kcur", il);
                cb(Qcur, "Vcur", il);

                cur = build_attn(inp_attn,
                        model.layers[il].wo, NULL,
                        Qcur, Kcur, Vcur, nullptr, nullptr, 1.0f/sqrtf(float(n_embd_head)), il);
            }

            if (il == n_layer - 1 && inp_out_ids) {
                residual = ggml_get_rows(ctx0, residual, inp_out_ids);
                cur      = ggml_get_rows(ctx0, cur,      inp_out_ids);
            }

            ggml_tensor * ffn_inp = ggml_add(ctx0, residual, cur);
            cb(ffn_inp, "ffn_inp", il);

            // feed-forward network
            {
                cur = build_norm(ffn_inp,
                        model.layers[il].ffn_norm, NULL,
                        LLM_NORM_RMS, il);
                cb(cur, "ffn_norm", il);

                cur = build_ffn(cur,
                        model.layers[il].ffn_up,   NULL, NULL,
                        model.layers[il].ffn_gate, NULL, NULL,
                        model.layers[il].ffn_down, NULL, NULL,
                        NULL,
                        LLM_FFN_SILU, LLM_FFN_PAR, il);
                cb(cur, "ffn_out", il);
            }

            cur = ggml_add(ctx0, cur, ffn_inp);

            cur = build_cvec(cur, il);
            cb(cur, "l_out", il);

            inpL = cur;
        }

        cur = inpL;

        // norm
        cur = build_norm(cur,
                model.output_norm, NULL,
                LLM_NORM_RMS, -1);

        cb(cur, "result_norm", -1);
        res->t_embd = cur;

        cur = build_lora_mm(model.output, cur);

        cb(cur, "result_output", -1);
        res->t_logits = cur;

        ggml_build_forward_expand(gf, cur);
    }
};

struct llm_build_gptneox : public llm_graph_context {
    llm_build_gptneox(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
        const int64_t n_embd_head = hparams.n_embd_head_v;
        const int64_t n_embd_gqa  = hparams.n_embd_v_gqa();

        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);

        ggml_tensor * cur;
        ggml_tensor * inpL;

        inpL = build_inp_embd(model.tok_embd);

        // inp_pos - contains the positions
        ggml_tensor * inp_pos = build_inp_pos();

        auto * inp_attn = build_attn_inp_kv_unified();

        ggml_tensor * inp_out_ids = build_inp_out_ids();

        for (int il = 0; il < n_layer; ++il) {
            cur = build_norm(inpL,
                    model.layers[il].attn_norm,
                    model.layers[il].attn_norm_b,
                    LLM_NORM, il);
            cb(cur, "attn_norm", il);

            // self-attention
            {
                cur = build_lora_mm(model.layers[il].wqkv, cur);
                cb(cur, "wqkv", il);

                cur = ggml_add(ctx0, cur, model.layers[il].bqkv);
                cb(cur, "bqkv", il);

                ggml_tensor * Qcur = ggml_view_3d(ctx0, cur, n_embd_head, n_head,    n_tokens, n_embd_head*sizeof(float), cur->nb[1], 0*sizeof(float)*(n_embd));
                ggml_tensor * Kcur = ggml_view_3d(ctx0, cur, n_embd_head, n_head_kv, n_tokens, n_embd_head*sizeof(float), cur->nb[1], 1*sizeof(float)*(n_embd));
                ggml_tensor * Vcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd_gqa, n_tokens, cur->nb[1], 1*sizeof(float)*(n_embd + n_embd_gqa)));

                Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

                Qcur = ggml_rope_ext(
                        ctx0, Qcur, inp_pos, nullptr,
                        n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow
                        );

                Kcur = ggml_rope_ext(
                        ctx0, Kcur, inp_pos, nullptr,
                        n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow
                        );

                cb(Qcur, "Qcur", il);
                cb(Kcur, "Kcur", il);
                cb(Vcur, "Vcur", il);

                cur = build_attn(inp_attn,
                        model.layers[il].wo, model.layers[il].bo,
                        Qcur, Kcur, Vcur, nullptr, nullptr, 1.0f/sqrtf(float(n_embd_head)), il);
            }

            if (il == n_layer - 1 && inp_out_ids) {
                cur  = ggml_get_rows(ctx0,  cur, inp_out_ids);
                inpL = ggml_get_rows(ctx0, inpL, inp_out_ids);
            }

            // ffn
            if (hparams.use_par_res) {
                // attention and ffn are computed in parallel
                // x = x + attn(ln1(x)) + ffn(ln2(x))

                ggml_tensor * attn_out = cur;

                cur = build_norm(inpL,
                        model.layers[il].ffn_norm,
                        model.layers[il].ffn_norm_b,
                        LLM_NORM, il);
                cb(cur, "ffn_norm", il);

                cur = build_ffn(cur,
                        model.layers[il].ffn_up,   model.layers[il].ffn_up_b,   NULL,
                        NULL,                      NULL,                        NULL,
                        model.layers[il].ffn_down, model.layers[il].ffn_down_b, NULL,
                        NULL,
                        LLM_FFN_GELU, LLM_FFN_SEQ, il);
                cb(cur, "ffn_out", il);

                cur = ggml_add(ctx0, cur, inpL);
                cb(cur, "ffn_out", il);

                cur = ggml_add(ctx0, cur, attn_out);

                cur = build_cvec(cur, il);
                cb(cur, "l_out", il);

                // input for next layer
                inpL = cur;
            } else {
                // attention and ffn are computed sequentially
                // x = x + attn(ln1(x))
                // x = x + ffn(ln2(x))

                ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpL);
                cb(ffn_inp, "ffn_inp", il);

                cur = build_norm(ffn_inp,
                        model.layers[il].ffn_norm,
                        model.layers[il].ffn_norm_b,
                        LLM_NORM, il);
                cb(cur, "ffn_norm", il);

                cur = build_ffn(cur,
                        model.layers[il].ffn_up,   model.layers[il].ffn_up_b,   NULL,
                        NULL,                      NULL,                        NULL,
                        model.layers[il].ffn_down, model.layers[il].ffn_down_b, NULL,
                        NULL,
                        LLM_FFN_GELU, LLM_FFN_SEQ, il);
                cb(cur, "ffn_out", il);

                cur = ggml_add(ctx0, cur, ffn_inp);

                cur = build_cvec(cur, il);
                cb(cur, "l_out", il);

                // input for next layer
                inpL = cur;
            }
        }

        cur = build_norm(inpL,
                model.output_norm,
                model.output_norm_b,
                LLM_NORM, -1);

        cb(cur, "result_norm", -1);
        res->t_embd = cur;

        cur = build_lora_mm(model.output, cur);

        cb(cur, "result_output", -1);
        res->t_logits = cur;

        ggml_build_forward_expand(gf, cur);
    }
};

struct llm_build_arctic : public llm_graph_context {
    llm_build_arctic(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
        const int64_t n_embd_head = hparams.n_embd_head_v;

        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
        GGML_ASSERT(n_embd_head == hparams.n_rot);

        ggml_tensor * cur;
        ggml_tensor * inpL;

        inpL = build_inp_embd(model.tok_embd);

        // inp_pos - contains the positions
        ggml_tensor * inp_pos = build_inp_pos();

        auto * inp_attn = build_attn_inp_kv_unified();

        ggml_tensor * inp_out_ids = build_inp_out_ids();

        for (int il = 0; il < n_layer; ++il) {
            ggml_tensor * inpSA = inpL;

            // norm
            cur = build_norm(inpL,
                    model.layers[il].attn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "attn_norm", il);

            // self-attention
            {
                // compute Q and K and RoPE them
                ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
                cb(Qcur, "Qcur", il);

                ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
                cb(Kcur, "Kcur", il);

                ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
                cb(Vcur, "Vcur", il);

                Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
                Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
                Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

                Qcur = ggml_rope_ext(
                        ctx0, Qcur, inp_pos, nullptr,
                        n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow
                        );

                Kcur = ggml_rope_ext(
                        ctx0, Kcur, inp_pos, nullptr,
                        n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow
                        );

                cb(Qcur, "Qcur", il);
                cb(Kcur, "Kcur", il);
                cb(Vcur, "Vcur", il);

                cur = build_attn(inp_attn,
                        model.layers[il].wo, NULL,
                        Qcur, Kcur, Vcur, nullptr, nullptr, 1.0f/sqrtf(float(n_embd_head)), il);
            }

            if (il == n_layer - 1 && inp_out_ids) {
                cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
                inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
            }

            ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
            cb(ffn_inp, "ffn_inp", il);

            // feed-forward network
            cur = build_norm(ffn_inp,
                    model.layers[il].ffn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "ffn_norm", il);

            cur = build_ffn(cur,
                    model.layers[il].ffn_up,   NULL, NULL,
                    model.layers[il].ffn_gate, NULL, NULL,
                    model.layers[il].ffn_down, NULL, NULL,
                    NULL,
                    LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(cur, "ffn_out", il);

            ggml_tensor * ffn_out = ggml_add(ctx0, cur, ffn_inp);
            cb(ffn_out, "ffn_out", il);

            // MoE
            cur = build_norm(inpSA,
                    model.layers[il].ffn_norm_exps, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "ffn_norm_exps", il);

            cur = build_moe_ffn(cur,
                    model.layers[il].ffn_gate_inp,
                    model.layers[il].ffn_up_exps,
                    model.layers[il].ffn_gate_exps,
                    model.layers[il].ffn_down_exps,
                    nullptr,
                    n_expert, n_expert_used,
                    LLM_FFN_SILU, true,
                    false, 0.0,
                    LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX,
                    il);
            cb(cur, "ffn_moe_out", il);

            cur = ggml_add(ctx0, cur, ffn_out);
            cb(cur, "ffn_out", il);

            cur = build_cvec(cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        cur = build_norm(cur,
                model.output_norm, NULL,
                LLM_NORM_RMS, -1);

        cb(cur, "result_norm", -1);
        res->t_embd = cur;

        // lm_head
        cur = build_lora_mm(model.output, cur);

        cb(cur, "result_output", -1);
        res->t_logits = cur;

        ggml_build_forward_expand(gf, cur);
    }
};

struct llm_build_deepseek : public llm_graph_context {
    llm_build_deepseek(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
        const int64_t n_embd_head = hparams.n_embd_head_v;

        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
        GGML_ASSERT(n_embd_head == hparams.n_rot);

        ggml_tensor * cur;
        ggml_tensor * inpL;

        inpL = build_inp_embd(model.tok_embd);

        // inp_pos - contains the positions
        ggml_tensor * inp_pos = build_inp_pos();

        auto * inp_attn = build_attn_inp_kv_unified();

        const float kq_scale = hparams.f_attention_scale == 0.0f ? 1.0f/sqrtf(float(n_embd_head)) : hparams.f_attention_scale;

        ggml_tensor * inp_out_ids = build_inp_out_ids();

        for (int il = 0; il < n_layer; ++il) {
            ggml_tensor * inpSA = inpL;

            // norm
            cur = build_norm(inpL,
                    model.layers[il].attn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "attn_norm", il);

            // self-attention
            {
                // rope freq factors for llama3; may return nullptr for llama2 and other models
                ggml_tensor * rope_factors = model.get_rope_factors(cparams, il);

                // compute Q and K and RoPE them
                ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
                cb(Qcur, "Qcur", il);
                if (model.layers[il].bq) {
                    Qcur = ggml_add(ctx0, Qcur, model.layers[il].bq);
                    cb(Qcur, "Qcur", il);
                }

                ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
                cb(Kcur, "Kcur", il);
                if (model.layers[il].bk) {
                    Kcur = ggml_add(ctx0, Kcur, model.layers[il].bk);
                    cb(Kcur, "Kcur", il);
                }

                ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
                cb(Vcur, "Vcur", il);
                if (model.layers[il].bv) {
                    Vcur = ggml_add(ctx0, Vcur, model.layers[il].bv);
                    cb(Vcur, "Vcur", il);
                }

                Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
                Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
                Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

                Qcur = ggml_rope_ext(
                        ctx0, Qcur, inp_pos, rope_factors,
                        n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow
                        );

                Kcur = ggml_rope_ext(
                        ctx0, Kcur, inp_pos, rope_factors,
                        n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow
                        );

                cb(Qcur, "Qcur", il);
                cb(Kcur, "Kcur", il);
                cb(Vcur, "Vcur", il);

                cur = build_attn(inp_attn,
                        model.layers[il].wo, model.layers[il].bo,
                        Qcur, Kcur, Vcur, nullptr, nullptr, kq_scale, il);
            }

            if (il == n_layer - 1 && inp_out_ids) {
                cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
                inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
            }

            ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
            cb(ffn_inp, "ffn_inp", il);

            cur = build_norm(ffn_inp,
                    model.layers[il].ffn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "ffn_norm", il);

            if ((uint32_t) il < hparams.n_layer_dense_lead) {
                cur = build_ffn(cur,
                        model.layers[il].ffn_up,   NULL, NULL,
                        model.layers[il].ffn_gate, NULL, NULL,
                        model.layers[il].ffn_down, NULL, NULL,
                        NULL,
                        LLM_FFN_SILU, LLM_FFN_PAR, il);
                cb(cur, "ffn_out", il);
            } else {
                // MoE branch
                ggml_tensor * moe_out =
                    build_moe_ffn(cur,
                            model.layers[il].ffn_gate_inp,
                            model.layers[il].ffn_up_exps,
                            model.layers[il].ffn_gate_exps,
                            model.layers[il].ffn_down_exps,
                            nullptr,
                            n_expert, n_expert_used,
                            LLM_FFN_SILU, false,
                            false, hparams.expert_weights_scale,
                            LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX,
                            il);
                cb(moe_out, "ffn_moe_out", il);

                // FFN shared expert
                {
                    ggml_tensor * ffn_shexp = build_ffn(cur,
                            model.layers[il].ffn_up_shexp,   NULL, NULL,
                            model.layers[il].ffn_gate_shexp, NULL, NULL,
                            model.layers[il].ffn_down_shexp, NULL, NULL,
                            NULL,
                            LLM_FFN_SILU, LLM_FFN_PAR, il);
                    cb(ffn_shexp, "ffn_shexp", il);

                    cur = ggml_add(ctx0, moe_out, ffn_shexp);
                    cb(cur, "ffn_out", il);
                }
            }

            cur = ggml_add(ctx0, cur, ffn_inp);

            cur = build_cvec(cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        cur = build_norm(cur,
                model.output_norm, NULL,
                LLM_NORM_RMS, -1);

        cb(cur, "result_norm", -1);
        res->t_embd = cur;

        // lm_head
        cur = build_lora_mm(model.output, cur);

        cb(cur, "result_output", -1);
        res->t_logits = cur;

        ggml_build_forward_expand(gf, cur);
    }
};

struct llm_build_deepseek2 : public llm_graph_context {
    llm_build_deepseek2(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
        bool is_lite = (hparams.n_layer == 27);

        const bool is_mla = (hparams.n_embd_head_k_mla != 0 && hparams.n_embd_head_v_mla != 0);

        // note: these are the actual head sizes you get when treating as MHA or after "decompression" using wv_b for MLA
        const int64_t n_embd_head_k = is_mla ? hparams.n_embd_head_k_mla : hparams.n_embd_head_k;
        const int64_t n_embd_head_v = is_mla ? hparams.n_embd_head_v_mla : hparams.n_embd_head_v;

        const int64_t n_embd_head_qk_rope = hparams.n_rot;
        const int64_t n_embd_head_qk_nope = n_embd_head_k - n_embd_head_qk_rope;

        const uint32_t kv_lora_rank = hparams.n_lora_kv;

        // We have to pre-scale kq_scale and attn_factor to make the YaRN RoPE work correctly.
        // See https://github.com/ggerganov/llama.cpp/discussions/7416 for detailed explanation.
        const float mscale = attn_factor * (1.0f + hparams.rope_yarn_log_mul * logf(1.0f / freq_scale));
        const float kq_scale = 1.0f*mscale*mscale/sqrtf(float(n_embd_head_k));
        const float attn_factor = 1.0f / (1.0f + 0.1f * logf(1.0f / freq_scale));

        ggml_tensor * cur;
        ggml_tensor * inpL;

        // {n_embd, n_tokens}
        inpL = build_inp_embd(model.tok_embd);

        // inp_pos - contains the positions
        ggml_tensor * inp_pos = build_inp_pos();

        auto * inp_attn = build_attn_inp_kv_unified();

        ggml_tensor * inp_out_ids = build_inp_out_ids();

        for (int il = 0; il < n_layer; ++il) {
            ggml_tensor * inpSA = inpL;

            // norm
            cur = build_norm(inpL,
                    model.layers[il].attn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "attn_norm", il);

            // self_attention
            {
                ggml_tensor * q = NULL;
                if (!is_lite) {
                    q = ggml_mul_mat(ctx0, model.layers[il].wq_a, cur);
                    cb(q, "q", il);

                    q = build_norm(q,
                            model.layers[il].attn_q_a_norm, nullptr,
                            LLM_NORM_RMS, il);
                    cb(q, "q", il);

                    q = ggml_mul_mat(ctx0, model.layers[il].wq_b, q);
                    cb(q, "q", il);
                } else {
                    q = ggml_mul_mat(ctx0, model.layers[il].wq, cur);
                    cb(q, "q", il);
                }

                // split into {n_embd_head_qk_nope, n_head, n_tokens}
                ggml_tensor * q_nope = ggml_view_3d(ctx0, q,
                        n_embd_head_qk_nope, n_head, n_tokens,
                        ggml_row_size(q->type, n_embd_head_k),
                        ggml_row_size(q->type, n_embd_head_k) * n_head,
                        0);
                cb(q_nope, "q_nope", il);

                // and {n_embd_head_qk_rope, n_head, n_tokens}
                ggml_tensor * q_pe = ggml_view_3d(ctx0, q,
                        n_embd_head_qk_rope, n_head, n_tokens,
                        ggml_row_size(q->type, n_embd_head_k),
                        ggml_row_size(q->type, n_embd_head_k) * n_head,
                        ggml_row_size(q->type, n_embd_head_qk_nope));
                cb(q_pe, "q_pe", il);

                ggml_tensor * kv_cmpr_pe = ggml_mul_mat(ctx0, model.layers[il].wkv_a_mqa, cur);
                cb(kv_cmpr_pe, "kv_cmpr_pe", il);

                // split into {kv_lora_rank, n_tokens}
                ggml_tensor * kv_cmpr = ggml_view_2d(ctx0, kv_cmpr_pe,
                        kv_lora_rank, n_tokens,
                        ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_embd_head_qk_rope),
                        0);
                cb(kv_cmpr, "kv_cmpr", il);

                // and {n_embd_head_qk_rope, 1, n_tokens}
                ggml_tensor * k_pe = ggml_view_3d(ctx0, kv_cmpr_pe,
                        n_embd_head_qk_rope, 1, n_tokens,
                        ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_embd_head_qk_rope),
                        ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_embd_head_qk_rope),
                        ggml_row_size(kv_cmpr_pe->type, kv_lora_rank));
                cb(k_pe, "k_pe", il);

                q_pe = ggml_rope_ext(ctx0, q_pe, inp_pos, nullptr,
                        n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(q_pe, "q_pe", il);

                k_pe = ggml_rope_ext(ctx0, k_pe, inp_pos, nullptr,
                        n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(k_pe, "k_pe", il);

                kv_cmpr = build_norm(kv_cmpr,
                        model.layers[il].attn_kv_a_norm, nullptr,
                        LLM_NORM_RMS, il);
                cb(kv_cmpr, "kv_cmpr", il);

                if (is_mla) {
                    // {n_embd_head_qk_nope, n_tokens, n_head}
                    q_nope = ggml_permute(ctx0, q_nope, 0, 2, 1, 3);
                    cb(q_nope, "q_nope_perm", il);

                    // {n_embd_head_qk_nope, kv_lora_rank, n_head} x {n_embd_head_qk_nope, n_tokens, n_head}
                    ggml_tensor * q_nope_absorbed = ggml_mul_mat(ctx0, model.layers[il].wk_b, q_nope);
                    cb(q_nope_absorbed, "q_nope_absorbed", il);

                    // {kv_lora_rank, n_head, n_tokens}
                    q_nope_absorbed = ggml_permute(ctx0, q_nope_absorbed, 0, 2, 1, 3);
                    cb(q_nope_absorbed, "q_nope_absorbed_perm", il);

                    // {n_embd_head_qk_rope + kv_lora_rank, n_head, n_tokens}
                    // note: rope must go first for in-place context shifting in build_rope_shift()
                    ggml_tensor * Qcur = ggml_concat(ctx0, q_pe, q_nope_absorbed, 0);
                    cb(Qcur, "Qcur", il);

                    kv_cmpr = ggml_reshape_3d(ctx0, kv_cmpr, kv_lora_rank, 1, n_tokens);
                    cb(kv_cmpr, "kv_cmpr_reshape", il);

                    // {n_embd_head_qk_rope + kv_lora_rank, 1, n_tokens}
                    ggml_tensor * Kcur = ggml_concat(ctx0, k_pe, kv_cmpr, 0);
                    cb(Kcur, "Kcur", il);

                    // {kv_lora_rank, 1, n_tokens}
                    ggml_tensor * Vcur = kv_cmpr;
                    cb(Vcur, "Vcur", il);

                    // note: MLA with the absorption optimzation converts into MQA (ie: GQA with 1 group)
                    cur = build_attn(inp_attn,
                            model.layers[il].wo, NULL,
                            Qcur, Kcur, Vcur, nullptr, model.layers[il].wv_b, kq_scale, il);
                } else {
                    ggml_tensor * kv = ggml_mul_mat(ctx0, model.layers[il].wkv_b, kv_cmpr);
                    cb(kv, "kv", il);

                    // split into {n_embd_head_qk_nope, n_head, n_tokens}
                    ggml_tensor * k_nope = ggml_view_3d(ctx0, kv,
                            n_embd_head_qk_nope, n_head, n_tokens,
                            ggml_row_size(kv->type, n_embd_head_qk_nope + n_embd_head_v),
                            ggml_row_size(kv->type, n_embd_head_qk_nope + n_embd_head_v) * n_head,
                            0);
                    cb(k_nope, "k_nope_view", il);

                    // and {n_embd_head_v, n_head, n_tokens}
                    ggml_tensor * Vcur = ggml_view_3d(ctx0, kv,
                            n_embd_head_v, n_head, n_tokens,
                            ggml_row_size(kv->type, n_embd_head_qk_nope + n_embd_head_v),
                            ggml_row_size(kv->type, n_embd_head_qk_nope + n_embd_head_v) * n_head,
                            ggml_row_size(kv->type, n_embd_head_qk_nope));
                    cb(Vcur, "Vcur_view", il);

                    Vcur = ggml_cont(ctx0, Vcur);
                    cb(Vcur, "Vcur_cont", il);

                    // note: rope must go first for in-place context shifting in build_rope_shift()
                    ggml_tensor * Qcur = ggml_concat(ctx0, q_pe, q_nope, 0);
                    cb(Qcur, "Qcur", il);

                    ggml_tensor * Kcur = ggml_concat(ctx0, ggml_repeat(ctx0, k_pe, q_pe), k_nope, 0);
                    cb(Kcur, "Kcur", il);

                    // note: MLA without the absorption optimization converts into MHA (ie: GQA with full n_head groups)
                    cur = build_attn(inp_attn,
                            model.layers[il].wo, NULL,
                            Qcur, Kcur, Vcur, nullptr, nullptr, kq_scale, il);
                }
            }

            if (il == n_layer - 1 && inp_out_ids) {
                cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
                inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
            }

            ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
            cb(ffn_inp, "ffn_inp", il);

            cur = build_norm(ffn_inp,
                    model.layers[il].ffn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "ffn_norm", il);

            if ((uint32_t) il < hparams.n_layer_dense_lead) {
                cur = build_ffn(cur,
                        model.layers[il].ffn_up,   NULL, NULL,
                        model.layers[il].ffn_gate, NULL, NULL,
                        model.layers[il].ffn_down, NULL, NULL,
                        NULL,
                        LLM_FFN_SILU, LLM_FFN_PAR, il);
                cb(cur, "ffn_out", il);
            } else {
                // MoE branch
                ggml_tensor * moe_out =
                    build_moe_ffn(cur,
                            model.layers[il].ffn_gate_inp,
                            model.layers[il].ffn_up_exps,
                            model.layers[il].ffn_gate_exps,
                            model.layers[il].ffn_down_exps,
                            model.layers[il].ffn_exp_probs_b,
                            n_expert, n_expert_used,
                            LLM_FFN_SILU, hparams.expert_weights_norm,
                            true, hparams.expert_weights_scale,
                            (llama_expert_gating_func_type) hparams.expert_gating_func,
                            il);
                cb(moe_out, "ffn_moe_out", il);

                // FFN shared expert
                {
                    ggml_tensor * ffn_shexp = build_ffn(cur,
                            model.layers[il].ffn_up_shexp,   NULL, NULL,
                            model.layers[il].ffn_gate_shexp, NULL, NULL,
                            model.layers[il].ffn_down_shexp, NULL, NULL,
                            NULL,
                            LLM_FFN_SILU, LLM_FFN_PAR, il);
                    cb(ffn_shexp, "ffn_shexp", il);

                    cur = ggml_add(ctx0, moe_out, ffn_shexp);
                    cb(cur, "ffn_out", il);
                }
            }

            cur = ggml_add(ctx0, cur, ffn_inp);

            cur = build_cvec(cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        cur = build_norm(cur,
                model.output_norm, NULL,
                LLM_NORM_RMS, -1);

        cb(cur, "result_norm", -1);
        res->t_embd = cur;

        // lm_head
        cur = ggml_mul_mat(ctx0, model.output, cur);

        cb(cur, "result_output", -1);
        res->t_logits = cur;

        ggml_build_forward_expand(gf, cur);
    }
};

struct llm_build_bitnet : public llm_graph_context {
    llm_build_bitnet(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
        const int64_t n_embd_head = hparams.n_embd_head_v;

        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);

        ggml_tensor * cur;
        ggml_tensor * inpL;

        inpL = build_inp_embd(model.tok_embd);

        // inp_pos - contains the positions
        ggml_tensor * inp_pos = build_inp_pos();

        auto * inp_attn = build_attn_inp_kv_unified();

        ggml_tensor * inp_out_ids = build_inp_out_ids();

        for (int il = 0; il < n_layer; ++il) {
            ggml_tensor * inpSA = inpL;

            cur = build_norm(inpL,
                    model.layers[il].attn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "attn_norm", il);

            // self-attention
            {
                // compute Q and K and RoPE them
                ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
                if (model.layers[il].wq_scale) {
                    Qcur = ggml_mul(ctx0, Qcur, model.layers[il].wq_scale);
                }
                cb(Qcur, "Qcur", il);
                if (model.layers[il].bq) {
                    Qcur = ggml_add(ctx0, Qcur, model.layers[il].bq);
                    cb(Qcur, "Qcur", il);
                }

                // B1.K
                ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
                if (model.layers[il].wk_scale) {
                    Kcur = ggml_mul(ctx0, Kcur, model.layers[il].wk_scale);
                }
                cb(Kcur, "Kcur", il);
                if (model.layers[il].bk) {
                    Kcur = ggml_add(ctx0, Kcur, model.layers[il].bk);
                    cb(Kcur, "Kcur", il);
                }

                // B1.V
                ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
                if (model.layers[il].wv_scale) {
                    Vcur = ggml_mul(ctx0, Vcur, model.layers[il].wv_scale);
                }
                cb(Vcur, "Vcur", il);
                if (model.layers[il].bv) {
                    Vcur = ggml_add(ctx0, Vcur, model.layers[il].bv);
                    cb(Vcur, "Vcur", il);
                }

                Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
                Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
                Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

                Qcur = ggml_rope_ext(
                        ctx0, Qcur, inp_pos, nullptr,
                        n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow
                        );

                Kcur = ggml_rope_ext(
                        ctx0, Kcur, inp_pos, nullptr,
                        n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow
                        );

                cb(Qcur, "Qcur", il);
                cb(Kcur, "Kcur", il);
                cb(Vcur, "Vcur", il);

                cur = build_attn(inp_attn,
                        NULL, NULL,
                        Qcur, Kcur, Vcur, nullptr, nullptr, 1.0f/sqrtf(float(n_embd_head)), il);

                cur = build_norm(cur,
                        model.layers[il].attn_sub_norm, NULL,
                        LLM_NORM_RMS, il);
                cb(cur, "attn_sub_norm", il);

                cur = build_lora_mm(model.layers[il].wo, cur);
                if (model.layers[il].wo_scale) {
                    cur = ggml_mul(ctx0, cur, model.layers[il].wo_scale);
                }
                if (model.layers[il].bo) {
                    cur = ggml_add(ctx0, cur, model.layers[il].bo);
                }
                cb(cur, "attn_o_out", il);
            }

            if (il == n_layer - 1 && inp_out_ids) {
                cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
                inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
            }

            ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
            cb(ffn_inp, "ffn_inp", il);

            // feed-forward forward
            cur = build_norm(ffn_inp,
                    model.layers[il].ffn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "ffn_norm", il);

            cur = build_ffn(cur,
                    model.layers[il].ffn_up,   NULL, model.layers[il].ffn_up_scale,
                    model.layers[il].ffn_gate, NULL, model.layers[il].ffn_gate_scale,
                    NULL,                      NULL, NULL,
                    NULL,
                    LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(cur, "ffn_sub_out", il);

            cur = build_norm(cur,
                    model.layers[il].ffn_sub_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "ffn_sub_norm", il);

            cur = build_lora_mm(model.layers[il].ffn_down, cur);
            if (model.layers[il].ffn_down_scale) {
                cur = ggml_mul(ctx0, cur, model.layers[il].ffn_down_scale);
            }
            cb(cur, "ffn_down", il);

            cur = ggml_add(ctx0, cur, ffn_inp);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        cur = build_norm(cur,
                model.output_norm, NULL,
                LLM_NORM_RMS, -1);

        cb(cur, "result_norm", -1);
        res->t_embd = cur;

        // lm_head
        // FIXME: do not use model.tok_embd directly, duplicate as model.output
        cur = build_lora_mm(model.tok_embd, cur);

        cb(cur, "result_output", -1);
        res->t_logits = cur;

        ggml_build_forward_expand(gf, cur);
    }
};

struct llm_build_t5_enc : public llm_graph_context {
    llm_build_t5_enc(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
        const int64_t n_embd_head = hparams.n_embd_head_v;

        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);

        ggml_tensor * cur;
        ggml_tensor * inpL;

        inpL = build_inp_embd(model.tok_embd);

        ggml_tensor * pos_bucket_enc = build_inp_pos_bucket_enc();

        auto * inp_attn = build_attn_inp_no_cache();

        ggml_tensor * inp_out_ids = build_inp_out_ids();

        for (int il = 0; il < n_layer; ++il) {
            ggml_tensor * inpSA = inpL;

            // norm
            cur = build_norm(inpL,
                    model.layers[il].attn_norm_enc, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "attn_norm", il);

            // self-attention
            {
                ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq_enc, cur);
                cb(Qcur, "Qcur", il);

                ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk_enc, cur);
                cb(Kcur, "Kcur", il);

                ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv_enc, cur);
                cb(Vcur, "Vcur", il);

                Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
                Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
                Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

                ggml_tensor * attn_rel_b = model.layers[il].attn_rel_b_enc ? model.layers[il].attn_rel_b_enc : model.layers[0].attn_rel_b_enc;
                ggml_tensor * kq_b = build_pos_bias(pos_bucket_enc, attn_rel_b);

                cur = build_attn(inp_attn,
                        model.layers[il].wo_enc, nullptr,
                        Qcur, Kcur, Vcur, kq_b, nullptr, 1.0f, il);
                cb(cur, "kqv_out", il);
            }

            if (il == n_layer - 1 && inp_out_ids) {
                cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
                inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
            }

            ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
            cb(ffn_inp, "ffn_inp", il);

            // feed-forward network
            {
                cur = build_norm(ffn_inp,
                        model.layers[il].ffn_norm_enc, NULL,
                        LLM_NORM_RMS, il);
                cb(cur, "ffn_norm", il);

                // T5 uses relu, flan-T5 uses gelu-gated
                cur = build_ffn(cur,
                        model.layers[il].ffn_up_enc,   NULL, NULL,
                        model.layers[il].ffn_gate_enc, NULL, NULL,
                        model.layers[il].ffn_down_enc, NULL, NULL,
                        NULL,
                        model.layers[il].ffn_gate_enc ? LLM_FFN_GELU : LLM_FFN_RELU,
                        model.layers[il].ffn_gate_enc ? LLM_FFN_PAR  : LLM_FFN_SEQ,
                        il);
                cb(cur, "ffn_out", il);
            }

            cur = ggml_add(ctx0, cur, ffn_inp);
            cb(cur, "ffn_out", il);

            cur = build_cvec(cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;
        cb(cur, "result_embd", -1);

        cur = build_norm(cur,
                model.output_norm_enc, NULL,
                LLM_NORM_RMS, -1);

        cb(cur, "result_norm", -1);
        res->t_embd = cur;

        ggml_build_forward_expand(gf, cur);
    }
};

struct llm_build_t5_dec : public llm_graph_context {
    llm_build_t5_dec(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
        const int64_t n_embd_head = hparams.n_embd_head_v;
        //const int64_t n_embd_gqa  = hparams.n_embd_v_gqa();

        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);

        ggml_tensor * cur;
        ggml_tensor * inpL;

        inpL = build_inp_embd(model.tok_embd);

        ggml_tensor * embd_enc       = build_inp_cross_embd();
        ggml_tensor * pos_bucket_dec = build_inp_pos_bucket_dec();

        const int64_t n_outputs_enc = embd_enc->ne[1];

        auto * inp_attn_self  = build_attn_inp_kv_unified();
        auto * inp_attn_cross = build_attn_inp_cross();

        ggml_tensor * inp_out_ids = build_inp_out_ids();

        for (int il = 0; il < n_layer; ++il) {
            ggml_tensor * inpSA = inpL;

            // norm
            cur = build_norm(inpL,
                    model.layers[il].attn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "attn_norm", il);

            // self-attention
            {
                ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
                cb(Qcur, "Qcur", il);

                ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
                cb(Kcur, "Kcur", il);

                ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
                cb(Vcur, "Vcur", il);

                Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
                Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
                Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

                ggml_tensor * attn_rel_b = model.layers[il].attn_rel_b ? model.layers[il].attn_rel_b : model.layers[0].attn_rel_b;
                ggml_tensor * kq_b = build_pos_bias(pos_bucket_dec, attn_rel_b);

                cur = build_attn(inp_attn_self,
                        model.layers[il].wo, model.layers[il].bo,
                        Qcur, Kcur, Vcur, kq_b, nullptr, 1.0f, il);
                cb(cur, "kqv_out", il);
            }

            cur = ggml_add(ctx0, cur, inpSA);
            cb(cur, "cross_inp", il);

            ggml_tensor * inpCA = cur;

            // norm
            cur = build_norm(cur,
                    model.layers[il].attn_norm_cross, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "attn_norm_cross", il);

            // cross-attention
            {
                ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq_cross, cur);
                cb(Qcur, "Qcur", il);

                ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk_cross, embd_enc);
                cb(Kcur, "Kcur", il);

                ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv_cross, embd_enc);
                cb(Vcur, "Vcur", il);

                Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
                Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_outputs_enc);
                Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_outputs_enc);

                cur = build_attn(inp_attn_cross,
                        model.layers[il].wo_cross, nullptr,
                        Qcur, Kcur, Vcur, nullptr, nullptr, 1.0f, il);
                cb(cur, "kqv_out", il);

                //ggml_tensor * q =                 ggml_permute(ctx0, Qcur, 0, 2, 1, 3);
                //ggml_tensor * k = ggml_cont(ctx0, ggml_permute(ctx0, Kcur, 0, 2, 1, 3));

                //ggml_tensor * kq = ggml_mul_mat(ctx0, k, q);
                //cb(kq, "kq", il);

                //kq = ggml_soft_max_ext(ctx0, kq, KQ_mask_cross, 1.0f, hparams.f_max_alibi_bias);
                //cb(kq, "kq_soft_max_ext", il);

                //ggml_tensor * v = ggml_cont(ctx0, ggml_transpose(ctx0, ggml_reshape_2d(ctx0, Vcur, n_embd_gqa, n_outputs_enc)));
                //cb(v, "v", il);

                //ggml_tensor * kqv = ggml_mul_mat(ctx0, ggml_reshape_3d(ctx0, v, n_outputs_enc, n_embd_head, n_head_kv), kq);
                //cb(kqv, "kqv", il);

                //ggml_tensor * kqv_merged = ggml_permute(ctx0, kqv, 0, 2, 1, 3);
                //cb(kqv_merged, "kqv_merged", il);

                //cur = ggml_cont_2d(ctx0, kqv_merged, n_embd_gqa, n_tokens);
                //cb(cur, "kqv_merged_cont", il);

                //ggml_build_forward_expand(gf, cur);

                //cur = build_lora_mm(model.layers[il].wo_cross, cur);
                //cb(cur, "kqv_out", il);
            }

            if (il == n_layer - 1 && inp_out_ids) {
                cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
                inpCA = ggml_get_rows(ctx0, inpCA, inp_out_ids);
            }

            ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpCA);
            cb(ffn_inp, "ffn_inp", il);

            // feed-forward network
            {
                cur = build_norm(ffn_inp,
                        model.layers[il].ffn_norm, NULL,
                        LLM_NORM_RMS, il);
                cb(cur, "ffn_norm", il);

                // T5 uses relu, flan-T5 uses gelu-gated
                cur = build_ffn(cur,
                        model.layers[il].ffn_up,   NULL, NULL,
                        model.layers[il].ffn_gate, NULL, NULL,
                        model.layers[il].ffn_down, NULL, NULL,
                        NULL,
                        model.layers[il].ffn_gate_enc ? LLM_FFN_GELU : LLM_FFN_RELU,
                        model.layers[il].ffn_gate_enc ? LLM_FFN_PAR : LLM_FFN_SEQ,
                        il);
                cb(cur, "ffn_out", il);
            }

            cur = ggml_add(ctx0, cur, ffn_inp);
            cb(cur, "ffn_out", il);

            cur = build_cvec(cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;
        cb(cur, "result_embd", -1);

        cur = build_norm(cur,
                model.output_norm, NULL,
                LLM_NORM_RMS, -1);

        cb(cur, "result_norm", -1);
        res->t_embd = cur;

        // lm_head
        cur = build_lora_mm(model.output, cur);

        cb(cur, "result_output", -1);
        res->t_logits = cur;

        ggml_build_forward_expand(gf, cur);
    }
};

struct llm_build_jais : public llm_graph_context {
    llm_build_jais(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
        const int64_t n_embd_head = hparams.n_embd_head_v;
        const int64_t n_embd_gqa  = hparams.n_embd_v_gqa();

        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);

        ggml_tensor * cur;
        ggml_tensor * inpL;

        inpL = build_inp_embd(model.tok_embd);

        auto * inp_attn = build_attn_inp_kv_unified();

        ggml_tensor * inp_out_ids = build_inp_out_ids();

        for (int il = 0; il < n_layer; ++il) {
            cur = build_norm(inpL,
                    model.layers[il].attn_norm,
                    model.layers[il].attn_norm_b,
                    LLM_NORM, il);
            cb(cur, "attn_norm", il);

            // self-attention
            {
                cur = build_lora_mm(model.layers[il].wqkv, cur);
                cb(cur, "wqkv", il);

                cur = ggml_add(ctx0, cur, model.layers[il].bqkv);
                cb(cur, "bqkv", il);

                ggml_tensor * Qcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd,     n_tokens, cur->nb[1], 0*cur->nb[0]*(n_embd)));
                ggml_tensor * Kcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd_gqa, n_tokens, cur->nb[1], 1*cur->nb[0]*(n_embd)));
                ggml_tensor * Vcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd_gqa, n_tokens, cur->nb[1], 1*cur->nb[0]*(n_embd + n_embd_gqa)));

                cb(Qcur, "Qcur", il);
                cb(Kcur, "Kcur", il);
                cb(Vcur, "Vcur", il);

                Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
                Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
                Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

                cur = build_attn(inp_attn,
                        model.layers[il].wo, model.layers[il].bo,
                        Qcur, Kcur, Vcur, nullptr, nullptr, 1.0f/float(n_embd_head), il);
            }

            if (il == n_layer - 1 && inp_out_ids) {
                cur  = ggml_get_rows(ctx0,  cur, inp_out_ids);
                inpL = ggml_get_rows(ctx0, inpL, inp_out_ids);
            }

            // add the input
            ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpL);
            cb(ffn_inp, "ffn_inp", il);

            // FF
            {
                cur = build_norm(ffn_inp,
                        model.layers[il].ffn_norm,
                        model.layers[il].ffn_norm_b,
                        LLM_NORM, il);
                cb(cur, "ffn_norm", il);

                cur = build_ffn(cur,
                        model.layers[il].ffn_up,   model.layers[il].ffn_up_b,   NULL,
                        model.layers[il].ffn_gate, model.layers[il].ffn_gate_b, NULL,
                        model.layers[il].ffn_down, model.layers[il].ffn_down_b, NULL,
                        NULL,
                        LLM_FFN_SILU, LLM_FFN_PAR, il);
                cb(cur, "ffn_out", il);
            }

            inpL = ggml_add(ctx0, cur, ffn_inp);
            cb(inpL, "l_out", il);
        }

        cur = build_norm(inpL,
                model.output_norm,
                model.output_norm_b,
                LLM_NORM, -1);

        cb(cur, "result_norm", -1);
        res->t_embd = cur;

        cur = build_lora_mm(model.output, cur);

        cb(cur, "result_output", -1);
        res->t_logits = cur;

        ggml_build_forward_expand(gf, cur);
    }
};

struct llm_build_chatglm : public llm_graph_context {
    llm_build_chatglm(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
        const int64_t n_embd_head = hparams.n_embd_head_v;
        const int64_t n_embd_gqa  = hparams.n_embd_v_gqa();

        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);

        ggml_tensor * cur;
        ggml_tensor * inpL;

        inpL = build_inp_embd(model.tok_embd);

        // inp_pos - contains the positions
        ggml_tensor * inp_pos = build_inp_pos();

        auto * inp_attn = build_attn_inp_kv_unified();

        ggml_tensor * inp_out_ids = build_inp_out_ids();

        for (int il = 0; il < n_layer; ++il) {
            ggml_tensor * inpSA = inpL;

            cur = build_norm(inpL,
                    model.layers[il].attn_norm,
                    NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "attn_norm", il);

            // self-attention
            {
                ggml_tensor * Qcur = nullptr;
                ggml_tensor * Kcur = nullptr;
                ggml_tensor * Vcur = nullptr;

                if (model.layers[il].wqkv == nullptr) {
                    Qcur = build_lora_mm(model.layers[il].wq, cur);
                    if (model.layers[il].bq) {
                        Qcur = ggml_add(ctx0, Qcur, model.layers[il].bq);
                    }
                    Kcur = build_lora_mm(model.layers[il].wk, cur);
                    if (model.layers[il].bk) {
                        Kcur = ggml_add(ctx0, Kcur, model.layers[il].bk);
                    }
                    Vcur = build_lora_mm(model.layers[il].wv, cur);
                    if (model.layers[il].bv) {
                        Vcur = ggml_add(ctx0, Vcur, model.layers[il].bv);
                    }
                    Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
                    Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
                } else {
                    cur = build_lora_mm(model.layers[il].wqkv, cur);
                    cb(cur, "wqkv", il);
                    if (model.layers[il].bqkv) {
                        cur = ggml_add(ctx0, cur, model.layers[il].bqkv);
                        cb(cur, "bqkv", il);
                    }
                    Qcur = ggml_view_3d(ctx0, cur, n_embd_head, n_head,    n_tokens, n_embd_head*sizeof(float), cur->nb[1], 0*sizeof(float)*(n_embd));
                    Kcur = ggml_view_3d(ctx0, cur, n_embd_head, n_head_kv, n_tokens, n_embd_head*sizeof(float), cur->nb[1], 1*sizeof(float)*(n_embd));
                    Vcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd_gqa, n_tokens, cur->nb[1], 1*sizeof(float)*(n_embd + n_embd_gqa)));
                }

                Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

                //printf("freq_base: %f freq_scale: %f ext_factor: %f attn_factor: %f\n", freq_base, freq_scale, ext_factor, attn_factor);
                Qcur = ggml_rope_ext(
                        ctx0, Qcur, inp_pos, nullptr,
                        n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow
                        );

                Kcur = ggml_rope_ext(
                        ctx0, Kcur, inp_pos, nullptr,
                        n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow
                        );

                cb(Qcur, "Qcur", il);
                cb(Kcur, "Kcur", il);
                cb(Vcur, "Vcur", il);

                cur = build_attn(inp_attn,
                        model.layers[il].wo, NULL,
                        Qcur, Kcur, Vcur, nullptr, nullptr, 1.0f/sqrtf(float(n_embd_head)), il);
            }

            if (il == n_layer - 1 && inp_out_ids) {
                cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
                inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
            }

            // Add the input
            ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
            cb(ffn_inp, "ffn_inp", il);

            // FF
            {
                cur = build_norm(ffn_inp,
                        model.layers[il].ffn_norm,
                        NULL,
                        LLM_NORM_RMS, il);
                cb(cur, "ffn_norm", il);

                cur = build_ffn(cur,
                        model.layers[il].ffn_up,   NULL, NULL,
                        NULL,                      NULL, NULL,
                        model.layers[il].ffn_down, NULL, NULL,
                        NULL,
                        LLM_FFN_SWIGLU, LLM_FFN_SEQ, il);
                cb(cur, "ffn_out", il);

            }

            inpL = ggml_add(ctx0, cur, ffn_inp);
            cb(inpL, "l_out", il);
        }

        cur = build_norm(inpL,
                model.output_norm,
                NULL,
                LLM_NORM_RMS, -1);

        cb(cur, "result_norm", -1);
        res->t_embd = cur;

        cur = build_lora_mm(model.output, cur);

        cb(cur, "result_output", -1);
        res->t_logits = cur;

        ggml_build_forward_expand(gf, cur);
    }
};

struct llm_build_glm4 : public llm_graph_context {
    llm_build_glm4(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
        const int64_t n_embd_head = hparams.n_embd_head_v;
        const int64_t n_embd_gqa  = hparams.n_embd_v_gqa();

        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);

        ggml_tensor * cur;
        ggml_tensor * inpL;

        inpL = build_inp_embd(model.tok_embd);

        // inp_pos - contains the positions
        ggml_tensor * inp_pos = build_inp_pos();

        auto * inp_attn = build_attn_inp_kv_unified();

        ggml_tensor * inp_out_ids = build_inp_out_ids();

        for (int il = 0; il < n_layer; ++il) {
            ggml_tensor * inpSA = inpL;

            // Pre-attention norm
            cur = build_norm(inpL,
                    model.layers[il].attn_norm,
                    NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "attn_norm", il);

            // self-attention
            {
                ggml_tensor * Qcur = nullptr;
                ggml_tensor * Kcur = nullptr;
                ggml_tensor * Vcur = nullptr;

                if (model.layers[il].wqkv == nullptr) {
                    Qcur = build_lora_mm(model.layers[il].wq, cur);
                    if (model.layers[il].bq) {
                        Qcur = ggml_add(ctx0, Qcur, model.layers[il].bq);
                    }
                    Kcur = build_lora_mm(model.layers[il].wk, cur);
                    if (model.layers[il].bk) {
                        Kcur = ggml_add(ctx0, Kcur, model.layers[il].bk);
                    }
                    Vcur = build_lora_mm(model.layers[il].wv, cur);
                    if (model.layers[il].bv) {
                        Vcur = ggml_add(ctx0, Vcur, model.layers[il].bv);
                    }
                    Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
                    Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
                } else {
                    cur = build_lora_mm(model.layers[il].wqkv, cur);
                    cb(cur, "wqkv", il);
                    if (model.layers[il].bqkv) {
                        cur = ggml_add(ctx0, cur, model.layers[il].bqkv);
                        cb(cur, "bqkv", il);
                    }
                    Qcur = ggml_view_3d(ctx0, cur, n_embd_head, n_head,    n_tokens, n_embd_head*sizeof(float), cur->nb[1], 0*sizeof(float)*(n_embd));
                    Kcur = ggml_view_3d(ctx0, cur, n_embd_head, n_head_kv, n_tokens, n_embd_head*sizeof(float), cur->nb[1], 1*sizeof(float)*(n_embd));
                    Vcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd_gqa, n_tokens, cur->nb[1], 1*sizeof(float)*(n_embd + n_embd_gqa)));
                }

                Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

                Qcur = ggml_rope_ext(
                        ctx0, Qcur, inp_pos, nullptr,
                        n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow
                        );

                Kcur = ggml_rope_ext(
                        ctx0, Kcur, inp_pos, nullptr,
                        n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow
                        );

                cb(Qcur, "Qcur", il);
                cb(Kcur, "Kcur", il);
                cb(Vcur, "Vcur", il);

                cur = build_attn(inp_attn,
                        model.layers[il].wo, NULL,
                        Qcur, Kcur, Vcur, nullptr, nullptr, 1.0f/sqrtf(float(n_embd_head)), il);
            }

            if (il == n_layer - 1 && inp_out_ids) {
                cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
                inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
            }

            // Post-attention norm (new!)
            cur = build_norm(cur,
                    model.layers[il].attn_post_norm,
                    NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "post_attn_norm", il);

            // Add the input (residual connection after post-attention norm)
            ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
            cb(ffn_inp, "ffn_inp", il);

            // FF
            {
                // Pre-MLP norm
                cur = build_norm(ffn_inp,
                        model.layers[il].ffn_norm,
                        NULL,
                        LLM_NORM_RMS, il);
                cb(cur, "ffn_norm", il);

                // MLP
                cur = build_ffn(cur,
                        model.layers[il].ffn_up,   NULL, NULL,
                        NULL,                      NULL, NULL,
                        model.layers[il].ffn_down, NULL, NULL,
                        NULL,
                        LLM_FFN_SWIGLU, LLM_FFN_SEQ, il);
                cb(cur, "ffn_out", il);

                // Post-MLP norm
                cur = build_norm(cur,
                        model.layers[il].ffn_post_norm,
                        NULL,
                        LLM_NORM_RMS, il);
                cb(cur, "post_mlp_norm", il);
            }

            // Add residual connection after post-MLP norm
            inpL = ggml_add(ctx0, cur, ffn_inp);
            cb(inpL, "l_out", il);
        }

        // Final norm
        cur = build_norm(inpL,
                model.output_norm,
                NULL,
                LLM_NORM_RMS, -1);

        cb(cur, "result_norm", -1);
        res->t_embd = cur;

        // Output projection
        cur = build_lora_mm(model.output, cur);

        cb(cur, "result_output", -1);
        res->t_logits = cur;

        ggml_build_forward_expand(gf, cur);
    }
};

struct llm_build_glm4_moe : public llm_graph_context {
    llm_build_glm4_moe(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
        const int64_t n_embd_head = hparams.n_embd_head_v;

        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);

        ggml_tensor * cur;
        ggml_tensor * inpL;

        inpL = build_inp_embd(model.tok_embd);

        // inp_pos - contains the positions
        ggml_tensor * inp_pos = build_inp_pos();

        auto * inp_attn = build_attn_inp_kv_unified();

        ggml_tensor * inp_out_ids = build_inp_out_ids();

        // Only process up to last layer (skip final NextN layer)
        // Final layer tensors are loaded but not processed in forward pass
        const int n_transformer_layers = n_layer - hparams.nextn_predict_layers;
        for (int il = 0; il < n_transformer_layers; ++il) {
            ggml_tensor * inpSA = inpL;

            // Pre-attention norm
            cur = build_norm(inpL, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, il);
            cb(cur, "attn_norm", il);

            // self-attention
            {
                ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
                if (model.layers[il].bq) {
                    Qcur = ggml_add(ctx0, Qcur, model.layers[il].bq);
                }
                cb(Qcur, "Qcur", il);

                ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
                if (model.layers[il].bk) {
                    Kcur = ggml_add(ctx0, Kcur, model.layers[il].bk);
                }
                cb(Kcur, "Kcur", il);

                ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
                if (model.layers[il].bv) {
                    Vcur = ggml_add(ctx0, Vcur, model.layers[il].bv);
                }
                cb(Vcur, "Vcur", il);

                Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens);
                Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
                Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

                // Apply Q/K norm if available (GLM-4.5 355B variant)
                if (model.layers[il].attn_q_norm) {
                    Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, NULL, LLM_NORM_RMS, il);
                    cb(Qcur, "Qcur_normed", il);
                }
                if (model.layers[il].attn_k_norm) {
                    Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, NULL, LLM_NORM_RMS, il);
                    cb(Kcur, "Kcur_normed", il);
                }

                Qcur = ggml_rope_ext(
                        ctx0, Qcur, inp_pos, nullptr,
                        n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow
                        );

                Kcur = ggml_rope_ext(
                        ctx0, Kcur, inp_pos, nullptr,
                        n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow
                        );

                cb(Qcur, "Qcur", il);
                cb(Kcur, "Kcur", il);
                cb(Vcur, "Vcur", il);

                cur = build_attn(inp_attn,
                        model.layers[il].wo, NULL,
                        Qcur, Kcur, Vcur, nullptr, nullptr, 1.0f/sqrtf(float(n_embd_head)), il);
            }

            if (il == n_transformer_layers - 1 && inp_out_ids) {
                cur   = ggml_get_rows(ctx0, cur, inp_out_ids);
                inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
            }

            ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
            cb(ffn_inp, "ffn_inp", il);

            // Post-attention norm
            cur = build_norm(ffn_inp, model.layers[il].attn_post_norm, NULL, LLM_NORM_RMS, il);
            cb(cur, "post_attn_norm", il);

            // Check if this is a dense layer (n_layer_dense_lead=1, so layer 0 is dense)
            if (static_cast<uint32_t>(il) < hparams.n_layer_dense_lead) {
                // Dense FFN layer
                cur = build_ffn(cur,
                        model.layers[il].ffn_up,   NULL, NULL,
                        model.layers[il].ffn_gate, NULL, NULL,
                        model.layers[il].ffn_down, NULL, NULL,
                        NULL,
                        LLM_FFN_SILU, LLM_FFN_PAR, il);
                cb(cur, "ffn_out", il);
            } else {
                // Process routed experts using existing MoE infrastructure
                ggml_tensor * routed_out = build_moe_ffn(cur,
                        model.layers[il].ffn_gate_inp,
                        model.layers[il].ffn_up_exps,
                        model.layers[il].ffn_gate_exps,
                        model.layers[il].ffn_down_exps,
                        model.layers[il].ffn_exp_probs_b,
                        n_expert, n_expert_used,
                        LLM_FFN_SILU, hparams.expert_weights_norm,
                        true, hparams.expert_weights_scale,
                        (llama_expert_gating_func_type) hparams.expert_gating_func,
                        il);
                cb(routed_out, "ffn_moe_out", il);

                // Process shared expert on original input
                ggml_tensor * shared_out = build_ffn(cur,
                        model.layers[il].ffn_up_shexp,   NULL, NULL,
                        model.layers[il].ffn_gate_shexp, NULL, NULL,
                        model.layers[il].ffn_down_shexp, NULL, NULL,
                        NULL,
                        LLM_FFN_SILU, LLM_FFN_PAR, il);
                cb(shared_out, "ffn_shexp_out", il);

                // Final output: routed_output + shared_output
                cur = ggml_add(ctx0, routed_out, shared_out);
                cb(cur, "ffn_out", il);
            }

            cur = ggml_add(ctx0, cur, ffn_inp);

            cur = build_cvec(cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;
        cur = build_norm(cur, model.output_norm, NULL, LLM_NORM_RMS, -1);

        cb(cur, "result_norm", -1);
        res->t_embd = cur;

        // lm_head
        cur = build_lora_mm(model.output, cur);

        cb(cur, "result_output", -1);
        res->t_logits = cur;

        ggml_build_forward_expand(gf, cur);
    }
};

struct llm_build_nemotron : public llm_graph_context {
    llm_build_nemotron(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
        const int64_t n_embd_head = hparams.n_embd_head_v;

        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
        //GGML_ASSERT(n_embd_head == hparams.n_rot);

        ggml_tensor * cur;
        ggml_tensor * inpL;

        inpL = build_inp_embd(model.tok_embd);

        // inp_pos - contains the positions
        ggml_tensor * inp_pos = build_inp_pos();

        auto * inp_attn = build_attn_inp_kv_unified();

        ggml_tensor * inp_out_ids = build_inp_out_ids();

        for (int il = 0; il < n_layer; ++il) {
            ggml_tensor * inpSA = inpL;

            // norm
            cur = build_norm(inpL,
                    model.layers[il].attn_norm,
                    model.layers[il].attn_norm_b,
                    LLM_NORM, il);
            cb(cur, "attn_norm", il);

            // self-attention
            {
                // compute Q and K and RoPE them
                ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
                cb(Qcur, "Qcur", il);
                if (model.layers[il].bq) {
                    Qcur = ggml_add(ctx0, Qcur, model.layers[il].bq);
                    cb(Qcur, "Qcur", il);
                }

                ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
                cb(Kcur, "Kcur", il);
                if (model.layers[il].bk) {
                    Kcur = ggml_add(ctx0, Kcur, model.layers[il].bk);
                    cb(Kcur, "Kcur", il);
                }

                ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
                cb(Vcur, "Vcur", il);
                if (model.layers[il].bv) {
                    Vcur = ggml_add(ctx0, Vcur, model.layers[il].bv);
                    cb(Vcur, "Vcur", il);
                }

                Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
                Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
                Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

                Qcur = ggml_rope_ext(
                        ctx0, Qcur, inp_pos, nullptr,
                        n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow
                        );

                Kcur = ggml_rope_ext(
                        ctx0, Kcur, inp_pos, nullptr,
                        n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow
                        );

                cb(Qcur, "Qcur", il);
                cb(Kcur, "Kcur", il);
                cb(Vcur, "Vcur", il);

                cur = build_attn(inp_attn,
                        model.layers[il].wo, model.layers[il].bo,
                        Qcur, Kcur, Vcur, nullptr, nullptr, 1.0f/sqrtf(float(n_embd_head)), il);
            }

            if (il == n_layer - 1 && inp_out_ids) {
                cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
                inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
            }

            ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
            cb(ffn_inp, "ffn_inp", il);

            // feed-forward network
            cur = build_norm(ffn_inp,
                    model.layers[il].ffn_norm,
                    model.layers[il].ffn_norm_b,
                    LLM_NORM, il);
            cb(cur, "ffn_norm", il);

            cur = build_ffn(cur,
                    model.layers[il].ffn_up,   model.layers[il].ffn_up_b,   NULL,
                    NULL,                      NULL,                        NULL,
                    model.layers[il].ffn_down, model.layers[il].ffn_down_b, NULL,
                    NULL,
                    LLM_FFN_RELU_SQR, LLM_FFN_SEQ, il);

            cur = ggml_add(ctx0, cur, ffn_inp);
            cb(cur, "ffn_out", il);

            cur = build_cvec(cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        cur = build_norm(cur,
                model.output_norm, model.output_norm_b,
                LLM_NORM, -1);

        cb(cur, "result_norm", -1);
        res->t_embd = cur;

        // lm_head
        cur = build_lora_mm(model.output, cur);

        cb(cur, "result_output", -1);
        res->t_logits = cur;

        ggml_build_forward_expand(gf, cur);
    }
};

struct llm_build_exaone : public llm_graph_context {
    llm_build_exaone(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
        const int64_t n_embd_head = hparams.n_embd_head_v;

        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
        GGML_ASSERT(n_embd_head == hparams.n_rot);

        ggml_tensor * cur;
        ggml_tensor * inpL;

        inpL = build_inp_embd(model.tok_embd);

        // inp_pos - contains the positions
        ggml_tensor * inp_pos = build_inp_pos();

        auto * inp_attn = build_attn_inp_kv_unified();

        ggml_tensor * inp_out_ids = build_inp_out_ids();

        for (int il = 0; il < n_layer; ++il) {
            ggml_tensor * inpSA = inpL;

            // norm
            cur = build_norm(inpL,
                    model.layers[il].attn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "attn_norm", il);

            // self-attention
            {
                // rope freq factors for llama3; may return nullptr for llama2 and other models
                ggml_tensor * rope_factors = model.get_rope_factors(cparams, il);

                // compute Q and K and RoPE them
                ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
                cb(Qcur, "Qcur", il);
                if (model.layers[il].bq) {
                    Qcur = ggml_add(ctx0, Qcur, model.layers[il].bq);
                    cb(Qcur, "Qcur", il);
                }

                ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
                cb(Kcur, "Kcur", il);
                if (model.layers[il].bk) {
                    Kcur = ggml_add(ctx0, Kcur, model.layers[il].bk);
                    cb(Kcur, "Kcur", il);
                }

                ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
                cb(Vcur, "Vcur", il);
                if (model.layers[il].bv) {
                    Vcur = ggml_add(ctx0, Vcur, model.layers[il].bv);
                    cb(Vcur, "Vcur", il);
                }

                Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
                Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
                Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

                Qcur = ggml_rope_ext(
                        ctx0, Qcur, inp_pos, rope_factors,
                        n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow
                        );

                Kcur = ggml_rope_ext(
                        ctx0, Kcur, inp_pos, rope_factors,
                        n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow
                        );

                cb(Qcur, "Qcur", il);
                cb(Kcur, "Kcur", il);
                cb(Vcur, "Vcur", il);

                cur = build_attn(inp_attn,
                        model.layers[il].wo, model.layers[il].bo,
                        Qcur, Kcur, Vcur, nullptr, nullptr, 1.0f/sqrtf(float(n_embd_head)), il);
            }

            if (il == n_layer - 1 && inp_out_ids) {
                cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
                inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
            }

            ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
            cb(ffn_inp, "ffn_inp", il);

            // feed-forward network
            cur = build_norm(ffn_inp,
                    model.layers[il].ffn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "ffn_norm", il);

            cur = build_ffn(cur,
                    model.layers[il].ffn_up,   NULL, NULL,
                    model.layers[il].ffn_gate, NULL, NULL,
                    model.layers[il].ffn_down, NULL, NULL,
                    NULL,
                    LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(cur, "ffn_out", il);

            cur = ggml_add(ctx0, cur, ffn_inp);
            cb(cur, "ffn_out", il);

            cur = build_cvec(cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        cur = build_norm(cur,
                model.output_norm, NULL,
                LLM_NORM_RMS, -1);

        cb(cur, "result_norm", -1);
        res->t_embd = cur;

        // lm_head
        cur = build_lora_mm(model.output, cur);

        cb(cur, "result_output", -1);
        res->t_logits = cur;

        ggml_build_forward_expand(gf, cur);
    }
};

template <bool iswa>
struct llm_build_exaone4 : public llm_graph_context {
    llm_build_exaone4(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
        const int64_t n_embd_head = hparams.n_embd_head_k;

        GGML_ASSERT(n_embd_head == hparams.n_embd_head_v);
        GGML_ASSERT(n_embd_head == hparams.n_rot);

        ggml_tensor * cur;
        ggml_tensor * inpL;

        inpL = build_inp_embd(model.tok_embd);

        // inp_pos - contains the positions
        ggml_tensor * inp_pos = build_inp_pos();

        using inp_attn_type = std::conditional_t<iswa, llm_graph_input_attn_kv_unified_iswa, llm_graph_input_attn_kv_unified>;
        inp_attn_type * inp_attn = nullptr;

        if constexpr (iswa) {
            inp_attn = build_attn_inp_kv_unified_iswa();
        } else {
            inp_attn = build_attn_inp_kv_unified();
        }

        ggml_tensor * inp_out_ids = build_inp_out_ids();

        for (int il = 0; il < n_layer; ++il) {
            ggml_tensor * inpSA = inpL;

            // use RoPE for SWA layers or non-SWA models
            const bool use_rope = hparams.is_swa(il) || hparams.swa_type == LLAMA_SWA_TYPE_NONE;

            cur = inpL;

            // self-attention
            {
                ggml_tensor * rope_factors = model.get_rope_factors(cparams, il);

                ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
                cb(Qcur, "Qcur", il);

                ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
                cb(Kcur, "Kcur", il);

                ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
                cb(Vcur, "Vcur", il);

                Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
                Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
                Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

                Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, NULL, LLM_NORM_RMS, il);
                Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, NULL, LLM_NORM_RMS, il);
                cb(Qcur, "Qcur_normed", il);
                cb(Kcur, "Kcur_normed", il);

                if (use_rope) {
                    Qcur = ggml_rope_ext(
                            ctx0, Qcur, inp_pos, rope_factors,
                            n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                            ext_factor, attn_factor, beta_fast, beta_slow
                            );

                    Kcur = ggml_rope_ext(
                            ctx0, Kcur, inp_pos, rope_factors,
                            n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                            ext_factor, attn_factor, beta_fast, beta_slow
                            );
                }

                cb(Qcur, "Qcur", il);
                cb(Kcur, "Kcur", il);
                cb(Vcur, "Vcur", il);

                cur = build_attn(inp_attn,
                        model.layers[il].wo, NULL,
                        Qcur, Kcur, Vcur, nullptr, nullptr, 1.0f/sqrtf(float(n_embd_head)), il);
                cb(cur, "attn_out", il);
            }

            if (il == n_layer - 1 && inp_out_ids) {
                cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
                inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
            }

            cur = build_norm(cur,
                    model.layers[il].attn_post_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "attn_post_norm", il);

            ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
            cb(ffn_inp, "ffn_inp", il);

            // feed-forward network
            cur = build_ffn(ffn_inp,
                    model.layers[il].ffn_up,   NULL, NULL,
                    model.layers[il].ffn_gate, NULL, NULL,
                    model.layers[il].ffn_down, NULL, NULL,
                    NULL,
                    LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(cur, "ffn_out", il);

            cur = build_norm(cur,
                    model.layers[il].ffn_post_norm, NULL,
                    LLM_NORM_RMS, -1);
            cb(cur, "ffn_post_norm", -1);

            cur = ggml_add(ctx0, cur, ffn_inp);

            cur = build_cvec(cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        cur = build_norm(cur,
                model.output_norm, NULL,
                LLM_NORM_RMS, -1);

        cb(cur, "result_norm", -1);
        res->t_embd = cur;

        // lm_head
        cur = build_lora_mm(model.output, cur);

        cb(cur, "result_output", -1);
        res->t_logits = cur;

        ggml_build_forward_expand(gf, cur);
    }
};

struct llm_build_rwkv6_base : public llm_graph_context {
    const llama_model & model;

    llm_build_rwkv6_base(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params), model(model) {
    }

    ggml_tensor * build_rwkv6_channel_mix(
            const llama_layer * layer,
            ggml_tensor * cur,
            ggml_tensor * x_prev,
            llm_arch arch) const {
        ggml_tensor * sx = ggml_sub(ctx0, x_prev, cur);
        switch (arch) {
            case LLM_ARCH_RWKV6:
                {
                    ggml_tensor * xk = ggml_add(ctx0, ggml_mul(ctx0, sx, layer->channel_mix_lerp_k), cur);
                    ggml_tensor * xr = ggml_add(ctx0, ggml_mul(ctx0, sx, layer->channel_mix_lerp_r), cur);

                    ggml_tensor * r = ggml_sigmoid(ctx0, build_lora_mm(layer->channel_mix_receptance, xr));
                    ggml_tensor * k = ggml_sqr(
                            ctx0,
                            ggml_relu(
                                ctx0,
                                build_lora_mm(layer->channel_mix_key, xk)
                                )
                            );
                    cur = ggml_mul(ctx0, r, build_lora_mm(layer->channel_mix_value, k));
                } break;
            default:
                GGML_ABORT("fatal error");
        }

        return cur;
    }

    ggml_tensor * build_rwkv6_time_mix(
            llm_graph_input_rs * inp,
            ggml_tensor * cur,
            ggml_tensor * x_prev,
            const llama_ubatch & ubatch,
            int   il) const {
        const auto * mctx_cur = static_cast<const llama_memory_recurrent_context *>(mctx);

        const auto n_tokens = ubatch.n_tokens;
        const auto n_seqs = ubatch.n_seqs;
        const auto n_seq_tokens = ubatch.n_seq_tokens;
        const auto n_embd = hparams.n_embd;
        const auto head_size = hparams.wkv_head_size;
        const auto n_head = n_embd / head_size;
        const auto n_head_kv = hparams.n_head_kv(il);

        const auto kv_head = mctx_cur->get_head();

        const auto & layer = model.layers[il];

        bool is_qrwkv = layer.time_mix_first == nullptr;

        ggml_tensor * sx = ggml_sub(ctx0, x_prev, cur);

        sx  = ggml_reshape_2d(ctx0, sx,  n_embd, n_tokens);
        cur = ggml_reshape_2d(ctx0, cur, n_embd, n_tokens);

        ggml_tensor * xxx = ggml_add(ctx0, ggml_mul(ctx0, sx, layer.time_mix_lerp_x), cur);

        xxx = ggml_reshape_4d(
                ctx0,
                ggml_tanh(
                    ctx0,
                    ggml_mul_mat(ctx0, layer.time_mix_w1, xxx)
                    ),
                layer.time_mix_w1->ne[1] / 5, 1, 5, n_tokens
                );

        xxx = ggml_cont(ctx0, ggml_permute(ctx0, xxx, 0, 1, 3, 2));

        xxx = ggml_mul_mat(
                ctx0,
                ggml_reshape_4d(
                    ctx0,
                    layer.time_mix_w2,
                    layer.time_mix_w2->ne[0], layer.time_mix_w2->ne[1], 1, 5
                    ),
                xxx
                );

        ggml_tensor *xw, *xk, *xv, *xr, *xg;
        if (layer.time_mix_lerp_fused) {
            // fusing these weights makes some performance improvement
            sx  = ggml_reshape_3d(ctx0, sx,  n_embd, 1, n_tokens);
            cur = ggml_reshape_3d(ctx0, cur, n_embd, 1, n_tokens);
            xxx = ggml_add(ctx0, ggml_mul(ctx0, ggml_add(ctx0, xxx, layer.time_mix_lerp_fused), sx), cur);
            xw = ggml_view_2d(ctx0, xxx, n_embd, n_tokens, xxx->nb[1], 0);
            xk = ggml_view_2d(ctx0, xxx, n_embd, n_tokens, xxx->nb[1], n_embd * n_tokens * sizeof(float));
            xv = ggml_view_2d(ctx0, xxx, n_embd, n_tokens, xxx->nb[1], n_embd * n_tokens * 2 * sizeof(float));
            xr = ggml_view_2d(ctx0, xxx, n_embd, n_tokens, xxx->nb[1], n_embd * n_tokens * 3 * sizeof(float));
            xg = ggml_view_2d(ctx0, xxx, n_embd, n_tokens, xxx->nb[1], n_embd * n_tokens * 4 * sizeof(float));
        } else {
            // for backward compatibility
            xw = ggml_view_2d(ctx0, xxx, n_embd, n_tokens, xxx->nb[1], 0);
            xk = ggml_view_2d(ctx0, xxx, n_embd, n_tokens, xxx->nb[1], n_embd * n_tokens * sizeof(float));
            xv = ggml_view_2d(ctx0, xxx, n_embd, n_tokens, xxx->nb[1], n_embd * n_tokens * 2 * sizeof(float));
            xr = ggml_view_2d(ctx0, xxx, n_embd, n_tokens, xxx->nb[1], n_embd * n_tokens * 3 * sizeof(float));
            xg = ggml_view_2d(ctx0, xxx, n_embd, n_tokens, xxx->nb[1], n_embd * n_tokens * 4 * sizeof(float));

            xw = ggml_add(ctx0, ggml_mul(ctx0, ggml_add(ctx0, xw, layer.time_mix_lerp_w), sx), cur);
            xk = ggml_add(ctx0, ggml_mul(ctx0, ggml_add(ctx0, xk, layer.time_mix_lerp_k), sx), cur);
            xv = ggml_add(ctx0, ggml_mul(ctx0, ggml_add(ctx0, xv, layer.time_mix_lerp_v), sx), cur);
            xr = ggml_add(ctx0, ggml_mul(ctx0, ggml_add(ctx0, xr, layer.time_mix_lerp_r), sx), cur);
            xg = ggml_add(ctx0, ggml_mul(ctx0, ggml_add(ctx0, xg, layer.time_mix_lerp_g), sx), cur);
        }

        ggml_tensor * r = build_lora_mm(layer.time_mix_receptance, xr);
        ggml_tensor * k = build_lora_mm(layer.time_mix_key,        xk);
        ggml_tensor * v = build_lora_mm(layer.time_mix_value,      xv);
        if (layer.time_mix_receptance_b) {
            r = ggml_add(ctx0, r, layer.time_mix_receptance_b);
        }
        if (layer.time_mix_key_b) {
            k = ggml_add(ctx0, k, layer.time_mix_key_b);
        }
        if (layer.time_mix_value_b) {
            v = ggml_add(ctx0, v, layer.time_mix_value_b);
        }

        ggml_tensor * g = build_lora_mm(layer.time_mix_gate, xg);
        if (is_qrwkv) {
            g = ggml_sigmoid(ctx0, g);
        } else {
            g = ggml_silu(ctx0, g);
        }

        if (n_head_kv != 0 && n_head_kv != n_head) {
            GGML_ASSERT(n_head % n_head_kv == 0);
            k = ggml_reshape_4d(ctx0, k, head_size, 1, n_head_kv, n_tokens);
            v = ggml_reshape_4d(ctx0, v, head_size, 1, n_head_kv, n_tokens);
            ggml_tensor * tmp = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, head_size, n_head / n_head_kv, n_head_kv, n_tokens);
            k = ggml_repeat(ctx0, k, tmp);
            v = ggml_repeat(ctx0, v, tmp);
        }

        k = ggml_reshape_3d(ctx0, k, head_size, n_head, n_tokens);
        v = ggml_reshape_3d(ctx0, v, head_size, n_head, n_tokens);
        r = ggml_reshape_3d(ctx0, r, head_size, n_head, n_tokens);

        ggml_tensor * w = ggml_mul_mat(
                ctx0,
                layer.time_mix_decay_w2,
                ggml_tanh(
                    ctx0,
                    ggml_mul_mat(ctx0, layer.time_mix_decay_w1, xw)
                    )
                );

        w = ggml_add(ctx0, w, layer.time_mix_decay);
        w = ggml_exp(ctx0, ggml_neg(ctx0, ggml_exp(ctx0, w)));
        w = ggml_reshape_3d(ctx0, w, head_size, n_head, n_tokens);

        if (is_qrwkv) {
            // k = k * (1 - w)
            k = ggml_sub(ctx0, k, ggml_mul(ctx0, k, w));
        }

        ggml_tensor * wkv_state = build_rs(
                inp, mctx_cur->get_s_l(il),
                hparams.n_embd_s(), n_seqs);

        ggml_tensor * wkv_output;
        if (is_qrwkv) {
            wkv_output = ggml_gated_linear_attn(ctx0, k, v, r, w, wkv_state, pow(head_size, -0.5f));
        } else {
            wkv_output = ggml_rwkv_wkv6(ctx0, k, v, r, layer.time_mix_first, w, wkv_state);
        }
        cur = ggml_view_1d(ctx0, wkv_output, n_embd * n_tokens, 0);
        wkv_state = ggml_view_1d(ctx0, wkv_output, n_embd * head_size * n_seqs, n_embd * n_tokens * sizeof(float));

        ggml_build_forward_expand(
                gf,
                ggml_cpy(
                    ctx0,
                    wkv_state,
                    ggml_view_1d(
                        ctx0,
                        mctx_cur->get_s_l(il),
                        hparams.n_embd_s() * n_seqs,
                        hparams.n_embd_s() * kv_head * ggml_element_size(mctx_cur->get_s_l(il))
                        )
                    )
                );

        if (!is_qrwkv) {
            // group norm with head_count groups
            cur = ggml_reshape_3d(ctx0, cur, n_embd / n_head, n_head, n_tokens);
            cur = ggml_norm(ctx0, cur, 64e-5f);

            // Convert back to regular vectors.
            cur = ggml_reshape_2d(ctx0, cur, n_embd, n_tokens);
            cur = ggml_add(ctx0, ggml_mul(ctx0, cur, layer.time_mix_ln), layer.time_mix_ln_b);
        } else {
            cur = ggml_reshape_2d(ctx0, cur, n_embd, n_tokens);
        }

        cur = ggml_mul(ctx0, cur, g);
        cur = build_lora_mm(layer.time_mix_output, cur);

        return ggml_reshape_3d(ctx0, cur, n_embd, n_seq_tokens, n_seqs);
    }
};

struct llm_build_rwkv6 : public llm_build_rwkv6_base {
    llm_build_rwkv6(const llama_model & model, const llm_graph_params & params) : llm_build_rwkv6_base(model, params) {
        GGML_ASSERT(hparams.token_shift_count == 2);

        ggml_tensor * cur;
        ggml_tensor * inpL;

        inpL = build_inp_embd(model.tok_embd);
        inpL = build_norm(inpL, model.tok_norm, model.tok_norm_b, LLM_NORM, -1);

        auto * rs_inp = build_rs_inp();

        const auto n_embd = hparams.n_embd;
        const auto n_seq_tokens = ubatch.n_seq_tokens;
        const auto n_seqs = ubatch.n_seqs;

        ggml_tensor * inp_out_ids = build_inp_out_ids();

        for (int il = 0; il < n_layer; ++il) {
            const llama_layer * layer = &model.layers[il];
            inpL = ggml_reshape_3d(ctx0, inpL, n_embd, n_seq_tokens, n_seqs);

            ggml_tensor * token_shift = build_rwkv_token_shift_load(rs_inp, ubatch, il);

            ggml_tensor * att_shift = ggml_view_3d(ctx0, token_shift, n_embd, 1, n_seqs, token_shift->nb[1], token_shift->nb[2], 0);
            ggml_tensor * ffn_shift = ggml_view_3d(ctx0, token_shift, n_embd, 1, n_seqs, token_shift->nb[1], token_shift->nb[2], n_embd * ggml_element_size(token_shift));

            ggml_tensor * att_norm = build_norm(inpL, layer->attn_norm, layer->attn_norm_b, LLM_NORM, il);
            cb(att_norm, "attn_norm", il);

            ggml_tensor * x_prev = ggml_concat(
                    ctx0,
                    att_shift,
                    ggml_view_3d(ctx0, att_norm, n_embd, n_seq_tokens - 1, n_seqs, att_norm->nb[1], att_norm->nb[2], 0),
                    1
                    );

            cur = build_rwkv6_time_mix(rs_inp, att_norm, x_prev, ubatch, il);

            ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpL);
            cb(ffn_inp, "ffn_inp", il);

            ggml_tensor * ffn_norm = build_norm(ffn_inp, layer->attn_norm_2, layer->attn_norm_2_b, LLM_NORM, il);
            cb(ffn_norm, "ffn_norm", il);

            x_prev = ggml_concat(
                    ctx0,
                    ffn_shift,
                    ggml_view_3d(ctx0, ffn_norm, n_embd, n_seq_tokens - 1, n_seqs, ffn_norm->nb[1], ffn_norm->nb[2], 0),
                    1
                    );

            token_shift = ggml_concat(ctx0,
                    ggml_view_3d(ctx0, att_norm, n_embd, 1, n_seqs, att_norm->nb[1], att_norm->nb[2], (n_seq_tokens-1)*n_embd*ggml_element_size(att_norm)),
                    ggml_view_3d(ctx0, ffn_norm, n_embd, 1, n_seqs, ffn_norm->nb[1], ffn_norm->nb[2], (n_seq_tokens-1)*n_embd*ggml_element_size(ffn_norm)),
                    1
                    );
            ggml_build_forward_expand(gf, build_rwkv_token_shift_store(token_shift, ubatch, il));

            ffn_inp  = ggml_reshape_2d(ctx0, ffn_inp,  n_embd, n_tokens);
            ffn_norm = ggml_reshape_2d(ctx0, ffn_norm, n_embd, n_tokens);
            x_prev   = ggml_reshape_2d(ctx0, x_prev,   n_embd, n_tokens);
            cur      = ggml_reshape_2d(ctx0, cur,      n_embd, n_tokens);

            if (il == n_layer - 1 && inp_out_ids) {
                ffn_inp  = ggml_get_rows(ctx0, ffn_inp,  inp_out_ids);
                ffn_norm = ggml_get_rows(ctx0, ffn_norm, inp_out_ids);
                x_prev   = ggml_get_rows(ctx0, x_prev,   inp_out_ids);
                cur      = ggml_get_rows(ctx0, cur,      inp_out_ids);
            }

            cur = build_rwkv6_channel_mix(layer, ffn_norm, x_prev, LLM_ARCH_RWKV6);
            cur = ggml_add(ctx0, cur, ffn_inp);

            if (hparams.rescale_every_n_layers != 0 && (il + 1) % hparams.rescale_every_n_layers == 0) {
                cur = ggml_scale(ctx0, cur, 0.5F);
            }

            cur = build_cvec(cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;
        cur = build_norm(cur, model.output_norm, model.output_norm_b, LLM_NORM, -1);

        cb(cur, "result_norm", -1);
        res->t_embd = cur;

        cur = build_lora_mm(model.output, cur);

        cb(cur, "result_output", -1);
        res->t_logits = cur;

        ggml_build_forward_expand(gf, cur);
    }
};

// ref: https://huggingface.co/recursal/QRWKV6-32B-Instruct-Preview-v0.1/blob/main/modeling_rwkv6qwen2.py
struct llm_build_rwkv6qwen2 : public llm_build_rwkv6_base {
    llm_build_rwkv6qwen2(const llama_model & model, const llm_graph_params & params) : llm_build_rwkv6_base(model, params) {
        GGML_ASSERT(n_embd == hparams.n_embd_r());

        ggml_tensor * cur;
        ggml_tensor * inpL;

        inpL = build_inp_embd(model.tok_embd);

        auto * rs_inp = build_rs_inp();

        const auto n_embd = hparams.n_embd;
        const auto n_seq_tokens = ubatch.n_seq_tokens;
        const auto n_seqs = ubatch.n_seqs;

        ggml_tensor * inp_out_ids = build_inp_out_ids();

        for (int il = 0; il < n_layer; ++il) {
            const llama_layer * layer = &model.layers[il];
            inpL = ggml_reshape_3d(ctx0, inpL, n_embd, n_seq_tokens, n_seqs);

            ggml_tensor * token_shift = build_rwkv_token_shift_load(rs_inp, ubatch, il);

            ggml_tensor * att_norm = build_norm(inpL, layer->attn_norm, layer->attn_norm_b, LLM_NORM_RMS, il);
            cb(att_norm, "attn_norm", il);

            ggml_tensor * x_prev = ggml_concat(
                    ctx0,
                    token_shift,
                    ggml_view_3d(ctx0, att_norm, n_embd, n_seq_tokens - 1, n_seqs, att_norm->nb[1], att_norm->nb[2], 0),
                    1
                    );

            cur = build_rwkv6_time_mix(rs_inp, att_norm, x_prev, ubatch, il);

            token_shift = ggml_view_3d(ctx0, att_norm, n_embd, 1, n_seqs, att_norm->nb[1], att_norm->nb[2], (n_seq_tokens-1)*n_embd*ggml_element_size(att_norm));
            ggml_build_forward_expand(gf, build_rwkv_token_shift_store(token_shift, ubatch, il));

            ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpL);
            cb(ffn_inp, "ffn_inp", il);

            cur     = ggml_reshape_2d(ctx0, cur,     n_embd, n_tokens);
            ffn_inp = ggml_reshape_2d(ctx0, ffn_inp, n_embd, n_tokens);

            if (il == n_layer - 1 && inp_out_ids) {
                cur     = ggml_get_rows(ctx0, cur,     inp_out_ids);
                ffn_inp = ggml_get_rows(ctx0, ffn_inp, inp_out_ids);
            }

            // feed-forward network
            cur = build_norm(ffn_inp,
                    model.layers[il].ffn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "ffn_norm", il);

            cur = build_ffn(cur,
                    model.layers[il].ffn_up,   NULL, NULL,
                    model.layers[il].ffn_gate, NULL, NULL,
                    model.layers[il].ffn_down, NULL, NULL,
                    NULL,
                    LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(cur, "ffn_out", il);

            cur = ggml_add(ctx0, cur, ffn_inp);

            cur = build_cvec(cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;
        cur = build_norm(cur, model.output_norm, model.output_norm_b, LLM_NORM_RMS, -1);

        cb(cur, "result_norm", -1);
        res->t_embd = cur;

        cur = build_lora_mm(model.output, cur);

        cb(cur, "result_output", -1);
        res->t_logits = cur;

        ggml_build_forward_expand(gf, cur);
    }
};

struct llm_build_rwkv7_base : public llm_graph_context {
    const llama_model & model;

    llm_build_rwkv7_base(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params), model(model) {
    }

    ggml_tensor * build_rwkv7_channel_mix(
            const llama_layer * layer,
            ggml_tensor * cur,
            ggml_tensor * x_prev,
            llm_arch arch) const {
        ggml_tensor * sx = ggml_sub(ctx0, x_prev, cur);
        switch (arch) {
            case LLM_ARCH_RWKV7:
                {
                    ggml_tensor * xk = ggml_add(ctx0, ggml_mul(ctx0, sx, layer->channel_mix_lerp_k), cur);

                    ggml_tensor * k = ggml_sqr(
                        ctx0,
                        ggml_relu(
                            ctx0,
                            build_lora_mm(layer->channel_mix_key, xk)
                        )
                    );

                    cur = build_lora_mm(layer->channel_mix_value, k);
                } break;
            default:
                GGML_ABORT("fatal error");
        }

        return cur;
    }

    ggml_tensor * build_rwkv7_time_mix(
            llm_graph_input_rs * inp,
            ggml_tensor * cur,
            ggml_tensor * x_prev,
            ggml_tensor *& first_layer_value,
            const llama_ubatch & ubatch,
            int   il) const {
        const auto * mctx_cur = static_cast<const llama_memory_recurrent_context *>(mctx);

        const auto n_tokens = ubatch.n_tokens;
        const auto n_seqs = ubatch.n_seqs;
        const auto n_embd = hparams.n_embd;
        const auto head_size = hparams.wkv_head_size;
        const auto head_count = n_embd / head_size;
        const auto n_seq_tokens = ubatch.n_seq_tokens;

        const auto kv_head = mctx_cur->get_head();

        const auto & layer = model.layers[il];

        bool has_gating = layer.time_mix_g1 && layer.time_mix_g2;

        ggml_tensor * sx = ggml_sub(ctx0, x_prev, cur);
        ggml_tensor * dummy = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, n_embd, n_seq_tokens, n_seqs, has_gating ? 6 : 5);
        sx = ggml_repeat(ctx0, sx, dummy);

        ggml_tensor * xxx = ggml_add(ctx0, ggml_mul(ctx0, sx, layer.time_mix_lerp_fused), cur);

        ggml_tensor * xr = ggml_view_2d(ctx0, xxx, n_embd, n_tokens, xxx->nb[1], 0);
        ggml_tensor * xw = ggml_view_2d(ctx0, xxx, n_embd, n_tokens, xxx->nb[1], n_embd * n_tokens * sizeof(float));
        ggml_tensor * xk = ggml_view_2d(ctx0, xxx, n_embd, n_tokens, xxx->nb[1], n_embd * n_tokens * 2 * sizeof(float));
        ggml_tensor * xv = ggml_view_2d(ctx0, xxx, n_embd, n_tokens, xxx->nb[1], n_embd * n_tokens * 3 * sizeof(float));
        ggml_tensor * xa = ggml_view_2d(ctx0, xxx, n_embd, n_tokens, xxx->nb[1], n_embd * n_tokens * 4 * sizeof(float));
        ggml_tensor * xg = has_gating ? ggml_view_2d(ctx0, xxx, n_embd, n_tokens, xxx->nb[1], n_embd * n_tokens * 5 * sizeof(float)) : nullptr;

        ggml_tensor * r = build_lora_mm(layer.time_mix_receptance, xr);
        ggml_tensor * w = ggml_add(
            ctx0,
            ggml_mul_mat(ctx0, layer.time_mix_w2, ggml_tanh(ctx0, ggml_mul_mat(ctx0, layer.time_mix_w1, xw))),
            layer.time_mix_w0
        );
        w = ggml_exp(ctx0, ggml_scale(ctx0, ggml_sigmoid(ctx0, w), -0.606531));

        ggml_tensor * k = build_lora_mm(layer.time_mix_key, xk);
        ggml_tensor * v = build_lora_mm(layer.time_mix_value, xv);
        if (first_layer_value == nullptr) {
            first_layer_value = v;
        } else {
            // Add the first layer value as a residual connection.
            v = ggml_add(ctx0, v,
                ggml_mul(ctx0,
                    ggml_sub(ctx0, first_layer_value, v),
                    ggml_sigmoid(ctx0, ggml_add(ctx0,
                            ggml_mul_mat(ctx0, layer.time_mix_v2, ggml_mul_mat(ctx0, layer.time_mix_v1, xv)),
                            layer.time_mix_v0
                        )
                    )
                )
            );
        }

        ggml_tensor * g = nullptr;
        if (layer.time_mix_g1 && layer.time_mix_g2) {
            g = ggml_mul_mat(ctx0, layer.time_mix_g2, ggml_sigmoid(ctx0, ggml_mul_mat(ctx0, layer.time_mix_g1, xg)));
        }

        ggml_tensor * a = ggml_sigmoid(ctx0,
            ggml_add(
                ctx0,
                ggml_mul_mat(ctx0, layer.time_mix_a2, ggml_mul_mat(ctx0, layer.time_mix_a1, xa)),
                layer.time_mix_a0
            )
        );

        ggml_tensor * kk = ggml_reshape_3d(ctx0, ggml_mul(ctx0, k, layer.time_mix_k_k), head_size, head_count, n_tokens);
        kk = ggml_l2_norm(ctx0, kk, 1e-12);

        ggml_tensor * ka = ggml_mul(ctx0, k, layer.time_mix_k_a);
        k = ggml_add(ctx0, k, ggml_sub(ctx0, ggml_mul(ctx0, a, ka), ka));

        r = ggml_reshape_3d(ctx0, r, head_size, head_count, n_tokens);
        w = ggml_reshape_3d(ctx0, w, head_size, head_count, n_tokens);
        k = ggml_reshape_3d(ctx0, k, head_size, head_count, n_tokens);
        v = ggml_reshape_3d(ctx0, v, head_size, head_count, n_tokens);
        a = ggml_reshape_3d(ctx0, a, head_size, head_count, n_tokens);

        ggml_tensor * wkv_state = build_rs(
                inp, mctx_cur->get_s_l(il),
                hparams.n_embd_s(), n_seqs);

        ggml_tensor * wkv_output = ggml_rwkv_wkv7(ctx0, r, w, k, v, ggml_neg(ctx0, kk), ggml_mul(ctx0, kk, a), wkv_state);
        cur = ggml_view_1d(ctx0, wkv_output, n_embd * n_tokens, 0);
        wkv_state = ggml_view_1d(ctx0, wkv_output, n_embd * head_size * n_seqs, n_embd * n_tokens * sizeof(float));

        ggml_build_forward_expand(
                gf,
                ggml_cpy(
                    ctx0,
                    wkv_state,
                    ggml_view_1d(
                        ctx0,
                        mctx_cur->get_s_l(il),
                        hparams.n_embd_s() * n_seqs,
                        hparams.n_embd_s() * kv_head * ggml_element_size(mctx_cur->get_s_l(il))
                        )
                    )
                );

        if (layer.time_mix_ln && layer.time_mix_ln_b) {
            // group norm with head_count groups
            cur = ggml_reshape_3d(ctx0, cur, n_embd / head_count, head_count, n_tokens);
            cur = ggml_norm(ctx0, cur, 64e-5f);

            // Convert back to regular vectors.
            cur = ggml_reshape_2d(ctx0, cur, n_embd, n_tokens);
            cur = ggml_add(ctx0, ggml_mul(ctx0, cur, layer.time_mix_ln), layer.time_mix_ln_b);
        } else {
            cur = ggml_reshape_2d(ctx0, cur, n_embd, n_tokens);
        }

        ggml_tensor * rk = ggml_sum_rows(ctx0,
                ggml_mul(ctx0, ggml_mul(ctx0, k, r), ggml_reshape_2d(ctx0, layer.time_mix_r_k, head_size, head_count)));
        cur = ggml_add(ctx0, cur, ggml_reshape_2d(ctx0, ggml_mul(ctx0, v, rk), n_embd, n_tokens));

        if (has_gating) {
            cur = ggml_mul(ctx0, cur, g);
        }
        cur = build_lora_mm(layer.time_mix_output, cur);

        return ggml_reshape_3d(ctx0, cur, n_embd, n_seq_tokens, n_seqs);
    }
};

struct llm_build_rwkv7 : public llm_build_rwkv7_base {
    llm_build_rwkv7(const llama_model & model, const llm_graph_params & params) : llm_build_rwkv7_base(model, params) {
        GGML_ASSERT(hparams.token_shift_count == 2);

        ggml_tensor * cur;
        ggml_tensor * inpL;
        ggml_tensor * v_first = nullptr;

        inpL = build_inp_embd(model.tok_embd);
        inpL = build_norm(inpL, model.tok_norm, model.tok_norm_b, LLM_NORM, -1);

        auto * rs_inp = build_rs_inp();

        const auto n_embd = hparams.n_embd;
        const auto n_seq_tokens = ubatch.n_seq_tokens;
        const auto n_seqs = ubatch.n_seqs;

        ggml_tensor * inp_out_ids = build_inp_out_ids();

        for (int il = 0; il < n_layer; ++il) {
            const llama_layer * layer = &model.layers[il];
            inpL = ggml_reshape_3d(ctx0, inpL, n_embd, n_seq_tokens, n_seqs);

            ggml_tensor * token_shift = build_rwkv_token_shift_load(rs_inp, ubatch, il);

            ggml_tensor * att_shift = ggml_view_3d(ctx0, token_shift, n_embd, 1, n_seqs, token_shift->nb[1], token_shift->nb[2], 0);
            ggml_tensor * ffn_shift = ggml_view_3d(ctx0, token_shift, n_embd, 1, n_seqs, token_shift->nb[1], token_shift->nb[2], n_embd * ggml_element_size(token_shift));

            ggml_tensor * att_norm = build_norm(inpL, layer->attn_norm, layer->attn_norm_b, LLM_NORM, il);
            cb(att_norm, "attn_norm", il);

            ggml_tensor * x_prev = ggml_concat(
                    ctx0,
                    att_shift,
                    ggml_view_3d(ctx0, att_norm, n_embd, n_seq_tokens - 1, n_seqs, att_norm->nb[1], att_norm->nb[2], 0),
                    1
                    );

            cur = build_rwkv7_time_mix(rs_inp, att_norm, x_prev, v_first, ubatch, il);

            ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpL);
            cb(ffn_inp, "ffn_inp", il);

            ggml_tensor * ffn_norm = build_norm(ffn_inp, layer->attn_norm_2, layer->attn_norm_2_b, LLM_NORM, il);
            cb(ffn_norm, "ffn_norm", il);

            x_prev = ggml_concat(
                    ctx0,
                    ffn_shift,
                    ggml_view_3d(ctx0, ffn_norm, n_embd, n_seq_tokens - 1, n_seqs, ffn_norm->nb[1], ffn_norm->nb[2], 0),
                    1
                    );

            token_shift = ggml_concat(ctx0,
                    ggml_view_3d(ctx0, att_norm, n_embd, 1, n_seqs, att_norm->nb[1], att_norm->nb[2], (n_seq_tokens-1)*n_embd*ggml_element_size(att_norm)),
                    ggml_view_3d(ctx0, ffn_norm, n_embd, 1, n_seqs, ffn_norm->nb[1], ffn_norm->nb[2], (n_seq_tokens-1)*n_embd*ggml_element_size(ffn_norm)),
                    1
                    );
            ggml_build_forward_expand(gf, build_rwkv_token_shift_store(token_shift, ubatch, il));

            ffn_inp  = ggml_reshape_2d(ctx0, ffn_inp,  n_embd, n_tokens);
            ffn_norm = ggml_reshape_2d(ctx0, ffn_norm, n_embd, n_tokens);
            x_prev   = ggml_reshape_2d(ctx0, x_prev,   n_embd, n_tokens);

            if (il == n_layer - 1 && inp_out_ids) {
                ffn_inp  = ggml_get_rows(ctx0, ffn_inp,  inp_out_ids);
                ffn_norm = ggml_get_rows(ctx0, ffn_norm, inp_out_ids);
                x_prev   = ggml_get_rows(ctx0, x_prev,   inp_out_ids);
            }

            cur = build_rwkv7_channel_mix(layer, ffn_norm, x_prev, LLM_ARCH_RWKV7);
            cur = ggml_add(ctx0, cur, ffn_inp);

            cur = build_cvec(cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;
        cur = build_norm(cur, model.output_norm, model.output_norm_b, LLM_NORM, -1);

        cb(cur, "result_norm", -1);
        res->t_embd = cur;

        cur = build_lora_mm(model.output, cur);

        cb(cur, "result_output", -1);
        res->t_logits = cur;

        ggml_build_forward_expand(gf, cur);
    }
};


struct llm_build_arwkv7 : public llm_build_rwkv7_base {
    llm_build_arwkv7(const llama_model & model, const llm_graph_params & params) : llm_build_rwkv7_base(model, params) {
        GGML_ASSERT(n_embd == hparams.n_embd_r());

        ggml_tensor * cur;
        ggml_tensor * inpL;
        ggml_tensor * v_first = nullptr;

        inpL = build_inp_embd(model.tok_embd);

        auto * rs_inp = build_rs_inp();

        const auto n_embd = hparams.n_embd;
        const auto n_seq_tokens = ubatch.n_seq_tokens;
        const auto n_seqs = ubatch.n_seqs;

        ggml_tensor * inp_out_ids = build_inp_out_ids();

        for (int il = 0; il < n_layer; ++il) {
            const llama_layer * layer = &model.layers[il];
            inpL = ggml_reshape_3d(ctx0, inpL, n_embd, n_seq_tokens, n_seqs);

            ggml_tensor * token_shift = build_rwkv_token_shift_load(rs_inp, ubatch, il);

            ggml_tensor * att_norm = build_norm(inpL, layer->attn_norm, layer->attn_norm_b, LLM_NORM_RMS, il);
            cb(att_norm, "attn_norm", il);

            ggml_tensor * x_prev = ggml_concat(
                    ctx0,
                    token_shift,
                    ggml_view_3d(ctx0, att_norm, n_embd, n_seq_tokens - 1, n_seqs, att_norm->nb[1], att_norm->nb[2], 0),
                    1
                    );

            cur = build_rwkv7_time_mix(rs_inp, att_norm, x_prev, v_first, ubatch, il);

            token_shift = ggml_view_3d(ctx0, att_norm, n_embd, 1, n_seqs, att_norm->nb[1], att_norm->nb[2], (n_seq_tokens-1)*n_embd*ggml_element_size(att_norm));
            ggml_build_forward_expand(gf, build_rwkv_token_shift_store(token_shift, ubatch, il));

            ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpL);
            cb(ffn_inp, "ffn_inp", il);

            cur     = ggml_reshape_2d(ctx0, cur,     n_embd, n_tokens);
            ffn_inp = ggml_reshape_2d(ctx0, ffn_inp, n_embd, n_tokens);

            if (il == n_layer - 1 && inp_out_ids) {
                cur     = ggml_get_rows(ctx0, cur,     inp_out_ids);
                ffn_inp = ggml_get_rows(ctx0, ffn_inp, inp_out_ids);
            }

            // feed-forward network
            cur = build_norm(ffn_inp,
                    model.layers[il].ffn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "ffn_norm", il);

            cur = build_ffn(cur,
                    model.layers[il].ffn_up,   NULL, NULL,
                    model.layers[il].ffn_gate, NULL, NULL,
                    model.layers[il].ffn_down, NULL, NULL,
                    NULL,
                    LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(cur, "ffn_out", il);

            cur = ggml_add(ctx0, cur, ffn_inp);

            cur = build_cvec(cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;
        cur = build_norm(cur, model.output_norm, model.output_norm_b, LLM_NORM_RMS, -1);

        cb(cur, "result_norm", -1);
        res->t_embd = cur;

        cur = build_lora_mm(model.output, cur);

        cb(cur, "result_output", -1);
        res->t_logits = cur;

        ggml_build_forward_expand(gf, cur);
    }
};

struct llm_build_granite : public llm_graph_context {
    llm_build_granite(
        const llama_model & model,
        const llm_graph_params & params)
        : llm_graph_context(params) {

        const int64_t n_embd_head = hparams.n_embd_head_v;

        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
        GGML_ASSERT(n_embd_head == hparams.n_rot);

        ggml_tensor * cur;
        ggml_tensor * inpL;

        inpL = build_inp_embd(model.tok_embd);

        // inp_pos - built only if rope enabled
        ggml_tensor * inp_pos = nullptr;
        if (hparams.rope_finetuned) {
            inp_pos = build_inp_pos();
        }

        auto * inp_attn = build_attn_inp_kv_unified();

        ggml_tensor * inp_out_ids = build_inp_out_ids();

        for (int il = 0; il < n_layer; ++il) {
            ggml_tensor * inpSA = inpL;

            // norm
            cur = build_norm(inpL,
                    model.layers[il].attn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "attn_norm", il);

            // self-attention
            cur = build_attention_layer(
                cur, inp_pos, inp_attn,
                model, n_embd_head, il);

            if (il == n_layer - 1 && inp_out_ids) {
                cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
                inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
            }

            // ffn
            cur = build_layer_ffn(cur, inpSA, model, il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        cur = build_norm(cur,
                model.output_norm, NULL,
                LLM_NORM_RMS, -1);

        cb(cur, "result_norm", -1);
        res->t_embd = cur;

        // lm_head
        cur = build_lora_mm(model.output, cur);

        // For Granite architectures - scale logits
        cur = ggml_scale(ctx0, cur, 1.0f / hparams.f_logit_scale);
        cb(cur, "result_output", -1);
        res->t_logits = cur;

        ggml_build_forward_expand(gf, cur);
    }

    ggml_tensor * build_attention_layer(
              ggml_tensor                     * cur,
              ggml_tensor                     * inp_pos,
              llm_graph_input_attn_kv_unified * inp_attn,
        const llama_model                     & model,
        const int64_t                           n_embd_head,
        const int                               il) {

        // compute Q and K and (optionally) RoPE them
        ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
        cb(Qcur, "Qcur", il);
        if (model.layers[il].bq) {
            Qcur = ggml_add(ctx0, Qcur, model.layers[il].bq);
            cb(Qcur, "Qcur", il);
        }

        ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
        cb(Kcur, "Kcur", il);
        if (model.layers[il].bk) {
            Kcur = ggml_add(ctx0, Kcur, model.layers[il].bk);
            cb(Kcur, "Kcur", il);
        }

        ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
        cb(Vcur, "Vcur", il);
        if (model.layers[il].bv) {
            Vcur = ggml_add(ctx0, Vcur, model.layers[il].bv);
            cb(Vcur, "Vcur", il);
        }

        Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, hparams.n_head(il),    n_tokens);
        Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, hparams.n_head_kv(il), n_tokens);
        Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, hparams.n_head_kv(il), n_tokens);

        const bool use_rope = hparams.rope_finetuned;
        if (use_rope) {
            ggml_tensor * rope_factors = model.get_rope_factors(cparams, il);
            Qcur = ggml_rope_ext(
                    ctx0, Qcur, inp_pos, rope_factors,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );

            Kcur = ggml_rope_ext(
                    ctx0, Kcur, inp_pos, rope_factors,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );
        }

        cb(Qcur, "Qcur", il);
        cb(Kcur, "Kcur", il);
        cb(Vcur, "Vcur", il);

        const float kq_scale = hparams.f_attention_scale == 0.0f ? 1.0f/sqrtf(float(n_embd_head)) : hparams.f_attention_scale;
        cur = build_attn(inp_attn,
                model.layers[il].wo, model.layers[il].bo,
                Qcur, Kcur, Vcur, nullptr, nullptr, kq_scale, il);
                cb(cur, "attn_out", il);
        return cur;
    }

    ggml_tensor * build_layer_ffn(
              ggml_tensor       * cur,
              ggml_tensor       * inpSA,
        const llama_model       & model,
        const int                 il) {

        // For Granite architectures - scale residual
        if (hparams.f_residual_scale) {
            cur = ggml_scale(ctx0, cur, hparams.f_residual_scale);
        }
        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        // feed-forward network (non-MoE)
        if (model.layers[il].ffn_gate_inp == nullptr) {

            cur = build_norm(ffn_inp,
                    model.layers[il].ffn_norm, NULL,
                    LLM_NORM_RMS, il);
                    cb(cur, "ffn_norm", il);

            cur = build_ffn(cur,
                    model.layers[il].ffn_up,   model.layers[il].ffn_up_b,   NULL,
                    model.layers[il].ffn_gate, model.layers[il].ffn_gate_b, NULL,
                    model.layers[il].ffn_down, model.layers[il].ffn_down_b, NULL,
                    NULL,
                    LLM_FFN_SILU, LLM_FFN_PAR, il);
                    cb(cur, "ffn_out", il);

        } else {
            // MoE branch
            cur = build_norm(ffn_inp,
                    model.layers[il].ffn_norm, NULL,
                    LLM_NORM_RMS, il);
                    cb(cur, "ffn_norm", il);

            ggml_tensor * moe_out = build_moe_ffn(cur,
                    model.layers[il].ffn_gate_inp,
                    model.layers[il].ffn_up_exps,
                    model.layers[il].ffn_gate_exps,
                    model.layers[il].ffn_down_exps,
                    nullptr,
                    n_expert, n_expert_used,
                    LLM_FFN_SILU, true,
                    false, 0.0,
                    LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX,
                    il);
            cb(moe_out, "ffn_moe_out", il);

            // For Granite MoE Shared
            if (hparams.n_ff_shexp > 0) {
                ggml_tensor * ffn_shexp = build_ffn(cur,
                    model.layers[il].ffn_up_shexp,   NULL, NULL,
                    model.layers[il].ffn_gate_shexp, NULL, NULL,
                    model.layers[il].ffn_down_shexp, NULL, NULL,
                    NULL,
                    LLM_FFN_SILU, LLM_FFN_PAR, il);
                cb(ffn_shexp, "ffn_shexp", il);

                cur = ggml_add(ctx0, moe_out, ffn_shexp);
                cb(cur, "ffn_out", il);
            } else {
                cur = moe_out;
            }
        }

        // For Granite architectures - scale residual
        if (hparams.f_residual_scale) {
            cur = ggml_scale(ctx0, cur, hparams.f_residual_scale);
        }
        cur = ggml_add(ctx0, cur, ffn_inp);
        cb(cur, "ffn_out", il);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        return cur;
    }
};

struct llm_build_granite_hybrid : public llm_graph_context_mamba {
    llm_build_granite_hybrid(
                 const llama_model & model,
            const llm_graph_params & params) :
        llm_graph_context_mamba(params) {

        const int64_t n_embd_head = hparams.n_embd_head_v;
        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);

        ggml_tensor * cur;
        ggml_tensor * inpL;

        inpL = build_inp_embd(model.tok_embd);

        auto * inp = build_inp_mem_hybrid();

        ggml_tensor * inp_out_ids = build_inp_out_ids();

        // Positional embeddings populated if rope enabled
        ggml_tensor * inp_pos = nullptr;
        if (hparams.rope_finetuned) {
            inp_pos = build_inp_pos();
        }

        for (int il = 0; il < n_layer; ++il) {
            struct ggml_tensor * inpSA = inpL;

            // norm
            cur = build_norm(inpL,
                    model.layers[il].attn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "attn_norm", il);

            if (hparams.is_recurrent(il)) {
                // ssm layer //
                cur = build_mamba2_layer(inp->get_recr(), cur, model, ubatch, il);
            } else {
                // attention layer //
                cur = build_attention_layer(
                    cur, inp_pos, inp->get_attn(), model,
                    n_embd_head, il);
            }

            if (il == n_layer - 1 && inp_out_ids) {
                cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
                inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
            }

            // ffn
            cur = build_layer_ffn(cur, inpSA, model, il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        cur = build_norm(cur,
                model.output_norm, NULL,
                LLM_NORM_RMS, -1);

        cb(cur, "result_norm", -1);
        res->t_embd = cur;

        // lm_head
        cur = build_lora_mm(model.output, cur);

        // For Granite architectures - scale logits
        if (hparams.f_logit_scale) {
            cur = ggml_scale(ctx0, cur, 1.0f / hparams.f_logit_scale);
        }
        cb(cur, "result_output", -1);
        res->t_logits = cur;

        ggml_build_forward_expand(gf, cur);
    }

    ggml_tensor * build_attention_layer(
              ggml_tensor                     * cur,
              ggml_tensor                     * inp_pos,
              llm_graph_input_attn_kv_unified * inp_attn,
        const llama_model                     & model,
        const int64_t                           n_embd_head,
        const int                               il) {

        // compute Q and K and (optionally) RoPE them
        ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
        cb(Qcur, "Qcur", il);
        if (model.layers[il].bq) {
            Qcur = ggml_add(ctx0, Qcur, model.layers[il].bq);
            cb(Qcur, "Qcur", il);
        }

        ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
        cb(Kcur, "Kcur", il);
        if (model.layers[il].bk) {
            Kcur = ggml_add(ctx0, Kcur, model.layers[il].bk);
            cb(Kcur, "Kcur", il);
        }

        ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
        cb(Vcur, "Vcur", il);
        if (model.layers[il].bv) {
            Vcur = ggml_add(ctx0, Vcur, model.layers[il].bv);
            cb(Vcur, "Vcur", il);
        }

        Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, hparams.n_head(il),    n_tokens);
        Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, hparams.n_head_kv(il), n_tokens);
        Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, hparams.n_head_kv(il), n_tokens);

        const bool use_rope = hparams.rope_finetuned;
        if (use_rope) {
            ggml_tensor * rope_factors = model.get_rope_factors(cparams, il);
            Qcur = ggml_rope_ext(
                    ctx0, Qcur, inp_pos, rope_factors,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );

            Kcur = ggml_rope_ext(
                    ctx0, Kcur, inp_pos, rope_factors,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );
        }

        cb(Qcur, "Qcur", il);
        cb(Kcur, "Kcur", il);
        cb(Vcur, "Vcur", il);

        const float kq_scale = hparams.f_attention_scale == 0.0f ? 1.0f/sqrtf(float(n_embd_head)) : hparams.f_attention_scale;
        cur = build_attn(inp_attn,
                model.layers[il].wo, model.layers[il].bo,
                Qcur, Kcur, Vcur, nullptr, nullptr, kq_scale, il);
                cb(cur, "attn_out", il);
        return cur;
    }

    ggml_tensor * build_layer_ffn(
              ggml_tensor       * cur,
              ggml_tensor       * inpSA,
        const llama_model       & model,
        const int                 il) {

        // For Granite architectures - scale residual
        if (hparams.f_residual_scale) {
            cur = ggml_scale(ctx0, cur, hparams.f_residual_scale);
        }
        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        // feed-forward network (non-MoE)
        if (model.layers[il].ffn_gate_inp == nullptr) {

            cur = build_norm(ffn_inp,
                    model.layers[il].ffn_norm, NULL,
                    LLM_NORM_RMS, il);
                    cb(cur, "ffn_norm", il);

            cur = build_ffn(cur,
                    model.layers[il].ffn_up,   model.layers[il].ffn_up_b,   NULL,
                    model.layers[il].ffn_gate, model.layers[il].ffn_gate_b, NULL,
                    model.layers[il].ffn_down, model.layers[il].ffn_down_b, NULL,
                    NULL,
                    LLM_FFN_SILU, LLM_FFN_PAR, il);
                    cb(cur, "ffn_out", il);

        } else {
            // MoE branch
            cur = build_norm(ffn_inp,
                    model.layers[il].ffn_norm, NULL,
                    LLM_NORM_RMS, il);
                    cb(cur, "ffn_norm", il);

            ggml_tensor * moe_out = build_moe_ffn(cur,
                    model.layers[il].ffn_gate_inp,
                    model.layers[il].ffn_up_exps,
                    model.layers[il].ffn_gate_exps,
                    model.layers[il].ffn_down_exps,
                    nullptr,
                    n_expert, n_expert_used,
                    LLM_FFN_SILU, true,
                    false, 0.0,
                    LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX,
                    il);
            cb(moe_out, "ffn_moe_out", il);

            // For Granite MoE Shared
            if (hparams.n_ff_shexp > 0) {
                ggml_tensor * ffn_shexp = build_ffn(cur,
                    model.layers[il].ffn_up_shexp,   NULL, NULL,
                    model.layers[il].ffn_gate_shexp, NULL, NULL,
                    model.layers[il].ffn_down_shexp, NULL, NULL,
                    NULL,
                    LLM_FFN_SILU, LLM_FFN_PAR, il);
                cb(ffn_shexp, "ffn_shexp", il);

                cur = ggml_add(ctx0, moe_out, ffn_shexp);
                cb(cur, "ffn_out", il);
            } else {
                cur = moe_out;
            }
        }

        // For Granite architectures - scale residual
        if (hparams.f_residual_scale) {
            cur = ggml_scale(ctx0, cur, hparams.f_residual_scale);
        }
        cur = ggml_add(ctx0, cur, ffn_inp);
        cb(cur, "ffn_out", il);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        return cur;
    }
};

struct llm_build_solar : public llm_graph_context {
    llm_build_solar(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
        const int64_t n_embd_head = hparams.n_embd_head_v;
        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
        GGML_ASSERT(n_embd_head == hparams.n_rot);

        struct ggml_tensor * cur;
        struct ggml_tensor * inpL;

        inpL = build_inp_embd(model.tok_embd);

        // inp_pos - contains the positions
        struct ggml_tensor * inp_pos = build_inp_pos();

        // KQ_mask (mask for 1 head, it will be broadcasted to all heads)
        auto * inp_attn = build_attn_inp_kv_unified();

        const float kq_scale = hparams.f_attention_scale == 0.0f ? 1.0f/sqrtf(float(n_embd_head)) : hparams.f_attention_scale;

        struct ggml_tensor * bskcn_1;
        struct ggml_tensor * bskcn_2;

        for (int il = 0; il < n_layer; ++il) {
            struct ggml_tensor * inpSA = inpL;

            if (hparams.n_bskcn(0, il)) {
                bskcn_1 = inpSA;
            }

            if (hparams.n_bskcn(1, il)) {
                bskcn_2 = inpSA;
            }

            if (hparams.n_bskcn(2, il)) {
                inpSA = ggml_add(
                   ctx0,
                   ggml_mul(ctx0, bskcn_1, ggml_view_1d(ctx0, model.layers[il].bskcn_tv, 1, 0)),
                   ggml_mul(ctx0, inpSA, ggml_view_1d(ctx0, model.layers[il].bskcn_tv, 1, ggml_element_size(model.layers[il].bskcn_tv))));
            }

            if (hparams.n_bskcn(3, il)) {
                inpSA = ggml_add(
                   ctx0,
                   ggml_mul(ctx0, bskcn_2, ggml_view_1d(ctx0, model.layers[il].bskcn_tv, 1, 0)),
                   ggml_mul(ctx0, inpSA, ggml_view_1d(ctx0, model.layers[il].bskcn_tv, 1, ggml_element_size(model.layers[il].bskcn_tv))));
            }

            // norm
            cur = build_norm(inpL,
                    model.layers[il].attn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "attn_norm", il);

            // self-attention
            {
                // rope freq factors for llama3; may return nullptr for llama2 and other models
                ggml_tensor * rope_factors = model.get_rope_factors(cparams, il);

                // compute Q and K and RoPE them
                ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
                cb(Qcur, "Qcur", il);
                if (model.layers[il].bq) {
                    Qcur = ggml_add(ctx0, Qcur, model.layers[il].bq);
                    cb(Qcur, "Qcur", il);
                }

                ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
                cb(Kcur, "Kcur", il);
                if (model.layers[il].bk) {
                    Kcur = ggml_add(ctx0, Kcur, model.layers[il].bk);
                    cb(Kcur, "Kcur", il);
                }

                ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
                cb(Vcur, "Vcur", il);
                if (model.layers[il].bv) {
                    Vcur = ggml_add(ctx0, Vcur, model.layers[il].bv);
                    cb(Vcur, "Vcur", il);
                }

                Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
                Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
                Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

                Qcur = ggml_rope_ext(
                        ctx0, Qcur, inp_pos, rope_factors,
                        n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow
                        );

                Kcur = ggml_rope_ext(
                        ctx0, Kcur, inp_pos, rope_factors,
                        n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow
                        );

                cb(Qcur, "Qcur", il);
                cb(Kcur, "Kcur", il);
                cb(Vcur, "Vcur", il);

                cur = build_attn(inp_attn,
                        model.layers[il].wo, model.layers[il].bo,
                        Qcur, Kcur, Vcur, nullptr, nullptr, kq_scale, il);
                cb(cur, "attn_out", il);
            }

            if (il == n_layer - 1) {
                // skip computing output for unused tokens
                ggml_tensor * inp_out_ids = build_inp_out_ids();
                cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
                inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
            }

            ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
            cb(ffn_inp, "ffn_inp", il);

            // feed-forward network
            cur = build_norm(ffn_inp,
                    model.layers[il].ffn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "ffn_norm", il);

            cur = build_ffn(cur,
                    model.layers[il].ffn_up,   model.layers[il].ffn_up_b,   NULL,
                    model.layers[il].ffn_gate, model.layers[il].ffn_gate_b, NULL,
                    model.layers[il].ffn_down, model.layers[il].ffn_down_b, NULL,
                    NULL,
                    LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(cur, "ffn_out", il);

            cur = ggml_add(ctx0, cur, ffn_inp);
            cb(cur, "ffn_out", il);

            cur = build_cvec(cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        cur = build_norm(cur,
                model.output_norm, NULL,
                LLM_NORM_RMS, -1);

        cb(cur, "result_norm", -1);
        res->t_embd = cur;

        // lm_head
        cur = build_lora_mm(model.output, cur);

        cb(cur, "result_output", -1);
        res->t_logits = cur;

        ggml_build_forward_expand(gf, cur);
    }
};

// ref: https://github.com/facebookresearch/chameleon
// based on the original build_llama() function, changes:
//   * qk-norm
//   * swin-norm
//   * removed bias
//   * removed MoE
struct llm_build_chameleon : public llm_graph_context {
    llm_build_chameleon(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
        const int64_t n_embd_head = hparams.n_embd_head_v;

        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
        GGML_ASSERT(n_embd_head == hparams.n_rot);

        ggml_tensor * cur;
        ggml_tensor * inpL;

        inpL = build_inp_embd(model.tok_embd);

        // inp_pos - contains the positions
        ggml_tensor * inp_pos = build_inp_pos();

        auto * inp_attn = build_attn_inp_kv_unified();

        ggml_tensor * inp_out_ids = build_inp_out_ids();

        for (int il = 0; il < n_layer; ++il) {
            ggml_tensor * inpSA = inpL;

            // norm
            if (hparams.swin_norm) {
                cur = inpL;
            } else {
                cur = build_norm(inpL,
                        model.layers[il].attn_norm, NULL,
                        LLM_NORM_RMS, il);
                cb(cur, "attn_norm", il);
            }

            // self-attention
            {
                // compute Q and K and RoPE them
                ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
                cb(Qcur, "Qcur", il);

                ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
                cb(Kcur, "Kcur", il);

                ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
                cb(Vcur, "Vcur", il);

                if (model.layers[il].attn_q_norm) {
                    Qcur = ggml_view_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens,
                            ggml_element_size(Qcur) * n_embd_head,
                            ggml_element_size(Qcur) * n_embd_head * n_head,
                            0);
                    cb(Qcur, "Qcur", il);

                    Qcur = build_norm(Qcur,
                            model.layers[il].attn_q_norm,
                            model.layers[il].attn_q_norm_b,
                            LLM_NORM, il);
                    cb(Qcur, "Qcur", il);
                }

                if (model.layers[il].attn_k_norm) {
                    Kcur = ggml_view_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens,
                            ggml_element_size(Kcur) * n_embd_head,
                            ggml_element_size(Kcur) * n_embd_head * n_head_kv,
                            0);
                    cb(Kcur, "Kcur", il);

                    Kcur = build_norm(Kcur,
                            model.layers[il].attn_k_norm,
                            model.layers[il].attn_k_norm_b,
                            LLM_NORM, il);
                    cb(Kcur, "Kcur", il);
                }

                Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
                Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
                Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

                Qcur = ggml_rope_ext(
                        ctx0, Qcur, inp_pos, nullptr,
                        n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow
                        );

                Kcur = ggml_rope_ext(
                        ctx0, Kcur, inp_pos, nullptr,
                        n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow
                        );

                cb(Qcur, "Qcur", il);
                cb(Kcur, "Kcur", il);
                cb(Vcur, "Vcur", il);

                cur = build_attn(inp_attn,
                        model.layers[il].wo, nullptr,
                        Qcur, Kcur, Vcur, nullptr, nullptr, 1.0f/sqrtf(float(n_embd_head)), il);
            }

            if (il == n_layer - 1 && inp_out_ids) {
                cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
                inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
            }

            if (hparams.swin_norm) {
                cur = build_norm(cur,
                        model.layers[il].attn_norm, NULL,
                        LLM_NORM_RMS, il);
            }

            ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
            cb(ffn_inp, "ffn_inp", il);

            // feed-forward network
            if (!hparams.swin_norm) {
                cur = build_norm(ffn_inp,
                        model.layers[il].ffn_norm, NULL,
                        LLM_NORM_RMS, il);
                cb(cur, "ffn_norm", il);
            }

            cur = build_ffn(cur,
                    model.layers[il].ffn_up,   NULL, NULL,
                    model.layers[il].ffn_gate, NULL, NULL,
                    model.layers[il].ffn_down, NULL, NULL,
                    NULL,
                    LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(cur, "ffn_out", il);

            if (hparams.swin_norm) {
                cur = build_norm(cur,
                        model.layers[il].ffn_norm, NULL,
                        LLM_NORM_RMS, il);
                cb(cur, "ffn_norm", il);
            }

            cur = ggml_add(ctx0, cur, ffn_inp);
            cb(cur, "ffn_out", il);

            cur = build_cvec(cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        cur = build_norm(cur,
                model.output_norm, NULL,
                LLM_NORM_RMS, -1);

        cb(cur, "result_norm", -1);
        res->t_embd = cur;

        // lm_head
        cur = build_lora_mm(model.output, cur);
        cb(cur, "result_output_with_img_logits", -1);

        // TODO: this suppresses the output of image tokens, which is required to enable text-only outputs.
        // Needs to be removed once image outputs are supported.
        int img_token_end_idx = 8196;
        int img_token_start_idx = 4;
        int num_img_tokens = img_token_end_idx - img_token_start_idx;
        // creates 1d tensor of size num_img_tokens and values -FLT_MAX,
        // which ensures that text token values are always at least larger than image token values
        ggml_tensor * img_logits = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, num_img_tokens);
        img_logits = ggml_clamp(ctx0, img_logits, -FLT_MAX, -FLT_MAX);
        cb(img_logits, "img_logits", -1);

        cur = ggml_set_1d(ctx0, cur, img_logits, ggml_element_size(cur) * img_token_start_idx);

        cb(cur, "result_output", -1);
        res->t_logits = cur;

        ggml_build_forward_expand(gf, cur);
    }
};

struct llm_build_wavtokenizer_dec : public llm_graph_context {
    llm_build_wavtokenizer_dec(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
        ggml_tensor * cur;
        ggml_tensor * inpL;

        inpL = build_inp_embd(model.tok_embd);

        cur = ggml_cont(ctx0, ggml_transpose(ctx0, inpL));

        cur = ggml_conv_1d_ph(ctx0, model.conv1d, cur, 1, 1);
        cur = ggml_add(ctx0, cur, model.conv1d_b);

        // posnet
        for (uint32_t il = 0; il < hparams.posnet.n_layer; ++il) {
            const auto & layer = model.layers[il].posnet;

            inpL = cur;

            switch (il) {
                case 0:
                case 1:
                case 3:
                case 4:
                    {
                        cur = build_norm(cur,
                                layer.norm1,
                                layer.norm1_b,
                                LLM_NORM_GROUP, 0);

                        cur = ggml_mul(ctx0, ggml_sigmoid(ctx0, cur), cur);

                        cur = ggml_conv_1d_ph(ctx0, layer.conv1, cur, 1, 1);
                        cur = ggml_add(ctx0, cur, layer.conv1_b);

                        cur = build_norm(cur,
                                layer.norm2,
                                layer.norm2_b,
                                LLM_NORM_GROUP, 0);

                        cur = ggml_mul(ctx0, ggml_sigmoid(ctx0, cur), cur);

                        cur = ggml_conv_1d_ph(ctx0, layer.conv2, cur, 1, 1);
                        cur = ggml_add(ctx0, cur, layer.conv2_b);

                        cur = ggml_add(ctx0, cur, inpL);
                    } break;
                case 2:
                    {
                        cur = build_norm(cur,
                                layer.attn_norm,
                                layer.attn_norm_b,
                                LLM_NORM_GROUP, 0);

                        ggml_tensor * q;
                        ggml_tensor * k;
                        ggml_tensor * v;

                        q = ggml_conv_1d_ph(ctx0, layer.attn_q, cur, 1, 1);
                        k = ggml_conv_1d_ph(ctx0, layer.attn_k, cur, 1, 1);
                        v = ggml_conv_1d_ph(ctx0, layer.attn_v, cur, 1, 1);

                        q = ggml_add(ctx0, q, layer.attn_q_b);
                        k = ggml_add(ctx0, k, layer.attn_k_b);
                        v = ggml_add(ctx0, v, layer.attn_v_b);

                        q = ggml_cont(ctx0, ggml_transpose(ctx0, q));
                        k = ggml_cont(ctx0, ggml_transpose(ctx0, k));

                        ggml_tensor * kq = ggml_mul_mat(ctx0, k, q);

                        kq = ggml_soft_max_ext(ctx0, kq, nullptr, 1.0f/sqrtf(float(hparams.posnet.n_embd)), 0.0f);

                        cur = ggml_mul_mat(ctx0, kq, v);

                        cur = ggml_conv_1d_ph(ctx0, layer.attn_o, cur, 1, 1);
                        cur = ggml_add(ctx0, cur, layer.attn_o_b);

                        cur = ggml_add(ctx0, cur, inpL);
                    } break;
                case 5:
                    {
                        cur = build_norm(cur,
                                layer.norm,
                                layer.norm_b,
                                LLM_NORM_GROUP, 0);
                    } break;
                default: GGML_ABORT("unknown posnet layer");
            };
        }

        cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur));

        cur = build_norm(cur,
                model.tok_norm,
                model.tok_norm_b,
                LLM_NORM, -1);

        cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur));

        inpL = cur;

        // convnext
        for (uint32_t il = 0; il < hparams.convnext.n_layer; ++il) {
            const auto & layer = model.layers[il].convnext;

            cur = inpL;

            cur = ggml_conv_1d_dw_ph(ctx0, layer.dw, cur, 1, 1);
            cur = ggml_add(ctx0, cur, layer.dw_b);

            cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur));

            cur = build_norm(cur,
                    layer.norm,
                    layer.norm_b,
                    LLM_NORM, -1);

            cur = build_ffn(cur,
                    layer.pw1, layer.pw1_b, NULL,
                    NULL,      NULL,        NULL,
                    layer.pw2, layer.pw2_b, NULL,
                    NULL,
                    LLM_FFN_GELU, LLM_FFN_SEQ, il);

            cur = ggml_mul(ctx0, cur, layer.gamma);

            cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur));

            inpL = ggml_add(ctx0, cur, inpL);
        }

        cur = inpL;

        cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur));

        cur = build_norm(cur,
                model.output_norm,
                model.output_norm_b,
                LLM_NORM, -1);

        // lm_head
        cur = build_lora_mm(model.output, cur);

        cur = ggml_add(ctx0, cur, model.output_b);

        cb(cur, "result_embd", -1);
        res->t_embd = cur;

        ggml_build_forward_expand(gf, cur);
    }
};

struct llm_build_plm : public llm_graph_context {
    llm_build_plm(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
        const float kq_scale = 1.0f/sqrtf(float(hparams.n_embd_head_k));

        const uint32_t n_embd_head_qk_rope = hparams.n_rot;
        const uint32_t n_embd_head_qk_nope = hparams.n_embd_head_k - hparams.n_rot;
        const uint32_t kv_lora_rank = hparams.n_lora_kv;

        ggml_tensor * cur;
        ggml_tensor * inpL;

        // {n_embd, n_tokens}
        inpL = build_inp_embd(model.tok_embd);

        // inp_pos - contains the positions
        ggml_tensor * inp_pos = build_inp_pos();

        auto * inp_attn = build_attn_inp_kv_unified();

        ggml_tensor * inp_out_ids = build_inp_out_ids();

        for (int il = 0; il < n_layer; ++il) {
            ggml_tensor * inpSA = inpL;

            // norm
            cur = build_norm(inpL,
                    model.layers[il].attn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "attn_norm", il);

            // self_attention
            {
                ggml_tensor * q = NULL;
                q = ggml_mul_mat(ctx0, model.layers[il].wq, cur);
                cb(q, "q", il);

                // split into {n_head * n_embd_head_qk_nope, n_tokens}
                ggml_tensor * q_nope = ggml_view_3d(ctx0, q, n_embd_head_qk_nope, n_head, n_tokens,
                        ggml_row_size(q->type, hparams.n_embd_head_k),
                        ggml_row_size(q->type, hparams.n_embd_head_k * n_head),
                        0);
                cb(q_nope, "q_nope", il);

                // and {n_head * n_embd_head_qk_rope, n_tokens}
                ggml_tensor * q_pe = ggml_view_3d(ctx0, q, n_embd_head_qk_rope, n_head, n_tokens,
                        ggml_row_size(q->type, hparams.n_embd_head_k),
                        ggml_row_size(q->type, hparams.n_embd_head_k * n_head),
                        ggml_row_size(q->type, n_embd_head_qk_nope));
                cb(q_pe, "q_pe", il);

                // {n_embd, kv_lora_rank + n_embd_head_qk_rope} * {n_embd, n_tokens} -> {kv_lora_rank + n_embd_head_qk_rope, n_tokens}
                ggml_tensor * kv_pe_compresseed = ggml_mul_mat(ctx0, model.layers[il].wkv_a_mqa, cur);
                cb(kv_pe_compresseed, "kv_pe_compresseed", il);

                // split into {kv_lora_rank, n_tokens}
                ggml_tensor * kv_compressed = ggml_view_2d(ctx0, kv_pe_compresseed, kv_lora_rank, n_tokens,
                        kv_pe_compresseed->nb[1],
                        0);
                cb(kv_compressed, "kv_compressed", il);

                // and {n_embd_head_qk_rope, n_tokens}
                ggml_tensor * k_pe = ggml_view_3d(ctx0, kv_pe_compresseed, n_embd_head_qk_rope, 1, n_tokens,
                        kv_pe_compresseed->nb[1],
                        kv_pe_compresseed->nb[1],
                        ggml_row_size(kv_pe_compresseed->type, kv_lora_rank));
                cb(k_pe, "k_pe", il);

                kv_compressed = build_norm(kv_compressed,
                        model.layers[il].attn_kv_a_norm, NULL,
                        LLM_NORM_RMS, il);
                cb(kv_compressed, "kv_compressed", il);

                // {kv_lora_rank, n_head * (n_embd_head_qk_nope + n_embd_head_v)} * {kv_lora_rank, n_tokens} -> {n_head * (n_embd_head_qk_nope + n_embd_head_v), n_tokens}
                ggml_tensor * kv = ggml_mul_mat(ctx0, model.layers[il].wkv_b, kv_compressed);
                cb(kv, "kv", il);

                // split into {n_head * n_embd_head_qk_nope, n_tokens}
                ggml_tensor * k_nope = ggml_view_3d(ctx0, kv, n_embd_head_qk_nope, n_head, n_tokens,
                        ggml_row_size(kv->type, n_embd_head_qk_nope + hparams.n_embd_head_v),
                        ggml_row_size(kv->type, n_head * (n_embd_head_qk_nope + hparams.n_embd_head_v)),
                        0);
                cb(k_nope, "k_nope", il);

                // and {n_head * n_embd_head_v, n_tokens}
                ggml_tensor * v_states = ggml_view_3d(ctx0, kv, hparams.n_embd_head_v, n_head, n_tokens,
                        ggml_row_size(kv->type, (n_embd_head_qk_nope + hparams.n_embd_head_v)),
                        ggml_row_size(kv->type, (n_embd_head_qk_nope + hparams.n_embd_head_v)*n_head),
                        ggml_row_size(kv->type, (n_embd_head_qk_nope)));
                cb(v_states, "v_states", il);

                v_states = ggml_cont(ctx0, v_states);
                cb(v_states, "v_states", il);

                v_states = ggml_view_2d(ctx0, v_states, hparams.n_embd_head_v * n_head, n_tokens,
                        ggml_row_size(kv->type, hparams.n_embd_head_v * n_head),
                        0);
                cb(v_states, "v_states", il);

                q_pe = ggml_rope_ext(
                        ctx0, q_pe, inp_pos, nullptr,
                        n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow
                        );
                cb(q_pe, "q_pe", il);

                // shared RoPE key
                k_pe = ggml_rope_ext(
                        ctx0, k_pe, inp_pos, nullptr,
                        n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow
                        );
                cb(k_pe, "k_pe", il);

                ggml_tensor * q_states = ggml_concat(ctx0, q_nope, q_pe, 0);
                cb(q_states, "q_states", il);

                ggml_tensor * k_states = ggml_concat(ctx0, k_nope, ggml_repeat(ctx0, k_pe, q_pe), 0);
                cb(k_states, "k_states", il);

                cur = build_attn(inp_attn,
                        model.layers[il].wo, NULL,
                        q_states, k_states, v_states, nullptr, nullptr, kq_scale, il);
            }

            if (il == n_layer - 1 && inp_out_ids) {
                cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
                inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
            }

            ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
            cb(ffn_inp, "ffn_inp", il);

            cur = build_norm(ffn_inp,
                    model.layers[il].ffn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "ffn_norm", il);

            cur = build_ffn(cur,
                    model.layers[il].ffn_up,   NULL, NULL,
                    NULL, NULL, NULL,
                    model.layers[il].ffn_down, NULL, NULL,
                    NULL,
                    LLM_FFN_RELU_SQR, LLM_FFN_SEQ, il);
            cb(cur, "ffn_out", il);

            cur = ggml_add(ctx0, cur, ffn_inp);

            cur = build_cvec(cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        cur = build_norm(cur,
                model.output_norm, NULL,
                LLM_NORM_RMS, -1);

        cb(cur, "result_norm", -1);
        res->t_embd = cur;

        cur = build_lora_mm(model.output, cur);

        cb(cur, "result_output", -1);
        res->t_logits = cur;

        ggml_build_forward_expand(gf, cur);
    }
};

struct llm_build_bailingmoe : public llm_graph_context {
    llm_build_bailingmoe(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
        ggml_tensor * cur;
        ggml_tensor * inpL;

        inpL = build_inp_embd(model.tok_embd);

        // inp_pos - contains the positions
        ggml_tensor * inp_pos = build_inp_pos();

        auto * inp_attn = build_attn_inp_kv_unified();

        ggml_tensor * inp_out_ids = build_inp_out_ids();

        for (int il = 0; il < n_layer; ++il) {
            ggml_tensor * inpSA = inpL;

            // norm
            cur = build_norm(inpL,
                    model.layers[il].attn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "attn_norm", il);

            // self-attention
            {
                // rope freq factors for llama3; may return nullptr for llama2 and other models
                ggml_tensor * rope_factors = model.get_rope_factors(cparams, il);

                // compute Q and K and RoPE them
                ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
                cb(Qcur, "Qcur", il);
                if (model.layers[il].bq) {
                    Qcur = ggml_add(ctx0, Qcur, model.layers[il].bq);
                    cb(Qcur, "Qcur", il);
                }

                ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
                cb(Kcur, "Kcur", il);
                if (model.layers[il].bk) {
                    Kcur = ggml_add(ctx0, Kcur, model.layers[il].bk);
                    cb(Kcur, "Kcur", il);
                }

                ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
                cb(Vcur, "Vcur", il);
                if (model.layers[il].bv) {
                    Vcur = ggml_add(ctx0, Vcur, model.layers[il].bv);
                    cb(Vcur, "Vcur", il);
                }

                Qcur = ggml_reshape_3d(ctx0, Qcur, n_rot, n_head,    n_tokens);
                Kcur = ggml_reshape_3d(ctx0, Kcur, n_rot, n_head_kv, n_tokens);
                Vcur = ggml_reshape_3d(ctx0, Vcur, n_rot, n_head_kv, n_tokens);

                Qcur = ggml_rope_ext(
                        ctx0, Qcur, inp_pos, rope_factors,
                        n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow
                        );

                Kcur = ggml_rope_ext(
                        ctx0, Kcur, inp_pos, rope_factors,
                        n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow
                        );

                cb(Qcur, "Qcur", il);
                cb(Kcur, "Kcur", il);
                cb(Vcur, "Vcur", il);

                cur = build_attn(inp_attn,
                        model.layers[il].wo, model.layers[il].bo,
                        Qcur, Kcur, Vcur, nullptr, nullptr, 1.0f/sqrtf(float(n_rot)), il);
            }

            if (il == n_layer - 1 && inp_out_ids) {
                cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
                inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
            }

            ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
            cb(ffn_inp, "ffn_inp", il);

            cur = build_norm(ffn_inp,
                    model.layers[il].ffn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "ffn_norm", il);

            ggml_tensor * moe_out =
                build_moe_ffn(cur,
                        model.layers[il].ffn_gate_inp,
                        model.layers[il].ffn_up_exps,
                        model.layers[il].ffn_gate_exps,
                        model.layers[il].ffn_down_exps,
                        nullptr,
                        n_expert, n_expert_used,
                        LLM_FFN_SILU, hparams.expert_weights_norm,
                        false, hparams.expert_weights_scale,
                        LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX,
                        il);
            cb(moe_out, "ffn_moe_out", il);

            // FFN shared expert
            {
                ggml_tensor * ffn_shexp = build_ffn(cur,
                        model.layers[il].ffn_up_shexp,   NULL, NULL,
                        model.layers[il].ffn_gate_shexp, NULL, NULL,
                        model.layers[il].ffn_down_shexp, NULL, NULL,
                        NULL,
                        LLM_FFN_SILU, LLM_FFN_PAR, il);
                cb(ffn_shexp, "ffn_shexp", il);

                cur = ggml_add(ctx0, moe_out, ffn_shexp);
                cb(cur, "ffn_out", il);
            }

            cur = ggml_add(ctx0, cur, ffn_inp);

            cur = build_cvec(cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        cur = build_norm(cur,
                model.output_norm, NULL,
                LLM_NORM_RMS, -1);

        cb(cur, "result_norm", -1);
        res->t_embd = cur;

        // lm_head
        cur = build_lora_mm(model.output, cur);

        cb(cur, "result_output", -1);
        res->t_logits = cur;

        ggml_build_forward_expand(gf, cur);
    }
};

struct llm_build_dots1 : public llm_graph_context {
    llm_build_dots1(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
        const int64_t n_embd_head = hparams.n_embd_head_v;

        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
        GGML_ASSERT(n_embd_head == hparams.n_rot);

        ggml_tensor * cur;
        ggml_tensor * inpL;

        inpL = build_inp_embd(model.tok_embd);

        // inp_pos - contains the positions
        ggml_tensor * inp_pos = build_inp_pos();

        auto * inp_attn = build_attn_inp_kv_unified();

        ggml_tensor * inp_out_ids = build_inp_out_ids();

        for (int il = 0; il < n_layer; ++il) {
            ggml_tensor * inpSA = inpL;

            // norm
            cur = build_norm(inpL,
                    model.layers[il].attn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "attn_norm", il);

            // self_attention
            {
                // compute Q and K and RoPE them
                ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
                cb(Qcur, "Qcur", il);

                ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
                cb(Kcur, "Kcur", il);

                ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
                cb(Vcur, "Vcur", il);

                Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
                Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
                Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

                Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, NULL, LLM_NORM_RMS, il);
                cb(Qcur, "Qcur_normed", il);

                Qcur = ggml_rope_ext(
                        ctx0, Qcur, inp_pos, nullptr,
                        n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow
                        );

                Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, NULL, LLM_NORM_RMS, il);
                cb(Kcur, "Kcur_normed", il);

                Kcur = ggml_rope_ext(
                        ctx0, Kcur, inp_pos, nullptr,
                        n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow
                        );

                cb(Qcur, "Qcur", il);
                cb(Kcur, "Kcur", il);
                cb(Vcur, "Vcur", il);

                cur = build_attn(inp_attn,
                        model.layers[il].wo, model.layers[il].bo,
                        Qcur, Kcur, Vcur, nullptr, nullptr, 1.0f/sqrtf(float(n_embd_head)), il);
            }

            if (il == n_layer - 1 && inp_out_ids) {
                cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
                inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
            }

            ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
            cb(ffn_inp, "ffn_inp", il);

            // MoE branch
            cur = build_norm(ffn_inp,
                    model.layers[il].ffn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "ffn_norm", il);

            if ((uint32_t) il < hparams.n_layer_dense_lead) {
                cur = build_ffn(cur,
                        model.layers[il].ffn_up,   NULL, NULL,
                        model.layers[il].ffn_gate, NULL, NULL,
                        model.layers[il].ffn_down, NULL, NULL,
                        NULL,
                        LLM_FFN_SILU, LLM_FFN_PAR, il);
                cb(cur, "ffn_out", il);
            } else {
                ggml_tensor * moe_out =
                    build_moe_ffn(cur,
                            model.layers[il].ffn_gate_inp,
                            model.layers[il].ffn_up_exps,
                            model.layers[il].ffn_gate_exps,
                            model.layers[il].ffn_down_exps,
                            model.layers[il].ffn_exp_probs_b,
                            n_expert, n_expert_used,
                            LLM_FFN_SILU, hparams.expert_weights_norm,
                            true, hparams.expert_weights_scale,
                            (llama_expert_gating_func_type) hparams.expert_gating_func,
                            il);
                cb(moe_out, "ffn_moe_out", il);

                {
                    ggml_tensor * ffn_shexp = build_ffn(cur,
                            model.layers[il].ffn_up_shexp,   NULL, NULL,
                            model.layers[il].ffn_gate_shexp, NULL, NULL,
                            model.layers[il].ffn_down_shexp, NULL, NULL,
                            NULL,
                            LLM_FFN_SILU, LLM_FFN_PAR, il);
                    cb(ffn_shexp, "ffn_shexp", il);

                    cur = ggml_add(ctx0, moe_out, ffn_shexp);
                    cb(cur, "ffn_out", il);
                }
            }

            cur = ggml_add(ctx0, cur, ffn_inp);

            cur = build_cvec(cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        cur = build_norm(cur,
                model.output_norm, NULL,
                LLM_NORM_RMS, -1);

        cb(cur, "result_norm", -1);
        res->t_embd = cur;

        // lm_head
        cur = build_lora_mm(model.output, cur);

        cb(cur, "result_output", -1);
        res->t_logits = cur;

        ggml_build_forward_expand(gf, cur);
    }
};

struct llm_build_ernie4_5 : public llm_graph_context {
    llm_build_ernie4_5(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
        const int64_t n_embd_head = hparams.n_embd_head_v;

        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
        GGML_ASSERT(n_embd_head == hparams.n_rot);

        ggml_tensor * cur;
        ggml_tensor * inpL;

        inpL = build_inp_embd(model.tok_embd);

        // inp_pos - contains the positions
        ggml_tensor * inp_pos = build_inp_pos();

        auto * inp_attn = build_attn_inp_kv_unified();

        for (int il = 0; il < n_layer; ++il) {
            ggml_tensor * inpSA = inpL;

            // norm
            {
                cur = build_norm(inpL,
                        model.layers[il].attn_norm, NULL,
                        LLM_NORM_RMS, il);
                cb(cur, "attn_norm", il);
            }

            // self-attention
            {
                ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
                cb(Qcur, "Qcur", il);
                if (model.layers[il].bq) {
                    Qcur = ggml_add(ctx0, Qcur, model.layers[il].bq);
                    cb(Qcur, "Qcur", il);
                }

                ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
                cb(Kcur, "Kcur", il);
                if (model.layers[il].bk) {
                    Kcur = ggml_add(ctx0, Kcur, model.layers[il].bk);
                    cb(Kcur, "Kcur", il);
                }

                ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
                cb(Vcur, "Vcur", il);
                if (model.layers[il].bv) {
                    Vcur = ggml_add(ctx0, Vcur, model.layers[il].bv);
                    cb(Vcur, "Vcur", il);
                }

                Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
                Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
                Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

                Qcur = ggml_rope_ext(
                        ctx0, Qcur, inp_pos, nullptr,
                        n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow
                        );

                Kcur = ggml_rope_ext(
                        ctx0, Kcur, inp_pos, nullptr,
                        n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow
                        );

                cb(Qcur, "Qcur", il);
                cb(Kcur, "Kcur", il);
                cb(Vcur, "Vcur", il);

                cur = build_attn(inp_attn,
                        model.layers[il].wo, NULL,
                        Qcur, Kcur, Vcur, nullptr, nullptr, 1.0f/sqrtf(float(n_embd_head)), il);
            }

            if (il == n_layer - 1) {
                // skip computing output for unused tokens
                ggml_tensor * inp_out_ids = build_inp_out_ids();
                cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
                inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
            }

            ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
            cb(ffn_inp, "ffn_inp", il);

            // feed-forward network
            {
                cur = build_norm(ffn_inp,
                        model.layers[il].ffn_norm, NULL,
                        LLM_NORM_RMS, il);
                cb(cur, "ffn_norm", il);

                cur = build_ffn(cur,
                        model.layers[il].ffn_up,   NULL, NULL,
                        model.layers[il].ffn_gate, NULL, NULL,
                        model.layers[il].ffn_down, NULL, NULL,
                        NULL,
                        LLM_FFN_SILU, LLM_FFN_PAR, il);
                cb(cur, "ffn_out", il);
            }

            cur = ggml_add(ctx0, cur, ffn_inp);

            cur = build_cvec(cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        cur = build_norm(cur,
                model.output_norm, NULL,
                LLM_NORM_RMS, -1);

        cb(cur, "result_norm", -1);
        res->t_embd = cur;

        // lm_head
        cur = build_lora_mm(model.output, cur);

        cb(cur, "result_output", -1);
        res->t_logits = cur;

        ggml_build_forward_expand(gf, cur);
    }
};

struct llm_build_ernie4_5_moe : public llm_graph_context {
    llm_build_ernie4_5_moe(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
        const int64_t n_embd_head = hparams.n_embd_head_v;

        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
        GGML_ASSERT(n_embd_head == hparams.n_rot);

        ggml_tensor * cur;
        ggml_tensor * inpL;

        inpL = build_inp_embd(model.tok_embd);

        // inp_pos - contains the positions
        ggml_tensor * inp_pos = build_inp_pos();

        auto * inp_attn = build_attn_inp_kv_unified();

        ggml_tensor * inp_out_ids = build_inp_out_ids();

        GGML_ASSERT(hparams.n_moe_layer_step > 0 && "Ernie 4.5 MoE requires n_moe_layer_step > 0");
        for (int il = 0; il < n_layer; ++il) {
            ggml_tensor * inpSA = inpL;
            // norm
            {
                cur = build_norm(inpL,
                        model.layers[il].attn_norm, NULL,
                        LLM_NORM_RMS, il);
                cb(cur, "attn_norm", il);
            }

            // self-attention
            {
                // compute Q and K and RoPE them
                ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
                cb(Qcur, "Qcur", il);
                if (model.layers[il].bq) {
                    Qcur = ggml_add(ctx0, Qcur, model.layers[il].bq);
                    cb(Qcur, "Qcur", il);
                }

                ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
                cb(Kcur, "Kcur", il);
                if (model.layers[il].bk) {
                    Kcur = ggml_add(ctx0, Kcur, model.layers[il].bk);
                    cb(Kcur, "Kcur", il);
                }

                ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
                cb(Vcur, "Vcur", il);
                if (model.layers[il].bv) {
                    Vcur = ggml_add(ctx0, Vcur, model.layers[il].bv);
                    cb(Vcur, "Vcur", il);
                }

                Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
                Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
                Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

                Qcur = ggml_rope_ext(
                        ctx0, Qcur, inp_pos, nullptr,
                        n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow
                        );

                Kcur = ggml_rope_ext(
                        ctx0, Kcur, inp_pos, nullptr,
                        n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow
                        );

                cb(Qcur, "Qcur", il);
                cb(Kcur, "Kcur", il);
                cb(Vcur, "Vcur", il);

                cur = build_attn(inp_attn,
                        model.layers[il].wo, NULL,
                        Qcur, Kcur, Vcur, nullptr, nullptr, 1.0f/sqrtf(float(n_embd_head)), il);
                cb(cur, "attn_out", il);
            }

            if (il == n_layer - 1 && inp_out_ids) {
                cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
                inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
            }

            ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
            cb(ffn_inp, "ffn_inp", il);

            // feed-forward network
            bool is_moe_layer = static_cast<uint32_t>(il) >= hparams.n_layer_dense_lead && (il + 1) % hparams.n_moe_layer_step == 0;

            if (!is_moe_layer) {
                cur = build_norm(ffn_inp,
                        model.layers[il].ffn_norm, NULL,
                        LLM_NORM_RMS, il);
                cb(cur, "ffn_norm", il);

                cur = build_ffn(cur,
                        model.layers[il].ffn_up,   NULL, NULL,
                        model.layers[il].ffn_gate, NULL, NULL,
                        model.layers[il].ffn_down, NULL, NULL,
                        NULL,
                        LLM_FFN_SILU, LLM_FFN_PAR, il);
                cb(cur, "ffn_out", il);
            } else {
                // MoE branch
                cur = build_norm(ffn_inp,
                        model.layers[il].ffn_norm, NULL,
                        LLM_NORM_RMS, il);
                cb(cur, "ffn_norm", il);

                ggml_tensor * moe_out = build_moe_ffn(cur,
                        model.layers[il].ffn_gate_inp,
                        model.layers[il].ffn_up_exps,
                        model.layers[il].ffn_gate_exps,
                        model.layers[il].ffn_down_exps,
                        model.layers[il].ffn_exp_probs_b,
                        n_expert, n_expert_used,
                        LLM_FFN_SILU, true,
                        false, 0.0,
                        LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX,
                        il);
                cb(moe_out, "ffn_moe_out", il);

                // Shared expert (if present)
                if (hparams.n_ff_shexp > 0) {
                    ggml_tensor * ffn_shexp = build_ffn(cur,
                        model.layers[il].ffn_up_shexp,   NULL, NULL,
                        model.layers[il].ffn_gate_shexp, NULL, NULL,
                        model.layers[il].ffn_down_shexp, NULL, NULL,
                        NULL,
                        LLM_FFN_SILU, LLM_FFN_PAR, il);
                    cb(ffn_shexp, "ffn_shexp", il);

                    cur = ggml_add(ctx0, moe_out, ffn_shexp);
                } else {
                    cur = moe_out;
                }
                cb(cur, "ffn_out", il);
            }

            cur = ggml_add(ctx0, cur, ffn_inp);
            cb(cur, "ffn_out", il);

            cur = build_cvec(cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        cur = build_norm(cur,
                model.output_norm, NULL,
                LLM_NORM_RMS, -1);

        cb(cur, "result_norm", -1);
        res->t_embd = cur;

        // lm_head
        cur = build_lora_mm(model.output, cur);

        cb(cur, "result_output", -1);
        res->t_logits = cur;

        ggml_build_forward_expand(gf, cur);
    }
};

struct llm_build_falcon_h1 : public llm_graph_context_mamba {
    llm_build_falcon_h1(const llama_model & model, const llm_graph_params & params) : llm_graph_context_mamba(params) {
        const int64_t n_embd_head = hparams.n_embd_head_v;

        ggml_tensor * cur;
        ggml_tensor * inpL;

        inpL = build_inp_embd(model.tok_embd);

        // inp_pos - contains the positions
        ggml_tensor * inp_pos = build_inp_pos();

        // Build the inputs in the recurrent & kv cache
        auto * inp = build_inp_mem_hybrid();

        const float kq_scale = hparams.f_attention_scale == 0.0f ? 1.0f/sqrtf(float(n_embd_head)) : hparams.f_attention_scale;

        ggml_tensor * inp_out_ids = build_inp_out_ids();

        for (int il = 0; il < n_layer; ++il) {
            ggml_tensor * inpSA = inpL;

            cur = build_norm(inpL,
                    model.layers[il].attn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "attn_norm", il);

            // self-attention
            ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
            cb(Qcur, "Qcur", il);

            ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
            cb(Kcur, "Kcur", il);

            ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
            cb(Vcur, "Vcur", il);

            Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
            Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);

            Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

            Qcur = ggml_rope_ext(
                    ctx0, Qcur, inp_pos, nullptr,
                    n_rot, hparams.rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow);

            Kcur = ggml_rope_ext(
                    ctx0, Kcur, inp_pos, nullptr,
                    n_rot, hparams.rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );

            cb(Qcur, "Qcur-post-rope", il);
            cb(Kcur, "Kcur-post-rope", il);
            cb(Vcur, "Vcur-post-rope", il);

            ggml_tensor * attn_out = build_attn(inp->get_attn(),
                    model.layers[il].wo, NULL,
                    Qcur, Kcur, Vcur, nullptr, nullptr, kq_scale, il);
            cb(attn_out, "attn_out", il);

            cur = build_norm(inpL,
                model.layers[il].attn_norm, NULL,
                LLM_NORM_RMS, il);
            // Mamba2 layer
            cb(cur, "ssm_in", il);

            ggml_tensor * ssm_out = build_mamba2_layer(inp->get_recr(), cur, model, ubatch, il);
            cb(ssm_out, "ssm_out", il);

            // // Aggregation
            cur = ggml_add(ctx0, attn_out, ssm_out);
            inpSA = ggml_add(ctx0, cur, inpSA);
            cb(cur, "layer_out", il);

            if (il == n_layer - 1 && inp_out_ids) {
                cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
                inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
            }

            ggml_tensor * ffn_inp = inpSA;
            cb(ffn_inp, "ffn_inp", il);

            // feed-forward network
            cur = build_norm(ffn_inp,
                    model.layers[il].ffn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "ffn_norm", il);

            cur = build_ffn(cur,
                    model.layers[il].ffn_up,   model.layers[il].ffn_up_b, NULL,
                    model.layers[il].ffn_gate, model.layers[il].ffn_gate_b, NULL,
                    model.layers[il].ffn_down, model.layers[il].ffn_down_b, NULL,
                    NULL,
                    LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(cur, "ffn_out", il);

            cur = ggml_add(ctx0, cur, inpSA);

            cur = build_cvec(cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        cur = build_norm(cur,
                model.output_norm, NULL,
                LLM_NORM_RMS, -1);

        cb(cur, "result_norm", -1);
        res->t_embd = cur;

        // lm_head
        cur = build_lora_mm(model.output, cur);

        cb(cur, "result_output", -1);
        res->t_logits = cur;

        ggml_build_forward_expand(gf, cur);
    }
};

struct llm_build_plamo2 : public llm_graph_context_mamba {
    llm_build_plamo2(const llama_model & model, const llm_graph_params & params) : llm_graph_context_mamba(params) {
        ggml_tensor * cur;
        ggml_tensor * inpL;

        // {n_embd, n_tokens}
        inpL = build_inp_embd(model.tok_embd);
        cb(inpL, "embedding_output", -1);

        ggml_tensor * inp_pos = build_inp_pos();

        auto * inp_hybrid = build_inp_mem_hybrid();

        ggml_tensor * inp_out_ids = build_inp_out_ids();

        for (int il = 0; il < n_layer; ++il) {
            ggml_tensor * residual = inpL;

            // ggml_graph_add_node(gf, model.layers[il].attn_norm);
            // cb(model.layers[il].attn_norm, "attn_norm", il);

            // pre_mixer_norm
            cur = build_norm(inpL, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, il);

            // check if this layer is Mamba or Attention
            bool is_mamba_layer = hparams.is_recurrent(il);

            if (is_mamba_layer) {
                // PLaMo-2 Mamba layer
                cur = build_plamo2_mamba_layer(inp_hybrid->get_recr(), cur, model, ubatch, il);
            } else {
                // PLaMo-2 Attention layer
                cur = build_plamo2_attn_layer(inp_hybrid->get_attn(), inp_pos, cur, model, il);
            }

            // post_mixer_norm
            cur = build_norm(cur, model.layers[il].attn_post_norm, NULL, LLM_NORM_RMS, il);
            cb(cur, "attn_post_norm", il);

            // residual connection
            cur = ggml_add(ctx0, cur, residual);
            cb(cur, "attn_residual", il);
            residual = cur;

            // pre-ffn norm
            cur = build_norm(cur, model.layers[il].ffn_norm, NULL, LLM_NORM_RMS, il);
            cb(cur, "ffn_pre_norm", il);

            // feed-forward network
            cur = build_ffn(cur,
                    model.layers[il].ffn_up,   NULL, NULL,
                    NULL,                      NULL, NULL,
                    model.layers[il].ffn_down, NULL, NULL,
                    NULL,
                    LLM_FFN_SWIGLU, LLM_FFN_SEQ, il);
            cb(cur, "ffn_out", il);

            // post ffn norm
            cur = build_norm(cur, model.layers[il].ffn_post_norm, NULL, LLM_NORM_RMS, il);
            cb(cur, "ffn_post_norm", il);

            if (il == n_layer - 1 && inp_out_ids) {
                cur  = ggml_get_rows(ctx0,  cur, inp_out_ids);
                residual = ggml_get_rows(ctx0, residual, inp_out_ids);
            }

            // residual connection
            cur = ggml_add(ctx0, cur, residual);
            cb(cur, "ffn_residual", il);

            inpL = cur;
        }

        cur = inpL;

        // final norm
        cur = build_norm(cur, model.output_norm, NULL, LLM_NORM_RMS, -1);
        cb(cur, "result_norm", -1);

        // lm_head
        cur = build_lora_mm(model.output, cur);
        cb(cur, "result_output", -1);

        // Explicitly mark as output tensor to ensure proper backend assignment
        ggml_set_output(cur);

        res->t_logits = cur;

        ggml_build_forward_expand(gf, cur);
    }

private:
    ggml_tensor * build_plamo2_attn_layer(
            llm_graph_input_attn_kv_unified * inp,
            ggml_tensor * inp_pos,
            ggml_tensor * cur,
            const llama_model & model,
            int il) {

        // self-attention
        {
            // PLaMo-2 uses combined QKV tensor
            ggml_tensor * qkv = build_lora_mm(model.layers[il].wqkv, cur);
            cb(qkv, "wqkv", il);

            // split QKV tensor into Q, K, V
            const int64_t n_embd_head_q = hparams.n_embd_head_k;
            const int64_t n_embd_head_k = hparams.n_embd_head_k;
            const int64_t n_embd_head_v = hparams.n_embd_head_v;
            int32_t n_head_kv = hparams.n_head_kv(il);

            const int64_t q_offset = 0;
            const int64_t k_offset = n_embd_head_q * n_head;
            const int64_t v_offset = k_offset + n_embd_head_k * n_head_kv;

            ggml_tensor * Qcur = ggml_view_3d(ctx0, qkv, n_embd_head_q, n_head, n_tokens, n_embd_head_q * sizeof(float), qkv->nb[1], q_offset * ggml_element_size(qkv));
            ggml_tensor * Kcur = ggml_view_3d(ctx0, qkv, n_embd_head_k, n_head_kv, n_tokens, n_embd_head_k * sizeof(float), qkv->nb[1], k_offset * ggml_element_size(qkv));
            ggml_tensor * Vcur = ggml_cont(ctx0, ggml_view_2d(ctx0, qkv, n_embd_head_v * n_head_kv, n_tokens, qkv->nb[1], v_offset * ggml_element_size(qkv)));

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head_v, n_head_kv, n_tokens);

            Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, NULL, LLM_NORM_RMS, il);
            cb(Qcur, "Qcur_normed", il);

            Qcur = ggml_rope_ext(
                    ctx0, Qcur, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );

            Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, NULL, LLM_NORM_RMS, il);
            cb(Kcur, "Kcur_normed", il);

            Kcur = ggml_rope_ext(
                    ctx0, Kcur, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );

            cur = build_attn(inp, model.layers[il].wo, NULL, Qcur, Kcur, Vcur, NULL, NULL, 1.0f/sqrtf(float(n_embd_head_v)), il);
        }

        cb(cur, "attn_out", il);

        return cur;
    }

    ggml_tensor * build_plamo2_mamba_layer(
         llm_graph_input_rs * inp,
               ggml_tensor * cur,
         const llama_model & model,
        const llama_ubatch & ubatch,
                       int   il) {

        const auto * mctx_cur = inp->mctx;

        const auto kv_head = mctx_cur->get_head();

        const int64_t d_conv   = hparams.ssm_d_conv;
        const int64_t d_inner  = hparams.ssm_d_inner;
        const int64_t d_state  = hparams.ssm_d_state;
        const int64_t n_heads  = hparams.ssm_dt_rank;
        const int64_t head_dim = d_inner / n_heads;
        const int64_t n_group  = hparams.ssm_n_group;
        const int64_t n_seqs   = ubatch.n_seqs;

        const int64_t n_seq_tokens = ubatch.n_seq_tokens;

        GGML_ASSERT(n_seqs != 0);
        GGML_ASSERT(ubatch.equal_seqs());
        GGML_ASSERT(ubatch.n_tokens == n_seq_tokens * n_seqs);

        ggml_tensor * conv_states_all = mctx_cur->get_r_l(il);
        ggml_tensor * ssm_states_all  = mctx_cur->get_s_l(il);

        ggml_tensor * conv = build_rs(inp, conv_states_all, hparams.n_embd_r(), n_seqs);
        conv = ggml_reshape_3d(ctx0, conv, d_conv - 1, d_inner + 2*n_group*d_state, n_seqs);

        // {n_embd, n_tokens} => {n_embd, n_seq_tokens, n_seqs}
        cur = ggml_reshape_3d(ctx0, cur, cur->ne[0], n_seq_tokens, n_seqs);

        // in_proj: {n_embd, 2*d_inner} @ {n_embd, n_seq_tokens, n_seqs} => {2*d_inner, n_seq_tokens, n_seqs}
        ggml_tensor * zx = build_lora_mm(model.layers[il].ssm_in, cur);
        cb(zx, "mamba_in_proj", il);
        // {8192, 5, 1, 1} -> {8192, 1, 5, 1}
        zx = ggml_permute(ctx0, zx, 0, 2, 1, 3);
        zx = ggml_cont(ctx0, zx);
        zx = ggml_reshape_4d(ctx0, zx, head_dim * 2, n_heads, n_seq_tokens, n_seqs);
        cb(zx, "mamba_in_proj_out", il);

        // split into z and x
        // => {head_dim * n_heads, n_seq_tokens, n_seqs}
        ggml_tensor * x = ggml_view_4d(ctx0, zx, head_dim, n_heads, n_seq_tokens, n_seqs, zx->nb[1], zx->nb[2], zx->nb[3], head_dim*ggml_element_size(zx));
        x = ggml_cont(ctx0, x);
        x = ggml_reshape_3d(ctx0, x, head_dim * n_heads, n_seq_tokens, n_seqs);
        // x = ggml_permute(ctx0, x, 0, 2, 1, 3);
        cb(x, "mamba_x_split", il);

        ggml_tensor * z = ggml_view_4d(ctx0, zx, head_dim, n_heads, n_seq_tokens, n_seqs, zx->nb[1], zx->nb[2], zx->nb[3], 0);
        cb(z, "mamba_z_split", il);

        // conv1d
        {
            // => {d_conv - 1 + n_seq_tokens, d_inner, n_seqs}
            ggml_tensor * conv_x = ggml_concat(ctx0, conv, ggml_transpose(ctx0, x), 0);
            cb(conv_x, "mamba_conv1d_input", il);

            // copy last (d_conv - 1) columns back into the state cache
            ggml_tensor * last_conv = ggml_view_3d(ctx0, conv_x, d_conv - 1, d_inner, n_seqs,
                    conv_x->nb[1], conv_x->nb[2], n_seq_tokens*(conv_x->nb[0]));

            ggml_build_forward_expand(gf,
                ggml_cpy(ctx0, last_conv,
                    ggml_view_1d(ctx0, conv_states_all,
                        (d_conv - 1)*(d_inner + 2*n_group*d_state)*(n_seqs),
                        kv_head*(d_conv - 1)*(d_inner + 2*n_group*d_state)*ggml_element_size(conv_states_all))));
            cb(conv_states_all, "mamba_conv1d_state", il);

            // 1D convolution
            x = ggml_ssm_conv(ctx0, conv_x, model.layers[il].ssm_conv1d);
            cb(x, "mamba_conv1d", il);

            x = ggml_silu(ctx0, x);
            cb(x, "mamba_conv1d_silu", il);
        }

        // SSM
        {
            // bcdt_proj: {d_inner, dt_rank + 2*d_state} @ {d_inner, n_seq_tokens, n_seqs} => {dt_rank + 2*d_state, n_seq_tokens, n_seqs}
            ggml_tensor * x_bcdt = build_lora_mm(model.layers[il].ssm_x, x);
            cb(x_bcdt, "mamba_bcdt_proj", il);

            // split into dt, B, C
            const int64_t dt_dim = std::max(64, int(hparams.n_embd / 16));
            ggml_tensor * B = ggml_view_3d(ctx0, x_bcdt, d_state, n_seq_tokens, n_seqs, x_bcdt->nb[1], x_bcdt->nb[2], 0);
            ggml_tensor * C  = ggml_view_3d(ctx0, x_bcdt, d_state, n_seq_tokens, n_seqs, x_bcdt->nb[1], x_bcdt->nb[2], ggml_element_size(x_bcdt)*d_state);
            ggml_tensor * dt  = ggml_view_3d(ctx0, x_bcdt, dt_dim, n_seq_tokens, n_seqs, x_bcdt->nb[1], x_bcdt->nb[2], ggml_element_size(x_bcdt)*(2*d_state));
            cb(B, "mamba_B_raw", il);
            cb(C, "mamba_C_raw", il);
            cb(dt, "mamba_dt_raw", il);

            // Apply RMS norm to dt, B, C (PLaMo-2 specific)
            B = build_norm(B, model.layers[il].ssm_b_norm, NULL, LLM_NORM_RMS, il);
            C = build_norm(C, model.layers[il].ssm_c_norm, NULL, LLM_NORM_RMS, il);
            dt = build_norm(dt, model.layers[il].ssm_dt_norm, NULL, LLM_NORM_RMS, il);
            cb(B, "mamba_B_normed", il);
            cb(C, "mamba_C_normed", il);
            cb(dt, "mamba_dt_normed", il);

            // dt_proj: {dt_rank, d_inner} @ {dt_rank, n_seq_tokens, n_seqs} => {d_inner, n_seq_tokens, n_seqs}
            dt = build_lora_mm(model.layers[il].ssm_dt, dt);
            dt = ggml_add(ctx0, dt, model.layers[il].ssm_dt_b);
            cb(dt, "mamba_dt_proj", il);

            ggml_tensor * A = ggml_reshape_2d(ctx0, model.layers[il].ssm_a, 1, n_heads);
            cb(A, "mamba_A", il);

            x = ggml_view_4d(ctx0, x, head_dim, n_heads, n_seq_tokens, n_seqs, head_dim * ggml_element_size(x), head_dim * n_heads * ggml_element_size(x), head_dim * n_heads * n_seq_tokens * ggml_element_size(x), 0);
            B = ggml_view_4d(ctx0, B, d_state, 1, n_seq_tokens, n_seqs, d_state * B->nb[0], B->nb[1], B->nb[2], 0);
            C = ggml_view_4d(ctx0, C, d_state, 1, n_seq_tokens, n_seqs, d_state * C->nb[0], C->nb[1], C->nb[2], 0);

            // use the states and the indices provided by build_recurrent_state
            // (this is necessary in order to properly use the states before they are overwritten,
            //  while avoiding to make unnecessary copies of the states)
            auto get_ssm_rows = [&](ggml_context * ctx, ggml_tensor * states, ggml_tensor * ids) {
                ggml_tensor * ssm = ggml_reshape_4d(ctx, states, d_state, head_dim, n_heads, mctx_cur->get_size());

                // Custom operator to optimize the parallel associative scan
                // as described in the Annex D of the Mamba paper.
                // => {d_inner, n_seq_tokens, n_seqs} and {d_state, d_inner, n_seqs}
                return ggml_ssm_scan(ctx, ssm, x, dt, A, B, C, ids);
            };

            ggml_tensor * y_ssm = build_rs(inp, ssm_states_all, hparams.n_embd_s(), ubatch.n_seqs, get_ssm_rows);
            cb(y_ssm, "mamba_ssm_scan", il);

            // store last states
            ggml_build_forward_expand(gf,
                ggml_cpy(ctx0,
                    ggml_view_1d(ctx0, y_ssm, n_heads*head_dim*d_state*n_seqs, n_heads*head_dim*n_seq_tokens*n_seqs*ggml_element_size(y_ssm)),
                    ggml_view_1d(ctx0, ssm_states_all, n_heads*head_dim*d_state*n_seqs, kv_head*n_seqs*n_heads*head_dim*d_state*ggml_element_size(ssm_states_all))));
            cb(ssm_states_all, "mamba_ssm_states", il);

            ggml_tensor * y = ggml_view_4d(ctx0, y_ssm, head_dim, n_heads, n_seq_tokens, n_seqs, head_dim * ggml_element_size(x), head_dim * n_heads * ggml_element_size(x), head_dim * n_heads * n_seq_tokens * ggml_element_size(x), 0);
            cb(y, "mamba_y_view", il);

            // Add D parameter and apply gating with z
            // {d_inner, n_seq_tokens, n_seqs} * {d_inner} => {d_inner, n_seq_tokens, n_seqs}
            ggml_tensor * D = ggml_reshape_2d(ctx0, model.layers[il].ssm_d, 1, n_heads);
            y = ggml_add(ctx0, y, ggml_mul(ctx0, x, D));
            cb(y, "mamba_y_add_d", il);

            y = ggml_swiglu_split(ctx0, ggml_cont(ctx0, z), y);
            cb(y, "mamba_y_swiglu_z", il);

            // out_proj: {d_inner, n_embd} @ {d_inner, n_seq_tokens, n_seqs} => {n_embd, n_seq_tokens, n_seqs}
            y = ggml_view_3d(ctx0, y, head_dim * n_heads, n_seq_tokens, n_seqs, y->nb[2], y->nb[3], 0);
            cur = build_lora_mm(model.layers[il].ssm_out, y);
            cb(cur, "mamba_out_proj", il);
        }

        // {n_embd, n_seq_tokens, n_seqs} => {n_embd, n_tokens}
        cur = ggml_reshape_2d(ctx0, cur, cur->ne[0], n_seq_tokens * n_seqs);
        cb(cur, "mamba_out", il);

        return cur;
    }
};

struct llm_build_arcee : public llm_graph_context {
    llm_build_arcee(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
        const int64_t n_embd_head = hparams.n_embd_head_v;

        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
        GGML_ASSERT(n_embd_head == hparams.n_rot);

        ggml_tensor * cur;
        ggml_tensor * inpL;

        inpL = build_inp_embd(model.tok_embd);

        // inp_pos - contains the positions
        ggml_tensor * inp_pos = build_inp_pos();

        auto * inp_attn = build_attn_inp_kv_unified();

        const float kq_scale = hparams.f_attention_scale == 0.0f ? 1.0f/sqrtf(float(n_embd_head)) : hparams.f_attention_scale;

        ggml_tensor * inp_out_ids = build_inp_out_ids();

        for (int il = 0; il < n_layer; ++il) {
            ggml_tensor * inpSA = inpL;

            // norm
            cur = build_norm(inpL,
                    model.layers[il].attn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "attn_norm", il);

            // self-attention
            {
                // rope freq factors for llama3; may return nullptr for llama2 and other models
                ggml_tensor * rope_factors = model.get_rope_factors(cparams, il);

                // compute Q and K and RoPE them
                ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
                cb(Qcur, "Qcur", il);
                if (model.layers[il].bq) {
                    Qcur = ggml_add(ctx0, Qcur, model.layers[il].bq);
                    cb(Qcur, "Qcur", il);
                }

                ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
                cb(Kcur, "Kcur", il);
                if (model.layers[il].bk) {
                    Kcur = ggml_add(ctx0, Kcur, model.layers[il].bk);
                    cb(Kcur, "Kcur", il);
                }

                ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
                cb(Vcur, "Vcur", il);
                if (model.layers[il].bv) {
                    Vcur = ggml_add(ctx0, Vcur, model.layers[il].bv);
                    cb(Vcur, "Vcur", il);
                }

                Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
                Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
                Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

                Qcur = ggml_rope_ext(
                        ctx0, Qcur, inp_pos, rope_factors,
                        n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow
                        );

                Kcur = ggml_rope_ext(
                        ctx0, Kcur, inp_pos, rope_factors,
                        n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow
                        );

                cb(Qcur, "Qcur", il);
                cb(Kcur, "Kcur", il);
                cb(Vcur, "Vcur", il);

                cur = build_attn(inp_attn,
                        model.layers[il].wo, model.layers[il].bo,
                        Qcur, Kcur, Vcur, nullptr, nullptr, kq_scale, il);
                cb(cur, "attn_out", il);
            }

            if (il == n_layer - 1 && inp_out_ids) {
                cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
                inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
            }

            ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
            cb(ffn_inp, "ffn_inp", il);

            // feed-forward network
            // ARCEE uses relu^2 instead of silu
            cur = build_norm(ffn_inp,
                    model.layers[il].ffn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "ffn_norm", il);

            cur = build_ffn(cur,
                    model.layers[il].ffn_up,   NULL, NULL,
                    NULL,                      NULL, NULL,
                    model.layers[il].ffn_down, NULL, NULL,
                    NULL,
                    LLM_FFN_RELU_SQR, LLM_FFN_SEQ, il);
            cb(cur, "ffn_out", il);

            cur = ggml_add(ctx0, cur, ffn_inp);
            cb(cur, "ffn_out", il);

            cur = build_cvec(cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        cur = build_norm(cur,
                model.output_norm, NULL,
                LLM_NORM_RMS, -1);

        cb(cur, "result_norm", -1);
        res->t_embd = cur;

        // lm_head
        cur = build_lora_mm(model.output, cur);

        cb(cur, "result_output", -1);
        res->t_logits = cur;

        ggml_build_forward_expand(gf, cur);
    }
};

struct llm_build_hunyuan_moe : public llm_graph_context {
    llm_build_hunyuan_moe(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
        const int64_t n_embd_head = hparams.n_embd_head_v;

        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
        GGML_ASSERT(n_embd_head == hparams.n_rot);

        ggml_tensor * cur;
        ggml_tensor * inpL;

        inpL = build_inp_embd(model.tok_embd);

        // inp_pos - contains the positions
        ggml_tensor * inp_pos = build_inp_pos();

        auto * inp_attn = build_attn_inp_kv_unified();

        const float kq_scale = 1.0f / sqrtf(float(n_embd_head));

        ggml_tensor * inp_out_ids = build_inp_out_ids();

        for (int il = 0; il < n_layer; ++il) {
            ggml_tensor * inpSA = inpL;

            // norm
            cur = build_norm(inpL,
                    model.layers[il].attn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "attn_norm", il);

            // self-attention
            {
                // rope freq factors for llama3; may return nullptr for llama2 and other models
                ggml_tensor * rope_factors = model.get_rope_factors(cparams, il);

                // compute Q and K and RoPE them
                ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
                cb(Qcur, "Qcur", il);
                if (model.layers[il].bq) {
                    Qcur = ggml_add(ctx0, Qcur, model.layers[il].bq);
                    cb(Qcur, "Qcur", il);
                }

                ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
                cb(Kcur, "Kcur", il);
                if (model.layers[il].bk) {
                    Kcur = ggml_add(ctx0, Kcur, model.layers[il].bk);
                    cb(Kcur, "Kcur", il);
                }

                ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
                cb(Vcur, "Vcur", il);
                if (model.layers[il].bv) {
                    Vcur = ggml_add(ctx0, Vcur, model.layers[il].bv);
                    cb(Vcur, "Vcur", il);
                }

                Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
                Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
                Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

                Qcur = ggml_rope_ext(
                        ctx0, Qcur, inp_pos, rope_factors,
                        n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow
                        );

                cb(Qcur, "Qcur", il);
                cb(Kcur, "Kcur", il);
                cb(Vcur, "Vcur", il);

                Kcur = ggml_rope_ext(
                        ctx0, Kcur, inp_pos, rope_factors,
                        n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow
                        );

                Kcur = build_norm(Kcur,
                        model.layers[il].attn_k_norm, nullptr,
                        LLM_NORM_RMS, il);
                cb(Kcur, "Kcur_norm", il);

                Qcur = build_norm(Qcur,
                        model.layers[il].attn_q_norm, nullptr,
                        LLM_NORM_RMS, il);
                cb(Qcur, "Qcur_norm", il);

                cur = build_attn(inp_attn,
                        model.layers[il].wo, model.layers[il].bo,
                        Qcur, Kcur, Vcur, nullptr, nullptr, kq_scale, il);
                cb(cur, "attn_out", il);
            }

            if (il == n_layer - 1 && inp_out_ids) {
                cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
                inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
            }

            ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
            cb(ffn_inp, "ffn_inp", il);

            cur = build_norm(ffn_inp,
                model.layers[il].ffn_norm, NULL,
                LLM_NORM_RMS, il);
            cb(cur, "ffn_norm", il);

            // feed-forward network (non-MoE)
            ggml_tensor * cur_mlp = build_ffn(cur,
                    model.layers[il].ffn_up_shexp,   NULL, NULL,
                    model.layers[il].ffn_gate_shexp, NULL, NULL,
                    model.layers[il].ffn_down_shexp, NULL, NULL,
                    NULL,
                    LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(cur_mlp, "ffn_mlp", il);

            // MoE branch
            ggml_tensor * cur_moe = build_moe_ffn(cur,
                    model.layers[il].ffn_gate_inp,
                    model.layers[il].ffn_up_exps,
                    model.layers[il].ffn_gate_exps,
                    model.layers[il].ffn_down_exps,
                    nullptr,
                    n_expert, n_expert_used,
                    LLM_FFN_SILU,
                    true, // norm_topk_prob
                    false,
                    0.0,
                    LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX,
                    il);
            cb(cur_moe, "ffn_moe_out", il);

            ggml_tensor * ffn_out = ggml_add(ctx0, cur_moe, cur_mlp);
            cb(ffn_out, "ffn_out", il);

            cur = ggml_add(ctx0, ffn_out, ffn_inp);

            cur = build_cvec(cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        cur = build_norm(cur,
                model.output_norm, NULL,
                LLM_NORM_RMS, -1);

        cb(cur, "result_norm", -1);
        res->t_embd = cur;

        // lm_head
        cur = build_lora_mm(model.output, cur);
        cb(cur, "result_output", -1);
        res->t_logits = cur;

        ggml_build_forward_expand(gf, cur);
    }
};

struct llm_build_hunyuan_dense : public llm_graph_context {
    llm_build_hunyuan_dense(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
        const int64_t n_embd_head = hparams.n_embd_head_v;

        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
        GGML_ASSERT(n_embd_head == hparams.n_rot);

        ggml_tensor * cur;
        ggml_tensor * inpL;

        inpL = build_inp_embd(model.tok_embd);

        // inp_pos - contains the positions
        ggml_tensor * inp_pos = build_inp_pos();

        auto * inp_attn = build_attn_inp_kv_unified();

        const float kq_scale = 1.0f / sqrtf(float(n_embd_head));

        ggml_tensor * inp_out_ids = build_inp_out_ids();

        for (int il = 0; il < n_layer; ++il) {
            ggml_tensor * inpSA = inpL;

            // norm
            cur = build_norm(inpL,
                    model.layers[il].attn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "attn_norm", il);
            // self-attention
            {
                // rope freq factors for llama3; may return nullptr for llama2 and other models
                ggml_tensor * rope_factors = model.get_rope_factors(cparams, il);

                // compute Q and K and RoPE them
                ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
                cb(Qcur, "Qcur", il);
                if (model.layers[il].bq) {
                    Qcur = ggml_add(ctx0, Qcur, model.layers[il].bq);
                    cb(Qcur, "Qcur", il);
                }

                ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
                cb(Kcur, "Kcur", il);
                if (model.layers[il].bk) {
                    Kcur = ggml_add(ctx0, Kcur, model.layers[il].bk);
                    cb(Kcur, "Kcur", il);
                }

                ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
                cb(Vcur, "Vcur", il);
                if (model.layers[il].bv) {
                    Vcur = ggml_add(ctx0, Vcur, model.layers[il].bv);
                    cb(Vcur, "Vcur", il);
                }

                Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
                Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
                Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

                Qcur = ggml_rope_ext(
                         ctx0, Qcur, inp_pos, rope_factors,
                         n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                         ext_factor, attn_factor, beta_fast, beta_slow
                         );

                cb(Qcur, "Qcur", il);
                cb(Kcur, "Kcur", il);
                cb(Vcur, "Vcur", il);

                Kcur = ggml_rope_ext(
                         ctx0, Kcur, inp_pos, rope_factors,
                         n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                         ext_factor, attn_factor, beta_fast, beta_slow
                         );

                Kcur = build_norm(Kcur,
                         model.layers[il].attn_k_norm, nullptr,
                         LLM_NORM_RMS, il);
                cb(Kcur, "Kcur_norm", il);

                Qcur = build_norm(Qcur,
                         model.layers[il].attn_q_norm, nullptr,
                         LLM_NORM_RMS, il);
                cb(Qcur, "Qcur_norm", il);

                cur = build_attn(inp_attn,
                        model.layers[il].wo, model.layers[il].bo,
                        Qcur, Kcur, Vcur, nullptr, nullptr, kq_scale, il);
                cb(cur, "attn_out", il);
            }

            if (il == n_layer - 1 && inp_out_ids) {
                cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
                inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
            }

            ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
            cb(ffn_inp, "ffn_inp", il);

            cur = build_norm(ffn_inp,
                    model.layers[il].ffn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "ffn_norm", il);
            // feed-forward network (non-MoE)
            ggml_tensor * cur_mlp = build_ffn(cur,
                        model.layers[il].ffn_up,   NULL, NULL,
                        model.layers[il].ffn_gate, NULL, NULL,
                        model.layers[il].ffn_down, NULL, NULL,
                        NULL,
                        LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(cur_mlp, "ffn_out", il);

            cur = ggml_add(ctx0, cur_mlp, ffn_inp);

            cur = build_cvec(cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }
        cur = inpL;

        cur = build_norm(cur,
                model.output_norm, NULL,
                LLM_NORM_RMS, -1);

        cb(cur, "result_norm", -1);
        res->t_embd = cur;
        // lm_head
        cur = build_lora_mm(model.output, cur);
        cb(cur, "result_output", -1);
        res->t_logits = cur;

        ggml_build_forward_expand(gf, cur);
    }
};

struct llm_build_smollm3 : public llm_graph_context {
    llm_build_smollm3(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
        const int64_t n_embd_head = hparams.n_embd_head_v;

        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
        GGML_ASSERT(n_embd_head == hparams.n_rot);

        ggml_tensor * cur;
        ggml_tensor * inpL;

        inpL = build_inp_embd(model.tok_embd);

        // inp_pos - contains the positions
        ggml_tensor * inp_pos = build_inp_pos();

        auto * inp_attn = build_attn_inp_kv_unified();

        const float kq_scale = hparams.f_attention_scale == 0.0f ? 1.0f/sqrtf(float(n_embd_head)) : hparams.f_attention_scale;

        ggml_tensor * inp_out_ids = build_inp_out_ids();

        for (int il = 0; il < n_layer; ++il) {
            ggml_tensor * inpSA = inpL;

            const bool use_rope = (il + 1) % hparams.n_no_rope_layer_step != 0;

            // norm
            cur = build_norm(inpL,
                    model.layers[il].attn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "attn_norm", il);

            // self-attention
            {
                // compute Q and K and RoPE them
                ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
                cb(Qcur, "Qcur", il);
                if (model.layers[il].bq) {
                    Qcur = ggml_add(ctx0, Qcur, model.layers[il].bq);
                    cb(Qcur, "Qcur", il);
                }

                ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
                cb(Kcur, "Kcur", il);
                if (model.layers[il].bk) {
                    Kcur = ggml_add(ctx0, Kcur, model.layers[il].bk);
                    cb(Kcur, "Kcur", il);
                }

                ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
                cb(Vcur, "Vcur", il);
                if (model.layers[il].bv) {
                    Vcur = ggml_add(ctx0, Vcur, model.layers[il].bv);
                    cb(Vcur, "Vcur", il);
                }

                Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
                Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
                Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

                if (use_rope) {
                    Qcur = ggml_rope_ext(
                            ctx0, Qcur, inp_pos, nullptr,
                            n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                            ext_factor, attn_factor, beta_fast, beta_slow
                            );

                    Kcur = ggml_rope_ext(
                            ctx0, Kcur, inp_pos, nullptr,
                            n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                            ext_factor, attn_factor, beta_fast, beta_slow
                            );
                }

                cb(Qcur, "Qcur", il);
                cb(Kcur, "Kcur", il);
                cb(Vcur, "Vcur", il);

                cur = build_attn(inp_attn,
                        model.layers[il].wo, model.layers[il].bo,
                        Qcur, Kcur, Vcur, nullptr, nullptr, kq_scale, il);
                cb(cur, "attn_out", il);
            }

            if (il == n_layer - 1 && inp_out_ids) {
                cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
                inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
            }

            ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
            cb(ffn_inp, "ffn_inp", il);

            // feed-forward network
            {
                cur = build_norm(ffn_inp,
                        model.layers[il].ffn_norm, NULL,
                        LLM_NORM_RMS, il);
                cb(cur, "ffn_norm", il);

                cur = build_ffn(cur,
                        model.layers[il].ffn_up,   model.layers[il].ffn_up_b,   NULL,
                        model.layers[il].ffn_gate, model.layers[il].ffn_gate_b, NULL,
                        model.layers[il].ffn_down, model.layers[il].ffn_down_b, NULL,
                        NULL,
                        LLM_FFN_SILU, LLM_FFN_PAR, il);
                cb(cur, "ffn_out", il);
            }

            cur = ggml_add(ctx0, cur, ffn_inp);
            cb(cur, "ffn_out", il);

            cur = build_cvec(cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        cur = build_norm(cur,
                model.output_norm, NULL,
                LLM_NORM_RMS, -1);

        cb(cur, "result_norm", -1);
        res->t_embd = cur;

        // lm_head
        cur = build_lora_mm(model.output, cur);

        cb(cur, "result_output", -1);
        res->t_logits = cur;

        ggml_build_forward_expand(gf, cur);
    }
};

struct llm_build_openai_moe_iswa : public llm_graph_context {
    llm_build_openai_moe_iswa(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
        ggml_tensor * cur;
        ggml_tensor * inpL;

        inpL = build_inp_embd(model.tok_embd);

        // inp_pos - contains the positions
        ggml_tensor * inp_pos = build_inp_pos();

        auto * inp_attn = build_attn_inp_kv_unified_iswa();

        for (int il = 0; il < n_layer; ++il) {
            ggml_tensor * inpSA = inpL;

            // norm
            cur = build_norm(inpL,
                    model.layers[il].attn_norm, nullptr,
                    LLM_NORM_RMS, il);
            cb(cur, "attn_norm", il);

            // self-attention
            {
                // compute Q and K and RoPE them
                ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
                cb(Qcur, "Qcur", il);
                if (model.layers[il].bq) {
                    Qcur = ggml_add(ctx0, Qcur, model.layers[il].bq);
                    cb(Qcur, "Qcur", il);
                }

                ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
                cb(Kcur, "Kcur", il);
                if (model.layers[il].bk) {
                    Kcur = ggml_add(ctx0, Kcur, model.layers[il].bk);
                    cb(Kcur, "Kcur", il);
                }

                ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
                cb(Vcur, "Vcur", il);
                if (model.layers[il].bv) {
                    Vcur = ggml_add(ctx0, Vcur, model.layers[il].bv);
                    cb(Vcur, "Vcur", il);
                }

                Qcur = ggml_reshape_3d(ctx0, Qcur, n_rot, n_head,    n_tokens);
                Kcur = ggml_reshape_3d(ctx0, Kcur, n_rot, n_head_kv, n_tokens);
                Vcur = ggml_reshape_3d(ctx0, Vcur, n_rot, n_head_kv, n_tokens);

                Qcur = ggml_rope_ext(
                        ctx0, Qcur, inp_pos, nullptr,
                        n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow
                        );

                Kcur = ggml_rope_ext(
                        ctx0, Kcur, inp_pos, nullptr,
                        n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow
                        );

                cb(Qcur, "Qcur", il);
                cb(Kcur, "Kcur", il);
                cb(Vcur, "Vcur", il);

                cur = build_attn_with_sinks(inp_attn,
                        model.layers[il].wo, model.layers[il].bo,
                        Qcur, Kcur, Vcur, nullptr, nullptr, model.layers[il].attn_sinks, 1.0f/sqrtf(float(n_rot)), il);

                cb(cur, "attn_out", il);
            }

            if (il == n_layer - 1) {
                // skip computing output for unused tokens
                ggml_tensor * inp_out_ids = build_inp_out_ids();
                cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
                inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
            }

            ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
            cb(ffn_inp, "ffn_inp", il);

            cur = ffn_inp;
            cur = build_norm(cur,
                    model.layers[il].attn_post_norm, nullptr,
                    LLM_NORM_RMS, il);
            cb(cur, "attn_post_norm", il);

            // MoE branch
            cur = build_moe_ffn(cur,
                    model.layers[il].ffn_gate_inp,  model.layers[il].ffn_gate_inp_b,
                    model.layers[il].ffn_up_exps,   model.layers[il].ffn_up_exps_b,
                    model.layers[il].ffn_gate_exps, model.layers[il].ffn_gate_exps_b,
                    model.layers[il].ffn_down_exps, model.layers[il].ffn_down_exps_b,
                    nullptr,
                    n_expert, n_expert_used,
                    LLM_FFN_SWIGLU_OAI_MOE, false,
                    false, 0.0,
                    LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX_WEIGHT,
                    il);
            cb(cur, "ffn_moe_out", il);

            cur = ggml_add(ctx0, cur, ffn_inp);

            cur = build_cvec(cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        cur = build_norm(cur,
                model.output_norm, NULL,
                LLM_NORM_RMS, -1);

        cb(cur, "result_norm", -1);
        res->t_embd = cur;

        // lm_head
        cur = build_lora_mm(model.output, cur);

        cb(cur, "result_output", -1);
        res->t_logits = cur;

        ggml_build_forward_expand(gf, cur);
    }
};

struct llm_build_lfm2 : public llm_graph_context {
    const llama_model & model;

    llm_build_lfm2(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params), model(model) {

        ggml_tensor * cur = build_inp_embd(model.tok_embd);
        cb(cur, "model.embed_tokens", -1);

        ggml_tensor * inp_pos     = build_inp_pos();
        auto        * inp_hybrid  = build_inp_mem_hybrid();
        ggml_tensor * inp_out_ids = build_inp_out_ids();

        for (int il = 0; il < n_layer; ++il) {
            auto * prev_cur = cur;
            cur = build_norm(cur, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, il);
            cb(cur, "model.layers.{}.operator_norm", il);

            cur = hparams.is_recurrent(il) ?
                build_shortconv_block(cur, inp_hybrid->get_recr(), il) :
                build_attn_block(cur, inp_pos, inp_hybrid->get_attn(), il) ;

            if (il == n_layer - 1 && inp_out_ids) {
                cur      = ggml_get_rows(ctx0,      cur, inp_out_ids);
                prev_cur = ggml_get_rows(ctx0, prev_cur, inp_out_ids);
            }

            cur = ggml_add(ctx0, prev_cur, cur);
            cur = ggml_add(ctx0, cur, build_feed_forward(cur, il));
        }

        cur = build_norm(cur, model.tok_norm, NULL, LLM_NORM_RMS, -1);
        cb(cur, "model.embedding_norm", -1);
        res->t_embd = cur;

        // lm_head is tied with embeddings
        cur = build_lora_mm(model.tok_embd, cur);
        cb(cur, "lm_head", -1);

        res->t_logits = cur;

        ggml_build_forward_expand(gf, cur);
    }

    ggml_tensor * build_feed_forward(ggml_tensor * cur,
                                     int           il) const {
        cur = build_norm(cur, model.layers[il].ffn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "model.layers.{}.ffn_norm", il);

        GGML_ASSERT(!model.layers[il].ffn_up_b);
        GGML_ASSERT(!model.layers[il].ffn_gate_b);
        GGML_ASSERT(!model.layers[il].ffn_down_b);
        cur = build_ffn(cur,
                model.layers[il].ffn_up,   NULL, NULL,
                model.layers[il].ffn_gate, NULL, NULL,
                model.layers[il].ffn_down, NULL, NULL,
                NULL,
                LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(cur, "model.layers.{}.feed_forward.w2", il);

        return cur;
    }

    ggml_tensor * build_attn_block(ggml_tensor                     * cur,
                                   ggml_tensor                     * inp_pos,
                                   llm_graph_input_attn_kv_unified * inp_attn,
                                   int                               il) const {
        GGML_ASSERT(hparams.n_embd_v_gqa(il) == hparams.n_embd_k_gqa(il));
        auto const n_embd_head = hparams.n_embd_head_v;
        auto const n_head_kv = hparams.n_head_kv(il);

        auto * q = build_lora_mm(model.layers[il].wq, cur);
        cb(q, "model.layers.{}.self_attn.q_proj", il);
        auto * k = build_lora_mm(model.layers[il].wk, cur);
        cb(k, "model.layers.{}.self_attn.k_proj", il);
        auto * v = build_lora_mm(model.layers[il].wv, cur);
        cb(v, "model.layers.{}.self_attn.v_proj", il);

        q = ggml_reshape_3d(ctx0, q, n_embd_head, n_head,    n_tokens);
        k = ggml_reshape_3d(ctx0, k, n_embd_head, n_head_kv, n_tokens);
        v = ggml_reshape_3d(ctx0, v, n_embd_head, n_head_kv, n_tokens);

        // qk norm
        q = build_norm(q, model.layers[il].attn_q_norm, NULL, LLM_NORM_RMS, il);
        cb(q, "model.layers.{}.self_attn.q_layernorm", il);
        k = build_norm(k, model.layers[il].attn_k_norm, NULL, LLM_NORM_RMS, il);
        cb(k, "model.layers.{}.self_attn.k_layernorm", il);

        // RoPE
        q = ggml_rope_ext(
                ctx0, q, inp_pos, nullptr,
                n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow
                );
        k = ggml_rope_ext(
                ctx0, k, inp_pos, nullptr,
                n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow
                );

        cur = build_attn(inp_attn, model.layers[il].wo, NULL,
                q, k, v, nullptr, nullptr, 1.0f/sqrtf(float(n_embd_head)), il);

        cb(cur, "model.layers.{}.self_attn.out_proj", il);

        return cur;
    }

    ggml_tensor * build_shortconv_block(ggml_tensor        * cur,
                                        llm_graph_input_rs * inp_recr,
                                        int                il) {
        const auto *   mctx_cur     = static_cast<const llama_memory_hybrid_context *>(mctx)->get_recr();
        const uint32_t kv_head      = mctx_cur->get_head();
        const int64_t  n_seq_tokens = ubatch.n_seq_tokens;
        const int64_t  n_seqs       = ubatch.n_seqs;
        GGML_ASSERT(n_seqs != 0);
        GGML_ASSERT(ubatch.equal_seqs());
        GGML_ASSERT(ubatch.n_tokens == n_seq_tokens * n_seqs);

        GGML_ASSERT(hparams.n_shortconv_l_cache > 1);
        const uint32_t d_conv = hparams.n_shortconv_l_cache - 1;

        // {n_embd, n_tokens} => {n_embd, n_seq_tokens, n_seqs}
        cur = ggml_reshape_3d(ctx0, cur, cur->ne[0], n_seq_tokens, n_seqs);

        auto * bcx = build_lora_mm(model.layers[il].shortconv.in_proj, cur);
        cb(bcx, "model.layers.{}.conv.in_proj", il);

        constexpr auto n_chunks = 3;
        GGML_ASSERT(bcx->ne[0] % n_chunks == 0);
        auto const chunk_size = bcx->ne[0] / n_chunks;
        auto * b = ggml_view_3d(ctx0, bcx, chunk_size, bcx->ne[1], bcx->ne[2], bcx->nb[1], bcx->nb[2], 0*chunk_size*ggml_element_size(bcx));
        auto * c = ggml_view_3d(ctx0, bcx, chunk_size, bcx->ne[1], bcx->ne[2], bcx->nb[1], bcx->nb[2], 1*chunk_size*ggml_element_size(bcx));
        auto * x = ggml_view_3d(ctx0, bcx, chunk_size, bcx->ne[1], bcx->ne[2], bcx->nb[1], bcx->nb[2], 2*chunk_size*ggml_element_size(bcx));

        auto * bx = ggml_transpose(ctx0, ggml_mul(ctx0, b, x));

        // read conv state
        auto * conv_state = mctx_cur->get_r_l(il);
        auto * conv_rs    = build_rs(inp_recr, conv_state, hparams.n_embd_r(), n_seqs);
        auto * conv       = ggml_reshape_3d(ctx0, conv_rs, d_conv, hparams.n_embd, n_seqs);

        bx = ggml_concat(ctx0, conv, bx, 0);
        GGML_ASSERT(bx->ne[0] > conv->ne[0]);

        // last d_conv columns is a new conv state
        auto * new_conv = ggml_view_3d(ctx0, bx, conv->ne[0], bx->ne[1], bx->ne[2], bx->nb[1], bx->nb[2], (bx->ne[0] - conv->ne[0])*ggml_element_size(bx));
        GGML_ASSERT(ggml_are_same_shape(conv, new_conv));

        // write new conv conv state
        ggml_build_forward_expand(
                gf,
                ggml_cpy(
                    ctx0,
                    new_conv,
                    ggml_view_1d(
                        ctx0,
                        conv_state,
                        ggml_nelements(new_conv),
                        kv_head*d_conv*n_embd*ggml_element_size(new_conv)
                        )
                    )
                );

        auto * conv_kernel = model.layers[il].shortconv.conv;
        auto * conv_out = ggml_ssm_conv(ctx0, bx, conv_kernel);
        cb(conv_out, "model.layers.{}.conv.conv", il);

        auto * y = ggml_mul(ctx0, c, conv_out);
        y = build_lora_mm(model.layers[il].shortconv.out_proj, y);
        cb(y, "model.layers.{}.conv.out_proj", il);
        // {n_embd, n_seq_tokens, n_seqs} => {n_embd, n_tokens}
        y = ggml_reshape_2d(ctx0, y, y->ne[0], n_seq_tokens * n_seqs);

        return y;
    }
};

template <bool iswa>
struct llm_build_smallthinker : public llm_graph_context{
    llm_build_smallthinker(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params){
        const int64_t n_embd_head = hparams.n_embd_head_v;

        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
        GGML_ASSERT(n_embd_head == hparams.n_rot);

        ggml_tensor * cur;
        ggml_tensor * inpL;

        inpL = build_inp_embd(model.tok_embd);

        // inp_pos - contains the positions
        ggml_tensor * inp_pos = build_inp_pos();

        using inp_attn_type = std::conditional_t<iswa, llm_graph_input_attn_kv_unified_iswa, llm_graph_input_attn_kv_unified>;
        inp_attn_type * inp_attn = nullptr;

        if constexpr (iswa) {
            inp_attn = build_attn_inp_kv_unified_iswa();
        } else {
            inp_attn = build_attn_inp_kv_unified();
        }

        ggml_tensor * inp_out_ids = build_inp_out_ids();

        for (int il = 0; il < n_layer; ++il) {
            ggml_tensor * inpSA  = inpL;
            ggml_tensor * probs  = nullptr;

            probs = build_lora_mm(model.layers[il].ffn_gate_inp, inpL);  // [n_expert, n_tokens]
            cb(probs, "ffn_moe_logits", il);

            // norm
            cur = build_norm(inpL,model.layers[il].attn_norm, NULL, LLM_NORM_RMS, il);
            cb(cur, "attn_norm", il);

            // self_attention
            {
                // compute Q and K and RoPE them
                struct ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
                cb(Qcur, "Qcur", il);

                struct ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
                cb(Kcur, "Kcur", il);

                struct ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
                cb(Vcur, "Vcur", il);

                Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens);
                Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
                Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

                if (hparams.n_no_rope_layer_step == n_layer || il % hparams.n_no_rope_layer_step != 0) {
                    Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                                     ext_factor, attn_factor, beta_fast, beta_slow);

                    Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                                     ext_factor, attn_factor, beta_fast, beta_slow);
                }

                cb(Qcur, "Qcur", il);
                cb(Kcur, "Kcur", il);

                cur = build_attn(inp_attn,
                        model.layers[il].wo, model.layers[il].bo,
                        Qcur, Kcur, Vcur, nullptr, nullptr, 1.0f / sqrtf(float(n_embd_head)), il);
            }

            if (il == n_layer - 1 && inp_out_ids) {
                cur = ggml_get_rows(ctx0, cur, inp_out_ids);
                inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
                probs = ggml_get_rows(ctx0, probs, inp_out_ids);
            }

            ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
            cb(ffn_inp, "ffn_inp", il);

            // MoE branch
            cur = build_norm(ffn_inp, model.layers[il].ffn_norm, NULL, LLM_NORM_RMS, il);
            cb(cur, "ffn_norm", il);

            ggml_tensor * ffn_out =
                build_moe_ffn(cur,
                        nullptr,
                        model.layers[il].ffn_up_exps,
                        model.layers[il].ffn_gate_exps,
                        model.layers[il].ffn_down_exps,
                        nullptr,
                        n_expert, n_expert_used,
                        LLM_FFN_RELU, true,
                        false, 0.0,
                        static_cast<llama_expert_gating_func_type>(hparams.expert_gating_func),
                        il, probs);

            cb(ffn_out, "ffn_out", il);
            cur = ffn_out;

            cur = ggml_add(ctx0, cur, ffn_inp);
            cur = build_cvec(cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        cur = build_norm(cur, model.output_norm, NULL, LLM_NORM_RMS, -1);
        cb(cur, "result_norm", -1);

        // lm_head
        cur = build_lora_mm(model.output, cur);
        cb(cur, "result_output", -1);
        res->t_logits = cur;

        ggml_build_forward_expand(gf, cur);
    }
};

llama_memory_i * llama_model::create_memory(const llama_memory_params & params, llama_cparams & cparams) const {
    llama_memory_i * res;

    switch (arch) {
        // Models that need specific instantiation should be handled in the
        // switch statement
        case LLM_ARCH_BERT:
        case LLM_ARCH_JINA_BERT_V2:
        case LLM_ARCH_NOMIC_BERT:
        case LLM_ARCH_NOMIC_BERT_MOE:
        case LLM_ARCH_NEO_BERT:
        case LLM_ARCH_WAVTOKENIZER_DEC:
        case LLM_ARCH_DREAM:
        case LLM_ARCH_LLADA:
            {
                res = nullptr;
            } break;
        // Models that need standard caching should rely on recurrent/hybrid
        // checks
        default:
            {
                if (llm_arch_is_recurrent(arch)) {
                    res = new llama_memory_recurrent(
                            *this,
                            nullptr,
                            GGML_TYPE_F32,
                            GGML_TYPE_F32,
                            cparams.offload_kqv,
                            std::max((uint32_t) 1, cparams.n_seq_max),
                            cparams.n_seq_max);
                } else if (llm_arch_is_hybrid(arch)) {
                    const auto padding = llama_kv_cache_unified::get_padding(cparams);

                    cparams.n_ctx = GGML_PAD(cparams.n_ctx, padding);

                    res = new llama_memory_hybrid(
                        /* model             */ *this,
                        /* attn_type_k       */ params.type_k,
                        /* attn_type_v       */ params.type_v,
                        /* attn_v_trans      */ !cparams.flash_attn,
                        /* attn_kv_size      */ cparams.n_ctx,
                        /* attn_n_pad        */ padding,
                        /* attn_n_swa        */ hparams.n_swa,
                        /* attn_swa_type     */ hparams.swa_type,
                        /* recurrent_type_k  */ GGML_TYPE_F32,
                        /* recurrent_type_v  */ GGML_TYPE_F32,
                        /* recurrent_kv_size */ std::max((uint32_t) 1, cparams.n_seq_max),
                        /* n_seq_max         */ cparams.n_seq_max,
                        /* offload           */ cparams.offload_kqv,
                        /* unified           */ cparams.kv_unified,
                        /* filter_attn       */ (arch == LLM_ARCH_FALCON_H1) ? [&](int32_t) { return true; } : (llama_memory_hybrid::layer_filter_cb)nullptr,
                        /* filter_recr       */ (arch == LLM_ARCH_FALCON_H1) ? [&](int32_t) { return true; } : (llama_memory_hybrid::layer_filter_cb)nullptr);
                } else {
                    const auto padding = llama_kv_cache_unified::get_padding(cparams);

                    uint32_t n_ctx_per_stream = cparams.n_ctx;

                    if (!cparams.kv_unified) {
                        n_ctx_per_stream = (cparams.n_ctx + cparams.n_seq_max - 1)/cparams.n_seq_max;
                        n_ctx_per_stream = GGML_PAD(n_ctx_per_stream, padding);

                        cparams.n_ctx = n_ctx_per_stream*cparams.n_seq_max;
                    } else {
                        n_ctx_per_stream = GGML_PAD(n_ctx_per_stream, padding);

                        cparams.n_ctx = n_ctx_per_stream;
                    }

                    LLAMA_LOG_DEBUG("%s: n_ctx = %u (padded)\n", __func__, cparams.n_ctx);

                    if (hparams.swa_type != LLAMA_SWA_TYPE_NONE) {
                        GGML_ASSERT(hparams.is_swa_any());

                        res = new llama_kv_cache_unified_iswa(
                                *this,
                                params.type_k,
                                params.type_v,
                                !cparams.flash_attn,
                                cparams.offload_kqv,
                                params.swa_full,
                                cparams.kv_unified,
                                n_ctx_per_stream,
                                cparams.n_seq_max,
                                cparams.n_ubatch,
                                padding);
                    } else {
                        GGML_ASSERT(!hparams.is_swa_any());

                        res = new llama_kv_cache_unified(
                                *this,
                                nullptr,
                                params.type_k,
                                params.type_v,
                                !cparams.flash_attn,
                                cparams.offload_kqv,
                                cparams.kv_unified,
                                n_ctx_per_stream,
                                cparams.n_seq_max,
                                padding,
                                hparams.n_swa,
                                hparams.swa_type);
                    }
                }
            }
    }

    return res;
}

ggml_cgraph * llama_model::build_graph(const llm_graph_params & params) const {
    std::unique_ptr<llm_graph_context> llm;

    switch (arch) {
        case LLM_ARCH_LLAMA:
            {
                llm = std::make_unique<llm_build_llama>(*this, params);
            } break;
        case LLM_ARCH_LLAMA4:
            {
                llm = std::make_unique<llm_build_llama_iswa>(*this, params);
            } break;
        case LLM_ARCH_DECI:
            {
                llm = std::make_unique<llm_build_deci>(*this, params);
            } break;
        case LLM_ARCH_BAICHUAN:
            {
                llm = std::make_unique<llm_build_baichuan>(*this, params);
            } break;
        case LLM_ARCH_FALCON:
            {
                llm = std::make_unique<llm_build_falcon>(*this, params);
            } break;
        case LLM_ARCH_GROK:
            {
                llm = std::make_unique<llm_build_grok>(*this, params);
            } break;
        case LLM_ARCH_STARCODER:
            {
                llm = std::make_unique<llm_build_starcoder>(*this, params);
            } break;
        case LLM_ARCH_REFACT:
            {
                llm = std::make_unique<llm_build_refact>(*this, params);
            } break;
        case LLM_ARCH_BERT:
        case LLM_ARCH_JINA_BERT_V2:
        case LLM_ARCH_NOMIC_BERT:
        case LLM_ARCH_NOMIC_BERT_MOE:
            {
                llm = std::make_unique<llm_build_bert>(*this, params);
            } break;
        case LLM_ARCH_NEO_BERT:
            {
                llm = std::make_unique<llm_build_neo_bert>(*this, params);
            } break;
        case LLM_ARCH_BLOOM:
            {
                llm = std::make_unique<llm_build_bloom>(*this, params);
            } break;
        case LLM_ARCH_MPT:
            {
                llm = std::make_unique<llm_build_mpt>(*this, params);
            } break;
        case LLM_ARCH_STABLELM:
            {
                llm = std::make_unique<llm_build_stablelm>(*this, params);
            } break;
        case LLM_ARCH_QWEN:
            {
                llm = std::make_unique<llm_build_qwen>(*this, params);
            } break;
        case LLM_ARCH_QWEN2:
            {
                llm = std::make_unique<llm_build_qwen2>(*this, params);
            } break;
        case LLM_ARCH_DREAM:
            {
                llm = std::make_unique<llm_build_dream>(*this, params);
            }
            break;
        case LLM_ARCH_LLADA:
            {
                llm = std::make_unique<llm_build_llada>(*this, params);
            }
            break;
        case LLM_ARCH_QWEN2VL:
            {
                llm = std::make_unique<llm_build_qwen2vl>(*this, params);
            } break;
        case LLM_ARCH_QWEN2MOE:
            {
                llm = std::make_unique<llm_build_qwen2moe>(*this, params);
            } break;
        case LLM_ARCH_QWEN3:
            {
                llm = std::make_unique<llm_build_qwen3>(*this, params);
            } break;
        case LLM_ARCH_QWEN3MOE:
            {
                llm = std::make_unique<llm_build_qwen3moe>(*this, params);
            } break;
        case LLM_ARCH_PHI2:
            {
                llm = std::make_unique<llm_build_phi2>(*this, params);
            } break;
        case LLM_ARCH_PHI3:
        case LLM_ARCH_PHIMOE:
            {
                if (hparams.swa_type != LLAMA_SWA_TYPE_NONE) {
                    llm = std::make_unique<llm_build_phi3<true>> (*this, params);
                } else {
                    llm = std::make_unique<llm_build_phi3<false>>(*this, params);
                }
            } break;
        case LLM_ARCH_PLAMO:
            {
                llm = std::make_unique<llm_build_plamo>(*this, params);
            } break;
        case LLM_ARCH_PLAMO2:
            {
                llm = std::make_unique<llm_build_plamo2>(*this, params);
            } break;
        case LLM_ARCH_GPT2:
            {
                llm = std::make_unique<llm_build_gpt2>(*this, params);
            } break;
        case LLM_ARCH_CODESHELL:
            {
                llm = std::make_unique<llm_build_codeshell>(*this, params);
            } break;
        case LLM_ARCH_ORION:
            {
                llm = std::make_unique<llm_build_orion>(*this, params);
            } break;
        case LLM_ARCH_INTERNLM2:
            {
                llm = std::make_unique<llm_build_internlm2>(*this, params);
            } break;
        case LLM_ARCH_MINICPM3:
            {
                llm = std::make_unique<llm_build_minicpm3>(*this, params);
            } break;
        case LLM_ARCH_GEMMA:
            {
                llm = std::make_unique<llm_build_gemma>(*this, params);
            } break;
        case LLM_ARCH_GEMMA2:
            {
                llm = std::make_unique<llm_build_gemma2_iswa>(*this, params);
            } break;
        case LLM_ARCH_GEMMA3:
            {
                llm = std::make_unique<llm_build_gemma3_iswa>(*this, params);
            } break;
        case LLM_ARCH_GEMMA3N:
            {
                llm = std::make_unique<llm_build_gemma3n_iswa>(*this, params);
            } break;
        case LLM_ARCH_STARCODER2:
            {
                llm = std::make_unique<llm_build_starcoder2>(*this, params);
            } break;
        case LLM_ARCH_MAMBA:
        case LLM_ARCH_MAMBA2:
            {
                llm = std::make_unique<llm_build_mamba>(*this, params);
            } break;
        case LLM_ARCH_JAMBA:
            {
                llm = std::make_unique<llm_build_jamba>(*this, params);
            } break;
        case LLM_ARCH_XVERSE:
            {
                llm = std::make_unique<llm_build_xverse>(*this, params);
            } break;
        case LLM_ARCH_COMMAND_R:
            {
                llm = std::make_unique<llm_build_command_r>(*this, params);
            } break;
        case LLM_ARCH_COHERE2:
            {
                llm = std::make_unique<llm_build_cohere2_iswa>(*this, params);
            } break;
        case LLM_ARCH_DBRX:
            {
                llm = std::make_unique<llm_build_dbrx>(*this, params);
            } break;
        case LLM_ARCH_OLMO:
            {
                llm = std::make_unique<llm_build_olmo>(*this, params);
            } break;
        case LLM_ARCH_OLMO2:
            {
                llm = std::make_unique<llm_build_olmo2>(*this, params);
            } break;
        case LLM_ARCH_OLMOE:
            {
                llm = std::make_unique<llm_build_olmoe>(*this, params);
            } break;
        case LLM_ARCH_OPENELM:
            {
                llm = std::make_unique<llm_build_openelm>(*this, params);
            } break;
        case LLM_ARCH_GPTNEOX:
            {
                llm = std::make_unique<llm_build_gptneox>(*this, params);
            } break;
        case LLM_ARCH_ARCTIC:
            {
                llm = std::make_unique<llm_build_arctic>(*this, params);
            } break;
        case LLM_ARCH_DEEPSEEK:
            {
                llm = std::make_unique<llm_build_deepseek>(*this, params);
            } break;
        case LLM_ARCH_DEEPSEEK2:
            {
                llm = std::make_unique<llm_build_deepseek2>(*this, params);
            } break;
        case LLM_ARCH_CHATGLM:
            {
                llm = std::make_unique<llm_build_chatglm>(*this, params);
            } break;
        case LLM_ARCH_GLM4:
            {
                llm = std::make_unique<llm_build_glm4>(*this, params);
            } break;
        case LLM_ARCH_GLM4_MOE:
            {
                llm = std::make_unique<llm_build_glm4_moe>(*this, params);
            } break;
        case LLM_ARCH_BITNET:
            {
                llm = std::make_unique<llm_build_bitnet>(*this, params);
            } break;
        case LLM_ARCH_T5:
            {
                switch (params.gtype) {
                    case LLM_GRAPH_TYPE_ENCODER:
                        llm = std::make_unique<llm_build_t5_enc>(*this, params);
                        break;
                    case LLM_GRAPH_TYPE_DEFAULT:
                    case LLM_GRAPH_TYPE_DECODER:
                        llm = std::make_unique<llm_build_t5_dec>(*this, params);
                        break;
                    default:
                        GGML_ABORT("invalid graph type");
                };
            } break;
        case LLM_ARCH_T5ENCODER:
            {
                llm = std::make_unique<llm_build_t5_enc>(*this, params);
            }
            break;
        case LLM_ARCH_JAIS:
            {
                llm = std::make_unique<llm_build_jais>(*this, params);
            } break;
        case LLM_ARCH_NEMOTRON:
            {
                llm = std::make_unique<llm_build_nemotron>(*this, params);
            } break;
        case LLM_ARCH_EXAONE:
            {
                llm = std::make_unique<llm_build_exaone>(*this, params);
            } break;
        case LLM_ARCH_EXAONE4:
            {
                if (hparams.swa_type == LLAMA_SWA_TYPE_STANDARD) {
                    llm = std::make_unique<llm_build_exaone4<true>>(*this, params);
                } else {
                    llm = std::make_unique<llm_build_exaone4<false>>(*this, params);
                }
            } break;
        case LLM_ARCH_RWKV6:
            {
                llm = std::make_unique<llm_build_rwkv6>(*this, params);
            } break;
        case LLM_ARCH_RWKV6QWEN2:
            {
                llm = std::make_unique<llm_build_rwkv6qwen2>(*this, params);
            } break;
        case LLM_ARCH_RWKV7:
            {
                llm = std::make_unique<llm_build_rwkv7>(*this, params);
            } break;
        case LLM_ARCH_ARWKV7:
            {
                llm = std::make_unique<llm_build_arwkv7>(*this, params);
            } break;
        case LLM_ARCH_GRANITE:
        case LLM_ARCH_GRANITE_MOE:
        case LLM_ARCH_MINICPM:
            {
                llm = std::make_unique<llm_build_granite>(*this, params);
            } break;
        case LLM_ARCH_GRANITE_HYBRID:
            {
                llm = std::make_unique<llm_build_granite_hybrid>(*this, params);
            } break;
        case LLM_ARCH_CHAMELEON:
            {
                llm = std::make_unique<llm_build_chameleon>(*this, params);
            } break;
        case LLM_ARCH_SOLAR:
            {
                llm = std::make_unique<llm_build_solar>(*this, params);
            } break;
        case LLM_ARCH_WAVTOKENIZER_DEC:
            {
                llm = std::make_unique<llm_build_wavtokenizer_dec>(*this, params);
            } break;
        case LLM_ARCH_PLM:
            {
                llm = std::make_unique<llm_build_plm>(*this, params);
            } break;
        case LLM_ARCH_BAILINGMOE:
            {
                llm = std::make_unique<llm_build_bailingmoe>(*this, params);
            } break;
        case LLM_ARCH_DOTS1:
            {
                llm = std::make_unique<llm_build_dots1>(*this, params);
            } break;
        case LLM_ARCH_ARCEE:
            {
                llm = std::make_unique<llm_build_arcee>(*this, params);
            } break;
        case LLM_ARCH_ERNIE4_5:
            {
                llm = std::make_unique<llm_build_ernie4_5>(*this, params);
            } break;
        case LLM_ARCH_ERNIE4_5_MOE:
            {
                llm = std::make_unique<llm_build_ernie4_5_moe>(*this, params);
            } break;
        case LLM_ARCH_HUNYUAN_MOE:
            {
                llm = std::make_unique<llm_build_hunyuan_moe>(*this, params);
            } break;
        case LLM_ARCH_HUNYUAN_DENSE:
            {
                llm = std::make_unique<llm_build_hunyuan_dense>(*this, params);
            } break;
        case LLM_ARCH_SMOLLM3:
            {
                llm = std::make_unique<llm_build_smollm3>(*this, params);
            } break;
        case LLM_ARCH_OPENAI_MOE:
            {
                llm = std::make_unique<llm_build_openai_moe_iswa>(*this, params);
            } break;
        case LLM_ARCH_FALCON_H1:
            {
                llm = std::make_unique<llm_build_falcon_h1>(*this, params);
            } break;
        case LLM_ARCH_LFM2:
            {
                llm = std::make_unique<llm_build_lfm2>(*this, params);
            } break;
        case LLM_ARCH_SMALLTHINKER:
            {
                if (hparams.swa_type == LLAMA_SWA_TYPE_STANDARD) {
                    llm = std::make_unique<llm_build_smallthinker<true>> (*this, params);
                } else {
                    llm = std::make_unique<llm_build_smallthinker<false>>(*this, params);
                }
            } break;
        default:
            GGML_ABORT("fatal error");
    }

    // add on pooling layer
    llm->build_pooling(cls, cls_b, cls_out, cls_out_b);

    return llm->res->get_gf();
}

//
// interface implementation
//

llama_model_params llama_model_default_params() {
    llama_model_params result = {
        /*.devices                     =*/ nullptr,
        /*.tensor_buft_overrides       =*/ nullptr,
        /*.n_gpu_layers                =*/ 0,
        /*.split_mode                  =*/ LLAMA_SPLIT_MODE_LAYER,
        /*.main_gpu                    =*/ 0,
        /*.tensor_split                =*/ nullptr,
        /*.progress_callback           =*/ nullptr,
        /*.progress_callback_user_data =*/ nullptr,
        /*.kv_overrides                =*/ nullptr,
        /*.vocab_only                  =*/ false,
        /*.use_mmap                    =*/ true,
        /*.use_mlock                   =*/ false,
        /*.check_tensors               =*/ false,
        /*.use_extra_bufts             =*/ true,
    };

#ifdef GGML_USE_METAL
    // note: we usually have plenty of VRAM, so by default offload all layers to the GPU
    result.n_gpu_layers = 999;
#endif

    return result;
}

const llama_vocab * llama_model_get_vocab(const llama_model * model) {
    return &model->vocab;
}

void llama_free_model(llama_model * model) {
    llama_model_free(model);
}

void llama_model_free(llama_model * model) {
    delete model;
}

int32_t llama_model_n_ctx_train(const llama_model * model) {
    return model->hparams.n_ctx_train;
}

int32_t llama_model_n_embd(const llama_model * model) {
    return model->hparams.n_embd;
}

int32_t llama_model_n_layer(const llama_model * model) {
    return model->hparams.n_layer;
}

int32_t llama_model_n_head(const llama_model * model) {
    return model->hparams.n_head();
}

int32_t llama_model_n_head_kv(const llama_model * model) {
    return model->hparams.n_head_kv();
}

int32_t llama_model_n_swa(const llama_model * model) {
    return model->hparams.n_swa;
}

uint32_t llama_model_n_cls_out(const struct llama_model * model) {
    return model->hparams.n_cls_out;
}

const char * llama_model_cls_label(const struct llama_model * model, uint32_t i) {
    if (i < model->classifier_labels.size()) {
        return model->classifier_labels[i].c_str();
    }

    return nullptr;
}

// deprecated
int32_t llama_n_ctx_train(const llama_model * model) {
    return llama_model_n_ctx_train(model);
}

// deprecated
int32_t llama_n_embd(const llama_model * model) {
    return llama_model_n_embd(model);
}

// deprecated
int32_t llama_n_layer(const llama_model * model) {
    return llama_model_n_layer(model);
}

// deprecated
int32_t llama_n_head(const llama_model * model) {
    return llama_model_n_head(model);
}

llama_rope_type llama_model_rope_type(const llama_model * model) {
    switch (model->arch) {
        // these models do not use RoPE
        case LLM_ARCH_GPT2:
        case LLM_ARCH_GPTJ:
        case LLM_ARCH_MPT:
        case LLM_ARCH_REFACT:
        case LLM_ARCH_BLOOM:
        case LLM_ARCH_MAMBA:
        case LLM_ARCH_MAMBA2:
        case LLM_ARCH_JAMBA:
        case LLM_ARCH_JINA_BERT_V2:
        case LLM_ARCH_T5:
        case LLM_ARCH_T5ENCODER:
        case LLM_ARCH_JAIS:
        case LLM_ARCH_RWKV6:
        case LLM_ARCH_RWKV6QWEN2:
        case LLM_ARCH_RWKV7:
        case LLM_ARCH_ARWKV7:
        case LLM_ARCH_WAVTOKENIZER_DEC:
            return LLAMA_ROPE_TYPE_NONE;

        // use what we call a normal RoPE, operating on pairs of consecutive head values
        case LLM_ARCH_LLAMA:
        case LLM_ARCH_LLADA:
        case LLM_ARCH_LLAMA4:
        case LLM_ARCH_DECI:
        case LLM_ARCH_BAICHUAN:
        case LLM_ARCH_STARCODER:
        case LLM_ARCH_INTERNLM2:
        case LLM_ARCH_MINICPM:
        case LLM_ARCH_XVERSE:
        case LLM_ARCH_COMMAND_R:
        case LLM_ARCH_COHERE2:
        case LLM_ARCH_OLMO:
        case LLM_ARCH_ARCTIC:
        case LLM_ARCH_DEEPSEEK:
        case LLM_ARCH_DEEPSEEK2:
        case LLM_ARCH_PLM:
        case LLM_ARCH_CHATGLM:
        case LLM_ARCH_GLM4:
        case LLM_ARCH_GRANITE:
        case LLM_ARCH_GRANITE_MOE:
        case LLM_ARCH_GRANITE_HYBRID:
        case LLM_ARCH_CHAMELEON:
        case LLM_ARCH_SOLAR:
        case LLM_ARCH_BAILINGMOE:
        case LLM_ARCH_NEO_BERT:
        case LLM_ARCH_SMOLLM3:
        case LLM_ARCH_ARCEE:
        case LLM_ARCH_ERNIE4_5:
        case LLM_ARCH_ERNIE4_5_MOE:
            return LLAMA_ROPE_TYPE_NORM;

        // the pairs of head values are offset by n_rot/2
        case LLM_ARCH_FALCON:
        case LLM_ARCH_FALCON_H1:
        case LLM_ARCH_GROK:
        case LLM_ARCH_DBRX:
        case LLM_ARCH_BERT:
        case LLM_ARCH_NOMIC_BERT:
        case LLM_ARCH_NOMIC_BERT_MOE:
        case LLM_ARCH_STABLELM:
        case LLM_ARCH_BITNET:
        case LLM_ARCH_QWEN:
        case LLM_ARCH_QWEN2:
        case LLM_ARCH_DREAM:
        case LLM_ARCH_QWEN2MOE:
        case LLM_ARCH_QWEN3:
        case LLM_ARCH_QWEN3MOE:
        case LLM_ARCH_OLMO2:
        case LLM_ARCH_OLMOE:
        case LLM_ARCH_PHI2:
        case LLM_ARCH_PHI3:
        case LLM_ARCH_PHIMOE:
        case LLM_ARCH_PLAMO:
        case LLM_ARCH_PLAMO2:
        case LLM_ARCH_GEMMA:
        case LLM_ARCH_GEMMA2:
        case LLM_ARCH_GEMMA3:
        case LLM_ARCH_GEMMA3N:
        case LLM_ARCH_STARCODER2:
        case LLM_ARCH_OPENELM:
        case LLM_ARCH_GPTNEOX:
        case LLM_ARCH_CODESHELL:
        case LLM_ARCH_ORION:
        case LLM_ARCH_NEMOTRON:
        case LLM_ARCH_EXAONE:
        case LLM_ARCH_EXAONE4:
        case LLM_ARCH_MINICPM3:
        case LLM_ARCH_DOTS1:
        case LLM_ARCH_HUNYUAN_MOE:
        case LLM_ARCH_OPENAI_MOE:
        case LLM_ARCH_HUNYUAN_DENSE:
        case LLM_ARCH_LFM2:
        case LLM_ARCH_SMALLTHINKER:
        case LLM_ARCH_GLM4_MOE:
            return LLAMA_ROPE_TYPE_NEOX;

        case LLM_ARCH_QWEN2VL:
            return LLAMA_ROPE_TYPE_MROPE;

        // all model arches should be listed explicitly here
        case LLM_ARCH_UNKNOWN:
            GGML_ABORT("unknown architecture");
    }

    return LLAMA_ROPE_TYPE_NONE;
}

float llama_model_rope_freq_scale_train(const llama_model * model) {
    return model->hparams.rope_freq_scale_train;
}

int32_t llama_model_meta_val_str(const llama_model * model, const char * key, char * buf, size_t buf_size) {
    const auto & it = model->gguf_kv.find(key);
    if (it == model->gguf_kv.end()) {
        if (buf_size > 0) {
            buf[0] = '\0';
        }
        return -1;
    }
    return snprintf(buf, buf_size, "%s", it->second.c_str());
}

int32_t llama_model_meta_count(const llama_model * model) {
    return (int)model->gguf_kv.size();
}

int32_t llama_model_meta_key_by_index(const llama_model * model, int i, char * buf, size_t buf_size) {
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

int32_t llama_model_meta_val_str_by_index(const llama_model * model, int32_t i, char * buf, size_t buf_size) {
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

int32_t llama_model_desc(const llama_model * model, char * buf, size_t buf_size) {
    return snprintf(buf, buf_size, "%s", model->desc().c_str());
}

uint64_t llama_model_size(const llama_model * model) {
    return model->size();
}

const char * llama_model_chat_template(const llama_model * model, const char * name) {
    const auto key = name ? LLM_KV(model->arch, name)(LLM_KV_TOKENIZER_CHAT_TEMPLATE)
        : LLM_KV(model->arch)(LLM_KV_TOKENIZER_CHAT_TEMPLATE);
    const auto & it = model->gguf_kv.find(key);
    if (it == model->gguf_kv.end()) {
        // one-off fix for very popular models (so we are not flooded with issues)
        // do not extend this list unless absolutely necessary
        // Mistral-Small-2503 does not have built-in chat template
        llama_vocab_pre_type pre_type = model->vocab.get_pre_type();
        if (!name && pre_type == LLAMA_VOCAB_PRE_TYPE_TEKKEN && model->layers.size() == 40) {
            return "mistral-v7-tekken";
        }

        return nullptr;
    }

    return it->second.c_str();
}

uint64_t llama_model_n_params(const llama_model * model) {
    return model->n_elements();
}

bool llama_model_has_encoder(const llama_model * model) {
    switch (model->arch) {
        case LLM_ARCH_T5:        return true;
        case LLM_ARCH_T5ENCODER: return true;
        default:                 return false;
    }
}

bool llama_model_has_decoder(const llama_model * model) {
    switch (model->arch) {
        case LLM_ARCH_T5ENCODER: return false;
        default:                 return true;
    }
}

llama_token llama_model_decoder_start_token(const llama_model * model) {
    return model->hparams.dec_start_token_id;
}

bool llama_model_is_recurrent(const llama_model * model) {
    return llm_arch_is_recurrent(model->arch);
}

bool llama_model_is_diffusion(const llama_model * model) {
    return llm_arch_is_diffusion(model->arch);
}

const std::vector<std::pair<std::string, ggml_tensor *>> & llama_internal_get_tensor_map(const llama_model * model) {
    return model->tensors_by_name;
}
