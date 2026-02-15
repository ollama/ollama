#include "llama-model.h"

#include "llama-impl.h"
#include "llama-mmap.h"
#include "llama-cparams.h"
#include "llama-model-loader.h"

#include "llama-kv-cache.h"
#include "llama-kv-cache-iswa.h"
#include "llama-memory-hybrid.h"
#include "llama-memory-recurrent.h"

#include "ggml-cpp.h"

#include "models/models.h"

#include <algorithm>
#include <cassert>
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
        case LLM_TYPE_140M:          return "140M";
        case LLM_TYPE_160M:          return "160M";
        case LLM_TYPE_190M:          return "190M";
        case LLM_TYPE_220M:          return "220M";
        case LLM_TYPE_250M:          return "250M";
        case LLM_TYPE_256M:          return "256M";
        case LLM_TYPE_270M:          return "270M";
        case LLM_TYPE_335M:          return "335M";
        case LLM_TYPE_350M:          return "350M";
        case LLM_TYPE_360M:          return "360M";
        case LLM_TYPE_410M:          return "410M";
        case LLM_TYPE_450M:          return "450M";
        case LLM_TYPE_475M:          return "475M";
        case LLM_TYPE_558M:          return "558M";
        case LLM_TYPE_700M:          return "700M";
        case LLM_TYPE_770M:          return "770M";
        case LLM_TYPE_780M:          return "780M";
        case LLM_TYPE_950M:          return "950M";
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
        case LLM_TYPE_2_6B:          return "2.6B";
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
        case LLM_TYPE_26B:           return "26B";
        case LLM_TYPE_27B:           return "27B";
        case LLM_TYPE_30B:           return "30B";
        case LLM_TYPE_32B:           return "32B";
        case LLM_TYPE_34B:           return "34B";
        case LLM_TYPE_35B:           return "35B";
        case LLM_TYPE_36B:           return "36B";
        case LLM_TYPE_40B:           return "40B";
        case LLM_TYPE_65B:           return "65B";
        case LLM_TYPE_70B:           return "70B";
        case LLM_TYPE_120B:          return "120B";
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
        case LLM_TYPE_7B_A1B:        return "7B.A1B";
        case LLM_TYPE_8B_A1B:        return "8B.A1B";
        case LLM_TYPE_16B_A1B:       return "16B.A1B";
        case LLM_TYPE_21B_A3B:       return "21B.A3B";
        case LLM_TYPE_30B_A3B:       return "30B.A3B";
        case LLM_TYPE_31B_A3_5B:     return "31B.A3.5B";
        case LLM_TYPE_80B_A3B:       return "80B.A3B";
        case LLM_TYPE_100B_A6B:      return "100B.A6B";
        case LLM_TYPE_106B_A12B:     return "106B.A12B";
        case LLM_TYPE_230B_A10B:     return "230B.A10B";
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
                const int n_embd_inp = hparams.n_embd_inp();
                ggml_tensor * b = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, n_embd_inp, w->ne[1], 1, 1);
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
static buft_list_t make_cpu_buft_list(const std::vector<ggml_backend_dev_t> & devices, bool use_extra_bufts, bool no_host) {
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
    if (!no_host) {
        for (auto * dev : devices) {
            ggml_backend_buffer_type_t buft = ggml_backend_dev_host_buffer_type(dev);
            if (buft) {
                buft_list.emplace_back(dev, buft);
                break;
            }
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

    // add the device extra buffer type (if any)
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    auto ggml_backend_dev_get_extra_bufts_fn = (ggml_backend_dev_get_extra_bufts_t)
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_dev_get_extra_bufts");

    if (ggml_backend_dev_get_extra_bufts_fn) {
        ggml_backend_buffer_type_t * extra_bufts = ggml_backend_dev_get_extra_bufts_fn(dev);
        while (extra_bufts && *extra_bufts) {
            buft_list.emplace_back(dev, *extra_bufts);
            ++extra_bufts;
        }
    }

    return buft_list;
}

struct llama_model::impl {
    impl() = default;
    ~impl() = default;

    uint64_t n_elements = 0;

    size_t n_bytes = 0;

    std::string desc_str;

    // model memory mapped files
    llama_mmaps mappings;

    // objects representing data potentially being locked in memory
    llama_mlocks mlock_bufs;
    llama_mlocks mlock_mmaps;

    // contexts where the model tensors metadata is stored as well ass the corresponding buffers:
    std::vector<std::pair<ggml_context_ptr, std::vector<ggml_backend_buffer_ptr>>> ctxs_bufs;

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

llama_model::~llama_model() = default;

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
    // for CLIP models, we only need to load tensors, no hparams
    if (hparams.vocab_only || ml.get_arch() == LLM_ARCH_CLIP) {
        return;
    }

    ml.get_key(LLM_KV_CONTEXT_LENGTH,          hparams.n_ctx_train);
    ml.get_key(LLM_KV_EMBEDDING_LENGTH,        hparams.n_embd);
    ml.get_key(LLM_KV_BLOCK_COUNT,             hparams.n_layer);
    ml.get_key(LLM_KV_EXPERT_COUNT,            hparams.n_expert,        false);
    ml.get_key(LLM_KV_EXPERT_USED_COUNT,       hparams.n_expert_used,   false);
    ml.get_key(LLM_KV_EXPERT_GROUP_COUNT,      hparams.n_expert_groups, false);
    ml.get_key(LLM_KV_EXPERT_GROUP_USED_COUNT, hparams.n_group_used,    false);

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
        GGML_ASSERT(hparams.n_expert_groups < hparams.n_expert);
        if (hparams.n_expert_groups > 1) {
            GGML_ASSERT(hparams.n_expert % hparams.n_expert_groups == 0);
            GGML_ASSERT(hparams.n_group_used > 0);
            GGML_ASSERT(hparams.n_group_used < hparams.n_expert_groups);
        }
    } else {
        GGML_ASSERT(hparams.n_expert_used == 0);
        GGML_ASSERT(hparams.n_expert_groups == 0);
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

    std::fill(hparams.xielu_alpha_n.begin(), hparams.xielu_alpha_n.end(), 0.0f);
    std::fill(hparams.xielu_alpha_p.begin(), hparams.xielu_alpha_p.end(), 0.0f);
    std::fill(hparams.xielu_beta.begin(), hparams.xielu_beta.end(), 0.0f);
    std::fill(hparams.xielu_eps.begin(), hparams.xielu_eps.end(), 0.0f);

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

                const bool found_swa = ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW, hparams.n_swa, false);
                if (found_swa && hparams.n_swa == 0) {
                    hparams.swa_type             = LLAMA_SWA_TYPE_NONE;
                    hparams.n_no_rope_layer_step = hparams.n_layer; // always use rope
                } else {
                    hparams.swa_type                = LLAMA_SWA_TYPE_CHUNKED;
                    hparams.n_swa                   = 8192;
                    hparams.n_attn_temp_floor_scale = 8192;
                    hparams.f_attn_temp_scale       = 0.1f;
                    hparams.f_attn_temp_offset      = 1.0f;
                    hparams.set_swa_pattern(4);   // pattern: 3 chunked - 1 full
                }

                switch (hparams.n_expert) {
                    case 0: {
                        // MobileLLM (no MoE)
                        switch (hparams.n_embd) {
                            case 2048: type = LLM_TYPE_140M; break;
                            case 4096: type = LLM_TYPE_360M; break;
                            case 6144: type = LLM_TYPE_950M; break;
                            default:   type = LLM_TYPE_UNKNOWN;
                        }
                    } break;
                    case 16:  type = LLM_TYPE_17B_16E; break;
                    case 128: type = LLM_TYPE_17B_128E; break;
                    default:  type = LLM_TYPE_UNKNOWN;
                }

                hparams.use_kq_norm = type != LLM_TYPE_17B_128E;
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
        case LLM_ARCH_AFMOE:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                ml.get_key(LLM_KV_LEADING_DENSE_BLOCK_COUNT,   hparams.n_layer_dense_lead);
                ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,  hparams.n_ff_exp);
                ml.get_key(LLM_KV_EXPERT_SHARED_COUNT,         hparams.n_expert_shared);
                ml.get_key(LLM_KV_EXPERT_GATING_FUNC,          hparams.expert_gating_func, false);
                ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE,        hparams.expert_weights_scale, false);
                ml.get_key(LLM_KV_EXPERT_WEIGHTS_NORM,         hparams.expert_weights_norm, false);
                ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW,    hparams.n_swa, false);

                // Set up interleaved sliding window attention (ISWA)
                // Pattern: 3 sliding - 1 full (global_attn_every_n_layers = 4)
                if (hparams.n_swa > 0) {
                    hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;
                    hparams.set_swa_pattern(4);
                } else {
                    hparams.swa_type = LLAMA_SWA_TYPE_NONE;
                }

                // Default to sigmoid if not set
                if (hparams.expert_gating_func == LLAMA_EXPERT_GATING_FUNC_TYPE_NONE) {
                    hparams.expert_gating_func = LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID;
                }

                switch (hparams.n_layer) {
                    case 56: type = LLM_TYPE_6B; break;
                    case 32: type = LLM_TYPE_26B; break;
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
                // Backward-compatible defaults for older MiniCPM GGUFs
                hparams.f_embedding_scale = 12.0f;
                hparams.f_residual_scale  = 1.4f / sqrtf(float(hparams.n_layer));
                hparams.f_logit_scale     = hparams.n_embd ? (256.0f / float(hparams.n_embd)) : 1.0f;

                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

                // Optional KV reads, override defaults if present in newer GGUF exports
                ml.get_key(LLM_KV_EMBEDDING_SCALE, hparams.f_embedding_scale, /*required=*/false);
                ml.get_key(LLM_KV_RESIDUAL_SCALE, hparams.f_residual_scale, /*required=*/false);
                ml.get_key(LLM_KV_LOGIT_SCALE, hparams.f_logit_scale, /*required=*/false);

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
                // defaults for old GGUFs
                hparams.yarn_beta_fast = 8.0f;
                hparams.f_logit_scale = 0.5773502691896257f;
                hparams.f_embedding_scale = 78.38367176906169f;
                hparams.f_attn_out_scale = 0.08838834764831845f;
                hparams.f_attn_logit_softcapping = 30.0f;
                hparams.f_router_logit_softcapping = 30.0f;
                // no final_logit_softcapping in grok-1
                hparams.f_final_logit_softcapping = 0.0f;

                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,  hparams.f_norm_rms_eps);
                ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,   hparams.n_ff_exp, false);
                ml.get_key(LLM_KV_LOGIT_SCALE,                  hparams.f_logit_scale, false);
                ml.get_key(LLM_KV_EMBEDDING_SCALE,              hparams.f_embedding_scale, false);
                ml.get_key(LLM_KV_ATTENTION_OUTPUT_SCALE,       hparams.f_attn_out_scale, false);
                ml.get_key(LLM_KV_ATTN_LOGIT_SOFTCAPPING,       hparams.f_attn_logit_softcapping, false);
                ml.get_key(LLM_KV_ROUTER_LOGIT_SOFTCAPPING,     hparams.f_router_logit_softcapping, false);
                ml.get_key(LLM_KV_FINAL_LOGIT_SOFTCAPPING,      hparams.f_final_logit_softcapping, false);

                ml.get_key(LLM_KV_ATTENTION_TEMPERATURE_LENGTH,  hparams.attn_temp_length, false);
                ml.get_key(LLM_KV_ROPE_SCALING_YARN_EXT_FACTOR,  hparams.yarn_ext_factor, false);
                ml.get_key(LLM_KV_ROPE_SCALING_YARN_ATTN_FACTOR, hparams.yarn_attn_factor, false);
                ml.get_key(LLM_KV_ROPE_SCALING_YARN_BETA_FAST,   hparams.yarn_beta_fast, false);
                ml.get_key(LLM_KV_ROPE_SCALING_YARN_BETA_SLOW,   hparams.yarn_beta_slow, false);

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
        case LLM_ARCH_JINA_BERT_V3:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS,    hparams.f_norm_eps);
                ml.get_key(LLM_KV_ATTENTION_CAUSAL,           hparams.causal_attn);
                ml.get_key(LLM_KV_POOLING_TYPE,               hparams.pooling_type, false);

                switch (hparams.n_layer) {
                    case 24:
                        type = LLM_TYPE_558M; break;
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
        case LLM_ARCH_LLADA_MOE:
            {
                ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp, false);

                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                // diffusion language model uses non-causal attention
                hparams.causal_attn = false;
                switch (hparams.n_layer) {
                    case 16: type = LLM_TYPE_A1_7B; break;
                    default: type = LLM_TYPE_UNKNOWN;
                }
            } break;
        case LLM_ARCH_RND1:
            {
                ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp, false);

                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                switch (hparams.n_layer) {
                    case 48: type = LLM_TYPE_30B_A3B; break;
                    default: type = LLM_TYPE_UNKNOWN;
                }
                // Set non-causal attention for diffusion models
                hparams.causal_attn = false;
            } break;
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
        case LLM_ARCH_QWEN3VL:
            {
                ml.get_key(LLM_KV_NUM_DEEPSTACK_LAYERS, hparams.n_deepstack_layers, false);
                ml.get_key_or_arr(LLM_KV_ROPE_DIMENSION_SECTIONS, hparams.rope_sections, 4, true);
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                switch (hparams.n_layer) {
                    case 28: type = LLM_TYPE_1_7B; break;
                    case 36: type = hparams.n_embd == 2560 ? LLM_TYPE_4B : LLM_TYPE_8B; break;
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
        case LLM_ARCH_QWEN3VLMOE:
            {
                ml.get_key(LLM_KV_NUM_DEEPSTACK_LAYERS, hparams.n_deepstack_layers, false);
                ml.get_key_or_arr(LLM_KV_ROPE_DIMENSION_SECTIONS, hparams.rope_sections, 4, true);
                ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp, false);
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

                // Load attention parameters
                ml.get_key(LLM_KV_ATTENTION_KEY_LENGTH,   hparams.n_embd_head_k, false);
                ml.get_key(LLM_KV_ATTENTION_VALUE_LENGTH, hparams.n_embd_head_v, false);
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
                const bool found_swa = ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW, hparams.n_swa, false);
                if (found_swa && hparams.n_swa > 0) {
                    hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;
                    hparams.set_swa_pattern(6);

                    hparams.rope_freq_base_train_swa  = 10000.0f;
                    hparams.rope_freq_scale_train_swa = 1.0f;
                } else {
                    hparams.swa_type = LLAMA_SWA_TYPE_NONE;
                }

                hparams.f_final_logit_softcapping = 0.0f;
                ml.get_key(LLM_KV_FINAL_LOGIT_SOFTCAPPING, hparams.f_final_logit_softcapping, false);
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

                switch (hparams.n_layer) {
                    case 18: type = LLM_TYPE_270M; break;
                    case 26: type = LLM_TYPE_1B; break;
                    case 32: type = LLM_TYPE_8B; break; // Rnj-1
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

                hparams.n_layer_kv_from_start     = 20;
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
        case LLM_ARCH_GEMMA_EMBEDDING:
            {
                hparams.swa_type = LLAMA_SWA_TYPE_SYMMETRIC;
                hparams.set_swa_pattern(6);

                hparams.causal_attn = false; // embeddings do not use causal attention
                hparams.rope_freq_base_train_swa = 10000.0f;
                hparams.rope_freq_scale_train_swa = 1.0f;

                ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW, hparams.n_swa);
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                ml.get_key(LLM_KV_POOLING_TYPE, hparams.pooling_type);

                //applied only if model converted with --sentence-transformers-dense-modules
                ml.get_key(LLM_KV_DENSE_2_FEAT_IN, hparams.dense_2_feat_in, false);
                ml.get_key(LLM_KV_DENSE_2_FEAT_OUT, hparams.dense_2_feat_out, false);
                ml.get_key(LLM_KV_DENSE_3_FEAT_IN, hparams.dense_3_feat_in, false);
                ml.get_key(LLM_KV_DENSE_3_FEAT_OUT, hparams.dense_3_feat_out, false);

                GGML_ASSERT((hparams.dense_2_feat_in == 0 || hparams.dense_2_feat_in == hparams.n_embd) && "dense_2_feat_in must be equal to n_embd");
                GGML_ASSERT((hparams.dense_3_feat_out == 0 || hparams.dense_3_feat_out == hparams.n_embd) && "dense_3_feat_out must be equal to n_embd");

                switch (hparams.n_layer) {
                    case 24: type = LLM_TYPE_0_3B; break;
                    default: type = LLM_TYPE_UNKNOWN;
                }
                hparams.f_attention_scale = 1.0f / std::sqrt(float(hparams.n_embd_head_k));

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

                const bool found_swa = ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW, hparams.n_swa, false);
                if (found_swa && hparams.n_swa > 0) {
                    hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;
                    hparams.set_swa_pattern(4);
                } else {
                    hparams.swa_type = LLAMA_SWA_TYPE_NONE;
                }

                switch (hparams.n_layer) {
                    case 16: type = LLM_TYPE_1B; break;
                    case 32: type = LLM_TYPE_7B; break;
                    case 40: type = LLM_TYPE_13B; break;
                    case 64: type = LLM_TYPE_32B; break;
                    default: type = LLM_TYPE_UNKNOWN;
                }
            } break;
        case LLM_ARCH_SEED_OSS:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                switch (hparams.n_layer) {
                    case 64: type = LLM_TYPE_36B; break;
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

                switch (hparams.n_ff_exp) {
                    case 1408: type = LLM_TYPE_16B; break;
                    case 1792: type = LLM_TYPE_20B; break;
                    default: type = LLM_TYPE_UNKNOWN;
                }
            } break;
        case LLM_ARCH_DEEPSEEK2:
            {
                // lite variants include DeepSeek-V2-Lite, GigaChat3-10B-A1.8B
                bool is_lite = (hparams.n_layer == 27 || hparams.n_layer == 26);
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

                if (ml.get_key(LLM_KV_ROPE_SCALING_YARN_LOG_MUL, hparams.rope_yarn_log_mul, 0.0f)) {
                    // [TAG_DEEPSEEK2_YARN_LOG_MUL_FIX]
                    // cancel the factor from the convert script
                    hparams.rope_yarn_log_mul /= 0.1f;
                }

                // (optional) temperature tuning - used by mistral-large
                ml.get_key(LLM_KV_ATTENTION_TEMPERATURE_SCALE,  hparams.f_attn_temp_scale,       false);
                ml.get_key(LLM_KV_ATTENTION_TEMPERATURE_LENGTH, hparams.n_attn_temp_floor_scale, false);

                hparams.f_attn_temp_offset = 0.0f;

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
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,    hparams.f_norm_rms_eps);
                ml.get_key_or_arr(LLM_KV_ROPE_DIMENSION_SECTIONS, hparams.rope_sections, 4, false);
                switch (hparams.n_layer) {
                    case 40: type = LLM_TYPE_9B; break;
                    case 61: type = LLM_TYPE_32B; break;
                    default: type = LLM_TYPE_UNKNOWN;
                }
            } break;
        case LLM_ARCH_GLM4_MOE:
            {
                ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,     hparams.n_ff_exp);
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,    hparams.f_norm_rms_eps);
                ml.get_key_or_arr(LLM_KV_ROPE_DIMENSION_SECTIONS, hparams.rope_sections, 4, false);

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
                    hparams.expert_gating_func =  LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID;
                }

                // NextN/MTP parameters
                ml.get_key(LLM_KV_NEXTN_PREDICT_LAYERS,        hparams.nextn_predict_layers, false);

                // TODO: when MTP is implemented, this should probably be updated if needed
                hparams.n_layer_kv_from_start = hparams.n_layer - hparams.nextn_predict_layers;

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

                hparams.dec_n_layer = hparams.n_layer;
                ml.get_key(LLM_KV_DECODER_BLOCK_COUNT, hparams.dec_n_layer, false);

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
        case LLM_ARCH_NEMOTRON_H:
        case LLM_ARCH_NEMOTRON_H_MOE:
            {
                ml.get_key(LLM_KV_SSM_CONV_KERNEL,    hparams.ssm_d_conv);
                ml.get_key(LLM_KV_SSM_INNER_SIZE,     hparams.ssm_d_inner);
                ml.get_key(LLM_KV_SSM_STATE_SIZE,     hparams.ssm_d_state);
                ml.get_key(LLM_KV_SSM_TIME_STEP_RANK, hparams.ssm_dt_rank);
                ml.get_key(LLM_KV_SSM_GROUP_COUNT,    hparams.ssm_n_group);

                // A layer is recurrent IFF the n_head_kv value is set to 0 and
                // the n_ff value is set to 0
                for (uint32_t i = 0; i < hparams.n_layer; ++i) {
                    hparams.recurrent_layer_arr[i] = (hparams.n_head_kv(i) == 0 && hparams.n_ff(i) == 0);
                }

                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

                ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,        hparams.n_ff_exp,        false);
                ml.get_key(LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH, hparams.n_ff_shexp,      false);
                ml.get_key(LLM_KV_EXPERT_SHARED_COUNT,               hparams.n_expert_shared, false);
                ml.get_key(LLM_KV_EXPERT_WEIGHTS_NORM,               hparams.expert_weights_norm, false);
                ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE,              hparams.expert_weights_scale, false);

                switch (hparams.n_layer) {
                    case 52: type = LLM_TYPE_31B_A3_5B; break; // Nemotron-H_MOE 31B
                    case 56: type = LLM_TYPE_9B; break;
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

                switch (hparams.n_embd) {
                    case 768: type = LLM_TYPE_350M; break;
                    case 1536: type = (hparams.n_embd == 2048 ? LLM_TYPE_7B_A1B : LLM_TYPE_1B); break;
                    case 2048: case 2560: type = LLM_TYPE_3B; break;
                    case 4096: type = LLM_TYPE_32B; break;
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
        case LLM_ARCH_BAILINGMOE2:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,       hparams.f_norm_rms_eps);
                ml.get_key(LLM_KV_LEADING_DENSE_BLOCK_COUNT,         hparams.n_layer_dense_lead);
                ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,        hparams.n_ff_exp);
                ml.get_key(LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH, hparams.n_ff_shexp);
                ml.get_key(LLM_KV_EXPERT_SHARED_COUNT,               hparams.n_expert_shared);
                ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE,              hparams.expert_weights_scale);
                ml.get_key(LLM_KV_EXPERT_WEIGHTS_NORM,               hparams.expert_weights_norm, false);
                ml.get_key(LLM_KV_EXPERT_GATING_FUNC,                hparams.expert_gating_func);
                ml.get_key(LLM_KV_NEXTN_PREDICT_LAYERS,              hparams.nextn_predict_layers, false);

                // TODO: when MTP is implemented, this should probably be updated if needed
                hparams.n_layer_kv_from_start = hparams.n_layer - hparams.nextn_predict_layers;

                switch (hparams.n_layer) {
                    case 20: type = LLM_TYPE_16B_A1B; break;
                    case 21: type = LLM_TYPE_16B_A1B; break;
                    case 32: type = LLM_TYPE_100B_A6B; break;
                    case 33: type = LLM_TYPE_100B_A6B; break;
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

                switch (hparams.n_layer) {
                    case 24: type = LLM_TYPE_20B; break;
                    case 36: type = LLM_TYPE_120B; break;
                    default: type = LLM_TYPE_UNKNOWN;
                }
            } break;
        case LLM_ARCH_LFM2:
            {
                ml.get_key(LLM_KV_SHORTCONV_L_CACHE,           hparams.n_shortconv_l_cache);
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                for (uint32_t il = 0; il < hparams.n_layer; ++il) {
                    hparams.recurrent_layer_arr[il] = hparams.n_head_kv(il) == 0;
                }
                hparams.n_layer_dense_lead = hparams.n_layer;
                switch (hparams.n_ff()) {
                    case  4608: type = LLM_TYPE_350M; break;
                    case  6912: type = LLM_TYPE_700M; break;
                    case  8192: type = LLM_TYPE_1_2B; break;
                    case 10752: type = LLM_TYPE_2_6B; break;
                    default:    type = LLM_TYPE_UNKNOWN;
                }
            } break;
        case LLM_ARCH_LFM2MOE:
            {
                ml.get_key(LLM_KV_SHORTCONV_L_CACHE,           hparams.n_shortconv_l_cache);
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                ml.get_key(LLM_KV_LEADING_DENSE_BLOCK_COUNT,   hparams.n_layer_dense_lead);
                ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,  hparams.n_ff_exp);
                ml.get_key(LLM_KV_EXPERT_GATING_FUNC,          hparams.expert_gating_func);

                for (uint32_t il = 0; il < hparams.n_layer; ++il) {
                    hparams.recurrent_layer_arr[il] = hparams.n_head_kv(il) == 0;
                }

                type = LLM_TYPE_8B_A1B;
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
        case LLM_ARCH_GROVEMOE:
            {
                ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,        hparams.n_ff_exp);
                ml.get_key(LLM_KV_EXPERT_CHUNK_FEED_FORWARD_LENGTH,  hparams.n_ff_chexp);
                ml.get_key(LLM_KV_EXPERT_GROUP_SCALE,                hparams.expert_group_scale);
                ml.get_key(LLM_KV_EXPERTS_PER_GROUP,                 hparams.n_group_experts);
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,       hparams.f_norm_rms_eps);

                switch (hparams.n_layer) {
                    case 48: type = LLM_TYPE_30B_A3B; break;
                    default: type = LLM_TYPE_UNKNOWN;
                }
            } break;
        case LLM_ARCH_APERTUS:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                ml.get_key_or_arr(LLM_KV_XIELU_ALPHA_N,        hparams.xielu_alpha_n, hparams.n_layer);
                ml.get_key_or_arr(LLM_KV_XIELU_ALPHA_P,        hparams.xielu_alpha_p, hparams.n_layer);
                ml.get_key_or_arr(LLM_KV_XIELU_BETA,           hparams.xielu_beta,    hparams.n_layer);
                ml.get_key_or_arr(LLM_KV_XIELU_EPS,            hparams.xielu_eps,     hparams.n_layer);

                switch (hparams.n_layer) {
                    case 32: type = LLM_TYPE_8B; break;
                    default: type = LLM_TYPE_UNKNOWN;
                }
            } break;
        case LLM_ARCH_MINIMAX_M2:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,  hparams.f_norm_rms_eps);
                ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,   hparams.n_ff_exp);
                ml.get_key(LLM_KV_EXPERT_GATING_FUNC,           hparams.expert_gating_func, false);

                switch (hparams.n_layer) {
                    case 62: type = LLM_TYPE_230B_A10B; break;
                    default: type = LLM_TYPE_UNKNOWN;
                }
            } break;
        case LLM_ARCH_COGVLM:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                switch (hparams.n_layer) {
                    case 32: type = LLM_TYPE_13B; break;
                    default: type = LLM_TYPE_UNKNOWN;
                }
            } break;
        case LLM_ARCH_PANGU_EMBED:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                switch (hparams.n_layer) {
                    case 26: type = LLM_TYPE_1B; break; // openPangu-Embedded-1B-V1.1
                    case 34: type = LLM_TYPE_7B; break; // openPangu-Embedded-7B-V1.1
                    default: type = LLM_TYPE_UNKNOWN;
                }
            } break;
        case LLM_ARCH_QWEN3NEXT:
            {
                ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,        hparams.n_ff_exp, false);
                ml.get_key(LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH, hparams.n_ff_shexp, false);
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,       hparams.f_norm_rms_eps);

                // Load linear attention (gated delta net) parameters
                ml.get_key(LLM_KV_SSM_CONV_KERNEL,    hparams.ssm_d_conv);
                ml.get_key(LLM_KV_SSM_INNER_SIZE,     hparams.ssm_d_inner);
                ml.get_key(LLM_KV_SSM_STATE_SIZE,     hparams.ssm_d_state);
                ml.get_key(LLM_KV_SSM_TIME_STEP_RANK, hparams.ssm_dt_rank);
                ml.get_key(LLM_KV_SSM_GROUP_COUNT,    hparams.ssm_n_group);

                // Mark recurrent layers (linear attention layers)
                for (uint32_t i = 0; i < hparams.n_layer; ++i) {
                    hparams.recurrent_layer_arr[i] = ((i + 1) % 4 != 0); // TODO: extract the magic 4 from "full_attention_interval"
                }

                switch (hparams.n_layer) {
                    case 48: type = LLM_TYPE_80B_A3B; break;
                    default: type = LLM_TYPE_UNKNOWN;
                }
            } break;
        case LLM_ARCH_MISTRAL3:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                ml.get_key(LLM_KV_ATTENTION_TEMPERATURE_SCALE, hparams.f_attn_temp_scale, false);

                ml.get_key(LLM_KV_ROPE_SCALING_YARN_BETA_FAST, hparams.yarn_beta_fast,    false);
                ml.get_key(LLM_KV_ROPE_SCALING_YARN_BETA_SLOW, hparams.yarn_beta_slow,    false);
                ml.get_key(LLM_KV_ROPE_SCALING_YARN_LOG_MUL,   hparams.rope_yarn_log_mul, 0.0f);

                hparams.f_attn_temp_offset = 0.0f;

                // TODO: maybe add n_attn_temp_floor_scale as a separate KV?
                if (hparams.f_attn_temp_scale != 0.0f) {
                    hparams.n_attn_temp_floor_scale = hparams.n_ctx_orig_yarn;
                    if (hparams.n_attn_temp_floor_scale == 0) {
                        throw std::runtime_error("invalid n_ctx_orig_yarn for attention temperature scaling");
                    }
                }

                switch (hparams.n_layer) {
                    case 26: type = LLM_TYPE_3B; break;
                    case 34: type = LLM_TYPE_8B; break;
                    case 40: type = LLM_TYPE_14B; break;
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
    pimpl->cpu_buft_list = make_cpu_buft_list(devices, params.use_extra_bufts, params.no_host);
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

    // define a comparator for the buft -> ctx map to ensure that the order is well-defined:
    struct ggml_backend_buft_comparator {
        bool operator()(const ggml_backend_buffer_type_t & lhs, const ggml_backend_buffer_type_t & rhs) const {
            return strcmp(ggml_backend_buft_name(lhs), ggml_backend_buft_name(rhs)) < 0;
        }
    };
    std::map<ggml_backend_buffer_type_t, ggml_context_ptr, ggml_backend_buft_comparator> ctx_map;

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

            ctx_map.emplace(buft, ctx);

            return ctx;
        }
        return it->second.get();
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
            case LLM_ARCH_MISTRAL3:
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
            case LLM_ARCH_LLADA_MOE:
                {
                    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

                    // output
                    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
                    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, 0);

                    GGML_ASSERT(n_expert > 0 && "n_expert must be > 0 for llada-moe");
                    GGML_ASSERT(n_expert_used > 0 && "n_expert_used must be > 0 for llada-moe");

                    for (int i = 0; i < n_layer; ++i) {
                        auto & layer = layers[i];

                        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

                        layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_embd}, 0);
                        layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_embd_gqa}, 0);
                        layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_gqa}, 0);
                        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd}, 0);
                        layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), {n_embd_head_k}, 0);
                        layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), {n_embd_head_k}, 0);

                        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);

                        layer.ffn_gate_inp = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP, "weight", i), {n_embd, n_expert}, 0);

                        const int64_t n_ff_exp = hparams.n_ff_exp ? hparams.n_ff_exp : n_ff / n_expert_used;

                        layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {  n_embd, n_ff_exp, n_expert}, 0);
                        layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {n_ff_exp,   n_embd, n_expert}, 0);
                        layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {  n_embd, n_ff_exp, n_expert}, 0);
                    }
                } break;
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

                    for (int i = 0; i < n_layer; ++i) {
                        bool is_moe_layer = hparams.n_moe_layer_step > 0 && (i + 1) % hparams.n_moe_layer_step == 0;

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

                    const int64_t n_ff_exp = hparams.n_ff_exp ? hparams.n_ff_exp : n_ff/* / n_expert_used*/; // grok-1 n_ff_exp == n_ff
                    for (int i = 0; i < n_layer; ++i) {
                        auto & layer = layers[i];

                        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

                        layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_embd}, 0);
                        layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_embd_gqa}, 0);
                        layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_gqa}, 0);
                        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd}, 0);

                        layer.attn_out_norm   = create_tensor(tn(LLM_TENSOR_ATTN_OUT_NORM, "weight", i), {n_embd}, 0);

                        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);

                        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd, n_ff}, TENSOR_NOT_REQUIRED);
                        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff,   n_embd}, TENSOR_NOT_REQUIRED);
                        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd, n_ff}, TENSOR_NOT_REQUIRED);

                        layer.ffn_gate_inp  = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,  "weight", i), {n_embd, n_expert}, 0);
                        layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {n_embd,   n_ff_exp, n_expert}, TENSOR_NOT_REQUIRED);
                        layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {n_ff_exp, n_embd,   n_expert}, 0);
                        layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {n_embd,   n_ff_exp, n_expert}, 0);

                        layer.ffn_post_norm = create_tensor(tn(LLM_TENSOR_LAYER_OUT_NORM, "weight", i), {n_embd}, TENSOR_NOT_REQUIRED);
                        if (!layer.ffn_post_norm) {
                            layer.ffn_post_norm = create_tensor(tn(LLM_TENSOR_FFN_POST_NORM, "weight", i), {n_embd}, 0);
                        }
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
            case LLM_ARCH_JINA_BERT_V3:
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
                        layer.bo = create_tensor(tn(LLM_TENSOR_ATTN_OUT,      "bias", i),   {n_embd}, TENSOR_NOT_REQUIRED);

                        layer.attn_out_norm   = create_tensor(tn(LLM_TENSOR_ATTN_OUT_NORM, "weight", i), {n_embd}, 0);
                        layer.attn_out_norm_b = create_tensor(tn(LLM_TENSOR_ATTN_OUT_NORM, "bias", i),   {n_embd}, 0);

                        if (hparams.moe_every_n_layers > 0 && i % hparams.moe_every_n_layers == 1) {
                            layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {  n_embd, n_ff,   n_expert}, 0);
                            layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {  n_ff,   n_embd, n_expert}, 0);
                            layer.ffn_gate_inp = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,   "weight", i), {n_embd, n_expert}, 0);
                        } else {
                            layer.ffn_up     = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd, n_ff}, 0);
                            layer.ffn_up_b   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "bias", i),   {n_ff}, TENSOR_NOT_REQUIRED);
                            layer.ffn_down   = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff, n_embd}, 0);
                            layer.ffn_down_b = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "bias", i),   {n_embd}, TENSOR_NOT_REQUIRED);

                            if (arch == LLM_ARCH_NOMIC_BERT) {
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
                        layer.bq = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "bias", i), {n_embd}, TENSOR_NOT_REQUIRED);
                        layer.bk = create_tensor(tn(LLM_TENSOR_ATTN_K,   "bias", i), {n_embd_gqa}, TENSOR_NOT_REQUIRED);
                        layer.bv = create_tensor(tn(LLM_TENSOR_ATTN_V,   "bias", i), {n_embd_gqa}, TENSOR_NOT_REQUIRED);

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
            case LLM_ARCH_QWEN3VL:
                {
                    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

                    // output
                    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
                    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);
                    // if output is NULL, init from the input tok embed
                    if (output == NULL) {
                        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
                    }

                    // output rerank head
                    cls_out = create_tensor(tn(LLM_TENSOR_CLS_OUT, "weight"), {n_embd, hparams.n_cls_out}, TENSOR_NOT_REQUIRED);

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
            case LLM_ARCH_QWEN3VLMOE:
            case LLM_ARCH_RND1:
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
                    // mamba parameters
                    const uint32_t d_conv             = hparams.ssm_d_conv;
                    const uint32_t d_state            = hparams.ssm_d_state;
                    const uint32_t num_heads          = hparams.ssm_dt_rank;
                    const uint32_t intermediate_size  = hparams.ssm_d_inner;
                    const int64_t dt_dim              = std::max(64, int(hparams.n_embd / 16));

                    // attention parameters
                    const uint32_t qk_dim = hparams.n_embd_head_k;
                    const uint32_t v_dim  = hparams.n_embd_head_v;

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
                            const int64_t num_attention_heads = hparams.n_head(i);
                            const int64_t q_num_heads         = num_attention_heads;
                            const int64_t num_key_value_heads = hparams.n_head_kv(i);
                            const int64_t k_num_heads         = num_key_value_heads;
                            const int64_t v_num_heads         = num_key_value_heads;
                            const int64_t q_proj_dim          = q_num_heads * qk_dim;
                            const int64_t k_proj_dim          = k_num_heads * qk_dim;
                            const int64_t v_proj_dim          = v_num_heads * v_dim;

                            layer.wqkv = create_tensor(tn(LLM_TENSOR_ATTN_QKV, "weight", i), {n_embd, q_proj_dim + k_proj_dim + v_proj_dim}, 0);
                            layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), {qk_dim, num_attention_heads}, 0);
                            layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), {qk_dim, k_num_heads}, 0);
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
            case LLM_ARCH_GEMMA_EMBEDDING:
                {
                    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

                    // output
                    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
                    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);

                    // if output is NULL, init from the input tok embed
                    if (output == NULL) {
                        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD,   "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
                    }

                    // Dense linear weights
                    dense_2_out_layers = create_tensor(tn(LLM_TENSOR_DENSE_2_OUT, "weight"), {n_embd, hparams.dense_2_feat_out}, TENSOR_NOT_REQUIRED);
                    dense_3_out_layers = create_tensor(tn(LLM_TENSOR_DENSE_3_OUT, "weight"), {hparams.dense_3_feat_in, n_embd}, TENSOR_NOT_REQUIRED);


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
            case LLM_ARCH_SEED_OSS:
                {
                    const uint32_t head_dim             = hparams.n_embd_head_k;
                    const int64_t n_qo_dim              = n_head * head_dim;
                    const int64_t n_kv_dim              = n_head_kv * head_dim;

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

                        layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_qo_dim}, 0);
                        layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_kv_dim}, 0);
                        layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_kv_dim}, 0);
                        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_qo_dim, n_embd}, 0);

                        layer.bq = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "bias", i), {n_qo_dim},   TENSOR_NOT_REQUIRED);
                        layer.bk = create_tensor(tn(LLM_TENSOR_ATTN_K,   "bias", i), {n_kv_dim},   TENSOR_NOT_REQUIRED);
                        layer.bv = create_tensor(tn(LLM_TENSOR_ATTN_V,   "bias", i), {n_kv_dim},   TENSOR_NOT_REQUIRED);

                        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);
                        layer.attn_post_norm = create_tensor(tn(LLM_TENSOR_ATTN_POST_NORM, "weight", i), {n_embd}, 0);

                        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff}, 0);
                        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff}, 0);
                        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd}, 0);
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
                    // lite variants include DeepSeek-V2-Lite, GigaChat3-10B-A1.8B
                    const bool is_lite = (hparams.n_layer == 27 || hparams.n_layer == 26);

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

                    // n_layer:     number of encoder_layers
                    // dec_n_layer: number of decoder_layers
                    const int dec_n_layer = hparams.dec_n_layer;
                    if (dec_n_layer > n_layer) {
                        layers.resize(dec_n_layer);
                    }

                    // load encoder layers
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

                    // load decoder layers
                    for (int i = 0; i < dec_n_layer; ++i) {
                        auto & layer = layers[i];

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
                            layer.nextn.enorm            = create_tensor(tn(LLM_TENSOR_NEXTN_ENORM, "weight", i), { n_embd }, flags);
                            layer.nextn.hnorm            = create_tensor(tn(LLM_TENSOR_NEXTN_HNORM, "weight", i), { n_embd }, flags);

                            // Optional tensors
                            layer.nextn.embed_tokens     = create_tensor(tn(LLM_TENSOR_NEXTN_EMBED_TOKENS, "weight", i), { n_embd, n_vocab }, flags | TENSOR_NOT_REQUIRED);
                            layer.nextn.shared_head_head = create_tensor(tn(LLM_TENSOR_NEXTN_SHARED_HEAD_HEAD, "weight", i), { n_embd, n_vocab }, flags | TENSOR_NOT_REQUIRED);
                            layer.nextn.shared_head_norm = create_tensor(tn(LLM_TENSOR_NEXTN_SHARED_HEAD_NORM, "weight", i), { n_embd }, flags | TENSOR_NOT_REQUIRED);
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
            case LLM_ARCH_NEMOTRON_H:
            case LLM_ARCH_NEMOTRON_H_MOE:
                {
                    // mamba2 Mixer SSM params
                    // NOTE: int64_t for tensor dimensions
                    const int64_t d_conv     = hparams.ssm_d_conv;
                    const int64_t d_inner    = hparams.ssm_d_inner;
                    const int64_t d_state    = hparams.ssm_d_state;
                    const int64_t n_ssm_head = hparams.ssm_dt_rank;
                    const int64_t n_group    = hparams.ssm_n_group;
                    const int64_t d_in_proj  = 2*d_inner + 2*n_group*d_state + n_ssm_head;

                    const int64_t n_ff_exp = hparams.n_ff_exp ? hparams.n_ff_exp : n_ff / n_expert_used;
                    const int64_t n_ff_shexp = hparams.n_ff_shexp;

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

                        // all blocks use the attn norm
                        layer.attn_norm  = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

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
                        } else if (hparams.n_ff(i) == 0) {
                            // attention layers (with optional bias)
                            const int64_t n_head_i = hparams.n_head(i);
                            const int64_t n_embd_k_gqa_i = hparams.n_embd_k_gqa(i);
                            const int64_t n_embd_v_gqa_i = hparams.n_embd_v_gqa(i);
                            layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_embd_head_k * n_head_i}, 0);
                            layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_embd_k_gqa_i}, 0);
                            layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_v_gqa_i}, 0);
                            layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head_k * n_head_i, n_embd}, 0);
                            layer.bq = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "bias",   i), {n_embd},         TENSOR_NOT_REQUIRED);
                            layer.bk = create_tensor(tn(LLM_TENSOR_ATTN_K,   "bias",   i), {n_embd_k_gqa_i}, TENSOR_NOT_REQUIRED);
                            layer.bv = create_tensor(tn(LLM_TENSOR_ATTN_V,   "bias",   i), {n_embd_v_gqa_i}, TENSOR_NOT_REQUIRED);
                            layer.bo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "bias",   i), {n_embd},         TENSOR_NOT_REQUIRED);
                        }  else {
                            if (n_expert != 0) {
                                layer.ffn_gate_inp    = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,  "weight", i), { n_embd, n_expert}, 0);
                                layer.ffn_exp_probs_b = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias", i), {n_expert         }, 0);

                                // MoE branch
                                layer.ffn_down_exps   = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {n_ff_exp,   n_embd, n_expert}, 0);
                                layer.ffn_up_exps     = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {  n_embd, n_ff_exp, n_expert}, 0);

                                // Shared expert branch
                                layer.ffn_down_shexp  = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i), {n_ff_shexp, n_embd}, 0);
                                layer.ffn_up_shexp    = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,   "weight", i), {n_embd, n_ff_shexp}, 0);

                            } else {
                                // mlp layers
                                layer.ffn_down   = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  hparams.n_ff(i), n_embd}, 0);
                                layer.ffn_up     = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   hparams.n_ff(i)}, 0);
                                layer.ffn_down_b = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "bias",   i), {n_embd}, TENSOR_NOT_REQUIRED);
                                layer.ffn_up_b   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "bias",   i), {hparams.n_ff(i)}, TENSOR_NOT_REQUIRED);
                            }
                        }
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
            case LLM_ARCH_BAILINGMOE2:
                {
                    const int64_t n_ff_exp        = hparams.n_ff_exp;
                    const int64_t n_expert_shared = hparams.n_expert_shared;

                    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

                    // output
                    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
                    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, 0);

                    GGML_ASSERT(n_expert > 0 && "n_expert must be > 0 for bailingmoe2");
                    GGML_ASSERT(n_expert_used > 0 && "n_expert_used must be > 0 for bailingmoe2");

                    for (int i = 0; i < n_layer; ++i) {
                        int flags = 0;
                        if (hparams.nextn_predict_layers > 0 && static_cast<uint32_t>(i) >= n_layer - hparams.nextn_predict_layers) {
                            // skip all tensors in the NextN layers
                            flags |= TENSOR_SKIP;
                        }

                        auto & layer = layers[i];

                        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, flags);

                        layer.wqkv = create_tensor(tn(LLM_TENSOR_ATTN_QKV, "weight", i), {n_embd, n_embd + 2*n_embd_gqa}, flags);
                        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head_k * n_head, n_embd}, flags);

                        layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), {n_embd_head_k}, flags);
                        layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), {n_embd_head_k}, flags);

                        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, flags);

                        if (static_cast<uint32_t>(i) >= hparams.n_layer_dense_lead) { // MoE layers
                            const int64_t n_ff_shexp = (hparams.n_ff_shexp ? hparams.n_ff_shexp : n_ff_exp) * n_expert_shared;

                            layer.ffn_gate_inp = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP, "weight", i), {n_embd, n_expert}, flags);
                            layer.ffn_exp_probs_b = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias", i), {n_expert}, TENSOR_NOT_REQUIRED | flags);

                            layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {  n_embd, n_ff_exp, n_expert}, flags);
                            layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {n_ff_exp,   n_embd, n_expert}, flags);
                            layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {  n_embd, n_ff_exp, n_expert}, flags);

                            layer.ffn_gate_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", i), {n_embd, n_ff_shexp}, flags);
                            layer.ffn_down_shexp = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i), {n_ff_shexp, n_embd}, flags);
                            layer.ffn_up_shexp   = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,   "weight", i), {n_embd, n_ff_shexp}, flags);
                        } else { // Dense layers
                            layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff}, flags);
                            layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd}, flags);
                            layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff}, flags);
                        }

                        // NextN/MTP tensors (preserved but unused) - conditionally load for last nextn_predict_layers
                        if (hparams.nextn_predict_layers > 0 && static_cast<uint32_t>(i) >= n_layer - hparams.nextn_predict_layers) {
                            layer.nextn.eh_proj          = create_tensor(tn(LLM_TENSOR_NEXTN_EH_PROJ, "weight", i), { 2 * n_embd, n_embd }, flags);
                            layer.nextn.embed_tokens     = create_tensor(tn(LLM_TENSOR_NEXTN_EMBED_TOKENS, "weight", i), { n_embd, n_vocab }, TENSOR_NOT_REQUIRED | flags);
                            layer.nextn.enorm            = create_tensor(tn(LLM_TENSOR_NEXTN_ENORM, "weight", i), { n_embd }, flags);
                            layer.nextn.hnorm            = create_tensor(tn(LLM_TENSOR_NEXTN_HNORM, "weight", i), { n_embd }, flags);
                            layer.nextn.shared_head_head = create_tensor(tn(LLM_TENSOR_NEXTN_SHARED_HEAD_HEAD, "weight", i), { n_embd, n_vocab }, TENSOR_NOT_REQUIRED | flags);
                            layer.nextn.shared_head_norm = create_tensor(tn(LLM_TENSOR_NEXTN_SHARED_HEAD_NORM, "weight", i), { n_embd }, TENSOR_NOT_REQUIRED | flags);
                            layer.layer_out_norm         = create_tensor(tn(LLM_TENSOR_LAYER_OUT_NORM, "weight", i), {n_embd}, flags);
                        }
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
            case LLM_ARCH_AFMOE:
                {
                    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

                    // output
                    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
                    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);

                    // if output is NULL, init from the input tok embed
                    if (output == NULL) {
                        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
                    }

                    const int64_t n_ff_exp = hparams.n_ff_exp;
                    const int64_t n_expert_shared = hparams.n_expert_shared;

                    for (int i = 0; i < n_layer; ++i) {
                        auto & layer = layers[i];

                        // dual attention normalization
                        layer.attn_norm      = create_tensor(tn(LLM_TENSOR_ATTN_NORM,      "weight", i), {n_embd}, 0);
                        layer.attn_post_norm = create_tensor(tn(LLM_TENSOR_ATTN_POST_NORM, "weight", i), {n_embd}, 0);

                        // attention projections
                        layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_embd_head_k * n_head}, 0);
                        layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_embd_k_gqa}, 0);
                        layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_v_gqa}, 0);
                        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head_k * n_head, n_embd}, 0);

                        // Q/K normalization
                        layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), {n_embd_head_k}, 0);
                        layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), {n_embd_head_k}, 0);

                        // attention gating
                        layer.wqkv_gate = create_tensor(tn(LLM_TENSOR_ATTN_GATE, "weight", i), {n_embd, n_embd_head_k * n_head}, 0);

                        // dual ffn normalization
                        layer.ffn_norm      = create_tensor(tn(LLM_TENSOR_FFN_NORM,      "weight", i), {n_embd}, 0);
                        layer.ffn_post_norm = create_tensor(tn(LLM_TENSOR_FFN_POST_NORM, "weight", i), {n_embd}, 0);

                        if (static_cast<uint32_t>(i) >= hparams.n_layer_dense_lead) {
                            // MoE layers
                            layer.ffn_gate_inp = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP, "weight", i), {n_embd, n_expert}, 0);
                            layer.ffn_exp_probs_b = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias", i), {n_expert}, 0);

                            // grouped expert weights
                            layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {n_embd, n_ff_exp, n_expert}, 0);
                            layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {n_ff_exp, n_embd, n_expert}, 0);
                            layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {n_embd, n_ff_exp, n_expert}, 0);

                            // shared expert
                            if (n_expert_shared > 0) {
                                const int64_t n_ff_shexp = n_ff_exp * n_expert_shared;
                                layer.ffn_gate_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", i), {n_embd, n_ff_shexp}, 0);
                                layer.ffn_down_shexp = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i), {n_ff_shexp, n_embd}, 0);
                                layer.ffn_up_shexp   = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,   "weight", i), {n_embd, n_ff_shexp}, 0);
                            }
                        } else {
                            // Dense layers
                            layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd, n_ff}, 0);
                            layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff, n_embd}, 0);
                            layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd, n_ff}, 0);
                        }
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
            case LLM_ARCH_LFM2MOE:
                {
                    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

                    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
                    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);

                    if (output == NULL) {
                        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
                    }

                    for (int i = 0; i < n_layer; ++i) {
                        auto & layer = layers[i];

                        const bool is_moe_layer = i >= static_cast<int>(hparams.n_layer_dense_lead);

                        // ffn/moe is same for transformer and conv layers
                        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);
                        if (is_moe_layer) {
                            GGML_ASSERT(n_expert && n_expert_used);
                            layer.ffn_gate_inp    = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP, "weight", i),  {n_embd, n_expert}, 0);
                            layer.ffn_gate_exps   = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {n_embd, hparams.n_ff_exp, n_expert}, 0);
                            layer.ffn_down_exps   = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {hparams.n_ff_exp,   n_embd, n_expert}, 0);
                            layer.ffn_up_exps     = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS, "weight", i),   {n_embd, hparams.n_ff_exp, n_expert}, 0);
                            layer.ffn_exp_probs_b = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias", i), {n_expert}, 0);
                        } else {  // dense
                            layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff}, 0);
                            layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd}, 0);
                            layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff}, 0);
                        }

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
            case LLM_ARCH_GROVEMOE:
                {
                    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

                    // output
                    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
                    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);
                    // if output is NULL, init from the input tok embed
                    if (output == NULL) {
                        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
                    }

                    GGML_ASSERT(n_expert > 0 && "n_expert must be > 0 for GROVEMOE");
                    GGML_ASSERT(n_expert_used > 0 && "n_expert_used must be > 0 for GROVEMOE");
                    GGML_ASSERT(hparams.n_group_experts > 0 && "n_group_experts must be > 0 for GROVEMOE");

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

                        // MoE branch
                        const int64_t n_ff_exp = hparams.n_ff_exp ? hparams.n_ff_exp : n_ff / n_expert_used;
                        const int64_t n_ff_chexp = hparams.n_ff_chexp ? hparams.n_ff_chexp : n_embd_head_k;
                        const int64_t n_chunk_expert = n_expert / hparams.n_group_experts;

                        layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {  n_embd, n_ff_exp, n_expert}, 0);
                        layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {n_ff_exp,   n_embd, n_expert}, 0);
                        layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {  n_embd, n_ff_exp, n_expert}, 0);

                        layer.ffn_gate_chexps = create_tensor(tn(LLM_TENSOR_FFN_GATE_CHEXPS, "weight", i), {  n_embd, n_ff_chexp, n_chunk_expert}, 0);
                        layer.ffn_down_chexps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_CHEXPS, "weight", i), {n_ff_chexp,   n_embd, n_chunk_expert}, 0);
                        layer.ffn_up_chexps   = create_tensor(tn(LLM_TENSOR_FFN_UP_CHEXPS,   "weight", i), {  n_embd, n_ff_chexp, n_chunk_expert}, 0);
                    }
                } break;
            case LLM_ARCH_APERTUS:
                {
                    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, 0);

                    // output
                    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), { n_embd }, 0);
                    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), { n_embd, n_vocab }, 0);

                    for (int i = 0; i < n_layer; ++i) {
                        auto & layer = layers[i];

                        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), { n_embd }, 0);

                        if (hparams.rope_scaling_type_train == LLAMA_ROPE_SCALING_TYPE_LONGROPE) {
                            layer.rope_long  = create_tensor(tn(LLM_TENSOR_ROPE_FACTORS_LONG,  "weight", i), { n_rot/2 }, TENSOR_NOT_REQUIRED | (i != 0 ? TENSOR_DUPLICATED : 0));
                            layer.rope_short = create_tensor(tn(LLM_TENSOR_ROPE_FACTORS_SHORT, "weight", i), { n_rot/2 }, TENSOR_NOT_REQUIRED | (i != 0 ? TENSOR_DUPLICATED : 0));
                        } else {
                            layer.rope_freqs = create_tensor(tn(LLM_TENSOR_ROPE_FREQS, "weight", i), { n_rot/2 }, TENSOR_NOT_REQUIRED | (i != 0 ? TENSOR_DUPLICATED : 0));
                        }

                        layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "weight", i), { n_embd, n_embd_head_k * n_head }, 0);
                        layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K,   "weight", i), { n_embd, n_embd_gqa }, 0);
                        layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V,   "weight", i), { n_embd, n_embd_gqa }, 0);
                        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), { n_embd_head_k * n_head, n_embd }, 0);

                        // optional bias tensors
                        layer.bq = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "bias", i), { n_embd },     TENSOR_NOT_REQUIRED);
                        layer.bk = create_tensor(tn(LLM_TENSOR_ATTN_K,   "bias", i), { n_embd_gqa }, TENSOR_NOT_REQUIRED);
                        layer.bv = create_tensor(tn(LLM_TENSOR_ATTN_V,   "bias", i), { n_embd_gqa }, TENSOR_NOT_REQUIRED);
                        layer.bo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "bias", i), { n_embd },     TENSOR_NOT_REQUIRED);

                        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), { n_embd }, 0);
                        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), { n_ff, n_embd }, 0);
                        layer.ffn_up = create_tensor(tn(LLM_TENSOR_FFN_UP, "weight", i), { n_embd, n_ff }, 0);

                        // Q and K layernorms for Apertus
                        layer.attn_q_norm   = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), { n_embd_head_k }, 0);
                        layer.attn_q_norm_b = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "bias",   i), { n_embd_head_k }, TENSOR_NOT_REQUIRED);
                        layer.attn_k_norm   = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), { n_embd_head_k }, 0);
                        layer.attn_k_norm_b = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "bias",   i), { n_embd_head_k }, TENSOR_NOT_REQUIRED);
                    }
                } break;
            case LLM_ARCH_MINIMAX_M2:
                {
                    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

                    // output
                    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
                    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, 0);

                    for (int i = 0; i < n_layer; ++i) {
                        auto & layer = layers[i];

                        layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q, "weight", i), { n_embd, n_embd_head_k * n_head }, 0);
                        layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K, "weight", i), { n_embd, n_embd_gqa }, 0);
                        layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V, "weight", i), { n_embd, n_embd_gqa }, 0);
                        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), { n_embd_head_k * n_head, n_embd }, 0);

                        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);
                        layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), {n_embd_head_k * n_head}, 0);
                        layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), {n_embd_k_gqa}, 0);

                        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);

                        layer.ffn_gate_inp = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP, "weight", i), {n_embd, n_expert}, 0);
                        layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {n_embd, n_ff,   n_expert}, 0);
                        layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {n_ff,   n_embd, n_expert}, 0);
                        layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {n_embd, n_ff,   n_expert}, 0);
                        layer.ffn_exp_probs_b = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias", i), {n_expert}, 0);
                    }
                } break;
            case LLM_ARCH_COGVLM:
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
                        layer.wqkv = create_tensor(tn(LLM_TENSOR_ATTN_QKV, "weight", i), {n_embd, n_embd_head_k * n_head * 3}, 0);
                        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head_k * n_head, n_embd}, 0);

                        layer.visexp_attn_wqkv = create_tensor(tn(LLM_TENSOR_VISEXP_ATTN_QKV, "weight", i), {n_embd, n_embd_head_k * n_head * 3}, 0);
                        layer.visexp_attn_wo = create_tensor(tn(LLM_TENSOR_VISEXP_ATTN_OUT, "weight", i), {n_embd_head_k * n_head, n_embd}, 0);

                        layer.rope_freqs = create_tensor(tn(LLM_TENSOR_ROPE_FREQS, "weight", i), {n_rot/2}, TENSOR_NOT_REQUIRED | (i != 0 ? TENSOR_DUPLICATED : 0));

                        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);
                        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff}, 0);
                        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd}, 0);
                        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff}, 0);

                        layer.visexp_ffn_gate = create_tensor(tn(LLM_TENSOR_VISEXP_FFN_GATE, "weight", i), {n_embd,   n_ff}, 0);
                        layer.visexp_ffn_down = create_tensor(tn(LLM_TENSOR_VISEXP_FFN_DOWN, "weight", i), {  n_ff, n_embd}, 0);
                        layer.visexp_ffn_up   = create_tensor(tn(LLM_TENSOR_VISEXP_FFN_UP,   "weight", i), {n_embd,   n_ff}, 0);
                    }
                } break;
            case LLM_ARCH_PANGU_EMBED:
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

                        // weight tensors
                        layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_embd_head_k * n_head}, 0);
                        layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_embd_k_gqa}, 0);
                        layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_v_gqa}, 0);
                        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head_k * n_head, n_embd}, 0);

                        // bias tensors
                        layer.bq = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "bias", i), {n_embd_head_k * n_head}, 0);
                        layer.bk = create_tensor(tn(LLM_TENSOR_ATTN_K,   "bias", i), {n_embd_gqa}, 0);
                        layer.bv = create_tensor(tn(LLM_TENSOR_ATTN_V,   "bias", i), {n_embd_gqa}, 0);
                        layer.bo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "bias", i), {n_embd}, 0);

                        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);

                        if (hparams.rope_scaling_type_train == LLAMA_ROPE_SCALING_TYPE_LONGROPE) {
                            layer.rope_long  = create_tensor(tn(LLM_TENSOR_ROPE_FACTORS_LONG,  "weight", i), {n_rot/2}, TENSOR_NOT_REQUIRED | (i != 0 ? TENSOR_DUPLICATED : 0));
                            layer.rope_short = create_tensor(tn(LLM_TENSOR_ROPE_FACTORS_SHORT, "weight", i), {n_rot/2}, TENSOR_NOT_REQUIRED | (i != 0 ? TENSOR_DUPLICATED : 0));
                        } else {
                            layer.rope_freqs = create_tensor(tn(LLM_TENSOR_ROPE_FREQS, "weight", i), {n_rot/2}, TENSOR_NOT_REQUIRED | (i != 0 ? TENSOR_DUPLICATED : 0));
                        }

                        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff}, 0);
                        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd}, 0);
                        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff}, 0);
                    }
                } break;
            case LLM_ARCH_QWEN3NEXT:
                {
                    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, 0);

                    // output
                    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), { n_embd }, 0);
                    output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), { n_embd, n_vocab }, TENSOR_NOT_REQUIRED);

                    // if output is NULL, init from the input tok embed
                    if (output == NULL) {
                        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, TENSOR_DUPLICATED);
                    }

                    const int64_t n_ff_exp = hparams.n_ff_exp ? hparams.n_ff_exp : n_ff / n_expert_used;

                    // Calculate dimensions from hyperparameters
                    const int64_t head_k_dim = hparams.ssm_d_state;
                    const int64_t head_v_dim = hparams.ssm_d_state;
                    const int64_t n_k_heads  = hparams.ssm_n_group;
                    const int64_t n_v_heads  = hparams.ssm_dt_rank;
                    const int64_t key_dim    = head_k_dim * n_k_heads;
                    const int64_t value_dim  = head_v_dim * n_v_heads;
                    const int64_t conv_dim   = key_dim * 2 + value_dim;

                    // Calculate projection sizes
                    const int64_t qkvz_dim = key_dim * 2 + value_dim * 2;
                    const int64_t ba_dim   = n_v_heads * 2;

                    for (int i = 0; i < n_layer; ++i) {
                        auto & layer = layers[i];

                        layer.attn_norm      = create_tensor(tn(LLM_TENSOR_ATTN_NORM,      "weight", i), { n_embd }, 0);
                        layer.attn_post_norm = create_tensor(tn(LLM_TENSOR_ATTN_POST_NORM, "weight", i), { n_embd }, 0);

                        if (!hparams.is_recurrent(i)) {
                            // Attention layers
                            layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "weight", i), { n_embd, n_embd_head_k * n_head * 2 }, 0);
                            layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K,   "weight", i), { n_embd, n_embd_k_gqa }, 0);
                            layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V,   "weight", i), { n_embd, n_embd_v_gqa }, 0);
                            layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), { n_embd_head_k * n_head, n_embd }, 0);

                            // Q/K normalization for attention layers
                            layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), { n_embd_head_k }, 0);
                            layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), { n_embd_head_k }, 0);
                        } else {
                            // Linear attention (gated delta net) specific tensors
                            // Create tensors with calculated dimensions
                            layer.ssm_in         = create_tensor(tn(LLM_TENSOR_SSM_IN,         "weight", i), { n_embd, qkvz_dim }, 0);
                            layer.ssm_conv1d     = create_tensor(tn(LLM_TENSOR_SSM_CONV1D,     "weight", i), { hparams.ssm_d_conv, conv_dim }, 0);
                            layer.ssm_dt         = create_tensor(tn(LLM_TENSOR_SSM_DT,         "bias",   i), { hparams.ssm_dt_rank }, 0);
                            layer.ssm_a          = create_tensor(tn(LLM_TENSOR_SSM_A_NOSCAN,             i), { hparams.ssm_dt_rank }, 0);
                            layer.ssm_beta_alpha = create_tensor(tn(LLM_TENSOR_SSM_BETA_ALPHA, "weight", i), { n_embd, ba_dim }, 0);
                            layer.ssm_norm       = create_tensor(tn(LLM_TENSOR_SSM_NORM,       "weight", i), { head_v_dim }, 0);
                            layer.ssm_out        = create_tensor(tn(LLM_TENSOR_SSM_OUT,        "weight", i), { value_dim, n_embd }, 0);
                        }

                        layer.ffn_gate_inp  = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,  "weight", i), { n_embd, n_expert }, 0);
                        layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), { n_embd, n_ff_exp, n_expert }, 0);
                        layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), { n_ff_exp, n_embd, n_expert }, 0);
                        layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), { n_embd, n_ff_exp, n_expert }, 0);

                        // Shared experts
                        layer.ffn_gate_inp_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP_SHEXP, "weight", i), { n_embd }, 0);
                        layer.ffn_gate_shexp     = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP,     "weight", i), { n_embd, hparams.n_ff_shexp }, 0);
                        layer.ffn_up_shexp       = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,       "weight", i), { n_embd, hparams.n_ff_shexp }, 0);
                        layer.ffn_down_shexp     = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP,     "weight", i), { hparams.n_ff_shexp, n_embd }, 0);
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
    std::vector<std::pair<ggml_context *, llama_buf_map>> ctx_buf_maps;
    ctx_buf_maps.reserve(ctx_map.size());

    // Ensure we have enough capacity for the maximum backend buffer we will potentially create
    const size_t n_max_backend_buffer = ctx_map.size() * ml.files.size();
    pimpl->ctxs_bufs.reserve(n_max_backend_buffer);

    for (auto & [buft, ctx_ptr] : ctx_map) {
        ggml_context * ctx = ctx_ptr.get();

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

        std::vector<ggml_backend_buffer_ptr> bufs;
        if (ml.use_mmap && use_mmap_buffer && buffer_from_host_ptr_supported && is_default_buft) {
            GGML_ASSERT(!ml.no_alloc);
            for (uint32_t idx = 0; idx < ml.files.size(); idx++) {
                // only the mmap region containing the tensors in the model is mapped to the backend buffer
                // this is important for metal with apple silicon: if the entire model could be mapped to a metal buffer,
                //     then we could just use metal for all layers
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
                bufs.emplace_back(buf);
                buf_map.emplace(idx, buf);
            }
        } else {
            ggml_backend_buffer_t buf;
            if (ml.no_alloc) {
                buf = ggml_backend_buft_alloc_buffer(buft, /*size =*/ 0); // dummy buffer
                for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
                    t->buffer = buf; // set dummy buffer for weights so that the backend scheduler won't try to allocate them
                }
            } else {
                buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft); // real buffer
            }
            if (buf == nullptr) {
                throw std::runtime_error(format("unable to allocate %s buffer", ggml_backend_buft_name(buft)));
            }
            if (use_mlock && ggml_backend_buffer_is_host(buf)) {
                pimpl->mlock_bufs.emplace_back(new llama_mlock);
                auto & mlock_buf = pimpl->mlock_bufs.back();
                mlock_buf->init   (ggml_backend_buffer_get_base(buf));
                mlock_buf->grow_to(ggml_backend_buffer_get_size(buf));
            }
            bufs.emplace_back(buf);
            for (uint32_t idx = 0; idx < ml.files.size(); idx++) {
                buf_map.emplace(idx, buf);
            }
        }
        pimpl->ctxs_bufs.emplace_back(std::move(ctx_ptr), std::move(bufs));

        for (auto & buf : buf_map) {
            // indicate that this buffer contains weights
            // this is used by ggml_backend_sched to improve op scheduling: ops that use a weight are preferably scheduled to the backend that contains the weight
            ggml_backend_buffer_set_usage(buf.second, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        }

        ctx_buf_maps.emplace_back(ctx, buf_map);
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
    for (auto & [_, bufs] : pimpl->ctxs_bufs) {
        for (auto & buf: bufs) {
            LLAMA_LOG_INFO("%s: %12s model buffer size = %8.2f MiB\n",
                __func__, ggml_backend_buffer_name(buf.get()), ggml_backend_buffer_get_size(buf.get()) / 1024.0 / 1024.0);
        }
    }

    // populate tensors_by_name
    for (auto & [ctx, _] : pimpl->ctxs_bufs) {
        for (auto * cur = ggml_get_first_tensor(ctx.get()); cur != NULL; cur = ggml_get_next_tensor(ctx.get(), cur)) {
            tensors_by_name.emplace_back(ggml_get_name(cur), cur);
        }
    }

    if (ml.no_alloc) {
        return true;
    }

    // load tensor data
    for (auto & [ctx, buf_map] : ctx_buf_maps) {
        if (!ml.load_all_data(ctx, buf_map, use_mlock ? &pimpl->mlock_mmaps : NULL, params.progress_callback, params.progress_callback_user_data)) {
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

std::map<ggml_backend_buffer_type_t, size_t> llama_model::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> ret;
    for (const auto & [ctx, bufs] : pimpl->ctxs_bufs) {
        if (hparams.no_alloc) {
            GGML_ASSERT(bufs.size() == 1);
            ggml_backend_buffer_t buf = bufs[0].get();
            GGML_ASSERT(ggml_backend_buffer_get_base(buf) == nullptr);
            ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(buf);
            ret[buft] += ggml_backend_alloc_ctx_tensors_from_buft_size(ctx.get(), buft);
        } else {
            for (const auto & buf : bufs) {
                // GGML_ASSERT(ggml_backend_buffer_get_base(buf.get()) != nullptr); // multi_buffer does not have a defined base
                ret[ggml_backend_buffer_get_type(buf.get())] += ggml_backend_buffer_get_size(buf.get());
            }
        }
    }
    return ret;
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
    LLAMA_LOG_INFO("%s: no_alloc         = %d\n",     __func__, hparams.no_alloc);

    if (!hparams.vocab_only) {
        LLAMA_LOG_INFO("%s: n_ctx_train      = %u\n",     __func__, hparams.n_ctx_train);
        LLAMA_LOG_INFO("%s: n_embd           = %u\n",     __func__, hparams.n_embd);
        LLAMA_LOG_INFO("%s: n_embd_inp       = %u\n",     __func__, hparams.n_embd_inp());
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
        LLAMA_LOG_INFO("%s: n_expert_groups  = %d\n",     __func__, hparams.n_expert_groups);
        LLAMA_LOG_INFO("%s: n_group_used     = %d\n",     __func__, hparams.n_group_used);
        LLAMA_LOG_INFO("%s: causal attn      = %d\n",     __func__, hparams.causal_attn);
        LLAMA_LOG_INFO("%s: pooling type     = %d\n",     __func__, hparams.pooling_type);
        LLAMA_LOG_INFO("%s: rope type        = %d\n",     __func__, hparams.rope_type);
        LLAMA_LOG_INFO("%s: rope scaling     = %s\n",     __func__, rope_scaling_type.c_str());
        LLAMA_LOG_INFO("%s: freq_base_train  = %.1f\n",   __func__, hparams.rope_freq_base_train);
        LLAMA_LOG_INFO("%s: freq_scale_train = %g\n",     __func__, hparams.rope_freq_scale_train);
        LLAMA_LOG_INFO("%s: n_ctx_orig_yarn  = %u\n",     __func__, hparams.n_ctx_orig_yarn);
        LLAMA_LOG_INFO("%s: rope_yarn_log_mul= %.4f\n",   __func__, hparams.rope_yarn_log_mul);
        LLAMA_LOG_INFO("%s: rope_finetuned   = %s\n",     __func__, hparams.rope_finetuned ? "yes" : "unknown");
        // MRoPE (Multi-axis Rotary Position Embedding) sections
        if (const auto & s = hparams.rope_sections; s[0] || s[1] || s[2] || s[3]) {
            LLAMA_LOG_INFO("%s: mrope sections   = [%d, %d, %d, %d]\n", __func__, s[0], s[1], s[2], s[3]);
        }
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
        arch == LLM_ARCH_GRANITE_HYBRID ||
        arch == LLM_ARCH_QWEN3NEXT ||
        arch == LLM_ARCH_NEMOTRON_H ||
        arch == LLM_ARCH_NEMOTRON_H_MOE) {
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
    }

    if (arch == LLM_ARCH_QWEN2MOE) {
        LLAMA_LOG_INFO("%s: n_ff_exp         = %d\n",     __func__, hparams.n_ff_exp);
        LLAMA_LOG_INFO("%s: n_ff_shexp       = %d\n",     __func__, hparams.n_ff_shexp);
    }

    if (arch == LLM_ARCH_QWEN3MOE || arch == LLM_ARCH_OPENAI_MOE || arch == LLM_ARCH_QWEN3VLMOE || arch == LLM_ARCH_RND1) {
        LLAMA_LOG_INFO("%s: n_ff_exp         = %d\n",     __func__, hparams.n_ff_exp);
    }

    if (arch == LLM_ARCH_MINICPM ||
        arch == LLM_ARCH_GRANITE ||
        arch == LLM_ARCH_GRANITE_MOE ||
        arch == LLM_ARCH_GRANITE_HYBRID ||
        arch == LLM_ARCH_NEMOTRON_H_MOE) {
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

    if (arch == LLM_ARCH_BAILINGMOE2) {
        LLAMA_LOG_INFO("%s: n_layer_dense_lead   = %d\n",     __func__, hparams.n_layer_dense_lead);
        LLAMA_LOG_INFO("%s: n_ff_exp             = %d\n",     __func__, hparams.n_ff_exp);
        LLAMA_LOG_INFO("%s: n_ff_shexp           = %d\n",     __func__, hparams.n_ff_shexp);
        LLAMA_LOG_INFO("%s: n_expert_shared      = %d\n",     __func__, hparams.n_expert_shared);
        LLAMA_LOG_INFO("%s: expert_weights_scale = %.1f\n",   __func__, hparams.expert_weights_scale);
        LLAMA_LOG_INFO("%s: expert_weights_norm  = %d\n",     __func__, hparams.expert_weights_norm);
        LLAMA_LOG_INFO("%s: expert_gating_func   = %s\n",     __func__, llama_expert_gating_func_name((llama_expert_gating_func_type) hparams.expert_gating_func));
        LLAMA_LOG_INFO("%s: nextn_predict_layers = %d\n",     __func__, hparams.nextn_predict_layers);
    }

    if (arch == LLM_ARCH_SMALLTHINKER || arch == LLM_ARCH_LFM2MOE) {
        LLAMA_LOG_INFO("%s: n_ff_exp             = %d\n",     __func__, hparams.n_ff_exp);
        LLAMA_LOG_INFO("%s: expert_gating_func   = %s\n",     __func__, llama_expert_gating_func_name((llama_expert_gating_func_type) hparams.expert_gating_func));
    }

    if (arch == LLM_ARCH_GROVEMOE) {
        LLAMA_LOG_INFO("%s: n_ff_exp             = %d\n",     __func__, hparams.n_ff_exp);
        LLAMA_LOG_INFO("%s: n_ff_chexp           = %d\n",     __func__, hparams.n_ff_chexp);
        LLAMA_LOG_INFO("%s: n_group_experts      = %d\n",     __func__, hparams.n_group_experts);
        LLAMA_LOG_INFO("%s: expert_group_scale   = %.2f\n",   __func__, hparams.expert_group_scale);
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
    const uint32_t n_ctx_seq = cparams.n_ctx_seq;

    // choose long/short freq factors based on the context size
    if (layers[il].rope_freqs != nullptr) {
        return layers[il].rope_freqs;
    }

    if (n_ctx_seq > hparams.n_ctx_orig_yarn) {
        return layers[il].rope_long;
    }

    return layers[il].rope_short;
}

llama_memory_i * llama_model::create_memory(const llama_memory_params & params, const llama_cparams & cparams) const {
    llama_memory_i * res;

    switch (arch) {
        // Models that need specific instantiation should be handled in the
        // switch statement
        case LLM_ARCH_BERT:
        case LLM_ARCH_JINA_BERT_V2:
        case LLM_ARCH_JINA_BERT_V3:
        case LLM_ARCH_NOMIC_BERT:
        case LLM_ARCH_NOMIC_BERT_MOE:
        case LLM_ARCH_NEO_BERT:
        case LLM_ARCH_WAVTOKENIZER_DEC:
        case LLM_ARCH_GEMMA_EMBEDDING:
        case LLM_ARCH_DREAM:
        case LLM_ARCH_LLADA:
        case LLM_ARCH_LLADA_MOE:
        case LLM_ARCH_RND1:
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
                            GGML_TYPE_F32,
                            GGML_TYPE_F32,
                            cparams.offload_kqv,
                            std::max((uint32_t) 1, cparams.n_seq_max),
                            cparams.n_seq_max,
                            nullptr);
                } else if (llm_arch_is_hybrid(arch)) {

                    // The main difference between hybrid architectures is the
                    // layer filters, so pick the right one here
                    llama_memory_hybrid::layer_filter_cb filter_attn = nullptr;
                    llama_memory_hybrid::layer_filter_cb filter_recr = nullptr;
                    if (arch == LLM_ARCH_FALCON_H1) {
                        filter_attn = [&](int32_t) { return true; };
                        filter_recr = [&](int32_t) { return true; };
                    } else if (arch == LLM_ARCH_NEMOTRON_H || arch == LLM_ARCH_NEMOTRON_H_MOE) {
                        filter_attn = [&](int32_t il) {
                            return !hparams.is_recurrent(il) && hparams.n_ff(il) == 0;
                        };
                        filter_recr = [&](int32_t il) {
                            return hparams.is_recurrent(il) && hparams.n_ff(il) == 0;
                        };
                    }

                    res = new llama_memory_hybrid(
                        /* model             */ *this,
                        /* attn_type_k       */ params.type_k,
                        /* attn_type_v       */ params.type_v,
                        /* attn_v_trans      */ !cparams.flash_attn,
                        /* attn_kv_size      */ cparams.n_ctx,
                        /* attn_n_pad        */ 1,
                        /* attn_n_swa        */ hparams.n_swa,
                        /* attn_swa_type     */ hparams.swa_type,
                        /* recurrent_type_k  */ GGML_TYPE_F32,
                        /* recurrent_type_v  */ GGML_TYPE_F32,
                        /* recurrent_kv_size */ std::max((uint32_t) 1, cparams.n_seq_max),
                        /* n_seq_max         */ cparams.n_seq_max,
                        /* offload           */ cparams.offload_kqv,
                        /* unified           */ cparams.kv_unified,
                        /* filter_attn       */ std::move(filter_attn),
                        /* filter_recr       */ std::move(filter_recr));
                } else {
                    llama_memory_i::layer_reuse_cb reuse = nullptr;

                    if (arch == LLM_ARCH_GEMMA3N) {
                        reuse = [&](int32_t il) {
                            if (il >= (int32_t) hparams.n_layer_kv_from_start) {
                                return (int32_t) hparams.n_layer_kv_from_start - (hparams.is_swa(il) ? 2 : 1);
                            }

                            return -1;
                        };
                    }

                    if (hparams.swa_type != LLAMA_SWA_TYPE_NONE) {
                        GGML_ASSERT(hparams.is_swa_any());

                        res = new llama_kv_cache_iswa(
                                *this,
                                params.type_k,
                                params.type_v,
                                !cparams.flash_attn,
                                cparams.offload_kqv,
                                params.swa_full,
                                cparams.kv_unified,
                                cparams.n_ctx_seq,
                                cparams.n_seq_max,
                                cparams.n_ubatch,
                                1,
                                nullptr,
                                reuse);
                    } else {
                        GGML_ASSERT(!hparams.is_swa_any());

                        res = new llama_kv_cache(
                                *this,
                                params.type_k,
                                params.type_v,
                                !cparams.flash_attn,
                                cparams.offload_kqv,
                                cparams.kv_unified,
                                cparams.n_ctx_seq,
                                cparams.n_seq_max,
                                1,
                                hparams.n_swa,
                                hparams.swa_type,
                                nullptr,
                                nullptr);
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
                if (hparams.swa_type == LLAMA_SWA_TYPE_NONE) {
                    llm = std::make_unique<llm_build_llama>(*this, params);
                } else {
                    llm = std::make_unique<llm_build_llama_iswa>(*this, params);
                }
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
        case LLM_ARCH_JINA_BERT_V3:
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
        case LLM_ARCH_LLADA_MOE:
            {
                llm = std::make_unique<llm_build_llada_moe>(*this, params);
            }
            break;
        case LLM_ARCH_RND1:
            {
                llm = std::make_unique<llm_build_rnd1>(*this, params);
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
        case LLM_ARCH_QWEN3VL:
            {
                llm = std::make_unique<llm_build_qwen3vl>(*this, params);
            } break;
        case LLM_ARCH_QWEN3VLMOE:
            {
                llm = std::make_unique<llm_build_qwen3vlmoe>(*this, params);
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
                if (hparams.swa_type == LLAMA_SWA_TYPE_STANDARD) {
                    llm = std::make_unique<llm_build_gemma3<true>>(*this, params);
                } else {
                    llm = std::make_unique<llm_build_gemma3<false>>(*this, params);
                }
            } break;
        case LLM_ARCH_GEMMA3N:
            {
                llm = std::make_unique<llm_build_gemma3n_iswa>(*this, params);
            } break;
        case LLM_ARCH_GEMMA_EMBEDDING:
            {
                llm = std::make_unique<llm_build_gemma_embedding>(*this, params);
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
                if (hparams.swa_type == LLAMA_SWA_TYPE_STANDARD) {
                    llm = std::make_unique<llm_build_olmo2<true>>(*this, params);
                } else {
                    llm = std::make_unique<llm_build_olmo2<false>>(*this, params);
                }
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
        case LLM_ARCH_NEMOTRON_H:
        case LLM_ARCH_NEMOTRON_H_MOE:
            {
                llm = std::make_unique<llm_build_nemotron_h>(*this, params);
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
        case LLM_ARCH_BAILINGMOE2:
            {
                llm = std::make_unique<llm_build_bailingmoe2>(*this, params);
            } break;
        case LLM_ARCH_SEED_OSS:
            {
                llm = std::make_unique<llm_build_seed_oss>(*this, params);
            } break;
        case LLM_ARCH_DOTS1:
            {
                llm = std::make_unique<llm_build_dots1>(*this, params);
            } break;
        case LLM_ARCH_ARCEE:
            {
                llm = std::make_unique<llm_build_arcee>(*this, params);
            } break;
        case LLM_ARCH_AFMOE:
            {
                llm = std::make_unique<llm_build_afmoe>(*this, params);
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
        case LLM_ARCH_LFM2MOE:
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
        case LLM_ARCH_GROVEMOE:
            {
                llm = std::make_unique<llm_build_grovemoe>(*this, params);
            } break;
        case LLM_ARCH_APERTUS:
            {
                llm = std::make_unique<llm_build_apertus>(*this, params);
            } break;
        case LLM_ARCH_MINIMAX_M2:
            {
                llm = std::make_unique<llm_build_minimax_m2>(*this, params);
            } break;
        case LLM_ARCH_COGVLM:
            {
                llm = std::make_unique<llm_build_cogvlm>(*this, params);
            } break;
        case LLM_ARCH_PANGU_EMBED:
            {
                llm = std::make_unique<llm_build_pangu_embedded>(*this, params);
            } break;
        case LLM_ARCH_QWEN3NEXT:
            {
                llm = std::make_unique<llm_build_qwen3next>(*this, params);
            } break;
        case LLM_ARCH_MISTRAL3:
            {
                llm = std::make_unique<llm_build_mistral3>(*this, params);
            } break;
        default:
            GGML_ABORT("fatal error");
    }

    // add on pooling layer
    llm->build_pooling(cls, cls_b, cls_out, cls_out_b);

    // if the gguf model was converted with --sentence-transformers-dense-modules
    // there will be two additional dense projection layers
    // dense linear projections are applied after pooling
    // TODO: move reranking logic here and generalize
    llm->build_dense_out(dense_2_out_layers, dense_3_out_layers);

    return llm->res->get_gf();
}


//
// interface implementation
//

llama_model_params llama_model_default_params() {
    llama_model_params result = {
        /*.devices                     =*/ nullptr,
        /*.tensor_buft_overrides       =*/ nullptr,
        /*.n_gpu_layers                =*/ 999,
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
        /*.no_host                     =*/ false,
        /*.no_alloc                    =*/ false,
    };

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

int32_t llama_model_n_embd_inp(const llama_model * model) {
    return model->hparams.n_embd_inp();
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
        case LLM_ARCH_CLIP:
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
        case LLM_ARCH_NEMOTRON_H:
        case LLM_ARCH_NEMOTRON_H_MOE:
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
        case LLM_ARCH_MISTRAL3:
            return LLAMA_ROPE_TYPE_NORM;

        // the pairs of head values are offset by n_rot/2
        case LLM_ARCH_FALCON:
        case LLM_ARCH_FALCON_H1:
        case LLM_ARCH_GROK:
        case LLM_ARCH_DBRX:
        case LLM_ARCH_BERT:
        case LLM_ARCH_JINA_BERT_V3:
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
        case LLM_ARCH_LLADA_MOE:
        case LLM_ARCH_RND1:
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
        case LLM_ARCH_GEMMA_EMBEDDING:
        case LLM_ARCH_STARCODER2:
        case LLM_ARCH_OPENELM:
        case LLM_ARCH_GPTNEOX:
        case LLM_ARCH_CODESHELL:
        case LLM_ARCH_ORION:
        case LLM_ARCH_NEMOTRON:
        case LLM_ARCH_EXAONE:
        case LLM_ARCH_EXAONE4:
        case LLM_ARCH_MINICPM3:
        case LLM_ARCH_BAILINGMOE2:
        case LLM_ARCH_DOTS1:
        case LLM_ARCH_HUNYUAN_MOE:
        case LLM_ARCH_OPENAI_MOE:
        case LLM_ARCH_HUNYUAN_DENSE:
        case LLM_ARCH_LFM2:
        case LLM_ARCH_LFM2MOE:
        case LLM_ARCH_SMALLTHINKER:
        case LLM_ARCH_SEED_OSS:
        case LLM_ARCH_GROVEMOE:
        case LLM_ARCH_APERTUS:
        case LLM_ARCH_MINIMAX_M2:
        case LLM_ARCH_COGVLM:
        case LLM_ARCH_PANGU_EMBED:
        case LLM_ARCH_AFMOE:
        case LLM_ARCH_QWEN3NEXT:
            return LLAMA_ROPE_TYPE_NEOX;

        case LLM_ARCH_QWEN2VL:
            return LLAMA_ROPE_TYPE_MROPE;
        case LLM_ARCH_QWEN3VL:
        case LLM_ARCH_QWEN3VLMOE:
            return LLAMA_ROPE_TYPE_IMROPE;

        case LLM_ARCH_GLM4:
            return model->hparams.use_mrope() ? LLAMA_ROPE_TYPE_MROPE : LLAMA_ROPE_TYPE_NORM;
        case LLM_ARCH_GLM4_MOE:
            return model->hparams.use_mrope() ? LLAMA_ROPE_TYPE_MROPE : LLAMA_ROPE_TYPE_NEOX;

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

const char * llama_model_meta_key_str(llama_model_meta_key key) {
    switch (key) {
        case LLAMA_MODEL_META_KEY_SAMPLING_SEQUENCE:        return "general.sampling.sequence";
        case LLAMA_MODEL_META_KEY_SAMPLING_TOP_K:           return "general.sampling.top_k";
        case LLAMA_MODEL_META_KEY_SAMPLING_TOP_P:           return "general.sampling.top_p";
        case LLAMA_MODEL_META_KEY_SAMPLING_MIN_P:           return "general.sampling.min_p";
        case LLAMA_MODEL_META_KEY_SAMPLING_XTC_PROBABILITY: return "general.sampling.xtc_probability";
        case LLAMA_MODEL_META_KEY_SAMPLING_XTC_THRESHOLD:   return "general.sampling.xtc_threshold";
        case LLAMA_MODEL_META_KEY_SAMPLING_TEMP:            return "general.sampling.temp";
        case LLAMA_MODEL_META_KEY_SAMPLING_PENALTY_LAST_N:  return "general.sampling.penalty_last_n";
        case LLAMA_MODEL_META_KEY_SAMPLING_PENALTY_REPEAT:  return "general.sampling.penalty_repeat";
        case LLAMA_MODEL_META_KEY_SAMPLING_MIROSTAT:        return "general.sampling.mirostat";
        case LLAMA_MODEL_META_KEY_SAMPLING_MIROSTAT_TAU:    return "general.sampling.mirostat_tau";
        case LLAMA_MODEL_META_KEY_SAMPLING_MIROSTAT_ETA:    return "general.sampling.mirostat_eta";
        default:                                            return nullptr;
    }
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

bool llama_model_is_hybrid(const llama_model * model) {
    return llm_arch_is_hybrid(model->arch);
}

bool llama_model_is_diffusion(const llama_model * model) {
    return llm_arch_is_diffusion(model->arch);
}

const std::vector<std::pair<std::string, ggml_tensor *>> & llama_internal_get_tensor_map(const llama_model * model) {
    return model->tensors_by_name;
}
