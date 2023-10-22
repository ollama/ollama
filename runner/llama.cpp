/**
 * llama.cpp - git 465219b9143ac01db0990bbcb0a081ef72ec2008
 *
 * MIT License
 *
 * Copyright (c) 2023 Georgi Gerganov
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#define LLAMA_API_INTERNAL
#include "llama.h"

#include "unicode.h"

#include "ggml.h"

#include "ggml-alloc.h"

#ifdef GGML_USE_CUBLAS
#  include "ggml-cuda.h"
#elif defined(GGML_USE_CLBLAST)
#  include "ggml-opencl.h"
#endif

#ifdef GGML_USE_METAL
#  include "ggml-metal.h"
#endif
#ifdef GGML_USE_MPI
#  include "ggml-mpi.h"
#endif
#ifdef GGML_USE_K_QUANTS
#  ifndef QK_K
#    ifdef GGML_QKK_64
#      define QK_K 64
#    else
#      define QK_K 256
#    endif
#  endif
#endif

#ifdef __has_include
    #if __has_include(<unistd.h>)
        #include <unistd.h>
        #if defined(_POSIX_MAPPED_FILES)
            #include <sys/mman.h>
        #endif
        #if defined(_POSIX_MEMLOCK_RANGE)
            #include <sys/resource.h>
        #endif
    #endif
#endif

#if defined(_WIN32)
    #define WIN32_LEAN_AND_MEAN
    #ifndef NOMINMAX
        #define NOMINMAX
    #endif
    #include <windows.h>
    #include <io.h>
    #include <stdio.h> // for _fseeki64
#endif

#include <algorithm>
#include <array>
#include <cassert>
#include <cinttypes>
#include <climits>
#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>
#include <initializer_list>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <queue>
#include <random>
#include <regex>
#include <sstream>
#include <thread>
#include <unordered_map>
#include <set>
#include <forward_list>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

#ifdef __GNUC__
#ifdef __MINGW32__
#define LLAMA_ATTRIBUTE_FORMAT(...) __attribute__((format(gnu_printf, __VA_ARGS__)))
#else
#define LLAMA_ATTRIBUTE_FORMAT(...) __attribute__((format(printf, __VA_ARGS__)))
#endif
#else
#define LLAMA_ATTRIBUTE_FORMAT(...)
#endif

//
// logging
//

LLAMA_ATTRIBUTE_FORMAT(2, 3)
static void llama_log_internal        (ggml_log_level level, const char* format, ...);
static void llama_log_callback_default(ggml_log_level level, const char * text, void * user_data);

#define LLAMA_LOG_INFO(...)  llama_log_internal(GGML_LOG_LEVEL_INFO , __VA_ARGS__)
#define LLAMA_LOG_WARN(...)  llama_log_internal(GGML_LOG_LEVEL_WARN , __VA_ARGS__)
#define LLAMA_LOG_ERROR(...) llama_log_internal(GGML_LOG_LEVEL_ERROR, __VA_ARGS__)

//
// helpers
//

static size_t utf8_len(char src) {
    const size_t lookup[] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4 };
    uint8_t highbits = static_cast<uint8_t>(src) >> 4;
    return lookup[highbits];
}

static void replace_all(std::string & s, const std::string & search, const std::string & replace) {
    std::string result;
    for (size_t pos = 0; ; pos += search.length()) {
        auto new_pos = s.find(search, pos);
        if (new_pos == std::string::npos) {
            result += s.substr(pos, s.size() - pos);
            break;
        }
        result += s.substr(pos, new_pos - pos) + replace;
        pos = new_pos;
    }
    s = std::move(result);
}

static bool is_float_close(float a, float b, float abs_tol) {
    // Check for non-negative tolerance
    if (abs_tol < 0.0) {
        throw std::invalid_argument("Tolerance must be non-negative");
    }

    // Exact equality check
    if (a == b) {
        return true;
    }

    // Check for infinities
    if (std::isinf(a) || std::isinf(b)) {
        return false;
    }

    // Regular comparison using the provided absolute tolerance
    return std::fabs(b - a) <= abs_tol;
}

#ifdef GGML_USE_CPU_HBM
#include <hbwmalloc.h>
#endif

static void zeros(std::ofstream & file, size_t n) {
    char zero = 0;
    for (size_t i = 0; i < n; ++i) {
        file.write(&zero, 1);
    }
}

LLAMA_ATTRIBUTE_FORMAT(1, 2)
static std::string format(const char * fmt, ...) {
    va_list ap;
    va_list ap2;
    va_start(ap, fmt);
    va_copy(ap2, ap);
    int size = vsnprintf(NULL, 0, fmt, ap);
    GGML_ASSERT(size >= 0 && size < INT_MAX); // NOLINT
    std::vector<char> buf(size + 1);
    int size2 = vsnprintf(buf.data(), size + 1, fmt, ap2);
    GGML_ASSERT(size2 == size);
    va_end(ap2);
    va_end(ap);
    return std::string(buf.data(), size);
}

//
// gguf constants (sync with gguf.py)
//

enum llm_arch {
    LLM_ARCH_LLAMA,
    LLM_ARCH_FALCON,
    LLM_ARCH_BAICHUAN,
    LLM_ARCH_GPT2,
    LLM_ARCH_GPTJ,
    LLM_ARCH_GPTNEOX,
    LLM_ARCH_MPT,
    LLM_ARCH_STARCODER,
    LLM_ARCH_PERSIMMON,
    LLM_ARCH_REFACT,
    LLM_ARCH_BLOOM,
    LLM_ARCH_UNKNOWN,
};

static std::map<llm_arch, std::string> LLM_ARCH_NAMES = {
    { LLM_ARCH_LLAMA,           "llama"     },
    { LLM_ARCH_FALCON,          "falcon"    },
    { LLM_ARCH_GPT2,            "gpt2"      },
    { LLM_ARCH_GPTJ,            "gptj"      },
    { LLM_ARCH_GPTNEOX,         "gptneox"   },
    { LLM_ARCH_MPT,             "mpt"       },
    { LLM_ARCH_BAICHUAN,        "baichuan"  },
    { LLM_ARCH_STARCODER,       "starcoder" },
    { LLM_ARCH_PERSIMMON,       "persimmon" },
    { LLM_ARCH_REFACT,          "refact"    },
    { LLM_ARCH_BLOOM,           "bloom"     },
};

enum llm_kv {
    LLM_KV_GENERAL_ARCHITECTURE,
    LLM_KV_GENERAL_QUANTIZATION_VERSION,
    LLM_KV_GENERAL_ALIGNMENT,
    LLM_KV_GENERAL_NAME,
    LLM_KV_GENERAL_AUTHOR,
    LLM_KV_GENERAL_URL,
    LLM_KV_GENERAL_DESCRIPTION,
    LLM_KV_GENERAL_LICENSE,
    LLM_KV_GENERAL_SOURCE_URL,
    LLM_KV_GENERAL_SOURCE_HF_REPO,

    LLM_KV_CONTEXT_LENGTH,
    LLM_KV_EMBEDDING_LENGTH,
    LLM_KV_BLOCK_COUNT,
    LLM_KV_FEED_FORWARD_LENGTH,
    LLM_KV_USE_PARALLEL_RESIDUAL,
    LLM_KV_TENSOR_DATA_LAYOUT,

    LLM_KV_ATTENTION_HEAD_COUNT,
    LLM_KV_ATTENTION_HEAD_COUNT_KV,
    LLM_KV_ATTENTION_MAX_ALIBI_BIAS,
    LLM_KV_ATTENTION_CLAMP_KQV,
    LLM_KV_ATTENTION_LAYERNORM_EPS,
    LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,

    LLM_KV_ROPE_DIMENSION_COUNT,
    LLM_KV_ROPE_FREQ_BASE,
    LLM_KV_ROPE_SCALE_LINEAR,

    LLM_KV_TOKENIZER_MODEL,
    LLM_KV_TOKENIZER_LIST,
    LLM_KV_TOKENIZER_TOKEN_TYPE,
    LLM_KV_TOKENIZER_SCORES,
    LLM_KV_TOKENIZER_MERGES,
    LLM_KV_TOKENIZER_BOS_ID,
    LLM_KV_TOKENIZER_EOS_ID,
    LLM_KV_TOKENIZER_UNK_ID,
    LLM_KV_TOKENIZER_SEP_ID,
    LLM_KV_TOKENIZER_PAD_ID,
    LLM_KV_TOKENIZER_HF_JSON,
    LLM_KV_TOKENIZER_RWKV,
};

static std::map<llm_kv, std::string> LLM_KV_NAMES = {
    { LLM_KV_GENERAL_ARCHITECTURE,          "general.architecture"                  },
    { LLM_KV_GENERAL_QUANTIZATION_VERSION,  "general.quantization_version"          },
    { LLM_KV_GENERAL_ALIGNMENT,             "general.alignment"                     },
    { LLM_KV_GENERAL_NAME,                  "general.name"                          },
    { LLM_KV_GENERAL_AUTHOR,                "general.author"                        },
    { LLM_KV_GENERAL_URL,                   "general.url"                           },
    { LLM_KV_GENERAL_DESCRIPTION,           "general.description"                   },
    { LLM_KV_GENERAL_LICENSE,               "general.license"                       },
    { LLM_KV_GENERAL_SOURCE_URL,            "general.source.url"                    },
    { LLM_KV_GENERAL_SOURCE_HF_REPO,        "general.source.huggingface.repository" },

    { LLM_KV_CONTEXT_LENGTH,                "%s.context_length"        },
    { LLM_KV_EMBEDDING_LENGTH,              "%s.embedding_length"      },
    { LLM_KV_BLOCK_COUNT,                   "%s.block_count"           },
    { LLM_KV_FEED_FORWARD_LENGTH,           "%s.feed_forward_length"   },
    { LLM_KV_USE_PARALLEL_RESIDUAL,         "%s.use_parallel_residual" },
    { LLM_KV_TENSOR_DATA_LAYOUT,            "%s.tensor_data_layout"    },

    { LLM_KV_ATTENTION_HEAD_COUNT,          "%s.attention.head_count"             },
    { LLM_KV_ATTENTION_HEAD_COUNT_KV,       "%s.attention.head_count_kv"          },
    { LLM_KV_ATTENTION_MAX_ALIBI_BIAS,      "%s.attention.max_alibi_bias"         },
    { LLM_KV_ATTENTION_CLAMP_KQV,           "%s.attention.clamp_kqv"              },
    { LLM_KV_ATTENTION_LAYERNORM_EPS,       "%s.attention.layer_norm_epsilon"     },
    { LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,   "%s.attention.layer_norm_rms_epsilon" },

    { LLM_KV_ROPE_DIMENSION_COUNT,          "%s.rope.dimension_count" },
    { LLM_KV_ROPE_FREQ_BASE,                "%s.rope.freq_base"       },
    { LLM_KV_ROPE_SCALE_LINEAR,             "%s.rope.scale_linear"    },

    { LLM_KV_TOKENIZER_MODEL,               "tokenizer.ggml.model"              },
    { LLM_KV_TOKENIZER_LIST,                "tokenizer.ggml.tokens"             },
    { LLM_KV_TOKENIZER_TOKEN_TYPE,          "tokenizer.ggml.token_type"         },
    { LLM_KV_TOKENIZER_SCORES,              "tokenizer.ggml.scores"             },
    { LLM_KV_TOKENIZER_MERGES,              "tokenizer.ggml.merges"             },
    { LLM_KV_TOKENIZER_BOS_ID,              "tokenizer.ggml.bos_token_id"       },
    { LLM_KV_TOKENIZER_EOS_ID,              "tokenizer.ggml.eos_token_id"       },
    { LLM_KV_TOKENIZER_UNK_ID,              "tokenizer.ggml.unknown_token_id"   },
    { LLM_KV_TOKENIZER_SEP_ID,              "tokenizer.ggml.seperator_token_id" },
    { LLM_KV_TOKENIZER_PAD_ID,              "tokenizer.ggml.padding_token_id"   },
    { LLM_KV_TOKENIZER_HF_JSON,             "tokenizer.huggingface.json"        },
    { LLM_KV_TOKENIZER_RWKV,                "tokenizer.rwkv.world"              },
};

struct LLM_KV {
    LLM_KV(llm_arch arch) : arch(arch) {}

    llm_arch arch;

    std::string operator()(llm_kv kv) const {
        return ::format(LLM_KV_NAMES[kv].c_str(), LLM_ARCH_NAMES[arch].c_str());
    }
};

enum llm_tensor {
    LLM_TENSOR_TOKEN_EMBD,
    LLM_TENSOR_TOKEN_EMBD_NORM,
    LLM_TENSOR_POS_EMBD,
    LLM_TENSOR_OUTPUT,
    LLM_TENSOR_OUTPUT_NORM,
    LLM_TENSOR_ROPE_FREQS,
    LLM_TENSOR_ATTN_Q,
    LLM_TENSOR_ATTN_K,
    LLM_TENSOR_ATTN_V,
    LLM_TENSOR_ATTN_QKV,
    LLM_TENSOR_ATTN_OUT,
    LLM_TENSOR_ATTN_NORM,
    LLM_TENSOR_ATTN_NORM_2,
    LLM_TENSOR_ATTN_ROT_EMBD,
    LLM_TENSOR_FFN_GATE,
    LLM_TENSOR_FFN_DOWN,
    LLM_TENSOR_FFN_UP,
    LLM_TENSOR_FFN_NORM,
    LLM_TENSOR_ATTN_Q_NORM,
    LLM_TENSOR_ATTN_K_NORM,
};

static std::map<llm_arch, std::map<llm_tensor, std::string>> LLM_TENSOR_NAMES = {
    {
        LLM_ARCH_LLAMA,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_OUTPUT,          "output" },
            { LLM_TENSOR_ROPE_FREQS,      "rope_freqs" },
            { LLM_TENSOR_ATTN_NORM,       "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_Q,          "blk.%d.attn_q" },
            { LLM_TENSOR_ATTN_K,          "blk.%d.attn_k" },
            { LLM_TENSOR_ATTN_V,          "blk.%d.attn_v" },
            { LLM_TENSOR_ATTN_OUT,        "blk.%d.attn_output" },
            { LLM_TENSOR_ATTN_ROT_EMBD,   "blk.%d.attn_rot_embd" },
            { LLM_TENSOR_FFN_NORM,        "blk.%d.ffn_norm" },
            { LLM_TENSOR_FFN_GATE,        "blk.%d.ffn_gate" },
            { LLM_TENSOR_FFN_DOWN,        "blk.%d.ffn_down" },
            { LLM_TENSOR_FFN_UP,          "blk.%d.ffn_up" },
        },
    },
    {
        LLM_ARCH_BAICHUAN,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_OUTPUT,          "output" },
            { LLM_TENSOR_ROPE_FREQS,      "rope_freqs" },
            { LLM_TENSOR_ATTN_NORM,       "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_Q,          "blk.%d.attn_q" },
            { LLM_TENSOR_ATTN_K,          "blk.%d.attn_k" },
            { LLM_TENSOR_ATTN_V,          "blk.%d.attn_v" },
            { LLM_TENSOR_ATTN_OUT,        "blk.%d.attn_output" },
            { LLM_TENSOR_ATTN_ROT_EMBD,   "blk.%d.attn_rot_embd" },
            { LLM_TENSOR_FFN_NORM,        "blk.%d.ffn_norm" },
            { LLM_TENSOR_FFN_GATE,        "blk.%d.ffn_gate" },
            { LLM_TENSOR_FFN_DOWN,        "blk.%d.ffn_down" },
            { LLM_TENSOR_FFN_UP,          "blk.%d.ffn_up" },
        },
    },
    {
        LLM_ARCH_FALCON,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_OUTPUT,          "output" },
            { LLM_TENSOR_ATTN_NORM,       "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_NORM_2,     "blk.%d.attn_norm_2" },
            { LLM_TENSOR_ATTN_QKV,        "blk.%d.attn_qkv" },
            { LLM_TENSOR_ATTN_OUT,        "blk.%d.attn_output" },
            { LLM_TENSOR_FFN_DOWN,        "blk.%d.ffn_down" },
            { LLM_TENSOR_FFN_UP,          "blk.%d.ffn_up" },
        },
    },
    {
        LLM_ARCH_GPT2,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
        },
    },
    {
        LLM_ARCH_GPTJ,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
        },
    },
    {
        LLM_ARCH_GPTNEOX,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_OUTPUT,          "output" },
            { LLM_TENSOR_ATTN_NORM,       "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_QKV,        "blk.%d.attn_qkv" },
            { LLM_TENSOR_ATTN_OUT,        "blk.%d.attn_output" },
            { LLM_TENSOR_FFN_NORM,        "blk.%d.ffn_norm" },
            { LLM_TENSOR_FFN_DOWN,        "blk.%d.ffn_down" },
            { LLM_TENSOR_FFN_UP,          "blk.%d.ffn_up" },
        },
    },
    {
        LLM_ARCH_PERSIMMON,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd"},
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm"},
            { LLM_TENSOR_OUTPUT,          "output"},
            { LLM_TENSOR_ATTN_NORM,       "blk.%d.attn_norm"},
            { LLM_TENSOR_ATTN_QKV,        "blk.%d.attn_qkv"},
            { LLM_TENSOR_ATTN_OUT,        "blk.%d.attn_output"},
            { LLM_TENSOR_ATTN_Q_NORM,     "blk.%d.attn_q_norm"},
            { LLM_TENSOR_ATTN_K_NORM,     "blk.%d.attn_k_norm"},
            { LLM_TENSOR_FFN_NORM,        "blk.%d.ffn_norm"},
            { LLM_TENSOR_FFN_DOWN,        "blk.%d.ffn_down"},
            { LLM_TENSOR_FFN_UP,          "blk.%d.ffn_up"},
            { LLM_TENSOR_ATTN_ROT_EMBD,   "blk.%d.attn_rot_embd"},
        },
    },
    {
        LLM_ARCH_MPT,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_OUTPUT,          "output" },
            { LLM_TENSOR_ATTN_NORM,       "blk.%d.attn_norm" },
            { LLM_TENSOR_FFN_NORM,        "blk.%d.ffn_norm" },
            { LLM_TENSOR_ATTN_QKV,        "blk.%d.attn_qkv" },
            { LLM_TENSOR_ATTN_OUT,        "blk.%d.attn_output" },
            { LLM_TENSOR_FFN_DOWN,        "blk.%d.ffn_down" },
            { LLM_TENSOR_FFN_UP,          "blk.%d.ffn_up" },
        },
    },
    {
        LLM_ARCH_STARCODER,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_POS_EMBD,        "position_embd" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_OUTPUT,          "output" },
            { LLM_TENSOR_ATTN_NORM,       "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_QKV,        "blk.%d.attn_qkv" },
            { LLM_TENSOR_ATTN_OUT,        "blk.%d.attn_output" },
            { LLM_TENSOR_FFN_NORM,        "blk.%d.ffn_norm" },
            { LLM_TENSOR_FFN_UP,          "blk.%d.ffn_up" },
            { LLM_TENSOR_FFN_DOWN,        "blk.%d.ffn_down" },
        },
    },
    {
        LLM_ARCH_REFACT,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_OUTPUT,          "output" },
            { LLM_TENSOR_ATTN_NORM,       "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_Q,          "blk.%d.attn_q" },
            { LLM_TENSOR_ATTN_K,          "blk.%d.attn_k" },
            { LLM_TENSOR_ATTN_V,          "blk.%d.attn_v" },
            { LLM_TENSOR_ATTN_OUT,        "blk.%d.attn_output" },
            { LLM_TENSOR_FFN_NORM,        "blk.%d.ffn_norm" },
            { LLM_TENSOR_FFN_GATE,        "blk.%d.ffn_gate" },
            { LLM_TENSOR_FFN_DOWN,        "blk.%d.ffn_down" },
            { LLM_TENSOR_FFN_UP,          "blk.%d.ffn_up" },
        },
    },
    {
        LLM_ARCH_BLOOM,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_TOKEN_EMBD_NORM, "token_embd_norm" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_OUTPUT,          "output" },
            { LLM_TENSOR_ATTN_NORM,       "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_QKV,        "blk.%d.attn_qkv" },
            { LLM_TENSOR_ATTN_OUT,        "blk.%d.attn_output" },
            { LLM_TENSOR_FFN_NORM,        "blk.%d.ffn_norm" },
            { LLM_TENSOR_FFN_UP,          "blk.%d.ffn_up" },
            { LLM_TENSOR_FFN_DOWN,        "blk.%d.ffn_down" },
        },
    },
    {
        LLM_ARCH_UNKNOWN,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
        },
    },
};

static llm_arch llm_arch_from_string(const std::string & name) {
    for (const auto & kv : LLM_ARCH_NAMES) { // NOLINT
        if (kv.second == name) {
            return kv.first;
        }
    }

    return LLM_ARCH_UNKNOWN;
}

// helper to handle gguf constants
// usage:
//
//   const auto tn = LLM_TN(LLM_ARCH_LLAMA);
//
//   std::string name = tn(LLM_TENSOR_OUTPUT);                     -> "output"
//   std::string name = tn(LLM_TENSOR_TOKEN_EMBD, "bias");         -> "token_embd.bias"
//   std::string name = tn(LLM_TENSOR_ATTN_NORM, "weight", 3);     -> "blk.3.attn_norm.weight"
//
struct LLM_TN {
    LLM_TN(llm_arch arch) : arch(arch) {}

    llm_arch arch;

    std::string operator()(llm_tensor tensor) const {
        return LLM_TENSOR_NAMES[arch].at(tensor);
    }

    std::string operator()(llm_tensor tensor, const std::string & suffix) const {
        return LLM_TENSOR_NAMES[arch].at(tensor) + "." + suffix;
    }

    std::string operator()(llm_tensor tensor, int bid) const {
        return ::format(LLM_TENSOR_NAMES[arch].at(tensor).c_str(), bid);
    }

    std::string operator()(llm_tensor tensor, const std::string & suffix, int bid) const {
        return ::format(LLM_TENSOR_NAMES[arch].at(tensor).c_str(), bid) + "." + suffix;
    }
};

//
// gguf helpers
//

#define GGUF_GET_KEY(ctx, dst, func, type, req, key) \
do { \
    const std::string skey(key); \
    const int kid = gguf_find_key(ctx, skey.c_str()); \
    if (kid >= 0) { \
        enum gguf_type ktype = gguf_get_kv_type(ctx, kid); \
        if (ktype != (type)) { \
            throw std::runtime_error(format("key %s has wrong type: %s", skey.c_str(), gguf_type_name(ktype))); \
        } \
        (dst) = func(ctx, kid); \
    } else if (req) { \
        throw std::runtime_error(format("key not found in model: %s", skey.c_str())); \
    } \
} while (0)

//
// ggml helpers
//

static void ggml_graph_compute_helper(std::vector<uint8_t> & buf, ggml_cgraph * graph, int n_threads) {
    struct ggml_cplan plan = ggml_graph_plan(graph, n_threads);

    if (plan.work_size > 0) {
        buf.resize(plan.work_size);
        plan.work_data = buf.data();
    }

    ggml_graph_compute(graph, &plan);
}

//
// llama helpers
//

#ifdef GGML_USE_CUBLAS
#   define llama_host_malloc(n)  ggml_cuda_host_malloc(n)
#   define llama_host_free(data) ggml_cuda_host_free(data)
#elif GGML_USE_METAL
#   define llama_host_malloc(n)  ggml_metal_host_malloc(n)
#   define llama_host_free(data) ggml_metal_host_free(data)
#elif GGML_USE_CPU_HBM
#   define llama_host_malloc(n)  hbw_malloc(n)
#   define llama_host_free(data) if (data != NULL) hbw_free(data)
#else
#   define llama_host_malloc(n)  malloc(n)
#   define llama_host_free(data) free(data)
#endif

#if defined(_WIN32)
static std::string llama_format_win_err(DWORD err) {
    LPSTR buf;
    size_t size = FormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
                                 NULL, err, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (LPSTR)&buf, 0, NULL);
    if (!size) {
        return "FormatMessageA failed";
    }
    std::string ret(buf, size);
    LocalFree(buf);
    return ret;
}
#endif

struct llama_buffer {
    void * data = NULL;
    size_t size = 0;

    // fallback to malloc / free
    // useful in cases where CUDA can try to allocate PINNED memory
    bool fallback = false;

    void resize(size_t n) {
        llama_host_free(data);

        data = llama_host_malloc(n);
        if (!data) {
            fallback = true;
            data = malloc(n);
        } else {
            fallback = false;
        }

        GGML_ASSERT(data);
        size = n;
    }

    ~llama_buffer() {
        if (data) {
            if (fallback) { // NOLINT
                free(data);
            } else {
                llama_host_free(data);
            }
        }

        data = NULL;
    }
};

struct llama_file {
    // use FILE * so we don't have to re-open the file to mmap
    FILE * fp;
    size_t size;

    llama_file(const char * fname, const char * mode) {
        fp = std::fopen(fname, mode);
        if (fp == NULL) {
            throw std::runtime_error(format("failed to open %s: %s", fname, strerror(errno)));
        }
        seek(0, SEEK_END);
        size = tell();
        seek(0, SEEK_SET);
    }

    size_t tell() const {
#ifdef _WIN32
        __int64 ret = _ftelli64(fp);
#else
        long ret = std::ftell(fp);
#endif
        GGML_ASSERT(ret != -1); // this really shouldn't fail
        return (size_t) ret;
    }

    void seek(size_t offset, int whence) const {
#ifdef _WIN32
        int ret = _fseeki64(fp, (__int64) offset, whence);
#else
        int ret = std::fseek(fp, (long) offset, whence);
#endif
        GGML_ASSERT(ret == 0); // same
    }

    void read_raw(void * ptr, size_t len) const {
        if (len == 0) {
            return;
        }
        errno = 0;
        std::size_t ret = std::fread(ptr, len, 1, fp);
        if (ferror(fp)) {
            throw std::runtime_error(format("read error: %s", strerror(errno)));
        }
        if (ret != 1) {
            throw std::runtime_error(std::string("unexpectedly reached end of file"));
        }
    }

    uint32_t read_u32() const {
        uint32_t ret;
        read_raw(&ret, sizeof(ret));
        return ret;
    }

    void write_raw(const void * ptr, size_t len) const {
        if (len == 0) {
            return;
        }
        errno = 0;
        size_t ret = std::fwrite(ptr, len, 1, fp);
        if (ret != 1) {
            throw std::runtime_error(format("write error: %s", strerror(errno)));
        }
    }

    void write_u32(std::uint32_t val) const {
        write_raw(&val, sizeof(val));
    }

    ~llama_file() {
        if (fp) {
            std::fclose(fp);
        }
    }
};

struct llama_mmap {
    void * addr;
    size_t size;

    llama_mmap(const llama_mmap &) = delete;

#ifdef _POSIX_MAPPED_FILES
    static constexpr bool SUPPORTED = true;

    llama_mmap(struct llama_file * file, size_t prefetch = (size_t) -1 /* -1 = max value */, bool numa = false) {
        size = file->size;
        int fd = fileno(file->fp);
        int flags = MAP_SHARED;
        // prefetch/readahead impairs performance on NUMA systems
        if (numa) { prefetch = 0; }
#ifdef __linux__
        if (prefetch) { flags |= MAP_POPULATE; }
#endif
        addr = mmap(NULL, file->size, PROT_READ, flags, fd, 0);
        if (addr == MAP_FAILED) {
            throw std::runtime_error(format("mmap failed: %s", strerror(errno)));
        }

        if (prefetch > 0) {
            // Advise the kernel to preload the mapped memory
            if (posix_madvise(addr, std::min(file->size, prefetch), POSIX_MADV_WILLNEED)) {
                fprintf(stderr, "warning: posix_madvise(.., POSIX_MADV_WILLNEED) failed: %s\n",
                        strerror(errno));
            }
        }
        if (numa) {
            // advise the kernel not to use readahead
            // (because the next page might not belong on the same node)
            if (posix_madvise(addr, file->size, POSIX_MADV_RANDOM)) {
                fprintf(stderr, "warning: posix_madvise(.., POSIX_MADV_RANDOM) failed: %s\n",
                        strerror(errno));
            }
        }
    }

    ~llama_mmap() {
        munmap(addr, size);
    }
#elif defined(_WIN32)
    static constexpr bool SUPPORTED = true;

    llama_mmap(struct llama_file * file, bool prefetch = true, bool numa = false) {
        (void) numa;

        size = file->size;

        HANDLE hFile = (HANDLE) _get_osfhandle(_fileno(file->fp));

        HANDLE hMapping = CreateFileMappingA(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
        DWORD error = GetLastError();

        if (hMapping == NULL) {
            throw std::runtime_error(format("CreateFileMappingA failed: %s", llama_format_win_err(error).c_str()));
        }

        addr = MapViewOfFile(hMapping, FILE_MAP_READ, 0, 0, 0);
        error = GetLastError();
        CloseHandle(hMapping);

        if (addr == NULL) {
            throw std::runtime_error(format("MapViewOfFile failed: %s", llama_format_win_err(error).c_str()));
        }

        if (prefetch) {
            // PrefetchVirtualMemory is only present on Windows 8 and above, so we dynamically load it
            BOOL (WINAPI *pPrefetchVirtualMemory) (HANDLE, ULONG_PTR, PWIN32_MEMORY_RANGE_ENTRY, ULONG);
            HMODULE hKernel32 = GetModuleHandleW(L"kernel32.dll");

            // may fail on pre-Windows 8 systems
            pPrefetchVirtualMemory = reinterpret_cast<decltype(pPrefetchVirtualMemory)> (GetProcAddress(hKernel32, "PrefetchVirtualMemory"));

            if (pPrefetchVirtualMemory) {
                // advise the kernel to preload the mapped memory
                WIN32_MEMORY_RANGE_ENTRY range;
                range.VirtualAddress = addr;
                range.NumberOfBytes = (SIZE_T)size;
                if (!pPrefetchVirtualMemory(GetCurrentProcess(), 1, &range, 0)) {
                    fprintf(stderr, "warning: PrefetchVirtualMemory failed: %s\n",
                            llama_format_win_err(GetLastError()).c_str());
                }
            }
        }
    }

    ~llama_mmap() {
        if (!UnmapViewOfFile(addr)) {
            fprintf(stderr, "warning: UnmapViewOfFile failed: %s\n",
                    llama_format_win_err(GetLastError()).c_str());
        }
    }
#else
    static constexpr bool SUPPORTED = false;

    llama_mmap(struct llama_file * file, bool prefetch = true, bool numa = false) {
        (void) file;
        (void) prefetch;
        (void) numa;

        throw std::runtime_error(std::string("mmap not supported"));
    }
#endif
};

// Represents some region of memory being locked using mlock or VirtualLock;
// will automatically unlock on destruction.
struct llama_mlock {
    void * addr = NULL;
    size_t size = 0;

    bool failed_already = false;

    llama_mlock() {}
    llama_mlock(const llama_mlock &) = delete;

    ~llama_mlock() {
        if (size) {
            raw_unlock(addr, size);
        }
    }

    void init(void * ptr) {
        GGML_ASSERT(addr == NULL && size == 0); // NOLINT
        addr = ptr;
    }

    void grow_to(size_t target_size) {
        GGML_ASSERT(addr);
        if (failed_already) {
            return;
        }
        size_t granularity = lock_granularity();
        target_size = (target_size + granularity - 1) & ~(granularity - 1);
        if (target_size > size) {
            if (raw_lock((uint8_t *) addr + size, target_size - size)) {
                size = target_size;
            } else {
                failed_already = true;
            }
        }
    }

#ifdef _POSIX_MEMLOCK_RANGE
    static constexpr bool SUPPORTED = true;

    static size_t lock_granularity() {
        return (size_t) sysconf(_SC_PAGESIZE);
    }

    #ifdef __APPLE__
        #define MLOCK_SUGGESTION \
            "Try increasing the sysctl values 'vm.user_wire_limit' and 'vm.global_user_wire_limit' and/or " \
            "decreasing 'vm.global_no_user_wire_amount'.  Also try increasing RLIMIT_MLOCK (ulimit -l).\n"
    #else
        #define MLOCK_SUGGESTION \
            "Try increasing RLIMIT_MLOCK ('ulimit -l' as root).\n"
    #endif

    bool raw_lock(const void * addr, size_t size) const {
        if (!mlock(addr, size)) {
            return true;
        }

        char* errmsg = std::strerror(errno);
        bool suggest = (errno == ENOMEM);

        // Check if the resource limit is fine after all
        struct rlimit lock_limit;
        if (suggest && getrlimit(RLIMIT_MEMLOCK, &lock_limit)) {
            suggest = false;
        }
        if (suggest && (lock_limit.rlim_max > lock_limit.rlim_cur + size)) {
            suggest = false;
        }

        fprintf(stderr, "warning: failed to mlock %zu-byte buffer (after previously locking %zu bytes): %s\n%s",
                size, this->size, errmsg, suggest ? MLOCK_SUGGESTION : "");
        return false;
    }

    #undef MLOCK_SUGGESTION

    static void raw_unlock(void * addr, size_t size) {
        if (munlock(addr, size)) {
            fprintf(stderr, "warning: failed to munlock buffer: %s\n", std::strerror(errno));
        }
    }
#elif defined(_WIN32)
    static constexpr bool SUPPORTED = true;

    static size_t lock_granularity() {
        SYSTEM_INFO si;
        GetSystemInfo(&si);
        return (size_t) si.dwPageSize;
    }

    bool raw_lock(void * ptr, size_t len) const {
        for (int tries = 1; ; tries++) {
            if (VirtualLock(ptr, len)) {
                return true;
            }
            if (tries == 2) {
                fprintf(stderr, "warning: failed to VirtualLock %zu-byte buffer (after previously locking %zu bytes): %s\n",
                    len, size, llama_format_win_err(GetLastError()).c_str());
                return false;
            }

            // It failed but this was only the first try; increase the working
            // set size and try again.
            SIZE_T min_ws_size, max_ws_size;
            if (!GetProcessWorkingSetSize(GetCurrentProcess(), &min_ws_size, &max_ws_size)) {
                fprintf(stderr, "warning: GetProcessWorkingSetSize failed: %s\n",
                        llama_format_win_err(GetLastError()).c_str());
                return false;
            }
            // Per MSDN: "The maximum number of pages that a process can lock
            // is equal to the number of pages in its minimum working set minus
            // a small overhead."
            // Hopefully a megabyte is enough overhead:
            size_t increment = len + 1048576;
            // The minimum must be <= the maximum, so we need to increase both:
            min_ws_size += increment;
            max_ws_size += increment;
            if (!SetProcessWorkingSetSize(GetCurrentProcess(), min_ws_size, max_ws_size)) {
                fprintf(stderr, "warning: SetProcessWorkingSetSize failed: %s\n",
                        llama_format_win_err(GetLastError()).c_str());
                return false;
            }
        }
    }

    static void raw_unlock(void * ptr, size_t len) {
        if (!VirtualUnlock(ptr, len)) {
            fprintf(stderr, "warning: failed to VirtualUnlock buffer: %s\n",
                    llama_format_win_err(GetLastError()).c_str());
        }
    }
#else
    static constexpr bool SUPPORTED = false;

    static size_t lock_granularity() {
        return (size_t) 65536;
    }

    bool raw_lock(const void * addr, size_t len) const {
        fprintf(stderr, "warning: mlock not supported on this system\n");
        return false;
    }

    static void raw_unlock(const void * addr, size_t len) {}
#endif
};

typedef void (*offload_func_t)(struct ggml_tensor * tensor);

static void llama_nop(struct ggml_tensor * tensor) { // don't offload by default
    (void) tensor;
}

static std::string llama_token_to_str(const struct llama_context * ctx, llama_token token) {
    std::vector<char> result(8, 0);
    const int n_tokens = llama_token_to_piece(llama_get_model(ctx), token, result.data(), result.size());
    if (n_tokens < 0) {
        result.resize(-n_tokens);
        int check = llama_token_to_piece(llama_get_model(ctx), token, result.data(), result.size());
        GGML_ASSERT(check == -n_tokens);
    } else {
        result.resize(n_tokens);
    }

    return std::string(result.data(), result.size());
}

//
// globals
//

struct llama_state {
    // We save the log callback globally
    ggml_log_callback log_callback = llama_log_callback_default;
    void * log_callback_user_data = nullptr;
};

static llama_state g_state;

// available llama models
enum e_model {
    MODEL_UNKNOWN,
    MODEL_1B,
    MODEL_3B,
    MODEL_7B,
    MODEL_8B,
    MODEL_13B,
    MODEL_15B,
    MODEL_30B,
    MODEL_34B,
    MODEL_40B,
    MODEL_65B,
    MODEL_70B,
};

static const size_t kB = 1024;
static const size_t MB = 1024*kB;
static const size_t GB = 1024*MB;

struct llama_hparams {
    bool     vocab_only;
    uint32_t n_vocab;
    uint32_t n_ctx_train; // context size the model was trained on
    uint32_t n_embd;
    uint32_t n_head;
    uint32_t n_head_kv;
    uint32_t n_layer;
    uint32_t n_rot;
    uint32_t n_ff;

    float f_norm_eps;
    float f_norm_rms_eps;

    float rope_freq_base_train;
    float rope_freq_scale_train;

    float f_clamp_kqv;
    float f_max_alibi_bias;

    bool operator!=(const llama_hparams & other) const {
        if (this->vocab_only  != other.vocab_only)  return true;
        if (this->n_vocab     != other.n_vocab)     return true;
        if (this->n_ctx_train != other.n_ctx_train) return true;
        if (this->n_embd      != other.n_embd)      return true;
        if (this->n_head      != other.n_head)      return true;
        if (this->n_head_kv   != other.n_head_kv)   return true;
        if (this->n_layer     != other.n_layer)     return true;
        if (this->n_rot       != other.n_rot)       return true;
        if (this->n_ff        != other.n_ff)        return true;

        const float EPSILON = 1e-9;

        if (!is_float_close(this->f_norm_eps,            other.f_norm_eps,            EPSILON)) return true;
        if (!is_float_close(this->f_norm_rms_eps,        other.f_norm_rms_eps,        EPSILON)) return true;
        if (!is_float_close(this->rope_freq_base_train,  other.rope_freq_base_train,  EPSILON)) return true;
        if (!is_float_close(this->rope_freq_scale_train, other.rope_freq_scale_train, EPSILON)) return true;

        return false;
    }

    uint32_t n_gqa() const {
        return n_head/n_head_kv;
    }

    uint32_t n_embd_head() const {
        return n_embd/n_head;
    }

    uint32_t n_embd_gqa() const {
        return n_embd/n_gqa();
    }
};

struct llama_cparams {
    uint32_t n_ctx;       // context size used during inference
    uint32_t n_batch;
    uint32_t n_threads;       // number of threads to use for generation
    uint32_t n_threads_batch; // number of threads to use for batch processing

    float rope_freq_base;
    float rope_freq_scale;

    bool mul_mat_q;
};

struct llama_layer {
    // normalization
    struct ggml_tensor * attn_norm;
    struct ggml_tensor * attn_norm_b;
    struct ggml_tensor * attn_norm_2;
    struct ggml_tensor * attn_norm_2_b;
    struct ggml_tensor * attn_q_norm;
    struct ggml_tensor * attn_q_norm_b;
    struct ggml_tensor * attn_k_norm;
    struct ggml_tensor * attn_k_norm_b;

    // attention
    struct ggml_tensor * wq;
    struct ggml_tensor * wk;
    struct ggml_tensor * wv;
    struct ggml_tensor * wo;
    struct ggml_tensor * wqkv;

    // attention bias
    struct ggml_tensor * bo;
    struct ggml_tensor * bqkv;

    // normalization
    struct ggml_tensor * ffn_norm;
    struct ggml_tensor * ffn_norm_b;

    // ff
    struct ggml_tensor * w1; // ffn_gate
    struct ggml_tensor * w2; // ffn_down
    struct ggml_tensor * w3; // ffn_up

    // ff bias
    struct ggml_tensor * b2; // ffn_down
    struct ggml_tensor * b3; // ffn_up
};

struct llama_kv_cell {
    llama_pos pos   = -1;
    llama_pos delta = 0;

    std::set<llama_seq_id> seq_id;

    bool has_seq_id(const llama_seq_id & id) const {
        return seq_id.find(id) != seq_id.end();
    }
};

// ring-buffer of cached KV data
struct llama_kv_cache {
    bool has_shift = false;

    // Note: The value of head isn't only used to optimize searching
    // for a free KV slot. llama_decode_internal also uses it, so it
    // cannot be freely changed after a slot has been allocated.
    uint32_t head = 0;
    uint32_t size = 0;

    // computed before each graph build
    uint32_t n = 0;

    std::vector<llama_kv_cell> cells;

    struct ggml_tensor * k = NULL;
    struct ggml_tensor * v = NULL;

    struct ggml_context * ctx = NULL;

    llama_buffer buf;

    ~llama_kv_cache() {
        if (ctx) {
            ggml_free(ctx);
        }

#ifdef GGML_USE_CUBLAS
        ggml_cuda_free_data(k);
        ggml_cuda_free_data(v);
#endif // GGML_USE_CUBLAS
    }
};

struct llama_vocab {
    using id    = int32_t;
    using token = std::string;
    using ttype = llama_token_type;

    struct token_data {
        token text;
        float score;
        ttype type;
    };

    enum llama_vocab_type type = LLAMA_VOCAB_TYPE_SPM;

    std::unordered_map<token, id> token_to_id;
    std::vector<token_data>       id_to_token;

    std::unordered_map<token, id> special_tokens_cache;

    std::map<std::pair<std::string, std::string>, int> bpe_ranks;

    // default LLaMA special tokens
    id special_bos_id = 1;
    id special_eos_id = 2;
    id special_unk_id = 0;
    id special_sep_id = -1;
    id special_pad_id = -1;

    id linefeed_id       = 13;
    id special_prefix_id = 32007;
    id special_middle_id = 32009;
    id special_suffix_id = 32008;
    id special_eot_id    = 32010;

    int find_bpe_rank(std::string token_left, std::string token_right) const {
        replace_all(token_left,  " ",  "\u0120");
        replace_all(token_left,  "\n", "\u010A");
        replace_all(token_right, " ",  "\u0120");
        replace_all(token_right, "\n", "\u010A");

        auto it = bpe_ranks.find(std::make_pair(token_left, token_right));
        if (it == bpe_ranks.end()) {
            return -1;
        }

        return it->second;
    }
};

struct llama_model {
    e_model     type  = MODEL_UNKNOWN;
    llm_arch    arch  = LLM_ARCH_UNKNOWN;
    llama_ftype ftype = LLAMA_FTYPE_ALL_F32;

    std::string name = "n/a";

    llama_hparams hparams = {};
    llama_vocab   vocab;

    struct ggml_tensor * tok_embeddings;
    struct ggml_tensor * pos_embeddings;
    struct ggml_tensor * tok_norm;
    struct ggml_tensor * tok_norm_b;

    struct ggml_tensor * output_norm;
    struct ggml_tensor * output_norm_b;
    struct ggml_tensor * output;

    std::vector<llama_layer> layers;

    int n_gpu_layers;

    // context
    struct ggml_context * ctx = NULL;

    // the model memory buffer
    llama_buffer buf;

    // model memory mapped file
    std::unique_ptr<llama_mmap> mapping;

    // objects representing data potentially being locked in memory
    llama_mlock mlock_buf;
    llama_mlock mlock_mmap;

    // for quantize-stats only
    std::vector<std::pair<std::string, struct ggml_tensor *>> tensors_by_name;

    int64_t t_load_us = 0;
    int64_t t_start_us = 0;

    ~llama_model() {
        if (ctx) {
            ggml_free(ctx);
        }

#ifdef GGML_USE_CUBLAS
        for (size_t i = 0; i < tensors_by_name.size(); ++i) {
            ggml_cuda_free_data(tensors_by_name[i].second);
        }
        ggml_cuda_free_scratch();
#elif defined(GGML_USE_CLBLAST)
        for (size_t i = 0; i < tensors_by_name.size(); ++i) {
            ggml_cl_free_data(tensors_by_name[i].second);
        }
#endif
    }
};

struct llama_context {
    llama_context(const llama_model & model) : model(model), t_start_us(model.t_start_us), t_load_us(model.t_load_us) {}
    ~llama_context() {
#ifdef GGML_USE_METAL
        if (ctx_metal) {
            ggml_metal_free(ctx_metal);
        }
#endif
        if (alloc) {
            ggml_allocr_free(alloc);
        }
    }

    llama_cparams cparams;

    const llama_model & model;

    // key + value cache for the self attention
    struct llama_kv_cache kv_self;

    std::mt19937 rng;

    bool has_evaluated_once = false;

    int64_t t_start_us;
    int64_t t_load_us;
    int64_t t_sample_us = 0;
    int64_t t_p_eval_us = 0;
    int64_t t_eval_us   = 0;

    int32_t n_sample = 0; // number of tokens sampled
    int32_t n_p_eval = 0; // number of tokens in eval calls for the prompt (with batch size > 1)
    int32_t n_eval   = 0; // number of eval calls

    // decode output (2-dimensional array: [n_tokens][n_vocab])
    std::vector<float> logits;
    bool logits_all = false;

    // input embedding (1-dimensional array: [n_embd])
    std::vector<float> embedding;

    // reusable buffer for `struct ggml_graph_plan.work_data`
    std::vector<uint8_t> work_buffer;

    // memory buffers used to evaluate the model
    llama_buffer buf_compute;

    llama_buffer buf_alloc;
    ggml_allocr * alloc = NULL;

#ifdef GGML_USE_METAL
    ggml_metal_context * ctx_metal = NULL;
#endif

#ifdef GGML_USE_MPI
    ggml_mpi_context * ctx_mpi = NULL;
#endif
};

//
// kv cache helpers
//

static bool llama_kv_cache_init(
        const struct llama_hparams & hparams,
             struct llama_kv_cache & cache,
                         ggml_type   wtype,
                          uint32_t   n_ctx,
                               int   n_gpu_layers) {
    const uint32_t n_embd  = hparams.n_embd_gqa();
    const uint32_t n_layer = hparams.n_layer;

    const int64_t n_mem      = n_layer*n_ctx;
    const int64_t n_elements = n_embd*n_mem;

    cache.has_shift = false;

    cache.head = 0;
    cache.size = n_ctx;

    cache.cells.clear();
    cache.cells.resize(n_ctx);

    cache.buf.resize(2u*n_elements*ggml_type_size(wtype) + 2u*ggml_tensor_overhead());
    memset(cache.buf.data, 0, cache.buf.size);

    struct ggml_init_params params;
    params.mem_size   = cache.buf.size;
    params.mem_buffer = cache.buf.data;
    params.no_alloc   = false;

    cache.ctx = ggml_init(params);

    if (!cache.ctx) {
        LLAMA_LOG_ERROR("%s: failed to allocate memory for kv cache\n", __func__);
        return false;
    }

    cache.k = ggml_new_tensor_1d(cache.ctx, wtype, n_elements);
    cache.v = ggml_new_tensor_1d(cache.ctx, wtype, n_elements);
    ggml_set_name(cache.k, "cache_k");
    ggml_set_name(cache.v, "cache_v");

    (void) n_gpu_layers;
#ifdef GGML_USE_CUBLAS
    size_t vram_kv_cache = 0;

    if (n_gpu_layers > (int)n_layer + 1) {
        ggml_cuda_assign_buffers_no_scratch(cache.v);
        LLAMA_LOG_INFO("%s: offloading v cache to GPU\n", __func__);
        vram_kv_cache += ggml_nbytes(cache.v);
    }
    if (n_gpu_layers > (int)n_layer + 2) {
        ggml_cuda_assign_buffers_no_scratch(cache.k);
        LLAMA_LOG_INFO("%s: offloading k cache to GPU\n", __func__);
        vram_kv_cache += ggml_nbytes(cache.k);
    }
    if (vram_kv_cache > 0) {
        LLAMA_LOG_INFO("%s: VRAM kv self = %.2f MB\n", __func__, vram_kv_cache / 1024.0 / 1024.0);
    }
#endif // GGML_USE_CUBLAS

    return true;
}

// find an empty slot of size "n_tokens" in the cache
// updates the cache head
// Note: On success, it's important that cache.head points
// to the first cell of the slot.
static bool llama_kv_cache_find_slot(
           struct llama_kv_cache & cache,
        const struct llama_batch & batch) {
    const uint32_t n_ctx    = cache.size;
    const uint32_t n_tokens = batch.n_tokens;

    if (n_tokens > n_ctx) {
        LLAMA_LOG_ERROR("%s: n_tokens=%d > n_ctx=%d\n", __func__, n_tokens, n_ctx);
        return false;
    }

    uint32_t n_tested = 0;

    while (true) {
        if (cache.head + n_tokens > n_ctx) {
            n_tested += n_ctx - cache.head;
            cache.head = 0;
            continue;
        }

        bool found = true;
        for (uint32_t i = 0; i < n_tokens; i++) {
            if (cache.cells[cache.head + i].pos >= 0) {
                found = false;
                cache.head += i + 1;
                n_tested   += i + 1;
                break;
            }
        }

        if (found) {
            break;
        }

        if (n_tested >= n_ctx) {
            //LLAMA_LOG_ERROR("%s: failed to find a slot for %d tokens\n", __func__, n_tokens);
            return false;
        }
    }

    for (uint32_t i = 0; i < n_tokens; i++) {
        cache.cells[cache.head + i].pos = batch.pos[i];

        for (int32_t j = 0; j < batch.n_seq_id[i]; j++) {
            cache.cells[cache.head + i].seq_id.insert(batch.seq_id[i][j]);
        }
    }

    return true;
}

// find how many cells are currently in use
static int32_t llama_kv_cache_cell_max(const struct llama_kv_cache & cache) {
    for (uint32_t i = cache.size - 1; i > 0; --i) {
        if (cache.cells[i].pos >= 0 && !cache.cells[i].seq_id.empty()) {
            return i + 1;
        }
    }

    return 0;
}

static void llama_kv_cache_tokens_rm(struct llama_kv_cache & cache, int32_t c0, int32_t c1) {
    if (c0 < 0) c0 = 0;
    if (c1 < 0) c1 = cache.size;

    for (int32_t i = c0; i < c1; ++i) {
        cache.cells[i].pos = -1;
        cache.cells[i].seq_id.clear();
    }

    // Searching for a free slot can start here since we know it will be empty.
    cache.head = uint32_t(c0);
}

static void llama_kv_cache_seq_rm(
        struct llama_kv_cache & cache,
                 llama_seq_id   seq_id,
                    llama_pos   p0,
                    llama_pos   p1) {
    uint32_t new_head = cache.size;

    if (p0 < 0) p0 = 0;
    if (p1 < 0) p1 = std::numeric_limits<llama_pos>::max();

    for (uint32_t i = 0; i < cache.size; ++i) {
        if (cache.cells[i].has_seq_id(seq_id) && cache.cells[i].pos >= p0 && cache.cells[i].pos < p1) {
            cache.cells[i].seq_id.erase(seq_id);
            if (cache.cells[i].seq_id.empty()) {
                cache.cells[i].pos = -1;
                if (new_head == cache.size) new_head = i;
            }
        }
    }

    // If we freed up a slot, set head to it so searching can start there.
    if (new_head != cache.size) cache.head = new_head;
}

static void llama_kv_cache_seq_cp(
        struct llama_kv_cache & cache,
                 llama_seq_id   seq_id_src,
                 llama_seq_id   seq_id_dst,
                    llama_pos   p0,
                    llama_pos   p1) {
    if (p0 < 0) p0 = 0;
    if (p1 < 0) p1 = std::numeric_limits<llama_pos>::max();

    cache.head = 0;

    for (uint32_t i = 0; i < cache.size; ++i) {
        if (cache.cells[i].has_seq_id(seq_id_src) && cache.cells[i].pos >= p0 && cache.cells[i].pos < p1) {
            cache.cells[i].seq_id.insert(seq_id_dst);
        }
    }
}

static void llama_kv_cache_seq_keep(struct llama_kv_cache & cache, llama_seq_id seq_id) {
    uint32_t new_head = cache.size;

    for (uint32_t i = 0; i < cache.size; ++i) {
        if (!cache.cells[i].has_seq_id(seq_id)) {
            cache.cells[i].pos = -1;
            cache.cells[i].seq_id.clear();
            if (new_head == cache.size) new_head = i;
        } else {
            cache.cells[i].seq_id.clear();
            cache.cells[i].seq_id.insert(seq_id);
        }
    }

    // If we freed up a slot, set head to it so searching can start there.
    if (new_head != cache.size) cache.head = new_head;
}

static void llama_kv_cache_seq_shift(
        struct llama_kv_cache & cache,
                 llama_seq_id   seq_id,
                    llama_pos   p0,
                    llama_pos   p1,
                    llama_pos   delta) {
    uint32_t new_head = cache.size;

    if (p0 < 0) p0 = 0;
    if (p1 < 0) p1 = std::numeric_limits<llama_pos>::max();

    for (uint32_t i = 0; i < cache.size; ++i) {
        if (cache.cells[i].has_seq_id(seq_id) && cache.cells[i].pos >= p0 && cache.cells[i].pos < p1) {
            cache.cells[i].pos += delta;
            if (cache.cells[i].pos < 0) {
                cache.cells[i].pos = -1;
                cache.cells[i].seq_id.clear();
                if (new_head == cache.size) new_head = i;
            } else {
                cache.has_shift = true;
                cache.cells[i].delta = delta;
            }
        }
    }

    // If we freed up a slot, set head to it so searching can start there.
    // Otherwise we just start the next search from the beginning.
    cache.head = new_head != cache.size ? new_head : 0;
}

//
// model loading and saving
//

enum llama_fver {
    GGUF_FILE_VERSION_V1 = 1,
    GGUF_FILE_VERSION_V2 = 2,
};

static const char * llama_file_version_name(llama_fver version) {
    switch (version) {
        case GGUF_FILE_VERSION_V1: return "GGUF V1 (support until nov 2023)";
        case GGUF_FILE_VERSION_V2: return "GGUF V2 (latest)";
    }

    return "unknown";
}

static std::string llama_format_tensor_shape(const std::vector<int64_t> & ne) {
    char buf[256];
    snprintf(buf, sizeof(buf), "%5" PRId64, ne.at(0));
    for (size_t i = 1; i < ne.size(); i++) {
        snprintf(buf + strlen(buf), sizeof(buf) - strlen(buf), ", %5" PRId64, ne.at(i));
    }
    return buf;
}

static std::string llama_format_tensor_shape(const struct ggml_tensor * t) {
    char buf[256];
    snprintf(buf, sizeof(buf), "%5" PRId64, t->ne[0]);
    for (int i = 1; i < GGML_MAX_DIMS; i++) {
        snprintf(buf + strlen(buf), sizeof(buf) - strlen(buf), ", %5" PRId64, t->ne[i]);
    }
    return buf;
}

struct llama_model_loader {
    int n_kv      = 0;
    int n_tensors = 0;
    int n_created = 0;

    int64_t n_elements = 0;
    size_t  n_bytes    = 0;

    bool use_mmap = false;

    llama_file  file;
    llama_ftype ftype;
    llama_fver  fver;

    std::unique_ptr<llama_mmap> mapping;

    struct gguf_context * ctx_gguf = NULL;
    struct ggml_context * ctx_meta = NULL;

    llama_model_loader(const std::string & fname, bool use_mmap) : file(fname.c_str(), "rb") {
        struct gguf_init_params params = {
            /*.no_alloc = */ true,
            /*.ctx      = */ &ctx_meta,
        };

        ctx_gguf = gguf_init_from_file(fname.c_str(), params);
        if (!ctx_gguf) {
            throw std::runtime_error(format("%s: failed to load model from %s\n", __func__, fname.c_str()));
        }

        n_kv      = gguf_get_n_kv(ctx_gguf);
        n_tensors = gguf_get_n_tensors(ctx_gguf);

        fver = (enum llama_fver ) gguf_get_version(ctx_gguf);

        for (int i = 0; i < n_tensors; i++) {
            const char * name = gguf_get_tensor_name(ctx_gguf, i);
            struct ggml_tensor * t = ggml_get_tensor(ctx_meta, name);
            n_elements += ggml_nelements(t);
            n_bytes    += ggml_nbytes(t);
        }

        LLAMA_LOG_INFO("%s: loaded meta data with %d key-value pairs and %d tensors from %s (version %s)\n",
                __func__, n_kv, n_tensors, fname.c_str(), llama_file_version_name(fver));

        // determine file type based on the number of tensors for each quantization and print meta data
        // TODO: make optional
        {
            std::map<enum ggml_type, uint32_t> n_type;

            uint32_t n_type_max = 0;
            enum ggml_type type_max = GGML_TYPE_F32;

            for (int i = 0; i < n_tensors; i++) {
                const char * name = gguf_get_tensor_name(ctx_gguf, i);
                struct ggml_tensor * meta = ggml_get_tensor(ctx_meta, name);

                n_type[meta->type]++;

                if (n_type_max < n_type[meta->type]) {
                    n_type_max = n_type[meta->type];
                    type_max   = meta->type;
                }

                LLAMA_LOG_INFO("%s: - tensor %4d: %32s %-8s [ %s ]\n", __func__, i, name, ggml_type_name(meta->type), llama_format_tensor_shape(meta).c_str());
            }

            switch (type_max) {
                case GGML_TYPE_F32:  ftype = LLAMA_FTYPE_ALL_F32;       break;
                case GGML_TYPE_F16:  ftype = LLAMA_FTYPE_MOSTLY_F16;    break;
                case GGML_TYPE_Q4_0: ftype = LLAMA_FTYPE_MOSTLY_Q4_0;   break;
                case GGML_TYPE_Q4_1: ftype = LLAMA_FTYPE_MOSTLY_Q4_1;   break;
                case GGML_TYPE_Q5_0: ftype = LLAMA_FTYPE_MOSTLY_Q5_0;   break;
                case GGML_TYPE_Q5_1: ftype = LLAMA_FTYPE_MOSTLY_Q5_1;   break;
                case GGML_TYPE_Q8_0: ftype = LLAMA_FTYPE_MOSTLY_Q8_0;   break;
                case GGML_TYPE_Q2_K: ftype = LLAMA_FTYPE_MOSTLY_Q2_K;   break;
                case GGML_TYPE_Q3_K: ftype = LLAMA_FTYPE_MOSTLY_Q3_K_M; break;
                case GGML_TYPE_Q4_K: ftype = LLAMA_FTYPE_MOSTLY_Q4_K_M; break;
                case GGML_TYPE_Q5_K: ftype = LLAMA_FTYPE_MOSTLY_Q5_K_M; break;
                case GGML_TYPE_Q6_K: ftype = LLAMA_FTYPE_MOSTLY_Q6_K;   break;
                default:
                     {
                         LLAMA_LOG_WARN("%s: unknown type %s\n", __func__, ggml_type_name(type_max));
                         ftype = LLAMA_FTYPE_ALL_F32;
                     } break;
            }

            // this is a way to mark that we have "guessed" the file type
            ftype = (llama_ftype) (ftype | LLAMA_FTYPE_GUESSED);

            {
                const int kid = gguf_find_key(ctx_gguf, "general.file_type");
                if (kid >= 0) {
                    ftype = (llama_ftype) gguf_get_val_u32(ctx_gguf, kid);
                }
            }

            for (int i = 0; i < n_kv; i++) {
                const char * name         = gguf_get_key(ctx_gguf, i);
                const enum gguf_type type = gguf_get_kv_type(ctx_gguf, i);

                LLAMA_LOG_INFO("%s: - kv %3d: %42s %-8s\n", __func__, i, name, gguf_type_name(type));
            }

            // print type counts
            for (auto & kv : n_type) {
                if (kv.second == 0) {
                    continue;
                }

                LLAMA_LOG_INFO("%s: - type %4s: %4d tensors\n", __func__, ggml_type_name(kv.first), kv.second);
            }
        }

        if (!llama_mmap::SUPPORTED) {
            LLAMA_LOG_WARN("%s: mmap is not supported on this platform\n", __func__);
            use_mmap = false;
        }

        this->use_mmap = use_mmap;
    }

    ~llama_model_loader() {
        if (ctx_gguf) {
            gguf_free(ctx_gguf);
        }
        if (ctx_meta) {
            ggml_free(ctx_meta);
        }
    }

    std::string get_arch_name() const {
        const auto kv = LLM_KV(LLM_ARCH_UNKNOWN);

        std::string arch_name;
        GGUF_GET_KEY(ctx_gguf, arch_name, gguf_get_val_str, GGUF_TYPE_STRING, false, kv(LLM_KV_GENERAL_ARCHITECTURE));

        return arch_name;
    }

    enum llm_arch get_arch() const {
        const std::string arch_name = get_arch_name();

        return llm_arch_from_string(arch_name);
    }

    const char * get_tensor_name(int i) const {
        return gguf_get_tensor_name(ctx_gguf, i);
    }

    struct ggml_tensor * get_tensor_meta(int i) const {
        return ggml_get_tensor(ctx_meta, get_tensor_name(i));
    }

    void calc_sizes(size_t & ctx_size_p, size_t & mmapped_size_p) const {
        ctx_size_p     = 0;
        mmapped_size_p = 0;

        for (int i = 0; i < n_tensors; i++) {
            struct ggml_tensor * meta = get_tensor_meta(i);
            ctx_size_p += sizeof(struct ggml_tensor) + GGML_OBJECT_SIZE;
            (use_mmap ? mmapped_size_p : ctx_size_p) += ggml_nbytes_pad(meta);
        }
    }

    struct ggml_tensor * create_tensor_for(struct ggml_context * ctx, struct ggml_tensor * meta, ggml_backend_type backend) {
        if (backend != GGML_BACKEND_CPU) {
            ggml_set_no_alloc(ctx, true);
        }

        struct ggml_tensor * tensor = ggml_dup_tensor(ctx, meta);
        tensor->backend = backend; // TODO: ggml_set_backend
        ggml_set_name(tensor, ggml_get_name(meta));

        if (backend != GGML_BACKEND_CPU) {
            ggml_set_no_alloc(ctx, use_mmap);
        }

        n_created++;

        return tensor;
    }

    struct ggml_tensor * create_tensor(struct ggml_context * ctx, const std::string & name, const std::vector<int64_t> & ne, ggml_backend_type backend) {
        struct ggml_tensor * cur = ggml_get_tensor(ctx_meta, name.c_str());

        if (cur == NULL) {
            throw std::runtime_error(format("%s: tensor '%s' not found", __func__, name.c_str()));
        }

        {
            bool is_ok = true;
            for (size_t i = 0; i < ne.size(); ++i) {
                if (ne[i] != cur->ne[i]) {
                    is_ok = false;
                    break;
                }
            }
            if (!is_ok) {
                throw std::runtime_error(
                        format("%s: tensor '%s' has wrong shape; expected %s, got %s",
                            __func__, name.c_str(),
                            llama_format_tensor_shape(ne).c_str(),
                            llama_format_tensor_shape(cur).c_str()));
            }
        }

        return create_tensor_for(ctx, cur, backend);
    }

    void done_getting_tensors() const {
        if (n_created != n_tensors) {
            throw std::runtime_error(format("%s: wrong number of tensors; expected %d, got %d", __func__, n_tensors, n_created));
        }
    }

    size_t file_offset(const char * name) const {
        const int idx = gguf_find_tensor(ctx_gguf, name);

        if (idx < 0) {
            throw std::runtime_error(format("%s: tensor '%s' not found in the file", __func__, name));
        }

        return gguf_get_data_offset(ctx_gguf) + gguf_get_tensor_offset(ctx_gguf, idx);
    }

    void load_data_for(struct ggml_tensor * cur) const {
        const size_t offs = file_offset(ggml_get_name(cur));

        if (use_mmap) {
            cur->data = (uint8_t *) mapping->addr + offs;
        } else {
            file.seek(offs, SEEK_SET);
            file.read_raw(cur->data, ggml_nbytes(cur));
        }
    }

    void load_all_data(struct ggml_context * ctx, llama_progress_callback progress_callback, void * progress_callback_user_data, llama_mlock * lmlock) {
        size_t size_data = 0;
        size_t size_lock = 0;
        size_t size_pref = 0; // prefetch

        for (int i = 0; i < gguf_get_n_tensors(ctx_gguf); i++) {
            struct ggml_tensor * cur = ggml_get_tensor(ctx, gguf_get_tensor_name(ctx_gguf, i));
            size_data += ggml_nbytes(cur);
            if (cur->backend == GGML_BACKEND_CPU) {
                size_pref += ggml_nbytes(cur);
            }
        }

        if (use_mmap) {
            mapping.reset(new llama_mmap(&file, size_pref, ggml_is_numa()));
            if (lmlock) {
                lmlock->init(mapping->addr);
            }
        }

        size_t done_size = 0;
        for (int i = 0; i < gguf_get_n_tensors(ctx_gguf); i++) {
            struct ggml_tensor * cur = ggml_get_tensor(ctx, gguf_get_tensor_name(ctx_gguf, i));
            GGML_ASSERT(cur); // unused tensors should have been caught by load_data already

            if (progress_callback) {
                progress_callback((float) done_size / size_data, progress_callback_user_data);
            }

            // allocate temp buffer if not using mmap
            if (!use_mmap && cur->data == NULL) {
                GGML_ASSERT(cur->backend != GGML_BACKEND_CPU);
                #ifdef GGML_USE_CPU_HBM
                cur->data = (uint8_t*)hbw_malloc(ggml_nbytes(cur));
                #else
                cur->data = (uint8_t*)malloc(ggml_nbytes(cur));
                #endif
            }

            load_data_for(cur);

            switch (cur->backend) {
                case GGML_BACKEND_CPU:
                    if (use_mmap && lmlock) {
                        size_lock += ggml_nbytes(cur);
                        lmlock->grow_to(size_lock);
                    }
                    break;
#ifdef GGML_USE_CUBLAS
                case GGML_BACKEND_GPU:
                case GGML_BACKEND_GPU_SPLIT:
                    // old code:
                    //ggml_cuda_transform_tensor(lt.data, lt.ggml_tensor);

                    // TODO: test if this works !!
                    ggml_cuda_transform_tensor(cur->data, cur);
                    if (!use_mmap) {
                        free(cur->data);
                    }
                    break;
#elif defined(GGML_USE_CLBLAST)
                case GGML_BACKEND_GPU:
                    ggml_cl_transform_tensor(cur->data, cur);
                    if (!use_mmap) {
                        free(cur->data);
                    }
                    break;
#endif
                default:
                    continue;
            }

            done_size += ggml_nbytes(cur);
        }
    }
};

//
// load LLaMA models
//

static std::string llama_model_arch_name(llm_arch arch) {
    auto it = LLM_ARCH_NAMES.find(arch);
    if (it == LLM_ARCH_NAMES.end()) {
        return "unknown";
    }
    return it->second;
}

static std::string llama_model_ftype_name(llama_ftype ftype) {
    if (ftype & LLAMA_FTYPE_GUESSED) {
        return llama_model_ftype_name((enum llama_ftype) (ftype & ~LLAMA_FTYPE_GUESSED)) + " (guessed)";
    }

    switch (ftype) {
        case LLAMA_FTYPE_ALL_F32:     return "all F32";
        case LLAMA_FTYPE_MOSTLY_F16:  return "mostly F16";
        case LLAMA_FTYPE_MOSTLY_Q4_0: return "mostly Q4_0";
        case LLAMA_FTYPE_MOSTLY_Q4_1: return "mostly Q4_1";
        case LLAMA_FTYPE_MOSTLY_Q4_1_SOME_F16:
                                      return "mostly Q4_1, some F16";
        case LLAMA_FTYPE_MOSTLY_Q5_0: return "mostly Q5_0";
        case LLAMA_FTYPE_MOSTLY_Q5_1: return "mostly Q5_1";
        case LLAMA_FTYPE_MOSTLY_Q8_0: return "mostly Q8_0";

        // K-quants
        case LLAMA_FTYPE_MOSTLY_Q2_K:   return "mostly Q2_K";
        case LLAMA_FTYPE_MOSTLY_Q3_K_S: return "mostly Q3_K - Small";
        case LLAMA_FTYPE_MOSTLY_Q3_K_M: return "mostly Q3_K - Medium";
        case LLAMA_FTYPE_MOSTLY_Q3_K_L: return "mostly Q3_K - Large";
        case LLAMA_FTYPE_MOSTLY_Q4_K_S: return "mostly Q4_K - Small";
        case LLAMA_FTYPE_MOSTLY_Q4_K_M: return "mostly Q4_K - Medium";
        case LLAMA_FTYPE_MOSTLY_Q5_K_S: return "mostly Q5_K - Small";
        case LLAMA_FTYPE_MOSTLY_Q5_K_M: return "mostly Q5_K - Medium";
        case LLAMA_FTYPE_MOSTLY_Q6_K:   return "mostly Q6_K";

        default: return "unknown, may not work";
    }
}

static const char * llama_model_type_name(e_model type) {
    switch (type) {
        case MODEL_1B:  return "1B";
        case MODEL_3B:  return "3B";
        case MODEL_7B:  return "7B";
        case MODEL_8B:  return "8B";
        case MODEL_13B: return "13B";
        case MODEL_15B: return "15B";
        case MODEL_30B: return "30B";
        case MODEL_34B: return "34B";
        case MODEL_40B: return "40B";
        case MODEL_65B: return "65B";
        case MODEL_70B: return "70B";
        default:        return "?B";
    }
}

static void llm_load_arch(llama_model_loader & ml, llama_model & model) {
    model.arch = ml.get_arch();
    if (model.arch == LLM_ARCH_UNKNOWN) {
        throw std::runtime_error("unknown model architecture: '" + ml.get_arch_name() + "'");
    }
}

static void llm_load_hparams(
        llama_model_loader & ml,
        llama_model & model) {
    struct gguf_context * ctx = ml.ctx_gguf;

    const auto kv = LLM_KV(model.arch);

    auto & hparams = model.hparams;

    // get general kv
    GGUF_GET_KEY(ctx, model.name, gguf_get_val_str, GGUF_TYPE_STRING, false, kv(LLM_KV_GENERAL_NAME));

    // get hparams kv
    GGUF_GET_KEY(ctx, hparams.n_vocab,        gguf_get_arr_n,   GGUF_TYPE_ARRAY,  true, kv(LLM_KV_TOKENIZER_LIST));
    GGUF_GET_KEY(ctx, hparams.n_ctx_train,    gguf_get_val_u32, GGUF_TYPE_UINT32, true, kv(LLM_KV_CONTEXT_LENGTH));
    GGUF_GET_KEY(ctx, hparams.n_embd,         gguf_get_val_u32, GGUF_TYPE_UINT32, true, kv(LLM_KV_EMBEDDING_LENGTH));
    GGUF_GET_KEY(ctx, hparams.n_ff,           gguf_get_val_u32, GGUF_TYPE_UINT32, true, kv(LLM_KV_FEED_FORWARD_LENGTH));
    GGUF_GET_KEY(ctx, hparams.n_head,         gguf_get_val_u32, GGUF_TYPE_UINT32, true, kv(LLM_KV_ATTENTION_HEAD_COUNT));
    GGUF_GET_KEY(ctx, hparams.n_layer,        gguf_get_val_u32, GGUF_TYPE_UINT32, true, kv(LLM_KV_BLOCK_COUNT));

    // n_head_kv is optional, default to n_head
    hparams.n_head_kv = hparams.n_head;
    GGUF_GET_KEY(ctx, hparams.n_head_kv, gguf_get_val_u32, GGUF_TYPE_UINT32, false, kv(LLM_KV_ATTENTION_HEAD_COUNT_KV));

    // rope_freq_base (optional)
    hparams.rope_freq_base_train = 10000.0f;
    GGUF_GET_KEY(ctx, hparams.rope_freq_base_train, gguf_get_val_f32, GGUF_TYPE_FLOAT32, false, kv(LLM_KV_ROPE_FREQ_BASE));

    // rope_freq_scale (inverse of the kv) is optional
    float ropescale = 1.0f;
    GGUF_GET_KEY(ctx, ropescale, gguf_get_val_f32, GGUF_TYPE_FLOAT32, false, kv(LLM_KV_ROPE_SCALE_LINEAR));
    hparams.rope_freq_scale_train = 1.0f/ropescale;

    // sanity check for n_rot (optional)
    {
        hparams.n_rot = hparams.n_embd / hparams.n_head;

        GGUF_GET_KEY(ctx, hparams.n_rot, gguf_get_val_u32, GGUF_TYPE_UINT32, false, kv(LLM_KV_ROPE_DIMENSION_COUNT));

        if (model.arch == LLM_ARCH_LLAMA || model.arch == LLM_ARCH_FALCON) {
            if (hparams.n_rot != hparams.n_embd / hparams.n_head) {
                throw std::runtime_error(format("invalid n_rot: %u, expected %u", hparams.n_rot, hparams.n_embd / hparams.n_head));
            }
        }
        // gpt-neox n_rot = rotary_pct * (n_embd / n_head)
        // gpt-j n_rot = rotary_dim
    }

    // arch-specific KVs
    switch (model.arch) {
        case LLM_ARCH_LLAMA:
            {
                GGUF_GET_KEY(ctx, hparams.f_norm_rms_eps, gguf_get_val_f32, GGUF_TYPE_FLOAT32, true, kv(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS));

                switch (hparams.n_layer) {
                    case 26: model.type = e_model::MODEL_3B; break;
                    case 32: model.type = e_model::MODEL_7B; break;
                    case 40: model.type = e_model::MODEL_13B; break;
                    case 48: model.type = e_model::MODEL_34B; break;
                    case 60: model.type = e_model::MODEL_30B; break;
                    case 80: model.type = hparams.n_head == hparams.n_head_kv ? e_model::MODEL_65B : e_model::MODEL_70B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_FALCON:
            {
                GGUF_GET_KEY(ctx, hparams.f_norm_eps, gguf_get_val_f32, GGUF_TYPE_FLOAT32, true, kv(LLM_KV_ATTENTION_LAYERNORM_EPS));

                switch (hparams.n_layer) {
                    case 32: model.type = e_model::MODEL_7B; break;
                    case 60: model.type = e_model::MODEL_40B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_BAICHUAN:
            {
                GGUF_GET_KEY(ctx, hparams.f_norm_rms_eps, gguf_get_val_f32, GGUF_TYPE_FLOAT32, true, kv(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS));
                switch (hparams.n_layer) {
                    case 32: model.type = e_model::MODEL_7B; break;
                    case 40: model.type = e_model::MODEL_13B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_STARCODER:
            {
                GGUF_GET_KEY(ctx, hparams.f_norm_eps, gguf_get_val_f32, GGUF_TYPE_FLOAT32, true, kv(LLM_KV_ATTENTION_LAYERNORM_EPS));
                switch (hparams.n_layer) {
                    case 24: model.type = e_model::MODEL_1B; break;
                    case 36: model.type = e_model::MODEL_3B; break;
                    case 42: model.type = e_model::MODEL_7B; break;
                    case 40: model.type = e_model::MODEL_15B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_PERSIMMON:
            {
                GGUF_GET_KEY(ctx, hparams.f_norm_eps, gguf_get_val_f32, GGUF_TYPE_FLOAT32, true, kv(LLM_KV_ATTENTION_LAYERNORM_EPS));
                switch (hparams.n_layer) {
                    case 36: model.type = e_model::MODEL_8B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_REFACT:
            {
                GGUF_GET_KEY(ctx, hparams.f_norm_rms_eps, gguf_get_val_f32, GGUF_TYPE_FLOAT32, true, kv(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS));
                switch (hparams.n_layer) {
                    case 32: model.type = e_model::MODEL_1B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_BLOOM:
            {
                GGUF_GET_KEY(ctx, hparams.f_norm_eps, gguf_get_val_f32, GGUF_TYPE_FLOAT32, true, kv(LLM_KV_ATTENTION_LAYERNORM_EPS));

                switch (hparams.n_layer) {
                    case 24: model.type = e_model::MODEL_1B; break;
                    case 30:
                        switch (hparams.n_embd) {
                            case 2560: model.type = e_model::MODEL_3B; break;
                            case 4096: model.type = e_model::MODEL_7B; break;
                        } break;
                }
            } break;
        case LLM_ARCH_MPT:
            {
                hparams.f_clamp_kqv = 0.0f;

                GGUF_GET_KEY(ctx, hparams.f_norm_eps, gguf_get_val_f32, GGUF_TYPE_FLOAT32, true, kv(LLM_KV_ATTENTION_LAYERNORM_EPS));
                GGUF_GET_KEY(ctx, hparams.f_clamp_kqv, gguf_get_val_f32, GGUF_TYPE_FLOAT32, false, kv(LLM_KV_ATTENTION_CLAMP_KQV));
                GGUF_GET_KEY(ctx, hparams.f_max_alibi_bias, gguf_get_val_f32, GGUF_TYPE_FLOAT32, true, kv(LLM_KV_ATTENTION_MAX_ALIBI_BIAS));

                switch (hparams.n_layer) {
                    case 32: model.type = e_model::MODEL_7B; break;
                    case 48: model.type = e_model::MODEL_30B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        default: (void)0;
    }

    model.ftype = ml.ftype;
}

// TODO: This should probably be in llama.h
static std::vector<llama_vocab::id> llama_tokenize_internal(const llama_vocab & vocab, std::string raw_text, bool bos, bool special = false);
static llama_token llama_byte_to_token(const llama_vocab & vocab, uint8_t ch);

static void llm_load_vocab(
        llama_model_loader & ml,
        llama_model & model) {
    auto & vocab = model.vocab;

    struct gguf_context * ctx = ml.ctx_gguf;

    const auto kv = LLM_KV(model.arch);

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

    // determine vocab type
    {
        std::string tokenizer_name;

        GGUF_GET_KEY(ctx, tokenizer_name, gguf_get_val_str, GGUF_TYPE_STRING, true, kv(LLM_KV_TOKENIZER_MODEL));

        if (tokenizer_name == "llama") {
            vocab.type = LLAMA_VOCAB_TYPE_SPM;

            // default special tokens
            vocab.special_bos_id = 1;
            vocab.special_eos_id = 2;
            vocab.special_unk_id = 0;
            vocab.special_sep_id = -1;
            vocab.special_pad_id = -1;
        } else if (tokenizer_name == "gpt2") {
            vocab.type = LLAMA_VOCAB_TYPE_BPE;

            // read bpe merges and populate bpe ranks
            const int merges_keyidx = gguf_find_key(ctx, kv(LLM_KV_TOKENIZER_MERGES).c_str());
            if (merges_keyidx == -1) {
                throw std::runtime_error("cannot find tokenizer merges in model file\n");
            }

            const int n_merges = gguf_get_arr_n(ctx, merges_keyidx);

            for (int i = 0; i < n_merges; i++) {
                const std::string word = gguf_get_arr_str(ctx, merges_keyidx, i);
                GGML_ASSERT(codepoints_from_utf8(word).size() > 0);

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
            vocab.special_bos_id = 11;
            vocab.special_eos_id = 11;
            vocab.special_unk_id = -1;
            vocab.special_sep_id = -1;
            vocab.special_pad_id = -1;
        } else {
            LLAMA_LOG_WARN("%s: unknown tokenizer: '%s'", __func__, tokenizer_name.c_str());
            LLAMA_LOG_WARN("%s: using default tokenizer: 'llama'", __func__);

            vocab.type = LLAMA_VOCAB_TYPE_SPM;
        }
    }

    const uint32_t n_vocab = gguf_get_arr_n(ctx, token_idx);

    vocab.id_to_token.resize(n_vocab);

    for (uint32_t i = 0; i < n_vocab; i++) {
        std::string word = gguf_get_arr_str(ctx, token_idx, i);
        GGML_ASSERT(codepoints_from_utf8(word).size() > 0);

        vocab.token_to_id[word] = i;

        auto & token_data = vocab.id_to_token[i];
        token_data.text  = std::move(word);
        token_data.score = scores ? scores[i] : 0.0f;
        token_data.type  = toktypes ? (llama_token_type) toktypes[i] : LLAMA_TOKEN_TYPE_NORMAL;
    }
    GGML_ASSERT(vocab.id_to_token.size() == vocab.token_to_id.size());

    // determine the newline token: LLaMA "<0x0A>" == 10 == '\n', Falcon 193 == '\n'
    if (vocab.type == LLAMA_VOCAB_TYPE_SPM) {
        vocab.linefeed_id = llama_byte_to_token(vocab, '\n');
    } else {
        vocab.linefeed_id = llama_tokenize_internal(vocab, "\u010A", false)[0];
    }

    // special tokens
    GGUF_GET_KEY(ctx, vocab.special_bos_id, gguf_get_val_u32, GGUF_TYPE_UINT32, false, kv(LLM_KV_TOKENIZER_BOS_ID));
    GGUF_GET_KEY(ctx, vocab.special_eos_id, gguf_get_val_u32, GGUF_TYPE_UINT32, false, kv(LLM_KV_TOKENIZER_EOS_ID));
    GGUF_GET_KEY(ctx, vocab.special_unk_id, gguf_get_val_u32, GGUF_TYPE_UINT32, false, kv(LLM_KV_TOKENIZER_UNK_ID));
    GGUF_GET_KEY(ctx, vocab.special_sep_id, gguf_get_val_u32, GGUF_TYPE_UINT32, false, kv(LLM_KV_TOKENIZER_SEP_ID));
    GGUF_GET_KEY(ctx, vocab.special_pad_id, gguf_get_val_u32, GGUF_TYPE_UINT32, false, kv(LLM_KV_TOKENIZER_PAD_ID));

    // build special tokens cache
    {
        // TODO: It is unclear (to me) at this point, whether special tokes are guaranteed to be of a deterministic type,
        //  and will always be correctly labeled in 'added_tokens.json' etc.
        // The assumption is, since special tokens aren't meant to be exposed to end user, they are designed
        //  to be unmatchable by the tokenizer, therefore tokens from the vocab, which are unmatchable by the tokenizer
        //  are special tokens.
        // From testing, this appears to corelate 1:1 with special tokens.
        //

        // Counting special tokens and verifying in only one direction
        //  is sufficient to detect difference in those two sets.
        //
        uint32_t special_tokens_count_by_type = 0;
        uint32_t special_tokens_count_from_verification = 0;

        bool special_tokens_definition_mismatch = false;

        for (const auto & t : vocab.token_to_id) {
            const auto & token = t.first;
            const auto & id    = t.second;

            // Count all non-normal tokens in the vocab while iterating
            if (vocab.id_to_token[id].type != LLAMA_TOKEN_TYPE_NORMAL) {
                special_tokens_count_by_type++;
            }

            // Skip single character tokens
            if (token.length() > 1) {
                bool is_tokenizable = false;

                // Split token string representation in two, in all possible ways
                //  and check if both halves can be matched to a valid token
                for (unsigned i = 1; i < token.length();) {
                    const auto left  = token.substr(0, i);
                    const auto right = token.substr(i);

                    // check if we didnt partition in the middle of a utf sequence
                    auto utf = utf8_len(left.at(left.length() - 1));

                    if (utf == 1) {
                        if (vocab.token_to_id.find(left)  != vocab.token_to_id.end() &&
                            vocab.token_to_id.find(right) != vocab.token_to_id.end() ) {
                            is_tokenizable = true;
                            break;
                        }
                        i++;
                    } else {
                        // skip over the rest of multibyte utf sequence
                        i += utf - 1;
                    }
                }

                if (!is_tokenizable) {
                    // Some tokens are multibyte, but they are utf sequences with equivalent text length of 1
                    //  it's faster to re-filter them here, since there are way less candidates now

                    // Calculate a total "utf" length of a token string representation
                    size_t utf8_str_len = 0;
                    for (unsigned i = 0; i < token.length();) {
                        utf8_str_len++;
                        i += utf8_len(token.at(i));
                    }

                    // And skip the ones which are one character
                    if (utf8_str_len > 1) {
                        // At this point what we have left are special tokens only
                        vocab.special_tokens_cache[token] = id;

                        // Count manually found special tokens
                        special_tokens_count_from_verification++;

                        // If this manually found special token is not marked as such, flag a mismatch
                        if (vocab.id_to_token[id].type == LLAMA_TOKEN_TYPE_NORMAL) {
                            special_tokens_definition_mismatch = true;
                        }
                    }
                }
            }
        }

        if (special_tokens_definition_mismatch || special_tokens_count_from_verification != special_tokens_count_by_type) {
            LLAMA_LOG_WARN("%s: mismatch in special tokens definition ( %u/%zu vs %u/%zu ).\n",
                __func__,
                special_tokens_count_from_verification, vocab.id_to_token.size(),
                special_tokens_count_by_type, vocab.id_to_token.size()
            );
        } else {
            LLAMA_LOG_INFO("%s: special tokens definition check successful ( %u/%zu ).\n",
                __func__,
                special_tokens_count_from_verification, vocab.id_to_token.size()
            );
        }
    }
}

static void llm_load_print_meta(llama_model_loader & ml, llama_model & model) {
    const auto & hparams = model.hparams;
    const auto & vocab   = model.vocab;

    // hparams
    LLAMA_LOG_INFO("%s: format           = %s\n",     __func__, llama_file_version_name(ml.fver));
    LLAMA_LOG_INFO("%s: arch             = %s\n",     __func__, LLM_ARCH_NAMES.at(model.arch).c_str());
    LLAMA_LOG_INFO("%s: vocab type       = %s\n",     __func__, vocab.type == LLAMA_VOCAB_TYPE_SPM ? "SPM" : "BPE"); // TODO: fix
    LLAMA_LOG_INFO("%s: n_vocab          = %u\n",     __func__, hparams.n_vocab);
    LLAMA_LOG_INFO("%s: n_merges         = %u\n",     __func__, (int) vocab.bpe_ranks.size());
    LLAMA_LOG_INFO("%s: n_ctx_train      = %u\n",     __func__, hparams.n_ctx_train);
    LLAMA_LOG_INFO("%s: n_embd           = %u\n",     __func__, hparams.n_embd);
    LLAMA_LOG_INFO("%s: n_head           = %u\n",     __func__, hparams.n_head);
    LLAMA_LOG_INFO("%s: n_head_kv        = %u\n",     __func__, hparams.n_head_kv);
    LLAMA_LOG_INFO("%s: n_layer          = %u\n",     __func__, hparams.n_layer);
    LLAMA_LOG_INFO("%s: n_rot            = %u\n",     __func__, hparams.n_rot); // a.k.a. n_embd_head, n_head_dim
    LLAMA_LOG_INFO("%s: n_gqa            = %u\n",     __func__, hparams.n_gqa());
    LLAMA_LOG_INFO("%s: f_norm_eps       = %.1e\n",   __func__, hparams.f_norm_eps);
    LLAMA_LOG_INFO("%s: f_norm_rms_eps   = %.1e\n",   __func__, hparams.f_norm_rms_eps);
    LLAMA_LOG_INFO("%s: f_clamp_kqv      = %.1e\n",   __func__, hparams.f_clamp_kqv);
    LLAMA_LOG_INFO("%s: f_max_alibi_bias = %.1e\n",   __func__, hparams.f_max_alibi_bias);
    LLAMA_LOG_INFO("%s: n_ff             = %u\n",     __func__, hparams.n_ff);
    LLAMA_LOG_INFO("%s: freq_base_train  = %.1f\n",   __func__, hparams.rope_freq_base_train);
    LLAMA_LOG_INFO("%s: freq_scale_train = %g\n",     __func__, hparams.rope_freq_scale_train);
    LLAMA_LOG_INFO("%s: model type       = %s\n",     __func__, llama_model_type_name(model.type));
    LLAMA_LOG_INFO("%s: model ftype      = %s\n",     __func__, llama_model_ftype_name(model.ftype).c_str());
    LLAMA_LOG_INFO("%s: model params     = %.2f B\n", __func__, ml.n_elements*1e-9);
    if (ml.n_bytes < GB) {
        LLAMA_LOG_INFO("%s: model size       = %.2f MiB (%.2f BPW) \n", __func__, ml.n_bytes/1024.0/1024.0, ml.n_bytes*8.0/ml.n_elements);
    } else {
        LLAMA_LOG_INFO("%s: model size       = %.2f GiB (%.2f BPW) \n", __func__, ml.n_bytes/1024.0/1024.0/1024.0, ml.n_bytes*8.0/ml.n_elements);
    }

    // general kv
    LLAMA_LOG_INFO("%s: general.name   = %s\n",    __func__, model.name.c_str());

    // special tokens
    if (vocab.special_bos_id != -1) { LLAMA_LOG_INFO( "%s: BOS token = %d '%s'\n", __func__, vocab.special_bos_id, vocab.id_to_token[vocab.special_bos_id].text.c_str() ); }
    if (vocab.special_eos_id != -1) { LLAMA_LOG_INFO( "%s: EOS token = %d '%s'\n", __func__, vocab.special_eos_id, vocab.id_to_token[vocab.special_eos_id].text.c_str() ); }
    if (vocab.special_unk_id != -1) { LLAMA_LOG_INFO( "%s: UNK token = %d '%s'\n", __func__, vocab.special_unk_id, vocab.id_to_token[vocab.special_unk_id].text.c_str() ); }
    if (vocab.special_sep_id != -1) { LLAMA_LOG_INFO( "%s: SEP token = %d '%s'\n", __func__, vocab.special_sep_id, vocab.id_to_token[vocab.special_sep_id].text.c_str() ); }
    if (vocab.special_pad_id != -1) { LLAMA_LOG_INFO( "%s: PAD token = %d '%s'\n", __func__, vocab.special_pad_id, vocab.id_to_token[vocab.special_pad_id].text.c_str() ); }
    if (vocab.linefeed_id    != -1) { LLAMA_LOG_INFO( "%s: LF token  = %d '%s'\n", __func__, vocab.linefeed_id,    vocab.id_to_token[vocab.linefeed_id].text.c_str() );    }
}

static void llm_load_tensors(
        llama_model_loader & ml,
        llama_model & model,
        int n_gpu_layers,
        int main_gpu,
        const float * tensor_split,
        bool use_mlock,
        llama_progress_callback progress_callback,
        void * progress_callback_user_data) {
    model.t_start_us = ggml_time_us();

    auto & ctx     = model.ctx;
    auto & hparams = model.hparams;

    model.n_gpu_layers = n_gpu_layers;

    size_t ctx_size;
    size_t mmapped_size;

    ml.calc_sizes(ctx_size, mmapped_size);

    LLAMA_LOG_INFO("%s: ggml ctx size = %7.2f MB\n", __func__, ctx_size/1024.0/1024.0);

    // create the ggml context
    {
        model.buf.resize(ctx_size);
        if (use_mlock) {
            model.mlock_buf.init   (model.buf.data);
            model.mlock_buf.grow_to(model.buf.size);
        }

        struct ggml_init_params params = {
            /*.mem_size   =*/ model.buf.size,
            /*.mem_buffer =*/ model.buf.data,
            /*.no_alloc   =*/ ml.use_mmap,
        };

        model.ctx = ggml_init(params);
        if (!model.ctx) {
            throw std::runtime_error(format("ggml_init() failed"));
        }
    }

    (void) main_gpu;
#ifdef GGML_USE_CUBLAS
    LLAMA_LOG_INFO("%s: using " GGML_CUDA_NAME " for GPU acceleration\n", __func__);
    ggml_cuda_set_main_device(main_gpu);
#define LLAMA_BACKEND_OFFLOAD       GGML_BACKEND_GPU
#define LLAMA_BACKEND_OFFLOAD_SPLIT GGML_BACKEND_GPU_SPLIT
#elif defined(GGML_USE_CLBLAST)
    LLAMA_LOG_INFO("%s: using OpenCL for GPU acceleration\n", __func__);
#define LLAMA_BACKEND_OFFLOAD       GGML_BACKEND_GPU
#define LLAMA_BACKEND_OFFLOAD_SPLIT GGML_BACKEND_GPU
#else
#define LLAMA_BACKEND_OFFLOAD       GGML_BACKEND_CPU
#define LLAMA_BACKEND_OFFLOAD_SPLIT GGML_BACKEND_CPU
#endif

    // prepare memory for the weights
    size_t vram_weights = 0;
    {
        const int64_t n_embd     = hparams.n_embd;
        const int64_t n_embd_gqa = hparams.n_embd_gqa();
        const int64_t n_layer    = hparams.n_layer;
        const int64_t n_vocab    = hparams.n_vocab;

        const auto tn = LLM_TN(model.arch);
        switch (model.arch) {
            case LLM_ARCH_LLAMA:
            case LLM_ARCH_REFACT:
                {
                    model.tok_embeddings = ml.create_tensor(ctx, tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, GGML_BACKEND_CPU);

                    // output
                    {
                        ggml_backend_type backend_norm;
                        ggml_backend_type backend_output;

                        if (n_gpu_layers > int(n_layer)) {
                            // norm is not performance relevant on its own but keeping it in VRAM reduces data copying
                            // on Windows however this is detrimental unless everything is on the GPU
#ifndef _WIN32
                            backend_norm = LLAMA_BACKEND_OFFLOAD;
#else
                            backend_norm = n_gpu_layers <= (int) n_layer + 2 ? GGML_BACKEND_CPU : LLAMA_BACKEND_OFFLOAD;
#endif // _WIN32

                            backend_output = LLAMA_BACKEND_OFFLOAD_SPLIT;
                        } else {
                            backend_norm   = GGML_BACKEND_CPU;
                            backend_output = GGML_BACKEND_CPU;
                        }

                        model.output_norm = ml.create_tensor(ctx, tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd},          backend_norm);
                        model.output      = ml.create_tensor(ctx, tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, backend_output);

                        if (backend_norm == GGML_BACKEND_GPU) {
                            vram_weights += ggml_nbytes(model.output_norm);
                        }
                        if (backend_output == GGML_BACKEND_GPU_SPLIT) {
                            vram_weights += ggml_nbytes(model.output);
                        }
                    }

                    const uint32_t n_ff = hparams.n_ff;

                    const int i_gpu_start = n_layer - n_gpu_layers;

                    model.layers.resize(n_layer);

                    for (uint32_t i = 0; i < n_layer; ++i) {
                        const ggml_backend_type backend = int(i) < i_gpu_start ? GGML_BACKEND_CPU : LLAMA_BACKEND_OFFLOAD; // NOLINT
                        const ggml_backend_type backend_split = int(i) < i_gpu_start ? GGML_BACKEND_CPU : LLAMA_BACKEND_OFFLOAD_SPLIT; // NOLINT

                        auto & layer = model.layers[i];

                        layer.attn_norm = ml.create_tensor(ctx, tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, backend);

                        layer.wq = ml.create_tensor(ctx, tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_embd},     backend_split);
                        layer.wk = ml.create_tensor(ctx, tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_embd_gqa}, backend_split);
                        layer.wv = ml.create_tensor(ctx, tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_gqa}, backend_split);
                        layer.wo = ml.create_tensor(ctx, tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd},     backend_split);

                        layer.ffn_norm = ml.create_tensor(ctx, tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, backend);

                        layer.w1 = ml.create_tensor(ctx, tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff}, backend_split);
                        layer.w2 = ml.create_tensor(ctx, tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd}, backend_split);
                        layer.w3 = ml.create_tensor(ctx, tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff}, backend_split);

                        if (backend == GGML_BACKEND_GPU) {
                            vram_weights +=
                                ggml_nbytes(layer.attn_norm) + ggml_nbytes(layer.wq) + ggml_nbytes(layer.wk)       +
                                ggml_nbytes(layer.wv)        + ggml_nbytes(layer.wo) + ggml_nbytes(layer.ffn_norm) +
                                ggml_nbytes(layer.w1)        + ggml_nbytes(layer.w2) + ggml_nbytes(layer.w3);
                        }
                    }
                } break;
            case LLM_ARCH_BAICHUAN:
                {
                    model.tok_embeddings = ml.create_tensor(ctx, tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, GGML_BACKEND_CPU);
                    {
                        ggml_backend_type backend_norm;
                        ggml_backend_type backend_output;

                        if (n_gpu_layers > int(n_layer)) {
                            // norm is not performance relevant on its own but keeping it in VRAM reduces data copying
                            // on Windows however this is detrimental unless everything is on the GPU
#ifndef _WIN32
                            backend_norm = LLAMA_BACKEND_OFFLOAD;
#else
                            backend_norm = n_gpu_layers <= (int) n_layer + 2 ? GGML_BACKEND_CPU : LLAMA_BACKEND_OFFLOAD;
#endif // _WIN32

                            backend_output = LLAMA_BACKEND_OFFLOAD_SPLIT;
                        } else {
                            backend_norm   = GGML_BACKEND_CPU;
                            backend_output = GGML_BACKEND_CPU;
                        }

                        model.output_norm = ml.create_tensor(ctx, tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd},          backend_norm);
                        model.output      = ml.create_tensor(ctx, tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, backend_output);

                        if (backend_norm == GGML_BACKEND_GPU) {
                            vram_weights += ggml_nbytes(model.output_norm);
                        }
                        if (backend_output == GGML_BACKEND_GPU_SPLIT) {
                            vram_weights += ggml_nbytes(model.output);
                        }
                    }

                    const uint32_t n_ff = hparams.n_ff;

                    const int i_gpu_start = n_layer - n_gpu_layers;

                    model.layers.resize(n_layer);

                    for (uint32_t i = 0; i < n_layer; ++i) {
                        const ggml_backend_type backend = int(i) < i_gpu_start ? GGML_BACKEND_CPU : LLAMA_BACKEND_OFFLOAD; // NOLINT
                        const ggml_backend_type backend_split = int(i) < i_gpu_start ? GGML_BACKEND_CPU : LLAMA_BACKEND_OFFLOAD_SPLIT; // NOLINT

                        auto & layer = model.layers[i];

                        layer.attn_norm = ml.create_tensor(ctx, tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, backend);

                        layer.wq = ml.create_tensor(ctx, tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_embd},     backend_split);
                        layer.wk = ml.create_tensor(ctx, tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_embd_gqa}, backend_split);
                        layer.wv = ml.create_tensor(ctx, tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_gqa}, backend_split);
                        layer.wo = ml.create_tensor(ctx, tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd},     backend_split);

                        layer.ffn_norm = ml.create_tensor(ctx, tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, backend);

                        layer.w1 = ml.create_tensor(ctx, tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff}, backend_split);
                        layer.w2 = ml.create_tensor(ctx, tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd}, backend_split);
                        layer.w3 = ml.create_tensor(ctx, tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff}, backend_split);

                        if (backend == GGML_BACKEND_GPU) {
                            vram_weights +=
                                ggml_nbytes(layer.attn_norm) + ggml_nbytes(layer.wq) + ggml_nbytes(layer.wk)       +
                                ggml_nbytes(layer.wv)        + ggml_nbytes(layer.wo) + ggml_nbytes(layer.ffn_norm) +
                                ggml_nbytes(layer.w1)        + ggml_nbytes(layer.w2) + ggml_nbytes(layer.w3);
                        }
                    }
                } break;
            case LLM_ARCH_FALCON:
                {
                    // TODO: CPU-only for now

                    model.tok_embeddings = ml.create_tensor(ctx, tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, GGML_BACKEND_CPU);

                    // output
                    {
                        ggml_backend_type backend_norm;
                        ggml_backend_type backend_output;

                        if (n_gpu_layers > int(n_layer)) {
                            // norm is not performance relevant on its own but keeping it in VRAM reduces data copying
                            // on Windows however this is detrimental unless everything is on the GPU
#ifndef _WIN32
                            backend_norm = LLAMA_BACKEND_OFFLOAD;
#else
                            backend_norm = n_gpu_layers <= (int) n_layer + 2 ? GGML_BACKEND_CPU : LLAMA_BACKEND_OFFLOAD;
#endif // _WIN32

                            backend_output = LLAMA_BACKEND_OFFLOAD_SPLIT;
                        } else {
                            backend_norm   = GGML_BACKEND_CPU;
                            backend_output = GGML_BACKEND_CPU;
                        }

                        model.output_norm   = ml.create_tensor(ctx, tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd},          backend_norm);
                        model.output_norm_b = ml.create_tensor(ctx, tn(LLM_TENSOR_OUTPUT_NORM, "bias"),   {n_embd},          backend_norm);
                        model.output        = ml.create_tensor(ctx, tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, backend_output);

                        if (backend_norm == GGML_BACKEND_GPU) {
                            vram_weights += ggml_nbytes(model.output_norm);
                            vram_weights += ggml_nbytes(model.output_norm_b);
                        }
                        if (backend_output == GGML_BACKEND_GPU_SPLIT) {
                            vram_weights += ggml_nbytes(model.output);
                        }
                    }

                    const uint32_t n_ff = hparams.n_ff;

                    const int i_gpu_start = n_layer - n_gpu_layers;

                    model.layers.resize(n_layer);

                    for (uint32_t i = 0; i < n_layer; ++i) {
                        const ggml_backend_type backend       = int(i) < i_gpu_start ? GGML_BACKEND_CPU : LLAMA_BACKEND_OFFLOAD; // NOLINT
                        const ggml_backend_type backend_split = int(i) < i_gpu_start ? GGML_BACKEND_CPU : LLAMA_BACKEND_OFFLOAD_SPLIT; // NOLINT

                        auto & layer = model.layers[i];

                        layer.attn_norm   = ml.create_tensor(ctx, tn(LLM_TENSOR_ATTN_NORM,   "weight", i), {n_embd}, backend);
                        layer.attn_norm_b = ml.create_tensor(ctx, tn(LLM_TENSOR_ATTN_NORM,   "bias", i),   {n_embd}, backend);

                        if (gguf_find_tensor(ml.ctx_gguf, tn(LLM_TENSOR_ATTN_NORM_2, "weight", i).c_str()) >= 0) {
                            layer.attn_norm_2   = ml.create_tensor(ctx, tn(LLM_TENSOR_ATTN_NORM_2, "weight", i), {n_embd}, backend);
                            layer.attn_norm_2_b = ml.create_tensor(ctx, tn(LLM_TENSOR_ATTN_NORM_2, "bias", i),   {n_embd}, backend);

                            if (backend == GGML_BACKEND_GPU) {
                                vram_weights += ggml_nbytes(layer.attn_norm_2);
                                vram_weights += ggml_nbytes(layer.attn_norm_2_b);
                            }
                        }

                        layer.wqkv = ml.create_tensor(ctx, tn(LLM_TENSOR_ATTN_QKV, "weight", i), {n_embd, n_embd + 2*n_embd_gqa}, backend_split);
                        layer.wo   = ml.create_tensor(ctx, tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd},                backend_split);

                        layer.w2 = ml.create_tensor(ctx, tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd}, backend_split);
                        layer.w3 = ml.create_tensor(ctx, tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff}, backend_split);

                        if (backend == GGML_BACKEND_GPU) {
                            vram_weights +=
                                ggml_nbytes(layer.attn_norm) + ggml_nbytes(layer.attn_norm_b) +
                                ggml_nbytes(layer.wqkv)      + ggml_nbytes(layer.wo)          +
                                ggml_nbytes(layer.w2)        + ggml_nbytes(layer.w3);
                        }
                    }
                } break;
            case LLM_ARCH_STARCODER:
                {
                    model.tok_embeddings = ml.create_tensor(ctx, tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, GGML_BACKEND_CPU);
                    model.pos_embeddings = ml.create_tensor(ctx, tn(LLM_TENSOR_POS_EMBD, "weight"), {n_embd, hparams.n_ctx_train}, GGML_BACKEND_CPU);

                    // output
                    {
                        ggml_backend_type backend_norm;
                        ggml_backend_type backend_output;

                        if (n_gpu_layers > int(n_layer)) {
                            // norm is not performance relevant on its own but keeping it in VRAM reduces data copying
                            // on Windows however this is detrimental unless everything is on the GPU
#ifndef _WIN32
                            backend_norm = LLAMA_BACKEND_OFFLOAD;
#else
                            backend_norm = n_gpu_layers <= (int) n_layer + 2 ? GGML_BACKEND_CPU : LLAMA_BACKEND_OFFLOAD;
#endif // _WIN32

                            backend_output = LLAMA_BACKEND_OFFLOAD_SPLIT;
                        } else {
                            backend_norm   = GGML_BACKEND_CPU;
                            backend_output = GGML_BACKEND_CPU;
                        }

                        model.output_norm   = ml.create_tensor(ctx, tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd},          backend_norm);
                        model.output_norm_b = ml.create_tensor(ctx, tn(LLM_TENSOR_OUTPUT_NORM, "bias"),   {n_embd},          backend_norm);
                        model.output        = ml.create_tensor(ctx, tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, backend_output);

                        if (backend_norm == GGML_BACKEND_GPU) {
                            vram_weights += ggml_nbytes(model.output_norm);
                            vram_weights += ggml_nbytes(model.output_norm_b);
                        }
                        if (backend_output == GGML_BACKEND_GPU_SPLIT) {
                            vram_weights += ggml_nbytes(model.output);
                        }
                    }

                    const uint32_t n_ff = hparams.n_ff;

                    const int i_gpu_start = n_layer - n_gpu_layers;

                    model.layers.resize(n_layer);

                    for (uint32_t i = 0; i < n_layer; ++i) {
                        const ggml_backend_type backend       = int(i) < i_gpu_start ? GGML_BACKEND_CPU : LLAMA_BACKEND_OFFLOAD; // NOLINT
                        const ggml_backend_type backend_split = int(i) < i_gpu_start ? GGML_BACKEND_CPU : LLAMA_BACKEND_OFFLOAD_SPLIT; // NOLINT

                        auto & layer = model.layers[i];

                        layer.attn_norm   = ml.create_tensor(ctx, tn(LLM_TENSOR_ATTN_NORM,   "weight", i), {n_embd}, backend);
                        layer.attn_norm_b = ml.create_tensor(ctx, tn(LLM_TENSOR_ATTN_NORM,   "bias", i),   {n_embd}, backend);

                        layer.wqkv = ml.create_tensor(ctx, tn(LLM_TENSOR_ATTN_QKV, "weight", i), {n_embd, n_embd + 2*n_embd_gqa}, backend_split);
                        layer.bqkv = ml.create_tensor(ctx, tn(LLM_TENSOR_ATTN_QKV, "bias", i),   {n_embd + 2*n_embd_gqa},         backend_split);

                        layer.wo   = ml.create_tensor(ctx, tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd},   backend_split);
                        layer.bo   = ml.create_tensor(ctx, tn(LLM_TENSOR_ATTN_OUT, "bias", i),   {n_embd},           backend_split);

                        layer.ffn_norm   = ml.create_tensor(ctx, tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, backend);
                        layer.ffn_norm_b = ml.create_tensor(ctx, tn(LLM_TENSOR_FFN_NORM, "bias", i),   {n_embd}, backend);

                        layer.w2 = ml.create_tensor(ctx, tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff, n_embd}, backend_split);
                        layer.b2 = ml.create_tensor(ctx, tn(LLM_TENSOR_FFN_DOWN, "bias", i),   {n_embd},       backend_split);

                        layer.w3 = ml.create_tensor(ctx, tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff}, backend_split);
                        layer.b3 = ml.create_tensor(ctx, tn(LLM_TENSOR_FFN_UP,   "bias", i),   {n_ff},           backend_split);

                        if (backend == GGML_BACKEND_GPU) {
                            vram_weights +=
                                ggml_nbytes(layer.attn_norm) + ggml_nbytes(layer.attn_norm_b) +
                                ggml_nbytes(layer.wqkv)      + ggml_nbytes(layer.bqkv)        +
                                ggml_nbytes(layer.wo)        + ggml_nbytes(layer.bo)          +
                                ggml_nbytes(layer.ffn_norm)  + ggml_nbytes(layer.ffn_norm_b)  +
                                ggml_nbytes(layer.w2)        + ggml_nbytes(layer.b2)          +
                                ggml_nbytes(layer.w3)        + ggml_nbytes(layer.b3);
                        }
                    }
                } break;
            case LLM_ARCH_PERSIMMON:
                {
                    model.tok_embeddings = ml.create_tensor(ctx, tn(LLM_TENSOR_TOKEN_EMBD, "weight"),  {n_embd, n_vocab}, GGML_BACKEND_CPU);

                    {
                        ggml_backend_type backend_norm;
                        ggml_backend_type backend_output;

                        if (n_gpu_layers > int(n_layer)) {
                            // norm is not performance relevant on its own but keeping it in VRAM reduces data copying
                            // on Windows however this is detrimental unless everything is on the GPU
#ifndef _WIN32
                            backend_norm = LLAMA_BACKEND_OFFLOAD;
#else
                            backend_norm = n_gpu_layers <= (int) n_layer + 2 ? GGML_BACKEND_CPU : LLAMA_BACKEND_OFFLOAD;
#endif // _WIN32

                            backend_output = LLAMA_BACKEND_OFFLOAD_SPLIT;
                        } else {
                            backend_norm   = GGML_BACKEND_CPU;
                            backend_output = GGML_BACKEND_CPU;
                        }

                        model.output_norm    = ml.create_tensor(ctx, tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd},          backend_norm);
                        model.output_norm_b  = ml.create_tensor(ctx, tn(LLM_TENSOR_OUTPUT_NORM, "bias"),   {n_embd},          backend_norm);
                        model.output         = ml.create_tensor(ctx, tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, backend_output);

                        if (backend_norm == GGML_BACKEND_GPU) {
                            vram_weights += ggml_nbytes(model.output_norm);
                            vram_weights += ggml_nbytes(model.output_norm_b);
                        }
                        if (backend_output == GGML_BACKEND_GPU_SPLIT) {
                            vram_weights += ggml_nbytes(model.output);
                        }
                    }

                    const uint32_t n_ff = hparams.n_ff;
                    const int i_gpu_start = n_layer - n_gpu_layers;
                    model.layers.resize(n_layer);
                    for (uint32_t i = 0; i < n_layer; ++i) {
                        const ggml_backend_type backend = int(i) < i_gpu_start ? GGML_BACKEND_CPU : LLAMA_BACKEND_OFFLOAD;
                        const ggml_backend_type backend_split = int(i) < i_gpu_start ? GGML_BACKEND_CPU : LLAMA_BACKEND_OFFLOAD_SPLIT;
                        auto & layer = model.layers[i];
                        layer.attn_norm   = ml.create_tensor(ctx, tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, backend);
                        layer.attn_norm_b = ml.create_tensor(ctx, tn(LLM_TENSOR_ATTN_NORM, "bias", i),   {n_embd}, backend);
                        layer.wqkv = ml.create_tensor(ctx, tn(LLM_TENSOR_ATTN_QKV, "weight", i), {n_embd, n_embd + 2*n_embd_gqa}, backend_split);
                        layer.bqkv = ml.create_tensor(ctx, tn(LLM_TENSOR_ATTN_QKV, "bias", i),   {n_embd + 2*n_embd_gqa},         backend_split);
                        layer.wo   = ml.create_tensor(ctx, tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd},   backend_split);
                        layer.bo   = ml.create_tensor(ctx, tn(LLM_TENSOR_ATTN_OUT, "bias", i),   {n_embd},           backend_split);
                        layer.w2 = ml.create_tensor(ctx, tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff, n_embd}, backend_split);
                        layer.b2 = ml.create_tensor(ctx, tn(LLM_TENSOR_FFN_DOWN, "bias", i),   {n_embd},       backend_split);
                        layer.w3 = ml.create_tensor(ctx, tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff}, backend_split);
                        layer.b3 = ml.create_tensor(ctx, tn(LLM_TENSOR_FFN_UP,   "bias", i),   {n_ff},           backend_split);
                        layer.ffn_norm   = ml.create_tensor(ctx, tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, backend);
                        layer.ffn_norm_b = ml.create_tensor(ctx, tn(LLM_TENSOR_FFN_NORM, "bias", i),   {n_embd}, backend);
                        layer.attn_q_norm   = ml.create_tensor(ctx, tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), {64}, backend);
                        layer.attn_q_norm_b = ml.create_tensor(ctx, tn(LLM_TENSOR_ATTN_Q_NORM, "bias", i),   {64}, backend);
                        layer.attn_k_norm   = ml.create_tensor(ctx, tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), {64}, backend);
                        layer.attn_k_norm_b = ml.create_tensor(ctx, tn(LLM_TENSOR_ATTN_K_NORM, "bias", i),   {64}, backend);
                    }
                } break;
            case LLM_ARCH_BLOOM:
                {
                    // TODO: CPU-only for now

                    model.tok_embeddings = ml.create_tensor(ctx, tn(LLM_TENSOR_TOKEN_EMBD,      "weight"), {n_embd, n_vocab}, GGML_BACKEND_CPU);
                    model.tok_norm       = ml.create_tensor(ctx, tn(LLM_TENSOR_TOKEN_EMBD_NORM, "weight"), {n_embd},          GGML_BACKEND_CPU);
                    model.tok_norm_b     = ml.create_tensor(ctx, tn(LLM_TENSOR_TOKEN_EMBD_NORM, "bias"),   {n_embd},          GGML_BACKEND_CPU);

                    // output
                    {
                        ggml_backend_type backend_norm;
                        ggml_backend_type backend_output;

                        if (n_gpu_layers > int(n_layer)) {
                            // norm is not performance relevant on its own but keeping it in VRAM reduces data copying
                            // on Windows however this is detrimental unless everything is on the GPU
#ifndef _WIN32
                            backend_norm = LLAMA_BACKEND_OFFLOAD;
#else
                            backend_norm = n_gpu_layers <= (int) n_layer + 2 ? GGML_BACKEND_CPU : LLAMA_BACKEND_OFFLOAD;
#endif // _WIN32

                            backend_output = LLAMA_BACKEND_OFFLOAD_SPLIT;
                        } else {
                            backend_norm   = GGML_BACKEND_CPU;
                            backend_output = GGML_BACKEND_CPU;
                        }

                        model.output_norm   = ml.create_tensor(ctx, tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd},          backend_norm);
                        model.output_norm_b = ml.create_tensor(ctx, tn(LLM_TENSOR_OUTPUT_NORM, "bias"),   {n_embd},          backend_norm);
                        model.output        = ml.create_tensor(ctx, tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, backend_output);

                        if (backend_norm == GGML_BACKEND_GPU) {
                            vram_weights += ggml_nbytes(model.output_norm);
                            vram_weights += ggml_nbytes(model.output_norm_b);
                        }
                        if (backend_output == GGML_BACKEND_GPU_SPLIT) {
                            vram_weights += ggml_nbytes(model.output);
                        }
                    }

                    const uint32_t n_ff = hparams.n_ff;

                    const int i_gpu_start = n_layer - n_gpu_layers;

                    model.layers.resize(n_layer);

                    for (uint32_t i = 0; i < n_layer; ++i) {
                        const ggml_backend_type backend       = int(i) < i_gpu_start ? GGML_BACKEND_CPU : LLAMA_BACKEND_OFFLOAD; // NOLINT
                        const ggml_backend_type backend_split = int(i) < i_gpu_start ? GGML_BACKEND_CPU : LLAMA_BACKEND_OFFLOAD_SPLIT; // NOLINT

                        auto & layer = model.layers[i];

                        layer.attn_norm   = ml.create_tensor(ctx, tn(LLM_TENSOR_ATTN_NORM,   "weight", i), {n_embd}, backend);
                        layer.attn_norm_b = ml.create_tensor(ctx, tn(LLM_TENSOR_ATTN_NORM,   "bias", i),   {n_embd}, backend);

                        layer.wqkv = ml.create_tensor(ctx, tn(LLM_TENSOR_ATTN_QKV, "weight", i), {n_embd, n_embd + 2*n_embd_gqa}, backend_split);
                        layer.bqkv = ml.create_tensor(ctx, tn(LLM_TENSOR_ATTN_QKV, "bias", i),   {n_embd + 2*n_embd_gqa},         backend_split);

                        layer.wo   = ml.create_tensor(ctx, tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd},                backend_split);
                        layer.bo   = ml.create_tensor(ctx, tn(LLM_TENSOR_ATTN_OUT, "bias", i),   {n_embd},                        backend_split);

                        layer.ffn_norm   = ml.create_tensor(ctx, tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, backend);
                        layer.ffn_norm_b = ml.create_tensor(ctx, tn(LLM_TENSOR_FFN_NORM, "bias", i),   {n_embd}, backend);

                        layer.w2 = ml.create_tensor(ctx, tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff, n_embd}, backend_split);
                        layer.b2 = ml.create_tensor(ctx, tn(LLM_TENSOR_FFN_DOWN, "bias", i),   {n_embd},       backend_split);

                        layer.w3 = ml.create_tensor(ctx, tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff}, backend_split);
                        layer.b3 = ml.create_tensor(ctx, tn(LLM_TENSOR_FFN_UP,   "bias", i),   {n_ff},           backend_split);

                        if (backend == GGML_BACKEND_GPU) {
                            vram_weights +=
                                ggml_nbytes(layer.attn_norm) + ggml_nbytes(layer.attn_norm_b) +
                                ggml_nbytes(layer.wqkv)      + ggml_nbytes(layer.bqkv)        +
                                ggml_nbytes(layer.wo)        + ggml_nbytes(layer.bo)          +
                                ggml_nbytes(layer.ffn_norm)  + ggml_nbytes(layer.ffn_norm_b)  +
                                ggml_nbytes(layer.w3)        + ggml_nbytes(layer.b3)          +
                                ggml_nbytes(layer.w2)        + ggml_nbytes(layer.b2);
                        }
                    }
                } break;
            case LLM_ARCH_MPT:
                {
                    model.tok_embeddings = ml.create_tensor(ctx, tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, GGML_BACKEND_CPU);

                    // output
                    {
                        ggml_backend_type backend_norm;
                        ggml_backend_type backend_output;

                        if (n_gpu_layers > int(n_layer)) {
                            // norm is not performance relevant on its own but keeping it in VRAM reduces data copying
                            // on Windows however this is detrimental unless everything is on the GPU
#ifndef _WIN32
                            backend_norm = LLAMA_BACKEND_OFFLOAD;
#else
                            backend_norm = n_gpu_layers <= (int) n_layer + 2 ? GGML_BACKEND_CPU : LLAMA_BACKEND_OFFLOAD;
#endif // _WIN32

                            backend_output = LLAMA_BACKEND_OFFLOAD_SPLIT;
                        } else {
                            backend_norm   = GGML_BACKEND_CPU;
                            backend_output = GGML_BACKEND_CPU;
                        }

                        model.output_norm   = ml.create_tensor(ctx, tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd},          backend_norm);
                        model.output        = ml.create_tensor(ctx, tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, backend_output);

                        if (backend_norm == GGML_BACKEND_GPU) {
                            vram_weights += ggml_nbytes(model.output_norm);
                        }
                        if (backend_output == GGML_BACKEND_GPU_SPLIT) {
                            vram_weights += ggml_nbytes(model.output);
                        }
                    }

                    const uint32_t n_ff = hparams.n_ff;

                    const int i_gpu_start = n_layer - n_gpu_layers;

                    model.layers.resize(n_layer);

                    for (uint32_t i = 0; i < n_layer; ++i) {
                        const ggml_backend_type backend = int(i) < i_gpu_start ? GGML_BACKEND_CPU : LLAMA_BACKEND_OFFLOAD; // NOLINT
                        const ggml_backend_type backend_split = int(i) < i_gpu_start ? GGML_BACKEND_CPU : LLAMA_BACKEND_OFFLOAD_SPLIT; // NOLINT

                        auto & layer = model.layers[i];

                        layer.attn_norm = ml.create_tensor(ctx, tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, backend);
                        layer.wqkv = ml.create_tensor(ctx, tn(LLM_TENSOR_ATTN_QKV, "weight", i), {n_embd, n_embd + 2*n_embd_gqa}, backend_split);
                        layer.wo   = ml.create_tensor(ctx, tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd},                backend_split);

                        layer.ffn_norm = ml.create_tensor(ctx, tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, backend);

                        layer.w2 = ml.create_tensor(ctx, tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd}, backend_split);
                        layer.w3 = ml.create_tensor(ctx, tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff}, backend_split);

                        if (backend == GGML_BACKEND_GPU) {
                            vram_weights +=
                                ggml_nbytes(layer.attn_norm) +
                                ggml_nbytes(layer.wqkv)      +
                                ggml_nbytes(layer.wo)        +
                                ggml_nbytes(layer.ffn_norm)  +
                                ggml_nbytes(layer.w2)        +
                                ggml_nbytes(layer.w3);
                        }
                    }
                } break;
            default:
                throw std::runtime_error("unknown architecture");
        }
    }

    ml.done_getting_tensors();

    // print memory requirements
    {
        // this is the total memory required to run the inference
        size_t mem_required =
            ctx_size +
            mmapped_size - vram_weights; // weights in VRAM not in memory

        LLAMA_LOG_INFO("%s: mem required  = %7.2f MB\n", __func__, mem_required / 1024.0 / 1024.0);

#if defined(GGML_USE_CUBLAS) || defined(GGML_USE_CLBLAST)
        const int n_gpu = std::min(n_gpu_layers, int(hparams.n_layer));

        LLAMA_LOG_INFO("%s: offloading %d repeating layers to GPU\n", __func__, n_gpu);
        if (n_gpu_layers > (int) hparams.n_layer) {
            LLAMA_LOG_INFO("%s: offloading non-repeating layers to GPU\n", __func__);
        }

#ifdef GGML_USE_CUBLAS
        const int max_backend_supported_layers = hparams.n_layer + 3;
        const int max_offloadable_layers = hparams.n_layer + 3;
#elif defined(GGML_USE_CLBLAST)
        const int max_backend_supported_layers = hparams.n_layer + 1;
        const int max_offloadable_layers = hparams.n_layer + 1;
#endif // GGML_USE_CUBLAS

        LLAMA_LOG_INFO("%s: offloaded %d/%d layers to GPU\n", __func__, std::min(n_gpu_layers, max_offloadable_layers), max_backend_supported_layers);
        LLAMA_LOG_INFO("%s: VRAM used: %.2f MB\n", __func__, vram_weights / 1024.0 / 1024.0);
#else
        (void) n_gpu_layers;
#endif // defined(GGML_USE_CUBLAS) || defined(GGML_USE_CLBLAST)
    }

    // populate `tensors_by_name`
    for (int i = 0; i < ml.n_tensors; ++i) {
        struct ggml_tensor * cur = ggml_get_tensor(ctx, ml.get_tensor_name(i));
        model.tensors_by_name.emplace_back(ggml_get_name(cur), cur);
    }

    (void) tensor_split;
#ifdef GGML_USE_CUBLAS
    {
        ggml_cuda_set_tensor_split(tensor_split);
    }
#endif

    ml.load_all_data(ctx, progress_callback, progress_callback_user_data, use_mlock ? &model.mlock_mmap : NULL);

    if (progress_callback) {
        progress_callback(1.0f, progress_callback_user_data);
    }

    model.mapping = std::move(ml.mapping);

    // loading time will be recalculate after the first eval, so
    // we take page faults deferred by mmap() into consideration
    model.t_load_us = ggml_time_us() - model.t_start_us;
}

static bool llama_model_load(
        const std::string & fname,
        llama_model & model,
        int n_gpu_layers,
        int main_gpu,
        const float * tensor_split,
        bool use_mmap,
        bool use_mlock,
        bool vocab_only,
        llama_progress_callback progress_callback,
        void *progress_callback_user_data) {
    try {
        llama_model_loader ml(fname, use_mmap);

        model.hparams.vocab_only = vocab_only;

        llm_load_arch   (ml, model);
        llm_load_hparams(ml, model);
        llm_load_vocab  (ml, model);

        llm_load_print_meta(ml, model);

        if (model.hparams.n_vocab != model.vocab.id_to_token.size()) {
            throw std::runtime_error("vocab size mismatch");
        }

        if (vocab_only) {
            LLAMA_LOG_INFO("%s: vocab only - skipping tensors\n", __func__);
            return true;
        }

        llm_load_tensors(
                ml, model, n_gpu_layers,
                main_gpu, tensor_split,
                use_mlock, progress_callback, progress_callback_user_data);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("error loading model: %s\n", err.what());
        return false;
    }

    return true;
}

static struct ggml_cgraph * llm_build_llama(
    llama_context & lctx,
    const llama_batch & batch) {
    const auto & model   = lctx.model;
    const auto & hparams = model.hparams;
    const auto & cparams = lctx.cparams;

    const auto & kv_self = lctx.kv_self;

    GGML_ASSERT(!!kv_self.ctx);

    const int64_t n_embd      = hparams.n_embd;
    const int64_t n_layer     = hparams.n_layer;
    const int64_t n_ctx       = cparams.n_ctx;
    const int64_t n_head      = hparams.n_head;
    const int64_t n_head_kv   = hparams.n_head_kv;
    const int64_t n_embd_head = hparams.n_embd_head();
    const int64_t n_embd_gqa  = hparams.n_embd_gqa();

    GGML_ASSERT(n_embd_head == hparams.n_rot);

    const float freq_base    = cparams.rope_freq_base;
    const float freq_scale   = cparams.rope_freq_scale;
    const float norm_rms_eps = hparams.f_norm_rms_eps;

    const int n_gpu_layers = model.n_gpu_layers;

    const int32_t n_tokens = batch.n_tokens;
    const int32_t n_kv     = ggml_allocr_is_measure(lctx.alloc) ? n_ctx            : kv_self.n;
    const int32_t kv_head  = ggml_allocr_is_measure(lctx.alloc) ? n_ctx - n_tokens : kv_self.head;

    const bool do_rope_shift = ggml_allocr_is_measure(lctx.alloc) || kv_self.has_shift;

    //printf("n_kv = %d\n", n_kv);

    auto & buf_compute = lctx.buf_compute;

    struct ggml_init_params params = {
        /*.mem_size   =*/ buf_compute.size,
        /*.mem_buffer =*/ buf_compute.data,
        /*.no_alloc   =*/ true,
    };

    struct ggml_context * ctx0 = ggml_init(params);

    ggml_cgraph * gf = ggml_new_graph(ctx0);

    struct ggml_tensor * cur;
    struct ggml_tensor * inpL;

    if (batch.token) {
        struct ggml_tensor * inp_tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);

        ggml_allocr_alloc(lctx.alloc, inp_tokens);
        if (!ggml_allocr_is_measure(lctx.alloc)) {
            memcpy(inp_tokens->data, batch.token, n_tokens*ggml_element_size(inp_tokens));
        }
        ggml_set_name(inp_tokens, "inp_tokens");

        inpL = ggml_get_rows(ctx0, model.tok_embeddings, inp_tokens);
    } else {
#ifdef GGML_USE_MPI
        GGML_ASSERT(false && "not implemented");
#endif

        inpL = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, n_tokens);

        ggml_allocr_alloc(lctx.alloc, inpL);
        if (!ggml_allocr_is_measure(lctx.alloc)) {
            memcpy(inpL->data, batch.embd, n_tokens * n_embd * ggml_element_size(inpL));
        }
    }

    const int i_gpu_start = n_layer - n_gpu_layers;
    (void) i_gpu_start;

    // offload functions set the tensor output backend to GPU
    // tensors are GPU-accelerated if any input or the output has been offloaded
    offload_func_t offload_func_nr = llama_nop; // nr = non-repeating
    offload_func_t offload_func_kq = llama_nop;
    offload_func_t offload_func_v  = llama_nop;

#ifdef GGML_USE_CUBLAS
    if (n_gpu_layers > n_layer) {
        offload_func_nr = ggml_cuda_assign_buffers_no_alloc;
    }
    if (n_gpu_layers > n_layer + 1) {
        offload_func_v  = ggml_cuda_assign_buffers_no_alloc;
    }
    if (n_gpu_layers > n_layer + 2) {
        offload_func_kq = ggml_cuda_assign_buffers_no_alloc;
    }
#endif // GGML_USE_CUBLAS

    // KQ_scale
    struct ggml_tensor * KQ_scale = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, 1);
    ggml_set_name(KQ_scale, "1/sqrt(n_embd_head)");
    ggml_allocr_alloc(lctx.alloc, KQ_scale);
    if (!ggml_allocr_is_measure(lctx.alloc)) {
        ggml_set_f32(KQ_scale, 1.0f/sqrtf(float(n_embd_head)));
    }

    // KQ_mask (mask for 1 head, it will be broadcasted to all heads)
    struct ggml_tensor * KQ_mask = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, n_kv, n_tokens, 1);
    offload_func_kq(KQ_mask);
    ggml_set_name(KQ_mask, "KQ_mask");
    ggml_allocr_alloc(lctx.alloc, KQ_mask);
    if (!ggml_allocr_is_measure(lctx.alloc)) {
        float * data = (float *) KQ_mask->data;
        memset(data, 0, ggml_nbytes(KQ_mask));

        for (int h = 0; h < 1; ++h) {
            for (int j = 0; j < n_tokens; ++j) {
                const llama_pos    pos    = batch.pos[j];
                const llama_seq_id seq_id = batch.seq_id[j][0];

                for (int i = 0; i < n_kv; ++i) {
                    if (!kv_self.cells[i].has_seq_id(seq_id) || kv_self.cells[i].pos > pos) {
                        data[h*(n_kv*n_tokens) + j*n_kv + i] = -INFINITY;
                    }
                }
            }
        }
    }

    // KQ_pos - contains the positions
    struct ggml_tensor * KQ_pos = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    offload_func_kq(KQ_pos);
    ggml_set_name(KQ_pos, "KQ_pos");
    ggml_allocr_alloc(lctx.alloc, KQ_pos);
    if (!ggml_allocr_is_measure(lctx.alloc)) {
        int * data = (int *) KQ_pos->data;
        for (int i = 0; i < n_tokens; ++i) {
            data[i] = batch.pos[i];
        }
    }

    // shift the entire K-cache if needed
    if (do_rope_shift) {
        struct ggml_tensor * K_shift = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_ctx);
        offload_func_kq(K_shift);
        ggml_set_name(K_shift, "K_shift");
        ggml_allocr_alloc(lctx.alloc, K_shift);
        if (!ggml_allocr_is_measure(lctx.alloc)) {
            int * data = (int *) K_shift->data;
            for (int i = 0; i < n_ctx; ++i) {
                data[i] = kv_self.cells[i].delta;
            }
        }

        for (int il = 0; il < n_layer; ++il) {
            struct ggml_tensor * tmp =
                    ggml_rope_custom_inplace(ctx0,
                        ggml_view_3d(ctx0, kv_self.k,
                            n_embd_head, n_head_kv, n_ctx,
                            ggml_element_size(kv_self.k)*n_embd_head,
                            ggml_element_size(kv_self.k)*n_embd_gqa,
                            ggml_element_size(kv_self.k)*n_embd_gqa*n_ctx*il),
                        K_shift, n_embd_head, 0, 0, freq_base, freq_scale);
            offload_func_kq(tmp);
            ggml_build_forward_expand(gf, tmp);
        }
    }

    for (int il = 0; il < n_layer; ++il) {
        ggml_format_name(inpL, "layer_inp_%d", il);

        offload_func_t offload_func = llama_nop;

#ifdef GGML_USE_CUBLAS
        if (il >= i_gpu_start) {
            offload_func = ggml_cuda_assign_buffers_no_alloc;
        }
#endif // GGML_USE_CUBLAS

        struct ggml_tensor * inpSA = inpL;

        // norm
        {
            cur = ggml_rms_norm(ctx0, inpL, norm_rms_eps);
            offload_func(cur);
            ggml_set_name(cur, "rms_norm_0");

            // cur = cur*attn_norm(broadcasted)
            cur = ggml_mul(ctx0, cur, model.layers[il].attn_norm);
            offload_func(cur);
            ggml_set_name(cur, "attention_norm_0");
        }

        // self-attention
        {
            // compute Q and K and RoPE them
            struct ggml_tensor * tmpk = ggml_mul_mat(ctx0, model.layers[il].wk, cur);
            offload_func_kq(tmpk);
            ggml_set_name(tmpk, "tmpk");

            struct ggml_tensor * tmpq = ggml_mul_mat(ctx0, model.layers[il].wq, cur);
            offload_func_kq(tmpq);
            ggml_set_name(tmpq, "tmpq");

            struct ggml_tensor * Kcur = ggml_rope_custom(ctx0, ggml_reshape_3d(ctx0, tmpk, n_embd_head, n_head_kv, n_tokens), KQ_pos, n_embd_head, 0, 0, freq_base, freq_scale);
            offload_func_kq(Kcur);
            ggml_set_name(Kcur, "Kcur");

            struct ggml_tensor * Qcur = ggml_rope_custom(ctx0, ggml_reshape_3d(ctx0, tmpq, n_embd_head, n_head,    n_tokens), KQ_pos, n_embd_head, 0, 0, freq_base, freq_scale);
            offload_func_kq(Qcur);
            ggml_set_name(Qcur, "Qcur");

            // store key and value to memory
            {
                // compute the transposed [n_tokens, n_embd] V matrix

                struct ggml_tensor * tmpv = ggml_mul_mat(ctx0, model.layers[il].wv, cur);
                offload_func_v(tmpv);
                ggml_set_name(tmpv, "tmpv");

                struct ggml_tensor * Vcur = ggml_transpose(ctx0, ggml_reshape_2d(ctx0, tmpv, n_embd_gqa, n_tokens));
                offload_func_v(Vcur);
                ggml_set_name(Vcur, "Vcur");

                struct ggml_tensor * k = ggml_view_1d(ctx0, kv_self.k, n_tokens*n_embd_gqa, (ggml_element_size(kv_self.k)*n_embd_gqa)*(il*n_ctx + kv_head));
                offload_func_kq(k);
                ggml_set_name(k, "k");

                struct ggml_tensor * v = ggml_view_2d(ctx0, kv_self.v, n_tokens, n_embd_gqa,
                        (   n_ctx)*ggml_element_size(kv_self.v),
                        (il*n_ctx)*ggml_element_size(kv_self.v)*n_embd_gqa + kv_head*ggml_element_size(kv_self.v));
                offload_func_v(v);
                ggml_set_name(v, "v");

                // important: storing RoPE-ed version of K in the KV cache!
                ggml_build_forward_expand(gf, ggml_cpy(ctx0, Kcur, k));
                ggml_build_forward_expand(gf, ggml_cpy(ctx0, Vcur, v));
            }

            struct ggml_tensor * Q = ggml_permute(ctx0, Qcur, 0, 2, 1, 3);
            offload_func_kq(Q);
            ggml_set_name(Q, "Q");

            struct ggml_tensor * K =
                ggml_view_3d(ctx0, kv_self.k,
                        n_embd_head, n_kv, n_head_kv,
                        ggml_element_size(kv_self.k)*n_embd_gqa,
                        ggml_element_size(kv_self.k)*n_embd_head,
                        ggml_element_size(kv_self.k)*n_embd_gqa*n_ctx*il);
            offload_func_kq(K);
            ggml_set_name(K, "K");

            // K * Q
            struct ggml_tensor * KQ = ggml_mul_mat(ctx0, K, Q);
            offload_func_kq(KQ);
            ggml_set_name(KQ, "KQ");

            // KQ_scaled = KQ / sqrt(n_embd_head)
            // KQ_scaled shape [n_kv, n_tokens, n_head, 1]
            struct ggml_tensor * KQ_scaled = ggml_scale(ctx0, KQ, KQ_scale);
            offload_func_kq(KQ_scaled);
            ggml_set_name(KQ_scaled, "KQ_scaled");

            // KQ_masked = mask_past(KQ_scaled)
            struct ggml_tensor * KQ_masked = ggml_add(ctx0, KQ_scaled, KQ_mask);
            offload_func_kq(KQ_masked);
            ggml_set_name(KQ_masked, "KQ_masked");

            // KQ = soft_max(KQ_masked)
            struct ggml_tensor * KQ_soft_max = ggml_soft_max(ctx0, KQ_masked);
            offload_func_v(KQ_soft_max);
            ggml_set_name(KQ_soft_max, "KQ_soft_max");

            // split cached V into n_head heads
            struct ggml_tensor * V =
                ggml_view_3d(ctx0, kv_self.v,
                        n_kv, n_embd_head, n_head_kv,
                        ggml_element_size(kv_self.v)*n_ctx,
                        ggml_element_size(kv_self.v)*n_ctx*n_embd_head,
                        ggml_element_size(kv_self.v)*n_ctx*n_embd_gqa*il);
            offload_func_v(V);
            ggml_set_name(V, "V");

#if 1
            struct ggml_tensor * KQV = ggml_mul_mat(ctx0, V, KQ_soft_max);
            offload_func_v(KQV);
            ggml_set_name(KQV, "KQV");
#else
            // make V contiguous in memory to speed up the matmul, however we waste time on the copy
            // on M1 this is faster for the perplexity computation, but ~5% slower for the single-token generation
            // is there a better way?
            struct ggml_tensor * V_cont = ggml_cpy(ctx0, V, ggml_new_tensor_3d(ctx0, kv_self.v->type, n_ctx, n_embd_head, n_head));
            struct ggml_tensor * KQV = ggml_mul_mat(ctx0, V_cont, KQ_soft_max);
#endif

            // KQV_merged = KQV.permute(0, 2, 1, 3)
            struct ggml_tensor * KQV_merged = ggml_permute(ctx0, KQV, 0, 2, 1, 3);
            offload_func_v(KQV_merged);
            ggml_set_name(KQV_merged, "KQV_merged");

            // cur = KQV_merged.contiguous().view(n_embd, n_tokens)
            cur = ggml_cont_2d(ctx0, KQV_merged, n_embd, n_tokens);
            offload_func_v(cur);
            ggml_set_name(cur, "KQV_merged_contiguous");

            // projection (no bias)
            cur = ggml_mul_mat(ctx0,
                    model.layers[il].wo,
                    cur);
            offload_func(cur);
            ggml_set_name(cur, "result_wo");
        }

        struct ggml_tensor * inpFF = ggml_add(ctx0, cur, inpSA);
        offload_func(inpFF);
        ggml_set_name(inpFF, "inpFF");

        // feed-forward network
        {
            // norm
            {
                cur = ggml_rms_norm(ctx0, inpFF, norm_rms_eps);
                offload_func(cur);
                ggml_set_name(cur, "rms_norm_1");

                // cur = cur*ffn_norm(broadcasted)
                cur = ggml_mul(ctx0, cur, model.layers[il].ffn_norm);
                offload_func(cur);
                ggml_set_name(cur, "ffn_norm");
            }

            struct ggml_tensor * tmp = ggml_mul_mat(ctx0,
                    model.layers[il].w3,
                    cur);
            offload_func(tmp);
            ggml_set_name(tmp, "result_w3");

            cur = ggml_mul_mat(ctx0,
                    model.layers[il].w1,
                    cur);
            offload_func(cur);
            ggml_set_name(cur, "result_w1");

            // SILU activation
            cur = ggml_silu(ctx0, cur);
            offload_func(cur);
            ggml_set_name(cur, "silu");

            cur = ggml_mul(ctx0, cur, tmp);
            offload_func(cur);
            ggml_set_name(cur, "silu_x_result_w3");

            cur = ggml_mul_mat(ctx0,
                    model.layers[il].w2,
                    cur);
            offload_func(cur);
            ggml_set_name(cur, "result_w2");
        }

        cur = ggml_add(ctx0, cur, inpFF);
        offload_func(cur);
        ggml_set_name(cur, "inpFF_+_result_w2");

        // input for next layer
        inpL = cur;
    }

    cur = inpL;

    // norm
    {
        cur = ggml_rms_norm(ctx0, cur, norm_rms_eps);
        offload_func_nr(cur);
        ggml_set_name(cur, "rms_norm_2");

        // cur = cur*norm(broadcasted)
        cur = ggml_mul(ctx0, cur, model.output_norm);
        // offload_func_nr(cur); // TODO CPU + GPU mirrored backend
        ggml_set_name(cur, "result_norm");
    }

    // lm_head
    cur = ggml_mul_mat(ctx0, model.output, cur);
    ggml_set_name(cur, "result_output");

    ggml_build_forward_expand(gf, cur);

    ggml_free(ctx0);

    return gf;
}

static struct ggml_cgraph * llm_build_baichaun(
         llama_context & lctx,
     const llama_batch & batch) {
    const auto & model   = lctx.model;
    const auto & hparams = model.hparams;
    const auto & cparams = lctx.cparams;

    const auto & kv_self = lctx.kv_self;

    GGML_ASSERT(!!kv_self.ctx);

    const int64_t n_embd      = hparams.n_embd;
    const int64_t n_layer     = hparams.n_layer;
    const int64_t n_ctx       = cparams.n_ctx;
    const int64_t n_head      = hparams.n_head;
    const int64_t n_head_kv   = hparams.n_head_kv;
    const int64_t n_embd_head = hparams.n_embd_head();
    const int64_t n_embd_gqa  = hparams.n_embd_gqa();

    GGML_ASSERT(n_embd_head == hparams.n_rot);

    const float freq_base    = cparams.rope_freq_base;
    const float freq_scale   = cparams.rope_freq_scale;
    const float norm_rms_eps = hparams.f_norm_rms_eps;

    const int n_gpu_layers = model.n_gpu_layers;

    const int32_t n_tokens = batch.n_tokens;
    const int32_t n_kv     = ggml_allocr_is_measure(lctx.alloc) ? n_ctx            : kv_self.n;
    const int32_t kv_head  = ggml_allocr_is_measure(lctx.alloc) ? n_ctx - n_tokens : kv_self.head;

    const bool do_rope_shift = ggml_allocr_is_measure(lctx.alloc) || kv_self.has_shift;

    auto & buf_compute = lctx.buf_compute;

    struct ggml_init_params params = {
        /*.mem_size   =*/ buf_compute.size,
        /*.mem_buffer =*/ buf_compute.data,
        /*.no_alloc   =*/ true,
    };

    struct ggml_context * ctx0 = ggml_init(params);

    ggml_cgraph * gf = ggml_new_graph(ctx0);

    struct ggml_tensor * cur;
    struct ggml_tensor * inpL;

    if (batch.token) {
        struct ggml_tensor * inp_tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);

        ggml_allocr_alloc(lctx.alloc, inp_tokens);
        if (!ggml_allocr_is_measure(lctx.alloc)) {
            memcpy(inp_tokens->data, batch.token, n_tokens*ggml_element_size(inp_tokens));
        }
        ggml_set_name(inp_tokens, "inp_tokens");

        inpL = ggml_get_rows(ctx0, model.tok_embeddings, inp_tokens);
    } else {
#ifdef GGML_USE_MPI
        GGML_ASSERT(false && "not implemented");
#endif

        inpL = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, n_tokens);

        ggml_allocr_alloc(lctx.alloc, inpL);
        if (!ggml_allocr_is_measure(lctx.alloc)) {
            memcpy(inpL->data, batch.embd, n_tokens * n_embd * ggml_element_size(inpL));
        }
    }

    const int i_gpu_start = n_layer - n_gpu_layers;
    (void) i_gpu_start;

    // offload functions set the tensor output backend to GPU
    // tensors are GPU-accelerated if any input or the output has been offloaded
    offload_func_t offload_func_nr = llama_nop; // nr = non-repeating
    offload_func_t offload_func_kq = llama_nop;
    offload_func_t offload_func_v  = llama_nop;

#ifdef GGML_USE_CUBLAS
    if (n_gpu_layers > n_layer) {
        offload_func_nr = ggml_cuda_assign_buffers_no_alloc;
    }
    if (n_gpu_layers > n_layer + 1) {
        offload_func_v  = ggml_cuda_assign_buffers_no_alloc;
    }
    if (n_gpu_layers > n_layer + 2) {
        offload_func_kq = ggml_cuda_assign_buffers_no_alloc;
    }
#endif // GGML_USE_CUBLAS

    // KQ_scale
    struct ggml_tensor * KQ_scale = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, 1);
    ggml_set_name(KQ_scale, "1/sqrt(n_embd_head)");
    ggml_allocr_alloc(lctx.alloc, KQ_scale);
    if (!ggml_allocr_is_measure(lctx.alloc)) {
        ggml_set_f32(KQ_scale, 1.0f/sqrtf(float(n_embd)/n_head));
    }

    // KQ_mask (mask for 1 head, it will be broadcasted to all heads)
    struct ggml_tensor * KQ_mask = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, n_kv, n_tokens, 1);
    offload_func_kq(KQ_mask);
    ggml_set_name(KQ_mask, "KQ_mask");
    ggml_allocr_alloc(lctx.alloc, KQ_mask);
    if (!ggml_allocr_is_measure(lctx.alloc)) {
        float * data = (float *) KQ_mask->data;
        memset(data, 0, ggml_nbytes(KQ_mask));

        for (int h = 0; h < 1; ++h) {
            for (int j = 0; j < n_tokens; ++j) {
                const llama_pos    pos    = batch.pos[j];
                const llama_seq_id seq_id = batch.seq_id[j][0];

                for (int i = 0; i < n_kv; ++i) {
                    if (!kv_self.cells[i].has_seq_id(seq_id) || kv_self.cells[i].pos > pos) {
                        data[h*(n_kv*n_tokens) + j*n_kv + i] = -INFINITY;
                    }
                }
            }
        }
    }

    // KQ_pos - contains the positions
    struct ggml_tensor * KQ_pos = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    offload_func_kq(KQ_pos);
    ggml_set_name(KQ_pos, "KQ_pos");
    ggml_allocr_alloc(lctx.alloc, KQ_pos);
    if (!ggml_allocr_is_measure(lctx.alloc)) {
        int * data = (int *) KQ_pos->data;
        for (int i = 0; i < n_tokens; ++i) {
            data[i] = batch.pos[i];
        }
    }

    // shift the entire K-cache if needed
    if (do_rope_shift) {
        struct ggml_tensor * K_shift = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_ctx);
        offload_func_kq(K_shift);
        ggml_set_name(K_shift, "K_shift");
        ggml_allocr_alloc(lctx.alloc, K_shift);
        if (!ggml_allocr_is_measure(lctx.alloc)) {
            int * data = (int *) K_shift->data;
            for (int i = 0; i < n_ctx; ++i) {
                data[i] = kv_self.cells[i].delta;
            }
        }

        for (int il = 0; il < n_layer; ++il) {
            struct ggml_tensor * tmp =
                    ggml_rope_custom_inplace(ctx0,
                        ggml_view_3d(ctx0, kv_self.k,
                            n_embd_head, n_head_kv, n_ctx,
                            ggml_element_size(kv_self.k)*n_embd_head,
                            ggml_element_size(kv_self.k)*n_embd_gqa,
                            ggml_element_size(kv_self.k)*n_embd_gqa*n_ctx*il),
                        K_shift, n_embd_head, 0, 0, freq_base, freq_scale);
            offload_func_kq(tmp);
            ggml_build_forward_expand(gf, tmp);
        }
    }

    for (int il = 0; il < n_layer; ++il) {
        ggml_format_name(inpL, "layer_inp_%d", il);

        offload_func_t offload_func = llama_nop;

#ifdef GGML_USE_CUBLAS
        if (il >= i_gpu_start) {
            offload_func = ggml_cuda_assign_buffers_no_alloc;
        }
#endif // GGML_USE_CUBLAS

        struct ggml_tensor * inpSA = inpL;

        // norm
        {
            cur = ggml_rms_norm(ctx0, inpL, norm_rms_eps);
            offload_func(cur);
            ggml_set_name(cur, "rms_norm_0");

            // cur = cur*attn_norm(broadcasted)
            cur = ggml_mul(ctx0, cur, model.layers[il].attn_norm);
            offload_func(cur);
            ggml_set_name(cur, "attention_norm_0");
        }

        // self-attention
        {
            // compute Q and K and RoPE them
            struct ggml_tensor * tmpk = ggml_mul_mat(ctx0, model.layers[il].wk, cur);
            offload_func_kq(tmpk);
            ggml_set_name(tmpk, "tmpk");

            struct ggml_tensor * tmpq = ggml_mul_mat(ctx0, model.layers[il].wq, cur);
            offload_func_kq(tmpq);
            ggml_set_name(tmpq, "tmpq");

            struct ggml_tensor * Kcur;
            struct ggml_tensor * Qcur;
            switch (model.type) {
                case MODEL_7B:
                    Kcur = ggml_rope_custom(ctx0, ggml_reshape_3d(ctx0, tmpk, n_embd_head, n_head_kv, n_tokens), KQ_pos, n_embd_head, 0, 0, freq_base, freq_scale);
                    Qcur = ggml_rope_custom(ctx0, ggml_reshape_3d(ctx0, tmpq, n_embd_head, n_head, n_tokens),    KQ_pos, n_embd_head, 0, 0, freq_base, freq_scale);
                    break;
                case MODEL_13B:
                    Kcur = ggml_reshape_3d(ctx0, tmpk, n_embd/n_head, n_head, n_tokens);
                    Qcur = ggml_reshape_3d(ctx0, tmpq, n_embd/n_head, n_head, n_tokens);
                    break;
                default:
                    GGML_ASSERT(false);
            }

            offload_func_kq(Kcur);
            ggml_set_name(Kcur, "Kcur");

            offload_func_kq(Qcur);
            ggml_set_name(Qcur, "Qcur");

            // store key and value to memory
            {
                // compute the transposed [n_tokens, n_embd] V matrix

                struct ggml_tensor * tmpv = ggml_mul_mat(ctx0, model.layers[il].wv, cur);
                offload_func_v(tmpv);
                ggml_set_name(tmpv, "tmpv");

                struct ggml_tensor * Vcur = ggml_transpose(ctx0, ggml_reshape_2d(ctx0, tmpv, n_embd_gqa, n_tokens));
                offload_func_v(Vcur);
                ggml_set_name(Vcur, "Vcur");

                struct ggml_tensor * k = ggml_view_1d(ctx0, kv_self.k, n_tokens*n_embd_gqa, (ggml_element_size(kv_self.k)*n_embd_gqa)*(il*n_ctx + kv_head));
                offload_func_kq(k);
                ggml_set_name(k, "k");

                struct ggml_tensor * v = ggml_view_2d(ctx0, kv_self.v, n_tokens, n_embd_gqa,
                        (   n_ctx)*ggml_element_size(kv_self.v),
                        (il*n_ctx)*ggml_element_size(kv_self.v)*n_embd_gqa + kv_head*ggml_element_size(kv_self.v));
                offload_func_v(v);
                ggml_set_name(v, "v");

                // important: storing RoPE-ed version of K in the KV cache!
                ggml_build_forward_expand(gf, ggml_cpy(ctx0, Kcur, k));
                ggml_build_forward_expand(gf, ggml_cpy(ctx0, Vcur, v));
            }

            struct ggml_tensor * Q = ggml_permute(ctx0, Qcur, 0, 2, 1, 3);
            offload_func_kq(Q);
            ggml_set_name(Q, "Q");

            struct ggml_tensor * K =
                ggml_view_3d(ctx0, kv_self.k,
                        n_embd_head, n_kv, n_head_kv,
                        ggml_element_size(kv_self.k)*n_embd_gqa,
                        ggml_element_size(kv_self.k)*n_embd_head,
                        ggml_element_size(kv_self.k)*n_embd_gqa*n_ctx*il);
            offload_func_kq(K);
            ggml_set_name(K, "K");

            // K * Q
            struct ggml_tensor * KQ = ggml_mul_mat(ctx0, K, Q);
            offload_func_kq(KQ);
            ggml_set_name(KQ, "KQ");

            // KQ_scaled = KQ / sqrt(n_embd_head)
            // KQ_scaled shape [n_past + n_tokens, n_tokens, n_head, 1]
            struct ggml_tensor * KQ_scaled = ggml_scale(ctx0, KQ, KQ_scale);
            offload_func_kq(KQ_scaled);
            ggml_set_name(KQ_scaled, "KQ_scaled");

            struct ggml_tensor * KQ_masked;
            struct ggml_tensor * KQ_scaled_alibi;

            switch (model.type) {
                case MODEL_7B:
                    KQ_masked = ggml_add(ctx0, KQ_scaled, KQ_mask);
                    break;
                case MODEL_13B:
                    // TODO: replace with ggml_add()
                    KQ_scaled_alibi = ggml_alibi(ctx0, KQ_scaled, /*n_past*/ 0, n_head, 8);
                    ggml_set_name(KQ_scaled_alibi, "KQ_scaled_alibi");
                    KQ_masked = ggml_add(ctx0, KQ_scaled_alibi, KQ_mask);
                    break;
                default:
                    GGML_ASSERT(false);
            }

            // KQ = soft_max(KQ_masked)
            struct ggml_tensor * KQ_soft_max = ggml_soft_max(ctx0, KQ_masked);
            offload_func_v(KQ_soft_max);
            ggml_set_name(KQ_soft_max, "KQ_soft_max");

            // split cached V into n_head heads
            struct ggml_tensor * V =
                ggml_view_3d(ctx0, kv_self.v,
                        n_kv, n_embd_head, n_head_kv,
                        ggml_element_size(kv_self.v)*n_ctx,
                        ggml_element_size(kv_self.v)*n_ctx*n_embd_head,
                        ggml_element_size(kv_self.v)*n_ctx*n_embd_gqa*il);
            offload_func_v(V);
            ggml_set_name(V, "V");

            struct ggml_tensor * KQV = ggml_mul_mat(ctx0, V, KQ_soft_max);
            offload_func_v(KQV);
            ggml_set_name(KQV, "KQV");

            // KQV_merged = KQV.permute(0, 2, 1, 3)
            struct ggml_tensor * KQV_merged = ggml_permute(ctx0, KQV, 0, 2, 1, 3);
            offload_func_v(KQV_merged);
            ggml_set_name(KQV_merged, "KQV_merged");

            // cur = KQV_merged.contiguous().view(n_embd, n_tokens)
            cur = ggml_cont_2d(ctx0, KQV_merged, n_embd, n_tokens);
            offload_func_v(cur);
            ggml_set_name(cur, "KQV_merged_contiguous");

            // projection (no bias)
            cur = ggml_mul_mat(ctx0,
                    model.layers[il].wo,
                    cur);
            offload_func(cur);
            ggml_set_name(cur, "result_wo");
        }

        struct ggml_tensor * inpFF = ggml_add(ctx0, cur, inpSA);
        offload_func(inpFF);
        ggml_set_name(inpFF, "inpFF");

        // feed-forward network
        {
            // norm
            {
                cur = ggml_rms_norm(ctx0, inpFF, norm_rms_eps);
                offload_func(cur);
                ggml_set_name(cur, "rms_norm_1");

                // cur = cur*ffn_norm(broadcasted)
                cur = ggml_mul(ctx0, cur, model.layers[il].ffn_norm);
                offload_func(cur);
                ggml_set_name(cur, "ffn_norm");
            }

            struct ggml_tensor * tmp = ggml_mul_mat(ctx0,
                    model.layers[il].w3,
                    cur);
            offload_func(tmp);
            ggml_set_name(tmp, "result_w3");

            cur = ggml_mul_mat(ctx0,
                    model.layers[il].w1,
                    cur);
            offload_func(cur);
            ggml_set_name(cur, "result_w1");

            // SILU activation
            cur = ggml_silu(ctx0, cur);
            offload_func(cur);
            ggml_set_name(cur, "silu");

            cur = ggml_mul(ctx0, cur, tmp);
            offload_func(cur);
            ggml_set_name(cur, "silu_x_result_w3");

            cur = ggml_mul_mat(ctx0,
                    model.layers[il].w2,
                    cur);
            offload_func(cur);
            ggml_set_name(cur, "result_w2");
        }

        cur = ggml_add(ctx0, cur, inpFF);
        offload_func(cur);
        ggml_set_name(cur, "inpFF_+_result_w2");

        // input for next layer
        inpL = cur;
    }

    cur = inpL;

    // norm
    {
        cur = ggml_rms_norm(ctx0, cur, norm_rms_eps);
        offload_func_nr(cur);
        ggml_set_name(cur, "rms_norm_2");

        // cur = cur*norm(broadcasted)
        cur = ggml_mul(ctx0, cur, model.output_norm);
        // offload_func_nr(cur); // TODO CPU + GPU mirrored backend
        ggml_set_name(cur, "result_norm");
    }

    // lm_head
    cur = ggml_mul_mat(ctx0, model.output, cur);
    ggml_set_name(cur, "result_output");

    ggml_build_forward_expand(gf, cur);

    ggml_free(ctx0);

    return gf;
}

static struct ggml_cgraph * llm_build_refact(
         llama_context & lctx,
     const llama_batch & batch) {
    const auto & model   = lctx.model;
    const auto & hparams = model.hparams;
    const auto & cparams = lctx.cparams;

    const auto & kv_self = lctx.kv_self;

    GGML_ASSERT(!!kv_self.ctx);

    const int64_t n_embd      = hparams.n_embd;
    const int64_t n_layer     = hparams.n_layer;
    const int64_t n_ctx       = cparams.n_ctx;
    const int64_t n_head      = hparams.n_head;
    const int64_t n_head_kv   = hparams.n_head_kv;
    const int64_t n_embd_head = hparams.n_embd_head();
    const int64_t n_embd_gqa  = hparams.n_embd_gqa();

    const float norm_rms_eps = hparams.f_norm_rms_eps;

    const int n_gpu_layers = model.n_gpu_layers;

    const int32_t n_tokens = batch.n_tokens;
    const int32_t n_kv     = ggml_allocr_is_measure(lctx.alloc) ? n_ctx            : kv_self.n;
    const int32_t kv_head  = ggml_allocr_is_measure(lctx.alloc) ? n_ctx - n_tokens : kv_self.head;

    // printf("n_kv = %d\n", n_kv);

    auto & buf_compute = lctx.buf_compute;

    struct ggml_init_params params = {
        /*.mem_size   =*/ buf_compute.size,
        /*.mem_buffer =*/ buf_compute.data,
        /*.no_alloc   =*/ true,
    };

    struct ggml_context * ctx0 = ggml_init(params);

    ggml_cgraph * gf = ggml_new_graph(ctx0);

    struct ggml_tensor * cur;
    struct ggml_tensor * inpL;

    if (batch.token) {
        struct ggml_tensor * inp_tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);

        ggml_allocr_alloc(lctx.alloc, inp_tokens);
        if (!ggml_allocr_is_measure(lctx.alloc)) {
            memcpy(inp_tokens->data, batch.token, n_tokens*ggml_element_size(inp_tokens));
        }
        ggml_set_name(inp_tokens, "inp_tokens");

        inpL = ggml_get_rows(ctx0, model.tok_embeddings, inp_tokens);
    } else {
#ifdef GGML_USE_MPI
        GGML_ASSERT(false && "not implemented");
#endif

        inpL = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, n_tokens);

        ggml_allocr_alloc(lctx.alloc, inpL);
        if (!ggml_allocr_is_measure(lctx.alloc)) {
            memcpy(inpL->data, batch.embd, n_tokens * n_embd * ggml_element_size(inpL));
        }
    }

    const int i_gpu_start = n_layer - n_gpu_layers;
    (void) i_gpu_start;

    // offload functions set the tensor output backend to GPU
    // tensors are GPU-accelerated if any input or the output has been offloaded
    offload_func_t offload_func_nr = llama_nop; // nr = non-repeating
    offload_func_t offload_func_kq = llama_nop;
    offload_func_t offload_func_v  = llama_nop;

#ifdef GGML_USE_CUBLAS
    if (n_gpu_layers > n_layer) {
        offload_func_nr = ggml_cuda_assign_buffers_no_alloc;
    }
    if (n_gpu_layers > n_layer + 1) {
        offload_func_v  = ggml_cuda_assign_buffers_no_alloc;
    }
    if (n_gpu_layers > n_layer + 2) {
        offload_func_kq = ggml_cuda_assign_buffers_no_alloc;
    }
#endif // GGML_USE_CUBLAS

    // KQ_scale
    struct ggml_tensor * KQ_scale = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, 1);
    ggml_set_name(KQ_scale, "1/sqrt(n_embd_head)");
    ggml_allocr_alloc(lctx.alloc, KQ_scale);
    if (!ggml_allocr_is_measure(lctx.alloc)) {
        ggml_set_f32(KQ_scale, 1.0f/sqrtf(float(n_embd_head)));
    }

    // KQ_mask (mask for 1 head, it will be broadcasted to all heads)
    struct ggml_tensor * KQ_mask = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, n_kv, n_tokens, 1);
    offload_func_kq(KQ_mask);
    ggml_set_name(KQ_mask, "KQ_mask");
    ggml_allocr_alloc(lctx.alloc, KQ_mask);
    if (!ggml_allocr_is_measure(lctx.alloc)) {
        float * data = (float *) KQ_mask->data;
        memset(data, 0, ggml_nbytes(KQ_mask));

        for (int h = 0; h < 1; ++h) {
            for (int j = 0; j < n_tokens; ++j) {
                const llama_pos    pos    = batch.pos[j];
                const llama_seq_id seq_id = batch.seq_id[j][0];

                for (int i = 0; i < n_kv; ++i) {
                    if (!kv_self.cells[i].has_seq_id(seq_id) || kv_self.cells[i].pos > pos) {
                        data[h*(n_kv*n_tokens) + j*n_kv + i] = -INFINITY;
                    }
                }
            }
        }
    }

    for (int il = 0; il < n_layer; ++il) {
        ggml_format_name(inpL, "layer_inp_%d", il);

        offload_func_t offload_func = llama_nop;

#ifdef GGML_USE_CUBLAS
        if (il >= i_gpu_start) {
            offload_func = ggml_cuda_assign_buffers_no_alloc;
        }
#endif // GGML_USE_CUBLAS

        struct ggml_tensor * inpSA = inpL;

        // norm
        {
            cur = ggml_rms_norm(ctx0, inpL, norm_rms_eps);
            offload_func(cur);
            ggml_set_name(cur, "rms_norm_0");

            // cur = cur*attn_norm(broadcasted)
            cur = ggml_mul(ctx0, cur, model.layers[il].attn_norm);
            offload_func(cur);
            ggml_set_name(cur, "attention_norm_0");
        }

        // self-attention
        {
            // compute Q and K
            struct ggml_tensor * tmpk = ggml_mul_mat(ctx0, model.layers[il].wk, cur);
            offload_func_kq(tmpk);
            ggml_set_name(tmpk, "tmpk");

            struct ggml_tensor * tmpq = ggml_mul_mat(ctx0, model.layers[il].wq, cur);
            offload_func_kq(tmpq);
            ggml_set_name(tmpq, "tmpq");

            struct ggml_tensor * Kcur = ggml_reshape_3d(ctx0, tmpk, n_embd_head, n_head_kv, n_tokens);
            offload_func_kq(Kcur);
            ggml_set_name(Kcur, "Kcur");

            struct ggml_tensor * Qcur = ggml_reshape_3d(ctx0, tmpq, n_embd_head, n_head,    n_tokens);
            offload_func_kq(Qcur);
            ggml_set_name(Qcur, "Qcur");

            // store key and value to memory
            {
                // compute the transposed [n_tokens, n_embd] V matrix

                struct ggml_tensor * tmpv = ggml_mul_mat(ctx0, model.layers[il].wv, cur);
                offload_func_v(tmpv);
                ggml_set_name(tmpv, "tmpv");

                struct ggml_tensor * Vcur = ggml_transpose(ctx0, ggml_reshape_2d(ctx0, tmpv, n_embd_gqa, n_tokens));
                offload_func_v(Vcur);
                ggml_set_name(Vcur, "Vcur");

                struct ggml_tensor * k = ggml_view_1d(ctx0, kv_self.k, n_tokens*n_embd_gqa, (ggml_element_size(kv_self.k)*n_embd_gqa)*(il*n_ctx + kv_head));
                offload_func_kq(k);
                ggml_set_name(k, "k");

                struct ggml_tensor * v = ggml_view_2d(ctx0, kv_self.v, n_tokens, n_embd_gqa,
                        (   n_ctx)*ggml_element_size(kv_self.v),
                        (il*n_ctx)*ggml_element_size(kv_self.v)*n_embd_gqa + kv_head*ggml_element_size(kv_self.v));
                offload_func_v(v);
                ggml_set_name(v, "v");

                ggml_build_forward_expand(gf, ggml_cpy(ctx0, Kcur, k));
                ggml_build_forward_expand(gf, ggml_cpy(ctx0, Vcur, v));
            }

            struct ggml_tensor * Q = ggml_permute(ctx0, Qcur, 0, 2, 1, 3);
            offload_func_kq(Q);
            ggml_set_name(Q, "Q");

            struct ggml_tensor * K =
                ggml_view_3d(ctx0, kv_self.k,
                        n_embd_head, n_kv, n_head_kv,
                        ggml_element_size(kv_self.k)*n_embd_gqa,
                        ggml_element_size(kv_self.k)*n_embd_head,
                        ggml_element_size(kv_self.k)*n_embd_gqa*n_ctx*il);
            offload_func_kq(K);
            ggml_set_name(K, "K");

            // K * Q
            struct ggml_tensor * KQ = ggml_mul_mat(ctx0, K, Q);
            offload_func_kq(KQ);
            ggml_set_name(KQ, "KQ");

            // KQ_scaled = KQ / sqrt(n_embd_head)
            // KQ_scaled shape [n_kv, n_tokens, n_head, 1]
            struct ggml_tensor * KQ_scaled = ggml_scale(ctx0, KQ, KQ_scale);
            offload_func_kq(KQ_scaled);
            ggml_set_name(KQ_scaled, "KQ_scaled");

            // KQ_masked = mask_past(KQ_scaled)
            struct ggml_tensor * KQ_scaled_alibi = ggml_alibi(ctx0, KQ_scaled, /*n_past*/ 0, n_head, 8);
            ggml_set_name(KQ_scaled_alibi, "KQ_scaled_alibi");

            struct ggml_tensor * KQ_masked = ggml_add(ctx0, KQ_scaled_alibi, KQ_mask);
            offload_func_kq(KQ_masked);
            ggml_set_name(KQ_masked, "KQ_masked");

            // KQ = soft_max(KQ_masked)
            struct ggml_tensor * KQ_soft_max = ggml_soft_max(ctx0, KQ_masked);
            offload_func_v(KQ_soft_max);
            ggml_set_name(KQ_soft_max, "KQ_soft_max");

            // split cached V into n_head heads
            struct ggml_tensor * V =
                ggml_view_3d(ctx0, kv_self.v,
                        n_kv, n_embd_head, n_head_kv,
                        ggml_element_size(kv_self.v)*n_ctx,
                        ggml_element_size(kv_self.v)*n_ctx*n_embd_head,
                        ggml_element_size(kv_self.v)*n_ctx*n_embd_gqa*il);
            offload_func_v(V);
            ggml_set_name(V, "V");

#if 1
            struct ggml_tensor * KQV = ggml_mul_mat(ctx0, V, KQ_soft_max);
            offload_func_v(KQV);
            ggml_set_name(KQV, "KQV");
#else
            // make V contiguous in memory to speed up the matmul, however we waste time on the copy
            // on M1 this is faster for the perplexity computation, but ~5% slower for the single-token generation
            // is there a better way?
            struct ggml_tensor * V_cont = ggml_cpy(ctx0, V, ggml_new_tensor_3d(ctx0, kv_self.v->type, n_ctx, n_embd_head, n_head));
            struct ggml_tensor * KQV = ggml_mul_mat(ctx0, V_cont, KQ_soft_max);
#endif

            // KQV_merged = KQV.permute(0, 2, 1, 3)
            struct ggml_tensor * KQV_merged = ggml_permute(ctx0, KQV, 0, 2, 1, 3);
            offload_func_v(KQV_merged);
            ggml_set_name(KQV_merged, "KQV_merged");

            // cur = KQV_merged.contiguous().view(n_embd, n_tokens)
            cur = ggml_cont_2d(ctx0, KQV_merged, n_embd, n_tokens);
            offload_func_v(cur);
            ggml_set_name(cur, "KQV_merged_contiguous");

            // projection (no bias)
            cur = ggml_mul_mat(ctx0,
                    model.layers[il].wo,
                    cur);
            offload_func(cur);
            ggml_set_name(cur, "result_wo");
        }

        struct ggml_tensor * inpFF = ggml_add(ctx0, cur, inpSA);
        offload_func(inpFF);
        ggml_set_name(inpFF, "inpFF");

        // feed-forward network
        {
            // norm
            {
                cur = ggml_rms_norm(ctx0, inpFF, norm_rms_eps);
                offload_func(cur);
                ggml_set_name(cur, "rms_norm_1");

                // cur = cur*ffn_norm(broadcasted)
                cur = ggml_mul(ctx0, cur, model.layers[il].ffn_norm);
                offload_func(cur);
                ggml_set_name(cur, "ffn_norm");
            }

            struct ggml_tensor * tmp = ggml_mul_mat(ctx0,
                    model.layers[il].w3,
                    cur);
            offload_func(tmp);
            ggml_set_name(tmp, "result_w3");

            cur = ggml_mul_mat(ctx0,
                    model.layers[il].w1,
                    cur);
            offload_func(cur);
            ggml_set_name(cur, "result_w1");

            // SILU activation
            cur = ggml_silu(ctx0, cur);
            offload_func(cur);
            ggml_set_name(cur, "silu");

            cur = ggml_mul(ctx0, cur, tmp);
            offload_func(cur);
            ggml_set_name(cur, "silu_x_result_w3");

            cur = ggml_mul_mat(ctx0,
                    model.layers[il].w2,
                    cur);
            offload_func(cur);
            ggml_set_name(cur, "result_w2");
        }

        cur = ggml_add(ctx0, cur, inpFF);
        offload_func(cur);
        ggml_set_name(cur, "inpFF_+_result_w2");

        // input for next layer
        inpL = cur;
    }

    cur = inpL;

    // norm
    {
        cur = ggml_rms_norm(ctx0, cur, norm_rms_eps);
        offload_func_nr(cur);
        ggml_set_name(cur, "rms_norm_2");

        // cur = cur*norm(broadcasted)
        cur = ggml_mul(ctx0, cur, model.output_norm);
        // offload_func_nr(cur); // TODO CPU + GPU mirrored backend
        ggml_set_name(cur, "result_norm");
    }

    // lm_head
    cur = ggml_mul_mat(ctx0, model.output, cur);
    ggml_set_name(cur, "result_output");

    ggml_build_forward_expand(gf, cur);

    ggml_free(ctx0);

    return gf;
}

static struct ggml_cgraph * llm_build_falcon(
         llama_context & lctx,
     const llama_batch & batch) {
    const auto & model   = lctx.model;
    const auto & hparams = model.hparams;
    const auto & cparams = lctx.cparams;

    const auto & kv_self = lctx.kv_self;

    GGML_ASSERT(!!kv_self.ctx);

    const int64_t n_embd      = hparams.n_embd;
    const int64_t n_layer     = hparams.n_layer;
    const int64_t n_ctx       = cparams.n_ctx;
    const int64_t n_head      = hparams.n_head;
    const int64_t n_head_kv   = hparams.n_head_kv;
    const int64_t n_embd_head = hparams.n_embd_head();
    const int64_t n_embd_gqa  = hparams.n_embd_gqa();

    GGML_ASSERT(n_embd_head == hparams.n_rot);

    const float freq_base  = cparams.rope_freq_base;
    const float freq_scale = cparams.rope_freq_scale;
    const float norm_eps   = hparams.f_norm_eps;

    const int n_gpu_layers = model.n_gpu_layers;

    const int32_t n_tokens = batch.n_tokens;
    const int32_t n_kv     = ggml_allocr_is_measure(lctx.alloc) ? n_ctx            : kv_self.n;
    const int32_t kv_head  = ggml_allocr_is_measure(lctx.alloc) ? n_ctx - n_tokens : kv_self.head;

    const bool do_rope_shift = ggml_allocr_is_measure(lctx.alloc) || kv_self.has_shift;

    //printf("kv_head = %d, n_kv = %d, n_tokens = %d, n_ctx = %d, is_measure = %d, has_shift = %d\n",
    //        kv_head, n_kv, n_tokens, n_ctx, ggml_allocr_is_measure(lctx.alloc), kv_self.has_shift);

    auto & buf_compute = lctx.buf_compute;

    struct ggml_init_params params = {
        /*.mem_size   =*/ buf_compute.size,
        /*.mem_buffer =*/ buf_compute.data,
        /*.no_alloc   =*/ true,
    };

    struct ggml_context * ctx0 = ggml_init(params);

    ggml_cgraph * gf = ggml_new_graph(ctx0);

    struct ggml_tensor * cur;
    struct ggml_tensor * inpL;

    if (batch.token) {
        struct ggml_tensor * inp_tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);

        ggml_allocr_alloc(lctx.alloc, inp_tokens);
        if (!ggml_allocr_is_measure(lctx.alloc)) {
            memcpy(inp_tokens->data, batch.token, n_tokens*ggml_element_size(inp_tokens));
        }
        ggml_set_name(inp_tokens, "inp_tokens");

        inpL = ggml_get_rows(ctx0, model.tok_embeddings, inp_tokens);
    } else {
#ifdef GGML_USE_MPI
        GGML_ASSERT(false && "not implemented");
#endif

        inpL = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, n_tokens);

        ggml_allocr_alloc(lctx.alloc, inpL);
        if (!ggml_allocr_is_measure(lctx.alloc)) {
            memcpy(inpL->data, batch.embd, n_tokens * n_embd * ggml_element_size(inpL));
        }
    }

    const int i_gpu_start = n_layer - n_gpu_layers;
    (void) i_gpu_start;

    // offload functions set the tensor output backend to GPU
    // tensors are GPU-accelerated if any input or the output has been offloaded
    offload_func_t offload_func_nr = llama_nop; // nr = non-repeating
    offload_func_t offload_func_kq = llama_nop;
    offload_func_t offload_func_v  = llama_nop;

#ifdef GGML_USE_CUBLAS
    if (n_gpu_layers > n_layer) {
        offload_func_nr = ggml_cuda_assign_buffers_no_alloc;
    }
    if (n_gpu_layers > n_layer + 1) {
        offload_func_v  = ggml_cuda_assign_buffers_no_alloc;
    }
    if (n_gpu_layers > n_layer + 2) {
        offload_func_kq = ggml_cuda_assign_buffers_no_alloc;
    }
#endif // GGML_USE_CUBLAS

    // KQ_scale
    struct ggml_tensor * KQ_scale = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, 1);
    ggml_set_name(KQ_scale, "1/sqrt(n_embd_head)");
    ggml_allocr_alloc(lctx.alloc, KQ_scale);
    if (!ggml_allocr_is_measure(lctx.alloc)) {
        ggml_set_f32(KQ_scale, 1.0f/sqrtf(float(n_embd)/n_head));
    }

    // KQ_mask (mask for 1 head, it will be broadcasted to all heads)
    struct ggml_tensor * KQ_mask = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, n_kv, n_tokens, 1);
    offload_func_kq(KQ_mask);
    ggml_set_name(KQ_mask, "KQ_mask");
    ggml_allocr_alloc(lctx.alloc, KQ_mask);
    if (!ggml_allocr_is_measure(lctx.alloc)) {
        float * data = (float *) KQ_mask->data;
        memset(data, 0, ggml_nbytes(KQ_mask));

        for (int h = 0; h < 1; ++h) {
            for (int j = 0; j < n_tokens; ++j) {
                const llama_pos    pos    = batch.pos[j];
                const llama_seq_id seq_id = batch.seq_id[j][0];

                for (int i = 0; i < n_kv; ++i) {
                    if (!kv_self.cells[i].has_seq_id(seq_id) || kv_self.cells[i].pos > pos) {
                        data[h*(n_kv*n_tokens) + j*n_kv + i] = -INFINITY;
                    }
                }
            }
        }
    }

    // KQ_pos - contains the positions
    struct ggml_tensor * KQ_pos = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    offload_func_kq(KQ_pos);
    ggml_set_name(KQ_pos, "KQ_pos");
    ggml_allocr_alloc(lctx.alloc, KQ_pos);
    if (!ggml_allocr_is_measure(lctx.alloc)) {
        int * data = (int *) KQ_pos->data;
        for (int i = 0; i < n_tokens; ++i) {
            data[i] = batch.pos[i];
        }
    }

    // shift the entire K-cache if needed
    if (do_rope_shift) {
        struct ggml_tensor * K_shift = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_ctx);
        offload_func_kq(K_shift);
        ggml_set_name(K_shift, "K_shift");
        ggml_allocr_alloc(lctx.alloc, K_shift);
        if (!ggml_allocr_is_measure(lctx.alloc)) {
            int * data = (int *) K_shift->data;
            for (int i = 0; i < n_ctx; ++i) {
                data[i] = kv_self.cells[i].delta;
            }
        }

        for (int il = 0; il < n_layer; ++il) {
            struct ggml_tensor * tmp =
                    ggml_rope_custom_inplace(ctx0,
                        ggml_view_3d(ctx0, kv_self.k,
                            n_embd_head, n_head_kv, n_ctx,
                            ggml_element_size(kv_self.k)*n_embd_head,
                            ggml_element_size(kv_self.k)*n_embd_gqa,
                            ggml_element_size(kv_self.k)*n_embd_gqa*n_ctx*il),
                        K_shift, n_embd_head, 2, 0, freq_base, freq_scale);
            offload_func_kq(tmp);
            ggml_build_forward_expand(gf, tmp);
        }
    }

    for (int il = 0; il < n_layer; ++il) {
        struct ggml_tensor * attn_norm;

        offload_func_t offload_func = llama_nop;

#ifdef GGML_USE_CUBLAS
        if (il >= i_gpu_start) {
            offload_func = ggml_cuda_assign_buffers_no_alloc;
        }
#endif // GGML_USE_CUBLAS

        // self-attention
        // TODO: refactor into common function (shared with LLaMA)
        {
            attn_norm = ggml_norm(ctx0, inpL, norm_eps);
            offload_func(attn_norm);

            attn_norm = ggml_add(ctx0,
                    ggml_mul(ctx0, attn_norm, model.layers[il].attn_norm),
                    model.layers[il].attn_norm_b);
            offload_func(attn_norm->src[0]);
            offload_func(attn_norm);

            if (model.layers[il].attn_norm_2) { // Falcon-40B
                cur = ggml_norm(ctx0, inpL, norm_eps);
                offload_func(cur);

                cur = ggml_add(ctx0,
                        ggml_mul(ctx0, cur, model.layers[il].attn_norm_2),
                        model.layers[il].attn_norm_2_b);
                offload_func(cur->src[0]);
                offload_func(cur);
            } else { // Falcon 7B
                cur = attn_norm;
            }

            // compute QKV

            cur = ggml_mul_mat(ctx0, model.layers[il].wqkv, cur);
            offload_func_kq(cur);

            // Note that the strides for Kcur, Vcur are set up so that the
            // resulting views are misaligned with the tensor's storage
            // (by applying the K/V offset we shift the tensor's original
            // view to stick out behind the viewed QKV tensor's allocated
            // memory, so to say). This is ok because no actual accesses
            // happen to that out-of-range memory, but it can require some
            // trickery when trying to accurately dump these views for
            // debugging.

            const size_t wsize = ggml_type_size(cur->type);

            // TODO: these 2 ggml_conts are technically not needed, but we add them until CUDA support for
            //       non-contiguous views is added for the rope operator
            struct ggml_tensor * tmpq = ggml_cont(ctx0, ggml_view_3d(
                ctx0, cur, n_embd_head, n_head, n_tokens,
                wsize * n_embd_head,
                wsize * n_embd_head * (n_head + 2 * n_head_kv),
                0));
            offload_func_kq(tmpq);

            struct ggml_tensor * tmpk = ggml_cont(ctx0, ggml_view_3d(
                ctx0, cur, n_embd_head, n_head_kv, n_tokens,
                wsize * n_embd_head,
                wsize * n_embd_head * (n_head + 2 * n_head_kv),
                wsize * n_embd_head *  n_head));
            offload_func_kq(tmpk);

            struct ggml_tensor * tmpv = ggml_view_3d(
                ctx0, cur, n_embd_head, n_head_kv, n_tokens,
                wsize * n_embd_head,
                wsize * n_embd_head * (n_head + 2 * n_head_kv),
                wsize * n_embd_head * (n_head +     n_head_kv));
            offload_func_v(tmpv);

            // using mode = 2 for neox mode
            struct ggml_tensor * Qcur = ggml_rope_custom(ctx0, tmpq, KQ_pos, n_embd_head, 2, 0, freq_base, freq_scale);
            offload_func_kq(Qcur);
            struct ggml_tensor * Kcur = ggml_rope_custom(ctx0, tmpk, KQ_pos, n_embd_head, 2, 0, freq_base, freq_scale);
            offload_func_kq(Kcur);

            {
                struct ggml_tensor * Vcur = ggml_transpose(ctx0, ggml_reshape_2d(ctx0, ggml_cont(ctx0, tmpv), n_embd_gqa, n_tokens));
                offload_func_v(Vcur);
                offload_func_v(Vcur->src[0]->src[0]);
                ggml_set_name(Vcur, "Vcur");

                struct ggml_tensor * k = ggml_view_1d(ctx0, kv_self.k, n_tokens*n_embd_gqa, (ggml_element_size(kv_self.k)*n_embd_gqa)*(il*n_ctx + kv_head));
                offload_func_kq(k);
                ggml_set_name(k, "k");

                struct ggml_tensor * v = ggml_view_2d(ctx0, kv_self.v, n_tokens, n_embd_gqa,
                        (   n_ctx)*ggml_element_size(kv_self.v),
                        (il*n_ctx)*ggml_element_size(kv_self.v)*n_embd_gqa + kv_head*ggml_element_size(kv_self.v));
                offload_func_v(v);

                ggml_build_forward_expand(gf, ggml_cpy(ctx0, Kcur, k));
                ggml_build_forward_expand(gf, ggml_cpy(ctx0, Vcur, v));
            }

            struct ggml_tensor * Q = ggml_permute(ctx0, Qcur, 0, 2, 1, 3);
            offload_func_kq(Q);
            ggml_set_name(Q, "Q");

            struct ggml_tensor * K =
                ggml_view_3d(ctx0, kv_self.k,
                        n_embd_head, n_kv, n_head_kv,
                        ggml_element_size(kv_self.k)*n_embd_gqa,
                        ggml_element_size(kv_self.k)*n_embd_head,
                        ggml_element_size(kv_self.k)*n_embd_gqa*n_ctx*il);
            offload_func_kq(K);
            ggml_set_name(K, "K");

            struct ggml_tensor * KQ = ggml_mul_mat(ctx0, K, Q);
            offload_func_kq(KQ);
            ggml_set_name(KQ, "KQ");

            struct ggml_tensor * KQ_scaled = ggml_scale(ctx0, KQ, KQ_scale);
            offload_func_kq(KQ_scaled);
            ggml_set_name(KQ_scaled, "KQ_scaled");

            struct ggml_tensor * KQ_masked = ggml_add(ctx0, KQ_scaled, KQ_mask);
            offload_func_kq(KQ_masked);
            ggml_set_name(KQ_masked, "KQ_masked");

            struct ggml_tensor * KQ_soft_max = ggml_soft_max(ctx0, KQ_masked);
            offload_func_v(KQ_soft_max);
            ggml_set_name(KQ_soft_max, "KQ_soft_max");

            struct ggml_tensor * V =
                ggml_view_3d(ctx0, kv_self.v,
                        n_kv, n_embd_head, n_head_kv,
                        ggml_element_size(kv_self.v)*n_ctx,
                        ggml_element_size(kv_self.v)*n_ctx*n_embd_head,
                        ggml_element_size(kv_self.v)*n_ctx*n_embd_gqa*il);
            offload_func_v(V);
            ggml_set_name(V, "V");

            struct ggml_tensor * KQV = ggml_mul_mat(ctx0, V, KQ_soft_max);
            offload_func_v(KQV);
            ggml_set_name(KQV, "KQV");

            struct ggml_tensor * KQV_merged = ggml_permute(ctx0, KQV, 0, 2, 1, 3);
            offload_func_v(KQV_merged);
            ggml_set_name(KQV_merged, "KQV_merged");

            cur = ggml_cont_2d(ctx0, KQV_merged, n_embd, n_tokens);
            offload_func_v(cur);
            ggml_set_name(cur, "KQV_merged_contiguous");

            cur = ggml_mul_mat(ctx0, model.layers[il].wo, cur);
            offload_func(cur);
            ggml_set_name(cur, "result_wo");
        }

        struct ggml_tensor * attn_out = cur;

        // feed forward
        {
            struct ggml_tensor * inpFF = attn_norm;

            cur = ggml_mul_mat(ctx0, model.layers[il].w3, inpFF);
            offload_func(cur);

            cur = ggml_gelu(ctx0, cur);
            offload_func(cur);
            cur = ggml_mul_mat(ctx0, model.layers[il].w2, cur);
            offload_func(cur);
        }

        cur = ggml_add(ctx0, cur, attn_out);
        offload_func(cur);
        cur = ggml_add(ctx0, cur, inpL);
        offload_func(cur);

        // input for next layer
        inpL = cur;
    }

    cur = inpL;

    // norm
    {
        cur = ggml_norm(ctx0, cur, norm_eps);
        offload_func_nr(cur);

        cur = ggml_add(ctx0,
                ggml_mul(ctx0, cur, model.output_norm),
                model.output_norm_b);
        ggml_set_name(cur, "result_norm");
    }

    cur = ggml_mul_mat(ctx0, model.output, cur);
    ggml_set_name(cur, "result_output");

    ggml_build_forward_expand(gf, cur);

    ggml_free(ctx0);

    return gf;
}

static struct ggml_cgraph * llm_build_starcoder(
         llama_context & lctx,
     const llama_batch & batch) {
    const auto & model   = lctx.model;
    const auto & hparams = model.hparams;
    const auto & cparams = lctx.cparams;

    const auto & kv_self = lctx.kv_self;

    GGML_ASSERT(!!kv_self.ctx);

    const int64_t n_embd      = hparams.n_embd;
    const int64_t n_layer     = hparams.n_layer;
    const int64_t n_ctx       = cparams.n_ctx;
    const int64_t n_head      = hparams.n_head;
    const int64_t n_head_kv   = hparams.n_head_kv;
    const int64_t n_embd_head = hparams.n_embd_head();
    const int64_t n_embd_gqa  = hparams.n_embd_gqa();

    GGML_ASSERT(n_embd_head == hparams.n_rot);

    const float norm_eps = hparams.f_norm_eps;

    const int32_t n_tokens = batch.n_tokens;
    const int32_t n_kv     = ggml_allocr_is_measure(lctx.alloc) ? n_ctx            : kv_self.n;
    const int32_t kv_head  = ggml_allocr_is_measure(lctx.alloc) ? n_ctx - n_tokens : kv_self.head;

    auto & buf_compute = lctx.buf_compute;

    struct ggml_init_params params = {
        /*.mem_size   =*/ buf_compute.size,
        /*.mem_buffer =*/ buf_compute.data,
        /*.no_alloc   =*/ true,
    };

    struct ggml_context * ctx0 = ggml_init(params);

    ggml_cgraph * gf = ggml_new_graph(ctx0);

    struct ggml_tensor * cur;
    struct ggml_tensor * token;
    struct ggml_tensor * position;
    struct ggml_tensor * inpL;

    if (batch.token) {
        struct ggml_tensor * inp_tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);

        ggml_allocr_alloc(lctx.alloc, inp_tokens);
        if (!ggml_allocr_is_measure(lctx.alloc)) {
            memcpy(inp_tokens->data, batch.token, n_tokens*ggml_element_size(inp_tokens));
        }
        ggml_set_name(inp_tokens, "inp_tokens");

        token = ggml_get_rows(ctx0, model.tok_embeddings, inp_tokens);
    } else {
#ifdef GGML_USE_MPI
        GGML_ASSERT(false && "not implemented");
#endif

        token = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, n_tokens);

        ggml_allocr_alloc(lctx.alloc, token);
        if (!ggml_allocr_is_measure(lctx.alloc)) {
            memcpy(token->data, batch.embd, n_tokens * n_embd * ggml_element_size(token));
        }
    }

    {
        // Compute position embeddings.
        struct ggml_tensor * inp_positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
        ggml_allocr_alloc(lctx.alloc, inp_positions);
        if (!ggml_allocr_is_measure(lctx.alloc)) {
            for (int i = 0; i < n_tokens; ++i) {
                ((int32_t *) inp_positions->data)[i] = batch.pos[i];
            }
        }
        ggml_set_name(inp_positions, "inp_positions");

        position = ggml_get_rows(ctx0, model.pos_embeddings, inp_positions);
    }

    // KQ_scale
    struct ggml_tensor * KQ_scale = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, 1);
    ggml_set_name(KQ_scale, "1/sqrt(n_embd_head)");
    ggml_allocr_alloc(lctx.alloc, KQ_scale);
    if (!ggml_allocr_is_measure(lctx.alloc)) {
        ggml_set_f32(KQ_scale, 1.0f/sqrtf(float(n_embd)/n_head));
    }

    // KQ_mask (mask for 1 head, it will be broadcasted to all heads)
    struct ggml_tensor * KQ_mask = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, n_kv, n_tokens, 1);
    ggml_set_name(KQ_mask, "KQ_mask");
    ggml_allocr_alloc(lctx.alloc, KQ_mask);
    if (!ggml_allocr_is_measure(lctx.alloc)) {
        float * data = (float *) KQ_mask->data;
        memset(data, 0, ggml_nbytes(KQ_mask));

        for (int h = 0; h < 1; ++h) {
            for (int j = 0; j < n_tokens; ++j) {
                const llama_pos    pos    = batch.pos[j];
                const llama_seq_id seq_id = batch.seq_id[j][0];

                for (int i = 0; i < n_kv; ++i) {
                    if (!kv_self.cells[i].has_seq_id(seq_id) || kv_self.cells[i].pos > pos) {
                        data[h*(n_kv*n_tokens) + j*n_kv + i] = -INFINITY;
                    }
                }
            }
        }
    }

    inpL = ggml_add(ctx0, token, position);
    ggml_set_name(inpL, "inpL");

    for (int il = 0; il < n_layer; ++il) {
        {
            // Norm
            cur = ggml_norm(ctx0, inpL, norm_eps);
            cur = ggml_add(ctx0, ggml_mul(ctx0, cur, model.layers[il].attn_norm), model.layers[il].attn_norm_b);
        }

        {
            // Self Attention
            cur = ggml_add(ctx0, ggml_mul_mat(ctx0, model.layers[il].wqkv, cur), model.layers[il].bqkv);

            struct ggml_tensor * tmpq = ggml_view_2d(ctx0, cur, n_embd, n_tokens, cur->nb[1], 0*sizeof(float)*n_embd);
            struct ggml_tensor * tmpk = ggml_view_2d(ctx0, cur, n_embd_gqa, n_tokens, cur->nb[1], sizeof(float)*n_embd);
            struct ggml_tensor * tmpv = ggml_view_2d(ctx0, cur, n_embd_gqa, n_tokens, cur->nb[1], sizeof(float)*(n_embd + n_embd_gqa));

            struct ggml_tensor * Qcur = tmpq;
            struct ggml_tensor * Kcur = tmpk;

            {
                struct ggml_tensor * Vcur = ggml_transpose(ctx0, ggml_reshape_2d(ctx0, ggml_cont(ctx0, tmpv), n_embd_gqa, n_tokens));
                ggml_set_name(Vcur, "Vcur");

                struct ggml_tensor * k = ggml_view_1d(ctx0, kv_self.k, n_tokens*n_embd_gqa, (ggml_element_size(kv_self.k)*n_embd_gqa)*(il*n_ctx + kv_head));
                ggml_set_name(k, "k");

                struct ggml_tensor * v = ggml_view_2d(ctx0, kv_self.v, n_tokens, n_embd_gqa,
                        (   n_ctx)*ggml_element_size(kv_self.v),
                        (il*n_ctx)*ggml_element_size(kv_self.v)*n_embd_gqa + kv_head*ggml_element_size(kv_self.v));

                ggml_build_forward_expand(gf, ggml_cpy(ctx0, Kcur, k));
                ggml_build_forward_expand(gf, ggml_cpy(ctx0, Vcur, v));
            }

            struct ggml_tensor * Q =
                ggml_permute(ctx0,
                        ggml_cpy(ctx0,
                            Qcur,
                            ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, n_embd_head, n_head, n_tokens)),
                        0, 2, 1, 3);
            ggml_set_name(Q, "Q");

            struct ggml_tensor * K =
                ggml_view_3d(ctx0, kv_self.k,
                        n_embd_head, n_kv, n_head_kv,
                        ggml_element_size(kv_self.k)*n_embd_gqa,
                        ggml_element_size(kv_self.k)*n_embd_head,
                        ggml_element_size(kv_self.k)*n_embd_gqa*n_ctx*il);
            ggml_set_name(K, "K");

            // K * Q
            struct ggml_tensor * KQ = ggml_mul_mat(ctx0, K, Q);
            ggml_set_name(KQ, "KQ");

            // KQ_scaled = KQ / sqrt(n_embd_head)
            // KQ_scaled shape [n_past + n_tokens, n_tokens, n_head, 1]
            struct ggml_tensor * KQ_scaled = ggml_scale_inplace(ctx0, KQ, KQ_scale);
            ggml_set_name(KQ_scaled, "KQ_scaled");

            // KQ_masked = mask_past(KQ_scaled)
            struct ggml_tensor * KQ_masked = ggml_add(ctx0, KQ_scaled, KQ_mask);
            ggml_set_name(KQ_masked, "KQ_masked");

            // KQ = soft_max(KQ_masked)
            struct ggml_tensor * KQ_soft_max = ggml_soft_max_inplace(ctx0, KQ_masked);
            ggml_set_name(KQ_soft_max, "KQ_soft_max");

            // split cached V into n_head heads
            struct ggml_tensor * V =
                ggml_view_3d(ctx0, kv_self.v,
                        n_kv, n_embd_head, n_head_kv,
                        ggml_element_size(kv_self.v)*n_ctx,
                        ggml_element_size(kv_self.v)*n_ctx*n_embd_head,
                        ggml_element_size(kv_self.v)*n_ctx*n_embd_gqa*il);
            ggml_set_name(V, "V");

            struct ggml_tensor * KQV = ggml_mul_mat(ctx0, V, KQ_soft_max);
            ggml_set_name(KQV, "KQV");

            // KQV_merged = KQV.permute(0, 2, 1, 3)
            struct ggml_tensor * KQV_merged = ggml_permute(ctx0, KQV, 0, 2, 1, 3);
            ggml_set_name(KQV_merged, "KQV_merged");

            // cur = KQV_merged.contiguous().view(n_embd, n_tokens)
            cur = ggml_cont_2d(ctx0, KQV_merged, n_embd, n_tokens);
            ggml_set_name(cur, "KQV_merged_contiguous");
        }

        // Projection
        cur = ggml_add(ctx0, ggml_mul_mat(ctx0, model.layers[il].wo, cur), model.layers[il].bo);

        // Add the input
        cur = ggml_add(ctx0, cur, inpL);

        struct ggml_tensor * inpFF = cur;

        // FF
        {
            // Norm
            {
                cur = ggml_norm(ctx0, inpFF, norm_eps);
                cur = ggml_add(ctx0, ggml_mul(ctx0, cur, model.layers[il].ffn_norm), model.layers[il].ffn_norm_b);
            }

            cur = ggml_add(ctx0, ggml_mul_mat(ctx0, model.layers[il].w3, cur), model.layers[il].b3);

            // GELU activation
            cur = ggml_gelu(ctx0, cur);

            // Projection
            cur = ggml_add(ctx0, ggml_mul_mat(ctx0, model.layers[il].w2, cur), model.layers[il].b2);
        }

        inpL = ggml_add(ctx0, cur, inpFF);
    }

    // Output Norm
    {
        cur = ggml_norm(ctx0, inpL, norm_eps);
        cur = ggml_add(ctx0, ggml_mul(ctx0, cur, model.output_norm), model.output_norm_b);
    }
    ggml_set_name(cur, "result_norm");

    cur = ggml_mul_mat(ctx0, model.output, cur);
    ggml_set_name(cur, "result_output");

    ggml_build_forward_expand(gf, cur);
    ggml_free(ctx0);

    return gf;
}

static struct ggml_cgraph * llm_build_persimmon(
         llama_context & lctx,
     const llama_batch & batch) {
    const auto & model = lctx.model;
    const auto & hparams = model.hparams;

    const auto & kv_self = lctx.kv_self;

    GGML_ASSERT(!!kv_self.ctx);

    const auto & cparams = lctx.cparams;
    const int64_t n_embd      = hparams.n_embd;
    const int64_t n_layer     = hparams.n_layer;
    const int64_t n_ctx       = cparams.n_ctx;
    const int64_t n_head_kv   = hparams.n_head_kv;
    const int64_t n_head      = hparams.n_head;
    const int64_t n_embd_head = hparams.n_embd_head();
    const int64_t n_embd_gqa  = hparams.n_embd_gqa();
    const size_t n_rot        = n_embd_head / 2;

    const float freq_base  = cparams.rope_freq_base;
    const float freq_scale = cparams.rope_freq_scale;
    const float norm_eps = hparams.f_norm_eps;

    const int n_gpu_layers = model.n_gpu_layers;


    const int32_t n_tokens    = batch.n_tokens;
    const int32_t n_kv        = ggml_allocr_is_measure(lctx.alloc) ? n_ctx            : kv_self.n;
    const int32_t kv_head     = ggml_allocr_is_measure(lctx.alloc) ? n_ctx - n_tokens : kv_self.head;

    const bool do_rope_shift  = ggml_allocr_is_measure(lctx.alloc) || kv_self.has_shift;

    auto & buf_compute = lctx.buf_compute;
    struct ggml_init_params params = {
        /*.mem_size   =*/ buf_compute.size,
        /*.mem_buffer =*/ buf_compute.data,
        /*.no_alloc   =*/ true,
    };

    struct ggml_context * ctx0 = ggml_init(params);

    ggml_cgraph * gf = ggml_new_graph(ctx0);

    struct ggml_tensor * cur;
    struct ggml_tensor * inpL;

    if (batch.token) {
        struct ggml_tensor * inp_tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);

        ggml_allocr_alloc(lctx.alloc, inp_tokens);
        if (!ggml_allocr_is_measure(lctx.alloc)) {
            memcpy(inp_tokens->data, batch.token, n_tokens*ggml_element_size(inp_tokens));
        }
        ggml_set_name(inp_tokens, "inp_tokens");
        inpL = ggml_get_rows(ctx0, model.tok_embeddings, inp_tokens);
    } else {
        inpL = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, n_tokens);
        ggml_allocr_alloc(lctx.alloc, inpL);
        if (!ggml_allocr_is_measure(lctx.alloc)) {
            memcpy(inpL->data, batch.embd, n_tokens * n_embd * ggml_element_size(inpL));
        }
    }
    const int i_gpu_start = n_layer - n_gpu_layers;
    (void) i_gpu_start;
    offload_func_t offload_func_nr = llama_nop; // nr = non-repeating
    offload_func_t offload_func_kq = llama_nop;
    offload_func_t offload_func_v  = llama_nop;
    // KQ_scale
    struct ggml_tensor * KQ_scale = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, 1);
    ggml_allocr_alloc(lctx.alloc, KQ_scale);
    if (!ggml_allocr_is_measure(lctx.alloc)) {
        ggml_set_f32(KQ_scale, 1.0f/sqrtf(float(n_embd_head)));
    }
    ggml_set_name(KQ_scale, "1/sqrt(n_embd_head)");
    struct ggml_tensor * KQ_mask = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, n_kv, n_tokens, 1);
    offload_func_kq(KQ_mask);
    ggml_set_name(KQ_mask, "KQ_mask");
    ggml_allocr_alloc(lctx.alloc, KQ_mask);

    if (!ggml_allocr_is_measure(lctx.alloc)) {
        float * data = (float *) KQ_mask->data;
        memset(data, 0, ggml_nbytes(KQ_mask));
        for (int h = 0; h < 1; ++h) {
            for (int j = 0; j < n_tokens; ++j) {
                const llama_pos    pos    = batch.pos[j];
                const llama_seq_id seq_id = batch.seq_id[j][0];
                for (int i = 0; i < n_kv; ++i) {
                    if (!kv_self.cells[i].has_seq_id(seq_id) || kv_self.cells[i].pos > pos) {
                        data[h*(n_kv*n_tokens) + j*n_kv + i] = -INFINITY;
                    }
                }
            }
        }
    }

    struct ggml_tensor * KQ_pos = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    offload_func_kq(KQ_pos);
    ggml_set_name(KQ_pos, "KQ_pos");
    ggml_allocr_alloc(lctx.alloc, KQ_pos);
    if (!ggml_allocr_is_measure(lctx.alloc)) {
        int * data = (int *) KQ_pos->data;
        for (int i = 0; i < n_tokens; ++i) {
            data[i] = batch.pos[i];
        }
    }
    if (do_rope_shift) {
        struct ggml_tensor * K_shift = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_ctx);
        offload_func_kq(K_shift);
        ggml_set_name(K_shift, "K_shift");
        ggml_allocr_alloc(lctx.alloc, K_shift);
        if (!ggml_allocr_is_measure(lctx.alloc)) {
            int * data = (int *) K_shift->data;
            for (int i = 0; i < n_ctx; ++i) {
                data[i] = kv_self.cells[i].delta;
            }
        }
        for (int il = 0; il < n_layer; ++il) {
            struct ggml_tensor * tmp =
                    // we rotate only the first n_rot dimensions.
                    ggml_rope_custom_inplace(ctx0,
                        ggml_view_3d(ctx0, kv_self.k,
                            n_rot, n_head, n_ctx,
                            ggml_element_size(kv_self.k)*n_embd_gqa,
                            ggml_element_size(kv_self.k)*n_embd_head,
                            ggml_element_size(kv_self.k)*(n_embd_head*n_ctx*il)
                        ),
                        K_shift, n_rot, 2, 0, freq_base, freq_scale);
            offload_func_kq(tmp);
            ggml_build_forward_expand(gf, tmp);
        }
    }
    for (int il=0; il < n_layer; ++il) {
        struct ggml_tensor * residual = inpL;
        offload_func_t offload_func = llama_nop;
        {
            cur = ggml_norm(ctx0, inpL, norm_eps);
            offload_func(cur);
            cur = ggml_mul(ctx0, cur, model.layers[il].attn_norm);
            offload_func(cur);
            cur = ggml_add(ctx0, cur, model.layers[il].attn_norm_b);
            offload_func(cur);
            ggml_format_name(cur, "input_layernorm_%d", il);
        }
        // self attention
        {
            cur = ggml_mul_mat(ctx0, model.layers[il].wqkv, cur);
            offload_func_kq(cur);
            cur = ggml_add(ctx0, cur, model.layers[il].bqkv);
            offload_func_kq(cur);

            // split qkv
            GGML_ASSERT(n_head_kv == n_head);
            ggml_set_name(cur, format("qkv_%d", il).c_str());
            struct ggml_tensor * tmpqkv = ggml_reshape_4d(ctx0, cur, n_embd_head, 3, n_head, n_tokens);
            offload_func_kq(tmpqkv);
            struct ggml_tensor * tmpqkv_perm = ggml_cont(ctx0, ggml_permute(ctx0, tmpqkv, 0, 3, 1, 2));
            offload_func_kq(tmpqkv_perm);
            ggml_format_name(tmpqkv_perm, "tmpqkv_perm_%d", il);
            struct ggml_tensor * tmpq = ggml_view_3d(
                    ctx0, tmpqkv_perm, n_embd_head, n_head, n_tokens,
                    ggml_element_size(tmpqkv_perm) * n_embd_head,
                    ggml_element_size(tmpqkv_perm) * n_embd_head * n_head,
                    0
                );
            offload_func_kq(tmpq);
            struct ggml_tensor * tmpk = ggml_view_3d(
                    ctx0, tmpqkv_perm, n_embd_head, n_head, n_tokens,
                    ggml_element_size(tmpqkv_perm) * n_embd_head,
                    ggml_element_size(tmpqkv_perm) * n_embd_head * n_head,
                    ggml_element_size(tmpqkv_perm) * n_embd_head * n_head * n_tokens
                );
            offload_func_kq(tmpk);
            // Q/K Layernorm
            tmpq = ggml_norm(ctx0, tmpq, norm_eps);
            offload_func_kq(tmpq);
            tmpq =  ggml_mul(ctx0, tmpq, model.layers[il].attn_q_norm);
            offload_func_kq(tmpq);
            tmpq =  ggml_add(ctx0, tmpq, model.layers[il].attn_q_norm_b);
            offload_func_kq(tmpq);

            tmpk = ggml_norm(ctx0, tmpk, norm_eps);
            offload_func_v(tmpk);
            tmpk =  ggml_mul(ctx0, tmpk, model.layers[il].attn_k_norm);
            offload_func_v(tmpk);
            tmpk =  ggml_add(ctx0, tmpk, model.layers[il].attn_k_norm_b);
            offload_func_v(tmpk);

            // RoPE the first n_rot of q/k, pass the other half, and concat.
            struct ggml_tensor * qrot = ggml_view_3d(
                ctx0, tmpq, n_rot, n_head, n_tokens,
                ggml_element_size(tmpq) * n_embd_head,
                ggml_element_size(tmpq) * n_embd_head * n_head,
                0
            );
            offload_func_kq(qrot);
            ggml_format_name(qrot, "qrot_%d", il);
            struct ggml_tensor * krot = ggml_view_3d(
                ctx0, tmpk, n_rot, n_head, n_tokens,
                ggml_element_size(tmpk) * n_embd_head,
                ggml_element_size(tmpk) * n_embd_head * n_head,
                0
            );
            offload_func_kq(krot);
            ggml_format_name(krot, "krot_%d", il);

            // get the second half of tmpq, e.g tmpq[n_rot:, :, :]
            struct ggml_tensor * qpass = ggml_view_3d(
                ctx0, tmpq, n_rot, n_head, n_tokens,
                ggml_element_size(tmpq) * n_embd_head,
                ggml_element_size(tmpq) * n_embd_head * n_head,
                ggml_element_size(tmpq) * n_rot
            );
            offload_func_kq(qpass);
            ggml_format_name(qpass, "qpass_%d", il);
            struct ggml_tensor * kpass = ggml_view_3d(
                ctx0, tmpk, n_rot, n_head, n_tokens,
                ggml_element_size(tmpk) * n_embd_head,
                ggml_element_size(tmpk) * n_embd_head * n_head,
                ggml_element_size(tmpk) * n_rot
            );
            offload_func_kq(kpass);
            ggml_format_name(kpass, "kpass_%d", il);

            struct ggml_tensor * qrotated =  ggml_rope_custom(
                    ctx0, qrot, KQ_pos, n_rot, 2, 0, freq_base, freq_scale
            );
            offload_func_kq(qrotated);
            struct ggml_tensor * krotated = ggml_rope_custom(
                    ctx0, krot, KQ_pos, n_rot, 2, 0, freq_base, freq_scale
            );
            offload_func_kq(krotated);
            // ggml currently only supports concatenation on dim=2
            // so we need to permute qrot, qpass, concat, then permute back.
            qrotated = ggml_cont(ctx0, ggml_permute(ctx0, qrotated, 2, 1, 0, 3));
            offload_func_kq(qrotated);
            krotated = ggml_cont(ctx0, ggml_permute(ctx0, krotated, 2, 1, 0, 3));
            offload_func_kq(krotated);

            qpass = ggml_cont(ctx0, ggml_permute(ctx0, qpass, 2, 1, 0, 3));
            offload_func_kq(qpass);
            kpass = ggml_cont(ctx0, ggml_permute(ctx0, kpass, 2, 1, 0, 3));
            offload_func_kq(kpass);

            struct ggml_tensor * Qcur = ggml_concat(ctx0, qrotated, qpass);
            offload_func_kq(Qcur);
            struct ggml_tensor * Kcur = ggml_concat(ctx0, krotated, kpass);
            offload_func_kq(Kcur);

            struct ggml_tensor * Q = ggml_cont(ctx0, ggml_permute(ctx0, Qcur, 1, 2, 0, 3));
            offload_func_kq(Q);

            Kcur = ggml_cont(ctx0, ggml_permute(ctx0, Kcur, 2, 1, 0, 3));
            offload_func_kq(Kcur);
            {
                struct ggml_tensor * tmpv = ggml_view_3d(
                        ctx0, tmpqkv_perm, n_embd_head, n_head, n_tokens,
                        ggml_element_size(tmpqkv_perm) * n_embd_head,
                        ggml_element_size(tmpqkv_perm) * n_embd_head * n_head,
                        ggml_element_size(tmpqkv_perm) * n_embd_head * n_head * n_tokens * 2
                    );
                offload_func_v(tmpv);
                // store K, V in cache
                struct ggml_tensor * Vcur = ggml_transpose(ctx0, ggml_reshape_2d(ctx0, tmpv, n_embd_gqa, n_tokens));
                offload_func_v(Vcur);
                ggml_set_name(Vcur, "Vcur");

                struct ggml_tensor * k = ggml_view_1d(
                    ctx0, kv_self.k, n_tokens*n_embd_gqa,
                    (ggml_element_size(kv_self.k)*n_embd_gqa)*(il*n_ctx + kv_head)
                );
                offload_func_kq(k);
                ggml_set_name(k, "k");

                struct ggml_tensor * v = ggml_view_2d(ctx0, kv_self.v, n_tokens, n_embd_gqa,
                        (   n_ctx)*ggml_element_size(kv_self.v),
                        (il*n_ctx)*ggml_element_size(kv_self.v)*n_embd_gqa + kv_head*ggml_element_size(kv_self.v));
                offload_func_v(v);
                ggml_set_name(v, "v");

                // important: storing RoPE-ed version of K in the KV cache!
                ggml_build_forward_expand(gf, ggml_cpy(ctx0, Kcur, k));
                ggml_build_forward_expand(gf, ggml_cpy(ctx0, Vcur, v));
            }
            struct ggml_tensor * K = ggml_view_3d(ctx0, kv_self.k,
                    n_embd_head, n_kv, n_head_kv,
                    ggml_element_size(kv_self.k)*n_embd_gqa,
                    ggml_element_size(kv_self.k)*n_embd_head,
                    ggml_element_size(kv_self.k)*n_embd_gqa*n_ctx*il);

            offload_func_kq(K);
            ggml_format_name(K, "K_%d", il);

            struct ggml_tensor * KQ = ggml_mul_mat(ctx0, K, Q);
            offload_func_kq(KQ);
            ggml_set_name(KQ, "KQ");

            struct ggml_tensor * KQ_scaled = ggml_scale(ctx0, KQ, KQ_scale);
            offload_func_kq(KQ_scaled);
            ggml_set_name(KQ_scaled, "KQ_scaled");

            struct ggml_tensor * KQ_masked = ggml_add(ctx0, KQ_scaled, KQ_mask);
            offload_func_kq(KQ_masked);
            ggml_set_name(KQ_masked, "KQ_masked");

            struct ggml_tensor * KQ_soft_max = ggml_soft_max_inplace(ctx0, KQ_masked);
            offload_func_kq(KQ_soft_max);
            ggml_set_name(KQ_soft_max, "KQ_soft_max");

            struct ggml_tensor * V =
                ggml_view_3d(ctx0, kv_self.v,
                        n_kv, n_embd_head, n_head_kv,
                        ggml_element_size(kv_self.v)*n_ctx,
                        ggml_element_size(kv_self.v)*n_ctx*n_embd_head,
                        ggml_element_size(kv_self.v)*n_ctx*n_embd_gqa*il);
            offload_func_v(V);
            ggml_set_name(V, "V");

            struct ggml_tensor * KQV = ggml_mul_mat(ctx0, V, KQ_soft_max);
            offload_func_v(KQV);
            ggml_set_name(KQV, "KQV");

            struct ggml_tensor * KQV_merged = ggml_permute(ctx0, KQV, 0, 2, 1, 3);
            offload_func_v(KQV_merged);
            ggml_set_name(KQV_merged, "KQV_merged");

            cur = ggml_cont_2d(ctx0, KQV_merged, n_embd, n_tokens);
            offload_func_v(cur);
            ggml_set_name(cur, "KQV_merged_contiguous");

            cur = ggml_mul_mat(ctx0, model.layers[il].wo, cur);
            offload_func(cur);
            cur = ggml_add(ctx0, cur, model.layers[il].bo);
            offload_func(cur);
            ggml_set_name(cur, "result_wo");
        }

        struct ggml_tensor * inpFF = ggml_add(ctx0, residual, cur);
        offload_func(inpFF);
        ggml_set_name(inpFF, "inpFF");
        {
            // MLP
            {
                // Norm
                cur = ggml_norm(ctx0, inpFF, norm_eps);
                offload_func(cur);
                cur = ggml_add(ctx0,
                    ggml_mul(ctx0, cur, model.layers[il].ffn_norm),
                    model.layers[il].ffn_norm_b
                );
                ggml_set_name(cur, "ffn_norm");
                offload_func(cur);
            }
            cur = ggml_mul_mat(ctx0, model.layers[il].w3, cur);
            offload_func(cur);

            cur = ggml_add(ctx0, cur, model.layers[il].b3);
            offload_func(cur);
            ggml_set_name(cur, "result_ffn_up");

            cur = ggml_sqr(ctx0, ggml_relu(ctx0, cur));
            ggml_set_name(cur, "result_ffn_act");
            offload_func(cur);
            offload_func(cur->src[0]);

            cur = ggml_mul_mat(ctx0, model.layers[il].w2, cur);
            offload_func(cur);
            cur = ggml_add(ctx0,
                cur,
                model.layers[il].b2);
            offload_func(cur);
            ggml_set_name(cur, "outFF");
        }
        cur = ggml_add(ctx0, cur, inpFF);
        offload_func(cur);
        ggml_set_name(cur, "inpFF_+_outFF");
        inpL = cur;
    }
    cur = inpL;
    {
        cur = ggml_norm(ctx0, cur, norm_eps);
        offload_func_nr(cur);
        cur = ggml_mul(ctx0, cur, model.output_norm);
        offload_func_nr(cur);

        cur = ggml_add(ctx0, cur, model.output_norm_b);
        // offload_func_nr(cur);

        ggml_set_name(cur, "result_norm");
    }
    cur = ggml_mul_mat(ctx0, model.output, cur);
    ggml_set_name(cur, "result_output");
    ggml_build_forward_expand(gf, cur);
    ggml_free(ctx0);
    return gf;
}

static struct ggml_cgraph * llm_build_bloom(
         llama_context & lctx,
     const llama_batch & batch) {
    const auto & model   = lctx.model;
    const auto & hparams = model.hparams;
    const auto & cparams = lctx.cparams;

    const auto & kv_self = lctx.kv_self;

    GGML_ASSERT(!!kv_self.ctx);

    const int64_t n_embd      = hparams.n_embd;
    const int64_t n_layer     = hparams.n_layer;
    const int64_t n_ctx       = cparams.n_ctx;
    const int64_t n_head      = hparams.n_head;
    const int64_t n_head_kv   = hparams.n_head_kv;
    const int64_t n_embd_head = hparams.n_embd_head();
    const int64_t n_embd_gqa  = hparams.n_embd_gqa();

    GGML_ASSERT(n_embd_head == hparams.n_rot);

    const float norm_eps = hparams.f_norm_eps;

    const int32_t n_tokens = batch.n_tokens;
    const int32_t n_kv     = ggml_allocr_is_measure(lctx.alloc) ? n_ctx            : kv_self.n;
    const int32_t kv_head  = ggml_allocr_is_measure(lctx.alloc) ? n_ctx - n_tokens : kv_self.head;

    auto & buf_compute = lctx.buf_compute;

    struct ggml_init_params params = {
        /*.mem_size   =*/ buf_compute.size,
        /*.mem_buffer =*/ buf_compute.data,
        /*.no_alloc   =*/ false,
    };

    params.no_alloc = true;

    struct ggml_context * ctx0 = ggml_init(params);

    ggml_cgraph * gf = ggml_new_graph(ctx0);

    struct ggml_tensor * cur;
    struct ggml_tensor * token;
    struct ggml_tensor * inpL;

    if (batch.token) {
        struct ggml_tensor * inp_tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);

        ggml_allocr_alloc(lctx.alloc, inp_tokens);
        if (!ggml_allocr_is_measure(lctx.alloc)) {
            memcpy(inp_tokens->data, batch.token, n_tokens*ggml_element_size(inp_tokens));
        }
        ggml_set_name(inp_tokens, "inp_tokens");

        token = ggml_get_rows(ctx0, model.tok_embeddings, inp_tokens);
    } else {
#ifdef GGML_USE_MPI
        GGML_ASSERT(false && "not implemented");
#endif

        token = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, n_tokens);

        ggml_allocr_alloc(lctx.alloc, token);
        if (!ggml_allocr_is_measure(lctx.alloc)) {
            memcpy(token->data, batch.embd, n_tokens * n_embd * ggml_element_size(token));
        }
    }

    // KQ_scale
    struct ggml_tensor * KQ_scale = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, 1);
    ggml_set_name(KQ_scale, "1/sqrt(n_embd_head)");
    ggml_allocr_alloc(lctx.alloc, KQ_scale);
    if (!ggml_allocr_is_measure(lctx.alloc)) {
        ggml_set_f32(KQ_scale, 1.0f/sqrtf(float(n_embd)/n_head));
    }

    // KQ_mask (mask for 1 head, it will be broadcasted to all heads)
    struct ggml_tensor * KQ_mask = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, n_kv, n_tokens, 1);
    ggml_set_name(KQ_mask, "KQ_mask");
    ggml_allocr_alloc(lctx.alloc, KQ_mask);
    if (!ggml_allocr_is_measure(lctx.alloc)) {
        float * data = (float *) KQ_mask->data;
        memset(data, 0, ggml_nbytes(KQ_mask));

        for (int h = 0; h < 1; ++h) {
            for (int j = 0; j < n_tokens; ++j) {
                const llama_pos    pos    = batch.pos[j];
                const llama_seq_id seq_id = batch.seq_id[j][0];

                for (int i = 0; i < n_kv; ++i) {
                    if (!kv_self.cells[i].has_seq_id(seq_id) || kv_self.cells[i].pos > pos) {
                        data[h*(n_kv*n_tokens) + j*n_kv + i] = -INFINITY;
                    }
                }
            }
        }
    }

    // norm
    {
        inpL = ggml_norm(ctx0, token, norm_eps);
        inpL = ggml_add(ctx0, ggml_mul(ctx0, inpL, model.tok_norm), model.tok_norm_b);
    }

    ggml_set_name(inpL, "inpL");

    for (int il = 0; il < n_layer; ++il) {
        {
            // Norm
            cur = ggml_norm(ctx0, inpL, norm_eps);
            cur = ggml_add(ctx0, ggml_mul(ctx0, cur, model.layers[il].attn_norm), model.layers[il].attn_norm_b);
        }

        {
            // Self Attention
            cur = ggml_add(ctx0, ggml_mul_mat(ctx0, model.layers[il].wqkv, cur), model.layers[il].bqkv);

            struct ggml_tensor * tmpq = ggml_view_2d(ctx0, cur, n_embd, n_tokens, cur->nb[1], 0*sizeof(float)*n_embd);
            struct ggml_tensor * tmpk = ggml_view_2d(ctx0, cur, n_embd_gqa, n_tokens, cur->nb[1], sizeof(float)*n_embd);
            struct ggml_tensor * tmpv = ggml_view_2d(ctx0, cur, n_embd_gqa, n_tokens, cur->nb[1], sizeof(float)*(n_embd + n_embd_gqa));

            struct ggml_tensor * Qcur = tmpq;
            struct ggml_tensor * Kcur = tmpk;

            // store key and value to memory
            {
                struct ggml_tensor * Vcur = ggml_transpose(ctx0, ggml_reshape_2d(ctx0, ggml_cont(ctx0, tmpv), n_embd_gqa, n_tokens));
                ggml_set_name(Vcur, "Vcur");

                struct ggml_tensor * k = ggml_view_1d(ctx0, kv_self.k, n_tokens*n_embd_gqa, (ggml_element_size(kv_self.k)*n_embd_gqa)*(il*n_ctx + kv_head));
                ggml_set_name(k, "k");

                struct ggml_tensor * v = ggml_view_2d(ctx0, kv_self.v, n_tokens, n_embd_gqa,
                        (   n_ctx)*ggml_element_size(kv_self.v),
                        (il*n_ctx)*ggml_element_size(kv_self.v)*n_embd_gqa + kv_head*ggml_element_size(kv_self.v));

                ggml_build_forward_expand(gf, ggml_cpy(ctx0, Kcur, k));
                ggml_build_forward_expand(gf, ggml_cpy(ctx0, Vcur, v));
            }

            struct ggml_tensor * Q =
                ggml_permute(ctx0,
                        ggml_cpy(ctx0,
                            Qcur,
                            ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, n_embd_head, n_head, n_tokens)),
                        0, 2, 1, 3);
            ggml_set_name(Q, "Q");

            struct ggml_tensor * K =
                ggml_view_3d(ctx0, kv_self.k,
                        n_embd_head, n_kv, n_head_kv,
                        ggml_element_size(kv_self.k)*n_embd_gqa,
                        ggml_element_size(kv_self.k)*n_embd_head,
                        ggml_element_size(kv_self.k)*n_embd_gqa*n_ctx*il);
            ggml_set_name(K, "K");

            // K * Q
            struct ggml_tensor * KQ = ggml_mul_mat(ctx0, K, Q);
            ggml_set_name(KQ, "KQ");

            // KQ_scaled = KQ / sqrt(n_embd_head)
            // KQ_scaled shape [n_past + n_tokens, n_tokens, n_head, 1]
            struct ggml_tensor * KQ_scaled = ggml_scale_inplace(ctx0, KQ, KQ_scale);
            ggml_set_name(KQ_scaled, "KQ_scaled");

            struct ggml_tensor * KQ_scaled_alibi = ggml_alibi(ctx0, KQ_scaled, /*n_past*/ kv_head, n_head, 8);
            ggml_set_name(KQ_scaled_alibi, "KQ_scaled_alibi");

            // KQ_masked = mask_past(KQ_scaled)
            struct ggml_tensor * KQ_masked = ggml_add(ctx0, KQ_scaled_alibi, KQ_mask);
            ggml_set_name(KQ_masked, "KQ_masked");

            // KQ = soft_max(KQ_masked)
            struct ggml_tensor * KQ_soft_max = ggml_soft_max_inplace(ctx0, KQ_masked);
            ggml_set_name(KQ_soft_max, "KQ_soft_max");

            // split cached V into n_head heads
            struct ggml_tensor * V =
                ggml_view_3d(ctx0, kv_self.v,
                        n_kv, n_embd_head, n_head_kv,
                        ggml_element_size(kv_self.v)*n_ctx,
                        ggml_element_size(kv_self.v)*n_ctx*n_embd_head,
                        ggml_element_size(kv_self.v)*n_ctx*n_embd_gqa*il);
            ggml_set_name(V, "V");

            struct ggml_tensor * KQV = ggml_mul_mat(ctx0, V, KQ_soft_max);
            ggml_set_name(KQV, "KQV");

            // KQV_merged = KQV.permute(0, 2, 1, 3)
            struct ggml_tensor * KQV_merged = ggml_permute(ctx0, KQV, 0, 2, 1, 3);
            ggml_set_name(KQV_merged, "KQV_merged");

            // cur = KQV_merged.contiguous().view(n_embd, n_tokens)
            cur = ggml_cont_2d(ctx0, KQV_merged, n_embd, n_tokens);
            ggml_set_name(cur, "KQV_merged_contiguous");
        }

        // Projection
        cur = ggml_add(ctx0, ggml_mul_mat(ctx0, model.layers[il].wo, cur), model.layers[il].bo);

        // Add the input
        cur = ggml_add(ctx0, cur, inpL);

        struct ggml_tensor * inpFF = cur;

        // FF
        {
            // Norm
            {
                cur = ggml_norm(ctx0, inpFF, norm_eps);
                cur = ggml_add(ctx0, ggml_mul(ctx0, cur, model.layers[il].ffn_norm), model.layers[il].ffn_norm_b);
            }

            cur = ggml_add(ctx0, ggml_mul_mat(ctx0, model.layers[il].w3, cur), model.layers[il].b3);

            // GELU activation
            cur = ggml_gelu(ctx0, cur);

            // Projection
            cur = ggml_add(ctx0, ggml_mul_mat(ctx0, model.layers[il].w2, cur), model.layers[il].b2);
        }

        inpL = ggml_add(ctx0, cur, inpFF);
    }

    // Output Norm
    {
        cur = ggml_norm(ctx0, inpL, norm_eps);
        cur = ggml_add(ctx0, ggml_mul(ctx0, cur, model.output_norm), model.output_norm_b);
    }
    ggml_set_name(cur, "result_norm");

    cur = ggml_mul_mat(ctx0, model.output, cur);
    ggml_set_name(cur, "result_output");

    ggml_build_forward_expand(gf, cur);

    ggml_free(ctx0);

    return gf;
}

static struct ggml_cgraph * llm_build_mpt(
         llama_context & lctx,
     const llama_batch & batch) {
    const auto & model   = lctx.model;
    const auto & hparams = model.hparams;
    const auto & cparams = lctx.cparams;

    const auto & kv_self = lctx.kv_self;

    GGML_ASSERT(!!kv_self.ctx);

    const int64_t n_embd      = hparams.n_embd;
    const int64_t n_layer     = hparams.n_layer;
    const int64_t n_ctx       = cparams.n_ctx;
    const int64_t n_head      = hparams.n_head;
    const int64_t n_head_kv   = hparams.n_head_kv;
    const int64_t n_embd_head = hparams.n_embd_head();
    const int64_t n_embd_gqa  = hparams.n_embd_gqa();

    const float norm_eps       = hparams.f_norm_eps;
    const float clamp_kqv      = hparams.f_clamp_kqv;
    const float max_alibi_bias = hparams.f_max_alibi_bias;

    const int n_gpu_layers = model.n_gpu_layers;

    const int32_t n_tokens = batch.n_tokens;
    const int32_t n_kv     = ggml_allocr_is_measure(lctx.alloc) ? n_ctx            : kv_self.n;
    const int32_t kv_head  = ggml_allocr_is_measure(lctx.alloc) ? n_ctx - n_tokens : kv_self.head;

    auto & buf_compute = lctx.buf_compute;

    struct ggml_init_params params = {
        /*.mem_size   =*/ buf_compute.size,
        /*.mem_buffer =*/ buf_compute.data,
        /*.no_alloc   =*/ false,
    };

    params.no_alloc = true;

    struct ggml_context * ctx0 = ggml_init(params);

    ggml_cgraph * gf = ggml_new_graph(ctx0);

    struct ggml_tensor * cur;
    struct ggml_tensor * inpL;

    //int warmup = 0;
    if (batch.token) {
        struct ggml_tensor * inp_tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);

        ggml_allocr_alloc(lctx.alloc, inp_tokens);
        if (!ggml_allocr_is_measure(lctx.alloc)) {
            memcpy(inp_tokens->data, batch.token, n_tokens*ggml_element_size(inp_tokens));
            //warmup = ((uint32_t*) inp_tokens->data)[0] == 0;
        }

        ggml_set_name(inp_tokens, "inp_tokens");

        inpL = ggml_get_rows(ctx0, model.tok_embeddings, inp_tokens);
    } else {
#ifdef GGML_USE_MPI
        GGML_ASSERT(false && "not implemented");
#endif

        inpL = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, n_tokens);

        ggml_allocr_alloc(lctx.alloc, inpL);
        if (!ggml_allocr_is_measure(lctx.alloc)) {
            memcpy(inpL->data, batch.embd, n_tokens * n_embd * ggml_element_size(inpL));
        }
    }

    const int i_gpu_start = n_layer - n_gpu_layers;
    (void) i_gpu_start;

    // offload functions set the tensor output backend to GPU
    // tensors are GPU-accelerated if any input or the output has been offloaded
    offload_func_t offload_func_nr = llama_nop; // nr = non-repeating
    offload_func_t offload_func_kq = llama_nop;
    offload_func_t offload_func_v  = llama_nop;

#ifdef GGML_USE_CUBLAS
    if (n_gpu_layers > n_layer) {
        offload_func_nr = ggml_cuda_assign_buffers_no_alloc;
    }
    if (n_gpu_layers > n_layer + 1) {
        offload_func_v  = ggml_cuda_assign_buffers_no_alloc;
    }
    if (n_gpu_layers > n_layer + 2) {
        offload_func_kq = ggml_cuda_assign_buffers_no_alloc;
    }
#endif // GGML_USE_CUBLAS

    // KQ_scale
    struct ggml_tensor * KQ_scale = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, 1);
    ggml_set_name(KQ_scale, "1/sqrt(n_embd_head)");
    ggml_allocr_alloc(lctx.alloc, KQ_scale);
    if (!ggml_allocr_is_measure(lctx.alloc)) {
        ggml_set_f32(KQ_scale, 1.0f/sqrtf(float(n_embd)/n_head));
    }

    // KQ_mask (mask for 1 head, it will be broadcasted to all heads)
    struct ggml_tensor * KQ_mask = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, n_kv, n_tokens, 1);
    offload_func_kq(KQ_mask);
    ggml_set_name(KQ_mask, "KQ_mask");
    ggml_allocr_alloc(lctx.alloc, KQ_mask);
    if (!ggml_allocr_is_measure(lctx.alloc)) {
        float * data = (float *) KQ_mask->data;
        memset(data, 0, ggml_nbytes(KQ_mask));

        for (int h = 0; h < 1; ++h) {
            for (int j = 0; j < n_tokens; ++j) {
                const llama_pos    pos    = batch.pos[j];
                const llama_seq_id seq_id = batch.seq_id[j][0];

                for (int i = 0; i < n_kv; ++i) {
                    if (!kv_self.cells[i].has_seq_id(seq_id) || kv_self.cells[i].pos > pos) {
                        data[h*(n_kv*n_tokens) + j*n_kv + i] = -INFINITY;
                    }
                }
            }
        }
    }

    for (int il = 0; il < n_layer; ++il) {
        struct ggml_tensor * attn_norm;

        offload_func_t offload_func = llama_nop;

#ifdef GGML_USE_CUBLAS
        if (il >= i_gpu_start) {
            offload_func = ggml_cuda_assign_buffers_no_alloc;
        }
#endif // GGML_USE_CUBLAS

        // self-attention
        // TODO: refactor into common function (shared with LLaMA)
        {
            attn_norm = ggml_norm(ctx0, inpL, norm_eps);
            offload_func(attn_norm);

            attn_norm = ggml_mul(ctx0, attn_norm, model.layers[il].attn_norm);
            offload_func(attn_norm);

            if (1) {
                cur = attn_norm;
            }

            // compute QKV

            cur = ggml_mul_mat(ctx0, model.layers[il].wqkv, cur);
            offload_func_kq(cur);

            if (clamp_kqv > 0.0f) {
                cur = ggml_clamp(ctx0, cur, -clamp_kqv, clamp_kqv);
                offload_func_kq(cur);
            }

            const size_t wsize = ggml_type_size(cur->type);

            struct ggml_tensor * Qcur = ggml_view_3d(
                ctx0, cur, n_embd_head, n_head, n_tokens,
                wsize * n_embd_head,
                wsize * n_embd_head * (n_head + 2 * n_head_kv),
                0);
            offload_func_kq(Qcur);

            struct ggml_tensor * Kcur = ggml_view_3d(
                ctx0, cur, n_embd_head, n_head_kv, n_tokens,
                wsize * n_embd_head,
                wsize * n_embd_head * (n_head + 2 * n_head_kv),
                wsize * n_embd_head *  n_head);
            offload_func_kq(Kcur);

            struct ggml_tensor * tmpv = ggml_view_3d(
                ctx0, cur, n_embd_head, n_head_kv, n_tokens,
                wsize * n_embd_head,
                wsize * n_embd_head * (n_head + 2 * n_head_kv),
                wsize * n_embd_head * (n_head +     n_head_kv));
            offload_func_kq(Kcur);

            ggml_set_name(Qcur, "Qcur");
            ggml_set_name(Kcur, "Kcur");

            {
                struct ggml_tensor * Vcur = ggml_transpose(ctx0, ggml_reshape_2d(ctx0, ggml_cont(ctx0, tmpv), n_embd_gqa, n_tokens));
                offload_func_v(Vcur);
                offload_func_v(Vcur->src[0]->src[0]);
                ggml_set_name(Vcur, "Vcur");

                struct ggml_tensor * k = ggml_view_1d(ctx0, kv_self.k, n_tokens*n_embd_gqa, (ggml_element_size(kv_self.k)*n_embd_gqa)*(il*n_ctx + kv_head));
                offload_func_kq(k);
                ggml_set_name(k, "k");

                struct ggml_tensor * v = ggml_view_2d(ctx0, kv_self.v, n_tokens, n_embd_gqa,
                        (   n_ctx)*ggml_element_size(kv_self.v),
                        (il*n_ctx)*ggml_element_size(kv_self.v)*n_embd_gqa + kv_head*ggml_element_size(kv_self.v));
                offload_func_v(v);

                ggml_build_forward_expand(gf, ggml_cpy(ctx0, Kcur, k));
                ggml_build_forward_expand(gf, ggml_cpy(ctx0, Vcur, v));
            }

            struct ggml_tensor * Q = ggml_permute(ctx0, Qcur, 0, 2, 1, 3);
            offload_func_kq(Q);
            ggml_set_name(Q, "Q");

            struct ggml_tensor * K =
                ggml_view_3d(ctx0, kv_self.k,
                        n_embd_head, n_kv, n_head_kv,
                        ggml_element_size(kv_self.k)*n_embd_gqa,
                        ggml_element_size(kv_self.k)*n_embd_head,
                        ggml_element_size(kv_self.k)*n_embd_gqa*n_ctx*il);
            offload_func_kq(K);
            ggml_set_name(K, "K");

            struct ggml_tensor * KQ = ggml_mul_mat(ctx0, K, Q);
            offload_func_kq(KQ);
            ggml_set_name(KQ, "KQ");

            struct ggml_tensor * KQ_scaled = ggml_scale(ctx0, KQ, KQ_scale);
            offload_func_kq(KQ_scaled);
            ggml_set_name(KQ_scaled, "KQ_scaled");

            // TODO: replace with ggml_add()
            struct ggml_tensor * KQ_scaled_alibi =
                ggml_alibi(ctx0, KQ_scaled, 0, n_head, max_alibi_bias);
            offload_func_kq(KQ_scaled_alibi);
            ggml_set_name(KQ_scaled_alibi, "KQ_scaled_alibi");

            struct ggml_tensor * KQ_masked = ggml_add(ctx0, KQ_scaled_alibi, KQ_mask);
            offload_func_kq(KQ_masked);
            ggml_set_name(KQ_masked, "KQ_masked");

            struct ggml_tensor * KQ_soft_max = ggml_soft_max(ctx0, KQ_masked);
            offload_func_v(KQ_soft_max);
            ggml_set_name(KQ_soft_max, "KQ_soft_max");

            struct ggml_tensor * V =
                ggml_view_3d(ctx0, kv_self.v,
                        n_kv, n_embd_head, n_head_kv,
                        ggml_element_size(kv_self.v)*n_ctx,
                        ggml_element_size(kv_self.v)*n_ctx*n_embd_head,
                        ggml_element_size(kv_self.v)*n_ctx*n_embd_gqa*il);
            offload_func_v(V);
            ggml_set_name(V, "V");

            struct ggml_tensor * KQV = ggml_mul_mat(ctx0, V, KQ_soft_max);
            offload_func_v(KQV);
            ggml_set_name(KQV, "KQV");

            struct ggml_tensor * KQV_merged = ggml_permute(ctx0, KQV, 0, 2, 1, 3);
            offload_func_v(KQV_merged);
            ggml_set_name(KQV_merged, "KQV_merged");

            cur = ggml_cont_2d(ctx0, KQV_merged, n_embd, n_tokens);
            offload_func_v(cur);
            ggml_set_name(cur, "KQV_merged_contiguous");

            cur = ggml_mul_mat(ctx0, model.layers[il].wo, cur);
            offload_func(cur);
            ggml_set_name(cur, "result_wo");
        }

        // Add the input
        cur = ggml_add(ctx0, cur, inpL);
        offload_func(cur);

        struct ggml_tensor * attn_out = cur;

        // feed forward
        {
            // Norm
            {
                cur = ggml_norm(ctx0, attn_out, norm_eps);
                offload_func(cur);

                cur = ggml_mul(ctx0, cur, model.layers[il].ffn_norm);
                offload_func(cur);
            }

            cur = ggml_mul_mat(ctx0, model.layers[il].w3, cur);
            offload_func(cur);

            cur = ggml_gelu(ctx0, cur);
            offload_func(cur);
            cur = ggml_mul_mat(ctx0, model.layers[il].w2, cur);
            offload_func(cur);
        }

        cur = ggml_add(ctx0, cur, attn_out);
        offload_func(cur);
        // input for next layer
        inpL = cur;
    }

    cur = inpL;

    // norm
    {
        cur = ggml_norm(ctx0, cur, norm_eps);
        offload_func_nr(cur);

        cur = ggml_mul(ctx0, cur, model.output_norm);
        ggml_set_name(cur, "result_norm");
    }

    cur = ggml_mul_mat(ctx0, model.output, cur);
    ggml_set_name(cur, "result_output");

    ggml_build_forward_expand(gf, cur);

    ggml_free(ctx0);

    return gf;
}

static struct ggml_cgraph * llama_build_graph(
         llama_context & lctx,
     const llama_batch & batch) {
    const auto & model = lctx.model;

    struct ggml_cgraph * result = NULL;

    switch (model.arch) {
        case LLM_ARCH_LLAMA:
            {
                result = llm_build_llama(lctx, batch);
            } break;
        case LLM_ARCH_BAICHUAN:
            {
                result = llm_build_baichaun(lctx, batch);
            } break;
        case LLM_ARCH_FALCON:
            {
                result = llm_build_falcon(lctx, batch);
            } break;
        case LLM_ARCH_STARCODER:
            {
                result = llm_build_starcoder(lctx, batch);
            } break;
        case LLM_ARCH_PERSIMMON:
            {
                result = llm_build_persimmon(lctx, batch);
            } break;
        case LLM_ARCH_REFACT:
            {
                result = llm_build_refact(lctx, batch);
            } break;
        case LLM_ARCH_BLOOM:
            {
                result = llm_build_bloom(lctx, batch);
            } break;
        case LLM_ARCH_MPT:
            {
                result = llm_build_mpt(lctx, batch);
            } break;
        default:
            GGML_ASSERT(false);
    }

    return result;
}

// decode a batch of tokens by evaluating the transformer
//
//   - lctx:      llama context
//   - batch:     batch to evaluate
//
// return 0 on success
// return positive int on warning
// return negative int on error
//
static int llama_decode_internal(
         llama_context & lctx,
           llama_batch   batch) {
    const uint32_t n_tokens = batch.n_tokens;

    if (n_tokens == 0) {
        LLAMA_LOG_ERROR("%s: n_tokens == 0", __func__);
        return -1;
    }

    const auto & model   = lctx.model;
    const auto & hparams = model.hparams;
    const auto & cparams = lctx.cparams;

    const auto n_batch = cparams.n_batch;

    GGML_ASSERT(n_tokens <= n_batch);

    int n_threads = n_tokens == 1 ? cparams.n_threads : cparams.n_threads_batch;
    GGML_ASSERT((!batch.token && batch.embd) || (batch.token && !batch.embd)); // NOLINT

    const int64_t t_start_us = ggml_time_us();

#ifdef GGML_USE_MPI
    // TODO: needs fix after #3228
    GGML_ASSERT(false && "not implemented");
    //ggml_mpi_eval_init(lctx.ctx_mpi, &n_tokens, &n_past, &n_threads);
#endif

    GGML_ASSERT(n_threads > 0);

    auto & kv_self = lctx.kv_self;

    GGML_ASSERT(!!kv_self.ctx);

    const int64_t n_embd  = hparams.n_embd;
    const int64_t n_vocab = hparams.n_vocab;

    // helpers for smoother batch API transistion
    // after deprecating the llama_eval calls, these will be removed
    std::vector<llama_pos> pos;

    std::vector<int32_t>                   n_seq_id;
    std::vector<llama_seq_id *>            seq_id_arr;
    std::vector<std::vector<llama_seq_id>> seq_id;

    if (batch.pos == nullptr) {
        pos.resize(n_tokens);
        for (uint32_t i = 0; i < n_tokens; i++) {
            pos[i] = batch.all_pos_0 + i*batch.all_pos_1;
        }

        batch.pos = pos.data();
    }

    if (batch.seq_id == nullptr) {
        n_seq_id.resize(n_tokens);
        seq_id.resize(n_tokens);
        seq_id_arr.resize(n_tokens);
        for (uint32_t i = 0; i < n_tokens; i++) {
            n_seq_id[i] = 1;
            seq_id[i].resize(1);
            seq_id[i][0] = batch.all_seq_id;
            seq_id_arr[i] = seq_id[i].data();
        }

        batch.n_seq_id = n_seq_id.data();
        batch.seq_id = seq_id_arr.data();
    }

    if (!llama_kv_cache_find_slot(kv_self, batch)) {
        return 1;
    }

    // a heuristic, to avoid attending the full cache if it is not yet utilized
    // after enough generations, the benefit from this heuristic disappears
    // if we start defragmenting the cache, the benefit from this will be more important
    //kv_self.n = std::max(32, GGML_PAD(llama_kv_cache_cell_max(kv_self), 32));   // TODO: this might be better for CUDA?
    kv_self.n = std::min((int32_t) cparams.n_ctx, std::max(32, llama_kv_cache_cell_max(kv_self)));

    //printf("kv_self.n = %d\n", kv_self.n);

    ggml_allocr_reset(lctx.alloc);

    ggml_cgraph * gf = llama_build_graph(lctx, batch);

    ggml_allocr_alloc_graph(lctx.alloc, gf);

    struct ggml_tensor * res        = gf->nodes[gf->n_nodes - 1];
    struct ggml_tensor * embeddings = gf->nodes[gf->n_nodes - 2];

    GGML_ASSERT(strcmp(res->name,        "result_output") == 0);
    GGML_ASSERT(strcmp(embeddings->name, "result_norm")   == 0);


#ifdef GGML_USE_CUBLAS
    for (int i = 0; i < gf->n_leafs; i++) {
        ggml_tensor * node = gf->leafs[i];
        if (node->backend == GGML_BACKEND_GPU && node->extra == NULL) {
            ggml_cuda_assign_scratch_offset(node, (char*)node->data - (char *) lctx.buf_alloc.data);
            ggml_cuda_copy_to_device(node);
        }
    }

    for (int i = 0; i < gf->n_nodes; i++) {
        ggml_tensor * node = gf->nodes[i];
        if (node->backend == GGML_BACKEND_GPU && node->extra == NULL) {
            ggml_cuda_assign_scratch_offset(node, (char*)node->data - (char *) lctx.buf_alloc.data);
        }
    }

    ggml_cuda_set_mul_mat_q(cparams.mul_mat_q);

    // HACK: ggml-alloc may change the tensor backend when reusing a parent, so force output to be on the CPU here if needed
    if (!lctx.embedding.empty()) {
        embeddings->backend = GGML_BACKEND_CPU;
    }
    res->backend = GGML_BACKEND_CPU;
#endif

    // LLAMA_LOG_INFO("graph build time: %.3f ms (%d nodes, %d leafs)\n", (ggml_time_us() - t_start_us)/1000.0, gf->n_nodes, gf->n_leafs);

    // for big prompts, if BLAS is enabled, it is better to use only one thread
    // otherwise, the threads are spin-lock waiting for the BLAS calls and are degrading the performance
    // TODO: this is mostly important for Apple Silicon where CBLAS is still performing very well
    //       we still need some threads to process all non-mul_mat ops, but not too much to avoid interfering
    //       with the BLAS calls. need a better solution
    if (n_tokens >= 32 && ggml_cpu_has_blas() && !ggml_cpu_has_gpublas()) {
        n_threads = std::min(4, n_threads);
    }

    // If all tensors can be run on the GPU then using more than 1 thread is detrimental.
    const bool full_offload_supported = model.arch == LLM_ARCH_LLAMA ||
        model.arch == LLM_ARCH_BAICHUAN ||
        model.arch == LLM_ARCH_FALCON ||
        model.arch == LLM_ARCH_REFACT ||
        model.arch == LLM_ARCH_MPT;
    const bool fully_offloaded = model.n_gpu_layers >= (int) hparams.n_layer + 3;
    if (ggml_cpu_has_cublas() && full_offload_supported && fully_offloaded) {
        n_threads = 1;
    }

#if GGML_USE_MPI
    const int64_t n_layer = hparams.n_layer;
    ggml_mpi_graph_compute_pre(lctx.ctx_mpi, gf, n_layer);
#endif

#ifdef GGML_USE_METAL
    if (lctx.ctx_metal) {
        ggml_metal_set_n_cb     (lctx.ctx_metal, n_threads);
        ggml_metal_graph_compute(lctx.ctx_metal, gf);
    } else {
        ggml_graph_compute_helper(lctx.work_buffer, gf, n_threads);
    }
#else
    ggml_graph_compute_helper(lctx.work_buffer, gf, n_threads);
#endif

#if GGML_USE_MPI
    ggml_mpi_graph_compute_post(lctx.ctx_mpi, gf, n_layer);
#endif

    // update the kv ring buffer
    lctx.kv_self.has_shift  = false;
    lctx.kv_self.head      += n_tokens;
    // Ensure kv cache head points to a valid index.
    if (lctx.kv_self.head >= lctx.kv_self.size) {
        lctx.kv_self.head = 0;
    }

#ifdef GGML_PERF
    // print timing information per ggml operation (for debugging purposes)
    // requires GGML_PERF to be defined
    ggml_graph_print(gf);
#endif

    // plot the computation graph in dot format (for debugging purposes)
    //if (n_past%100 == 0) {
    //    ggml_graph_dump_dot(gf, NULL, "llama.dot");
    //}

    // extract logits
    {
        auto & logits_out = lctx.logits;

        if (batch.logits) {
            logits_out.resize(n_vocab * n_tokens);
            for (uint32_t i = 0; i < n_tokens; i++) {
                if (batch.logits[i] == 0) {
                    continue;
                }
                memcpy(logits_out.data() + (n_vocab*i), (float *) ggml_get_data(res) + (n_vocab*i), sizeof(float)*n_vocab);
            }
        } else if (lctx.logits_all) {
            logits_out.resize(n_vocab * n_tokens);
            memcpy(logits_out.data(), (float *) ggml_get_data(res), sizeof(float)*n_vocab*n_tokens);
        } else {
            logits_out.resize(n_vocab);
            memcpy(logits_out.data(), (float *) ggml_get_data(res) + (n_vocab*(n_tokens - 1)), sizeof(float)*n_vocab);
        }
    }

    // extract embeddings
    if (!lctx.embedding.empty()) {
        auto & embedding_out = lctx.embedding;

        embedding_out.resize(n_embd);
        memcpy(embedding_out.data(), (float *) ggml_get_data(embeddings) + (n_embd*(n_tokens - 1)), sizeof(float)*n_embd);
    }

    // measure the performance only for the single-token evals
    if (n_tokens == 1) {
        lctx.t_eval_us += ggml_time_us() - t_start_us;
        lctx.n_eval++;
    }
    else if (n_tokens > 1) {
        lctx.t_p_eval_us += ggml_time_us() - t_start_us;
        lctx.n_p_eval += n_tokens;
    }

    // get a more accurate load time, upon first eval
    // TODO: fix this
    if (!lctx.has_evaluated_once) {
        lctx.t_load_us = ggml_time_us() - lctx.t_start_us;
        lctx.has_evaluated_once = true;
    }

    return 0;
}

//
// tokenizer
//

static enum llama_vocab_type llama_vocab_get_type(const llama_vocab & vocab) {
    return vocab.type;
}

static bool llama_is_normal_token(const llama_vocab & vocab, llama_token id) {
    return vocab.id_to_token[id].type == LLAMA_TOKEN_TYPE_NORMAL;
}

static bool llama_is_unknown_token(const llama_vocab & vocab, llama_token id) {
    return vocab.id_to_token[id].type == LLAMA_TOKEN_TYPE_UNKNOWN;
}

static bool llama_is_control_token(const llama_vocab & vocab, llama_token id) {
    return vocab.id_to_token[id].type == LLAMA_TOKEN_TYPE_CONTROL;
}

static bool llama_is_byte_token(const llama_vocab & vocab, llama_token id) {
    return vocab.id_to_token[id].type == LLAMA_TOKEN_TYPE_BYTE;
}

static bool llama_is_user_defined_token(const llama_vocab& vocab, llama_token id) {
    return vocab.id_to_token[id].type == LLAMA_TOKEN_TYPE_USER_DEFINED;
}

static uint8_t llama_token_to_byte(const llama_vocab& vocab, llama_token id) {
    GGML_ASSERT(llama_is_byte_token(vocab, id));
    const auto& token_data = vocab.id_to_token.at(id);
    switch (llama_vocab_get_type(vocab)) {
    case LLAMA_VOCAB_TYPE_SPM: {
        auto buf = token_data.text.substr(3, 2);
        return strtol(buf.c_str(), NULL, 16);
    }
    case LLAMA_VOCAB_TYPE_BPE: {
        GGML_ASSERT(false);
        return unicode_to_bytes_bpe(token_data.text);
    }
    default:
        GGML_ASSERT(false);
    }
}

static llama_token llama_byte_to_token(const llama_vocab & vocab, uint8_t ch) {
    switch (llama_vocab_get_type(vocab)) {
    case LLAMA_VOCAB_TYPE_SPM: {
        char buf[7];
        int result = snprintf(buf, sizeof(buf), "<0x%02X>", ch);
        GGML_ASSERT(0 <= result && result < 7);
        return vocab.token_to_id.at(buf);
    }
    case LLAMA_VOCAB_TYPE_BPE: {
        return vocab.token_to_id.at(bytes_to_unicode_bpe(ch));
    }
    default:
        GGML_ASSERT(false);
    }
}

static void llama_escape_whitespace(std::string & text) {
    replace_all(text, " ", "\xe2\x96\x81");
}

static void llama_unescape_whitespace(std::string & word) {
    replace_all(word, "\xe2\x96\x81", " ");
}

struct llm_symbol {
    using index = int;
    index prev;
    index next;
    const char * text;
    size_t n;
};

static_assert(std::is_trivially_copyable<llm_symbol>::value, "llm_symbol is not trivially copyable");

// SPM tokenizer
// original implementation:
// https://github.com/ggerganov/llama.cpp/commit/074bea2eb1f1349a0118239c4152914aecaa1be4

struct llm_bigram_spm {
    struct comparator {
        bool operator()(llm_bigram_spm & l, llm_bigram_spm & r) {
            return (l.score < r.score) || (l.score == r.score && l.left > r.left);
        }
    };
    using queue_storage = std::vector<llm_bigram_spm>;
    using queue = std::priority_queue<llm_bigram_spm, queue_storage, comparator>;
    llm_symbol::index left;
    llm_symbol::index right;
    float score;
    size_t size;
};

struct llm_tokenizer_spm {
    llm_tokenizer_spm(const llama_vocab & vocab): vocab(vocab) {}

    void tokenize(const std::string & text, std::vector<llama_vocab::id> & output) {
        // split string into utf8 chars
        int index = 0;
        size_t offs = 0;
        while (offs < text.size()) {
            llm_symbol sym;
            size_t len = utf8_len(text[offs]);
            sym.text = text.c_str() + offs;
            sym.n = std::min(len, text.size() - offs);
            offs += sym.n;
            sym.prev = index - 1;
            sym.next = offs == text.size() ? -1 : index + 1;
            index++;
            symbols.emplace_back(sym);
        }

        // seed the work queue with all possible 2-character tokens.
        for (size_t i = 1; i < symbols.size(); ++i) {
            try_add_bigram(i - 1, i);
        }

        // keep substituting the highest frequency pairs for as long as we can.
        while (!work_queue.empty()) {
            auto bigram = work_queue.top();
            work_queue.pop();

            auto & left_sym = symbols[bigram.left];
            auto & right_sym = symbols[bigram.right];

            // if one of the symbols already got merged, skip it.
            if (left_sym.n == 0 || right_sym.n == 0 ||
                left_sym.n + right_sym.n != bigram.size) {
                continue;
            }

            // merge the right sym into the left one
            left_sym.n += right_sym.n;
            right_sym.n = 0;

            //LLAMA_LOG_INFO("left = '%*s' size = %zu\n", (int) left_sym.n, left_sym.text, bigram.size);

            // remove the right sym from the chain
            left_sym.next = right_sym.next;
            if (right_sym.next >= 0) {
                symbols[right_sym.next].prev = bigram.left;
            }

            // find more substitutions
            try_add_bigram(left_sym.prev, bigram.left);
            try_add_bigram(bigram.left, left_sym.next);
        }

        for (int i = 0; i != -1; i = symbols[i].next) {
            auto & symbol = symbols[i];
            resegment(symbol, output);
        }
    }

private:
    void resegment(llm_symbol & symbol, std::vector<llama_vocab::id> & output) {
        auto text = std::string(symbol.text, symbol.n);
        auto token = vocab.token_to_id.find(text);

        // Do we need to support is_unused?
        if (token != vocab.token_to_id.end()) {
            output.push_back((*token).second);
            return;
        }

        const auto p = rev_merge.find(text);

        if (p == rev_merge.end()) {
            // output any symbols that did not form tokens as bytes.
            for (int j = 0; j < (int)symbol.n; ++j) {
                llama_vocab::id token_id = llama_byte_to_token(vocab, symbol.text[j]);
                output.push_back(token_id);
            }
            return;
        }

        resegment(symbols[p->second.first],  output);
        resegment(symbols[p->second.second], output);
    }

    void try_add_bigram(int left, int right) {
        if (left == -1 || right == -1) {
            return;
        }

        const std::string text = std::string(symbols[left].text, symbols[left].n + symbols[right].n);
        auto token = vocab.token_to_id.find(text);

        if (token == vocab.token_to_id.end()) {
            return;
        }

        if (static_cast<size_t>((*token).second) >= vocab.id_to_token.size()) {
            return;
        }

        const auto & tok_data = vocab.id_to_token[(*token).second];

        llm_bigram_spm bigram;
        bigram.left  = left;
        bigram.right = right;
        bigram.score = tok_data.score;
        bigram.size  = text.size();

        work_queue.push(bigram);

        // Do we need to support is_unused?
        rev_merge[text] = std::make_pair(left, right);
    }

    const llama_vocab & vocab;

    std::vector<llm_symbol> symbols;
    llm_bigram_spm::queue work_queue;

    std::map<std::string, std::pair<int, int>> rev_merge;
};

// BPE tokenizer
// adapted from https://github.com/cmp-nct/ggllm.cpp [MIT License]
// tried to simplify unicode stuff, so most likely does not work 100% correctly!

// TODO: there are a lot of common parts between spm and bpe tokenizers, should be refactored and reused

struct llm_bigram_bpe {
    struct comparator {
        bool operator()(const llm_bigram_bpe & l, const llm_bigram_bpe & r) const {
            return l.rank > r.rank || (l.rank == r.rank && l.left > r.left);
        }
    };

    using queue_storage = std::vector<llm_bigram_bpe>;
    using queue = std::priority_queue<llm_bigram_bpe, queue_storage, comparator>;
    llm_symbol::index left;
    llm_symbol::index right;
    std::string text;
    int rank;
    size_t size;
};

struct llm_tokenizer_bpe {
    llm_tokenizer_bpe(const llama_vocab & vocab): vocab(vocab) {}

    void tokenize(const std::string & text, std::vector<llama_vocab::id> & output) {
        int final_prev_index = -1;
        auto word_collection = bpe_gpt2_preprocess(text);

        symbols_final.clear();

        for (auto & word : word_collection) {
            work_queue = llm_bigram_bpe::queue();
            symbols.clear();

            int index = 0;
            size_t offset = 0;

            while (offset < word.size()) {
                llm_symbol sym;
                size_t char_len = std::min(word.size() - offset, (size_t) ::utf8_len(word[offset]));
                sym.text = word.c_str() + offset;
                sym.n = char_len;
                offset += sym.n;
                sym.prev = index - 1;
                sym.next = offset == word.size() ? -1 : index + 1;
                index++;
                symbols.emplace_back(sym);
            }
            for (size_t i = 1; i < symbols.size(); ++i) {
                add_new_bigram(i - 1, i);
            }

            // build token(s)
            while (!work_queue.empty()) {
                auto bigram = work_queue.top();
                work_queue.pop();

                auto & left_symbol = symbols[bigram.left];
                auto & right_symbol = symbols[bigram.right];

                if (left_symbol.n == 0 || right_symbol.n == 0) {
                    continue;
                }
                std::string left_token = std::string(left_symbol.text, left_symbol.n);
                std::string right_token = std::string(right_symbol.text, right_symbol.n);
                if (left_token + right_token != bigram.text) {
                    continue;  // Skip this bigram if it's outdated
                }

                // merge the right sym into the left one
                left_symbol.n += right_symbol.n;
                right_symbol.n = 0;

                // remove the right sym from the chain
                left_symbol.next = right_symbol.next;
                if (right_symbol.next >= 0) {
                    symbols[right_symbol.next].prev = bigram.left;
                }

                add_new_bigram(left_symbol.prev, bigram.left);  // left side of current symbol
                add_new_bigram(bigram.left, left_symbol.next);  // right side of current symbol
            }

            // add the fnished tokens to the final list keeping correct order for next and prev
            for (auto & sym : symbols) {
                if (sym.n > 0) {
                    sym.prev = final_prev_index;
                    sym.next = -1;
                    if (final_prev_index != -1) {
                        symbols_final[final_prev_index].next = symbols_final.size();
                    }
                    symbols_final.emplace_back(sym);
                    final_prev_index = symbols_final.size() - 1;
                }
            }
        }

        symbols = symbols_final;

        if (!symbols.empty()) {
            for (int i = 0; i != -1; i = symbols[i].next) {
                auto & symbol = symbols[i];
                if (symbol.n == 0) {
                    continue;
                }

                const std::string str = std::string(symbol.text, symbol.n);
                const auto token = vocab.token_to_id.find(str);

                if (token == vocab.token_to_id.end()) {
                    for (auto j = str.begin(); j != str.end(); ++j) {
                        std::string byte_str(1, *j);
                        auto token_multibyte = vocab.token_to_id.find(byte_str);
                        if (token_multibyte == vocab.token_to_id.end()) {
                            throw std::runtime_error("ERROR: byte not found in vocab");
                        }
                        output.push_back((*token_multibyte).second);
                    }
                } else {
                    output.push_back((*token).second);
                }
            }
        }
    }

private:
    void add_new_bigram(int left, int right) {
        if (left == -1 || right == -1) {
            return;
        }

        std::string left_token  = std::string(symbols[left].text,  symbols[left].n);
        std::string right_token = std::string(symbols[right].text, symbols[right].n);

        int rank_found = -1;

        rank_found = vocab.find_bpe_rank(left_token, right_token);

        if (rank_found < 0) {
            return;
        }

        llm_bigram_bpe bigram;

        bigram.left  = left;
        bigram.right = right;
        bigram.text  = left_token + right_token;
        bigram.size  = left_token.size() + right_token.size();
        bigram.rank  = rank_found;

        work_queue.push(bigram);
    }

    std::vector<std::string> bpe_gpt2_preprocess(const std::string & text) {
        std::vector<std::string> bpe_words;
        std::vector<std::string> bpe_encoded_words;

        std::string token = "";
        // GPT2 system regex:  's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
        bool collecting_numeric = false;
        bool collecting_letter = false;
        bool collecting_special = false;
        bool collecting_whitespace_lookahead = false;
        bool collecting = false;

        std::vector<std::string> text_utf;
        text_utf.reserve(text.size());
        bpe_words.reserve(text.size());
        bpe_encoded_words.reserve(text.size());

        auto cps = codepoints_from_utf8(text);
        for (size_t i = 0; i < cps.size(); ++i)
            text_utf.emplace_back(codepoint_to_utf8(cps[i]));

        for (int i = 0; i < (int)text_utf.size(); i++) {
            const std::string & utf_char = text_utf[i];
            bool split_condition = false;
            int bytes_remain = text_utf.size() - i;
            // forward backward lookups
            const std::string & utf_char_next = (i + 1 < (int)text_utf.size()) ? text_utf[i + 1] : "";
            const std::string & utf_char_next_next = (i + 2 < (int)text_utf.size()) ? text_utf[i + 2] : "";

            // handling contractions
            if (!split_condition && bytes_remain >= 2) {
                // 's|'t|'m|'d
                if (utf_char == "\'" && (utf_char_next == "s" || utf_char_next == "t" || utf_char_next == "m" || utf_char_next == "d")) {
                    split_condition = true;
                }
                if (split_condition) {
                    if (token.size()) {
                        bpe_words.emplace_back(token); // push previous content as token
                    }
                    token = utf_char + utf_char_next;
                    bpe_words.emplace_back(token);
                    token = "";
                    i++;
                    continue;
                }
            }
            if (!split_condition && bytes_remain >= 3) {
                // 're|'ve|'ll
                if (utf_char == "\'" && (
                    (utf_char_next == "r" && utf_char_next_next == "e") ||
                    (utf_char_next == "v" && utf_char_next_next == "e") ||
                    (utf_char_next == "l" && utf_char_next_next == "l"))
                    ) {
                    split_condition = true;
                }
                if (split_condition) {
                    // current token + next token can be defined
                    if (token.size()) {
                        bpe_words.emplace_back(token); // push previous content as token
                    }
                    token = utf_char + utf_char_next + utf_char_next_next;
                    bpe_words.emplace_back(token); // the contraction
                    token = "";
                    i += 2;
                    continue;
                }
            }

            if (!split_condition && !collecting) {
                if (codepoint_type(utf_char) == CODEPOINT_TYPE_LETTER || (!token.size() && utf_char == " " && codepoint_type(utf_char_next) == CODEPOINT_TYPE_LETTER)) {
                    collecting_letter = true;
                    collecting = true;
                }
                else if (codepoint_type(utf_char) == CODEPOINT_TYPE_DIGIT || (!token.size() && utf_char == " " && codepoint_type(utf_char_next) == CODEPOINT_TYPE_DIGIT)) {
                    collecting_numeric = true;
                    collecting = true;
                }
                else if (
                    ((codepoint_type(utf_char) != CODEPOINT_TYPE_LETTER && codepoint_type(utf_char) != CODEPOINT_TYPE_DIGIT) && (codepoint_type(utf_char) != CODEPOINT_TYPE_WHITESPACE)) ||
                    (!token.size() && utf_char == " " && codepoint_type(utf_char_next) != CODEPOINT_TYPE_LETTER && codepoint_type(utf_char_next) != CODEPOINT_TYPE_DIGIT && codepoint_type(utf_char_next) != CODEPOINT_TYPE_WHITESPACE)
                    ) {
                    collecting_special = true;
                    collecting = true;
                }
                else if (codepoint_type(utf_char) == CODEPOINT_TYPE_WHITESPACE && codepoint_type(utf_char_next) == CODEPOINT_TYPE_WHITESPACE) {
                    collecting_whitespace_lookahead = true;
                    collecting = true;
                }
                else if (codepoint_type(utf_char) == CODEPOINT_TYPE_WHITESPACE) {
                    split_condition = true;
                }
            }
            else if (!split_condition && collecting) {
                if (collecting_letter && codepoint_type(utf_char) != CODEPOINT_TYPE_LETTER) {
                    split_condition = true;
                }
                else if (collecting_numeric && codepoint_type(utf_char) != CODEPOINT_TYPE_DIGIT) {
                    split_condition = true;
                }
                else if (collecting_special && (codepoint_type(utf_char) == CODEPOINT_TYPE_LETTER || codepoint_type(utf_char) == CODEPOINT_TYPE_DIGIT || codepoint_type(utf_char) == CODEPOINT_TYPE_WHITESPACE)) {
                    split_condition = true;
                }
                else if (collecting_whitespace_lookahead && (codepoint_type(utf_char_next) == CODEPOINT_TYPE_LETTER || codepoint_type(utf_char_next) == CODEPOINT_TYPE_DIGIT)) {
                    split_condition = true;
                }
            }

            if (utf_char_next == "") {
                split_condition = true; // final
                token += utf_char;
            }

            if (split_condition) {
                if (token.size()) {
                    bpe_words.emplace_back(token);
                }
                token = utf_char;
                collecting = false;
                collecting_letter = false;
                collecting_numeric = false;
                collecting_special = false;
                collecting_whitespace_lookahead = false;
            }
            else {
                token += utf_char;
            }
        }

        for (std::string & word : bpe_words) {
            std::string encoded_token = "";
            for (char & c : word) {
                encoded_token += bytes_to_unicode_bpe(c);
            }
            bpe_encoded_words.emplace_back(encoded_token);
        }

        return bpe_encoded_words;
    }

    const llama_vocab & vocab;

    std::vector<llm_symbol> symbols;
    std::vector<llm_symbol> symbols_final;

    llm_bigram_bpe::queue work_queue;
};

typedef enum FRAGMENT_BUFFER_VARIANT_TYPE{
    FRAGMENT_BUFFER_VARIANT_TYPE_TOKEN,
    FRAGMENT_BUFFER_VARIANT_TYPE_RAW_TEXT
} FRAGMENT_BUFFER_VARIANT_TYPE;

struct fragment_buffer_variant{
    fragment_buffer_variant(llama_vocab::id _token)
    :
        type(FRAGMENT_BUFFER_VARIANT_TYPE_TOKEN),
        token(_token),
        raw_text(_dummy),
        offset(0),
        length(0){}
    fragment_buffer_variant(const std::string & _raw_text, int64_t _offset, int64_t _length)
    :
        type(FRAGMENT_BUFFER_VARIANT_TYPE_RAW_TEXT),
        token((llama_vocab::id)-1),
        raw_text(_raw_text),
        offset(_offset),
        length(_length){
            GGML_ASSERT( _offset >= 0 );
            GGML_ASSERT( _length >= 1 );
            GGML_ASSERT( offset + length <= raw_text.length() );
        }

    const FRAGMENT_BUFFER_VARIANT_TYPE type;
    const llama_vocab::id token;
    const std::string _dummy;
    const std::string & raw_text;
    const uint64_t offset;
    const uint64_t length;
};

// #define PRETOKENIZERDEBUG

static void tokenizer_st_partition(const llama_vocab & vocab, std::forward_list<fragment_buffer_variant> & buffer)
{
    // for each special token
    for (const auto & st: vocab.special_tokens_cache) {
        const auto & special_token = st.first;
        const auto & special_id    = st.second;

        // for each text fragment
        std::forward_list<fragment_buffer_variant>::iterator it = buffer.begin();
        while (it != buffer.end()) {
            auto & fragment = (*it);

            // if a fragment is text ( not yet processed )
            if (fragment.type == FRAGMENT_BUFFER_VARIANT_TYPE_RAW_TEXT) {
                auto * raw_text = &(fragment.raw_text);

                auto raw_text_base_offset = fragment.offset;
                auto raw_text_base_length = fragment.length;

                // loop over the text
                while (true) {
                    // find the first occurence of a given special token in this fragment
                    //  passing offset argument only limit the "search area" but match coordinates
                    //  are still relative to the source full raw_text
                    auto match = raw_text->find(special_token, raw_text_base_offset);

                    // no occurences found, stop processing this fragment for a given special token
                    if (match == std::string::npos) break;

                    // check if match is within bounds of offset <-> length
                    if (match + special_token.length() > raw_text_base_offset + raw_text_base_length) break;

#ifdef PRETOKENIZERDEBUG
                    fprintf(stderr, "FF: (%ld %ld %ld) '%s'\n", raw_text->length(), raw_text_base_offset, raw_text_base_length, raw_text->substr(raw_text_base_offset, raw_text_base_length).c_str());
#endif
                    auto source = std::distance(buffer.begin(), it);

                    // if match is further than base offset
                    //  then we have some text to the left of it
                    if (match > raw_text_base_offset) {
                        // left
                        const int64_t left_reminder_offset = raw_text_base_offset + 0;
                        const int64_t left_reminder_length = match - raw_text_base_offset;
                        buffer.emplace_after(it, (*raw_text), left_reminder_offset, left_reminder_length);

#ifdef PRETOKENIZERDEBUG
                        fprintf(stderr, "FL: (%ld %ld) '%s'\n", left_reminder_offset, left_reminder_length, raw_text->substr(left_reminder_offset, left_reminder_length).c_str());
#endif
                        it++;
                    }

                    // special token
                    buffer.emplace_after(it, special_id);
                    it++;

                    // right
                    if (match + special_token.length() < raw_text_base_offset + raw_text_base_length) {
                        const int64_t right_reminder_offset = match + special_token.length();
                        const int64_t right_reminder_length = raw_text_base_length - ((match - raw_text_base_offset) + special_token.length());
                        buffer.emplace_after(it, (*raw_text), right_reminder_offset, right_reminder_length);

#ifdef PRETOKENIZERDEBUG
                        fprintf(stderr, "FR: (%ld %ld) '%s'\n", right_reminder_offset, right_reminder_length, raw_text->substr(right_reminder_offset, right_reminder_length).c_str());
#endif

                        it++;

                        if (source == 0) {
                            buffer.erase_after(buffer.before_begin());
                        } else {
                            buffer.erase_after(std::next(buffer.begin(), (source-1)));
                        }

                        // repeat for the right side
                        raw_text_base_offset = right_reminder_offset;
                        raw_text_base_length = right_reminder_length;

#ifdef PRETOKENIZERDEBUG
                        fprintf(stderr, "RR: (%ld %ld) '%s'\n", raw_text_base_offset, raw_text_base_length, raw_text->substr(raw_text_base_offset, raw_text_base_length).c_str());
#endif
                    } else {
                        if (source == 0) {
                            buffer.erase_after(buffer.before_begin());
                        } else {
                            buffer.erase_after(std::next(buffer.begin(), (source-1)));
                        }
                        break;
                    }
                }
            }
            it++;
        }
    }
}

static std::vector<llama_vocab::id> llama_tokenize_internal(const llama_vocab & vocab, std::string raw_text, bool bos, bool special) {
    std::vector<llama_vocab::id> output;

    // OG tokenizer behavior:
    //
    // tokenizer.encode('', add_bos=True)  returns [1]
    // tokenizer.encode('', add_bos=False) returns []

    if (bos && vocab.special_bos_id != -1) {
        output.push_back(vocab.special_bos_id);
    }

    if (raw_text.empty()) {
        return output;
    }

    std::forward_list<fragment_buffer_variant> fragment_buffer;
    fragment_buffer.emplace_front( raw_text, 0, raw_text.length() );

    if (special) tokenizer_st_partition( vocab, fragment_buffer );

    switch (vocab.type) {
        case LLAMA_VOCAB_TYPE_SPM:
            {
                for (const auto & fragment: fragment_buffer)
                {
                    if (fragment.type == FRAGMENT_BUFFER_VARIANT_TYPE_RAW_TEXT)
                    {
                        // without adding this leading whitespace, we do not get the same results as the original tokenizer

                        // TODO: It's likely possible to get rid of this string copy entirely
                        //  by modifying llm_tokenizer_x to operate with string offsets like pre-tokenizer
                        //  and passing 'add space prefix' as bool argument
                        //
                        auto raw_text = (special ? "" : " ") + fragment.raw_text.substr(fragment.offset, fragment.length);

#ifdef PRETOKENIZERDEBUG
                        fprintf(stderr,"TT: (%ld %ld %ld) '%s'\n", raw_text.length(), fragment.offset, fragment.length, raw_text.c_str());
#endif
                        llm_tokenizer_spm tokenizer(vocab);
                        llama_escape_whitespace(raw_text);
                        tokenizer.tokenize(raw_text, output);
                    }
                    else // if (fragment.type == FRAGMENT_BUFFER_VARIANT_TYPE_TOKEN)
                    {
                        output.push_back(fragment.token);
                    }
                }
            } break;
        case LLAMA_VOCAB_TYPE_BPE:
            {
                for (const auto & fragment: fragment_buffer)
                {
                    if (fragment.type == FRAGMENT_BUFFER_VARIANT_TYPE_RAW_TEXT)
                    {
                        auto raw_text = fragment.raw_text.substr(fragment.offset, fragment.length);

#ifdef PRETOKENIZERDEBUG
                        fprintf(stderr,"TT: (%ld %ld %ld) '%s'\n", raw_text.length(), fragment.offset, fragment.length, raw_text.c_str());
#endif
                        llm_tokenizer_bpe tokenizer(vocab);
                        tokenizer.tokenize(raw_text, output);
                    }
                    else // if (fragment.type == FRAGMENT_BUFFER_VARIANT_TYPE_TOKEN)
                    {
                        output.push_back(fragment.token);
                    }
                }
            } break;
    }

    return output;
}

//
// grammar - internal
//

struct llama_partial_utf8 {
    uint32_t value;    // bit value so far (unshifted)
    int      n_remain; // num bytes remaining; -1 indicates invalid sequence
};

struct llama_grammar {
    const std::vector<std::vector<llama_grammar_element>>   rules;
    std::vector<std::vector<const llama_grammar_element *>> stacks;

    // buffer for partially generated UTF-8 sequence from accepted tokens
    llama_partial_utf8                                      partial_utf8;
};

struct llama_grammar_candidate {
    size_t               index;
    const uint32_t     * code_points;
    llama_partial_utf8   partial_utf8;
};

// Decodes a UTF-8 string which may end in an incomplete sequence. Adds a terminating 0 for use as
// pointer. If an invalid sequence is encountered, returns `llama_partial_utf8.n_remain == -1`.
static std::pair<std::vector<uint32_t>, llama_partial_utf8> decode_utf8(
        const char         * src,
        llama_partial_utf8   partial_start) {
    static const int      lookup[] = { 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 2, 2, 3, 4 };
    const char          * pos      = src;
    std::vector<uint32_t> code_points;
    uint32_t              value    = partial_start.value;
    int                   n_remain = partial_start.n_remain;

    // continue previous decode, if applicable
    while (*pos != 0 && n_remain > 0) {
        uint8_t next_byte = static_cast<uint8_t>(*pos);
        if ((next_byte >> 6) != 2) {
            // invalid sequence, abort
            code_points.push_back(0);
            return std::make_pair(std::move(code_points), llama_partial_utf8{ 0, -1 });
        }
        value = (value << 6) + (next_byte & 0x3F);
        ++pos;
        --n_remain;
    }

    if (partial_start.n_remain > 0 && n_remain == 0) {
        code_points.push_back(value);
    }

    // decode any subsequent utf-8 sequences, which may end in an incomplete one
    while (*pos != 0) {
        uint8_t  first_byte = static_cast<uint8_t>(*pos);
        uint8_t  highbits   = first_byte >> 4;
                 n_remain   = lookup[highbits] - 1;

        if (n_remain < 0) {
            // invalid sequence, abort
            code_points.clear();
            code_points.push_back(0);
            return std::make_pair(std::move(code_points), llama_partial_utf8{ 0, n_remain });
        }

        uint8_t  mask       = (1 << (7 - n_remain)) - 1;
                 value      = first_byte & mask;
        ++pos;
        while (*pos != 0 && n_remain > 0) {
            value = (value << 6) + (static_cast<uint8_t>(*pos) & 0x3F);
            ++pos;
            --n_remain;
        }
        if (n_remain == 0) {
            code_points.push_back(value);
        }
    }
    code_points.push_back(0);

    return std::make_pair(std::move(code_points), llama_partial_utf8{ value, n_remain });
}

// returns true iff pos points to the end of one of the definitions of a rule
static bool llama_grammar_is_end_of_sequence(const llama_grammar_element * pos) {
    switch (pos->type) {
        case LLAMA_GRETYPE_END: return true;  // NOLINT
        case LLAMA_GRETYPE_ALT: return true;  // NOLINT
        default:                return false;
    }
}

// returns true iff chr satisfies the char range at pos (regular or inverse range)
// asserts that pos is pointing to a char range element
static std::pair<bool, const llama_grammar_element *> llama_grammar_match_char(
        const llama_grammar_element * pos,
        const uint32_t                chr) {

    bool found            = false;
    bool is_positive_char = pos->type == LLAMA_GRETYPE_CHAR;

    GGML_ASSERT(is_positive_char || pos->type == LLAMA_GRETYPE_CHAR_NOT); // NOLINT

    do {
        if (pos[1].type == LLAMA_GRETYPE_CHAR_RNG_UPPER) {
            // inclusive range, e.g. [a-z]
            found = found || (pos->value <= chr && chr <= pos[1].value);
            pos += 2;
        } else {
            // exact char match, e.g. [a] or "a"
            found = found || pos->value == chr;
            pos += 1;
        }
    } while (pos->type == LLAMA_GRETYPE_CHAR_ALT);

    return std::make_pair(found == is_positive_char, pos);
}

// returns true iff some continuation of the given partial UTF-8 sequence could satisfy the char
// range at pos (regular or inverse range)
// asserts that pos is pointing to a char range element
static bool llama_grammar_match_partial_char(
        const llama_grammar_element * pos,
        const llama_partial_utf8      partial_utf8) {

    bool is_positive_char = pos->type == LLAMA_GRETYPE_CHAR;
    GGML_ASSERT(is_positive_char || pos->type == LLAMA_GRETYPE_CHAR_NOT);

    uint32_t partial_value = partial_utf8.value;
    int      n_remain      = partial_utf8.n_remain;

    // invalid sequence or 7-bit char split across 2 bytes (overlong)
    if (n_remain < 0 || (n_remain == 1 && partial_value < 2)) {
        return false;
    }

    // range of possible code points this partial UTF-8 sequence could complete to
    uint32_t low  = partial_value << (n_remain * 6);
    uint32_t high = low | ((1 << (n_remain * 6)) - 1);

    if (low == 0) {
        if (n_remain == 2) {
            low = 1 << 11;
        } else if (n_remain == 3) {
            low = 1 << 16;
        }
    }

    do {
        if (pos[1].type == LLAMA_GRETYPE_CHAR_RNG_UPPER) {
            // inclusive range, e.g. [a-z]
            if (pos->value <= high && low <= pos[1].value) {
                return is_positive_char;
            }
            pos += 2;
        } else {
            // exact char match, e.g. [a] or "a"
            if (low <= pos->value && pos->value <= high) {
                return is_positive_char;
            }
            pos += 1;
        }
    } while (pos->type == LLAMA_GRETYPE_CHAR_ALT);

    return !is_positive_char;
}


// transforms a grammar pushdown stack into N possible stacks, all ending
// at a character range (terminal element)
static void llama_grammar_advance_stack(
        const std::vector<std::vector<llama_grammar_element>>   & rules,
        const std::vector<const llama_grammar_element *>        & stack,
        std::vector<std::vector<const llama_grammar_element *>> & new_stacks) {

    if (stack.empty()) {
        new_stacks.emplace_back(stack);
        return;
    }

    const llama_grammar_element * pos = stack.back();

    switch (pos->type) {
        case LLAMA_GRETYPE_RULE_REF: {
            const size_t                  rule_id = static_cast<size_t>(pos->value);
            const llama_grammar_element * subpos  = rules[rule_id].data();
            do {
                // init new stack without the top (pos)
                std::vector<const llama_grammar_element *> new_stack(stack.begin(), stack.end() - 1);
                if (!llama_grammar_is_end_of_sequence(pos + 1)) {
                    // if this rule ref is followed by another element, add that to stack
                    new_stack.push_back(pos + 1);
                }
                if (!llama_grammar_is_end_of_sequence(subpos)) {
                    // if alternate is nonempty, add to stack
                    new_stack.push_back(subpos);
                }
                llama_grammar_advance_stack(rules, new_stack, new_stacks);
                while (!llama_grammar_is_end_of_sequence(subpos)) {
                    // scan to end of alternate def
                    subpos++;
                }
                if (subpos->type == LLAMA_GRETYPE_ALT) {
                    // there's another alternate def of this rule to process
                    subpos++;
                } else {
                    break;
                }
            } while (true);
            break;
        }
        case LLAMA_GRETYPE_CHAR:
        case LLAMA_GRETYPE_CHAR_NOT:
            new_stacks.emplace_back(stack);
            break;
        default:
            // end of alternate (LLAMA_GRETYPE_END, LLAMA_GRETYPE_ALT) or middle of char range
            // (LLAMA_GRETYPE_CHAR_ALT, LLAMA_GRETYPE_CHAR_RNG_UPPER); stack should never be left on
            // those
            GGML_ASSERT(false);
    }
}

// takes a set of possible pushdown stacks on a grammar, which are required to
// be positioned at a character range (see `llama_grammar_advance_stack`), and
// produces the N possible stacks if the given char is accepted at those
// positions
static std::vector<std::vector<const llama_grammar_element *>> llama_grammar_accept(
        const std::vector<std::vector<llama_grammar_element>>         & rules,
        const std::vector<std::vector<const llama_grammar_element *>> & stacks,
        const uint32_t                                                  chr) {

    std::vector<std::vector<const llama_grammar_element *>> new_stacks;

    for (const auto & stack : stacks) {
        if (stack.empty()) {
            continue;
        }

        auto match = llama_grammar_match_char(stack.back(), chr);
        if (match.first) {
            const llama_grammar_element * pos = match.second;

            // update top of stack to next element, if any
            std::vector<const llama_grammar_element *> new_stack(stack.begin(), stack.end() - 1);
            if (!llama_grammar_is_end_of_sequence(pos)) {
                new_stack.push_back(pos);
            }
            llama_grammar_advance_stack(rules, new_stack, new_stacks);
        }
    }

    return new_stacks;
}

static std::vector<llama_grammar_candidate> llama_grammar_reject_candidates(
        const std::vector<std::vector<llama_grammar_element>>         & rules,
        const std::vector<std::vector<const llama_grammar_element *>> & stacks,
        const std::vector<llama_grammar_candidate>                    & candidates);

static std::vector<llama_grammar_candidate> llama_grammar_reject_candidates_for_stack(
        const std::vector<std::vector<llama_grammar_element>> & rules,
        const std::vector<const llama_grammar_element *>      & stack,
        const std::vector<llama_grammar_candidate>            & candidates) {

    std::vector<llama_grammar_candidate> rejects;

    if (stack.empty()) {
        for (const auto & tok : candidates) {
            if (*tok.code_points != 0 || tok.partial_utf8.n_remain != 0) {
                rejects.push_back(tok);
            }
        }
        return rejects;
    }

    const llama_grammar_element * stack_pos = stack.back();

    std::vector<llama_grammar_candidate> next_candidates;
    for (const auto & tok : candidates) {
        if (*tok.code_points == 0) {
            // reached end of full codepoints in token, reject iff it ended in a partial sequence
            // that cannot satisfy this position in grammar
            if (tok.partial_utf8.n_remain != 0 &&
                    !llama_grammar_match_partial_char(stack_pos, tok.partial_utf8)) {
                rejects.push_back(tok);
            }
        } else if (llama_grammar_match_char(stack_pos, *tok.code_points).first) {
            next_candidates.push_back({ tok.index, tok.code_points + 1, tok.partial_utf8 });
        } else {
            rejects.push_back(tok);
        }
    }

    const auto * stack_pos_after = llama_grammar_match_char(stack_pos, 0).second;

    // update top of stack to next element, if any
    std::vector<const llama_grammar_element *> stack_after(stack.begin(), stack.end() - 1);
    if (!llama_grammar_is_end_of_sequence(stack_pos_after)) {
        stack_after.push_back(stack_pos_after);
    }
    std::vector<std::vector<const llama_grammar_element *>> next_stacks;
    llama_grammar_advance_stack(rules, stack_after, next_stacks);

    auto next_rejects = llama_grammar_reject_candidates(rules, next_stacks, next_candidates);
    for (const auto & tok : next_rejects) {
        rejects.push_back({ tok.index, tok.code_points - 1, tok.partial_utf8 });
    }

    return rejects;
}

static std::vector<llama_grammar_candidate> llama_grammar_reject_candidates(
        const std::vector<std::vector<llama_grammar_element>>         & rules,
        const std::vector<std::vector<const llama_grammar_element *>> & stacks,
        const std::vector<llama_grammar_candidate>                    & candidates) {
    GGML_ASSERT(!stacks.empty()); // REVIEW

    if (candidates.empty()) {
        return std::vector<llama_grammar_candidate>();
    }

    auto rejects = llama_grammar_reject_candidates_for_stack(rules, stacks.front(), candidates);

    for (size_t i = 1, size = stacks.size(); i < size; ++i) {
        rejects = llama_grammar_reject_candidates_for_stack(rules, stacks[i], rejects);
    }
    return rejects;
}

//
// grammar - external
//

struct llama_grammar * llama_grammar_init(
            const llama_grammar_element ** rules,
                                 size_t    n_rules,
                                 size_t    start_rule_index) {
    const llama_grammar_element * pos;

    // copy rule definitions into vectors
    std::vector<std::vector<llama_grammar_element>> vec_rules(n_rules);
    for (size_t i = 0; i < n_rules; i++) {
        for (pos = rules[i]; pos->type != LLAMA_GRETYPE_END; pos++) {
            vec_rules[i].push_back(*pos);
        }
        vec_rules[i].push_back({LLAMA_GRETYPE_END, 0});
    }

    // loop over alternates of start rule to build initial stacks
    std::vector<std::vector<const llama_grammar_element *>> stacks;
    pos = rules[start_rule_index];
    do {
        std::vector<const llama_grammar_element *> stack;
        if (!llama_grammar_is_end_of_sequence(pos)) {
            // if alternate is nonempty, add to stack
            stack.push_back(pos);
        }
        llama_grammar_advance_stack(vec_rules, stack, stacks);
        while (!llama_grammar_is_end_of_sequence(pos)) {
            // scan to end of alternate def
            pos++;
        }
        if (pos->type == LLAMA_GRETYPE_ALT) {
            // there's another alternate def of this rule to process
            pos++;
        } else {
            break;
        }
    } while (true);

    return new llama_grammar{ std::move(vec_rules), std::move(stacks), {} };
}

void llama_grammar_free(struct llama_grammar * grammar) {
    delete grammar;
}

struct llama_grammar * llama_grammar_copy(const struct llama_grammar * grammar) {
    llama_grammar * result = new llama_grammar{ grammar->rules, grammar->stacks, grammar->partial_utf8 };

    // redirect elements in stacks to point to new rules
    for (size_t is = 0; is < result->stacks.size(); is++) {
        for (size_t ie = 0; ie < result->stacks[is].size(); ie++) {
            for (size_t ir0 = 0; ir0 < grammar->rules.size(); ir0++) {
                for (size_t ir1 = 0; ir1 < grammar->rules[ir0].size(); ir1++) {
                    if (grammar->stacks[is][ie] == &grammar->rules[ir0][ir1]) {
                         result->stacks[is][ie]  =  &result->rules[ir0][ir1];
                    }
                }
            }
        }
    }

    return result;
}

//
// sampling
//

void llama_set_rng_seed(struct llama_context * ctx, uint32_t seed) {
    if (seed == LLAMA_DEFAULT_SEED) {
        seed = time(NULL);
    }
    ctx->rng.seed(seed);
}

void llama_sample_softmax(struct llama_context * ctx, llama_token_data_array * candidates) {
    GGML_ASSERT(candidates->size > 0);

    const int64_t t_start_sample_us = ggml_time_us();

    // Sort the logits in descending order
    if (!candidates->sorted) {
        std::sort(candidates->data, candidates->data + candidates->size, [](const llama_token_data & a, const llama_token_data & b) {
            return a.logit > b.logit;
        });
        candidates->sorted = true;
    }

    float max_l = candidates->data[0].logit;
    float cum_sum = 0.0f;
    for (size_t i = 0; i < candidates->size; ++i) {
        float p = expf(candidates->data[i].logit - max_l);
        candidates->data[i].p = p;
        cum_sum += p;
    }
    for (size_t i = 0; i < candidates->size; ++i) {
        candidates->data[i].p /= cum_sum;
    }

    if (ctx) {
        ctx->t_sample_us += ggml_time_us() - t_start_sample_us;
    }
}

void llama_sample_top_k(struct llama_context * ctx, llama_token_data_array * candidates, int k, size_t min_keep) {
    const int64_t t_start_sample_us = ggml_time_us();

    k = std::max(k, (int) min_keep);
    k = std::min(k, (int) candidates->size);

    // Sort scores in descending order
    if (!candidates->sorted) {
        auto comp = [](const llama_token_data & a, const llama_token_data & b) {
            return a.logit > b.logit;
        };
        if (k == (int) candidates->size) {
            std::sort(candidates->data, candidates->data + candidates->size, comp);
        } else {
            std::partial_sort(candidates->data, candidates->data + k, candidates->data + candidates->size, comp);
        }
        candidates->sorted = true;
    }
    candidates->size = k;

    if (ctx) {
        ctx->t_sample_us += ggml_time_us() - t_start_sample_us;
    }
}

void llama_sample_top_p(struct llama_context * ctx, llama_token_data_array * candidates, float p, size_t min_keep) {
    if (p >= 1.0f) {
        return;
    }

    llama_sample_softmax(ctx, candidates);

    const int64_t t_start_sample_us = ggml_time_us();

    // Compute the cumulative probabilities
    float cum_sum = 0.0f;
    size_t last_idx = candidates->size;

    for (size_t i = 0; i < candidates->size; ++i) {
        cum_sum += candidates->data[i].p;

        // Check if the running sum is at least p or if we have kept at least min_keep tokens
        // we set the last index to i+1 to indicate that the current iterate should be included in the set
        if (cum_sum >= p && i + 1 >= min_keep) {
            last_idx = i + 1;
            break;
        }
    }

    // Resize the output vector to keep only the top-p tokens
    candidates->size = last_idx;

    if (ctx) {
        ctx->t_sample_us += ggml_time_us() - t_start_sample_us;
    }
}

void llama_sample_tail_free(struct llama_context * ctx, llama_token_data_array * candidates, float z, size_t min_keep) {
    if (z >= 1.0f || candidates->size <= 2) {
        return;
    }

    llama_sample_softmax(nullptr, candidates);
    const int64_t t_start_sample_us = ggml_time_us();

    // Compute the first and second derivatives
    std::vector<float> first_derivatives(candidates->size - 1);
    std::vector<float> second_derivatives(candidates->size - 2);

    for (size_t i = 0; i < first_derivatives.size(); ++i) {
        first_derivatives[i] = candidates->data[i].p - candidates->data[i + 1].p;
    }
    for (size_t i = 0; i < second_derivatives.size(); ++i) {
        second_derivatives[i] = first_derivatives[i] - first_derivatives[i + 1];
    }

    // Calculate absolute value of second derivatives
    for (size_t i = 0; i < second_derivatives.size(); ++i) {
        second_derivatives[i] = std::abs(second_derivatives[i]);
    }

    // Normalize the second derivatives
    {
        const float second_derivatives_sum = std::accumulate(second_derivatives.begin(), second_derivatives.end(), 0.0f);

        if (second_derivatives_sum > 1e-6f) {
            for (float & value : second_derivatives) {
                value /= second_derivatives_sum;
            }
        } else {
            for (float & value : second_derivatives) {
                value = 1.0f / second_derivatives.size();
            }
        }
    }

    float cum_sum = 0.0f;
    size_t last_idx = candidates->size;
    for (size_t i = 0; i < second_derivatives.size(); ++i) {
        cum_sum += second_derivatives[i];

        // Check if the running sum is greater than z or if we have kept at least min_keep tokens
        if (cum_sum > z && i >= min_keep) {
            last_idx = i;
            break;
        }
    }

    // Resize the output vector to keep only the tokens above the tail location
    candidates->size = last_idx;

    if (ctx) {
        ctx->t_sample_us += ggml_time_us() - t_start_sample_us;
    }
}

void llama_sample_typical(struct llama_context * ctx, llama_token_data_array * candidates, float p, size_t min_keep) {
    // Reference implementation:
    // https://github.com/huggingface/transformers/compare/main...cimeister:typical-sampling:typical-pr
    if (p >= 1.0f) {
        return;
    }

    // Compute the softmax of logits and calculate entropy
    llama_sample_softmax(nullptr, candidates);

    const int64_t t_start_sample_us = ggml_time_us();

    float entropy = 0.0f;
    for (size_t i = 0; i < candidates->size; ++i) {
        entropy += -candidates->data[i].p * logf(candidates->data[i].p);
    }

    // Compute the absolute difference between negative log probability and entropy for each candidate
    std::vector<float> shifted_scores;
    for (size_t i = 0; i < candidates->size; ++i) {
        float shifted_score = fabsf(-logf(candidates->data[i].p) - entropy);
        shifted_scores.push_back(shifted_score);
    }

    // Sort tokens based on the shifted_scores and their corresponding indices
    std::vector<size_t> indices(candidates->size);
    std::iota(indices.begin(), indices.end(), 0);

    std::sort(indices.begin(), indices.end(), [&](size_t a, size_t b) {
        return shifted_scores[a] < shifted_scores[b];
    });

    // Compute the cumulative probabilities
    float cum_sum = 0.0f;
    size_t last_idx = indices.size();

    for (size_t i = 0; i < indices.size(); ++i) {
        size_t idx = indices[i];
        cum_sum += candidates->data[idx].p;

        // Check if the running sum is greater than typical or if we have kept at least min_keep tokens
        if (cum_sum > p && i >= min_keep - 1) {
            last_idx = i + 1;
            break;
        }
    }

    // Resize the output vector to keep only the locally typical tokens
    std::vector<llama_token_data> new_candidates;
    for (size_t i = 0; i < last_idx; ++i) {
        size_t idx = indices[i];
        new_candidates.push_back(candidates->data[idx]);
    }

    // Replace the data in candidates with the new_candidates data
    std::copy(new_candidates.begin(), new_candidates.end(), candidates->data);
    candidates->size = new_candidates.size();

    if (ctx) {
        ctx->t_sample_us += ggml_time_us() - t_start_sample_us;
    }
}

void llama_sample_temp(struct llama_context * ctx, llama_token_data_array * candidates_p, float temp) {
    const int64_t t_start_sample_us = ggml_time_us();

    for (size_t i = 0; i < candidates_p->size; ++i) {
        candidates_p->data[i].logit /= temp;
    }

    if (ctx) {
        ctx->t_sample_us += ggml_time_us() - t_start_sample_us;
    }
}

void llama_sample_temperature(struct llama_context * ctx, llama_token_data_array * candidates_p, float temp) {
    llama_sample_temp(ctx, candidates_p, temp);
}

void llama_sample_repetition_penalties(
            struct llama_context * ctx,
          llama_token_data_array * candidates,
               const llama_token * last_tokens,
                          size_t   penalty_last_n,
                           float   penalty_repeat,
                           float   penalty_freq,
                           float   penalty_present) {
    if (penalty_last_n == 0 || (penalty_repeat == 1.0f && penalty_freq == 0.0f && penalty_present == 0.0f)) {
        return;
    }

    const int64_t t_start_sample_us = ggml_time_us();

    // Create a frequency map to count occurrences of each token in last_tokens
    std::unordered_map<llama_token, int> token_count;
    for (size_t i = 0; i < penalty_last_n; ++i) {
        token_count[last_tokens[i]]++;
    }

    // Apply frequency and presence penalties to the candidates
    for (size_t i = 0; i < candidates->size; ++i) {
        const auto token_iter = token_count.find(candidates->data[i].id);
        if (token_iter == token_count.end()) {
            continue;
        }

        const int count = token_iter->second;

        // The academic publication that described this technique actually just only divided, but that would cause tokens with negative logits to become more likely, which is obviously wrong.
        // This is common fix for this problem, which is to multiply by the penalty instead of dividing.
        if (candidates->data[i].logit <= 0) {
            candidates->data[i].logit *= penalty_repeat;
        } else {
            candidates->data[i].logit /= penalty_repeat;
        }

        candidates->data[i].logit -= float(count) * penalty_freq + float(count > 0) * penalty_present;
    }

    candidates->sorted = false;

    if (ctx) {
        ctx->t_sample_us += ggml_time_us() - t_start_sample_us;
    }
}

void llama_sample_grammar(struct llama_context * ctx, llama_token_data_array * candidates, const struct llama_grammar * grammar) {
    GGML_ASSERT(ctx);
    const int64_t t_start_sample_us = ggml_time_us();

    bool allow_eos = false;
    for (const auto & stack : grammar->stacks) {
        if (stack.empty()) {
            allow_eos = true;
            break;
        }
    }

    const llama_token eos = llama_token_eos(ctx);

    std::vector<std::pair<std::vector<uint32_t>, llama_partial_utf8>> candidates_decoded;
    std::vector<llama_grammar_candidate>                              candidates_grammar;

    for (size_t i = 0; i < candidates->size; ++i) {
        const llama_token id    = candidates->data[i].id;
        const std::string piece = llama_token_to_str(ctx, id);
        if (id == eos) {
            if (!allow_eos) {
                candidates->data[i].logit = -INFINITY;
            }
        } else if (piece.empty() || piece[0] == 0) {
            candidates->data[i].logit = -INFINITY;
        } else {
            candidates_decoded.push_back(decode_utf8(piece.c_str(), grammar->partial_utf8));
            candidates_grammar.push_back({ i, candidates_decoded.back().first.data(), candidates_decoded.back().second });
        }
    }

    const auto rejects = llama_grammar_reject_candidates(grammar->rules, grammar->stacks, candidates_grammar);
    for (const auto & reject : rejects) {
        candidates->data[reject.index].logit = -INFINITY;
    }

    ctx->t_sample_us += ggml_time_us() - t_start_sample_us;
}

static void llama_log_softmax(float * array, size_t size) {
    float max_l = *std::max_element(array, array + size);
    float sum = 0.f;
    for (size_t i = 0; i < size; ++i) {
        float p = expf(array[i] - max_l);
        sum += p;
        array[i] = p;
    }

    for (size_t i = 0; i < size; ++i) {
        array[i] = logf(array[i] / sum);
    }
}

void llama_sample_classifier_free_guidance(
          struct llama_context * ctx,
        llama_token_data_array * candidates,
          struct llama_context * guidance_ctx,
                         float   scale) {
    int64_t t_start_sample_us = ggml_time_us();

    GGML_ASSERT(ctx);

    auto n_vocab = llama_n_vocab(llama_get_model(ctx));

    GGML_ASSERT(n_vocab == (int)candidates->size);
    GGML_ASSERT(!candidates->sorted);

    std::vector<float> logits_base;
    logits_base.reserve(candidates->size);
    for (size_t i = 0; i < candidates->size; ++i) {
        logits_base.push_back(candidates->data[i].logit);
    }
    llama_log_softmax(logits_base.data(), candidates->size);

    float* logits_guidance = llama_get_logits(guidance_ctx);
    llama_log_softmax(logits_guidance, n_vocab);

    for (int i = 0; i < n_vocab; ++i) {
        float logit_guidance = logits_guidance[i];
        float logit_base = logits_base[i];
        candidates->data[i].logit = scale * (logit_base - logit_guidance) + logit_guidance;
    }

    if (ctx) {
        ctx->t_sample_us += ggml_time_us() - t_start_sample_us;
    }
}

llama_token llama_sample_token_mirostat(struct llama_context * ctx, llama_token_data_array * candidates, float tau, float eta, int m, float * mu) {
    GGML_ASSERT(ctx);

    auto N = float(llama_n_vocab(llama_get_model(ctx)));
    int64_t t_start_sample_us;
    t_start_sample_us = ggml_time_us();

    llama_sample_softmax(nullptr, candidates);

    // Estimate s_hat using the most probable m tokens
    float s_hat = 0.0;
    float sum_ti_bi = 0.0;
    float sum_ti_sq = 0.0;
    for (size_t i = 0; i < size_t(m - 1) && i < candidates->size - 1; ++i) {
        float t_i = logf(float(i + 2) / float(i + 1));
        float b_i = logf(candidates->data[i].p / candidates->data[i + 1].p);
        sum_ti_bi += t_i * b_i;
        sum_ti_sq += t_i * t_i;
    }
    s_hat = sum_ti_bi / sum_ti_sq;

    // Compute k from the estimated s_hat and target surprise value
    float epsilon_hat = s_hat - 1;
    float k = powf((epsilon_hat * powf(2, *mu)) / (1 - powf(N, -epsilon_hat)), 1 / s_hat);

    // Sample the next word X using top-k sampling
    llama_sample_top_k(nullptr, candidates, int(k), 1);
    if (ctx) {
        ctx->t_sample_us += ggml_time_us() - t_start_sample_us;
    }
    llama_token X = llama_sample_token(ctx, candidates);
    t_start_sample_us = ggml_time_us();

    // Compute error as the difference between observed surprise and target surprise value
    size_t X_idx = std::distance(candidates->data, std::find_if(candidates->data, candidates->data + candidates->size, [&](const llama_token_data & candidate) {
        return candidate.id == X;
    }));
    float observed_surprise = -log2f(candidates->data[X_idx].p);
    float e = observed_surprise - tau;

    // Update mu using the learning rate and error
    *mu = *mu - eta * e;

    if (ctx) {
        ctx->t_sample_us += ggml_time_us() - t_start_sample_us;
    }
    return X;
}

llama_token llama_sample_token_mirostat_v2(struct llama_context * ctx, llama_token_data_array * candidates, float tau, float eta, float * mu) {
    int64_t t_start_sample_us;
    t_start_sample_us = ggml_time_us();

    llama_sample_softmax(ctx, candidates);

    // Truncate the words with surprise values greater than mu
    candidates->size = std::distance(candidates->data, std::find_if(candidates->data, candidates->data + candidates->size, [&](const llama_token_data & candidate) {
        return -log2f(candidate.p) > *mu;
    }));

    if (candidates->size == 0) {
        candidates->size = 1;
    }

    if (ctx) {
        ctx->t_sample_us += ggml_time_us() - t_start_sample_us;
    }

    // Normalize the probabilities of the remaining words
    llama_sample_softmax(ctx, candidates);

    // Sample the next word X from the remaining words
    llama_token X = llama_sample_token(ctx, candidates);
    t_start_sample_us = ggml_time_us();

    // Compute error as the difference between observed surprise and target surprise value
    size_t X_idx = std::distance(candidates->data, std::find_if(candidates->data, candidates->data + candidates->size, [&](const llama_token_data & candidate) {
        return candidate.id == X;
    }));
    float observed_surprise = -log2f(candidates->data[X_idx].p);
    float e = observed_surprise - tau;

    // Update mu using the learning rate and error
    *mu = *mu - eta * e;

    if (ctx) {
        ctx->t_sample_us += ggml_time_us() - t_start_sample_us;
    }
    return X;
}

llama_token llama_sample_token_greedy(struct llama_context * ctx, llama_token_data_array * candidates) {
    const int64_t t_start_sample_us = ggml_time_us();

    // Find max element
    auto * max_iter = std::max_element(candidates->data, candidates->data + candidates->size, [](const llama_token_data & a, const llama_token_data & b) {
        return a.logit < b.logit;
    });

    llama_token result = max_iter->id;
    if (ctx) {
        ctx->t_sample_us += ggml_time_us() - t_start_sample_us;
        ctx->n_sample++;
    }
    return result;
}

llama_token llama_sample_token(struct llama_context * ctx, llama_token_data_array * candidates) {
    GGML_ASSERT(ctx);

    const int64_t t_start_sample_us = ggml_time_us();
    llama_sample_softmax(nullptr, candidates);

    std::vector<float> probs;
    probs.reserve(candidates->size);
    for (size_t i = 0; i < candidates->size; ++i) {
        probs.push_back(candidates->data[i].p);
    }

    std::discrete_distribution<> dist(probs.begin(), probs.end());
    auto & rng = ctx->rng;
    int idx = dist(rng);

    llama_token result = candidates->data[idx].id;

    ctx->t_sample_us += ggml_time_us() - t_start_sample_us;
    ctx->n_sample++;
    return result;
}

void llama_grammar_accept_token(struct llama_context * ctx, struct llama_grammar * grammar, llama_token token) {
    const int64_t t_start_sample_us = ggml_time_us();

    if (token == llama_token_eos(ctx)) {
        for (const auto & stack : grammar->stacks) {
            if (stack.empty()) {
                return;
            }
        }
        GGML_ASSERT(false);
    }

    const std::string piece = llama_token_to_str(ctx, token);

    // Note terminating 0 in decoded string
    const auto   decoded     = decode_utf8(piece.c_str(), grammar->partial_utf8);
    const auto & code_points = decoded.first;
    for (auto it = code_points.begin(), end = code_points.end() - 1; it != end; ++it) {
        grammar->stacks = llama_grammar_accept(grammar->rules, grammar->stacks, *it);
    }
    grammar->partial_utf8 = decoded.second;
    GGML_ASSERT(!grammar->stacks.empty());

    ctx->t_sample_us += ggml_time_us() - t_start_sample_us;
}

//
// Beam search
//

struct llama_beam {
    std::vector<llama_token> tokens;
    float p;  // Cumulative beam probability (renormalized relative to all beams)
    bool eob; // Initialize end-of-beam to false. Callback sets this to true.
    // Sort beams by probability. In case of ties, prefer beams at eob.
    bool operator<(const llama_beam & rhs) const {
        return std::make_pair(p, eob) < std::make_pair(rhs.p, rhs.eob);
    }
    // Shift off first n tokens and discard them.
    void shift_tokens(const size_t n) {
        if (n) {
            std::copy(tokens.begin() + n, tokens.end(), tokens.begin());
            tokens.resize(tokens.size() - n);
        }
    }
    llama_beam_view view() const { return {tokens.data(), tokens.size(), p, eob}; }
};

// A struct for calculating logit-related info.
struct llama_logit_info {
    const float * const logits;
    const int n_vocab;
    const float max_l;
    const float normalizer;
    struct sum_exp {
        float max_l;
        float operator()(float sum, float l) const { return sum + std::exp(l - max_l); }
    };
    llama_logit_info(llama_context * ctx)
      : logits(llama_get_logits(ctx))
      , n_vocab(llama_n_vocab(llama_get_model(ctx)))
      , max_l(*std::max_element(logits, logits + n_vocab))
      , normalizer(1.0f / std::accumulate(logits, logits + n_vocab, 0.0f, sum_exp{max_l}))
      { }
    llama_token_data get_token_data(const llama_token token_id) const {
        constexpr auto p = std::numeric_limits<float>::quiet_NaN();  // never used
        return {token_id, logits[token_id], p};
    }
    // Return top k token_data by logit.
    std::vector<llama_token_data> top_k(size_t k) {
        std::vector<llama_token_data> min_heap;  // min-heap by logit
        const llama_token k_min = std::min(static_cast<llama_token>(k), n_vocab);
        min_heap.reserve(k_min);
        for (llama_token token_id = 0 ; token_id < k_min ; ++token_id) {
            min_heap.push_back(get_token_data(token_id));
        }
        auto comp = [](const llama_token_data & a, const llama_token_data & b) { return a.logit > b.logit; };
        std::make_heap(min_heap.begin(), min_heap.end(), comp);
        for (llama_token token_id = k_min ; token_id < n_vocab ; ++token_id) {
            if (min_heap.front().logit < logits[token_id]) {
                std::pop_heap(min_heap.begin(), min_heap.end(), comp);
                min_heap.back().id = token_id;
                min_heap.back().logit = logits[token_id];
                std::push_heap(min_heap.begin(), min_heap.end(), comp);
            }
        }
        return min_heap;
    }
    float probability_from_logit(float logit) const {
        return normalizer * std::exp(logit - max_l);
    }
};

struct llama_beam_search_data {
    llama_context * ctx;
    size_t n_beams;
    int n_past;
    int n_predict;
    std::vector<llama_beam> beams;
    std::vector<llama_beam> next_beams;

    // Re-calculated on each loop iteration
    size_t common_prefix_length;

    // Used to communicate to/from callback on beams state.
    std::vector<llama_beam_view> beam_views;

    llama_beam_search_data(llama_context * ctx, size_t n_beams, int n_past, int n_predict)
      : ctx(ctx)
      , n_beams(n_beams)
      , n_past(n_past)
      , n_predict(n_predict)
      , beam_views(n_beams) {
        beams.reserve(n_beams);
        next_beams.reserve(n_beams);
    }

    // Collapse beams to a single beam given by index.
    void collapse_beams(const size_t beam_idx) {
        if (0u < beam_idx) {
            std::swap(beams[0], beams[beam_idx]);
        }
        beams.resize(1);
    }

    // Min-heaps are used to efficiently collect the top-k elements (k=n_beams).
    // The repetative patterns below reflect the 2 stages of heaps:
    //  * Gather elements until the vector is full, then call std::make_heap() on it.
    //  * If the heap is full and a new element is found that should be included, pop the
    //    least element to the back(), replace it with the new, then push it into the heap.
    void fill_next_beams_by_top_probabilities(llama_beam & beam) {
        // Min-heaps use a greater-than comparator.
        const auto comp = [](const llama_beam & a, const llama_beam & b) { return a.p > b.p; };
        if (beam.eob) {
            // beam is at end-of-sentence, so just copy it to next_beams if its probability is high enough.
            if (next_beams.size() < n_beams) {
                next_beams.push_back(std::move(beam));
                if (next_beams.size() == n_beams) {
                    std::make_heap(next_beams.begin(), next_beams.end(), comp);
                }
            } else if (next_beams.front().p < beam.p) {
                std::pop_heap(next_beams.begin(), next_beams.end(), comp);
                next_beams.back() = std::move(beam);
                std::push_heap(next_beams.begin(), next_beams.end(), comp);
            }
        } else {
            // beam is not at end-of-sentence, so branch with next top_k tokens.
            if (!beam.tokens.empty()) {
                llama_decode(ctx, llama_batch_get_one(beam.tokens.data(), beam.tokens.size(), n_past, 0));
            }
            llama_logit_info logit_info(ctx);
            std::vector<llama_token_data> next_tokens = logit_info.top_k(n_beams);
            size_t i=0;
            if (next_beams.size() < n_beams) {
                for (; next_beams.size() < n_beams ; ++i) {
                    llama_beam next_beam = beam;
                    next_beam.tokens.push_back(next_tokens[i].id);
                    next_beam.p *= logit_info.probability_from_logit(next_tokens[i].logit);
                    next_beams.push_back(std::move(next_beam));
                }
                std::make_heap(next_beams.begin(), next_beams.end(), comp);
            } else {
                for (; next_beams.front().p == 0.0f ; ++i) {
                    std::pop_heap(next_beams.begin(), next_beams.end(), comp);
                    next_beams.back() = beam;
                    next_beams.back().tokens.push_back(next_tokens[i].id);
                    next_beams.back().p *= logit_info.probability_from_logit(next_tokens[i].logit);
                    std::push_heap(next_beams.begin(), next_beams.end(), comp);
                }
            }
            for (; i < n_beams ; ++i) {
                const float next_p = beam.p * logit_info.probability_from_logit(next_tokens[i].logit);
                if (next_beams.front().p < next_p) {
                    std::pop_heap(next_beams.begin(), next_beams.end(), comp);
                    next_beams.back() = beam;
                    next_beams.back().tokens.push_back(next_tokens[i].id);
                    next_beams.back().p = next_p;
                    std::push_heap(next_beams.begin(), next_beams.end(), comp);
                }
            }
        }
    }

    // Find common_prefix_length based on beams.
    // Requires beams is not empty.
    size_t find_common_prefix_length() {
        size_t common_prefix_length = beams[0].tokens.size();
        for (size_t i = 1 ; i < beams.size() ; ++i) {
            common_prefix_length = std::min(common_prefix_length, beams[i].tokens.size());
            for (size_t j = 0 ; j < common_prefix_length ; ++j) {
                if (beams[0].tokens[j] != beams[i].tokens[j]) {
                    common_prefix_length = j;
                    break;
                }
            }
        }
        return common_prefix_length;
    }

    // Construct beams_state to send back to caller via the callback function.
    // Side effect: set common_prefix_length = find_common_prefix_length();
    llama_beams_state get_beams_state(const bool last_call) {
        for (size_t i = 0 ; i < beams.size() ; ++i) {
            beam_views[i] = beams[i].view();
        }
        common_prefix_length = find_common_prefix_length();
        return {beam_views.data(), beams.size(), common_prefix_length, last_call};
    }

    // Loop:
    //  * while i < n_predict, AND
    //  * any of the beams have not yet reached end-of-beam (eob), AND
    //  * the highest probability beam(s) (plural in case of ties) are not at end-of-sentence
    //    (since all other beam probabilities can only decrease)
    void loop(const llama_beam_search_callback_fn_t callback, void * const callback_data) {
        beams.push_back({{}, 1.0f, false});  // Start with one empty beam w/ probability = 1.0 and !eob.
        const auto not_eob = [](const llama_beam & beam) { return !beam.eob; };
        for (int i = 0 ; i < n_predict && std::any_of(beams.begin(),beams.end(),not_eob) &&
                       !beams[top_beam_index()].eob ; ++i) {
            callback(callback_data, get_beams_state(false));  // Sets common_prefix_length
            update_beams_from_beam_views();   // Update values (p,eob) that callback may have changed.
            if (common_prefix_length) {
                llama_decode(ctx, llama_batch_get_one(beams[0].tokens.data(), common_prefix_length, n_past, 0));
                n_past += common_prefix_length;
            }
            // Zero-out next_beam probabilities to place them last in following min-heap.
            std::for_each(next_beams.begin(), next_beams.end(), [](llama_beam & beam) { beam.p = 0.0f; });
            for (llama_beam & beam : beams) {
                beam.shift_tokens(common_prefix_length);
                fill_next_beams_by_top_probabilities(beam);
            }
            // next_beams become the beams of next/final iteration. Swap them to re-use memory.
            beams.swap(next_beams);
            renormalize_beam_probabilities(beams);
        }
        collapse_beams(top_beam_index());
        callback(callback_data, get_beams_state(true));
    }

    // As beams grow, the cumulative probabilities decrease.
    // Renormalize them to avoid floating point underflow.
    static void renormalize_beam_probabilities(std::vector<llama_beam> & beams) {
        const auto sum_p = [](float sum, llama_beam & beam) { return sum + beam.p; };
        const float inv_sum = 1.0f / std::accumulate(beams.begin(), beams.end(), 0.0f, sum_p);
        std::for_each(beams.begin(), beams.end(), [=](llama_beam & beam) { beam.p *= inv_sum; });
    }

    // Assumes beams is non-empty.  Uses llama_beam::operator<() for ordering.
    size_t top_beam_index() {
        return std::max_element(beams.begin(), beams.end()) - beams.begin();
    }

    // Copy (p,eob) for each beam which may have been changed by the callback.
    void update_beams_from_beam_views() {
        for (size_t i = 0 ; i < beams.size() ; ++i) {
            beams[i].p = beam_views[i].p;
            beams[i].eob = beam_views[i].eob;
        }
    }
};

void llama_beam_search(llama_context * ctx,
                       llama_beam_search_callback_fn_t callback, void * callback_data,
                       size_t n_beams, int n_past, int n_predict) {
    assert(ctx);
    const int64_t t_start_sample_us = ggml_time_us();

    llama_beam_search_data beam_search_data(ctx, n_beams, n_past, n_predict);

    beam_search_data.loop(callback, callback_data);

    ctx->t_sample_us += ggml_time_us() - t_start_sample_us;
    ctx->n_sample++;
}

//
// quantization
//

template <typename T>
struct no_init {
    T value;
    no_init() { /* do nothing */ }
};

static void llama_convert_tensor_internal(
    struct ggml_tensor * tensor, std::vector<no_init<float>> & output, std::vector<std::thread> & workers,
    const size_t nelements, const int nthread
) {
    if (output.size() < nelements) {
        output.resize(nelements);
    }
    float * f32_output = (float *) output.data();

    ggml_type_traits_t qtype;
    if (ggml_is_quantized(tensor->type)) {
        qtype = ggml_internal_get_type_traits(tensor->type);
        if (qtype.to_float == NULL) {
            throw std::runtime_error(format("type %s unsupported for integer quantization: no dequantization available", ggml_type_name(tensor->type)));
        }
    } else if (tensor->type != GGML_TYPE_F16) {
        throw std::runtime_error(format("cannot dequantize/convert tensor type %s", ggml_type_name(tensor->type)));
    }

    if (nthread < 2) {
        if (tensor->type == GGML_TYPE_F16) {
            ggml_fp16_to_fp32_row((ggml_fp16_t *)tensor->data, f32_output, nelements);
        } else if (ggml_is_quantized(tensor->type)) {
            qtype.to_float(tensor->data, f32_output, nelements);
        } else {
            GGML_ASSERT(false); // unreachable
        }
        return;
    }

    auto block_size = tensor->type == GGML_TYPE_F16 ? 1 : (size_t)ggml_blck_size(tensor->type);
    auto block_size_bytes = ggml_type_size(tensor->type);

    GGML_ASSERT(nelements % block_size == 0);
    auto nblocks = nelements / block_size;
    auto blocks_per_thread = nblocks / nthread;
    auto spare_blocks = nblocks - (blocks_per_thread * nthread); // if blocks aren't divisible by thread count

    for (auto tnum = 0, in_buff_offs = 0, out_buff_offs = 0; tnum < nthread; tnum++) {
        auto thr_blocks = blocks_per_thread + (tnum == nthread - 1 ? spare_blocks : 0); // num blocks for this thread
        auto thr_elems = thr_blocks * block_size; // number of elements for this thread
        auto thr_block_bytes = thr_blocks * block_size_bytes; // number of input bytes for this thread

        auto compute = [qtype] (ggml_type typ, uint8_t * inbuf, float * outbuf, int nels) {
            if (typ == GGML_TYPE_F16) {
                ggml_fp16_to_fp32_row((ggml_fp16_t *)inbuf, outbuf, nels);
            } else {
                qtype.to_float(inbuf, outbuf, nels);
            }
        };
        workers.emplace_back(compute, tensor->type, (uint8_t *) tensor->data + in_buff_offs, f32_output + out_buff_offs, thr_elems);
        in_buff_offs += thr_block_bytes;
        out_buff_offs += thr_elems;
    }
    for (auto & w : workers) { w.join(); }
    workers.clear();
}

#ifdef GGML_USE_K_QUANTS
static ggml_type get_k_quant_type(
    ggml_type new_type, const ggml_tensor * tensor, const llama_model & model, llama_ftype ftype, int * i_attention_wv,
    int n_attention_wv, int * i_feed_forward_w2, int n_feed_forward_w2
) {
    const std::string name = ggml_get_name(tensor);
    // TODO: avoid hardcoded tensor names - use the TN_* constants
    const auto tn = LLM_TN(model.arch);

    auto use_more_bits = [](int i_layer, int num_layers) -> bool {
        return i_layer < num_layers/8 || i_layer >= 7*num_layers/8 || (i_layer - num_layers/8)%3 == 2;
    };

    if (name == tn(LLM_TENSOR_OUTPUT, "weight")) {
        int nx = tensor->ne[0];
        if (model.arch == LLM_ARCH_FALCON || nx % QK_K != 0) {
            new_type = GGML_TYPE_Q8_0;
        }
        else if (new_type != GGML_TYPE_Q8_0) {
            new_type = GGML_TYPE_Q6_K;
        }
    } else if (name.find("attn_v.weight") != std::string::npos) {
        if      (ftype == LLAMA_FTYPE_MOSTLY_Q2_K) new_type = GGML_TYPE_Q3_K;
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q3_K_M) {
            new_type = *i_attention_wv < 2 ? GGML_TYPE_Q5_K : GGML_TYPE_Q4_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q3_K_L) new_type = GGML_TYPE_Q5_K;
        else if ((ftype == LLAMA_FTYPE_MOSTLY_Q4_K_M || ftype == LLAMA_FTYPE_MOSTLY_Q5_K_M) &&
                use_more_bits(*i_attention_wv, n_attention_wv)) new_type = GGML_TYPE_Q6_K;
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q4_K_S && *i_attention_wv < 4) new_type = GGML_TYPE_Q5_K;
        else if (QK_K == 64 && (ftype == LLAMA_FTYPE_MOSTLY_Q4_K_S || ftype == LLAMA_FTYPE_MOSTLY_Q3_K_S) &&
                (*i_attention_wv < n_attention_wv/8 || *i_attention_wv >= 7*n_attention_wv/8)) new_type = GGML_TYPE_Q6_K;
        if (model.type == MODEL_70B) {
            // In the 70B model we have 8 heads sharing the same attn_v weights. As a result, the attn_v.weight tensor is
            // 8x smaller compared to attn_q.weight. Hence, we can get a nice boost in quantization accuracy with
            // nearly negligible increase in model size by quantizing this tensor with more bits:
            if (new_type == GGML_TYPE_Q3_K || new_type == GGML_TYPE_Q4_K) new_type = GGML_TYPE_Q5_K;
        }
        ++*i_attention_wv;
    } else if (name.find("ffn_down.weight") != std::string::npos) {
        if      (ftype == LLAMA_FTYPE_MOSTLY_Q2_K) new_type = GGML_TYPE_Q3_K;
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q3_K_M) {
            new_type = *i_feed_forward_w2 < 2 ? GGML_TYPE_Q5_K
                     : model.arch != LLM_ARCH_FALCON || use_more_bits(*i_feed_forward_w2, n_feed_forward_w2) ? GGML_TYPE_Q4_K
                     : GGML_TYPE_Q3_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q3_K_L) {
            new_type = model.arch == LLM_ARCH_FALCON ? GGML_TYPE_Q4_K : GGML_TYPE_Q5_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q4_K_M) {
            if (model.arch == LLM_ARCH_FALCON) {
                new_type = *i_feed_forward_w2 < 2 ? GGML_TYPE_Q6_K :
                           use_more_bits(*i_feed_forward_w2, n_feed_forward_w2) ? GGML_TYPE_Q5_K : GGML_TYPE_Q4_K;
            } else {
                if (use_more_bits(*i_feed_forward_w2, n_feed_forward_w2)) new_type = GGML_TYPE_Q6_K;
            }
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q5_K_M && use_more_bits(*i_feed_forward_w2, n_feed_forward_w2)) new_type = GGML_TYPE_Q6_K;
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q4_K_S && model.arch != LLM_ARCH_FALCON && *i_feed_forward_w2 < 4) {
            new_type = GGML_TYPE_Q5_K;
        }
        ++*i_feed_forward_w2;
    } else if (name.find("attn_output.weight") != std::string::npos) {
        if (model.arch != LLM_ARCH_FALCON) {
            if      (ftype == LLAMA_FTYPE_MOSTLY_Q2_K  ) new_type = GGML_TYPE_Q3_K;
            else if (ftype == LLAMA_FTYPE_MOSTLY_Q3_K_M) new_type = GGML_TYPE_Q4_K;
            else if (ftype == LLAMA_FTYPE_MOSTLY_Q3_K_L) new_type = GGML_TYPE_Q5_K;
        } else {
            if (ftype == LLAMA_FTYPE_MOSTLY_Q3_K_L) new_type = GGML_TYPE_Q4_K;
        }
    }
    else if (name.find("attn_qkv.weight") != std::string::npos) {
        if (ftype == LLAMA_FTYPE_MOSTLY_Q3_K_M || ftype == LLAMA_FTYPE_MOSTLY_Q3_K_L) new_type = GGML_TYPE_Q4_K;
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q4_K_M) new_type = GGML_TYPE_Q5_K;
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q5_K_M) new_type = GGML_TYPE_Q6_K;
    }
    else if (name.find("ffn_gate.weight") != std::string::npos || name.find("ffn_up.weight") != std::string::npos) {
        if (ftype == LLAMA_FTYPE_MOSTLY_Q2_K) new_type = GGML_TYPE_Q3_K;
    }
    // This can be used to reduce the size of the Q5_K_S model.
    // The associated PPL increase is fully in line with the size reduction
    //else {
    //    if (ftype == LLAMA_FTYPE_MOSTLY_Q5_K_S) new_type = GGML_TYPE_Q4_K;
    //}
    bool convert_incompatible_tensor = false;
    if (new_type == GGML_TYPE_Q2_K || new_type == GGML_TYPE_Q3_K || new_type == GGML_TYPE_Q4_K ||
        new_type == GGML_TYPE_Q5_K || new_type == GGML_TYPE_Q6_K) {
        int nx = tensor->ne[0];
        int ny = tensor->ne[1];
        if (nx % QK_K != 0) {
            LLAMA_LOG_WARN("\n\n%s : tensor cols %d x %d are not divisible by %d, required for k-quants\n", __func__, nx, ny, QK_K);
            convert_incompatible_tensor = true;
        }
    }
    if (convert_incompatible_tensor) {
        if (name == tn(LLM_TENSOR_OUTPUT, "weight")) {
            new_type = GGML_TYPE_F16; //fall back to F16 instead of just failing.
            LLAMA_LOG_WARN("F16 will be used for this tensor instead.\n");
        } else if (name == tn(LLM_TENSOR_TOKEN_EMBD, "weight")) {
            new_type = GGML_TYPE_Q4_0; //fall back to Q4_0 instead of just failing.
            LLAMA_LOG_WARN("Q4_0 will be used for this tensor instead.\n");
        } else {
            throw std::runtime_error("Unsupported tensor size encountered\n");
        }
    }

    return new_type;
}
#endif

static void llama_model_quantize_internal(const std::string & fname_inp, const std::string & fname_out, const llama_model_quantize_params * params) {
    ggml_type quantized_type;
    llama_ftype ftype = params->ftype;

    switch (params->ftype) {
        case LLAMA_FTYPE_MOSTLY_Q4_0: quantized_type = GGML_TYPE_Q4_0; break;
        case LLAMA_FTYPE_MOSTLY_Q4_1: quantized_type = GGML_TYPE_Q4_1; break;
        case LLAMA_FTYPE_MOSTLY_Q5_0: quantized_type = GGML_TYPE_Q5_0; break;
        case LLAMA_FTYPE_MOSTLY_Q5_1: quantized_type = GGML_TYPE_Q5_1; break;
        case LLAMA_FTYPE_MOSTLY_Q8_0: quantized_type = GGML_TYPE_Q8_0; break;
        case LLAMA_FTYPE_MOSTLY_F16:  quantized_type = GGML_TYPE_F16;  break;
        case LLAMA_FTYPE_ALL_F32:     quantized_type = GGML_TYPE_F32;  break;

#ifdef GGML_USE_K_QUANTS
        // K-quants
        case LLAMA_FTYPE_MOSTLY_Q2_K:   quantized_type = GGML_TYPE_Q2_K; break;
        case LLAMA_FTYPE_MOSTLY_Q3_K_S:
        case LLAMA_FTYPE_MOSTLY_Q3_K_M:
        case LLAMA_FTYPE_MOSTLY_Q3_K_L: quantized_type = GGML_TYPE_Q3_K; break;
        case LLAMA_FTYPE_MOSTLY_Q4_K_S:
        case LLAMA_FTYPE_MOSTLY_Q4_K_M: quantized_type = GGML_TYPE_Q4_K; break;
        case LLAMA_FTYPE_MOSTLY_Q5_K_S:
        case LLAMA_FTYPE_MOSTLY_Q5_K_M: quantized_type = GGML_TYPE_Q5_K; break;
        case LLAMA_FTYPE_MOSTLY_Q6_K:   quantized_type = GGML_TYPE_Q6_K; break;
#endif
        default: throw std::runtime_error(format("invalid output file type %d\n", ftype));
    }

    int nthread = params->nthread;

    if (nthread <= 0) {
        nthread = std::thread::hardware_concurrency();
    }

    // mmap consistently increases speed Linux, and also increases speed on Windows with
    // hot cache. It may cause a slowdown on macOS, possibly related to free memory.
#if defined(__linux__) || defined(_WIN32)
    constexpr bool use_mmap = true;
#else
    constexpr bool use_mmap = false;
#endif

    llama_model_loader ml(fname_inp, use_mmap);
    if (ml.use_mmap) {
        ml.mapping.reset(new llama_mmap(&ml.file, /* prefetch */ 0, ggml_is_numa()));
    }

    llama_model model;
    llm_load_arch(ml, model);
    llm_load_hparams(ml, model);

    if (params->only_copy) {
        ftype = model.ftype;
    }

    const size_t align = GGUF_DEFAULT_ALIGNMENT;
    struct gguf_context * ctx_out = gguf_init_empty();

    // copy the KV pairs from the input file
    gguf_set_kv     (ctx_out, ml.ctx_gguf);
    gguf_set_val_u32(ctx_out, "general.quantization_version", GGML_QNT_VERSION);
    gguf_set_val_u32(ctx_out, "general.file_type", ftype);

#ifdef GGML_USE_K_QUANTS
    int n_attention_wv    = 0;
    int n_feed_forward_w2 = 0;

    for (int i = 0; i < ml.n_tensors; ++i) {
        struct ggml_tensor * meta = ml.get_tensor_meta(i);

        const std::string name = ggml_get_name(meta);

        // TODO: avoid hardcoded tensor names - use the TN_* constants
        if (name.find("attn_v.weight") != std::string::npos || name.find("attn_qkv.weight") != std::string::npos) {
            ++n_attention_wv;
        }
        else if (name.find("ffn_down.weight") != std::string::npos) {
            ++n_feed_forward_w2;
        }
    }
    if (n_attention_wv != n_feed_forward_w2 || (uint32_t)n_attention_wv != model.hparams.n_layer) {
        LLAMA_LOG_WARN("%s ============ Strange model: n_attention_wv = %d, n_feed_forward_w2 = %d, hparams.n_layer = %d\n",
                __func__, n_attention_wv, n_feed_forward_w2, model.hparams.n_layer);
    }

    int i_attention_wv = 0;
    int i_feed_forward_w2 = 0;
#endif

    size_t total_size_org = 0;
    size_t total_size_new = 0;
    std::vector<int64_t> hist_all(1 << 4, 0);

    std::vector<std::thread> workers;
    workers.reserve(nthread);
    std::mutex mutex;

    int idx = 0;

    std::vector<no_init<uint8_t>> read_data;
    std::vector<no_init<uint8_t>> work;
    std::vector<no_init<float>> f32_conv_buf;

    // populate the original tensors so we get an initial meta data
    for (int i = 0; i < ml.n_tensors; ++i) {
        struct ggml_tensor * meta = ml.get_tensor_meta(i);
        gguf_add_tensor(ctx_out, meta);
    }

    std::ofstream fout(fname_out, std::ios::binary);
    fout.exceptions(std::ofstream::failbit); // fail fast on write errors

    const size_t meta_size = gguf_get_meta_size(ctx_out);

    LLAMA_LOG_INFO("%s: meta size = %zu bytes\n", __func__, meta_size);

    // placeholder for the meta data
    ::zeros(fout, meta_size);

    for (int i = 0; i < ml.n_tensors; ++i) {
        struct ggml_tensor * tensor = ml.get_tensor_meta(i);

        const std::string name = ggml_get_name(tensor);

        if (!ml.use_mmap) {
            if (read_data.size() < ggml_nbytes(tensor)) {
                read_data.resize(ggml_nbytes(tensor));
            }
            tensor->data = read_data.data();
        }
        ml.load_data_for(tensor);

        LLAMA_LOG_INFO("[%4d/%4d] %36s - [%s], type = %6s, ",
               ++idx, ml.n_tensors,
               ggml_get_name(tensor),
               llama_format_tensor_shape(tensor).c_str(),
               ggml_type_name(tensor->type));

        // This used to be a regex, but <regex> has an extreme cost to compile times.
        bool quantize = name.rfind("weight") == name.size() - 6; // ends with 'weight'?

        // quantize only 2D tensors
        quantize &= (tensor->n_dims == 2);
        quantize &= params->quantize_output_tensor || name != "output.weight";
        quantize &= !params->only_copy;

        enum ggml_type new_type;
        void * new_data;
        size_t new_size;

        if (quantize) {
            new_type = quantized_type;
#ifdef GGML_USE_K_QUANTS
            new_type = get_k_quant_type(
                new_type, tensor, model, ftype, &i_attention_wv, n_attention_wv, &i_feed_forward_w2, n_feed_forward_w2
            );
#endif
            // If we've decided to quantize to the same type the tensor is already
            // in then there's nothing to do.
            quantize = tensor->type != new_type;
        }
        if (!quantize) {
            new_type = tensor->type;
            new_data = tensor->data;
            new_size = ggml_nbytes(tensor);
            LLAMA_LOG_INFO("size = %8.3f MB\n", ggml_nbytes(tensor)/1024.0/1024.0);
        } else {
            const size_t nelements = ggml_nelements(tensor);

            float * f32_data;

            if (tensor->type == GGML_TYPE_F32) {
                f32_data = (float *) tensor->data;
            } else if (ggml_is_quantized(tensor->type) && !params->allow_requantize) {
                throw std::runtime_error(format("requantizing from type %s is disabled", ggml_type_name(tensor->type)));
            } else {
                llama_convert_tensor_internal(tensor, f32_conv_buf, workers, nelements, nthread);
                f32_data = (float *) f32_conv_buf.data();
            }

            LLAMA_LOG_INFO("quantizing to %s .. ", ggml_type_name(new_type));
            fflush(stdout);

            if (work.size() < nelements * 4) {
                work.resize(nelements * 4); // upper bound on size
            }
            new_data = work.data();
            std::array<int64_t, 1 << 4> hist_cur = {};

            static const int chunk_size = 32 * 512;
            const int nchunk = (nelements + chunk_size - 1)/chunk_size;
            const int nthread_use = nthread > 1 ? std::max(1, std::min(nthread, nchunk)) : 1;
            if (nthread_use < 2) {
                new_size = ggml_quantize_chunk(new_type, f32_data, new_data, 0, nelements, hist_cur.data());
            } else {
                size_t counter = 0;
                new_size = 0;
                auto compute = [&mutex, &counter, &hist_cur, &new_size, new_type, f32_data, new_data, nelements]() {
                    std::array<int64_t, 1 << 4> local_hist = {};
                    size_t local_size = 0;
                    while (true) {
                        std::unique_lock<std::mutex> lock(mutex);
                        size_t first = counter; counter += chunk_size;
                        if (first >= nelements) {
                            if (local_size > 0) {
                                for (int j=0; j<int(local_hist.size()); ++j) {
                                    hist_cur[j] += local_hist[j];
                                }
                                new_size += local_size;
                            }
                            break;
                        }
                        lock.unlock();
                        size_t last = std::min(nelements, first + chunk_size);
                        local_size += ggml_quantize_chunk(new_type, f32_data, new_data, first, last - first, local_hist.data());
                    }
                };
                for (int it = 0; it < nthread_use - 1; ++it) {
                    workers.emplace_back(compute);
                }
                compute();
                for (auto & w : workers) { w.join(); }
                workers.clear();
            }

            LLAMA_LOG_INFO("size = %8.2f MB -> %8.2f MB | hist: ", ggml_nbytes(tensor)/1024.0/1024.0, new_size/1024.0/1024.0);
            int64_t tot_count = 0;
            for (size_t i = 0; i < hist_cur.size(); i++) {
                hist_all[i] += hist_cur[i];
                tot_count += hist_cur[i];
            }

            if (tot_count > 0) {
                for (size_t i = 0; i < hist_cur.size(); i++) {
                    LLAMA_LOG_INFO("%5.3f ", hist_cur[i] / float(nelements));
                }
            }
            LLAMA_LOG_INFO("\n");
        }
        total_size_org += ggml_nbytes(tensor);
        total_size_new += new_size;

        // update the gguf meta data as we go
        gguf_set_tensor_type(ctx_out, name.c_str(), new_type);
        gguf_set_tensor_data(ctx_out, name.c_str(), new_data, new_size);

        // write tensor data + padding
        fout.write((const char *) new_data, new_size);
        zeros(fout, GGML_PAD(new_size, align) - new_size);
    }

    // go back to beginning of file and write the updated meta data
    {
        fout.seekp(0);
        std::vector<uint8_t> data(gguf_get_meta_size(ctx_out));
        gguf_get_meta_data(ctx_out, data.data());
        fout.write((const char *) data.data(), data.size());
    }

    fout.close();

    gguf_free(ctx_out);

    LLAMA_LOG_INFO("%s: model size  = %8.2f MB\n", __func__, total_size_org/1024.0/1024.0);
    LLAMA_LOG_INFO("%s: quant size  = %8.2f MB\n", __func__, total_size_new/1024.0/1024.0);

    // print histogram for all tensors
    {
        int64_t sum_all = 0;
        for (size_t i = 0; i < hist_all.size(); i++) {
            sum_all += hist_all[i];
        }

        if (sum_all > 0) {
            LLAMA_LOG_INFO("%s: hist: ", __func__);
            for (size_t i = 0; i < hist_all.size(); i++) {
                LLAMA_LOG_INFO("%5.3f ", hist_all[i] / float(sum_all));
            }
            LLAMA_LOG_INFO("\n");
        }
    }
}

static int llama_apply_lora_from_file_internal(
    const struct llama_model & model, const char * path_lora, float scale, const char * path_base_model, int n_threads
) {
    LLAMA_LOG_INFO("%s: applying lora adapter from '%s' - please wait ...\n", __func__, path_lora);

    const int64_t t_start_lora_us = ggml_time_us();

    auto fin = std::ifstream(path_lora, std::ios::binary);
    if (!fin) {
        LLAMA_LOG_ERROR("%s: failed to open '%s'\n", __func__, path_lora);
        return 1;
    }

    // verify magic and version
    {
        uint32_t magic;
        fin.read((char *) &magic, sizeof(magic));
        uint32_t format_version;
        fin.read((char *) &format_version, sizeof(format_version));

        if (format_version != 1) {
            LLAMA_LOG_ERROR("%s: unsupported file version\n", __func__ );
            return 1;
        }
    }

    int32_t lora_r;
    int32_t lora_alpha;
    fin.read((char *) &lora_r, sizeof(lora_r));
    fin.read((char *) &lora_alpha, sizeof(lora_alpha));
    float scaling = scale * (float)lora_alpha / (float)lora_r;

    LLAMA_LOG_INFO("%s: r = %d, alpha = %d, scaling = %.2f\n", __func__, lora_r, lora_alpha, scaling);

    // create a temporary ggml context to store the lora tensors
    // todo: calculate size from biggest possible tensor
    std::vector<uint8_t> lora_buf(1024ull * 1024ull * 1024ull);
    struct ggml_init_params params;
    params.mem_size   = lora_buf.size();
    params.mem_buffer = lora_buf.data();
    params.no_alloc   = false;

    ggml_context * lora_ctx = ggml_init(params);
    std::unordered_map<std::string, struct ggml_tensor *> lora_tensors;

    // create a name -> tensor map of the model to accelerate lookups
    std::unordered_map<std::string, struct ggml_tensor*> model_tensors;
    for (const auto & kv : model.tensors_by_name) {
        model_tensors.insert(kv);
    }

    // load base model
    std::unique_ptr<llama_model_loader> ml;
    ggml_context * base_ctx = NULL;
    std::vector<uint8_t> base_buf;
    if (path_base_model) {
        LLAMA_LOG_INFO("%s: loading base model from '%s'\n", __func__, path_base_model);
        ml.reset(new llama_model_loader(path_base_model, /*use_mmap*/ true));

        size_t ctx_size;
        size_t mmapped_size;
        ml->calc_sizes(ctx_size, mmapped_size);
        base_buf.resize(ctx_size);

        ggml_init_params base_params;
        base_params.mem_size   = base_buf.size();
        base_params.mem_buffer = base_buf.data();
        base_params.no_alloc   = ml->use_mmap;

        base_ctx = ggml_init(base_params);

        // maybe this should in llama_model_loader
        if (ml->use_mmap) {
            ml->mapping.reset(new llama_mmap(&ml->file, /* prefetch */ 0, ggml_is_numa()));
        }
    }

    // read tensors and apply
    bool warned = false;
    int n_tensors = 0;

    std::vector<uint8_t> work_buffer;

    while (true) {
        int32_t n_dims;
        int32_t length;
        int32_t ftype;

        fin.read(reinterpret_cast<char *>(&n_dims), sizeof(n_dims));
        fin.read(reinterpret_cast<char *>(&length), sizeof(length));
        fin.read(reinterpret_cast<char *>(&ftype),  sizeof(ftype));
        if (fin.eof()) {
            break;
        }

        int32_t ne[2] = { 1, 1 };
        for (int i = 0; i < n_dims; ++i) {
            fin.read(reinterpret_cast<char *>(&ne[i]), sizeof(ne[i]));
        }

        std::string name;
        {
            char buf[1024];
            fin.read(buf, length);
            name = std::string(buf, length);
        }

        // check for lora suffix and get the type of tensor
        const std::string lora_suffix = ".lora";
        size_t pos = name.rfind(lora_suffix);
        if (pos == std::string::npos) {
            LLAMA_LOG_ERROR("%s: error: '%s' is not a lora tensor\n", __func__, name.c_str());
            return 1;
        }

        std::string lora_type = name.substr(pos + lora_suffix.length());
        std::string base_name = name;
        base_name.erase(pos);
        // LLAMA_LOG_INFO("%s: %s => %s (lora type %s) \n", __func__, name.c_str(),base_name.c_str(), lora_type.c_str());

        if (model_tensors.find(base_name) == model_tensors.end()) {
            LLAMA_LOG_ERROR("%s: unknown tensor '%s' in lora adapter\n", __func__, name.data());
            return 1;
        }

        // create ggml tensor
        ggml_type wtype;
        switch (ftype) {
            case 0: wtype = GGML_TYPE_F32;  break;
            case 1: wtype = GGML_TYPE_F16;  break;
            default:
                    {
                        LLAMA_LOG_ERROR("%s: invalid tensor data type '%d'\n",
                                __func__, ftype);
                        return false;
                    }
        }
        ggml_tensor * lora_tensor;
        if (n_dims == 2) {
            lora_tensor = ggml_new_tensor_2d(lora_ctx, wtype, ne[0], ne[1]);
        }
        else {
            LLAMA_LOG_ERROR("%s: unsupported tensor dimension %d\n", __func__, n_dims);
            return 1;
        }
        ggml_set_name(lora_tensor, "lora_tensor");

        // load tensor data
        size_t offset = fin.tellg();
        size_t tensor_data_size = ggml_nbytes(lora_tensor);
        offset = (offset + 31) & -32;
        fin.seekg(offset);
        fin.read((char*)lora_tensor->data, tensor_data_size);

        lora_tensors[name] = lora_tensor;

        // check if we have both A and B tensors and apply
        if (lora_tensors.find(base_name + ".loraA") != lora_tensors.end() &&
            lora_tensors.find(base_name + ".loraB") != lora_tensors.end()) {

            ggml_tensor * dest_t = model_tensors[base_name];

            offload_func_t offload_func = llama_nop;
            offload_func_t offload_func_force_inplace = llama_nop;

#ifdef GGML_USE_CUBLAS
            if (dest_t->backend == GGML_BACKEND_GPU || dest_t->backend == GGML_BACKEND_GPU_SPLIT) {
                if (dest_t->type != GGML_TYPE_F16) {
                    throw std::runtime_error(format(
                        "%s: error: the simultaneous use of LoRAs and GPU acceleration is only supported for f16 models", __func__));
                }
                offload_func = ggml_cuda_assign_buffers;
                offload_func_force_inplace = ggml_cuda_assign_buffers_force_inplace;
            }
#endif // GGML_USE_CUBLAS

            ggml_tensor * base_t;
            if (ml) {
                struct gguf_context * ctx_gguf = ml->ctx_gguf;

                // load from base model
                if (gguf_find_tensor(ctx_gguf, base_name.c_str()) < 0) {
                    // TODO: throw
                    LLAMA_LOG_ERROR("%s: error: tensor '%s' not found in base model\n", __func__, base_name.c_str());
                    return 1;
                }

                // TODO: not tested!! maybe not working!
                base_t = ml->create_tensor(base_ctx, base_name, { (uint32_t)dest_t->ne[0], (uint32_t)dest_t->ne[1] }, GGML_BACKEND_CPU);
                ml->load_data_for(base_t);
            } else {
                base_t = dest_t;
            }

            if (ggml_is_quantized(base_t->type)) {
                if (!warned) {
                    LLAMA_LOG_WARN("%s: warning: using a lora adapter with a quantized model may result in poor quality, "
                                   "use a f16 or f32 base model with --lora-base\n", __func__);
                    warned = true;
                }
            }

            ggml_tensor * loraA = lora_tensors[base_name + ".loraA"];
            GGML_ASSERT(loraA->type == GGML_TYPE_F32);
            ggml_set_name(loraA, "loraA");

            ggml_tensor * loraB = lora_tensors[base_name + ".loraB"];
            GGML_ASSERT(loraB->type == GGML_TYPE_F32);
            ggml_set_name(loraB, "loraB");

            if (base_t->ne[0] != loraA->ne[1] || base_t->ne[1] != loraB->ne[1]) {
                LLAMA_LOG_ERROR("%s: incompatible tensor dimensions (%" PRId64 " and %" PRId64 ");"
                                " are you sure that this adapter is for this model?\n", __func__, base_t->ne[0], loraA->ne[1]);
                return 1;
            }

            // w = w + BA*s
            ggml_tensor * BA = ggml_mul_mat(lora_ctx, loraA, loraB);
            offload_func(BA);
            ggml_set_name(BA, "BA");

            if (scaling != 1.0f) {
                ggml_tensor * scale_tensor = ggml_new_f32(lora_ctx, scaling);
                ggml_set_name(scale_tensor, "scale_tensor");

                BA = ggml_scale_inplace(lora_ctx, BA, scale_tensor);
                offload_func(BA);
                ggml_set_name(BA, "BA_scaled");
            }

            ggml_tensor * r;
            if (base_t == dest_t) {
                r = ggml_add_inplace(lora_ctx, dest_t, BA);
                offload_func_force_inplace(r);
                ggml_set_name(r, "r_add_inplace");
            }
            else {
                r = ggml_add(lora_ctx, base_t, BA);
                offload_func(r);
                ggml_set_name(r, "r_add");

                r = ggml_cpy(lora_ctx, r, dest_t);
                offload_func(r);
                ggml_set_name(r, "r_cpy");
            }

            struct ggml_cgraph * gf = ggml_new_graph(lora_ctx);
            ggml_build_forward_expand(gf, r);

            ggml_graph_compute_helper(work_buffer, gf, n_threads);

            // we won't need these tensors again, reset the context to save memory
            ggml_free(lora_ctx);
            lora_ctx = ggml_init(params);
            lora_tensors.clear();

            n_tensors++;
            if (n_tensors % 4 == 0) {
                LLAMA_LOG_INFO(".");
            }
        }
    }

    // TODO: this should be in a destructor, it will leak on failure
    ggml_free(lora_ctx);
    if (base_ctx) {
        ggml_free(base_ctx);
    }

    const int64_t t_lora_us = ggml_time_us() - t_start_lora_us;
    LLAMA_LOG_INFO(" done (%.2f ms)\n", t_lora_us / 1000.0);

    return 0;
}

//
// interface implementation
//
struct llama_model_params llama_model_default_params() {
    struct llama_model_params result = {
        /*.n_gpu_layers                =*/ 0,
        /*.main_gpu                    =*/ 0,
        /*.tensor_split                =*/ nullptr,
        /*.progress_callback           =*/ nullptr,
        /*.progress_callback_user_data =*/ nullptr,
        /*.vocab_only                  =*/ false,
        /*.use_mmap                    =*/ true,
        /*.use_mlock                   =*/ false,
    };

#ifdef GGML_USE_METAL
    result.n_gpu_layers = 1;
#endif

    return result;
}

struct llama_context_params llama_context_default_params() {
    struct llama_context_params result = {
        /*.seed                        =*/ LLAMA_DEFAULT_SEED,
        /*.n_ctx                       =*/ 512,
        /*.n_batch                     =*/ 512,
        /*.n_threads                   =*/ GGML_DEFAULT_N_THREADS, // TODO: better default
        /*.n_threads_batch             =*/ GGML_DEFAULT_N_THREADS,
        /*.rope_freq_base              =*/ 0.0f,
        /*.rope_freq_scale             =*/ 0.0f,
        /*.mul_mat_q                   =*/ true,
        /*.f16_kv                      =*/ true,
        /*.logits_all                  =*/ false,
        /*.embedding                   =*/ false,
    };

    return result;
}

struct llama_model_quantize_params llama_model_quantize_default_params() {
    struct llama_model_quantize_params result = {
        /*.nthread                     =*/ 0,
        /*.ftype                       =*/ LLAMA_FTYPE_MOSTLY_Q5_1,
        /*.allow_requantize            =*/ false,
        /*.quantize_output_tensor      =*/ true,
        /*.only_copy                   =*/ false,
    };

    return result;
}

int llama_max_devices(void) {
    return LLAMA_MAX_DEVICES;
}

bool llama_mmap_supported(void) {
    return llama_mmap::SUPPORTED;
}

bool llama_mlock_supported(void) {
    return llama_mlock::SUPPORTED;
}

void llama_backend_init(bool numa) {
    ggml_time_init();

    // needed to initialize f16 tables
    {
        struct ggml_init_params params = { 0, NULL, false };
        struct ggml_context * ctx = ggml_init(params);
        ggml_free(ctx);
    }

    if (numa) {
        ggml_numa_init();
    }

#ifdef GGML_USE_MPI
    ggml_mpi_backend_init();
#endif
}

void llama_backend_free(void) {
#ifdef GGML_USE_MPI
    ggml_mpi_backend_free();
#endif
}

int64_t llama_time_us(void) {
    return ggml_time_us();
}

struct llama_model * llama_load_model_from_file(
                             const char * path_model,
              struct llama_model_params   params) {
    ggml_time_init();

    llama_model * model = new llama_model;

    unsigned cur_percentage = 0;
    if (params.progress_callback == NULL) {
        params.progress_callback_user_data = &cur_percentage;
        params.progress_callback = [](float progress, void * ctx) {
            unsigned * cur_percentage_p = (unsigned *) ctx;
            unsigned percentage = (unsigned) (100 * progress);
            while (percentage > *cur_percentage_p) {
                *cur_percentage_p = percentage;
                LLAMA_LOG_INFO(".");
                if (percentage >= 100) {
                    LLAMA_LOG_INFO("\n");
                }
            }
        };
    }

    if (!llama_model_load(path_model, *model, params.n_gpu_layers,
                params.main_gpu, params.tensor_split,
                params.use_mmap, params.use_mlock, params.vocab_only,
                params.progress_callback, params.progress_callback_user_data)) {
        LLAMA_LOG_ERROR("%s: failed to load model\n", __func__);
        delete model;
        return nullptr;
    }

    return model;
}

void llama_free_model(struct llama_model * model) {
    delete model;
}

struct llama_context * llama_new_context_with_model(
                 struct llama_model * model,
        struct llama_context_params   params) {

    if (!model) {
        return nullptr;
    }

    llama_context * ctx = new llama_context(*model);

    const auto & hparams = model->hparams;
    auto       & cparams = ctx->cparams;

    cparams.n_batch         = params.n_batch;
    cparams.n_ctx           = params.n_ctx == 0           ? hparams.n_ctx_train           : params.n_ctx;
    cparams.rope_freq_base  = params.rope_freq_base == 0  ? hparams.rope_freq_base_train  : params.rope_freq_base;
    cparams.rope_freq_scale = params.rope_freq_scale == 0 ? hparams.rope_freq_scale_train : params.rope_freq_scale;
    cparams.n_threads       = params.n_threads;
    cparams.n_threads_batch = params.n_threads_batch;
    cparams.mul_mat_q       = params.mul_mat_q;

    if (params.seed == LLAMA_DEFAULT_SEED) {
        params.seed = time(NULL);
    }

    LLAMA_LOG_INFO("%s: n_ctx      = %u\n",     __func__, cparams.n_ctx);
    LLAMA_LOG_INFO("%s: freq_base  = %.1f\n",   __func__, cparams.rope_freq_base);
    LLAMA_LOG_INFO("%s: freq_scale = %g\n",     __func__, cparams.rope_freq_scale);

    ctx->rng = std::mt19937(params.seed);
    ctx->logits_all = params.logits_all;

    ggml_type memory_type = params.f16_kv ? GGML_TYPE_F16 : GGML_TYPE_F32;

    // reserve memory for context buffers
    if (!hparams.vocab_only) {
        if (!llama_kv_cache_init(ctx->model.hparams, ctx->kv_self, memory_type, cparams.n_ctx, model->n_gpu_layers)) {
            LLAMA_LOG_ERROR("%s: llama_kv_cache_init() failed for self-attention cache\n", __func__);
            llama_free(ctx);
            return nullptr;
        }

        {
            const size_t memory_size = ggml_nbytes(ctx->kv_self.k) + ggml_nbytes(ctx->kv_self.v);
            LLAMA_LOG_INFO("%s: kv self size  = %7.2f MB\n", __func__, memory_size / 1024.0 / 1024.0);
        }

        // resized during inference
        if (params.logits_all) {
            ctx->logits.reserve(cparams.n_ctx*hparams.n_vocab);
        } else {
            ctx->logits.reserve(hparams.n_vocab);
        }

        if (params.embedding){
            ctx->embedding.resize(hparams.n_embd);
        }

        {
            static const size_t tensor_alignment = 32;
            // the compute buffer is used to store the tensor and graph structs, while the allocator buffer is used for the tensor data
            ctx->buf_compute.resize(ggml_tensor_overhead()*GGML_MAX_NODES + ggml_graph_overhead());

            // create measure allocator
            ctx->alloc = ggml_allocr_new_measure(tensor_alignment);

            // build worst-case graph
            int n_tokens = (int)std::min(cparams.n_ctx, cparams.n_batch);
            int n_past = cparams.n_ctx - n_tokens;
            llama_token token = llama_token_bos(ctx); // not actually used by llama_build_graph, but required to choose between token and embedding inputs graph
            ggml_cgraph * gf = llama_build_graph(*ctx, llama_batch_get_one(&token, n_tokens, n_past, 0));

#ifdef GGML_USE_METAL
            if (model->n_gpu_layers > 0) {
                ggml_metal_log_set_callback(llama_log_callback_default, NULL);

                ctx->ctx_metal = ggml_metal_init(1);
                if (!ctx->ctx_metal) {
                    LLAMA_LOG_ERROR("%s: ggml_metal_init() failed\n", __func__);
                    llama_free(ctx);
                    return NULL;
                }
                //ggml_metal_graph_find_concurrency(ctx->ctx_metal, gf, false);
                //ggml_allocr_set_parse_seq(ctx->alloc, ggml_metal_get_concur_list(ctx->ctx_metal), ggml_metal_if_optimized(ctx->ctx_metal));
            }
#endif
            // measure memory requirements for the graph
            size_t alloc_size = ggml_allocr_alloc_graph(ctx->alloc, gf) + tensor_alignment;

            LLAMA_LOG_INFO("%s: compute buffer total size = %.2f MB\n", __func__, (ctx->buf_compute.size + alloc_size) / 1024.0 / 1024.0);

            // recreate allocator with exact memory requirements
            ggml_allocr_free(ctx->alloc);

            ctx->buf_alloc.resize(alloc_size);
            ctx->alloc = ggml_allocr_new(ctx->buf_alloc.data, ctx->buf_alloc.size, tensor_alignment);
#ifdef GGML_USE_METAL
            if (ctx->ctx_metal) {
                //ggml_allocr_set_parse_seq(ctx->alloc, ggml_metal_get_concur_list(ctx->ctx_metal), ggml_metal_if_optimized(ctx->ctx_metal));
            }
#endif
#ifdef GGML_USE_CUBLAS
            ggml_cuda_set_scratch_size(alloc_size);
            LLAMA_LOG_INFO("%s: VRAM scratch buffer: %.2f MB\n", __func__, alloc_size / 1024.0 / 1024.0);

            // calculate total VRAM usage
            auto add_tensor = [](const ggml_tensor * t, size_t & size) {
                if (t->backend == GGML_BACKEND_GPU || t->backend == GGML_BACKEND_GPU_SPLIT) {
                    size += ggml_nbytes(t);
                }
            };
            size_t model_vram_size = 0;
            for (const auto & kv : model->tensors_by_name) {
                add_tensor(kv.second, model_vram_size);
            }

            size_t kv_vram_size = 0;
            add_tensor(ctx->kv_self.k, kv_vram_size);
            add_tensor(ctx->kv_self.v, kv_vram_size);

            size_t ctx_vram_size = alloc_size + kv_vram_size;
            size_t total_vram_size = model_vram_size + ctx_vram_size;

            LLAMA_LOG_INFO("%s: total VRAM used: %.2f MB (model: %.2f MB, context: %.2f MB)\n", __func__,
                    total_vram_size / 1024.0 / 1024.0,
                    model_vram_size / 1024.0 / 1024.0,
                    ctx_vram_size / 1024.0 / 1024.0);
#endif
        }

#ifdef GGML_USE_METAL
        if (model->n_gpu_layers > 0) {
            // this allocates all Metal resources and memory buffers

            void * data_ptr  = NULL;
            size_t data_size = 0;

            if (ctx->model.mapping) {
                data_ptr  = ctx->model.mapping->addr;
                data_size = ctx->model.mapping->size;
            } else {
                data_ptr  = ggml_get_mem_buffer(ctx->model.ctx);
                data_size = ggml_get_mem_size  (ctx->model.ctx);
            }

            const size_t max_size = ggml_get_max_tensor_size(ctx->model.ctx);

            LLAMA_LOG_INFO("%s: max tensor size = %8.2f MB\n", __func__, max_size/1024.0/1024.0);

#define LLAMA_METAL_CHECK_BUF(result)                            \
            if (!(result)) {                                             \
                LLAMA_LOG_ERROR("%s: failed to add buffer\n", __func__); \
                llama_free(ctx);                                         \
                return NULL;                                             \
            }

            LLAMA_METAL_CHECK_BUF(ggml_metal_add_buffer(ctx->ctx_metal, "data",  data_ptr, data_size, max_size));
            LLAMA_METAL_CHECK_BUF(ggml_metal_add_buffer(ctx->ctx_metal, "kv",    ctx->kv_self.buf.data, ctx->kv_self.buf.size, 0));
            LLAMA_METAL_CHECK_BUF(ggml_metal_add_buffer(ctx->ctx_metal, "alloc", ctx->buf_alloc.data, ctx->buf_alloc.size, 0));
#undef LLAMA_METAL_CHECK_BUF
        }
#endif
    }

#ifdef GGML_USE_MPI
    ctx->ctx_mpi = ggml_mpi_init();

    if (ggml_mpi_rank(ctx->ctx_mpi) > 0) {
        // Enter a blocking eval loop with dummy input, letting rank=0 drive the process
        // TODO: needs fix after #3228
        GGML_ASSERT(false && "not implemented");
        //const std::vector<llama_token> tmp(ctx->model.hparams.n_ctx, llama_token_bos(ctx));
        //while (!llama_eval(ctx, tmp.data(), tmp.size(), 0, 0)) {};
        llama_backend_free();
        exit(1);
    }
#endif

    return ctx;
}

void llama_free(struct llama_context * ctx) {
    delete ctx;
}

const llama_model * llama_get_model(const struct llama_context * ctx) {
    return &ctx->model;
}

int llama_n_ctx(const struct llama_context * ctx) {
    return ctx->cparams.n_ctx;
}

enum llama_vocab_type llama_vocab_type(const struct llama_model * model) {
    return model->vocab.type;
}

int llama_n_vocab(const struct llama_model * model) {
    return model->vocab.id_to_token.size();
}

int llama_n_ctx_train(const struct llama_model * model) {
    return model->hparams.n_ctx_train;
}

int llama_n_embd(const struct llama_model * model) {
    return model->hparams.n_embd;
}

float llama_rope_freq_scale_train(const struct llama_model * model) {
    return model->hparams.rope_freq_scale_train;
}

int llama_model_desc(const struct llama_model * model, char * buf, size_t buf_size) {
    return snprintf(buf, buf_size, "%s %s %s",
            llama_model_arch_name(model->arch).c_str(),
            llama_model_type_name(model->type),
            llama_model_ftype_name(model->ftype).c_str());
}

uint64_t llama_model_size(const struct llama_model * model) {
    uint64_t size = 0;
    for (const auto & it : model->tensors_by_name) {
        size += ggml_nbytes(it.second);
    }
    return size;
}

uint64_t llama_model_n_params(const struct llama_model * model) {
    uint64_t nparams = 0;
    for (const auto & it : model->tensors_by_name) {
        nparams += ggml_nelements(it.second);
    }
    return nparams;
}

struct ggml_tensor * llama_get_model_tensor(struct llama_model * model, const char * name) {
    return ggml_get_tensor(model->ctx, name);
}

int llama_model_quantize(
        const char * fname_inp,
        const char * fname_out,
        const llama_model_quantize_params * params) {
    try {
        llama_model_quantize_internal(fname_inp, fname_out, params);
        return 0;
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: failed to quantize: %s\n", __func__, err.what());
        return 1;
    }
}

int llama_apply_lora_from_file(struct llama_context * ctx, const char * path_lora, float scale, const char * path_base_model, int n_threads) {
    try {
        return llama_apply_lora_from_file_internal(ctx->model, path_lora, scale, path_base_model, n_threads);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: failed to apply lora adapter: %s\n", __func__, err.what());
        return 1;
    }
}

int llama_model_apply_lora_from_file(const struct llama_model * model, const char * path_lora, float scale, const char * path_base_model, int n_threads) {
    try {
        return llama_apply_lora_from_file_internal(*model, path_lora, scale, path_base_model, n_threads);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: failed to apply lora adapter: %s\n", __func__, err.what());
        return 1;
    }
}

int llama_get_kv_cache_token_count(const struct llama_context * ctx) {
    return ctx->kv_self.head;
}

void llama_kv_cache_tokens_rm(struct llama_context * ctx, int32_t c0, int32_t c1) {
    llama_kv_cache_tokens_rm(ctx->kv_self, c0, c1);
}

void llama_kv_cache_seq_rm(struct llama_context * ctx, llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    llama_kv_cache_seq_rm(ctx->kv_self, seq_id, p0, p1);
}

void llama_kv_cache_seq_cp(struct llama_context * ctx, llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    if (seq_id_src == seq_id_dst) {
        return;
    }
    llama_kv_cache_seq_cp(ctx->kv_self, seq_id_src, seq_id_dst, p0, p1);
}

void llama_kv_cache_seq_keep(struct llama_context * ctx, llama_seq_id seq_id) {
    llama_kv_cache_seq_keep(ctx->kv_self, seq_id);
}

void llama_kv_cache_seq_shift(struct llama_context * ctx, llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos delta) {
    llama_kv_cache_seq_shift(ctx->kv_self, seq_id, p0, p1, delta);
}

// Returns the *maximum* size of the state
size_t llama_get_state_size(const struct llama_context * ctx) {
    // we don't know size of rng until we actually serialize it. so reserve more than enough memory for its serialized state.
    // for reference, std::mt19937(1337) serializes to 6701 bytes.
    const size_t s_rng_size        = sizeof(size_t);
    const size_t s_rng             = LLAMA_MAX_RNG_STATE;
    const size_t s_logits_capacity = sizeof(size_t);
    const size_t s_logits_size     = sizeof(size_t);
    const size_t s_logits          = ctx->logits.capacity() * sizeof(float);
    const size_t s_embedding_size  = sizeof(size_t);
    const size_t s_embedding       = ctx->embedding.size() * sizeof(float);
    const size_t s_kv_size         = sizeof(size_t);
    const size_t s_kv_ntok         = sizeof(int);
    const size_t s_kv              = ctx->kv_self.buf.size;

    const size_t s_total = (
        + s_rng_size
        + s_rng
        + s_logits_capacity
        + s_logits_size
        + s_logits
        + s_embedding_size
        + s_embedding
        + s_kv_size
        + s_kv_ntok
        + s_kv
    );

    return s_total;
}

// llama_context_data
struct llama_data_context {
    virtual void write(const void * src, size_t size) = 0;
    virtual size_t get_size_written() = 0;
    virtual ~llama_data_context() = default;
};

struct llama_data_buffer_context : llama_data_context {
    uint8_t * ptr;
    size_t size_written = 0;

    llama_data_buffer_context(uint8_t * p) : ptr(p) {}

    void write(const void * src, size_t size) override {
        memcpy(ptr, src, size);
        ptr += size;
        size_written += size;
    }

    size_t get_size_written() override {
        return size_written;
    }
};

struct llama_data_file_context : llama_data_context {
    llama_file * file;
    size_t size_written = 0;

    llama_data_file_context(llama_file * f) : file(f) {}

    void write(const void * src, size_t size) override {
        file->write_raw(src, size);
        size_written += size;
    }

    size_t get_size_written() override {
        return size_written;
    }
};

/** copy state data into either a buffer or file depending on the passed in context
 *
 * file context:
 * llama_file file("/path", "wb");
 * llama_data_file_context data_ctx(&file);
 * llama_copy_state_data(ctx, &data_ctx);
 *
 * buffer context:
 * std::vector<uint8_t> buf(max_size, 0);
 * llama_data_buffer_context data_ctx(&buf.data());
 * llama_copy_state_data(ctx, &data_ctx);
 *
*/
static void llama_copy_state_data_internal(struct llama_context * ctx, llama_data_context * data_ctx) {
    // copy rng
    {
        std::stringstream rng_ss;
        rng_ss << ctx->rng;

        const size_t rng_size = rng_ss.str().size();
        char rng_buf[LLAMA_MAX_RNG_STATE];

        memset(&rng_buf[0], 0, LLAMA_MAX_RNG_STATE);
        memcpy(&rng_buf[0], rng_ss.str().data(), rng_ss.str().size());

        data_ctx->write(&rng_size,   sizeof(rng_size));
        data_ctx->write(&rng_buf[0], LLAMA_MAX_RNG_STATE);
    }

    // copy logits
    {
        const size_t logits_cap  = ctx->logits.capacity();
        const size_t logits_size = ctx->logits.size();

        data_ctx->write(&logits_cap,  sizeof(logits_cap));
        data_ctx->write(&logits_size, sizeof(logits_size));

        if (logits_size) {
            data_ctx->write(ctx->logits.data(), logits_size * sizeof(float));
        }

        // If there is a gap between the size and the capacity, write padding
        size_t padding_size = (logits_cap - logits_size) * sizeof(float);
        if (padding_size > 0) {
            std::vector<uint8_t> padding(padding_size, 0); // Create a buffer filled with zeros
            data_ctx->write(padding.data(), padding_size);
        }
    }

    // copy embeddings
    {
        const size_t embedding_size = ctx->embedding.size();

        data_ctx->write(&embedding_size, sizeof(embedding_size));

        if (embedding_size) {
            data_ctx->write(ctx->embedding.data(), embedding_size * sizeof(float));
        }
    }

    // copy kv cache
    {
        const auto & kv_self = ctx->kv_self;
        const auto & hparams = ctx->model.hparams;
        const auto & cparams = ctx->cparams;

        const auto   n_layer = hparams.n_layer;
        const auto   n_embd  = hparams.n_embd_gqa();
        const auto   n_ctx   = cparams.n_ctx;

        const size_t   kv_buf_size = kv_self.buf.size;
        const uint32_t kv_head     = kv_self.head;
        const uint32_t kv_size     = kv_self.size;

        data_ctx->write(&kv_buf_size, sizeof(kv_buf_size));
        data_ctx->write(&kv_head,     sizeof(kv_head));
        data_ctx->write(&kv_size,     sizeof(kv_size));

        if (kv_buf_size) {
            const size_t elt_size = ggml_element_size(kv_self.k);

            ggml_context * cpy_ctx = ggml_init({ 4096, NULL, /* no_alloc */ true });
            ggml_cgraph gf{};

            ggml_tensor * kout3d = ggml_new_tensor_3d(cpy_ctx, kv_self.k->type, n_embd, kv_head, n_layer);
            std::vector<uint8_t> kout3d_data(ggml_nbytes(kout3d), 0);
            kout3d->data = kout3d_data.data();

            ggml_tensor * vout3d = ggml_new_tensor_3d(cpy_ctx, kv_self.v->type, kv_head, n_embd, n_layer);
            std::vector<uint8_t> vout3d_data(ggml_nbytes(vout3d), 0);
            vout3d->data = vout3d_data.data();

            ggml_tensor * k3d = ggml_view_3d(cpy_ctx, kv_self.k,
                n_embd, kv_head, n_layer,
                elt_size*n_embd, elt_size*n_embd*n_ctx, 0);

            ggml_tensor * v3d = ggml_view_3d(cpy_ctx, kv_self.v,
                kv_head, n_embd, n_layer,
                elt_size*n_ctx, elt_size*n_ctx*n_embd, 0);

            ggml_build_forward_expand(&gf, ggml_cpy(cpy_ctx, k3d, kout3d));
            ggml_build_forward_expand(&gf, ggml_cpy(cpy_ctx, v3d, vout3d));
            ggml_graph_compute_helper(ctx->work_buffer, &gf, /*n_threads*/ 1);

            ggml_free(cpy_ctx);

            // our data is now in the kout3d_data and vout3d_data buffers
            // write them to file
            data_ctx->write(kout3d_data.data(), kout3d_data.size());
            data_ctx->write(vout3d_data.data(), vout3d_data.size());
        }

        for (uint32_t i = 0; i < kv_size; ++i) {
            const auto & cell = kv_self.cells[i];

            const llama_pos pos         = cell.pos;
            const size_t    seq_id_size = cell.seq_id.size();

            data_ctx->write(&pos,         sizeof(pos));
            data_ctx->write(&seq_id_size, sizeof(seq_id_size));

            for (auto seq_id : cell.seq_id) {
                data_ctx->write(&seq_id, sizeof(seq_id));
            }
        }
    }
}

size_t llama_copy_state_data(struct llama_context * ctx, uint8_t * dst) {
    llama_data_buffer_context data_ctx(dst);
    llama_copy_state_data_internal(ctx, &data_ctx);

    return data_ctx.get_size_written();
}

// Sets the state reading from the specified source address
size_t llama_set_state_data(struct llama_context * ctx, uint8_t * src) {
    uint8_t * inp = src;

    // set rng
    {
        size_t rng_size;
        char   rng_buf[LLAMA_MAX_RNG_STATE];

        memcpy(&rng_size,   inp, sizeof(rng_size));    inp += sizeof(rng_size);
        memcpy(&rng_buf[0], inp, LLAMA_MAX_RNG_STATE); inp += LLAMA_MAX_RNG_STATE;

        std::stringstream rng_ss;
        rng_ss.str(std::string(&rng_buf[0], rng_size));
        rng_ss >> ctx->rng;

        GGML_ASSERT(!rng_ss.fail());
    }

    // set logits
    {
        size_t logits_cap;
        size_t logits_size;

        memcpy(&logits_cap,  inp, sizeof(logits_cap));  inp += sizeof(logits_cap);
        memcpy(&logits_size, inp, sizeof(logits_size)); inp += sizeof(logits_size);

        GGML_ASSERT(ctx->logits.capacity() == logits_cap);

        if (logits_size) {
            ctx->logits.resize(logits_size);
            memcpy(ctx->logits.data(), inp, logits_size * sizeof(float));
        }

        inp += logits_cap * sizeof(float);
    }

    // set embeddings
    {
        size_t embedding_size;

        memcpy(&embedding_size, inp, sizeof(embedding_size)); inp += sizeof(embedding_size);

        GGML_ASSERT(ctx->embedding.capacity() == embedding_size);

        if (embedding_size) {
            memcpy(ctx->embedding.data(), inp, embedding_size * sizeof(float));
            inp += embedding_size * sizeof(float);
        }
    }

    // set kv cache
    {
        const auto & kv_self = ctx->kv_self;
        const auto & hparams = ctx->model.hparams;
        const auto & cparams = ctx->cparams;

        const int    n_layer = hparams.n_layer;
        const int    n_embd  = hparams.n_embd_gqa();
        const int    n_ctx   = cparams.n_ctx;

        size_t   kv_buf_size;
        uint32_t kv_head;
        uint32_t kv_size;

        memcpy(&kv_buf_size, inp, sizeof(kv_buf_size)); inp += sizeof(kv_buf_size);
        memcpy(&kv_head,     inp, sizeof(kv_head));     inp += sizeof(kv_head);
        memcpy(&kv_size,     inp, sizeof(kv_size));     inp += sizeof(kv_size);

        if (kv_buf_size) {
            GGML_ASSERT(kv_self.buf.size == kv_buf_size);

            const size_t elt_size = ggml_element_size(kv_self.k);

            ggml_context * cpy_ctx = ggml_init({ 4096, NULL, /* no_alloc */ true });
            ggml_cgraph gf{};

            ggml_tensor * kin3d = ggml_new_tensor_3d(cpy_ctx, kv_self.k->type, n_embd, kv_head, n_layer);
            kin3d->data = (void *) inp;
            inp += ggml_nbytes(kin3d);

            ggml_tensor * vin3d = ggml_new_tensor_3d(cpy_ctx, kv_self.v->type, kv_head, n_embd, n_layer);
            vin3d->data = (void *) inp;
            inp += ggml_nbytes(vin3d);

            ggml_tensor * k3d = ggml_view_3d(cpy_ctx, kv_self.k,
                n_embd, kv_head, n_layer,
                elt_size*n_embd, elt_size*n_embd*n_ctx, 0);

            ggml_tensor * v3d = ggml_view_3d(cpy_ctx, kv_self.v,
                kv_head, n_embd, n_layer,
                elt_size*n_ctx, elt_size*n_ctx*n_embd, 0);

            ggml_build_forward_expand(&gf, ggml_cpy(cpy_ctx, kin3d, k3d));
            ggml_build_forward_expand(&gf, ggml_cpy(cpy_ctx, vin3d, v3d));
            ggml_graph_compute_helper(ctx->work_buffer, &gf, /*n_threads*/ 1);

            ggml_free(cpy_ctx);
        }

        ctx->kv_self.head = kv_head;
        ctx->kv_self.size = kv_size;

        ctx->kv_self.cells.resize(kv_size);

        for (uint32_t i = 0; i < kv_size; ++i) {
            llama_pos pos;
            size_t    seq_id_size;

            memcpy(&pos,         inp, sizeof(pos));         inp += sizeof(pos);
            memcpy(&seq_id_size, inp, sizeof(seq_id_size)); inp += sizeof(seq_id_size);

            ctx->kv_self.cells[i].pos = pos;

            llama_seq_id seq_id;

            for (size_t j = 0; j < seq_id_size; ++j) {
                memcpy(&seq_id, inp, sizeof(seq_id)); inp += sizeof(seq_id);
                ctx->kv_self.cells[i].seq_id.insert(seq_id);
            }
        }
    }

    const size_t nread    = inp - src;
    const size_t max_size = llama_get_state_size(ctx);

    GGML_ASSERT(nread <= max_size);

    return nread;
}

static bool llama_load_session_file_internal(struct llama_context * ctx, const char * path_session, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    llama_file file(path_session, "rb");

    // sanity checks
    {
        const uint32_t magic   = file.read_u32();
        const uint32_t version = file.read_u32();

        if (magic != LLAMA_SESSION_MAGIC || version != LLAMA_SESSION_VERSION) {
            LLAMA_LOG_ERROR("%s : unknown (magic, version) for session file: %08x, %08x\n", __func__, magic, version);
            return false;
        }

        llama_hparams session_hparams;
        file.read_raw(&session_hparams, sizeof(llama_hparams));

        if (session_hparams != ctx->model.hparams) {
            LLAMA_LOG_INFO("%s : model hparams didn't match from session file!\n", __func__);
            return false;
        }
    }

    // load the prompt
    {
        const uint32_t n_token_count = file.read_u32();

        if (n_token_count > n_token_capacity) {
            LLAMA_LOG_ERROR("%s : token count in session file exceeded capacity! %u > %zu\n", __func__, n_token_count, n_token_capacity);
            return false;
        }

        file.read_raw(tokens_out, sizeof(llama_token) * n_token_count);
        *n_token_count_out = n_token_count;
    }

    // restore the context state
    {
        const size_t n_state_size_cur = file.size - file.tell();
        const size_t n_state_size_max = llama_get_state_size(ctx);

        if (n_state_size_cur > n_state_size_max) {
            LLAMA_LOG_ERROR("%s : the state size in session file is too big! max %zu, got %zu\n", __func__, n_state_size_max, n_state_size_cur);
            return false;
        }

        std::vector<uint8_t> state_data(n_state_size_max);
        file.read_raw(state_data.data(), n_state_size_cur);

        llama_set_state_data(ctx, state_data.data());
    }

    return true;
}

bool llama_load_session_file(struct llama_context * ctx, const char * path_session, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    try {
        return llama_load_session_file_internal(ctx, path_session, tokens_out, n_token_capacity, n_token_count_out);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("error loading session file: %s\n", err.what());
        return false;
    }
}

bool llama_save_session_file(struct llama_context * ctx, const char * path_session, const llama_token * tokens, size_t n_token_count) {
    llama_file file(path_session, "wb");

    file.write_u32(LLAMA_SESSION_MAGIC);
    file.write_u32(LLAMA_SESSION_VERSION);

    file.write_raw(&ctx->model.hparams, sizeof(llama_hparams));

    // save the prompt
    file.write_u32((uint32_t) n_token_count);
    file.write_raw(tokens, sizeof(llama_token) * n_token_count);

    // save the context state using stream saving
    llama_data_file_context data_ctx(&file);
    llama_copy_state_data_internal(ctx, &data_ctx);

    return true;
}

int llama_eval(
        struct llama_context * ctx,
                 llama_token * tokens,
                     int32_t   n_tokens,
                         int   n_past) {
    llama_kv_cache_tokens_rm(ctx->kv_self, n_past, -1);

    const int ret = llama_decode_internal(*ctx, llama_batch_get_one(tokens, n_tokens, n_past, 0));
    if (ret < 0) {
        LLAMA_LOG_ERROR("%s: failed to decode, ret = %d\n", __func__, ret);
    }

    return ret;
}

int llama_eval_embd(
            struct llama_context * ctx,
                           float * embd,
                         int32_t   n_tokens,
                             int   n_past) {
    llama_kv_cache_tokens_rm(ctx->kv_self, n_past, -1);

    llama_batch batch = { n_tokens, nullptr, embd, nullptr, nullptr, nullptr, nullptr, n_past, 1, 0, };

    const int ret = llama_decode_internal(*ctx, batch);
    if (ret < 0) {
        LLAMA_LOG_ERROR("%s: failed to decode, ret = %d\n", __func__, ret);
    }

    return ret;
}

void llama_set_n_threads(struct llama_context * ctx, uint32_t n_threads, uint32_t n_threads_batch) {
    ctx->cparams.n_threads       = n_threads;
    ctx->cparams.n_threads_batch = n_threads_batch;
}

struct llama_batch llama_batch_get_one(
             llama_token * tokens,
                 int32_t   n_tokens,
               llama_pos   pos_0,
            llama_seq_id   seq_id) {
    return {
        /*n_tokens       =*/ n_tokens,
        /*tokens         =*/ tokens,
        /*embd           =*/ nullptr,
        /*pos            =*/ nullptr,
        /*n_seq_id       =*/ nullptr,
        /*seq_id         =*/ nullptr,
        /*logits         =*/ nullptr,
        /*all_pos_0      =*/ pos_0,
        /*all_pos_1      =*/ 1,
        /*all_seq_id     =*/ seq_id,
    };
}

struct llama_batch llama_batch_init(int32_t n_tokens, int32_t embd, int32_t n_seq_max) {
    llama_batch batch = { 0, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 0, 0, 0, };

    if (embd) {
        batch.embd = (float *) malloc(sizeof(float) * n_tokens * embd);
    } else {
        batch.token = (llama_token *) malloc(sizeof(llama_token) * n_tokens);
    }

    batch.pos      = (llama_pos *)     malloc(sizeof(llama_pos)      * n_tokens);
    batch.n_seq_id = (int32_t *)       malloc(sizeof(int32_t)        * n_tokens);
    batch.seq_id   = (llama_seq_id **) malloc(sizeof(llama_seq_id *) * n_tokens);
    for (int i = 0; i < n_tokens; ++i) {
        batch.seq_id[i] = (llama_seq_id *) malloc(sizeof(llama_seq_id) * n_seq_max);
    }
    batch.logits   = (int8_t *)        malloc(sizeof(int8_t)         * n_tokens);

    return batch;
}

void llama_batch_free(struct llama_batch batch) {
    if (batch.token)    free(batch.token);
    if (batch.embd)     free(batch.embd);
    if (batch.pos)      free(batch.pos);
    if (batch.n_seq_id) free(batch.n_seq_id);
    if (batch.seq_id) {
        for (int i = 0; i < batch.n_tokens; ++i) {
            free(batch.seq_id[i]);
        }
        free(batch.seq_id);
    }
    if (batch.logits)   free(batch.logits);
}

int llama_decode(
        struct llama_context * ctx,
          struct llama_batch   batch) {
    const int ret = llama_decode_internal(*ctx, batch);
    if (ret < 0) {
        LLAMA_LOG_ERROR("%s: failed to decode, ret = %d\n", __func__, ret);
    }

    return ret;
}

float * llama_get_logits(struct llama_context * ctx) {
    return ctx->logits.data();
}

float * llama_get_logits_ith(struct llama_context * ctx, int32_t i) {
    return ctx->logits.data() + i*ctx->model.hparams.n_vocab;
}

float * llama_get_embeddings(struct llama_context * ctx) {
    return ctx->embedding.data();
}

const char * llama_token_get_text(const struct llama_context * ctx, llama_token token) {
    return ctx->model.vocab.id_to_token[token].text.c_str();
}

float llama_token_get_score(const struct llama_context * ctx, llama_token token) {
    return ctx->model.vocab.id_to_token[token].score;
}

llama_token_type llama_token_get_type(const struct llama_context * ctx, llama_token token) {
    return ctx->model.vocab.id_to_token[token].type;
}

llama_token llama_token_bos(const struct llama_context * ctx) {
    return ctx->model.vocab.special_bos_id;
}

llama_token llama_token_eos(const struct llama_context * ctx) {
    return ctx->model.vocab.special_eos_id;
}

llama_token llama_token_nl(const struct llama_context * ctx) {
    return ctx->model.vocab.linefeed_id;
}
llama_token llama_token_prefix(const struct llama_context * ctx) {
    return ctx->model.vocab.special_prefix_id;
}

llama_token llama_token_middle(const struct llama_context * ctx) {
    return ctx->model.vocab.special_middle_id;
}

llama_token llama_token_suffix(const struct llama_context * ctx) {
    return ctx->model.vocab.special_suffix_id;
}

llama_token llama_token_eot(const struct llama_context * ctx) {
    return ctx->model.vocab.special_eot_id;
}

int llama_tokenize(
    const struct llama_model * model,
                  const char * text,
                         int   text_len,
                 llama_token * tokens,
                         int   n_max_tokens,
                        bool   add_bos,
                        bool   special) {
    auto res = llama_tokenize_internal(model->vocab, std::string(text, text_len), add_bos, special);

    if (n_max_tokens < (int) res.size()) {
        // LLAMA_LOG_ERROR("%s: too many tokens\n", __func__);
        return -((int) res.size());
    }

    for (size_t i = 0; i < res.size(); i++) {
        tokens[i] = res[i];
    }

    return res.size();
}

static std::string llama_decode_text(const std::string & text) {
    std::string decoded_text;
    auto unicode_sequences = codepoints_from_utf8(text);
    for (auto& unicode_sequence : unicode_sequences) {
        decoded_text += unicode_to_bytes_bpe(codepoint_to_utf8(unicode_sequence));
    }

    return decoded_text;
}

// does not write null-terminator to buf
int llama_token_to_piece(const struct llama_model * model, llama_token token, char * buf, int length) {
    if (0 <= token && token < llama_n_vocab(model)) {
        switch (llama_vocab_get_type(model->vocab)) {
        case LLAMA_VOCAB_TYPE_SPM: {
            if (llama_is_normal_token(model->vocab, token)) {
                std::string result = model->vocab.id_to_token[token].text;
                llama_unescape_whitespace(result);
                if (length < (int) result.length()) {
                    return -result.length();
                }
                memcpy(buf, result.c_str(), result.length());
                return result.length();
            } else if (llama_is_unknown_token(model->vocab, token)) { // NOLINT
                if (length < 3) {
                    return -3;
                }
                memcpy(buf, "\xe2\x96\x85", 3);
                return 3;
            } else if (llama_is_control_token(model->vocab, token)) {
                ;
            } else if (llama_is_byte_token(model->vocab, token)) {
                if (length < 1) {
                    return -1;
                }
                buf[0] = llama_token_to_byte(model->vocab, token);
                return 1;
            } else {
                // TODO: for now we accept all unsupported token types,
                // suppressing them like CONTROL tokens.
                // GGML_ASSERT(false);
            }
            break;
        }
        case LLAMA_VOCAB_TYPE_BPE: {
            if (llama_is_normal_token(model->vocab, token)) {
                std::string result = model->vocab.id_to_token[token].text;
                result = llama_decode_text(result);
                if (length < (int) result.length()) {
                    return -result.length();
                }
                memcpy(buf, result.c_str(), result.length());
                return result.length();
            } else if (llama_is_control_token(model->vocab, token)) {
                ;
            } else {
                // TODO: for now we accept all unsupported token types,
                // suppressing them like CONTROL tokens.
                // GGML_ASSERT(false);
            }
            break;
        }
        default:
            GGML_ASSERT(false);
        }
    }
    return 0;
}

struct llama_timings llama_get_timings(struct llama_context * ctx) {
    struct llama_timings result = {
        /*.t_start_ms  =*/ 1e-3 * ctx->t_start_us,
        /*.t_end_ms    =*/ 1.00 * ggml_time_ms(),
        /*.t_load_ms   =*/ 1e-3 * ctx->t_load_us,
        /*.t_sample_ms =*/ 1e-3 * ctx->t_sample_us,
        /*.t_p_eval_ms =*/ 1e-3 * ctx->t_p_eval_us,
        /*.t_eval_ms   =*/ 1e-3 * ctx->t_eval_us,

        /*.n_sample =*/ std::max(1, ctx->n_sample),
        /*.n_p_eval =*/ std::max(1, ctx->n_p_eval),
        /*.n_eval   =*/ std::max(1, ctx->n_eval),
    };

    return result;
}

void llama_print_timings(struct llama_context * ctx) {
    const llama_timings timings = llama_get_timings(ctx);

    LLAMA_LOG_INFO("\n");
    LLAMA_LOG_INFO("%s:        load time = %10.2f ms\n", __func__, timings.t_load_ms);
    LLAMA_LOG_INFO("%s:      sample time = %10.2f ms / %5d runs   (%8.2f ms per token, %8.2f tokens per second)\n",
            __func__, timings.t_sample_ms, timings.n_sample, timings.t_sample_ms / timings.n_sample, 1e3 / timings.t_sample_ms * timings.n_sample);
    LLAMA_LOG_INFO("%s: prompt eval time = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n",
            __func__, timings.t_p_eval_ms, timings.n_p_eval, timings.t_p_eval_ms / timings.n_p_eval, 1e3 / timings.t_p_eval_ms * timings.n_p_eval);
    LLAMA_LOG_INFO("%s:        eval time = %10.2f ms / %5d runs   (%8.2f ms per token, %8.2f tokens per second)\n",
            __func__, timings.t_eval_ms, timings.n_eval, timings.t_eval_ms / timings.n_eval, 1e3 / timings.t_eval_ms * timings.n_eval);
    LLAMA_LOG_INFO("%s:       total time = %10.2f ms\n", __func__, (timings.t_end_ms - timings.t_start_ms));
}

void llama_reset_timings(struct llama_context * ctx) {
    ctx->t_start_us = ggml_time_us();
    ctx->t_sample_us = ctx->n_sample = 0;
    ctx->t_eval_us   = ctx->n_eval   = 0;
    ctx->t_p_eval_us = ctx->n_p_eval = 0;
}

const char * llama_print_system_info(void) {
    static std::string s;

    s  = "";
    s += "AVX = "         + std::to_string(ggml_cpu_has_avx())         + " | ";
    s += "AVX2 = "        + std::to_string(ggml_cpu_has_avx2())        + " | ";
    s += "AVX512 = "      + std::to_string(ggml_cpu_has_avx512())      + " | ";
    s += "AVX512_VBMI = " + std::to_string(ggml_cpu_has_avx512_vbmi()) + " | ";
    s += "AVX512_VNNI = " + std::to_string(ggml_cpu_has_avx512_vnni()) + " | ";
    s += "FMA = "         + std::to_string(ggml_cpu_has_fma())         + " | ";
    s += "NEON = "        + std::to_string(ggml_cpu_has_neon())        + " | ";
    s += "ARM_FMA = "     + std::to_string(ggml_cpu_has_arm_fma())     + " | ";
    s += "F16C = "        + std::to_string(ggml_cpu_has_f16c())        + " | ";
    s += "FP16_VA = "     + std::to_string(ggml_cpu_has_fp16_va())     + " | ";
    s += "WASM_SIMD = "   + std::to_string(ggml_cpu_has_wasm_simd())   + " | ";
    s += "BLAS = "        + std::to_string(ggml_cpu_has_blas())        + " | ";
    s += "SSE3 = "        + std::to_string(ggml_cpu_has_sse3())        + " | ";
    s += "SSSE3 = "       + std::to_string(ggml_cpu_has_ssse3())       + " | ";
    s += "VSX = "         + std::to_string(ggml_cpu_has_vsx())         + " | ";

    return s.c_str();
}

void llama_dump_timing_info_yaml(FILE * stream, const llama_context * ctx) {
    fprintf(stream, "\n");
    fprintf(stream, "###########\n");
    fprintf(stream, "# Timings #\n");
    fprintf(stream, "###########\n");
    fprintf(stream, "\n");

    fprintf(stream, "mst_eval: %.2f  # ms / token during generation\n",
            1.0e-3 * ctx->t_eval_us / ctx->n_eval);
    fprintf(stream, "mst_p_eval: %.2f  # ms / token during prompt processing\n",
            1.0e-3 * ctx->t_p_eval_us / ctx->n_p_eval);
    fprintf(stream, "mst_sample: %.2f  # ms / token during sampling\n",
            1.0e-3 * ctx->t_sample_us / ctx->n_sample);
    fprintf(stream, "n_eval: %d  # number of tokens generated (excluding the first one)\n", ctx->n_eval);
    fprintf(stream, "n_p_eval: %d  # number of tokens processed in batches at the beginning\n", ctx->n_p_eval);
    fprintf(stream, "n_sample: %d  # number of sampled tokens\n", ctx->n_sample);
    fprintf(stream, "t_eval_us: %" PRId64 "  # total microseconds spent generating tokens\n", ctx->t_eval_us);
    fprintf(stream, "t_load_us: %" PRId64 "  # total microseconds spent loading the model\n", ctx->t_load_us);
    fprintf(stream, "t_p_eval_us: %" PRId64 "  # total microseconds spent prompt processing\n", ctx->t_p_eval_us);
    fprintf(stream, "t_sample_us: %" PRId64 "  # total microseconds spent sampling\n", ctx->t_sample_us);
    fprintf(stream, "ts_eval: %.2f  # tokens / second during generation\n",
            1.0e6 * ctx->n_eval / ctx->t_eval_us);
    fprintf(stream, "ts_p_eval: %.2f  # tokens / second during prompt processing\n",
            1.0e6 * ctx->n_p_eval / ctx->t_p_eval_us);
    fprintf(stream, "ts_sample: %.2f  # tokens / second during sampling\n",
            1.0e6 * ctx->n_sample / ctx->t_sample_us);
}

// For internal test use
const std::vector<std::pair<std::string, struct ggml_tensor *>> & llama_internal_get_tensor_map(
    struct llama_context * ctx
) {
    return ctx->model.tensors_by_name;
}

void llama_log_set(ggml_log_callback log_callback, void * user_data) {
    g_state.log_callback = log_callback ? log_callback : llama_log_callback_default;
    g_state.log_callback_user_data = user_data;
}

static void llama_log_internal_v(ggml_log_level level, const char * format, va_list args) {
    va_list args_copy;
    va_copy(args_copy, args);
    char buffer[128];
    int len = vsnprintf(buffer, 128, format, args);
    if (len < 128) {
        g_state.log_callback(level, buffer, g_state.log_callback_user_data);
    } else {
        char* buffer2 = new char[len+1];
        vsnprintf(buffer2, len+1, format, args_copy);
        buffer2[len] = 0;
        g_state.log_callback(level, buffer2, g_state.log_callback_user_data);
        delete[] buffer2;
    }
    va_end(args_copy);
}

static void llama_log_internal(ggml_log_level level, const char * format, ...) {
    va_list args;
    va_start(args, format);
    llama_log_internal_v(level, format, args);
    va_end(args);
}

static void llama_log_callback_default(ggml_log_level level, const char * text, void * user_data) {
    (void) level;
    (void) user_data;
    fputs(text, stderr);
    fflush(stderr);
}
