#include "constraint.h"

#include <dlpack/dlpack.h>
#include <xgrammar/xgrammar.h>

#include <algorithm>
#include <exception>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

thread_local std::string last_error;

constexpr int32_t kMaxVocabSize = 1 << 20;
constexpr size_t kMaxSourceSize = 1 << 20;

template <typename F>
int protect(F&& fn) {
    try {
        last_error.clear();
        fn();
        return 0;
    } catch (const std::exception& e) {
        last_error = e.what();
    } catch (...) {
        last_error = "unknown constraint library error";
    }
    return 1;
}

}  // namespace

struct ollama_constraint_model {
    int32_t vocab_size;
    xgrammar::TokenizerInfo tokenizer;
    xgrammar::GrammarCompiler compiler;

    ollama_constraint_model(std::vector<std::string> vocab, int32_t size, std::vector<int32_t> stops)
        : vocab_size(size),
          tokenizer(vocab, xgrammar::VocabType::RAW, size, std::move(stops)),
          compiler(tokenizer, /*max_threads=*/8, /*cache_enabled=*/false) {}
};

struct ollama_constraint_matcher {
    int32_t vocab_size;
    xgrammar::GrammarMatcher matcher;

    ollama_constraint_matcher(int32_t size, xgrammar::CompiledGrammar grammar)
        : vocab_size(size), matcher(grammar) {}
};

extern "C" {

const char* ollama_constraint_last_error(void) {
    return last_error.c_str();
}

int ollama_constraint_model_new(
    const char* token_data,
    size_t token_data_size,
    const uint64_t* token_offsets,
    size_t token_count,
    int32_t vocab_size,
    const int32_t* stop_token_ids,
    size_t stop_token_count,
    ollama_constraint_model** model) {
    return protect([&] {
        if (model == nullptr) {
            throw std::invalid_argument("model output is null");
        }
        *model = nullptr;
        if (vocab_size <= 0 || vocab_size > kMaxVocabSize ||
            token_count > static_cast<size_t>(vocab_size)) {
            throw std::invalid_argument("invalid vocabulary size");
        }
        if (token_count > 0 && token_offsets == nullptr) {
            throw std::invalid_argument("token offsets are null");
        }
        if (token_data_size > 0 && token_data == nullptr) {
            throw std::invalid_argument("token data is null");
        }
        if (stop_token_count == 0 || stop_token_ids == nullptr) {
            throw std::invalid_argument("at least one stop token is required");
        }
        if (stop_token_count > static_cast<size_t>(vocab_size)) {
            throw std::invalid_argument("too many stop tokens");
        }

        std::vector<std::string> vocab;
        vocab.reserve(token_count);
        const char* base = token_data == nullptr ? "" : token_data;
        uint64_t begin = 0;
        for (size_t i = 0; i < token_count; ++i) {
            uint64_t end = token_offsets[i];
            if (end < begin || end > token_data_size) {
                throw std::invalid_argument("invalid token offsets");
            }
            vocab.emplace_back(base + begin, static_cast<size_t>(end - begin));
            begin = end;
        }
        if (begin != token_data_size) {
            throw std::invalid_argument("token offsets do not consume token data");
        }

        std::vector<int32_t> stops(stop_token_ids, stop_token_ids + stop_token_count);
        for (int32_t id : stops) {
            if (id < 0 || id >= vocab_size) {
                throw std::invalid_argument("stop token is outside vocabulary");
            }
        }
        *model = new ollama_constraint_model(std::move(vocab), vocab_size, std::move(stops));
    });
}

void ollama_constraint_model_free(ollama_constraint_model* model) {
    delete model;
}

int ollama_constraint_matcher_new(
    ollama_constraint_model* model,
    ollama_constraint_kind kind,
    const char* source,
    size_t source_size,
    ollama_constraint_matcher** matcher) {
    return protect([&] {
        if (model == nullptr || matcher == nullptr) {
            throw std::invalid_argument("model or matcher output is null");
        }
        *matcher = nullptr;
        if (source_size > 0 && source == nullptr) {
            throw std::invalid_argument("constraint source is null");
        }
        if (source_size > kMaxSourceSize) {
            throw std::invalid_argument("constraint source is too large");
        }
        const char* source_data = source == nullptr ? "" : source;

        xgrammar::CompiledGrammar compiled = [&]() -> xgrammar::CompiledGrammar {
            switch (kind) {
            case OLLAMA_CONSTRAINT_JSON:
                return model->compiler.CompileBuiltinJSONGrammar();
            case OLLAMA_CONSTRAINT_JSON_SCHEMA:
                return model->compiler.CompileJSONSchema(std::string(source_data, source_size));
            default:
                throw std::invalid_argument("unknown constraint kind");
            }
        }();
        *matcher = new ollama_constraint_matcher(model->vocab_size, std::move(compiled));
    });
}

void ollama_constraint_matcher_free(ollama_constraint_matcher* matcher) {
    delete matcher;
}

int ollama_constraint_matcher_fill(
    ollama_constraint_matcher* matcher,
    int32_t* bitmask,
    size_t bitmask_words,
    int* needs_apply) {
    return protect([&] {
        if (matcher == nullptr || bitmask == nullptr || needs_apply == nullptr) {
            throw std::invalid_argument("matcher, bitmask, or result is null");
        }
        size_t expected = static_cast<size_t>(xgrammar::GetBitmaskSize(matcher->vocab_size));
        if (bitmask_words != expected) {
            throw std::invalid_argument("incorrect bitmask size");
        }

        int64_t shape[] = {static_cast<int64_t>(bitmask_words)};
        DLTensor tensor{};
        tensor.data = bitmask;
        tensor.device = DLDevice{kDLCPU, 0};
        tensor.ndim = 1;
        tensor.dtype = xgrammar::GetBitmaskDLType();
        tensor.shape = shape;
        tensor.strides = nullptr;
        tensor.byte_offset = 0;
        *needs_apply = matcher->matcher.FillNextTokenBitmask(&tensor) ? 1 : 0;

        bool any = false;
        for (int32_t id = 0; id < matcher->vocab_size; ++id) {
            if ((static_cast<uint32_t>(bitmask[id / 32]) & (uint32_t{1} << (id % 32))) != 0) {
                any = true;
                break;
            }
        }
        if (!any) {
            throw std::runtime_error("constraint rejects every vocabulary token");
        }
    });
}

int ollama_constraint_matcher_accept(
    ollama_constraint_matcher* matcher,
    int32_t token_id,
    int* accepted) {
    return protect([&] {
        if (matcher == nullptr || accepted == nullptr) {
            throw std::invalid_argument("matcher or result is null");
        }
        if (token_id < 0 || token_id >= matcher->vocab_size) {
            throw std::invalid_argument("token is outside vocabulary");
        }
        *accepted = matcher->matcher.AcceptToken(token_id) ? 1 : 0;
    });
}

}  // extern "C"
