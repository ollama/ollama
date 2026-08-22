#include "xgrammar.h"

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

// Errors travel into API responses and logs: strip xgrammar's console
// decoration — "[HH:MM:SS] file:line: message\n" — and cap the length,
// since messages can embed schema fragments.
constexpr size_t kMaxErrorSize = 512;

void sanitize_error(std::string& s) {
    if (s.size() >= 11 && s[0] == '[' && s[9] == ']' && s[10] == ' ') {
        for (size_t i = 11; (i = s.find(':', i)) != std::string::npos; ++i) {
            size_t j = i + 1;
            while (j < s.size() && s[j] >= '0' && s[j] <= '9') {
                ++j;
            }
            if (j > i + 1 && j + 1 < s.size() && s[j] == ':' && s[j + 1] == ' ') {
                s.erase(0, j + 2);
                break;
            }
        }
    }
    while (!s.empty() && s.back() == '\n') {
        s.pop_back();
    }
    if (s.size() > kMaxErrorSize) {
        s.resize(kMaxErrorSize);
        // Do not leave a torn UTF-8 sequence at the cut.
        while (!s.empty() && (static_cast<unsigned char>(s.back()) & 0xc0) == 0x80) {
            s.pop_back();
        }
        if (!s.empty() && static_cast<unsigned char>(s.back()) >= 0xc0) {
            s.pop_back();
        }
        s += "...";
    }
}

template <typename F>
int protect(F&& fn) {
    try {
        last_error.clear();
        fn();
        return 0;
    } catch (const std::exception& e) {
        last_error = e.what();
        sanitize_error(last_error);
    } catch (...) {
        last_error = "unknown xgrammar library error";
    }
    return 1;
}

}  // namespace

struct ollama_xgrammar_compiler {
    int32_t vocab_size;
    xgrammar::TokenizerInfo tokenizer;
    xgrammar::GrammarCompiler compiler;

    ollama_xgrammar_compiler(std::vector<std::string> vocab, int32_t size, std::vector<int32_t> stops,
                          int32_t max_threads, int64_t cache_bytes)
        : vocab_size(size),
          tokenizer(vocab, xgrammar::VocabType::RAW, size, std::move(stops)),
          compiler(tokenizer, max_threads, /*cache_enabled=*/cache_bytes > 0,
                   /*max_memory_bytes=*/cache_bytes > 0 ? cache_bytes : 0) {}
};

struct ollama_xgrammar_matcher {
    int32_t vocab_size;
    xgrammar::GrammarMatcher matcher;

    ollama_xgrammar_matcher(int32_t size, xgrammar::CompiledGrammar grammar)
        : vocab_size(size), matcher(grammar) {}
};

#ifndef OLLAMA_XGRAMMAR_VERSION
#define OLLAMA_XGRAMMAR_VERSION "unknown"
#endif

extern "C" {

const char* ollama_xgrammar_version(void) {
    return OLLAMA_XGRAMMAR_VERSION;
}

const char* ollama_xgrammar_last_error(void) {
    return last_error.c_str();
}

int ollama_xgrammar_compiler_new(
    const char* token_data,
    size_t token_data_size,
    const uint64_t* token_offsets,
    size_t token_count,
    int32_t vocab_size,
    const int32_t* stop_token_ids,
    size_t stop_token_count,
    int32_t max_threads,
    int64_t cache_bytes,
    ollama_xgrammar_compiler** compiler) {
    return protect([&] {
        if (compiler == nullptr) {
            throw std::invalid_argument("compiler output is null");
        }
        *compiler = nullptr;
        if (token_count > 0 && token_offsets == nullptr) {
            throw std::invalid_argument("token offsets are null");
        }
        if (token_data_size > 0 && token_data == nullptr) {
            throw std::invalid_argument("token data is null");
        }
        if (stop_token_count > 0 && stop_token_ids == nullptr) {
            throw std::invalid_argument("stop token ids are null");
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
        *compiler = new ollama_xgrammar_compiler(
            std::move(vocab), vocab_size, std::move(stops), max_threads, cache_bytes);
    });
}

int ollama_xgrammar_matcher_new(
    ollama_xgrammar_compiler* compiler,
    ollama_xgrammar_kind kind,
    const char* source,
    size_t source_size,
    ollama_xgrammar_matcher** matcher) {
    return protect([&] {
        if (compiler == nullptr || matcher == nullptr) {
            throw std::invalid_argument("compiler or matcher output is null");
        }
        *matcher = nullptr;
        if (source_size > 0 && source == nullptr) {
            throw std::invalid_argument("grammar source is null");
        }
        const char* source_data = source == nullptr ? "" : source;

        xgrammar::CompiledGrammar compiled = [&]() -> xgrammar::CompiledGrammar {
            switch (kind) {
            case OLLAMA_XGRAMMAR_JSON_SCHEMA:
                return compiler->compiler.CompileJSONSchema(std::string(source_data, source_size));
            default:
                throw std::invalid_argument("unknown grammar kind");
            }
        }();
        *matcher = new ollama_xgrammar_matcher(compiler->vocab_size, std::move(compiled));
    });
}

int ollama_xgrammar_matcher_fill(
    ollama_xgrammar_matcher* matcher,
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
    });
}

int ollama_xgrammar_matcher_accept(
    ollama_xgrammar_matcher* matcher,
    int32_t token_id,
    int* accepted) {
    return protect([&] {
        if (matcher == nullptr || accepted == nullptr) {
            throw std::invalid_argument("matcher or result is null");
        }
        *accepted = matcher->matcher.AcceptToken(token_id) ? 1 : 0;
    });
}

void ollama_xgrammar_matcher_free(ollama_xgrammar_matcher* matcher) {
    delete matcher;
}

void ollama_xgrammar_compiler_free(ollama_xgrammar_compiler* compiler) {
    delete compiler;
}

}  // extern "C"
