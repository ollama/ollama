#ifndef OLLAMA_XGRAMMAR_H
#define OLLAMA_XGRAMMAR_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#if defined(OLLAMA_XGRAMMAR_BUILD)
#define OLLAMA_XGRAMMAR_API __declspec(dllexport)
#else
#define OLLAMA_XGRAMMAR_API __declspec(dllimport)
#endif
#else
#define OLLAMA_XGRAMMAR_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ollama_xgrammar_compiler ollama_xgrammar_compiler;
typedef struct ollama_xgrammar_matcher ollama_xgrammar_matcher;

typedef enum ollama_xgrammar_kind {
    OLLAMA_XGRAMMAR_JSON_SCHEMA = 0,
} ollama_xgrammar_kind;

// The pinned xgrammar release the library was built from.
OLLAMA_XGRAMMAR_API const char* ollama_xgrammar_version(void);

OLLAMA_XGRAMMAR_API const char* ollama_xgrammar_last_error(void);

OLLAMA_XGRAMMAR_API int ollama_xgrammar_compiler_new(
    const char* token_data,
    size_t token_data_size,
    const uint64_t* token_offsets,
    size_t token_count,
    int32_t vocab_size,
    const int32_t* stop_token_ids,
    size_t stop_token_count,
    int32_t max_threads,
    int64_t cache_bytes,
    ollama_xgrammar_compiler** compiler);
OLLAMA_XGRAMMAR_API void ollama_xgrammar_compiler_free(ollama_xgrammar_compiler* compiler);

OLLAMA_XGRAMMAR_API int ollama_xgrammar_matcher_new(
    ollama_xgrammar_compiler* compiler,
    ollama_xgrammar_kind kind,
    const char* source,
    size_t source_size,
    ollama_xgrammar_matcher** matcher);
OLLAMA_XGRAMMAR_API void ollama_xgrammar_matcher_free(ollama_xgrammar_matcher* matcher);

OLLAMA_XGRAMMAR_API int ollama_xgrammar_matcher_fill(
    ollama_xgrammar_matcher* matcher,
    int32_t* bitmask,
    size_t bitmask_words,
    int* needs_apply);
OLLAMA_XGRAMMAR_API int ollama_xgrammar_matcher_accept(
    ollama_xgrammar_matcher* matcher,
    int32_t token_id,
    int* accepted);

#ifdef __cplusplus
}
#endif

#endif
