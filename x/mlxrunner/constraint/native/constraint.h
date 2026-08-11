#ifndef OLLAMA_CONSTRAINT_H
#define OLLAMA_CONSTRAINT_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#if defined(OLLAMA_CONSTRAINT_BUILD)
#define OLLAMA_CONSTRAINT_API __declspec(dllexport)
#else
#define OLLAMA_CONSTRAINT_API __declspec(dllimport)
#endif
#else
#define OLLAMA_CONSTRAINT_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ollama_constraint_model ollama_constraint_model;
typedef struct ollama_constraint_matcher ollama_constraint_matcher;

typedef enum ollama_constraint_kind {
    OLLAMA_CONSTRAINT_JSON = 0,
    OLLAMA_CONSTRAINT_JSON_SCHEMA = 1,
} ollama_constraint_kind;

OLLAMA_CONSTRAINT_API const char* ollama_constraint_last_error(void);

OLLAMA_CONSTRAINT_API int ollama_constraint_model_new(
    const char* token_data,
    size_t token_data_size,
    const uint64_t* token_offsets,
    size_t token_count,
    int32_t vocab_size,
    const int32_t* stop_token_ids,
    size_t stop_token_count,
    ollama_constraint_model** model);
OLLAMA_CONSTRAINT_API void ollama_constraint_model_free(ollama_constraint_model* model);

OLLAMA_CONSTRAINT_API int ollama_constraint_matcher_new(
    ollama_constraint_model* model,
    ollama_constraint_kind kind,
    const char* source,
    size_t source_size,
    ollama_constraint_matcher** matcher);
OLLAMA_CONSTRAINT_API void ollama_constraint_matcher_free(ollama_constraint_matcher* matcher);

OLLAMA_CONSTRAINT_API int ollama_constraint_matcher_fill(
    ollama_constraint_matcher* matcher,
    int32_t* bitmask,
    size_t bitmask_words,
    int* needs_apply);
OLLAMA_CONSTRAINT_API int ollama_constraint_matcher_accept(
    ollama_constraint_matcher* matcher,
    int32_t token_id,
    int* accepted);

#ifdef __cplusplus
}
#endif

#endif
