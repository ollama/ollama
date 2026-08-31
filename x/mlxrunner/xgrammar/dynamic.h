#ifndef OLLAMA_XGRAMMAR_DYNAMIC_H
#define OLLAMA_XGRAMMAR_DYNAMIC_H

#include "native/xgrammar.h"

typedef struct ollama_xgrammar_dynamic_handle {
    void* ctx;
} ollama_xgrammar_dynamic_handle;

int ollama_xgrammar_dynamic_load(ollama_xgrammar_dynamic_handle* handle, const char* path);
const char* ollama_xgrammar_dynamic_version(void);
const char* ollama_xgrammar_dynamic_error(void);

int ollama_xgrammar_dynamic_compiler_new(
    const char*, size_t, const uint64_t*, size_t, int32_t,
    const int32_t*, size_t, int32_t, int64_t, ollama_xgrammar_compiler**, char**);
void ollama_xgrammar_dynamic_compiler_free(ollama_xgrammar_compiler*);
int ollama_xgrammar_dynamic_matcher_new(
    ollama_xgrammar_compiler*, ollama_xgrammar_kind, const char*, size_t,
    ollama_xgrammar_matcher**, char**);
void ollama_xgrammar_dynamic_matcher_free(ollama_xgrammar_matcher*);
int ollama_xgrammar_dynamic_matcher_fill(ollama_xgrammar_matcher*, int32_t*, size_t, int*, char**);
int ollama_xgrammar_dynamic_matcher_accept(ollama_xgrammar_matcher*, int32_t, int*, char**);

#endif
