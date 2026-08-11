#ifndef OLLAMA_CONSTRAINT_DYNAMIC_H
#define OLLAMA_CONSTRAINT_DYNAMIC_H

#include "native/constraint.h"

typedef struct ollama_constraint_dynamic_handle {
    void* ctx;
} ollama_constraint_dynamic_handle;

int ollama_constraint_dynamic_load(ollama_constraint_dynamic_handle* handle, const char* path);
const char* ollama_constraint_dynamic_error(void);

int ollama_constraint_dynamic_model_new(
	const char*, size_t, const uint64_t*, size_t, int32_t,
	const int32_t*, size_t, ollama_constraint_model**, char**);
void ollama_constraint_dynamic_model_free(ollama_constraint_model*);
int ollama_constraint_dynamic_matcher_new(
	ollama_constraint_model*, ollama_constraint_kind, const char*, size_t,
	ollama_constraint_matcher**, char**);
void ollama_constraint_dynamic_matcher_free(ollama_constraint_matcher*);
int ollama_constraint_dynamic_matcher_fill(ollama_constraint_matcher*, int32_t*, size_t, int*, char**);
int ollama_constraint_dynamic_matcher_accept(ollama_constraint_matcher*, int32_t, int*, char**);

#endif
