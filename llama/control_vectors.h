#ifndef CONTROL_VECTORS_H
#define CONTROL_VECTORS_H

#include "llama.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
Since we probably can't directly import common.h as it is a c++ file, we need to
expose a C api for constructing control vector structs and applying them to the
model.
*/

LLAMA_API int32_t llama_apply_control_vector(
    const struct llama_model * model,
    struct llama_context * ctx,
    char* path,
    float strength
);

#ifdef __cplusplus
}
#endif

#endif


