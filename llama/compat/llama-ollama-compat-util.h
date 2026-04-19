#pragma once

// Internal helpers shared by the per-architecture handlers in
// llama-ollama-compat.cpp. Not part of the public API.
//
// Everything lives under namespace llama_ollama_compat::detail. The
// definitions live in llama-ollama-compat-util.cpp, which also owns the
// registry globals (tensor skip list, load-op table) that need a single
// translation unit.
//
// ---- Non-public API dependencies (see also README.md "Maintenance") ----
//
// Mostly public: gguf_* and ggml_* accessors from ggml/include/ are all
// stable. `ggml_backend_*` and `ggml_fp16_to_fp32` are stable too.
//
// Three pieces we rely on that aren't strictly guaranteed public:
//
//  1. Direct writes to `ggml_tensor::type`, `ne[]`, `nb[]` — the struct is
//     public and fields are spec'd, but there's no sanctioned mutator for
//     them post-creation. Used in set_tensor_type / set_tensor_shape /
//     reclaim_slot_as. Risk: upstream could in principle introduce an
//     opaque-tensor mode; in practice it hasn't in years.
//
//  2. `const_cast<char *>(gguf_get_tensor_name(...))` in rename_tensor.
//     The pointer returned points into a mutable char[GGML_MAX_NAME]
//     buffer inside a std::vector element. Defined behavior as long as
//     upstream keeps name storage in-line (has done so forever).
//
//  3. `llama_model_loader` forward decl from src/llama-model-loader.h
//     (internal, not llama.h). Only used as an opaque pointer key for
//     the skip-prefix registry — we never dereference it. Could swap for
//     `const void *` if upstream ever moved that type around.
//
// All three are trivially replaceable if upstream changes out from under
// us. See llama/compat/README.md for the escape hatches.

#include <cstddef>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <string>
#include <vector>

#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"

struct llama_model_loader;

namespace llama_ollama_compat::detail {

// -- gguf_context KV helpers --
bool has_key(const gguf_context * meta, const char * key);
void copy_u32_kv(gguf_context * meta, const char * src, const char * dst);
void copy_f32_kv(gguf_context * meta, const char * src, const char * dst);
void inject_u32_if_missing (gguf_context * meta, const char * key, uint32_t v);
void inject_f32_if_missing (gguf_context * meta, const char * key, float    v);
void inject_str_if_missing (gguf_context * meta, const char * key, const char * v);
void inject_bool_if_missing(gguf_context * meta, const char * key, bool     v);
void inject_f32_arr_if_missing(gguf_context * meta, const char * key,
                               const float * data, size_t n);
void truncate_str_arr (gguf_context * meta, const char * key, size_t new_n);
void truncate_data_arr(gguf_context * meta, const char * key,
                       gguf_type elem_type, size_t elem_size, size_t new_n);

// -- ggml_context tensor scans --
bool any_tensor_with_prefix(const ggml_context * ctx, const char * prefix);

// -- Tensor renaming / reshaping (mutates both gguf_context and ggml_context) --
void rename_tensor(gguf_context * meta, ggml_context * ctx,
                   const char * old_name, const char * new_name);
void rename_tensors_containing(gguf_context * meta, ggml_context * ctx,
                               const char * needle, const char * replacement);
void set_tensor_type (ggml_tensor * t, ggml_type type);
void set_tensor_shape(ggml_tensor * t, std::initializer_list<int64_t> shape);
bool reclaim_slot_as (gguf_context * meta, ggml_context * ctx,
                      const char * orphan_name, const char * new_name,
                      std::initializer_list<int64_t> shape, ggml_type type);

// -- File-offset capture (before rename) --
size_t tensor_file_offset(const gguf_context * meta, const char * name);

// -- Per-loader skip-prefix registry --
void add_skip_prefix(const llama_model_loader * ml, std::string prefix);
bool should_skip_tensor_prefix(const llama_model_loader * ml, const char * name);

// -- Load-time transform registry --
struct LoadOp {
    std::function<bool(const char * src_file, void * dst, size_t dst_size)> apply;
    const char * description;
};
void register_load_op(std::string dest_name, LoadOp op);
bool take_load_op    (const char * dest_name, LoadOp & out); // removes + returns

// Read `size` bytes at `offset` from `path` into `dst`. Used by LoadOps.
bool read_at(const char * path, size_t offset, void * dst, size_t size);

// -- Common high-level transforms --

// F16 -> F32 promotion. Captures the source file offset at registration
// time so later renames/reshapes of this tensor don't invalidate the read.
void promote_tensor_to_f32(gguf_context * meta, ggml_context * ctx, const char * name);

// Concatenate N source tensors into one destination. Captures each source's
// file offset + byte size at registration time. Layout assumption: sources
// concatenate cleanly along the destination's slow ggml axis, which in
// C order means the destination bytes are src[0] || src[1] || ... .
void register_concat_load(const gguf_context * meta, std::string dest_name,
                          const std::vector<std::string> & src_names);

} // namespace llama_ollama_compat::detail
