#pragma once

// Ollama-format GGUF compatibility shim.
//
// Older Ollama builds ship GGUFs that differ from upstream in a handful of
// ways per-architecture (arch names, KV keys, tensor names, file layout).
// This shim detects those files during load and translates them in-memory
// so the rest of llama.cpp can load them unmodified.
//
// Three upstream hook points call into this namespace — one per insertion:
//
//   1. llama-model-loader.cpp (main model load):
//        translate_metadata()        — mutate KVs / tensor metadata
//        should_skip_tensor()        — filter weights_map population
//
//   2. tools/mtmd/clip.cpp (mmproj load):
//        translate_clip_metadata()   — rewrite KVs + tensor names for clip
//        maybe_load_tensor()         — override file read (e.g. F16->F32)
//
// Detection is per-arch; for any non-Ollama file every entry point is a
// no-op. Per-arch logic lives in anonymous-namespace handle_<arch>()
// functions in the .cpp; adding a new arch is a new handler plus one
// dispatch line in each translate_* entry point.

#include <cstddef>
#include <string>

#include "ggml-backend.h" // for ggml_backend_buffer_type_t

struct gguf_context;
struct ggml_context;
struct ggml_tensor;
struct llama_model_loader;

namespace llama_ollama_compat {

// Called from llama_model_loader's constructor, right after the arch is read.
void translate_metadata(const llama_model_loader * ml,
                        gguf_context * meta,
                        ggml_context * ctx,
                        std::string & arch_name);

// Called from llama_model_loader's weights_map population loop. Returns
// true to drop a tensor from the loader — used to hide embedded vision
// tensors from the text model's view without modifying the gguf_context.
bool should_skip_tensor(const llama_model_loader * ml, const char * tensor_name);

// Called from clip_model_loader's constructor. Rewrites the clip-facing
// view of the metadata (arch=clip, clip.vision.* KVs, renamed tensors)
// so the rest of clip.cpp can load an Ollama monolithic GGUF unchanged.
void translate_clip_metadata(gguf_context * meta, ggml_context * ctx);

// Called from clip.cpp's tensor-loading loop, before the normal file read.
// If this tensor was marked for type promotion by translate_clip_metadata
// (e.g. F16->F32), performs the conversion and writes the result into
// `cur` (host memcpy or backend_tensor_set based on `buft`). Returns true
// when the tensor was handled — caller should skip its normal read path.
bool maybe_load_tensor(ggml_tensor * cur,
                       const char * source_file,
                       size_t file_offset,
                       ggml_backend_buffer_type_t buft);

} // namespace llama_ollama_compat
