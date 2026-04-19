#pragma once

// Ollama-format GGUF compatibility shim.
//
// Older Ollama builds ship GGUFs that differ from upstream in a handful of
// ways per-architecture. This shim detects those files during load and
// translates them in-memory so the rest of llama.cpp can load them
// unmodified. Single entry point per hook; all logic is data-driven from
// per-architecture rules.
//
// Two hooks:
//   1. translate_metadata() — runs after gguf_init_from_file, mutates KVs
//      and (optionally) tensor names on the gguf_context / ggml_context.
//   2. apply_tensor_transforms() — runs after load_all_data, rewrites
//      tensor data that differs numerically (e.g. gemma3 RMSNorm +1).

#include <cstdint>
#include <string>
#include <vector>

struct gguf_context;
struct ggml_context;
struct ggml_tensor;
struct llama_model_loader;

namespace llama_ollama_compat {

// Inspect and mutate the just-loaded gguf_context. May update arch_name if
// the file uses an Ollama-specific architecture name. Safe to call for any
// model — no-op when no Ollama markers are present.
//
// If compat was applied, registers any tensor transforms against `ml` for
// apply_tensor_transforms() to consume later.
void translate_metadata(const llama_model_loader * ml,
                        gguf_context * meta,
                        ggml_context * ctx,
                        std::string & arch_name);

// Returns true if the loader should skip this tensor entirely (not add to
// weights_map, not count toward n_tensors). Used to drop embedded vision
// tensors from the text model without physically removing them.
bool should_skip_tensor(const llama_model_loader * ml, const char * tensor_name);

// Called after load_all_data returns for a model context. Applies any
// registered transforms (read tensor data from the backend buffer, modify,
// write back) to tensors in `ctx`. Call once per model context.
void apply_tensor_transforms(const llama_model_loader * ml, ggml_context * ctx);

// Called from the clip loader (tools/mtmd/clip.cpp). If the file is an
// Ollama-format monolithic GGUF (text + embedded vision), rewrites the
// clip-facing view of the metadata so the clip loader sees it as a normal
// mmproj file. Safe to call unconditionally — no-op when not an Ollama file.
//
// Operations:
//   - sets general.architecture = "clip"
//   - sets clip.has_vision_encoder, clip.projector_type, clip.use_gelu
//   - copies gemma3.vision.* KVs into clip.vision.*
//   - renames vision tensors (v.patch_embedding -> v.patch_embd, etc.)
//   - promotes specific F16 tensors to F32 in the ggml_context so clip
//     allocates the correct buffer size
//
// Non-vision text tensors remain in the gguf but are never looked up by
// clip, so they cost nothing.
void translate_clip_metadata(gguf_context * meta, ggml_context * ctx);

// Called from clip.cpp's tensor-loading loop, before reading bytes from the
// file. If this tensor was marked for type promotion by translate_clip_metadata,
// fills `out` with the promoted data (e.g. F16→F32) and returns true. The
// caller should then use `out` instead of reading from the file.
//
// `file_offset` is the absolute file offset of the original (pre-promotion)
// tensor data in the source GGUF.
bool supply_promoted_tensor_data(const ggml_tensor * cur,
                                 const char * source_file,
                                 size_t file_offset,
                                 std::vector<uint8_t> & out);

} // namespace llama_ollama_compat
