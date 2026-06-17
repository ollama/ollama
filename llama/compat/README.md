# llama.cpp compatibility layer

This directory holds a temporary in-process compatibility layer for existing
published Ollama GGUFs whose metadata or tensor layout does not yet match what
llama.cpp expects directly. The layer translates those files in memory at load
time so users do not need to re-pull or re-create models during the transition
to llama-server.

This patch model is intended to be short lived. The target end state is that
published models and newly created models use llama.cpp-compatible metadata and
tensor layouts on disk, and this directory can be removed.

The layer is applied automatically at build time via CMake `FetchContent`'s
`PATCH_COMMAND` for normal fetched builds. If CMake is pointed at a source
override through `FETCHCONTENT_SOURCE_DIR_LLAMA_CPP`, the same patch is applied
during configure. If `OLLAMA_LLAMA_CPP_SOURCE` is set, the patch is
intentionally skipped so a developer can iterate on a local llama.cpp tree.

## Files

- `llama-ollama-compat.h`, `llama-ollama-compat.cpp` - the compatibility
  entry points and per-architecture handlers.
- `llama-ollama-compat-util.h`, `llama-ollama-compat-util.cpp` - helpers for
  KV edits, tensor renames, skip-prefix tracking, tensor load operations, and
  small tensor repacking primitives.
- `001-llama-cpp-hooks.patch` - small additive call-site edits in llama.cpp files.
  It currently touches `src/llama-model-loader.cpp` and `tools/mtmd/clip.cpp`.
- `002-llama-cpp-ui-empty-assets.patch` - lets the llama.cpp UI embed helper
  generate an empty asset table when no UI assets are present.
- `compat.cmake`, `apply-patch.cmake` - CMake glue and an idempotent applier
  (used by `llama/server/CMakeLists.txt`) that applies every `*.patch` under
  this directory by numeric filename order — the hooks patch plus each
  `models/` architecture patch.
- `models/` - the sibling **new-architecture** layer: implementations of
  architectures llama.cpp doesn't support yet, each added via a small
  registration patch. (Those files *add* archs; the files above *translate*
  existing GGUFs onto archs llama.cpp already has.)

The compatibility source files stay in this directory and are linked into the
fetched llama.cpp targets. The patch file only adds call sites.

## Load-Time Hooks

The layer runs at a small set of loader hook points:

1. Main model constructor: `translate_metadata` inspects the parsed metadata
   and mutates the in-memory `gguf_context` and `ggml_context` when a handler
   recognizes an existing published model format. It can also request mmap
   disablement when a handler needs writable backend buffers for transformed
   tensor data.
2. Main model tensor indexing: `should_skip_tensor` hides embedded projector,
   vision, audio, MTP, or other tensors that the text loader should not claim.
3. Main model tensor reads: `maybe_load_text_tensor` applies registered
   text-side load operations, such as FFN concat or dtype promotion, before
   the normal llama.cpp file read. This is wired into both full model loading
   and single-tensor reads used by tools such as `llama-quantize`.
4. `mtmd/clip` constructor: `translate_clip_metadata` rewrites a clip-facing
   view of monolithic GGUFs into the mmproj form expected by llama.cpp.
5. `mtmd/clip` tensor load loop: `maybe_load_tensor` applies clip-side load
   operations, such as F16 to F32 promotion, QKV merge, tensor repack, tensor
   split, or zero-fill.

Files that do not match a supported published-model marker are left unchanged.
Setting `OLLAMA_LLAMA_CPP_COMPAT=0` disables the hook bodies for internal
create-time validation and for models that are already known to be
llama.cpp-compatible on disk.

## Supported Transformations

This table tracks the dispatch surface. Keep it brief; the handler comments in
`llama-ollama-compat.cpp` are the source of truth for exact KV and tensor maps.

| Internal arch / marker | Text handling | Clip/mmproj handling |
|---|---|---|
| `gemma3` | Normalizes Gemma 3 metadata, tokenizer fields, and embedded vision/projector tensors. | Gemma 3 projector translation. |
| `gemma3` + embedding markers (`embeddinggemma`) | Maps to `gemma-embedding` metadata and fixes embedding dense/norm tensors. | n/a |
| `bert` + Snowflake markers (`snowflake-arctic-embed2`) | Fixes Snowflake Arctic Embed 2 tokenizer metadata. | n/a |
| `gemma3n` | Normalizes tokenizer/EOS metadata, truncates vocab-shaped tensors, and hides unused embedded vision/audio/projector tensors. | n/a |
| `gemma4` | Normalizes tokenizer metadata and hides embedded audio/vision/projector tensors from the text loader. | Gemma 4 vision/audio projector translation for GGUF blobs. |
| `gptoss` | Maps to `gpt-oss`, copies KVs, injects missing expert FFN metadata, and renames tensors. | n/a |
| `lfm2` | Renames norm tensors and fixes feed-forward metadata. | n/a |
| `olmo3` | Maps to the OLMo2-compatible loader path. | n/a |
| `mistral3` | Fixes RoPE/YaRN metadata and hides embedded vision/projector tensors. | Pixtral-style projector translation. |
| `qwen35`, `qwen35moe` | Fixes Qwen3.5/Qwen3-VL-style text metadata, translates embedded MTP tensors, and hides embedded vision/projector tensors. | Qwen3-VL merger-style projector translation. |
| `qwen3next` | Normalizes hybrid attention KV-head metadata and renames SSM dt tensors to the names expected by llama.cpp. | n/a |
| `qwen25vl` | Maps to `qwen2vl` metadata conventions. | Qwen2.5-VL projector translation. |
| `qwen3vl`, `qwen3vlmoe` | Adds missing Qwen3-VL metadata and hides embedded vision/projector tensors. | Qwen3-VL projector translation, including QKV merge and patch-embedding split/repack. |
| `deepseekocr` | Maps to `deepseek2-ocr`, injects missing OCR/MoE metadata, and hides embedded SAM/vision/projector tensors. | DeepSeek OCR projector translation. |
| `glmocr` | Maps GLM OCR metadata/tensors to the llama.cpp-compatible view. | GLM OCR projector translation. |
| `glm4moelite` | Maps GLM-4.7 Flash MLA metadata to the `deepseek2` path and fixes special-token metadata. | n/a |
| `nemotron_h_moe` | Fixes latent-FFN variants and hides MTP tensors. | n/a |
| `nemotron_h_omni` | Selects the Nemotron text loader and hides audio/vision/projector tensors from the text loader. | Nemotron V2 VL projector translation; audio remains disabled. |
| `llama` with Llama 3 markers | Fixes Llama 3 tokenizer metadata. | n/a |
| `llama4` | Hides embedded vision/projector tensors from the text loader. | Llama 4 projector translation. |
| `clip` projector without `clip.projector_type` | n/a | Defaults LLaVA/BakLLaVA projectors to `clip.projector_type=mlp`. |

Usage:

```sh
llama-server --model /path/to/ollama-blob --mmproj /path/to/ollama-blob
```

Passing the same monolithic GGUF as both `--model` and `--mmproj` works because
each loader applies its own translation.

Additional architectures are added by implementing a `handle_<arch>()` and,
for vision models, `handle_<arch>_clip()` in `llama-ollama-compat.cpp`, then
dispatching them from `translate_metadata` / `translate_clip_metadata`. For
monolithic vision models, also update the `compatClipArches` allowlist in
`llm/llama_server.go` so Ollama passes the main GGUF as `--mmproj`.

## Regenerating the Patch File

After a llama.cpp bump moves the insertion points, re-apply the edits to a
fresh checkout and run:

```sh
cd /path/to/llama.cpp
git diff -- \
    src/llama-model-loader.cpp \
    tools/mtmd/clip.cpp \
    > /path/to/ollama/llama/compat/001-llama-cpp-hooks.patch
```

## Implementation Notes

The compatibility code is mostly written against public APIs (`gguf.h`,
`ggml.h`, `ggml-backend.h`). A few operations rely on implementation details
because the public API does not expose equivalent mutators:

| Dependency | Use | Replacement if needed |
|---|---|---|
| Direct writes to `ggml_tensor::type` / `ne[]` / `nb[]` | Post-creation tensor reshape/retype for in-memory translation. | Add public tensor shape/type mutators. |
| `const_cast<char *>(gguf_get_tensor_name(...))` in `rename_tensor` | Renames gguf tensors in place. | Add a public `gguf_rename_tensor` helper. |
| `llama_model_loader` forward declaration from `src/llama-model-loader.h` | Opaque key for per-loader registries. The pointer is never dereferenced. | Replace registry keys with `const void *`. |

Two helpers need extra context:

- `reclaim_slot_as` repurposes an orphaned tensor slot as a synthesized tensor
  when a clip handler splits one source tensor into multiple destination
  tensors. This is needed because clip metadata loading allocates exactly enough
  tensor slots for the source file.
- Load-op registry overrides ignore the caller-provided `file_offset` when a
  registered operation exists. The operations capture their own source offsets
  at translation time, before renames change tensor names.
