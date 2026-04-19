# llama.cpp compatibility shim

This directory holds an in-process compatibility layer that lets upstream
`llama-server` load GGUFs produced by older versions of Ollama (and files
pulled from the Ollama registry) without re-converting or re-downloading.

The layer is applied automatically at build time via CMake `FetchContent`'s
`PATCH_COMMAND` — there is no separate "apply patches" step.

## Files

- `llama-ollama-compat.h`, `llama-ollama-compat.cpp` — the shim itself. These
  are regular source files owned by Ollama; they get copied into the fetched
  llama.cpp source tree during configure.
- `upstream-edits.patch` — small additive edits to upstream files so the
  shim gets called. Currently ~48 lines touching 6 files. Kept as a real
  `git` patch so re-generation on upstream bumps is one command.

## What the shim does

The shim runs at two well-defined points in the loader:

1. **After `gguf_init_from_file`**, for both the main model loader and the
   `mtmd/clip` loader: inspects the just-parsed metadata and decides whether
   the file is an Ollama-format GGUF. If so, it mutates the in-memory
   `gguf_context` and `ggml_context` (KV names, tensor names, tensor types)
   so the rest of the loader sees an upstream-shape file.

2. **After `load_all_data`**: applies any numerical fix-ups that need the
   tensors in their final backend buffers (e.g. RMSNorm `+1` if a future
   arch needs it — gemma3 doesn't).

Non-Ollama files are detected by the absence of Ollama-specific KV keys
(e.g. `gemma3.mm.tokens_per_image`) or embedded `v.*` / `mm.*` tensors in
the main model file. When no markers are present every compat function is
an immediate no-op.

## Currently supported architectures

| Arch | Text loader | Clip (mmproj) loader |
|---|---|---|
| `gemma3` | KV injection (`layer_norm_rms_epsilon`, `rope.freq_base`, `rope.freq_base_swa`), tokenizer vocab truncation, drop `v.*`/`mm.*` tensors | Arch rewrite to `clip`, KV synthesis (`clip.vision.*`, `clip.projector_type=gemma3`), tensor renames (`v.patch_embedding`→`v.patch_embd`, `mlp.fc{1,2}`→`ffn_{down,up}`, etc.), F16→F32 promotion for patch/position embeddings (Metal IM2COL requirement) |

Usage:

```
llama-server --model /path/to/ollama-blob --mmproj /path/to/ollama-blob
```

Passing the same monolithic GGUF as both `--model` and `--mmproj` works —
each loader applies its own translation.

Additional architectures are added by implementing a `handle_<arch>()`
and (for vision models) `handle_<arch>_clip()` in `llama-ollama-compat.cpp`
and dispatching them from `translate_metadata` / `translate_clip_metadata`.

## Regenerating `upstream-edits.patch`

After upstream changes the insertion points (rare), re-apply the edits to
a fresh checkout and run:

```
cd /path/to/llama.cpp
git diff -- \
    ggml/include/gguf.h \
    ggml/src/gguf.cpp \
    src/CMakeLists.txt \
    src/llama-model-loader.cpp \
    src/llama-model.cpp \
    tools/mtmd/clip.cpp \
    > /path/to/ollama/llama/compat/upstream-edits.patch
```

## Why not fork llama.cpp or vendor it?

Forking means tracking upstream manually. Vendoring means snapshotting all of
llama.cpp's source in the Ollama tree (the old `llama/llama.cpp/` layout).
This shim keeps upstream unmodified on disk and the Ollama-specific logic
isolated in two files plus a small diff — upstream bumps are usually just
`LLAMA_CPP_VERSION` changes.
