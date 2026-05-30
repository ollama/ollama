# Interim model-architecture support

Ollama runs GGUF models with upstream llama.cpp, pinned at `LLAMA_CPP_VERSION`.
Occasionally a model arrives whose architecture isn't in that pinned version
yet. To keep that model working in the meantime, we carry a small, temporary
addition here that teaches the fetched llama.cpp about it, following llama.cpp's
own model conventions so the same work can be offered upstream.

This is deliberately interim. As soon as the architecture is available in
llama.cpp, the files below are deleted and the model loads on stock llama.cpp.

> Its counterpart, `llama/compat/`, handles the opposite case: models llama.cpp
> *already* supports, whose older GGUF files just need their metadata translated.

## What's here, per architecture

- `<arch>.cpp` — the model implementation (hparams, tensors, compute graph). It
  lives in our tree and is compiled into llama.cpp via CMake, so a llama.cpp
  version bump leaves it untouched.
- `llama-cpp-<arch>.patch` — the few edits that register the architecture in
  llama.cpp's own tables: the arch name, the model factory, the rope type, and —
  only if the model needs them — a tensor name or tokenizer entry.

The build applies the patch and links the source automatically.

## Adding one

Work against a llama.cpp checkout at the pinned `LLAMA_CPP_VERSION`:

1. Implement the architecture there as an ordinary llama.cpp model (a
   `src/models/<arch>.cpp` plus the registration edits), modeled on the closest
   existing architecture and reusing its building blocks. Iterate with
   `llama-cli` until it loads and generates correctly.
2. Move the implementation to `llama/models/<arch>.cpp`, changing its include
   from `"models.h"` to `"models/models.h"`.
3. Capture the registration edits as the patch — only the files you changed:
   ```sh
   git diff -- src/llama-arch.h src/llama-arch.cpp src/llama-model.cpp \
       src/models/models.h src/llama-vocab.h src/llama-vocab.cpp \
     > llama/models/llama-cpp-<arch>.patch
   ```
4. Build from the repo root (`cmake -B build . && cmake --build build`) and
   confirm `ollama run <model>` works and stops cleanly.

Keep the footprint small: put the logic in `<arch>.cpp`, and prefer reusing
llama.cpp's existing hparams, tensor names, and graph builders over adding new
ones. The smaller the patch, the less there is to redo on a version bump.

## After a llama.cpp bump

Re-apply the registration edits to the new checkout and re-capture the diff; the
`<arch>.cpp` usually needs no change. If the architecture has landed upstream by
then, simply delete `<arch>.cpp` and its patch.

## Current architectures

- `laguna` — poolside Laguna. Remove once upstream llama.cpp supports it.
