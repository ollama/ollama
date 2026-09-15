# Architecture notes (NOLAI fork)

Working reference for how this repo is structured, built up while
investigating video-input support. Not official docs (see `docs/` for
those) and not a plan — a durable map of how the pieces fit together, so a
future session (human or Claude) doesn't have to re-derive it from scratch.
Update this file as understanding improves; it will drift out of date
otherwise.

## The two inference backends

Ollama serves models through exactly two backends, chosen per-model based on
the manifest's layer format (`Model.IsMLX()`, `server/images.go`: true when
`Config.ModelFormat == "safetensors"`).

### 1. GGML / GGUF models → vendored `llama-server` subprocess

This is the backend for the vast majority of models (Gemma, Qwen*-VL,
Llama, Mistral, etc. in their GGUF form). There is **no Go-native CGO model
implementation anymore** — it was removed in commit `9db4bdbad6`
("runner: Remove CGO engines, use llama-server exclusively for GGML
models", #16031). Before that commit, Ollama had its own Go model/runner
code under `model/models/*` and `runner/ollamarunner/`; those packages no
longer exist. Any PR or reference material written against that
architecture (e.g. abandoned upstream PR ollama/ollama#12962) targets a
subsystem that is gone and cannot be merged as-is.

Today, GGUF inference works by Ollama spawning a real `llama-server`
subprocess (llama.cpp's own HTTP server binary) and talking to it over
HTTP — `/completion` and `/v1/chat/completions`. Ollama's job is: convert
models to GGUF (`convert/`), manage the manifest/capabilities
(`server/images.go`), launch and supervise the subprocess with the right
flags (`llm/llama_server.go`, `server/sched.go`), and translate between
Ollama's own API shapes and llama-server's wire format.

### 2. Safetensors models → MLX runner (Apple Silicon / CUDA via MLX)

A second, independent, Go-native inference path for models kept in
safetensors format, under `x/models/*` (model implementations) and
`x/mlxrunner/` (the runner/server harness, analogous in role to
`llm/llama_server.go` but talking to Ollama's own in-process MLX-backed Go
code rather than a subprocess). This is a real, actively developed path —
it's not legacy or secondary. Some model families exist on both backends
(e.g. Gemma4 has both a GGUF converter and an `x/models/gemma4/` MLX
implementation); others are MLX-only (e.g. Qwen3.8/`qwen4_exp` — no GGUF
converter exists for it at all, and its architecture has components with no
equivalent in llama.cpp — see "Model-specific findings" below).

**When investigating "how does X work for model Y", always check which
backend Y actually uses first** — the two paths share some upstream
plumbing (`api.Message`, `llm.MediaData`/`MediaKind`, `server/prompt.go`'s
`[img-N]` tagging) but diverge completely below that point. Mixing up which
backend a model uses is the single easiest way to investigate the wrong
code path.

## The vendoring mechanism (`llama/`)

This is a common point of confusion, worth getting right:

- `llama/vendor/` is **not committed source**. It's the on-disk name CMake
  uses as a fetch destination; there's no git history for it because
  nothing under it is tracked (see `llama/.gitignore`). If you `grep` under
  `llama/vendor/` expecting to find llama.cpp source, **it will only be
  there if someone has already run a configure step in this working
  directory** — and if it's there, it reflects whatever was fetched *at
  that time*, which can silently be stale relative to the current
  `LLAMA_CPP_VERSION` file if nobody has reconfigured since the pin last
  changed. Always check `LLAMA_CPP_VERSION`'s current value, and when in
  doubt, do a fresh configure rather than trust an existing checkout.
- The actual fetch: `LLAMA_CPP_VERSION` (single line, repo root) is read by
  `llama/server/CMakeLists.txt` and passed as `GIT_TAG` to a CMake
  `FetchContent_Declare(llama_cpp GIT_REPOSITORY
  https://github.com/ggml-org/llama.cpp.git GIT_TAG ...)`. Running
  `cmake -S llama/server --preset cpu` (or another backend preset) clones
  the pinned ref into `build/<preset-name>/_deps/llama_cpp-src`.
  Configuring is a real network operation (full clone, though shallow) —
  budget a few minutes.
- **Ollama patches llama.cpp before building it**, via `llama/compat/` —
  see `llama/compat/compat.cmake` and `llama/compat/llama-ollama-compat.cpp`
  (a large C++ shim, not just a patch file) plus `.patch` files. This
  handles things like: translating Ollama's own GGUF tensor-naming
  convention into what llama.cpp's loader expects for models where Ollama's
  converter and llama.cpp's expected format diverge (`detect_ollama_gemma4
  ()`/`handle_gemma4_clip()` are examples — real, production code, not
  stubs), RPC changes, DRM-based VRAM detection, etc.
- **Ollama also overrides several of llama.cpp's own CMake options**,
  forcing some on and some off regardless of llama.cpp's own defaults — see
  `llama/server/CMakeLists.txt:164-185`. This is where the video-support
  investigation hit a real wall (see below): llama.cpp's own `LLAMA_SUBPROCESS`
  option defaults to ON on Linux/macOS, but Ollama forces it **OFF** with an
  explicit comment explaining a real portability bug (glibc symbol
  `posix_spawn_file_actions_addchdir_np` missing before glibc 2.29, breaking
  Ollama's AlmaLinux 8 (glibc 2.28) build images) — and a second override,
  `MTMD_VIDEO OFF`, exists specifically *because* mtmd's video code depends
  on the same subprocess header. **This means bumping `LLAMA_CPP_VERSION`
  alone never enables llama.cpp's native video support in Ollama's build,
  no matter how new the pinned version is**, until this override is
  revisited.
- `llama/README.md` documents the supported workflow for bumping the pin:
  configure, verify the fetched ref, diff the upstream changes, rebuild,
  run Go tests, and run integration tests on real hardware. It's a real,
  moderately involved process (model loading, GPU discovery, scheduler
  inputs, and streaming behavior can all be affected by a version bump),
  not a one-line edit — treat it accordingly.

## The multimodal pipeline (images/audio today, video investigated)

For the GGML/llama-server path specifically (MLX has its own parallel but
structurally similar pipeline under `x/mlxrunner/media.go` +
per-model `PrepareMedia`/`EncodeMedia`):

1. `api.Message.Images []ImageData` (`api/types.go`) is a **generic
   media-bytes carrier**, not image-specific despite the name — audio
   already piggybacks on it (sniffed by content, not a separate field).
2. `server/prompt.go`'s `imageTaggedMessages()` is the single conversion
   point from "raw bytes on a message" to "`[img-N]` markers in the
   rendered prompt + a parallel `[]llm.MediaData` slice." Both
   chat-prompt-building code paths in `server/routes.go` converge here.
3. `llm/media.go`'s `DetectMediaKind()` sniffs each blob's kind (currently
   `MediaKindImage`/`MediaKindAudio`/`MediaKindUnknown` — no video kind
   yet) via magic bytes / `http.DetectContentType`.
4. `llm/llama_server.go` sends the tagged prompt + media to the
   `llama-server` subprocess, either via `/completion` (Ollama's own
   `[img-N]` markers get swapped for llama-server's internal marker, media
   sent as base64 in a `MultimodalData` array) or `/v1/chat/completions`
   (OpenAI-style `image_url`/`input_audio` content parts).
5. llama-server's mtmd library (`llama/vendor/tools/mtmd/` once fetched)
   does the actual vision/audio encoding — CLIP-style vision towers per
   architecture (e.g. `models/gemma4v.cpp` for Gemma4, with its own
   `PROJECTOR_TYPE_GEMMA4V`), auto-wired via `--mmproj` when
   `llm/llama_server.go`'s `compatClipArches` allowlist includes the
   model's architecture.

Capability advertisement (`model.CapabilityVision`/`CapabilityAudio`) is
purely derived from GGUF metadata presence (`vision.block_count` /
`audio.block_count` KV keys, `server/images.go` `ggufCapabilities()`) — not
manually maintained per model family.

### Video input — investigated, not yet implemented

Findings from investigating how to add this, current as of this session:

- **Video is unimplemented anywhere in Ollama's own Go code** — no
  `MediaKindVideo`, no video-aware renderer logic (though
  `model/renderers/qwen35.go`/`qwen3vl.go` have literal `// TODO: support
  videos` markers at the point images are tagged).
- **llama.cpp itself gained real native video support upstream**: PR
  ggml-org/llama.cpp#24269 ("xsn/mtmd-helper-video-input") merged June 8,
  2026 (commit `8f83d6c271d1...`) into `master`, and is present in the
  `v0.4.1` release tag (verified by actually fetching it locally). It adds:
  - `mtmd_helper_video_init*`/`mtmd_bitmap_init_lazy` — mtmd decodes video
    itself (via ffmpeg as a subprocess, not a linked codec library) and
    lazily expands one input marker into multiple frame chunks during
    tokenization. Frames still go through the ordinary per-image bitmap/CLIP
    path — no new temporal position-encoding was added to any vision
    transformer graph (`clip-graph.h:48` still has `// TODO [QWEN_VIDEO]:
    improve this in the future`).
  - Server-side: a real `"input_video"` OpenAI-compat content-part type
    (`tools/server/server-common.cpp`), gated by `opt.allow_video`, with
    per-model capability exposed as `"video": meta.has_inp_video`.
  - CLI: `mtmd-cli.cpp` has a `/video <path>` command.
- **But it's compiled out in Ollama's build**, specifically and
  deliberately, via the `LLAMA_SUBPROCESS`/`MTMD_VIDEO` CMake overrides
  described above (`llama/server/CMakeLists.txt:179-185`). Verified locally:
  bumping `LLAMA_CPP_VERSION` to `v0.4.1` and reconfiguring still shows
  `MTMD_VIDEO:BOOL=OFF` in `CMakeCache.txt`, because `LLAMA_SUBPROCESS` is
  force-off regardless of the pinned version.
- **Open question, not yet resolved**: how to get real video support while
  keeping the AlmaLinux 8 (glibc 2.28) build target working. Candidate
  directions, none evaluated in depth yet:
  1. Check whether a newer llama.cpp version fixed the missing glibc
     version-guard around `posix_spawn_file_actions_addchdir_np` in
     `subprocess.h` — if so, `LLAMA_SUBPROCESS` might be safely
     re-enabled without breaking the AlmaLinux 8 build. Needs checking
     against the actual current upstream source, not assumed.
  2. Whether AlmaLinux 8 support is still a hard requirement, or could be
     relaxed for a build variant — a product/release decision, not a
     technical one.
  3. Whether `MTMD_VIDEO` could be decoupled from generic
     `LLAMA_SUBPROCESS`/router-mode machinery via an `llama/compat/` patch,
     e.g. making mtmd's video helper spawn ffmpeg through a narrower path
     that doesn't trip the glibc issue, rather than the general-purpose
     subprocess header. Not scoped.
  4. Fallback: decompose video into frames on **Ollama's Go side** instead
     (before the request reaches llama-server), reusing the existing
     per-image `[img-N]` pipeline unchanged, extracting frames via a
     runtime `exec.LookPath("ffmpeg")` shell-out (no CGO, no vendored
     ffmpeg build). This avoids the glibc/subprocess issue entirely since
     it never touches llama.cpp's C++ subprocess code, at the cost of
     duplicating (in Go) something llama.cpp now does natively in C++. A
     full design for this was drafted in an earlier planning pass this
     session; treat it as a known-viable fallback, not the preferred
     direction, now that native upstream support is confirmed to exist.

## Model-specific findings (from this investigation)

- **Gemma4**: has a complete, working GGUF vision pipeline already (see
  pipeline section above) — no new C++ work needed to serve Gemma4 images
  today. Good target for follow-on multimodal work generally.
- **Qwen3.8 / `qwen4_exp`**: MLX-only, no GGUF converter exists. Its
  architecture has three components with no equivalent anywhere in
  llama.cpp: "hyper-connections" (a learned multi-stream residual mixing
  mechanism, replacing the usual single residual stream — would require
  pervasive changes to llama.cpp's graph-construction scaffolding, not a
  single new op), an "Engram" hashed n-gram embedding cache (token-ID
  XOR/hash → sharded embedding table, persisted across forward calls — no
  analog to any existing llama.cpp cache type), and "QSA" sparse attention
  (structurally similar to DeepSeek's sparse-attention indexer, which
  llama.cpp has tensor-loading plumbing for but never wired into an actual
  compute graph — `glm-dsa.cpp` inherits `deepseek2.cpp`'s dense-only graph
  builder). Porting this would be genuine multi-week new C++
  graph-construction work in llama.cpp, not a converter-writing exercise.
  Ruled out as a target for this reason.
- **Ornith 9B**: no dedicated model code at all — purely a renderer/parser
  wrapper (`model/renderers/ornith.go`) around Qwen3.5's existing renderer
  and parser. Rides on whatever backend Qwen3.5 uses (both GGUF and MLX
  exist for that family).

## Session state left behind

- `LLAMA_CPP_VERSION` was changed from `b10740` to `v0.4.1` in the working
  tree (uncommitted) during this investigation, to verify the video API's
  presence. This is a plausible independent improvement (newer llama.cpp)
  but was **not requested or confirmed as a standalone change** — don't
  assume it should be committed without checking with the user and running
  the full `llama/README.md` validation workflow first (a version bump can
  affect model loading, GPU discovery, scheduler inputs, and streaming
  behavior well beyond video).
- A `build/llama-server-cpu/` directory exists locally from a configure
  step done during this investigation (gitignored, not committed, safe to
  delete/reconfigure).
