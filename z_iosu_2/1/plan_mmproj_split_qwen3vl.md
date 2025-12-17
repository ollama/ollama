# Plan: `mmproj.gguf` split support (Qwen3-VL)

Note: From now on, I will write documents in English.

## Context (what the GitHub comment is asking for)
The comment points out that on HuggingFace, most “Image-Text-to-Text” models compatible with Ollama are published as:

- a **text GGUF** (main model)
- a **separate vision/projector GGUF** (typically named `mmproj.gguf`)

It also suggests that if we want to support split vision projectors, we should look at how `llama.cpp/tools/mtmd/clip.cpp` reads `mmproj.gguf`, and compare tensor names/metadata between community `mmproj.gguf` files and Ollama-shipped blobs.

## What I will do (summary)
I will make Ollama able to **load community `mmproj.gguf`** (with slightly different naming conventions) without forcing an “all-in-one” repack. The approach:

1. **Accept and map alternative tensor names** (aliasing) so the Go model code can find weights even if the `mmproj.gguf` uses different names.
2. **Ensure the Go model structs are correctly re-bound** to tensors after the secondary GGUF is loaded (`LoadSecondary`).
3. **Detect split projectors via metadata** (e.g. `general.type = mmproj` + `general.architecture = clip` and/or `clip.projector_type`) in addition to the current detection.

I will keep inference logic changes minimal; the goal is name/metadata compatibility.

## Why this is needed
In `llama.cpp/tools/mtmd/clip.cpp`, the Qwen3VL projector wiring uses LLaVA-style projector tensor naming and, for Qwen3VL specifically, it pulls two layers:

- `mm_0_*` from `TN_LLAVA_PROJ` index 0
- `mm_1_*` from `TN_LLAVA_PROJ` index 2

On the Go side, Qwen3VL expects tensors like `mm.0` and `mm.2` (and other prefixes like `v.patch_embd.*`). Community models may instead provide:

- `mm.1` instead of `mm.2` (or other variants)
- metadata keys under `clip.*` instead of `vision.*` (or vice versa)

If the runner loads a secondary file but the expected names are not found, the model ends up “without a projector” and image input fails.

## Planned changes (files and approach)
### 1) Tensor aliasing for the projector
Goal: tolerate different tensor naming in `mmproj.gguf`.

- Add extra aliases in the Qwen3VL split path *before* `LoadSecondary`, for example:
  - Map `mm.2` ↔ `mm.1` (only if it matches the actual `mmproj.gguf` we want to support)
  - Keep existing aliases (`v.patch_embd` ↔ `v.patch_embed`, etc.)

Candidate file:
- `model/models/qwen3vl/model.go` (inside `ensureVisionReady`, before `LoadSecondary`)

### 2) More robust projector detection via metadata
Today, “projector” classification is primarily:
- `f.KV().Kind() == "projector"` or `vision.block_count` without `block_count`

But many `mmproj.gguf` files use:
- `general.type = mmproj`
- `general.architecture = clip`

Adjustment:
- treat `general.type == "mmproj"` as a projector layer as well when setting the layer media type.

Candidate file:
- `server/create.go` (layer `mediatype` detection)

### 3) Validation / tests
- Run focused Go tests (by package) to ensure we don’t break current behavior.

Candidates:
- `go test ./server -run Test...` (there are tests around projector info)
- `go test ./...` if the change remains small

## Risks / non-goals
- I won’t add aggressive heuristics that mix “similar-looking” tensors; only clear, intentional aliases.
- I won’t change the format of models shipped by Ollama.
- I won’t touch `llama.cpp` unless it becomes strictly necessary (preference: solve in Go).

## Checkpoint (your OK)
If you agree with this approach, the next step is to apply minimal changes in:

- `model/models/qwen3vl/model.go`
- `server/create.go`

…and then run tests.
