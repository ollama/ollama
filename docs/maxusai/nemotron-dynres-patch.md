# Nemotron dynamic-resolution patch: lifting the fixed 256-token vision budget

MaxusAI-fork reference (fork-only; does not exist upstream). Written 2026-08-01.
Status: **implemented on `feat/nemotron-dynres-vision-budget`, build-verified, not yet
runtime-validated** — see [Validation checklist](#validation-checklist) before deploying.

> **The one thing to take away:** Nemotron 3 Nano Omni does **not** tile. The tiling patch
> sketched in [vision-token-budgets-by-arch.md](vision-token-budgets-by-arch.md) targeted
> the wrong mechanism — the model's reference preprocessing is a *single* native-aspect
> dynamic-resolution resize, bounded by 1,024…13,312 pre-merge patches (= **256…3,328
> visual tokens** per image). `llama/compat/002-llama-cpp-nemotron-dynres.patch` implements
> exactly that, and makes `--image-{min,max}-tokens` live for `nemotron_h_omni`.

## What the reference actually does (and why the tiling plan was wrong)

Verified 2026-08-01 against NVIDIA's HF repo
(`nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16`: `image_processing.py`,
`processing.py`, `config.json`), vLLM's `nano_nemotron_vl.py`, and the tech report
(arXiv 2604.24954):

- **Single resize, no tiles, no thumbnail.** Each image is bicubic-antialias resized —
  aspect preserved, **never letterboxed** — to a patch grid within
  `min_num_patches=1024` … `max_num_patches=13312` (16 px patches), then 2×2
  pixel-shuffled: 256…3,328 tokens. Small images are **upscaled** to the floor. vLLM
  asserts `use_thumbnail is False` on this path; the `use_thumbnail: true` in config.json
  is a legacy key the V3 dynamic path never reads.
- **The 12-vs-13 question dissolves.** `max_num_patches=13312` is a per-image patch cap,
  not a tile count. The GGUF's `max_tiles=12` is fabricated by a converter default
  (`convert/convert_nemotron_h.go` `cmp.Or(MaxNumTiles, 12)`); the V3 preprocessor config
  has no such key. Tiling (12 tiles + thumbnail on top) is the older
  Nemotron-Nano-**V2-VL-12B** scheme, a different model.
- **Markers.** Every image — including a single small one — is wrapped
  `<img>`(id 19) + `<image>`(id 18)×N + `</img>`(id 20). Stock llama.cpp emits **no**
  markers for nemotron (that is why measured counts were exactly 256, not 256+2), so
  stock behavior was not even the correct degenerate case: wrong framing *and* black
  letterbox bars.
- **Graph change is required** (the by-arch doc's "graph side needs no change" held only
  for the abandoned tiling design): the GGUF's position embeddings are pre-downsampled to
  the fixed 32×32 (512 px) grid, so variable-resolution input needs runtime interpolation.

Two corrections to the by-arch doc's patch notes, for the record: feeding the patch counts
(1024/13312) into the tile-count keys would **pass** the `GGML_ASSERT` at the INTERNVL
hparams branch (1024 ≤ 13312 < INT32_MAX) — the actual failure is an *empty*
`image_res_candidates` aborting in `mtmd_image_preprocessor_internvl::preprocess`. And the
"internvl repoint" dead end is confirmed (missing `mm_{0,1,3}_b` bias tensors abort load).

## Provenance

Upstream already solved this once: llama.cpp **PR #23638** (author SyrupAnon), closed
unmerged 2026-05-25 on process grounds (AI-disclosure/reference-link dispute), not
technical ones. Its final commit `66b9b344` ("use native dynamic resolution for
nemotron_v2_vl") is the basis of our 002 patch. Tracking issue llama.cpp **#25317** (open,
uncommented since 2026-07-05) independently reproduces the fork's measurements. Upstream
master, tag `b10091` (our pin), and upstream ollama main were all still fixed-256 as of
2026-08-01 — waiting for upstream is not a near-term path. Resubmitting the corrected
patch upstream (with the HF reference links the original PR lacked) is worth considering
so the fork does not carry 002 forever.

## What the 002 patch changes

`llama/compat/002-llama-cpp-nemotron-dynres.patch`, generated against llama.cpp `b10091`
(the `LLAMA_CPP_VERSION` pin), applied automatically by `apply-patch.cmake`'s glob after
001. Three hunks:

1. **`tools/mtmd/clip.cpp`** — the `PROJECTOR_TYPE_NEMOTRON_V2_VL` hparams branch calls
   `hparams.set_limit_image_tokens(256, 3328)`. This differs deliberately from upstream
   commit `66b9b344`, which assigned raw `image_{min,max}_pixels` (262144/3407872 — the
   same budget: tokens × patch_size²·n_merge² = tokens × 1024). Going through
   `set_limit_image_tokens()` honors `custom_image_{min,max}_tokens`, i.e. the
   `--image-{min,max}-tokens` flags become **live** for nemotron, matching how gemma4 and
   the qwen family are wired. It also sets `warmup_image_size` to the ceiling (~1846 px),
   so warmup exercises the worst case at load — fail-fast for memory, but expect a heavier
   model load than the old 512² warmup.
2. **`tools/mtmd/mtmd.cpp`** — dispatch to `mtmd_image_preprocessor_dyn_size` (replacing
   `fixed_size`), and set `img_beg="<img>"` / `img_end="</img>"`. Each image now costs
   its grid product + 2 marker tokens, like the other arches.
3. **`tools/mtmd/models/nemotron-v2-vl.cpp`** — `resize_position_embeddings(GGML_SCALE_MODE_BICUBIC)`
   instead of the raw `ggml_add` of the baked 32×32 grid. A no-op at 512×512.

The Ollama-format GGUF path needs **no compat-layer change**: `handle_nemotron_h_omni_clip()`
already produces the same baked 32×32 position-embedding tensor as upstream conversion, so
patched behavior is identical for both GGUF formats. (Fidelity note: NVIDIA's C-RADIOv4-H
interpolates from the full 128×128 grid at inference, bilinear `align_corners=False`.
Keeping the 128×128 grid through the compat layer and interpolating from it would be
*more* faithful than upstream's own patch — a follow-up, only worth doing with an A/B.)

## Go side

`visionServerArgs()` ([`llm/llama_server.go`](../../llm/llama_server.go)) now has a
`nemotron_h_omni` case passing `--image-min-tokens` / `--image-max-tokens`, resolved by
`nemotronImageTokenBudget()`:

- Defaults **256 / 3328** (the model's native bounds).
- The shared `ImageMinTokens`/`ImageMaxTokens` options arrive as the gemma4-shaped
  DefaultOptions values (40/1120) when the caller left them alone; those exact values are
  treated as unset for this arch. Consequence: **explicitly requesting 40 or 1120 on
  nemotron is not expressible** — pick an adjacent value (41, 1119, 1121…). Any other
  value passes through; min clamps down to max.
- Values above 3328 are passed through, but exceed the model's training distribution —
  the reference never produces more than 13,312 patches. Don't raise the ceiling; only
  lower it (e.g. to trade detail for context on busy hosts, via a Modelfile
  `PARAMETER image_max_tokens 2048`, same manifest-pinning advice as
  [vision-token-budget-measurements.md](vision-token-budget-measurements.md)).
- **On an unpatched payload the flags are parsed but inert** — same silent-no-op behavior
  the measurements doc documents for `image_max_tokens` against stock gemma4. The Go
  change is therefore safe to ship via the overlay image (where it does nothing for
  nemotron); the *budget lift itself* is C++ and is not.

`TestVisionServerArgs` was updated accordingly — the old `nemotron_h_omni → nil` case and
its "must stay absent" rationale are gone.

## Deployment constraints

- **Cannot ship via the overlay image.** `Dockerfile.gemma4budget` copies the C++ payload
  from the `ollama/ollama` base; the 002 patch only takes effect in a full CMake build.
  Adding the patch file also makes the overlay's payload-pristine proof
  (`git diff HEAD v0.32.5 -- LLAMA_CPP_VERSION llama/ …`) non-empty — a **true positive**:
  do not build overlays from a tree containing 002.
- **The gfx1151 host is gated at 0.32.1** (payload = llama.cpp `b9888`) per
  [amd-upgrade-gate.md](amd-upgrade-gate.md). The 002 hunks were generated against
  `b10091`; the three touched regions are identical at `b9888`, but `mtmd-image.cpp`
  differs between the tags, so **re-run `git apply --check` against the exact payload tag
  you build** (for a 0.32.1-era build, apply onto `b9888` and regenerate if fuzz).
- Expect per-image context cost up to 3,328 tokens (plus 2 markers). Ollama's Go-side
  truncation heuristics count images as **zero** tokens for this arch (no projector
  layer ⇒ `ProjectorPaths` empty ⇒ the 768/image heuristic never applies), so budget
  `num_ctx` for image-heavy workloads explicitly.

## Validation checklist

Build-time (done on this branch): `002` applies after `001` on pristine `b10091`; reverse
`--check` idempotence holds; `llama-server` CPU target compiles with both patches;
`go build` / `go test ./llm` green.

Runtime (required before the deploy gate, on the ROCm host, per the A/B discipline in
[gemma4-budget-image.md](gemma4-budget-image.md)):

1. **Token counts** via `prompt_eval_count` minus the 18-token text baseline: expect
   ≈ grid product + 2, e.g. 640×480 → 40×30/4 + 2 = **302**, 1920×1080 → ≈ 2042,
   1568×1568 → ≈ 2404, 3000×2000 → ceiling ≈ 3330. (The PR author measured 2040 visual
   tokens at 1920×1080 on CUDA.) Constant-256 means the payload is unpatched.
2. **`ggml_interpolate` bicubic on ROCm/gfx1151** — exercised at every non-512² grid via
   `resize_position_embeddings`; watch for backend fallback or garbage output.
3. **Load-time warmup** at ~1846²: VRAM headroom and load latency on the 8060S.
4. **Output quality A/B** (≥6 rows, 0 degenerate): OCR/fine-detail should improve
   markedly; also verify no double `<img>` wrapping (check the served nemotron3 template
   does not already emit the markers — mtmd now adds them).
5. **Batch splitting**: a 3,328-embedding image chunk vs `-b` 256–2048 — confirm the mtmd
   helper chunks it (llama.cpp behavior, unverified in code review).
6. **Audio unaffected**: compat force-sets `clip.has_audio_encoder=false`; smoke-test an
   audio prompt anyway if audio is ever enabled.

## See also

- [vision-token-budgets-by-arch.md](vision-token-budgets-by-arch.md) — the arch-gating
  mechanism; its "Notes for a future tiling patch" section is superseded by this doc.
- [vision-token-budget-measurements.md](vision-token-budget-measurements.md) — measured
  costs (pre-patch for nemotron) and the routing policy.
- llama.cpp PR #23638 / commit `66b9b344` / issue #25317 — upstream provenance.
