# ADR 0001: Lift the nemotron 256-token vision cap with native dynamic resolution, not tiling

- **Status:** accepted — implemented on `feat/nemotron-dynres-vision-budget`; runtime
  validation tracked in [nemotron-test-image.md](../nemotron-test-image.md)
- **Date:** 2026-08-01
- **Deciders:** MaxusAI fork maintainers
- **Supersedes:** the "Notes for a future tiling patch" plan in
  [vision-token-budgets-by-arch.md](../vision-token-budgets-by-arch.md) and decision 3 of
  [vision-token-budget-measurements.md](../vision-token-budget-measurements.md)

## Context

`nemotron3:33b` (arch `nemotron_h_omni`, llama.cpp projector
`PROJECTOR_TYPE_NEMOTRON_V2_VL`) encodes every image at a structural 256 visual tokens:
llama.cpp letterboxes onto one 512×512 canvas and its hparams branch never consumes the
`--image-{min,max}-tokens` flags. At 3000×2000 that is ~23,400 px/token — OCR-grade
inputs are unusable, and the routing policy had to steer all large images away from
nemotron. The fork's July 2026 analysis proposed an InternVL-style tiling patch and left
two open questions (12-vs-13 tile budget; trained tile markers).

An adversarially-verified investigation (2026-08-01; four parallel deep-dives over the
pinned llama.cpp source, the fork compat layer, upstream llama.cpp/ollama, and NVIDIA's
reference implementation) established:

1. **The model does not tile.** NVIDIA's reference (`image_processing.py` in
   `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16`, vLLM's
   `nano_nemotron_vl.py`, arXiv 2604.24954) performs a *single* aspect-preserving
   bicubic-antialiased resize within 1,024…13,312 pre-merge 16px patches (= 256…3,328
   tokens after the 2×2 pixel shuffle), upscales small images to the floor, adds no
   thumbnail, and never letterboxes. Tiling (12 + thumbnail) is the older
   Nemotron-Nano-**V2-VL-12B** scheme — a different model. The GGUF's `max_tiles=12` is a
   converter-fabricated default; the 12-vs-13 question dissolves.
2. **Stock llama.cpp is not even the correct degenerate case:** the model was trained
   with every image wrapped `<img>`(19)…`</img>`(20); llama.cpp emits no markers and pads
   with black bars.
3. **Upstream will not fix it soon.** llama.cpp PR #23638 implemented exactly the right
   fix and was closed unmerged on process grounds (2026-05-25); issue #25317 has been
   open and uncommented since 2026-07-05; master/b10091/ollama-main were all unchanged as
   of 2026-08-01.

## Decision

Carry a fork-local compat patch, `llama/compat/002-llama-cpp-nemotron-dynres.patch`
(three hunks, ported from PR #23638's final commit `66b9b344` onto the `b10091` pin):

1. `clip.cpp`: the NEMOTRON_V2_VL hparams branch calls
   **`set_limit_image_tokens(256, 3328)`** — deliberately diverging from the upstream
   commit's raw `image_{min,max}_pixels` assignment so that `--image-{min,max}-tokens`
   (and therefore Ollama's `image_min_tokens`/`image_max_tokens` options) become live.
2. `mtmd.cpp`: dispatch to `mtmd_image_preprocessor_dyn_size`; emit `<img>`/`</img>`.
3. `models/nemotron-v2-vl.cpp`: `resize_position_embeddings(GGML_SCALE_MODE_BICUBIC)`
   over the baked 32×32 grid (no-op at 512²).

Plus a `nemotron_h_omni` case in `visionServerArgs()` (defaults 256/3328; the
gemma4-shaped DefaultOptions values 40/1120 are treated as unset). Normative behavior:
[nemotron-dynres-patch.md § Spec](../nemotron-dynres-patch.md#spec--normative-behaviour).

## Alternatives considered

- **InternVL-style tiling patch** (the July plan): implements the wrong model's
  preprocessing; the served LLM was not trained on tiles. Rejected on reference evidence.
- **Wait for upstream / bump `LLAMA_CPP_VERSION`:** no fix exists upstream at any tag
  through b10216/master (2026-08-01). Rejected as a near-term path; resubmitting the
  corrected patch upstream (with the reference links the closed PR lacked) remains the
  exit strategy for carrying 002.
- **Repoint `clip.projector_type` to `"internvl"`:** aborts at tensor load (missing
  `mm_{0,1,3}_b` biases) and the graphs differ numerically. Confirmed dead end.
- **Client-side N-square-crops workaround:** still valid on unpatched payloads but feeds
  the model out-of-distribution input; superseded by the patch.
- **Raw pixel assignment as upstream wrote it:** identical budget, but leaves the flags
  dead and the knob arch-inconsistent with gemma4/qwen. Rejected for the
  `set_limit_image_tokens` variant.
- **GGUF-metadata-driven bounds** (reading the forwarded `min/max_num_patches` keys):
  rejected — for the shipped model they equal the hardcoded values exactly, and binding
  the budget to metadata would let a future variant GGUF silently declare bounds the
  graph was never validated at.

## Consequences

- Per-image cost becomes dynamic 256…3,328 (+2 marker tokens): up to 13× today's context
  consumption per image. Go-side truncation heuristics count images as zero for this arch
  (monolithic blob ⇒ no `ProjectorPaths`), so `num_ctx` must be budgeted explicitly.
- Warmup now probes the ceiling (~1846² image): worst-case memory surfaces at load
  instead of first request; load gets slower and heavier.
- **The overlay image cannot ship this** — the payload-pristine proof goes non-empty
  (true positive). Any deployment carrying 002 requires the full Dockerfile build
  ([nemotron-test-image.md](../nemotron-test-image.md) builds `ollama-rocm-nemotron`).
  Overlay-based builds must come from trees without the patch file.
- The 002 hunks target `b10091`; a build against the AMD-gate-pinned `b9888` payload
  needs a `git apply --check` (and possibly regeneration) against that tag. On the
  gfx1151 host this is not optional: the b10091 payload produced degenerate vision output
  there (rolled back 2026-07-31), so the deployable artifact is 002 cherry-picked onto
  the `85ebcb79`/0.32.1 lineage — the b10091-based test image validates mechanics only.
- The routing policy ("never send large images to nemotron3") and the measured
  px/token tables become stale for patched payloads and must be re-measured.
- The pixel ceiling is enforced before the per-dimension 32px minimum clamp, so
  degenerate aspect ratios (≈100:1+) can exceed it and exhaust memory — an inherited
  `dyn_size`-family edge (qwen/kimivl share it) that nemotron was previously immune to;
  absurd-aspect inputs must be filtered upstream of Ollama.
- Risks accepted pending runtime validation: `ggml_interpolate` bicubic on ROCm/gfx1151;
  interpolated-position-embedding quality vs the reference's 128×128-grid interpolation
  (a possible follow-up: keep the full grid through the compat layer); rounding-parity
  differences between `calc_size_preserved_ratio` and NVIDIA's exact snap algorithm
  (token counts land close but not always identical to the reference).
