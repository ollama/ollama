# Vision token budgets are per-architecture, not a general knob

MaxusAI-fork reference (fork-only; does not exist upstream). Written 2026-07-30 after
tracing why `nemotron3` charges exactly 256 visual tokens for every image regardless of
input size, and finding that `--image-max-tokens` cannot change it.

> **Update 2026-08-01:** the nemotron cap is now lifted by
> `llama/compat/002-llama-cpp-nemotron-dynres.patch` — and the fix is **not** the tiling
> patch sketched below; the model natively uses single-image dynamic resolution. See
> [nemotron-dynres-patch.md](nemotron-dynres-patch.md). The mechanism analysis in this doc
> remains correct for unpatched payloads; the [tiling notes](#notes-for-a-future-tiling-patch)
> are superseded and kept as a historical record with corrections annotated inline.

> **The one thing to take away:** `api.Options.ImageMinTokens` / `ImageMaxTokens` read like
> a general vision-budget control. They are not. `visionServerArgs()`
> ([`llm/llama_server.go`](../../llm/llama_server.go)) switches on **`modelArch`**, and an
> arch that is not in that switch gets no flags at all. Adding an arch to the switch is only
> half the job — the projector must also *consume* the flags on the llama.cpp side.

For what each arch actually **costs** at real image sizes — stock vs fork vs pinned, in
visual tokens and pixels-per-token — plus the routing policy that follows, see
[vision-token-budget-measurements.md](vision-token-budget-measurements.md).

## What each arch actually gets

| `modelArch` | flags ollama passes | effective budget | set by |
|---|---|---|---|
| `gemma4` | `--image-min-tokens` / `--image-max-tokens`, defaults **40 / 1120** | 40 … 1,120 tokens | `set_limit_image_tokens(40, 280)`, ceiling raised by our flags |
| `qwen2vl`, `qwen25vl`, `qwen3vl`, `qwen3vlmoe`, `qwen35`, `qwen35moe` | `--image-min-tokens 1024` (fixed) | 1,024 … 4,096 tokens | `set_limit_image_tokens(8, 4096)`, floor raised by our flag |
| **`nemotron_h_omni`** | `--image-min-tokens` / `--image-max-tokens`, defaults **256 / 3328** | 256 … 3,328 with the 002 patch; **exactly 256 (flags inert) on an unpatched payload** | `set_limit_image_tokens(256, 3328)` added by `llama/compat/002-llama-cpp-nemotron-dynres.patch` |
| `mistral3` | none | 8 … 1,024 **grid** tokens; per-image cost is grid + rows (see below), worst case **2,048** | `set_limit_image_tokens(8, 1024)` (pixtral projector) |
| `glmocr` | none | 8 … 4,096 tokens | `set_limit_image_tokens(8, 4096)` (glm4v projector) |
| `llama4` | none | structural: 144 per 336² tile + 144 overview + 2 markers ⇒ 146 (untiled) … 1,442 (3×3 grid) | 336² tiling (`set_llava_uhd_res_candidates(3)`), not a pixel budget |
| `deepseekocr` | none | structural: 273 + fused rows + 1 ⇒ 274 … 1,264 | hardcoded 1024² global + 640² tiles (2…9), not a pixel budget |
| everything else | none | whatever the projector defaults to | llama.cpp |

## The non-budgeted compat arches (traced 2026-08-01, llama.cpp b10091)

None of these four consume `--image-{min,max}-tokens`, so `api.Options.ImageMinTokens` /
`ImageMaxTokens` are dead knobs for them (`visionServerArgs()` passes nothing). Exact
per-image context cost — replicated by `llm.ImageTokensForSize()` and pinned by
`TestImageTokensForSize`; the replication was byte-diffed against the verbatim b10091
C++ over a ~5,500-size sweep:

- **`mistral3`** (PROJECTOR_TYPE_PIXTRAL via `handle_mistral3_clip`): dyn_size
  `smart_resize` on a 28 px grid (patch 14 × `mistral3.spatial_merge_size` 2), budget
  8…1,024 grid tokens. Cost = `cols·rows + rows`: the grid, **plus one `[IMG_BREAK]`
  embedding per row except the last, plus a lone `[IMG_END]` text token** (there is no
  image-begin marker) — not grid + 2. A 1024×1024 image costs 32×32 + 32 = 1,056; the
  worst case is a one-column strip at the pixel budget, 1,024 + 1,024 = 2,048. The old
  flat 768/image heuristic under-counted large images by ~40%.
- **`glmocr`** (PROJECTOR_TYPE_GLM4V via `handle_glmocr_clip`): qwen-shaped dyn_size on
  the same 28 px grid (patch 14 × merge 2) but with the stock floor of 8 — this arch
  does **not** get the qwen `--image-min-tokens 1024` pin. Cost = grid + 2, ceiling
  4,096 + 2. At 1920×1080 that is 69×39 + 2 = 2,693 — 3.5× the old 768 heuristic.
- **`llama4`** (PROJECTOR_TYPE_LLAMA4 via `handle_llama4_clip`): llava_uhd tiling, 336²
  tiles of (336/14)²/2² = 144 embeddings. ≤ 336² ⇒ overview only (146 with markers);
  larger images pick the best-fit grid from (x·336, y·336), x,y ∈ 1…3 minus 1×1, and pay
  144 per tile plus the trailing 144-token overview. **At b10091 the
  `MTMD_SLICE_TMPL_LLAMA4` tile separators are gone from `mtmd.cpp`** — only
  `<|image_start|>`/`<|image_end|>` wrap the embeddings. Ceiling 144·10 + 2 = 1,442.
- **`deepseekocr`** (PROJECTOR_TYPE_DEEPSEEKOCR via `handle_deepseekocr_clip`): geometry
  hardcoded in `clip.cpp` (patch 16, base 1024, tile 640, 2…9 tiles), so GGUF KVs cannot
  move it. Every image pays the padded 1024² global view: 16² grid + 16 image-newlines +
  1 view separator = 273 embeddings. Images over 640 px in either dimension add the
  closest-aspect tile grid as **fused rows** of (10·gridW + 1)·10 embeddings each. The
  only text marker is a trailing `"\n"`. Ceiling is the 1×9 grid: 273 + 9·110 + 1 =
  1,264. Note the cost is *not* monotonic in image size — the grid choice tracks aspect
  ratio, not area, so 800×600 (3×2 grid, 894) costs more than the larger 1000×1000
  (2×2 grid, 694).

These are analytical values validated against the b10091 source, not yet end-to-end
measurements; when one of these models is next served, spot-check with the
`prompt_eval_count` method in
[vision-token-budget-measurements.md](vision-token-budget-measurements.md) and extend its
tables.

## Why nemotron was stuck at 256 (mechanism; applies to unpatched payloads)

`--image-{min,max}-tokens` land in `hparams.custom_image_{min,max}_tokens`, and the only
consumer that affects behaviour is `clip_hparams::set_limit_image_tokens()`. The
`PROJECTOR_TYPE_NEMOTRON_V2_VL` hparams branch **never calls it** — it reads only
`KEY_PROJ_SCALE_FACTOR`. mtmd then dispatches that projector to
`mtmd_image_preprocessor_fixed_size`, which resizes every image onto one
`image_size × image_size` canvas, so the token count is structural:

```
(image_size / patch_size)² / n_merge²  =  (512 / 16)² / 2²  =  256 tokens
```

Measured end to end on `nemotron3:33b-q4_K_M`: constant across a 576× area range, and
`image_max_tokens` at 2048, 4096 and even 64 all left it at exactly 256 — it cannot even be
lowered. Attaching N images costs exactly N × 256.

**Superseded 2026-08-01:** `visionServerArgs()` now *does* have a `case "nemotron_h_omni"`,
because the 002 patch makes the projector consume the flags (the old
`TestVisionServerArgs/nemotron_h_omni` want-nil case is gone). On an unpatched payload the
flags are parsed but inert — the same silent no-op the measurements doc records for stock
gemma4. Upstream `master` was re-checked 2026-08-01: still unchanged on every relevant line
(the one upstream fix, PR #23638, was closed unmerged on process grounds; issue #25317 sits
uncommented), so a `LLAMA_CPP_VERSION` bump alone still cannot fix it.

## Notes for a future tiling patch

> **Superseded 2026-08-01 — kept as a historical record.** The shipped fix
> ([nemotron-dynres-patch.md](nemotron-dynres-patch.md), `llama/compat/002-…`) is *not*
> this tiling design: NVIDIA's reference does a single dynamic-resolution resize, no
> tiles, no thumbnail. Specific corrections to the notes below, verified against code:
> **(a)** the "12 vs 13" question dissolves — 13312 is a per-image patch cap, and the
> GGUF's `max_tiles=12` is a converter-fabricated default; **(b)** feeding 13312 into the
> tile-count keys would *pass* the `GGML_ASSERT` (1024 ≤ 13312 < INT32_MAX) — the real
> failure is an empty `image_res_candidates` aborting later in
> `mtmd_image_preprocessor_internvl::preprocess`; **(c)** "the graph side needs no
> change" held only for this tiling design — the shipped fix must interpolate the baked
> 32×32 position embeddings (`resize_position_embeddings`); **(d)** the model wraps every
> image in `<img>`/`</img>` (token ids 19/20), which stock llama.cpp omits entirely.

`handle_nemotron_h_omni_clip()` in `llama/compat/llama-ollama-compat.cpp` forwards six
`nemotron_h_omni.vision.*` keys into the `clip.vision.*` namespace — `image_token_id`,
`image_start_token_id`, `image_end_token_id`, `max_tiles`, `min_num_patches`,
`max_num_patches`. **Nothing reads any of them**, so the block reads like dynamic tiling is
wired up when it is not.

> 🛑 These notes live here rather than as a comment in that file **on purpose**. `llama/` is
> under a byte-equality invariant against the upstream tag — it is upstream's code, and the
> overlay image in [gemma4-budget-image.md](gemma4-budget-image.md) is only valid because the
> fork's payload is identical to the base image's. A comment there is enough to make the
> pre-rebuild payload proof non-empty and block a rebuild. **Do not document things inside
> `llama/`.**

The values are worth knowing, because they are the record that this model *ships* a
dynamic-tiling configuration that llama.cpp does not implement:

```
max_tiles        = 12          use_thumbnail   = true      (not forwarded; nothing reads it)
min_num_patches  = 1024        =  1 tile  × (512/16)²
max_num_patches  = 13312       = 13 tiles × (512/16)²      = 12 tiles + 1 thumbnail
```

After the 2×2 merge that is 13,312 / 4 = **3,328 visual tokens**, 13× what it gets today. A
patch would need to:

1. **Dispatch to a tiling preprocessor** in `mtmd.cpp` instead of `fixed_size` —
   `mtmd_image_preprocessor_internvl` is the natural candidate, being the llava-uhd subclass
   nemotron is already grouped with elsewhere in clip.cpp.
2. **Read tile bounds** in the `PROJECTOR_TYPE_NEMOTRON_V2_VL` hparams branch and populate
   `image_res_candidates`, as the `INTERNVL` branch does.
3. **Convert the units.** clip.cpp reads `clip.vision.preproc_{min,max}_tiles`
   (`KEY_PREPROC_{MIN,MAX}_TILES`), and **those are tile counts** — InternVL's defaults are 1
   and 12 — whereas `min/max_num_patches` above are **patch counts**. Divide by
   `(image_size/patch_size)² = 1024` first. A rename alone feeds `13312` into a tile count and
   trips the `min <= max && max < INT32_MAX` assertion at `clip.cpp:1335`.
4. **Emit tile markers.** nemotron sets no `img_beg`/`img_end` in mtmd today; multi-tile input
   needs the begin/end and per-tile markers the LLM was trained on.

Two things deliberately left open, because guessing them in the translation layer would be
worse than leaving them visible:

- **12 vs 13.** Should the tile budget be 12 (`max_tiles`, thumbnail excluded, as InternVL
  counts it) or 13 (`max_num_patches/1024`, thumbnail included)? Genuinely ambiguous from the
  metadata alone.
- **The graph side needs no change** — tiles are each 512², so `clip_graph_nemotron_v2_vl` and
  the ViT position embeddings are used exactly as now, just N times. That is what makes this
  tractable at all.

One tempting shortcut is a **dead end**: re-pointing `clip.projector_type` at `"internvl"` to
borrow its tiling fails, because the INTERNVL tensor loader requires `mm_0_b`/`mm_1_b`/`mm_3_b`
biases that the nemotron projector does not have, so model load aborts.

⚠️ Also note a patch here **cannot ship via the overlay image**. `Dockerfile.gemma4budget`
rebuilds only the Go binary and takes the C++/CUDA payload from the `ollama/ollama` base, so a
clip.cpp/mtmd.cpp change needs a full Dockerfile build.

⚠️ nemotron also **letterboxes** onto its square canvas (`PAD_CEIL` + black, the
`clip_hparams` defaults it never overrides). Aspect ratio is preserved, but for a 16:9
photo roughly 44% of the 256 tokens encode black bars. Since each attached image costs
exactly 256, sending square crops as multiple images recovers them without any patch.

## How token count actually works

For the arches that do have a budget:

```
tokens = round_half_up(w / S) × round_half_up(h / S)      where S = patch_size × n_merge
         with w,h first scaled so that  min_tokens ≤ pixels / S² ≤ max_tokens
```

`image_max_pixels = max_tokens × S²`, which is why the server logs a pixel figure at load
(`2580480` for gemma4 at budget 1120 — that is 1120 × 48²). The pixel ceiling and the token
ceiling are **the same constraint in different units**, not two different limits.

Consequences worth knowing:

- **Tune by total pixel count, not by how long one edge looks.** A 4032×189 panorama is
  762,048 px — well *under* gemma4's 2,580,480 ceiling — so it is never downscaled and simply
  costs `area / S²` ≈ 331 tokens. Wide images run out of pixels before they run out of budget.
- **The nominal ceiling is not always reachable**, because the grid is integral. At 16:9,
  gemma4 tops out at 44×25 = 1,100 of its 1,120. It *is* exactly attainable at other shapes
  (1536×1680 gives 32×35 = 1,120).

## Checking a new arch before you promise a knob

1. Is it in `compatClipArches` (`llm/llama_server.go`)? That only means the fork can point
   `--mmproj` at an Ollama-format GGUF — it says nothing about budgets.
2. What `clip.projector_type` does `llama/compat` give it? That decides which clip.cpp
   hparams branch runs.
3. Does that branch call `set_limit_image_tokens(...)`? **If not, the flags are inert and no
   Go change can help.**
4. Quick empirical check: if the model-load log prints no `image_min_pixels` /
   `image_max_pixels` line, it has no token budget — those lines are gated on the values
   being > 0, and a projector that never calls `set_limit_image_tokens()` leaves them at the
   `-1` sentinel.

## See also

- [`gemma4-budget-image.md`](gemma4-budget-image.md) — building/deploying the patched image,
  and the measured gemma4 budget behaviour.
- [`../design/gemma4-vision-token-budgets-upstream-rebase.md`](../design/gemma4-vision-token-budgets-upstream-rebase.md)
  — why the feature is Go-only rather than a C++ patch.
