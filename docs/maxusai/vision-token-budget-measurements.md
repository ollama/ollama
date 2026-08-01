# Vision token budgets: measured cost, and the routing policy that follows

MaxusAI-fork reference (fork-only; does not exist upstream). Written 2026-07-31 from
end-to-end measurements on the gfx1151 host (Ryzen AI Max+ 395 / Radeon 8060S, ROCm),
serving `0.32.1-gemma4budget-85ebcb79` and `0.32.5-gemma4budget-0d23f7a6`.

Companion to [vision-token-budgets-by-arch.md](vision-token-budgets-by-arch.md), which
explains *why* the budget is arch-gated. This one answers *what it costs* — and what to do
about it.

> **The one thing to take away:** the fork's patch and the `image_min_tokens` pin solve
> **different** problems. The patch raises the *ceiling* (280 → 1120) and only helps images
> above ~800 kpx. The pin raises the *floor* (40 → 1088) and only helps images below that.
> Neither substitutes for the other, and at 640×480 the patch alone changes nothing.

## Method

`prompt_eval_count` from `/api/generate` with `num_predict: 1`, minus that model's
text-only baseline measured the same way. Baselines differ per model (gemma4 17,
qwen3.5 11, nemotron3 18) and must be subtracted, or small-image numbers are badly
distorted. Solid-colour PNGs: token cost is a function of geometry, not content.

Counts are **grid-quantised** — they land on discrete cells, so nearby budget values
produce identical results (e.g. `image_min_tokens` anywhere in 1056–1088 gives the same
count). Do not read small differences as meaningful.

**These numbers are the grid product + 2.** The measured values reconcile exactly with the
`round_half_up(w/S) × round_half_up(h/S)` formula in
[vision-token-budgets-by-arch.md](vision-token-budgets-by-arch.md#how-token-count-actually-works),
plus a constant 2 for the image begin/end markers, which `prompt_eval_count` includes and a
pure grid calculation does not:

| size | grid product | measured here |
|---|---|---|
| 640×480 | 13 × 10 = 130 | 132 |
| 896×896 | 19 × 19 = 361 | 363 |
| 1568×1568 | 33 × 33 = 1089 | 1091 |
| 1920×1080, pinned | 44 × 25 = 1100 | 1102 |

Compute the grid on the **post-scaling** dimensions: when a floor or ceiling binds, the
image is resized first, so the source width and height are not the ones that reach the grid.
The 1920×1080 row above is that doc's own worked 16:9 example.

## Measurements — visual tokens

| model | 640×480 | 896×896 | 1920×1080 | 1568×1568 | 3000×2000 |
|---|---|---|---|---|---|
| `gemma4:*` — **stock upstream, no fork** (40…**280**) | 132 | 258 | 266 | 258 | 262 |
| `gemma4:*` — fork, default (40…1120) | 132 | 363 | 922 | 1091 | 1082 |
| `gemma4:*` — fork, pinned (1088/1120) | 1133 | 1091 | 1102 | 1091 | 1082 |
| `qwen3.5:*` / `qwen3.6:*` — **before** `87cf1100` | 302 | 786 | 2042 | 2403 | 4058 |
| `qwen3.5:*` / `qwen3.6:*` — **after** `87cf1100` | 1038 | 1026 | 2042 | 2403 | 4058 |
| `nemotron3:33b-q8` (`nemotron_h_omni`) | 256 | 256 | 256 | 256 | 256 |

Stock never exceeds 266 — the 280 ceiling, landing just under it on the patch grid.
Verified on `ollama/ollama:0.32.5-rocm`, which reports version `0.32.5`.

The `nemotron3` row is **pre-002-patch** (all deployed payloads to date). With
[the 002 dynamic-resolution patch](nemotron-dynres-patch.md) expect grid+2 like the other
arches: ≈302 at 640×480, ≈2042 at 1920×1080, ≈3330 at the ceiling — re-measure and extend
this table when a patched payload ships.

## Measurements — pixels encoded per visual token

Source pixels ÷ measured visual tokens. **Lower is finer detail retained.** This is the
number to reason about when choosing a model for a given image.

| model | 640×480 | 896×896 | 1920×1080 | 1568×1568 | 3000×2000 |
|---|---|---|---|---|---|
| `gemma4:*` — stock upstream | 2,327 | 3,112 | 7,795 | 9,530 | **22,901** |
| `gemma4:*` — fork, default | 2,327 | 2,212 | 2,249 | 2,254 | 5,545 |
| `gemma4:*` — fork, pinned | **271** | 783 | 1,882 | 2,254 | 5,545 |
| `qwen3.5:*` — before | 1,017 | 1,021 | 1,015 | 1,023 | 1,479 |
| `qwen3.5:*` — after | 296 | 783 | 1,015 | 1,023 | 1,479 |
| `nemotron3:33b-q8` | 1,200 | 3,136 | 8,100 | 9,604 | **23,437** |

## What the numbers say

**What the fork actually buys.** 258 → 1091 at 1568² (**4.2×**), 266 → 922 at 1920×1080
(3.5×). That is the entire justification for the patch, and it is invisible if you only
compare fork builds against each other.

**Unpatched gemma4 is as coarse as nemotron on large images** — 22,901 vs 23,437 px/token
at 3000×2000, a 2% difference. The arch everyone treats as the weak one was, without the
patch, indistinguishable from gemma4 at high resolution.

**The patch buys nothing at 640×480.** Stock and patched both give 132: the ceiling never
binds and the floor of 40 does not lift it. Only the pin helps there (132 → 1133, **8.6×**).

**Default gemma4 is worse than nemotron on small images** — 2,327 vs 1,200 px/token at
640×480, because the floor of 40 lets a small image collapse to 132 tokens. Feeding
thumbnails or small crops to gemma4 at defaults yields *less* detail than nemotron.

**`image_max_tokens` is silently ignored upstream.** Requesting 1120 against stock returns
258, versus 1091 on a patched server. No error, no warning — the option does not exist
upstream and is dropped. A client cannot tell it had no effect.

**`87cf1100` only bites below ~1 MP.** At 1568² and above, qwen35 was already past the 1024
floor. The gain is 302 → 1038 at 640×480 (3.4×) and 786 → 1026 at 896² (1.3×). At
3000×2000 it sits at 4058, effectively the 4096 ceiling.

**nemotron is fixed at 256 and cannot be moved — on unpatched payloads** (every deployed
payload when these rows were measured). Constant across a 19.6× area range
(307 kpx → 6 Mpx), and `image_min_tokens`/`image_max_tokens` at 4096 *and* 64 both leave it
at exactly 256 — it cannot even be lowered. It also letterboxes onto a square canvas, so
for 16:9 roughly 44% of those 256 tokens encode black bars. See
[vision-token-budgets-by-arch.md](vision-token-budgets-by-arch.md) for the mechanism;
with `llama/compat/002-llama-cpp-nemotron-dynres.patch` all of this changes — dynamic
256…3,328, no letterbox, live flags ([nemotron-dynres-patch.md](nemotron-dynres-patch.md)).

## Spec — normative behaviour

For `modelArch` in `visionServerArgs()` ([`llm/llama_server.go`](../../llm/llama_server.go)):

| arch | flags passed | resulting budget |
|---|---|---|
| `gemma4` | `--image-min-tokens` / `--image-max-tokens`, from `api.Options`, defaulting **40 / 1120** | 40 … 1,120 |
| `qwen2vl`, `qwen25vl`, `qwen3vl`, `qwen3vlmoe`, `qwen35`, `qwen35moe` | `--image-min-tokens 1024` (fixed, not tunable) | 1,024 … 4,096 |
| `nemotron_h_omni` | `--image-min-tokens` / `--image-max-tokens`, defaults **256 / 3328** | 256 … 3,328 on a payload with the 002 patch; exactly 256 (flags inert) unpatched — see [nemotron-dynres-patch.md](nemotron-dynres-patch.md) |
| everything else | none | projector default |

Option resolution: `ImageMinTokens` / `ImageMaxTokens` are plain `int` with `omitempty`.
`gemma4ImageTokenBudget()` treats `<= 0` as unset and substitutes the defaults, so a JSON
`null` — or an omitted field — yields **40 / 1120**, not "no limit". Both are **Runner**
options: changing either reloads the runner. `nemotronImageTokenBudget()` additionally
treats the exact gemma4-shaped defaults (40/1120) as unset — explicit 40 or 1120 is not
expressible for that arch — and clamps the ceiling to the trained 3,328; see the
[normative spec in nemotron-dynres-patch.md](nemotron-dynres-patch.md#spec--normative-behaviour).

## Decision record

**Context.** The budget is arch-gated and the defaults are asymmetric: the ceiling was
raised to 1120 but the floor left at llama.cpp's 40. Measurement shows the floor, not the
ceiling, dominates cost for sub-megapixel images — and that three separate diagnostic cycles
in July 2026 misread the floor as a wrong or unpinned build.

**Decisions.**

1. **Route by image size, not by model reputation.** Above ~1 MP prefer `qwen3.5`/`qwen3.6`
   (up to 4,096) or patched `gemma4` (up to 1,120). Do not send large or detail-critical
   images to `nemotron3`.
2. **Pin the floor server-side when uniformity matters**, via a model manifest rather than
   per-request options, so clients may keep sending `null`:
   ```bash
   printf 'FROM gemma4:e2b-it-q4_K_M\nPARAMETER image_min_tokens 1088\nPARAMETER image_max_tokens 1120\n' \
     | ollama create gemma4:e2b-it-q4_K_M-budget -f -
   ```
   Pin once in the manifest; do not vary per request mid-run, or every change reloads the
   runner.
3. **Do not add `case "nemotron_h_omni"`** to `visionServerArgs()`. The projector never
   consumes the flags, so it would ship a knob that does nothing.
   `TestVisionServerArgs/nemotron_h_omni` asserts it stays absent. For nemotron, recover
   detail by tiling — each attached image costs a flat 256, so N square crops cost N × 256.

   > **Superseded 2026-08-01.** The case now exists and the flags are live on payloads
   > carrying `llama/compat/002-llama-cpp-nemotron-dynres.patch`
   > ([nemotron-dynres-patch.md](nemotron-dynres-patch.md)); the regression test now
   > asserts the flags are *present* (defaults 256/3328). The N-square-crops workaround
   > remains valid for unpatched payloads only — the model was not trained on tiles, so
   > prefer the patch. Decision 1's "do not send large images to nemotron3" also needs
   > re-measuring once a patched payload ships.
4. **Treat "budget → constant token count" as false.** Pinning tightens the spread from
   132–1091 (Δ959) to 1082–1162 (Δ80); it does not flatten it. Cost tracks pixel **area**,
   not orientation — every transposed pair measured identical (1920×1080 ≡ 1080×1920).

**Consequences.** Deploying `87cf1100` raises image token cost, and therefore context
consumption, for the six qwen35/qwen35moe vision models on sub-megapixel inputs. That is the
upstream-recommended setting for grounding and counting, but it is not a no-op — budget for
it.

## Scope of these measurements

- `gemma4:e2b-it-q4_K_M` and `gemma4:31b-it-q4_K_M` returned **identical** counts on every
  row, so within an arch the budget does not vary by parameter count or quant.
- qwen35 was probed with `qwen3.5:0.8b-q8_0` only. The flag applies to all six
  qwen35/qwen35moe models; identical counts across them is likely (it held for gemma4) but
  **not verified**.
- nemotron was probed on `q8`; the by-arch doc measured `q4_K_M`. Both give 256, so the
  behaviour is not quant-specific.
- Aspect ratios covered: 1:1, 4:3, 3:4, 16:9, 9:16, 3:1, 1:3, 3:2. Audio input on
  `nemotron_h_omni` was **not** tested.
