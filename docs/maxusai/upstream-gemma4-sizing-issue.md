# Draft upstream issue: gemma4 image sizing diverges from reference, breaking box_2d grounding

Fork-internal draft for filing against **ggml-org/llama.cpp** (written 2026-08-07,
against `b10091`; re-verify against master before filing — the gemma4 hparams
branch was unchanged on master as of this date). File as one issue with two
defects; the fix for both is one patch:
[`llama/compat/004-llama-cpp-gemma4-budget-fill.patch`](../../llama/compat/004-llama-cpp-gemma4-budget-fill.patch).

---

**Title:** Gemma 4 (mtmd): image sizing diverges from reference preprocessing —
off-budget patch grids break `box_2d` grounding; letterbox padding skews
coordinates

## Summary

Gemma 4's reference preprocessor (HF transformers `image_processing_gemma4.py`,
`get_aspect_ratio_preserving_size` + `aspect_ratio_preserving_resize`) always
resizes an image — up **or** down — so the 48-aligned patch grid fills the
visual-token budget, and it resizes **directly** (`tvF.resize` to exact target;
no padding — its independent per-axis floor is deliberately anisotropic).
Supported budgets are exactly 70/140/280/560/1120
(ai.google.dev/gemma/docs/core/model_card_4).

llama.cpp diverges twice for `PROJECTOR_TYPE_GEMMA4V`/`GEMMA4UV`:

1. **Under-budget images keep their natural rounded grid**
   (`img_tool::calc_size_preserved_ratio`, the Qwen2-VL `smart_resize` clone,
   `tools/mtmd/mtmd-image.cpp`). A 1920×1080 image at the 40…1120 default range
   lands on a 40×23 = 920-token grid. No supported budget can produce that grid:
   reference grids satisfy `cols·rows ≤ B < (cols+1)·(rows+1)`. The model
   measurably cannot ground on such "off-ladder" grids — its `box_2d` vertical
   coordinates acquire a 5–19% scale error while horizontal stays exact.
   Additionally, pinning `min == max` overshoots the budget via the
   `< min_pixels` branch's `ceil_by_factor` (requesting 1120 delivers 1170).
2. **Letterbox padding.** The gemma4 branch never sets `image_resize_pad`, so it
   inherits `PAD_CEIL` (`clip-model.h`): aspect-preserving fit + centered black
   bars. The model emits norm-1000 `box_2d` against the padded canvas; decoding
   against the original image (as Google's own examples do) inherits a
   `content/canvas` scale error on the padded axis — measured at 2–5%, matching
   the prediction to 3–4 significant figures across 12 cells (4 budgets × 3
   model sizes) and snapping to exactly 1.000 wherever the pad vanishes.

## Minimal reproduction (defect 1)

Two synthetic scenes, identical content, both dimensions multiples of 48, budget
range 40…1120 so `target == source` (no resample, no pad — pure grid effect):

| input | grid | tokens | reachable at a supported budget? | bbox IoU (12B) | y-scale |
|---|---|---|---|---|---|
| 2160×1152 | 45×24 | 1080 | yes (1080 ≤ 1120 < 46·25) | **0.943** | 0.9938 |
| 2064×1152 | 43×24 | 1032 | **no** (no budget in [1032, 44·25)) | 0.799 | 0.9497 |

Fewer tokens, same content, same aspect — the only difference is grid
reachability. With the sizing fixed (fill-to-budget), a 1920×1080 image lands on
44×25 = 1100 and mean IoU on a six-object scene goes 0.504 → 0.885 (12B),
0.729 → 0.961 (31B), 0.810 → 0.970 (26B A4B), measured at temperature 0 against
pixel ground truth.

## Fix

Mirror the reference: `factor = sqrt(budget·48² / (W·H))`, floor each axis to
48, snap the requested ceiling down to the supported ladder, and resize with
`PAD_NONE`. Patch attached (three hunks: `clip-model.h` opt-in flag, `clip.cpp`
gemma4 branch, `mtmd-image.cpp` `calc_size_budget_fill` + dyn_size branch); the
flag defaults off so no other projector changes behaviour (verified:
nemotron_v2_vl `prompt_eval_count` byte-identical).

Note the trade: reference behaviour upscales small images to the budget, so a
640×480 image costs ~1064 tokens at budget 1120 instead of 132. That is what the
model was trained on, and per the model card users select lower budgets for
speed.

## Evidence quality

- Every claim above measured on Gemma 4 12B/26B/31B (q4_K_M), M-series Metal,
  temperature 0, six-object synthetic scene + invoice document with pixel ground
  truth; full write-up in the fork:
  `docs/maxusai/gemma4-bbox-investigation-findings.md` (MaxusAI/ollama).
- Reference behaviour verified in transformers source (torchvision and PIL
  backends), and consistent with Google DeepMind's JAX reference and KerasHub
  (all resize directly to the aligned target; none pad).
