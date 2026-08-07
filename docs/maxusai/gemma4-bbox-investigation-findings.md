# Findings: gemma4 bbox geometry vs visual token budget

Resolution of [gemma4-bbox-aspect-investigation.md](gemma4-bbox-aspect-investigation.md)
(brief opened 7820acf5, 26B cell added 27e05deb). Investigated 2026-08-07.

**Status: RESOLVED. The brief's hypothesis is refuted; the actual cause is
identified and verified by prediction: the model only grounds accurately on patch
grids its five supported budgets can produce (`c·r ≤ B < (c+1)·(r+1)` for
B ∈ {70, 140, 280, 560, 1120}), and llama.cpp's sizing routinely delivers grids
outside that set. A faithful 1120 — grid 44×24 — scores IoU 0.952 on 31B, the best
bbox result of the campaign. A separate, smaller letterbox-padding bug is confirmed
in source and quantified on the horizontal axis.**

Method note: most of this needed no new model runs. The gitignored `resp_*.json`
and `scores_*.json` under `vision-suite/` already contained every emitted box and
`prompt_eval_count` for every rung. Two new controlled experiments were run on 12B
for the parts the archive could not settle.

## Summary

| Claim | Verdict |
|---|---|
| Per-dimension rounding distorts aspect → breaks norm-1000 decode | **Refuted** — geometrically impossible, and contradicted by measurement |
| Not in the vision encoder | **Confirmed** (27e05deb reached this independently; this doc adds the per-size vertical decomposition) |
| Aspect-correlated; the 1:1 document is immune | **Refuted** — a metric artifact; the square image degrades too |
| Padding is an unisolated confound | **Confirmed as a real bug**, but it is *not* the cause of the collapse |
| Pinned 1120 upscales beyond native | **Confirmed** (1170-patch grid = 45×26 = 2160×1248) |

## 1. Aspect distortion cannot be the mechanism

Normalized coordinates are invariant to anisotropic resize. If content `W0×H0` is
resized by any pair of factors to fill a `Wt×Ht` canvas, a feature at `(x0,y0)`
lands at `(x0·Wt/W0, y0·Ht/H0)`, whose norm-1000 coordinate is `1000·x0/W0` —
independent of `Wt` and `Ht`. The distortion cancels exactly. "The model emits
box_2d against the distorted canvas; the scorer decodes against the original
dimensions" describes a no-op.

Measurement agrees. Budget 280 produces a `1056×576` canvas — a **+3.13%** aspect
error, among the largest on the ladder — and its measured vertical mapping is exact
(y-scale 0.9994 on 12B). This is why aspect-error magnitude never rank-ordered IoU:
it is not a causal variable.

## 2. The canvas table is now observed, not replicated

`prompt_eval_count` decomposes exactly as `text + patch_grid + 16`, where the 16 is
a constant per-image framing overhead (verified on both tests at all six
configurations; text-only is 568 tokens for the scene prompt, 342 for the document
prompt). The patch-grid counts factor uniquely onto the 48px grid:

| rung | scene grid | canvas | document grid | canvas |
|---|---|---|---|---|
| 70 | 11×6 = 66 | 528×288 | 8×8 = 64 | 384² |
| 140 | 15×8 = 120 | 720×384 | 11×11 = 121 | 528² |
| 280 | 22×12 = 264 | 1056×576 | 16×16 = 256 | 768² |
| 560 | 31×17 = 527 | 1488×816 | 23×23 = 529 | 1104² |
| 1120 | 45×26 = 1170 | 2160×1248 | 34×34 = 1156 | 1632² |
| shipped `40…1120` | 40×23 = 920 | 1920×1104 | 33×33 = 1089 | 1584² |

This confirms every row of the brief's predicted table as an *observation* rather
than a replication of the arithmetic, and it reconciles the token counts already in
[vision-token-budget-measurements.md](vision-token-budget-measurements.md): those
are image-block counts (grid + 16), so the brief's "~936" for the shipped range is
correct — 920 patches plus framing. Next-step #1 is discharged without instrumenting
C++.

Incidentally: the 1120 rung produces a **1170-patch grid — more than the 1120
requested**. When the `< min_pixels` branch fires it uses `ceil_by_factor`, which can
overshoot `max_pixels`. Small separate upstream bug.

## 3. Letterbox padding is real and is a genuine coordinate bug

gemma4 never sets `image_resize_pad`, so it inherits the `clip-model.h:65` default
`PAD_CEIL`. `img_tool::resize` (`mtmd-image.cpp:39–117`) under `PAD_CEIL` does an
aspect-preserving fit and a **centered black letterbox**:

```cpp
float scale = std::min(scale_w, scale_h);
new_width  = std::min((int)std::ceil(src.w * scale), target_resolution.width);
new_height = std::min((int)std::ceil(src.h * scale), target_resolution.height);
fill(dst, pad_color);                                   // default {0,0,0} — black
offset_x = (target_resolution.width  - new_width)  / 2; // centered
offset_y = (target_resolution.height - new_height) / 2;
composite(dst, resized_image, offset_x, offset_y);
```

The model emits norm-1000 `box_2d` against the **full padded canvas**; the scorer —
and any consumer — decodes against the **original** dimensions. The pad therefore
becomes a scale-plus-offset error on the padded axis:

    axis scale  = content_extent / canvas_extent
    axis offset = (pad/2) · original_extent / canvas_extent

Measured horizontal scale against that prediction, scene test, all three sizes:

| rung | canvas | x pad | predicted | 12B | 26B | 31B |
|---|---|---|---|---|---|---|
| 140 | 720×384 | 37px | 0.9481 | 0.9526 | 0.9449 | 0.9483 |
| 280 | 1056×576 | 32px | 0.9697 | 0.9689 | 0.9658 | 0.9736 |
| 560 | 1488×816 | 37px | 0.9749 | 0.9733 | 0.9730 | 0.9749 |
| 1120 | 2160×1248 | none | 1.0000 | 1.0004 | 1.0015 | 1.0003 |
| shipped range | 1920×1104 | none | 1.0000 | 1.0022 | — | — |

Twelve cells tracking the prediction to 3–4 significant figures, snapping to exactly
1.000 precisely where the x-pad disappears. The centered-pad offset is confirmed too
(280: predicted +29.1px, measured +25.7px). A model-side box-tightness bias cannot
explain this — it would shrink both axes and would not vanish exactly when the pad does.

**This answers the brief's next-step #4**: padding *is* applied, and the offset is
*not* corrected during box decode. It costs 2–5% on whichever axis gets padded.

## 4. …but padding is not what causes the collapse

Controlled A/B, 12B, shipped default budget, same content, same font, same grid
(40×23). The second arm is pad-free by construction: `target == source`, so
`img_tool::resize` takes its `dst.get_size() == src.get_size()` early return — no
resample, no letterbox.

| arm | canvas | padding | IoU | y-scale | y-offset |
|---|---|---|---|---|---|
| 1920×1080 | 1920×1104 | 12px black bars | 0.505 | 0.8186 | +12.0 (pad predicts +11.7) |
| 1920×1104 | 1920×1104 | **none** | 0.501 | 0.8435 | −1.3 (pad predicts 0) |

The padded arm reproduces the archived `native` run to within noise (0.504 / 0.8250),
so the re-render is faithful. Removing padding entirely removes only the **offset**.
The y-**scale** error — the thing that actually destroys IoU — survives untouched.

## 5. The error is vertical-only and not aspect-related

Three pad-free arms at ~880–900 patches (every dimension a multiple of 48):

| arm | size | grid | short axis | x-scale | y-scale | IoU |
|---|---|---|---|---|---|---|
| landscape | 1920×1056 | 40×22 | y | 1.0021 | 0.9200 | 0.578 |
| portrait | 1056×1920 | 22×40 | x | 1.0017 | 0.8112 | 0.406 |
| **square** | 1440×1440 | 30×30 | — | 1.0026 | 0.9057 | 0.477 |

x-scale is 1.002 ± 0.001 in all three, including portrait where x is the *short*
axis. The **square** arm — no aspect error possible, no padding, no resampling —
still loses 9.4% vertically. The error follows neither the short axis nor the aspect
ratio. It is specifically the vertical coordinate, and it grows with patch rows.

## 6. "The 1:1 document is immune" is a metric artifact

This is asserted by the brief's constraint 3 and restated in 27e05deb ("The 1:1
document still shows no collapse at 1120 on any size"). Both rest on
`name_bbox_hits`, which `score_doc` computes as a coarse band test
(`bb[1] > 250 and bb[3] < 700 and bb[0] < 500`) rather than an IoU. It cannot detect
a 5% scale error.

Measuring the five item-name rows directly out of `document.png` and comparing against
the emitted `name_bbox` gives mean absolute vertical error, in px on a 1568 axis:

| rung | 12B | 26B | 31B |
|---|---|---|---|
| 140 | 209.3 | 42.8 | 31.9 |
| 280 | 10.5 | 10.7 | 4.3 |
| 560 | **4.9** | **2.7** | **3.7** |
| 1120 | 27.4 | 10.5 | 25.4 |
| shipped range | 20.4 | — | — |

The document degrades at 1120 too, with the same U-shape and the same 560 optimum. It
passed the band test because its item rows sit at y 354–559 — near the top, where a
scale error about y = 0 produces little absolute displacement. **The aspect contrast
that motivated the whole investigation does not survive measurement.**

## 7. The loss is coordinate-space, not localization

For each run, fit two independent least-squares lines to the decoded boxes — one per
axis, two parameters each — over the 12 x-coordinates and 12 y-coordinates of the 6
matched boxes:

    pred_x = a_x·gt_x + b_x        pred_y = a_y·gt_y + b_y

A global affine can only absorb a *coordinate-space* error. Genuine mislocalization is
idiosyncratic per box and no single line fits it. Inverting the fitted transform and
re-scoring:

| run | raw IoU | after affine |
|---|---|---|
| 12B 280 | 0.883 | 0.960 |
| 12B 560 | 0.894 | 0.963 |
| 12B 1120 | 0.719 | 0.943 |
| 26B 1120 | 0.810 | 0.980 |
| 31B 1120 | 0.729 | 0.974 |
| shipped range | 0.504 | 0.927 |

**Read that column carefully.** The affine is fitted to the same six boxes it is then
evaluated on — four parameters against 24 numbers, with no held-out set. It is *not* a
prediction of what fixing a bug would yield. It says only that a single global rescale
accounts for most of the loss.

The fit-independent statistic is the residual scatter, in px:

| rung | 12B x | 12B y | 26B x | 26B y | 31B x | 31B y |
|---|---|---|---|---|---|---|
| 140 | 4.6 | 8.8 | 5.0 | 5.0 | 7.9 | 6.8 |
| 280 | 3.9 | 3.0 | 5.2 | 3.9 | 5.3 | 3.3 |
| 560 | 3.2 | 2.7 | 1.9 | 2.1 | 3.1 | 3.5 |
| 1120 | **2.3** | 8.6 | **1.3** | **2.0** | **2.3** | **2.0** |

Horizontal scatter improves ~2–4× from 140 to 1120 on all three sizes, and vertical
does the same on 26B and 31B — so precision genuinely improves with budget while
absolute accuracy collapses. **12B's vertical axis is the exception**: it rises again
at 1120 (2.7 → 8.6), consistent with the tall-grid noise noted below.

The horizontal half of this is the strong half: `a_x` was *predicted in advance* from
the padding geometry (§3) and matched to 3–4 significant figures, so it is not
curve-fitting. The vertical affine is purely fitted and should be read as a
description, not a mechanism.

This also discharges the brief's constraint 1 (non-monotonicity): the shipped range
scores worst not because it uses fewer tokens than pinned 1120, but because a 40×23
grid lands in the worst part of the vertical-error curve.

## 8. Per-size vertical decomposition

27e05deb ruled the encoder out from IoU alone. The vertical scale error at 1120
orders the same way and sharpens it:

| size | encoder | y-scale @1120 | IoU @1120 |
|---|---|---|---|
| 26B A4B | ~550M, MoE | 1.0186 | 0.810 |
| 31B | ~550M, dense | 1.0468 | 0.729 |
| 12B | none | 1.0710 | 0.719 |

The encoder-free 12B is worst and the encoder-bearing MoE is best, but 31B sits
between them — so the encoder does not partition the results, consistent with
27e05deb's conclusion. The magnitude tracks robustness, not architecture.

## 9. The cause: off-ladder patch grids are out of distribution

Two earlier framings of the driver — "patch rows", then "total patch count with a
threshold in 530–880" — were each falsified by later cells. In particular a
44×24 = 1056-patch arm scored 0.856 where the threshold story demanded degradation.
That cell was the signal, not noise (credit: Glenn's read of the sweep).

The model card says the **supported** token budgets are exactly 70/140/280/560/1120,
and HF transformers' Gemma 4 processor (verified in source this session) always
resizes to **fill** the chosen budget: `factor = sqrt(budget_px / total_px)`, floor
each axis to a multiple of 48. So the only grids the model ever saw in training are
those this procedure can emit at a supported budget:

    grid (c, r) is in-distribution  ⇔  c·r ≤ B < (c+1)·(r+1)  for some B in the ladder

Retrodicting **every** vertical-accuracy cell measured in this campaign against that
rule: 17/19 fit, and the two exceptions (34×34 = 1156, 44×24 at −4.3%) are the two
*mildest* errors in their groups, both sitting just past a ladder grid — consistent
with error growing with distance off-ladder rather than with a binary rule.

Then the rule was tested by prediction, including a cell chosen to falsify it:
45×24 = 1080 is reachable at 1120, while 43×24 = 1032 is not (no budget lies in
[1032, 1100)) — even though 1032 is *fewer* patches. All arms native-rendered,
pad-free, one budget pinned so nothing reloads:

| model | grid | patches | on ladder? | IoU | y-scale |
|---|---|---|---|---|---|
| 12B | 45×24 | 1080 | **yes** (1120) | **0.943** | 0.9938 |
| 12B | 43×24 | 1032 | no | 0.799 | 0.9497 |
| 12B | 22×12 | 264 | **yes** (280) | **0.887** | 0.9952 |
| 12B | 21×11 | 231 | no | 0.669 | 0.9384 |
| 12B | 24×13 | 312 | no | 0.726 | 0.9031 |
| 31B | 44×24 | 1056 | **yes** (1120) | **0.952** | 0.9972 |

Every prediction holds, the bands exist below 560 as well, and the error's sign
follows the direction off-ladder (below the nearest band → y-scale < 1; the two
overshooting cells, 1156 and 1170, are the only ones with y-scale > 1).

**The 31B row is the finding.** 44×24 = 2112×1152 is exactly the grid the reference
preprocessor produces for a 16:9 image at budget 1120. On it, 31B scores **0.952 —
better than its own 560 rung (0.906)** and the best bbox result of the campaign.
There is nothing wrong with 1120. What is wrong is that llama.cpp never delivers it:

- **Range mode** (`40…1120`, the shipped config) keeps an under-budget image at its
  natural rounded grid — 1920×1080 → 40×23 = 920, unreachable at any budget, mid-gap
  between the 560 band and the 1120 band → the worst cell measured (0.504).
- **Pinned mode** (`min == max`, the sweep config) overshoots through the
  `< min_pixels` branch's `ceil_by_factor` — 45×26 = 1170 > 1120, past the top of
  the ladder → 0.719.

llama.cpp's sizing (`calc_size_preserved_ratio`, a Qwen2-VL `smart_resize` clone)
differs from the Gemma 4 reference precisely here: the reference **always scales to
the budget** (up or down) and floors; llama.cpp leaves under-budget images at their
natural size. That divergence, not the padding, is the primary bug. Why the error
expresses on the vertical axis only remains unexplained — x-scale is 1.002 ± 0.002
in every pad-free cell regardless of grid — but it no longer needs explaining to fix
the problem.

Caveats: one seed, temperature 0, one content family per cell; 26B not run on the
discriminator cells; error magnitude off-ladder varies non-monotonically (the 38×21
arm at 0.589 with x also broken is an outlier in degree).

## Fine text pulls the opposite way — 1120 is not merely a worse 560

`runs/finetext-2026-08-02.log` compares only upstream's 280 against the fork's shipped
range. **560 was never probed**, and 560 is exactly what a lowered ceiling would pick.
Swept here on 12B against a regenerated compliance page (same layout and seed as
`finetext_probe.py`; Courier New rather than DejaVuSansMono, which is not on this host,
so absolute recall is *not* comparable to the 2026-08-02 log — the across-budget
ordering within this run is the result):

| arm | patches | canvas | 22px | 16px | 12px | 9px | 7px |
|---|---|---|---|---|---|---|---|
| 280 pinned | 256 | 768² | 0/4 | 0/4 | 0/4 | 0/4 | 0/4 |
| 560 pinned | 529 | 1104² | 4/4 | 0/4 | 0/4 | 0/4 | 0/4 |
| 70…560 range | 529 | 1104² | 4/4 | 0/4 | 0/4 | 0/4 | 0/4 |
| 40…1120 shipped | 1089 | 1584² | 4/4 | **4/4** | **2/4** | 0/4 | 0/4 |
| 1120 pinned | 1156 | 1632² | 4/4 | 3/4 | 1/4 | 0/4 | 0/4 |

A 1568px page is downscaled to 1104px at 560 — a 30% linear reduction, so 16px text
renders at ~11px and 12px text at ~8px. **Dropping the ceiling to 560 costs the entire
16px tier and the 12px partial.** That is the workload ADR 0003 and
[ollama#15626](https://github.com/ollama/ollama/issues/15626) raised the ceiling for.

This looked like a fundamental opposition — geometry seemed to need ≤ ~530 patches
while fine text needs as many as possible. **§9 dissolves it.** Geometry does not
need few patches; it needs an on-ladder grid. A faithful 1120 (44×24) delivers
1584px-class resolution for text *and* the best geometry measured. The conflict was
an artifact of llama.cpp never producing an on-ladder grid above 560.

## 10. Fix implemented and validated (2026-08-07)

`llama/compat/004-llama-cpp-gemma4-budget-fill.patch` implements exactly the fix §9
prescribes, following the [nemotron-dynres-patch.md](nemotron-dynres-patch.md)
precedent: `img_tool::calc_size_budget_fill` mirrors the reference formula
(`factor = sqrt(B·48²/(W·H))`, per-axis `floor(·/48)·48`, B snapped down to the
ladder, sub-70 requests clamped up to 70), gated by a new `image_budget_fill`
hparam that only `PROJECTOR_TYPE_GEMMA4V/GEMMA4UV` set, plus
`image_resize_pad = PAD_NONE`. `--image-min-tokens` becomes a no-op for gemma4.

Validated end-to-end on a native Metal build (all four `llama/compat: applied`
lines in the build log), original `scene_hd.png` and archived ground truth, so
numbers compare directly with the unpatched sweep. Delivered grids confirmed from
`prompt_eval_count`: 527 = 31×17 at 560, **1100 = 44×25** at 1120 (the reference
grid for 16:9 — note it is 44×25, not the 44×24 §9's arms happened to use; both
are reachable). An off-ladder request (`max 900`) snaps to 560 and reproduces the
560 run byte-for-byte; `min == max == 1120` no longer overshoots to 1170.

| scene IoU | 560 unpatched | 560 patched | 1120 unpatched (pin / shipped) | 1120 patched |
|---|---|---|---|---|
| 12B | 0.894 | **0.940** | 0.719 / 0.504 | 0.885 |
| 26B A4B | 0.914 | **0.961** | 0.810 / — | **0.970** |
| 31B | 0.906 | 0.934 | 0.729 / — | **0.961** |

- Every cell improves; 26B at faithful 1120 (0.970) is the best bbox result of the
  campaign, and x-scale is 0.997–1.002 everywhere (the §3 letterbox error is gone —
  that alone is the +0.03–0.05 at 560).
- On 26B and 31B, patched 1120 now **beats** patched 560 while fine text holds:
  patched `max 1120` reads the 22px + full 16px + half the 12px tiers (33×33 grid,
  same as the old shipped range), where 560 reads only 22px. 1120 strictly
  dominates — the model card's "higher budgets for OCR" guidance is restored.
- Honest exception: 12B at 1120 scores 0.885 with y-scale +5.4% — below its own 560
  (0.940). The encoder-free 12B is the least robust at the top rung in every
  measurement; on 12B, 560 remains the better bbox setting even patched.

## Remaining next steps

1. **Report both divergences upstream** (ggml-org/llama.cpp): under-budget images are
   left on unreachable grids (primary, breaks grounding), and `PAD_CEIL` letterboxing
   is applied where the reference resizes directly (secondary, 2–5% on the padded
   axis). The 45×24-vs-43×24 pair is the minimal reproduction; 004 is the fix.
2. **Revisit ADR 0007 with §10's numbers.** Its 560 ceiling was the correct
   mitigation for unpatched llama.cpp. On a build carrying 004, `min 70 / max 1120`
   matches the model card's guidance and strictly dominates on 26B/31B (geometry
   *and* fine text); on 12B, 560 still wins bbox (0.940 vs 0.885). The ADR's own
   "revisit when the error is fixed" clause anticipates exactly this — one open
   question is whether the default should be size-aware given the 12B exception.
3. **Fix `score_doc`.** `name_bbox_hits` is a band test that hid §6 for the whole
   investigation. It should score IoU against measured row geometry.
4. **Why vertical?** The mechanism behind the OOD error expressing only on the
   y-axis — and 12B's residual +5.4% at 1120 even on-ladder — remains unexplained.

## Reproduction

Experiment scripts (`padtest.py`, `axistest.py`, `ftsweep.py`, `threshold.py`,
`laddertest.py`) are not committed — they run against the fork server on `:11435`
with `gemma4:12b-it-q4_K_M` (laddertest also `31b`), think off, temperature 0, and
reuse `vision_suite.SCENE_PROMPT`. Each renders its own inputs so the only variable
between arms is geometry. The archival analysis reads `resp_*.json` /
`scores_*.json` in `vision-suite/`, which are gitignored
(`vision-suite/.gitignore:5`) and therefore local to whoever ran the sweep.
