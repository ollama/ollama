# Investigation brief: gemma4 bbox geometry degrades with visual token budget

MaxusAI-fork investigation brief (fork-only). Opened 2026-08-07 from the Apple
Silicon benchmark campaign.

> **Status: CLOSED 2026-08-07 — the hypothesis below is refuted. See
> [gemma4-bbox-investigation-findings.md](gemma4-bbox-investigation-findings.md).**
>
> Kept as the record of how the question was framed. Three of its claims did not
> survive measurement, and the corrections matter to anyone reading it:
>
> - **The aspect hypothesis is geometrically void.** Normalized coordinates are
>   invariant to anisotropic resize, so per-dimension rounding cannot move a
>   norm-1000 box. That is why AR-error magnitude never rank-ordered IoU.
> - **Constraint 3 is wrong.** The 1:1 document is *not* immune; `name_bbox_hits`
>   is a band test that cannot see a 5% scale error. A square, unpadded, unresampled
>   image still loses 9.4% vertically.
> - **Padding is real but is not the cause.** `PAD_CEIL` letterboxing is confirmed in
>   source and quantified (it explains the horizontal axis exactly), but a pad-free
>   control reproduces the collapse.
>
> The actual defect is a vertical-only coordinate error that grows with patch rows.

## The observation

Raising gemma4's visual token budget improves fine-text recall and *degrades*
bounding-box geometry. Measured on an M5 Max, native Metal build, llama.cpp b10091,
`vision-suite` scene test, think off, temperature 0, `min == max` pinned per rung:

| budget | 12B (encoder-free) | 26B A4B (~550M, MoE) | 31B (~550M, dense) |
|---|---|---|---|
| 70 | 0.000 | 0.000 | 0.000 |
| 140 | 0.780 | 0.814 | 0.830 |
| 280 | 0.883 | 0.885 | 0.902 |
| **560** | **0.894** | **0.914** | **0.906** |
| 1120 | 0.719 | 0.810 | 0.729 |
| **280→1120 cost** | **+0.164** | **+0.075** | **+0.173** |

The 14px serial is found from 280 on 26B and 31B, and only from 560 on 12B.

Four properties any explanation must account for:

1. **Not monotonic in token count.** The shipped *range* `40…1120` selects ~936 image
   tokens and scores IoU 0.504; *pinned* 1120 uses ~1170 and scores 0.719. More
   tokens, better geometry.
2. **Not in the vision encoder — ruled out by 26B.** 12B is encoder-free; 26B A4B and
   31B share a ~550M encoder. All three peak at 560 and collapse at 1120. Critically,
   the two *encoder-bearing* sizes differ **most** in collapse magnitude (+0.075 vs
   +0.173) while the encoder-free 12B sits between them on peak IoU. If the encoder
   drove this, 26B and 31B would pair off against 12B. They do not. The **shape** is
   universal; the **magnitude** varies non-systematically.
3. **Aspect-correlated.** The 16:9 scene degrades. The 1:1 document does not
   (`name_bbox_hits` 4→4 across the same rungs).
4. **Reproduces exactly.** A budget-matched control at 280 reproduces stock's 0.883
   to three decimals, so it is not payload drift (b10091 vs b10242 contributes
   nothing measurable).

## Hypothesis: per-dimension rounding breaks the coordinate mapping

`img_tool::calc_size_preserved_ratio` ([`tools/mtmd/mtmd-image.cpp:161`], the
4-argument overload) rounds width and height **independently** to multiples of
`align_size`:

```cpp
int h_bar = std::max(align_size, round_by_factor(height));
int w_bar = std::max(align_size, round_by_factor(width));

if (h_bar * w_bar > max_pixels) {
    const auto beta = std::sqrt(float(height * width) / max_pixels);
    h_bar = std::max(align_size, floor_by_factor(height / beta));
    w_bar = std::max(align_size, floor_by_factor(width  / beta));
} else if (h_bar * w_bar < min_pixels) {
    const auto beta = std::sqrt(float(min_pixels) / (height * width));
    h_bar = ceil_by_factor(height * beta);
    w_bar = ceil_by_factor(width * beta);
}
```

Independent rounding does **not** preserve the ratio. For gemma4,
`align = patch_size * n_merge = 16 * 3 = 48` — matching Google's documented
"both dimensions divisible by 48".

Predicted aspect error (replicating the function exactly, `min == max` pinned):

| budget | scene 1920×1080 → | AR err | document 1568² → | AR err | chart 1280×960 → | AR err |
|---|---|---|---|---|---|---|
| 70 | 528×288 | +3.13% | 384×384 | 0.00% | 432×336 | −3.57% |
| 140 | 720×384 | +5.47% | 528×528 | 0.00% | 624×480 | −2.50% |
| 280 | 1056×576 | +3.13% | 768×768 | 0.00% | 912×672 | +1.79% |
| 560 | 1488×816 | +2.57% | 1104×1104 | 0.00% | 1344×1008 | 0.00% |
| 1120 | 2160×1248 | **−2.64%** | 1632×1632 | 0.00% | 1872×1392 | +0.86% |

**A square image is distorted 0.00% at every rung.** That is the cleanest support
for the hypothesis: the one test that does not degrade is the one that suffers no
aspect error.

The proposed failure path: the model emits norm-1000 `box_2d` coordinates relative
to the **distorted** canvas; the scorer maps them back onto the **original**
1920×1080. A 2–5% axis-dependent scale error moves box edges by 20–50px on a 1080px
axis, which is enough to cost IoU without affecting label or colour recall — matching
the observed pattern exactly (labels/colours stay 6/6 while IoU halves).

### What the hypothesis does NOT yet explain

Be honest about this before building on it:

- AR-error **magnitude** does not rank-order IoU. 560 (+2.57%) scores 0.894 and 1120
  (−2.64%) scores 0.719 — similar magnitudes, very different IoU. Sign, or something
  correlated with it, matters more than magnitude.
- The shipped range `40…1120` leaves the canvas at 1920×1104 (−2.18% AR) and scores
  **0.504** — worse than pinned 1120's −2.64% at 0.719. Aspect error alone therefore
  cannot be the whole story.
- Padding is a confound not yet isolated: gemma4 does **not** set `image_resize_pad`
  in its `PROJECTOR_TYPE_GEMMA4V` branch ([`clip.cpp:1445`]), so it inherits the
  `clip-model.h:65` default `PAD_CEIL`. Whether padding is applied, and whether the
  pad offset is accounted for in coordinate space, is unverified.

## Methodological trap found while measuring

**Pinning `min == max` can UPSCALE the image.** At budget 1120 the scene's rounded
size (1920×1104 = 2,119,680 px) is *below* `min_pixels` (1120 × 48² = 2,580,480), so
the `< min_pixels` branch fires and `ceil_by_factor` scales it **up** to 2160×1248 —
beyond native resolution. The shipped default (`min = 40`) never reaches that branch.

So `run_budget_sweep.sh`'s pinned rungs do not reproduce production behaviour at the
top of the ladder. Any follow-up must sweep **both** regimes and label them. This
also partly explains the range-vs-pinned discrepancy noted above.

## What to do next

1. **Instrument the actual target size.** Log `target_size` from
   `mtmd_image_preprocessor_dyn_size::preprocess` and confirm the predicted table
   against a real run. Everything above is a replication of the arithmetic, not an
   observation of it.
2. **Isolate aspect from content.** Build `gen_aspect.py`: the same six labelled
   shapes rendered at 1:1, 4:3, 16:9 and 21:9 with geometry scaled *proportionally*
   (objects occupy the same fraction of canvas in every variant) and matching ground
   truth. Sweep budget × aspect. Naive letterboxing is not adequate — it confounds
   aspect with object scale.
3. **Test the coordinate-mapping theory directly.** Feed a pre-distorted image whose
   dimensions are already multiples of 48 (e.g. 1920×1056, AR error 0%). If IoU
   recovers at 1120, the mapping is confirmed as the mechanism.
4. **Check padding.** Determine whether `PAD_CEIL` pads gemma4 images, and if so
   whether the pad offset is corrected for when boxes are decoded. Compare against
   the nemotron finding in
   [nemotron-dynres-patch.md](nemotron-dynres-patch.md), where the unpatched
   512-letterbox payload carried the padding offset in its y-axis — a known precedent
   for exactly this class of bug.
5. **Compare with the reference implementation.** llama.cpp's comment calls this
   "smart_resize in transformers code". Diff against HF transformers' Gemma 4 image
   processor and Google's own preprocessing to see whether upstream rounds
   per-dimension too, or preserves the ratio and pads. If llama.cpp diverges, this is
   an upstream bug worth reporting rather than a fork concern.
6. **Re-decide the default.** `api.DefaultImageMaxTokens = 1120` is currently the
   worst usable rung for bbox work on both sizes. 560 dominates it — better IoU,
   identical fine-text recall. If the mapping bug is fixed, re-measure before
   changing; if it is not, 560 is the better default and superseding ADR 0003 should
   be considered on its own merits.

## References

- Gemma 4 model card — <https://ai.google.dev/gemma/docs/core/model_card_4>
  (budget ladder 70/140/280/560/1120; 12B encoder-free, 26B A4B and 31B ~550M encoder)
- vLLM Gemma 4 recipe — <https://docs.vllm.ai/projects/recipes/en/stable/Google/Gemma4.html>
  (`mm_processor_kwargs` key `max_soft_tokens`, same ladder)
- Upstream issue for the knob — <https://github.com/ollama/ollama/issues/15626>
- [ADR 0003](adr/0003-vision-image-token-budget-policy.md) — budget policy, and the
  2026-08-07 addendum carrying these numbers
- [SPEC §2.1](spec/vision-image-token-budgets.md) — the normative ladder
- [vision-token-budget-measurements.md](vision-token-budget-measurements.md) — full sweep
- Harness: `vision-suite/run_budget_sweep.sh`, `vision-suite/run_compare.sh`

## Not covered

- **26B A4B has since been swept (2026-08-07)** and is included above. It was the
  discriminating cell and it is what rules the encoder out. No longer a gap.
- Only `q4_K_M`. Quality numbers are not assumed to transfer to `nvfp4`.
- One image per aspect, one seed. Single-sample scores on 5–6 item tasks are coarse;
  the document `name_bbox` column moves ±1 between adjacent rungs and should not be
  read as a trend.
