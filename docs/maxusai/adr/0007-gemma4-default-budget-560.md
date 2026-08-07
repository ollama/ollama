# ADR 0007: gemma4's default vision budget drops to 70…560, as a mitigation for a coordinate bug

- **Status:** superseded by [ADR 0008](0008-gemma4-budget-fill-restores-1120.md)
  (2026-08-07, same day: the 004 budget-fill patch fixed the defect this ADR
  mitigated; defaults returned to 70/1120). Accepted 2026-08-07.
- **Date:** 2026-08-07
- **Deciders:** MaxusAI fork maintainers
- **Supersedes:** the *default values* set under
  [ADR 0003](0003-vision-image-token-budget-policy.md). ADR 0003's **policy**
  (budgets are per-arch, opt-in, and empirically verified) is unchanged and still
  governs — this decision is an exercise of it, not a reversal.
- **Related:** [SPEC §2.1](../spec/vision-image-token-budgets.md),
  [findings](../gemma4-bbox-investigation-findings.md),
  [measurements](../vision-token-budget-measurements.md)
- **Vendor reference:** <https://ai.google.dev/gemma/docs/core/model_card_4>

## Context

ADR 0003 raised gemma4's ceiling from llama.cpp's 280 to **1120**, the vendor
maximum, on fine-text evidence: upstream reads nothing below 16px on a 1568² page
while the budgeted build transcribes to 7–9px. Its validation addendum recorded the
cost as "~0.12 mean IoU on synthetic shape boxes" and let the policy stand.

Two things have since been measured that the addendum could not have known.

**1. 560 is the optimum, not 1120 — on every size and both tests.** Sweeping
Google's documented ladder (70/140/280/560/1120) with `min == max` pinned,
`bbox_mean_iou` on the scene test:

| budget | 12B (encoder-free) | 26B A4B (~550M, MoE) | 31B (~550M, dense) |
|---|---|---|---|
| 70 | 0.000 | 0.000 | 0.000 |
| 140 | 0.780 | 0.814 | 0.830 |
| 280 | 0.883 | 0.885 | 0.902 |
| **560** | **0.894** | **0.914** | **0.906** |
| 1120 | 0.719 | 0.810 | 0.729 |

The document test agrees once measured properly — its `name_bbox_hits` is a coarse
band test that hid the effect; direct vertical error in px on a 1568 axis also
bottoms out at 560 (12B 4.9, 26B 2.7, 31B 3.7).

**2. The shipped configuration was the worst one measured.** The default was a
*range*, `40…1120`, not a pinned rung. On a 1920×1080 input that range selects a
40×23 patch grid and scores **IoU 0.504** — below every pinned rung including 1120
(0.719). The production default was worse than any budget an operator was likely to
set by hand.

**Crucially, this is not a perception limit.** Fitting and removing a single
per-axis affine per run lifts 1120 to the *best* corrected IoU on 26B (0.980) and
31B (0.974). More tokens localise better; a vertical coordinate error in llama.cpp
masks it. The cause of that error is still open — aspect distortion, letterbox
padding, resampling, the vision encoder and `resize_position_embeddings` are all
ruled out. It tracks patch **rows**, and 560 keeps a 1920×1080 input on a 31×17
grid, in the low-error part of that curve, where the scene's measured y-scale is
0.996.

Separately, `DefaultImageMinTokens = 40` is **below the model card's documented
floor of 70**. It came from llama.cpp's `set_limit_image_tokens(40, 280)`, not from
Google. As a floor it binds only on small inputs, so this is a correctness tidy-up
rather than a fix.

## Decision

1. **`api.DefaultImageMaxTokens` = 560** (was 1120).
2. **`api.DefaultImageMinTokens` = 70** (was 40).
3. **Both values must be rungs on the vendor ladder.** 70/140/280/560/1120.
   `gemma4ImageTokenBudget()` still does not *enforce* the ladder — off-ladder
   values remain accepted for per-request tuning — but the shipped defaults are
   ladder values, and changing them to a non-rung requires superseding this ADR.
4. **This is a mitigation, not a finding that the model reads 560 better.** It is
   contingent on the llama.cpp vertical coordinate error. When that is fixed, both
   values MUST be re-measured — the corrected-IoU evidence says 1120 would then
   win.
5. **1120 remains reachable per request.** Nothing is removed from the API; callers
   who want the vendor maximum set `image_max_tokens` explicitly.

## Alternatives considered

- **Leave 1120 and wait for the coordinate fix.** Rejected: the shipped *range*
  measured worst of all (0.504), so waiting means knowingly shipping the worst
  configuration for an unbounded period. The fix has no owner or date.
- **Pin `min == max == 560`.** Rejected: pinning sends inputs smaller than the
  budget through llama.cpp's `< min_pixels` branch, which **upscales beyond native
  resolution** (a 1920×1080 input becomes 2160×1248 at 1120). A floor-and-ceiling
  range avoids that; a pinned rung invites it.
- **Raise the floor to 560 instead of lowering the ceiling.** Rejected on
  measurement: on a 1920×1080 input the projector already selects ~936 tokens under
  `40…1120`, so a 560 floor never binds and the bad 40×23 grid is unchanged.
- **Drop the ceiling to 280** (llama.cpp's own default). Rejected: 280 loses the
  fine-text recall that justified ADR 0003 in the first place — on 12B the 14px
  serial is only found from 560 up — and scores below 560 on bbox at every size.
- **Fix the coordinate error first, then re-decide.** Preferred in principle, but
  the cause is unidentified after a full investigation and may sit upstream. This
  decision is explicitly reversible when that lands.

## Consequences

- **Images that hit the ceiling get roughly half as many visual tokens.**
  1920×1080 goes 922 → 529, 1568² 1091 → 531, 3000×2000 1082 → 534. Images whose
  natural grid already fits under 560 are **unchanged** (640×480 stays 132, 896²
  stays 363), so this is not a blanket reduction.
- **Fine-text recall is retained.** The 14px serial that motivated ADR 0003 is
  still found at 560 on all three sizes.
- **Context and latency improve.** Roughly half the image tokens per large image
  means more conversation fits and prefill is cheaper. `TestChatPrompt`'s gemma4
  truncation case had to have its context limit halved to keep truncating at all.
- **`nemotronImageTokenBudget()` keeps working**, because its "caller left the
  gemma4 defaults alone" sentinel compares against `api.Default*` symbolically
  rather than against literal 40/1120. The sentinel's *values* move with this ADR;
  its behaviour does not.
- **Every gemma4 vision measurement recorded before 2026-08-07 was taken at
  40…1120** and does not describe the shipped default any more. Re-measure before
  comparing across that boundary.
- **Test coverage keeps the old configuration reachable.** `TestImageTokensForSize`
  retains explicit `40/1120` rows so the pre-ADR behaviour stays pinned and the
  delta stays visible.
