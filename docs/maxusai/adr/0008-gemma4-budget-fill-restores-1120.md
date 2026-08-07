# ADR 0008: the budget-fill payload restores gemma4's default ceiling to 1120

- **Status:** accepted, 2026-08-07
- **Supersedes:** [ADR 0007](0007-gemma4-default-budget-560.md)'s *default
  values* (70…560). Its mechanism analysis stands; its mitigation is no longer
  needed on a payload carrying
  [`004-llama-cpp-gemma4-budget-fill.patch`](../../../llama/compat/004-llama-cpp-gemma4-budget-fill.patch).
- **Depends on:** the 004 patch being in the compiled payload. The fork's build
  applies it unconditionally (`llama/compat` glob); overlay builds are already
  forbidden while compat patches are in-tree
  ([apple-silicon-build spec](../spec/apple-silicon-build.md)).

## Context

ADR 0007 lowered the ceiling from 1120 to 560 as a mitigation for a coordinate
defect that grew with the visual token budget. The investigation
([findings](../gemma4-bbox-investigation-findings.md), §9–§10) then isolated the
actual cause: llama.cpp left under-budget images on patch grids the model never
saw in training ("off-ladder" — not satisfying `c·r ≤ B < (c+1)·(r+1)` for a
supported budget B ∈ {70, 140, 280, 560, 1120}), and pinned budgets overshot the
ladder entirely. Off-ladder grids break `box_2d` vertical grounding; the ladder
itself was never the problem.

004 replaces the sizing with the reference behaviour (snap the ceiling down to
the ladder, scale — up or down — to fill it, floor each axis to 48, no
letterbox). Validated end-to-end on the original suite (findings §10):

| scene IoU | 560 | 1120 |
|---|---|---|
| 12B | 0.940 | 0.885 |
| 26B A4B | 0.961 | **0.970** |
| 31B | 0.934 | **0.961–0.963** |

Fine text at 1120 reads the 22px, 16px and half the 12px tiers; 560 reads only
22px. On 26B/31B, 1120 now strictly dominates 560. Portrait recovers (0.406 →
0.905), off-ladder requests snap down, `min == max` no longer overshoots, and
nemotron (002's shared dyn_size path) is byte-identical — no regression.

## Decision

- `api.DefaultImageMinTokens = 70`, `api.DefaultImageMaxTokens = 1120` — the
  model card's floor and ceiling, matching its guidance to use higher budgets
  for OCR/document work and lower ones per request where speed matters.
- The Go token estimator (`ImageTokensForSize`) mirrors 004's
  `calc_size_budget_fill` (ladder snap + sqrt fill + per-axis floor), and
  `MaxImageTokens` snaps the requested ceiling, so scheduling charges what
  llama-server actually delivers.

## Alternatives considered

- **Keep 560.** Rejected: it pays a permanent fine-text cost (blind below
  ~22px on a 1568² page) to mitigate a defect that is now fixed, and it is
  dominated on 26B/31B.
- **Size-aware default (560 for 12B, 1120 otherwise).** The encoder-free 12B
  still prefers 560 for pure bbox work (0.940 vs 0.885; a residual +5.4%
  vertical error at 1120 survives 004 on that size only). Rejected for now:
  defaults should not fork on model size without a second dataset confirming
  the 12B exception; callers can pin `image_max_tokens: 560` per request.

## Consequences

- **Small images now cost the full budget.** Budget-fill upscales: a 640×480
  image is 1066 tokens at the default ceiling where it used to be 132. That is
  the reference behaviour and what makes grounding accurate, but prefill cost
  rises accordingly; latency-sensitive callers should request a lower rung.
- `image_min_tokens` is a no-op for gemma4 on a 004 payload (kept as a flag for
  unpatched payloads).
- These defaults are only correct **with** 004. A Go binary at 70/1120 driving
  an unpatched payload reproduces the shipped-range worst case ADR 0007 fled
  (IoU 0.504 at 920 tokens); within this fork that combination cannot be built.
- Warmup exercises the snapped ceiling rather than 256 tokens: heavier load,
  fail-fast on memory.
