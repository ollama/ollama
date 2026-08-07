# ADR 0003: Per-architecture vision budgets are opt-in and empirically verified; lineage backports take only the arch-specific half

- **Status:** accepted 2026-08-02, with the qwen gap closed on the release lineage
  (`a4788474` on `release/0.32.1-dynres`, backporting the qwen half of `87cf1100`).
- **Date:** 2026-08-02
- **Deciders:** MaxusAI fork maintainers
- **Related:** [ADR 0001](0001-nemotron-vision-dynamic-resolution.md) (nemotron
  dynamic resolution), [SPEC: vision image-token budgets](../spec/vision-image-token-budgets.md)
- **Vendor reference:** Gemma 4 model card —
  <https://ai.google.dev/gemma/docs/core/model_card_4> (supported visual token
  budgets **70 / 140 / 280 / 560 / 1120**; 12B is encoder-free, 26B A4B and 31B
  carry a ~550M vision encoder)

## Context

`visionServerArgs()` switches on `modelArch` to decide which `--image-{min,max}-tokens`
flags a llama-server launch receives. Arches absent from that switch get no flags and
run at whatever their projector defaults to. The switch is shared by both maintained
lineages, but the llama.cpp payloads that *consume* the flags are versioned
independently: `main` tracks b10091 and `release/0.32.1-dynres` tracks b9888, and each
carries its own copy of `llama/compat/002-llama-cpp-nemotron-dynres.patch`. A lineage
can therefore gain or lose a consumer without the Go switch changing at all — `main`
itself had a pristine `llama/` between `5ad093b0` and `2487dd56`.

Two facts forced this decision:

1. **A silent omission.** `qwen35`/`qwen35moe` (qwen3.6) were the only vision arches
   in `compatClipArches` never listed in the switch, so they fell through to
   `default: return nil`. They load as `PROJECTOR_TYPE_QWEN3VL`, the branch that
   explicitly warns "requires at minimum 1024 image tokens to function correctly on
   grounding tasks". `main` fixed this in `87cf1100` (2026-07-30); the release lineage
   never received it, so production qwen3.6 ran with no floor. Measured on b9888 + 002
   (baseline 14 tokens): a 224² image cost **51** visual tokens, 448² cost 198, 896²
   cost 786 — all far under the threshold the projector itself asks for.

2. **Backporting the fix wholesale would have broken the branch.** `87cf1100` also
   adds a guard asserting `nemotron_h_omni` receives **no** budget flags. That was
   correct on `main` *as it stood on 2026-07-30*, whose `llama/` was then pristine, so
   `PROJECTOR_TYPE_NEMOTRON_V2_VL` never called `set_limit_image_tokens()` and the
   flags would have advertised a dead knob. It was already false on the release
   lineage, where compat/002 makes exactly those flags live and the branch's own tests
   assert they are passed — and it is false on `main` too since `2487dd56` restored
   compat/002 there. The assertion was payload-specific and short-lived; the arch
   entry it travelled with was not.

## Decision

1. **Budgets stay per-arch and opt-in.** No global "vision budget" knob. An arch gets
   flags only when its projector demonstrably consumes them.
2. **Adding an arch requires an empirical check**, not a code reading: the fingerprint
   procedure in the SPEC (§3) must show the token count moving in the predicted
   direction, plus a re-measure at corpus sizes to bound the blast radius.
3. **`qwen35`/`qwen35moe` get the 1024 floor on every lineage.** Verified on b9888:
   51/198/786 → 1026/1026/1026, while 1920×1080 (2042) and 1568×1568 (2403) stay above
   the floor and are untouched.
4. **Dynamic resolution stays nemotron-only.** compat/002 is gated on
   `PROJECTOR_TYPE_NEMOTRON_V2_VL`; gemma4 and qwen keep their own separate
   mechanisms. Nothing about ADR 0001 generalises to other arches.
5. **Cross-lineage backports carry arch entries, not projector assertions.** When a
   commit both adds an arch and asserts what a *different* arch's projector does, take
   only the arch-specific half and record the divergence in the commit message.

## Alternatives considered

- **Cherry-pick `87cf1100` unchanged.** Rejected: its nemotron guard contradicts the
  release lineage's patched projector and its existing tests. Taking the qwen half
  keeps both branches honest about their own payloads.
- **Pass budget flags to every vision arch.** Rejected: flags that a projector ignores
  turn `ImageMinTokens`/`ImageMaxTokens` into knobs that silently do nothing, which is
  precisely the confusion
  [vision-token-budgets-by-arch.md](../vision-token-budgets-by-arch.md) was written to
  dispel. The single deliberate exception (nemotron on unpatched payloads) is
  documented in the arch table.
- **Leave the release lineage without the floor.** Rejected once measured: this is not
  a tuning preference but a grounding-quality defect on sub-1MP inputs, on the lineage
  that actually serves production.
- **Make the budget a per-request option for qwen too.** Rejected for now: the floor
  is a correctness threshold the projector itself names, not a tunable, and Runner
  options force a reload (SPEC B3).

## Consequences

- Vision budgets are launch-time flags, so this policy applies identically to
  `/api/chat`, `/api/generate` and the `/v1` endpoints; measurements taken through one
  endpoint transfer to the others (SPEC B2).
- Sub-1MP images on qwen3.6 now cost ~1026 visual tokens instead of 51–786: better
  grounding, more context consumed. Corpus-sized inputs are unchanged, so every vision
  measurement previously recorded on this lineage still stands.
- `TestVisionServerArgs` expectations are a function of the lineage's payload, not a
  constant. They happen to agree today — both lineages carry compat/002 and assert
  nemotron's flags — but any change to a lineage's `llama/` must be accompanied by a
  re-check, in both directions.
- Both maintained lineages now carry all three mechanisms (qwen floor, gemma4 budget,
  nemotron dynres) and differ only in llama.cpp pin: `main` at b10091 since compat/002
  was ported forward in `2487dd56`, `release/0.32.1-dynres` at b9888. Which payload may
  ship to the gfx1151 host remains governed by
  [amd-upgrade-gate.md](../amd-upgrade-gate.md), independently of this policy.

## Validation addendum (2026-08-02)

The gemma4 280→1120 budget was re-validated against genuine upstream with the
dense fine-text A/B: upstream reads nothing below 16px on a 1568² page and
confabulates code-like strings; the budgeted build transcribes correctly to
7–9px. Cost: ~0.12 mean IoU on synthetic shape boxes. Policy stands. Details:
[vision-campaign-2026-08-02.md](../vision-campaign-2026-08-02.md) §4.

## Addendum — vendor ladder and size dependence (2026-08-07)

Two things learned after the decision, neither of which reverses it, but both of
which change how the IoU cost above should be read.

**1. The budget is a vendor-documented ladder.** Google's model card
(<https://ai.google.dev/gemma/docs/core/model_card_4>) states: *"The supported
token budgets are: 70, 140, 280, 560, and 1120."* So the 280→1120 raise moves
from llama.cpp's default rung to the **documented maximum** — it is a ladder
step, not an invented ceiling. This strengthens decision 1: the knob has a
vendor-defined domain.

Two off-ladder facts follow, recorded rather than acted on here:

- `gemma4ImageTokenBudget()` accepts any integer; it does not clamp to the ladder.
  Off-ladder values are undocumented territory.
- `api.DefaultImageMinTokens = 40` sits **below the documented floor of 70**. It
  derives from llama.cpp's `set_limit_image_tokens(40, 280)`, not the model card.
  As a floor it binds only on very small images, so the practical impact is
  narrow — but it is off-ladder.

**2. The IoU cost depends on the REGIME, not on model size — and 560 beats 1120.**

First, the payload is exonerated. Measured 2026-08-07 on `gemma4:12b-it-q4_K_M`
(Metal, M5 Max), scene test, with a budget-matched control:

| arm | budget | `prompt_eval_count` | `bbox_mean_iou` | 14px serial |
|---|---|---|---|---|
| stock (b10242) | llama.cpp default | 848 | 0.883 | ✗ |
| fork (b10091) | 40 / 1120 (range) | 1504 | **0.504** | ✓ |
| control (b10091) | 40 / 280 (range) | 848 | 0.883 | ✗ |

The control reproduces stock **exactly**, so the 151-build llama.cpp gap between
the lineages contributes nothing measurable and the whole delta is the budget.

Then the ladder sweep (`run_budget_sweep.sh`, `min == max` pinned per rung):

| budget | image tok | 12B IoU | 31B IoU | 12B serial | 31B serial |
|---|---|---|---|---|---|
| 70 | ~82 | 0.000 | 0.000 | ✗ | ✗ |
| 140 | ~136 | 0.780 | 0.830 | ✗ | ✗ |
| 280 | 280 | 0.883 | 0.902 | ✗ | ✓ |
| **560** | ~543 | **0.894** | **0.906** | ✓ | ✓ |
| 1120 | ~1186 | 0.719 | 0.729 | ✓ | ✓ |

Three corrections to the reading above follow:

- **Not size-dependent.** Both sizes trace the same curve — peak at 560, collapse
  at 1120 — and the 280→1120 cost is +0.164 (12B) vs +0.173 (31B). An earlier
  draft of this addendum attributed the ~0.12-vs-0.379 gap to 12B being
  encoder-free. That was wrong.
- **Regime-dependent instead.** ~0.379 came from the *range* `40…1120`, where the
  projector selected ~936 image tokens; the *pinned* 1120 (~1186 tokens) costs only
  ~0.164. **More tokens produced better geometry**, so the effect is not monotonic
  in token count and "higher resolution costs localisation" is the wrong model of
  it. The shipped default's range happens to select a bad intermediate grid.
- **Not in the vision encoder.** 12B is encoder-free and 31B carries ~550M, yet both
  collapse identically at 1120. The defect therefore sits in the shared path — grid
  selection during preprocessing, or the norm-1000 `box_2d`/yxyx decode — which
  rules out a large part of the search space.

**560 dominates 1120 on both sizes**: higher IoU *and* the same fine-text recall.
The bottom two rungs are unusable for this workload (70 scores zero on every
metric; 140 loses labels and line items).

**Decisions 1–5 stand unchanged**, but decision 2's "empirical check" standard now
argues against the value in decision-adjacent defaults: on this evidence
`api.DefaultImageMaxTokens = 1120` is the wrong rung, and 560 is the candidate.
Changing a shipped default is out of scope for an addendum — raise it as its own
decision. Until then, do not quote "the budget raise costs ~0.12 IoU" without
stating the regime and the rung.
