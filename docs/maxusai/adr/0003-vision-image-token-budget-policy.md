# ADR 0003: Per-architecture vision budgets are opt-in and empirically verified; lineage backports take only the arch-specific half

- **Status:** accepted 2026-08-02, with the qwen gap closed on the release lineage
  (`a4788474` on `release/0.32.1-dynres`, backporting the qwen half of `87cf1100`).
- **Date:** 2026-08-02
- **Deciders:** MaxusAI fork maintainers
- **Related:** [ADR 0001](0001-nemotron-vision-dynamic-resolution.md) (nemotron
  dynamic resolution), [SPEC: vision image-token budgets](../spec/vision-image-token-budgets.md)

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
