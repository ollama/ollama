# ADR 0006: `release/0.32.1-dynres` is a maintained lineage, never merged into `main`

- **Status:** accepted 2026-08-02
- **Date:** 2026-08-02
- **Deciders:** MaxusAI fork maintainers
- **Related:** [ADR 0003](0003-vision-image-token-budget-policy.md) (lineage backports take
  only the arch-specific half), [ADR 0001](0001-nemotron-vision-dynamic-resolution.md)
  (nemotron dynres, which forces full builds),
  [AMD upgrade gate](../amd-upgrade-gate.md) (why the 0.32.1 line exists at all)

## Context

The fork maintains **two lineages**, and this was implicit until now — eight documents
reference `release/0.32.1-dynres`, but none stated how it relates to `main`. The question
was asked directly on 2026-08-02: should the release branch be merged into `main`?

The lineages differ where it matters most — the llama.cpp payload:

| | `main` | `release/0.32.1-dynres` |
|---|---|---|
| `LLAMA_CPP_VERSION` | **b10091** | **b9888** |
| ollama base | 0.32.5 line | 0.32.1 line |
| `--direct-io` (upstream `72116baf`) | present | **absent** (predates it) |
| AMD/gfx1151 deployable | **no — gated** | **yes — this is the pinned line** |

The release line is not a stale branch that fell behind. It is pinned to b9888
*deliberately*, because b10091 is the payload the [AMD upgrade gate](../amd-upgrade-gate.md)
blocks: it produced degenerate vision output on gfx1151 and was rolled back on 2026-07-31.
The gate's blockers — ollama/ollama#17459 and #17475 — are still open.

## Evidence

Measured 2026-08-02 with `origin/main` at `db51b585` and the release tip at `63dc9b27`
(14 commits ahead):

**Nothing would be gained.** `git cherry` reports **7 of the 14 commits already have
patch-equivalent counterparts on `main`**. Every feature the branch carries is already
there — the qwen35/qwen35moe image-token floor, `llama/compat/002-llama-cpp-nemotron-dynres.patch`
(byte-identical blob), per-model `kv_cache_type`, the `qwen35` and `nemotron3nano` parsers,
and ADR 0003. Not one file on the branch is absent from `main`.

**A merge would cost, not gain.** It produces **8 conflicts**, five of them `add/add` on
documents authored independently in both lineages:

```
CONFLICT (add/add): adr/0002 · adr/0003 · generate-think-format-empty-response
                    spec/structured-output-with-thinking · spec/vision-image-token-budgets
CONFLICT (content): .gitignore · llm/llama_server.go · server/routes_generate_test.go
```

**`main` is far ahead on shared code**, so a merge risks dragging 0.32.1-adapted code
forward over newer work:

| file | lines only on release | lines only on `main` |
|---|---|---|
| `llm/llama_server.go` | 25 | **332** |
| `server/routes.go` | 27 | 20 |
| `api/types.go` | 1 | 3 |

Those 25 lines are not stale — they are **0.32.1-specific adaptations** (e.g. `258534eb`
adapts per-model `kv_cache_type` from `main`'s `fd93fb85`+`31c8ea79`). They are correct for
b9888 and wrong for b10091. That is precisely why the lineage exists.

## Decision

**`release/0.32.1-dynres` is a long-lived release lineage. It is never merged into `main`.**

Normative rules:

1. **Changes flow one way: `main` → release, by cherry-pick.** Never open a PR from the
   release lineage to `main`. Backports adapt to b9888 and take only the applicable half —
   see [ADR 0003](0003-vision-image-token-budget-policy.md).
2. **The release lineage is the deployable line for AMD/gfx1151** while the
   [AMD gate](../amd-upgrade-gate.md) holds. `main` is not deployable there.
3. **Both lineages carry their own `llama/compat/002-*.patch`.** They may be byte-identical
   at a given moment; that is coincidence, not coupling. Do not assume one implies the other.
4. **Divergence is expected and is not drift.** Do not "catch the branch up" to `main` by
   merging. Only cherry-pick specific, adapted fixes.
5. **Retire, don't merge.** When the AMD gate lifts and gfx1151 moves onto a b10091-or-later
   payload, the lineage is *retired* — the branch is archived, not merged.

## Build portability (asked 2026-08-02: "is it suitable for CUDA and ROCm at the same time?")

**The source is backend-agnostic; the image is not.**

- Patch 002 touches only `tools/mtmd/clip.cpp`, `models/nemotron-v2-vl.cpp` and `mtmd.cpp` —
  CPU-side image preprocessing. **Zero** CUDA/HIP/ROCm references, and zero backend-specific
  lines in the branch's Go diff. `apply-patch.cmake` runs at configure time for every preset,
  so CUDA and ROCm builds both receive it from the same tree.
- The Dockerfile builds **per-backend presets** (`rocm_v7_2_linux`, `llama_cuda_v12_linux`,
  `llama_cuda_v13_linux`, `vulkan`, jetpack…), selected by `FLAVOR`. One image is one flavor.
  A CUDA host needs its own build from the same commit.

⚠️ **The overlay shortcut is unavailable on this lineage.** Because 002 is C++ compiled into
`llama-server`/`libmtmd`, the Go-binary-only overlay cannot deliver it, and the
payload-pristine proof goes non-empty *correctly* (a true positive — see
[ADR 0001](0001-nemotron-vision-dynamic-resolution.md)). Any deployment carrying 002 requires
the **full Dockerfile build** — roughly 45–70 GB of scratch per flavor, against ~7 GB for an
overlay. `Dockerfile.overlay`, `make proof` and `make gate` in the deployment repo apply to
the *overlay* line, not to a 002-carrying tree.

## Consequences

- `main` and the release line diverge permanently until the gate lifts. That is intended.
- Every fix wanted on AMD must be cherry-picked and adapted; it will not arrive by merging.
- Duplicate documents will keep appearing in both lineages (the five `add/add` conflicts).
  This is the visible cost of the split and is accepted — do not resolve it by merging.
- Anyone running `git merge release/0.32.1-dynres` on `main` should stop and read this ADR;
  the eight conflicts are a signal, not an obstacle to push through.
- Retirement, when it comes, is a deletion-and-archive, leaving `main` untouched.
