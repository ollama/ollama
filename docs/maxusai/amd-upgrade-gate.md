# Upgrade gate for AMD / gfx1151: 0.32.5 is blocked

MaxusAI-fork reference (fork-only; does not exist upstream). Written 2026-07-31 after
`0.32.5-gemma4budget-4259c191` produced degenerate output on the gfx1151 host and was rolled
back to `0.32.1-gemma4budget-85ebcb79`.

> **The one thing to take away:** do **not** upgrade the AMD/gfx1151 deployment past
> **0.32.1** until [#17459](https://github.com/ollama/ollama/issues/17459) and
> [#17475](https://github.com/ollama/ollama/issues/17475) are closed **and**
> `--direct-io` is confirmed safe on ROCm iGPUs. The pinned image is
> `maxusai-ollama:0.32.1-rocm-gemma4budget`. This gate is about the **upstream payload**,
> not about anything the fork changed.

## Status

| | |
|---|---|
| Pinned image | `maxusai-ollama:0.32.1-rocm-gemma4budget` |
| Pinned version | `0.32.1-gemma4budget-85ebcb79` |
| Blocked target | `0.32.5-gemma4budget-4259c191` (built, verified, **rolled back**) |
| Host | Ryzen AI Max+ 395 / Radeon 8060S, **gfx1151**, ROCm, Linux |
| Gate lifts when | #17459 **and** #17475 closed, **and** `--direct-io` re-validated on ROCm iGPU |

## What was observed

Vision requests driven from `10.8.0.6` against `10.8.0.4`, `qwen3.6:35b-a3b-q4_k_m`
(`qwen35moe`), `/api/chat`:

| attempt | build | result |
|---|---|---|
| 1 | `0.32.5-…-4259c191` | 2 ok / 8 → then **6 degenerate** |
| 2 | `0.32.5-…-4259c191` | 0 ok → **degenerate inside row 1** |
| 3 | rollback landed mid-run | refused: `version_mismatch` |
| 4 | `0.32.1-…-85ebcb79` | **6 ok / 6, 0 degenerate** |

Every degeneration occurred on the new build; none on the old one. `prompt_eval` on the good
rows spanned 5,317–11,941, varying with image and tags, so vision was genuinely active
throughout rather than silently skipped.

**This is correlation, not a controlled experiment.** The rollback was not scheduled as part
of the test, and a container restart alone clearing the state cannot be excluded. It is
recorded as strong evidence, not proof.

## Ground-truth reproduction (2026-08-01)

The failure class was reproduced deterministically on this host with the ground-truth
vision suite in [nemotron-test-image.md](nemotron-test-image.md): on a b10091-payload
build, `gemma4:31b` — at budgets identical to the gated 0.32.1 build — dropped from
6/6 labels + a perfect invoice extraction + 5/5 chart values to 3/6 / 0/5 / 0/5, emitted
degenerate token salad on the multi-image test, and twice produced responses describing
a **previous request's** image (the #17475 shared-slot signature, also seen on
nemotron3). Temperature 0, same model blob, same prompts. The gate's "degenerate vision
output" observation is therefore not workload-specific and is now regression-testable.

## Corroborating upstream reports

Both open, both 0.32.5-specific, both matching this deployment:

**[#17459](https://github.com/ollama/ollama/issues/17459) — repeated-token degeneration.**
Reported on **Framework Desktop Max+ 395 / AMD Radeon 8060S** — the same SoC and the same
iGPU as this host — on ollama 0.32.5 via `/api/chat`: Gemma 4 emits repeated `<unused49>`
tokens when the request carries `think: false`. Same silicon, same version, same endpoint,
same symptom class.

**[#17475](https://github.com/ollama/ollama/issues/17475) — shared-slot cross-request
corruption.** Its runner config is nearly identical to ours — `-np 1`,
`--context-shift --keep 4`, `--direct-io`, vision + mmproj — and it names `qwen3.6:35b` among
the models involved. Reproduces 8/8 under concurrent submission + client aborts + model
swapping, and 0/6 when calls are serialized. Explains the
`erased invalidated context checkpoint` / `cached n_tokens = 0` churn seen in our logs.

## Why `--direct-io` is a no-go on AMD for now

[`72116baf`](https://github.com/ollama/ollama/pull/17286) (Daniel Hiltgen, upstream, not
MaxusAI) adds to `llm/llama_server.go`:

```go
if runtime.GOOS == "linux" && g.Integrated && (CUDA || ROCm) {
    params = append(params, "--direct-io")
}
```

**gfx1151 satisfies every clause** — Linux, integrated, ROCm — so the flag is applied
unconditionally here with no opt-out. Verified in the live runner cmdline: **present on
0.32.5, absent on 0.32.1.**

Its stated purpose is sound ("avoid double memory consumption by enabling direct IO for
iGPUs" — on shared-memory parts the page cache double-buffers weights). The objection is not
to the intent:

1. **It is unconditional and unexposed.** No env var or option disables it. On this host it
   arrives purely as a side effect of a version bump.
2. **It changes the weight-load path on the exact hardware class that regressed**, and
   O_DIRECT is alignment-sensitive — a silently short or misaligned read yields corrupted
   weights, whose signature is degenerate output rather than an error.
3. **It is present in #17475's runner config too**, on a DGX Spark GB10 — also a unified-memory
   part, also dio-enabled by the same commit. Two affected systems, both with dio on.
4. **It is unvalidated on ROCm iGPUs by us.** It was never measured on gfx1151 before it
   arrived; the 0.32.5 verification checked offload and throughput, not load-path integrity.

It is **not** proven to be the cause — #17475 attributes its corruption to slot sharing, and
degeneration could equally come from the llama.cpp payload bump. It is listed as a blocker
because it is an unvalidated, non-optional change to the load path on the affected hardware.

## What is *not* the cause

**The `qwen35`/`qwen35moe` 1024 image-token floor is exonerated.**
[`87cf1100`](https://github.com/MaxusAI/ollama/pull/10) was the *intended* change in the
0.32.5 cutover and drew the initial suspicion, wrongly. It alters prompt length
(313 → 1049 tokens for a 640×480 image), not output coherence, and appears in neither
upstream issue. The real risk was the **incidental payload** that rode along:

| layer | 0.32.1 → 0.32.5 | rebuilt by the overlay? |
|---|---|---|
| Fork runtime code | 1 commit (`87cf1100`) | yes (Go binary) |
| Upstream Go | 9 serving-path commits, incl. `72116baf` dio | yes (Go binary) |
| **llama.cpp payload** | **`b9888` → `b10091`** (~200 builds) | **no — comes from the base image** |

The overlay rebuilds only the Go binary, so the llama.cpp bump is never compiled or reviewed
here; it arrives wholesale inside `ollama/ollama:0.32.5-rocm`. That is the largest and least
inspected surface in any base bump, and it is where slot, checkpoint and prompt-cache logic
lives.

## Spec — normative gate

Before moving the AMD/gfx1151 deployment past `0.32.1`, **all** must hold:

1. [#17459](https://github.com/ollama/ollama/issues/17459) is **closed**, with the fix in the
   target tag.
2. [#17475](https://github.com/ollama/ollama/issues/17475) is **closed**, with the fix in the
   target tag.
3. `--direct-io` is either (a) absent for ROCm iGPUs in the target, (b) opt-out-able and
   disabled here, or (c) validated on gfx1151 — a load-path integrity check, not just
   throughput.
4. A vision A/B of **≥6 consecutive rows per build** on `qwen35moe` shows **0 degenerate** on
   the candidate, run with the rollback boundary controlled — the deficiency in the
   2026-07-31 evidence.
5. `make proof` passes against the new `BASETAG`, and `BASETAG`/`STOCK` moved together.

Gates 1–2 are upstream-dependent; 3–5 are ours. Record the outcome of each here when the gate
is next evaluated.

## Decision record

**Context.** `0.32.5-gemma4budget-4259c191` was built, verified (identity, 96 gfx1151
kernels, 100% GPU offload, budget behaviour) and deployed on 2026-07-31. It passed every
check we had. Those checks measured *plumbing* — flags, offload, token counts, throughput —
and none of them would detect degenerate output, which is what a downstream consumer hit
within the hour.

**Decisions.**

1. **Pin AMD/gfx1151 to `0.32.1-gemma4budget-85ebcb79`.** Recorded in
   `docker/ollama-rocm/.env` with the full image chain.
2. **Treat a base-image bump as a payload change, not a version bump.** The overlay makes
   base bumps look cheap — one build arg — while silently importing every llama.cpp change
   between tags. Sync deliberately, and read the upstream issue tracker for the target tag
   before deploying.
3. **Add output-quality checks to the deploy gate.** Plumbing checks passed while the build
   was producing garbage. Verification must include generating and inspecting real output,
   not just counting tokens and VRAM.
4. **Do not backport the 0.32.5 payload to get the qwen35 floor.** If the floor is wanted
   before the gate lifts, cherry-pick `87cf1100` onto `85ebcb79` and rebuild on the
   **0.32.1-rocm** base — one switch statement, none of the payload.

**Consequences.** The AMD host stays on `0.32.1`, so `qwen35`/`qwen35moe` keep the
llama.cpp default floor and small images stay cheap (313 rather than 1049 tokens at
640×480). Non-AMD deployments are **not** covered by this gate — `10.8.0.6` (Blackwell,
CUDA) is unaffected by clause 3 and may track a different version; note that #17475 was
reported on CUDA, so clauses 1–2 still apply there.

## See also

- [gemma4-budget-image.md](gemma4-budget-image.md) — build/verify/deploy runbook
- [vision-token-budget-measurements.md](vision-token-budget-measurements.md) — measured token
  cost, including the 313 → 1049 delta this gate defers
- [vision-token-budgets-by-arch.md](vision-token-budgets-by-arch.md) — why the budget is
  arch-gated
