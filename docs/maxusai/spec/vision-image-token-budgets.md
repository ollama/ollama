# SPEC: vision image-token budgets

MaxusAI-fork specification. Status: **implemented** on both maintained lineages
(fork `main`, `release/0.32.1-dynres`). Written 2026-08-02.

Normative contract for how many visual tokens an image costs and who decides it.
Measured costs per arch and size live in
[vision-token-budget-measurements.md](../vision-token-budget-measurements.md); the
mechanism analysis is in
[vision-token-budgets-by-arch.md](../vision-token-budgets-by-arch.md); the decisions
are [ADR 0001](../adr/0001-nemotron-vision-dynamic-resolution.md) and
[ADR 0003](../adr/0003-vision-image-token-budget-policy.md).

## 1. Where the budget is decided

**B1 — Budgets are launch-time.** `visionServerArgs(modelArch, opts)` contributes
`--image-min-tokens` / `--image-max-tokens` to the llama-server process command line
when the runner starts. Budgets are therefore a property of the **loaded runner**, not
of a request.

**B2 — Budgets are endpoint-independent.** Because of B1, `/api/chat`,
`/api/generate` and every `/v1` endpoint MUST observe identical per-image costs for
the same model and image. Measurements taken through one endpoint are valid for all.
(Verified: nemotron3 + dynres, 1920×1080 image, `prompt_eval_count` 2061 on both
`/api/generate` and `/api/chat`.)

**B3 — Budget options are Runner options.** `ImageMinTokens` / `ImageMaxTokens`
changes reload the runner. Clients MUST expect a reload when they vary them per
request.

**B4 — Two sides must agree.** A flag only has effect if the loaded projector calls
`set_limit_image_tokens()` and consumes it. Adding an arch to `visionServerArgs`
is **half** the change; the other half belongs to llama.cpp (upstream or a
`llama/compat` patch). An arch whose projector ignores the flags MUST NOT be given
them, so the API does not advertise a knob that does nothing — with the deliberate
exception in B5.

**B5 — Forward-compatible flags are permitted where a lineage patch makes them
live.** `nemotron_h_omni` receives budget flags on every maintained lineage: both
currently carry `llama/compat/002-llama-cpp-nemotron-dynres.patch`, which consumes
them. Against a pristine `llama/` (upstream stock, or `main` between `5ad093b0` and
`2487dd56`) llama-server still parses them but the projector ignores them and the cost
is a structural 256/image. Whether the flags are live is therefore a property of the
payload, not of the arch — recorded in the table rather than silently tolerated.

## 2. Per-architecture contract

| `modelArch` | flags | effective budget | consumed by |
|---|---|---|---|
| `gemma4` | max from `api.Options`, defaults **70 / 1120** ([ADR 0008](../adr/0008-gemma4-budget-fill-restores-1120.md)); min is a no-op on the 004 payload | ladder rung ≤ max, grid fills it | gemma4v projector + `llama/compat/004` budget-fill (snap to ladder, scale up/down, `PAD_NONE`) |
| `qwen2vl`, `qwen25vl`, `qwen3vl`, `qwen3vlmoe`, `qwen35`, `qwen35moe` | `--image-min-tokens 1024` | 1,024 … 4,096 | `PROJECTOR_TYPE_QWEN3VL`, `set_limit_image_tokens(8, 4096)` with the floor raised |
| `nemotron_h_omni` | min/max from `api.Options`, defaults 256 / 3328 | 256 … 3,328 with compat/002; exactly 256, flags inert, without it | `PROJECTOR_TYPE_NEMOTRON_V2_VL` as patched (ADR 0001) |
| `mistral3`, `glmocr`, `llama4`, `deepseekocr`, all others | none | projector default / structural | — |

`qwen35` and `qwen35moe` are in the Qwen row because `llama/compat`'s
`handle_qwen35_like_clip()` sets `clip.projector_type = "qwen3vl_merger"`, so they
load as `PROJECTOR_TYPE_QWEN3VL` — the branch that emits the "requires at minimum
1024 image tokens" warning.

### 2.1 Gemma 4's vendor-documented budget ladder

Google's model card — **<https://ai.google.dev/gemma/docs/core/model_card_4>** —
specifies the supported visual token budgets as a **discrete ladder**, not a
continuous range:

> The supported token budgets are: **70, 140, 280, 560, and 1120.**

Consequences for this spec:

- **B6 — The shipped gemma4 defaults MUST be ladder values.** The fork ships
  **70 / 1120** ([ADR 0008](../adr/0008-gemma4-budget-fill-restores-1120.md)) —
  the documented floor and maximum. Moving a default to a non-rung requires
  superseding that ADR. Higher rungs preserve detail for OCR; lower rungs suit
  classification and video, and are selectable per request.
- **B7 — Delivered grids MUST be ladder-reachable.** The model only grounds
  `box_2d` accurately on grids satisfying `c·r ≤ B < (c+1)·(r+1)` for a supported
  budget B ([findings §9](../gemma4-bbox-investigation-findings.md)). The
  `llama/compat/004` budget-fill patch enforces this: the requested ceiling snaps
  down to the ladder (sub-70 clamps up), the image scales — up or down — to fill
  it, no letterbox. ADR 0007's interim 560 ceiling (a mitigation for the
  off-ladder defect, superseded by ADR 0008) applied only to unpatched payloads.
- **Off-ladder request values are accepted but snapped.** `gemma4ImageTokenBudget()`
  passes integers through; the payload snaps the ceiling (900 → 560) and ignores
  `min`. `MaxImageTokens`/`ImageTokensForSize` mirror the snap-and-fill so
  scheduling charges what llama-server delivers — including the upscale of small
  images to the budget (a 640×480 costs ~1064 tokens at ceiling 1120, not 132).

**Model sizes are not interchangeable for vision results.** Per the same card:

| variant | params | vision encoder | context |
|---|---|---|---|
| E2B | 2.3B effective (5.1B w/ embeddings) | ~150M | 128K |
| E4B | 4.5B effective (8B w/ embeddings) | ~150M | 128K |
| 12B Unified | 11.95B | **encoder-free** | 256K |
| 26B A4B (MoE) | 25.2B total / 3.8B active | ~550M | 256K |
| 31B Dense | 30.7B | ~550M | 256K |

The 12B is **encoder-free** while 26B/31B carry a ~550M vision encoder. These are
different vision paths, so a budget/bbox measurement taken on one size MUST NOT be
generalised to another without re-measuring — see
[vision-token-budget-measurements.md](../vision-token-budget-measurements.md) for the
per-size sweep and `run_budget_sweep.sh` for the harness.

## 3. Verifying that a flag binds

Adding an arch to the switch MUST be accompanied by an empirical check, because B4
cannot be established by reading the Go side.

Procedure — the fingerprint method:

1. Measure a text-only request with `num_predict: 1`; record `prompt_eval_count` as
   the baseline.
2. Measure the same request with a **sub-budget** image (small enough that the
   proposed floor would bind, or large enough that a ceiling would). Image tokens are
   `prompt_eval_count − baseline`.
3. Repeat with the flag applied. The count MUST change in the predicted direction. If
   it does not, the projector is ignoring the flag and the arch MUST NOT be added
   (B4).
4. Re-measure at corpus-representative sizes to confirm the change is confined to the
   intended range.

Worked example (qwen3.6 `qwen35moe`, b9888 + 002, baseline 14):

| image | without floor | with `--image-min-tokens 1024` |
|---|---|---|
| 224×224 | 51 | 1026 |
| 448×448 | 198 | 1026 |
| 896×896 | 786 | 1026 |
| 1920×1080 | — | 2042 (floor does not bind) |
| 1568×1568 | — | 2403 (floor does not bind) |

## 4. Lineage rule

The maintained lineages pin different llama.cpp versions and patch them independently,
so the *consumers* of these flags can differ even though the Go-side switch is shared.
Consequences:

- Arch entries MAY be backported freely between lineages; **assertions about a
  projector's behaviour MAY NOT**. A change that adds an arch and also asserts what a
  *different* arch's projector does must be split, taking only the arch-specific half.
  (Concretely: `87cf1100` added the qwen floor *and* asserted `nemotron_h_omni` gets no
  flags. The first half was portable; the second described only `main`'s
  then-pristine `llama/` and was already false elsewhere.)
- Every lineage MUST keep `TestVisionServerArgs` expectations consistent with its own
  payload, and MUST re-check them whenever that payload changes. Expectations agreeing
  across lineages today is a coincidence of both carrying compat/002, not an invariant.

## 5. Conformance

- `TestVisionServerArgs` — the §2 table, per lineage per §4.
- `TestImageTokensForSize` — the replicated cost model for non-budgeted compat arches.
- The §3 fingerprint procedure, recorded in
  [vision-token-budget-measurements.md](../vision-token-budget-measurements.md) when
  an arch is added or a payload changes.
