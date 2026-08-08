# `ollama-rocm-nemotron`: the dynres validation image

MaxusAI-fork reference (fork-only). Written 2026-08-01. Companion to
[nemotron-dynres-patch.md](nemotron-dynres-patch.md) and
[ADR 0001](adr/0001-nemotron-vision-dynamic-resolution.md).

> **The one thing to take away:** this image exists to answer one question — *does the
> 002 patch actually lift the 256-token cap on the gfx1151 host* — and it answers it for
> **mechanics only**. It is built from the 0.32.5-synced branch (llama.cpp `b10091`), the
> payload family the AMD gate exists to keep off this host. Do **not** promote it, tag it
> into the deployment chain, or draw output-quality conclusions from it.

## What it is

The repo's own `Dockerfile` with `--build-arg FLAVOR=rocm` — i.e. the same recipe as
`ollama/ollama:*-rocm` — built from `feat/nemotron-dynres-vision-budget`, so the payload
contains `llama/compat/001` + `002` (patched `llama-server`) and the Go binary carries the
`visionServerArgs()` nemotron case:

```bash
docker build --build-arg FLAVOR=rocm \
  -t ollama-rocm-nemotron -t ollama-rocm-nemotron:0.32.5-<sha> \
  /opt/github/MaxusAI/ollama
```

This is a **full build** (the overlay cannot carry the C++ patch — see
[gemma4-budget-image.md](gemma4-budget-image.md)); ROCm flavor skips the CUDA/Vulkan/MLX
stages.

## Running it — isolated from the live deployment

The live `ollama-rocm` container (port 11434, gated `0.32.1` image) must not be touched.
The test container differs deliberately: its own name, port **11435**, the host model
store mounted **read-only** (pull/rm impossible), and a scratch dir for the daemon home.
Device/group/ipc recipe mirrors `~/deployments/ollama/docker/ollama-rocm/docker-compose.yml`:

```bash
mkdir -p /tmp/ollama-nemotron-home
docker run -d --name ollama-rocm-nemotron -p 11435:11434 \
  --device /dev/kfd --device /dev/dri \
  --group-add 44 --group-add 992 \
  --security-opt seccomp=unconfined --cap-add SYS_PTRACE \
  --ipc host --shm-size 16g \
  -v /tmp/ollama-nemotron-home:/root/.ollama \
  -v /opt/ollama/.ollama/models:/root/.ollama/models:ro \
  -e OLLAMA_DEBUG=1 -e ROCR_VISIBLE_DEVICES=0 \
  -e OLLAMA_FLASH_ATTENTION=1 -e OLLAMA_KV_CACHE_TYPE=q8_0 \
  ollama-rocm-nemotron
```

GPU memory is shared with the live daemon — check `docker exec ollama-rocm ollama ps`
first and prefer `nemotron3:33b-q4_K_M` (already in the host store) over `q8`.

## Test protocol

Method identical to [vision-token-budget-measurements.md](vision-token-budget-measurements.md):
solid-colour PNGs, `/api/generate` with `num_predict: 1`, `prompt_eval_count` minus the
text-only baseline (nemotron3 baseline: 18). Grid-quantised — do not read small deltas.

1. **Budget fingerprint at load:** the model-load log must print `image_min_pixels: 262144`
   and `image_max_pixels: 3407872`. Their absence means the payload is unpatched (the
   `-1` sentinels are never logged).
2. **Token counts** (pass = within a few tokens of expectation; hard fail = flat 256):

   | image | px | expected grid | expected Δtokens (grid + 2) |
   |---|---|---|---|
   | 640×480 | 307,200 | 20×15 | **302** |
   | 896×896 | 802,816 | 28×28 | **786** |
   | 1920×1080 | 2,073,600 | 60×34 (approx) | **≈2,042** |
   | 1568×1568 | 2,458,624 | 49×49 | **2,403** |
   | 3000×2000 | 6,000,000 → scaled to ≤3,407,872 | ≈70×47 | **≈3,292** |
   | 320×240 (floor) | 76,800 → upscaled ≥262,144 | ≈ min grid | **≈258–320** |

   (Pre-patch, every row measured exactly **256**. The +2 is the new `<img>`/`</img>`
   markers. Expectations use `calc_size_preserved_ratio`'s 32px alignment; NVIDIA's exact
   snap rounding can differ by one grid row/column — that is parity noise, not failure.)
3. **`image_max_tokens` knob:** request with `"options": {"image_max_tokens": 1024}` on a
   large image → count drops to ≈1,026; on an unpatched payload this was silently ignored.
4. **Coherence smoke** (not a quality verdict — b10091 confound): one short caption per
   image; log output; anything degenerate/repetitive is recorded but attributed per
   [amd-upgrade-gate.md](amd-upgrade-gate.md) blockers, not automatically to 002.
5. **Warmup/VRAM:** note load time and `rocm-smi` VRAM at idle-after-load vs the live
   container's nemotron figures — warmup now probes ~1846².

## Results

**2026-08-01 — native host build (same branch `fe261904`, same 001+002 patches, gfx1151-only
`rocm_v7_2_user_arch` preset), served via `go`-built binary on :11435, gfx1151/ROCm 7.2,
`nemotron3:33b-q4_K_M`, `OLLAMA_FLASH_ATTENTION=1`, `OLLAMA_KV_CACHE_TYPE=q8_0`.**
This validates the patch mechanics on the exact target GPU; the containerized
`ollama-rocm-nemotron` run should reproduce these numbers (append below when run).

Text-only baseline: 18. `visual+markers` = `prompt_eval_count` − 18. **VERDICT: the 256
cap is lifted — mechanics behave exactly as specified.**

| image | measured | expected | note |
|---|---|---|---|
| 320×240 | **270** | floor upscale | pre-patch: 256 flat |
| 640×480 | **304** | ≈302 | |
| 896×896 | **788** | ≈786 | |
| 1568×1568 | **2,405** | ≈2,403 | 9.4× pre-patch |
| 1920×1080 | **2,044** | ≈2,042 | PR author measured 2,040 visual on CUDA |
| 2048×1664 | **3,332** | 3,330 | **exact 3,328 ceiling reached** |
| 3000×2000 | **3,294** | ≈3,292 | 12.9× pre-patch |
| 3200×32 (100:1) | **324** | bounded | no degenerate-aspect blowup at this ratio |
| 1920×1080 @ `image_max_tokens=1024` | **1,012** | ≈1,010 | **knob live** (2,044 → 1,012); silently ignored pre-patch |

- Every row sits a constant +2 above the pure grid+2 arithmetic — one extra token pair
  from prompt assembly, grid-quantisation-level noise, consistent across all sizes.
- Bicubic `resize_position_embeddings` exercised on ROCm at every non-512² grid
  (all counts correct, output coherent) — no fallback, no garbage.
- Coherence smoke (`think:false`, solid-red 1920×1080): "The image is predominantly red
  with a black border." Correct color, fluent, no repetition. (With default reasoning on,
  small `num_predict` is consumed by thinking and `response` comes back empty — use
  `think:false` or read the `thinking` field when smoke-testing.) Not a quality verdict —
  the b10091 confound still applies to sustained workloads.
- VRAM with model resident after worst-case warmup: **27.2 GiB** of the 96 GiB pool.
- The `image_min_pixels`/`image_max_pixels` load lines were not visible in Ollama's
  captured server log at `OLLAMA_DEBUG=1`; use `prompt_eval_count` as the fingerprint
  instead (flat 256 = unpatched).

### Content-quality A/B (2026-08-01, ground-truth suite)

Long-prompt JSON+bbox suite (`vision_suite.py`, three synthetic ground-truth images) run
at temperature 0 / `think:false` / `format:"json"` against both payloads — same model
blob, both served via llama-server+mtmd (verified in launch logs):

| metric | patched (dynres, b10091) | unpatched (512² fixed, b9888 live) |
|---|---|---|
| scene: 20px labels read | **3/6** | 0/6 |
| scene: 14px serial | near-miss (`SN-4921-KK`) | not seen |
| scene: objects enumerated | 3/6 | **6/6** |
| scene: bbox hits / colors | 0 / 1 | 0 / 0 |
| invoice: header + invoice-no | **exact** | exact |
| invoice: 17px fine print | **verbatim** (multi run) | not seen |
| invoice: line items faithful | 0/5 (hallucinated names) | **3/5 partial, total exact** |
| chart: values read (multi) | 2/5 | **5/5** |
| multi: cross-image q1 (find INV code) | right | right |

**Reading (revised after the control matrix below):** the budget lift delivers exactly
what it promises on *fine text* (labels, serial, fine print — all invisible at 256
tokens), but on this payload **global spatial structure degrades** (objects missed,
attributes scrambled, confabulated line items). Cause attribution, in order of proof:

1. **The b10091 payload is vision-broken independently of 002 — proven by the gemma4
   control.** gemma4:31b ran the same suite at *identical* budgets on both payloads
   (prompt counts equal; 002 does not touch gemma4) and collapsed only on b10091:
   scene 6/6→3/6 labels, invoice 5/5-items-perfect→0/5 with a confabulated refusal,
   chart 5/5→0/5, output lengths 993→42 tokens, including outright degenerate token
   salad (`{"thought}<channel|>...`) — the same "degenerate vision output" that forced
   the 2026-07-31 rollback, now reproduced cold on ground truth. The nemotron
   structural regression above is therefore confounded and must be re-measured on
   **b9888+002** (the production candidate) before further attribution.
2. **Cross-request contamination on b10091, reproduced on two models** — the warm
   nemotron document run contained the *previous request's* scene labels (cold rerun
   did not), and the gemma4-patched document response described the scene image
   outright. That is the [#17475](https://github.com/ollama/ollama/issues/17475)
   shared-slot signature, independently reinforcing
   [amd-upgrade-gate.md](amd-upgrade-gate.md).
3. **Position-embedding fidelity (now an unproven secondary hypothesis):** the compat
   layer bakes the 128×128 RADIO grid to 32×32 at load and the 002 graph
   bicubic-upsamples that intermediate to grids up to 115×115, where the reference
   interpolates 128²→target directly (max-dim-then-crop for non-square). Symptoms were
   consistent (fine texture readable, layout scrambled, worst on the 60×34 grid), but
   the gemma4 control shows the payload alone explains most of it. Keep the fork-local
   fix (native 128² grid in `handle_nemotron_h_omni_clip()`'s pos-embed load-op) as an
   experiment to run on b9888+002 only if structure has not recovered there. Grammar
   constraint and reasoning mode were ruled out (identical scores without
   `format:"json"`).

### Control matrix: gemma4 + qwen3.6, same suite, both payloads

Both models run at **identical budgets on both payloads** (equal `prompt_eval_count`
per test; 002 touches neither arch), so these columns isolate the payload version.
gemma4 with model-default reasoning; qwen3.6 with `think:false` (with reasoning on,
both payloads returned empty responses — thinking consumed `num_predict`; harness
note, not a payload signal).

| metric | gemma4:31b b9888-live | gemma4:31b b10091 | qwen3.6:35b b9888-live | qwen3.6:35b b10091 |
|---|---|---|---|---|
| scene labels (20px) | **6/6** | 3/6 | **6/6** | 3/6 |
| scene colors | **6/6** | 3/6 | **6/6** | 2/6 |
| scene serial (14px) | – | found | **found** | found |
| invoice line items | **5/5**, qty/price 5/5, total exact | 0/5 (described the *previous request's image*) | **5/5**, qty/price 5/5, total exact | 0/5 (142-token collapse) |
| chart values | **5/5** | 0/5 (degenerate token salad) | **5/5** | 4/5 |
| cross-image q1/q2 | right/right | wrong/wrong | right/right | right/wrong |

**Conclusion:** the b10091 payload degrades vision quality across all three models with
one signature — half the objects, collapsed/confabulated outputs, cross-request
leakage — at unchanged token budgets. This (a) reproduces the AMD gate's rollback
rationale deterministically ([amd-upgrade-gate.md](amd-upgrade-gate.md)), (b) confounds
the nemotron structural regression above, and (c) makes **b9888+002** the decisive next
build: it escapes the broken payload, tests 002 on a healthy substrate, and is the
production candidate anyway. On the healthy b9888 payload, gemma4:31b and qwen3.6:35b
both ace this suite — consistent with the routing policy in
[vision-token-budget-measurements.md](vision-token-budget-measurements.md) — while
nemotron's 256-token cap leaves it structure-capable but fine-text-blind there.

Suite + images: [`vision-suite/`](vision-suite/) (committed; ground truth embedded).

### VERDICT — b9888+002 (2026-08-01, native build from `feat/nemotron-dynres-0321`)

Token protocol: **byte-identical** to the b10091+002 runs (270…3,332 dynamic, exact
ceiling at 2048×1664, knob 2,044→1,012) — the 002 patch behaves the same on b9888.
Ground-truth suite (`think:false`, cold):

| test | b9888 stock (256 cap) | b10091+002 | **b9888+002** |
|---|---|---|---|
| nemotron scene labels / colors / objects | 0/6, 0/6, 6 | 3/6, 1/6, 3 | **6/6, 6/6, 6 (+serial exact)** |
| nemotron invoice | partial, totals only | 0/5 hallucinated | **5/5 items, 5/5 qty/price, total exact** |
| nemotron chart (multi) + q1/q2 | 5/5, right/right | 2/5, right/wrong | **5/5, right/right** |

Controls on the same b9888+002 build: gemma4:31b and qwen3.6:35b (`think:false`) both
score **perfect** — gemma4 identical to b9888-live cell-for-cell (build is healthy), and
both even pick up the 14px serial. Conclusions:

1. **b9888+002 is quality-positive on every axis**: the good payload's structure plus
   the patch's fine-text gains. It is the validated production candidate.
2. **The pos-embed double-resample hypothesis is refuted as a blocker** — structure is
   fully intact with the baked-32² interpolation. The queued compat experiment is
   closed unless future OCR-grade grounding work reopens it.
3. **Think-mode finding (all payloads):** `think:true` + `format:"json"` returns an
   empty `response` for nemotron3 and qwen3.6 (thinking ends without a JSON body,
   eval 500–1,200 « the 4,000 budget); gemma4 works in both modes, and gemma4
   `think:false` shows no #17459 degeneration on b9888. For JSON extraction on the
   reasoning models, serve with `think:false`.
4. ~~Known residual: pixel-bbox localization is weak~~ **Superseded 2026-08-02**: the
   weak bbox scores were a scorer decode artifact — each model answers in its trained
   coordinate dialect regardless of instructions (qwen3.6: `bbox_2d` xyxy norm-1000,
   IoU ≈ 0.95; gemma4: Gemini `box_2d` **yxyx** norm-1000, IoU ≈ 0.78; nemotron3:
   self-chosen `bbox_2d` xyxy norm-1000 **of its input canvas**, so stock letterboxed
   payloads skew the y-axis by the padding — another point for dynres). The suite
   scorer now searches key × space × order ([vision-suite/](vision-suite/) README has
   the dialect map).

**2026-08-01 — containerized run, `ollama-rocm-nemotron:gfx1151-host-8d97cdea`** (the
host-artifact image variant: ubuntu:24.04 final stage + the native build's payload,
because the canonical almalinux toolchain pull was hours from completing; provenance in
the image LABELs). Run with the isolated recipe above (port 11435, models `:ro`).
**Every measurement byte-for-byte identical to the native run** — all eight geometries,
the knob (2,044 → 1,012), and the coherence sample ("predominantly red with a thin black
border", `think:false`). gfx1151 discovered from inside the container
(`library=ROCm compute=gfx1151`, 96 GiB pool). Append the canonical all-target image's
run here when its build lands; expect identical numbers.

## Promotion path (do not skip)

Passing here proves the mechanism, not deployability. The deployable artifact for this
host is 002 + the Go commits cherry-picked onto the **`85ebcb79` / 0.32.1 / `b9888`**
lineage (re-`git apply --check` the patch against `b9888`), full-built, then taken
through the deploy gate in [amd-upgrade-gate.md](amd-upgrade-gate.md) including the
output-quality A/B — on a payload without the b10091 degeneration confound.

### Addendum 2026-08-02 — think:on cells resolved (generate think+format fix)

The misfiled think:on cells were an Ollama bug, not model failures:
`/api/generate` (+`/v1/completions`) with thinking + `format` never worked in any
release — root cause, fix (two-pass stop-split), and history in
[generate-think-format-empty-response.md](generate-think-format-empty-response.md)
and [ADR 0002](adr/0002-deferred-format-constraining.md) (mechanism since moved
to the routes layer,
[ADR 0004](adr/0004-routes-layer-think-format-double-request.md)); fix merged to
main (PR #22, re-architected in the routes-layer PR) and cherry-picked to
`release/0.32.1-dynres`.

On the fixed b9888+002 build (`ollama-dynres-genfix*`), think:on + format:json,
temperature 0:

| test | nemotron3 | qwen3.6 |
|---|---|---|
| scene | 6/6 labels, 6/6 colors, serial exact, **5/6 bbox center-hits** (pixel-space, IoU ≈ 0.3) | 6/6 + serial, bbox IoU ≈ 0.95 once norm-1000-decoded |
| invoice | 5/5 items, 5/5 qty/price, total exact, **5/5 name-bbox** | 5/5 + total exact |
| 3-image | all answers right — needs a ≈16K generation budget | all right — needs > 8K |

Reasoning materially improves nemotron's localization (centers 0–1 → 5/6) at 4–16×
generation cost. Completing the earlier matrix: nemotron stock think:on also misfiles
(the bug exists on 0.32.1 — it predates every payload here); gemma4 b9888 think:off is
perfect including the serial; gemma4 b10091 think:off collapses on scene/invoice with
the multi test partly intact. Serving guidance: on fixed builds, think:on + format is
legitimate and preferred when grounding matters; on unfixed builds keep `think:false`.

### Addendum 2026-08-02 (2) — routes-layer fix cherry-picked, image rebuilt + canaried

The think+format mechanism moved to the routes layer
([ADR 0004](adr/0004-routes-layer-think-format-double-request.md), fork PR #31):
`release/0.32.1-dynres` now carries `82158bd8` (routes-layer double request) +
`ae797815` (runner-split removal) on top of the earlier v1 cherry-picks. Full
`server`/`llm`/`model/parsers` suites pass on the branch.

Image: **`maxusai-ollama:0.32.1-rocm-dynres-ae797815`** — built with the overlay
recipe (`Dockerfile.overlay`, Go binary only) on base
`maxusai-ollama:0.32.1-rocm-dynres-a4788474`, so the b9888+002 llama-server
payload is bit-identical to the canonical image and every payload-level verdict
above stands. Canaried per the isolation recipe (own name, port 11442, models
`:ro`), then removed:

| probe (temp 0) | result |
|---|---|
| nemotron3 `/api/generate` think+`"json"` | thinking (309 chars) + `{"capital": "Paris"}`, stop, prompt_eval 31 / eval 85 |
| qwen3.6 `/api/chat` think+`"json"` | thinking (817 chars) + `{"capital": "Paris"}`, stop, **prompt_eval 24** (honest — pre-fix chat reported the cache-inclusive continuation prefill) / eval 247 |
| qwen3.6 `/v1/completions` `json_object` | valid JSON, stop, usage 24/223/247 |

The canonical container on :11435 still runs `…-a4788474`; promoting
`…-ae797815` into it goes through the usual deploy gate, not this doc.

### Addendum 2026-08-02 (3) — full-factorial campaign vs genuine upstream

The complete build × model × thinking × endpoint × num_ctx campaign (144 cells,
baseline corrected to genuine `ollama/ollama:0.32.1-rocm`, dialect-corrected Q4
scoring, max-context arm, and the qwen runaway root cause: q8_0 KV cache) is
consolidated in [vision-campaign-2026-08-02.md](vision-campaign-2026-08-02.md),
with raw logs and parsed JSON under [vision-suite/runs/](vision-suite/runs/).

### Addendum 2026-08-03 — kv_cache_type image built, canaried, promoted; f16 fleet-wide

Green-lit deploy: overlay image **`maxusai-ollama:0.32.1-rocm-dynres-258534eb`**
(tag `v0.32.1-dynres.2`; Dockerfile.overlay on base `…-a4788474`, payload
bit-identical; Go binary carries the routes-layer v2 think+format fix and the
per-model/request `kv_cache_type` option incl. K/V pair syntax). Canary on
:11442: version stamp OK; default emits NO `--cache-type` flags (llama-server
f16 default); `options.kv_cache_type="q8_0/f16"` → `--cache-type-k q8_0
--cache-type-v f16` on the live runner; nemotron generate+think+`"json"` →
valid JSON with thinking intact. Promoted to the canonical container on
**:11435** with explicit `OLLAMA_KV_CACHE_TYPE=f16` and a durable home dir
(`~/deployments/ollama/homes/canonical-11435`). Prod **:11434** recreated by
the operator with `OLLAMA_KV_CACHE_TYPE=f16` (same gemma4budget image;
rollback container `ollama-rocm-q8backup` retained). The q8_0 reasoning-
inflation exposure (campaign §6) is closed fleet-wide; q8_0 remains available
per-request via the pair syntax where memory matters.

---

## Deploy record — `35d9e58e` promoted to prod `:11434` (2026-08-08)

**First full build to reach production.** `maxusai-ollama:0.32.1-rocm-dynres-35d9e58e`
(version `0.32.1-dynres-35d9e58e`), a `FLAVOR=rocm` Dockerfile build from
`release/0.32.1-dynres`, replacing the overlay `…-gemma4budget-85ebcb79`. Carries compat
**002** (nemotron dynamic resolution — was a hard 256 tokens/image, now up to 3328), **004**
(gemma4 budget-fill sizing), **005** (pinned-budget overshoot). The overlay cannot carry any
of them: they are C++ under `llama/compat`, and the overlay rebuilds only the Go binary.

Gate-safe: payload **b9888**, `--direct-io` absent. This is a move *within* the b9888
lineage, not an upgrade past it, so [amd-upgrade-gate.md](amd-upgrade-gate.md) is unaffected.

**Build cost, measured** (corrects the ~45–70 GB figure, which is the all-flavours build):
~24 min wall, ~12 GB net disk (82 G → 70 G free), 3.08 GB image, cold cache, base already
pulled.

**Verified before cutover** (bench container on `:11435`, store `:ro`, `NUM_PARALLEL=1`):

| model | scene IoU | doc IoU | img tok | notes |
|---|---|---|---|---|
| gemma4:31b @1120 | 0.961 | 0.728 | 1100 / 1089 | both grids on-ladder (SPEC B7) |
| qwen3.6 35B-A3B | 0.953 | 0.320 | 2031 | W3 all correct |
| nemotron3 33B | 0.840 | 0.058 | 2026 | W3 all correct |

005 confirmed: `pinned 3328` delivers **3254** (≤ ceiling), against `3388 ⚠` pre-005 in the
baseline §4.3.

**The decisive check was `qwen3.6`** — the model `10.8.0.6` actually drives, and the one
that degenerated in the July incident. Its document IoU of 0.320 looked like a regression
against the 0.686 recorded on Metal, so it was A/B'd on the *same hardware* against the
outgoing image: **0.317 vs 0.320**, `prompt_eval` byte-identical (2615 / 2743). Platform,
not patches. qwen3.6 touches none of 002/004/005 (004 is gemma4-only; 002/005 are
nemotron/dyn_size). Full cell: [vision-benchmark-baseline.md](vision-benchmark-baseline.md) §4.5.

**Not verified:** the exit-255 crash seen on `dynres-258534eb` (inside `process_mtmd`,
`n_tokens = 8601`) did **not** recur, but the suite's heaviest leg reached only 6203 tokens,
so the triggering conditions were never reproduced. No-crash here is not evidence of a fix.

Deploy notes: the outgoing container had lost its compose labels (recreated outside compose
on 2026-08-02), so `docker compose up -d` could not adopt it and left an orphan; it needed an
explicit `docker rm -f` and a brief stop rather than a seamless recreate. It is
compose-managed again. Rollback images retained; `~/deployments` `.env` carries the chain.
