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

Suite + images: `gen_scenes.py` / `vision_suite.py` (session scratchpad; ground truth
embedded). Re-run all three models on b9888+002; run the pos-embed experiment only if
nemotron structure has not recovered there.

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
