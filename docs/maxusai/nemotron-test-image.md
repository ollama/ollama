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
