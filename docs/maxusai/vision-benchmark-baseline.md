# Vision benchmark baseline — performance overview and regression reference

Compiled 2026-08-07. **Purpose:** one table of clean, attributed measurements to
(a) compare models/configs and (b) detect regressions. Supersedes the throughput
columns of [vision-token-budget-measurements.md](vision-token-budget-measurements.md)
(see "Reading rules" — its tps cells are load-contaminated; its accuracy cells stand).

## Platform

| | |
|---|---|
| Host | Apple M5 Max, 128 GB, macOS 26.6 (Metal; MLX metal_v4) |
| Servers | `:11434` stock ollama 0.32.6 (llama.cpp b10242) · `:11435` fork unpatched (b10091 + compat 001–003) · `:11436` fork patched (b10091 + 001–**004**, `0.32.5-gemma4fill-dev`) |
| Method | vision-suite prompts, temperature 0, think off, `format:json`; scene 1920×1080 (6 labelled shapes + 14px serial), document 1568² invoice, multi = 3 images |
| Key commits | 004 patch `14dc92a1` · ADR 0008 `b282ca97` · doc-IoU metric `62f01182` (all merged, PR #40) |

## Main matrix (scene test unless noted)

| model | quant | server / patch | budget → img tok | scene IoU | doc name_bbox IoU | 14px serial | gen tok/s | prefill tok/s |
|---|---|---|---|---|---|---|---|---|
| qwen3.6 35B-A3B | q4_K_M | patched | (qwen path) ~2031 | **0.975** | 0.686 | ✓ | **109.7** | 1269 |
| nemotron3 33B | q4_K_M | fork (002) | dynres ~2090 | 0.857 | — | ✓ | 90.4 | 885 |
| gemma4 31B | q4_K_M | patched | 1120 → 1100 | 0.963 | **0.712** | ✓ | 20.8 | 266 |
| gemma4 31B | q4_K_M | patched | 560 → 527 | 0.934 | — | ✓ | ~21 | — |
| gemma4 31B | q4_K_M | patched | 280 → 264 | 0.934 | — | ✗ | 22.6 | ~1200* |
| gemma4 31B | q4_K_M | unpatched | 280 pinned → 264 | 0.902 | — | ✗ | 23.0 | 432 |
| gemma4 26B A4B | q4_K_M | patched | 1120 → 1100 | 0.970 | — | ✓ | — | — |
| gemma4 26B A4B | q4_K_M | patched | 560 → 527 | 0.961 | — | ✓ | — | — |
| gemma4 12B | q4_K_M | patched | 1120 → 1100 | 0.885 | 0.101 | ✓ | 52.6 | 1314 |
| gemma4 12B | q4_K_M | patched | 560 → 527 | 0.940 | ~0.64 | ✓ | — | — |
| gemma4 12B | q4_K_M | **stock** (max 280) | 280 → 264 | 0.883 | 0.414 | ✗ | 53.0 | 908 |
| gemma4 12B | q4_K_M | unpatched shipped 40…1120 | → 920 (off-ladder) | 0.504 | 0.101 | ✓ | — | — |
| gemma4 12B/31B, qwen3.6 | **nvfp4** | MLX runner | — | **vision-blind** | — | — | — | — |

\* prefill tok/s on ≤264-token images is dominated by fixed overhead; treat as scatter.

Extras (patched, 12B): portrait 1056×1920 @1120 → 24×45 grid, IoU 0.905.
Fine text (1568² page): @560 reads 22px tier only; @1120 reads 22 + 16 + half of
12px. Multi-image (patched): all questions correct on 12B/31B/qwen3.6.

## Reading rules (regression testing)

1. **Accuracy cells are deterministic** per (payload, backend, budget, image) at
   temperature 0 — an IoU shift ≥0.01 on the same config is a real regression.
   `prompt_eval_count` must decompose as `text + patch_grid + 16` per image
   (scene text = 584 incl. framing, document = 358); a grid change means the
   sizing changed. On a 004 payload every gemma4 grid must satisfy
   `c·r ≤ B < (c+1)·(r+1)` for a ladder B (SPEC B7).
2. **Throughput cells are only valid from a clean single request after warm
   load.** The bsweep archive's tps columns are load-contaminated (measured 31B
   @280 at 8.1 tok/s under reload pressure vs 23.0 clean — ~3× off). Never read
   tps from sweep runs.
3. Decode tok/s is independent of image budget and of 004; prefill scales with
   image tokens. Cross-server tps (different payloads) is indicative only.
4. nvfp4 = MLX runner = images silently dropped (model confabulates). Any
   nvfp4 "vision result" is invalid until MLX vision lands.
5. Known model caveats: 12B (encoder-free) degrades at 1120 (pin 560 for bbox);
   single scene/seed per cell — treat ±0.01 IoU as noise floor, not trend.

## Repro

Patched server: worktree build per [apple-silicon-build spec](spec/apple-silicon-build.md),
`OLLAMA_MODELS=~/.ollama/models-mlx`, then
`THINK=false python3 vision-suite/vision_suite.py <host> <tag> <model>`; scores land
in gitignored `scores_<tag>.json`.
