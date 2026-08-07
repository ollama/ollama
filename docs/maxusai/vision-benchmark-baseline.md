# Vision benchmark baseline — performance overview and regression reference

Compiled 2026-08-07, gaps filled same day. **Purpose:** one table of clean,
attributed measurements to (a) compare models/configs and (b) detect regressions.
Supersedes the throughput columns of
[vision-token-budget-measurements.md](vision-token-budget-measurements.md)
(its tps cells are load-contaminated; its accuracy cells stand).

## Platform

| | |
|---|---|
| Host | Apple M5 Max, 128 GB, macOS 26.6 — **High Power Mode** (`pmset -g` powermode 2). Re-measure if power mode differs; all tps cells assume it. |
| Servers | `:11434` stock ollama 0.32.6 (llama.cpp b10242) · `:11435` fork unpatched (b10091 + compat 001–003) · `:11436` fork patched (b10091 + 001–**004**, `0.32.5-gemma4fill-dev`) |
| Method | vision-suite prompts, temperature 0, think off, `format:json`; scene 1920×1080 (6 labelled shapes + 14px serial), document 1568² invoice, multi = 3 images. Clean single request per cell, warm load, no concurrent traffic. |
| Key commits | 004 patch `14dc92a1` · ADR 0008 `b282ca97` · doc-IoU metric `62f01182` (merged, PR #40) |

## GGUF inventory (both servers share the store)

| family | q4_K_M | nvfp4 |
|---|---|---|
| gemma4 | 12b-it 7.6 GB · 26b-a4b-it 18.0 GB · 31b-it 19.9 GB | 12b 7.7 GB · 26b 17.6 GB · 31b 18.6 GB |
| qwen3.6 | 35b-a3b 23.9 GB | 35b-a3b 21.9 GB |
| nemotron3 | 33b 27.6 GB | — |

All **nvfp4** variants route to the MLX runner, which silently drops images
(vision-blind, model confabulates) — no valid vision cell exists for any nvfp4.

## Main matrix (scene test unless noted; all cells clean-measured)

| model | quant | server / patch | budget → img tok | scene IoU | doc name_bbox IoU | 14px serial | gen tok/s | prefill tok/s |
|---|---|---|---|---|---|---|---|---|
| qwen3.6 35B-A3B | q4_K_M | patched | (qwen path) ~2031 | **0.975** | 0.686 | ✓ | **109.7** | 1269 |
| nemotron3 33B | q4_K_M | fork (002) | dynres ~2090 | 0.857 | — | ✓ | 90.4 | 885 |
| gemma4 26B A4B | q4_K_M | patched | 1120 → 1100 | 0.970 | — | ✓ | 102.9 | 657 |
| gemma4 26B A4B | q4_K_M | patched | 560 → 527 | 0.961 | — | ✓ | 106.4 | 1104 |
| gemma4 26B A4B | q4_K_M | **stock** (max 280) | 280 → 264 | 0.885 | — | ✗ | 106.3 | 1102 |
| gemma4 31B | q4_K_M | patched | 1120 → 1100 | 0.963 | **0.712** | ✓ | 20.8 | 266 |
| gemma4 31B | q4_K_M | patched | 560 → 527 | 0.934 | — | ✓ | 21.6 | (cache hit)* |
| gemma4 31B | q4_K_M | patched | 280 → 264 | 0.934 | — | ✗ | 22.6 | ~1200* |
| gemma4 31B | q4_K_M | unpatched | 280 pinned → 264 | 0.902 | — | ✗ | 23.0 | 432 |
| gemma4 31B | q4_K_M | **stock** (max 280) | 280 → 264 | 0.902 | — | ✗ | 18.9 | 381 |
| gemma4 12B | q4_K_M | patched | 1120 → 1100 | 0.885 | 0.101 | ✓ | 52.6 | 1314 |
| gemma4 12B | q4_K_M | patched | 560 → 527 | 0.940 | ~0.64 | ✓ | 53.5 | 1306 |
| gemma4 12B | q4_K_M | **stock** (max 280) | 280 → 264 | 0.883 | 0.414 | ✗ | 53.0 | 908 |
| gemma4 12B | q4_K_M | unpatched shipped 40…1120 | → 920 (off-ladder) | 0.504 | 0.101 | ✓ | — | — |
| any | **nvfp4** | MLX runner | — | vision-blind | — | — | — | — |

\* prefill on ≤264-token images is overhead-dominated; the 31b@560 cell hit a warm
KV cache (0.2s load) — both invalid as prefill measurements.

**Stock 31B confirms the control-arm equivalence exactly**: 0.902 on stock b10242
matches unpatched-fork b10091 pinned @280 to three decimals.

Extras (patched): portrait 12B @1120 → 24×45 grid, IoU 0.905. Fine text (1568²):
@560 reads 22px only; @1120 adds 16px + half of 12px. Multi-image: all questions
correct on 12B/31B/qwen3.6.

## Reading rules (regression testing)

1. **Accuracy cells are deterministic** per (payload, backend, budget, image) at
   temperature 0 — reproduced to three decimals across reruns this campaign. An
   IoU shift ≥0.01 on the same config is a real regression. `prompt_eval_count`
   must decompose as `text + patch_grid + 16` per image (scene text = 584 incl.
   framing, document = 358); a grid change means the sizing changed. On a 004
   payload every gemma4 grid must satisfy `c·r ≤ B < (c+1)·(r+1)` for a ladder B
   (SPEC B7).
2. **Cross-image variance ≈ ±0.01**: 31B @1120 scored 0.952 on a re-rendered
   scene variant vs 0.963 on the suite scene. Same-config drift beyond that is
   signal.
3. **Throughput cells are only valid from a clean single request after warm
   load, in High Power Mode.** Sweep-archive tps is load-contaminated (31B @280:
   8.1 tok/s under reload pressure vs 23.0 clean, ~3× off). Watch for KV-cache
   hits: `load_duration` ≪ 1s with an implausible prefill tok/s means the image
   prefix was cached — discard the prefill cell.
4. Decode tok/s is independent of image budget and of 004; prefill scales with
   image tokens. Cross-payload tps (stock vs fork) is indicative only — e.g.
   stock 31B decodes 18.9 vs fork 21–23 on different llama.cpp builds.
5. nvfp4 = MLX runner = images silently dropped (model confabulates). Any nvfp4
   "vision result" is invalid until MLX vision lands.
6. Model caveats: 12B (encoder-free) degrades at 1120 — pin 560 for bbox. 26B
   A4B is the gemma4 speed/accuracy sweet spot (~105 tok/s, 0.970 @1120).
   Single scene/seed per cell.

## Repro

Patched server: worktree build per [apple-silicon-build spec](spec/apple-silicon-build.md),
`OLLAMA_MODELS=~/.ollama/models-mlx`, then
`THINK=false python3 vision-suite/vision_suite.py <host> <tag> <model>`; scores land
in gitignored `scores_<tag>.json`. Verify High Power Mode first: `pmset -g | grep powermode` → 2.
