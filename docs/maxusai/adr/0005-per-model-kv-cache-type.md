# ADR 0005: per-model KV cache type, and KV policy for reasoning models

Date: 2026-08-02 · Status: accepted (shipped with the `kv_cache_type` option)

## Context

The 2026-08-02 vision campaign root-caused qwen3.6:35b's think-mode "runaways"
to `OLLAMA_KV_CACHE_TYPE=q8_0`: on grounding-heavy prompts the reasoning phase
converges at 19,160 tokens under q8_0 vs 3,320 under f16 (~5.8× thinking-cost
inflation, deterministic, build-independent —
[vision-campaign-2026-08-02.md](../vision-campaign-2026-08-02.md) §6).
nemotron3 and gemma4 showed no q8_0 sensitivity in 96 cells. The env var is
server-wide; the sensitivity is model-specific. Upstream ollama's default is
f16 — q8_0 on this host was our own VRAM optimization, worth ~3 GB per model
at 32K ctx.

## Decision

1. **Per-model/load `kv_cache_type` option**
   ([SPEC](../kv-cache-type-per-model.md), Go-only, overlay-compatible):
   Modelfile `PARAMETER kv_cache_type …` or request `options.kv_cache_type`,
   overriding the env at runner launch; allowlist-validated with env fallback
   so typos cannot brick a load; `K/V` pair syntax (`q8_0/f16`) exposes
   llama.cpp's independent `--cache-type-k`/`--cache-type-v` (quantized V
   requires flash attention; K does not).
2. **Policy on gfx1151**: the instance may keep `q8_0` as its default;
   reasoning models — today qwen3.6 — get `kv_cache_type f16` (conservative)
   or `q8_0/f16` (validated sweet spot: half the KV memory, no FA
   requirement, no measured inflation — see the SPEC's attribution results)
   at the model level.
3. **Suite guardrail**: benchmark runs must record the KV type; think-mode
   qwen results under q8_0 with caps below ~27K are censored and must not be
   read as model failures.

## Consequences

- One instance can serve memory-cheap non-reasoning models and
  precision-sensitive reasoning models correctly at the same time.
- Changing `kv_cache_type` between requests respawns the runner (Runner-block
  comparison) — same cost profile as a `num_ctx` change; per-request toggling
  is possible but load-heavy, so per-model defaults are the intended use.
- The K/V pair syntax attributed the qwen degradation same-day: neither side
  alone inflates; only combined q8_0 does (superadditive). `q8_0/f16` is
  validated clean on both the mild and severe prompts.
- Other reasoning models may have undiscovered q8_0 cliffs; the vision suite
  plus this knob make that a one-command check per model.
