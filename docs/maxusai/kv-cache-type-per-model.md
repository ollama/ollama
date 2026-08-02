# SPEC: per-model `kv_cache_type` option

## Motivation

`OLLAMA_KV_CACHE_TYPE` is server-wide, but KV quantization tolerance is
model-specific: qwen3.6:35b's vision reasoning degrades ~6× under q8_0
(thinking converges at 19,160 tokens vs 3,320 at f16 on the identical prompt,
temp 0 — [vision-campaign-2026-08-02.md](vision-campaign-2026-08-02.md) §6),
while nemotron3 and gemma4 showed no q8_0 sensitivity in 96 cells. Operators
should not have to choose one KV type for every model on an instance.

## Design

- New load-time option `kv_cache_type` (string) in `api.Options`' `Runner`
  block ([api/types.go](../../api/types.go)), so it works everywhere runner
  options already work:
  - `PARAMETER kv_cache_type f16` in a Modelfile (per-model default),
  - `"options": {"kv_cache_type": "f16"}` on generate/chat (per request).
- Resolution at runner launch (`resolveKVCacheType`,
  [llm/server.go](../../llm/server.go)): option overrides
  `OLLAMA_KV_CACHE_TYPE`; empty option keeps the env value; both empty keeps
  llama-server's own default (**f16**). Values are trimmed + lower-cased and
  validated against what `--cache-type-k/v` accepts at the pinned build
  (f32, f16, bf16, q8_0, q4_0, q4_1, q5_0, q5_1, iq4_nl); an invalid value
  logs a warning and falls back to the env so a Modelfile typo cannot make a
  model unloadable.
- The launch still passes the two separate flags `--cache-type-k <t>
  --cache-type-v <t>` (there is no combined `--cache-type-kv` in llama.cpp).
- Reload semantics come free: the field lives in `Runner`, and the scheduler's
  `needsReload` compares `Runner` blocks with `reflect.DeepEqual` — changing
  `kv_cache_type` between requests respawns the runner exactly like a
  `num_ctx` change.

## K and V independence (source-verified at the b9888 pin)

`--cache-type-k` and `--cache-type-v` are fully independent llama-server
flags backed by separate params (`common/arg.cpp:2174,2187`); **different
types for K and V are legal**. There is no combined `--cache-type-kv` flag —
"KV cache type" being one value is an *ollama* convention (the env applies a
single type to both), not a llama.cpp constraint. The one asymmetry
(`src/llama-context.cpp:3550`): **a quantized V cache requires flash
attention** — llama-server errors with "V cache quantization requires
flash_attn" otherwise. K-only quantization has no FA requirement (only the V
cache is stored transposed in the non-FA path, `src/llama-kv-cache.cpp:1501`),
so `q8_0/f16` (K quantized, V full) is the classic "half the memory savings,
no FA needed" configuration.

This fork's `kv_cache_type` option therefore accepts both forms:

- `PARAMETER kv_cache_type f16` — one type for both caches (env parity);
- `PARAMETER kv_cache_type q8_0/f16` — `K/V` pair, mapped onto the two flags.

Both halves of a pair are validated against the allowlist; any invalid half
falls back to the env value as a whole. The pair syntax also enables the
attribution experiment for the qwen finding: `q8_0/f16` vs `f16/q8_0`
determines whether the ~6× reasoning degradation comes from K or V
quantization.

## Caveats

- Quantized **V**-cache requires flash attention in llama.cpp; the value is
  forwarded unchecked, matching the existing env behavior. With
  `OLLAMA_FLASH_ATTENTION` off, quantized types fail the same way they would
  via the env.
- Case toggling between requests ("F16" vs "f16") triggers a spurious reload
  (DeepEqual compares the raw option string); harmless.
- Go-only change — fits the overlay image recipe with the llama-server
  payload byte-equality proof intact.

## Recommended deployment (gfx1151)

Prod keeps `OLLAMA_KV_CACHE_TYPE=q8_0`; qwen3.6 reasoning models get
`PARAMETER kv_cache_type f16`. KV cost at 32,768 ctx: ~3 GB → ~6 GB.

## Attribution results (2026-08-02, run via this feature)

The K/V pair syntax ran the attribution experiment on the release lineage
(`258534eb` cherry-pick, native b9888 binary, env q8_0, per-request overrides;
raw log `vision-suite/runs/kv-attrib-2026-08-02.log`). Thinking tokens to
natural convergence, qwen3.6 think-on, generate, temp 0:

| prompt | f16/f16 | q8_0-K/f16-V | f16-K/q8_0-V | q8_0/q8_0 |
|---|---|---|---|---|
| scene | 3,320 (IoU .927) | 2,488 (.969) | 4,472 (.958) | 19,160 |
| multi | 9,050 | 6,687 | 3,655 | >131,072 (never) |

**Neither cache alone causes the inflation — only both quantized.** The
effect is superadditive: each single-sided arm converges promptly (with
equal-or-better box quality than pure f16), so the reasoning chain tolerates
one noise source but not the combined error. Operationally, **`kv_cache_type
q8_0/f16` is the sweet spot for qwen reasoning**: half the KV memory of f16,
no flash-attention requirement (V stays f16), and no inflation on either the
mild or the severe prompt. Pure f16 remains the conservative default.

Live canary (same log): env default honored, option override reloads the
runner (`--cache-type-k f16 --cache-type-v f16` observed), pair splits
correctly (`--cache-type-k q8_0 --cache-type-v f16`), invalid value warns
once and falls back to env. The first canary attempt caught a real defect —
an incomplete cherry-pick that silently dropped the pair-syntax commit —
before anything shipped.
