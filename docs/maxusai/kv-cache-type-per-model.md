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
