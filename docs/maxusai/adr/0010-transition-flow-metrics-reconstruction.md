# ADR 0010: Reconstruct cancelled-pass metrics in the transition flow

- **Status:** accepted, implemented + live-validated 2026-08-08 on fork `main`
  lineage. Amends the transition-flow metrics behaviour of
  [ADR 0004](0004-routes-layer-think-format-double-request.md); 0004's marker
  flow is unchanged.
- **Date:** 2026-08-08
- **Deciders:** MaxusAI fork maintainers

## Context

ADR 0004's double request satisfies spec R6 in the marker flow because pass
one terminates at a stop string: the runner's final chunk arrives and its
metrics feed the `pass1` summing. The transition flow (no known think-close
marker, e.g. gemma4) instead **cancels** pass one at the parsed
thinking→content transition — and both runner clients deliver metrics only on
a completion's final chunk (llama-server timings ride the stop event,
`llm/llama_server.go`; the MLX pipeline's final response,
`x/mlxrunner/pipeline.go`). The cancel discards that chunk, so the response
forwarded pass two raw: `prompt_eval_count` was the continuation's
cache-inclusive prefill (reasoning re-counted as prompt) and `eval_count`
omitted every reasoning token. Verified live on gemma4:12b-nvfp4 (MLX):
`eval_count` 5 with ~300 reasoning tokens — violating R6 in both directions.

## Decision

Reconstruct pass-one metrics at the routes layer, at the restart point, in
the textual form — `transitionPassMetrics()` in `server/routes.go`, shared by
both handlers:

- `prompt_eval_count`: derived at done time as pass two's runner-reported
  prefill minus the pure-text token delta between the continuation prompt
  and the request's own prompt. Both runners report prefill cache-inclusive
  (llama-server timings `cache_n + prompt_n`; the MLX pipeline
  `len(request.Tokens)`), so the reported count carries image-embedding
  tokens, and the two tokenized prompts differ only in appended text (image
  placeholders cancel) — the subtraction therefore recovers the request's
  own prompt cost *including* its image tokens, which no text tokenization
  can count. The restart also tokenizes the request's own prompt as a
  textual stand-in: it serves the exits pass two never reaches (context
  full) and any degenerate pass-two report (zero, or smaller than the
  delta).
- `eval_count`: the raw pass-one stream (thinking plus any leaked content
  fragment), tokenized. Every generated token counts once per R6; the
  discarded fragment was genuinely generated, so it counts.
- durations: wall-clock split at the first streamed chunk — prefill before
  it, decode after.

The synthesized value feeds ADR 0004's existing `pass1` summing, so both
flows converge on identical done-time logic. Reconstruction is best-effort:
on a tokenize error the response falls back to pass-two-only metrics (the
prior behaviour). The `pass1 == nil` guard also covers chat's
marker-armed-but-leaked fallback, and running before the context pre-check
gives length-limited transition results pass-one counts instead of zeros.

Alternatives rejected:

- **Runner protocol change (per-chunk running counters).** llama-server
  partial chunks carry no timings without `timings_per_token` overhead, and a
  mid-stream counter would leak into every streamed API chunk through the
  handlers' shared metrics copy.
- **Counting streamed chunks.** Chunk ≠ token (UTF-8 buffering in both
  runners' detokenizers) and it embeds a protocol assumption tokenization
  avoids.
- **Status quo (raw forwarding, upstream #12460/#14288 shape).** Fails R6 by
  the entire reasoning span.

## Consequences

- Counts are R6-correct on both endpoints and both engines. Live (patched
  build, temp 0): MLX gemma4:12b-nvfp4 → eval 776 / prompt 31 (was eval 5);
  llama-server gemma4:12b-it-q4_K_M → eval 371 / prompt 31 with valid
  constrained JSON. The reconstructed eval rate (~111 tok/s) matches the
  model's measured ~113 tok/s warm decode.
- `eval_count` is a textual-form approximation, not a runner token identity:
  re-tokenization can differ by a token or two at chunk boundaries. The same
  caveat applies to `prompt_eval_count` only at the continuation seam and in
  the fallback cases; the prefill-minus-delta derivation otherwise counts in
  the runner's own token space, image-embedding tokens included (a vision
  request no longer understates by ~2042 tokens per nemotron3 image or 256+
  per gemma4 image).
- Conformance: `Test{Generate,Chat}ThinkFormatTransitionMetrics`
  (`server/routes_generate_test.go`, spec §5) — vision-shaped: the mock
  runner reports cache-inclusive prefills carrying an image-token surplus,
  and the tests require it in the final `prompt_eval_count`.
