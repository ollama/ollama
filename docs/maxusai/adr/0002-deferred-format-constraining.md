# ADR 0002: Defer `format` constraining for implicit-thinking generates via a stop-split continuation, not lazy grammars

- **Status:** accepted, validated 2026-08-01 on fork `main` (b10091 runners) and
  `release/0.32.1-dynres` (b9888+002 runners). Full evidence in
  [generate-think-format-empty-response.md](../generate-think-format-empty-response.md);
  the resulting contract is specified in
  [SPEC: structured output combined with thinking](../spec/structured-output-with-thinking.md).
- **Date:** 2026-08-01
- **Deciders:** MaxusAI fork maintainers

## Context

With thinking enabled and `format` set, `/api/generate` applied the format grammar
from token 0. Implicit-thinking models (parsers `nemotron-3-nano`, `qwen3.5`) can then
never emit their think-close marker, so the parser classifies the entire constrained
output as thinking and `response` is empty. Upstream fixed chat only (#12460
double-request, v0.12.4); generate was never fixed in any released version. We need
reasoning to run unconstrained while the answer is still grammar-forced, on both the
b10091-pinned `main` and the b9888-pinned release branch, for `format:"json"` *and*
JSON-schema formats, streaming included.

## Decision

Split a format-constrained completion for an implicit-thinking generation into two
passes over the same prompt inside the llama-server runner
(`llm/llama_server.go`):

1. unconstrained pass with the parser's think-close marker added as a stop string;
2. if (and only if) generation stopped on that marker (`stopping_word`), a
   constrained continuation of `prompt + emitted thinking + marker` with the
   grammar/schema applied eagerly. `cache_prompt:true` makes the continuation prefill
   a KV-cache hit; the marker is injected into the callback stream so parsers close
   thinking; final metrics merge both passes.

Parsers opt in through `parsers.ImplicitThinkingParser` (marker exposed after
`Init`); generic template models participate when the prompt prefills the opening
tag. A reclassification safety net in `GenerateHandler` (non-streaming: `format`
active + `done_reason:"stop"` + empty response + valid-JSON thinking → thinking
becomes the response) covers runners without the deferral. `/api/chat` keeps its
existing double-request mechanism.

## Alternatives considered

1. **llama-server lazy grammars (`grammar_lazy` + `grammar_triggers`)** — the
   initially preferred mechanism; implemented and rejected on evidence. A PATTERN
   trigger activates on a regex over generated text and feeds the grammar from the
   first non-empty capture group — necessarily a *freely sampled* character. When the
   model's first post-`</think>` character isn't valid for the grammar root,
   `llama_grammar_accept_str` throws and the slot dies (reproduced with a schema
   format: empty response, generation cut at the marker). The robust form wraps the
   grammar root to accept the marker itself (upstream's tool-call pattern), which
   works for our own `grammarJSON` but is impossible for `format:{schema}`: the
   schema→GBNF conversion happens inside llama-server and the Go side has no
   converter (CGO `SchemaToGrammar` was removed with the engines). A half-lazy
   (json-only) + half-something-else design was rejected for mechanism duplication.
2. **Port the chat double-request to generate** — re-renders the prompt through
   template/renderer machinery with an assistant thinking message; correctness then
   depends on every affected model's template disambiguating "continue thinking" vs
   "answer now" (gpt-oss needs an explicit prefill hack even in chat). The stop-split
   achieves the same semantics by pure text continuation, template-free.
3. **Parser-side reclassification only** — trivially small, but the model never
   actually reasons (the grammar suppresses thinking from token 0), so it fixes the
   field mix-up while silently destroying the quality benefit thinking was enabled
   for. Kept as the safety net, not the fix.
4. **Status quo + document `think:false` for JSON** — the routing policy already said
   this; it forfeits reasoning quality for structured extraction and leaves the API
   returning nonsense (`response:""` with JSON in `thinking`).

## Consequences

- `format` + thinking now works on `/api/generate` for nemotron3 and qwen3.5/3.6,
  streaming and non-streaming, `"json"` and schema formats; thinking quality is
  preserved (reasoning runs with no grammar).
- Two `/completion` requests per such generate; the second's prefill is a prompt-cache
  hit (~1–2 tokens). Sampler state (RNG) restarts for the continuation — same
  property as chat's double-request; irrelevant at temperature 0.
- Long thinking can fill the context window before the continuation runs; the runner
  pre-checks and ends as a length-limited thinking-only result rather than erroring
  (`925a669a`). Reported metrics count each token once (prompt eval from pass one,
  generated tokens summed) — thinking tokens appear in `eval_count`, never in
  `prompt_eval_count`.
- Validation surfaced a quality datum worth keeping: with reasoning enabled through
  this path, nemotron's localisation improved markedly (scene bbox center-hits 5/6
  vs 0–1 in every think:false run; invoice name-bbox 5/5) — reasoning helps spatial
  grounding, so think:on is now a quality *option* for extraction, not a forbidden
  mode.
- The mechanism needs only stop strings + `stopping_word` + `cache_prompt`: no
  llama.cpp feature dependency, so the identical commit cherry-picks to b9888-era
  branches (done: `d1ef5557` on `release/0.32.1-dynres`).
- Prompts truncated to token arrays by context shift can't be continued textually and
  keep the old eager behavior (plus safety net) — a pre-existing degraded edge.
- If a model ends thinking with a tool-call tag instead of the marker (nemotron can),
  generate has no tools so this doesn't arise; revisit if generate ever grows tools.
- Upstream ollama shares the bug through v0.32.5; the fix is upstreamable as-is
  (issue draft in the companion doc).
