# ADR 0009: Constrain MLX generation with a pure-Go JSON grammar, reject what it cannot constrain

- **Status:** accepted (2026-08-07)
- **Date:** 2026-08-07
- **Deciders:** MaxusAI fork maintainers

## Context

`llm.CompletionRequest.Format` is never forwarded by the MLX runner client
(`x/mlxrunner/client.go`): the wire `CompletionRequest` has no `Format` field, so
`/api/chat` and `/api/generate` with `format:"json"` or a JSON Schema against
safetensors/MLX models return unconstrained text (verified live 2026-08-07 on
`gemma4:12b-nvfp4`; models often wrap JSON in markdown fences). The llama-server
path grammar-constrains the same requests. Silently ignoring a load-bearing
request field violates the fork's no-silent-drop principle.

Constraints on the solution space:

- The fork removed the CGO engines (upstream #16031); llama.cpp's grammar engine
  (`llama-grammar.cpp`, 1471 lines) and schema converter
  (`json-schema-to-grammar.cpp`, 1153 lines) are no longer in the binary. GGUF
  models delegate both to the external llama-server process; MLX models have no
  such process to delegate to.
- [SPEC: structured output combined with thinking](../spec/structured-output-with-thinking.md)
  puts the thinking/format split at the routes layer (ADR 0004). The runner-side
  contract is only R9: constrain from the first token whenever `format` is set on
  the request the runner receives. No thinking awareness is needed in the runner.
- The MLX runner samples on the GPU (`x/mlxrunner/sample`), pipelined one token
  ahead, with optional speculative decoding.

## Decision

Implement constrained sampling for the MLX path in pure Go, and return an
explicit HTTP 400 for any `format` the implementation cannot faithfully
constrain. Never silently ignore a constraint.

1. **New package `x/structured`** (no MLX or cgo dependency):
   - `Compile(format json.RawMessage) (*Grammar, error)` accepts `"json"` or a
     JSON Schema object and produces a byte-level grammar IR. `"json"` ports the
     `grammarJSON` GBNF used by the llama-server path (root is an object). Schemas
     port the semantics of llama.cpp b10091 `json-schema-to-grammar` (the version
     the fork pins), including its fixed property order: required and optional
     properties are emitted in schema declaration order, and objects with
     `properties` and no `additionalProperties` are closed.
   - A matcher advances the grammar byte by byte using sets of pushdown stacks
     (the same formalization as `llama-grammar.cpp`, over bytes instead of
     codepoints). Alternation forks stacks; rule references push frames, so
     recursive `$ref`s work. References in tail position reuse the parent
     continuation instead of pushing — star recursions (string content,
     whitespace) are tail calls, so those states are genuine self-loops.
     Without the collapse the stack grows one frame per consumed byte, mask
     caching never hits inside strings, and long generations decay
     quadratically (observed live: ~160 ms/token at 900 characters, ~20
     ms/token — model-bound — after).
   - A vocabulary trie (token id → decoded bytes) computes the per-step
     allowed-token mask by walking the trie under the matcher; masks are
     memoized by canonical matcher-state key (in-string and in-whitespace states
     recur constantly, so the hit rate is high). EOS tokens are allowed exactly
     when the grammar can complete; non-EOS special tokens and empty-piece
     tokens are never allowed.
2. **Runner integration (`x/mlxrunner`)**:
   - The wire `CompletionRequest` gains a `Format` field; `client.go` forwards
     `llm.CompletionRequest.Format` verbatim. A non-empty
     `llm.CompletionRequest.Grammar` (raw GBNF, llama-server-only) is rejected
     with 400 — nothing sets it today, but if something starts to, it must not
     be dropped.
   - The runner HTTP handler compiles `Format` at admission and returns 400 with
     the compile error before any generation starts.
   - A constrained decoder replaces the pipelined decoder for constrained
     requests: the next forward pass is dispatched on the GPU, then the CPU
     reads the previous token, advances the matcher, and builds the logit bias
     (0 for allowed, −∞ for disallowed) while the forward runs; the bias is
     added to the logits before `Sampler.Sample`. Penalties, temperature,
     top-k/top-p/min-p compose correctly with −∞ logits. Speculative decoding is
     parked for constrained requests (draft KV maintained, no proposals), the
     same treatment logprobs requests already get.
3. **Scope of schema support.** The converter is a function-by-function port
   of b10091's `common_schema_converter` (the deleted
   `json-schema-to-grammar.cpp` recovered from git history), so the supported
   set is b10091's, not a subset invented here:
   - Supported: `type` (including type arrays), `properties` + `required` with
     declaration-order emission, `additionalProperties` (absent/false =
     closed; true or schema = extra pairs whose keys exclude the declared
     names, ported trie construction included), `items` (schema or tuple
     array) and `prefixItems`, `minItems`/`maxItems`, `enum`, `const`,
     `anyOf`/`oneOf`, `allOf` (property/enum merge, `anyOf`-nested components
     optional), `$ref` to any internal `#/` pointer (recursion allowed),
     `minLength`/`maxLength` (with `type:"string"`), integer
     `minimum`/`maximum`/`exclusiveMinimum`/`exclusiveMaximum` (ported
     digit-decomposition), string `format` `date`/`time`/`date-time`/`uuid`,
     empty schema or bare `object` = free-form object.
   - Ignored exactly where b10091 ignores them (annotation semantics):
     `title`, `description`, `default`, `examples`; numeric bounds on
     `number` (only `integer` is enforced); `minLength` without
     `type:"string"`; unknown sibling keywords next to a recognized
     structural keyword (e.g. `if`/`then` beside `type:"object"`).
   - Rejected with 400, naming the reason: `pattern` (b10091 compiles
     regexes; this port does not — the one deliberate feature cut),
     external (`http(s)://`) `$ref`s (llama-server also refuses to fetch at
     runtime), and any schema b10091 itself errors on ("unrecognized
     schema"), plus any `format` value that is neither `"json"` nor a JSON
     object, and raw `Grammar`.
   - Byte-level relaxation: inside JSON strings any byte ≥ 0x20 except `"` and
     `\` is accepted, so the grammar does not itself verify multi-byte UTF-8
     well-formedness (llama.cpp matches codepoints). Tokenizer pieces are
     almost always valid UTF-8, and the decode path already buffers partial
     sequences (`utf8_buffer.go`). Enum/const literals reproduce the schema's
     own number formatting (source text) rather than nlohmann's re-serialized
     dump; property-name literals and dumps match for ASCII names.

4. **Two adjacent silent drops fixed in passing** (live verification of the
   ADR 0004 flow on MLX exposed both):
   - `Options.Stop` was ignored by the MLX runner entirely. The routes layer's
     marker flow injects the think-close marker as a stop string, and user
     stop sequences are documented API. The decode loop now runs the stream
     through a stop matcher (truncate before the match, hold back partial
     suffixes, `done_reason:"stop"` on match, flush held text at end).
   - The MLX client kept delivering already-buffered response chunks after
     the caller cancelled the request context. The structured-outputs
     transition flow cancels pass one at the thinking→content boundary, and
     the buffered chunks leaked pass-one content into the final response.
     The scanner loop now checks the context before each callback, as the
     llama-server client does.

## Alternatives considered

- **400-only floor** (reject every `format` on the MLX path). Honest but drops a
  flagship API feature for the model class this fork treats as first-class; the
  routes-layer think+format machinery would be dead code for MLX models.
  Retained only as the behaviour for the unsupported subset.
- **Reintroduce llama.cpp grammar via cgo.** Reverses the deliberate CGO-engine
  removal (#16031); couples the MLX runner to llama.cpp builds again.
- **Delegate grammar to a llama-server sidecar per request.** A second process
  and a vocabulary round-trip per token for a feature that is CPU-trivial
  relative to decode; operationally absurd.
- **Rejection sampling (sample, test against the matcher, resample).** No mask
  computation, but worst case is unbounded GPU round-trips per token in tight
  grammar states (after `{` only a handful of bytes are legal); the mask is one
  round-trip always.

## Consequences

- `format:"json"` and supported schemas produce grammar-valid output on MLX
  models from the first token; combined with thinking, the ADR 0004
  routes-layer split applies unchanged (pass one arrives format-free, pass two
  constrained).
- Unsupported schemas fail fast with a named reason instead of degrading into
  prose-wrapped JSON.
- Constrained decode gives up token-level sampling pipelining but overlaps mask
  computation with the forward pass; the first constrained request per loaded
  model pays a one-time vocabulary index build. Measured on M5 Max with a
  260k-piece vocabulary: ~12 ms for a cold mask in the widest (in-string)
  state, ~170 ns warm; a representative object generation averages well under
  1 ms/token amortized. Cold masks hide under a 12B model's per-token forward
  pass.
- A pure-Go grammar engine enters the tree (~2k lines with tests) that must
  track the pinned llama.cpp version's converter semantics when the pin moves.

## Conformance

- `x/structured` unit tests: matcher acceptance against `encoding/json` ground
  truth, schema compilation fixtures (property order, closed objects, enum,
  `$ref` recursion, rejection list), mask correctness on a synthetic vocabulary
  (step-by-step allowed sets, EOS gating), and a greedy-walk test that any
  masked path terminates in schema-valid JSON.
- Wire tests: `Format` forwarded by the client, 400 surfaced for raw `Grammar`
  and for unsupported schemas.
- Manual probe (spec §5), performed 2026-08-08 on `gemma4:12b-nvfp4` at
  temperature 0: `/api/generate` with `format:"json"` (`think:false`) returns
  a pure JSON object with `done_reason:"stop"`; with `think:true` the
  transition flow separates 1.2k characters of `thinking` from a valid
  constrained `response`; a JSON Schema with bounded integers and `maxItems`
  produces declaration-ordered, range-respecting output; `/api/chat` with
  `stream:true` streams only constrained chunks; `/v1/chat/completions` with
  `response_format json_schema` conforms; a `pattern` schema returns HTTP 400
  naming the keyword; an unconstrained request is byte-identical in behaviour
  to before. The GGUF twin (`gemma4:12b-it-q4_K_M`, llama-server path)
  produces the same shape for the same schema request.
- Known residual: the transition flow reports pass-two-only `eval_count`
  (routes-layer behaviour shared with the llama-server path, R6 conformance
  gap for the transition flow generally; the marker flow sums correctly).
