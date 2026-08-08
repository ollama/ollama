# SPEC: structured output combined with thinking

MaxusAI-fork specification. Status: **implemented** (fork `main` lineage and
`release/0.32.1-dynres`). Written 2026-08-02. Both runner backends satisfy
the runner-side obligations: llama-server natively, the MLX runner since
2026-08-08 ([ADR 0009](../adr/0009-mlx-pure-go-constrained-sampling.md) —
constrain-from-first-token, stop strings, and post-cancel chunk discipline).

Normative contract for requests that set `format` **and** run a model that emits
reasoning. Rationale and history live in
[generate-think-format-empty-response.md](../generate-think-format-empty-response.md);
the design decision is [ADR 0002](../adr/0002-deferred-format-constraining.md). This
document states only what the implementation must do, so it can be verified or
re-implemented without reading either.

Key words MUST / MUST NOT / SHOULD follow their usual RFC-2119 sense.

## 1. Scope

Applies when a request sets `format` to `"json"` or a JSON Schema object, and the
model either has a builtin parser with thinking support or a template whose thinking
tags `thinking.InferTags` recognises.

Two model shapes matter:

- **implicit-open** — generation begins *inside* thinking; no opening marker is
  emitted; thinking ends at a close marker (`</think>`). Parsers `nemotron-3-nano`,
  `qwen3.5`, and any template model whose prompt prefills the opening tag.
- **explicit-open** — the model emits an opening marker first, so a grammar that
  excludes it simply suppresses thinking.

The failure this contract exists to prevent is specific to implicit-open models: a
grammar applied from the first token makes the close marker unreachable, so the model
can never leave thinking and the whole grammar-shaped answer is classified as
reasoning.

## 2. Requirements

**R1 — Reasoning MUST be unconstrained.** No grammar or schema may restrict sampling
while the model is producing reasoning. A conforming implementation MUST NOT rely on
a grammar whose root merely *tolerates* the reasoning span if that grammar can still
reject a freely sampled reasoning token.

**R2 — The response MUST satisfy the format.** Once reasoning has ended, generation
MUST be constrained so the emitted content conforms to `format`.

**R3 — Fields MUST be separated.** Reasoning MUST be reported in `thinking`
(`message.thinking` on chat) and the constrained answer in `response`
(`message.content`). A completion that satisfies the format MUST NOT be reported as
thinking.

**R4 — Marker semantics MUST match the parsers.** The boundary between reasoning and
content is the **first textual occurrence** of the close marker, whether the model
emits it as a special token or spells out its characters. Constraining MUST begin at
exactly the boundary the parser uses; the two MUST NOT be able to disagree.

**R5 — Termination MUST be honest.** If reasoning consumes the token budget or the
context window, the request MUST end with `done_reason: "length"` and MUST NOT report
`"stop"`. Exhaustion MUST NOT surface as a transport or runner error after content
has already been streamed.

**R6 — Metrics MUST count each token once.** `prompt_eval_count` MUST report the
tokens of the request's own prompt. Reasoning tokens MUST appear in `eval_count` and
MUST NOT also appear in `prompt_eval_count`, even when an implementation re-submits
reasoning as prompt text internally. `eval_count` MUST be the total generated across
any internal passes.

**R7 — Streaming order MUST be preserved.** Reasoning MUST stream to the client as it
is produced, before any content. The close marker MUST reach downstream parsers so
they transition states at the right point.

**R8 — Endpoint behaviour MUST be uniform.** All endpoints MUST satisfy R1–R7:

| endpoint | handler | mechanism |
|---|---|---|
| `/api/chat` | `ChatHandler` | double request (upstream #12460) |
| `/v1/chat/completions`, `/v1/responses`, `/v1/messages` | `ChatHandler` | as above |
| `/api/generate` | `GenerateHandler` | deferred constraining (ADR 0002) |
| `/v1/completions` | `GenerateHandler` | as above |

**R9 — Non-thinking paths MUST be unchanged.** With thinking disabled or absent, or
with no `format`, generation MUST behave exactly as it did before: a single
completion, constrained from the first token when `format` is set.

## 3. Implementation at the routes layer

Satisfied by a two-request split in `GenerateHandler`/`ChatHandler` (ADR 0004,
originally a runner-layer split per ADR 0002): pass one strips `format` and, when
the model's close marker is known (`parsers.ImplicitThinkingParser`, or the
generic thinking parser with a prefilled opening tag), carries the marker as an
extra stop string; on `done_reason:"stop"` with no parsed content, pass two
continues `prompt + reasoning + marker` (generate) or the re-rendered prompt
with the reasoning as an assistant message (chat) with the grammar applied
eagerly. Prompt caching makes the second prefill a cache hit in the textual
form. Models without a known marker use the thinking→content transition with a
re-rendered continuation (upstream #12460/#14288 flow).

Requirements on that implementation:

- The second pass MUST NOT carry the marker as a stop string (R2: a JSON string value
  containing the marker must not truncate the answer).
- Before the second pass, the continuation prompt MUST be checked against the loaded
  runner's context length (request options may report 0 = auto); if it does not fit,
  the request ends per R5.
- On `/api/chat`, requests carrying tools MUST NOT use the marker stop (it would
  preempt a tool call following the marker); they keep the transition flow.

The split places two obligations on every runner backend: `format` on the
request the runner receives MUST constrain from the first token (R9), and
`Options.Stop` MUST truncate the stream at the first textual match without
emitting the stop text (the marker stop of pass one). The transition flow
additionally requires that a runner client deliver no further chunks once the
handler cancels the request context — buffered pass-one content past the
cancellation would leak into the constrained response.

## 4. Defensive reclassification

For flows the double request does not cover, a non-streaming generate response
that has `format` active, `done_reason: "stop"`, an empty response, and thinking that
is itself valid JSON MUST be reclassified so the thinking becomes the response. This
is a safety net for R3 only; it does not satisfy R1, because such a generation was
constrained throughout.

## 5. Conformance

Automated (`server/routes_generate_test.go`):

- `TestGenerateThinkFormatMarkerFlow` (+`Streaming`) — pass one format-free and
  marker-stopped, pass two's continuation prompt and format, streamed order
  (R1, R2, R4, R7), metrics (R6).
- `TestGenerateThinkFormatContextFull` — R5.
- `TestChatThinkFormatMarkerStop` — the chat marker flow and metrics (R6).
- `TestChatThinkFormatLengthNoContinuation` — budget exhaustion inside
  reasoning (R5).
- The pre-existing chat `structured outputs restart` tests — the transition
  fallback.
- `Test{Nemotron3Nano,Qwen35}Parser*ThinkingCloseMarker` — marker exposure (R4).
- `TestReclassifyConstrainedThinking` — §4.

Manual probe (any implicit-open model, temperature 0): `/api/generate` with
`think:true` and `format:"json"` must return reasoning in `thinking` and parseable
JSON in `response`, with `done_reason:"stop"`; repeat with a JSON Schema, with
`stream:true`, and against `/api/chat` and `/v1/chat/completions`.

MLX runner probe (2026-08-08, `gemma4:12b-nvfp4`, explicit-open → transition
flow): `format:"json"` and JSON Schema requests return constrained JSON on
`/api/generate`, `/api/chat` (streaming included), and `/v1/chat/completions`;
`think:true` separates reasoning from a valid constrained response; budget
exhaustion reports `done_reason:"length"` (R5); the GGUF twin
(`gemma4:12b-it-q4_K_M`) produces the same shapes for the same schema. Full
matrix and runner-side mechanism in ADR 0009.

## 6. Non-goals

Tool calls combined with `format` on `/api/generate` (generate has no tools);
guaranteeing that a model's reasoning is *useful*; and constraining reasoning itself
to any grammar — R1 forbids it.
