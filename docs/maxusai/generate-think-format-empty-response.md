# `/api/generate` + `think` + `format` returned an empty response — root cause and fix

MaxusAI-fork investigation and fix (2026-08-01). Companion to
[ADR 0002](adr/0002-deferred-format-constraining.md) (the original runner-layer
decision), [ADR 0004](adr/0004-routes-layer-think-format-double-request.md)
(the 2026-08-02 move to the routes layer — the current architecture, described
in [its own section below](#the-fix-v2--routes-layer-double-request-2026-08-02)),
and [SPEC: structured output combined with thinking](spec/structured-output-with-thinking.md)
(the normative contract this behaviour must satisfy). Fix branches:
`fix/generate-think-format-lazy-grammar` (runner layer, 2026-08-01) then
`fix/generate-think-format-routes-layer` (routes layer, 2026-08-02), both off
`main`, cherry-picked to `release/0.32.1-dynres`.

> **The one thing to take away:** with thinking enabled and `format` set,
> `/api/generate` constrained the model with the format grammar from the very first
> token. Implicit-thinking models (nemotron3, qwen3.5/3.6) can then never emit
> `</think>`, so their marker-based parsers classify the entire — perfectly valid —
> JSON output as `thinking`, and `response` comes back empty. `/api/chat` has had a
> chat-only workaround since v0.12.4, which is why OpenWebUI / Python `ChatOllama`
> users never saw it. It is **not a regression**: no released version ever got this
> right on `/api/generate`, on either the old Go engine or the llama-server runner.

## Symptom

`nemotron3:33b-q4_K_M`, `/api/generate`, `think: true`, `format: "json"` (probe
2026-08-01, fork `main` @ `3f6ea735`, temperature 0):

```json
{"response": "", "thinking": "{\"capital\": \"Paris\"}", "done_reason": "stop", "eval_count": 7}
```

The complete, correct JSON answer lands in `thinking`; `response` is empty. Identical
mechanism for `qwen3.6:35b-a3b-q4_k_m` (parser `qwen3.5`). The ground-truth vision
suite hit the same wall, which is why its whole grid was recorded at `think:false`
("gray" think-on cells) and the routing policy said "serve JSON extraction with
`think:false`".

## Mechanism

Three pieces interlock:

1. **Eager grammar.** `llm/llama_server.go` translated `format` into a GBNF grammar
   (`format:"json"` → `grammarJSON`) or a `json_schema` on the llama-server
   `/completion` request. llama-server applies that constraint from token 0.
2. **Grammar excludes the markers.** A JSON grammar admits no `<think>`/`</think>`
   tokens anywhere, so the model cannot ever close (or open) a thinking block.
3. **Implicit-open parsers.** `nemotron-3-nano` and `qwen3.5` parsers start every
   generation in *collecting-thinking* state when thinking is enabled and only leave
   it on `</think>` (or `<tool_call>`). The marker never arrives → everything the
   grammar forced out is classified as thinking.

Explicit-open models (e.g. qwen3's generic template parser waiting for `<think>`) show
the inverse artifact: the JSON arrives in `response` but thinking is silently skipped —
same root cause, different visibility.

## Why `/api/chat` was fine (and OpenWebUI/ChatOllama memories are correct)

Upstream ollama fixed this for chat only, in `77060d46` "routes: structured outputs for
gpt-oss (#12460)" (2025-10-08, first in v0.12.4): `ChatHandler` sends the first
completion **without** the format, watches the parser for the thinking→content
transition, then cancels and re-issues a second completion with the format applied and
the collected thinking prepended as an assistant message
(`structuredOutputsState` machine, `server/routes.go`). OpenWebUI and Python
`ChatOllama` use `/api/chat`, so they always rode this path. Verified live on this
host: same model, same prompt, `/api/chat` returns thinking + JSON correctly on the
unfixed build.

`GenerateHandler` never received the equivalent: it passed `Format: req.Format`
straight into the completion request.

## Version archaeology — this never worked

- **Old Go engine (removed 2026-05-29, `9db4bdba`, upstream #16031):** `llm/server.go`
  converted `format:"json"` → `req.Grammar = grammarJSON` and the runner's sampler
  (`sample.NewGrammarSampler`) applied it from token 0. No thinking-awareness. Same
  failure by construction.
- **`nemotron-3-nano` parser landed 2025-12-15 (`7e3ea813`, v0.13.4)** — during the old
  engine era — and `GenerateHandler` already ran builtin parsers without any format
  deferral. So the very first version able to run nemotron3 already had the bug; the
  qwen3.5 parser (2026-03) inherited it.
- The llama-server runner switch changed nothing about this path; it reproduced
  byte-for-byte the old eager-grammar semantics.
- Independently confirmed on the unfixed 0.32.1 lineage (2026-08-01, parallel
  benchmarking session): the live `maxusai-ollama:0.32.1-rocm-gemma4budget` container
  (b9888, stock 256-token vision path) misfiled all three vision-suite tests at
  think:on + `format:"json"` (eval 486–735, responses empty/JSON-invalid) — the bug is
  lineage-independent, exactly as the code history predicts.
- Local docker archaeology is moot: `ollama/ollama:0.12.6-rocm` predates the affected
  parser class entirely (it cannot run nemotron3), and the reproduction on current
  `main` plus the code history above date the behavior conclusively.

### Upstream state (checked 2026-08-01 via GitHub API)

Upstream knows the problem class and left the generate side unfixed **knowingly**:

- [#10538](https://github.com/ollama/ollama/issues/10538) (open since 2025-05):
  umbrella "structured outputs for reasoning models". Maintainer ParthSareen
  (2025-05-16): "In an ideal world we can have thinking enabled and then do
  structured outputs after that portion."
- [#11691](https://github.com/ollama/ollama/issues/11691) (open, 87 comments,
  gpt-oss): ParthSareen (2025-08-12) sketches the intended endgame — "bring some of
  the token parsing down to the runner level… grammar sampling and token parsing
  closer together to know when to start the constrained sampling". That is
  runner-level deferred constraining, i.e. the shape of our fix.
- [PR #12460](https://github.com/ollama/ollama/pull/12460) (merged 2025-10-09,
  v0.12.4): the chat double-request. Its own description records the decision:
  "**does not update generate handler for now**".
- [PR #14288](https://github.com/ollama/ollama/pull/14288) (open since 2026-02-17,
  `veeceey`): ports the double-request to `/api/generate`, citing
  [#11691](https://github.com/ollama/ollama/issues/11691). **Zero maintainer reviews
  in ~6 months**; author bumped it 2026-03-10.
- The think:**false** twin bug (`format` silently ignored when thinking is disabled)
  ran a separate arc: [#15260](https://github.com/ollama/ollama/issues/15260) fixed
  gemma4-only in [PR #15678](https://github.com/ollama/ollama/pull/15678) (v0.21.1,
  ParthSareen: "the cases for other models may vary, so a long term solution must be
  added on a parser level"), then
  [#14645](https://github.com/ollama/ollama/issues/14645) (qwen3.5/3.6) fixed
  generally in [PR #15901](https://github.com/ollama/ollama/pull/15901) (merged
  2026-07-07, v0.31.2). Our fork's `forceImmediate` chat logic and working
  think:false probes are downstream of that arc.
- Upstream `main`'s `GenerateHandler` still has no deferral as of 2026-08-01
  (verified against raw `server/routes.go`: `structuredOutputsState` appears only in
  `ChatHandler`). Users in #14645/#10538 report migrating to llama.cpp over this.

Related but distinct threads:
[#10929](https://github.com/ollama/ollama/issues/10929) (invalid JSON under
thinking+format), [#15288](https://github.com/ollama/ollama/issues/15288) (gemma4 on
the OpenAI endpoint), [#10976](https://github.com/ollama/ollama/issues/10976)
(thinking+tools), and the recurring ask to expose raw GBNF
([PR #12055](https://github.com/ollama/ollama/pull/12055), open unmerged; earlier
GBNF requests #6237/#11911/#5899 went nowhere) — users cite exactly this bug class
when asking for it.

Upstream move: support/review [PR #14288](https://github.com/ollama/ollama/pull/14288)
rather than filing a duplicate; our stop-split is an alternative implementation worth
offering there (single mechanism for `"json"` + schema, no template re-render, works
for raw-prompt generates where the double-request's re-render does not apply).

**Upstream port verified (2026-08-01):** commit `928c7494` cherry-picks *cleanly* onto
`ollama/ollama` main (`8d8c701d`) — branch `feat/upstream-generate-think-format`,
local commit `2aff0a70`. Builds, all `model/parsers` + `llm` + `server` tests pass,
and (upstream main pins the same llama.cpp `b10091` as our fork) a live E2E against
the fork-built runners returns thinking + correct JSON for both `format:"json"` and
schema formats on nemotron3. The offer on #14288 can point at a ready branch.

### How llama.cpp itself avoids the bug (and why users migrating there are happy)

llama-server's own chat endpoint (`/v1/chat/completions`) never applies a bare format
grammar. Its auto-parser layer builds a **format-aware composite grammar** from the
chat template: with `response_format` set, the response parser is literally
`reasoning-block + whitespace + schema-JSON` and the reasoning rule is
`optional("<think>" + until("</think>") + "</think>")`
(`common/chat-auto-parser-generator.cpp:139-144,167-168` in the pinned tree). The
single eager grammar's root therefore *admits the entire thinking span* — reasoning
is only "constrained" to eventually close its tag — and hard constraining begins at
the JSON. One request, no double-pass; the server also splits `reasoning_content`
from `content` server-side. (Marker semantics are the same as ours: the reasoning
rule ends at the first textual `</think>`.)

Ollama cannot simply adopt that because it deliberately bypasses llama-server's chat
layer: it renders prompts with its own Go templates/renderers (jinja disabled for
parser models), calls raw `/completion`, and parses thinking/tools Go-side. Matching
llama.cpp's trick at the `/completion` level would mean generating an
"anything-until-`</think>`, then format" GBNF in Go — expressible for our own
`grammarJSON` (GBNF has no negative lookahead, so `until()` compiles to an awkward
character-class expansion llama.cpp generates programmatically), but impossible for
`format:{schema}` whose grammar is produced inside llama-server. The stop-split is
the pragmatic equivalent one layer down, with the bonus that reasoning runs with no
grammar in the sampler at all.

## The fix v1 — deferred constraining in the llama-server runner (historical)

> **Superseded 2026-08-02** by the routes-layer double request
> ([next section](#the-fix-v2--routes-layer-double-request-2026-08-02),
> [ADR 0004](adr/0004-routes-layer-think-format-double-request.md)). The
> semantics below carried over; the mechanism moved up a layer and the runner
> code was removed. Kept for the mechanism record and because the lazy-grammar
> analysis still applies.

`llm/llama_server.go` ran a format-constrained completion for an
implicit-thinking generation in **two passes over one prompt**:

1. **Pass 1 — think free:** the completion runs *unconstrained* with the parser's
   think-close marker (`</think>`) added as an extra stop string. llama-server stops
   exactly when reasoning ends (`stopping_word` tells us it was our marker).
2. **Pass 2 — constrained continuation:** the same prompt + the emitted thinking +
   the marker is resubmitted with the grammar/schema applied **eagerly**. With
   `cache_prompt: true` the prefill is a KV-cache hit over pass 1, so the second pass
   costs a couple of prompt tokens. The marker is injected into the callback stream
   between passes so the parsers observe the close and switch to content; final
   metrics merge both passes' eval counts/durations.

If pass 1 ends any other way (EOS mid-thinking, `num_predict` exhausted, a user stop
string), there is no continuation and the result is reported honestly.

Robustness + metrics (`925a669a`, from cross-session validation findings): if the
continuation prompt (prompt + thinking + marker) no longer fits `num_ctx`, the
completion ends as a length-limited thinking-only result instead of surfacing
llama-server's rejection as a 500 (pre-checked via `/tokenize`, with a defensive
fallback on a 400). Metrics count every token once: `prompt_eval_count` is pass
one's (the true prompt — the continuation's prefill re-reads pass one's output from
cache and is not re-counted), `eval_count` is the sum of both passes' generated
tokens. Multi-image / long-thinking workloads should still size `num_ctx` for
prompt + thinking + answer.

Wiring: parsers that begin generation inside thinking expose their marker via the new
`parsers.ImplicitThinkingParser` interface (`nemotron-3-nano`, `qwen3.5`);
`GenerateHandler` also participates for generic template models whose prompt prefills
the opening tag (`thinking.InferTags` closing tag). The tag travels as
`llm.CompletionRequest.ThinkCloseTag`. `/api/chat` is untouched (its double-request
mechanism keeps working).

Safety net: `reclassifyConstrainedThinking` in `server/routes.go` — non-streaming
generate responses that come back with `format` active, `done_reason:"stop"`, empty
response and *valid-JSON* thinking are reclassified as response. This covers runners
without the deferral (e.g. MLX, or a llama-server predating stop-string reporting).

### Why not `grammar_lazy` + `grammar_triggers`?

That was the first implementation (the pinned llama-server supports both, verified at
`server-schema.cpp` and in the b9888 libs). It failed structurally:

- A PATTERN trigger feeds the grammar from the **first non-empty capture group** — the
  first *freely sampled* character after `</think>`. If the model doesn't happen to
  emit `{` there, `llama_grammar_accept_str` throws "Unexpected empty grammar stack"
  and the slot dies. Reproduced: schema-format probe returned empty response with the
  reasoning intact and generation cut at 0 constrained tokens.
- The robust variant wraps the grammar root so it accepts the marker itself
  (`root ::= "</think>" ws fmt-root`, trigger group starting at the marker — the
  upstream tool-call pattern). That works for our own `grammarJSON` but **cannot work
  for `format:{schema}`**: the schema→GBNF conversion happens inside llama-server and
  the Go side has no converter to wrap its root (the old CGO `SchemaToGrammar` was
  removed with the engines).

The stop-split needs only stop strings + `stopping_word` + `cache_prompt` — features
b9888 and b10091 both have — and fixes `"json"` and schema formats identically. Full
trade-off record in [ADR 0002](adr/0002-deferred-format-constraining.md).

## The fix v2 — routes-layer double request (2026-08-02)

The mechanism moved from the runner to `server/routes.go`
([ADR 0004](adr/0004-routes-layer-think-format-double-request.md)): upstream
fixed chat at the routes layer (#12460) and the open generate port
([#14288](https://github.com/ollama/ollama/pull/14288)) mirrors it, so that is
the architecture upstream will accept — and it covers every engine, not just
the llama-server runner. The fork's version is a **superset of #14288**: same
state machine, plus the three hardenings the upstream PR lacks.

**GenerateHandler** loops chat-style over up to two completion requests:

1. When the model's think-close marker is known — `parsers.ImplicitThinkingParser`
   (`nemotron-3-nano`, `qwen3.5`) or the generic thinking parser with a
   prefilled opening tag — **pass one strips `format` and injects the marker as
   a stop string** (per-request `Options.Stop` copy; `PreservedTokens` already
   keeps the marker textual). On `done_reason:"stop"` with no parsed content,
   the handler feeds the held-back content plus the marker through the parser
   (thinking closes exactly where it would have), then continues with
   `prompt + raw pass-one output + marker` and the format applied — the same
   textual continuation as v1, now built at the routes layer. An EOS still
   inside thinking takes the same path (recovery the v1 split left to the
   reclassify net); `length` ends honestly with no continuation.
2. Models without a marker (harmony/explicit thinking) use exactly upstream's
   flow: cancel at the thinking→content transition, re-render via `chatPrompt`
   with the thinking as an assistant message (+ the harmony final-channel
   prefill), rerun constrained.

**ChatHandler** keeps its accepted double request and gains the marker stop on
pass one — the fix for the measured qwen3.6 chat runaway (a model that never
closes thinking has no thinking→content transition to trigger on, so pass one
burned to `num_predict`: eval pinned at 16000, empty response, ~16k reported as
`prompt_eval_count`). Tools requests keep the transition flow: a stop at the
marker would preempt a tool call that follows `</think>`.

**Hardenings, both handlers** (carried from v1, now engine-agnostic):

- Continuation pre-checked with `Tokenize` against the **loaded runner's
  `ContextLength()`** (request options may still hold `0` = auto — using them
  raw would trip the check on every request); if the thinking filled the
  window, the request ends `done_reason:"length"` with the streamed thinking
  preserved instead of a 500.
- Final metrics count each token once: pass one's `prompt_eval_count`, summed
  `eval_count`/durations. On chat this fixes the 16,181-vs-~888 inflation; the
  cancel-path (no pass-one final) keeps upstream's raw forwarding.

The runner-layer split was removed in the follow-up commit (routes stopped
passing `ThinkCloseTag`, making it unreachable; then the field and machinery
went away, −367 lines). `reclassifyConstrainedThinking` remains as the net for
flows the double request does not cover.

## Validation

Functional matrix (temperature 0, `num_predict` 512, both fixed builds:
fork `main`+fix on b10091 runners, `release/0.32.1-dynres`+fix on b9888+002 runners):

| probe | before | after |
|---|---|---|
| nemotron3 generate think+`"json"` | `response:""`, JSON in thinking, eval 7 | thinking + `{"capital": "Paris"}`, eval 38 |
| nemotron3 generate think+schema | empty response (slot death under lazy attempt; empty pre-fix) | thinking + `{"capital":"Paris"}` |
| nemotron3 generate think+`"json"` streaming | — | thinking chunks → JSON chunks, one done |
| nemotron3 generate think:false+`"json"` | JSON, no thinking | unchanged |
| nemotron3 generate think, no format | thinking + prose | unchanged |
| nemotron3 chat think+`"json"` | worked (double request) | unchanged |
| qwen3.6 generate think+`"json"` (b9888+002+fix) | `response:""` | 431-token reasoning + `{"capital": "Paris"}` |
| qwen3:0.6b / deepseek-r1:1.5b generate think+`"json"` | JSON in response (thinking skipped) | unchanged |

qwen3.6 on fork `main` (b10091 runners) emits degenerate 4-token output with or
without the fix and on `/api/chat` too — that is the known b10091 payload regression
([amd-upgrade-gate.md](amd-upgrade-gate.md)), orthogonal to this bug.

Ground-truth vision suite, think-on cells (`THINK=on NUM_PREDICT=4096`, suite as
committed in `docs/maxusai/vision-suite/`, against `release/0.32.1-dynres`+fix,
b9888+002 runners, port 11440):

| cell (think:on) | pre-fix | post-fix |
|---|---|---|
| nemotron scene: json valid / labels / colors / bbox / serial | response empty (misfiled) | **valid / 6/6 / 6/6 / 5/6 / found** (`NUM_PREDICT=4096`, eval 4098) |
| nemotron invoice: json / invoice-no / items / qty-price / total / name-bbox | response empty (misfiled) | **valid / found / 5/5 / 5/5 / exact / 5/5** (`NUM_PREDICT=12288`, thinking 13.9k chars, eval 4969, stop) |
| nemotron multi-image: json / q1 / q2 / chart values / q4-bbox | response empty (misfiled) | **valid / right / right / 5/5** / miss (`NUM_PREDICT=24576`, thinking 27.7k chars, eval 10996, stop) |
| qwen3.6 scene: json / labels / colors / serial | response empty (misfiled) | **valid / 6/6 / 6/6 / found** (thinking 4.4k chars, eval 2458, stop) |
| qwen3.6 invoice: json / items / qty-price / total / name-bbox | response empty (misfiled) | **valid / 5/5 / 5/5 / exact / 4/5** (thinking 3.4k chars, eval 1970, stop) |
| qwen3.6 multi-image: json / q1 / q2 / chart values | response empty (misfiled) | **valid / right / right / 5/5** (thinking 19.5k chars, eval 9050, stop) |

(qwen3.6 cells at `NUM_PREDICT=12288`/`num_ctx=32768`; independently reproduced by
the parallel benchmarking session's own runs on the same server. Cells recorded
before `925a669a` report the continuation's cache-inclusive `prompt_eval_count` —
inflated by the thinking tokens; scoring fields are unaffected.)

The think:on cells don't just match the recorded think:false verdict — they exceed
it where localisation is concerned: 5/6 scene bbox center-hits (0 historically) and
5/5 invoice name-bbox hits (first ever). Budget note: with thinking on, nemotron
spends thousands of tokens reasoning over dense extraction tasks (13.9k chars on the
invoice); at `NUM_PREDICT=4096` the document/multi cells exhausted the budget inside
thinking (honest `done_reason:"length"`, empty response — the same would happen on
`/api/chat`). Think mode on extraction workloads is a latency/budget trade, not a
correctness workaround anymore; size `num_predict` (and `num_ctx`) for reasoning +
answer.

Unit tests (v1, removed with the runner split): `TestLlamaServerCompletionDeferredFormat`
(+`ContextFull`, `ThinkingOnly`). Still present: `TestApplyCompletionFormat`,
`Test{Nemotron3Nano,Qwen35}Parser*ThinkingCloseMarker`, and
`TestReclassifyConstrainedThinking`.

### Validation v2 — routes layer (2026-08-02)

Fork-`main` Go binary served over the b9888+002 payload
(`/opt/github/MaxusAI/ollama-0321/build/lib/ollama`, test server :11441,
gfx1151/ROCm, `OLLAMA_FLASH_ATTENTION=1`, `OLLAMA_KV_CACHE_TYPE=q8_0`); suite
as committed, `THINK=on NUM_PREDICT=16000`, temperature 0, `format:"json"`.
Raw scores/responses archived in the session's `validation-11441/`.

| cell (think:on) | ENDPOINT=generate | ENDPOINT=chat |
|---|---|---|
| nemotron scene: json / labels / colors / bbox-hits / IoU / serial | **valid / 6/6 / 6/6 / 6/6 / 0.768 / found** | **identical, same counts** |
| nemotron scene counts (prompt / eval) | 2,674 / 11,373 | 2,674 / 11,373 |
| qwen3.6 invoice: json / items / qty-price / total / name-bbox | **valid / 5/5 / 5/5 / exact / 4/5** | **valid / 5/5 / 5/5 / exact / 4/5** |
| qwen3.6 invoice counts (prompt / eval) | 2,741 / 7,322 | 2,741 / 3,139 |

- Nemotron's chat and generate cells are literally identical — the two
  continuation styles (textual vs `chatPrompt` re-render) converge at temp 0.
  Its bbox line improved over the v1 record (6/6 center-hits, IoU 0.768,
  norm-1000; v1 recorded 5/6 pixel-space IoU ≈ 0.3).
- **Chat runaway fixed and honest:** the scene prompt drives qwen3.6 chat into
  a marker-stop pass one that ended at 13,515 generated tokens (< 16,000 —
  previously pinned at exactly 16000) followed by a real constrained
  continuation, with `prompt_eval_count` reported as the true **2,613** (the
  pre-fix measurement reported 16,181 vs ~888 real on a smaller prompt — the
  continuation's cache-inclusive prefill).
- **Honest residuals, reported as data:** qwen3.6 + this scene prompt at
  temp 0 is a pathological thinker — at `num_ctx 32768` it produced 16,000
  thinking tokens without ever emitting `</think>` (honest
  `done_reason:"length"`, empty response, true counts); at `num_ctx 16384` the
  continuation started legally with ~250 tokens of headroom and was cut by the
  window mid-JSON. Neither is the bug: the first is the model never closing
  (nothing can force a close the model won't emit — only `num_predict` bounds
  it now), the second is the documented size-`num_ctx`-for-thinking+answer
  trade. The invoice cells show the same model+endpoints producing perfect
  constrained output when its thinking terminates.

Unit tests (v2): `TestGenerateThinkFormatMarkerFlow` (+`Streaming`,
`ContextFull`) and `TestChatThinkFormatMarkerStop` /
`TestChatThinkFormatLengthNoContinuation` in `server/routes_generate_test.go` —
pass-one format-strip + marker-stop assertions, continuation prompt equality,
merged metrics, context-full length end, and the no-continuation-on-length
rule, against the mock runner. The pre-existing chat `structured outputs
restart` tests cover the transition fallback unchanged.

## Upstream engagement draft

Comment for [PR #14288](https://github.com/ollama/ollama/pull/14288) (do not open a
new issue — #10538/#11691 already track this and #14288 is the pending fix).
Positioned as a **collaborative superset of #14288**: we adopted its routes-layer
double-request architecture and carry three hardenings on top, each backed by a
measurement. Rewritten 2026-08-02 for the routes-layer implementation
(supersedes the earlier runner-layer pitch). **Do not post without Glenn's
explicit go.**

> Confirming this is still needed on v0.32.x, and adding measurements from a
> downstream deployment that shipped exactly this PR's architecture — plus three
> hardenings we needed in practice. First, the failure is worse than "garbage
> output" for one class of model: with `/api/generate` + `think:true` +
> `format:"json"`, models whose parser starts *inside* thinking (nemotron3,
> qwen3.5/3.6) return `{"response": "", "thinking": "{…the model's correct JSON…}"}`.
> The eager grammar admits no `</think>` token, so the model can never leave thinking
> and the marker-based parser files the entire grammar-forced answer as reasoning.
> The blast radius is wider than the native endpoint: `/v1/completions` also routes
> to `GenerateHandler`, so the OpenAI-compat completions API is affected too, while
> `/v1/chat/completions`, `/v1/responses` and `/v1/messages` all reach `ChatHandler`
> and are fine — which is why this has stayed invisible to most users.
>
> We run this PR's double-request shape in production and suggest folding in three
> hardenings, each of which corresponds to a failure we hit:
>
> 1. **Stop pass one at the think-close marker instead of waiting for content.**
>    The transition signal (parsed thinking → parsed content) never comes for a
>    model that never closes its thinking, and pass one burns to `num_predict`
>    unconstrained — we measured qwen3.6 on `/api/chat` (which has this same gap
>    since #12460) pinned at eval_count 16000 with an empty response,
>    prompt-dependently 2 runs out of 3. Parsers that know their marker expose it
>    (a one-method `ImplicitThinkingParser` interface on the parser registry:
>    `nemotron-3-nano`, `qwen3.5`; the generic thinking parser participates when
>    the prompt prefills the opening tag), and pass one carries it as a stop
>    string. Generation then ends the moment reasoning closes — also saving the
>    throwaway unconstrained content the cancel flow generates — and the handler
>    feeds the marker through the parser and continues. A model that still never
>    closes ends as an honest `done_reason:"length"`. The same one-line stop fixes
>    the chat handler's runaway.
> 2. **Pre-check the continuation against the context window.** If the thinking
>    (re-submitted as prompt for pass two) no longer fits `num_ctx`, the runner
>    rejects the request and the client gets a 500 *after* thinking already
>    streamed — reproduced live. A `Tokenize` pre-check (against the loaded
>    runner's context length, not request options, which may still hold 0=auto)
>    turns that into a clean `done_reason:"length"` with the thinking preserved.
> 3. **Merge metrics so each token is counted once.** Forwarding pass-two's
>    counts reports the continuation's cache-inclusive prefill as
>    `prompt_eval_count` — we measured 16,181 reported vs ~888 real on chat.
>    Report pass one's `prompt_eval_count` and sum `eval_count`/durations.
>
> One shape difference worth discussing: for models with a known marker, pass two
> can continue **textually** (`prompt + emitted thinking + marker`, format
> applied) instead of re-rendering with an assistant thinking message. That keeps
> the continuation an exact prompt-cache hit, is independent of whether a
> template round-trips thinking, and needs no per-model "continue thinking vs
> answer now" disambiguation (the reason gpt-oss needs the final-channel prefill
> hack). Models without a marker (harmony) keep this PR's re-render flow — the
> two compose cleanly as fast path + fallback; in our validation both styles
> produced identical outputs and identical merged counts on nemotron3.
>
> Validated end-to-end over a ground-truth vision-extraction suite (nemotron3 +
> qwen3.6, `"json"` and schema formats, streaming and non-streaming, both
> endpoints): think-on structured output goes from empty responses to valid JSON
> at full extraction quality, with reasoning measurably improving nemotron's
> bbox grounding. A branch with the combined shape (this PR + the three
> hardenings + regression tests for the chat runaway, context-full end, and
> metrics merge) is ready — happy to PR it, or to break any subset into this one,
> whichever the maintainers prefer.

The ready branch is `feat/upstream-generate-think-format`, rebased onto current
upstream `main` with the routes-layer shape (see ADR 0004); full `llm` +
`server` + `model/parsers` suites pass, and it excludes these fork-only docs.

## What if the model *mentions* `</think>` inside its thinking?

Considered explicitly; the stop-split introduces **no new hazard** because it uses the
same text-level, first-occurrence semantics every other layer already uses:

- Whether the model emits the *special token* `</think>` or spells out the literal
  characters (e.g. "I must remember to write `</think>` before answering"), both
  detokenize to the same text: the parsers' preserved-token handling renders the
  special token as its text, and llama-server stop strings match on detokenized text.
  There is no way to distinguish "quoting the marker" from "closing the block" at
  the text level — for the model, emitting that token *is* closing the block.
- The Go parsers (`nemotron-3-nano`, `qwen3.5`, the generic template parser), the
  `/api/chat` double-request transition, and llama.cpp's own
  `until("</think>")` grammar rule all end thinking at the first occurrence. The
  stop-split's phase-1 stop string agrees with them by construction — parser state
  and constraining can't disagree about where thinking ended.
- Worst case (model "quotes" the marker mid-reasoning): thinking is cut early and the
  constrained answer starts at that point — a valid-JSON response with truncated
  reasoning, never a crash, never an empty response. The same generation would be cut
  at the same point on `/api/chat` today, and with no format at all the parser would
  still end thinking there.
- The reverse direction is also safe: phase 2 (the constrained continuation) does
  **not** carry the marker stop string (unit-tested), so a JSON *string value*
  containing `</think>` cannot truncate the response; the parser is in content state
  and passes it through.
- Residual edge: nemotron's parser also treats `<tool_call>` as a thinking
  terminator. `/api/generate` has no tools, so phase 1 does not stop on it; if a
  model ever emitted it mid-thinking under format, phase 1 would run to its natural
  end and the output would degrade to pre-fix classification for that request (no
  crash). On `/api/chat` the v2 marker flow is disabled whenever the request
  carries tools (the stop would preempt a tool call that follows `</think>`);
  those requests keep the upstream transition flow. Revisit if generate grows
  tool support.

## Operational notes

- The routing-policy amendment "serve JSON extraction with `think:false`"
  (nemotron-test-image.md, 2026-08-01) is lifted by this fix on builds that carry it;
  keep it for any binary without the fix.
- `release/0.32.1-dynres` carried v1 as `d1ef5557` (cherry-pick of `928c7494`);
  the v2 routes-layer commits are cherry-picked on top (see the branch log) and
  the image rebuilt per
  [nemotron-test-image.md](nemotron-test-image.md) conventions — llama-server
  payload untouched (b9888+002), so existing quality verdicts stand.
