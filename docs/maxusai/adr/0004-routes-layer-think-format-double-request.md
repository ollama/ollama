# ADR 0004: Move the think+format double request to the routes layer

- **Status:** accepted, validated 2026-08-02 on fork `main` (Go binary over the
  b9888+002 payload, test server :11441). Supersedes the *mechanism* of
  [ADR 0002](0002-deferred-format-constraining.md) (runner-layer stop-split);
  0002's problem analysis and alternatives record remain the reference.
- **Date:** 2026-08-02
- **Deciders:** MaxusAI fork maintainers

## Context

ADR 0002 fixed `/api/generate` + `think` + `format` with a two-pass stop-split
inside the llama-server runner (`llm/llama_server.go`). It worked, but at the
wrong layer for the long game:

- **Upstream alignment.** Upstream fixed the same bug for `/api/chat` at the
  routes layer (#12460, v0.12.4) and the open generate port
  ([#14288](https://github.com/ollama/ollama/pull/14288)) mirrors that shape.
  A runner-layer fix would always be a divergent architecture to defend.
- **Engine coverage.** The runner split only covered the llama-server runner;
  MLX (and upstream's ollamarunner) fell through to the reclassification net.
  The routes layer sits above every engine.
- **Chat asymmetry.** The measured qwen3.6 chat runaway (eval pinned at 16000,
  empty response, prompt-dependent 2-of-3) lived in chat's double request,
  which the runner fix could not touch.

Upstream's #14288, however, lacks all three hardenings we shipped and measured:
pass one waits for the parser's thinking→content transition (a model that never
closes thinking burns to `num_predict`), there is no context-overflow pre-check
(the 500 after streamed thinking reproduces live), and per-pass token counts are
forwarded raw (16,181 reported vs ~888 real prompt on chat).

## Decision

Re-architect to the routes layer (`server/routes.go`) as a superset of the
upstream pattern, and retire the runner-layer split:

1. **GenerateHandler** runs format-constrained completions as a chat-style
   state-machine loop. When the model's think-close marker is known
   (`parsers.ImplicitThinkingParser`, or the generic thinking parser when the
   prompt prefills the opening tag), **pass one strips the format and injects
   the marker as a stop string**; the continuation re-submits the exact token
   stream (`prompt + raw pass-one output + marker`) with the format applied —
   textual continuation, no template round-trip, prompt-cache friendly. Models
   without a marker (harmony/explicit thinking) use upstream's transition-cancel
   + `chatPrompt` re-render path unchanged, so the fork stays a clean superset
   of #14288.
2. **ChatHandler** keeps its accepted double request but gains the same marker
   stop on pass one (when tools are absent — a stop at the marker would preempt
   a post-thinking tool call), honest metrics, and the continuation pre-check.
3. **Hardenings, both handlers:** the continuation is pre-checked with
   `Tokenize` against the loaded runner's `ContextLength()` (request options may
   still hold 0 = auto) and ends as `done_reason:"length"` with the streamed
   thinking preserved when it cannot fit (headroom 8, as in ADR 0002); final
   metrics report pass one's `prompt_eval_count` and summed eval counts and
   durations, so every token is counted once.
4. **Pass-one end conditions:** `done_reason:"stop"` with no parsed content
   continues to pass two (the marker fired, or an EOS still inside thinking —
   the latter is a recovery the runner split routed to the reclassify net);
   `length` ends honestly with no continuation. Without `stopping_word` at this
   layer a user stop string firing mid-thinking also continues — acceptable,
   since with `format` set the contract is a formatted answer.
5. **Runner cleanup:** routes stopped passing `ThinkCloseTag`, making the
   runner split unreachable; a follow-up commit removes the split and the
   `CompletionRequest.ThinkCloseTag` field. `reclassifyConstrainedThinking`
   stays for flows the double request does not cover.

## Consequences

- Everything ADR 0002 delivered still holds, now engine-agnostic and symmetric
  with chat. Live validation (b9888+002 payload, think:on, `format:"json"`,
  temp 0): nemotron scene = valid JSON 6/6 labels/colors/bbox-hits + serial on
  **both endpoints** with identical honest counts (prompt 2674, eval 11373);
  qwen3.6 invoice = valid JSON 5/5 on **both endpoints** (prompt 2741, chat
  eval 3139); qwen chat `prompt_eval_count` is now the real 2,613 for the scene
  prompt, not the continuation's cache-inclusive 16k.
- The qwen chat runaway is structurally fixed: pass one ends at the marker the
  moment thinking closes. When the model *never* closes (qwen3.6 scene at
  temp 0 keeps reasoning past 16000 tokens — reproduced at 32k ctx), the
  request ends as an honest `length` with true counts instead of an empty
  response; that residual is model behavior, not the bug.
- Chat's pass two still re-renders via `chatPrompt` (accepted upstream shape,
  templates render the thinking round-trip); generate's marker flow continues
  textually — both were observed producing identical totals on nemotron.
- format+tools+thinking on chat intentionally keeps the transition flow (no
  marker stop) so tool calls that follow `</think>` are not cut off.
- The runner-layer mechanism, its tests, and the `ThinkCloseTag` wire field are
  gone (−367 lines); `llm/llama_server.go` is back to a single `/completion`
  per completion, which is also the shape the upstream-facing branch carries.
- The upstream branch (`feat/upstream-generate-think-format`) rebases onto this
  architecture and is positioned as a collaborative superset of #14288 — same
  state machine, plus the marker stop, pre-check, and honest metrics.
