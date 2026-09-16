# Renderer And Parser Parity

Part of the MLX porting guide (`x/models/PORTING_GUIDE.md`). Bring-up
evidence lives beside these docs in `x/models/bringup/<model>/` — git-ignored,
never committed; see `artifacts.md`.

Renderer and parser tests should be treated as part of model correctness, not
as API polish. When a model ships a `chat_template.jinja` or tokenizer chat
template, walk every meaningful output-producing branch in that template and
add renderer tests for it before creating the Ollama model. Cover at least:

- first system message handling and additional system messages
- plain user-only and multi-turn chat
- generation prompt or assistant-prefill behavior
- thinking enabled, disabled, and omitted, including exact delimiter prefill
- tool registry rendering
- assistant messages with content, thinking text, and tool calls
- tool response messages
- variant-specific template differences, such as BF16 and quantized drops with
  different tool-call formats

Prefer exact rendered-output tests over substring checks. For complex templates,
follow the pattern in `model/renderers/gemma4_reference_test.go`: commit or
locate the reference template, render representative messages through the
reference template engine or `AutoTokenizer.apply_chat_template`, and compare it
to the Go renderer.

Reference parity applies to canonical transcripts the template supports; it
does not make every publisher `raise_exception` branch Ollama product policy.
Audit renderer hard failures separately. Normalize common API roles when their
hierarchy and order can be preserved; otherwise, if a message can be serialized
without dropping or corrupting it, warn and pass it through in the model's turn
format. Reject only when faithful serialization is unsafe or ambiguous. Verify
that application-shaped requests avoid the warning in integration tests, not
with renderer unit tests that capture logs.

The expanded parity suites are gated behind an env var and skip silently
without it — the gate is only satisfied when they actually run against the
live template engine:

```bash
VERIFY_JINJA2=1 go test -count=1 ./model/renderers/ -run '(?i)<model>'
```

(Requires a `.venv/bin/python` with transformers 5.x, or `uv` available to
fetch one.) If Ollama intentionally diverges from the template, such as
prefilling a thinking open or close tag to steer generation, assert that
difference explicitly in a separate "known differences" test. Never normalize a
divergence away inside the test harness — a converter that mirrors renderer
policy turns the parity test into self-verification. Runtime smoke tests are
not a substitute for this renderer coverage because parser fallbacks can hide
malformed prompts until a user exercises a different chat flow.

## Anti-Circularity Invariants

These structure rules keep a parity test from drifting into
self-verification, where a converter that mirrors renderer policy stays green
while the renderer diverges from the template:

- The converter feeding the reference template engine is a mechanical field
  mapping only. It never injects messages, appends instruction text, or
  wraps content. Note gemma4's converter is only safe because its template
  natively accepts `tools` and `enable_thinking`.
- When the template lacks an input Ollama needs (no tools argument, no
  thinking control), centralize that policy in one documented production
  function (e.g. `<model>PrepareMessages`) called by both the renderer and the
  parity test.
- Always render with `add_generation_prompt=True` and compare the template
  output verbatim. The harness never overrides the generation prompt or
  post-processes the reference side; a different prompt ending is a
  known-difference case, not a harness override.
- With tools present, cover transcripts ending in each terminal role (user,
  assistant, tool). Tools + trailing tool result is every turn of a real
  agentic loop.
- Renderer-side invented behavior (guidance prose, prefills, injected
  messages) gets the same discipline as parser leniency: mark it
  Ollama-invented, keep it minimal and centralized, and validate it with
  reference replay/logprob evidence. Structural transcript deviations are
  presumptively wrong.
- Capture the exact rendered prompt from a failing live session before
  editing renderer code, so before/after renderings stay diffable.

## Tooling Gaps

Missing tooling is not a waiver — build a disposable harness and note it in
the review report:

- **Template-byte + token-ID compare tool.** Decoded-text equality cannot
  detect pretokenizer differences (for example at ASCII-space boundaries
  before punctuation). One command should compare exact rendered bytes AND
  production token IDs against the reference tokenizer.
- **Parser split/round-trip harness.** Streaming-parser bugs at chunk
  boundaries (tool-call wrappers split mid-token) are found ad hoc. A harness
  should replay a reference continuation at every split point and diff parsed
  output (see the equivalence gate, part 4).
