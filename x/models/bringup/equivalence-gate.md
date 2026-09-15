# Reference Equivalence Gate

Part of the MLX porting guide (`x/models/PORTING_GUIDE.md`). Bring-up
evidence lives beside these docs in `x/models/bringup/<model>/` — git-ignored,
never committed; see `artifacts.md`.

This gate must pass on unquantized BF16 before quantization, performance work,
or a claim that the port is complete. Integration tests, perplexity, and
generation samples are supporting evidence only. Treat every discrepancy as a
defect in the port until the evidence proves otherwise.

1. **Prompt bytes and token IDs**
   - Compare exact reference-template bytes to the Go renderer.
   - Tokenize those exact prompts with the reference training tokenizer and
     Ollama's production tokenizer; require identical token IDs, including
     special tokens, BOS/EOS, and the generation prompt.
   - Decoded-text equality is not sufficient. Include code/tool output,
     indentation, whitespace before punctuation, newlines, contractions,
     non-ASCII text, empty content, and special-token-looking strings.
   - For tools, cover multiple definitions and transcripts ending in user,
     assistant, and tool roles, especially a tool result containing JSON or
     source code followed by the next generation prompt.

2. **Prefill, cache, and output-head parity**
   - Run identical token IDs through short one-shot prefill,
     production-style chunked prefill, and cached decode.
   - Compare embeddings, selected layer boundaries, final hidden state, raw
     LM-head output, and final post-transform logits.
   - Assert every dtype transition from the operation ledger. Capturing
     tensors as float32 for comparison must not erase original dtype metadata.
   - `--skip-logits` is acceptable while localizing layers, never for the
     final gate. Capture last-position logits when full-sequence logits are too
     large.
   - Report top-1 and its reference margin, top-k overlap, expected-token rank,
     logprob delta p50/p95/max, and KL when stable. Cosine similarity alone
     cannot validate output transforms or token ranking.

3. **Teacher-forced trajectory parity**
   - Force the same realistic continuation token IDs through both
     implementations and compare post-transform logits at every position.
   - Cover at least 128 positions across prose and code-like content. Tool
     models must cross every supported channel, a tool call, a code/JSON tool
     result, and the next assistant decision.
   - Save top-1 agreement, mean top-k overlap, expected-token ranks and
     logprob deltas, plus reference margins for every mismatch.
   - Investigate every mismatch. Accept only low-margin swaps supported by
     independently measured backend drift; do not label unexplained
     high-margin differences "accumulation."
   - After BF16 passes, replay the same trajectory on every production
     quantization against the verified BF16 port. Quantized integration success
     alone is not a quality-parity check.

4. **Protocol and parser replay**
   - Feed well-formed reference continuations through the production streaming
     parser at every meaningful split point through control tokens, tool
     wrappers, arguments, and UTF-8 text.
   - Compare content, thinking, tool calls/arguments, turn actions, and
     continuation prompts. Round-trip parsed messages into the next renderer
     turn and repeat prompt-byte/token-ID parity.
   - Include multiple calls, malformed/truncated output, and ordinary content
     containing control-token-like strings. Hardening may improve malformed
     recovery but must not alter well-formed reference output.

5. **Deterministic reference replay**
   - Use identical prompt token IDs and publisher-equivalent greedy sampling.
     Record raw generated token IDs, not only decoded output.
   - For tool models, replay a bounded multi-turn scenario with fixed tool
     results. A bad realistic workload is a trigger for this analysis; a good
     workload never proves parity.

6. **Discrepancy protocol**
   - Freeze prompt bytes, token/continuation IDs, model and reference hashes,
     dtypes, sampling settings, and raw output before editing code.
   - Locate the first divergence in order: rendered bytes, token IDs,
     embedding, layer/submodule, raw LM head, transformed logits, sampler,
     parser.
   - Fix the earliest divergence and rerun the whole gate. Never mask an
     upstream mismatch with renderer steering or parser recovery.
   - If the reference appears defective, require a minimal native
     reproduction, disagreement between publisher references, or publisher
     confirmation. Missing tooling is not a waiver; extend it or build a
     disposable reference harness.

## Tooling Gaps

Missing tooling is not a waiver — build a disposable harness and note it in
the review report:

- **Top-k/rank/KL report helpers.** Cosine similarity cannot see
  token-ranking flips, and logprob diagnostics get re-invented per port.
  Shared helpers should emit top-1 margin, top-k overlap, expected rank,
  logprob deltas, and KL.
- **Teacher-forced trajectory command.** The gate's strongest check (part 3)
  is a bespoke script every time. One command should take prompt +
  continuation token IDs and emit a per-position agreement report.
