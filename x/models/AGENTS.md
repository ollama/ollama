# MLX Model Porting Agent Guide

This directory contains experimental MLX model implementations and the tooling
used to validate new architecture ports. Treat this file as the local process
guide for humans and coding agents working under `x/models`.

## Porting Standard

- Use a reference-driven workflow. The source of truth is the canonical
  `transformers` implementation for the target architecture unless a maintainer
  explicitly names another reference.
- Keep model ports readable and local to the new architecture. Do not refactor
  shared utilities, cache behavior, tokenizer code, or existing model files
  unless the task explicitly requires it.
- Prefer the patterns already used by nearby MLX models (`llama`, `qwen3_5`,
  `gemma4`) over new abstractions.
- Preserve reviewer signal. Do not claim a port is complete without recording
  the commands, model revision or local path, prompt, dtype, numerical
  comparisons, perplexity results, generation samples, known skips, and known
  limitations.

## Required Workflow

1. Read `x/models/PORTING_GUIDE.md`.
2. Inspect the model before implementing:

   ```bash
   python3 x/models/scripts/inspect_model.py \
       --model /path/to/hf/model-or-variant \
       --output /tmp/ollama_port/<arch>
   ```

3. Generate PyTorch reference activations with `dump_activations.py`. Keep the
   generated sidecar manifest next to the safetensors output. If reference
   settings such as attention backend or cache mode are in question, compare
   the resulting artifacts with `compare_activations.py`. For cache-specific
   checks, use `dump_activations.py --decode-text` or `--decode-token-id` so
   the reference captures a decode pass after cached prefill.
4. Implement the smallest model-specific Go code needed to reproduce the
   reference forward pass.
5. Add forward-pass tests that load the reference activations and compare layer
   outputs with `x/models/testutil`.
6. Add long-sequence, cache, quantized, thinking, or multimodal validation when
   the architecture has those risks.
7. Run `x/cmd/ppl` for end-to-end quality once the forward pass matches.
8. Run integration tests with `OLLAMA_TEST_MODEL` against a created local tag
   as final validation. Do this after focused tests pass; integration tests are
   too slow and broad to replace model-specific unit and reference tests.
9. Produce a reviewer report with `summarize_validation.py`.

## Validation Rules

- Exact element-wise tolerance is appropriate for single operations such as
  embedding lookup, matmul, and normalization.
- Use cosine similarity for accumulated layer outputs and final hidden states.
- Use per-position reports for RoPE, sliding-window, cache-offset, and
  long-context failures.
- Do not treat generation quality or perplexity deltas as hard CI gates unless
  the task explicitly sets thresholds. Record the evidence and explain the
  judgment.
- For integration testing, make sure the created model advertises only the
  capabilities it really supports. Audio, vision, tools, embedding, and
  thinking tests rely on capabilities to decide whether to run or skip.

## Review Discipline

- Disclose agent assistance in PR notes when applicable.
- Include exact commands and file paths for every generated artifact.
- If a test skips because model weights, references, MLX libraries, or corpora
  are unavailable, name the missing input and the command to regenerate it.
- If a reviewer asks for a design change, reason from the code and evidence.
  Do not blindly re-run an agent and paste the result.
