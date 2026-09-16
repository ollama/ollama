# MLX Model Porting Agent Guide

This directory contains experimental MLX model implementations and the tooling
used to validate new architecture ports. Treat this file as the local conduct
guide for humans and coding agents working under `x/models`.

The porting process itself lives in exactly one place: read
`x/models/PORTING_GUIDE.md` first and follow its workflow — each step links a
deep-dive document in `x/models/bringup/`. Every artifact the work produces
belongs under `x/models/bringup/<model>/` (see `bringup/artifacts.md`); never
`/tmp` or scratch directories.

## Working Rules

- Use a reference-driven workflow. The source of truth is the implementation
  the publisher designates as authoritative (or the one a maintainer
  explicitly names); record that choice and its revision. See
  `bringup/reference-capture.md`.
- Keep model ports readable and local to the new architecture. Do not refactor
  shared utilities, cache behavior, tokenizer code, or existing model files
  unless the task explicitly requires it.
- Prefer the patterns already used by nearby MLX models (`llama`, `qwen3_5`,
  `gemma4`) over new abstractions.
- Treat custom kernels as a last resort; follow the optimization order in
  `bringup/performance.md` and keep a kernel only with measured incremental
  benefit and explicit backend, OS, correctness, and fallback coverage.
- Treat tensor namespaces and config-schema shape as artifact ABI, not
  cosmetic metadata. A permissive loader that accepts two layouts does not
  make those layouts interchangeable for published tags. See
  `bringup/release-gates.md`.
- Reference parity tests must not re-implement renderer policy in their
  template-input converters, and must never override the template's
  generation prompt. See `bringup/renderer-parity.md`.
- Do not treat generation quality or perplexity deltas as hard CI gates unless
  the task explicitly sets thresholds. Record the evidence and explain the
  judgment.

## Review Discipline

- Disclose agent assistance in PR notes when applicable.
- Include exact commands and file paths for every generated artifact.
- Do not claim a port is complete without the evidence the guide's Definition
  Of Done requires; the bring-up archive is that evidence.
- If a test skips because model weights, references, MLX libraries, or corpora
  are unavailable, name the missing input and the command to regenerate it.
- If a reviewer asks for a design change, reason from the code and evidence.
  Do not blindly re-run an agent and paste the result.
