# MLX Model Porting Guide

This guide describes the repeatable process for bringing a new language model
architecture up on Ollama's MLX runner. The process is reference-driven: start
from the canonical `transformers` implementation, generate reproducible
reference artifacts, implement the same forward pass in Go/MLX, and collect
enough evidence that a reviewer can understand both the code and the validation.

The supporting pieces are:

- `x/models/scripts/inspect_model.py`: inspects one or more Hugging Face model
  directories or Hub IDs and writes a porting manifest.
- `x/models/scripts/dump_activations.py`: captures per-submodule PyTorch
  outputs into safetensors plus a deterministic sidecar manifest.
- `x/models/scripts/compare_activations.py`: compares two activation artifacts
  and ranks tensor or position drift.
- `x/models/testutil`: Go helpers for loading safetensors directories or
  Ollama tags, comparing tensors, and finding the first layer/position where
  MLX drifts from the reference.
- `x/cmd/ppl`: perplexity CLI for end-to-end quality validation against either
  lm-evaluation-harness or llama.cpp methodology.
- `x/models/scripts/summarize_validation.py`: builds a reviewer-ready Markdown
  report from non-agentic validation artifacts.

## Workflow

### 1. Discovery

Start by inspecting representative model variants:

```bash
python3 x/models/scripts/inspect_model.py \
    --model models/<org>/<small-base> \
    --model models/<org>/<larger-base> \
    --output /tmp/ollama_port/<architecture>
```

Review `porting_manifest.md` before coding. Look for:

- config fields that vary across variants
- tensor prefixes and tied embedding behavior
- RoPE parameters, partial rotation, and scaling
- sliding-window or hybrid layer patterns
- MoE expert shapes and routing config
- attention bias, grouped-query attention, MLA, recurrent or convolution state
- multimodal processor fields
- thinking tags in chat templates
- safetensors dtype histogram and quantization metadata

### 2. Reference Capture

Generate a PyTorch reference from the `transformers` implementation:

```bash
.venv/bin/python3 x/models/scripts/dump_activations.py \
    --model models/<org>/<name> \
    [--model-class MyModelForCausalLM] \
    [--attn-implementation eager] \
    [--transformers-path path/to/custom/transformers/src] \
    --skip-logits
```

Default output:

- `/tmp/ollama_ref/<variant>/activations.safetensors`
- `/tmp/ollama_ref/<variant>/activations.safetensors.manifest.json`

Use a short prompt first. Add a long prompt when the model has sliding-window,
RoPE scaling, recurrent state, or other context-sensitive behavior. Re-run with
filters when drilling into a failing layer:

Pin `--attn-implementation` when the reference model supports multiple
Transformers attention backends. Backends such as SDPA and eager can produce
different long-context numerics even in the official implementation. The sidecar
manifest records both the requested and resolved backend, plus whether the
reference forward pass used cache state. Activation references default to
`use_cache=false`; pass `--use-cache` only for cache-specific reference captures.

For decode/cache references, use the prompt as a cached prefill and capture the
follow-up token or text:

```bash
.venv/bin/python3 x/models/scripts/dump_activations.py \
    --model models/<name> \
    --attn-implementation eager \
    --prompt "$(cat /tmp/long-prompt.txt)" \
    --decode-text " the" \
    --skip-logits
```

```bash
.venv/bin/python3 x/models/scripts/dump_activations.py \
    --model models/<name> \
    --filter "model.layers.5.self_attn.*" \
    --filter "model.layers.5.mlp.*"
```

Use `--list-modules` to map Python hook names to Go fields:

```bash
.venv/bin/python3 x/models/scripts/dump_activations.py \
    --model models/<name> \
    --list-modules
```

When a model is sensitive to reference settings, compare the resulting
activation dumps directly:

```bash
python3 x/models/scripts/compare_activations.py \
    --got /tmp/ollama_ref/<variant>/activations-eager.safetensors \
    --want /tmp/ollama_ref/<variant>/activations-sdpa.safetensors \
    --filter "model.layers.*" \
    --json-output /tmp/ollama_port/<architecture>/activation-comparison.json \
    --markdown-output /tmp/ollama_port/<architecture>/activation-comparison.md
```

This is a reference-quality check, not a replacement for Go unit tests. It
helps distinguish an unstable reference from a model implementation defect.

### 3. Implementation

Implement the Go model in `x/models/<name>/` using nearby models as templates.
Keep the first version self-contained and model-specific. Shared utility
changes should be small, justified by more than one model, and called out in
the review report.

`testutil.LoadModelFromDir(t, dir)` loads any registered architecture from a
directory containing `config.json`, `tokenizer.json`, and `*.safetensors`.
It creates a synthetic manifest and uses the standard `base.New()` factory
dispatch.

### 4. Layer Comparison

Write forward-pass tests using `testutil`:

```go
model := testutil.LoadModelFromDir(t, testutil.ModelDir(t, "MODEL_DIR", "models/<name>"))
ref := testutil.LoadReference(t, filepath.Join(testutil.DefaultRefDir("variant"), "activations.safetensors"))

h := model.Forward(tokens, caches)
testutil.CompareArraysCosineSim(t, "final_hidden", h, ref["model.norm"], 0.999)

embed := model.EmbedTokens.Forward(tokens)
testutil.CompareArrays(t, "embed_tokens", embed, ref["model.embed_tokens"], testutil.BFloat16Tol())
```

Debug failures layer by layer. Use `CompareLayersPerPosition`,
`LogDriftRanks`, and `EarliestOutlierPosition` to localize the first
divergence. Per-position output is especially useful for sliding-window,
attention-mask, cache-offset, and RoPE bugs.

When a long-context run drifts but no single operation is obviously wrong,
isolate candidate layers with reference inputs before changing the model. Load
the reference output from layer `N-1`, cast it back to the model dtype, rebuild
any side inputs from reference tensors when possible, and run only layer `N`.
If the isolated layer passes but the chained run drifts, the evidence points to
accumulated numerical drift or backend/reference sensitivity rather than a
localized layer implementation bug. If the isolated layer still fails, drill
into that layer's submodules and compare the smallest operation that reproduces
the failure.

### 5. Context, Cache, And Special Behavior

Add focused tests when the architecture has the corresponding risk:

- long-sequence test: sliding window, RoPE scaling, recurrent state, or sparse
  layer schedules
- cache test: generation-time KV offsets, shared KV layers, recurrent caches,
  or sliding windows
- quantized test: model-specific packed tensors, expert stacking, or mixed
  quantization metadata
- thinking test: chat templates or renderer behavior that affects thinking
  delimiters
- multimodal test: image/audio token accounting or processor-dependent inputs

For cache patterns, see `x/mlxrunner/cache_test.go`.

### 6. Perplexity Validation

After layer comparisons pass, validate end-to-end quality with `x/cmd/ppl`:

```bash
# Default: lm-evaluation-harness wikitext task
go run ./x/cmd/ppl -model mymodel:base-mlx-bf16 -cache-dir /tmp/ollama-ppl-cache

# llama.cpp-compatible mode
go run ./x/cmd/ppl -model mymodel:base-mlx-bf16 -mode llamacpp -cache-dir /tmp/ollama-ppl-cache

# Load directly from a Hugging Face directory
go run ./x/cmd/ppl -model-dir models/mymodel-Base -format json > /tmp/ollama_port/ppl.json
```

Cross-check by running the same model through `lm-eval --tasks wikitext` for
harness mode or `llama-perplexity` for llamacpp mode. A small PPL delta can be
acceptable, but record the baseline, absolute delta, relative delta, corpus,
mode, and context length.

For new models, add a small CI smoke test like
`x/models/gemma4/perplexity_test.go`: it should load a local tag when present,
score a tiny synthetic corpus, and assert only that the result is finite and
plausible.

### 7. Integration Validation

Run integration tests against the created Ollama model tag as final validation,
after the focused reference, cache, quantized, thinking, multimodal, and
perplexity checks above pass. Integration tests are slower and broader; they
should catch API-level packaging and capability problems, not serve as a crutch
for missing unit tests.

Build the local binary first:

```bash
go build .
```

Then run the integration package with the new model override:

```bash
OLLAMA_TEST_MODEL=mymodel:base-mlx-bf16 \
  go test -tags=integration -v -count=1 -timeout 30m ./integration
```

When testing against an already-running local or remote server:

```bash
OLLAMA_TEST_EXISTING=1 \
OLLAMA_HOST=http://127.0.0.1:11434 \
OLLAMA_TEST_MODEL=mymodel:base-mlx-bf16 \
  go test -tags=integration -v -count=1 -timeout 30m ./integration
```

Capability metadata matters. The model manifest/config must advertise only the
capabilities the port actually supports:

- completion models should advertise `completion`
- vision models should also advertise `vision`
- audio models should also advertise `audio`
- tool-capable chat models should advertise `tools`
- thinking models should advertise `thinking`
- embedding models should advertise `embedding`

The integration tests use those capabilities to decide whether tests for
vision, audio, tool calling, embeddings, and thinking should run or skip. If a
completion-only model runs vision/audio/tool tests, fix the advertised
capabilities. If a model supports one of those features but the related tests
skip, fix the created model metadata before treating integration validation as
complete.

Record the integration command, whether `OLLAMA_TEST_EXISTING` was used, the
model tag, and any capability-based skips in the review report.

### 8. Reviewer Report

Collect validation artifacts into a Markdown report:

```bash
python3 x/models/scripts/summarize_validation.py \
    --manifest /tmp/ollama_port/<architecture>/porting_manifest.json \
    --activation-manifest /tmp/ollama_ref/<variant>/activations.safetensors.manifest.json \
    --activation-comparison-json /tmp/ollama_port/activation-comparison.json \
    --go-test-json /tmp/ollama_port/go-test.json \
    --go-test-json /tmp/ollama_port/integration-test.json \
    --ppl-json /tmp/ollama_port/ppl.json \
    --generation-transcript /tmp/ollama_port/generation.md \
    --output /tmp/ollama_port/review_report.md
```

The report should be factual. Do not use it to hide failed checks; list known
skips and unresolved limitations explicitly.

## Reference Key Naming

The Python hooks capture outputs keyed by `model.named_modules()` paths. For
multimodal models, the text model is often nested under
`model.language_model.layers.0`. Map these paths to Go struct fields manually
when writing tests.

## Tolerance Guidelines

| Scenario | Primary check | Threshold |
| --- | --- | --- |
| Single op (embedding, matmul, norm) | `CompareArrays` | `BFloat16Tol()` (atol=5e-3) |
| Single decoder layer output, BF16 | `CompareArraysCosineSim` | 0.9999 |
| Full forward pass, BF16, any depth | `CompareArraysCosineSim` | 0.999 |
| Long-sequence accumulated output | `CompareArraysCosineSim` | 0.99, with diagnostics |
| Quantized model end-to-end | `CompareArraysCosineSim` | 0.99 |

For tensors that have been through a final per-channel scaled norm, use cosine
similarity rather than element-wise tolerance. Element-wise diffs at those
positions are dominated by per-channel norm weights and are not a useful bug
signal. A real bug usually collapses cosine similarity well below 0.99.

## Common Pitfalls

- Embedding scaling: Hugging Face may include `sqrt(hidden_size)` scaling
  inside the embedding module.
- RoPE conventions: Hugging Face often uses `rotate_half` split at the
  midpoint; MLX uses paired rotation. Check partial rotation dimensions.
- Weight prefixes: models use `model.`, `language_model.`, or nested
  multimodal prefixes. Do not hard-code one prefix until inspection confirms
  it.
- Norm scale shift: some norms use `1 + weight`, others use `weight` directly.
- Logits comparison: full logits tensors are huge. Prefer final hidden state
  or `--skip-logits` unless logits are specifically under investigation.
- Dtype contamination: accidental float32 operations can preserve quality but
  hurt speed. Check output dtype and profile if performance is unexpectedly
  poor.
- `Floats()` only works on float32 arrays. `testutil` casts automatically; if
  comparing manually, use `.AsType(mlx.DTypeFloat32)` first.

## Definition Of Done

A new MLX model port is ready for review when:

- representative variants have an inspection manifest
- PyTorch activation references and sidecar manifests exist
- forward-pass tests compare embedding, layer outputs, and final hidden state
- long-context/cache/quantized/thinking/multimodal tests are added when the
  architecture needs them, or the report explains why they are not applicable
- `x/cmd/ppl` has been run against a base model and baseline when available
- integration tests have been run with `OLLAMA_TEST_MODEL` against the created
  tag, with capability-driven skips reviewed
- at least one generation sample is captured for reviewer sanity checking
- all commands, model paths or revisions, prompts, dtype choices, skips, and
  unresolved limitations are recorded

## PR / Review Report Template

````markdown
## Summary
- Architecture:
- Source model(s):
- Transformers revision or package version:
- Agent-assisted: yes/no

## Variant And Config Coverage
- Variants inspected:
- Config differences that affect implementation:
- Tensor prefixes and dtype histogram:
- Risk flags:

## Validation Commands
```bash
# inspect_model
# dump_activations
# compare_activations
# go test
# OLLAMA_TEST_MODEL integration test
# x/cmd/ppl
```

## Numerical Results
- Forward/layer comparison:
- Long-context/cache:
- Quantized:
- Perplexity:
- Integration:

## Generation Samples
- Prompt:
- Output:

## Known Skips Or Limitations
- Missing local artifacts:
- Unsupported variants:
- Follow-up work:
````
