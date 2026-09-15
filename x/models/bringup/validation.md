# Perplexity And Integration Validation

Part of the MLX porting guide (`x/models/PORTING_GUIDE.md`). Bring-up
evidence lives beside these docs in `x/models/bringup/<model>/` — git-ignored,
never committed; see `artifacts.md`.

## Perplexity Validation

After layer comparisons pass, validate end-to-end quality with `x/cmd/ppl`:

```bash
# Default: lm-evaluation-harness wikitext task
go run ./x/cmd/ppl -model mymodel:base-mlx-bf16 -cache-dir /tmp/ollama-ppl-cache

# Window mode for cross-ecosystem comparison
go run ./x/cmd/ppl -model mymodel:base-mlx-bf16 -mode window -cache-dir /tmp/ollama-ppl-cache

# Load directly from a Hugging Face directory
go run ./x/cmd/ppl -model-dir models/mymodel-Base -format json > x/models/bringup/<model>/parity/ppl.json
```

For model-quality comparisons, use the canonical downloaded corpus by omitting
`-corpus`, and use either the full default run or an explicitly documented
multi-document subset such as `-max-docs 8` for iteration. Do not use the tiny
synthetic corpus from smoke tests for quality tables; it is only a finite-result
sanity check and can give misleading ordering between quantization variants.

Cross-check harness mode by running the same model through
`lm-eval --tasks wikitext`; window mode is directly comparable with numbers
published by other runtimes' perplexity tools. A small PPL delta can be
acceptable, but record the baseline, absolute delta, relative delta, corpus,
mode, and context length. Running both modes is itself a useful cross-check:
their token PPLs should agree closely on the same model and corpus.

Compare on **token perplexity** in window mode: its word/byte perplexity
denominators span the whole corpus while only about half the tokens are
scored, so those two sub-metrics are not meaningful there.

For new models, add a small CI smoke test like
`x/models/gemma4/perplexity_test.go`: it should load a local tag when present,
score a tiny synthetic corpus, and assert only that the result is finite and
plausible.

## Integration Validation

Run integration tests against the created Ollama model tag as final validation,
after the focused reference, cache, quantized, thinking, multimodal, and
perplexity checks above pass, and after the artifact ABI report has been
produced. Integration tests are slower and broader; they should catch
API-level packaging and capability problems, not serve as a crutch for missing
unit tests or metadata compatibility checks.

Before running integration, compare the created tag's manifest/config/tensor
metadata against the publisher source and, for replacement tags, against the
existing public tag. Record tensor namespace, config-schema, tokenizer/
processor/generation metadata, params, capabilities, auxiliary-component, and
`REQUIRES` deltas in `x/models/bringup/<model>/parity/artifact-abi.md`.
Passing integration tests through a permissive loader does not prove the
artifact ABI is compatible.

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
