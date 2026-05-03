# ollama-ppl

A perplexity measurement CLI for MLX-loaded language models.

`ollama-ppl` loads any registered MLX model architecture directly (no
HTTP server, no separate inference process) and runs end-to-end perplexity
on a corpus. It supports two scoring methodologies:

- **`harness`** (default): reproduces EleutherAI lm-evaluation-harness'
  `wikitext` task. Document-level rolling loglikelihood with `context_len=1`,
  scoring every prediction position in each window. The default corpus is
  fetched from the canonical `EleutherAI/wikitext_document_level` Hugging Face
  dataset.
- **`llamacpp`**: reproduces llama.cpp's `llama-perplexity` tool. Concatenates
  the corpus into one stream, splits into non-overlapping `n_ctx`-token chunks
  with BOS substituted at chunk position 0, scores only the second half of
  each chunk. The default corpus is the `wiki.test.raw` flat file from the
  `ggml-org/ci` Hugging Face dataset (the source the llama.cpp community uses).

Note: for optimal results, use the base trained model, not the instruction-tuned version.

## Examples

```bash
# Default: harness mode on an ollama-stored model
go run ./x/cmd/ppl -model mymodel:base-mlx-bf16

# Llama.cpp-compatible mode for direct comparison with llama-perplexity
go run ./x/cmd/ppl -model mymodel:base-mlx-bf16 -mode llamacpp

# Load from a HuggingFace directory (no ollama tag required)
go run ./x/cmd/ppl -model-dir models/SomeOrg/SomeModel-Base

# Keep downloaded corpora in a sandbox-friendly cache
go run ./x/cmd/ppl -model-dir models/SomeOrg/SomeModel-Base \
  -cache-dir /tmp/ollama-ppl-cache

# Compare against a prior JSON result and fail if relative token PPL drifts >1%
go run ./x/cmd/ppl -model mymodel:base-mlx-bf16 \
  -baseline /tmp/baseline-ppl.json \
  -max-rel-delta 0.01
```

## Cross-validation

To compare against a canonical reference:

```bash
# lm-evaluation-harness reference (default harness mode)
pip install lm-eval
python -m lm_eval --model hf \
    --model_args pretrained=models/SomeOrg/SomeModel-Base,dtype=bfloat16,trust_remote_code=True,max_length=2048 \
    --device mps --tasks wikitext

# llama.cpp reference (llamacpp mode), after converting HF→GGUF
python ~/code/llama.cpp/convert_hf_to_gguf.py models/SomeOrg/SomeModel-Base \
    --outtype f16 --outfile /tmp/model-f16.gguf
llama.cpp/build/bin/llama-perplexity \
    -m /tmp/model-f16.gguf \
    -f /tmp/ollama-bench-data/wiki.test.raw -c 512 -ngl 99
```
