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
- **`window`**: scores a concatenated token stream in fixed windows.
  Concatenates the corpus into one stream, splits into non-overlapping
  fixed-size chunks with BOS substituted at chunk position 0, and scores only
  the second half of each chunk. This methodology is widely used by other
  runtimes' perplexity tools, so results are directly comparable across
  ecosystems. The default corpus is a standard `wiki.test.raw`
  (wikitext-2-raw) mirror.

Note: for optimal results, use the base trained model, not the instruction-tuned version.

## Examples

```bash
# Default: harness mode on an ollama-stored model
go run ./x/cmd/ppl -model mymodel:base-mlx-bf16

# Window mode for cross-ecosystem comparison
go run ./x/cmd/ppl -model mymodel:base-mlx-bf16 -mode window

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

To compare harness mode against an independent reference:

```bash
pip install lm-eval
python -m lm_eval --model hf \
    --model_args pretrained=models/SomeOrg/SomeModel-Base,dtype=bfloat16,trust_remote_code=True,max_length=2048 \
    --device mps --tasks wikitext
```

Window mode uses the same corpus, chunking, and scoring conventions as other
runtimes' perplexity tools, so those numbers can be compared directly when a
reference implementation publishes them.
