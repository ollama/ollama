# x/models/scripts

Helper scripts for porting and validating MLX model implementations.
None of these are required at runtime — they exist to support the
implementation/debugging workflow described in
[`../PORTING_GUIDE.md`](../PORTING_GUIDE.md).

## Available scripts

### `inspect_model.py`

Reads one or more Hugging Face model directories (or Hub IDs when
`huggingface_hub` is installed), inspects config/tokenizer files and
safetensors headers, and writes `porting_manifest.json` plus
`porting_manifest.md`.

```
python3 x/models/scripts/inspect_model.py \
    --model models/MyOrg/MyModel-Base \
    --output /tmp/ollama_port/my-model
```

### `dump_activations.py`

Captures per-module forward outputs from a PyTorch reference
implementation (any model loadable through `transformers`) and writes
them to a `.safetensors` file. The Go-side `forward_test.go` files load
this reference via `testutil.LoadReference` and compare it position by
position against the MLX implementation's outputs.

```
.venv/bin/python3 x/models/scripts/dump_activations.py \
    --model models/MyOrg/MyModel-Base \
    --attn-implementation eager \
    --model-class MyModelForCausalLM
```

See the script's `--help` for the full flag list, including filter
patterns for capturing only a subset of submodules and a
`--transformers-path` override for using a custom `transformers` source
checkout. Pin the attention implementation when Transformers offers multiple
backends for the model. The script also writes a deterministic sidecar manifest
next to the safetensors output by default, including the resolved attention
backend and whether the forward pass used cache state.

For cache validation, run the prompt as a cached prefill and capture a
follow-up decode pass:

```
.venv/bin/python3 x/models/scripts/dump_activations.py \
    --model models/MyOrg/MyModel-Base \
    --attn-implementation eager \
    --prompt "$(cat /tmp/long-prompt.txt)" \
    --decode-text " the" \
    --skip-logits
```

### `compare_activations.py`

Compares two safetensors activation artifacts one tensor at a time and writes
ranked JSON/Markdown drift reports. Use it to compare two reference backends
or to summarize a focused candidate-vs-reference activation dump without
loading the whole artifact set into memory at once.

```
python3 x/models/scripts/compare_activations.py \
    --got /tmp/ollama_ref/my-model/eager.safetensors \
    --want /tmp/ollama_ref/my-model/sdpa.safetensors \
    --filter "model.layers.*" \
    --json-output /tmp/ollama_port/my-model/activation-comparison.json \
    --markdown-output /tmp/ollama_port/my-model/activation-comparison.md
```

### `summarize_validation.py`

Builds a reviewer-ready Markdown report from the non-agentic artifacts
produced during a port: the inspection manifest, activation manifests,
activation comparison JSON, `go test -json` output, `x/cmd/ppl -format json`
output, and optional generation samples.

```
python3 x/models/scripts/summarize_validation.py \
    --manifest /tmp/ollama_port/my-model/porting_manifest.json \
    --activation-manifest /tmp/ollama_ref/my-model/activations.safetensors.manifest.json \
    --activation-comparison-json /tmp/ollama_port/my-model/activation-comparison.json \
    --go-test-json /tmp/ollama_port/my-model/go-test.json \
    --ppl-json /tmp/ollama_port/my-model/ppl.json \
    --output /tmp/ollama_port/my-model/review_report.md
```
