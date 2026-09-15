# Discovery And Reference Capture

Part of the MLX porting guide (`x/models/PORTING_GUIDE.md`). Bring-up
evidence lives beside these docs in `x/models/bringup/<model>/` — git-ignored,
never committed; see `artifacts.md`.

## Discovery

Start by inspecting representative model variants:

```bash
python3 x/models/scripts/inspect_model.py \
    --model models/<org>/<small-base> \
    --model models/<org>/<larger-base> \
    --output x/models/bringup/<model>/reference
```

Review `porting_manifest.md` before coding. Look for:

- config fields that vary across variants
- tensor prefixes/namespaces, config-schema shape, and tied embedding behavior
- RoPE parameters, partial rotation, and scaling
- sliding-window or hybrid layer patterns
- MoE expert shapes and routing config
- attention bias, grouped-query attention, MLA, recurrent or convolution state
- multimodal processor fields
- thinking tags in chat templates
- safetensors dtype histogram and quantization metadata

## Reference Capture

Identify the publisher-designated source of truth before capturing anything.
Publishers often provide several implementations — native PyTorch,
Transformers, vLLM, example runners — and they can differ in subtle ways.
Use whichever one the publisher designates as authoritative, and record
revisions, checkpoint hashes, and the stated authority order.

Be especially careful with pre-release and partner drops: they typically
aggregate contributions from multiple teams, so a download set can contain
the primary reference alongside derived implementations — converted
checkpoints, compatibility ports, example scripts — and multiple branches at
different stages. Derived implementations can lag the primary or carry their
own defects, and a checkout can silently include stale secondary copies.
Identify the primary explicitly (exact repository, branch, and revision),
treat it as the only source of truth, and validate every derived
implementation against it before relying on one. Re-establish which artifact
is primary on every refresh. Build a forward-operation ledger
before coding: map each authoritative reference stage to Ollama code and a
test, including input/output dtype, explicit casts, scales, normalization and
residual order, masks, RoPE, cache updates, LM head, output
multipliers/softcaps, and token suppression. On every publisher refresh, diff
the authoritative code, config, tokenizer, template, and inference defaults
before reusing reference artifacts; unchanged tensor shapes do not prove
unchanged semantics.

Tensor namespace and config-schema shape are part of the reference contract.
Do not normalize names, flatten nested configs, or treat two publisher
distributions as equivalent because the runtime can be taught to load both.
If a derived checkpoint uses different tensor names or config key paths from
the publisher-designated source, record that as artifact ABI drift and resolve
which layout the shipped tag must preserve before writing model or create code.
For replacements of existing public tags, the public tag's current layout is
also a compatibility contract.

Generate a PyTorch reference from the `transformers` implementation:

```bash
.venv/bin/python3 x/models/scripts/dump_activations.py \
    --model models/<org>/<name> \
    [--model-class MyModelForCausalLM] \
    [--attn-implementation eager] \
    [--transformers-path path/to/custom/transformers/src] \
    --skip-logits
```

Pass `--output x/models/bringup/<model>/reference/<variant>/activations.safetensors`
so the dump and its sidecar manifest land in the bring-up tree (the default
is `/tmp/ollama_ref/<variant>/`, which does not survive to review time).

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
    --json-output x/models/bringup/<model>/parity/activation-comparison.json \
    --markdown-output x/models/bringup/<model>/parity/activation-comparison.md
```

This is a reference-quality check, not a replacement for Go unit tests. It
helps distinguish an unstable reference from a model implementation defect.

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

## Tooling Gaps

Known gaps in `dump_activations.py`; missing tooling is not a waiver — extend
the script or build a disposable harness and note it in the review report:

- **Original dtype metadata.** Captures re-cast to float32 can erase the
  original dtypes, which hides cast differences on the output path. The
  sidecar manifest should record each tensor's original dtype.
- **Last-position logit capture.** Full-sequence logits are so large that
  `--skip-logits` becomes the habit — which hides output-transform bugs. Add
  a mode capturing raw LM-head and post-transform logits for just the last
  position.
