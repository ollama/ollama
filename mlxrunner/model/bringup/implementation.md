# Implementation And Layer Comparison

Part of the MLX porting guide (`mlxrunner/model/PORTING_GUIDE.md`). Bring-up
evidence lives beside these docs in `mlxrunner/model/bringup/<model>/` — git-ignored,
never committed; see `artifacts.md`.

## Implementation

Implement the Go model in `mlxrunner/model/<name>/` using nearby models as templates.
Keep the first version self-contained and model-specific. Shared utility
changes should be small, justified by more than one model, and called out in
the review report.

`testutil.LoadModelFromDir(t, dir)` loads any registered architecture from a
directory containing `config.json`, `tokenizer.json`, and `*.safetensors`.
It creates a synthetic manifest and uses the standard `model.New()` factory
dispatch.

## Layer Comparison

Write forward-pass tests using `testutil`. Every MLX-touching test runs its
body in `mlxtest.Run(t, func(t *mlxtest.T) { ... })` — it skips when MLX is
unavailable AND runs the body on the shared MLX thread. That thread is
load-bearing: MLX default streams are thread-local, and MLX work on any other
thread panics at eval with "There is no Stream(...) in current thread"
(threading contract: `mlx` package doc). The testutil helpers take a
`testutil.TB`, which the `*mlxtest.T` satisfies.

```go
mlxtest.Run(t, func(t *mlxtest.T) {
mlx.Scoped(func() { // frees the model, references, and activations at test end
model := testutil.LoadModelFromDir(t, testutil.ModelDir(t, "MODEL_DIR", "models/<name>"))
ref := testutil.LoadReference(t, filepath.Join(testutil.DefaultRefDir("variant"), "activations.safetensors"))

h, _ := model.Forward(&batch.Batch{
    InputIDs:     tokens,
    SeqOffsets:   []int32{0},
    SeqQueryLens: []int32{seqLen},
}, caches)
testutil.CompareArraysCosineSim(t, "final_hidden", h, ref["model.norm"], 0.999)

embed := model.EmbedTokens.Forward(tokens)
testutil.CompareArrays(t, "embed_tokens", embed, ref["model.embed_tokens"], testutil.BFloat16Tol())
})
})
```

Loader choice matters more than it looks:

- **Prefer the production import path.** Create the Ollama tag first and load
  it with `testutil.LoadModelByNameOrErr`. `LoadModelFromDir` on a raw HF
  directory bypasses the create-path import transform — for architectures
  whose runtime layout depends on it (renamed or synthesized tensors), the
  dir-loaded model is silently wrong: it loads, runs, and produces garbage
  (a final-hidden cosine of ~0.17 looks exactly like a model bug). Validate
  through the tag path first; only use dir-loading once it's proven
  equivalent for the architecture.
  Loading through the production tag is still not an artifact metadata check:
  it proves the loader accepts the tag, not that the tag preserves the
  publisher or existing-public tensor namespace/config schema. Pair every
  production-tag parity run with the artifact ABI report from
  `release-gates.md`.
- **Arrays live until their MLX scope ends** (`mlx` package, `scope.go`).
  Arrays a test creates outside any scope live until the process exits; wrap
  the test body in `mlx.Scoped` so the model, references, and activations are
  freed when the test ends, and use `mlx.Scoped`/`mlx.ScopedArrays` around
  per-iteration work in loops. An array used after its scope ended panics
  with "used after its scope ended".
- **Very large models: one test per process if memory is tight.** A test
  whose body is not scoped holds its weights until the process exits, so
  sequential loads of a 30B-class model OOM by the third test. Scope the
  body (the maintained examples do), or run with `-run 'TestName'` per
  invocation.

`mlxrunner/model/gemma4/forward_test.go` and `mlxrunner/model/qwen3_5/forward_test.go` are
maintained examples of this pattern, including per-layer walks that mirror the
production forward loop.

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

## Context, Cache, And Special Behavior

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

For cache patterns, see `mlxrunner/cache/cache_test.go`. Renderer and parser
coverage is model correctness too — see `renderer-parity.md`.

For multimodal models, mind prefill buffer-lifetime ordering (see the
commented block in `mlxrunner/pipeline.go`): the first eval that touches
the media graph must run after the sweep — an eval before the sweep cannot
free any buffer the chunk's live handles retain, which on media chunks pins
the entire vision tower and shows up as a memory explosion, not a wrong
answer. Release media buffers only after the drafter's committed callback,
since a deferred draft flush may still embed rows from them. Memory-lifetime
bugs like this pass every numerical-parity test; validate multimodal ports
with a peak-memory check on a large-media request, not correctness tests
alone.
