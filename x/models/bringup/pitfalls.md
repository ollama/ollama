# Common Pitfalls

Part of the MLX porting guide (`x/models/PORTING_GUIDE.md`). Bring-up
evidence lives beside these docs in `x/models/bringup/<model>/` — git-ignored,
never committed; see `artifacts.md`.

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
- `ollama create` with `FROM <existing-tag>` preserves all layers and merges
  PARAMETERs — the cheap way to add or adjust parameters on a finished tag.
  Recreating with `FROM <file>` starts from scratch: every PARAMETER,
  RENDERER, PARSER, REQUIRES, and DRAFT line must be respecified or it is
  silently lost.
- After renaming registered architecture strings or rewriting stored model
  configs, rebuild and restart every serving binary (including side builds
  used by benches and soak tests). A binary built before the rename cannot
  load the renamed models and fails at model-load time, not at startup.
- `hf download` applies `--include` to a single pattern; additional
  arguments are treated as explicit filenames and the include filter is
  skipped (a warning is printed). After any multi-file download, verify the
  byte sizes of everything you expected.
- Every MLX-touching entry point needs its goroutine pinned to an OS thread
  (`mlxtest.Setup` in tests, `runtime.LockOSThread` at a CLI main, or the
  `x/internal/mlxthread` worker in production). The failure is a runtime
  panic on first eval, so a tool can compile and sit broken until someone
  actually runs it — see the threading contract in the `x/mlxrunner/mlx`
  package doc.
- `testutil.LoadModelFromDir` bypasses the create-path import transform;
  for architectures that need it, the dir-loaded model is silently wrong.
  Validate via a created tag + `LoadModelByNameOrErr` first (see Layer
  Comparison in `implementation.md`).
- After rebasing this tooling onto a newer runner, execute at least one
  exemplar test with real weights — compiling is not proof. Weights-gated
  tests skip in CI, so interface-adjacent breakage (thread pins, changed
  layer signatures, memory behavior) only surfaces on a live run.
