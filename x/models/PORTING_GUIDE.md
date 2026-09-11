# MLX Model Porting Guide

This is the overview of the repeatable process for bringing a new model
architecture up on Ollama's MLX runner. The process is reference-driven:
start from the publisher's authoritative implementation, capture reproducible
reference artifacts, implement the same forward pass in Go/MLX, and collect
enough evidence that a reviewer can verify both the code and the validation.
Correctness comes before performance tuning. Treat every discrepancy as a
defect in the port until the evidence proves otherwise.

Each step below links to a deep-dive document in `x/models/bringup/` — read
the deep dive when you reach that step, not before. Every artifact the
workflow produces lives under `x/models/bringup/<model>/` from day one
(git-ignored, never committed, NOT disposable — it becomes the reviewer
archive attached to the PR; see `bringup/artifacts.md`). Do not write
bring-up evidence to `/tmp` or session scratch directories.

Supporting tools: `x/models/scripts/` (inspect_model, dump_activations,
compare_activations, summarize_validation — see `scripts/README.md`),
`x/models/testutil` (Go comparison helpers), and `x/cmd/ppl` (perplexity CLI).
Agent standards live in `x/models/AGENTS.md`.

## Workflow

1. **Discovery** — inspect representative variants with `inspect_model.py`;
   review the porting manifest for config variance, tensor prefixes, RoPE,
   sliding windows, MoE, multimodal fields, and dtype histograms before
   coding. → `bringup/reference-capture.md`

2. **Reference capture** — record the publisher's authority order, revisions,
   and checkpoint hashes; build a forward-operation/dtype ledger; dump
   PyTorch activations with sidecar manifests into the bring-up tree.
   → `bringup/reference-capture.md`

3. **Implementation** — implement the Go model in `x/models/<name>/` using
   nearby models as templates; keep it self-contained and model-specific.
   → `bringup/implementation.md`

4. **Layer comparison** — forward-pass tests against the reference dumps;
   localize the first divergence layer by layer, isolating layers with
   reference inputs before blaming accumulation.
   → `bringup/implementation.md`

5. **Context, cache, and special behavior** — focused long-sequence, cache,
   quantized, thinking, and multimodal tests where the architecture carries
   the risk; renderer/parser parity is model correctness, with strict
   anti-circularity rules for parity tests.
   → `bringup/implementation.md`, `bringup/renderer-parity.md`

6. **Reference equivalence gate** — mandatory on BF16 before quantization,
   performance work, or any completeness claim: exact prompt bytes and token
   IDs, prefill/cache/output-head parity, teacher-forced trajectories, parser
   replay, deterministic replay, and a first-divergence discrepancy protocol.
   → `bringup/equivalence-gate.md`

7. **Perplexity validation** — `x/cmd/ppl` end-to-end quality scoring with a
   recorded baseline and corpus methodology. → `bringup/validation.md`

8. **Integration validation** — `OLLAMA_TEST_MODEL` against the created tag
   as final validation; capability metadata drives test selection, so verify
   the advertised capability set exactly. Before this step, produce the
   artifact ABI report comparing publisher source metadata to the created
   Ollama tag, and old public tag metadata to any replacement tag.
   → `bringup/validation.md`, `bringup/release-gates.md`

9. **Performance tuning** — second pass only: baseline first, prefer existing
   MLX ops and compiled closures over custom kernels, audit host-side tensor
   scaling, and re-run correctness after every optimization.
   → `bringup/performance.md`

10. **Reviewer report and archive** — build the validation report, complete
    `MANIFEST.json`, and attach the bring-up archive to the PR.
    → `bringup/artifacts.md`

For models that ship as published tags (especially embargoed partner models),
run the release gates — provenance pinning, codename hygiene, published
sampling defaults, capability intent, shipping-stack benchmarks, artifact
metadata — before any push. → `bringup/release-gates.md`

Known traps live in `bringup/pitfalls.md`.

## Definition Of Done

A new MLX model port is ready for review when:

- representative variants have an inspection manifest
- the authoritative implementation and revision are recorded, and the
  forward-operation ledger accounts for every explicit dtype cast and output
  transform
- PyTorch activation references and sidecar manifests exist
- forward-pass tests compare embedding, layer outputs, and final hidden state
- the reference equivalence gate has passed on BF16: exact prompt bytes and
  production token IDs, prefill and cached decode, raw and transformed
  logits, a teacher-forced trajectory, streaming parser round-trip, and
  deterministic raw-token replay
- every divergence is either fixed in the port or supported by a preserved
  minimal reference-defect/backend-drift reproduction; there are no
  unexplained high-margin token-ranking mismatches
- long-context/cache/quantized/thinking/multimodal tests are added when the
  architecture needs them, or the report explains why they are not applicable
- `x/cmd/ppl` has been run against a base model and baseline when available
- integration tests have been run with `OLLAMA_TEST_MODEL` against the created
  tag, with capability-driven skips reviewed
- the created artifact's tensor names, config schema, tokenizer/processor/
  generation metadata, params, capabilities, auxiliary components, and
  `REQUIRES` have been compared against the publisher source and any previous
  public tag; every drift is explicitly approved or fixed
- at least one generation sample is captured for reviewer sanity checking
- performance baseline and any tuning results are recorded after correctness
  validation, or the report explains why performance tuning was deferred
- host-built tensors are audited for scaling class (nothing O(n²) in tokens
  or patches is constructed CPU-side), and peak memory plus upload volume
  were recorded for a representative large-media/long-context request
- all commands, model paths or revisions, prompts, dtype choices, skips, and
  unresolved limitations are recorded
- the bring-up tree under `x/models/bringup/<model>/` is complete, its
  `MANIFEST.json` lists every artifact as archived or regenerable, and the
  archive is attached to the PR (template in `bringup/artifacts.md`)
- for published tags, the release gates have run: provenance hashes pinned,
  codename sweep clean, publisher sampling defaults asserted, capability
  intent verified, artifact ABI report complete, and ship benchmarks taken
  through the ollama stack
