# Release Gates For Published Models

Part of the MLX porting guide (`x/models/PORTING_GUIDE.md`). Bring-up
evidence lives beside these docs in `x/models/bringup/<model>/` — git-ignored,
never committed; see `artifacts.md`.

These gates apply when the port ships as published tags, and matter most for
embargoed partner models. Run them early and re-run them before any push —
they are quick, and they make a launch predictable.

- **Provenance pinning.** Identify the publisher's PRIMARY artifact (usually
  the native training checkpoint) versus derived distributions (converted
  checkpoints, quantized drops); record per-file content hashes for every
  drop in the bring-up `LEDGER.md`. On any refresh, diff hashes per file
  rather than relying on repository timestamps or commit messages. When a
  derivation must be regenerated, run the publisher's own conversion tooling
  from the primary and byte/numeric compare against the previously used
  artifacts before shipping.
- **Published artifact ABI.** Tensor names and namespaces, config-schema
  shape, tokenizer/processor/generation metadata, params, capabilities,
  auxiliary layers, and `REQUIRES` are part of the artifact ABI. Before any
  push, produce a report comparing:
  1. the publisher source artifact to the final Ollama tag, and
  2. the existing public tag to the replacement tag when overwriting.
  A loader that accepts multiple tensor layouts does not make those layouts
  interchangeable for published models. If the created tag renames tensors,
  flattens or nests config differently, drops or synthesizes metadata, changes
  auxiliary-component layout, or changes the minimum runtime requirement, stop
  until the drift is explicitly approved and documented. Replacing a public
  tag must preserve its current ABI unless the release owner approves a
  breaking metadata migration.
- **Embargo and codename hygiene.** Before any push or code publication,
  sweep for internal codenames across: source code, every JSON layer of every
  model manifest (configs, processor configs, draft configs), embedded
  artifact metadata (including chat templates), and user-visible surfaces
  (`ollama show`, `/api/show` model_info). Take final naming from the
  publisher's own tooling — their converter or transformers package defines
  the intended class names and model_type.
- **Published-defaults gate.** Extract the publisher's recommended sampling
  parameters (model card best practices, generation_config) and assert that
  every tag carries them as PARAMETERs, so users get the publisher's
  recommended experience out of the box.
- **Capability-intent assertion.** Write down the intended capability set per
  tag and assert `show` reports exactly that. Some capabilities are derived
  at load time from artifact contents and can change when an auxiliary
  component (for example a vision projector) is absent — verify the full
  intended set on every tag, and make sure the test harness reports
  capability-based skips loudly so coverage gaps are visible.
- **Benchmark through the shipping stack.** Ship/no-ship performance numbers
  come from the full server path that users run, not from standalone runner
  invocations. Standalone flag recipes can select different code paths —
  speculative-decoding settings are especially sensitive — so treat them as
  diagnostics only.
- **Artifact metadata.** Conversion and quantization runs should set the
  intended model name up front (`--model-name` or a properly named source
  directory); tools embed the source name in artifact metadata, and embedded
  metadata can only be changed by rewriting the artifact.
