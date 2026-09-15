#!/bin/sh
# Scaffold a bring-up tree for a new model port:
#   x/models/scripts/new_port.sh <model-name>
# Creates x/models/bringup/<model-name>/ with the standard layout and a
# LEDGER.md template. Everything under the tree is git-ignored by design;
# see x/models/bringup/artifacts.md.
set -e
if [ -z "$1" ]; then
    echo "usage: $0 <model-name>" >&2
    exit 2
fi
ROOT="x/models/bringup/$1"
if [ -e "$ROOT/LEDGER.md" ]; then
    echo "$ROOT already scaffolded" >&2
    exit 1
fi
mkdir -p "$ROOT/reference" "$ROOT/parity" "$ROOT/perf" "$ROOT/scripts" "$ROOT/sessions"
cat > "$ROOT/LEDGER.md" <<'TEMPLATE'
# <model> — provenance and operation ledger

## Authority order (publisher-designated)

1. PRIMARY: <artifact, repo, revision, per-file content hashes>
2. Derived: <converted checkpoints and the tooling that produced them>

Reference implementation for activation capture: <package/wheel + class>.

## Architecture summary

<layers, attention layout, dims, vocab, special scales/softcaps, modalities>

## Forward-operation/dtype ledger

| Stage | Reference behavior | Ollama code |
| --- | --- | --- |
| Embedding | | |
| Norms / QK handling | | |
| RoPE | | |
| Attention schedule | | |
| LM head + output transforms | | |

## Verification log

- [ ] Source provenance pinned (hashes recorded above)
- [ ] Activation reference dump (publisher implementation, eager)
- [ ] Forward-pass tests (embed, per-layer, final hidden)
- [ ] Output-head parity (raw + post-transform logits)
- [ ] Reference equivalence gate (see bringup/equivalence-gate.md)
- [ ] Renderer/parser parity (VERIFY_JINJA2 run, not skipped)
- [ ] Perplexity baselines (harness + window)
- [ ] Artifact ABI report (publisher source vs created tag; old public tag vs replacement)
- [ ] Integration pass with capability skips reviewed
TEMPLATE
echo "scaffolded $ROOT"
