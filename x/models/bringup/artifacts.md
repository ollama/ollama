# Bring-Up Artifacts, Reviewer Report, And PR Archive

Part of the MLX porting guide (`x/models/PORTING_GUIDE.md`). This directory
holds the working evidence produced while porting a model; everything here
except these guide documents and `.gitignore` is git-ignored.

Bring-up evidence — reference activation dumps, parity and trajectory
reports, disposable harness scripts, bench results, session notes — is
deliberately kept out of repository history, but it is NOT disposable: the
archive built from this tree is the artifact a reviewer or the model
publisher uses to verify that the port faithfully implements the reference.

## Rules

- All bring-up work for a port lives under `x/models/bringup/<model>/` from
  day one. Do not scatter evidence across `/tmp`, session scratch
  directories, or ad-hoc cache paths — those do not survive to review time.
- Nothing under `<model>/` is ever committed. The PR carries an archive
  instead.

Scaffold a new port's tree (layout below plus a LEDGER template):

```bash
x/models/scripts/new_port.sh <model>
```

## Layout

```
x/models/bringup/<model>/
  LEDGER.md      # provenance: source repos, revisions, per-file content
                 # hashes, authority order; the forward-operation/dtype
                 # ledger; running decision log
  MANIFEST.json  # every file: sha256, bytes, archived|regenerable, and the
                 # command that regenerates it
  reference/     # activation dumps + sidecar manifests, reference configs,
                 # tokenizer/template captures
  parity/        # artifact ABI reports, renderer byte/token-ID reports,
                 # logit and teacher-forced trajectory comparisons, parser
                 # replay outputs, quantization-vs-BF16 replays
  perf/          # benchmark outputs, A/B results, profile summaries
  scripts/       # every disposable harness written during bring-up
  sessions/      # engineering/agent session notes (internal-only; excluded
                 # from the external archive flavor)
```

## Reviewer Report

Collect validation artifacts into a Markdown report:

```bash
python3 x/models/scripts/summarize_validation.py \
    --manifest x/models/bringup/<model>/reference/porting_manifest.json \
    --activation-manifest x/models/bringup/<model>/reference/<variant>/activations.safetensors.manifest.json \
    --activation-comparison-json x/models/bringup/<model>/parity/activation-comparison.json \
    --go-test-json x/models/bringup/<model>/parity/go-test.json \
    --go-test-json x/models/bringup/<model>/parity/integration-test.json \
    --artifact-abi-report x/models/bringup/<model>/parity/artifact-abi.md \
    --ppl-json x/models/bringup/<model>/parity/ppl.json \
    --generation-transcript x/models/bringup/<model>/parity/generation.md \
    --output x/models/bringup/<model>/review_report.md
```

The report should be factual. Do not use it to hide failed checks; list known
skips and unresolved limitations explicitly.

## Archive

Before opening the model-support PR, generate `MANIFEST.json`
(`python3 x/models/scripts/make_manifest.py x/models/bringup/<model>`; large
regenerable artifacts and their rebuild commands go in the tree's
`ARCHIVE_EXCLUDE.json`), build the archive, and attach it:

```bash
tar --exclude='reference/*.safetensors' \
    --exclude='sessions' \
    -czf /tmp/<model>-bringup-$(date +%Y%m%d).tgz \
    -C x/models/bringup <model>
```

- Target size is ~20MB or less: summaries, manifests, ledgers, and scripts
  always go in; multi-GB raw tensor dumps stay out but must be listed in
  `MANIFEST.json` as regenerable with the exact command.
- Attach the archive to the PR. If it exceeds the attachment limit, place it
  in the team artifact share and link it from the PR body.
- Drop `--exclude='sessions'` only for internal review copies; the external
  flavor (for the model publisher) always excludes `sessions/`.

The archive — not the repository — is where reviewers and the model
publisher verify the port against the reference implementation.

## PR / Review Report Template

````markdown
## Summary
- Architecture:
- Source model(s):
- Transformers revision or package version:
- Agent-assisted: yes/no

## Variant And Config Coverage
- Variants inspected:
- Config differences that affect implementation:
- Tensor prefixes and dtype histogram:
- Risk flags:

## Validation Commands
```bash
# inspect_model
# artifact ABI comparison
# dump_activations
# compare_activations
# go test
# OLLAMA_TEST_MODEL integration test
# x/cmd/ppl
```

## Numerical Results
- Forward/layer comparison:
- Long-context/cache:
- Quantized:
- Perplexity:
- Integration:

## Artifact ABI
- Publisher source vs created tag:
- Previous public tag vs replacement tag:
- Approved metadata drift:

## Generation Samples
- Prompt:
- Output:

## Known Skips Or Limitations
- Missing local artifacts:
- Unsupported variants:
- Follow-up work:

## Bring-Up Archive
- Attachment or artifact-share link:
- MANIFEST.json regenerable-artifact count:
````

## Tooling Gaps

- **`summarize_validation.py` gate completeness.** Extend the report to mark
  the equivalence gate INCOMPLETE whenever a required artifact (token-ID
  parity, trajectory report, parser replay, raw-token replay) is missing, so
  absent evidence is always visible to reviewers.
