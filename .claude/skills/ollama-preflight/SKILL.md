---
name: ollama-preflight
description: Validate a freshly built ollama image before deploying it — run the in-repo pre-deploy regression harness against a host, read the pass/fail diff, and diagnose failures. Use when someone has built a new ollama image and wants it verified, asks to "check this build", "run preflight", "regress the new image", or is deciding whether a tag is safe to deploy to the CUDA, ROCm/gfx1151, or Apple Silicon host. Also use when a vision check disagrees between hosts, or when a harness run reports CONTENTION, NEEDS_BASELINE, or an unknown (platform, version) combination. NOT for authoring new probes or changing expected values — that is a code change in docs/maxusai/vision-suite/preflight/.
---

# Pre-deploy validation for ollama images

This skill owns the **procedure**. The assertions and expected values live in the
repo at `docs/maxusai/vision-suite/preflight/` and are versioned with the payload
they describe. Never re-implement a check here, and never edit an expected value
to make a run go green — see the harness README's "Adding an expectation".

## Before you start

Establish three things. Guessing any of them wastes a twenty-minute run.

1. **Which host, and is it free?** The harness detects contention, but a
   contended run still costs you the time. The live service on `:11434` is
   frequently down and must not be started as a side effect of validating.
2. **Which port is actually the build under test?** 11434, 11435 and 11436 are
   all occupied on 10.8.0.6 — `:11435` currently answers version `0.9.6`, an
   entirely different server. Never assume; the harness asserts the version
   string first and aborts if it disagrees, which is the safety net, not a
   substitute for checking.
3. **Which platform profile applies.** `cuda`, `rocm`, `apple-silicon`, or
   `apple-silicon-mlx`. ROCm is gated at the 0.32.1 base
   (`docs/maxusai/amd-upgrade-gate.md`) and served from `release/0.32.1-dynres`.
   Note what the gate does and does not pin: it blocks the 0.32.5 base and its
   b10091 payload, **not** the compat patches — that lineage carries 002/004/005
   as adapted backports. So ROCm has the *same patch list* as CUDA over a
   *different payload*, and its numbers are still not the CUDA ones. Identical
   patch lists do not imply identical token counts. On a Mac the platform names
   the serving stack, which version alone cannot: `apple-silicon` is the
   llama.cpp path, `apple-silicon-mlx` is the MLX-store server (conventionally
   `:11436`, `OLLAMA_MODELS=~/.ollama/models-mlx`).

## Order of work

Cheapest and most decisive first.

**1. The no-GPU gates (seconds).** Run these before touching a server; they catch
regressions with no model load at all.

```bash
python3 docs/maxusai/vision-suite/preflight/test_verdicts.py
```

That pins the harness's own verdict logic and the internal consistency of
`expectations.toml`. Run it after *any* edit to the expectations file — it is
what catches a ladder edited to flat without its `scaling` field being updated,
which would silently invert the verdict.

```bash
go test ./llm/ -run TestImageTokensForSize
```

That pins the sizing formulas. Go is not installed on this host — run it in a
container (see the project's Go verification note; the `-u 1000:1000` flag
matters).

**2. A smoke run against the target (~2 minutes per arch).** Skips the probes
that force model reloads.

```bash
python3 docs/maxusai/vision-suite/preflight/preflight.py \
    --host http://127.0.0.1:11437 --platform cuda \
    --image-tag maxusai/ollama:4987dd49-dynres \
    --skip-pinned --out runs/smoke.json
```

If the version gate fails, stop. Nothing below it is attributable to this build.

**3. The full run (~5-10 minutes per arch; longer with `--quality`).** Drop
`--skip-pinned` to include the pinned-budget probe — that is the 005 defect
class, and it costs two full model reloads. Add `--quality` to score extraction
via `vision_suite.py`; that is the slow part and can be run separately.

Long runs must be detached. A backgrounded run has been SIGTERM'd (exit 143)
mid-suite:

```bash
setsid nohup python3 docs/maxusai/vision-suite/preflight/preflight.py \
    --host http://127.0.0.1:11437 --platform cuda --quality \
    --out runs/rc1.json > runs/rc1.log 2>&1 < /dev/null &
```

**Do not poll with `pgrep -f preflight.py`** — the pattern matches the checking
shell's own command line, so the loop never exits. Two waiters hung for hours on
this. The result file is written after every check, so just read it:

```bash
python3 -c "import json;d=json.load(open('runs/rc1.json'));print(d['summary']);[print(r['status'],r['check'],r.get('arch') or '',r['summary']) for r in d['results']]"
```

## Reading the verdict

| exit | meaning | what to do |
|---|---|---|
| 0 | passed | deploy |
| 1 | a check failed | read the diagnosis; see below |
| 2 | unknown `(platform, version)`, or the version gate rejected the server | you are pointed at the wrong port, or this build needs a new profile |
| 3 | contention | another client is on the endpoint; stop it and re-run |
| 4 | `NEEDS_BASELINE` | this combination has never been measured — it is not validated |

Exit 4 is not a harness bug. It means the expectations file honestly records that
nobody has measured this (platform, arch) combination, and the harness refuses to
imply a pass. Either measure it and add it, or acknowledge the gap explicitly
with `--allow-unmeasured`.

## Diagnosing failures

The harness prints a diagnosis with each failure. Trust it over your intuition on
these three, which have each burned real time:

- **A flat token ladder is not automatically a failure.** It means an unpatched
  payload for `nemotron_h_omni`, but it is the *correct* result for `gemma4`
  under 004, which budget-fills every image to the ceiling. The verdict is
  per-arch and the harness knows which is which; do not "fix" a gemma4 flat row.
- **An empty `response` on the think+format check may be the `num_predict` trap,
  not a vision failure.** If `eval_count` equals `num_predict` and `thinking` is
  non-empty, the whole allowance was spent inside an unclosed thinking block.
  The harness says so explicitly. Raise `num_predict` in the expectations file;
  do not conclude the build is broken. (Measured on the canary: this probe needs
  ~1,250 tokens; the floor is 600 and nemotron is configured at 4,000.)
- **A whole-suite timeout with a healthy server is queue starvation.** A run once
  failed all three tests at exactly 3 × 1800 s while another client was
  saturating the single slot. The harness reports `CONTENTION` (exit 3) rather
  than a false failure — believe it, find the other client, re-run.

If a token count moved but the *shape* is right, that is a behaviour change, not
a broken payload. Re-measure deliberately and update the expectations file with
provenance, per the harness README.

## Aggregating across hosts

There is no CI yet; the matrix spans CUDA, ROCm and Apple Silicon on self-hosted
machines. Every run writes one JSON file with a `meta` block naming host,
platform, profile, patchset and version. Collect them from each host into one
directory and compare summaries — the README has a one-liner for this. Report
per-host rather than merging into a single verdict: the ROCm host is on a
different payload by design, and a merged pass/fail hides that.

## Reference

- Why expectations are code and not knowledge in this file, plus the normative
  rules: `docs/maxusai/adr/0011-preflight-expectations-are-versioned-code.md`
- Harness and the maintenance path: `docs/maxusai/vision-suite/preflight/README.md`
- Expected values: `docs/maxusai/vision-suite/preflight/expectations.toml`
- Reference passing run: canary `maxusai/ollama:4987dd49-dynres` on `:11437`
- ROCm gate: `docs/maxusai/amd-upgrade-gate.md`
- Apple Silicon build: `docs/maxusai/spec/apple-silicon-build.md`
