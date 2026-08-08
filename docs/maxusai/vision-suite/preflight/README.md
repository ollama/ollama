# Pre-deploy regression harness

Validates a freshly built ollama image before it is deployed, so verification
stops being a sequence of hand-typed prompts repeated per host.

One entry point, one data file:

```bash
./preflight.py --host http://127.0.0.1:11437 --platform cuda \
               --image-tag maxusai/ollama:4987dd49-dynres
```

It resolves a **profile** from `(platform, server version string)`, runs the
checks that apply to that profile, prints an expected-vs-actual diff, writes a
machine-readable result file to `runs/`, and exits non-zero on failure.

**The expectations live in [`expectations.toml`](expectations.toml), in this
repo, on purpose** — the decision and its normative rules are
[ADR 0011](../../adr/0011-preflight-expectations-are-versioned-code.md). They
legitimately change when a compat patch lands — 004 made gemma4 flat at 1,102
tokens, 005 moved nemotron's pinned cost from 3,390 to 3,270 — so they are
versioned alongside the payload they describe. A harness living outside the repo
silently rots against those changes. The
[`ollama-preflight` skill](../../../../.claude/skills/ollama-preflight/SKILL.md)
sits on top and owns the *procedure* (which host, which order, how to read
results); this code owns the *assertions*.

This extends the vision suite rather than replacing it. `measure.py` reports;
this asserts. `vision_suite.py` scores; this applies thresholds to its scores.

> **Getting the subtrahend right takes two corrections, not one.** Both are now in
> `probes.Ollama.image_prefix()`, and SPEC B8 makes them normative.
>
> 1. **One prompt everywhere.** `measure.py` used to baseline with `"Hi"` and probe
>    with `"Describe briefly."`, so the *prompt*-length difference landed in every
>    row (18 vs 21 tokens on `nemotron3:33b-q8`). Fixed upstream of this branch.
> 2. **The text prefix is not the text-only count.** Attaching an image can change
>    how the template renders the surrounding text. Same prompt, same model:
>    text-only 21, but the prefix inside an image request **20**. A matched-prompt
>    text-only baseline therefore reads every nemotron image exactly 1 token *low*.
>    `gemma4:31b` measures 19 both ways, so this is arch-specific and cannot be
>    hardcoded.
>
> This harness calibrates the prefix from a two-image difference —
> `count(A) + count(B) − count(A,B)`, which cancels it without trusting a text-only
> probe or assuming a grid. Until 2026-08-08 it did (2) wrong, which is why the
> recorded ladder read `[265, 265, 577, 2305, 3269]`: uniformly 1 low, and briefly
> written off as grid quantisation. Corrected to `[266, 266, 578, 2306, 3270]`,
> which agrees with the pinned `expect_tokens` and `TestImageTokensForSize`.

## Relationship to the no-GPU gates

Two cheap gates run before any model loads:

```bash
go test ./llm/ -run TestImageTokensForSize   # the sizing formulas
python3 test_verdicts.py                     # the harness's own verdict logic
```

`TestImageTokensForSize` pins the sizing formulas. `test_verdicts.py` pins *this
harness*: that a flat ladder diagnoses an unpatched payload for a `dynamic` arch
and passes for a `flat` one, that the pinned-budget ceiling invariant catches an
overshoot at a value nobody has measured, that the `num_predict` trap is named as
the trap rather than as a vision failure — plus consistency checks on
`expectations.toml` itself (every declared arch has a block, every
`image_max_pixels` equals `max_tokens × S²`, every `scaling` field agrees with the
ladder recorded beside it, every `unmeasured` block carries a reason).

That last set is the guard on the maintenance path: editing a ladder to flat
without updating `scaling` is exactly how the verdict logic would silently
invert, and the test fails on it.

This harness covers what neither can: that the built *image* actually carries the
payload and behaves as measured.

## Checks

| check | what it proves | how it fails |
|---|---|---|
| `version` | the server on this port is the build under test | **gates the run** — aborts, exit 2 |
| `image_tag` | the container serving this port runs the named image | FAIL |
| `go_patch_marker` | `grep -c -- --image-max-tokens /usr/bin/ollama` is 1 | 0 means a stock binary |
| `payload_proof` | `load_hparams: image_max_pixels: N (custom value)` where `N == max_tokens * S²` | FAIL, with the derivation printed |
| `token_ladder` | same image at five 16:9 geometries vs a text-only baseline | FAIL, **per-arch** verdict |
| `pinned_budget` | `image_min_tokens == image_max_tokens` never exceeds the ceiling | FAIL, the 005 defect class |
| `think_format` | `think:true` + `format:"json"` returns a non-empty valid JSON body | FAIL, distinguishes the fork fix from the `num_predict` trap |
| `extraction_quality` | `vision_suite.py` scores clear their floors (`--quality`) | FAIL |
| `endpoint_exclusive` | no other client was competing for the slot | CONTENTION, exit 3 |

### Two things the checks deliberately do *not* do

**The payload proof never inspects the binary.** Static inspection of
`libmtmd.so` is unreliable and must not be reintroduced: `strings` is absent from
the ollama images, so in-container greps return misleading zeros, and
`<img>`/`</img>` literals appear in stock too (InternVL uses them). An RTTI
occurrence-count delta (`fixed_size` 9→8) was suggestive but never proof. The
model-load log is the proof, and the harness forces a fresh model load first so
it cannot read a *previous* build's line.

**The ladder verdict is per-arch and is never shared.** A flat ladder means an
unpatched payload for `nemotron_h_omni`, but is the *correct* result for `gemma4`
under 004, which budget-fills every image to the ceiling. Each arch declares
`scaling = "dynamic" | "flat"` and the diagnosis follows from that. A shared
heuristic gets this backwards.

## Exit codes

| code | meaning |
|---|---|
| 0 | all applicable checks passed |
| 1 | one or more checks failed |
| 2 | harness/config error — unknown `(platform, version)`, unreachable, or version gate rejected |
| 3 | endpoint contention — results not trustworthy, re-run exclusively |
| 4 | an applicable expectation has never been measured (`NEEDS_BASELINE`) |

Exit 4 exists so an unvalidatable combination can never read as a pass. Use
`--allow-unmeasured` to acknowledge the gap deliberately.

## Operational notes

These are encoded in the harness, not left to the operator to remember:

- **`num_predict` is floored at 600** (`min_num_predict` in the data file). At 120,
  three probes returned `"response": ""` with `eval_count` exactly 120 — the whole
  allowance spent inside an unclosed thinking block. That reads as a vision
  failure and is not one, so `thinking` is judged alongside `response` and the
  diagnosis names the trap when it sees it.
- **Probes are grouped by budget.** Changing `image_{min,max}_tokens` is a Runner
  option and forces a full model reload, tens of seconds to minutes. Every
  default-budget probe runs first; the pinned arms run last. Use `--skip-pinned`
  to save two reloads when you only want a smoke check.
- **Contention is detected, not measured through.** A `vision_suite.py` run once
  failed all three tests with "timed out" at exactly 3 × 1800 s while the server
  was perfectly healthy — another client was saturating the single slot. Every
  probe records wall-clock minus the server's own `total_duration`; that gap is
  queue time, and a run with a large one reports `CONTENTION` (exit 3) instead of
  a false failure.
- **Results are written after every check**, so a killed run still leaves usable
  data.
- **Detach long runs.** A backgrounded run has been SIGTERM'd (exit 143) mid-suite:

  ```bash
  setsid nohup ./preflight.py --host http://127.0.0.1:11437 --platform cuda \
      --quality --out runs/rc1.json > runs/rc1.log 2>&1 < /dev/null &
  ```

  Poll `runs/rc1.json`. **Do not use `pgrep -f preflight.py`** to test whether
  your own job is running — the pattern matches the checking shell's own command
  line, so the loop never exits. Two waiters hung for hours on this. Use
  `ps -eo args | grep '[p]reflight.py'` or just watch the result file.
- **Never trust a port.** 11434, 11435 and 11436 are all occupied on 10.8.0.6;
  a canary on 11436 once answered from the wrong server and only a mismatched
  version string caught it. The version gate runs first and aborts the run.

## Aggregating across hosts

Until CI exists, results are collected by hand. Every run writes one JSON file
with a `meta` block (host, platform, profile, patchset, version, container) and
one record per check. To compare hosts:

```bash
python3 - <<'EOF'
import json, glob
for f in sorted(glob.glob("runs/*.json")):
    d = json.load(open(f))
    m, s = d["meta"], d["summary"]
    print(f"{m.get('profile','?'):28} {m.get('version','?'):24} {s}")
EOF
```

## Adding an expectation when a compat patch lands

This is the maintenance path. Follow it and the harness stays true; skip it and
it rots.

**1. Decide whether the patch changes a value or adds a combination.**

- A patch that changes measured behaviour on an existing profile (004, 005) →
  update the values in that profile's `[expect.<profile>.<arch>]` block.
- A patch that produces a genuinely different payload, or a new platform → add a
  new `[profiles.<id>]` and its own `[expect.<id>.<arch>]` blocks. Do not reuse a
  profile whose `patchset` no longer describes the build.

**2. Measure. Do not derive, and do not edit a number to make a run go green.**
Build the image, start it on a free port, and take the ladder by hand:

```bash
python3 - <<'EOF'
import sys; sys.path.insert(0, ".")
from probes import Ollama
c = Ollama("http://127.0.0.1:11437")
print("version:", c.version())
model = "nemotron3:33b-q8"
base, text_only, _ = c.image_prefix(model)   # NOT text_baseline() -- see B8
print(f"in-image prefix: {base}  (text-only: {text_only})")
for size in ["256x144", "512x288", "1024x576", "2048x1152", "3072x1728"]:
    delta, _ = c.visual_tokens(model, size, base)
    print(f"  {size:>10}: {delta}")
EOF
```

For the payload proof, read the load log directly — and force a reload first, or
you may be reading the previous build's line:

```bash
docker logs ollama-dynres-canary 2>&1 | grep -E 'image_(min|max)_pixels'
```

**3. Write the values in, with provenance.** Every arch block carries
`status = "measured"` and `measured_on`. Set `image_{min,max}_pixels` to the
values the log printed and record the derivation inputs (`patch_stride`,
`budget_{min,max}_tokens`) next to them, so the next person can check
`N == max_tokens × S²` without rediscovering that `S = patch_size × n_merge`.

**4. Set `scaling` correctly.** `"dynamic"` if cost should track resolution,
`"flat"` if the payload budget-fills (gemma4 post-004) or is structurally capped
(nemotron pre-002). This single field is what makes the verdict per-arch.

**5. If you cannot measure it, say so.** Set `status = "unmeasured"` with a
`reason`. The harness reports `NEEDS_BASELINE` and exits 4. That is correct
behaviour — an unvalidatable check must never read as a pass. Copying another
profile's numbers across is the exact failure this file exists to prevent, and is
why the ROCm gemma4 and both Apple Silicon rows are currently marked unmeasured
rather than filled in from the CUDA profile.

**6. Cross-reference the ADR or patch** in the block's comment, so the next
reader can find out *why* the number moved.

### Worked example — what 005 looked like

Before 005, nemotron pinned to 3328 delivered 3390: 3,388 grid tokens against its
own 3,328 ceiling, a 60-token overshoot. The change was:

```toml
[expect.cuda-dynres-005.nemotron_h_omni.pinned]
size = "2048x1152"
pin_tokens = 3328
expect_tokens = 3270          # was 3390 pre-005
tolerance = 4
enforce_ceiling_invariant = true
control_expect_tokens = 2306  # unpinned control — unchanged by 005
```

Note `enforce_ceiling_invariant`. The exact value is a regression pin; the
invariant (`delivered − markers ≤ ceiling`) is a *class* assertion that catches a
future overshoot at a number nobody has measured yet. Keep both.
