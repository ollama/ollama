# Gemma 4 visual token budgets — upstream rebase & forward-port notes

> **HISTORICAL.** Records the decision to rebase onto the last Go-runner base rather than
> forward-port. That branch was retired in 2026-07 once the feature shipped a different
> way; this file is its only surviving record. **The "Forward-porting" section at the end
> reached a conclusion that later turned out to be wrong — see the note there.**

Decision record for keeping the `image_min_tokens` / `image_max_tokens` feature alive as
upstream Ollama removed the architecture it was built on. Companion to
[gemma4-vision-token-budgets.md](gemma4-vision-token-budgets.md) (the feature's own design).

## TL;DR

- The feature was implemented against Ollama's **Go-native inference runner**
  (`runner/ollamarunner`) and the Go model path (`model/model.go`,
  `model/models/gemma4/`). Upstream has since **deleted that entire path** and now
  delegates multimodal inference to `llama-server` (llama.cpp / `mtmd`).
- Because of that, the original branch **cannot be rebased onto current `main`** — its
  core commits patch files that no longer exist.
- This branch (`feat/gemma4-visual-token-budgets-last-go-runner`) rebases the 9 feature
  commits onto **`f63eea3d`**, the last upstream commit where the full Go stack is present
  **and wired**, so the feature actually functions. It is verified building, testing, and
  wired end-to-end.
- A forward-port to current `main` is a **re-implementation against `mtmd` (C++)**, not a
  rebase. See [Forward-port](#forward-porting-to-current-main) below.

## How upstream removed the foundation (two stages)

The Go inference path was removed in two separate upstream PRs:

| Upstream commit | Date | PR | What it removed |
| --- | --- | --- | --- |
| `9db4bdba` | 2026-05-29 | #16031 — "runner: Remove CGO engines, use llama-server exclusively for GGML models" | `runner/ollamarunner/` (the Go runner that drives the feature) |
| `7b22ac96` | 2026-07-02 | #17007 — "llama: clean up dead code from llama-server work" | `model/model.go`, `model/models/gemma4/*`, `kvcache/`, `ml/nn/` (now-dead Go model code) |

After #16031, `model/model.go` and `model/models/gemma4/` still existed but were **dead
code** (no runner drove them); #17007 then deleted them. So "the last version that
supported `model.go`" is ambiguous, which drove the base choice below.

## Base selection

Two candidate bases were evaluated:

| | **A — literal last-with-`model.go`** | **B — last fully-wired stack (chosen)** |
| --- | --- | --- |
| Commit | `a2b3a5e9` (= `7b22ac96^`), 2026-07-02 | `f63eea3d` (= `9db4bdba^`), 2026-05-24 |
| Behind `main` at time of writing | ~20 commits | ~125 commits |
| `model/model.go`, `model/models/gemma4/` | present | present |
| `runner/ollamarunner/` | **already deleted** | **present & wired** |
| Feature applies cleanly? | No — `feat(ollamarunner)` commit conflicts (target file gone) | Yes — all 9 commits apply |
| Feature actually runs? | **No** — Go model path is dead code here | **Yes** — the Go runner drives the Go gemma4 model |

**Chosen: B (`f63eea3d`).** A is more current but yields a branch where the feature is
present-but-non-functional (the engine that consumes the budgets was removed 5 weeks
earlier). B is the last point where the feature is coherent and runnable.

## This branch

- **`feat/gemma4-visual-token-budgets-last-go-runner`**, tip `580ca88e`.
- Base `f63eea3d` + the 9 feature commits, replayed with **zero conflicts**;
  `git range-diff` confirms every patch is byte-identical to the original
  `feat/gemma4-visual-token-budgets` (`e90d7d95`).
- Original branches are untouched.

Reproduce:

```sh
git fetch upstream            # upstream = https://github.com/ollama/ollama.git
git checkout -B feat/gemma4-visual-token-budgets-last-go-runner origin/feat/gemma4-visual-token-budgets
git rebase --onto f63eea3d 9ba5a049
```

## Verification

Adversarial verification (four independent checks) on `580ca88e`:

- **Wiring — PASS.** Feature is live for GGUF gemma4: options flow
  `api.Options` → server → `runner/ollamarunner` → `model.MultimodalBudgetEncoder.
  EncodeMultimodalWithBudgets` → `ProcessImageWithBudgets`. `ollamarunner` is the runner
  actually selected for gemma4 (`fs/ggml/ggml.go` `OllamaEngineRequired("gemma4") == true`),
  so budgets take effect. `server/sched.go` reloads the runner when the budgets change.
- **Upstream drift — PASS.** In the 36 commits between the old base (`9ba5a049`) and
  `f63eea3d`, only `api/types.go` and `x/mlxrunner/server.go` were touched upstream; both
  integrate cleanly (upstream's `Options`/`Seed` changes are preserved, no collisions). The
  five core files are byte-identical between the two bases.
- **Build & test — PASS.** `go build ./...` (whole module) succeeds; `go vet` clean on all
  affected packages; feature unit tests pass; 30 scheduler tests pass
  (`go test ./server/ -run 'Sched|Image|Token|Reload'`).
- **Feature logic — PASS with one CONCERN** (see below).

### Known risk (pre-existing, not introduced by the rebase)

> **Moot for the shipped path, but the underlying question is still open.** The unclamped
> lookup described below is in `model/models/gemma4/model_vision.go`, which no longer
> exists — the shipped path resizes and tokenizes inside llama.cpp `mtmd`/`clip` instead.
> The deployed 1120-budget config has been exercised on a real gemma4:31b vision request
> without incident, but that is **one image**, not the wide/extreme-aspect-ratio case this
> warns about. The suggested smoke test is still worth doing against the shipped path.

The feature raises the default max visual-token budget to **560** (worst case **1120**),
2–4× the base gemma4's documented reference max of **280**. The vision
position-embedding lookup (`model/models/gemma4/model_vision.go:323`) indexes a fixed-size
table **without clamping**. If a model's GGUF position-embedding table was exported for the
~280-token resolution, a large or extreme-aspect-ratio image at a 560/1120-token budget
could index out of bounds (garbage output or a crash). This risk is identical in the
original branch and this rebase — it is not a rebase regression.

**Action:** smoke-test a real Gemma 4 vision model with `image_max_tokens: 1120` on a wide
image before relying on high budgets. Consider clamping table indices as a follow-up.

## Forward-porting to current `main`

> **⚠️ SUPERSEDED BY EVIDENCE — this section's conclusion was wrong.**
>
> The analysis below concluded that a forward-port would require patching C++ `mtmd`/`clip`,
> on the grounds that Gemma ignores the `--image-min/max-tokens` levers. **It does not.**
> PR #2 (`d06138a9`) simply passes those flags through to `llama-server` for the `gemma4`
> arch — Go-only, no C++ patch — and it measurably works: `prompt_eval_count` rises from
> 1,435 (~220 image tokens) to 2,233 (~1,015), matching the reference server exactly.
> See `docs/maxusai/gemma4-budget-image.md`.
>
> Its closing instinct was the right one, though, and is what resolved this: *"first confirm
> what the pinned `mtmd` actually does for Gemma today."* The rest of the section is left
> unedited as a record of the reasoning.

Current `main` no longer has any Go-side hook to influence Gemma image tokens:

- Ollama ships **raw image bytes** to `llama-server`; all resize + tokenization happens
  inside llama.cpp `mtmd`/`clip` (C++).
- For Gemma, `mtmd` produces a **fixed** token count and ignores the min/max-token levers;
  those levers (`--image-min/max-tokens`) are per-process and only honored by dynamic-
  resolution projectors (Qwen-VL, etc.), not Gemma.
- Upstream's `vision.min_image_tokens` / `vision.max_image_tokens` GGUF KV is **write-only
  and inert** in Ollama today (only `convert_lfm2_vl.go` sets it, nothing reads it), and
  gemma4 does not emit it.
- llama.cpp is **fetched at build** (pinned by `LLAMA_CPP_VERSION`, see
  `cmake/local.cmake`); it is not vendored in-tree, and `llama/compat/` contains no
  vision/clip/mtmd patches.

So a faithful forward-port requires changing the C++ `mtmd`/`clip` layer (either as a
maintained `llama/compat/` patch or upstreamed to llama.cpp), plus per-request plumbing on
both sides of the `/completion` boundary — a re-implementation, not a rebase. If the goal is
simply "let big images use more tokens," first confirm what the pinned `mtmd` actually does
for Gemma today; it may already vary tokens by resolution, which would change the calculus.
