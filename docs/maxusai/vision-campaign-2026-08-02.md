# Vision campaign 2026-08-02 — upstream 0.32.1 vs canonical b9888+patches

Full-factorial ground-truth campaign on the gfx1151 host, plus a max-context arm,
a true-stock baseline correction, and the qwen think-runaway bisect that ended in
a root cause. Raw logs and parsed JSON: [vision-suite/runs/](vision-suite/runs/).
Method: [vision-suite/README.md](vision-suite/README.md) — temp 0, cold container
restarts per block, dialect-aware bbox scoring, both think modes everywhere.

## Axes

| Axis | Values |
|---|---|
| Build | **upstream** `ollama/ollama:0.32.1-rocm` (genuine baseline) vs **canonical** `maxusai-ollama:0.32.1-rocm-dynres-a4788474` (b9888 + dynres 002 + gemma4 budget + qwen floor + generate think+format fix) |
| Model | `nemotron3:33b-q4_K_M`, `gemma4:31b-it-q4_K_M`, `qwen3.6:35b-a3b-q4_k_m` |
| Thinking | off / on |
| Endpoint | `/api/generate` / `/api/chat` |
| num_ctx | 32,768 (main) · per-model max 131,072 / 262,144 (ctx arm) |

144 scored cells (72 per build) + 36 ctx-arm cells + 10 bisect/control runs.
Containers: canonical :11435, upstream :11439 (`ollama-truestock`), throwaway
:11440/:11441 for bisect arms. Prod :11434 untouched. Campaign env matched prod:
`OLLAMA_FLASH_ATTENTION=1`, `OLLAMA_KV_CACHE_TYPE=q8_0` — the latter turned out
to be a finding, not a constant (see §6).

An earlier draft of this campaign used `maxusai-ollama:0.32.1-rocm-gemma4budget`
as the baseline; per review, the baseline must be genuine upstream. The re-run
proved the nemotron and qwen columns of that draft were already token-identical
to upstream (the gemma4budget image only touches gemma), so only gemma's
baseline numbers changed. Both datasets are in `runs/`.

## 1. Headline: dynres transforms nemotron

Scene photo 1920×1080, think off (identical on both endpoints):

| | upstream (256 visual tokens) | canonical (2,042 visual tokens) |
|---|---|---|
| Objects found | 0 / 6 | **6 / 6** |
| Boxes hit · mean IoU | 0 · 0.00 | **6 · 0.84** |
| Colors right | 0 / 6 | **6 / 6** |
| Serial `SN-4921-XK` | missed | **read** |
| Invoice items (doc test) | 3 / 5, prices 0/5 | **5 / 5, prices 5/5** |
| px per visual token | 8,100 | 1,015 |

Multi-image keeps full per-image budgets on canonical (2,042 + 2,403 + ~1,200
visual + ~556 text tokens = measured prompt 6,203; upstream's 1,324 is the same
text + 3×256 visual) and answers every cross-reference question.

**Think-mode grounding addendum (dialect probe, `runs/dialect-2026-08-02.log`):**
the think-on box degradation (6/6 · IoU 0.84 → 4/6 · 0.39) is NOT a reasoning
deficit — it is prompt-dialect obedience. Only reasoning obeys the prompt's
pixel-coordinate instruction; nemotron's pixel-space geometry is weak. Asking
for its native **norm-1000** coordinates instead restores think-on to
**6/6 · IoU 0.812** (think-off control 0.864) and cuts thinking from 11,749 to
4,083 tokens — the chain no longer does coordinate arithmetic. Suite guidance:
prompt bounding boxes in norm-1000 for all three models (each answers in it
natively regardless; the scorer verifies via `bbox_space`). The one canonical "regression" ever observed (multi
Q4 bbox) was a scorer artifact — see §5.

## 2. The generate think+format misfiling is an upstream bug, now proven on upstream

On genuine `ollama/ollama:0.32.1`, every model that emits think markers returns
an **empty response** on `/api/generate` with `think:true` + `format:"json"`
(nemotron 3/3 cells, qwen 3/3 cells; gemma is unaffected because it emits no
markers on generate). Token counts show normal generation (486–950 evals) —
the answer is generated but misfiled into `thinking` because the JSON grammar
constrains output from token 0. `/api/chat` recovered long ago upstream via the
double-request fix; `/api/generate` never did. The canonical build's fix
rescues all nemotron cells (3/3 valid) and qwen's document cell; details in
[generate-think-format-empty-response.md](generate-think-format-empty-response.md).

## 3. Chat hides its thinking cost in prompt_eval

With think+format on `/api/chat`, upstream's double-request re-evaluates the
thinking as prompt in pass 2. The eval_count looks cheap; prompt_eval balloons:

| nemotron scene, think on | prompt_eval | eval_count | true total |
|---|---|---|---|
| generate (canonical fix, single pass reported honestly) | 2,674 | 11,749 | 14,423 |
| chat (upstream double request) | 15,427 | 518 | 15,945 |

Same compute, different bookkeeping — compare cells only on prompt+output
totals. The v2 routes-layer implementation (`82158bd8`+`ae797815`, image
`…-dynres-ae797815`) unifies the mechanics at the routes layer.

## 4. Gemma budget patch: mixed verdict on this suite — re-justify or revert

Genuine upstream gemma runs this suite at roughly half the prompt tokens
(scene 848 vs 1,504) and is **not worse overall**:

| gemma4:31b | upstream 0.32.1 | canonical (budget patch) |
|---|---|---|
| Scene objects/boxes | 6/6 · IoU **0.90** | 6/6 · IoU 0.78 |
| Document (think off) | 5/5 items, total ✓, 3 name boxes | 5/5 items, total ✓, 4 name boxes |
| Document (think on, generate) | 4/5 items, 1 name box | **5/5 items, 4 name boxes** |
| Multi-image | all correct | all correct |

(Multi-image Q4 verdicts here use the corrected scoring of §5; the as-run
`campaign-*.parsed.json` predates the scorer fix — `final-matrix-2026-08-02.json`
is the authoritative dataset for Q4.)

Upstream wins box sharpness on these clean synthetic images; the patch wins
fine-text under thinking. **Verdict settled same day by the dense fine-text
A/B** (`vision-suite/finetext_probe.py`, 20 codes at 22/16/12/9/7 px on a
1568² page, `runs/finetext-2026-08-02.log`):

| exact-match recall | 22px | 16px | 12px | 9px | 7px |
|---|---|---|---|---|---|
| upstream (280 visual tokens) | 4/4 | 2/4 | 0/4 | 0/4 | 0/4 |
| canonical (1,120 visual tokens) | 4/4 | 4/4 | 4/4 | 4/4 | 3/4 |

Identical across endpoints and think modes. Upstream is blind below 16px and
**confabulates** — it still returns 18 plausible-looking codes, i.e. silent
misreads, and think-on burns ~4,400 hidden tokens without gaining a single
small-tier code. **The budget patch is re-justified**: ~0.12 IoU on synthetic
shape boxes buys correct dense-text transcription down to 7–9px without
confabulation. It stays in the release lineage (ADR 0003 policy).

## 5. Q4 scorer correction (affects earlier reported matrices)

The multi-image Q4 check (bbox of the "DYNAMO" shape) compared answers against
pixel-space ground truth only, while models answer in norm-1000 regardless of
prompt instructions. All three models on canonical answered ~[115, 552, 252,
798] ≈ ground truth [220,600,480,860]px normalized — correct, scored as miss.
Upstream nemotron's apparent "hit" was a garbage box whose center landed inside
the target. Fixed in `vision_suite.py` (same dialect search as scene boxes,
reported as `q4_bbox_space`); 16 historical cells re-scored. After correction,
**every valid multi-image response in the main + true-stock datasets
(`final-matrix-2026-08-02.json`) hits Q4**, with two exceptions: upstream
nemotron think-on/chat answers `null` (label illegible at 256 tokens), and in
the ctx arm nemotron think-on at 131,072/generate produced one genuinely wrong
box under both dialects — think-mode boxes remain nemotron's noisiest metric
(cf. IoU 0.39 in §1). As-run parsed files for blocks scored before the fix
landed (~13:05) keep pre-correction Q4 values; `final-matrix-2026-08-02.json`
carries the corrected scores.

## 6. Qwen think-runaway: root cause is q8_0 KV cache quantization

Qwen3.6 with think on runs its reasoning to the token limit on the two
box-heavy prompts (scene, multi) while always terminating on the extraction
prompt (document). This reproduced deterministically across builds, endpoints,
images, and num_ctx up to 262,144 — and was NOT a code regression:

| Arm (scene cell, think on, generate) | KV cache | Result |
|---|---|---|
| genfix `d1ef5557` · ctx 16,384 | q8_0 | runaway to ctx-full (13,771 evals) |
| genfix2 `925a669a` · ctx 16,384 | q8_0 | runaway, identical 13,771 |
| routes `ae797815` · ctx 16,384 | q8_0 | runaway, identical 13,771 |
| canonical image · ctx 16,384 | q8_0 | runaway, identical 13,771 |
| all four · ctx 32,768 | q8_0 | runaway to num_predict (16,000) |
| genfix `d1ef5557` · ctx 16,384 | **f16** | **valid, 3,320 evals, 6/6, IoU 0.927** |
| canonical image · ctx 32,768 | **f16** | **valid, 3,320 evals (same count), IoU 0.927** |

The exact minimal pair is the canonical image at 32,768: identical prompt
(2,613 tokens) with only `OLLAMA_KV_CACHE_TYPE` flipped — q8_0 runs away to
16,000, f16 terminates at 3,320 with valid output. The native `d1ef5557` f16
run confirms the effect on a second build with the same eval_count and IoU
(its reported prompt_eval of 5,325 is that binary's known double-counted
pass-2 prefill; the q8_0 arm shows 2,613 precisely because pass 2 never ran).
**Uncapped follow-up (same day, `runs/uncapped-2026-08-02.log`):** removing
the arbitrary 16,000 cap (probes at num_predict 131,072, ctx 262,144)
resolves the "runaway" into a **prompt-severity spectrum of q8_0 reasoning
inflation**, not a uniform loop:

| prompt (think on, temp 0) | f16 thinking | q8_0 thinking | inflation |
|---|---|---|---|
| document (extraction) | 4,384 | 3,139 | none (q8_0 slightly shorter) |
| scene (6-object grounding) | 3,320 | **19,160, valid JSON** | ~5.8× |
| multi (3-image grounding) | 9,050 | **no convergence ≤131,072** | ≥14.5×, unbounded in practice |

The scene cells in the earlier matrices were censoring artifacts of the
16,000 cap under a finite ~19K convergence (the 16,384-ctx arms additionally
window-bound); the multi cells are genuine non-termination within any
practical budget. Multi's convergence sits near the 27,352 backfill cap —
upstream chat squeaked under it (thinking done, 1,228-token answer) while the
code-identical canonical run did not, so cold 25K+ chains straddle it.

With budgets above each prompt's ceiling, **both builds behave identically on
chat (3/3 valid) and the only remaining build difference is upstream's
generate misfiling (0/3 at every budget and both KV types — 606/492/951
evals q8_0, 606/487/1,157 f16)** — i.e. with the environment held right, the
measured patch effect on qwen is exactly the generate think+format fix, and
the f16 full matrix is 6/6 valid on canonical vs 3/6 on upstream.

### Prior art (searched 2026-08-02; no exact prior report found)

No published report combines KV-cache-only q8_0, thinking-length inflation
(rather than accuracy loss or premature stop), a Qwen3.5/3.6-class model, and
task-dependence (grounding affected, extraction not). Nearest neighbors:

- **Mechanism, academic:** [arXiv 2606.00206](https://arxiv.org/abs/2606.00206)
  — quantization noise at high-entropy positions over-samples hesitation
  tokens and inflates CoT up to 4.5× on R1-distills/QwQ (weight/end-to-end
  PTQ, KV never isolated); [arXiv 2606.25519](https://arxiv.org/abs/2606.25519)
  — "CoT token inflation" up to ~2.9× at INT3 incl. Qwen3-30B-A3B-Thinking
  (weight-only). Our result is the KV-cache-only instance of this phenomenon.
- **Same arch/knob/backend, inverse symptom:**
  [ollama#17347](https://github.com/ollama/ollama/issues/17347) — quantized KV
  on qwen35/qwen35moe under ROCm flips the stop decision the other way
  (premature EOS mid-turn). Natural cross-reference if we file upstream.
- **Counter-evidence that sharpens novelty:**
  [arXiv 2504.04823](https://arxiv.org/abs/2504.04823) finds KV8/KV4
  near-lossless with *no* length inflation on reasoning benchmarks (text-only,
  no vision grounding, moderate lengths);
  [llama.cpp#21385](https://github.com/ggml-org/llama.cpp/issues/21385)
  reports q4_0 KV "lossless" on Qwen3.5-9B — noting only 8/32 layers carry KV
  in the hybrid arch — on multi-turn chat, not 19K-token reasoning chains.
  A single-step KL benchmark ([r/LocalLLaMA](https://www.reddit.com/r/LocalLLaMA/comments/1suh3sz/gemma_4_and_qwen_36_with_q8_0_and_q4_0_kv_cache/))
  measured qwen3.6 KL &lt; 0.04 at q8_0 — consistent with our view that the
  effect is accumulation over a long chain, invisible single-step.
- **Confound ruled out:** [Qwen3#1700](https://github.com/QwenLM/Qwen3/issues/1700)
  shows JSON-constrained non-termination *without* KV quantization exists as a
  separate failure class — our f16 control (same grammar, terminates at 3,320)
  excludes it here.
- **Related per-phase sensitivity:**
  [llama.cpp#21679](https://github.com/ggml-org/llama.cpp/issues/21679)
  measured think-phase KV entries as more quantization-sensitive than
  answer-phase on R1-distill — an independent hint toward per-phase or
  per-model KV precision, which our `kv_cache_type` option (ADR 0005) now
  makes operable per model.

**Operational recommendation (gfx1151 prod runs q8_0):** serve qwen3.6
vision+reasoning workloads with `OLLAMA_KV_CACHE_TYPE=f16` (or per-model
override once available). Nemotron and gemma showed no q8_0 sensitivity in 96
cells. Memory cost: qwen KV at f16 doubles vs q8_0 (≈25 GB → ≈50 GB at 262K;
proportionally less at 32K).

## 7. Max-context arm: quality-free, pay in load time and VRAM

At per-model max ctx (nemotron 131,072; gemma/qwen 262,144), 12 of 18
think-false cells are token-identical to 32,768, and the six that differ are
equal or better: qwen scene keeps 6/6 hits with IoU rising 0.953 → 0.973/0.971,
qwen multi gives the same all-correct answers at +223 output tokens, and the
nemo/gemma chat multi cells drift by ≤7 tokens with identical scores. No cell
regresses. The mamba-hybrid dividend is real:

| Model · ctx | VRAM loaded | first pass (load + 3 tests) |
|---|---|---|
| nemotron3 · 131,072 | 28.8 GB | 89 s |
| gemma4 · 262,144 | 30.3 GB | 443 s |
| qwen3.6 · 262,144 | ~25 GB (delta; prod co-resident) | 142 s |

## 8. False-cell ledger — fully resolved

Every invalid cell in 144 has exactly one of two explanations:
1. **Upstream generate misfiling** (§2): 6 upstream cells (nemo 3, qwen 3),
   fixed on canonical (4 of 6 rescued; the other 2 fall into class 2).
2. **q8_0-KV-degraded qwen reasoning exceeding the caps** (§6): scene+multi
   with think on, on both builds and endpoints (upstream chat included — not
   caused by any fork code). Uncapped, scene converges at 19,160 evals (cap
   censoring); multi does not converge within 131,072 (practically unbounded
   under q8_0, vs 9,050 at f16).

Zero unexplained failures. Determinism throughout: temp-0 repeats were
token-identical across builds sharing a code path (e.g. qwen chat document:
542 evals in four independent configurations).
