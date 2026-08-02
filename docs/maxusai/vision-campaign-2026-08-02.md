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
text + 3×256 visual) and answers every cross-reference question. The one canonical "regression" ever observed (multi
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

Upstream wins box sharpness; the patch wins fine-text under thinking. Nothing
here reproduces a decisive uplift on synthetic clean imagery. Action: either
produce the dense fine-text case that motivated the 280→1120 budget as a suite
scene, or drop the patch from the release lineage.

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
**Uncapped follow-up (same day):** raising num_predict to 131,072 (ctx
262,144) shows q8_0 reasoning is NOT a non-terminating loop — it converges at
**19,160 evals with valid JSON** (canonical scene cell; reproduced eval-exact
in the backfill), vs 3,320 under f16. The correct statement is a **~5.8×
thinking-cost degradation under q8_0**, and every "runaway" cell in the
matrices above is a censoring artifact of the arbitrary 16,000 cap sitting
below a ~19K natural convergence (the 16,384-ctx arms were additionally
window-bound). Uncapped matrix for both builds/endpoints plus the f16
comparison: `runs/uncapped-2026-08-02.log`.

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
2. **q8_0-KV-degraded qwen reasoning exceeding the 16,000 cap** (§6):
   scene+multi with think on, on both builds and endpoints (upstream chat
   included — not caused by any fork code). Uncapped, these cells converge at
   ~19K evals; the "failures" are cap censoring of a q8_0-inflated but finite
   reasoning phase.

Zero unexplained failures. Determinism throughout: temp-0 repeats were
token-identical across builds sharing a code path (e.g. qwen chat document:
542 evals in four independent configurations).
