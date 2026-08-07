# Vision benchmark suite

Reproducible ground-truth benchmarks behind the measured tables in
[nemotron-test-image.md](../nemotron-test-image.md) and the amendments in
[vision-token-budget-measurements.md](../vision-token-budget-measurements.md).

## Files

- `gen_scenes.py` — deterministically renders the three test images into `visimgs/`
  plus `ground_truth.json`: a 1920×1080 labeled-shapes scene (20px labels, 14px corner
  serial), a 1568×1568 fake invoice (22px line items, 17px fine print), a 1280×960 bar
  chart (19px values). Needs Pillow + DejaVu fonts
  (`/usr/share/fonts/truetype/dejavu/`). Regenerate any time; edit sizes/content to
  extend coverage — scoring reads `ground_truth.json`, not hardcoded values.
- `vision_suite.py <host> <tag> [model] [test]` — runs three long-prompt JSON
  extractions (single scene w/ pixel bboxes, single invoice, 3-image cross-analysis)
  and scores objectively (label recall, color accuracy, qty/price exactness, bbox
  center-hits, cross-image answers). Env: `THINK=on|false` (default `false`),
  `NUM_PREDICT` (default 2200; ≥4000 with `THINK=on`, 16000 for think-on multi-image),
  `IMAGE_MIN_TOKENS` / `IMAGE_MAX_TOKENS` (fork-only per-request vision budget,
  arch-gated to gemma4 and nemotron_h_omni; unset = build default. Recorded in the
  scores as `req_image_*_tokens` so a control run is identifiable after the fact),
  `ENDPOINT=generate|chat` (default `generate` — `/api/chat` is what OpenWebUI and
  ChatOllama use, and it has carried the upstream think+format two-pass fix since
  v0.12.4, so think-on cells differ by endpoint on builds without the generate-side
  fix). Writes `resp_<tag>_<test>.json` + `scores_<tag>.json` beside the script.
- `gen_geoms.py [outdir]` — renders the eight geometries `measure.py` reads into `testimgs/`
  (deterministic noise + gridlines + corner markers, so the payload size is realistic and
  letterboxing is visible). Run it before `measure.py`; needs Pillow.
- `measure.py <host> [model]` — the token-budget protocol: `prompt_eval_count` with
  `num_predict:1` minus the text-only baseline, over 8 geometries + the
  `image_max_tokens` knob check. Flat 256 on nemotron = unpatched payload. It *reports*
  rather than asserts — for a pass/fail gate on the same formulas with no GPU and no server,
  use `go test ./llm/ -run TestImageTokensForSize`. Note the `image_max_tokens` probe is a
  Runner option, so it forces a full model reload.
- `extbench.py <host> <tag> [model] [benchmark]` — slices of four external benchmarks
  (`ocrbench`, `countbenchqa`, `chartqa`, `refcoco`) pulled from the HF datasets-server REST
  API (stdlib only, no `datasets`, no HF token) and scored locally: contains-match, integer
  match, relaxed accuracy, and dialect-aware bbox IoU respectively. Env: `LIMIT` (50),
  `OFFSET`, `SLEEP` (yield the GPU between requests), plus the same `THINK` / `ENDPOINT` /
  `NUM_PREDICT` / `NUM_CTX` knobs as `vision_suite.py`. Writes `ext_<tag>_<bench>.json`. The
  `refcoco` mode reports the winning coordinate dialect and JSON key per item, so it doubles
  as a dialect probe. See [../vision-benchmark-survey.md](../vision-benchmark-survey.md) for
  why the external harnesses' own grounding scorers cannot be trusted with our models.
- `run_grid.sh` — model × think-mode grid against one host, with an optional restart
  hook between runs (see below).
- `run_compare.sh <tag-prefix>` — **stock vs fork, with a budget-matched control arm.**
  Use this rather than eyeballing two separate runs: a bare stock-vs-fork comparison
  moves two variables at once. See "Comparing against stock" below.
- `variants.py <host> <nogrammar|thinkon> [model]` — scene-test probes that isolate
  the `format:"json"` grammar constraint and reasoning mode as variables.

## Method (match this or numbers aren't comparable)

- `temperature 0`, `format:"json"`, `num_ctx 16384`, `NUM_PREDICT=4000` for grids.
- **Cold server per model run** when payloads under test have cross-request leakage
  (upstream #17475 reproduced on b10091): restart the serving container/process
  between runs — `run_grid.sh` does this via `RESTART_CMD`.
- **Always run both think modes.** `think:true` + `format:"json"` yields an *empty*
  `response` for nemotron3 and qwen3.6 **on stock builds** (thinking ends without a
  JSON body, well under the token budget); gemma4 handles both. Report empty cells as
  data.

  > **Updated 2026-08-07 — this is FIXED on the fork; do not expect empty cells from a
  > fork build.** Measured on `nemotron3:33b-q4_K_M`, all three tests, both a native
  > Metal build and the CPU container:
  >
  > | build | `json_valid` | `eval_count` |
  > |---|---|---|
  > | stock 0.32.6 | **False** ×3 | 562 / 485 / 833 |
  > | fork (Metal) | **True** ×3 | 5233 / 10110 / 7668 |
  > | fork (CPU container) | **True** ×3 | 5134 / 7370 / 4889 |
  >
  > Stock still generates tokens — it thinks and then emits no JSON. The fork thinks
  > and then emits valid JSON. See
  > [generate-think-format-empty-response.md](../generate-think-format-empty-response.md),
  > [ADR 0002](../adr/0002-deferred-format-constraining.md) and
  > [ADR 0004](../adr/0004-routes-layer-think-format-double-request.md).
  >
  > **Budget accordingly.** A fork think-on cell does real work where stock returns
  > almost immediately, so it is far slower — not a hang. Same run: stock 21 s for all
  > three tests, fork on Metal ~7 min, fork on the CPU container ~39 min. Raise
  > `HTTP_TIMEOUT` for CPU think-on runs.
- Subtract each model's text-only baseline when reading `prompt_eval_count`
  (nemotron3: 18); counts are grid-quantised — ignore ±2.
- Bbox scoring is dual-space: models emit their trained coordinate conventions
  regardless of prompt instructions — qwen3.6 answers in 0-1000 normalized (IoU ~0.95
  once decoded; near-perfect grounding), nemotron3 with reasoning answers in pixels
  (center-accurate, IoU ~0.3). The scorer tries both spaces, keeps the better, and
  reports `bbox_space` + `bbox_mean_iou` alongside center-hits. Accepted schema-key
  dialects: `bbox`, `bbox_2d` (qwen, and nemotron's self-chosen key), `box_2d`
  (gemma4/Gemini — note its [y1,x1,y2,x2] order, searched automatically), plus
  `name_bbox`/`name_bbox_2d` on the invoice. Measured dialect map (2026-08-02):
  qwen3.6 = bbox_2d, xyxy, norm-1000 (IoU ~0.95); gemma4 = box_2d, yxyx, norm-1000
  (IoU ~0.78); nemotron3 = bbox_2d, xyxy, norm-1000 of its input canvas — on the
  unpatched 512-letterbox payload the y-axis carries the padding offset, on dynres
  payloads the canvas is the image itself; under prompted reasoning it can emit
  coarse pixel-space boxes instead. Key choice alone did not change quality; the
  space/order decode did.

## Example: full grid against an isolated test server

```bash
python3 gen_scenes.py
RESTART_CMD="docker restart my-test-container" \
  MODELS="nemotron3:33b-q4_K_M gemma4:31b-it-q4_K_M qwen3.6:35b-a3b-q4_k_m" \
  ./run_grid.sh http://127.0.0.1:11435 mytag
```

The isolated-container recipe (own port, model store mounted read-only, GPU
passthrough) is in [nemotron-test-image.md](../nemotron-test-image.md).

## Comparing against stock (use the control arm)

A bare stock-vs-fork comparison moves **two** variables at once:

1. **our vision token budget** — `visionServerArgs` adds `gemma4` and
   `nemotron_h_omni` branches that upstream does not have *at all* (checked
   2026-08-07: both `v0.32.5:llm/llama_server.go:994` and `v0.32.6:…:999` contain
   only `qwenVLServerArgs`, handling qwen arches); and
2. **the llama.cpp payload** — `LLAMA_CPP_VERSION` differs whenever the fork is not
   synced to the release the stock server runs. Measured 2026-08-07: fork `b10091`
   (v0.32.5) vs stock `b10242` (v0.32.6), 151 builds apart.

So "the fork detects the fine text that stock misses" and "the fork's bbox IoU is
worse than stock's" are individually uninterpretable — either could be ours or
upstream's drift.

`run_compare.sh` adds a third arm that pins the fork's budget to upstream's
effective defaults. A delta that **disappears** under the control was ours; a delta
that **survives** is the payload.

```bash
STOCK=http://127.0.0.1:11434 FORK=http://127.0.0.1:11435 \
  MODEL=gemma4:12b-it-q4_K_M CONTROL_MIN=40 CONTROL_MAX=280 \
  ./run_compare.sh mytag
```

Control values are **per-arch**, and wrong ones silently invalidate the arm:

| arch | control min | control max | why |
|---|---|---|---|
| `gemma4` | 40 | 280 | llama.cpp `set_limit_image_tokens(40, 280)` |
| `nemotron_h_omni` | 256 | 256 | unpatched payload is a structural flat 256; 002 makes it (256, 3328), so pinning both bounds reproduces stock |

The knobs are arch-gated, so on any other arch the control arm is a no-op that
duplicates the fork arm. That is a valid result — but do not read it as "no budget
effect" on an arch that was never wired into `visionServerArgs`.

**Backend caveat.** A CPU arm can differ from a Metal arm on identical inputs with
identical `prompt_eval_count` — greedy sampling diverges on backend floating point.
Always check `prompt_eval_count` before attributing such a delta to a patch.

## Runs archive and harness knobs (2026-08-02)

- `runs/` holds raw campaign logs plus `*.parsed.json` (one object per scored
  cell) for the 2026-08-02 campaign, max-context arm, true-stock baseline,
  and runaway bisect; as-run parsed files keep the scorer outputs of their
  time (Q4 is pre-correction in blocks scored before the dialect fix);
  `final-matrix-2026-08-02.json` is the merged, Q4-corrected dataset behind
  the published matrix.
- `ONLY_TESTS=scene_single[,document_single,...]` runs a subset of the suite —
  used by the bisect harness. `HTTP_TIMEOUT` (seconds, default 1800) bounds a
  single request — raise it for uncapped think-mode probes. `KV_CACHE_TYPE`
  passes a per-request `options.kv_cache_type` (fork feature, ADR 0005) —
  single type or K/V pair like `q8_0/f16`.
- Multi-image Q4 is scored dialect-aware like scene boxes (`q4_bbox_space`
  reports the matched space); models answer norm-1000 regardless of prompt.
- Caveat: with `OLLAMA_KV_CACHE_TYPE=q8_0`, qwen3.6 think-on inflates
  prompt-dependently: document unaffected, scene ~19K thinking tokens (vs
  3.3K at f16), multi no convergence within 131K (vs 9.0K at f16). Use f16
  KV for qwen reasoning runs; no practical num_predict rescues multi on
  q8_0. See vision-campaign-2026-08-02.md §6.

## Fine-text probe and coordinate-dialect guidance (2026-08-02)

- `finetext_probe.py` generates a 1568² dense-text page (20 reference codes at
  22/16/12/9/7 px, seeded) and scores exact-match recall per size tier — the
  test that separates real transcription from confabulation. `gen` regenerates
  the image; run form matches vision_suite env knobs.
- Prompt bounding boxes in **norm-1000**, not pixels: all three models answer
  norm-1000 natively; nemotron additionally OBEYS a pixel instruction when
  thinking and loses geometry doing so (IoU .39 pixel-prompt vs .81
  norm-1000-prompt, think on). The scorer's `bbox_space` field verifies what
  came back.
