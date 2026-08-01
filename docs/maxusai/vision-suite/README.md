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
  `NUM_PREDICT` (default 2200; use ≥4000 with `THINK=on`). Writes
  `resp_<tag>_<test>.json` + `scores_<tag>.json` beside the script.
- `measure.py <host> [model]` — the token-budget protocol: `prompt_eval_count` with
  `num_predict:1` minus the text-only baseline, over 8 geometries + the
  `image_max_tokens` knob check. Flat 256 on nemotron = unpatched payload.
- `run_grid.sh` — model × think-mode grid against one host, with an optional restart
  hook between runs (see below).
- `variants.py <host> <nogrammar|thinkon> [model]` — scene-test probes that isolate
  the `format:"json"` grammar constraint and reasoning mode as variables.

## Method (match this or numbers aren't comparable)

- `temperature 0`, `format:"json"`, `num_ctx 16384`, `NUM_PREDICT=4000` for grids.
- **Cold server per model run** when payloads under test have cross-request leakage
  (upstream #17475 reproduced on b10091): restart the serving container/process
  between runs — `run_grid.sh` does this via `RESTART_CMD`.
- **Always run both think modes.** Known result: `think:true` + `format:"json"` yields
  an *empty* `response` for nemotron3 and qwen3.6 on every payload tested (thinking
  ends without a JSON body, well under the token budget); gemma4 handles both. Report
  empty cells as data.
- Subtract each model's text-only baseline when reading `prompt_eval_count`
  (nemotron3: 18); counts are grid-quantised — ignore ±2.
- Bbox caveat: all tested models emit their trained coordinate conventions rather than
  requested absolute pixels, so `bbox_hits` is a weak signal pending a
  normalized-coordinate prompt iteration.

## Example: full grid against an isolated test server

```bash
python3 gen_scenes.py
RESTART_CMD="docker restart my-test-container" \
  MODELS="nemotron3:33b-q4_K_M gemma4:31b-it-q4_K_M qwen3.6:35b-a3b-q4_k_m" \
  ./run_grid.sh http://127.0.0.1:11435 mytag
```

The isolated-container recipe (own port, model store mounted read-only, GPU
passthrough) is in [nemotron-test-image.md](../nemotron-test-image.md).
