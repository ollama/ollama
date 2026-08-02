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
  `ENDPOINT=generate|chat` (default `generate` — `/api/chat` is what OpenWebUI and
  ChatOllama use, and it has carried the upstream think+format two-pass fix since
  v0.12.4, so think-on cells differ by endpoint on builds without the generate-side
  fix). Writes `resp_<tag>_<test>.json` + `scores_<tag>.json` beside the script.
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
