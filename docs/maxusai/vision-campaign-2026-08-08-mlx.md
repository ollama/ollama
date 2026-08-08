# Vision campaign 2026-08-08: MLX vs GGUF — bbox, document, OCR, multi-image

First engine-parity measurement after gemma4 MLX vision landed
(`claude/admiring-khorana-9b856c`): the full
[vision-suite](vision-suite/README.md) scene/document/multi extractions plus
the fine-text probe, MLX (safetensors nvfp4) against llama-server (GGUF
q4_K_M) on the same host, same fixtures, same method.

## Method

- 8 models, **cold server per model** (fork `main`-lineage build, Metal,
  M-series; MLX store `~/.ollama/models-mlx`), `temperature 0`,
  `ENDPOINT=chat`, `THINK=false`, default vision budgets (gemma4: ADR 0008
  70…1120), suite defaults otherwise.
- Runner: [`vision-suite/run_engine_compare.sh`](vision-suite/run_engine_compare.sh);
  tables rendered by
  [`vision-suite/summarize_engine_compare.py`](vision-suite/summarize_engine_compare.py)
  from the raw scores in
  [`vision-suite/runs/mlx-compare-2026-08-08/`](vision-suite/runs/mlx-compare-2026-08-08/).
  Reproduce:

  ```sh
  cd docs/maxusai/vision-suite
  MODELS="gemma4:12b-nvfp4 gemma4:26b-nvfp4 gemma4:31b-nvfp4 \
    gemma4:12b-it-q4_K_M gemma4:26b-a4b-it-q4_K_M gemma4:31b-it-q4_K_M \
    qwen3.6:35b-a3b-q4_K_M nemotron3:33b-q4_K_M" \
  RESTART_CMD='pkill -f "ollama serve"; sleep 2; (cd ../../.. && \
    OLLAMA_MODELS=$HOME/.ollama/models-mlx OLLAMA_HOST=127.0.0.1:11499 \
    ./ollama serve >> /tmp/serve.log 2>&1 &)' \
    ./run_engine_compare.sh http://127.0.0.1:11499
  python3 summarize_engine_compare.py <the same model list>
  ```

- All 8 models produced valid JSON on every test. The MLX cells were scored
  with the harness's markdown-fence tolerance (`fenced: true` in the scores):
  at campaign time the MLX runner did not enforce `format:"json"` (no Format
  field on its wire protocol — fixed by x/structured, ADR 0009); the fenced
  bodies were well-formed JSON in all 12 cells, and stripping is a no-op for
  grammar-constrained llama-server output.

## Scene grounding (six objects, norm-1000 boxes) + document extraction

| Model | Engine | Scene bbox IoU | Boxes / labels / colors | Serial | Invoice (items · qty+price · total) | name_bbox hits |
|---|---|---|---|---|---|---|
| gemma4:12b-nvfp4 | **MLX** | **0.947** | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4 |
| gemma4:26b-nvfp4 | **MLX** | **0.968** | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4 |
| gemma4:31b-nvfp4 | **MLX** | **0.964** | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4 |
| gemma4:12b-it-q4_K_M | GGUF | 0.885 | 6/6 · 6/6 · 5/6 | ✅ | 5/5 · 5/5 · ✅ | 3 |
| gemma4:26b-a4b-it-q4_K_M | GGUF | 0.970 | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4 |
| gemma4:31b-it-q4_K_M | GGUF | 0.963 | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4 |
| qwen3.6:35b-a3b-q4_K_M | GGUF | 0.975 | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4 |
| nemotron3:33b-q4_K_M | GGUF | 0.857 | 6/6 · 6/6 · 6/6 | ❌ | 5/5 · 5/5 · ✅ | 4 |

## Fine-text OCR (exact-match recall per size tier, /4) + multi-image + throughput

| Model | Engine | 22px | 16px | 12px | 9px | 7px | Multi-image (3 imgs) | Gen tok/s | Prefill tok/s |
|---|---|---|---|---|---|---|---|---|---|
| gemma4:12b-nvfp4 | **MLX** | 4 | 4 | 2 | 0 | 0 | ✅ all Qs + bbox | 121 | 1766 |
| gemma4:26b-nvfp4 | **MLX** | 4 | 4 | 4 | 2 | 2 | ✅ all Qs + bbox | 151 | 2961 |
| gemma4:31b-nvfp4 | **MLX** | 4 | 4 | 4 | 2 | 1 | ✅ all Qs + bbox | 50 | 519 |
| gemma4:12b-it-q4_K_M | GGUF | 4 | 4 | 3 | 0 | 0 | ✅ all Qs + bbox | 50 | 1338 |
| gemma4:26b-a4b-it-q4_K_M | GGUF | 4 | 4 | 4 | 2 | 2 | ✅ all Qs + bbox | 93 | 496 |
| gemma4:31b-it-q4_K_M | GGUF | 4 | 4 | 4 | 3 | 2 | ✅ all Qs + bbox | 20 | 312 |
| qwen3.6:35b-a3b-q4_K_M | GGUF | 4 | 4 | 4 | 3 | 0 | ✅ all Qs + bbox | 105 | 1012 |
| nemotron3:33b-q4_K_M | GGUF | 4 | 4 | 3 | 1 | 0 | ✅ all Qs + bbox | 107 | 867 |

## Reading the results

- **MLX gemma4 matches its GGUF sibling on every quality axis.** 26b: 0.968
  vs 0.970; 31b: 0.964 vs 0.963 — within single-run noise, on the
  [findings §10](gemma4-bbox-investigation-findings.md) reference band
  (0.970 / 0.961–0.963). Fine-text tiers differ by at most one code per tier
  in either direction — expected, since the weight quantizations differ too
  (nvfp4 vs q4_K_M).
- **Token-cost parity is exact.** Scene prompt cost was 1684 tokens for all
  six gemma4 runs, both engines: the Go preprocessor (`llm.BudgetFillSize`)
  and llama.cpp's 004 patch land on the same ADR 0008 ladder grid.
- **The 12b cell is a finding.** [ADR 0008](adr/0008-gemma4-budget-fill-restores-1120.md)
  records a residual +5.4% vertical error on 12B at budget 1120 that
  "survives 004" on the llama.cpp payload — its 0.885 here reproduces that
  number. The MLX path scores **0.947** at the same budget: the residual does
  **not** reproduce on MLX, suggesting it is llama.cpp-payload-specific
  rather than inherent to the weights. One seed, one scene — a candidate for
  a follow-up sweep before amending the findings doc.
- **Cross-architecture ordering matches the
  [2026-08-02 campaign](vision-campaign-2026-08-02.md)**: qwen3.6 leads
  grounding (0.975), nemotron3 trails (0.857, only model to miss the 14px
  corner serial), gemma4 26b/31b sit just below qwen. Everyone aces the
  invoice.
- **Throughput:** MLX decodes ~2.4× faster than llama-server at matched sizes
  (12b: 121 vs 50 tok/s; 31b: 50 vs 20) and prefills faster. Single samples,
  directional only.

## Caveats

- Engine and weights quantization move together (nvfp4 on MLX vs q4_K_M on
  GGUF) — inherent to comparing these store artifacts.
- One seed, temperature 0, one scene/document/page per test — same scope as
  the suite's other campaigns; treat single-cell deltas ≤ one code / ≤ 0.01
  IoU as noise.
- Warm-decode tok/s come from the scene cell (model already loaded by the
  time it generates); load time is excluded by the cold-server method.
