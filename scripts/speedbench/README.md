# SPEED-Bench shim for Ollama

`ollama_speedbench.py` replays a prepared [SPEED-Bench](https://huggingface.co/datasets/nvidia/SPEED-Bench)
dataset against Ollama's OpenAI-compatible endpoint and reports
speculative-decoding metrics per category: draft acceptance, tokens per step
(acceptance length, bonus token included), time per step, eval rate, prefill
rate, TTFT, and latency. Aggregates are computed from summed counts, never
from averaged ratios. It is runtime-agnostic: it works against any
OpenAI-compatible server, and prints `n/a` for metrics the server does not
report (Ollama reports step counts only where the runner counts them; some
runners report draft counts without step counts).

It is stdlib-only; no Python dependencies.

## Preparing the dataset

SPEED-Bench is governed by the NVIDIA Evaluation Dataset License Agreement,
and the Hugging Face copy carries redistribution placeholders for parts of the
data. Use NVIDIA's official tooling to build the full dataset:

```sh
curl -LsSf https://raw.githubusercontent.com/NVIDIA-NeMo/Skills/refs/heads/main/nemo_skills/dataset/speed-bench/prepare.py | python3 - --output_dir ./speed-bench-data
```

This produces JSONL splits (e.g. `qualitative.jsonl`, `throughput_2k.jsonl`).
Each prepared record must end up with:

```json
{"question_id": "...", "category": "coding", "turns": ["first user message", "optional follow-up"]}
```

`turns` may hold 1–5 user messages; multi-turn records are replayed in order
with the live assistant replies as history. Records still containing the
placeholder notice are skipped. A plain `{"prompt": "...", "category": "..."}`
line is also accepted for smoke testing.

## Running

```sh
python3 scripts/speedbench/ollama_speedbench.py \
  --model gemma4:12b-mtp \
  --input ./speed-bench-data/qualitative.jsonl \
  --json-out report.json
```

Options: `--url` (default `http://localhost:11434`), `--temperature` (default
0, matching the SPEED-Bench non-reasoning methodology), `--max-tokens`
(default 4096), `--category` (repeatable), `--limit`, `--timeout`.

Requests run serially, so time-per-step numbers describe an idle server;
concurrency is deliberately not offered yet.

## Reading the report

- **draft acceptance**: drafted tokens accepted ÷ drafted. Not comparable
  across drafting methods (it depends on draft depth); use eval rate to
  compare methods.
- **tokens per step** (acceptance length): tokens committed per target
  forward pass, including steps that did not draft.
- **time per step**: wall time per target forward pass.
- tokens per step ÷ time per step equals the eval rate exactly when the
  server reports steps; that identity is one way to sanity-check a run.

## Provenance

- Dataset: `nvidia/SPEED-Bench` (Hugging Face), prepared with NVIDIA
  NeMo-Skills `speed-bench/prepare.py`. Paper: arXiv:2604.09557.
- The shim does not ship any dataset content; `--input` is always a local,
  user-prepared file.
