# Model Test Log 001 — s390x (IBM Z)

**Date:** 2026-07-07 / 2026-07-08
**Tester:** Justin Veltri
**Platform:** IBM Z (s390x), z15 LPAR, 1 TB RAM, 32 logical CPUs
**Container:** Debian-based, AIU-accelerated (12 VFs)
**Ollama build:** from source, `main` branch

---

## Methodology

- Server restarted (`pkill -f "ollama serve"`) before each new model to ensure clean AIU state
- Each model ran 15 consecutive benchmark requests using:
  - Prompt: `"List 3 facts about the ocean."`
  - `num_predict: 80`
  - `stream: false`
- Tok/s = median of warm runs (first 1–2 excluded due to AIU JIT warmup)
- RAM = VmRSS read from `/proc/$(pgrep llama-server)/status` after a completed generation
- Benchmark command:
  ```sh
  for i in $(seq 1 15); do
    curl -s http://localhost:11434/api/generate \
      -d '{"model":"<tag>","prompt":"List 3 facts about the ocean.","options":{"num_predict":80},"stream":false}' \
      | awk -F: '{for(i=1;i<=NF;i++){if($i~/"eval_count"/) count=$(i+1)+0; if($i~/"eval_duration"/) dur=$(i+1)+0}} END {printf "%d tokens in %.1fs = %.1f tok/s\n", count, dur/1e9, count/(dur/1e9)}'
  done
  ```

---

## Results

| Model | Tag | Quant | Status | Tok/s (median) | RAM (VmRSS) |
|-------|-----|-------|--------|----------------|-------------|
| SmolLM 135M | `smollm:135m` | Q4_0 | ✅ | 104.6 | 178 MiB |
| SmolLM 360M | `smollm:360m` | Q4_0 | ✅ | 77.9 | 340 MiB |
| Llama 3.2 1B | `llama3.2:1b` | Q4_K_M | ✅ | 17.6 | 1.5 GiB |
| Llama 3.2 1B (Q8_0) | `llama3.2:1b-instruct-q8_0` | Q8_0 | ✅ | 22.75 | 1.5 GiB |
| Llama 3.2 1B (Q5_K_M) | `llama3.2:1b-instruct-q5_k_m` | Q5_K_M | ✅ | 21.6 | 1.1 GiB |
| Llama 3.2 1B (Q2_K) | `llama3.2:1b-instruct-q2_k` | Q2_K | ⚠️ | 4.4 | 781 MiB |
| Llama 3.2 1B (F16) | `llama3.2:1b-instruct-fp16` | F16 | ✅ | 4.9 | 2.5 GiB |
| Llama 3.2 3B | `llama3.2:3b` | Q4_K_M | ✅ | 12.2 | 2.4 GiB |
| Llama 3.2 3B (Q8_0) | `llama3.2:3b-instruct-q8_0` | Q8_0 | ✅ | 7.5 | 3.7 GiB |
| Granite 3.3 2B | `granite3.3:2b` | Q4_K_M | ✅ | 12.25 | 1.9 GiB |
| Mistral 7B | `mistral:7b` | Q4_K_M | ✅ | 5.8 | 4.6 GiB |
| Qwen2.5 0.5B | `qwen2.5:0.5b` | Q4_K_M | ❌ | — | — |

---

## Observations

### AIU Behavior
- AIU JIT-compiles or caches the compute graph on first 1–2 inferences. Subsequent runs are significantly faster.
- Throughput is non-deterministic — other workloads on the z15 LPAR contend for AIU VFs, causing occasional spikes down to ~1–3 tok/s.
- Restarting `ollama serve` between models ensures a clean AIU state for benchmarking.
- `ollama ps` reports `100% GPU` even though ollama sees `total_vram="0 B"` — AIU is transparent to ollama's scheduler.

### Quant Format Findings
- **Q8_0 outperforms Q4_K_M** for the same model (22.75 vs 17.6 tok/s for llama3.2:1b) — AIU appears to handle higher-precision formats more efficiently.
- **Q2_K is unstable** — throughput ranged 1.4–11.4 tok/s across 15 runs. Not recommended for production use on s390x.
- **F16 is slower than Q8_0** despite higher precision — likely due to larger memory footprint reducing AIU cache efficiency.
- **Q4_0 models (SmolLM) are the fastest** — 77–104 tok/s, well within interactive use range.

### Failures
- **qwen2.5:0.5b** — Produced garbage output on first inference, then server crashed on subsequent requests. Root cause unknown; may be a tensor layout incompatibility with the byteswap patch.

---

## Quant Coverage Summary

| Format | Status |
|--------|--------|
| Q4_0 | ✅ Working |
| Q4_K_M | ✅ Working |
| Q8_0 | ✅ Working |
| Q5_K_M | ✅ Working |
| F16 | ✅ Working |
| Q2_K | ⚠️ Unstable |
| IQ4_XS | ❌ Not supported |

---

## Recommended Demo Models

| Model | Tag | Reason |
|-------|-----|--------|
| Granite 3.3 2B | `granite3.3:2b` | IBM model; strong quality, ~12 tok/s, 1.9 GiB |
| Llama 3.2 1B (Q8_0) | `llama3.2:1b-instruct-q8_0` | Best throughput among Q4K+ models at 22.75 tok/s |
