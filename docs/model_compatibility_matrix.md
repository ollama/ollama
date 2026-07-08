# Model Compatibility Matrix — s390x (IBM Z)

> **Tok/s** — median of 15 consecutive runs, 80-token generation, prompt: `"List 3 facts about the ocean."`. First 1–2 runs excluded (AIU JIT warmup). AIU-accelerated (12 VFs); results vary with host load.
> **RAM** — VmRSS of `llama-server` process read after a completed generation via `/proc/$(pgrep llama-server)/status`.

## Results

| Model | Tag | Quant | Params | Status | Tok/s | RAM |
|-------|-----|-------|--------|--------|-------|-----|
| SmolLM 135M | `smollm:135m` | Q4_0 | 135M | ✅ | 104.6 | 178 MiB |
| SmolLM 360M | `smollm:360m` | Q4_0 | 360M | ✅ | 77.9 | 340 MiB |
| Llama 3.2 1B | `llama3.2:1b` | Q4_K_M | 1B | ✅ | 17.6 | 1.5 GiB |
| Llama 3.2 1B (Q8_0) | `llama3.2:1b-instruct-q8_0` | Q8_0 | 1B | ✅ | 22.75 | 1.5 GiB |
| Llama 3.2 1B (Q5_K_M) | `llama3.2:1b-instruct-q5_k_m` | Q5_K_M | 1B | ✅ | 21.6 | 1.1 GiB |
| Llama 3.2 1B (Q2_K) | `llama3.2:1b-instruct-q2_k` | Q2_K | 1B | ⚠️ | 4.4 | 781 MiB |
| Llama 3.2 1B (F16) | `llama3.2:1b-instruct-fp16` | F16 | 1B | ✅ | 4.9 | 2.5 GiB |
| Llama 3.2 3B | `llama3.2:3b` | Q4_K_M | 3B | ✅ | 12.2 | 2.4 GiB |
| Llama 3.2 3B (Q8_0) | `llama3.2:3b-instruct-q8_0` | Q8_0 | 3B | ✅ | 7.5 | 3.7 GiB |
| Granite 3.3 2B | `granite3.3:2b` | Q4_K_M | 2B | ✅ | 12.25 | 1.9 GiB |
| Mistral 7B | `mistral:7b` | Q4_K_M | 7B | ✅ | 5.8 | 4.6 GiB |
| Qwen2.5 0.5B | `qwen2.5:0.5b` | Q4_K_M | 500M | ❌ | — | — |

**Status:** ✅ Working · ⚠️ Unstable · ❌ Fails

**Q2_K note:** Highly variable throughput (1.4–11.4 tok/s range).  
**Qwen2.5 note:** Garbage output and server crash after first inference.

---

## Quant Coverage

| Format | Status |
|--------|--------|
| Q4_0 | ✅ |
| Q4_K_M | ✅ |
| Q8_0 | ✅ |
| Q5_K_M | ✅ |
| F16 | ✅ |
| Q2_K | ⚠️ |
| IQ4_XS | ❌ |

---

## Recommended Demo Models

| Model | Tag | Why |
|-------|-----|-----|
| Granite 3.3 2B | `granite3.3:2b` | IBM model; best for s390x demo |
| Llama 3.2 1B | `llama3.2:1b` | Industry-standard baseline |

---

## Known Issues

1. **`ollama --version` shows `0.0.0`** — Dev build from source; no version tag.
2. **`llama-server --list-devices` exits 127** — Benign; CPU fallback works correctly.
3. **AIU throughput varies with host load** — Other workloads on the z15 LPAR contend for AIU VFs.
