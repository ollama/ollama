# s390x Big-Endian Inference Support

## Overview

This document describes the changes made on branch `justin-main-experiments` to enable correct LLM inference on IBM Z (s390x) — a big-endian architecture. Before this work, Ollama would build and start on s390x but every model response was a stream of dots (`.....................`) with no real text.

---

## Background

### The endianness problem

GGUF model files are always stored in **little-endian** byte order, regardless of the host platform. On little-endian hosts (x86, ARM) this is transparent — values read directly from disk are immediately usable. On big-endian hosts like IBM Z (s390x), multi-byte values read from disk have their bytes in the wrong order and must be swapped before the CPU can interpret them correctly.

### What was already handled

The existing `002-gguf-big-endian-byteswap.patch` already fixed the **GGUF metadata** layer — the key-value pairs in the file header (architecture name, hyperparameters, tokenizer data, etc.) are byteswapped correctly when the file is opened. The model would therefore load, tokenize input, and start producing tokens — but the tokens were garbage.

### What was missing

The **tensor data** — the actual weights that the model computes with — was not being byteswapped. Every floating-point scale value in every quantized block was byte-reversed, so matrix multiplications and dequantization produced numerically meaningless results. The model was running, but on corrupted weights.

---

## Root Cause Analysis

### Quantized block layouts

GGUF stores weights in quantized block formats. Each block contains:
- **Integer quant bits** — packed nibbles or bytes representing discretized weight values. These are **byte arrays with no endian sensitivity** and need no swapping.
- **FP16/FP32 scale fields** (`d`, `dmin`) — floating-point values used to dequantize the integer bits back to real numbers. These are **multi-byte values that must be byteswapped** on big-endian hosts.

The scale fields sit at different byte offsets depending on the quantization type:

| Type | Block size | Scale field position |
|------|-----------|---------------------|
| F16 / BF16 | 2 bytes each | entire value |
| F32 | 4 bytes each | entire value |
| Q4_0, Q5_0, Q8_0 | 18–34 B | `d` at offset 0 |
| Q4_1, Q5_1 | 20–36 B | `d` at offset 0, `m` at offset 2 |
| Q4_K | 144 B | `d` at offset 0, `dmin` at offset 2 |
| Q5_K | 176 B | `d` at offset 0, `dmin` at offset 2 |
| Q2_K | 84 B | `d` at offset 80, `dmin` at offset 82 |
| Q3_K | 110 B | `d` at offset 108 |
| Q6_K | 210 B | `d` at offset 208 |

Most Ollama-downloaded models use **Q4_K_M** format, which is `GGML_TYPE_Q4_K`. This type was entirely absent from the initial byteswap attempt, explaining why results were always garbage.

### The mmap problem

llama.cpp uses `mmap()` by default to load model files on Linux. mmap maps the file directly into the process's virtual address space as a **read-only region**. There is no separate buffer to swap — the tensor `data` pointer points straight into the file mapping. Attempting to byteswap it would fault. The byteswap must happen into a **writable buffer**, which only exists when mmap is disabled and tensors are loaded via `read()`.

### The wrong hook point

llama.cpp has two tensor loading code paths:
- `load_data_for()` — loads a single tensor, used by offline tools like `llama-quantize`
- `load_all_data()` — loads all tensors in a loop, used by the **actual inference server**

The initial implementation hooked only `load_data_for`. Inference goes through `load_all_data`, so the byteswap never ran during a real `ollama run`.

---

## Changes Made

### 1. `llama/compat/003-tensor-data-big-endian-byteswap.patch`

A patch applied to llama.cpp at build time (pinned to `b9840`) that injects byteswap logic directly into `src/llama-model-loader.cpp`.

**Two functions are added:**

```c
// Inner function — works on a raw buffer
static void bswap_buf(ggml_type type, uint8_t * data, size_t nbytes);

// Thin wrapper for ggml_tensor* callers
static void bswap_tensor_data(struct ggml_tensor * t);
```

Both are compiled only on big-endian hosts via:
```c
#if defined(__BYTE_ORDER__) && defined(__ORDER_BIG_ENDIAN__) && \
    (__BYTE_ORDER__ == __ORDER_BIG_ENDIAN__)
```
On little-endian hosts both functions compile to no-ops with zero overhead.

**Three call sites are patched in:**

1. `load_data_for()` — after `file->read_raw()`, before `check_tensors` validation
2. `load_all_data()`, host-buffer path — after `file->read_raw(cur->data, n_size)`
3. `load_all_data()`, non-host-buffer path — after `file->read_raw(read_buf.data(), n_size)`, before `ggml_backend_tensor_set()`

### 2. `llama/compat/llama-ollama-compat.cpp`

In the `translate_metadata()` function, which runs at model load time before any tensor data is read:

```cpp
#if defined(__BYTE_ORDER__) && defined(__ORDER_BIG_ENDIAN__) && \
    (__BYTE_ORDER__ == __ORDER_BIG_ENDIAN__)
    disable_mmap_for(ml);
#endif
```

This forces llama.cpp off the zero-copy mmap path for all models on big-endian hosts, ensuring every tensor read goes through a writable buffer where `bswap_tensor_data` can operate. The existing `translate_metadata` return-value mechanism already propagates this to `use_mmap = false` in the model loader constructor via the hook in `001-llama-cpp-hooks.patch`.

---

## How the Patches Are Delivered

The patches live in `llama/compat/` and are applied automatically by `apply-patch.cmake` during `cmake -B build` via FetchContent's `PATCH_COMMAND`. They are applied in numeric filename order against a fresh clone of llama.cpp at the pinned commit (`LLAMA_CPP_VERSION = b9840`):

```
001-llama-cpp-hooks.patch                  — compat layer call-site insertions
002-gguf-big-endian-byteswap.patch         — GGUF metadata byteswap (pre-existing)
003-tensor-data-big-endian-byteswap.patch  — tensor weight byteswap (this work)
```

The patches are idempotent — re-running cmake does not re-apply already-applied patches.

---

## Verification

On a running s390x build:

**Terminal 1:**
```sh
./ollama serve
```

Look for this line in the server log when a model loads — it confirms mmap was disabled and the byteswap path will run:
```
compat patch disabled mmap for transformed text tensors
```

**Terminal 2:**
```sh
./ollama run smollm:135m "Hello, how are you?"
```

Before this fix: `.....................`  
After this fix: coherent text response

---

## Limitations and Future Work

- **No GPU support** — s390x has no GGML GPU backend. All inference is CPU-only. The byteswap assumes the non-host-buffer path (`read_buf` via `ggml_backend_tensor_set`) is the only non-CPU path, which is true on s390x today.
- **mmap disabled globally** — Disabling mmap increases memory usage because tensors are copied into heap allocations rather than mapped directly. On s390x this is unavoidable until llama.cpp has a first-class byteswap hook in the loading pipeline.
- **iQuant types not covered** — IQ2_XXS, IQ3_S, IQ4_XS and other iQuant formats are not yet handled in `bswap_buf`. These formats are uncommon in Ollama-published models but may need to be added if they appear.
- **This patch is temporary** — The right long-term fix is for GGUF or llama.cpp to handle big-endian natively. This patch bridges the gap until that happens upstream.

---

## File Reference

| File | Purpose |
|------|---------|
| `llama/compat/003-tensor-data-big-endian-byteswap.patch` | Injects `bswap_buf` / `bswap_tensor_data` into llama.cpp's tensor load paths |
| `llama/compat/llama-ollama-compat.cpp` | Disables mmap on big-endian in `translate_metadata()` |
| `llama/compat/apply-patch.cmake` | Idempotent patch applier invoked by FetchContent |
| `llama/compat/compat.cmake` | Wires the patch command into the FetchContent declaration |
| `LLAMA_CPP_VERSION` | Pinned llama.cpp commit (`b9840`) the patches are written against |
