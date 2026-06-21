# TurboQuant Integration — Build & Test Guide

## Changes Made

| File | Change |
|------|--------|
| `llama/server/CMakeLists.txt:180` | FetchContent → `nomadstar/llama-cpp-turboquant` tag `feature/triattention` |
| `llm/server.go:118-121` | Default `kvCacheType` to `turbo3_0` when `OLLAMA_KV_CACHE_TYPE` empty |
| `envconfig/config.go:317` | Updated default doc string |
| `LLAMA_CPP_VERSION` | Point to turboquant fork |
| `llama/compat/llama-cpp-hooks.patch` | Adjusted `clip.cpp` hunks for turboquant line offsets |

## Build

```bash
cd ~/Github/ollama

# CUDA build (NVIDIA 4GB)
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# OR: ROCm build (AMD 16GB)
cmake -B build -DGGML_HIPBLAS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

## Test

```bash
# Run a small model with turbo3_0 default
OLLAMA_KV_CACHE_TYPE=turbo3_0 ./build/bin/ollama run llama3.2:1b

# Compare against original f16
OLLAMA_KV_CACHE_TYPE=f16 ./build/bin/ollama run llama3.2:1b

# Try other turboquant variants
OLLAMA_KV_CACHE_TYPE=turbo2_0 ./build/bin/ollama run llama3.2:1b
OLLAMA_KV_CACHE_TYPE=turbo4_0 ./build/bin/ollama run llama3.2:1b
```

## Verify the Fork Is Active

Check llama-server logs for: `cache_type_k = t32` (or whatever turboquant type).

```bash
# Verbose output shows cache type
OLLAMA_DEBUG=1 ./build/bin/ollama run llama3.2:1b 2>&1 | grep -i cache
```

## GPU Memory Check

```bash
watch -n 1 nvidia-smi
```

Expected improvement: turbo3_0 compresses KV cache ~4.6× vs f16, so context
lengths that previously OOM'd should now fit in 4GB.

## If Build Fails

Most likely cause: the compat patch (`llama-cpp-hooks.patch`) doesn't match the
turboquant fork. Rebuild from scratch:

```bash
git clean -fdx llama/  # nuke cached llama.cpp
cmake -B build -DGGML_CUDA=ON
cmake --build build -j$(nproc)
```

## Important Notes

- The turboquant fork (`nomadstar/llama-cpp-turboquant`, branch
  `feature/triattention`) is at
  `https://github.com/nomadstar/llama-cpp-turboquant.git`
- Ollama fetches it via CMake at build time — no submodule needed
- The patch in `llama/compat/llama-cpp-hooks.patch` applies on top of
  the turboquant fork, adding Ollama's monolithic-GGUF loading support

## Known Issues & TODO

- **Qwen2.5 / Newer Architectures Support**:
  - **Issue:** Models like `Qwen2.5-Coder-1.5B-Instruct` fail (producing gibberish or repetitive outputs) even under standard `f16` cache mode. This is due to the older upstream `llama.cpp` base version used in this fork (which lacks support for Qwen2.5 RoPE scaling, metadata, and architecture details).
  - **TODO:** Upgrade the base `llama.cpp` code in the `llama-cpp-turboquant` repository to a newer upstream release, and adjust the custom TurboQuant/TriAttention patches accordingly.

