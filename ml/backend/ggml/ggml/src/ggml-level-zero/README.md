# Intel Level Zero GGML Backend

This directory contains the Intel Level Zero (oneAPI L0) backend for GGML, enabling
Ollama to run inference on Intel Arc discrete GPUs, Intel Iris Xe integrated GPUs,
and Intel NPU (Meteor Lake / Lunar Lake / Arrow Lake VPU silicon).

## Build Prerequisites

1. **Intel Level Zero development headers**

   Debian/Ubuntu:
   ```
   apt-get install level-zero-dev
   ```
   RHEL/Fedora:
   ```
   dnf install intel-oneapi-level-zero-devel
   ```
   Windows (MSVC):
   Set `ONEAPI_ROOT` to your Intel oneAPI Base Toolkit installation directory
   and re-run CMake.

2. **Clang with SPIR-V target** (for AOT kernel compilation)

   The Intel oneAPI Base Toolkit ships a clang build with `-target spir64` support.
   Install `intel-oneapi-compiler-dpcpp-cpp` or the full Base Toolkit.

   If clang-spirv is not found, CMake emits a WARNING and the backend falls back
   to runtime JIT (see ADR-L0-003). Performance is identical after warmup.

3. **CMake >= 3.21**

## Build

```bash
cmake -B build --preset "Level Zero"
cmake --build build --config Release -- -j$(nproc)
```

To include NPU support (Intel Meteor Lake / Lunar Lake):
```bash
cmake -B build --preset "Level Zero NPU"
cmake --build build --config Release -- -j$(nproc)
```

Debug build:
```bash
cmake -B build --preset "Level Zero Debug"
cmake --build build
```

## Runtime Prerequisites

Install the Intel Compute Runtime and Level Zero loader on the target system:

Debian/Ubuntu:
```
apt-get install intel-opencl-icd intel-level-zero-gpu level-zero
```

RHEL/Fedora:
```
dnf install intel-opencl intel-level-zero-gpu level-zero
```

The loader library (`libze_loader.so.1` on Linux, `ze_loader.dll` on Windows) is
discovered at runtime via `dlopen` — it is NOT linked at build time. If the loader
is not on `LD_LIBRARY_PATH` or the system library path, the backend logs a debug
message and reports zero Level Zero devices. No error is surfaced to the user.

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `ZE_AFFINITY_MASK` | (unset) | Intel Level Zero pass-through. Comma-separated list of device indices to expose. Example: `ZE_AFFINITY_MASK=0` exposes only the first device. |
| `OLLAMA_L0_DEVICE_INDEX` | (unset) | Restrict Ollama to a single Level Zero device by 0-based index. |
| `OLLAMA_L0_NPU_ENABLE` | `0` | Set to `1` to enumerate Intel NPU (VPU-type) devices. NPU enumeration is opt-in because performance is model-size-sensitive. See Known Limits below. |

## Device Ordering

When `OLLAMA_L0_NPU_ENABLE=0` (the default):
- Only GPU-type L0 devices (Arc dGPU, Iris Xe iGPU) are enumerated.
- Device indices are assigned in driver-enumeration order (typically Arc before Iris Xe
  on systems with both).

When `OLLAMA_L0_NPU_ENABLE=1`:
- GPU devices are enumerated first, then NPU (VPU) devices.
- The scheduler uses free-memory heuristics to decide placement — large models will
  not fit on the NPU's smaller memory and will be placed on the GPU or CPU instead.

## Known Limits

- **NPU model size cap**: The Intel NPU (Meteor Lake / Lunar Lake) supports approximately
  8B-parameter Q4 models comfortably. The backend clamps reported NPU memory to 3 GB to
  prevent the scheduler from over-committing large models to the NPU. FP32 compute is
  not efficient on some NPU SKUs; the backend returns `supports_fp16 = 1` where
  applicable.

- **SPIR-V JIT fallback**: On Intel Gen9 integrated GPUs with older driver versions,
  `zeModuleCreate` may return `ZE_RESULT_ERROR_MODULE_BUILD_FAILURE` for the AOT SPIR-V
  blobs. The backend automatically retries with the inlined OpenCL C JIT path and logs
  a WARNING. The JIT path adds approximately 500 ms latency on the first inference per
  kernel (one-time cost, cached thereafter).

- **NPU and FP32**: Some Intel NPU firmware releases do not support FP32 accumulation
  efficiently. The GGML `supports_op()` hook returns `false` for F32 ops when
  `supports_fp16 = 0`, causing the scheduler to fall back to the CPU for those layers.

## Troubleshooting

**`ze_loader` not found**

Symptom: `ollama ps` shows no Level Zero devices, and Ollama debug logs contain
`ZE_OLLAMA_ERR_LOADER_MISSING`.

Fix: Install the `level-zero` runtime package and ensure `libze_loader.so.1` is on
`LD_LIBRARY_PATH` or `/etc/ld.so.conf.d/`.

**Zero devices enumerated**

Symptom: `ze_ollama_init()` succeeds but `ze_ollama_enumerate_devices()` returns 0.

Possible causes:
1. Intel Compute Runtime not installed — install `intel-level-zero-gpu`.
2. Running in a container without device passthrough — add `--device=/dev/dri` and
   optionally `--device=/dev/accel` for NPU.
3. `ZE_AFFINITY_MASK` set to an empty value or non-existent index.

**SPIR-V module build failure**

Symptom: WARNING logged `SPIR-V module build failure on device X — falling back to JIT`.

Fix: Update Intel Compute Runtime to version 24.39 or newer.
     Check version: `clinfo | grep 'Driver Version'`.

## License

MIT. See `SPDX-License-Identifier: MIT` in every source file.

The Intel Level Zero loader (`libze_loader`) and Intel Compute Runtime driver are
also MIT-licensed. SPIR-V headers (build-time only) are Apache-2.0 WITH LLVM-exception,
which is MIT-compatible. No GPL or LGPL code is pulled in at any stage.
