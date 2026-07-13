# Issue #17116 — "CUDA error: the provided PTX was compiled with an unsupported toolchain"

## 1. Root cause

Ollama's Windows CUDA 12 build compiles the older compute targets as **PTX only**
(`llama/server/CMakePresets.json`, `llama_cuda_v12_windows`):

```
CMAKE_CUDA_ARCHITECTURES = 50-virtual;52-virtual;60-virtual;61-virtual;70;75;80;86;89;90;90a;120
```

So on a Maxwell/Pascal card (CC 5.x/6.x — e.g. the GTX 1060 in the near-identical
issue #17012) there is no SASS in the fatbin; the NVIDIA driver must JIT the PTX at
model load. That PTX is emitted by the CUDA 12.8 toolkit, and a driver older than
570 cannot consume it. The driver rejects it with `CUDA_ERROR_UNSUPPORTED_PTX_VERSION`
("the provided PTX was compiled with an unsupported toolchain"), ggml's `CUDA_CHECK`
calls `abort()`, and the user sees llama-server die with `0xc0000409`
(STATUS_STACK_BUFFER_OVERRUN — how the Windows CRT reports `abort()`).

`discover.filterOldCUDADriver` (`discover/cuda_compat.go`) exists precisely to catch
this: it drops CUDA devices whose driver is below the floor implied by the CUDA
runtime about to be loaded (550 for compressed fatbins, 570 for JIT-ing legacy
compute targets). **On Windows it silently did nothing.**

The gate is keyed off the CUDA runtime *minor* version, which `cudaRuntimeVersion`
(`discover/llama_server.go`) infers from the filenames staged in the variant
directory. Two bugs made it report `12.0` on every Windows install:

1. `cudaRuntimeDLLRegex` was `^cudart64_(\d{2})(\d)\.dll$` — three digits, which is
   the **CUDA ≤ 11** naming scheme (`cudart64_110.dll`). CUDA 12 and 13 ship
   `cudart64_12.dll` / `cudart64_13.dll`, carrying only the major. The regex never
   matched, so the shipped runtime was not seen at all.
2. `cudaRuntimeVersion` then fell through to its directory-name fallback
   (`cuda_v12`) and called `update(major, 0)` — **fabricating a minor of `0`**. It
   had no way to express "major known, minor unknown", and its `libcudart.so.<major>`
   path had the same defect (`matches[2] == ""` → `minor = 0`).

`filterOldCUDADriver` therefore concluded the runtime was CUDA **12.0**: too old to
use compressed fatbins (needs ≥ 12.4) and too old to JIT legacy compute (needs ≥ 12.8).
Both guards evaluated false, it hit the early return, and **no device was filtered**.
The Pascal GPU was handed to llama-server, which then aborted JIT-ing the PTX.

This is why the earlier fix for the same crash (#16994, which added
`filterOldCUDADriver`) appeared to work and why the existing tests all passed: they
only ever exercise the Linux layout (`libcudart.so.12.8.90`), which parses correctly.
Linux was fixed; Windows — the platform where Pascal is PTX-only and where the crash
was actually reported — was not.

## 2. The fix and why

Make "minor version unknown" a first-class value instead of silently coercing it to `0`:

- **`discover/llama_server.go`**
  - `cudaRuntimeDLLRegex` → `^cudart64_(\d{2})(\d)?\.dll$`, so it matches both the
    legacy `cudart64_110.dll` and the modern `cudart64_12.dll` / `cudart64_13.dll`.
  - New `cudaRuntimeMinorUnknown = -1`. `cudaRuntimeVersion` now returns it whenever
    only the major is discoverable (CUDA 12+ DLL, bare `libcudart.so.12`, or the
    directory-name fallback) rather than claiming `.0`. Its `update` closure already
    keeps the highest value seen, so a real minor from a fully-versioned `.so` still
    wins over an unknown from the directory name.
  - Guarded the one other consumer so the sentinel never leaks into
    `DeviceInfo.DriverMinor`.

- **`discover/cuda_compat.go`** — an unknown minor is now treated as *at least* the
  runtime we ship, via a `runtimeMinorAtLeast` helper. This is the safety-critical
  half: guessing low disables the driver floors and crashes the runner, whereas
  guessing high drops the GPU with the existing `"NVIDIA driver too old"` warning and
  falls back to CPU. A wrong guess in the safe direction costs performance; a wrong
  guess in the unsafe direction costs the whole server.

The source-build carve-out the gate was written for still works wherever the minor is
actually discoverable — i.e. every Linux build, which is where it was aimed
(`libcudart.so.12.2.140` → 12.2 → no filtering, as covered by the existing
"source build with older CUDA runtime keeps devices" test).

I deliberately did **not** touch the CMake arch lists. Adding real SASS for Pascal to
the Windows build would also fix the crash but inflates the binary for every user, and
the driver gate is the mechanism the project already chose for this.

## 3. Files changed

| File | Change |
|---|---|
| `discover/llama_server.go` | Fix `cudaRuntimeDLLRegex` for CUDA 12+ DLL naming; add `cudaRuntimeMinorUnknown`; return it from `cudaRuntimeVersion` instead of a fabricated `0`; guard the `DriverMinor` assignment. |
| `discover/cuda_compat.go` | Treat an unknown runtime minor as "at least what we ship" so the 550/570 driver floors are applied instead of skipped. |
| `discover/cuda_compat_test.go` | Two regression cases on the Windows `cudart64_12.dll` layout: driver 560 drops the GTX 1060 but keeps the RTX 4060 Ti; driver 535 drops all CUDA devices. |
| `discover/llama_server_test.go` | Extend the `cuda runtime version` subtest: legacy `cudart64_110.dll` → 11.0; `cudart64_12.dll` → 12.unknown; directory fallback → unknown, not `.0`. |

## 4. Risk / uncertainty

- **The behavioural trade-off is real but strongly favourable.** A Windows user who
  *builds from source* with an old CUDA toolkit (12.0–12.7) **and** an old driver now
  has their GPU dropped to CPU, because on Windows the minor genuinely cannot be read
  from the file layout. Previously they'd have worked, since a source build defaults
  to `CMAKE_CUDA_ARCHITECTURES=native` (real SASS, no JIT). This is a small, silent-
  failure-free population (they get an explicit "NVIDIA driver too old" warning), and
  the alternative is that every Windows *release* user on Pascal/Maxwell with a
  pre-570 driver hard-crashes. If that trade-off is unacceptable, the precise fix is to
  read the DLL's Win32 `VERSIONINFO` resource (`cudart64_12.dll` does carry the full
  `12.8.90`), which removes the guess entirely — I left it out as it's a larger,
  platform-specific change I can't exercise from Linux.
- **I could not run this on real Windows/NVIDIA hardware.** The crash path itself
  (driver rejecting the PTX) is not reproducible here; I verified the *decision* that
  leads into it, which is the code under test.
- The CUDA 12 DLL naming (`cudart64_12.dll`, major only) is the load-bearing external
  fact. Confirmed against NVIDIA's toolkit layout
  (`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin\cudart64_12.dll`)
  rather than assumed.
- `cuda_v13` is still never passed through `filterOldCUDADriver` (`discover/runner.go:119`
  gates on `cuda_v12` exactly). That's benign today — a driver too old for CUDA 13
  fails device enumeration outright, so the variant drops out on its own — but it is a
  latent asymmetry. Left alone to keep this change scoped to the reported issue.

## 5. How I verified it

Go toolchain at `/home/pjsump/go-sdk/go/bin` (go1.26.0).

**Proved the new tests actually catch the bug** — reverted the two source files,
kept the new tests, and ran them against the unfixed code:

```
--- FAIL: TestFilterOldCUDADriver/cuda_12_runtime_without_a_minor_version_filters_older_CUDA_devices
    cuda_compat_test.go:174: got 2 devices, want 1: [... "NVIDIA GeForce GTX 1060 6GB", ComputeMajor:6, ComputeMinor:1, NVIDIADriverMajor:560 ...]
--- FAIL: TestFilterOldCUDADriver/cuda_12_runtime_without_a_minor_version_filters_all_CUDA_devices_on_a_pre-compression_driver
    cuda_compat_test.go:174: got 1 devices, want 0: [... "NVIDIA V100" ... NVIDIADriverMajor:535 ...]
--- FAIL: TestLlamaServerDiscovery/cuda_runtime_version
    llama_server_test.go:472: cudaRuntimeVersion cuda 12 dll = 12.0, true, want 12.unknown, true
```

The first failure *is* issue #17116: the GTX 1060 on driver 560 is kept, and is then
handed to llama-server to JIT PTX its driver cannot consume.

**With the fix applied:**

- `go test ./discover/... ./ml/... ./llm/...` → all `ok`. No existing test needed its
  expectations changed except the one that pinned the buggy `minor = 0` fallback.
- `go test ./discover/ -run 'TestFilterOldCUDADriver|TestLlamaServerDiscovery' -v` →
  all 7 `TestFilterOldCUDADriver` cases pass, including both new ones, alongside the
  pre-existing arch-filter and source-build cases.
- `go vet ./discover/...` clean; `gofmt -l discover/` clean.
- `GOOS=windows go build ./discover/... ./llm/... ./ml/...` → OK (the fix is
  Windows-facing, so this matters).

Two failures I confirmed are **pre-existing and unrelated** by reproducing them on a
stashed clean tree: `GOOS=windows go vet` flags `native_probe_windows.go:456`
(already carries a `//nolint:govet`), and `GOOS=darwin go build` reports
`undefined: GetCPUMem` (it lives in the cgo-gated `gpu_darwin.go`, excluded when
cross-compiling with `CGO_ENABLED=0`).
