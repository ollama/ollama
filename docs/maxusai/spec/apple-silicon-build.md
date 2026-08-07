# Running the patched fork on Apple Silicon

MaxusAI-fork spec (fork-only; does not exist upstream). Written 2026-08-07.

Two artifacts, because one cannot do both jobs:

| | Native | Container |
|---|---|---|
| Target | this Mac, run directly | any Apple M-family Mac, under Docker |
| Preset | `darwin` | `cpu` |
| Compute | Metal GPU + MLX `metal_v4` | **CPU only** |
| llama.cpp patches (001–005) | yes | yes |
| Go patches | yes | yes |

## Why two

Containers on Apple Silicon run inside a Linux VM and **there is no Metal
passthrough to a Linux guest**. Any container on a Mac is CPU-only, whatever the
runtime (Docker, Podman, Apple `container`). Metal comes only from a native macOS
arm64 build — see [`docs/development.md`](../../development.md) ("On macOS arm64,
this builds Metal inference").

So: native for speed, container for distribution to other M-family Macs.

## The patch mechanism is shared

Both artifacts get the full patch set from one place.
[`llama/compat/apply-patch.cmake`](../../../llama/compat/apply-patch.cmake) uses
`file(GLOB_RECURSE)` over `llama/compat/*.patch` and applies the results in
numeric filename order during llama.cpp's `FetchContent`, wired in by
[`compat.cmake`](../../../llama/compat/compat.cmake) at
[`llama/server/CMakeLists.txt:134`](../../../llama/server/CMakeLists.txt).

The glob is **recursive**, so subdirectory patches are included. The set is five,
applied in numeric filename order:

| Patch | Purpose |
|---|---|
| `001-llama-cpp-hooks.patch` | compat call-sites |
| `002-llama-cpp-nemotron-dynres.patch` | nemotron dynamic resolution |
| `models/003-llama-cpp-laguna-metal.patch` | Laguna, Metal path |
| `004-llama-cpp-gemma4-budget-fill.patch` | gemma4 reference sizing: ladder snap + budget fill + `PAD_NONE` |
| `005-llama-cpp-dynres-pinned-overshoot.patch` | dyn_size pinned budgets: never exceed max_pixels |

Both the `darwin` and `cpu` presets route through `llama/server`, so all five are
applied to the fetched source in each build. No bespoke plumbing, and no
divergence between the two artifacts.

Note that 003 patches the Metal path specifically. It applies in the container
build too — patching is a source-level step — but the code it touches is not
compiled there, since the container has no Metal backend.

## The overlay shortcut is invalid here — do not use it

[`Dockerfile.gemma4budget`](../../../Dockerfile.gemma4budget) copies the compiled
C++ payload from the `ollama/ollama` base image and replaces only
`/usr/bin/ollama`. On a tree containing
[`002-llama-cpp-nemotron-dynres.patch`](../../../llama/compat/002-llama-cpp-nemotron-dynres.patch)
that produces an image which **looks patched and is not** — 002 changes the
compiled payload, so an overlay silently drops it.

This is already documented as a hard constraint in
[`nemotron-dynres-patch.md`](../nemotron-dynres-patch.md#deployment-constraints):
*"do not build overlays from a tree containing 002."*

The Apple-ARM container therefore does a **full CMake build**. This is affordable
only because it needs one backend: `cpu`. The 45–70 GB figure that motivated the
overlay applies to the full fleet matrix (ROCm base + CUDA 12 + CUDA 13 + Vulkan +
MLX), none of which a Mac can use.

## Native build (macOS arm64, Metal + MLX)

### Prerequisites

| Requirement | Verified 2026-08-07 |
|---|---|
| Go ≥ 1.26.0 (`go.mod`) | 1.26.5 (brew) |
| Ninja | 1.13.2 (brew) |
| Full Xcode, selected | 26.6, `/Applications/Xcode.app` |
| Metal toolchain | 17F109, `installed` |

The Metal toolchain is **not optional**. `ollama_default_mlx_backends()` in
[`cmake/local.cmake`](../../../cmake/local.cmake) calls
`ollama_check_metal_toolchain()` unconditionally on macOS arm64, so configure
hard-fails without it:

```sh
xcodebuild -downloadComponent MetalToolchain
```

### MLX backend selection

Automatic — no flags. `metal_v4` is chosen when both macOS and the macOS SDK are
≥ 26.2, otherwise `metal_v3`. On this host (macOS 26.6, SDK 26.5) that resolves to
**`metal_v4`**. The version regex captures `major.minor`, so `26.5 ≥ 26.2` compares
as intended.

### Build

```sh
cmake -B build .
cmake --build build --parallel 18
go build -trimpath -ldflags="-X=github.com/ollama/ollama/version.Version=<version>" -o ollama .
```

Stamp the version the way [`Dockerfile.gemma4budget`](../../../Dockerfile.gemma4budget)
does, so `ollama --version` identifies the build as a fork artifact rather than
reporting `0.0.0`.

## Container (`Dockerfile.applearm`)

Fork-only file, following the `Dockerfile.gemma4budget` precedent: the fork keeps
its build files separate from the upstream-tracked [`Dockerfile`](../../../Dockerfile)
so upstream syncs do not conflict.

The stock `Dockerfile` cannot be reused as-is: its `arm64` assembly stage hardcodes
CUDA v12, CUDA v13, Jetpack 5 and Jetpack 6 — all NVIDIA, none reachable from a Mac.

### Stages

1. **Build base** — `almalinux:8` (arm64) + `gcc-toolset-13-{gcc,gcc-c++,binutils}`,
   cmake, ninja, git. Verified available for aarch64 at `13.3.1-2.2.el8_10`.
   Omit the NVIDIA CUDA sbsa repo that the stock `base-arm64` adds; nothing needs it.
2. **llama-server** — `cmake -S llama/server --preset cpu`. Applies 001–005.
3. **Go** — build the fork binary with the version stamp.
4. **Runtime** — `ubuntu:24.04` + `ca-certificates` + `libopenblas0`.
   No `libvulkan1`; there is no Vulkan backend in this image.

Building on glibc 2.28 (AlmaLinux 8) and running on glibc 2.39 (Ubuntu 24.04) is
forward-compatible, and is what the stock `Dockerfile` already does.

The file header must state that this is **not** an overlay and must not be
converted into one while 002 is in-tree.

### Docker resource floor

Docker Desktop defaults are far below what this build needs. Measured 2026-08-07:
12 CPUs, **8.2 GB RAM** on a 128 GB host. `GGML_CPU_ALL_VARIANTS=ON` compiles
several CPU variants in parallel; at `-j12` in 8 GB this is expected to OOM.
Raise the VM memory allocation before building, or lower build parallelism.

## Verification

Each patch tier is checked independently. "It compiled" is not evidence that a
patch is live.

| What | Check | Pass |
|---|---|---|
| Go patches | `go test ./api/... ./llm/... ./server/...` | green |
| Patches applied — native | build output (superbuild: appears during `cmake --build`, not root configure) | all five `llama/compat: applied …` lines |
| Patches applied — container | **source inspection, not log grep** — see below | 001–005 hunks present in the fetched source |
| 002 live | nemotron vision request, per-image token cost | up to 3,328 (stock shows 256) |
| Metal live (native) | `build/lib/ollama` contents; serve logs | `mlx_metal_v4` present, Metal backend selected |
| Container identity | `docker run --rm <img> --version` | fork version string |

A `FATAL_ERROR` from `apply-patch.cmake` means a patch no longer fits the pinned
`LLAMA_CPP_VERSION` (currently `b10091`) and must be regenerated — the message says
so explicitly. Because it is fatal, a build that *completes* has necessarily applied
every patch. That is the strongest guarantee available, and it is worth more than
the log lines.

### Do not verify the container by grepping the build log

The `llama/compat: applied …` lines **do not appear** in the container build output.
Confirmed 2026-08-07 with a forced `--no-cache-filter --progress=plain` rebuild: the
stage genuinely re-ran (37.6 s configure) and the lines were still absent. The
container uses CMake 3.31, whose `FetchContent` routes the populate sub-build's
output to a log file instead of the console; the native build uses CMake 4.4.2,
which forwards it. BuildKit also truncates long step output with `#NN ...` markers,
which masks the same thing.

Verify against the **source the build actually compiled** instead. Build the
`llama-server-cpu` stage as a target, then inspect `_deps/llama_cpp-src`:

```sh
docker build -f Dockerfile.applearm --platform linux/arm64 \
    --target llama-server-cpu -t applearm-patchproof .

docker run --rm --entrypoint sh --platform linux/arm64 applearm-patchproof -c '
S=build/llama-server-cpu/_deps/llama_cpp-src
grep -n "set_limit_image_tokens(256, 3328)" $S/tools/mtmd/clip.cpp                 # 002
grep -n "resize_position_embeddings(GGML_SCALE_MODE_BICUBIC)" \
    $S/tools/mtmd/models/nemotron-v2-vl.cpp                                        # 002
grep -c "llama_ollama_compat" $S/src/llama-model-loader.cpp                        # 001
grep -n "GGML_USE_METAL" $S/src/models/laguna.cpp                                  # 003
'
```

Note the path is 10 directories deep — a `find -maxdepth 8` will miss it.

Do **not** use the string `"<img>"` as a 002 marker. It occurs twice in `mtmd.cpp`
(another projector uses it), so its presence proves nothing.

### Two things that do not transfer

- **The artifacts verify separately.** The container cannot validate Metal
  behavior; the native build cannot validate the Linux CPU payload. Neither
  substitutes for the other.
- **The vision-suite baselines are not valid here.** Every number under
  [`docs/maxusai/vision-suite/runs/`](../vision-suite/runs/) was measured on the
  Linux CUDA/ROCm fleet. Metal and CPU-arm64 are different backends; reusing those
  parsed JSONs as expected values will manufacture false regressions. Any vision
  measurement on Apple Silicon needs its own baseline.

## Scope

CPU-only, Apple M-family only. No CUDA switch is included — there is no Apple ARM
target that could use one. If a non-Apple linux/arm64 target appears later (Grace,
Jetson), that is a separate decision, not a flag on this file.
