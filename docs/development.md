# Development

Install prerequisites:

- [Go](https://go.dev/doc/install)
- [CMake](https://cmake.org/download/) 3.24 or newer
- C/C++ compiler: Clang on macOS, Visual Studio 2022 C++ tools on Windows, or GCC/Clang on Linux
- [Ninja](https://github.com/ninja-build/ninja/releases) in `PATH` is recommended, especially on Windows
- For gRPC development (see `docs/grpc-phased-reliable-approach.md`): [buf](https://buf.build/docs/installation) (for proto lint/generate). Run `go get -tool google.golang.org/protobuf/cmd/protoc-gen-go connectrpc.com/connect/cmd/protoc-gen-connect-go` (or ensure in PATH via go bin). Re-run `~/.local/bin/logloom build && ~/.local/bin/logloom report` (or equiv) after any changes to `server/routes.go`, `server/sched.go`, or new gRPC code to measure observability lift per the reliable overlay. Always follow the phased approach in `docs/grpc-phased-reliable-approach.md` for implementation (use todo_write, reliable checklist, gates).

For pure Go iteration against an existing native payload, run Ollama from the repository root:

```shell
go run . serve
```

> [!NOTE]
> Ollama includes native code compiled with CGO.  From time to time these data structures can change and CGO can get out of sync resulting in unexpected crashes.  You can force a full build of the native code by running `go clean -cache` first. 

## Quick local iteration (recommended on this branch)

When working on gRPC changes, adapters, converters, the scheduler, clients, or any code that needs **real model execution** (non-zero `prompt_eval_count` / `eval_count`, actual streaming, tools, etc.), you need both the Go API layer **and** a fresh `llama-server` native payload (Metal on Apple Silicon).

Use the convenience target / script:

```shell
make local          # or: make dev
# or directly:
./scripts/build-local.sh
```

This runs the full cmake + Go build and produces:
- `./ollama` (the Go binary with REST + gRPC/Connect surfaces)
- `build/lib/ollama/llama-server` (and supporting Metal libs) — the runner the Go code actually spawns

Both the script and the Makefile are the supported workflow for fast iteration. See `./scripts/build-local.sh --help` for flags:

- `make go` (or `./scripts/build-local.sh --go-only`) — extremely fast Go-only rebuild once the native payload has been built once. Use this for most day-to-day gRPC handler / client / converter work.
- `make clean`
- Passing extra args: `OLLAMA_MLX_BACKENDS= ./scripts/build-local.sh` (lighter build, no MLX).

After the build you can immediately test the dual surface exactly as used in the gRPC reports and harness:

```shell
OLLAMA_GRPC_HOST=127.0.0.1:11435 ./ollama serve
```

Or run the integration matrix:

```shell
OLLAMA_GRPC_HOST=127.0.0.1:11435 go test -tags=integration -run TestGRPCStreaming ./integration -count=1 -timeout=5m
```

**Quality + parity + edge case comparison script (recommended for validating gRPC transport fixes, token counts, and advanced behaviors with real model runs):**

After `./ollama serve` (with gRPC enabled), run the comparison from repo root. It exercises REST vs gRPC (Connect client) side-by-side for the same inputs, produces metrics tables (prompt/completion tokens, TTFT, TPS, wire bytes, tool calls, etc.), and covers edges. Supports separate-port (default) and SAMEPORT.

```shell
# Separate ports (most common for dev/reports)
OLLAMA_GRPC_HOST=127.0.0.1:11435 ./ollama serve &
go run scripts/quality-grpc-comparison.go -model llama3.2:1b

# SAMEPORT (cmux on primary port)
OLLAMA_GRPC_SAMEPORT=1 ./ollama serve &
go run scripts/quality-grpc-comparison.go -sameport -rest http://127.0.0.1:11434 -grpc 127.0.0.1:11434 -model llama3.2:1b
```

See script header comments and `docs/grpc-phased-reliable-approach.md` for full usage, expected matching tokens (core is shared), and edge monitoring (tools in gRPC streams, mid-gen cancellation reporting partial tokens, OTEL/metrics/pprof probes if exposed).

All artifacts (`build/`, the root `/ollama` binary, `dist/`, `integration/ollama`, etc.) are covered by `.gitignore`.

## Native build model (manual equivalent)

For a fresh checkout, or after changing native code, build from the repository root. On macOS arm64, this builds Metal inference. On all other platforms this builds CPU-only inference. It builds the Go binary at the repository root and installs the native runtime payload under `build/lib/ollama`.

```shell
cmake -B build .
cmake --build build --parallel 8
./ollama serve
```

To install into a standard prefix layout:

```shell
cmake --install build --prefix /path/to/install
```

On all platforms except macOS arm64, to build GPU backends select the backends explicitly:

```shell
cmake -B build . -DOLLAMA_LLAMA_BACKENDS="cuda_v13;vulkan"
cmake --build build --parallel 8
```

Supported backend values are `cuda_v12`, `cuda_v13`, `rocm_v7_1`, `rocm_v7_2`, `vulkan`, `cuda_jetpack5`, and `cuda_jetpack6`.

Use standard CMake architecture overrides to narrow GPU builds for local hardware:

```shell
# CUDA
cmake -B build . -DOLLAMA_LLAMA_BACKENDS=cuda_v13 -DCMAKE_CUDA_ARCHITECTURES=native

# ROCm / HIP
cmake -B build . -DOLLAMA_LLAMA_BACKENDS=rocm_v7_2 -DCMAKE_HIP_ARCHITECTURES=gfx1100
```

You can tune GGML build options by setting `GGML_*` values during configure. For example, to build CUDA v12 for Pascal without flash attention kernels:

```shell
cmake -B build . -DOLLAMA_LLAMA_BACKENDS=cuda_v12 -DCMAKE_CUDA_ARCHITECTURES=61 -DGGML_CUDA_FA=OFF
```

## macOS (Apple Silicon)

Additional prerequisites:

MLX Metal requires the Metal toolchain. Install [Xcode](https://developer.apple.com/xcode/) first, then:

```shell
xcodebuild -downloadComponent MetalToolchain
```

## Windows

Additional prerequisites:

- [Visual Studio 2022](https://visualstudio.microsoft.com/downloads/) including the Native Desktop Workload
- (Optional) AMD GPU support
    - [ROCm](https://rocm.docs.amd.com/en/latest/)
- (Optional) NVIDIA GPU support
    - [CUDA SDK](https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64&target_type=exe_network)
- (Optional) Vulkan GPU support
    - [Vulkan SDK](https://vulkan.lunarg.com/sdk/home) - useful for AMD/Intel GPUs
- (Optional) MLX engine support
    - [CUDA 13+ SDK](https://developer.nvidia.com/cuda-downloads)
    - [cuDNN 9+](https://developer.nvidia.com/cudnn)

For Ninja builds, run CMake from a Developer PowerShell/Command Prompt or another shell where the Visual Studio compiler is available.

> Building for Vulkan requires VULKAN_SDK environment variable:
> 
> PowerShell
> ```powershell
> $env:VULKAN_SDK="C:\VulkanSDK\<version>"
> ```
> CMD
> ```cmd
> set VULKAN_SDK=C:\VulkanSDK\<version>
> ```

## Windows (ARM)

Windows ARM does not support additional acceleration libraries at this time.

## Linux

Additional prerequisites:

- (Optional) AMD GPU support
    - [ROCm](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/quick-start.html)
- (Optional) NVIDIA GPU support
    - [CUDA SDK](https://developer.nvidia.com/cuda-downloads)
- (Optional) Vulkan GPU support
    - [Vulkan SDK](https://vulkan.lunarg.com/sdk/home) - useful for AMD/Intel GPUs
    - Or install via package manager: `sudo apt install vulkan-sdk` (Ubuntu/Debian) or `sudo dnf install vulkan-sdk` (Fedora/CentOS)
- (Optional) MLX engine support
    - [CUDA 13+ SDK](https://developer.nvidia.com/cuda-downloads)
    - [cuDNN 9+](https://developer.nvidia.com/cudnn)
    - OpenBLAS/LAPACK: `sudo apt install libopenblas-dev liblapack-dev liblapacke-dev` (Ubuntu/Debian)
> [!IMPORTANT]
> Ensure prerequisites are in `PATH` before running CMake.

## MLX Engine (Optional)

The MLX engine enables running safetensor based models. On macOS arm64, MLX is enabled by default. On other platforms, MLX backends are selected with `OLLAMA_MLX_BACKENDS`.

### CUDA

Requires CUDA 13+ and [cuDNN](https://developer.nvidia.com/cudnn) 9+.

```shell
cmake -B build . -DOLLAMA_MLX_BACKENDS=cuda_v13
cmake --build build --parallel 8
```

### Local MLX source overrides

To build against a local checkout of MLX and/or MLX-C (useful for development), set environment variables before running CMake:

```shell
export OLLAMA_MLX_SOURCE=/path/to/mlx
export OLLAMA_MLX_C_SOURCE=/path/to/mlx-c
```

On macOS arm64:

```shell
OLLAMA_MLX_SOURCE=../mlx OLLAMA_MLX_C_SOURCE=../mlx-c cmake -B build .
cmake --build build --parallel 8
```

For CUDA:

```powershell
$env:OLLAMA_MLX_SOURCE="../mlx"
$env:OLLAMA_MLX_C_SOURCE="../mlx-c"
cmake -B build . -DOLLAMA_MLX_BACKENDS=cuda_v13
cmake --build build --parallel 8
```

## Docker

```shell
docker build .
```

### ROCm

```shell
docker build --build-arg FLAVOR=rocm .
```

## Running tests

To run tests, use `go test`:

```shell
go test ./...
```

## Library detection

Ollama looks for native helper binaries and acceleration libraries in installed and local development layouts:

* `../lib/ollama` for standard installs where `ollama` is under `bin/`
* `./lib/ollama` for Windows release-style payloads and local dist output
* `.` for macOS release artifacts that colocate helpers with `ollama`
* `build/lib/ollama` and `dist/<platform>/lib/ollama` for local development builds

If the libraries are not found, Ollama will not run with any acceleration libraries.
