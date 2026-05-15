# Development

Install prerequisites:

- [Go](https://go.dev/doc/install)
- [CMake](https://cmake.org/download/) 3.24 or newer
- C/C++ compiler: Clang on macOS, Visual Studio 2022 C++ tools on Windows, or GCC/Clang on Linux
- [Ninja](https://github.com/ninja-build/ninja/releases) in `PATH` is recommended, especially on Windows

For pure Go iteration against an existing native payload, run Ollama from the repository root:

```shell
go run . serve
```

> [!NOTE]
> Ollama includes native code compiled with CGO.  From time to time these data structures can change and CGO can get out of sync resulting in unexpected crashes.  You can force a full build of the native code by running `go clean -cache` first. 

## Native build model

For a fresh checkout, or after changing native code, build from the repository root. Use an explicit job count instead of bare `--parallel`; increase `4` only if the machine has enough CPU and memory headroom.

```shell
cmake -B build .
cmake --build build --parallel 4
./ollama serve
```

To build `llama-server` GPU backends through the same root build, select the backend and target explicitly:

```shell
cmake -B build-gpu . -DOLLAMA_LLAMA_SERVER_BACKENDS=vulkan
cmake --build build-gpu --target ollama-llama-server-vulkan --parallel 4
```

Supported backend values are `cuda-v12`, `cuda-v13`, `cuda-v13-windows`, `rocm`, `rocm-windows`, `vulkan`, `jetpack5`, and `jetpack6`.

Use standard CMake architecture overrides to narrow GPU builds for local hardware:

```shell
# CUDA
cmake -B build-gpu . -DOLLAMA_LLAMA_SERVER_BACKENDS=cuda-v13 -DCMAKE_CUDA_ARCHITECTURES=native

# ROCm / HIP
cmake -B build-gpu . -DOLLAMA_LLAMA_SERVER_BACKENDS=rocm -DCMAKE_HIP_ARCHITECTURES=gfx1100
```

`AMDGPU_TARGETS` is also accepted for ROCm when matching llama.cpp-specific target strings is necessary.

## macOS (Apple Silicon)

macOS Apple Silicon supports Metal for local native builds. For a release-style payload:

```shell
./scripts/build_darwin.sh -a arm64
```

## macOS (Intel)

Install prerequisites:

- [CMake](https://cmake.org/download/) or `brew install cmake`

Then build the Darwin payload:

```shell
./scripts/build_darwin.sh -a amd64
```

Lastly, run Ollama:

```shell
dist/darwin-amd64/ollama serve
```

## Windows

Install prerequisites:

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

Then build a minimal CPU payload and Go binary:

```powershell
.\scripts\build_windows.ps1 cpu ollama
```

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

> [!IMPORTANT]
> Prefer the build script for release-style GPU payloads. It wires the platform-specific compiler, SDK, and install layout details.

Lastly, run Ollama:

```powershell
.\dist\windows-amd64\ollama.exe serve
```

For native CMake iteration, use the repository-root CMake build shown in [Native build model](#native-build-model). Ninja is recommended when available.

## Windows (ARM)

Windows ARM does not support additional acceleration libraries at this time. The Windows build script can cross-compile the CPU llama-server payload when the ARM64 cross-compile toolchain is installed; otherwise it skips that payload for local developer builds.

## Linux

Install prerequisites:

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


For a release-style Linux payload, use the Docker-backed build script:

```shell
./scripts/build_linux.sh
```

For native CMake iteration, use the repository-root CMake build shown in [Native build model](#native-build-model).

## MLX Engine (Optional)

The MLX engine enables running safetensor based models. It requires building the [MLX](https://github.com/ml-explore/mlx) and [MLX-C](https://github.com/ml-explore/mlx-c) shared libraries via the repository-root CMake presets. The root project delegates MLX-specific rules to `cmake/mlx`. On macOS, MLX leverages the Metal library to run on the GPU, and on Windows and Linux, runs on NVIDIA GPUs via CUDA v13.

### macOS (Apple Silicon)

Requires the Metal toolchain. Install [Xcode](https://developer.apple.com/xcode/) first, then:

```shell
xcodebuild -downloadComponent MetalToolchain
```

Verify it's installed correctly (should print "no input files"):

```shell
xcrun metal
```

Then build:

```shell
cmake --preset MLX
cmake --build --preset MLX --parallel 4
cmake --install build --component MLX
cmake --install build --component MLX_VENDOR
```

> [!NOTE]
> Without the Metal toolchain, cmake will silently complete with Metal disabled. Check the cmake output for `Setting MLX_BUILD_METAL=OFF` which indicates the toolchain is missing.

### Windows / Linux (CUDA)

Requires CUDA 13+ and [cuDNN](https://developer.nvidia.com/cudnn) 9+.

```shell
cmake --preset "MLX CUDA 13"
cmake --build --preset "MLX CUDA 13" --parallel 4
cmake --install build --component MLX --strip
cmake --install build --component MLX_VENDOR
```

### Local MLX source overrides

To build against a local checkout of MLX and/or MLX-C (useful for development), set environment variables before running CMake:

```shell
export OLLAMA_MLX_SOURCE=/path/to/mlx
export OLLAMA_MLX_C_SOURCE=/path/to/mlx-c
```

For example, using the helper scripts with local mlx and mlx-c repos:
```shell
OLLAMA_MLX_SOURCE=../mlx OLLAMA_MLX_C_SOURCE=../mlx-c ./scripts/build_linux.sh

OLLAMA_MLX_SOURCE=../mlx OLLAMA_MLX_C_SOURCE=../mlx-c ./scripts/build_darwin.sh
```

```powershell
$env:OLLAMA_MLX_SOURCE="../mlx"
$env:OLLAMA_MLX_C_SOURCE="../mlx-c"
./scripts/build_windows.ps1
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
