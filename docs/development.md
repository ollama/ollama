# Development

Install prerequisites:

- [Go](https://go.dev/doc/install)
- C/C++ Compiler e.g. Clang on macOS, [TDM-GCC](https://jmeubank.github.io/tdm-gcc/download/) on x86_64 or [llvm-mingw](https://github.com/mstorsjo/llvm-mingw) on ARM on Windows, GCC/Clang on Linux

Then build Ollama from the root directory of the repository:

```
go run . serve
```

## macOS (Apple Silicon)

macOS Apple Silicon supports Metal which is built-in to the Ollama binary.

## macOS (Intel)

Install prerequisites:

- [CMake](https://cmake.org/download/) or `brew install cmake`

Then, configure and build the project:

```
cmake -B build
cmake --build build
```

Lastly, run Ollama:

```
go run . serve
```

## Windows

Install prerequisites:

- [CMake](https://cmake.org/download/)
- [Visual Studio 2022](https://visualstudio.microsoft.com/downloads/) including the Native Desktop Workload
- (Optional) AMD GPU support
    - [ROCm](https://rocm.github.io/install.html)
    - [Ninja](https://github.com/ninja-build/ninja/releases)
- (Optional) NVIDIA GPU support
    - [CUDA SDK](https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_network)

> [!IMPORTANT]
> Ensure prerequisites are in `PATH` before running CMake.

> [!IMPORTANT]
> ROCm is not compatible with Visual Studio CMake generators. Use `-GNinja` when configuring the project.

> [!IMPORTANT]
> CUDA is only compatible with Visual Studio CMake generators.

Then, configure and build the project:

```
cmake -B build
cmake --build build --config Release
```

Lastly, run Ollama:

```
go run . serve
```

## Windows (ARM)

Windows ARM does not support additional acceleration libraries at this time.

## Linux

Install prerequisites:

- [CMake](https://cmake.org/download/) or `sudo apt install cmake` or `sudo dnf install cmake`
- (Optional) AMD GPU support
    - [ROCm](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/quick-start.html)
- (Optional) NVIDIA GPU support
    - [CUDA SDK](https://developer.nvidia.com/cuda-downloads)

> [!IMPORTANT]
> Ensure prerequisites are in `PATH` before running CMake.


Then, configure and build the project:

```
cmake -B build
cmake --build build
```

Lastly, run Ollama:

```
go run . serve
```

## Docker

```
docker build .
```

### ROCm

```
docker build --build-arg FLAVOR=rocm .
```

## Running tests

To run tests, use `go test`:

```
go test ./...
```
