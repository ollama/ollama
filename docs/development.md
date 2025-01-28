# Development

## macOS

Install [Go](https://go.dev/doc/install), then build Ollama from the root directory of the repository:

```
go run . serve
```

> To silence the `ld: warning: ignoring duplicate libraries: '-lobjc'` [warning](https://github.com/golang/go/issues/67799), use `export CGO_LDFLAGS="-Wl,-no_warn_duplicate_libraries"`

### Intel CPU acceleration

To build CPU acceleration libraries for older Macs with Intel CPU, first install CMake:

```
brew install cmake
```

Then, build the CPU acceleration libraries:

```
cmake -B build
cmake --build build
```

Now, run Ollama, and the acceleration libraries will be loaded automatically:

```
go run . serve
```

## Windows

Install [Go](https://go.dev/doc/install) and a version of [MinGW](https://jmeubank.github.io/tdm-gcc/download/). Then run Ollama:

```
go run . serve
```

### Hardware acceleration

Install prerequisites:

- [CMake](https://cmake.org/download/)
- [Visual Studio 2022](https://visualstudio.microsoft.com/downloads/) including the Native Desktop Workload

Next, build the acceleration libraries:

```
cmake -B build
cmake --build build --config Release
```

Lastly, run Ollama:

```
go run . serve
```

#### CUDA

Install the [CUDA SDK](https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_network)

Then, rebuild the GPU libraries:

```
rm -r build
cmake -B build
cmake --build build
```

and finally run Ollama:

```
go run . serve
```

#### ROCm

Install the prerequisites:

- [HIP SDK](https://www.amd.com/en/developer/resources/rocm-hub/hip-sdk.html)
- [Ninja](https://github.com/ninja-build/ninja/releases)

Then, build the GPU libraries:

```
rm -r build
cmake --preset ROCm -G Ninja
cmake --build build
```

Finally, run Ollama:

```
go run . serve
```

## Windows (ARM)

Install the arm64 version of [Go](https://go.dev/dl/) and [llvm-mingw](https://github.com/mstorsjo/llvm-mingw) (make sure to add its `bin` directory to the `PATH`).

Then, build and run Ollama:

```
go run . serve
```

## Linux

Install `gcc`, `g++` and [Go](https://go.dev/doc/install):

```
sudo apt-get install gcc g++ build-essential
```

Then, run Ollama:

```
go run . serve
```

### Hardware acceleration

Install prerequisites:

```
sudo apt-get install cmake
```

Then, build the acceleration libraries:

```
cmake -B build
cmake --build build
```

Lastly, run Ollama:

```
go run . serve
```

#### CUDA

Install the [CUDA SDK](https://developer.nvidia.com/cuda-downloads), and CUDA to `PATH`:

```
export PATH="$PATH:/usr/local/cuda/bin"
```

Then, build the acceleration libraries:

```
rm -r build
cmake -B build
cmake --build build
```

Finally, build and run Ollama:

```
go run . serve
```

#### ROCm

Install [ROCm](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/quick-start.html), then run:

```
rm -r build
cmake -B build
cmake --build build
```

After building, run Ollama:

```
go run . serve
```

## Docker

```
docker build .
```

### ROCm

```
FLAVOR=rocm docker build .
```

## Running tests

To run tests, use `go test`:

```
go test ./...
```
