# `llama`

<<<<<<< Updated upstream
This package integrates llama.cpp as a Go package that's easy to build with tags for different CPU and GPU processors.
=======
This package integrates the [llama.cpp](https://github.com/ggerganov/llama.cpp) library as a Go package and makes it easy to build it with tags for different CPU and GPU processors.
>>>>>>> Stashed changes

Supported:

- [x] CPU
- [x] avx, avx2
- [x] macOS Metal
- [x] Windows CUDA
- [x] Windows ROCm
- [x] Linux CUDA
- [x] Linux ROCm
- [x] Llava
<<<<<<< Updated upstream
=======
- [ ] Parallel Requests

Extra build steps are required for CUDA and ROCm on Windows since `nvcc` and `hipcc` both require using msvc as the host compiler. For these small dlls are created:

- `ggml-cuda.dll`
- `ggml-hipblas.dll`
>>>>>>> Stashed changes

> Note: it's important that memory is allocated and freed by the same compiler (e.g. entirely by code compiled with msvc or mingw). Issues from this should be rare, but there are some places where pointers are returned by the CUDA or HIP runtimes and freed elsewhere, causing a a crash. In a future change the same runtime should be used in both cases to avoid crashes.

## Building

```
go build .
```

### AVX

```shell
go build -tags avx .
```

### AVX2

```shell
# go doesn't recognize `-mfma` as a valid compiler flag
# see https://github.com/golang/go/issues/17895
go env -w "CGO_CFLAGS_ALLOW=-mfma|-mf16c"
go env -w "CGO_CXXFLAGS_ALLOW=-mfma|-mf16c"
go build -tags=avx,avx2 .
```

## Linux

### CUDA

<<<<<<< Updated upstream
Install the [CUDA toolkit v11.3.1](https://developer.nvidia.com/cuda-11-3-1-download-archive):
=======
Install the [CUDA toolkit v11.3.1](https://developer.nvidia.com/cuda-11-3-1-download-archive) then build `libggml-cuda.so`:

```shell
./build_cuda.sh
```
>>>>>>> Stashed changes

Then build the package with the `cuda` tag:

```shell
go build -tags=cuda .
```

## Windows

<<<<<<< Updated upstream
Download [w64devkit](https://github.com/skeeto/w64devkit/releases/latest) for a simple MinGW development environment.

=======
>>>>>>> Stashed changes
### CUDA

Install the [CUDA toolkit v11.3.1](https://developer.nvidia.com/cuda-11-3-1-download-archive) then build the cuda code:

Build `ggml-cuda.dll`:

```shell
<<<<<<< Updated upstream
make ggml_cuda.dll
=======
./build_cuda.ps1
>>>>>>> Stashed changes
```

Then build the package with the `cuda` tag:

```shell
go build -tags=cuda .
```

### ROCm

<<<<<<< Updated upstream
Install [ROCm 5.7.1](https://rocm.docs.amd.com/en/docs-5.7.1/).

```shell
make ggml_hipblas.dll
=======
Install [ROCm 5.7.1](https://rocm.docs.amd.com/en/docs-5.7.1/) and [Strawberry Perl](https://strawberryperl.com/).

Then, build `ggml-hipblas.dll`:

```shell
./build_hipblas.sh
>>>>>>> Stashed changes
```

Then build the package with the `rocm` tag:

```shell
go build -tags=rocm .
```

## Syncing with llama.cpp

<<<<<<< Updated upstream
To update this package to the latest llama.cpp code, use the `sync_llama.sh` script from the root of this repo:

```
./sync_llama.sh ../../llama.cpp
=======
To update this package to the latest llama.cpp code, use the `scripts/sync_llama.sh` script from the root of this repo, providing the location of a llama.cpp checkout:

```
cd ollama
./scripts/sync_llama.sh ../llama.cpp
>>>>>>> Stashed changes
```
