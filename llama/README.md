# `llama`

This package integrates the [llama.cpp](https://github.com/ggerganov/llama.cpp) library as a Go package and makes it easy to build it with tags for different CPU and GPU processors.

Supported:

- [x] CPU
- [x] avx, avx2
- [x] macOS Metal
- [x] Windows CUDA
- [x] Windows ROCm
- [x] Linux CUDA
- [x] Linux ROCm
- [x] Llava

Extra build steps are required for CUDA and ROCm on Windows since `nvcc` and `hipcc` both require using msvc as the host compiler. For these shared libraries are created:

- `ggml_cuda.dll` on Windows or `ggml_cuda.so` on Linux
- `ggml_hipblas.dll` on Windows or `ggml_hipblas.so` on Linux

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

Install the [CUDA toolkit v11.3.1](https://developer.nvidia.com/cuda-11-3-1-download-archive):

```shell
make ggml_cuda.so
go build -tags avx,cuda .
```

### ROCm

Install the [CUDA toolkit v11.3.1](https://developer.nvidia.com/cuda-11-3-1-download-archive):

```shell
make ggml_hipblas.so
go build -tags avx,rocm .
```

## Windows

Download [w64devkit](https://github.com/skeeto/w64devkit/releases/latest) for a simple MinGW development environment.

### CUDA

Install the [CUDA toolkit v11.3.1](https://developer.nvidia.com/cuda-11-3-1-download-archive) then build the cuda code:

```shell
make ggml_cuda.dll
go build -tags avx,cuda .
```

### ROCm

Install [ROCm 5.7.1](https://rocm.docs.amd.com/en/docs-5.7.1/).

```shell
make ggml_hipblas.dll
go build -tags avx,rocm .
```

## Building runners

```shell
# build all runners for this platform
make -j
```

## Syncing with llama.cpp

To update this package to the latest llama.cpp code, use the `sync.sh` script:

```
./sync.sh ../../llama.cpp
```
