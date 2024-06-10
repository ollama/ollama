# `llama`

This package integrates llama.cpp as a Go package that's easy to build with tags for different CPU and GPU processors.

Supported:

- [x] CPU
- [x] avx, avx2
- [x] macOS Metal
- [x] Windows CUDA
- [x] Windows ROCm
- [x] Linux CUDA
- [x] Linux ROCm
- [x] Llava

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

Then build the package with the `cuda` tag:

```shell
go build -tags=cuda .
```

## Windows

Download [w64devkit](https://github.com/skeeto/w64devkit/releases/latest) for a simple MinGW development environment.

### CUDA

Install the [CUDA toolkit v11.3.1](https://developer.nvidia.com/cuda-11-3-1-download-archive) then build the cuda code:

Build `ggml-cuda.dll`:

```shell
make ggml_cuda.dll
```

Then build the package with the `cuda` tag:

```shell
go build -tags=cuda .
```

### ROCm

Install [ROCm 5.7.1](https://rocm.docs.amd.com/en/docs-5.7.1/) and [Strawberry Perl](https://strawberryperl.com/).

```shell
make ggml_hipblas.dll
```

Then build the package with the `rocm` tag:

```shell
go build -tags=rocm .
```

## Syncing with llama.cpp

To update this package to the latest llama.cpp code, use the `sync_llama.sh` script from the root of this repo:

```
./sync_llama.sh ../../llama.cpp
```
