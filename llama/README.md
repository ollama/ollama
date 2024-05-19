# `llama`

This package integrates llama.cpp as a Go package that's easy to build with different tags for different CPU and GPU processors.

- [x] CPU
- [x] avx, avx2
- [ ] avx512
- [x] macOS Metal
- [x] Windows CUDA
- [x] Windows ROCm
- [ ] Linux CUDA
- [ ] Linux ROCm

Extra build steps are required for CUDA and ROCm on Windows since `nvcc` and `hipcc` both require using msvc as the host compiler. For these small dlls are created:

- `ggml-cuda.dll`
- `ggml-hipblas.dll`

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
go env -w "CGO_CFLAGS_ALLOW=-mfma"
go env -w "CGO_CXXFLAGS_ALLOW=-mfma"
go build -tags=avx2 .
```

### CUDA

Install the [CUDA toolkit v11.3.1](https://developer.nvidia.com/cuda-11-3-1-download-archive) then build ggml-cuda:

Build `ggml-cuda.dll`:

```shell
./cuda.sh
```

Then build the package with the `cuda` tag:

```shell
go build -tags=cuda .
```

### ROCm

Install [ROCm 5.7.1](https://rocm.docs.amd.com/en/docs-5.7.1/) and [Strawberry Perl](https://strawberryperl.com/):

Build `ggml-hipblas.dll`:

```shell
./hipblas.sh
```

Then build the package with the `rocm` tag:

```shell
go build -tags=rocm .
```

## Syncing with llama.cpp

To update this package to the latest llama.cpp code, use the `sync.sh` script.

```
./sync.sh ../../llama.cpp
```
