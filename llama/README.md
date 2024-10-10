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

- `ggml_cuda.dll` on Windows or `libggml_cuda.so` on Linux
- `ggml_hipblas.dll` on Windows or `libggml_hipblas.so` on Linux

> Note: it's important that memory is allocated and freed by the same compiler (e.g. entirely by code compiled with msvc or mingw). Issues from this should be rare, but there are some places where pointers are returned by the CUDA or HIP runtimes and freed elsewhere, causing a crash. In a future change the same runtime should be used in both cases to avoid crashes.

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

To update this package to the latest llama.cpp code, use the `sync.sh` script.

> Note: the upstream commit is defined by HEAD in ./llm/llama.cpp today, but will be adjusted to a manifest file soon.  You must update this before running the sync script.

```
./sync.sh 
```

When updating, sometimes the existing patches wont apply cleanly due to upstream changes.  The general sequence to rebase a patch is:

1. Rename the offending `./patches/NN-xxx.diff` patch so it doesn't end with ".diff" temporarily which will result in the sync script ignoring it
2. Run the sync script so it completes successfully (disable all patches necessary for a clean run)
3. Manually apply the patch (e.g. `patch -p1 < ./patches/NN-xxx.tmp`) and resolve the conflicts
4. Build and test without committing the changes in the native code
5. Generate a replacement diff `git diff file1.cpp file2.cpp > ./patches/NN-xxx.diff`
6. Repeat for additional patches until all are cleaned up
7. Re-run the sync script to verify everything worked correctly.

## Modifying the native code

If you are fixing a bug or adding a capability which impacts the vendored code from llama.cpp, you'll need to generate a patch for the native code.  Changes should be contributed upstream where possible to reduce the number of patches we carry.  The general approach is:

1. Edit the relevant native code, without committing the changes
2. Build and test until the code is working properly and ready for an ollama PR
3. Generate a new diff, with a prefix number larger than the existing numbers. e.g. `git diff file1.ccp file2.cpp > ./patches/42-something.diff`
4. Commit the changes to the native files, but don't let them drift from the patch content
5. Post the PR to Ollama - CI will verify the patch reproduces the same content as what is committed to the file to ensure all changes have patches.