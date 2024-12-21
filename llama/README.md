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
go env -w "CGO_CPPFLAGS_ALLOW=-mfma|-mf16c"
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

Install [ROCm](https://rocm.docs.amd.com/en/latest/).

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

Install [ROCm](https://rocm.docs.amd.com/en/latest/).

```shell
make ggml_hipblas.dll
go build -tags avx,rocm .
```

## Building runners

```shell
# build all runners for this platform
make -j
```

## Vendoring

Ollama currently vendors [llama.cpp](https://github.com/ggerganov/llama.cpp/) and [ggml](https://github.com/ggerganov/ggml) through a vendoring model. While we generally strive to contribute changes back upstream to avoid drift, we cary a small set of patches which are applied to the tracking commit. A set of make targets are available to aid developers in updating to a newer tracking commit, or to work on changes.

If you update the vendoring code, start by running the following command to establish the tracking llama.cpp repo in the `./vendor/` directory.

```
make apply-patches
```

### Updating Base Commit

**Pin to new base commit**

To update to a newer base commit, select the upstream git tag or commit and update `llama/vendoring`

#### Applying patches

When updating to a newer base commit, the existing patches may not apply cleanly and require manual merge resolution.

Start by applying the patches. If any of the patches have conflicts, the `git am` will stop at the first failure.

```
make apply-patches
```

If you see an error message about a conflict, go into the `./vendor/` directory, and perform merge resolution using your preferred tool to the patch commit which failed. Save the file(s) and continue the patch series with `git am --continue` . If any additional patches fail, follow the same pattern until the full patch series is applied. Once finished, run a final `create-patches` and `sync` target to ensure everything is updated.

```
make create-patches sync
```

Build and test Ollama, and make any necessary changes to the Go code based on the new base commit. Submit your PR to the Ollama repo.

### Generating Patches

When working on new fixes or features that impact vendored code, use the following model. First get a clean tracking repo with all current patches applied:

```
make apply-patches
```

Now edit the upstream native code in the `./vendor/` directory. You do not need to commit every change in order to build, a dirty working tree in the tracking repo is OK while developing. Simply save in your editor, and run the following to refresh the vendored code with your changes, build the backend(s) and build ollama:

```
make sync
make -j 8
go build .
```

> [!IMPORTANT]
> Do **NOT** run `apply-patches` while you're iterating as that will reset the tracking repo. It will detect a dirty tree and abort, but if your tree is clean and you accidentally ran this target, use `git reflog` to recover your commit(s).

Iterate until you're ready to submit PRs. Once your code is ready, commit a change in the `./vendor/` directory, then generate the patches for ollama with

```
make create-patches
```

> [!IMPORTANT]
> Once you have completed this step, it is safe to run `apply-patches` since your change is preserved in the patches.

In your `./vendor/` directory, create a branch, and cherry-pick the new commit to that branch, then submit a PR upstream to llama.cpp.

Commit the changes in the ollama repo and submit a PR to Ollama, which will include the vendored code update with your change, along with the patches.

After your PR upstream is merged, follow the **Updating Base Commit** instructions above, however first remove your patch before running `apply-patches` since the new base commit contains your change already.
