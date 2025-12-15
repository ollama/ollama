# Experimental ROCm iGPU Support

This branch adds a ROCm backend path geared toward AMD APUs that only expose a small VRAM aperture but share a large UMA pool with the CPU. The steps below outline how to reproduce the build and how to run Ollama with the staged ROCm runtime.

> **Warning**
> Upstream ROCm does not officially support these APUs yet. Expect driver updates, kernel parameters, or environment variables such as `HSA_OVERRIDE_GFX_VERSION` to change between releases.

## 1. Stage the ROCm runtime

We avoid touching the system installation by unpacking the required RPMs into `build/rocm-stage`.

```bash
mkdir -p build/rocm-stage build/rpm-tmp
cd build/rpm-tmp
dnf download \
  hipblas hipblas-devel hipblas-common-devel \
  rocblas rocblas-devel \
  rocsolver rocsolver-devel \
  rocm-hip-devel rocm-device-libs rocm-comgr rocm-comgr-devel

cd ../rocm-stage
for rpm in ../rpm-tmp/*.rpm; do
  echo "extracting ${rpm}"
  rpm2cpio "${rpm}" | bsdtar -xf -
done
```

Important staged paths after extraction:

| Purpose                  | Location                                        |
| ------------------------ | ----------------------------------------------- |
| HIP/rocBLAS libraries    | `build/rocm-stage/lib64`                        |
| Tensile kernels (rocBLAS)| `build/rocm-stage/lib64/rocblas/library`        |
| Headers (`hip`, `rocblas`)| `build/rocm-stage/include`                     |

## 2. Build the ROCm backend

Configure CMake with the preset that targets ROCm 6.x and point it at the staged HIP compiler:

```bash
cmake --preset "ROCm 6" -B build/rocm \
  -DGGML_VULKAN=OFF \
  -DCMAKE_INSTALL_PREFIX=/usr/local \
  -DCMAKE_HIP_COMPILER=/usr/bin/hipcc \
  -DCMAKE_PREFIX_PATH="$PWD/build/rocm-stage"

cmake --build build/rocm --target ggml-hip -j$(nproc)
```

Artifacts land in `build/lib/ollama/rocm` (and mirrored in `dist/lib/ollama/rocm` when packaging). These include `libggml-hip.so`, CPU fallback variants, Vulkan, and `librocsolver.so`.

## 3. Run Ollama on ROCm

The runner needs to see both the GGML plugins and the staged ROCm runtime. The following environment block works for an AMD Radeon 760M with a UMA carve-out:

```bash
export BASE=$HOME/ollama-gpu
export OLLAMA_LIBRARY_PATH=$BASE/build/lib/ollama/rocm:$BASE/build/lib/ollama
export LD_LIBRARY_PATH=$OLLAMA_LIBRARY_PATH:$BASE/build/rocm-stage/lib64:${LD_LIBRARY_PATH:-}
export ROCBLAS_TENSILE_LIBPATH=$BASE/build/rocm-stage/lib64/rocblas/library
export ROCBLAS_TENSILE_PATH=$ROCBLAS_TENSILE_LIBPATH

export HSA_OVERRIDE_GFX_VERSION=11.0.0   # spoof gfx1100 for Phoenix
export GGML_HIP_FORCE_GTT=1             # force GTT allocations for UMA memory
export OLLAMA_GPU_DRIVER=rocm
export OLLAMA_GPU=100                   # opt into GPU-only scheduling
export OLLAMA_LLM_LIBRARY=rocm          # skip CUDA/Vulkan discovery noise
export OLLAMA_VULKAN=0                  # optional: suppress Vulkan backend

$BASE/build/ollama serve
```

On launch you should see log lines similar to:

```
library=ROCm compute=gfx1100 name=ROCm0 description="AMD Radeon 760M Graphics"
ggml_hip_get_device_memory using GTT memory for 0000:0e:00.0 (total=16352354304 free=15034097664)
```

If the runner crashes before enumerating devices:

- Double-check that `ROCBLAS_TENSILE_LIBPATH` points to the staged `rocblas/library`.
- Ensure no other `LD_LIBRARY_PATH` entries override `libamdhip64.so`.
- Try unsetting `HSA_OVERRIDE_GFX_VERSION` to confirm whether the kernel patch is still needed on your system.

## 4. Sharing this build

- Keep the staged RPMs alongside the branch so others can reproduce the exact runtime.
- Include `/tmp/ollama_rocm_run.log` or similar discovery logs in issues/PRs to help maintainers understand the UMA setup.
- Mention any kernel parameters (e.g., large UMA buffer in firmware) when opening upstream tickets.

