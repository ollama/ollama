# Build Troubleshooting Guide — ROCm 7.x / gfx1201

## Issue: rocWMMA FATTN compiler crash (exit code 1, no error message)

**Symptoms:**
- Build fails on `fattn-mma-f16-instance-*.cu` files
- Compiler exits with code 1 but prints no error
- System has <32GB RAM
- 568+ warnings before crash

**Root Cause:**
rocWMMA Flash Attention templates are massive template expansions. Each `.cu` file
needs ~8GB RAM to compile. With `-j4` or higher, multiple instances compete for RAM
and the OOM killer silently terminates the compiler.

**Solutions (in order):**

### 1. Use Ninja with serial pool (RECOMMENDED)
```bash
cmake -B build -G Ninja \
  -DGGML_HIP=ON \
  -DGGML_HIP_ROCWMMA_FATTN=ON \
  -DGGML_TURBOQUANT=ON \
  -DAMDGPU_TARGETS=gfx1201

# The ggml_fattn_pool.cmake automatically creates a "fattn_serial" pool
# with depth=1, so FATTN files compile one at a time
ninja -C build
```

### 2. Use the build script with auto-detection
```bash
chmod +x build_ollama_rocm.sh
./build_ollama_rocm.sh
```
This script:
- Detects available RAM
- Uses serial compilation if <32GB
- Retries with `-j2` if parallel fails
- Falls back to `-j1` if needed
- Disables FATTN only as last resort

### 3. Manual serial compilation
```bash
# Configure with serial flags
cmake -B build \
  -DGGML_HIP=ON \
  -DGGML_HIP_ROCWMMA_FATTN=ON \
  -DCMAKE_HIP_FLAGS="-parallel-jobs=1 --amdgpu-unroll-threshold-local=500"

# Build with single job
cmake --build build --parallel 1
```

### 4. Disable FATTN (fallback — still gets all other optimizations)
```bash
cmake -B build \
  -DGGML_HIP=ON \
  -DGGML_HIP_ROCWMMA_FATTN=OFF \
  -DGGML_HIP_GRAPHS=ON \
  -DGGML_TURBOQUANT=ON \
  -DAMDGPU_TARGETS=gfx1201

cmake --build build --parallel
```
You still get:
- TurboQuant KV cache compression
- HIP Graphs execution
- Split-K matmul
- Persistent batching
- All compiler optimizations

Only missing: the ~65% prompt processing boost from rocWMMA flash attention.

## Issue: "undefined reference to turboquant_*" linker errors

**Fix:** Ensure `libggml_turboquant.so` is built and linked:
```bash
./build_all.sh  # builds and installs the extension libraries
```

Or manually:
```bash
hipcc -O3 -fPIC -shared -o libggml_turboquant.so ggml_turboquant.cpp
export LD_LIBRARY_PATH=$(pwd):$LD_LIBRARY_PATH
```

## Issue: "HIP error 700" (illegal address) at runtime

**Causes:**
- Paged KV page size mismatch
- RoPE cache head_dim mismatch
- TurboQuant block size mismatch

**Fix:** Ensure all `TURBOQUANT_BLOCK_SIZE`, `KV_PAGE_SIZE`, and `head_dim`
values match between the extension library and the patched llama.cpp.

## Issue: Vulkan extensions not loading

**Fix:**
```bash
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
# Or wherever libvulkan.so lives on your system
```

## Issue: CMake can't find HIP

**Fix:**
```bash
export HIP_PATH=/opt/rocm
export PATH=$HIP_PATH/bin:$PATH
cmake -B build -DCMAKE_PREFIX_PATH=$HIP_PATH ...
```
