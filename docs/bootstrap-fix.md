# s390x CMake Flag Support

## Summary

`CMakeLists.txt`, `cmake/local.cmake`, and `llama/server/CMakeLists.txt` had no
awareness of the IBM Z / LinuxONE accelerator stack. The build would silently
produce a generic CPU binary with no SIMD, no BLAS, and no zDNN support even
when running on z-series hardware.

The changes described here detect the `s390x` processor at configure time and
wire up the correct GGML flags for each IBM accelerator tier.

---

### What Changed

#### [`CMakeLists.txt`](../CMakeLists.txt)

Three new top-level options replace the old single `OLLAMA_S390X_BIGENDIAN`
block. All three auto-default based on `CMAKE_SYSTEM_PROCESSOR`:

| Option | Default on s390x | Maps to |
|---|---|---|
| `OLLAMA_S390X_BIGENDIAN` | `ON` | GGUF big-endian byte-swap (existing) |
| `OLLAMA_S390X_VXE` | `ON` | `GGML_VXE` — VX/VXE/VXE2 SIMD (z15+) |
| `OLLAMA_S390X_ZDNN` | `OFF` | `GGML_ZDNN` — zDNN/zAIU co-processor (z17+, opt-in) |

On detection, cmake now prints:

```
-- s390x target detected
--   OLLAMA_S390X_BIGENDIAN = ON
--   OLLAMA_S390X_VXE       = ON  (GGML_VXE; z15+ SIMD)
--   OLLAMA_S390X_ZDNN      = OFF  (GGML_ZDNN; z17+ zAIU co-processor)
```

#### [`cmake/local.cmake`](../cmake/local.cmake)

Two areas changed:

1. **Flag forwarding** — the `_cmake_args` block inside
   `ollama_add_llama_server_build()` is now split into named groups:
   - `_cmake_args_paths` — source/install paths
   - `_cmake_args_s390x` — all three `OLLAMA_S390X_*` flags forwarded verbatim
   - `_cmake_args_ggml_base` — superbuild-level `GGML_NATIVE=OFF`, `GGML_OPENMP=OFF`

2. **s390x CPU branch** — the `_cpu_args` platform selection is now a proper
   three-way `if/elseif/else`:
   - **macOS arm64** — unchanged
   - **s390x** — passes `GGML_VXE`, `GGML_ZDNN`, `GGML_BLAS=ON`,
     `GGML_BLAS_VENDOR=OpenBLAS` to the sub-project; emits a `WARNING` if
     OpenBLAS is missing; emits a `FATAL_ERROR` early if `GGML_ZDNN=ON` but
     `libzdnn` is not found on the host
   - **everything else** — unchanged

3. **Debuggability** — every `ExternalProject_Add` call for llama-server now
   prints the full resolved configure command and build directory at configure
   time, and adds step-separator banners to build/install output:

   ```
   -- [ollama-llama-server-local] configure command:
     /usr/bin/cmake -S ... -DGGML_VXE=ON -DGGML_BLAS=ON ...
   -- [ollama-llama-server-local] build dir: .../llama-server-local
   -- [ollama-llama-server-local] targets:   llama-server;llama-quantize
   ```

#### [`llama/server/CMakeLists.txt`](../llama/server/CMakeLists.txt)

Declares the same three `OLLAMA_S390X_*` options (with identical
processor-conditional defaults) and applies them to the GGML cache **before**
`FetchContent_MakeAvailable` so llama.cpp picks them up:

```cmake
set(GGML_VXE  ON  CACHE BOOL ... FORCE)   # when OLLAMA_S390X_VXE=ON
set(GGML_ZDNN ON  CACHE BOOL ... FORCE)   # when OLLAMA_S390X_ZDNN=ON
```

This ensures standalone `cmake --preset cpu_s390x` invocations work identically
to superbuild-driven ones.

#### [`llama/server/CMakePresets.json`](../llama/server/CMakePresets.json)

Four new presets added:

| Preset | Purpose |
|---|---|
| `cpu_s390x_base` *(hidden)* | Base: big-endian, `GGML_BLAS=OpenBLAS`, `GGML_CPU_ALL_VARIANTS=ON` |
| `cpu_s390x` | z15+ with VXE SIMD (`GGML_VXE=ON`) |
| `cpu_s390x_zdnn` | z17+ with VXE + zDNN (`GGML_VXE=ON`, `GGML_ZDNN=ON`) |
| `cpu_s390x_novxe` | Scalar-only debug / z14 (`GGML_VXE=OFF`) |

---

### Tests and Results

All tests were run inside the dev container
(`root@e876594f140c:/workspace/ollama-s390x`) on an arm64 host with
`-DCMAKE_SYSTEM_PROCESSOR=s390x` to simulate z-series detection without
requiring real IBM Z hardware. All four tests are configure-only (no compile).

---

#### Test 1 — Default s390x flags (VXE=ON, ZDNN=OFF, BIGENDIAN=ON)

```bash
cmake -S . -B /tmp/test-s390x-default \
  -DCMAKE_SYSTEM_PROCESSOR=s390x \
  -DOLLAMA_BUILD_GO=OFF \
  2>&1 | grep -E "s390x|VXE|ZDNN|BIGENDIAN|local build|configure command"
```

Output:

```
-- s390x target detected
--   OLLAMA_S390X_BIGENDIAN = ON
--   OLLAMA_S390X_VXE       = ON  (GGML_VXE; z15+ SIMD)
--   OLLAMA_S390X_ZDNN      = OFF  (GGML_ZDNN; z17+ zAIU co-processor)
-- [local build] platform: s390x / IBM Z & LinuxONE
--   SIMD (GGML_VXE)  : ON
--   zDNN (GGML_ZDNN) : OFF
  s390x build: OpenBLAS not found on this host.
    SIMD (VXE) acceleration depends on BLAS; performance will be degraded.
-- [ollama-llama-server-local] configure command:
  /usr/bin/cmake -S /workspace/ollama-s390x/llama/server -B <BINARY_DIR> \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/tmp/test-s390x-default \
    -DOLLAMA_LIB_DIR:STRING=lib/ollama \
    -DOLLAMA_RUNNER_DIR= \
    -DFETCHCONTENT_SOURCE_DIR_LLAMA_CPP=/tmp/test-s390x-default/_deps/llama_cpp-src \
    -DOLLAMA_LLAMA_CPP_SKIP_COMPAT_PATCH=ON \
    -DOLLAMA_S390X_BIGENDIAN=ON \
    -DOLLAMA_S390X_VXE=ON \
    -DOLLAMA_S390X_ZDNN=OFF \
    -DGGML_NATIVE=OFF \
    -DGGML_OPENMP=OFF \
    -DBUILD_SHARED_LIBS=ON \
    -DGGML_BACKEND_DL=ON \
    -DGGML_CPU_ALL_VARIANTS=ON \
    -DGGML_VXE=ON \
    -DGGML_ZDNN=OFF \
    -DGGML_BLAS=ON \
    -DGGML_BLAS_VENDOR=OpenBLAS
-- [ollama-llama-server-local] build dir: /tmp/test-s390x-default/llama-server-local
```

**Result: PASS** — `GGML_VXE=ON`, `GGML_BLAS=ON`, `OLLAMA_S390X_BIGENDIAN=ON`
all present in the forwarded configure command. OpenBLAS warning is expected
on a non-z dev machine.

---

#### Test 2 — Scalar-only build (VXE=OFF, simulates z14 / debug mode)

```bash
cmake -S . -B /tmp/test-s390x-novxe \
  -DCMAKE_SYSTEM_PROCESSOR=s390x \
  -DOLLAMA_S390X_VXE=OFF \
  -DOLLAMA_BUILD_GO=OFF \
  2>&1 | grep -E "VXE|ZDNN|local build"
```

Output:

```
--   OLLAMA_S390X_VXE       = OFF  (GGML_VXE; z15+ SIMD)
--   OLLAMA_S390X_ZDNN      = OFF  (GGML_ZDNN; z17+ zAIU co-processor)
-- [local build] platform: s390x / IBM Z & LinuxONE
--   SIMD (GGML_VXE)  : OFF
--   zDNN (GGML_ZDNN) : OFF
    SIMD (VXE) acceleration depends on BLAS; performance will be degraded.
  /usr/bin/cmake ... -DOLLAMA_S390X_VXE=OFF ... -DGGML_VXE=OFF ...
```

**Result: PASS** — `GGML_VXE=OFF` correctly forwarded when
`-DOLLAMA_S390X_VXE=OFF` is set.

---

#### Test 3 — zDNN requested without libzdnn installed (expected FATAL_ERROR)

```bash
cmake -S . -B /tmp/test-s390x-zdnn \
  -DCMAKE_SYSTEM_PROCESSOR=s390x \
  -DOLLAMA_S390X_ZDNN=ON \
  -DOLLAMA_BUILD_GO=OFF \
  2>&1 | grep -E "FATAL|zdnn|zDNN"
```

Output:

```
--   zDNN (GGML_ZDNN) : ON
  s390x build: GGML_ZDNN=ON but the IBM zDNN library (libzdnn) was not found.
    Install zDNN: https://github.com/IBM/zDNN
    Or disable zDNN acceleration with -DOLLAMA_S390X_ZDNN=OFF
See also "/tmp/test-s390x-zdnn/CMakeFiles/CMakeOutput.log".
See also "/tmp/test-s390x-zdnn/CMakeFiles/CMakeError.log".
```

**Result: PASS** — configure exits immediately with a clear, actionable error.
No cryptic linker failure deep in the llama.cpp sub-project.

---

#### Test 4 — Flag forwarding verification (full configure command inspection)

```bash
cmake -S . -B /tmp/test-s390x-fwd \
  -DCMAKE_SYSTEM_PROCESSOR=s390x \
  -DOLLAMA_BUILD_GO=OFF \
  2>&1 | grep -A2 "ollama-llama-server-local.*configure command"
```

Output:

```
-- [ollama-llama-server-local] configure command:
  /usr/bin/cmake -S /workspace/ollama-s390x/llama/server -B <BINARY_DIR> \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/tmp/test-s390x-fwd \
    -DOLLAMA_LIB_DIR:STRING=lib/ollama \
    -DOLLAMA_RUNNER_DIR= \
    -DFETCHCONTENT_SOURCE_DIR_LLAMA_CPP=/tmp/test-s390x-fwd/_deps/llama_cpp-src \
    -DOLLAMA_LLAMA_CPP_SKIP_COMPAT_PATCH=ON \
    -DOLLAMA_S390X_BIGENDIAN=ON \
    -DOLLAMA_S390X_VXE=ON \
    -DOLLAMA_S390X_ZDNN=OFF \
    -DGGML_NATIVE=OFF \
    -DGGML_OPENMP=OFF \
    -DBUILD_SHARED_LIBS=ON \
    -DGGML_BACKEND_DL=ON \
    -DGGML_CPU_ALL_VARIANTS=ON \
    -DGGML_VXE=ON \
    -DGGML_ZDNN=OFF \
    -DGGML_BLAS=ON \
    -DGGML_BLAS_VENDOR=OpenBLAS
-- [ollama-llama-server-local] build dir: /tmp/test-s390x-fwd/llama-server-local
```

**Result: PASS** — all three `OLLAMA_S390X_*` flags and their derived `GGML_*`
values are present and correctly valued in the forwarded command.

---

### Test Summary

| Test | Command | Expected | Result |
|---|---|---|---|
| Default s390x flags | `-DCMAKE_SYSTEM_PROCESSOR=s390x` | `GGML_VXE=ON`, `GGML_BLAS=ON`, `BIGENDIAN=ON` | ✅ PASS |
| Scalar-only (VXE=OFF) | `-DOLLAMA_S390X_VXE=OFF` | `GGML_VXE=OFF` forwarded | ✅ PASS |
| zDNN without libzdnn | `-DOLLAMA_S390X_ZDNN=ON` | `FATAL_ERROR` with install hint | ✅ PASS |
| Flag forwarding | inspect configure command | All flags reach llama/server | ✅ PASS |

> **Note:** Tests 1, 2, and 4 produce an OpenBLAS `WARNING` because
> `libopenblas-dev` is not installed in the dev container. This is expected
> and does not affect configure success. Install it on the real z-series build
> host with `apt install libopenblas-dev` before compiling.

---

## Bootstrap Dev Environment Fix

### Summary

Running `make run` on the z-Spyre runtime container and then executing the
bootstrap script produced two errors inside `ollama-dev`:

```
bash: git: command not found
```

```
Waiting for cache lock: Could not get lock /var/lib/dpkg/lock-frontend.
It is held by process 101 (apt-get)
```

---

## Root Cause

The original `bootstrap_dev_env.sh` wrote a `compose.yml` where the
`ollama-dev` service installed all packages — `git`, `cmake`, `gcc`, Go, etc.
— **inside the container's `command:` entrypoint**, which runs as a background
shell script every time the container starts:

```yaml
# Before (broken)
ollama-dev:
  image: docker.io/library/debian:bookworm
  command:
    - sh
    - -c
    - |
      apt-get update && apt-get install -y curl git cmake gcc g++ ... &&
      wget https://go.dev/dl/go1.22.5.linux-s390x.tar.gz ... &&
      sleep infinity
```

This caused two cascading problems:

### Problem 1 — `git: command not found`

When `podman compose exec ollama-dev bash` was run to enter the container, the
background `apt-get install` **had not finished yet**. The packages were still
being downloaded and installed, so `git` (and every other tool) was unavailable
until the entire install completed — which could take several minutes on a slow
network.

### Problem 2 — `apt` lock held by PID 101

Because the entrypoint's `apt-get install` was still running as PID 101, any
attempt to run `apt install git -y` interactively caused both processes to race
for `/var/lib/dpkg/lock-frontend`. The interactive `apt` command would block
indefinitely with:

```
Waiting for cache lock: Could not get lock /var/lib/dpkg/lock-frontend.
It is held by process 101 (apt-get)
```

---

## Fix

The fix moves all dependency installation out of the container entrypoint and
into a **`Dockerfile.dev`** that is generated alongside `compose.yml`. Package
installation now happens once at **image build time** (`podman compose up
--build`), not at container startup.

### What changed — [`scripts/bootstrap_dev_env.sh`](../scripts/bootstrap_dev_env.sh)

The `create_compose_file()` function was updated to:

1. **Detect the host architecture** (`s390x`, `x86_64`, `aarch64`) and map it
   to the correct Go archive suffix.
2. **Write a `Dockerfile.dev`** that bakes all dependencies into the image
   layer — no runtime `apt` calls:

```dockerfile
FROM docker.io/library/debian:bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl git vim htop ca-certificates wget tar \
        cmake ninja-build gcc g++ make pkg-config \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN wget -q https://go.dev/dl/go1.22.5.linux-s390x.tar.gz -O /tmp/go.tar.gz \
    && rm -rf /usr/local/go \
    && tar -C /usr/local -xzf /tmp/go.tar.gz \
    && rm /tmp/go.tar.gz

ENV PATH=$PATH:/usr/local/go/bin
CMD ["sleep", "infinity"]
```

3. **Switch the compose service** from `image:` to `build:` so Podman Compose
   builds the image from `Dockerfile.dev`:

```yaml
# After (fixed)
ollama-dev:
  build:
    context: .
    dockerfile: Dockerfile.dev
  volumes:
    - ./ollama-src:/workspace/ollama-s390x:Z
    - ./ollama-models:/root/.ollama:Z
  working_dir: /workspace/ollama-s390x
  restart: unless-stopped
```

---

## Immediate Workaround (existing container)

If you are already inside the old container and hitting the lock, **wait for
PID 101 to finish**:

```sh
# Monitor until the background apt process exits
watch -n2 'ps aux | grep apt | grep -v grep'

# Once it is gone, git is available
git --version
```

---

## Re-running After the Fix is Pushed

```sh
# 1. Tear down the old stack
podman compose down

# 2. Re-bootstrap — Dockerfile.dev is generated and the image is built once
curl -fsSL https://raw.githubusercontent.com/Brice12347/ollama-s390x/main/scripts/bootstrap_dev_env.sh | bash

# 3. Enter the container — all tools are ready immediately
podman compose exec ollama-dev bash
git --version        # works instantly
go version           # works instantly
cmake --version      # works instantly
```

---

## Before vs After

| | Before | After |
|---|---|---|
| `git` availability | Only after entrypoint `apt` finishes (minutes) | Immediately on `exec` |
| `apt` lock races | Yes — entrypoint and interactive `apt` conflict | No — no runtime `apt` calls |
| Image rebuild needed | Never (always pulls `debian:bookworm`) | Only when `Dockerfile.dev` changes |
| Architecture support | Hard-coded `linux-s390x` or `linux-amd64` | Auto-detected from `uname -m` |
