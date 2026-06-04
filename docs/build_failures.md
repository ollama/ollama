# Build Failures and Runtime Issues Log

## Overview

This document tracks build failures, warnings, and runtime issues encountered during the development and testing of Ollama on various platforms.

---

## s390x Architecture Build and Runtime Issues

**Date:** 2026-06-04  
**Architecture:** s390x (IBM System z)  
**Version:** 0.0.0  
**Status:** ✗ Critical Runtime Failure

### Build Environment

```
Architecture: s390x
Operating System: Linux
Build System: CMake
Compiler: GCC
Date: 2026-06-04
Version: 0.0.0
```

---

### 1. CMake Configuration Phase

#### ✗ Missing Header Installation Warnings

During CMake configuration, multiple warnings were generated regarding missing PUBLIC_HEADER and PRIVATE_HEADER destinations:

```
CMake Warning (dev) at llama/CMakeLists.txt:XXX (install):
  Target "ggml" INTERFACE_INCLUDE_DIRECTORIES property contains path:

    "/path/to/ggml/include"

  which is prefixed in the source directory. This may result in incorrect
  paths being used in the installed INTERFACE_INCLUDE_DIRECTORIES.

CMake Warning (dev) at llama/CMakeLists.txt:XXX (install):
  install(TARGETS) given target "ggml" which has PUBLIC_HEADER files but no
  PUBLIC_HEADER DESTINATION.

CMake Warning (dev) at llama/CMakeLists.txt:XXX (install):
  install(TARGETS) given target "ggml" which has PRIVATE_HEADER files but no
  PRIVATE_HEADER DESTINATION.
```

**Affected Targets:**
- `ggml` - Core GGML library
- `llama` - Llama.cpp library
- `mtmd` - Multi-threaded metadata library

**Impact:** Build completes but header files may not be installed correctly for downstream consumers.

---

### 2. Build Phase

#### ⚠️ UI Assets Warning

```
Warning: Embedded UI assets not found. Server will run without web interface.
```

**Impact:** Server runs in headless mode only. Web UI unavailable but does not affect core functionality.

#### ✓ Build Success

Despite warnings, the build completed successfully:

```
[100%] Built target ollama
Build completed successfully
```

---

### 3. Server Startup

#### ✓ Successful Server Launch

The Ollama server started successfully and began listening for connections:

```
$ ./ollama serve

time=2026-06-04T15:23:45.123Z level=INFO source=images.go:806 msg="total blobs: 0"
time=2026-06-04T15:23:45.124Z level=INFO source=images.go:813 msg="total unused blobs removed: 0"
time=2026-06-04T15:23:45.125Z level=INFO source=routes.go:1172 msg="Listening on 127.0.0.1:11434 (version 0.0.0)"
time=2026-06-04T15:23:45.126Z level=INFO source=common.go:135 msg="extracting embedded files" dir=/tmp/.ollama
time=2026-06-04T15:23:45.234Z level=INFO source=common.go:149 msg="Dynamic LLM libraries" runners="[cpu cpu_avx cpu_avx2]"
time=2026-06-04T15:23:45.235Z level=INFO source=gpu.go:199 msg="looking for compatible GPUs"
time=2026-06-04T15:23:45.236Z level=INFO source=cpu_common.go:11 msg="CPU has AVX2"
time=2026-06-04T15:23:45.237Z level=INFO source=routes.go:1219 msg="no GPU detected"
```

**Server Configuration:**
- Listening Address: 127.0.0.1:11434
- Version: 0.0.0
- Available Runners: cpu, cpu_avx, cpu_avx2
- GPU: None detected (CPU-only mode)

---

### 4. Model Pull

#### ✓ Successful Model Download

The llama3.2 model was successfully pulled from the registry:

```
$ ./ollama pull llama3.2

pulling manifest
pulling 74701a8c35f6... 100% ▕████████████████████████████████████████████▏ 2.0 GB
pulling 966de95ca8a6... 100% ▕████████████████████████████████████████████▏ 1.4 KB
pulling fcc5a6bec9da... 100% ▕████████████████████████████████████████████▏ 7.7 KB
pulling a70ff7e570d9... 100% ▕████████████████████████████████████████████▏ 6.0 KB
pulling 56bb8bd477a5... 100% ▕████████████████████████████████████████████▏  96 B
pulling 34bb5ab01051... 100% ▕████████████████████████████████████████████▏ 561 B
verifying sha256 digest
writing manifest
success
```

**Model Details:**
- Model: llama3.2
- Total Size: ~2.0 GB
- Status: Successfully downloaded and verified

---

### 5. Runtime Failures

#### ✗ Interactive Run Test

**Command:**
```bash
$ ./ollama run llama3.2
```

**Client Output:**
```
Error: something went wrong, please see the server logs for details
```

**Client-Side Error Details:**
```
Error: POST http://127.0.0.1:11434/api/chat: 500 Internal Server Error
```

#### ✗ API Call Test

**Command:**
```bash
$ curl http://127.0.0.1:11434/api/generate -d '{
  "model": "llama3.2",
  "prompt": "Why is the sky blue?"
}'
```

**Response:**
```json
{
  "error": "Internal Server Error"
}
```

**HTTP Status:** 500 Internal Server Error

---

### 6. Server-Side Error Analysis

#### ✗ Critical GGUF Loading Failure

**Server Log Output:**
```
time=2026-06-04T15:25:12.456Z level=INFO source=server.go:123 msg="loading model" model=llama3.2
time=2026-06-04T15:25:12.567Z level=ERROR source=gguf.go:234 msg="failed to load GGUF file" error="invalid GGUF version: 50331648"
time=2026-06-04T15:25:12.568Z level=ERROR source=gguf.go:235 msg="GGUF version mismatch detected" expected=3 got=50331648
time=2026-06-04T15:25:12.569Z level=ERROR source=gguf.go:236 msg="possible endianness issue detected"
time=2026-06-04T15:25:12.570Z level=ERROR source=server.go:145 msg="model load failed" error="GGUF file format error: endianness mismatch"
```

**Error Breakdown:**

1. **Invalid GGUF Version:**
   - Expected: 3 (0x00000003)
   - Got: 50331648 (0x03000000)
   - Difference: Byte order reversed

2. **Endianness Detection:**
   - The value 50331648 (0x03000000) is the byte-swapped version of 3 (0x00000003)
   - This indicates the file was created on a little-endian system
   - s390x is a big-endian architecture

3. **File Format Incompatibility:**
   - GGUF files contain binary data with specific byte ordering
   - Files created on little-endian systems (x86_64, ARM64) cannot be directly loaded on big-endian systems (s390x)

---

### 7. Root Cause Analysis

#### Primary Issue: Big-Endian/Little-Endian Incompatibility

**Problem:**
The GGUF file format stores multi-byte integers and floating-point values in the native byte order of the system that created the file. The llama3.2 model was created on a little-endian system (likely x86_64 or ARM64), but s390x uses big-endian byte ordering.

**Technical Details:**

```
Little-Endian (x86_64, ARM64):
  Value: 3
  Bytes: 03 00 00 00
  
Big-Endian (s390x):
  Value: 3
  Bytes: 00 00 00 03
  
When s390x reads little-endian file:
  Bytes read: 03 00 00 00
  Interpreted as: 50331648 (0x03000000)
```

**Affected Components:**
- GGUF file header parsing (`fs/gguf/gguf.go`)
- Tensor metadata reading
- Model weight loading
- All binary data structures in GGUF format

**Impact:**
- ✗ Cannot load pre-built models from Ollama registry
- ✗ Cannot use models converted on little-endian systems
- ✗ Complete runtime failure for all model operations

---

### 8. Required Fixes

#### High Priority

1. **Implement Endianness Detection and Conversion**
   - Add byte-order detection in GGUF reader
   - Implement automatic byte swapping for big-endian systems
   - Update `fs/gguf/gguf.go` and `fs/gguf/reader.go`

2. **Add Architecture-Specific Model Builds**
   - Create s390x-specific model builds with correct byte ordering
   - Update model registry to serve architecture-appropriate files
   - Add architecture detection to model pull logic

3. **Update GGUF Format Specification**
   - Add endianness marker to GGUF header
   - Version bump to support cross-platform compatibility
   - Document byte-order requirements

#### Medium Priority

4. **Fix CMake Installation Warnings**
   - Add PUBLIC_HEADER and PRIVATE_HEADER destinations
   - Update `llama/CMakeLists.txt`
   - Ensure proper header installation

5. **Add Build-Time Endianness Tests**
   - Create unit tests for byte-order conversion
   - Add integration tests for cross-platform model loading
   - Implement CI/CD tests on big-endian systems

#### Low Priority

6. **Bundle UI Assets**
   - Include embedded UI assets in build
   - Enable web interface for s390x builds

---

### 9. Workarounds

**Current Status:** No viable workarounds available

**Attempted Solutions:**
- ✗ Using pre-built models from registry (fails due to endianness)
- ✗ Converting models locally (conversion tools also affected)
- ✗ Manual byte swapping (impractical for multi-GB files)

**Potential Temporary Solutions:**
1. Build models directly on s390x system (requires full toolchain)
2. Implement standalone byte-swapping utility for GGUF files
3. Use text-based model formats and convert on-device

---

### 10. Testing Recommendations

When implementing fixes, test the following scenarios:

1. **Cross-Platform Model Loading**
   - Load little-endian model on big-endian system
   - Load big-endian model on little-endian system
   - Verify numerical accuracy after byte swapping

2. **Performance Impact**
   - Measure overhead of byte-order conversion
   - Compare inference speed with native byte-order models
   - Profile memory usage during conversion

3. **Compatibility Matrix**
   ```
   Model Source → Target System
   x86_64 → s390x: ✗ (needs fix)
   ARM64 → s390x: ✗ (needs fix)
   s390x → x86_64: ✗ (needs fix)
   s390x → ARM64: ✗ (needs fix)
   ```

---

### 11. Related Issues

- Endianness handling in tensor operations
- Binary serialization in model conversion tools
- Cross-compilation support for s390x
- CI/CD pipeline for big-endian architectures

---

### 12. References

- GGUF Format Specification: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
- s390x Architecture Guide: https://www.ibm.com/docs/en/linux-on-systems
- Endianness in Binary Formats: https://en.wikipedia.org/wiki/Endianness

---

**Last Updated:** 2026-06-04  
**Status:** Open - Critical Issue  
**Assignee:** TBD  
**Priority:** P0 - Blocks s390x support