# Ollama s390x Installation Testing - Attempt 001

**Date:** 2026-06-24  
**Platform:** IBM Z (s390x) architecture  
**Environment:** z-spyre-runtime container

## Overview

Testing Ollama build and installation on s390x architecture after modifying install.sh and environment setup scripts.

## Setup Process

### Environment Setup

1. **SSH Connection**
   - Connected to triframe system
   - Navigation path:
     ```
     cd /Wonder
     cd/bricepatchou
     cd/workspace
     ```

2. **Container Launch**
   - Launched z-spyre-runtime container
   - Instructions: https://pages.github.ibm.com/zosdev/z-spyre-docs/docs/main/containers/runtime/overview

3. **Repository Setup**
   - Changed to workspace directory
   - Cloned repository:
     ```bash
     git clone git@github.com:Brice12347/ollama-s390x.git
     cd ollama-s390x
     ```

4. **Dependencies Installation**
   - Updated package manager:
     ```bash
     apt update
     ```
   - Attempted to install dependencies:
     ```bash
     apt install golang-go cmake ninja-build
     ```
     - **Result:** `apt not found`
   - Used alternative package manager:
     ```bash
     dnf install golang cmake ninja-build
     ```
     - **Result:** Successfully installed

## Build Process

### Native Build Model

1. **CMake Configuration**
   ```bash
   cmake -B build .
   ```
   - **Result:** ✅ Works

2. **Build Execution**
   ```bash
   cmake --build build --parallel 8
   ```
   - **Result:** ✅ Works

3. **Server Launch**
   ```bash
   ./ollama serve
   ```
   - **Result:** ✅ Server started successfully
   - Server listening on: `127.0.0.1:11434`
   - Version: `0.0.0`

### Server Output

```
time=2026-06-24T10:38:17.196-04:00 level=INFO source=routes.go:1919 msg="server config" env="map[CUDA_VISIBLE_DEVICES: GGML_VK_VISIBLE_DEVICES: GPU_DEVICE_ORDINAL: HIP_VISIBLE_DEVICES: HSA_OVERRIDE_GFX_VERSION: HTTPS_PROXY: HTTP_PROXY: LLAMA_ARG_FIT: LLAMA_ARG_FIT_TARGET: NO_PROXY: OLLAMA_CONTEXT_LENGTH:0 OLLAMA_DEBUG:INFO OLLAMA_DEBUG_LOG_REQUESTS:false OLLAMA_EDITOR: OLLAMA_FLASH_ATTENTION:false OLLAMA_GO_TEMPLATE:true OLLAMA_GPU_OVERHEAD:0 OLLAMA_HOST:http://127.0.0.1:11434 OLLAMA_IGPU_ENABLE: OLLAMA_KEEP_ALIVE:5m0s OLLAMA_KV_CACHE_TYPE: OLLAMA_LLM_LIBRARY: OLLAMA_LOAD_TIMEOUT:5m0s OLLAMA_MAX_LOADED_MODELS:0 OLLAMA_MAX_QUEUE:512 OLLAMA_MAX_TRANSFER_STREAMS:4 OLLAMA_MODELS:/root/.ollama/models OLLAMA_NOHISTORY:false OLLAMA_NOPRUNE:false OLLAMA_NO_CLOUD:false OLLAMA_NUM_PARALLEL:1 OLLAMA_ORIGINS:[http://localhost https://localhost http://localhost:* https://localhost:* http://127.0.0.1 https://127.0.0.1 http://127.0.0.1:* https://127.0.0.1:* http://0.0.0.0 https://0.0.0.0 http://0.0.0.0:* https://0.0.0.0:* app://* file://* tauri://* vscode-webview://* vscode-file://*] OLLAMA_REMOTES:[ollama.com] OLLAMA_SCHED_SPREAD:false OLLAMA_VULKAN:true ROCR_VISIBLE_DEVICES: http_proxy: https_proxy: no_proxy:]"

time=2026-06-24T10:38:17.196-04:00 level=INFO source=routes.go:1921 msg="Ollama cloud disabled: false"

time=2026-06-24T10:38:17.196-04:00 level=INFO source=images.go:864 msg="total blobs: 0"

time=2026-06-24T10:38:17.196-04:00 level=INFO source=images.go:871 msg="total unused blobs removed: 0"

time=2026-06-24T10:38:17.196-04:00 level=INFO source=routes.go:1981 msg="Listening on 127.0.0.1:11434 (version 0.0.0)"

time=2026-06-24T10:38:17.197-04:00 level=INFO source=model_list_cache.go:111 msg="model list cache hydration complete" models=0 failures=0 elapsed=936.16µs

time=2026-06-24T10:38:17.197-04:00 level=INFO source=runner.go:60 msg="discovering available GPUs..."

time=2026-06-24T10:38:17.364-04:00 level=INFO source=types.go:50 msg="inference compute" id=cpu library=cpu compute="" name=cpu description=cpu libdirs=ollama driver="" pci_id="" type="" total="1007.0 GiB" available="1005.6 GiB"

time=2026-06-24T10:38:17.365-04:00 level=INFO source=routes.go:2031 msg="vram-based default context" total_vram="0 B" default_num_ctx=4096

time=2026-06-24T10:38:17.645-04:00 level=INFO source=model_recommendations.go:177 msg="model recommendations cache sleep scheduled" wait=3h48m29.444531001s consecutive_failures=0
```

## Testing

### Test 1: Server Status Check

**Setup:**
- Launched separate CLI in z-spyre-runtime container
- Changed to ollama-s390x directory

**Command:**
```bash
curl http://127.0.0.1:11434/api/version
```

**Output:**
```json
{"version":"0.0.0"}
```

**Result:** ✅ Server responding correctly

### Test 2: List Available Models

**Command:**
```bash
curl http://127.0.0.1:11434/api/tags
```

**Output:**
```json
{
  "models": [{
    "name": "llama3.2:1b",
    "model": "llama3.2:1b",
    "modified_at": "2026-06-24T10:54:40.826670871-04:00",
    "size": 1321098329,
    "digest": "baf6a787fdffd633537aa2eb51cfd54cb93ff08e2804009542bb63daf552878",
    "details": {
      "parent_model": "",
      "format": "gguf",
      "family": "llama",
      "families": ["llama"],
      "parameter_size": "1.2B",
      "quantization_level": "Q8_0",
      "context_length": 131072,
      "embedding_length": 2048,
      "capabilities": ["completion", "tools"]
    }
  }]
}
```

**Result:** ✅ Model list retrieved successfully

### Test 3: Run Model

**Setup:**
- Launched separate CLI in z-spyre-runtime container
- Changed to ollama-s390x directory

**Command:**
```bash
./ollama run llama3.2:1b
```

**Process:**
1. ✅ Successfully pulled manifest
2. ✅ Successfully wrote manifest
3. ❌ **Error occurred:**

**Error Message:**
```
Error: 500 Internal Server Error: llama-server process has terminated: exit status 1: error loading model: llama_model_loader: failed to load model from /root/.ollama/models/blobs/sha256-74701a8c35f6c8d9a4b91f3f3497643001d63e0c7a84e085bed452548fa88d45
```

**API Test:**
```bash
curl http://127.0.0.1:11434/api/generate -d '{
  "model": "llama3.2:1b",
  "prompt": "Why is the sky blue?"
}'
```

**Error Response:**
```json
{
  "error": "llama-server process has terminated: exit status 1: error loading model: llama_model_loader: failed to load model from /root/.ollama/models/blobs/sha256-74701a8c35f6c8d9a4b91f3f3497643001d63e0c7a84e085bed452548fa88d45\nerror loading model: llama_model_loader: failed to load model from /root/.ollama/models/blobs/sha256-74701a8c35f6c8d9a4b91f3f3497643001d63e0c7a84e085bed452548fa88d45"
}
```

## Key Finding

**The build is successful, but all pre-built models from Ollama's registry are incompatible with s390x.**

The models available in Ollama's public registry are compiled for x86_64/ARM architectures and cannot run on s390x. The GGUF model files themselves are architecture-specific in their binary format.

## Testing install.sh Changes

### Test 4: Modified Install Script

**Setup:**
- Launched z-spyre-runtime container
- Instructions: https://pages.github.ibm.com/zosdev/z-spyre-docs/docs/main/containers/runtime/overview

**Command:**
```bash
curl -fsSL https://raw.githubusercontent.com/Brice12347/ollama-s390x/main/scripts/install.sh | sh
```

**Output:**
```
>>> Detected IBM Z (s390x) architecture
>>> Installing ollama to /usr/local
>>> Downloading ollama-linux-s390x.tgz
##O#-#                                curl: (22) The requested URL returned error: 404

gzip: stdin: unexpected end of file
tar: Child returned status 1
tar: Error is not recoverable: exiting now
```

**Result:** ❌ Failed - Binary package not available

## Blockers for install.sh to Work on s390x

### 1. No s390x Binaries Exist
- `ollama-linux-s390x.tgz` not available at https://ollama.com/download/
- Official Ollama releases only provide x86_64 and ARM64 binaries

### 2. Build Must Complete Successfully
- Need working s390x build before packaging binaries
- Current build works, but needs validation

### 3. Binary Packaging Required
- Must create `.tar.zst` or `.tgz` archive with proper structure
- Archive should contain:
  - `ollama` binary
  - Required libraries
  - Installation metadata

### 4. Hosting Infrastructure Needed
- Binaries must be hosted at accessible URL
- Options:
  - GitHub Releases
  - Custom hosting server
  - IBM internal repository

## Next Steps

1. **Model Compatibility:**
   - Need to build/convert models specifically for s390x architecture
   - Investigate GGUF format requirements for s390x
   - Consider building models from scratch using llama.cpp on s390x

2. **Binary Distribution:**
   - Package the successful build into distributable format
   - Set up hosting for s390x binaries
   - Update install.sh to point to hosted binaries

3. **Testing:**
   - Test with locally built/converted models
   - Validate model inference on s390x
   - Performance benchmarking

## Conclusions

- ✅ Ollama builds successfully on s390x
- ✅ Server starts and responds to API calls
- ❌ Pre-built models from registry are incompatible
- ❌ Install script cannot work without hosted s390x binaries
- **Critical Path:** Need s390x-compatible model files to validate full functionality