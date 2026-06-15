# Build Attempt 003 - Failure Analysis

**Date:** 2026-06-15  
**Platform:** s390x (IBM Z Architecture)  
**Environment:** Jupyter Notebook via Podman  
**Objective:** Build and run Ollama with custom model on s390x

## Executive Summary

This build attempt encountered multiple critical errors related to CMake configuration, model downloading, GGUF format validation, and port conflicts. While the basic build environment was successfully configured, the attempt to run a custom model failed at multiple stages, revealing important insights about the s390x build process and model handling.

## Errors Encountered

### 1. CMake Preset Not Found

**Error:**
```
CMake Error: Could not find preset "cpu"
Available presets: Default, MLX Metal
```

**What Happened:**  
The build attempted to use a CMake preset named "cpu" which does not exist in the current CMakePresets.json configuration file.

**Why It Occurred:**  
The CMakePresets.json file only defines two presets:
- `Default`: Standard build configuration
- `MLX Metal`: Apple Silicon GPU acceleration (not applicable to s390x)

The "cpu" preset was likely referenced from documentation or examples targeting x86_64 architecture, where CPU-only builds are explicitly configured to distinguish them from CUDA/ROCm GPU builds.

**What Was Learned:**  
- s390x builds should use the "Default" preset, not "cpu"
- CMake preset names are architecture and platform-specific
- The correct build command for s390x is:
  ```bash
  cmake -B build .
  cmake --build build --parallel 8
  ```

**Resolution:**  
Use the Default preset or omit the preset specification entirely, allowing CMake to use default configuration.

---

### 2. HuggingFace Model Download Authentication Failure

**Error:**
```
HTTP request sent, awaiting response... 401 Unauthorized
```

**What Happened:**  
Attempted to download a model from HuggingFace using wget, but received a 401 Unauthorized error.

**Command Used:**
```bash
wget https://huggingface.co/username/model-name/resolve/main/model.gguf
```

**Why It Occurred:**  
HuggingFace requires authentication for many models, especially:
- Gated models (requiring explicit access approval)
- Private repositories
- Models with usage restrictions

The wget command did not include authentication credentials (API token).

**What Was Learned:**  
- HuggingFace models often require authentication
- wget alone is insufficient for authenticated downloads
- Two authentication methods are available:
  1. Use HuggingFace CLI with token: `huggingface-cli download`
  2. Include token in wget: `wget --header="Authorization: Bearer YOUR_TOKEN"`

**Resolution:**  
Use the HuggingFace CLI or include authentication token in the download request:
```bash
# Method 1: HuggingFace CLI (recommended)
pip install huggingface-hub
huggingface-cli login
huggingface-cli download username/model-name model.gguf

# Method 2: wget with token
wget --header="Authorization: Bearer hf_YOUR_TOKEN" \
  https://huggingface.co/username/model-name/resolve/main/model.gguf
```

---

### 3. GGUF Format Error - HTML Instead of Binary

**Error:**
```
error: invalid GGUF file format
Expected binary model file, received HTML error page
```

**What Happened:**  
The downloaded file was not a valid GGUF model file. Instead, it was an HTML error page from HuggingFace.

**Why It Occurred:**  
When wget encounters a 401 Unauthorized error, it still creates a file with the specified filename. However, instead of containing the model data, the file contains the HTML error page returned by the server. This is a common pitfall when downloading from authenticated endpoints.

**File Contents:**
```html
<!DOCTYPE html>
<html>
<head><title>401 Unauthorized</title></head>
<body>
<h1>Repository not found or access denied</h1>
...
</body>
</html>
```

**What Was Learned:**  
- Always verify file integrity after download
- Check file size and magic bytes before attempting to use downloaded models
- wget does not fail when it receives HTML error pages
- GGUF files should start with the magic bytes "GGUF" (0x47475546)

**Verification Commands:**
```bash
# Check file size (GGUF models are typically GB-sized)
ls -lh model.gguf

# Check magic bytes (should show "GGUF")
xxd -l 4 model.gguf

# Check if file is HTML
file model.gguf
head -n 5 model.gguf
```

**Resolution:**  
1. Authenticate properly before downloading
2. Verify file integrity after download
3. Use checksums when available
4. Implement download validation in scripts

---

### 4. Model Loading Error - Invalid Magic Characters

**Error:**
```
error loading model: invalid magic characters 'Repo'
expected 'GGUF' but got 'Repo'
```

**What Happened:**  
The llama-server attempted to load the downloaded file as a GGUF model but found invalid magic characters at the start of the file.

**Technical Details:**  
GGUF (GPT-Generated Unified Format) files must begin with the 4-byte magic number:
- Expected: `0x47 0x47 0x55 0x46` ("GGUF" in ASCII)
- Received: `0x52 0x65 0x70 0x6F` ("Repo" in ASCII)

The "Repo" prefix indicates the file started with "Repository" - part of the HTML error message "Repository not found or access denied".

**Why It Occurred:**  
This error is a direct consequence of Error #3. The file was an HTML error page, not a binary GGUF model file. The model loader correctly identified the format mismatch.

**What Was Learned:**  
- GGUF format validation is strict and catches corrupted/invalid files immediately
- Magic number checking is the first validation step in model loading
- Error messages clearly indicate format mismatches
- The llama-server has robust error handling for invalid model files

**GGUF Format Structure:**
```
Offset | Size | Content
-------|------|--------
0x00   | 4    | Magic: "GGUF" (0x47475546)
0x04   | 4    | Version number
0x08   | 8    | Tensor count
0x10   | 8    | Metadata KV count
...    | ...  | Model data
```

**Resolution:**  
Download a valid GGUF model file using proper authentication. Verify the magic bytes before attempting to load.

---

### 5. Port Binding Conflict

**Error:**
```
Error: listen tcp :11434: bind: address already in use
```

**What Happened:**  
Attempted to start llama-server on port 11434, but the port was already bound by another process.

**Why It Occurred:**  
A previous instance of the Ollama server or llama-server was still running on port 11434. This commonly happens when:
- A previous server instance wasn't properly terminated
- Multiple terminal sessions are running servers
- The server crashed but the port wasn't released
- Another application is using the same port

**What Was Learned:**  
- Always check for running instances before starting a new server
- Implement proper cleanup in development workflows
- Port conflicts are common in containerized environments
- Multiple server instances can cause confusion during debugging

**Diagnostic Commands:**
```bash
# Check what's using port 11434
lsof -i :11434
netstat -tulpn | grep 11434
ss -tulpn | grep 11434

# Find Ollama processes
ps aux | grep ollama
ps aux | grep llama-server

# Kill existing processes
pkill ollama
pkill llama-server
# or
kill -9 <PID>
```

**Resolution:**  
1. Identify and terminate the existing process
2. Use a different port with `--port` flag
3. Implement proper server lifecycle management
4. Add port checking to startup scripts

---

## Root Cause Analysis

The experiment failed due to a cascade of issues:

1. **Configuration Mismatch**: Using x86_64-specific CMake presets on s390x
2. **Authentication Gap**: Missing HuggingFace credentials for model download
3. **Silent Failure**: wget creating HTML files without failing the command
4. **Validation Bypass**: Not verifying downloaded file integrity before use
5. **Process Management**: Inadequate cleanup of running server instances

## Lessons Learned

### Technical Insights

1. **Architecture-Specific Configuration**: Build configurations are not portable across architectures. Always verify preset availability for the target platform.

2. **Authentication is Critical**: Modern ML model repositories require proper authentication. Implement token-based authentication in automated workflows.

3. **Validate Downloads**: Never assume a download succeeded based on file existence alone. Always verify:
   - File size matches expected size
   - Magic bytes are correct
   - Checksums match (when available)

4. **Error Propagation**: wget's behavior of creating files even on HTTP errors can mask failures. Use `wget --content-disposition` and check exit codes.

5. **Port Management**: In development environments, implement proper cleanup procedures to avoid port conflicts.

### Process Improvements

1. **Pre-flight Checks**: Add validation steps before attempting builds:
   ```bash
   # Verify CMake presets
   cmake --list-presets
   
   # Check authentication
   huggingface-cli whoami
   
   # Verify ports are free
   lsof -i :11434
   ```

2. **Download Validation Script**:
   ```bash
   #!/bin/bash
   download_and_verify() {
       local url=$1
       local output=$2
       local expected_magic=$3
       
       wget "$url" -O "$output"
       
       # Check file size
       if [ $(stat -f%z "$output") -lt 1000000 ]; then
           echo "Error: File too small"
           return 1
       fi
       
       # Check magic bytes
       if ! xxd -l 4 "$output" | grep -q "$expected_magic"; then
           echo "Error: Invalid magic bytes"
           return 1
       fi
       
       return 0
   }
   ```

3. **Cleanup Procedures**: Add cleanup steps to development workflow:
   ```bash
   # Stop all Ollama processes
   pkill -9 ollama llama-server
   
   # Clear temporary files
   rm -f /tmp/*.gguf
   
   # Reset environment
   unset OLLAMA_*
   ```

## Recommendations

### Immediate Actions

1. Update build documentation to specify "Default" preset for s390x
2. Add HuggingFace authentication setup to prerequisites
3. Create download validation utilities
4. Implement port conflict detection in startup scripts

### Long-term Improvements

1. Create architecture-specific build profiles
2. Develop automated model download and validation pipeline
3. Implement comprehensive pre-build validation suite
4. Add process management utilities for development environment

### Documentation Updates

1. Add troubleshooting section for common errors
2. Document authentication requirements clearly
3. Provide model download best practices
4. Include port management guidelines

## Next Steps

1. Configure HuggingFace authentication properly
2. Download a verified GGUF model using authenticated method
3. Verify model file integrity before loading
4. Ensure clean environment (no port conflicts)
5. Retry build with corrected configuration
6. Document successful build process

## References

- [GGUF Format Specification](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)
- [HuggingFace Authentication](https://huggingface.co/docs/huggingface_hub/quick-start#authentication)
- [CMake Presets Documentation](https://cmake.org/cmake/help/latest/manual/cmake-presets.7.html)
- [Ollama Development Guide](../docs/development.md)

## Appendix: Error Timeline

```
[T+00:00] Build initiated with "cpu" preset
[T+00:05] CMake preset error - switched to Default
[T+00:10] Build completed successfully
[T+00:15] Attempted model download from HuggingFace
[T+00:16] 401 Unauthorized error
[T+00:17] File created but contains HTML
[T+00:20] Attempted to load model
[T+00:21] GGUF magic number validation failed
[T+00:25] Attempted to start llama-server
[T+00:26] Port 11434 conflict detected
[T+00:30] Build attempt terminated
```

## Status

**Build Status:** ❌ Failed  
**Environment Setup:** ✅ Successful  
**Model Download:** ❌ Failed  
**Model Loading:** ❌ Failed  
**Server Start:** ❌ Failed (port conflict)

**Overall Assessment:** The build environment was successfully configured, but model acquisition and deployment failed due to authentication and validation issues. All errors are understood and resolvable.