# Build Attempt 004 - Install Script Failure

**Date:** 2026-06-25  
**Time:** 16:28:07 - 16:28:20 EDT  
**Environment:** Podman container (ollama:local)  
**Architecture:** s390x (IBM Z)

## Summary

Attempted to run the install.sh script inside the ollama development container after successful bootstrap. The installation failed due to a 404 error when trying to download the pre-built s390x binary package.

## Bootstrap Phase - SUCCESS ✓

The bootstrap script (`bootstrap_dev_env.sh`) completed successfully:

1. **Container Setup:**
   - Removed existing ollama container
   - Built new container image from `quay.io/podman/stable`
   - Installed build dependencies: git, golang, cmake, ninja-build, gcc, gcc-c++, make, curl, ca-certificates
   - Cloned repository: `https://github.com/Brice12347/ollama-s390x.git`
   - Container started successfully on port 11434

2. **Container Details:**
   - Name: `ollama`
   - Status: Running
   - Ports: `0.0.0.0:11434->11434/tcp`
   - Working Directory: `/workspace/ollama-s390x`

## Install Script Phase - FAILED ✗

### Command Executed
```bash
curl -fsSL https://raw.githubusercontent.com/Brice12347/ollama-s390x/main/scripts/install.sh | sh
```

### Output
```
>>> Detected IBM Z (s390x) architecture
>>> Cleaning up old version at /usr/local/lib/ollama
>>> Installing ollama to /usr/local
>>> Downloading ollama-linux-s390x.tgz
##O#-#                                                                         
curl: (22) The requested URL returned error: 404

gzip: stdin: unexpected end of file
tar: Child returned status 1
tar: Error is not recoverable: exiting now
```

## Root Cause Analysis

The install.sh script attempted to download a pre-built binary package `ollama-linux-s390x.tgz` from the repository, but this file does not exist (404 error). This indicates:

1. **Missing Pre-built Binary:** The repository does not have a pre-compiled s390x binary package available for download
2. **Script Assumption:** The install.sh script assumes pre-built binaries exist for all architectures
3. **Build Required:** For s390x architecture, the binary must be built from source rather than downloaded

## Error Chain

1. `curl` receives HTTP 404 (file not found)
2. Empty/incomplete data passed to `gzip`
3. `gzip` fails with "unexpected end of file"
4. `tar` receives invalid input from `gzip`
5. `tar` exits with error status 1

## Next Steps

Since pre-built binaries are not available for s390x, we need to:

1. **Build from Source:** Use the development environment that was successfully set up
2. **Follow Build Instructions:**
   ```bash
   cd /workspace/ollama-s390x
   cmake -B build .
   cmake --build build --parallel 8
   ./ollama serve
   ```
3. **Alternative:** Modify install.sh to detect missing binaries and fall back to building from source for s390x

## Conclusion

The bootstrap phase was successful, creating a proper development environment. However, the install.sh script is not suitable for s390x as it expects pre-built binaries that don't exist. The solution is to build ollama from source using the cmake build system within the container.

## Related Files
- `scripts/bootstrap_dev_env.sh` - Successfully created dev environment
- `scripts/install.sh` - Failed due to missing s390x binary package
- Container logs: `/root/.ollama-bootstrap/logs/run-20260625-162807.log`