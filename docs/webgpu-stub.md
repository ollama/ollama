# Experimental WebGPU Backend (stub)

This document describes a placeholder implementation for a future WebGPU-accelerated inference backend. It is intentionally minimal – the goal is to provide a clear entry point for the community to build on.

## Goals
- Show how a WebGPU provider could be registered in the Ollama codebase.
- Keep the implementation lightweight so it does not affect existing builds.
- Encourage contributors to flesh out the real logic.

## Design Sketch
1. A new provider class (WebGPUProvider) implements the generic provider interface.
2. It registers under the ID webgpu in ackend/provider/catalog.go (or the equivalent Go registry).
3. The provider expects an env var OLLAMA_WEBGPU_ENDPOINT pointing at a local WebGPU inference server.
4. When the endpoint is reachable, the provider lists a single placeholder model (webgpu-alpha-1).
5. Actual model loading is left for future work.

## Next steps for contributors
- Add the Go source file (ackend/provider/webgpu.go).
- Implement a simple health-check against the endpoint.
- Expose the provider in the UI settings panel.
- Write tests for registration and error handling.

---
*This stub is deliberately minimal; contributions that flesh it out are welcome!*
