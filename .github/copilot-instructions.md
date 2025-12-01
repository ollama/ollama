# Ollama Development Instructions

## Overview
Ollama is a Go application for running large language models locally. It provides a CLI and REST API for model management and inference.

## Architecture
- **CLI (cmd/)**: Command-line interface using Cobra, handles user commands like `pull`, `run`, `serve`
- **Server (server/)**: HTTP server using Gin, provides REST API for model operations
- **Runner (runner/)**: Manages model execution and GPU/CPU resource allocation
- **LLM Backends (llama/, llm/)**: Interfaces to underlying model libraries (llama.cpp, etc.)
- **Convert (convert/)**: Handles model format conversions (GGUF, Safetensors)
- **API (api/)**: Client library for interacting with Ollama server

## Data Flows
- Models downloaded from registry via `server/download.go` and stored in local cache
- Inference requests flow: API -> Server routes -> Scheduler -> Runner -> LLM backend
- Model loading: Lazy loading with memory management via `kvcache/`
- Cross-platform builds using CMake for C++ components (`CMakeLists.txt`)

## Critical Workflows
- **Build**: `go build .` (requires Go 1.24.1+), or use `Makefile.sync` for full build
- **Run server**: `./ollama serve` starts HTTP server on :11434
- **Pull models**: `ollama pull <model>` downloads from ollama.com/library
- **Run models**: `ollama run <model>` for interactive chat
- **Debug**: Set `OLLAMA_DEBUG=1` for verbose logging

## Conventions
- **Model formats**: GGUF for imported models, Modelfile for customization
- **Error handling**: Uses `errtypes` package for structured errors
- **Logging**: slog for structured logging, configurable via env vars
- **Testing**: Go tests with testify, integration tests in `integration/`
- **Dependencies**: Go modules, C++ components via CMake

## Integration Points
- **OpenAI API**: Compatible endpoints in `openai/` package
- **Model registry**: `server/internal/registry/` handles model discovery
- **External tools**: `tools/` for model quantization and optimization
- **Auth**: `auth/` for user authentication and model access control

## Examples
- Create custom model: `ollama create mymodel -f Modelfile` (see `Modelfile` in root)
- API call: `curl http://localhost:11434/api/generate -d '{"model":"llama3.2","prompt":"hello"}'`
- Debug build: `go build -tags debug .`

## File References
- `server/routes.go` - API endpoint definitions
- `cmd/cmd.go` - CLI command implementations
- `runner/runner.go` - Model execution logic
- `go.mod` - Dependencies and Go version
- `README.md` - User-facing documentation