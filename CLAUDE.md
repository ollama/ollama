# CLAUDE.md

This file provides guidance for AI assistants working with the Ollama codebase.

## Project Overview

Ollama is a multi-platform LLM serving application that provides:
- REST API for model inference (including OpenAI-compatible endpoints)
- CLI for model management and interactive chat
- Desktop applications for macOS and Windows
- Model format conversion and optimization tools

**Primary Languages:** Go (1.24.1) with C/C++ components via CGO (llama.cpp bindings)

## Project Structure

```
/                       # Root - main.go entry point
├── cmd/                # CLI implementation (Cobra-based commands)
├── server/             # HTTP API server and route handlers
├── api/                # Go client library and type definitions
├── model/              # Model inference, tokenization, and processing
│   ├── models/         # Per-architecture model implementations
│   ├── parsers/        # Response parsers for different model formats
│   ├── renderers/      # Output formatting
│   ├── input/          # Input processing
│   └── imageproc/      # Image preprocessing for multimodal models
├── llama/              # CGO bindings to llama.cpp (vendored with patches)
├── ml/                 # ML backends (GGML, neural network utilities)
├── convert/            # Model format conversion (GGUF, SafeTensors, Torch)
├── runner/             # Model execution runtime HTTP interface
├── app/                # Desktop application (macOS/Windows)
├── openai/             # OpenAI-compatible API layer
├── anthropic/          # Anthropic API compatibility layer
├── parser/             # Modelfile parsing (like Dockerfile for models)
├── tools/              # Tool calling and template execution
├── types/              # Core type definitions and errors
├── fs/                 # Filesystem utilities (GGUF/GGML file handling)
├── x/                  # Experimental modules (image generation, agents)
├── docs/               # API documentation and guides
└── integration/        # Integration tests
```

## Key Entry Points

- **`main.go`** - Root entry point, initializes Cobra CLI
- **`cmd/cmd.go`** - CLI commands (serve, run, pull, push, create, list, show, etc.)
- **`server/routes.go`** - API route handlers for all endpoints
- **`server/sched.go`** - Model scheduling and GPU/CPU memory management

## Building and Running

### Quick Start (Development)

```shell
go run . serve
```

### Full Build with GPU Support

```shell
# Configure and build C/C++ components
cmake -B build
cmake --build build

# Run
go run . serve
```

### CGO Cache Issues

If you encounter unexpected crashes, clear the CGO cache:

```shell
go clean -cache
```

### GPU Backends

- **Metal** (macOS Apple Silicon) - built-in, no extra steps
- **CUDA** (NVIDIA) - requires CUDA SDK
- **ROCm** (AMD) - requires ROCm and Ninja build
- **Vulkan** (cross-platform) - requires Vulkan SDK

## Running Tests

```shell
# Unit tests
go test ./...

# Integration tests (requires built ollama binary)
go test -tags=integration ./...

# With model tests (longer timeout needed)
go test -tags=integration,models -timeout 60m ./...

# With synctest for CI parity
GOEXPERIMENT=synctest go test ./...
```

## Code Style and Conventions

### Commit Messages

Format: `<package>: <short description>`

- Package is the most affected Go package (or directory for non-Go changes)
- Description starts lowercase, completes "This changes Ollama to..."

**Good:**
```
llm/backend/mlx: support the llama architecture
server: add streaming support for embeddings
```

**Bad:**
```
feat: add more emoji
fix: was not using famous web framework
```

### Linting

The project uses golangci-lint with strict settings. Key enabled linters:
- `gofmt`, `gofumpt` - formatting
- `bodyclose` - HTTP response body closing
- `misspell` - spelling errors
- `unconvert` - unnecessary conversions

Run linting:
```shell
golangci-lint run
```

### Testing Philosophy

- Test behavior, not implementation
- Use table-driven tests where appropriate
- Integration tests in `/integration/` directory with build tags

## Architecture Patterns

### Request/Response Streaming

Most APIs support streaming via `"stream": true`:
- JSON objects streamed line-by-line
- Harmony parser (`/harmony/`) handles streamed protocol

### Model Management

- Models stored in `~/.ollama/models/` (configurable via OLLAMA_MODELS)
- Modelfile format for customization (similar to Dockerfile)
- Version tags: `model:tag` (e.g., `llama3:70b`)

### API Endpoints

Key endpoints in `server/routes.go`:
- `/api/generate` - Text completion
- `/api/chat` - Chat completion
- `/api/embeddings` - Embedding generation
- `/api/pull` - Model downloading
- `/api/create` - Model creation from Modelfile
- `/api/tags` - List local models
- `/v1/chat/completions` - OpenAI-compatible chat

### Model Parsers

Each model family has a parser in `/model/parsers/`:
- Handles model-specific response formats
- Manages tool call parsing
- Examples: `qwen3coder.go`, `llama4.go`, `gemma3.go`

## Critical Considerations

### Vendored llama.cpp

The `/llama/` directory contains patched llama.cpp code. When modifying:

```shell
# Apply existing patches
make -f Makefile.sync apply-patches

# After making changes, generate new patches
make -f Makefile.sync format-patches sync
```

Patches are tracked in `/llama/patches/`. Changes should ideally be contributed upstream.

### CGO Dependency

Code depends on C/C++ via CGO. Data structure changes can cause cache invalidation issues and crashes. Use `go clean -cache` if you encounter issues.

### Platform-Specific Code

Many packages have platform-specific implementations:
- `*_darwin.go` - macOS
- `*_windows.go` - Windows
- `*_linux.go` - Linux

### Backward Compatibility

From CONTRIBUTING.md:
- Changes that break API backwards compatibility may not be accepted
- This includes the OpenAI-compatible API
- New features add maintenance burden and should be proposed via issues first

## Common Development Tasks

### Adding a New Model Architecture

1. Add model implementation in `/model/models/`
2. Add parser in `/model/parsers/` if needed
3. Add conversion support in `/convert/`
4. Update model registration

### Adding an API Endpoint

1. Add handler in `server/routes.go`
2. Add types in `api/types.go`
3. Add client method in `api/client.go`
4. Add tests in `server/routes_*_test.go`

### Working with Model Files

- Modelfile syntax documented in `docs/modelfile.mdx`
- Parser implementation in `/parser/`

## Environment Variables

- `OLLAMA_HOST` - Server bind address (default: 127.0.0.1:11434)
- `OLLAMA_MODELS` - Model storage directory
- `OLLAMA_DEBUG` - Enable debug logging
- `OLLAMA_KEEP_ALIVE` - Model keep-alive duration
- `OLLAMA_NUM_PARALLEL` - Number of parallel requests

## File Naming Conventions

- `*_test.go` - Test files (164 total)
- `*_darwin.go`, `*_windows.go`, `*_linux.go` - Platform-specific
- `zzz_*.go` - Auto-generated files (e.g., `zzz_generate.go`)

## Dependencies

Key Go dependencies:
- `github.com/gin-gonic/gin` - HTTP framework
- `github.com/spf13/cobra` - CLI framework
- `github.com/mattn/go-sqlite3` - SQLite (desktop app)
- `github.com/stretchr/testify` - Test assertions

## Getting Help

- GitHub Issues: https://github.com/ollama/ollama/issues
- Discord: https://discord.gg/ollama
- API Documentation: `docs/api.md`
- Development Guide: `docs/development.md`
