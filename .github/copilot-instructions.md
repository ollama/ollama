# Ollama Copilot Instructions

This file configures GitHub Copilot for the Ollama repository.
Focus: https://github.com/ollama/ollama.git - All guidance applies only to this repository.

## Project Overview

**Ollama** is an open-source project for building with large language models. It provides:
- CLI tool and REST API for running models locally
- Multi-platform support (macOS, Windows, Linux, Docker)
- Model management, downloading, and inference
- OpenAI-compatible API endpoints

Key characteristics:
- Written in **Go** (Go 1.24.1+)
- Cobra CLI framework for commands
- Built with performance and simplicity in mind
- Cross-platform support with OS-specific implementations

## Code Style & Conventions

### Go Style Guide
- Follow [Effective Go](https://golang.org/doc/effective_go)
- Use `go fmt` for formatting (enforced by CI)
- Packages align to functionality, not naming patterns
- No single-letter variable names except in ranges and short closures
- Comments explain "why", not "what" (code should be self-documenting)

### Commit Messages
Format: `<package>: <description>` (lowercase, imperative)

Examples:
- `llm/backend/mlx: support the llama architecture`
- `cmd: add interactive mode for model selection`
- `api/client: improve error handling for connection timeouts`

Avoid:
- Generic messages: "fix: bug", "feat: improvement"
- Unnecessary brevity that loses context

### File Organization
- `cmd/` - CLI commands and TUI logic
- `server/` - HTTP API and request handling
- `llm/` - LLM inference and model loading
- `convert/` - Model format conversions (separate converters per architecture)
- `api/` - REST API types and client implementations
- `model/` - Model manifest and metadata management
- `auth/` - Authentication and authorization
- `app/` - Desktop app (macOS app kit, Windows installer)
- `runner/` - Model execution runtime
- `template/` - Model parameter templates
- `internal/` - Vendored or internal utilities
- `vendor/` - Go dependencies (if used)

## Architecture

### Core Components

1. **CLI Layer** (`cmd/`)
   - Main entry point via Cobra
   - Interactive shell with readline support
   - Live output and streaming responses

2. **API Server** (`server/`, `api/`)
   - REST API (port 11434 by default)
   - WebSocket support for streaming
   - Middleware for logging, CORS, auth
   - OpenAI-compatible endpoints

3. **Model Management** (`model/`, `manifest/`)
   - Ollama model format (GGUF, SafeTensors, etc.)
   - Manifest parsing and caching
   - Multi-model support with isolation

4. **Inference Engine** (`llm/`, `runner/`, `tokenizer/`)
   - Model loading and unloading
   - Tokenization and token streaming
   - Backend selection (MLX, CUDA, CPU)
   - GPU memory management

5. **Model Conversion** (`convert/`)
   - Model architecture detection
   - Format conversion pipelines
   - Architecture-specific converters

6. **Platform Support** (`app/`, `darwin/`, Windows TUI)
   - Native macOS app wrapper
   - Windows installer and system tray
   - Linux systemd integration

### Key Design Patterns
- **Streaming responses**: Use `io.Writer` interfaces for token output
- **Context propagation**: Always pass `context.Context` for cancellation
- **Error handling**: Wrap errors with `fmt.Errorf()` and include context
- **Goroutine safety**: Use sync primitives carefully; document race conditions

## Build and Test

### Prerequisites
```bash
# Go 1.24.1 or later
go version

# macOS: Xcode command line tools
xcode-select --install

# Linux: Build tools
sudo apt install build-essential
```

### Building
```bash
# Build the CLI
go build ./cmd

# Build with specific backend
OLLAMA_BACKEND=mlx go build ./cmd

# Build tests
go test ./...
```

### Testing
```bash
# Run all tests
go test ./...

# Run specific package tests
go test ./llm/...

# Run with verbose output
go test -v ./...

# Run with race detector
go test -race ./...

# Run benchmarks
go test -bench=. -benchmem ./cmd/bench
```

### Key Commands
- `go fmt ./...` - Format all code
- `go vet ./...` - Static analysis
- `golangci-lint run` - Full linting (see `.golangci.yaml`)
- `make` - See Makefile.sync for build targets

## Testing Guidelines

### Test Requirements
- Include tests for new functionality (strive for behavior, not implementation)
- Place tests in `*_test.go` files alongside code
- Test both happy path and error cases
- Use table-driven tests for multiple scenarios

### Test Structure
```go
// Good: descriptive test names, clear assertions
func TestModelLoad_InferenceCompletes(t *testing.T) {
    m, err := LoadModel("llama")
    require.NoError(t, err)

    resp, err := m.Generate(ctx, prompt)
    require.NoError(t, err)
    assert.NotEmpty(t, resp.Text)
}

// Avoid: generic names, unclear intent
func TestModel(t *testing.T) { ... }
```

## Dependencies

### Policy
- Add dependencies sparingly
- Justify why existing alternatives don't work
- Prefer stdlib over external packages when feasible
- Common dependencies: Cobra, Gin, sqlite3, protocol buffers

### Adding Dependencies
Before adding a new dependency:
1. Check if stdlib provides similar functionality
2. Verify it's actively maintained
3. Check for security issues
4. Document in PR why it's necessary
5. Update `go.mod` with `go get`

## Key Files and Patterns

### Server Implementation (`server/`)
- HTTP handlers use Gin framework
- Middleware in `middleware/`
- OpenAI compatibility layer in `openai/`
- Example: `/v1/chat/completions` routes through compatibility layer

### Model Loading (`model/`, `manifest/`)
- Manifests are YAML/JSON metadata files
- Model registry at `~/.ollama/models/`
- Cache invalidation via version/hash checking

### CLI Commands (`cmd/`)
- Each major command is a Cobra command
- Reusable runners in `runner/`
- Interactive mode in `interactive.go`

### Type Definitions
- API types in `api/types.go`
- Internal types colocated with logic
- Use `json` tags for API serialization

## Common Additions/Changes

### Adding a New CLI Command
1. Create handler in `cmd/` (e.g., `cmd_newcommand.go`)
2. Register with root command in `cmd.go`
3. Add tests in `cmd_newcommand_test.go`
4. Document in README if user-facing

### Adding a New API Endpoint
1. Add handler in `server/`
2. Register route in `server.go`
3. Add OpenAI compatibility if applicable
4. Include tests with HTTP client simulation

### Supporting a New Model Architecture
1. Create converter in `convert/convert_modelname.go`
2. Register in `convert.go`
3. Add tokenizer if needed
4. Include architecture-specific inference in `llm/`

## Repository-Specific Guidelines

### Multi-Platform Development
- Use platform-specific files: `file_unix.go`, `file_windows.go`, `file_darwin.go`
- Test on all platforms before submitting
- Use build tags: `//go:build darwin`

### Performance Considerations
- GPU memory is precious; implement proper cleanup
- Streaming responses for large outputs (don't buffer entire response)
- Benchmark changes affecting inference speed
- Profile memory leaks with `pprof`

### Security & Credentials
- Never commit `.env`, `.key`, `.pem` files
- Use environment variables for secrets
- See `SECURITY.md` for vulnerability reporting
- Files in `x/agent/` enforce credential file access controls

### Documentation
- Update `README.md` for user-facing changes
- Update `CONTRIBUTING.md` if process changes
- Keep `docs/` current with new features
- Use inline code comments for non-obvious logic

## Before Submitting a Pull Request

1. **Commit messages**: Follow `<package>: <description>` format
2. **Tests**: Include tests for behavior changes
3. **Build**: Ensure `go build ./cmd` succeeds
4. **Format**: Run `go fmt ./...`
5. **Lint**: Run `golangci-lint run`
6. **Cross-platform**: Test on macOS/Windows/Linux if affected
7. **Breaking changes**: Avoid changes breaking the REST API or CLI interface
8. **Dependencies**: Justify any new dependencies in PR description
9. **Documentation**: Update docs if behavior changes

## Review & Merge Criteria

### Mandatory Main Deployment Rule
- Every merge to `main` or direct push to `main` must trigger a production redeploy through the approved deployment workflow.
- A change is not considered fully complete until that redeploy succeeds for the resulting `main` commit.

**Ideal PRs** (reviewed quickly):
- Bug fixes with clear minimal changes
- Performance improvements with benchmarks
- Documentation updates
- Small refactorings improving clarity

**Harder to review** (may take longer):
- New features (add surface area, harder to maintain)
- Large refactorings
- Major documentation additions

**Unlikely to be merged**:
- Breaking API/CLI changes without discussion
- Features not aligned with Ollama's goals
- PRs without tests or clear rationale

## Getting Help

- **Issues**: Open a GitHub issue first for non-trivial changes
- **Discord**: Join [Discord](https://discord.gg/ollama) for community support
- **Development Docs**: See `docs/development.md` for deep dives
- **Contributing Guide**: See `CONTRIBUTING.md` for detailed guidelines

## Repository Information

- **Repository**: https://github.com/ollama/ollama.git
- **Language**: Go 1.24.1+
- **License**: MIT
- **Main Branch**: main
- **CI/CD**: GitHub Actions (`.github/workflows/`)

---

**Last Updated**: 2026-04-17
**Scope**: ollama/ollama repository only
