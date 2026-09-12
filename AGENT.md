# AGENTS.md

> **Purpose**: This document provides structured information for AI coding assistants (like Claude, GitHub Copilot, etc.) to understand the Ollama repository structure, conventions, and development workflows.

## Repository Overview

**Project Name**: Ollama  
**Repository**: https://github.com/ollama/ollama  
**Description**: Ollama enables users to run large language models (LLMs) locally. It supports models like Llama, Mistral, Gemma, DeepSeek, and many others across macOS, Linux, and Windows platforms.  
**Primary Language**: Go  
**License**: MIT License  
**Current Go Version Required**: 1.24.1 (1.22+ minimum supported)

## Project Structure

```
ollama/
├── api/              # API definitions and client implementations
├── app/              # Application-specific code (desktop app)
├── cmd/              # Command-line interface implementations
├── convert/          # Model format conversion utilities
├── discover/         # Hardware and runtime discovery
├── docs/             # Documentation files
├── format/           # Model format handlers
├── fs/               # File system utilities
├── harmony/          # Additional functionality modules
├── integration/      # Integration tests
├── kvcache/          # Key-value cache implementation
├── llm/              # LLM inference engine
├── ml/               # Machine learning backend infrastructure
│   └── backend/      # GGML backend for GPU acceleration
├── model/            # Model definitions and configurations
├── parser/           # Modelfile parser
├── readline/         # Terminal readline implementation
├── runner/           # Model runner implementations
├── scripts/          # Build and utility scripts
├── server/           # HTTP server implementation
├── template/         # Prompt template system
├── thinking/         # Thinking/reasoning functionality
├── types/            # Type definitions and utilities
└── version/          # Version information
```

## Tech Stack

- **Language**: Go 1.24.1 (1.22+ compatible)
- **GPU Acceleration**: 
  - CUDA (NVIDIA)
  - ROCm (AMD)
  - Metal (Apple Silicon)
  - GGML Backend (`ml/backend/ggml/`)
- **HTTP Framework**: `github.com/gin-gonic/gin` + standard `net/http`
- **CLI Framework**: `github.com/spf13/cobra`
- **Testing**: Go's standard testing package
- **Linting**: golangci-lint
- **Package Management**: Go Modules

## Development Environment Setup

### Prerequisites

1. **Install Go**: Version 1.24.1 or higher (1.22+ minimum)
   ```bash
   # macOS
   brew install go
   
   # Linux
   wget https://go.dev/dl/go1.24.1.linux-amd64.tar.gz
   sudo tar -C /usr/local -xzf go1.24.1.linux-amd64.tar.gz
   ```

2. **Clone the Repository**:
   ```bash
   git clone https://github.com/ollama/ollama.git
   cd ollama
   ```

3. **Install Dependencies**:
   ```bash
   go mod download
   ```

### Platform-Specific Requirements

**macOS**:
- Xcode Command Line Tools
- CMake (for building llama.cpp)

**Linux**:
- GCC or Clang
- CMake
- CUDA Toolkit (for NVIDIA GPU support)
- ROCm (for AMD GPU support)

**Windows**:
- Visual Studio 2019 or later with C++ tools
- CMake
- CUDA Toolkit (for NVIDIA GPU support)

## Build Commands

### Standard Build
```bash
# Build the main binary
go generate ./...
go build .

# Build for specific OS/architecture
GOOS=linux GOARCH=amd64 go build .
GOOS=darwin GOARCH=arm64 go build .
GOOS=windows GOARCH=amd64 go build .
```

### Development Build
```bash
# Build with debug symbols
go build -gcflags="all=-N -l" .
```

### Clean Build
```bash
go clean -cache
go build .
```

## Test Commands

### Run All Tests
```bash
go test ./...
```

### Run Tests with Verbose Output
```bash
go test -v ./...
```

### Run Tests with Coverage
```bash
go test -cover ./...

# Generate HTML coverage report
go test -coverprofile=coverage.out ./...
go tool cover -html=coverage.out -o coverage.html
```

### Run Specific Package Tests
```bash
go test ./server/...
go test ./api/...
```

### Run Integration Tests
```bash
go test ./integration/...
```

### Run Tests with Race Detector
```bash
go test -race ./...
```

### Benchmarks
```bash
go test -bench=. ./...
```

## Linting and Code Quality

### Run Linter
```bash
# Install golangci-lint if not already installed
go install github.com/golangci/golangci-lint/cmd/golangci-lint@latest

# Run linter
golangci-lint run

# Run linter with auto-fix
golangci-lint run --fix
```

### Linter Configuration
The project uses `.golangci.yaml` for configuration. Key linters enabled:
- `gofmt` - Code formatting
- `goimports` - Import organization
- `govet` - Static analysis
- `errcheck` - Unchecked error handling
- `ineffassign` - Unused variable assignments
- `staticcheck` - Advanced static analysis
- `misspell` - Common misspellings
- `gosec` - Security issues

### Common Linter Errors and Fixes
```bash
# Error: "unused variable"
# Fix: Remove unused variable or use _ for intentional ignoring
x := someFunc() // linter error
_ = someFunc()  // correct

# Error: "missing error check"
# Fix: Always check error returns
file, err := os.Open("test.txt")
if err != nil {
    return err
}

# Error: "package not imported"
# Fix: Run goimports to auto-fix
goimports -w .
```

### Running Pre-commit Checks
```bash
# Run all quality checks before committing
go fmt ./...
go vet ./...
golangci-lint run
go test ./...
```

### Format Code
```bash
# Format all Go files
go fmt ./...

# Using goimports (recommended)
go install golang.org/x/tools/cmd/goimports@latest
goimports -w .
```

### Vet Code
```bash
go vet ./...
```

## Coding Standards

### Go Style Guide
- Follow the official [Effective Go](https://golang.org/doc/effective_go.html) guidelines
- Use [Uber's Go Style Guide](https://github.com/uber-go/guide/blob/master/style.md) for additional best practices

### Naming Conventions
- **Packages**: Short, lowercase, single-word names (e.g., `server`, `api`, `model`)
- **Files**: Lowercase with underscores (e.g., `model_file.go`, `http_server.go`)
- **Functions**: CamelCase, exported functions start with uppercase (e.g., `CreateModel`, `parseModelfile`)
- **Variables**: CamelCase, descriptive names (e.g., `modelName`, `ctx`, `err`)
- **Constants**: CamelCase or ALL_CAPS for exported constants

### Error Handling Patterns
```go
// Pattern 1: Simple error check
if err != nil {
    return fmt.Errorf("operation failed: %w", err)
}

// Pattern 2: Error with context
if err := loadModel(ctx, name); err != nil {
    return fmt.Errorf("failed to load model %q: %w", name, err)
}

// Pattern 3: Error wrapping with additional data
if err := save(file); err != nil {
    slog.Error("failed to save file", "path", file, "error", err)
    return fmt.Errorf("save failed: %w", err)
}

// Pattern 4: Defer-based cleanup and error handling
func process(path string) error {
    file, err := os.Open(path)
    if err != nil {
        return fmt.Errorf("cannot open file: %w", err)
    }
    defer file.Close()  // Always runs, even on error
    
    // Process file...
    return nil
}
```

### Context Usage
```go
// Always pass context.Context as the first parameter
func ProcessModel(ctx context.Context, modelName string) error {
    // Check for cancellation
    select {
    case <-ctx.Done():
        return ctx.Err()
    default:
    }
    // ... rest of function
}
```

### Logging
```go
// Use structured logging with slog for consistency
import "log/slog"

slog.Info("loading model", "name", modelName, "size", modelSize)
slog.Error("failed to load model", "error", err, "name", modelName)

// Note: Some legacy code may use standard log package, but slog is preferred for new code
```

### Comments
```go
// Package-level comments
// Package server implements the HTTP server for Ollama.
package server

// Function comments for exported functions
// CreateModel creates a new model from the given Modelfile.
// It returns an error if the Modelfile is invalid or the model already exists.
func CreateModel(ctx context.Context, name string, modelfile string) error {
    // Implementation
}
```

### Comment Conventions
- **Package Comments**: Required for all packages; start with "Package <name>"
  ```go
  // Package server provides HTTP server implementation for Ollama.
  package server
  ```
- **Function/Method Comments**: Required for all exported functions; explain purpose, parameters, and return values
  ```go
  // LoadModel loads a model from disk and prepares it for inference.
  // Returns an error if the model doesn't exist or is corrupted.
  func LoadModel(ctx context.Context, name string) (*Model, error) {}
  ```
- **Type Comments**: Required for exported types
  ```go
  // ModelConfig holds configuration for a model instance.
  type ModelConfig struct {
      Name    string
      Layers  int
  }
  ```
- **Inline Comments**: Use `//` for single-line, `/* */` for multi-line (sparingly)
  ```go
  // Bad: Comments state the obvious
  i++ // increment i
  
  // Good: Explain why, not what
  i++ // skip hidden layers that don't require gradient updates
  ```
- **Deprecated Items**: Mark with `// Deprecated:` comment
  ```go
  // Deprecated: Use LoadModelV2 instead.
  func LoadModel(ctx context.Context, name string) error {}
  ```
- **TODO Comments**: Include context and responsible person if known
  ```go
  // TODO(john): Optimize memory usage for large batch sizes (issue #1234)
  ```

## Common Patterns

### API Request/Response Pattern
```go
type CreateRequest struct {
    Name      string `json:"name"`
    Modelfile string `json:"modelfile"`
    Stream    bool   `json:"stream"`
}

type CreateResponse struct {
    Status string `json:"status"`
}

func (s *Server) CreateHandler(w http.ResponseWriter, r *http.Request) {
    var req CreateRequest
    if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
        http.Error(w, err.Error(), http.StatusBadRequest)
        return
    }
    // Process request...
}
```

### Model Loading Pattern
```go
func LoadModel(ctx context.Context, name string) (*Model, error) {
    // Check if model exists
    manifest, err := GetManifest(name)
    if err != nil {
        return nil, fmt.Errorf("model not found: %w", err)
    }
    
    // Load model layers
    model := &Model{Name: name}
    for _, layer := range manifest.Layers {
        if err := model.LoadLayer(ctx, layer); err != nil {
            return nil, fmt.Errorf("failed to load layer: %w", err)
        }
    }
    
    return model, nil
}
```

### Streaming Response Pattern
```go
func (s *Server) GenerateHandler(w http.ResponseWriter, r *http.Request) {
    w.Header().Set("Content-Type", "application/x-ndjson")
    w.Header().Set("Transfer-Encoding", "chunked")
    
    encoder := json.NewEncoder(w)
    for token := range generateTokens() {
        if err := encoder.Encode(token); err != nil {
            return
        }
        if f, ok := w.(http.Flusher); ok {
            f.Flush()
        }
    }
}
```

## Documentation Standards

### Code Documentation
- All exported functions, types, and constants must have GoDoc comments
- Comments should explain "why" not just "what"
- Include examples for complex functions

### README Updates
- Update README.md for user-facing features
- Include usage examples
- Document new CLI commands or API endpoints

### API Documentation
- Document all API endpoints in `docs/api.md`
- Include request/response examples
- Specify error codes and meanings

## Testing Standards

### Unit Tests
```go
func TestCreateModel(t *testing.T) {
    tests := []struct {
        name      string
        modelName string
        modelfile string
        wantErr   bool
    }{
        {
            name:      "valid model",
            modelName: "test-model",
            modelfile: "FROM llama2",
            wantErr:   false,
        },
        {
            name:      "invalid modelfile",
            modelName: "test-model",
            modelfile: "INVALID",
            wantErr:   true,
        },
    }
    
    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            err := CreateModel(context.Background(), tt.modelName, tt.modelfile)
            if (err != nil) != tt.wantErr {
                t.Errorf("CreateModel() error = %v, wantErr %v", err, tt.wantErr)
            }
        })
    }
}
```

### Integration Tests
- Place in `integration/` directory
- Use `-tags=integration` for build constraint
- Test end-to-end workflows

### Test Coverage Goals
- Aim for >70% overall coverage
- Critical paths should have >90% coverage
- All exported functions should have tests

## Git Workflow

### Branch Naming
- `feature/description` - New features
- `fix/description` - Bug fixes
- `docs/description` - Documentation updates
- `refactor/description` - Code refactoring

### Commit Messages
Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
type(scope): subject

body (optional)

footer (optional)
```

Examples:
```
feat(api): add streaming support for chat endpoint
fix(server): resolve memory leak in model loading
docs(readme): update installation instructions
test(api): add tests for pull command
refactor(llm): optimize token generation loop
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `test`: Adding or updating tests
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `chore`: Build process or auxiliary tool changes

### Pull Request Process

1. **Before Creating PR**:
   - Run tests: `go test ./...`
   - Run linter: `golangci-lint run`
   - Format code: `go fmt ./...`
   - Update documentation if needed

2. **PR Title**: Follow commit message format
   ```
   feat(api): add new endpoint for model comparison
   ```

3. **PR Description Template**:
   ```markdown
   ## Description
   Brief description of changes
   
   ## Type of Change
   - [ ] Bug fix
   - [ ] New feature
   - [ ] Breaking change
   - [ ] Documentation update
   
   ## Testing
   - [ ] Unit tests added/updated
   - [ ] Integration tests added/updated
   - [ ] Manual testing performed
   
   ## Checklist
   - [ ] Code follows style guidelines
   - [ ] Self-review completed
   - [ ] Comments added for complex code
   - [ ] Documentation updated
   - [ ] No new warnings generated
   - [ ] Tests pass locally
   ```

4. **Review Process**:
   - At least one approval required from maintainers
   - Address all review comments
   - Keep PR scope focused and reasonable size

## Dependencies Management

### Adding Dependencies
```bash
# Add a new dependency
go get github.com/example/package

# Add specific version
go get github.com/example/package@v1.2.3

# Update go.mod and go.sum
go mod tidy
```

### Updating Dependencies
```bash
# Update all dependencies
go get -u ./...

# Update specific dependency
go get -u github.com/example/package

# Clean up
go mod tidy
```

### Vendoring (if used)
```bash
go mod vendor
```

### Build Flags and Environment Variables

```bash
# Common build flags
go build -ldflags="-s -w"                    # Strip symbols for smaller binary
go build -ldflags="-X main.Version=v1.0.0"   # Embed version at build time

# Common environment variables for building
GO111MODULE=on go build .                    # Ensure module mode (default in 1.24)
CGO_ENABLED=1 go build .                     # Enable C bindings (needed for GPU)
GOOS=linux GOARCH=amd64 go build .           # Cross-compile
GODEBUG=gocachehash=1 go build .             # Debug cache misses
```

### Checking Build Environment

```bash
# View all Go environment variables
go env

# Check specific variables
go env GOVERSION
go env GOPATH
go env GOPROXY
go env GOFLAGS

# Verify build toolchain
go version
go list -m all      # List all dependencies
```

## Performance Considerations

### CPU Optimization
- Use goroutines for concurrent operations
- Implement worker pools for parallel processing
- Profile with `pprof` to identify bottlenecks

### Memory Management
- Use sync.Pool for frequently allocated objects
- Avoid memory leaks by closing resources
- Profile memory usage with `go test -memprofile`

### GPU Utilization
- Batch requests when possible
- Keep model loaded in GPU memory
- Use appropriate batch sizes for inference

## Security Best Practices

### Input Validation
```go
// Validate and sanitize all inputs
func ValidateModelName(name string) error {
    if name == "" {
        return errors.New("model name cannot be empty")
    }
    if matched, _ := regexp.MatchString(`^[a-zA-Z0-9._-]+$`, name); !matched {
        return errors.New("invalid model name format")
    }
    return nil
}
```

### Secure File Operations
```go
// Use secure file permissions
os.WriteFile(path, data, 0600)

// Validate file paths to prevent directory traversal
filepath.Clean(userPath)
```

### API Security
- Validate all API inputs
- Implement rate limiting
- Use HTTPS in production
- Sanitize error messages (don't leak sensitive info)

## Release Process

### Version Numbering
- Follow Semantic Versioning (SemVer)
- Format: `vMAJOR.MINOR.PATCH`
- Example: `v0.1.25`

### Building Releases
```bash
# Build for all platforms
./scripts/build.sh

# Build specific platform
GOOS=linux GOARCH=amd64 go build -o ollama-linux-amd64
```

## Useful Resources

### Official Documentation
- [Ollama Documentation](https://github.com/ollama/ollama/tree/main/docs)
- [API Documentation](https://github.com/ollama/ollama/blob/main/docs/api.md)
- [Modelfile Documentation](https://github.com/ollama/ollama/blob/main/docs/modelfile.md)

### Go Resources
- [Effective Go](https://golang.org/doc/effective_go.html)
- [Go Code Review Comments](https://github.com/golang/go/wiki/CodeReviewComments)
- [Uber Go Style Guide](https://github.com/uber-go/guide/blob/master/style.md)

### Community
- [GitHub Issues](https://github.com/ollama/ollama/issues)
- [GitHub Discussions](https://github.com/ollama/ollama/discussions)
- [Discord Server](https://discord.gg/ollama)

## Common Issues and Solutions

### Build Issues
```bash
# Clean cache and rebuild
go clean -cache -modcache
go mod download
go generate ./...
go build .
```

## Common Issues and Solutions

### Build Issues
```bash
# Clean cache and rebuild
go clean -cache -modcache
go mod download
go generate ./...
go build .
```

### Build Failure Debugging

#### Issue: "undefined reference to..." or linker errors
```bash
# Check if go generate was run
go generate ./...

# Rebuild everything from scratch
go clean -cache
go build .

# For GPU-related errors, ensure CUDA/ROCm is installed
echo $LD_LIBRARY_PATH  # Check library path on Linux
```

#### Issue: "import cycle detected"
```bash
# Visualize import dependencies
go mod graph | grep <package-name>

# Solution strategies:
# 1. Move shared types to a new package (e.g., types/)
# 2. Use interfaces to decouple dependencies
# 3. Refactor to remove circular dependency
```

#### Issue: "module not found" or dependency errors
```bash
# Update go.mod and go.sum
go mod tidy

# Check for replaced modules in go.mod
cat go.mod | grep -i replace

# Verify specific dependency
go mod why github.com/example/package
```

#### Issue: CGO build failures (for GPU/ML code)
```bash
# Disable CGO for testing (may limit functionality)
CGO_ENABLED=0 go build .

# Force rebuild with CGO (requires C compiler)
CGO_ENABLED=1 go build .

# Check CGO-related environment
go env CGO_ENABLED
go env CC
go env CXX
```

#### Issue: "permission denied" on build scripts
```bash
# Make scripts executable
chmod +x ./scripts/*.sh

# Or run with bash
bash ./scripts/build.sh
```

### Debugging Build Output
```bash
# Verbose build output
go build -v

# Very verbose with all executed commands
go build -x

# Build with tags
go build -tags=integration .

# Check what will be built
go list -m all
```

### Test Failures
```bash
# Run tests with verbose output
go test -v -run TestName ./...

# Run specific test
go test -v -run TestCreateModel ./server/

# Debug with race detector
go test -race ./...

# Get detailed error output
go test -v -count=1 ./...  # disable test caching
```

### Debugging with Delve

```bash
# Install delve
go install github.com/go-delve/delve/cmd/dlv@latest

# Debug a test
dlv test ./server -- -test.v

# Debug the main application
dlv debug . -- serve

# Delve interactive commands
(dlv) break main.main              # Set breakpoint
(dlv) break file.go:10             # Break at line
(dlv) condition 1 err != nil       # Conditional breakpoint
(dlv) continue                      # Resume execution
(dlv) next                          # Next line
(dlv) step                          # Step into function
(dlv) print variableName            # Print variable value
(dlv) backtrace                     # Show stack trace
(dlv) goroutines                    # List goroutines
```

### Runtime Debugging

```bash
# Enable verbose/debug logging
OLLAMA_DEBUG=1 ./ollama serve

# Set log level
OLLAMA_DEBUG=1 OLLAMA_LOG_LEVEL=debug ./ollama serve

# Log to file
OLLAMA_DEBUG=1 ./ollama serve > debug.log 2>&1

# Use structured logging to filter output
OLLAMA_DEBUG=1 ./ollama serve 2>&1 | grep "component:server"
```

### Profiling for Performance Issues

```bash
# CPU profiling
go test -cpuprofile=cpu.prof ./...
go tool pprof cpu.prof

# Memory profiling
go test -memprofile=mem.prof ./...
go tool pprof mem.prof

# Interactive pprof commands
(pprof) top              # Show top consumers
(pprof) list function    # Show function source with stats
(pprof) web              # Generate graph (requires graphviz)
```

### Import Cycle Errors
- Refactor code to break circular dependencies
- Move shared types to a common package
- Use interfaces to decouple packages

## AI Agent Instructions

When generating code for this repository:

1. **Always use Go 1.24+ features** when appropriate
2. **Include error handling** for all operations that can fail
3. **Add context.Context** as the first parameter for functions that may block
4. **Write tests** for new functions using table-driven test pattern
5. **Document exported symbols** with GoDoc comments
6. **Follow existing patterns** in the codebase for consistency
7. **Use structured logging** with `log/slog` package (preferred) or `log` package (legacy)
8. **Validate inputs** before processing
9. **Return wrapped errors** using `fmt.Errorf` with `%w`
10. **Keep functions focused** - single responsibility principle

### Example Code Generation Template

```go
// Package description
package packagename

import (
    "context"
    "fmt"
    "log/slog"
)

// FunctionName performs a specific task.
// It returns an error if the operation fails.
func FunctionName(ctx context.Context, param1 string) error {
    // Validate inputs
    if param1 == "" {
        return fmt.Errorf("param1 cannot be empty")
    }
    
    // Check context cancellation
    select {
    case <-ctx.Done():
        return ctx.Err()
    default:
    }
    
    // Perform operation with structured logging
    slog.Info("performing operation", "param", param1)
    
    // Handle errors with context wrapping
    if err := someOperation(); err != nil {
        return fmt.Errorf("failed to perform operation: %w", err)
    }
    
    return nil
}
```

---

**Maintainers**: Ollama Team  
**Contributing**: Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.
