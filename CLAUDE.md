# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common commands

```shell
# Run the server (compiles native code via CGO on first invocation)
go run . serve

# CLI entry point (same binary, Cobra subcommands: serve, run, pull, push, create, list, rm, ps, ...)
go run . <subcommand>

# Build a release binary
go build .

# Unit tests only (integration tests are tag-gated)
go test ./...

# Single package / single test
go test ./server/...
go test -run TestName ./server

# Integration tests (require a built `ollama` binary at repo root)
go build .
go test -tags=integration -v -count 1 -timeout 15m ./integration/
OLLAMA_TEST_MODEL=mymodel go test -tags=integration -v ./integration/
OLLAMA_TEST_EXISTING=1 OLLAMA_HOST=... go test -tags=integration ./integration/

# Model-matrix integration tests (slower)
go test -tags=integration,models -timeout 60m ./integration/

# Enable synctest package (required for a small number of time-sensitive tests — CI runs with it)
GOEXPERIMENT=synctest go test ./...

# Lint — golangci-lint config lives in .golangci.yaml
golangci-lint run

# Format
gofumpt -l -w .

# Native/GPU acceleration build (GGML + optional CUDA/ROCm/Vulkan/MLX)
cmake -B build
cmake --build build --config Release     # Windows needs --config Release
# then `go run . serve` picks up libs from build/lib/ollama

# Force full rebuild after CGO struct layout changes
go clean -cache
```

Only build with `cmake` when you need GPU acceleration libraries or have edited anything under `ml/backend/ggml/`, `llama/`, or `x/mlxrunner/`. Pure-Go changes do not need cmake.

## Vendored llama.cpp / ggml

`llama/vendor/`, `llama/llama.cpp/`, and `ml/backend/ggml/ggml/` are synced from upstream `ggml-org/llama.cpp` via patches. Never edit those directories directly — edit in `llama/vendor/` and regenerate patches:

```shell
make -f Makefile.sync apply-patches          # apply our patches to tracking tree
make -f Makefile.sync format-patches sync    # regenerate patches + rsync into repo
make -f Makefile.sync clean apply-patches    # reset tracking tree
```

To move to a newer upstream commit, bump `FETCH_HEAD` in `Makefile.sync`, then `apply-patches` (resolving any conflicts in `llama/vendor/` + `git am --continue`), then `format-patches sync`.

## Architecture

Ollama is a single Go binary that acts as both CLI and HTTP server. The server loads a model once, then spawns a subprocess (the "runner") that owns the native inference loop.

### Top-level flow
1. `main.go` → `cmd.NewCLI()` dispatches Cobra subcommands.
2. `ollama serve` → `server/routes.go` starts a Gin HTTP server on `OLLAMA_HOST` (default `127.0.0.1:11434`). Routes cover the native API (`/api/generate`, `/api/chat`, `/api/embed`, `/api/pull`, `/api/create`, ...) and an OpenAI-compatible layer (`openai/`).
3. Requests go through `server/sched.go`, which schedules loaded models across available GPUs (see `discover/`), and `llm/server.go`, which spawns a runner subprocess for each loaded model.
4. The runner binary is the **same `ollama` executable** re-invoked with a `runner` arg (`runner/runner.go`). `Execute` picks a backend based on flags: `--ollama-engine` (native Go ML), `--mlx-engine` (Apple/CUDA via MLX), `--imagegen-engine`, or default llama.cpp.

### Three inference backends
All three live side-by-side. A model's format dictates which runner loads it.

- `runner/llamarunner/` — CGO bindings to vendored llama.cpp (`llama/`). Handles GGUF via llama.cpp's graph execution. Default path for most models.
- `runner/ollamarunner/` — Native-Go inference engine. Uses `ml/` abstractions (`ml/backend/ggml/` does the heavy lifting in C via ggml). Model architectures implemented in `model/models/` (e.g. `model/models/gemma3/`, `model/models/qwen3/`). Selected with `--ollama-engine`.
- `x/mlxrunner/` — MLX backend (Metal on Apple Silicon, CUDA 13+ on Linux/Windows). Loads safetensors. Requires `cmake -B build --preset MLX` (see `docs/development.md`).

When implementing a **new model architecture** for the native engine, put it under `model/models/<arch>/` implementing `model.Model` (see `model/model.go`) and wire it in via the registration list. Vision/audio models additionally implement `MultimodalProcessor`. Run `OLLAMA_TEST_MODEL=<name> go test -tags=integration ./integration/` to validate.

### Key packages

| Package | Purpose |
|---|---|
| `api/` | Public client + request/response types (Go SDK, also serialized to TypeScript via `typescriptify`) |
| `server/` | HTTP routes, scheduler, model resolver, blob upload/download, prompt building |
| `openai/` | OpenAI-compatible `/v1/*` wrapper around native API |
| `cmd/` | CLI subcommands, interactive chat TUI (`cmd/tui/` — Bubbletea), `launch` integrations |
| `runner/` | Inference subprocess dispatcher; `llamarunner/` + `ollamarunner/` are the two first-party runners |
| `llm/` | Spawns and supervises runner subprocesses, detects GPU libs |
| `discover/` | GPU enumeration + capability detection (CUDA, ROCm, Metal, Vulkan, MLX) |
| `ml/` | Native Go ML tensor/backend abstraction (`ml.Backend`, `ml.Tensor`); `ml/backend/ggml/` is the C backend |
| `model/` | `Model` interface, per-architecture implementations, multimodal preprocessing, chat templates, parsers/renderers |
| `llama/` | CGO wrapper over vendored llama.cpp (not upstream-pristine — carries patches in `llama/patches/`) |
| `convert/` | Model conversion (HF/safetensors → GGUF) |
| `fs/ggml/` | GGUF file-format reader/writer |
| `template/` + `model/renderers/` + `model/parsers/` | Chat prompt templating (Go text/template) and streaming output parsing (e.g. Harmony, thinking tags) |
| `thinking/` + `tools/` + `harmony/` | Reasoning-trace handling, function-calling, Harmony message format |
| `kvcache/` | KV-cache management for the native runner |
| `tokenizer/` | BPE/SentencePiece tokenizers for native models |
| `envconfig/` | All `OLLAMA_*` env vars (single source of truth — don't read env vars elsewhere) |
| `app/` | Desktop app shell (macOS tray + Windows tray app wrapping the binary); has its own React UI under `app/ui/` |
| `x/` | Newer/experimental subpackages (MLX runner, imagegen, agent features, cloud server helpers) |

### Testing conventions

- Unit tests live beside code (`foo.go` + `foo_test.go`). Focus tests on behavior, not implementation (per `CONTRIBUTING.md`).
- Integration tests in `integration/` are gated by `-tags=integration`. Some need extra tags (`models`, etc.) — grep `go:build` headers. They shell out to a built `ollama` binary at the repo root on Unix; on Windows they require `OLLAMA_HOST` to point at an already-running server.
- The CI matrix builds Linux (CPU / CUDA / ROCm / Vulkan / MLX-CUDA), macOS, and Windows. `changes` job in `.github/workflows/test.yaml` skips vendored-code rebuilds when the llama.cpp/ggml tree is unchanged.

### Commit message convention

Per `CONTRIBUTING.md`: `<package>: <short lowercase continuation>` — e.g. `server: fix nil prompt in embed route`, `llm/backend/mlx: support the llama architecture`. Title is a continuation of "This changes Ollama to …". Avoid `feat:`, `fix:`, `chore:` prefixes.
