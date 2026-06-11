# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

This is the Ollama source tree (Go module `github.com/ollama/ollama`). This worktree was created at **v0.30.2** but has since been merged up to **latest main** (branch `llama-server-notes`, v0.30.5+, as of 2026-06-04). See `.claude/docs/` (start at `README.md`) for the local build/testing context this worktree was created for (the 0.30.x `llama-server` / GGUF packaging investigation); root `NOTES.md` is now just a pointer there.

**GitHub issue/PR tracking** for upstream ollama/ollama (and downstream Homebrew/Termux/llama.cpp) — the status of our open PRs/issues, the watch-list, and dated re-pull snapshots — lives in **`.claude/docs/github/github-tracking.md`** (the cross-repo hub) with per-repo detail in `.claude/docs/github/ollama-ollama.md`. Update these when re-checking issue/PR status; do **not** post anything to GitHub without explicit ask (low profile on this repo).

## Build & run

```shell
go run . serve            # pure-Go iteration against an already-built native payload
go build .                # build the ./ollama binary (integration tests expect it at repo root)
```

Native code is built with CMake, **not** `go generate`/Make. The build installs the native runtime payload under `build/lib/ollama` (where the server discovers it at runtime):

```shell
cmake -B build .                        # macOS arm64 → Metal; everything else → CPU-only
cmake --build build --parallel 8
./ollama serve
```

- GPU backends are opt-in except on macOS arm64: `cmake -B build . -DOLLAMA_LLAMA_BACKENDS="cuda_v13;vulkan"` (values: `cuda_v12`, `cuda_v13`, `rocm_v7_1`, `rocm_v7_2`, `vulkan`, `cuda_jetpack5`, `cuda_jetpack6`).
- MLX backends: enabled by default on macOS arm64; elsewhere `-DOLLAMA_MLX_BACKENDS=cuda_v13`.
- CGO is involved. After changing native code, if you hit unexplained crashes run `go clean -cache` to force a full native rebuild.
- Library discovery searches `../lib/ollama`, `./lib/ollama`, `.`, `build/lib/ollama`, and `dist/<platform>/lib/ollama`. Without these the server runs with no acceleration.
- `LLAMA_CPP_VERSION` pins the upstream llama.cpp commit (currently `b9509`); local patches live under `llama/`.

## Test & lint

```shell
go test ./...                                   # unit tests only (integration is build-tagged off)
go test ./server/ -run TestName                 # a single package / test
go test -tags=integration ./...                 # integration tests (need a built ./ollama at repo root)
go test -tags=integration,models ./...          # broad model suite; expect 60m+
golangci-lint run                               # config in .golangci.yaml; gofmt/gofumpt/goimports enforced
```

Integration tests (`integration/`, `//go:build integration`) start a server on a random port by default; set `OLLAMA_TEST_EXISTING` to run against a running server (honors `OLLAMA_HOST`), `OLLAMA_TEST_MODEL` to target a specific model, `OLLAMA_TEST_LOG_SERVER=1` to dump server logs.

## Commit messages

Format: `<package>: <short description>`, lowercase, continuing "This changes Ollama to…". The package is the most-affected Go package (or directory/root-file name if non-Go). E.g. `llm/backend/mlx: support the llama architecture`. Avoid Conventional-Commits prefixes (`feat:`, `fix:`, `chore:`).

## Architecture

Single Go binary (`main.go` → `cmd.NewCLI()`, cobra). One process can act as the CLI client, the HTTP server (`ollama serve`), or a hidden inference `runner` subprocess — selected by subcommand.

**Request flow:** client → HTTP API (`server/routes.go`, gin) → scheduler → inference runner subprocess → response (streamed). The server never links the inference engines directly; it spawns subprocesses and talks to them over HTTP/stdio.

- **`server/`** — HTTP handlers and the heart of the daemon. `routes.go` wires both the native Ollama API (`/api/generate`, `/api/chat`, `/api/embed`, `/api/pull`, `/api/show`, …) and OpenAI/Anthropic-compatible routes (`/v1/chat/completions`, `/v1/responses`, `/v1/messages`, `/v1/images/...`). `sched.go` is the scheduler: it decides which models are resident, allocates GPU/VRAM, manages concurrency, and starts/stops runners. `images.go`/`model.go`/`create.go`/`download.go` handle the model store (manifests + blobs).
- **Inference engines** are separate subprocesses the scheduler launches:
  - **GGUF models → external `llama-server`** (upstream llama.cpp). Managed from `llm/` (`NewLlamaServer` in `llm/llama_server.go` spawns and supervises the `llama-server` binary from the native payload). This is the path the local `NOTES.md` work concerns.
  - **MLX / safetensors → native `x/mlxrunner`**, invoked as the hidden subcommand `ollama runner --mlx-engine` (dispatch in `runner/runner.go`).
  - **Image generation → `x/imagegen`**, via `ollama runner --imagegen-engine`.
  - There is also a native pure-Go ML backend (`ml/`, `ml/backend/ggml`) with Go model implementations under `model/models/` (e.g. `gemma4`, `laguna`) and `model/` (input processing, parsers, renderers).
- **`middleware/`** — translates OpenAI (`openai.go`) and Anthropic (`anthropic.go`) request/response shapes to/from Ollama's internal API; mounted as gin middleware on the `/v1` and `/v1/messages` routes.
- **`api/`** — the Go API client and shared request/response types (`types.go`); also consumed by the CLI handlers in `cmd/`.
- **`cmd/`** — CLI command implementations. `cmd/launch/` powers `ollama launch <integration>` (Claude Code, Codex, Copilot, OpenClaw, OpenCode, Droid, …) by writing each tool's config to point at the local server. `cmd/tui/` is the interactive terminal UI; `cmd/interactive.go` the `ollama run` REPL.
- **`convert/`** — converts safetensors/other source formats into GGUF (one `convert_<arch>.go` per architecture). **`fs/ggml`** reads/writes GGUF.
- **`x/`** — newer/experimental subsystems (`mlxrunner`, `imagegen`, `safetensors`, `tokenizer`, `transfer`, `agent`, `server`, …).
- Supporting packages: `discover/` (GPU/device probing, also the hidden `gpu-discover` subcommand), `template/` & `thinking/` & `tools/` & `harmony/` (prompt/template/tool-call & reasoning handling), `kvcache/`, `tokenizer/`, `envconfig/` (all `OLLAMA_*` env vars), `auth/`.

## Conventions

- New API surface (fields, env vars, endpoints) is treated as a long-term maintenance/compat cost — backwards compatibility of the API (including the OpenAI-compatible one) is not broken. Discuss non-trivial changes via an issue before a PR (see `CONTRIBUTING.md`).
- Add dependencies sparingly and justify them.
- Tests should exercise behavior, not implementation.
