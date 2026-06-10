# Detailed Technical Review Report: Adding gRPC Support to the ollamas Fork (feature/grpc-initial branch)

**Date of review**: 2026-06-03. This review was performed against the source tree in this repository.  
**Scope**: Exhaustive hands-on static analysis via `list_dir`, `grep` (with path/glob limits), `read_file` (full and offset/limited chunks across dozens of files), targeted `run_terminal_command` (git, find, grep -n, ls), and selective MCP GitHub search via `search_tool` + `use_tool` (for upstream context only). No files were created or modified. All exploration tracked via internal `todo_write` (9 phases, marked complete sequentially; started broad with root + git, narrowed to key files like `server/routes.go`, cross-checked callers/tests/build). Branch confirmed `feature/grpc-initial` with (near-)clean tree (only local dotfiles ignored in "clean" claim). No pre-existing gRPC code (confirmed via targeted greps returning 0 matches in `**/*.go` for `grpc|gRPC|google.golang.org/grpc` in source; only transitive in `go.sum` + tokenizer testdata + sentencepiece generated pb).

**Summary of findings**: The codebase is a faithful but extended fork of ollama/ollama (heavy recent llama.cpp/CMake activity, cloud/remote model features via `internal/modelref` + `model_resolver.go`, experimental runners in `x/`). HTTP API surface (gin-based) is centralized and the *only* production transport. No multi-listener, cmux, or gRPC scaffolding. Request path is tightly coupled to `gin.Context` (bind + writer + abort + streaming channels). Core business logic (model resolution, scheduling, prompt rendering, inference dispatch) lives in `server/` and is reusable in-package. Adding gRPC is very feasible with minimal disruption to existing HTTP users if done via a thin protocol adapter + shared scheduler/*Server* instance. Separate ports (easiest, lowest risk) or later cmux are viable. **Single best entrypoint identified below**.

Upstream (via GitHub MCP searches on `ollama/ollama` for grpc/gRPC/protobuf): No active gRPC API PRs or issues proposing server-side gRPC (false positives only; e.g., one old TLS PR #5912 mentions "pkcs8 ... required by grpc" in its test script for mTLS certs; searches returned unrelated bugs/logs). No standardized upstream pattern yet—this fork can define it. (Searches used `grok_com_github__search_pull_requests` / `search_issues` after `search_tool` discovery.)

---

### 1. Current Architecture (Textual Diagram of Request Path + Lifecycle)

```
[CLI / SDK / OpenAI/Anthropic clients]
          |
          | HTTP (default http://127.0.0.1:11434 or OLLAMA_HOST)
          v
cmd/cmd.go:1996 RunServer()
  |
  +-- net.Listen("tcp", envconfig.Host().Host)  // 11434 default; supports scheme/host/port/path
  |
  v
server/routes.go:1917 Serve(ln net.Listener) error
  |
  +-- slog.SetDefault(...) + envconfig.Values() logging
  +-- fixBlobs + optional PruneLayers + manifest prune (unless OLLAMA_NOPRUNE)
  +-- s := &Server{ addr: ln.Addr(), modelCaches: newModelCaches(), ... }  // 1951
  +-- (useClient2 exp) ollama.DefaultRegistry()
  +-- h, _ := s.GenerateRoutes(rc)  // 1968  <<-- gin setup + all routes + middleware
  +-- http.Handle("/", h)  // onto DefaultServeMux (for pprof too)
  +-- ctx setup + sched := InitScheduler(schedCtx) ; s.sched = sched ; s.modelCaches.Start
  +-- discover.GPUDevices + LogDetails + VRAM-tier defaultNumCtx (4k/32k/256k)
  +-- signal handler (SIGINT/TERM) -> srvr.Close() + sched.unloadAllRunners()
  +-- s.sched.Run(schedCtx)
  +-- image.RegisterFormat("webp"...)
  +-- srvr := &http.Server{Handler: nil} ; srvr.Serve(ln)  // 2033 (blocks)
          |
          v
  (in GenerateRoutes:1794)
  r := gin.Default()
  r.Use(cors, allowedHostsMiddleware(addr))
  // Core /api/* (local + some cloud passthrough)
  r.POST("/api/chat", withLogging, s.ChatHandler)  // 1862
  r.POST("/api/generate", ...)  // 1861
  r.POST("/api/embed", s.EmbedHandler)
  r.GET("/api/tags", s.ListHandler) ...
  r.POST("/api/pull", ...) , /ps, /version, /blobs, /create, experimental/...
  // OpenAI compat (/v1/*)
  r.POST("/v1/chat/completions", cloudPassthroughMW, middleware.ChatMiddleware(), s.ChatHandler)
  ... /v1/completions, /embeddings, /models, /responses, /images/*, /audio/transcriptions
  // Anthropic
  r.POST("/v1/messages", ..., middleware.AnthropicMessagesMiddleware(), s.ChatHandler)
  return r, nil
          |
          v
  Handler (e.g. ChatHandler:2422):
    bind api.ChatRequest (c.ShouldBindJSON)
    modelRef, _ := parseAndValidateModelRef (server/model_resolver.go:36; supports :cloud suffix via internal/modelref)
    if cloud/remote -> proxyCloudJSONRequest (or legacy) using api.NewClient
    m, _ := GetModel (server/images.go:622)
    if keepalive=0 expire etc.
    caps + scheduleRunner:205 (ctx, name, caps, opts, keepalive) ->
      GetModel + CheckCapabilities
      s.sched.getRunner(...)  // server/sched.go:165 (queues LlmRequest)
      returns llm.LlamaServer (runnerRef.llama), *Model, *api.Options
    render prompt (chatPrompt, template, parsers, thinking, tools, harmony for gpt-oss etc.)
    if native (MLX or rendered) -> handleNativeChat
    else:
      ch := make(chan any)
      go func() {
        defer close(ch)
        for { ... structured outputs dance ...
          r.Completion(ctx, llm.CompletionRequest{...}, func(llm.CompletionResponse){ ch <- api.ChatResponse{...} })
        }
      }()
      writeChatResponse(c, req, ch)  // 2917
        if !stream { collect + c.JSON } else { streamResponse(c, ch) }
  streamResponse:2070 (c.Stream, ndjson, special gin.H{"error", "status"} handling, flush)
  (similar for GenerateHandler:256 ~751, Embed:754, Pull/Push use waitForStream, List/Ps/Show etc.)

  Scheduler (sched.go:61): pending/finished/expired chans, loaded map, loadFn, newServerFn=llm.NewLlamaServer
    getRunner -> LlmRequest -> processPending -> load (evict, VRAM calc, mmap, OOM retry paths, spread etc.)
    -> llamaServerRunner (llm/llama_server.go:112) : exec subprocess (ollama runner or llama-server bin), find free TCP port, http.Client to http://127.0.0.1:port (semaphore for NumParallel), Completion/Embedding/ etc. POSTs with streaming callbacks, ctx cancel propagates, status polling, log parsing for VRAM/mem.

  Down to: discover/ (GPU probe + llama_server.go), llm/ (NewLlamaServer, llama binary mgmt via CMake-built llama.cpp server), runner/ (for --imagegen/--mlx), x/mlxrunner + x/imagegen (alt runners with their own http servers on random ports).

  Cloud bits: internal/cloud (NoCloud env), cloud_proxy.go (signing with ed25519 ssh key from ~/.ollama/id_ed25519, zstd, proxy to ollama.com), passthrough MVs in routes, model_resolver.

Other paths (pull/push/create use registry + progress chans; blobs are direct).

**Key observation**: All inference + most admin flows converge on `*Server` methods + `*Scheduler` + `llm.LlamaServer` interface. Transport (gin + body rewrite hacks in middleware/openai.go + anthropic.go) is the only variable. OpenAI/Anthropic are *adapters* that mutate c.Request.Body + c.Writer then c.Next() into the *same* core handlers.

---

### 2. Entrypoints + Lifecycle (Code Pointers)

**Primary server start**:
- `cmd/cmd.go` (around `RunServer`): the listener + serve entrypoint. Also related cloud keypair initialization.
- `main.go`: thin CLI entry (`cmd.NewCLI().Execute`).
- `NewCLI` (~2334): wires "serve" (alias start) -> RunServer; PreRun checks for other cmds use heartbeat + auto background `startApp` (via app/ or exec).
- Desktop: `app/server/*.go` (manages `ollama serve` subprocess for tray/app); `app/cmd/app/app.go:284` (separate *UI* http server on its own port/ln, not inference); `cmd/launch/` (TUI selectors, no direct serve); `cmd/start*.go` + background_*.go.

**Core server surface**:
- `server/routes.go` (around `Serve`): central lifecycle (server creation, scheduler init, route registration on the mux, GPU discovery, signal handling). This is the primary place to add gRPC listener wiring.
- `server/routes.go` (around `GenerateRoutes`): all HTTP route registration (core endpoints + /v1 compat + middleware). gRPC registration should happen alongside or via a parallel path on the same `*Server` instance.
- `Server` struct (routes.go:100): tiny (`addr net.Addr; sched *Scheduler; defaultNumCtx int; requestLogger; modelCaches`). No transport inside.
- Handlers (examples): `ChatHandler:2422`, `GenerateHandler:256`, `EmbedHandler:754`, `ListHandler:1585`, `PsHandler:2239`, `PullHandler:1049`, `StatusHandler:2119` etc. All `*gin.Context`.
- `scheduleRunner:205` (private helper used by handlers).
- Scheduler: `server/sched.go:91` (`InitScheduler`), `getRunner:165`, `load:492` (complex; delegates to `llm.NewLlamaServer`).
- Other server files: `images.go` (GetModel:622, Model struct:63), `model_resolver.go:36` (parse + cloud/local), `cloud_proxy.go`, `auth.go`, create.go, etc.
- `llm/server.go:59` (`type LlamaServer interface { Completion, Chat, Embed, Load, ... }`); impl in `llama_server.go`.
- Env: `envconfig/config.go:22` (`Host()` parses OLLAMA_HOST; default 127.0.0.1:11434; also KeepAlive, MaxQueue, NumParallel, UseAuth, NoCloud, etc.). No gRPC yet. `ConnectableHost` for clients.
- Health/version: root `/`, `/api/version`, `/api/status` (cloud disabled flag), heartbeat in client.
- Auth/obs: `auth/auth.go` (ed25519 ssh keys + Sign/Nonce for cloud/registry); `logutil/logutil.go` (slog text + TRACE); `server/auth.go` (registry challenges).
- Middleware: `middleware/openai.go` (body rewrite + *Writer interceptors for response conversion; Chat/Completions/Embed/List etc.), similar for anthropic + cloud passthroughs.
- Client compat: `api/client.go:81` (NewClient + FromEnvironment; http + ndjson streaming; signs for ollama.com if UseAuth); `api/types.go` (all request/response structs; the source of truth).
- Experimental/other servers: `x/imagegen/server.go` (subprocess http on random port for imagegen runner), `x/mlxrunner/server.go` (similar), `runner/runner.go` (dispatch), `cmd/launch/*`.

**Callers of server logic**: Almost exclusively internal to `server/` + single external import in `cmd/cmd.go:54`. Integration tests (`integration/utils_test.go:457`) exec the *binary* (`ollama serve` with OLLAMA_HOST + mutex for serial); no direct `server.Serve` in _test.go files. Easy to evolve.

**Build/tests**: Flat layout (top-level `server/`, `llm/`, `api/`, `discover/`, `model/`, `fs/`, `x/`, `internal/` limited). Heavy `CMakeLists.txt` + `llama/` + scripts/build_*.sh for llama.cpp (C++ side untouched by gRPC). Tests: unit + `integration/` (build-tagged, spawn binary), many `*_test.go`. `go.mod`: gin, cobra, protobuf (google.golang.org/protobuf + indirect golang/protobuf), no grpc.

**Existing proto**: Only `convert/sentencepiece_model.proto` (proto2, LITE; generated `convert/sentencepiece/sentencepiece_model.pb.go` using `google.golang.org/protobuf/proto`; used only in `convert/tokenizer_spm.go` + tests for SPM tokenizer. No buf/protoc in build; committed generated. No `//go:generate` for API protos.

---

### 3. Recommended Entrypoint + Integration Approaches (Pros/Cons)

**The SINGLE best, most efficient + standardized entrypoint**: Extend the existing `server.Serve` lifecycle + `*Server` (the natural "service" holder) rather than GenerateRoutes or a completely separate path.

- **Primary plug point**: `server/routes.go` (`Serve` + supporting inits for Server creation, scheduler, `GenerateRoutes`). Add gRPC listener wiring + registration after core init but before/parallel to HTTP serve (e.g. via a `RegisterGRPC` helper or `ServeGRPC` that reuses the `*Server` instance).
- **Wiring point**: `cmd/cmd.go` (`RunServer`): after/beside the HTTP listener, start the gRPC listener (goroutine + errgroup for shutdown), share the `*Server`.
- **Config point**: `envconfig/config.go` (add `GRPCHost()` / `OLLAMA_GRPC_HOST` mirroring the existing `Host()` pattern; default e.g. `127.0.0.1:11435`).
- **Logic extraction**: Core of handlers (Chat:2422, Generate:256, Embed:754, scheduleRunner:205, List/Ps/Show etc.) + prompt funcs + model ref handling. (scheduleRunner + sched already protocol-agnostic.)

This keeps HTTP 100% unchanged (same routes, gin, middleware, behavior, port, pprof). gRPC is additive. All exploration confirmed this centralization.

**Port strategy evaluation** (prioritized for "without disrupting existing users"):
- **Recommended initial: Separate configurable port (e.g. default 11435 via OLLAMA_GRPC_HOST or always-on with env override)**. Listen two independent `net.Listener`s in RunServer. gRPC on its own `*grpc.Server`. Shutdown via signals + Close on both. Use `errgroup` or simple goroutine + select.
  - Pros: Trivial, zero new deps, no http2/h2c/tls contention, independent (can disable grpc easily), easy to test (different ports), matches runner subprocess pattern (random ports in llm/x/imagegen), graceful for "gRPC opt-in".
  - Cons: Two ports (users may need to configure firewall/docs), slightly more cmd code.
- **Alternative 1: Same port via cmux (later iteration)**. `cmux.New(ln).Match(cmux.HTTP2HeaderFieldPrefix("content-type", "application/grpc"))` for gRPC, else HTTP1/2 to gin. Or h2c handler.
  - Pros: Single port (nice UX, one OLLAMA_HOST), "modern" dual-protocol feel.
  - Cons: New dep (`github.com/soheilhy/cmux`), complexity (http2 for grpc plaintext, mux order, pprof on default mux, tls interaction with existing https support from TLS PR history, error handling on match failures). Higher risk of breaking existing HTTP users during initial landing.
- **Alternative 2: Standard gRPC port (50051) hardcoded + optional HTTP**. Or always derive gRPC port = httpPort+1.
  - Pros: Follows gRPC convention.
  - Cons: Surprises users (non-11434x port); less "Ollama-native".
- **Not recommended (initially)**: grpc-gateway (would reverse the direction—generate HTTP from proto, but we have mature gin + compat middleware already; high duplication risk). Single-http2-only server (breaks plain HTTP clients).

**Refactoring suggestions** (to share logic + avoid duplication with OpenAI/Anthropic middleware hacks):
- Extract *protocol-agnostic* methods on `*Server` (or a small `inferenceService` interface). Example sketch (in routes.go or new server/inference.go):
  ```go
  // Returns chan for streaming (re-use existing internal any or typed); errors via chan or separate.
  func (s *Server) Chat(ctx context.Context, req api.ChatRequest) (<-chan any, error)
  func (s *Server) Generate(...) ...
  func (s *Server) Embed(...) (*api.EmbedResponse, error) // non-stream
  // Internal: the schedule + prompt + Completion loop extracted from ChatHandler/GenerateHandler.
  // Cloud/remote paths can stay gin-specific or get a separate proxy method.
  ```
- Gin handlers become thin: bind/validate -> call extracted -> `streamResponse` or `c.JSON` or `writeChatResponse`.
- Middleware stay HTTP-only (body rewrite + writer wrappers are gin-specific).
- gRPC impl (new `server/grpc.go`): `type ollamaServer struct { s *Server; pb.Unimplemented... }`; `func (g *ollamaServer) Chat(req *pb.ChatRequest, stream pb.Ollama_ChatServer) error { ch, _ := g.s.Chat(stream.Context(), convert(req)); for r := range ch { stream.Send(convert(r)) } }`.
- Use `api.*` structs as the internal lingua franca (or small converters for pb <-> api to keep protos "clean"/efficient). This prevents drift.
- For streaming semantics: gRPC server streams are superior (typed, backpressure, status codes); map current `chan any` + gin.H errors to proper gRPC status + messages.
- DI: `*Server` already acts as holder (sched injected at init). No full DI framework needed; keep simple.
- Versioning: Proto package `ollama.v1`; messages can version fields later. Keep /api/ unversioned for HTTP compat.
- Other: Make `scheduleRunner` or a wrapper exported (lowercase -> upper in same package ok); centralize cloud checks if possible.

**Pros/cons of integration approaches (high-level)**:
- **Approach A (extract + delegate from thin gRPC handlers, separate port, shared *Server*/sched)**: Most efficient. Pros: minimal surface change, reuses 95%+ logic/scheduling/GPU mgmt/inference, easy incremental (start with chat/embed), no HTTP breakage, idiomatic Go (one service impl). Cons: upfront extraction work (but handlers are already somewhat repetitive).
- **Approach B (duplicate handler logic into gRPC for speed-to-first-proto)**: Faster initial spike. Pros: none really. Cons: duplication hell (will rot vs. OpenAI middleware changes + cloud passthroughs), maintenance burden, violates "standardized".
- **Approach C (cmux + full mux in Serve from day 1)**: Overkill for v1. Pros: single port. Cons: risk, dep, complexity (see ports above); defer.

**Minimal v1 gRPC surface** (focus inference first, then admin): Mirror core `/api/*` — `Chat` (server stream), `Generate` (stream), `Embed`/`Embeddings`, `List` (tags), `Show`, `Ps`, `Version`. Add `Pull`/`Push` (progress streams) + `Create`/`Delete`/`Copy`/`Blobs` for completeness. Defer full OpenAI/Anthropic translation or imagegen (binary responses) unless trivial. Support same model ref syntax (incl. cloud passthrough errors or proxy).

---

### 4. Checklist of Files Likely Needing Changes (for Implementation)

- **Config/entry**: `envconfig/config.go` (GRPCHost + OLLAMA_GRPC_HOST, AsMap, tests), `cmd/cmd.go` (RunServer listen + serve both + shutdown + keypair already there), `server/routes.go` (Serve mods + new Register/ServeGRPC + extractions from handlers).
- **New gRPC surface**: `proto/ollama.proto` (or `api/proto/v1/ollama.proto`; recommend root `proto/` for buf friendliness; syntax proto3; service + messages mirroring key api/ types + streaming), generated `pb/ollama.pb.go` + `ollama_grpc.pb.go` (commit them; add go_package).
- **Impl**: New `server/grpc.go` (or in routes.go; service struct + conversions + delegation) + registration call.
- **go.mod** (add `google.golang.org/grpc` + ensure protobuf); possibly `go.sum`.
- **Tests/docs**: `integration/utils_test.go` (grpc port env, or OLLAMA_GRPC_HOST="" for tests; mutex impact), server tests if any direct, `docs/api.md` + `docs/openapi.yaml` (mention grpc), README, perhaps new grpc client example in `api/examples/`.
- **Build**: Update scripts/build_*.sh or CMake? (unlikely, pure Go); add protoc/buf step or document (assume installed for gen).
- **Other potential**: `server/model_resolver.go` / `images.go` / `sched.go` (minor if extraction touches), `api/client.go` (future grpc client support?), `auth/` (if grpc metadata signing), cloud_proxy (passthrough for grpc?), `logutil` / health (grpc health checking standard?).
- **Not needed initially**: Changes to llm/, discover/, model/, CMake, x/ runners, template/ (core inference untouched), most middleware (HTTP-only).

---

### 5. Discovered Gotchas + Pain Points (from Fork's Current State)

- **Gin coupling everywhere**: BindJSON, AbortWithStatusJSON, c.Header, c.Stream, c.Writer replacement (in middleware), c.Request.Context(), gin.H error sentinel in streams. Direct reuse of handlers from gRPC would be painful/fragile. (See writeChatResponse:2369, streamResponse:2070, all handlers.)
- **Cloud / remote model interwining** (big one in this fork): `modelRef.Source == modelSourceCloud` (model_resolver.go:13 + routes.go:278 etc. in *every* inference handler) triggers `proxyCloudJSONRequest` (uses http api client + signing). gRPC paths must decide: error for cloud models, or implement parallel proxy (dupe risk). See cloud_proxy.go, internal/cloud, routes_cloud_test.go, recent model resolver addition.
- **Experimental features**: `useClient2`, `experimentEnabled`, imagegen (x/imagegen + flag in run), mlxrunner, harmony parser, web_search experimental, gemma4 special cases, native chat mode, structured outputs dance in chat loop. gRPC must not regress; test matrix large.
- **Streaming + error semantics**: ndjson + special `gin.H{"error":.., "status":..}` before/after content. gRPC streams + `status.Error` cleaner but conversion needed. Context cancel everywhere (good for gRPC).
- **Scheduler/GPU contention**: Serialized loads, evictions, OOM retries, VRAM accounting, semaphores per-runner, MaxRunners/NumParallel/Spread. Sharing the *exact* sched (via *Server*) is mandatory; parallel grpc/http listeners on same models would fight otherwise.
- **Auth**: Only ed25519 ssh keypair (cmd:2014, auth/, client signing for ollama.com + registry). No per-request local auth (OLLAMA_AUTH limited). gRPC would use metadata interceptors if adding (different from HTTP headers).
- **Large/binary responses**: Image gen (v1/images -> GenerateHandler + ImageWriter; api.GenerateResponse has Image), embeddings (floats), future. gRPC bytes/messages fine; chunk if huge.
- **Integration + test impact**: Binary exec + OLLAMA_HOST + global serverMutex/serverCmd (utils_test.go:430+). Parallel tests or grpc port may collide (use distinct ports or disable grpc in integ by default). Many routes_*_test.go assume HTTP.
- **Build/observability**: slog set globally in Serve; pprof on DefaultServeMux (comment in 1983); version from `version/` package. gRPC server should reuse logger.
- **Other**: `allowedHostsMiddleware` (loopback bias); registry client2 exp; NoCloud; remote models in manifests; heavy C++ side (untouched); flat pkg layout (easy to add `proto/` or `grpc/`); protobuf already present (good); sentencepiece only for tokenizer.
- **Compatibility layers**: OpenAI/Anthropic middleware are *one-way* adapters into core. gRPC should be direct (or future "compat" service if wanted). Types in api/ are canonical (used by x/ too).
- **Fork-specific**: Recent cloud modelref changes, gemma4/llama patches, launch migrations. Clean tree means we start fresh.

---

### 6. Clear Recommendation + Next Steps

**The most efficient and standardized entrypoint is the `server.Serve` / `*Server` lifecycle + handler cores in `server/routes.go` (1917 + 1794 + 2422/256/754/etc.) + `cmd/cmd.go:1996` + `envconfig/config.go`, using a shared `*Server` instance for all protocol handlers after light extraction of business logic.** This reuses the scheduler, model loading, prompt/rendering, llm dispatch, cloud handling (where sensible), and GPU mgmt with zero duplication risk. Separate port (11435 via OLLAMA_GRPC_HOST, default-on or opt-in) for v1 to guarantee no HTTP disruption. Proto at `proto/` (buf-friendly), generated committed, thin gRPC adapter in server pkg.

This is principal-architect level: low-risk incremental addition to a large fork, preserves the "Ollama HTTP API is king for existing users" while adding modern gRPC (excellent for Go clients, streaming, efficiency, typed contracts).

**Immediate next steps (minimal viable slice)**:
1. Add grpc dep + define minimal .proto (Chat/Generate/Embed + List/Version as start; messages based on api/types.go + streaming).
2. `envconfig` + `cmd/cmd.go` plumbing for second listener (non-blocking start of grpc goroutine).
3. Extract 1-2 core paths (e.g. chat local path) + implement gRPC server struct + Register in routes.go Serve.
4. Wire registration + basic ServeGRPC.
5. Build/test locally (go build, integration with grpc port override, manual grpcurl or Go client).
6. Expand surface, add converters, error mapping, auth metadata if needed, docs.
7. (Later) cmux opt-in behind env, full client in api/, health checking, reflection, etc.

**Key references in the source** (relative to repository root):
- Entrypoint: `server/routes.go` (around line 1917 for `Serve`), `server/routes.go:1794` (`GenerateRoutes`), `server/sched.go` (scheduler), `server/routes.go` (handlers such as ChatHandler).
- Lifecycle start: `cmd/cmd.go` (around `RunServer` / listen + serve).
- Config: `envconfig/config.go` (mirror `Host()` pattern for `GRPCHost()` / `OLLAMA_GRPC_HOST`; default e.g. `127.0.0.1:11435`).
- Pipeline core: `server/sched.go`, `llm/server.go` (LlamaServer interface), `server/images.go` (model loading).
- Compat layers: `middleware/openai.go`, `api/types.go` (canonical request/response types).
- Model resolution: `server/model_resolver.go`.
- Existing protobuf work: `convert/sentencepiece_model.proto` (and the new `proto/ollama/api/v1/` files).

All recommendations derived directly from code reads/greps. This enables adding gRPC "the Ollama way"—centralized, non-breaking, leveraging the fork's existing strengths. Ready for implementation.

(subagent id: 019e8fd4-e6fa-78f2-9a97-28e0991375d2)

---

## Post-Implementation Review: Implemented gRPC/Connect Integration Changes

Following the static analysis, the gRPC/Connect integration has been fully implemented in the `feature/grpc-initial` branch. The following robust features, error handling protocols, and compliance standards have been successfully coded and verified:

### 1. Robust Client-Side Features (Resiliency & Tuning)
* **Client-Side Circuit Breaking**:
  - Implemented a localized, thread-safe `circuitBreaker` struct per client instance in `api/grpc_client.go`.
  - Prevents upstream cascading failures by failing fast (throwing error) after 5 consecutive transient failures within a 30-second window.
  - Automatically resets after a 10-second cooldown period. Since it is scoped per-client, transient issues on one endpoint or model do not poison calls to others.
* **Plaintext HTTP/2 Cleartext (h2c) Dialing**:
  - Centralized the transport configuration in `api/h2c.go` (exposed via `api.H2CClient()`).
  - Configures an `http.Client` with the custom `DialTLSContext` override required to dial plaintext gRPC/Connect ports correctly, avoiding the "server gave HTTP response to HTTPS client" error common in standard Go HTTP/2 setups.
* **Keepalive Connection Tuning**:
  - Configured `ReadIdleTimeout` (30 seconds) and `PingTimeout` (15 seconds) on the client transport. This keeps connection paths active during long idle periods (common in interactive reasoning or slow generation tasks) and cleanly tears down stale connections.

### 2. Protocol Compliance & Diagnostics
* **Graceful Context / Deadline Handling**:
  - In `server/grpc.go`, mapped standard errors properly: `context.DeadlineExceeded` is now correctly mapped to the standard gRPC status `CodeDeadlineExceeded`, while `context.Canceled` is mapped to `CodeCanceled`.
* **Standard gRPC Health Checking**:
  - Added full support for the standard gRPC Health Checking protocol.
  - Registers `grpchealth.NewStaticChecker` for all core inference and management services (`ChatService`, `GenerateService`, `EmbedService`, `ModelsService`) so Kubernetes, load balancers, and external tools can query the serving status (`grpcurl ... grpc.health.v1.Health/Check`).
* **Server Reflection**:
  - Wired in `grpcreflect.NewStaticReflector` to publish gRPC service descriptors.
  - Developers can run CLI tools (e.g. `grpcurl` or `buf curl`) to interactively discover and debug APIs, gated behind the environment variable `OLLAMA_GRPC_REFLECTION=1`.

### 3. Rich Observability & Trace Instrumentation
* **Token-Level OpenTelemetry (OTEL) Span Attributes**:
  - Embedded detailed metrics directly into OTEL trace spans for both Unary and Streaming RPCs (`ChatService`, `GenerateService`).
  - Upon completion of a request or stream, the server extracts tokens and duration stats from the final generation chunk and attaches them as standard attributes:
    - `ollama.prompt_eval_count` (int64)
    - `ollama.eval_count` (int64)
    - `ollama.ttft_ms` (int64)
    - `ollama.tps` (float64)
* **Structured Decision logging (`slog`)**:
  - Standardized all logs using the structured `log/slog` framework.
  - Decisions around queue allocation, evictions, scheduler delays, and retry backoffs are logged with contextual metadata (`component="grpc"`, `model`, `rpc`, `stream_id`, `duration_ms`).

### 4. Engine & Scheduler Error Mapping
* **OOM Detection**:
  - Leverages engine-level `llm.IsOutOfMemory(err)` checks during runner allocation.
  - If a model fails to load due to insufficient host or device memory (VRAM/RAM), the gRPC server maps this to `connect.CodeResourceExhausted` (or `codes.ResourceExhausted`), allowing clients to differentiate transient resource constraint states from static invalid configurations.

