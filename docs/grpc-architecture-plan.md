# Technical Architecture and Implementation Plan: Adding gRPC Support to ollamas (feature/grpc-initial)

**Date:** 2026-06-03  
**Author:** Staff-level Systems Architect (Grok Build subagent synthesis)  
**Branch:** feature/grpc-initial (clean, no pre-existing gRPC per exhaustive grep)  
**Status:** Master plan for implementers. Executable, phased, production-grade, <2w effort for MVP.  
**Inputs synthesized (read first):**  
- `/tmp/grpc-research-report-1.md` (market/xai-proto research; hereafter [grpc-research])  
- `/tmp/ollamas-grpc-codebase-review.md` (codebase entrypoints, gotchas, lines; hereafter [ollamas-review])  
- `logloom-graph.json` (AST of log sites/callgraph; hereafter LogLoom)  
- Local verification via list_dir/grep/read_file on server/routes.go, cmd/cmd.go, envconfig/config.go, api/types.go, llm/server.go, server/sched.go, go.mod, server/images.go, server/model_resolver.go, server/cloud_proxy.go, server/prompt.go, logutil/logutil.go, integration/utils_test.go, CONTRIBUTING.md, docs/development.md, docs/api.md, etc.  
- Reliable Go Systems skill: `/Users/jonathandoughty/.grok/skills/reliable-go-systems/SKILL.md` (ctx-first, %w errors, structured slog, bounded, no fire-and-forget, errgroup, rich reasoning logs, checklist).  
- xai-proto patterns (https://github.com/xai-org/xai-proto, buf configs fetched via tools; commit ~543b901 per research).  
- Upstream context: ollama/ollama#10085 (gRPC API request for microservices/low-latency).

All claims cite exact paths/lines from inputs + verification reads. Plan prioritizes "extract once, serve many protocols" + shared `*Server` + separate port for safety (no breakage to HTTP on 11434).

---

## 1. Executive Summary & Goals

### Why gRPC for ollamas
- **Performance for streaming**: True server-streaming with backpressure/flow control vs. current NDJSON/SSE hacks in `server/routes.go:2070` (`streamResponse` using `c.Stream`, `gin.H` error sentinels, manual flushes). gRPC excels for token-by-token (research [grpc-research:73]).
- **Typed contracts for SDKs**: Enables first-class Python/TS/Go clients (like xai-sdk-python over xai-proto), strong typing for complex sampling/tools/multimodal (oneofs), vs. json unmarshal drift in `api/types.go`.
- **Microservices / local clustering / parity**: Supports patterns from Triton (KServe gRPC), vLLM (internal), xAI (public gRPC surface + HTTP compat). Community demand via #10085; wrappers like OLOL exist because no native (research [grpc-research:79], [ollamas-review:79]).
- **Observability & ops**: Built-in health/reflection, OTEL (metrics/tracing per-RPC), rich metadata, deadlines. Aligns with production LLM servers.
- **Efficiency**: Binary protobuf + multiplexing for large contexts/multimodal (images in `api/types.go:98`); lower latency for high-concurrency local use.

**Non-goals** (strict):
- Do **not** replace HTTP/OpenAI/Anthropic surface (`/api/*`, `/v1/*` in `GenerateRoutes:1794`).
- Full backward compat: zero regression on port 11434 (existing curl, OpenAI SDKs, `api/client.go`, integration tests, desktop/app subprocesses).
- No breakage to C++ runners (`llm/`, `discover/`, `x/imagegen/`, `x/mlxrunner/`), experimental features, cloud passthrough, or model loading.
- No immediate gRPC client in `api/` or full admin (Pull progress etc.); defer.
- Do not touch llama.cpp/CMake (C++ isolation).

### Success Metrics (MVP in <2w)
- Zero regression: all existing `go test ./server -run 'Test.*(Route|Chat|Generate|Embed|List)'`, `integration/*` (with serial mutex), HTTP paths pass unchanged.
- Working gRPC: Chat (server-stream), Generate (stream), Embed (unary), Models/List + Version; testable via `grpcurl` / `buf curl` / small Go `connect` client on default 11435.
- Same scheduler/GPU behavior: single `*Scheduler` instance (from `server/sched.go:91` `InitScheduler`), VRAM accounting, evictions, `llm.LlamaServer` iface (`llm/server.go:59`) shared.
- Observable: structured slog (model, grpc_method, tokens, duration_ms, stream_id) + basic OTEL spans/interceptors on gRPC paths; health/reflection enabled.
- Effort: Phase 0-4 land MVP; Phase 5 optional. Total implementer time <2 weeks (S/M/L per phase).
- Backcompat: `OLLAMA_GRPC_HOST` controls (default-on separate port); HTTP unchanged.

---

## 2. Recommended Architecture (Synthesized)

### Framework Choice: ConnectRPC (connect-go + buf) Primary
**Primary rec (from [grpc-research:11,30,37]):** `connectrpc.com/connect` (with `protoc-gen-connect-go`) + Buf.build.  
**Rationale (synthesized vs. alternatives):**
- One `net/http` handler supports **Connect + gRPC + gRPC-Web** protocols (incl. streaming) out-of-box. Excellent for local (grpcurl, buf curl, curl/HTTP1.1 debug, browsers). No separate grpc-go server needed.
- First-class Buf (lint/breaking/gen for Go + future Python/TS stubs like xai-proto [grpc-research:47-48]).
- Simpler integration than grpc-go (smaller generated, generics, easy middleware/interceptors for net/http).
- Coexistence: on *separate port* trivial (plain `http.Server` + h2c for unencrypted HTTP/2 per connect getting-started example); future same-port via cmux *or* http mux (less complex than pure grpc-go + cmux).
- Vs. grpc-go: mature but HTTP/2 friction, larger, requires cmux/proxies for mixed (table [grpc-research:36]).
- Vs. grpc-gateway: heavier transcoding; we already have mature Gin + OpenAI compat middleware (`middleware/openai.go:408`); gateway would duplicate.
- Vs. cmux day-1: overkill/risky for v1 (review [ollamas-review:143-145,173]); defer to Phase 5 opt-in.
- xAI alignment: "xAI uses gRPC wire but Connect philosophy aligns" (research [grpc-research:26]); their buf + rich LLM protos are model.
- Prod notes: growing adoption (Buf, YC, fintech); conformance-tested.

**Implementation note:** Use `http.Server` with `Protocols: h2c + HTTP1` (see connect docs example). gRPC clients work via gRPC protocol compat; Connect clients default to Connect protocol.

**go.mod additions (runtime):** `connectrpc.com/connect`, `connectrpc.com/otelconnect` (for OTEL), `google.golang.org/grpc` (for `codes`, `status`, health if needed; indirect often), `go.opentelemetry.io/otel` + sdk/exporters (optional full; interceptors work with providers).

### Buf + Proto Layout + Versioning
- **Dir:** `proto/ollama/api/v1/` (root `proto/` for buf friendliness; mirrors `proto/xai/api/v1/` in xai-proto [grpc-research:46]).
  - Split files for clarity (like xai): `chat.proto`, `generate.proto` (or unify), `embed.proto`, `models.proto`.
  - Or single `ollama.proto` for MVP simplicity.
- **Versioning:** Major-version dirs (`v1/`, future `v2/`) + `buf breaking --against` (FILE strategy [grpc-research:47]). Package `ollama.api.v1`; go_package with version suffix or managed prefix.
- **buf.yaml (v2, at repo root):**
  ```
  version: v2
  modules:
    - path: proto
  deps:
    - buf.build/googleapis/googleapis
    - buf.build/grpc-ecosystem/grpc-gateway  # if needed for any
  lint:
    use: [MINIMAL]
    except: [PACKAGE_DIRECTORY_MATCH]
  breaking:
    use: [FILE]
  ```
- **buf.gen.yaml (v2):**
  ```
  version: v2
  clean: true
  managed:
    enabled: true
    override:
      - file_option: go_package_prefix
        value: github.com/ollama/ollama/gen/proto
  plugins:
    - local: [go, tool, protoc-gen-go]
      out: gen/proto
      opt: paths=source_relative
    - local: [go, tool, protoc-gen-connect-go]
      out: gen/proto
      opt:
        - paths=source_relative
        - simple
  ```
- **Codegen:** `buf dep update && buf lint && buf generate`. Commit generated (`gen/proto/ollama/api/v1/...pb.go` + `...connect.go`). (Matches sentencepiece pattern `convert/sentencepiece_model.proto` + generated [ollamas-review:124].)
- **Workflow:** Add to dev docs (`docs/development.md`); optional `//go:generate` or `scripts/`. Buf in CI (lint + breaking against main) in Phase 5.
- **Why not api/proto/?** `proto/` at root is standard for Buf modules; keeps API contracts separate from Go `api/`.

**Data model mapping:**  
**Canonical:** `api/types.go` (GenerateRequest:62, ChatRequest:147, EmbedRequest:613, Message, Tool*, Options map, ImageData, StatusError, etc.) + internal (`server/model.go`, `llm/server.go` llm.ChatRequest).  
**Strategy:** Extract protocol-agnostic methods on `*Server` (e.g. `func (s *Server) Chat(ctx context.Context, req api.ChatRequest) (<-chan api.ChatResponse, error)` or callback writer style) that operate **only on api.* + ctx**.  
gRPC (connect) layer: **thin adapter** only (pb <-> api converters + error mapping + cloud check). No logic dupe.  
**Proto design:** Mirror api + enrich per xai (oneof Content for multimodal [text/image_url/etc], Tool oneof [Function/WebSearch/...], rich Usage/Sampling, FinishReason enum, Think, Logprobs, Format as oneof or bytes). Use `google.protobuf.Struct` or map for `options`; bytes for images. Converters handle orderedmap (`ToolCallFunctionArguments`), json.RawMessage, etc.  
This prevents drift, reuses prompt/render/schedule/renderers in `server/prompt.go:23` (chatPrompt), `server/images.go:622` (GetModel), etc.

### High-Level Components
- **Shared core:** `*Server` (routes.go:100: tiny holder for `sched *Scheduler`, `defaultNumCtx`, caches, addr; no transport). `scheduleRunner:205` (ctx-first, calls GetModel + sched.getRunner).
- **New:** `server/grpc.go` (or connect.go): connect handler impls (`ChatServiceHandler` etc.), converters, interceptors. `func (s *Server) RegisterGRPC(mux *http.ServeMux)` or equiv.
- **Lifecycle:** Extend `Serve:1917` + `GenerateRoutes:1794` minimally; new `Setup()` + `ServeGRPC(ln, s *Server)`.
- **Wiring:** `cmd/cmd.go:1996` (RunServer) for dual listeners + errgroup + signals + graceful.
- **Config:** `envconfig/config.go` (mirror Host:22).
- **Obs/Auth:** Interceptors (slog + otelconnect); metadata for auth (ed25519 reuse for cloud).
- **Other:** Health (standard), reflection (for grpcurl), error mapping (from gin.H patterns + StatusError:22).

### Port/Listener Strategy (Safety First)
**Initial (MVP):** Separate configurable port. Default `127.0.0.1:11435` via `OLLAMA_GRPC_HOST` (always-on unless explicitly disabled via empty/invalid).  
- Pros (review [ollamas-review:140-142]): Trivial, zero risk to 11434, independent disable/test, no h2 contention with existing http.Server/pprof on DefaultServeMux, matches runner subprocess ports.
- Cons: Two ports (docs/firewall note); users set `OLLAMA_GRPC_HOST=127.0.0.1:11435` or rely on default.

**Future (Phase 5 opt-in):** cmux on single 11434 (or http mux + protocol detect). `OLLAMA_GRPC_MUX=1`. Use `soheilhy/cmux` for HTTP2HeaderFieldPrefix("content-type", "application/grpc") -> gRPC handler; else Gin. (cmux essential per [grpc-research:39]).

**Diagram (text):**
```
[CLI/SDK/OpenAI clients (HTTP)]          [gRPC clients (connect/grpcurl)]
          | 11434 (OLLAMA_HOST)                     | 11435 (OLLAMA_GRPC_HOST, default)
          v                                       v
cmd/cmd.go:1996 RunServer()
  |
  +-- lnHTTP = net.Listen(Host().Host)     +-- lnGRPC = net.Listen(GRPCHost().Host)  [if set]
  |
  v
server/routes.go:1917 Serve(lnHTTP)  [refactored to Setup + http serve]
  +-- slog, prune, s := &Server{...} (1951)
  +-- h = GenerateRoutes:1794 (gin + /api + /v1 middlewares + cloudPassthrough)
  +-- sched = InitScheduler (sched.go:91); s.sched = sched; s.modelCaches.Start
  +-- GPU discover, defaultNumCtx
  +-- signals -> srvr.Close() + sched.unloadAllRunners()
  +-- s.sched.Run(); srvr.Serve(lnHTTP)  [blocks]
          |
          v  (handlers e.g. ChatHandler:2422)
             bind api.ChatRequest -> parseAndValidateModelRef (model_resolver.go:36)
             if cloud: proxyCloudJSONRequest (cloud_proxy.go:158, gin-tied)
             GetModel (images.go:622) -> scheduleRunner:205 (ctx, caps, opts) -> s.sched.getRunner (sched.go:165)
             -> render (prompt.go:23 chatPrompt) -> r.Chat/Completion (llm iface) -> ch any -> streamResponse:2070 (ndjson + gin.H errors)

Parallel gRPC path (new):
  server.ServeGRPC(lnGRPC, s)  [http.Server + connect handlers or cmux later]
    +-- interceptors (log/otel/auth)
    +-- register Chat/Embed/Models handlers
    +-- thin: pbReq -> apiReq (converters) -> if cloud: err("HTTP only for cloud") -> s.Chat(ctx, apiReq) [extracted, reuses schedule/render/llm]
    +-- stream.Send(convert)  [backpressure via gRPC flow control; ctx prop to sched/llm]
    +-- GracefulStop on shutdown
```
**Callouts:** All inference converges on `*Server` + `*Scheduler` + `llm.LlamaServer` (review diagram [ollamas-review:14-91]). OpenAI/Anthropic are *adapters* into same handlers (middleware rewrite body/writer then Next()). gRPC is direct (no compat middleware).

### Request Flow gRPC vs HTTP (Detailed)
1. **Common (post-extract):** `ctx` (with deadlines) -> parse/validate modelRef (local/remote/cloud) -> GetModel + CheckCapabilities -> scheduleRunner (ctx) -> sched.getRunner (LlmRequest{ctx, ...}) -> load/evict/VRAM/semaphore/llm.NewLlamaServer -> prompt render (tools, thinking, harmony, templates) -> native or llm.Chat/Completion(ctx, req, callback) -> usage/finish.
2. **HTTP:** gin bind + middleware (cloudPassthrough, ChatMiddleware etc.) -> handler (e.g. 2422) -> cloud proxy or extracted -> ch any or direct JSON -> streamResponse (ndjson, special error gin.H at 2050) or c.JSON.
3. **gRPC:** connect req (or stream) -> interceptor (slog/otel/ctx metadata) -> adapter (convert + cloud guard) -> extracted *Server method (ctx first) -> callback/stream.Send (typed pb) or unary resp. Errors -> connect.NewError(codes.XXX, ...) or status.
4. **Ctx/deadlines critical:** Propagate from gRPC stream.Context() all the way (research [grpc-research:138]); sched LlmRequest holds ctx (containedctx noted); llm runners use for cancel to avoid GPU leaks/hung runners. Timeouts on loads (envconfig.LoadTimeout).

**Cloud passthrough:** HTTP-only for MVP (gin-tied + json proxy in cloud_proxy.go). gRPC adapters: early return clear error for `modelSourceCloud` (model_resolver.go:13) or remoteHost models. Future: non-gin proxy using api client or direct.

**OpenAI compat:** Stays HTTP-only (middleware + handlers).

**Experimental runners (x/):** Untouched (own http servers on random ports).

### Observability
- **slog (existing):** Preserve custom (logutil/logutil.go:14 NewLogger, TRACE level -8, basename source). In gRPC: rich attrs on *all* paths (model, grpc_service/method, stream_id or req uuid, tokens_in/out, load_duration, total_duration, finish_reason, error, client metadata if any). E.g. `slog.Info("grpc chat started", "model", req.Model, "component", "grpc", "stream", req.Stream, "think", req.Think)`.
- **LogLoom gaps:** Only ~28 instrumented points in `server/routes.go` (grep on LogLoom); sched has more (~73) but clustered in processPending/Run. Many handlers (Chat:2422 full body) sparse. **Plan:** New gRPC paths + extracted methods get comprehensive coverage + "reasoning" style logs (e.g. `slog.Debug("decision: local inference path", "reason", "no :cloud suffix", "model", ...)`) for audit/debug (per reliable skill + xai telemetry). This improves overall observability.
- **OTEL:** Add `otelconnect` interceptors (tracing + metrics, GenAI semconv where possible like xai-sdk [grpc-research:130]). Bridge to slog (custom handler or dual). Spans around schedule/load/inference. Export optional (console/OTLP). Attributes: model, usage (prompt/completion/reasoning/cached), duration, status.
- **Health/Reflection:** Enable gRPC health checking + server reflection (for grpcurl/discovery/K8s). (Standard in grpc-go; for connect use wrappers or dual.)
- **Other:** Request IDs via ctx; pprof stays on HTTP DefaultServeMux.

### Auth
- Metadata-based (incoming `authorization` or `x-ollama-auth`; interceptor).
- Local: permissive (like allowedHostsMiddleware loopback bias at routes:1826; or allow all on dedicated gRPC port).
- Cloud/ed25519: reuse `auth/` + `initializeKeypair` (cmd:2014) for signing if gRPC cloud passthrough added later. For now, passthrough errors.
- Interceptors early in chain.

### Error Handling
- Map from existing: `api.StatusError:22`, gin.H{"error":, "status":} (streamResponse:2050, handleScheduleError:3080, writeModelRefParseError:69), context.Canceled -> 499, ErrMaxQueue -> 503, not found -> 404, caps/required -> 400, etc.
- gRPC: `connect.NewError(connect.CodeInvalidArgument, err)` (or grpc `status.Error(codes.InvalidArgument, ...)` + details). Use `google.rpc.Status` for rich (xai pattern [grpc-research:56]).
- Streaming: send error msg on stream before close (or trailer status).
- Always wrap: `fmt.Errorf("chat %s: %w", model, err)`.

### Streaming Backpressure, Ctx, Deadlines, Resources (Critical)
- gRPC Send blocks on slow client -> blocks producer -> must not leak GPU: 
  - Always `ctx, cancel := context.WithCancel(stream.Context())`; pass derived ctx down.
  - Bounded internal chans (cap 16-64 for tokens).
  - In extracted writer/callback: `select { case ch <- r: case <-ctx.Done(): return ctx.Err() }`.
  - llm callback must respect ctx (existing does via r.Chat(ctx,...)).
  - On handler return (client cancel/Send err): cancel() to stop generation promptly.
- No fire-and-forget goroutines (reliable checklist): every `go func()` has defer close, ctx owner, select on done.
- Scheduler semaphores/NumParallel/MaxQueue respected (shared).
- Timeouts: LoadTimeout, per-RPC deadlines propagated.

### Integration with Existing
- Shared `*Server`/sched (mandatory to avoid contention [ollamas-review:198]).
- Cloud/modelref: handled (error for gRPC MVP).
- OpenAI/Anthropic/compat: HTTP-only.
- Experimental: untouched.
- Logging global (slog.SetDefault in Serve:1918).
- No changes to llm/, discover/, model/, template/, x/, CMake, most middleware.

### Use of LogLoom
LogLoom is the starting AST for call sites/models (e.g. nodes for Server.GenerateHandler:17888, Serve:18126, streamResponse, sched.Run/processPending). Use it to identify uninstrumented paths (~50 in routes per review [ollamas-review:201] + confirmed ~28 logged nodes). In gRPC impl + extracts: add structured logs at decision points (parse, schedule success/fail, cloud branch, render choice, finish reason, error paths). Suggest post-impl: run logloom to capture new nodes + add "reasoning" tags for agentic/debug flows.

---

## 3. Detailed Implementation Plan (Phased, Actionable)

**General rules across phases (reliable-go + contrib):**
- Every I/O/long func: `ctx context.Context` **first** param. Never store ctx in structs.
- Errors: check all, `%w` wrap, `errors.Is/As`, lowercase no punct, sentinels for branches.
- Structured slog + ids: `model`, `component:"grpc"`, `rpc`, `stream_id:uuid`, `duration_ms`, `tokens`, `status`.
- Bounded: chans, workers, timeouts. `defer` resources. `errgroup` for lifecycles.
- No fire-and-forget. Idempotent where possible.
- Tests: `-race`, table-driven. `golangci-lint` clean.
- Compile iface checks.
- PR: explain new dep (per CONTRIBUTING:80), tests, draft docs.
- Per phase: touch list (with :lines), new files, cmds, verif, risks/mitigations, effort (S/M/L).
- Track in todo_write during impl.
- Feature: always-on separate port (env OLLAMA_GRPC_HOST controls addr; set to empty/disable logic).

### Phase 0: Foundations (Deps, Proto Skeleton, Buf, Codegen, Minimal Build) — Effort: S (1-2 days)
**Goal:** Buildable skeleton, no runtime change.

**Tasks (numbered):**
1. Update `go.mod` (and tidy): `go get connectrpc.com/connect@latest`; `go get connectrpc.com/otelconnect@latest`; `go get google.golang.org/grpc@latest` (for codes/status); `go get go.opentelemetry.io/otel@latest` (and sdk if full); `go get -tool google.golang.org/protobuf/cmd/protoc-gen-go@latest`; `go get -tool connectrpc.com/connect/cmd/protoc-gen-connect-go@latest`. (Explain in PR: "for gRPC surface per plan; buf-managed codegen".)
2. Install buf (dev): follow https://buf.build/docs/installation (or `go install github.com/bufbuild/buf/cmd/buf@latest` + PATH). Update `docs/development.md:5` prereqs + "for gRPC: buf + protoc-gen-* (go get -tool)".
3. Create `proto/ollama/api/v1/` (mkdir).
4. Create `proto/ollama/api/v1/chat.proto` (minimal MVP; others later or in one file):
   ```
   syntax = "proto3";
   package ollama.api.v1;
   option go_package = "github.com/ollama/ollama/gen/proto/ollama/api/v1;apiv1";

   import "google/protobuf/timestamp.proto";

   message Message {
     string role = 1;
     string content = 2;
     // TODO: enrich with oneof parts { text, image {bytes, detail}, ... } per xai [grpc-research:55]
     repeated bytes images = 3;  // initial mirror api
     // tool_calls etc.
   }
   message ChatRequest {
     string model = 1;
     repeated Message messages = 2;
     bool stream = 3;
     // KeepAlive, Options (map or Struct), Think, Format (bytes or oneof), Tools, Logprobs, TopLogprobs, Truncate, Shift, etc. Mirror api/types.go:147 + xai sampling.
     map<string, string> options = 10;  // or google.protobuf.Struct
     // ...
   }
   message ChatResponse {
     string model = 1;
     Message message = 2;
     bool done = 3;
     string done_reason = 4;
     // usage, created_at (Timestamp), ...
   }
   service ChatService {
     rpc Chat(ChatRequest) returns (ChatResponse);  // unary non-stream
     rpc ChatStream(ChatRequest) returns (stream ChatResponse);  // streaming MVP
   }
   // Similar for GenerateService, EmbedService, ModelsService { rpc List(ListRequest) returns (ListResponse); rpc Show... }
   // EmbedRequest/Response mirror api:613.
   // Add common: Version, Usage messages, FinishReason enum, Error details.
   ```
   (Start close to api; evolve oneofs in Phase 3/5. Use buf validate if added.)
5. `buf.yaml` + `buf.gen.yaml` at root (as above; adjust go_package_prefix to `github.com/ollama/ollama/gen/proto`).
6. `buf dep update && buf lint && buf generate`. Creates `gen/proto/ollama/api/v1/...`.
7. `go mod tidy && go build .` (stubs only; no impl yet). Add `gen/proto` to any .gitignore? No — commit.
8. Add basic `//go:generate` comment or note in README/CONTRIBUTING.

**Files to touch:**
- `go.mod` / `go.sum` (new requires)
- `proto/ollama/api/v1/chat.proto` (new)
- `proto/ollama/api/v1/models.proto` (new, minimal List/Show)
- `proto/ollama/api/v1/embed.proto` (new)
- `buf.yaml` (new)
- `buf.gen.yaml` (new)
- `docs/development.md` (add buf/prereqs)
- (Optional) `.github/workflows/*.yml` later for buf.

**Commands:**
```sh
go get connectrpc.com/connect connectrpc.com/otelconnect google.golang.org/grpc go.opentelemetry.io/otel
go get -tool google.golang.org/protobuf/cmd/protoc-gen-go connectrpc.com/connect/cmd/protoc-gen-connect-go
buf --version  # or install
buf dep update
buf lint
buf generate
go mod tidy
go build .
```

**Verification:**
- `buf build`, `buf lint` clean.
- `ls gen/proto/ollama/api/v1/` shows pb + connect.go.
- `go build .` succeeds (no import yet).
- `git status` shows new proto/gen (commit generated).
- Manual: `grep -r grpc gen/proto` or strings.

**Risks/Mitigations:**
- Buf install friction: document exactly; use go tool for gens.
- Proto design drift: keep MVP minimal, map 1:1 to api/* first; enrich later. Use LogLoom? No, for fields use api/types + xai research.
- Effort S.

### Phase 1: Listener + Lifecycle Wiring (envconfig, cmd/cmd.go RunServer Dual Listen + Graceful, Serve Extension) — Effort: M (2-3 days)
**Goal:** Dual listeners, shared state, graceful shutdown, zero HTTP change. Refactor for extractability.

**Tasks:**
1. `envconfig/config.go`: Add `GRPCHost() *url.URL` (copy/adapt Host:22; default "127.0.0.1:11435"; support OLLAMA_GRPC_HOST; handle scheme/port like Host; Connectable if needed). Add `NoGRPC()` or disable logic (if Var("OLLAMA_GRPC_HOST") == "" { return nil }). Update `AsMap():311` (add entry like OLLAMA_HOST:322), `Values()`, docs strings. Add test in config_test.go.
2. Refactor `server/routes.go` (critical):
   - Extract setup: `func setupServer() (*Server, http.Handler, *Scheduler, context.CancelFunc, func(), error)` — moves prune/fixBlobs (1923), s:=1951, initRequestLogging, GenerateRoutes:1968, http.Handle? (keep in serve), ctx/sched creation 1975, modelCaches, GPU 2013, defaultNumCtx 2021-2030, signals? (move up). Returns ready s, routes h, sched, done funcs.
   - Keep/adapt `Serve(ln net.Listener) error` to use setup + start http srvr.Serve (for compat).
   - Add `func (s *Server) ServeGRPC(ln net.Listener) error` (or package func): create mux, register (will fill Phase 2), http.Server with h2c Protocols (per connect example), ListenAndServe or Serve. (Interceptors later.)
   - Add `func (s *Server) registerServices(mux *http.ServeMux)` stub (empty for now).
   - Keep signal/unload in Serve or centralize in cmd.
   - Add imports as needed (context, net/http, etc.).
3. `cmd/cmd.go:1996 RunServer`:
   - After initializeKeypair:2014.
   - lnHTTP, err := net.Listen("tcp", envconfig.Host().Host)
   - gh := envconfig.GRPCHost()
   - var lnGRPC net.Listener
   - if gh != nil && gh.Host != "" { lnGRPC, _ = net.Listen("tcp", gh.Host) }
   - Use `g, ctx := errgroup.WithContext(context.Background())` (x/sync already in go.mod).
   - Setup: s, h, sched, schedDone, cleanup, err := server.Setup() (or inline calls to new setup).
   - http.Handle("/", h)  [or move]
   - s.sched.Run(...) if not in setup.
   - g.Go( func() error { srvr := &http.Server{Handler: nil}; /* signals? */ return srvr.Serve(lnHTTP) } )
   - if lnGRPC { g.Go( func() error { return server.ServeGRPC(lnGRPC, s) } ) }
   - Signal handling (enhance existing 1995): notify, srvr.Close(), if grpcSrv !=nil { grpcSrv.Shutdown or Graceful (for connect http: Shutdown(ctx)) }, schedDone(), sched.unloadAllRunners(), done.
   - return g.Wait()
   - Handle ErrServerClosed etc.
   - Update env docs append (2550 area): add envVars["OLLAMA_GRPC_HOST"] for serveCmd.
4. Minor: update `server/routes.go:1983` pprof comment if needed; ensure slog set once.
5. Add basic healthz or version over gRPC stub (later full).

**Files to touch (with anchors):**
- `envconfig/config.go:22` (Host model), ~311 (AsMap), 366 (Values) — add GRPCHost + entry.
- `cmd/cmd.go:1996` (RunServer), 2001 (listen), 2548 (envVars), 2569 (appendEnvDocs for serveCmd).
- `server/routes.go:100` (Server struct if extend), 1794 (GenerateRoutes), 1917 (Serve — refactor), 1951 (s creation), 1975 (ctx/sched), 1995 (signals), 2033 (Serve ln), + new setup/ServeGRPC/register ~new lines.
- `server/routes_test.go` etc (if direct Serve calls; mostly not).
- `integration/utils_test.go:452` (env set; later).
- `docs/development.md`, `docs/api.md` (mention), cmd help.

**New files:** None yet (or server/grpc.go stub).

**Commands:**
```sh
go test ./envconfig -run Test  # after GRPCHost
go build .
go run . serve  # test default; curl localhost:11434 ; check logs for grpc? (later)
# In another: OLLAMA_GRPC_HOST=127.0.0.1:11436 go run . serve
```

**Verification steps:**
- `OLLAMA_GRPC_HOST=127.0.0.1:11435 go run . serve` starts; `lsof -i :11435` shows; HTTP still on 11434.
- Signals (ctrl-c): graceful, unload logs, exit 0.
- `env | grep OLLAMA` shows new; `./ollama --help` or serve docs updated.
- No regression: `go test ./server -run TestRoutes` or full.
- Dual: two listeners, shared s.sched (add temp log or debug).
- Errgroup: no leaked goroutines.

**Risks/Mitigations:**
- Refactor breaks HTTP: do minimal moves; keep Serve signature/behavior identical for now; extensive tests + manual run before/after. Use httptest in unit.
- Dual sched risk: enforce single via setup; never call Serve twice.
- Shutdown races: use errgroup + ctx + explicit closes + <-done; test with kill.
- Env parse edge (ports, ipv6): copy Host exactly + tests.
- Effort M (refactor careful).

### Phase 2: Core Logic Extraction + Thin Adapters — Effort: L (4-5 days, core value)
**Goal:** Protocol-agnostic *Server methods; gin handlers thin; gRPC skeleton delegates; converters; first end-to-end (even if errors).

**Tasks:**
1. **Extraction in `server/routes.go`** (heavy, target Chat/Generate/Embed first; List/Show/Ps easier):
   - After cloud/modelRef checks + GetModel (dupe in handlers), call extracted.
   - Introduce (private or exported for tests):
     ```go
     // chat handles core chat after bind/cloud validation. write is called for each response (streaming or final).
     // Respects ctx for cancel/backpressure. Returns err (mapped by caller).
     func (s *Server) chat(ctx context.Context, req api.ChatRequest, write func(api.ChatResponse) error) error {
       // move: name resolution, GetModel (again? dedupe), scheduleRunner:2616, render (msgs, harmony, tools, filterThink), 
       // if len==0 early resp, native? handleNative, else r, m, opts := schedule...
       // ch or direct: use write in llm callback or after collect.
       // error paths: use write for gin.H style? or return err; let caller map.
       // prompt render calls, structured outputs dance, etc.
       // Ensure: all ctx passed (c.Request.Context() -> ctx), no gin.
       // Add rich slog at branches.
     }
     ```
     Similar for `generate(ctx, api.GenerateRequest, write func(api.GenerateResponse) error) error`.
     For Embed: `embed(ctx, api.EmbedRequest) (*api.EmbedResponse, error)` (unary, simpler).
     For List: `list(ctx) (api.ListResponse, error)` etc.
   - Update **gin handlers to thin wrappers**:
     - ChatHandler:2422: bind/validate (TopLogprobs etc), modelRef parse (cloud proxy return), name get, m=GetModel (keep some), then if !*stream { var final; err:=s.chat(ctx, req, func(r){final=r}); c.JSON... } else { go? adapt to ch or direct stream; use existing writeChatResponse/streamResponse which expect ch/gin.H. Bridge: make ch, s.chat(..., func(r){ch<-r}); close; stream... } or refactor stream too.
     - Goal: zero behavior change for HTTP. Test heavily.
   - Handle internal ch any / gin.H errors: in extracted, return err; for stream compat, keep gin paths using ch + special. Or evolve ch to typed + error sentinel.
   - Extract helpers if needed (filterThinkTags, shouldUseHarmony, etc stay pkg).
   - Add logs per LogLoom gaps + reliable: e.g. after schedule "scheduled runner", "model", req.Model, "load_ms", time.Since...
   - For non-stream early "load" resp etc: preserve.
2. **Thin gRPC adapters `server/grpc.go` (new):**
   ```go
   package server
   import (
     "context"
     "net/http"
     chatv1 "github.com/ollama/ollama/gen/proto/ollama/api/v1/ollamachatv1connect"  // adjust
     "connectrpc.com/connect"
     // pb "....pb"
   )
   type chatHandler struct { s *Server }
   func (h *chatHandler) Chat(ctx context.Context, req *connect.Request[pb.ChatRequest]) (*connect.Response[pb.ChatResponse], error) {
     apiReq := convertToAPI(req.Msg)
     // cloud guard
     if /* cloud */ { return nil, connect.NewError(connect.CodeUnimplemented, errors.New("cloud models require HTTP API (remove :cloud or use /v1)")) }
     var resp api.ChatResponse
     err := h.s.chat(ctx, apiReq, func(r api.ChatResponse){ resp = r })  // for unary
     if err != nil { return nil, errToConnect(err) }
     return connect.NewResponse( convertToPB(resp) ), nil
   }
   func (h *chatHandler) ChatStream(ctx context.Context, req *connect.Request[pb.ChatRequest], stream *connect.ServerStream[pb.ChatResponse]) error {
     apiReq := ...
     return h.s.chat(ctx, apiReq, func(r api.ChatResponse) error {
       if err := stream.Send(convertToPB(r)); err != nil { return err }
       return nil
     })
   }
   // Similar for other services.
   func (s *Server) registerServices(mux *http.ServeMux) {
     path, h := chatv1.NewChatServiceHandler(&chatHandler{s}, connect.WithInterceptors( loggingInterceptor(), otelconnect.NewInterceptor(otelconnect.WithTracerProvider(...)) ))
     mux.Handle(path, h)
     // embed, models...
   }
   ```
   - Converters: `func convertToAPI(*pb.ChatRequest) api.ChatRequest { ... }` (map fields, handle oneofs, json for format/options, images bytes). Reverse. Put in grpc.go or new `server/convert_grpc.go`.
   - Error mapper: `func errToConnect(err error) error { switch { case errors.Is(err, context.Canceled): return connect.NewError(connect.CodeCanceled, err) ... case isStatusErr: ... default: return connect.NewError(connect.CodeInternal, err) } }` (use codes from grpc or connect.Code*).
   - Interceptors (basic first): func loggingInterceptor() connect.Interceptor { return connect.UnaryInterceptorFunc( func(next connect.UnaryFunc) connect.UnaryFunc { return func(ctx, req) { start:=time.Now(); slog.Info("grpc unary start", "rpc", req.Spec().Procedure, "model", extractModel(req)); res,err:=next...; slog.Info("grpc done", "dur_ms", time.Since(start).Milliseconds(), "err", err); return } }) } . Streaming similar. Enrich with ids.
   - For stream backpressure/ctx: in ChatStream, use ctx from stream; pass to s.chat; in write func select on ctx if needed; bounded if internal ch used.
3. Update calls in other places if any (e.g. imagegen uses GenerateHandler?).
4. Add compile checks: `var _ chatv1.ChatServiceHandler = (*chatHandler)(nil)`
5. Unit tests for converters (table on api <-> pb roundtrip for key cases: simple text, tools, images, think, errors).
6. Temp: in extracted, for stream paths may still use some ch; clean in polish.

**Files to touch:**
- `server/routes.go:2422` (ChatHandler -> thin), 256 (Generate), 754 (Embed), 205 (scheduleRunner stays), 2616 (schedule call site), 1794?, 1917 (register call), new extracted funcs ~ e.g. after 244.
- `server/grpc.go` (new, ~200-400 LOC for MVP + converts).
- `server/convert_grpc.go` (new, optional).
- `server/routes_test.go`, `routes_generate_test.go` etc (verify no change).
- `api/types.go` (minor? if extend for internal).

**New files:** `server/grpc.go`, `server/convert_grpc.go` (if split), `proto/...` already in 0.

**Commands:**
```sh
go test ./server -run 'Test(Chat|Generate|Embed|Route)' -race -count=1
go build .
# manual: start server (grpc port), implement temp client or use buf curl/grpcurl on ChatStream (after phase3 proto impl)
```

**Verification:**
- HTTP identical (diff responses pre/post, full test matrix incl. tools, vision, thinking, streaming, errors, cloud error paths).
- gRPC path reaches schedule (add temp log "grpc reached core chat"); basic unary/stream "works" (even stub responses).
- Converters roundtrip key structs.
- Ctx cancel during "stream" stops promptly (test in unit if possible).
- Logs: new structured entries visible.
- No dupe logic.

**Risks/Mitigations (biggest phase):**
- Extraction changes HTTP behavior: copy-paste first into new func, then delete old; pair program or review diffs; run *all* routes_*_test + integ + manual (harmony, gemma4, structured, web_search exp, remote models).
- Stream error semantics (gin.H): map in bridge for HTTP; gRPC uses proper status + optional msg on stream.
- Long funcs: keep extracted <60 LOC where possible; helper funcs.
- Tool/orderedmap/JSON quirks: test converters exhaustively.
- Cloud checks dupe: centralize in modelRef parse or new `validateForGRPC` .
- Effort L (Chat logic is the beast: 2422 + render + loops + native).

### Phase 3: Proto Surface + Streaming MVP — Effort: M (2-3 days)
**Goal:** Full proto defs + working gRPC Chat stream + others; testable.

**Tasks:**
1. Flesh protos (Phase 0 skeleton -> complete for MVP): add all fields from api/types (ChatRequest/Response, Generate*, Embed*, ListResponse models, Version, Progress for future, Usage, Options as Struct or repeated, Message with images/tool_calls, Tool/ToolCall/Function, ThinkValue (oneof bool/string?), Format, Logprobs etc. Enrich: add oneof content, Tool oneof per xai [grpc-research:51], FinishReason enum, SamplingUsage. Use imports for timestamp/any/struct.
2. Regenerate: `buf generate` (update gen/).
3. Complete adapters in `server/grpc.go`: impl all MVP rpcs (Chat/ChatStream, Generate/GenerateStream, Embed, List, Show, Ps, Version). Use extracted from Phase 2.
4. Converters complete + tests.
5. Interceptors full (add to register).
6. Basic health/reflection (if grpc side: import grpc health/reflection; or connect equiv; register on mux).
7. Test:
   - `grpcurl -plaintext -d '{"model":"llama3.2"}' localhost:11435 ollama.api.v1.ChatService/ChatStream` (or buf curl --protocol grpc).
   - Small Go client in `api/examples/grpc_client.go` or temp _test.
   - Unit: mock *Server or use real with test sched? (hard; use integration later).
   - Cancel: client cancels mid-stream -> server stops (check logs/GPU?).
   - Error cases: bad model -> NotFound code; caps -> Invalid; etc.
8. Update `server/routes.go` register call in ServeGRPC/setup.

**Files:**
- `proto/ollama/api/v1/*.proto` (expand)
- `server/grpc.go` (impls)
- `gen/proto/...` (regenerated, commit)
- New: `integration/grpc_test.go` (or extend utils; build-tagged?).
- `api/examples/` (optional client).

**Commands:**
```sh
buf generate
go test ./server -race
go build .
# run server
grpcurl --plaintext --proto proto/ollama/api/v1/chat.proto list localhost:11435
buf curl --schema proto/ollama/api/v1 --protocol grpc --http2-prior-knowledge --data '{"model":"tinyllama", "messages":[{"role":"user","content":"hi"}]}' http://localhost:11435/ollama.api.v1.ChatService/ChatStream
go test -c -tags=integration ./integration  # later
```

**Verification:** Working streams (tokens arrive incrementally, done=true on last, usage), unary, errors have codes, reflection works (`grpcurl describe`), same scheduler behavior (ps shows loads from both ports).

**Risks:** Proto evolvability (use reserved, deprecate fields); streaming chunking (match api deltas); buf gen diffs (always commit clean).

### Phase 4: Polish, Compat, Observability, Tests — Effort: M (2 days)
1. Full auth metadata interceptor (even if no-op for local; parse bearer or custom).
2. OTEL: full interceptor setup (provider from env or noop); custom attrs + span around inference; slog bridge if possible.
3. Error mapping complete (all from handleScheduleError, StatusError, AuthorizationError, etc.).
4. Integration tests: update `integration/utils_test.go:430` startServer to also `t.Setenv("OLLAMA_GRPC_HOST", "127.0.0.1:11436")` or conditional (add OLLAMA_TEST_GRPC=0 to disable for matrix); new grpc tests using connect client (spawn binary or direct if refactored); port mutex safe (serial anyway).
5. Unit tests expand (converters, extractors with &Server{sched: mock or real init?}).
6. Docs: `docs/api.md` add "## gRPC API" section (endpoints, examples with grpcurl/buf curl, protos, ports, env); `docs/api/grpc.mdx`?; update openapi? (no, http only); README mention; envconfig docs.
7. Linting: ensure `golangci-lint run` (add if needed); `buf lint`.
8. Observability demo: logs with rich attrs; basic trace if exporter.
9. Backcompat: confirm HTTP 100% (diff large test runs); gRPC additive.
10. Version surface: gRPC Version() returns version.Version.

**Files:** `integration/utils_test.go`, `server/grpc.go` (polish), `docs/api.md`, `envconfig/config.go` (docs), tests, perhaps `logutil` if bridge.

**Verif:** Full test suite green; manual dual port use (http + grpc concurrent on same model -> shared sched, no OOM); logs/obs visible; grpcurl works cleanly.

**Risks:** Test port collisions (use random or disable grpc in integ by default, opt-in new tests); OTEL dep bloat (optional providers).

### Phase 5: Advanced/Optional — Effort: L (post-MVP, 1-2w+)
- cmux: add `github.com/soheilhy/cmux`; in cmd: cmux.New(lnHTTP), match grpc -> grpc handler, else gin; update ServeGRPC to not listen but take matched. Opt-in `OLLAMA_GRPC_SAMEPORT=1`. (High risk to existing; thorough soak.)
- Full admin: Pull/Push as server-stream Progress; Create/Delete etc.
- Image/Video: mirror xai deferred (StartDeferred + poll), batch.
- Buf CI: `.github/workflows` buf lint/breaking/format; pre-commit.
- Official clients: update `api/client.go`? or new grpc client; Python etc (via buf).
- Richer protos: full xai-style (search, citations, encrypted_content, agent_count, max_turns, store_messages, deferred for long).
- Auth: mTLS or full ed25519 metadata signing for cloud gRPC.
- Perf: load tests (slow client stream, high conn, cancel storm); consider API/engine split if backpressure issues.
- Reflection/health full prod.
- Breaking changes: reserved fields, semver via v2 dir.

**Risks:** cmux complexity (http2 plaintext, tls from prior PRs, pprof mux); defer until stable.

---

## 4. Risks, Gotchas, Mitigations (Synthesized from Reports)

**From [ollamas-review:192-205] + [grpc-research:136-148]:**
- **Gin coupling (bind, c.Stream, gin.H errors, writer hacks in middleware):** Mit: Phase 2 extraction to ctx+api only; thin wrappers. Test diff.
- **Cloud intertwining in *every* handler (modelRef.Source==Cloud -> proxyCloudJSONRequest gin-tied):** Mit: early guard in adapters + extracted (error "HTTP-only for cloud"); centralize parse. No gRPC cloud MVP.
- **Scheduler/GPU contention:** Mit: **mandatory single *Server* + sched shared** (extract setup); never dual init. Concurrent http+grpc loads on same model must serialize via existing logic.
- **Streaming backpressure/GPU leaks/hung runners:** Mit: ctx-first everywhere + cancel prop (research critical); bounded chans (reliable); select on done in writers; test explicit cancel mid-gen; monitor runner ctx in llm layer.
- **Low log coverage (LogLoom: ~28 nodes in routes.go vs many paths; sched clustered):** Mit: add rich slog (model+ids+reasoning) in *all* gRPC + extracted paths; improves coverage; use TraceContext.
- **Integration test mutex/ports/serial binary exec (utils_test.go:444):** Mit: set OLLAMA_GRPC_HOST in startServer (distinct port) or env-disable; new grpc tests opt-in; serial ok.
- **C++ side isolation:** Mit: pure Go addition; runners http unchanged.
- **Refactor/HTTP regression risk:** Mit: thin extract; pre/post test runs; feature additive (separate port).
- **Binary protos harder debug:** Mit: Connect (JSON over http too), reflection + grpcurl/buf curl, rich errors.
- **Auth/security (often unauthed):** Mit: interceptor early; local permissive (loopback); docs for prod (firewall, mTLS later).
- **High conn/long streams:** Mit: watch limits; keepalive; ctx deadlines.
- **Versioning/compat:** Mit: buf FILE breaking; major dirs; additive only.
- **Reliable-go violations:** Mit: explicit checklist in each phase (above); ctx, errs, slog, bounded, errgroup, no fire-forget, rich logs.
- **Deps/maintenance dual surface:** Mit: thin + extract minimizes; http primary.
- **Build/CI:** Mit: buf in dev prereqs; commit gen; verify in PR (buf generate --diff?).

**Other:** Experimental (harmony, web_search, imagegen) not regressed (test); remote models handled; large responses (embed floats, images) ok in pb.

Apply checklist before any edit: ctx first? errs wrapped? logs with ids? bounded? etc.

---

## 5. Dependencies & Tooling
- **go.mod (new runtime):** connectrpc.com/connect, connectrpc.com/otelconnect, go.opentelemetry.io/otel{,/sdk,/exporters/...}, google.golang.org/grpc (codes/status), golang.org/x/sync (errgroup already present).
- **Tool (codegen, not runtime):** buf (cli), protoc-gen-go + protoc-gen-connect-go (go get -tool).
- **Buf:** buf.yaml/buf.gen.yaml (as sketched); `buf lint`/`buf generate`/`buf breaking` (Phase 5 CI).
- **Codegen workflow:** Edit .proto -> `buf generate` -> commit gen/ + .proto. (No protoc direct.)
- **Testing:** `go test -race`; integration (binary + grpc client); manual grpcurl/buf curl; future load/soak.
- **Lint:** golangci-lint (existing); buf lint.
- **Other:** (Optional) protovalidate for rules.

No change to CMake/llama.cpp.

---

## 6. Phased Rollout & Backcompat Strategy
- **Env/flag:** `OLLAMA_GRPC_HOST` (default "127.0.0.1:11435" -> always-on separate). Disable: `OLLAMA_GRPC_HOST=""` or invalid port (add guard). No OLLAMA_ENABLE_GRPC needed initially (additive, low risk).
- **Dual maintenance:** HTTP primary/unchanged forever (or until far future). gRPC additive surface. Keep both in sync via shared extracts.
- **Testing matrix:** Integ serial; add grpc-specific tests behind build tag or env.
- **Docs/UX:** Announce in release notes: "gRPC API (experimental, port 11435 default, see docs/api.md)". grpcurl/buf curl examples.
- **Deprecation:** None for REST (non-goal). Future: if desired, mark some /api as legacy but keep.
- **Clients:** Existing unchanged. New gRPC users use generated or high-level (Phase 5).
- **Monitoring:** Watch adoption (logs?), issues on gRPC paths.

---

## 7. References
- Subagent reports: `/tmp/grpc-research-report-1.md` (esp. exec [11], xai lessons [44-62], table [34], pitfalls [136], recs [164]); `/tmp/ollamas-grpc-codebase-review.md` (arch [12], entrypoints [96-121], rec [209-212], gotchas [192], files [179], absolute paths [224]).
- LogLoom: `logloom-graph.json` (routes/s sched nodes; gaps analysis).
- xai-proto: https://github.com/xai-org/xai-proto (protos, buf.yaml [fetched], chat.proto modeling [grpc-research:50]); xai-sdk-python for telemetry/wrappers.
- Ollama: #10085; workspace files with lines as cited (e.g. server/routes.go:1917 Serve per both reports).
- Frameworks: connectrpc.com (getting-started, otel), buf.build/docs, grpc.io (perf, health, deadlines, OTEL), soheilhy/cmux.
- Reliable: `/Users/jonathandoughty/.grok/skills/reliable-go-systems/SKILL.md` (checklist applied).
- Other: Triton/vLLM/KServe patterns [grpc-research:66-71]; CONTRIBUTING.md (backcompat, new deps).

**Citations in text use [report:line] shorthand.** All grounded in primary reads.

---

**End of Plan.** Implementer can follow phases sequentially without reading 1000s LOC (key anchors + extracts provided). After Phase 4: working, observable, safe gRPC Chat/Embed/Models on dedicated port, zero HTTP impact, shared core/scheduler.

Next: land Phase 0 PR (proto + deps + buf), then iterate.

(Tracked via 8+ todo_write items; synthesis complete.)
