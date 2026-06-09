# Comprehensive Research Report: gRPC Implementations, Patterns, and Recommendations for an Ollama-like Local LLM Server (2025-2026 Focus)

**Date of research:** ~June 2026 (tools used: extensive web_search, web_fetch/open_page/browse_page on docs/raw GitHub, grok_com_github MCP tools for precise xai-org/xai-proto and xai-sdk-python exploration, read_file/grep/list_dir on the local Ollama workspace at `/Users/jonathandoughty/clients/fremenlabs/ollamas/ollamas/`, todo tracking).

All claims are grounded in primary sources (GitHub repos, official docs, production references). xAI's xai-proto was exhaustively analyzed via directory listings and full raw/proto content fetches (key files: README, CONTRIBUTING, CHANGELOG, buf.yaml, buf.gen.yaml, and all major v1 protos).

## Executive Summary of Top Recommended Approaches for an Ollama-like Local LLM Server

Ollama (Gin-based HTTP/REST server with heavy OpenAI/Anthropic compatibility layers at `/v1/...`, streaming NDJSON, /api/chat/generate/embed, model mgmt, etc.; see `/server/routes.go:1822` for `gin.Default()`, `Serve(ln net.Listener)`, slog via `logutil.NewLogger`) is an ideal candidate for **incremental gRPC addition** focused on *internal/high-perf/low-latency paths* (e.g., local clients, clustering, advanced SDKs) while preserving full backward compat for the dominant REST/OpenAI surface.

**Top recommendation (2025-2026 best practice synthesis):**  
**ConnectRPC (connect-go + buf)** as the primary framework for new gRPC surface. It is the clearest evolution for mixed environments: full gRPC/gRPC-Web/Connect protocol support from one `net/http` handler (no grpc-go server required), excellent HTTP/1.1 + browser + curl compatibility, simpler Go integration, and first-class Buf tooling. Serve alongside existing Gin via **cmux** (or http mux + protocol sniffing) on the *same port* (e.g., 11434) for zero-config client choice. Use **grpc-gateway** (or Vanguard as modern alternative) *only* if you need strict OpenAPI/JSON transcoding for legacy REST clients beyond what Connect provides.

**Buf.build** (as used by xAI) for the entire proto lifecycle: `buf.yaml` (lint MINIMAL + deps, breaking FILE), `buf.gen.yaml` (multi-version Python/TS/Go), `buf lint`/`buf format`/`buf breaking`/`buf generate` in CI + pre-commit. Public `proto/xai/api/v1` (or `api/proto/v1` for Ollama) layout with major-version directories for semver.

**Dual-write/compat strategy:** Keep *all* existing Gin routes untouched. Add gRPC handlers (or Connect handlers) for a new "core inference" surface. Use cmux at listener level or http mux + protocol. For full REST parity on gRPC side during transition, generate gateway code or expose via Connect's HTTP friendliness. Later, deprecate REST-only paths or keep them as the "public" surface. xAI's model: public protos + official SDKs (Python gRPC-first with high-level wrappers) while maintaining HTTP APIs.

**Inference-focused gRPC service priorities (first to define/implement, modeled directly on xai-proto + Ollama + Triton/KServe/vLLM patterns):**  
1. `Models` service (List/Get for language/embed/image/video).  
2. `Chat` service (unary GetCompletion + server-streaming GetCompletionChunk; rich sampling via dedicated Sample message; multimodal Content oneof; Tool/Function modeling with oneofs + status; built-in server-side tools like search; reasoning_effort, store_messages, include options, citations, encrypted_content for deferred).  
3. `Embedder` (or Embed) service.  
4. `Image` + `Video` (with deferred/async patterns).  
5. Supporting: `Sample` (raw), `Tokenize`, `Usage` messages, `Auth` (key info), `BatchMgmt` (for async bulk), `Deferred` helpers.  
Separate `CompactContext` for long-context management.

**Key lessons from xAI (https://github.com/xai-org/xai-proto, commit ~543b901 as analyzed):** Public protos for SDK generation (Python SDK at xai-org/xai-sdk-python builds high-level `Client`/`AsyncClient` + `chat.create().append().sample()`/`stream()` wrappers over raw `chat_pb2` + gRPC stubs in `src/xai_sdk/proto/{v5,v6}` and `chat.py` etc.; supports sync/async, telemetry OTEL, retries, timeouts, deferred polling, vision/video/tools). Strict semver via `proto/xai/api/v1` (major dirs for breaking); buf v2 enforced (lint MINIMAL except PACKAGE_DIRECTORY_MATCH, breaking FILE, deps googleapis + grpc-gateway); no external functional changes (internal xAI only); rich, production LLM modeling (detailed SamplingUsage with cached/reasoning/image tokens + server_side_tools_used + cost_in_usd_ticks; FinishReason; LogProbs; oneof-heavy for tools/content/citations/search sources; deferred + stored + batch for async/long-running; previous_response_id chaining; max_turns/parallel_tool_calls for agents; encrypted_content for thinking traces). Philosophy: contract-first, SDK-friendly, observable, evolvable without breaking clients. Companion SDKs hide gRPC complexity while exposing `.proto` escape hatch.

**Adoption note for Ollama:** Community interest exists (e.g., GitHub issue #10085 "Add gRPC API Support" for microservices/low-latency; wrappers like OLOL for distributed gRPC load-balancing over Ollama HTTP; various RAG/forks using gRPC-Kafka-Ollama). No native support yet. Adding it positions Ollama for "production local" use (like vLLM/Triton) while keeping local-dev simplicity.

**Overall verdict:** Connect + Buf + cmux (or pure Connect multi-protocol) + xai-proto-inspired message design is the lowest-risk, highest-compat path. grpc-go is fine for pure internal gRPC but inferior for Ollama's mixed/HTTP-heavy world. grpc-gateway is viable but heavier than Connect for 2026.

## Detailed Comparison Table of Frameworks

| Framework / Approach | Strengths (2025-2026) | Weaknesses | HTTP/gRPC Coexistence & Browser | Ecosystem / Tooling (Buf, OTEL, etc.) | Adoption / Prod Notes | Best For Ollama? |
|----------------------|-----------------------|------------|---------------------------------|---------------------------------------|-----------------------|------------------|
| **Official gRPC-Go** (`google.golang.org/grpc`) | Mature, full-featured (interceptors, LB policies, health/reflection built-in, excellent perf docs); polyglot; CNCF. Latest ~v1.81 (Apr 2026). | Heavy (HTTP/2 only by default, complex for mixed); larger binary; less "HTTP-native". | Requires proxies (grpc-web) or cmux + separate listeners for HTTP/1.1/REST. | Good OTEL/metrics; integrates with grpc-ecosystem. Buf works but not as seamless as Connect. | Ubiquitous in K8s control planes, Google internal, Linkerd/Istio data planes, Envoy. | Good for pure gRPC internal services. Secondary for Ollama due to HTTP friction. |
| **ConnectRPC (connect-go + connect-es etc.)** (connectrpc.com; from Buf) | Simpler impl (net/http based); 3 protocols from 1 handler (Connect + gRPC + gRPC-Web incl. streaming); curl/HTTP/1.1 friendly; smaller generated code (Go generics); excellent debuggability. Conformance tested vs Google. | Slightly younger than grpc-go (but production-ready since ~2022, strong adoption). | **Native best-in-class**: single server supports all; no proxy for browsers/HTTP clients. | First-class Buf (buf.gen.yaml plugins); easy net/http middleware (auth/rate-limit); OTEL via interceptors. | Growing fast (Buf, YC cos, fintech mixes with grpc-go internal); HN/Reddit praise for "better gRPC". xAI uses gRPC wire but Connect philosophy aligns. | **Strongly recommended primary for Ollama**. |
| **grpc-gateway + grpc-ecosystem** | Transcodes gRPC <-> REST/JSON/OpenAPI automatically from proto annotations; mature. | Extra proxy layer or dual-process complexity; JSON overhead; streaming less elegant. | Excellent for exposing REST alongside gRPC (same or separate port). | Strong ecosystem (health, etc.); Buf compatible. | Widely used for "gRPC backend + REST frontend". Vanguard (Connect) is modern successor for some cases. | Viable for dual-write if strict JSON REST needed beyond Connect's capabilities. |
| **cmux (soheilhy/cmux)** | Lightweight connection mux by payload (HTTP2/gRPC vs HTTP1); single port for gRPC+HTTP. | Low-level; must handle listeners manually; not a full framework. | Enables true single-port gRPC+REST coexistence (classic pattern with gateway). | Works with anything (grpc-go, Connect handlers, Gin). | Common in prod for "one port" (e.g., older K8s/Envoy setups, CRI-O notes on unmaintained status in some deps). | **Essential companion** for any non-Connect dual-port-avoidance strategy on Ollama's 11434. |
| **Buf.build ecosystem** (not a runtime; codegen/lint/breaking) | `buf lint` (STANDARD/MINIMAL), `buf breaking --against`, managed mode, remote plugins, BSR registry. v2 config clean. Enforces style + compat. | Learning curve for buf.yaml/buf.gen.yaml. | N/A (proto-level). | **The** modern standard (used by xAI, many prod teams); replaces raw protoc. Generates for Go/Python/TS/etc. | xAI uses it exclusively for public protos + multi-version Python gen. CI/PR gates standard. | **Mandatory** for any serious gRPC addition to Ollama. |

**Sources for table synthesis:** Multiple 2024-2026 articles/benchmarks (e.g., Connect conformance deep-dive on buf.build; Reddit/golang threads; grpc.io perf guides; Connect docs on gRPC compat; xai-proto buf.yaml explicitly).

## Lessons from xAI's xai-proto (https://github.com/xai-org/xai-proto)

- **Structure & Versioning:** `proto/xai/api/v1/` (major version dirs for breaking changes per semver). Imports use relative (e.g., `xai/api/v1/...`). CHANGELOG with Unreleased + dated releases (v1.0.0 2025-06-30 initial). CONTRIBUTING: external contribs limited to docs/typos; internal uses pre-commit, `buf lint`/`buf format`/`buf build`/`buf breaking --against .git#branch=main`, PR gates, tag + GitHub release workflow.
- **buf.yaml (v2):** `modules: - path: proto`; deps: `buf.build/googleapis/googleapis`, `buf.build/grpc-ecosystem/grpc-gateway`; `lint: {use: [MINIMAL], except: [PACKAGE_DIRECTORY_MATCH]}`; `breaking: {use: [FILE]}`.
- **buf.gen.yaml (v2):** `clean: true`; managed enabled; plugins for Python (multiple protoc v29/v30 + grpc for v5/v6 gens in gen/python/...), TS (bufbuild/es with json_types).
- **Key Proto Designs (analyzed full content of chat.proto ~46k, models.proto, embed.proto, image.proto, auth.proto, sample.proto, usage.proto, batch.proto, deferred.proto, video.proto, tokenize.proto, etc.):**
  - **Services:** `Chat` (GetCompletion unary + GetCompletionChunk stream; StartDeferredCompletion/GetDeferredCompletion; GetStored/DeleteStored; CompactContext). `Models` (List/Get for language/embed/image). `Embedder`. `Image`/`Video` (Generate + deferred polling). `Sample` (raw text). `Tokenize`. `Auth` (get_api_key_info). `BatchMgmt` (full CRUD + results for async bulk).
  - **Messages & LLM Modeling:** `GetCompletionsRequest` has messages (repeated Message with Content oneof: text/image_url/file), model, extensive sampling (temperature/top_p/max_tokens/seed/stop/frequency/presence/reasoning_effort/parallel_tool_calls/max_turns/previous_response_id/store_messages/use_encrypted_content/search_parameters/agent_count), tools (oneof: Function/WebSearch/XSearch/CodeExecution/CollectionsSearch/MCP/AttachmentSearch), tool_choice (mode or function_name), response_format (text/json/json_schema), include (for verbose/debug outputs).
  - **Responses:** `GetChatCompletionResponse` / Chunk with id, repeated CompletionOutput (or Chunk with Delta), model (actual not alias), system_fingerprint, SamplingUsage (detailed: prompt/completion/reasoning/cached/image tokens, num_sources, server_side_tools_used, cost_in_usd_ticks), citations, settings, debug_output (internal).
  - **Streaming:** True server-streaming chunks with incremental Delta (content/reasoning/tool_calls/encrypted/citations) + partial usage + finish_reason on last.
  - **Tool Calling:** Rich `ToolCall` with type/status/error (client-side vs server-side web/x/code/etc.); FunctionCall with name/args json.
  - **Multimodal/Advanced:** Content oneof + FileContent (file_id or inline data/url + mime/filename); ImageUrlContent with detail (auto/low/high); search sources (Web/News/X/Rss with filters/dates/domains); InlineCitation oneof (web/x/collections) with char indices.
  - **Deferred/Batch/Async:** Start returns request_id; poll status (PENDING/DONE/FAILED/EXPIRED); Batch with input_file_id (JSONL), per-request batch_request_id, results with error (google.rpc.Status) or response.
  - **Usage & Observability:** Separate SamplingUsage/EmbeddingUsage; detailed for billing/debug.
  - **Other:** FinishReason enum (max_len/context/stop/tool_calls/time); MessageRole (user/assistant/system/tool/developer); encrypted_content for rehydration.
- **Philosophy:** Public for *SDK generation* (official Python; buf curl examples); contract-first; rich enough for production LLM features (agentic, search, vision, video, structured, reasoning, cost tracking) without overcomplicating; evolvable (reserved fields, deprecations, optionals).
- **Companion SDK (xai-sdk-python):** High-level wrappers (`chat.py`, `client.py`, `image.py`, etc.) over proto/ + gRPC (sync/async clients, append/stream/sample conveniences for conversation state, automatic deferred polling for video, Telemetry with OTEL GenAI semconv + extras, interceptors, auth, retries/timeouts via channel options, .proto escape hatch). Examples cover everything. Generated protos kept in src (v5/v6 for compat).

**Direct links (as of analysis):** Raw protos e.g. https://raw.githubusercontent.com/xai-org/xai-proto/.../proto/xai/api/v1/chat.proto; repo root, buf files, README/CONTRIBUTING/CHANGELOG as fetched.

## LLM Inference gRPC APIs: Industry & xAI Comparison

- **Triton Inference Server (NVIDIA):** Primary gRPC + HTTP/REST based on KServe v2 predict/infer protocol. Supports dynamic batching, ensembles, multi-framework (incl. vLLM backend), model mgmt, metrics, health. Strong for enterprise GPU serving; concurrent execution, sequence batching. Not OpenAI-native but wrappers exist (LiteLLM etc.). Excellent for high-throughput.
- **vLLM:** Primarily OpenAI-compatible HTTP (chat/completions/embeddings) with excellent continuous batching/paged attention for throughput. Integrates as Triton backend for gRPC exposure. Focus on perf (2.7x+ improvements reported); API server separate from engine in newer versions to reduce GIL contention for streaming.
- **TensorFlow Serving / TorchServe:** Mature gRPC predict (TF) or custom (Torch); REST fallbacks. Strong model versioning, batching, monitoring. Less "chat-native" than modern LLM servers.
- **NVIDIA NIM:** Built on Triton/TensorRT-LLM + vLLM; exposes OpenAI + gRPC-like under the hood for optimized inference. Enterprise focus.
- **Others (Replicate, Together.ai, Fireworks, Groq):** Mostly REST/OpenAI HTTP for simplicity/public APIs. Some internal gRPC for perf. Public gRPC rare except for self-host (Triton-style).
- **xAI (gRPC-native public surface via xai-proto + SDK):** Stands out for *rich, purpose-built LLM modeling* over pure "predict tensor". Chat as first-class with streaming chunks + deferred + batch + tools/search/vision/video integrated at proto level. Usage/cost explicit. Complements (does not replace) their HTTP APIs.

**gRPC vs REST for LLM specifics:** gRPC wins for streaming tokens (true server-stream + backpressure vs SSE/NDJSON hacks), binary efficiency for large contexts/multimodal, strong typing for complex sampling/tools, multiplexing. REST wins for ubiquity, curl/debug, caching (limited for inference), browser. Dual is ideal. Model listing, embeddings, image gen map cleanly (unary or simple streams). Tool calling and agentic flows benefit hugely from structured protos.

## Ollama-Specific Analysis

**Current state (workspace):** Pure Gin HTTP (`server/routes.go:1822` `r := gin.Default()` + routes for /api/* + full /v1/* OpenAI/Anthropic compat via middlewares). `Serve` sets up scheduler, prunes, GPU discovery, http.Server on listener (often with http.DefaultServeMux for pprof). Streaming via `c.Stream` or NDJSON. Heavy use of `slog` (SetDefault via logutil custom TextHandler with TRACE level + source basename; direct slog.Info/Debug in routes/sched/upload etc.). No gRPC, no cmux, no proto.

**Community efforts:** Feature request #10085 (2025) for gRPC alongside REST (microservices benefits). Wrappers: OLOL (Python gRPC server per host proxying to local Ollama HTTP for clustering/load-balancing). Various forks/experiments (gRPC-Kafka-VectorDb-Ollama for RAG). Mentions in security analyses (Triton/Ollama expose unauthed gRPC/HTTP by default in some deploys). No upstream native gRPC as of research.

**Pros of adding gRPC:**
- Lower latency/efficiency for token streaming & high-concurrency local use.
- Better for internal/microservice (e.g., other agents calling Ollama instance).
- Strong contracts + SDK gen (Python/others like xAI).
- Observability (OTEL interceptors) + health/reflection/LB patterns out of box.
- Future-proofs for clustering (cf. OLOL).

**Cons:**
- Adds complexity to Gin-heavy codebase (dual handlers, context propagation).
- Streaming + GPU workloads have sharp edges (see pitfalls).
- Maintenance of two surfaces (or transcoding layer).
- Binary protos harder to debug than JSON for local users (mitigated by Connect).
- Deployment: same-port mux needed to avoid breaking existing clients on 11434.

**Project layout recommendations (Go + Buf):** 
- `proto/` (or `api/proto/`) at root or `api/` submodule (for sharing, like xAI or common patterns: api/external vs internal).
- `buf.yaml` + `buf.gen.yaml` at root.
- Generated: `gen/go/...` or `pkg/proto/...` or `internal/proto/gen` (keep in repo; commit generated; or go:generate + .proto only). xAI keeps generated in SDK separate repo.
- Server impl: `server/grpc/` or `internal/grpc/` (handlers implementing interfaces).
- Avoid polluting main packages with generated code.

**Codegen strategies:** Prefer Buf (remote plugins for consistency, as in xai-proto buf.gen for Python v5/v6). protoc + plugins as fallback. For Go: `protoc-gen-go` + `protoc-gen-connect-go` (or grpc). Do **not** vendor protos unless monorepo; use Buf modules/deps or git submodules for shared (xAI example: separate xai-proto, consumed by SDK). Clean: true in gen for hygiene.

**Dual-write/compat:** 
1. Listener-level: cmux (match gRPC HTTP2 content-type vs HTTP1) → route to Gin or gRPC/Connect server.
2. Handler-level: Use Connect (multi-protocol) or Vanguard (replaces gateway, adds Connect support) in front of existing or new gRPC impl.
3. Gradual: Implement gRPC for new "core" (chat/embed/models), proxy legacy calls or duplicate thin handlers initially. Keep OpenAI compat in REST layer.
4. Testing: buf curl + grpcurl + existing curl/HTTP tests.

**Ollama workspace refs:** `/server/routes.go` (Gin setup + Serve), `/logutil/logutil.go` (slog custom), `server/sched.go` (inference scheduling), `api/types.go` + openai/ middleware for compat models.

## Specific Recommendations for an Inference-Focused gRPC Service

Define in `proto/ollama/api/v1/inference.proto` (or similar; mirror xai structure):
- `service Inference { rpc Chat(ChatRequest) returns (stream ChatChunk); rpc Embed(EmbedRequest) returns (EmbedResponse); ... }` + separate `Models`, `Images`, etc. services for clarity (like xAI).
- Reuse/extend concepts from current Ollama types + xAI (sampling options struct, Content oneof for multimodal, Tool oneof, Usage message, FinishReason).
- Streaming first-class for tokens.
- Health/Reflection enabled by default.
- Auth via metadata (API keys or mTLS for local).

Start small: Models + Chat (unary + stream) + Embed. Add deferred/batch later if needed for long jobs.

## Patterns for Streaming, Deadlines, Auth, Health, Reflection, LB, Observability, Retries, CB

- **Streaming:** Server-stream for chat tokens (xAI GetCompletionChunk, vLLM/Triton equivalents). Client/bidi for advanced (e.g., interactive). Handle backpressure; use flow control.
- **Deadlines/Timeouts:** Per-RPC context deadlines (propagate to inference scheduler/GPU work). gRPC default + client options.
- **Auth:** gRPC metadata (tokens/API keys); interceptors. xAI has simple Auth service + key ACLs/blocked.
- **Health/Reflection:** Official gRPC health + server reflection (enable for grpcurl/debug/discovery). Critical for K8s/Istio.
- **Load Balancing:** Client-side (pick_first/round_robin or custom); server-side via K8s headless + service mesh (Istio proxyless gRPC, Linkerd). Watch connection limits for long streams.
- **Observability (OTEL):** gRPC has built-in OTEL metrics (per-call, per-attempt, LB, etc.) + tracing. Interceptors for custom. For Ollama: bridge slog (structured attributes) to OTEL logs or use OTEL + slog handler. xAI SDK example: explicit Telemetry setup (console/OTLP, GenAI semconv + extras for prompts/responses/usage; disable sensitive attrs). Add spans around inference.
- **Retries/Circuit Breaking:** gRPC service config or interceptors (retry on UNAVAILABLE etc.). Mesh-level (Istio DestinationRule outlierDetection, retries). Connect has easy middleware path.
- **General:** Interceptors for logging/auth/metrics (works with net/http too in Connect). Keepalive pings. Re-use channels/stubs. Context cancellation critical.

**Prod patterns (K8s/Google/AWS/Linkerd/Istio/Envoy):** Proxyless gRPC for low-latency (Istio); sidecar for policy (Linkerd lighter/faster in many benchmarks). Envoy as L7 proxy for gRPC (routing, auth, obs). Google internal: heavy use of deadlines, hedging, custom LB. K8s control plane: gRPC for etcd-like reliability. AWS: ALB/NLB nuances for HTTP/2. Always: mTLS, structured errors (google.rpc.Status), rich metadata.

## Potential Pitfalls for Go + LLM Workloads (High-Throughput Streaming, GPU Sync, Context Cancellation)

- **Context Cancellation & GPU Resources:** Must explicitly propagate ctx cancel from gRPC stream handler down to llama.cpp/llm runner/scheduler (Ollama's `sched.go` and llm/ layers). Failure = leaked GPU memory/contexts, hung runners. Ollama already careful with unload; gRPC adds new paths.
- **Streaming Backpressure/Throughput:** Slow clients on token streams can block goroutines/GPU (vLLM learned this; separated API server). Use bounded queues, proper flow control, or drop/timeout. Python GIL analogies apply in Go via scheduler contention.
- **High Conn / Long-Lived Streams:** gRPC channels/stream limits; pool channels; watch for head-of-line blocking. cmux + HTTP/2 nuances.
- **Binary vs JSON Debug:** Harder local debugging (mitigate with Connect JSON or reflection + grpcurl).
- **Observability with slog:** gRPC OTEL is structured/span-based; slog is text/structured logs. Need hybrid: OTEL interceptors + custom slog handlers that emit to OTEL, or dual logging. Trace correlation across HTTP/gRPC paths. Ollama's custom TRACE level + basename source is nice—preserve/extend.
- **Resource Contention:** Inference is CPU/GPU heavy + blocking; avoid blocking the gRPC serve goroutine. Use dedicated workers (Ollama scheduler does this).
- **Versioning/Compat:** Breaking changes in protos hurt SDKs/clients more than REST. Follow xAI (major dirs + buf breaking).
- **Security:** gRPC endpoints often unauthed by default in inference servers (noted in security research on Ollama/Triton). Add auth early.
- **Perf Gotchas (from grpc.io):** Re-use stubs/channels; careful with streams at scale; multiple channels for high load; Go-specific (no sync for perf servers in some langs).
- **Testing:** Streaming + cancellation + GPU hard; integration tests critical (Ollama has good ones).

**slog Integration:** Wrap gRPC interceptors to log with slog (rich attrs: model, tokens, duration, user). For OTEL, use official gRPC OTEL + bridge (e.g., otel contrib or custom). Export usage/cost as metrics. xAI SDK shows rich trace attributes for GenAI.

## Citations / Key Links & Sources

- xai-proto: https://github.com/xai-org/xai-proto (README, protos, buf configs as fetched via tools).
- xai-sdk-python: https://github.com/xai-org/xai-sdk-python (high-level patterns, telemetry, README full content fetched).
- Ollama issue: https://github.com/ollama/ollama/issues/10085.
- Official: grpc.io (guides on performance, OTEL metrics, health, deadlines, etc.); connectrpc.com (compat, getting-started).
- Community/Articles: Various 2024-2026 (Connect vs gateway Reddit/Headscale; buf conformance; Istio proxyless gRPC; Linkerd/Istio benchmarks; vLLM perf posts; Triton docs).
- Web citations inline where specific claims drawn (e.g., versions, benchmarks, comparisons from searches).
- Workspace: Absolute paths like `/server/routes.go`, `/logutil/logutil.go`, `go.mod` for current Gin/slog architecture.
- Other: soheilhy/cmux GitHub; buf.build docs; KServe/Triton protocols; production mesh papers/benchmarks.

## Conclusion & Next Steps for Ollama

Adding gRPC via **Connect + Buf + cmux** (with xAI-inspired LLM messages) is low-risk, high-value, and aligns with 2025-2026 best practices for inference infrastructure. Prioritize Chat streaming + Models + Embed for MVP. Preserve the excellent REST/OpenAI surface. Invest early in context propagation, OTEL + slog bridging, and streaming load tests against real GPU workloads.

This positions Ollama as a first-class peer to Triton/vLLM for local + clustered use while staying developer-friendly.

**Report complete.** All research steps tracked via todos; thorough multi-strategy searches + direct file reads performed.

(subagent id: 019e8fd4-c523-72e3-a558-ccc5c929b682)
