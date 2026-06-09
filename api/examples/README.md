# Ollama API Examples

Run the examples in this directory with:

```shell
go run example_name/main.go
```

## Chat - Chat with a model
- [chat/main.go](chat/main.go)

## Generate - Generate text from a model
- [generate/main.go](generate/main.go)
- [generate-streaming/main.go](generate-streaming/main.go)

## Pull - Pull a model
- [pull-progress/main.go](pull-progress/main.go)

## gRPC Client (Phase 4 agent/Flume productionization, report sec4 item 9)
Official gRPC/Connect client support skeleton added in `api/grpc_client.go` (uses generated apiv1connect for Chat/Generate/Embed/Models unary+streams; no new deps).
See source comments + `docs/grpc-phased-reliable-approach.md` "Agent/Flume..." subsec for details (ctx first, retry on errToConnect codes + jitter, rich slog).

**Flume/agent examples patterns (ctx/cancel/structured logs/health; use in your agent loops):**
- **Ctx prop + mid-stream cancel (for GPU stop, no leak):** `ctx, cancel := context.WithCancel(ctx); stream, _ := c.ChatStream(ctx, connect.NewRequest(&v1.ChatRequest{Model: m, Stream: true, Messages: ...})); for stream.Receive() { if shouldStop() { cancel() }; ... }` (exercises handler's streamCtx, select on Done, cancel to core per phased p334 + report p66 + SKILL Context is King).
- **Structured/reason logs for LogAgentReasoning:** Server (P2+) + client emit `slog` with `component`, `reason` (at decisions: schedule/evict/retry/ctx/cancel/done_reason), `stream_id`, `model`, `rpc`, `duration_ms`, `status`, `tokens`. Agents/Flume feed to graphs/postmortems (SKILL p35/57 + phased p5/374 + report p67-68). E.g. client "grpc client retry decision" reason="connect code Unavailable from errToConnect (queue backpressure); jitter...".
- **Health check:** `if err := c.Heartbeat(ctx); err != nil { ... };` then stream (or Version/List). Complements server health skeleton (P3 + this phase; gated refl + always health in registerServices).
- Run examples with gRPC: set OLLAMA_GRPC_HOST (or SAMEPORT=1); construct via NewGRPCClient("http://"+host, connect.WithGRPC()) or GRPCClientFromEnvironment. Fallback to HTTP Client for :cloud etc.
- Load/soak under agent conc: see phased doc soak skeleton (concurrent GRPCClient streams, backpressure via MaxQueue->CodeUnavailable->retry, pprof/metrics, health, -race, rich logs). Harness serial; manual for N-way.

All follow reliable-go-systems/SKILL.md (10pt: ctx first I/O, %w+Is/As classify no ignores, bounded no fire-forget, rich slog+reason everywhere, idempotent safe retry w/ jitter, small units, no new globals, verif -race/table, Flume LogAgent + runner respect).

## Notes
- These examples use the HTTP `api.Client`. For gRPC equivalent, import the connect client or the new GRPCClient wrapper once wired in examples.
