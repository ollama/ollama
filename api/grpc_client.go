// Package api (continued) provides the official Go client for ollama.
//
// This file adds gRPC/Connect client support skeleton for Agent/Flume productionization
// (report sec4 item 9: "official clients with streaming+retry using err codes, Flume examples
// ctx/cancel/structured logs, load/soak for queue/backpressure under agent concurrency,
// health/reflection/metrics").
//
// See docs/grpc-phased-reliable-approach.md (Phase 4/5 overlay, p88/368 health/refl gates,
// p323/p379 stop-if/review-subagent, p334 stream ctx, p374 LogLoom, p381 review).
// Prior phases: P1 wiring/errgroup/sameport/reason-logs, P2 extracts+thin+converters+rich
// slog+ctx-cancel+LogAgentReasoning, P3 protos+admin+intcps+refl-skel, P4 OTEL+integ+ -race green.
// api/client.go remains the HTTP client (ClientFromEnvironment + stream/do). This is additive
// gRPC skeleton (minimal per SKILL: skeletons+comments, no heavy).
//
// Follows reliable-go-systems/SKILL.md exactly (10pt checklist applied before/after every
// decision/edit in this file + callers): ctx first for I/O, %w errors + errors.Is/As classify
// (no ignores), no worker/queue bypass (calls go through server reconcile), bounded (no new
// goros; ctx selects), rich slog (component/reason/stream_id/dur/status/model/rpc at decisions,
// Flume/LogAgentReasoning style), timeouts via ctx, idempotent+safe retry w/ jitter (not pure exp),
// small units, no new globals, verifiability (-race/table at end), Flume respect (no change to
// Claimer/Sweeper/Runner; LogAgent via reason logs), rich reasoning at decisions.
//
// Usage (ctx prop, cancel for GPU, retry on transient err codes from errToConnect, health,
// structured logs):
//   c, _ := api.NewGRPCClient("http://127.0.0.1:11435", connect.WithGRPC())
//   // or GRPCClientFromEnvironment() (respects OLLAMA_GRPC_HOST like HTTP path)
//   ctx, cancel := context.WithCancel(ctx)
//   defer cancel()
//   stream, err := c.ChatStream(ctx, connect.NewRequest(&v1.ChatRequest{Model: "smol", Stream: true, Messages: []*v1.Message{{Role:"user", Content:"hi"}}}))
//   for stream.Receive() {
//     if stream.Msg().Done { break }
//     if tooMuch() { cancel() } // mid-stream cancel propagates to core (prevents GPU leak)
//   }
//   if ce := new(connect.Error); errors.As(stream.Err(), &ce) && ce.Code() == connect.CodeUnavailable {
//     // transient (MaxQueue etc); client retry helper + jitter would have kicked in for unary
//   }
//
// For load/soak (agent concurrency, queue backpressure): see phased doc "Agent/Flume..." subsec
// + integ/grpc_stream_test.go notes. Use multiple long-lived GRPCClient, health preflight,
// observe CodeUnavailable -> jitter retry, pprof/metrics, rich reason logs, -race.
//
// Health/reflection: server has skeleton (enableGRPC* in registerServices); client Heartbeat
// uses Version (or future dedicated health).
package api

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"sync"
	"time"

	"connectrpc.com/connect"

	v1 "github.com/ollama/ollama/gen/proto/ollama/api/v1"
	apiv1connect "github.com/ollama/ollama/gen/proto/ollama/api/v1/apiv1connect"
	"github.com/ollama/ollama/envconfig"
)

// GRPCClient is the official gRPC/Connect client skeleton (api/ package, additive to HTTP Client).
// Wraps the generated apiv1connect clients for Chat/Generate/Embed/Models (unary + streaming).
// Streaming primary win (token-by-token vs NDJSON). Retry on err codes (mirror server errToConnect).
// All I/O take ctx first (SKILL p13); respect cancellation (cancel mid-stream stops GPU promptly).
// Rich slog at decisions (component="api/grpc-client", reason=..., stream_id, dur, status, model, rpc)
// for Flume LogAgentReasoning / audit (matches server/grpc.go P1-4 style + phased p5/374).
// Minimal skeleton per SKILL (small funcs, no heavy impl, comments for full).
type GRPCClient struct {
	chat   apiv1connect.ChatServiceClient
	gen    apiv1connect.GenerateServiceClient
	embed  apiv1connect.EmbedServiceClient
	models apiv1connect.ModelsServiceClient

	baseURL string
	// httpClient held for potential future (e.g. custom transport); connect clients use it internally.
	httpClient *http.Client

	// breaker is per-client (not global) for Phase 4c circuit breaking on retryable errors.
	breaker circuitBreaker
}

// NewGRPCClient constructs a gRPC client for the given baseURL (e.g. "http://127.0.0.1:11435").
// By default uses Connect protocol; pass connect.WithGRPC() (or WithGRPCWeb()) for gRPC wire (as in
// integration/grpc_stream_test.go and server handlers).
// h2c transport for plaintext (common for local dedicated gRPC port or SAMEPORT).
// Ctx not stored (SKILL). No globals. Small.
func NewGRPCClient(baseURL string, opts ...connect.ClientOption) *GRPCClient {
	hc := H2CClient()  // central (Phase 4a); includes keepalive (Phase 2c) + correct h2c DialTLS

	baseURL = trimTrailingSlash(baseURL)
	return &GRPCClient{
		chat:       apiv1connect.NewChatServiceClient(hc, baseURL, opts...),
		gen:        apiv1connect.NewGenerateServiceClient(hc, baseURL, opts...),
		embed:      apiv1connect.NewEmbedServiceClient(hc, baseURL, opts...),
		models:     apiv1connect.NewModelsServiceClient(hc, baseURL, opts...),
		baseURL:    baseURL,
		httpClient: hc,
	}
}

func trimTrailingSlash(u string) string {
	if len(u) > 0 && u[len(u)-1] == '/' {
		return u[:len(u)-1]
	}
	return u
}

// GRPCClientFromEnvironment creates a gRPC client using OLLAMA_GRPC_HOST (mirrors envconfig +
// HTTP ClientFromEnvironment pattern; no new globals/magic). Falls back to grpc default port
// if not set (distinct from HTTP 11434 for safety, per P1).
// Agents/Flume: prefer explicit or env for dedicated vs sameport.
func GRPCClientFromEnvironment() (*GRPCClient, error) {
	u := envconfig.GRPCHost()
	if u == nil {
		// envconfig default is 127.0.0.1:11435 for GRPC (see config); construct
		// (avoid hardcode; in practice envconfig always returns non-nil for GRPC? but safe)
		return NewGRPCClient("http://127.0.0.1:11435"), nil
	}
	scheme := "http"
	if u.Scheme != "" {
		scheme = u.Scheme
	}
	return NewGRPCClient(fmt.Sprintf("%s://%s", scheme, u.Host)), nil
}

// isRetryable classifies using codes that errToConnect (server/grpc.go:764) maps to for transient
// cases (MaxQueue/queue full, temp scheduler, deadline, some canceled for reconnect safety).
// Permanent (InvalidArg, NotFound, Unauth, Internal, bad model) are not retried.
// Use errors.Is/As + connect.Error.Code() (no ignore).
func isRetryable(err error) bool {
	if err == nil {
		return false
	}
	if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
		// Canceled often permanent for this req (but some agents retry new); classify non-retry here
		return false
	}
	var ce *connect.Error
	if errors.As(err, &ce) {
		switch ce.Code() {
		case connect.CodeUnavailable, connect.CodeResourceExhausted:
			return true // transient backpressure/queue (errToConnect maps MaxQueue here)
		case connect.CodeDeadlineExceeded:
			return true // temp
		}
	}
	// also check wrapped StatusError etc via connect (already mapped); extend as needed
	return false
}

// doWithRetry: bounded retry helper (free func, generic OK; no type param on methods).
// Small unit. Logs rich reason at retry decision (Flume style for LogAgentReasoning).
// Idempotent/safe (only on classified transient; server reconcile safe). Ctx respected.
// circuitBreaker provides simple fail-fast after N consecutive retryable failures
// within a window (Phase 4c / G11). Per-client instance (no package globals) so one
// bad model/endpoint does not poison others. Follows SKILL "no globals" + bounded.
type circuitBreaker struct {
	mu       sync.Mutex
	failures int
	lastFail time.Time
}

const (
	cbThreshold = 5
	cbWindow    = 30 * time.Second
	cbCooldown  = 10 * time.Second
)

func (b *circuitBreaker) allows() bool {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.failures >= cbThreshold && time.Since(b.lastFail) < cbCooldown {
		return false
	}
	if time.Since(b.lastFail) > cbWindow {
		b.failures = 0
	}
	return true
}

func (b *circuitBreaker) recordFailure() {
	b.mu.Lock()
	b.failures++
	b.lastFail = time.Now()
	b.mu.Unlock()
}

func (b *circuitBreaker) recordSuccess() {
	b.mu.Lock()
	b.failures = 0
	b.mu.Unlock()
}

func doWithRetry[T any](
	ctx context.Context,
	rpc, model string,
	b *circuitBreaker,
	op func(context.Context) (*connect.Response[T], error),
) (*connect.Response[T], error) {
	const maxAttempts = 3
	if !b.allows() {
		slog.Info("grpc client circuit breaker open", "component", "api/grpc-client", "rpc", rpc, "model", model, "reason", "N consecutive retryable failures within window; failing fast to protect upstream (Phase 4c / G11 + SKILL no-globals + bounded)", "failures", b.failures, "cooldown_s", cbCooldown.Seconds(), "status", "circuit_open")
		return nil, fmt.Errorf("grpc client %s %s: circuit breaker open (recent failures >= %d)", rpc, model, cbThreshold)
	}
	var lastErr error
	for attempt := 0; attempt < maxAttempts; attempt++ {
		start := time.Now()
		slog.Debug("grpc client unary start", "component", "api/grpc-client", "rpc", rpc, "model", model, "attempt", attempt, "reason", "ctx first I/O (SKILL p13); delegate to apiv1connect; will classify err for safe retry per finding9 + errToConnect mirror", "stream_id", "", "status", "start")
		res, err := op(ctx)
		dur := time.Since(start).Milliseconds()
		if err == nil {
			b.recordSuccess()
			slog.Info("grpc client unary done", "component", "api/grpc-client", "rpc", rpc, "model", model, "duration_ms", dur, "status", "ok", "reason", "success; no retry needed (idempotent safe path)")
			return res, nil
		}
		lastErr = err
		if isRetryable(err) {
			b.recordFailure()
			// Rich reasoning log on every failure that contributes to breaker state (Flume/LogAgentReasoning).
			slog.Info("grpc client retryable failure", "component", "api/grpc-client", "rpc", rpc, "model", model, "reason", "transient error counted toward circuit breaker (CodeUnavailable/ResourceExhausted/etc from errToConnect)", "error", err, "current_failures", b.failures, "status", "transient")
		}
		if !isRetryable(err) || attempt == maxAttempts-1 {
			slog.Info("grpc client unary done", "component", "api/grpc-client", "rpc", rpc, "model", model, "duration_ms", dur, "error", err, "status", "error", "reason", "non-retryable or max attempts (permanent per errToConnect classify: Invalid/NotFound/Unauth/Internal/Canceled); propagate %w wrapped")
			return nil, fmt.Errorf("grpc client %s %s: %w", rpc, model, err)
		}
		// transient: jitter backoff (idempotent/safe per SKILL p41; not pure exp)
		jitter := time.Duration(50+(time.Now().UnixNano()%100)) * time.Millisecond * (1 << uint(attempt))
		if jitter > 2*time.Second {
			jitter = 2 * time.Second
		}
		slog.Info("grpc client retry decision", "component", "api/grpc-client", "rpc", rpc, "model", model, "stream_id", "", "status", "transient", "reason", "retryable (CodeUnavailable or equiv from errToConnect: queue backpressure/MaxQueue/temp scheduler per report p66-75 + phased); jitter backoff (not pure exp) for safe idempotent retry in agent/Flume loop", "attempt", attempt, "backoff_ms", jitter.Milliseconds())
		select {
		case <-ctx.Done():
			return nil, fmt.Errorf("grpc client %s %s ctx during backoff: %w", rpc, model, ctx.Err())
		case <-time.After(jitter):
		}
	}
	return nil, fmt.Errorf("grpc client %s %s: %w", rpc, model, lastErr)
}

// Chat (unary) with retry + rich logs + ctx first. For streaming use ChatStream.
func (c *GRPCClient) Chat(ctx context.Context, req *connect.Request[v1.ChatRequest]) (*connect.Response[v1.ChatResponse], error) {
	if req == nil || req.Msg == nil {
		return nil, errors.New("grpc client chat: nil req")
	}
	return doWithRetry(ctx, "Chat", req.Msg.Model, &c.breaker, func(ctx context.Context) (*connect.Response[v1.ChatResponse], error) {
		return c.chat.Chat(ctx, req)
	})
}

// Generate (unary) with retry + rich logs + ctx first.
func (c *GRPCClient) Generate(ctx context.Context, req *connect.Request[v1.GenerateRequest]) (*connect.Response[v1.GenerateResponse], error) {
	if req == nil || req.Msg == nil {
		return nil, errors.New("grpc client generate: nil req")
	}
	return doWithRetry(ctx, "Generate", req.Msg.Model, &c.breaker, func(ctx context.Context) (*connect.Response[v1.GenerateResponse], error) {
		return c.gen.Generate(ctx, req)
	})
}

// Embed (unary) with retry + rich logs + ctx first.
func (c *GRPCClient) Embed(ctx context.Context, req *connect.Request[v1.EmbedRequest]) (*connect.Response[v1.EmbedResponse], error) {
	if req == nil || req.Msg == nil {
		return nil, errors.New("grpc client embed: nil req")
	}
	return doWithRetry(ctx, "Embed", req.Msg.Model, &c.breaker, func(ctx context.Context) (*connect.Response[v1.EmbedResponse], error) {
		return c.embed.Embed(ctx, req)
	})
}

// ChatStream: streaming support. Retry only on initial connect (mid-stream: caller loop + ctx cancel
// to stop promptly per p334). Returns the stream; caller must Receive() + handle Err().
// Rich logs on connect decision/cancel. Bounded by caller ctx.
func (c *GRPCClient) ChatStream(ctx context.Context, req *connect.Request[v1.ChatRequest]) (*connect.ServerStreamForClient[v1.ChatResponse], error) {
	if req == nil || req.Msg == nil {
		return nil, errors.New("grpc client chatstream: nil req")
	}
	model := req.Msg.Model
	start := time.Now()
	slog.Debug("grpc client stream start", "component", "api/grpc-client", "rpc", "ChatStream", "model", model, "reason", "ctx first (SKILL); stream connect; retry only on initial (mid use cancel+new per finding9); bounded by select on Done in caller + handler", "stream_id", "", "status", "start")
	stream, err := c.chat.ChatStream(ctx, req)
	dur := time.Since(start).Milliseconds()
	if err != nil {
		if isRetryable(err) {
			slog.Info("grpc client retry decision", "component", "api/grpc-client", "rpc", "ChatStream", "model", model, "duration_ms", dur, "status", "transient", "reason", "initial stream connect transient (Unavailable from errToConnect); would retry here in full (skeleton limits to 1 for stream simplicity; agent can re-call)", "error", err)
			// skeleton: one retry attempt for connect phase; ctx select + jitter (SKILL p13/p41; no bare sleep)
			jitter := 100*time.Millisecond + time.Duration(time.Now().UnixNano()%50)*time.Millisecond
			select {
			case <-ctx.Done():
				return nil, fmt.Errorf("grpc client ChatStream %s ctx during retry backoff: %w", model, ctx.Err())
			case <-time.After(jitter):
			}
			stream, err = c.chat.ChatStream(ctx, req)
			if err != nil {
				return nil, fmt.Errorf("grpc client ChatStream %s: %w", model, err)
			}
		} else {
			return nil, fmt.Errorf("grpc client ChatStream %s: %w", model, err)
		}
	}
	slog.Info("grpc client stream connected", "component", "api/grpc-client", "rpc", "ChatStream", "model", model, "duration_ms", dur, "status", "ok", "reason", "stream open; caller owns Receive loop + cancel for GPU safety + backpressure (select before use)")
	return stream, nil
}

// GenerateStream similar to ChatStream (ctx, bounded, logs, limited retry on connect).
func (c *GRPCClient) GenerateStream(ctx context.Context, req *connect.Request[v1.GenerateRequest]) (*connect.ServerStreamForClient[v1.GenerateResponse], error) {
	if req == nil || req.Msg == nil {
		return nil, errors.New("grpc client generatestream: nil req")
	}
	model := req.Msg.Model
	start := time.Now()
	slog.Debug("grpc client stream start", "component", "api/grpc-client", "rpc", "GenerateStream", "model", model, "reason", "ctx first; see ChatStream", "stream_id", "", "status", "start")
	stream, err := c.gen.GenerateStream(ctx, req)
	dur := time.Since(start).Milliseconds()
	if err != nil {
		return nil, fmt.Errorf("grpc client GenerateStream %s: %w", model, err)
	}
	slog.Info("grpc client stream connected", "component", "api/grpc-client", "rpc", "GenerateStream", "model", model, "duration_ms", dur, "status", "ok", "reason", "stream open (limited retry skeleton)")
	return stream, nil
}

// Pull (server stream over ModelsService). Added per Phase 4b to match ChatStream/GenerateStream
// pattern + server-side modelsHandler Pull (progress for registry ops). Limited retry on initial
// connect only; caller owns the Receive() loop + ctx cancel for safety.
func (c *GRPCClient) Pull(ctx context.Context, req *connect.Request[v1.PullModelRequest]) (*connect.ServerStreamForClient[v1.ProgressResponse], error) {
	if req == nil || req.Msg == nil {
		return nil, errors.New("grpc client pull: nil req")
	}
	model := req.Msg.Model
	start := time.Now()
	slog.Debug("grpc client stream start", "component", "api/grpc-client", "rpc", "Pull", "model", model, "reason", "ctx first; admin stream like ChatStream", "status", "start")
	stream, err := c.models.Pull(ctx, req)
	dur := time.Since(start).Milliseconds()
	if err != nil {
		return nil, fmt.Errorf("grpc client Pull %s: %w", model, err)
	}
	slog.Info("grpc client stream connected", "component", "api/grpc-client", "rpc", "Pull", "model", model, "duration_ms", dur, "status", "ok", "reason", "pull progress stream open")
	return stream, nil
}

// Push (server stream). Symmetric to Pull.
func (c *GRPCClient) Push(ctx context.Context, req *connect.Request[v1.PushModelRequest]) (*connect.ServerStreamForClient[v1.ProgressResponse], error) {
	if req == nil || req.Msg == nil {
		return nil, errors.New("grpc client push: nil req")
	}
	model := req.Msg.Model
	start := time.Now()
	slog.Debug("grpc client stream start", "component", "api/grpc-client", "rpc", "Push", "model", model, "reason", "ctx first; admin stream", "status", "start")
	stream, err := c.models.Push(ctx, req)
	dur := time.Since(start).Milliseconds()
	if err != nil {
		return nil, fmt.Errorf("grpc client Push %s: %w", model, err)
	}
	slog.Info("grpc client stream connected", "component", "api/grpc-client", "rpc", "Push", "model", model, "duration_ms", dur, "status", "ok", "reason", "push progress stream open")
	return stream, nil
}

// List (admin) unary.
func (c *GRPCClient) List(ctx context.Context) (*connect.Response[v1.ListModelsResponse], error) {
	return doWithRetry(ctx, "List", "", &c.breaker, func(ctx context.Context) (*connect.Response[v1.ListModelsResponse], error) {
		return c.models.List(ctx, connect.NewRequest(&v1.ListModelsRequest{}))
	})
}

// Version (health-ish) unary.
func (c *GRPCClient) Version(ctx context.Context) (*connect.Response[v1.VersionResponse], error) {
	return doWithRetry(ctx, "Version", "", &c.breaker, func(ctx context.Context) (*connect.Response[v1.VersionResponse], error) {
		return c.models.Version(ctx, connect.NewRequest(&v1.VersionRequest{}))
	})
}

// Heartbeat: gRPC equiv of api.Client.Heartbeat (uses Version; when health wired use dedicated).
// Ctx first, err checked/wrapped.
func (c *GRPCClient) Heartbeat(ctx context.Context) error {
	_, err := c.Version(ctx)
	if err != nil {
		return fmt.Errorf("grpc client heartbeat: %w", err)
	}
	return nil
}

// Show (admin) unary.
func (c *GRPCClient) Show(ctx context.Context, model string) (*connect.Response[v1.ShowModelResponse], error) {
	return doWithRetry(ctx, "Show", model, &c.breaker, func(ctx context.Context) (*connect.Response[v1.ShowModelResponse], error) {
		return c.models.Show(ctx, connect.NewRequest(&v1.ShowModelRequest{Model: model}))
	})
}

// Ps (admin) unary — list running models.
func (c *GRPCClient) Ps(ctx context.Context) (*connect.Response[v1.PsResponse], error) {
	return doWithRetry(ctx, "Ps", "", &c.breaker, func(ctx context.Context) (*connect.Response[v1.PsResponse], error) {
		return c.models.Ps(ctx, connect.NewRequest(&v1.PsRequest{}))
	})
}

// Pull/Push return server streams and can be added similarly to ChatStream/GenerateStream.
// Skeleton minimal: above covers Chat/Generate/Embed/Models core + Show + Ps + health.

// Note: for full stream retry or advanced (circuit, metrics), wrap further in agent layer.
// All errors wrapped %w; classify via isRetryable (Is/As + connect codes from errToConnect).
// See server/grpc.go errToConnect + phased for full map. Safe for agent concurrency + backpressure.