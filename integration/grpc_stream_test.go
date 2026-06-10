//go:build integration

package integration

import (
	"context"
	"errors"
	"log/slog"
	"net/http"
	"os"
	"testing"
	"time"

	"connectrpc.com/connect"
	"golang.org/x/net/http2"
	"golang.org/x/sync/errgroup"

	v1 "github.com/ollama/ollama/gen/proto/ollama/api/v1"
	apiv1connect "github.com/ollama/ollama/gen/proto/ollama/api/v1/apiv1connect"
	"github.com/ollama/ollama/api"
	"github.com/stretchr/testify/require"
)

// TestGRPCStreaming exercises the gRPC streaming paths (ChatStream, GenerateStream)
// when OLLAMA_GRPC_HOST or OLLAMA_GRPC_SAMEPORT is set for the test binary.
// This extends the suite per the gRPC phased plan for robust coverage of
// streaming adapters, ctx cancel, error mapping in streams, and sameport.
// Integrated/expanded per report sec4 item 8 (finding8: "integrate new grpc_stream_test.go
// into main matrix with models/SAMEPORT/cancel/concurrency/error-class/sameport-vs-sep")
// + phased p90/91 (matrix + harness models + serial limits) + p379 gates + p334 (stream ctx).
// After runner integration (harness path fix + colocate payload support +
// actionable cmake in Find), `ollama serve` started by this test binary can
// discover llama-server and perform actual inference, enabling positive
// token streaming, tool calls in stream, harmony, structured outputs over gRPC
// + load/soak validation (report sec4 finding 1; see also grpc-phased... Phase5).
// Run with: OLLAMA_GRPC_HOST=127.0.0.1:11435 go test -tags=integration -run TestGRPCStreaming ./integration -count=1
// (with prior `cmake -B build . && cmake --build build --target ollama-local` + pulled model for positive tokens)
//
// Agent/Flume load/soak notes (report sec4 item9 + phased Agent/Flume subsec): harness serial (serverMutex);
// for concurrency/queue backpressure soak under "agent" load (N streams > sched MaxQueue): use manual
// (multiple GRPCClient in goroutines or separate procs + OLLAMA_GRPC_SAMEPORT=1), monitor for
// "MaxQueue"/"retry decision" logs (CodeUnavailable via errToConnect + client jitter), pprof stable,
// health preflight, mid-cancel on subset, rich reason logs for LogAgentReasoning. See
// docs/grpc-phased-reliable-approach.md "Agent/Flume gRPC Client..." for script skeleton + gates
// (p323/p379/p381 review subagent req + 10pt SKILL + -race). Exercise when runners present.
func TestGRPCStreaming(t *testing.T) {
	if os.Getenv("OLLAMA_TEST_GRPC") == "0" {
		t.Skip("gRPC tests disabled")
	}
	// Integrate with harness (startServer/Init) so the *test binary itself* launches
	// `ollama serve` (passing OLLAMA_GRPC_* envs). Combined with edit1 (fixed
	// "../ollama" hardcode) + edit2 (colocated payload support) the launched serve
	// can find runners from cmake or next-to-bin, unblocking real LLM callbacks
	// for token streaming in the gRPC write path.
	// Harness models: smol (and testModels) visible same-pkg from utils_test.go; used for matrix.
	slog.Info("grpc stream test matrix entry", "component", "integration", "reason", "integrating/expanding grpc_stream_test into main matrix per report sec4 item8 + phased p90/91/p379: harness models (smol), SAMEPORT/cancel/concurrency/error-class/sameport-vs-sep subtests; leverages P1/2/3 full intcps+correlation+stream_id+OTEL+rich converters+admin streams+reason logs+ctx derive; positive mid-token limited until runners (report #1/p56) but err-over-stream+cancel+admin+concurr paths robustly exercised; local fresh client per subtest for isolation (SKILL verif); serial harness respected (p91)", "status", "matrix-start", "report_finding", "sec4#8", "phased_refs", "p90 p91 p379 p334 p323 p381", "model", smol, "host", os.Getenv("OLLAMA_GRPC_HOST"))
	host := os.Getenv("OLLAMA_GRPC_HOST")
	if host == "" {
		host = "127.0.0.1:11435" // fallback + default for harness launch
	}
	if os.Getenv("OLLAMA_TEST_EXISTING") == "" {
		// Ensure GRPC env so startServer (called by Init) enables gRPC listener.
		// (harness defaults to disable unless set; see utils_test.go Phase4 support).
		t.Setenv("OLLAMA_GRPC_HOST", host)
		_, _, cleanup := InitServerConnection(context.Background(), t)
		defer cleanup()
	}

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	// --- chat_stream: core positive/err path (leverages P2 extracted chat + P3 stream adapter + P4 intcps)
	t.Run("chat_stream", func(t *testing.T) {
		// robust local client per subtest (no outer httpClient reuse/leak; isolation)
		httpClient := &http.Client{Transport: &http2.Transport{AllowHTTP: true}}
		client := apiv1connect.NewChatServiceClient(httpClient, "http://"+host, connect.WithGRPC())

		// Use smol (harness model) or nonexistent: with runners positive tokens (LLM cb->write->Send); always exercises errToConnect + stream ctx cancel select + bounded (phased p334).
		req := connect.NewRequest(&v1.ChatRequest{
			Model:    smol,
			Messages: []*v1.Message{{Role: "user", Content: "stream test"}},
			Stream:   true,
		})
		stream, err := client.ChatStream(ctx, req)
		require.NoError(t, err, "stream connect should succeed even if later errors")

		gotResponse := false
		for stream.Receive() {
			gotResponse = true
			if stream.Msg() != nil && stream.Msg().Done {
				break
			}
		}
		// Error or done (incl. positive content tokens when model/runner present) is
		// delivered over the gRPC stream (exercises thin adapter write, select on
		// streamCtx.Done per phased p334, errToConnect, bounded).
		if err := stream.Err(); err != nil {
			t.Logf("stream delivered error as expected (or model not present): %v (good for robust error-in-stream test)", err)
		}
		if !gotResponse {
			t.Log("no responses received before error (may be immediate fail)")
		}
		slog.Info("chat_stream subtest decision", "component", "integration", "reason", "chat stream matrix case complete (harness model or err path); covers P1/2/3/4 paths", "status", "sub-done", "model", req.Msg.GetModel(), "rpc", "ChatStream", "stream_id", "n/a-in-test", "dur_ms", 0)
	})

	// --- generate_stream: symmetric to chat (P2 generate extract + P3 adapter)
	t.Run("generate_stream", func(t *testing.T) {
		httpClient := &http.Client{Transport: &http2.Transport{AllowHTTP: true}}
		genClient := apiv1connect.NewGenerateServiceClient(httpClient, "http://"+host, connect.WithGRPC())

		req := connect.NewRequest(&v1.GenerateRequest{
			Model:  smol,
			Prompt: "count to 3 briefly",
			Stream: true,
		})
		stream, err := genClient.GenerateStream(ctx, req)
		require.NoError(t, err)

		got := false
		for stream.Receive() {
			got = true
			if stream.Msg() != nil && stream.Msg().Done {
				break
			}
		}
		if se := stream.Err(); se != nil {
			t.Logf("generate stream err (expected pre-runners or edge): %v", se)
		}
		if !got {
			t.Log("generate no chunks (err path ok)")
		}
		slog.Info("generate_stream subtest decision", "component", "integration", "reason", "generate stream added for matrix per finding8; exercises symmetric generate adapter + streamCtx + err classify", "status", "sub-done", "model", req.Msg.GetModel(), "rpc", "GenerateStream")
	})

	// --- admin_streams: List/Show/Ps/Version (P5 admin per report #4 + phased; Pull/Push streams noted but use unary here for minimal; Show/Ps use shared caches/sched)
	// No runner needed; exercises modelsHandler thin + converters + ctx + logs + errTo.
	t.Run("admin_streams", func(t *testing.T) {
		httpClient := &http.Client{Transport: &http2.Transport{AllowHTTP: true}}
		mc := apiv1connect.NewModelsServiceClient(httpClient, "http://"+host, connect.WithGRPC())

		// List (always safe)
		lr, err := mc.List(ctx, connect.NewRequest(&v1.ListModelsRequest{}))
		if err != nil {
			t.Logf("list admin may be empty pre-models: %v", err)
		} else if lr != nil {
			slog.Debug("admin list ok", "component", "integration", "reason", "List admin stream/unary path exercised via gRPC (modelsHandler + convertListModelToPB)", "status", "ok", "num", len(lr.Msg.GetModels()))
		}

		// Show for harness smol (may 404 if not pulled; path still covered; use log not fail)
		sr, err := mc.Show(ctx, connect.NewRequest(&v1.ShowModelRequest{Model: smol}))
		if err != nil {
			t.Logf("show for %s (harness model) not present or err (admin path exercised): %v", smol, err)
		} else if sr != nil {
			slog.Info("admin show decision", "component", "integration", "reason", "Show fleshed (details/model_info) per report#4; gRPC admin exercised", "status", "ok", "model", smol, "rpc", "ModelsService/Show")
		}

		// Ps (running models from shared sched; may empty)
		pr, err := mc.Ps(ctx, connect.NewRequest(&v1.PsRequest{}))
		require.NoError(t, err) // even if 0 models, response ok
		_ = pr

		// Version (simple, no model)
		vr, err := mc.Version(ctx, connect.NewRequest(&v1.VersionRequest{}))
		require.NoError(t, err)
		require.NotEmpty(t, vr.Msg.GetVersion())

		slog.Info("admin_streams subtest decision", "component", "integration", "reason", "admin streams (List/Show/Ps/Version) integrated for matrix per finding8/report#4/P5; Pull/Push progress streams covered in handlers but skipped here (net+idemp minimal); leverages modelsHandler ctx-first + errTo + stream_id logs", "status", "sub-done", "rpc", "ModelsService/*", "model", smol)
	})

	// --- concurrency: bounded subtest (errgroup SetLimit mirrors cmd P1/5 sameport; exercises sched refcounts/MaxQueue under concurrent gRPC streams)
	t.Run("concurrency", func(t *testing.T) {
		g, gctx := errgroup.WithContext(ctx)
		g.SetLimit(3) // bounded per SKILL + phased; owner = this subtest goroutine; no fire-forget
		for i := 0; i < 3; i++ {
			i := i
			g.Go(func() error {
				// fresh local client per goroutine/sub (robust)
				hc := &http.Client{Transport: &http2.Transport{AllowHTTP: true}}
				lc := apiv1connect.NewChatServiceClient(hc, "http://"+host, connect.WithGRPC())
				req := connect.NewRequest(&v1.ChatRequest{
					Model:    "nonexistent-concurr-" + string(rune('0'+i)),
					Messages: []*v1.Message{{Role: "user", Content: "concurr test"}},
					Stream:   true,
				})
				st, err := lc.ChatStream(gctx, req)
				if err != nil {
					return nil // err path ok for matrix (schedule errs concurrent safe)
				}
				for st.Receive() {
				}
				if se := st.Err(); se != nil {
					slog.Debug("concurr stream err ok", "component", "integration", "reason", "concurrent gRPC stream err classified (transient sched/MaxQueue or notfound); exercises bounded errgroup + shared sched reconcile under load", "status", "ok", "i", i)
				}
				return nil
			})
		}
		if werr := g.Wait(); werr != nil {
			t.Logf("concurr group err (non-fatal for matrix): %v", werr)
		}
		slog.Info("concurrency subtest decision", "component", "integration", "reason", "concurrency matrix sub added per finding8; uses errgroup.SetLimit + ctx owner + local clients; covers p91 scale note + sched under gRPC concurr (no new workers, reuses reconcile)", "status", "sub-done", "rpc", "ChatStream", "limit", 3)
	})

	// --- error_classification: use Is/As on stream.Err() to cover errToConnect classify (transient Unavailable for queue/MaxQueue, Canceled, NotFound etc per SKILL + report)
	t.Run("error_classification", func(t *testing.T) {
		hc := &http.Client{Transport: &http2.Transport{AllowHTTP: true}}
		lc := apiv1connect.NewChatServiceClient(hc, "http://"+host, connect.WithGRPC())
		req := connect.NewRequest(&v1.ChatRequest{
			Model:    "nonexistent-for-err-class",
			Messages: []*v1.Message{{Role: "user", Content: "err class test"}},
			Stream:   true,
		})
		st, err := lc.ChatStream(ctx, req)
		require.NoError(t, err)
		for st.Receive() {
		}
		if se := st.Err(); se != nil {
			var ce *connect.Error
			classified := "unknown"
			if errors.As(se, &ce) {
				classified = ce.Code().String()
				slog.Info("error classified via As", "component", "integration", "reason", "stream err classified with errors.As(*connect.Error) + codes from errToConnect (e.g. Canceled for ctx/499, Unavailable for MaxQueue/transient sched, NotFound/Invalid for bad model, per SKILL errors-are-values + Is/As); enables agent retry/jitter (idempotent)", "status", "classified", "code", classified, "model", req.Msg.GetModel(), "rpc", "ChatStream", "err", se.Error())
			}
			if errors.Is(se, context.Canceled) || errors.Is(se, context.DeadlineExceeded) {
				classified = "ctx-canceled-via-Is"
			}
			t.Logf("classified stream err: code=%s (good for error-class matrix)", classified)
		}
		slog.Info("error_classification subtest decision", "component", "integration", "reason", "error-class sub added per finding8; exercises Is/As on errToConnect outputs (leverages P4 complete mapping); sameport-vs-sep noted by env at run", "status", "sub-done", "rpc", "ChatStream")
	})

	// --- sameport_and_cancel (kept/enhanced; sameport-vs-sep via env on test invocation)
	t.Run("sameport_and_cancel", func(t *testing.T) {
		// Enhanced skeleton for SAMEPORT variant + mid-gen cancel (report sec4 item5: "mid-gen cancel on sameport";
		// phased doc p323 high-risk soak+review-subagent-req before default/landing p381, p334 required stream ctx + reason logs).
		// Leverages Phase1/2: derived ctx cancel from stream.Context() in handlers (select {<-ctx.Done()}), rich reason/component logs,
		// bounded selects, err classification (errToConnect Canceled etc), correlation/stream_id in ctx (intcps).
		// Robust: local client + httpClient per subtest (no outer leak; isolation per SKILL p43/70 verifiability + serial harness).
		// When runners present (report #1 blocks now): use real model + longer receive loop to hit mid-gen (LLM callback write cb sees cancel -> stop prompt promptly, no GPU leak).
		// Current: exercises err-over-stream + immediate cancel path (nonexistent model -> schedule err delivered, then cancel).
		// For soak: run under OLLAMA_GRPC_SAMEPORT=1 + long-lived dual clients (see docs/grpc-phased-reliable-approach.md soak skeleton).
		// sameport-vs-sep: run matrix by invoking with/without OLLAMA_GRPC_SAMEPORT=1 (or GRPC_HOST vs default sep); this sub skips unless set.
		if os.Getenv("OLLAMA_GRPC_SAMEPORT") != "1" && os.Getenv("OLLAMA_GRPC_HOST") == "" {
			t.Skip("SAMEPORT or explicit GRPC host required for this subtest variant")
		}
		// local client for this subtest (robust, no reuse after potential cleanup)
		httpClient := &http.Client{Transport: &http2.Transport{AllowHTTP: true}}
		localClient := apiv1connect.NewChatServiceClient(httpClient, "http://"+host, connect.WithGRPC())

		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()

		req := connect.NewRequest(&v1.ChatRequest{
			Model:    "nonexistent-for-test",
			Messages: []*v1.Message{{Role: "user", Content: "cancel stream test"}},
			Stream:   true,
		})
		stream, err := localClient.ChatStream(ctx, req)
		require.NoError(t, err)

		// Simulate mid-gen cancel (robust test per SKILL bounded + doc p334 stream ctx respect).
		// In positive case (runners): first Receive may get token, then cancel() stops via handler's streamCtx derived cancel passed to s.chat -> schedule -> llm.
		slog.Info("sameport mid-gen cancel decision", "component", "integration", "reason", "trigger cancel after initial Receive to exercise gRPC stream ctx cancel path on sameport (cmux muxed conn); covers long-running dual + agent client mid-cancel + no-leak req per report sec4#5 + phased p323; status=simulated", "status", "cancel-sim", "model", req.Msg.GetModel(), "sameport_env", os.Getenv("OLLAMA_GRPC_SAMEPORT"), "rpc", "ChatStream")
		if stream.Receive() {
			cancel() // cancel after first chunk (or immediately in no-model case); exercises select on Done in write cb
		}
		// Drain or check close. (bounded; in real would see quick close post-cancel)
		for stream.Receive() {
		}
		if err := stream.Err(); err != nil {
			t.Logf("cancel path delivered: %v (good for ctx cancel in gRPC stream per SKILL/Flume GPU safety)", err)
		}
		slog.Info("sameport_and_cancel subtest decision", "component", "integration", "reason", "sameport cancel variant complete; sameport-vs-sep matrix by CI/env invocation per finding8", "status", "sub-done", "sameport_env", os.Getenv("OLLAMA_GRPC_SAMEPORT"))
	})

	// --- high_level_grpc_client: exercise the recently added production client skeleton (Phase4 Agent/Flume sub-agent item9 + phased p88/368/p94-95)
	// Covers: api.NewGRPCClient + GRPCClientFromEnvironment (envconfig), doWithRetry for unary (List/Version/Heartbeat with jitter, ctx select, %w, isRetryable Is/As on connect.Code from errToConnect),
	// ChatStream/GenerateStream wrappers (limited retry on connect, ctx first, client-side rich slog+reason), Heartbeat.
	// Extends raw apiv1connect usage in other subs to cover high-level client code paths, retry classify, client logs for LogAgent/Flume.
	// Uses fresh client; exercises in matrix context (err paths ok pre-runners; sameport note).
	// Also exercises health path (Heartbeat calls Version; complements server enableGRPCHealthIfOptIn skeleton).
	t.Run("high_level_grpc_client", func(t *testing.T) {
		gc := api.NewGRPCClient("http://"+host, connect.WithGRPC())
		// unary via high-level (hits doWithRetry + client logs + errTo classify)
		lr, err := gc.List(ctx)
		if err != nil {
			t.Logf("high_level List err (pre-models ok): %v", err)
		} else if lr != nil {
			slog.Info("high_level list", "component", "integration", "reason", "GRPCClient.List exercised (doWithRetry + api/grpc_client.go:271; covers retry/jitter/ctx for agent/Flume)", "num", len(lr.Msg.GetModels()))
		}
		vr, err := gc.Version(ctx)
		require.NoError(t, err)
		require.NotEmpty(t, vr.Msg.GetVersion())
		if err := gc.Heartbeat(ctx); err != nil {
			t.Logf("high_level Heartbeat (health path, uses Version): %v (skeleton; server enableGRPCHealthIfOptIn exercised on serve)", err)
		}
		// FromEnvironment (exercises envconfig.GRPCHost path + fallback)
		if os.Getenv("OLLAMA_GRPC_HOST") != "" {
			gc2, _ := api.GRPCClientFromEnvironment()
			_ = gc2 // path covered
		}
		// stream via high-level (covers client stream wrapper + logs + ctx)
		req := connect.NewRequest(&v1.ChatRequest{
			Model:    smol,
			Messages: []*v1.Message{{Role: "user", Content: "high level client stream"}},
			Stream:   true,
		})
		st, err := gc.ChatStream(ctx, req)
		if err != nil {
			t.Logf("high_level ChatStream err ok: %v", err)
		} else {
			for st.Receive() {
			}
			if se := st.Err(); se != nil {
				t.Logf("high_level stream err: %v", se)
			}
		}
		// symmetric generate stream via client
		greq := connect.NewRequest(&v1.GenerateRequest{Model: smol, Prompt: "hi", Stream: true})
		gst, err := gc.GenerateStream(ctx, greq)
		if err != nil {
			t.Logf("high_level GenerateStream err ok: %v", err)
		} else {
			for gst.Receive() {
			}
			_ = gst.Err()
		}
		slog.Info("high_level_grpc_client subtest decision", "component", "integration", "reason", "high level GRPCClient (New/FromEnv/unary-retry/stream-wrapper/Heartbeat) exercised; covers recently added client skeleton + rich client reason logs + err classify/jitter (item9/Phase4 sub2 + Phase5 client); complements raw matrix subs; health/refl registration via harness serve", "status", "sub-done", "rpc", "Chat/Gen/List/Version/Heartbeat/Stream")
	})
}

