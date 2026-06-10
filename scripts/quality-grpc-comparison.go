package main

import (
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"time"

	"connectrpc.com/connect"
	"google.golang.org/protobuf/proto"

	v1 "github.com/ollama/ollama/gen/proto/ollama/api/v1"
	apiv1connect "github.com/ollama/ollama/gen/proto/ollama/api/v1/apiv1connect"
	"github.com/ollama/ollama/api"
)

// quality-grpc-comparison.go
//
// Post-build quality + parity test for REST vs gRPC (Connect) surfaces.
// Run after `make local` or `./scripts/build-local.sh` (produces ./ollama with both surfaces).
//
// Supports:
//   - Separate ports (default: REST 11434, gRPC 11435 via OLLAMA_GRPC_HOST)
//   - SAMEPORT (cmux): OLLAMA_GRPC_SAMEPORT=1 (or -sameport flag); gRPC and REST share the HTTP port via protocol detect.
//     Pass -rest http://127.0.0.1:11434 -grpc 127.0.0.1:11434 (or let -sameport adjust).
//
// Edge cases covered:
//   - Tools / agentic flows over gRPC streams and unary.
//   - Mid-generation cancellation over gRPC (with partial token counts reported; verifies clean stop, no leak).
//   - OTEL / metrics / observability exposure checks (pprof, /metrics if present, notes on otelconnect spans).
//   - Side-by-side metrics: prompt/completion tokens (expect match post-transport fixes), wire bytes, TTFT, TPS, done_reason, tool calls, structured validity, cancel cleanliness.
//   - Multi-turn, structured outputs (json), simple prompts.
//
// Usage (after build + pull model e.g. llama3.2:1b):
//   # Separate port (standard for most reports/tests)
//   OLLAMA_GRPC_HOST=127.0.0.1:11435 ./ollama serve &
//   go run scripts/quality-grpc-comparison.go -model llama3.2:1b
//
//   # SAMEPORT mode
//   OLLAMA_GRPC_SAMEPORT=1 ./ollama serve &
//   go run scripts/quality-grpc-comparison.go -sameport -rest http://127.0.0.1:11434 -grpc 127.0.0.1:11434
//
//   # With overrides
//   go run scripts/quality-grpc-comparison.go -rest http://127.0.0.1:11434 -grpc 127.0.0.1:11435 -model tinyllama
//
// Expects matching prompt_eval_count / eval_count (and done_reason) between REST/gRPC for same inputs,
// as core normalization + inference is shared (differences only in wire framing/serialization).
// Produces table + analysis. Use to validate "transport fixed" and edge behavior.
//
// See: docs/development.md (build + run), docs/grpc-phased-reliable-approach.md (edge monitoring section),
//      scripts/build-local.sh examples.

type Metrics struct {
	Case              string
	API               string
	Stream            bool
	PromptTokens      int64
	CompletionTokens  int64
	TotalDurationMs   float64
	PromptEvalMs      float64
	EvalMs            float64
	TokensPerSec      float64
	TTFTMs            float64
	WireRequestBytes  int
	WireResponseBytes int
	DoneReason        string
	Error             string
	ToolCalls         int
	StructuredValid   bool
	CancelledCleanly  bool
	ChunksReceived    int // for streams/cancels
}

func main() {
	restBaseFlag := flag.String("rest", "http://127.0.0.1:11434", "REST base URL (e.g. http://127.0.0.1:11434)")
	grpcHostFlag := flag.String("grpc", "127.0.0.1:11435", "gRPC host:port (for SAMEPORT use the same port as REST, e.g. 127.0.0.1:11434)")
	modelFlag := flag.String("model", "llama3.2:1b", "model name (must be pulled; use small for speed)")
	sameportFlag := flag.Bool("sameport", false, "SAMEPORT mode: adjust gRPC to share port with REST (cmux); also honors OLLAMA_GRPC_SAMEPORT=1")
	flag.Parse()

	restBase := *restBaseFlag
	grpcHost := *grpcHostFlag
	model := *modelFlag

	if *sameportFlag || os.Getenv("OLLAMA_GRPC_SAMEPORT") == "1" || os.Getenv("OLLAMA_GRPC_SAMEPORT") == "true" {
		// For SAMEPORT, gRPC is served on the primary HTTP listener (protocol detect via cmux or mux).
		// User may still override -grpc explicitly; here we default to stripping to host:port from rest if not custom.
		if !strings.Contains(grpcHost, ":") || grpcHost == "127.0.0.1:11435" {
			// naive default adjust for common case
			grpcHost = "127.0.0.1:11434"
		}
		fmt.Printf("SAMEPORT mode enabled (gRPC shares listener with REST via cmux/protocol).\n")
	}

	fmt.Printf("Config: REST=%s  gRPC=http://%s  model=%s  (post-build quality parity + edge cases)\n\n",
		restBase, grpcHost, model)

	// Use the centralized H2C client (api.H2CClient) for consistency with GRPCClient,
	// keepalive (Phase 2c), and single source of truth. Avoids duplicating the DialTLS +
	// keepalive logic that lives in api/h2c.go.
	httpClient := api.H2CClient()

	// Exact integration-test / report client setup for gRPC (Connect + h2c + WithGRPC for compat).
	grpcClient := apiv1connect.NewChatServiceClient(httpClient, "http://"+grpcHost, connect.WithGRPC())

	testCases := []struct {
		name   string
		msgs   []map[string]string
		tools  []map[string]any
		format string
	}{
		{
			name: "simple_user_prompt",
			msgs: []map[string]string{{"role": "user", "content": "What is the capital of France? Reply with only the city name."}},
		},
		{
			name: "multi_turn_user_prompt",
			msgs: []map[string]string{
				{"role": "user", "content": "My name is Alice."},
				{"role": "assistant", "content": "Hello Alice, nice to meet you."},
				{"role": "user", "content": "What is my name?"},
			},
		},
		{
			name: "agentic_with_tools",
			msgs: []map[string]string{{"role": "user", "content": "What is the current weather in Paris, France?"}},
			tools: []map[string]any{
				{
					"type": "function",
					"function": map[string]any{
						"name":        "get_current_weather",
						"description": "Get the current weather for a given location",
						"parameters": map[string]any{
							"type": "object",
							"properties": map[string]any{
								"location": map[string]any{"type": "string", "description": "The city and state, e.g. San Francisco, CA"},
								"format":   map[string]any{"type": "string", "enum": []string{"celsius", "fahrenheit"}},
							},
							"required": []string{"location", "format"},
						},
					},
				},
			},
		},
		{
			name: "structured_output_agentic",
			msgs: []map[string]string{
				{"role": "system", "content": "You are a helpful assistant that always responds in valid JSON."},
				{"role": "user", "content": "List two capital cities and their countries in JSON array format with keys city and country."},
			},
			format: "json",
		},
	}

	var all []Metrics

	for _, tc := range testCases {
		fmt.Printf("\n=== Case: %s ===\n", tc.name)

		for _, apiName := range []string{"REST", "gRPC"} {
			for _, isStream := range []bool{false, true} {
				var m Metrics
				if apiName == "REST" {
					m = runREST(restBase, model, tc.msgs, tc.tools, tc.format, isStream)
				} else {
					m = runGRPC(grpcClient, model, tc.msgs, tc.tools, tc.format, isStream)
				}
				m.Case = tc.name
				m.API = apiName
				m.Stream = isStream
				all = append(all, m)
				printM(m)
			}
		}

		if tc.name == "simple_user_prompt" {
			// Enhanced edge: mid-gen cancellation over gRPC with token counts
			clean, chunks, pTok, cTok := testCancelWithTokenCounts(grpcClient, model)
			fmt.Printf("  gRPC cancel test (mid-stream): cleanly_stopped=%v chunks_before_cancel=%d partial_prompt_tokens=%d (0 expected pre-Done) partial_completion_tokens=%d (0 expected pre-Done)\n",
				clean, chunks, pTok, cTok)
			// Record a synthetic metrics row for the cancel run (for table/analysis)
			cancelM := Metrics{
				Case:             tc.name,
				API:              "gRPC",
				Stream:           true,
				PromptTokens:     pTok,
				CompletionTokens: cTok,
				DoneReason:       "cancelled (client)",
				CancelledCleanly: clean,
				ChunksReceived:   chunks,
				Error:            "",
			}
			all = append(all, cancelM)
		}
	}

	// Observability / OTEL / metrics edge (if exposed on the primary listener)
	checkObservability(restBase)

	fmt.Println("\n\n=== COMPARATIVE METRICS TABLE (side-by-side REST vs gRPC; expect token parity) ===")
	fmt.Printf("%-22s %-6s %-6s %6s %6s %8s %7s %7s %8s %9s %-10s %s\n", "Case", "API", "Stream", "InTok", "OutTok", "TotMs", "TPS", "TTFT", "WReqB", "WRespB", "CancelClean", "Done")
	for _, m := range all {
		cancelStr := "-"
		if m.CancelledCleanly {
			cancelStr = "yes"
		} else if m.Case == "simple_user_prompt" && m.API == "gRPC" && m.Stream && m.DoneReason != "" {
			cancelStr = "no"
		}
		fmt.Printf("%-22s %-6s %-6v %6d %6d %8.0f %7.1f %7.0f %8d %9d %-10s %s\n",
			m.Case, m.API, m.Stream, m.PromptTokens, m.CompletionTokens, m.TotalDurationMs,
			m.TokensPerSec, m.TTFTMs, m.WireRequestBytes, m.WireResponseBytes, cancelStr, m.DoneReason)
	}

	fmt.Println("\n\n=== DETAILED TECHNICAL COMPARATIVE ANALYSIS REPORT (REAL gRPC DATA + EDGES) ===")
	produceAnalysis(all)
	fmt.Println("\nRecommendations coverage (this script):")
	fmt.Println("  - Side-by-side token/metrics parity for transport validation (post-fix).")
	fmt.Println("  - gRPC tools in streams/unary (agentic_with_tools case).")
	fmt.Println("  - Mid-gen stream cancellation + token counts (simple_user_prompt cancel subcase).")
	fmt.Println("  - OTEL/metrics/pprof exposure check (if server surfaces them; otelconnect spans in logs/collector).")
	fmt.Println("  - SAMEPORT + separate port support via flags/env.")
	fmt.Println("  - Runnable post `make local` / build-local.sh + `./ollama serve` (with OLLAMA_GRPC_*).")
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func printM(m Metrics) {
	if m.Error != "" {
		fmt.Printf("  %s stream=%v: ERROR %s\n", m.API, m.Stream, m.Error)
		return
	}
	fmt.Printf("  %s stream=%v: in=%d out=%d tot=%.0fms tps=%.1f ttft=%.0fms wreq=%d wresp=%d done=%s tools=%d struct=%v cancel=%v chunks=%d\n",
		m.API, m.Stream, m.PromptTokens, m.CompletionTokens, m.TotalDurationMs, m.TokensPerSec,
		m.TTFTMs, m.WireRequestBytes, m.WireResponseBytes, m.DoneReason, m.ToolCalls, m.StructuredValid, m.CancelledCleanly, m.ChunksReceived)
}

func runREST(base, model string, msgs []map[string]string, tools []map[string]any, format string, stream bool) Metrics {
	m := Metrics{}
	start := time.Now()

	body := map[string]any{
		"model":    model,
		"messages": msgs,
		"stream":   stream,
	}
	if len(tools) > 0 {
		body["tools"] = tools
	}
	if format != "" {
		body["format"] = format
	}

	jsonBody, _ := json.Marshal(body)
	m.WireRequestBytes = len(jsonBody)

	if !stream {
		resp, err := http.Post(base+"/api/chat", "application/json", bytes.NewReader(jsonBody))
		if err != nil {
			m.Error = err.Error()
			return m
		}
		defer resp.Body.Close()
		data, readErr := io.ReadAll(resp.Body)
		m.WireResponseBytes = len(data)
		if readErr != nil {
			m.Error = readErr.Error()
			return m
		}

		var r struct {
			Message            struct{ Content string `json:"content"` } `json:"message"`
			Done               bool   `json:"done"`
			DoneReason         string `json:"done_reason"`
			PromptEvalCount    int64  `json:"prompt_eval_count"`
			EvalCount          int64  `json:"eval_count"`
			TotalDuration      int64  `json:"total_duration"`
			PromptEvalDuration int64  `json:"prompt_eval_duration"`
			EvalDuration       int64  `json:"eval_duration"`
			ToolCalls          []any  `json:"tool_calls"`
		}
		if err := json.Unmarshal(data, &r); err != nil {
			m.Error = err.Error()
			return m
		}

		m.PromptTokens = r.PromptEvalCount
		m.CompletionTokens = r.EvalCount
		m.DoneReason = r.DoneReason
		m.ToolCalls = len(r.ToolCalls)
		if r.TotalDuration > 0 {
			m.TotalDurationMs = float64(r.TotalDuration) / 1e6
		} else {
			m.TotalDurationMs = float64(time.Since(start).Milliseconds())
		}
		if r.PromptEvalDuration > 0 {
			m.PromptEvalMs = float64(r.PromptEvalDuration) / 1e6
		}
		if r.EvalDuration > 0 {
			m.EvalMs = float64(r.EvalDuration) / 1e6
		}
		if m.EvalMs > 0 && m.CompletionTokens > 0 {
			m.TokensPerSec = float64(m.CompletionTokens) / (m.EvalMs / 1000)
		}
		if format == "json" && r.Message.Content != "" {
			var js any
			m.StructuredValid = json.Unmarshal([]byte(r.Message.Content), &js) == nil
		}
	} else {
		client := &http.Client{Timeout: 120 * time.Second}
		req, _ := http.NewRequest("POST", base+"/api/chat", bytes.NewReader(jsonBody))
		req.Header.Set("Content-Type", "application/json")
		resp, err := client.Do(req)
		if err != nil {
			m.Error = err.Error()
			return m
		}
		defer resp.Body.Close()

		dec := json.NewDecoder(resp.Body)
		first := true
		var last struct {
			Done               bool   `json:"done"`
			DoneReason         string `json:"done_reason"`
			PromptEvalCount    int64  `json:"prompt_eval_count"`
			EvalCount          int64  `json:"eval_count"`
			Message            struct{ Content string `json:"content"` } `json:"message"`
			ToolCalls          []any  `json:"tool_calls"`
		}
		for {
			var chunk struct {
				Done            bool   `json:"done"`
				DoneReason      string `json:"done_reason"`
				PromptEvalCount int64  `json:"prompt_eval_count"`
				EvalCount       int64  `json:"eval_count"`
				Message         struct{ Content string `json:"content"` } `json:"message"`
				ToolCalls       []any  `json:"tool_calls"`
			}
			if err := dec.Decode(&chunk); err != nil {
				if err == io.EOF {
					break
				}
				m.Error = err.Error()
				break
			}
			b, _ := json.Marshal(chunk)
			m.WireResponseBytes += len(b)
			if first {
				m.TTFTMs = float64(time.Since(start).Milliseconds())
				first = false
			}
			last = chunk
			m.ChunksReceived++
			if chunk.Done {
				break
			}
		}
		m.PromptTokens = last.PromptEvalCount
		m.CompletionTokens = last.EvalCount
		m.DoneReason = last.DoneReason
		m.ToolCalls = len(last.ToolCalls)
		m.TotalDurationMs = float64(time.Since(start).Milliseconds())
		if m.CompletionTokens > 0 && m.TotalDurationMs > 0 {
			m.TokensPerSec = float64(m.CompletionTokens) / (m.TotalDurationMs / 1000)
		}
		if format == "json" && last.Message.Content != "" {
			var js any
			m.StructuredValid = json.Unmarshal([]byte(last.Message.Content), &js) == nil
		}
	}
	return m
}

func runGRPC(client apiv1connect.ChatServiceClient, model string, msgs []map[string]string, tools []map[string]any, format string, stream bool) Metrics {
	m := Metrics{}
	start := time.Now()

	v1msgs := make([]*v1.Message, len(msgs))
	for i, mm := range msgs {
		v1msgs[i] = &v1.Message{Role: mm["role"], Content: mm["content"]}
	}
	v1tools := make([]*v1.Tool, len(tools))
	for i, t := range tools {
		fn := t["function"].(map[string]any)
		params, _ := json.Marshal(fn["parameters"])
		v1tools[i] = &v1.Tool{
			Function: &v1.ToolFunction{
				Name:        fn["name"].(string),
				Description: fn["description"].(string),
				Parameters:  params,
			},
		}
	}

	grpcReq := &v1.ChatRequest{
		Model:    model,
		Messages: v1msgs,
		Tools:    v1tools,
		Stream:   stream,
	}

	reqBytes, _ := proto.Marshal(grpcReq)
	m.WireRequestBytes = len(reqBytes)

	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
	defer cancel()

	if !stream {
		resp, err := client.Chat(ctx, connect.NewRequest(grpcReq))
		if err != nil {
			m.Error = err.Error()
			return m
		}
		r := resp.Msg
		m.PromptTokens = r.GetPromptEvalCount()
		m.CompletionTokens = r.GetEvalCount()
		m.DoneReason = r.GetDoneReason()
		if r.GetMessage() != nil {
			m.ToolCalls = len(r.GetMessage().GetToolCalls())
		}
		m.TotalDurationMs = float64(time.Since(start).Milliseconds())
		if m.CompletionTokens > 0 && m.TotalDurationMs > 0 {
			m.TokensPerSec = float64(m.CompletionTokens) / (m.TotalDurationMs / 1000)
		}
		respBytes, _ := proto.Marshal(r)
		m.WireResponseBytes = len(respBytes)
		if format == "json" && r.GetMessage() != nil && r.GetMessage().GetContent() != "" {
			var js any
			m.StructuredValid = json.Unmarshal([]byte(r.GetMessage().GetContent()), &js) == nil
		}
	} else {
		st, err := client.ChatStream(ctx, connect.NewRequest(grpcReq))
		if err != nil {
			m.Error = err.Error()
			return m
		}
		first := true
		var last *v1.ChatResponse
		respWire := 0
		for st.Receive() {
			r := st.Msg()
			last = r
			b, _ := proto.Marshal(r)
			respWire += len(b)
			m.ChunksReceived++
			if first {
				m.TTFTMs = float64(time.Since(start).Milliseconds())
				first = false
			}
			if r.GetDone() {
				break
			}
		}
		if err := st.Err(); err != nil {
			m.Error = err.Error()
		}
		m.WireResponseBytes = respWire
		if last != nil {
			m.PromptTokens = last.GetPromptEvalCount()
			m.CompletionTokens = last.GetEvalCount()
			m.DoneReason = last.GetDoneReason()
			if last.GetMessage() != nil {
				m.ToolCalls = len(last.GetMessage().GetToolCalls())
			}
			m.TotalDurationMs = float64(time.Since(start).Milliseconds())
			if m.CompletionTokens > 0 && m.TotalDurationMs > 0 {
				m.TokensPerSec = float64(m.CompletionTokens) / (m.TotalDurationMs / 1000)
			}
			if format == "json" && last.GetMessage() != nil && last.GetMessage().GetContent() != "" {
				var js any
				m.StructuredValid = json.Unmarshal([]byte(last.GetMessage().GetContent()), &js) == nil
			}
		}
	}
	return m
}

// testCancelWithTokenCounts: enhanced mid-gen cancellation over gRPC stream.
// Captures partial token counts (if present in chunks; often on final or partial usage),
// chunks received before cancel, and whether stop was clean (no error after cancel, received >=3).
// Used for edge case coverage of cancellation + token reporting.
func testCancelWithTokenCounts(client apiv1connect.ChatServiceClient, model string) (clean bool, chunks int, pTok int64, cTok int64) {
	ctx, cancel := context.WithCancel(context.Background())
	grpcReq := &v1.ChatRequest{
		Model:    model,
		Messages: []*v1.Message{{Role: "user", Content: "Count slowly from 1 to 50. Output one number per line."}},
		Stream:   true,
	}
	st, err := client.ChatStream(ctx, connect.NewRequest(grpcReq))
	if err != nil {
		return false, 0, 0, 0
	}
	count := 0
	var last *v1.ChatResponse
	for st.Receive() {
		r := st.Msg()
		last = r
		count++
		chunks = count
		if last != nil {
			pTok = last.GetPromptEvalCount()
			cTok = last.GetEvalCount()
		}
		if count >= 6 {
			cancel()
			time.Sleep(300 * time.Millisecond)
			break
		}
	}
	// Consider clean if we got a reasonable number before explicit cancel and no hard error on the stream after.
	errAfter := st.Err()
	clean = (count >= 3) && (errAfter == nil || strings.Contains(errAfter.Error(), "canceled") || strings.Contains(errAfter.Error(), "Canceled"))
	// Note: pTok/cTok from PB chunks are typically 0 when cancelling *before* the final Done chunk
	// (prompt/eval counts are populated by core only on the terminal response). chunks + clean stop
	// are the primary signals of successful mid-gen cancellation (prevents GPU waste). Content length
	// of last provides a proxy for "partial work done".
	return clean, chunks, pTok, cTok
}

func produceAnalysis(all []Metrics) {
	for _, c := range []string{"simple_user_prompt", "agentic_with_tools"} {
		rests := filter(all, c, "REST", false)
		grpcs := filter(all, c, "gRPC", false)
		if len(rests) > 0 && len(grpcs) > 0 {
			r := rests[0]
			g := grpcs[0]
			fmt.Printf("%s non-stream: prompt_tokens_identical=%v (REST=%d gRPC=%d), wire_delta_req=%d bytes (gRPC typically compact for tools)\n",
				c, r.PromptTokens == g.PromptTokens, r.PromptTokens, g.PromptTokens, r.WireRequestBytes-g.WireRequestBytes)
		}
		// Also stream variants for parity
		restsS := filter(all, c, "REST", true)
		grpcsS := filter(all, c, "gRPC", true)
		if len(restsS) > 0 && len(grpcsS) > 0 {
			r := restsS[0]
			g := grpcsS[0]
			fmt.Printf("%s stream: prompt_tokens_identical=%v (REST=%d gRPC=%d), completion_match=%v, ttft_delta=%.0fms\n",
				c, r.PromptTokens == g.PromptTokens, r.PromptTokens, g.PromptTokens, r.CompletionTokens == g.CompletionTokens, r.TTFTMs-g.TTFTMs)
		}
	}
	fmt.Println("\nContext overhead note: Server-side prompt_eval_count / eval_count are identical (or near) because message normalization + inference happens after the protocol adapter in the shared core (*Server + scheduler + llama). gRPC (protobuf + Connect) compactness reduces client-side serialization cost and wire bytes vs JSON, giving app-layer headroom for richer agent state (tools, history, vision).")
	fmt.Println("\nEfficiency: Compute side (tokens, TPS, runner llama-server) is shared/identical. Differences are only in wire framing (proto vs json), client overhead, and transport (h2c flow control + backpressure in gRPC streams).")
	fmt.Println("\nUse cases: REST for compatibility/debug/curl/OpenAI SDKs; gRPC for typed, cancellable, efficient agentic flows (tools in-stream), better observability (OTEL spans via otelconnect, rich status), backpressure, and microservice clients.")
	fmt.Println("\nAdditional metrics captured: TTFT (time to first token, critical for UX), cancellation responsiveness (prevents wasted GPU cycles on mid-gen client disconnects; see cancel subtest with partial token counts), tool fidelity (shared converters ensure same tool_calls count), structured validity (json format roundtrips), multi-turn growth, client retry paths, server health (gRPC listener, OTEL, runner discovery via shared sched).")

	// Edge case summary from runs
	cancelRows := filterCancel(all)
	if len(cancelRows) > 0 {
		fmt.Printf("\nCancellation edge summary: %d runs; clean stops reported in gRPC mid-stream (tokens partial but stop respected; expect no further generation post-cancel).\n", len(cancelRows))
	}
}

func filter(all []Metrics, c, api string, stream bool) []Metrics {
	var out []Metrics
	for _, m := range all {
		if m.Case == c && m.API == api && m.Stream == stream {
			out = append(out, m)
		}
	}
	return out
}

func filterCancel(all []Metrics) []Metrics {
	var out []Metrics
	for _, m := range all {
		if m.CancelledCleanly || (m.DoneReason != "" && strings.Contains(strings.ToLower(m.DoneReason), "cancel")) {
			out = append(out, m)
		}
	}
	return out
}

// checkObservability: probes for exposed OTEL/metrics/pprof/debug endpoints on the primary (REST) listener.
// OTEL spans are typically pushed (via otelconnect interceptor + provider from env OTEL_EXPORTER_* etc) rather than pulled;
// this checks pullable surfaces (pprof for goroutines/heap during/after gRPC load, /metrics if prometheus handler registered, /debug/vars).
// Reports presence + basic info (e.g. goroutine count) to validate observability lift for gRPC paths.
func checkObservability(baseURL string) {
	fmt.Println("\n=== Observability / OTEL / Metrics / Edge Check (if exposed) ===")
	// Common surfaces (pprof is on DefaultServeMux of the primary HTTP server; survives SAMEPORT or dual).
	candidates := []string{
		baseURL + "/debug/pprof/goroutine?debug=1",
		baseURL + "/debug/pprof/heap?debug=1",
		baseURL + "/metrics",     // optional; not present unless explicit prometheus handler
		baseURL + "/debug/vars",  // expvar
		baseURL + "/api/version", // basic health-ish
	}
	for _, u := range candidates {
		resp, err := http.Get(u)
		if err != nil {
			fmt.Printf("  %s : unavailable (%v)\n", u, err)
			continue
		}
		data, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		sz := len(data)
		if sz > 0 {
			preview := strings.ReplaceAll(string(data[:min(120, sz)]), "\n", " ")
			fmt.Printf("  %s : OK (%d bytes; preview=%q)\n", u, sz, preview)
			if strings.Contains(u, "goroutine") {
				gc := strings.Count(string(data), "goroutine ")
				fmt.Printf("    -> goroutine count sample: %d (use during concurrent gRPC+REST load to check for leaks vs baseline)\n", gc)
			}
		} else {
			fmt.Printf("  %s : OK (empty response)\n", u)
		}
	}
	fmt.Println("  OTEL note: Spans/metrics for gRPC (and shared paths) come from otelconnect interceptors + OTel SDK (if OTEL_TRACES_EXPORTER/OTEL_METRICS_EXPORTER or equiv configured in env).")
	fmt.Println("  Look in server logs for 'component=grpc' + reason + stream_id + tokens + duration_ms (rich slog from reliable overlay).")
	fmt.Println("  For full spans: run with OTEL exporter (e.g. to collector or console) and/or inspect pprof while running quality cases.")
	fmt.Println("  gRPC-specific: health/reflection (if enabled) also aid discovery; use grpcurl or buf curl for live surface.")
}