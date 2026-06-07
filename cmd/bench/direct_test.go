package main

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/ollama/ollama/mlxrunner/wire"
)

// mockRunner is an MLX runner stand-in that records every completion request.
type mockRunner struct {
	mu       sync.Mutex
	requests []wire.CompletionRequest
	cached   int
}

func (m *mockRunner) start(t *testing.T) string {
	t.Helper()
	mux := http.NewServeMux()
	mux.HandleFunc("GET /v1/status", func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(map[string]any{"Status": 0, "Progress": 100, "ContextLength": 4096, "Memory": 1 << 30})
	})
	mux.HandleFunc("POST /v1/completions", func(w http.ResponseWriter, r *http.Request) {
		var req wire.CompletionRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		m.mu.Lock()
		m.requests = append(m.requests, req)
		cached := m.cached
		m.mu.Unlock()

		enc := json.NewEncoder(w)
		evalCount := max(req.Options.NumPredict, 0)
		if evalCount > 0 {
			enc.Encode(wire.CompletionResponse{Content: "x"})
		}
		enc.Encode(wire.CompletionResponse{
			Done:                  true,
			PromptEvalCount:       100,
			PromptEvalCachedCount: &cached,
			PromptEvalDuration:    10 * time.Millisecond,
			EvalCount:             evalCount,
			EvalDuration:          20 * time.Millisecond,
		})
	})
	server := httptest.NewServer(mux)
	t.Cleanup(server.Close)
	return strings.TrimPrefix(server.URL, "http://")
}

func directTestOptions(t *testing.T, mode string, m *mockRunner) flagOptions {
	fOpt := createTestFlagOptions()
	addr := m.start(t)
	fOpt.runner = &addr
	fOpt.mode = &mode
	epochs := 3
	fOpt.epochs = &epochs
	return fOpt
}

func TestDirectParams(t *testing.T) {
	fOpt := createTestFlagOptions()
	ignoreEOS := true
	fOpt.ignoreEOS = &ignoreEOS

	p := directParams(fOpt, modeBoth, "p")
	if p.numPredict != 50 || !p.ignoreEOS || p.seed != 42 {
		t.Errorf("both: got numPredict=%d ignoreEOS=%v seed=%d", p.numPredict, p.ignoreEOS, p.seed)
	}

	p = directParams(fOpt, modePrefill, "p")
	if p.numPredict != 0 || p.ignoreEOS {
		t.Errorf("prefill: got numPredict=%d ignoreEOS=%v, want 0 false", p.numPredict, p.ignoreEOS)
	}

	seed := 0
	fOpt.seed = &seed
	if p := directParams(fOpt, modeBoth, "p"); p.seed != -1 {
		t.Errorf("seed 0 should map to random (-1), got %d", p.seed)
	}
}

func TestBenchmarkDirect_DecodeModeReusesPrompt(t *testing.T) {
	m := &mockRunner{}
	fOpt := directTestOptions(t, modeDecode, m)

	captureOutput(func() {
		if err := benchmarkDirect(fOpt, &strings.Builder{}); err != nil {
			t.Fatal(err)
		}
	})

	// One priming warmup plus three epochs, all on the same prompt.
	if len(m.requests) != 4 {
		t.Fatalf("got %d requests, want 4", len(m.requests))
	}
	for i, req := range m.requests {
		if req.Prompt != m.requests[0].Prompt {
			t.Errorf("request %d prompt differs from the first", i)
		}
	}
}

func TestBenchmarkDirect_PrefillModeUniquePrefillOnly(t *testing.T) {
	m := &mockRunner{}
	fOpt := directTestOptions(t, modePrefill, m)

	captureOutput(func() {
		if err := benchmarkDirect(fOpt, &strings.Builder{}); err != nil {
			t.Fatal(err)
		}
	})

	seen := map[string]bool{}
	for i, req := range m.requests {
		if req.Options.NumPredict != 0 {
			t.Errorf("request %d num_predict=%d, want 0", i, req.Options.NumPredict)
		}
		if seen[req.Prompt] {
			t.Errorf("request %d repeats an earlier prompt", i)
		}
		seen[req.Prompt] = true
	}
}

func TestBenchmarkDirect_ReportsCachedTokens(t *testing.T) {
	m := &mockRunner{cached: 60}
	fOpt := directTestOptions(t, modeDecode, m)
	ignoreEOS := true
	fOpt.ignoreEOS = &ignoreEOS

	var out strings.Builder
	captureOutput(func() {
		if err := benchmarkDirect(fOpt, &out); err != nil {
			t.Fatal(err)
		}
	})

	if !strings.Contains(out.String(), "40 processed-prompt-token 60 cached-prompt-token") {
		t.Errorf("prefill line missing cached split:\n%s", out.String())
	}
	for i, req := range m.requests {
		if !req.IgnoreEOS {
			t.Errorf("request %d did not forward IgnoreEOS", i)
		}
		if req.Options.TopK != 40 {
			t.Errorf("request %d TopK=%d, want API default 40", i, req.Options.TopK)
		}
	}
}

func TestBenchmarkDirect_RejectsUnknownMode(t *testing.T) {
	m := &mockRunner{}
	fOpt := directTestOptions(t, "sideways", m)
	if err := benchmarkDirect(fOpt, &strings.Builder{}); err == nil {
		t.Fatal("expected an error for an unknown mode")
	}
}
