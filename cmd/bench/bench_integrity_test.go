package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
)

// TestBenchmarkModel_OutputFileTruncates verifies that reused output files
// cannot retain stale records from an earlier, longer run.
func TestBenchmarkModel_OutputFileTruncates(t *testing.T) {
	opts := createTestFlagOptions()
	output := filepath.Join(t.TempDir(), "result.bench")
	stale := []byte(strings.Repeat("OLD-BENCHMARK-DATA", 1000))
	if err := os.WriteFile(output, stale, 0o600); err != nil {
		t.Fatal(err)
	}
	opts.outputFile = &output

	server := createMockOllamaServer(t, mockServerOptions{chatResponses: defaultChatResponses()})
	defer server.Close()
	t.Setenv("OLLAMA_HOST", server.URL)

	if err := BenchmarkModel(opts); err != nil {
		t.Fatal(err)
	}
	got, err := os.ReadFile(output)
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Contains(got, []byte("OLD-BENCHMARK-DATA")) {
		t.Fatal("output retained stale data from a previous run")
	}
	if !bytes.Contains(got, []byte("BenchmarkModel/name=test-model/step=total")) {
		t.Fatalf("output does not contain expected benchmark metrics: %s", got)
	}
}

// TestBenchmarkModel_PartialEpochsFail keeps already-produced measurements,
// but requires an error return when one of the requested epochs fails.
func TestBenchmarkModel_PartialEpochsFail(t *testing.T) {
	opts := createTestFlagOptions()
	epochs, maxTokens := 2, 0
	opts.epochs = &epochs
	opts.maxTokens = &maxTokens
	var calls atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		switch r.URL.Path {
		case "/api/chat":
			if calls.Add(1) == 2 {
				http.Error(w, "backend unavailable", http.StatusServiceUnavailable)
				return
			}
			_ = json.NewEncoder(w).Encode(api.ChatResponse{
				Model:   "test-model",
				Message: api.Message{Role: "assistant", Content: "response"},
				Done:    true,
				Metrics: api.Metrics{
					PromptEvalCount:    10,
					PromptEvalDuration: 100 * time.Millisecond,
					EvalCount:          50,
					EvalDuration:       500 * time.Millisecond,
					TotalDuration:      700 * time.Millisecond,
				},
			})
		case "/api/show":
			_ = json.NewEncoder(w).Encode(api.ShowResponse{})
		case "/api/ps":
			_ = json.NewEncoder(w).Encode(api.ProcessResponse{})
		case "/api/generate":
			_ = json.NewEncoder(w).Encode(api.GenerateResponse{Done: true})
		default:
			http.NotFound(w, r)
		}
	}))
	defer server.Close()
	t.Setenv("OLLAMA_HOST", server.URL)

	output := captureOutput(func() {
		err := BenchmarkModel(opts)
		if err == nil || !strings.Contains(err.Error(), "test-model: 1/2 epochs completed") {
			t.Errorf("expected incomplete benchmark error, got %v", err)
		}
	})
	if calls.Load() != 2 {
		t.Fatalf("expected 2 chat requests, got %d", calls.Load())
	}
	if !strings.Contains(output, "BenchmarkModel/name=test-model/step=total") {
		t.Fatalf("successful epoch metrics were lost: %s", output)
	}
}

func TestBenchmarkModel_MissingFinalMetricsFail(t *testing.T) {
	opts := createTestFlagOptions()
	maxTokens := 0
	opts.maxTokens = &maxTokens
	server := createMockOllamaServer(t, mockServerOptions{chatResponses: []api.ChatResponse{
		{Model: "test-model", Message: api.Message{Role: "assistant", Content: "partial"}, Done: false},
	}})
	defer server.Close()
	t.Setenv("OLLAMA_HOST", server.URL)

	captureOutput(func() {
		err := BenchmarkModel(opts)
		if err == nil || !strings.Contains(err.Error(), "test-model: 0/1 epochs completed") {
			t.Errorf("expected missing final metrics to fail, got %v", err)
		}
	})
}

// TestBenchmarkModel_MultipleModelsPartialFailure verifies that a failed model
// does not hide a successful model or turn the aggregate exit status into PASS.
func TestBenchmarkModel_MultipleModelsPartialFailure(t *testing.T) {
	opts := createTestFlagOptions()
	models, maxTokens := "bad-model,good-model", 0
	opts.models, opts.maxTokens = &models, &maxTokens
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		switch r.URL.Path {
		case "/api/show":
			_ = json.NewEncoder(w).Encode(api.ShowResponse{})
		case "/api/ps":
			_ = json.NewEncoder(w).Encode(api.ProcessResponse{})
		case "/api/generate":
			_ = json.NewEncoder(w).Encode(api.GenerateResponse{Done: true})
		case "/api/chat":
			var req api.ChatRequest
			if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
				http.Error(w, "invalid request", http.StatusBadRequest)
				return
			}
			if req.Model == "bad-model" {
				http.Error(w, "backend unavailable", http.StatusServiceUnavailable)
				return
			}
			_ = json.NewEncoder(w).Encode(api.ChatResponse{
				Model:   req.Model,
				Message: api.Message{Role: "assistant", Content: "result"},
				Done:    true,
				Metrics: api.Metrics{
					PromptEvalCount:    8,
					PromptEvalDuration: 20 * time.Millisecond,
					EvalCount:          10,
					EvalDuration:       100 * time.Millisecond,
					TotalDuration:      160 * time.Millisecond,
				},
			})
		default:
			http.NotFound(w, r)
		}
	}))
	defer server.Close()
	t.Setenv("OLLAMA_HOST", server.URL)

	output := captureOutput(func() {
		err := BenchmarkModel(opts)
		if err == nil || !strings.Contains(err.Error(), "bad-model: 0/1 epochs completed") {
			t.Errorf("expected aggregate failure for bad-model, got %v", err)
		}
	})
	if !strings.Contains(output, "BenchmarkModel/name=good-model/step=total") {
		t.Fatalf("second model was not measured after first failed: %s", output)
	}
}
// TestBenchmarkOpenAI_IncompleteEpochs checks that both fully failed and
// partially completed OpenAI-compatible runs return a failure while retaining
// metrics produced by successful epochs.
func TestBenchmarkOpenAI_IncompleteEpochs(t *testing.T) {
	for _, tc := range []struct {
		name       string
		successful int32
	}{
		{name: "all failed", successful: 0},
		{name: "partially completed", successful: 1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var calls atomic.Int32
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.URL.Path != "/v1/chat/completions" {
					http.NotFound(w, r)
					return
				}
				if calls.Add(1) > tc.successful {
					http.Error(w, "backend unavailable", http.StatusServiceUnavailable)
					return
				}
				w.Header().Set("Content-Type", "text/event-stream")
				_, _ = w.Write([]byte("data: {\"choices\":[{\"delta\":{\"content\":\"hello\"}}]}\n\n"))
				_, _ = w.Write([]byte("data: {\"choices\":[],\"usage\":{\"prompt_tokens\":12,\"completion_tokens\":5}}\n\n"))
				_, _ = w.Write([]byte("data: [DONE]\n\n"))
			}))
			defer server.Close()

			opts := createTestFlagOptions()
			epochs, maxTokens := 2, 0
			opts.epochs, opts.maxTokens = &epochs, &maxTokens
			openaiURL := server.URL + "/v1"
			opts.openaiURL = &openaiURL
			var out bytes.Buffer
			err := benchmarkOpenAI(opts, []string{"test-model"}, &out)
			want := fmt.Sprintf("incomplete benchmark: test-model: %d/2 epochs completed", tc.successful)
			if err == nil || !strings.Contains(err.Error(), want) {
				t.Fatalf("benchmarkOpenAI() error = %v, want %q", err, want)
			}
			if calls.Load() != 2 {
				t.Fatalf("requests = %d, want 2", calls.Load())
			}
			if got := strings.Count(out.String(), "step=total"); got != int(tc.successful) {
				t.Errorf("completed epoch results = %d, want %d; output: %s", got, tc.successful, out.String())
			}
		})
	}
}
