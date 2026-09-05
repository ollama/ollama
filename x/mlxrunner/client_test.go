package mlxrunner

import (
	"encoding/json"
	"net"
	"net/http"
	"net/http/httptest"
	"strconv"
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llm"
)

func TestRunnerArgsContextLengthPrecedence(t *testing.T) {
	for _, tt := range []struct {
		name        string
		hardContext int
		want        []string
	}{
		{
			name: "automatic soft context leaves runner at model capacity",
			want: []string{"runner", "--mlx-engine", "--model", "qwen3.5:27b", "--port", "49152"},
		},
		{
			name:        "explicit hard context is enforced",
			hardContext: 65536,
			want:        []string{"runner", "--mlx-engine", "--model", "qwen3.5:27b", "--port", "49152", "--ctx-size", "65536"},
		},
	} {
		t.Run(tt.name, func(t *testing.T) {
			got := runnerArgs("qwen3.5:27b", 49152, tt.hardContext)
			if len(got) != len(tt.want) {
				t.Fatalf("runnerArgs = %q, want %q", got, tt.want)
			}
			for i := range tt.want {
				if got[i] != tt.want[i] {
					t.Fatalf("runnerArgs[%d] = %q, want %q", i, got[i], tt.want[i])
				}
			}
		})
	}
}

func TestReportedContextLengthPrecedence(t *testing.T) {
	for _, tt := range []struct {
		name         string
		softContext  int
		hardContext  int
		modelContext int
		want         int
	}{
		{name: "automatic context remains a soft reporting limit", softContext: 32768, modelContext: 262144, want: 32768},
		{name: "model capacity caps automatic soft context", softContext: 524288, modelContext: 262144, want: 262144},
		{name: "explicit hard context takes precedence over soft", softContext: 32768, hardContext: 65536, modelContext: 65536, want: 65536},
		{name: "model capacity caps explicit hard context", softContext: 32768, hardContext: 524288, modelContext: 262144, want: 262144},
	} {
		t.Run(tt.name, func(t *testing.T) {
			client := Client{softContextLength: tt.softContext, hardContextLength: tt.hardContext}
			if got := client.reportedContextLength(tt.modelContext); got != tt.want {
				t.Fatalf("reportedContextLength(%d) = %d, want %d", tt.modelContext, got, tt.want)
			}
		})
	}
}

func TestPingReportsRunnerContextLength(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/status" {
			t.Errorf("path = %q, want /v1/status", r.URL.Path)
		}
		if err := json.NewEncoder(w).Encode(statusResponse{ContextLength: 65536, Memory: 42}); err != nil {
			t.Errorf("encode response: %v", err)
		}
	}))
	t.Cleanup(srv.Close)

	_, portString, err := net.SplitHostPort(srv.Listener.Addr().String())
	if err != nil {
		t.Fatalf("parse server port: %v", err)
	}
	port, err := strconv.Atoi(portString)
	if err != nil {
		t.Fatalf("parse server port: %v", err)
	}
	client := &Client{port: port, client: srv.Client()}
	if err := client.Ping(t.Context()); err != nil {
		t.Fatalf("Ping: %v", err)
	}
	if got := client.ContextLength(); got != 65536 {
		t.Fatalf("ContextLength = %d, want 65536", got)
	}
	if got := client.memory.Load(); got != 42 {
		t.Fatalf("memory = %d, want 42", got)
	}
}

func testIntPtr(v int) *int {
	return &v
}

func TestRequestGrammar(t *testing.T) {
	schema := `{"type":"object","properties":{"answer":{"type":"string"}}}`
	tag := `{"type":"structural_tag","format":{"type":"json_schema","json_schema":` + schema + `}}`
	for _, tt := range []struct {
		name string
		req  llm.CompletionRequest
		want string
	}{
		{name: "unset"},
		{name: "null", req: llm.CompletionRequest{Format: json.RawMessage(`null`)}},
		{name: "empty", req: llm.CompletionRequest{Format: json.RawMessage(`""`)}},
		{
			name: "json",
			req:  llm.CompletionRequest{Format: json.RawMessage(`"json"`)},
			want: `{"type":"structural_tag","format":{"type":"json_schema","json_schema":{"type":"object"}}}`,
		},
		{name: "schema", req: llm.CompletionRequest{Format: json.RawMessage(schema)}, want: tag},
	} {
		t.Run(tt.name, func(t *testing.T) {
			if got := string(requestGrammar(tt.req)); got != tt.want {
				t.Fatalf("requestGrammar = %s, want %s", got, tt.want)
			}
		})
	}
}

func TestClientCompletionRequestsIntermediateMetrics(t *testing.T) {
	var request CompletionRequest
	want := CompletionResponse{
		Done:                  true,
		PromptEvalCount:       10,
		PromptEvalCachedCount: testIntPtr(4),
	}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			t.Errorf("decode request: %v", err)
			return
		}
		if err := json.NewEncoder(w).Encode(want); err != nil {
			t.Errorf("encode response: %v", err)
		}
	}))
	t.Cleanup(srv.Close)

	_, portString, err := net.SplitHostPort(srv.Listener.Addr().String())
	if err != nil {
		t.Fatalf("parse server port: %v", err)
	}
	port, err := strconv.Atoi(portString)
	if err != nil {
		t.Fatalf("parse server port: %v", err)
	}
	client := &Client{port: port, client: srv.Client()}
	opts := api.DefaultOptions()
	var got llm.CompletionResponse
	if err := client.Completion(t.Context(), llm.CompletionRequest{
		Options:                    &opts,
		IncludeIntermediateMetrics: true,
	}, func(response llm.CompletionResponse) { got = response }); err != nil {
		t.Fatalf("Completion: %v", err)
	}
	if !request.IncludeIntermediateMetrics {
		t.Fatal("metrics per token was not forwarded to the MLX runner")
	}
	if got.PromptEvalCount != want.PromptEvalCount || got.PromptEvalCachedCount == nil || *got.PromptEvalCachedCount != *want.PromptEvalCachedCount {
		t.Errorf("prompt counts = (%d, %v), want (%d, %d)", got.PromptEvalCount, got.PromptEvalCachedCount, want.PromptEvalCount, *want.PromptEvalCachedCount)
	}
}
