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
