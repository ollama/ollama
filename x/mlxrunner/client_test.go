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

func TestClientCompletionRequestsIntermediateMetrics(t *testing.T) {
	var request CompletionRequest
	want := CompletionResponse{
		Done:                  true,
		PromptEvalCount:       10,
		PromptEvalCachedCount: 4,
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
	if got.PromptEvalCount != want.PromptEvalCount || got.PromptEvalCachedCount != want.PromptEvalCachedCount {
		t.Errorf("prompt counts = (%d, %d), want (%d, %d)", got.PromptEvalCount, got.PromptEvalCachedCount, want.PromptEvalCount, want.PromptEvalCachedCount)
	}
}
