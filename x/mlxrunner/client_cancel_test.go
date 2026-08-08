package mlxrunner

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llm"
)

// TestCompletionStopsCallbacksAfterCancel reproduces the routes-layer
// structured-outputs abort: the caller cancels the context from inside the
// callback, and chunks the client has already buffered must NOT keep
// flowing afterwards (they leak pass-one content into the response).
func TestCompletionStopsCallbacksAfterCancel(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		enc := json.NewEncoder(w)
		// All three lines land in one write, so the client buffers them
		// together before the cancellation takes effect transport-side.
		enc.Encode(CompletionResponse{Content: "one"})               //nolint:errcheck
		enc.Encode(CompletionResponse{Content: "two"})               //nolint:errcheck
		enc.Encode(CompletionResponse{Content: "three", Done: true}) //nolint:errcheck
	}))
	defer srv.Close()

	c := &Client{
		port:   srv.Listener.Addr().(*net.TCPAddr).Port,
		client: http.DefaultClient,
		status: llm.NewStatusWriter(io.Discard),
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	var calls []string
	err := c.Completion(ctx, llm.CompletionRequest{Prompt: "p", Options: &api.Options{}}, func(r llm.CompletionResponse) {
		calls = append(calls, r.Content)
		cancel()
	})
	if err == nil || !errors.Is(err, context.Canceled) {
		t.Fatalf("err = %v, want context.Canceled", err)
	}
	if len(calls) != 1 {
		t.Fatalf("callback ran %d times (%v), want 1: buffered chunks leaked past cancellation", len(calls), calls)
	}
}
