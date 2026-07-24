package server

import (
	"encoding/json"
	"net/http"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/fs/ggml"
)

func createEmbedTestModel(t *testing.T, s *Server, name string) {
	t.Helper()

	createMinimalGGUFModel(t, s, name, ggml.KV{
		"add_bos_token": false,
		"add_eos_token": false,
	}, "", map[string]any{
		"capabilities": []string{"embedding"},
	})
}

func TestEmbedHandlerExplicitNumBatchTruncatesInput(t *testing.T) {
	t.Setenv("OLLAMA_CONTEXT_LENGTH", "8192")
	gin.SetMode(gin.TestMode)

	mock := mockRunner{}
	s := newServerWithMockRunner(t, &mock)
	createEmbedTestModel(t, s, "embed-batch-truncate")

	truncate := true
	w := createRequest(t, s.EmbedHandler, api.EmbedRequest{
		Model:    "embed-batch-truncate",
		Input:    "one two three four five",
		Truncate: &truncate,
		Options: map[string]any{
			"num_batch": float64(3),
		},
	})
	if w.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d: %s", w.Code, w.Body.String())
	}

	var resp api.EmbedResponse
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatal(err)
	}
	if len(resp.Embeddings) != 1 {
		t.Fatalf("expected one embedding, got %#v", resp.Embeddings)
	}
	if got := mock.EmbeddingInputs; len(got) != 1 || got[0] != "x x x" {
		t.Fatalf("embedding inputs = %#v, want one truncated input", got)
	}
}

func TestEmbedHandlerExplicitNumBatchNoTruncateErrors(t *testing.T) {
	t.Setenv("OLLAMA_CONTEXT_LENGTH", "8192")
	gin.SetMode(gin.TestMode)

	mock := mockRunner{}
	s := newServerWithMockRunner(t, &mock)
	createEmbedTestModel(t, s, "embed-batch-no-truncate")

	truncate := false
	w := createRequest(t, s.EmbedHandler, api.EmbedRequest{
		Model:    "embed-batch-no-truncate",
		Input:    "one two three four five",
		Truncate: &truncate,
		Options: map[string]any{
			"num_batch": float64(3),
		},
	})
	if w.Code != http.StatusBadRequest {
		t.Fatalf("expected status 400, got %d: %s", w.Code, w.Body.String())
	}
	if got := mock.EmbeddingInputs; len(got) != 0 {
		t.Fatalf("embedding inputs = %#v, want none", got)
	}
	if !strings.Contains(w.Body.String(), "exceeds configured num_batch") {
		t.Fatalf("expected num_batch error, got %s", w.Body.String())
	}
}
