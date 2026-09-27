package server

import (
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llm"
)

// A model that states its thinking controls has an unsupported think value
// replaced by its default before rendering. A thinking-token budget is never
// one of those controls, so resolving first and budgeting second dropped every
// budget on such a model, and let the default level arm a budget the caller
// never asked for. The budget is read from what the caller sent; the resolved
// value only decides whether the model thinks at all.
func TestThinkBudgetSurvivesThinkingResolution(t *testing.T) {
	gin.SetMode(gin.TestMode)
	t.Setenv("OLLAMA_MODELS", t.TempDir())
	t.Setenv("OLLAMA_CONTEXT_LENGTH", "4096")
	mock := mockRunner{CompletionResponse: llm.CompletionResponse{Content: "reason</think>answer", Done: true, DoneReason: llm.DoneReasonStop}}
	s := newServerWithMockRunner(t, &mock)
	createMinimalGGUFModel(t, s, "thinking-base", nil, "{{ .Prompt }}", nil)
	// qwen3.8 states [false, "low", "medium", "xhigh"] with "medium" as default.
	if w := createRequest(t, s.CreateHandler, api.CreateRequest{Model: "thinking-qwen", From: "thinking-base", Renderer: "qwen3.8", Parser: "qwen3.5", Stream: &stream}); w.Code != http.StatusOK {
		t.Fatalf("create: %s", w.Body.String())
	}

	for _, tt := range []struct {
		name   string
		think  any
		budget int
	}{
		{"integer budget", 1500, 1500},
		{"level the model does not state", "max", 4096 * 4 / 5},
		{"level the model states", "low", 4096 / 8},
		{"omitted: the model's default level is not a budget", nil, 0},
		{"true", true, 0},
		{"false", false, 0},
	} {
		for _, endpoint := range []string{"chat", "generate"} {
			t.Run(endpoint+"/"+tt.name, func(t *testing.T) {
				var think *api.ThinkValue
				if tt.think != nil {
					think = &api.ThinkValue{Value: tt.think}
				}
				var w *httptest.ResponseRecorder
				if endpoint == "chat" {
					w = createRequest(t, s.ChatHandler, api.ChatRequest{Model: "thinking-qwen", Messages: []api.Message{{Role: "user", Content: "hello"}}, Think: think, Stream: &stream})
				} else {
					w = createRequest(t, s.GenerateHandler, api.GenerateRequest{Model: "thinking-qwen", Prompt: "hello", Think: think, Stream: &stream})
				}
				if code := w.Code; code != http.StatusOK {
					t.Fatalf("status %d", code)
				}
				if got := mock.CompletionRequest.ThinkBudget; got != tt.budget {
					t.Errorf("runner got think budget %d, want %d", got, tt.budget)
				}
			})
		}
	}
}
