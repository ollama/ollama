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
// replaced by its default before rendering. None of that touches the budget:
// the bound is think_budget, whatever level the request names and whatever the
// model resolves it to. A level never arms a budget, and never disarms one.
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
		name    string
		think   any
		options map[string]any
		budget  int
	}{
		{"a level the model does not state is not a budget", "max", nil, 0},
		{"a level the model states is not a budget", "low", nil, 0},
		{"omitted: the model's default level is not a budget", nil, nil, 0},
		{"true", true, nil, 0},
		{"false", false, nil, 0},
		{"a count with thinking on", true, map[string]any{"think_budget": 1500}, 1500},
		{"a share with thinking on", true, map[string]any{"think_budget": "max"}, 4096 * 4 / 5},
		{"a count beside a level the model states", "low", map[string]any{"think_budget": 1500}, 1500},
		{"a count beside a level the model does not state", "max", map[string]any{"think_budget": 1500}, 1500},
		{"a count with think omitted", nil, map[string]any{"think_budget": 1500}, 1500},
		{"a count with thinking off", false, map[string]any{"think_budget": 1500}, 0},
	} {
		for _, endpoint := range []string{"chat", "generate"} {
			t.Run(endpoint+"/"+tt.name, func(t *testing.T) {
				var think *api.ThinkValue
				if tt.think != nil {
					think = &api.ThinkValue{Value: tt.think}
				}
				var w *httptest.ResponseRecorder
				if endpoint == "chat" {
					w = createRequest(t, s.ChatHandler, api.ChatRequest{Model: "thinking-qwen", Messages: []api.Message{{Role: "user", Content: "hello"}}, Think: think, Options: tt.options, Stream: &stream})
				} else {
					w = createRequest(t, s.GenerateHandler, api.GenerateRequest{Model: "thinking-qwen", Prompt: "hello", Think: think, Options: tt.options, Stream: &stream})
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
