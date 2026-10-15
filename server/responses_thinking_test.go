package server

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/middleware"
	"github.com/ollama/ollama/openai"
)

func TestResponsesThinkingControls(t *testing.T) {
	gin.SetMode(gin.TestMode)
	t.Setenv("OLLAMA_MODELS", t.TempDir())
	t.Setenv("OLLAMA_CONTEXT_LENGTH", "4096")
	mock := mockRunner{}
	s := newServerWithMockRunner(t, &mock)
	createMinimalGGUFModel(t, s, "legacy", nil, "{{ .Prompt }}", nil)
	for _, model := range []struct{ name, renderer string }{{"boolean", "qwen3.5"}, {"named", "qwen3.8"}} {
		w := createRequest(t, s.CreateHandler, api.CreateRequest{Model: model.name, From: "legacy", Renderer: model.renderer, Parser: "qwen3.5", Stream: &stream})
		if w.Code != http.StatusOK {
			t.Fatal(w.Body.String())
		}
	}
	router := gin.New()
	router.POST("/v1/responses", middleware.ResponsesMiddleware(lookupThinking), s.ChatHandler)

	for _, model := range []string{"legacy", "boolean", "named"} {
		for _, effort := range []string{"", "none", "minimal", "low", "medium", "high", "xhigh", "ultra", "max", "future"} {
			for _, streaming := range []bool{false, true} {
				t.Run(fmt.Sprintf("%s/%s/stream=%t", model, effort, streaming), func(t *testing.T) {
					thinking := model != "legacy" && effort != "none"
					mock.CompletionResponse = llm.CompletionResponse{Content: "answer", Done: true, DoneReason: llm.DoneReasonStop}
					if thinking {
						mock.CompletionResponse.Content = "reason</think>answer"
					}
					body := fmt.Sprintf(`{"model":%q,"input":"hello","reasoning":{"effort":%q},"stream":%t}`, model, effort, streaming)
					w := NewRecorder()
					router.ServeHTTP(w, httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(body)))
					if model == "legacy" && effort == "future" {
						if w.Code != http.StatusBadRequest || !strings.Contains(w.Body.String(), "invalid reasoning value") {
							t.Fatalf("expected effort validation error: %d %s", w.Code, w.Body.String())
						}
						return
					}
					if w.Code != http.StatusOK {
						t.Fatalf("status %d: %s", w.Code, w.Body.String())
					}
					if model != "legacy" {
						prompt := mock.CompletionRequest.Prompt
						if strings.HasSuffix(prompt, "<think>\n") != thinking {
							t.Fatalf("unexpected thinking prefill: %s", prompt)
						}
						for _, level := range []string{"low", "xhigh"} {
							if strings.Contains(prompt, "Reasoning effort is set to "+level) != (model == "named" && effort == level) {
								t.Fatalf("unexpected %s effort: %s", level, prompt)
							}
						}
					}
					var response openai.ResponsesResponse
					if streaming {
						var answer strings.Builder
						for line := range strings.SplitSeq(w.Body.String(), "\n") {
							data, ok := strings.CutPrefix(line, "data: ")
							if !ok {
								continue
							}
							var event struct {
								Type     string                   `json:"type"`
								Delta    string                   `json:"delta"`
								Response openai.ResponsesResponse `json:"response"`
							}
							if err := json.Unmarshal([]byte(data), &event); err != nil {
								t.Fatal(err)
							}
							if event.Type == "response.output_text.delta" {
								answer.WriteString(event.Delta)
							}
							if event.Type == "response.completed" {
								response = event.Response
							}
						}
						if answer.String() != "answer" {
							t.Fatalf("answer deltas = %q", answer.String())
						}
					} else if err := json.Unmarshal(w.Body.Bytes(), &response); err != nil {
						t.Fatal(err)
					}
					var answer, reasoning string
					for _, item := range response.Output {
						for _, content := range item.Content {
							answer += content.Text
						}
						for _, summary := range item.Summary {
							reasoning += summary.Text
						}
					}
					wantReasoning := ""
					if thinking {
						wantReasoning = "reason"
					}
					if response.Status != "completed" || answer != "answer" || reasoning != wantReasoning {
						t.Fatalf("status=%s answer=%q reasoning=%q", response.Status, answer, reasoning)
					}
				})
			}
		}
	}
}

func TestResponsesThinkingOverrideAndScope(t *testing.T) {
	gin.SetMode(gin.TestMode)
	t.Setenv("OLLAMA_MODELS", t.TempDir())
	mock := mockRunner{CompletionResponse: llm.CompletionResponse{Content: "answer", Done: true, DoneReason: llm.DoneReasonStop}}
	s := newServerWithMockRunner(t, &mock)
	createMinimalGGUFModel(t, s, "legacy", nil, "{{ .Prompt }}", nil)
	router := gin.New()
	router.POST("/v1/responses", middleware.ResponsesMiddleware(lookupThinking), s.ChatHandler)
	router.POST("/v1/chat/completions", middleware.ChatMiddleware(lookupThinking), s.ChatHandler)
	router.POST("/api/chat", s.ChatHandler)
	router.POST("/api/generate", s.GenerateHandler)
	for _, tt := range []struct {
		path, fields string
		status       int
	}{
		{"/v1/responses", `"think":true,"reasoning":{"effort":"none"}`, 400},
		{"/v1/responses", `"think":"high","reasoning":{"effort":"none"}`, 400},
		{"/v1/responses", `"think":"future","reasoning":{"effort":"none"}`, 400},
		{"/v1/responses", `"think":false,"reasoning":{"effort":"high"}`, 200},
		{"/v1/responses", `"think":false,"reasoning":{"effort":"future"}`, 200},
		{"/v1/chat/completions", `"reasoning_effort":"high"`, 400},
		{"/api/chat", `"think":true`, 400},
		{"/api/generate", `"think":"high"`, 400},
	} {
		for _, streaming := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s/%s/stream=%t", tt.path, tt.fields, streaming), func(t *testing.T) {
				body := fmt.Sprintf(`{"model":"legacy","input":"hello","messages":[{"role":"user","content":"hello"}],"prompt":"hello","stream":%t,%s}`, streaming, tt.fields)
				w := NewRecorder()
				router.ServeHTTP(w, httptest.NewRequest(http.MethodPost, tt.path, strings.NewReader(body)))
				if w.Code != tt.status {
					t.Fatalf("status %d, want %d: %s", w.Code, tt.status, w.Body.String())
				}
			})
		}
	}
}
