package server

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/middleware"
)

func TestResponsesNonthinkingEffort(t *testing.T) {
	gin.SetMode(gin.TestMode)
	t.Setenv("OLLAMA_MODELS", t.TempDir())
	t.Setenv("OLLAMA_CONTEXT_LENGTH", "4096")
	mock := mockRunner{CompletionResponse: llm.CompletionResponse{Content: "answer", Done: true, DoneReason: llm.DoneReasonStop}}
	s := newServerWithMockRunner(t, &mock)
	createMinimalGGUFModel(t, s, "nonthinking", nil, "{{ .Prompt }}", nil)
	router := gin.New()
	router.POST("/v1/responses", middleware.ResponsesMiddleware(lookupThinking), s.ChatHandler)

	for _, tt := range []struct {
		name, fields string
		status       int
		response     string
	}{
		{"inherited effort", `"reasoning":{"effort":"high"}`, http.StatusOK, "answer"},
		{"explicit think on", `"think":true,"reasoning":{"effort":"none"}`, http.StatusBadRequest, "does not support thinking"},
		{"explicit think off", `"think":false,"reasoning":{"effort":"high"}`, http.StatusOK, "answer"},
		{"invalid effort", `"reasoning":{"effort":"future"}`, http.StatusBadRequest, "invalid reasoning value"},
	} {
		for _, streaming := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s/stream=%t", tt.name, streaming), func(t *testing.T) {
				body := fmt.Sprintf(`{"model":"nonthinking","input":"hello","stream":%t,%s}`, streaming, tt.fields)
				w := NewRecorder()
				router.ServeHTTP(w, httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(body)))
				if w.Code != tt.status || !strings.Contains(w.Body.String(), tt.response) {
					t.Fatalf("got %d %s, want %d containing %q", w.Code, w.Body.String(), tt.status, tt.response)
				}
			})
		}
	}
}

func TestResponsesNonthinkingEffortWebSearch(t *testing.T) {
	gin.SetMode(gin.TestMode)
	for _, streaming := range []bool{false, true} {
		t.Run(fmt.Sprintf("stream=%t", streaming), func(t *testing.T) {
			t.Setenv("OLLAMA_MODELS", t.TempDir())
			t.Setenv("OLLAMA_CONTEXT_LENGTH", "4096")
			var completions, searches atomic.Int32
			mock := mockRunner{CompletionFn: func(_ context.Context, req llm.CompletionRequest, yield func(llm.CompletionResponse)) error {
				content := "answer"
				if completions.Add(1) == 1 {
					content = `{"name":"web_search","arguments":{"query":"test"}}`
				} else if !strings.Contains(req.Prompt, "test search result") {
					return fmt.Errorf("follow-up is missing the search result")
				}
				yield(llm.CompletionResponse{Content: content, Done: true, DoneReason: llm.DoneReasonStop})
				return nil
			}}
			s := newServerWithMockRunner(t, &mock)
			createMinimalGGUFModel(t, s, "nonthinking", nil, `{{ if .Tools }}{{ .Tools }}{{ end }}
{{ range .Messages }}{{ .Role }}: {{ .Content }}
{{ range .ToolCalls }}{"name":"{{ .Function.Name }}","arguments":{{ .Function.Arguments }}}{{ end }}
{{ end }}`, nil)
			router := gin.New()
			router.POST("/v1/responses", middleware.ResponsesMiddleware(lookupThinking), s.ChatHandler)
			router.POST("/api/chat", s.ChatHandler)
			router.POST("/api/experimental/web_search", func(c *gin.Context) {
				searches.Add(1)
				c.JSON(http.StatusOK, api.WebSearchResponse{Results: []api.WebSearchResult{{
					Title: "Test", URL: "https://example.com", Content: "test search result",
				}}})
			})
			backend := httptest.NewServer(router)
			defer backend.Close()
			t.Setenv("OLLAMA_HOST", backend.URL)

			body := fmt.Sprintf(`{"model":"nonthinking","input":"search test","tools":[{"type":"web_search"}],"reasoning":{"effort":"high"},"stream":%t}`, streaming)
			w := NewRecorder()
			router.ServeHTTP(w, httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(body)))
			if w.Code != http.StatusOK || !strings.Contains(w.Body.String(), "answer") || strings.Contains(w.Body.String(), "response.failed") {
				t.Fatalf("expected successful search follow-up, got %d %s", w.Code, w.Body.String())
			}
			if streaming && !strings.Contains(w.Body.String(), "response.completed") {
				t.Fatalf("missing completion event: %s", w.Body.String())
			}
			if searches.Load() != 1 || completions.Load() != 2 {
				t.Fatalf("searches=%d completions=%d, want 1 search and 2 completions", searches.Load(), completions.Load())
			}
		})
	}
}
