package server

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"
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
