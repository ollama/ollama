package middleware

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/ollama/ollama/openai"
)

func TestChatMiddlewareRejectsInvalidContentWithToolCalls(t *testing.T) {
	for _, content := range []string{`42`, `true`, `{"text":"do not discard"}`} {
		t.Run(content, func(t *testing.T) {
			called := false
			router := gin.New()
			router.POST("/v1/chat/completions", ChatMiddleware(), func(c *gin.Context) {
				called = true
			})
			body := `{"model":"test","messages":[{"role":"assistant","content":` + content + `,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"lookup","arguments":"{}"}}]}]}`
			req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
			req.Header.Set("Content-Type", "application/json")
			w := httptest.NewRecorder()
			router.ServeHTTP(w, req)
			if w.Code != http.StatusBadRequest || called {
				t.Fatalf("status = %d, downstream called = %v; want 400 before generation", w.Code, called)
			}
			var response openai.ErrorResponse
			if err := json.Unmarshal(w.Body.Bytes(), &response); err != nil {
				t.Fatal(err)
			}
			if response.Error.Type != "invalid_request_error" || !strings.Contains(response.Error.Message, "invalid message content type") {
				t.Fatalf("unexpected error response: %+v", response)
			}
		})
	}
}
