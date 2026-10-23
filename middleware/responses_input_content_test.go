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

func TestResponsesMiddlewareRejectsUnrepresentableHistory(t *testing.T) {
	gin.SetMode(gin.TestMode)
	for _, tt := range []struct{ name, item, want string }{
		{"image reference", `{"role":"user","content":[{"type":"input_text","text":"Keep this."},{"type":"input_image","file_id":"file_private"}]}`, "file_id"},
		{"unknown reasoning", `{"type":"reasoning","content":[{"type":"future_reasoning","text":"private-content"}]}`, "future_reasoning"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			router := gin.New()
			router.POST("/v1/responses", ResponsesMiddleware(), func(c *gin.Context) {
				t.Error("invalid history reached inference")
			})
			// Both parsing and conversion errors must return JSON before streaming starts.
			body := `{"model":"test","stream":true,"input":[{"role":"user","content":"Before."},` + tt.item + `,{"role":"user","content":"After."}]}`
			req := httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(body))
			req.Header.Set("Content-Type", "application/json")
			recorder := httptest.NewRecorder()
			router.ServeHTTP(recorder, req)
			var response openai.ErrorResponse
			if err := json.Unmarshal(recorder.Body.Bytes(), &response); err != nil {
				t.Fatal(err)
			}
			if recorder.Code != http.StatusBadRequest || response.Error.Type != "invalid_request_error" || !strings.Contains(response.Error.Message, tt.want) || strings.Contains(recorder.Body.String(), "private-content") || strings.Contains(recorder.Body.String(), "file_private") {
				t.Fatalf("unexpected error: status=%d body=%s", recorder.Code, recorder.Body.String())
			}
		})
	}
}
