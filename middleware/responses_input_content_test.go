package middleware

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/google/go-cmp/cmp"
	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/openai"
)

func TestResponsesMiddlewareRejectsUnrepresentableHistory(t *testing.T) {
	gin.SetMode(gin.TestMode)
	for _, tt := range []struct{ name, item, want string }{
		{"missing image", `{"role":"user","content":[{"type":"input_text","text":"Keep this."},{"type":"input_image"}]}`, "image_url"},
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

func TestResponsesMiddlewareContinuesWithUnavailableImage(t *testing.T) {
	gin.SetMode(gin.TestMode)
	for _, stream := range []bool{false, true} {
		var captured api.ChatRequest
		router := gin.New()
		router.POST("/v1/responses", ResponsesMiddleware(), func(c *gin.Context) {
			if err := json.NewDecoder(c.Request.Body).Decode(&captured); err != nil {
				t.Fatal(err)
			}
			c.JSON(http.StatusOK, api.ChatResponse{Model: "test", Message: api.Message{Role: "assistant", Content: "continued"}, Done: true})
		})
		history := []json.RawMessage{
			json.RawMessage(`{"role":"user","content":"Earlier useful text."}`),
			json.RawMessage(`{"role":"user","content":[{"type":"input_text","text":"Inspect this."},{"type":"input_image","file_id":"file_private"}]}`),
		}
		for turn := range 2 {
			body, err := json.Marshal(map[string]any{"model": "test", "stream": stream, "input": history})
			if err != nil {
				t.Fatal(err)
			}
			req := httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(string(body)))
			req.Header.Set("Content-Type", "application/json")
			recorder := httptest.NewRecorder()
			router.ServeHTTP(recorder, req)
			if recorder.Code != http.StatusOK || !strings.Contains(recorder.Body.String(), "continued") {
				t.Fatalf("stream=%v turn=%d status=%d body=%s", stream, turn, recorder.Code, recorder.Body.String())
			}
			want := []api.Message{
				{Role: "user", Content: "Earlier useful text."},
				{Role: "user", Content: "Inspect this.\n\n[This image is unavailable. Attach the image again, or continue without the image.]\n\n"},
			}
			if turn == 1 {
				want = append(want, api.Message{Role: "user", Content: "hi"})
			}
			if diff := cmp.Diff(want, captured.Messages); diff != "" {
				t.Fatalf("history changed (-want +got):\n%s", diff)
			}
			if !strings.Contains(string(history[1]), "file_private") {
				t.Fatal("the original image reference changed")
			}
			history = append(history, json.RawMessage(`{"role":"user","content":"hi"}`))
		}
	}
}
