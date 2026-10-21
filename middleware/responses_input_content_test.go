package middleware

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/openai"
)

func TestResponsesMiddlewareReadableHistory(t *testing.T) {
	gin.SetMode(gin.TestMode)
	for _, stream := range []bool{false, true} {
		t.Run(fmt.Sprintf("stream=%t", stream), func(t *testing.T) {
			router := gin.New()
			calls := 0
			router.POST("/v1/responses", ResponsesMiddleware(), func(c *gin.Context) {
				calls++
				var req api.ChatRequest
				if err := c.ShouldBindJSON(&req); err != nil {
					t.Fatal(err)
				}
				if len(req.Messages) != 4 || req.Messages[0].Content != "Before." || req.Messages[1].Thinking != "Keep the original note." || req.Messages[1].Content != "I cannot delete the note." || len(req.Messages[1].ToolCalls) != 1 || req.Messages[1].ToolCalls[0].ID != "call_read" || req.Messages[2].ToolCallID != "call_read" || req.Messages[2].Content != "note contents" || req.Messages[3].Content != "Continue." {
					t.Fatalf("history changed: %+v", req.Messages)
				}
				c.JSON(http.StatusOK, api.ChatResponse{Model: "test", Message: api.Message{Role: "assistant", Content: "The note is intact."}, Done: true})
			})
			body := fmt.Sprintf(`{"model":"test","stream":%t,"input":[
				{"role":"user","content":"Before."},
				{"type":"reasoning","content":[{"type":"reasoning_text","text":"Keep the original note."}]},
				{"type":"custom_tool_call","call_id":"call_read","name":"read_note","input":"note.txt"},
				{"role":"assistant","content":[{"type":"refusal","refusal":"I cannot delete the note."}]},
				{"type":"custom_tool_call_output","call_id":"call_read","output":"note contents"},
				{"role":"user","content":"Continue."}
			]}`, stream)
			req := httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(body))
			req.Header.Set("Content-Type", "application/json")
			recorder := httptest.NewRecorder()
			router.ServeHTTP(recorder, req)
			if recorder.Code != http.StatusOK || calls != 1 || !strings.Contains(recorder.Body.String(), "The note is intact.") {
				t.Fatalf("status=%d calls=%d body=%s", recorder.Code, calls, recorder.Body.String())
			}
			if stream && !strings.Contains(recorder.Body.String(), "event: response.completed") {
				t.Fatalf("stream did not complete: %s", recorder.Body.String())
			}
		})
	}
}

func TestResponsesMiddlewareRejectsUnrepresentableHistory(t *testing.T) {
	gin.SetMode(gin.TestMode)
	for _, tt := range []struct{ name, item, want string }{
		{"image reference", `{"role":"user","content":[{"type":"input_text","text":"Keep this."},{"type":"input_image","file_id":"file_private"}]}`, "file_id"},
		{"audio", `{"role":"user","content":[{"type":"input_audio","audio_url":"private-content"}]}`, "input_audio"},
		{"unknown reasoning", `{"type":"reasoning","content":[{"type":"future_reasoning","text":"private-content"}]}`, "future_reasoning"},
		{"unknown item", `{"type":"future_item","text":"private-content"}`, "future_item"},
	} {
		for _, stream := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s/stream=%t", tt.name, stream), func(t *testing.T) {
				router := gin.New()
				router.POST("/v1/responses", ResponsesMiddleware(), func(c *gin.Context) {
					t.Error("invalid history reached inference")
				})
				body := fmt.Sprintf(`{"model":"test","stream":%t,"input":[{"role":"user","content":"Before."},%s,{"role":"user","content":"After."}]}`, stream, tt.item)
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
}
