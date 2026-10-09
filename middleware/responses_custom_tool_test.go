package middleware

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/google/go-cmp/cmp"

	"github.com/ollama/ollama/api"
)

func TestResponsesMiddlewareCustomToolHistory(t *testing.T) {
	for _, tt := range []struct {
		name    string
		history string
	}{
		{"function", `{"type":"function_call","call_id":"call_history_test","name":"read_note","arguments":"{\"input\":\"Read synthetic marker.\"}"},
		 {"type":"function_call_output","call_id":"call_history_test","output":"SYNTHETIC_TOOL_MARKER_731946"}`},
		{"custom", `{"type":"custom_tool_call","call_id":"call_history_test","name":"read_note","input":"Read synthetic marker."},
		 {"type":"custom_tool_call_output","call_id":"call_history_test","output":"SYNTHETIC_TOOL_MARKER_731946"}`},
	} {
		for _, stream := range []string{"false", "true"} {
			t.Run(tt.name+"/stream="+stream, func(t *testing.T) {
				var captured *api.ChatRequest
				router := gin.New()
				router.POST("/v1/responses", ResponsesMiddleware(), captureRequestMiddleware(&captured), func(c *gin.Context) {
					c.Status(http.StatusOK)
				})
				body := `{"model":"test","stream":` + stream + `,"input":[
					{"role":"user","content":"Read the synthetic marker."},
					` + tt.history + `,
					{"role":"user","content":"Reply with only the marker returned by the tool."}
				]}`
				request := httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(body))
				request.Header.Set("Content-Type", "application/json")
				response := httptest.NewRecorder()
				router.ServeHTTP(response, request)
				if response.Code != http.StatusOK {
					t.Fatalf("status = %d, body = %s", response.Code, response.Body.String())
				}
				if captured == nil {
					t.Fatal("request was not converted")
				}
				want := []api.Message{
					{Role: "user", Content: "Read the synthetic marker."},
					{Role: "assistant", ToolCalls: []api.ToolCall{{
						ID: "call_history_test", Function: api.ToolCallFunction{
							Name: "read_note", Arguments: testArgs(map[string]any{"input": "Read synthetic marker."}),
						},
					}}},
					{Role: "tool", ToolCallID: "call_history_test", Content: "SYNTHETIC_TOOL_MARKER_731946"},
					{Role: "user", Content: "Reply with only the marker returned by the tool."},
				}
				if diff := cmp.Diff(want, captured.Messages, argsComparer); diff != "" {
					t.Fatalf("tool history changed (-want +got):\n%s", diff)
				}
			})
		}
	}
}
