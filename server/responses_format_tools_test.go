package server

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/middleware"
)

func TestResponsesFormatWithToolInvestigation(t *testing.T) {
	gin.SetMode(gin.TestMode)
	const schema = `{"type":"object","properties":{"decision":{"enum":["hold"]}},"required":["decision"],"additionalProperties":false}`
	for _, parser := range []string{"qwen3.5", "qwen3-coder", "qwen3", "qwen3-vl-instruct"} {
		t.Run(parser, func(t *testing.T) {
			mock := &mockRunner{contextLength: 32768}
			s := newServerWithMockRunner(t, mock)
			s.defaultNumCtx = 32768
			createParserModel(t, s, "format-tools", parser)
			renderer := parser
			if parser == "qwen3" {
				renderer = "qwen3-vl-instruct"
			}
			stream := false
			w := createRequest(t, s.CreateHandler, api.CreateRequest{Model: "format-tools", From: "format-tools", Renderer: renderer, Stream: &stream})
			if w.Code != http.StatusOK {
				t.Fatalf("setting renderer: %d: %s", w.Code, w.Body.String())
			}
			router := gin.New()
			router.POST("/v1/responses", middleware.ResponsesMiddleware(), s.ChatHandler)

			call := 0
			mock.CompletionFn = func(_ context.Context, r llm.CompletionRequest, fn func(llm.CompletionResponse)) error {
				call++
				if !bytes.Equal(r.Format, []byte(schema)) {
					t.Errorf("completion %d lost the output schema: %s", call, r.Format)
				}
				if call <= 2 {
					if r.ToolCallOpen != "<tool_call>" || r.ToolCallClose != "</tool_call>" {
						t.Errorf("completion %d excludes tool calls from the format: %q %q", call, r.ToolCallOpen, r.ToolCallClose)
					}
				} else if r.ToolCallOpen != "" || r.ToolCallClose != "" {
					t.Error("tools must not be permitted when none were supplied")
				}
				if call == 1 {
					body := "<function=inspect><parameter=path>evidence.txt</parameter></function>"
					if parser == "qwen3" || parser == "qwen3-vl-instruct" {
						body = `{"name":"inspect","arguments":{"path":"evidence.txt"}}`
					}
					fn(llm.CompletionResponse{Content: "<tool_call>" + body + "</tool_call>"})
				} else {
					if call == 2 && !strings.Contains(r.Prompt, "INSPECTED_EVIDENCE") {
						t.Error("continuation lost the investigation result")
					}
					fn(llm.CompletionResponse{Content: `{"decision":"hold"}`})
				}
				fn(llm.CompletionResponse{Done: true, DoneReason: llm.DoneReasonStop})
				return nil
			}

			request := map[string]any{
				"model": "format-tools", "think": false, "stream": false,
				"input": []any{map[string]any{"role": "user", "content": "Inspect evidence.txt before deciding."}},
				"tools": []any{map[string]any{"type": "function", "name": "inspect", "parameters": map[string]any{
					"type": "object", "properties": map[string]any{"path": map[string]any{"type": "string"}}, "required": []string{"path"},
				}}},
				"text": map[string]any{"format": map[string]any{"type": "json_schema", "name": "decision", "schema": json.RawMessage(schema)}},
			}
			post := func() []map[string]any {
				t.Helper()
				body, err := json.Marshal(request)
				if err != nil {
					t.Fatal(err)
				}
				w := httptest.NewRecorder()
				router.ServeHTTP(w, httptest.NewRequest(http.MethodPost, "/v1/responses", bytes.NewReader(body)))
				if w.Code != http.StatusOK {
					t.Fatalf("Responses returned %d: %s", w.Code, w.Body.String())
				}
				var response struct {
					Output []map[string]any `json:"output"`
				}
				if err := json.Unmarshal(w.Body.Bytes(), &response); err != nil {
					t.Fatal(err)
				}
				return response.Output
			}
			output := post()
			if len(output) != 1 || output[0]["type"] != "function_call" || output[0]["name"] != "inspect" || output[0]["arguments"] != `{"path":"evidence.txt"}` {
				t.Fatalf("expected investigation tool call, got %v", output)
			}
			request["input"] = append(request["input"].([]any), output[0], map[string]any{
				"type": "function_call_output", "call_id": output[0]["call_id"], "output": "INSPECTED_EVIDENCE: hold",
			})
			for range 2 {
				output = post()
				if len(output) != 1 || output[0]["type"] != "message" {
					t.Fatalf("expected structured final text, got %v", output)
				}
				content := output[0]["content"].([]any)
				if len(content) != 1 || content[0].(map[string]any)["text"] != `{"decision":"hold"}` {
					t.Fatalf("final text changed: %v", content)
				}
				delete(request, "tools")
			}
		})
	}
}
