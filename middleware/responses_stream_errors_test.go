package middleware

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/openai"
)

func TestResponsesMiddlewareStreamFailure(t *testing.T) {
	toolCall := api.ToolCall{ID: "call_weather", Function: api.ToolCallFunction{Name: "weather", Arguments: testArgs(map[string]any{"city": "SF"})}}
	searchCall := api.ToolCall{ID: "call_search", Function: api.ToolCallFunction{Name: "web_search", Arguments: testArgs(map[string]any{"query": "weather"})}}
	for _, test := range []struct {
		name      string
		chunks    []api.ChatResponse
		webSearch bool
		empty     bool
		wantTypes []string
	}{
		{name: "before content", wantTypes: []string{}},
		{name: "empty error", empty: true, wantTypes: []string{}},
		{name: "reasoning", chunks: []api.ChatResponse{{Message: api.Message{Thinking: "partial thought"}}}, wantTypes: []string{"reasoning"}},
		{name: "text", chunks: []api.ChatResponse{{Message: api.Message{Content: "partial answer"}}}, wantTypes: []string{"message"}},
		{name: "reasoning then text", chunks: []api.ChatResponse{{Message: api.Message{Thinking: "partial thought"}}, {Message: api.Message{Content: "partial answer"}}}, wantTypes: []string{"reasoning", "message"}},
		{name: "text then tool", chunks: []api.ChatResponse{{Message: api.Message{Content: "checking weather"}}, {Message: api.Message{ToolCalls: []api.ToolCall{toolCall}}}}, wantTypes: []string{"message", "function_call"}},
		{name: "web search enabled", webSearch: true, chunks: []api.ChatResponse{{Message: api.Message{Content: "partial answer"}}}, wantTypes: []string{"message"}},
		{name: "web search pending", webSearch: true, chunks: []api.ChatResponse{{Message: api.Message{Content: "searching"}}, {Message: api.Message{ToolCalls: []api.ToolCall{searchCall}}}}, wantTypes: []string{"message"}},
	} {
		t.Run(test.name, func(t *testing.T) {
			gin.SetMode(gin.TestMode)
			router := gin.New()
			router.POST("/v1/responses", ResponsesMiddleware(), func(c *gin.Context) {
				encoder := json.NewEncoder(c.Writer)
				for _, chunk := range test.chunks {
					if err := encoder.Encode(chunk); err != nil {
						t.Fatal(err)
					}
				}
				message := "model runner exited"
				if test.empty {
					message = ""
				}
				if err := encoder.Encode(gin.H{"error": message}); err != nil {
					t.Fatal(err)
				}
				// Late chunks must not reopen a failed response or emit another terminal event.
				if !test.webSearch {
					for _, late := range []any{api.ChatResponse{Message: api.Message{Content: "late answer"}, Done: true}, gin.H{"error": "late error"}} {
						if err := encoder.Encode(late); err != nil {
							t.Fatal(err)
						}
					}
				}
			})
			request := map[string]any{"model": "test-model", "input": "hello", "stream": true}
			if test.webSearch {
				request["tools"] = []openai.ResponsesTool{{Type: "web_search"}}
			}
			body, err := json.Marshal(request)
			if err != nil {
				t.Fatal(err)
			}
			recorder := httptest.NewRecorder()
			req := httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(string(body)))
			req.Header.Set("Content-Type", "application/json")
			router.ServeHTTP(recorder, req)
			if recorder.Code != http.StatusOK || !strings.Contains(recorder.Header().Get("Content-Type"), "text/event-stream") {
				t.Fatalf("status=%d headers=%v body=%s", recorder.Code, recorder.Header(), recorder.Body.String())
			}
			message := "model runner exited"
			if test.empty {
				message = ""
			}
			response := assertFailedResponsesStream(t, recorder.Body.String(), "server_error", message, test.wantTypes)
			if len(test.chunks) > 0 && test.chunks[0].Message.Thinking != "" {
				if response.Output[0].Summary[0].Text != "partial thought" {
					t.Fatalf("reasoning lost: %#v", response.Output)
				}
			}
			for _, item := range response.Output {
				if item.Type == "message" {
					wantStatus := "incomplete"
					if test.name == "text then tool" {
						wantStatus = "completed"
					}
					if item.Status != wantStatus || item.Content[0].Text == "" {
						t.Fatalf("partial message not preserved: %#v", item)
					}
				}
			}
		})
	}
}

// Check the wire lifecycle as a client would: matching item IDs and indices,
// a single terminal event, and final output containing the streamed items.
func assertFailedResponsesStream(t *testing.T, body, code, message string, wantTypes []string) openai.ResponsesResponse {
	t.Helper()
	added := map[int]openai.ResponsesOutputItem{}
	done := map[int]openai.ResponsesOutputItem{}
	var response openai.ResponsesResponse
	blocks := strings.Split(strings.TrimSuffix(body, "\n\n"), "\n\n")
	for i, block := range blocks {
		lines := strings.Split(block, "\n")
		if len(lines) != 2 || !strings.HasPrefix(lines[0], "event: ") || !strings.HasPrefix(lines[1], "data: ") {
			t.Fatalf("malformed SSE block: %q", block)
		}
		var event struct {
			Type           string                     `json:"type"`
			SequenceNumber int                        `json:"sequence_number"`
			OutputIndex    int                        `json:"output_index"`
			Item           openai.ResponsesOutputItem `json:"item"`
			Response       openai.ResponsesResponse   `json:"response"`
		}
		if err := json.Unmarshal([]byte(strings.TrimPrefix(lines[1], "data: ")), &event); err != nil {
			t.Fatal(err)
		}
		if event.Type != strings.TrimPrefix(lines[0], "event: ") || event.SequenceNumber != i {
			t.Fatalf("event %d: type=%s sequence=%d", i, event.Type, event.SequenceNumber)
		}
		if i == 0 && event.Type != "response.created" || i == 1 && event.Type != "response.in_progress" {
			t.Fatalf("missing stream opening events: %s", body)
		}
		switch event.Type {
		case "response.output_item.added":
			if _, exists := added[event.OutputIndex]; exists {
				t.Fatalf("output index %d reused: %s", event.OutputIndex, body)
			}
			added[event.OutputIndex] = event.Item
		case "response.output_item.done":
			if opened, ok := added[event.OutputIndex]; !ok || opened.ID != event.Item.ID || opened.Type != event.Item.Type {
				t.Fatalf("closed an unknown output item: %#v", event)
			}
			if _, exists := done[event.OutputIndex]; exists {
				t.Fatalf("output item closed twice: %#v", event)
			}
			done[event.OutputIndex] = event.Item
		case "response.completed", "response.incomplete":
			t.Fatalf("failed generation reported success or truncation: %s", body)
		case "response.failed":
			if i != len(blocks)-1 {
				t.Fatalf("events after failure: %s", body)
			}
			response = event.Response
		}
	}
	if response.Status != "failed" || response.Error == nil || response.Error.Code != code || response.Error.Message != message || response.CompletedAt != nil {
		t.Fatalf("missing or incorrect terminal failure: %s", body)
	}
	if response.Output == nil || len(response.Output) != len(wantTypes) || len(added) != len(done) || len(done) != len(wantTypes) {
		t.Fatalf("output items lost or left open: added=%v done=%v response=%s", added, done, body)
	}
	for i, item := range response.Output {
		if item.Type != wantTypes[i] || !reflect.DeepEqual(item, done[i]) {
			t.Fatalf("terminal output[%d] disagrees with streamed output: %#v, done=%#v", i, item, done[i])
		}
	}
	return response
}

func TestWebSearchResponsesStreamFailure(t *testing.T) {
	for _, failSearch := range []bool{true, false} {
		name := "follow-up generation"
		if failSearch {
			name = "search execution"
		}
		t.Run(name, func(t *testing.T) {
			gin.SetMode(gin.TestMode)
			recorder := httptest.NewRecorder()
			ctx, _ := gin.CreateTestContext(recorder)
			stream := true
			request := openai.ResponsesRequest{Model: "test-model", Stream: &stream, Tools: []openai.ResponsesTool{{Type: "web_search"}}}
			inner := &ResponsesWriter{BaseWriter: BaseWriter{ResponseWriter: ctx.Writer}, converter: openai.NewResponsesStreamConverter("resp_test", "msg_test", request.Model, request), model: request.Model, stream: true, responseID: "resp_test", itemID: "msg_test", request: request}
			writer := &WebSearchResponsesWriter{
				BaseWriter: BaseWriter{ResponseWriter: ctx.Writer}, inner: inner, req: request,
				chat: &api.ChatRequest{Model: request.Model, Tools: api.Tools{openai.WebSearchFunctionTool()}},
				search: func(context.Context, string) (*api.WebSearchResponse, error) {
					if failSearch {
						return nil, errors.New("search failed")
					}
					return &api.WebSearchResponse{}, nil
				},
				followUpStream: func(_ context.Context, _ []api.Message, _ api.Tools, yield func(api.ChatResponse) error) error {
					if err := yield(api.ChatResponse{Message: api.Message{Content: "partial answer"}}); err != nil {
						return err
					}
					return errors.New("generation failed")
				},
			}
			data, err := json.Marshal(api.ChatResponse{Done: true, Message: api.Message{ToolCalls: []api.ToolCall{{ID: "call_search", Function: api.ToolCallFunction{Name: "web_search", Arguments: testArgs(map[string]any{"query": "weather"})}}}}})
			if err != nil {
				t.Fatal(err)
			}
			if _, err := writer.Write(data); err != nil {
				t.Fatal(err)
			}
			message := "generation failed"
			wantTypes := []string{"web_search_call", "message"}
			if failSearch {
				message = "search failed"
				wantTypes = []string{"web_search_call"}
			}
			response := assertFailedResponsesStream(t, recorder.Body.String(), "api_error", message, wantTypes)
			wantStatus := "completed"
			if failSearch {
				wantStatus = "failed"
			}
			if response.Output[0].Status != wantStatus {
				t.Fatalf("search status=%q, want %q", response.Output[0].Status, wantStatus)
			}
			if !failSearch && (response.Output[1].Status != "incomplete" || response.Output[1].Content[0].Text != "partial answer") {
				t.Fatalf("follow-up content lost: %#v", response.Output)
			}
		})
	}
}
