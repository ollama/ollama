package middleware

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/openai"
)

func assertResponsesAuthenticationError(t *testing.T, body string, stream bool, signinURL string) {
	t.Helper()
	data := []byte(body)
	if stream {
		data = nil
		for _, line := range strings.Split(body, "\n") {
			if !strings.HasPrefix(line, "data: ") {
				continue
			}
			var event struct {
				Type     string          `json:"type"`
				Response json.RawMessage `json:"response"`
			}
			if err := json.Unmarshal([]byte(strings.TrimPrefix(line, "data: ")), &event); err != nil {
				t.Fatal(err)
			}
			if event.Type == "response.completed" {
				t.Fatalf("authentication failure emitted completion: %s", body)
			}
			if event.Type == "response.failed" {
				if data != nil {
					t.Fatalf("multiple terminal failures: %s", body)
				}
				data = event.Response
			}
		}
	}
	var response openai.ResponsesErrorResponse
	if err := json.Unmarshal(data, &response); err != nil {
		t.Fatalf("invalid authentication error: %v: %s", err, body)
	}
	if response.Error.Message != "This model requires an Ollama account. Please sign in to Ollama to use this model." || response.Error.Code == nil || *response.Error.Code != "authentication_error" {
		t.Fatalf("unexpected authentication error: %s", body)
	}
	if response.SigninURL != signinURL {
		t.Fatalf("sign-in metadata = %q, want %q", response.SigninURL, signinURL)
	}
}

func TestResponsesMiddlewareNativeAuthenticationErrors(t *testing.T) {
	gin.SetMode(gin.TestMode)
	const signinURL = "https://ollama.com/connect?key=synthetic-secret"
	for _, test := range []struct {
		name                            string
		stream, webSearch, afterContent bool
	}{
		{"JSON", false, false, false},
		{"streaming request", true, false, false},
		{"web search JSON", false, true, false},
		{"web search before stream", true, true, false},
		{"web search during stream", true, true, true},
	} {
		t.Run(test.name, func(t *testing.T) {
			router := gin.New()
			router.POST("/v1/responses", ResponsesMiddleware(), func(c *gin.Context) {
				if test.afterContent {
					data, _ := json.Marshal(api.ChatResponse{Message: api.Message{Role: "assistant", Content: "partial answer"}})
					_, _ = c.Writer.Write(data)
					c.Writer.Flush()
				}
				c.JSON(http.StatusUnauthorized, gin.H{"error": "unauthorized", "signin_url": signinURL})
				if test.afterContent {
					_, _ = c.Writer.Write([]byte(`{"done":true}`))
				}
			})
			tools := "[]"
			if test.webSearch {
				tools = `[{"type":"web_search"}]`
			}
			body := fmt.Sprintf(`{"model":"test","input":"hello","stream":%t,"tools":%s}`, test.stream, tools)
			req := httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(body))
			req.Header.Set("Content-Type", "application/json")
			recorder := httptest.NewRecorder()
			router.ServeHTTP(recorder, req)
			streamError := test.stream && test.webSearch
			wantStatus := http.StatusUnauthorized
			if streamError {
				wantStatus = http.StatusOK
			}
			if recorder.Code != wantStatus {
				t.Fatalf("status = %d, want %d: %s", recorder.Code, wantStatus, recorder.Body.String())
			}
			assertResponsesAuthenticationError(t, recorder.Body.String(), streamError, signinURL)
			if test.afterContent && !strings.Contains(recorder.Body.String(), "partial answer") {
				t.Fatal("previous output was discarded")
			}
		})
	}
}

func TestResponsesMiddlewareDoesNotInferAuthenticationFromMessage(t *testing.T) {
	gin.SetMode(gin.TestMode)
	router := gin.New()
	router.POST("/v1/responses", ResponsesMiddleware(), func(c *gin.Context) {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "unauthorized"})
	})
	req := httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(`{"model":"test","input":"hello"}`))
	req.Header.Set("Content-Type", "application/json")
	recorder := httptest.NewRecorder()
	router.ServeHTTP(recorder, req)

	var response openai.ErrorResponse
	if err := json.Unmarshal(recorder.Body.Bytes(), &response); err != nil {
		t.Fatal(err)
	}
	if recorder.Code != http.StatusInternalServerError || response.Error.Type != "api_error" || response.Error.Message != "unauthorized" {
		t.Fatalf("non-authentication error was reclassified: status=%d body=%s", recorder.Code, recorder.Body.String())
	}
}

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
		{name: "text", chunks: []api.ChatResponse{{Message: api.Message{Content: "partial "}}, {Message: api.Message{Content: "answer"}}}, wantTypes: []string{"message"}},
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
			wantText := ""
			for _, chunk := range test.chunks {
				wantText += chunk.Message.Content
			}
			for _, item := range response.Output {
				if item.Type == "message" {
					wantStatus := "incomplete"
					if test.name == "text then tool" {
						wantStatus = "completed"
					}
					if item.Status != wantStatus || len(item.Content) != 1 || item.Content[0].Text != wantText {
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
