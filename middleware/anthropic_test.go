package middleware

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"

	"github.com/ollama/ollama/anthropic"
	"github.com/ollama/ollama/api"
)

func captureAnthropicRequest(capturedRequest any) gin.HandlerFunc {
	return func(c *gin.Context) {
		bodyBytes, _ := io.ReadAll(c.Request.Body)
		c.Request.Body = io.NopCloser(bytes.NewReader(bodyBytes))
		_ = json.Unmarshal(bodyBytes, capturedRequest)
		c.Next()
	}
}

// testProps creates ToolPropertiesMap from a map (convenience function for tests)
func testProps(m map[string]api.ToolProperty) *api.ToolPropertiesMap {
	props := api.NewToolPropertiesMap()
	for k, v := range m {
		props.Set(k, v)
	}
	return props
}

func TestAnthropicMessagesMiddleware(t *testing.T) {
	type testCase struct {
		name string
		body string
		req  api.ChatRequest
		err  anthropic.ErrorResponse
	}

	var capturedRequest *api.ChatRequest
	stream := true

	testCases := []testCase{
		{
			name: "basic message",
			body: `{
				"model": "test-model",
				"max_tokens": 1024,
				"messages": [
					{"role": "user", "content": "Hello"}
				]
			}`,
			req: api.ChatRequest{
				Model: "test-model",
				Messages: []api.Message{
					{Role: "user", Content: "Hello"},
				},
				Options: map[string]any{"num_predict": 1024},
				Stream:  &False,
			},
		},
		{
			name: "with system prompt",
			body: `{
				"model": "test-model",
				"max_tokens": 1024,
				"system": "You are helpful.",
				"messages": [
					{"role": "user", "content": "Hello"}
				]
			}`,
			req: api.ChatRequest{
				Model: "test-model",
				Messages: []api.Message{
					{Role: "system", Content: "You are helpful."},
					{Role: "user", Content: "Hello"},
				},
				Options: map[string]any{"num_predict": 1024},
				Stream:  &False,
			},
		},
		{
			name: "with options",
			body: `{
				"model": "test-model",
				"max_tokens": 2048,
				"temperature": 0.7,
				"top_p": 0.9,
				"top_k": 40,
				"stop_sequences": ["\n", "END"],
				"messages": [
					{"role": "user", "content": "Hello"}
				]
			}`,
			req: api.ChatRequest{
				Model: "test-model",
				Messages: []api.Message{
					{Role: "user", Content: "Hello"},
				},
				Options: map[string]any{
					"num_predict": 2048,
					"temperature": 0.7,
					"top_p":       0.9,
					"top_k":       40,
					"stop":        []string{"\n", "END"},
				},
				Stream: &False,
			},
		},
		{
			name: "streaming",
			body: `{
				"model": "test-model",
				"max_tokens": 1024,
				"stream": true,
				"messages": [
					{"role": "user", "content": "Hello"}
				]
			}`,
			req: api.ChatRequest{
				Model: "test-model",
				Messages: []api.Message{
					{Role: "user", Content: "Hello"},
				},
				Options: map[string]any{"num_predict": 1024},
				Stream:  &stream,
			},
		},
		{
			name: "with tools",
			body: `{
				"model": "test-model",
				"max_tokens": 1024,
				"messages": [
					{"role": "user", "content": "What's the weather?"}
				],
				"tools": [{
					"name": "get_weather",
					"description": "Get current weather",
					"input_schema": {
						"type": "object",
						"properties": {
							"location": {"type": "string"}
						},
						"required": ["location"]
					}
				}]
			}`,
			req: api.ChatRequest{
				Model: "test-model",
				Messages: []api.Message{
					{Role: "user", Content: "What's the weather?"},
				},
				Tools: []api.Tool{
					{
						Type: "function",
						Function: api.ToolFunction{
							Name:        "get_weather",
							Description: "Get current weather",
							Parameters: api.ToolFunctionParameters{
								Type:     "object",
								Required: []string{"location"},
								Properties: testProps(map[string]api.ToolProperty{
									"location": {Type: api.PropertyType{"string"}},
								}),
							},
						},
					},
				},
				Options: map[string]any{"num_predict": 1024},
				Stream:  &False,
			},
		},
		{
			name: "with tool result",
			body: `{
				"model": "test-model",
				"max_tokens": 1024,
				"messages": [
					{"role": "user", "content": "What's the weather?"},
					{"role": "assistant", "content": [
						{"type": "tool_use", "id": "call_123", "name": "get_weather", "input": {"location": "Paris"}}
					]},
					{"role": "user", "content": [
						{"type": "tool_result", "tool_use_id": "call_123", "content": "Sunny, 22°C"}
					]}
				]
			}`,
			req: api.ChatRequest{
				Model: "test-model",
				Messages: []api.Message{
					{Role: "user", Content: "What's the weather?"},
					{
						Role: "assistant",
						ToolCalls: []api.ToolCall{
							{
								ID: "call_123",
								Function: api.ToolCallFunction{
									Name:      "get_weather",
									Arguments: testArgs(map[string]any{"location": "Paris"}),
								},
							},
						},
					},
					{Role: "tool", Content: "Sunny, 22°C", ToolCallID: "call_123"},
				},
				Options: map[string]any{"num_predict": 1024},
				Stream:  &False,
			},
		},
		{
			name: "with thinking enabled",
			body: `{
				"model": "test-model",
				"max_tokens": 1024,
				"thinking": {"type": "enabled", "budget_tokens": 1000},
				"messages": [
					{"role": "user", "content": "Hello"}
				]
			}`,
			req: api.ChatRequest{
				Model: "test-model",
				Messages: []api.Message{
					{Role: "user", Content: "Hello"},
				},
				Options: map[string]any{"num_predict": 1024},
				Stream:  &False,
				Think:   &api.ThinkValue{Value: true},
			},
		},
		{
			name: "missing model error",
			body: `{
				"max_tokens": 1024,
				"messages": [
					{"role": "user", "content": "Hello"}
				]
			}`,
			err: anthropic.ErrorResponse{
				Type: "error",
				Error: anthropic.Error{
					Type:    "invalid_request_error",
					Message: "model is required",
				},
			},
		},
		{
			name: "missing max_tokens error",
			body: `{
				"model": "test-model",
				"messages": [
					{"role": "user", "content": "Hello"}
				]
			}`,
			err: anthropic.ErrorResponse{
				Type: "error",
				Error: anthropic.Error{
					Type:    "invalid_request_error",
					Message: "max_tokens is required and must be positive",
				},
			},
		},
		{
			name: "missing messages error",
			body: `{
				"model": "test-model",
				"max_tokens": 1024
			}`,
			err: anthropic.ErrorResponse{
				Type: "error",
				Error: anthropic.Error{
					Type:    "invalid_request_error",
					Message: "messages is required",
				},
			},
		},
		{
			name: "tool_use missing id error",
			body: `{
				"model": "test-model",
				"max_tokens": 1024,
				"messages": [
					{"role": "assistant", "content": [
						{"type": "tool_use", "name": "test"}
					]}
				]
			}`,
			err: anthropic.ErrorResponse{
				Type: "error",
				Error: anthropic.Error{
					Type:    "invalid_request_error",
					Message: "tool_use block missing required 'id' field",
				},
			},
		},
	}

	endpoint := func(c *gin.Context) {
		c.Status(http.StatusOK)
	}

	gin.SetMode(gin.TestMode)
	router := gin.New()
	router.Use(AnthropicMessagesMiddleware(), captureAnthropicRequest(&capturedRequest))
	router.Handle(http.MethodPost, "/v1/messages", endpoint)

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			req, _ := http.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(tc.body))
			req.Header.Set("Content-Type", "application/json")

			defer func() { capturedRequest = nil }()

			resp := httptest.NewRecorder()
			router.ServeHTTP(resp, req)

			if tc.err.Type != "" {
				// Expect error
				if resp.Code == http.StatusOK {
					t.Fatalf("expected error response, got 200 OK")
				}
				var errResp anthropic.ErrorResponse
				if err := json.Unmarshal(resp.Body.Bytes(), &errResp); err != nil {
					t.Fatalf("failed to unmarshal error: %v", err)
				}
				if errResp.Type != tc.err.Type {
					t.Errorf("expected error type %q, got %q", tc.err.Type, errResp.Type)
				}
				if errResp.Error.Type != tc.err.Error.Type {
					t.Errorf("expected error.type %q, got %q", tc.err.Error.Type, errResp.Error.Type)
				}
				if errResp.Error.Message != tc.err.Error.Message {
					t.Errorf("expected error.message %q, got %q", tc.err.Error.Message, errResp.Error.Message)
				}
				return
			}

			if resp.Code != http.StatusOK {
				t.Fatalf("unexpected status code: %d, body: %s", resp.Code, resp.Body.String())
			}

			if capturedRequest == nil {
				t.Fatal("request was not captured")
			}

			// Compare relevant fields
			if capturedRequest.Model != tc.req.Model {
				t.Errorf("model mismatch: got %q, want %q", capturedRequest.Model, tc.req.Model)
			}

			if diff := cmp.Diff(tc.req.Messages, capturedRequest.Messages,
				cmpopts.IgnoreUnexported(api.ToolCallFunctionArguments{}, api.ToolPropertiesMap{})); diff != "" {
				t.Errorf("messages mismatch (-want +got):\n%s", diff)
			}

			if tc.req.Stream != nil && capturedRequest.Stream != nil {
				if *tc.req.Stream != *capturedRequest.Stream {
					t.Errorf("stream mismatch: got %v, want %v", *capturedRequest.Stream, *tc.req.Stream)
				}
			}

			if tc.req.Think != nil {
				if capturedRequest.Think == nil {
					t.Error("expected Think to be set")
				} else if capturedRequest.Think.Value != tc.req.Think.Value {
					t.Errorf("Think mismatch: got %v, want %v", capturedRequest.Think.Value, tc.req.Think.Value)
				}
			}
		})
	}
}

func TestAnthropicMessagesMiddleware_Headers(t *testing.T) {
	gin.SetMode(gin.TestMode)

	t.Run("streaming sets correct headers", func(t *testing.T) {
		router := gin.New()
		router.Use(AnthropicMessagesMiddleware())
		router.POST("/v1/messages", func(c *gin.Context) {
			// Check headers were set
			if c.Writer.Header().Get("Content-Type") != "text/event-stream" {
				t.Errorf("expected Content-Type text/event-stream, got %q", c.Writer.Header().Get("Content-Type"))
			}
			if c.Writer.Header().Get("Cache-Control") != "no-cache" {
				t.Errorf("expected Cache-Control no-cache, got %q", c.Writer.Header().Get("Cache-Control"))
			}
			c.Status(http.StatusOK)
		})

		body := `{"model": "test", "max_tokens": 100, "stream": true, "messages": [{"role": "user", "content": "Hi"}]}`
		req, _ := http.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(body))
		req.Header.Set("Content-Type", "application/json")

		resp := httptest.NewRecorder()
		router.ServeHTTP(resp, req)
	})
}

func TestAnthropicMessagesMiddleware_InvalidJSON(t *testing.T) {
	gin.SetMode(gin.TestMode)
	router := gin.New()
	router.Use(AnthropicMessagesMiddleware())
	router.POST("/v1/messages", func(c *gin.Context) {
		c.Status(http.StatusOK)
	})

	req, _ := http.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(`{invalid json`))
	req.Header.Set("Content-Type", "application/json")

	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	if resp.Code != http.StatusBadRequest {
		t.Errorf("expected status 400, got %d", resp.Code)
	}

	var errResp anthropic.ErrorResponse
	if err := json.Unmarshal(resp.Body.Bytes(), &errResp); err != nil {
		t.Fatalf("failed to unmarshal error: %v", err)
	}

	if errResp.Type != "error" {
		t.Errorf("expected type 'error', got %q", errResp.Type)
	}
	if errResp.Error.Type != "invalid_request_error" {
		t.Errorf("expected error type 'invalid_request_error', got %q", errResp.Error.Type)
	}
}

func TestAnthropicWriter_NonStreaming(t *testing.T) {
	gin.SetMode(gin.TestMode)

	router := gin.New()
	router.Use(AnthropicMessagesMiddleware())
	router.POST("/v1/messages", func(c *gin.Context) {
		// Simulate Ollama response
		resp := api.ChatResponse{
			Model: "test-model",
			Message: api.Message{
				Role:    "assistant",
				Content: "Hello there!",
			},
			Done:       true,
			DoneReason: "stop",
			Metrics: api.Metrics{
				PromptEvalCount: 10,
				EvalCount:       5,
			},
		}
		data, _ := json.Marshal(resp)
		c.Writer.WriteHeader(http.StatusOK)
		_, _ = c.Writer.Write(data)
	})

	body := `{"model": "test-model", "max_tokens": 100, "messages": [{"role": "user", "content": "Hi"}]}`
	req, _ := http.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")

	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	if resp.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d", resp.Code)
	}

	var result anthropic.MessagesResponse
	if err := json.Unmarshal(resp.Body.Bytes(), &result); err != nil {
		t.Fatalf("failed to unmarshal response: %v", err)
	}

	if result.Type != "message" {
		t.Errorf("expected type 'message', got %q", result.Type)
	}
	if result.Role != "assistant" {
		t.Errorf("expected role 'assistant', got %q", result.Role)
	}
	if len(result.Content) != 1 {
		t.Fatalf("expected 1 content block, got %d", len(result.Content))
	}
	if result.Content[0].Text == nil || *result.Content[0].Text != "Hello there!" {
		t.Errorf("expected text 'Hello there!', got %v", result.Content[0].Text)
	}
	if result.StopReason != "end_turn" {
		t.Errorf("expected stop_reason 'end_turn', got %q", result.StopReason)
	}
	if result.Usage.InputTokens != 10 {
		t.Errorf("expected input_tokens 10, got %d", result.Usage.InputTokens)
	}
	if result.Usage.OutputTokens != 5 {
		t.Errorf("expected output_tokens 5, got %d", result.Usage.OutputTokens)
	}
}

// TestAnthropicWriter_ErrorFromRoutes tests error handling when routes.go sends
// gin.H{"error": "message"} without a StatusCode field (which is the common case)
func TestAnthropicWriter_ErrorFromRoutes(t *testing.T) {
	gin.SetMode(gin.TestMode)

	tests := []struct {
		name          string
		statusCode    int
		errorPayload  any
		wantErrorType string
		wantMessage   string
	}{
		// routes.go sends errors without StatusCode in JSON, so we must use HTTP status
		{
			name:          "404 with gin.H error (model not found)",
			statusCode:    http.StatusNotFound,
			errorPayload:  gin.H{"error": "model 'nonexistent' not found"},
			wantErrorType: "not_found_error",
			wantMessage:   "model 'nonexistent' not found",
		},
		{
			name:          "400 with gin.H error (bad request)",
			statusCode:    http.StatusBadRequest,
			errorPayload:  gin.H{"error": "model is required"},
			wantErrorType: "invalid_request_error",
			wantMessage:   "model is required",
		},
		{
			name:          "500 with gin.H error (internal error)",
			statusCode:    http.StatusInternalServerError,
			errorPayload:  gin.H{"error": "something went wrong"},
			wantErrorType: "api_error",
			wantMessage:   "something went wrong",
		},
		{
			name:       "404 with api.StatusError",
			statusCode: http.StatusNotFound,
			errorPayload: api.StatusError{
				StatusCode:   http.StatusNotFound,
				ErrorMessage: "model not found via StatusError",
			},
			wantErrorType: "not_found_error",
			wantMessage:   "model not found via StatusError",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			router := gin.New()
			router.Use(AnthropicMessagesMiddleware())
			router.POST("/v1/messages", func(c *gin.Context) {
				// Simulate what routes.go does - set status and write error JSON
				data, _ := json.Marshal(tt.errorPayload)
				c.Writer.WriteHeader(tt.statusCode)
				_, _ = c.Writer.Write(data)
			})

			body := `{"model": "test-model", "max_tokens": 100, "messages": [{"role": "user", "content": "Hi"}]}`
			req, _ := http.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(body))
			req.Header.Set("Content-Type", "application/json")

			resp := httptest.NewRecorder()
			router.ServeHTTP(resp, req)

			if resp.Code != tt.statusCode {
				t.Errorf("expected status %d, got %d", tt.statusCode, resp.Code)
			}

			var errResp anthropic.ErrorResponse
			if err := json.Unmarshal(resp.Body.Bytes(), &errResp); err != nil {
				t.Fatalf("failed to unmarshal error response: %v\nbody: %s", err, resp.Body.String())
			}

			if errResp.Type != "error" {
				t.Errorf("expected type 'error', got %q", errResp.Type)
			}
			if errResp.Error.Type != tt.wantErrorType {
				t.Errorf("expected error type %q, got %q", tt.wantErrorType, errResp.Error.Type)
			}
			if errResp.Error.Message != tt.wantMessage {
				t.Errorf("expected message %q, got %q", tt.wantMessage, errResp.Error.Message)
			}
		})
	}
}

func TestAnthropicMessagesMiddleware_SetsRelaxThinkingFlag(t *testing.T) {
	gin.SetMode(gin.TestMode)

	var flagSet bool
	router := gin.New()
	router.Use(AnthropicMessagesMiddleware())
	router.POST("/v1/messages", func(c *gin.Context) {
		_, flagSet = c.Get("relax_thinking")
		c.Status(http.StatusOK)
	})

	body := `{"model": "test-model", "max_tokens": 100, "messages": [{"role": "user", "content": "Hi"}]}`
	req, _ := http.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")

	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	if !flagSet {
		t.Error("expected relax_thinking flag to be set in context")
	}
}

// Web Search Tests

func TestHasWebSearchTool(t *testing.T) {
	tests := []struct {
		name     string
		tools    []anthropic.Tool
		expected bool
	}{
		{
			name:     "no tools",
			tools:    nil,
			expected: false,
		},
		{
			name: "regular tool only",
			tools: []anthropic.Tool{
				{Type: "custom", Name: "get_weather"},
			},
			expected: false,
		},
		{
			name: "web search tool",
			tools: []anthropic.Tool{
				{Type: "web_search_20250305", Name: "web_search"},
			},
			expected: true,
		},
		{
			name: "mixed tools",
			tools: []anthropic.Tool{
				{Type: "custom", Name: "get_weather"},
				{Type: "web_search_20250305", Name: "web_search"},
			},
			expected: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := hasWebSearchTool(tt.tools)
			if result != tt.expected {
				t.Errorf("expected %v, got %v", tt.expected, result)
			}
		})
	}
}

func TestExtractQueryFromToolCall(t *testing.T) {
	tests := []struct {
		name     string
		tc       *api.ToolCall
		expected string
	}{
		{
			name: "valid query",
			tc: &api.ToolCall{
				Function: api.ToolCallFunction{
					Name:      "web_search",
					Arguments: makeArgs("query", "test search"),
				},
			},
			expected: "test search",
		},
		{
			name: "empty arguments",
			tc: &api.ToolCall{
				Function: api.ToolCallFunction{
					Name: "web_search",
				},
			},
			expected: "",
		},
		{
			name: "no query key",
			tc: &api.ToolCall{
				Function: api.ToolCallFunction{
					Name:      "web_search",
					Arguments: makeArgs("other", "value"),
				},
			},
			expected: "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := extractQueryFromToolCall(tt.tc)
			if result != tt.expected {
				t.Errorf("expected %q, got %q", tt.expected, result)
			}
		})
	}
}

// makeArgs is a test helper that creates ToolCallFunctionArguments
func makeArgs(key string, value any) api.ToolCallFunctionArguments {
	args := api.NewToolCallFunctionArguments()
	args.Set(key, value)
	return args
}

// --- Web Search Integration Tests ---

// TestWebSearchServerToolUseID tests the ID derivation logic.
func TestWebSearchServerToolUseID(t *testing.T) {
	tests := []struct {
		msgID    string
		expected string
	}{
		{"msg_abc123", "srvtoolu_abc123"},
		{"msg_", "srvtoolu_"},
		{"nomsgprefix", "srvtoolu_nomsgprefix"},
	}
	for _, tt := range tests {
		got := serverToolUseID(tt.msgID)
		if got != tt.expected {
			t.Errorf("serverToolUseID(%q) = %q, want %q", tt.msgID, got, tt.expected)
		}
	}
}

// TestWebSearchNoWebSearchTool verifies that when there is no web_search tool,
// requests pass through to the normal AnthropicWriter without interception.
func TestWebSearchNoWebSearchTool(t *testing.T) {
	gin.SetMode(gin.TestMode)

	router := gin.New()
	router.Use(AnthropicMessagesMiddleware())
	router.POST("/v1/messages", func(c *gin.Context) {
		resp := api.ChatResponse{
			Model: "test-model",
			Message: api.Message{
				Role:    "assistant",
				Content: "Normal response",
			},
			Done:       true,
			DoneReason: "stop",
			Metrics:    api.Metrics{PromptEvalCount: 10, EvalCount: 5},
		}
		data, _ := json.Marshal(resp)
		c.Writer.WriteHeader(http.StatusOK)
		_, _ = c.Writer.Write(data)
	})

	body := `{"model":"test-model","max_tokens":100,"messages":[{"role":"user","content":"Hello"}]}`
	req, _ := http.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")

	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	if resp.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", resp.Code, resp.Body.String())
	}

	var result anthropic.MessagesResponse
	if err := json.Unmarshal(resp.Body.Bytes(), &result); err != nil {
		t.Fatalf("unmarshal error: %v", err)
	}

	if result.Type != "message" {
		t.Errorf("expected type 'message', got %q", result.Type)
	}
	if len(result.Content) != 1 || result.Content[0].Type != "text" {
		t.Fatalf("expected single text block, got %d blocks", len(result.Content))
	}
	if *result.Content[0].Text != "Normal response" {
		t.Errorf("expected text 'Normal response', got %q", *result.Content[0].Text)
	}
}

// TestWebSearchToolPresent_ModelDoesNotCallIt_NonStreaming verifies that when
// the web_search tool is present but the model does not call it, the response
// passes through normally (non-streaming case).
func TestWebSearchToolPresent_ModelDoesNotCallIt_NonStreaming(t *testing.T) {
	gin.SetMode(gin.TestMode)

	router := gin.New()
	router.Use(AnthropicMessagesMiddleware())
	router.POST("/v1/messages", func(c *gin.Context) {
		resp := api.ChatResponse{
			Model: "test-model",
			Message: api.Message{
				Role:    "assistant",
				Content: "I can answer that without searching.",
			},
			Done:       true,
			DoneReason: "stop",
			Metrics:    api.Metrics{PromptEvalCount: 12, EvalCount: 8},
		}
		data, _ := json.Marshal(resp)
		c.Writer.WriteHeader(http.StatusOK)
		_, _ = c.Writer.Write(data)
	})

	body := `{
		"model":"test-model:cloud",
		"max_tokens":100,
		"messages":[{"role":"user","content":"What is 2+2?"}],
		"tools":[{"type":"web_search_20250305","name":"web_search"}]
	}`
	req, _ := http.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")

	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	if resp.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", resp.Code, resp.Body.String())
	}

	var result anthropic.MessagesResponse
	if err := json.Unmarshal(resp.Body.Bytes(), &result); err != nil {
		t.Fatalf("unmarshal error: %v", err)
	}

	if result.Type != "message" {
		t.Errorf("expected type 'message', got %q", result.Type)
	}
	if len(result.Content) != 1 || result.Content[0].Type != "text" {
		t.Fatalf("expected single text block, got %+v", result.Content)
	}
	if *result.Content[0].Text != "I can answer that without searching." {
		t.Errorf("unexpected text: %q", *result.Content[0].Text)
	}
	if result.StopReason != "end_turn" {
		t.Errorf("expected stop_reason 'end_turn', got %q", result.StopReason)
	}
}

// TestWebSearchToolPresent_ModelDoesNotCallIt_Streaming verifies the streaming
// pass-through case when the model does not invoke web_search.
func TestWebSearchToolPresent_ModelDoesNotCallIt_Streaming(t *testing.T) {
	gin.SetMode(gin.TestMode)

	router := gin.New()
	router.Use(AnthropicMessagesMiddleware())
	router.POST("/v1/messages", func(c *gin.Context) {
		// Simulate streaming: two partial chunks then a final chunk
		chunks := []api.ChatResponse{
			{
				Model:   "test-model",
				Message: api.Message{Role: "assistant", Content: "Hello "},
				Done:    false,
			},
			{
				Model:   "test-model",
				Message: api.Message{Role: "assistant", Content: "world"},
				Done:    false,
			},
			{
				Model:      "test-model",
				Message:    api.Message{Role: "assistant", Content: ""},
				Done:       true,
				DoneReason: "stop",
				Metrics:    api.Metrics{PromptEvalCount: 10, EvalCount: 5},
			},
		}
		c.Writer.WriteHeader(http.StatusOK)
		for _, chunk := range chunks {
			data, _ := json.Marshal(chunk)
			_, _ = c.Writer.Write(data)
		}
	})

	body := `{
		"model":"test-model:cloud",
		"max_tokens":100,
		"stream":true,
		"messages":[{"role":"user","content":"Hi"}],
		"tools":[{"type":"web_search_20250305","name":"web_search"}]
	}`
	req, _ := http.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")

	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	if resp.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", resp.Code, resp.Body.String())
	}

	// Parse SSE events
	events := parseSSEEvents(t, resp.Body.String())

	// Should have standard streaming event flow
	if len(events) == 0 {
		t.Fatal("expected SSE events, got none")
	}

	// First event should be message_start
	if events[0].event != "message_start" {
		t.Errorf("first event should be message_start, got %q", events[0].event)
	}

	// Should have content_block_start for text
	hasTextStart := false
	hasTextDelta := false
	hasMessageStop := false
	for _, e := range events {
		if e.event == "content_block_start" {
			var cbs anthropic.ContentBlockStartEvent
			if err := json.Unmarshal([]byte(e.data), &cbs); err == nil {
				if cbs.ContentBlock.Type == "text" {
					hasTextStart = true
				}
			}
		}
		if e.event == "content_block_delta" {
			var cbd anthropic.ContentBlockDeltaEvent
			if err := json.Unmarshal([]byte(e.data), &cbd); err == nil {
				if cbd.Delta.Type == "text_delta" {
					hasTextDelta = true
				}
			}
		}
		if e.event == "message_stop" {
			hasMessageStop = true
		}
	}
	if !hasTextStart {
		t.Error("expected content_block_start with text type")
	}
	if !hasTextDelta {
		t.Error("expected content_block_delta with text_delta")
	}
	if !hasMessageStop {
		t.Error("expected message_stop event")
	}
}

// TestWebSearchToolPresent_ModelCallsIt_NonStreaming tests the full web search flow
// in non-streaming mode. It mocks the followup /api/chat call using a local HTTP server.
func TestWebSearchToolPresent_ModelCallsIt_NonStreaming(t *testing.T) {
	gin.SetMode(gin.TestMode)

	// Create a mock Ollama server that responds to the followup /api/chat call
	followupServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		resp := api.ChatResponse{
			Model: "test-model",
			Message: api.Message{
				Role:    "assistant",
				Content: "Based on my search, the answer is 42.",
			},
			Done:       true,
			DoneReason: "stop",
			Metrics:    api.Metrics{PromptEvalCount: 50, EvalCount: 20},
		}
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer followupServer.Close()

	// Set OLLAMA_HOST to our mock server so the followup call goes there
	t.Setenv("OLLAMA_HOST", followupServer.URL)

	// Also mock the web search API
	searchServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		resp := anthropic.OllamaWebSearchResponse{
			Results: []anthropic.OllamaWebSearchResult{
				{Title: "Test Result", URL: "https://example.com/result", Content: "Some content"},
			},
		}
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer searchServer.Close()

	// Point DoWebSearch at our mock search server
	originalEndpoint := anthropic.WebSearchEndpoint
	anthropic.WebSearchEndpoint = searchServer.URL
	defer func() { anthropic.WebSearchEndpoint = originalEndpoint }()

	router := gin.New()
	router.Use(AnthropicMessagesMiddleware())
	router.POST("/v1/messages", func(c *gin.Context) {
		resp := api.ChatResponse{
			Model: "test-model",
			Message: api.Message{
				Role: "assistant",
				ToolCalls: []api.ToolCall{
					{
						ID: "call_ws_001",
						Function: api.ToolCallFunction{
							Name:      "web_search",
							Arguments: makeArgs("query", "meaning of life"),
						},
					},
				},
			},
			Done:       true,
			DoneReason: "stop",
			Metrics:    api.Metrics{PromptEvalCount: 15, EvalCount: 3},
		}
		data, _ := json.Marshal(resp)
		c.Writer.WriteHeader(http.StatusOK)
		_, _ = c.Writer.Write(data)
	})

	body := `{
		"model":"test-model:cloud",
		"max_tokens":100,
		"messages":[{"role":"user","content":"What is the meaning of life?"}],
		"tools":[{"type":"web_search_20250305","name":"web_search"}]
	}`
	req, _ := http.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")

	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	if resp.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", resp.Code, resp.Body.String())
	}

	var result anthropic.MessagesResponse
	if err := json.Unmarshal(resp.Body.Bytes(), &result); err != nil {
		t.Fatalf("unmarshal error: %v\nbody: %s", err, resp.Body.String())
	}

	if result.Type != "message" {
		t.Errorf("expected type 'message', got %q", result.Type)
	}
	if result.Role != "assistant" {
		t.Errorf("expected role 'assistant', got %q", result.Role)
	}

	// Should have 3 blocks: server_tool_use + web_search_tool_result + text
	if len(result.Content) != 3 {
		t.Fatalf("expected 3 content blocks, got %d: %+v", len(result.Content), result.Content)
	}

	if result.Content[0].Type != "server_tool_use" {
		t.Errorf("expected first block type 'server_tool_use', got %q", result.Content[0].Type)
	}
	if result.Content[0].Name != "web_search" {
		t.Errorf("expected name 'web_search', got %q", result.Content[0].Name)
	}

	if result.Content[1].Type != "web_search_tool_result" {
		t.Errorf("expected second block type 'web_search_tool_result', got %q", result.Content[1].Type)
	}
	if result.Content[1].ToolUseID != result.Content[0].ID {
		t.Errorf("tool_use_id mismatch: %q != %q", result.Content[1].ToolUseID, result.Content[0].ID)
	}

	if result.Content[2].Type != "text" {
		t.Errorf("expected third block type 'text', got %q", result.Content[2].Type)
	}
	if result.Content[2].Text == nil || *result.Content[2].Text == "" {
		t.Error("expected non-empty text in third block")
	}

	if result.StopReason != "end_turn" {
		t.Errorf("expected stop_reason 'end_turn', got %q", result.StopReason)
	}
}

// TestWebSearchToolPresent_ModelCallsIt_Streaming tests the streaming SSE output
// when the model calls web_search with mocked search and followup endpoints.
func TestWebSearchToolPresent_ModelCallsIt_Streaming(t *testing.T) {
	gin.SetMode(gin.TestMode)

	// Mock followup /api/chat server
	followupServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		resp := api.ChatResponse{
			Model:      "test-model",
			Message:    api.Message{Role: "assistant", Content: "Here are the latest news."},
			Done:       true,
			DoneReason: "stop",
			Metrics:    api.Metrics{PromptEvalCount: 40, EvalCount: 15},
		}
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer followupServer.Close()
	t.Setenv("OLLAMA_HOST", followupServer.URL)

	// Mock web search API
	searchServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		resp := anthropic.OllamaWebSearchResponse{
			Results: []anthropic.OllamaWebSearchResult{
				{Title: "News Result", URL: "https://example.com/news", Content: "Breaking news"},
			},
		}
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer searchServer.Close()
	originalEndpoint := anthropic.WebSearchEndpoint
	anthropic.WebSearchEndpoint = searchServer.URL
	defer func() { anthropic.WebSearchEndpoint = originalEndpoint }()

	router := gin.New()
	router.Use(AnthropicMessagesMiddleware())
	router.POST("/v1/messages", func(c *gin.Context) {
		// Simulate buffered streaming: non-final chunk then final with tool call
		chunks := []api.ChatResponse{
			{
				Model:   "test-model",
				Message: api.Message{Role: "assistant"},
				Done:    false,
			},
			{
				Model: "test-model",
				Message: api.Message{
					Role: "assistant",
					ToolCalls: []api.ToolCall{
						{
							ID: "call_ws_002",
							Function: api.ToolCallFunction{
								Name:      "web_search",
								Arguments: makeArgs("query", "latest news"),
							},
						},
					},
				},
				Done:       true,
				DoneReason: "stop",
				Metrics:    api.Metrics{PromptEvalCount: 10, EvalCount: 2},
			},
		}
		c.Writer.WriteHeader(http.StatusOK)
		for _, chunk := range chunks {
			data, _ := json.Marshal(chunk)
			_, _ = c.Writer.Write(data)
		}
	})

	body := `{
		"model":"test-model:cloud",
		"max_tokens":100,
		"stream":true,
		"messages":[{"role":"user","content":"What is the latest news?"}],
		"tools":[{"type":"web_search_20250305","name":"web_search"}]
	}`
	req, _ := http.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")

	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	if resp.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", resp.Code, resp.Body.String())
	}

	events := parseSSEEvents(t, resp.Body.String())

	// Success path: 10 events (3 blocks: server_tool_use, web_search_tool_result, text with delta)
	expectedEventTypes := []string{
		"message_start",
		"content_block_start", // server_tool_use
		"content_block_stop",
		"content_block_start", // web_search_tool_result
		"content_block_stop",
		"content_block_start", // text (empty)
		"content_block_delta", // text_delta with actual content
		"content_block_stop",
		"message_delta",
		"message_stop",
	}

	if len(events) != len(expectedEventTypes) {
		t.Fatalf("expected %d events, got %d.\nEvents: %v", len(expectedEventTypes), len(events), eventNames(events))
	}

	for i, expected := range expectedEventTypes {
		if events[i].event != expected {
			t.Errorf("event[%d]: expected %q, got %q", i, expected, events[i].event)
		}
	}

	// Verify text delta has the followup model's content
	var textDelta anthropic.ContentBlockDeltaEvent
	if err := json.Unmarshal([]byte(events[6].data), &textDelta); err != nil {
		t.Fatalf("failed to parse text delta: %v", err)
	}
	if textDelta.Delta.Type != "text_delta" {
		t.Errorf("expected delta type 'text_delta', got %q", textDelta.Delta.Type)
	}
	if textDelta.Delta.Text != "Here are the latest news." {
		t.Errorf("expected followup text, got %q", textDelta.Delta.Text)
	}
}

// TestWebSearchStreamResponse tests the streamResponse method directly by constructing
// a WebSearchAnthropicWriter and calling streamResponse with a known response.
func TestWebSearchStreamResponse(t *testing.T) {
	gin.SetMode(gin.TestMode)

	text := "Here is the answer."

	response := anthropic.MessagesResponse{
		ID:    "msg_test123",
		Type:  "message",
		Role:  "assistant",
		Model: "test-model",
		Content: []anthropic.ContentBlock{
			{
				Type:  "server_tool_use",
				ID:    "srvtoolu_test123",
				Name:  "web_search",
				Input: map[string]any{"query": "test query"},
			},
			{
				Type:      "web_search_tool_result",
				ToolUseID: "srvtoolu_test123",
				Content: []anthropic.WebSearchResult{
					{Type: "web_search_result", URL: "https://example.com", Title: "Example"},
				},
			},
			{
				Type: "text",
				Text: &text,
			},
		},
		StopReason: "end_turn",
		Usage:      anthropic.Usage{InputTokens: 20, OutputTokens: 10},
	}

	rec := httptest.NewRecorder()
	ginCtx, _ := gin.CreateTestContext(rec)

	innerWriter := &AnthropicWriter{
		BaseWriter: BaseWriter{ResponseWriter: ginCtx.Writer},
		stream:     true,
		id:         "msg_test123",
	}
	wsWriter := &WebSearchAnthropicWriter{
		BaseWriter: BaseWriter{ResponseWriter: ginCtx.Writer},
		inner:      innerWriter,
		stream:     true,
		req:        anthropic.MessagesRequest{Model: "test-model"},
	}

	if err := wsWriter.streamResponse(response); err != nil {
		t.Fatalf("streamResponse error: %v", err)
	}

	events := parseSSEEvents(t, rec.Body.String())

	// Verify full event sequence
	expectedEventTypes := []string{
		"message_start",
		"content_block_start", // server_tool_use (index 0)
		"content_block_stop",  // index 0
		"content_block_start", // web_search_tool_result (index 1)
		"content_block_stop",  // index 1
		"content_block_start", // text (index 2)
		"content_block_delta", // text_delta
		"content_block_stop",  // index 2
		"message_delta",
		"message_stop",
	}

	if len(events) != len(expectedEventTypes) {
		t.Fatalf("expected %d events, got %d.\nEvents: %v", len(expectedEventTypes), len(events), eventNames(events))
	}

	for i, expected := range expectedEventTypes {
		if events[i].event != expected {
			t.Errorf("event[%d]: expected %q, got %q", i, expected, events[i].event)
		}
	}

	// Verify message_start content
	var msgStart anthropic.MessageStartEvent
	if err := json.Unmarshal([]byte(events[0].data), &msgStart); err != nil {
		t.Fatalf("failed to parse message_start: %v", err)
	}
	if msgStart.Message.ID != "msg_test123" {
		t.Errorf("expected message ID 'msg_test123', got %q", msgStart.Message.ID)
	}
	if msgStart.Message.Role != "assistant" {
		t.Errorf("expected role 'assistant', got %q", msgStart.Message.Role)
	}
	if len(msgStart.Message.Content) != 0 {
		t.Errorf("expected empty content in message_start, got %d blocks", len(msgStart.Message.Content))
	}

	// Verify content_block_start for server_tool_use (event index 1)
	var toolStart anthropic.ContentBlockStartEvent
	if err := json.Unmarshal([]byte(events[1].data), &toolStart); err != nil {
		t.Fatalf("failed to parse server_tool_use start: %v", err)
	}
	if toolStart.Index != 0 {
		t.Errorf("expected index 0, got %d", toolStart.Index)
	}
	if toolStart.ContentBlock.Type != "server_tool_use" {
		t.Errorf("expected type 'server_tool_use', got %q", toolStart.ContentBlock.Type)
	}
	if toolStart.ContentBlock.ID != "srvtoolu_test123" {
		t.Errorf("expected ID 'srvtoolu_test123', got %q", toolStart.ContentBlock.ID)
	}

	// Verify content_block_start for web_search_tool_result (event index 3)
	var searchStart anthropic.ContentBlockStartEvent
	if err := json.Unmarshal([]byte(events[3].data), &searchStart); err != nil {
		t.Fatalf("failed to parse web_search_tool_result start: %v", err)
	}
	if searchStart.Index != 1 {
		t.Errorf("expected index 1, got %d", searchStart.Index)
	}
	if searchStart.ContentBlock.Type != "web_search_tool_result" {
		t.Errorf("expected type 'web_search_tool_result', got %q", searchStart.ContentBlock.Type)
	}

	// Verify text block: content_block_start (event index 5)
	var textStart anthropic.ContentBlockStartEvent
	if err := json.Unmarshal([]byte(events[5].data), &textStart); err != nil {
		t.Fatalf("failed to parse text start: %v", err)
	}
	if textStart.Index != 2 {
		t.Errorf("expected index 2, got %d", textStart.Index)
	}
	if textStart.ContentBlock.Type != "text" {
		t.Errorf("expected type 'text', got %q", textStart.ContentBlock.Type)
	}
	// Text in start should be empty
	if textStart.ContentBlock.Text == nil || *textStart.ContentBlock.Text != "" {
		t.Errorf("expected empty text in content_block_start, got %v", textStart.ContentBlock.Text)
	}

	// Verify text delta (event index 6)
	var textDelta anthropic.ContentBlockDeltaEvent
	if err := json.Unmarshal([]byte(events[6].data), &textDelta); err != nil {
		t.Fatalf("failed to parse text delta: %v", err)
	}
	if textDelta.Index != 2 {
		t.Errorf("expected index 2, got %d", textDelta.Index)
	}
	if textDelta.Delta.Type != "text_delta" {
		t.Errorf("expected delta type 'text_delta', got %q", textDelta.Delta.Type)
	}
	if textDelta.Delta.Text != "Here is the answer." {
		t.Errorf("expected delta text 'Here is the answer.', got %q", textDelta.Delta.Text)
	}

	// Verify message_delta (event index 8)
	var msgDelta anthropic.MessageDeltaEvent
	if err := json.Unmarshal([]byte(events[8].data), &msgDelta); err != nil {
		t.Fatalf("failed to parse message_delta: %v", err)
	}
	if msgDelta.Delta.StopReason != "end_turn" {
		t.Errorf("expected stop_reason 'end_turn', got %q", msgDelta.Delta.StopReason)
	}
	if msgDelta.Usage.OutputTokens != 10 {
		t.Errorf("expected output_tokens 10, got %d", msgDelta.Usage.OutputTokens)
	}
}

// TestWebSearchSendError_NonStreaming tests sendError produces correct response shape.
func TestWebSearchSendError_NonStreaming(t *testing.T) {
	gin.SetMode(gin.TestMode)

	rec := httptest.NewRecorder()
	ginCtx, _ := gin.CreateTestContext(rec)

	innerWriter := &AnthropicWriter{
		BaseWriter: BaseWriter{ResponseWriter: ginCtx.Writer},
		stream:     false,
		id:         "msg_err001",
	}
	wsWriter := &WebSearchAnthropicWriter{
		BaseWriter: BaseWriter{ResponseWriter: ginCtx.Writer},
		inner:      innerWriter,
		stream:     false,
		req:        anthropic.MessagesRequest{Model: "test-model"},
	}

	if err := wsWriter.sendError("unavailable", "test query"); err != nil {
		t.Fatalf("sendError error: %v", err)
	}

	var result anthropic.MessagesResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &result); err != nil {
		t.Fatalf("unmarshal error: %v\nbody: %s", err, rec.Body.String())
	}

	if result.Type != "message" {
		t.Errorf("expected type 'message', got %q", result.Type)
	}
	if result.ID != "msg_err001" {
		t.Errorf("expected ID 'msg_err001', got %q", result.ID)
	}

	// Should have exactly 2 blocks: server_tool_use + web_search_tool_result
	if len(result.Content) != 2 {
		t.Fatalf("expected 2 content blocks, got %d", len(result.Content))
	}

	// Block 0: server_tool_use
	if result.Content[0].Type != "server_tool_use" {
		t.Errorf("expected 'server_tool_use', got %q", result.Content[0].Type)
	}
	expectedToolID := "srvtoolu_err001"
	if result.Content[0].ID != expectedToolID {
		t.Errorf("expected ID %q, got %q", expectedToolID, result.Content[0].ID)
	}
	if result.Content[0].Name != "web_search" {
		t.Errorf("expected name 'web_search', got %q", result.Content[0].Name)
	}
	// Verify input contains the query
	inputMap, ok := result.Content[0].Input.(map[string]any)
	if !ok {
		t.Fatalf("expected Input to be map, got %T", result.Content[0].Input)
	}
	if inputMap["query"] != "test query" {
		t.Errorf("expected query 'test query', got %v", inputMap["query"])
	}

	// Block 1: web_search_tool_result with error
	if result.Content[1].Type != "web_search_tool_result" {
		t.Errorf("expected 'web_search_tool_result', got %q", result.Content[1].Type)
	}
	if result.Content[1].ToolUseID != expectedToolID {
		t.Errorf("expected tool_use_id %q, got %q", expectedToolID, result.Content[1].ToolUseID)
	}

	// The Content field should be a WebSearchToolResultError
	contentJSON, _ := json.Marshal(result.Content[1].Content)
	var errContent anthropic.WebSearchToolResultError
	if err := json.Unmarshal(contentJSON, &errContent); err != nil {
		t.Fatalf("failed to parse error content: %v\nraw: %s", err, string(contentJSON))
	}
	if errContent.Type != "web_search_tool_result_error" {
		t.Errorf("expected error type 'web_search_tool_result_error', got %q", errContent.Type)
	}
	if errContent.ErrorCode != "unavailable" {
		t.Errorf("expected error_code 'unavailable', got %q", errContent.ErrorCode)
	}

	if result.StopReason != "end_turn" {
		t.Errorf("expected stop_reason 'end_turn', got %q", result.StopReason)
	}
}

// TestWebSearchSendError_Streaming tests sendError in streaming mode produces proper SSE.
func TestWebSearchSendError_Streaming(t *testing.T) {
	gin.SetMode(gin.TestMode)

	rec := httptest.NewRecorder()
	ginCtx, _ := gin.CreateTestContext(rec)

	innerWriter := &AnthropicWriter{
		BaseWriter: BaseWriter{ResponseWriter: ginCtx.Writer},
		stream:     true,
		id:         "msg_err002",
	}
	wsWriter := &WebSearchAnthropicWriter{
		BaseWriter: BaseWriter{ResponseWriter: ginCtx.Writer},
		inner:      innerWriter,
		stream:     true,
		req:        anthropic.MessagesRequest{Model: "test-model"},
	}

	if err := wsWriter.sendError("invalid_request", "bad query"); err != nil {
		t.Fatalf("sendError error: %v", err)
	}

	events := parseSSEEvents(t, rec.Body.String())

	// Error response has 2 blocks: server_tool_use + web_search_tool_result
	// Expected events: message_start,
	//   content_block_start(server_tool_use), content_block_stop,
	//   content_block_start(web_search_tool_result), content_block_stop,
	//   message_delta, message_stop
	expectedEventTypes := []string{
		"message_start",
		"content_block_start",
		"content_block_stop",
		"content_block_start",
		"content_block_stop",
		"message_delta",
		"message_stop",
	}

	if len(events) != len(expectedEventTypes) {
		t.Fatalf("expected %d events, got %d.\nEvents: %v", len(expectedEventTypes), len(events), eventNames(events))
	}

	for i, expected := range expectedEventTypes {
		if events[i].event != expected {
			t.Errorf("event[%d]: expected %q, got %q", i, expected, events[i].event)
		}
	}

	// Verify the server_tool_use block
	var toolStart anthropic.ContentBlockStartEvent
	if err := json.Unmarshal([]byte(events[1].data), &toolStart); err != nil {
		t.Fatalf("failed to parse server_tool_use start: %v", err)
	}
	if toolStart.ContentBlock.Type != "server_tool_use" {
		t.Errorf("expected 'server_tool_use', got %q", toolStart.ContentBlock.Type)
	}

	// Verify the web_search_tool_result block
	var resultStart anthropic.ContentBlockStartEvent
	if err := json.Unmarshal([]byte(events[3].data), &resultStart); err != nil {
		t.Fatalf("failed to parse web_search_tool_result start: %v", err)
	}
	if resultStart.ContentBlock.Type != "web_search_tool_result" {
		t.Errorf("expected 'web_search_tool_result', got %q", resultStart.ContentBlock.Type)
	}
}

// TestWebSearchSendError_EmptyQuery tests sendError with an empty query.
func TestWebSearchSendError_EmptyQuery(t *testing.T) {
	gin.SetMode(gin.TestMode)

	rec := httptest.NewRecorder()
	ginCtx, _ := gin.CreateTestContext(rec)

	innerWriter := &AnthropicWriter{
		BaseWriter: BaseWriter{ResponseWriter: ginCtx.Writer},
		stream:     false,
		id:         "msg_empty001",
	}
	wsWriter := &WebSearchAnthropicWriter{
		BaseWriter: BaseWriter{ResponseWriter: ginCtx.Writer},
		inner:      innerWriter,
		stream:     false,
		req:        anthropic.MessagesRequest{Model: "test-model"},
	}

	if err := wsWriter.sendError("invalid_request", ""); err != nil {
		t.Fatalf("sendError error: %v", err)
	}

	var result anthropic.MessagesResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &result); err != nil {
		t.Fatalf("unmarshal error: %v", err)
	}

	if len(result.Content) != 2 {
		t.Fatalf("expected 2 content blocks, got %d", len(result.Content))
	}

	// Verify the input has empty query
	inputMap, ok := result.Content[0].Input.(map[string]any)
	if !ok {
		t.Fatalf("expected Input to be map, got %T", result.Content[0].Input)
	}
	if inputMap["query"] != "" {
		t.Errorf("expected empty query, got %v", inputMap["query"])
	}
}

// --- SSE parsing helpers ---

type sseEvent struct {
	event string
	data  string
}

// parseSSEEvents parses Server-Sent Events from a string.
func parseSSEEvents(t *testing.T, body string) []sseEvent {
	t.Helper()
	var events []sseEvent
	var currentEvent string
	var currentData strings.Builder

	for _, line := range strings.Split(body, "\n") {
		if strings.HasPrefix(line, "event: ") {
			currentEvent = strings.TrimPrefix(line, "event: ")
		} else if strings.HasPrefix(line, "data: ") {
			currentData.WriteString(strings.TrimPrefix(line, "data: "))
		} else if line == "" && currentEvent != "" {
			events = append(events, sseEvent{event: currentEvent, data: currentData.String()})
			currentEvent = ""
			currentData.Reset()
		}
	}
	return events
}

// eventNames returns a list of event type names for debugging.
func eventNames(events []sseEvent) []string {
	names := make([]string, len(events))
	for i, e := range events {
		names[i] = e.event
	}
	return names
}

// TestWebSearchCloudModelGating tests that web_search tool is rejected for non-cloud models.
func TestWebSearchCloudModelGating(t *testing.T) {
	gin.SetMode(gin.TestMode)

	t.Run("local model rejected", func(t *testing.T) {
		handlerCalled := false
		router := gin.New()
		router.Use(AnthropicMessagesMiddleware())
		router.POST("/v1/messages", func(c *gin.Context) {
			handlerCalled = true
		})

		body := `{"model":"llama3.2","max_tokens":100,"messages":[{"role":"user","content":"hello"}],"tools":[{"type":"web_search_20250305","name":"web_search"}]}`
		req, _ := http.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(body))
		req.Header.Set("Content-Type", "application/json")
		resp := httptest.NewRecorder()
		router.ServeHTTP(resp, req)

		if resp.Code != http.StatusBadRequest {
			t.Errorf("expected 400, got %d: %s", resp.Code, resp.Body.String())
		}
		if handlerCalled {
			t.Error("handler should not be called for non-cloud model")
		}
		var errResp anthropic.ErrorResponse
		if err := json.Unmarshal(resp.Body.Bytes(), &errResp); err != nil {
			t.Fatalf("failed to parse error response: %v", err)
		}
		if !strings.Contains(errResp.Error.Message, "cloud models") {
			t.Errorf("expected error about cloud models, got: %q", errResp.Error.Message)
		}
	})

	t.Run("local model with tag rejected", func(t *testing.T) {
		router := gin.New()
		router.Use(AnthropicMessagesMiddleware())
		router.POST("/v1/messages", func(c *gin.Context) {
			t.Error("handler should not be called for non-cloud model")
		})

		body := `{"model":"llama3.2:latest","max_tokens":100,"messages":[{"role":"user","content":"hello"}],"tools":[{"type":"web_search_20250305","name":"web_search"}]}`
		req, _ := http.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(body))
		req.Header.Set("Content-Type", "application/json")
		resp := httptest.NewRecorder()
		router.ServeHTTP(resp, req)

		if resp.Code != http.StatusBadRequest {
			t.Errorf("expected 400, got %d: %s", resp.Code, resp.Body.String())
		}
	})

	t.Run("model ending in cloud without cloud suffix rejected", func(t *testing.T) {
		router := gin.New()
		router.Use(AnthropicMessagesMiddleware())
		router.POST("/v1/messages", func(c *gin.Context) {
			t.Error("handler should not be called for invalid cloud suffix model")
		})

		body := `{"model":"notreallycloud","max_tokens":100,"messages":[{"role":"user","content":"hello"}],"tools":[{"type":"web_search_20250305","name":"web_search"}]}`
		req, _ := http.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(body))
		req.Header.Set("Content-Type", "application/json")
		resp := httptest.NewRecorder()
		router.ServeHTTP(resp, req)

		if resp.Code != http.StatusBadRequest {
			t.Errorf("expected 400, got %d: %s", resp.Code, resp.Body.String())
		}
	})

	t.Run("cloud model with size tag allowed", func(t *testing.T) {
		handlerCalled := false
		router := gin.New()
		router.Use(AnthropicMessagesMiddleware())
		router.POST("/v1/messages", func(c *gin.Context) {
			handlerCalled = true
			resp := api.ChatResponse{
				Model:      "gpt-oss:120b",
				Message:    api.Message{Role: "assistant", Content: "hello"},
				Done:       true,
				DoneReason: "stop",
				Metrics:    api.Metrics{PromptEvalCount: 10, EvalCount: 5},
			}
			data, _ := json.Marshal(resp)
			c.Writer.WriteHeader(http.StatusOK)
			_, _ = c.Writer.Write(data)
		})

		body := `{"model":"gpt-oss:120b-cloud","max_tokens":100,"messages":[{"role":"user","content":"hello"}],"tools":[{"type":"web_search_20250305","name":"web_search"}]}`
		req, _ := http.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(body))
		req.Header.Set("Content-Type", "application/json")
		resp := httptest.NewRecorder()
		router.ServeHTTP(resp, req)

		if !handlerCalled {
			t.Error("handler should be called for cloud model")
		}
		if resp.Code != http.StatusOK {
			t.Errorf("expected 200, got %d: %s", resp.Code, resp.Body.String())
		}
	})

	t.Run("cloud model allowed", func(t *testing.T) {
		handlerCalled := false
		router := gin.New()
		router.Use(AnthropicMessagesMiddleware())
		router.POST("/v1/messages", func(c *gin.Context) {
			handlerCalled = true
			// Return a simple response so the middleware doesn't error
			resp := api.ChatResponse{
				Model:      "kimi-k2.5",
				Message:    api.Message{Role: "assistant", Content: "hello"},
				Done:       true,
				DoneReason: "stop",
				Metrics:    api.Metrics{PromptEvalCount: 10, EvalCount: 5},
			}
			data, _ := json.Marshal(resp)
			c.Writer.WriteHeader(http.StatusOK)
			_, _ = c.Writer.Write(data)
		})

		body := `{"model":"kimi-k2.5:cloud","max_tokens":100,"messages":[{"role":"user","content":"hello"}],"tools":[{"type":"web_search_20250305","name":"web_search"}]}`
		req, _ := http.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(body))
		req.Header.Set("Content-Type", "application/json")
		resp := httptest.NewRecorder()
		router.ServeHTTP(resp, req)

		if !handlerCalled {
			t.Error("handler should be called for cloud model")
		}
		if resp.Code != http.StatusOK {
			t.Errorf("expected 200, got %d: %s", resp.Code, resp.Body.String())
		}
	})

	t.Run("cloud disabled blocks web search even for cloud model", func(t *testing.T) {
		t.Setenv("OLLAMA_NO_CLOUD", "1")

		handlerCalled := false
		router := gin.New()
		router.Use(AnthropicMessagesMiddleware())
		router.POST("/v1/messages", func(c *gin.Context) {
			handlerCalled = true
		})

		body := `{"model":"kimi-k2.5:cloud","max_tokens":100,"messages":[{"role":"user","content":"hello"}],"tools":[{"type":"web_search_20250305","name":"web_search"}]}`
		req, _ := http.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(body))
		req.Header.Set("Content-Type", "application/json")
		resp := httptest.NewRecorder()
		router.ServeHTTP(resp, req)

		if resp.Code != http.StatusForbidden {
			t.Fatalf("expected 403, got %d: %s", resp.Code, resp.Body.String())
		}
		if handlerCalled {
			t.Fatal("handler should not be called when cloud is disabled")
		}

		var errResp anthropic.ErrorResponse
		if err := json.Unmarshal(resp.Body.Bytes(), &errResp); err != nil {
			t.Fatalf("failed to parse error response: %v", err)
		}
		if !strings.Contains(errResp.Error.Message, "ollama cloud is disabled") {
			t.Fatalf("expected cloud disabled error, got: %q", errResp.Error.Message)
		}
	})
}

// TestWebSearchSearchAPIError tests that a failing search API returns a proper error response.
func TestWebSearchSearchAPIError(t *testing.T) {
	gin.SetMode(gin.TestMode)

	// Mock search server that returns 500
	searchServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "internal error", http.StatusInternalServerError)
	}))
	defer searchServer.Close()
	originalEndpoint := anthropic.WebSearchEndpoint
	anthropic.WebSearchEndpoint = searchServer.URL
	defer func() { anthropic.WebSearchEndpoint = originalEndpoint }()

	router := gin.New()
	router.Use(AnthropicMessagesMiddleware())
	router.POST("/v1/messages", func(c *gin.Context) {
		resp := api.ChatResponse{
			Model: "test-model",
			Message: api.Message{
				Role: "assistant",
				ToolCalls: []api.ToolCall{
					{
						ID: "call_err",
						Function: api.ToolCallFunction{
							Name:      "web_search",
							Arguments: makeArgs("query", "test"),
						},
					},
				},
			},
			Done:       true,
			DoneReason: "stop",
			Metrics:    api.Metrics{PromptEvalCount: 10, EvalCount: 2},
		}
		data, _ := json.Marshal(resp)
		c.Writer.WriteHeader(http.StatusOK)
		_, _ = c.Writer.Write(data)
	})

	body := `{
		"model":"test-model:cloud",
		"max_tokens":100,
		"messages":[{"role":"user","content":"test"}],
		"tools":[{"type":"web_search_20250305","name":"web_search"}]
	}`
	req, _ := http.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")

	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	if resp.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", resp.Code, resp.Body.String())
	}

	var result anthropic.MessagesResponse
	if err := json.Unmarshal(resp.Body.Bytes(), &result); err != nil {
		t.Fatalf("unmarshal error: %v", err)
	}

	// Error response: server_tool_use + web_search_tool_result with error
	if len(result.Content) != 2 {
		t.Fatalf("expected 2 content blocks for error, got %d", len(result.Content))
	}
	if result.Content[0].Type != "server_tool_use" {
		t.Errorf("expected 'server_tool_use', got %q", result.Content[0].Type)
	}
	if result.Content[1].Type != "web_search_tool_result" {
		t.Errorf("expected 'web_search_tool_result', got %q", result.Content[1].Type)
	}
}

func TestWebSearchStreamingImmediateTakeover(t *testing.T) {
	gin.SetMode(gin.TestMode)

	followupServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		resp := api.ChatResponse{
			Model:      "test-model",
			Message:    api.Message{Role: "assistant", Content: "After search."},
			Done:       true,
			DoneReason: "stop",
			Metrics:    api.Metrics{PromptEvalCount: 20, EvalCount: 10},
		}
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer followupServer.Close()
	t.Setenv("OLLAMA_HOST", followupServer.URL)

	searchServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		resp := anthropic.OllamaWebSearchResponse{
			Results: []anthropic.OllamaWebSearchResult{
				{Title: "Result", URL: "https://example.com", Content: "content"},
			},
		}
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer searchServer.Close()
	originalEndpoint := anthropic.WebSearchEndpoint
	anthropic.WebSearchEndpoint = searchServer.URL
	defer func() { anthropic.WebSearchEndpoint = originalEndpoint }()

	router := gin.New()
	router.Use(AnthropicMessagesMiddleware())
	router.POST("/v1/messages", func(c *gin.Context) {
		chunks := []api.ChatResponse{
			{
				Model:   "test-model",
				Message: api.Message{Role: "assistant", Content: "Preface "},
				Done:    false,
			},
			{
				Model: "test-model",
				Message: api.Message{
					Role: "assistant",
					ToolCalls: []api.ToolCall{
						{
							ID: "call_ws_stream_1",
							Function: api.ToolCallFunction{
								Name:      "web_search",
								Arguments: makeArgs("query", "latest updates"),
							},
						},
					},
				},
				Done: false,
			},
			{
				Model:   "test-model",
				Message: api.Message{Role: "assistant", Content: "ignored chunk"},
				Done:    false,
			},
			{
				Model:      "test-model",
				Message:    api.Message{Role: "assistant"},
				Done:       true,
				DoneReason: "stop",
				Metrics:    api.Metrics{PromptEvalCount: 9, EvalCount: 4},
			},
		}
		c.Writer.WriteHeader(http.StatusOK)
		for _, chunk := range chunks {
			data, _ := json.Marshal(chunk)
			_, _ = c.Writer.Write(data)
		}
	})

	body := `{
		"model":"test-model:cloud",
		"max_tokens":100,
		"stream":true,
		"messages":[{"role":"user","content":"Find updates"}],
		"tools":[{"type":"web_search_20250305","name":"web_search"}]
	}`
	req, _ := http.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")

	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	if resp.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", resp.Code, resp.Body.String())
	}

	events := parseSSEEvents(t, resp.Body.String())
	if countEventsByName(events, "message_start") != 1 {
		t.Fatalf("expected exactly one message_start, got %d", countEventsByName(events, "message_start"))
	}
	if countEventsByName(events, "message_stop") != 1 {
		t.Fatalf("expected exactly one message_stop, got %d", countEventsByName(events, "message_stop"))
	}

	textDeltas := collectTextDeltas(t, events)
	if !containsString(textDeltas, "Preface ") {
		t.Fatalf("expected passthrough text delta, got %v", textDeltas)
	}
	if !containsString(textDeltas, "After search.") {
		t.Fatalf("expected post-search text delta, got %v", textDeltas)
	}
	if containsString(textDeltas, "ignored chunk") {
		t.Fatalf("unexpected text from chunks after takeover: %v", textDeltas)
	}
}

func TestWebSearchMixedToolCallsPreferWebSearch(t *testing.T) {
	gin.SetMode(gin.TestMode)

	followupServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		resp := api.ChatResponse{
			Model:      "test-model",
			Message:    api.Message{Role: "assistant", Content: "Search answer."},
			Done:       true,
			DoneReason: "stop",
			Metrics:    api.Metrics{PromptEvalCount: 11, EvalCount: 6},
		}
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer followupServer.Close()
	t.Setenv("OLLAMA_HOST", followupServer.URL)

	searchServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		resp := anthropic.OllamaWebSearchResponse{
			Results: []anthropic.OllamaWebSearchResult{
				{Title: "Result", URL: "https://example.com", Content: "content"},
			},
		}
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer searchServer.Close()
	originalEndpoint := anthropic.WebSearchEndpoint
	anthropic.WebSearchEndpoint = searchServer.URL
	defer func() { anthropic.WebSearchEndpoint = originalEndpoint }()

	router := gin.New()
	router.Use(AnthropicMessagesMiddleware())
	router.POST("/v1/messages", func(c *gin.Context) {
		resp := api.ChatResponse{
			Model: "test-model",
			Message: api.Message{
				Role: "assistant",
				ToolCalls: []api.ToolCall{
					{
						ID: "call_other",
						Function: api.ToolCallFunction{
							Name:      "get_weather",
							Arguments: makeArgs("location", "SF"),
						},
					},
					{
						ID: "call_ws_mixed",
						Function: api.ToolCallFunction{
							Name:      "web_search",
							Arguments: makeArgs("query", "latest weather"),
						},
					},
				},
			},
			Done:       true,
			DoneReason: "stop",
			Metrics:    api.Metrics{PromptEvalCount: 10, EvalCount: 2},
		}
		data, _ := json.Marshal(resp)
		c.Writer.WriteHeader(http.StatusOK)
		_, _ = c.Writer.Write(data)
	})

	body := `{
		"model":"test-model:cloud",
		"max_tokens":100,
		"messages":[{"role":"user","content":"Weather?"}],
		"tools":[
			{"type":"web_search_20250305","name":"web_search"},
			{"type":"custom","name":"get_weather","input_schema":{"type":"object"}}
		]
	}`
	req, _ := http.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")

	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	if resp.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", resp.Code, resp.Body.String())
	}

	var result anthropic.MessagesResponse
	if err := json.Unmarshal(resp.Body.Bytes(), &result); err != nil {
		t.Fatalf("unmarshal error: %v", err)
	}

	if len(result.Content) < 3 {
		t.Fatalf("expected at least 3 blocks, got %d", len(result.Content))
	}
	if result.Content[0].Type != "server_tool_use" {
		t.Fatalf("expected server_tool_use first, got %q", result.Content[0].Type)
	}
	if result.Content[1].Type != "web_search_tool_result" {
		t.Fatalf("expected web_search_tool_result second, got %q", result.Content[1].Type)
	}

	for _, block := range result.Content {
		if block.Type == "tool_use" && block.Name == "get_weather" {
			t.Fatalf("did not expect get_weather tool_use in mixed web_search-preferred path: %+v", result.Content)
		}
	}
}

func TestWebSearchFollowupClientToolStopReasonToolUse(t *testing.T) {
	gin.SetMode(gin.TestMode)

	followupServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		resp := api.ChatResponse{
			Model: "test-model",
			Message: api.Message{
				Role: "assistant",
				ToolCalls: []api.ToolCall{
					{
						ID: "call_weather_final",
						Function: api.ToolCallFunction{
							Name:      "get_weather",
							Arguments: makeArgs("location", "New York"),
						},
					},
				},
			},
			Done:       true,
			DoneReason: "stop",
			Metrics:    api.Metrics{PromptEvalCount: 25, EvalCount: 7},
		}
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer followupServer.Close()
	t.Setenv("OLLAMA_HOST", followupServer.URL)

	searchServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		resp := anthropic.OllamaWebSearchResponse{
			Results: []anthropic.OllamaWebSearchResult{
				{Title: "Result", URL: "https://example.com", Content: "content"},
			},
		}
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer searchServer.Close()
	originalEndpoint := anthropic.WebSearchEndpoint
	anthropic.WebSearchEndpoint = searchServer.URL
	defer func() { anthropic.WebSearchEndpoint = originalEndpoint }()

	router := gin.New()
	router.Use(AnthropicMessagesMiddleware())
	router.POST("/v1/messages", func(c *gin.Context) {
		resp := api.ChatResponse{
			Model: "test-model",
			Message: api.Message{
				Role: "assistant",
				ToolCalls: []api.ToolCall{
					{
						ID: "call_ws_tool_use",
						Function: api.ToolCallFunction{
							Name:      "web_search",
							Arguments: makeArgs("query", "forecast"),
						},
					},
				},
			},
			Done:       true,
			DoneReason: "stop",
			Metrics:    api.Metrics{PromptEvalCount: 15, EvalCount: 3},
		}
		data, _ := json.Marshal(resp)
		c.Writer.WriteHeader(http.StatusOK)
		_, _ = c.Writer.Write(data)
	})

	body := `{
		"model":"test-model:cloud",
		"max_tokens":100,
		"messages":[{"role":"user","content":"Do I need an umbrella?"}],
		"tools":[
			{"type":"web_search_20250305","name":"web_search"},
			{"type":"custom","name":"get_weather","input_schema":{"type":"object"}}
		]
	}`
	req, _ := http.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")

	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	if resp.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", resp.Code, resp.Body.String())
	}

	var result anthropic.MessagesResponse
	if err := json.Unmarshal(resp.Body.Bytes(), &result); err != nil {
		t.Fatalf("unmarshal error: %v", err)
	}

	if result.StopReason != "tool_use" {
		t.Fatalf("expected stop_reason tool_use, got %q", result.StopReason)
	}
	if len(result.Content) < 3 {
		t.Fatalf("expected server blocks + tool_use, got %d blocks", len(result.Content))
	}
	last := result.Content[len(result.Content)-1]
	if last.Type != "tool_use" {
		t.Fatalf("expected final block tool_use, got %q", last.Type)
	}
	if last.Name != "get_weather" {
		t.Fatalf("expected final tool name get_weather, got %q", last.Name)
	}
	if result.Usage.InputTokens != 40 || result.Usage.OutputTokens != 10 {
		t.Fatalf("unexpected aggregated usage: %+v", result.Usage)
	}
}

func TestWebSearchMultiIterationLoop(t *testing.T) {
	gin.SetMode(gin.TestMode)

	followupCall := 0
	followupServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		followupCall++
		switch followupCall {
		case 1:
			resp := api.ChatResponse{
				Model: "test-model",
				Message: api.Message{
					Role: "assistant",
					ToolCalls: []api.ToolCall{
						{
							ID: "call_ws_2",
							Function: api.ToolCallFunction{
								Name:      "web_search",
								Arguments: makeArgs("query", "loop query 2"),
							},
						},
					},
				},
				Done:       true,
				DoneReason: "stop",
				Metrics:    api.Metrics{PromptEvalCount: 20, EvalCount: 2},
			}
			_ = json.NewEncoder(w).Encode(resp)
		case 2:
			resp := api.ChatResponse{
				Model:      "test-model",
				Message:    api.Message{Role: "assistant", Content: "Final answer after 2 searches."},
				Done:       true,
				DoneReason: "stop",
				Metrics:    api.Metrics{PromptEvalCount: 30, EvalCount: 3},
			}
			_ = json.NewEncoder(w).Encode(resp)
		default:
			t.Fatalf("unexpected extra followup call: %d", followupCall)
		}
	}))
	defer followupServer.Close()
	t.Setenv("OLLAMA_HOST", followupServer.URL)

	searchServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		resp := anthropic.OllamaWebSearchResponse{
			Results: []anthropic.OllamaWebSearchResult{
				{Title: "Result", URL: "https://example.com", Content: "content"},
			},
		}
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer searchServer.Close()
	originalEndpoint := anthropic.WebSearchEndpoint
	anthropic.WebSearchEndpoint = searchServer.URL
	defer func() { anthropic.WebSearchEndpoint = originalEndpoint }()

	router := gin.New()
	router.Use(AnthropicMessagesMiddleware())
	router.POST("/v1/messages", func(c *gin.Context) {
		resp := api.ChatResponse{
			Model: "test-model",
			Message: api.Message{
				Role: "assistant",
				ToolCalls: []api.ToolCall{
					{
						ID: "call_ws_1",
						Function: api.ToolCallFunction{
							Name:      "web_search",
							Arguments: makeArgs("query", "loop query 1"),
						},
					},
				},
			},
			Done:       true,
			DoneReason: "stop",
			Metrics:    api.Metrics{PromptEvalCount: 10, EvalCount: 1},
		}
		data, _ := json.Marshal(resp)
		c.Writer.WriteHeader(http.StatusOK)
		_, _ = c.Writer.Write(data)
	})

	body := `{
		"model":"test-model:cloud",
		"max_tokens":100,
		"messages":[{"role":"user","content":"do multiple searches"}],
		"tools":[{"type":"web_search_20250305","name":"web_search"}]
	}`
	req, _ := http.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")

	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	if resp.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", resp.Code, resp.Body.String())
	}
	if followupCall != 2 {
		t.Fatalf("expected 2 followup calls, got %d", followupCall)
	}

	var result anthropic.MessagesResponse
	if err := json.Unmarshal(resp.Body.Bytes(), &result); err != nil {
		t.Fatalf("unmarshal error: %v", err)
	}

	serverToolUses := 0
	webResults := 0
	for _, block := range result.Content {
		if block.Type == "server_tool_use" {
			serverToolUses++
		}
		if block.Type == "web_search_tool_result" {
			webResults++
		}
	}
	if serverToolUses != 2 || webResults != 2 {
		t.Fatalf("expected two search iterations, got server_tool_use=%d web_search_tool_result=%d", serverToolUses, webResults)
	}

	if result.Usage.InputTokens != 60 || result.Usage.OutputTokens != 6 {
		t.Fatalf("unexpected aggregated usage: %+v", result.Usage)
	}
}

func TestWebSearchLoopMaxLimit(t *testing.T) {
	gin.SetMode(gin.TestMode)

	followupCall := 0
	followupServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		followupCall++
		resp := api.ChatResponse{
			Model: "test-model",
			Message: api.Message{
				Role: "assistant",
				ToolCalls: []api.ToolCall{
					{
						ID: "call_ws_loop_limit",
						Function: api.ToolCallFunction{
							Name:      "web_search",
							Arguments: makeArgs("query", "loop query next"),
						},
					},
				},
			},
			Done:       true,
			DoneReason: "stop",
			Metrics:    api.Metrics{PromptEvalCount: 7, EvalCount: 2},
		}
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer followupServer.Close()
	t.Setenv("OLLAMA_HOST", followupServer.URL)

	searchServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		resp := anthropic.OllamaWebSearchResponse{
			Results: []anthropic.OllamaWebSearchResult{
				{Title: "Result", URL: "https://example.com", Content: "content"},
			},
		}
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer searchServer.Close()
	originalEndpoint := anthropic.WebSearchEndpoint
	anthropic.WebSearchEndpoint = searchServer.URL
	defer func() { anthropic.WebSearchEndpoint = originalEndpoint }()

	router := gin.New()
	router.Use(AnthropicMessagesMiddleware())
	router.POST("/v1/messages", func(c *gin.Context) {
		resp := api.ChatResponse{
			Model: "test-model",
			Message: api.Message{
				Role: "assistant",
				ToolCalls: []api.ToolCall{
					{
						ID: "call_ws_initial",
						Function: api.ToolCallFunction{
							Name:      "web_search",
							Arguments: makeArgs("query", "loop query 1"),
						},
					},
				},
			},
			Done:       true,
			DoneReason: "stop",
			Metrics:    api.Metrics{PromptEvalCount: 5, EvalCount: 1},
		}
		data, _ := json.Marshal(resp)
		c.Writer.WriteHeader(http.StatusOK)
		_, _ = c.Writer.Write(data)
	})

	body := `{
		"model":"test-model:cloud",
		"max_tokens":100,
		"messages":[{"role":"user","content":"keep searching"}],
		"tools":[{"type":"web_search_20250305","name":"web_search"}]
	}`
	req, _ := http.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")

	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	if resp.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", resp.Code, resp.Body.String())
	}
	if followupCall != 3 {
		t.Fatalf("expected 3 followup calls before max loop error, got %d", followupCall)
	}

	var result anthropic.MessagesResponse
	if err := json.Unmarshal(resp.Body.Bytes(), &result); err != nil {
		t.Fatalf("unmarshal error: %v", err)
	}

	last := result.Content[len(result.Content)-1]
	if last.Type != "web_search_tool_result" {
		t.Fatalf("expected last block web_search_tool_result, got %q", last.Type)
	}
	contentJSON, _ := json.Marshal(last.Content)
	var errContent anthropic.WebSearchToolResultError
	if err := json.Unmarshal(contentJSON, &errContent); err != nil {
		t.Fatalf("failed to parse web search error content: %v", err)
	}
	if errContent.ErrorCode != "max_uses_exceeded" {
		t.Fatalf("expected max_uses_exceeded error, got %q", errContent.ErrorCode)
	}
	if result.StopReason != "end_turn" {
		t.Fatalf("expected end_turn, got %q", result.StopReason)
	}
}

func TestWebSearchStreamingFinalStopReasonToolUse(t *testing.T) {
	gin.SetMode(gin.TestMode)

	followupServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		resp := api.ChatResponse{
			Model: "test-model",
			Message: api.Message{
				Role: "assistant",
				ToolCalls: []api.ToolCall{
					{
						ID: "call_weather_stream",
						Function: api.ToolCallFunction{
							Name:      "get_weather",
							Arguments: makeArgs("location", "Seattle"),
						},
					},
				},
			},
			Done:       true,
			DoneReason: "stop",
			Metrics:    api.Metrics{PromptEvalCount: 14, EvalCount: 5},
		}
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer followupServer.Close()
	t.Setenv("OLLAMA_HOST", followupServer.URL)

	searchServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		resp := anthropic.OllamaWebSearchResponse{
			Results: []anthropic.OllamaWebSearchResult{
				{Title: "Result", URL: "https://example.com", Content: "content"},
			},
		}
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer searchServer.Close()
	originalEndpoint := anthropic.WebSearchEndpoint
	anthropic.WebSearchEndpoint = searchServer.URL
	defer func() { anthropic.WebSearchEndpoint = originalEndpoint }()

	router := gin.New()
	router.Use(AnthropicMessagesMiddleware())
	router.POST("/v1/messages", func(c *gin.Context) {
		chunks := []api.ChatResponse{
			{
				Model:   "test-model",
				Message: api.Message{Role: "assistant", Content: "Let me check. "},
				Done:    false,
			},
			{
				Model: "test-model",
				Message: api.Message{
					Role: "assistant",
					ToolCalls: []api.ToolCall{
						{
							ID: "call_ws_stream_tool_use",
							Function: api.ToolCallFunction{
								Name:      "web_search",
								Arguments: makeArgs("query", "weather seattle"),
							},
						},
					},
				},
				Done: false,
			},
		}
		c.Writer.WriteHeader(http.StatusOK)
		for _, chunk := range chunks {
			data, _ := json.Marshal(chunk)
			_, _ = c.Writer.Write(data)
		}
	})

	body := `{
		"model":"test-model:cloud",
		"max_tokens":100,
		"stream":true,
		"messages":[{"role":"user","content":"Should I take a jacket?"}],
		"tools":[
			{"type":"web_search_20250305","name":"web_search"},
			{"type":"custom","name":"get_weather","input_schema":{"type":"object"}}
		]
	}`
	req, _ := http.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")

	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	if resp.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", resp.Code, resp.Body.String())
	}

	events := parseSSEEvents(t, resp.Body.String())
	if countEventsByName(events, "message_start") != 1 {
		t.Fatalf("expected exactly one message_start, got %d", countEventsByName(events, "message_start"))
	}

	var messageDelta anthropic.MessageDeltaEvent
	foundMessageDelta := false
	foundToolUse := false
	for _, event := range events {
		if event.event == "message_delta" {
			foundMessageDelta = true
			if err := json.Unmarshal([]byte(event.data), &messageDelta); err != nil {
				t.Fatalf("failed to unmarshal message_delta: %v", err)
			}
		}
		if event.event == "content_block_start" {
			var start anthropic.ContentBlockStartEvent
			if err := json.Unmarshal([]byte(event.data), &start); err != nil {
				t.Fatalf("failed to unmarshal content_block_start: %v", err)
			}
			if start.ContentBlock.Type == "tool_use" && start.ContentBlock.Name == "get_weather" {
				foundToolUse = true
			}
		}
	}

	if !foundMessageDelta {
		t.Fatal("expected message_delta event")
	}
	if messageDelta.Delta.StopReason != "tool_use" {
		t.Fatalf("expected stop_reason tool_use, got %q", messageDelta.Delta.StopReason)
	}
	if !foundToolUse {
		t.Fatal("expected tool_use content block for get_weather")
	}
}

func TestWebSearchFollowupNon200ReturnsApiError(t *testing.T) {
	gin.SetMode(gin.TestMode)

	followupServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "boom", http.StatusInternalServerError)
	}))
	defer followupServer.Close()
	t.Setenv("OLLAMA_HOST", followupServer.URL)

	searchServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		resp := anthropic.OllamaWebSearchResponse{
			Results: []anthropic.OllamaWebSearchResult{
				{Title: "Result", URL: "https://example.com", Content: "content"},
			},
		}
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer searchServer.Close()
	originalEndpoint := anthropic.WebSearchEndpoint
	anthropic.WebSearchEndpoint = searchServer.URL
	defer func() { anthropic.WebSearchEndpoint = originalEndpoint }()

	router := gin.New()
	router.Use(AnthropicMessagesMiddleware())
	router.POST("/v1/messages", func(c *gin.Context) {
		resp := api.ChatResponse{
			Model: "test-model",
			Message: api.Message{
				Role: "assistant",
				ToolCalls: []api.ToolCall{
					{
						ID: "call_ws_non200",
						Function: api.ToolCallFunction{
							Name:      "web_search",
							Arguments: makeArgs("query", "test"),
						},
					},
				},
			},
			Done:       true,
			DoneReason: "stop",
			Metrics:    api.Metrics{PromptEvalCount: 9, EvalCount: 1},
		}
		data, _ := json.Marshal(resp)
		c.Writer.WriteHeader(http.StatusOK)
		_, _ = c.Writer.Write(data)
	})

	body := `{
		"model":"test-model:cloud",
		"max_tokens":100,
		"messages":[{"role":"user","content":"test"}],
		"tools":[{"type":"web_search_20250305","name":"web_search"}]
	}`
	req, _ := http.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")

	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	if resp.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", resp.Code, resp.Body.String())
	}

	var result anthropic.MessagesResponse
	if err := json.Unmarshal(resp.Body.Bytes(), &result); err != nil {
		t.Fatalf("unmarshal error: %v", err)
	}
	if len(result.Content) != 2 {
		t.Fatalf("expected 2 blocks in error response, got %d", len(result.Content))
	}

	contentJSON, _ := json.Marshal(result.Content[1].Content)
	var errContent anthropic.WebSearchToolResultError
	if err := json.Unmarshal(contentJSON, &errContent); err != nil {
		t.Fatalf("failed to parse error content: %v", err)
	}
	if errContent.ErrorCode != "api_error" {
		t.Fatalf("expected api_error, got %q", errContent.ErrorCode)
	}
}

func countEventsByName(events []sseEvent, eventName string) int {
	count := 0
	for _, event := range events {
		if event.event == eventName {
			count++
		}
	}
	return count
}

func collectTextDeltas(t *testing.T, events []sseEvent) []string {
	t.Helper()

	var deltas []string
	for _, event := range events {
		if event.event != "content_block_delta" {
			continue
		}

		var delta anthropic.ContentBlockDeltaEvent
		if err := json.Unmarshal([]byte(event.data), &delta); err != nil {
			t.Fatalf("failed to unmarshal content_block_delta: %v", err)
		}
		if delta.Delta.Type == "text_delta" {
			deltas = append(deltas, delta.Delta.Text)
		}
	}

	return deltas
}

func containsString(values []string, target string) bool {
	for _, value := range values {
		if value == target {
			return true
		}
	}
	return false
}
