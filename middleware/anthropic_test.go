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
