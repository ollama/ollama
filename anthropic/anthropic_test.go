package anthropic

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/google/go-cmp/cmp"

	"github.com/ollama/ollama/api"
)

var (
	False = false
	True  = true
)

func captureRequestMiddleware(capturedRequest any) gin.HandlerFunc {
	return func(c *gin.Context) {
		bodyBytes, _ := io.ReadAll(c.Request.Body)
		c.Request.Body = io.NopCloser(bytes.NewReader(bodyBytes))
		err := json.Unmarshal(bodyBytes, capturedRequest)
		if err != nil {
			c.AbortWithStatusJSON(http.StatusInternalServerError, "failed to unmarshal request")
		}
		c.Next()
	}
}

func TestMessagesMiddleware(t *testing.T) {
	type testCase struct {
		name string
		body string
		req  api.ChatRequest
		err  ErrorResponse
	}

	var capturedRequest *api.ChatRequest

	testCases := []testCase{
		{
			name: "basic message request",
			body: `{
				"model": "claude-3-sonnet",
				"max_tokens": 100,
				"messages": [
					{"role": "user", "content": "Hello, how are you?"}
				]
			}`,
			req: api.ChatRequest{
				Model: "claude-3-sonnet",
				Messages: []api.Message{
					{
						Role:    "user",
						Content: "Hello, how are you?",
					},
					},
				Options: map[string]any{
					"num_predict": 100.0,
				},
				Stream: &False,
			},
		},
		{
			name: "message request with system prompt",
			body: `{
				"model": "claude-3-sonnet",
				"max_tokens": 100,
				"system": "You are a helpful assistant.",
				"messages": [
					{"role": "user", "content": "Hello"}
				]
			}`,
			req: api.ChatRequest{
				Model: "claude-3-sonnet",
				Messages: []api.Message{
					{
						Role:    "system",
						Content: "You are a helpful assistant.",
					},
					{
						Role:    "user",
						Content: "Hello",
					},
					},
				Options: map[string]any{
					"num_predict": 100.0,
				},
				Stream: &False,
			},
		},
		{
			name: "message request with options",
			body: `{
				"model": "claude-3-sonnet",
				"max_tokens": 200,
				"messages": [
					{"role": "user", "content": "Hello"}
				],
				"temperature": 0.7,
				"top_p": 0.9,
				"stop_sequences": ["\n", "stop"],
				"stream": true
			}`,
			req: api.ChatRequest{
				Model: "claude-3-sonnet",
				Messages: []api.Message{
					{
						Role:    "user",
						Content: "Hello",
					},
					},
				Options: map[string]any{
					"num_predict": 200.0,
					"temperature": 0.7,
					"top_p":       0.9,
					"stop":        []any{"\n", "stop"},
				},
				Stream: &True,
			},
		},
		{
			name: "message request with image content",
			body: `{
				"model": "claude-3-sonnet",
				"max_tokens": 100,
				"messages": [
					{
						"role": "user",
						"content": [
							{
								"type": "text",
								"text": "What is in this image?"
							},
							{
								"type": "image",
								"source": {
									"type": "base64",
									"media_type": "image/jpeg",
									"data": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
								}
							}
						]
					}
				]
			}`,
			req: api.ChatRequest{
				Model: "claude-3-sonnet",
				Messages: []api.Message{
					{
						Role:    "user",
						Content: "What is in this image?",
						Images: []api.ImageData{
							func() []byte {
								img, _ := base64.StdEncoding.DecodeString("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=")
								return img
							}(),
						},
					},
					},
				Options: map[string]any{
					"num_predict": 100.0,
				},
				Stream: &False,
			},
		},
		{
			name: "message request with tools",
			body: `{
				"model": "claude-3-sonnet",
				"max_tokens": 100,
				"messages": [
					{"role": "user", "content": "What's the weather like in Paris?"}
				],
				"tools": [
					{
						"name": "get_weather",
						"description": "Get the current weather",
						"input_schema": {
							"type": "object",
							"properties": {
								"location": {"type": "string"}
							},
							"required": ["location"]
						}
					}
				]
			}`,
			req: api.ChatRequest{
				Model: "claude-3-sonnet",
				Messages: []api.Message{
					{
						Role:    "user",
						Content: "What's the weather like in Paris?",
					},
					},
				Options: map[string]any{
					"num_predict": 100.0,
				},
				Tools: []api.Tool{
					{
						Type: "function",
						Function: api.ToolFunction{
							Name:        "get_weather",
							Description: "Get the current weather",
							Parameters: struct {
								Type       string   `json:"type"`
								Defs       any      `json:"$defs,omitempty"`
								Items      any      `json:"items,omitempty"`
								Required   []string `json:"required"`
								Properties map[string]struct {
									Type        api.PropertyType `json:"type"`
									Items       any              `json:"items,omitempty"`
									Description string           `json:"description"`
									Enum        []any            `json:"enum,omitempty"`
								} `json:"properties"`
							}{
								Type:     "object",
								Required: []string{"location"},
								Properties: map[string]struct {
									Type        api.PropertyType `json:"type"`
									Items       any              `json:"items,omitempty"`
									Description string           `json:"description"`
									Enum        []any            `json:"enum,omitempty"`
								}{
									"location": {
										Type:        []string{"string"},
										Description: "location",
									},
								},
							},
						},
					},
				},
				Stream: &False,
			},
		},
		{
			name: "missing max_tokens",
			body: `{
				"model": "claude-3-sonnet",
				"messages": [
					{"role": "user", "content": "Hello"}
				]
			}`,
		err: ErrorResponse{
			Error: Error{
				Message: "max_tokens is required and must be greater than 0",
				Type:    "api_error",
			},
		},
		},
		{
			name: "invalid content type",
			body: `{
				"model": "claude-3-sonnet",
				"max_tokens": 100,
				"messages": [
					{"role": "user", "content": 123}
				]
			}`,
		err: ErrorResponse{
			Error: Error{
				Message: "invalid content type: float64",
				Type:    "api_error",
			},
		},
		},
		{
			name: "empty messages",
			body: `{
				"model": "claude-3-sonnet",
				"max_tokens": 100,
				"messages": []
			}`,
		err: ErrorResponse{
			Error: Error{
				Message: "messages field is required",
				Type:    "api_error",
			},
		},
		},
	}

	endpoint := func(c *gin.Context) {
		c.Status(http.StatusOK)
	}

	gin.SetMode(gin.TestMode)
	router := gin.New()
	router.Use(MessagesMiddleware(), captureRequestMiddleware(&capturedRequest))
	router.Handle(http.MethodPost, "/api/chat", endpoint)

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			req, _ := http.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(tc.body))
			req.Header.Set("Content-Type", "application/json")
			req.Header.Set("Anthropic-Version", "2023-06-01")

			defer func() { capturedRequest = nil }()

			resp := httptest.NewRecorder()
			router.ServeHTTP(resp, req)

			var errResp ErrorResponse
			if resp.Code != http.StatusOK {
				if err := json.Unmarshal(resp.Body.Bytes(), &errResp); err != nil {
					t.Fatal(err)
				}
				if diff := cmp.Diff(tc.err, errResp); diff != "" {
					t.Fatalf("errors did not match for %s:\n%s", tc.name, diff)
				}
				return
			}

			if diff := cmp.Diff(&tc.req, capturedRequest); diff != "" {
				t.Fatalf("requests did not match for %s:\n%s", tc.name, diff)
			}
		})
	}
}

func TestToMessageResponse(t *testing.T) {
	testCases := []struct {
		name     string
		response api.ChatResponse
		expected MessageResponse
	}{
		{
			name: "basic response",
			response: api.ChatResponse{
				Model:      "test-model",
				Message:    api.Message{Role: "assistant", Content: "Hello there!"},
				DoneReason: "stop",
				Done:       true,
				Metrics: api.Metrics{
					PromptEvalCount: 10,
					EvalCount:       5,
				},
			},
			expected: MessageResponse{
				Type:       "message",
				Role:       "assistant",
				Model:      "test-model",
				Content:    []ContentBlock{{Type: "text", Text: "Hello there!"}},
				StopReason: strPtr("end_turn"),
				Usage:      Usage{InputTokens: 10, OutputTokens: 5},
			},
		},
		{
			name: "response with max_tokens stop reason",
			response: api.ChatResponse{
				Model:      "test-model",
				Message:    api.Message{Role: "assistant", Content: "Partial response"},
				DoneReason: "length",
				Done:       true,
				Metrics: api.Metrics{
					PromptEvalCount: 8,
					EvalCount:       12,
				},
			},
			expected: MessageResponse{
				Type:       "message",
				Role:       "assistant",
				Model:      "test-model",
				Content:    []ContentBlock{{Type: "text", Text: "Partial response"}},
				StopReason: strPtr("max_tokens"),
				Usage:      Usage{InputTokens: 8, OutputTokens: 12},
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result := toMessageResponse("test-id", tc.response)
			
			// Check specific fields since ID is generated
			if result.Type != tc.expected.Type {
				t.Errorf("Type mismatch: got %v, want %v", result.Type, tc.expected.Type)
			}
			if result.Role != tc.expected.Role {
				t.Errorf("Role mismatch: got %v, want %v", result.Role, tc.expected.Role)
			}
			if result.Model != tc.expected.Model {
				t.Errorf("Model mismatch: got %v, want %v", result.Model, tc.expected.Model)
			}
			if !reflect.DeepEqual(result.Content, tc.expected.Content) {
				t.Errorf("Content mismatch: got %v, want %v", result.Content, tc.expected.Content)
			}
			if result.Usage != tc.expected.Usage {
				t.Errorf("Usage mismatch: got %v, want %v", result.Usage, tc.expected.Usage)
			}
			
			if tc.expected.StopReason != nil {
				if result.StopReason == nil || *result.StopReason != *tc.expected.StopReason {
					t.Errorf("StopReason mismatch: got %v, want %v", result.StopReason, tc.expected.StopReason)
				}
			}
		})
	}
}

func strPtr(s string) *string {
	return &s
}

func TestFromAnthropicMessages(t *testing.T) {
	testCases := []struct {
		name     string
		messages []Message
		system   string
		expected []api.Message
		wantErr  bool
	}{
		{
			name: "basic messages",
			messages: []Message{
				{Role: "user", Content: "Hello"},
				{Role: "assistant", Content: "Hi there!"},
			},
			expected: []api.Message{
				{Role: "user", Content: "Hello", Images: []api.ImageData{}},
				{Role: "assistant", Content: "Hi there!", Images: []api.ImageData{}},
			},
			wantErr: false,
		},
		{
			name: "messages with system message in array",
			messages: []Message{
				{Role: "system", Content: "You are a helpful assistant."},
				{Role: "user", Content: "Hello"},
			},
			system: "You are a helpful assistant.",
			expected: []api.Message{
				{Role: "user", Content: "Hello", Images: []api.ImageData{}},
			},
			wantErr: false,
		},
		{
			name: "invalid content type",
			messages: []Message{
				{Role: "user", Content: 123},
			},
			expected: nil,
			wantErr:  true,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result, system, err := fromAnthropicMessages(tc.messages)
			
			if tc.wantErr {
				if err == nil {
					t.Error("expected error, got nil")
				}
				return
			}
			
			if err != nil {
				t.Errorf("unexpected error: %v", err)
				return
			}
			
			// Check messages
			if len(result) != len(tc.expected) {
				t.Errorf("message count mismatch: got %d, want %d", len(result), len(tc.expected))
			}
			
			for i, msg := range result {
				if i >= len(tc.expected) {
					break
				}
				expected := tc.expected[i]
				
				if msg.Role != expected.Role {
					t.Errorf("Message %d role mismatch: got %q, want %q", i, msg.Role, expected.Role)
				}
				if msg.Content != expected.Content {
					t.Errorf("Message %d content mismatch: got %q, want %q", i, msg.Content, expected.Content)
				}
				if msg.Thinking != expected.Thinking {
					t.Errorf("Message %d thinking mismatch: got %q, want %q", i, msg.Thinking, expected.Thinking)
				}
				if msg.ToolName != expected.ToolName {
					t.Errorf("Message %d tool name mismatch: got %q, want %q", i, msg.ToolName, expected.ToolName)
				}
				if !reflect.DeepEqual(msg.Images, expected.Images) {
					t.Errorf("Message %d images mismatch: got %v, want %v", i, msg.Images, expected.Images)
				}
				if !reflect.DeepEqual(msg.ToolCalls, expected.ToolCalls) {
					t.Errorf("Message %d tool calls mismatch: got %v, want %v", i, msg.ToolCalls, expected.ToolCalls)
				}
			}
			
			// Check system prompt
			if system != tc.system {
				t.Errorf("system prompt mismatch: got %q, want %q", system, tc.system)
			}
		})
	}
}

func TestContentBlocks(t *testing.T) {
	testCases := []struct {
		name      string
		content   string
		toolCalls []api.ToolCall
		expected  []ContentBlock
	}{
		{
			name:    "text content only",
			content: "Hello world",
			expected: []ContentBlock{
				{Type: "text", Text: "Hello world"},
			},
		},
		{
			name:  "tool calls only",
			content: "",
			toolCalls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name: "get_weather",
						Arguments: map[string]interface{}{
							"location": "Paris",
						},
					},
				},
			},
			expected: []ContentBlock{
				{
					Type:  "tool_use",
					Name:  "get_weather",
					Input: map[string]interface{}{"location": "Paris"},
				},
			},
		},
		{
			name:    "both text and tool calls",
			content: "I'll check the weather for you.",
			toolCalls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name: "get_weather",
						Arguments: map[string]interface{}{
							"location": "Paris",
						},
					},
				},
			},
			expected: []ContentBlock{
				{Type: "text", Text: "I'll check the weather for you."},
				{
					Type:  "tool_use",
					Name:  "get_weather",
					Input: map[string]interface{}{"location": "Paris"},
				},
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result := toContentBlocks(tc.content, tc.toolCalls)
			
			// Check content blocks without ID (since it's generated)
			for i := range result {
				if result[i].Type == "tool_use" {
					// Clear ID for comparison
					result[i].ID = ""
				}
			}
			
			if !reflect.DeepEqual(result, tc.expected) {
				t.Errorf("content blocks mismatch: got %v, want %v", result, tc.expected)
			}
		})
	}
}