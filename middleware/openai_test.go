package middleware

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
	"time"

	"github.com/gin-gonic/gin"
	"github.com/google/go-cmp/cmp"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/openai"
)

// testPropsMap creates a ToolPropertiesMap from a map (convenience function for tests)
func testPropsMap(m map[string]api.ToolProperty) *api.ToolPropertiesMap {
	props := api.NewToolPropertiesMap()
	for k, v := range m {
		props.Set(k, v)
	}
	return props
}

// testArgs creates ToolCallFunctionArguments from a map (convenience function for tests)
func testArgs(m map[string]any) api.ToolCallFunctionArguments {
	args := api.NewToolCallFunctionArguments()
	for k, v := range m {
		args.Set(k, v)
	}
	return args
}

// argsComparer provides cmp options for comparing ToolCallFunctionArguments by value
var argsComparer = cmp.Comparer(func(a, b api.ToolCallFunctionArguments) bool {
	return cmp.Equal(a.ToMap(), b.ToMap())
})

// propsComparer provides cmp options for comparing ToolPropertiesMap by value
var propsComparer = cmp.Comparer(func(a, b *api.ToolPropertiesMap) bool {
	if a == nil && b == nil {
		return true
	}
	if a == nil || b == nil {
		return false
	}
	return cmp.Equal(a.ToMap(), b.ToMap())
})

const (
	prefix = `data:image/jpeg;base64,`
	image  = `iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=`
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

func TestChatMiddleware(t *testing.T) {
	type testCase struct {
		name string
		body string
		req  api.ChatRequest
		err  openai.ErrorResponse
	}

	var capturedRequest *api.ChatRequest

	testCases := []testCase{
		{
			name: "chat handler",
			body: `{
				"model": "test-model",
				"messages": [
					{"role": "user", "content": "Hello"}
				]
			}`,
			req: api.ChatRequest{
				Model: "test-model",
				Messages: []api.Message{
					{
						Role:    "user",
						Content: "Hello",
					},
				},
				Options: map[string]any{
					"temperature": 1.0,
					"top_p":       1.0,
				},
				Stream: &False,
			},
		},
		{
			name: "chat handler with options",
			body: `{
				"model": "test-model",
				"messages": [
					{"role": "user", "content": "Hello"}
				],
				"stream":            true,
				"max_tokens":        999,
				"seed":              123,
				"stop":              ["\n", "stop"],
				"temperature":       3.0,
				"frequency_penalty": 4.0,
				"presence_penalty":  5.0,
				"top_p":             6.0,
				"response_format":   {"type": "json_object"}
			}`,
			req: api.ChatRequest{
				Model: "test-model",
				Messages: []api.Message{
					{
						Role:    "user",
						Content: "Hello",
					},
				},
				Options: map[string]any{
					"num_predict":       999.0, // float because JSON doesn't distinguish between float and int
					"seed":              123.0,
					"stop":              []any{"\n", "stop"},
					"temperature":       3.0,
					"frequency_penalty": 4.0,
					"presence_penalty":  5.0,
					"top_p":             6.0,
				},
				Format: json.RawMessage(`"json"`),
				Stream: &True,
			},
		},
		{
			name: "chat handler with streaming usage",
			body: `{
				"model": "test-model",
				"messages": [
					{"role": "user", "content": "Hello"}
				],
				"stream":            true,
				"stream_options":    {"include_usage": true},
				"max_tokens":        999,
				"seed":              123,
				"stop":              ["\n", "stop"],
				"temperature":       3.0,
				"frequency_penalty": 4.0,
				"presence_penalty":  5.0,
				"top_p":             6.0,
				"response_format":   {"type": "json_object"}
			}`,
			req: api.ChatRequest{
				Model: "test-model",
				Messages: []api.Message{
					{
						Role:    "user",
						Content: "Hello",
					},
				},
				Options: map[string]any{
					"num_predict":       999.0, // float because JSON doesn't distinguish between float and int
					"seed":              123.0,
					"stop":              []any{"\n", "stop"},
					"temperature":       3.0,
					"frequency_penalty": 4.0,
					"presence_penalty":  5.0,
					"top_p":             6.0,
				},
				Format: json.RawMessage(`"json"`),
				Stream: &True,
			},
		},
		{
			name: "chat handler with image content",
			body: `{
				"model": "test-model",
				"messages": [
					{
						"role": "user",
						"content": [
							{
								"type": "text",
								"text": "Hello"
							},
							{
								"type": "image_url",
								"image_url": {
									"url": "` + prefix + image + `"
								}
							}
						]
					}
				]
			}`,
			req: api.ChatRequest{
				Model: "test-model",
				Messages: []api.Message{
					{
						Role:    "user",
						Content: "Hello",
					},
					{
						Role: "user",
						Images: []api.ImageData{
							func() []byte {
								img, _ := base64.StdEncoding.DecodeString(image)
								return img
							}(),
						},
					},
				},
				Options: map[string]any{
					"temperature": 1.0,
					"top_p":       1.0,
				},
				Stream: &False,
			},
		},
		{
			name: "chat handler with tools",
			body: `{
				"model": "test-model",
				"messages": [
					{"role": "user", "content": "What's the weather like in Paris Today?"},
					{"role": "assistant", "tool_calls": [{"id": "id", "type": "function", "function": {"name": "get_current_weather", "arguments": "{\"location\": \"Paris, France\", \"format\": \"celsius\"}"}}]}
				]
			}`,
			req: api.ChatRequest{
				Model: "test-model",
				Messages: []api.Message{
					{
						Role:    "user",
						Content: "What's the weather like in Paris Today?",
					},
					{
						Role: "assistant",
						ToolCalls: []api.ToolCall{
							{
								ID: "id",
								Function: api.ToolCallFunction{
									Name: "get_current_weather",
									Arguments: testArgs(map[string]any{
										"location": "Paris, France",
										"format":   "celsius",
									}),
								},
							},
						},
					},
				},
				Options: map[string]any{
					"temperature": 1.0,
					"top_p":       1.0,
				},
				Stream: &False,
			},
		},
		{
			name: "chat handler with tools and content",
			body: `{
				"model": "test-model",
				"messages": [
					{"role": "user", "content": "What's the weather like in Paris Today?"},
					{"role": "assistant", "content": "Let's see what the weather is like in Paris", "tool_calls": [{"id": "id", "type": "function", "function": {"name": "get_current_weather", "arguments": "{\"location\": \"Paris, France\", \"format\": \"celsius\"}"}}]}
				]
			}`,
			req: api.ChatRequest{
				Model: "test-model",
				Messages: []api.Message{
					{
						Role:    "user",
						Content: "What's the weather like in Paris Today?",
					},
					{
						Role:    "assistant",
						Content: "Let's see what the weather is like in Paris",
						ToolCalls: []api.ToolCall{
							{
								ID: "id",
								Function: api.ToolCallFunction{
									Name: "get_current_weather",
									Arguments: testArgs(map[string]any{
										"location": "Paris, France",
										"format":   "celsius",
									}),
								},
							},
						},
					},
				},
				Options: map[string]any{
					"temperature": 1.0,
					"top_p":       1.0,
				},
				Stream: &False,
			},
		},
		{
			name: "chat handler with tools and empty content",
			body: `{
				"model": "test-model",
				"messages": [
					{"role": "user", "content": "What's the weather like in Paris Today?"},
					{"role": "assistant", "content": "", "tool_calls": [{"id": "id", "type": "function", "function": {"name": "get_current_weather", "arguments": "{\"location\": \"Paris, France\", \"format\": \"celsius\"}"}}]}
				]
			}`,
			req: api.ChatRequest{
				Model: "test-model",
				Messages: []api.Message{
					{
						Role:    "user",
						Content: "What's the weather like in Paris Today?",
					},
					{
						Role: "assistant",
						ToolCalls: []api.ToolCall{
							{
								ID: "id",
								Function: api.ToolCallFunction{
									Name: "get_current_weather",
									Arguments: testArgs(map[string]any{
										"location": "Paris, France",
										"format":   "celsius",
									}),
								},
							},
						},
					},
				},
				Options: map[string]any{
					"temperature": 1.0,
					"top_p":       1.0,
				},
				Stream: &False,
			},
		},
		{
			name: "chat handler with tools and thinking content",
			body: `{
				"model": "test-model",
				"messages": [
					{"role": "user", "content": "What's the weather like in Paris Today?"},
					{"role": "assistant", "reasoning": "Let's see what the weather is like in Paris", "tool_calls": [{"id": "id", "type": "function", "function": {"name": "get_current_weather", "arguments": "{\"location\": \"Paris, France\", \"format\": \"celsius\"}"}}]}
				]
			}`,
			req: api.ChatRequest{
				Model: "test-model",
				Messages: []api.Message{
					{
						Role:    "user",
						Content: "What's the weather like in Paris Today?",
					},
					{
						Role:     "assistant",
						Thinking: "Let's see what the weather is like in Paris",
						ToolCalls: []api.ToolCall{
							{
								ID: "id",
								Function: api.ToolCallFunction{
									Name: "get_current_weather",
									Arguments: testArgs(map[string]any{
										"location": "Paris, France",
										"format":   "celsius",
									}),
								},
							},
						},
					},
				},
				Options: map[string]any{
					"temperature": 1.0,
					"top_p":       1.0,
				},
				Stream: &False,
			},
		},
		{
			name: "tool response with call ID",
			body: `{
				"model": "test-model",
				"messages": [
					{"role": "user", "content": "What's the weather like in Paris Today?"},
					{"role": "assistant", "tool_calls": [{"id": "id_abc", "type": "function", "function": {"name": "get_current_weather", "arguments": "{\"location\": \"Paris, France\", \"format\": \"celsius\"}"}}]},
					{"role": "tool", "tool_call_id": "id_abc", "content": "The weather in Paris is 20 degrees Celsius"}
				]
			}`,
			req: api.ChatRequest{
				Model: "test-model",
				Messages: []api.Message{
					{
						Role:    "user",
						Content: "What's the weather like in Paris Today?",
					},
					{
						Role: "assistant",
						ToolCalls: []api.ToolCall{
							{
								ID: "id_abc",
								Function: api.ToolCallFunction{
									Name: "get_current_weather",
									Arguments: testArgs(map[string]any{
										"location": "Paris, France",
										"format":   "celsius",
									}),
								},
							},
						},
					},
					{
						Role:       "tool",
						Content:    "The weather in Paris is 20 degrees Celsius",
						ToolName:   "get_current_weather",
						ToolCallID: "id_abc",
					},
				},
				Options: map[string]any{
					"temperature": 1.0,
					"top_p":       1.0,
				},
				Stream: &False,
			},
		},
		{
			name: "tool response with name",
			body: `{
				"model": "test-model",
				"messages": [
					{"role": "user", "content": "What's the weather like in Paris Today?"},
					{"role": "assistant", "tool_calls": [{"id": "id", "type": "function", "function": {"name": "get_current_weather", "arguments": "{\"location\": \"Paris, France\", \"format\": \"celsius\"}"}}]},
					{"role": "tool", "name": "get_current_weather", "content": "The weather in Paris is 20 degrees Celsius"}
				]
			}`,
			req: api.ChatRequest{
				Model: "test-model",
				Messages: []api.Message{
					{
						Role:    "user",
						Content: "What's the weather like in Paris Today?",
					},
					{
						Role: "assistant",
						ToolCalls: []api.ToolCall{
							{
								ID: "id",
								Function: api.ToolCallFunction{
									Name: "get_current_weather",
									Arguments: testArgs(map[string]any{
										"location": "Paris, France",
										"format":   "celsius",
									}),
								},
							},
						},
					},
					{
						Role:     "tool",
						Content:  "The weather in Paris is 20 degrees Celsius",
						ToolName: "get_current_weather",
					},
				},
				Options: map[string]any{
					"temperature": 1.0,
					"top_p":       1.0,
				},
				Stream: &False,
			},
		},
		{
			name: "chat handler with streaming tools",
			body: `{
				"model": "test-model",
				"messages": [
					{"role": "user", "content": "What's the weather like in Paris?"}
				],
				"stream": true,
				"tools": [{
					"type": "function",
					"function": {
						"name": "get_weather",
						"description": "Get the current weather",
						"parameters": {
							"type": "object",
							"required": ["location"],
							"properties": {
								"location": {
									"type": "string",
									"description": "The city and state"
								},
								"unit": {
									"type": "string",
									"enum": ["celsius", "fahrenheit"]
								}
							}
						}
					}
				}]
			}`,
			req: api.ChatRequest{
				Model: "test-model",
				Messages: []api.Message{
					{
						Role:    "user",
						Content: "What's the weather like in Paris?",
					},
				},
				Tools: []api.Tool{
					{
						Type: "function",
						Function: api.ToolFunction{
							Name:        "get_weather",
							Description: "Get the current weather",
							Parameters: api.ToolFunctionParameters{
								Type:     "object",
								Required: []string{"location"},
								Properties: testPropsMap(map[string]api.ToolProperty{
									"location": {
										Type:        api.PropertyType{"string"},
										Description: "The city and state",
									},
									"unit": {
										Type: api.PropertyType{"string"},
										Enum: []any{"celsius", "fahrenheit"},
									},
								}),
							},
						},
					},
				},
				Options: map[string]any{
					"temperature": 1.0,
					"top_p":       1.0,
				},
				Stream: &True,
			},
		},
		{
			name: "chat handler error forwarding",
			body: `{
				"model": "test-model",
				"messages": [
					{"role": "user", "content": 2}
				]
			}`,
			err: openai.ErrorResponse{
				Error: openai.Error{
					Message: "invalid message content type: float64",
					Type:    "invalid_request_error",
				},
			},
		},
	}

	endpoint := func(c *gin.Context) {
		c.Status(http.StatusOK)
	}

	gin.SetMode(gin.TestMode)
	router := gin.New()
	router.Use(ChatMiddleware(), captureRequestMiddleware(&capturedRequest))
	router.Handle(http.MethodPost, "/api/chat", endpoint)

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			req, _ := http.NewRequest(http.MethodPost, "/api/chat", strings.NewReader(tc.body))
			req.Header.Set("Content-Type", "application/json")

			defer func() { capturedRequest = nil }()

			resp := httptest.NewRecorder()
			router.ServeHTTP(resp, req)

			var errResp openai.ErrorResponse
			if resp.Code != http.StatusOK {
				if err := json.Unmarshal(resp.Body.Bytes(), &errResp); err != nil {
					t.Fatal(err)
				}
				return
			}
			if diff := cmp.Diff(&tc.req, capturedRequest, argsComparer, propsComparer); diff != "" {
				t.Fatalf("requests did not match: %+v", diff)
			}
			if diff := cmp.Diff(tc.err, errResp); diff != "" {
				t.Fatalf("errors did not match for %s:\n%s", tc.name, diff)
			}
		})
	}
}

func TestCompletionsMiddleware(t *testing.T) {
	type testCase struct {
		name string
		body string
		req  api.GenerateRequest
		err  openai.ErrorResponse
	}

	var capturedRequest *api.GenerateRequest

	testCases := []testCase{
		{
			name: "completions handler",
			body: `{
				"model": "test-model",
				"prompt": "Hello",
				"temperature": 0.8,
				"stop": ["\n", "stop"],
				"suffix": "suffix"
			}`,
			req: api.GenerateRequest{
				Model:  "test-model",
				Prompt: "Hello",
				Options: map[string]any{
					"frequency_penalty": 0.0,
					"presence_penalty":  0.0,
					"temperature":       0.8,
					"top_p":             1.0,
					"stop":              []any{"\n", "stop"},
				},
				Suffix: "suffix",
				Stream: &False,
			},
		},
		{
			name: "completions handler stream",
			body: `{
				"model": "test-model",
				"prompt": "Hello",
				"stream": true,
				"temperature": 0.8,
				"stop": ["\n", "stop"],
				"suffix": "suffix"
			}`,
			req: api.GenerateRequest{
				Model:  "test-model",
				Prompt: "Hello",
				Options: map[string]any{
					"frequency_penalty": 0.0,
					"presence_penalty":  0.0,
					"temperature":       0.8,
					"top_p":             1.0,
					"stop":              []any{"\n", "stop"},
				},
				Suffix: "suffix",
				Stream: &True,
			},
		},
		{
			name: "completions handler stream with usage",
			body: `{
				"model": "test-model",
				"prompt": "Hello",
				"stream": true,
				"stream_options": {"include_usage": true},
				"temperature": 0.8,
				"stop": ["\n", "stop"],
				"suffix": "suffix"
			}`,
			req: api.GenerateRequest{
				Model:  "test-model",
				Prompt: "Hello",
				Options: map[string]any{
					"frequency_penalty": 0.0,
					"presence_penalty":  0.0,
					"temperature":       0.8,
					"top_p":             1.0,
					"stop":              []any{"\n", "stop"},
				},
				Suffix: "suffix",
				Stream: &True,
			},
		},
		{
			name: "completions handler error forwarding",
			body: `{
				"model": "test-model",
				"prompt": "Hello",
				"temperature": null,
				"stop": [1, 2],
				"suffix": "suffix"
			}`,
			err: openai.ErrorResponse{
				Error: openai.Error{
					Message: "invalid type for 'stop' field: float64",
					Type:    "invalid_request_error",
				},
			},
		},
	}

	endpoint := func(c *gin.Context) {
		c.Status(http.StatusOK)
	}

	gin.SetMode(gin.TestMode)
	router := gin.New()
	router.Use(CompletionsMiddleware(), captureRequestMiddleware(&capturedRequest))
	router.Handle(http.MethodPost, "/api/generate", endpoint)

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			req, _ := http.NewRequest(http.MethodPost, "/api/generate", strings.NewReader(tc.body))
			req.Header.Set("Content-Type", "application/json")

			resp := httptest.NewRecorder()
			router.ServeHTTP(resp, req)

			var errResp openai.ErrorResponse
			if resp.Code != http.StatusOK {
				if err := json.Unmarshal(resp.Body.Bytes(), &errResp); err != nil {
					t.Fatal(err)
				}
			}

			if capturedRequest != nil && !reflect.DeepEqual(tc.req, *capturedRequest) {
				t.Fatal("requests did not match")
			}

			if !reflect.DeepEqual(tc.err, errResp) {
				t.Fatal("errors did not match")
			}

			capturedRequest = nil
		})
	}
}

func TestEmbeddingsMiddleware(t *testing.T) {
	type testCase struct {
		name string
		body string
		req  api.EmbedRequest
		err  openai.ErrorResponse
	}

	var capturedRequest *api.EmbedRequest

	testCases := []testCase{
		{
			name: "embed handler single input",
			body: `{
				"input": "Hello",
				"model": "test-model"
			}`,
			req: api.EmbedRequest{
				Input: "Hello",
				Model: "test-model",
			},
		},
		{
			name: "embed handler batch input",
			body: `{
				"input": ["Hello", "World"],
				"model": "test-model"
			}`,
			req: api.EmbedRequest{
				Input: []any{"Hello", "World"},
				Model: "test-model",
			},
		},
		{
			name: "embed handler error forwarding",
			body: `{
				"model": "test-model"
			}`,
			err: openai.ErrorResponse{
				Error: openai.Error{
					Message: "invalid input",
					Type:    "invalid_request_error",
				},
			},
		},
	}

	endpoint := func(c *gin.Context) {
		c.Status(http.StatusOK)
	}

	gin.SetMode(gin.TestMode)
	router := gin.New()
	router.Use(EmbeddingsMiddleware(), captureRequestMiddleware(&capturedRequest))
	router.Handle(http.MethodPost, "/api/embed", endpoint)

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			req, _ := http.NewRequest(http.MethodPost, "/api/embed", strings.NewReader(tc.body))
			req.Header.Set("Content-Type", "application/json")

			resp := httptest.NewRecorder()
			router.ServeHTTP(resp, req)

			var errResp openai.ErrorResponse
			if resp.Code != http.StatusOK {
				if err := json.Unmarshal(resp.Body.Bytes(), &errResp); err != nil {
					t.Fatal(err)
				}
			}

			if capturedRequest != nil && !reflect.DeepEqual(tc.req, *capturedRequest) {
				t.Fatal("requests did not match")
			}

			if !reflect.DeepEqual(tc.err, errResp) {
				t.Fatal("errors did not match")
			}

			capturedRequest = nil
		})
	}
}

func TestListMiddleware(t *testing.T) {
	type testCase struct {
		name     string
		endpoint func(c *gin.Context)
		resp     string
	}

	testCases := []testCase{
		{
			name: "list handler",
			endpoint: func(c *gin.Context) {
				c.JSON(http.StatusOK, api.ListResponse{
					Models: []api.ListModelResponse{
						{
							Name:       "test-model",
							ModifiedAt: time.Unix(int64(1686935002), 0).UTC(),
						},
					},
				})
			},
			resp: `{
				"object": "list",
				"data": [
					{
						"id": "test-model",
						"object": "model",
						"created": 1686935002,
						"owned_by": "library"
					}
				]
			}`,
		},
		{
			name: "list handler empty output",
			endpoint: func(c *gin.Context) {
				c.JSON(http.StatusOK, api.ListResponse{})
			},
			resp: `{
				"object": "list",
				"data": null
			}`,
		},
	}

	gin.SetMode(gin.TestMode)

	for _, tc := range testCases {
		router := gin.New()
		router.Use(ListMiddleware())
		router.Handle(http.MethodGet, "/api/tags", tc.endpoint)
		req, _ := http.NewRequest(http.MethodGet, "/api/tags", nil)

		resp := httptest.NewRecorder()
		router.ServeHTTP(resp, req)

		var expected, actual map[string]any
		err := json.Unmarshal([]byte(tc.resp), &expected)
		if err != nil {
			t.Fatalf("failed to unmarshal expected response: %v", err)
		}

		err = json.Unmarshal(resp.Body.Bytes(), &actual)
		if err != nil {
			t.Fatalf("failed to unmarshal actual response: %v", err)
		}

		if !reflect.DeepEqual(expected, actual) {
			t.Errorf("responses did not match\nExpected: %+v\nActual: %+v", expected, actual)
		}
	}
}

func TestRetrieveMiddleware(t *testing.T) {
	type testCase struct {
		name     string
		endpoint func(c *gin.Context)
		resp     string
	}

	testCases := []testCase{
		{
			name: "retrieve handler",
			endpoint: func(c *gin.Context) {
				c.JSON(http.StatusOK, api.ShowResponse{
					ModifiedAt: time.Unix(int64(1686935002), 0).UTC(),
				})
			},
			resp: `{
				"id":"test-model",
				"object":"model",
				"created":1686935002,
				"owned_by":"library"}
			`,
		},
		{
			name: "retrieve handler error forwarding",
			endpoint: func(c *gin.Context) {
				c.JSON(http.StatusBadRequest, gin.H{"error": "model not found"})
			},
			resp: `{
				"error": {
				  "code": null,
				  "message": "model not found",
				  "param": null,
				  "type": "api_error"
				}
			}`,
		},
	}

	gin.SetMode(gin.TestMode)

	for _, tc := range testCases {
		router := gin.New()
		router.Use(RetrieveMiddleware())
		router.Handle(http.MethodGet, "/api/show/:model", tc.endpoint)
		req, _ := http.NewRequest(http.MethodGet, "/api/show/test-model", nil)

		resp := httptest.NewRecorder()
		router.ServeHTTP(resp, req)

		var expected, actual map[string]any
		err := json.Unmarshal([]byte(tc.resp), &expected)
		if err != nil {
			t.Fatalf("failed to unmarshal expected response: %v", err)
		}

		err = json.Unmarshal(resp.Body.Bytes(), &actual)
		if err != nil {
			t.Fatalf("failed to unmarshal actual response: %v", err)
		}

		if !reflect.DeepEqual(expected, actual) {
			t.Errorf("responses did not match\nExpected: %+v\nActual: %+v", expected, actual)
		}
	}
}

func TestImageGenerationsMiddleware(t *testing.T) {
	type testCase struct {
		name string
		body string
		req  api.GenerateRequest
		err  openai.ErrorResponse
	}

	var capturedRequest *api.GenerateRequest

	testCases := []testCase{
		{
			name: "image generation basic",
			body: `{
				"model": "test-model",
				"prompt": "a beautiful sunset"
			}`,
			req: api.GenerateRequest{
				Model:  "test-model",
				Prompt: "a beautiful sunset",
			},
		},
		{
			name: "image generation with size",
			body: `{
				"model": "test-model",
				"prompt": "a beautiful sunset",
				"size": "512x768"
			}`,
			req: api.GenerateRequest{
				Model:  "test-model",
				Prompt: "a beautiful sunset",
				Width:  512,
				Height: 768,
			},
		},
		{
			name: "image generation missing prompt",
			body: `{
				"model": "test-model"
			}`,
			err: openai.ErrorResponse{
				Error: openai.Error{
					Message: "prompt is required",
					Type:    "invalid_request_error",
				},
			},
		},
		{
			name: "image generation missing model",
			body: `{
				"prompt": "a beautiful sunset"
			}`,
			err: openai.ErrorResponse{
				Error: openai.Error{
					Message: "model is required",
					Type:    "invalid_request_error",
				},
			},
		},
	}

	endpoint := func(c *gin.Context) {
		c.Status(http.StatusOK)
	}

	gin.SetMode(gin.TestMode)
	router := gin.New()
	router.Use(ImageGenerationsMiddleware(), captureRequestMiddleware(&capturedRequest))
	router.Handle(http.MethodPost, "/api/generate", endpoint)

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			req, _ := http.NewRequest(http.MethodPost, "/api/generate", strings.NewReader(tc.body))
			req.Header.Set("Content-Type", "application/json")

			defer func() { capturedRequest = nil }()

			resp := httptest.NewRecorder()
			router.ServeHTTP(resp, req)

			if tc.err.Error.Message != "" {
				var errResp openai.ErrorResponse
				if err := json.Unmarshal(resp.Body.Bytes(), &errResp); err != nil {
					t.Fatal(err)
				}
				if diff := cmp.Diff(tc.err, errResp); diff != "" {
					t.Fatalf("errors did not match:\n%s", diff)
				}
				return
			}

			if resp.Code != http.StatusOK {
				t.Fatalf("expected status 200, got %d: %s", resp.Code, resp.Body.String())
			}

			if diff := cmp.Diff(&tc.req, capturedRequest); diff != "" {
				t.Fatalf("requests did not match:\n%s", diff)
			}
		})
	}
}

func TestImageWriterResponse(t *testing.T) {
	gin.SetMode(gin.TestMode)

	// Test that ImageWriter transforms GenerateResponse to OpenAI format
	endpoint := func(c *gin.Context) {
		resp := api.GenerateResponse{
			Model:     "test-model",
			CreatedAt: time.Unix(1234567890, 0).UTC(),
			Done:      true,
			Image:     "dGVzdC1pbWFnZS1kYXRh", // base64 of "test-image-data"
		}
		data, _ := json.Marshal(resp)
		c.Writer.Write(append(data, '\n'))
	}

	router := gin.New()
	router.Use(ImageGenerationsMiddleware())
	router.Handle(http.MethodPost, "/api/generate", endpoint)

	body := `{"model": "test-model", "prompt": "test"}`
	req, _ := http.NewRequest(http.MethodPost, "/api/generate", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")

	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	if resp.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d: %s", resp.Code, resp.Body.String())
	}

	var imageResp openai.ImageGenerationResponse
	if err := json.Unmarshal(resp.Body.Bytes(), &imageResp); err != nil {
		t.Fatalf("failed to unmarshal response: %v", err)
	}

	if imageResp.Created != 1234567890 {
		t.Errorf("expected created 1234567890, got %d", imageResp.Created)
	}

	if len(imageResp.Data) != 1 {
		t.Fatalf("expected 1 image, got %d", len(imageResp.Data))
	}

	if imageResp.Data[0].B64JSON != "dGVzdC1pbWFnZS1kYXRh" {
		t.Errorf("expected image data 'dGVzdC1pbWFnZS1kYXRh', got %s", imageResp.Data[0].B64JSON)
	}
}
