package openai

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
)

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
		err  ErrorResponse
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
								Function: api.ToolCallFunction{
									Name: "get_current_weather",
									Arguments: map[string]interface{}{
										"location": "Paris, France",
										"format":   "celsius",
									},
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
							Parameters: struct {
								Type       string   `json:"type"`
								Required   []string `json:"required"`
								Properties map[string]struct {
									Type        string   `json:"type"`
									Description string   `json:"description"`
									Enum        []string `json:"enum,omitempty"`
								} `json:"properties"`
							}{
								Type:     "object",
								Required: []string{"location"},
								Properties: map[string]struct {
									Type        string   `json:"type"`
									Description string   `json:"description"`
									Enum        []string `json:"enum,omitempty"`
								}{
									"location": {
										Type:        "string",
										Description: "The city and state",
									},
									"unit": {
										Type: "string",
										Enum: []string{"celsius", "fahrenheit"},
									},
								},
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
			err: ErrorResponse{
				Error: Error{
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

			var errResp ErrorResponse
			if resp.Code != http.StatusOK {
				if err := json.Unmarshal(resp.Body.Bytes(), &errResp); err != nil {
					t.Fatal(err)
				}
				return
			}
			if diff := cmp.Diff(&tc.req, capturedRequest); diff != "" {
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
		err  ErrorResponse
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
			err: ErrorResponse{
				Error: Error{
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

			var errResp ErrorResponse
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
		err  ErrorResponse
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
			err: ErrorResponse{
				Error: Error{
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

			var errResp ErrorResponse
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

// Test generated using Keploy
func TestToolCallId_UniqueID(t *testing.T) {
    id1 := toolCallId()
    id2 := toolCallId()

    if !strings.HasPrefix(id1, "call_") || len(id1) != 13 {
        t.Errorf("Expected ID to have prefix 'call_' and length 13, got %s", id1)
    }

    if !strings.HasPrefix(id2, "call_") || len(id2) != 13 {
        t.Errorf("Expected ID to have prefix 'call_' and length 13, got %s", id2)
    }

    if id1 == id2 {
        t.Errorf("Expected unique IDs, but got identical IDs: %s and %s", id1, id2)
    }
}


// Test generated using Keploy
func TestToToolCalls_Conversion(t *testing.T) {
    input := []api.ToolCall{
        {
            Function: api.ToolCallFunction{
                Name:      "testFunction",
                Arguments: map[string]interface{}{"key": "value"},
            },
        },
    }

    result := toToolCalls(input)

    if len(result) != 1 {
        t.Fatalf("Expected 1 ToolCall, got %d", len(result))
    }

    if result[0].Function.Name != "testFunction" {
        t.Errorf("Expected function name 'testFunction', got %s", result[0].Function.Name)
    }

    expectedArgs := `{"key":"value"}`
    if result[0].Function.Arguments != expectedArgs {
        t.Errorf("Expected arguments %s, got %s", expectedArgs, result[0].Function.Arguments)
    }

    if !strings.HasPrefix(result[0].ID, "call_") {
        t.Errorf("Expected ID to have prefix 'call_', got %s", result[0].ID)
    }
}


// Test generated using Keploy
func TestToChunk_Conversion(t *testing.T) {
    input := api.ChatResponse{
        Model: "test-model",
        Message: api.Message{
            Content: "Hello",
            ToolCalls: []api.ToolCall{
                {
                    Function: api.ToolCallFunction{
                        Name:      "testFunction",
                        Arguments: map[string]interface{}{"key": "value"},
                    },
                },
            },
        },
        DoneReason: "completed",
    }

    result := toChunk("test-id", input)

    if result.Id != "test-id" {
        t.Errorf("Expected ID 'test-id', got %s", result.Id)
    }

    if result.Model != "test-model" {
        t.Errorf("Expected model 'test-model', got %s", result.Model)
    }

    if len(result.Choices) != 1 {
        t.Fatalf("Expected 1 choice, got %d", len(result.Choices))
    }

    if result.Choices[0].Delta.Content != "Hello" {
        t.Errorf("Expected content 'Hello', got %s", result.Choices[0].Delta.Content)
    }

    if result.Choices[0].FinishReason == nil || *result.Choices[0].FinishReason != "completed" {
        t.Errorf("Expected finish reason 'completed', got %v", result.Choices[0].FinishReason)
    }

    if len(result.Choices[0].Delta.ToolCalls) != 1 {
        t.Fatalf("Expected 1 tool call, got %d", len(result.Choices[0].Delta.ToolCalls))
    }

    if result.Choices[0].Delta.ToolCalls[0].Function.Name != "testFunction" {
        t.Errorf("Expected tool call function name 'testFunction', got %s", result.Choices[0].Delta.ToolCalls[0].Function.Name)
    }
}

// Test generated using Keploy
func TestToChunk_NoDoneReasonOrToolCalls(t *testing.T) {
    input := api.ChatResponse{
        Model: "test-model",
        Message: api.Message{
            Content: "Hello",
            ToolCalls: nil,
        },
        DoneReason: "",
    }

    result := toChunk("test-id", input)

    if result.Id != "test-id" {
        t.Errorf("Expected ID 'test-id', got %s", result.Id)
    }

    if result.Model != "test-model" {
        t.Errorf("Expected model 'test-model', got %s", result.Model)
    }

    if len(result.Choices) != 1 {
        t.Fatalf("Expected 1 choice, got %d", len(result.Choices))
    }

    if result.Choices[0].Delta.Content != "Hello" {
        t.Errorf("Expected content 'Hello', got %s", result.Choices[0].Delta.Content)
    }

    if result.Choices[0].FinishReason != nil {
        t.Errorf("Expected nil finish reason, got %v", result.Choices[0].FinishReason)
    }
}


// Test generated using Keploy
func TestToChatCompletion_ToolCallsFinishReason(t *testing.T) {
    input := api.ChatResponse{
        Model: "test-model",
        CreatedAt: time.Now(),
        Message: api.Message{
            Role:    "assistant",
            Content: "Hello",
            ToolCalls: []api.ToolCall{
                {
                    Function: api.ToolCallFunction{
                        Name:      "testFunction",
                        Arguments: map[string]interface{}{"key": "value"},
                    },
                },
            },
        },
        DoneReason: "",
    }

    result := toChatCompletion("test-id", input)

    if result.Id != "test-id" {
        t.Errorf("Expected ID 'test-id', got %s", result.Id)
    }

    if result.Model != "test-model" {
        t.Errorf("Expected model 'test-model', got %s", result.Model)
    }

    if len(result.Choices) != 1 {
        t.Fatalf("Expected 1 choice, got %d", len(result.Choices))
    }

    if result.Choices[0].FinishReason == nil || *result.Choices[0].FinishReason != "tool_calls" {
        t.Errorf("Expected finish reason 'tool_calls', got %v", result.Choices[0].FinishReason)
    }
}


