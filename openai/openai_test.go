package openai

import (
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

func capture(req any) gin.HandlerFunc {
	return func(c *gin.Context) {
		body, _ := io.ReadAll(c.Request.Body)
		_ = json.Unmarshal(body, req)
		c.Next()
	}
}

func TestChatMiddleware(t *testing.T) {
	type test struct {
		name string
		body string
		req  api.ChatRequest
		err  ErrorResponse
	}

	tests := []test{
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
				Stream: func() *bool { f := false; return &f }(),
			},
		},
		{
			name: "chat handler with large context",
			body: `{
				"model": "test-model",
				"messages": [
					{"role": "user", "content": "Hello"}
				],
				"max_tokens": 16384
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

					// TODO (jmorganca): because we use a map[string]any for options
					// the values need to be floats for the test comparison to work.
					"num_predict": 16384.0,
					"num_ctx":     16384.0,
				},
				Stream: func() *bool { f := false; return &f }(),
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
									"url": "data:image/jpeg;base64,ZGF0YQo="
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
								img, _ := base64.StdEncoding.DecodeString("ZGF0YQo=")
								return img
							}(),
						},
					},
				},
				Options: map[string]any{
					"temperature": 1.0,
					"top_p":       1.0,
				},
				Stream: func() *bool { f := false; return &f }(),
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
				Stream: func() *bool { f := false; return &f }(),
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

	gin.SetMode(gin.TestMode)

	for _, tt := range tests {
		var req api.ChatRequest

		router := gin.New()
		router.Use(ChatMiddleware(), capture(&req))
		router.Handle(http.MethodPost, "/api/chat", func(c *gin.Context) {
			c.Status(http.StatusOK)
		})

		t.Run(tt.name, func(t *testing.T) {
			r, _ := http.NewRequest(http.MethodPost, "/api/chat", strings.NewReader(tt.body))
			r.Header.Set("Content-Type", "application/json")
			resp := httptest.NewRecorder()
			router.ServeHTTP(resp, r)

			var err ErrorResponse
			if resp.Code != http.StatusOK {
				if err := json.Unmarshal(resp.Body.Bytes(), &err); err != nil {
					t.Fatal(err)
				}
			}

			if diff := cmp.Diff(tt.req, req); diff != "" {
				t.Errorf("mismatch (-want +got):\n%s", diff)
			}

			if diff := cmp.Diff(tt.err, err); diff != "" {
				t.Errorf("mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestCompletionsMiddleware(t *testing.T) {
	type test struct {
		name string
		body string
		req  api.GenerateRequest
		err  ErrorResponse
	}

	tests := []test{
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
					"temperature":       1.6,
					"top_p":             1.0,
					"stop":              []any{"\n", "stop"},
				},
				Suffix: "suffix",
				Stream: func() *bool { f := false; return &f }(),
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

	gin.SetMode(gin.TestMode)

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var req api.GenerateRequest

			router := gin.New()
			router.Use(CompletionsMiddleware(), capture(&req))
			router.Handle(http.MethodPost, "/api/generate", func(c *gin.Context) {
				c.Status(http.StatusOK)
			})

			r, _ := http.NewRequest(http.MethodPost, "/api/generate", strings.NewReader(tt.body))
			r.Header.Set("Content-Type", "application/json")

			res := httptest.NewRecorder()
			router.ServeHTTP(res, r)

			var errResp ErrorResponse
			if res.Code != http.StatusOK {
				if err := json.Unmarshal(res.Body.Bytes(), &errResp); err != nil {
					t.Fatal(err)
				}
			}

			if !cmp.Equal(tt.req, req) {
				t.Fatalf("requests did not match:\n%s", cmp.Diff(tt.req, req))
			}

			if !cmp.Equal(tt.err, errResp) {
				t.Fatalf("errors did not match:\n%s", cmp.Diff(tt.err, errResp))
			}
		})
	}
}

func TestEmbeddingsMiddleware(t *testing.T) {
	type test struct {
		name string
		body string
		req  api.EmbedRequest
		err  ErrorResponse
	}

	tests := []test{
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

	for _, tt := range tests {
		var req api.EmbedRequest

		router := gin.New()
		router.Use(EmbeddingsMiddleware(), capture(&req))
		router.Handle(http.MethodPost, "/api/embed", endpoint)

		t.Run(tt.name, func(t *testing.T) {
			r, _ := http.NewRequest(http.MethodPost, "/api/embed", strings.NewReader(tt.body))
			r.Header.Set("Content-Type", "application/json")

			resp := httptest.NewRecorder()
			router.ServeHTTP(resp, r)

			var errResp ErrorResponse
			if resp.Code != http.StatusOK {
				if err := json.Unmarshal(resp.Body.Bytes(), &errResp); err != nil {
					t.Fatal(err)
				}
			}
			if diff := cmp.Diff(tt.req, req); diff != "" {
				t.Errorf("request mismatch (-want +got):\n%s", diff)
			}

			if diff := cmp.Diff(tt.err, errResp); diff != "" {
				t.Errorf("error mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestListMiddleware(t *testing.T) {
	type test struct {
		name    string
		handler gin.HandlerFunc
		body    string
	}

	tests := []test{
		{
			name: "list handler",
			handler: func(c *gin.Context) {
				c.JSON(http.StatusOK, api.ListResponse{
					Models: []api.ListModelResponse{
						{
							Name:       "test-model",
							ModifiedAt: time.Unix(int64(1686935002), 0).UTC(),
						},
					},
				})
			},
			body: `{
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
			handler: func(c *gin.Context) {
				c.JSON(http.StatusOK, api.ListResponse{
					Models: []api.ListModelResponse{},
				})
			},
			body: `{
				"object": "list",
				"data": null
			}`,
		},
	}

	gin.SetMode(gin.TestMode)

	for _, tt := range tests {
		router := gin.New()
		router.Use(ListMiddleware())
		router.Handle(http.MethodGet, "/api/tags", tt.handler)
		req, _ := http.NewRequest(http.MethodGet, "/api/tags", nil)

		resp := httptest.NewRecorder()
		router.ServeHTTP(resp, req)

		var expected, actual map[string]any
		err := json.Unmarshal([]byte(tt.body), &expected)
		if err != nil {
			t.Fatalf("failed to unmarshal expected response: %v", err)
		}

		err = json.Unmarshal(resp.Body.Bytes(), &actual)
		if err != nil {
			t.Fatalf("failed to unmarshal actual response: %v", err)
		}

		if diff := cmp.Diff(expected, actual); diff != "" {
			t.Errorf("responses did not match (-want +got):\n%s", diff)
		}
	}
}

func TestRetrieveMiddleware(t *testing.T) {
	type test struct {
		name    string
		handler gin.HandlerFunc
		body    string
	}

	tests := []test{
		{
			name: "retrieve handler",
			handler: func(c *gin.Context) {
				c.JSON(http.StatusOK, api.ShowResponse{
					ModifiedAt: time.Unix(int64(1686935002), 0).UTC(),
				})
			},
			body: `{
				"id":"test-model",
				"object":"model",
				"created":1686935002,
				"owned_by":"library"}
			`,
		},
		{
			name: "retrieve handler error forwarding",
			handler: func(c *gin.Context) {
				c.JSON(http.StatusBadRequest, gin.H{"error": "model not found"})
			},
			body: `{
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

	for _, tt := range tests {
		router := gin.New()
		router.Use(RetrieveMiddleware())
		router.Handle(http.MethodGet, "/api/show/:model", tt.handler)
		req, _ := http.NewRequest(http.MethodGet, "/api/show/test-model", nil)

		resp := httptest.NewRecorder()
		router.ServeHTTP(resp, req)

		var expected, actual map[string]any
		err := json.Unmarshal([]byte(tt.body), &expected)
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
