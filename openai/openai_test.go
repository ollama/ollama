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
	"github.com/ollama/ollama/api"
	"github.com/stretchr/testify/assert"
)

const prefix = `data:image/jpeg;base64,`
const image = `iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=`

var False = false

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
					"temperature":       1.6,
					"top_p":             1.0,
					"stop":              []any{"\n", "stop"},
				},
				Suffix: "suffix",
				Stream: &False,
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

func TestMiddlewareResponses(t *testing.T) {
	type testCase struct {
		Name     string
		Method   string
		Path     string
		TestPath string
		Handler  func() gin.HandlerFunc
		Endpoint func(c *gin.Context)
		Expected func(t *testing.T, resp *httptest.ResponseRecorder)
	}

	testCases := []testCase{
		{
			Name:     "list handler",
			Method:   http.MethodGet,
			Path:     "/api/tags",
			TestPath: "/api/tags",
			Handler:  ListMiddleware,
			Endpoint: func(c *gin.Context) {
				c.JSON(http.StatusOK, api.ListResponse{
					Models: []api.ListModelResponse{
						{
							Name: "Test Model",
						},
					},
				})
			},
			Expected: func(t *testing.T, resp *httptest.ResponseRecorder) {
				var listResp ListCompletion
				if err := json.NewDecoder(resp.Body).Decode(&listResp); err != nil {
					t.Fatal(err)
				}

				if listResp.Object != "list" {
					t.Fatalf("expected list, got %s", listResp.Object)
				}

				if len(listResp.Data) != 1 {
					t.Fatalf("expected 1, got %d", len(listResp.Data))
				}

				if listResp.Data[0].Id != "Test Model" {
					t.Fatalf("expected Test Model, got %s", listResp.Data[0].Id)
				}
			},
		},
		{
			Name:     "retrieve model",
			Method:   http.MethodGet,
			Path:     "/api/show/:model",
			TestPath: "/api/show/test-model",
			Handler:  RetrieveMiddleware,
			Endpoint: func(c *gin.Context) {
				c.JSON(http.StatusOK, api.ShowResponse{
					ModifiedAt: time.Date(2024, 6, 17, 13, 45, 0, 0, time.UTC),
				})
			},
			Expected: func(t *testing.T, resp *httptest.ResponseRecorder) {
				var retrieveResp Model
				if err := json.NewDecoder(resp.Body).Decode(&retrieveResp); err != nil {
					t.Fatal(err)
				}

				if retrieveResp.Object != "model" {
					t.Fatalf("Expected object to be model, got %s", retrieveResp.Object)
				}

				if retrieveResp.Id != "test-model" {
					t.Fatalf("Expected id to be test-model, got %s", retrieveResp.Id)
				}
			},
		},
	}

	gin.SetMode(gin.TestMode)
	router := gin.New()

	for _, tc := range testCases {
		t.Run(tc.Name, func(t *testing.T) {
			router = gin.New()
			router.Use(tc.Handler())
			router.Handle(tc.Method, tc.Path, tc.Endpoint)
			req, _ := http.NewRequest(tc.Method, tc.TestPath, nil)

			resp := httptest.NewRecorder()
			router.ServeHTTP(resp, req)

			assert.Equal(t, http.StatusOK, resp.Code)

			tc.Expected(t, resp)
		})
	}
}
