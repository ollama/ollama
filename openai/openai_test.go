package openai

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/ollama/ollama/api"
	"github.com/stretchr/testify/assert"
)

const prefix = `data:image/jpeg;base64,`
const image = `iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=`
const imageURL = prefix + image

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
		Name     string
		Input    string
		Expected func(t *testing.T, req *api.ChatRequest, err *ErrorResponse)
	}

	var capturedRequest *api.ChatRequest

	testCases := []testCase{
		{
			Name: "chat handler",
			Input: `{
				"model": "test-model",
				"messages": [
					{"role": "user", "content": "Hello"}
				]
			}`,
			Expected: func(t *testing.T, req *api.ChatRequest, err *ErrorResponse) {
				if err != nil {
					t.Fatalf("expected no error, got %s", err.Error.Message)
				}

				if req.Model != "test-model" {
					t.Fatalf("expected 'test-model', got %s", req.Model)
				}

				if req.Messages[0].Role != "user" {
					t.Fatalf("expected 'user', got %s", req.Messages[0].Role)
				}

				if req.Messages[0].Content != "Hello" {
					t.Fatalf("expected 'Hello', got %s", req.Messages[0].Content)
				}
			},
		},
		{
			Name: "chat handler with image content",
			Input: `{
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
									"url": "` + imageURL + `"
								}
							}
						]
					}
				]
			}`,
			Expected: func(t *testing.T, req *api.ChatRequest, err *ErrorResponse) {
				if err != nil {
					t.Fatalf("expected no error, got %s", err.Error.Message)
				}

				if req.Messages[0].Role != "user" {
					t.Fatalf("expected 'user', got %s", req.Messages[0].Role)
				}

				if req.Messages[0].Content != "Hello" {
					t.Fatalf("expected 'Hello', got %s", req.Messages[0].Content)
				}

				img, _ := base64.StdEncoding.DecodeString(imageURL[len(prefix):])

				if req.Messages[1].Role != "user" {
					t.Fatalf("expected 'user', got %s", req.Messages[1].Role)
				}

				if !bytes.Equal(req.Messages[1].Images[0], img) {
					t.Fatalf("expected image encoding, got %s", req.Messages[1].Images[0])
				}
			},
		},
		{
			Name: "chat handler with tools",
			Input: `{
				"model": "test-model",
				"messages": [
					{"role": "user", "content": "What's the weather like in Paris Today?"},
					{"role": "assistant", "tool_calls": [{"id": "id", "type": "function", "function": {"name": "get_current_weather", "arguments": "{\"location\": \"Paris, France\", \"format\": \"celsius\"}"}}]}
				]
			}`,
			Expected: func(t *testing.T, req *api.ChatRequest, err *ErrorResponse) {
				if err != nil {
					t.Fatalf("expected no error, got %s", err.Error.Message)
				}

				if req.Messages[0].Content != "What's the weather like in Paris Today?" {
					t.Fatalf("expected What's the weather like in Paris Today?, got %s", req.Messages[0].Content)
				}

				if req.Messages[1].ToolCalls[0].Function.Arguments["location"] != "Paris, France" {
					t.Fatalf("expected 'Paris, France', got %v", req.Messages[1].ToolCalls[0].Function.Arguments["location"])
				}

				if req.Messages[1].ToolCalls[0].Function.Arguments["format"] != "celsius" {
					t.Fatalf("expected celsius, got %v", req.Messages[1].ToolCalls[0].Function.Arguments["format"])
				}
			},
		},
		{
			Name: "chat handler error forwarding",
			Input: `{
				"model": "test-model",
				"messages": [
					{"role": "user", "content": 2}
				]
			}`,
			Expected: func(t *testing.T, req *api.ChatRequest, err *ErrorResponse) {
				if err == nil {
					t.Fatal("expected error, got nil")
				}

				if !strings.Contains(err.Error.Message, "invalid message content type") {
					t.Fatal("error was not forwarded")
				}
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
		t.Run(tc.Name, func(t *testing.T) {
			req, _ := http.NewRequest(http.MethodPost, "/api/chat", strings.NewReader(tc.Input))
			req.Header.Set("Content-Type", "application/json")

			resp := httptest.NewRecorder()
			router.ServeHTTP(resp, req)

			var errResp *ErrorResponse
			if resp.Code != http.StatusOK {
				if err := json.Unmarshal(resp.Body.Bytes(), &errResp); err != nil {
					t.Fatal(err)
				}
			}

			tc.Expected(t, capturedRequest, errResp)
			capturedRequest = nil
		})
	}
}

func TestCompletionsMiddleware(t *testing.T) {
	type testCase struct {
		Name     string
		Input    string
		Expected func(t *testing.T, req *api.GenerateRequest, err *ErrorResponse)
	}

	var capturedRequest *api.GenerateRequest

	testCases := []testCase{
		{
			Name: "completions handler",
			Input: `{
				"model": "test-model",
				"prompt": "Hello",
				"temperature": 0.8,
				"stop": ["\n", "stop"],
				"suffix": "suffix"
			}`,
			Expected: func(t *testing.T, req *api.GenerateRequest, err *ErrorResponse) {
				if err != nil {
					t.Fatalf("expected no error, got %s", err.Error.Message)
				}

				if req.Options["temperature"] != 1.6 {
					t.Fatalf("expected 1.6, got %f", req.Options["temperature"])
				}

				stopTokens, ok := req.Options["stop"].([]any)

				if !ok {
					t.Fatalf("expected stop tokens to be a list")
				}

				if stopTokens[0] != "\n" || stopTokens[1] != "stop" {
					t.Fatalf("expected ['\\n', 'stop'], got %v", stopTokens)
				}

				if req.Suffix != "suffix" {
					t.Fatalf("expected 'suffix', got %s", req.Suffix)
				}
			},
		},
		{
			Name: "completions handler error forwarding",
			Input: `{
				"model": "test-model",
				"prompt": "Hello",
				"temperature": null,
				"stop": [1, 2],
				"suffix": "suffix"
			}`,
			Expected: func(t *testing.T, req *api.GenerateRequest, err *ErrorResponse) {
				if err == nil {
					t.Fatal("expected error, got nil")
				}

				if !strings.Contains(err.Error.Message, "invalid type for 'stop' field") {
					t.Fatalf("error was not forwarded")
				}
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
		t.Run(tc.Name, func(t *testing.T) {
			req, _ := http.NewRequest(http.MethodPost, "/api/generate", strings.NewReader(tc.Input))
			req.Header.Set("Content-Type", "application/json")

			resp := httptest.NewRecorder()
			router.ServeHTTP(resp, req)

			var errResp *ErrorResponse
			if resp.Code != http.StatusOK {
				if err := json.Unmarshal(resp.Body.Bytes(), &errResp); err != nil {
					t.Fatal(err)
				}
			}

			tc.Expected(t, capturedRequest, errResp)
			capturedRequest = nil
		})
	}
}

func TestEmbeddingsMiddleware(t *testing.T) {
	type testCase struct {
		Name     string
		Input    string
		Expected func(t *testing.T, req *api.EmbedRequest, err *ErrorResponse)
	}

	var capturedRequest *api.EmbedRequest

	testCases := []testCase{
		{
			Name: "embed handler single input",
			Input: `{
				"input": "Hello",
				"model": "test-model"
			}`,
			Expected: func(t *testing.T, req *api.EmbedRequest, err *ErrorResponse) {
				if err != nil {
					t.Fatalf("expected no error, got %s", err.Error.Message)
				}

				if req.Input != "Hello" {
					t.Fatalf("expected 'Hello', got %s", req.Input)
				}

				if req.Model != "test-model" {
					t.Fatalf("expected 'test-model', got %s", req.Model)
				}
			},
		},
		{
			Name: "embed handler batch input",
			Input: `{
				"input": ["Hello", "World"],
				"model": "test-model"
			}`,
			Expected: func(t *testing.T, req *api.EmbedRequest, err *ErrorResponse) {
				if err != nil {
					t.Fatalf("expected no error, got %s", err.Error.Message)
				}

				input, ok := req.Input.([]any)

				if !ok {
					t.Fatalf("expected input to be a list")
				}

				if input[0].(string) != "Hello" {
					t.Fatalf("expected 'Hello', got %s", input[0])
				}

				if input[1].(string) != "World" {
					t.Fatalf("expected 'World', got %s", input[1])
				}

				if req.Model != "test-model" {
					t.Fatalf("expected 'test-model', got %s", req.Model)
				}
			},
		},
		{
			Name: "embed handler error forwarding",
			Input: `{
				"model": "test-model"
			}`,
			Expected: func(t *testing.T, req *api.EmbedRequest, err *ErrorResponse) {
				if err == nil {
					t.Fatal("expected error, got nil")
				}

				if !strings.Contains(err.Error.Message, "invalid input") {
					t.Fatalf("error was not forwarded")
				}
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
		t.Run(tc.Name, func(t *testing.T) {
			req, _ := http.NewRequest(http.MethodPost, "/api/embed", strings.NewReader(tc.Input))
			req.Header.Set("Content-Type", "application/json")

			resp := httptest.NewRecorder()
			router.ServeHTTP(resp, req)

			var errResp *ErrorResponse
			if resp.Code != http.StatusOK {
				if err := json.Unmarshal(resp.Body.Bytes(), &errResp); err != nil {
					t.Fatal(err)
				}
			}

			tc.Expected(t, capturedRequest, errResp)
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
