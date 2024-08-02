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
	"github.com/stretchr/testify/assert"

	"github.com/ollama/ollama/api"
)

const (
	prefix   = `data:image/jpeg;base64,`
	image    = `iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=`
	imageURL = prefix + image
)

func prepareRequest(req *http.Request, body any) {
	bodyBytes, _ := json.Marshal(body)
	req.Body = io.NopCloser(bytes.NewReader(bodyBytes))
	req.Header.Set("Content-Type", "application/json")
}

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
		Setup    func(t *testing.T, req *http.Request)
		Expected func(t *testing.T, req *api.ChatRequest, resp *httptest.ResponseRecorder)
	}

	var capturedRequest *api.ChatRequest

	testCases := []testCase{
		{
			Name: "chat handler",
			Setup: func(t *testing.T, req *http.Request) {
				body := ChatCompletionRequest{
					Model:    "test-model",
					Messages: []Message{{Role: "user", Content: "Hello"}},
				}
				prepareRequest(req, body)
			},
			Expected: func(t *testing.T, req *api.ChatRequest, resp *httptest.ResponseRecorder) {
				if resp.Code != http.StatusOK {
					t.Fatalf("expected 200, got %d", resp.Code)
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
			Setup: func(t *testing.T, req *http.Request) {
				body := ChatCompletionRequest{
					Model: "test-model",
					Messages: []Message{
						{
							Role: "user", Content: []map[string]any{
								{"type": "text", "text": "Hello"},
								{"type": "image_url", "image_url": map[string]string{"url": imageURL}},
							},
						},
					},
				}
				prepareRequest(req, body)
			},
			Expected: func(t *testing.T, req *api.ChatRequest, resp *httptest.ResponseRecorder) {
				if resp.Code != http.StatusOK {
					t.Fatalf("expected 200, got %d", resp.Code)
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
			Setup: func(t *testing.T, req *http.Request) {
				body := ChatCompletionRequest{
					Model: "test-model",
					Messages: []Message{
						{Role: "user", Content: "What's the weather like in Paris Today?"},
						{Role: "assistant", ToolCalls: []ToolCall{{
							ID:   "id",
							Type: "function",
							Function: struct {
								Name      string `json:"name"`
								Arguments string `json:"arguments"`
							}{
								Name:      "get_current_weather",
								Arguments: "{\"location\": \"Paris, France\", \"format\": \"celsius\"}",
							},
						}}},
					},
				}
				prepareRequest(req, body)
			},
			Expected: func(t *testing.T, req *api.ChatRequest, resp *httptest.ResponseRecorder) {
				if resp.Code != 200 {
					t.Fatalf("expected 200, got %d", resp.Code)
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
			Setup: func(t *testing.T, req *http.Request) {
				body := ChatCompletionRequest{
					Model:    "test-model",
					Messages: []Message{{Role: "user", Content: 2}},
				}
				prepareRequest(req, body)
			},
			Expected: func(t *testing.T, req *api.ChatRequest, resp *httptest.ResponseRecorder) {
				if resp.Code != http.StatusBadRequest {
					t.Fatalf("expected 400, got %d", resp.Code)
				}

				if !strings.Contains(resp.Body.String(), "invalid message content type") {
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
	router.Use(ChatMiddleware(), captureRequestMiddleware(&capturedRequest))
	router.Handle(http.MethodPost, "/api/chat", endpoint)

	for _, tc := range testCases {
		t.Run(tc.Name, func(t *testing.T) {
			req, _ := http.NewRequest(http.MethodPost, "/api/chat", nil)

			tc.Setup(t, req)

			resp := httptest.NewRecorder()
			router.ServeHTTP(resp, req)

			tc.Expected(t, capturedRequest, resp)

			capturedRequest = nil
		})
	}
}

func TestCompletionsMiddleware(t *testing.T) {
	type testCase struct {
		Name     string
		Setup    func(t *testing.T, req *http.Request)
		Expected func(t *testing.T, req *api.GenerateRequest, resp *httptest.ResponseRecorder)
	}

	var capturedRequest *api.GenerateRequest

	testCases := []testCase{
		{
			Name: "completions handler",
			Setup: func(t *testing.T, req *http.Request) {
				temp := float32(0.8)
				body := CompletionRequest{
					Model:       "test-model",
					Prompt:      "Hello",
					Temperature: &temp,
					Stop:        []string{"\n", "stop"},
					Suffix:      "suffix",
				}
				prepareRequest(req, body)
			},
			Expected: func(t *testing.T, req *api.GenerateRequest, resp *httptest.ResponseRecorder) {
				if req.Prompt != "Hello" {
					t.Fatalf("expected 'Hello', got %s", req.Prompt)
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
			Setup: func(t *testing.T, req *http.Request) {
				body := CompletionRequest{
					Model:       "test-model",
					Prompt:      "Hello",
					Temperature: nil,
					Stop:        []int{1, 2},
					Suffix:      "suffix",
				}
				prepareRequest(req, body)
			},
			Expected: func(t *testing.T, req *api.GenerateRequest, resp *httptest.ResponseRecorder) {
				if resp.Code != http.StatusBadRequest {
					t.Fatalf("expected 400, got %d", resp.Code)
				}

				if !strings.Contains(resp.Body.String(), "invalid type for 'stop' field") {
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
			req, _ := http.NewRequest(http.MethodPost, "/api/generate", nil)

			tc.Setup(t, req)

			resp := httptest.NewRecorder()
			router.ServeHTTP(resp, req)

			tc.Expected(t, capturedRequest, resp)

			capturedRequest = nil
		})
	}
}

func TestEmbeddingsMiddleware(t *testing.T) {
	type testCase struct {
		Name     string
		Setup    func(t *testing.T, req *http.Request)
		Expected func(t *testing.T, req *api.EmbedRequest, resp *httptest.ResponseRecorder)
	}

	var capturedRequest *api.EmbedRequest

	testCases := []testCase{
		{
			Name: "embed handler single input",
			Setup: func(t *testing.T, req *http.Request) {
				body := EmbedRequest{
					Input: "Hello",
					Model: "test-model",
				}
				prepareRequest(req, body)
			},
			Expected: func(t *testing.T, req *api.EmbedRequest, resp *httptest.ResponseRecorder) {
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
			Setup: func(t *testing.T, req *http.Request) {
				body := EmbedRequest{
					Input: []string{"Hello", "World"},
					Model: "test-model",
				}
				prepareRequest(req, body)
			},
			Expected: func(t *testing.T, req *api.EmbedRequest, resp *httptest.ResponseRecorder) {
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
			Setup: func(t *testing.T, req *http.Request) {
				body := EmbedRequest{
					Model: "test-model",
				}
				prepareRequest(req, body)
			},
			Expected: func(t *testing.T, req *api.EmbedRequest, resp *httptest.ResponseRecorder) {
				if resp.Code != http.StatusBadRequest {
					t.Fatalf("expected 400, got %d", resp.Code)
				}

				if !strings.Contains(resp.Body.String(), "invalid input") {
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
			req, _ := http.NewRequest(http.MethodPost, "/api/embed", nil)

			tc.Setup(t, req)

			resp := httptest.NewRecorder()
			router.ServeHTTP(resp, req)

			tc.Expected(t, capturedRequest, resp)

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
		Setup    func(t *testing.T, req *http.Request)
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

			if tc.Setup != nil {
				tc.Setup(t, req)
			}

			resp := httptest.NewRecorder()
			router.ServeHTTP(resp, req)

			assert.Equal(t, http.StatusOK, resp.Code)

			tc.Expected(t, resp)
		})
	}
}
