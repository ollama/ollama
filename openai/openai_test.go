package openai

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/ollama/ollama/api"
	"github.com/stretchr/testify/assert"
)

func TestMiddleware(t *testing.T) {
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
			Name:     "chat handler",
			Method:   http.MethodPost,
			Path:     "/api/chat",
			TestPath: "/api/chat",
			Handler:  Middleware,
			Endpoint: func(c *gin.Context) {
				var chatReq api.ChatRequest
				if err := c.ShouldBindJSON(&chatReq); err != nil {
					c.JSON(http.StatusBadRequest, gin.H{"error": "invalid request"})
					return
				}

				userMessage := chatReq.Messages[0].Content
				var assistantMessage string

				switch userMessage {
				case "Hello":
					assistantMessage = "Hello!"
				default:
					assistantMessage = "I'm not sure how to respond to that."
				}

				c.JSON(http.StatusOK, api.ChatResponse{
					Message: api.Message{
						Role:    "assistant",
						Content: assistantMessage,
					},
				})
			},
			Setup: func(t *testing.T, req *http.Request) {
				body := ChatCompletionRequest{
					Model:    "test-model",
					Messages: []Message{{Role: "user", Content: "Hello"}},
				}

				bodyBytes, _ := json.Marshal(body)

				req.Body = io.NopCloser(bytes.NewReader(bodyBytes))
				req.Header.Set("Content-Type", "application/json")
			},
			Expected: func(t *testing.T, resp *httptest.ResponseRecorder) {
				var chatResp ChatCompletion
				if err := json.NewDecoder(resp.Body).Decode(&chatResp); err != nil {
					t.Fatal(err)
				}

				if chatResp.Object != "chat.completion" {
					t.Fatalf("expected chat.completion, got %s", chatResp.Object)
				}

				if chatResp.Choices[0].Message.Content != "Hello!" {
					t.Fatalf("expected Hello!, got %s", chatResp.Choices[0].Message.Content)
				}
			},
		},
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
			Name:     "embedding handler (single embedding)",
			Method:   http.MethodPost,
			Path:     "/api/embeddings",
			TestPath: "/api/embeddings",
			Handler:  EmbeddingMiddleware,
			Endpoint: func(c *gin.Context) {
				c.JSON(http.StatusOK, api.EmbeddingResponse{
					Embedding: []float64{0.1, 0.2, 0.3},
				})
			},
			Setup: func(t *testing.T, req *http.Request) {
				body := EmbeddingRequest{
					Input: "Hello",
					Model: "test-model",
				}

				bodyBytes, _ := json.Marshal(body)

				req.Body = io.NopCloser(bytes.NewReader(bodyBytes))
				req.Header.Set("Content-Type", "application/json")
			},
			Expected: func(t *testing.T, resp *httptest.ResponseRecorder) {
				var embeddingResp EmbeddingList
				if err := json.NewDecoder(resp.Body).Decode(&embeddingResp); err != nil {
					t.Fatal(err)
				}

				if embeddingResp.Object != "list" {
					t.Fatalf("expected list, got %s", embeddingResp.Object)
				}

				if len(embeddingResp.Data) != 1 {
					t.Fatalf("expected 1 embedding, got %d", len(embeddingResp.Data))
				}

				if embeddingResp.Data[0].Object != "embedding" {
					t.Fatalf("expected embedding, got %s", embeddingResp.Data[0].Object)
				}

				if embeddingResp.Data[0].Embedding[0] != 0.1 {
					t.Fatalf("expected 0.1, got %f", embeddingResp.Data[0])
				}

				if embeddingResp.Model != "test-model" {
					t.Fatalf("expected test-model, got %s", embeddingResp.Model)
				}
			},
		},
		{
			Name:     "embedding handler (batch embedding)",
			Method:   http.MethodPost,
			Path:     "/api/embeddings",
			TestPath: "/api/embeddings",
			Handler:  EmbeddingMiddleware,
			Endpoint: func(c *gin.Context) {
				c.JSON(http.StatusOK, api.EmbeddingResponse{
					EmbeddingBatch: [][]float64{
						{0.1, 0.2, 0.3},
						{0.4, 0.5, 0.6},
						{0.7, 0.8, 0.9},
					},
				})
			},
			Setup: func(t *testing.T, req *http.Request) {
				body := EmbeddingRequest{
					Input: []string{"Hello", "World", "Ollama"},
					Model: "test-model",
				}

				bodyBytes, _ := json.Marshal(body)

				req.Body = io.NopCloser(bytes.NewReader(bodyBytes))
				req.Header.Set("Content-Type", "application/json")
			},
			Expected: func(t *testing.T, resp *httptest.ResponseRecorder) {
				var embeddingResp EmbeddingList
				if err := json.NewDecoder(resp.Body).Decode(&embeddingResp); err != nil {
					t.Fatal(err)
				}

				if embeddingResp.Object != "list" {
					t.Fatalf("expected list, got %s", embeddingResp.Object)
				}

				if len(embeddingResp.Data) != 3 {
					t.Fatalf("expected 3 embeddings, got %d", len(embeddingResp.Data))
				}

				if embeddingResp.Data[0].Object != "embedding" {
					t.Fatalf("expected embedding, got %s", embeddingResp.Data[0].Object)
				}

				if embeddingResp.Data[0].Embedding[0] != 0.1 {
					t.Fatalf("expected 0.1, got %f", embeddingResp.Data[0])
				}

				if embeddingResp.Model != "test-model" {
					t.Fatalf("expected test-model, got %s", embeddingResp.Model)
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
