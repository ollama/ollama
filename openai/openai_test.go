package openai

import (
	"bytes"
	"encoding/json"
	"fmt"
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
			Handler:  ChatMiddleware,
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
				assert.Equal(t, http.StatusOK, resp.Code)

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
			Name:     "completions handler",
			Method:   http.MethodPost,
			Path:     "/api/generate",
			TestPath: "/api/generate",
			Handler:  CompletionsMiddleware,
			Endpoint: func(c *gin.Context) {
				c.JSON(http.StatusOK, api.GenerateResponse{
					Response: "Hello!",
				})
			},
			Setup: func(t *testing.T, req *http.Request) {
				body := CompletionRequest{
					Model:  "test-model",
					Prompt: "Hello",
				}

				bodyBytes, _ := json.Marshal(body)

				req.Body = io.NopCloser(bytes.NewReader(bodyBytes))
				req.Header.Set("Content-Type", "application/json")
			},
			Expected: func(t *testing.T, resp *httptest.ResponseRecorder) {
				assert.Equal(t, http.StatusOK, resp.Code)
				var completionResp Completion
				if err := json.NewDecoder(resp.Body).Decode(&completionResp); err != nil {
					t.Fatal(err)
				}

				if completionResp.Object != "text_completion" {
					t.Fatalf("expected text_completion, got %s", completionResp.Object)
				}

				if completionResp.Choices[0].Text != "Hello!" {
					t.Fatalf("expected Hello!, got %s", completionResp.Choices[0].Text)
				}
			},
		},
		{
			Name:     "completions handler with params",
			Method:   http.MethodPost,
			Path:     "/api/generate",
			TestPath: "/api/generate",
			Handler:  CompletionsMiddleware,
			Endpoint: func(c *gin.Context) {
				var generateReq api.GenerateRequest
				if err := c.ShouldBindJSON(&generateReq); err != nil {
					c.JSON(http.StatusBadRequest, gin.H{"error": "invalid request"})
					return
				}

				temperature := generateReq.Options["temperature"].(float64)
				var assistantMessage string

				switch temperature {
				case 1.6:
					assistantMessage = "Received temperature of 1.6"
				default:
					assistantMessage = fmt.Sprintf("Received temperature of %f", temperature)
				}

				c.JSON(http.StatusOK, api.GenerateResponse{
					Response: assistantMessage,
				})
			},
			Setup: func(t *testing.T, req *http.Request) {
				temp := float32(0.8)
				body := CompletionRequest{
					Model:       "test-model",
					Prompt:      "Hello",
					Temperature: &temp,
				}

				bodyBytes, _ := json.Marshal(body)

				req.Body = io.NopCloser(bytes.NewReader(bodyBytes))
				req.Header.Set("Content-Type", "application/json")
			},
			Expected: func(t *testing.T, resp *httptest.ResponseRecorder) {
				assert.Equal(t, http.StatusOK, resp.Code)
				var completionResp Completion
				if err := json.NewDecoder(resp.Body).Decode(&completionResp); err != nil {
					t.Fatal(err)
				}

				if completionResp.Object != "text_completion" {
					t.Fatalf("expected text_completion, got %s", completionResp.Object)
				}

				if completionResp.Choices[0].Text != "Received temperature of 1.6" {
					t.Fatalf("expected Received temperature of 1.6, got %s", completionResp.Choices[0].Text)
				}
			},
		},
		{
			Name:     "completions handler with error",
			Method:   http.MethodPost,
			Path:     "/api/generate",
			TestPath: "/api/generate",
			Handler:  CompletionsMiddleware,
			Endpoint: func(c *gin.Context) {
				c.JSON(http.StatusBadRequest, gin.H{"error": "invalid request"})
			},
			Setup: func(t *testing.T, req *http.Request) {
				body := CompletionRequest{
					Model:  "test-model",
					Prompt: "Hello",
				}

				bodyBytes, _ := json.Marshal(body)

				req.Body = io.NopCloser(bytes.NewReader(bodyBytes))
				req.Header.Set("Content-Type", "application/json")
			},
			Expected: func(t *testing.T, resp *httptest.ResponseRecorder) {
				if resp.Code != http.StatusBadRequest {
					t.Fatalf("expected 400, got %d", resp.Code)
				}

				if !strings.Contains(resp.Body.String(), `"invalid request"`) {
					t.Fatalf("error was not forwarded")
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
				assert.Equal(t, http.StatusOK, resp.Code)

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

			tc.Expected(t, resp)
		})
	}
}
