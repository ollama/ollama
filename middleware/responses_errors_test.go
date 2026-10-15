package middleware

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/openai"
)

func assertResponsesAuthenticationError(t *testing.T, body string, stream bool, signinURL string) {
	t.Helper()
	data := []byte(body)
	if stream {
		data = nil
		for _, line := range strings.Split(body, "\n") {
			if !strings.HasPrefix(line, "data: ") {
				continue
			}
			var event struct {
				Type     string          `json:"type"`
				Response json.RawMessage `json:"response"`
			}
			if err := json.Unmarshal([]byte(strings.TrimPrefix(line, "data: ")), &event); err != nil {
				t.Fatal(err)
			}
			if event.Type == "response.completed" {
				t.Fatalf("authentication failure emitted completion: %s", body)
			}
			if event.Type == "response.failed" {
				if data != nil {
					t.Fatalf("multiple terminal failures: %s", body)
				}
				data = event.Response
			}
		}
	}
	var response openai.ResponsesErrorResponse
	if err := json.Unmarshal(data, &response); err != nil {
		t.Fatalf("invalid authentication error: %v: %s", err, body)
	}
	if response.Error.Message != "This model requires an Ollama account. Please sign in to Ollama to use this model." || response.Error.Code == nil || *response.Error.Code != "authentication_error" {
		t.Fatalf("unexpected authentication error: %s", body)
	}
	if response.SigninURL != signinURL {
		t.Fatalf("sign-in metadata = %q, want %q", response.SigninURL, signinURL)
	}
}

func TestResponsesMiddlewareNativeAuthenticationErrors(t *testing.T) {
	gin.SetMode(gin.TestMode)
	const signinURL = "https://ollama.com/connect?key=synthetic-secret"
	for _, test := range []struct {
		name                            string
		stream, webSearch, afterContent bool
	}{
		{"JSON", false, false, false},
		{"streaming request", true, false, false},
		{"web search JSON", false, true, false},
		{"web search before stream", true, true, false},
		{"web search during stream", true, true, true},
	} {
		t.Run(test.name, func(t *testing.T) {
			router := gin.New()
			router.POST("/v1/responses", ResponsesMiddleware(), func(c *gin.Context) {
				if test.afterContent {
					data, _ := json.Marshal(api.ChatResponse{Message: api.Message{Role: "assistant", Content: "partial answer"}})
					_, _ = c.Writer.Write(data)
					c.Writer.Flush()
				}
				c.JSON(http.StatusUnauthorized, gin.H{"error": "unauthorized", "signin_url": signinURL})
				if test.afterContent {
					_, _ = c.Writer.Write([]byte(`{"done":true}`))
				}
			})
			tools := "[]"
			if test.webSearch {
				tools = `[{"type":"web_search"}]`
			}
			body := fmt.Sprintf(`{"model":"test","input":"hello","stream":%t,"tools":%s}`, test.stream, tools)
			req := httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(body))
			req.Header.Set("Content-Type", "application/json")
			recorder := httptest.NewRecorder()
			router.ServeHTTP(recorder, req)
			streamError := test.stream && test.webSearch
			wantStatus := http.StatusUnauthorized
			if streamError {
				wantStatus = http.StatusOK
			}
			if recorder.Code != wantStatus {
				t.Fatalf("status = %d, want %d: %s", recorder.Code, wantStatus, recorder.Body.String())
			}
			assertResponsesAuthenticationError(t, recorder.Body.String(), streamError, signinURL)
			if test.afterContent && !strings.Contains(recorder.Body.String(), "partial answer") {
				t.Fatal("previous output was discarded")
			}
		})
	}
}

func TestResponsesMiddlewareDoesNotInferAuthenticationFromMessage(t *testing.T) {
	gin.SetMode(gin.TestMode)
	router := gin.New()
	router.POST("/v1/responses", ResponsesMiddleware(), func(c *gin.Context) {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "unauthorized"})
	})
	req := httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(`{"model":"test","input":"hello"}`))
	req.Header.Set("Content-Type", "application/json")
	recorder := httptest.NewRecorder()
	router.ServeHTTP(recorder, req)

	var response openai.ErrorResponse
	if err := json.Unmarshal(recorder.Body.Bytes(), &response); err != nil {
		t.Fatal(err)
	}
	if recorder.Code != http.StatusInternalServerError || response.Error.Type != "api_error" || response.Error.Message != "unauthorized" {
		t.Fatalf("non-authentication error was reclassified: status=%d body=%s", recorder.Code, recorder.Body.String())
	}
}
