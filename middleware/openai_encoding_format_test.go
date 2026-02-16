package middleware

import (
	"encoding/base64"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/openai"
)

func TestEmbeddingsMiddleware_EncodingFormats(t *testing.T) {
	testCases := []struct {
		name           string
		encodingFormat string
		expectType     string // "array" or "string"
		verifyBase64   bool
	}{
		{"float format", "float", "array", false},
		{"base64 format", "base64", "string", true},
		{"default format", "", "array", false},
	}

	gin.SetMode(gin.TestMode)

	endpoint := func(c *gin.Context) {
		resp := api.EmbedResponse{
			Embeddings:      [][]float32{{0.1, -0.2, 0.3}},
			PromptEvalCount: 5,
		}
		c.JSON(http.StatusOK, resp)
	}

	router := gin.New()
	router.Use(EmbeddingsMiddleware())
	router.Handle(http.MethodPost, "/api/embed", endpoint)

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			body := `{"input": "test", "model": "test-model"`
			if tc.encodingFormat != "" {
				body += `, "encoding_format": "` + tc.encodingFormat + `"`
			}
			body += `}`

			req, _ := http.NewRequest(http.MethodPost, "/api/embed", strings.NewReader(body))
			req.Header.Set("Content-Type", "application/json")

			resp := httptest.NewRecorder()
			router.ServeHTTP(resp, req)

			if resp.Code != http.StatusOK {
				t.Fatalf("expected status 200, got %d", resp.Code)
			}

			var result openai.EmbeddingList
			if err := json.Unmarshal(resp.Body.Bytes(), &result); err != nil {
				t.Fatalf("failed to unmarshal response: %v", err)
			}

			if len(result.Data) != 1 {
				t.Fatalf("expected 1 embedding, got %d", len(result.Data))
			}

			switch tc.expectType {
			case "array":
				if _, ok := result.Data[0].Embedding.([]interface{}); !ok {
					t.Errorf("expected array, got %T", result.Data[0].Embedding)
				}
			case "string":
				embStr, ok := result.Data[0].Embedding.(string)
				if !ok {
					t.Errorf("expected string, got %T", result.Data[0].Embedding)
				} else if tc.verifyBase64 {
					decoded, err := base64.StdEncoding.DecodeString(embStr)
					if err != nil {
						t.Errorf("invalid base64: %v", err)
					} else if len(decoded) != 12 {
						t.Errorf("expected 12 bytes, got %d", len(decoded))
					}
				}
			}
		})
	}
}

func TestEmbeddingsMiddleware_BatchWithBase64(t *testing.T) {
	gin.SetMode(gin.TestMode)

	endpoint := func(c *gin.Context) {
		resp := api.EmbedResponse{
			Embeddings: [][]float32{
				{0.1, 0.2},
				{0.3, 0.4},
				{0.5, 0.6},
			},
			PromptEvalCount: 10,
		}
		c.JSON(http.StatusOK, resp)
	}

	router := gin.New()
	router.Use(EmbeddingsMiddleware())
	router.Handle(http.MethodPost, "/api/embed", endpoint)

	body := `{
		"input": ["hello", "world", "test"],
		"model": "test-model",
		"encoding_format": "base64"
	}`

	req, _ := http.NewRequest(http.MethodPost, "/api/embed", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")

	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	if resp.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d", resp.Code)
	}

	var result openai.EmbeddingList
	if err := json.Unmarshal(resp.Body.Bytes(), &result); err != nil {
		t.Fatalf("failed to unmarshal response: %v", err)
	}

	if len(result.Data) != 3 {
		t.Fatalf("expected 3 embeddings, got %d", len(result.Data))
	}

	// All should be base64 strings
	for i := range 3 {
		embeddingStr, ok := result.Data[i].Embedding.(string)
		if !ok {
			t.Errorf("embedding %d: expected string, got %T", i, result.Data[i].Embedding)
			continue
		}

		// Verify it's valid base64
		if _, err := base64.StdEncoding.DecodeString(embeddingStr); err != nil {
			t.Errorf("embedding %d: invalid base64: %v", i, err)
		}

		// Check index
		if result.Data[i].Index != i {
			t.Errorf("embedding %d: expected index %d, got %d", i, i, result.Data[i].Index)
		}
	}
}

func TestEmbeddingsMiddleware_InvalidEncodingFormat(t *testing.T) {
	gin.SetMode(gin.TestMode)

	endpoint := func(c *gin.Context) {
		c.Status(http.StatusOK)
	}

	router := gin.New()
	router.Use(EmbeddingsMiddleware())
	router.Handle(http.MethodPost, "/api/embed", endpoint)

	testCases := []struct {
		name           string
		encodingFormat string
		shouldFail     bool
	}{
		{"valid: float", "float", false},
		{"valid: base64", "base64", false},
		{"valid: FLOAT (uppercase)", "FLOAT", false},
		{"valid: BASE64 (uppercase)", "BASE64", false},
		{"valid: Float (mixed)", "Float", false},
		{"valid: Base64 (mixed)", "Base64", false},
		{"invalid: json", "json", true},
		{"invalid: hex", "hex", true},
		{"invalid: invalid_format", "invalid_format", true},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			body := `{
				"input": "test",
				"model": "test-model",
				"encoding_format": "` + tc.encodingFormat + `"
			}`

			req, _ := http.NewRequest(http.MethodPost, "/api/embed", strings.NewReader(body))
			req.Header.Set("Content-Type", "application/json")

			resp := httptest.NewRecorder()
			router.ServeHTTP(resp, req)

			if tc.shouldFail {
				if resp.Code != http.StatusBadRequest {
					t.Errorf("expected status 400, got %d", resp.Code)
				}

				var errResp openai.ErrorResponse
				if err := json.Unmarshal(resp.Body.Bytes(), &errResp); err != nil {
					t.Fatalf("failed to unmarshal error response: %v", err)
				}

				if errResp.Error.Type != "invalid_request_error" {
					t.Errorf("expected error type 'invalid_request_error', got %q", errResp.Error.Type)
				}

				if !strings.Contains(errResp.Error.Message, "encoding_format") {
					t.Errorf("expected error message to mention encoding_format, got %q", errResp.Error.Message)
				}
			} else {
				if resp.Code != http.StatusOK {
					t.Errorf("expected status 200, got %d: %s", resp.Code, resp.Body.String())
				}
			}
		})
	}
}
