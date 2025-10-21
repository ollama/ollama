package middleware

import (
	"encoding/base64"
	"encoding/json"
	"math"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/openai"
)

func TestEmbeddingsMiddleware_EncodingFormat_Float(t *testing.T) {
	gin.SetMode(gin.TestMode)

	// Mock handler that returns embeddings
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

	body := `{
		"input": "test",
		"model": "test-model",
		"encoding_format": "float"
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

	if len(result.Data) != 1 {
		t.Fatalf("expected 1 embedding, got %d", len(result.Data))
	}

	// Check it's a float array
	embeddingSlice, ok := result.Data[0].Embedding.([]interface{})
	if !ok {
		t.Fatalf("expected embedding to be array, got %T", result.Data[0].Embedding)
	}

	if len(embeddingSlice) != 3 {
		t.Errorf("expected 3 floats, got %d", len(embeddingSlice))
	}
}

func TestEmbeddingsMiddleware_EncodingFormat_Base64(t *testing.T) {
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

	body := `{
		"input": "test",
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

	if len(result.Data) != 1 {
		t.Fatalf("expected 1 embedding, got %d", len(result.Data))
	}

	// Check it's a base64 string
	embeddingStr, ok := result.Data[0].Embedding.(string)
	if !ok {
		t.Fatalf("expected embedding to be string, got %T", result.Data[0].Embedding)
	}

	// Verify it's valid base64
	decoded, err := base64.StdEncoding.DecodeString(embeddingStr)
	if err != nil {
		t.Fatalf("failed to decode base64: %v", err)
	}

	// Should be 3 floats * 4 bytes = 12 bytes
	if len(decoded) != 12 {
		t.Errorf("expected 12 bytes, got %d", len(decoded))
	}

	// Verify values
	expected := []float32{0.1, -0.2, 0.3}
	for i := 0; i < 3; i++ {
		offset := i * 4
		bits := uint32(decoded[offset]) |
			uint32(decoded[offset+1])<<8 |
			uint32(decoded[offset+2])<<16 |
			uint32(decoded[offset+3])<<24
		decodedFloat := math.Float32frombits(bits)

		if math.Abs(float64(decodedFloat-expected[i])) > 1e-6 {
			t.Errorf("float[%d]: expected %f, got %f", i, expected[i], decodedFloat)
		}
	}
}

func TestEmbeddingsMiddleware_EncodingFormat_Default(t *testing.T) {
	gin.SetMode(gin.TestMode)

	endpoint := func(c *gin.Context) {
		resp := api.EmbedResponse{
			Embeddings: [][]float32{{0.1, -0.2, 0.3}},
		}
		c.JSON(http.StatusOK, resp)
	}

	router := gin.New()
	router.Use(EmbeddingsMiddleware())
	router.Handle(http.MethodPost, "/api/embed", endpoint)

	// No encoding_format specified
	body := `{
		"input": "test",
		"model": "test-model"
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

	// Should default to float format (array)
	_, ok := result.Data[0].Embedding.([]interface{})
	if !ok {
		t.Errorf("expected default format to be array, got %T", result.Data[0].Embedding)
	}
}

// TestEmbeddingsMiddleware_TokenizedInputRejection removed - OpenAI actually accepts tokenized input

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
	for i := 0; i < 3; i++ {
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
