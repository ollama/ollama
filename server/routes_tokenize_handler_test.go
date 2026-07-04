package server

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/ollama/ollama/server/tokenizerloader"
)

func TestTokenizeHandler_UsesVocabOnly_NoFallback(t *testing.T) {
	gin.SetMode(gin.TestMode)

	// Arrange: make vocab-only path return our fake
	wantTok := &fakeTokHTTP{tokens: map[string][]int{"hello": {7080, 29477}}}
	tokenizerloader.ResetForTest()
	tokenizerloader.SetOpenVocabOnlyForTest(func(ctx context.Context, model string) (tokenizerloader.Tokenizer, error) {
		return wantTok, nil
	})
	tokenizerloader.RegisterFallbackHooks(
		func(model, text string) ([]int, error) { return nil, fmt.Errorf("should not hit fallback") },
		func(model string, ids []int) (string, error) { return "", fmt.Errorf("should not hit fallback") },
	)

	s := &Server{} // zero value ok; handler calls tokenizerloader.Get
	r := gin.New()
	r.POST("/api/tokenize", s.TokenizeHandler)
	r.POST("/api/detokenize", s.DetokenizeHandler)

	// Act: call /api/tokenize
	body := `{"model":"mistral:latest","content":"hello"}`
	req := httptest.NewRequest(http.MethodPost, "/api/tokenize", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	r.ServeHTTP(w, req)

	// Assert
	if w.Code != 200 {
		t.Fatalf("status = %d, body=%s", w.Code, w.Body.String())
	}
	var resp struct {
		Model  string `json:"model"`
		Tokens []int  `json:"tokens"`
	}
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatalf("json: %v", err)
	}
	if resp.Model != "mistral:latest" {
		t.Fatalf("model mismatch: %s", resp.Model)
	}
	if len(resp.Tokens) != 2 || resp.Tokens[0] != 7080 || resp.Tokens[1] != 29477 {
		t.Fatalf("tokens mismatch: %+v", resp.Tokens)
	}
}

func TestTokenizeHandler_FallbackWhenVocabOnlyUnavailable(t *testing.T) {
	gin.SetMode(gin.TestMode)

	// Arrange: make vocab-only path return error so fallback is used
	tokenizerloader.ResetForTest()
	tokenizerloader.SetOpenVocabOnlyForTest(func(ctx context.Context, model string) (tokenizerloader.Tokenizer, error) {
		return nil, tokenizerloader.ErrVocabOnlyUnavailable
	})
	tokenizerloader.RegisterFallbackHooks(
		func(model, text string) ([]int, error) { return []int{2050}, nil },
		func(model string, ids []int) (string, error) { return " fam", nil },
	)

	s := &Server{} // zero value ok; handler calls tokenizerloader.Get
	r := gin.New()
	r.POST("/api/tokenize", s.TokenizeHandler)
	r.POST("/api/detokenize", s.DetokenizeHandler)

	// Act: call /api/tokenize
	body := `{"model":"mistral:latest","content":"hello"}`
	req := httptest.NewRequest(http.MethodPost, "/api/tokenize", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	r.ServeHTTP(w, req)

	// Assert
	if w.Code != 200 {
		t.Fatalf("status = %d, body=%s", w.Code, w.Body.String())
	}
	var resp struct {
		Model  string `json:"model"`
		Tokens []int  `json:"tokens"`
	}
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatalf("json: %v", err)
	}
	if resp.Model != "mistral:latest" {
		t.Fatalf("model mismatch: %s", resp.Model)
	}
	if len(resp.Tokens) != 1 || resp.Tokens[0] != 2050 {
		t.Fatalf("tokens mismatch: %+v", resp.Tokens)
	}
}

// --- helpers ---

type fakeTokHTTP struct {
	tokens map[string][]int
}

func (f *fakeTokHTTP) Close() error { return nil }

func (f *fakeTokHTTP) Tokenize(text string) ([]int, error) {
	if t, ok := f.tokens[text]; ok {
		return t, nil
	}
	return []int{42}, nil
}

func (f *fakeTokHTTP) Detokenize(ids []int) (string, error) {
	if len(ids) == 1 && ids[0] == 2050 {
		return " fam", nil
	}
	return "hello", nil
}
