package server

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"sync/atomic"
	"testing"

	"github.com/ollama/ollama/server/tokenizerloader"
)

func TestTokenizeHandler_UsesVocabOnly_NoFallback(t *testing.T) {
	_ = os.Setenv("OLLAMA_TOKENIZER_DEBUG", "1")
	t.Cleanup(func() { _ = os.Unsetenv("OLLAMA_TOKENIZER_DEBUG") })

	// Reset tokenizer state
	tokenizerloader.ResetForTest()

	// Inject vocab-only success
	tokenizerloader.SetOpenVocabOnlyForTest(func(ctx context.Context, model string) (tokenizerloader.Tokenizer, error) {
		return &fakeTokHTTP{tokens: []int{1234}}, nil
	})

	// Fallback hooks that would bump a counter if triggered
	var fallbackHit atomic.Int32
	tokenizerloader.RegisterFallbackHooks(
		func(ctx context.Context, model, text string) ([]int, error) {
			fallbackHit.Add(1)
			return []int{9}, nil
		},
		func(ctx context.Context, model string, ids []int) (string, error) {
			fallbackHit.Add(1)
			return "fallback", nil
		},
	)

	// Build a lightweight server with just the routes
	s := &Server{}
	rc := newTestRegistry(t) // provide a minimal registry if your GenerateRoutes needs one
	h, err := s.GenerateRoutes(rc)
	if err != nil {
		t.Fatalf("GenerateRoutes: %v", err)
	}

	body := map[string]any{
		"model":   "mistral:latest",
		"content": "hello",
	}
	b, _ := json.Marshal(body)
	req := httptest.NewRequest(http.MethodPost, "/api/tokenize", bytes.NewReader(b))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("unexpected status: %d, body=%s", w.Code, w.Body.String())
	}

	// Ensure fallback was NOT used
	if fallbackHit.Load() != 0 {
		t.Fatalf("fallback should not be used when vocab-only is available")
	}

	// Basic shape check on response
	var resp struct {
		Model  string `json:"model"`
		Tokens []int  `json:"tokens"`
	}
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatalf("bad json: %v", err)
	}
	if resp.Model != "mistral:latest" {
		t.Fatalf("wrong model in response: %q", resp.Model)
	}
	if len(resp.Tokens) != 1 || resp.Tokens[0] != 1234 {
		t.Fatalf("unexpected tokens in response: %v", resp.Tokens)
	}
}

// --- helpers ---

type fakeTokHTTP struct{ tokens []int }

func (f *fakeTokHTTP) Tokenize(ctx context.Context, s string) ([]int, error) { return f.tokens, nil }
func (f *fakeTokHTTP) Detokenize(ctx context.Context, ids []int) (string, error) {
	return "unused", nil
}

// newTestRegistry returns a minimal registry acceptable to GenerateRoutes.
// If your code needs a real registry, adapt accordingly (or stub its methods).
func newTestRegistry(t *testing.T) *Registry {
	t.Helper()
	return &Registry{} // adjust if your Server.GenerateRoutes requires fields
}
