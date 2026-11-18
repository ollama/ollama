package server

import (
	"bytes"
	"context"
	"encoding/json"
	"math"
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
	"testing"

	"github.com/ollama/ollama/llm"
)

type embeddingTestRunner struct {
	llm.LlamaServer
	embeddings map[string][]float32
}

func (r *embeddingTestRunner) Embedding(_ context.Context, input string) ([]float32, int, error) {
	return slices.Clone(r.embeddings[input]), 1, nil
}

func TestEmbeddingHandlersRejectInvalidVectors(t *testing.T) {
	for _, endpoint := range []string{"/api/embed", "/v1/embeddings", "/api/embeddings"} {
		for _, tc := range []struct {
			name   string
			vector []float32
			want   string
		}{
			{"zero", []float32{0, 0, 0}, "zero norm"},
			{"empty", nil, "empty embedding"},
			{"nan", []float32{1, float32(math.NaN())}, "NaN or Inf"},
			{"infinity", []float32{1, float32(math.Inf(1))}, "NaN or Inf"},
		} {
			t.Run(endpoint+"/"+tc.name, func(t *testing.T) {
				h := embeddingTestHandler(t, map[string][]float32{"bad": tc.vector, "good": {3, 4, 0}})
				body := map[string]any{"model": "embed-test", "input": []string{"good", "bad"}}
				if endpoint == "/api/embeddings" {
					body = map[string]any{"model": "embed-test", "prompt": "bad"}
				}
				w := embeddingTestRequest(t, h, endpoint, body)
				if w.Code != http.StatusInternalServerError {
					t.Fatalf("status = %d, want 500; body = %s", w.Code, w.Body.String())
				}
				if !strings.Contains(w.Body.String(), tc.want) {
					t.Fatalf("body = %s, want error containing %q", w.Body.String(), tc.want)
				}
			})
		}
	}
}

func TestEmbedValidVectorsAndPreload(t *testing.T) {
	h := embeddingTestHandler(t, map[string][]float32{"good": {3, 4, 0}})
	for _, tc := range []struct {
		name       string
		input      any
		dimensions int
		want       [][]float32
	}{
		{"single", "good", 0, [][]float32{{0.6, 0.8, 0}}},
		{"batch", []string{"good", "good"}, 0, [][]float32{{0.6, 0.8, 0}, {0.6, 0.8, 0}}},
		{"dimensions", "good", 1, [][]float32{{1}}},
		{"preload", "", 0, [][]float32{}},
		{"empty_batch", []string{}, 0, [][]float32{}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			w := embeddingTestRequest(t, h, "/api/embed", map[string]any{
				"model": "embed-test", "input": tc.input, "dimensions": tc.dimensions,
			})
			if w.Code != http.StatusOK {
				t.Fatalf("status = %d, want 200; body = %s", w.Code, w.Body.String())
			}
			var response struct {
				Embeddings [][]float32 `json:"embeddings"`
			}
			if err := json.Unmarshal(w.Body.Bytes(), &response); err != nil {
				t.Fatal(err)
			}
			if !slices.EqualFunc(response.Embeddings, tc.want, slices.Equal[[]float32]) {
				t.Fatalf("embeddings = %v, want %v", response.Embeddings, tc.want)
			}
		})
	}
	for _, prompt := range []string{"", "good"} {
		w := embeddingTestRequest(t, h, "/api/embeddings", map[string]any{"model": "embed-test", "prompt": prompt})
		if w.Code != http.StatusOK {
			t.Fatalf("legacy embedding status = %d; body = %s", w.Code, w.Body.String())
		}
		var response struct {
			Embedding []float64 `json:"embedding"`
		}
		if err := json.Unmarshal(w.Body.Bytes(), &response); err != nil {
			t.Fatal(err)
		}
		want := []float64{}
		if prompt != "" {
			want = []float64{3, 4, 0}
		}
		if !slices.Equal(response.Embedding, want) {
			t.Fatalf("legacy embedding = %v, want %v", response.Embedding, want)
		}
	}
}

func TestEmbedRejectsZeroNormAfterTruncation(t *testing.T) {
	h := embeddingTestHandler(t, map[string][]float32{"good": {0, 0, 1}})
	for _, endpoint := range []string{"/api/embed", "/v1/embeddings"} {
		w := embeddingTestRequest(t, h, endpoint, map[string]any{
			"model": "embed-test", "input": "good", "dimensions": 2,
		})
		if w.Code != http.StatusBadRequest {
			t.Fatalf("%s: status = %d, want 400; body = %s", endpoint, w.Code, w.Body.String())
		}
		if !strings.Contains(w.Body.String(), "cannot normalize embedding with dimensions 2") {
			t.Fatalf("%s: unexpected error: %s", endpoint, w.Body.String())
		}
	}
}

func embeddingTestHandler(t *testing.T, embeddings map[string][]float32) http.Handler {
	t.Helper()
	t.Setenv("OLLAMA_MODELS", t.TempDir())
	mock := &mockRunner{LlamaServer: &embeddingTestRunner{embeddings: embeddings}}
	s := newServerWithMockRunner(t, mock)
	createMinimalGGUFModel(t, s, "embed-test", nil, "", nil)
	h, err := s.GenerateRoutes()
	if err != nil {
		t.Fatal(err)
	}
	return h
}

func embeddingTestRequest(t *testing.T, h http.Handler, endpoint string, body any) *httptest.ResponseRecorder {
	t.Helper()
	data, err := json.Marshal(body)
	if err != nil {
		t.Fatal(err)
	}
	req := httptest.NewRequest(http.MethodPost, endpoint, bytes.NewReader(data))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)
	return w
}
