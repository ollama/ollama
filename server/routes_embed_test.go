package server

import (
	"bytes"
	"context"
	"errors"
	"net/http"
	"strings"
	"sync"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/google/go-cmp/cmp"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/fs/ggml"
)

// TestEmbedBatchLimit verifies that embedding inputs are bounded by the
// physical batch size (num_batch) before they are sent to the runner.
// llama-server cannot split a pooled embedding across ubatches, and inputs
// longer than the batch can crash the runner instead of returning an error.
func TestEmbedBatchLimit(t *testing.T) {
	gin.SetMode(gin.TestMode)

	const numBatch = 8

	var mu sync.Mutex
	var embedInputs []string

	mock := mockRunner{
		EmbeddingFn: func(_ context.Context, input string) ([]float32, int, error) {
			mu.Lock()
			embedInputs = append(embedInputs, input)
			mu.Unlock()

			n := len(strings.Fields(input))
			if n > numBatch {
				// A pooled embedding that exceeds the physical batch size
				// aborts llama-server, so the request fails without an
				// api.StatusError that the retry path could act on.
				return nil, 0, errors.New("do embedding request: EOF")
			}
			return []float32{0.1, 0.2, 0.3, 0.4}, n, nil
		},
		DetokenizeFn: func(_ context.Context, tokens []int) (string, error) {
			// One whitespace-separated field per token, so a truncated token
			// slice round-trips to text with a matching field count.
			return strings.TrimSpace(strings.Repeat("x ", len(tokens))), nil
		},
	}

	resetEmbedInputs := func() {
		mu.Lock()
		embedInputs = nil
		mu.Unlock()
	}

	s := newServerWithMockRunner(t, &mock)

	// add_eos_token is disabled so token limits are not adjusted for an
	// implicit eos, keeping the truncation lengths below exact.
	_, digest := createBinFile(t, ggml.KV{
		"general.architecture":          "llama",
		"llama.block_count":             uint32(1),
		"llama.context_length":          uint32(8192),
		"llama.embedding_length":        uint32(4096),
		"llama.attention.head_count":    uint32(32),
		"llama.attention.head_count_kv": uint32(8),
		"llama.add_eos_token":           false,
		"tokenizer.ggml.tokens":         []string{""},
		"tokenizer.ggml.scores":         []float32{0},
		"tokenizer.ggml.token_type":     []int32{0},
	}, []*ggml.Tensor{
		{Name: "token_embd.weight", Shape: []uint64{1}, WriterTo: bytes.NewReader(make([]byte, 4))},
		{Name: "blk.0.attn_norm.weight", Shape: []uint64{1}, WriterTo: bytes.NewReader(make([]byte, 4))},
		{Name: "blk.0.ffn_down.weight", Shape: []uint64{1}, WriterTo: bytes.NewReader(make([]byte, 4))},
		{Name: "blk.0.ffn_gate.weight", Shape: []uint64{1}, WriterTo: bytes.NewReader(make([]byte, 4))},
		{Name: "blk.0.ffn_up.weight", Shape: []uint64{1}, WriterTo: bytes.NewReader(make([]byte, 4))},
		{Name: "blk.0.ffn_norm.weight", Shape: []uint64{1}, WriterTo: bytes.NewReader(make([]byte, 4))},
		{Name: "blk.0.attn_k.weight", Shape: []uint64{1}, WriterTo: bytes.NewReader(make([]byte, 4))},
		{Name: "blk.0.attn_output.weight", Shape: []uint64{1}, WriterTo: bytes.NewReader(make([]byte, 4))},
		{Name: "blk.0.attn_q.weight", Shape: []uint64{1}, WriterTo: bytes.NewReader(make([]byte, 4))},
		{Name: "blk.0.attn_v.weight", Shape: []uint64{1}, WriterTo: bytes.NewReader(make([]byte, 4))},
		{Name: "output.weight", Shape: []uint64{1}, WriterTo: bytes.NewReader(make([]byte, 4))},
	})

	w := createRequest(t, s.CreateHandler, api.CreateRequest{
		Model:  "test",
		Files:  map[string]string{"file.gguf": digest},
		Stream: &stream,
	})

	if w.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d", w.Code)
	}

	t.Run("long input is truncated to num_batch", func(t *testing.T) {
		resetEmbedInputs()

		w := createRequest(t, s.EmbedHandler, api.EmbedRequest{
			Model:   "test",
			Input:   strings.TrimSpace(strings.Repeat("a ", 12)),
			Options: map[string]any{"num_batch": numBatch},
		})

		if w.Code != http.StatusOK {
			t.Fatalf("expected status 200, got %d: %s", w.Code, w.Body.String())
		}

		mu.Lock()
		defer mu.Unlock()
		if len(embedInputs) != 1 {
			t.Fatalf("expected 1 embedding call, got %d", len(embedInputs))
		}
		if n := len(strings.Fields(embedInputs[0])); n != numBatch {
			t.Errorf("expected input truncated to %d tokens, got %d", numBatch, n)
		}
	})

	t.Run("short input is not truncated", func(t *testing.T) {
		resetEmbedInputs()

		w := createRequest(t, s.EmbedHandler, api.EmbedRequest{
			Model:   "test",
			Input:   "a b c d",
			Options: map[string]any{"num_batch": numBatch},
		})

		if w.Code != http.StatusOK {
			t.Fatalf("expected status 200, got %d: %s", w.Code, w.Body.String())
		}

		mu.Lock()
		defer mu.Unlock()
		if len(embedInputs) != 1 {
			t.Fatalf("expected 1 embedding call, got %d", len(embedInputs))
		}
		if embedInputs[0] != "a b c d" {
			t.Errorf("expected input to be unchanged, got %q", embedInputs[0])
		}
	})

	t.Run("truncate disabled returns 400 when input exceeds num_batch", func(t *testing.T) {
		resetEmbedInputs()

		truncate := false
		w := createRequest(t, s.EmbedHandler, api.EmbedRequest{
			Model:    "test",
			Input:    strings.TrimSpace(strings.Repeat("a ", 12)),
			Truncate: &truncate,
			Options:  map[string]any{"num_batch": numBatch},
		})

		if w.Code != http.StatusBadRequest {
			t.Fatalf("expected status 400, got %d: %s", w.Code, w.Body.String())
		}
		if diff := cmp.Diff(w.Body.String(), `{"error":"the input length exceeds the context length"}`); diff != "" {
			t.Errorf("mismatch (-got +want):\n%s", diff)
		}

		mu.Lock()
		defer mu.Unlock()
		if len(embedInputs) != 0 {
			t.Errorf("expected no embedding calls, got %d", len(embedInputs))
		}
	})

	t.Run("truncate disabled allows input within num_batch", func(t *testing.T) {
		resetEmbedInputs()

		truncate := false
		w := createRequest(t, s.EmbedHandler, api.EmbedRequest{
			Model:    "test",
			Input:    strings.TrimSpace(strings.Repeat("a ", numBatch)),
			Truncate: &truncate,
			Options:  map[string]any{"num_batch": numBatch},
		})

		if w.Code != http.StatusOK {
			t.Fatalf("expected status 200, got %d: %s", w.Code, w.Body.String())
		}
	})
}
