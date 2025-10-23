//go:build integration

package integration

import (
	"context"
	"strings"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
)

func TestEmbedTruncationRefactor(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Minute)
	defer cancel()
	client, _, cleanup := InitServerConnection(ctx, t)
	defer cleanup()

	t.Run("single input token count", func(t *testing.T) {
		req := api.EmbedRequest{
			Model: "all-minilm",
			Input: "why is the sky blue?",
		}

		res, err := embedTestHelper(ctx, client, t, req)
		if err != nil {
			t.Fatal(err)
		}

		if res.PromptEvalCount <= 0 {
			t.Fatalf("expected positive token count, got %d", res.PromptEvalCount)
		}
	})

	t.Run("batch parallel token counting", func(t *testing.T) {
		req := api.EmbedRequest{
			Model: "all-minilm",
			Input: []string{"cat", "dog and mouse", "bird"},
		}

		res, err := embedTestHelper(ctx, client, t, req)
		if err != nil {
			t.Fatal(err)
		}

		if len(res.Embeddings) != 3 {
			t.Fatalf("expected 3 embeddings, got %d", len(res.Embeddings))
		}

		if res.PromptEvalCount <= 0 {
			t.Fatalf("expected positive token count, got %d", res.PromptEvalCount)
		}
	})

	t.Run("truncation single input", func(t *testing.T) {
		truncTrue := true
		longInput := strings.Repeat("word ", 100)

		req := api.EmbedRequest{
			Model:    "all-minilm",
			Input:    longInput,
			Truncate: &truncTrue,
			Options:  map[string]any{"num_ctx": 50},
		}

		res, err := embedTestHelper(ctx, client, t, req)
		if err != nil {
			t.Fatal(err)
		}

		if res.PromptEvalCount > 50 {
			t.Fatalf("expected tokens <= 50 after truncation, got %d", res.PromptEvalCount)
		}

		if res.PromptEvalCount == 0 {
			t.Fatal("expected non-zero token count after truncation")
		}
	})

	t.Run("truncation batch", func(t *testing.T) {
		truncTrue := true
		req := api.EmbedRequest{
			Model:    "all-minilm",
			Input:    []string{"short", strings.Repeat("long ", 100), "medium text"},
			Truncate: &truncTrue,
			Options:  map[string]any{"num_ctx": 30},
		}

		res, err := embedTestHelper(ctx, client, t, req)
		if err != nil {
			t.Fatal(err)
		}

		if len(res.Embeddings) != 3 {
			t.Fatalf("expected 3 embeddings, got %d", len(res.Embeddings))
		}

		if res.PromptEvalCount > 90 {
			t.Fatalf("expected tokens <= 90 (3 × 30 max), got %d", res.PromptEvalCount)
		}
	})

	t.Run("truncate false error", func(t *testing.T) {
		truncFalse := false
		req := api.EmbedRequest{
			Model:    "all-minilm",
			Input:    strings.Repeat("word ", 100),
			Truncate: &truncFalse,
			Options:  map[string]any{"num_ctx": 10},
		}

		_, err := embedTestHelper(ctx, client, t, req)
		if err == nil {
			t.Fatal("expected error when truncate=false with long input")
		}

		if !strings.Contains(err.Error(), "exceeds maximum context length") {
			t.Fatalf("expected context length error, got: %v", err)
		}
	})

	t.Run("runner token count accuracy", func(t *testing.T) {
		baseline := api.EmbedRequest{Model: "all-minilm", Input: "test"}
		baseRes, err := embedTestHelper(ctx, client, t, baseline)
		if err != nil {
			t.Fatal(err)
		}

		batch := api.EmbedRequest{
			Model: "all-minilm",
			Input: []string{"test", "test", "test"},
		}
		batchRes, err := embedTestHelper(ctx, client, t, batch)
		if err != nil {
			t.Fatal(err)
		}

		expectedCount := baseRes.PromptEvalCount * 3
		if batchRes.PromptEvalCount < expectedCount-2 || batchRes.PromptEvalCount > expectedCount+2 {
			t.Fatalf("expected ~%d tokens (3 × %d), got %d",
				expectedCount, baseRes.PromptEvalCount, batchRes.PromptEvalCount)
		}
	})
}
