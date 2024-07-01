//go:build integration

package integration

import (
	"context"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
)

func TestAllMiniLMEmbed(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()

	req := api.EmbedRequest{
		Model: "all-minilm",
		Input: "why is the sky blue?",
	}

	res := EmbedTestHelper(ctx, t, req)

	if len(res.Embeddings) != 1 {
		t.Fatalf("expected 1 embedding, got %d", len(res.Embeddings))
	}

	if len(res.Embeddings[0]) != 384 {
		t.Fatalf("expected 384 floats, got %d", len(res.Embeddings[0]))
	}
}

func TestAllMiniLMBatchEmbed(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()

	req := api.EmbedRequest{
		Model: "all-minilm",
		Input: []string{"why is the sky blue?", "why is the grass green?"},
	}

	res := EmbedTestHelper(ctx, t, req)

	if len(res.Embeddings) != 2 {
		t.Fatalf("expected 2 embeddings, got %d", len(res.Embeddings))
	}

	if len(res.Embeddings[0]) != 384 {
		t.Fatalf("expected 384 floats, got %d", len(res.Embeddings[0]))
	}
}
