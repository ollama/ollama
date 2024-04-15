//go:build integration

package integration

import (
	"context"
	"net/http"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
)

func TestAllMiniLMEmbedding(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()

	req := api.EmbeddingRequest{
		Model:  "all-minilm",
		Prompt: "why is the sky blue?",
		Options: map[string]interface{}{
			"temperature": 0,
			"seed":        123,
		},
	}

	res := EmbeddingTestHelper(ctx, t, &http.Client{}, req)

	if len(res.Embedding) != 384 {
		t.Fatalf("Expected 384 floats to be returned, got %v", len(res.Embedding))
	}

	if res.Embedding[0] != 0.146763876080513 {
		t.Fatalf("Expected first embedding float to be 0.146763876080513, got %v", res.Embedding[0])
	}
}

func TestAllMiniLMEmbeddings(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()

	req := api.EmbeddingRequest{
		Model:   "all-minilm",
		Prompts: []string{"why is the sky blue?", "why is the sky blue?"},
		Options: map[string]interface{}{
			"temperature": 0,
			"seed":        123,
		},
	}

	res := EmbeddingTestHelper(ctx, t, &http.Client{}, req)

	if len(res.Embeddings) != 2 {
		t.Fatal("Expected 2 embeddings to be returned")
	}

	if len(res.Embeddings[0]) != 384 {
		t.Fatalf("Expected first embedding to have 384 floats, got %v", len(res.Embeddings[0]))
	}

	if res.Embeddings[0][0] != 0.146763876080513 && res.Embeddings[1][0] != 0.146763876080513 {
		t.Fatalf("Expected first embedding floats to be 0.146763876080513, got %v, %v", res.Embeddings[0][0], res.Embeddings[1][0])
	}
}
