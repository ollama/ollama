//go:build integration

package integration

import (
	"context"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
)

// SmolLM2-135M-Instruct-GGUF-Split is a small public repository whose only
// quant is sharded, so pulling it exercises the sharded GGUF fallback against
// the real registry for about 92 MB. The registry rejects the tag at manifest
// resolution, the shards are fetched from Hugging Face directly, and the create
// path merges them into a single model layer.
const shardedModel = "hf.co/owalsh/SmolLM2-135M-Instruct-GGUF-Split:Q4_0"

func TestPullShardedGGUF(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
	defer cancel()

	client, _, cleanup := InitServerConnection(ctx, t)
	defer cleanup()

	if err := PullIfMissing(ctx, client, shardedModel); err != nil {
		t.Fatalf("pulling sharded model: %v", err)
	}

	// The merged model must be usable, and must be a single model layer rather
	// than one layer per shard.
	show, err := client.Show(ctx, &api.ShowRequest{Model: shardedModel})
	if err != nil {
		t.Fatalf("show: %v", err)
	}
	if show.Details.Family == "" {
		t.Error("merged model reports no family, suggesting a bad merge")
	}

	req := api.GenerateRequest{
		Model:   shardedModel,
		Prompt:  "why is the sky blue?",
		Stream:  &stream,
		Options: map[string]any{"temperature": 0, "seed": 123},
	}
	DoGenerate(ctx, t, client, req, []string{"rayleigh", "scatter", "blue", "light"}, 120*time.Second, 30*time.Second)
}
