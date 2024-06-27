//go:build integration

package integration

import (
	"context"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/stretchr/testify/require"
)

func TestContextExhaustion(t *testing.T) {
	// Longer needed for small footprint GPUs
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()
	// Set up the test data
	req := api.GenerateRequest{
		Model:  "llama2",
		Prompt: "Write me a story with a ton of emojis?",
		Stream: &stream,
		Options: map[string]interface{}{
			"temperature": 0,
			"seed":        123,
			"num_ctx":     128,
		},
	}
	client, _, cleanup := InitServerConnection(ctx, t)
	defer cleanup()
	if err := PullIfMissing(ctx, client, req.Model); err != nil {
		t.Fatalf("PullIfMissing failed: %v", err)
	}
	DoGenerate(ctx, t, client, req, []string{"once", "upon", "lived"}, 120*time.Second, 10*time.Second)
}

func TestContextRoundTrip(t *testing.T) {
	req := api.GenerateRequest{
		Model:     "orca-mini",
		Prompt:    "why is the sky blue?",
		Stream:    &stream,
		KeepAlive: &api.Duration{Duration: 10 * time.Second},
		Options: map[string]interface{}{
			"seed":        42,
			"temperature": 0.0,
		},
	}
	resp := []string{"sunlight"}
	ctx, cancel := context.WithTimeout(context.Background(), 1*time.Minute)
	defer cancel()

	client, _, cleanup := InitServerConnection(ctx, t)
	defer cleanup()

	require.NoError(t, PullIfMissing(ctx, client, req.Model))

	context := DoGenerate(ctx, t, client, req, resp, 60*time.Second, 10*time.Second)
	require.NotEmpty(t, context)

	req.Prompt = "what about the ocean"
	req.Context = context
	resp = []string{"blue"}

	DoGenerate(ctx, t, client, req, resp, 60*time.Second, 10*time.Second)
}
