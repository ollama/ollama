//go:build integration

package integration

import (
	"context"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
)

func TestContextExhaustion(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 4*time.Minute) // Longer needed for small footprint GPUs
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
	GenerateTestHelper(ctx, t, req, []string{"once", "upon", "lived"})
}
