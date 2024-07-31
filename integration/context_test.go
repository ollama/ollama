<<<<<<< HEAD
=======
<<<<<<< HEAD
>>>>>>> 73e3b128c2a287e5cf76ba2f9fcda3086476a289
//go:build integration

package integration

import (
	"context"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
)

func TestContextExhaustion(t *testing.T) {
<<<<<<< HEAD
	// Longer needed for small footprint GPUs
	ctx, cancel := context.WithTimeout(context.Background(), 6*time.Minute)
=======
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute) // TODO maybe shorter?
>>>>>>> 73e3b128c2a287e5cf76ba2f9fcda3086476a289
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
<<<<<<< HEAD
=======
=======
//go:build integration

package integration

import (
	"context"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
)

func TestContextExhaustion(t *testing.T) {
	// Longer needed for small footprint GPUs
	ctx, cancel := context.WithTimeout(context.Background(), 6*time.Minute)
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
>>>>>>> 0b01490f7a487eae06890c5eabcd2270e32605a5
>>>>>>> 73e3b128c2a287e5cf76ba2f9fcda3086476a289
