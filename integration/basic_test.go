//go:build integration

package integration

import (
	"context"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
)

func TestOrcaMiniBlueSky(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()
	// Set up the test data
	req := api.GenerateRequest{
		Model:  "orca-mini",
		Prompt: "why is the sky blue?",
		Stream: &stream,
		Options: map[string]interface{}{
			"temperature": 0,
			"seed":        123,
		},
	}
	GenerateTestHelper(ctx, t, req, []string{"rayleigh", "scattering"})
}
