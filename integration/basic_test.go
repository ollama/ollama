//go:build integration

package integration

import (
	"context"
	"net/http"
	"testing"
	"time"

	"github.com/jmorganca/ollama/api"
)

func TestOrcaMiniBlueSky(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 1*time.Minute)
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
	GenerateTestHelper(ctx, t, &http.Client{}, req, []string{"rayleigh"})
}
