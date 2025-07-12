//go:build integration && library

package integration

import (
	"context"
	"log/slog"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
)

// First run of this scenario on a target system will take a long time to download
// ~1.5TB of models.  Set a sufficiently large -timeout for your network speed
func TestLibraryModelsGenerate(t *testing.T) {
	softTimeout, hardTimeout := getTimeouts(t)
	slog.Info("Setting timeouts", "soft", softTimeout, "hard", hardTimeout)
	ctx, cancel := context.WithTimeout(context.Background(), hardTimeout)
	defer cancel()
	client, _, cleanup := InitServerConnection(ctx, t)
	defer cleanup()

	chatModels := libraryChatModels
	for _, model := range chatModels {
		t.Run(model, func(t *testing.T) {
			if time.Now().Sub(started) > softTimeout {
				t.Skip("skipping remaining tests to avoid excessive runtime")
			}
			if err := PullIfMissing(ctx, client, model); err != nil {
				t.Fatalf("pull failed %s", err)
			}
			req := api.GenerateRequest{
				Model:     model,
				Prompt:    "why is the sky blue?",
				KeepAlive: &api.Duration{Duration: 10 * time.Second},
				Options: map[string]interface{}{
					"temperature": 0.1,
					"seed":        123,
				},
			}
			anyResp := []string{"rayleigh", "scatter", "atmosphere", "nitrogen", "oxygen", "wavelength"}
			// Special cases
			if model == "duckdb-nsql" {
				anyResp = []string{"select", "from"}
			} else if model == "granite3-guardian" || model == "shieldgemma" || model == "llama-guard3" || model == "bespoke-minicheck" {
				anyResp = []string{"yes", "no", "safe", "unsafe"}
			} else if model == "openthinker" || model == "nexusraven" {
				anyResp = []string{"plugin", "im_sep", "components", "function call"}
			} else if model == "starcoder" || model == "starcoder2" || model == "magicoder" || model == "deepseek-coder" {
				req.Prompt = "def fibonacci():"
				anyResp = []string{"f(n)", "sequence", "n-1", "main()", "__main__", "while"}
			}
			DoGenerate(ctx, t, client, req, anyResp, 120*time.Second, 30*time.Second)
		})
	}
}
