//go:build integration && library

package integration

import (
	"context"
	"fmt"
	"log/slog"
	"os"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
)

// First run of this scenario on a target system will take a long time to download
// ~1.5TB of models.  Set a sufficiently large -timeout for your network speed
func TestLibraryModelsChat(t *testing.T) {
	softTimeout, hardTimeout := getTimeouts(t)
	slog.Info("Setting timeouts", "soft", softTimeout, "hard", hardTimeout)
	ctx, cancel := context.WithTimeout(context.Background(), hardTimeout)
	defer cancel()
	client, _, cleanup := InitServerConnection(ctx, t)
	defer cleanup()
	targetArch := os.Getenv("OLLAMA_TEST_ARCHITECTURE")

	chatModels := libraryChatModels
	for _, model := range chatModels {
		t.Run(model, func(t *testing.T) {
			if time.Now().Sub(started) > softTimeout {
				t.Skip("skipping remaining tests to avoid excessive runtime")
			}
			if err := PullIfMissing(ctx, client, model); err != nil {
				t.Fatalf("pull failed %s", err)
			}
			if targetArch != "" {
				resp, err := client.Show(ctx, &api.ShowRequest{Name: model})
				if err != nil {
					t.Fatalf("unable to show model: %s", err)
				}
				arch := resp.ModelInfo["general.architecture"].(string)
				if arch != targetArch {
					t.Skip(fmt.Sprintf("Skipping %s architecture %s != %s", model, arch, targetArch))
				}
			}
			req := api.ChatRequest{
				Model: model,
				Messages: []api.Message{
					{
						Role:    "user",
						Content: blueSkyPrompt,
					},
				},
				KeepAlive: &api.Duration{Duration: 10 * time.Second},
				Options: map[string]interface{}{
					"temperature": 0.1,
					"seed":        123,
				},
			}
			anyResp := blueSkyExpected
			// Special cases
			if model == "duckdb-nsql" {
				anyResp = []string{"select", "from"}
			} else if model == "granite3-guardian" || model == "shieldgemma" || model == "llama-guard3" || model == "bespoke-minicheck" {
				anyResp = []string{"yes", "no", "safe", "unsafe"}
			} else if model == "openthinker" {
				anyResp = []string{"plugin", "im_sep", "components", "function call"}
			} else if model == "starcoder" || model == "starcoder2" || model == "magicoder" || model == "deepseek-coder" {
				req.Messages[0].Content = "def fibonacci():"
				anyResp = []string{"f(n)", "sequence", "n-1", "main()", "__main__", "while"}
			}
			DoChat(ctx, t, client, req, anyResp, 120*time.Second, 30*time.Second)
		})
	}
}
