//go:build integration

package integration

import (
	"context"
	"log/slog"
	"os"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
)

func registerChatCases(models []string) {
	registerModelIntegrationCases("chat", models, runChatModel)
}

func runChatModel(t *testing.T, model string) {
	t.Helper()

	softTimeout, hardTimeout := getTimeouts(t)
	slog.Info("Setting timeouts", "soft", softTimeout, "hard", hardTimeout)
	ctx, cancel := context.WithTimeout(context.Background(), hardTimeout)
	defer cancel()
	client, _, cleanup := InitServerConnection(ctx, t)
	defer cleanup()

	if time.Since(started) > softTimeout {
		t.Skip("skipping remaining tests to avoid excessive runtime")
	}
	skipRegisteredMinVRAM(t, model)
	requireCapability(ctx, t, client, model, "completion")
	skipIfTargetArchitecture(ctx, t, client, model)
	skipIfModelTooLargeForSweepVRAM(ctx, t, client, model)

	initialTimeout := 120 * time.Second
	streamTimeout := 30 * time.Second
	preloadGenerateModel(ctx, t, client, api.GenerateRequest{Model: model, KeepAlive: &api.Duration{Duration: 10 * time.Second}})
	defer func() {
		client.Generate(ctx, &api.GenerateRequest{Model: model, KeepAlive: &api.Duration{Duration: 0}}, func(rsp api.GenerateResponse) error { return nil })
	}()

	gpuPercent := getGPUPercent(ctx, t, client, model)
	if gpuPercent < 80 {
		slog.Warn("Low GPU percentage - increasing timeouts", "percent", gpuPercent)
		initialTimeout = 240 * time.Second
		streamTimeout = 40 * time.Second
	}

	req, anyResp := chatModelRequest(model)
	DoChat(ctx, t, client, req, anyResp, initialTimeout, streamTimeout)
}

func chatModelRequest(model string) (api.ChatRequest, []string) {
	req := api.ChatRequest{
		Model: model,
		Messages: []api.Message{
			{
				Role:    "user",
				Content: blueSkyPrompt,
			},
		},
		KeepAlive: &api.Duration{Duration: 10 * time.Second},
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

	return req, anyResp
}

func skipIfTargetArchitecture(ctx context.Context, t *testing.T, client *api.Client, model string) {
	t.Helper()

	targetArch := os.Getenv("OLLAMA_TEST_ARCHITECTURE")
	if targetArch == "" {
		return
	}

	resp, err := client.Show(ctx, &api.ShowRequest{Name: model})
	if err != nil {
		t.Fatalf("unable to show model: %s", err)
	}
	arch := resp.ModelInfo["general.architecture"].(string)
	if arch != targetArch {
		t.Skipf("Skipping %s architecture %s != %s", model, arch, targetArch)
	}
}
