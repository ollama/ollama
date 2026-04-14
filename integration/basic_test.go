//go:build integration

package integration

import (
	"context"
	"log/slog"
	"os"
	"runtime"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
)

func TestBlueSky(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()
	// Set up the test data
	req := api.ChatRequest{
		Model: smol,
		Messages: []api.Message{
			{
				Role:    "user",
				Content: blueSkyPrompt,
			},
		},
		Stream: &stream,
		Options: map[string]any{
			"temperature": 0,
			"seed":        123,
		},
	}
	ChatTestHelper(ctx, t, req, blueSkyExpected)
}

func TestUnicode(t *testing.T) {
	if testModel != "" {
		t.Skip("uses hardcoded model, not applicable with model override")
	}
	skipUnderMinVRAM(t, 6)
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Minute)
	defer cancel()
	// Set up the test data
	req := api.ChatRequest{
		// DeepSeek has a Unicode tokenizer regex, making it a unicode torture test
		Model: "deepseek-coder-v2:16b-lite-instruct-q2_K", // TODO is there an ollama-engine model we can switch to and keep the coverage?
		Messages: []api.Message{
			{
				Role:    "user",
				Content: "天空为什么是蓝色的?", // Why is the sky blue?
			},
		},
		Stream: &stream,
		Options: map[string]any{
			"temperature": 0,
			"seed":        123,
			// Workaround deepseek context shifting bug
			"num_ctx":     8192,
			"num_predict": 2048,
		},
	}
	client, _, cleanup := InitServerConnection(ctx, t)
	defer cleanup()
	pullOrSkip(ctx, t, client, req.Model)
	slog.Info("loading", "model", req.Model)
	err := client.Generate(ctx, &api.GenerateRequest{Model: req.Model}, func(response api.GenerateResponse) error { return nil })
	if err != nil {
		t.Fatalf("failed to load model %s: %s", req.Model, err)
	}
	defer func() {
		// best effort unload once we're done with the model
		client.Generate(ctx, &api.GenerateRequest{Model: req.Model, KeepAlive: &api.Duration{Duration: 0}}, func(rsp api.GenerateResponse) error { return nil })
	}()

	skipIfNotGPULoaded(ctx, t, client, req.Model, 100)

	DoChat(ctx, t, client, req, []string{
		"散射", // scattering
		"频率", // frequency
	}, 120*time.Second, 120*time.Second)
}

func TestExtendedUnicodeOutput(t *testing.T) {
	if testModel != "" {
		t.Skip("uses hardcoded model, not applicable with model override")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()
	// Set up the test data
	req := api.ChatRequest{
		Model: "gemma2:2b",
		Messages: []api.Message{
			{
				Role:    "user",
				Content: "Output some smily face emoji",
			},
		},
		Stream: &stream,
		Options: map[string]any{
			"temperature": 0,
			"seed":        123,
		},
	}
	client, _, cleanup := InitServerConnection(ctx, t)
	defer cleanup()
	pullOrSkip(ctx, t, client, req.Model)
	DoChat(ctx, t, client, req, []string{"😀", "😊", "😁", "😂", "😄", "😃"}, 120*time.Second, 120*time.Second)
}

func TestUnicodeModelDir(t *testing.T) {
	// This is only useful for Windows with utf-16 characters, so skip this test for other platforms
	if runtime.GOOS != "windows" {
		t.Skip("Unicode test only applicable to windows")
	}
	// Only works for local testing
	if os.Getenv("OLLAMA_TEST_EXISTING") != "" {
		t.Skip("TestUnicodeModelDir only works for local testing, skipping")
	}

	modelDir, err := os.MkdirTemp("", "ollama_埃")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(modelDir)
	slog.Info("unicode", "OLLAMA_MODELS", modelDir)

	t.Setenv("OLLAMA_MODELS", modelDir)

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()

	req := api.ChatRequest{
		Model: smol,
		Messages: []api.Message{
			{
				Role:    "user",
				Content: blueSkyPrompt,
			},
		},
		Stream: &stream,
		Options: map[string]any{
			"temperature": 0,
			"seed":        123,
		},
	}
	ChatTestHelper(ctx, t, req, blueSkyExpected)
}

// TestNumPredict verifies that when num_predict is set, the model generates
// exactly that many tokens. It uses logprobs to count the actual tokens output.
func TestNumPredict(t *testing.T) {
	if testModel != "" {
		t.Skip("uses hardcoded model, not applicable with model override")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()

	client, _, cleanup := InitServerConnection(ctx, t)
	defer cleanup()

	pullOrSkip(ctx, t, client, "qwen3:0.6b")

	req := api.GenerateRequest{
		Model:    "qwen3:0.6b",
		Prompt:   "Write a long story.",
		Stream:   &stream,
		Logprobs: true,
		Options: map[string]any{
			"num_predict": 10,
			"temperature": 0,
			"seed":        123,
		},
	}

	logprobCount := 0
	var finalResponse api.GenerateResponse
	err := client.Generate(ctx, &req, func(resp api.GenerateResponse) error {
		logprobCount += len(resp.Logprobs)
		if resp.Done {
			finalResponse = resp
		}
		return nil
	})
	if err != nil {
		t.Fatalf("generate failed: %v", err)
	}

	if logprobCount != 10 {
		t.Errorf("expected 10 tokens (logprobs), got %d (EvalCount=%d, DoneReason=%s)",
			logprobCount, finalResponse.EvalCount, finalResponse.DoneReason)
	}
}
