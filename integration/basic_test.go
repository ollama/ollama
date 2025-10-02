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
	req := api.GenerateRequest{
		Model:  smol,
		Prompt: blueSkyPrompt,
		Stream: &stream,
		Options: map[string]any{
			"temperature": 0,
			"seed":        123,
		},
	}
	GenerateTestHelper(ctx, t, req, blueSkyExpected)
}

func TestUnicode(t *testing.T) {
	skipUnderMinVRAM(t, 6)
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Minute)
	defer cancel()
	// Set up the test data
	req := api.GenerateRequest{
		// DeepSeek has a Unicode tokenizer regex, making it a unicode torture test
		Model:  "deepseek-coder-v2:16b-lite-instruct-q2_K", // TODO is there an ollama-engine model we can switch to and keep the coverage?
		Prompt: "天空为什么是蓝色的?",                               // Why is the sky blue?
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
	if err := PullIfMissing(ctx, client, req.Model); err != nil {
		t.Fatal(err)
	}
	slog.Info("loading", "model", req.Model)
	err := client.Generate(ctx, &api.GenerateRequest{Model: req.Model}, func(response api.GenerateResponse) error { return nil })
	if err != nil {
		t.Fatalf("failed to load model %s: %s", req.Model, err)
	}
	skipIfNotGPULoaded(ctx, t, client, req.Model, 100)

	DoGenerate(ctx, t, client, req, []string{
		"散射", // scattering
		"频率", // frequency
	}, 120*time.Second, 120*time.Second)
}

func TestExtendedUnicodeOutput(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()
	// Set up the test data
	req := api.GenerateRequest{
		Model:  "gemma2:2b",
		Prompt: "Output some smily face emoji",
		Stream: &stream,
		Options: map[string]any{
			"temperature": 0,
			"seed":        123,
		},
	}
	client, _, cleanup := InitServerConnection(ctx, t)
	defer cleanup()
	if err := PullIfMissing(ctx, client, req.Model); err != nil {
		t.Fatal(err)
	}
	DoGenerate(ctx, t, client, req, []string{"😀", "😊", "😁", "😂", "😄", "😃"}, 120*time.Second, 120*time.Second)
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

	req := api.GenerateRequest{
		Model:  smol,
		Prompt: blueSkyPrompt,
		Stream: &stream,
		Options: map[string]any{
			"temperature": 0,
			"seed":        123,
		},
	}
	GenerateTestHelper(ctx, t, req, blueSkyExpected)
}
