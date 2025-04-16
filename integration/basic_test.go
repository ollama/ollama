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
	"github.com/stretchr/testify/require"
)

func TestBlueSky(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()
	// Set up the test data
	req := api.GenerateRequest{
		Model:  smol,
		Prompt: "why is the sky blue?",
		Stream: &stream,
		Options: map[string]any{
			"temperature": 0,
			"seed":        123,
		},
	}
	GenerateTestHelper(ctx, t, req, []string{"rayleigh", "scattering"})
}

func TestUnicode(t *testing.T) {
	skipUnderMinVRAM(t, 6)
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Minute)
	defer cancel()
	// Set up the test data
	req := api.GenerateRequest{
		// DeepSeek has a Unicode tokenizer regex, making it a unicode torture test
		Model:  "deepseek-coder-v2:16b-lite-instruct-q2_K",
		Prompt: "天空为什么是蓝色的?",
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
	require.NoError(t, PullIfMissing(ctx, client, req.Model))
	DoGenerate(ctx, t, client, req, []string{"散射", "频率"}, 120*time.Second, 120*time.Second)
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
	require.NoError(t, PullIfMissing(ctx, client, req.Model))
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
	require.NoError(t, err)
	defer os.RemoveAll(modelDir)
	slog.Info("unicode", "OLLAMA_MODELS", modelDir)

	t.Setenv("OLLAMA_MODELS", modelDir)

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()

	req := api.GenerateRequest{
		Model:  smol,
		Prompt: "why is the sky blue?",
		Stream: &stream,
		Options: map[string]any{
			"temperature": 0,
			"seed":        123,
		},
	}
	GenerateTestHelper(ctx, t, req, []string{"rayleigh", "scattering"})
}
