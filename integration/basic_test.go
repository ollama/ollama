//go:build integration

package integration

import (
	"context"
	"encoding/base64"
	"log/slog"
	"os"
	"runtime"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
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
		Options: map[string]interface{}{
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
		Options: map[string]interface{}{
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
		Options: map[string]interface{}{
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
		Options: map[string]interface{}{
			"temperature": 0,
			"seed":        123,
		},
	}
	GenerateTestHelper(ctx, t, req, []string{"rayleigh", "scattering"})
}

// Run through the models supported by the new engine
func TestNewEngineModels(t *testing.T) {
	softTimeout, hardTimeout := getTimeouts(t)
	if !envconfig.NewEngine() {
		t.Skip("New engine not enabled via OLLAMA_NEW_ENGINE=1, skipping new-engine tests")
	}
	ctx, cancel := context.WithTimeout(context.Background(), hardTimeout)
	defer cancel()
	client, _, cleanup := InitServerConnection(ctx, t)
	defer cleanup()

	newEngineChatModels := []string{
		"llama3.1:latest",
		"llama3.2:3b-instruct-q4_K_M", // aka latest
		"llama3.2:1b-instruct-fp16",
		"llama3.2:1b-instruct-q2_K",
		"llama3.2:1b-instruct-q3_K_L",
		"llama3.2:1b-instruct-q3_K_M",
		"llama3.2:1b-instruct-q3_K_S",
		"llama3.2:1b-instruct-q4_0",
		"llama3.2:1b-instruct-q4_1",
		"llama3.2:1b-instruct-q4_K_S",
		"llama3.2:1b-instruct-q5_0",
		"llama3.2:1b-instruct-q5_1",
		"llama3.2:1b-instruct-q5_K_M",
		"llama3.2:1b-instruct-q5_K_S",
		"llama3.2:1b-instruct-q6_K",
		"llama3.2:1b-instruct-q8_0",
		"gemma2:latest",
	}
	newEngineVisionModels := []string{
		"llama3.2-vision:latest",
		"gemma3:latest",
	}

	for _, model := range newEngineChatModels {
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
				KeepAlive: &api.Duration{Duration: 0},
				Options: map[string]interface{}{
					"temperature": 0,
					"seed":        123,
				},
			}
			anyResp := []string{"rayleigh", "scattering", "atmosphere", "nitrogen", "oxygen"}
			DoGenerate(ctx, t, client, req, anyResp, 120*time.Second, 30*time.Second)
		})
	}
	for _, model := range newEngineVisionModels {
		t.Run(model, func(t *testing.T) {
			image, err := base64.StdEncoding.DecodeString(imageEncoding)
			require.NoError(t, err)
			req := api.GenerateRequest{
				Model:     model,
				Prompt:    "what does the text in this image say?",
				KeepAlive: &api.Duration{Duration: 0},
				Options: map[string]interface{}{
					"temperature": 0,
					"seed":        123,
				},
				Images: []api.ImageData{
					image,
				},
			}
			resp := "the ollamas"
			require.NoError(t, PullIfMissing(ctx, client, req.Model))
			// llava models on CPU can be quite slow to start,
			DoGenerate(ctx, t, client, req, []string{resp}, 240*time.Second, 30*time.Second)
		})
	}
}
