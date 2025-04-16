//go:build integration && models

package integration

import (
	"context"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"log/slog"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/format"
)

var (
	started    = time.Now()
	chatModels = []string{
		"granite3-moe:latest",
		"granite-code:latest",
		"nemotron-mini:latest",
		"command-r:latest",
		"gemma2:latest",
		"gemma:latest",
		"internlm2:latest",
		"phi3.5:latest",
		"phi3:latest",
		// "phi:latest", // flaky, sometimes generates no response on first query
		"stablelm2:latest", // Predictions are off, crashes on small VRAM GPUs
		"falcon:latest",
		"falcon2:latest",
		"minicpm-v:latest",
		"mistral:latest",
		"orca-mini:latest",
		"llama2:latest",
		"llama3.1:latest",
		"llama3.2:latest",
		"llama3.2-vision:latest",
		"qwen2.5-coder:latest",
		"qwen:latest",
		"solar-pro:latest",
	}
)

func getTimeouts(t *testing.T) (soft time.Duration, hard time.Duration) {
	deadline, hasDeadline := t.Deadline()
	if !hasDeadline {
		return 8 * time.Minute, 10 * time.Minute
	} else if deadline.Compare(time.Now().Add(2*time.Minute)) <= 0 {
		t.Skip("too little time")
		return time.Duration(0), time.Duration(0)
	}
	return -time.Since(deadline.Add(-2 * time.Minute)), -time.Since(deadline.Add(-20 * time.Second))
}

func TestModelsGenerate(t *testing.T) {
	softTimeout, hardTimeout := getTimeouts(t)
	slog.Info("Setting timeouts", "soft", softTimeout, "hard", hardTimeout)
	ctx, cancel := context.WithTimeout(context.Background(), hardTimeout)
	defer cancel()
	client, _, cleanup := InitServerConnection(ctx, t)
	defer cleanup()

	// TODO use info API eventually
	var maxVram uint64
	var err error
	if s := os.Getenv("OLLAMA_MAX_VRAM"); s != "" {
		maxVram, err = strconv.ParseUint(s, 10, 64)
		if err != nil {
			t.Fatalf("invalid  OLLAMA_MAX_VRAM %v", err)
		}
	} else {
		slog.Warn("No VRAM info available, testing all models, so larger ones might timeout...")
	}

	for _, model := range chatModels {
		t.Run(model, func(t *testing.T) {
			if time.Now().Sub(started) > softTimeout {
				t.Skip("skipping remaining tests to avoid excessive runtime")
			}
			if err := PullIfMissing(ctx, client, model); err != nil {
				t.Fatalf("pull failed %s", err)
			}
			if maxVram > 0 {
				resp, err := client.List(ctx)
				if err != nil {
					t.Fatalf("list models failed %v", err)
				}
				for _, m := range resp.Models {
					if m.Name == model && float32(m.Size)*1.2 > float32(maxVram) {
						t.Skipf("model %s is too large for available VRAM: %s > %s", model, format.HumanBytes(m.Size), format.HumanBytes(int64(maxVram)))
					}
				}
			}
			// TODO - fiddle with context size
			req := api.GenerateRequest{
				Model:  model,
				Prompt: "why is the sky blue?",
				Options: map[string]interface{}{
					"temperature": 0,
					"seed":        123,
				},
			}
			anyResp := []string{"rayleigh", "scattering", "atmosphere", "nitrogen", "oxygen"}
			DoGenerate(ctx, t, client, req, anyResp, 120*time.Second, 30*time.Second)
		})
	}
}

func TestModelsEmbed(t *testing.T) {
	softTimeout, hardTimeout := getTimeouts(t)
	ctx, cancel := context.WithTimeout(context.Background(), hardTimeout)
	defer cancel()
	client, _, cleanup := InitServerConnection(ctx, t)
	defer cleanup()

	// TODO use info API eventually
	var maxVram uint64
	var err error
	if s := os.Getenv("OLLAMA_MAX_VRAM"); s != "" {
		maxVram, err = strconv.ParseUint(s, 10, 64)
		if err != nil {
			t.Fatalf("invalid  OLLAMA_MAX_VRAM %v", err)
		}
	} else {
		slog.Warn("No VRAM info available, testing all models, so larger ones might timeout...")
	}

	data, err := ioutil.ReadFile(filepath.Join("testdata", "embed.json"))
	if err != nil {
		t.Fatalf("failed to open test data file: %s", err)
	}
	testCase := map[string][]float64{}
	err = json.Unmarshal(data, &testCase)
	if err != nil {
		t.Fatalf("failed to load test data: %s", err)
	}
	for model, expected := range testCase {

		t.Run(model, func(t *testing.T) {
			if time.Now().Sub(started) > softTimeout {
				t.Skip("skipping remaining tests to avoid excessive runtime")
			}
			if err := PullIfMissing(ctx, client, model); err != nil {
				t.Fatalf("pull failed %s", err)
			}
			if maxVram > 0 {
				resp, err := client.List(ctx)
				if err != nil {
					t.Fatalf("list models failed %v", err)
				}
				for _, m := range resp.Models {
					if m.Name == model && float32(m.Size)*1.2 > float32(maxVram) {
						t.Skipf("model %s is too large for available VRAM: %s > %s", model, format.HumanBytes(m.Size), format.HumanBytes(int64(maxVram)))
					}
				}
			}
			req := api.EmbeddingRequest{
				Model:  model,
				Prompt: "why is the sky blue?",
				Options: map[string]interface{}{
					"temperature": 0,
					"seed":        123,
				},
			}
			resp, err := client.Embeddings(ctx, &req)
			if err != nil {
				t.Fatalf("embeddings call failed %s", err)
			}
			if len(resp.Embedding) == 0 {
				t.Errorf("zero length embedding response")
			}
			if len(expected) != len(resp.Embedding) {
				expStr := make([]string, len(resp.Embedding))
				for i, v := range resp.Embedding {
					expStr[i] = fmt.Sprintf("%0.6f", v)
				}
				// When adding new models, use this output to populate the testdata/embed.json
				fmt.Printf("expected\n%s\n", strings.Join(expStr, ", "))
				t.Fatalf("expected %d, got %d", len(expected), len(resp.Embedding))
			}
			sim := cosineSimilarity(resp.Embedding, expected)
			if sim < 0.99 {
				t.Fatalf("expected %v, got %v (similarity: %f)", expected[0:5], resp.Embedding[0:5], sim)
			}
		})
	}

}
