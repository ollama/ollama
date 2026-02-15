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

func TestModelsChat(t *testing.T) {
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

	var chatModels []string
	if s := os.Getenv("OLLAMA_NEW_ENGINE"); s != "" {
		chatModels = ollamaEngineChatModels
	} else {
		chatModels = append(ollamaEngineChatModels, llamaRunnerChatModels...)
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
			initialTimeout := 120 * time.Second
			streamTimeout := 30 * time.Second
			slog.Info("loading", "model", model)
			err := client.Generate(ctx,
				&api.GenerateRequest{Model: model, KeepAlive: &api.Duration{Duration: 10 * time.Second}},
				func(response api.GenerateResponse) error { return nil },
			)
			if err != nil {
				t.Fatalf("failed to load model %s: %s", model, err)
			}
			gpuPercent := getGPUPercent(ctx, t, client, model)
			if gpuPercent < 80 {
				slog.Warn("Low GPU percentage - increasing timeouts", "percent", gpuPercent)
				initialTimeout = 240 * time.Second
				streamTimeout = 40 * time.Second
			}

			// TODO - fiddle with context size
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
					"temperature": 0,
					"seed":        123,
				},
			}
			DoChat(ctx, t, client, req, blueSkyExpected, initialTimeout, streamTimeout)
			// best effort unload once we're done with the model
			client.Generate(ctx, &api.GenerateRequest{Model: req.Model, KeepAlive: &api.Duration{Duration: 0}}, func(rsp api.GenerateResponse) error { return nil })
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
				Model:     model,
				Prompt:    "why is the sky blue?",
				KeepAlive: &api.Duration{Duration: 10 * time.Second},
				Options: map[string]interface{}{
					"temperature": 0,
					"seed":        123,
				},
			}
			resp, err := client.Embeddings(ctx, &req)
			if err != nil {
				t.Fatalf("embeddings call failed %s", err)
			}
			defer func() {
				// best effort unload once we're done with the model
				client.Generate(ctx, &api.GenerateRequest{Model: req.Model, KeepAlive: &api.Duration{Duration: 0}}, func(rsp api.GenerateResponse) error { return nil })
			}()
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
