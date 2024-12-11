//go:build integration

package integration

import (
	"context"
	"log/slog"
	"os"
	"strconv"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/format"
)

func TestMultiModelConcurrency(t *testing.T) {
	var (
		req = [2]api.GenerateRequest{
			{
				Model:     "orca-mini",
				Prompt:    "why is the ocean blue?",
				Stream:    &stream,
				KeepAlive: &api.Duration{Duration: 10 * time.Second},
				Options: map[string]interface{}{
					"seed":        42,
					"temperature": 0.0,
				},
			}, {
				Model:     "tinydolphin",
				Prompt:    "what is the origin of the us thanksgiving holiday?",
				Stream:    &stream,
				KeepAlive: &api.Duration{Duration: 10 * time.Second},
				Options: map[string]interface{}{
					"seed":        42,
					"temperature": 0.0,
				},
			},
		}
		resp = [2][]string{
			{"sunlight"},
			{"england", "english", "massachusetts", "pilgrims", "british", "festival"},
		}
	)
	var wg sync.WaitGroup
	wg.Add(len(req))
	ctx, cancel := context.WithTimeout(context.Background(), time.Second*240)
	defer cancel()

	client, _, cleanup := InitServerConnection(ctx, t)
	defer cleanup()

	for i := 0; i < len(req); i++ {
		require.NoError(t, PullIfMissing(ctx, client, req[i].Model))
	}

	for i := 0; i < len(req); i++ {
		go func(i int) {
			defer wg.Done()
			// Note: CPU based inference can crawl so don't give up too quickly
			DoGenerate(ctx, t, client, req[i], resp[i], 90*time.Second, 30*time.Second)
		}(i)
	}
	wg.Wait()
}

func TestIntegrationConcurrentPredictOrcaMini(t *testing.T) {
	req, resp := GenerateRequests()
	reqLimit := len(req)
	iterLimit := 5

	if s := os.Getenv("OLLAMA_MAX_VRAM"); s != "" {
		maxVram, err := strconv.ParseUint(s, 10, 64)
		require.NoError(t, err)
		// Don't hammer on small VRAM cards...
		if maxVram < 4*format.GibiByte {
			reqLimit = min(reqLimit, 2)
			iterLimit = 2
		}
	}

	ctx, cancel := context.WithTimeout(context.Background(), 9*time.Minute)
	defer cancel()
	client, _, cleanup := InitServerConnection(ctx, t)
	defer cleanup()

	// Get the server running (if applicable) warm the model up with a single initial request
	DoGenerate(ctx, t, client, req[0], resp[0], 60*time.Second, 10*time.Second)

	var wg sync.WaitGroup
	wg.Add(reqLimit)
	for i := 0; i < reqLimit; i++ {
		go func(i int) {
			defer wg.Done()
			for j := 0; j < iterLimit; j++ {
				slog.Info("Starting", "req", i, "iter", j)
				// On slower GPUs it can take a while to process the concurrent requests
				// so we allow a much longer initial timeout
				DoGenerate(ctx, t, client, req[i], resp[i], 120*time.Second, 20*time.Second)
			}
		}(i)
	}
	wg.Wait()
}

// Stress the system if we know how much VRAM it has, and attempt to load more models than will fit
func TestMultiModelStress(t *testing.T) {
	s := os.Getenv("OLLAMA_MAX_VRAM") // TODO - discover actual VRAM
	if s == "" {
		t.Skip("OLLAMA_MAX_VRAM not specified, can't pick the right models for the stress test")
	}

	maxVram, err := strconv.ParseUint(s, 10, 64)
	if err != nil {
		t.Fatal(err)
	}
	if maxVram < 2*format.GibiByte {
		t.Skip("VRAM less than 2G, skipping model stress tests")
	}

	type model struct {
		name string
		size uint64 // Approximate amount of VRAM they typically use when fully loaded in VRAM
	}

	smallModels := []model{
		{
			name: "orca-mini",
			size: 2992 * format.MebiByte,
		},
		{
			name: "phi",
			size: 2616 * format.MebiByte,
		},
		{
			name: "gemma:2b",
			size: 2364 * format.MebiByte,
		},
		{
			name: "stable-code:3b",
			size: 2608 * format.MebiByte,
		},
		{
			name: "starcoder2:3b",
			size: 2166 * format.MebiByte,
		},
	}
	mediumModels := []model{
		{
			name: "llama2",
			size: 5118 * format.MebiByte,
		},
		{
			name: "mistral",
			size: 4620 * format.MebiByte,
		},
		{
			name: "orca-mini:7b",
			size: 5118 * format.MebiByte,
		},
		{
			name: "dolphin-mistral",
			size: 4620 * format.MebiByte,
		},
		{
			name: "gemma:7b",
			size: 5000 * format.MebiByte,
		},
		{
			name: "codellama:7b",
			size: 5118 * format.MebiByte,
		},
	}

	// These seem to be too slow to be useful...
	// largeModels := []model{
	// 	{
	// 		name: "llama2:13b",
	// 		size: 7400 * format.MebiByte,
	// 	},
	// 	{
	// 		name: "codellama:13b",
	// 		size: 7400 * format.MebiByte,
	// 	},
	// 	{
	// 		name: "orca-mini:13b",
	// 		size: 7400 * format.MebiByte,
	// 	},
	// 	{
	// 		name: "gemma:7b",
	// 		size: 5000 * format.MebiByte,
	// 	},
	// 	{
	// 		name: "starcoder2:15b",
	// 		size: 9100 * format.MebiByte,
	// 	},
	// }

	var chosenModels []model
	switch {
	case maxVram < 10000*format.MebiByte:
		slog.Info("selecting small models")
		chosenModels = smallModels
	// case maxVram < 30000*format.MebiByte:
	default:
		slog.Info("selecting medium models")
		chosenModels = mediumModels
		// default:
		// 	slog.Info("selecting large models")
		// 	chosenModels = largeModels
	}

	req, resp := GenerateRequests()

	for i := range req {
		if i > len(chosenModels) {
			break
		}
		req[i].Model = chosenModels[i].name
	}

	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Minute) // TODO baseline -- 10m too short
	defer cancel()
	client, _, cleanup := InitServerConnection(ctx, t)
	defer cleanup()

	// Make sure all the models are pulled before we get started
	for _, r := range req {
		require.NoError(t, PullIfMissing(ctx, client, r.Model))
	}

	var wg sync.WaitGroup
	consumed := uint64(256 * format.MebiByte) // Assume some baseline usage
	for i := 0; i < len(req); i++ {
		// Always get at least 2 models, but don't overshoot VRAM too much or we'll take too long
		if i > 1 && consumed > maxVram {
			slog.Info("achieved target vram exhaustion", "count", i, "vram", format.HumanBytes2(maxVram), "models", format.HumanBytes2(consumed))
			break
		}
		consumed += chosenModels[i].size
		slog.Info("target vram", "count", i, "vram", format.HumanBytes2(maxVram), "models", format.HumanBytes2(consumed))

		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			for j := 0; j < 3; j++ {
				slog.Info("Starting", "req", i, "iter", j, "model", req[i].Model)
				DoGenerate(ctx, t, client, req[i], resp[i], 120*time.Second, 5*time.Second)
			}
		}(i)
	}
	go func() {
		for {
			time.Sleep(2 * time.Second)
			select {
			case <-ctx.Done():
				return
			default:
				models, err := client.ListRunning(ctx)
				if err != nil {
					slog.Warn("failed to list running models", "error", err)
					continue
				}
				for _, m := range models.Models {
					slog.Info("loaded model snapshot", "model", m)
				}
			}
		}
	}()
	wg.Wait()
}
