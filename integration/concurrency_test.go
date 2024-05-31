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

	"github.com/ollama/ollama/api"
	"github.com/stretchr/testify/require"
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
			[]string{"sunlight"},
			[]string{"england", "english", "massachusetts", "pilgrims", "british"},
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
			DoGenerate(ctx, t, client, req[i], resp[i], 60*time.Second, 10*time.Second)
		}(i)
	}
	wg.Wait()
}

func TestIntegrationConcurrentPredictOrcaMini(t *testing.T) {
	req, resp := GenerateRequests()
	reqLimit := len(req)
	iterLimit := 5

	vram := os.Getenv("OLLAMA_MAX_VRAM")
	if vram != "" {
		max, err := strconv.ParseUint(vram, 10, 64)
		require.NoError(t, err)
		// Don't hammer on small VRAM cards...
		if max < 4*1024*1024*1024 {
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
	vram := os.Getenv("OLLAMA_MAX_VRAM")
	if vram == "" {
		t.Skip("OLLAMA_MAX_VRAM not specified, can't pick the right models for the stress test")
	}
	max, err := strconv.ParseUint(vram, 10, 64)
	require.NoError(t, err)
	const MB = uint64(1024 * 1024)
	type model struct {
		name string
		size uint64 // Approximate amount of VRAM they typically use when fully loaded in VRAM
	}

	smallModels := []model{
		{
			name: "orca-mini",
			size: 2992 * MB,
		},
		{
			name: "phi",
			size: 2616 * MB,
		},
		{
			name: "gemma:2b",
			size: 2364 * MB,
		},
		{
			name: "stable-code:3b",
			size: 2608 * MB,
		},
		{
			name: "starcoder2:3b",
			size: 2166 * MB,
		},
	}
	mediumModels := []model{
		{
			name: "llama2",
			size: 5118 * MB,
		},
		{
			name: "mistral",
			size: 4620 * MB,
		},
		{
			name: "orca-mini:7b",
			size: 5118 * MB,
		},
		{
			name: "dolphin-mistral",
			size: 4620 * MB,
		},
		{
			name: "gemma:7b",
			size: 5000 * MB,
		},
		// TODO - uncomment this once #3565 is merged and this is rebased on it
		// {
		// 	name: "codellama:7b",
		// 	size: 5118 * MB,
		// },
	}

	// These seem to be too slow to be useful...
	// largeModels := []model{
	// 	{
	// 		name: "llama2:13b",
	// 		size: 7400 * MB,
	// 	},
	// 	{
	// 		name: "codellama:13b",
	// 		size: 7400 * MB,
	// 	},
	// 	{
	// 		name: "orca-mini:13b",
	// 		size: 7400 * MB,
	// 	},
	// 	{
	// 		name: "gemma:7b",
	// 		size: 5000 * MB,
	// 	},
	// 	{
	// 		name: "starcoder2:15b",
	// 		size: 9100 * MB,
	// 	},
	// }

	var chosenModels []model
	switch {
	case max < 10000*MB:
		slog.Info("selecting small models")
		chosenModels = smallModels
	// case max < 30000*MB:
	default:
		slog.Info("selecting medium models")
		chosenModels = mediumModels
		// default:
		// 	slog.Info("selecting large models")
		// 	chosenModels = largModels
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
	consumed := uint64(256 * MB) // Assume some baseline usage
	for i := 0; i < len(req); i++ {
		// Always get at least 2 models, but dont' overshoot VRAM too much or we'll take too long
		if i > 1 && consumed > max {
			slog.Info("achieved target vram exhaustion", "count", i, "vramMB", max/1024/1024, "modelsMB", consumed/1024/1024)
			break
		}
		consumed += chosenModels[i].size
		slog.Info("target vram", "count", i, "vramMB", max/1024/1024, "modelsMB", consumed/1024/1024)

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
