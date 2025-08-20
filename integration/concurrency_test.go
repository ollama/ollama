//go:build integration

package integration

import (
	"context"
	"fmt"
	"log/slog"
	"math"
	"math/rand"
	"os"
	"strconv"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/format"
)

// Send multiple requests in parallel (concurrently) to a single model and ensure responses are expected
func TestConcurrentGenerate(t *testing.T) {
	// Assumes all requests have the same model
	req, resp := GenerateRequests()
	numParallel := int(envconfig.NumParallel() + 1)
	iterLimit := 3

	softTimeout, hardTimeout := getTimeouts(t)
	ctx, cancel := context.WithTimeout(context.Background(), hardTimeout)
	defer cancel()
	client, _, cleanup := InitServerConnection(ctx, t)
	defer cleanup()

	// Get the server running (if applicable) warm the model up with a single initial request
	slog.Info("loading", "model", req[0].Model)
	err := client.Generate(ctx,
		&api.GenerateRequest{Model: req[0].Model, KeepAlive: &api.Duration{Duration: 10 * time.Second}},
		func(response api.GenerateResponse) error { return nil },
	)
	if err != nil {
		t.Fatalf("failed to load model %s: %s", req[0].Model, err)
	}

	var wg sync.WaitGroup
	r := rand.New(rand.NewSource(0))
	wg.Add(numParallel)
	for i := range numParallel {
		go func(i int) {
			defer wg.Done()
			for j := 0; j < iterLimit; j++ {
				if time.Now().Sub(started) > softTimeout {
					slog.Info("exceeded soft timeout, winding down test")
					return
				}
				k := r.Int() % len(req)
				slog.Info("Starting", "thread", i, "iter", j)
				// On slower GPUs it can take a while to process the concurrent requests
				// so we allow a much longer initial timeout
				DoGenerate(ctx, t, client, req[k], resp[k], 120*time.Second, 20*time.Second)
			}
		}(i)
	}
	wg.Wait()
}

// Stress the scheduler and attempt to load more models than will fit to cause thrashing
// This test will always load at least 2 models even on CPU based systems
func TestMultiModelStress(t *testing.T) {
	s := os.Getenv("OLLAMA_MAX_VRAM")
	if s == "" {
		s = "0"
	}

	maxVram, err := strconv.ParseUint(s, 10, 64)
	if err != nil {
		t.Fatal(err)
	}

	smallModels := []string{
		"llama3.2:1b",
		"qwen3:0.6b",
		"gemma:2b",
		"deepseek-r1:1.5b",
		"starcoder2:3b",
	}
	mediumModels := []string{
		"qwen3:8b",
		"llama2",
		"deepseek-r1:7b",
		"mistral",
		"dolphin-mistral",
		"gemma:7b",
		"codellama:7b",
	}

	var chosenModels []string
	switch {
	case maxVram < 10000*format.MebiByte:
		slog.Info("selecting small models")
		chosenModels = smallModels
	default:
		slog.Info("selecting medium models")
		chosenModels = mediumModels
	}

	softTimeout, hardTimeout := getTimeouts(t)
	ctx, cancel := context.WithTimeout(context.Background(), hardTimeout)
	defer cancel()
	client, _, cleanup := InitServerConnection(ctx, t)
	defer cleanup()

	// Make sure all the models are pulled before we get started
	for _, model := range chosenModels {
		require.NoError(t, PullIfMissing(ctx, client, model))
	}

	// Determine how many models we can load in parallel before we exceed VRAM
	// The intent is to go 1 over what can fit so we force the scheduler to thrash
	targetLoadCount := 0
	slog.Info("Loading models to find how many can fit in VRAM before overflowing")
	for i, model := range chosenModels {
		req := &api.GenerateRequest{Model: model}
		slog.Info("loading", "model", model)
		err = client.Generate(ctx, req, func(response api.GenerateResponse) error { return nil })
		if err != nil {
			t.Fatalf("failed to load model %s: %s", model, err)
		}
		targetLoadCount++
		if i > 0 {
			models, err := client.ListRunning(ctx)
			if err != nil {
				t.Fatalf("failed to list running models: %s", err)
			}
			if len(models.Models) < targetLoadCount {
				loaded := []string{}
				for _, m := range models.Models {
					loaded = append(loaded, m.Name)
				}
				slog.Info("found model load capacity", "target", targetLoadCount, "current", loaded, "chosen", chosenModels[:targetLoadCount])
				break
			}
		}
	}
	if targetLoadCount == len(chosenModels) {
		// TODO consider retrying the medium models
		slog.Warn("all models being used without exceeding VRAM, set OLLAMA_MAX_VRAM so test can pick larger models")
	}

	r := rand.New(rand.NewSource(0))
	var wg sync.WaitGroup
	for i := range targetLoadCount {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			reqs, resps := GenerateRequests()
			for j := 0; j < 3; j++ {
				if time.Now().Sub(started) > softTimeout {
					slog.Info("exceeded soft timeout, winding down test")
					return
				}
				k := r.Int() % len(reqs)
				reqs[k].Model = chosenModels[i]
				slog.Info("Starting", "model", reqs[k].Model, "iteration", j, "request", reqs[k].Prompt)
				DoGenerate(ctx, t, client, reqs[k], resps[k],
					120*time.Second, // Be extra patient for the model to load initially
					10*time.Second,  // Once results start streaming, fail if they stall
				)
			}
		}(i)
	}
	go func() {
		for {
			time.Sleep(10 * time.Second)
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
					var procStr string
					switch {
					case m.SizeVRAM == 0:
						procStr = "100% CPU"
					case m.SizeVRAM == m.Size:
						procStr = "100% GPU"
					case m.SizeVRAM > m.Size || m.Size == 0:
						procStr = "Unknown"
					default:
						sizeCPU := m.Size - m.SizeVRAM
						cpuPercent := math.Round(float64(sizeCPU) / float64(m.Size) * 100)
						procStr = fmt.Sprintf("%d%%/%d%%", int(cpuPercent), int(100-cpuPercent))
					}

					slog.Info("loaded model snapshot", "model", m.Name, "CPU/GPU", procStr, "expires", format.HumanTime(m.ExpiresAt, "Never"))
				}
			}
		}
	}()
	wg.Wait()
}
