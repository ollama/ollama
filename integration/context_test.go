//go:build integration

package integration

import (
	"context"
	"log/slog"
	"sync"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
)

func TestLongInputContext(t *testing.T) {
	// Setting NUM_PARALLEL to 1 ensures the allocated context is exactly what
	// we asked for and there is nothing extra that we could spill over into
	t.Setenv("OLLAMA_NUM_PARALLEL", "1")

	// Longer needed for small footprint GPUs
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()
	// Set up the test data
	req := api.ChatRequest{
		Model: smol,
		Messages: []api.Message{
			{
				Role:    "user",
				Content: "Oh, don’t speak to me of Austria. Perhaps I don’t understand things, but Austria never has wished, and does not wish, for war. She is betraying us! Russia alone must save Europe. Our gracious sovereign recognizes his high vocation and will be true to it. That is the one thing I have faith in! Our good and wonderful sovereign has to perform the noblest role on earth, and he is so virtuous and noble that God will not forsake him. He will fulfill his vocation and crush the hydra of revolution, which has become more terrible than ever in the person of this murderer and villain! We alone must avenge the blood of the just one.... Whom, I ask you, can we rely on?... England with her commercial spirit will not and cannot understand the Emperor Alexander’s loftiness of soul. She has refused to evacuate Malta. She wanted to find, and still seeks, some secret motive in our actions. What answer did Novosíltsev get? None. The English have not understood and cannot understand the self-abnegation of our Emperor who wants nothing for himself, but only desires the good of mankind. And what have they promised? Nothing! And what little they have promised they will not perform! Prussia has always declared that Buonaparte is invincible, and that all Europe is powerless before him.... And I don’t believe a word that Hardenburg says, or Haugwitz either. This famous Prussian neutrality is just a trap. I have faith only in God and the lofty destiny of our adored monarch. He will save Europe! What country is this referring to?",
			},
		},
		Stream: &stream,
		Options: map[string]any{
			"temperature": 0,
			"seed":        123,
			"num_ctx":     128,
		},
	}
	client, _, cleanup := InitServerConnection(ctx, t)
	defer cleanup()
	if err := PullIfMissing(ctx, client, req.Model); err != nil {
		t.Fatalf("PullIfMissing failed: %v", err)
	}
	DoChat(ctx, t, client, req, []string{"russia", "german", "france", "england", "austria", "prussia", "europe", "individuals", "coalition", "conflict"}, 120*time.Second, 10*time.Second)
}

func TestContextExhaustion(t *testing.T) {
	// Setting NUM_PARALLEL to 1 ensures the allocated context is exactly what
	// we asked for and there is nothing extra that we could spill over into
	t.Setenv("OLLAMA_NUM_PARALLEL", "1")

	// Longer needed for small footprint GPUs
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()
	// Set up the test data
	req := api.ChatRequest{
		Model: smol,
		Messages: []api.Message{
			{
				Role:    "user",
				Content: "Write me a story in english with a lot of emojis",
			},
		},
		Stream: &stream,
		Options: map[string]any{
			"temperature": 0,
			"seed":        123,
			"num_ctx":     128,
		},
	}
	client, _, cleanup := InitServerConnection(ctx, t)
	defer cleanup()
	if err := PullIfMissing(ctx, client, req.Model); err != nil {
		t.Fatalf("PullIfMissing failed: %v", err)
	}
	DoChat(ctx, t, client, req, []string{"once", "upon", "lived", "sunny", "cloudy", "clear", "water", "time", "travel", "world"}, 120*time.Second, 10*time.Second)
}

// Send multiple generate requests with prior context and ensure the response is coherant and expected
func TestParallelGenerateWithHistory(t *testing.T) {
	modelName := "gpt-oss:20b"
	req, resp := GenerateRequests()
	numParallel := 2
	iterLimit := 2

	softTimeout, hardTimeout := getTimeouts(t)
	ctx, cancel := context.WithTimeout(context.Background(), hardTimeout)
	defer cancel()
	client, _, cleanup := InitServerConnection(ctx, t)
	defer cleanup()
	initialTimeout := 120 * time.Second
	streamTimeout := 20 * time.Second

	// Get the server running (if applicable) warm the model up with a single initial request
	slog.Info("loading", "model", modelName)
	err := client.Generate(ctx,
		&api.GenerateRequest{Model: modelName, KeepAlive: &api.Duration{Duration: 10 * time.Second}},
		func(response api.GenerateResponse) error { return nil },
	)
	if err != nil {
		t.Fatalf("failed to load model %s: %s", modelName, err)
	}
	gpuPercent := getGPUPercent(ctx, t, client, modelName)
	if gpuPercent < 80 {
		slog.Warn("Low GPU percentage - increasing timeouts", "percent", gpuPercent)
		initialTimeout = 240 * time.Second
		streamTimeout = 30 * time.Second
	}

	var wg sync.WaitGroup
	wg.Add(numParallel)
	for i := range numParallel {
		go func(i int) {
			defer wg.Done()
			k := i % len(req)
			req[k].Model = modelName
			for j := 0; j < iterLimit; j++ {
				if time.Now().Sub(started) > softTimeout {
					slog.Info("exceeded soft timeout, winding down test")
					return
				}
				slog.Info("Starting", "thread", i, "iter", j)
				// On slower GPUs it can take a while to process the concurrent requests
				// so we allow a much longer initial timeout
				c := DoGenerate(ctx, t, client, req[k], resp[k], initialTimeout, streamTimeout)
				req[k].Context = c
				req[k].Prompt = "tell me more!"
			}
		}(i)
	}
	wg.Wait()
}

// Send generate requests with prior context and ensure the response is coherant and expected
func TestGenerateWithHistory(t *testing.T) {
	req := api.GenerateRequest{
		Model:     smol,
		Prompt:    rainbowPrompt,
		Stream:    &stream,
		KeepAlive: &api.Duration{Duration: 10 * time.Second},
		Options: map[string]any{
			"num_ctx": 16384,
		},
	}

	softTimeout, hardTimeout := getTimeouts(t)
	ctx, cancel := context.WithTimeout(context.Background(), hardTimeout)
	defer cancel()
	client, _, cleanup := InitServerConnection(ctx, t)
	defer cleanup()

	// Get the server running (if applicable) warm the model up with a single initial request
	slog.Info("loading", "model", req.Model)
	err := client.Generate(ctx,
		&api.GenerateRequest{Model: req.Model, KeepAlive: &api.Duration{Duration: 10 * time.Second}, Options: req.Options},
		func(response api.GenerateResponse) error { return nil },
	)
	if err != nil {
		t.Fatalf("failed to load model %s: %s", req.Model, err)
	}

	req.Context = DoGenerate(ctx, t, client, req, rainbowExpected, 30*time.Second, 20*time.Second)

	for i := 0; i < len(rainbowFollowups); i++ {
		req.Prompt = rainbowFollowups[i]
		if time.Now().Sub(started) > softTimeout {
			slog.Info("exceeded soft timeout, winding down test")
			return
		}
		req.Context = DoGenerate(ctx, t, client, req, rainbowExpected, 30*time.Second, 20*time.Second)
	}
}

// Send multiple chat requests with prior context and ensure the response is coherant and expected
func TestParallelChatWithHistory(t *testing.T) {
	modelName := "gpt-oss:20b"
	req, resp := ChatRequests()
	numParallel := 2
	iterLimit := 2

	softTimeout, hardTimeout := getTimeouts(t)
	ctx, cancel := context.WithTimeout(context.Background(), hardTimeout)
	defer cancel()
	client, _, cleanup := InitServerConnection(ctx, t)
	defer cleanup()
	initialTimeout := 120 * time.Second
	streamTimeout := 20 * time.Second

	// Get the server running (if applicable) warm the model up with a single initial empty request
	slog.Info("loading", "model", modelName)
	err := client.Generate(ctx,
		&api.GenerateRequest{Model: modelName, KeepAlive: &api.Duration{Duration: 10 * time.Second}},
		func(response api.GenerateResponse) error { return nil },
	)
	if err != nil {
		t.Fatalf("failed to load model %s: %s", modelName, err)
	}
	gpuPercent := getGPUPercent(ctx, t, client, modelName)
	if gpuPercent < 80 {
		slog.Warn("Low GPU percentage - increasing timeouts", "percent", gpuPercent)
		initialTimeout = 240 * time.Second
		streamTimeout = 30 * time.Second
	}

	var wg sync.WaitGroup
	wg.Add(numParallel)
	for i := range numParallel {
		go func(i int) {
			defer wg.Done()
			k := i % len(req)
			req[k].Model = modelName
			for j := 0; j < iterLimit; j++ {
				if time.Now().Sub(started) > softTimeout {
					slog.Info("exceeded soft timeout, winding down test")
					return
				}
				slog.Info("Starting", "thread", i, "iter", j)
				// On slower GPUs it can take a while to process the concurrent requests
				// so we allow a much longer initial timeout
				assistant := DoChat(ctx, t, client, req[k], resp[k], initialTimeout, streamTimeout)
				if assistant == nil {
					t.Fatalf("didn't get an assistant response for context")
				}
				req[k].Messages = append(req[k].Messages,
					*assistant,
					api.Message{Role: "user", Content: "tell me more!"},
				)
			}
		}(i)
	}
	wg.Wait()
}

// Send generate requests with prior context and ensure the response is coherant and expected
func TestChatWithHistory(t *testing.T) {
	req := api.ChatRequest{
		Model:     smol,
		Stream:    &stream,
		KeepAlive: &api.Duration{Duration: 10 * time.Second},
		Options: map[string]any{
			"num_ctx": 16384,
		},
		Messages: []api.Message{
			{
				Role:    "user",
				Content: rainbowPrompt,
			},
		},
	}

	softTimeout, hardTimeout := getTimeouts(t)
	ctx, cancel := context.WithTimeout(context.Background(), hardTimeout)
	defer cancel()
	client, _, cleanup := InitServerConnection(ctx, t)
	defer cleanup()

	// Get the server running (if applicable) warm the model up with a single initial request
	slog.Info("loading", "model", req.Model)
	err := client.Generate(ctx,
		&api.GenerateRequest{Model: req.Model, KeepAlive: &api.Duration{Duration: 10 * time.Second}, Options: req.Options},
		func(response api.GenerateResponse) error { return nil },
	)
	if err != nil {
		t.Fatalf("failed to load model %s: %s", req.Model, err)
	}

	assistant := DoChat(ctx, t, client, req, rainbowExpected, 30*time.Second, 20*time.Second)

	for i := 0; i < len(rainbowFollowups); i++ {
		if time.Now().Sub(started) > softTimeout {
			slog.Info("exceeded soft timeout, winding down test")
			return
		}
		req.Messages = append(req.Messages,
			*assistant,
			api.Message{Role: "user", Content: rainbowFollowups[i]},
		)

		assistant = DoChat(ctx, t, client, req, rainbowExpected, 30*time.Second, 20*time.Second)
		if assistant == nil {
			t.Fatalf("didn't get an assistant response for context")
		}
	}
}
