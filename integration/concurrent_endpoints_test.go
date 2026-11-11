//go:build integration && release

package integration

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/types/model"
)

func runConcurrentEmbed(t *testing.T, modelName string) {
	ctx, cancel := context.WithTimeout(t.Context(), 5*time.Minute)
	defer cancel()
	client, _, cleanup := InitServerConnection(ctx, t)
	defer cleanup()
	requireCapability(ctx, t, client, modelName, "embedding")
	skipIfModelTooLargeForVRAM(ctx, t, client, modelName)
	info, err := client.Show(ctx, &api.ShowRequest{Model: modelName})
	if err != nil {
		t.Fatal(err)
	}
	inputs := []any{"A red bicycle.", "The library closes at six on Saturdays.", "A recipe for bread made with flour, water, salt, and yeast."}
	if slices.Contains(info.Capabilities, model.CapabilityVision) {
		image, _, _ := decodeTestImages(t)
		inputs = append(inputs, map[string]string{"text": "A street scene.", "image": base64.StdEncoding.EncodeToString(image)})
	}
	if slices.Contains(info.Capabilities, model.CapabilityAudio) {
		inputs = append(inputs, map[string]string{"audio": base64.StdEncoding.EncodeToString(decodeTestAudio(t))})
	}
	want := make([][]float32, len(inputs))
	for i, input := range inputs {
		response, err := client.Embed(ctx, &api.EmbedRequest{Model: modelName, Input: input})
		if err != nil {
			skipIfMLXUnsupported(t, err)
			t.Fatal(err)
		}
		if len(response.Embeddings) != 1 || len(response.Embeddings[0]) == 0 {
			t.Fatal("empty baseline embedding")
		}
		want[i] = response.Embeddings[0]
	}
	t.Run("concurrent", func(t *testing.T) {
		for i, input := range inputs {
			t.Run(fmt.Sprint(i), func(t *testing.T) {
				t.Parallel()
				got, err := client.Embed(ctx, &api.EmbedRequest{Model: modelName, Input: input})
				if err != nil {
					t.Fatal(err)
				}
				if len(got.Embeddings) != 1 || len(got.Embeddings[0]) != len(want[i]) {
					t.Fatal("embedding dimensions changed")
				}
				for j, value := range got.Embeddings[0] {
					if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
						t.Fatalf("embedding %d value %d is not finite: %g", i, j, value)
					}
				}
				// Quantized kernels can round differently for a wider batch.
				// Use the same whole-vector parity check as the embed suite.
				requireSimilarEmbedding(t, want[i], got.Embeddings[0])
			})
		}
	})
}

func runConcurrentSpeech(t *testing.T, modelName string) {
	ctx, cancel := context.WithTimeout(t.Context(), 5*time.Minute)
	defer cancel()
	client, endpoint, cleanup := InitServerConnection(ctx, t)
	defer cleanup()
	requireCapability(ctx, t, client, modelName, "speech")
	skipIfModelTooLargeForVRAM(ctx, t, client, modelName)
	info, err := client.Show(ctx, &api.ShowRequest{Model: modelName})
	if err != nil {
		t.Fatal(err)
	}
	speech := func(ctx context.Context, text string) (*http.Response, error) {
		body, err := json.Marshal(map[string]string{"model": modelName, "input": text, "response_format": "pcm"})
		if err != nil {
			return nil, err
		}
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, "http://"+endpoint+"/v1/audio/speech", bytes.NewReader(body))
		if err != nil {
			return nil, err
		}
		req.Header.Set("Content-Type", "application/json")
		return http.DefaultClient.Do(req)
	}
	check := func(t *testing.T, ctx context.Context, text string) {
		t.Helper()
		res, err := speech(ctx, text)
		if err != nil {
			t.Fatal(err)
		}
		defer res.Body.Close()
		pcm, err := io.ReadAll(io.LimitReader(res.Body, 8<<20))
		if err != nil {
			t.Fatal(err)
		}
		if res.StatusCode != http.StatusOK {
			err := fmt.Errorf("speech HTTP %d: %s", res.StatusCode, pcm)
			skipIfMLXUnsupported(t, err)
			t.Fatal(err)
		}
		if len(pcm) < 4800 || len(pcm)%2 != 0 || len(pcm) == 8<<20 || bytes.Equal(pcm, make([]byte, len(pcm))) {
			t.Fatalf("invalid or silent PCM stream (%d bytes)", len(pcm))
		}
	}
	// Load on the parent goroutine so an unsupported platform skips the case.
	check(t, ctx, "Hello there.")
	for round := range 2 {
		t.Run(fmt.Sprint(round), func(t *testing.T) {
			if slices.Contains(info.Capabilities, model.CapabilityCompletion) {
				t.Run("chat", func(t *testing.T) {
					t.Parallel()
					var text strings.Builder
					err := client.Chat(ctx, &api.ChatRequest{
						Model: modelName, Messages: []api.Message{{Role: "user", Content: "The box is red. What color is the box? Reply with just its color."}},
						Think: &api.ThinkValue{Value: false}, Options: map[string]any{"temperature": 0, "num_predict": 32},
					}, func(r api.ChatResponse) error { text.WriteString(r.Message.Content); return nil })
					if err != nil || !strings.Contains(strings.ToLower(text.String()), "red") {
						t.Fatalf("chat alongside speech = %q, error %v", text.String(), err)
					}
				})
			}
			for i, text := range []string{"The red bicycle is waiting outside.", "Please bring the books back to the library before six.", "This request is canceled while its peers continue speaking."} {
				t.Run(fmt.Sprint(i), func(t *testing.T) {
					t.Parallel()
					if i < 2 {
						check(t, ctx, text)
						return
					}
					ctx, cancel := context.WithCancel(ctx)
					defer cancel()
					res, err := speech(ctx, text)
					if err != nil {
						t.Fatal(err)
					}
					defer res.Body.Close()
					if res.StatusCode != http.StatusOK {
						t.Fatalf("cancel request HTTP %d", res.StatusCode)
					}
					var sample [2]byte
					if _, err := io.ReadFull(res.Body, sample[:]); err != nil {
						t.Fatal(err)
					}
					cancel()
				})
			}
		})
	}
	check(t, ctx, "All streams have finished.")
}
