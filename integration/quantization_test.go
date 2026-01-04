//go:build integration && models

package integration

import (
	"bytes"
	"context"
	"fmt"
	"log/slog"
	"strings"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
)

func TestQuantization(t *testing.T) {
	sourceModels := []string{
		"qwen2.5:0.5b-instruct-fp16",
	}
	quantizations := []string{
		"Q8_0",
		"Q4_K_S",
		"Q4_K_M",
		"Q4_K",
	}
	softTimeout, hardTimeout := getTimeouts(t)
	started := time.Now()
	slog.Info("Setting timeouts", "soft", softTimeout, "hard", hardTimeout)
	ctx, cancel := context.WithTimeout(context.Background(), hardTimeout)
	defer cancel()
	client, _, cleanup := InitServerConnection(ctx, t)
	defer cleanup()

	for _, base := range sourceModels {
		if err := PullIfMissing(ctx, client, base); err != nil {
			t.Fatalf("pull failed %s", err)
		}
		for _, quant := range quantizations {
			newName := fmt.Sprintf("%s__%s", base, quant)
			t.Run(newName, func(t *testing.T) {
				if time.Now().Sub(started) > softTimeout {
					t.Skip("skipping remaining tests to avoid excessive runtime")
				}
				req := &api.CreateRequest{
					Model:        newName,
					Quantization: quant,
					From:         base,
				}
				fn := func(resp api.ProgressResponse) error {
					// fmt.Print(".")
					return nil
				}
				t.Logf("quantizing: %s -> %s", base, quant)
				if err := client.Create(ctx, req, fn); err != nil {
					t.Fatalf("create failed %s", err)
				}
				defer func() {
					req := &api.DeleteRequest{
						Model: newName,
					}
					t.Logf("deleting: %s -> %s", base, quant)
					if err := client.Delete(ctx, req); err != nil {
						t.Logf("failed to clean up %s: %s", req.Model, err)
					}
				}()
				// Check metadata on the model
				resp, err := client.Show(ctx, &api.ShowRequest{Name: newName})
				if err != nil {
					t.Fatalf("unable to show model: %s", err)
				}
				if !strings.Contains(resp.Details.QuantizationLevel, quant) {
					t.Fatalf("unexpected quantization for %s:\ngot: %s", newName, resp.Details.QuantizationLevel)
				}

				stream := true
				chatReq := api.ChatRequest{
					Model: newName,
					Messages: []api.Message{
						{
							Role:    "user",
							Content: blueSkyPrompt,
						},
					},
					KeepAlive: &api.Duration{Duration: 3 * time.Second},
					Options: map[string]any{
						"seed":        42,
						"temperature": 0.0,
					},
					Stream: &stream,
				}
				t.Logf("verifying: %s -> %s", base, quant)

				// Some smaller quantizations can cause models to have poor quality
				// or get stuck in repetition loops, so we stop as soon as we have any matches
				reqCtx, reqCancel := context.WithCancel(ctx)
				atLeastOne := false
				var buf bytes.Buffer
				chatfn := func(response api.ChatResponse) error {
					buf.Write([]byte(response.Message.Content))
					fullResp := strings.ToLower(buf.String())
					for _, resp := range blueSkyExpected {
						if strings.Contains(fullResp, resp) {
							atLeastOne = true
							t.Log(fullResp)
							reqCancel()
							break
						}
					}
					return nil
				}

				done := make(chan int)
				var genErr error
				go func() {
					genErr = client.Chat(reqCtx, &chatReq, chatfn)
					done <- 0
				}()

				select {
				case <-done:
					if genErr != nil && !atLeastOne {
						t.Fatalf("failed with %s request prompt %s ", chatReq.Model, chatReq.Messages[0].Content)
					}
				case <-ctx.Done():
					t.Error("outer test context done while waiting for generate")
				}

				t.Logf("passed")

			})
		}
	}
}
