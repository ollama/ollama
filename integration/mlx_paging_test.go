//go:build integration

package integration

import (
	"context"
	"fmt"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/types/model"
)

// Unload as soon as the response completes. Requiring disk files or sleeps here
// would hide loss of the background write during a normal model switch. This
// uses only public APIs so the same test can exercise a remote server.
func runMLXPaging(t *testing.T, modelName string) {
	_, timeout := getTimeouts(t)
	ctx, cancel := context.WithTimeout(t.Context(), timeout)
	defer cancel()
	client, _, cleanup := InitServerConnection(ctx, t)
	defer cleanup()
	pullOrSkip(ctx, t, client, modelName)
	info, err := client.Show(ctx, &api.ShowRequest{Model: modelName})
	if err != nil {
		t.Fatal(err)
	}
	if info.Details.Format != "safetensors" {
		t.Skip("persistent cache checks require an MLX model")
	}
	unload := func(t *testing.T) {
		t.Helper()
		request := api.GenerateRequest{Model: modelName, KeepAlive: &api.Duration{}}
		if err := client.Generate(ctx, &request, func(api.GenerateResponse) error { return nil }); err != nil {
			t.Fatal(err)
		}
		// /api/ps proves this isn't just another resident-cache hit.
		for {
			ps, err := client.ListRunning(ctx)
			if err != nil {
				t.Fatal(err)
			}
			loaded := false
			for _, m := range ps.Models {
				loaded = loaded || model.ParseName(m.Name).EqualFold(model.ParseName(modelName))
			}
			if !loaded {
				break
			}
			select {
			case <-ctx.Done():
				t.Fatal(ctx.Err())
			case <-time.After(10 * time.Millisecond):
			}
		}
	}
	// Prior tests may leave pending writes for another conversation. Give this
	// restart check its own runner and bounded shutdown window.
	unload(t)
	req := api.GenerateRequest{
		Model: modelName, Runner: "mlx", Think: &api.ThinkValue{Value: false},
		KeepAlive: &api.Duration{Duration: time.Minute},
		Options:   map[string]any{"num_ctx": 4096, "num_predict": 32, "temperature": 0},
	}
	preloadConcurrentModel(ctx, t, client, req)
	req.Prompt = fmt.Sprintf("Reference %d. Read these notes.\n", time.Now().UnixNano()) +
		strings.Repeat("Each cache entry has a key and an immutable value. Readers check the key before using the value. Writers publish complete entries atomically.\n", 20) +
		"\nWrite a Python function with the signature def sha256_hex(text: str) -> str: that returns the SHA256 hex digest of text encoded as UTF-8. Import hashlib. Use double quotes for string literals. Put the code in a python markdown code block, without a docstring or comments."
	type pagingCase struct {
		name   string
		images []api.ImageData
	}
	cases := []pagingCase{{name: "text"}}
	if slices.Contains(info.Capabilities, model.CapabilityVision) {
		first, second, _ := decodeTestImages(t)
		cases = append(cases, pagingCase{"image", []api.ImageData{first}}, pagingCase{"changed-image", []api.ImageData{second}})
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			req.Images = tc.images
			var baseline string
			var generated int
			// Full prefill and prefix reuse can differ numerically. Use resident
			// reuse as the reference for disk restoration instead of full prefill.
			for round := range 3 {
				var text strings.Builder
				var final api.GenerateResponse
				err := client.Generate(ctx, &req, func(chunk api.GenerateResponse) error {
					text.WriteString(chunk.Response)
					if chunk.Done {
						final = chunk
					}
					return nil
				})
				if err != nil || !final.Done || final.EvalCount == 0 || final.PromptEvalCachedCount == nil {
					t.Fatalf("round %d incomplete: done=%v, generated=%d, cached=%v, error=%v", round, final.Done, final.EvalCount, final.PromptEvalCachedCount, err)
				}
				if round == 0 {
					if *final.PromptEvalCachedCount >= final.PromptEvalCount-16 {
						t.Fatal("new prompt or image unexpectedly reused the complete prefix")
					}
				} else {
					if *final.PromptEvalCachedCount < final.PromptEvalCount-16 {
						t.Fatalf("round %d reused %d of %d prompt tokens", round, *final.PromptEvalCachedCount, final.PromptEvalCount)
					}
					if round == 1 {
						baseline, generated = text.String(), final.EvalCount
					} else if text.String() != baseline || final.EvalCount != generated {
						t.Fatalf("restore changed greedy output: got %q (%d tokens), want %q (%d tokens)", text.String(), final.EvalCount, baseline, generated)
					}
				}
				t.Logf("round %d: prompt=%d cached=%d generated=%d", round, final.PromptEvalCount, *final.PromptEvalCachedCount, final.EvalCount)
				if round == 0 {
					continue
				}
				unload(t)
			}
		})
	}
}
