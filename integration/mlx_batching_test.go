//go:build integration

package integration

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/ollama/ollama/types/model"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
)

// runMLXBatching adds cache reuse and slot lifecycle checks to the release
// concurrency group. Capacity one exercises the same requests serially.
func runMLXBatching(t *testing.T, modelName string) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()
	client, _, cleanup := InitServerConnection(ctx, t)
	defer cleanup()
	pullOrSkip(ctx, t, client, modelName)
	info, err := client.Show(ctx, &api.ShowRequest{Model: modelName})
	if err != nil {
		t.Fatal(err)
	}
	if info.Details.Format != "safetensors" {
		t.Skip("cache reuse checks require an MLX model")
	}
	request := func(prompt string, n int) api.GenerateRequest {
		return api.GenerateRequest{
			Model: modelName, Runner: "mlx", Prompt: prompt,
			Think: &api.ThinkValue{Value: false}, KeepAlive: &api.Duration{Duration: time.Minute},
			Options: map[string]any{"num_predict": n, "temperature": 0, "num_ctx": 4096},
		}
	}
	preloadConcurrentModel(ctx, t, client, request("", 128))
	cities := []string{"Paris", "Tokyo", "Rome", "Madrid", "London", "Berlin", "Ottawa", "Cairo"}
	countries := []string{"France", "Japan", "Italy", "Spain", "the United Kingdom", "Germany", "Canada", "Egypt"}
	requests := make([]api.GenerateRequest, len(cities))
	for i := range requests {
		requests[i] = request("What is the capital of "+countries[i]+"? Answer with just the city name.", 128)
	}
	t.Run("isolation", func(t *testing.T) {
		type result struct {
			text  string
			final api.GenerateResponse
			err   error
		}
		for range 2 {
			results := make(chan result, len(requests))
			start := make(chan struct{})
			for i, req := range requests {
				go func() {
					<-start
					r := result{}
					r.err = client.Generate(ctx, &req, func(chunk api.GenerateResponse) error {
						r.text += chunk.Response
						if chunk.Done {
							r.final = chunk
						}
						return nil
					})
					if !strings.Contains(r.text, cities[i]) {
						r.err = fmt.Errorf("want %s, got %q (%v)", cities[i], r.text, r.err)
					}
					results <- r
				}()
			}
			close(start)
			for range requests {
				r := <-results
				if r.err != nil || !r.final.Done {
					t.Fatalf("response: %+v, error: %v", r, r.err)
				}
			}
		}
	})
	t.Run("reuse", func(t *testing.T) {
		// When every slot is free, a repeat can select its cached prefix. A
		// concurrent repeat may instead land on another available slot.
		for _, req := range requests {
			for round := range 2 {
				var final api.GenerateResponse
				err := client.Generate(ctx, &req, func(chunk api.GenerateResponse) error {
					if chunk.Done {
						final = chunk
					}
					return nil
				})
				if err != nil || !final.Done {
					t.Fatalf("repeat response: %+v, %v", final, err)
				}
				if round == 1 {
					cached := final.PromptEvalCachedCount
					if cached == nil {
						t.Fatal("repeat did not report cached tokens")
					}
					if *cached < final.PromptEvalCount-8 {
						t.Fatalf("repeat reused %d of %d prompt tokens", *cached, final.PromptEvalCount)
					}
				}
			}
		}
	})
	for _, tc := range []struct {
		name   string
		cancel bool
	}{{"cancel", true}, {"finish", false}} {
		t.Run("join_and_leave/"+tc.name, func(t *testing.T) {
			longCtx, stop := context.WithCancel(ctx)
			defer stop()
			first := make(chan struct{}, 1)
			finished := make(chan error, 1)
			var final api.GenerateResponse
			long := request("Write a detailed 2000-word story about a lighthouse keeper. Start the story immediately and keep writing.", 256)
			go func() {
				finished <- client.Generate(longCtx, &long, func(chunk api.GenerateResponse) error {
					if chunk.Response != "" || chunk.Thinking != "" {
						select {
						case first <- struct{}{}:
						default:
						}
					}
					if chunk.Done {
						final = chunk
					}
					return nil
				})
			}()
			select {
			case <-first:
			case err := <-finished:
				t.Fatalf("long request ended before output: %v", err)
			case <-ctx.Done():
				t.Fatal(ctx.Err())
			}
			// Capacity one queues the peer; cancellation still frees the slot.
			if tc.cancel && envconfig.NumParallel() <= 1 {
				stop()
			}
			short := requests[1]
			var text string
			err := client.Generate(ctx, &short, func(chunk api.GenerateResponse) error { text += chunk.Response; return nil })
			if err != nil || !strings.Contains(text, "Tokyo") {
				t.Fatalf("joining request: %q, %v", text, err)
			}
			if envconfig.NumParallel() > 1 {
				select {
				case err := <-finished:
					t.Fatalf("long request finished before short joining request: %v", err)
				default:
				}
			}
			if tc.cancel {
				stop()
			}
			err = <-finished
			if tc.cancel {
				if !errors.Is(err, context.Canceled) {
					t.Fatalf("cancel: %v", err)
				}
			} else if err != nil || !final.Done || final.EvalCount != 256 {
				t.Fatalf("surviving request: %+v, %v", final, err)
			}
			text = ""
			err = client.Generate(ctx, &requests[2], func(chunk api.GenerateResponse) error { text += chunk.Response; return nil })
			if err != nil || !strings.Contains(text, "Rome") {
				t.Fatalf("slot reuse: %q, %v", text, err)
			}
		})
	}

	t.Run("independent_grammar", func(t *testing.T) {
		errs := make(chan error, 2)
		for _, city := range cities[:2] {
			go func() {
				req := request("Return the city in JSON.", 32)
				req.Format = []byte(fmt.Sprintf(`{"type":"object","properties":{"city":{"type":"string","enum":[%q]}},"required":["city"],"additionalProperties":false}`, city))
				var text string
				err := client.Generate(ctx, &req, func(chunk api.GenerateResponse) error { text += chunk.Response; return nil })
				if err == nil && !strings.Contains(text, city) {
					err = fmt.Errorf("grammar want %s, got %q", city, text)
				}
				errs <- err
			}()
		}
		for range 2 {
			if err := <-errs; err != nil {
				t.Fatal(err)
			}
		}
	})
}

// TestMLXBatchingVision mixes text and two different images at the same prompt
// position. Repeats exercise both cached image identity and per-request layout.
func TestMLXBatchingVision(t *testing.T) {
	if testModel == "" {
		t.Skip("set OLLAMA_TEST_MODEL to a published MLX vision model")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()
	client, _, cleanup := InitServerConnection(ctx, t)
	defer cleanup()
	info, err := client.Show(ctx, &api.ShowRequest{Model: testModel})
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Contains(info.Capabilities, model.CapabilityVision) {
		t.Skip("model has no vision capability")
	}
	road, desk, _ := decodeTestImages(t)
	cases := []struct {
		image        api.ImageData
		prompt, want string
	}{
		{road, "Is this scene crossing a road or working at a desk? Begin with CROSSING or WORKING, then describe the image in detail.", "crossing"},
		{desk, "Is this scene crossing a road or working at a desk? Begin with CROSSING or WORKING, then describe the image in detail.", "working"},
		{nil, "What is the capital of Japan? Begin with the city name and then describe the city's history in detail.", "tokyo"},
	}
	for round := range 2 {
		start := make(chan struct{})
		results := make(chan error, len(cases))
		for _, tc := range cases {
			go func() {
				<-start
				req := api.GenerateRequest{Model: testModel, Runner: "mlx", Prompt: tc.prompt, Think: &api.ThinkValue{Value: false}, KeepAlive: &api.Duration{Duration: time.Minute}, Options: map[string]any{"num_predict": 128, "num_ctx": 4096, "temperature": 0}}
				if tc.image != nil {
					req.Images = []api.ImageData{tc.image}
				}
				var text string
				var final api.GenerateResponse
				err := client.Generate(ctx, &req, func(chunk api.GenerateResponse) error {
					text += chunk.Response
					if chunk.Done {
						final = chunk
					}
					return nil
				})
				if err == nil && (!final.Done || final.EvalCount == 0 || !strings.Contains(strings.ToLower(text), tc.want)) {
					err = fmt.Errorf("want %q in image/text response, got %q (done=%v tokens=%d)", tc.want, text, final.Done, final.EvalCount)
				}
				results <- err
			}()
		}
		close(start)
		for range cases {
			if err := <-results; err != nil {
				t.Errorf("round %d: %v", round, err)
			}
		}
		if t.Failed() {
			return
		}
	}
}

// TestMLXBatchingLongPrompts crosses the prefill chunk and sparse-attention
// boundaries with unequal histories. Longer replies keep the short-prompt row
// generating while its peer processes later prompt chunks.
func TestMLXBatchingLongPrompts(t *testing.T) {
	if testModel == "" {
		t.Skip("set OLLAMA_TEST_MODEL to a published MLX model")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 8*time.Minute)
	defer cancel()
	client, _, cleanup := InitServerConnection(ctx, t)
	defer cleanup()
	start := make(chan struct{})
	cities := []string{"Paris", "Tokyo", "Rome", "Madrid", "London", "Berlin", "Ottawa", "Cairo"}
	countries := []string{"France", "Japan", "Italy", "Spain", "the United Kingdom", "Germany", "Canada", "Egypt"}
	rows := max(2, min(len(cities), int(envconfig.NumParallel())))
	t.Logf("keeping %d requests active across unequal prompt lengths", rows)
	results := make(chan error, rows)
	for i, city := range cities[:rows] {
		go func() {
			repetitions := 20
			if i%2 == 1 {
				repetitions = 180
			}
			prompt := strings.Repeat("The island has an old lighthouse, a harbor, and a village beside the sea. ", repetitions)
			req := api.GenerateRequest{Model: testModel, Runner: "mlx", Prompt: prompt + "\nIgnore the preceding background. What is the capital of " + countries[i] + "? Begin with the city name, then write a detailed 1000-word history of the city.", Think: &api.ThinkValue{Value: false}, KeepAlive: &api.Duration{Duration: time.Minute}, Options: map[string]any{"num_predict": 512, "num_ctx": 8192, "temperature": 0}}
			<-start
			var text, thinking string
			var final api.GenerateResponse
			err := client.Generate(ctx, &req, func(chunk api.GenerateResponse) error {
				text += chunk.Response
				thinking += chunk.Thinking
				if chunk.Done {
					final = chunk
				}
				return nil
			})
			if err == nil && (!final.Done || !strings.Contains(text, city)) {
				err = fmt.Errorf("row %d want %s, got %q (done=%v tokens=%d reason=%s thinking=%q)", i, city, text, final.Done, final.EvalCount, final.DoneReason, thinking)
			}
			if err == nil && i%2 == 1 && final.PromptEvalCount <= 2048 {
				err = fmt.Errorf("long prompt only has %d tokens; did not cross chunk/sparse boundary", final.PromptEvalCount)
			}
			results <- err
		}()
	}
	close(start)
	for range rows {
		if err := <-results; err != nil {
			t.Error(err)
		}
	}
}
