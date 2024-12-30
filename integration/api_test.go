//go:build integration

package integration

import (
	"bytes"
	"context"
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
)

func TestAPIGenerate(t *testing.T) {
	initialTimeout := 60 * time.Second
	streamTimeout := 30 * time.Second
	ctx, cancel := context.WithTimeout(context.Background(), 1*time.Minute)
	defer cancel()
	// Set up the test data
	req := api.GenerateRequest{
		Model:  "orca-mini",
		Prompt: "why is the sky blue?",
		Options: map[string]interface{}{
			"temperature": 0,
			"seed":        123,
		},
	}
	anyResp := []string{"rayleigh", "scattering"}

	client, _, cleanup := InitServerConnection(ctx, t)
	defer cleanup()
	if err := PullIfMissing(ctx, client, req.Model); err != nil {
		t.Fatalf("pull failed %s", err)
	}

	tests := []struct {
		name   string
		stream bool
	}{
		{
			name:   "stream",
			stream: true,
		},
		{
			name:   "no_stream",
			stream: false,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			stallTimer := time.NewTimer(initialTimeout)
			var buf bytes.Buffer
			fn := func(response api.GenerateResponse) error {
				// fmt.Print(".")
				// Fields that must always be present
				if response.Model == "" {
					t.Errorf("response missing model: %#v", response)
				}
				if response.Done {
					// Required fields for final updates:
					if response.DoneReason == "" && *req.Stream {
						// TODO - is the lack of done reason on non-stream a bug?
						t.Errorf("final response missing done_reason: %#v", response)
					}
					if response.Metrics.TotalDuration == 0 {
						t.Errorf("final response missing total_duration: %#v", response)
					}
					if response.Metrics.LoadDuration == 0 {
						t.Errorf("final response missing load_duration: %#v", response)
					}
					if response.Metrics.PromptEvalDuration == 0 {
						t.Errorf("final response missing prompt_eval_duration: %#v", response)
					}
					if response.Metrics.EvalCount == 0 {
						t.Errorf("final response missing eval_count: %#v", response)
					}
					if response.Metrics.EvalDuration == 0 {
						t.Errorf("final response missing eval_duration: %#v", response)
					}
					if len(response.Context) == 0 {
						t.Errorf("final response missing context: %#v", response)
					}

					// Note: caching can result in no prompt eval count, so this can't be verified reliably
					// if response.Metrics.PromptEvalCount == 0 {
					// 	t.Errorf("final response missing prompt_eval_count: %#v", response)
					// }

				} // else incremental response, nothing to check right now...
				buf.Write([]byte(response.Response))
				if !stallTimer.Reset(streamTimeout) {
					return fmt.Errorf("stall was detected while streaming response, aborting")
				}
				return nil
			}

			done := make(chan int)
			var genErr error
			go func() {
				req.Stream = &test.stream
				genErr = client.Generate(ctx, &req, fn)
				done <- 0
			}()

			select {
			case <-stallTimer.C:
				if buf.Len() == 0 {
					t.Errorf("generate never started.  Timed out after :%s", initialTimeout.String())
				} else {
					t.Errorf("generate stalled.  Response so far:%s", buf.String())
				}
			case <-done:
				if genErr != nil {
					t.Fatalf("failed with %s request prompt %s ", req.Model, req.Prompt)
				}
				// Verify the response contains the expected data
				response := buf.String()
				atLeastOne := false
				for _, resp := range anyResp {
					if strings.Contains(strings.ToLower(response), resp) {
						atLeastOne = true
						break
					}
				}
				if !atLeastOne {
					t.Errorf("none of %v found in %s", anyResp, response)
				}
			case <-ctx.Done():
				t.Error("outer test context done while waiting for generate")
			}
		})
	}

	// Validate PS while we're at it...
	resp, err := client.ListRunning(ctx)
	if err != nil {
		t.Errorf("list models API error: %s", err)
	}
	if resp == nil || len(resp.Models) == 0 {
		t.Errorf("list models API returned empty list while model should still be loaded")
	}
	// Find the model we just loaded and verify some attributes
	found := false
	for _, model := range resp.Models {
		if strings.Contains(model.Name, req.Model) {
			found = true
			if model.Model == "" {
				t.Errorf("model field omitted: %#v", model)
			}
			if model.Size == 0 {
				t.Errorf("size omitted: %#v", model)
			}
			if model.Digest == "" {
				t.Errorf("digest omitted: %#v", model)
			}
			verifyModelDetails(t, model.Details)
			var nilTime time.Time
			if model.ExpiresAt == nilTime {
				t.Errorf("expires_at omitted: %#v", model)
			}
			// SizeVRAM could be zero.
		}
	}
	if !found {
		t.Errorf("unable to locate running model: %#v", resp)
	}
}

func TestAPIChat(t *testing.T) {
	initialTimeout := 60 * time.Second
	streamTimeout := 30 * time.Second
	ctx, cancel := context.WithTimeout(context.Background(), 1*time.Minute)
	defer cancel()
	// Set up the test data
	req := api.ChatRequest{
		Model: "orca-mini",
		Messages: []api.Message{
			{
				Role:    "user",
				Content: "why is the sky blue?",
			},
		},
		Options: map[string]interface{}{
			"temperature": 0,
			"seed":        123,
		},
	}
	anyResp := []string{"rayleigh", "scattering"}

	client, _, cleanup := InitServerConnection(ctx, t)
	defer cleanup()
	if err := PullIfMissing(ctx, client, req.Model); err != nil {
		t.Fatalf("pull failed %s", err)
	}

	tests := []struct {
		name   string
		stream bool
	}{
		{
			name:   "stream",
			stream: true,
		},
		{
			name:   "no_stream",
			stream: false,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			stallTimer := time.NewTimer(initialTimeout)
			var buf bytes.Buffer
			fn := func(response api.ChatResponse) error {
				// fmt.Print(".")
				// Fields that must always be present
				// slog.Info("XXX response", "response", response)
				if response.Model == "" {
					t.Errorf("response missing model: %#v", response)
				}
				if response.Done {
					// Required fields for final updates:
					var nilTime time.Time
					if response.CreatedAt == nilTime {
						t.Errorf("final response missing total_duration: %#v", response)
					}
					if response.DoneReason == "" {
						t.Errorf("final response missing done_reason: %#v", response)
					}
					if response.Metrics.TotalDuration == 0 {
						t.Errorf("final response missing total_duration: %#v", response)
					}
					if response.Metrics.LoadDuration == 0 {
						t.Errorf("final response missing load_duration: %#v", response)
					}
					if response.Metrics.PromptEvalDuration == 0 {
						t.Errorf("final response missing prompt_eval_duration: %#v", response)
					}
					if response.Metrics.EvalCount == 0 {
						t.Errorf("final response missing eval_count: %#v", response)
					}
					if response.Metrics.EvalDuration == 0 {
						t.Errorf("final response missing eval_duration: %#v", response)
					}

					// Note: caching can result in no prompt eval count, so this can't be verified reliably
					// if response.Metrics.PromptEvalCount == 0 {
					// 	t.Errorf("final response missing prompt_eval_count: %#v", response)
					// }
				} // else incremental response, nothing to check right now...
				buf.Write([]byte(response.Message.Content))
				if !stallTimer.Reset(streamTimeout) {
					return fmt.Errorf("stall was detected while streaming response, aborting")
				}
				return nil
			}

			done := make(chan int)
			var genErr error
			go func() {
				req.Stream = &test.stream
				genErr = client.Chat(ctx, &req, fn)
				done <- 0
			}()

			select {
			case <-stallTimer.C:
				if buf.Len() == 0 {
					t.Errorf("chat never started.  Timed out after :%s", initialTimeout.String())
				} else {
					t.Errorf("chat stalled.  Response so far:%s", buf.String())
				}
			case <-done:
				if genErr != nil {
					t.Fatalf("failed with %s request prompt %v", req.Model, req.Messages)
				}
				// Verify the response contains the expected data
				response := buf.String()
				atLeastOne := false
				for _, resp := range anyResp {
					if strings.Contains(strings.ToLower(response), resp) {
						atLeastOne = true
						break
					}
				}
				if !atLeastOne {
					t.Errorf("none of %v found in %s", anyResp, response)
				}
			case <-ctx.Done():
				t.Error("outer test context done while waiting for chat")
			}
		})
	}
}

func TestAPIListModels(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	client, _, cleanup := InitServerConnection(ctx, t)
	defer cleanup()

	// Make sure we have at least one model so an empty list can be considered a failure
	if err := PullIfMissing(ctx, client, "orca-mini"); err != nil {
		t.Fatalf("pull failed %s", err)
	}

	resp, err := client.List(ctx)
	if err != nil {
		t.Fatalf("unable to list models: %s", err)
	}
	if len(resp.Models) == 0 {
		t.Fatalf("list should not be empty")
	}
	model := resp.Models[0]
	if model.Name == "" {
		t.Errorf("first model name empty: %#v", model)
	}
	var nilTime time.Time
	if model.ModifiedAt == nilTime {
		t.Errorf("first model modified_at empty: %#v", model)
	}
	if model.Size == 0 {
		t.Errorf("first model size empty: %#v", model)
	}
	if model.Digest == "" {
		t.Errorf("first model digest empty: %#v", model)
	}
	verifyModelDetails(t, model.Details)
}

func verifyModelDetails(t *testing.T, details api.ModelDetails) {
	// Not always populated?
	// if model.Details.ParentModel == "" {
	// 	t.Errorf("first model details.parent_model empty: %#v", model)
	// }
	if details.Format == "" {
		t.Errorf("first model details.format empty: %#v", details)
	}
	if details.Family == "" {
		t.Errorf("first model details.family empty: %#v", details)
	}
	// Sometimes empty?
	// if len(details.Families) == 0 {
	// 	t.Errorf("first model details.families empty: %#v", details)
	// }
	if details.ParameterSize == "" {
		t.Errorf("first model details.parameter_size empty: %#v", details)
	}
	if details.QuantizationLevel == "" {
		t.Errorf("first model details.quantization_level empty: %#v", details)
	}
}

func TestAPIShowModel(t *testing.T) {
	modelName := "llama3.2"
	ctx, cancel := context.WithTimeout(context.Background(), 1*time.Minute)
	defer cancel()
	client, _, cleanup := InitServerConnection(ctx, t)
	defer cleanup()

	if err := PullIfMissing(ctx, client, modelName); err != nil {
		t.Fatalf("pull failed %s", err)
	}
	resp, err := client.Show(ctx, &api.ShowRequest{Name: modelName})
	if err != nil {
		t.Fatalf("unable to show model: %s", err)
	}
	if resp.License == "" {
		t.Errorf("%s missing license: %#v", modelName, resp)
	}
	if resp.Modelfile == "" {
		t.Errorf("%s missing modelfile: %#v", modelName, resp)
	}
	if resp.Parameters == "" {
		t.Errorf("%s missing parameters: %#v", modelName, resp)
	}
	if resp.Template == "" {
		t.Errorf("%s missing template: %#v", modelName, resp)
	}
	// llama3 missing system
	// if resp.System == "" {
	// 	t.Errorf("%s missing system: %#v", modelName, resp)
	// }
	verifyModelDetails(t, resp.Details)
	// llama3 missing messages
	// if len(resp.Messages) == 0 {
	// 	t.Errorf("%s missing messages: %#v", modelName, resp)
	// }
	if len(resp.ModelInfo) == 0 {
		t.Errorf("%s missing model_info: %#v", modelName, resp)
	}
	// llama3 missing projectors
	// if len(resp.ProjectorInfo) == 0 {
	// 	t.Errorf("%s missing projector_info: %#v", modelName, resp)
	// }
	var nilTime time.Time
	if resp.ModifiedAt == nilTime {
		t.Errorf("%s missing modified_at: %#v", modelName, resp)
	}
}

func TestAPIEmbeddings(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 1*time.Minute)
	defer cancel()
	client, _, cleanup := InitServerConnection(ctx, t)
	defer cleanup()
	req := api.EmbeddingRequest{
		Model:  "orca-mini",
		Prompt: "why is the sky blue?",
		Options: map[string]interface{}{
			"temperature": 0,
			"seed":        123,
		},
	}

	if err := PullIfMissing(ctx, client, req.Model); err != nil {
		t.Fatalf("pull failed %s", err)
	}

	resp, err := client.Embeddings(ctx, &req)
	if err != nil {
		t.Fatalf("embeddings call failed %s", err)
	}
	if len(resp.Embedding) == 0 {
		t.Errorf("zero length embedding response")
	}
}
