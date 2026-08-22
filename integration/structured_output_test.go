//go:build integration

package integration

import (
	"bytes"
	"context"
	"encoding/json"
	"math"
	"testing"

	"github.com/ollama/ollama/api"
)

var structuredOutputSchema = json.RawMessage(`{
  "type": "object",
  "properties": {
    "color": {"type": "string", "enum": ["blue", "violet"]},
    "count": {"type": "integer", "minimum": 1, "maximum": 3}
  },
  "required": ["color", "count"],
  "additionalProperties": false
}`)

func validateStructuredObject(t *testing.T, content string) {
	t.Helper()
	if !json.Valid([]byte(content)) {
		t.Fatalf("response is not valid JSON: %q", content)
	}
	var object map[string]json.RawMessage
	if err := json.Unmarshal([]byte(content), &object); err != nil {
		t.Fatal(err)
	}
	if len(object) != 2 || object["color"] == nil || object["count"] == nil {
		t.Fatalf("response has the wrong fields: %s", content)
	}
	var color string
	if err := json.Unmarshal(object["color"], &color); err != nil || (color != "blue" && color != "violet") {
		t.Fatalf("color = %q, %v", color, err)
	}
	var count int
	if err := json.Unmarshal(object["count"], &count); err != nil || count < 1 || count > 3 {
		t.Fatalf("count = %d, %v", count, err)
	}
}

func validateConstrainedLogprobs(t *testing.T, logprobs []api.Logprob) {
	t.Helper()
	if len(logprobs) == 0 {
		t.Fatal("constrained response did not include logprobs")
	}
	for i, entry := range logprobs {
		if math.IsInf(entry.Logprob, 0) || math.IsNaN(entry.Logprob) {
			t.Fatalf("logprob[%d] is not finite: %v", i, entry.Logprob)
		}
		for j, alternative := range entry.TopLogprobs {
			if math.IsInf(alternative.Logprob, 0) || math.IsNaN(alternative.Logprob) {
				t.Fatalf("logprob[%d].top_logprobs[%d] is not finite: %v", i, j, alternative.Logprob)
			}
		}
	}
}

const structuredOutputMLXModel = "qwen3.5:2b-nvfp4"

func registerStructuredOutputCases() {
	registerModelMinVRAM([]integrationModel{{Name: structuredOutputMLXModel, MinVRAMGB: 4}})
	registerModelIntegrationCases("structured-output", testModels([]string{smol, structuredOutputMLXModel}), runStructuredOutput)
}

func runStructuredOutput(t *testing.T, model string) {
	skipRegisteredMinVRAM(t, model)
	ctx, cancel := context.WithTimeout(context.Background(), apiTestTimeout)
	defer cancel()
	client, _, cleanup := InitServerConnection(ctx, t)
	defer cleanup()
	pullOrSkip(ctx, t, client, model)
	noThink := api.ThinkValue{Value: false}
	preloadGenerateModel(ctx, t, client, api.GenerateRequest{
		Model:  model,
		Prompt: "Respond with one word.",
		Think:  &noThink,
		Options: map[string]any{
			"temperature": 0,
			"num_predict": 1,
		},
	})

	t.Run("generate schema adversarial prompt", func(t *testing.T) {
		// The prompt asks for prose, so only grammar enforcement can make
		// the response satisfy the schema.
		stream := false
		req := api.GenerateRequest{
			Model:  model,
			Prompt: "Say hi",
			Stream: &stream,
			Format: structuredOutputSchema,
			Think:  &noThink,
			Options: map[string]any{
				"temperature": 0,
				"seed":        17,
				"num_predict": 96,
			},
		}
		var content bytes.Buffer
		if err := client.Generate(ctx, &req, func(response api.GenerateResponse) error {
			content.WriteString(response.Response)
			return nil
		}); err != nil {
			t.Fatal(err)
		}
		validateStructuredObject(t, content.String())
	})

	t.Run("generate builtin JSON streaming", func(t *testing.T) {
		stream := true
		req := api.GenerateRequest{
			Model:  model,
			Prompt: "Return the smallest possible JSON value. Output JSON only.",
			Stream: &stream,
			Format: json.RawMessage(`"json"`),
			Think:  &noThink,
			Options: map[string]any{
				"temperature": 0,
				"seed":        17,
				"num_predict": 96,
			},
		}
		var content bytes.Buffer
		if err := client.Generate(ctx, &req, func(response api.GenerateResponse) error {
			content.WriteString(response.Response)
			return nil
		}); err != nil {
			t.Fatal(err)
		}
		if !json.Valid(content.Bytes()) {
			t.Fatalf("response is not valid JSON: %q", content.String())
		}
	})

	t.Run("generate schema greedy with logprobs", func(t *testing.T) {
		stream := false
		req := api.GenerateRequest{
			Model:       model,
			Prompt:      "Return an object with a color and a small count. Output JSON only.",
			Stream:      &stream,
			Format:      structuredOutputSchema,
			Think:       &noThink,
			Logprobs:    true,
			TopLogprobs: 20,
			Options: map[string]any{
				"temperature": 0,
				"seed":        23,
				"num_predict": 96,
			},
		}
		var content bytes.Buffer
		var logprobs []api.Logprob
		if err := client.Generate(ctx, &req, func(response api.GenerateResponse) error {
			content.WriteString(response.Response)
			logprobs = append(logprobs, response.Logprobs...)
			return nil
		}); err != nil {
			t.Fatal(err)
		}
		validateStructuredObject(t, content.String())
		validateConstrainedLogprobs(t, logprobs)
	})

	t.Run("generate schema sampled streaming", func(t *testing.T) {
		stream := true
		req := api.GenerateRequest{
			Model:  model,
			Prompt: "Return a color and count as a JSON object. Output JSON only.",
			Stream: &stream,
			Format: structuredOutputSchema,
			Think:  &noThink,
			Options: map[string]any{
				"temperature": 0.7,
				"seed":        29,
				"num_predict": 96,
			},
		}
		var content bytes.Buffer
		if err := client.Generate(ctx, &req, func(response api.GenerateResponse) error {
			content.WriteString(response.Response)
			return nil
		}); err != nil {
			t.Fatal(err)
		}
		validateStructuredObject(t, content.String())
	})

	t.Run("chat schema greedy", func(t *testing.T) {
		stream := false
		req := api.ChatRequest{
			Model: model,
			Messages: []api.Message{{
				Role:    "user",
				Content: "Return an object with a color and a small count. Output JSON only.",
			}},
			Stream: &stream,
			Format: structuredOutputSchema,
			Think:  &noThink,
			Options: map[string]any{
				"temperature": 0,
				"seed":        31,
				"num_predict": 96,
			},
		}
		var content bytes.Buffer
		if err := client.Chat(ctx, &req, func(response api.ChatResponse) error {
			content.WriteString(response.Message.Content)
			return nil
		}); err != nil {
			t.Fatal(err)
		}
		validateStructuredObject(t, content.String())
	})
}
