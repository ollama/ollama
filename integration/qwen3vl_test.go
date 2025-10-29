//go:build integration

package integration

import (
	"context"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
)

// var MODEL = "qwen3vl-odc-dev"
// var MODEL = "qwen3vl-thinking-odc-dev"

// TestQwen3VLScenarios exercises common Qwen3-VL cases using integration helpers
func TestQwen3VLScenarios(t *testing.T) {
	tests := []struct {
		name     string
		messages []api.Message
		tools    []api.Tool
		image    string
		anyResp  []string
	}{
		{
			name: "Text-Only Scenario",
			messages: []api.Message{
				{Role: "system", Content: "You are a helpful assistant."},
				{Role: "user", Content: "Write a short haiku about autumn."},
			},
			anyResp: []string{"haiku", "autumn", "fall"},
		},
		{
			name: "Single Image Scenario",
			messages: []api.Message{
				{
					Role:    "system",
					Content: "You are a helpful assistant that can see images.",
				},
				{
					Role:    "user",
					Content: "What is this flower? Is it poisonous to cats?",
				},
			},
			image:   "testdata/question.png",
			anyResp: []string{"flower", "plant", "poison", "cat"},
		},
		{
			name: "Tools Scenario",
			messages: []api.Message{
				{
					Role:    "system",
					Content: "You can call tools when needed. Return tool calls when actions are needed.",
				},
				{Role: "user", Content: "What's the weather in San Francisco now?"},
			},
			tools: []api.Tool{
				{
					Type: "function",
					Function: api.ToolFunction{
						Name:        "get_weather",
						Description: "Get current weather for a city.",
						Parameters: api.ToolFunctionParameters{
							Type: "object",
							Properties: map[string]api.ToolProperty{
								"city": {
									Type:        api.PropertyType{"string"},
									Description: "The city to get the weather for",
								},
							},
							Required: []string{"city"},
						},
					},
				},
			},
			anyResp: []string{"san francisco", "weather", "temperature"},
		},
	}

	// models := []string{"qwen3-vl:8b", "qwen3-vl:30b"}
	models := []string{"qwen3vl-odc-dev"} // , "qwen3vl-thinking-odc-dev"}

	for _, model := range models {
		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				// Load image if specified
				if tt.image != "" {
					imageData := loadImageData(t, tt.image)
					// Add image to the last user message
					if len(tt.messages) > 0 {
						lastMessage := &tt.messages[len(tt.messages)-1]
						if lastMessage.Role == "user" {
							lastMessage.Images = []api.ImageData{imageData}
						}
					}
				}

				// Build chat request
				req := api.ChatRequest{
					Model:    model,
					Messages: tt.messages,
					Tools:    tt.tools,
					Stream:   &stream,
					Options: map[string]any{
						"seed":        42,
						"temperature": 0.0,
					},
				}

				// Use integration helpers
				ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
				defer cancel()
				client, _, cleanup := InitServerConnection(ctx, t)
				defer cleanup()

				// Skip pulling/preloading when using an existing (cloud) server
				if os.Getenv("OLLAMA_TEST_EXISTING") == "" {
					if err := PullIfMissing(ctx, client, req.Model); err != nil {
						t.Fatal(err)
					}
					// Preload model once to reduce startup latency
					_ = client.Generate(ctx, &api.GenerateRequest{Model: req.Model}, func(r api.GenerateResponse) error { return nil })
				}

				// If this is a tools scenario, validate tool_calls instead of content
				if len(tt.tools) > 0 {
					var gotCalls []api.ToolCall
					err := client.Chat(ctx, &req, func(r api.ChatResponse) error {
						if len(r.Message.ToolCalls) > 0 {
							gotCalls = append(gotCalls, r.Message.ToolCalls...)
						}
						return nil
					})
					if err != nil {
						t.Fatalf("chat error: %v", err)
					}
					if len(gotCalls) == 0 {
						t.Fatalf("expected at least one tool call, got none")
					}
					// Optionally validate the first tool name matches the offered tool
					if gotCalls[0].Function.Name == "" {
						t.Fatalf("tool call missing function name: %#v", gotCalls[0])
					}
					return
				}

				// Otherwise, validate content contains any of the expected substrings
				DoChat(ctx, t, client, req, toLowerSlice(tt.anyResp), 240*time.Second, 30*time.Second)
			})
		}
	}
}

// loadImageData loads image data from a file path
func loadImageData(t *testing.T, imagePath string) []byte {
	data, err := os.ReadFile(imagePath)
	if err != nil {
		t.Fatalf("Failed to load image %s: %v", imagePath, err)
	}
	return data
}

func toLowerSlice(in []string) []string {
	out := make([]string, len(in))
	for i, s := range in {
		out[i] = strings.ToLower(s)
	}
	return out
}
