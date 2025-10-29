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

// TestQwen3VLStreaming tests Qwen3-VL with streaming enabled
func TestQwen3VLStreaming(t *testing.T) {
	runQwen3VLTests(t, true)
}

// TestQwen3VLNonStreaming tests Qwen3-VL with streaming disabled
func TestQwen3VLNonStreaming(t *testing.T) {
	runQwen3VLTests(t, false)
}

func runQwen3VLTests(t *testing.T, stream bool) {
	models := []string{"qwen3vl-odc-dev"} // , "qwen3vl-thinking-odc-dev", "qwen3-vl:8b"}

	tests := []struct {
		name     string
		messages []api.Message
		tools    []api.Tool
		images   []string
	}{
		{
			name: "Text-Only Scenario",
			messages: []api.Message{
				{Role: "system", Content: "You are a helpful assistant."},
				{Role: "user", Content: "Write a short haiku about autumn."},
			},
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
					Content: "What is the answer to this question?",
				},
			},
			images: []string{"testdata/question.png"},
		},
		{
			name: "Multiple Images Scenario",
			messages: []api.Message{
				{
					Role:    "system",
					Content: "You are a helpful assistant that can see images.",
				},
				{
					Role:    "user",
					Content: "Use both images to answer the question.",
				},
			},
			images: []string{"testdata/question.png", "testdata/menu.png"},
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
		},
		{
			name: "Multi-Turn Tools With Image",
			messages: []api.Message{
				{Role: "system", Content: "Use tools when actions are required."},
				{Role: "user", Content: "What's the current temperature in San Francisco?"},
				{Role: "assistant", Content: "", ToolCalls: []api.ToolCall{
					{Function: api.ToolCallFunction{
						Name: "get_weather",
						Arguments: api.ToolCallFunctionArguments{
							"city": "San Francisco",
						},
					}},
				}},
				{Role: "tool", ToolName: "get_weather", Content: "Sunny"},
				{Role: "user", Content: "Given that weather, what are the top 10 activities to do in San Francisco? Consider this photo as context."},
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
				{
					Type: "function",
					Function: api.ToolFunction{
						Name:        "get_top_10_activities",
						Description: "Get the top 10 activities for a city given the weather.",
						Parameters: api.ToolFunctionParameters{
							Type: "object",
							Properties: map[string]api.ToolProperty{
								"weather": {
									Type:        api.PropertyType{"string"},
									Description: "The weather in the city",
								},
								"city": {
									Type:        api.PropertyType{"string"},
									Description: "The city to get the activities for",
								},
								"image": {
									Type:        api.PropertyType{"base64"},
									Description: "The image of the city",
								},
							},
							Required: []string{"weather", "city", "image"},
						},
					},
				},
			},
			images: []string{"testdata/sf-city.jpeg"},
		},
	}

	for _, model := range models {
		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				// Load and attach images if specified
				if len(tt.images) > 0 {
					var imgs []api.ImageData
					for _, path := range tt.images {
						imgs = append(imgs, loadImageData(t, path))
					}
					if len(tt.messages) > 0 {
						lastMessage := &tt.messages[len(tt.messages)-1]
						if lastMessage.Role == "user" {
							lastMessage.Images = imgs
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

				isRemote := os.Getenv("OLLAMA_TEST_EXISTING") != ""

				// Use integration helpers
				ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
				defer cancel()
				client, _, cleanup := InitServerConnection(ctx, t)
				defer cleanup()

				// Skip pulling/preloading when using an existing (cloud) server
				if !isRemote {
					if err := PullIfMissing(ctx, client, req.Model); err != nil {
						t.Fatal(err)
					}
					// Preload model once to reduce startup latency
					_ = client.Generate(ctx, &api.GenerateRequest{Model: req.Model}, func(r api.GenerateResponse) error { return nil })
				}

				var contentBuf, thinkingBuf strings.Builder
				var gotCalls []api.ToolCall

				err := client.Chat(ctx, &req, func(r api.ChatResponse) error {
					contentBuf.WriteString(r.Message.Content)
					thinkingBuf.WriteString(r.Message.Thinking)
					if len(r.Message.ToolCalls) > 0 {
						gotCalls = append(gotCalls, r.Message.ToolCalls...)
					}
					return nil
				})
				if err != nil {
					t.Fatalf("chat error: %v", err)
				}

				// Log responses (truncated)
				content := contentBuf.String()
				thinking := thinkingBuf.String()
				const maxLog = 800
				if len(thinking) > 0 {
					if len(thinking) > maxLog {
						thinking = thinking[:maxLog] + "... [truncated]"
					}
					t.Logf("Thinking: %s", thinking)
				}
				if len(content) > 0 {
					if len(content) > maxLog {
						content = content[:maxLog] + "... [truncated]"
					}
					t.Logf("Content: %s", content)
				}
				if len(gotCalls) > 0 {
					t.Logf("Tool calls: %d", len(gotCalls))
					for i, call := range gotCalls {
						t.Logf("  [%d] %s(%+v)", i, call.Function.Name, call.Function.Arguments)
					}
				}

				// If this is a tools scenario, validate tool_calls
				if len(tt.tools) > 0 {
					if len(gotCalls) == 0 {
						t.Fatalf("expected at least one tool call, got none")
					}
					if gotCalls[0].Function.Name == "" {
						t.Fatalf("tool call missing function name: %#v", gotCalls[0])
					}
				}
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
