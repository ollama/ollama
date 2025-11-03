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

// getTestConfig returns model and streaming mode based on environment variables or defaults
func getTestConfig() (model string, stream bool) {
	model = os.Getenv("QWEN3VL_MODEL")
	if model == "" {
		model = "qwen3-vl:235b-cloud" // default
	}

	streamStr := os.Getenv("QWEN3VL_STREAM")
	stream = streamStr != "false" // default to true

	return model, stream
}

func TestQwen3VL(t *testing.T) {
	model, stream := getTestConfig()

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
					Content: "What is in this image?",
				},
			},
			images: []string{"testdata/menu.png"},
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
			images: []string{"testdata/satmath1.png", "testdata/satmath2.png"},
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

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Load and attach images to last user message
			messages := tt.messages
			if len(tt.images) > 0 {
				var imgs []api.ImageData
				for _, path := range tt.images {
					imgs = append(imgs, loadImageData(t, path))
				}
				// Find last user message and attach images
				for i := len(messages) - 1; i >= 0; i-- {
					if messages[i].Role == "user" {
						messages[i].Images = imgs
						break
					}
				}
			}

			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
			defer cancel()
			client, _, cleanup := InitServerConnection(ctx, t)
			defer cleanup()

			// Pull/preload model if not using remote server
			if os.Getenv("OLLAMA_TEST_EXISTING") == "" {
				if err := PullIfMissing(ctx, client, model); err != nil {
					t.Fatal(err)
				}
				// Preload to reduce startup latency
				_ = client.Generate(ctx, &api.GenerateRequest{Model: model}, func(api.GenerateResponse) error { return nil })
			}

			// Build and execute chat request
			req := &api.ChatRequest{
				Model:    model,
				Messages: messages,
				Tools:    tt.tools,
				Stream:   &stream,
				Options:  map[string]any{"seed": 42, "temperature": 0.0},
			}

			var contentBuf, thinkingBuf strings.Builder
			var toolCalls []api.ToolCall

			err := client.Chat(ctx, req, func(r api.ChatResponse) error {
				contentBuf.WriteString(r.Message.Content)
				thinkingBuf.WriteString(r.Message.Thinking)
				toolCalls = append(toolCalls, r.Message.ToolCalls...)
				return nil
			})
			if err != nil {
				t.Fatalf("chat error: %v", err)
			}

			// Log truncated responses
			logTruncated := func(label, text string) {
				if text != "" {
					if len(text) > 800 {
						text = text[:800] + "... [truncated]"
					}
					t.Logf("%s: %s", label, text)
				}
			}
			logTruncated("Thinking", thinkingBuf.String())
			logTruncated("Content", contentBuf.String())

			if len(toolCalls) > 0 {
				t.Logf("Tool calls: %d", len(toolCalls))
				for i, call := range toolCalls {
					t.Logf("  [%d] %s(%+v)", i, call.Function.Name, call.Function.Arguments)
				}
			}

			// Validate tool calls if tools were provided
			if len(tt.tools) > 0 {
				if len(toolCalls) == 0 {
					t.Fatal("expected at least one tool call, got none")
				}
				if toolCalls[0].Function.Name == "" {
					t.Fatalf("tool call missing function name: %#v", toolCalls[0])
				}
			}
		})
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
