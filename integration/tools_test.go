//go:build integration

package integration

import (
	"context"
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
)

var libraryToolsModels = []string{
	"qwen3-vl",
	"gpt-oss:20b",
	"gpt-oss:120b",
	"qwen3",
	"llama3.1",
	"llama3.2",
	"mistral",
	"qwen2.5",
	"qwen2",
	"mistral-nemo",
	"mistral-small",
	"mixtral:8x22b",
	"qwq",
	"granite3.3",
}

func TestAPIToolCalling(t *testing.T) {
	initialTimeout := 60 * time.Second
	streamTimeout := 60 * time.Second
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
	defer cancel()

	client, _, cleanup := InitServerConnection(ctx, t)
	defer cleanup()

	for _, model := range libraryToolsModels {
		t.Run(model, func(t *testing.T) {
			if v, ok := minVRAM[model]; ok {
				skipUnderMinVRAM(t, v)
			}

			if err := PullIfMissing(ctx, client, model); err != nil {
				t.Fatalf("pull failed %s", err)
			}

			tools := []api.Tool{
				{
					Type: "function",
					Function: api.ToolFunction{
						Name:        "get_weather",
						Description: "Get the current weather in a given location",
						Parameters: api.ToolFunctionParameters{
							Type:     "object",
							Required: []string{"location"},
							Properties: map[string]api.ToolProperty{
								"location": {
									Type:        api.PropertyType{"string"},
									Description: "The city and state, e.g. San Francisco, CA",
								},
							},
						},
					},
				},
			}

			req := api.ChatRequest{
				Model: model,
				Messages: []api.Message{
					{
						Role:    "user",
						Content: "Call get_weather with location set to San Francisco.",
					},
				},
				Tools: tools,
				Options: map[string]any{
					"temperature": 0,
				},
			}

			stallTimer := time.NewTimer(initialTimeout)
			var gotToolCall bool
			var lastToolCall api.ToolCall

			fn := func(response api.ChatResponse) error {
				if len(response.Message.ToolCalls) > 0 {
					gotToolCall = true
					lastToolCall = response.Message.ToolCalls[len(response.Message.ToolCalls)-1]
				}
				if !stallTimer.Reset(streamTimeout) {
					return fmt.Errorf("stall was detected while streaming response, aborting")
				}
				return nil
			}

			stream := true
			req.Stream = &stream
			done := make(chan int)
			var genErr error
			go func() {
				genErr = client.Chat(ctx, &req, fn)
				done <- 0
			}()

			select {
			case <-stallTimer.C:
				t.Errorf("tool-calling chat never started. Timed out after: %s", initialTimeout.String())
			case <-done:
				if genErr != nil {
					t.Fatalf("chat failed: %v", genErr)
				}

				if !gotToolCall {
					t.Fatalf("expected at least one tool call, got none")
				}

				if lastToolCall.Function.Name != "get_weather" {
					t.Errorf("unexpected tool called: got %q want %q", lastToolCall.Function.Name, "get_weather")
				}

				if _, ok := lastToolCall.Function.Arguments["location"]; !ok {
					t.Errorf("expected tool arguments to include 'location', got: %s", lastToolCall.Function.Arguments.String())
				}
			case <-ctx.Done():
				t.Error("outer test context done while waiting for tool-calling chat")
			}
		})
	}
}

func TestAPIToolCallingMultiTurn(t *testing.T) {
	initialTimeout := 60 * time.Second
	streamTimeout := 60 * time.Second
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
	defer cancel()

	client, _, cleanup := InitServerConnection(ctx, t)
	defer cleanup()

	for _, model := range libraryToolsModels {
		t.Run(model, func(t *testing.T) {
			if v, ok := minVRAM[model]; ok {
				skipUnderMinVRAM(t, v)
			}

			if err := PullIfMissing(ctx, client, model); err != nil {
				t.Fatalf("pull failed %s", err)
			}

			tools := []api.Tool{
				{
					Type: "function",
					Function: api.ToolFunction{
						Name:        "get_weather",
						Description: "Get the current weather in a given location",
						Parameters: api.ToolFunctionParameters{
							Type:     "object",
							Required: []string{"location"},
							Properties: map[string]api.ToolProperty{
								"location": {
									Type:        api.PropertyType{"string"},
									Description: "The city and state, e.g. San Francisco, CA",
								},
							},
						},
					},
				},
			}

			userMessage := api.Message{
				Role:    "user",
				Content: "What's the weather like in San Francisco?",
			}

			req := api.ChatRequest{
				Model:    model,
				Messages: []api.Message{userMessage},
				Tools:    tools,
				Options: map[string]any{
					"temperature": 0,
				},
			}

			stallTimer := time.NewTimer(initialTimeout)
			var assistantMessage api.Message
			var gotToolCall bool
			var toolCallID string

			fn := func(response api.ChatResponse) error {
				if response.Message.Content != "" {
					assistantMessage.Content += response.Message.Content
					assistantMessage.Role = "assistant"
				}
				if len(response.Message.ToolCalls) > 0 {
					gotToolCall = true
					assistantMessage.ToolCalls = response.Message.ToolCalls
					assistantMessage.Role = "assistant"
					toolCallID = response.Message.ToolCalls[0].ID
				}
				if !stallTimer.Reset(streamTimeout) {
					return fmt.Errorf("stall was detected while streaming response, aborting")
				}
				return nil
			}

			stream := true
			req.Stream = &stream
			done := make(chan int)
			var genErr error
			go func() {
				genErr = client.Chat(ctx, &req, fn)
				done <- 0
			}()

			select {
			case <-stallTimer.C:
				t.Fatalf("first turn chat never started. Timed out after: %s", initialTimeout.String())
			case <-done:
				if genErr != nil {
					t.Fatalf("first turn chat failed: %v", genErr)
				}

				if !gotToolCall {
					t.Fatalf("expected at least one tool call in first turn, got none")
				}

				if len(assistantMessage.ToolCalls) == 0 {
					t.Fatalf("expected tool calls in assistant message, got none")
				}

				firstToolCall := assistantMessage.ToolCalls[0]
				if firstToolCall.Function.Name != "get_weather" {
					t.Errorf("unexpected tool called: got %q want %q", firstToolCall.Function.Name, "get_weather")
				}

				location, ok := firstToolCall.Function.Arguments["location"]
				if !ok {
					t.Fatalf("expected tool arguments to include 'location', got: %s", firstToolCall.Function.Arguments.String())
				}

				toolResult := `{"temperature": 72, "condition": "sunny", "humidity": 65}`
				toolMessage := api.Message{
					Role:       "tool",
					Content:    toolResult,
					ToolName:   "get_weather",
					ToolCallID: toolCallID,
				}

				messages := []api.Message{
					userMessage,
					assistantMessage,
					toolMessage,
				}

				req2 := api.ChatRequest{
					Model:    model,
					Messages: messages,
					Tools:    tools,
					Options: map[string]any{
						"temperature": 0,
					},
				}

				stallTimer2 := time.NewTimer(initialTimeout)
				var finalResponse string
				var gotSecondToolCall bool

				fn2 := func(response api.ChatResponse) error {
					if len(response.Message.ToolCalls) > 0 {
						gotSecondToolCall = true
					}
					if response.Message.Content != "" {
						finalResponse += response.Message.Content
					}
					if !stallTimer2.Reset(streamTimeout) {
						return fmt.Errorf("stall was detected while streaming response, aborting")
					}
					return nil
				}

				req2.Stream = &stream
				done2 := make(chan int)
				var genErr2 error
				go func() {
					genErr2 = client.Chat(ctx, &req2, fn2)
					done2 <- 0
				}()

				select {
				case <-stallTimer2.C:
					t.Fatalf("second turn chat never started. Timed out after: %s", initialTimeout.String())
				case <-done2:
					if genErr2 != nil {
						t.Fatalf("second turn chat failed: %v", genErr2)
					}

					if gotSecondToolCall {
						t.Errorf("expected no tool calls in second turn, but got tool calls. Model should respond with natural language after receiving tool result.")
					}

					if finalResponse == "" {
						t.Fatalf("expected natural language response in second turn, got empty response")
					}

					responseLower := strings.ToLower(finalResponse)
					expectedKeywords := []string{"72", "sunny", "temperature", "weather", "san francisco", "fahrenheit"}
					foundKeyword := false
					for _, keyword := range expectedKeywords {
						if strings.Contains(responseLower, strings.ToLower(keyword)) {
							foundKeyword = true
							break
						}
					}
					if !foundKeyword {
						t.Logf("response: %s", finalResponse)
						t.Logf("location from tool call: %v", location)
					}
				case <-ctx.Done():
					t.Error("outer test context done while waiting for second turn")
				}
			case <-ctx.Done():
				t.Error("outer test context done while waiting for first turn")
			}
		})
	}
}
