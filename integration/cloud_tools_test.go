//go:build integration

package integration

import (
	"context"
	"fmt"
	"net/http"
	"net/url"
	"os"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
)

type authTransport struct {
	apiKey string
	base   http.RoundTripper
}

func (t *authTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	req.Header.Set("Authorization", "Bearer "+t.apiKey)
	return t.base.RoundTrip(req)
}

var cloudToolsModels = []string{
	"glm-4.6",
	"kimi-k2:1t",
	"deepseek-v3.1:671b",
	"gpt-oss:120b",
	"minimax-m2",
}

func setupCloudClient(t *testing.T) *api.Client {
	origHost := os.Getenv("OLLAMA_HOST")

	// Restore original env after test
	t.Cleanup(func() {
		if origHost == "" {
			os.Unsetenv("OLLAMA_HOST")
		} else {
			os.Setenv("OLLAMA_HOST", origHost)
		}
	})

	// Configure for cloud proxy server
	os.Setenv("OLLAMA_HOST", "localhost:8080")

	apiKey := os.Getenv("OLLAMA_API_KEY_LOCAL")
	if apiKey == "" {
		t.Fatal("OLLAMA_API_KEY_LOCAL environment variable required for cloud tests")
	}

	return api.NewClient(
		&url.URL{Scheme: "http", Host: "localhost:8080"},
		&http.Client{
			Transport: &authTransport{
				apiKey: apiKey,
				base:   http.DefaultTransport,
			},
		},
	)
}

func TestCloudToolCallingSingle(t *testing.T) {
	client := setupCloudClient(t)
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Minute)
	defer cancel()

	weatherTool := api.Tool{
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
	}

	for _, model := range cloudToolsModels {
		t.Run(model, func(t *testing.T) {
			req := api.ChatRequest{
				Model: model,
				Messages: []api.Message{
					{Role: "user", Content: "What is the weather in San Francisco?"},
				},
				Tools:   []api.Tool{weatherTool},
				Options: map[string]any{"temperature": 0, "seed": 42},
			}

			var (
				gotToolCall    bool
				lastToolCall   api.ToolCall
				initialTimeout = 120 * time.Second
				streamTimeout  = 120 * time.Second
			)

			stallTimer := time.NewTimer(initialTimeout)
			defer stallTimer.Stop()

			fn := func(resp api.ChatResponse) error {
				if len(resp.Message.ToolCalls) > 0 {
					gotToolCall = true
					lastToolCall = resp.Message.ToolCalls[len(resp.Message.ToolCalls)-1]
				}
				if !stallTimer.Reset(streamTimeout) {
					return fmt.Errorf("stall detected")
				}
				return nil
			}

			stream := true
			req.Stream = &stream
			done := make(chan error, 1)

			go func() {
				done <- client.Chat(ctx, &req, fn)
			}()

			select {
			case <-stallTimer.C:
				t.Fatalf("timeout after %s", initialTimeout)
			case err := <-done:
				if err != nil {
					t.Fatalf("chat failed: %v", err)
				}

				// Validate tool call
				if !gotToolCall {
					t.Fatal("expected tool call, got none")
				}
				t.Logf("Tool: %s(%+v)", lastToolCall.Function.Name, lastToolCall.Function.Arguments)

				if lastToolCall.Function.Name != "get_weather" {
					t.Errorf("expected get_weather, got %q", lastToolCall.Function.Name)
				}
				location, ok := lastToolCall.Function.Arguments["location"]
				if !ok {
					t.Errorf("missing 'location' in arguments: %v", lastToolCall.Function.Arguments)
				} else if location != "San Francisco, CA" {
					t.Logf("note: got location=%q, expected 'San Francisco, CA'", location)
				}
			case <-ctx.Done():
				t.Fatal("context timeout")
			}
		})
	}
}

func TestCloudToolCallingMultiTurn(t *testing.T) {
	client := setupCloudClient(t)
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Minute)
	defer cancel()

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
		{
			Type: "function",
			Function: api.ToolFunction{
				Name:        "get_time",
				Description: "Get the current time in a given timezone",
				Parameters: api.ToolFunctionParameters{
					Type:     "object",
					Required: []string{"timezone"},
					Properties: map[string]api.ToolProperty{
						"timezone": {
							Type:        api.PropertyType{"string"},
							Description: "The timezone, e.g. America/Los_Angeles",
						},
					},
				},
			},
		},
	}

	for _, model := range cloudToolsModels {
		t.Run(model, func(t *testing.T) {
			messages := []api.Message{
				{Role: "user", Content: "What's the weather in San Francisco?"},
				{
					Role:    "assistant",
					Content: "",
					ToolCalls: []api.ToolCall{
						{
							Function: api.ToolCallFunction{
								Name:      "get_weather",
								Arguments: api.ToolCallFunctionArguments{"location": "San Francisco, CA"},
							},
						},
					},
				},
				{Role: "tool", Content: "The weather in San Francisco is sunny, 72°F", ToolName: "get_weather"},
				{Role: "assistant", Content: "The weather in San Francisco is currently sunny with a temperature of 72°F."},
				{Role: "user", Content: "Great! Now what time is it there?"},
			}

			req := api.ChatRequest{
				Model:    model,
				Messages: messages,
				Tools:    tools,
				Options:  map[string]any{"temperature": 0, "seed": 42},
			}

			var (
				gotToolCall    bool
				lastToolCall   api.ToolCall
				allToolCalls   []api.ToolCall
				initialTimeout = 300 * time.Second
				streamTimeout  = 300 * time.Second
			)

			stallTimer := time.NewTimer(initialTimeout)
			defer stallTimer.Stop()

			fn := func(resp api.ChatResponse) error {
				if len(resp.Message.ToolCalls) > 0 {
					gotToolCall = true
					allToolCalls = append(allToolCalls, resp.Message.ToolCalls...)
					lastToolCall = resp.Message.ToolCalls[len(resp.Message.ToolCalls)-1]
				}
				if !stallTimer.Reset(streamTimeout) {
					return fmt.Errorf("stall detected")
				}
				return nil
			}

			stream := true
			req.Stream = &stream
			done := make(chan error, 1)

			go func() {
				done <- client.Chat(ctx, &req, fn)
			}()

			select {
			case <-stallTimer.C:
				t.Fatalf("timeout after %s", initialTimeout)
			case err := <-done:
				if err != nil {
					t.Fatalf("chat failed: %v", err)
				}

				if !gotToolCall {
					t.Fatal("expected tool call, got none")
				}

				// Log all tool calls
				t.Logf("Multi-turn: %d tool calls", len(allToolCalls))
				for i, tc := range allToolCalls {
					t.Logf("  [%d] %s(%+v)", i, tc.Function.Name, tc.Function.Arguments)
				}

				// Validate last tool call should be get_time
				if lastToolCall.Function.Name != "get_time" {
					t.Errorf("expected get_time, got %q", lastToolCall.Function.Name)
				}
				timezone, ok := lastToolCall.Function.Arguments["timezone"]
				if !ok {
					t.Errorf("missing 'timezone' in arguments: %v", lastToolCall.Function.Arguments)
				} else if timezone != "America/Los_Angeles" {
					t.Logf("note: got timezone=%q, expected 'America/Los_Angeles'", timezone)
				}
			case <-ctx.Done():
				t.Fatal("context timeout")
			}
		})
	}
}
