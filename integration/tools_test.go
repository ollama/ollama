//go:build integration

package integration

import (
	"context"
	"fmt"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
)

// testPropsMap creates a ToolPropertiesMap from a map (convenience function for tests)
func testPropsMap(m map[string]api.ToolProperty) *api.ToolPropertiesMap {
	props := api.NewToolPropertiesMap()
	for k, v := range m {
		props.Set(k, v)
	}
	return props
}

var toolsMinVRAM = map[string]uint64{
	"gemma4":        8,
	"qwen3-vl":      16,
	"gpt-oss:20b":   16,
	"gpt-oss:120b":  70,
	"qwen3":         6,
	"llama3.1":      8,
	"llama3.2":      4,
	"mistral":       6,
	"qwen2.5":       6,
	"qwen2":         6,
	"ministral-3":   20,
	"mistral-nemo":  9,
	"mistral-small": 16,
	"mixtral:8x22b": 80,
	"qwq":           20,
	"granite3.3":    7,
}

func TestAPIToolCalling(t *testing.T) {
	initialTimeout := 60 * time.Second
	streamTimeout := 60 * time.Second
	softTimeout, hardTimeout := getTimeouts(t)
	ctx, cancel := context.WithTimeout(context.Background(), hardTimeout)
	defer cancel()

	client, _, cleanup := InitServerConnection(ctx, t)
	defer cleanup()

	models := testModels(libraryToolsModels)

	for _, model := range models {
		t.Run(model, func(t *testing.T) {
			if time.Now().Sub(started) > softTimeout {
				t.Skip("skipping remaining tests to avoid excessive runtime")
				return
			}

			runAPIToolCallingModelWithClient(t, ctx, client, model, initialTimeout, streamTimeout, toolsMinVRAM)
		})
	}
}

func runAPIToolCallingModelWithClient(t *testing.T, ctx context.Context, client *api.Client, model string, initialTimeout, streamTimeout time.Duration, minVRAM map[string]uint64) {
	t.Helper()

	if v, ok := minVRAM[model]; ok {
		skipUnderMinVRAM(t, v)
	}
	if testModel != "" {
		requireCapability(ctx, t, client, model, "tools")
	}
	testBasicToolCall(t, ctx, client, model, initialTimeout, streamTimeout)
}

func basicToolCallTools() []api.Tool {
	return []api.Tool{
		{
			Type: "function",
			Function: api.ToolFunction{
				Name:        "get_weather",
				Description: "Get the current weather in a given location",
				Parameters: api.ToolFunctionParameters{
					Type:     "object",
					Required: []string{"location"},
					Properties: testPropsMap(map[string]api.ToolProperty{
						"location": {
							Type:        api.PropertyType{"string"},
							Description: "The city and state, e.g. San Francisco, CA",
						},
					}),
				},
			},
		},
	}
}

func testBasicToolCall(t *testing.T, ctx context.Context, client *api.Client, model string, initialTimeout, streamTimeout time.Duration) {
	t.Helper()
	testBasicToolCallWithNumPredict(t, ctx, client, model, initialTimeout, streamTimeout, 0)
}

func testBasicToolCallWithNumPredict(t *testing.T, ctx context.Context, client *api.Client, model string, initialTimeout, streamTimeout time.Duration, numPredict int) {
	t.Helper()

	options := map[string]any{
		"temperature": 0,
	}
	if numPredict > 0 {
		options["num_predict"] = numPredict
	}

	req := api.ChatRequest{
		Model: model,
		Messages: []api.Message{
			{
				Role:    "user",
				Content: "Call get_weather with location set to San Francisco.",
			},
		},
		Tools:     basicToolCallTools(),
		Options:   options,
		KeepAlive: &api.Duration{Duration: 10 * time.Second},
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

		if _, ok := lastToolCall.Function.Arguments.Get("location"); !ok {
			t.Errorf("expected tool arguments to include 'location', got: %s", lastToolCall.Function.Arguments.String())
		}
	case <-ctx.Done():
		t.Error("outer test context done while waiting for tool-calling chat")
	}
}
