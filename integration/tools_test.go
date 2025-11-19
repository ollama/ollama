//go:build integration

package integration

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/openai"
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

func float64Ptr(v float64) *float64 {
	return &v
}

func sendOpenAIChatRequest(ctx context.Context, endpoint string, req openai.ChatCompletionRequest) (*openai.ChatCompletion, error) {
	jsonData, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", endpoint+"/v1/chat/completions", bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")

	client := &http.Client{
		Timeout: 10 * time.Minute,
	}

	resp, err := client.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("failed to send request: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("API error: status=%d, body=%s", resp.StatusCode, string(body))
	}

	var chatResp openai.ChatCompletion
	if err := json.Unmarshal(body, &chatResp); err != nil {
		return nil, fmt.Errorf("failed to unmarshal response: %w, body: %s", err, string(body))
	}

	return &chatResp, nil
}

func sendOpenAIChatStreamRequest(ctx context.Context, endpoint string, req openai.ChatCompletionRequest, fn func(openai.ChatCompletionChunk) error) error {
	jsonData, err := json.Marshal(req)
	if err != nil {
		return fmt.Errorf("failed to marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", endpoint+"/v1/chat/completions", bytes.NewBuffer(jsonData))
	if err != nil {
		return fmt.Errorf("failed to create request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Accept", "text/event-stream")

	client := &http.Client{
		Timeout: 0, // No timeout for streaming
	}

	resp, err := client.Do(httpReq)
	if err != nil {
		return fmt.Errorf("failed to send request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("API error: status=%d, body=%s", resp.StatusCode, string(body))
	}

	decoder := resp.Body
	reader := bytes.NewBuffer([]byte{})
	buf := make([]byte, 4096)

	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
			n, err := decoder.Read(buf)
			if n > 0 {
				reader.Write(buf[:n])

				// Process complete lines
				for {
					line, err := reader.ReadString('\n')
					if err != nil {
						// Not a complete line yet
						reader.WriteString(line)
						break
					}

					line = strings.TrimSpace(line)
					if strings.HasPrefix(line, "data: ") {
						data := strings.TrimPrefix(line, "data: ")

						if data == "[DONE]" {
							return nil
						}

						var streamResp openai.ChatCompletionChunk
						if err := json.Unmarshal([]byte(data), &streamResp); err != nil {
							return fmt.Errorf("failed to unmarshal stream response: %w", err)
						}

						if err := fn(streamResp); err != nil {
							return err
						}
					}
				}
			}

			if err != nil {
				if err != io.EOF {
					return fmt.Errorf("error reading stream: %w", err)
				}
				break
			}
		}
	}

	return nil
}

// TestToolCallingAllAPIs tests both Ollama and OpenAI APIs with shared model loading
func TestToolCallingAllAPIs(t *testing.T) {
	initialTimeout := 60 * time.Second
	streamTimeout := 60 * time.Second
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
	defer cancel()

	client, endpoint, cleanup := InitServerConnection(ctx, t)
	defer cleanup()

	for _, model := range libraryToolsModels {
		t.Run(model, func(t *testing.T) {
			// Skip if insufficient VRAM
			if v, ok := minVRAM[model]; ok {
				skipUnderMinVRAM(t, v)
			}

			// Pull model if missing - only do this once per model
			if err := PullIfMissing(ctx, client, model); err != nil {
				t.Fatalf("pull failed %s", err)
			}

			t.Run("OllamaAPI", func(t *testing.T) {
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

			t.Run("OpenAIAPI", func(t *testing.T) {
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

				req := openai.ChatCompletionRequest{
					Model: model,
					Messages: []openai.Message{
						{
							Role:    "user",
							Content: "Call get_weather with location set to San Francisco.",
						},
					},
					Tools:       tools,
					Stream:      true,
					Temperature: float64Ptr(0),
				}

				stallTimer := time.NewTimer(initialTimeout)
				var gotToolCall bool
				var lastToolCall openai.ToolCall

				fn := func(response openai.ChatCompletionChunk) error {
					if len(response.Choices) > 0 && len(response.Choices[0].Delta.ToolCalls) > 0 {
						gotToolCall = true
						toolCalls := response.Choices[0].Delta.ToolCalls
						lastToolCall = toolCalls[len(toolCalls)-1]
					}
					if !stallTimer.Reset(streamTimeout) {
						return fmt.Errorf("stall was detected while streaming response, aborting")
					}
					return nil
				}

				done := make(chan int)
				var genErr error
				go func() {
					genErr = sendOpenAIChatStreamRequest(ctx, "http://"+endpoint, req, fn)
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

					if !strings.Contains(lastToolCall.Function.Arguments, "location") {
						t.Errorf("expected tool arguments to include 'location', got: %s", lastToolCall.Function.Arguments)
					}

					if !strings.Contains(lastToolCall.Function.Arguments, "San Francisco") {
						t.Errorf("expected tool arguments to include 'San Francisco', got: %s", lastToolCall.Function.Arguments)
					}
				case <-ctx.Done():
					t.Error("outer test context done while waiting for tool-calling chat")
				}
			})
		})
	}
}
