//go:build integration

package integration

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"os"
	"runtime"
	"strings"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
)

const (
	toolRouteName       = "get_weather"
	toolRouteLocation   = "Boston"
	toolRoutePreviousID = "call_previous"
)

func registerToolRouteCases(models []string) {
	registerModelIntegrationCases("tools-routes", models, runToolRoutesModel)
}

func runToolRoutesModel(t *testing.T, model string) {
	softTimeout, hardTimeout := getTimeouts(t)
	if time.Since(started) > softTimeout {
		t.Skip("skipping remaining tests to avoid excessive runtime")
	}
	ctx, cancel := context.WithTimeout(context.Background(), hardTimeout)
	defer cancel()

	client, endpoint, cleanup := InitServerConnection(ctx, t)
	t.Cleanup(func() {
		cleanup()
		assertNoUnexpectedRoleWarnings(t)
	})

	if v, ok := toolsMinVRAM[model]; ok {
		skipUnderMinVRAM(t, v)
	}
	requireCapability(ctx, t, client, model, "tools")

	t.Run("ollama", func(t *testing.T) {
		runOllamaToolRoute(t, ctx, client, model)
	})
	t.Run("openai_chat_completions", func(t *testing.T) {
		runOpenAIChatToolRoute(t, ctx, endpoint, model)
	})
	t.Run("openai_responses", func(t *testing.T) {
		runOpenAIResponsesToolRoute(t, ctx, endpoint, model)
	})
	t.Run("anthropic_messages", func(t *testing.T) {
		runAnthropicToolRoute(t, ctx, endpoint, model)
	})
}

func assertNoUnexpectedRoleWarnings(t *testing.T) {
	t.Helper()
	if os.Getenv("OLLAMA_TEST_EXISTING") != "" || runtime.GOOS == "windows" {
		return
	}

	serverMutex.Lock()
	defer serverMutex.Unlock()
	if strings.Contains(serverLog.String(), "unexpected message role") {
		t.Error("tool route request emitted an unexpected message role warning")
	}
}

func toolRouteTool() api.Tool {
	return newTool(toolRouteName, "Get the current weather for a location", []string{"location"}, map[string]api.ToolProperty{
		"location": {Type: api.PropertyType{"string"}, Description: "The city name"},
	})
}

func toolRouteArguments(location string) api.ToolCallFunctionArguments {
	arguments := api.NewToolCallFunctionArguments()
	arguments.Set("location", location)
	return arguments
}

func runOllamaToolRoute(t *testing.T, ctx context.Context, client *api.Client, model string) {
	t.Helper()
	stream := true
	req := api.ChatRequest{
		Model: model,
		Messages: []api.Message{
			{Role: "system", Content: toolRouteSystemPrompt()},
			{Role: "user", Content: "Call get_weather for Paris."},
			{
				Role: "assistant",
				ToolCalls: []api.ToolCall{{
					ID: toolRoutePreviousID,
					Function: api.ToolCallFunction{
						Name:      toolRouteName,
						Arguments: toolRouteArguments("Paris"),
					},
				}},
			},
			{Role: "tool", Content: `{"condition":"sunny"}`, ToolName: toolRouteName, ToolCallID: toolRoutePreviousID},
			{Role: "user", Content: toolRouteUserPrompt()},
		},
		Tools:  []api.Tool{toolRouteTool()},
		Stream: &stream,
		Think:  &api.ThinkValue{Value: false},
		Options: map[string]any{
			"temperature": 0,
			"num_ctx":     contextLength(16384),
			"num_predict": 256,
		},
		KeepAlive: &api.Duration{Duration: 10 * time.Second},
	}

	var calls []api.ToolCall
	var content strings.Builder
	if err := client.Chat(ctx, &req, func(response api.ChatResponse) error {
		calls = append(calls, response.Message.ToolCalls...)
		content.WriteString(response.Message.Content)
		return nil
	}); err != nil {
		t.Fatalf("Ollama chat failed: %v", err)
	}
	checkNoLeakedTags(t, content.String())
	if len(calls) == 0 {
		t.Fatalf("Ollama chat returned no tool call; content=%q", truncate(content.String(), 300))
	}
	assertToolRouteCall(t, calls[len(calls)-1].Function.Name, calls[len(calls)-1].Function.Arguments.String())
}

func runOpenAIChatToolRoute(t *testing.T, ctx context.Context, endpoint, model string) {
	t.Helper()
	payload := map[string]any{
		"model": model,
		"messages": []any{
			map[string]any{"role": "system", "content": "Tool protocol compatibility test."},
			map[string]any{"role": "developer", "content": toolRouteSystemPrompt()},
			map[string]any{"role": "user", "content": []any{map[string]any{"type": "text", "text": "Call get_weather for Paris."}}},
			map[string]any{
				"role":    "assistant",
				"content": nil,
				"tool_calls": []any{map[string]any{
					"id":   toolRoutePreviousID,
					"type": "function",
					"function": map[string]any{
						"name":      toolRouteName,
						"arguments": `{"location":"Paris"}`,
					},
				}},
			},
			map[string]any{
				"role":         "tool",
				"tool_call_id": toolRoutePreviousID,
				"content":      []any{map[string]any{"type": "text", "text": `{"condition":"sunny"}`}},
			},
			map[string]any{"role": "user", "content": toolRouteUserPrompt()},
		},
		"tools":            []any{openAIChatToolRouteDefinition()},
		"stream":           true,
		"temperature":      0,
		"max_tokens":       256,
		"reasoning_effort": "none",
	}

	events := postToolRouteSSE(t, ctx, endpoint, "/v1/chat/completions", payload, nil)
	var name, arguments string
	for _, event := range events {
		var chunk struct {
			Choices []struct {
				Delta struct {
					ToolCalls []struct {
						Function struct {
							Name      string `json:"name"`
							Arguments string `json:"arguments"`
						} `json:"function"`
					} `json:"tool_calls"`
				} `json:"delta"`
			} `json:"choices"`
		}
		if err := json.Unmarshal(event.Data, &chunk); err != nil {
			t.Fatalf("decode OpenAI Chat event: %v", err)
		}
		for _, choice := range chunk.Choices {
			for _, call := range choice.Delta.ToolCalls {
				name += call.Function.Name
				arguments += call.Function.Arguments
			}
		}
	}
	assertToolRouteCall(t, name, arguments)
}

func runOpenAIResponsesToolRoute(t *testing.T, ctx context.Context, endpoint, model string) {
	t.Helper()
	payload := map[string]any{
		"model":        model,
		"instructions": "Tool protocol compatibility test.",
		"input": []any{
			map[string]any{"type": "message", "role": "developer", "content": toolRouteSystemPrompt()},
			map[string]any{"type": "message", "role": "user", "content": []any{map[string]any{"type": "input_text", "text": "Call get_weather for Paris."}}},
			map[string]any{
				"type":      "function_call",
				"call_id":   toolRoutePreviousID,
				"name":      toolRouteName,
				"arguments": `{"location":"Paris"}`,
			},
			map[string]any{
				"type":    "function_call_output",
				"call_id": toolRoutePreviousID,
				"output":  []any{map[string]any{"type": "input_text", "text": `{"condition":"sunny"}`}},
			},
			map[string]any{"type": "message", "role": "user", "content": toolRouteUserPrompt()},
		},
		"tools": []any{map[string]any{
			"type":        "function",
			"name":        toolRouteName,
			"description": "Get the current weather for a location",
			"strict":      false,
			"parameters":  toolRouteParameters(),
		}},
		"stream":            true,
		"temperature":       0,
		"max_output_tokens": 256,
		"reasoning":         map[string]any{"effort": "none"},
	}

	events := postToolRouteSSE(t, ctx, endpoint, "/v1/responses", payload, nil)
	var name, arguments string
	for _, event := range events {
		if event.Event != "response.output_item.done" {
			continue
		}
		var data struct {
			Item struct {
				Type      string `json:"type"`
				Name      string `json:"name"`
				Arguments string `json:"arguments"`
			} `json:"item"`
		}
		if err := json.Unmarshal(event.Data, &data); err != nil {
			t.Fatalf("decode OpenAI Responses event: %v", err)
		}
		if data.Item.Type == "function_call" {
			name = data.Item.Name
			arguments = data.Item.Arguments
		}
	}
	assertToolRouteCall(t, name, arguments)
}

func runAnthropicToolRoute(t *testing.T, ctx context.Context, endpoint, model string) {
	t.Helper()
	payload := map[string]any{
		"model":      model,
		"max_tokens": 256,
		"system": []any{
			map[string]any{"type": "text", "text": "Tool protocol compatibility test."},
			map[string]any{"type": "text", "text": toolRouteSystemPrompt(), "cache_control": map[string]any{"type": "ephemeral"}},
		},
		"messages": []any{
			map[string]any{"role": "user", "content": []any{map[string]any{"type": "text", "text": "Call get_weather for Paris."}}},
			map[string]any{"role": "system", "content": []any{map[string]any{"type": "text", "text": "Runtime token budget update."}}},
			map[string]any{"role": "assistant", "content": []any{map[string]any{
				"type":  "tool_use",
				"id":    toolRoutePreviousID,
				"name":  toolRouteName,
				"input": map[string]any{"location": "Paris"},
			}}},
			map[string]any{"role": "user", "content": []any{
				map[string]any{
					"type":        "tool_result",
					"tool_use_id": toolRoutePreviousID,
					"content":     []any{map[string]any{"type": "text", "text": `{"condition":"sunny"}`}},
				},
				map[string]any{"type": "text", "text": toolRouteUserPrompt()},
			}},
		},
		"tools": []any{map[string]any{
			"name":         toolRouteName,
			"description":  "Get the current weather for a location",
			"input_schema": toolRouteParameters(),
		}},
		"stream":      true,
		"temperature": 0,
		"thinking":    map[string]any{"type": "disabled"},
	}

	events := postToolRouteSSE(t, ctx, endpoint, "/v1/messages", payload, map[string]string{
		"anthropic-version": "2023-06-01",
	})
	var name, arguments string
	var toolIndex int
	var foundTool bool
	for _, event := range events {
		switch event.Event {
		case "content_block_start":
			var data struct {
				Index        int `json:"index"`
				ContentBlock struct {
					Type string `json:"type"`
					Name string `json:"name"`
				} `json:"content_block"`
			}
			if err := json.Unmarshal(event.Data, &data); err != nil {
				t.Fatalf("decode Anthropic content start: %v", err)
			}
			if data.ContentBlock.Type == "tool_use" {
				foundTool = true
				toolIndex = data.Index
				name = data.ContentBlock.Name
			}
		case "content_block_delta":
			if !foundTool {
				continue
			}
			var data struct {
				Index int `json:"index"`
				Delta struct {
					Type        string `json:"type"`
					PartialJSON string `json:"partial_json"`
				} `json:"delta"`
			}
			if err := json.Unmarshal(event.Data, &data); err != nil {
				t.Fatalf("decode Anthropic content delta: %v", err)
			}
			if data.Index == toolIndex && data.Delta.Type == "input_json_delta" {
				arguments += data.Delta.PartialJSON
			}
		}
	}
	assertToolRouteCall(t, name, arguments)
}

func toolRouteSystemPrompt() string {
	return "When the user asks for weather, call get_weather. Return only the tool call and do not answer in text."
}

func toolRouteUserPrompt() string {
	return "Now call get_weather with location exactly Boston. Return only the tool call."
}

func openAIChatToolRouteDefinition() map[string]any {
	return map[string]any{
		"type": "function",
		"function": map[string]any{
			"name":        toolRouteName,
			"description": "Get the current weather for a location",
			"parameters":  toolRouteParameters(),
		},
	}
}

func toolRouteParameters() map[string]any {
	return map[string]any{
		"type":       "object",
		"required":   []string{"location"},
		"properties": map[string]any{"location": map[string]any{"type": "string"}},
	}
}

type toolRouteSSEEvent struct {
	Event string
	Data  json.RawMessage
}

func postToolRouteSSE(t *testing.T, ctx context.Context, endpoint, path string, payload any, headers map[string]string) []toolRouteSSEEvent {
	t.Helper()
	body, err := json.Marshal(payload)
	if err != nil {
		t.Fatal(err)
	}

	requestCtx, cancel := context.WithTimeout(ctx, 4*time.Minute)
	defer cancel()
	req, err := http.NewRequestWithContext(requestCtx, http.MethodPost, "http://"+endpoint+path, bytes.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Content-Type", "application/json")
	for name, value := range headers {
		req.Header.Set(name, value)
	}

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("POST %s failed: %v", path, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		data, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
		t.Fatalf("POST %s returned %s: %s", path, resp.Status, data)
	}

	var events []toolRouteSSEEvent
	var eventName string
	var eventData strings.Builder
	flush := func() {
		data := strings.TrimSpace(eventData.String())
		if data != "" && data != "[DONE]" {
			events = append(events, toolRouteSSEEvent{Event: eventName, Data: json.RawMessage(data)})
		}
		eventName = ""
		eventData.Reset()
	}

	scanner := bufio.NewScanner(resp.Body)
	scanner.Buffer(make([]byte, 64<<10), 2<<20)
	for scanner.Scan() {
		line := scanner.Text()
		if line == "" {
			flush()
			continue
		}
		if value, ok := strings.CutPrefix(line, "event:"); ok {
			eventName = strings.TrimSpace(value)
			continue
		}
		if value, ok := strings.CutPrefix(line, "data:"); ok {
			if eventData.Len() > 0 {
				eventData.WriteByte('\n')
			}
			eventData.WriteString(strings.TrimSpace(value))
		}
	}
	if err := scanner.Err(); err != nil {
		t.Fatalf("read %s stream: %v", path, err)
	}
	flush()
	if len(events) == 0 {
		t.Fatalf("POST %s returned no SSE events", path)
	}
	return events
}

func assertToolRouteCall(t *testing.T, name, arguments string) {
	t.Helper()
	if name != toolRouteName {
		t.Fatalf("tool name = %q, want %q; arguments=%q", name, toolRouteName, arguments)
	}
	var values map[string]any
	if err := json.Unmarshal([]byte(arguments), &values); err != nil {
		t.Fatalf("tool arguments are not JSON: %q: %v", arguments, err)
	}
	location, _ := values["location"].(string)
	if !strings.EqualFold(location, toolRouteLocation) {
		t.Fatalf("tool location = %q, want %q; arguments=%s", location, toolRouteLocation, arguments)
	}
}
