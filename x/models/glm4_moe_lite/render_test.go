//go:build mlx

package glm4_moe_lite

import (
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
)

func TestRendererSimple(t *testing.T) {
	r := &Renderer{}

	messages := []api.Message{
		{Role: "user", Content: "Hello"},
	}

	// Thinking enabled (default)
	result, err := r.Render(messages, nil, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	expected := "[gMASK]<sop><|user|>Hello<|assistant|><think>"
	if result != expected {
		t.Errorf("result = %q, want %q", result, expected)
	}
}

func TestRendererThinkingDisabled(t *testing.T) {
	r := &Renderer{}

	messages := []api.Message{
		{Role: "user", Content: "Hello"},
	}

	tv := &api.ThinkValue{Value: false}

	result, err := r.Render(messages, nil, tv)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	expected := "[gMASK]<sop><|user|>Hello<|assistant|></think>"
	if result != expected {
		t.Errorf("result = %q, want %q", result, expected)
	}
}

func TestRendererMultiTurn(t *testing.T) {
	r := &Renderer{}

	messages := []api.Message{
		{Role: "user", Content: "What is 2+2?"},
		{Role: "assistant", Content: "4", Thinking: "Let me calculate: 2+2=4"},
		{Role: "user", Content: "And 3+3?"},
	}

	result, err := r.Render(messages, nil, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Check key parts
	if !strings.Contains(result, "[gMASK]<sop>") {
		t.Error("missing [gMASK]<sop> prefix")
	}
	if !strings.Contains(result, "<|user|>What is 2+2?") {
		t.Error("missing first user message")
	}
	if !strings.Contains(result, "<|assistant|><think>Let me calculate: 2+2=4</think>4") {
		t.Error("missing assistant message with thinking")
	}
	if !strings.Contains(result, "<|user|>And 3+3?") {
		t.Error("missing second user message")
	}
	if !strings.HasSuffix(result, "<|assistant|><think>") {
		t.Errorf("should end with <|assistant|><think>, got suffix: %q", result[len(result)-30:])
	}
}

func TestRendererWithSystem(t *testing.T) {
	r := &Renderer{}

	messages := []api.Message{
		{Role: "system", Content: "You are a helpful assistant."},
		{Role: "user", Content: "Hello"},
	}

	result, err := r.Render(messages, nil, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if !strings.Contains(result, "<|system|>You are a helpful assistant.") {
		t.Error("missing system message")
	}
}

func TestRendererWithTools(t *testing.T) {
	r := &Renderer{}

	messages := []api.Message{
		{Role: "user", Content: "What's the weather?"},
	}

	props := api.NewToolPropertiesMap()
	props.Set("location", api.ToolProperty{Type: api.PropertyType{"string"}, Description: "The city"})
	tools := []api.Tool{
		{
			Function: api.ToolFunction{
				Name:        "get_weather",
				Description: "Get the weather for a location",
				Parameters: api.ToolFunctionParameters{
					Type:       "object",
					Properties: props,
					Required:   []string{"location"},
				},
			},
		},
	}

	result, err := r.Render(messages, tools, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Check for tool system prompt
	if !strings.Contains(result, "<|system|>") {
		t.Error("missing system tag for tools")
	}
	if !strings.Contains(result, "# Tools") {
		t.Error("missing tools header")
	}
	if !strings.Contains(result, "<tools>") {
		t.Error("missing tools tag")
	}
	if !strings.Contains(result, "get_weather") {
		t.Error("missing tool name")
	}
	if !strings.Contains(result, "</tools>") {
		t.Error("missing closing tools tag")
	}
}

func TestRendererWithToolCalls(t *testing.T) {
	r := &Renderer{}

	args := api.NewToolCallFunctionArguments()
	args.Set("location", "San Francisco")

	messages := []api.Message{
		{Role: "user", Content: "What's the weather in SF?"},
		{
			Role: "assistant",
			ToolCalls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name:      "get_weather",
						Arguments: args,
					},
				},
			},
		},
		{Role: "tool", Content: "Sunny, 72F"},
	}

	result, err := r.Render(messages, nil, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if !strings.Contains(result, "<tool_call>get_weather") {
		t.Error("missing tool call")
	}
	if !strings.Contains(result, "<arg_key>location</arg_key>") {
		t.Error("missing arg_key")
	}
	if !strings.Contains(result, "<arg_value>San Francisco</arg_value>") {
		t.Error("missing arg_value")
	}
	if !strings.Contains(result, "</tool_call>") {
		t.Error("missing tool call closing tag")
	}
	if !strings.Contains(result, "<|observation|>") {
		t.Error("missing observation tag")
	}
	if !strings.Contains(result, "<tool_response>Sunny, 72F</tool_response>") {
		t.Error("missing tool response")
	}
}

func TestFormatToolJSON(t *testing.T) {
	input := []byte(`{"name":"test","value":123}`)
	result := formatToolJSON(input)

	// Should add spaces after : and ,
	if !strings.Contains(result, ": ") {
		t.Error("should add space after colon")
	}
	if !strings.Contains(result, ", ") {
		t.Error("should add space after comma")
	}
}
