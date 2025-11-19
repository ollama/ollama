package parsers

import (
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/ollama/ollama/api"
)

func TestCogitoParser(t *testing.T) {
	tests := []struct {
		name              string
		input             string
		expectedContent   string
		expectedThinking  string
		expectedToolCalls []api.ToolCall
		tools             []api.Tool
		lastMessage       *api.Message
	}{
		{
			name:             "simple_content",
			input:            "This is a simple response.",
			expectedContent:  "This is a simple response.",
			expectedThinking: "",
		},
		{
			name:             "thinking_only",
			input:            "<think>This is thinking content.</think>This is response content.",
			expectedContent:  "This is response content.",
			expectedThinking: "This is thinking content.",
		},
		{
			name: "tool_call_simple",
			input: `<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather
` + "```json\n" + `{"location":"Paris"}
` + "```" + `<｜tool▁call▁end｜><｜tool▁calls▁end｜>`,
			expectedToolCalls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name: "get_weather",
						Arguments: api.ToolCallFunctionArguments{
							"location": "Paris",
						},
					},
				},
			},
			tools: []api.Tool{
				{
					Type: "function",
					Function: api.ToolFunction{
						Name: "get_weather",
						Parameters: api.ToolFunctionParameters{
							Properties: map[string]api.ToolProperty{
								"location": {Type: api.PropertyType{"string"}},
							},
						},
					},
				},
			},
		},
		{
			name: "thinking_with_tool_call",
			input: `<think>I need to check the weather.</think><｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather
` + "```json\n" + `{"location":"Paris"}
` + "```" + `<｜tool▁call▁end｜><｜tool▁calls▁end｜>`,
			expectedContent:  "",
			expectedThinking: "I need to check the weather.",
			expectedToolCalls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name: "get_weather",
						Arguments: api.ToolCallFunctionArguments{
							"location": "Paris",
						},
					},
				},
			},
			tools: []api.Tool{
				{
					Type: "function",
					Function: api.ToolFunction{
						Name: "get_weather",
						Parameters: api.ToolFunctionParameters{
							Properties: map[string]api.ToolProperty{
								"location": {Type: api.PropertyType{"string"}},
							},
						},
					},
				},
			},
		},
		{
			name: "multiple_tool_calls",
			input: `<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather
` + "```json\n" + `{"location":"Paris"}
` + "```" + `<｜tool▁call▁end｜>
<｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather
` + "```json\n" + `{"location":"London"}
` + "```" + `<｜tool▁call▁end｜><｜tool▁calls▁end｜>`,
			expectedToolCalls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name: "get_weather",
						Arguments: api.ToolCallFunctionArguments{
							"location": "Paris",
						},
					},
				},
				{
					Function: api.ToolCallFunction{
						Name: "get_weather",
						Arguments: api.ToolCallFunctionArguments{
							"location": "London",
						},
					},
				},
			},
			tools: []api.Tool{
				{
					Type: "function",
					Function: api.ToolFunction{
						Name: "get_weather",
						Parameters: api.ToolFunctionParameters{
							Properties: map[string]api.ToolProperty{
								"location": {Type: api.PropertyType{"string"}},
							},
						},
					},
				},
			},
		},
		{
			name: "complex_tool_arguments",
			input: `<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>process_data
` + "```json\n" + `{"items":["item1","item2"],"config":{"enabled":true,"threshold":0.95},"count":42}
` + "```" + `<｜tool▁call▁end｜><｜tool▁calls▁end｜>`,
			expectedToolCalls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name: "process_data",
						Arguments: api.ToolCallFunctionArguments{
							"items":  []interface{}{"item1", "item2"},
							"config": map[string]interface{}{"enabled": true, "threshold": 0.95},
							"count":  42.0,
						},
					},
				},
			},
		},
		{
			name:             "tool_output_parsing",
			input:            `<｜tool▁outputs▁begin｜><｜tool▁output▁begin｜>{"temperature": 22, "condition": "sunny"}<｜tool▁output▁end｜><｜tool▁outputs▁end｜>`,
			expectedContent:  "",
			expectedThinking: "",
		},
		{
			name: "thinking_with_multiline_content",
			input: `<think>This is line 1
This is line 2
This is line 3</think>Final response here.`,
			expectedContent:  "Final response here.",
			expectedThinking: "This is line 1\nThis is line 2\nThis is line 3",
		},
		{
			name:             "empty_thinking_tags",
			input:            "<think></think>This is content.",
			expectedContent:  "This is content.",
			expectedThinking: "",
		},
		{
			name:            "prefill_content_only",
			input:           "Continuing from previous content.",
			expectedContent: "Continuing from previous content.",
			lastMessage: &api.Message{
				Role:    "assistant",
				Content: "Previous content",
			},
		},
		{
			name:             "prefill_with_thinking",
			input:            "<think>Continuing thinking</think>Continuing content.",
			expectedContent:  "Continuing content.",
			expectedThinking: "Continuing thinking",
			lastMessage: &api.Message{
				Role: "assistant",
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			parser := &CogitoParser{}
			parser.Init(tt.tools, tt.lastMessage)

			content, thinking, toolCalls, err := parser.Add(tt.input, true)
			if err != nil {
				t.Fatalf("Add() error = %v", err)
			}

			if diff := cmp.Diff(tt.expectedContent, content); diff != "" {
				t.Errorf("content mismatch (-want +got):\n%s", diff)
			}

			if diff := cmp.Diff(tt.expectedThinking, thinking); diff != "" {
				t.Errorf("thinking mismatch (-want +got):\n%s", diff)
			}

			if diff := cmp.Diff(tt.expectedToolCalls, toolCalls); diff != "" {
				t.Errorf("tool calls mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestCogitoParser_Streaming(t *testing.T) {
	parser := &CogitoParser{}
	parser.Init(nil, nil)

	chunks := []string{
		"<think>This is ",
		"thinking content",
		".</think>This is ",
		"content.<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>test_tool\n```json\n{\"arg\":\"value\"}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>",
	}

	var finalContent, finalThinking strings.Builder
	var finalToolCalls []api.ToolCall

	for i, chunk := range chunks {
		done := i == len(chunks)-1
		content, thinking, toolCalls, err := parser.Add(chunk, done)
		if err != nil {
			t.Fatalf("Add() error on chunk %d: %v", i, err)
		}

		finalContent.WriteString(content)
		finalThinking.WriteString(thinking)
		finalToolCalls = append(finalToolCalls, toolCalls...)
	}

	expectedContent := "This is content."
	expectedThinking := "This is thinking content."
	expectedToolCalls := []api.ToolCall{
		{
			Function: api.ToolCallFunction{
				Name: "test_tool",
				Arguments: api.ToolCallFunctionArguments{
					"arg": "value",
				},
			},
		},
	}

	if finalContent.String() != expectedContent {
		t.Errorf("expected content %q, got %q", expectedContent, finalContent.String())
	}

	if finalThinking.String() != expectedThinking {
		t.Errorf("expected thinking %q, got %q", expectedThinking, finalThinking.String())
	}

	if diff := cmp.Diff(expectedToolCalls, finalToolCalls); diff != "" {
		t.Errorf("tool calls mismatch (-want +got):\n%s", diff)
	}
}

func TestCogitoParser_HasToolSupport(t *testing.T) {
	parser := &CogitoParser{}
	if !parser.HasToolSupport() {
		t.Error("CogitoParser should support tools")
	}
}

func TestCogitoParser_HasThinkingSupport(t *testing.T) {
	parser := &CogitoParser{}
	if !parser.HasThinkingSupport() {
		t.Error("CogitoParser should support thinking")
	}
}

func TestCogitoParser_Init(t *testing.T) {
	parser := &CogitoParser{}

	tools := []api.Tool{
		{Function: api.ToolFunction{Name: "test_tool"}},
	}

	lastMessage := &api.Message{Role: "assistant", Content: "previous"}

	returnedTools := parser.Init(tools, lastMessage)

	if len(returnedTools) != len(tools) {
		t.Errorf("expected %d tools returned, got %d", len(tools), len(returnedTools))
	}
}
