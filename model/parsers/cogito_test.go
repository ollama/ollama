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
			input:            "This is thinking content.</think>This is response content.",
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
			input: `I need to check the weather.</think><｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather
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
			input: `This is line 1
This is line 2
This is line 3</think>Final response here.`,
			expectedContent:  "Final response here.",
			expectedThinking: "This is line 1\nThis is line 2\nThis is line 3",
		},
		{
			name:             "no_thinking_simple",
			input:            "This is content.",
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
			input:            "Continuing thinking</think>Continuing content.",
			expectedContent:  "Continuing content.",
			expectedThinking: "Continuing thinking",
			lastMessage: &api.Message{
				Role: "assistant",
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Use thinking-enabled parser for tests that expect thinking
			hasThinking := tt.expectedThinking != ""
			parser := &CogitoParser{hasThinkingSupport: hasThinking}
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
	parser := &CogitoParser{hasThinkingSupport: true}
	parser.Init(nil, nil)

	chunks := []string{
		"This is ",
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
	// Test thinking-enabled parser
	thinkingParser := &CogitoParser{hasThinkingSupport: true}
	if !thinkingParser.HasThinkingSupport() {
		t.Error("CogitoParser with hasThinkingSupport=true should support thinking")
	}

	// Test non-thinking parser
	nonThinkingParser := &CogitoParser{hasThinkingSupport: false}
	if nonThinkingParser.HasThinkingSupport() {
		t.Error("CogitoParser with hasThinkingSupport=false should not support thinking")
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

func TestCogitoParser_parseToolCallContent(t *testing.T) {
	tests := []struct {
		name        string
		content     string
		expected    api.ToolCall
		expectError bool
	}{
		{
			name: "valid_tool_call_standard_format",
			content: `function<｜tool▁sep｜>get_weather
` + "```json\n" + `{"location":"Paris"}
` + "```",
			expected: api.ToolCall{
				Function: api.ToolCallFunction{
					Name: "get_weather",
					Arguments: api.ToolCallFunctionArguments{
						"location": "Paris",
					},
				},
			},
			expectError: false,
		},
		{
			name:    "valid_tool_call_without_newlines",
			content: `function<｜tool▁sep｜>get_weather` + "```json" + `{"location":"Paris"}` + "```",
			expected: api.ToolCall{
				Function: api.ToolCallFunction{
					Name: "get_weather",
					Arguments: api.ToolCallFunctionArguments{
						"location": "Paris",
					},
				},
			},
			expectError: false,
		},
		{
			name: "valid_tool_call_complex_args",
			content: `function<｜tool▁sep｜>process_data
` + "```json\n" + `{"items":["item1","item2"],"config":{"enabled":true},"count":42}
` + "```",
			expected: api.ToolCall{
				Function: api.ToolCallFunction{
					Name: "process_data",
					Arguments: api.ToolCallFunctionArguments{
						"items":  []interface{}{"item1", "item2"},
						"config": map[string]interface{}{"enabled": true},
						"count":  42.0,
					},
				},
			},
			expectError: false,
		},
		{
			name: "valid_tool_call_empty_args",
			content: `function<｜tool▁sep｜>no_args_tool
` + "```json\n" + `{}
` + "```",
			expected: api.ToolCall{
				Function: api.ToolCallFunction{
					Name:      "no_args_tool",
					Arguments: api.ToolCallFunctionArguments{},
				},
			},
			expectError: false,
		},
		{
			name:        "missing_separator",
			content:     `functionget_weather` + "```json\n" + `{"location":"Paris"}` + "\n```",
			expected:    api.ToolCall{},
			expectError: true,
		},
		{
			name:        "invalid_function_type",
			content:     `not_function<｜tool▁sep｜>get_weather` + "```json\n" + `{"location":"Paris"}` + "\n```",
			expected:    api.ToolCall{},
			expectError: true,
		},
		{
			name:        "missing_json_block_start",
			content:     `function<｜tool▁sep｜>get_weather{"location":"Paris"}` + "```",
			expected:    api.ToolCall{},
			expectError: true,
		},
		{
			name:        "missing_json_block_end",
			content:     `function<｜tool▁sep｜>get_weather` + "```json\n" + `{"location":"Paris"}`,
			expected:    api.ToolCall{},
			expectError: true,
		},
		{
			name:        "invalid_json",
			content:     `function<｜tool▁sep｜>get_weather` + "```json\n" + `{location:Paris}` + "\n```",
			expected:    api.ToolCall{},
			expectError: true,
		},
		{
			name:        "empty_function_type",
			content:     `<｜tool▁sep｜>get_weather` + "```json\n" + `{"location":"Paris"}` + "\n```",
			expected:    api.ToolCall{},
			expectError: true,
		},
		{
			name: "tool_with_spaces_in_name",
			content: `function<｜tool▁sep｜>  get_weather  
` + "```json\n" + `{"location":"Paris"}
` + "```",
			expected: api.ToolCall{
				Function: api.ToolCallFunction{
					Name: "get_weather",
					Arguments: api.ToolCallFunctionArguments{
						"location": "Paris",
					},
				},
			},
			expectError: false,
		},
		{
			name: "tool_with_multiline_json",
			content: `function<｜tool▁sep｜>get_weather
` + "```json\n" + `{
  "location": "Paris",
  "units": "metric"
}
` + "```",
			expected: api.ToolCall{
				Function: api.ToolCallFunction{
					Name: "get_weather",
					Arguments: api.ToolCallFunctionArguments{
						"location": "Paris",
						"units":    "metric",
					},
				},
			},
			expectError: false,
		},
		{
			name: "tool_with_nested_objects",
			content: `function<｜tool▁sep｜>complex_tool
` + "```json\n" + `{"nested":{"deep":{"value":123}}}
` + "```",
			expected: api.ToolCall{
				Function: api.ToolCallFunction{
					Name: "complex_tool",
					Arguments: api.ToolCallFunctionArguments{
						"nested": map[string]interface{}{
							"deep": map[string]interface{}{
								"value": 123.0,
							},
						},
					},
				},
			},
			expectError: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			parser := &CogitoParser{}

			result, err := parser.parseToolCallContent(tt.content)

			if tt.expectError {
				if err == nil {
					t.Errorf("expected error but got none")
				}
				return
			}

			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			if diff := cmp.Diff(tt.expected, result); diff != "" {
				t.Errorf("tool call mismatch (-want +got):\n%s", diff)
			}
		})
	}
}
