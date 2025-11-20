package parsers

import (
	"testing"

	"github.com/google/go-cmp/cmp"

	"github.com/ollama/ollama/api"
)

func TestDeepSeekParser(t *testing.T) {
	tests := []struct {
		name             string
		input            string
		expectedContent  string
		expectedThinking string
		expectedCalls    []api.ToolCall
		hasThinking      bool
	}{
		{
			name:            "simple_content",
			input:           "Hello, how are you?",
			expectedContent: "Hello, how are you?",
			hasThinking:     false,
		},
		{
			name:             "thinking_content",
			input:            "I need to think about this...</think>The answer is 42.",
			expectedThinking: "I need to think about this...",
			expectedContent:  "The answer is 42.",
			hasThinking:      true,
		},
		{
			name:            "no_thinking_simple",
			input:           "Just a regular response.",
			expectedContent: "Just a regular response.",
			hasThinking:     false,
		},
		{
			name:             "thinking_with_newlines",
			input:            "Let me think:\n- Point 1\n- Point 2</think>\n\nHere's my answer.",
			expectedThinking: "Let me think:\n- Point 1\n- Point 2",
			expectedContent:  "Here's my answer.",
			hasThinking:      true,
		},
		{
			name:            "tool_call_simple",
			input:           "I'll check the weather.<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>{\"location\":\"Paris\"}<｜tool▁call▁end｜><｜tool▁calls▁end｜>",
			expectedContent: "I'll check the weather.",
			expectedCalls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name: "get_weather",
						Arguments: api.ToolCallFunctionArguments{
							"location": "Paris",
						},
					},
				},
			},
			hasThinking: false,
		},
		{
			name:            "multiple_tool_calls",
			input:           "Getting weather for both cities.<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>{\"location\":\"Paris\"}<｜tool▁call▁end｜><｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>{\"location\":\"London\"}<｜tool▁call▁end｜><｜tool▁calls▁end｜>",
			expectedContent: "Getting weather for both cities.",
			expectedCalls: []api.ToolCall{
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
			hasThinking: false,
		},
		{
			name:            "tool_output",
			input:           "Here's the weather: <｜tool▁output▁begin｜>Temperature: 22°C, Sunny<｜tool▁output▁end｜> Hope that helps!",
			expectedContent: "Here's the weather: Temperature: 22°C, SunnyHope that helps!",
			hasThinking:     false,
		},
		{
			name:            "complex_tool_arguments",
			input:           "Processing data.<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>process_data<｜tool▁sep｜>{\"items\":[\"item1\",\"item2\"],\"config\":{\"enabled\":true,\"threshold\":0.95}}<｜tool▁call▁end｜><｜tool▁calls▁end｜>",
			expectedContent: "Processing data.",
			expectedCalls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name: "process_data",
						Arguments: api.ToolCallFunctionArguments{
							"items":  []interface{}{"item1", "item2"},
							"config": map[string]interface{}{"enabled": true, "threshold": 0.95},
						},
					},
				},
			},
			hasThinking: false,
		},
		{
			name:             "thinking_with_tool_call", // technically this can't happen, but the parser can handle it
			input:            "Let me check the weather...</think>I'll get that for you.<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>{\"location\":\"Paris\"}<｜tool▁call▁end｜><｜tool▁calls▁end｜>",
			expectedThinking: "Let me check the weather...",
			expectedContent:  "I'll get that for you.",
			expectedCalls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name: "get_weather",
						Arguments: api.ToolCallFunctionArguments{
							"location": "Paris",
						},
					},
				},
			},
			hasThinking: true,
		},
		{
			name:            "empty_content",
			input:           "",
			expectedContent: "",
			hasThinking:     false,
		},
		{
			name:             "only_thinking",
			input:            "Just thinking content</think>",
			expectedThinking: "Just thinking content",
			expectedContent:  "",
			hasThinking:      true,
		},
		{
			name:            "multiple_tool_outputs",
			input:           "Results: <｜tool▁output▁begin｜>Paris: 22°C<｜tool▁output▁end｜> and <｜tool▁output▁begin｜>London: 18°C<｜tool▁output▁end｜>",
			expectedContent: "Results: Paris: 22°Cand London: 18°C",
			hasThinking:     false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			parser := &DeepSeekParser{hasThinkingSupport: tt.hasThinking}
			parser.Init([]api.Tool{}, nil, &api.ThinkValue{Value: tt.hasThinking})

			content, thinking, calls, err := parser.Add(tt.input, true)
			if err != nil {
				t.Fatalf("Add() error = %v", err)
			}

			if diff := cmp.Diff(tt.expectedContent, content); diff != "" {
				t.Errorf("Content mismatch (-want +got):\n%s", diff)
			}

			if diff := cmp.Diff(tt.expectedThinking, thinking); diff != "" {
				t.Errorf("Thinking mismatch (-want +got):\n%s", diff)
			}

			if diff := cmp.Diff(tt.expectedCalls, calls); diff != "" {
				t.Errorf("Tool calls mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestDeepSeekParser_Streaming(t *testing.T) {
	tests := []struct {
		name             string
		chunks           []string
		expectedContent  string
		expectedThinking string
		expectedCalls    []api.ToolCall
		hasThinking      bool
	}{
		{
			name:            "streaming_simple_content",
			chunks:          []string{"Hello, ", "how are ", "you?"},
			expectedContent: "Hello, how are you?",
			hasThinking:     false,
		},
		{
			name:             "streaming_thinking",
			chunks:           []string{"I need to ", "think about this", "...</think>", "The answer is 42."},
			expectedThinking: "I need to think about this...",
			expectedContent:  "The answer is 42.",
			hasThinking:      true,
		},
		{
			name:            "streaming_tool_call",
			chunks:          []string{"I'll check weather.", "<｜tool▁calls▁begin｜>", "<｜tool▁call▁begin｜>get_weather", "<｜tool▁sep｜>{\"location\":\"Paris\"}", "<｜tool▁call▁end｜><｜tool▁calls▁end｜>"},
			expectedContent: "I'll check weather.",
			expectedCalls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name: "get_weather",
						Arguments: api.ToolCallFunctionArguments{
							"location": "Paris",
						},
					},
				},
			},
			hasThinking: false,
		},
		{
			name:             "streaming_thinking_with_partial_tag",
			chunks:           []string{"Thinking about this", "...</", "think>", "Done thinking."},
			expectedThinking: "Thinking about this...",
			expectedContent:  "Done thinking.",
			hasThinking:      true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			parser := &DeepSeekParser{hasThinkingSupport: tt.hasThinking}
			parser.Init([]api.Tool{}, nil, &api.ThinkValue{Value: tt.hasThinking})

			var allContent, allThinking string
			var allCalls []api.ToolCall

			for i, chunk := range tt.chunks {
				done := i == len(tt.chunks)-1
				content, thinking, calls, err := parser.Add(chunk, done)
				if err != nil {
					t.Fatalf("Add() error = %v", err)
				}

				allContent += content
				allThinking += thinking
				allCalls = append(allCalls, calls...)
			}

			if diff := cmp.Diff(tt.expectedContent, allContent); diff != "" {
				t.Errorf("Content mismatch (-want +got):\n%s", diff)
			}

			if diff := cmp.Diff(tt.expectedThinking, allThinking); diff != "" {
				t.Errorf("Thinking mismatch (-want +got):\n%s", diff)
			}

			if diff := cmp.Diff(tt.expectedCalls, allCalls); diff != "" {
				t.Errorf("Tool calls mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestDeepSeekParser_HasThinkingSupport(t *testing.T) {
	tests := []struct {
		name            string
		hasThinking     bool
		expectedSupport bool
	}{
		{
			name:            "thinking_enabled",
			hasThinking:     true,
			expectedSupport: true,
		},
		{
			name:            "thinking_disabled",
			hasThinking:     false,
			expectedSupport: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			parser := &DeepSeekParser{hasThinkingSupport: tt.hasThinking}
			if got := parser.HasThinkingSupport(); got != tt.expectedSupport {
				t.Errorf("HasThinkingSupport() = %v, want %v", got, tt.expectedSupport)
			}
		})
	}
}

func TestDeepSeekParser_HasToolSupport(t *testing.T) {
	parser := &DeepSeekParser{}
	if !parser.HasToolSupport() {
		t.Error("HasToolSupport() should return true")
	}
}

func TestDeepSeekParser_Init(t *testing.T) {
	parser := &DeepSeekParser{hasThinkingSupport: true}
	tools := []api.Tool{
		{
			Type: "function",
			Function: api.ToolFunction{
				Name: "test_tool",
			},
		},
	}

	returnedTools := parser.Init(tools, nil, &api.ThinkValue{Value: true})

	if diff := cmp.Diff(tools, returnedTools); diff != "" {
		t.Errorf("Init() returned tools mismatch (-want +got):\n%s", diff)
	}

	// Test initial state is set to thinking when enabled
	if parser.state != DeepSeekCollectingThinking {
		t.Errorf("Expected initial state to be DeepSeekCollectingThinking, got %v", parser.state)
	}
}

func TestDeepSeekParser_parseToolCallContent(t *testing.T) {
	tests := []struct {
		name        string
		content     string
		expected    api.ToolCall
		expectError bool
	}{
		{
			name:    "valid_tool_call",
			content: "get_weather<｜tool▁sep｜>{\"location\":\"Paris\"}",
			expected: api.ToolCall{
				Function: api.ToolCallFunction{
					Name: "get_weather",
					Arguments: api.ToolCallFunctionArguments{
						"location": "Paris",
					},
				},
			},
		},
		{
			name:    "complex_arguments",
			content: "process_data<｜tool▁sep｜>{\"items\":[\"a\",\"b\"],\"config\":{\"enabled\":true}}",
			expected: api.ToolCall{
				Function: api.ToolCallFunction{
					Name: "process_data",
					Arguments: api.ToolCallFunctionArguments{
						"items":  []interface{}{"a", "b"},
						"config": map[string]interface{}{"enabled": true},
					},
				},
			},
		},
		{
			name:        "invalid_format_no_separator",
			content:     "get_weather{\"location\":\"Paris\"}",
			expectError: true,
		},
		{
			name:        "invalid_json",
			content:     "get_weather<｜tool▁sep｜>{invalid json}",
			expectError: true,
		},
	}

	parser := &DeepSeekParser{}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := parser.parseToolCallContent(tt.content)

			if tt.expectError {
				if err == nil {
					t.Error("Expected error but got none")
				}
				return
			}

			if err != nil {
				t.Fatalf("Unexpected error: %v", err)
			}

			if diff := cmp.Diff(tt.expected, result); diff != "" {
				t.Errorf("parseToolCallContent() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestDeepSeekParser_EdgeCases(t *testing.T) {
	tests := []struct {
		name             string
		input            string
		expectedContent  string
		expectedThinking string
		hasThinking      bool
	}{
		{
			name:             "nested_think_tags_in_thinking",
			input:            "Outer thinking <think>inner</think> content</think>Final content",
			expectedThinking: "Outer thinking <think>inner",
			expectedContent:  "content</think>Final content",
			hasThinking:      true,
		},
		{
			name:             "multiple_think_close_tags",
			input:            "First thought</think>Second thought</think>Final content",
			expectedThinking: "First thought",
			expectedContent:  "Second thought</think>Final content",
			hasThinking:      true,
		},
		{
			name:             "empty_thinking_content",
			input:            "</think>Just content",
			expectedThinking: "",
			expectedContent:  "Just content",
			hasThinking:      true,
		},
		{
			name:            "thinking_disabled_with_think_tags",
			input:           "Some content</think>More content",
			expectedContent: "Some content</think>More content",
			hasThinking:     false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			parser := &DeepSeekParser{hasThinkingSupport: tt.hasThinking}
			parser.Init([]api.Tool{}, nil, &api.ThinkValue{Value: tt.hasThinking})

			content, thinking, _, err := parser.Add(tt.input, true)
			if err != nil {
				t.Fatalf("Add() error = %v", err)
			}

			if diff := cmp.Diff(tt.expectedContent, content); diff != "" {
				t.Errorf("Content mismatch (-want +got):\n%s", diff)
			}

			if diff := cmp.Diff(tt.expectedThinking, thinking); diff != "" {
				t.Errorf("Thinking mismatch (-want +got):\n%s", diff)
			}
		})
	}
}
