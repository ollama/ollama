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
						Index: 0,
						Name:  "get_weather",
						Arguments: testArgs(map[string]any{
							"location": "Paris",
						}),
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
						Arguments: testArgs(map[string]any{
							"location": "Paris",
						}),
					},
				},
				{
					Function: api.ToolCallFunction{
						Index: 1,
						Name:  "get_weather",
						Arguments: testArgs(map[string]any{
							"location": "London",
						}),
					},
				},
			},
			hasThinking: false,
		},
		{
			name:            "tool_output",
			input:           "Here's the weather: <｜tool▁output▁begin｜>Temperature: 22°C, Sunny<｜tool▁output▁end｜> Hope that helps!",
			expectedContent: "Here's the weather: Temperature: 22°C, Sunny Hope that helps!",
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
						Arguments: testArgs(map[string]any{
							"items":  []interface{}{"item1", "item2"},
							"config": map[string]interface{}{"enabled": true, "threshold": 0.95},
						}),
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
						Arguments: testArgs(map[string]any{
							"location": "Paris",
						}),
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
			expectedContent: "Results: Paris: 22°C and London: 18°C",
			hasThinking:     false,
		},
		{
			name:            "unicode_content",
			input:           "مرحبا بالعالم! 你好世界! 🌍",
			expectedContent: "مرحبا بالعالم! 你好世界! 🌍",
			hasThinking:     false,
		},
		{
			name:            "emoji_passthrough",
			input:           "Task completed ✅ 🎉",
			expectedContent: "Task completed ✅ 🎉",
			hasThinking:     false,
		},
		{
			name:            "emoji_after_tool_call",
			input:           "I'll help you.<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>{\"location\":\"Tokyo\"}<｜tool▁call▁end｜><｜tool▁calls▁end｜>完成 ✅",
			expectedContent: "I'll help you.完成 ✅",
			expectedCalls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name: "get_weather",
						Arguments: testArgs(map[string]any{
							"location": "Tokyo",
						}),
					},
				},
			},
			hasThinking: false,
		},
		{
			name:            "newlines_and_whitespace",
			input:           "Line 1\n\nLine 3\t\tTabbed content",
			expectedContent: "Line 1\n\nLine 3\t\tTabbed content",
			hasThinking:     false,
		},
		{
			name:             "thinking_with_unicode",
			input:            "我在思考这个问题...</think>答案是42。",
			expectedThinking: "我在思考这个问题...",
			expectedContent:  "答案是42。",
			hasThinking:      true,
		},
		{
			name:            "tool_call_with_unicode_args",
			input:           "Searching for information.<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>search<｜tool▁sep｜>{\"query\":\"北京天气\",\"language\":\"中文\"}<｜tool▁call▁end｜><｜tool▁calls▁end｜>",
			expectedContent: "Searching for information.",
			expectedCalls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name: "search",
						Arguments: testArgs(map[string]any{
							"query":    "北京天气",
							"language": "中文",
						}),
					},
				},
			},
			hasThinking: false,
		},
		{
			name:            "tool_output_with_unicode",
			input:           "天气信息: <｜tool▁output▁begin｜>北京: 25°C, 晴天<｜tool▁output▁end｜> 希望对您有帮助!",
			expectedContent: "天气信息: 北京: 25°C, 晴天 希望对您有帮助!",
			hasThinking:     false,
		},
		{
			name:            "mixed_content_with_special_chars",
			input:           "Price: $100 & tax @ 10% = $110 <｜tool▁output▁begin｜>Total: $110<｜tool▁output▁end｜> (final)",
			expectedContent: "Price: $100 & tax @ 10% = $110 Total: $110 (final)",
			hasThinking:     false,
		},
		{
			name:            "tool_call_with_special_chars",
			input:           "Processing data.<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>execute_command<｜tool▁sep｜>{\"command\":\"ls && echo \\\"done\\\"\",\"path\":\"/home/user\"}<｜tool▁call▁end｜><｜tool▁calls▁end｜>",
			expectedContent: "Processing data.",
			expectedCalls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name: "execute_command",
						Arguments: testArgs(map[string]any{
							"command": "ls && echo \"done\"",
							"path":    "/home/user",
						}),
					},
				},
			},
			hasThinking: false,
		},
		{
			name:             "thinking_with_special_chars",
			input:            "Let me calculate: 2+2=4 & 3*3=9...</think>The results are correct!",
			expectedThinking: "Let me calculate: 2+2=4 & 3*3=9...",
			expectedContent:  "The results are correct!",
			hasThinking:      true,
		},
		{
			name:            "empty_tool_call_args",
			input:           "Pinging server.<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>ping<｜tool▁sep｜>{}<｜tool▁call▁end｜><｜tool▁calls▁end｜>",
			expectedContent: "Pinging server.",
			expectedCalls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name:      "ping",
						Arguments: api.NewToolCallFunctionArguments(),
					},
				},
			},
			hasThinking: false,
		},
		{
			name:            "empty_tool_output",
			input:           "Checking status: <｜tool▁output▁begin｜><｜tool▁output▁end｜> No output received.",
			expectedContent: "Checking status:  No output received.",
			hasThinking:     false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			parser := &DeepSeek3Parser{hasThinkingSupport: tt.hasThinking}
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

			if diff := cmp.Diff(tt.expectedCalls, calls, argsComparer); diff != "" {
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
						Arguments: testArgs(map[string]any{
							"location": "Paris",
						}),
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
		{
			name:            "streaming_tool_output",
			chunks:          []string{"Weather info: ", "<｜tool▁output▁begin｜>", "25°C, Sunny", "<｜tool▁output▁end｜>", " Enjoy!"},
			expectedContent: "Weather info: 25°C, Sunny Enjoy!",
			hasThinking:     false,
		},
		{
			name:            "streaming_with_split_tags",
			chunks:          []string{"Content before ", "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>test", "<｜tool▁sep｜>{}", "<｜tool▁call▁end｜><｜tool▁calls▁end｜>", " after"},
			expectedContent: "Content before  after",
			expectedCalls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name:      "test",
						Arguments: api.NewToolCallFunctionArguments(),
					},
				},
			},
			hasThinking: false,
		},
		{
			name:             "streaming_thinking_with_split_end_tag",
			chunks:           []string{"Thinking content", "</th", "ink>", "Regular content"},
			expectedThinking: "Thinking content",
			expectedContent:  "Regular content",
			hasThinking:      true,
		},
		{
			name:            "streaming_unicode_content",
			chunks:          []string{"مرحبا ", "بالعالم! ", "你好", "世界!"},
			expectedContent: "مرحبا بالعالم! 你好世界!",
			hasThinking:     false,
		},
		{
			name:            "streaming_multiple_tool_outputs",
			chunks:          []string{"Results: ", "<｜tool▁output▁begin｜>", "Paris: 22°C", "<｜tool▁output▁end｜>", " and ", "<｜tool▁output▁begin｜>", "London: 18°C", "<｜tool▁output▁end｜>"},
			expectedContent: "Results: Paris: 22°C and London: 18°C",
			hasThinking:     false,
		},
		{
			name:            "streaming_tool_call_with_split_json",
			chunks:          []string{"Processing.", "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>calc<｜tool▁sep｜>{\"x\":", "42,\"y\":", "24}<｜tool▁call▁end｜><｜tool▁calls▁end｜>"},
			expectedContent: "Processing.",
			expectedCalls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name: "calc",
						Arguments: testArgs(map[string]any{
							"x": float64(42),
							"y": float64(24),
						}),
					},
				},
			},
			hasThinking: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			parser := &DeepSeek3Parser{hasThinkingSupport: tt.hasThinking}
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

			if diff := cmp.Diff(tt.expectedCalls, allCalls, argsComparer); diff != "" {
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
			parser := &DeepSeek3Parser{hasThinkingSupport: tt.hasThinking}
			if got := parser.HasThinkingSupport(); got != tt.expectedSupport {
				t.Errorf("HasThinkingSupport() = %v, want %v", got, tt.expectedSupport)
			}
		})
	}
}

func TestDeepSeekParser_HasToolSupport(t *testing.T) {
	parser := &DeepSeek3Parser{}
	if !parser.HasToolSupport() {
		t.Error("HasToolSupport() should return true")
	}
}

func TestDeepSeekParser_Init(t *testing.T) {
	parser := &DeepSeek3Parser{hasThinkingSupport: true}
	tools := []api.Tool{
		{
			Type: "function",
			Function: api.ToolFunction{
				Name: "test_tool",
			},
		},
	}

	returnedTools := parser.Init(tools, nil, &api.ThinkValue{Value: true})

	if diff := cmp.Diff(tools, returnedTools, toolsComparer); diff != "" {
		t.Errorf("Init() returned tools mismatch (-want +got):\n%s", diff)
	}

	// Test initial state is set to thinking when enabled
	if parser.state != DeepSeekCollectingThinking {
		t.Errorf("Expected initial state to be DeepSeekCollectingThinking, got %v", parser.state)
	}
}

func TestDeepSeek3Parser_parseToolCallContent(t *testing.T) {
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
					Arguments: testArgs(map[string]any{
						"location": "Paris",
					}),
				},
			},
		},
		{
			name:    "complex_arguments",
			content: "process_data<｜tool▁sep｜>{\"items\":[\"a\",\"b\"],\"config\":{\"enabled\":true}}",
			expected: api.ToolCall{
				Function: api.ToolCallFunction{
					Name: "process_data",
					Arguments: testArgs(map[string]any{
						"items":  []interface{}{"a", "b"},
						"config": map[string]interface{}{"enabled": true},
					}),
				},
			},
		},
		{
			name:    "empty_arguments",
			content: "ping<｜tool▁sep｜>{}",
			expected: api.ToolCall{
				Function: api.ToolCallFunction{
					Name:      "ping",
					Arguments: api.NewToolCallFunctionArguments(),
				},
			},
		},
		{
			name:    "unicode_in_tool_name",
			content: "获取天气<｜tool▁sep｜>{\"城市\":\"北京\"}",
			expected: api.ToolCall{
				Function: api.ToolCallFunction{
					Name: "获取天气",
					Arguments: testArgs(map[string]any{
						"城市": "北京",
					}),
				},
			},
		},
		{
			name:    "special_chars_in_arguments",
			content: "execute<｜tool▁sep｜>{\"command\":\"ls && echo \\\"done\\\"\",\"path\":\"/home/user\"}",
			expected: api.ToolCall{
				Function: api.ToolCallFunction{
					Name: "execute",
					Arguments: testArgs(map[string]any{
						"command": "ls && echo \"done\"",
						"path":    "/home/user",
					}),
				},
			},
		},
		{
			name:    "numeric_arguments",
			content: "calculate<｜tool▁sep｜>{\"x\":3.14,\"y\":42,\"enabled\":true}",
			expected: api.ToolCall{
				Function: api.ToolCallFunction{
					Name: "calculate",
					Arguments: testArgs(map[string]any{
						"x":       3.14,
						"y":       float64(42),
						"enabled": true,
					}),
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
		{
			name:        "empty_tool_name",
			content:     "<｜tool▁sep｜>{\"arg\":\"value\"}",
			expectError: false, // This should work, just empty name
			expected: api.ToolCall{
				Function: api.ToolCallFunction{
					Name: "",
					Arguments: testArgs(map[string]any{
						"arg": "value",
					}),
				},
			},
		},
		{
			name:        "missing_json_part",
			content:     "tool_name<｜tool▁sep｜>",
			expectError: true,
		},
	}

	parser := &DeepSeek3Parser{}
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

			if diff := cmp.Diff(tt.expected, result, argsComparer); diff != "" {
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
		{
			name:            "malformed_tool_call_missing_sep",
			input:           "Testing.<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>bad_tool{\"arg\":\"value\"}<｜tool▁call▁end｜><｜tool▁calls▁end｜>",
			expectedContent: "Testing.",
			hasThinking:     false,
		},
		{
			name:            "malformed_tool_call_invalid_json",
			input:           "Testing.<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>bad_tool<｜tool▁sep｜>{invalid json}<｜tool▁call▁end｜><｜tool▁calls▁end｜>",
			expectedContent: "Testing.",
			hasThinking:     false,
		},
		{
			name:            "partial_tool_tag_at_end",
			input:           "Content with partial <｜tool▁calls▁",
			expectedContent: "Content with partial <｜tool▁calls▁",
			hasThinking:     false,
		},
		{
			name:            "partial_think_tag_at_end",
			input:           "Thinking content</th",
			expectedContent: "Thinking content</th",
			hasThinking:     false,
		},
		{
			name:             "partial_think_tag_at_end_with_thinking",
			input:            "Thinking content</th",
			expectedThinking: "Thinking content",
			expectedContent:  "",
			hasThinking:      true,
		},
		{
			name:            "whitespace_only_content",
			input:           "   \n\t   ",
			expectedContent: "   \n\t   ",
			hasThinking:     false,
		},
		{
			name:            "tool_output_with_newlines",
			input:           "Output:\n<｜tool▁output▁begin｜>Line 1\nLine 2\nLine 3<｜tool▁output▁end｜>\nDone.",
			expectedContent: "Output:\nLine 1\nLine 2\nLine 3\nDone.",
			hasThinking:     false,
		},
		{
			name:            "consecutive_tool_calls",
			input:           "First.<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>tool1<｜tool▁sep｜>{}<｜tool▁call▁end｜><｜tool▁calls▁end｜>Second.<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>tool2<｜tool▁sep｜>{}<｜tool▁call▁end｜><｜tool▁calls▁end｜>",
			expectedContent: "First.",
			hasThinking:     false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			parser := &DeepSeek3Parser{hasThinkingSupport: tt.hasThinking}
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

func TestDeepSeekParser_ThinkSuppression(t *testing.T) {
	t.Run("think false suppresses reasoning", func(t *testing.T) {
		parser := &DeepSeek3Parser{hasThinkingSupport: true}
		parser.Init(nil, nil, &api.ThinkValue{Value: false})

		content, thinking, _, err := parser.Add("hidden reasoning...</think>visible answer", true)
		if err != nil {
			t.Fatal(err)
		}
		if thinking != "" {
			t.Errorf("expected thinking to be suppressed, got: %q", thinking)
		}
		if content != "visible answer" {
			t.Errorf("expected content without thinking, got: %q", content)
		}
	})

	t.Run("think true returns reasoning", func(t *testing.T) {
		parser := &DeepSeek3Parser{hasThinkingSupport: true}
		parser.Init(nil, nil, &api.ThinkValue{Value: true})

		content, thinking, _, err := parser.Add("reasoning here...</think>answer", true)
		if err != nil {
			t.Fatal(err)
		}
		if thinking != "reasoning here..." {
			t.Errorf("expected thinking content, got: %q", thinking)
		}
		if content != "answer" {
			t.Errorf("expected content, got: %q", content)
		}
	})

	t.Run("no thinking support unaffected", func(t *testing.T) {
		parser := &DeepSeek3Parser{hasThinkingSupport: false}
		parser.Init(nil, nil, &api.ThinkValue{Value: false})

		content, thinking, _, err := parser.Add("plain text", true)
		if err != nil {
			t.Fatal(err)
		}
		if thinking != "" {
			t.Errorf("expected no thinking, got: %q", thinking)
		}
		if content != "plain text" {
			t.Errorf("expected plain text, got: %q", content)
		}
	})

	t.Run("streaming think false suppresses across chunks", func(t *testing.T) {
		parser := &DeepSeek3Parser{hasThinkingSupport: true}
		parser.Init(nil, nil, &api.ThinkValue{Value: false})

		chunks := []string{"hidden ", "reasoning...</think>", "visible ", "answer"}
		var allContent, allThinking string
		for i, chunk := range chunks {
			content, thinking, _, err := parser.Add(chunk, i == len(chunks)-1)
			if err != nil {
				t.Fatal(err)
			}
			allContent += content
			allThinking += thinking
		}
		if allThinking != "" {
			t.Errorf("expected no thinking in streaming, got: %q", allThinking)
		}
		if allContent != "visible answer" {
			t.Errorf("expected 'visible answer', got: %q", allContent)
		}
	})
}
