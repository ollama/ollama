package parsers

import (
	"testing"

	"github.com/google/go-cmp/cmp"

	"github.com/ollama/ollama/api"
)

func TestLFM2Parser(t *testing.T) {
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
			input:           "I'll check the weather.<|tool_call_start|>{\"name\":\"get_weather\",\"arguments\":{\"location\":\"Paris\"}}<|tool_call_end|>",
			expectedContent: "I'll check the weather.",
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
			name:            "multiple_tool_calls",
			input:           "Getting weather for both cities.<|tool_call_start|>{\"name\":\"get_weather\",\"arguments\":{\"location\":\"Paris\"}}<|tool_call_end|><|tool_call_start|>{\"name\":\"get_weather\",\"arguments\":{\"location\":\"London\"}}<|tool_call_end|>",
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
						Name: "get_weather",
						Arguments: testArgs(map[string]any{
							"location": "London",
						}),
					},
				},
			},
			hasThinking: false,
		},
		{
			name:            "complex_tool_arguments",
			input:           "Processing data.<|tool_call_start|>{\"name\":\"process_data\",\"arguments\":{\"items\":[\"item1\",\"item2\"],\"config\":{\"enabled\":true,\"threshold\":0.95}}}<|tool_call_end|>",
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
			name:             "thinking_with_tool_call",
			input:            "Let me check the weather...</think>I'll get that for you.<|tool_call_start|>{\"name\":\"get_weather\",\"arguments\":{\"location\":\"Paris\"}}<|tool_call_end|>",
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
			input:           "Searching for information.<|tool_call_start|>{\"name\":\"search\",\"arguments\":{\"query\":\"北京天气\",\"language\":\"中文\"}}<|tool_call_end|>",
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
			name:             "thinking_with_special_chars",
			input:            "Let me calculate: 2+2=4 & 3*3=9...</think>The results are correct!",
			expectedThinking: "Let me calculate: 2+2=4 & 3*3=9...",
			expectedContent:  "The results are correct!",
			hasThinking:      true,
		},
		{
			name:            "empty_tool_call_args",
			input:           "Pinging server.<|tool_call_start|>{\"name\":\"ping\",\"arguments\":{}}<|tool_call_end|>",
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
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			parser := &LFM2Parser{hasThinkingSupport: tt.hasThinking}
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

func TestLFM2Parser_Streaming(t *testing.T) {
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
			chunks:          []string{"I'll check weather.", "<|tool_call_start|>", "{\"name\":\"get_weather\",", "\"arguments\":{\"location\":\"Paris\"}}", "<|tool_call_end|>"},
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
			name:            "streaming_unicode_content",
			chunks:          []string{"مرحبا ", "بالعالم! ", "你好", "世界!"},
			expectedContent: "مرحبا بالعالم! 你好世界!",
			hasThinking:     false,
		},
		{
			name:            "streaming_tool_call_with_split_json",
			chunks:          []string{"Processing.", "<|tool_call_start|>{\"name\":\"calc\",\"arguments\":{\"x\":", "42,\"y\":", "24}}<|tool_call_end|>"},
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
		{
			// Test that leading whitespace after <think> is trimmed even when in separate chunks
			name:             "streaming_thinking_whitespace_after_tag",
			chunks:           []string{"<think>", "\n\n  ", "Actual thinking content", "</think>", "Response"},
			expectedThinking: "Actual thinking content",
			expectedContent:  "Response",
			hasThinking:      true,
		},
		{
			// Test whitespace between </think> and content in streaming
			name:             "streaming_whitespace_after_close_tag",
			chunks:           []string{"<think>Thinking</think>", "\n\n\n", "Response content"},
			expectedThinking: "Thinking",
			expectedContent:  "Response content",
			hasThinking:      true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			parser := &LFM2Parser{hasThinkingSupport: tt.hasThinking}
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

func TestLFM2Parser_HasThinkingSupport(t *testing.T) {
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
			parser := &LFM2Parser{hasThinkingSupport: tt.hasThinking}
			if got := parser.HasThinkingSupport(); got != tt.expectedSupport {
				t.Errorf("HasThinkingSupport() = %v, want %v", got, tt.expectedSupport)
			}
		})
	}
}

func TestLFM2Parser_HasToolSupport(t *testing.T) {
	parser := &LFM2Parser{}
	if !parser.HasToolSupport() {
		t.Error("HasToolSupport() should return true")
	}
}

func TestLFM2Parser_Init(t *testing.T) {
	parser := &LFM2Parser{hasThinkingSupport: true}
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
	if parser.state != LFM2CollectingThinking {
		t.Errorf("Expected initial state to be LFM2CollectingThinking, got %v", parser.state)
	}
}

func TestLFM2Parser_parseToolCallContent(t *testing.T) {
	tests := []struct {
		name        string
		content     string
		expected    api.ToolCall
		expectError bool
	}{
		{
			name:    "valid_tool_call",
			content: `{"name":"get_weather","arguments":{"location":"Paris"}}`,
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
			content: `{"name":"process_data","arguments":{"items":["a","b"],"config":{"enabled":true}}}`,
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
			content: `{"name":"ping","arguments":{}}`,
			expected: api.ToolCall{
				Function: api.ToolCallFunction{
					Name:      "ping",
					Arguments: api.NewToolCallFunctionArguments(),
				},
			},
		},
		{
			name:    "unicode_in_tool_name",
			content: `{"name":"获取天气","arguments":{"城市":"北京"}}`,
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
			name:    "numeric_arguments",
			content: `{"name":"calculate","arguments":{"x":3.14,"y":42,"enabled":true}}`,
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
			name:        "invalid_json",
			content:     `{invalid json}`,
			expectError: true,
		},
		{
			name:        "missing_name",
			content:     `{"arguments":{"arg":"value"}}`,
			expectError: true,
		},
		{
			name:        "empty_name",
			content:     `{"name":"","arguments":{"arg":"value"}}`,
			expectError: true,
		},
	}

	parser := &LFM2Parser{}
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

func TestLFM2Parser_parseToolCallsContent(t *testing.T) {
	tests := []struct {
		name        string
		content     string
		expected    []api.ToolCall
		expectError bool
	}{
		{
			name:    "multiple_python_style_calls",
			content: `[bash(command='curl google.com'),bash(command='curl example.com')]`,
			expected: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name: "bash",
						Arguments: testArgs(map[string]any{
							"command": "curl google.com",
						}),
					},
				},
				{
					Function: api.ToolCallFunction{
						Name: "bash",
						Arguments: testArgs(map[string]any{
							"command": "curl example.com",
						}),
					},
				},
			},
		},
		{
			name:    "single_python_style_call",
			content: `bash(command='ls -la')`,
			expected: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name: "bash",
						Arguments: testArgs(map[string]any{
							"command": "ls -la",
						}),
					},
				},
			},
		},
		{
			name:    "single_bracketed_call",
			content: `[bash(command='pwd')]`,
			expected: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name: "bash",
						Arguments: testArgs(map[string]any{
							"command": "pwd",
						}),
					},
				},
			},
		},
		{
			name:    "multiple_different_functions",
			content: `[get_weather(location='Paris'),search(query='news')]`,
			expected: []api.ToolCall{
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
						Name: "search",
						Arguments: testArgs(map[string]any{
							"query": "news",
						}),
					},
				},
			},
		},
		{
			name:    "nested_parentheses_in_arg",
			content: `bash(command='echo "(hello)"')`,
			expected: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name: "bash",
						Arguments: testArgs(map[string]any{
							"command": `echo "(hello)"`,
						}),
					},
				},
			},
		},
		{
			name:    "comma_inside_quotes",
			content: `bash(command='echo "hello, world"')`,
			expected: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name: "bash",
						Arguments: testArgs(map[string]any{
							"command": `echo "hello, world"`,
						}),
					},
				},
			},
		},
		{
			name:    "equals_inside_quotes",
			content: `bash(command='export FOO=bar')`,
			expected: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name: "bash",
						Arguments: testArgs(map[string]any{
							"command": `export FOO=bar`,
						}),
					},
				},
			},
		},
		{
			name:    "double_quotes_with_single_inside",
			content: `bash(command="echo 'hello'")`,
			expected: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name: "bash",
						Arguments: testArgs(map[string]any{
							"command": `echo 'hello'`,
						}),
					},
				},
			},
		},
		{
			name:    "multiple_args",
			content: `bash(command='ls', flag='-la', count=42)`,
			expected: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name: "bash",
						Arguments: testArgs(map[string]any{
							"command": "ls",
							"flag":    "-la",
							"count":   int64(42),
						}),
					},
				},
			},
		},
		{
			name:    "no_args",
			content: `ping()`,
			expected: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name:      "ping",
						Arguments: api.NewToolCallFunctionArguments(),
					},
				},
			},
		},
		{
			name:    "three_calls",
			content: `[a(x='1'),b(y='2'),c(z='3')]`,
			expected: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name:      "a",
						Arguments: testArgs(map[string]any{"x": "1"}),
					},
				},
				{
					Function: api.ToolCallFunction{
						Name:      "b",
						Arguments: testArgs(map[string]any{"y": "2"}),
					},
				},
				{
					Function: api.ToolCallFunction{
						Name:      "c",
						Arguments: testArgs(map[string]any{"z": "3"}),
					},
				},
			},
		},
		{
			// Note: backslash escapes are preserved as-is, not processed
			name:    "escaped_quote_in_value",
			content: `bash(command='echo \'hello\'')`,
			expected: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name: "bash",
						Arguments: testArgs(map[string]any{
							"command": `echo \'hello\'`,
						}),
					},
				},
			},
		},
	}

	parser := &LFM2Parser{}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := parser.parseToolCallsContent(tt.content)

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
				t.Errorf("parseToolCallsContent() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestLFM2Parser_EdgeCases(t *testing.T) {
	tests := []struct {
		name             string
		input            string
		expectedContent  string
		expectedThinking string
		hasThinking      bool
	}{
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
			name:            "whitespace_only_content",
			input:           "   \n\t   ",
			expectedContent: "   \n\t   ",
			hasThinking:     false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			parser := &LFM2Parser{hasThinkingSupport: tt.hasThinking}
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
