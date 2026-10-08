package parsers

import (
	"slices"
	"testing"

	"github.com/google/go-cmp/cmp"

	"github.com/ollama/ollama/api"
)

func minicpm5TestTools() []api.Tool {
	return []api.Tool{
		{
			Type: "function",
			Function: api.ToolFunction{
				Name:        "get_weather",
				Description: "Get the weather for a city",
				Parameters: api.ToolFunctionParameters{
					Properties: testPropsMap(map[string]api.ToolProperty{
						"city": {Type: api.PropertyType{"string"}},
					}),
				},
			},
		},
	}
}

func TestMiniCPM5Parser(t *testing.T) {
	tests := []struct {
		name             string
		input            string
		expectedContent  string
		expectedThinking string
		expectedCalls    []api.ToolCall
		tools            []api.Tool
		hasThinking      bool
	}{
		{
			name:            "simple_content",
			input:           "Hello, how are you?",
			expectedContent: "Hello, how are you?",
			tools:           minicpm5TestTools(),
			hasThinking:     false,
		},
		{
			name:             "thinking_then_content",
			input:            "<think>Let me think about this.</think>The answer is 42.",
			expectedThinking: "Let me think about this.",
			expectedContent:  "The answer is 42.",
			tools:            minicpm5TestTools(),
			hasThinking:      true,
		},
		{
			name:             "thinking_without_open_tag",
			input:            "The user wants the weather.</think>It is sunny.",
			expectedThinking: "The user wants the weather.",
			expectedContent:  "It is sunny.",
			tools:            minicpm5TestTools(),
			hasThinking:      true,
		},
		{
			name:             "native_tool_call",
			input:            "<think>The user wants the weather in Paris.</think><function name=\"get_weather\"><param name=\"city\">Paris</param></function>",
			expectedThinking: "The user wants the weather in Paris.",
			expectedContent:  "",
			expectedCalls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Index: 0,
						Name:  "get_weather",
						Arguments: testArgs(map[string]any{
							"city": "Paris",
						}),
					},
				},
			},
			tools:       minicpm5TestTools(),
			hasThinking: true,
		},
		{
			name:            "tool_call_without_thinking",
			input:           "<function name=\"get_weather\"><param name=\"city\">Paris</param></function>",
			expectedContent: "",
			expectedCalls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Index: 0,
						Name:  "get_weather",
						Arguments: testArgs(map[string]any{
							"city": "Paris",
						}),
					},
				},
			},
			tools:       minicpm5TestTools(),
			hasThinking: false,
		},
		{
			name:            "tool_call_with_content_before",
			input:           "I'll check the weather for you.<function name=\"get_weather\"><param name=\"city\">Paris</param></function>",
			expectedContent: "I'll check the weather for you.",
			expectedCalls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Index: 0,
						Name:  "get_weather",
						Arguments: testArgs(map[string]any{
							"city": "Paris",
						}),
					},
				},
			},
			tools:       minicpm5TestTools(),
			hasThinking: false,
		},
		{
			name:            "tool_call_with_arguments_wrapper",
			input:           "<function name=\"get_weather\"><arguments><param name=\"city\">Paris</param></arguments></function>",
			expectedContent: "",
			expectedCalls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Index: 0,
						Name:  "get_weather",
						Arguments: testArgs(map[string]any{
							"city": "Paris",
						}),
					},
				},
			},
			tools:       minicpm5TestTools(),
			hasThinking: false,
		},
		{
			name:            "tool_call_with_cdata_param",
			input:           "<function name=\"get_weather\"><param name=\"city\"><![CDATA[Paris & <surroundings>]]></param></function>",
			expectedContent: "",
			expectedCalls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Index: 0,
						Name:  "get_weather",
						Arguments: testArgs(map[string]any{
							"city": "Paris & <surroundings>",
						}),
					},
				},
			},
			tools:       minicpm5TestTools(),
			hasThinking: false,
		},
		{
			name:            "multiple_tool_calls",
			input:           "<function name=\"get_weather\"><param name=\"city\">Paris</param></function><function name=\"get_weather\"><param name=\"city\">London</param></function>",
			expectedContent: "",
			expectedCalls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Index: 0,
						Name:  "get_weather",
						Arguments: testArgs(map[string]any{
							"city": "Paris",
						}),
					},
				},
				{
					Function: api.ToolCallFunction{
						Index: 1,
						Name:  "get_weather",
						Arguments: testArgs(map[string]any{
							"city": "London",
						}),
					},
				},
			},
			tools:       minicpm5TestTools(),
			hasThinking: false,
		},
		{
			name:            "unknown_tool_name_falls_back_to_content",
			input:           "<function name=\"unknown_tool\"><param name=\"city\">Paris</param></function>",
			expectedContent: "<function name=\"unknown_tool\"><param name=\"city\">Paris</param></function>",
			tools:           minicpm5TestTools(),
			hasThinking:     false,
		},
		{
			name:            "content_mentioning_function_word",
			input:           "The function of this tool is simple.",
			expectedContent: "The function of this tool is simple.",
			tools:           minicpm5TestTools(),
			hasThinking:     false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			parser := &MiniCPM5Parser{hasThinkingSupport: tt.hasThinking}
			parser.Init(tt.tools, nil, &api.ThinkValue{Value: tt.hasThinking})

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

func TestMiniCPM5Parser_Streaming(t *testing.T) {
	tests := []struct {
		name             string
		chunks           []string
		expectedContent  string
		expectedThinking string
		expectedCalls    []api.ToolCall
		tools            []api.Tool
		hasThinking      bool
	}{
		{
			name:            "streaming_simple_content",
			chunks:          []string{"Hello, ", "how are ", "you?"},
			expectedContent: "Hello, how are you?",
			tools:           minicpm5TestTools(),
			hasThinking:     false,
		},
		{
			name:             "streaming_thinking_split_close_tag",
			chunks:           []string{"<think>Let me ", "think about this.</th", "ink>", "The answer is 42."},
			expectedThinking: "Let me think about this.",
			expectedContent:  "The answer is 42.",
			tools:            minicpm5TestTools(),
			hasThinking:      true,
		},
		{
			name:             "streaming_tool_call_split_tags",
			chunks:           []string{"<think>The user wants the weather.</think><func", "tion name=\"get_weather\">", "<param name=\"city\">Par", "is</param></function>"},
			expectedThinking: "The user wants the weather.",
			expectedCalls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Index: 0,
						Name:  "get_weather",
						Arguments: testArgs(map[string]any{
							"city": "Paris",
						}),
					},
				},
			},
			tools:       minicpm5TestTools(),
			hasThinking: true,
		},
		{
			name:            "streaming_tool_call_per_token",
			chunks:          []string{"<", "function", " name=", "\"get_weather\"", ">", "<param", " name=\"city\"", ">Paris<", "/param>", "</", "function>"},
			expectedContent: "",
			expectedCalls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Index: 0,
						Name:  "get_weather",
						Arguments: testArgs(map[string]any{
							"city": "Paris",
						}),
					},
				},
			},
			tools:       minicpm5TestTools(),
			hasThinking: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			parser := &MiniCPM5Parser{hasThinkingSupport: tt.hasThinking}
			parser.Init(tt.tools, nil, &api.ThinkValue{Value: tt.hasThinking})

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

func TestMiniCPM5Parser_PreservedTokens(t *testing.T) {
	parser := &MiniCPM5Parser{}
	tokens := parser.PreservedTokens()

	for _, want := range []string{
		"<tool_call>", "</tool_call>",
		"<function", "</function>",
		"<param", "</param>",
		"<arguments>", "</arguments>",
	} {
		if !slices.Contains(tokens, want) {
			t.Errorf("PreservedTokens() missing %q, got %v", want, tokens)
		}
	}
}

func TestMiniCPM5Parser_Registered(t *testing.T) {
	p := ParserForName("minicpm5")
	if p == nil {
		t.Fatal("ParserForName(\"minicpm5\") returned nil")
	}
	if !p.HasToolSupport() {
		t.Error("minicpm5 parser should have tool support")
	}
}
