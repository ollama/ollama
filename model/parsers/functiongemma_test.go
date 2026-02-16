package parsers

import (
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/ollama/ollama/api"
	"github.com/stretchr/testify/assert"
)

func TestFunctionGemmaParser(t *testing.T) {
	tests := []struct {
		name          string
		chunks        []string
		tools         []api.Tool
		expectedCalls []api.ToolCall
		expectedText  string
	}{
		{
			name:          "plain_content",
			chunks:        []string{"H", "e", "l", "l", "o", ",", " ", "w", "o", "r", "l", "d", "!"},
			expectedCalls: nil,
			expectedText:  "Hello, world!",
		},
		{
			name: "simple_tool_call",
			chunks: []string{
				"<", "start", "_", "function", "_", "call", ">",
				"call", ":", "get", "_", "weather", "{",
				"city", ":", "<", "escape", ">", "Paris", "<", "escape", ">",
				"}", "<", "end", "_", "function", "_", "call", ">",
			},
			tools: []api.Tool{
				{
					Type: "function",
					Function: api.ToolFunction{
						Name: "get_weather",
						Parameters: api.ToolFunctionParameters{
							Type: "object",
							Properties: testPropsMap(map[string]api.ToolProperty{
								"city": {Type: api.PropertyType{"string"}},
							}),
						},
					},
				},
			},
			expectedCalls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name:      "get_weather",
						Arguments: testArgs(map[string]any{"city": "Paris"}),
					},
				},
			},
			expectedText: "",
		},
		{
			name: "content_before_tool_call",
			chunks: []string{
				"L", "et", " ", "me", " ", "check", ".",
				"<", "start", "_", "function", "_", "call", ">",
				"call", ":", "get", "_", "weather", "{",
				"city", ":", "<", "escape", ">", "Paris", "<", "escape", ">",
				"}", "<", "end", "_", "function", "_", "call", ">",
			},
			expectedCalls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name:      "get_weather",
						Arguments: testArgs(map[string]any{"city": "Paris"}),
					},
				},
			},
			expectedText: "Let me check.",
		},
		{
			name: "numeric_arguments",
			chunks: []string{
				"<", "start", "_", "function", "_", "call", ">",
				"call", ":", "add", "{",
				"a", ":", "1", ",", "b", ":", "2",
				"}", "<", "end", "_", "function", "_", "call", ">",
			},
			expectedCalls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name:      "add",
						Arguments: testArgs(map[string]any{"a": int64(1), "b": int64(2)}),
					},
				},
			},
			expectedText: "",
		},
		{
			name: "boolean_arguments",
			chunks: []string{
				"<", "start", "_", "function", "_", "call", ">",
				"call", ":", "set", "_", "flag", "{",
				"enabled", ":", "true", ",", "verbose", ":", "false",
				"}", "<", "end", "_", "function", "_", "call", ">",
			},
			expectedCalls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name:      "set_flag",
						Arguments: testArgs(map[string]any{"enabled": true, "verbose": false}),
					},
				},
			},
			expectedText: "",
		},
		{
			name: "multiple_tool_calls",
			chunks: []string{
				"<", "start", "_", "function", "_", "call", ">",
				"call", ":", "get", "_", "weather", "{",
				"city", ":", "<", "escape", ">", "Paris", "<", "escape", ">",
				"}", "<", "end", "_", "function", "_", "call", ">",
				"<", "start", "_", "function", "_", "call", ">",
				"call", ":", "get", "_", "weather", "{",
				"city", ":", "<", "escape", ">", "London", "<", "escape", ">",
				"}", "<", "end", "_", "function", "_", "call", ">",
			},
			expectedCalls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name:      "get_weather",
						Arguments: testArgs(map[string]any{"city": "Paris"}),
					},
				},
				{
					Function: api.ToolCallFunction{
						Name:      "get_weather",
						Arguments: testArgs(map[string]any{"city": "London"}),
					},
				},
			},
			expectedText: "",
		},
		{
			name: "array_argument",
			chunks: []string{
				"<", "start", "_", "function", "_", "call", ">",
				"call", ":", "process", "{",
				"items", ":", "[",
				"<", "escape", ">", "a", "<", "escape", ">", ",",
				"<", "escape", ">", "b", "<", "escape", ">", ",",
				"<", "escape", ">", "c", "<", "escape", ">",
				"]",
				"}", "<", "end", "_", "function", "_", "call", ">",
			},
			expectedCalls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name:      "process",
						Arguments: testArgs(map[string]any{"items": []any{"a", "b", "c"}}),
					},
				},
			},
			expectedText: "",
		},
		{
			name: "object_argument",
			chunks: []string{
				"<", "start", "_", "function", "_", "call", ">",
				"call", ":", "update", "{",
				"data", ":", "{",
				"name", ":", "<", "escape", ">", "test", "<", "escape", ">", ",",
				"value", ":", "42",
				"}",
				"}", "<", "end", "_", "function", "_", "call", ">",
			},
			expectedCalls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name: "update",
						Arguments: testArgs(map[string]any{
							"data": map[string]any{"name": "test", "value": int64(42)},
						}),
					},
				},
			},
			expectedText: "",
		},
		{
			name:          "empty_input",
			chunks:        []string{},
			expectedCalls: nil,
			expectedText:  "",
		},
		{
			name: "tool_call_with_no_arguments",
			chunks: []string{
				"<", "start", "_", "function", "_", "call", ">",
				"call", ":", "get", "_", "time", "{", "}",
				"<", "end", "_", "function", "_", "call", ">",
			},
			expectedCalls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name:      "get_time",
						Arguments: api.NewToolCallFunctionArguments(),
					},
				},
			},
			expectedText: "",
		},
		{
			name: "content_with_angle_brackets",
			chunks: []string{
				"The", " ", "result", " ", "is", " ", "a", " ", "<", "value", ">", " ", "tag",
			},
			expectedCalls: nil,
			expectedText:  "The result is a <value> tag",
		},
		{
			name: "float_argument",
			chunks: []string{
				"<", "start", "_", "function", "_", "call", ">",
				"call", ":", "set", "_", "temp", "{",
				"value", ":", "3", ".", "14",
				"}", "<", "end", "_", "function", "_", "call", ">",
			},
			expectedCalls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name:      "set_temp",
						Arguments: testArgs(map[string]any{"value": 3.14}),
					},
				},
			},
			expectedText: "",
		},
		{
			name: "content_after_tool_call",
			chunks: []string{
				"<", "start", "_", "function", "_", "call", ">",
				"call", ":", "test", "{", "}",
				"<", "end", "_", "function", "_", "call", ">",
				"Done", "!",
			},
			expectedCalls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name:      "test",
						Arguments: api.NewToolCallFunctionArguments(),
					},
				},
			},
			expectedText: "Done!",
		},
		{
			name: "unicode_content_and_arguments",
			chunks: []string{
				"こんにちは", " ",
				"<", "start", "_", "function", "_", "call", ">",
				"call", ":", "greet", "{",
				"name", ":", "<", "escape", ">", "日本語", "<", "escape", ">",
				"}", "<", "end", "_", "function", "_", "call", ">",
			},
			expectedCalls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name:      "greet",
						Arguments: testArgs(map[string]any{"name": "日本語"}),
					},
				},
			},
			expectedText: "こんにちは ",
		},
		{
			name: "multiple_params_sorted",
			chunks: []string{
				"<", "start", "_", "function", "_", "call", ">",
				"call", ":", "search", "{",
				"query", ":", "<", "escape", ">", "test", "<", "escape", ">", ",",
				"limit", ":", "10", ",",
				"offset", ":", "0",
				"}", "<", "end", "_", "function", "_", "call", ">",
			},
			expectedCalls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name: "search",
						Arguments: testArgs(map[string]any{
							"query":  "test",
							"limit":  int64(10),
							"offset": int64(0),
						}),
					},
				},
			},
			expectedText: "",
		},
		{
			name: "nested_object_argument",
			chunks: []string{
				"<", "start", "_", "function", "_", "call", ">",
				"call", ":", "create", "{",
				"config", ":", "{",
				"settings", ":", "{",
				"enabled", ":", "true", ",",
				"name", ":", "<", "escape", ">", "test", "<", "escape", ">",
				"}",
				"}",
				"}", "<", "end", "_", "function", "_", "call", ">",
			},
			expectedCalls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name: "create",
						Arguments: testArgs(map[string]any{
							"config": map[string]any{
								"settings": map[string]any{
									"enabled": true,
									"name":    "test",
								},
							},
						}),
					},
				},
			},
			expectedText: "",
		},
		{
			name: "partial_start_tag_in_content",
			chunks: []string{
				"Hello", " ", "<", "start", " ", "world",
			},
			expectedCalls: nil,
			expectedText:  "Hello <start world",
		},
		{
			name: "parallel_tool_calls",
			chunks: []string{
				"<", "start", "_", "function", "_", "call", ">",
				"call", ":", "get", "_", "weather", "{",
				"city", ":", "<", "escape", ">", "Paris", "<", "escape", ">",
				"}", "<", "end", "_", "function", "_", "call", ">",
				"<", "start", "_", "function", "_", "call", ">",
				"call", ":", "get", "_", "time", "{",
				"timezone", ":", "<", "escape", ">", "UTC", "<", "escape", ">",
				"}", "<", "end", "_", "function", "_", "call", ">",
			},
			expectedCalls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name:      "get_weather",
						Arguments: testArgs(map[string]any{"city": "Paris"}),
					},
				},
				{
					Function: api.ToolCallFunction{
						Name:      "get_time",
						Arguments: testArgs(map[string]any{"timezone": "UTC"}),
					},
				},
			},
			expectedText: "",
		},
		{
			name: "content_between_tool_calls",
			chunks: []string{
				"<", "start", "_", "function", "_", "call", ">",
				"call", ":", "first", "{", "}",
				"<", "end", "_", "function", "_", "call", ">",
				"Some", " ", "text", " ", "here",
				"<", "start", "_", "function", "_", "call", ">",
				"call", ":", "second", "{", "}",
				"<", "end", "_", "function", "_", "call", ">",
			},
			expectedCalls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name:      "first",
						Arguments: api.NewToolCallFunctionArguments(),
					},
				},
				{
					Function: api.ToolCallFunction{
						Name:      "second",
						Arguments: api.NewToolCallFunctionArguments(),
					},
				},
			},
			expectedText: "Some text here",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			parser := &FunctionGemmaParser{}
			parser.Init(tt.tools, nil, nil)

			var allContent string
			var allCalls []api.ToolCall

			for i, chunk := range tt.chunks {
				done := i == len(tt.chunks)-1
				content, _, calls, err := parser.Add(chunk, done)
				assert.NoError(t, err)
				allContent += content
				allCalls = append(allCalls, calls...)
			}

			// Handle empty chunks case
			if len(tt.chunks) == 0 {
				content, _, calls, err := parser.Add("", true)
				assert.NoError(t, err)
				allContent = content
				allCalls = calls
			}

			assert.Equal(t, tt.expectedText, allContent)
			if diff := cmp.Diff(tt.expectedCalls, allCalls, argsComparer); diff != "" {
				t.Errorf("calls mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestFunctionGemmaParser_HasSupport(t *testing.T) {
	parser := &FunctionGemmaParser{}
	assert.True(t, parser.HasToolSupport())
	assert.False(t, parser.HasThinkingSupport())
}
