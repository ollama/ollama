package renderers

import (
	"testing"

	"github.com/google/go-cmp/cmp"

	"github.com/ollama/ollama/api"
)

func TestOlmo3Renderer(t *testing.T) {
	tests := []struct {
		name     string
		msgs     []api.Message
		tools    []api.Tool
		expected string
	}{
		{
			name: "basic without system - adds default system",
			msgs: []api.Message{
				{Role: "user", Content: "Hello!"},
			},
			expected: "<|im_start|>system\n" +
				"You are a helpful function-calling AI assistant. You do not currently have access to any functions. <functions></functions><|im_end|>\n" +
				"<|im_start|>user\n" +
				"Hello!<|im_end|>\n" +
				"<|im_start|>assistant\n",
		},
		{
			name: "with system message no tools",
			msgs: []api.Message{
				{Role: "system", Content: "You are a helpful assistant."},
				{Role: "user", Content: "Hello!"},
			},
			expected: "<|im_start|>system\n" +
				"You are a helpful assistant.<|im_end|>\n" +
				"<|im_start|>user\n" +
				"Hello!<|im_end|>\n" +
				"<|im_start|>assistant\n",
		},
		{
			name: "with system message and tools",
			msgs: []api.Message{
				{Role: "system", Content: "You are a helpful assistant."},
				{Role: "user", Content: "What is the weather?"},
			},
			tools: []api.Tool{
				{
					Type: "function",
					Function: api.ToolFunction{
						Name:        "get_weather",
						Description: "Get the current weather",
						Parameters: api.ToolFunctionParameters{
							Type:     "object",
							Required: []string{"location"},
							Properties: testPropsMap(map[string]api.ToolProperty{
								"location": {Type: api.PropertyType{"string"}, Description: "The city"},
							}),
						},
					},
				},
			},
			expected: "<|im_start|>system\n" +
				`You are a helpful assistant.<functions>[{"type": "function", "function": {"name": "get_weather", "description": "Get the current weather", "parameters": {"type": "object", "required": ["location"], "properties": {"location": {"type": "string", "description": "The city"}}}}}]</functions><|im_end|>` + "\n" +
				"<|im_start|>user\n" +
				"What is the weather?<|im_end|>\n" +
				"<|im_start|>assistant\n",
		},
		{
			name: "default system with tools - includes function instruction",
			msgs: []api.Message{
				{Role: "user", Content: "What is the weather?"},
			},
			tools: []api.Tool{
				{
					Type: "function",
					Function: api.ToolFunction{
						Name:        "get_weather",
						Description: "Get the current weather",
						Parameters: api.ToolFunctionParameters{
							Type:     "object",
							Required: []string{"location"},
							Properties: testPropsMap(map[string]api.ToolProperty{
								"location": {Type: api.PropertyType{"string"}, Description: "The city"},
							}),
						},
					},
				},
			},
			expected: "<|im_start|>system\n" +
				"You are a helpful function-calling AI assistant. " +
				"You are provided with function signatures within <functions></functions> XML tags. You may call one or more functions to assist with the user query. Output any function calls within <function_calls></function_calls> XML tags. Do not make assumptions about what values to plug into functions." +
				`<functions>[{"type": "function", "function": {"name": "get_weather", "description": "Get the current weather", "parameters": {"type": "object", "required": ["location"], "properties": {"location": {"type": "string", "description": "The city"}}}}}]</functions><|im_end|>` + "\n" +
				"<|im_start|>user\n" +
				"What is the weather?<|im_end|>\n" +
				"<|im_start|>assistant\n",
		},
		{
			name: "assistant with tool calls - function call syntax",
			msgs: []api.Message{
				{Role: "system", Content: "You are a helpful assistant."},
				{Role: "user", Content: "What is the weather in SF?"},
				{
					Role:    "assistant",
					Content: "Let me check the weather.",
					ToolCalls: []api.ToolCall{
						{
							ID: "call_1",
							Function: api.ToolCallFunction{
								Name: "get_weather",
								Arguments: testArgs(map[string]any{
									"location": "San Francisco",
								}),
							},
						},
					},
				},
				{Role: "tool", Content: `{"temperature": 68}`, ToolName: "get_weather"},
			},
			tools: []api.Tool{
				{
					Type: "function",
					Function: api.ToolFunction{
						Name:        "get_weather",
						Description: "Get the current weather",
						Parameters: api.ToolFunctionParameters{
							Type:     "object",
							Required: []string{"location"},
							Properties: testPropsMap(map[string]api.ToolProperty{
								"location": {Type: api.PropertyType{"string"}, Description: "The city"},
							}),
						},
					},
				},
			},
			expected: "<|im_start|>system\n" +
				`You are a helpful assistant.<functions>[{"type": "function", "function": {"name": "get_weather", "description": "Get the current weather", "parameters": {"type": "object", "required": ["location"], "properties": {"location": {"type": "string", "description": "The city"}}}}}]</functions><|im_end|>` + "\n" +
				"<|im_start|>user\n" +
				"What is the weather in SF?<|im_end|>\n" +
				"<|im_start|>assistant\n" +
				`Let me check the weather.<function_calls>get_weather(location="San Francisco")</function_calls><|im_end|>` + "\n" +
				"<|im_start|>environment\n" +
				`{"temperature": 68}<|im_end|>` + "\n" +
				"<|im_start|>assistant\n",
		},
		{
			name: "multi-turn conversation",
			msgs: []api.Message{
				{Role: "system", Content: "You are a helpful assistant."},
				{Role: "user", Content: "Hello"},
				{Role: "assistant", Content: "Hi there!"},
				{Role: "user", Content: "How are you?"},
			},
			expected: "<|im_start|>system\n" +
				"You are a helpful assistant.<|im_end|>\n" +
				"<|im_start|>user\n" +
				"Hello<|im_end|>\n" +
				"<|im_start|>assistant\n" +
				"Hi there!<|im_end|>\n" +
				"<|im_start|>user\n" +
				"How are you?<|im_end|>\n" +
				"<|im_start|>assistant\n",
		},
		{
			name: "parallel tool calls - newline separated",
			msgs: []api.Message{
				{Role: "user", Content: "Get weather in SF and NYC"},
				{
					Role: "assistant",
					ToolCalls: []api.ToolCall{
						{
							ID: "call_1",
							Function: api.ToolCallFunction{
								Name:      "get_weather",
								Arguments: testArgs(map[string]any{"location": "San Francisco"}),
							},
						},
						{
							ID: "call_2",
							Function: api.ToolCallFunction{
								Name:      "get_weather",
								Arguments: testArgs(map[string]any{"location": "New York"}),
							},
						},
					},
				},
				{Role: "tool", Content: `{"temperature": 68}`, ToolName: "get_weather"},
				{Role: "tool", Content: `{"temperature": 55}`, ToolName: "get_weather"},
			},
			tools: []api.Tool{
				{
					Type: "function",
					Function: api.ToolFunction{
						Name: "get_weather",
						Parameters: api.ToolFunctionParameters{
							Type: "object",
							Properties: testPropsMap(map[string]api.ToolProperty{
								"location": {Type: api.PropertyType{"string"}},
							}),
						},
					},
				},
			},
			expected: "<|im_start|>system\n" +
				"You are a helpful function-calling AI assistant. " +
				"You are provided with function signatures within <functions></functions> XML tags. You may call one or more functions to assist with the user query. Output any function calls within <function_calls></function_calls> XML tags. Do not make assumptions about what values to plug into functions." +
				`<functions>[{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {"location": {"type": "string"}}}}}]</functions><|im_end|>` + "\n" +
				"<|im_start|>user\n" +
				"Get weather in SF and NYC<|im_end|>\n" +
				"<|im_start|>assistant\n" +
				`<function_calls>get_weather(location="San Francisco")` + "\n" +
				`get_weather(location="New York")</function_calls><|im_end|>` + "\n" +
				"<|im_start|>environment\n" +
				`{"temperature": 68}<|im_end|>` + "\n" +
				"<|im_start|>environment\n" +
				`{"temperature": 55}<|im_end|>` + "\n" +
				"<|im_start|>assistant\n",
		},
		{
			name: "tool call with multiple arguments",
			msgs: []api.Message{
				{Role: "user", Content: "Book a flight"},
				{
					Role: "assistant",
					ToolCalls: []api.ToolCall{
						{
							ID: "call_1",
							Function: api.ToolCallFunction{
								Name: "book_flight",
								Arguments: testArgsOrdered([]orderedArg{
									{"from", "SFO"},
									{"to", "NYC"},
								}),
							},
						},
					},
				},
			},
			tools: []api.Tool{
				{
					Type: "function",
					Function: api.ToolFunction{
						Name: "book_flight",
						Parameters: api.ToolFunctionParameters{
							Type: "object",
							Properties: testPropsOrdered([]orderedProp{
								{"from", api.ToolProperty{Type: api.PropertyType{"string"}}},
								{"to", api.ToolProperty{Type: api.PropertyType{"string"}}},
							}),
						},
					},
				},
			},
			expected: "<|im_start|>system\n" +
				"You are a helpful function-calling AI assistant. " +
				"You are provided with function signatures within <functions></functions> XML tags. You may call one or more functions to assist with the user query. Output any function calls within <function_calls></function_calls> XML tags. Do not make assumptions about what values to plug into functions." +
				`<functions>[{"type": "function", "function": {"name": "book_flight", "parameters": {"type": "object", "properties": {"from": {"type": "string"}, "to": {"type": "string"}}}}}]</functions><|im_end|>` + "\n" +
				"<|im_start|>user\n" +
				"Book a flight<|im_end|>\n" +
				"<|im_start|>assistant\n" +
				`<function_calls>book_flight(from="SFO", to="NYC")</function_calls><|im_end|>` + "\n" +
				"<|im_start|>assistant\n",
		},
		{
			name: "assistant prefill - no generation prompt",
			msgs: []api.Message{
				{Role: "user", Content: "Hello"},
				{Role: "assistant", Content: "Hi there!"},
			},
			expected: "<|im_start|>system\n" +
				"You are a helpful function-calling AI assistant. You do not currently have access to any functions. <functions></functions><|im_end|>\n" +
				"<|im_start|>user\n" +
				"Hello<|im_end|>\n" +
				"<|im_start|>assistant\n" +
				"Hi there!",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			rendered, err := (&Olmo3Renderer{}).Render(tt.msgs, tt.tools, nil)
			if err != nil {
				t.Fatal(err)
			}
			if diff := cmp.Diff(rendered, tt.expected); diff != "" {
				t.Errorf("mismatch (-got +want):\n%s", diff)
			}
		})
	}
}
