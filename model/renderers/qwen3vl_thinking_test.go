package renderers

import (
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/ollama/ollama/api"
)

func TestQwen3VLThinkingRenderer(t *testing.T) {
	tests := []struct {
		name     string
		msgs     []api.Message
		images   []api.ImageData
		tools    []api.Tool
		expected string
	}{
		{
			name: "basic",
			msgs: []api.Message{
				{Role: "system", Content: "You are a helpful assistant."},
				{Role: "user", Content: "Hello, how are you?"},
			},
			expected: `<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Hello, how are you?<|im_end|>
<|im_start|>assistant
<think>
`,
		},
		{
			name: "With thinking, end assistant.",
			msgs: []api.Message{
				{Role: "user", Content: "Tell me a story in two sentences."},
				{Role: "assistant", Content: "abc", Thinking: "To make this story interesting, I will speak in poetry."},
			},
			expected: `<|im_start|>user
Tell me a story in two sentences.<|im_end|>
<|im_start|>assistant
<think>
To make this story interesting, I will speak in poetry.
</think>

abc`,
		},
		{
			name: "With thinking, end assistant.",
			msgs: []api.Message{
				{Role: "user", Content: "Tell me a story in two sentences."},
				{Role: "assistant", Thinking: "To make this story interesting, I will speak in poetry."},
			},
			expected: `<|im_start|>user
Tell me a story in two sentences.<|im_end|>
<|im_start|>assistant
<think>
To make this story interesting, I will speak in poetry.`,
		},
		{
			name: "Multiple thinking",
			msgs: []api.Message{
				{Role: "user", Content: "Tell me a story in two sentences."},
				{Role: "assistant", Content: "abc", Thinking: "To make this story interesting, I will speak in poetry.<think>And I will speak in poetry after the first sentence.</think>"},
			},
			expected: `<|im_start|>user
Tell me a story in two sentences.<|im_end|>
<|im_start|>assistant
<think>
To make this story interesting, I will speak in poetry.<think>And I will speak in poetry after the first sentence.</think>
</think>

abc`, // NOTE: the second thinking tag is not captured
		},
		{
			name: "Multiple thinking, multiple messages.",
			msgs: []api.Message{
				{Role: "user", Content: "Tell me a story in two sentences."},
				{Role: "assistant", Thinking: "To make this story interesting, I will speak in poetry.", Content: "abc"},
				{Role: "user", Content: "What is the weather like in San Francisco?"},
				{Role: "assistant", Thinking: "Speak poetry after the first sentence.</think><think>Speak poetry after the second sentence."},
			},
			expected: `<|im_start|>user
Tell me a story in two sentences.<|im_end|>
<|im_start|>assistant
abc<|im_end|>
<|im_start|>user
What is the weather like in San Francisco?<|im_end|>
<|im_start|>assistant
<think>
Speak poetry after the first sentence.</think><think>Speak poetry after the second sentence.`,
		},
		// NOTE: Servers automatically prepend a [img-<n>] tag
		// 		{
		// 			name: "Image",
		// 			msgs: []api.Message{
		// 				{Role: "user", Content: "Describe this image.", Images: []api.ImageData{api.ImageData(IMAGE2_BASE64)}},
		// 			},
		// 			expected: `<|im_start|>user
		// [img-0]Describe this image.<|im_end|>
		// <|im_start|>assistant
		// <think>
		// `,
		// 		},

		// NOTE: Servers automatically prepend a [img-<n>] tag
		// 		{
		// 			name: "Multiple images",
		// 			msgs: []api.Message{
		// 				{Role: "user", Content: "Describe these images.", Images: []api.ImageData{api.ImageData(IMAGE1_BASE64), api.ImageData(IMAGE2_BASE64)}},
		// 			},
		// 			expected: `<|im_start|>user
		// [img-0][img-1]Describe these images.<|im_end|>
		// <|im_start|>assistant
		// <think>
		// `,
		// 		},

		// NOTE: solved with #12518: https://github.com/ollama/ollama/compare/main...drifkin/stable-tool-args
		// 		{
		// 			name: "with tools and response",
		// 			msgs: []api.Message{
		// 				{Role: "system", Content: "You are a helpful assistant with access to tools."},
		// 				{Role: "user", Content: "What's the weather like in New York?"},
		// 				{
		// 					Role:    "assistant",
		// 					Content: "I'll check the weather in New York for you.",
		// 					ToolCalls: []api.ToolCall{
		// 						{
		// 							Function: api.ToolCallFunction{
		// 								Name: "get-current-weather",
		// 								Arguments: testArgs(map[string]any{
		// 									"location": "New York",
		// 									"unit":     "fahrenheit",
		// 								}),
		// 							},
		// 						},
		// 					},
		// 				},
		// 				{Role: "tool", Content: "80", ToolName: "get-current-weather"},
		// 				{Role: "user", Content: "That sounds nice! What about San Francisco?"},
		// 			},
		// 			tools: []api.Tool{
		// 				{
		// 					Type: "function",
		// 					Function: api.ToolFunction{
		// 						Name:        "get-current-weather",
		// 						Description: "Get the current weather for a location",
		// 						Parameters: api.ToolFunctionParameters{
		// 							Type:     "object",
		// 							Required: []string{"location"},
		// 							Properties: testPropsMap(map[string]api.ToolProperty{
		// 								"location": {
		// 									Type:        api.PropertyType{"string"},
		// 									Description: "The city and state, e.g. San Francisco, CA",
		// 								},
		// 								"unit": {
		// 									Type:        api.PropertyType{"string"},
		// 									Enum:        []any{"celsius", "fahrenheit"},
		// 									Description: "The temperature unit",
		// 								},
		// 							}),
		// 						},
		// 					},
		// 				},
		// 			},
		// 			expected: `<|im_start|>system
		// You are a helpful assistant with access to tools.

		// # Tools

		// You may call one or more functions to assist with the user query.

		// You are provided with function signatures within <tools></tools> XML tags:
		// <tools>
		// {"type": "function", "function": {"name": "get-current-weather", "description": "Get the current weather for a location", "parameters": {"type": "object", "properties": {"location": {"type": "string", "description": "The city and state, e.g. San Francisco, CA"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "The temperature unit"}}, "required": ["location"]}}}
		// </tools>

		// For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
		// <tool_call>
		// {"name": <function-name>, "arguments": <args-json-object>}
		// </tool_call><|im_end|>
		// <|im_start|>user
		// What's the weather like in New York?<|im_end|>
		// <|im_start|>assistant
		// I'll check the weather in New York for you.
		// <tool_call>
		// {"name": "get-current-weather", "arguments": {"location": "New York", "unit": "fahrenheit"}}
		// </tool_call><|im_end|>
		// <|im_start|>user
		// <tool_response>
		// 80
		// </tool_response><|im_end|>
		// <|im_start|>user
		// That sounds nice! What about San Francisco?<|im_end|>
		// <|im_start|>assistant
		// <think>
		// `,
		// 		},

		// NOTE: solved with #12518: https://github.com/ollama/ollama/compare/main...drifkin/stable-tool-args
		// 		{
		// 			name: "With tools and response, multiple tool calls",
		// 			msgs: []api.Message{
		// 				{
		// 					Role:    "system",
		// 					Content: "You are a helpful assistant with access to tools.",
		// 				},
		// 				{
		// 					Role:    "user",
		// 					Content: "Call two tools for me: add and multiply.",
		// 				},
		// 				{
		// 					Role:    "assistant",
		// 					Content: "Sure, I'll call both tools for you.",
		// 					ToolCalls: []api.ToolCall{
		// 						{
		// 							Function: api.ToolCallFunction{
		// 								Name: "add",
		// 								Arguments: testArgs(map[string]any{
		// 									"a": 2,
		// 									"b": 3,
		// 								}),
		// 							},
		// 						},
		// 						{
		// 							Function: api.ToolCallFunction{
		// 								Name: "multiply",
		// 								Arguments: testArgs(map[string]any{
		// 									"x": 4,
		// 									"y": 5,
		// 								}),
		// 							},
		// 						},
		// 					},
		// 				},
		// 				{
		// 					Role:     "tool",
		// 					Content:  "5",
		// 					ToolName: "add",
		// 				},
		// 				{
		// 					Role:     "tool",
		// 					Content:  "20",
		// 					ToolName: "multiply",
		// 				},
		// 				{
		// 					Role:    "user",
		// 					Content: "Thanks! What are the results?",
		// 				},
		// 			},
		// 			tools: []api.Tool{
		// 				{
		// 					Type: "function",
		// 					Function: api.ToolFunction{
		// 						Name:        "add",
		// 						Description: "Add two numbers",
		// 						Parameters: api.ToolFunctionParameters{
		// 							Type:     "object",
		// 							Required: []string{"a", "b"},
		// 							Properties: testPropsMap(map[string]api.ToolProperty{
		// 								"a": {Type: api.PropertyType{"integer"}, Description: "First number"},
		// 								"b": {Type: api.PropertyType{"integer"}, Description: "Second number"},
		// 							}),
		// 						},
		// 					},
		// 				},
		// 				{
		// 					Type: "function",
		// 					Function: api.ToolFunction{
		// 						Name:        "multiply",
		// 						Description: "Multiply two numbers",
		// 						Parameters: api.ToolFunctionParameters{
		// 							Type:     "object",
		// 							Required: []string{"x", "y"},
		// 							Properties: testPropsMap(map[string]api.ToolProperty{
		// 								"x": {Type: api.PropertyType{"integer"}, Description: "First factor"},
		// 								"y": {Type: api.PropertyType{"integer"}, Description: "Second factor"},
		// 							}),
		// 						},
		// 					},
		// 				},
		// 			},
		// 			expected: `<|im_start|>system
		// You are a helpful assistant with access to tools.

		// # Tools

		// You may call one or more functions to assist with the user query.

		// You are provided with function signatures within <tools></tools> XML tags:
		// <tools>
		// {"type": "function", "function": {"name": "add", "description": "Add two numbers", "parameters": {"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}, "required": ["a", "b"]}}}
		// {"type": "function", "function": {"name": "multiply", "description": "Multiply two numbers", "parameters": {"type": "object", "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}}, "required": ["x", "y"]}}}
		// </tools>

		// For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
		// <tool_call>
		// {"name": <function-name>, "arguments": <args-json-object>}
		// </tool_call><|im_end|>
		// <|im_start|>user
		// Call two tools for me: add and multiply.<|im_end|>
		// <|im_start|>assistant
		// Sure, I'll call both tools for you.
		// <tool_call>
		// {"name": "add", "arguments": {"a": 2, "b": 3}}
		// </tool_call>
		// <tool_call>
		// {"name": "multiply", "arguments": {"x": 4, "y": 5}}
		// </tool_call><|im_end|>
		// <|im_start|>user
		// <tool_response>
		// 5
		// </tool_response>
		// <tool_response>
		// 20
		// </tool_response><|im_end|>
		// <|im_start|>user
		// Thanks! What are the results?<|im_end|>
		// <|im_start|>assistant
		// <think>
		// `,
		// 		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			rendered, err := (&Qwen3VLRenderer{isThinking: true}).Render(tt.msgs, tt.tools, nil)
			if err != nil {
				t.Fatal(err)
			}
			if diff := cmp.Diff(rendered, tt.expected); diff != "" {
				t.Errorf("mismatch (-got +want):\n%s", diff)
			}
		})
	}
}

func TestFormatToolCallArgumentThinkingVL(t *testing.T) {
	tests := []struct {
		name     string
		arg      any
		expected string
	}{
		{
			name:     "string",
			arg:      "foo",
			expected: "foo",
		},
		{
			name:     "map",
			arg:      map[string]any{"foo": "bar"},
			expected: "{\"foo\":\"bar\"}",
		},
		{
			name:     "number",
			arg:      1,
			expected: "1",
		},
		{
			name:     "boolean",
			arg:      true,
			expected: "true",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := formatToolCallArgument(tt.arg)
			if got != tt.expected {
				t.Errorf("formatToolCallArgument(%v) = %v, want %v", tt.arg, got, tt.expected)
			}
		})
	}
}
