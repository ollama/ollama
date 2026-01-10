package renderers

import (
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/ollama/ollama/api"
)

func TestQwen3VLNonThinkingRenderer(t *testing.T) {
	tests := []struct {
		name       string
		msgs       []api.Message
		images     []api.ImageData
		tools      []api.Tool
		useImgTags bool
		expected   string
	}{
		{
			name: "prefill",
			msgs: []api.Message{
				{Role: "system", Content: "You are a helpful assistant."},
				{Role: "user", Content: "Tell me something interesting."},
				{Role: "assistant", Content: "I'll tell you something interesting about cats"},
			},
			expected: `<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Tell me something interesting.<|im_end|>
<|im_start|>assistant
I'll tell you something interesting about cats`,
		},
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
`,
		},
		{
			name: "With thinking, end assistant.",
			msgs: []api.Message{
				{Role: "user", Content: "Tell me a story in two sentences."},
				{Role: "assistant", Content: "abc<think>To make this story interesting, I will speak in poetry.</think>"}, // does the thinking even work?
			},
			expected: `<|im_start|>user
Tell me a story in two sentences.<|im_end|>
<|im_start|>assistant
abc<think>To make this story interesting, I will speak in poetry.</think>`,
		},
		{
			name: "Multiple thinking",
			msgs: []api.Message{
				{Role: "user", Content: "Tell me a story in two sentences."},
				{Role: "assistant", Content: "abc<think>To make this story interesting, I will speak in poetry.</think><think>And I will speak in poetry after the first sentence.</think>"},
			},
			expected: `<|im_start|>user
Tell me a story in two sentences.<|im_end|>
<|im_start|>assistant
abc<think>To make this story interesting, I will speak in poetry.</think><think>And I will speak in poetry after the first sentence.</think>`, // NOTE: the second thinking tag is not captured
		},
		{
			name: "Multiple thinking, multiple messages.",
			msgs: []api.Message{
				{Role: "user", Content: "Tell me a story in two sentences."},
				{Role: "assistant", Content: "abc<think>To make this story interesting, I will speak in poetry.</think><think>And I will speak in poetry after the first sentence.</think>"},
				{Role: "user", Content: "What is the weather like in San Francisco? <think>I will check the weather in San Francisco for you.</think>"},
				{Role: "assistant", Content: "I'll check the weather in San Francisco for you.<think>Speak poetry after the first sentence.</think><think>Speak poetry after the second sentence.</think>"},
			},
			expected: `<|im_start|>user
Tell me a story in two sentences.<|im_end|>
<|im_start|>assistant
abc<think>To make this story interesting, I will speak in poetry.</think><think>And I will speak in poetry after the first sentence.</think><|im_end|>
<|im_start|>user
What is the weather like in San Francisco? <think>I will check the weather in San Francisco for you.</think><|im_end|>
<|im_start|>assistant
I'll check the weather in San Francisco for you.<think>Speak poetry after the first sentence.</think><think>Speak poetry after the second sentence.</think>`,
		},
		{
			name: "Image",
			msgs: []api.Message{
				{Role: "user", Content: "Describe this image.", Images: []api.ImageData{api.ImageData("img2")}},
				{Role: "assistant", Content: "Let me analyze this image."},
			},
			expected: `<|im_start|>user
<|vision_start|><|image_pad|><|vision_end|>Describe this image.<|im_end|>
<|im_start|>assistant
Let me analyze this image.`,
		},
		{
			name: "Image with image tags",
			msgs: []api.Message{
				{Role: "user", Content: "Describe this image.", Images: []api.ImageData{api.ImageData("img2")}},
				{Role: "assistant", Content: "Let me analyze this image."},
			},
			useImgTags: true,
			expected: `<|im_start|>user
[img]Describe this image.<|im_end|>
<|im_start|>assistant
Let me analyze this image.`,
		},
		{
			name: "Multiple images",
			msgs: []api.Message{
				{Role: "user", Content: "Describe these images.", Images: []api.ImageData{api.ImageData("img1"), api.ImageData("img2")}},
			},
			expected: `<|im_start|>user
<|vision_start|><|image_pad|><|vision_end|><|vision_start|><|image_pad|><|vision_end|>Describe these images.<|im_end|>
<|im_start|>assistant
`,
		},
		{
			name: "Multiple images with image tags",
			msgs: []api.Message{
				{Role: "user", Content: "Describe these images.", Images: []api.ImageData{api.ImageData("img1"), api.ImageData("img2")}},
				{Role: "assistant", Content: "Let me analyze this image."},
			},
			useImgTags: true,
			expected: `<|im_start|>user
[img][img]Describe these images.<|im_end|>
<|im_start|>assistant
Let me analyze this image.`,
		},
		// 		// NOTE: solved with #12518: https://github.com/ollama/ollama/compare/main...drifkin/stable-tool-args
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
		// 								Arguments: map[string]any{
		// 									"location": "New York",
		// 									"unit":     "fahrenheit",
		// 								},
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
		// 							Properties: map[string]api.ToolProperty{
		// 								"location": {
		// 									Type:        api.PropertyType{"string"},
		// 									Description: "The city and state, e.g. San Francisco, CA",
		// 								},
		// 								"unit": {
		// 									Type:        api.PropertyType{"string"},
		// 									Enum:        []any{"celsius", "fahrenheit"},
		// 									Description: "The temperature unit",
		// 								},
		// 							},
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
		// `,
		// 		},
		// 		// NOTE: solved with #12518: https://github.com/ollama/ollama/compare/main...drifkin/stable-tool-args
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
		// 								Arguments: map[string]any{
		// 									"a": 2,
		// 									"b": 3,
		// 								},
		// 							},
		// 						},
		// 						{
		// 							Function: api.ToolCallFunction{
		// 								Name: "multiply",
		// 								Arguments: map[string]any{
		// 									"x": 4,
		// 									"y": 5,
		// 								},
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
		// 							Properties: map[string]api.ToolProperty{
		// 								"a": {Type: api.PropertyType{"integer"}, Description: "First number"},
		// 								"b": {Type: api.PropertyType{"integer"}, Description: "Second number"},
		// 							},
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
		// 							Properties: map[string]api.ToolProperty{
		// 								"x": {Type: api.PropertyType{"integer"}, Description: "First factor"},
		// 								"y": {Type: api.PropertyType{"integer"}, Description: "Second factor"},
		// 							},
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
		// {"type": "function", "function": {"name": "multiply", "description": "Multiply two numbers", "parameters": {"type": "object", "properties": {"x": {"description": "First factor"}, "y": {"description": "Second factor"}}, "required": ["x", "y"]}}}
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
		// `,
		// 		},
		{
			name: "user tool_response block preserved",
			msgs: []api.Message{
				{Role: "user", Content: "What's the weather?"},
				{
					Role:    "assistant",
					Content: "I'll check.",
					ToolCalls: []api.ToolCall{
						{Function: api.ToolCallFunction{Name: "get-current-weather", Arguments: testArgsOrdered([]orderedArg{{"location", "Paris"}, {"unit", "celsius"}})}},
					},
				},
				{Role: "user", Content: "<tool_response>\n18\n</tool_response>"},
				{Role: "user", Content: "Thanks!"},
			},
			expected: `<|im_start|>user
What's the weather?<|im_end|>
<|im_start|>assistant
I'll check.
<tool_call>
{"name": "get-current-weather", "arguments": {"location": "Paris", "unit": "celsius"}}
</tool_call><|im_end|>
<|im_start|>user
<tool_response>
18
</tool_response><|im_end|>
<|im_start|>user
Thanks!<|im_end|>
<|im_start|>assistant
`,
		},
		{
			name: "assistant with multiple tool calls and content",
			msgs: []api.Message{
				{Role: "user", Content: "Hi"},
				{
					Role:    "assistant",
					Content: "before",
					ToolCalls: []api.ToolCall{
						{Function: api.ToolCallFunction{Name: "add", Arguments: testArgsOrdered([]orderedArg{{"a", 2}, {"b", 3}})}},
						{Function: api.ToolCallFunction{Name: "mul", Arguments: testArgsOrdered([]orderedArg{{"x", 4}, {"y", 5}})}},
					},
				},
			},
			expected: `<|im_start|>user
Hi<|im_end|>
<|im_start|>assistant
before
<tool_call>
{"name": "add", "arguments": {"a": 2, "b": 3}}
</tool_call>
<tool_call>
{"name": "mul", "arguments": {"x": 4, "y": 5}}
</tool_call>`,
		},
		{
			name: "consecutive tool responses grouped",
			msgs: []api.Message{
				{Role: "user", Content: "Compute results"},
				{Role: "assistant", Content: "ok", ToolCalls: []api.ToolCall{{Function: api.ToolCallFunction{Name: "job", Arguments: testArgs(map[string]any{"n": 1})}}}},
				{Role: "tool", Content: "5", ToolName: "job"},
				{Role: "tool", Content: "6", ToolName: "job"},
			},
			expected: `<|im_start|>user
Compute results<|im_end|>
<|im_start|>assistant
ok
<tool_call>
{"name": "job", "arguments": {"n": 1}}
</tool_call><|im_end|>
<|im_start|>user
<tool_response>
5
</tool_response>
<tool_response>
6
</tool_response><|im_end|>
<|im_start|>assistant
`,
		},
		{
			name: "last message is tool then prefill",
			msgs: []api.Message{
				{Role: "user", Content: "run"},
				{Role: "assistant", Content: "ok", ToolCalls: []api.ToolCall{{Function: api.ToolCallFunction{Name: "exec", Arguments: testArgs(map[string]any{"cmd": "ls"})}}}},
				{Role: "tool", Content: "done", ToolName: "exec"},
			},
			expected: `<|im_start|>user
run<|im_end|>
<|im_start|>assistant
ok
<tool_call>
{"name": "exec", "arguments": {"cmd": "ls"}}
</tool_call><|im_end|>
<|im_start|>user
<tool_response>
done
</tool_response><|im_end|>
<|im_start|>assistant
`,
		},
		{
			name: "user with multiple images",
			msgs: []api.Message{
				{Role: "user", Content: "Describe.", Images: []api.ImageData{api.ImageData("img1"), api.ImageData("img2")}},
			},
			expected: `<|im_start|>user
<|vision_start|><|image_pad|><|vision_end|><|vision_start|><|image_pad|><|vision_end|>Describe.<|im_end|>
<|im_start|>assistant
`,
		},
		{
			name: "user tool_response, no whitespace",
			msgs: []api.Message{
				{Role: "user", Content: "What's the weather?"},
				{
					Role:    "assistant",
					Content: "I'll check.",
					ToolCalls: []api.ToolCall{
						{Function: api.ToolCallFunction{Name: "get-current-weather", Arguments: testArgsOrdered([]orderedArg{{"location", "Paris"}, {"unit", "celsius"}})}},
					},
				},
				{Role: "user", Content: "<tool_response>\n18\n</tool_response>"},
				{Role: "user", Content: "Thanks!"},
			},
			expected: `<|im_start|>user
What's the weather?<|im_end|>
<|im_start|>assistant
I'll check.
<tool_call>
{"name": "get-current-weather", "arguments": {"location": "Paris", "unit": "celsius"}}
</tool_call><|im_end|>
<|im_start|>user
<tool_response>
18
</tool_response><|im_end|>
<|im_start|>user
Thanks!<|im_end|>
<|im_start|>assistant
`,
		},
		{
			name: "user tool_response with surrounding whitespace",
			msgs: []api.Message{
				{Role: "user", Content: "What's the weather?"},
				{
					Role:    "assistant",
					Content: "I'll check.",
					ToolCalls: []api.ToolCall{
						{Function: api.ToolCallFunction{Name: "get-current-weather", Arguments: testArgsOrdered([]orderedArg{{"location", "Paris"}, {"unit", "celsius"}})}},
					},
				},
				{Role: "user", Content: "\n\n\n\n<tool_response>\n18\n</tool_response> extra\n\n\n\n\n\n"},
			},
			expected: `<|im_start|>user
What's the weather?<|im_end|>
<|im_start|>assistant
I'll check.
<tool_call>
{"name": "get-current-weather", "arguments": {"location": "Paris", "unit": "celsius"}}
</tool_call><|im_end|>
<|im_start|>user




<tool_response>
18
</tool_response> extra





<|im_end|>
<|im_start|>assistant
`,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			rendered, err := (&Qwen3VLRenderer{isThinking: false, useImgTags: tt.useImgTags}).Render(tt.msgs, tt.tools, nil)
			if err != nil {
				t.Fatal(err)
			}
			if diff := cmp.Diff(rendered, tt.expected); diff != "" {
				t.Errorf("mismatch (-got +want):\n%s", diff)
			}
		})
	}
}
