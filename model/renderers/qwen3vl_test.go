package renderers

import (
	"testing"

	"encoding/base64"

	"github.com/google/go-cmp/cmp"
	"github.com/ollama/ollama/api"
)

var IMAGE1_BASE64 = base64.StdEncoding.EncodeToString([]byte("image1"))
var IMAGE2_BASE64 = "iVBORw0KGgoAAAANSUhEUgAAAQAAAAEACAIAAADTED8xAAADMElEQVR4nOzVwQnAIBQFQYXff81RUkQCOyDj1YOPnbXWPmeTRef+/3O/OyBjzh3CD95BfqICMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMK0CMO0TAAD//2Anhf4QtqobAAAAAElFTkSuQmCC"

// TODO:
// - [ ] test videos?

func TestQwen3VLRenderer(t *testing.T) {
	tests := []struct {
		name     string
		msgs     []api.Message
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
				// {Role: "system", Content: "You are a helpful assistant."},
				{Role: "user", Content: "Tell me a story in two sentences."},
				{Role: "assistant", Content: "abc<think>To make this story interesting, I will speak in poetry.</think>"}, // does the thinking even work?
			},
			expected: `<|im_start|>user
Tell me a story in two sentences.<|im_end|>
<|im_start|>assistant
<think>
To make this story interesting, I will speak in poetry.
</think>

<|im_end|>
<|im_start|>assistant
<think>
`,
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
<think>
To make this story interesting, I will speak in poetry.
</think>

<|im_end|>
<|im_start|>assistant
<think>
`, // the second thinking tag is not captured
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
<|im_end|>
<|im_start|>user
What is the weather like in San Francisco? <think>I will check the weather in San Francisco for you.</think><|im_end|>
<|im_start|>assistant
<think>
Speak poetry after the first sentence.
</think>

<|im_end|>
<|im_start|>assistant
<think>`,
		},
		{
			name: "Image",
			msgs: []api.Message{
				{Role: "user", Content: "Describe this image.", Images: []api.ImageData{api.ImageData(IMAGE2_BASE64)}}, // does this work?
			}, // this is actually a local test, remote model may need to be different
			expected: `<|im_start|>user
[img-0]Describe this image.<|im_end|>
<|im_start|>assistant
<think>
`,
		}, // there's no way to do videos?
		{
			name: "with tools and response",
			msgs: []api.Message{
				{Role: "system", Content: "You are a helpful assistant with access to tools."},
				{Role: "user", Content: "What's the weather like in New York?"},
				{
					Role:    "assistant",
					Content: "I'll check the weather in New York for you.",
					ToolCalls: []api.ToolCall{
						{
							Function: api.ToolCallFunction{
								Name: "get-current-weather",
								Arguments: map[string]any{
									"location": "New York",
									"unit":     "fahrenheit",
								},
							},
						},
					},
				},
				{Role: "tool", Content: "80", ToolName: "get-current-weather"},
				{Role: "user", Content: "That sounds nice! What about San Francisco?"},
			},
			tools: []api.Tool{
				{
					Type: "function",
					Function: api.ToolFunction{
						Name:        "get-current-weather",
						Description: "Get the current weather for a location",
						Parameters: api.ToolFunctionParameters{
							Type:     "object",
							Required: []string{"location"},
							Properties: map[string]api.ToolProperty{
								"location": {
									Type:        api.PropertyType{"string"},
									Description: "The city and state, e.g. San Francisco, CA",
								},
								"unit": {
									Type:        api.PropertyType{"string"},
									Enum:        []any{"celsius", "fahrenheit"},
									Description: "The temperature unit",
								},
							},
						},
					},
				},
			},
			expected: `<|im_start|>system
You are a helpful assistant with access to tools.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "get-current-weather", "description": "Get the current weather for a location", "parameters": {"type": "object", "properties": {"location": {"type": "string", "description": "The city and state, e.g. San Francisco, CA"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "The temperature unit"}}, "required": ["location"]}}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call><|im_end|>
<|im_start|>user
What's the weather like in New York?<|im_end|>
<|im_start|>assistant
I'll check the weather in New York for you.
<tool_call>
{"name": "get-current-weather", "arguments": {"location": "New York", "unit": "fahrenheit"}}
</tool_call><|im_end|>
<|im_start|>user
<tool_response>
80
</tool_response><|im_end|>
<|im_start|>user
That sounds nice! What about San Francisco?<|im_end|>
<|im_start|>assistant
<think>
`,
		},
		{
			name: "With tools and response, multiple tool calls",
			msgs: []api.Message{
				{
					Role:    "system",
					Content: "You are a helpful assistant with access to tools.",
				},
				{
					Role:    "user",
					Content: "Call two tools for me: add and multiply.",
				},
				{
					Role:    "assistant",
					Content: "Sure, I'll call both tools for you.",
					ToolCalls: []api.ToolCall{
						{
							Function: api.ToolCallFunction{
								Name: "add",
								Arguments: map[string]any{
									"a": 2,
									"b": 3,
								},
							},
						},
						{
							Function: api.ToolCallFunction{
								Name: "multiply",
								Arguments: map[string]any{
									"x": 4,
									"y": 5,
								},
							},
						},
					},
				},
				{
					Role:     "tool",
					Content:  "5",
					ToolName: "add",
				},
				{
					Role:     "tool",
					Content:  "20",
					ToolName: "multiply",
				},
				{
					Role:    "user",
					Content: "Thanks! What are the results?",
				},
			},
			tools: []api.Tool{
				{
					Type: "function",
					Function: api.ToolFunction{
						Name:        "add",
						Description: "Add two numbers",
						Parameters: api.ToolFunctionParameters{
							Type:     "object",
							Required: []string{"a", "b"},
							Properties: map[string]api.ToolProperty{
								"a": {Type: api.PropertyType{"integer"}, Description: "First number"},
								"b": {Type: api.PropertyType{"integer"}, Description: "Second number"},
							},
						},
					},
				},
				{
					Type: "function",
					Function: api.ToolFunction{
						Name:        "multiply",
						Description: "Multiply two numbers",
						Parameters: api.ToolFunctionParameters{
							Type:     "object",
							Required: []string{"x", "y"},
							Properties: map[string]api.ToolProperty{
								"x": {Type: api.PropertyType{"integer"}, Description: "First factor"},
								"y": {Type: api.PropertyType{"integer"}, Description: "Second factor"},
							},
						},
					},
				},
			},
			expected: `<|im_start|>system
You are a helpful assistant with access to tools.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "add", "description": "Add two numbers", "parameters": {"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}, "required": ["a", "b"]}}}
{"type": "function", "function": {"name": "multiply", "description": "Multiply two numbers", "parameters": {"type": "object", "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}}, "required": ["x", "y"]}}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call><|im_end|>
<|im_start|>user
Call two tools for me: add and multiply.<|im_end|>
<|im_start|>assistant
Sure, I'll call both tools for you.
<tool_call>
{"name": "add", "arguments": {"a": 2, "b": 3}}
</tool_call>
<tool_call>
{"name": "multiply", "arguments": {"x": 4, "y": 5}}
</tool_call><|im_end|>
<|im_start|>user
<tool_response>
5
</tool_response>
<tool_response>
20
</tool_response><|im_end|>
<|im_start|>user
Thanks! What are the results?<|im_end|>
<|im_start|>assistant
<think>
`,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			rendered, err := Qwen3VLRenderer(tt.msgs, tt.tools, nil)
			if err != nil {
				t.Fatal(err)
			}
			if diff := cmp.Diff(rendered, tt.expected); diff != "" {
				t.Errorf("mismatch (-got +want):\n%s", diff)
			}
		})
	}
}

// what is this function for?

func TestFormatToolCallArgumentVL(t *testing.T) {
	tests := []struct {
		name     string
		arg      any
		expected string
	}{
		{
			name: "string",
			arg:  "foo",
			// notice no quotes around the string
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

// func TestQwen3VL_AssistantToolCall_NoContent(t *testing.T) {
//     msgs := []api.Message{
//         {Role: "user", Content: "What is the weather like in New York?"},
//         {Role: "assistant", Content: "", ToolCalls: []api.ToolCall{
//             {Function: api.ToolCallFunction{Name: "get_weather", Arguments: map[string]any{"location": "New York", "unit": "F"}}},
//         }},
//     }

//     got, err := Qwen3VLRenderer(msgs, nil, nil)
//     if err != nil { t.Fatal(err) }

//     // Expect assistant block with a single tool_call (no leading blank line), then prefill
//     wantContains := []string{
//         "<|im_start|>user\nWhat is the weather like in New York?<|im_end|>\n",
//         "<|im_start|>assistant\n<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"location\": \"New York\", \"unit\": \"F\"}}\n</tool_call><|im_end|>\n",
//         "<|im_start|>assistant\n<think>\n",
//     }
//     for _, sub := range wantContains {
//         if !strings.Contains(got, sub) {
//             t.Fatalf("missing substring: %q\nGot:\n%s", sub, got)
//         }
//     }
// }

// func TestQwen3VL_ToolResponses_Grouped(t *testing.T) {
//     msgs := []api.Message{
//         {Role: "user", Content: "Call two tools."},
//         {Role: "assistant", Content: "", ToolCalls: []api.ToolCall{
//             {Function: api.ToolCallFunction{Name: "tool_a", Arguments: map[string]any{"x": 1}}},
//         }},
//         {Role: "tool", Content: "result_a", ToolName: "tool_a"},
//         {Role: "tool", Content: "result_b", ToolName: "tool_b"},
//     }

//     got, err := Qwen3VLRenderer(msgs, nil, nil)
//     if err != nil { t.Fatal(err) }

//     // Expect a single user block containing two <tool_response> entries
//     want := "<|im_start|>user\n<tool_response>\nresult_a\n</tool_response>\n<tool_response>\nresult_b\n</tool_response><|im_end|>\n"
//     if !strings.Contains(got, want) {
//         t.Fatalf("grouped tool responses not found\nWant contains:\n%s\nGot:\n%s", want, got)
//     }
// }

// func TestQwen3VL_ToolsHeader_JSONFormatting(t *testing.T) {
//     tools := []api.Tool{ {
//         Function: api.ToolFunction{
//             Name:        "get_weather",
//             Description: "Get current weather",
//             Parameters: api.ToolFunctionParameters{
//                 Type:     "object",
//                 Required: []string{"location"},
//                 Properties: map[string]api.ToolProperty{
//                     "location": {Type: api.PropertyType{"string"}, Description: "city, state"},
//                 },
//             },
//         },
//         Type: "function",
//     } }

//     msgs := []api.Message{
//         {Role: "system", Content: "You are a helpful assistant."},
//         {Role: "user", Content: "Hello"},
//     }

//     got, err := Qwen3VLRenderer(msgs, tools, nil)
//     if err != nil { t.Fatal(err) }

//     // Must include a <tools> block and JSON with spaces after ':' and ',' (marshalWithSpaces)
//     if !strings.Contains(got, "<tools>") || !strings.Contains(got, "</tools>") {
//         t.Fatalf("missing <tools> block in output:\n%s", got)
//     }
//     // Heuristic: JSON pretty spacing inside
//     if !strings.Contains(got, ": ") || !strings.Contains(got, ", ") {
//         t.Fatalf("tool JSON appears not to contain spaced separators\nGot:\n%s", got)
//     }
// }

// func TestQwen3VL_ThinkExtraction_FromContent(t *testing.T) {
//     msgs := []api.Message{
//         {Role: "user", Content: "Tell me a story."},
//         {Role: "assistant", Content: "<think>Reasoning</think>Answer"},
//     }

//     got, err := Qwen3VLRenderer(msgs, nil, nil)
//     if err != nil { t.Fatal(err) }

//     // Expect reasoning moved into <think> block and remaining content after it
//     wantContains := []string{
//         "<|im_start|>assistant\n<think>\nReasoning\n</think>\n\nAnswer",
//     }
//     for _, sub := range wantContains {
//         if !strings.Contains(got, sub) {
//             t.Fatalf("missing substring: %q\nGot:\n%s", sub, got)
//         }
//     }
// }
