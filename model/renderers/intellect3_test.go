package renderers

import (
	"testing"

	"github.com/google/go-cmp/cmp"

	"github.com/ollama/ollama/api"
)

func TestIntellect3Renderer(t *testing.T) {
	tests := []struct {
		name     string
		msgs     []api.Message
		tools    []api.Tool
		expected string
	}{
		{
			name: "basic user message",
			msgs: []api.Message{
				{Role: "user", Content: "Hello!"},
			},
			expected: "<|im_start|>user\n" +
				"Hello!<|im_end|>\n" +
				"<|im_start|>assistant\n" +
				"<think>",
		},
		{
			name: "with system message",
			msgs: []api.Message{
				{Role: "system", Content: "You are helpful."},
				{Role: "user", Content: "Hi"},
			},
			expected: "<|im_start|>system\n" +
				"You are helpful.<|im_end|>\n" +
				"<|im_start|>user\n" +
				"Hi<|im_end|>\n" +
				"<|im_start|>assistant\n" +
				"<think>",
		},
		{
			name: "multi-turn conversation",
			msgs: []api.Message{
				{Role: "user", Content: "Hello"},
				{Role: "assistant", Content: "Hi!"},
				{Role: "user", Content: "Bye"},
			},
			expected: "<|im_start|>user\n" +
				"Hello<|im_end|>\n" +
				"<|im_start|>assistant\n" +
				"Hi!<|im_end|>\n" +
				"<|im_start|>user\n" +
				"Bye<|im_end|>\n" +
				"<|im_start|>assistant\n" +
				"<think>",
		},
		{
			name: "with tools no system message",
			msgs: []api.Message{
				{Role: "user", Content: "Weather?"},
			},
			tools: []api.Tool{
				{
					Type: "function",
					Function: api.ToolFunction{
						Name:        "get_weather",
						Description: "Get weather",
						Parameters: api.ToolFunctionParameters{
							Type: "object",
							Properties: map[string]api.ToolProperty{
								"location": {Type: api.PropertyType{"string"}},
							},
						},
					},
				},
			},
			expected: "<|im_start|>system\n" +
				"You are INTELLECT-3, a helpful assistant developed by Prime Intellect, that can interact with a computer to solve tasks.\n\n" +
				"# Tools\n\n" +
				"You have access to the following functions:\n\n" +
				"<tools>\n" +
				"<function>\n" +
				"<name>get_weather</name>\n" +
				"<description>Get weather</description>\n" +
				"<parameters>\n" +
				"<parameter>\n" +
				"<name>location</name>\n" +
				"<type>string</type>\n" +
				"</parameter>\n" +
				"</parameters>\n" +
				"</function>\n" +
				"</tools>\n\n" +
				"If you choose to call a function ONLY reply in the following format with NO suffix:\n\n" +
				"<tool_call>\n" +
				"<function=example_function_name>\n" +
				"<parameter=example_parameter_1>\n" +
				"value_1\n" +
				"</parameter>\n" +
				"<parameter=example_parameter_2>\n" +
				"This is the value for the second parameter\n" +
				"that can span\n" +
				"multiple lines\n" +
				"</parameter>\n" +
				"</function>\n" +
				"</tool_call>\n\n" +
				"<IMPORTANT>\n" +
				"Reminder:\n" +
				"- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n" +
				"- Required parameters MUST be specified\n" +
				"- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n" +
				"- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n" +
				"</IMPORTANT><|im_end|>\n" +
				"<|im_start|>user\n" +
				"Weather?<|im_end|>\n" +
				"<|im_start|>assistant\n" +
				"<think>",
		},
		{
			name: "tool call and response",
			msgs: []api.Message{
				{Role: "user", Content: "Weather?"},
				{
					Role:    "assistant",
					Content: "Checking.",
					ToolCalls: []api.ToolCall{
						{
							ID: "1",
							Function: api.ToolCallFunction{
								Name:      "get_weather",
								Arguments: map[string]any{"location": "SF"},
							},
						},
					},
				},
				{Role: "tool", Content: `{"temp": 68}`, ToolCallID: "1"},
			},
			tools: []api.Tool{
				{
					Type: "function",
					Function: api.ToolFunction{
						Name: "get_weather",
						Parameters: api.ToolFunctionParameters{
							Type: "object",
							Properties: map[string]api.ToolProperty{
								"location": {Type: api.PropertyType{"string"}},
							},
						},
					},
				},
			},
			expected: "<|im_start|>system\n" +
				"You are INTELLECT-3, a helpful assistant developed by Prime Intellect, that can interact with a computer to solve tasks.\n\n" +
				"# Tools\n\n" +
				"You have access to the following functions:\n\n" +
				"<tools>\n" +
				"<function>\n" +
				"<name>get_weather</name>\n" +
				"<parameters>\n" +
				"<parameter>\n" +
				"<name>location</name>\n" +
				"<type>string</type>\n" +
				"</parameter>\n" +
				"</parameters>\n" +
				"</function>\n" +
				"</tools>\n\n" +
				"If you choose to call a function ONLY reply in the following format with NO suffix:\n\n" +
				"<tool_call>\n" +
				"<function=example_function_name>\n" +
				"<parameter=example_parameter_1>\n" +
				"value_1\n" +
				"</parameter>\n" +
				"<parameter=example_parameter_2>\n" +
				"This is the value for the second parameter\n" +
				"that can span\n" +
				"multiple lines\n" +
				"</parameter>\n" +
				"</function>\n" +
				"</tool_call>\n\n" +
				"<IMPORTANT>\n" +
				"Reminder:\n" +
				"- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n" +
				"- Required parameters MUST be specified\n" +
				"- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n" +
				"- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n" +
				"</IMPORTANT><|im_end|>\n" +
				"<|im_start|>user\n" +
				"Weather?<|im_end|>\n" +
				"<|im_start|>assistant\n" +
				"Checking.\n\n" +
				"<tool_call>\n" +
				"<function=get_weather>\n" +
				"<parameter=location>\n" +
				"SF\n" +
				"</parameter>\n" +
				"</function>\n" +
				"</tool_call><|im_end|>\n" +
				"<|im_start|>user\n" +
				"<tool_response>\n" +
				`{"temp": 68}` + "\n" +
				"</tool_response>\n" +
				"<|im_end|>\n" +
				"<|im_start|>assistant\n" +
				"<think>",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			rendered, err := (&Intellect3Renderer{}).Render(tt.msgs, tt.tools, nil)
			if err != nil {
				t.Fatal(err)
			}
			if diff := cmp.Diff(rendered, tt.expected); diff != "" {
				t.Errorf("mismatch (-got +want):\n%s", diff)
			}
		})
	}
}
