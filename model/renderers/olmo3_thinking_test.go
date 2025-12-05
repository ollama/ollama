package renderers

import (
	"testing"

	"github.com/google/go-cmp/cmp"

	"github.com/ollama/ollama/api"
)

func TestOlmo3ThinkingRenderer(t *testing.T) {
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
				"You are OLMo, a helpful function-calling AI assistant built by Ai2. Your date cutoff is November 2024, and your model weights are available at https://huggingface.co/allenai. You do not currently have access to any functions. <functions></functions><|im_end|>\n" +
				"<|im_start|>user\n" +
				"Hello!<|im_end|>\n" +
				"<|im_start|>assistant\n" +
				"<think>",
		},
		{
			name: "with system message no tools",
			msgs: []api.Message{
				{Role: "system", Content: "You are a helpful assistant."},
				{Role: "user", Content: "Hello!"},
			},
			expected: "<|im_start|>system\n" +
				"You are a helpful assistant. You do not currently have access to any functions. <functions></functions><|im_end|>\n" +
				"<|im_start|>user\n" +
				"Hello!<|im_end|>\n" +
				"<|im_start|>assistant\n" +
				"<think>",
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
							Properties: map[string]api.ToolProperty{
								"location": {Type: api.PropertyType{"string"}, Description: "The city"},
							},
						},
					},
				},
			},
			expected: "<|im_start|>system\n" +
				`You are a helpful assistant. <functions>[{"type": "function", "function": {"name": "get_weather", "description": "Get the current weather", "parameters": {"type": "object", "required": ["location"], "properties": {"location": {"type": "string", "description": "The city"}}}}}]</functions><|im_end|>` + "\n" +
				"<|im_start|>user\n" +
				"What is the weather?<|im_end|>\n" +
				"<|im_start|>assistant\n" +
				"<think>",
		},
		{
			name: "assistant with tool calls",
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
								Arguments: map[string]any{
									"location": "San Francisco",
								},
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
							Properties: map[string]api.ToolProperty{
								"location": {Type: api.PropertyType{"string"}, Description: "The city"},
							},
						},
					},
				},
			},
			expected: "<|im_start|>system\n" +
				`You are a helpful assistant. <functions>[{"type": "function", "function": {"name": "get_weather", "description": "Get the current weather", "parameters": {"type": "object", "required": ["location"], "properties": {"location": {"type": "string", "description": "The city"}}}}}]</functions><|im_end|>` + "\n" +
				"<|im_start|>user\n" +
				"What is the weather in SF?<|im_end|>\n" +
				"<|im_start|>assistant\n" +
				`Let me check the weather.<function_calls>[{"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": "{\"location\":\"San Francisco\"}"}}]</function_calls><|im_end|>` + "\n" +
				"<|im_start|>environment\n" +
				`{"temperature": 68}<|im_end|>` + "\n" +
				"<|im_start|>assistant\n" +
				"<think>",
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
				"You are a helpful assistant. You do not currently have access to any functions. <functions></functions><|im_end|>\n" +
				"<|im_start|>user\n" +
				"Hello<|im_end|>\n" +
				"<|im_start|>assistant\n" +
				"Hi there!<|im_end|>\n" +
				"<|im_start|>user\n" +
				"How are you?<|im_end|>\n" +
				"<|im_start|>assistant\n" +
				"<think>",
		},
		{
			name: "parallel tool calls",
			msgs: []api.Message{
				{Role: "user", Content: "Get weather in SF and NYC"},
				{
					Role: "assistant",
					ToolCalls: []api.ToolCall{
						{
							ID: "call_1",
							Function: api.ToolCallFunction{
								Name:      "get_weather",
								Arguments: map[string]any{"location": "San Francisco"},
							},
						},
						{
							ID: "call_2",
							Function: api.ToolCallFunction{
								Name:      "get_weather",
								Arguments: map[string]any{"location": "New York"},
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
							Properties: map[string]api.ToolProperty{
								"location": {Type: api.PropertyType{"string"}},
							},
						},
					},
				},
			},
			expected: "<|im_start|>system\n" +
				`You are OLMo, a helpful function-calling AI assistant built by Ai2. Your date cutoff is November 2024, and your model weights are available at https://huggingface.co/allenai. <functions>[{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {"location": {"type": "string"}}}}}]</functions><|im_end|>` + "\n" +
				"<|im_start|>user\n" +
				"Get weather in SF and NYC<|im_end|>\n" +
				"<|im_start|>assistant\n" +
				`<function_calls>[{"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": "{\"location\":\"San Francisco\"}"}}, {"id": "call_2", "type": "function", "function": {"name": "get_weather", "arguments": "{\"location\":\"New York\"}"}}]</function_calls><|im_end|>` + "\n" +
				"<|im_start|>environment\n" +
				`{"temperature": 68}<|im_end|>` + "\n" +
				"<|im_start|>environment\n" +
				`{"temperature": 55}<|im_end|>` + "\n" +
				"<|im_start|>assistant\n" +
				"<think>",
		},
		{
			name: "assistant message only content no tool calls",
			msgs: []api.Message{
				{Role: "user", Content: "Tell me a joke"},
				{Role: "assistant", Content: "Why did the chicken cross the road?"},
				{Role: "user", Content: "I don't know, why?"},
			},
			expected: "<|im_start|>system\n" +
				"You are OLMo, a helpful function-calling AI assistant built by Ai2. Your date cutoff is November 2024, and your model weights are available at https://huggingface.co/allenai. You do not currently have access to any functions. <functions></functions><|im_end|>\n" +
				"<|im_start|>user\n" +
				"Tell me a joke<|im_end|>\n" +
				"<|im_start|>assistant\n" +
				"Why did the chicken cross the road?<|im_end|>\n" +
				"<|im_start|>user\n" +
				"I don't know, why?<|im_end|>\n" +
				"<|im_start|>assistant\n" +
				"<think>",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			rendered, err := (&Olmo3ThinkingRenderer{}).Render(tt.msgs, tt.tools, nil)
			if err != nil {
				t.Fatal(err)
			}
			if diff := cmp.Diff(rendered, tt.expected); diff != "" {
				t.Errorf("mismatch (-got +want):\n%s", diff)
			}
		})
	}
}
