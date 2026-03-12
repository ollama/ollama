package renderers

import (
	"testing"

	"github.com/google/go-cmp/cmp"

	"github.com/ollama/ollama/api"
)

func TestOlmo3ThinkRenderer(t *testing.T) {
	tests := []struct {
		name     string
		variant  Olmo3ThinkVariant
		msgs     []api.Message
		tools    []api.Tool
		expected string
	}{
		{
			name:    "7b_basic_without_system",
			variant: Olmo31Think,
			msgs: []api.Message{
				{Role: "user", Content: "Hello!"},
			},
			expected: "<|im_start|>system\n" +
				"You are Olmo, a helpful AI assistant built by Ai2. Your date cutoff is December 2024, and your model weights are available at https://huggingface.co/allenai.<|im_end|>\n" +
				"<|im_start|>user\n" +
				"Hello!<|im_end|>\n" +
				"<|im_start|>assistant\n" +
				"<think>",
		},
		{
			name:    "7b_with_custom_system",
			variant: Olmo31Think,
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
			name:    "7b_tools_ignored",
			variant: Olmo31Think,
			msgs: []api.Message{
				{Role: "user", Content: "What is the weather?"},
			},
			tools: []api.Tool{
				{
					Type: "function",
					Function: api.ToolFunction{
						Name:        "get_weather",
						Description: "Get the current weather",
					},
				},
			},
			expected: "<|im_start|>system\n" +
				"You are Olmo, a helpful AI assistant built by Ai2. Your date cutoff is December 2024, and your model weights are available at https://huggingface.co/allenai.<|im_end|>\n" +
				"<|im_start|>user\n" +
				"What is the weather?<|im_end|>\n" +
				"<|im_start|>assistant\n" +
				"<think>",
		},
		{
			name:    "7b_tool_calls_and_tool_messages_ignored",
			variant: Olmo31Think,
			msgs: []api.Message{
				{Role: "user", Content: "What is the weather in SF?"},
				{
					Role:    "assistant",
					Content: "Let me check the weather.",
					ToolCalls: []api.ToolCall{
						{
							ID: "call_1",
							Function: api.ToolCallFunction{
								Name:      "get_weather",
								Arguments: testArgs(map[string]any{"location": "San Francisco"}),
							},
						},
					},
				},
				{Role: "tool", Content: `{"temperature": 68}`},
			},
			expected: "<|im_start|>system\n" +
				"You are Olmo, a helpful AI assistant built by Ai2. Your date cutoff is December 2024, and your model weights are available at https://huggingface.co/allenai.<|im_end|>\n" +
				"<|im_start|>user\n" +
				"What is the weather in SF?<|im_end|>\n" +
				"<|im_start|>assistant\n" +
				"Let me check the weather.<|im_end|>\n" +
				"<|im_start|>assistant\n" +
				"<think>",
		},
		{
			name:    "7b_multi_turn_conversation",
			variant: Olmo31Think,
			msgs: []api.Message{
				{Role: "user", Content: "Hello"},
				{Role: "assistant", Content: "Hi there!"},
				{Role: "user", Content: "How are you?"},
			},
			expected: "<|im_start|>system\n" +
				"You are Olmo, a helpful AI assistant built by Ai2. Your date cutoff is December 2024, and your model weights are available at https://huggingface.co/allenai.<|im_end|>\n" +
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
			name:    "32b_basic_without_system",
			variant: Olmo3Think32B,
			msgs: []api.Message{
				{Role: "user", Content: "Hello!"},
			},
			expected: "<|im_start|>system\n" +
				"You are a helpful AI assistant.<|im_end|>\n" +
				"<|im_start|>user\n" +
				"Hello!<|im_end|>\n" +
				"<|im_start|>assistant\n" +
				"<think>",
		},
		{
			name:    "32b_with_custom_system_gets_suffix",
			variant: Olmo3Think32B,
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
			name:    "31_basic_without_system",
			variant: Olmo31Think,
			msgs: []api.Message{
				{Role: "user", Content: "Hello!"},
			},
			expected: "<|im_start|>system\n" +
				"You are Olmo, a helpful AI assistant built by Ai2. Your date cutoff is December 2024, and your model weights are available at https://huggingface.co/allenai.<|im_end|>\n" +
				"<|im_start|>user\n" +
				"Hello!<|im_end|>\n" +
				"<|im_start|>assistant\n" +
				"<think>",
		},
		{
			name:    "31_with_custom_system_gets_suffix",
			variant: Olmo31Think,
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
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			rendered, err := (&Olmo3ThinkRenderer{Variant: tt.variant}).Render(tt.msgs, tt.tools, nil)
			if err != nil {
				t.Fatal(err)
			}
			if diff := cmp.Diff(rendered, tt.expected); diff != "" {
				t.Errorf("mismatch (-got +want):\n%s", diff)
			}
		})
	}
}
