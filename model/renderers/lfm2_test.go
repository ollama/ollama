package renderers

import (
	"testing"

	"github.com/google/go-cmp/cmp"

	"github.com/ollama/ollama/api"
)

func TestLFM2Renderer(t *testing.T) {
	tests := []struct {
		name       string
		messages   []api.Message
		tools      []api.Tool
		thinkValue *api.ThinkValue
		expected   string
	}{
		{
			name: "basic user message",
			messages: []api.Message{
				{Role: "user", Content: "Hello!"},
			},
			thinkValue: &api.ThinkValue{Value: false},
			expected:   "<|im_start|>user\nHello!<|im_end|>\n<|im_start|>assistant\n",
		},
		{
			name: "basic with system message",
			messages: []api.Message{
				{Role: "system", Content: "You are a helpful assistant."},
				{Role: "user", Content: "Hello!"},
			},
			thinkValue: &api.ThinkValue{Value: false},
			expected:   "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nHello!<|im_end|>\n<|im_start|>assistant\n",
		},
		{
			name: "multiple system messages rendered separately",
			messages: []api.Message{
				{Role: "system", Content: "First instruction."},
				{Role: "system", Content: "Second instruction."},
				{Role: "user", Content: "Hello!"},
			},
			thinkValue: &api.ThinkValue{Value: false},
			expected:   "<|im_start|>system\nFirst instruction.<|im_end|>\n<|im_start|>system\nSecond instruction.<|im_end|>\n<|im_start|>user\nHello!<|im_end|>\n<|im_start|>assistant\n",
		},
		{
			name: "multi-turn conversation",
			messages: []api.Message{
				{Role: "user", Content: "What is 2+2?"},
				{Role: "assistant", Content: "The answer is 4."},
				{Role: "user", Content: "Thanks!"},
			},
			thinkValue: &api.ThinkValue{Value: false},
			expected:   "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\nThe answer is 4.<|im_end|>\n<|im_start|>user\nThanks!<|im_end|>\n<|im_start|>assistant\n",
		},
		{
			name: "only system message",
			messages: []api.Message{
				{Role: "system", Content: "You are helpful."},
			},
			thinkValue: &api.ThinkValue{Value: false},
			expected:   "<|im_start|>system\nYou are helpful.<|im_end|>\n<|im_start|>assistant\n",
		},
		{
			// When assistant is the LAST assistant, thinking is preserved (even with keep_past_thinking=false)
			name: "user-assistant-user: last assistant preserves thinking",
			messages: []api.Message{
				{Role: "user", Content: "Q1"},
				{Role: "assistant", Content: "<think>reasoning</think>A1"},
				{Role: "user", Content: "Q2"},
			},
			thinkValue: &api.ThinkValue{Value: false},
			expected:   "<|im_start|>user\nQ1<|im_end|>\n<|im_start|>assistant\n<think>reasoning</think>A1<|im_end|>\n<|im_start|>user\nQ2<|im_end|>\n<|im_start|>assistant\n",
		},
		{
			// With two assistants, first is stripped (not last), second preserved (is last)
			name: "multi-turn thinking: first stripped, second preserved",
			messages: []api.Message{
				{Role: "user", Content: "Q1"},
				{Role: "assistant", Content: "<think>reason1</think>A1"},
				{Role: "user", Content: "Q2"},
				{Role: "assistant", Content: "<think>reason2</think>A2"},
			},
			thinkValue: &api.ThinkValue{Value: false},
			expected:   "<|im_start|>user\nQ1<|im_end|>\n<|im_start|>assistant\nA1<|im_end|>\n<|im_start|>user\nQ2<|im_end|>\n<|im_start|>assistant\n<think>reason2</think>A2<|im_end|>\n<|im_start|>assistant\n",
		},
		{
			// With thinking enabled (keep_past_thinking=true), both preserved
			name: "multi-turn thinking: both preserved when thinking enabled",
			messages: []api.Message{
				{Role: "user", Content: "Q1"},
				{Role: "assistant", Content: "<think>reason1</think>A1"},
				{Role: "user", Content: "Q2"},
				{Role: "assistant", Content: "<think>reason2</think>A2"},
			},
			thinkValue: &api.ThinkValue{Value: true},
			expected:   "<|im_start|>user\nQ1<|im_end|>\n<|im_start|>assistant\n<think>reason1</think>A1<|im_end|>\n<|im_start|>user\nQ2<|im_end|>\n<|im_start|>assistant\n<think>reason2</think>A2<|im_end|>\n<|im_start|>assistant\n",
		},
		{
			name: "assistant with tool calls",
			messages: []api.Message{
				{Role: "user", Content: "What's the weather?"},
				{
					Role: "assistant",
					ToolCalls: []api.ToolCall{
						{
							Function: api.ToolCallFunction{
								Name: "get_weather",
								Arguments: testArgs(map[string]any{
									"location": "Paris",
								}),
							},
						},
					},
				},
			},
			thinkValue: &api.ThinkValue{Value: false},
			expected:   `<|im_start|>user` + "\n" + `What's the weather?<|im_end|>` + "\n" + `<|im_start|>assistant` + "\n" + `<|tool_call_start|>{"arguments":{"location":"Paris"},"name":"get_weather"}<|tool_call_end|><|im_end|>` + "\n" + `<|im_start|>assistant` + "\n",
		},
		{
			name: "assistant with content and tool calls",
			messages: []api.Message{
				{Role: "user", Content: "What's the weather in Paris?"},
				{
					Role:    "assistant",
					Content: "Let me check.",
					ToolCalls: []api.ToolCall{
						{
							Function: api.ToolCallFunction{
								Name: "get_weather",
								Arguments: testArgs(map[string]any{
									"location": "Paris",
								}),
							},
						},
					},
				},
			},
			thinkValue: &api.ThinkValue{Value: false},
			expected:   `<|im_start|>user` + "\n" + `What's the weather in Paris?<|im_end|>` + "\n" + `<|im_start|>assistant` + "\n" + `Let me check.<|tool_call_start|>{"arguments":{"location":"Paris"},"name":"get_weather"}<|tool_call_end|><|im_end|>` + "\n" + `<|im_start|>assistant` + "\n",
		},
		{
			name: "tool response",
			messages: []api.Message{
				{Role: "user", Content: "What's the weather?"},
				{Role: "assistant", Content: "Let me check."},
				{Role: "tool", Content: "22C, Sunny"},
			},
			thinkValue: &api.ThinkValue{Value: false},
			expected:   "<|im_start|>user\nWhat's the weather?<|im_end|>\n<|im_start|>assistant\nLet me check.<|im_end|>\n<|im_start|>tool\n22C, Sunny<|im_end|>\n<|im_start|>assistant\n",
		},
		{
			name: "multiple tool calls",
			messages: []api.Message{
				{Role: "user", Content: "Get weather for Paris and London"},
				{
					Role: "assistant",
					ToolCalls: []api.ToolCall{
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
				},
			},
			thinkValue: &api.ThinkValue{Value: false},
			expected:   `<|im_start|>user` + "\n" + `Get weather for Paris and London<|im_end|>` + "\n" + `<|im_start|>assistant` + "\n" + `<|tool_call_start|>{"arguments":{"location":"Paris"},"name":"get_weather"}<|tool_call_end|><|tool_call_start|>{"arguments":{"location":"London"},"name":"get_weather"}<|tool_call_end|><|im_end|>` + "\n" + `<|im_start|>assistant` + "\n",
		},
		{
			name: "tools definitions with system message",
			messages: []api.Message{
				{Role: "system", Content: "You are helpful."},
				{Role: "user", Content: "What's the weather?"},
			},
			tools: []api.Tool{
				{
					Type: "function",
					Function: api.ToolFunction{
						Name:        "get_weather",
						Description: "Get current weather",
						Parameters: api.ToolFunctionParameters{
							Type: "object",
							Properties: testPropsMap(map[string]api.ToolProperty{
								"location": {
									Type:        api.PropertyType{"string"},
									Description: "City name",
								},
							}),
							Required: []string{"location"},
						},
					},
				},
			},
			thinkValue: &api.ThinkValue{Value: false},
			expected:   `<|im_start|>system` + "\n" + `You are helpful.` + "\n" + `List of tools: [{"type":"function","function":{"name":"get_weather","description":"Get current weather","parameters":{"type":"object","required":["location"],"properties":{"location":{"type":"string","description":"City name"}}}}}]<|im_end|>` + "\n" + `<|im_start|>user` + "\n" + `What's the weather?<|im_end|>` + "\n" + `<|im_start|>assistant` + "\n",
		},
		{
			name: "tools definitions without system message",
			messages: []api.Message{
				{Role: "user", Content: "What's the weather?"},
			},
			tools: []api.Tool{
				{
					Type: "function",
					Function: api.ToolFunction{
						Name:        "get_weather",
						Description: "Get current weather",
						Parameters: api.ToolFunctionParameters{
							Type: "object",
							Properties: testPropsMap(map[string]api.ToolProperty{
								"location": {
									Type:        api.PropertyType{"string"},
									Description: "City name",
								},
							}),
							Required: []string{"location"},
						},
					},
				},
			},
			thinkValue: &api.ThinkValue{Value: false},
			expected:   `<|im_start|>system` + "\n" + `List of tools: [{"type":"function","function":{"name":"get_weather","description":"Get current weather","parameters":{"type":"object","required":["location"],"properties":{"location":{"type":"string","description":"City name"}}}}}]<|im_end|>` + "\n" + `<|im_start|>user` + "\n" + `What's the weather?<|im_end|>` + "\n" + `<|im_start|>assistant` + "\n",
		},
		{
			name: "multiple tools without system message",
			messages: []api.Message{
				{Role: "user", Content: "Hello"},
			},
			tools: []api.Tool{
				{
					Type: "function",
					Function: api.ToolFunction{
						Name:        "get_weather",
						Description: "Get weather",
					},
				},
				{
					Type: "function",
					Function: api.ToolFunction{
						Name:        "get_time",
						Description: "Get time",
					},
				},
			},
			thinkValue: &api.ThinkValue{Value: false},
			expected:   "<|im_start|>system\nList of tools: [{\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"description\":\"Get weather\",\"parameters\":{\"type\":\"\",\"properties\":null}}}, {\"type\":\"function\",\"function\":{\"name\":\"get_time\",\"description\":\"Get time\",\"parameters\":{\"type\":\"\",\"properties\":null}}}]<|im_end|>\n<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n",
		},
		{
			name: "user-tool sequence",
			messages: []api.Message{
				{Role: "user", Content: "Check weather"},
				{Role: "tool", Content: "22C"},
			},
			thinkValue: &api.ThinkValue{Value: false},
			expected:   "<|im_start|>user\nCheck weather<|im_end|>\n<|im_start|>tool\n22C<|im_end|>\n<|im_start|>assistant\n",
		},
		{
			name: "full tool call cycle",
			messages: []api.Message{
				{Role: "user", Content: "Check weather"},
				{Role: "assistant", Content: "Let me check"},
				{Role: "tool", Content: "22C"},
				{Role: "assistant", Content: "It's 22C"},
			},
			thinkValue: &api.ThinkValue{Value: false},
			expected:   "<|im_start|>user\nCheck weather<|im_end|>\n<|im_start|>assistant\nLet me check<|im_end|>\n<|im_start|>tool\n22C<|im_end|>\n<|im_start|>assistant\nIt's 22C<|im_end|>\n<|im_start|>assistant\n",
		},
		{
			name: "unicode content",
			messages: []api.Message{
				{Role: "user", Content: "你好世界! مرحبا 🌍"},
				{Role: "assistant", Content: "Hello! 👋"},
			},
			thinkValue: &api.ThinkValue{Value: false},
			expected:   "<|im_start|>user\n你好世界! مرحبا 🌍<|im_end|>\n<|im_start|>assistant\nHello! 👋<|im_end|>\n<|im_start|>assistant\n",
		},
		{
			name: "newlines in content",
			messages: []api.Message{
				{Role: "user", Content: "Line 1\nLine 2\n\nLine 4"},
				{Role: "assistant", Content: "Response with\nmultiple\nlines"},
			},
			thinkValue: &api.ThinkValue{Value: false},
			expected:   "<|im_start|>user\nLine 1\nLine 2\n\nLine 4<|im_end|>\n<|im_start|>assistant\nResponse with\nmultiple\nlines<|im_end|>\n<|im_start|>assistant\n",
		},
		{
			name: "empty assistant content",
			messages: []api.Message{
				{Role: "user", Content: "Hello"},
				{Role: "assistant", Content: ""},
				{Role: "user", Content: "OK"},
			},
			thinkValue: &api.ThinkValue{Value: false},
			expected:   "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n<|im_end|>\n<|im_start|>user\nOK<|im_end|>\n<|im_start|>assistant\n",
		},
		{
			// Generation prompt does NOT include <think> - model outputs it
			name: "generation prompt has no think tag",
			messages: []api.Message{
				{Role: "user", Content: "Think hard"},
			},
			thinkValue: &api.ThinkValue{Value: true},
			expected:   "<|im_start|>user\nThink hard<|im_end|>\n<|im_start|>assistant\n",
		},
		{
			// Interleaved: thinking before tool call - last assistant preserves thinking
			name: "thinking before tool call (last assistant)",
			messages: []api.Message{
				{Role: "user", Content: "What's the weather?"},
				{
					Role:    "assistant",
					Content: "<think>I need to check the weather</think>",
					ToolCalls: []api.ToolCall{
						{
							Function: api.ToolCallFunction{
								Name: "get_weather",
								Arguments: testArgs(map[string]any{
									"location": "Paris",
								}),
							},
						},
					},
				},
			},
			thinkValue: &api.ThinkValue{Value: false},
			expected:   "<|im_start|>user\nWhat's the weather?<|im_end|>\n<|im_start|>assistant\n<think>I need to check the weather</think><|tool_call_start|>{\"arguments\":{\"location\":\"Paris\"},\"name\":\"get_weather\"}<|tool_call_end|><|im_end|>\n<|im_start|>assistant\n",
		},
		{
			// Two assistants with tool calls - first has thinking stripped
			name: "two assistants with tools: first thinking stripped",
			messages: []api.Message{
				{Role: "user", Content: "What's the weather?"},
				{
					Role:    "assistant",
					Content: "<think>checking</think>",
					ToolCalls: []api.ToolCall{
						{
							Function: api.ToolCallFunction{
								Name: "get_weather",
								Arguments: testArgs(map[string]any{
									"location": "Paris",
								}),
							},
						},
					},
				},
				{Role: "tool", Content: "22C"},
				{Role: "assistant", Content: "<think>got result</think>It's 22C!"},
			},
			thinkValue: &api.ThinkValue{Value: false},
			expected:   "<|im_start|>user\nWhat's the weather?<|im_end|>\n<|im_start|>assistant\n<|tool_call_start|>{\"arguments\":{\"location\":\"Paris\"},\"name\":\"get_weather\"}<|tool_call_end|><|im_end|>\n<|im_start|>tool\n22C<|im_end|>\n<|im_start|>assistant\n<think>got result</think>It's 22C!<|im_end|>\n<|im_start|>assistant\n",
		},
		{
			// Two assistants with tools - both preserved when thinking enabled
			name: "two assistants with tools: both preserved when thinking enabled",
			messages: []api.Message{
				{Role: "user", Content: "What's the weather?"},
				{
					Role:    "assistant",
					Content: "<think>checking</think>",
					ToolCalls: []api.ToolCall{
						{
							Function: api.ToolCallFunction{
								Name: "get_weather",
								Arguments: testArgs(map[string]any{
									"location": "Paris",
								}),
							},
						},
					},
				},
				{Role: "tool", Content: "22C"},
				{Role: "assistant", Content: "<think>got result</think>It's 22C!"},
			},
			thinkValue: &api.ThinkValue{Value: true},
			expected:   "<|im_start|>user\nWhat's the weather?<|im_end|>\n<|im_start|>assistant\n<think>checking</think><|tool_call_start|>{\"arguments\":{\"location\":\"Paris\"},\"name\":\"get_weather\"}<|tool_call_end|><|im_end|>\n<|im_start|>tool\n22C<|im_end|>\n<|im_start|>assistant\n<think>got result</think>It's 22C!<|im_end|>\n<|im_start|>assistant\n",
		},
		{
			// Content before thinking before tool call
			name: "content then thinking then tool call",
			messages: []api.Message{
				{Role: "user", Content: "What's the weather?"},
				{
					Role:    "assistant",
					Content: "Let me check.<think>Using weather API</think>",
					ToolCalls: []api.ToolCall{
						{
							Function: api.ToolCallFunction{
								Name: "get_weather",
								Arguments: testArgs(map[string]any{
									"location": "Paris",
								}),
							},
						},
					},
				},
			},
			thinkValue: &api.ThinkValue{Value: false},
			expected:   "<|im_start|>user\nWhat's the weather?<|im_end|>\n<|im_start|>assistant\nLet me check.<think>Using weather API</think><|tool_call_start|>{\"arguments\":{\"location\":\"Paris\"},\"name\":\"get_weather\"}<|tool_call_end|><|im_end|>\n<|im_start|>assistant\n",
		},
	}

	renderer := &LFM2Renderer{IsThinking: true}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			rendered, err := renderer.Render(tt.messages, tt.tools, tt.thinkValue)
			if err != nil {
				t.Fatalf("Render() error = %v", err)
			}
			if diff := cmp.Diff(tt.expected, rendered); diff != "" {
				t.Errorf("Render() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}
