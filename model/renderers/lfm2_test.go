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

// TestLFM2Renderer_GoldenTests validates renderer output against the exact expected output
// from the LFM2.5-1.2B-Thinking chat_template.jinja file.
// Note: BOS token (<|startoftext|>) is added by the tokenizer, not the renderer.
func TestLFM2Renderer_GoldenTests(t *testing.T) {
	tests := []struct {
		name       string
		messages   []api.Message
		tools      []api.Tool
		thinkValue *api.ThinkValue
		// expected is the jinja template output WITHOUT the BOS token
		// (tokenizer adds BOS, not the renderer)
		expected string
	}{
		{
			// Jinja Test 1: Simple user message
			// Input: [{"role": "user", "content": "Hello"}]
			// Jinja output: '<|startoftext|><|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n'
			name: "jinja_test_1_simple_user",
			messages: []api.Message{
				{Role: "user", Content: "Hello"},
			},
			thinkValue: &api.ThinkValue{Value: false},
			expected:   "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n",
		},
		{
			// Jinja Test 2: With system message
			// Input: [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "Hello"}]
			// Jinja output: '<|startoftext|><|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n'
			name: "jinja_test_2_with_system",
			messages: []api.Message{
				{Role: "system", Content: "You are a helpful assistant."},
				{Role: "user", Content: "Hello"},
			},
			thinkValue: &api.ThinkValue{Value: false},
			expected:   "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n",
		},
		{
			// Jinja Test 6: Tool call and response
			// Input: [{"role": "user", "content": "What is the weather?"}, {"role": "assistant", "content": "<|tool_call_start|>{...}<|tool_call_end|>"}, {"role": "tool", "content": "Sunny, 72F"}]
			// Jinja output: '<|startoftext|><|im_start|>user\nWhat is the weather?<|im_end|>\n<|im_start|>assistant\n<|tool_call_start|>{"name": "get_weather", "arguments": {"location": "NYC"}}<|tool_call_end|><|im_end|>\n<|im_start|>tool\nSunny, 72F<|im_end|>\n<|im_start|>assistant\n'
			name: "jinja_test_6_tool_call_response",
			messages: []api.Message{
				{Role: "user", Content: "What is the weather?"},
				{
					Role: "assistant",
					ToolCalls: []api.ToolCall{
						{
							Function: api.ToolCallFunction{
								Name:      "get_weather",
								Arguments: testArgs(map[string]any{"location": "NYC"}),
							},
						},
					},
				},
				{Role: "tool", Content: "Sunny, 72F"},
			},
			thinkValue: &api.ThinkValue{Value: false},
			expected:   "<|im_start|>user\nWhat is the weather?<|im_end|>\n<|im_start|>assistant\n<|tool_call_start|>{\"arguments\":{\"location\":\"NYC\"},\"name\":\"get_weather\"}<|tool_call_end|><|im_end|>\n<|im_start|>tool\nSunny, 72F<|im_end|>\n<|im_start|>assistant\n",
		},
		{
			// Jinja Test 8: Last assistant with thinking (NOT stripped)
			// Input: [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "<think>Thinking...</think>Hi there!"}]
			// add_generation_prompt=False
			// Jinja output: '<|startoftext|><|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n<think>Thinking...</think>Hi there!<|im_end|>\n'
			// Note: We always add generation prompt, so we test this indirectly
			name: "jinja_test_8_last_assistant_thinking_kept",
			messages: []api.Message{
				{Role: "user", Content: "Hello"},
				{Role: "assistant", Content: "<think>Thinking...</think>Hi there!"},
			},
			thinkValue: &api.ThinkValue{Value: false},
			// The assistant IS the last assistant, so thinking is preserved
			expected: "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n<think>Thinking...</think>Hi there!<|im_end|>\n<|im_start|>assistant\n",
		},
		{
			// Jinja Test 9: Multiple assistants - only past stripped
			// Input: [{"role": "user", "content": "First question"}, {"role": "assistant", "content": "<think>First thought</think>First answer"}, {"role": "user", "content": "Second question"}, {"role": "assistant", "content": "<think>Second thought</think>Second answer"}]
			// add_generation_prompt=False
			// Jinja output: '<|startoftext|><|im_start|>user\nFirst question<|im_end|>\n<|im_start|>assistant\nFirst answer<|im_end|>\n<|im_start|>user\nSecond question<|im_end|>\n<|im_start|>assistant\n<think>Second thought</think>Second answer<|im_end|>\n'
			name: "jinja_test_9_multiple_assistants_past_stripped",
			messages: []api.Message{
				{Role: "user", Content: "First question"},
				{Role: "assistant", Content: "<think>First thought</think>First answer"},
				{Role: "user", Content: "Second question"},
				{Role: "assistant", Content: "<think>Second thought</think>Second answer"},
			},
			thinkValue: &api.ThinkValue{Value: false},
			// First assistant is NOT the last, so thinking stripped. Second IS the last, thinking kept.
			expected: "<|im_start|>user\nFirst question<|im_end|>\n<|im_start|>assistant\nFirst answer<|im_end|>\n<|im_start|>user\nSecond question<|im_end|>\n<|im_start|>assistant\n<think>Second thought</think>Second answer<|im_end|>\n<|im_start|>assistant\n",
		},
		{
			// Jinja Test 10: Thinking + tool call (past assistant)
			// When there's a tool response after, the assistant is NOT the last
			// Input: user asks, assistant thinks+calls tool, tool responds
			// Jinja output: '<|startoftext|><|im_start|>user\nWhat\'s the weather?<|im_end|>\n<|im_start|>assistant\n<think>I should check the weather</think><|tool_call_start|>{"name": "get_weather", "arguments": {"location": "NYC"}}<|tool_call_end|><|im_end|>\n<|im_start|>tool\nSunny<|im_end|>\n<|im_start|>assistant\n'
			// Note: In this case the assistant IS still the last assistant (tool is not assistant)
			name: "jinja_test_10_thinking_with_tool_call",
			messages: []api.Message{
				{Role: "user", Content: "What's the weather?"},
				{
					Role:    "assistant",
					Content: "<think>I should check the weather</think>",
					ToolCalls: []api.ToolCall{
						{
							Function: api.ToolCallFunction{
								Name:      "get_weather",
								Arguments: testArgs(map[string]any{"location": "NYC"}),
							},
						},
					},
				},
				{Role: "tool", Content: "Sunny"},
			},
			thinkValue: &api.ThinkValue{Value: false},
			// The assistant IS the last assistant (tool messages don't count), so thinking preserved
			expected: "<|im_start|>user\nWhat's the weather?<|im_end|>\n<|im_start|>assistant\n<think>I should check the weather</think><|tool_call_start|>{\"arguments\":{\"location\":\"NYC\"},\"name\":\"get_weather\"}<|tool_call_end|><|im_end|>\n<|im_start|>tool\nSunny<|im_end|>\n<|im_start|>assistant\n",
		},
		{
			// Multi-assistant with tool call cycle: first assistant thinking stripped
			// user -> assistant (think + tool) -> tool -> assistant (think + answer)
			// First assistant is NOT last, second IS last
			name: "multi_assistant_tool_cycle",
			messages: []api.Message{
				{Role: "user", Content: "What's the weather?"},
				{
					Role:    "assistant",
					Content: "<think>Let me check</think>",
					ToolCalls: []api.ToolCall{
						{
							Function: api.ToolCallFunction{
								Name:      "get_weather",
								Arguments: testArgs(map[string]any{"location": "NYC"}),
							},
						},
					},
				},
				{Role: "tool", Content: "Sunny"},
				{Role: "assistant", Content: "<think>Got it</think>It's sunny!"},
			},
			thinkValue: &api.ThinkValue{Value: false},
			// First assistant: NOT last -> thinking stripped
			// Second assistant: IS last -> thinking kept
			expected: "<|im_start|>user\nWhat's the weather?<|im_end|>\n<|im_start|>assistant\n<|tool_call_start|>{\"arguments\":{\"location\":\"NYC\"},\"name\":\"get_weather\"}<|tool_call_end|><|im_end|>\n<|im_start|>tool\nSunny<|im_end|>\n<|im_start|>assistant\n<think>Got it</think>It's sunny!<|im_end|>\n<|im_start|>assistant\n",
		},
		{
			// Tools list in system - matches jinja format exactly
			// Note: Go json.Marshal includes all struct fields, including empty parameters
			name: "tools_list_format",
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
			// Tools are JSON serialized as array: List of tools: [{...}, {...}]
			// Note: json.Marshal includes empty parameters field
			expected: "<|im_start|>system\nList of tools: [{\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"description\":\"Get weather\",\"parameters\":{\"type\":\"\",\"properties\":null}}}, {\"type\":\"function\",\"function\":{\"name\":\"get_time\",\"description\":\"Get time\",\"parameters\":{\"type\":\"\",\"properties\":null}}}]<|im_end|>\n<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n",
		},
		{
			// System + tools combined (jinja Test 7 equivalent)
			name: "system_plus_tools",
			messages: []api.Message{
				{Role: "system", Content: "You are a weather assistant."},
				{Role: "user", Content: "What is the weather?"},
			},
			tools: []api.Tool{
				{
					Type: "function",
					Function: api.ToolFunction{
						Name: "get_weather",
					},
				},
			},
			thinkValue: &api.ThinkValue{Value: false},
			// System content + newline + tools list
			// Note: json.Marshal includes empty parameters field
			expected: "<|im_start|>system\nYou are a weather assistant.\nList of tools: [{\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"parameters\":{\"type\":\"\",\"properties\":null}}}]<|im_end|>\n<|im_start|>user\nWhat is the weather?<|im_end|>\n<|im_start|>assistant\n",
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
