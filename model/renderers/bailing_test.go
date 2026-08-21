package renderers

import (
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/ollama/ollama/api"
)

func TestBailingRenderer(t *testing.T) {
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
				{Role: "user", Content: "Hello"},
			},
			expected: "<role>SYSTEM</role>detailed thinking on<|role_end|><role>HUMAN</role>Hello<|role_end|><role>ASSISTANT</role>\n<think>",
		},
		{
			name: "thinking disabled",
			messages: []api.Message{
				{Role: "user", Content: "Hello"},
			},
			thinkValue: &api.ThinkValue{Value: false},
			expected:   "<role>SYSTEM</role>detailed thinking off<|role_end|><role>HUMAN</role>Hello<|role_end|><role>ASSISTANT</role>\n<think></think>",
		},
		{
			name: "system and user",
			messages: []api.Message{
				{Role: "system", Content: "You are helpful."},
				{Role: "user", Content: "Hello"},
			},
			expected: "<role>SYSTEM</role>You are helpful.\ndetailed thinking on<|role_end|><role>HUMAN</role>Hello<|role_end|><role>ASSISTANT</role>\n<think>",
		},
		{
			name: "system message already carrying the thinking directive",
			messages: []api.Message{
				{Role: "system", Content: "Be brief.\ndetailed thinking off"},
				{Role: "user", Content: "Hello"},
			},
			expected: "<role>SYSTEM</role>Be brief.\ndetailed thinking off<|role_end|><role>HUMAN</role>Hello<|role_end|><role>ASSISTANT</role>\n<think>",
		},
		{
			name: "multi-turn conversation",
			messages: []api.Message{
				{Role: "user", Content: "Hi"},
				{Role: "assistant", Content: "Hello there"},
				{Role: "user", Content: "How are you?"},
			},
			expected: "<role>SYSTEM</role>detailed thinking on<|role_end|><role>HUMAN</role>Hi<|role_end|><role>ASSISTANT</role>\n<think></think>Hello there<|role_end|><role>HUMAN</role>How are you?<|role_end|><role>ASSISTANT</role>\n<think>",
		},
		{
			name: "assistant with explicit thinking",
			messages: []api.Message{
				{Role: "user", Content: "Answer with reasoning."},
				{Role: "assistant", Thinking: "Plan.", Content: "Done."},
				{Role: "user", Content: "Next"},
			},
			expected: "<role>SYSTEM</role>detailed thinking on<|role_end|><role>HUMAN</role>Answer with reasoning.<|role_end|><role>ASSISTANT</role>\n<think>Plan.</think>Done.<|role_end|><role>HUMAN</role>Next<|role_end|><role>ASSISTANT</role>\n<think>",
		},
		{
			name: "assistant with inline think block in content",
			messages: []api.Message{
				{Role: "user", Content: "Q"},
				{Role: "assistant", Content: "<think>hmm</think>\nAnswer"},
				{Role: "user", Content: "Next"},
			},
			expected: "<role>SYSTEM</role>detailed thinking on<|role_end|><role>HUMAN</role>Q<|role_end|><role>ASSISTANT</role>\n<think>hmm</think>Answer<|role_end|><role>HUMAN</role>Next<|role_end|><role>ASSISTANT</role>\n<think>",
		},
		{
			name: "mid-conversation system message",
			messages: []api.Message{
				{Role: "user", Content: "Hi"},
				{Role: "system", Content: "Switch style."},
				{Role: "user", Content: "Go"},
			},
			expected: "<role>SYSTEM</role>detailed thinking on<|role_end|><role>HUMAN</role>Hi<|role_end|><role>SYSTEM</role>Switch style.<|role_end|><role>HUMAN</role>Go<|role_end|><role>ASSISTANT</role>\n<think>",
		},
		{
			name: "tool call round trip",
			messages: []api.Message{
				{Role: "system", Content: "You are helpful."},
				{Role: "user", Content: "Weather?"},
				{
					Role: "assistant",
					ToolCalls: []api.ToolCall{
						{
							Function: api.ToolCallFunction{
								Name:      "get_weather",
								Arguments: args(`{"location": "Tokyo", "unit": "celsius"}`),
							},
						},
					},
				},
				{Role: "tool", Content: `{"temperature":22}`},
			},
			tools: []api.Tool{
				{
					Type: "function",
					Function: api.ToolFunction{
						Name:        "get_weather",
						Description: "Get weather",
						Parameters: api.ToolFunctionParameters{
							Type:       "object",
							Required:   []string{"location"},
							Properties: propsMap(`{"location": {"type": "string"}}`),
						},
					},
				},
			},
			expected: "<role>SYSTEM</role>You are helpful.\n" +
				"# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n" +
				`{"type": "function", "function": {"name": "get_weather", "description": "Get weather", "parameters": {"type": "object", "required": ["location"], "properties": {"location": {"type": "string"}}}}}` +
				"\n</tools>\n\nIf none of the functions can be used, point it out. If the given question lacks the parameters required by the function, also point it out.\nIf you need to use a function, for each function call, output the function name and arguments within the following XML format:\n<tool_call>{function-name}\n<arg_key>{arg-key-1}</arg_key>\n<arg_value>{arg-value-1}</arg_value>\n<arg_key>{arg-key-2}</arg_key>\n<arg_value>{arg-value-2}</arg_value>\n...\n</tool_call>\n" +
				"detailed thinking on<|role_end|>" +
				"<role>HUMAN</role>Weather?<|role_end|>" +
				"<role>ASSISTANT</role>\n<think></think>" +
				"<tool_call>get_weather<arg_key>location</arg_key>\n<arg_value>Tokyo</arg_value><arg_key>unit</arg_key>\n<arg_value>celsius</arg_value>\n</tool_call><|role_end|>" +
				"<role>OBSERVATION</role>\n<tool_response>\n{\"temperature\":22}\n</tool_response><|role_end|>" +
				"<role>ASSISTANT</role>\n<think>",
		},
		{
			name: "tool call with content and non-string argument",
			messages: []api.Message{
				{Role: "user", Content: "Weather?"},
				{
					Role:    "assistant",
					Content: "Let me check",
					ToolCalls: []api.ToolCall{
						{
							Function: api.ToolCallFunction{
								Name:      "get_weather",
								Arguments: args(`{"days": 3}`),
							},
						},
					},
				},
				{Role: "tool", Content: "sunny"},
			},
			expected: "<role>SYSTEM</role>detailed thinking on<|role_end|>" +
				"<role>HUMAN</role>Weather?<|role_end|>" +
				"<role>ASSISTANT</role>\n<think></think>Let me check\n" +
				"<tool_call>get_weather<arg_key>days</arg_key>\n<arg_value>3</arg_value>\n</tool_call><|role_end|>" +
				"<role>OBSERVATION</role>\n<tool_response>\nsunny\n</tool_response><|role_end|>" +
				"<role>ASSISTANT</role>\n<think>",
		},
		{
			name: "prefill assistant content",
			messages: []api.Message{
				{Role: "user", Content: "Q"},
				{Role: "assistant", Content: "The answer is"},
			},
			expected: "<role>SYSTEM</role>detailed thinking on<|role_end|><role>HUMAN</role>Q<|role_end|><role>ASSISTANT</role>\n<think></think>The answer is",
		},
		{
			name: "prefill assistant thinking",
			messages: []api.Message{
				{Role: "user", Content: "Q"},
				{Role: "assistant", Thinking: "Let me think"},
			},
			expected: "<role>SYSTEM</role>detailed thinking on<|role_end|><role>HUMAN</role>Q<|role_end|><role>ASSISTANT</role>\n<think>Let me think",
		},
		{
			name: "prefill assistant thinking with thinking disabled closes the block",
			messages: []api.Message{
				{Role: "user", Content: "Q"},
				{Role: "assistant", Thinking: "Let me think"},
			},
			thinkValue: &api.ThinkValue{Value: false},
			expected:   "<role>SYSTEM</role>detailed thinking off<|role_end|><role>HUMAN</role>Q<|role_end|><role>ASSISTANT</role>\n<think>Let me think</think>",
		},
		{
			name: "bare assistant prefill matches the generation prompt",
			messages: []api.Message{
				{Role: "user", Content: "Q"},
				{Role: "assistant"},
			},
			expected: "<role>SYSTEM</role>detailed thinking on<|role_end|><role>HUMAN</role>Q<|role_end|><role>ASSISTANT</role>\n<think>",
		},
		{
			name: "bare assistant prefill with thinking disabled",
			messages: []api.Message{
				{Role: "user", Content: "Q"},
				{Role: "assistant"},
			},
			thinkValue: &api.ThinkValue{Value: false},
			expected:   "<role>SYSTEM</role>detailed thinking off<|role_end|><role>HUMAN</role>Q<|role_end|><role>ASSISTANT</role>\n<think></think>",
		},
	}

	renderer := &BailingRenderer{}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			rendered, err := renderer.Render(tt.messages, tt.tools, tt.thinkValue)
			if err != nil {
				t.Fatalf("Render() error: %v", err)
			}
			if diff := cmp.Diff(tt.expected, rendered); diff != "" {
				t.Errorf("rendered prompt mismatch (-want +got):\n%s", diff)
			}
		})
	}
}
