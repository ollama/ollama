package renderers

import (
	"testing"

	"github.com/google/go-cmp/cmp"

	"github.com/ollama/ollama/api"
)

func TestDeepSeekRenderer(t *testing.T) {
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
				{Role: "user", Content: "Hello, how are you?"},
			},
			thinkValue: &api.ThinkValue{Value: false},
			expected:   `<｜begin▁of▁sentence｜><｜User｜>Hello, how are you?<｜Assistant｜></think>`,
		},
		{
			name: "basic with system message",
			messages: []api.Message{
				{Role: "system", Content: "You are a helpful assistant."},
				{Role: "user", Content: "Hello, how are you?"},
			},
			thinkValue: &api.ThinkValue{Value: false},
			expected:   `<｜begin▁of▁sentence｜>You are a helpful assistant.<｜User｜>Hello, how are you?<｜Assistant｜></think>`,
		},
		{
			name: "multiple system messages",
			messages: []api.Message{
				{Role: "system", Content: "First instruction"},
				{Role: "system", Content: "Second instruction"},
				{Role: "user", Content: "Hello"},
			},
			thinkValue: &api.ThinkValue{Value: false},
			expected: `<｜begin▁of▁sentence｜>First instruction

Second instruction<｜User｜>Hello<｜Assistant｜></think>`,
		},
		{
			name: "thinking enabled",
			messages: []api.Message{
				{Role: "user", Content: "Hello, how are you?"},
			},
			thinkValue: &api.ThinkValue{Value: true},
			expected:   `<｜begin▁of▁sentence｜><｜User｜>Hello, how are you?<｜Assistant｜><think>`,
		},
		{
			name: "thinking enabled with system",
			messages: []api.Message{
				{Role: "system", Content: "You are a helpful assistant."},
				{Role: "user", Content: "Hello, how are you?"},
			},
			thinkValue: &api.ThinkValue{Value: true},
			expected:   `<｜begin▁of▁sentence｜>You are a helpful assistant.<｜User｜>Hello, how are you?<｜Assistant｜><think>`,
		},
		{
			name: "conversation with assistant response",
			messages: []api.Message{
				{Role: "user", Content: "What is the capital of France?"},
				{Role: "assistant", Content: "The capital of France is Paris."},
				{Role: "user", Content: "Fantastic!"},
			},
			thinkValue: &api.ThinkValue{Value: false},
			expected:   `<｜begin▁of▁sentence｜><｜User｜>What is the capital of France?<｜Assistant｜></think>The capital of France is Paris.<｜end▁of▁sentence｜><｜User｜>Fantastic!<｜Assistant｜></think>`,
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
								Arguments: api.ToolCallFunctionArguments{
									"location": "Paris",
								},
							},
						},
					},
				},
			},
			thinkValue: &api.ThinkValue{Value: false},
			expected:   `<｜begin▁of▁sentence｜><｜User｜>What's the weather?<｜Assistant｜></think><｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>{"location":"Paris"}<｜tool▁call▁end｜><｜tool▁calls▁end｜><｜end▁of▁sentence｜>`,
		},
		{
			name: "assistant with content and tool calls",
			messages: []api.Message{
				{Role: "user", Content: "What's the weather in Paris?"},
				{
					Role:    "assistant",
					Content: "I'll check the weather for you.",
					ToolCalls: []api.ToolCall{
						{
							Function: api.ToolCallFunction{
								Name: "get_weather",
								Arguments: api.ToolCallFunctionArguments{
									"location": "Paris",
								},
							},
						},
					},
				},
			},
			thinkValue: &api.ThinkValue{Value: false},
			expected:   `<｜begin▁of▁sentence｜><｜User｜>What's the weather in Paris?<｜Assistant｜></think>I'll check the weather for you.<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>{"location":"Paris"}<｜tool▁call▁end｜><｜tool▁calls▁end｜><｜end▁of▁sentence｜>`,
		},
		{
			name: "tool response",
			messages: []api.Message{
				{Role: "user", Content: "What's the weather?"},
				{
					Role: "assistant",
					ToolCalls: []api.ToolCall{
						{
							Function: api.ToolCallFunction{
								Name: "get_weather",
								Arguments: api.ToolCallFunctionArguments{
									"location": "Paris",
								},
							},
						},
					},
				},
				{Role: "tool", Content: "Temperature: 22°C, Sunny"},
			},
			thinkValue: &api.ThinkValue{Value: false},
			expected:   `<｜begin▁of▁sentence｜><｜User｜>What's the weather?<｜Assistant｜></think><｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>{"location":"Paris"}<｜tool▁call▁end｜><｜tool▁calls▁end｜><｜end▁of▁sentence｜><｜tool▁output▁begin｜>Temperature: 22°C, Sunny<｜tool▁output▁end｜>`,
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
								Arguments: api.ToolCallFunctionArguments{
									"location": "Paris",
								},
							},
						},
						{
							Function: api.ToolCallFunction{
								Name: "get_weather",
								Arguments: api.ToolCallFunctionArguments{
									"location": "London",
								},
							},
						},
					},
				},
				{Role: "tool", Content: "Paris: 22°C, Sunny"},
				{Role: "tool", Content: "London: 18°C, Cloudy"},
			},
			thinkValue: &api.ThinkValue{Value: false},
			expected:   `<｜begin▁of▁sentence｜><｜User｜>Get weather for Paris and London<｜Assistant｜></think><｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>{"location":"Paris"}<｜tool▁call▁end｜><｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>{"location":"London"}<｜tool▁call▁end｜><｜tool▁calls▁end｜><｜end▁of▁sentence｜><｜tool▁output▁begin｜>Paris: 22°C, Sunny<｜tool▁output▁end｜><｜tool▁output▁begin｜>London: 18°C, Cloudy<｜tool▁output▁end｜>`,
		},
		{
			name: "content with </think> tag removal",
			messages: []api.Message{
				{Role: "user", Content: "Think about this"},
				{Role: "assistant", Content: "I'm thinking about this.</think>The answer is 42."},
			},
			thinkValue: &api.ThinkValue{Value: false},
			expected:   `<｜begin▁of▁sentence｜><｜User｜>Think about this<｜Assistant｜></think>The answer is 42.<｜end▁of▁sentence｜>`,
		},
		{
			name: "empty system message",
			messages: []api.Message{
				{Role: "system", Content: ""},
				{Role: "user", Content: "Hello"},
			},
			thinkValue: &api.ThinkValue{Value: false},
			expected:   `<｜begin▁of▁sentence｜><｜User｜>Hello<｜Assistant｜></think>`,
		},
		{
			name: "empty assistant content",
			messages: []api.Message{
				{Role: "user", Content: "Hello"},
				{Role: "assistant", Content: ""},
			},
			thinkValue: &api.ThinkValue{Value: false},
			expected:   `<｜begin▁of▁sentence｜><｜User｜>Hello<｜Assistant｜></think><｜end▁of▁sentence｜>`,
		},
		{
			name: "special characters",
			messages: []api.Message{
				{Role: "user", Content: "What about <|special|> tokens and \"quotes\"?"},
				{Role: "assistant", Content: "They're handled normally."},
			},
			thinkValue: &api.ThinkValue{Value: false},
			expected:   `<｜begin▁of▁sentence｜><｜User｜>What about <|special|> tokens and "quotes"?<｜Assistant｜></think>They're handled normally.<｜end▁of▁sentence｜>`,
		},
		{
			name: "tool calls with null content",
			messages: []api.Message{
				{Role: "user", Content: "Get weather"},
				{
					Role: "assistant",
					ToolCalls: []api.ToolCall{
						{
							Function: api.ToolCallFunction{
								Name: "get_weather",
								Arguments: api.ToolCallFunctionArguments{
									"location": "Paris",
								},
							},
						},
					},
				},
			},
			thinkValue: &api.ThinkValue{Value: false},
			expected:   `<｜begin▁of▁sentence｜><｜User｜>Get weather<｜Assistant｜></think><｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>{"location":"Paris"}<｜tool▁call▁end｜><｜tool▁calls▁end｜><｜end▁of▁sentence｜>`,
		},
		{
			name: "assistant after tool context",
			messages: []api.Message{
				{Role: "user", Content: "Process data"},
				{
					Role: "assistant",
					ToolCalls: []api.ToolCall{
						{
							Function: api.ToolCallFunction{
								Name: "process",
								Arguments: api.ToolCallFunctionArguments{
									"data": "test",
								},
							},
						},
					},
				},
				{Role: "tool", Content: "Success"},
				{Role: "assistant", Content: "Done"},
			},
			thinkValue: &api.ThinkValue{Value: false},
			expected:   `<｜begin▁of▁sentence｜><｜User｜>Process data<｜Assistant｜></think><｜tool▁calls▁begin｜><｜tool▁call▁begin｜>process<｜tool▁sep｜>{"data":"test"}<｜tool▁call▁end｜><｜tool▁calls▁end｜><｜end▁of▁sentence｜><｜tool▁output▁begin｜>Success<｜tool▁output▁end｜>Done<｜end▁of▁sentence｜>`,
		},
		{
			name:       "no messages",
			messages:   []api.Message{},
			thinkValue: &api.ThinkValue{Value: false},
			expected:   `<｜begin▁of▁sentence｜>`,
		},
		{
			name: "only system messages",
			messages: []api.Message{
				{Role: "system", Content: "System instruction"},
			},
			thinkValue: &api.ThinkValue{Value: false},
			expected:   `<｜begin▁of▁sentence｜>System instruction`,
		},
		{
			name: "multiple think tags in content",
			messages: []api.Message{
				{Role: "user", Content: "Complex question"},
				{Role: "assistant", Content: "First thought</think>Second thought</think>Final answer"},
			},
			thinkValue: &api.ThinkValue{Value: false},
			expected:   `<｜begin▁of▁sentence｜><｜User｜>Complex question<｜Assistant｜></think>Second thought</think>Final answer<｜end▁of▁sentence｜>`,
		},
	}

	renderer := &DeepSeekRenderer{isThinking: true}
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
