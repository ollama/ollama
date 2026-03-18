package renderers

import (
	"testing"

	"github.com/google/go-cmp/cmp"

	"github.com/ollama/ollama/api"
)

func TestCogitoRenderer(t *testing.T) {
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
			expected:   `<｜begin▁of▁sentence｜>You are Cogito, an AI assistant created by Deep Cogito, which is an AI research lab based in San Francisco.<｜User｜>Hello, how are you?<｜Assistant｜>`,
		},
		{
			name: "basic with system message",
			messages: []api.Message{
				{Role: "system", Content: "You are a helpful assistant."},
				{Role: "user", Content: "Hello, how are you?"},
			},
			thinkValue: &api.ThinkValue{Value: false},
			expected: `<｜begin▁of▁sentence｜>You are Cogito, an AI assistant created by Deep Cogito, which is an AI research lab based in San Francisco.

You are a helpful assistant.<｜User｜>Hello, how are you?<｜Assistant｜>`,
		},
		{
			name: "conversation with assistant response",
			messages: []api.Message{
				{Role: "user", Content: "What is the capital of France?"},
				{Role: "assistant", Content: "The capital of France is Paris."},
				{Role: "user", Content: "Fantastic!"},
			},
			thinkValue: &api.ThinkValue{Value: false},
			expected:   `<｜begin▁of▁sentence｜>You are Cogito, an AI assistant created by Deep Cogito, which is an AI research lab based in San Francisco.<｜User｜>What is the capital of France?<｜Assistant｜>The capital of France is Paris.<｜end▁of▁sentence｜><｜User｜>Fantastic!<｜Assistant｜>`,
		},
		{
			name: "thinking enabled without system",
			messages: []api.Message{
				{Role: "user", Content: "Hello, how are you?"},
			},
			thinkValue: &api.ThinkValue{Value: true},
			expected: `<｜begin▁of▁sentence｜>Enable deep thinking subroutine.

You are Cogito, an AI assistant created by Deep Cogito, which is an AI research lab based in San Francisco.<｜User｜>Hello, how are you?<｜Assistant｜><think>
`,
		},
		{
			name: "thinking enabled with system",
			messages: []api.Message{
				{Role: "system", Content: "You are a helpful assistant."},
				{Role: "user", Content: "Hello, how are you?"},
			},
			thinkValue: &api.ThinkValue{Value: true},
			expected: `<｜begin▁of▁sentence｜>Enable deep thinking subroutine.

You are Cogito, an AI assistant created by Deep Cogito, which is an AI research lab based in San Francisco.

You are a helpful assistant.

<｜User｜>Hello, how are you?<｜Assistant｜><think>
`,
		},
		{
			name: "thinking disabled",
			messages: []api.Message{
				{Role: "user", Content: "Hello, how are you?"},
			},
			thinkValue: &api.ThinkValue{Value: false},
			expected:   `<｜begin▁of▁sentence｜>You are Cogito, an AI assistant created by Deep Cogito, which is an AI research lab based in San Francisco.<｜User｜>Hello, how are you?<｜Assistant｜>`,
		},
		{
			name: "with tools",
			messages: []api.Message{
				{Role: "user", Content: "What's the weather like?"},
			},
			thinkValue: &api.ThinkValue{Value: false},
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
			expected: `<｜begin▁of▁sentence｜>You are Cogito, an AI assistant created by Deep Cogito, which is an AI research lab based in San Francisco.
You have the following functions available:
` + "```json\n" + `{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name"
                }
            },
            "required": [
                "location"
            ]
        }
    }
}
` + "```\n" + `<｜User｜>What's the weather like?<｜Assistant｜>`,
		},
		{
			name: "assistant with tool calls",
			messages: []api.Message{
				{Role: "user", Content: "What's the weather in Paris?"},
				{
					Role:    "assistant",
					Content: "I'll check the weather in Paris for you.",
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
			expected: `<｜begin▁of▁sentence｜>You are Cogito, an AI assistant created by Deep Cogito, which is an AI research lab based in San Francisco.<｜User｜>What's the weather in Paris?<｜Assistant｜>I'll check the weather in Paris for you.<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather
` + "```json\n" + `{"location":"Paris"}
` + "```" + `<｜tool▁call▁end｜><｜tool▁calls▁end｜><｜end▁of▁sentence｜><｜Assistant｜>`,
		},
		{
			name: "tool response",
			messages: []api.Message{
				{Role: "user", Content: "What's the weather in Paris?"},
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
				{Role: "tool", Content: "Temperature: 22°C, Sunny"},
			},
			thinkValue: &api.ThinkValue{Value: false},
			expected: `<｜begin▁of▁sentence｜>You are Cogito, an AI assistant created by Deep Cogito, which is an AI research lab based in San Francisco.<｜User｜>What's the weather in Paris?<｜Assistant｜><｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather
` + "```json\n" + `{"location":"Paris"}
` + "```" + `<｜tool▁call▁end｜><｜tool▁calls▁end｜><｜end▁of▁sentence｜><｜tool▁outputs▁begin｜><｜tool▁output▁begin｜>Temperature: 22°C, Sunny<｜tool▁output▁end｜><｜tool▁outputs▁end｜><｜Assistant｜>`,
		},
		{
			name: "multiple tool responses",
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
				{Role: "tool", Content: "Paris: 22°C, Sunny"},
				{Role: "tool", Content: "London: 18°C, Cloudy"},
			},
			thinkValue: &api.ThinkValue{Value: false},
			expected: `<｜begin▁of▁sentence｜>You are Cogito, an AI assistant created by Deep Cogito, which is an AI research lab based in San Francisco.<｜User｜>Get weather for Paris and London<｜Assistant｜><｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather
` + "```json\n" + `{"location":"Paris"}
` + "```" + `<｜tool▁call▁end｜>
<｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather
` + "```json\n" + `{"location":"London"}
` + "```" + `<｜tool▁call▁end｜><｜tool▁calls▁end｜><｜end▁of▁sentence｜><｜tool▁outputs▁begin｜><｜tool▁output▁begin｜>Paris: 22°C, Sunny<｜tool▁output▁end｜>
<｜tool▁output▁begin｜>London: 18°C, Cloudy<｜tool▁output▁end｜><｜tool▁outputs▁end｜><｜Assistant｜>`,
		},
		{
			name: "thinking with tools",
			messages: []api.Message{
				{Role: "user", Content: "What's the weather like?"},
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
			thinkValue: &api.ThinkValue{Value: true},
			expected: `<｜begin▁of▁sentence｜>Enable deep thinking subroutine.

You are Cogito, an AI assistant created by Deep Cogito, which is an AI research lab based in San Francisco.
You have the following functions available:
` + "```json\n" + `{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name"
                }
            },
            "required": [
                "location"
            ]
        }
    }
}
` + "```\n" + `<｜User｜>What's the weather like?<｜Assistant｜><think>
`,
		},
		// test cases based on cogito
		{
			name: "single_turn_thinking_false",
			messages: []api.Message{
				{Role: "user", Content: "Hello"},
			},
			thinkValue: &api.ThinkValue{Value: false},
			expected:   `<｜begin▁of▁sentence｜>You are Cogito, an AI assistant created by Deep Cogito, which is an AI research lab based in San Francisco.<｜User｜>Hello<｜Assistant｜>`,
		},
		{
			name: "single_turn_thinking_true",
			messages: []api.Message{
				{Role: "user", Content: "Hello"},
			},
			thinkValue: &api.ThinkValue{Value: true},
			expected: `<｜begin▁of▁sentence｜>Enable deep thinking subroutine.

You are Cogito, an AI assistant created by Deep Cogito, which is an AI research lab based in San Francisco.<｜User｜>Hello<｜Assistant｜><think>
`,
		},
		{
			name: "multi_turn_thinking_false",
			messages: []api.Message{
				{Role: "user", Content: "Hello"},
				{Role: "assistant", Content: "Hi there!"},
				{Role: "user", Content: "How are you?"},
			},
			thinkValue: &api.ThinkValue{Value: false},
			expected:   `<｜begin▁of▁sentence｜>You are Cogito, an AI assistant created by Deep Cogito, which is an AI research lab based in San Francisco.<｜User｜>Hello<｜Assistant｜>Hi there!<｜end▁of▁sentence｜><｜User｜>How are you?<｜Assistant｜>`,
		},
		{
			name: "multi_turn_thinking_true",
			messages: []api.Message{
				{Role: "user", Content: "Hello"},
				{Role: "assistant", Content: "Hi there!"},
				{Role: "user", Content: "How are you?"},
			},
			thinkValue: &api.ThinkValue{Value: true},
			expected: `<｜begin▁of▁sentence｜>Enable deep thinking subroutine.

You are Cogito, an AI assistant created by Deep Cogito, which is an AI research lab based in San Francisco.<｜User｜>Hello<｜Assistant｜>Hi there!<｜end▁of▁sentence｜><｜User｜>How are you?<｜Assistant｜><think>
`,
		},
		{
			name: "multi_with_system_thinking_false",
			messages: []api.Message{
				{Role: "system", Content: "You are a helpful assistant"},
				{Role: "user", Content: "Start"},
				{Role: "assistant", Content: "Okay"},
			},
			thinkValue: &api.ThinkValue{Value: false},
			expected: `<｜begin▁of▁sentence｜>You are Cogito, an AI assistant created by Deep Cogito, which is an AI research lab based in San Francisco.

You are a helpful assistant<｜User｜>Start<｜Assistant｜>Okay<｜end▁of▁sentence｜><｜Assistant｜>`,
		},
		{
			name: "multi_with_system_thinking_true",
			messages: []api.Message{
				{Role: "system", Content: "You are a helpful assistant"},
				{Role: "user", Content: "Start"},
				{Role: "assistant", Content: "Okay"},
			},
			thinkValue: &api.ThinkValue{Value: true},
			expected: `<｜begin▁of▁sentence｜>Enable deep thinking subroutine.

You are Cogito, an AI assistant created by Deep Cogito, which is an AI research lab based in San Francisco.

You are a helpful assistant

<｜User｜>Start<｜Assistant｜>Okay<｜end▁of▁sentence｜><｜Assistant｜><think>
`,
		},
		{
			name: "multi_with_system2_thinking_false",
			messages: []api.Message{
				{Role: "system", Content: "You are a pirate chatbot who always responds in pirate speak!"},
				{Role: "user", Content: "Give me a short introduction to LLMs."},
				{Role: "assistant", Content: "Arrr! I'm a pirate"},
				{Role: "user", Content: "Tell me more about LLMs."},
			},
			thinkValue: &api.ThinkValue{Value: false},
			expected: `<｜begin▁of▁sentence｜>You are Cogito, an AI assistant created by Deep Cogito, which is an AI research lab based in San Francisco.

You are a pirate chatbot who always responds in pirate speak!<｜User｜>Give me a short introduction to LLMs.<｜Assistant｜>Arrr! I'm a pirate<｜end▁of▁sentence｜><｜User｜>Tell me more about LLMs.<｜Assistant｜>`,
		},
		{
			name: "multi_with_system2_thinking_true",
			messages: []api.Message{
				{Role: "system", Content: "You are a pirate chatbot who always responds in pirate speak!"},
				{Role: "user", Content: "Give me a short introduction to LLMs."},
				{Role: "assistant", Content: "Arrr! I'm a pirate"},
				{Role: "user", Content: "Tell me more about LLMs."},
			},
			thinkValue: &api.ThinkValue{Value: true},
			expected: `<｜begin▁of▁sentence｜>Enable deep thinking subroutine.

You are Cogito, an AI assistant created by Deep Cogito, which is an AI research lab based in San Francisco.

You are a pirate chatbot who always responds in pirate speak!

<｜User｜>Give me a short introduction to LLMs.<｜Assistant｜>Arrr! I'm a pirate<｜end▁of▁sentence｜><｜User｜>Tell me more about LLMs.<｜Assistant｜><think>
`,
		},
		// tools
		{
			name: "tool_calls_only_no_content",
			messages: []api.Message{
				{Role: "user", Content: "Get weather for Paris"},
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
			expected: `<｜begin▁of▁sentence｜>You are Cogito, an AI assistant created by Deep Cogito, which is an AI research lab based in San Francisco.<｜User｜>Get weather for Paris<｜Assistant｜><｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather
` + "```json\n" + `{"location":"Paris"}
` + "```" + `<｜tool▁call▁end｜><｜tool▁calls▁end｜><｜end▁of▁sentence｜><｜Assistant｜>`,
		},
		{
			name: "complex_tool_arguments",
			messages: []api.Message{
				{Role: "user", Content: "Process complex data"},
				{
					Role: "assistant",
					ToolCalls: []api.ToolCall{
						{
							Function: api.ToolCallFunction{
								Name: "process_data",
								Arguments: testArgsOrdered([]orderedArg{
									{"config", map[string]any{
										"enabled":   true,
										"threshold": 0.95,
										"tags":      []string{"important", "urgent"},
									}},
									{"items", []any{"item1", "item2", "item3"}},
								}),
							},
						},
					},
				},
			},
			thinkValue: &api.ThinkValue{Value: false},
			expected: `<｜begin▁of▁sentence｜>You are Cogito, an AI assistant created by Deep Cogito, which is an AI research lab based in San Francisco.<｜User｜>Process complex data<｜Assistant｜><｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>process_data
` + "```json\n" + `{"config":{"enabled":true,"tags":["important","urgent"],"threshold":0.95},"items":["item1","item2","item3"]}
` + "```" + `<｜tool▁call▁end｜><｜tool▁calls▁end｜><｜end▁of▁sentence｜><｜Assistant｜>`,
		},
		{
			name: "empty_messages",
			messages: []api.Message{
				{Role: "system", Content: ""},
				{Role: "user", Content: "Hello"},
				{Role: "assistant", Content: ""},
			},
			thinkValue: &api.ThinkValue{Value: false},
			expected:   `<｜begin▁of▁sentence｜>You are Cogito, an AI assistant created by Deep Cogito, which is an AI research lab based in San Francisco.<｜User｜>Hello<｜Assistant｜><｜end▁of▁sentence｜><｜Assistant｜>`,
		},
		{
			name: "thinking_with_empty_assistant_content",
			messages: []api.Message{
				{Role: "user", Content: "Think about this"},
				{Role: "assistant", Content: ""},
			},
			thinkValue: &api.ThinkValue{Value: true},
			expected: `<｜begin▁of▁sentence｜>Enable deep thinking subroutine.

You are Cogito, an AI assistant created by Deep Cogito, which is an AI research lab based in San Francisco.<｜User｜>Think about this<｜Assistant｜><｜end▁of▁sentence｜><｜Assistant｜><think>
`,
		},
		{
			name: "multiple_system_messages",
			messages: []api.Message{
				{Role: "system", Content: "First instruction"},
				{Role: "system", Content: "Second instruction"},
				{Role: "user", Content: "Hello"},
			},
			thinkValue: &api.ThinkValue{Value: false},
			expected: `<｜begin▁of▁sentence｜>You are Cogito, an AI assistant created by Deep Cogito, which is an AI research lab based in San Francisco.

First instruction<｜User｜>Hello<｜Assistant｜>`,
		},
		{
			name: "special_characters_in_content",
			messages: []api.Message{
				{Role: "user", Content: "What about <|special|> tokens and \"quotes\"?"},
				{Role: "assistant", Content: "They're handled normally in content."},
			},
			thinkValue: &api.ThinkValue{Value: false},
			expected:   `<｜begin▁of▁sentence｜>You are Cogito, an AI assistant created by Deep Cogito, which is an AI research lab based in San Francisco.<｜User｜>What about <|special|> tokens and "quotes"?<｜Assistant｜>They're handled normally in content.<｜end▁of▁sentence｜><｜Assistant｜>`,
		},
		{
			name: "long_conversation_multiple_rounds",
			messages: []api.Message{
				{Role: "user", Content: "Hi"},
				{Role: "assistant", Content: "Hello!"},
				{Role: "user", Content: "How are you?"},
				{Role: "assistant", Content: "Good, thanks!"},
				{Role: "user", Content: "What's the weather?"},
			},
			thinkValue: &api.ThinkValue{Value: false},
			expected:   `<｜begin▁of▁sentence｜>You are Cogito, an AI assistant created by Deep Cogito, which is an AI research lab based in San Francisco.<｜User｜>Hi<｜Assistant｜>Hello!<｜end▁of▁sentence｜><｜User｜>How are you?<｜Assistant｜>Good, thanks!<｜end▁of▁sentence｜><｜User｜>What's the weather?<｜Assistant｜>`,
		},
	}

	renderer := &CogitoRenderer{isThinking: true}
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
