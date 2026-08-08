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
								Arguments: testArgs(map[string]any{
									"location": "Paris",
								}),
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
								Arguments: testArgs(map[string]any{
									"location": "Paris",
								}),
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
			expected:   `<｜begin▁of▁sentence｜><｜User｜>Get weather for Paris and London<｜Assistant｜></think><｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>{"location":"Paris"}<｜tool▁call▁end｜><｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>{"location":"London"}<｜tool▁call▁end｜><｜tool▁calls▁end｜><｜end▁of▁sentence｜><｜tool▁output▁begin｜>Paris: 22°C, Sunny<｜tool▁output▁end｜><｜tool▁output▁begin｜>London: 18°C, Cloudy<｜tool▁output▁end｜>`,
		},
		{
			name: "literal </think> in content is preserved",
			messages: []api.Message{
				{Role: "user", Content: "Think about this"},
				{Role: "assistant", Content: "I'm thinking about this.</think>The answer is 42."},
			},
			thinkValue: &api.ThinkValue{Value: false},
			expected:   `<｜begin▁of▁sentence｜><｜User｜>Think about this<｜Assistant｜></think>I'm thinking about this.</think>The answer is 42.<｜end▁of▁sentence｜>`,
		},
		{
			name: "replayed reasoning block in content is stripped",
			messages: []api.Message{
				{Role: "user", Content: "Think about this"},
				{Role: "assistant", Content: "<think>I'm thinking about this.</think>The answer is 42."},
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
								Arguments: testArgs(map[string]any{
									"location": "Paris",
								}),
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
								Arguments: testArgs(map[string]any{
									"data": "test",
								}),
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
			name: "multiple literal think tags in content are preserved",
			messages: []api.Message{
				{Role: "user", Content: "Complex question"},
				{Role: "assistant", Content: "First thought</think>Second thought</think>Final answer"},
			},
			thinkValue: &api.ThinkValue{Value: false},
			expected:   `<｜begin▁of▁sentence｜><｜User｜>Complex question<｜Assistant｜></think>First thought</think>Second thought</think>Final answer<｜end▁of▁sentence｜>`,
		},
		{
			name: "replayed reasoning block with literal </think> after it",
			messages: []api.Message{
				{Role: "user", Content: "Complex question"},
				{Role: "assistant", Content: "<think>First thought</think>Second thought</think>Final answer"},
			},
			thinkValue: &api.ThinkValue{Value: false},
			expected:   `<｜begin▁of▁sentence｜><｜User｜>Complex question<｜Assistant｜></think>Second thought</think>Final answer<｜end▁of▁sentence｜>`,
		},
		{
			name: "thinking enabled after tool call - should render thinking",
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
				{Role: "assistant", Content: "Based on the weather data, it's sunny in Paris."},
				{Role: "user", Content: "Now tell me about London weather too."},
			},
			thinkValue: &api.ThinkValue{Value: true},
			expected:   `<｜begin▁of▁sentence｜><｜User｜>What's the weather in Paris?<｜Assistant｜></think><｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>{"location":"Paris"}<｜tool▁call▁end｜><｜tool▁calls▁end｜><｜end▁of▁sentence｜><｜tool▁output▁begin｜>Temperature: 22°C, Sunny<｜tool▁output▁end｜>Based on the weather data, it's sunny in Paris.<｜end▁of▁sentence｜><｜User｜>Now tell me about London weather too.<｜Assistant｜><think>`,
		},
		{
			name: "thinking disabled after tool call - should not render thinking",
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
				{Role: "assistant", Content: "Based on the weather data, it's sunny in Paris."},
				{Role: "user", Content: "Now tell me about London weather too."},
			},
			thinkValue: &api.ThinkValue{Value: false},
			expected:   `<｜begin▁of▁sentence｜><｜User｜>What's the weather in Paris?<｜Assistant｜></think><｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>{"location":"Paris"}<｜tool▁call▁end｜><｜tool▁calls▁end｜><｜end▁of▁sentence｜><｜tool▁output▁begin｜>Temperature: 22°C, Sunny<｜tool▁output▁end｜>Based on the weather data, it's sunny in Paris.<｜end▁of▁sentence｜><｜User｜>Now tell me about London weather too.<｜Assistant｜></think>`,
		},
		{
			name: "thinking enabled but messages without thinking content",
			messages: []api.Message{
				{Role: "user", Content: "First question about cats"},
				{Role: "assistant", Content: "Cats are wonderful pets."},
				{Role: "user", Content: "What about dogs?"},
				{Role: "assistant", Content: "Dogs are loyal companions."},
				{Role: "user", Content: "Final question about birds"},
			},
			thinkValue: &api.ThinkValue{Value: true},
			expected:   `<｜begin▁of▁sentence｜><｜User｜>First question about cats<｜Assistant｜></think>Cats are wonderful pets.<｜end▁of▁sentence｜><｜User｜>What about dogs?<｜Assistant｜></think>Dogs are loyal companions.<｜end▁of▁sentence｜><｜User｜>Final question about birds<｜Assistant｜><think>`,
		},
		{
			name: "thinking disabled for all assistant responses",
			messages: []api.Message{
				{Role: "user", Content: "First question about cats"},
				{Role: "assistant", Content: "Cats are wonderful pets."},
				{Role: "user", Content: "What about dogs?"},
				{Role: "assistant", Content: "Dogs are loyal companions."},
				{Role: "user", Content: "Final question about birds"},
			},
			thinkValue: &api.ThinkValue{Value: false},
			expected:   `<｜begin▁of▁sentence｜><｜User｜>First question about cats<｜Assistant｜></think>Cats are wonderful pets.<｜end▁of▁sentence｜><｜User｜>What about dogs?<｜Assistant｜></think>Dogs are loyal companions.<｜end▁of▁sentence｜><｜User｜>Final question about birds<｜Assistant｜></think>`,
		},
		{
			name: "complex conversation with tool calls and thinking enabled",
			messages: []api.Message{
				{Role: "user", Content: "Tell me about the weather"},
				{Role: "assistant", Content: "I'll check the weather for you."},
				{Role: "user", Content: "Actually, get Paris weather specifically"},
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
				{Role: "tool", Content: "Paris: 22°C, Sunny"},
				{Role: "assistant", Content: "The weather in Paris is great!"},
				{Role: "user", Content: "What about the forecast for tomorrow?"},
			},
			thinkValue: &api.ThinkValue{Value: true},
			expected:   `<｜begin▁of▁sentence｜><｜User｜>Tell me about the weather<｜Assistant｜></think>I'll check the weather for you.<｜end▁of▁sentence｜><｜User｜>Actually, get Paris weather specifically<｜Assistant｜></think><｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>{"location":"Paris"}<｜tool▁call▁end｜><｜tool▁calls▁end｜><｜end▁of▁sentence｜><｜tool▁output▁begin｜>Paris: 22°C, Sunny<｜tool▁output▁end｜>The weather in Paris is great!<｜end▁of▁sentence｜><｜User｜>What about the forecast for tomorrow?<｜Assistant｜><think>`,
		},
		{
			name: "tool call without subsequent user message - no thinking",
			messages: []api.Message{
				{Role: "user", Content: "Get the weather"},
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
				{Role: "tool", Content: "22°C, Sunny"},
			},
			thinkValue: &api.ThinkValue{Value: true},
			expected:   `<｜begin▁of▁sentence｜><｜User｜>Get the weather<｜Assistant｜></think><｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>{"location":"Paris"}<｜tool▁call▁end｜><｜tool▁calls▁end｜><｜end▁of▁sentence｜><｜tool▁output▁begin｜>22°C, Sunny<｜tool▁output▁end｜>`,
		},
		{
			name: "messages with thinking content, no thinking in render",
			messages: []api.Message{
				{Role: "user", Content: "Solve this math problem: 15 * 23"},
				{
					Role:     "assistant",
					Content:  "The answer is 345.",
					Thinking: "Let me calculate 15 * 23. I can break this down: 15 * 20 = 300, and 15 * 3 = 45, so 300 + 45 = 345.",
				},
				{Role: "user", Content: "What about 12 * 34?"},
			},
			thinkValue: &api.ThinkValue{Value: false},
			expected:   `<｜begin▁of▁sentence｜><｜User｜>Solve this math problem: 15 * 23<｜Assistant｜></think>The answer is 345.<｜end▁of▁sentence｜><｜User｜>What about 12 * 34?<｜Assistant｜></think>`,
		},
		{
			name: "conversation with mix of thinking and no thinking",
			messages: []api.Message{
				{Role: "user", Content: "Explain quantum physics"},
				{
					Role:     "assistant",
					Content:  "Quantum physics is the study of matter and energy at the smallest scales.",
					Thinking: "This is a complex topic. I should start with basic concepts and avoid overwhelming technical details.",
				},
				{Role: "user", Content: "What about photons?"},
				{
					Role:    "assistant",
					Content: "Photons are particles of light with no mass.",
				},
				{Role: "user", Content: "How do they interact with matter?"},
			},
			thinkValue: &api.ThinkValue{Value: true},
			expected:   `<｜begin▁of▁sentence｜><｜User｜>Explain quantum physics<｜Assistant｜></think>Quantum physics is the study of matter and energy at the smallest scales.<｜end▁of▁sentence｜><｜User｜>What about photons?<｜Assistant｜></think>Photons are particles of light with no mass.<｜end▁of▁sentence｜><｜User｜>How do they interact with matter?<｜Assistant｜><think>`,
		},
		{
			name: "tool call with thinking content in response",
			messages: []api.Message{
				{Role: "user", Content: "What's the weather in Tokyo and New York?"},
				{
					Role:     "assistant",
					Content:  "I'll check the weather for both cities.",
					Thinking: "I need to call the weather API for two different cities. Let me make parallel calls.",
					ToolCalls: []api.ToolCall{
						{
							Function: api.ToolCallFunction{
								Name: "get_weather",
								Arguments: testArgs(map[string]any{
									"location": "Tokyo",
								}),
							},
						},
						{
							Function: api.ToolCallFunction{
								Name: "get_weather",
								Arguments: testArgs(map[string]any{
									"location": "New York",
								}),
							},
						},
					},
				},
				{Role: "tool", Content: "Tokyo: 18°C, Cloudy"},
				{Role: "tool", Content: "New York: 22°C, Sunny"},
				{
					Role:     "assistant",
					Content:  "Based on the weather data: Tokyo is cloudy at 18°C, while New York is sunny at 22°C.",
					Thinking: "The data shows a nice contrast between the two cities. Tokyo is cooler and overcast while NYC has better weather.",
				},
			},
			thinkValue: &api.ThinkValue{Value: true},
			expected:   `<｜begin▁of▁sentence｜><｜User｜>What's the weather in Tokyo and New York?<｜Assistant｜></think>I'll check the weather for both cities.<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>{"location":"Tokyo"}<｜tool▁call▁end｜><｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>{"location":"New York"}<｜tool▁call▁end｜><｜tool▁calls▁end｜><｜end▁of▁sentence｜><｜tool▁output▁begin｜>Tokyo: 18°C, Cloudy<｜tool▁output▁end｜><｜tool▁output▁begin｜>New York: 22°C, Sunny<｜tool▁output▁end｜>Based on the weather data: Tokyo is cloudy at 18°C, while New York is sunny at 22°C.<｜end▁of▁sentence｜>`,
		},
		{
			name: "empty thinking field",
			messages: []api.Message{
				{Role: "user", Content: "Simple question"},
				{
					Role:     "assistant",
					Content:  "Simple answer.",
					Thinking: "", // Empty thinking content
				},
			},
			thinkValue: &api.ThinkValue{Value: true},
			expected:   `<｜begin▁of▁sentence｜><｜User｜>Simple question<｜Assistant｜></think>Simple answer.<｜end▁of▁sentence｜>`,
		},
		{
			name: "with tools definitions",
			messages: []api.Message{
				{Role: "system", Content: "You are a helpful assistant."},
				{Role: "user", Content: "What's the weather like?"},
			},
			tools: []api.Tool{
				{
					Type: "function",
					Function: api.ToolFunction{
						Name:        "get_weather",
						Description: "Get current weather information",
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
			expected: `<｜begin▁of▁sentence｜>You are a helpful assistant.

## Tools
You have access to the following tools:

### get_weather
Description: Get current weather information

Parameters: {"type":"object","required":["location"],"properties":{"location":{"type":"string","description":"City name"}}}

IMPORTANT: ALWAYS adhere to this exact format for tool use:
<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>tool_call_name<｜tool▁sep｜>tool_call_arguments<｜tool▁call▁end｜>{{additional_tool_calls}}<｜tool▁calls▁end｜>

Where:

- ` + "`tool_call_name`" + ` must be an exact match to one of the available tools
- ` + "`tool_call_arguments`" + ` must be valid JSON that strictly follows the tool's Parameters Schema
- For multiple tool calls, chain them directly without separators or spaces
<｜User｜>What's the weather like?<｜Assistant｜></think>`,
		},
		{
			name: "tools definitions with thinking enabled",
			messages: []api.Message{
				{Role: "system", Content: "You are a helpful assistant."},
				{Role: "user", Content: "What's the weather in Paris?"},
			},
			tools: []api.Tool{
				{
					Type: "function",
					Function: api.ToolFunction{
						Name:        "get_weather",
						Description: "Get current weather information",
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
			expected: `<｜begin▁of▁sentence｜>You are a helpful assistant.

## Tools
You have access to the following tools:

### get_weather
Description: Get current weather information

Parameters: {"type":"object","required":["location"],"properties":{"location":{"type":"string","description":"City name"}}}

IMPORTANT: ALWAYS adhere to this exact format for tool use:
<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>tool_call_name<｜tool▁sep｜>tool_call_arguments<｜tool▁call▁end｜>{{additional_tool_calls}}<｜tool▁calls▁end｜>

Where:

- ` + "`tool_call_name`" + ` must be an exact match to one of the available tools
- ` + "`tool_call_arguments`" + ` must be valid JSON that strictly follows the tool's Parameters Schema
- For multiple tool calls, chain them directly without separators or spaces
<｜User｜>What's the weather in Paris?<｜Assistant｜><think>`,
		},
		{
			name: "tools definitions with actual tool call",
			messages: []api.Message{
				{Role: "system", Content: "You are a helpful assistant."},
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
			},
			tools: []api.Tool{
				{
					Type: "function",
					Function: api.ToolFunction{
						Name:        "get_weather",
						Description: "Get current weather information",
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
			expected: `<｜begin▁of▁sentence｜>You are a helpful assistant.

## Tools
You have access to the following tools:

### get_weather
Description: Get current weather information

Parameters: {"type":"object","required":["location"],"properties":{"location":{"type":"string","description":"City name"}}}

IMPORTANT: ALWAYS adhere to this exact format for tool use:
<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>tool_call_name<｜tool▁sep｜>tool_call_arguments<｜tool▁call▁end｜>{{additional_tool_calls}}<｜tool▁calls▁end｜>

Where:

- ` + "`tool_call_name`" + ` must be an exact match to one of the available tools
- ` + "`tool_call_arguments`" + ` must be valid JSON that strictly follows the tool's Parameters Schema
- For multiple tool calls, chain them directly without separators or spaces
<｜User｜>What's the weather in Paris?<｜Assistant｜></think><｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>{"location":"Paris"}<｜tool▁call▁end｜><｜tool▁calls▁end｜><｜end▁of▁sentence｜>`,
		},
		{
			name: "tools definitions with full conversation cycle",
			messages: []api.Message{
				{Role: "system", Content: "You are a helpful assistant."},
				{Role: "user", Content: "What's the weather in Paris?"},
				{
					Role:    "assistant",
					Content: "I'll check the weather for you.",
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
				{Role: "assistant", Content: "The weather in Paris is 22°C and sunny!"},
			},
			tools: []api.Tool{
				{
					Type: "function",
					Function: api.ToolFunction{
						Name:        "get_weather",
						Description: "Get current weather information",
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
			expected: `<｜begin▁of▁sentence｜>You are a helpful assistant.

## Tools
You have access to the following tools:

### get_weather
Description: Get current weather information

Parameters: {"type":"object","required":["location"],"properties":{"location":{"type":"string","description":"City name"}}}

IMPORTANT: ALWAYS adhere to this exact format for tool use:
<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>tool_call_name<｜tool▁sep｜>tool_call_arguments<｜tool▁call▁end｜>{{additional_tool_calls}}<｜tool▁calls▁end｜>

Where:

- ` + "`tool_call_name`" + ` must be an exact match to one of the available tools
- ` + "`tool_call_arguments`" + ` must be valid JSON that strictly follows the tool's Parameters Schema
- For multiple tool calls, chain them directly without separators or spaces
<｜User｜>What's the weather in Paris?<｜Assistant｜></think>I'll check the weather for you.<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>{"location":"Paris"}<｜tool▁call▁end｜><｜tool▁calls▁end｜><｜end▁of▁sentence｜><｜tool▁output▁begin｜>Temperature: 22°C, Sunny<｜tool▁output▁end｜>The weather in Paris is 22°C and sunny!<｜end▁of▁sentence｜>`,
		},
		{
			name: "tools with thinking and full conversation",
			messages: []api.Message{
				{Role: "system", Content: "You are a helpful assistant."},
				{Role: "user", Content: "Check the weather in Tokyo"},
				{
					Role:     "assistant",
					Thinking: "The user wants weather info for Tokyo. I should use the get_weather tool.",
					Content:  "Let me check that for you.",
					ToolCalls: []api.ToolCall{
						{
							Function: api.ToolCallFunction{
								Name: "get_weather",
								Arguments: testArgs(map[string]any{
									"location": "Tokyo",
								}),
							},
						},
					},
				},
				{Role: "tool", Content: "Temperature: 18°C, Cloudy"},
				{
					Role:     "assistant",
					Thinking: "The weather data shows it's cloudy and cool. I should present this clearly.",
					Content:  "In Tokyo, it's currently 18°C and cloudy.",
				},
				{Role: "user", Content: "What about London?"},
			},
			tools: []api.Tool{
				{
					Type: "function",
					Function: api.ToolFunction{
						Name:        "get_weather",
						Description: "Get current weather information",
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
			expected: `<｜begin▁of▁sentence｜>You are a helpful assistant.

## Tools
You have access to the following tools:

### get_weather
Description: Get current weather information

Parameters: {"type":"object","required":["location"],"properties":{"location":{"type":"string","description":"City name"}}}

IMPORTANT: ALWAYS adhere to this exact format for tool use:
<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>tool_call_name<｜tool▁sep｜>tool_call_arguments<｜tool▁call▁end｜>{{additional_tool_calls}}<｜tool▁calls▁end｜>

Where:

- ` + "`tool_call_name`" + ` must be an exact match to one of the available tools
- ` + "`tool_call_arguments`" + ` must be valid JSON that strictly follows the tool's Parameters Schema
- For multiple tool calls, chain them directly without separators or spaces
<｜User｜>Check the weather in Tokyo<｜Assistant｜></think>Let me check that for you.<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>{"location":"Tokyo"}<｜tool▁call▁end｜><｜tool▁calls▁end｜><｜end▁of▁sentence｜><｜tool▁output▁begin｜>Temperature: 18°C, Cloudy<｜tool▁output▁end｜>In Tokyo, it's currently 18°C and cloudy.<｜end▁of▁sentence｜><｜User｜>What about London?<｜Assistant｜><think>`,
		},
		{
			name: "multiple tools definitions",
			messages: []api.Message{
				{Role: "system", Content: "You are a helpful assistant with access to multiple tools."},
				{Role: "user", Content: "What can you help me with?"},
			},
			tools: []api.Tool{
				{
					Type: "function",
					Function: api.ToolFunction{
						Name:        "get_weather",
						Description: "Get current weather information",
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
				{
					Type: "function",
					Function: api.ToolFunction{
						Name:        "calculate",
						Description: "Perform mathematical calculations",
						Parameters: api.ToolFunctionParameters{
							Type: "object",
							Properties: testPropsMap(map[string]api.ToolProperty{
								"expression": {
									Type:        api.PropertyType{"string"},
									Description: "Mathematical expression to evaluate",
								},
							}),
							Required: []string{"expression"},
						},
					},
				},
			},
			thinkValue: &api.ThinkValue{Value: false},
			expected: `<｜begin▁of▁sentence｜>You are a helpful assistant with access to multiple tools.

## Tools
You have access to the following tools:

### get_weather
Description: Get current weather information

Parameters: {"type":"object","required":["location"],"properties":{"location":{"type":"string","description":"City name"}}}

### calculate
Description: Perform mathematical calculations

Parameters: {"type":"object","required":["expression"],"properties":{"expression":{"type":"string","description":"Mathematical expression to evaluate"}}}

IMPORTANT: ALWAYS adhere to this exact format for tool use:
<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>tool_call_name<｜tool▁sep｜>tool_call_arguments<｜tool▁call▁end｜>{{additional_tool_calls}}<｜tool▁calls▁end｜>

Where:

- ` + "`tool_call_name`" + ` must be an exact match to one of the available tools
- ` + "`tool_call_arguments`" + ` must be valid JSON that strictly follows the tool's Parameters Schema
- For multiple tool calls, chain them directly without separators or spaces
<｜User｜>What can you help me with?<｜Assistant｜></think>`,
		},
		{
			name: "multiple tools with multiple tool calls",
			messages: []api.Message{
				{Role: "user", Content: "Get weather for Paris and calculate 25 * 4"},
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
								Name: "calculate",
								Arguments: testArgs(map[string]any{
									"expression": "25 * 4",
								}),
							},
						},
					},
				},
				{Role: "tool", Content: "Temperature: 22°C, Sunny"},
				{Role: "tool", Content: "Result: 100"},
			},
			tools: []api.Tool{
				{
					Type: "function",
					Function: api.ToolFunction{
						Name:        "get_weather",
						Description: "Get current weather information",
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
				{
					Type: "function",
					Function: api.ToolFunction{
						Name:        "calculate",
						Description: "Perform mathematical calculations",
						Parameters: api.ToolFunctionParameters{
							Type: "object",
							Properties: testPropsMap(map[string]api.ToolProperty{
								"expression": {
									Type:        api.PropertyType{"string"},
									Description: "Mathematical expression to evaluate",
								},
							}),
							Required: []string{"expression"},
						},
					},
				},
			},
			thinkValue: &api.ThinkValue{Value: false},
			expected: `<｜begin▁of▁sentence｜>

## Tools
You have access to the following tools:

### get_weather
Description: Get current weather information

Parameters: {"type":"object","required":["location"],"properties":{"location":{"type":"string","description":"City name"}}}

### calculate
Description: Perform mathematical calculations

Parameters: {"type":"object","required":["expression"],"properties":{"expression":{"type":"string","description":"Mathematical expression to evaluate"}}}

IMPORTANT: ALWAYS adhere to this exact format for tool use:
<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>tool_call_name<｜tool▁sep｜>tool_call_arguments<｜tool▁call▁end｜>{{additional_tool_calls}}<｜tool▁calls▁end｜>

Where:

- ` + "`tool_call_name`" + ` must be an exact match to one of the available tools
- ` + "`tool_call_arguments`" + ` must be valid JSON that strictly follows the tool's Parameters Schema
- For multiple tool calls, chain them directly without separators or spaces
<｜User｜>Get weather for Paris and calculate 25 * 4<｜Assistant｜></think><｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>{"location":"Paris"}<｜tool▁call▁end｜><｜tool▁call▁begin｜>calculate<｜tool▁sep｜>{"expression":"25 * 4"}<｜tool▁call▁end｜><｜tool▁calls▁end｜><｜end▁of▁sentence｜><｜tool▁output▁begin｜>Temperature: 22°C, Sunny<｜tool▁output▁end｜><｜tool▁output▁begin｜>Result: 100<｜tool▁output▁end｜>`,
		},
		{
			name: "tools without system message",
			messages: []api.Message{
				{Role: "user", Content: "What's the weather?"},
			},
			tools: []api.Tool{
				{
					Type: "function",
					Function: api.ToolFunction{
						Name:        "get_weather",
						Description: "Get current weather information",
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
			expected: `<｜begin▁of▁sentence｜>

## Tools
You have access to the following tools:

### get_weather
Description: Get current weather information

Parameters: {"type":"object","required":["location"],"properties":{"location":{"type":"string","description":"City name"}}}

IMPORTANT: ALWAYS adhere to this exact format for tool use:
<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>tool_call_name<｜tool▁sep｜>tool_call_arguments<｜tool▁call▁end｜>{{additional_tool_calls}}<｜tool▁calls▁end｜>

Where:

- ` + "`tool_call_name`" + ` must be an exact match to one of the available tools
- ` + "`tool_call_arguments`" + ` must be valid JSON that strictly follows the tool's Parameters Schema
- For multiple tool calls, chain them directly without separators or spaces
<｜User｜>What's the weather?<｜Assistant｜></think>`,
		},
		{
			name: "multi-turn conversation with thinking content on each turn",
			messages: []api.Message{
				{Role: "user", Content: "hey!"},
				{
					Role:     "assistant",
					Content:  "Hey! 😊 How's it going? What's on your mind today?",
					Thinking: "Hmm, the user just said \"hey!\" which is a simple greeting. This is a straightforward opening where they're likely just starting a conversation or testing the interaction.",
				},
				{Role: "user", Content: "fantastic, how has yours been"},
				{
					Role:     "assistant",
					Content:  "Glad to hear you're having a fantastic day! That's awesome.\n\nMine's been great, thanks for asking! Just buzzing along, helping people out and having conversations like this one. So what's making your day so fantastic? Anything fun happening?",
					Thinking: "Ah, the user is responding warmly and asking about my \"day.\" Since I'm an AI, I need to gently remind them I don't experience time like a human, but frame it positively to keep the conversation flowing.",
				},
				{Role: "user", Content: "awesome, can you tell me a 10 word story?"},
			},
			thinkValue: &api.ThinkValue{Value: true},
			expected: `<｜begin▁of▁sentence｜><｜User｜>hey!<｜Assistant｜></think>Hey! 😊 How's it going? What's on your mind today?<｜end▁of▁sentence｜><｜User｜>fantastic, how has yours been<｜Assistant｜></think>Glad to hear you're having a fantastic day! That's awesome.

Mine's been great, thanks for asking! Just buzzing along, helping people out and having conversations like this one. So what's making your day so fantastic? Anything fun happening?<｜end▁of▁sentence｜><｜User｜>awesome, can you tell me a 10 word story?<｜Assistant｜><think>`,
		},
		{
			name: "vLLM documentation example - multi-turn with full thinking content",
			messages: []api.Message{
				{Role: "system", Content: "You are a helpful assistant"},
				{Role: "user", Content: "Who are you?"},
				{
					Role:     "assistant",
					Content:  "I am DeepSeek",
					Thinking: "Hmm",
				},
				{Role: "user", Content: "9.11 and 9.8, which is greater?"},
				{
					Role:     "assistant",
					Content:  "9.8 is greater than 9.11.\n\nTo compare them easily, you can align the decimal places:  \n- 9.11 has a tenths digit of 1 and a hundredths digit of 1.  \n- 9.8 can be written as 9.80, which has a tenths digit of 8 and a hundredths digit of 0.  \n\nSince the whole number part (9) is the same, compare the tenths place: 8 is greater than 1, so 9.80 (or 9.8) is greater than 9.11.",
					Thinking: "First, the user is asking which number is greater between 9.11 and 9.8. These are decimal numbers.\n\nI need to compare 9.11 and 9.8. To make it easier, I should think of them as decimals with the same number of decimal places.\n\n9.11 has two decimal places, and 9.8 has one decimal place. I can write 9.8 as 9.80 to make it comparable.\n\nSo, 9.11 versus 9.80.\n\nNow, comparing the whole number parts: both have 9, so they are equal in the units place.\n\nNext, compare the tenths place: for 9.11, the tenths digit is 1. For 9.80, the tenths digit is 8. Since 8 is greater than 1, 9.80 is greater than 9.11.\n\nTherefore, 9.8 is greater than 9.11.\n\nI can also think of them as fractions: 9.11 is 911/100, and 9.8 is 98/10 or 980/100. Comparing 911/100 and 980/100, 980/100 is larger, so 9.8 is greater.\n\nSo, the answer should be that 9.8 is greater than 9.11.\n\nNow, I need to respond helpfully. Since the user might be learning or need clarification, I should explain briefly.\n\nFinally, my response should be in English, as the query is in English.",
				},
				{Role: "user", Content: "Thanks! Can you tell me a 10 word story?"},
				{
					Role:     "assistant",
					Content:  "Of course. Here is a 10-word story:\n\nHe found the key, unlocking a door to forgotten memories.",
					Thinking: "Hmm, the user just asked for a 10-word story after the previous number comparison question. This is a quick creative task with a strict word count constraint. \n\nThe story needs to be exactly 10 words while maintaining coherence and a hint of narrative. A micro-story about finding a key could work - it implies a larger unseen story. \n\nChecking the word count: \"He found the key, unlocking a door to forgotten memories.\" That's 10 words with a beginning, middle and implied end. It fits the requirement while leaving room for imagination.",
				},
				{Role: "user", Content: "That was beautiful! Now can you write a haiku?"},
			},
			thinkValue: &api.ThinkValue{Value: true},
			expected: `<｜begin▁of▁sentence｜>You are a helpful assistant<｜User｜>Who are you?<｜Assistant｜></think>I am DeepSeek<｜end▁of▁sentence｜><｜User｜>9.11 and 9.8, which is greater?<｜Assistant｜></think>9.8 is greater than 9.11.

To compare them easily, you can align the decimal places:  
- 9.11 has a tenths digit of 1 and a hundredths digit of 1.  
- 9.8 can be written as 9.80, which has a tenths digit of 8 and a hundredths digit of 0.  

Since the whole number part (9) is the same, compare the tenths place: 8 is greater than 1, so 9.80 (or 9.8) is greater than 9.11.<｜end▁of▁sentence｜><｜User｜>Thanks! Can you tell me a 10 word story?<｜Assistant｜></think>Of course. Here is a 10-word story:

He found the key, unlocking a door to forgotten memories.<｜end▁of▁sentence｜><｜User｜>That was beautiful! Now can you write a haiku?<｜Assistant｜><think>`,
		},
		{
			name: "no system prompt - content with embedded thinking tags",
			messages: []api.Message{
				{Role: "user", Content: "Who are you?"},
				{Role: "assistant", Content: "<think>Hmm</think>I am DeepSeek"},
				{Role: "user", Content: "Thanks! Can you tell me a 10 word story?"},
			},
			thinkValue: &api.ThinkValue{Value: true},
			expected:   `<｜begin▁of▁sentence｜><｜User｜>Who are you?<｜Assistant｜></think>I am DeepSeek<｜end▁of▁sentence｜><｜User｜>Thanks! Can you tell me a 10 word story?<｜Assistant｜><think>`,
		},
	}

	renderer := &DeepSeek3Renderer{IsThinking: true, Variant: Deepseek31}
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
