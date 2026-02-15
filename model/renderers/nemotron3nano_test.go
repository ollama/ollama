package renderers

import (
	"testing"

	"github.com/google/go-cmp/cmp"

	"github.com/ollama/ollama/api"
)

func TestNemotron3NanoRenderer(t *testing.T) {
	tests := []struct {
		name       string
		msgs       []api.Message
		tools      []api.Tool
		thinkValue *api.ThinkValue
		expected   string
	}{
		{
			name: "basic user message - thinking mode",
			msgs: []api.Message{
				{Role: "user", Content: "Hello!"},
			},
			thinkValue: &api.ThinkValue{Value: true},
			expected: "<|im_start|>system\n<|im_end|>\n" +
				"<|im_start|>user\nHello!<|im_end|>\n" +
				"<|im_start|>assistant\n<think>\n",
		},
		{
			name: "basic user message - no thinking",
			msgs: []api.Message{
				{Role: "user", Content: "Hello!"},
			},
			thinkValue: nil,
			expected: "<|im_start|>system\n<|im_end|>\n" +
				"<|im_start|>user\nHello!<|im_end|>\n" +
				"<|im_start|>assistant\n<think></think>",
		},
		{
			name: "with system message",
			msgs: []api.Message{
				{Role: "system", Content: "You are a helpful assistant."},
				{Role: "user", Content: "Hello!"},
			},
			thinkValue: &api.ThinkValue{Value: true},
			expected: "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n" +
				"<|im_start|>user\nHello!<|im_end|>\n" +
				"<|im_start|>assistant\n<think>\n",
		},
		{
			name: "multi-turn conversation",
			msgs: []api.Message{
				{Role: "user", Content: "Hi"},
				{Role: "assistant", Content: "Hello! How can I help?"},
				{Role: "user", Content: "Tell me a joke"},
			},
			thinkValue: &api.ThinkValue{Value: true},
			expected: "<|im_start|>system\n<|im_end|>\n" +
				"<|im_start|>user\nHi<|im_end|>\n" +
				"<|im_start|>assistant\n<think></think>Hello! How can I help?<|im_end|>\n" +
				"<|im_start|>user\nTell me a joke<|im_end|>\n" +
				"<|im_start|>assistant\n<think>\n",
		},
		{
			name: "with tools",
			msgs: []api.Message{
				{Role: "user", Content: "What's the weather in Paris?"},
			},
			tools: []api.Tool{
				{
					Type: "function",
					Function: api.ToolFunction{
						Name:        "get_weather",
						Description: "Get the current weather",
						Parameters: api.ToolFunctionParameters{
							Type:     "object",
							Required: []string{"city"},
							Properties: testPropsMap(map[string]api.ToolProperty{
								"city": {Type: api.PropertyType{"string"}, Description: "The city name"},
							}),
						},
					},
				},
			},
			thinkValue: &api.ThinkValue{Value: true},
			expected: "<|im_start|>system\n" +
				"# Tools\n\nYou have access to the following functions:\n\n<tools>\n" +
				"<function>\n<name>get_weather</name>\n" +
				"<description>Get the current weather</description>\n" +
				"<parameters>\n" +
				"<parameter>\n<name>city</name>\n<type>string</type>\n<description>The city name</description>\n</parameter>\n" +
				"<required>[\"city\"]</required>\n" +
				"</parameters>\n</function>\n</tools>\n\n" +
				"If you choose to call a function ONLY reply in the following format with NO suffix:\n\n" +
				"<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n" +
				"<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n" +
				"</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n" +
				"- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n" +
				"- Required parameters MUST be specified\n" +
				"- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n" +
				"- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n" +
				"</IMPORTANT><|im_end|>\n" +
				"<|im_start|>user\nWhat's the weather in Paris?<|im_end|>\n" +
				"<|im_start|>assistant\n<think>\n",
		},
		{
			name: "tool call with response",
			msgs: []api.Message{
				{Role: "user", Content: "What's the weather in Paris?"},
				{
					Role: "assistant",
					ToolCalls: []api.ToolCall{
						{
							Function: api.ToolCallFunction{
								Name:      "get_weather",
								Arguments: testArgs(map[string]any{"city": "Paris"}),
							},
						},
					},
				},
				{Role: "tool", Content: "Sunny, 72F"},
			},
			tools: []api.Tool{
				{
					Type: "function",
					Function: api.ToolFunction{
						Name:        "get_weather",
						Description: "Get the current weather",
						Parameters: api.ToolFunctionParameters{
							Type:     "object",
							Required: []string{"city"},
							Properties: testPropsMap(map[string]api.ToolProperty{
								"city": {Type: api.PropertyType{"string"}, Description: "The city name"},
							}),
						},
					},
				},
			},
			thinkValue: &api.ThinkValue{Value: true},
			expected: "<|im_start|>system\n" +
				"# Tools\n\nYou have access to the following functions:\n\n<tools>\n" +
				"<function>\n<name>get_weather</name>\n" +
				"<description>Get the current weather</description>\n" +
				"<parameters>\n" +
				"<parameter>\n<name>city</name>\n<type>string</type>\n<description>The city name</description>\n</parameter>\n" +
				"<required>[\"city\"]</required>\n" +
				"</parameters>\n</function>\n</tools>\n\n" +
				"If you choose to call a function ONLY reply in the following format with NO suffix:\n\n" +
				"<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n" +
				"<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n" +
				"</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n" +
				"- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n" +
				"- Required parameters MUST be specified\n" +
				"- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n" +
				"- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n" +
				"</IMPORTANT><|im_end|>\n" +
				"<|im_start|>user\nWhat's the weather in Paris?<|im_end|>\n" +
				"<|im_start|>assistant\n<think></think>\n" +
				"<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>\n<|im_end|>\n" +
				"<|im_start|>user\n<tool_response>\nSunny, 72F\n</tool_response>\n<|im_end|>\n" +
				"<|im_start|>assistant\n<think>\n",
		},
		{
			name: "assistant with content and tool call",
			msgs: []api.Message{
				{Role: "user", Content: "What's the weather?"},
				{
					Role:    "assistant",
					Content: "Let me check that for you.",
					ToolCalls: []api.ToolCall{
						{
							Function: api.ToolCallFunction{
								Name:      "get_weather",
								Arguments: testArgs(map[string]any{"city": "Paris"}),
							},
						},
					},
				},
				{Role: "tool", Content: "Sunny"},
			},
			tools: []api.Tool{
				{
					Type: "function",
					Function: api.ToolFunction{
						Name: "get_weather",
						Parameters: api.ToolFunctionParameters{
							Type: "object",
							Properties: testPropsMap(map[string]api.ToolProperty{
								"city": {Type: api.PropertyType{"string"}},
							}),
						},
					},
				},
			},
			thinkValue: &api.ThinkValue{Value: true},
			expected: "<|im_start|>system\n" +
				"# Tools\n\nYou have access to the following functions:\n\n<tools>\n" +
				"<function>\n<name>get_weather</name>\n" +
				"<parameters>\n" +
				"<parameter>\n<name>city</name>\n<type>string</type>\n</parameter>\n" +
				"</parameters>\n</function>\n</tools>\n\n" +
				"If you choose to call a function ONLY reply in the following format with NO suffix:\n\n" +
				"<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n" +
				"<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n" +
				"</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n" +
				"- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n" +
				"- Required parameters MUST be specified\n" +
				"- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n" +
				"- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n" +
				"</IMPORTANT><|im_end|>\n" +
				"<|im_start|>user\nWhat's the weather?<|im_end|>\n" +
				"<|im_start|>assistant\n<think></think>Let me check that for you.\n" +
				"<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>\n<|im_end|>\n" +
				"<|im_start|>user\n<tool_response>\nSunny\n</tool_response>\n<|im_end|>\n" +
				"<|im_start|>assistant\n<think>\n",
		},
		{
			name: "thinking in history is truncated",
			msgs: []api.Message{
				{Role: "user", Content: "Hi"},
				{Role: "assistant", Content: "Hello!", Thinking: "Let me think about this..."},
				{Role: "user", Content: "How are you?"},
			},
			thinkValue: &api.ThinkValue{Value: true},
			expected: "<|im_start|>system\n<|im_end|>\n" +
				"<|im_start|>user\nHi<|im_end|>\n" +
				"<|im_start|>assistant\n<think></think>Hello!<|im_end|>\n" +
				"<|im_start|>user\nHow are you?<|im_end|>\n" +
				"<|im_start|>assistant\n<think>\n",
		},
		{
			name: "parallel tool calls",
			msgs: []api.Message{
				{Role: "user", Content: "Weather in Paris and London?"},
				{
					Role: "assistant",
					ToolCalls: []api.ToolCall{
						{
							Function: api.ToolCallFunction{
								Name:      "get_weather",
								Arguments: testArgs(map[string]any{"city": "Paris"}),
							},
						},
						{
							Function: api.ToolCallFunction{
								Name:      "get_weather",
								Arguments: testArgs(map[string]any{"city": "London"}),
							},
						},
					},
				},
				{Role: "tool", Content: "Sunny"},
				{Role: "tool", Content: "Rainy"},
			},
			tools: []api.Tool{
				{
					Type: "function",
					Function: api.ToolFunction{
						Name: "get_weather",
						Parameters: api.ToolFunctionParameters{
							Type: "object",
							Properties: testPropsMap(map[string]api.ToolProperty{
								"city": {Type: api.PropertyType{"string"}},
							}),
						},
					},
				},
			},
			thinkValue: &api.ThinkValue{Value: true},
			expected: "<|im_start|>system\n" +
				"# Tools\n\nYou have access to the following functions:\n\n<tools>\n" +
				"<function>\n<name>get_weather</name>\n" +
				"<parameters>\n" +
				"<parameter>\n<name>city</name>\n<type>string</type>\n</parameter>\n" +
				"</parameters>\n</function>\n</tools>\n\n" +
				"If you choose to call a function ONLY reply in the following format with NO suffix:\n\n" +
				"<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n" +
				"<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n" +
				"</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n" +
				"- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n" +
				"- Required parameters MUST be specified\n" +
				"- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n" +
				"- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n" +
				"</IMPORTANT><|im_end|>\n" +
				"<|im_start|>user\nWeather in Paris and London?<|im_end|>\n" +
				"<|im_start|>assistant\n<think></think>\n" +
				"<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>\n" +
				"<tool_call>\n<function=get_weather>\n<parameter=city>\nLondon\n</parameter>\n</function>\n</tool_call>\n<|im_end|>\n" +
				"<|im_start|>user\n<tool_response>\nSunny\n</tool_response>\n<tool_response>\nRainy\n</tool_response>\n<|im_end|>\n" +
				"<|im_start|>assistant\n<think>\n",
		},
		{
			name: "thinking disabled when user doesn't request it",
			msgs: []api.Message{
				{Role: "user", Content: "Hello!"},
			},
			thinkValue: nil,
			expected: "<|im_start|>system\n<|im_end|>\n" +
				"<|im_start|>user\nHello!<|im_end|>\n" +
				"<|im_start|>assistant\n<think></think>",
		},
		{
			name: "complex message history with thinking, tools, tool calls, tool results and content",
			msgs: []api.Message{
				{Role: "user", Content: "What's the weather in Paris and London? Also, what's 2+2?"},
				{Role: "assistant", Content: "", Thinking: "I need to check the weather for both cities and calculate 2+2. Let me start with the weather calls.", ToolCalls: []api.ToolCall{
					{Function: api.ToolCallFunction{Name: "get_weather", Arguments: testArgs(map[string]any{"city": "Paris"})}},
					{Function: api.ToolCallFunction{Name: "get_weather", Arguments: testArgs(map[string]any{"city": "London"})}},
				}},
				{Role: "tool", Content: "Sunny, 22°C", ToolCallID: "call1"},
				{Role: "tool", Content: "Rainy, 15°C", ToolCallID: "call2"},
				{Role: "assistant", Content: "", Thinking: "Now I have the weather data. Let me calculate 2+2.", ToolCalls: []api.ToolCall{
					{Function: api.ToolCallFunction{Name: "calculate", Arguments: testArgs(map[string]any{"expression": "2+2"})}},
				}},
				{Role: "tool", Content: "4", ToolCallID: "call3"},
				{Role: "assistant", Content: "Based on the weather data, Paris is sunny at 22°C and London is rainy at 15°C. Also, 2+2 equals 4.", Thinking: "Perfect! I have all the information needed to provide a complete answer."},
			},
			tools: []api.Tool{
				{
					Type: "function",
					Function: api.ToolFunction{
						Name: "get_weather",
						Parameters: api.ToolFunctionParameters{
							Type: "object",
							Properties: testPropsMap(map[string]api.ToolProperty{
								"city": {Type: api.PropertyType{"string"}},
							}),
						},
					},
				},
				{
					Type: "function",
					Function: api.ToolFunction{
						Name: "calculate",
						Parameters: api.ToolFunctionParameters{
							Type: "object",
							Properties: testPropsMap(map[string]api.ToolProperty{
								"expression": {Type: api.PropertyType{"string"}},
							}),
						},
					},
				},
			},
			thinkValue: &api.ThinkValue{Value: true},
			expected: "<|im_start|>system\n" +
				"# Tools\n\nYou have access to the following functions:\n\n<tools>\n" +
				"<function>\n<name>get_weather</name>\n" +
				"<parameters>\n" +
				"<parameter>\n<name>city</name>\n<type>string</type>\n</parameter>\n" +
				"</parameters>\n</function>\n" +
				"<function>\n<name>calculate</name>\n" +
				"<parameters>\n" +
				"<parameter>\n<name>expression</name>\n<type>string</type>\n</parameter>\n" +
				"</parameters>\n</function>\n</tools>\n\n" +
				"If you choose to call a function ONLY reply in the following format with NO suffix:\n\n" +
				"<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n" +
				"<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n" +
				"</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n" +
				"- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n" +
				"- Required parameters MUST be specified\n" +
				"- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n" +
				"- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n" +
				"</IMPORTANT><|im_end|>\n" +
				"<|im_start|>user\nWhat's the weather in Paris and London? Also, what's 2+2?<|im_end|>\n" +
				"<|im_start|>assistant\n" +
				"<think>\nI need to check the weather for both cities and calculate 2+2. Let me start with the weather calls.\n</think>\n" +
				"<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>\n" +
				"<tool_call>\n<function=get_weather>\n<parameter=city>\nLondon\n</parameter>\n</function>\n</tool_call>\n<|im_end|>\n" +
				"<|im_start|>user\n<tool_response>\nSunny, 22°C\n</tool_response>\n<tool_response>\nRainy, 15°C\n</tool_response>\n<|im_end|>\n" +
				"<|im_start|>assistant\n" +
				"<think>\nNow I have the weather data. Let me calculate 2+2.\n</think>\n" +
				"<tool_call>\n<function=calculate>\n<parameter=expression>\n2+2\n</parameter>\n</function>\n</tool_call>\n<|im_end|>\n" +
				"<|im_start|>user\n<tool_response>\n4\n</tool_response>\n<|im_end|>\n" +
				"<|im_start|>assistant\n" +
				"<think>\nPerfect! I have all the information needed to provide a complete answer.\n</think>\n" +
				"Based on the weather data, Paris is sunny at 22°C and London is rainy at 15°C. Also, 2+2 equals 4.<|im_end|>\n" +
				"<|im_start|>assistant\n<think>\n",
		},
		{
			name:       "empty messages list",
			msgs:       []api.Message{},
			thinkValue: nil,
			expected:   "<|im_start|>system\n<|im_end|>\n<|im_start|>assistant\n<think></think>",
		},
		{
			name: "tool result with JSON content",
			msgs: []api.Message{
				{Role: "user", Content: "Get user info"},
				{
					Role: "assistant",
					ToolCalls: []api.ToolCall{
						{Function: api.ToolCallFunction{Name: "get_user", Arguments: testArgs(map[string]any{"id": "123"})}},
					},
				},
				{Role: "tool", Content: `{"name": "John", "age": 30, "active": true}`},
			},
			tools: []api.Tool{
				{
					Type: "function",
					Function: api.ToolFunction{
						Name: "get_user",
						Parameters: api.ToolFunctionParameters{
							Type:       "object",
							Properties: testPropsMap(map[string]api.ToolProperty{"id": {Type: api.PropertyType{"string"}}}),
						},
					},
				},
			},
			thinkValue: &api.ThinkValue{Value: true},
			expected: "<|im_start|>system\n" +
				"# Tools\n\nYou have access to the following functions:\n\n<tools>\n" +
				"<function>\n<name>get_user</name>\n<parameters>\n" +
				"<parameter>\n<name>id</name>\n<type>string</type>\n</parameter>\n" +
				"</parameters>\n</function>\n</tools>\n\n" +
				"If you choose to call a function ONLY reply in the following format with NO suffix:\n\n" +
				"<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n" +
				"<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n" +
				"</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n" +
				"- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n" +
				"- Required parameters MUST be specified\n" +
				"- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n" +
				"- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n" +
				"</IMPORTANT><|im_end|>\n" +
				"<|im_start|>user\nGet user info<|im_end|>\n" +
				"<|im_start|>assistant\n<think></think>\n" +
				"<tool_call>\n<function=get_user>\n<parameter=id>\n123\n</parameter>\n</function>\n</tool_call>\n<|im_end|>\n" +
				"<|im_start|>user\n<tool_response>\n{\"name\": \"John\", \"age\": 30, \"active\": true}\n</tool_response>\n<|im_end|>\n" +
				"<|im_start|>assistant\n<think>\n",
		},
		{
			name: "assistant message with only thinking no content",
			msgs: []api.Message{
				{Role: "user", Content: "Think about this"},
				{Role: "assistant", Thinking: "Deep thoughts here...", Content: ""},
				{Role: "user", Content: "What did you think?"},
			},
			thinkValue: &api.ThinkValue{Value: true},
			expected: "<|im_start|>system\n<|im_end|>\n" +
				"<|im_start|>user\nThink about this<|im_end|>\n" +
				"<|im_start|>assistant\n<think></think><|im_end|>\n" +
				"<|im_start|>user\nWhat did you think?<|im_end|>\n" +
				"<|im_start|>assistant\n<think>\n",
		},
		{
			name: "tool call with complex nested argument",
			msgs: []api.Message{
				{Role: "user", Content: "Create data"},
				{
					Role: "assistant",
					ToolCalls: []api.ToolCall{
						{Function: api.ToolCallFunction{
							Name: "create",
							Arguments: testArgs(map[string]any{
								"data": map[string]any{"nested": "value", "count": 42},
							}),
						}},
					},
				},
				{Role: "tool", Content: "Created"},
			},
			tools: []api.Tool{
				{
					Type: "function",
					Function: api.ToolFunction{
						Name: "create",
						Parameters: api.ToolFunctionParameters{
							Type:       "object",
							Properties: testPropsMap(map[string]api.ToolProperty{"data": {Type: api.PropertyType{"object"}}}),
						},
					},
				},
			},
			thinkValue: &api.ThinkValue{Value: true},
			expected: "<|im_start|>system\n" +
				"# Tools\n\nYou have access to the following functions:\n\n<tools>\n" +
				"<function>\n<name>create</name>\n<parameters>\n" +
				"<parameter>\n<name>data</name>\n<type>object</type>\n</parameter>\n" +
				"</parameters>\n</function>\n</tools>\n\n" +
				"If you choose to call a function ONLY reply in the following format with NO suffix:\n\n" +
				"<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n" +
				"<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n" +
				"</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n" +
				"- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n" +
				"- Required parameters MUST be specified\n" +
				"- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n" +
				"- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n" +
				"</IMPORTANT><|im_end|>\n" +
				"<|im_start|>user\nCreate data<|im_end|>\n" +
				"<|im_start|>assistant\n<think></think>\n" +
				"<tool_call>\n<function=create>\n<parameter=data>\n{\"count\":42,\"nested\":\"value\"}\n</parameter>\n</function>\n</tool_call>\n<|im_end|>\n" +
				"<|im_start|>user\n<tool_response>\nCreated\n</tool_response>\n<|im_end|>\n" +
				"<|im_start|>assistant\n<think>\n",
		},
		{
			name: "content explaining the format itself",
			msgs: []api.Message{
				{Role: "user", Content: "How do I format a tool call?"},
				{Role: "assistant", Content: "To call a tool, use <tool_call> tags with <function=name> inside."},
				{Role: "user", Content: "Thanks!"},
			},
			thinkValue: &api.ThinkValue{Value: true},
			expected: "<|im_start|>system\n<|im_end|>\n" +
				"<|im_start|>user\nHow do I format a tool call?<|im_end|>\n" +
				"<|im_start|>assistant\n<think></think>To call a tool, use <tool_call> tags with <function=name> inside.<|im_end|>\n" +
				"<|im_start|>user\nThanks!<|im_end|>\n" +
				"<|im_start|>assistant\n<think>\n",
		},
		{
			name: "unicode in content and tool args",
			msgs: []api.Message{
				{Role: "user", Content: "Translate 你好"},
				{
					Role: "assistant",
					ToolCalls: []api.ToolCall{
						{Function: api.ToolCallFunction{Name: "translate", Arguments: testArgs(map[string]any{"text": "你好"})}},
					},
				},
				{Role: "tool", Content: "Hello"},
			},
			tools: []api.Tool{
				{
					Type: "function",
					Function: api.ToolFunction{
						Name: "translate",
						Parameters: api.ToolFunctionParameters{
							Type: "object",
							Properties: testPropsMap(map[string]api.ToolProperty{
								"text": {Type: api.PropertyType{"string"}},
							}),
						},
					},
				},
			},
			thinkValue: &api.ThinkValue{Value: true},
			expected: "<|im_start|>system\n" +
				"# Tools\n\nYou have access to the following functions:\n\n<tools>\n" +
				"<function>\n<name>translate</name>\n<parameters>\n" +
				"<parameter>\n<name>text</name>\n<type>string</type>\n</parameter>\n" +
				"</parameters>\n</function>\n</tools>\n\n" +
				"If you choose to call a function ONLY reply in the following format with NO suffix:\n\n" +
				"<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n" +
				"<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n" +
				"</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n" +
				"- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n" +
				"- Required parameters MUST be specified\n" +
				"- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n" +
				"- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n" +
				"</IMPORTANT><|im_end|>\n" +
				"<|im_start|>user\nTranslate 你好<|im_end|>\n" +
				"<|im_start|>assistant\n<think></think>\n" +
				"<tool_call>\n<function=translate>\n<parameter=text>\n你好\n</parameter>\n</function>\n</tool_call>\n<|im_end|>\n" +
				"<|im_start|>user\n<tool_response>\nHello\n</tool_response>\n<|im_end|>\n" +
				"<|im_start|>assistant\n<think>\n",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			renderer := &Nemotron3NanoRenderer{}
			rendered, err := renderer.Render(tt.msgs, tt.tools, tt.thinkValue)
			if err != nil {
				t.Fatal(err)
			}
			if diff := cmp.Diff(rendered, tt.expected); diff != "" {
				t.Errorf("mismatch (-got +want):\n%s", diff)
			}
		})
	}
}
