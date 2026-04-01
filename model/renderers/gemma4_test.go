package renderers

import (
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/stretchr/testify/assert"
)

func TestGemma4Renderer(t *testing.T) {
	q := `<|"|>` // string delimiter shorthand for readability

	tests := []struct {
		name     string
		messages []api.Message
		tools    []api.Tool
		think    *api.ThinkValue
		expected string
	}{
		{
			name: "basic_user_message",
			messages: []api.Message{
				{Role: "user", Content: "Hello!"},
			},
			expected: "<|turn>user\nHello!<turn|>\n<|turn>model\n",
		},
		{
			name: "with_system_message",
			messages: []api.Message{
				{Role: "system", Content: "You are helpful"},
				{Role: "user", Content: "Hello!"},
			},
			expected: "<|turn>system\nYou are helpful<turn|>\n<|turn>user\nHello!<turn|>\n<|turn>model\n",
		},
		{
			name: "with_developer_role",
			messages: []api.Message{
				{Role: "developer", Content: "You are a coding assistant"},
				{Role: "user", Content: "Hello!"},
			},
			expected: "<|turn>system\nYou are a coding assistant<turn|>\n<|turn>user\nHello!<turn|>\n<|turn>model\n",
		},
		{
			name: "multi_turn",
			messages: []api.Message{
				{Role: "user", Content: "Hi"},
				{Role: "assistant", Content: "Hello!"},
				{Role: "user", Content: "More"},
			},
			expected: "<|turn>user\nHi<turn|>\n<|turn>model\nHello!<turn|>\n<|turn>user\nMore<turn|>\n<|turn>model\n",
		},
		{
			name: "assistant_last_message_no_close",
			messages: []api.Message{
				{Role: "user", Content: "Hi"},
				{Role: "assistant", Content: "Hello!"},
			},
			expected: "<|turn>user\nHi<turn|>\n<|turn>model\nHello!",
		},
		{
			name:     "empty_messages",
			messages: []api.Message{},
			expected: "<|turn>model\n",
		},
		{
			name: "thinking_enabled",
			messages: []api.Message{
				{Role: "user", Content: "Think hard"},
			},
			think:    thinkTrue(),
			expected: "<|turn>system\n<|think|><turn|>\n<|turn>user\nThink hard<turn|>\n<|turn>model\n",
		},
		{
			name: "thinking_with_system",
			messages: []api.Message{
				{Role: "system", Content: "Be careful"},
				{Role: "user", Content: "Think hard"},
			},
			think:    thinkTrue(),
			expected: "<|turn>system\n<|think|>Be careful<turn|>\n<|turn>user\nThink hard<turn|>\n<|turn>model\n",
		},
		{
			// Tools with no system message — tool declarations follow immediately after system\n
			name: "with_tools",
			messages: []api.Message{
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
							Properties: testPropsMap(map[string]api.ToolProperty{
								"city": {Type: api.PropertyType{"string"}, Description: "City"},
							}),
						},
					},
				},
			},
			expected: "<|turn>system\n<|tool>declaration:get_weather{description:" + q + "Get weather" + q + ",parameters:{properties:{city:{description:" + q + "City" + q + ",type:" + q + "STRING" + q + "}},type:" + q + "OBJECT" + q + "}}<tool|><turn|>\n<|turn>user\nWeather?<turn|>\n<|turn>model\n",
		},
		{
			// System message with tools — tools follow directly after system content (no newline)
			name: "system_message_with_tools",
			messages: []api.Message{
				{Role: "system", Content: "You are a weather expert."},
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
							Properties: testPropsMap(map[string]api.ToolProperty{
								"city": {Type: api.PropertyType{"string"}, Description: "City"},
							}),
						},
					},
				},
			},
			expected: "<|turn>system\nYou are a weather expert.<|tool>declaration:get_weather{description:" + q + "Get weather" + q + ",parameters:{properties:{city:{description:" + q + "City" + q + ",type:" + q + "STRING" + q + "}},type:" + q + "OBJECT" + q + "}}<tool|><turn|>\n<|turn>user\nWeather?<turn|>\n<|turn>model\n",
		},
		{
			// Tool call + tool response: response is inline in the model turn, no separate <|turn>tool
			// Non-JSON tool response falls back to {value:<|"|>...<|"|>}
			name: "tool_call",
			messages: []api.Message{
				{Role: "user", Content: "Weather?"},
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
				{Role: "tool", Content: "Sunny"},
			},
			tools: []api.Tool{
				{
					Type: "function",
					Function: api.ToolFunction{
						Name:        "get_weather",
						Description: "Get weather",
						Parameters: api.ToolFunctionParameters{
							Type: "object",
							Properties: testPropsMap(map[string]api.ToolProperty{
								"city": {Type: api.PropertyType{"string"}, Description: "City"},
							}),
						},
					},
				},
			},
			expected: "<|turn>system\n<|tool>declaration:get_weather{description:" + q + "Get weather" + q + ",parameters:{properties:{city:{description:" + q + "City" + q + ",type:" + q + "STRING" + q + "}},type:" + q + "OBJECT" + q + "}}<tool|><turn|>\n" +
				"<|turn>user\nWeather?<turn|>\n" +
				"<|turn>model\n<|tool_call>call:get_weather{city:" + q + "Paris" + q + "}<tool_call|>" +
				"<|tool_response>response:get_weather{value:" + q + "Sunny" + q + "}<tool_response|>",
		},
		{
			// Assistant content + tool call + tool response inline
			name: "assistant_content_with_tool_call",
			messages: []api.Message{
				{Role: "user", Content: "Weather?"},
				{
					Role:    "assistant",
					Content: "Let me check.",
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
						Name:        "get_weather",
						Description: "Get weather",
						Parameters: api.ToolFunctionParameters{
							Type: "object",
							Properties: testPropsMap(map[string]api.ToolProperty{
								"city": {Type: api.PropertyType{"string"}, Description: "City"},
							}),
						},
					},
				},
			},
			expected: "<|turn>system\n<|tool>declaration:get_weather{description:" + q + "Get weather" + q + ",parameters:{properties:{city:{description:" + q + "City" + q + ",type:" + q + "STRING" + q + "}},type:" + q + "OBJECT" + q + "}}<tool|><turn|>\n" +
				"<|turn>user\nWeather?<turn|>\n" +
				"<|turn>model\nLet me check.<|tool_call>call:get_weather{city:" + q + "Paris" + q + "}<tool_call|>" +
				"<|tool_response>response:get_weather{value:" + q + "Sunny" + q + "}<tool_response|>",
		},
		{
			// Parallel tool calls — both responses inline
			name: "parallel_tool_calls",
			messages: []api.Message{
				{Role: "user", Content: "Weather and time?"},
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
								Name:      "get_time",
								Arguments: testArgs(map[string]any{"timezone": "UTC"}),
							},
						},
					},
				},
				{Role: "tool", Content: "Sunny"},
				{Role: "tool", Content: "12:00"},
			},
			tools: []api.Tool{
				{
					Type: "function",
					Function: api.ToolFunction{
						Name:        "get_weather",
						Description: "Get weather",
						Parameters: api.ToolFunctionParameters{
							Type: "object",
							Properties: testPropsMap(map[string]api.ToolProperty{
								"city": {Type: api.PropertyType{"string"}, Description: "City"},
							}),
						},
					},
				},
				{
					Type: "function",
					Function: api.ToolFunction{
						Name:        "get_time",
						Description: "Get current time",
						Parameters: api.ToolFunctionParameters{
							Type: "object",
							Properties: testPropsMap(map[string]api.ToolProperty{
								"timezone": {Type: api.PropertyType{"string"}, Description: "Timezone"},
							}),
						},
					},
				},
			},
			expected: "<|turn>system\n<|tool>declaration:get_weather{description:" + q + "Get weather" + q + ",parameters:{properties:{city:{description:" + q + "City" + q + ",type:" + q + "STRING" + q + "}},type:" + q + "OBJECT" + q + "}}<tool|><|tool>declaration:get_time{description:" + q + "Get current time" + q + ",parameters:{properties:{timezone:{description:" + q + "Timezone" + q + ",type:" + q + "STRING" + q + "}},type:" + q + "OBJECT" + q + "}}<tool|><turn|>\n" +
				"<|turn>user\nWeather and time?<turn|>\n" +
				"<|turn>model\n<|tool_call>call:get_weather{city:" + q + "Paris" + q + "}<tool_call|><|tool_call>call:get_time{timezone:" + q + "UTC" + q + "}<tool_call|>" +
				"<|tool_response>response:get_weather{value:" + q + "Sunny" + q + "}<tool_response|>" +
				"<|tool_response>response:get_time{value:" + q + "12:00" + q + "}<tool_response|>",
		},
		{
			// Numeric arguments — JSON tool response with individual key:value pairs
			name: "numeric_arguments",
			messages: []api.Message{
				{Role: "user", Content: "Add"},
				{
					Role: "assistant",
					ToolCalls: []api.ToolCall{
						{
							Function: api.ToolCallFunction{
								Name:      "add",
								Arguments: testArgs(map[string]any{"a": float64(1), "b": float64(2)}),
							},
						},
					},
				},
				{Role: "tool", Content: `{"result":3}`},
			},
			tools: []api.Tool{
				{
					Type: "function",
					Function: api.ToolFunction{
						Name:        "add",
						Description: "Add numbers",
						Parameters: api.ToolFunctionParameters{
							Type: "object",
							Properties: testPropsMap(map[string]api.ToolProperty{
								"a": {Type: api.PropertyType{"number"}},
								"b": {Type: api.PropertyType{"number"}},
							}),
						},
					},
				},
			},
			expected: "<|turn>system\n<|tool>declaration:add{description:" + q + "Add numbers" + q + ",parameters:{properties:{a:{type:" + q + "NUMBER" + q + "},b:{type:" + q + "NUMBER" + q + "}},type:" + q + "OBJECT" + q + "}}<tool|><turn|>\n" +
				"<|turn>user\nAdd<turn|>\n" +
				"<|turn>model\n<|tool_call>call:add{a:1,b:2}<tool_call|>" +
				"<|tool_response>response:add{result:3}<tool_response|>",
		},
		{
			// Boolean argument — non-JSON tool response
			name: "boolean_argument",
			messages: []api.Message{
				{Role: "user", Content: "Set flag"},
				{
					Role: "assistant",
					ToolCalls: []api.ToolCall{
						{
							Function: api.ToolCallFunction{
								Name:      "set_flag",
								Arguments: testArgs(map[string]any{"enabled": true}),
							},
						},
					},
				},
				{Role: "tool", Content: "done"},
			},
			tools: []api.Tool{
				{
					Type: "function",
					Function: api.ToolFunction{
						Name:        "set_flag",
						Description: "Set a flag",
						Parameters: api.ToolFunctionParameters{
							Type: "object",
							Properties: testPropsMap(map[string]api.ToolProperty{
								"enabled": {Type: api.PropertyType{"boolean"}, Description: "Flag value"},
							}),
						},
					},
				},
			},
			expected: "<|turn>system\n<|tool>declaration:set_flag{description:" + q + "Set a flag" + q + ",parameters:{properties:{enabled:{description:" + q + "Flag value" + q + ",type:" + q + "BOOLEAN" + q + "}},type:" + q + "OBJECT" + q + "}}<tool|><turn|>\n" +
				"<|turn>user\nSet flag<turn|>\n" +
				"<|turn>model\n<|tool_call>call:set_flag{enabled:true}<tool_call|>" +
				"<|tool_response>response:set_flag{value:" + q + "done" + q + "}<tool_response|>",
		},
		{
			name: "tool_with_required_params",
			messages: []api.Message{
				{Role: "user", Content: "Weather?"},
			},
			tools: []api.Tool{
				{
					Type: "function",
					Function: api.ToolFunction{
						Name:        "get_weather",
						Description: "Gets the weather for a given city",
						Parameters: api.ToolFunctionParameters{
							Type:     "object",
							Required: []string{"city"},
							Properties: testPropsMap(map[string]api.ToolProperty{
								"city":    {Type: api.PropertyType{"string"}, Description: "City Name"},
								"country": {Type: api.PropertyType{"string"}, Description: "Country Name"},
							}),
						},
					},
				},
			},
			expected: "<|turn>system\n<|tool>declaration:get_weather{description:" + q + "Gets the weather for a given city" + q + ",parameters:{properties:{city:{description:" + q + "City Name" + q + ",type:" + q + "STRING" + q + "},country:{description:" + q + "Country Name" + q + ",type:" + q + "STRING" + q + "}},required:[" + q + "city" + q + "],type:" + q + "OBJECT" + q + "}}<tool|><turn|>\n" +
				"<|turn>user\nWeather?<turn|>\n<|turn>model\n",
		},
		{
			name: "tool_with_enum",
			messages: []api.Message{
				{Role: "user", Content: "Test"},
			},
			tools: []api.Tool{
				{
					Type: "function",
					Function: api.ToolFunction{
						Name:        "set_mode",
						Description: "Set mode",
						Parameters: api.ToolFunctionParameters{
							Type: "object",
							Properties: testPropsMap(map[string]api.ToolProperty{
								"mode": {Type: api.PropertyType{"string"}, Description: "The mode", Enum: []any{"fast", "slow"}},
							}),
						},
					},
				},
			},
			expected: "<|turn>system\n<|tool>declaration:set_mode{description:" + q + "Set mode" + q + ",parameters:{properties:{mode:{description:" + q + "The mode" + q + ",enum:[" + q + "fast" + q + "," + q + "slow" + q + "],type:" + q + "STRING" + q + "}},type:" + q + "OBJECT" + q + "}}<tool|><turn|>\n" +
				"<|turn>user\nTest<turn|>\n<|turn>model\n",
		},
		{
			name: "unicode_content",
			messages: []api.Message{
				{Role: "user", Content: "こんにちは"},
			},
			expected: "<|turn>user\nこんにちは<turn|>\n<|turn>model\n",
		},
		{
			name: "newlines_in_content",
			messages: []api.Message{
				{Role: "user", Content: "Line 1\nLine 2\nLine 3"},
			},
			expected: "<|turn>user\nLine 1\nLine 2\nLine 3<turn|>\n<|turn>model\n",
		},
		{
			// Thinking + tools — <|think|> immediately followed by tool declarations
			name: "thinking_with_tools",
			messages: []api.Message{
				{Role: "user", Content: "Weather?"},
			},
			think: thinkTrue(),
			tools: []api.Tool{
				{
					Type: "function",
					Function: api.ToolFunction{
						Name:        "get_weather",
						Description: "Get weather",
						Parameters: api.ToolFunctionParameters{
							Type: "object",
							Properties: testPropsMap(map[string]api.ToolProperty{
								"city": {Type: api.PropertyType{"string"}, Description: "City"},
							}),
						},
					},
				},
			},
			expected: "<|turn>system\n<|think|><|tool>declaration:get_weather{description:" + q + "Get weather" + q + ",parameters:{properties:{city:{description:" + q + "City" + q + ",type:" + q + "STRING" + q + "}},type:" + q + "OBJECT" + q + "}}<tool|><turn|>\n" +
				"<|turn>user\nWeather?<turn|>\n<|turn>model\n",
		},
		{
			name: "image_tags_when_enabled",
			messages: []api.Message{
				{Role: "user", Content: "What is this?", Images: []api.ImageData{[]byte("fake")}},
			},
			expected: "<|turn>user\n[img-0]What is this?<turn|>\n<|turn>model\n",
		},
		{
			// JSON tool response — parsed into individual key:value pairs
			name: "json_tool_response",
			messages: []api.Message{
				{Role: "user", Content: "Weather?"},
				{
					Role: "assistant",
					ToolCalls: []api.ToolCall{
						{
							Function: api.ToolCallFunction{
								Name:      "get_weather",
								Arguments: testArgs(map[string]any{"city": "Tokyo"}),
							},
						},
					},
				},
				{Role: "tool", Content: `{"temperature":15,"weather":"sunny"}`},
				{Role: "user", Content: "Thanks!"},
			},
			tools: []api.Tool{
				{
					Type: "function",
					Function: api.ToolFunction{
						Name:        "get_weather",
						Description: "Get weather",
						Parameters: api.ToolFunctionParameters{
							Type: "object",
							Properties: testPropsMap(map[string]api.ToolProperty{
								"city": {Type: api.PropertyType{"string"}, Description: "City"},
							}),
						},
					},
				},
			},
			// Matches HF reference: tool response inline, JSON fields as key:value, no <turn|> before next user
			expected: "<|turn>system\n<|tool>declaration:get_weather{description:" + q + "Get weather" + q + ",parameters:{properties:{city:{description:" + q + "City" + q + ",type:" + q + "STRING" + q + "}},type:" + q + "OBJECT" + q + "}}<tool|><turn|>\n" +
				"<|turn>user\nWeather?<turn|>\n" +
				"<|turn>model\n<|tool_call>call:get_weather{city:" + q + "Tokyo" + q + "}<tool_call|>" +
				"<|tool_response>response:get_weather{temperature:15,weather:" + q + "sunny" + q + "}<tool_response|>" +
				"<|turn>user\nThanks!<turn|>\n" +
				"<|turn>model\n",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			renderer := &Gemma4Renderer{useImgTags: true}
			result, err := renderer.Render(tt.messages, tt.tools, tt.think)
			assert.NoError(t, err)
			assert.Equal(t, tt.expected, result)
		})
	}
}

func thinkTrue() *api.ThinkValue {
	return &api.ThinkValue{Value: true}
}
