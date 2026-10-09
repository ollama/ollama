package renderers

import (
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestFunctionGemmaRenderer(t *testing.T) {
	tests := []struct {
		name     string
		messages []api.Message
		tools    []api.Tool
		expected string
	}{
		{
			name: "basic_user_message",
			messages: []api.Message{
				{Role: "user", Content: "Hello!"},
			},
			expected: "<bos><start_of_turn>user\nHello!<end_of_turn>\n<start_of_turn>model\n",
		},
		{
			name: "with_system_message",
			messages: []api.Message{
				{Role: "system", Content: "You are helpful"},
				{Role: "user", Content: "Hello!"},
			},
			expected: "<bos><start_of_turn>developer\nYou are helpful<end_of_turn>\n<start_of_turn>user\nHello!<end_of_turn>\n<start_of_turn>model\n",
		},
		{
			name: "with_developer_role",
			messages: []api.Message{
				{Role: "developer", Content: "You are a coding assistant"},
				{Role: "user", Content: "Hello!"},
			},
			expected: "<bos><start_of_turn>developer\nYou are a coding assistant<end_of_turn>\n<start_of_turn>user\nHello!<end_of_turn>\n<start_of_turn>model\n",
		},
		{
			name: "custom_system_message_with_tools",
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
			// Custom system message is preserved, tools are appended
			expected: "<bos><start_of_turn>developer\nYou are a weather expert.\nYou can do function calling with the following functions:<start_function_declaration>declaration:get_weather{description:<escape>Get weather<escape>,parameters:{properties:{city:{description:<escape>City<escape>,type:<escape>STRING<escape>}},type:<escape>OBJECT<escape>}}<end_function_declaration><end_of_turn>\n<start_of_turn>user\nWeather?<end_of_turn>\n<start_of_turn>model\n",
		},
		{
			name: "developer_role_with_tools",
			messages: []api.Message{
				{Role: "developer", Content: "Be concise."},
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
			// Developer role message is preserved, tools are appended
			expected: "<bos><start_of_turn>developer\nBe concise.\nYou can do function calling with the following functions:<start_function_declaration>declaration:get_weather{description:<escape>Get weather<escape>,parameters:{properties:{city:{description:<escape>City<escape>,type:<escape>STRING<escape>}},type:<escape>OBJECT<escape>}}<end_function_declaration><end_of_turn>\n<start_of_turn>user\nWeather?<end_of_turn>\n<start_of_turn>model\n",
		},
		{
			name: "multi_turn",
			messages: []api.Message{
				{Role: "user", Content: "Hi"},
				{Role: "assistant", Content: "Hello!"},
				{Role: "user", Content: "More"},
			},
			expected: "<bos><start_of_turn>user\nHi<end_of_turn>\n<start_of_turn>model\nHello!<end_of_turn>\n<start_of_turn>user\nMore<end_of_turn>\n<start_of_turn>model\n",
		},
		{
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
			expected: "<bos><start_of_turn>developer\nYou can do function calling with the following functions:<start_function_declaration>declaration:get_weather{description:<escape>Get weather<escape>,parameters:{properties:{city:{description:<escape>City<escape>,type:<escape>STRING<escape>}},type:<escape>OBJECT<escape>}}<end_function_declaration><end_of_turn>\n<start_of_turn>user\nWeather?<end_of_turn>\n<start_of_turn>model\n",
		},
		{
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
			expected: "<bos><start_of_turn>developer\nYou can do function calling with the following functions:<start_function_declaration>declaration:get_weather{description:<escape>Get weather<escape>,parameters:{properties:{city:{description:<escape>City<escape>,type:<escape>STRING<escape>}},type:<escape>OBJECT<escape>}}<end_function_declaration><end_of_turn>\n<start_of_turn>user\nWeather?<end_of_turn>\n<start_of_turn>model\n<start_function_call>call:get_weather{city:<escape>Paris<escape>}<end_function_call><start_function_response>response:get_weather{<escape>Sunny<escape>}<end_function_response>",
		},
		{
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
			expected: "<bos><start_of_turn>developer\nYou can do function calling with the following functions:<start_function_declaration>declaration:get_weather{description:<escape>Get weather<escape>,parameters:{properties:{city:{description:<escape>City<escape>,type:<escape>STRING<escape>}},type:<escape>OBJECT<escape>}}<end_function_declaration><end_of_turn>\n<start_of_turn>user\nWeather?<end_of_turn>\n<start_of_turn>model\nLet me check.<start_function_call>call:get_weather{city:<escape>Paris<escape>}<end_function_call><start_function_response>response:get_weather{<escape>Sunny<escape>}<end_function_response>",
		},
		{
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
				{Role: "tool", Content: "3"},
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
			expected: "<bos><start_of_turn>developer\nYou can do function calling with the following functions:<start_function_declaration>declaration:add{description:<escape>Add numbers<escape>,parameters:{properties:{a:{description:<escape><escape>,type:<escape>NUMBER<escape>},b:{description:<escape><escape>,type:<escape>NUMBER<escape>}},type:<escape>OBJECT<escape>}}<end_function_declaration><end_of_turn>\n<start_of_turn>user\nAdd<end_of_turn>\n<start_of_turn>model\n<start_function_call>call:add{a:1,b:2}<end_function_call><start_function_response>response:add{<escape>3<escape>}<end_function_response>",
		},
		{
			name:     "empty_messages",
			messages: []api.Message{},
			expected: "<bos><start_of_turn>model\n",
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
			// Required params are escaped: required:[<escape>city<escape>]
			expected: "<bos><start_of_turn>developer\nYou can do function calling with the following functions:<start_function_declaration>declaration:get_weather{description:<escape>Gets the weather for a given city<escape>,parameters:{properties:{city:{description:<escape>City Name<escape>,type:<escape>STRING<escape>},country:{description:<escape>Country Name<escape>,type:<escape>STRING<escape>}},required:[<escape>city<escape>],type:<escape>OBJECT<escape>}}<end_function_declaration><end_of_turn>\n<start_of_turn>user\nWeather?<end_of_turn>\n<start_of_turn>model\n",
		},
		{
			name: "multiple_tools",
			messages: []api.Message{
				{Role: "user", Content: "Weather and time?"},
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
			// Multiple tool declarations are consecutive
			expected: "<bos><start_of_turn>developer\nYou can do function calling with the following functions:<start_function_declaration>declaration:get_weather{description:<escape>Get weather<escape>,parameters:{properties:{city:{description:<escape>City<escape>,type:<escape>STRING<escape>}},type:<escape>OBJECT<escape>}}<end_function_declaration><start_function_declaration>declaration:get_time{description:<escape>Get current time<escape>,parameters:{properties:{timezone:{description:<escape>Timezone<escape>,type:<escape>STRING<escape>}},type:<escape>OBJECT<escape>}}<end_function_declaration><end_of_turn>\n<start_of_turn>user\nWeather and time?<end_of_turn>\n<start_of_turn>model\n",
		},
		{
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
			// Multiple tool calls and responses are consecutive
			expected: "<bos><start_of_turn>developer\nYou can do function calling with the following functions:<start_function_declaration>declaration:get_weather{description:<escape>Get weather<escape>,parameters:{properties:{city:{description:<escape>City<escape>,type:<escape>STRING<escape>}},type:<escape>OBJECT<escape>}}<end_function_declaration><start_function_declaration>declaration:get_time{description:<escape>Get current time<escape>,parameters:{properties:{timezone:{description:<escape>Timezone<escape>,type:<escape>STRING<escape>}},type:<escape>OBJECT<escape>}}<end_function_declaration><end_of_turn>\n<start_of_turn>user\nWeather and time?<end_of_turn>\n<start_of_turn>model\n<start_function_call>call:get_weather{city:<escape>Paris<escape>}<end_function_call><start_function_call>call:get_time{timezone:<escape>UTC<escape>}<end_function_call><start_function_response>response:get_weather{<escape>Sunny<escape>}<end_function_response><start_function_response>response:get_time{<escape>12:00<escape>}<end_function_response>",
		},
		{
			name: "user_after_tool_response",
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
				{Role: "user", Content: "Thanks! What about London?"},
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
			// User message after tool response gets concatenated (user reverted to this behavior)
			expected: "<bos><start_of_turn>developer\nYou can do function calling with the following functions:<start_function_declaration>declaration:get_weather{description:<escape>Get weather<escape>,parameters:{properties:{city:{description:<escape>City<escape>,type:<escape>STRING<escape>}},type:<escape>OBJECT<escape>}}<end_function_declaration><end_of_turn>\n<start_of_turn>user\nWeather?<end_of_turn>\n<start_of_turn>model\n<start_function_call>call:get_weather{city:<escape>Paris<escape>}<end_function_call><start_function_response>response:get_weather{<escape>Sunny<escape>}<end_function_response>Thanks! What about London?<end_of_turn>\n<start_of_turn>model\n",
		},
		// Edge cases
		{
			name: "tool_empty_properties",
			messages: []api.Message{
				{Role: "user", Content: "Test"},
			},
			tools: []api.Tool{
				{
					Type: "function",
					Function: api.ToolFunction{
						Name:        "test_fn",
						Description: "",
						Parameters: api.ToolFunctionParameters{
							Type:       "object",
							Properties: testPropsMap(map[string]api.ToolProperty{}),
						},
					},
				},
			},
			// Empty properties are omitted
			expected: "<bos><start_of_turn>developer\nYou can do function calling with the following functions:<start_function_declaration>declaration:test_fn{description:<escape><escape>,parameters:{type:<escape>OBJECT<escape>}}<end_function_declaration><end_of_turn>\n<start_of_turn>user\nTest<end_of_turn>\n<start_of_turn>model\n",
		},
		{
			name: "unicode_content",
			messages: []api.Message{
				{Role: "user", Content: "こんにちは 🎉"},
			},
			expected: "<bos><start_of_turn>user\nこんにちは 🎉<end_of_turn>\n<start_of_turn>model\n",
		},
		{
			name: "newlines_in_content",
			messages: []api.Message{
				{Role: "user", Content: "Line 1\nLine 2\nLine 3"},
			},
			expected: "<bos><start_of_turn>user\nLine 1\nLine 2\nLine 3<end_of_turn>\n<start_of_turn>model\n",
		},
		{
			name: "special_chars_in_content",
			messages: []api.Message{
				{Role: "user", Content: "Test <tag> & \"quotes\" chars"},
			},
			expected: "<bos><start_of_turn>user\nTest <tag> & \"quotes\" chars<end_of_turn>\n<start_of_turn>model\n",
		},
		{
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
			expected: "<bos><start_of_turn>developer\nYou can do function calling with the following functions:<start_function_declaration>declaration:set_flag{description:<escape>Set a flag<escape>,parameters:{properties:{enabled:{description:<escape>Flag value<escape>,type:<escape>BOOLEAN<escape>}},type:<escape>OBJECT<escape>}}<end_function_declaration><end_of_turn>\n<start_of_turn>user\nSet flag<end_of_turn>\n<start_of_turn>model\n<start_function_call>call:set_flag{enabled:true}<end_function_call><start_function_response>response:set_flag{<escape>done<escape>}<end_function_response>",
		},
		{
			name: "multiple_required_params",
			messages: []api.Message{
				{Role: "user", Content: "Test"},
			},
			tools: []api.Tool{
				{
					Type: "function",
					Function: api.ToolFunction{
						Name:        "test",
						Description: "Test",
						Parameters: api.ToolFunctionParameters{
							Type:     "object",
							Required: []string{"a", "b", "c"},
							Properties: testPropsMap(map[string]api.ToolProperty{
								"a": {Type: api.PropertyType{"string"}, Description: "A"},
								"b": {Type: api.PropertyType{"string"}, Description: "B"},
								"c": {Type: api.PropertyType{"string"}, Description: "C"},
							}),
						},
					},
				},
			},
			expected: "<bos><start_of_turn>developer\nYou can do function calling with the following functions:<start_function_declaration>declaration:test{description:<escape>Test<escape>,parameters:{properties:{a:{description:<escape>A<escape>,type:<escape>STRING<escape>},b:{description:<escape>B<escape>,type:<escape>STRING<escape>},c:{description:<escape>C<escape>,type:<escape>STRING<escape>}},required:[<escape>a<escape>,<escape>b<escape>,<escape>c<escape>],type:<escape>OBJECT<escape>}}<end_function_declaration><end_of_turn>\n<start_of_turn>user\nTest<end_of_turn>\n<start_of_turn>model\n",
		},
		{
			name: "array_type_param",
			messages: []api.Message{
				{Role: "user", Content: "Test"},
			},
			tools: []api.Tool{
				{
					Type: "function",
					Function: api.ToolFunction{
						Name:        "test",
						Description: "Test",
						Parameters: api.ToolFunctionParameters{
							Type: "object",
							Properties: testPropsMap(map[string]api.ToolProperty{
								"items": {Type: api.PropertyType{"array"}, Description: "List of items"},
							}),
						},
					},
				},
			},
			expected: "<bos><start_of_turn>developer\nYou can do function calling with the following functions:<start_function_declaration>declaration:test{description:<escape>Test<escape>,parameters:{properties:{items:{description:<escape>List of items<escape>,type:<escape>ARRAY<escape>}},type:<escape>OBJECT<escape>}}<end_function_declaration><end_of_turn>\n<start_of_turn>user\nTest<end_of_turn>\n<start_of_turn>model\n",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			renderer := &FunctionGemmaRenderer{}
			result, err := renderer.Render(tt.messages, tt.tools, nil)
			assert.NoError(t, err)
			assert.Equal(t, tt.expected, result)
		})
	}
}

// TestFunctionGemmaRendererToolResultOrder verifies a tool result is attributed to
// the tool call named by its tool_call_id, not to whichever call happens to sit at
// the same position.
//
// The OpenAI-compatible API does not require the tool messages that follow a
// multi-call assistant turn to mirror the call order, so pairing by position
// silently swaps the payloads. Callers that send no tool_call_id (the /api/chat
// shape) keep the positional behaviour, which the first case pins down.
func TestFunctionGemmaRendererToolResultOrder(t *testing.T) {
	calls := []api.ToolCall{
		{ID: "call_tokyo", Function: api.ToolCallFunction{Name: "get_tokyo", Arguments: testArgs(map[string]any{"city": "Tokyo"})}},
		{ID: "call_toronto", Function: api.ToolCallFunction{Name: "get_toronto", Arguments: testArgs(map[string]any{"city": "Toronto"})}},
	}

	tests := []struct {
		name     string
		toolMsgs []api.Message
		want     []string
	}{
		{
			name: "reversed_results_match_their_call_id",
			toolMsgs: []api.Message{
				{Role: "tool", ToolName: "get_toronto", ToolCallID: "call_toronto", Content: "sunny"},
				{Role: "tool", ToolName: "get_tokyo", ToolCallID: "call_tokyo", Content: "snow"},
			},
			want: []string{"response:get_toronto", "response:get_tokyo"},
		},
		{
			name: "in_order_results_match_their_call_id",
			toolMsgs: []api.Message{
				{Role: "tool", ToolName: "get_tokyo", ToolCallID: "call_tokyo", Content: "snow"},
				{Role: "tool", ToolName: "get_toronto", ToolCallID: "call_toronto", Content: "sunny"},
			},
			want: []string{"response:get_tokyo", "response:get_toronto"},
		},
		{
			name: "results_without_call_ids_stay_positional",
			toolMsgs: []api.Message{
				{Role: "tool", Content: "snow"},
				{Role: "tool", Content: "sunny"},
			},
			want: []string{"response:get_tokyo", "response:get_toronto"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			messages := []api.Message{
				{Role: "user", Content: "weather?"},
				{Role: "assistant", ToolCalls: calls},
			}
			messages = append(messages, tt.toolMsgs...)

			got, err := (&FunctionGemmaRenderer{}).Render(messages, nil, nil)
			require.NoError(t, err)

			// Every response must still be rendered, and in the order supplied.
			assert.Equal(t, len(tt.want), strings.Count(got, "<start_function_response>"))
			prev := -1
			for _, want := range tt.want {
				idx := strings.Index(got, want)
				require.Greater(t, idx, prev, "response %q missing or out of order in:\n%s", want, got)
				prev = idx
			}
		})
	}
}
