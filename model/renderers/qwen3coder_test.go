package renderers

import (
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/ollama/ollama/api"
)

func TestQwen3CoderRenderer(t *testing.T) {
	tests := []struct {
		name     string
		msgs     []api.Message
		tools    []api.Tool
		expected string
	}{
		{
			name: "basic",
			msgs: []api.Message{
				{Role: "system", Content: "You are a helpful assistant."},
				{Role: "user", Content: "Hello, how are you?"},
			},
			expected: `<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Hello, how are you?<|im_end|>
<|im_start|>assistant
`,
		},
		{
			name: "with tools and response",
			msgs: []api.Message{
				{Role: "system", Content: "You are a helpful assistant with access to tools."},
				{Role: "user", Content: "What is the weather like in San Francisco?"},
				{
					Role:    "assistant",
					Content: "I'll check the weather in San Francisco for you.",
					ToolCalls: []api.ToolCall{
						{
							Function: api.ToolCallFunction{
								Name: "get_weather",
								Arguments: testArgs(map[string]any{
									"unit": "fahrenheit",
								}),
							},
						},
					},
				},
				{Role: "tool", Content: "{\"location\": \"San Francisco, CA\", \"temperature\": 68, \"condition\": \"partly cloudy\", \"humidity\": 65, \"wind_speed\": 12}", ToolName: "get_weather"},
				{Role: "user", Content: "That sounds nice! What about New York?"},
			},
			tools: []api.Tool{
				{Function: api.ToolFunction{
					Name:        "get_weather",
					Description: "Get the current weather in a given location",
					Parameters: api.ToolFunctionParameters{
						Required: []string{"unit"},
						Properties: testPropsMap(map[string]api.ToolProperty{
							"unit": {Type: api.PropertyType{"string"}, Enum: []any{"celsius", "fahrenheit"}, Description: "The unit of temperature"},
							// TODO(drifkin): add multiple params back once we have predictable
							// order via some sort of ordered map type (see
							// <https://github.com/ollama/ollama/issues/12244>)
							/*
								"location": {Type: api.PropertyType{"string"}, Description: "The city and state, e.g. San Francisco, CA"},
							*/
						}),
					},
				}},
			},
			expected: `<|im_start|>system
You are a helpful assistant with access to tools.

# Tools

You have access to the following functions:

<tools>
<function>
<name>get_weather</name>
<description>Get the current weather in a given location</description>
<parameters>
<parameter>
<name>unit</name>
<type>string</type>
<description>The unit of temperature</description>
<enum>["celsius","fahrenheit"]</enum>
</parameter>
<required>["unit"]</required>
</parameters>
</function>
</tools>

If you choose to call a function ONLY reply in the following format with NO suffix:

<tool_call>
<function=example_function_name>
<parameter=example_parameter_1>
value_1
</parameter>
<parameter=example_parameter_2>
This is the value for the second parameter
that can span
multiple lines
</parameter>
</function>
</tool_call>

<IMPORTANT>
Reminder:
- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags
- Required parameters MUST be specified
- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after
- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls
</IMPORTANT><|im_end|>
<|im_start|>user
What is the weather like in San Francisco?<|im_end|>
<|im_start|>assistant
I'll check the weather in San Francisco for you.

<tool_call>
<function=get_weather>
<parameter=unit>
fahrenheit
</parameter>
</function>
</tool_call><|im_end|>
<|im_start|>user
<tool_response>
{"location": "San Francisco, CA", "temperature": 68, "condition": "partly cloudy", "humidity": 65, "wind_speed": 12}
</tool_response>
<|im_end|>
<|im_start|>user
That sounds nice! What about New York?<|im_end|>
<|im_start|>assistant
`,
		},
		{
			name: "parallel tool calls",
			msgs: []api.Message{
				{Role: "system", Content: "You are a helpful assistant with access to tools."},
				{Role: "user", Content: "call double(1) and triple(2)"},
				{Role: "assistant", Content: "I'll call double(1) and triple(2) for you.", ToolCalls: []api.ToolCall{
					{Function: api.ToolCallFunction{Name: "double", Arguments: testArgs(map[string]any{"number": "1"})}},
					{Function: api.ToolCallFunction{Name: "triple", Arguments: testArgs(map[string]any{"number": "2"})}},
				}},
				{Role: "tool", Content: "{\"number\": 2}", ToolName: "double"},
				{Role: "tool", Content: "{\"number\": 6}", ToolName: "triple"},
			},
			tools: []api.Tool{
				{Function: api.ToolFunction{Name: "double", Description: "Double a number", Parameters: api.ToolFunctionParameters{Properties: testPropsMap(map[string]api.ToolProperty{
					"number": {Type: api.PropertyType{"string"}, Description: "The number to double"},
				})}}},
				{Function: api.ToolFunction{Name: "triple", Description: "Triple a number", Parameters: api.ToolFunctionParameters{Properties: testPropsMap(map[string]api.ToolProperty{
					"number": {Type: api.PropertyType{"string"}, Description: "The number to triple"},
				})}}},
			},
			expected: `<|im_start|>system
You are a helpful assistant with access to tools.

# Tools

You have access to the following functions:

<tools>
<function>
<name>double</name>
<description>Double a number</description>
<parameters>
<parameter>
<name>number</name>
<type>string</type>
<description>The number to double</description>
</parameter>
</parameters>
</function>
<function>
<name>triple</name>
<description>Triple a number</description>
<parameters>
<parameter>
<name>number</name>
<type>string</type>
<description>The number to triple</description>
</parameter>
</parameters>
</function>
</tools>

If you choose to call a function ONLY reply in the following format with NO suffix:

<tool_call>
<function=example_function_name>
<parameter=example_parameter_1>
value_1
</parameter>
<parameter=example_parameter_2>
This is the value for the second parameter
that can span
multiple lines
</parameter>
</function>
</tool_call>

<IMPORTANT>
Reminder:
- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags
- Required parameters MUST be specified
- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after
- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls
</IMPORTANT><|im_end|>
<|im_start|>user
call double(1) and triple(2)<|im_end|>
<|im_start|>assistant
I'll call double(1) and triple(2) for you.

<tool_call>
<function=double>
<parameter=number>
1
</parameter>
</function>
</tool_call>
<tool_call>
<function=triple>
<parameter=number>
2
</parameter>
</function>
</tool_call><|im_end|>
<|im_start|>user
<tool_response>
{"number": 2}
</tool_response>
<tool_response>
{"number": 6}
</tool_response>
<|im_end|>
<|im_start|>assistant
`,
		},
		{
			name: "prefill",
			msgs: []api.Message{
				{Role: "system", Content: "You are a helpful assistant."},
				{Role: "user", Content: "Tell me something interesting."},
				{Role: "assistant", Content: "I'll tell you something interesting about cats"},
			},
			expected: `<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Tell me something interesting.<|im_end|>
<|im_start|>assistant
I'll tell you something interesting about cats`,
		},
		{
			name: "complex tool call arguments should remain json encoded",
			msgs: []api.Message{
				{Role: "user", Content: "call tool"},
				{Role: "assistant", ToolCalls: []api.ToolCall{
					{Function: api.ToolCallFunction{
						Name: "echo",
						Arguments: testArgs(map[string]any{
							"payload": map[string]any{"foo": "bar"},
						}),
					}},
				}},
				{Role: "tool", Content: "{\"payload\": {\"foo\": \"bar\"}}", ToolName: "echo"},
			},
			expected: `<|im_start|>user
call tool<|im_end|>
<|im_start|>assistant

<tool_call>
<function=echo>
<parameter=payload>
{"foo":"bar"}
</parameter>
</function>
</tool_call><|im_end|>
<|im_start|>user
<tool_response>
{"payload": {"foo": "bar"}}
</tool_response>
<|im_end|>
<|im_start|>assistant
`,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			rendered, err := (&Qwen3CoderRenderer{}).Render(tt.msgs, tt.tools, nil)
			if err != nil {
				t.Fatal(err)
			}
			if diff := cmp.Diff(rendered, tt.expected); diff != "" {
				t.Errorf("mismatch (-got +want):\n%s", diff)
			}
		})
	}
}

func TestFormatToolCallArgument(t *testing.T) {
	tests := []struct {
		name     string
		arg      any
		expected string
	}{
		{
			name: "string",
			arg:  "foo",
			// notice no quotes around the string
			expected: "foo",
		},
		{
			name:     "map",
			arg:      map[string]any{"foo": "bar"},
			expected: "{\"foo\":\"bar\"}",
		},
		{
			name:     "number",
			arg:      1,
			expected: "1",
		},
		{
			name:     "boolean",
			arg:      true,
			expected: "true",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := formatToolCallArgument(tt.arg)
			if got != tt.expected {
				t.Errorf("formatToolCallArgument(%v) = %v, want %v", tt.arg, got, tt.expected)
			}
		})
	}
}

func TestQwen3ToolDefinitionTypes(t *testing.T) {
	tests := []struct {
		name         string
		propertyType api.PropertyType
		expected     string
	}{
		{
			name:         "simple",
			propertyType: api.PropertyType{"string"},
			expected:     "string",
		},
		{
			name:         "multiple",
			propertyType: api.PropertyType{"string", "number"},
			expected:     "[\"string\",\"number\"]",
		},
		{
			name:         "empty",
			propertyType: api.PropertyType{},
			expected:     "[]",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := formatToolDefinitionType(tt.propertyType)
			if got != tt.expected {
				t.Errorf("formatToolDefinitionType() = %v, want %v", got, tt.expected)
			}
		})
	}
}
