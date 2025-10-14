package renderers

import (
	"testing"

	"github.com/google/go-cmp/cmp"

	"github.com/ollama/ollama/api"
)

// Helper function to create ordered arguments for tests
func makeArgs(pairs ...any) api.ToolCallFunctionArguments {
	args := api.NewToolCallFunctionArguments()
	for i := 0; i < len(pairs); i += 2 {
		key := pairs[i].(string)
		value := pairs[i+1]
		args.Set(key, value)
	}
	return args
}

// Helper function to create ordered properties for tests
func makeProps(pairs ...any) *api.ToolProperties {
	props := api.NewToolProperties()
	for i := 0; i < len(pairs); i += 2 {
		key := pairs[i].(string)
		value := pairs[i+1].(api.ToolProperty)
		props.Set(key, value)
	}
	return props
}

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
								Name:      "get_weather",
								Arguments: makeArgs("unit", "fahrenheit"),
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
					Parameters: api.NewToolFunctionParametersWithProps(
						"object",
						[]string{"unit"},
						makeProps(
							"unit", api.ToolProperty{Type: api.PropertyType{"string"}, Enum: []any{"celsius", "fahrenheit"}, Description: "The unit of temperature"},
						),
					),
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
					{Function: api.ToolCallFunction{Name: "double", Arguments: makeArgs("number", "1")}},
					{Function: api.ToolCallFunction{Name: "triple", Arguments: makeArgs("number", "2")}},
				}},
				{Role: "tool", Content: "{\"number\": 2}", ToolName: "double"},
				{Role: "tool", Content: "{\"number\": 6}", ToolName: "triple"},
			},
			tools: []api.Tool{
				{Function: api.ToolFunction{Name: "double", Description: "Double a number", Parameters: api.NewToolFunctionParametersWithProps("object", nil, makeProps(
					"number", api.ToolProperty{Type: api.PropertyType{"string"}, Description: "The number to double"},
				))}},
				{Function: api.ToolFunction{Name: "triple", Description: "Triple a number", Parameters: api.NewToolFunctionParametersWithProps("object", nil, makeProps(
					"number", api.ToolProperty{Type: api.PropertyType{"string"}, Description: "The number to triple"},
				))}},
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
						Name:      "echo",
						Arguments: makeArgs("payload", map[string]any{"foo": "bar"}),
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

func TestMultipleParametersNonDeterministic(t *testing.T) {
	// This test demonstrates that tools with multiple parameters are rendered
	// non-deterministically due to Go's map iteration order.
	// See https://github.com/ollama/ollama/issues/12244

	tools := []api.Tool{
		{Function: api.ToolFunction{
			Name:        "get_weather",
			Description: "Get the current weather",
			Parameters: api.NewToolFunctionParametersWithProps(
				"object",
				[]string{"location", "unit"},
				makeProps(
					"location", api.ToolProperty{Type: api.PropertyType{"string"}, Description: "The city and state"},
					"unit", api.ToolProperty{Type: api.PropertyType{"string"}, Description: "The temperature unit"},
					"format", api.ToolProperty{Type: api.PropertyType{"string"}, Description: "The output format"},
				),
			),
		}},
	}

	msgs := []api.Message{
		{Role: "user", Content: "What's the weather?"},
		{Role: "assistant", ToolCalls: []api.ToolCall{
			{Function: api.ToolCallFunction{
				Name: "get_weather",
				Arguments: makeArgs(
					"location", "San Francisco, CA",
					"unit", "fahrenheit",
					"format", "detailed",
				),
			}},
		}},
	}

	// Run the renderer multiple times and collect unique outputs
	outputs := make(map[string]bool)
	for i := 0; i < 15; i++ {
		rendered, err := Qwen3CoderRenderer(msgs, tools, nil)
		if err != nil {
			t.Fatal(err)
		}
		outputs[rendered] = true
	}

	// The renderer should be deterministic - we should only get one unique output
	if len(outputs) > 1 {
		// Show the first two different outputs for comparison
		count := 0
		for output := range outputs {
			if count < 2 {
				t.Logf("\nOutput variant %d:\n%s", count+1, output)
				count++
			}
		}
		t.Fatalf("Renderer produced %d different outputs across 15 runs (expected deterministic output)", len(outputs))
	}
}
