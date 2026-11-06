package renderers

import (
	"strings"
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
</tool_response><|im_end|>
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
</tool_response><|im_end|>
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
</tool_response><|im_end|>
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

func TestQwen3CoderRendererToolResponseNoTrailingNewline(t *testing.T) {
	msgs := []api.Message{
		{Role: "user", Content: "call tool"},
		{Role: "assistant", ToolCalls: []api.ToolCall{
			{Function: api.ToolCallFunction{
				Name:      "echo",
				Arguments: testArgs(map[string]any{"payload": "ok"}),
			}},
		}},
		{Role: "tool", Content: "{\"payload\":\"ok\"}", ToolName: "echo"},
	}

	rendered, err := (&Qwen3CoderRenderer{}).Render(msgs, nil, nil)
	if err != nil {
		t.Fatal(err)
	}

	if strings.Contains(rendered, "</tool_response>\n<|im_end|>") {
		t.Fatalf("expected no newline after </tool_response>, got:\n%s", rendered)
	}
	if !strings.Contains(rendered, "</tool_response><|im_end|>") {
		t.Fatalf("expected </tool_response> to be immediately followed by <|im_end|>, got:\n%s", rendered)
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

// toolWithExtraSchemaKeys is the shape from #18430: two or more keys the
// renderer does not write itself, both on a property (`properties` beside
// `required`) and on `parameters` (`$defs` beside `required`). Tools without
// them have at most one additional key per object and so cannot expose an
// ordering problem.
func toolWithExtraSchemaKeys() []api.Tool {
	inner := api.NewToolPropertiesMap()
	inner.Set("city", api.ToolProperty{Type: api.PropertyType{"string"}})

	props := api.NewToolPropertiesMap()
	props.Set("place", api.ToolProperty{
		Type:        api.PropertyType{"object"},
		Description: "Where.",
		Properties:  inner,
		Required:    []string{"city"},
	})
	// `items` is declared before `enum` on ToolProperty but sorts after it, so
	// this property is what separates "the order json.Marshal wrote" from "any
	// other stable order" -- sorting the keys would also be deterministic, and
	// would also be wrong.
	props.Set("tags", api.ToolProperty{
		Type:        api.PropertyType{"array"},
		Description: "Labels.",
		Items:       map[string]any{"type": "string"},
		Enum:        []any{"a", "b"},
	})

	return []api.Tool{{Function: api.ToolFunction{
		Name:        "get_weather",
		Description: "Get the forecast.",
		Parameters: api.ToolFunctionParameters{
			Type:       "object",
			Defs:       map[string]any{"unit": map[string]any{"type": "string"}},
			Required:   []string{"place"},
			Properties: props,
		},
	}}}
}

// Rendering is the prompt-cache key: an unchanged request that renders to a
// different string reprocesses everything from the first differing byte. The
// additional-key loop used to range over a map[string]any, and Go randomizes
// map iteration, so identical requests rendered in different orders.
func TestQwen3CoderRendererAdditionalKeysAreStableAcrossRenders(t *testing.T) {
	renderer := &Qwen3CoderRenderer{}
	msgs := []api.Message{{Role: "user", Content: "What is the weather in Lisbon?"}}

	first, err := renderer.Render(msgs, toolWithExtraSchemaKeys(), nil)
	if err != nil {
		t.Fatalf("render: %v", err)
	}

	// One render has two independent orderings to get wrong (the property's
	// and `parameters`'), so a single repeat already fails the old code half
	// the time; 100 makes it a certainty rather than a coin flip.
	for i := 1; i < 100; i++ {
		got, err := renderer.Render(msgs, toolWithExtraSchemaKeys(), nil)
		if err != nil {
			t.Fatalf("render %d: %v", i, err)
		}
		if diff := cmp.Diff(first, got); diff != "" {
			t.Fatalf("render %d differs from render 0 (-first +got):\n%s", i, diff)
		}
	}
}

// The order itself is part of the contract, not just its stability: it is the
// order `encoding/json` writes the struct's fields in, which is what the
// reference implementation's ordering comment on renderAdditionalKeys means.
func TestQwen3CoderRendererAdditionalKeysFollowFieldOrder(t *testing.T) {
	renderer := &Qwen3CoderRenderer{}
	got, err := renderer.Render(
		[]api.Message{{Role: "user", Content: "What is the weather in Lisbon?"}},
		toolWithExtraSchemaKeys(),
		nil,
	)
	if err != nil {
		t.Fatalf("render: %v", err)
	}

	// Declaration order: ToolProperty is anyOf, type, items, description,
	// enum, properties, required; ToolFunctionParameters is type, $defs,
	// items, required, properties. `items` before `enum` on `tags` is the
	// pair that sorting the keys would swap.
	want := `<parameters>
<parameter>
<name>place</name>
<type>object</type>
<description>Where.</description>
<properties>{"city":{"type":"string"}}</properties>
<required>["city"]</required>
</parameter>
<parameter>
<name>tags</name>
<type>array</type>
<description>Labels.</description>
<items>{"type":"string"}</items>
<enum>["a","b"]</enum>
</parameter>
<$defs>{"unit":{"type":"string"}}</$defs>
<required>["place"]</required>
</parameters>`

	if !strings.Contains(got, want) {
		t.Errorf("rendered tool block does not match\nwant substring:\n%s\n\ngot:\n%s", want, got)
	}
}
