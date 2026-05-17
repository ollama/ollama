package tools

import (
	"strings"
	"testing"
	"text/template"

	"github.com/google/go-cmp/cmp"
	"github.com/ollama/ollama/api"
)

// argsComparer provides cmp options for comparing ToolCallFunctionArguments by value (order-insensitive)
var argsComparer = cmp.Comparer(func(a, b api.ToolCallFunctionArguments) bool {
	return cmp.Equal(a.ToMap(), b.ToMap())
})

// testPropsMap creates a ToolPropertiesMap from a map (convenience function for tests, order not preserved)
func testPropsMap(m map[string]api.ToolProperty) *api.ToolPropertiesMap {
	props := api.NewToolPropertiesMap()
	for k, v := range m {
		props.Set(k, v)
	}
	return props
}

// testArgs creates ToolCallFunctionArguments from a map (convenience function for tests, order not preserved)
func testArgs(m map[string]any) api.ToolCallFunctionArguments {
	args := api.NewToolCallFunctionArguments()
	for k, v := range m {
		args.Set(k, v)
	}
	return args
}

func TestParser(t *testing.T) {
	qwen, err := template.New("qwen").Parse(`{{if .ToolCalls}}<tool_call>{{range .ToolCalls}}{"name": "{{.Function.Name}}", "arguments": {{.Function.Arguments}}}{{end}}</tool_call>{{end}}`)
	if err != nil {
		t.Fatalf("Failed to parse template: %v", err)
	}

	deepseek, err := template.New("deepseek").Parse("{{if .ToolCalls}}<|tool▁calls▁begin|>{{range .ToolCalls}}<|tool▁call▁begin|>function<|tool▁sep|>get_current_weather\n```json\n{\"location\": \"Tokyo\"}\n```<|tool▁call▁end|>{{end}}<|tool▁calls▁end|><|end▁of▁sentence|>{{end}}")
	if err != nil {
		t.Fatalf("Failed to parse template: %v", err)
	}

	json, err := template.New("json").Parse(`{{if .ToolCalls}}{{range .ToolCalls}}{"name": "{{.Function.Name}}", "arguments": {{.Function.Arguments}}}{{end}}{{end}}`)
	if err != nil {
		t.Fatalf("Failed to parse template: %v", err)
	}

	mistral, err := template.New("mistral").Parse(`{{if .ToolCalls}}[TOOL_CALLS] [{{range .ToolCalls}}{"name": "{{.Function.Name}}", "arguments": {{.Function.Arguments}}}{{end}}][/TOOL_CALLS]{{end}}`)
	if err != nil {
		t.Fatalf("Failed to parse template: %v", err)
	}

	list, err := template.New("list").Parse(`{{if .ToolCalls}}[{{range .ToolCalls}}{"name": "{{.Function.Name}}", "arguments": {{.Function.Arguments}}}{{end}}]{{end}}`)
	if err != nil {
		t.Fatalf("Failed to parse template: %v", err)
	}

	tools := []api.Tool{
		{
			Type: "function",
			Function: api.ToolFunction{
				Name:        "get_temperature",
				Description: "Retrieve the temperature for a given location",
				Parameters: api.ToolFunctionParameters{
					Type:     "object",
					Required: []string{"city"},
					Properties: testPropsMap(map[string]api.ToolProperty{
						"format": {
							Type:        api.PropertyType{"string"},
							Description: "The format to return the temperature in",
							Enum:        []any{"fahrenheit", "celsius"},
						},
						"city": {
							Type:        api.PropertyType{"string"},
							Description: "The city to get the temperature for",
						},
					}),
				},
			},
		},
		{
			Type: "function",
			Function: api.ToolFunction{
				Name:        "get_conditions",
				Description: "Retrieve the current weather conditions for a given location",
				Parameters: api.ToolFunctionParameters{
					Type: "object",
					Properties: testPropsMap(map[string]api.ToolProperty{
						"location": {
							Type:        api.PropertyType{"string"},
							Description: "The location to get the weather conditions for",
						},
					}),
				},
			},
		},
		{
			Type: "function",
			Function: api.ToolFunction{
				Name:        "say_hello",
				Description: "Say hello",
			},
		},
		{
			Type: "function",
			Function: api.ToolFunction{
				Name:        "say_hello_world",
				Description: "Say hello world",
			},
		},
		{
			Type: "function",
			Function: api.ToolFunction{
				Name:        "get_address",
				Description: "Get the address of a given location",
				Parameters: api.ToolFunctionParameters{
					Type: "object",
					Properties: testPropsMap(map[string]api.ToolProperty{
						"location": {
							Type:        api.PropertyType{"string"},
							Description: "The location to get the address for",
						},
					}),
				},
			},
		},
		{
			Type: "function",
			Function: api.ToolFunction{
				Name:        "add",
				Description: "Add two numbers",
				Parameters: api.ToolFunctionParameters{
					Type: "object",
					Properties: testPropsMap(map[string]api.ToolProperty{
						"a": {
							Type:        api.PropertyType{"string"},
							Description: "The first number to add",
						},
						"b": {
							Type:        api.PropertyType{"string"},
							Description: "The second number to add",
						},
					}),
				},
			},
		},
	}

	tests := []struct {
		name    string
		inputs  []string
		tmpl    *template.Template
		content string
		calls   []api.ToolCall
	}{
		{
			name:    "no tool calls - just text",
			inputs:  []string{"Hello, how can I help you today?"},
			content: "Hello, how can I help you today?",
			tmpl:    qwen,
			calls:   nil,
		},
		{
			name:    "empty input",
			inputs:  []string{""},
			content: "",
			tmpl:    qwen,
			calls:   nil,
		},
		{
			name:    "tool call",
			inputs:  []string{`<tool_call>{"name": "get_conditions", "arguments": {"location": "San Francisco"}}</tool_call>`},
			content: "",
			tmpl:    qwen,
			calls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Index: 0,
						Name:  "get_conditions",
						Arguments: testArgs(map[string]any{
							"location": "San Francisco",
						}),
					},
				},
			},
		},
		{
			name:    "empty args",
			inputs:  []string{`<tool_call>{"name": "get_conditions", "arguments": {}}</tool_call>`},
			content: "",
			tmpl:    qwen,
			calls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Index:     0,
						Name:      "get_conditions",
						Arguments: api.NewToolCallFunctionArguments(),
					},
				},
			},
		},
		{
			name:    "text before tool call",
			inputs:  []string{`Let me check the weather. <tool_call>{"name": "get_temperature", "arguments": {"city": "New York"}}</tool_call>`},
			content: "Let me check the weather. ",
			tmpl:    qwen,
			calls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Index: 0,
						Name:  "get_temperature",
						Arguments: testArgs(map[string]any{
							"city": "New York",
						}),
					},
				},
			},
		},
		{
			name:    "qwen no args with text",
			inputs:  []string{"Let me say hello to the user. I'll use the say_hello tool. "},
			content: "Let me say hello to the user. I'll use the say_hello tool. ",
			tmpl:    qwen,
			calls:   nil,
		},
		{
			name:    "two tool calls in a list",
			inputs:  []string{`[TOOL_CALLS] [{"name": "get_temperature", "arguments": {"city": "London", "format": "fahrenheit"}}, {"name": "get_conditions", "arguments": {"location": "Tokyo"}}][/TOOL_CALLS]`},
			content: "",
			tmpl:    mistral,
			calls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Index: 0,
						Name:  "get_temperature",
						Arguments: testArgs(map[string]any{
							"city":   "London",
							"format": "fahrenheit",
						}),
					},
				},
				{
					Function: api.ToolCallFunction{
						Index: 1,
						Name:  "get_conditions",
						Arguments: testArgs(map[string]any{
							"location": "Tokyo",
						}),
					},
				},
			},
		},
		{
			name:    "qwen two tool calls",
			inputs:  []string{`Okay, let's call both tools! <tool_call>{"name": "get_temperature", "arguments": {"city": "London", "format": "fahrenheit"}}</tool_call><tool_call>{"name": "get_conditions", "arguments": {"location": "Tokyo"}}</tool_call>`},
			content: "Okay, let's call both tools! ",
			tmpl:    qwen,
			calls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Index: 0,
						Name:  "get_temperature",
						Arguments: testArgs(map[string]any{
							"city":   "London",
							"format": "fahrenheit",
						}),
					},
				},
				{
					Function: api.ToolCallFunction{
						Index: 1,
						Name:  "get_conditions",
						Arguments: testArgs(map[string]any{
							"location": "Tokyo",
						}),
					},
				},
			},
		},
		{
			name:    "empty args followed by args",
			inputs:  []string{`Let me say hello and check the weather. <tool_call>{"name": "say_hello", "arguments": {}}</tool_call><tool_call>{"name": "get_temperature", "arguments": {"city": "London", "format": "fahrenheit"}}</tool_call>`},
			content: "Let me say hello and check the weather. ",
			tmpl:    qwen,
			calls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Index:     0,
						Name:      "say_hello",
						Arguments: api.NewToolCallFunctionArguments(),
					},
				},
				{
					Function: api.ToolCallFunction{
						Index: 1,
						Name:  "get_temperature",
						Arguments: testArgs(map[string]any{
							"city":   "London",
							"format": "fahrenheit",
						}),
					},
				},
			},
		},
		{
			name:    "qwen empty followed by args",
			inputs:  []string{`Let me check the weather. <tool_call>{"name": "get_conditions", "arguments": {}}</tool_call><tool_call>{"name": "get_conditions", "arguments": {"location": "Tokyo"}}`},
			content: "Let me check the weather. ",
			tmpl:    qwen,
			calls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Index:     0,
						Name:      "get_conditions",
						Arguments: api.NewToolCallFunctionArguments(),
					},
				},
				{
					Function: api.ToolCallFunction{
						Index: 1,
						Name:  "get_conditions",
						Arguments: testArgs(map[string]any{
							"location": "Tokyo",
						}),
					},
				},
			},
		},
		{
			name:    "deepseek",
			inputs:  []string{"<think>Wait, I need to call a tool</think><|tool▁calls▁begin|><|tool▁call▁begin|>function<|tool▁sep|>get_temperature\n```json\n{\"city\": \"Tokyo\"}\n```<|tool▁call▁end|><|tool▁calls▁end|><|end▁of▁sentence|>"},
			content: "<think>Wait, I need to call a tool</think>",
			tmpl:    deepseek,
			calls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Index: 0,
						Name:  "get_temperature",
						Arguments: testArgs(map[string]any{
							"city": "Tokyo",
						}),
					},
				},
			},
		},
		{
			name: "deepseek incremental",
			inputs: []string{
				"<think>Wait",
				", I need",
				" to call",
				" a tool</think><|too",
				"l▁calls▁begin",
				"|>",
				"<|tool▁call▁begin|>function<|tool▁sep|>get_temperature\n",
				"```json\n",
				"{\"city\": \"Tokyo\"}\n",
				"```",
				"<|tool▁c", "all▁end|>",
				"<|tool▁calls▁end|>",
				"<|end▁of▁sentence|>",
			},
			content: "<think>Wait, I need to call a tool</think>",
			tmpl:    deepseek,
			calls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Index: 0,
						Name:  "get_temperature",
						Arguments: testArgs(map[string]any{
							"city": "Tokyo",
						}),
					},
				},
			},
		},
		{
			name: "json",
			inputs: []string{
				"{",
				"\"name\": \"get_temperature\",",
				"\"arguments\": {",
				"\"city\": \"Tokyo\"",
				"}",
				"}",
			},
			content: "",
			tmpl:    json,
			calls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Index: 0,
						Name:  "get_temperature",
						Arguments: testArgs(map[string]any{
							"city": "Tokyo",
						}),
					},
				},
			},
		},
		{
			name: "json maybe a tool call",
			inputs: []string{
				"{",
				"\"name\": \"get_temperature\",",
				"\"arguments\": {",
			},
			content: "",
			tmpl:    json,
			calls:   nil,
		},
		{
			name: "json not a tool call",
			inputs: []string{
				"{",
				"\"name\": \"search\", ",
				"\"arguments\": {",
				"\"query\": \"What is the capital of Canada?\"",
				"}",
				"}",
			},
			content: "{\"name\": \"search\", \"arguments\": {\"query\": \"What is the capital of Canada?\"}}",
			tmpl:    json,
			calls:   nil,
		},
		{
			name: "json object followed by tool call",
			inputs: []string{
				"{\"name\": \"jeff\"}",
				"{\"name\": \"get_conditions\", \"arguments\": {\"location\": \"San Francisco\"}}",
			},
			content: "{\"name\": \"jeff\"}{\"name\": \"get_conditions\", \"arguments\": {\"location\": \"San Francisco\"}}",
			tmpl:    json,
		},
		{
			name: "json object followed by tool call split",
			inputs: []string{
				"{\"name\": \"jeff\"} {",
				"\"name\": \"get_conditions\", \"arguments\": {\"location\": \"San Francisco\"}}",
			},
			content: "{\"name\": \"jeff\"} {\"name\": \"get_conditions\", \"arguments\": {\"location\": \"San Francisco\"}}",
			tmpl:    json,
		},
		{
			name: "json code",
			inputs: []string{
				"for { fmt.Println(\"hello\") }",
			},
			content: "for { fmt.Println(\"hello\") }",
			tmpl:    json,
		},
		{
			name: "list multiple",
			inputs: []string{
				"[",
				"{",
				"\"name\": \"get_temperature\", ",
				"\"arguments\": {",
				"\"city\": \"London\"",
				"}",
				"},",
				"{",
				"\"name\": \"get_conditions\", ",
				"\"arguments\": {",
				"\"location\": \"Tokyo\"",
				"}",
				"}]",
			},
			content: "",
			tmpl:    list,
			calls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Index: 0,
						Name:  "get_temperature",
						Arguments: testArgs(map[string]any{
							"city": "London",
						}),
					},
				},
				{
					Function: api.ToolCallFunction{
						Index: 1,
						Name:  "get_conditions",
						Arguments: testArgs(map[string]any{
							"location": "Tokyo",
						}),
					},
				},
			},
		},
		{
			name: "list partial",
			inputs: []string{
				"[{",
				"\"name\": \"get_conditions\", ",
				"\"arguments\": {",
				"\"location\": \"Tokyo\"",
				"}",
				"}",
			},
			content: "",
			tmpl:    list,
			calls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Index: 0,
						Name:  "get_conditions",
						Arguments: testArgs(map[string]any{
							"location": "Tokyo",
						}),
					},
				},
			},
		},
		{
			name: "list invalid",
			inputs: []string{
				"[",
				"{",
				"\"name\": \"search\", ",
				"\"arguments\": {",
				"\"query\": \"What is the capital of Canada?\"",
				"}",
				"}",
			},
			content: "",
			tmpl:    list,
			calls:   nil,
		},
		{
			name: "list trailing ]",
			inputs: []string{
				"[",
				"{",
				"\"name\": \"get_conditions\", ",
				"\"arguments\": {",
				"\"location\": \"Tokyo\"",
				"}",
				"}",
				"]",
				"]",
			},
			content: "",
			tmpl:    list,
			calls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Index: 0,
						Name:  "get_conditions",
						Arguments: testArgs(map[string]any{
							"location": "Tokyo",
						}),
					},
				},
			},
		},
		{
			name: "list not a tool call",
			inputs: []string{
				"[special",
				" del",
				"ivery]",
			},
			content: "[special delivery]",
			tmpl:    list,
			calls:   nil,
		},
		{
			name: "tool name with collision",
			inputs: []string{
				"<tool_call>",
				"{",
				"\"name\": \"say_hello",
				"_world\",",
				"\"arguments\": {}}",
				"}",
			},
			content: "",
			tmpl:    qwen,
			calls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Index:     0,
						Name:      "say_hello_world",
						Arguments: api.NewToolCallFunctionArguments(),
					},
				},
			},
		},
		{
			name: "tool name with collision multiple",
			inputs: []string{
				"<tool_call>",
				"{",
				"\"name\": \"say_hello",
				"_world\",",
				"\"arguments\": {}}",
				"</tool_call>",
				"<tool_call>",
				"{",
				"\"name\": \"say_hello",
				"\",",
				"\"arguments\": {}}",
				"</tool_call>",
			},
			content: "",
			tmpl:    qwen,
			calls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Index:     0,
						Name:      "say_hello_world",
						Arguments: api.NewToolCallFunctionArguments(),
					},
				},
				{
					Function: api.ToolCallFunction{
						Index:     1,
						Name:      "say_hello",
						Arguments: api.NewToolCallFunctionArguments(),
					},
				},
			},
		},
		{
			name: "tool name with collision non streaming",
			inputs: []string{
				`<tool_call>{"name": "say_hello`,
			},
			content: "",
			tmpl:    qwen,
			calls:   nil,
		},
		{
			name: "tool name with collision non streaming multiple",
			inputs: []string{
				`<tool_call>{"name": "say_hello", "arguments": {}}</tool_call><tool_call>{"name": "say_hello_world", "arguments": {}}`,
			},
			content: "",
			tmpl:    qwen,
			calls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Index:     0,
						Name:      "say_hello",
						Arguments: api.NewToolCallFunctionArguments(),
					},
				},
				{
					Function: api.ToolCallFunction{
						Index:     1,
						Name:      "say_hello_world",
						Arguments: api.NewToolCallFunctionArguments(),
					},
				},
			},
		},
		{
			name: "tool name with collision non streaming shorter",
			inputs: []string{
				`<tool_call>{"name": "say_hello", "arguments": {}}</tool_call>`,
			},
			content: "",
			tmpl:    qwen,
			calls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Index:     0,
						Name:      "say_hello",
						Arguments: api.NewToolCallFunctionArguments(),
					},
				},
			},
		},
		{
			name: "tool name with collision non streaming longer",
			inputs: []string{
				`<tool_call>{"name": "say_hello_world", "arguments": {}}</tool_call>`,
			},
			content: "",
			tmpl:    qwen,
			calls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Index:     0,
						Name:      "say_hello_world",
						Arguments: api.NewToolCallFunctionArguments(),
					},
				},
			},
		},
		{
			name: "tool name with substring of another",
			inputs: []string{
				"{",
				"\"name\": \"get_address\",",
				"\"arguments\": {",
				"\"location\": \"London\"",
				"}",
				"}",
			},
			content: "",
			tmpl:    json,
			calls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Index: 0,
						Name:  "get_address",
						Arguments: testArgs(map[string]any{
							"location": "London",
						}),
					},
				},
			},
		},
		{
			name: "tool name with substring of another",
			inputs: []string{
				`<tool_call>{"name": "get_address", "arguments": {"location": "London"}}</tool_call>`,
			},
			content: "",
			tmpl:    qwen,
			calls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Index: 0,
						Name:  "get_address",
						Arguments: testArgs(map[string]any{
							"location": "London",
						}),
					},
				},
			},
		},
		{
			name: "args before name",
			inputs: []string{
				`<tool_call>{"arguments": {"a": "5", "b": "10"}, "name": "add"}</tool_call>`,
			},
			content: "",
			tmpl:    qwen,
			calls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Index: 0,
						Name:  "add",
						Arguments: testArgs(map[string]any{
							"a": "5",
							"b": "10",
						}),
					},
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			parser := NewParser(tt.tmpl, tools)

			var calls []api.ToolCall
			var content string
			for _, input := range tt.inputs {
				tcs, c := parser.Add(input)
				calls = append(calls, tcs...)
				content += c
			}

			if content != tt.content {
				t.Errorf("Expected content %q, got %q", tt.content, content)
			}

			if len(calls) != len(tt.calls) {
				t.Fatalf("Expected %d tool calls, got %d", len(tt.calls), len(calls))
			}

			for i, want := range tt.calls {
				if diff := cmp.Diff(calls[i], want, argsComparer); diff != "" {
					t.Errorf("Tool call %d mismatch (-got +want):\n%s", i, diff)
				}
			}
		})
	}
}

func TestDone(t *testing.T) {
	tests := []struct {
		name   string
		tag    string
		buffer []byte
		want   bool
	}{
		{
			name:   "empty",
			tag:    "<tool_call>",
			buffer: []byte{},
			want:   false,
		},
		{
			name:   "empty",
			tag:    "<tool_call>",
			buffer: []byte{},
			want:   false,
		},
		{
			name:   "json open",
			tag:    "{",
			buffer: []byte("{\"name\": \"get_weather\""),
			want:   false,
		},
		{
			name:   "json closed",
			tag:    "{",
			buffer: []byte("{\"name\": \"get_weather\"}"),
			want:   true,
		},
		{
			name:   "json empty",
			tag:    "{",
			buffer: []byte("{}"),
			want:   true,
		},
		{
			name:   "list open",
			tag:    "[",
			buffer: []byte("[{\"name\": \"get_weather\""),
			want:   false,
		},
		{
			name:   "list closed",
			tag:    "[",
			buffer: []byte("[{\"name\": \"get_weather\"}]"),
			want:   true,
		},
		{
			name:   "list empty",
			tag:    "[",
			buffer: []byte("[]"),
			want:   true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			parser := &Parser{
				tag:    tt.tag,
				buffer: tt.buffer,
			}
			got := parser.done()
			if got != tt.want {
				t.Errorf("done() = %t, want %t", got, tt.want)
			}
		})
	}
}

func TestContent(t *testing.T) {
	tests := []struct {
		name    string
		tag     string
		content []byte
		want    string
		n       int
	}{
		{
			name:    "empty",
			content: []byte{},
			tag:     "{",
			want:    "",
			n:       0,
		},
		{
			name:    "tag",
			tag:     "<tool_call>",
			content: []byte("<tool_call>{\"name\": \"get_temperature\""),
			want:    "",
			n:       0,
		},
		{
			name:    "json object",
			tag:     "{",
			content: []byte("{\"name\": \"get_temperature\"}"),
			want:    "{\"name\": \"get_temperature\"}",
			n:       0,
		},
		{
			name:    "json object after called",
			tag:     "{",
			content: []byte("{\"hello\": \"world\"}"),
			want:    "{\"hello\": \"world\"}",
			n:       0,
		},
		{
			name:    "json object after called",
			tag:     "{",
			content: []byte("{\"hello\": \"world\"}"),
			want:    "",
			n:       1,
		},
		{
			name:    "list",
			tag:     "[",
			content: []byte("[{\"name\": \"get_temperature\"}]"),
			want:    "[{\"name\": \"get_temperature\"}]",
			n:       0,
		},
		{
			name:    "code",
			tag:     "{",
			content: []byte("{ fmt.Println(\"hello\")"),
			want:    "{ fmt.Println(\"hello\")",
			n:       0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			parser := &Parser{
				tag:    tt.tag,
				buffer: tt.content,
				n:      tt.n,
			}
			got := parser.Content()
			if got != tt.want {
				t.Errorf("Content() = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestFindTag(t *testing.T) {
	cases := []struct {
		name   string
		buffer []byte
		tag    string
		i      int
		found  bool
	}{
		{
			name:   "no overlap",
			buffer: []byte("hello world"),
			tag:    "<tool_call>",
			i:      -1,
			found:  false,
		},
		{
			name:   "full overlap",
			buffer: []byte("<tool_call>"),
			tag:    "<tool_call>",
			i:      0,
			found:  true,
		},
		{
			name:   "whitespace",
			buffer: []byte("    <tool_call>\n {\"name\": \"bob\"}"),
			tag:    "<tool_call>",
			i:      4,
			found:  true,
		},
		{
			name:   "over",
			buffer: []byte("<tool_call>{\"name\""),
			tag:    "<tool_call>",
			i:      0,
			found:  true,
		},
		{
			name:   "partial overlap",
			buffer: []byte("text <tool_call>"),
			tag:    "<tool_call>",
			i:      5,
			found:  true,
		},
		{
			name:   "overlap with extra",
			buffer: []byte("<tool_calls><tool_call>"),
			tag:    "<tool_calls>",
			i:      0,
			found:  true,
		},
		{
			name:   "delimiter longer than string",
			buffer: []byte("<tool>"),
			tag:    "<tool_call>",
			i:      -1,
			found:  false,
		},
		{
			name:   "empty string",
			buffer: []byte{},
			tag:    "<tool_call>",
			i:      -1,
			found:  false,
		},
		{
			name:   "single char overlap",
			buffer: []byte("test<"),
			tag:    "<tool_call>",
			i:      4,
			found:  false,
		},
		{
			name:   "partial tool call",
			buffer: []byte("hello <tool_"),
			tag:    "<tool_call>",
			i:      6,
			found:  false,
		},
		{
			name:   "square bracket",
			buffer: []byte("calling tools: ["),
			tag:    "[",
			i:      15,
			found:  true,
		},
		{
			name:   "bracket",
			buffer: []byte("{\"name\": \"bob\""),
			tag:    "{",
			i:      0,
			found:  true,
		},
		{
			name:   "bracket with whitespace",
			buffer: []byte("\n\n{\n\"name\": \"bob\""),
			tag:    "{",
			i:      2,
			found:  true,
		},
	}

	for _, tt := range cases {
		t.Run(tt.name, func(t *testing.T) {
			parser := &Parser{
				tag:    tt.tag,
				buffer: tt.buffer,
				n:      0,
			}
			i, found := parser.findTag()
			if i != tt.i {
				t.Errorf("findTag(%q, %q) = %d; want %d", tt.buffer, tt.tag, i, tt.i)
			}
			if found != tt.found {
				t.Errorf("findTag(%q, %q) = %t; want %t", tt.buffer, tt.tag, found, tt.found)
			}
		})
	}
}

func TestFindArguments(t *testing.T) {
	tests := []struct {
		name   string
		buffer []byte
		want   map[string]any
		tool   string
	}{
		{
			name:   "empty string",
			buffer: []byte{},
			want:   nil,
		},
		{
			name:   "whitespace only",
			buffer: []byte("   \n\t  "),
			want:   nil,
		},
		{
			name:   "unbalanced braces - missing closing",
			buffer: []byte(`{"format": "fahrenheit", "location": "San Francisco"`),
			want:   nil,
		},
		{
			name:   "unbalanced braces - extra closing",
			buffer: []byte(`{"format": "fahrenheit"}}`),
			want: map[string]any{
				"format": "fahrenheit",
			},
		},
		{
			name:   "invalid JSON",
			buffer: []byte(`{format: fahrenheit, location: "San Francisco"}`),
			want:   nil,
		},
		{
			name:   "valid json",
			buffer: []byte(`{"name": "get_temperature", "arguments": {"format": "fahrenheit", "location": "San Francisco, CA"}}`),
			want: map[string]any{
				"format":   "fahrenheit",
				"location": "San Francisco, CA",
			},
		},
		{
			name:   "valid arguments with special tokens",
			buffer: []byte(`[tool]get_temperature[args]{"format": "fahrenheit", "location": "San Francisco, CA"}[end]`),
			want: map[string]any{
				"format":   "fahrenheit",
				"location": "San Francisco, CA",
			},
		},
		{
			name:   "valid arguments in array",
			buffer: []byte(`[{"name": "get_temperature", "arguments": {"format": "fahrenheit", "location": "San Francisco, CA"}}`),
			want: map[string]any{
				"format":   "fahrenheit",
				"location": "San Francisco, CA",
			},
		},
		{
			name:   "nested deep",
			buffer: []byte(`{"function": {"name": "get_temperature", "arguments": {"format": "fahrenheit", "location": "San Francisco, CA"}}}`),
			want: map[string]any{
				"format":   "fahrenheit",
				"location": "San Francisco, CA",
			},
		},
		{
			name:   "one arg",
			buffer: []byte(`get_temperature({"location": "San Francisco, CA"})`),
			want: map[string]any{
				"location": "San Francisco, CA",
			},
		},
		{
			name:   "two args",
			buffer: []byte(`[{"name": "get_temperature", "arguments": {"location": "San Francisco, CA", "format": "fahrenheit"}}, {"name": "get_weather", "arguments": {"location": "San Francisco, CA", "format": "fahrenheit"}}]`),
			want: map[string]any{
				"location": "San Francisco, CA",
				"format":   "fahrenheit",
			},
		},
		{
			name:   "deepseek",
			buffer: []byte("<|tool▁calls▁begin|><|tool▁call▁begin|>function<|tool▁sep|>get_temperature\n```json\n{\"location\": \"Tokyo\"}\n```<|tool▁call▁end|><|tool▁calls▁end|><|end▁of▁sentence|>"),
			want: map[string]any{
				"location": "Tokyo",
			},
		},
		{
			name:   "deepseek",
			buffer: []byte(`"arguments": {"location": "Tokyo"}}</tool_call>`),
			want: map[string]any{
				"location": "Tokyo",
			},
		},
		{
			name:   "string with braces",
			buffer: []byte(`{"name": "process_code", "arguments": {"code": "if (x > 0) { return true; }"}}`),
			want: map[string]any{
				"code": "if (x > 0) { return true; }",
			},
		},
		{
			name:   "string with nested json",
			buffer: []byte(`{"name": "send_data", "arguments": {"payload": "{\"nested\": {\"key\": \"value\"}}"}}`),
			want: map[string]any{
				"payload": `{"nested": {"key": "value"}}`,
			},
		},
		{
			name:   "string with escaped quotes and braces",
			buffer: []byte(`{"name": "analyze", "arguments": {"text": "The JSON is: {\"key\": \"val{ue}\"}"}}`),
			want: map[string]any{
				"text": `The JSON is: {"key": "val{ue}"}`,
			},
		},
		{
			name:   "multiple objects with string containing braces",
			buffer: []byte(`{"name": "test", "arguments": {"query": "find } in text"}} {"name": "other"}`),
			want: map[string]any{
				"query": "find } in text",
			},
		},
		{
			name:   "unmatched closing brace in string",
			buffer: []byte(`{"name": "search", "arguments": {"pattern": "regex: }"}}`),
			want: map[string]any{
				"pattern": "regex: }",
			},
		},
		{
			name:   "complex nested with mixed braces",
			buffer: []byte(`{"name": "analyze", "arguments": {"data": "{\"items\": [{\"value\": \"}\"}, {\"code\": \"if (x) { return y; }\"}]}"}}`),
			want: map[string]any{
				"data": `{"items": [{"value": "}"}, {"code": "if (x) { return y; }"}]}`,
			},
		},
		{
			name:   "string with newline and braces",
			buffer: []byte(`{"name": "format", "arguments": {"template": "{\n  \"key\": \"value\"\n}"}}`),
			want: map[string]any{
				"template": "{\n  \"key\": \"value\"\n}",
			},
		},
		{
			name:   "string with unicode escape",
			buffer: []byte(`{"name": "test", "arguments": {"text": "Unicode: \u007B and \u007D"}}`),
			want: map[string]any{
				"text": "Unicode: { and }",
			},
		},
		{
			name:   "array arguments",
			buffer: []byte(`{"name": "batch", "arguments": ["item1", "item2", "{\"nested\": true}"]}`),
			want:   nil, // This should return nil because arguments is not a map
		},
		{
			name:   "escaped backslash before quote",
			buffer: []byte(`{"name": "path", "arguments": {"dir": "C:\\Program Files\\{App}\\"}}`),
			want: map[string]any{
				"dir": `C:\Program Files\{App}\`,
			},
		},
		{
			name:   "single quotes not treated as string delimiters",
			buffer: []byte(`{"name": "query", "arguments": {"sql": "SELECT * FROM users WHERE name = '{admin}'"}}`),
			want: map[string]any{
				"sql": "SELECT * FROM users WHERE name = '{admin}'",
			},
		},
		{
			name:   "incomplete json at buffer end",
			buffer: []byte(`{"name": "test", "arguments": {"data": "some {"`),
			want:   nil,
		},
		{
			name:   "multiple escaped quotes",
			buffer: []byte(`{"name": "echo", "arguments": {"msg": "He said \"Hello {World}\" loudly"}}`),
			want: map[string]any{
				"msg": `He said "Hello {World}" loudly`,
			},
		},
		{
			name:   "json with comments style string",
			buffer: []byte(`{"name": "code", "arguments": {"snippet": "// This is a comment with { and }"}}`),
			want: map[string]any{
				"snippet": "// This is a comment with { and }",
			},
		},
		{
			name:   "consecutive escaped backslashes",
			buffer: []byte(`{"name": "test", "arguments": {"path": "C:\\\\{folder}\\\\"}}`),
			want: map[string]any{
				"path": `C:\\{folder}\\`,
			},
		},
		{
			name:   "empty string with braces after",
			buffer: []byte(`{"name": "test", "arguments": {"a": "", "b": "{value}"}}`),
			want: map[string]any{
				"a": "",
				"b": "{value}",
			},
		},
		{
			name:   "unicode in key names",
			buffer: []byte(`{"name": "test", "arguments": {"key{": "value", "key}": "value2"}}`),
			want: map[string]any{
				"key{": "value",
				"key}": "value2",
			},
		},
		{
			name:   "very long string with braces",
			buffer: []byte(`{"name": "test", "arguments": {"data": "` + strings.Repeat("a{b}c", 100) + `"}}`),
			want: map[string]any{
				"data": strings.Repeat("a{b}c", 100),
			},
		},
		{
			name:   "tab characters and braces",
			buffer: []byte(`{"name": "test", "arguments": {"code": "\tif (true) {\n\t\treturn;\n\t}"}}`),
			want: map[string]any{
				"code": "\tif (true) {\n\t\treturn;\n\t}",
			},
		},
		{
			name:   "null byte in string",
			buffer: []byte(`{"name": "test", "arguments": {"data": "before\u0000{after}"}}`),
			want: map[string]any{
				"data": "before\x00{after}",
			},
		},
		{
			name:   "escaped quote at end of string",
			buffer: []byte(`{"name": "test", "arguments": {"data": "text with quote at end\\\""}}`),
			want: map[string]any{
				"data": `text with quote at end\"`,
			},
		},
		{
			name:   "mixed array and object in arguments",
			buffer: []byte(`{"name": "test", "arguments": {"items": ["{", "}", {"key": "value"}]}}`),
			want: map[string]any{
				"items": []any{"{", "}", map[string]any{"key": "value"}},
			},
		},
		{
			name:   "stringified arguments",
			buffer: []byte(`{"name": "get_temperature", "arguments": "{\"format\": \"fahrenheit\", \"location\": \"San Francisco, CA\"}"}`),
			want: map[string]any{
				"format":   "fahrenheit",
				"location": "San Francisco, CA",
			},
		},
		{
			name:   "stringified parameters",
			buffer: []byte(`{"name": "get_temperature", "parameters": "{\"format\": \"fahrenheit\", \"location\": \"San Francisco, CA\"}"}`),
			want: map[string]any{
				"format":   "fahrenheit",
				"location": "San Francisco, CA",
			},
		},
		{
			name:   "simple tool call",
			tool:   "get_temperature",
			buffer: []byte(`{"get_temperature": {"format": "fahrenheit", "location": "San Francisco, CA"}}`),
			want: map[string]any{
				"format":   "fahrenheit",
				"location": "San Francisco, CA",
			},
		},
		{
			name:   "stringified simple tool call",
			tool:   "get_temperature",
			buffer: []byte(`{"get_temperature": "{\"format\": \"fahrenheit\", \"location\": \"San Francisco, CA\"}"}`),
			want: map[string]any{
				"format":   "fahrenheit",
				"location": "San Francisco, CA",
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, _ := findArguments(&api.Tool{Function: api.ToolFunction{Name: tt.tool}}, tt.buffer)

			if diff := cmp.Diff(got, tt.want); diff != "" {
				t.Errorf("findArguments() args mismatch (-got +want):\n%s", diff)
			}
		})
	}
}
