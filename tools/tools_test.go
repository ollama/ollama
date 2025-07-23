package tools

import (
	"testing"
	"text/template"

	"github.com/google/go-cmp/cmp"
	"github.com/ollama/ollama/api"
)

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
				Parameters: struct {
					Type       string   `json:"type"`
					Defs       any      `json:"$defs,omitempty"`
					Items      any      `json:"items,omitempty"`
					Required   []string `json:"required"`
					Properties map[string]struct {
						Type        api.PropertyType `json:"type"`
						Items       any              `json:"items,omitempty"`
						Description string           `json:"description"`
						Enum        []any            `json:"enum,omitempty"`
					} `json:"properties"`
				}{
					Type:     "object",
					Required: []string{"city"},
					Properties: map[string]struct {
						Type        api.PropertyType `json:"type"`
						Items       any              `json:"items,omitempty"`
						Description string           `json:"description"`
						Enum        []any            `json:"enum,omitempty"`
					}{
						"format": {
							Type:        api.PropertyType{"string"},
							Description: "The format to return the temperature in",
							Enum:        []any{"fahrenheit", "celsius"},
						},
						"city": {
							Type:        api.PropertyType{"string"},
							Description: "The city to get the temperature for",
						},
					},
				},
			},
		},
		{
			Type: "function",
			Function: api.ToolFunction{
				Name:        "get_conditions",
				Description: "Retrieve the current weather conditions for a given location",
				Parameters: struct {
					Type       string   `json:"type"`
					Defs       any      `json:"$defs,omitempty"`
					Items      any      `json:"items,omitempty"`
					Required   []string `json:"required"`
					Properties map[string]struct {
						Type        api.PropertyType `json:"type"`
						Items       any              `json:"items,omitempty"`
						Description string           `json:"description"`
						Enum        []any            `json:"enum,omitempty"`
					} `json:"properties"`
				}{
					Type: "object",
					Properties: map[string]struct {
						Type        api.PropertyType `json:"type"`
						Items       any              `json:"items,omitempty"`
						Description string           `json:"description"`
						Enum        []any            `json:"enum,omitempty"`
					}{
						"location": {
							Type:        api.PropertyType{"string"},
							Description: "The location to get the weather conditions for",
						},
					},
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
				Parameters: struct {
					Type       string   `json:"type"`
					Defs       any      `json:"$defs,omitempty"`
					Items      any      `json:"items,omitempty"`
					Required   []string `json:"required"`
					Properties map[string]struct {
						Type        api.PropertyType `json:"type"`
						Items       any              `json:"items,omitempty"`
						Description string           `json:"description"`
						Enum        []any            `json:"enum,omitempty"`
					} `json:"properties"`
				}{
					Type: "object",
					Properties: map[string]struct {
						Type        api.PropertyType `json:"type"`
						Items       any              `json:"items,omitempty"`
						Description string           `json:"description"`
						Enum        []any            `json:"enum,omitempty"`
					}{
						"location": {
							Type:        api.PropertyType{"string"},
							Description: "The location to get the address for",
						},
					},
				},
			},
		},
		{
			Type: "function",
			Function: api.ToolFunction{
				Name:        "add",
				Description: "Add two numbers",
				Parameters: struct {
					Type       string   `json:"type"`
					Defs       any      `json:"$defs,omitempty"`
					Items      any      `json:"items,omitempty"`
					Required   []string `json:"required"`
					Properties map[string]struct {
						Type        api.PropertyType `json:"type"`
						Items       any              `json:"items,omitempty"`
						Description string           `json:"description"`
						Enum        []any            `json:"enum,omitempty"`
					} `json:"properties"`
				}{
					Type: "object",
					Properties: map[string]struct {
						Type        api.PropertyType `json:"type"`
						Items       any              `json:"items,omitempty"`
						Description string           `json:"description"`
						Enum        []any            `json:"enum,omitempty"`
					}{
						"a": {
							Type:        api.PropertyType{"string"},
							Description: "The first number to add",
						},
						"b": {
							Type:        api.PropertyType{"string"},
							Description: "The second number to add",
						},
					},
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
						Arguments: api.ToolCallFunctionArguments{
							"location": "San Francisco",
						},
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
						Arguments: api.ToolCallFunctionArguments{},
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
						Arguments: api.ToolCallFunctionArguments{
							"city": "New York",
						},
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
						Arguments: api.ToolCallFunctionArguments{
							"city":   "London",
							"format": "fahrenheit",
						},
					},
				},
				{
					Function: api.ToolCallFunction{
						Index: 1,
						Name:  "get_conditions",
						Arguments: api.ToolCallFunctionArguments{
							"location": "Tokyo",
						},
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
						Arguments: api.ToolCallFunctionArguments{
							"city":   "London",
							"format": "fahrenheit",
						},
					},
				},
				{
					Function: api.ToolCallFunction{
						Index: 1,
						Name:  "get_conditions",
						Arguments: api.ToolCallFunctionArguments{
							"location": "Tokyo",
						},
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
						Arguments: api.ToolCallFunctionArguments{},
					},
				},
				{
					Function: api.ToolCallFunction{
						Index: 1,
						Name:  "get_temperature",
						Arguments: api.ToolCallFunctionArguments{
							"city":   "London",
							"format": "fahrenheit",
						},
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
						Arguments: api.ToolCallFunctionArguments{},
					},
				},
				{
					Function: api.ToolCallFunction{
						Index: 1,
						Name:  "get_conditions",
						Arguments: api.ToolCallFunctionArguments{
							"location": "Tokyo",
						},
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
						Arguments: api.ToolCallFunctionArguments{
							"city": "Tokyo",
						},
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
						Arguments: api.ToolCallFunctionArguments{
							"city": "Tokyo",
						},
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
						Arguments: api.ToolCallFunctionArguments{
							"city": "Tokyo",
						},
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
						Arguments: api.ToolCallFunctionArguments{
							"city": "London",
						},
					},
				},
				{
					Function: api.ToolCallFunction{
						Index: 1,
						Name:  "get_conditions",
						Arguments: api.ToolCallFunctionArguments{
							"location": "Tokyo",
						},
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
						Arguments: api.ToolCallFunctionArguments{
							"location": "Tokyo",
						},
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
						Arguments: api.ToolCallFunctionArguments{
							"location": "Tokyo",
						},
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
						Arguments: api.ToolCallFunctionArguments{},
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
						Arguments: api.ToolCallFunctionArguments{},
					},
				},
				{
					Function: api.ToolCallFunction{
						Index:     1,
						Name:      "say_hello",
						Arguments: api.ToolCallFunctionArguments{},
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
						Arguments: api.ToolCallFunctionArguments{},
					},
				},
				{
					Function: api.ToolCallFunction{
						Index:     1,
						Name:      "say_hello_world",
						Arguments: api.ToolCallFunctionArguments{},
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
						Arguments: api.ToolCallFunctionArguments{},
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
						Arguments: api.ToolCallFunctionArguments{},
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
						Arguments: api.ToolCallFunctionArguments{
							"location": "London",
						},
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
						Arguments: api.ToolCallFunctionArguments{
							"location": "London",
						},
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
						Arguments: api.ToolCallFunctionArguments{
							"a": "5",
							"b": "10",
						},
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
				if diff := cmp.Diff(calls[i], want); diff != "" {
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
			buffer: []byte(`", "arguments": {"location": "Tokyo"}}</tool_call>`),
			want: map[string]any{
				"location": "Tokyo",
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, _ := findArguments(tt.buffer)

			if diff := cmp.Diff(got, tt.want); diff != "" {
				t.Errorf("scanArguments() args mismatch (-got +want):\n%s", diff)
			}
		})
	}
}
