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
					Type: "object",
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
			name:    "two tool calls",
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

func TestContent(t *testing.T) {
	tests := []struct {
		name   string
		parser *Parser
		want   string
	}{
		{
			name: "empty",
			parser: &Parser{
				tag:    "<tool_call>",
				buffer: "",
				n:      0,
			},
			want: "",
		},
		{
			name: "regular content",
			parser: &Parser{
				tag:    "<tool_call>",
				buffer: "Here is some regular content:",
				n:      0,
			},
			want: "Here is some regular content:",
		},
		{
			name: "tools called",
			parser: &Parser{
				tag:    "<tool_call>",
				buffer: "I will call some tools. <tool_call>",
				n:      1,
			},
			want: "",
		},
		{
			name: "no tools called but tag found",
			parser: &Parser{
				tag:    "<tool_call>",
				buffer: "I will call some tools. <tool_call>{\"name\": \"get_temperature\"",
				n:      0,
			},
			want: "I will call some tools. ",
		},
		{
			name: "{ tag  with no tools",
			parser: &Parser{
				tag:    "{",
				buffer: "Here is an example json object: {\"name\": \"bob\"",
				n:      0,
			},
			want: "Here is an example json object: {\"name\": \"bob\"",
		},
		{
			name: "[ tag with no tools",
			parser: &Parser{
				tag:    "[",
				buffer: "Here is an example list of json objects: [{\"name\": \"bob\"",
				n:      0,
			},
			want: "Here is an example list of json objects: [{\"name\": \"bob\"",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := tt.parser.Content()
			if got != tt.want {
				t.Errorf("Content() = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestFindTag(t *testing.T) {
	cases := []struct {
		name   string
		buffer string
		tag    string
		want   int
	}{
		{
			name:   "no overlap",
			buffer: "hello world",
			tag:    "<tool_call>",
			want:   -1,
		},
		{
			name:   "full overlap",
			buffer: "<tool_call>",
			tag:    "<tool_call>",
			want:   0,
		},
		{
			name:   "over",
			buffer: "<tool_call>{\"name\"",
			tag:    "<tool_call>",
			want:   0,
		},
		{
			name:   "partial overlap",
			buffer: "text <tool_call>",
			tag:    "<tool_call>",
			want:   5,
		},
		{
			name:   "overlap with extra",
			buffer: "<tool_calls><tool_call>",
			tag:    "<tool_calls>",
			want:   0,
		},
		{
			name:   "delimiter longer than string",
			buffer: "<tool>",
			tag:    "<tool_call>",
			want:   -1,
		},
		{
			name:   "empty string",
			buffer: "",
			tag:    "<tool_call>",
			want:   -1,
		},
		{
			name:   "single char overlap",
			buffer: "test<",
			tag:    "<tool_call>",
			want:   4,
		},
		{
			name:   "partial tool call",
			buffer: "hello <tool_",
			tag:    "<tool_call>",
			want:   6,
		},
	}

	for _, tt := range cases {
		t.Run(tt.name, func(t *testing.T) {
			parser := &Parser{
				tag:    tt.tag,
				buffer: tt.buffer,
				n:      0,
			}
			got := parser.findTag()
			if got != tt.want {
				t.Errorf("findTag(%q, %q) = %d; want %d", tt.buffer, tt.tag, got, tt.want)
			}
		})
	}
}

func TestFindArguments(t *testing.T) {
	parser := &Parser{
		properties: []string{"format", "location"},
	}

	tests := []struct {
		name  string
		input string
		want  map[string]any
	}{
		{
			name:  "empty string",
			input: "",
			want:  nil,
		},
		{
			name:  "whitespace only",
			input: "   \n\t  ",
			want:  nil,
		},
		{
			name:  "unbalanced braces - missing closing",
			input: `{"format": "fahrenheit", "location": "San Francisco"`,
			want:  nil,
		},
		{
			name:  "unbalanced braces - extra closing",
			input: `{"format": "fahrenheit"}}`,
			want: map[string]any{
				"format": "fahrenheit",
			},
		},
		{
			name:  "invalid JSON",
			input: `{format: fahrenheit, location: "San Francisco"}`,
			want:  nil,
		},
		{
			name:  "valid arguments field",
			input: `{"name": "get_temperature", "arguments": {"format": "fahrenheit", "location": "San Francisco, CA"}}`,
			want: map[string]any{
				"format":   "fahrenheit",
				"location": "San Francisco, CA",
			},
		},
		{
			name:  "valid arguments field",
			input: `[tool]get_temperature[args]{"format": "fahrenheit", "location": "San Francisco, CA"}[end]`,
			want: map[string]any{
				"format":   "fahrenheit",
				"location": "San Francisco, CA",
			},
		},
		{
			name:  "valid arguments field in array",
			input: `[{"arguments": {"format": "fahrenheit", "location": "San Francisco, CA"}}`,
			want: map[string]any{
				"format":   "fahrenheit",
				"location": "San Francisco, CA",
			},
		},
		{
			name:  "nested deep",
			input: `{"function": {"name": "get_temperature", "arguments": {"format": "fahrenheit", "location": "San Francisco, CA"}}}`,
			want: map[string]any{
				"format":   "fahrenheit",
				"location": "San Francisco, CA",
			},
		},
		{
			name:  "one arg",
			input: `get_weather({"location": "San Francisco, CA"})`,
			want: map[string]any{
				"location": "San Francisco, CA",
			},
		},
		{
			name:  "deepseek",
			input: "<|tool▁calls▁begin|><|tool▁call▁begin|>function<|tool▁sep|>get_current_weather\n```json\n{\"location\": \"Tokyo\"}\n```<|tool▁call▁end|><|tool▁calls▁end|><|end▁of▁sentence|>",
			want: map[string]any{
				"location": "Tokyo",
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, _ := parser.findArguments(tt.input)

			if diff := cmp.Diff(got, tt.want); diff != "" {
				t.Errorf("scanArguments() args mismatch (-got +want):\n%s", diff)
			}
		})
	}
}
