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

	// deepseek, err := template.New("deepseek").Parse("{{if .ToolCalls}}<|tool▁calls▁begin|>{{range .ToolCalls}}<|tool▁call▁begin|>function<|tool▁sep|>get_current_weather\n```json\n{\"location\": \"Tokyo\"}\n```<|tool▁call▁end|>{{end}}<|tool▁calls▁end|><|end▁of▁sentence|>{{end}}")
	// if err != nil {
	// 	t.Fatalf("Failed to parse template: %v", err)
	// }

	// json, err := template.New("json").Parse(`{{if .ToolCalls}}{{range .ToolCalls}}{"name": "{{.Function.Name}}", "arguments": {{.Function.Arguments}}}{{end}}{{end}}`)
	// if err != nil {
	// 	t.Fatalf("Failed to parse template: %v", err)
	// }

	// list, err := template.New("list").Parse(`{{if .ToolCalls}}[{{range .ToolCalls}}{"name": "{{.Function.Name}}", "arguments": {{.Function.Arguments}}}{{end}}]{{end}}`)
	// if err != nil {
	// 	t.Fatalf("Failed to parse template: %v", err)
	// }

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
		input   string
		tmpl    *template.Template
		content string
		calls   []api.ToolCall
	}{
		{
			name:    "no tool calls - just text",
			input:   "Hello, how can I help you today?",
			content: "Hello, how can I help you today?",
			tmpl:    qwen,
			calls:   nil,
		},
		{
			name:    "empty input",
			input:   "",
			content: "",
			tmpl:    qwen,
			calls:   nil,
		},
		{
			name:    "tool call",
			input:   `<tool_call>{"name": "get_conditions", "arguments": {"location": "San Francisco"}}</tool_call>`,
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
			input:   `Let me check the weather. <tool_call>{"name": "get_temperature", "arguments": {"city": "New York"}}</tool_call>`,
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
			input:   `Okay, let's call both tools! <tool_call>{"name": "get_temperature", "arguments": {"city": "London", "format": "fahrenheit"}}</tool_call><tool_call>{"name": "get_conditions", "arguments": {"location": "Tokyo"}}</tool_call>`,
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
		// {
		// 	name:            "text with two tool calls",
		// 	input:           `I'll check both cities. <tool_call>{"name": "get_weather", "arguments": {"location": "Paris"}}</tool_call> and also <tool_call>{"name": "get_weather", "arguments": {"location": "Berlin"}}</tool_call>`,
		// 	expectedContent: "I'll check both cities. ",
		// 	expectedCalls: []api.ToolCall{
		// 		{
		// 			Function: api.ToolCallFunction{
		// 				Index: 0,
		// 				Name:  "get_weather",
		// 				Arguments: api.ToolCallFunctionArguments{
		// 					"location": "Paris",
		// 				},
		// 			},
		// 		},
		// 		{
		// 			Function: api.ToolCallFunction{
		// 				Index: 1,
		// 				Name:  "get_weather",
		// 				Arguments: api.ToolCallFunctionArguments{
		// 					"location": "Berlin",
		// 				},
		// 			},
		// 		},
		// 	},
		// },
		// {
		// 	name:            "three tool calls with text",
		// 	input:           `Weather check: <tool_call>{"name": "get_weather", "arguments": {"location": "Miami"}}</tool_call><tool_call>{"name": "get_weather", "arguments": {"location": "Seattle"}}</tool_call>, and <tool_call>{"name": "get_weather", "arguments": {"location": "Denver"}}</tool_call>`,
		// 	expectedContent: "Weather check: ",
		// 	expectedCalls: []api.ToolCall{
		// 		{
		// 			Function: api.ToolCallFunction{
		// 				Index: 0,
		// 				Name:  "get_weather",
		// 				Arguments: api.ToolCallFunctionArguments{
		// 					"location": "Miami",
		// 				},
		// 			},
		// 		},
		// 		{
		// 			Function: api.ToolCallFunction{
		// 				Index: 1,
		// 				Name:  "get_weather",
		// 				Arguments: api.ToolCallFunctionArguments{
		// 					"location": "Seattle",
		// 				},
		// 			},
		// 		},
		// 		{
		// 			Function: api.ToolCallFunction{
		// 				Index: 2,
		// 				Name:  "get_weather",
		// 				Arguments: api.ToolCallFunctionArguments{
		// 					"location": "Denver",
		// 				},
		// 			},
		// 		},
		// 	},
		// },
		// {
		// 	name:            "invalid tool call - unknown function",
		// 	input:           `<tool_call>{"name": "unknown_function", "arguments": {"param": "value"}}</tool_call>`,
		// 	expectedContent: "",
		// 	expectedCalls:   nil,
		// },
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create a fresh parser for each test
			parser := NewParser(tt.tmpl, tools)

			calls, content := parser.Add(tt.input)

			// Verify content
			if content != tt.content {
				t.Errorf("Expected content %q, got %q", tt.content, content)
			}

			// Verify tool calls
			if len(calls) != len(tt.calls) {
				t.Fatalf("Expected %d tool calls, got %d", len(tt.calls), len(calls))
			}

			for i, expectedCall := range tt.calls {
				if diff := cmp.Diff(calls[i], expectedCall); diff != "" {
					t.Errorf("Tool call %d mismatch (-got +want):\n%s", i, diff)
				}
			}
		})
	}
}

func TestIndexOverlap(t *testing.T) {
	cases := []struct {
		name   string
		s      string
		prefix string
		want   int
	}{
		{
			name:   "no overlap",
			s:      "hello world",
			prefix: "<tool_call>",
			want:   -1,
		},
		{
			name:   "full overlap",
			s:      "<tool_call>",
			prefix: "<tool_call>",
			want:   0,
		},
		{
			name:   "over",
			s:      "<tool_call>{\"name\"",
			prefix: "<tool_call>",
			want:   -1,
		},
		{
			name:   "partial overlap",
			s:      "text <tool_call>",
			prefix: "<tool_call>",
			want:   5,
		},
		{
			name:   "delimiter longer than string",
			s:      "<tool>",
			prefix: "<tool_call>",
			want:   -1,
		},
		{
			name:   "empty string",
			s:      "",
			prefix: "<tool_call>",
			want:   -1,
		},
		{
			name:   "empty delimiter",
			s:      "<tool_call>",
			prefix: "",
			want:   -1,
		},
		{
			name:   "single char overlap",
			s:      "test<",
			prefix: "<tool_call>",
			want:   4,
		},
		{
			name:   "partial tool call",
			s:      "hello <tool_",
			prefix: "<tool_call>",
			want:   6,
		},
	}

	for _, tt := range cases {
		t.Run(tt.name, func(t *testing.T) {
			got := indexOverlap(tt.s, tt.prefix)
			if got != tt.want {
				t.Errorf("indexOverlap(%q, %q) = %d; want %d", tt.s, tt.prefix, got, tt.want)
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
