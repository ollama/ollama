package server

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/template"
)

func readFile(t *testing.T, base, name string) *bytes.Buffer {
	t.Helper()

	bts, err := os.ReadFile(filepath.Join(base, name))
	if err != nil {
		t.Fatal(err)
	}

	return bytes.NewBuffer(bts)
}

func TestExecuteWithTools(t *testing.T) {
	p := filepath.Join("testdata", "tools")
	cases := []struct {
		model      string
		output     string
		ok         bool
		wellFormed bool
	}{
		{"mistral", `[TOOL_CALLS] [{"name": "get_current_weather", "arguments": {"format":"fahrenheit","location":"San Francisco, CA"}},{"name": "get_current_weather", "arguments": {"format":"celsius","location":"Toronto, Canada"}}]`, true, true},
		{"mistral", `[TOOL_CALLS] [{"name": "get_current_weather", "arguments": {"format":"fahrenheit","location":"San Francisco, CA"}},{"name": "get_current_weather", "arguments": {"format":"celsius","location":"Toronto, Canada"}}]

The temperature in San Francisco, CA is 70°F and in Toronto, Canada is 20°C.`, true, false},
		{"mistral", `[TOOL_CALLS] [{"name": "get_current_weather", "arguments": {"format":"fahrenheit","location":"San Francisco, CA"}},{"name": "get_current_weather", "arguments": {"format":"celsius","location":"Toronto, Canada"}},{"name": "get_current_weather", "arguments": {"format":"celsius","location":"To }]`, false, false},
		{"mistral", `I'm not aware of that information. However, I can suggest searching for the weather using the "get_current_weather" function:

		[{"name": "get_current_weather", "arguments": {"format":"fahrenheit","location":"San Francisco, CA"}},{"name": "get_current_weather", "arguments": {"format":"celsius","location":"Toronto, Canada"}}]`, true, false},
		{"mistral", " The weather in San Francisco, CA is 70°F and in Toronto, Canada is 20°C.", false, false},
		{"command-r-plus", "Action: ```json" + `
[
    {
        "tool_name": "get_current_weather",
        "parameters": {
            "format": "fahrenheit",
            "location": "San Francisco, CA"
        }
    },
    {
        "tool_name": "get_current_weather",
        "parameters": {
            "format": "celsius",
            "location": "Toronto, Canada"
        }
    }
]
` + "```", true, true},
		{"command-r-plus", " The weather in San Francisco, CA is 70°F and in Toronto, Canada is 20°C.", false, false},
		{"firefunction", ` functools[{"name": "get_current_weather", "arguments": {"format":"fahrenheit","location":"San Francisco, CA"}},{"name": "get_current_weather", "arguments": {"format":"celsius","location":"Toronto, Canada"}}]`, true, true},
		{"firefunction", " The weather in San Francisco, CA is 70°F and in Toronto, Canada is 20°C.", false, false},
		{"llama3-groq-tool-use", `<tool_call>
{"name": "get_current_weather", "arguments": {"format":"fahrenheit","location":"San Francisco, CA"}}
{"name": "get_current_weather", "arguments": {"format":"celsius","location":"Toronto, Canada"}}
</tool_call>`, true, true},
		{"xlam", `### Response:
{"tool_calls": [{"name": "get_current_weather", "arguments": {"format":"fahrenheit","location":"San Francisco, CA"}},{"name": "get_current_weather", "arguments": {"format":"celsius","location":"Toronto, Canada"}}]}`, true, true},
		{"nemotron", `<toolcall> {"name": "get_current_weather", "arguments": {"format":"fahrenheit","location":"San Francisco, CA"}},{"name": "get_current_weather", "arguments": {"format":"celsius","location":"Toronto, Canada"}}]} </toolcall>`, true, true},
	}

	var tools []api.Tool
	if err := json.Unmarshal(readFile(t, p, "tools.json").Bytes(), &tools); err != nil {
		t.Fatal(err)
	}

	var messages []api.Message
	if err := json.Unmarshal(readFile(t, p, "messages.json").Bytes(), &messages); err != nil {
		t.Fatal(err)
	}

	calls := []api.ToolCall{
		{
			Function: api.ToolCallFunction{
				Name: "get_current_weather",
				Arguments: api.ToolCallFunctionArguments{
					"format":   "fahrenheit",
					"location": "San Francisco, CA",
				},
			},
		},
		{
			Function: api.ToolCallFunction{
				Name: "get_current_weather",
				Arguments: api.ToolCallFunctionArguments{
					"format":   "celsius",
					"location": "Toronto, Canada",
				},
			},
		},
	}

	for _, tt := range cases {
		t.Run(tt.model, func(t *testing.T) {
			tmpl, err := template.Parse(readFile(t, p, fmt.Sprintf("%s.gotmpl", tt.model)).String())
			if err != nil {
				t.Fatal(err)
			}

			t.Run("template", func(t *testing.T) {
				var actual bytes.Buffer
				if err := tmpl.Execute(&actual, template.Values{Tools: tools, Messages: messages}); err != nil {
					t.Fatal(err)
				}

				if diff := cmp.Diff(actual.String(), readFile(t, p, fmt.Sprintf("%s.out", tt.model)).String()); diff != "" {
					t.Errorf("mismatch (-got +want):\n%s", diff)
				}
			})

			t.Run("prefix", func(t *testing.T) {
				m := &Model{Template: tmpl}
				m.addToolPrefix()

				if tt.wellFormed {
					if len(m.ToolPrefix) == 0 {
						t.Fatalf("No tool prefix detected")
					}

					if !strings.HasPrefix(strings.TrimSpace(tt.output), m.ToolPrefix) {
						t.Fatalf("incorrect tool prefix: \"%s\", \"%s\"", m.ToolPrefix, tt.output)
					}
				}
			})

			t.Run("parse", func(t *testing.T) {
				m := &Model{Template: tmpl}
				actual, ok := m.parseToolCalls(tt.output)
				if ok != tt.ok {
					t.Fatalf("expected %t, got %t", tt.ok, ok)
				}

				if tt.ok {
					if diff := cmp.Diff(actual, calls); diff != "" {
						t.Errorf("mismatch (-got +want):\n%s", diff)
					}
				}
			})
		})
	}
}

func TestParseObjects(t *testing.T) {
	tests := []struct {
		input string
		want  []map[string]any
	}{
		{
			input: `[{"name": "get_current_weather", "arguments": {"format":"fahrenheit","location":"San Francisco, CA"}},{"name": "get_current_weather", "arguments": {"format":"celsius","location":"Toronto, Canada"}}]`,
			want: []map[string]any{
				{"name": "get_current_weather", "arguments": map[string]any{"format": "fahrenheit", "location": "San Francisco, CA"}},
				{"name": "get_current_weather", "arguments": map[string]any{"format": "celsius", "location": "Toronto, Canada"}},
			},
		},
		{
			input: `<toolcall>{"name": "get_current_weather", "arguments": {"format":"fahrenheit","location":"San Francisco, CA"}} </toolcall>`,
			want: []map[string]any{
				{"name": "get_current_weather", "arguments": map[string]any{"format": "fahrenheit", "location": "San Francisco, CA"}},
			},
		},
		{
			input: `<toolcall>{"name": "get_current_weather", "arguments": {"format":"fahrenheit","location":"San Francisco, CA"}} </toolcall> <toolcall>{"name": "get_current_weather", "arguments": {"format":"celsius","location":"Toronto, ON"}} </toolcall>`,
			want: []map[string]any{
				{"name": "get_current_weather", "arguments": map[string]any{"format": "fahrenheit", "location": "San Francisco, CA"}},
				{"name": "get_current_weather", "arguments": map[string]any{"format": "celsius", "location": "Toronto, ON"}},
			},
		},
		{
			input: `{"name": "get_current_weather", "arguments": `,
			want:  nil,
		},
	}

	for _, tc := range tests {
		t.Run(tc.input, func(t *testing.T) {
			got := parseObjects(tc.input)

			if diff := cmp.Diff(got, tc.want); diff != "" {
				t.Errorf("mismatch (-got +want):\n%s", diff)
			}
		})
	}
}

func TestAddToolPrefix(t *testing.T) {
	tests := []struct {
		name     string
		template string
		want     string
	}{
		{
			name:     "prefix_from_previous_text_node",
			template: `Previous text node{{- range .ToolCalls}}{{.name}}{{end}}`,
			want:     "Previous text node",
		},
		{
			name:     "prefix_from_range_node",
			template: `{{- range .ToolCalls}}[TOOL_CALLS]{{.name}}{{end}}`,
			want:     "[TOOL_CALLS]",
		},
		{
			name:     "prefix_with_extra_whitespace",
			template: `    Previous text with spaces    {{- range .ToolCalls}}{{.name}}{{end}}`,
			want:     "Previous text with spaces",
		},
		{
			name:     "prefix_with_newlines",
			template: "First line\nSecond line\n{{- range .ToolCalls}}{{.name}}{{end}}",
			want:     "First line\nSecond line",
		},
		{
			name: "tool_calls_json_template",
			template: `{{ if .Content }}{{ .Content }}{{- else if .ToolCalls }}<tool_call>
{{ range .ToolCalls }}{"name": "{{ .Function.Name }}", "arguments": {{ .Function.Arguments }}}{{ end }}</tool_call>
{{ end }}`,
			want: `<tool_call>`,
		},
		{
			name: "mistral_tool_calls_template",
			template: `{{- if .Content }} {{ .Content }}
{{- else if .ToolCalls }}[TOOL_CALLS] [
{{- range .ToolCalls }}{"name": "{{ .Function.Name }}", "arguments": {{ .Function.Arguments }}}
{{- end }}]
{{- end }}</s>`,
			want: "[TOOL_CALLS] [",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tmpl, err := template.Parse(tt.template)
			if err != nil {
				t.Fatalf("failed to parse template: %v", err)
			}

			m := &Model{Template: tmpl}
			m.addToolPrefix()

			if m.ToolPrefix != tt.want {
				t.Errorf("incorrect tool prefix:\ngot:  %q\nwant: %q", m.ToolPrefix, tt.want)
			}
		})
	}
}
