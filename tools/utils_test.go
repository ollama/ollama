package tools

import (
	"testing"
	gotmpl "text/template"

	"github.com/google/go-cmp/cmp"
	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/template"
)

func TestExtractToolCallsFormat(t *testing.T) {
	cases := []struct {
		name     string
		template string
		want     string
		found    bool
	}{
		{
			name:     "nil template",
			template: "",
			want:     "",
			found:    false,
		},
		{
			name:     "basic tool call with text",
			template: "{{if .ToolCalls}}Hello world{{end}}",
			want:     "Hello world",
			found:    true,
		},
		{
			name:     "tool call with json format",
			template: "{{if .ToolCalls}}```json\n{{end}}",
			want:     "```json\n",
			found:    true,
		},
		{
			name:     "tool call in range",
			template: "{{range .ToolCalls}}tool: {{.}}{{end}}",
			want:     "",
			found:    false,
		},
		{
			name:     "tool call with multiple text nodes",
			template: "{{if .ToolCalls}}First text{{if .Something}}inner{{end}}Second text{{end}}",
			want:     "First text",
			found:    true,
		},
		{
			name:     "nested if without tool calls",
			template: "{{if .Something}}{{if .OtherThing}}text{{end}}{{end}}",
			want:     "",
			found:    false,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			tmpl, err := gotmpl.New("test").Parse(tc.template)
			if err != nil && tc.template != "" {
				t.Fatalf("failed to parse template: %v", err)
			}

			got, found := extractToolCallsFormat(tmpl)
			if got != tc.want {
				t.Errorf("got text %q, want %q", got, tc.want)
			}
			if found != tc.found {
				t.Errorf("got found %v, want %v", found, tc.found)
			}
		})
	}
}

func TestToolPrefix(t *testing.T) {
	cases := []struct {
		name     string
		template string
		want     string
	}{
		{
			name:     "basic tool call with action prefix",
			template: "{{if .ToolCalls}}Action: ```json{{end}}",
			want:     "Action:",
		},
		{
			name:     "incomplete functools bracket",
			template: "{{if .ToolCalls}}functools[{{end}}",
			want:     "functools",
		},
		{
			name:     "tool call with angle brackets",
			template: "{{if .ToolCalls}}Hello, world! <tool_call>{{end}}",
			want:     "<tool_call>",
		},
		{
			name:     "multiple tool call formats",
			template: "{{if .ToolCalls}}[tool_call] <tool_call>{{end}}",
			want:     "[tool_call]",
		},
		{
			name:     "single angle bracket tool call",
			template: "{{if .ToolCalls}}<tool_call>{{end}}",
			want:     "<tool_call>",
		},
		{
			name:     "incomplete angle bracket after tool call",
			template: "{{if .ToolCalls}}[tool_call] <{{end}}",
			want:     "[tool_call]",
		},
		{
			name:     "angle bracket prefix with tool call",
			template: "{{if .ToolCalls}}> <tool_call>{{end}}",
			want:     "<tool_call>",
		},
		{
			name:     "uppercase tool call with incomplete bracket",
			template: "{{if .ToolCalls}}[TOOL_CALL] [{{end}}",
			want:     "[TOOL_CALL]",
		},
		{
			name:     "uppercase tool call with adjacent bracket",
			template: "{{if .ToolCalls}}[TOOL_CALL][{{end}}",
			want:     "[TOOL_CALL]",
		},
		{
			name:     "tool call with pipe delimiters",
			template: "{{if .ToolCalls}}<|tool_call|>{{end}}",
			want:     "<|tool_call|>",
		},
		{
			name:     "tool with no prefix",
			template: "{{if .ToolCalls}}{{end}}",
			want:     "",
		},
	}

	for _, tt := range cases {
		t.Run(tt.name, func(t *testing.T) {
			tmpl, err := gotmpl.New("test").Parse(tt.template)
			if err != nil {
				t.Fatalf("failed to parse template: %v", err)
			}
			got := toolPrefix(tmpl)
			if got != tt.want {
				t.Errorf("ToolToken(%q) = %q; want %q", tt.template, got, tt.want)
			}
		})
	}
}

func TestToolTemplate(t *testing.T) {
	cases := []struct {
		name     string
		template string
		want     bool
	}{
		{
			name:     "basic tool call range",
			template: "{{range .ToolCalls}}test{{end}}",
			want:     true,
		},
		{
			name:     "no tool calls",
			template: "{{range .Other}}test{{end}}",
			want:     false,
		},
		{
			name:     "nested tool calls",
			template: "{{range .Outer}}{{range .ToolCalls}}test{{end}}{{end}}",
			want:     true,
		},
		{
			name:     "empty template",
			template: "",
			want:     false,
		},
		{
			name:     "tool calls in if statement",
			template: "{{if .ToolCalls}}test{{end}}",
			want:     false,
		},
	}

	for _, tt := range cases {
		t.Run(tt.name, func(t *testing.T) {
			tmpl, err := gotmpl.New("test").Parse(tt.template)
			if err != nil {
				t.Fatalf("failed to parse template: %v", err)
			}

			parsed, err := template.Parse(tmpl.Root.String())
			if err != nil {
				t.Fatalf("failed to parse template: %v", err)
			}

			_, err = toolTemplate(parsed)
			if err != nil && tt.want {
				t.Errorf("toolTemplate() = %v; want %v", err, tt.want)
			}
		})
	}
}

func TestSuffixOverlap(t *testing.T) {
	cases := []struct {
		name string
		s    string
		d    string
		want int
	}{
		{
			name: "no overlap",
			s:    "hello world",
			d:    "<tool_call>",
			want: -1,
		},
		{
			name: "full overlap",
			s:    "<tool_call>",
			d:    "<tool_call>",
			want: 0,
		},
		{
			name: "partial overlap",
			s:    "text <tool_call>",
			d:    "<tool_call>",
			want: 5,
		},
		{
			name: "delimiter longer than string",
			s:    "<tool>",
			d:    "<tool_call>",
			want: -1,
		},
		{
			name: "empty string",
			s:    "",
			d:    "<tool_call>",
			want: -1,
		},
		{
			name: "empty delimiter",
			s:    "<tool_call>",
			d:    "",
			want: -1,
		},
		{
			name: "single char overlap",
			s:    "test<",
			d:    "<tool_call>",
			want: 4,
		},
		{
			name: "partial tool call",
			s:    "hello <tool_",
			d:    "<tool_call>",
			want: 6,
		},
	}

	for _, tt := range cases {
		t.Run(tt.name, func(t *testing.T) {
			got := suffixOverlap(tt.s, tt.d)
			if got != tt.want {
				t.Errorf("suffixOverlap(%q, %q) = %d; want %d", tt.s, tt.d, got, tt.want)
			}
		})
	}
}

func TestExtractToolArgs(t *testing.T) {
	cases := []struct {
		name     string
		template string
		want     string
		ok       bool
	}{
		{
			name:     "basic tool call with text after",
			template: `{{if .ToolCalls}}tool response{{end}}`,
			want:     "tool response",
			ok:       true,
		},
		{
			name:     "tool call with mixed content after",
			template: `{{if .ToolCalls}}<tool_call>{{.Something}}{{end}}`,
			want:     "<tool_call>",
			ok:       true,
		},
		{
			name:     "tool call with no text after",
			template: `{{if .ToolCalls}}{{.Something}}{{end}}`,
			want:     "",
			ok:       true,
		},
		{
			name:     "nested tool call",
			template: `{{if .Something}}{{if .ToolCalls}}[TOOL_CALL]{{end}}{{end}}`,
			want:     "[TOOL_CALL]",
			ok:       true,
		},
		{
			name:     "no tool calls",
			template: `{{if .Something}}no tools here{{end}}`,
			want:     "",
			ok:       false,
		},
		{
			name:     "empty template",
			template: ``,
			want:     "",
			ok:       false,
		},
		{
			name:     "multiple tool calls sections",
			template: `{{if .ToolCalls}}first{{end}}{{if .ToolCalls}}second{{end}}`,
			want:     "first",
			ok:       true,
		},
		{
			name:     "range over tool calls",
			template: `{{if .ToolCalls}}{{range .ToolCalls}}tool{{end}}{{end}}`,
			want:     "",
			ok:       true,
		},
		{
			name:     "tool calls with pipe delimiters",
			template: `{{if .ToolCalls}}<|tool|>{{end}}`,
			want:     "<|tool|>",
			ok:       true,
		},
		{
			name:     "tool calls with nested template",
			template: `{{if .ToolCalls}}{{template "tool" .}}{{end}}`,
			want:     "",
			ok:       true,
		},
		{
			name:     "tool calls with whitespace variations",
			template: `{{if .ToolCalls}}  tool  {{end}}`,
			want:     "  tool  ",
			ok:       true,
		},
	}

	for _, tt := range cases {
		t.Run(tt.name, func(t *testing.T) {
			tmpl, err := gotmpl.New("test").Parse(tt.template)
			if err != nil {
				t.Fatalf("failed to parse template: %v", err)
			}

			got, ok := extractToolCallsFormat(tmpl)
			if got != tt.want {
				t.Errorf("TextAfterToolCalls() got = %q, want %q", got, tt.want)
			}
			if ok != tt.ok {
				t.Errorf("TextAfterToolCalls() ok = %v, want %v", ok, tt.ok)
			}
		})
	}
}

func TestCollect(t *testing.T) {
	cases := []struct {
		name string
		obj  any
		want []map[string]any
	}{
		{
			name: "simple map",
			obj: map[string]any{
				"key": "value",
			},
			want: []map[string]any{
				{"key": "value"},
			},
		},
		{
			name: "nested map",
			obj: map[string]any{
				"outer": map[string]any{
					"inner": "value",
				},
			},
			want: []map[string]any{
				{"outer": map[string]any{"inner": "value"}},
				{"inner": "value"},
			},
		},
		{
			name: "array of maps",
			obj: []any{
				map[string]any{"key1": "val1"},
				map[string]any{"key2": "val2"},
			},
			want: []map[string]any{
				{"key1": "val1"},
				{"key2": "val2"},
			},
		},
		{
			name: "deeply nested",
			obj: map[string]any{
				"l1": map[string]any{
					"l2": map[string]any{
						"l3": "value",
					},
				},
			},
			want: []map[string]any{
				{"l1": map[string]any{"l2": map[string]any{"l3": "value"}}},
				{"l2": map[string]any{"l3": "value"}},
				{"l3": "value"},
			},
		},
		{
			name: "non-map value",
			obj:  "string",
			want: nil,
		},
	}

	for _, tt := range cases {
		t.Run(tt.name, func(t *testing.T) {
			got := collect(tt.obj)
			if len(got) != len(tt.want) {
				t.Errorf("collect() got %d maps, want %d", len(got), len(tt.want))
				return
			}

			// Compare each map in the result
			for i := range tt.want {
				if !mapsEqual(got[i], tt.want[i]) {
					t.Errorf("collect() map[%d] = %v, want %v", i, got[i], tt.want[i])
				}
			}
		})
	}
}

// mapsEqual compares two maps for deep equality
func mapsEqual(m1, m2 map[string]any) bool {
	if len(m1) != len(m2) {
		return false
	}
	for k, v1 := range m1 {
		v2, ok := m2[k]
		if !ok {
			return false
		}
		switch val1 := v1.(type) {
		case map[string]any:
			val2, ok := v2.(map[string]any)
			if !ok || !mapsEqual(val1, val2) {
				return false
			}
		default:
			if v1 != v2 {
				return false
			}
		}
	}
	return true
}

func TestParseJSONToolCalls(t *testing.T) {
	tests := []struct {
		name          string
		input         string
		nameField     string
		argsField     string
		wantToolCalls []api.ToolCall
		wantErr       error
	}{
		{
			name:      "valid single tool call",
			input:     `{"name": "test_tool", "arguments": {"arg1": "value1"}}`,
			nameField: "name",
			argsField: "arguments",
			wantToolCalls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name: "test_tool",
						Arguments: map[string]any{
							"arg1": "value1",
						},
					},
				},
			},
			wantErr: nil,
		},
		{
			name:          "incomplete JSON",
			input:         `{"name": "test_tool", "arguments": {"arg1": `,
			nameField:     "name",
			argsField:     "arguments",
			wantToolCalls: nil,
			wantErr:       errAccumulateMore,
		},
		{
			name:          "invalid JSON",
			input:         `not json at all`,
			nameField:     "name",
			argsField:     "arguments",
			wantToolCalls: nil,
			wantErr:       errInvalidToolCall,
		},
		{
			name:          "missing required fields",
			input:         `{"other": "field"}`,
			nameField:     "name",
			argsField:     "arguments",
			wantToolCalls: nil,
			wantErr:       errInvalidToolCall,
		},
		// Unlikely to hit this case as the parse would have already parsed the first JSON
		{
			name: "multiple tool calls in array",
			input: `[
				{"name": "tool1", "arguments": {"arg1": 1}},
				{"name": "tool2", "arguments": {"arg2": "value"}}
			]`,
			nameField: "name",
			argsField: "arguments",
			wantToolCalls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name: "tool1",
						Arguments: map[string]any{
							"arg1": float64(1),
						},
					},
				},
				{
					Function: api.ToolCallFunction{
						Name: "tool2",
						Arguments: map[string]any{
							"arg2": "value",
						},
					},
				},
			},
			wantErr: nil,
		},
		{
			name: "multiple tool calls without array",
			input: `
				{"name": "tool1", "arguments": {"arg1": 1}},
				{"name": "tool2", "arguments": {"arg2": "value"}}
			`,
			nameField: "name",
			argsField: "arguments",
			wantToolCalls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name: "tool1",
						Arguments: map[string]any{
							"arg1": float64(1),
						},
					},
				},
				{
					Function: api.ToolCallFunction{
						Name: "tool2",
						Arguments: map[string]any{
							"arg2": "value",
						},
					},
				},
			},
			wantErr: nil,
		},
		{
			name: "multiple tool calls with text after",
			input: `
				{"name": "tool1", "arguments": {"arg1": 1}} text
				{"name": "tool2", "arguments": {"arg2": "value"}} text
			`,
			nameField: "name",
			argsField: "arguments",
			wantToolCalls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name: "tool1",
						Arguments: map[string]any{
							"arg1": float64(1),
						},
					},
				},
				{
					Function: api.ToolCallFunction{
						Name: "tool2",
						Arguments: map[string]any{
							"arg2": "value",
						},
					},
				},
			},
			wantErr: nil,
		},
		{
			name: "second tool call in array",
			input: `
				, {"name": "tool2", "arguments": {"arg2": "value"}}
			`,
			nameField: "name",
			argsField: "arguments",
			wantToolCalls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name: "tool2",
						Arguments: map[string]any{
							"arg2": "value",
						},
					},
				},
			},
			wantErr: nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotCalls, err := parseJSONToolCalls(tt.input, tt.nameField, tt.argsField)

			if err != tt.wantErr {
				t.Errorf("parseJSONToolCalls() error = %v, want %v", err, tt.wantErr)
			}

			if len(gotCalls) != 0 && tt.wantErr != nil {
				t.Errorf("parseJSONToolCalls() valid = %v, want %v", len(gotCalls) == 0, tt.wantErr == nil)
			}

			if diff := cmp.Diff(gotCalls, tt.wantToolCalls); diff != "" {
				t.Errorf("parseJSONToolCalls() tool calls mismatch (-got +want):\n%s", diff)
			}
		})
	}
}
