package server

// import (
// 	"testing"
// 	gotmpl "text/template"
// )

// func TestToolToken(t *testing.T) {
// 	cases := []struct {
// 		name     string
// 		template string
// 		want     string
// 		ok       bool
// 	}{
// 		{
// 			name:     "basic tool call with action prefix",
// 			template: "{{if .ToolCalls}}Action: ```json{{end}}",
// 			want:     "Action:",
// 			ok:       true,
// 		},
// 		{
// 			name:     "incomplete functools bracket",
// 			template: "{{if .ToolCalls}}functools[{{end}}",
// 			want:     "functools",
// 			ok:       true,
// 		},
// 		{
// 			name:     "tool call with angle brackets",
// 			template: "{{if .ToolCalls}}Hello, world! <tool_call>{{end}}",
// 			want:     "<tool_call>",
// 			ok:       true,
// 		},
// 		{
// 			name:     "multiple tool call formats",
// 			template: "{{if .ToolCalls}}[tool_call] <tool_call>{{end}}",
// 			want:     "[tool_call]",
// 			ok:       true,
// 		},
// 		{
// 			name:     "single angle bracket tool call",
// 			template: "{{if .ToolCalls}}<tool_call>{{end}}",
// 			want:     "<tool_call>",
// 			ok:       true,
// 		},
// 		{
// 			name:     "incomplete angle bracket after tool call",
// 			template: "{{if .ToolCalls}}[tool_call] <{{end}}",
// 			want:     "[tool_call]",
// 			ok:       true,
// 		},
// 		{
// 			name:     "angle bracket prefix with tool call",
// 			template: "{{if .ToolCalls}}> <tool_call>{{end}}",
// 			want:     "<tool_call>",
// 			ok:       true,
// 		},
// 		{
// 			name:     "uppercase tool call with incomplete bracket",
// 			template: "{{if .ToolCalls}}[TOOL_CALL] [{{end}}",
// 			want:     "[TOOL_CALL]",
// 			ok:       true,
// 		},
// 		{
// 			name:     "uppercase tool call with adjacent bracket",
// 			template: "{{if .ToolCalls}}[TOOL_CALL][{{end}}",
// 			want:     "[TOOL_CALL]",
// 			ok:       true,
// 		},
// 		{
// 			name:     "tool call with pipe delimiters",
// 			template: "{{if .ToolCalls}}<|tool_call|>{{end}}",
// 			want:     "<|tool_call|>",
// 			ok:       true,
// 		},
// 	}

// 	for _, tt := range cases {
// 		t.Run(tt.name, func(t *testing.T) {
// 			tmpl, err := gotmpl.New("test").Parse(tt.template)
// 			if err != nil {
// 				t.Fatalf("failed to parse template: %v", err)
// 			}
// 			got, ok := ToolPrefix(tmpl)
// 			if got != tt.want {
// 				t.Errorf("ToolToken(%q) = %q; want %q", tt.template, got, tt.want)
// 			}
// 			if ok != tt.ok {
// 				t.Errorf("ToolToken(%q) = %v; want %v", tt.template, ok, tt.ok)
// 			}
// 		})
// 	}
// }

// func TestTextAfterToolCalls(t *testing.T) {
// 	cases := []struct {
// 		name     string
// 		template string
// 		want     string
// 		ok       bool
// 	}{
// 		{
// 			name:     "basic tool call with text after",
// 			template: `{{if .ToolCalls}}tool response{{end}}`,
// 			want:     "tool response",
// 			ok:       true,
// 		},
// 		{
// 			name:     "tool call with mixed content after",
// 			template: `{{if .ToolCalls}}<tool_call>{{.Something}}{{end}}`,
// 			want:     "<tool_call>",
// 			ok:       true,
// 		},
// 		{
// 			name:     "tool call with no text after",
// 			template: `{{if .ToolCalls}}{{.Something}}{{end}}`,
// 			want:     "",
// 			ok:       true,
// 		},
// 		{
// 			name:     "nested tool call",
// 			template: `{{if .Something}}{{if .ToolCalls}}[TOOL_CALL]{{end}}{{end}}`,
// 			want:     "[TOOL_CALL]",
// 			ok:       true,
// 		},
// 		{
// 			name:     "no tool calls",
// 			template: `{{if .Something}}no tools here{{end}}`,
// 			want:     "",
// 			ok:       false,
// 		},
// 		{
// 			name:     "empty template",
// 			template: ``,
// 			want:     "",
// 			ok:       false,
// 		},
// 		{
// 			name:     "multiple tool calls sections",
// 			template: `{{if .ToolCalls}}first{{end}}{{if .ToolCalls}}second{{end}}`,
// 			want:     "first",
// 			ok:       true,
// 		},
// 		{
// 			name:     "range over tool calls",
// 			template: `{{if .ToolCalls}}{{range .ToolCalls}}tool{{end}}{{end}}`,
// 			want:     "",
// 			ok:       true,
// 		},
// 		{
// 			name:     "tool calls with pipe delimiters",
// 			template: `{{if .ToolCalls}}<|tool|>{{end}}`,
// 			want:     "<|tool|>",
// 			ok:       true,
// 		},
// 		{
// 			name:     "tool calls with nested template",
// 			template: `{{if .ToolCalls}}{{template "tool" .}}{{end}}`,
// 			want:     "",
// 			ok:       true,
// 		},
// 		{
// 			name:     "tool calls with whitespace variations",
// 			template: `{{if .ToolCalls}}  tool  {{end}}`,
// 			want:     "  tool  ",
// 			ok:       true,
// 		},
// 	}

// 	for _, tt := range cases {
// 		t.Run(tt.name, func(t *testing.T) {
// 			tmpl, err := gotmpl.New("test").Parse(tt.template)
// 			if err != nil {
// 				t.Fatalf("failed to parse template: %v", err)
// 			}

// 			got, ok := extractToolCallsTemplate(tmpl)
// 			if got != tt.want {
// 				t.Errorf("TextAfterToolCalls() got = %q, want %q", got, tt.want)
// 			}
// 			if ok != tt.ok {
// 				t.Errorf("TextAfterToolCalls() ok = %v, want %v", ok, tt.ok)
// 			}
// 		})
// 	}
// }
