package tools

import (
	"testing"
	"text/template"
)

func TestParseTag(t *testing.T) {
	cases := []struct {
		name     string
		template string
		want     string
	}{
		{
			name:     "empty",
			template: "",
			want:     "{",
		},
		{
			name:     "no tag",
			template: "{{if .ToolCalls}}{{end}}",
			want:     "{",
		},
		{
			name:     "no tag with range",
			template: "{{if .ToolCalls}}{{range .ToolCalls}}{{ . }}{{end}}{{end}}",
			want:     "{",
		},
		{
			name:     "tool call with json format",
			template: "{{if .ToolCalls}}```json\n{{end}}",
			want:     "```json",
		},
		{
			name:     "square brackets",
			template: "{{if .ToolCalls}}[{{range .ToolCalls}}{{ . }}{{end}}]{{end}}",
			want:     "[",
		},
		{
			name:     "square brackets with whitespace",
			template: "{{if .ToolCalls}}\n [ {{range .ToolCalls}}{{ . }}{{end}}]{{end}}",
			want:     "[",
		},
		{
			name:     "tailing ]",
			template: "{{if .ToolCalls}}{{range .ToolCalls}}{{ . }}{{end}}]{{end}}",
			want:     "{",
		},
		{
			name:     "whitespace only",
			template: "{{if .ToolCalls}} {{range .ToolCalls}}{{ . }}{{end}}{{end}}",
			want:     "{",
		},
		{
			name:     "whitespace only in range",
			template: "{{if .ToolCalls}}{{range .ToolCalls}}\n{{ . }}\n{{end}}{{end}}",
			want:     "{",
		},
		{
			name:     "json objects",
			template: `{{if .ToolCalls}}{{range .ToolCalls}}{"name": "{{ .Function.Name }}", "arguments": {{ .Function.Arguments }}}{{end}}{{end}}`,
			want:     "{",
		},
		{
			name:     "json objects with whitespace",
			template: "{{if .ToolCalls}}{{range .ToolCalls}}\n{\"name\": \"{{ .Function.Name }}\", \"arguments\": {{ .Function.Arguments }}}{{end}}{{end}}",
			want:     "{",
		},
		{
			name:     "json objects with CRLF",
			template: "{{if .ToolCalls}}{{range .ToolCalls}}\r\n{\"name\": \"{{ .Function.Name }}\", \"arguments\": {{ .Function.Arguments }}}{{end}}{{end}}",
			want:     "{",
		},
		{
			name:     "json objects with whitespace before and after range",
			template: "{{if .ToolCalls}}\n{{range .ToolCalls}}\n{\"name\": \"{{ .Function.Name }}\", \"arguments\": {{ .Function.Arguments }}}\r\n{{end}}\r\n{{end}}",
			want:     "{",
		},
		{
			name:     "before and after range",
			template: "{{if .ToolCalls}}<|tool▁calls▁begin|>{{range .ToolCalls}}<|tool▁call▁begin|>functionget_current_weather\n```json\n{\"location\": \"Tokyo\"}\n```<|tool▁call▁end|>\n{{end}}<|tool▁calls▁end|>{{end}}",
			want:     "<|tool▁calls▁begin|>",
		},
		{
			name:     "after range",
			template: "{{if .ToolCalls}}{{range .ToolCalls}}<tool_call>{\"name\": \"{{ .Function.Name }}\", \"arguments\": {{ .Function.Arguments }}}</tool_call>{{end}}{{end}}",
			want:     "<tool_call>",
		},
		{
			name:     "after range with leading whitespace before range",
			template: "{{if .ToolCalls}}\n{{range .ToolCalls}}<tool_call>{\"name\": \"{{ .Function.Name }}\", \"arguments\": {{ .Function.Arguments }}}</tool_call>{{end}}{{end}}",
			want:     "<tool_call>",
		},
		{
			name:     "tool call in range with {",
			template: `{{if .ToolCalls}}{{range .ToolCalls}}<tool_call>{"name": "{{ .Function.Name }}", "arguments": {{ .Function.Arguments }}}<tool_call>{{end}}{{end}}`,
			want:     "<tool_call>",
		},
		{
			name:     "tool call with multiple text nodes",
			template: "{{if .ToolCalls}}First text{{if .Something}}inner{{end}}Second text{{end}}",
			want:     "First text",
		},
		{
			name:     "action tag",
			template: "{{if .ToolCalls}}Action: ```json{{end}}",
			want:     "Action: ```json",
		},
		{
			name:     "incomplete functools bracket",
			template: "{{if .ToolCalls}}functools[{{end}}",
			want:     "functools[",
		},
		{
			name:     "uppercase tool call with incomplete bracket",
			template: "{{if .ToolCalls}}[TOOL_CALL] [{{end}}",
			want:     "[TOOL_CALL] [",
		},
		{
			name:     "uppercase tool call with adjacent bracket",
			template: "{{if .ToolCalls}}[TOOL_CALL][{{end}}",
			want:     "[TOOL_CALL][",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			tmpl, err := template.New("test").Parse(tc.template)
			if err != nil && tc.template != "" {
				t.Fatalf("failed to parse template: %v", err)
			}

			got := parseTag(tmpl)
			if got != tc.want {
				t.Errorf("got text %q, want %q", got, tc.want)
			}
		})
	}
}
