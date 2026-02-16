package thinking

import (
	"testing"
	"text/template"
)

func TestInferThinkingTags(t *testing.T) {
	cases := []struct {
		desc           string
		tmplString     string
		wantOpeningTag string
		wantClosingTag string
	}{
		{
			desc: "basic",
			tmplString: `
			{{ if .Thinking}}
				/think
			{{ end }}
			{{- range $i, $_ := .Messages }}
				{{- $last := eq (len (slice $.Messages $i)) 1 -}}
				{{ if and $last .Thinking }}
					<think>{{ .Thinking }}</think>
				{{ end }}
			{{ end }}
		`,
			wantOpeningTag: "<think>",
			wantClosingTag: "</think>",
		},
		{
			desc: "doubly nested range",
			tmplString: `
			{{ if .Thinking}}
				/think
			{{ end }}
			{{- range $i, $_ := .Messages }}
				{{- range $j, $_ := .NotMessages }}
					{{- $last := eq (len (slice $.Messages $i)) 1 -}}
					{{ if and $last .Thinking }}
						<think>{{ .Thinking }}</think>
					{{ end }}
				{{ end }}
			{{ end }}
		`,
			wantOpeningTag: "",
			wantClosingTag: "",
		},
		{
			desc: "whitespace is trimmed",
			tmplString: `
			{{ if .Thinking}}
				/think
			{{ end }}
			{{- range $i, $_ := .Messages }}
				{{- $last := eq (len (slice $.Messages $i)) 1 -}}
				{{ if and $last .Thinking }}
					Some text before   {{ .Thinking }}    Some text after
				{{ end }}
			{{ end }}
		`,
			wantOpeningTag: "Some text before",
			wantClosingTag: "Some text after",
		},
		{
			desc: "qwen3",
			tmplString: `
{{- if or .System .Tools .Thinking }}<|im_start|>system
{{- if .System }}
{{ .System }}
{{- end }}
{{- if .Tools }}

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{{- range .Tools }}
{"type": "function", "function": {{ .Function }}}
{{- end }}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>
{{- end }}
{{- if .Thinking }}
/think
{{- else }}
/no_think
{{- end }}<|im_end|>
{{ end }}
{{- range $i, $_ := .Messages }}
{{- $last := eq (len (slice $.Messages $i)) 1 -}}
{{- if eq .Role "user" }}<|im_start|>user
{{ .Content }}<|im_end|>
{{ else if eq .Role "assistant" }}<|im_start|>assistant
{{ if and $last .Thinking }}
<think>{{ .Thinking }}</think>
{{ end }}
{{ if .Content }}{{ .Content }}
{{- else if .ToolCalls }}<tool_call>
{{ range .ToolCalls }}{"name": "{{ .Function.Name }}", "arguments": {{ .Function.Arguments }}}
{{ end }}</tool_call>
{{- end }}{{ if not $last }}<|im_end|>
{{ end }}
{{- else if eq .Role "tool" }}<|im_start|>user
<tool_response>
{{ .Content }}
</tool_response><|im_end|>
{{ end }}
{{- if and (ne .Role "assistant") $last }}<|im_start|>assistant
{{ end }}
{{- end }}
			`,
			wantOpeningTag: "<think>",
			wantClosingTag: "</think>",
		},
	}
	for _, c := range cases {
		tmpl := template.Must(template.New("test").Parse(c.tmplString))
		openingTag, closingTag := InferTags(tmpl)
		if openingTag != c.wantOpeningTag || closingTag != c.wantClosingTag {
			t.Errorf("case %q: got (%q,%q), want (%q,%q)", c.desc, openingTag, closingTag, c.wantOpeningTag, c.wantClosingTag)
		}
	}
}
