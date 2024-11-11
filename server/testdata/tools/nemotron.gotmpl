{{- if (or .Tools .System) }}<extra_id_0>System
{{ if .System }}{{ .System }}


{{ end }}
{{- if .Tools }}
{{- range .Tools }}<tool> {{ . }} </tool>{{ end }}


{{ end }}
{{- end }}
{{- range $i, $m := .Messages }}
{{- $last := eq (len (slice $.Messages $i)) 1 -}}
{{- if eq .Role "user" }}<extra_id_1>User
{{ .Content }}
{{- if $last }}
<extra_id_1>Assistant
{{- end }}
{{ else if eq .Role "tool" }}<extra_id_1>Tool
{{ .Content }}
{{- if $last }}
<extra_id_1>Assistant
{{- end }}
{{ else if eq .Role "assistant" }}<extra_id_1>Assistant
{{- if .ToolCalls }}
{{ range .ToolCalls }}<toolcall> {"name": "{{ .Function.Name }}", "arguments": {{ .Function.Arguments }}} </toolcall> {{ end }}
{{ else }}
{{ .Content }}
{{- if not $last }}
{{ end }}
{{- end }}
{{- end }}
{{- end }}