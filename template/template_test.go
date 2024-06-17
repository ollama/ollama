package template

import (
	"bufio"
	"bytes"
	"encoding/json"
	"io"
	"os"
	"path/filepath"
	"slices"
	"testing"
	"text/template"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llm"
)

func TestNamed(t *testing.T) {
	f, err := os.Open(filepath.Join("testdata", "templates.jsonl"))
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		var ss map[string]string
		if err := json.Unmarshal(scanner.Bytes(), &ss); err != nil {
			t.Fatal(err)
		}

		for k, v := range ss {
			t.Run(k, func(t *testing.T) {
				kv := llm.KV{"tokenizer.chat_template": v}
				s := kv.ChatTemplate()
				r, err := Named(s)
				if err != nil {
					t.Fatal(err)
				}

				if r.Name != k {
					t.Errorf("expected %q, got %q", k, r.Name)
				}

				var b bytes.Buffer
				if _, err := io.Copy(&b, r.Reader()); err != nil {
					t.Fatal(err)
				}

				tmpl, err := template.New(s).Parse(b.String())
				if err != nil {
					t.Fatal(err)
				}

				if tmpl.Tree.Root.String() == "" {
					t.Errorf("empty %s template", k)
				}
			})
		}
	}
}

func TestParse(t *testing.T) {
	cases := []struct {
		template string
		vars     []string
	}{
		{"{{ .Prompt }}", []string{"prompt", "response"}},
		{"{{ .System }} {{ .Prompt }}", []string{"prompt", "response", "system"}},
		{"{{ .System }} {{ .Prompt }} {{ .Response }}", []string{"prompt", "response", "system"}},
		{"{{ with .Tools }}{{ . }}{{ end }} {{ .System }} {{ .Prompt }}", []string{"prompt", "response", "system", "tools"}},
		{"{{ range .Messages }}{{ .Role }} {{ .Content }}{{ end }}", []string{"content", "messages", "role"}},
		{"{{ range .Messages }}{{ if eq .Role \"system\" }}SYSTEM: {{ .Content }}{{ else if eq .Role \"user\" }}USER: {{ .Content }}{{ else if eq .Role \"assistant\" }}ASSISTANT: {{ .Content }}{{ end }}{{ end }}", []string{"content", "messages", "role"}},
	}

	for _, tt := range cases {
		t.Run("", func(t *testing.T) {
			tmpl, err := Parse(tt.template)
			if err != nil {
				t.Fatal(err)
			}

			vars := tmpl.Vars()
			if !slices.Equal(tt.vars, vars) {
				t.Errorf("expected %v, got %v", tt.vars, vars)
			}
		})
	}
}

func TestExecuteWithMessages(t *testing.T) {
	cases := []struct {
		templates []string
		values    Values
		expected  string
	}{
		{
			[]string{
				`[INST] {{ if .System }}{{ .System }}{{ print "\n\n" }}{{ end }}{{ .Prompt }}[/INST] `,
				`[INST] {{ if .System }}{{ .System }}{{ print "\n\n" }}{{ end }}{{ .Prompt }}[/INST] {{ .Response }}`,
				`{{- range .Messages }}
{{- if eq .Role "user" }}[INST] {{ if and (isLastMessage $.Messages .) $.System }}{{ $.System }}{{ print "\n\n" }}
{{- end }}{{ .Content }}[/INST] {{ else if eq .Role "assistant" }}{{ .Content }}
{{- end }}
{{- end }}`,
			},
			Values{
				Messages: []api.Message{
					{Role: "user", Content: "Hello friend!"},
					{Role: "assistant", Content: "Hello human!"},
					{Role: "user", Content: "Yay!"},
				},
			},
			`[INST] Hello friend![/INST] Hello human![INST] Yay![/INST] `,
		},
		{
			[]string{
				`[INST] {{ if .System }}{{ .System }}{{ print "\n\n" }}{{ end }}{{ .Prompt }}[/INST] `,
				`[INST] {{ if .System }}{{ .System }}{{ print "\n\n" }}{{ end }}{{ .Prompt }}[/INST] {{ .Response }}`,
				`
{{- range .Messages }}
{{- if eq .Role "user" }}[INST] {{ if and (isLastMessage $.Messages .) $.System }}{{ $.System }}{{ print "\n\n" }}
{{- end }}{{ .Content }}[/INST] {{ else if eq .Role "assistant" }}{{ .Content }}
{{- end }}
{{- end }}`,
			},
			Values{
				Messages: []api.Message{
					{Role: "system", Content: "You are a helpful assistant!"},
					{Role: "user", Content: "Hello friend!"},
					{Role: "assistant", Content: "Hello human!"},
					{Role: "user", Content: "Yay!"},
				},
			},
			`[INST] Hello friend![/INST] Hello human![INST] You are a helpful assistant!

Yay![/INST] `,
		},
		{
			[]string{
				`{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
{{ .Response }}<|im_end|>
`,
				`
{{- range .Messages }}
{{- if and (eq .Role "user") (isLastMessage $.Messages .) $.System }}<|im_start|>system
{{ $.System }}<|im_end|>{{ print "\n" }}
{{- end }}<|im_start|>{{ .Role }}
{{ .Content }}<|im_end|>{{ print "\n" }}
{{- end }}<|im_start|>assistant
`,
			},
			Values{
				Messages: []api.Message{
					{Role: "system", Content: "You are a helpful assistant!"},
					{Role: "user", Content: "Hello friend!"},
					{Role: "assistant", Content: "Hello human!"},
					{Role: "user", Content: "Yay!"},
				},
			},
			`<|im_start|>user
Hello friend!<|im_end|>
<|im_start|>assistant
Hello human!<|im_end|>
<|im_start|>system
You are a helpful assistant!<|im_end|>
<|im_start|>user
Yay!<|im_end|>
<|im_start|>assistant
`,
		},
		{
			[]string{
				`{{ if .Prompt }}Question: {{ .Prompt }}

{{ end }}Answer: {{ .Response }}

`,
				`
{{- range .Messages }}
{{- if eq .Role "user" }}Question: {{ .Content }}{{ print "\n\n" }}
{{- else if eq .Role "assistant" }}Answer: {{ .Content }}{{ print "\n\n" }}
{{- end }}
{{- end }}Answer: `,
			},
			Values{
				Messages: []api.Message{
					{Role: "user", Content: "What's in this image?", Images: []api.ImageData{[]byte("")}},
					{Role: "assistant", Content: "It's a hot dog."},
					{Role: "user", Content: "What's in _this_ image?"},
					{Role: "user", Images: []api.ImageData{[]byte("")}},
					{Role: "user", Content: "Is it a hot dog?"},
				},
			},
			`Question: [img-0] What's in this image?

Answer: It's a hot dog.

Question: What's in _this_ image?

[img-1]

Is it a hot dog?

Answer: `,
		},
	}

	for _, tt := range cases {
		t.Run("", func(t *testing.T) {
			for _, tmpl := range tt.templates {
				t.Run("", func(t *testing.T) {
					tmpl, err := Parse(tmpl)
					if err != nil {
						t.Fatal(err)
					}

					var b bytes.Buffer
					if err := tmpl.Execute(&b, tt.values); err != nil {
						t.Fatal(err)
					}

					if b.String() != tt.expected {
						t.Errorf("expected\n%s,\ngot\n%s", tt.expected, b.String())
					}
				})
			}
		})
	}
}
