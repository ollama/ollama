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
	type template struct {
		name     string
		template string
	}
	cases := []struct {
		name      string
		templates []template
		values    Values
		expected  string
	}{
		{
			"mistral",
			[]template{
				{"no response", `[INST] {{ if .System }}{{ .System }}{{ "\n\n" }}{{ end }}{{ .Prompt }}[/INST] `},
				{"response", `[INST] {{ if .System }}{{ .System }}{{ "\n\n" }}{{ end }}{{ .Prompt }}[/INST] {{ .Response }}`},
				{"messages", `{{- range $index, $_ := .Messages }}
{{- if eq .Role "user" }}[INST] {{ if and (eq (len (slice $.Messages $index)) 1) $.System }}{{ $.System }}{{ "\n\n" }}
{{- end }}{{ .Content }}[/INST] {{ else if eq .Role "assistant" }}{{ .Content }}
{{- end }}
{{- end }}`},
			},
			Values{
				Messages: []api.Message{
					{Role: "user", Content: "Hello friend!"},
					{Role: "assistant", Content: "Hello human!"},
					{Role: "user", Content: "What is your name?"},
				},
			},
			`[INST] Hello friend![/INST] Hello human![INST] What is your name?[/INST] `,
		},
		{
			"mistral system",
			[]template{
				{"no response", `[INST] {{ if .System }}{{ .System }}{{ "\n\n" }}{{ end }}{{ .Prompt }}[/INST] `},
				{"response", `[INST] {{ if .System }}{{ .System }}{{ "\n\n" }}{{ end }}{{ .Prompt }}[/INST] {{ .Response }}`},
				{"messages", `
{{- range $index, $_ := .Messages }}
{{- if eq .Role "user" }}[INST] {{ if and (eq (len (slice $.Messages $index)) 1) $.System }}{{ $.System }}{{ "\n\n" }}
{{- end }}{{ .Content }}[/INST] {{ else if eq .Role "assistant" }}{{ .Content }}
{{- end }}
{{- end }}`},
			},
			Values{
				Messages: []api.Message{
					{Role: "system", Content: "You are a helpful assistant!"},
					{Role: "user", Content: "Hello friend!"},
					{Role: "assistant", Content: "Hello human!"},
					{Role: "user", Content: "What is your name?"},
				},
			},
			`[INST] Hello friend![/INST] Hello human![INST] You are a helpful assistant!

What is your name?[/INST] `,
		},
		{
			"chatml",
			[]template{
				// this does not have a "no response" test because it's impossible to render the same output
				{"response", `{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
{{ .Response }}<|im_end|>
`},
				{"messages", `
{{- range $index, $_ := .Messages }}
{{- if and (eq .Role "user") (eq (len (slice $.Messages $index)) 1) $.System }}<|im_start|>system
{{ $.System }}<|im_end|>{{ "\n" }}
{{- end }}<|im_start|>{{ .Role }}
{{ .Content }}<|im_end|>{{ "\n" }}
{{- end }}<|im_start|>assistant
`},
			},
			Values{
				Messages: []api.Message{
					{Role: "system", Content: "You are a helpful assistant!"},
					{Role: "user", Content: "Hello friend!"},
					{Role: "assistant", Content: "Hello human!"},
					{Role: "user", Content: "What is your name?"},
				},
			},
			`<|im_start|>user
Hello friend!<|im_end|>
<|im_start|>assistant
Hello human!<|im_end|>
<|im_start|>system
You are a helpful assistant!<|im_end|>
<|im_start|>user
What is your name?<|im_end|>
<|im_start|>assistant
`,
		},
		{
			"moondream",
			[]template{
				// this does not have a "no response" test because it's impossible to render the same output
				{"response", `{{ if .Prompt }}Question: {{ .Prompt }}

{{ end }}Answer: {{ .Response }}

`},
				{"messages", `
{{- range .Messages }}
{{- if eq .Role "user" }}Question: {{ .Content }}{{ "\n\n" }}
{{- else if eq .Role "assistant" }}Answer: {{ .Content }}{{ "\n\n" }}
{{- end }}
{{- end }}Answer: `},
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
		t.Run(tt.name, func(t *testing.T) {
			for _, ttt := range tt.templates {
				t.Run(ttt.name, func(t *testing.T) {
					tmpl, err := Parse(ttt.template)
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
