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
		{"{{ .Prompt }}", []string{"prompt"}},
		{"{{ .System }} {{ .Prompt }}", []string{"prompt", "system"}},
		{"{{ .System }} {{ .Prompt }} {{ .Response }}", []string{"prompt", "response", "system"}},
		{"{{ with .Tools }}{{ . }}{{ end }} {{ .System }} {{ .Prompt }}", []string{"prompt", "system", "tools"}},
		{"{{ range .Messages }}{{ .Role }} {{ .Content }}{{ end }}", []string{"content", "messages", "role"}},
		{"{{ range .Messages }}{{ if eq .Role \"system\" }}SYSTEM: {{ .Content }}{{ else if eq .Role \"user\" }}USER: {{ .Content }}{{ else if eq .Role \"assistant\" }}ASSISTANT: {{ .Content }}{{ end }}{{ end }}", []string{"content", "messages", "role"}},
		{"{{ .Prompt }} {{ .Suffix }}", []string{"prompt", "suffix"}},
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
