package template

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/fs/ggml"
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
				kv := ggml.KV{"tokenizer.chat_template": v}
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

				tmpl, err := Parse(b.String())
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

func mustToolFunction(t *testing.T, input string) api.Tool {
	fun := api.ToolFunction{}
	if err := json.Unmarshal([]byte(input), &fun); err != nil {
		t.Fatal(err)
	}
	return api.Tool{
		Type:     "function",
		Function: fun,
	}
}

func TestTools(t *testing.T) {
	get_weather := mustToolFunction(t, `
{
  "name": "get_weather",
  "description": "Get current weather information for a location",
  "parameters": {
    "type": "object",
    "properties": {
      "location": {
        "type": "string",
        "description": "The city and state, e.g. San Francisco, CA"
      },
      "unit": {
        "type": "string",
        "enum": [
          "celsius",
          "fahrenheit"
        ],
        "description": "The unit of temperature to use"
      }
    },
    "required": [
      "location"
    ]
  }
}
`)
	cases := map[string]struct {
		tools    api.Tools
		messages []api.Message
	}{
		"single": {
			tools:    []api.Tool{get_weather},
			messages: []api.Message{},
		},
		"single-call": {
			messages: []api.Message{
				{Role: "assistant", ToolCalls: []api.ToolCall{
					{
						Function: api.ToolCallFunction{
							Name: "get_weather",
							Arguments: api.ToolCallFunctionArguments{
								"location": "San Francisco, CA",
							},
						},
					},
				}},
			},
		},
		"single-result": {
			messages: []api.Message{
				{Role: "tool", Content: `{"temperature": 22, "condition": "Sunny", "humidity": 45, "wind_speed": 10}`},
			},
		},
	}

	templates, err := filepath.Glob("*.gotmpl")
	if err != nil {
		t.Fatal(err)
	}

	casenames := make([]string, len(cases))
	for n := range cases {
		casenames = append(casenames, n)
	}

	for _, tplname := range templates {
		haveTools := false
		for _, n := range casenames {
			if _, err := os.Stat(filepath.Join("testdata", tplname, fmt.Sprintf("tools-%v", n))); err == nil {
				haveTools = true
				break
			}
		}
		if !haveTools {
			continue
		}

		t.Run(tplname, func(t *testing.T) {
			bts, err := os.ReadFile(tplname)
			if err != nil {
				t.Fatal(err)
			}

			tmpl, err := Parse(string(bts))
			if err != nil {
				t.Fatal(err)
			}

			for n := range cases {
				tc := cases[n]

				t.Run(n, func(t *testing.T) {
					expect, err := os.ReadFile(filepath.Join("testdata", tplname, fmt.Sprintf("tools-%v", n)))
					if err != nil {
						if os.IsNotExist(err) {
							t.SkipNow()
						}
						t.Fatal(err)
					}

					var actual bytes.Buffer
					if err := tmpl.Execute(&actual, Values{Messages: tc.messages, Tools: tc.tools}); err != nil {
						t.Fatal(err)
					}

					bts := actual.Bytes()

					if len(expect) == 0 {
						if err := os.WriteFile(filepath.Join("testdata", tplname, fmt.Sprintf("tools-%v", n)), bts, 0666); err != nil {
							t.Fatal(err)
						}
						return
					}

					if diff := cmp.Diff(string(bts), string(expect)); diff != "" {
						t.Errorf("mismatch (-got +want):\n%s", diff)
					}
				})
			}
		})
	}
}

func TestTemplate(t *testing.T) {
	cases := make(map[string][]api.Message)
	for _, mm := range [][]api.Message{
		{
			{Role: "user", Content: "Hello, how are you?"},
		},
		{
			{Role: "user", Content: "Hello, how are you?"},
			{Role: "assistant", Content: "I'm doing great. How can I help you today?"},
			{Role: "user", Content: "I'd like to show off how chat templating works!"},
		},
		{
			{Role: "system", Content: "You are a helpful assistant."},
			{Role: "user", Content: "Hello, how are you?"},
			{Role: "assistant", Content: "I'm doing great. How can I help you today?"},
			{Role: "user", Content: "I'd like to show off how chat templating works!"},
		},
	} {
		var roles []string
		for _, m := range mm {
			roles = append(roles, m.Role)
		}

		cases[strings.Join(roles, "-")] = mm
	}

	matches, err := filepath.Glob("*.gotmpl")
	if err != nil {
		t.Fatal(err)
	}

	for _, match := range matches {
		t.Run(match, func(t *testing.T) {
			bts, err := os.ReadFile(match)
			if err != nil {
				t.Fatal(err)
			}

			tmpl, err := Parse(string(bts))
			if err != nil {
				t.Fatal(err)
			}

			for n, tt := range cases {
				var actual bytes.Buffer
				t.Run(n, func(t *testing.T) {
					if err := tmpl.Execute(&actual, Values{Messages: tt}); err != nil {
						t.Fatal(err)
					}

					expect, err := os.ReadFile(filepath.Join("testdata", match, n))
					if err != nil {
						t.Fatal(err)
					}

					bts := actual.Bytes()

					if slices.Contains([]string{"chatqa.gotmpl", "llama2-chat.gotmpl", "mistral-instruct.gotmpl", "openchat.gotmpl", "vicuna.gotmpl"}, match) && bts[len(bts)-1] == ' ' {
						t.Log("removing trailing space from output")
						bts = bts[:len(bts)-1]
					}

					if diff := cmp.Diff(string(bts), string(expect)); diff != "" {
						t.Errorf("mismatch (-got +want):\n%s", diff)
					}
				})

				t.Run("legacy", func(t *testing.T) {
					t.Skip("legacy outputs are currently default outputs")
					var legacy bytes.Buffer
					if err := tmpl.Execute(&legacy, Values{Messages: tt, forceLegacy: true}); err != nil {
						t.Fatal(err)
					}

					legacyBytes := legacy.Bytes()
					if slices.Contains([]string{"chatqa.gotmpl", "openchat.gotmpl", "vicuna.gotmpl"}, match) && legacyBytes[len(legacyBytes)-1] == ' ' {
						t.Log("removing trailing space from legacy output")
						legacyBytes = legacyBytes[:len(legacyBytes)-1]
					} else if slices.Contains([]string{"codellama-70b-instruct.gotmpl", "llama2-chat.gotmpl", "mistral-instruct.gotmpl"}, match) {
						t.Skip("legacy outputs cannot be compared to messages outputs")
					}

					if diff := cmp.Diff(legacyBytes, actual.Bytes()); diff != "" {
						t.Errorf("mismatch (-got +want):\n%s", diff)
					}
				})
			}
		})
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
		{`{{- range .Messages }}
{{- if eq .Role "system" }}SYSTEM:
{{- else if eq .Role "user" }}USER:
{{- else if eq .Role "assistant" }}ASSISTANT:
{{- end }} {{ .Content }}
{{- end }}`, []string{"content", "messages", "role"}},
		{`{{- if .Messages }}
{{- range .Messages }}<|im_start|>{{ .Role }}
{{ .Content }}<|im_end|>
{{ end }}<|im_start|>assistant
{{ else -}}
{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
{{ .Response }}<|im_end|>
{{- end -}}`, []string{"content", "messages", "prompt", "response", "role", "system"}},
	}

	for _, tt := range cases {
		t.Run("", func(t *testing.T) {
			tmpl, err := Parse(tt.template)
			if err != nil {
				t.Fatal(err)
			}

			if diff := cmp.Diff(tmpl.Vars(), tt.vars); diff != "" {
				t.Errorf("mismatch (-got +want):\n%s", diff)
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
				{"no response", `[INST] {{ if .System }}{{ .System }}

{{ end }}{{ .Prompt }}[/INST] `},
				{"response", `[INST] {{ if .System }}{{ .System }}

{{ end }}{{ .Prompt }}[/INST] {{ .Response }}`},
				{"messages", `[INST] {{ if .System }}{{ .System }}

{{ end }}
{{- range .Messages }}
{{- if eq .Role "user" }}{{ .Content }}[/INST] {{ else if eq .Role "assistant" }}{{ .Content }}[INST] {{ end }}
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
				{"no response", `[INST] {{ if .System }}{{ .System }}

{{ end }}{{ .Prompt }}[/INST] `},
				{"response", `[INST] {{ if .System }}{{ .System }}

{{ end }}{{ .Prompt }}[/INST] {{ .Response }}`},
				{"messages", `[INST] {{ if .System }}{{ .System }}

{{ end }}
{{- range .Messages }}
{{- if eq .Role "user" }}{{ .Content }}[/INST] {{ else if eq .Role "assistant" }}{{ .Content }}[INST] {{ end }}
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
			`[INST] You are a helpful assistant!

Hello friend![/INST] Hello human![INST] What is your name?[/INST] `,
		},
		{
			"mistral assistant",
			[]template{
				{"no response", `[INST] {{ .Prompt }}[/INST] `},
				{"response", `[INST] {{ .Prompt }}[/INST] {{ .Response }}`},
				{"messages", `
{{- range $i, $m := .Messages }}
{{- if eq .Role "user" }}[INST] {{ .Content }}[/INST] {{ else if eq .Role "assistant" }}{{ .Content }}{{ end }}
{{- end }}`},
			},
			Values{
				Messages: []api.Message{
					{Role: "user", Content: "Hello friend!"},
					{Role: "assistant", Content: "Hello human!"},
					{Role: "user", Content: "What is your name?"},
					{Role: "assistant", Content: "My name is Ollama and I"},
				},
			},
			`[INST] Hello friend![/INST] Hello human![INST] What is your name?[/INST] My name is Ollama and I`,
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
{{- range $index, $_ := .Messages }}<|im_start|>{{ .Role }}
{{ .Content }}<|im_end|>
{{ end }}<|im_start|>assistant
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
			`<|im_start|>system
You are a helpful assistant!<|im_end|>
<|im_start|>user
Hello friend!<|im_end|>
<|im_start|>assistant
Hello human!<|im_end|>
<|im_start|>user
What is your name?<|im_end|>
<|im_start|>assistant
`,
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

					if diff := cmp.Diff(b.String(), tt.expected); diff != "" {
						t.Errorf("mismatch (-got +want):\n%s", diff)
					}
				})
			}
		})
	}
}

func TestExecuteWithSuffix(t *testing.T) {
	tmpl, err := Parse(`{{- if .Suffix }}<PRE> {{ .Prompt }} <SUF>{{ .Suffix }} <MID>
{{- else }}{{ .Prompt }}
{{- end }}`)
	if err != nil {
		t.Fatal(err)
	}

	cases := []struct {
		name   string
		values Values
		expect string
	}{
		{
			"message", Values{Messages: []api.Message{{Role: "user", Content: "hello"}}}, "hello",
		},
		{
			"prompt suffix", Values{Prompt: "def add(", Suffix: "return x"}, "<PRE> def add( <SUF>return x <MID>",
		},
	}

	for _, tt := range cases {
		t.Run(tt.name, func(t *testing.T) {
			var b bytes.Buffer
			if err := tmpl.Execute(&b, tt.values); err != nil {
				t.Fatal(err)
			}

			if diff := cmp.Diff(b.String(), tt.expect); diff != "" {
				t.Errorf("mismatch (-got +want):\n%s", diff)
			}
		})
	}
}
