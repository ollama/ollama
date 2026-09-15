package template_test

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/openai"
	prompttemplate "github.com/ollama/ollama/template"
)

func TestDeveloperInstructions(t *testing.T) {
	fixtures := []struct{ name, text string }{
		{"legacy", `{{if .System}}[system]{{.System}}[/system]{{end}}[user]{{.Prompt}}[/user][assistant]{{.Response}}`},
		{"system_and_messages", `{{if .System}}[system]{{.System}}[/system]{{end}}{{range .Messages}}{{if or (eq .Role "user") (eq .Role "assistant") (eq .Role "tool")}}[{{.Role}}]{{.Content}}[/{{.Role}}]{{end}}{{end}}[assistant]`},
		{"messages_only", `{{range .Messages}}{{if or (eq .Role "system") (eq .Role "user") (eq .Role "assistant") (eq .Role "tool")}}[{{.Role}}]{{.Content}}[/{{.Role}}]{{end}}{{end}}[assistant]`},
	}
	cases := []struct {
		name     string
		messages []api.Message
	}{
		{"leading", []api.Message{
			{Role: "developer", Content: "instruction-alpha"},
			{Role: "user", Content: "question-one"},
		}},
		{"consecutive", []api.Message{
			{Role: "developer", Content: "instruction-alpha"},
			{Role: "developer", Content: "instruction-beta"},
			{Role: "user", Content: "question-one"},
		}},
		{"mixed_instructions", []api.Message{
			{Role: "system", Content: "instruction-system-one"},
			{Role: "developer", Content: "instruction-alpha"},
			{Role: "system", Content: "instruction-system-two"},
			{Role: "developer", Content: "instruction-beta"},
			{Role: "user", Content: "question-one"},
		}},
		{"later_turn", []api.Message{
			{Role: "user", Content: "question-one"},
			{Role: "assistant", Content: "answer-one"},
			{Role: "developer", Content: "instruction-later"},
			{Role: "user", Content: "question-two"},
		}},
		{"unicode_and_newlines", []api.Message{
			{Role: "developer", Content: "تعليمات خاصة\nKeep café and 日本語 intact."},
			{Role: "user", Content: "question-one"},
		}},
	}
	for _, fixture := range fixtures {
		for _, tc := range cases {
			for _, viaResponses := range []bool{false, true} {
				path := "direct"
				if viaResponses {
					path = "responses"
				}
				t.Run(fixture.name+"/"+tc.name+"/"+path, func(t *testing.T) {
					render := func(messages []api.Message) string {
						t.Helper()
						// Execute may collate its input in place; isolate each render.
						messages = append([]api.Message(nil), messages...)
						if viaResponses {
							input := make([]map[string]string, len(messages))
							for i, m := range messages {
								input[i] = map[string]string{"type": "message", "role": m.Role, "content": m.Content}
							}
							body, err := json.Marshal(map[string]any{"model": "fixture", "stream": false, "input": input})
							if err != nil {
								t.Fatal(err)
							}
							var request openai.ResponsesRequest
							if err := json.Unmarshal(body, &request); err != nil {
								t.Fatal(err)
							}
							chat, err := openai.FromResponsesRequest(request)
							if err != nil {
								t.Fatal(err)
							}
							messages = chat.Messages
						}
						tmpl, err := prompttemplate.Parse(fixture.text)
						if err != nil {
							t.Fatal(err)
						}
						var output bytes.Buffer
						if err := tmpl.Execute(&output, prompttemplate.Values{Messages: messages}); err != nil {
							t.Fatal(err)
						}
						return output.String()
					}
					// The compatibility contract is parity with the same conversation
					// expressed using the template's existing system role.
					reference := append([]api.Message(nil), tc.messages...)
					for i := range reference {
						if reference[i].Role == "developer" {
							reference[i].Role = "system"
						}
					}
					want := render(reference)
					got := render(tc.messages)
					for _, m := range tc.messages {
						if !strings.Contains(want, m.Content) {
							t.Fatalf("control prompt lost content %q: %q", m.Content, want)
						}
						if !strings.Contains(got, m.Content) {
							t.Errorf("prompt lost content %q: %q", m.Content, got)
						}
					}
					if got != want {
						t.Errorf("developer/system rendering differs\nwant: %q\n got: %q", want, got)
					}
				})
			}
		}
	}
}
