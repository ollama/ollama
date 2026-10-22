package template

import (
	"bytes"
	"io/fs"
	"reflect"
	"slices"
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
)

func TestBundledTemplatesDeveloperRoleFallback(t *testing.T) {
	names, err := fs.Glob(templatesFS, "*.gotmpl")
	if err != nil {
		t.Fatal(err)
	}
	cases := map[string][]api.Message{
		"leading": {{Role: "developer", Content: "DEVELOPER_POLICY"}, {Role: "user", Content: "Hello"}},
		"after system": {
			{Role: "system", Content: "SYSTEM_POLICY"},
			{Role: "developer", Content: "DEVELOPER_POLICY"},
			{Role: "user", Content: "Hello"},
		},
		"after conversation": {
			{Role: "user", Content: "Hello"},
			{Role: "assistant", Content: "Earlier answer"},
			{Role: "developer", Content: "DEVELOPER_POLICY"},
			{Role: "user", Content: "Continue"},
		},
	}
	for _, name := range names {
		t.Run(name, func(t *testing.T) {
			source, err := templatesFS.ReadFile(name)
			if err != nil {
				t.Fatal(err)
			}
			for _, lineEnding := range []string{"LF", "CRLF"} {
				t.Run(lineEnding, func(t *testing.T) {
					text := string(source)
					if lineEnding == "CRLF" {
						text = strings.ReplaceAll(text, "\n", "\r\n")
					}
					tmpl, err := Parse(text)
					if err != nil {
						t.Fatal(err)
					}
					for label, messages := range cases {
						t.Run(label, func(t *testing.T) {
							original := slices.Clone(messages)
							systemMessages := slices.Clone(messages)
							for i := range systemMessages {
								if systemMessages[i].Role == "developer" {
									systemMessages[i].Role = "system"
								}
							}
							var got, want bytes.Buffer
							if err := tmpl.Execute(&got, Values{Messages: messages}); err != nil {
								t.Fatal(err)
							}
							if err := tmpl.Execute(&want, Values{Messages: systemMessages}); err != nil {
								t.Fatal(err)
							}
							if got.String() != want.String() {
								t.Fatalf("developer prompt = %q, system prompt = %q", got.String(), want.String())
							}
							if label == "leading" && !strings.Contains(got.String(), "DEVELOPER_POLICY") {
								t.Fatal("leading developer policy was lost")
							}
							if !reflect.DeepEqual(messages, original) {
								t.Fatal("template fallback mutated the source history")
							}
						})
					}
				})
			}
		})
	}
}

func TestDeveloperInstructionsInLegacyTemplate(t *testing.T) {
	tmpl, err := Parse(`{{ .System }}|{{ .Prompt }}|{{ .Response }}`)
	if err != nil {
		t.Fatal(err)
	}
	var out bytes.Buffer
	err = tmpl.Execute(&out, Values{Messages: []api.Message{
		{Role: "system", Content: "SYSTEM_POLICY"},
		{Role: "developer", Content: "DEVELOPER_POLICY"},
		{Role: "user", Content: "First request"},
		{Role: "assistant", Content: "First answer"},
		{Role: "developer", Content: "LATER_POLICY"},
		{Role: "user", Content: "Continue"},
	}})
	if err != nil {
		t.Fatal(err)
	}
	for _, marker := range []string{"SYSTEM_POLICY\n\nDEVELOPER_POLICY", "LATER_POLICY", "Continue"} {
		if !strings.Contains(out.String(), marker) {
			t.Errorf("missing %q: %s", marker, out.String())
		}
	}
}

func TestLegacyTemplateInstructionCompatibility(t *testing.T) {
	for _, role := range []string{"system", "developer"} {
		for _, toolLoop := range []bool{false, true} {
			name := role + "/adjacent"
			messages := []api.Message{{Role: "system", Content: "FIRST"}}
			want := "FIRST\n\nSECOND|REQUEST|"
			if toolLoop {
				name = role + "/tool_loop"
				want = "SECOND|REQUEST|"
				messages = append(messages,
					api.Message{Role: "assistant", ToolCalls: []api.ToolCall{{ID: "call_1", Function: api.ToolCallFunction{
						Name: "check", Arguments: api.NewToolCallFunctionArguments(),
					}}}},
					api.Message{Role: "tool", ToolCallID: "call_1", Content: "OK"},
				)
			}
			messages = append(messages, api.Message{Role: role, Content: "SECOND"}, api.Message{Role: "user", Content: "REQUEST"})
			t.Run(name, func(t *testing.T) {
				tmpl, err := Parse(`{{ .System }}|{{ .Prompt }}|{{ .Response }}`)
				if err != nil {
					t.Fatal(err)
				}
				original := slices.Clone(messages)
				var out bytes.Buffer
				if err := tmpl.Execute(&out, Values{Messages: messages}); err != nil {
					t.Fatal(err)
				}
				if out.String() != want {
					t.Fatalf("got %q, want the existing system behavior %q", out.String(), want)
				}
				if role == "developer" && !reflect.DeepEqual(messages, original) {
					t.Fatal("legacy fallback mutated the source history")
				}
			})
		}
	}
}

func TestMessagesTemplatePreservesDeveloperRole(t *testing.T) {
	tmpl, err := Parse(`{{ range .Messages }}{{ .Role }}:{{ .Content }};{{ end }}`)
	if err != nil {
		t.Fatal(err)
	}
	var out bytes.Buffer
	err = tmpl.Execute(&out, Values{Messages: []api.Message{
		{Role: "system", Content: "System policy"}, {Role: "developer", Content: "Developer policy"}, {Role: "user", Content: "Hello"},
	}})
	if err != nil {
		t.Fatal(err)
	}
	if want := "system:System policy;developer:Developer policy;user:Hello;"; out.String() != want {
		t.Fatalf("got %q, want %q", out.String(), want)
	}
}

func TestCustomTemplateKeepsInstructionRolesSeparate(t *testing.T) {
	// A custom template owns its role semantics. Developer turns must not also
	// appear in System when the template renders them from Messages.
	tmpl, err := Parse(`{{ .System }}|{{ range .Messages }}{{ if eq .Role "developer" }}developer:{{ .Content }};{{ else if eq .Role "user" }}user:{{ .Content }};{{ end }}{{ end }}`)
	if err != nil {
		t.Fatal(err)
	}
	messages := []api.Message{
		{Role: "system", Content: "Base policy"},
		{Role: "developer", Content: "Project policy"},
		{Role: "user", Content: "First request"},
		{Role: "assistant", Content: "First answer"},
		{Role: "developer", Content: "Updated policy"},
		{Role: "user", Content: "Continue"},
	}
	var out bytes.Buffer
	if err := tmpl.Execute(&out, Values{Messages: messages}); err != nil {
		t.Fatal(err)
	}
	if want := "Base policy|developer:Project policy;user:First request;developer:Updated policy;user:Continue;"; out.String() != want {
		t.Fatalf("got %q, want %q", out.String(), want)
	}
	if messages[1].Role != "developer" || messages[4].Role != "developer" {
		t.Fatal("template changed the source instruction roles")
	}
}
