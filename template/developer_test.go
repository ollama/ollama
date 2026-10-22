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

func TestTemplatesDeveloperRoleFallback(t *testing.T) {
	names, err := fs.Glob(templatesFS, "*.gotmpl")
	if err != nil {
		t.Fatal(err)
	}
	cases := map[string][]api.Message{
		"leading": {{Role: "developer", Content: "DEVELOPER_POLICY"}, {Role: "user", Content: "Hello"}},
		"mixed history": {
			{Role: "system", Content: "SYSTEM_POLICY"},
			{Role: "developer", Content: "DEVELOPER_POLICY"},
			{Role: "user", Content: "Hello"},
			{Role: "assistant", Content: "Earlier answer"},
			{Role: "developer", Content: "LATER_POLICY"},
			{Role: "user", Content: "Continue"},
		},
	}
	sources := map[string]string{"legacy": `{{ .System }}|{{ .Prompt }}|{{ .Response }}`}
	for _, name := range names {
		source, err := templatesFS.ReadFile(name)
		if err != nil {
			t.Fatal(err)
		}
		sources[name] = string(source)
	}
	for name, source := range sources {
		t.Run(name, func(t *testing.T) {
			tmpl, err := Parse(source)
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
					position := 0
					for _, message := range original {
						if message.Role != "system" && message.Role != "developer" {
							continue
						}
						index := strings.Index(got.String()[position:], message.Content)
						if index < 0 || strings.Count(got.String(), message.Content) != 1 {
							t.Fatalf("instruction %q was lost, duplicated, or reordered: %q", message.Content, got.String())
						}
						position += index + len(message.Content)
					}
					if !reflect.DeepEqual(messages, original) {
						t.Fatal("template fallback mutated the source history")
					}
				})
			}
		})
	}
}

func TestSavedTemplateDeveloperFallbackWithCRLF(t *testing.T) {
	source, err := templatesFS.ReadFile("chatml.gotmpl")
	if err != nil {
		t.Fatal(err)
	}
	tmpl, err := Parse(" \r\n" + strings.ReplaceAll(string(source), "\n", "\r\n") + "\r\n ")
	if err != nil {
		t.Fatal(err)
	}
	var out bytes.Buffer
	if err := tmpl.Execute(&out, Values{Messages: []api.Message{{Role: "developer", Content: "POLICY"}, {Role: "user", Content: "Hello"}}}); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out.String(), "<|im_start|>system\r\nPOLICY<|im_end|>") {
		t.Fatalf("saved template did not recognize developer fallback: %q", out.String())
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

func TestCustomTemplateKeepsInstructionRolesSeparate(t *testing.T) {
	// A custom template owns its role semantics. Developer turns must not also
	// appear in System when the template renders them from Messages.
	tmpl, err := Parse(`{{ .System }}|{{ range .Messages }}{{ .Role }}:{{ .Content }};{{ end }}`)
	if err != nil {
		t.Fatal(err)
	}
	messages := []api.Message{
		{Role: "system", Content: "Base policy"},
		{Role: "developer", Content: "Project policy"},
		{Role: "developer", Content: "Supplemental policy"},
		{Role: "user", Content: "First request"},
		{Role: "assistant", Content: "First answer"},
		{Role: "developer", Content: "Updated policy"},
		{Role: "user", Content: "Continue"},
	}
	original := slices.Clone(messages)
	var out bytes.Buffer
	if err := tmpl.Execute(&out, Values{Messages: messages}); err != nil {
		t.Fatal(err)
	}
	if want := "Base policy|system:Base policy;developer:Project policy\n\nSupplemental policy;user:First request;assistant:First answer;developer:Updated policy;user:Continue;"; out.String() != want {
		t.Fatalf("got %q, want %q", out.String(), want)
	}
	if !reflect.DeepEqual(messages, original) {
		t.Fatal("template mutated the source history")
	}
}
