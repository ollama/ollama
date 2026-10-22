package template

import (
	"bytes"
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
)

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

func TestSystemSlotPreservesDeveloperInstructionsWithMessages(t *testing.T) {
	// Older Llama and Qwen templates use System for instructions and Messages
	// for user/assistant/tool turns. Developer content must reach that slot
	// even though the message history retains its original roles.
	tmpl, err := Parse(`{{ .System }}|{{ range .Messages }}{{ if eq .Role "user" }}{{ .Content }};{{ end }}{{ end }}`)
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
	if want := "Base policy\n\nProject policy\n\nUpdated policy|First request;Continue;"; out.String() != want {
		t.Fatalf("got %q, want %q", out.String(), want)
	}
	if messages[1].Role != "developer" || messages[4].Role != "developer" {
		t.Fatal("template changed the source instruction roles")
	}
}
