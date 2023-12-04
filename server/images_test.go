package server

import (
	"testing"
)

func TestModelPrompt(t *testing.T) {
	m := Model{
		Template: "a{{ .Prompt }}b",
	}
	s, err := m.Prompt(PromptVars{
		Prompt: "<h1>",
	})
	if err != nil {
		t.Fatal(err)
	}
	want := "a<h1>b"
	if s != want {
		t.Errorf("got %q, want %q", s, want)
	}
}
