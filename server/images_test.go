package server

import (
	"testing"
)

func TestModelPrompt(t *testing.T) {
	var m Model
	s, err := m.Prompt(PromptVars{
		First:  true,
		Prompt: "<h1>",
	}, "a{{ .Prompt }}b")
	if err != nil {
		t.Fatal(err)
	}
	want := "a<h1>b"
	if s != want {
		t.Errorf("got %q, want %q", s, want)
	}
}
