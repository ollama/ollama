package server

import (
	"testing"

	"github.com/jmorganca/ollama/api"
)

func TestModelPrompt(t *testing.T) {
	var m Model
	req := api.GenerateRequest{
		Template: "a{{ .Prompt }}b",
		Prompt:   "<h1>",
	}
	s, err := m.Prompt(req)
	if err != nil {
		t.Fatal(err)
	}
	want := "a<h1>b"
	if s != want {
		t.Errorf("got %q, want %q", s, want)
	}
}
