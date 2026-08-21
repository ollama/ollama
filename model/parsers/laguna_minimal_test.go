package parsers

import (
	"testing"

	"github.com/ollama/ollama/api"
)

func TestLagunaMinimalJSONContent(t *testing.T) {
	tools := []api.Tool{{Function: api.ToolFunction{Name: "web_search"}}}
	p := &LagunaV8Parser{}
	p.Init(tools, nil, nil)

	chunks := []string{"The env field:", ` {"`, "TOKEN", `": "`, "x", `"}`, " done"}
	var content string
	var calls []api.ToolCall
	for i, tok := range chunks {
		c, _, cl, err := p.Add(tok, i == len(chunks)-1)
		content += c
		calls = append(calls, cl...)
		if err != nil {
			t.Fatalf("error at chunk %d (%q): %v", i+1, tok, err)
		}
	}
	if len(calls) > 0 {
		t.Errorf("content turned into %d tool call(s): %+v", len(calls), calls)
	}
	want := `The env field: {"TOKEN": "x"} done`
	if content != want {
		t.Errorf("content altered\n got: %q\nwant: %q", content, want)
	}
}
