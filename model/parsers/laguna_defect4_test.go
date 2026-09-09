package parsers

import (
	"testing"

	"github.com/ollama/ollama/api"
)

// TestLagunaHeldCandidateEmitsNoCall feeds one Add whose buffer holds prose
// followed by the start of a JSON candidate. The held path emits the prose;
// it must not emit a tool call alongside it.
func TestLagunaHeldCandidateEmitsNoCall(t *testing.T) {
	tools := []api.Tool{{Function: api.ToolFunction{Name: "web_search"}}}
	p := &LagunaV8Parser{}
	p.Init(tools, nil, nil)

	content, _, calls, err := p.Add(`checking {"na`, false)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	t.Logf("content=%q calls=%+v", content, calls)
	if len(calls) > 0 {
		t.Errorf("held candidate produced %d call(s): %+v", len(calls), calls)
	}
}
