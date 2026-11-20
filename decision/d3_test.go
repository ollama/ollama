package decision

import (
	"encoding/json"
	"slices"
	"testing"

	"github.com/ollama/ollama/llm"
)

func TestD3CompilePreservesQuestionAndOptionOrder(t *testing.T) {
	var req Request
	if err := json.Unmarshal([]byte(`{"model":"d3","state":"","questions":{"team":{"type":"choice","instructions":"Route it.","criteria":{"second":"Later","first":"Earlier"}},"approved":{"type":"noul"},"rating":{"type":"score","criteria":["Low","Medium","High"]}}}`), &req); err != nil {
		t.Fatal(err)
	}
	compiled, err := Compile(req, "d3")
	if err != nil {
		t.Fatal(err)
	}
	if compiled.NeedsChatRender() || compiled.Request.State != "(empty)" {
		t.Fatalf("D3 compile mode or empty state changed: %+v", compiled.Request)
	}
	if got := compiled.Request.Rows[0].Question.Options; !slices.Equal(got, []string{"second: Later", "first: Earlier"}) {
		t.Fatalf("choice options = %v", got)
	}
	if got := compiled.Request.Rows[1].Question.Options; !slices.Equal(got, []string{"No / false", "Yes / true"}) {
		t.Fatalf("noul options = %v", got)
	}
	if got := compiled.Request.Rows[2].Question.Options; !slices.Equal(got, []string{"Low", "Medium", "High"}) {
		t.Fatalf("score levels = %v", got)
	}
	response, err := compiled.Answer("d3", llm.ScoreResponse{Logits: [][]float32{{2, 1}, {0, 1}, {0, 1, 2}}})
	if err != nil {
		t.Fatal(err)
	}
	team, ok := response.Answers.Get("team")
	if !ok || team.(ChoiceAnswer).Choice != "second" {
		t.Fatalf("choice answer = %+v", team)
	}
}
