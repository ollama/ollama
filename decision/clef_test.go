package decision

import (
	"encoding/json"
	"math"
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llm"
)

func TestCompileClef(t *testing.T) {
	var req Request
	if err := json.Unmarshal([]byte(`{"model":"renamed-model","state":{"z":2,"a":"café & <x>"},"questions":{"team":{"type":"choice","instructions":"Route it","criteria":{"z":null,"a":"First"}},"yes":{"type":"noul","instructions":{"z":2,"a":1}},"rating":{"type":"score","instructions":"Rate it","criteria":["Low","High"]}}}`), &req); err != nil {
		t.Fatal(err)
	}
	c, err := Compile(req, "clef")
	if err != nil {
		t.Fatal(err)
	}
	if err := c.Render(func([]api.Message) (string, error) { t.Fatal("Clef must not use a chat template"); return "", nil }); err != nil {
		t.Fatal(err)
	}
	input := c.Request
	if len(input.Fields) != 3 || len(c.Request.Rows) != 0 {
		t.Fatal("expected one joint request")
	}
	if input.Segments[1] != `{"a":"café & <x>","z":2}` {
		t.Fatalf("state = %s", input.Segments[1])
	}
	option := func(f, i int) string {
		return strings.Join(input.Segments[input.Fields[f].Options[i][0]:input.Fields[f].Options[i][1]], "")
	}
	if option(0, 0) != `{"description":"First","option_id":"a"}` || option(0, 1) != `{"option_id":"z"}` {
		t.Fatalf("choice ordering / null semantics changed: %s, %s", option(0, 0), option(0, 1))
	}
	if option(1, 0) != `{"description":"The proposition is true or the answer is yes.","option_id":"true"}` {
		t.Fatal(option(1, 0))
	}
	if input.Fields[0].Type != 1 || input.Fields[1].Type != 0 || input.Fields[2].Type != 2 {
		t.Fatal("incorrect type ids")
	}
	prompt := strings.Join(input.Segments, "")
	if !strings.HasSuffix(prompt, "<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:") || !strings.Contains(prompt, "INSTRUCTION: {\"a\":1,\"z\":2}\nALLOWED OPTIONS:\n") {
		t.Fatal(prompt)
	}
	response, err := c.Answer(req.Model, llm.ScoreResponse{Logits: [][]float32{{2, 0}, {2, 0}, {0, 2}}, InputTokens: 99})
	if err != nil {
		t.Fatal(err)
	}
	a, _ := response.Answers.Get("team")
	choice := a.(ChoiceAnswer)
	n, _ := response.Answers.Get("yes")
	yes := n.(NoulAnswer)
	if choice.Choice != "a" || math.Abs(choice.Confidence-0.4729346589968384) > 1e-8 || yes.Noul < 0.88 || response.Usage.OutputTokens != 0 {
		t.Fatalf("incorrect joint answer: %+v %+v", choice, yes)
	}
}
