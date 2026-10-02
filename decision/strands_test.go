package decision

import (
	"encoding/json"
	"fmt"
	"math"
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llm"
)

func TestCompileStrands(t *testing.T) {
	var req Request
	err := json.Unmarshal([]byte(`{"model":"renamed","state":{"z":1e6,"a":["café & <x>",true,null]},"questions":{"team":{"type":"choice","instructions":"  Route it  ","criteria":{"z":" Last\n team ","a":""}},"yes":{"type":"noul","instructions":"Is it urgent?"},"rating":{"type":"score","instructions":["Rate it"],"criteria":["Low","High"]}}}`), &req)
	if err != nil {
		t.Fatal(err)
	}
	c, err := CompileWithEncoder(req, "strands")
	if err != nil {
		t.Fatal(err)
	}
	if err := c.Render(func([]api.Message) (string, error) { t.Fatal("unexpected chat template"); return "", nil }); err != nil {
		t.Fatal(err)
	}
	if len(c.Request.PointerRows) != 3 || len(c.Request.Rows) != 0 || len(c.Request.Fields) != 0 {
		t.Fatal("expected three independent pointer rows")
	}
	row := c.Request.PointerRows[0]
	if want := "<state>\n{\n  \"z\": 1000000.0,\n  \"a\": [\n    \"café & <x>\",\n    true,\n    null\n  ]\n}\n</state>\n"; row.Prefix != want {
		t.Fatalf("prefix = %q", row.Prefix)
	}
	want := "<question type=\"choice\">\nSelect exactly one option.\nRoute it\n<options>\n1. z — Last team\n2. a\n</options>\n</question>\n<answer>"
	if row.Prompt != want {
		t.Fatalf("prompt = %q", row.Prompt)
	}
	for i, want := range []string{"1. z — Last team", "2. a"} {
		span := row.Options[i]
		if got := row.Prompt[span[0]:span[1]]; got != want {
			t.Fatalf("option = %q", got)
		}
	}
	if c.Request.PointerRows[0].Type != 1 || c.Request.PointerRows[1].Type != 0 || c.Request.PointerRows[2].Type != 2 {
		t.Fatal("incorrect question types")
	}
	if !strings.Contains(c.Request.PointerRows[1].Prompt, "1. false — the statement does not hold for this state\n2. true — the statement holds for this state") {
		t.Fatal("wrong noul defaults")
	}
	c.OrdinalSmoothing = 0.1
	answer, err := c.Answer(req.Model, llm.ScoreResponse{Logits: [][]float32{{0, 2}, {0, 2}, {0, 2}}, InputTokens: 99})
	if err != nil {
		t.Fatal(err)
	}
	v, _ := answer.Answers.Get("team")
	choice := v.(ChoiceAnswer)
	if choice.Choice != "a" || math.Abs(choice.Confidence-0.7615941559557646) > 1e-8 {
		t.Fatalf("wrong choice: %+v", choice)
	}
	v, _ = answer.Answers.Get("rating")
	score := v.(ScoreAnswer)
	if math.Abs(score.Score-0.8807970779778823) > 1e-8 || math.Abs(score.Confidence-0.9575595798883334) > 1e-8 {
		t.Fatalf("wrong score: %+v", score)
	}
	if answer.Usage.InputTokens != 99 || answer.Usage.OutputTokens != 0 {
		t.Fatal(answer.Usage)
	}
}

func TestStrandsContent(t *testing.T) {
	for _, tt := range []struct{ input, want string }{
		{`"  text\n"`, "text"}, {`[]`, "[]"}, {`{}`, "{}"},
		{`{"z":{},"a":[1.0,-0.0,1e-5]}`, "{\n  \"z\": {},\n  \"a\": [\n    1.0,\n    -0.0,\n    1e-05\n  ]\n}"},
	} {
		got, err := strandsContent(json.RawMessage(tt.input))
		if err != nil || got != tt.want {
			t.Fatalf("%s: got %q, %v; want %q", tt.input, got, err, tt.want)
		}
	}
}

func TestStrandsValidation(t *testing.T) {
	for _, q := range []string{
		`{"type":"noul"}`, `{"type":"noul","instructions":null}`, `{"type":"unknown","instructions":"x"}`,
		`{"type":"noul","instructions":"x","criteria":{"true":null}}`,
		`{"type":"noul","instructions":"x","criteria":{"maybe":"yes"}}`,
		`{"type":"choice","instructions":"x","criteria":{"a":"only"}}`,
		`{"type":"choice","instructions":"x","criteria":{"a":null,"b":"two"}}`,
		`{"type":"score","instructions":"x","criteria":["one",null]}`,
		`{"type":"score","instructions":"x","criteria":["0","1","2","3","4","5","6","7","8","9","10"]}`,
	} {
		var req Request
		if err := json.Unmarshal([]byte(`{"model":"s","state":"x","questions":{"q":`+q+`}}`), &req); err != nil {
			t.Fatal(err)
		}
		if _, err := CompileWithEncoder(req, "strands"); err == nil {
			t.Fatalf("accepted %s", q)
		}
	}
	for _, n := range []int{26, 27, 255, 256} {
		criteria := make([]string, n)
		for i := range criteria {
			criteria[i] = fmt.Sprintf(`"%d":""`, i)
		}
		q := Question{Type: "choice", Instructions: json.RawMessage(`"choose"`), Criteria: json.RawMessage("{" + strings.Join(criteria, ",") + "}")}
		_, err := strandsField("q", q)
		if (err == nil) != (n <= 255) {
			t.Fatalf("%d options: %v", n, err)
		}
	}
	req := testRequest(t)
	req.Images = []api.ImageData{[]byte("image")}
	if _, err := CompileWithEncoder(req, "strands"); err == nil {
		t.Fatal("accepted images")
	}
}

func TestStrandsConfidence(t *testing.T) {
	for _, tt := range []struct {
		p               []float64
		typ             string
		smoothing, want float64
	}{
		{[]float64{0.5, 0.5}, "choice", 0, 0}, {[]float64{0, 1, 0}, "choice", 0, 1},
		{[]float64{0.5, 0, 0.5}, "score", 0.1, 0},
		{[]float64{0.05, 0.9, 0.05}, "score", 0.1, 1},
		{[]float64{0.5, 0.5}, "score", 1, 1},
	} {
		if got := strandsConfidence(tt.p, tt.typ, tt.smoothing); math.Abs(got-tt.want) > 1e-8 {
			t.Fatalf("%+v: got %v", tt, got)
		}
	}
}
