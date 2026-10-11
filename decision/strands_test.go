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
	err := json.Unmarshal([]byte(`{"model":"renamed","state":{"z":1e6,"a":["café & <x>",true,null]},"questions":{"team":{"type":"choice","instructions":"  Route it  ","criteria":{"z":" Last\n team ","a":null}},"yes":{"type":"noul","instructions":"Is it urgent?"},"rating":{"type":"score","instructions":["Rate it"],"criteria":["Low","High"]}}}`), &req)
	if err != nil {
		t.Fatal(err)
	}
	c, err := Compile(req, "strands")
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
	answer, err := c.Answer(req.Model, llm.ScoreResponse{Logits: [][]float32{{0, 2}, {0, 2}, {0, 2}}, InputTokens: 99})
	if err != nil {
		t.Fatal(err)
	}
	v, _ := answer.Answers.Get("team")
	choice := v.(ChoiceAnswer)
	if choice.Choice != "a" || math.Abs(choice.Confidence-0.4729346589968384) > 1e-8 {
		t.Fatalf("wrong choice: %+v", choice)
	}
	v, _ = answer.Answers.Get("rating")
	score := v.(ScoreAnswer)
	if math.Abs(score.Score-0.8807970779778823) > 1e-8 || math.Abs(score.Confidence-choice.Confidence) > 1e-8 {
		t.Fatalf("wrong score: %+v", score)
	}
	if answer.Usage.InputTokens != 99 || answer.Usage.OutputTokens != 0 {
		t.Fatal(answer.Usage)
	}
}

func TestStrandsChoiceWithoutDescriptions(t *testing.T) {
	for _, description := range []string{`null`, `""`} {
		t.Run(description, func(t *testing.T) {
			var req Request
			input := fmt.Sprintf(`{"model":"strands-decider:2b-mlx-bf16","state":"Our checkout has returned 500 errors since 9am.","questions":{"label":{"type":"choice","instructions":"Which label fits this ticket?","criteria":{"billing":%s,"bug":%s,"account":%s}}}}`, description, description, description)
			if err := json.Unmarshal([]byte(input), &req); err != nil {
				t.Fatal(err)
			}
			c, err := Compile(req, "strands")
			if err != nil {
				t.Fatal(err)
			}
			row := c.Request.PointerRows[0]
			want := "<question type=\"choice\">\nSelect exactly one option.\nWhich label fits this ticket?\n<options>\n1. billing\n2. bug\n3. account\n</options>\n</question>\n<answer>"
			if row.Prompt != want {
				t.Fatalf("prompt = %q, want %q", row.Prompt, want)
			}
			if len(row.Options) != 3 {
				t.Fatalf("option count = %d, want 3", len(row.Options))
			}
			for i, want := range []string{"1. billing", "2. bug", "3. account"} {
				span := row.Options[i]
				if got := row.Prompt[span[0]:span[1]]; got != want {
					t.Fatalf("option %d = %q, want %q", i, got, want)
				}
			}
		})
	}
}

func TestStrandsContent(t *testing.T) {
	depth := maxStrandsContentDepth + 1
	if _, err := strandsContent(json.RawMessage(strings.Repeat("[", depth) + "0" + strings.Repeat("]", depth))); err == nil {
		t.Fatal("accepted excessive content nesting")
	}
	for _, tt := range []struct{ input, want string }{
		{`"  text\n"`, "text"},
		{`[]`, "[]"},
		{`{}`, "{}"},
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
		`{"type":"noul","instructions":""}`,
		`{"type":"noul","instructions":" \t\n "}`,
		`{"type":"choice","instructions":" \t\n ","criteria":{"a":"one","b":"two"}}`,
		`{"type":"score","instructions":" \t\n ","criteria":["one","two"]}`,
		`{"type":"noul","instructions":"x","criteria":{"true":null}}`,
		`{"type":"noul","instructions":"x","criteria":{"maybe":"yes"}}`,
		`{"type":"choice","instructions":"x","criteria":{"a":"only"}}`,
		`{"type":"choice","instructions":"x","criteria":{"a":42,"b":"two"}}`,
		`{"type":"choice","instructions":"x","criteria":{"":"one","valid":"two"}}`,
		`{"type":"choice","instructions":"x","criteria":{" \t\n ":"one","valid":"two"}}`,
		`{"type":"score","instructions":"x","criteria":["one",null]}`,
		`{"type":"score","instructions":"x","criteria":["0","1","2","3","4","5","6","7","8","9","10"]}`,
	} {
		var req Request
		if err := json.Unmarshal([]byte(`{"model":"s","state":"x","questions":{"q":`+q+`}}`), &req); err != nil {
			t.Fatal(err)
		}
		if _, err := Compile(req, "strands"); err == nil {
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
	if _, err := Compile(req, "strands"); err == nil {
		t.Fatal("accepted images")
	}
}

func FuzzStrandsContent(f *testing.F) {
	for _, seed := range []string{`"text"`, `{"z":1e6,"a":["café",true,null]}`, `[]`, `{"x":[[[0]]]}`, `null`, `{"x":`} {
		f.Add(seed)
	}
	f.Fuzz(func(t *testing.T, input string) {
		if len(input) > 64<<10 {
			t.Skip()
		}
		output, err := strandsContent(json.RawMessage(input))
		if err != nil {
			return
		}
		raw := strings.TrimSpace(input)
		if (strings.HasPrefix(raw, "{") || strings.HasPrefix(raw, "[")) && !json.Valid([]byte(output)) {
			t.Fatalf("rendered invalid JSON: %q", output)
		}
	})
}
