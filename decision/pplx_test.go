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

func TestCompilePPLX(t *testing.T) {
	var req Request
	err := json.Unmarshal([]byte(`{"model":"renamed-decider","state":{"z":2,"a":"café & <x>"},"questions":{"team":{"type":"choice","instructions":"Route it","criteria":{"z":null,"a":"First"}},"yes":{"type":"noul","instructions":{"z":2,"a":1},"criteria":{"false":""}},"rating":{"type":"score","instructions":"Rate it","criteria":["Low","High"]}}}`), &req)
	if err != nil {
		t.Fatal(err)
	}
	req.Images = []api.ImageData{{1, 2, 3}}
	c, err := CompileWithEncoder(req, "pplx")
	if err != nil {
		t.Fatal(err)
	}
	if err := c.Render(func([]api.Message) (string, error) { t.Fatal("PPLX uses its reference prompt"); return "", nil }); err != nil {
		t.Fatal(err)
	}
	if !c.Request.Readout || len(c.Request.Rows) != 3 || len(c.Request.Images) != 1 {
		t.Fatalf("unexpected request: %+v", c.Request)
	}
	want := pplxPrefix + "State:\n{\"z\": 2, \"a\": \"café & <x>\"}\n\nQuestion:\nRoute it\n\nOptions:\nA: z\nB: a: First\n\nReturn only the letter code of the best option.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
	if c.Request.Rows[0].Prompt != want {
		t.Fatalf("prompt = %q; want %q", c.Request.Rows[0].Prompt, want)
	}
	if !strings.Contains(c.Request.Rows[1].Prompt, "Question:\n{\"z\": 2, \"a\": 1}\n\nOptions:\nA: No / false\nB: Yes / true") {
		t.Fatal(c.Request.Rows[1].Prompt)
	}
	if !strings.Contains(c.Request.Rows[2].Prompt, "A: Low\nB: High") {
		t.Fatal(c.Request.Rows[2].Prompt)
	}
	result, err := c.Answer(req.Model, llm.ScoreResponse{Logits: [][]float32{{0, 2}, {0, 2}, {0, 2}}, InputTokens: 111})
	if err != nil {
		t.Fatal(err)
	}
	team, _ := result.Answers.Get("team")
	yes, _ := result.Answers.Get("yes")
	rating, _ := result.Answers.Get("rating")
	p := 1 / (1 + math.Exp(-2))
	if team.(ChoiceAnswer).Choice != "a" || math.Abs(team.(ChoiceAnswer).Confidence-(2*p-1)) > 1e-7 || math.Abs(yes.(NoulAnswer).Noul-p) > 1e-7 || math.Abs(rating.(ScoreAnswer).Confidence-(2*p-1)) > 1e-7 || result.Usage.OutputTokens != 0 {
		t.Fatalf("wrong answers: %+v %+v %+v", team, yes, rating)
	}
}

func TestPPLXOptionLimits(t *testing.T) {
	for _, count := range []int{1, 26, 27, 255, 256} {
		t.Run(fmt.Sprint(count), func(t *testing.T) {
			var criteria []string
			for i := 0; i < count; i++ {
				criteria = append(criteria, fmt.Sprintf("%q:null", fmt.Sprint(i)))
			}
			q := &Questions{}
			q.Set("pick", Question{Type: "choice", Instructions: json.RawMessage(`"Pick"`), Criteria: json.RawMessage("{" + strings.Join(criteria, ",") + "}")})
			c, err := CompileWithEncoder(Request{Model: "pplx", State: json.RawMessage(`"State"`), Questions: q}, "pplx")
			if count > 255 {
				if err == nil {
					t.Fatal("accepted 256 options")
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if len(c.Request.Rows[0].Candidates) != count || c.Request.Rows[0].Candidates[count-1] != pplxCodes[count-1] {
				t.Fatal("incorrect answer vocabulary")
			}
			if count == 1 {
				result, err := c.Answer("pplx", llm.ScoreResponse{Logits: [][]float32{{0}}})
				if err != nil {
					t.Fatal(err)
				}
				answer, _ := result.Answers.Get("pick")
				if answer.(ChoiceAnswer).Confidence != 1 {
					t.Fatal(answer)
				}
			}
		})
	}
}

func TestPPLXStructuredContent(t *testing.T) {
	for _, tt := range []struct{ input, want string }{
		{`{"z":["caf\u00e9", "a,b:c", "\"\\"], "a": "<tag>"}`, `{"z": ["café", "a,b:c", "\"\\"], "a": "<tag>"}`},
		{`{"z":1e6,"a":[-0,1e-5,1e16,-0.0,9007199254740993]}`, `{"z": 1000000.0, "a": [0, 1e-05, 1e+16, -0.0, 9007199254740993]}`},
		{`["\u2028\u2029","\\u2028",true,null]`, "[\"\u2028\u2029\", \"\\\\u2028\", true, null]"},
		{`"plain text"`, "plain text"},
		{`null`, "null"},
	} {
		got, err := pplxContent(json.RawMessage(tt.input))
		if err != nil || got != tt.want {
			t.Fatalf("content = %q, %v; want %q", got, err, tt.want)
		}
	}
	for _, raw := range []string{"", `{`, `null true`} {
		if _, err := pplxContent(json.RawMessage(raw)); err == nil {
			t.Fatalf("accepted invalid JSON %q", raw)
		}
	}
}

func TestPPLXSchema(t *testing.T) {
	var req Request
	if err := json.Unmarshal([]byte(`{"model":"pplx","state":null,"questions":{"team":{"type":"choice","criteria":{"z":null,"a":{"text":"First","weight":1e6}}},"yes":{"type":"noul","instructions":null,"criteria":{"true":null,"false":[]}},"rating":{"type":"score","instructions":{},"criteria":[null,{"z":1e6,"a":"High"}]}}}`), &req); err != nil {
		t.Fatal(err)
	}
	c, err := CompileWithEncoder(req, "pplx")
	if err != nil {
		t.Fatal(err)
	}
	for i, options := range []string{
		"A: z\nB: a: {\"text\": \"First\", \"weight\": 1000000.0}",
		"A: No / false\nB: Yes / true",
		"A: null\nB: {\"z\": 1000000.0, \"a\": \"High\"}",
	} {
		if !strings.Contains(c.Request.Rows[i].Prompt, "State:\nnull\n\nQuestion:\nChoose the best matching option.\n\nOptions:\n"+options) {
			t.Fatal(c.Request.Rows[i].Prompt)
		}
	}
	result, err := c.Answer(req.Model, llm.ScoreResponse{Logits: [][]float32{{0, 2}, {0, 2}, {0, 2}}})
	if err != nil {
		t.Fatal(err)
	}
	rating, _ := result.Answers.Get("rating")
	legend, err := json.Marshal(rating.(ScoreAnswer).Legend)
	if err != nil || string(legend) != `{"0":null,"1":{"z":1e6,"a":"High"}}` {
		t.Fatalf("legend = %s, %v", legend, err)
	}
	if _, err := Compile(req); err == nil {
		t.Fatal("broadened the candidate-scoring schema")
	}
}

func TestPPLXValidation(t *testing.T) {
	for _, question := range []string{
		`{"type":"other"}`, `{"type":"choice"}`, `{"type":"choice","criteria":null}`,
		`{"type":"choice","criteria":{}}`, `{"type":"choice","criteria":{"":null}}`,
		`{"type":"choice","criteria":[]}`, `{"type":"noul","criteria":[]}`,
		`{"type":"noul","criteria":{"yes":null}}`, `{"type":"score","criteria":{}}`,
		`{"type":"score","criteria":["only"]}`,
	} {
		var req Request
		if err := json.Unmarshal([]byte(`{"model":"pplx","state":"x","questions":{"q":`+question+`}}`), &req); err != nil {
			t.Fatal(err)
		}
		if _, err := CompileWithEncoder(req, "pplx"); err == nil {
			t.Fatalf("accepted invalid question %s", question)
		}
	}
}

func TestPPLXConfidence(t *testing.T) {
	var req Request
	if err := json.Unmarshal([]byte(`{"model":"pplx","state":"state","questions":{"pick":{"type":"choice","instructions":"choose","criteria":{"a":null,"b":null,"c":null}},"rating":{"type":"score","instructions":"rate","criteria":["low","medium","high"]}}}`), &req); err != nil {
		t.Fatal(err)
	}
	c, err := CompileWithEncoder(req, "pplx")
	if err != nil {
		t.Fatal(err)
	}
	logits := []float32{float32(math.Log(.1)), float32(math.Log(.2)), float32(math.Log(.7))}
	result, err := c.Answer("pplx", llm.ScoreResponse{Logits: [][]float32{logits, logits}})
	if err != nil {
		t.Fatal(err)
	}
	choice, _ := result.Answers.Get("pick")
	score, _ := result.Answers.Get("rating")
	if math.Abs(choice.(ChoiceAnswer).Confidence-.55) > 1e-7 || math.Abs(score.(ScoreAnswer).Confidence-.4) > 1e-7 || math.Abs(score.(ScoreAnswer).Score-1.6) > 1e-7 {
		t.Fatalf("unexpected answers: %+v %+v", choice, score)
	}
}
