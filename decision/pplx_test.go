package decision

import (
	"encoding/json"
	"math"
	"slices"
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llm"
)

func TestPplxCompilePreservesQuestionAndOptionOrder(t *testing.T) {
	var req Request
	if err := json.Unmarshal([]byte(`{
		"model":"pplx-test",
		"state":{"name":"Ada","active":true},
		"questions":{
			"category":{"type":"choice","instructions":"Select one.","criteria":{"second":"Later","first":"Earlier"}},
			"approved":{"type":"noul","criteria":{"true":"Approved"}},
			"rating":{"type":"score","instructions":"Rate it.","criteria":["Low","Medium","High"]}
		}
	}`), &req); err != nil {
		t.Fatal(err)
	}
	c, err := Compile(req, "pplx")
	if err != nil {
		t.Fatal(err)
	}
	if c.Request.State != `{"name": "Ada", "active": true}` {
		t.Errorf("state = %q", c.Request.State)
	}
	if len(c.Request.Rows) != 3 || !slices.Equal(c.Request.Rows[0].Question.Options, []string{"second: Later", "first: Earlier"}) {
		t.Fatalf("choice options = %+v", c.Request.Rows)
	}
	if !slices.Equal(c.Request.Rows[1].Question.Options, []string{"No / false", "Approved"}) {
		t.Errorf("noul options = %v", c.Request.Rows[1].Question.Options)
	}
	if !slices.Equal(c.Request.Rows[2].Question.Options, []string{"Low", "Medium", "High"}) {
		t.Errorf("score options = %v", c.Request.Rows[2].Question.Options)
	}
	if !slices.Equal([]string{c.fields[0].Name, c.fields[1].Name, c.fields[2].Name}, []string{"category", "approved", "rating"}) {
		t.Errorf("question order changed: %+v", c.fields)
	}
	if err := c.Render(func([]api.Message) (string, error) {
		t.Fatal("PPLX prompts must be rendered by the model scorer")
		return "", nil
	}); err != nil {
		t.Fatal(err)
	}
}

func TestPplxAnswerUsesDecisionConfidence(t *testing.T) {
	var req Request
	if err := json.Unmarshal([]byte(`{"model":"pplx-test","state":"x","questions":{"kind":{"type":"choice","criteria":{"a":"A","b":"B","c":"C"}},"rating":{"type":"score","criteria":["Low","Medium","High"]}}}`), &req); err != nil {
		t.Fatal(err)
	}
	c, err := Compile(req, "pplx")
	if err != nil {
		t.Fatal(err)
	}
	got, err := c.Answer("pplx-test", llm.ScoreResponse{Logits: [][]float32{{2, 1, 0}, {0, 1, 2}}})
	if err != nil {
		t.Fatal(err)
	}
	choiceValue, _ := got.Answers.Get("kind")
	scoreValue, _ := got.Answers.Get("rating")
	choice := choiceValue.(ChoiceAnswer)
	score := scoreValue.(ScoreAnswer)
	p := math.Exp(2) / (math.Exp(2) + math.E + 1)
	wantChoice := (p - 1.0/3) / (1 - 1.0/3)
	if choice.Choice != "a" || math.Abs(choice.Confidence-wantChoice) > 1e-6 {
		t.Errorf("choice = %+v, want confidence %g", choice, wantChoice)
	}
	if score.Score <= 1 || score.Confidence <= 0 || score.Confidence >= 1 {
		t.Errorf("score = %+v", score)
	}
}

func TestPplxFalsyDescriptionsUseDefaults(t *testing.T) {
	var req Request
	if err := json.Unmarshal([]byte(`{"model":"pplx-test","state":"x","questions":{"approved":{"type":"noul","instructions":false,"criteria":{"false":[],"true":0}}}}`), &req); err != nil {
		t.Fatal(err)
	}
	c, err := Compile(req, "pplx")
	if err != nil {
		t.Fatal(err)
	}
	if got := c.Request.Rows[0].Question.Instructions; got != "Choose the best matching option." {
		t.Errorf("default instructions = %q", got)
	}
	if got := c.Request.Rows[0].Question.Options; !slices.Equal(got, []string{"No / false", "Yes / true"}) {
		t.Errorf("default noul options = %v", got)
	}
}

func TestPplxScoreNeedsTwoLevels(t *testing.T) {
	var req Request
	if err := json.Unmarshal([]byte(`{"model":"pplx-test","state":"x","questions":{"rating":{"type":"score","criteria":["Only"]}}}`), &req); err != nil {
		t.Fatal(err)
	}
	if _, err := Compile(req, "pplx"); err == nil || !strings.Contains(err.Error(), "2–10 levels") {
		t.Fatalf("Compile() error = %v, want two-level rejection", err)
	}
}

func TestPplxKeepsSharedScoreLevelLimit(t *testing.T) {
	var req Request
	levels := strings.TrimSuffix(strings.Repeat(`"level",`, 11), ",")
	if err := json.Unmarshal([]byte(`{"model":"pplx-test","state":"x","questions":{"rating":{"type":"score","criteria":[`+levels+`]}}}`), &req); err != nil {
		t.Fatal(err)
	}
	for _, encoding := range []string{"pplx", "tev1"} {
		t.Run(encoding, func(t *testing.T) {
			if _, err := Compile(req, encoding); err == nil || !strings.Contains(err.Error(), "2–10 levels") {
				t.Fatalf("Compile(%q) error = %v, want shared limit", encoding, err)
			}
		})
	}
}

func TestPplxRejectsEmptyImageBeforeScoring(t *testing.T) {
	var req Request
	if err := json.Unmarshal([]byte(`{"model":"pplx-test","state":"x","questions":{"kind":{"type":"choice","criteria":{"a":"A","b":"B"}}},"images":[""]}`), &req); err != nil {
		t.Fatal(err)
	}
	if _, err := Compile(req, "pplx"); err == nil || !strings.Contains(err.Error(), "image 0 must not be empty") {
		t.Fatalf("Compile() error = %v, want empty-image rejection", err)
	}
}

func FuzzPplxCompile(f *testing.F) {
	f.Add([]byte(`{"model":"pplx-test","state":"x","questions":{"kind":{"type":"choice","criteria":{"a":"A","b":"B"}}}}`))
	f.Add([]byte(`{"model":"pplx-test","state":{},"questions":{"rating":{"type":"score","criteria":[0,1]}}}`))
	f.Fuzz(func(t *testing.T, data []byte) {
		if len(data) > 1<<16 {
			return
		}
		var req Request
		if json.Unmarshal(data, &req) == nil {
			_, _ = Compile(req, "pplx")
		}
	})
}
