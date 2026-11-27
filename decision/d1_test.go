package decision

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llm"
)

func d1Request(t *testing.T, body string) Request {
	t.Helper()
	var req Request
	if err := json.Unmarshal([]byte(body), &req); err != nil {
		t.Fatal(err)
	}
	return req
}

func TestCompileD1PassesQuestionsInOrder(t *testing.T) {
	req := d1Request(t, `{
		"model": "d1",
		"state": {"ticket": "Charged twice", "order": 4417},
		"questions": {
			"urgency": {"type":"score","instructions":"How urgent?","criteria":["low","high"]},
			"refund": {"type":"noul","instructions":"Is a refund requested?"},
			"team": {"type":"choice","instructions":"Which team?","criteria":{"billing":null,"shipping":"Parcels"}}
		}
	}`)
	c, err := Compile(req, "d1")
	if err != nil {
		t.Fatal(err)
	}
	native := c.Request.Native
	if native == nil || string(native.State) != string(req.State) {
		t.Fatalf("native request %+v does not carry the state", native)
	}
	var names []string
	for _, q := range native.Questions {
		names = append(names, q.Name)
	}
	if strings.Join(names, ",") != "urgency,refund,team" {
		t.Fatalf("question order %v, want request order", names)
	}
	var team api.SystemOneQuestion
	if err := json.Unmarshal(native.Questions[2].Question, &team); err != nil || string(team.Criteria) != `{"billing":null,"shipping":"Parcels"}` {
		t.Fatalf("team question %+v, %v; want its criteria in request order", team, err)
	}

	answers := []json.RawMessage{
		json.RawMessage(`{"type":"score","score":0.2}`),
		json.RawMessage(`{"type":"noul","noul":0.9}`),
		json.RawMessage(`{"type":"choice","choice":"billing"}`),
	}
	res, err := c.Answer("d1", llm.ScoreResponse{Answers: answers, InputTokens: 42})
	if err != nil {
		t.Fatal(err)
	}
	if res.Usage.InputTokens != 42 || res.Answers.Len() != 3 {
		t.Fatalf("unexpected response %+v", res)
	}
	if got, _ := res.Answers.Get("team"); string(got.(json.RawMessage)) != string(answers[2]) {
		t.Fatalf("team answer %s, want %s", got, answers[2])
	}
	if _, err := c.Answer("d1", llm.ScoreResponse{Answers: answers[:2]}); err == nil {
		t.Fatal("expected an error for a missing answer")
	}
}

func TestCompileD1ValidatesLikeOtherModels(t *testing.T) {
	for _, body := range []string{
		`{"model":"m","state":"x","questions":{"q":{"type":"text","instructions":"Who?"}}}`,
		`{"model":"m","state":"x","questions":{"q":{"type":"choice","instructions":"Which?","criteria":{"only":null}}}}`,
		`{"model":"m","state":"x","questions":{"q":{"type":"score","instructions":"How?","criteria":["one"]}}}`,
		`{"model":"m","state":"x","questions":{"q":{"type":"noul","instructions":"  "}}}`,
		`{"model":"m","state":"x","questions":{"q":{"type":"noul","instructions":"Is it?","criteria":{"maybe":"x"}}}}`,
		`{"model":"m","state":"","questions":{"q":{"type":"noul","instructions":"Is it?"}}}`,
		`{"model":"m","state":null,"questions":{"q":{"type":"noul","instructions":"Is it?"}}}`,
		`{"model":"m","state":"x","questions":{}}`,
	} {
		req := d1Request(t, body)
		if _, err := Compile(req, ""); err == nil {
			t.Fatalf("%s: the default encoder accepts it; the case tests nothing", body)
		}
		if _, err := Compile(req, "d1"); err == nil {
			t.Errorf("%s: d1 accepted what other decision models reject", body)
		}
	}
}

func TestCompileD1Images(t *testing.T) {
	req := d1Request(t, `{"model":"d1","state":"","questions":{"cat":{"type":"noul","instructions":"Is there a cat?"}}}`)
	req.Images = []api.ImageData{[]byte("picture")}
	c, err := Compile(req, "d1")
	if err != nil {
		t.Fatal(err)
	}
	if len(c.Request.Images) != 1 {
		t.Fatal("images were not passed to the model")
	}
}
