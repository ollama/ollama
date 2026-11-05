package decision

import (
	"encoding/json"
	"fmt"
	"strings"
	"testing"

	"github.com/ollama/ollama/llm"
)

func hindsightRequest(t *testing.T, body string) Request {
	t.Helper()
	var req Request
	if err := json.Unmarshal([]byte(body), &req); err != nil {
		t.Fatal(err)
	}
	return req
}

func TestHindsightCompileKeepsRequestNative(t *testing.T) {
	req := hindsightRequest(t, `{"model":"hs","state":{"b":1,"a":"é"},"questions":{
		"keep":{"type":"noul"},
		"team":{"type":"choice","instructions":"Which?","criteria":{"billing":null,"bug":"Errors"}},
		"urgency":{"type":"score","criteria":["low",{"level":"high"}]}}}`)
	c, err := Compile(req, "hindsight")
	if err != nil {
		t.Fatal(err)
	}
	native := c.Request.Native
	if native == nil || len(native.Questions) != 3 || native.Questions[1].Name != "team" {
		t.Fatalf("native request = %+v", native)
	}
	if string(native.State) != `{"b":1,"a":"é"}` {
		t.Fatalf("state = %s", native.State)
	}
	var values []string
	for _, f := range c.fields {
		for _, choice := range f.Choices {
			values = append(values, fmt.Sprint(choice.Value))
		}
	}
	if got := strings.Join(values, ","); got != "false,true,billing,bug,0,1" {
		t.Fatalf("choices = %s", got)
	}
}

func TestHindsightAnswerConfidence(t *testing.T) {
	req := hindsightRequest(t, `{"model":"hs","state":"s","questions":{
		"q":{"type":"choice","criteria":{"a":"x","b":"y","c":"z"}}}}`)
	c, err := Compile(req, "hindsight")
	if err != nil {
		t.Fatal(err)
	}
	response, err := c.Answer("hs", llm.ScoreResponse{Logits: [][]float32{{2, 0, 0}}})
	if err != nil {
		t.Fatal(err)
	}
	answer, _ := response.Answers.Get("q")
	got := answer.(ChoiceAnswer)
	// (n·p_max − 1)/(n − 1) with p_max = e²/(e²+2).
	want := (3*0.786986042484952 - 1) / 2
	if got.Choice != "a" || got.Confidence < want-1e-9 || got.Confidence > want+1e-9 {
		t.Fatalf("answer = %+v, want choice a with confidence %v", got, want)
	}
}

func TestHindsightLimits(t *testing.T) {
	criteria := map[string]string{}
	for i := range 256 {
		criteria[fmt.Sprint("k", i)] = "d"
	}
	many, _ := json.Marshal(criteria)
	for name, body := range map[string]string{
		"one option":    `{"model":"hs","state":"s","questions":{"q":{"type":"choice","criteria":{"a":"x"}}}}`,
		"256 options":   `{"model":"hs","state":"s","questions":{"q":{"type":"choice","criteria":` + string(many) + `}}}`,
		"one level":     `{"model":"hs","state":"s","questions":{"q":{"type":"score","criteria":["a"]}}}`,
		"eleven levels": `{"model":"hs","state":"s","questions":{"q":{"type":"score","criteria":["0","1","2","3","4","5","6","7","8","9","10"]}}}`,
		"bad type":      `{"model":"hs","state":"s","questions":{"q":{"type":"rank"}}}`,
		"no questions":  `{"model":"hs","state":"s","questions":{}}`,
	} {
		if _, err := Compile(hindsightRequest(t, body), "hindsight"); err == nil {
			t.Errorf("%s: compiled without error", name)
		}
	}
	criteria = map[string]string{}
	for i := range 255 {
		criteria[fmt.Sprint("k", i)] = "d"
	}
	most, _ := json.Marshal(criteria)
	if _, err := Compile(hindsightRequest(t, `{"model":"hs","state":"s","questions":{"q":{"type":"choice","criteria":`+string(most)+`}}}`), "hindsight"); err != nil {
		t.Errorf("255 options: %v", err)
	}
}
