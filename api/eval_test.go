package api

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"
)

func TestEvalClient(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost || r.URL.Path != "/api/eval" {
			t.Errorf("unexpected request: %s %s", r.Method, r.URL.Path)
		}
		var body map[string]json.RawMessage
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Error(err)
		}
		if len(body) != 3 || string(body["state"]) != `["text"]` || body["questions"] == nil || string(body["model"]) != `"local-evaluator"` {
			t.Errorf("unexpected request body: %s", body)
		}
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"model":"local-evaluator","answers":{"urgent":{"type":"noul","noul":0}},"usage":{"input_tokens":10,"output_tokens":4}}`))
	}))
	defer server.Close()
	u, err := url.Parse(server.URL)
	if err != nil {
		t.Fatal(err)
	}
	resp, err := NewClient(u, server.Client()).Eval(context.Background(), &EvalRequest{
		Model: "local-evaluator", State: json.RawMessage(`["text"]`),
		Questions: map[string]EvalQuestion{"urgent": {Type: "noul", Instructions: json.RawMessage(`"Urgent?"`)}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if resp.Answers["urgent"].Noul == nil || *resp.Answers["urgent"].Noul != 0 || resp.Usage.InputTokens != 10 {
		t.Fatalf("unexpected response: %+v", resp)
	}
}

func TestEvalRequest(t *testing.T) {
	// The question IDs are caller-defined, including IDs that are not Go identifiers.
	const body = `{
		"model":"local-evaluator",
		"state":{"messages":["Help! My payouts have been failing for 3 days."]},
		"questions":{
			"is urgent?":{"type":"noul","instructions":"Does this convey urgency?","criteria":{"true":"Time-sensitive","false":"No urgency"}},
			"department":{"type":"choice","instructions":{"task":"Choose a team"},"criteria":{"billing":"Payments","technical":null}},
			"frustration":{"type":"score","instructions":["Rate frustration"],"criteria":["Calm","Frustrated","Angry"]}
		}
	}`
	var req EvalRequest
	if err := json.Unmarshal([]byte(body), &req); err != nil {
		t.Fatal(err)
	}
	if err := req.Validate(); err != nil {
		t.Fatal(err)
	}
	for _, state := range []string{`"text"`, `[]`, `{}`} {
		req.State = json.RawMessage(state)
		if err := req.Validate(); err != nil {
			t.Fatalf("%s: %v", state, err)
		}
	}
}

func TestEvalValidation(t *testing.T) {
	for _, body := range []string{
		`{}`,
		`{"model":"m","state":null,"questions":{"q":{"type":"noul","instructions":"?"}}}`,
		`{"model":"m","state":1,"questions":{"q":{"type":"noul","instructions":"?"}}}`,
		`{"model":"m","state":true,"questions":{"q":{"type":"noul","instructions":"?"}}}`,
		`{"model":"m","state":"text","questions":{}}`,
		`{"model":"m","state":"text","questions":{"q":null}}`,
	} {
		var req EvalRequest
		if err := json.Unmarshal([]byte(body), &req); err != nil {
			t.Fatal(err)
		}
		if err := req.Validate(); err == nil {
			t.Fatalf("accepted %s", body)
		}
	}
	for _, question := range []string{
		`{"type":"entity","instructions":"?"}`,
		`{"type":"noul"}`, `{"type":"noul","instructions":null}`,
		`{"type":"noul","instructions":false}`,
		`{"type":"noul","instructions":"?","criteria":null}`,
		`{"type":"noul","instructions":"?","criteria":{"true":null}}`,
		`{"type":"noul","instructions":"?","criteria":{"yes":"yes"}}`,
		`{"type":"choice","instructions":"?"}`,
		`{"type":"choice","instructions":"?","criteria":{}}`,
		`{"type":"choice","instructions":"?","criteria":{"a":1}}`,
		`{"type":"score","instructions":"?","criteria":["low"]}`,
		`{"type":"score","instructions":"?","criteria":["low",null]}`,
	} {
		var q EvalQuestion
		if err := json.Unmarshal([]byte(question), &q); err != nil {
			t.Fatal(err)
		}
		r := EvalRequest{Model: "m", State: json.RawMessage(`"text"`), Questions: map[string]EvalQuestion{"my-id": q}}
		if err := r.Validate(); err == nil || !strings.Contains(err.Error(), `questions["my-id"]`) {
			t.Fatalf("%s: %v", question, err)
		}
	}
}

func TestEvalZeroAnswer(t *testing.T) {
	zero := 0.0
	for _, tc := range []struct {
		answer EvalAnswer
		want   string
	}{
		{EvalAnswer{Type: "noul", Noul: &zero}, `{"type":"noul","noul":0}`},
		{EvalAnswer{Type: "score", Score: &zero, Confidence: &zero}, `{"type":"score","score":0,"confidence":0}`},
	} {
		got, err := json.Marshal(tc.answer)
		if err != nil || string(got) != tc.want {
			t.Fatalf("got %s, %v; want %s", got, err, tc.want)
		}
	}
}
