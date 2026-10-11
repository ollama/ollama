package decision

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
)

func TestPublisherPrompts(t *testing.T) {
	var req Request
	if err := json.Unmarshal([]byte(`{"model":"decision","state":"Paid €20 & <tag>. Literal \\u2028, quote \" and colon: comma,","questions":{"refund":{"type":"noul","instructions":"Refund requested?"}}}`), &req); err != nil {
		t.Fatal(err)
	}
	// Publisher contracts: Together's examples/decide.py payload() and Bespoke's
	// parallel_schema.prepare_prompts(), with false/true mapped to No/Yes.
	for _, tc := range []struct{ renderer, want string }{
		{"tev1", `{"state": "Paid €20 & <tag>. Literal \\u2028, quote \" and colon: comma,", "question": "Refund requested?", "options": [{"label": "A", "key": "false", "description": "No"}, {"label": "B", "key": "true", "description": "Yes"}]}`},
		{"", `{"context": "Paid €20 & \u003ctag\u003e. Literal \\u2028, quote \" and colon: comma,", "schema": [{"name": "refund", "description": "Refund requested?", "choices": [{"code": "A", "value": false, "description": "No"}, {"code": "B", "value": true, "description": "Yes"}]}]}` + "\n\nRequested field: \"refund\""},
	} {
		t.Run(tc.renderer, func(t *testing.T) {
			compiled, err := Compile(req, tc.renderer)
			if err != nil {
				t.Fatal(err)
			}
			if err := compiled.Render(func(messages []api.Message) (string, error) { return messages[0].Content, nil }); err != nil {
				t.Fatal(err)
			}
			if got := compiled.Request.Rows[0].Prompt; got != tc.want {
				t.Fatalf("prompt differs from publisher format:\ngot  %s\nwant %s", got, tc.want)
			}
		})
	}
	got, err := promptJSON("\u2028\u2029\\u2028\\u2029")
	if want := "\"\u2028\u2029\\\\u2028\\\\u2029\""; err != nil || got != want {
		t.Fatalf("Unicode escaping = %q, %v; want %q", got, err, want)
	}
}

func TestTevQuestions(t *testing.T) {
	req := testRequest(t)
	req.State = json.RawMessage(`{"frames":["first","second"]}`)
	compiled, err := Compile(req, "tev1")
	if err != nil {
		t.Fatal(err)
	}
	if err := compiled.Render(func(messages []api.Message) (string, error) { return messages[0].Content, nil }); err != nil {
		t.Fatal(err)
	}
	for i, row := range compiled.Request.Rows {
		var payload struct {
			State    string
			Question string
			Options  []struct{ Label, Key, Description string }
		}
		if err := json.Unmarshal([]byte(row.Prompt), &payload); err != nil {
			t.Fatal(err)
		}
		f := compiled.fields[i]
		if payload.State != `{"frames":["first","second"]}` || payload.Question != f.Description || len(payload.Options) != len(f.Choices) {
			t.Fatalf("question %d was not compiled independently: %s", i, row.Prompt)
		}
		for j, option := range payload.Options {
			if option.Label != row.Candidates[j] || option.Description != f.Choices[j].Description || option.Key == "" {
				t.Fatalf("question %d option %d: %+v", i, j, option)
			}
		}
	}
	for _, count := range []int{24, 25, 26} {
		q, _ := req.Questions.Get("urgency")
		q.Criteria = json.RawMessage(`["x"` + strings.Repeat(`,"x"`, count-1) + `]`)
		req.Questions.Set("urgency", q)
		_, err := Compile(req, "tev1")
		if (err == nil) != (count == 24) {
			t.Errorf("Tev1 accepted %d candidates: %v", count, err)
		}
	}
}
