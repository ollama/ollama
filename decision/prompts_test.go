package decision

import (
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
)

func TestCompiledPromptBudget(t *testing.T) {
	req := testRequest(t)
	req.State = json.RawMessage(`"` + strings.Repeat("x", 1<<20) + `"`)
	req.Questions = &Questions{}
	for i := range 64 {
		req.Questions.Set(fmt.Sprintf("q%d", i), Question{Type: "noul", Instructions: json.RawMessage(`"q"`)})
	}
	for _, encoding := range []string{"", "tev1", "strands"} {
		t.Run(encoding, func(t *testing.T) {
			_, err := Compile(req, encoding)
			if !errors.Is(err, ErrPromptTooLarge) {
				t.Fatalf("Compile() error = %v, want prompt budget error", err)
			}
		})
	}
}

func TestRenderedPromptBudget(t *testing.T) {
	compiled, err := Compile(testRequest(t), "")
	if err != nil {
		t.Fatal(err)
	}
	large := strings.Repeat("x", maxCompiledPromptBytes/2+1)
	if err := compiled.Render(func([]api.Message) (string, error) { return large, nil }); !errors.Is(err, ErrPromptTooLarge) {
		t.Fatalf("Render() error = %v, want prompt budget error", err)
	}
}

func TestPublisherPrompts(t *testing.T) {
	var req Request
	if err := json.Unmarshal([]byte(`{"model":"decision","state":"Paid €20 & <tag>. Literal \\u2028, quote \" and colon: comma,","questions":{"refund":{"type":"noul","instructions":"Refund requested?"}}}`), &req); err != nil {
		t.Fatal(err)
	}
	// Publisher contracts: Together's examples/decide.py payload(), with
	// false/true described as No/Yes, and Bespoke's native_schema(), which
	// describes only the choices the request describes.
	for _, tc := range []struct{ renderer, want string }{
		{"tev1", `{"state": "Paid €20 & <tag>. Literal \\u2028, quote \" and colon: comma,", "question": "Refund requested?", "options": [{"label": "A", "key": "false", "description": "No"}, {"label": "B", "key": "true", "description": "Yes"}]}`},
		{"", `{"context": "Paid €20 & \u003ctag\u003e. Literal \\u2028, quote \" and colon: comma,", "schema": [{"name": "refund", "description": "Refund requested?", "choices": [{"code": "A", "value": false}, {"code": "B", "value": true}]}]}` + "\n\nRequested field: \"refund\""},
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
	descriptions := [][]string{{"Charges and refunds", "technical"}, {"No", "Yes"}, {"Routine", "Urgent", "Emergency"}}
	for i, row := range compiled.Request.Rows {
		var payload struct {
			State    json.RawMessage
			Question string
			Options  []struct{ Label, Key, Description string }
		}
		if err := json.Unmarshal([]byte(row.Prompt), &payload); err != nil {
			t.Fatal(err)
		}
		f := compiled.fields[i]
		if string(payload.State) != `{"frames": ["first", "second"]}` || payload.Question != f.Description || len(payload.Options) != len(f.Choices) {
			t.Fatalf("question %d was not compiled independently: %s", i, row.Prompt)
		}
		for j, option := range payload.Options {
			if option.Label != row.Candidates[j] || option.Description != descriptions[i][j] || option.Key == "" {
				t.Fatalf("question %d option %d: %+v", i, j, option)
			}
		}
	}
	for _, count := range []int{24, 25, 26} {
		q, _ := req.Questions.Get("department")
		criteria := make([]string, count)
		for i := range criteria {
			criteria[i] = fmt.Sprintf(`"%d":"x"`, i)
		}
		q.Criteria = json.RawMessage(`{` + strings.Join(criteria, ",") + `}`)
		req.Questions.Set("department", q)
		_, err := Compile(req, "tev1")
		if (err == nil) != (count == 24) {
			t.Errorf("Tev1 accepted %d candidates: %v", count, err)
		}
	}
}
