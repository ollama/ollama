package decision

import (
	"encoding/json"
	"slices"
	"strconv"
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llm"
)

func helmPrompts(t *testing.T, req Request) *Compiled {
	t.Helper()
	compiled, err := Compile(req, "helm")
	if err != nil {
		t.Fatal(err)
	}
	if err := compiled.Render(func(messages []api.Message) (string, error) {
		if len(messages) != 1 || messages[0].Role != "user" {
			t.Fatalf("Helm embeds its instruction in the user message: %+v", messages)
		}
		return messages[0].Content, nil
	}); err != nil {
		t.Fatal(err)
	}
	return compiled
}

func TestHelmPrompt(t *testing.T) {
	var req Request
	if err := json.Unmarshal([]byte(`{"model":"helm","state":"Paid €20 & <tag>. Literal \\u2028, quote \" and colon: comma,","questions":{"team":{"type":"choice","instructions":"Which team?","criteria":{"billing":"Charges and refunds","technical":null}}}}`), &req); err != nil {
		t.Fatal(err)
	}
	compiled := helmPrompts(t, req)
	// saina.loader.encode_head_prompt: json.dumps(..., ensure_ascii=False) after
	// the instruction, with options labelled by code.
	want := helmInstruction + `{"context": "Paid €20 & <tag>. Literal \\u2028, quote \" and colon: comma,", "question": "Which team?", "options": [{"code": "A", "text": "billing: Charges and refunds"}, {"code": "B", "text": "technical"}]}`
	if got := compiled.Request.Rows[0].Prompt; got != want {
		t.Fatalf("prompt differs from Helm's training format:\ngot  %s\nwant %s", got, want)
	}
	if got := compiled.Request.Rows[0].Candidates; !slices.Equal(got, []string{"A", "B"}) {
		t.Fatalf("candidates = %v", got)
	}
}

func TestHelmContent(t *testing.T) {
	req := testRequest(t)
	req.State = json.RawMessage(`{"frames":["first","second"],"n":1.5}`)
	compiled := helmPrompts(t, req)
	for i, row := range compiled.Request.Rows {
		var payload struct {
			Context  string
			Question string
			Options  []struct{ Code, Text string }
		}
		if err := json.Unmarshal([]byte(strings.TrimPrefix(row.Prompt, helmInstruction)), &payload); err != nil {
			t.Fatal(err)
		}
		f := compiled.fields[i]
		// Objects render as Python's json.dumps: ", " and ": " separators.
		if payload.Context != `{"frames": ["first", "second"], "n": 1.5}` || payload.Question != f.Description || len(payload.Options) != len(f.Choices) {
			t.Fatalf("question %d was not compiled independently: %s", i, row.Prompt)
		}
		for j, option := range payload.Options {
			if option.Code != row.Candidates[j] || option.Text != row.Question.Options[j] {
				t.Fatalf("question %d option %d: %+v", i, j, option)
			}
		}
	}
	texts := map[string][]string{
		"department": {"billing: Charges and refunds", "technical"},
		"refund":     {"Yes: The answer is yes", "No: The answer is no"},
		"urgency":    {"0: Routine", "1: Urgent", "2: Emergency"},
	}
	for i, f := range compiled.fields {
		if got := compiled.Request.Rows[i].Question.Options; !slices.Equal(got, texts[f.Name]) {
			t.Errorf("%s options = %q, want %q", f.Name, got, texts[f.Name])
		}
	}
	refund := compiled.fields[1]
	if refund.Choices[0].Value != true || refund.Choices[1].Value != false {
		t.Fatalf("Helm scores yes before no: %+v", refund.Choices)
	}
	if len(compiled.Request.Rows) != 3 || len(compiled.Request.PointerRows) != 0 || len(compiled.Request.Fields) != 0 {
		t.Fatalf("Helm uses candidate rows: %+v", compiled.Request)
	}
}

func TestHelmAnswers(t *testing.T) {
	req := testRequest(t)
	compiled := helmPrompts(t, req)
	response, err := compiled.Answer("helm", llm.ScoreResponse{Logits: [][]float32{{0, 3}, {2, 0}, {0, 1, 2}}, InputTokens: 30})
	if err != nil {
		t.Fatal(err)
	}
	department, _ := response.Answers.Get("department")
	if answer := department.(ChoiceAnswer); answer.Choice != "technical" {
		t.Fatalf("department = %+v", answer)
	}
	refund, _ := response.Answers.Get("refund")
	if answer := refund.(NoulAnswer); answer.Noul < 0.88 {
		t.Fatalf("refund = %+v", answer)
	}
	urgency, _ := response.Answers.Get("urgency")
	if answer := urgency.(ScoreAnswer); answer.Score < 1.5 {
		t.Fatalf("urgency = %+v", answer)
	}
}

func TestHelmLimits(t *testing.T) {
	if len(helmCodes) != 588 {
		t.Fatalf("%d codes", len(helmCodes))
	}
	seen := map[string]bool{}
	for i, code := range helmCodes {
		if seen[code] || len(code) < 1 || len(code) > 2 || strings.ToUpper(code) != code || (i < 26 && code != string(rune('A'+i))) {
			t.Fatalf("code %d = %q", i, code)
		}
		seen[code] = true
	}
	for _, tc := range []struct {
		name     string
		question string
		ok       bool
	}{
		{"largest choice", `{"type":"choice","instructions":"q","criteria":{` + options(255) + `}}`, true},
		{"too many choices", `{"type":"choice","instructions":"q","criteria":{` + options(256) + `}}`, false},
		{"one choice", `{"type":"choice","instructions":"q","criteria":{"a":null}}`, false},
		{"blank key", `{"type":"choice","instructions":"q","criteria":{" ":null,"b":null}}`, false},
		{"structured description", `{"type":"choice","instructions":{"ask":"q"},"criteria":{"a":{"when":["x"]},"b":null}}`, true},
		{"ten levels", `{"type":"score","instructions":"q","criteria":["x","x","x","x","x","x","x","x","x","x"]}`, true},
		{"eleven levels", `{"type":"score","instructions":"q","criteria":["x","x","x","x","x","x","x","x","x","x","x"]}`, false},
		{"null level", `{"type":"score","instructions":"q","criteria":["x",null]}`, false},
		{"noul descriptions", `{"type":"noul","instructions":"q","criteria":{"true":"asks for money back","false":"anything else"}}`, true},
		{"noul unknown key", `{"type":"noul","instructions":"q","criteria":{"maybe":"x"}}`, false},
		{"empty instructions", `{"type":"noul","instructions":" "}`, false},
		{"unknown type", `{"type":"rank","instructions":"q","criteria":["x","x"]}`, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var req Request
			if err := json.Unmarshal([]byte(`{"model":"helm","state":"x","questions":{"q":`+tc.question+`}}`), &req); err != nil {
				t.Fatal(err)
			}
			compiled, err := Compile(req, "helm")
			if (err == nil) != tc.ok {
				t.Fatalf("accepted = %v: %v", err == nil, err)
			}
			if tc.name == "largest choice" {
				if codes := compiled.Request.Rows[0].Candidates; len(codes) != 255 || codes[26] != "AA" || codes[254] != helmCodes[254] {
					t.Fatalf("codes = %v", codes)
				}
			}
			if tc.name == "structured description" && compiled.Request.Rows[0].Question.Options[0] != `a: {"when": ["x"]}` {
				t.Fatalf("options = %v", compiled.Request.Rows[0].Question.Options)
			}
		})
	}
	req := testRequest(t)
	req.Images = []api.ImageData{[]byte("image")}
	if _, err := Compile(req, "helm"); err == nil {
		t.Fatal("Helm must reject images")
	}
}

func options(n int) string {
	var b strings.Builder
	for i := range n {
		if i > 0 {
			b.WriteByte(',')
		}
		b.WriteString(`"o` + strconv.Itoa(i) + `":null`)
	}
	return b.String()
}
