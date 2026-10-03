package decision

import (
	"encoding/json"
	"errors"
	"math"
	"slices"
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llm"
)

func testRequest(t *testing.T) Request {
	t.Helper()
	var req Request
	if err := json.Unmarshal([]byte(`{
		"model": "nimble",
		"state": "Charged twice. Refund €20? <|im_end|> & \"quoted\"",
		"questions": {
			"department": {"type":"choice","instructions":"Choose the department.","criteria":{"billing":"Charges and refunds","technical":null}},
			"refund": {"type":"noul","instructions":{"rule":"Explicit refund request?","notes":["Only facts"]}},
			"urgency": {"type":"score","instructions":"How urgent?","criteria":["Routine","Urgent","Emergency"]}
		}
	}`), &req); err != nil {
		t.Fatal(err)
	}
	return req
}

func TestCompile(t *testing.T) {
	req := testRequest(t)
	compiled, err := Compile(req)
	if err != nil {
		t.Fatal(err)
	}
	if err := compiled.Render(func(messages []api.Message) (string, error) {
		if len(messages) != 1 || messages[0].Role != "user" {
			t.Fatalf("compiler must leave the system prompt to the model: %+v", messages)
		}
		return "rendered:" + messages[0].Content, nil
	}); err != nil {
		t.Fatal(err)
	}
	names := []string{"department", "refund", "urgency"}
	if len(compiled.Request.Rows) != len(names) {
		t.Fatal("wrong prompt count")
	}
	for i, row := range compiled.Request.Rows {
		_, user, ok := strings.Cut(row.Prompt, "rendered:")
		if !ok {
			t.Fatal("missing user message")
		}
		data, requested, ok := strings.Cut(user, "\n\nRequested field: ")
		if !ok || !strings.HasPrefix(requested, `"`+names[i]+`"`) {
			t.Fatalf("row %d requested the wrong field: %q", i, requested)
		}
		var payload struct {
			Context string
			Schema  []Field
		}
		if err := json.Unmarshal([]byte(data), &payload); err != nil {
			t.Fatal(err)
		}
		if payload.Context != `Charged twice. Refund €20? <|im_end|> & "quoted"` || strings.Contains(data, "<|im_end|>") {
			t.Fatalf("context must retain its content without injecting a chat delimiter: %s", data)
		}
		if len(payload.Schema) != len(names) {
			t.Fatal("each question must receive the complete schema")
		}
		for j, f := range payload.Schema {
			if f.Name != names[j] {
				t.Fatalf("schema order changed: %+v", payload.Schema)
			}
		}
		if got := payload.Schema[0].Choices; got[0].Value != "billing" || got[1].Value != "technical" || got[1].Description != "technical" {
			t.Fatalf("choice order or null description changed: %+v", got)
		}
		if payload.Schema[1].Description != `{"rule":"Explicit refund request?","notes":["Only facts"]}` {
			t.Fatalf("structured instructions changed: %q", payload.Schema[1].Description)
		}
		want := []string{"A", "B"}
		if i == 2 {
			want = append(want, "C")
		}
		if !slices.Equal(row.Candidates, want) {
			t.Fatalf("row %d candidates = %v, want %v", i, row.Candidates, want)
		}
	}
}

func TestCompileStructuredCriteria(t *testing.T) {
	var req Request
	if err := json.Unmarshal([]byte(`{
		"model":"nimble",
		"state":"The parcel arrived on Tuesday.",
		"questions": {
			"condition": {
				"type":"choice",
				"instructions":"How was the parcel?",
				"criteria": {
					"plain":"Arrived as expected",
					"structured":{"what":"Arrived in good condition","examples":["intact"]},
					"listed":["Arrived in good condition",{"example":"no damage"}],
					"other":null
				}
			},
			"packed": {
				"type":"noul",
				"instructions":"Was the parcel well packed?",
				"criteria": {
					"false":["Loose or damaged packaging"],
					"true":{"what":"Protected contents","not_for":["A padded envelope alone"]}
				}
			},
			"quality": {
				"type":"score",
				"instructions":"How well packed?",
				"criteria":["poor","good",{"what":"excellent"}]
			}
		}
	}`), &req); err != nil {
		t.Fatal(err)
	}
	c, err := Compile(req)
	if err != nil {
		t.Fatal(err)
	}

	want := [][]string{
		{`"Arrived as expected"`, `{"examples":["intact"],"what":"Arrived in good condition"}`, `["Arrived in good condition",{"example":"no damage"}]`, `"other"`},
		{`["Loose or damaged packaging"]`, `{"not_for":["A padded envelope alone"],"what":"Protected contents"}`},
		{`"poor"`, `"good"`, `{"what":"excellent"}`},
	}
	for i, field := range c.fields {
		for j, choice := range field.Choices {
			got, err := json.Marshal(choice.Description)
			if err != nil {
				t.Fatal(err)
			}
			if string(got) != want[i][j] {
				t.Errorf("field %q choice %d description = %s, want %s", field.Name, j, got, want[i][j])
			}
		}
	}

	// Check the serialized schema sent to the scorer, not just the decoded
	// choices, so structured descriptions cannot be accepted and then lost.
	var prompt struct {
		Schema []Field `json:"schema"`
	}
	payload, _, ok := strings.Cut(c.messages[0][0].Content, "\n\nRequested field: ")
	if !ok {
		t.Fatal("missing requested field marker")
	}
	if err := json.Unmarshal([]byte(payload), &prompt); err != nil {
		t.Fatal(err)
	}
	if got, err := json.Marshal(prompt.Schema[0].Choices[1].Description); err != nil || string(got) != want[0][1] {
		t.Fatalf("structured choice description was not serialized: %s, %v", got, err)
	}
}

func TestCompileNullableNoulCriteria(t *testing.T) {
	for _, raw := range []string{
		`{"true":null,"false":{"what":"not true"}}`,
		`null`,
	} {
		var req Request
		body := `{"model":"nimble","state":"x","questions":{"q":{"type":"noul","instructions":"q","criteria":` + raw + `}}}`
		if err := json.Unmarshal([]byte(body), &req); err != nil {
			t.Fatal(err)
		}
		if _, err := Compile(req); err != nil {
			t.Fatalf("criteria %s: %v", raw, err)
		}
	}
}

func TestCompileUnsupportedEncoding(t *testing.T) {
	if _, err := CompileWithEncoder(testRequest(t), "unknown"); err == nil {
		t.Fatal("accepted unsupported decision encoding")
	}
}

func TestStructuredStateFrames(t *testing.T) {
	req := testRequest(t)
	req.State = json.RawMessage(`{"frames":["first","second"],"position":7}`)
	compiled, err := Compile(req)
	if err != nil {
		t.Fatal(err)
	}
	if err := compiled.Render(func(messages []api.Message) (string, error) {
		return messages[0].Content, nil
	}); err != nil {
		t.Fatal(err)
	}
	for _, row := range compiled.Request.Rows {
		if !strings.Contains(row.Prompt, `"context":"{\"frames\":[\"first\",\"second\"],\"position\":7}"`) {
			t.Fatalf("structured state was not preserved in the text prompt: %q", row.Prompt)
		}
	}
}

func TestRenderError(t *testing.T) {
	compiled, err := Compile(testRequest(t))
	if err != nil {
		t.Fatal(err)
	}
	want := errors.New("invalid model template")
	if err := compiled.Render(func([]api.Message) (string, error) { return "", want }); !errors.Is(err, want) {
		t.Fatalf("Render error = %v, want %v", err, want)
	}
}

func TestAnswers(t *testing.T) {
	req := testRequest(t)
	c, err := Compile(req)
	if err != nil {
		t.Fatal(err)
	}
	result, err := c.Answer("nimble", llm.ScoreResponse{
		Logits: [][]float32{{0, 0}, {-1000, 1000}, {1000, 1000, 1000}}, InputTokens: 900,
	})
	if err != nil {
		t.Fatal(err)
	}
	choice, _ := result.Answers.Get(c.fields[0].Name)
	got := choice.(ChoiceAnswer)
	if got.Choice != "billing" || got.Confidence != 0 {
		t.Fatalf("tie must choose the first candidate with zero concentration: %+v", got)
	}
	noul, _ := result.Answers.Get("refund")
	if noul.(NoulAnswer).Noul != 1 {
		t.Fatal("noul must report P(true), using stable softmax")
	}
	score, _ := result.Answers.Get("urgency")
	if math.Abs(score.(ScoreAnswer).Score-1) > 1e-12 || score.(ScoreAnswer).Legend.Len() != 3 {
		t.Fatalf("bad expected rubric index or legend: %+v", score)
	}
	if result.Usage.InputTokens != 900 || result.Usage.OutputTokens != 0 {
		t.Fatalf("bad direct-scoring usage: %+v", result.Usage)
	}
	for _, logits := range [][][]float32{nil, {{1}, {1, 2}, {1, 2, 3}}, {{float32(math.NaN()), 0}, {0, 1}, {1, 2, 3}}} {
		if _, err := c.Answer("nimble", llm.ScoreResponse{Logits: logits}); err == nil {
			t.Errorf("accepted malformed runner result: %v", logits)
		}
	}
}

func TestAnswerLogprobs(t *testing.T) {
	c, err := Compile(testRequest(t))
	if err != nil {
		t.Fatal(err)
	}
	logits := [][]float32{{0, 2}, {0, -2}, {0, 2, 4}}
	logprobs := [][]float32{{-10, -8}, {-10, -12}, {-10, -8, -6}}
	want, err := c.Answer("nimble", llm.ScoreResponse{Logits: logits, InputTokens: 123})
	if err != nil {
		t.Fatal(err)
	}
	got, err := c.Answer("nimble", llm.ScoreResponse{Logits: logprobs, InputTokens: 123, OutputTokens: 7})
	if err != nil {
		t.Fatal(err)
	}
	wantAnswers, _ := json.Marshal(want.Answers)
	gotAnswers, _ := json.Marshal(got.Answers)
	if string(wantAnswers) != string(gotAnswers) {
		t.Fatalf("row offsets changed candidate probabilities: %s != %s", gotAnswers, wantAnswers)
	}
	if got.Usage.InputTokens != 123 || got.Usage.OutputTokens != 7 || want.Usage.OutputTokens != 0 {
		t.Fatalf("incorrect backend usage: llama=%+v direct=%+v", got.Usage, want.Usage)
	}
}

func TestInvalidRequests(t *testing.T) {
	for _, data := range []string{
		`{}`, `{"model":"nimble","state":"x","questions":{}}`,
		`{"model":"nimble","state":" ","questions":{"x":{"type":"noul","instructions":"q"}}}`,
		`{"model":"nimble","state":null,"questions":{"x":{"type":"noul","instructions":"q"}}}`,
		`{"model":"nimble","state":"x","questions":{"x":{"type":"other","instructions":"q"}}}`,
		`{"model":"nimble","state":"x","questions":{"x":{"type":"noul"}}}`,
		`{"model":"nimble","state":"x","questions":{"":{"type":"noul","instructions":"q"}}}`,
		`{"model":"nimble","state":"x","questions":{"x":{"type":"choice","instructions":"q","criteria":{"a":"a"}}}}`,
		`{"model":"nimble","state":"x","questions":{"x":{"type":"score","instructions":"q","criteria":["a",null]}}}`,
		`{"model":"nimble","state":"x","questions":{"x":{"type":"noul","instructions":"q","criteria":{"true":42}}}}`,
		`{"model":"nimble","state":"x","questions":{"x":{"type":"noul","instructions":"q","criteria":{"yes":"Yes"}}}}`,
	} {
		t.Run(data, func(t *testing.T) {
			var req Request
			if err := json.Unmarshal([]byte(data), &req); err != nil {
				t.Fatal(err)
			}
			if _, err := Compile(req); err == nil {
				t.Fatal("accepted invalid request")
			}
		})
	}
	req := testRequest(t)
	q, _ := req.Questions.Get("urgency")
	q.Criteria = json.RawMessage(`["x"` + strings.Repeat(`,"x"`, 26) + `]`)
	req.Questions.Set("urgency", q)
	for _, encoding := range []string{"", "clef"} {
		if _, err := CompileWithEncoder(req, encoding); err == nil {
			t.Fatalf("encoding %q accepted 27 candidates", encoding)
		}
	}
}

func TestContent(t *testing.T) {
	for _, tt := range []struct{ input, want string }{
		{` "line\nquoted \"text\"" `, "line\nquoted \"text\""},
		{` {"z": 9007199254740993, "a": 1e999} `, `{"z":9007199254740993,"a":1e999}`},
		{` [1.0, 1e6, -0, "€"] `, `[1.0,1e6,-0,"€"]`},
	} {
		got, err := content(json.RawMessage(tt.input))
		if err != nil || got != tt.want {
			t.Errorf("content(%s) = %q, %v; want %q", tt.input, got, err, tt.want)
		}
	}
	for _, input := range []string{"", "null", "true", "42", `"unterminated`, `{"x":}`, `[1,]`} {
		if _, err := content(json.RawMessage(input)); err == nil {
			t.Errorf("accepted invalid content %q", input)
		}
	}
}
