package decision

import (
	"encoding/json"
	"math"
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llm"
)

func TestCompileClef(t *testing.T) {
	var req Request
	if err := json.Unmarshal([]byte(`{"model":"renamed-model","state":{"z":2,"a":"café & <x>"},"questions":{"team":{"type":"choice","instructions":"Route it","criteria":{"z":null,"a":"First"}},"yes":{"type":"noul","instructions":{"z":2,"a":1}},"rating":{"type":"score","instructions":"Rate it","criteria":["Low","High"]}}}`), &req); err != nil {
		t.Fatal(err)
	}
	c, err := CompileWithEncoder(req, "clef")
	if err != nil {
		t.Fatal(err)
	}
	if err := c.Render(func([]api.Message) (string, error) { t.Fatal("Clef must not use a chat template"); return "", nil }); err != nil {
		t.Fatal(err)
	}
	input := c.Request
	if len(input.Fields) != 3 || len(c.Request.Rows) != 0 {
		t.Fatal("expected one joint request")
	}
	if input.Segments[1] != `{"a":"café & <x>","z":2}` {
		t.Fatalf("state = %s", input.Segments[1])
	}
	option := func(f, i int) string {
		return strings.Join(input.Segments[input.Fields[f].Options[i][0]:input.Fields[f].Options[i][1]], "")
	}
	if option(0, 0) != `{"description":"First","option_id":"a"}` || option(0, 1) != `{"option_id":"z"}` {
		t.Fatalf("choice ordering / null semantics changed: %s, %s", option(0, 0), option(0, 1))
	}
	if option(1, 0) != `{"description":"The proposition is true or the answer is yes.","option_id":"true"}` {
		t.Fatal(option(1, 0))
	}
	if input.Fields[0].Type != 1 || input.Fields[1].Type != 0 || input.Fields[2].Type != 2 {
		t.Fatal("incorrect type ids")
	}
	prompt := strings.Join(input.Segments, "")
	if !strings.HasSuffix(prompt, "<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:") || !strings.Contains(prompt, "INSTRUCTION: {\"a\":1,\"z\":2}\nALLOWED OPTIONS:\n") {
		t.Fatal(prompt)
	}
	response, err := c.Answer(req.Model, llm.ScoreResponse{Logits: [][]float32{{2, 0}, {2, 0}, {0, 2}}, InputTokens: 99})
	if err != nil {
		t.Fatal(err)
	}
	a, _ := response.Answers.Get("team")
	choice := a.(ChoiceAnswer)
	n, _ := response.Answers.Get("yes")
	yes := n.(NoulAnswer)
	if choice.Choice != "a" || math.Abs(choice.Confidence-0.4729346589968384) > 1e-8 || yes.Noul < 0.88 || response.Usage.OutputTokens != 0 {
		t.Fatalf("incorrect joint answer: %+v %+v", choice, yes)
	}
}

func TestCompileClefImages(t *testing.T) {
	req := testRequest(t)
	req.Images = []api.ImageData{[]byte("first image"), []byte("second image")}
	c, err := CompileWithEncoder(req, "clef")
	if err != nil {
		t.Fatal(err)
	}
	if c.Request.ImagePosition != 1 || len(c.Request.Images) != 2 || string(c.Request.Images[1]) != "second image" {
		t.Fatalf("images were not preserved before state: %+v", c.Request)
	}
	if _, err := Compile(req); err == nil {
		t.Fatal("candidate scoring silently ignored images")
	}
}

func TestClefCanonicalJSON(t *testing.T) {
	for _, tt := range []struct{ input, want string }{
		{`{"v":1e6}`, `{"v":1000000.0}`},
		{`{"v":-0}`, `{"v":0}`},
		{`{"v":1e-5}`, `{"v":1e-05}`},
		{`[1.0,-0.0,1e15,1e16,1e-4,1e-999,1e999,-1e999]`, `[1.0,-0.0,1000000000000000.0,1e+16,0.0001,0.0,Infinity,-Infinity]`},
		{`{"z":9007199254740993,"a":[true,false,null]}`, `{"a":[true,false,null],"z":9007199254740993}`},
		{`{"v":"\u2028\u2029"}`, "{\"v\":\"  \"}"},
		{`{"v":"\\u2028\\u2029"}`, `{"v":"\\u2028\\u2029"}`},
		{`{"v":"\\\u2028"}`, "{\"v\":\"\\\\ \"}"},
		{`{"v":"\u0000\b\f\n\r\t\"\\\u007f<>&café"}`, "{\"v\":\"\\u0000\\b\\f\\n\\r\\t\\\"\\\\\x7f<>&café\"}"},
		{`null`, `null`},
		{`true`, `true`},
		{`"plain text"`, `plain text`},
	} {
		t.Run(tt.input, func(t *testing.T) {
			got, err := clefContent(json.RawMessage(tt.input))
			if err != nil || got != tt.want {
				t.Fatalf("got %q, %v; want %q", got, err, tt.want)
			}
		})
	}
	for _, input := range []string{"", `{`, `null true`} {
		if _, err := clefContent(json.RawMessage(input)); err == nil {
			t.Fatalf("accepted invalid JSON %q", input)
		}
	}
}

func TestClefSchema(t *testing.T) {
	var req Request
	if err := json.Unmarshal([]byte(`{"model":"clef","state":null,"questions":{"team":{"type":"choice","criteria":{"z":null,"a":{"text":"First","weight":1e6}}},"yes":{"type":"noul","instructions":null,"criteria":{"true":null}},"rating":{"type":"score","instructions":"","criteria":[null,{"text":"High"}]}}}`), &req); err != nil {
		t.Fatal(err)
	}
	c, err := CompileWithEncoder(req, "clef")
	if err != nil {
		t.Fatal(err)
	}
	if c.Request.Segments[1] != "null" {
		t.Fatal("lost null state")
	}
	for i, name := range []string{"team", "yes", "rating"} {
		if got := c.Request.Segments[c.Request.Fields[i].Question[0]]; got != name {
			t.Fatalf("fallback instructions = %q, want %q", got, name)
		}
	}
	for _, tt := range []struct {
		field, option int
		want          string
	}{
		{0, 0, `{"description":{"text":"First","weight":1000000.0},"option_id":"a"}`},
		{0, 1, `{"option_id":"z"}`},
		{1, 0, `{"option_id":"true"}`},
		{2, 0, `{"option_id":"0"}`},
		{2, 1, `{"description":{"text":"High"},"option_id":"1"}`},
	} {
		span := c.Request.Fields[tt.field].Options[tt.option]
		if got := c.Request.Segments[span[0]]; got != tt.want {
			t.Fatalf("option = %q, want %q", got, tt.want)
		}
	}
	answer, err := c.Answer(req.Model, llm.ScoreResponse{Logits: [][]float32{{2, 0}, {2, 0}, {0, 2}}})
	if err != nil {
		t.Fatal(err)
	}
	rating, _ := answer.Answers.Get("rating")
	legend, err := json.Marshal(rating.(ScoreAnswer).Legend)
	if err != nil || string(legend) != `{"0":null,"1":{"text":"High"}}` {
		t.Fatalf("legend = %s, %v", legend, err)
	}
	if _, err := Compile(req); err == nil {
		t.Fatal("broadened the candidate-scoring schema")
	}
}

func TestClefValidation(t *testing.T) {
	for _, raw := range []string{
		`{}`, `{"model":"clef","state":null,"questions":{}}`,
		`{"model":"clef","questions":{"q":{"type":"noul"}}}`,
		`{"model":"clef","state":null,"questions":{"":{"type":"noul"}}}`,
		`{"model":"clef","state":null,"questions":{"q":{"type":"other"}}}`,
		`{"model":"clef","state":null,"questions":{"q":{"type":"choice","criteria":{"a":null}}}}`,
		`{"model":"clef","state":null,"questions":{"q":{"type":"score","criteria":{}}}}`,
		`{"model":"clef","state":null,"questions":{"q":{"type":"noul","criteria":[true,false]}}}`,
		`{"model":"clef","state":null,"questions":{"q":{"type":"noul","criteria":{"yes":null}}}}`,
	} {
		var req Request
		if err := json.Unmarshal([]byte(raw), &req); err != nil {
			t.Fatal(err)
		}
		if _, err := CompileWithEncoder(req, "clef"); err == nil {
			t.Fatalf("accepted invalid request %s", raw)
		}
	}
}

func TestDecisionRejectsVideo(t *testing.T) {
	for _, encoding := range []string{"", "clef"} {
		req := testRequest(t)
		req.Videos = []json.RawMessage{json.RawMessage(`"video.mp4"`)}
		if _, err := CompileWithEncoder(req, encoding); err == nil || !strings.Contains(err.Error(), "video") {
			t.Fatalf("encoding %q silently accepted video: %v", encoding, err)
		}
		req.Videos = nil
		if _, err := CompileWithEncoder(req, encoding); err != nil {
			t.Fatal(err)
		}
	}
}
