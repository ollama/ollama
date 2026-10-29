package decision

import (
	"encoding/json"
	"fmt"
	"math"
	"slices"
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llm"
)

func TestKevRequestAndAnswers(t *testing.T) {
	var req Request
	input := `{"model":"kev:test","state":{"subject":"Charged twice","details":["café",{"count":1e6,"urgent":true}]},"questions":{"team":{"type":"choice","instructions":{"goal":"Pick a team"},"criteria":{"billing":"Payments","shipping":null}},"urgent":{"type":"noul","instructions":"Urgent?","criteria":{"true":"escalate"}},"level":{"type":"score","instructions":"How urgent?","criteria":["low",null,{"high":true}]}}}`
	if err := json.Unmarshal([]byte(input), &req); err != nil {
		t.Fatal(err)
	}
	c, err := Compile(req, "kev")
	if err != nil {
		t.Fatal(err)
	}
	if err := c.Render(func([]api.Message) (string, error) { t.Fatal("Kev does not use a chat renderer"); return "", nil }); err != nil {
		t.Fatal(err)
	}
	if want := "subject: Charged twice\ndetails:\n  - café\n  - count: 1000000.0\n    urgent: True"; c.Request.State != want {
		t.Fatalf("state = %q, want %q", c.Request.State, want)
	}
	if got := c.Request.Rows[0].Question; got.Instructions != "goal: Pick a team" || !slices.Equal(got.Options, []string{"billing: Payments", "shipping"}) {
		t.Fatalf("choice = %+v", got)
	}
	if got := c.Request.Rows[1].Question.Options; !slices.Equal(got, []string{"no", "yes: escalate"}) {
		t.Fatalf("noul options = %v", got)
	}
	if got := c.Request.Rows[2].Question.Options; !slices.Equal(got, []string{"low", "", "high: True"}) {
		t.Fatalf("score options = %v", got)
	}
	answer, err := c.Answer(req.Model, llm.ScoreResponse{Logits: [][]float32{{0, 2}, {0, 2}, {0, 1, 0}}, InputTokens: 123})
	if err != nil {
		t.Fatal(err)
	}
	team, _ := answer.Answers.Get("team")
	choice := team.(ChoiceAnswer)
	if choice.Choice != "shipping" || math.Abs(choice.Confidence-0.7616) > 1e-4 {
		t.Fatalf("choice answer = %+v", choice)
	}
	urgent, _ := answer.Answers.Get("urgent")
	if urgent.(NoulAnswer).Noul != 0.8808 {
		t.Fatalf("noul answer = %+v", urgent)
	}
	level, _ := answer.Answers.Get("level")
	score := level.(ScoreAnswer)
	if score.Score != 1 || math.Abs(score.Confidence-0.3642) > 1e-4 {
		t.Fatalf("score answer = %+v", score)
	}
	if answer.Usage.InputTokens != 123 {
		t.Fatal(answer.Usage)
	}
}

func TestKevLimits(t *testing.T) {
	for _, state := range []string{`null`, `""`, `[]`} {
		t.Run("state "+state, func(t *testing.T) {
			var req Request
			if err := json.Unmarshal([]byte(`{"model":"kev","state":`+state+`,"questions":{"q":{"type":"choice","criteria":{"a":null}}}}`), &req); err != nil {
				t.Fatal(err)
			}
			if _, err := Compile(req, "kev"); err != nil {
				t.Fatalf("Compile() error = %v, want acceptance of state %s", err, state)
			}
		})
	}
	for _, count := range []int{1, 255, 256} {
		t.Run(fmt.Sprintf("%d options", count), func(t *testing.T) {
			var req Request
			criteria := make([]string, count)
			for i := range criteria {
				criteria[i] = `"x"`
			}
			input := `{"model":"kev","state":"x","questions":{"q":{"type":"score","criteria":[` + strings.Join(criteria, ",") + `]}}}`
			if err := json.Unmarshal([]byte(input), &req); err != nil {
				t.Fatal(err)
			}
			_, err := Compile(req, "kev")
			if (err == nil) != (count <= 255) {
				t.Fatalf("Compile() error = %v for %d options", err, count)
			}
		})
	}
	for _, tc := range []struct {
		name, input, want string
	}{
		{"blank question", `{"model":"kev","state":"x","questions":{" ":{"type":"noul"}}}`, "field name must not be empty"},
		{"blank choice key", `{"model":"kev","state":"x","questions":{"q":{"type":"choice","criteria":{" ":null}}}}`, "choice keys must not be empty"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var req Request
			if err := json.Unmarshal([]byte(tc.input), &req); err != nil {
				t.Fatal(err)
			}
			if _, err := Compile(req, "kev"); err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("Compile() error = %v, want %q", err, tc.want)
			}
		})
	}
}

func FuzzKevContent(f *testing.F) {
	for _, seed := range []string{``, `"text"`, `{"subject":"café","details":[1e6,true,null]}`, `[]`, `{"x":[[[0]]]}`, `null`, `{"x":`} {
		f.Add(seed)
	}
	f.Fuzz(func(t *testing.T, input string) {
		if len(input) > 64<<10 {
			t.Skip()
		}
		_, err := kevContent(json.RawMessage(input))
		if err == nil && input != "" && !json.Valid([]byte(input)) {
			t.Fatalf("accepted invalid JSON: %q", input)
		}
	})
}
