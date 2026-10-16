package decisiontest

import (
	"encoding/json"
	"testing"

	"github.com/ollama/ollama/decision"
	"github.com/ollama/ollama/llm"
)

func TestCacheExpectation(t *testing.T) {
	zero, half, one := 0.0, 0.5, 1.0
	full, partial := 100, 1
	for _, tt := range []struct {
		name        string
		expectation *CacheExpectation
		cached      *int
		wantErr     bool
	}{
		{"unreported without requirement", nil, nil, false},
		{"missing telemetry", &CacheExpectation{Min: &half}, nil, true},
		{"broken reuse", &CacheExpectation{Min: &half}, new(int), true},
		{"full hit", &CacheExpectation{Min: &one}, &full, false},
		{"unexpected hit", &CacheExpectation{Max: &zero}, &partial, true},
	} {
		t.Run(tt.name, func(t *testing.T) {
			c := Case{Cache: tt.expectation}
			response := Response{PromptEvalCachedCount: tt.cached}
			response.Usage.InputTokens = 100
			if err := c.CheckCache(response); (err != nil) != tt.wantErr {
				t.Fatalf("CheckCache=%v, wantErr=%v", err, tt.wantErr)
			}
		})
	}
}

// Exercise actual API responses so the test harness does not impose one model's
// confidence formula or output precision on other decision models.
func TestResponseEncodings(t *testing.T) {
	var req decision.Request
	if err := json.Unmarshal([]byte(`{"model":"test","state":"An ordinary message.","questions":{
        "topic":{"type":"choice","instructions":"Classify the message.","criteria":{"a":"A","b":"B","c":"C"}},
        "rating":{"type":"score","instructions":"Rate the message.","criteria":["Low","Medium","High"]},
        "message":{"type":"noul","instructions":"Is this a message?","criteria":{}}
    }}`), &req); err != nil {
		t.Fatal(err)
	}
	for _, encoding := range []string{"", "tev1", "clef", "strands", "kev"} {
		t.Run(encoding, func(t *testing.T) {
			compiled, err := decision.Compile(req, encoding)
			if err != nil {
				t.Fatal(err)
			}
			for _, logits := range [][][]float32{
				{{0, 0, 0}, {0, 0, 0}, {0, 0}},
				{{1, 2, 3}, {1, 2, 3}, {1, 2}},
			} {
				response, err := compiled.Answer(req.Model, llm.ScoreResponse{InputTokens: 42, Logits: logits})
				if err != nil {
					t.Fatal(err)
				}
				body, err := json.Marshal(response)
				if err != nil {
					t.Fatal(err)
				}
				if _, err := decode(req, body); err != nil {
					t.Fatalf("rejected valid API response: %v: %s", err, body)
				}
			}
		})
	}
}
