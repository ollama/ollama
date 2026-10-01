package llm

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
)

func TestLlamaServerReadout(t *testing.T) {
	for _, tc := range []struct {
		name      string
		tokens    int
		logits    string
		wantError bool
	}{
		{"complete", 3, `[[0,2]]`, false},
		{"truncated", 2, `[[0,2]]`, true},
		{"wrong options", 3, `[[0]]`, true},
		{"extra row", 3, `[[0,2],[1,0]]`, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			runner := newLlamaScoreTestRunner(t, func(http.ResponseWriter, *http.Request) { t.Error("must not generate tokens") })
			transport := runner.httpClient().Transport
			runner.client = &http.Client{Transport: readoutRoundTripper(func(req *http.Request) (*http.Response, error) {
				if req.URL.Path != "/embedding" {
					return transport.RoundTrip(req)
				}
				var body struct {
					Input   []int `json:"input"`
					Options int   `json:"score_options"`
				}
				if err := json.NewDecoder(req.Body).Decode(&body); err != nil {
					t.Fatal(err)
				}
				if len(body.Input) != 3 || body.Options != 2 {
					t.Fatalf("wrong readout request: %+v", body)
				}
				w := httptest.NewRecorder()
				fmt.Fprintf(w, `[{"logits":%s,"tokens_evaluated":%d}]`, tc.logits, tc.tokens)
				return w.Result(), nil
			})}
			result, err := runner.Score(t.Context(), ScoreRequest{Readout: true, MaxTokens: 3, Rows: []ScoreRow{{Prompt: "abc", Candidates: []string{"A", "B"}}}})
			if (err != nil) != tc.wantError {
				t.Fatalf("error = %v", err)
			}
			if !tc.wantError && (result.InputTokens != 3 || result.OutputTokens != 0 || result.Logits[0][1] != 2) {
				t.Fatal(result)
			}
		})
	}
}

func TestLlamaServerReadoutRejectsInvalidInput(t *testing.T) {
	runner := newLlamaScoreTestRunner(t, func(http.ResponseWriter, *http.Request) { t.Fatal("unexpected generation") })
	for _, input := range []ScoreRequest{
		{Readout: true, MaxTokens: 3, Rows: []ScoreRow{{Prompt: "long", Candidates: []string{"A"}}}},
		{Readout: true, MaxTokens: 3, Rows: []ScoreRow{{Prompt: "abc", Candidates: strings.Fields(strings.Repeat("A ", 256))}}},
		{Readout: true, MaxTokens: 3, Rows: []ScoreRow{{Prompt: "abc", Candidates: []string{"A"}}}, Images: []api.ImageData{[]byte("not an image")}},
	} {
		if _, err := runner.Score(t.Context(), input); err == nil {
			t.Fatal("accepted invalid input")
		}
	}
}

type readoutRoundTripper func(*http.Request) (*http.Response, error)

func (f readoutRoundTripper) RoundTrip(req *http.Request) (*http.Response, error) { return f(req) }
