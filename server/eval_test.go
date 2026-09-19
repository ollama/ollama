package server

import (
	"context"
	"encoding/json"
	"errors"
	"math"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/types/model"
)

func TestEvalHandlerValidation(t *testing.T) {
	for _, body := range []string{
		`{`, `null`, `{}`, `[]`,
		`{"model":"m","state":"text","questions":{"q":{"type":"noul","instructions":"?"}}} {}`,
		`{"model":"m","state":"text","questions":{"q":{"type":"choice","instructions":"?"}}}`,
		`{"model":"m","state":"text","questions":{"q":{"type":"score","instructions":"?","criteria":["low"]}}}`,
		`{"model":"gliner","input":"text","labels":["person"]}`,
	} {
		w := httptest.NewRecorder()
		c, _ := gin.CreateTestContext(w)
		c.Request = httptest.NewRequest(http.MethodPost, "/api/eval", strings.NewReader(body))
		(&Server{}).EvalHandler(c) // no scheduler: validation must precede loading
		if w.Code != http.StatusUnprocessableEntity {
			t.Fatalf("%s: %d %s", body, w.Code, w.Body)
		}
	}
}

func TestEvalRejectsExtractionBeforeLoading(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())
	config, err := manifest.NewLayer(strings.NewReader(`{"model_format":"safetensors","capabilities":["extraction"]}`), "application/vnd.docker.container.image.v1+json")
	if err != nil {
		t.Fatal(err)
	}
	if err := manifest.WriteManifest(model.ParseName("gliner"), config, nil); err != nil {
		t.Fatal(err)
	}
	w := createRequest(t, (&Server{}).EvalHandler, api.EvalRequest{
		Model: "gliner", State: json.RawMessage(`"John works at Google."`),
		Questions: map[string]api.EvalQuestion{"person": {Type: "noul", Instructions: json.RawMessage(`"Is a person mentioned?"`)}},
	})
	if w.Code != http.StatusBadRequest || !strings.Contains(w.Body.String(), "/api/extract") {
		t.Fatalf("%d %s", w.Code, w.Body)
	}
}

func TestEvalHandlerAnswers(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())
	mock := &mockRunner{ChatFn: func(ctx context.Context, req llm.ChatRequest, fn func(llm.ChatResponse)) error {
		if req.Think == nil || req.Think.Bool() || req.Shift || len(req.Format) == 0 {
			t.Fatalf("unexpected generation settings: %+v", req)
		}
		if strings.Contains(req.Messages[1].Content, "question-key-") {
			t.Fatal("question IDs must not reach the model")
		}
		var schema struct {
			Properties struct {
				Probabilities struct {
					MinItems int `json:"minItems"`
				} `json:"probabilities"`
			} `json:"properties"`
		}
		if err := json.Unmarshal(req.Format, &schema); err != nil {
			t.Fatal(err)
		}
		distribution := `[0.1,0.9]`
		if schema.Properties.Probabilities.MinItems == 3 {
			distribution = `[0.05,0.3,0.65]`
		}
		fn(llm.ChatResponse{Message: api.Message{Content: `{"probabilities":`}})
		fn(llm.ChatResponse{Message: api.Message{Content: distribution + `}`}, Done: true, PromptEvalCount: 50, EvalCount: 10})
		return nil
	}}
	s := newServerWithMockRunner(t, mock)
	createMinimalGGUFModel(t, s, "eval-model", nil, "", nil)
	w := createRequest(t, s.EvalHandler, api.EvalRequest{
		Model: "eval-model", State: json.RawMessage(`{"message":"My payouts have been failing for three days."}`),
		Questions: map[string]api.EvalQuestion{
			"question-key-noul":   {Type: "noul", Instructions: json.RawMessage(`"Urgent?"`)},
			"question-key-choice": {Type: "choice", Instructions: json.RawMessage(`"Team?"`), Criteria: json.RawMessage(`{"billing":null,"technical":"Bugs"}`)},
			"question-key-score":  {Type: "score", Instructions: json.RawMessage(`"Frustration?"`), Criteria: json.RawMessage(`["Calm","Frustrated","Angry"]`)},
		},
	})
	if w.Code != http.StatusOK {
		t.Fatalf("%d %s", w.Code, w.Body)
	}
	var resp api.EvalResponse
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatal(err)
	}
	if len(resp.Answers) != 3 || resp.Model != "eval-model" || resp.Usage.InputTokens != 150 || resp.Usage.OutputTokens != 30 {
		t.Fatalf("response: %+v", resp)
	}
	if a := resp.Answers["question-key-noul"]; a.Noul == nil || *a.Noul != 0.9 || a.Confidence != nil {
		t.Fatalf("noul: %+v", a)
	}
	if a := resp.Answers["question-key-choice"]; a.Choice == nil || *a.Choice != "technical" || a.Probabilities["billing"] != 0.1 {
		t.Fatalf("choice: %+v", a)
	}
	if a := resp.Answers["question-key-score"]; a.Score == nil || math.Abs(*a.Score-1.6) > 1e-12 || a.Legend["2"] != "Angry" {
		t.Fatalf("score: %+v", a)
	}
}

func TestEvalAnswerRejectsInvalidModelOutput(t *testing.T) {
	for _, output := range []string{
		`oops`, `null`, `{}`, `{"probabilities":[1]}`, `{"probabilities":[0.1,0.2,0.7]}`,
		`{"probabilities":[null,1]}`, `{"probabilities":["0",1]}`,
		`{"probabilities":[-1,2]}`, `{"probabilities":[0,0]}`, `{"probabilities":[0.1,0.1]}`,
	} {
		if _, err := evalAnswer("choice", []string{"a", "b"}, []string{"", ""}, output); err == nil {
			t.Fatalf("accepted %s", output)
		}
	}
	for _, tc := range []struct {
		output string
		want   float64
	}{
		{`{"probabilities":[0.5,0.5]}`, 0},
		{`{"probabilities":[0,1]}`, 1},
	} {
		a, err := evalAnswer("choice", []string{"a", "b"}, []string{"", ""}, tc.output)
		if err != nil || a.Confidence == nil || math.Abs(*a.Confidence-tc.want) > 1e-12 {
			t.Fatalf("%s: %+v, %v", tc.output, a, err)
		}
	}
	a, err := evalAnswer("choice", []string{"a", "b", "c"}, nil, `{"probabilities":[0.333,0.333,0.333]}`)
	if err != nil || math.Abs(a.Probabilities["a"]+a.Probabilities["b"]+a.Probabilities["c"]-1) > 1e-12 {
		t.Fatalf("rounded distribution: %+v, %v", a, err)
	}
}

func TestEvalCompletionFailures(t *testing.T) {
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	opts := api.DefaultOptions()
	if _, err := evaluate(ctx, nil, nil, &opts, api.EvalRequest{Questions: map[string]api.EvalQuestion{"q": {Type: "noul"}}}); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancellation: %v", err)
	}
	for _, response := range []llm.ChatResponse{
		{Message: api.Message{Content: `{"probabilities":[0,1]}`}},
		{Done: true, DoneReason: llm.DoneReasonLength},
		{Done: true, DoneReason: llm.DoneReasonConnectionClosed},
	} {
		mock := &mockRunner{ChatResponse: response}
		_, _, err := evalCompletion(t.Context(), &Model{}, mock, &opts, []api.Message{{Role: "user", Content: "?"}}, nil)
		if err == nil {
			t.Fatalf("accepted incomplete generation: %+v", response)
		}
	}
}
