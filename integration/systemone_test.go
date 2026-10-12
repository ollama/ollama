//go:build integration && release

package integration

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"net/http"
	"slices"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/decision"
	"github.com/ollama/ollama/types/model"
)

func postSystemOne(ctx context.Context, endpoint string, input decision.Request) ([]byte, int, error) {
	body, err := json.Marshal(input)
	if err != nil {
		return nil, 0, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, "http://"+endpoint+"/v1/systemone", bytes.NewReader(body))
	if err != nil {
		return nil, 0, err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, 0, err
	}
	defer resp.Body.Close()
	body, err = io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	return body, resp.StatusCode, err
}

func runAPISystemOne(t *testing.T, modelName string) {
	ctx, cancel := context.WithTimeout(t.Context(), apiTestTimeout)
	defer cancel()
	client, endpoint, cleanup := InitServerConnection(ctx, t)
	defer cleanup()
	if err := PullIfMissing(ctx, client, modelName); err != nil {
		t.Fatal(err)
	}
	info, err := client.Show(ctx, &api.ShowRequest{Model: modelName})
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Contains(info.Capabilities, model.CapabilityDecision) {
		t.Fatalf("model %q does not declare the decision capability", modelName)
	}

	var input decision.Request
	if err := json.Unmarshal([]byte(`{
		"state":"Returns are allowed within 30 days. This item was bought 12 days ago. It arrived undamaged.",
		"questions":{
			"eligible":{"type":"noul","instructions":"Is this item within the store return window?","criteria":{}},
			"department":{"type":"choice","instructions":"Which department handles a return?","criteria":{"returns":"Product returns and refunds","technical":"Software installation"}},
			"damage":{"type":"score","instructions":"How damaged was the item?","criteria":["Undamaged","Minor damage","Destroyed"]}
		}
	}`), &input); err != nil {
		t.Fatal(err)
	}
	input.Model = modelName
	type response struct {
		Model   string
		Answers struct {
			Eligible   decision.NoulAnswer
			Department decision.ChoiceAnswer
			Damage     decision.ScoreAnswer
		}
		Usage decision.Usage
	}
	score := func(t *testing.T, req decision.Request) response {
		t.Helper()
		body, status, err := postSystemOne(ctx, endpoint, req)
		if err != nil {
			t.Fatal(err)
		}
		if status != http.StatusOK {
			t.Fatalf("System One returned HTTP %d: %s", status, body)
		}
		// Missing or null numbers must not pass as legitimate zero values.
		result := response{Usage: decision.Usage{InputTokens: -1, OutputTokens: -1}}
		result.Answers.Eligible.Noul = math.NaN()
		result.Answers.Department.Confidence = math.NaN()
		result.Answers.Damage = decision.ScoreAnswer{Score: math.NaN(), Confidence: math.NaN()}
		if err := json.Unmarshal(body, &result); err != nil {
			t.Fatalf("decode System One response: %v: %s", err, body)
		}
		if result.Model != modelName {
			t.Fatalf("unexpected model or answer fields: %s", body)
		}
		// MLX scores without generating; llama.cpp can report primer/answer
		// tokens. Neither backend may omit input accounting or report negatives.
		if result.Usage.InputTokens <= 0 || result.Usage.OutputTokens < 0 {
			t.Fatalf("missing or invalid usage: %s", body)
		}
		answers := result.Answers
		if answers.Eligible.Type != "noul" || answers.Department.Type != "choice" || answers.Damage.Type != "score" {
			t.Fatalf("unexpected answer types: %s", body)
		}
		for _, answer := range []struct {
			name          string
			probabilities *decision.Probabilities
			confidence    float64
			keys          []string
		}{
			{"department", answers.Department.Probabilities, answers.Department.Confidence, []string{"returns", "technical"}},
			{"damage", answers.Damage.Probabilities, answers.Damage.Confidence, []string{"0", "1", "2"}},
		} {
			if answer.probabilities.Len() != len(answer.keys) {
				t.Fatalf("%s probabilities: got %v, want keys %v", answer.name, answer.probabilities, answer.keys)
			}
			var sum float64
			for _, key := range answer.keys {
				p, ok := answer.probabilities.Get(key)
				if !ok || !(p >= 0 && p <= 1) {
					t.Fatalf("%s probability %q missing or invalid: %s", answer.name, key, body)
				}
				sum += p
			}
			if math.Abs(sum-1) > 1e-6 || !(answer.confidence >= 0 && answer.confidence <= 1) {
				t.Fatalf("%s: probability sum=%v or missing/invalid confidence: %s", answer.name, sum, body)
			}
		}
		if p := answers.Eligible.Noul; !(p >= 0 && p <= 1) {
			t.Fatalf("missing or invalid eligible probability: %s", body)
		}
		if value := answers.Damage.Score; !(value >= 0 && value <= 2) {
			t.Fatalf("missing or invalid damage score: %s", body)
		}
		if diff := cmp.Diff(map[string]any{"0": "Undamaged", "1": "Minor damage", "2": "Destroyed"}, answers.Damage.Legend.ToMap()); diff != "" {
			t.Fatalf("damage legend (-want +got):\n%s", diff)
		}
		return result
	}

	baseline := score(t, input)
	t.Run("decisions", func(t *testing.T) {
		if baseline.Answers.Eligible.Noul <= 0.5 || baseline.Answers.Department.Choice != "returns" || baseline.Answers.Damage.Score >= 0.5 {
			t.Fatalf("unexpected decisions for an undamaged, in-window return: %+v", baseline.Answers)
		}
		late := input
		late.State = json.RawMessage(`"Returns are allowed within 30 days. This item was bought 90 days ago. It arrived undamaged."`)
		if result := score(t, late); result.Answers.Eligible.Noul >= 0.5 {
			t.Fatalf("late return should be ineligible: %+v", result.Answers.Eligible)
		}
	})

	unchanged := func(t *testing.T) {
		t.Helper()
		got := score(t, input)
		// Allow 0.01 absolute probability drift from different BF16 cache/prefill
		// shapes while requiring identical choices.
		if diff := cmp.Diff(baseline.Answers, got.Answers, cmpopts.EquateApprox(0, 0.01),
			cmpopts.AcyclicTransformer("probabilities", (*decision.Probabilities).ToMap),
			cmpopts.AcyclicTransformer("legend", (*decision.Legend).ToMap),
		); diff != "" {
			t.Errorf("scoring changed (-baseline +got):\n%s", diff)
		}
		if got.Usage.InputTokens != baseline.Usage.InputTokens {
			t.Errorf("input token count changed: got %d, want %d", got.Usage.InputTokens, baseline.Usage.InputTokens)
		}
	}
	t.Run("repeat", unchanged)
	t.Run("concurrent", func(t *testing.T) {
		for i := range 4 {
			t.Run(fmt.Sprint(i), func(t *testing.T) {
				t.Parallel()
				unchanged(t)
			})
		}
	})
	t.Run("after_generation", func(t *testing.T) {
		noStream := false
		err := client.Generate(ctx, &api.GenerateRequest{
			Model: modelName, Prompt: "Reply with one word: hello", Stream: &noStream,
			Think:   &api.ThinkValue{Value: false},
			Options: map[string]any{"temperature": 0, "num_predict": 8},
		}, func(api.GenerateResponse) error { return nil })
		if !slices.Contains(info.Capabilities, model.CapabilityCompletion) {
			var status api.StatusError
			if !errors.As(err, &status) || status.StatusCode != http.StatusBadRequest {
				t.Fatalf("decision-only generation: got %v, want HTTP 400", err)
			}
		} else if err != nil {
			t.Fatal(err)
		}
		unchanged(t)
	})

	t.Run("invalid_state", func(t *testing.T) {
		invalid := input
		invalid.State = json.RawMessage(`""`)
		body, status, err := postSystemOne(ctx, endpoint, invalid)
		if err != nil {
			t.Fatal(err)
		}
		if status != http.StatusBadRequest {
			t.Fatalf("got HTTP %d, want %d: %s", status, http.StatusBadRequest, body)
		}
		unchanged(t)
	})
}
