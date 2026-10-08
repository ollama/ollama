//go:build integration && (release || migration)

package integration

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"maps"
	"math"
	"net/http"
	"os"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/decision"
	"github.com/ollama/ollama/internal/decisiontest"
	"github.com/ollama/ollama/types/model"
)

func runAPISystemOne(t *testing.T, modelName string) {
	path := os.Getenv("OLLAMA_TEST_DECISION_FILE")
	if path == "" {
		path = "../.cache/decision-data/regression.jsonl"
	}
	runDecisionCorpus(t, modelName, path)
}

func runAPISystemOneVision(t *testing.T, modelName string) {
	abbeyRoad, docs, _ := decodeTestImages(t)
	var cases []decisiontest.Case
	for _, fixture := range []struct {
		name   string
		images []api.ImageData
		choice string
		laptop bool
	}{
		{"road", []api.ImageData{abbeyRoad}, "crossing", false},
		{"desk", []api.ImageData{docs}, "working", true},
		{"road_then_desk", []api.ImageData{abbeyRoad, docs}, "crossing", false},
		{"desk_then_road", []api.ImageData{docs, abbeyRoad}, "working", true},
	} {
		var req decision.Request
		if err := json.Unmarshal([]byte(`{
			"state":"Inspect the first attached image.",
			"questions":{
				"activity":{"type":"choice","instructions":"What are the animals doing in the first image?","criteria":{"crossing":"Walking across a road","working":"Working or resting at a desk"}},
				"laptop":{"type":"noul","instructions":"Is there a laptop computer in the first image?","criteria":{}}
			}
		}`), &req); err != nil {
			t.Fatal(err)
		}
		req.Images = fixture.images
		cases = append(cases, decisiontest.Case{ID: fixture.name, Dataset: "ollama-vision", Request: req, Expected: map[string]decisiontest.Expected{
			"activity": {Choice: &fixture.choice}, "laptop": {Noul: &fixture.laptop},
		}})
	}
	runDecisionCases(t, modelName, decisiontest.Corpus{Cases: cases})
}

func runDecisionCorpus(t *testing.T, modelName, path string) {
	if _, err := os.Stat(path); errors.Is(err, os.ErrNotExist) {
		t.Skipf("decision corpus unavailable: %s", path)
	} else if err != nil {
		t.Fatal(err)
	}
	corpus, err := decisiontest.Load(path)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("decision corpus=%s sha256=%s", path, corpus.SHA256)
	runDecisionCases(t, modelName, corpus)
}

func runDecisionCases(t *testing.T, modelName string, corpus decisiontest.Corpus) {
	ctx := t.Context()
	client, _, cleanup := InitServerConnection(ctx, t)
	defer cleanup()
	loadCtx, cancel := context.WithTimeout(ctx, apiTestTimeout)
	defer cancel()
	requireCapability(loadCtx, t, client, modelName, model.CapabilityDecision)
	skipIfModelTooLargeForVRAM(loadCtx, t, client, modelName)
	if corpus.HasImages() {
		requireCapability(loadCtx, t, client, modelName, model.CapabilityVision)
	}
	validateDecisionCases(ctx, t, client, modelName, corpus)
}

func validateDecisionCases(ctx context.Context, t *testing.T, client *api.Client, modelName string, corpus decisiontest.Corpus) {
	t.Helper()
	_, endpoint := GetTestEndpoint()
	endpoint = "http://" + endpoint + "/v1/systemone"
	keepAlive := &api.Duration{Duration: 10 * time.Second}
	score := func(t *testing.T, req decision.Request) decisiontest.Response {
		t.Helper()
		req.Model = modelName
		req.KeepAlive = keepAlive
		requestCtx, cancel := context.WithTimeout(ctx, apiTestTimeout)
		defer cancel()
		result, _, err := decisiontest.Do(requestCtx, endpoint, req)
		if err != nil {
			t.Fatal(err)
		}
		return result
	}
	var tally decisiontest.Tally
	datasets := make(map[string]decisiontest.Tally)
	t.Logf("decision regression cases=%d", len(corpus.Cases))
	t.Run("corpus", func(t *testing.T) {
		for _, c := range corpus.Cases {
			t.Run(c.ID, func(t *testing.T) {
				req := c.Request
				req.Model = modelName
				req.KeepAlive = keepAlive
				requestCtx, cancel := context.WithTimeout(ctx, apiTestTimeout)
				defer cancel()
				result, _, err := decisiontest.Do(requestCtx, endpoint, req)
				var wrong []string
				if err == nil {
					err = c.CheckCache(result)
					if err == nil {
						wrong = c.Mismatches(result)
					}
				}
				tally.Add(c, wrong, err)
				d := datasets[c.Dataset]
				d.Add(c, wrong, err)
				datasets[c.Dataset] = d
				if err != nil {
					t.Fatal(err)
				}
				if len(wrong) == 0 {
					return
				}
				t.Error(strings.Join(wrong, "; "))
			})
		}
	})
	t.Logf("decision score: %s, expected cases=%d", tally, len(corpus.Cases))
	for _, name := range slices.Sorted(maps.Keys(datasets)) {
		t.Logf("decision dataset=%s: %s", name, datasets[name])
	}
	if t.Failed() {
		return
	}
	input := corpus.Cases[0].Request
	input.Model = modelName
	input.KeepAlive = keepAlive
	input.Questions = &decision.Questions{}
	for _, c := range corpus.Cases {
		for _, q := range c.Request.Questions.All() {
			if _, ok := input.Questions.Get(q.Type); !ok {
				input.Questions.Set(q.Type, q)
			}
		}
	}
	baseline := score(t, input)
	checkUnchanged := func(t *testing.T, got decisiontest.Response) {
		t.Helper()
		// Cache/prefill shapes can shift BF16 probabilities slightly; choices
		// must stay identical while numbers retain the existing 0.01 tolerance.
		if diff := cmp.Diff(baseline.Answers, got.Answers, cmpopts.EquateApprox(0, .01)); diff != "" {
			t.Errorf("answers changed (-baseline +got): %s", diff)
		}
		if got.Usage.InputTokens != baseline.Usage.InputTokens {
			t.Errorf("input tokens changed: %d -> %d", baseline.Usage.InputTokens, got.Usage.InputTokens)
		}
	}
	unchanged := func(t *testing.T) {
		t.Helper()
		checkUnchanged(t, score(t, input))
	}
	t.Run("repeat", unchanged)
	t.Run("concurrent", func(t *testing.T) {
		// The managed harness uses t.Setenv, which forbids parallel subtests.
		type result struct {
			response decisiontest.Response
			err      error
		}
		results := make(chan result, 4)
		for range cap(results) {
			go func() {
				requestCtx, cancel := context.WithTimeout(ctx, apiTestTimeout)
				defer cancel()
				response, _, err := decisiontest.Do(requestCtx, endpoint, input)
				results <- result{response, err}
			}()
		}
		for range cap(results) {
			got := <-results
			if got.err != nil {
				t.Error(got.err)
				continue
			}
			checkUnchanged(t, got.response)
		}
	})
	t.Run("after_generation", func(t *testing.T) {
		info, err := client.Show(ctx, &api.ShowRequest{Model: modelName})
		if err != nil {
			t.Fatal(err)
		}
		noStream := false
		err = client.Generate(ctx, &api.GenerateRequest{
			Model: modelName, Prompt: "Reply with one word: hello", Stream: &noStream,
			KeepAlive: keepAlive,
			Think:     &api.ThinkValue{Value: false},
			Options:   map[string]any{"temperature": 0, "num_predict": 8},
		}, func(api.GenerateResponse) error { return nil })
		if !slices.Contains(info.Capabilities, model.CapabilityCompletion) {
			var status api.StatusError
			if !errors.As(err, &status) || status.StatusCode != 400 {
				t.Fatalf("decision-only generation: got %v, want HTTP 400", err)
			}
		} else if err != nil {
			t.Fatal(err)
		}
		unchanged(t)
	})
	for _, tc := range []struct {
		name      string
		questions *decision.Questions
	}{
		{"invalid_request", nil},
		{"invalid_questions", &decision.Questions{}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			invalid := input
			invalid.Questions = tc.questions
			requestCtx, cancel := context.WithTimeout(ctx, apiTestTimeout)
			defer cancel()
			_, _, err := decisiontest.Do(requestCtx, endpoint, invalid)
			var status api.StatusError
			if !errors.As(err, &status) || status.StatusCode != http.StatusBadRequest {
				t.Fatalf("missing or empty questions: got %v, want HTTP 400", err)
			}
			unchanged(t)
		})
	}
	t.Run("invalid_state", func(t *testing.T) {
		info, err := client.Show(ctx, &api.ShowRequest{Model: modelName})
		if err != nil {
			t.Fatal(err)
		}
		invalid := map[string]any{"model": modelName, "questions": input.Questions}
		if info.Details.Family != "kev" {
			// Kev accepts empty and null state; only absence is invalid.
			invalid["state"] = ""
		}
		body, err := json.Marshal(invalid)
		if err != nil {
			t.Fatal(err)
		}
		requestCtx, cancel := context.WithTimeout(ctx, apiTestTimeout)
		defer cancel()
		req, err := http.NewRequestWithContext(requestCtx, http.MethodPost, endpoint, bytes.NewReader(body))
		if err != nil {
			t.Fatal(err)
		}
		req.Header.Set("Content-Type", "application/json")
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			t.Fatal(err)
		}
		defer resp.Body.Close()
		body, err = io.ReadAll(io.LimitReader(resp.Body, 1<<20))
		if err != nil || resp.StatusCode != http.StatusBadRequest {
			t.Fatalf("invalid state: HTTP %d: %s: %v", resp.StatusCode, body, err)
		}
		unchanged(t)
	})
}

// Migration must not skip inference when the optional release corpus is absent.
func validateSystemOne(ctx context.Context, t *testing.T, client *api.Client, modelName string, info *api.ShowResponse) {
	t.Helper()
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
	eligible, ineligible, department := true, false, "returns"
	minDamage, maxDamage := 0.0, math.Nextafter(0.5, 0)
	late := input
	late.State = json.RawMessage(`"Returns are allowed within 30 days. This item was bought 90 days ago. It arrived undamaged."`)
	corpus := decisiontest.Corpus{Cases: []decisiontest.Case{
		{ID: "in_window", Request: input, Expected: map[string]decisiontest.Expected{
			"eligible": {Noul: &eligible}, "department": {Choice: &department},
			"damage": {Min: &minDamage, Max: &maxDamage},
		}},
		{ID: "late_return", Request: late, Expected: map[string]decisiontest.Expected{
			"eligible": {Noul: &ineligible},
		}},
	}}
	if slices.Contains(info.Capabilities, model.CapabilityVision) {
		image, _, _ := decodeTestImages(t)
		questions := &decision.Questions{}
		questions.Set("word", decision.Question{
			Type: "choice", Instructions: json.RawMessage(`"Which word appears in the image?"`),
			Criteria: json.RawMessage(`{"ollama":"OLLAMA","nasa":"NASA"}`),
		})
		word := "ollama"
		corpus.Cases = append(corpus.Cases, decisiontest.Case{
			ID: "vision", Request: decision.Request{
				State: json.RawMessage(`"Read the attached image."`), Images: []api.ImageData{image}, Questions: questions,
			},
			Expected: map[string]decisiontest.Expected{"word": {Choice: &word}},
		})
	}
	validateDecisionCases(ctx, t, client, modelName, corpus)
}
