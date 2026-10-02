//go:build integration && release

package integration

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"os"
	"slices"
	"strings"
	"testing"

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
	client, endpoint, cleanup := InitServerConnection(ctx, t)
	defer cleanup()
	loadCtx, cancel := context.WithTimeout(ctx, apiTestTimeout)
	defer cancel()
	requireCapability(loadCtx, t, client, modelName, model.CapabilityDecision)
	skipIfModelTooLargeForVRAM(loadCtx, t, client, modelName)
	if corpus.HasImages() {
		requireCapability(loadCtx, t, client, modelName, model.CapabilityVision)
	}
	endpoint = "http://" + endpoint + "/v1/systemone"
	score := func(t *testing.T, req decision.Request) decisiontest.Response {
		t.Helper()
		req.Model = modelName
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
				requestCtx, cancel := context.WithTimeout(t.Context(), apiTestTimeout)
				defer cancel()
				result, _, err := decisiontest.Do(requestCtx, endpoint, req)
				var wrong []string
				if err == nil {
					wrong = c.Mismatches(result)
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
	input.Questions = &decision.Questions{}
	for _, c := range corpus.Cases {
		for _, q := range c.Request.Questions.All() {
			if _, ok := input.Questions.Get(q.Type); !ok {
				input.Questions.Set(q.Type, q)
			}
		}
	}
	baseline := score(t, input)
	unchanged := func(t *testing.T) {
		t.Helper()
		got := score(t, input)
		// Cache/prefill shapes can shift BF16 probabilities slightly; choices
		// must stay identical while numbers retain the existing 0.01 tolerance.
		if diff := cmp.Diff(baseline.Answers, got.Answers, cmpopts.EquateApprox(0, .01)); diff != "" {
			t.Errorf("answers changed (-baseline +got): %s", diff)
		}
		if got.Usage.InputTokens != baseline.Usage.InputTokens {
			t.Errorf("input tokens changed: %d -> %d", baseline.Usage.InputTokens, got.Usage.InputTokens)
		}
	}
	t.Run("repeat", unchanged)
	t.Run("concurrent", func(t *testing.T) {
		for i := range 4 {
			t.Run(fmt.Sprint(i), func(t *testing.T) { t.Parallel(); unchanged(t) })
		}
	})
	t.Run("invalid_request", func(t *testing.T) {
		invalid := input
		invalid.Model = modelName
		invalid.Questions = nil
		requestCtx, cancel := context.WithTimeout(ctx, apiTestTimeout)
		defer cancel()
		_, _, err := decisiontest.Do(requestCtx, endpoint, invalid)
		var status api.StatusError
		if !errors.As(err, &status) || status.StatusCode != 400 {
			t.Fatalf("missing questions: got %v, want HTTP 400", err)
		}
		unchanged(t)
	})
}
