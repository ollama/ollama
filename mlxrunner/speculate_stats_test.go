package mlxrunner

import (
	"context"
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlx/mlxtest"
	sampler "github.com/ollama/ollama/mlxrunner/sample"
)

// endRound with fixedDepth skips the depth controller, so these sessions
// exercise only the stats accounting. resume's drained plain round mutates
// iterations directly in speculate.go and is not reproducible here without
// an inner decoder; its accounting is covered by the decode-level tests.
func TestSpecStatsDraftSteps(t *testing.T) {
	s := &speculationSession{enabled: true, fixedDepth: true}

	s.endRound(3, 2, 3) // drafting round
	s.endRound(4, 4, 4) // drafting round
	s.endRound(2, 0, 0) // drafting round, all rejected
	s.endRound(0, 0, 0) // plain round

	if s.stats.iterations != 4 {
		t.Errorf("iterations = %d, want 4", s.stats.iterations)
	}
	if s.stats.draftSteps != 3 {
		t.Errorf("draftSteps = %d, want 3 (only rounds that drafted)", s.stats.draftSteps)
	}
	if s.stats.drafted != 9 {
		t.Errorf("drafted = %d, want 9", s.stats.drafted)
	}
	if s.stats.accepted != 6 {
		t.Errorf("accepted = %d, want 6", s.stats.accepted)
	}
}

func TestReportCounts(t *testing.T) {
	t.Run("nil session reports nothing", func(t *testing.T) {
		var s *speculationSession
		var resp CompletionResponse
		s.reportCounts(&resp)
		if resp.DraftCount != nil || resp.DraftAcceptedCount != nil || resp.DraftStepCount != nil || resp.StepCount != nil {
			t.Errorf("resp = %+v, want all speculative counts nil", resp)
		}
	})

	t.Run("disabled session reports nothing", func(t *testing.T) {
		// A logprobs request parks permanently but still accumulates rounds.
		s := &speculationSession{enabled: false}
		s.endRound(0, 0, 0)
		var resp CompletionResponse
		s.reportCounts(&resp)
		if resp.DraftCount != nil || resp.DraftAcceptedCount != nil || resp.DraftStepCount != nil || resp.StepCount != nil {
			t.Errorf("resp = %+v, want all speculative counts nil", resp)
		}
	})

	t.Run("enabled session that never drafted reports zeros", func(t *testing.T) {
		s := &speculationSession{enabled: true, fixedDepth: true}
		s.countForward()
		s.countForward()
		s.endRound(0, 0, 0)
		s.endRound(0, 0, 0)
		var resp CompletionResponse
		s.reportCounts(&resp)
		for name, got := range map[string]*int{
			"DraftCount":         resp.DraftCount,
			"DraftAcceptedCount": resp.DraftAcceptedCount,
			"DraftStepCount":     resp.DraftStepCount,
			"StepCount":          resp.StepCount,
		} {
			if got == nil {
				t.Errorf("%s = nil, want zero", name)
			}
		}
		if got := *resp.DraftCount; got != 0 {
			t.Errorf("DraftCount = %d, want 0", got)
		}
		if got := *resp.StepCount; got != 2 {
			t.Errorf("StepCount = %d, want 2", got)
		}
	})

	t.Run("enabled session reports counts as copies", func(t *testing.T) {
		s := &speculationSession{enabled: true, fixedDepth: true}
		s.countForward()
		s.countForward()
		s.endRound(5, 3, 5)
		s.endRound(0, 0, 0)

		var resp CompletionResponse
		s.reportCounts(&resp)

		s.countForward()
		s.endRound(2, 1, 2) // later rounds must not mutate the response already sent
		if got := *resp.DraftCount; got != 5 {
			t.Errorf("DraftCount = %d, want 5", got)
		}
		if got := *resp.DraftAcceptedCount; got != 3 {
			t.Errorf("DraftAcceptedCount = %d, want 3", got)
		}
		if got := *resp.DraftStepCount; got != 1 {
			t.Errorf("DraftStepCount = %d, want 1", got)
		}
		if got := *resp.StepCount; got != 2 {
			t.Errorf("StepCount = %d, want 2", got)
		}
	})
}

// Steps are target forward passes over the same window eval_duration covers,
// recorded where each forward is dispatched rather than derived from decode
// rounds. A round count misses the pipelined decoder's seed forward, which no
// round spans, so a response that ends parked -- a short run, or one whose
// num_predict leaves no room to draft -- completes more forwards than rounds.
// The reported steps must therefore be checked against the model's own count.
func TestStepCountMatchesForwardPasses(t *testing.T) {
	const eos int32 = 7
	predict := map[int32]int32{1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: eos, eos: 0}

	for _, tt := range []struct {
		name       string
		numPredict int
	}{
		{"drafting run", 40},
		{"short run ends parked", 3},
		{"no budget to draft", 1},
	} {
		t.Run(tt.name, func(t *testing.T) {
			mlxtest.Run(t, func(t *mlxtest.T) {
				r := mtpTestRunner(t, predict, []int32{eos}, sampler.Options{})
				caches, _ := newMTPTestCaches(1)
				r.cache.caches = caches
				r.spec = newSpeculation(r, pipelineSlot, &fakeMTPDraft{predict: predict}, caches, nil)
				session, ch := newMTPTestSession(caches)

				req := request{
					Responses:         ch,
					Tokens:            []int32{0},
					CompletionRequest: CompletionRequest{Options: api.Options{NumPredict: tt.numPredict}},
					SamplerOpts:       sampler.Options{},
				}
				spec := r.spec.open(req, nil)
				if spec == nil || !spec.enabled {
					t.Fatal("speculation session is not drafting")
				}
				pinDraftLimit(spec, 4)
				d := spec.decoder(mlx.FromValues([]int32{1}, 1), 1, nil)
				if err := driveTestDecoder(context.Background(), r, req, session, d, 0); err != nil {
					t.Fatalf("decode: %v", err)
				}

				_, final := collectResponses(ch)
				if final.StepCount == nil {
					t.Fatalf("final response = %+v, want a step count", final)
				}
				forwards := len(r.Model.(*fakeMTPModel).forwards)
				if got := *final.StepCount; got != forwards {
					t.Errorf("StepCount = %d, target forward passes = %d", got, forwards)
				}
			})
		})
	}
}
