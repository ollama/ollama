package mlxrunner

import (
	"testing"
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
		s.endRound(5, 3, 5)
		s.endRound(0, 0, 0)

		var resp CompletionResponse
		s.reportCounts(&resp)

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
