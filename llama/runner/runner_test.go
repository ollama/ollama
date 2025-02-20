package runner

import (
	"math"
	"testing"
)

func TestProbs(t *testing.T) {
	// Input test data
	logits := []float32{1.0, 2.0, 0.5, -1.0}
	vocabSize := 4
	want := []TokenProbs{
		{TokenID: 1, Logit: 2.0},  // Highest logit
		{TokenID: 0, Logit: 1.0},  // Second highest
		{TokenID: 2, Logit: 0.5},  // Third
		{TokenID: 3, Logit: -1.0}, // Lowest
	}

	got := probs(logits, vocabSize)

	// Test 1: Check sorting order
	for i := 0; i < len(got)-1; i++ {
		if got[i].Logit < got[i+1].Logit {
			t.Errorf("probs not properly sorted: logit at pos %d (%f) < logit at pos %d (%f)",
				i, got[i].Logit, i+1, got[i+1].Logit)
		}
	}

	// Test 2: Check probability normalization
	var sum float32
	for _, p := range got {
		sum += p.Prob
	}
	if math.Abs(float64(sum-1.0)) > 1e-6 {
		t.Errorf("probabilities do not sum to 1: got %v", sum)
	}

	// Test 3: Check token IDs match expected order
	for i, want := range want {
		if got[i].TokenID != want.TokenID {
			t.Errorf("wrong token ID at position %d: got %d, want %d",
				i, got[i].TokenID, want.TokenID)
		}
		if got[i].Logit != want.Logit {
			t.Errorf("wrong logit at position %d: got %f, want %f",
				i, got[i].Logit, want.Logit)
		}
	}

	// Test 4: Check log probs are correctly calculated
	for i, p := range got {
		expectedLogProb := float32(math.Log(float64(p.Prob)))
		if math.Abs(float64(p.LogProb-expectedLogProb)) > 1e-6 {
			t.Errorf("wrong log prob at position %d: got %f, want %f",
				i, p.LogProb, expectedLogProb)
		}
	}
}
