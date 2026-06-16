package diffusiongemma

import (
	"math"
	"math/rand"
	"testing"
)

func TestTempAt(t *testing.T) {
	const nSteps = 48
	tMin, tMax := float32(0.4), float32(0.8)
	if got := tempAt(nSteps, nSteps, tMin, tMax); math.Abs(float64(got-tMax)) > 1e-6 {
		t.Errorf("tempAt(nSteps) = %v, want tMax=%v", got, tMax)
	}
	// Step 1 is the tMin floor + one increment, NOT tMin exactly (matches reference).
	want1 := tMin + (tMax-tMin)/float32(nSteps)
	if got := tempAt(1, nSteps, tMin, tMax); math.Abs(float64(got-want1)) > 1e-6 {
		t.Errorf("tempAt(1) = %v, want %v", got, want1)
	}
	if got := tempAt(nSteps/2, nSteps, tMin, tMax); got <= tMin || got >= tMax {
		t.Errorf("tempAt(mid) = %v, want in (%v,%v)", got, tMin, tMax)
	}
}

func TestNumCanvases(t *testing.T) {
	cases := []struct{ nPredict, canvas, want int }{
		{0, 256, 1},
		{-5, 256, 1},
		{1, 256, 1},
		{256, 256, 1},
		{257, 256, 2},
		{1024, 256, 4},
	}
	for _, c := range cases {
		if got := numCanvases(c.nPredict, c.canvas); got != c.want {
			t.Errorf("numCanvases(%d,%d) = %d, want %d", c.nPredict, c.canvas, got, c.want)
		}
	}
}

func TestEntropyBoundAccept(t *testing.T) {
	// Distinct entropies: accept the ascending prefix while the running sum is
	// <= bound when tested (so the last accepted may exceed it).
	// [0.0, 0.15, 0.3], bound 0.1: accept 0.0 (sum->0), accept 0.15 (0<=0.1, sum->0.15),
	// reject 0.3 (0.15>0.1). -> [true, true, false].
	got := entropyBoundAccept([]float32{0.0, 0.15, 0.3}, 0.1)
	want := []bool{true, true, false}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("entropyBoundAccept = %v, want %v", got, want)
		}
	}

	// All-zero entropy: every position accepted regardless of bound.
	for _, b := range entropyBoundAccept([]float32{0, 0, 0, 0}, 0.1) {
		if !b {
			t.Fatalf("all-zero entropy should accept everything")
		}
	}

	// The off-by-one: a single high-entropy position with bound 0 is still
	// accepted (bound tested before adding), but the next is not.
	got = entropyBoundAccept([]float32{0.5, 0.5}, 0.0)
	if !got[0] && !got[1] {
		t.Fatalf("expected exactly one accepted with bound 0, got %v", got)
	}
	if got[0] && got[1] {
		t.Fatalf("expected exactly one accepted with bound 0, got %v", got)
	}
}

func TestStableAndConfident(t *testing.T) {
	lowEnt := []float32{0.001, 0.001}
	highEnt := []float32{0.5, 0.5}
	am := []int32{3, 7}

	if stableAndConfident(am, nil, lowEnt, 0.005, 1) {
		t.Error("first step (prevArgmax nil) must not be stable")
	}
	if !stableAndConfident(am, []int32{3, 7}, lowEnt, 0.005, 1) {
		t.Error("equal argmax + low entropy should be stable&confident")
	}
	if stableAndConfident(am, []int32{3, 9}, lowEnt, 0.005, 1) {
		t.Error("changed argmax should not be stable")
	}
	if stableAndConfident(am, []int32{3, 7}, highEnt, 0.005, 1) {
		t.Error("high entropy should not be confident")
	}
	if !stableAndConfident(am, nil, lowEnt, 0.005, 0) {
		t.Error("stabThresh=0 should be always-stable (confident still required)")
	}
}

func TestRenoise(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	sampled := []int32{5, 6, 7, 8}
	accept := []bool{true, false, true, false}
	next := renoise(sampled, accept, rng, 100)
	if next[0] != 5 || next[2] != 7 {
		t.Errorf("accepted positions must keep their token, got %v", next)
	}
	for _, i := range []int{1, 3} {
		if next[i] < 0 || next[i] >= 100 {
			t.Errorf("renoised token %d out of range: %d", i, next[i])
		}
	}
}

func TestSampleCanvas(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	canvasLen, vocab := 2, 3
	// pos0 peaked on token 0 (low entropy); pos1 uniform (max entropy ~ln3).
	logits := []float32{
		10, 0, 0,
		0, 0, 0,
	}
	s := sampleCanvas(logits, canvasLen, vocab, 1.0, 2, rng)

	if s.argmax[0] != 0 {
		t.Errorf("argmax[0] = %d, want 0", s.argmax[0])
	}
	if s.entropy[0] >= s.entropy[1] {
		t.Errorf("peaked position should have lower entropy: %v vs %v", s.entropy[0], s.entropy[1])
	}
	if math.Abs(float64(s.entropy[1])-math.Log(3)) > 1e-3 {
		t.Errorf("uniform entropy = %v, want ~ln3=%v", s.entropy[1], math.Log(3))
	}
	// Top-1 self-cond prob at the peaked position should be ~1.
	if s.scProbs[0] < 0.99 {
		t.Errorf("peaked top-1 prob = %v, want ~1", s.scProbs[0])
	}
	if s.scIDs[0] != 0 {
		t.Errorf("peaked top-1 id = %d, want 0", s.scIDs[0])
	}
	for j := range canvasLen {
		if s.sampled[j] < 0 || s.sampled[j] >= int32(vocab) {
			t.Errorf("sampled[%d] out of range: %d", j, s.sampled[j])
		}
	}
}
