//go:build mlx

package sample

import (
	"math"
	"sort"
	"testing"

	"github.com/ollama/ollama/x/mlxrunner/mlx"
)

// logprobEntry is the (token id, logprob) pair returned by the sampler's
// top-K extraction, used after the test-side descending sort.
type logprobEntry struct {
	id      int
	logprob float64
}

// runSampleLogprobs drives Sample on a fresh Sampler configured for logprobs
// and returns the greedily-sampled token id, its logprob, and the top-K
// entries sorted descending by logprob. Logits must be a [vocab]-shaped
// slice; the helper reshapes it to [1, vocab] before calling the sampler.
func runSampleLogprobs(t *testing.T, logits []float32, topK int) (int, float64, []logprobEntry) {
	t.Helper()

	s := New(Options{Logprobs: true, TopLogprobs: topK})
	defer func() {
		s.Free()
		mlx.Sweep()
	}()

	tensor := mlx.FromValues(logits, 1, len(logits))
	res := s.Sample(tensor)

	mlx.Pin(res.Arrays()...)
	defer mlx.Unpin(res.Arrays()...)
	mlx.Sweep()
	mlx.Eval(res.Arrays()...)

	selected := res.Token.Int()
	selLP := float64(res.Logprob.Floats()[0])

	var top []logprobEntry
	if topK > 0 && res.TopTokens != nil {
		ids := res.TopTokens.Ints()
		vals := res.TopLogprobs.Floats()
		top = make([]logprobEntry, len(ids))
		for i, id := range ids {
			top[i] = logprobEntry{id: id, logprob: float64(vals[i])}
		}
		sort.Slice(top, func(i, j int) bool { return top[i].logprob > top[j].logprob })
	}
	return selected, selLP, top
}

func TestSampleLogprobsBasic(t *testing.T) {
	tests := []struct {
		name           string
		logits         []float32
		topK           int
		wantSelectedID int
		wantTopLen     int
	}{
		{
			name:           "single token without top logprobs",
			logits:         []float32{1.0, 0.5, 0.3, 0.1},
			topK:           0,
			wantSelectedID: 0,
			wantTopLen:     0,
		},
		{
			name:           "single token with top logprobs",
			logits:         []float32{1.0, 0.5, 0.3, 0.1},
			topK:           3,
			wantSelectedID: 0,
			wantTopLen:     3,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			selected, _, top := runSampleLogprobs(t, tt.logits, tt.topK)
			if selected != tt.wantSelectedID {
				t.Errorf("selected = %d, want %d", selected, tt.wantSelectedID)
			}
			if len(top) != tt.wantTopLen {
				t.Errorf("top-K length = %d, want %d", len(top), tt.wantTopLen)
			}
		})
	}
}

func TestSampleLogprobsNumericalStability(t *testing.T) {
	logits := []float32{1000.0, 999.0, 998.0}
	_, selLP, top := runSampleLogprobs(t, logits, 3)

	if math.IsInf(selLP, 0) || math.IsNaN(selLP) {
		t.Errorf("selected logprob is not finite: %f", selLP)
	}
	for i, e := range top {
		if math.IsInf(e.logprob, 0) || math.IsNaN(e.logprob) {
			t.Errorf("top[%d] logprob is not finite: %f", i, e.logprob)
		}
	}
	for i := 1; i < len(top); i++ {
		if top[i].logprob > top[i-1].logprob {
			t.Errorf("top logprobs not descending: %f > %f", top[i].logprob, top[i-1].logprob)
		}
	}
}

func TestSampleLogprobsProbabilityCorrectness(t *testing.T) {
	tests := []struct {
		name   string
		logits []float32
	}{
		{"uniform", []float32{1.0, 1.0, 1.0, 1.0}},
		{"different", []float32{2.0, 1.0, 0.5, 0.1}},
		{"negative", []float32{-1.0, -2.0, -3.0, -4.0}},
		{"mixed", []float32{5.0, -5.0, 0.0, 2.5}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			selected, selLP, top := runSampleLogprobs(t, tt.logits, len(tt.logits))

			if selLP > 0 {
				t.Errorf("selected logprob should be <= 0, got %f", selLP)
			}
			for i, e := range top {
				if e.logprob > 0 {
					t.Errorf("top[%d] logprob should be <= 0, got %f", i, e.logprob)
				}
			}

			if tt.name == "uniform" {
				want := 1.0 / float64(len(tt.logits))
				got := math.Exp(selLP)
				if math.Abs(got-want) > 1e-6 {
					t.Errorf("uniform logits: selected prob = %f, want %f", got, want)
				}
			}

			for i := 1; i < len(top); i++ {
				if top[i].logprob > top[i-1].logprob {
					t.Errorf("top logprobs not descending at %d: %f > %f",
						i, top[i].logprob, top[i-1].logprob)
				}
			}

			found := false
			for _, e := range top {
				if e.id == selected {
					found = true
					if math.Abs(e.logprob-selLP) > 1e-6 {
						t.Errorf("selected logprob mismatch: selLP=%f top=%f", selLP, e.logprob)
					}
					break
				}
			}
			if !found {
				t.Errorf("selected token %d not present in top-K", selected)
			}
		})
	}
}

func TestSampleLogprobsSoftmaxCorrectness(t *testing.T) {
	tests := []struct {
		name   string
		logits []float32
	}{
		{"small vocabulary", []float32{1.0, 2.0, 3.0}},
		{"large differences", []float32{10.0, 0.0, -10.0}},
		{"all equal", []float32{5.0, 5.0, 5.0, 5.0, 5.0}},
		{"very large values", []float32{500.0, 499.0, 498.0}},
		{"very small values", []float32{-500.0, -499.0, -498.0}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, _, top := runSampleLogprobs(t, tt.logits, len(tt.logits))
			if len(top) != len(tt.logits) {
				t.Fatalf("top-K length = %d, want %d", len(top), len(tt.logits))
			}

			var sum float64
			for _, e := range top {
				p := math.Exp(e.logprob)
				if p < 0 || p > 1 {
					t.Errorf("token %d: probability %f out of [0,1]", e.id, p)
				}
				sum += p
			}

			if math.Abs(sum-1.0) > 1e-5 {
				t.Errorf("probabilities sum = %f, want 1.0", sum)
			}
		})
	}
}

func TestSampleLogprobsSelectedTokenCorrectness(t *testing.T) {
	logits := []float32{3.0, 1.0, 2.0, 0.5}

	maxIdx := 0
	for i, v := range logits[1:] {
		if v > logits[maxIdx] {
			maxIdx = i + 1
		}
	}

	selected, selLP, top := runSampleLogprobs(t, logits, len(logits))

	if selected != maxIdx {
		t.Errorf("selected = %d, want argmax %d", selected, maxIdx)
	}

	if top[0].id != maxIdx {
		t.Errorf("top[0].id = %d, want argmax %d", top[0].id, maxIdx)
	}
	if math.Abs(top[0].logprob-selLP) > 1e-6 {
		t.Errorf("top[0].logprob = %f, want selected %f", top[0].logprob, selLP)
	}
}

func TestSampleLogprobsTopKOrdering(t *testing.T) {
	// Logits chosen so argmax order differs from index order.
	logits := []float32{2.0, 5.0, 1.0, 4.0, 3.0}
	wantOrder := []int{1, 3, 4, 0, 2}

	_, _, top := runSampleLogprobs(t, logits, len(logits))

	if len(top) != len(wantOrder) {
		t.Fatalf("top-K length = %d, want %d", len(top), len(wantOrder))
	}
	for i, e := range top {
		if e.id != wantOrder[i] {
			t.Errorf("top[%d].id = %d, want %d", i, e.id, wantOrder[i])
		}
	}
	for i := 1; i < len(top); i++ {
		if top[i].logprob > top[i-1].logprob {
			t.Errorf("top[%d].logprob (%f) > top[%d].logprob (%f)",
				i, top[i].logprob, i-1, top[i-1].logprob)
		}
	}
}
