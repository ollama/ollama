package lfm3speech

import (
	"math"
	"math/rand/v2"
	"slices"
	"testing"
)

func testSampler(t *testing.T, opts SampleOptions) *FrameSampler {
	t.Helper()
	cfg := &Config{NumCodebooks: 3}
	s, err := NewFrameSampler(cfg, opts, rand.New(rand.NewPCG(1, 2)))
	if err != nil {
		t.Fatal(err)
	}
	return s
}

func TestRepetitionPenalty(t *testing.T) {
	s := testSampler(t, SampleOptions{RepetitionPenalty: 2, RepetitionWindow: 2})
	// Codebook 1 used code 0 twice and code 2 once; the penalty applies once
	// per code, dividing positive and multiplying negative scores.
	s.remember([]int32{4, 0, 9})
	s.remember([]int32{4, 2, 9})
	s.remember([]int32{4, 0, 9})
	if len(s.history) != 2 {
		t.Fatalf("history holds %d frames, want the window of 2", len(s.history))
	}
	logits := []float32{1.2, 1, -1, 0.5}
	if got := s.sample(logits, 1); got != 1 {
		t.Fatalf("greedy code %d, want 1", got)
	}
	want := []float32{0.6, 1, -2, 0.5}
	for i := range want {
		if logits[i] != want[i] {
			t.Fatalf("penalized logits %v, want %v", logits, want)
		}
	}
}

func TestSampleTemperatureAndTopK(t *testing.T) {
	s := testSampler(t, SampleOptions{Temperature: 1, TopK: 2})
	counts := make([]int, 4)
	for range 4000 {
		counts[s.sample([]float32{2, 0, 1, -5}, 0)]++
	}
	if counts[1] != 0 || counts[3] != 0 {
		t.Fatalf("top-k sampled outside the top two: %v", counts)
	}
	// p(0)/p(2) = e at unit temperature.
	if ratio := float64(counts[0]) / float64(counts[2]); math.Abs(ratio-math.E) > 0.3 {
		t.Fatalf("sampled ratio %.2f, want about e (%v)", ratio, counts)
	}
}

func TestSamplerOrder(t *testing.T) {
	cfg := &Config{NumCodebooks: 3}
	if _, err := NewFrameSampler(cfg, SampleOptions{Order: []int{0, 0, 2}}, nil); err == nil {
		t.Fatal("accepted a non-permutation order")
	}
	if _, err := NewFrameSampler(cfg, SampleOptions{Order: []int{0, 1}}, nil); err == nil {
		t.Fatal("accepted a short order")
	}
	if _, err := NewFrameSampler(cfg, SampleOptions{Order: []int{}}, nil); err == nil {
		t.Fatal("accepted an empty order")
	}
	s := testSampler(t, SampleOptions{})
	seen := map[[3]int]bool{}
	for range 200 {
		o := s.order(3)
		seen[[3]int(o)] = true
	}
	if len(seen) != 6 {
		t.Fatalf("random orders cover %d of 6 permutations", len(seen))
	}
}

func TestKthSmallestMatchesSort(t *testing.T) {
	rng := rand.New(rand.NewPCG(5, 6))
	for _, size := range []int{1, 2, 3, 17, 2017} {
		for _, distinct := range []int{1, 3, 1 << 30} {
			v := make([]float32, size)
			for i := range v {
				v[i] = float32(rng.IntN(distinct)) - float32(distinct/2)
			}
			sorted := slices.Sorted(slices.Values(v))
			for _, n := range []int{0, size / 2, size - 1, max(0, size-128)} {
				if got := kthSmallest(slices.Clone(v), n); got != sorted[n] {
					t.Fatalf("size %d distinct %d: kthSmallest(%d) = %v, want %v", size, distinct, n, got, sorted[n])
				}
			}
		}
	}
}
