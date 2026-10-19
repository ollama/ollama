package lfm3speech

import (
	"fmt"
	"math"
	"math/rand/v2"
	"slices"
)

// SampleOptions controls how each codebook of a speech frame is drawn.
type SampleOptions struct {
	// Temperature <= 0 decodes greedily.
	Temperature float32
	// TopK <= 0 keeps every code.
	TopK int
	// RepetitionPenalty applies to the codes a codebook used in the last
	// RepetitionWindow frames; 0 or 1 disables it.
	RepetitionPenalty float32
	RepetitionWindow  int
	// Order fixes the codebook reveal order of every frame; nil draws a
	// random permutation per frame as the publisher does.
	Order []int
}

// DefaultSampleOptions are the publisher's recommended speech settings,
// which differ between Omni and Audio.
func DefaultSampleOptions(omni bool) SampleOptions {
	if omni {
		return SampleOptions{Temperature: 0.4, RepetitionPenalty: 1.2, RepetitionWindow: 64}
	}
	return SampleOptions{Temperature: 1, TopK: 128, RepetitionPenalty: 1.1, RepetitionWindow: 4096}
}

// FrameSampler carries one stream's sampling state across frames: options,
// randomness, and the recent frames the repetition penalty reads. Keep one
// per generated response, across speech segments of an interleaved answer.
type FrameSampler struct {
	opts    SampleOptions
	rng     *rand.Rand
	history [][]int32
	scratch []float32
}

// NewFrameSampler validates opts against cfg. A nil rng is seeded randomly.
func NewFrameSampler(cfg *Config, opts SampleOptions, rng *rand.Rand) (*FrameSampler, error) {
	if math.IsNaN(float64(opts.Temperature)) || math.IsInf(float64(opts.Temperature), 0) {
		return nil, fmt.Errorf("invalid speech temperature %v", opts.Temperature)
	}
	if p := opts.RepetitionPenalty; p < 0 || math.IsNaN(float64(p)) || math.IsInf(float64(p), 0) {
		return nil, fmt.Errorf("invalid speech repetition penalty %v", p)
	}
	if opts.RepetitionWindow < 0 {
		return nil, fmt.Errorf("invalid speech repetition window %d", opts.RepetitionWindow)
	}
	if opts.Order != nil {
		sorted := slices.Sorted(slices.Values(opts.Order))
		if len(sorted) != cfg.NumCodebooks {
			return nil, fmt.Errorf("codebook order %v is not a permutation of %d codebooks", opts.Order, cfg.NumCodebooks)
		}
		for i, v := range sorted {
			if v != i {
				return nil, fmt.Errorf("codebook order %v is not a permutation of %d codebooks", opts.Order, cfg.NumCodebooks)
			}
		}
	}
	if rng == nil {
		rng = rand.New(rand.NewPCG(rand.Uint64(), rand.Uint64()))
	}
	return &FrameSampler{opts: opts, rng: rng}, nil
}

func (s *FrameSampler) order(n int) []int {
	if s.opts.Order != nil {
		return s.opts.Order
	}
	return s.rng.Perm(n)
}

// remember records a non-EOS frame for the repetition penalty.
func (s *FrameSampler) remember(frame []int32) {
	if s.opts.RepetitionWindow == 0 {
		return
	}
	if len(s.history) == s.opts.RepetitionWindow {
		s.history = slices.Delete(s.history, 0, 1)
	}
	s.history = append(s.history, slices.Clone(frame))
}

// sample draws codebook k's code from its logits, which it may modify.
func (s *FrameSampler) sample(logits []float32, k int) int32 {
	if p := s.opts.RepetitionPenalty; p != 0 && p != 1 {
		seen := make(map[int32]bool, len(s.history))
		for _, frame := range s.history {
			c := frame[k]
			if seen[c] {
				continue
			}
			seen[c] = true
			if logits[c] < 0 {
				logits[c] *= p
			} else {
				logits[c] /= p
			}
		}
	}
	if s.opts.Temperature <= 0 {
		return int32(argmax(logits))
	}
	t := s.opts.Temperature
	for i := range logits {
		logits[i] /= t
	}
	if k := s.opts.TopK; k > 0 && k < len(logits) {
		s.scratch = append(s.scratch[:0], logits...)
		threshold := kthSmallest(s.scratch, len(logits)-k)
		for i, v := range logits {
			if v < threshold {
				logits[i] = float32(math.Inf(-1))
			}
		}
	}
	peak := logits[argmax(logits)]
	probs := make([]float64, len(logits))
	var total float64
	for i, v := range logits {
		probs[i] = math.Exp(float64(v - peak))
		total += probs[i]
	}
	r := s.rng.Float64() * total
	for i, p := range probs {
		if r < p {
			return int32(i)
		}
		r -= p
	}
	return int32(argmax(logits))
}

// kthSmallest returns the value v[n] would hold if v were sorted ascending,
// reordering v. It selects in linear expected time; a full sort of every
// codebook's logits was a measurable share of speech decoding.
func kthSmallest(v []float32, n int) float32 {
	lo, hi := 0, len(v)-1
	for lo < hi {
		pivot := v[lo+(hi-lo)/2]
		i, j := lo, hi
		for i <= j {
			for v[i] < pivot {
				i++
			}
			for v[j] > pivot {
				j--
			}
			if i <= j {
				v[i], v[j] = v[j], v[i]
				i++
				j--
			}
		}
		switch {
		case n <= j:
			hi = j
		case n >= i:
			lo = i
		default:
			return v[n]
		}
	}
	return v[n]
}

// argmax returns the first index of the largest value.
func argmax(v []float32) int {
	best := 0
	for i, x := range v {
		if x > v[best] {
			best = i
		}
	}
	return best
}
