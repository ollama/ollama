package rope

import "github.com/ollama/ollama/ml"

// Options contains optional parameters for RoPE function
type Options struct {
	Type    int
	Factors ml.Tensor

	// YaRN options
	YaRN struct {
		OriginalContextLength int
		ExtrapolationFactor,
		AttentionFactor,
		BetaFast,
		BetaSlow float32
	}

	// MRoPE options
	MRoPE struct {
		Sections []int
	}
}

// WithTypeNeoX sets RoPE type to NeoX
func WithTypeNeoX() func(*Options) {
	return func(opts *Options) {
		opts.Type = 0b0010
	}
}

// WithTypeMRoPE sets RoPE type to MRoPE
func WithTypeMRoPE() func(*Options) {
	return func(opts *Options) {
		opts.Type = 0b1000
	}
}

// WithTypeVision sets RoPE type to Vision
func WithTypeVision() func(*Options) {
	return func(opts *Options) {
		opts.Type = 0b11000
	}
}

// WithFactors sets custom rope factors
func WithFactors(factors ml.Tensor) func(*Options) {
	return func(opts *Options) {
		if factors != nil {
			opts.Factors = factors
		}
	}
}

// WithOriginalContextLength sets a custom context length
func WithOriginalContextLength(n int) func(*Options) {
	return func(opts *Options) {
		opts.YaRN.OriginalContextLength = n
	}
}

func WithExtrapolationFactor(extrapolationFactor float32) func(*Options) {
	return func(opts *Options) {
		opts.YaRN.ExtrapolationFactor = extrapolationFactor
	}
}

func WithAttentionFactor(attentionFactor float32) func(*Options) {
	return func(opts *Options) {
		opts.YaRN.AttentionFactor = attentionFactor
	}
}

func WithMRoPESections(sections []int) func(*Options) {
	return func(opts *Options) {
		opts.MRoPE.Sections = sections
	}
}
