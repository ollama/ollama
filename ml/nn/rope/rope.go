package rope

import "github.com/ollama/ollama/ml"

// Options contains optional parameters for RoPE function
type Options struct {
	Type                  int
	Factors               ml.Tensor
	OriginalContextLength int

	// YaRN options
	ExtrapolationFactor,
	AttentionFactor,
	BetaFast,
	BetaSlow float32
}

// WithOriginalContextLength sets a custom context length
func WithOriginalContextLength(n int) func(*Options) {
	return func(opts *Options) {
		opts.OriginalContextLength = n
	}
}

// WithTypeNeoX sets RoPE type to NeoX
func WithTypeNeoX() func(*Options) {
	return func(opts *Options) {
		opts.Type = 2
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

func WithExtrapolationFactor(extrapolationFactor float32) func(*Options) {
	return func(opts *Options) {
		opts.ExtrapolationFactor = extrapolationFactor
	}
}

func WithAttentionFactor(attentionFactor float32) func(*Options) {
	return func(opts *Options) {
		opts.AttentionFactor = attentionFactor
	}
}
