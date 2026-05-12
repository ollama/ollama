// Package rope provides options for RoPE
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

func WithBetaFast(betaFast float32) func(*Options) {
	return func(opts *Options) {
		opts.YaRN.BetaFast = betaFast
	}
}

func WithBetaSlow(betaSlow float32) func(*Options) {
	return func(opts *Options) {
		opts.YaRN.BetaSlow = betaSlow
	}
}

func WithMRoPE(sections []int) func(*Options) {
	return func(opts *Options) {
		opts.Type |= 1 << 3
		opts.MRoPE.Sections = sections
	}
}

func WithVision(sections []int) func(*Options) {
	return func(opts *Options) {
		opts.Type |= 1<<3 | 1<<4
		opts.MRoPE.Sections = sections
	}
}

func WithInterleaveMRoPE(sections []int) func(*Options) {
	return func(opts *Options) {
		opts.Type |= 1<<3 | 1<<5
		opts.MRoPE.Sections = sections
	}
}
