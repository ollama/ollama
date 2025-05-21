package rope

import "github.com/ollama/ollama/ml"

// Options contains optional parameters for RoPE function
type Options struct {
	OriginalContextLength int
	Type                  int
	Factors               ml.Tensor
}

// WithOriginalContextLength sets a custom context length
func WithOriginalContextLength(n int) func(*Options) {
	return func(opts *Options) {
		opts.OriginalContextLength = n
	}
}

// WithType sets RoPE type to NeoX
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
