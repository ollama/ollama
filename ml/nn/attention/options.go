package attention

import (
	"github.com/ollama/ollama/ml"
)

type Options struct {
	// Scale is a scaling factor applied to the attention scores. Default is 1/√d_k.
	Scale float64

	// LogitSoftcap is used to apply a soft cap to the logits before softmax.
	LogitSoftcap float32

	// Mask is used in some attention mechanisms to mask out certain positions.
	Mask ml.Tensor

	// Sinks is used in some attention mechanisms to store additional data.
	Sinks ml.Tensor

	// MLA is used in some attention mechanisms for multi-latent attention.
	MLA ml.Tensor

	// Cached indicates whether key/value were retrieved from cache.
	Cached bool
}

func WithScale(scale float64) func(*Options) {
	return func(o *Options) {
		o.Scale = scale
	}
}

func WithSinks(sinks ml.Tensor) func(*Options) {
	return func(o *Options) {
		o.Sinks = sinks
	}
}

func WithMLA(mla ml.Tensor) func(*Options) {
	return func(o *Options) {
		o.MLA = mla
	}
}

func WithMask(mask ml.Tensor) func(*Options) {
	return func(o *Options) {
		o.Mask = mask
	}
}

func WithLogitSoftcap(softcap float32) func(*Options) {
	return func(o *Options) {
		o.LogitSoftcap = softcap
	}
}
