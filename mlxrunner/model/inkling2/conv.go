package inkling2

import (
	"math"
	"slices"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlxrunner/batch"
	"github.com/ollama/ollama/mlxrunner/cache"
	"github.com/ollama/ollama/mlxrunner/nn"
)

type shortConv struct {
	Weight *mlx.Array // [taps, channels]
	heads  int
}

// forward applies the convolution to x. A dynamic convolution also takes its
// per-token tap projection [rows, tokens, heads*taps].
func (s shortConv) forward(x, dynamic *mlx.Array, b *batch.Batch, layerCache cache.Layer) *mlx.Array {
	rows, tokens, channels := x.Dim(0), x.Dim(1), x.Dim(2)
	taps := s.Weight.Dim(0)
	tail := taps - 1
	if recurrent, ok := layerCache.(cache.Recurrent); ok && mlx.MetalIsAvailable() &&
		!slices.ContainsFunc(b.SeqQueryLens, func(n int32) bool { return int(n) != tokens }) {
		terms := taps + 1
		if dynamic != nil {
			terms += taps
		}
		history := recurrent.Get(b, x.DType())
		y, next := mlx.ShortConvolution(history.ConvState(), x, s.Weight, dynamic, s.heads, float32(1/math.Sqrt(float64(terms))))
		if y != nil {
			putConvStates(recurrent, b, history, x, next)
			return y
		}
	}
	var history *nn.RecurrentHistory
	var options []nn.RecurrentOption
	if layerCache == nil {
		options = append(options, nn.WithRecurrentState(mlx.Zeros(x.DType(), rows, tail, channels), nil))
	} else {
		recurrent := layerCache.(cache.Recurrent)
		history = recurrent.Get(b, x.DType())
		options = append(options, nn.WithRecurrentHistory(history), nn.WithSnapshotSplits(recurrent.SnapshotSplits(tokens)))
	}
	window, states := nn.CausalConvWindow(b, x, tail, options...)
	if history != nil {
		delta := slices.Repeat([]*mlx.Array{history.DeltaState()}, len(states))
		layerCache.(cache.Recurrent).Put(b, states, delta)
	}

	var perToken *mlx.Array
	if dynamic != nil {
		perToken = dynamic.AsType(mlx.DTypeFloat32).Reshape(rows, tokens, s.heads, taps)
	}
	inputs := convTapArgs(window, x, s.Weight, perToken)
	if mlx.MetalIsAvailable() {
		return compiledConvTaps(inputs...)[0]
	}
	return convTaps(inputs...)[0]
}

// forwardAdd returns sumFP32(residual, forward(x)) for fixed taps, adding in
// the convolution's launch where it can.
func (s shortConv) forwardAdd(x, residual *mlx.Array, dtype mlx.DType, b *batch.Batch, layerCache cache.Layer) *mlx.Array {
	tokens := x.Dim(1)
	if recurrent, ok := layerCache.(cache.Recurrent); ok && mlx.MetalIsAvailable() &&
		!slices.ContainsFunc(b.SeqQueryLens, func(n int32) bool { return int(n) != tokens }) {
		history := recurrent.Get(b, x.DType())
		scale := float32(1 / math.Sqrt(float64(s.Weight.Dim(0)+1)))
		if sum, next := mlx.ShortConvolutionAdd(history.ConvState(), x, s.Weight, residual, scale, dtype); sum != nil {
			putConvStates(recurrent, b, history, x, next)
			return sum
		}
	}
	return sumFP32(residual, s.forward(x, nil, b, layerCache), dtype)
}

// putConvStates stores a kernel's final conv state. A forward with snapshot
// splits, such as draft verification, also stores the state after each split:
// the trailing inputs of [state; x] through it. Short forwards keep them as
// views of one small concat; longer ones copy them out, so a held snapshot
// does not keep the whole forward's input alive.
func putConvStates(recurrent cache.Recurrent, b *batch.Batch, history *nn.RecurrentHistory, x, next *mlx.Array) {
	splits := recurrent.SnapshotSplits(x.Dim(1))
	states := make([]*mlx.Array, 0, len(splits)+1)
	if len(splits) > 0 {
		prior := history.ConvState()
		tail := prior.Dim(1)
		full := mlx.Concatenate([]*mlx.Array{prior, x}, 1)
		for _, split := range splits {
			state := full.Slice(mlx.Slice(), mlx.Slice(split, split+tail), mlx.Slice())
			if x.Dim(1) > shortTargetRows {
				state = mlx.Contiguous(state, false)
			}
			states = append(states, state)
		}
	}
	states = append(states, next)
	recurrent.Put(b, states, slices.Repeat([]*mlx.Array{history.DeltaState()}, len(states)))
}

// convTapArgs returns convTaps' inputs: x, the scale, then each tap's lagged
// window and coefficients. Slicing outside the arithmetic lets one shapeless
// trace serve every token count.
func convTapArgs(window, x, weight, perToken *mlx.Array) []*mlx.Array {
	rows, tokens, channels := x.Dim(0), x.Dim(1), x.Dim(2)
	taps := weight.Dim(0)
	tail, terms := taps-1, taps+1
	if perToken != nil {
		terms += taps
	}
	args := []*mlx.Array{x, mlx.FromValue(float32(1 / math.Sqrt(float64(terms))))}
	for tap := range taps {
		start := tap
		coef := weight.Slice(mlx.Slice(tap, tap+1), mlx.Slice())
		if perToken != nil {
			// Projected convolutions index weights by lag. The fixed
			// convolutions store their weights in chronological order.
			heads := perToken.Dim(2)
			start = tail - tap
			shift := perToken.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(), mlx.Slice(tap, tap+1))
			coef = mlx.Add(coef.AsType(mlx.DTypeFloat32).Reshape(1, 1, heads, channels/heads), shift).Reshape(rows, tokens, channels)
		}
		args = append(args, window.Slice(mlx.Slice(), mlx.Slice(start, start+tokens), mlx.Slice()), coef)
	}
	return args
}

// DepthwiseConvSiLU has fixed taps and a SiLU output. Inkling adds per-token tap
// weights and a scaled residual, so it fuses its own graph.
var compiledConvTaps = mlx.Compile("InklingConvTaps", convTaps, mlx.Shapeless())

// convTaps sums the tap products in tap order, adds x and scales the sum, all
// in FP32, and rounds the result to x's dtype.
func convTaps(in ...*mlx.Array) []*mlx.Array {
	x, scale := in[0], in[1]
	var sum *mlx.Array
	for i := 2; i < len(in); i += 2 {
		product := mlx.Mul(in[i].AsType(mlx.DTypeFloat32), in[i+1].AsType(mlx.DTypeFloat32))
		if sum == nil {
			sum = product
		} else {
			sum = mlx.Add(sum, product)
		}
	}
	sum = mlx.Add(sum, x.AsType(mlx.DTypeFloat32))
	return []*mlx.Array{mlx.Mul(sum, scale).AsType(x.DType())}
}
