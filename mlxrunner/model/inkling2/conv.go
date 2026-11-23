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
	Weight  *mlx.Array // [taps, channels]
	Dynamic *nn.Linear
	heads   int
}

func (s shortConv) forward(x, conditioning *mlx.Array, b *batch.Batch, layerCache cache.Layer) *mlx.Array {
	rows, tokens, channels := x.Dim(0), x.Dim(1), x.Dim(2)
	taps := s.Weight.Dim(0)
	tail := taps - 1
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

	inputs := []*mlx.Array{window, x, s.Weight}
	if s.Dynamic != nil {
		perToken := projectTarget(s.Dynamic, conditioning).AsType(mlx.DTypeFloat32).Reshape(rows, tokens, s.heads, taps)
		inputs = append(inputs, perToken)
	}
	if mlx.MetalIsAvailable() {
		return compiledConvTaps(inputs...)[0]
	}
	return convTaps(inputs...)[0]
}

// DepthwiseConvSiLU uses fixed taps and a SiLU output. Inkling can add per-token
// tap weights and needs a scaled residual, so fuse those operations in one graph.
var compiledConvTaps = mlx.Compile("InklingConvTaps", convTaps)

func convTaps(inputs ...*mlx.Array) []*mlx.Array {
	window, x, weight := inputs[0], inputs[1], inputs[2]
	rows, tokens, channels := x.Dim(0), x.Dim(1), x.Dim(2)
	taps := weight.Dim(0)
	tail, terms := taps-1, taps+1
	var perToken *mlx.Array
	var heads int
	if len(inputs) == 4 {
		perToken = inputs[3]
		heads = perToken.Dim(2)
		terms += taps
	}

	var sum *mlx.Array
	for tap := range taps {
		start := tap
		coef := weight.Slice(mlx.Slice(tap, tap+1), mlx.Slice()).AsType(mlx.DTypeFloat32)
		if perToken != nil {
			// Projected convolutions index weights by lag. The fixed
			// convolutions store their weights in chronological order.
			start = tail - tap
			shift := perToken.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(), mlx.Slice(tap, tap+1))
			coef = mlx.Add(coef.Reshape(1, 1, heads, channels/heads), shift).Reshape(rows, tokens, channels)
		}
		lagged := window.Slice(mlx.Slice(), mlx.Slice(start, start+tokens), mlx.Slice()).AsType(mlx.DTypeFloat32)
		product := mlx.Mul(lagged, coef)
		if sum == nil {
			sum = product
		} else {
			sum = mlx.Add(sum, product)
		}
	}
	sum = mlx.Add(sum, x.AsType(mlx.DTypeFloat32))
	return []*mlx.Array{mlx.MulScalar(sum, float32(1/math.Sqrt(float64(terms)))).AsType(x.DType())}
}
