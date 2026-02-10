//go:build mlx

package sample

import (
	"math"

	"github.com/ollama/ollama/x/mlxrunner/mlx"
)

type Sampler interface {
	Sample(*mlx.Array) *mlx.Array
}

func New(temp, top_p, min_p float32, top_k int) Sampler {
	if temp == 0 {
		return greedy{}
	}

	var samplers []Sampler
	if top_p > 0 && top_p < 1 {
		samplers = append(samplers, TopP(top_p))
	}

	if min_p != 0 {
		samplers = append(samplers, MinP(min_p))
	}

	if top_k > 0 {
		samplers = append(samplers, TopK(top_k))
	}

	samplers = append(samplers, Temperature(temp))
	return chain(samplers)
}

type greedy struct{}

func (greedy) Sample(logits *mlx.Array) *mlx.Array {
	return logits.Argmax(-1, false)
}

type chain []Sampler

func (c chain) Sample(logits *mlx.Array) *mlx.Array {
	for _, sampler := range c {
		logits = sampler.Sample(logits)
	}
	return logits
}

type Temperature float32

func (t Temperature) Sample(logits *mlx.Array) *mlx.Array {
	return logits.Multiply(mlx.FromValue(1 / float32(t))).Categorical(-1)
}

type TopP float32

func (p TopP) Sample(logprobs *mlx.Array) *mlx.Array {
	// TODO: implement
	return logprobs
}

type MinP float32

func (p MinP) Sample(logprobs *mlx.Array) *mlx.Array {
	// TODO: implement
	return logprobs
}

type TopK int

func (k TopK) Sample(logprobs *mlx.Array) *mlx.Array {
	mask := logprobs.Negative().ArgpartitionAxis(int(k)-1, -1).Slice(mlx.Slice(), mlx.Slice(int(k), 0))
	return logprobs.PutAlongAxis(mask, mlx.FromValue(float32(math.Inf(-1))), -1)
}
