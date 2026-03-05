//go:build mlx

package sample

import (
	"math"

	"github.com/ollama/ollama/x/mlxrunner/mlx"
)

type Transform func(*Sampler, *mlx.Array) *mlx.Array

type Sampler struct {
	Temperature     float32
	TopP            float32
	MinP            float32
	TopK            int
	RepeatLastN     int
	PresencePenalty float32

	history    []int32
	transforms []Transform
}

func New(temp, top_p, min_p float32, top_k, repeatLastN int, presencePenalty float32) *Sampler {
	s := &Sampler{
		Temperature:     temp,
		TopP:            top_p,
		MinP:            min_p,
		TopK:            top_k,
		RepeatLastN:     repeatLastN,
		PresencePenalty: presencePenalty,
	}

	var transforms []Transform
	if presencePenalty != 0 {
		transforms = append(transforms, penalty)
	}

	if top_p > 0 && top_p < 1 {
		transforms = append(transforms, topP)
	}

	if min_p != 0 {
		transforms = append(transforms, minP)
	}

	if top_k > 0 {
		transforms = append(transforms, topK)
	}

	if temp == 0 {
		transforms = append(transforms, greedy)
	} else {
		transforms = append(transforms, temperature)
	}

	s.transforms = transforms
	return s
}

func (s *Sampler) ResetHistory(history []int32) {
	s.history = append(s.history[:0], history...)
}

func (s *Sampler) AppendToken(token int32) {
	s.history = append(s.history, token)
}

func (s *Sampler) Sample(logits *mlx.Array) *mlx.Array {
	for _, transform := range s.transforms {
		logits = transform(s, logits)
	}
	return logits
}

func greedy(_ *Sampler, logits *mlx.Array) *mlx.Array {
	return logits.Argmax(-1, false)
}

func temperature(s *Sampler, logits *mlx.Array) *mlx.Array {
	return mlx.DivScalar(logits, s.Temperature).Categorical(-1)
}

func topP(s *Sampler, logprobs *mlx.Array) *mlx.Array {
	if s.TopP <= 0 || s.TopP >= 1 {
		return logprobs
	}

	order := logprobs.Negative().ArgsortAxis(-1)
	sortedLogprobs := logprobs.TakeAlongAxis(order, -1)
	sortedProbs := mlx.SoftmaxAxis(sortedLogprobs, -1, true)
	prevCumProbs := sortedProbs.Cumsum(-1, false, true).Subtract(sortedProbs)
	keep := prevCumProbs.Less(mlx.FromValue(s.TopP))
	filtered := mlx.Where(keep, sortedLogprobs, mlx.FromValue(float32(math.Inf(-1))))
	return logprobs.PutAlongAxis(order, filtered, -1)
}

func minP(_ *Sampler, logprobs *mlx.Array) *mlx.Array {
	// TODO: implement
	return logprobs
}

func topK(s *Sampler, logprobs *mlx.Array) *mlx.Array {
	if s.TopK <= 0 {
		return logprobs
	}

	vocab := logprobs.Dim(logprobs.NumDims() - 1)
	if s.TopK >= vocab {
		return logprobs
	}

	mask := logprobs.Negative().ArgpartitionAxis(s.TopK-1, -1).Slice(mlx.Slice(), mlx.Slice(s.TopK, 0))
	return logprobs.PutAlongAxis(mask, mlx.FromValue(float32(math.Inf(-1))), -1)
}

func penalty(s *Sampler, logprobs *mlx.Array) *mlx.Array {
	if len(s.history) == 0 {
		return logprobs
	}

	if s.PresencePenalty == 0 {
		return logprobs
	}

	vocab := logprobs.Dim(logprobs.NumDims() - 1)
	if vocab <= 0 {
		return logprobs
	}

	start := 0
	if s.RepeatLastN > 0 && s.RepeatLastN < len(s.history) {
		start = len(s.history) - s.RepeatLastN
	}

	seen := make(map[int32]struct{}, len(s.history)-start)
	indices := make([]int32, 0, len(s.history)-start)
	for _, token := range s.history[start:] {
		if token < 0 || token >= int32(vocab) {
			continue
		}
		if _, ok := seen[token]; ok {
			continue
		}
		seen[token] = struct{}{}
		indices = append(indices, token)
	}

	if len(indices) == 0 {
		return logprobs
	}

	indexShape := []int32{int32(len(indices))}
	if logprobs.NumDims() > 1 {
		indexShape = []int32{1, int32(len(indices))}
	}

	tokenIndices := mlx.NewArrayInt32(indices, indexShape)
	selected := logprobs.TakeAlongAxis(tokenIndices, -1)
	adjusted := mlx.AddScalar(selected, -s.PresencePenalty)
	return logprobs.PutAlongAxis(tokenIndices, adjusted, -1)
}
