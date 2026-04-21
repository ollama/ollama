package sample

import (
	"math"

	"github.com/ollama/ollama/x/mlxrunner/mlx"
)

type Transform func(*Sampler, *mlx.Array) *mlx.Array

type Sampler struct {
	Temperature      float32
	TopP             float32
	MinP             float32
	TopK             int
	RepeatLastN      int
	RepeatPenalty    float32
	PresencePenalty  float32
	FrequencyPenalty float32

	history    *mlx.Array
	historyLen int
	transforms []Transform
}

func New(temp, top_p, min_p float32, top_k, repeatLastN int, repeatPenalty, presencePenalty, frequencyPenalty float32) *Sampler {
	if repeatPenalty <= 0 {
		repeatPenalty = 1
	}

	s := &Sampler{
		Temperature:      temp,
		TopP:             top_p,
		MinP:             min_p,
		TopK:             top_k,
		RepeatLastN:      repeatLastN,
		RepeatPenalty:    repeatPenalty,
		PresencePenalty:  presencePenalty,
		FrequencyPenalty: frequencyPenalty,
	}

	var transforms []Transform
	if s.usesHistory() {
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

func (s *Sampler) usesHistory() bool {
	return s.RepeatPenalty != 1 || s.PresencePenalty != 0 || s.FrequencyPenalty != 0
}

func (s *Sampler) setHistory(history *mlx.Array, historyLen int) {
	if history != nil {
		mlx.Pin(history)
	}
	if s.history != nil {
		mlx.Unpin(s.history)
	}
	s.history = history
	s.historyLen = historyLen
}

func (s *Sampler) ResetHistory(history []int32) {
	if !s.usesHistory() {
		return
	}
	if s.RepeatLastN > 0 && len(history) > s.RepeatLastN {
		history = history[len(history)-s.RepeatLastN:]
	}
	if len(history) == 0 {
		s.setHistory(nil, 0)
		return
	}

	tokens := append([]int32(nil), history...)
	s.setHistory(mlx.NewArrayInt32(tokens, []int32{int32(len(tokens))}), len(tokens))
}

func (s *Sampler) AppendToken(token *mlx.Array) {
	if !s.usesHistory() || token == nil {
		return
	}

	next := token.AsType(mlx.DTypeInt32)
	nextLen := next.Size()

	if s.history != nil && s.historyLen > 0 {
		next = s.history.Concatenate(0, next)
		nextLen += s.historyLen
	}

	if s.RepeatLastN > 0 && nextLen > s.RepeatLastN {
		trim := nextLen - s.RepeatLastN
		next = next.Slice(mlx.Slice(trim, nextLen))
		nextLen = s.RepeatLastN
	}

	s.setHistory(next, nextLen)
}

func (s *Sampler) Free() {
	s.setHistory(nil, 0)
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

func topP(s *Sampler, logits *mlx.Array) *mlx.Array {
	if s.TopP <= 0 || s.TopP >= 1 {
		return logits
	}

	order := logits.Negative().ArgsortAxis(-1)
	sortedLogits := logits.TakeAlongAxis(order, -1)
	sortedProbs := mlx.SoftmaxAxis(sortedLogits, -1, true)
	prevCumProbs := sortedProbs.Cumsum(-1, false, true).Subtract(sortedProbs)
	keep := prevCumProbs.Less(mlx.FromValue(s.TopP))
	filtered := mlx.Where(keep, sortedLogits, mlx.FromValue(float32(math.Inf(-1))))
	return logits.PutAlongAxis(order, filtered, -1)
}

func minP(s *Sampler, logits *mlx.Array) *mlx.Array {
	if s.MinP <= 0 || s.MinP > 1 {
		return logits
	}

	maxLogits := logits.TakeAlongAxis(logits.Argmax(-1, true), -1)
	minLogits := mlx.AddScalar(maxLogits, float32(math.Log(float64(s.MinP))))

	return mlx.Where(
		logits.Less(minLogits),
		mlx.FromValue(float32(math.Inf(-1))),
		logits,
	)
}

func topK(s *Sampler, logits *mlx.Array) *mlx.Array {
	if s.TopK <= 0 {
		return logits
	}

	vocab := logits.Dim(logits.NumDims() - 1)
	if s.TopK >= vocab {
		return logits
	}

	mask := logits.Negative().ArgpartitionAxis(s.TopK-1, -1).Slice(mlx.Slice(), mlx.Slice(s.TopK, mlx.End))
	return logits.PutAlongAxis(mask, mlx.FromValue(float32(math.Inf(-1))), -1)
}

func penalty(s *Sampler, logits *mlx.Array) *mlx.Array {
	if s.historyLen == 0 {
		return logits
	}

	tokenIndices := s.history
	if logits.NumDims() > 1 {
		tokenIndices = tokenIndices.ExpandDims(0)
	}

	if s.RepeatPenalty != 1 || s.PresencePenalty != 0 {
		adjusted := logits.TakeAlongAxis(tokenIndices, -1)
		if s.RepeatPenalty != 1 {
			factor := mlx.Where(
				adjusted.Less(mlx.FromValue(float32(0))),
				mlx.FromValue(s.RepeatPenalty),
				mlx.FromValue(1/s.RepeatPenalty),
			)
			adjusted = adjusted.Multiply(factor)
		}
		if s.PresencePenalty != 0 {
			adjusted = mlx.AddScalar(adjusted, -s.PresencePenalty)
		}
		logits = logits.PutAlongAxis(tokenIndices, adjusted, -1)
	}

	if s.FrequencyPenalty != 0 {
		logits = logits.ScatterAddAxis(tokenIndices, mlx.FromValue(-s.FrequencyPenalty), -1)
	}

	return logits
}
