package sample

import (
	"errors"
	"math"

	"golang.org/x/exp/rand"
	"gonum.org/v1/gonum/stat/sampleuv"
)

type Sampler interface {
	Sample([]float32) (int32, error)
}

type weighted struct {
	src        rand.Source
	transforms []Transform
}

// TODO(parthsareen): remove uv sample dependency https://github.com/ollama/ollama/issues/9279
func Weighted(seed *uint64, transforms ...Transform) Sampler {
	var src rand.Source
	if seed != nil {
		src = rand.NewSource(*seed)
	}
	return weighted{src: src, transforms: transforms}
}

func (s weighted) Sample(logits []float32) (int32, error) {
	logits64 := make([]float64, len(logits))
	for i, v := range logits {
		logits64[i] = float64(v)
	}

	for _, t := range s.transforms {
		logits64 = t.Apply(logits64)
	}

	logitsCopy := make([]float64, 0, len(logits))
	indices := make([]int, 0, len(logits))
	for i, logit := range logits64 {
		if !math.IsInf(logit, -1) {
			logitsCopy = append(logitsCopy, logit)
			indices = append(indices, i)
		}
	}

	if len(logitsCopy) == 0 {
		return -1, errors.New("no valid logits found for weighed sampling")
	}

	probs := softmax(logitsCopy)
	w := sampleuv.NewWeighted(probs, s.src)
	if idx, ok := w.Take(); ok {
		return int32(indices[idx]), nil
	}
	return -1, errors.New("weighed sampler failed, no valid token found")
}

type greedy struct {
	transforms []Transform
}

func Greedy(transforms ...Transform) Sampler {
	return greedy{transforms: transforms}
}

func (s greedy) Sample(logits []float32) (int32, error) {
	logits64 := make([]float64, len(logits))
	for i, v := range logits {
		logits64[i] = float64(v)
	}

	for _, t := range s.transforms {
		logits64 = t.Apply(logits64)
	}

	var maxIdx int
	var maxLogit float64
	for i, logit := range logits64 {
		if logit > maxLogit {
			maxLogit = logit
			maxIdx = i
		}
	}

	if maxLogit == math.Inf(-1) {
		return -1, errors.New("no valid logits found for greedy sampling")
	}

	return int32(maxIdx), nil
}

// TODO(parthsareen): update sampler interface to use json unmarshal https://github.com/ollama/ollama/issues/9278
func NewSampler(temperature float32, topK int, topP float32, minP float32, seed int) (Sampler, error) {
	transforms := []Transform{}
	if temperature < 0 || temperature > 2 {
		return nil, errors.New("temperature must be between 0 and 2")
	}

	if temperature != 0 {
		transforms = append(transforms, Temperature(temperature))
	}

	if topK != 0 {
		if topK <= 0 {
			return nil, errors.New("topK must be greater than 0")
		}
		transforms = append(transforms, TopK(topK))
	}

	if topP != 0 {
		if topP < 0 || topP >= 1 {
			return nil, errors.New("topP must be between 0 and 1")
		}
		transforms = append(transforms, TopP(topP))
	}

	if minP != 0 {
		if minP < 0 || minP >= 1 {
			return nil, errors.New("minP must be between 0 and 1")
		}
		transforms = append(transforms, MinP(minP))
	}

	if len(transforms) == 0 {
		return nil, errors.New("at least one transform is required")
	}

	if temperature == 0 {
		return Greedy(transforms...), nil
	}

	if seed != 0 {
		seed64 := uint64(seed)
		return Weighted(&seed64, transforms...), nil
	}
	return Weighted(nil, transforms...), nil
}
