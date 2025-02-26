package sample

import (
	"errors"

	"golang.org/x/exp/rand"
	"gonum.org/v1/gonum/stat/sampleuv"
)

type Sampler interface {
	Sample([]float32) (int32, error)
}

type tokenInfo struct {
	id    int
	logit float64
	prob  float64
}

// TODO: see if this is needed to check if things are sorted or not
type tokenSliceInfo struct {
	tokens []tokenInfo
	sorted bool
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

	probs := softmax(logits64)

	tokens := make([]tokenInfo, len(logits))
	for i, v := range logits {
		tokens[i] = tokenInfo{
			id:    i,
			logit: float64(v),
			prob:  probs[i],
		}
	}

	for _, t := range s.transforms {
		tokens = t.Apply(tokens)
	}

	// logitsCopy := make([]float64, 0, len(logits))
	// indices := make([]int, 0, len(logits))
	// for i, logit := range logits64 {
	// 	if !math.IsInf(logit, -1) {
	// 		logitsCopy = append(logitsCopy, logit)
	// 		indices = append(indices, i)
	// 	}
	// }

	if len(tokens) == 0 {
		return -1, errors.New("no valid logits found for weighed sampling")
	}

	filteredProbs := make([]float64, len(tokens))
	indices := make([]int, len(tokens))
	for i, token := range tokens {
		filteredProbs[i] = token.prob
		indices[i] = token.id
	}

	// probs := softmax(logitsCopy)
	w := sampleuv.NewWeighted(filteredProbs, s.src)
	if idx, ok := w.Take(); ok {
		return int32(indices[idx]), nil
	}
	return -1, errors.New("weighted sampler failed, no valid token found")
}

type greedy struct{}

func Greedy() Sampler {
	return greedy{}
}

// Sample returns the index of the maximum value in logits.
func (s greedy) Sample(logits []float32) (int32, error) {
	if len(logits) == 0 {
		return -1, errors.New("no logits provided for greedy sampling")
	}

	maxIdx := 0
	for i := range logits {
		if logits[i] > logits[maxIdx] {
			maxIdx = i
		}
	}

	return int32(maxIdx), nil
}

// TODO(parthsareen): update sampler interface to use json unmarshal https://github.com/ollama/ollama/issues/9278
func NewSampler(temperature float32, topK int, topP float32, minP float32, seed int) (Sampler, error) {
	if temperature == 0 {
		return Greedy(), nil
	}

	if temperature < 0 || temperature > 2 {
		return nil, errors.New("temperature must be between 0 and 2")
	}

	transforms := []Transform{Temperature(temperature)}

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

	if seed >= 0 {
		seed64 := uint64(seed)
		return Weighted(&seed64, transforms...), nil
	}
	return Weighted(nil, transforms...), nil
}
