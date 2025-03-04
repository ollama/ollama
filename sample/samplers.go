package sample

import (
	"errors"
	"math/rand"
)

// Sampler is not thread-safe. Each goroutine should have its own instance.
type Sampler interface {
	Sample([]float32) (int32, error)
}

type tokenInfo struct {
	id    int32
	logit float32
	prob  float32
}

type tokenSliceInfo struct {
	tokens []tokenInfo
	sorted bool
}

type weighted struct {
	r          float32
	transforms []Transform
}

func Weighted(r float32, transforms ...Transform) Sampler {
	return &weighted{
		r:          r,
		transforms: transforms,
	}
}

func (s *weighted) Sample(logits []float32) (int32, error) {
	tokens := make([]tokenInfo, len(logits))
	probs := make([]float32, len(logits))

	for i, v := range logits {
		tokens[i] = tokenInfo{
			id:    int32(i),
			logit: v,
			prob:  probs[i],
		}
	}

	tokensInfo := tokenSliceInfo{tokens: tokens, sorted: false}
	for _, t := range s.transforms {
		tokensInfo = t.Apply(tokensInfo)
	}

	if len(tokensInfo.tokens) == 0 {
		return -1, errors.New("no valid logits found for weighted sampling")
	}

	// Cumulative distribution function based sampling
	sumProbs := make([]float32, len(tokensInfo.tokens))
	var sum float32
	for i, token := range tokensInfo.tokens {
		sum += token.prob
		sumProbs[i] = sum
	}

	s.r *= sumProbs[len(tokensInfo.tokens)-1]

	// Binary search for the selected index
	left, right := 0, len(tokensInfo.tokens)-1
	for left < right {
		mid := (left + right) / 2
		if sumProbs[mid] < s.r {
			left = mid + 1
		} else {
			right = mid
		}
	}

	return int32(tokensInfo.tokens[left].id), nil
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

	transforms = append(transforms, softmax{})

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

	var seed64 *int64
	if seed != 0 {
		s := int64(seed)
		seed64 = &s
	}

	var r float32
	if seed64 == nil {
		r = rand.Float32()
	} else {
		// Use the seed to initialize a random source
		rng := rand.New(rand.NewSource(*seed64))
		// Increment the seed for next call to ensure different results
		r = rng.Float32()
	}

	return Weighted(r, transforms...), nil
}
