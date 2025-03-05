package sample

import (
	"errors"
	"math/rand/v2"
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
	sum    float32
}

type weighted struct {
	rng        *rand.Rand
	transforms []transform
}

func Weighted(rng *rand.Rand, transforms ...transform) Sampler {
	return &weighted{
		rng:        rng,
		transforms: transforms,
	}
}

func (s *weighted) Sample(logits []float32) (int32, error) {
	tokens := make([]tokenInfo, len(logits))
	for i, v := range logits {
		tokens[i].id = int32(i)
		tokens[i].logit = v
	}

	tokensInfo := tokenSliceInfo{tokens: tokens, sorted: false}
	for _, t := range s.transforms {
		tokensInfo = t.Apply(tokensInfo)
	}

	if len(tokensInfo.tokens) == 0 {
		return -1, errors.New("no valid logits found for weighted sampling")
	}

	var r float32
	if s.rng != nil {
		r = s.rng.Float32()
	} else {
		r = rand.Float32()
	}
	r *= tokensInfo.sum

	// Binary search for the selected index
	left, right := 0, len(tokensInfo.tokens)-1
	for left < right {
		mid := (left + right) / 2
		if tokensInfo.tokens[mid].prob < r {
			left = mid + 1
		} else {
			right = mid
		}
	}

	return tokensInfo.tokens[left].id, nil
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
	maxVal := logits[0]
	for i := 1; i < len(logits); i++ {
		if logits[i] > maxVal {
			maxVal = logits[i]
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

	transforms := []transform{}
	if topK > 0 {
		transforms = append(transforms, TopK(topK))
	} else {
		transforms = append(transforms, sortTokens{})
	}

	if temperature < 0 || temperature > 2 {
		return nil, errors.New("temperature must be between 0 and 2")
	}

	// tokens must be sorted by logits before next steps
	transforms = append(transforms, Temperature(temperature), softmax{})

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

	var rng *rand.Rand
	if seed != -1 {
		bits := uint64(float64(seed) * float64(1<<32))
		rng = rand.New(rand.NewPCG(bits, bits))
	}

	return Weighted(rng, transforms...), nil
}
