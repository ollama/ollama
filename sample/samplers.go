package sample

import (
	"errors"
	"math/rand/v2"
	"slices"
)

// Sampler is not thread-safe. Each goroutine should have its own instance
type Sampler interface {
	Sample([]float32) (int32, error)
}

// logit represents information about a single token during sampling
type logit struct {
	id    int32   // The token's unique identifier
	value float32 // The raw logit or probability from the model
}

type weighted struct {
	rng         *rand.Rand
	tokens      []logit
	topK        int
	topP        float32
	minP        float32
	temperature float32
}

func (s *weighted) Sample(logits []float32) (int32, error) {
	if len(s.tokens) < len(logits) {
		s.tokens = make([]logit, len(logits))
	}

	tokens := s.tokens[:len(logits)]

	for i, v := range logits {
		tokens[i].id = int32(i)
		tokens[i].value = v
	}

	// Tokens are sorted by logits in TopK or SortTokens
	if s.topK > 0 {
		tokens = topK(tokens, s.topK)
	} else {
		sortTokens(tokens)
	}

	tokens = temperature(tokens, s.temperature)
	tokens = softmax(tokens)

	tokens = topP(tokens, s.topP)
	tokens = minP(tokens, s.minP)

	if len(tokens) == 0 {
		return -1, errors.New("no valid logits found for weighted sampling")
	}

	// TODO(parthsareen): fix seeded sampling https://github.com/ollama/ollama/issues/9554
	var r float32
	if s.rng != nil {
		r = s.rng.Float32()
	} else {
		r = rand.Float32()
	}

	// Calculate cumulative sum of probabilities
	var sum float32
	for i := range tokens {
		sum += tokens[i].value
		tokens[i].value = sum
	}
	r *= tokens[len(tokens)-1].value

	idx, _ := slices.BinarySearchFunc(tokens, r, func(token logit, target float32) int {
		// Compare cumulative probabilities
		if token.value < target {
			return -1
		}
		// First token that exceeds target
		return 1
	})

	if idx >= len(tokens) {
		idx = len(tokens) - 1
	}

	return tokens[idx].id, nil
}

type greedy struct{}

// Greedy sample returns the index of the maximum value in logits.
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
func NewSampler(temperature float32, topK int, topP float32, minP float32, seed int) Sampler {
	if temperature == 0 {
		return &greedy{}
	}

	var rng *rand.Rand
	if seed != -1 {
		// PCG requires two parameters: sequence and stream
		// Use original seed for sequence
		sequence := uint64(seed)
		// Use golden ratio hash to generate statistically independent seeds
		rng = rand.New(rand.NewPCG(sequence, sequence^0x9E3779B9))
	}
	temperature = max(temperature, 1)

	if topP < 0.0 {
		topP = 0.0
	}
	if topP >= 1.0 {
		topP = 1.0
	}

	if minP < 0.0 {
		minP = 0.0
	}
	if minP >= 1.0 {
		minP = 1.0
	}

	return &weighted{
		rng:         rng,
		topK:        topK,
		topP:        topP,
		minP:        minP,
		temperature: temperature,
	}
}
