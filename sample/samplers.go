package sample

import (
	"errors"
	"math"
	"math/rand/v2"
	"slices"

	"github.com/ollama/ollama/llama"
	"github.com/ollama/ollama/tokenizer"
)

// token represents information about a single token during sampling
type token struct {
	id    int32   // The token's unique identifier
	value float32 // The raw logit or probability from the model
}

const DefaultPenaltyLookback = 64

type Sampler struct {
	rng         *rand.Rand
	topK        int
	topP        float32
	minP        float32
	temperature float32
	repeat      float32
	presence    float32
	frequency   float32
	history     []int32
	grammar     *GrammarSampler
}

func (s *Sampler) Reset() {
	s.history = s.history[:0]
}

func (s *Sampler) Accept(token int32) {
	s.history = append(s.history, token)
	if len(s.history) > DefaultPenaltyLookback {
		copy(s.history, s.history[len(s.history)-DefaultPenaltyLookback:])
		s.history = s.history[:DefaultPenaltyLookback]
	}
}

func (s *Sampler) Sample(logits []float32) (int32, error) {
	if len(logits) == 0 {
		return -1, errors.New("sample: no logits provided to sample")
	}

	counts := tokenCounts(s.history, len(logits))

	tokens := make([]token, len(logits))
	for i := range logits {
		value := logits[i]
		if count := counts[int32(i)]; count > 0 {
			value = applyPenalty(value, count, s.repeat, s.presence, s.frequency)
		}

		tokens[i].id = int32(i)
		tokens[i].value = value
	}

	t, err := s.sample(tokens)
	if err != nil {
		return -1, err
	}

	if s.grammar != nil {
		// optimization: first check if the max logit is accepted by the grammar
		// if the max logit is rejected, apply the grammar to all logits (slower)
		top := []token{t}
		s.grammar.Apply(top)
		if !math.IsInf(float64(top[0].value), -1) {
			s.grammar.Accept(top[0].id)
			return top[0].id, nil
		}

		// since .sample has side effects of modifying the tokens
		// we need to reset them before applying the grammar and
		// sampling again
		for i := range logits {
			value := logits[i]
			if count := counts[int32(i)]; count > 0 {
				value = applyPenalty(value, count, s.repeat, s.presence, s.frequency)
			}
			tokens[i].id = int32(i)
			tokens[i].value = value
		}
		s.grammar.Apply(tokens)
		t, err = s.sample(tokens)
		if err != nil {
			return -1, err
		}
		s.grammar.Accept(t.id)
	}

	return t.id, nil
}

// greedy returns the highest probability token from the tokens
func greedy(tokens []token) token {
	max := tokens[0]
	for i := 1; i < len(tokens); i++ {
		if tokens[i].value > max.value {
			max = tokens[i]
		}
	}

	return max
}

// sample returns the highest probability token from the tokens
// given sampler parameters. It also has side effects of modifying the tokens
func (s *Sampler) sample(tokens []token) (token, error) {
	if s.temperature == 0 {
		return greedy(tokens), nil
	}

	// topK also sorts the tokens in descending order of logits
	tokens = topK(tokens, s.topK)

	// scale and normalize the tokens in place
	temperature(tokens, s.temperature)
	softmax(tokens)

	tokens = topP(tokens, s.topP)
	tokens = minP(tokens, s.minP)

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

	idx, _ := slices.BinarySearchFunc(tokens, r, func(token token, target float32) int {
		if token.value < target {
			return -1
		}
		return 1
	})

	if math.IsNaN(float64(sum)) {
		return token{}, errors.New("sample: logits sum to NaN, check model output")
	}
	return tokens[idx], nil
}

// TODO(parthsareen): update sampler interface to use json unmarshal https://github.com/ollama/ollama/issues/9278
func NewSampler(temperature float32, topK int, topP float32, minP float32, repeatPenalty float32, presencePenalty float32, frequencyPenalty float32, seed int, grammar *GrammarSampler) Sampler {
	var rng *rand.Rand
	if seed != -1 {
		// PCG requires two parameters: sequence and stream
		// Use original seed for sequence
		sequence := uint64(seed)
		// Use golden ratio hash to generate statistically independent seeds
		rng = rand.New(rand.NewPCG(sequence, sequence^0x9E3779B9))
	}
	if temperature < 0.0 {
		temperature = 0.0
	}

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

	if repeatPenalty <= 0 {
		repeatPenalty = 1.0
	}

	return Sampler{
		rng:         rng,
		topK:        topK,
		topP:        topP,
		minP:        minP,
		temperature: temperature,
		repeat:      repeatPenalty,
		presence:    presencePenalty,
		frequency:   frequencyPenalty,
		grammar:     grammar,
	}
}

type GrammarSampler struct {
	grammar *llama.Grammar
}

func NewGrammarSampler(tok tokenizer.Tokenizer, grammarStr string) (*GrammarSampler, error) {
	vocabIds := make([]uint32, len(tok.Vocabulary().Values))
	pieces := make([]string, len(tok.Vocabulary().Values))
	for i := range tok.Vocabulary().Values {
		pieces[i], _ = tok.Decode([]int32{int32(i)})
		vocabIds[i] = uint32(i)
	}

	grammar := llama.NewGrammar(grammarStr, vocabIds, pieces, tok.Vocabulary().EOS)
	if grammar == nil {
		return nil, errors.New("sample: failed to initialize grammar")
	}

	return &GrammarSampler{grammar: grammar}, nil
}

func (g *GrammarSampler) Apply(tokens []token) {
	tds := make([]llama.TokenData, len(tokens))
	for i, token := range tokens {
		tds[i].ID = token.id
		tds[i].Logit = token.value
	}
	g.grammar.Apply(tds)

	for i := range tokens {
		tokens[i].value = tds[i].Logit
	}
}

func (g *GrammarSampler) Accept(token int32) {
	g.grammar.Accept(token)
}

func (g *GrammarSampler) Free() {
	g.grammar.Free()
}
