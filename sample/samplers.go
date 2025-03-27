package sample

import (
	"errors"
	"math"
	"math/rand/v2"
	"slices"
	"sync"

	"github.com/ollama/ollama/llama"
)

// token represents information about a single token during sampling
type token struct {
	id    int32   // The token's unique identifier
	value float32 // The raw logit or probability from the model
}

type Sampler struct {
	rng         *rand.Rand
	topK        int
	topP        float32
	minP        float32
	temperature float32
	grammar     *Grammar
}

func (s *Sampler) Sample(logits []float32) (int32, error) {
	if len(logits) == 0 {
		return -1, errors.New("sample: no logits provided to sample")
	}

	tokens := make([]token, len(logits))
	for i := range logits {
		tokens[i].id = int32(i)
		tokens[i].value = logits[i]
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
			tokens[i].id = int32(i)
			tokens[i].value = logits[i]
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
func NewSampler(temperature float32, topK int, topP float32, minP float32, seed int, grammar *Grammar) Sampler {
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

	return Sampler{
		rng:         rng,
		topK:        topK,
		topP:        topP,
		minP:        minP,
		temperature: temperature,
		grammar:     grammar,
	}
}

type Grammar struct {
	vocab   *Vocab
	grammar string
	sampler *llama.Sampler
}

func NewGrammar(vocab *Vocab, grammar string) (*Grammar, error) {
	v, err := vocab.Load()
	if err != nil {
		return nil, err
	}

	return &Grammar{
		vocab:   vocab,
		grammar: grammar,
		sampler: llama.NewGrammarSampler(v, grammar),
	}, nil
}

func (g *Grammar) Apply(tokens []token) {
	tds := make([]llama.TokenData, len(tokens))
	for i, token := range tokens {
		tds[i].Id = token.id
		tds[i].Logit = token.value
	}

	g.sampler.Apply(tds)

	for i := range tokens {
		tokens[i].value = tds[i].Logit
	}
}

func (g *Grammar) Accept(token int32) {
	g.sampler.Accept(token)
}

type Vocab struct {
	once  sync.Once
	vocab *llama.Vocab
	err   error
	path  string
}

func NewVocab(path string) *Vocab {
	return &Vocab{path: path}
}

// Load returns the lazily-loaded vocabulary
func (v *Vocab) Load() (*llama.Vocab, error) {
	v.once.Do(func() {
		vocab, err := llama.LoadVocabFromFile(v.path)
		if err != nil {
			v.err = err
			return
		}
		v.vocab = vocab
	})
	return v.vocab, v.err
}
