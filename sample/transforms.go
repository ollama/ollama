package sample

import (
	"cmp"
	"math"

	pq "github.com/emirpasic/gods/v2/queues/priorityqueue"
)

type Transform interface {
	Apply([]tokenInfo) []tokenInfo
}

// TODO(parthsareen): potentially cache softmax values
func softmax(logits []float64) []float64 {
	var sum float64
	probs := make([]float64, len(logits))
	for i, v := range logits {
		probs[i] = math.Exp(v)
		sum += probs[i]
	}

	for i := range probs {
		probs[i] /= sum
	}

	return probs
}

type Temperature float64

func (t Temperature) Apply(tokens []tokenInfo) []tokenInfo {
	temp := math.Max(float64(t), 1e-7)

	// subtracting max logit to avoid under/overflow
	maxLogit := math.Inf(-1)
	for _, token := range tokens {
		if token.logit > maxLogit {
			maxLogit = token.logit
		}
	}

	for i := range tokens {
		tokens[i].logit = (tokens[i].logit - maxLogit) / temp
	}

	return tokens
}

type logitMap struct {
	index int
	logit float64
}

type TopK int

func (k TopK) Apply(tokens []tokenInfo) []tokenInfo {
	if int(k) >= len(tokens) {
		return tokens
	}
	q := pq.NewWith(func(a, b tokenInfo) int {
		return -cmp.Compare(a.logit, b.logit)
	})

	// TODO: can do a sort instead and make use of the sorted in other transforms
	validTokens := make([]tokenInfo, 0, int(k))
	for _, token := range tokens {
		q.Enqueue(token)
	}
	for range k {
		token, _ := q.Dequeue()
		validTokens = append(validTokens, token)
	}

	return validTokens
}

type TopP float64

func (p TopP) Apply(tokens []tokenInfo) []tokenInfo {
	// probs := softmax(logits)
	indices := make([]int, len(tokens))
	for i := range indices {
		indices[i] = i
	}

	// sort in descending order
	// todo: check and see if tokens are sorted
	// slices.SortFunc(indices, func(i, j int) int {
	// 	return -cmp.Compare(tokens[i].prob, tokens[j].prob)
	// })

	var sum float64
	var cutoffIndex int
	for i, idx := range indices {
		sum += tokens[idx].prob
		if sum > float64(p) {
			cutoffIndex = i + 1
			break
		}
	}

	// If we didn't reach the threshold, keep all tokens
	if cutoffIndex == 0 {
		cutoffIndex = len(indices)
	}

	// TODO: this only works if sorted
	tokens = tokens[:cutoffIndex]

	return tokens
}

type MinP float64

// TODO: remove alloc in here
func (p MinP) Apply(tokens []tokenInfo) []tokenInfo {
	// probs := softmax(logits)
	maxProb := math.Inf(-1)
	for _, token := range tokens {
		if token.prob > maxProb {
			maxProb = token.prob
		}
	}

	threshold := maxProb * float64(p)

	newTokens := make([]tokenInfo, 0, len(tokens))
	for i, token := range tokens {
		if token.prob >= threshold {
			newTokens = append(newTokens, tokens[i])
		}
	}

	return newTokens
}
