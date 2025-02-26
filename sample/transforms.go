package sample

import (
	"cmp"
	"math"
	"slices"

	pq "github.com/emirpasic/gods/v2/queues/priorityqueue"
)

type Transform interface {
	Apply(tokenSliceInfo) tokenSliceInfo
}

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

func (t Temperature) Apply(ts tokenSliceInfo) tokenSliceInfo {
	if t == 1 {
		return ts
	}

	temp := math.Max(float64(t), 1e-7)

	// subtracting max logit to avoid under/overflow
	maxLogit := math.Inf(-1)
	for _, token := range ts.tokens {
		if token.logit > maxLogit {
			maxLogit = token.logit
		}
	}

	for i := range ts.tokens {
		ts.tokens[i].logit = (ts.tokens[i].logit - maxLogit) / temp
	}

	return ts
}

type logitMap struct {
	index int
	logit float64
}

type TopK int

func (k TopK) Apply(ts tokenSliceInfo) tokenSliceInfo {
	if int(k) >= len(ts.tokens) {
		return ts
	}
	q := pq.NewWith(func(a, b tokenInfo) int {
		return -cmp.Compare(a.logit, b.logit)
	})

	validTokens := make([]tokenInfo, 0, int(k))
	for _, token := range ts.tokens {
		if q.Size() < int(k) {
			q.Enqueue(token)
		} else if min, ok := q.Peek(); ok && token.logit > min.logit {
			q.Dequeue()
			q.Enqueue(token)
		}
	}

	for !q.Empty() {
		token, _ := q.Dequeue()
		validTokens = append([]tokenInfo{token}, validTokens...)
	}

	return tokenSliceInfo{tokens: validTokens, sorted: true}
}

type TopP float64

func (p TopP) Apply(ts tokenSliceInfo) tokenSliceInfo {
	indices := make([]int, len(ts.tokens))
	for i := range indices {
		indices[i] = i
	}

	if !ts.sorted {
		// sort in descending order
		slices.SortFunc(indices, func(i, j int) int {
			return cmp.Compare(ts.tokens[j].prob, ts.tokens[i].prob)
		})
	}

	newTokens := make([]tokenInfo, 0, len(ts.tokens))
	var sum float64
	for _, idx := range indices {
		sum += ts.tokens[idx].prob
		newTokens = append(newTokens, ts.tokens[idx])
		if sum > float64(p) {
			break
		}
	}

	ts.tokens = newTokens
	ts.sorted = true

	return ts
}

type MinP float64

func (p MinP) Apply(ts tokenSliceInfo) tokenSliceInfo {
	maxProb := math.Inf(-1)
	for _, token := range ts.tokens {
		if token.prob > maxProb {
			maxProb = token.prob
		}
	}

	threshold := maxProb * float64(p)

	newTokens := make([]tokenInfo, 0, len(ts.tokens))
	for i, token := range ts.tokens {
		if token.prob >= threshold {
			newTokens = append(newTokens, ts.tokens[i])
		}
	}

	return tokenSliceInfo{tokens: newTokens, sorted: ts.sorted}
}
