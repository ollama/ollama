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

type softmax struct{}

func (s softmax) Apply(ts tokenSliceInfo) tokenSliceInfo {
	var sum float32
	for i, v := range ts.tokens {
		ts.tokens[i].prob = float32(math.Exp(float64(v.logit)))
		sum += ts.tokens[i].prob
	}

	for i := range ts.tokens {
		ts.tokens[i].prob /= sum
	}

	return ts
}

type Temperature float64

func (t Temperature) Apply(ts tokenSliceInfo) tokenSliceInfo {
	if t == 1 {
		return ts
	}

	temp := float32(math.Max(float64(t), 1e-7))

	// subtracting max logit to avoid under/overflow
	maxLogit := float32(math.Inf(-1))
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
	var sum float32
	for _, idx := range indices {
		sum += ts.tokens[idx].prob
		newTokens = append(newTokens, ts.tokens[idx])
		if sum > float32(p) {
			break
		}
	}

	ts.tokens = newTokens
	ts.sorted = true

	return ts
}

type MinP float32

func (p MinP) Apply(ts tokenSliceInfo) tokenSliceInfo {
	maxProb := float32(math.Inf(-1))
	for _, token := range ts.tokens {
		if token.prob > maxProb {
			maxProb = token.prob
		}
	}

	threshold := maxProb * float32(p)

	// Filter tokens in-place
	validTokens := ts.tokens[:0]
	for i, token := range ts.tokens {
		if token.prob >= threshold {
			validTokens = append(validTokens, ts.tokens[i])
		}
	}

	return tokenSliceInfo{tokens: validTokens, sorted: ts.sorted}
}
