package sample

import (
	"cmp"
	"container/heap"
	"math"
	"slices"
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
	// if called after top-k, the tokens are already sorted
	if !ts.sorted {
		slices.SortFunc(ts.tokens, func(i, j tokenInfo) int {
			return cmp.Compare(j.logit, i.logit) // Sort in descending order
		})
		ts.sorted = true
	}

	// subtracting max logit to avoid under/overflow
	for i := range ts.tokens {
		ts.tokens[i].logit = (ts.tokens[i].logit - ts.tokens[0].logit) / temp
	}

	return ts
}

// minHeap implements container/heap.Interface
type minHeap []tokenInfo

func (h minHeap) Len() int           { return len(h) }
func (h minHeap) Less(i, j int) bool { return h[i].logit < h[j].logit }
func (h minHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }
func (h *minHeap) Push(x any)        { *h = append(*h, x.(tokenInfo)) }
func (h *minHeap) Pop() any {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[0 : n-1]
	return x
}

type TopK int

func (k TopK) Apply(ts tokenSliceInfo) tokenSliceInfo {
	kk := int(k)
	if kk >= len(ts.tokens) {
		return ts
	}

	// Create a min-heap with the first k elements
	h := make(minHeap, kk)
	copy(h, ts.tokens[:kk])
	heap.Init(&h)

	// Process remaining elements
	for i := kk; i < len(ts.tokens); i++ {
		if ts.tokens[i].logit > h[0].logit {
			h[0] = ts.tokens[i]
			heap.Fix(&h, 0)
		}
	}

	// Copy back k largest elements
	copy(ts.tokens[:kk], h)

	// Store in descending order
	slices.Reverse(ts.tokens[:kk])

	ts.tokens = ts.tokens[:kk]
	ts.sorted = true
	return ts
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
