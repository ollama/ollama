package sample

import (
	"container/heap"
	"math"
	"math/rand"
	"slices"
)

// tokenHeap implements heap.Interface and holds tokens as a min-heap to track k largest elements
type tokenHeap []token

func (h tokenHeap) Len() int           { return len(h) }
func (h tokenHeap) Less(i, j int) bool { return h[i].value < h[j].value }
func (h tokenHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }

func (h *tokenHeap) Push(x any) {
	*h = append(*h, x.(token))
}

func (h *tokenHeap) Pop() any {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[0 : n-1]
	return x
}

// topK limits the number of tokens considered to the k highest logits
func topK(ts []token, k int) []token {
	if k >= len(ts) || k <= 0 {
		slices.SortFunc(ts, func(a, b token) int {
			switch {
			case a.value < b.value:
				return 1
			case a.value > b.value:
				return -1
			default:
				return 0
			}
		})
		return ts
	}

	// Initialize min-heap with first k elements
	h := make(tokenHeap, k)
	copy(h, ts[:k])
	heap.Init(&h)

	// Process remaining elements
	for i := k; i < len(ts); i++ {
		if ts[i].value > h[0].value {
			heap.Pop(&h)
			heap.Push(&h, ts[i])
		}
	}

	// Convert heap to sorted slice in descending order
	result := make([]token, len(h))
	for i := k - 1; i >= 0; i-- {
		result[i] = heap.Pop(&h).(token)
	}

	return result
}

// topP limits tokens to those with cumulative probability p
func topP(ts []token, p float32) []token {
	if p == 1.0 {
		return ts
	}

	// Find cutoff index where cumulative sum exceeds p
	var sum float32
	for i, t := range ts {
		sum += t.value
		if sum > float32(p) {
			ts = ts[:i+1]
			return ts
		}
	}

	return ts
}

// minP limits tokens to those with cumulative probability p
func minP(ts []token, p float32) []token {
	if p == 1.0 {
		return ts
	}

	maxProb := float32(math.Inf(-1))
	for _, token := range ts {
		if token.value > maxProb {
			maxProb = token.value
		}
	}

	threshold := maxProb * float32(p)

	// Filter tokens in-place
	validTokens := ts[:0]
	for i, token := range ts {
		if token.value >= threshold {
			validTokens = append(validTokens, ts[i])
		}
	}

	ts = validTokens
	return ts
}

func temperature(ts []token, temp float32) {
	for i := range ts {
		ts[i].value /= temp
	}
}

func softmax(ts []token) {
	if len(ts) == 0 {
		return
	}

	// Find max logit for numerical stability
	maxLogit := ts[0].value
	for _, t := range ts {
		if t.value > maxLogit {
			maxLogit = t.value
		}
	}

	// Compute exp(logit - maxLogit) and sum them
	var sumExp float32
	for i, t := range ts {
		expVal := float32(math.Exp(float64(t.value - maxLogit)))
		ts[i].value = expVal
		sumExp += expVal
	}

	// Normalize probabilities
	for i := range ts {
		ts[i].value /= sumExp
	}
}

// applyDist selects a token based on probabilities and seed
func dist(ts []token, seed int64) int {
	rng := rand.New(rand.NewSource(seed))

	cdf := make([]float32, len(ts))
	var cumSum float32
	for i, t := range ts {
		cumSum += t.value
		cdf[i] = cumSum
	}

	r := rng.Float32() * cumSum

	// Select token based on CDF
	for i, probSum := range cdf {
		if r < probSum {
			return i
		}
	}

	return len(ts) - 1
}
