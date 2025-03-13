package sample

import (
	"container/heap"
	"math"
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

// temperature applies scaling to the logits
func temperature(ts []token, temp float32) []token {
	// Ensure temperature clipping near 0 to avoid numerical instability
	temp = max(temp, 1e-7)
	for i := range ts {
		ts[i].value = ts[i].value / temp
	}
	return ts
}

// softmax applies normalization to the logits
func softmax(ts []token) []token {
	// Find max logit for numerical stability
	maxLogit := float32(math.Inf(-1))
	for _, t := range ts {
		if t.value > maxLogit {
			maxLogit = t.value
		}
	}

	// Compute exp(x - max)
	var sum float32
	for i, v := range ts {
		ts[i].value = float32(math.Exp(float64(v.value - maxLogit)))
		sum += ts[i].value
	}

	// exp(x - max) / sum(exp(x - max))
	for i := range ts {
		ts[i].value /= sum
	}

	return ts
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
