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
func temperature(ts []token, temp float32) {
	// Ensure temperature clipping near 0 to avoid numerical instability
	temp = max(temp, 1e-7)
	for i := range ts {
		ts[i].value = ts[i].value / temp
	}
}

// softmax applies normalization to the logits
func softmax(ts []token) {
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
// requires ts to be sorted in descending order of probabilities
func topP(ts []token, p float32) []token {
	if p == 1.0 {
		return ts
	}

	// Find cutoff index where cumulative sum exceeds p
	var sum float32
	for i, t := range ts {
		sum += t.value
		if sum > float32(p) {
			return ts[:i+1]
		}
	}

	return ts
}

// minP filters tokens with probabilities >= p * max_prob
// requires ts to be sorted in descending order of probabilities
func minP(ts []token, p float32) []token {
	maxProb := ts[0].value

	threshold := maxProb * p

	for i, t := range ts {
		if t.value < threshold {
			return ts[:i]
		}
	}
	return ts
}
