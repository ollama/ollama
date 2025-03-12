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
func (h tokenHeap) Less(i, j int) bool { return h[i].value < h[j].value } // Use < for min-heap to track largest elements
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
	if k >= len(ts) {
		sortLogits(ts)
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
	result := make([]token, k)
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

// partialSortLogits uses quickselect to efficiently find and sort the top n tokens
func partialSortLogits(ts []token, n int) []token {
	if n >= len(ts) {
		n = len(ts)
	}

	left, right := 0, len(ts)-1
	target := n - 1

	// Quickselect algorithm to partition array around pivot
	for left < right {
		// Choose middle element as pivot and move it to the end
		pivot := left + (right-left)/2
		ts[pivot], ts[right] = ts[right], ts[pivot]

		// storeIndex tracks where to put next element greater than pivot
		storeIndex := left
		pivotValue := ts[right].value

		// Partition array into elements >= pivot and < pivot
		// Elements >= pivot go to the left side
		for i := left; i < right; i++ {
			if ts[i].value >= pivotValue {
				ts[storeIndex], ts[i] = ts[i], ts[storeIndex]
				storeIndex++
			}
		}

		// Move pivot to its final position
		ts[right], ts[storeIndex] = ts[storeIndex], ts[right]

		// If pivot is at target position, we're done
		// Otherwise recursively partition the half containing target
		if storeIndex == target {
			break
		} else if storeIndex < target {
			left = storeIndex + 1 // Target is in right half
		} else {
			right = storeIndex - 1 // Target is in left half
		}
	}

	// Sort just the top n elements in descending order
	slices.SortFunc(ts[:n], func(a, b token) int {
		if a.value > b.value {
			return -1
		}
		if a.value < b.value {
			return 1
		}
		return 0
	})

	return ts[:n]
}

// sortLogits uses partialSortLogits to efficiently sort tokens
// It sorts approximately sqrt(len(tokens)) elements which balances
// between having enough tokens for sampling while avoiding full sort
func sortLogits(ts []token) {
	// Use sqrt of token length as a heuristic for partial sort size
	// This provides a good balance between performance and having enough tokens
	n := int(math.Sqrt(float64(len(ts)))) + 1

	// Ensure we have at least 100 tokens and at most 1000
	switch {
	case n < 100:
		n = 100
	case n > 1000:
		n = 1000
	}

	partialSortLogits(ts, n)
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
