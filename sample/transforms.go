package sample

import (
	"container/heap"
	"math"
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

// temperature applies scaling and softmax to the logits
func temperature(ts []token, temp float32) []token {
	// Find max logit for numerical stability
	maxLogit := float32(math.Inf(-1))
	for _, t := range ts {
		if t.value > maxLogit {
			maxLogit = t.value
		}
	}

	// Apply temperature and compute exp(x - max)
	temp = max(temp, 1e-7)
	var sum float32
	for i, v := range ts {
		ts[i].value = float32(math.Exp(float64((v.value - maxLogit) / temp)))
		sum += ts[i].value
	}

	// Normalize
	for i := range ts {
		ts[i].value /= sum
	}

	return ts
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

// TODO(parthsareen): possibly replace with simpler implementation https://github.com/ollama/ollama/issues/9584
// sortLogits sorts implementation to sort tokens by logits using counting sort
// counting sort is faster than built-in sort for this use case
func sortLogits(tokens []token) {
	if len(tokens) <= 1 {
		return
	}

	// Find max/min in a single pass
	minLogit, maxLogit := tokens[0].value, tokens[0].value
	for _, t := range tokens[1:] {
		if t.value < minLogit {
			minLogit = t.value
		} else if t.value > maxLogit {
			maxLogit = t.value
		}
	}

	// Calculate scaling to map to uint32 range
	logitRange := maxLogit - minLogit
	if logitRange < 1e-6 {
		return // All values effectively equal
	}

	// Count frequencies directly from tokens
	const maxInt = (1 << 24) - 1 // Use 24 bits for good granularity
	var counts [256]int          // For first byte

	// First pass: count frequencies
	for _, t := range tokens {
		// Map to [0, maxInt] range
		score := min(uint32((t.value-minLogit)*float32(maxInt)/logitRange), maxInt)
		counts[score>>16]++
	}

	// Calculate offsets
	var offset int
	for i := range counts {
		count := counts[i]
		counts[i] = offset
		offset += count
	}

	// Second pass: place elements in correct position
	output := make([]token, len(tokens))
	// Track current positions
	countsCopy := counts

	for i, t := range tokens {
		score := min(uint32((t.value-minLogit)*float32(maxInt)/logitRange), maxInt)

		pos := countsCopy[score>>16]
		countsCopy[score>>16]++
		output[len(tokens)-1-pos] = tokens[i]
	}

	copy(tokens, output)
}
