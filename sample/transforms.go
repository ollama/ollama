package sample

import (
	"math"
	"slices"
)

func softmax(ts []logit) []logit {
	var sum float32
	for i, v := range ts {
		ts[i].value = float32(math.Exp(float64(v.value)))
		sum += ts[i].value
	}

	for i := range ts {
		ts[i].value /= sum
	}

	return ts
}

func temperature(ti []logit, t float32) []logit {
	if t == 1 {
		return ti
	}

	temp := max(t, 1e-7)
	maxLogit := float32(math.Inf(-1))
	for _, token := range ti {
		if token.value > maxLogit {
			maxLogit = token.value
		}
	}

	// subtracting max logit to avoid under/overflow
	for i := range ti {
		ti[i].value = (ti[i].value - maxLogit) / temp
	}

	return ti
}

// siftDown maintains a min-heap property by recursively moving larger elements down the heap.
//
// The heap is represented as an array where for any node at index i:
// - Left child is at index 2i + 1
// - Right child is at index 2i + 2
// - Parent is at index (i-1)/2
//
// The function compares a node with its children and:
// 1. Finds the smallest value between the node and its children
// 2. If the node is not the smallest, swaps it with its smallest child
// 3. Continues this process down the affected path until the min-heap property is restored
func siftDown(data []logit, start, end int) {
	root := start
	for {
		child := 2*root + 1
		if child >= end {
			break
		}
		// Find smaller child (we want min heap)
		if child+1 < end && data[child+1].value < data[child].value {
			child++
		}
		// Exit if root is already smaller than children
		if data[root].value <= data[child].value {
			break
		}
		// Swap with smaller child and continue
		data[root], data[child] = data[child], data[root]
		root = child
	}
}

// topK limits the number of tokens considered to the k highest logits
func topK(ts []logit, k int) []logit {
	if k >= len(ts) {
		return ts
	}
	// Heapify + siftDown - O(nlog(k))
	// Build min-heap of first k elements
	heap := ts[:k]
	for i := k/2 - 1; i >= 0; i-- {
		siftDown(heap, i, k)
	}

	// Process remaining elements - if larger than heap root, replace root
	for i := k; i < len(ts); i++ {
		if ts[i].value > heap[0].value {
			heap[0] = ts[i]
			siftDown(heap, 0, k)
		}
	}

	slices.Reverse(heap)

	ts = heap
	return ts
}

// topP limits tokens to those with cumulative probability p
func topP(ts []logit, p float32) []logit {
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
func minP(ts []logit, p float32) []logit {
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

// Conting sort implementation to sort tokens by logits
func sortLogits(tokens []logit) {
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
	output := make([]logit, len(tokens))
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
