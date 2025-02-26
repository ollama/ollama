package sample

import (
	"cmp"
	"math"
	"slices"
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

	// tokens := make([]tokenInfo, len(ts.tokens))
	// copy(tokens, ts.tokens)
	tokens := ts.tokens

	// Partial sort to get top-k tokens
	partialSort(tokens, int(k), func(a, b tokenInfo) bool {
		return a.logit > b.logit // Sort in descending order
	})

	return tokenSliceInfo{tokens: tokens[:int(k)], sorted: true}
}

// siftDown implements the sift-down operation for a heap
func siftDown(tokens []tokenInfo, start, end int, less func(a, b tokenInfo) bool) {
	current := start
	for {
		// Calculate child indices
		child1 := 2*current + 1
		child2 := 2*current + 2

		// Find the largest/smallest child based on comparator
		largest := current
		if child1 < end && less(tokens[child1], tokens[largest]) {
			largest = child1
		}
		if child2 < end && less(tokens[child2], tokens[largest]) {
			largest = child2
		}

		// If current is already in the right position, we're done
		if largest == current {
			break
		}

		// Swap and continue sifting down
		tokens[current], tokens[largest] = tokens[largest], tokens[current]
		current = largest
	}
}

// partialSort sorts the first k elements of the slice
func partialSort(tokens []tokenInfo, k int, less func(a, b tokenInfo) bool) {
	// Build heap with first k elements
	for i := k/2 - 1; i >= 0; i-- {
		siftDown(tokens, i, k, less)
	}

	// For each remaining element, if it's "better" than the root,
	// replace root and sift down to maintain heap property
	for i := k; i < len(tokens); i++ {
		if less(tokens[i], tokens[0]) {
			tokens[0] = tokens[i]
			siftDown(tokens, 0, k, less)
		}
	}

	// Sort the heap (in-place)
	for i := k - 1; i > 0; i-- {
		tokens[0], tokens[i] = tokens[i], tokens[0]
		siftDown(tokens, 0, i, less)
	}
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
