package sample

import (
	"math"
	"slices"
)

type transform interface {
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

	ts.sum = sum

	return ts
}

type Temperature float64

func (t Temperature) Apply(ts tokenSliceInfo) tokenSliceInfo {
	if t == 1 {
		return ts
	}

	temp := float32(math.Max(float64(t), 1e-7))
	maxLogit := float32(math.Inf(-1))
	for _, token := range ts.tokens {
		if token.logit > maxLogit {
			maxLogit = token.logit
		}
	}

	// subtracting max logit to avoid under/overflow
	for i := range ts.tokens {
		ts.tokens[i].logit = (ts.tokens[i].logit - maxLogit) / temp
	}

	return ts
}

type TopK int

// siftDown maintains min-heap property by pushing larger elements down
func siftDown(data []tokenInfo, start, end int) {
	root := start
	for {
		child := 2*root + 1
		if child >= end {
			break
		}
		// Find smaller child (we want min heap)
		if child+1 < end && data[child+1].logit < data[child].logit {
			child++
		}
		// If root is already smaller than children, we're done
		if data[root].logit <= data[child].logit {
			break
		}
		// Otherwise swap with smaller child and continue
		data[root], data[child] = data[child], data[root]
		root = child
	}
}

func (k TopK) Apply(ts tokenSliceInfo) tokenSliceInfo {
	kk := int(k)
	if kk >= len(ts.tokens) {
		return ts
	}

	// Build min-heap of first k elements
	heap := ts.tokens[:kk]
	for i := kk/2 - 1; i >= 0; i-- {
		siftDown(heap, i, kk)
	}

	// Process remaining elements - if larger than heap root, replace root
	for i := kk; i < len(ts.tokens); i++ {
		if ts.tokens[i].logit > heap[0].logit {
			heap[0] = ts.tokens[i]
			siftDown(heap, 0, kk)
		}
	}

	slices.Reverse(heap)

	ts.tokens = heap
	ts.sorted = true
	return ts
}

type TopP float64

func (p TopP) Apply(ts tokenSliceInfo) tokenSliceInfo {
	// Find cutoff index where cumulative sum exceeds p
	var sum float32
	for i, t := range ts.tokens {
		sum += t.prob
		if sum > float32(p) {
			ts.tokens = ts.tokens[:i+1]
			return ts
		}
	}

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

type sortTokens struct{}

func (s sortTokens) Apply(ts tokenSliceInfo) tokenSliceInfo {
	fastSort(ts.tokens)
	ts.sorted = true
	return ts
}

// Counting sort
func fastSort(tokens []tokenInfo) {
	if len(tokens) <= 1 {
		return
	}

	// Find max/min in a single pass
	minLogit, maxLogit := tokens[0].logit, tokens[0].logit
	for _, t := range tokens[1:] {
		if t.logit < minLogit {
			minLogit = t.logit
		} else if t.logit > maxLogit {
			maxLogit = t.logit
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
		score := uint32((t.logit - minLogit) * float32(maxInt) / logitRange)
		if score > maxInt { // Handle float precision edge cases
			score = maxInt
		}
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
	output := make([]tokenInfo, len(tokens))
	// Track current positions
	countsCopy := counts

	for i, t := range tokens {
		score := uint32((t.logit - minLogit) * float32(maxInt) / logitRange)
		if score > maxInt {
			score = maxInt
		}

		pos := countsCopy[score>>16]
		countsCopy[score>>16]++
		output[len(tokens)-1-pos] = tokens[i]
	}

	copy(tokens, output)
}
