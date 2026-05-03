package sample

import (
	"math"
	"slices"
)

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
	if k <= 0 || k >= len(ts) {
		return ts
	}
	return ts[:k]
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
