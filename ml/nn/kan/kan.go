package kan

import (
	"math"
	"sync"
)

// Layer is a multi-head Geometric KAN layer that replaces softmax in attention.
// It operates on pre-softmax attention logits (QK^T / sqrt(d_k) + mask)
// and produces attention weights that sum to 1 per query position.
//
// Multiple heads cooperate additively in log-space: each head applies its own
// B-spline basis functions with independently normalized coefficients, and
// their outputs are summed before exp-normalize. This allows each head to
// specialize on a different part of the error surface.
//
// New heads are spawned dynamically when loss plateaus (see ShadowTrainer),
// initialized to zero so they don't disrupt existing heads.
type Layer struct {
	Grid  *BSplineGrid
	Heads []*Coefficients
	mu    sync.RWMutex
}

// NewLayer creates a KAN layer with one head initialized to approximate softmax.
func NewLayer(cfg Config) *Layer {
	grid := NewBSplineGrid(cfg.Order, cfg.NumBasis, cfg.GridMin, cfg.GridMax)
	return &Layer{
		Grid:  grid,
		Heads: []*Coefficients{NewCoefficients(grid)},
	}
}

// NewLayerFromWeights creates a KAN layer with pre-trained weights.
// If len(weights) == numBasis, creates a single head.
// If len(weights) == N * numBasis, creates N heads.
func NewLayerFromWeights(cfg Config, weights []float32) *Layer {
	grid := NewBSplineGrid(cfg.Order, cfg.NumBasis, cfg.GridMin, cfg.GridMax)

	numBasis := cfg.NumBasis
	numHeads := len(weights) / numBasis
	if numHeads < 1 {
		numHeads = 1
	}

	heads := make([]*Coefficients, numHeads)
	for h := 0; h < numHeads; h++ {
		start := h * numBasis
		end := start + numBasis
		if end > len(weights) {
			end = len(weights)
		}
		heads[h] = NewCoefficientsFromWeights(weights[start:end])
	}

	return &Layer{
		Grid:  grid,
		Heads: heads,
	}
}

// NumHeads returns the current number of cooperative heads.
func (l *Layer) NumHeads() int {
	l.mu.RLock()
	defer l.mu.RUnlock()
	return len(l.Heads)
}

// AddHead spawns a new cooperative head initialized to zero (no-op).
// Returns the new head count.
func (l *Layer) AddHead(numBasis int) int {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.Heads = append(l.Heads, NewZeroCoefficients(numBasis))
	return len(l.Heads)
}

// Forward applies the multi-head Geometric KAN to attention logits (CPU-side).
//
// Input: logits as a flat float32 slice.
//
//	Shape semantics: [seqK * seqQ] flattened, where seqQ is the number of
//	query positions and seqK is the number of key positions.
//
// Output: attention weights (same shape), each row sums to 1.
//
// The computation for each scalar logit x:
//
//	f(x) = Σ_head Σ_i (c_hi * B_i(x))
//
// Each head contributes additively in log-space. After computing the combined
// KAN values, each row is normalized via stable exp + L1 normalization
// (same as softmax).
func (l *Layer) Forward(logits []float32, seqK, seqQ int) []float32 {
	l.mu.RLock()
	// Snapshot all heads' coefficients
	allCoeffs := make([][]float32, len(l.Heads))
	for h, head := range l.Heads {
		allCoeffs[h] = make([]float32, len(head.Weights))
		copy(allCoeffs[h], head.Weights)
	}
	l.mu.RUnlock()

	// Phase 1: Evaluate multi-head B-spline KAN for each logit
	// Each head contributes additively in log-space
	rawScores := make([]float32, len(logits))
	for i, x := range logits {
		basis := l.Grid.Evaluate(x)
		var val float32
		for _, coeffs := range allCoeffs {
			for j, b := range basis {
				if j < len(coeffs) {
					val += coeffs[j] * b
				}
			}
		}
		rawScores[i] = val
	}

	// Phase 2: Row-wise numerically stable exp + normalization
	output := make([]float32, len(logits))
	for q := 0; q < seqQ; q++ {
		rowStart := q * seqK
		rowEnd := rowStart + seqK
		if rowEnd > len(rawScores) {
			rowEnd = len(rawScores)
		}

		// Find row max
		rowMax := rawScores[rowStart]
		for i := rowStart + 1; i < rowEnd; i++ {
			if rawScores[i] > rowMax {
				rowMax = rawScores[i]
			}
		}

		// Exp with max subtracted + sum
		var rowSum float64
		for i := rowStart; i < rowEnd; i++ {
			v := math.Exp(float64(rawScores[i] - rowMax))
			output[i] = float32(v)
			rowSum += v
		}

		// Normalize
		if rowSum > 1e-10 {
			invSum := float32(1.0 / rowSum)
			for i := rowStart; i < rowEnd; i++ {
				output[i] *= invSum
			}
		}
	}

	return output
}

// ForwardSingleRow applies the KAN to a single row of logits and normalizes.
func (l *Layer) ForwardSingleRow(logits []float32) []float32 {
	return l.Forward(logits, len(logits), 1)
}

// EvaluateRaw computes the raw multi-head B-spline transform for a single point.
// Returns Σ_head Σ_i (c_hi * B_i(x)) without any exp or normalization.
func (l *Layer) EvaluateRaw(x float32) float32 {
	l.mu.RLock()
	allCoeffs := make([][]float32, len(l.Heads))
	for h, head := range l.Heads {
		allCoeffs[h] = make([]float32, len(head.Weights))
		copy(allCoeffs[h], head.Weights)
	}
	l.mu.RUnlock()

	basis := l.Grid.Evaluate(x)
	var val float32
	for _, coeffs := range allCoeffs {
		for j, b := range basis {
			if j < len(coeffs) {
				val += coeffs[j] * b
			}
		}
	}
	return val
}

// UpdateCoefficients thread-safely replaces all heads' coefficients.
// The flat slice is split into chunks of numBasis, one per head.
// Each head's coefficients are independently normalized.
func (l *Layer) UpdateCoefficients(newWeights []float32) {
	l.mu.Lock()
	defer l.mu.Unlock()

	numBasis := l.Grid.NumBasis
	for h, head := range l.Heads {
		start := h * numBasis
		end := start + numBasis
		if start >= len(newWeights) {
			break
		}
		if end > len(newWeights) {
			end = len(newWeights)
		}
		copy(head.Weights, newWeights[start:end])
		head.NormalizeAndRedistribute()
	}
}

// GetCoefficients returns a copy of all heads' coefficients concatenated.
func (l *Layer) GetCoefficients() []float32 {
	l.mu.RLock()
	defer l.mu.RUnlock()

	total := 0
	for _, head := range l.Heads {
		total += len(head.Weights)
	}

	w := make([]float32, 0, total)
	for _, head := range l.Heads {
		w = append(w, head.Weights...)
	}
	return w
}
