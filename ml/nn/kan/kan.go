package kan

import (
	"math"
	"sync"
)

// Layer is a single Geometric KAN layer that replaces softmax in attention.
// It operates on pre-softmax attention logits (QK^T / sqrt(d_k) + mask)
// and produces attention weights that sum to 1 per query position.
//
// The KAN applies learnable B-spline basis functions to each logit value,
// with coefficients managed by the geometric mean normalization scheme.
type Layer struct {
	Grid         *BSplineGrid
	Coefficients *Coefficients
	mu           sync.RWMutex
}

// NewLayer creates a KAN layer initialized to approximate softmax behavior.
func NewLayer(cfg Config) *Layer {
	grid := NewBSplineGrid(cfg.Order, cfg.NumBasis, cfg.GridMin, cfg.GridMax)
	return &Layer{
		Grid:         grid,
		Coefficients: NewCoefficients(grid),
	}
}

// NewLayerFromWeights creates a KAN layer with pre-trained weights.
func NewLayerFromWeights(cfg Config, weights []float32) *Layer {
	grid := NewBSplineGrid(cfg.Order, cfg.NumBasis, cfg.GridMin, cfg.GridMax)
	return &Layer{
		Grid:         grid,
		Coefficients: NewCoefficientsFromWeights(weights),
	}
}

// Forward applies the Geometric KAN to attention logits (CPU-side).
//
// Input: logits as a flat float32 slice (the pre-softmax QK^T scores for one head).
//        Shape semantics: [seqK * seqQ] flattened, where seqQ is the number of
//        query positions and seqK is the number of key positions.
//        seqK is the "row" dimension -- each row of seqK values gets normalized.
//
// Output: attention weights (same shape), each row sums to 1.
//
// The computation for each scalar logit x:
//   kan(x) = sum_i(c_i * B_i(x))
// where B_i are cubic B-spline basis functions and c_i are the
// geometrically-normalized coefficients.
//
// After computing KAN values, each row is normalized via L1 normalization
// (divide by sum of absolute values) to maintain the probabilistic
// interpretation of attention weights, then clamped to [0, 1].
func (l *Layer) Forward(logits []float32, seqK, seqQ int) []float32 {
	l.mu.RLock()
	coeffs := make([]float32, len(l.Coefficients.Weights))
	copy(coeffs, l.Coefficients.Weights)
	l.mu.RUnlock()

	output := make([]float32, len(logits))

	// For each element, evaluate the B-spline KAN
	for i, x := range logits {
		basis := l.Grid.Evaluate(x)
		var val float32
		for j, b := range basis {
			if j < len(coeffs) {
				val += coeffs[j] * b
			}
		}
		// Apply exp to maintain non-negativity (KAN learns in log-space effectively)
		output[i] = float32(math.Exp(float64(val)))
	}

	// Row-wise L1 normalization: each row of seqK values sums to 1
	// This preserves the attention weight semantics
	for q := 0; q < seqQ; q++ {
		rowStart := q * seqK
		rowEnd := rowStart + seqK
		if rowEnd > len(output) {
			rowEnd = len(output)
		}

		var rowSum float32
		for i := rowStart; i < rowEnd; i++ {
			rowSum += output[i]
		}

		if rowSum > 1e-10 {
			invSum := 1.0 / rowSum
			for i := rowStart; i < rowEnd; i++ {
				output[i] *= invSum
			}
		}
	}

	return output
}

// ForwardSingleRow applies the KAN to a single row of logits and normalizes.
// This is the hot path used after graduation (replacing softmax).
func (l *Layer) ForwardSingleRow(logits []float32) []float32 {
	return l.Forward(logits, len(logits), 1)
}

// UpdateCoefficients thread-safely replaces the coefficients after a training step.
func (l *Layer) UpdateCoefficients(newWeights []float32) {
	l.mu.Lock()
	defer l.mu.Unlock()

	copy(l.Coefficients.Weights, newWeights)
	l.Coefficients.NormalizeAndRedistribute()
}

// GetCoefficients returns a copy of the current coefficients.
func (l *Layer) GetCoefficients() []float32 {
	l.mu.RLock()
	defer l.mu.RUnlock()

	w := make([]float32, len(l.Coefficients.Weights))
	copy(w, l.Coefficients.Weights)
	return w
}
