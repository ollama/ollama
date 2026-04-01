package kan

import "math"

// BSplineGrid holds a uniform knot vector and evaluates cubic B-spline basis functions.
// All computation is done in pure Go on CPU since the coefficient tensors are tiny.
type BSplineGrid struct {
	Order    int
	NumBasis int
	Knots    []float32 // length = NumBasis + Order + 1
}

// NewBSplineGrid creates a uniform B-spline grid with the given parameters.
// For cubic B-splines (order=3) with numBasis=8, this creates 12 knots
// uniformly spaced in [gridMin, gridMax] with extended boundary knots.
func NewBSplineGrid(order, numBasis int, gridMin, gridMax float32) *BSplineGrid {
	numInterior := numBasis - order
	if numInterior < 1 {
		numInterior = 1
	}

	step := (gridMax - gridMin) / float32(numInterior)
	numKnots := numBasis + order + 1
	knots := make([]float32, numKnots)

	for i := range knots {
		knots[i] = gridMin + float32(i-order)*step
	}

	return &BSplineGrid{
		Order:    order,
		NumBasis: numBasis,
		Knots:    knots,
	}
}

// Evaluate computes the values of all basis functions at a single point x.
// Uses the Cox-de Boor recursion algorithm.
// Returns a slice of length NumBasis.
func (g *BSplineGrid) Evaluate(x float32) []float32 {
	k := g.Order + 1 // B-spline order (degree + 1), e.g., 4 for cubic
	n := g.NumBasis
	t := g.Knots

	// Start with degree-0 basis functions (piecewise constant)
	numIntervals := len(t) - 1
	basis := make([]float32, numIntervals)
	for i := 0; i < numIntervals; i++ {
		if (x >= t[i] && x < t[i+1]) || (i == numIntervals-1 && x == t[i+1]) {
			basis[i] = 1.0
		}
	}

	// Cox-de Boor recursion for degrees 1..order
	for d := 1; d < k; d++ {
		newBasis := make([]float32, numIntervals-d)
		for i := range newBasis {
			var left, right float32

			denom1 := t[i+d] - t[i]
			if denom1 > 0 {
				left = (x - t[i]) / denom1 * basis[i]
			}

			denom2 := t[i+d+1] - t[i+1]
			if denom2 > 0 {
				right = (t[i+d+1] - x) / denom2 * basis[i+1]
			}

			newBasis[i] = left + right
		}
		basis = newBasis
	}

	// Ensure we return exactly NumBasis values
	if len(basis) > n {
		basis = basis[:n]
	}
	for len(basis) < n {
		basis = append(basis, 0)
	}

	return basis
}

// EvaluateBatch computes basis function values for a batch of input points.
// Returns a [len(xs)][NumBasis] matrix (row-major).
func (g *BSplineGrid) EvaluateBatch(xs []float32) [][]float32 {
	result := make([][]float32, len(xs))
	for i, x := range xs {
		result[i] = g.Evaluate(x)
	}
	return result
}

// InitSoftmaxApprox returns initial B-spline coefficients that roughly approximate
// the softmax-like behavior exp(x) / normalizing_constant over the grid range.
// This gives the KAN a head start rather than random initialization.
func InitSoftmaxApprox(grid *BSplineGrid) []float32 {
	n := grid.NumBasis
	coeffs := make([]float32, n)

	// Sample exp(x) at the grid centers and use those as initial coefficients
	step := (grid.Knots[len(grid.Knots)-1] - grid.Knots[0]) / float32(n+1)
	start := grid.Knots[0]

	for i := range coeffs {
		x := start + float32(i+1)*step
		coeffs[i] = float32(math.Exp(float64(x)))
	}

	// Normalize so geometric mean = 1 (consistent with the constraint)
	logSum := float64(0)
	for _, c := range coeffs {
		if c > 1e-10 {
			logSum += math.Log(float64(c))
		}
	}
	geoMean := float32(math.Exp(logSum / float64(n)))
	if geoMean > 0 {
		for i := range coeffs {
			coeffs[i] /= geoMean
		}
	}

	return coeffs
}
