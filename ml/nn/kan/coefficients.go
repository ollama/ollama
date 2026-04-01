package kan

import "math"

// Coefficients manages B-spline coefficients with the geometric mean constraint
// and clip-redistribute weight shifting scheme.
//
// The key invariant: the geometric mean of |coefficients| is anchored at 1.
// Weights exceeding an adaptive threshold are clipped, and the excess is
// redistributed proportionally via each coefficient's share of the geometric mean.
type Coefficients struct {
	Weights []float32
}

// NewCoefficients creates a new coefficient set initialized to approximate softmax.
func NewCoefficients(grid *BSplineGrid) *Coefficients {
	c := &Coefficients{
		Weights: InitSoftmaxApprox(grid),
	}
	c.NormalizeAndRedistribute()
	return c
}

// NewCoefficientsFromWeights creates a coefficient set from existing weights.
func NewCoefficientsFromWeights(weights []float32) *Coefficients {
	w := make([]float32, len(weights))
	copy(w, weights)
	return &Coefficients{Weights: w}
}

// NormalizeAndRedistribute applies the full geometric mean normalization
// and clip-redistribute pipeline:
//  1. Normalize so geometric mean of |weights| = 1
//  2. Compute adaptive clip threshold (mean + 2*std of |weights|)
//  3. Clip weights exceeding threshold
//  4. Redistribute clipped excess proportionally via geometric mean shares
//  5. Re-normalize to restore geometric mean = 1
func (c *Coefficients) NormalizeAndRedistribute() {
	if len(c.Weights) == 0 {
		return
	}

	c.normalizeGeoMean()
	c.clipAndRedistribute()
	// Re-normalize after redistribution to maintain the invariant
	c.normalizeGeoMean()
}

// normalizeGeoMean scales all coefficients so that the geometric mean
// of their absolute values equals 1.
//
// geometric_mean(|w|) = exp(mean(log(|w|))) = 1
// => mean(log(|w|)) = 0
// => divide each w_i by current geometric mean
func (c *Coefficients) normalizeGeoMean() {
	n := len(c.Weights)
	if n == 0 {
		return
	}

	logSum := float64(0)
	for _, w := range c.Weights {
		absW := float64(w)
		if absW < 0 {
			absW = -absW
		}
		if absW < 1e-10 {
			absW = 1e-10
		}
		logSum += math.Log(absW)
	}

	geoMean := math.Exp(logSum / float64(n))
	if geoMean < 1e-10 {
		return
	}

	scale := float32(1.0 / geoMean)
	for i := range c.Weights {
		c.Weights[i] *= scale
	}
}

// clipAndRedistribute applies adaptive clipping and redistributes the excess.
//
// Threshold = mean(|w|) + 2*std(|w|)
//
// For each weight exceeding the threshold:
//   - Clip to threshold (preserving sign)
//   - Accumulate the excess
//
// The total excess is then redistributed to all coefficients proportionally
// to each coefficient's share of the geometric mean:
//
//	share_i = |w_i| / sum(|w_j|)
//	w_i += sign(w_i) * share_i * total_excess
//
// This ensures no single basis function dominates while preserving
// the total energy of the coefficient vector.
func (c *Coefficients) clipAndRedistribute() {
	n := len(c.Weights)
	if n == 0 {
		return
	}

	// Compute absolute values
	abs := make([]float64, n)
	for i, w := range c.Weights {
		abs[i] = math.Abs(float64(w))
	}

	// Compute mean and std of absolute values
	mean := float64(0)
	for _, a := range abs {
		mean += a
	}
	mean /= float64(n)

	variance := float64(0)
	for _, a := range abs {
		d := a - mean
		variance += d * d
	}
	variance /= float64(n)
	std := math.Sqrt(variance)

	// Adaptive threshold: mean + 2*std
	threshold := mean + 2.0*std
	if threshold < 1e-10 {
		return
	}

	// Clip and accumulate excess
	totalExcess := float64(0)
	for i := range c.Weights {
		if abs[i] > threshold {
			excess := abs[i] - threshold
			totalExcess += excess
			if c.Weights[i] > 0 {
				c.Weights[i] = float32(threshold)
			} else {
				c.Weights[i] = float32(-threshold)
			}
			abs[i] = threshold
		}
	}

	if totalExcess < 1e-10 {
		return
	}

	// Compute sum of absolute values after clipping for proportional redistribution
	absSum := float64(0)
	for _, a := range abs {
		absSum += a
	}
	if absSum < 1e-10 {
		return
	}

	// Redistribute excess proportionally to each coefficient's geometric mean share
	for i := range c.Weights {
		share := abs[i] / absSum
		redistribution := float32(share * totalExcess)
		if c.Weights[i] >= 0 {
			c.Weights[i] += redistribution
		} else {
			c.Weights[i] -= redistribution
		}
	}
}

// GeometricMean returns the current geometric mean of |weights|.
// Should be ~1.0 after normalization.
func (c *Coefficients) GeometricMean() float64 {
	n := len(c.Weights)
	if n == 0 {
		return 0
	}

	logSum := float64(0)
	for _, w := range c.Weights {
		absW := math.Abs(float64(w))
		if absW < 1e-10 {
			absW = 1e-10
		}
		logSum += math.Log(absW)
	}

	return math.Exp(logSum / float64(n))
}

// Clone returns a deep copy of the coefficients.
func (c *Coefficients) Clone() *Coefficients {
	w := make([]float32, len(c.Weights))
	copy(w, c.Weights)
	return &Coefficients{Weights: w}
}
