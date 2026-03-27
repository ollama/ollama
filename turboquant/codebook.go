package turboquant

import "math"

// Lloyd-Max optimal quantizer centroids for the standard normal distribution N(0,1).
//
// After rotation by Pi, each coordinate of a unit-sphere vector follows a Beta
// distribution that converges to N(0, 1/d) in high dimensions (Lemma 1 in the
// paper). These centroids are the optimal solution to the continuous 1D k-means
// problem (Equation 4): minimizing MSE over the Gaussian PDF.
//
// At runtime, centroids are scaled by 1/sqrt(d) to match the N(0, 1/d) distribution.
//
// The map key is the bit-width b of the scalar quantizer (NOT the total TurboQuant
// bit-width). For TurboQuant_prod at total bit-width B, the MSE stage uses b = B-1
// bits (the remaining 1 bit goes to QJL).
var stdGaussianCentroids = map[int][]float64{
	// b=1: 2 centroids. Optimal for N(0,1): +/- E[|X|] = +/- sqrt(2/pi)
	1: {-0.7978845608, 0.7978845608},
	// b=2: 4 centroids. Paper confirms: {+/-0.453/sqrt(d), +/-1.51/sqrt(d)} for N(0,1/d)
	2: {-1.5104176088, -0.4527800398, 0.4527800398, 1.5104176088},
	// b=3: 8 centroids
	3: {-2.1519481310, -1.3439092613, -0.7560052489, -0.2451209526,
		0.2451209526, 0.7560052489, 1.3439092613, 2.1519481310},
	// b=4: 16 centroids
	4: {-2.7326368500, -2.0690790327, -1.6180234170, -1.2562091030,
		-0.9423520268, -0.6567903640, -0.3880823390, -0.1284185740,
		0.1284185740, 0.3880823390, 0.6567903640, 0.9423520268,
		1.2562091030, 1.6180234170, 2.0690790327, 2.7326368500},
}

// Lloyd-Max optimal quantizer boundaries for N(0,1). Each boundary is the midpoint
// between consecutive centroids (Voronoi tessellation). A coordinate value x is
// assigned to centroid i when boundaries[i-1] < x <= boundaries[i].
var stdGaussianBoundaries = map[int][]float64{
	1: {0},
	2: {-0.9815988243, 0, 0.9815988243},
	3: {-1.7479286962, -1.0499572551, -0.5005631008, 0,
		0.5005631008, 1.0499572551, 1.7479286962},
	4: {-2.4008579413, -1.8435512249, -1.4371162600, -1.0992995649,
		-0.7995711954, -0.5224363515, -0.2582504565, 0,
		0.2582504565, 0.5224363515, 0.7995711954, 1.0992995649,
		1.4371162600, 1.8435512249, 2.4008579413},
}

// Codebook holds precomputed quantizer parameters scaled for a specific head dimension.
type Codebook struct {
	BitWidth   int
	Dim        int
	Centroids  []float32
	Boundaries []float32
}

// NewCodebook creates a codebook for the given total TurboQuant bit-width and vector
// dimension. The MSE stage uses (bitWidth - 1) bits for the scalar quantizer; the
// remaining 1 bit is reserved for the QJL residual.
func NewCodebook(bitWidth, dim int) *Codebook {
	mseBits := bitWidth - 1
	if mseBits < 1 || mseBits > 4 {
		mseBits = 2
	}

	stdC := stdGaussianCentroids[mseBits]
	stdB := stdGaussianBoundaries[mseBits]

	// Scale from N(0,1) to N(0, 1/d) by multiplying by 1/sqrt(d)
	scale := 1.0 / math.Sqrt(float64(dim))

	centroids := make([]float32, len(stdC))
	for i, c := range stdC {
		centroids[i] = float32(c * scale)
	}

	boundaries := make([]float32, len(stdB))
	for i, b := range stdB {
		boundaries[i] = float32(b * scale)
	}

	return &Codebook{
		BitWidth:   bitWidth,
		Dim:        dim,
		Centroids:  centroids,
		Boundaries: boundaries,
	}
}

// NumCentroids returns 2^(bitWidth-1), the number of MSE quantization levels.
func (cb *Codebook) NumCentroids() int {
	return len(cb.Centroids)
}
