package turboquant

import (
	"math"
	"sync"
)

type Rotation struct {
	Dim    int
	Seed   uint64
	Matrix []float32 // row-major orthogonal matrix
}

type rotationCacheKey struct {
	dim  int
	seed uint64
}

var rotationCache sync.Map

func BuildRotation(dim int, seed uint64) Rotation {
	key := rotationCacheKey{dim: dim, seed: seed}
	if cached, ok := rotationCache.Load(key); ok {
		return cached.(Rotation)
	}

	rot := Rotation{
		Dim:    dim,
		Seed:   seed,
		Matrix: buildOrthogonalMatrix(dim, seed),
	}
	actual, _ := rotationCache.LoadOrStore(key, rot)
	return actual.(Rotation)
}

func ApplyRotation(x []float32, rot Rotation) []float32 {
	if len(x) != rot.Dim {
		panic("turboquant: vector length does not match rotation dimension")
	}

	out := make([]float32, rot.Dim)
	for row := range rot.Dim {
		base := row * rot.Dim
		var sum float32
		for col, value := range x {
			sum += rot.Matrix[base+col] * value
		}
		out[row] = sum
	}
	return out
}

func ApplyInverseRotation(y []float32, rot Rotation) []float32 {
	if len(y) != rot.Dim {
		panic("turboquant: vector length does not match rotation dimension")
	}

	out := make([]float32, rot.Dim)
	for row := range rot.Dim {
		yVal := y[row]
		base := row * rot.Dim
		// The inverse rotation accumulates in float32 on the reconstruction path; keep this FP32-accumulate behavior explicit while long-generation corruption audits remain active.
		for col := range rot.Dim {
			out[col] += rot.Matrix[base+col] * yVal
		}
	}
	return out
}

// buildOrthogonalMatrix returns a dim×dim orthogonal matrix derived from the
// given seed using Householder QR factorisation. The input is the same seeded
// Gaussian matrix used by the previous Gram-Schmidt path, but the QR Q-factor
// is numerically unconditionally orthogonal. This algorithm replaced the
// classical Gram-Schmidt used through BlockVersion 3; the Householder path was
// introduced at BlockVersion 4 (current: BlockVersion 6).
//
// Algorithm: apply dim-1 Householder reflectors H_1…H_{dim-1} from the left
// to reduce A to upper triangular form R. Simultaneously accumulate
// Q = H_1 * H_2 * … * H_{dim-1} by right-multiplying each reflector into Q,
// starting from the identity. The resulting Q is the orthogonal factor in
// A = QR and has orthonormal rows and columns.
func buildOrthogonalMatrix(dim int, seed uint64) []float32 {
	if dim <= 0 {
		return nil
	}

	// Initialise A with the same seeded Gaussian rows as before.
	a := make([][]float64, dim)
	for row := range dim {
		a[row] = make([]float64, dim)
		rng := splitmix64(seed ^ uint64(dim)<<32 ^ uint64(row+1)*0x9e3779b97f4a7c15)
		for col := range dim {
			a[row][col] = gaussianFloat64(&rng)
		}
	}

	// Q starts as the identity; we accumulate Q = H_1 * … * H_{dim-1}.
	q := make([][]float64, dim)
	for i := range q {
		q[i] = make([]float64, dim)
		q[i][i] = 1.0
	}

	v := make([]float64, dim) // scratch buffer for the Householder vector

	for k := range dim - 1 {
		n := dim - k

		// Build Householder vector for column k, rows k..dim-1.
		// Sign chosen to avoid cancellation: sigma = −sign(a[k][k]) * ||x||.
		for i := range n {
			v[i] = a[k+i][k]
		}
		sigma := vectorNorm64(v[:n])
		if v[0] >= 0 {
			sigma = -sigma
		}
		v[0] -= sigma
		vnorm2 := dotFloat64(v[:n], v[:n])
		if vnorm2 < 1e-28 {
			continue // column already zeroed — no reflector needed
		}
		beta := 2.0 / vnorm2

		// Apply H_k to a[k:, k:] from the left.
		for j := k; j < dim; j++ {
			var dot float64
			for i := range n {
				dot += v[i] * a[k+i][j]
			}
			dot *= beta
			for i := range n {
				a[k+i][j] -= dot * v[i]
			}
		}

		// Apply H_k to q[:, k:] from the right: q ← q * H_k.
		for i := range dim {
			var dot float64
			for j := range n {
				dot += q[i][k+j] * v[j]
			}
			dot *= beta
			for j := range n {
				q[i][k+j] -= dot * v[j]
			}
		}
	}

	// Sign-normalise each row so the first non-negligible element is positive.
	// This convention matches the previous Gram-Schmidt path and makes the
	// output deterministic despite the reflector sign ambiguity.
	for row := range dim {
		for _, value := range q[row] {
			if math.Abs(value) <= 1e-12 {
				continue
			}
			if value < 0 {
				for col := range dim {
					q[row][col] = -q[row][col]
				}
			}
			break
		}
	}

	out := make([]float32, dim*dim)
	for row := range dim {
		for col := range dim {
			out[row*dim+col] = float32(q[row][col])
		}
	}
	return out
}

func dotFloat64(a, b []float64) float64 {
	var out float64
	for i := range a {
		out += a[i] * b[i]
	}
	return out
}

func vectorNorm64(values []float64) float64 {
	var sum float64
	for _, value := range values {
		sum += value * value
	}
	return math.Sqrt(sum)
}

func gaussianFloat64(rng *splitmix64) float64 {
	u1 := unitUniform64(rng)
	u2 := unitUniform64(rng)
	return math.Sqrt(-2*math.Log(u1)) * math.Cos(2*math.Pi*u2)
}

func unitUniform64(rng *splitmix64) float64 {
	const scale = 1.0 / (1 << 53)
	return (float64(rng.next()>>11) + 0.5) * scale
}

type splitmix64 uint64

func (s *splitmix64) next() uint64 {
	*s += 0x9e3779b97f4a7c15
	z := uint64(*s)
	z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9
	z = (z ^ (z >> 27)) * 0x94d049bb133111eb
	return z ^ (z >> 31)
}
