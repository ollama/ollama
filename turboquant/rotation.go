package turboquant

import (
	"math"
	"math/rand/v2"
)

// GenerateRotation creates a d x d orthogonal rotation matrix using
// the Randomized Hadamard Transform: Pi = (1/sqrt(d)) * H_d * D
// where H_d is the Hadamard matrix and D is a diagonal matrix of random +/-1 signs.
//
// The returned slice is in column-major (GGML) order: data[j*d + i] = Pi[i][j].
// Returns nil if dim is not a power of 2.
func GenerateRotation(dim int, seed uint64) []float32 {
	if !isPowerOf2(dim) {
		return nil
	}

	h := hadamard(dim)
	rng := rand.New(rand.NewPCG(seed, seed^0xDEAD_BEEF_CAFE_BABE))

	signs := make([]float32, dim)
	for i := range signs {
		if rng.Float64() < 0.5 {
			signs[i] = -1
		} else {
			signs[i] = 1
		}
	}

	scale := float32(1.0 / math.Sqrt(float64(dim)))

	// Pi[i][j] = scale * H[i][j] * signs[j]
	// Column-major: result[j*dim + i] = Pi[i][j]
	result := make([]float32, dim*dim)
	for i := 0; i < dim; i++ {
		for j := 0; j < dim; j++ {
			result[j*dim+i] = h[i*dim+j] * signs[j] * scale
		}
	}

	return result
}

// GenerateRotationTranspose creates the transpose of the rotation matrix Pi^T.
// Pi^T[i][j] = Pi[j][i] = scale * H[j][i] * signs[i] = scale * H[i][j] * signs[i]
// (H is symmetric).
//
// The returned slice is in column-major order: data[j*d + i] = Pi^T[i][j].
func GenerateRotationTranspose(dim int, seed uint64) []float32 {
	if !isPowerOf2(dim) {
		return nil
	}

	h := hadamard(dim)
	rng := rand.New(rand.NewPCG(seed, seed^0xDEAD_BEEF_CAFE_BABE))

	signs := make([]float32, dim)
	for i := range signs {
		if rng.Float64() < 0.5 {
			signs[i] = -1
		} else {
			signs[i] = 1
		}
	}

	scale := float32(1.0 / math.Sqrt(float64(dim)))

	// Pi^T[i][j] = scale * H[i][j] * signs[i]
	// Column-major: result[j*dim + i] = Pi^T[i][j]
	result := make([]float32, dim*dim)
	for i := 0; i < dim; i++ {
		for j := 0; j < dim; j++ {
			result[j*dim+i] = h[i*dim+j] * signs[i] * scale
		}
	}

	return result
}

// hadamard generates the unnormalized Hadamard matrix of size n x n in row-major order.
// Requires n to be a power of 2.
func hadamard(n int) []float32 {
	if n == 1 {
		return []float32{1}
	}

	half := hadamard(n / 2)
	m := n / 2
	result := make([]float32, n*n)

	for i := 0; i < m; i++ {
		for j := 0; j < m; j++ {
			val := half[i*m+j]
			result[i*n+j] = val       // top-left
			result[i*n+j+m] = val     // top-right
			result[(i+m)*n+j] = val   // bottom-left
			result[(i+m)*n+j+m] = -val // bottom-right
		}
	}

	return result
}

// VerifyOrthogonal checks that Pi^T * Pi ≈ I within the given tolerance.
// Useful for testing.
func VerifyOrthogonal(piColMajor []float32, dim int, tol float64) bool {
	for i := 0; i < dim; i++ {
		for j := 0; j < dim; j++ {
			var dot float64
			for k := 0; k < dim; k++ {
				// Pi^T[k][i] * Pi[k][j] in column-major:
				// piColMajor[i*dim+k] * piColMajor[j*dim+k]
				dot += float64(piColMajor[i*dim+k]) * float64(piColMajor[j*dim+k])
			}

			expected := 0.0
			if i == j {
				expected = 1.0
			}

			if math.Abs(dot-expected) > tol {
				return false
			}
		}
	}

	return true
}
