package turboquant

import (
	"fmt"
	"math"
	"testing"
)

func TestHadamardOrthogonality(t *testing.T) {
	for _, dim := range []int{1, 2, 4, 8, 16, 32, 64, 128} {
		t.Run(fmt.Sprintf("dim=%d", dim), func(t *testing.T) {
			h := hadamard(dim)

			if len(h) != dim*dim {
				t.Fatalf("expected %d elements, got %d", dim*dim, len(h))
			}

			// H^T * H = dim * I for unnormalized Hadamard matrices
			for i := 0; i < dim; i++ {
				for j := 0; j < dim; j++ {
					var dot float64
					for k := 0; k < dim; k++ {
						dot += float64(h[i*dim+k]) * float64(h[j*dim+k])
					}

					expected := 0.0
					if i == j {
						expected = float64(dim)
					}

					if math.Abs(dot-expected) > 1e-6 {
						t.Errorf("H^T*H[%d][%d] = %f, want %f", i, j, dot, expected)
					}
				}
			}
		})
	}
}

func TestHadamardEntries(t *testing.T) {
	// All entries of an unnormalized Hadamard matrix are +1 or -1
	for _, dim := range []int{2, 4, 8, 16} {
		h := hadamard(dim)
		for i, v := range h {
			if v != 1 && v != -1 {
				t.Errorf("dim=%d: entry[%d] = %f, want +/-1", dim, i, v)
			}
		}
	}
}

func TestRotationIsOrthogonal(t *testing.T) {
	for _, dim := range []int{4, 8, 16, 32, 64, 128} {
		t.Run(fmt.Sprintf("dim=%d", dim), func(t *testing.T) {
			piData := GenerateRotation(dim, TurboQuantSeed)
			if piData == nil {
				t.Fatal("GenerateRotation returned nil")
			}

			if !VerifyOrthogonal(piData, dim, 1e-5) {
				t.Error("rotation matrix is not orthogonal (Pi^T * Pi != I)")
			}
		})
	}
}

func TestRotationIsDeterministic(t *testing.T) {
	dim := 64
	seed := uint64(12345)

	pi1 := GenerateRotation(dim, seed)
	pi2 := GenerateRotation(dim, seed)

	for i := range pi1 {
		if pi1[i] != pi2[i] {
			t.Fatalf("index %d: %f != %f", i, pi1[i], pi2[i])
		}
	}
}

func TestRotationRejectsNonPowerOf2(t *testing.T) {
	for _, dim := range []int{3, 7, 12, 65, 127} {
		if GenerateRotation(dim, TurboQuantSeed) != nil {
			t.Errorf("dim=%d: expected nil for non-power-of-2", dim)
		}
	}
}

func TestRotationAndTransposeAreInverse(t *testing.T) {
	dim := 32
	seed := TurboQuantSeed

	pi := GenerateRotation(dim, seed)
	piT := GenerateRotationTranspose(dim, seed)

	// Verify (Pi^T * Pi)[i][j] = delta_{ij} using column-major data.
	// Column-major: data[col*dim + row] = M[row][col]
	for i := 0; i < dim; i++ {
		for j := 0; j < dim; j++ {
			var dot float64
			for k := 0; k < dim; k++ {
				// Pi^T[i][k] = piT[k*dim + i]
				// Pi[k][j]   = pi[j*dim + k]
				dot += float64(piT[k*dim+i]) * float64(pi[j*dim+k])
			}

			expected := 0.0
			if i == j {
				expected = 1.0
			}

			if math.Abs(dot-expected) > 1e-5 {
				t.Errorf("(Pi^T * Pi)[%d][%d] = %f, want %f", i, j, dot, expected)
			}
		}
	}
}

func TestRotationPreservesNorm(t *testing.T) {
	// Rotating a vector should preserve its L2 norm (orthogonal invariant)
	dim := 16
	pi := GenerateRotation(dim, TurboQuantSeed)

	x := make([]float64, dim)
	for i := range x {
		x[i] = float64(i+1) / float64(dim)
	}

	var normBefore float64
	for _, v := range x {
		normBefore += v * v
	}
	normBefore = math.Sqrt(normBefore)

	// Apply rotation: y = Pi * x
	// In column-major: pi[j*dim + i] = Pi[i][j]
	// y[i] = sum_j Pi[i][j] * x[j] = sum_j pi[j*dim+i] * x[j]
	y := make([]float64, dim)
	for i := 0; i < dim; i++ {
		for j := 0; j < dim; j++ {
			y[i] += float64(pi[j*dim+i]) * x[j]
		}
	}

	var normAfter float64
	for _, v := range y {
		normAfter += v * v
	}
	normAfter = math.Sqrt(normAfter)

	if math.Abs(normBefore-normAfter) > 1e-5 {
		t.Errorf("norm changed: before=%f after=%f", normBefore, normAfter)
	}
}
