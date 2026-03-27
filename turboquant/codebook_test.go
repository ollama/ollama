package turboquant

import (
	"fmt"
	"math"
	"testing"
)

func TestCodebookCentroidsAreSymmetric(t *testing.T) {
	for bits := 1; bits <= 4; bits++ {
		t.Run(fmt.Sprintf("mse_bits=%d", bits), func(t *testing.T) {
			cb := NewCodebook(bits+1, 128)
			centroids := cb.Centroids
			n := len(centroids)

			if n != 1<<bits {
				t.Fatalf("expected %d centroids, got %d", 1<<bits, n)
			}

			for i := 0; i < n/2; i++ {
				sum := float64(centroids[i]) + float64(centroids[n-1-i])
				if math.Abs(sum) > 1e-6 {
					t.Errorf("centroids not symmetric: c[%d]=%f + c[%d]=%f = %f",
						i, centroids[i], n-1-i, centroids[n-1-i], sum)
				}
			}
		})
	}
}

func TestCodebookCentroidsAreStrictlyIncreasing(t *testing.T) {
	for bits := 1; bits <= 4; bits++ {
		t.Run(fmt.Sprintf("mse_bits=%d", bits), func(t *testing.T) {
			cb := NewCodebook(bits+1, 128)
			for i := 1; i < len(cb.Centroids); i++ {
				if cb.Centroids[i] <= cb.Centroids[i-1] {
					t.Errorf("not sorted at index %d: %f <= %f",
						i, cb.Centroids[i], cb.Centroids[i-1])
				}
			}
		})
	}
}

func TestCodebookScalesWithDimension(t *testing.T) {
	// Centroids scale as 1/sqrt(d). Ratio between dim=64 and dim=128
	// should be sqrt(128/64) = sqrt(2)
	cb64 := NewCodebook(4, 64)
	cb128 := NewCodebook(4, 128)

	ratio := math.Sqrt(128.0 / 64.0)

	for i := range cb64.Centroids {
		expected := float64(cb128.Centroids[i]) * ratio
		actual := float64(cb64.Centroids[i])

		if math.Abs(actual-expected)/math.Max(math.Abs(expected), 1e-10) > 1e-5 {
			t.Errorf("centroid[%d]: got %f, want %f (ratio sqrt(2))", i, actual, expected)
		}
	}
}

func TestCodebookMatchesPaperValues(t *testing.T) {
	// The paper states that for N(0, 1/d), the b=2 centroids are
	// {+/-0.453/sqrt(d), +/-1.51/sqrt(d)}.
	// Verify our codebook matches this for dim=128.
	cb := NewCodebook(3, 128) // TQ3 uses MSE bits = 2
	d := 128.0
	sqrtD := math.Sqrt(d)

	expectedCentroids := []float64{
		-1.51 / sqrtD,
		-0.453 / sqrtD,
		0.453 / sqrtD,
		1.51 / sqrtD,
	}

	for i, expected := range expectedCentroids {
		actual := float64(cb.Centroids[i])
		// Allow 1% tolerance since paper rounds the values
		if math.Abs(actual-expected)/math.Abs(expected) > 0.01 {
			t.Errorf("centroid[%d] = %f, paper says ~%f (1%% tolerance)", i, actual, expected)
		}
	}
}

func TestCodebookNumCentroids(t *testing.T) {
	tests := []struct {
		bitWidth int
		expected int
	}{
		{3, 4},  // TQ3: MSE uses 2 bits -> 4 centroids
		{4, 8},  // TQ4: MSE uses 3 bits -> 8 centroids
		{5, 16}, // hypothetical TQ5: MSE uses 4 bits -> 16 centroids
	}

	for _, tc := range tests {
		cb := NewCodebook(tc.bitWidth, 128)
		if cb.NumCentroids() != tc.expected {
			t.Errorf("bitWidth=%d: got %d centroids, want %d",
				tc.bitWidth, tc.expected, cb.NumCentroids())
		}
	}
}
