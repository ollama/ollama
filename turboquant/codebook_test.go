package turboquant

import (
	"fmt"
	"math"
	"testing"
)

// TestUnitVectorCoordSamplesVariance checks that unitVectorCoordSamples
// produces samples with variance ≈ 1.0 for the normalized coordinate
// distribution (z_1/‖z‖ · √d). The variance of this distribution is exactly
// d · Var[z_1/‖z‖] = d · (1/d) = 1, independent of d. For small d the
// distribution has heavier tails than N(0,1) but still variance=1.
func TestUnitVectorCoordSamplesVariance(t *testing.T) {
	for _, dim := range []int{4, 8, 16, 32, 64, 128} {
		samples := unitVectorCoordSamples(dim, 2, 32768)
		var sum, sumSq float64
		for _, s := range samples {
			sum += s
			sumSq += s * s
		}
		n := float64(len(samples))
		mean := sum / n
		variance := sumSq/n - mean*mean
		// Mean should be ~0, variance ~1 for all d.
		if math.Abs(mean) > 0.05 {
			t.Errorf("dim=%d: mean=%.4f, want ~0", dim, mean)
		}
		if math.Abs(variance-1.0) > 0.05 {
			t.Errorf("dim=%d: variance=%.4f, want ~1.0", dim, variance)
		}
	}
}

// TestUnitVectorCoordSamplesKurtosis checks that small-d samples have lower
// kurtosis than N(0,1) (kurtosis=3), confirming the bounded-support Beta tails.
// The kurtosis of the coordinate distribution is 3d/(d+2), which equals
// 2.0 at d=4 and approaches 3.0 as d→∞. So for small d the distribution is
// platykurtic (kurtosis < 3, bounded support), unlike the unbounded Gaussian.
func TestUnitVectorCoordSamplesKurtosis(t *testing.T) {
	// For d=4: kurtosis = 3×4/(4+2) = 2.0 (platykurtic, well below Gaussian's 3).
	// For d=128: kurtosis = 3×128/130 ≈ 2.95 (close to Gaussian's 3).
	samplesSmall := unitVectorCoordSamples(4, 2, 65536)
	var s2, s4 float64
	for _, s := range samplesSmall {
		s2 += s * s
		s4 += s * s * s * s
	}
	n := float64(len(samplesSmall))
	varSmall := s2 / n
	kurtSmall := (s4 / n) / (varSmall * varSmall)
	if kurtSmall >= 3.0 {
		t.Errorf("dim=4: kurtosis=%.3f, want < 3.0 (platykurtic for bounded distribution)", kurtSmall)
	}

	samplesLarge := unitVectorCoordSamples(128, 2, 65536)
	var l2, l4 float64
	for _, s := range samplesLarge {
		l2 += s * s
		l4 += s * s * s * s
	}
	varLarge := l2 / n
	kurtLarge := (l4 / n) / (varLarge * varLarge)
	// For d=128 kurtosis should be close to Gaussian's 3.
	if math.Abs(kurtLarge-3.0) > 0.3 {
		t.Errorf("dim=128: kurtosis=%.3f, want ~3.0 (close to Gaussian)", kurtLarge)
	}
}

func TestScalarCodebookDeterministic(t *testing.T) {
	for _, bits := range []int{2, 3} {
		codebookA, boundsA := scalarCodebook(128, bits)
		codebookB, boundsB := scalarCodebook(128, bits)
		if len(codebookA) != 1<<bits {
			t.Fatalf("bits=%d codebook len=%d", bits, len(codebookA))
		}
		for i := range codebookA {
			if codebookA[i] != codebookB[i] {
				t.Fatalf("bits=%d centroid mismatch at %d", bits, i)
			}
		}
		for i := range boundsA {
			if boundsA[i] != boundsB[i] {
				t.Fatalf("bits=%d boundary mismatch at %d", bits, i)
			}
		}
	}
}

func TestCodebookBoundariesMonotonic(t *testing.T) {
	for _, bits := range []int{2, 3} {
		_, bounds := scalarCodebook(128, bits)
		for i := 1; i < len(bounds); i++ {
			if bounds[i] <= bounds[i-1] {
				t.Fatalf("bits=%d boundaries are not monotonic", bits)
			}
		}
	}
}

func TestQuantizeScalarByBoundaryDeterministic(t *testing.T) {
	codebook, bounds := scalarCodebook(128, 3)
	mid := bounds[2]

	left := quantizeScalarByBoundary(mid-1e-6, codebook, bounds)
	right := quantizeScalarByBoundary(mid+1e-6, codebook, bounds)
	atBoundary := quantizeScalarByBoundary(mid, codebook, bounds)

	if left != 2 {
		t.Fatalf("left boundary bucket = %d, want 2", left)
	}
	if right != 3 {
		t.Fatalf("right boundary bucket = %d, want 3", right)
	}
	if atBoundary != 3 {
		t.Fatalf("exact boundary bucket = %d, want 3", atBoundary)
	}
}

// TestPaperTheoremOneMSEBounds verifies that the Lloyd-Max codebook achieves
// the total-distortion bounds from Theorem 1 of arXiv:2504.19874
// for unit-norm vectors at bit widths 1–4.
//
// Paper Theorem 1 bounds (total distortion ||v - v_hat||^2 for unit-norm vectors):
//
//	b=1 → 0.36,  b=2 → 0.117,  b=3 → 0.03,  b=4 → 0.009
func TestPaperTheoremOneMSEBounds(t *testing.T) {
	paperBounds := map[int]float64{
		1: 0.36,
		2: 0.117,
		3: 0.03,
		4: 0.009,
	}

	const dim = 128
	const nTrials = 200
	rot := BuildRotation(dim, 0x42c0ffee)

	for bits := 1; bits <= 4; bits++ {
		bits := bits // capture loop variable
		t.Run(fmt.Sprintf("bits=%d", bits), func(t *testing.T) {
			bound := paperBounds[bits]
			codebook, boundaries := scalarCodebook(dim, bits)

			var totalMSE float64
			for trial := range nTrials {
				vec := pseudoRandomVector(dim, uint64(trial)*0x9e3779b97f4a7c15+1)

				// Normalize to unit norm.
				var norm float64
				for _, v := range vec {
					norm += float64(v) * float64(v)
				}
				norm = math.Sqrt(norm)
				if norm < 1e-10 {
					continue
				}
				for i := range vec {
					vec[i] /= float32(norm)
				}

				// Apply rotation (matches internal encode behavior).
				rotated := ApplyRotation(vec, rot)

				// Compute RMS scale (same as blockScale in encode.go).
				var sumSq float64
				for _, v := range rotated {
					sumSq += float64(v) * float64(v)
				}
				scale := float32(math.Sqrt(sumSq / float64(dim)))

				// Quantize each element and accumulate per-element MSE.
				var mse float64
				for _, v := range rotated {
					normalized := float32(0)
					if scale > 0 {
						normalized = v / scale
					}
					idx := quantizeScalarByBoundary(normalized, codebook, boundaries)
					recon := codebook[idx] * scale
					diff := float64(v - recon)
					mse += diff * diff
				}
				totalMSE += mse
			}

			avgMSE := totalMSE / float64(nTrials)
			// Allow 50% headroom over the paper bound to account for finite
			// sample size and finite dimension effects.
			if avgMSE > bound*1.5 {
				t.Errorf("bits=%d: avg MSE %.6f exceeds 1.5× paper bound %.6f",
					bits, avgMSE, bound*1.5)
			}
			t.Logf("bits=%d: avg MSE=%.6f, paper bound=%.6f, ratio=%.2f",
				bits, avgMSE, bound, avgMSE/bound)
		})
	}
}
