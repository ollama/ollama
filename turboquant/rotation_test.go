package turboquant

import (
	"fmt"
	"math"
	"testing"
)

// testDims covers small, medium, and the primary production dims.
var testDims = []int{4, 8, 16, 32, 64, 128}

func TestBuildRotationDeterministic(t *testing.T) {
	for _, dim := range testDims {
		a := BuildRotation(dim, 123)
		b := BuildRotation(dim, 123)
		if a.Dim != b.Dim || a.Seed != b.Seed || len(a.Matrix) != len(b.Matrix) {
			t.Fatalf("rotation metadata mismatch for dim %d", dim)
		}
		for i := range a.Matrix {
			if a.Matrix[i] != b.Matrix[i] {
				t.Fatalf("rotation mismatch at dim=%d idx=%d", dim, i)
			}
		}
	}
}

func TestBuildRotationDifferentSeedsDiffer(t *testing.T) {
	a := BuildRotation(16, 111)
	b := BuildRotation(16, 222)
	different := false
	for i := range a.Matrix {
		if a.Matrix[i] != b.Matrix[i] {
			different = true
			break
		}
	}
	if !different {
		t.Fatal("different seeds produced the same orthogonal matrix")
	}
}

func TestApplyInverseRotation(t *testing.T) {
	for _, dim := range testDims {
		values := pseudoRandomVector(dim, uint64(dim)*17)
		rot := BuildRotation(dim, uint64(dim)*19)
		got := ApplyInverseRotation(ApplyRotation(values, rot), rot)
		for i := range values {
			if abs32(values[i]-got[i]) > 1e-4 {
				t.Fatalf("dim=%d idx=%d got=%v want=%v", dim, i, got[i], values[i])
			}
		}
	}
}

func TestRotationPreservesNorm(t *testing.T) {
	for _, dim := range testDims {
		values := pseudoRandomVector(dim, uint64(dim)*23)
		rot := BuildRotation(dim, uint64(dim)*29)
		before := vectorNorm(values)
		after := vectorNorm(ApplyRotation(values, rot))
		if abs32(before-after) > 1e-3 {
			t.Fatalf("dim=%d norm drift=%v", dim, abs32(before-after))
		}
	}
}

// TestAttentionScoreRotationInvariance verifies the exact mathematical
// invariant: dot(Q,K) == dot(R@Q, R@K) for an orthogonal matrix R. Because R
// is orthogonal, R^T R = I, so the inner product is preserved exactly (up to
// floating-point rounding). This is the property that makes rotating K before
// quantization safe when Q is also rotated at attention time.
func TestAttentionScoreRotationInvariance(t *testing.T) {
	for _, dim := range []int{64, 128, 256} {
		t.Run(fmt.Sprintf("dim=%d", dim), func(t *testing.T) {
			seed := uint64(0x42c0ffee)
			q := pseudoRandomVector(dim, seed)
			k := pseudoRandomVector(dim, seed^0xbeef)

			rot := BuildRotation(dim, seed+1)
			qRot := ApplyRotation(q, rot)
			kRot := ApplyRotation(k, rot)

			var dotOrig float64
			for i := range q {
				dotOrig += float64(q[i]) * float64(k[i])
			}

			var dotRot float64
			for i := range qRot {
				dotRot += float64(qRot[i]) * float64(kRot[i])
			}

			relErr := math.Abs(dotOrig-dotRot) / (math.Abs(dotOrig) + 1e-10)
			if relErr > 1e-4 {
				t.Errorf("dim=%d: dot(Q,K)=%.6f dot(RQ,RK)=%.6f relErr=%.2e",
					dim, dotOrig, dotRot, relErr)
			}
		})
	}
}

// TestQuantizedAttentionScorePreservation verifies the practical end-to-end
// path: encode K per-head (which stores R@k), rotate Q (giving R@q), then
// compute (R@q)·(R@k_quant) and compare against the true dot(Q,K). Single
// trials can have high per-sample error from quantization noise, so we average
// over many trials and check the mean relative error, matching the pattern used
// by TestEncodeKeyPerHeadRoundTrip.
func TestQuantizedAttentionScorePreservation(t *testing.T) {
	const dim = 128
	const trials = 100

	for _, preset := range []Preset{PresetTQ2, PresetTQ3} {
		t.Run(preset.Name, func(t *testing.T) {
			rng := splitmix64(0x1111feed)
			rot := BuildRotation(dim, preset.RotationSeed)
			var totalAbsErr, totalAbsDot float64

			for trial := range trials {
				q := make([]float32, dim)
				k := make([]float32, dim)
				for j := range q {
					q[j] = float32(gaussianFloat64(&rng))
					k[j] = float32(gaussianFloat64(&rng))
				}

				// True attention score.
				var trueScore float64
				for i := range q {
					trueScore += float64(q[i]) * float64(k[i])
				}

				// Quantized path: encode K in rotated space, rotate Q, compute score.
				packed, scale, err := EncodeKeyPerHead(k, preset)
				if err != nil {
					t.Fatalf("trial %d EncodeKeyPerHead: %v", trial, err)
				}
				kRecon := DequantKeyPerHead(packed, scale, dim, preset.KeyPrimaryBits)
				qRot := ApplyRotation(q, rot)

				var quantScore float64
				for i := range qRot {
					quantScore += float64(qRot[i]) * float64(kRecon[i])
				}

				totalAbsErr += math.Abs(trueScore - quantScore)
				totalAbsDot += math.Abs(trueScore)
			}

			avgRelErr := totalAbsErr / (totalAbsDot + 1e-10)
			t.Logf("preset=%s avg relative dot error = %.4f over %d trials", preset.Name, avgRelErr, trials)

			// tq3 (3-bit) is tighter than tq2 (2-bit); these thresholds match the
			// codec quality validated by TestEncodeKeyPerHeadRoundTrip.
			maxRelErr := 0.45
			if preset.KeyPrimaryBits >= 3 {
				maxRelErr = 0.25
			}
			if avgRelErr > maxRelErr {
				t.Errorf("preset=%s avg relative dot error = %.4f, want <= %.4f",
					preset.Name, avgRelErr, maxRelErr)
			}
		})
	}
}

// TestBuildRotationIsOrthogonal verifies that Q satisfies Q*Q^T = I (rows are
// orthonormal). This is the core invariant required by the TurboQuant encoding
// and is guaranteed unconditionally by the Householder QR algorithm.
func TestBuildRotationIsOrthogonal(t *testing.T) {
	// Include dim=256 to exercise the algorithm well beyond typical head_dim.
	for _, dim := range append(testDims, 256) {
		rot := BuildRotation(dim, uint64(dim)*31)
		for i := range dim {
			for j := i; j < dim; j++ {
				var dot float32
				for k := range dim {
					dot += rot.Matrix[i*dim+k] * rot.Matrix[j*dim+k]
				}
				if i == j {
					if math.Abs(float64(dot-1)) > 5e-5 {
						t.Fatalf("dim=%d row %d: self-dot=%.6f, want 1.0", dim, i, dot)
					}
				} else if math.Abs(float64(dot)) > 5e-5 {
					t.Fatalf("dim=%d rows %d,%d: cross-dot=%.6f, want 0.0", dim, i, j, dot)
				}
			}
		}
	}
}
