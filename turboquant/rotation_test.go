package turboquant

import (
	"fmt"
	"math"
	"testing"
)

// testDims covers small and production dims. WHT requires powers of 2.
var testDims = []int{4, 8, 16, 32, 64, 128}

func TestBuildRotationDeterministic(t *testing.T) {
	for _, dim := range testDims {
		a := BuildRotation(dim, 123)
		b := BuildRotation(dim, 123)
		if a.Dim != b.Dim || a.Seed != b.Seed || len(a.Signs) != len(b.Signs) {
			t.Fatalf("rotation metadata mismatch for dim %d", dim)
		}
		for i := range a.Signs {
			if a.Signs[i] != b.Signs[i] {
				t.Fatalf("rotation sign mismatch at dim=%d idx=%d", dim, i)
			}
		}
	}
}

func TestBuildRotationDifferentSeedsDiffer(t *testing.T) {
	a := BuildRotation(16, 111)
	b := BuildRotation(16, 222)
	different := false
	for i := range a.Signs {
		if a.Signs[i] != b.Signs[i] {
			different = true
			break
		}
	}
	if !different {
		t.Fatal("different seeds produced the same sign vector")
	}
}

func TestBuildRotationSignsArePlusMinus1(t *testing.T) {
	for _, dim := range testDims {
		rot := BuildRotation(dim, uint64(dim)*31)
		for i, s := range rot.Signs {
			if s != 1.0 && s != -1.0 {
				t.Fatalf("dim=%d idx=%d sign=%v, want ±1", dim, i, s)
			}
		}
	}
}

// TestApplyInverseRotation verifies self-inverse property: F(F(x)) == x.
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

// TestAttentionScoreRotationInvariance verifies dot(F(q), F(k)) == dot(q, k).
// Because F is orthogonal, the inner product is preserved. This is the core
// property that makes rotating K before quantization safe when Q is also rotated.
func TestAttentionScoreRotationInvariance(t *testing.T) {
	for _, dim := range []int{64, 128, 256, 512} {
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
				t.Errorf("dim=%d: dot(q,k)=%.6f dot(F(q),F(k))=%.6f relErr=%.2e",
					dim, dotOrig, dotRot, relErr)
			}
		})
	}
}

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
