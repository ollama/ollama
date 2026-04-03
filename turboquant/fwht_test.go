package turboquant

import (
	"math"
	"testing"
)

func TestFWHTRoundTrip(t *testing.T) {
	// Forward then inverse should recover the original vector
	for _, dim := range []int{4, 8, 16, 32, 64, 128, 256} {
		t.Run(dimName(dim), func(t *testing.T) {
			original := make([]float32, dim)
			for i := range original {
				original[i] = float32(i) * 0.1
			}

			data := make([]float32, dim)
			copy(data, original)

			ApplyFWHT(data, TurboQuantSeed, false)  // forward
			ApplyFWHT(data, TurboQuantSeed, true)    // inverse

			for i := range data {
				diff := math.Abs(float64(data[i] - original[i]))
				if diff > 1e-4 {
					t.Fatalf("dim=%d elem[%d]: got %.6f, want %.6f (diff %.6f)",
						dim, i, data[i], original[i], diff)
				}
			}
		})
	}
}

func TestFWHTNormalizeRoundTrip(t *testing.T) {
	for _, dim := range []int{4, 8, 16, 32, 64, 128, 256} {
		t.Run(dimName(dim), func(t *testing.T) {
			original := make([]float32, dim)
			for i := range original {
				original[i] = float32(i) * 0.1
			}

			normBefore := float32(l2norm(original))
			if normBefore < 1e-12 {
				normBefore = 1.0
			}
			
			// Normalize
			data := make([]float32, dim)
			for i := range data {
				data[i] = original[i] / normBefore
			}

			// Forward
			ApplyFWHT(data, TurboQuantSeed, false)
			
			// Inverse
			ApplyFWHT(data, TurboQuantSeed, true)
			
			// De-normalize
			for i := range data {
				data[i] = data[i] * normBefore
			}

			for i := range data {
				diff := math.Abs(float64(data[i] - original[i]))
				if diff > 1e-4 {
					t.Fatalf("dim=%d elem[%d]: got %.6f, want %.6f (diff %.6f)",
						dim, i, data[i], original[i], diff)
				}
			}
		})
	}
}

func TestFWHTPreservesNorm(t *testing.T) {
	for _, dim := range []int{8, 32, 128, 256} {
		t.Run(dimName(dim), func(t *testing.T) {
			data := make([]float32, dim)
			for i := range data {
				data[i] = float32(i+1) * 0.01
			}

			normBefore := l2norm(data)
			ApplyFWHT(data, TurboQuantSeed, false)
			normAfter := l2norm(data)

			relErr := math.Abs(normAfter-normBefore) / normBefore
			if relErr > 1e-5 {
				t.Fatalf("norm not preserved: before=%.6f after=%.6f relErr=%.6f",
					normBefore, normAfter, relErr)
			}
		})
	}
}

func TestFWHTIsDeterministic(t *testing.T) {
	dim := 64
	data1 := make([]float32, dim)
	data2 := make([]float32, dim)
	for i := range data1 {
		data1[i] = float32(i) * 0.5
		data2[i] = float32(i) * 0.5
	}

	ApplyFWHT(data1, TurboQuantSeed, false)
	ApplyFWHT(data2, TurboQuantSeed, false)

	for i := range data1 {
		if data1[i] != data2[i] {
			t.Fatalf("not deterministic at elem[%d]: %.6f vs %.6f", i, data1[i], data2[i])
		}
	}
}

func TestFWHTDifferentSeedsProduceDifferentResults(t *testing.T) {
	dim := 64
	data1 := make([]float32, dim)
	data2 := make([]float32, dim)
	for i := range data1 {
		data1[i] = float32(i) + 1.0
		data2[i] = float32(i) + 1.0
	}

	ApplyFWHT(data1, TurboQuantSeed, false)
	ApplyFWHT(data2, TurboQuantSeed+1, false)

	same := true
	for i := range data1 {
		if data1[i] != data2[i] {
			same = false
			break
		}
	}
	if same {
		t.Fatal("different seeds produced identical results")
	}
}

func TestFWHTDistributesInformationUniformly(t *testing.T) {
	// A vector with a single outlier should spread its energy across all dimensions
	dim := 128
	data := make([]float32, dim)
	data[0] = 100.0 // single large outlier

	ApplyFWHT(data, TurboQuantSeed, false)

	// After rotation, no single element should dominate
	maxAbs := float32(0)
	for _, v := range data {
		if abs := float32(math.Abs(float64(v))); abs > maxAbs {
			maxAbs = abs
		}
	}

	// Expected: each element ≈ 100/√128 ≈ 8.84
	// Allow generous tolerance but ensure max is well below 100
	expected := float32(100.0 / math.Sqrt(float64(dim)))
	if maxAbs > expected*2.0 {
		t.Fatalf("outlier not distributed: maxAbs=%.2f expected≈%.2f", maxAbs, expected)
	}
}

func TestFWHTRejectsNonPowerOf2(t *testing.T) {
	data := make([]float32, 3)
	data[0] = 1.0
	data[1] = 2.0
	data[2] = 3.0

	original := make([]float32, 3)
	copy(original, data)

	ApplyFWHT(data, TurboQuantSeed, false)

	// Should be unchanged (no-op for non-power-of-2)
	for i := range data {
		if data[i] != original[i] {
			t.Fatalf("should be no-op for dim=3, elem[%d] changed: %.6f -> %.6f",
				i, original[i], data[i])
		}
	}
}

func BenchmarkFWHT128(b *testing.B) {
	benchmarkFWHT(b, 128)
}

func BenchmarkFWHT256(b *testing.B) {
	benchmarkFWHT(b, 256)
}

func BenchmarkDenseRotation128(b *testing.B) {
	benchmarkDense(b, 128)
}

func BenchmarkDenseRotation256(b *testing.B) {
	benchmarkDense(b, 256)
}

func benchmarkFWHT(b *testing.B, dim int) {
	data := make([]float32, dim)
	for i := range data {
		data[i] = float32(i) * 0.01
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ApplyFWHT(data, TurboQuantSeed, false)
	}
}

func benchmarkDense(b *testing.B, dim int) {
	matrix := GenerateRotation(dim, TurboQuantSeed)
	x := make([]float32, dim)
	result := make([]float32, dim)
	for i := range x {
		x[i] = float32(i) * 0.01
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		// Dense matrix-vector multiply (column-major)
		for j := 0; j < dim; j++ {
			var sum float32
			for k := 0; k < dim; k++ {
				sum += matrix[j*dim+k] * x[k]
			}
			result[j] = sum
		}
	}
}

func l2norm(v []float32) float64 {
	var sum float64
	for _, x := range v {
		sum += float64(x) * float64(x)
	}
	return math.Sqrt(sum)
}

func dimName(d int) string {
	switch {
	case d >= 1024:
		return string(rune('0'+d/1024)) + "k"
	default:
		return string(rune('0'+d/100)) + string(rune('0'+d%100/10)) + string(rune('0'+d%10))
	}
}
