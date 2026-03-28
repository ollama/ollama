package turboquant

import (
	"math"
	"math/rand/v2"
)

// GenerateQJLProjection creates a d x d random Gaussian projection matrix S
// for the Quantized Johnson-Lindenstrauss transform.
//
// Each entry S[i][j] ~ N(0, 1) i.i.d., seeded deterministically.
// The returned slice is in column-major (GGML) order: data[j*d + i] = S[i][j].
func GenerateQJLProjection(dim int, seed uint64) []float32 {
	qjlSeed := seed ^ 0x514A_4C5F_5052_4F4A // "QJL_PROJ" XOR mask
	rng := rand.New(rand.NewPCG(qjlSeed, qjlSeed^0xABCD_EF01_2345_6789))

	result := make([]float32, dim*dim)
	for j := 0; j < dim; j++ {
		for i := 0; i < dim; i++ {
			result[j*dim+i] = float32(rng.NormFloat64())
		}
	}

	return result
}

// QJLDequantScale returns the scalar (sqrt(pi/2) / d) used in QJL dequantization.
// The full dequantization is: x_qjl = scale * gamma * S^T * signs
// where gamma is the residual L2 norm and signs are the 1-bit quantized values.
func QJLDequantScale(dim int) float64 {
	return math.Sqrt(math.Pi/2.0) / float64(dim)
}
