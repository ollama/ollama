package turboquant

import "math"

// ApplyFWHT applies the Fast Walsh-Hadamard Transform rotation in-place.
// This is a Go reference implementation matching the GGML FWHT op exactly
// (same xorshift64 PRNG, same butterfly structure).
//
// Forward (inverse=false): y = (1/√d) * H * D * x
// Inverse (inverse=true):  y = (1/√d) * D * H * x
//
// where D is a random ±1 diagonal from seed, H is the Hadamard matrix
// applied via butterfly operations.
func ApplyFWHT(data []float32, seed uint64, inverse bool) {
	d := len(data)
	if d == 0 || !isPowerOf2(d) {
		return
	}

	signs := generateSigns(d, seed)
	scale := float32(1.0 / math.Sqrt(float64(d)))

	if !inverse {
		// Forward: D first, then H
		for i := 0; i < d; i++ {
			data[i] *= signs[i]
		}
		butterflyHadamard(data)
	} else {
		// Inverse: H first, then D
		butterflyHadamard(data)
		for i := 0; i < d; i++ {
			data[i] *= signs[i]
		}
	}

	for i := 0; i < d; i++ {
		data[i] *= scale
	}
}

// butterflyHadamard applies the in-place Walsh-Hadamard butterfly transform.
func butterflyHadamard(data []float32) {
	d := len(data)
	for length := 1; length < d; length <<= 1 {
		for j := 0; j < d; j += length << 1 {
			for k := 0; k < length; k++ {
				a := data[j+k]
				b := data[j+k+length]
				data[j+k] = a + b
				data[j+k+length] = a - b
			}
		}
	}
}

// generateSigns generates the random ±1 diagonal using xorshift64,
// matching the GGML FWHT kernel exactly.
func generateSigns(d int, seed uint64) []float32 {
	rng := seed
	signs := make([]float32, d)
	for i := 0; i < d; i++ {
		rng ^= rng << 13
		rng ^= rng >> 7
		rng ^= rng << 17
		if rng&1 != 0 {
			signs[i] = 1.0
		} else {
			signs[i] = -1.0
		}
	}
	return signs
}
