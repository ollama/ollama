package turboquant

import "math"

// ApplyFWHT applies the Fast Walsh-Hadamard Transform rotation in-place.
// This is a Go reference implementation matching the GGML FWHT op exactly
// (same splitmix64 PRNG, same butterfly structure).
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

	scale := float32(1.0 / math.Sqrt(float64(d)))

	if !inverse {
		// Forward: D first, then H
		for i := 0; i < d; i++ {
			data[i] *= splitmixSign(seed, i)
		}
		butterflyHadamard(data)
	} else {
		// Inverse: H first, then D
		butterflyHadamard(data)
		for i := 0; i < d; i++ {
			data[i] *= splitmixSign(seed, i)
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

// splitmixSign returns +1 or -1 for position pos using splitmix64 finalizer.
// Matches the GGML FWHT/TQ kernels exactly (position-independent, O(1) per element).
func splitmixSign(seed uint64, pos int) float32 {
	x := seed + uint64(pos)*0x9E3779B97F4A7C15
	x ^= x >> 30
	x *= 0xBF58476D1CE4E5B9
	x ^= x >> 27
	x *= 0x94D049BB133111EB
	x ^= x >> 31
	if x&1 != 0 {
		return 1.0
	}
	return -1.0
}
