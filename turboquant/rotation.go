package turboquant

import (
	"math"
	"sync"
)

// Rotation holds the randomised Walsh-Hadamard sign vector for TurboQuant.
// The transform F(x) = S·H·S·x/√Dim (symmetric variant) is self-inverse:
// F(F(x)) = x. This eliminates the separate R / R^T pair from the Householder
// QR path and keeps the dot-product invariant F(q)·F(k) = q·k.
// Signs are ±1.0 float32 derived from Seed via splitmix64.
type Rotation struct {
	Dim   int
	Seed  uint64
	Signs []float32 // [Dim] ±1.0
}

type rotationCacheKey struct {
	dim  int
	seed uint64
}

var rotationCache sync.Map

func BuildRotation(dim int, seed uint64) Rotation {
	key := rotationCacheKey{dim: dim, seed: seed}
	if cached, ok := rotationCache.Load(key); ok {
		return cached.(Rotation)
	}

	var signs []float32
	if dim > 0 && (dim&(dim-1)) == 0 {
		signs = buildSignVector(dim, seed)
	}
	// Non-power-of-2 dims: Signs == nil → identity (no rotation).

	rot := Rotation{Dim: dim, Seed: seed, Signs: signs}
	actual, _ := rotationCache.LoadOrStore(key, rot)
	return actual.(Rotation)
}

// ApplyRotation applies the symmetric randomised WHT F(x) = S·H·S·x/√Dim.
// F is self-inverse, so ApplyInverseRotation calls this function.
// If rot.Signs is nil (non-power-of-2 dim), returns a copy with no rotation.
func ApplyRotation(x []float32, rot Rotation) []float32 {
	if len(x) != rot.Dim {
		panic("turboquant: vector length does not match rotation dimension")
	}
	out := make([]float32, rot.Dim)
	copy(out, x)
	if rot.Signs != nil {
		applySHSInPlace(out, rot.Signs)
	}
	return out
}

// ApplyInverseRotation is identical to ApplyRotation because F is self-inverse.
func ApplyInverseRotation(y []float32, rot Rotation) []float32 {
	return ApplyRotation(y, rot)
}

// applySHSInPlace applies S·H·S·x/√n in-place where S = diag(signs).
func applySHSInPlace(x []float32, signs []float32) {
	n := len(x)
	scale := float32(1.0 / math.Sqrt(float64(n)))

	// First S: elementwise multiply by signs
	for i := range n {
		x[i] *= signs[i]
	}

	// H: Walsh-Hadamard butterfly
	whtInPlace(x)

	// Second S and normalise
	for i := range n {
		x[i] *= signs[i] * scale
	}
}

// whtInPlace applies the unnormalised Walsh-Hadamard transform in-place.
// n must be a power of 2.
func whtInPlace(x []float32) {
	n := len(x)
	for stride := 1; stride < n; stride <<= 1 {
		for i := 0; i < n; i += stride * 2 {
			for j := i; j < i+stride; j++ {
				a, b := x[j], x[j+stride]
				x[j], x[j+stride] = a+b, a-b
			}
		}
	}
}

// buildSignVector generates a [dim] ±1 float32 sign vector from seed.
// Each bit of the splitmix64 output maps to one sign.
func buildSignVector(dim int, seed uint64) []float32 {
	signs := make([]float32, dim)
	rng := splitmix64(seed ^ uint64(dim)<<32 ^ 0x9e3779b97f4a7c15)
	for i := 0; i < dim; i += 64 {
		bits := rng.next()
		for j := 0; j < 64 && i+j < dim; j++ {
			if (bits>>j)&1 == 1 {
				signs[i+j] = 1.0
			} else {
				signs[i+j] = -1.0
			}
		}
	}
	return signs
}

type splitmix64 uint64

func (s *splitmix64) next() uint64 {
	*s += 0x9e3779b97f4a7c15
	z := uint64(*s)
	z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9
	z = (z ^ (z >> 27)) * 0x94d049bb133111eb
	return z ^ (z >> 31)
}

func gaussianFloat64(rng *splitmix64) float64 {
	u1 := unitUniform64(rng)
	u2 := unitUniform64(rng)
	return math.Sqrt(-2*math.Log(u1)) * math.Cos(2*math.Pi*u2)
}

func unitUniform64(rng *splitmix64) float64 {
	const scale = 1.0 / (1 << 53)
	return (float64(rng.next()>>11) + 0.5) * scale
}
