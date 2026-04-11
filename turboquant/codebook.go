package turboquant

import (
	"math"
	"slices"
	"sync"
)

type codebookCacheKey struct {
	dim  int
	bits int
}

type scalarCodebookCacheValue struct {
	codebook   []float32
	boundaries []float32
}

var scalarCodebookCache sync.Map

// ExportCodebook returns the Lloyd-Max codebook centroids for the given
// dim and bits. Used by the CUDA dequant kernel (loaded into GPU constant memory).
func ExportCodebook(dim, bits int) []float32 {
	cb, _ := scalarCodebook(dim, bits)
	return cb
}

// ExportBoundaries returns the Lloyd-Max decision boundaries for the given
// dim and bits. Used by the CUDA encode kernel for binary-search quantization.
// Boundaries are the midpoints between adjacent centroids; len = (1<<bits) - 1.
func ExportBoundaries(dim, bits int) []float32 {
	_, boundaries := scalarCodebook(dim, bits)
	return boundaries
}

func scalarCodebook(dim int, bits int) ([]float32, []float32) {
	key := codebookCacheKey{dim: dim, bits: bits}
	if cached, ok := scalarCodebookCache.Load(key); ok {
		value := cached.(scalarCodebookCacheValue)
		return append([]float32(nil), value.codebook...), append([]float32(nil), value.boundaries...)
	}

	codebook := buildLloydMaxCodebook(dim, bits)
	value := scalarCodebookCacheValue{
		codebook:   codebook,
		boundaries: codebookBoundaries(codebook),
	}
	actual, _ := scalarCodebookCache.LoadOrStore(key, value)
	cached := actual.(scalarCodebookCacheValue)
	return append([]float32(nil), cached.codebook...), append([]float32(nil), cached.boundaries...)
}

func buildLloydMaxCodebook(dim int, bits int) []float32 {
	levels := 1 << bits
	if levels <= 1 {
		return []float32{0}
	}

	samples := unitVectorCoordSamples(dim, bits, 65536)
	slices.Sort(samples)

	centroids := make([]float64, levels)
	for level := 0; level < levels; level++ {
		begin := level * len(samples) / levels
		end := (level + 1) * len(samples) / levels
		if end <= begin {
			end = begin + 1
		}
		centroids[level] = meanFloat64(samples[begin:end])
	}

	for iter := 0; iter < 48; iter++ {
		slices.Sort(centroids)
		bounds := make([]float64, levels-1)
		for i := range bounds {
			bounds[i] = (centroids[i] + centroids[i+1]) / 2
		}

		sums := make([]float64, levels)
		counts := make([]int, levels)
		for _, sample := range samples {
			idx := quantizeScalarFloat64(sample, bounds)
			sums[idx] += sample
			counts[idx]++
		}

		maxDelta := 0.0
		for i := range centroids {
			next := centroids[i]
			if counts[i] > 0 {
				next = sums[i] / float64(counts[i])
			} else if i == 0 {
				next = bounds[0] - 0.25
			} else if i == len(centroids)-1 {
				next = bounds[len(bounds)-1] + 0.25
			} else {
				next = (bounds[i-1] + bounds[i]) / 2
			}
			maxDelta = math.Max(maxDelta, math.Abs(next-centroids[i]))
			centroids[i] = next
		}
		if maxDelta < 1e-6 {
			break
		}
	}

	slices.Sort(centroids)
	codebook := make([]float32, len(centroids))
	for i := range centroids {
		codebook[i] = float32(centroids[i])
	}
	return codebook
}

// unitVectorCoordSamples returns count samples from the exact marginal
// distribution of a single coordinate of a uniformly random unit vector in R^d:
//
//	z_1 / ‖z‖ · √d,  z ~ N(0, I_d)
//
// For large d this converges to N(0,1); for smaller d the heavier tails of the
// Beta((d-3)/2,(d-3)/2) coordinate distribution are preserved. Using this
// distribution — rather than pure N(0,1) — produces the optimal Lloyd-Max
// codebook for the actual coordinate distribution that arises after RMS
// normalization and random rotation (Paper §3.1, Eq. 4, Lemma 1).
func unitVectorCoordSamples(dim int, bits int, count int) []float64 {
	rng := splitmix64(uint64(bits+1)<<48 ^ uint64(dim+1)<<16 ^ 0x4d595df4d0f33173)
	out := make([]float64, count)
	if dim <= 1 {
		for i := range out {
			out[i] = gaussianFloat64(&rng)
		}
		return out
	}
	sqrtDim := math.Sqrt(float64(dim))
	for i := range out {
		z0 := gaussianFloat64(&rng)
		sumSq := z0 * z0
		for k := 1; k < dim; k++ {
			g := gaussianFloat64(&rng)
			sumSq += g * g
		}
		norm := math.Sqrt(sumSq)
		if norm < 1e-15 {
			out[i] = 0
		} else {
			out[i] = z0 / norm * sqrtDim
		}
	}
	return out
}

func meanFloat64(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}
	total := 0.0
	for _, value := range values {
		total += value
	}
	return total / float64(len(values))
}

func codebookBoundaries(codebook []float32) []float32 {
	if len(codebook) < 2 {
		return nil
	}

	out := make([]float32, len(codebook)-1)
	for i := range out {
		out[i] = (codebook[i] + codebook[i+1]) / 2
	}
	return out
}

func quantizeScalarByBoundary(v float32, codebook []float32, boundaries []float32) uint8 {
	if len(codebook) == 0 {
		return 0
	}
	if len(boundaries) != len(codebook)-1 {
		return quantizeScalarNearest(v, codebook)
	}

	idx := 0
	for idx < len(boundaries) && v >= boundaries[idx] {
		idx++
	}
	return uint8(idx)
}

func quantizeScalarFloat64(v float64, boundaries []float64) int {
	idx := 0
	for idx < len(boundaries) && v >= boundaries[idx] {
		idx++
	}
	return idx
}

func quantizeScalarNearest(v float32, codebook []float32) uint8 {
	best := 0
	bestDist := float32(math.MaxFloat32)
	for i, centroid := range codebook {
		d := abs32(v - centroid)
		if d < bestDist {
			bestDist = d
			best = i
		}
	}
	return uint8(best)
}

func dequantizeScalar(idx uint8, codebook []float32) float32 {
	if int(idx) >= len(codebook) {
		return 0
	}
	return codebook[idx]
}

