package llm

import (
	"math"
	"sync"
)

// maxCalibrationSamples bounds how many context lengths are remembered per key. A straight
// line needs two; the rest are kept so a fit is not dominated by one unusual load.
const maxCalibrationSamples = 8

// CalibrationKey identifies loads whose memory use is comparable.
//
// Everything that moves the curve belongs in here. Samples are looked up by the whole key,
// so changing any of these selects a different bucket and the old samples stop being
// consulted — that is how a sample is invalidated when the thing it described changes.
// ModelSize is included because a path can be rewritten to hold different weights.
type CalibrationKey struct {
	Model          string
	ModelSize      uint64
	KVCacheType    string
	FlashAttention bool
	NumBatch       int
	NumParallel    int
	NumGPU         int
}

type calibrationSample struct {
	numCtx int
	vram   uint64
}

// VRAMCalibration remembers how much VRAM a model actually used, so that a later load of
// the same model can be predicted from measurement rather than from metadata alone.
//
// Measured VRAM is close to linear in context length: across two architectures at four
// context lengths, the residual of a straight-line fit stayed under 30 MiB. Two loads at
// different context lengths are therefore enough to predict a third, including behaviour
// the metadata does not describe.
//
// That matters because the metadata prediction models one KV cache of the published
// dimensions. An architecture that allocates several caches, or whose compute buffers grow
// with context, uses more than that, by an amount proportional to context — and in the
// direction that overcommits a device. A measurement covers all of it without needing to
// know which effect produced it.
//
// The zero value is not usable; call NewVRAMCalibration.
type VRAMCalibration struct {
	mu      sync.Mutex
	samples map[CalibrationKey][]calibrationSample
}

func NewVRAMCalibration() *VRAMCalibration {
	return &VRAMCalibration{samples: make(map[CalibrationKey][]calibrationSample)}
}

// Record stores what a completed load actually used. A second measurement at a context
// length already recorded replaces it rather than being added alongside, so a key always
// reflects the most recent load at each context length.
func (c *VRAMCalibration) Record(key CalibrationKey, numCtx int, vram uint64) {
	if c == nil || numCtx <= 0 || vram == 0 {
		return
	}

	c.mu.Lock()
	defer c.mu.Unlock()

	samples := c.samples[key]
	for i := range samples {
		if samples[i].numCtx == numCtx {
			samples[i].vram = vram
			return
		}
	}

	samples = append(samples, calibrationSample{numCtx: numCtx, vram: vram})
	if len(samples) > maxCalibrationSamples {
		samples = samples[len(samples)-maxCalibrationSamples:]
	}
	c.samples[key] = samples
}

// Predict estimates VRAM for a load at numCtx, preferring measurement over metadata.
//
// base and bytesPerToken are the metadata prediction, used as the prior: with no samples
// it is returned unchanged, and with a single sample it supplies the slope through that
// measured point. The second return reports whether measurement was used, which is worth
// logging — a prediction backed by a sample deserves different trust than one that is not.
func (c *VRAMCalibration) Predict(key CalibrationKey, numCtx int, base, bytesPerToken uint64) (uint64, bool) {
	prior := addScaled(base, bytesPerToken, max(numCtx, 0))
	if c == nil || numCtx <= 0 {
		return prior, false
	}

	c.mu.Lock()
	samples := append([]calibrationSample(nil), c.samples[key]...)
	c.mu.Unlock()

	switch {
	case len(samples) == 0:
		return prior, false

	case !hasDistinctContexts(samples):
		// One measured point, so the slope has to come from the prior. Anchoring on the
		// measurement still corrects the whole context-independent part, which is the
		// larger of the two terms for every model of consequence.
		s := samples[0]
		return addScaled(s.vram, bytesPerToken, numCtx-s.numCtx), true

	default:
		intercept, slope := leastSquares(samples)
		if slope < 0 {
			// Memory does not shrink as context grows; a negative slope means the samples
			// disagree for some reason this model cannot express, so fall back rather than
			// extrapolate a nonsense line.
			return prior, false
		}
		return clampFloat(intercept + slope*float64(numCtx)), true
	}
}

func hasDistinctContexts(samples []calibrationSample) bool {
	for _, s := range samples[1:] {
		if s.numCtx != samples[0].numCtx {
			return true
		}
	}
	return false
}

// leastSquares fits vram = intercept + slope*numCtx over the samples.
func leastSquares(samples []calibrationSample) (intercept, slope float64) {
	n := float64(len(samples))
	var sumX, sumY float64
	for _, s := range samples {
		sumX += float64(s.numCtx)
		sumY += float64(s.vram)
	}
	meanX, meanY := sumX/n, sumY/n

	var num, den float64
	for _, s := range samples {
		dx := float64(s.numCtx) - meanX
		num += dx * (float64(s.vram) - meanY)
		den += dx * dx
	}
	if den == 0 {
		return meanY, 0
	}
	slope = num / den
	return meanY - slope*meanX, slope
}

// addScaled returns base + perToken*tokens, saturating instead of wrapping. tokens may be
// negative, which is how a single sample is extrapolated down to a shorter context. These
// are sizes in bytes, so a result that would go below zero or past the width of the type is
// meaningless either way; saturating keeps a nonsense input from becoming a plausible-
// looking number.
func addScaled(base, perToken uint64, tokens int) uint64 {
	if tokens == 0 || perToken == 0 {
		return base
	}

	magnitude := uint64(tokens)
	if tokens < 0 {
		magnitude = uint64(-tokens)
	}

	product := perToken * magnitude
	if product/magnitude != perToken {
		product = math.MaxUint64
	}

	if tokens < 0 {
		if product > base {
			return 0
		}
		return base - product
	}
	if base > math.MaxUint64-product {
		return math.MaxUint64
	}
	return base + product
}

func clampFloat(v float64) uint64 {
	switch {
	case math.IsNaN(v), v <= 0:
		return 0
	case v >= math.MaxUint64:
		return math.MaxUint64
	default:
		return uint64(v)
	}
}
