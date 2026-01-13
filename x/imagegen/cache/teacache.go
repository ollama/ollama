//go:build mlx

// Package cache provides caching mechanisms for diffusion model inference.
package cache

import (
	"github.com/ollama/ollama/x/imagegen/mlx"
)

// TeaCache implements Timestep Embedding Aware Caching for diffusion models.
// It caches the transformer output and reuses it when timestep values
// are similar between consecutive steps.
//
// For CFG (classifier-free guidance), it caches pos and neg predictions
// separately and always computes CFG fresh to avoid error amplification.
//
// Reference: "Timestep Embedding Tells: It's Time to Cache for Video Diffusion Model"
// https://github.com/ali-vilab/TeaCache
type TeaCache struct {
	// Cached transformer output from last computed step (non-CFG mode)
	cachedOutput *mlx.Array

	// Cached CFG outputs (pos and neg separately)
	cachedPosOutput *mlx.Array
	cachedNegOutput *mlx.Array

	// Previous timestep value for difference calculation
	prevTimestep float32

	// Accumulated difference for rescaling
	accumulatedDiff float32

	// Configuration
	threshold      float32 // Threshold for recomputation decision
	rescaleFactor  float32 // Model-specific rescaling factor
	skipEarlySteps int     // Number of early steps to never cache

	// Statistics
	cacheHits   int
	cacheMisses int
}

// TeaCacheConfig holds configuration for TeaCache.
type TeaCacheConfig struct {
	// Threshold for recomputation. Lower = more cache hits, potential quality loss.
	// Recommended: 0.05-0.15 for image models
	Threshold float32

	// Rescale factor to adjust timestep embedding differences.
	// Model-specific, typically 1.0-2.0
	RescaleFactor float32

	// SkipEarlySteps: number of early steps to always compute (never cache).
	// Set to 2-3 for CFG mode to preserve structure. 0 = no skipping.
	SkipEarlySteps int
}

// DefaultTeaCacheConfig returns default configuration for TeaCache.
func DefaultTeaCacheConfig() *TeaCacheConfig {
	return &TeaCacheConfig{
		Threshold:     0.1,
		RescaleFactor: 1.0,
	}
}

// NewTeaCache creates a new TeaCache instance.
func NewTeaCache(cfg *TeaCacheConfig) *TeaCache {
	if cfg == nil {
		cfg = DefaultTeaCacheConfig()
	}
	return &TeaCache{
		threshold:      cfg.Threshold,
		rescaleFactor:  cfg.RescaleFactor,
		skipEarlySteps: cfg.SkipEarlySteps,
	}
}

// ShouldCompute determines if we should compute the full forward pass
// or reuse the cached output based on timestep similarity.
//
// Algorithm:
// 1. First step always computes
// 2. Subsequent steps compare |currTimestep - prevTimestep| * rescaleFactor
// 3. If accumulated difference > threshold, compute new output
// 4. Otherwise, reuse cached output
func (tc *TeaCache) ShouldCompute(step int, timestep float32) bool {
	// Always compute early steps (critical for structure)
	// Check both regular cache and CFG cache
	hasCachedOutput := tc.cachedOutput != nil || tc.HasCFGCache()
	if step < tc.skipEarlySteps || step == 0 || !hasCachedOutput {
		return true
	}

	// Compute absolute difference between current and previous timestep
	diff := timestep - tc.prevTimestep
	if diff < 0 {
		diff = -diff
	}

	// Apply rescaling factor
	scaledDiff := diff * tc.rescaleFactor

	// Accumulate difference (helps track drift over multiple cached steps)
	tc.accumulatedDiff += scaledDiff

	// Decision based on accumulated difference
	if tc.accumulatedDiff > tc.threshold {
		tc.accumulatedDiff = 0 // Reset accumulator
		return true
	}

	return false
}

// UpdateCache stores the computed output for potential reuse (non-CFG mode).
func (tc *TeaCache) UpdateCache(output *mlx.Array, timestep float32) {
	// Free previous cached output
	if tc.cachedOutput != nil {
		tc.cachedOutput.Free()
	}

	// Store new cached values
	tc.cachedOutput = output
	tc.prevTimestep = timestep
	tc.cacheMisses++
}

// UpdateCFGCache stores pos and neg outputs separately for CFG mode.
// This allows CFG to be computed fresh each step, avoiding error amplification.
func (tc *TeaCache) UpdateCFGCache(posOutput, negOutput *mlx.Array, timestep float32) {
	// Free previous cached outputs
	if tc.cachedPosOutput != nil {
		tc.cachedPosOutput.Free()
	}
	if tc.cachedNegOutput != nil {
		tc.cachedNegOutput.Free()
	}

	// Store new cached values
	tc.cachedPosOutput = posOutput
	tc.cachedNegOutput = negOutput
	tc.prevTimestep = timestep
	tc.cacheMisses++
}

// GetCached returns the cached output (non-CFG mode).
func (tc *TeaCache) GetCached() *mlx.Array {
	tc.cacheHits++
	return tc.cachedOutput
}

// GetCFGCached returns cached pos and neg outputs for CFG mode.
func (tc *TeaCache) GetCFGCached() (pos, neg *mlx.Array) {
	tc.cacheHits++
	return tc.cachedPosOutput, tc.cachedNegOutput
}

// HasCFGCache returns true if CFG cache is available.
func (tc *TeaCache) HasCFGCache() bool {
	return tc.cachedPosOutput != nil && tc.cachedNegOutput != nil
}

// Arrays returns all arrays that should be kept alive.
func (tc *TeaCache) Arrays() []*mlx.Array {
	var arrays []*mlx.Array
	if tc.cachedOutput != nil {
		arrays = append(arrays, tc.cachedOutput)
	}
	if tc.cachedPosOutput != nil {
		arrays = append(arrays, tc.cachedPosOutput)
	}
	if tc.cachedNegOutput != nil {
		arrays = append(arrays, tc.cachedNegOutput)
	}
	return arrays
}

// Stats returns cache hit/miss statistics.
func (tc *TeaCache) Stats() (hits, misses int) {
	return tc.cacheHits, tc.cacheMisses
}

// Free releases all cached arrays.
func (tc *TeaCache) Free() {
	if tc.cachedOutput != nil {
		tc.cachedOutput.Free()
		tc.cachedOutput = nil
	}
	if tc.cachedPosOutput != nil {
		tc.cachedPosOutput.Free()
		tc.cachedPosOutput = nil
	}
	if tc.cachedNegOutput != nil {
		tc.cachedNegOutput.Free()
		tc.cachedNegOutput = nil
	}
}
