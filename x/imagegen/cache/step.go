//go:build mlx

package cache

import "github.com/ollama/ollama/x/imagegen/mlx"

// StepCache caches layer outputs across diffusion denoising steps.
// Based on DeepCache (CVPR 2024) and Learning-to-Cache (NeurIPS 2024):
// shallow layers change little between consecutive steps, so we can
// cache their outputs and skip recomputation on non-refresh steps.
//
// Supports both single-stream (Z-Image) and dual-stream (Qwen-Image) architectures:
//   - Single-stream: use Get/Set for the single output per layer
//   - Dual-stream: use Get/Set for stream 1 (imgH), Get2/Set2 for stream 2 (txtH)
//
// Usage (single-stream):
//
//	cache := NewStepCache(15)  // cache first 15 layers
//	for step := 0; step < numSteps; step++ {
//	    refresh := cache.ShouldRefresh(step, 3)  // refresh every 3 steps
//	    for i, layer := range layers {
//	        if i < 15 && !refresh && cache.Get(i) != nil {
//	            output = cache.Get(i)  // reuse cached
//	        } else {
//	            output = layer.Forward(input)
//	            if i < 15 && refresh {
//	                cache.Set(i, output)
//	            }
//	        }
//	    }
//	}
//	cache.Free()  // cleanup when done
//
// Usage (dual-stream):
//
//	cache := NewStepCache(15)
//	for step := 0; step < numSteps; step++ {
//	    refresh := cache.ShouldRefresh(step, 3)
//	    for i, layer := range layers {
//	        if i < 15 && !refresh && cache.Get(i) != nil {
//	            imgH, txtH = cache.Get(i), cache.Get2(i)
//	        } else {
//	            imgH, txtH = layer.Forward(imgH, txtH, ...)
//	            if i < 15 && refresh {
//	                cache.Set(i, imgH)
//	                cache.Set2(i, txtH)
//	            }
//	        }
//	    }
//	}
type StepCache struct {
	layers   []*mlx.Array // cached layer outputs (stream 1)
	layers2  []*mlx.Array // cached layer outputs (stream 2, for dual-stream models)
	constant *mlx.Array   // optional constant (e.g., text embeddings)
}

// NewStepCache creates a cache for the given number of layers.
func NewStepCache(numLayers int) *StepCache {
	return &StepCache{
		layers:  make([]*mlx.Array, numLayers),
		layers2: make([]*mlx.Array, numLayers),
	}
}

// ShouldRefresh returns true if the cache should be refreshed at this step.
// Refresh happens on step 0, interval, 2*interval, etc.
func (c *StepCache) ShouldRefresh(step, interval int) bool {
	return step%interval == 0
}

// Get returns the cached output for a layer, or nil if not cached.
func (c *StepCache) Get(layer int) *mlx.Array {
	if layer < len(c.layers) {
		return c.layers[layer]
	}
	return nil
}

// Set stores a layer output (stream 1), freeing any previous value.
func (c *StepCache) Set(layer int, arr *mlx.Array) {
	if layer < len(c.layers) {
		if c.layers[layer] != nil {
			c.layers[layer].Free()
		}
		c.layers[layer] = arr
	}
}

// Get2 returns the cached output for a layer (stream 2), or nil if not cached.
// Used for dual-stream architectures like Qwen-Image.
func (c *StepCache) Get2(layer int) *mlx.Array {
	if layer < len(c.layers2) {
		return c.layers2[layer]
	}
	return nil
}

// Set2 stores a layer output (stream 2), freeing any previous value.
// Used for dual-stream architectures like Qwen-Image.
func (c *StepCache) Set2(layer int, arr *mlx.Array) {
	if layer < len(c.layers2) {
		if c.layers2[layer] != nil {
			c.layers2[layer].Free()
		}
		c.layers2[layer] = arr
	}
}

// GetConstant returns the cached constant value.
func (c *StepCache) GetConstant() *mlx.Array {
	return c.constant
}

// SetConstant stores a constant value, freeing any previous value.
func (c *StepCache) SetConstant(arr *mlx.Array) {
	if c.constant != nil {
		c.constant.Free()
	}
	c.constant = arr
}

// Arrays returns all non-nil cached arrays (for pool.Keep).
func (c *StepCache) Arrays() []*mlx.Array {
	var result []*mlx.Array
	if c.constant != nil {
		result = append(result, c.constant)
	}
	for _, arr := range c.layers {
		if arr != nil {
			result = append(result, arr)
		}
	}
	for _, arr := range c.layers2 {
		if arr != nil {
			result = append(result, arr)
		}
	}
	return result
}

// Free releases all cached arrays. Call when generation completes.
func (c *StepCache) Free() {
	if c.constant != nil {
		c.constant.Free()
		c.constant = nil
	}
	for i, arr := range c.layers {
		if arr != nil {
			arr.Free()
			c.layers[i] = nil
		}
	}
	for i, arr := range c.layers2 {
		if arr != nil {
			arr.Free()
			c.layers2[i] = nil
		}
	}
}

// NumLayers returns the number of layers this cache can store.
func (c *StepCache) NumLayers() int {
	return len(c.layers)
}
