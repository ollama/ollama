//go:build mlx

package mlxrunner

import (
	"fmt"
	"log/slog"

	"github.com/ollama/ollama/logutil"
	"github.com/ollama/ollama/x/mlxrunner/cache"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/mlxrunner/model/base"
)

type kvCache struct {
	// For now we only support a single entry, so this is just one sequence
	tokens []int32
	caches []cache.Cache
}

// cacheSession manages caches for a single pipeline run.
// Callers should append generated tokens to outputs and
// defer close to save the cache state.
type cacheSession struct {
	cache   *kvCache
	inputs  []int32
	outputs []int32

	caches    []cache.Cache
	remaining []int32
}

func (c *kvCache) free() {
	for i, kv := range c.caches {
		if kv == nil {
			continue
		}
		kv.Free()
		c.caches[i] = nil
	}
	c.caches = nil
	c.tokens = nil
}

func (c *kvCache) cachesCanTrim() bool {
	for _, kv := range c.caches {
		if kv == nil {
			continue
		}
		if !kv.CanTrim() {
			return false
		}
	}
	return true
}

func (c *kvCache) trimToPrefix(prefix int) {
	for _, kv := range c.caches {
		if kv == nil || !kv.CanTrim() {
			continue
		}
		if trim := kv.Offset() - prefix; trim > 0 {
			kv.Trim(trim)
		}
	}
	if prefix < len(c.tokens) {
		c.tokens = c.tokens[:prefix]
	}
}

// begin prepares caches for a new request. It finds the nearest
// matching cache or creates new caches if none match.
func (c *kvCache) begin(m base.Model, inputs []int32) *cacheSession {
	if len(c.caches) == 0 {
		if cacheFactory, ok := m.(interface{ NewCaches() []cache.Cache }); ok {
			c.caches = cacheFactory.NewCaches()
		} else {
			c.caches = make([]cache.Cache, m.NumLayers())
			for i := range c.caches {
				c.caches[i] = cache.NewKVCache()
			}
		}
	}

	remaining := c.findRemaining(inputs)

	return &cacheSession{
		cache:     c,
		inputs:    inputs,
		caches:    c.caches,
		remaining: remaining,
	}
}

// close saves the token state if the forward pass ran.
func (s *cacheSession) close() {
	if len(s.caches) == 0 {
		return
	}

	offset := -1
	arrays := make([]*mlx.Array, 0, 2*len(s.caches))
	for _, kv := range s.caches {
		if kv == nil {
			continue
		}
		if off := kv.Offset(); offset < 0 || off < offset {
			offset = off
		}
		arrays = append(arrays, kv.Materialize()...)
	}
	if offset <= 0 {
		return
	}

	// Ensure that if we have run the forward pass and set the metadata
	// that we also actually have the data.
	mlx.AsyncEval(arrays...)

	stored := append(s.inputs, s.outputs...)
	if offset > len(stored) {
		offset = len(stored)
	}
	s.cache.tokens = stored[:offset]
}

// findRemaining finds the longest common prefix between tokens and the cached
// sequence, trims stale cache entries, and returns the remaining tokens.
func (c *kvCache) findRemaining(tokens []int32) []int32 {
	prefix := 0
	for prefix < len(tokens) && prefix < len(c.tokens) && tokens[prefix] == c.tokens[prefix] {
		prefix++
	}

	// Always keep at least one token to re-evaluate so the
	// pipeline can seed token generation from it.
	if prefix == len(tokens) && prefix > 0 {
		prefix--
	}

	if prefix < len(c.tokens) {
		if c.cachesCanTrim() {
			c.trimToPrefix(prefix)
		} else {
			c.free()
			slog.Info("Cache miss", "left", len(tokens), "matched", prefix, "reason", "non_trimmable_divergence")
			return tokens
		}
	}

	if prefix == 0 {
		slog.Info("Cache miss", "left", len(tokens))
	} else {
		slog.Info("Cache hit", "total", len(tokens), "cached", prefix, "left", len(tokens[prefix:]))
	}
	return tokens[prefix:]
}

func (c *kvCache) log() {
	if len(c.caches) == 0 {
		return
	}
	offset := -1
	var totalBytes int
	for _, kv := range c.caches {
		if kv == nil {
			continue
		}
		if off := kv.Offset(); offset < 0 || off < offset {
			offset = off
		}
		for _, a := range kv.Materialize() {
			totalBytes += a.NumBytes()
		}
	}
	if offset < 0 {
		return
	}
	logutil.Trace(fmt.Sprintf("kv cache tokens: %d, size: %s", offset, mlx.PrettyBytes(totalBytes)))
}
