//go:build mlx

package mlxrunner

import (
	"fmt"
	"log/slog"

	"github.com/ollama/ollama/logutil"
	"github.com/ollama/ollama/x/mlxrunner/cache"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
)

// CacheEntry stores a single sequence
type CacheEntry struct {
	Tokens []int32
	Caches []cache.Cache
}

// FindNearestCache finds the longest common prefix between tokens and the cached sequence
func (r *Runner) FindNearestCache(tokens []int32) ([]cache.Cache, []int32) {
	if r.cache == nil {
		slog.Info("Cache miss", "left", len(tokens))
		return nil, tokens
	}

	// Find longest common prefix
	prefix := 0
	for prefix < len(tokens) && prefix < len(r.cache.Tokens) && tokens[prefix] == r.cache.Tokens[prefix] {
		prefix++
	}

	switch {
	case prefix == 0:
		for _, c := range r.cache.Caches {
			c.Free()
		}
		r.cache = nil
		slog.Info("Cache miss", "left", len(tokens))
		return nil, tokens
	case prefix < len(r.cache.Tokens):
		trim := len(r.cache.Tokens) - prefix
		for _, c := range r.cache.Caches {
			c.Trim(trim)
		}
		r.cache.Tokens = r.cache.Tokens[:prefix]
	}

	slog.Info("Cache hit", "total", len(tokens), "cached", prefix, "left", len(tokens[prefix:]))
	return r.cache.Caches, tokens[prefix:]
}

func (r *Runner) InsertCache(tokens []int32, caches []cache.Cache) {
	r.cache = &CacheEntry{
		Tokens: tokens,
		Caches: caches,
	}
}

func (c *CacheEntry) LogCache() {
	var totalBytes int
	for _, kv := range c.Caches {
		k, v := kv.State()
		totalBytes += k.NumBytes() + v.NumBytes()
	}
	logutil.Trace(fmt.Sprintf("kv cache tokens: %d, size: %s", c.Caches[0].Offset(), mlx.PrettyBytes(totalBytes)))
}
