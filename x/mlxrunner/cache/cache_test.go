//go:build mlx

package cache

import "testing"

func TestKVCacheGrowDebugEnabled(t *testing.T) {
	t.Setenv("OLLAMA_MLX_DEBUG_CACHE_GROW", "")
	if kvCacheGrowDebugEnabled() {
		t.Fatal("kvCacheGrowDebugEnabled() = true, want false")
	}

	t.Setenv("OLLAMA_MLX_DEBUG_CACHE_GROW", "1")
	if !kvCacheGrowDebugEnabled() {
		t.Fatal("kvCacheGrowDebugEnabled() = false, want true")
	}
}
