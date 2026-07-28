package base

import (
	"testing"

	"github.com/ollama/ollama/x/mlxrunner/mlx"
)

func clearMediaFeatureCache(cache *MediaFeatureCache) {
	cache.mu.Lock()
	defer cache.mu.Unlock()

	for cache.lru.Len() > 0 {
		cache.removeLocked(cache.lru.Back().Value.(*mediaFeatureEntry))
	}
}

func TestMediaFeatureKey(t *testing.T) {
	key := NewMediaFeatureKey([]byte("image"), "tokens=280")
	if key != NewMediaFeatureKey([]byte("image"), "tokens=280") {
		t.Fatal("same input produced different keys")
	}
	if key == NewMediaFeatureKey([]byte("different"), "tokens=280") {
		t.Fatal("different media produced the same key")
	}
	if key == NewMediaFeatureKey([]byte("image"), "tokens=560") {
		t.Fatal("different preprocessing variants produced the same key")
	}
}

func TestMediaFeatureCacheLeaseLifetime(t *testing.T) {
	cache := newMediaFeatureCache(1, 4)
	features := mlx.FromValues([]float32{1}, 1, 1)
	lease := cache.Store(NewMediaFeatureKey([]byte("a"), ""), features, 1)

	clearMediaFeatureCache(cache)
	mlx.Sweep()
	if !features.Valid() {
		t.Fatal("features were freed while leased")
	}

	lease.Release()
	lease.Release()
	mlx.Sweep()
	if features.Valid() {
		t.Fatal("evicted features remained valid after their lease was released")
	}
}

func TestMediaFeatureCacheLRU(t *testing.T) {
	cache := newMediaFeatureCache(2, 1<<20)
	keys := []MediaFeatureKey{
		NewMediaFeatureKey([]byte("a"), ""),
		NewMediaFeatureKey([]byte("b"), ""),
		NewMediaFeatureKey([]byte("c"), ""),
	}

	for i := range 2 {
		lease := cache.Store(keys[i], mlx.FromValues([]float32{float32(i)}, 1, 1), 1)
		lease.Release()
	}
	recent, ok := cache.Acquire(keys[0])
	if !ok {
		t.Fatal("failed to acquire resident entry")
	}
	recent.Release()

	inserted := cache.Store(keys[2], mlx.FromValues([]float32{2}, 1, 1), 1)
	inserted.Release()

	if lease, ok := cache.Acquire(keys[1]); ok {
		lease.Release()
		t.Fatal("least recently used entry was not evicted")
	}
	for _, key := range []MediaFeatureKey{keys[0], keys[2]} {
		lease, ok := cache.Acquire(key)
		if !ok {
			t.Fatal("recent entry was evicted")
		}
		lease.Release()
	}

	clearMediaFeatureCache(cache)
	mlx.Sweep()
}

func TestMediaFeatureCacheByteLimit(t *testing.T) {
	cache := newMediaFeatureCache(4, 4)
	first := mlx.FromValues([]float32{1}, 1, 1)
	firstLease := cache.Store(NewMediaFeatureKey([]byte("a"), ""), first, 1)
	firstLease.Release()

	second := mlx.FromValues([]float32{2}, 1, 1)
	secondLease := cache.Store(NewMediaFeatureKey([]byte("b"), ""), second, 1)
	secondLease.Release()
	mlx.Sweep()

	if first.Valid() {
		t.Fatal("oldest features were not evicted at the byte limit")
	}
	if !second.Valid() {
		t.Fatal("newest features were evicted")
	}

	clearMediaFeatureCache(cache)
	mlx.Sweep()
}

func TestMediaFeatureCacheMultipleLeases(t *testing.T) {
	cache := newMediaFeatureCache(1, 4)
	key := NewMediaFeatureKey([]byte("a"), "")
	features := mlx.FromValues([]float32{1}, 1, 1)
	first := cache.Store(key, features, 1)
	second, ok := cache.Acquire(key)
	if !ok {
		t.Fatal("failed to acquire second lease")
	}

	clearMediaFeatureCache(cache)
	first.Release()
	mlx.Sweep()
	if !features.Valid() {
		t.Fatal("features were freed before the final lease was released")
	}

	second.Release()
	mlx.Sweep()
	if features.Valid() {
		t.Fatal("features remained valid after the final lease was released")
	}
}

func TestMediaFeatureCacheStoreExisting(t *testing.T) {
	cache := newMediaFeatureCache(1, 1<<20)
	key := NewMediaFeatureKey([]byte("a"), "")
	firstFeatures := mlx.FromValues([]float32{1}, 1, 1)
	first := cache.Store(key, firstFeatures, 1)

	secondFeatures := mlx.FromValues([]float32{2}, 1, 1)
	second := cache.Store(key, secondFeatures, 1)
	if second.Features() != firstFeatures {
		t.Fatal("second store did not reuse resident features")
	}

	clearMediaFeatureCache(cache)
	first.Release()
	mlx.Sweep()
	if !firstFeatures.Valid() {
		t.Fatal("features were freed while another lease was active")
	}

	second.Release()
	mlx.Sweep()
	if firstFeatures.Valid() {
		t.Fatal("resident features remained valid after the final lease was released")
	}
	if secondFeatures.Valid() {
		t.Fatal("unused duplicate features remained valid")
	}
}
