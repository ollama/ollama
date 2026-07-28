package base

import (
	"container/list"
	"crypto/sha256"
	"encoding/binary"
	"sync"
	"sync/atomic"

	"github.com/ollama/ollama/x/mlxrunner/mlx"
)

const (
	// Keep only a few recent attachments and cap their projected features so
	// this stopgap cannot consume a material share of the model's memory.
	mediaFeatureCacheEntries = 4
	mediaFeatureCacheBytes   = 128 << 20
)

// MediaFeatureKey identifies projected media features within one loaded model.
// variant must describe preprocessing choices that can change the features.
type MediaFeatureKey [sha256.Size]byte

// NewMediaFeatureKey hashes media bytes and their preprocessing variant.
func NewMediaFeatureKey(data []byte, variant string) MediaFeatureKey {
	hash := sha256.New()
	var size [8]byte
	binary.LittleEndian.PutUint64(size[:], uint64(len(variant)))
	_, _ = hash.Write(size[:])
	_, _ = hash.Write([]byte(variant))
	_, _ = hash.Write(data)

	var key MediaFeatureKey
	copy(key[:], hash.Sum(nil))
	return key
}

// MediaFeatureCache is a bounded stopgap that retains projected media features
// while multimodal requests use a request-local KV cache. It is intentionally
// independent of the runner caches and should be removed or absorbed once
// multimodal inputs participate in their normal lifecycle.
type MediaFeatureCache struct {
	// mu guards the cache accounting, LRU, entries map, and every mutable
	// mediaFeatureEntry field.
	mu sync.Mutex

	maxEntries int
	maxBytes   int
	usedBytes  int
	entries    map[MediaFeatureKey]*mediaFeatureEntry
	lru        list.List
}

type mediaFeatureEntry struct {
	key        MediaFeatureKey
	features   *mlx.Array
	tokenCount int
	size       int
	leases     int
	element    *list.Element
	resident   bool
}

// MediaFeatureLease keeps cached features alive while a prepared prompt uses
// them. Release must be called when the prompt is closed.
type MediaFeatureLease struct {
	cache    *MediaFeatureCache
	entry    *mediaFeatureEntry
	released atomic.Bool
}

// NewMediaFeatureCache creates a cache with fixed entry and memory bounds.
func NewMediaFeatureCache() *MediaFeatureCache {
	return newMediaFeatureCache(mediaFeatureCacheEntries, mediaFeatureCacheBytes)
}

func newMediaFeatureCache(maxEntries, maxBytes int) *MediaFeatureCache {
	if maxEntries <= 0 {
		panic("media feature cache requires at least one entry")
	}
	if maxBytes <= 0 {
		panic("media feature cache requires a positive byte limit")
	}
	return &MediaFeatureCache{
		maxEntries: maxEntries,
		maxBytes:   maxBytes,
		entries:    make(map[MediaFeatureKey]*mediaFeatureEntry),
	}
}

// Acquire returns a lease for cached features and updates their recency.
func (c *MediaFeatureCache) Acquire(key MediaFeatureKey) (*MediaFeatureLease, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()

	entry, ok := c.entries[key]
	if !ok {
		return nil, false
	}
	entry.leases++
	c.lru.MoveToFront(entry.element)
	return &MediaFeatureLease{cache: c, entry: entry}, true
}

// Store retains features and returns a lease for the caller. If another caller
// stored the key first, Store returns the existing features instead.
func (c *MediaFeatureCache) Store(key MediaFeatureKey, features *mlx.Array, tokenCount int) *MediaFeatureLease {
	// The three panics below are here only because this stopgap cache has
	// no error channel back to the pipeline; when multimodal inputs join the
	// normal cache lifecycle these become returned errors.
	if features == nil || !features.Valid() {
		panic("cannot cache invalid media features")
	}
	if tokenCount <= 0 {
		panic("cannot cache media features without tokens")
	}

	c.mu.Lock()
	defer c.mu.Unlock()

	if entry, ok := c.entries[key]; ok {
		if entry.tokenCount != tokenCount {
			panic("media feature cache token count changed for the same key")
		}
		entry.leases++
		c.lru.MoveToFront(entry.element)
		return &MediaFeatureLease{cache: c, entry: entry}
	}

	mlx.Pin(features)
	entry := &mediaFeatureEntry{
		key:        key,
		features:   features,
		tokenCount: tokenCount,
		size:       features.NumBytes(),
		leases:     1,
		resident:   true,
	}
	entry.element = c.lru.PushFront(entry)
	c.entries[key] = entry
	c.usedBytes += entry.size
	c.evictLocked()

	return &MediaFeatureLease{cache: c, entry: entry}
}

// Features returns the projected features while the lease is active.
func (l *MediaFeatureLease) Features() *mlx.Array {
	if l == nil || l.released.Load() {
		return nil
	}
	return l.entry.features
}

// TokenCount returns the number of soft tokens represented by the features.
func (l *MediaFeatureLease) TokenCount() int {
	if l == nil || l.released.Load() {
		return 0
	}
	return l.entry.tokenCount
}

// Release relinquishes a lease. It is safe to call more than once.
func (l *MediaFeatureLease) Release() {
	if l == nil || !l.released.CompareAndSwap(false, true) {
		return
	}

	l.cache.mu.Lock()
	defer l.cache.mu.Unlock()

	l.entry.leases--
	if l.entry.leases < 0 {
		// Request-goroutine panic, kept only for the stopgap's shape; see Store.
		panic("media feature cache lease count became negative")
	}
	if !l.entry.resident && l.entry.leases == 0 {
		mlx.Unpin(l.entry.features)
	}
}

func (c *MediaFeatureCache) evictLocked() {
	for len(c.entries) > c.maxEntries || c.usedBytes > c.maxBytes {
		c.removeLocked(c.lru.Back().Value.(*mediaFeatureEntry))
	}
}

func (c *MediaFeatureCache) removeLocked(entry *mediaFeatureEntry) {
	delete(c.entries, entry.key)
	c.lru.Remove(entry.element)
	c.usedBytes -= entry.size
	entry.element = nil
	entry.resident = false
	if entry.leases == 0 {
		mlx.Unpin(entry.features)
	}
}
