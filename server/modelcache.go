package server

import (
	"os"
	"sync"
	"time"
)

// modelCacheEntry stores a GetModel result alongside the manifest file's
// modification time captured at population. On lookup, a single os.Stat
// syscall (~1μs) validates freshness without re-reading blobs, JSON
// decoding, SHA256 hashing, or template parsing (~60–200μs).
type modelCacheEntry struct {
	model           *Model
	manifestPath    string
	manifestModTime time.Time
}

// modelCache provides concurrency-safe caching for GetModel results.
//
// Staleness detection uses manifest file ModTime: if the manifest file is
// unchanged since the entry was populated, all blob references within it
// are still valid and the cached Model can be reused. This eliminates the
// need for explicit invalidation in mutation handlers (create, delete,
// pull) because those operations modify the manifest file, which
// automatically changes its ModTime and causes the next lookup to miss.
//
// sync.RWMutex is chosen over sync.Map because the access pattern is
// read-heavy (many concurrent inference requests) with rare writes (model
// mutations), and RWMutex allows bounded, predictable memory usage with
// straightforward iteration for future eviction policies.
type modelCache struct {
	mu      sync.RWMutex
	entries map[string]*modelCacheEntry
}

var globalModelCache = &modelCache{
	entries: make(map[string]*modelCacheEntry),
}

// get returns a cached Model if the manifest file has not been modified
// since the entry was stored. Returns (nil, false) on cache miss or stale
// entry. On hit, returns a shallow copy of the Model struct so callers can
// safely mutate value-type fields (e.g. Config.Parser) without corrupting
// the cached original.
func (c *modelCache) get(name string) (*Model, bool) {
	c.mu.RLock()
	entry, ok := c.entries[name]
	c.mu.RUnlock()
	if !ok {
		return nil, false
	}

	// Validate freshness: one stat syscall (~1μs) vs full reload (~60–200μs).
	fi, err := os.Stat(entry.manifestPath)
	if err != nil || !fi.ModTime().Equal(entry.manifestModTime) {
		c.mu.Lock()
		delete(c.entries, name)
		c.mu.Unlock()
		return nil, false
	}

	// Shallow copy prevents caller mutations (e.g. m.Config.Parser = "harmony")
	// from corrupting the cached entry. ConfigV2 is a value type so it is fully
	// copied. Slice and map fields (AdapterPaths, Options, Messages, etc.) share
	// underlying storage but are never mutated in place by callers.
	copied := *entry.model
	return &copied, true
}

// put stores a Model in the cache keyed by name, recording the manifest
// file path and its current modification time for future staleness checks.
func (c *modelCache) put(name string, m *Model, manifestPath string, modTime time.Time) {
	c.mu.Lock()
	c.entries[name] = &modelCacheEntry{
		model:           m,
		manifestPath:    manifestPath,
		manifestModTime: modTime,
	}
	c.mu.Unlock()
}
