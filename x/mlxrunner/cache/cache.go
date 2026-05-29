package cache

import (
	"github.com/ollama/ollama/x/mlxrunner/batch"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/models/nn"
)

// Cache is common state management shared by every cache kind. Writers
// live on the specific caches
type Cache interface {
	// State returns the cache-owned state roots that should be kept/evaluated.
	State() []*mlx.Array
	Free()
	Offset() int

	// Snapshot copies cache state from fromOffset to current offset into
	// pinned VRAM arrays. The active cache is unchanged.
	Snapshot(fromOffset int) Snapshot

	// Restore brings the cache to target. If snapshot is nil, rewinds
	// using the cache's own live state. Returns false if the target is
	// unreachable (e.g. target > current offset, or negative).
	Restore(snapshot Snapshot, target int) bool

	// Merge combines two sequential snapshots [a,b) and [b,c) into [a,c).
	// Takes ownership of both inputs.
	Merge(parent, child Snapshot) Snapshot

	// Split divides a snapshot [a,c) at offset b into [a,b) and [b,c).
	// Takes ownership of the input. Cache types that cannot split
	// (e.g. recurrent) return (nil, snapshot).
	Split(snapshot Snapshot, at int) (parent, child Snapshot)
}

// Snapshot is paged-out cache state that can be restored later.
type Snapshot interface {
	// Size returns the byte size of the paged-out data (in VRAM).
	Size() int
	// Close unpins the snapshot's arrays so they can be freed by Sweep.
	Close()
}

// Viewer exposes a read-only attention history for a cache.
type Viewer interface {
	View(b *batch.Batch) *nn.KVHistory
}

type speculativeCommitter interface {
	Cache
	commit(n int)
}

// Speculation is an isolated cache transaction for speculative target
// validation. Updates record generated K/V without mutating the live caches;
// Commit appends only the accepted prefix to the live caches.
type Speculation struct {
	layers []speculativeCommitter
}

// BeginSpeculation returns cache wrappers suitable for a speculative target
// forward. The returned caches must only be used for that forward.
func BeginSpeculation(caches []Cache) ([]Cache, *Speculation, bool) {
	specCaches := make([]Cache, len(caches))
	layers := make([]speculativeCommitter, len(caches))

	for i, c := range caches {
		switch c := c.(type) {
		case nil:
		case *RotatingKVCache:
			sc := newSpeculativeRotatingKVCache(c)
			specCaches[i] = sc
			layers[i] = sc
		case *KVCache:
			sc := newSpeculativeKVCache(c)
			specCaches[i] = sc
			layers[i] = sc
		default:
			return nil, nil, false
		}
	}

	return specCaches, &Speculation{layers: layers}, true
}

// BeginIsolatedSpeculation returns cache wrappers that never mutate live cache
// state. It is intended for correctness instrumentation, not the hot path.
func BeginIsolatedSpeculation(caches []Cache) ([]Cache, bool) {
	specCaches := make([]Cache, len(caches))

	for i, c := range caches {
		switch c := c.(type) {
		case nil:
		case *RotatingKVCache:
			specCaches[i] = newSpeculativeRotatingKVCache(c)
		case *KVCache:
			specCaches[i] = newIsolatedKVCache(c)
		default:
			return nil, false
		}
	}

	return specCaches, true
}

// Commit appends the accepted prefix from the speculative forward to the live
// caches. The target bonus token is intentionally not committed.
func (s *Speculation) Commit(n int) {
	if s == nil {
		return
	}
	for _, layer := range s.layers {
		if layer != nil {
			layer.commit(n)
		}
	}
}

func concatKV(prev, next *mlx.Array) *mlx.Array {
	if prev == nil {
		return next
	}
	return prev.Concatenate(2, next)
}

func prefixKV(a *mlx.Array, n int) *mlx.Array {
	return a.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(0, n), mlx.Slice())
}
