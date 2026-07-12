package cache

import (
	"fmt"

	"github.com/ollama/ollama/x/mlxrunner/mlx"
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

	// PrepareSnapshots schedules the cache to capture a snapshot as its
	// storage offset reaches each listed offset during subsequent writes.
	// Offsets are storage offsets, must not already be passed (current
	// offset <= offset), and must be sorted ascending and unique. Scheduled
	// offsets persist across multiple writes until TakeSnapshots is called;
	// captures accumulate. The previous schedule must be drained first.
	//
	// Unlike Snapshot, capture happens at the moment a write crosses the
	// scheduled offset, so it can record states interior to a batched write
	// without the caller breaking the write into pieces.
	PrepareSnapshots(offsets []int)

	// TakeSnapshots returns the snapshots captured since PrepareSnapshots,
	// one per scheduled offset in the caller's order, and clears the
	// schedule. An entry is nil when its scheduled offset captured a
	// zero-width range (the offset equalled the previous boundary, e.g. an
	// offset scheduled at the current position): rolling back there needs
	// only a live rewind, so there is nothing to page out.
	TakeSnapshots() []Snapshot

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
	// Size returns the byte size of the paged-out data (in VRAM). A lazy
	// snapshot that still indexes a live cache buffer returns 0 — it owns
	// no extra memory yet. Once materialized (the cache copies the range
	// out before overwriting its slots), Size returns the owned bytes.
	Size() int

	// SetMaterializeHook installs a callback fired once when a lazy
	// snapshot materializes (allocates its owned arrays). delta is the
	// newly-allocated byte count. Pass nil to detach. Snapshots that are
	// never lazy may treat this as a no-op.
	SetMaterializeHook(func(delta int))

	// Close unpins the snapshot's arrays so they can be freed by Sweep.
	Close()
}

// pendingSnapshots holds the per-token snapshot capture state shared by all
// cache kinds. The owning cache calls capture(offset) from its write path
// after each token's storage is in place; capture materializes a snapshot
// for every scheduled offset that the write has now reached.
type pendingSnapshots struct {
	offsets  []int      // scheduled storage offsets, in caller order
	captured []Snapshot // captured[i] corresponds to offsets[i]; nil until reached
	base     int        // running capture cursor: the from-offset of the next capture
}

// prepare schedules the listed storage offsets. See Cache.PrepareSnapshots.
func (p *pendingSnapshots) prepare(currentOffset int, offsets []int) {
	if p.captured != nil {
		panic("PrepareSnapshots: previous schedule not drained; TakeSnapshots first or its captures leak")
	}
	for i, o := range offsets {
		if o < currentOffset {
			panic(fmt.Sprintf("PrepareSnapshots: offset %d already passed (current %d)", o, currentOffset))
		}
		// captureReached and scheduledIn walk offsets in storage order and rely
		// on it being ascending and unique.
		if i > 0 && o <= offsets[i-1] {
			panic(fmt.Sprintf("PrepareSnapshots: offsets must be sorted and unique, got %v", offsets))
		}
	}
	p.offsets = append([]int(nil), offsets...)
	p.captured = make([]Snapshot, len(offsets))
	p.base = currentOffset
}

// take returns the captured snapshots and clears the schedule. See
// Cache.TakeSnapshots. Captures fire in ascending offset order, so by take time
// every scheduled offset the writes crossed has been visited; a nil entry is a
// zero-width capture, not a missed one.
func (p *pendingSnapshots) take() []Snapshot {
	out := p.captured
	p.offsets, p.captured = nil, nil
	return out
}

// captureReached materializes a snapshot for every scheduled offset that equals
// reached and hasn't been captured yet, using snap to produce the rollback
// state. The capture cursor base holds the previous scheduled boundary: snap
// reads it (yielding a [base, reached) range for position-sliceable caches),
// then base advances to reached so the next capture starts there. base only
// advances when a capture actually fires — write boundaries between scheduled
// offsets must not move it, or a later capture would record a fromOffset past
// the previous scheduled offset and break the alignment between each capture's
// range and the trie edge the caller attaches it to.
func (p *pendingSnapshots) captureReached(reached int, snap func(offset int) Snapshot) {
	captured := false
	for i, o := range p.offsets {
		if p.captured[i] == nil && o == reached {
			p.captured[i] = snap(o)
			captured = true
		}
	}
	if captured {
		p.base = reached
	}
}

// scheduledIn returns the scheduled offsets in [start, end], ascending. A write
// spanning [start, end) drives captureReached at each of these so any capture
// due there fires; for position-sliceable caches, the resulting snapshots cover
// [previous-scheduled-offset, this-offset) via captureReached's running cursor.
// p.offsets is ascending and unique (prepare enforces it), so the range filter
// preserves both.
func (p *pendingSnapshots) scheduledIn(start, end int) []int {
	var boundaries []int
	for _, o := range p.offsets {
		if o >= start && o <= end {
			boundaries = append(boundaries, o)
		}
	}
	return boundaries
}
