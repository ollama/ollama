package cache

import "github.com/ollama/ollama/mlx"

// HiddenCache retains backbone outputs for heads that read earlier tokens.
// It follows the same prefix snapshots as the attention and recurrent caches.
// Arrays are immutable: appending or restoring builds a new view or concatenation,
// so a snapshot never observes a later request's writes.
type HiddenCache struct {
	hidden       *mlx.Array
	scope        *mlx.Scope
	backingBytes int
	snapshots    pendingSnapshots
}

func NewHiddenCache() *HiddenCache { return &HiddenCache{scope: mlx.NewScope()} }

func (c *HiddenCache) Append(hidden *mlx.Array) {
	start := c.Offset()
	if c.hidden != nil {
		hidden = mlx.Concatenate([]*mlx.Array{c.hidden, hidden}, 1)
	}
	c.replace(hidden, hidden.NumBytes())
	for _, offset := range c.snapshots.scheduledIn(start, c.Offset()) {
		c.snapshots.captureReached(offset, func(int) Snapshot {
			return c.snapshot(c.snapshots.base, offset)
		})
	}
}

func (c *HiddenCache) replace(hidden *mlx.Array, backingBytes int) {
	if c.hidden == nil {
		c.hidden = hidden
		c.scope.Attach(hidden)
	} else {
		c.hidden.Set(hidden)
	}
	c.backingBytes = backingBytes
}

func (c *HiddenCache) State() []*mlx.Array {
	if c.hidden == nil {
		return nil
	}
	return []*mlx.Array{c.hidden}
}

func (c *HiddenCache) Offset() int {
	if c.hidden == nil {
		return 0
	}
	return c.hidden.Dim(1)
}

func (c *HiddenCache) Free() {
	for _, s := range c.TakeSnapshots() {
		if s != nil {
			s.Close()
		}
	}
	c.scope.Close()
	c.scope = mlx.NewScope()
	c.hidden = nil
	c.backingBytes = 0
}

func (c *HiddenCache) PrepareSnapshots(offsets []int) { c.snapshots.prepare(c.Offset(), offsets) }
func (c *HiddenCache) TakeSnapshots() []Snapshot      { return c.snapshots.take() }

type hiddenSnapshot struct {
	hidden       *mlx.Array
	scope        *mlx.Scope
	start        int
	backingBytes int
}

// Charge the whole backing array even for a view. This may overcount shared
// storage, but eviction must never undercount a small slice retaining a full row.
func (s *hiddenSnapshot) Size() int                    { return s.backingBytes }
func (s *hiddenSnapshot) Close()                       { s.scope.Close() }
func (s *hiddenSnapshot) SetMaterializeHook(func(int)) {}

func newHiddenSnapshot(hidden *mlx.Array, start, backingBytes int) *hiddenSnapshot {
	s := &hiddenSnapshot{hidden: hidden, scope: mlx.NewScope(), start: start, backingBytes: backingBytes}
	s.scope.Attach(hidden)
	return s
}

func (c *HiddenCache) snapshot(start, end int) Snapshot {
	if start == end {
		return nil
	}
	return newHiddenSnapshot(c.hidden.Slice(mlx.Slice(), mlx.Slice(start, end), mlx.Slice()), start, c.backingBytes)
}

func (c *HiddenCache) Snapshot(start int) Snapshot { return c.snapshot(start, c.Offset()) }

func (c *HiddenCache) Restore(snapshot Snapshot, target int) bool {
	if target < 0 {
		return false
	}
	if target == 0 {
		c.Free()
		return true
	}
	if target <= c.Offset() {
		c.replace(c.hidden.Slice(mlx.Slice(), mlx.Slice(0, target), mlx.Slice()), c.backingBytes)
		return true
	}
	if snapshot == nil {
		return false
	}
	s := snapshot.(*hiddenSnapshot)
	if s.start > c.Offset() || target > s.start+s.hidden.Dim(1) {
		return false
	}
	part := s.hidden.Slice(mlx.Slice(), mlx.Slice(c.Offset()-s.start, target-s.start), mlx.Slice())
	if c.hidden == nil {
		c.replace(part, s.backingBytes)
	} else {
		c.Append(part)
	}
	return true
}

func (c *HiddenCache) Split(snapshot Snapshot, at int) (Snapshot, Snapshot) {
	if snapshot == nil {
		return nil, nil
	}
	s := snapshot.(*hiddenSnapshot)
	n := at - s.start
	if n <= 0 {
		return nil, s
	}
	if n >= s.hidden.Dim(1) {
		return s, nil
	}
	left := newHiddenSnapshot(s.hidden.Slice(mlx.Slice(), mlx.Slice(0, n), mlx.Slice()), s.start, s.backingBytes)
	right := newHiddenSnapshot(s.hidden.Slice(mlx.Slice(), mlx.Slice(n, mlx.End), mlx.Slice()), at, s.backingBytes)
	s.Close()
	return left, right
}

func (c *HiddenCache) Merge(parent, child Snapshot) Snapshot {
	if parent == nil || child == nil {
		if parent != nil {
			parent.Close()
		}
		if child != nil {
			child.Close()
		}
		return nil
	}
	a, b := parent.(*hiddenSnapshot), child.(*hiddenSnapshot)
	joined := mlx.Concatenate([]*mlx.Array{a.hidden, b.hidden}, 1)
	s := newHiddenSnapshot(joined, a.start, joined.NumBytes())
	mlx.AsyncEval(joined)
	a.Close()
	b.Close()
	return s
}
