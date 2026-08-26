package qwen4_exp

import (
	"fmt"

	"github.com/ollama/ollama/x/mlxrunner/batch"
	"github.com/ollama/ollama/x/mlxrunner/cache"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
)

// engramCache keeps the two raw token IDs and nine normalized
// convolution inputs preceding the current chunk. Both Engram operations span
// forward boundaries, so this state follows the model cache lifecycle.
type engramCache struct {
	history     *mlx.Array
	convHistory *mlx.Array
	offset      int
	eosID       int64
	width       int
	convTail    int
	convDim     int

	scheduled []int
	captured  []cache.Snapshot
}

type engramSnapshot struct {
	history     *mlx.Array
	convHistory *mlx.Array
	offset      int
}

func newEngramCache(width, convTail, convDim int, eosID int64) *engramCache {
	return &engramCache{width: width, convTail: convTail, convDim: convDim, eosID: eosID}
}

func (c *engramCache) setHistory(value *mlx.Array) {
	value = value.Clone()
	mlx.Pin(value)
	mlx.Unpin(c.history)
	c.history = value
}

func (c *engramCache) setConvHistory(value *mlx.Array) {
	value = value.Clone()
	mlx.Pin(value)
	mlx.Unpin(c.convHistory)
	c.convHistory = value
}

func (c *engramCache) get(b *batch.Batch) *mlx.Array {
	if c.history == nil {
		batchSize := b.InputIDs.Dim(0)
		values := make([]int64, batchSize*c.width)
		for i := range values {
			values[i] = c.eosID
		}
		c.setHistory(mlx.FromValues(values, batchSize, c.width))
	}
	return c.history
}

func (c *engramCache) getConvHistory(b *batch.Batch, dtype mlx.DType) *mlx.Array {
	if c.convHistory == nil {
		c.setConvHistory(mlx.Zeros(dtype, b.InputIDs.Dim(0), c.convTail, c.convDim))
	}
	return c.convHistory
}

func tailAfter(history, values *mlx.Array, count, width int) *mlx.Array {
	if count > 0 && count < values.Dim(1) {
		start := make([]int32, values.NumDims())
		stop := make([]int32, values.NumDims())
		for i, dim := range values.Dims() {
			stop[i] = int32(dim)
		}
		stop[1] = int32(count)
		values = mlx.SliceStartStop(values, start, stop)
	}
	joined := history
	if count > 0 {
		joined = mlx.Concatenate([]*mlx.Array{history, values}, 1)
	}
	dims := joined.Dims()
	start := make([]int32, len(dims))
	stop := make([]int32, len(dims))
	for i, dim := range dims {
		stop[i] = int32(dim)
	}
	start[1] = int32(joined.Dim(1) - width)
	return mlx.SliceStartStop(joined, start, stop)
}

func (c *engramCache) put(b *batch.Batch, inputIDs, convInput *mlx.Array) {
	if b.InputIDs.Dim(0) != 1 || len(b.SeqQueryLens) != 1 {
		panic("Engram cache requires a single sequence")
	}
	history := c.get(b)
	convHistory := c.getConvHistory(b, convInput.DType())
	length := int(b.SeqQueryLens[0])
	start, end := c.offset, c.offset+length
	for i, boundary := range c.scheduled {
		if c.captured[i] != nil || boundary < start || boundary > end {
			continue
		}
		c.captured[i] = newEngramSnapshot(
			tailAfter(history, inputIDs, boundary-start, c.width),
			tailAfter(convHistory, convInput, boundary-start, c.convTail),
			boundary,
		)
	}
	c.setHistory(tailAfter(history, inputIDs, length, c.width))
	c.setConvHistory(tailAfter(convHistory, convInput, length, c.convTail))
	c.offset = end
}

func (c *engramCache) State() []*mlx.Array {
	state := []*mlx.Array{c.history, c.convHistory}
	for _, snapshot := range c.captured {
		if snapshot != nil {
			value := snapshot.(*engramSnapshot)
			state = append(state, value.history, value.convHistory)
		}
	}
	return state
}

func (c *engramCache) Free() {
	mlx.Unpin(c.history, c.convHistory)
	c.history = nil
	c.convHistory = nil
	c.offset = 0
	c.scheduled = nil
	c.captured = nil
}

func (c *engramCache) Offset() int { return c.offset }

func (c *engramCache) PrepareSnapshots(offsets []int) {
	if c.captured != nil {
		panic("Engram cache: previous snapshot schedule was not drained")
	}
	for i, offset := range offsets {
		if offset < c.offset || (i > 0 && offset <= offsets[i-1]) {
			panic(fmt.Sprintf("Engram cache: invalid snapshot offsets %v at current offset %d", offsets, c.offset))
		}
	}
	c.scheduled = append([]int(nil), offsets...)
	c.captured = make([]cache.Snapshot, len(offsets))
	for i, offset := range offsets {
		if offset == c.offset && c.history != nil && c.convHistory != nil {
			c.captured[i] = newEngramSnapshot(c.history, c.convHistory, c.offset)
		}
	}
}

func (c *engramCache) TakeSnapshots() []cache.Snapshot {
	result := c.captured
	c.scheduled = nil
	c.captured = nil
	return result
}

func (c *engramCache) Snapshot(int) cache.Snapshot {
	if c.history == nil || c.convHistory == nil {
		return nil
	}
	return newEngramSnapshot(c.history, c.convHistory, c.offset)
}

func (c *engramCache) Restore(snapshot cache.Snapshot, target int) bool {
	if snapshot == nil {
		return target == c.offset
	}
	value, ok := snapshot.(*engramSnapshot)
	if !ok || value.offset != target {
		return false
	}
	c.setHistory(value.history)
	c.setConvHistory(value.convHistory)
	c.offset = target
	return true
}

func (c *engramCache) Merge(parent, child cache.Snapshot) cache.Snapshot {
	if parent != nil {
		parent.Close()
	}
	return child
}

func (c *engramCache) Split(snapshot cache.Snapshot, _ int) (cache.Snapshot, cache.Snapshot) {
	return nil, snapshot
}

func newEngramSnapshot(history, convHistory *mlx.Array, offset int) *engramSnapshot {
	snapshot := &engramSnapshot{history: history.Clone(), convHistory: convHistory.Clone(), offset: offset}
	mlx.Pin(snapshot.history, snapshot.convHistory)
	mlx.AsyncEval(snapshot.history, snapshot.convHistory)
	return snapshot
}

func (s *engramSnapshot) Size() int                    { return s.history.NumBytes() + s.convHistory.NumBytes() }
func (s *engramSnapshot) SetMaterializeHook(func(int)) {}
func (s *engramSnapshot) Close()                       { mlx.Unpin(s.history, s.convHistory) }
