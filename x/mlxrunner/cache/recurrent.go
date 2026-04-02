package cache

import (
	"fmt"

	"github.com/ollama/ollama/x/mlxrunner/batch"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
)

// RecurrentCache stores state for linear-recurrent layers using pool tensors.
//
// convState: [poolSize, convTail, convDim]
// deltaState: [poolSize, numVHeads, headVDim, headKDim]
//
// Row i in the pool belongs to seqOrder[i]. seqOffsets[i] tracks
// how many tokens sequence i has processed.
type RecurrentCache struct {
	convState  *mlx.Array
	deltaState *mlx.Array

	seqOffsets []int
	seqOrder   []int

	convTail  int
	convDim   int
	numVHeads int
	headVDim  int
	headKDim  int
}

func NewRecurrentCache(convTail, convDim, numVHeads, headVDim, headKDim int32) *RecurrentCache {
	return &RecurrentCache{
		convTail:  int(convTail),
		convDim:   int(convDim),
		numVHeads: int(numVHeads),
		headVDim:  int(headVDim),
		headKDim:  int(headKDim),
	}
}

func (c *RecurrentCache) setState(old, v *mlx.Array) *mlx.Array {
	if v == nil || !v.Valid() {
		return old
	}
	v = v.Clone()
	mlx.Pin(v)
	mlx.Unpin(old)
	return v
}

// seqIndex returns the pool row index for a seqID, or -1 if not found.
func (c *RecurrentCache) seqIndex(seqID int) int {
	for i, id := range c.seqOrder {
		if id == seqID {
			return i
		}
	}
	return -1
}

// ensure grows pool tensors if needed to match poolSize, preserving existing rows.
func (c *RecurrentCache) ensure(poolSize int, dtype mlx.DType) {
	if poolSize <= 0 {
		return
	}
	if c.convState != nil && c.convState.Valid() && c.convState.DType() == dtype && c.convState.Dim(0) == poolSize {
		return
	}

	grow := poolSize
	if c.convState != nil && c.convState.Valid() {
		if c.convState.DType() != dtype {
			// Dtype changed — replace entire pool
			c.convState = c.setState(c.convState, mlx.Zeros(dtype, poolSize, c.convTail, c.convDim))
			c.deltaState = c.setState(c.deltaState, mlx.Zeros(dtype, poolSize, c.numVHeads, c.headVDim, c.headKDim))
			return
		}
		grow = poolSize - c.convState.Dim(0)
	}

	if grow <= 0 {
		return
	}

	newConvRows := mlx.Zeros(dtype, grow, c.convTail, c.convDim)
	newDeltaRows := mlx.Zeros(dtype, grow, c.numVHeads, c.headVDim, c.headKDim)

	if c.convState != nil && c.convState.Valid() {
		c.convState = c.setState(c.convState, c.convState.Concatenate(0, newConvRows))
		c.deltaState = c.setState(c.deltaState, c.deltaState.Concatenate(0, newDeltaRows))
	} else {
		c.convState = c.setState(c.convState, newConvRows)
		c.deltaState = c.setState(c.deltaState, newDeltaRows)
	}
}

// batchExtent returns the smallest [start, end) range of pool rows covering all batch sequences.
func (c *RecurrentCache) batchExtent(b *batch.ForwardBatch) (int, int) {
	minIdx := -1
	maxIdx := 0
	for _, seqID := range b.SeqIDs {
		idx := c.seqIndex(seqID)
		if idx < 0 {
			panic(fmt.Sprintf("RecurrentCache.batchExtent: sequence %d not found in cache", seqID))
		}
		if minIdx < 0 || idx < minIdx {
			minIdx = idx
		}
		if idx+1 > maxIdx {
			maxIdx = idx + 1
		}
	}
	if minIdx < 0 {
		minIdx = 0
	}
	return minIdx, maxIdx
}

// stateHistory builds KVHistory mapping batch positions to pool rows,
// remapped relative to sliceStart.
func (c *RecurrentCache) stateHistory(b *batch.ForwardBatch, sliceStart int) mlx.KVHistory {
	n := len(b.SeqIDs)
	indices := make([]int32, n)
	seqLens := make([]int, n)
	for i, seqID := range b.SeqIDs {
		idx := c.seqIndex(seqID)
		if idx < 0 {
			panic(fmt.Sprintf("RecurrentCache.stateHistory: sequence %d not found in cache", seqID))
		}
		indices[i] = int32(idx - sliceStart)
		seqLens[i] = c.seqOffsets[idx]
	}
	return mlx.KVHistory{
		PageTable: mlx.NewArrayInt32(indices, []int32{int32(n), 1}),
		SeqLens:   seqLens,
	}
}

func (c *RecurrentCache) ConvState(b *batch.ForwardBatch, dtype mlx.DType) (*mlx.Array, mlx.KVHistory) {
	c.ensure(len(c.seqOrder), dtype)
	sliceStart, sliceEnd := c.batchExtent(b)
	return c.convState.Slice(mlx.Slice(sliceStart, sliceEnd), mlx.Slice(), mlx.Slice()),
		c.stateHistory(b, sliceStart)
}

func (c *RecurrentCache) SetConvState(b *batch.ForwardBatch, v *mlx.Array) {
	n := int32(len(b.SeqIDs))
	indices := c.batchIndices(b)
	// Reshape to [N, 1, 1] for broadcasting with [poolSize, convTail, convDim]
	indices = mlx.Reshape(indices, n, 1, 1)
	c.convState.Set(c.convState.PutAlongAxis(indices, v, 0))
}

func (c *RecurrentCache) DeltaState(b *batch.ForwardBatch, dtype mlx.DType) (*mlx.Array, mlx.KVHistory) {
	c.ensure(len(c.seqOrder), dtype)
	sliceStart, sliceEnd := c.batchExtent(b)
	return c.deltaState.Slice(mlx.Slice(sliceStart, sliceEnd), mlx.Slice(), mlx.Slice(), mlx.Slice()),
		c.stateHistory(b, sliceStart)
}

func (c *RecurrentCache) SetDeltaState(b *batch.ForwardBatch, v *mlx.Array) {
	n := int32(len(b.SeqIDs))
	indices := c.batchIndices(b)
	// Reshape to [N, 1, 1, 1] for broadcasting with [poolSize, numVHeads, headVDim, headKDim]
	indices = mlx.Reshape(indices, n, 1, 1, 1)
	c.deltaState.Set(c.deltaState.PutAlongAxis(indices, v, 0))
}

// batchIndices returns an int32 tensor mapping each batch position to its
// pool row index, for use with PutAlongAxis scatter.
func (c *RecurrentCache) batchIndices(b *batch.ForwardBatch) *mlx.Array {
	idx := make([]int32, len(b.SeqIDs))
	for i, seqID := range b.SeqIDs {
		idx[i] = int32(c.seqIndex(seqID))
	}
	return mlx.NewArrayInt32(idx, []int32{int32(len(idx))})
}

func (c *RecurrentCache) Advance(b *batch.ForwardBatch) {
	for i, seqID := range b.SeqIDs {
		idx := c.seqIndex(seqID)
		if idx >= 0 {
			c.seqOffsets[idx] += b.SeqLens[i]
		}
	}
}

func (c *RecurrentCache) Update(_ *batch.ForwardBatch, keys, values *mlx.Array) (*mlx.Array, *mlx.Array, mlx.KVHistory) {
	return keys, values, mlx.KVHistory{}
}

func (c *RecurrentCache) State() []*mlx.Array {
	return []*mlx.Array{c.convState, c.deltaState}
}

// recurrentSnapshot holds paged-out recurrent state for one sequence.
type recurrentSnapshot struct {
	convState, deltaState *mlx.Array
	offset                int
}

func (s *recurrentSnapshot) Size() int {
	n := 0
	if s.convState != nil {
		n += s.convState.NumBytes()
	}
	if s.deltaState != nil {
		n += s.deltaState.NumBytes()
	}
	return n
}

func (s *recurrentSnapshot) Close() { mlx.Unpin(s.convState, s.deltaState) }

func (c *RecurrentCache) Snapshot(seqID int, fromOffset int) Snapshot {
	idx := c.seqIndex(seqID)
	if idx < 0 {
		return nil
	}
	snap := &recurrentSnapshot{offset: c.seqOffsets[idx]}
	if c.convState != nil && c.convState.Valid() {
		row := c.convState.Slice(mlx.Slice(idx, idx+1), mlx.Slice(), mlx.Slice())
		snap.convState = mlx.Contiguous(row, false)
		mlx.Pin(snap.convState)
	}
	if c.deltaState != nil && c.deltaState.Valid() {
		row := c.deltaState.Slice(mlx.Slice(idx, idx+1), mlx.Slice(), mlx.Slice(), mlx.Slice())
		snap.deltaState = mlx.Contiguous(row, false)
		mlx.Pin(snap.deltaState)
	}
	mlx.AsyncEval(snap.convState, snap.deltaState)
	return snap
}

func (c *RecurrentCache) Restore(seqID int, snapshot Snapshot, target int) bool {
	idx := c.seqIndex(seqID)
	if idx < 0 {
		return false
	}

	if snapshot == nil {
		return target == c.seqOffsets[idx]
	}

	snap := snapshot.(*recurrentSnapshot)
	if target != snap.offset {
		return false
	}

	if snap.convState != nil {
		if c.convState == nil {
			c.ensure(len(c.seqOrder), snap.convState.DType())
		}
		c.convState.Set(c.convState.SliceUpdate(snap.convState,
			mlx.Slice(idx, idx+1), mlx.Slice(), mlx.Slice()))
	}
	if snap.deltaState != nil {
		if c.deltaState == nil {
			c.ensure(len(c.seqOrder), snap.deltaState.DType())
		}
		c.deltaState.Set(c.deltaState.SliceUpdate(snap.deltaState,
			mlx.Slice(idx, idx+1), mlx.Slice(), mlx.Slice(), mlx.Slice()))
	}
	c.seqOffsets[idx] = snap.offset
	return true
}

func (c *RecurrentCache) Merge(parent, child Snapshot) Snapshot {
	if parent != nil {
		parent.Close()
	}
	return child
}

func (c *RecurrentCache) Split(snapshot Snapshot, at int) (Snapshot, Snapshot) {
	return nil, snapshot
}

func (c *RecurrentCache) Free() {
	mlx.Unpin(c.convState, c.deltaState)
	c.convState, c.deltaState = nil, nil
	// Preserve sequence registration — callers may Restore after Free.
	for i := range c.seqOffsets {
		c.seqOffsets[i] = 0
	}
}

func (c *RecurrentCache) Offsets(seqIDs ...int) []int32 {
	offsets := make([]int32, len(seqIDs))
	for i, seqID := range seqIDs {
		idx := c.seqIndex(seqID)
		if idx >= 0 {
			offsets[i] = int32(c.seqOffsets[idx])
		}
	}
	return offsets
}

func (c *RecurrentCache) SetSeqs(seqIDs []int) {
	wanted := make(map[int]bool, len(seqIDs))
	for _, id := range seqIDs {
		wanted[id] = true
	}

	changed := len(seqIDs) != len(c.seqOrder)
	if !changed {
		for _, id := range c.seqOrder {
			if !wanted[id] {
				changed = true
				break
			}
		}
	}
	if !changed {
		return
	}

	// Build new order: preserve existing order for survivors, then append new
	newOrder := make([]int, 0, len(seqIDs))
	newOffsets := make([]int, 0, len(seqIDs))
	var survivingRows []int32
	for i, id := range c.seqOrder {
		if wanted[id] {
			survivingRows = append(survivingRows, int32(i))
			newOrder = append(newOrder, id)
			newOffsets = append(newOffsets, c.seqOffsets[i])
		}
	}
	added := make(map[int]bool, len(newOrder))
	for _, id := range newOrder {
		added[id] = true
	}
	for _, id := range seqIDs {
		if !added[id] {
			newOrder = append(newOrder, id)
			newOffsets = append(newOffsets, 0)
		}
	}

	c.seqOrder = newOrder
	c.seqOffsets = newOffsets

	// Rebuild pool tensor if it exists
	if c.convState != nil && c.convState.Valid() {
		dtype := c.convState.DType()
		if len(survivingRows) == 0 {
			mlx.Unpin(c.convState, c.deltaState)
			c.convState, c.deltaState = nil, nil
		} else if len(survivingRows) != c.convState.Dim(0) {
			takeIdx := mlx.NewArrayInt32(survivingRows, []int32{int32(len(survivingRows))})
			c.convState = c.setState(c.convState, c.convState.TakeAxis(takeIdx, 0))
			c.deltaState = c.setState(c.deltaState, c.deltaState.TakeAxis(takeIdx, 0))
		}
		if len(c.seqOrder) > 0 {
			c.ensure(len(c.seqOrder), dtype)
		}
	}
}
