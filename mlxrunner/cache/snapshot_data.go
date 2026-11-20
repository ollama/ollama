package cache

import (
	"slices"

	"github.com/ollama/ollama/mlx"
)

// SnapshotData describes immutable state for persistence. Arrays are borrowed
// only for the current MLX scope; exporting does not materialize lazy snapshots
// or change the live cache. SequenceAxis is -1 for whole-state checkpoints.
type SnapshotData struct {
	Arrays       []*mlx.Array `json:"-"`
	Start, End   int
	Cursor       int
	SequenceAxis int
}

// SnapshotCodec exposes cache state without coupling disk storage to a model.
// RestoreData borrows independently owned writable arrays, never file mappings.
// It preserves Restore's prefix-continuity and endpoint restrictions.
// Passing nil to SnapshotData exports the current live state.
type SnapshotCodec interface {
	SnapshotData(Snapshot) SnapshotData
	RestoreData(SnapshotData, int) bool
}

func (c *KVCache) SnapshotData(snapshot Snapshot) SnapshotData {
	if snapshot == nil {
		return SnapshotData{Arrays: c.State(), End: c.offset, SequenceAxis: 2}
	}
	s := snapshot.(*kvSnapshot)
	k, v := s.keys, s.values
	if s.cache != nil {
		k = s.cache.keys.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(s.fromOffset, s.toOffset), mlx.Slice())
		v = s.cache.values.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(s.fromOffset, s.toOffset), mlx.Slice())
	}
	return SnapshotData{Arrays: []*mlx.Array{k, v}, Start: s.fromOffset, End: s.toOffset, SequenceAxis: 2}
}

func (c *KVCache) RestoreData(data SnapshotData, target int) bool {
	if !attentionData(data) || data.SequenceAxis != 2 || data.Cursor != 0 || data.Arrays[0].Dim(2) != data.End-data.Start {
		return false
	}
	s := &kvSnapshot{keys: data.Arrays[0], values: data.Arrays[1], fromOffset: data.Start, toOffset: data.End}
	return c.Restore(s, target)
}

func (c *RotatingKVCache) SnapshotData(snapshot Snapshot) SnapshotData {
	var s *rotatingSnapshot
	if snapshot == nil {
		state := c.State()
		if len(state) == 0 {
			return SnapshotData{SequenceAxis: -1}
		}
		s = &rotatingSnapshot{keys: state[0], values: state[1], toOffset: c.offset, idx: c.idx}
	} else {
		s = snapshot.(*rotatingSnapshot)
	}
	k, v := s.keys, s.values
	if s.cache != nil {
		k = s.cache.keys.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(s.sliceStart, s.sliceEnd), mlx.Slice())
		v = s.cache.values.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(s.sliceStart, s.sliceEnd), mlx.Slice())
	}
	data := SnapshotData{Arrays: []*mlx.Array{k, v}, Start: s.fromOffset, End: s.toOffset, Cursor: s.idx, SequenceAxis: -1}
	// A multi-token forward may retain history preceding its final window.
	// Such storage is linear, so export only its restorable tail. A ring at
	// decode size keeps its original order and write cursor.
	if start := k.Dim(2) - c.maxSize; start > 0 {
		for i, a := range data.Arrays {
			data.Arrays[i] = a.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(start, mlx.End), mlx.Slice())
		}
		data.Cursor -= start
	}
	return data
}

func (c *RotatingKVCache) RestoreData(data SnapshotData, target int) bool {
	if !attentionData(data) || data.SequenceAxis != -1 || data.Cursor < 0 || data.Cursor > data.Arrays[0].Dim(2) || data.Arrays[0].Dim(2) != min(data.End, c.maxSize) {
		return false
	}
	s := &rotatingSnapshot{keys: data.Arrays[0], values: data.Arrays[1], fromOffset: data.Start, toOffset: data.End, idx: data.Cursor}
	return c.Restore(s, target)
}

func attentionData(data SnapshotData) bool {
	if !snapshotArrays(data, 2) {
		return false
	}
	k, v := data.Arrays[0], data.Arrays[1]
	return k.NumDims() == 4 && v.NumDims() == 4 && k.Dim(0) == 1 && v.Dim(0) == 1 && k.Dim(1) == v.Dim(1) && k.Dim(2) == v.Dim(2) && k.DType() == v.DType()
}

func (c *RecurrentCache) SnapshotData(snapshot Snapshot) SnapshotData {
	if snapshot == nil {
		return SnapshotData{Arrays: []*mlx.Array{c.convState, c.deltaState}, End: c.offset, SequenceAxis: -1}
	}
	s := snapshot.(*recurrentSnapshot)
	return SnapshotData{Arrays: []*mlx.Array{s.convState, s.deltaState}, End: s.offset, SequenceAxis: -1}
}

func (c *RecurrentCache) RestoreData(data SnapshotData, target int) bool {
	if !snapshotArrays(data, 2) || data.SequenceAxis != -1 || data.Start != 0 || data.Cursor != 0 ||
		!slices.Equal(data.Arrays[0].Dims(), []int{1, c.convTail, c.convDim}) ||
		!slices.Equal(data.Arrays[1].Dims(), []int{1, c.numVHeads, c.headVDim, c.headKDim}) || data.Arrays[1].DType() != mlx.DTypeFloat32 {
		return false
	}
	s := &recurrentSnapshot{convState: data.Arrays[0], deltaState: data.Arrays[1], offset: data.End}
	return c.Restore(s, target)
}

func (c *HiddenCache) SnapshotData(snapshot Snapshot) SnapshotData {
	if snapshot == nil {
		return SnapshotData{Arrays: c.State(), End: c.Offset(), SequenceAxis: 1}
	}
	s := snapshot.(*hiddenSnapshot)
	return SnapshotData{Arrays: []*mlx.Array{s.hidden}, Start: s.start, End: s.start + s.hidden.Dim(1), SequenceAxis: 1}
}

func (c *HiddenCache) RestoreData(data SnapshotData, target int) bool {
	if !snapshotArrays(data, 1) || data.SequenceAxis != 1 || data.Cursor != 0 || data.Arrays[0].NumDims() != 3 || data.Arrays[0].Dim(0) != 1 || data.Arrays[0].Dim(1) != data.End-data.Start {
		return false
	}
	s := &hiddenSnapshot{hidden: data.Arrays[0], start: data.Start, backingBytes: data.Arrays[0].NumBytes()}
	return c.Restore(s, target)
}

func snapshotArrays(data SnapshotData, count int) bool {
	return data.Start >= 0 && data.End > data.Start && len(data.Arrays) == count && !slices.Contains(data.Arrays, nil)
}
