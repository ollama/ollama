package cache

import (
	"fmt"

	"github.com/ollama/ollama/x/mlxrunner/batch"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/models/nn"
)

// Recurrent is the subset of RecurrentCache used by model Forward paths.
type Recurrent interface {
	Get(b *batch.Batch, dtype mlx.DType) *nn.RecurrentHistory
	Put(b *batch.Batch, convStates, deltaStates []*mlx.Array)
	SnapshotSplits(forwardLen int) []int
}

// MultiSeqRecurrent fans a batched recurrent Forward across per-sequence
// RecurrentCaches. Each sequence keeps B=1 state so the active set can change.
type MultiSeqRecurrent struct {
	seqs []*RecurrentCache

	convTail  int32
	convDim   int32
	numVHeads int32
	headVDim  int32
	headKDim  int32
}

// NewMultiSeqRecurrent builds a multi-sequence recurrent cache.
func NewMultiSeqRecurrent(seqs []*RecurrentCache) *MultiSeqRecurrent {
	if len(seqs) == 0 {
		panic("cache.NewMultiSeqRecurrent: empty")
	}
	s0 := seqs[0]
	return &MultiSeqRecurrent{
		seqs:      seqs,
		convTail:  int32(s0.convTail),
		convDim:   int32(s0.convDim),
		numVHeads: int32(s0.numVHeads),
		headVDim:  int32(s0.headVDim),
		headKDim:  int32(s0.headKDim),
	}
}

func (m *MultiSeqRecurrent) NumSeq() int { return len(m.seqs) }

func (m *MultiSeqRecurrent) ResetSeq(i int) error {
	if i < 0 || i >= len(m.seqs) {
		return fmt.Errorf("sequence %d out of range", i)
	}
	m.seqs[i].Free()
	m.seqs[i] = NewRecurrentCache(m.convTail, m.convDim, m.numVHeads, m.headVDim, m.headKDim)
	return nil
}

func (m *MultiSeqRecurrent) seqIndex(b *batch.Batch, row int) int {
	if b != nil && row < len(b.SeqIDs) {
		return int(b.SeqIDs[row])
	}
	return row
}

func (m *MultiSeqRecurrent) Get(b *batch.Batch, dtype mlx.DType) *nn.RecurrentHistory {
	B := b.InputIDs.Dim(0)
	var convRows, deltaRows []*mlx.Array
	for i := range B {
		seq := m.seqIndex(b, i)
		rowBatch := &batch.Batch{
			InputIDs:     b.InputIDs.Slice(mlx.Slice(i, i+1), mlx.Slice()),
			SeqOffsets:   []int32{b.SeqOffsets[i]},
			SeqQueryLens: []int32{b.SeqQueryLens[i]},
		}
		h := m.seqs[seq].Get(rowBatch, dtype)
		convRows = append(convRows, h.ConvState())
		deltaRows = append(deltaRows, h.DeltaState())
	}
	return nn.NewRecurrentHistory(stackRows(convRows), stackRows(deltaRows))
}

func (m *MultiSeqRecurrent) Put(b *batch.Batch, convStates, deltaStates []*mlx.Array) {
	// Parallel path does not segment recurrent kernels for prefix snapshots.
	if len(convStates) == 0 || len(deltaStates) == 0 {
		panic("cache.MultiSeqRecurrent: empty boundary states")
	}
	convEnd := convStates[len(convStates)-1]
	deltaEnd := deltaStates[len(deltaStates)-1]
	B := b.InputIDs.Dim(0)
	for i := range B {
		seq := m.seqIndex(b, i)
		rowBatch := &batch.Batch{
			InputIDs:     b.InputIDs.Slice(mlx.Slice(i, i+1), mlx.Slice()),
			SeqOffsets:   []int32{b.SeqOffsets[i]},
			SeqQueryLens: []int32{b.SeqQueryLens[i]},
		}
		m.seqs[seq].Put(rowBatch,
			[]*mlx.Array{convEnd.Slice(mlx.Slice(i, i+1), mlx.Slice(), mlx.Slice())},
			[]*mlx.Array{deltaEnd.Slice(mlx.Slice(i, i+1), mlx.Slice(), mlx.Slice(), mlx.Slice())},
		)
	}
}

func (m *MultiSeqRecurrent) SnapshotSplits(int) []int { return nil }

func stackRows(rows []*mlx.Array) *mlx.Array {
	if len(rows) == 1 {
		return rows[0]
	}
	out := rows[0]
	for _, r := range rows[1:] {
		out = out.Concatenate(0, r)
	}
	return out
}

func (m *MultiSeqRecurrent) State() []*mlx.Array {
	var out []*mlx.Array
	for _, s := range m.seqs {
		out = append(out, s.State()...)
	}
	return out
}

func (m *MultiSeqRecurrent) Free() {
	for _, s := range m.seqs {
		s.Free()
	}
}

func (m *MultiSeqRecurrent) Offset() int {
	max := 0
	for _, s := range m.seqs {
		if o := s.Offset(); o > max {
			max = o
		}
	}
	return max
}

func (m *MultiSeqRecurrent) Snapshot(int) Snapshot {
	panic("cache.MultiSeqRecurrent: Snapshot not supported")
}
func (m *MultiSeqRecurrent) PrepareSnapshots([]int) {
	panic("cache.MultiSeqRecurrent: PrepareSnapshots not supported")
}
func (m *MultiSeqRecurrent) TakeSnapshots() []Snapshot {
	panic("cache.MultiSeqRecurrent: TakeSnapshots not supported")
}
func (m *MultiSeqRecurrent) Restore(Snapshot, int) bool {
	panic("cache.MultiSeqRecurrent: Restore not supported")
}
func (m *MultiSeqRecurrent) Merge(Snapshot, Snapshot) Snapshot {
	panic("cache.MultiSeqRecurrent: Merge not supported")
}
func (m *MultiSeqRecurrent) Split(Snapshot, int) (Snapshot, Snapshot) {
	panic("cache.MultiSeqRecurrent: Split not supported")
}
