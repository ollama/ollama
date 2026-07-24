package cache

import (
	"fmt"

	"github.com/ollama/ollama/x/mlxrunner/batch"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/models/nn"
)

// MultiSeq fans a batched Forward across independent per-sequence Attention
// caches and returns a stacked KVHistory for SDPA. Used when OLLAMA_NUM_PARALLEL > 1.
type MultiSeq struct {
	seqs []Attention
}

// NewMultiSeq builds a multi-sequence cache from per-sequence Attention caches.
func NewMultiSeq(seqs []Attention) *MultiSeq {
	if len(seqs) == 0 {
		panic("cache.NewMultiSeq: empty")
	}
	return &MultiSeq{seqs: seqs}
}

// NumSeq returns the number of sequence slots.
func (m *MultiSeq) NumSeq() int { return len(m.seqs) }

// ResetSeq frees and replaces one sequence's Attention cache with a fresh
// instance of the same type (KV or rotating).
func (m *MultiSeq) ResetSeq(i int) error {
	if i < 0 || i >= len(m.seqs) {
		return fmt.Errorf("sequence %d out of range", i)
	}
	fresh, ok := cloneAttention(m.seqs[i])
	if !ok {
		return fmt.Errorf("sequence %d: unsupported attention cache %T", i, m.seqs[i])
	}
	m.seqs[i].Free()
	m.seqs[i] = fresh
	return nil
}

func cloneAttention(c Attention) (Attention, bool) {
	switch t := c.(type) {
	case *KVCache:
		return NewKVCache(), true
	case *RotatingKVCache:
		return NewRotatingKVCache(t.maxSize), true
	default:
		return nil, false
	}
}

func (m *MultiSeq) seqIndex(b *batch.Batch, row int) int {
	if b != nil && row < len(b.SeqIDs) {
		return int(b.SeqIDs[row])
	}
	return row
}

func (m *MultiSeq) Update(b *batch.Batch, keys, values *mlx.Array) *nn.KVHistory {
	B := keys.Dim(0)
	for i := range B {
		seq := m.seqIndex(b, i)
		if seq < 0 || seq >= len(m.seqs) {
			panic(fmt.Sprintf("cache.MultiSeq: seq id %d out of range [0,%d)", seq, len(m.seqs)))
		}
		rowBatch := &batch.Batch{
			InputIDs:     b.InputIDs,
			SeqOffsets:   []int32{b.SeqOffsets[i]},
			SeqQueryLens: []int32{b.SeqQueryLens[i]},
		}
		k := keys.Slice(mlx.Slice(i, i+1), mlx.Slice(), mlx.Slice(), mlx.Slice())
		v := values.Slice(mlx.Slice(i, i+1), mlx.Slice(), mlx.Slice(), mlx.Slice())
		_ = m.seqs[seq].Update(rowBatch, k, v)
	}
	return m.stack(b)
}

func (m *MultiSeq) View(b *batch.Batch) *nn.KVHistory {
	return m.stack(b)
}

func (m *MultiSeq) stack(b *batch.Batch) *nn.KVHistory {
	B := len(b.SeqOffsets)
	histories := make([]*nn.KVHistory, B)
	kLens := make([]int32, B)
	maxK := 0
	dtype := mlx.DTypeFloat16
	var H, Dk, Dv int
	for i := range B {
		seq := m.seqIndex(b, i)
		rowBatch := &batch.Batch{
			SeqOffsets:   []int32{b.SeqOffsets[i]},
			SeqQueryLens: []int32{b.SeqQueryLens[i]},
		}
		h := m.seqs[seq].View(rowBatch)
		histories[i] = h
		if h == nil || h.K() == nil {
			kLens[i] = 0
			continue
		}
		k := h.K()
		dtype = k.DType()
		H, Dk = k.Dim(1), k.Dim(3)
		Dv = h.V().Dim(3)
		kLens[i] = int32(k.Dim(2))
		if int(kLens[i]) > maxK {
			maxK = int(kLens[i])
		}
	}
	if maxK == 0 {
		return nn.NewKVHistory(nil, nil, nil)
	}

	stackedK := mlx.Zeros(dtype, B, H, maxK, Dk)
	stackedV := mlx.Zeros(dtype, B, H, maxK, Dv)
	for i, h := range histories {
		if h == nil || h.K() == nil || kLens[i] == 0 {
			continue
		}
		n := int(kLens[i])
		stackedK.Set(stackedK.SliceUpdate(h.K(), mlx.Slice(i, i+1), mlx.Slice(), mlx.Slice(0, n), mlx.Slice()))
		stackedV.Set(stackedV.SliceUpdate(h.V(), mlx.Slice(i, i+1), mlx.Slice(), mlx.Slice(0, n), mlx.Slice()))
	}
	return nn.NewKVHistory(stackedK, stackedV, kLensApplier{b: b, K: maxK, kLens: kLens, dtype: dtype})
}

type kLensApplier struct {
	b     *batch.Batch
	K     int
	kLens []int32
	dtype mlx.DType
}

func (a kLensApplier) ApplyMask(logical nn.AttentionMask) nn.AttentionMask {
	return logical.Intersect(nn.KPaddingMask(a.b, a.K, a.kLens, a.dtype))
}

func (m *MultiSeq) State() []*mlx.Array {
	var out []*mlx.Array
	for _, s := range m.seqs {
		out = append(out, s.State()...)
	}
	return out
}

func (m *MultiSeq) Free() {
	for _, s := range m.seqs {
		s.Free()
	}
}

func (m *MultiSeq) Offset() int {
	max := 0
	for _, s := range m.seqs {
		if o := s.Offset(); o > max {
			max = o
		}
	}
	return max
}

// Snapshot/Prepare/Take/Restore/Merge/Split are unused when the prefix trie is
// disabled for parallel > 1. They panic so misuse is obvious.
func (m *MultiSeq) Snapshot(int) Snapshot {
	panic("cache.MultiSeq: Snapshot not supported")
}
func (m *MultiSeq) PrepareSnapshots([]int) {
	panic("cache.MultiSeq: PrepareSnapshots not supported")
}
func (m *MultiSeq) TakeSnapshots() []Snapshot {
	panic("cache.MultiSeq: TakeSnapshots not supported")
}
func (m *MultiSeq) Restore(Snapshot, int) bool {
	panic("cache.MultiSeq: Restore not supported")
}
func (m *MultiSeq) Merge(Snapshot, Snapshot) Snapshot {
	panic("cache.MultiSeq: Merge not supported")
}
func (m *MultiSeq) Split(Snapshot, int) (Snapshot, Snapshot) {
	panic("cache.MultiSeq: Split not supported")
}

// WrapParallelCaches replaces each supported layer with an n-way multi-seq
// cache. Supports *KVCache, *RotatingKVCache, and *RecurrentCache. Returns
// ok=false when any layer type is unsupported.
func WrapParallelCaches(caches []Cache, n int) ([]Cache, bool) {
	if n <= 1 {
		return caches, true
	}
	out := make([]Cache, len(caches))
	for i, c := range caches {
		switch t := c.(type) {
		case *KVCache:
			seqs := make([]Attention, n)
			seqs[0] = t
			for j := 1; j < n; j++ {
				seqs[j] = NewKVCache()
			}
			out[i] = NewMultiSeq(seqs)
		case *RotatingKVCache:
			seqs := make([]Attention, n)
			seqs[0] = t
			for j := 1; j < n; j++ {
				seqs[j] = NewRotatingKVCache(t.maxSize)
			}
			out[i] = NewMultiSeq(seqs)
		case *RecurrentCache:
			seqs := make([]*RecurrentCache, n)
			seqs[0] = t
			for j := 1; j < n; j++ {
				seqs[j] = NewRecurrentCache(int32(t.convTail), int32(t.convDim), int32(t.numVHeads), int32(t.headVDim), int32(t.headKDim))
			}
			out[i] = NewMultiSeqRecurrent(seqs)
		default:
			return nil, false
		}
	}
	return out, true
}
