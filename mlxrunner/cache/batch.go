package cache

import (
	"fmt"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlxrunner/batch"
	"github.com/ollama/ollama/mlxrunner/nn"
)

// Batcher adapts model-specific cache state to a shared decode forward. Batch
// must validate every row before returning, without mutating any owner. The
// returned Layer borrows the owners and does not manage their lifetime.
type Batcher interface {
	Batch([]Cache) (Layer, error)
}

// Batch borrows one cache bundle per request for an unpadded, single-token
// decode forward. It neither owns nor rearranges the original caches. The
// caller must keep every owner alive until the forward and its state evaluate.
// Rebuild the views when the participating requests or their order changes.
// A single row returns its original layers and retains ordinary prefill support.
//
// Standard attention and recurrent caches are adapted here. Model-specific
// caches implement Batcher. This validates cache compatibility, not whether a
// model's computation supports multiple rows.
func Batch(rows [][]Cache) ([]Layer, error) {
	if len(rows) == 0 {
		return nil, fmt.Errorf("cache batch: no rows")
	}
	if len(rows) == 1 {
		return Layers(rows[0]), nil
	}
	for i, row := range rows {
		if len(row) != len(rows[0]) {
			return nil, fmt.Errorf("cache batch: row %d has %d layers, want %d", i, len(row), len(rows[0]))
		}
	}
	layers := make([]Layer, len(rows[0]))
	seen := make(map[Cache]bool)
	for l, first := range rows[0] {
		view := batchedLayer{rows: make([]Cache, len(rows))}
		for i, row := range rows {
			c := row[l]
			var compatible bool
			switch a := first.(type) {
			case nil:
				compatible = c == nil
			case *KVCache:
				_, compatible = c.(*KVCache)
			case *RotatingKVCache:
				b, ok := c.(*RotatingKVCache)
				compatible = ok && a.maxSize == b.maxSize
			case *RecurrentCache:
				b, ok := c.(*RecurrentCache)
				compatible = ok && a.convTail == b.convTail && a.convDim == b.convDim &&
					a.numVHeads == b.numVHeads && a.headVDim == b.headVDim && a.headKDim == b.headKDim
			case Batcher:
				compatible = true // The model validates its rows below.
			default:
				return nil, fmt.Errorf("cache batch: unsupported layer %d cache %T", l, first)
			}
			if !compatible {
				return nil, fmt.Errorf("cache batch: incompatible row %d layer %d cache %T (first %T)", i, l, c, first)
			}
			if c != nil {
				if seen[c] {
					return nil, fmt.Errorf("cache batch: reused cache owner in row %d layer %d", i, l)
				}
				seen[c] = true
			}
			view.rows[i] = c
		}
		switch first := first.(type) {
		case *KVCache, *RotatingKVCache:
			layers[l] = &attentionBatch{view}
		case *RecurrentCache:
			layers[l] = &recurrentBatch{view}
		case Batcher:
			var err error
			layers[l], err = first.Batch(view.rows)
			if err != nil {
				return nil, fmt.Errorf("cache batch: layer %d: %w", l, err)
			}
		}
	}
	return layers, nil
}

type batchedLayer struct {
	rows []Cache
}

func (c *batchedLayer) State() []*mlx.Array {
	var state []*mlx.Array
	for _, row := range c.rows {
		state = append(state, row.State()...)
	}
	return state
}

func (c *batchedLayer) check(b *batch.Batch) {
	if b.InputIDs.Dim(0) != len(c.rows) || b.InputIDs.Dim(1) != 1 ||
		len(b.SeqOffsets) != len(c.rows) || len(b.SeqQueryLens) != len(c.rows) {
		panic("cache batch: expected one decode token per cache row")
	}
	for _, n := range b.SeqQueryLens {
		if n != 1 {
			panic("cache batch: padded decode rows are not supported")
		}
	}
}

type attentionBatch struct{ batchedLayer }

func (c *attentionBatch) Update(b *batch.Batch, keys, values *mlx.Array) *nn.KVHistory {
	c.check(b)
	if keys.Dim(0) != len(c.rows) || values.Dim(0) != len(c.rows) || keys.Dim(2) != 1 || values.Dim(2) != 1 {
		panic("cache batch: expected one key/value token per cache row")
	}
	histories := make([]*nn.KVHistory, len(c.rows))
	for i, row := range c.rows {
		histories[i] = row.(Attention).Update(b.Row(i),
			keys.Slice(mlx.Slice(i, i+1), mlx.Slice(), mlx.Slice(), mlx.Slice()),
			values.Slice(mlx.Slice(i, i+1), mlx.Slice(), mlx.Slice(), mlx.Slice()))
	}
	return nn.NewRowKVHistory(histories)
}

func (c *attentionBatch) View(b *batch.Batch) *nn.KVHistory {
	c.check(b)
	histories := make([]*nn.KVHistory, len(c.rows))
	for i, row := range c.rows {
		histories[i] = row.(Attention).View(b.Row(i))
	}
	return nn.NewRowKVHistory(histories)
}

type recurrentBatch struct{ batchedLayer }

func (c *recurrentBatch) Get(b *batch.Batch, dtype mlx.DType) *nn.RecurrentHistory {
	c.check(b)
	conv, delta := make([]*mlx.Array, len(c.rows)), make([]*mlx.Array, len(c.rows))
	for i, row := range c.rows {
		h := row.(Recurrent).Get(b.Row(i), dtype)
		conv[i], delta[i] = h.ConvState(), h.DeltaState()
	}
	return nn.NewRecurrentHistory(mlx.Concatenate(conv, 0), mlx.Concatenate(delta, 0))
}

func (c *recurrentBatch) SnapshotSplits(forwardLen int) []int {
	if forwardLen != 1 {
		panic("cache batch: recurrent forwards must contain one decode token")
	}
	// A single-token forward has no interior snapshot boundary. Each owner
	// still captures its independently scheduled end boundary in Put.
	return nil
}

func (c *recurrentBatch) Put(b *batch.Batch, convStates, deltaStates []*mlx.Array) {
	c.check(b)
	if len(convStates) != 1 || len(deltaStates) != 1 {
		panic("cache batch: expected one recurrent boundary state")
	}
	if convStates[0].Dim(0) != len(c.rows) || deltaStates[0].Dim(0) != len(c.rows) {
		panic("cache batch: recurrent state row count does not match caches")
	}
	for i, row := range c.rows {
		// Gather a compact row so a parked request or its snapshots do not
		// retain the entire batch's recurrent-state allocation.
		index := mlx.FromValues([]int32{int32(i)}, 1)
		row.(Recurrent).Put(b.Row(i),
			[]*mlx.Array{convStates[0].TakeAxis(index, 0)},
			[]*mlx.Array{deltaStates[0].TakeAxis(index, 0)})
	}
}
