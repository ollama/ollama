package cache

import (
	"slices"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlxrunner/batch"
	"github.com/ollama/ollama/mlxrunner/nn"
)

// RecurrentReplay is available only inside a speculative verification batch.
// Ordinary cache writes and persistent snapshots retain their existing contract.
type RecurrentReplay interface {
	Recurrent
	PutReplay(*batch.Batch, []*mlx.Array, *mlx.Array, nn.GatedDeltaReplay)
}

// SpeculativeBatch owns deferred recurrent inputs until acceptance selects each
// row's exact storage offset. Resolve installs ordinary live states; Close only
// releases these transient inputs. It never publishes recipe snapshots.
type SpeculativeBatch struct {
	Layers    []Layer
	scope     *mlx.Scope
	recurrent []*speculativeRecurrent
	resolved  bool
}

// NewSpeculativeBatch wraps validated multirow recurrent layers. A one-row or
// attention-only batch needs no wrapper and returns nil.
func NewSpeculativeBatch(layers []Layer) *SpeculativeBatch {
	s := &SpeculativeBatch{Layers: slices.Clone(layers), scope: mlx.NewScope()}
	for i, layer := range layers {
		if recurrent, ok := layer.(*recurrentBatch); ok {
			wrapped := &speculativeRecurrent{recurrentBatch: recurrent, batch: s}
			s.Layers[i] = wrapped
			s.recurrent = append(s.recurrent, wrapped)
		}
	}
	if len(s.recurrent) == 0 {
		s.Close()
		return nil
	}
	return s
}

func (s *SpeculativeBatch) keep(a *mlx.Array) *mlx.Array {
	a = a.Clone()
	s.scope.Attach(a)
	return a
}

func (s *SpeculativeBatch) Close() {
	if s != nil {
		s.scope.Close()
	}
}

type speculativeRecurrent struct {
	*recurrentBatch
	batch      *SpeculativeBatch
	inputs     *nn.GatedDeltaReplay
	conv       []*mlx.Array
	end        *mlx.Array
	starts     []int
	boundaries []int
}

func (c *speculativeRecurrent) PutReplay(b *batch.Batch, conv []*mlx.Array, end *mlx.Array, inputs nn.GatedDeltaReplay) {
	c.check(b)
	if c.inputs != nil || c.batch.resolved {
		panic("speculative recurrence: forward already supplied")
	}
	n := b.InputIDs.Dim(1)
	c.boundaries = append(c.SnapshotSplits(n), n)
	if len(conv) != len(c.boundaries) || end.Dim(0) != len(c.rows) {
		panic("speculative recurrence: boundary states do not match rows")
	}
	c.starts = make([]int, len(c.rows))
	for i, owner := range c.rows {
		c.starts[i] = owner.Offset()
		if c.starts[i] != int(b.SeqOffsets[i]) {
			panic("speculative recurrence: input offset does not match owner")
		}
	}
	inputs.Packed, inputs.BA = c.batch.keep(inputs.Packed), c.batch.keep(inputs.BA)
	inputs.DtBias, inputs.AExp = c.batch.keep(inputs.DtBias), c.batch.keep(inputs.AExp)
	inputs.Prior = c.batch.keep(inputs.Prior)
	inputs.ValidLengths = slices.Clone(inputs.ValidLengths)
	c.inputs, c.end = &inputs, c.batch.keep(end)
	for _, state := range conv {
		c.conv = append(c.conv, c.batch.keep(state))
	}
}

func (c *speculativeRecurrent) State() []*mlx.Array {
	if c.inputs != nil && !c.batch.resolved {
		panic("speculative recurrence: resolve before requesting state")
	}
	return c.recurrentBatch.State()
}

// Resolve builds at most one replay per deferred layer, then installs the
// selected states together. No owner changes while the result graphs are built.
func (s *SpeculativeBatch) Resolve(offsets []int) {
	if s == nil {
		return
	}
	if s.resolved {
		panic("speculative recurrence: batch already resolved")
	}
	for _, c := range s.recurrent {
		if c.inputs == nil { // An ordinary Put, including other recurrent models.
			continue
		}
		if len(offsets) != len(c.rows) {
			panic("speculative recurrence: selected offsets do not match rows")
		}
		for i, offset := range offsets {
			n := offset - c.starts[i]
			if n < 0 || n > c.inputs.Packed.Dim(1) || (n != 0 && !slices.Contains(c.boundaries, n)) {
				panic("speculative recurrence: selected offset is not a boundary")
			}
		}
	}
	mlx.Scoped(func() {
		type selected struct {
			owner       *RecurrentCache
			conv, delta *mlx.Array
			offset      int
		}
		var states []selected
		for _, c := range s.recurrent {
			if c.inputs == nil {
				continue
			}
			L := c.inputs.Packed.Dim(1)
			prefixes := make([]int32, len(c.rows))
			interior := false
			for i, offset := range offsets {
				prefixes[i] = int32(offset - c.starts[i])
				interior = interior || (prefixes[i] > 0 && int(prefixes[i]) < L)
			}
			var replay *mlx.Array
			if interior {
				replay = c.inputs.Replay(prefixes)
			}
			for i, row := range c.rows {
				owner := row.(*RecurrentCache)
				conv, delta := owner.convState, owner.deltaState
				if n := int(prefixes[i]); n > 0 {
					index := mlx.FromValues([]int32{int32(i)}, 1)
					at, _ := slices.BinarySearch(c.boundaries, n)
					conv = c.conv[at].TakeAxis(index, 0)
					state := replay
					if n == L {
						state = c.end
					}
					delta = state.TakeAxis(index, 0)
				}
				states = append(states, selected{owner, conv, delta, offsets[i]})
			}
		}
		for _, state := range states {
			c := state.owner
			c.convState = c.setState(c.convState, state.conv)
			c.deltaState = c.setState(c.deltaState, state.delta)
			c.offset, c.needsCompact = state.offset, false
		}
	})
	s.resolved = true
}
