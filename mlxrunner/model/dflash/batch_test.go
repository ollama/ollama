package dflash

import (
	"fmt"
	"slices"
	"testing"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlx/mlxtest"
	"github.com/ollama/ollama/mlxrunner/batch"
	"github.com/ollama/ollama/mlxrunner/cache"
	"github.com/ollama/ollama/mlxrunner/nn"
)

func TestDraftBatchIndependentContexts(t *testing.T) {
	for _, convolutions := range []bool{false, true} {
		mlxtest.RunSubtest(t, fmt.Sprintf("convolutions=%v", convolutions), func(t *mlxtest.T) {
			m := newDraftRotaryTestModel(t, `,"no_rope_layers":[1]`)
			m.SlidingWindow = 4
			if convolutions {
				for _, layer := range m.Layers {
					conv := &convolution{
						Base: mlx.FromValues([]float32{1, 1, 0.1, 0.2, 1, 1, 0.2, 0.1}, 2, 2, 2),
						Projection: nn.NewLinear(mlx.FromValues([]float32{
							0.1, 0.2, 0.2, 0.1, -0.1, 0.1, 0.1, -0.1,
							0.2, 0.1, 0.1, 0.2, 0.1, -0.1, -0.1, 0.1,
						}, 8, 2), nil),
						Group: 1, Kernel: 2,
					}
					layer.AttentionConv, layer.MLPConv = conv, conv
					identity := mlx.FromValues([]float32{1, 0, 0, 1}, 2, 2)
					layer.MLP = &MLP{GateUpProj: nn.NewLinear(identity.Concatenate(0, identity), nil), DownProj: nn.NewLinear(identity, nil)}
				}
			}

			compare := func(label string, got, want *mlx.Array) {
				if !slices.Equal(got.Dims(), want.Dims()) {
					t.Fatalf("%s shape %v, want %v", label, got.Dims(), want.Dims())
				}
				values := mlx.Contiguous(want, false).AsType(mlx.DTypeFloat32).Floats()
				expected := make([]float64, len(values))
				for i, v := range values {
					expected[i] = float64(v)
				}
				checkDraftRotaryArray(t, label, got, expected)
			}
			features := func(row, count, round int) *mlx.Array {
				if count == 0 {
					return nil
				}
				values := make([]float32, count*2)
				for i := range values {
					values[i] = float32((i+3*row+round)%7-3) / 4
				}
				return mlx.FromValues(values, 1, count, 2)
			}
			const rows = 3
			serial, together := make([][]cache.Cache, rows), make([][]cache.Cache, rows)
			for i, seed := range []int{0, 3, 9} {
				serial[i], together[i] = m.NewCaches(), m.NewCaches()
				for _, caches := range [][]cache.Cache{serial[i], together[i]} {
					for _, c := range caches {
						defer c.Free()
					}
					if seed > 0 {
						m.Forward(&batch.Batch{SeqOffsets: []int32{0}, Hidden: features(i, seed, 0)}, nil, cache.Layers(caches))
					}
				}
			}

			for round, counts := range [][]int{{2, 0, 1}, {0, 3, 2}, {1, 2, 0}, {0, 0, 0}} {
				inputs := make([]*batch.Batch, rows)
				want := make([]*mlx.Array, rows)
				for i, count := range counts {
					offset := serial[i][0].Offset()
					inputs[i] = &batch.Batch{
						InputIDs:   mlx.FromValues([]int32{int32(i), 0, 1, 2, 0, 2, 1, 0}, 1, 8),
						SeqOffsets: []int32{int32(offset)}, SeqQueryLens: []int32{8}, Hidden: features(i, count, round+1),
					}
					for _, caches := range [][]cache.Cache{serial[i], together[i]} {
						for _, c := range caches {
							c.PrepareSnapshots([]int{offset + count})
						}
					}
					want[i], _ = m.Forward(inputs[i], nil, cache.Layers(serial[i]))
				}
				got := m.ForwardBatch(inputs, together)
				for i := range rows {
					label := fmt.Sprintf("round %d row %d", round, i)
					compare(label+" hidden", got.Slice(mlx.Slice(i, i+1), mlx.Slice(), mlx.Slice()), want[i])
					for layer := range m.Layers {
						a, b := together[i][layer], serial[i][layer]
						if a.Offset() != b.Offset() {
							t.Fatalf("%s layer %d offset=%d want=%d", label, layer, a.Offset(), b.Offset())
						}
						for state, v := range a.State() {
							compare(fmt.Sprintf("%s layer %d state %d", label, layer, state), v, b.State()[state])
						}
						boundary := int(inputs[i].SeqOffsets[0]) + counts[i]
						for _, c := range []cache.Cache{a, b} {
							snapshots := c.TakeSnapshots()
							if len(snapshots) != 1 {
								t.Fatalf("%s snapshots=%d want=1", label, len(snapshots))
							}
							if !c.Restore(nil, boundary) && !c.Restore(snapshots[0], boundary) {
								t.Fatalf("%s failed rollback to %d", label, boundary)
							}
							if snapshots[0] != nil {
								snapshots[0].Close()
							}
						}
						for state, v := range a.State() {
							compare(fmt.Sprintf("%s restored layer %d state %d", label, layer, state), v, b.State()[state])
						}
					}
				}
			}
		})
	}
}
