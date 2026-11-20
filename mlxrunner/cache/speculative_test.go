package cache

import (
	"fmt"
	"math"
	"testing"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlx/mlxtest"
	"github.com/ollama/ollama/mlxrunner/batch"
	"github.com/ollama/ollama/mlxrunner/nn"
)

func TestSpeculativeBatchOwnsSelectedRows(t *testing.T) {
	for _, keep := range []int{3, 8} {
		mlxtest.RunSubtest(t, fmt.Sprintf("keep=%d", keep), func(t *mlxtest.T) {
			const B, L, Hk, Hv, D = 8, 8, 16, 48, 128
			const survivor = 3
			const rowBytes = Hv * D * D * 4
			mlx.DefaultStream().Synchronize()
			mlx.ClearCache()
			before := mlx.ActiveMemory()
			var retained *RecurrentCache
			mlx.Scoped(func() {
				owners, bundles := make([]*RecurrentCache, B), make([][]Cache, B)
				var transaction *SpeculativeBatch
				out := mlx.ScopedArrays(func() []*mlx.Array {
					b := &batch.Batch{InputIDs: mlx.Zeros(mlx.DTypeInt32, B, L)}
					for row := range B {
						owner := NewRecurrentCache(2, 3, Hv, D, D)
						owners[row], bundles[row] = owner, []Cache{owner}
						h := owner.Get(newKVBatch(0, 1), mlx.DTypeBFloat16)
						owner.Put(newKVBatch(0, 1), []*mlx.Array{h.ConvState()}, []*mlx.Array{mlx.AddScalar(h.DeltaState(), float32(row+1)*.01)})
						b.SeqOffsets = append(b.SeqOffsets, 1)
						b.SeqQueryLens = append(b.SeqQueryLens, L)
						owner.PrepareSnapshots([]int{2, 3, 4, 5, 6, 7, 8})
					}
					layers, err := Batch(bundles)
					if err != nil {
						t.Fatal(err)
					}
					transaction = NewSpeculativeBatch(layers)
					c := transaction.Layers[0].(RecurrentReplay)
					h := c.Get(b, mlx.DTypeBFloat16)
					packed := mlx.AddScalar(mlx.Zeros(mlx.DTypeBFloat16, B, L, (2*Hk+Hv)*D), .125)
					ba := mlx.AddScalar(mlx.Zeros(mlx.DTypeBFloat16, B, L, 2*Hv), .1)
					dt := mlx.Zeros(mlx.DTypeBFloat16, Hv)
					a := mlx.AddScalar(mlx.Zeros(mlx.DTypeFloat32, Hv), .2)
					y, end, inputs := nn.GatedDeltaForReplay(b, packed, ba, dt, a, nn.WithRecurrentHistory(h))
					conv := make([]*mlx.Array, L)
					for i := range conv {
						conv[i] = mlx.AddScalar(h.ConvState(), float32(i+1))
					}
					c.PutReplay(b, conv, end, inputs)
					return []*mlx.Array{y}
				})
				mlx.Eval(out...)
				selected := make([]int, B)
				for i := range selected {
					selected[i] = 1 + 1 + i%L
				}
				selected[survivor] = 1 + keep
				transaction.Resolve(selected)
				var states []*mlx.Array
				for _, owner := range owners {
					for _, snap := range owner.TakeSnapshots() {
						if snap != nil {
							t.Fatal("deferred state escaped as a snapshot")
						}
					}
					states = append(states, owner.State()...)
				}
				transaction.Close()
				mlx.AsyncEval(states...)
				for i, owner := range owners {
					if i == survivor {
						retained = owner
					} else {
						owner.Free()
					}
				}
			})
			defer retained.Free()
			mlx.DefaultStream().Synchronize()
			mlx.ClearCache()
			// Measure before value reads can accidentally detach a backing view.
			held := mlx.ActiveMemory() - before
			t.Logf("keep=%d row_state_bytes=%d retained_active_bytes=%d bound=%d", keep, rowBytes, held, 2*rowBytes+256*1024)
			if held > 2*rowBytes+256*1024 {
				t.Fatalf("one %d-byte state retains %d bytes after all peer/input owners close", rowBytes, held)
			}
			if retained.Offset() != 1+keep || retained.State()[1].NumBytes() != rowBytes || retained.PrepareCompaction() != nil {
				t.Fatal("selected state lost its compact owner or exact offset")
			}
			for _, value := range retained.State()[1].Floats() {
				if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
					t.Fatal("selected state is not finite")
				}
			}
		})
	}
}

func TestSpeculativeBatchOrdinaryPut(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		a, b := NewRecurrentCache(1, 1, 1, 1, 1), NewRecurrentCache(1, 1, 1, 1, 1)
		defer a.Free()
		defer b.Free()
		layers, err := Batch([][]Cache{{a}, {b}})
		if err != nil {
			t.Fatal(err)
		}
		transaction := NewSpeculativeBatch(layers)
		defer transaction.Close()
		c := transaction.Layers[0].(Recurrent)
		input := &batch.Batch{InputIDs: mlx.Zeros(mlx.DTypeInt32, 2, 1), SeqOffsets: []int32{0, 0}, SeqQueryLens: []int32{1, 1}}
		h := c.Get(input, mlx.DTypeFloat32)
		c.Put(input, []*mlx.Array{h.ConvState()}, []*mlx.Array{mlx.AddScalar(h.DeltaState(), 2)})
		transaction.Resolve([]int{1, 1})
		mlx.Eval(c.State()...)
		for _, owner := range []*RecurrentCache{a, b} {
			if owner.Offset() != 1 || owner.State()[1].Floats()[0] != 2 {
				t.Fatal("ordinary recurrent Put was altered")
			}
		}
	})
}
