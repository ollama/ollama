package cache

import (
	"fmt"
	"math"
	"slices"
	"testing"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlx/mlxtest"
	"github.com/ollama/ollama/mlxrunner/batch"
	"github.com/ollama/ollama/mlxrunner/nn"
)

func compareRowArrays(got, want *mlx.Array) error {
	if !slices.Equal(got.Dims(), want.Dims()) {
		return fmt.Errorf("shape = %v, want %v", got.Dims(), want.Dims())
	}
	mlx.Eval(got, want)
	g, w := got.Floats(), want.Floats()
	for i := range w {
		if math.IsNaN(float64(g[i])) || math.Abs(float64(g[i]-w[i])) > 1e-5 {
			return fmt.Errorf("element %d = %g, want %g", i, g[i], w[i])
		}
	}
	return nil
}

func TestBatchAttentionRows(t *testing.T) {
	for _, kind := range []string{"full", "rotating"} {
		mlxtest.RunSubtest(t, kind, func(t *mlxtest.T) {
			newCache := func() Cache {
				if kind == "rotating" {
					return NewRotatingKVCache(4)
				}
				return NewKVCache()
			}
			owners, refs := make([]Cache, 3), make([]Cache, 3)
			for i := range owners {
				owners[i], refs[i] = newCache(), newCache()
				defer owners[i].Free()
				defer refs[i].Free()
			}
			kv := func(row, pos int) (*mlx.Array, *mlx.Array) {
				values := []float32{float32(row+1) * .2, float32(pos+1) * .1, -.4, .7}
				k := mlx.FromValues(values, 1, 1, 1, 4)
				return k, mlx.MulScalar(k, float32(row+2))
			}
			// Different history lengths and ring positions before requests meet.
			for row, n := range []int{1, 7, 3} {
				for pos := range n {
					k, v := kv(row, pos)
					for _, c := range []Cache{owners[row], refs[row]} {
						c.(Attention).Update(newKVBatch(pos, 1), k, v)
					}
				}
			}
			for _, active := range [][]int{{0, 1}, {1, 0}, {2, 1}, {0}, {2, 0}} {
				rows := make([][]Cache, len(active))
				b := &batch.Batch{
					InputIDs: mlx.Zeros(mlx.DTypeInt32, len(active), 1),
				}
				var ks, vs, qs []*mlx.Array
				for i, row := range active {
					rows[i] = []Cache{owners[row]}
					pos := owners[row].Offset()
					b.SeqOffsets = append(b.SeqOffsets, int32(pos))
					b.SeqQueryLens = append(b.SeqQueryLens, 1)
					k, v := kv(row, pos)
					ks, vs = append(ks, k), append(vs, v)
					qs = append(qs, mlx.AddScalar(k, .3))
				}
				layers, err := Batch(rows)
				if err != nil {
					t.Fatal(err)
				}
				c := layers[0].(Attention)
				h := c.Update(b, mlx.Concatenate(ks, 0), mlx.Concatenate(vs, 0))
				q := mlx.Concatenate(qs, 0)
				got := nn.ScaledDotProductAttention(b, q, .5, nn.WithKVHistory(h), nn.WithMask(nn.CausalMask()))
				// View is also the read-only shared-history path used by donors.
				view := nn.ScaledDotProductAttention(b, q, .5, nn.WithKVHistory(c.View(b)), nn.WithMask(nn.CausalMask()))
				var want []*mlx.Array
				for i, row := range active {
					rb := newKVBatch(refs[row].Offset(), 1)
					rh := refs[row].(Attention).Update(rb, ks[i], vs[i])
					want = append(want, nn.ScaledDotProductAttention(rb, qs[i], .5, nn.WithKVHistory(rh), nn.WithMask(nn.CausalMask())))
					if owners[row].Offset() != refs[row].Offset() {
						t.Fatalf("row %d offset = %d, want %d", row, owners[row].Offset(), refs[row].Offset())
					}
				}
				expected := mlx.Concatenate(want, 0)
				if err := compareRowArrays(got, expected); err != nil {
					t.Fatal(err)
				}
				if err := compareRowArrays(view, expected); err != nil {
					t.Fatal(err)
				}
				for row := range owners {
					for i, state := range owners[row].State() {
						if err := compareRowArrays(state, refs[row].State()[i]); err != nil {
							t.Fatalf("row %d state %d: %v", row, i, err)
						}
					}
				}
			}
		})
	}
}

func TestBatchRecurrentOwnersAndSnapshots(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		a, b := NewRecurrentCache(2, 3, 1, 2, 2), NewRecurrentCache(2, 3, 1, 2, 2)
		defer a.Free()
		defer b.Free()
		for i, c := range []*RecurrentCache{a, b} {
			rb := newKVBatch(0, i+1)
			h := c.Get(rb, mlx.DTypeFloat32)
			c.Put(rb, []*mlx.Array{mlx.AddScalar(h.ConvState(), float32(i+1))},
				[]*mlx.Array{mlx.AddScalar(h.DeltaState(), float32(10+i))})
			c.PrepareSnapshots([]int{c.Offset() + 1})
		}
		layers, err := Batch([][]Cache{{b}, {a}})
		if err != nil {
			t.Fatal(err)
		}
		rb := &batch.Batch{
			InputIDs: mlx.Zeros(mlx.DTypeInt32, 2, 1), SeqOffsets: []int32{2, 1}, SeqQueryLens: []int32{1, 1},
		}
		c := layers[0].(Recurrent)
		h := c.Get(rb, mlx.DTypeFloat32)
		if err := compareRowArrays(h.ConvState(), mlx.Concatenate([]*mlx.Array{b.State()[0], a.State()[0]}, 0)); err != nil {
			t.Fatal(err)
		}
		c.Put(rb, []*mlx.Array{mlx.AddScalar(h.ConvState(), 3)}, []*mlx.Array{mlx.AddScalar(h.DeltaState(), 7)})
		if a.Offset() != 2 || b.Offset() != 3 {
			t.Fatalf("offsets = %d, %d, want 2, 3", a.Offset(), b.Offset())
		}
		mlx.Eval(c.State()...)
		sa, sb := a.TakeSnapshots()[0], b.TakeSnapshots()[0]
		defer sa.Close()
		defer sb.Close()
		// Keep b parked while a advances and restores its own boundary.
		before := b.State()[1]
		ah := a.Get(newKVBatch(2, 1), mlx.DTypeFloat32)
		a.Put(newKVBatch(2, 1), []*mlx.Array{mlx.AddScalar(ah.ConvState(), 100)},
			[]*mlx.Array{mlx.AddScalar(ah.DeltaState(), 100)})
		if !a.Restore(sa, 2) {
			t.Fatal("cannot restore row a's captured boundary")
		}
		if err := compareRowArrays(b.State()[1], before); err != nil {
			t.Fatal(err)
		}
		if err := compareRowArrays(a.State()[0], mlx.AddScalar(mlx.Zeros(mlx.DTypeFloat32, 1, 2, 3), 4)); err != nil {
			t.Fatal(err)
		}
		if err := compareRowArrays(a.State()[1], mlx.AddScalar(mlx.Zeros(mlx.DTypeFloat32, 1, 1, 2, 2), 17)); err != nil {
			t.Fatal(err)
		}
		if b.Offset() != 3 {
			t.Fatal("restoring a changed b's offset")
		}
	})
}

func TestBatchRejectsIncompatibleCaches(t *testing.T) {
	if Layers(nil) != nil {
		t.Fatal("no-cache forwards must retain a nil cache bundle")
	}
	mlxtest.Run(t, func(t *mlxtest.T) {
		a, b := NewKVCache(), NewRotatingKVCache(4)
		defer a.Free()
		defer b.Free()
		for _, rows := range [][][]Cache{nil, {{a}, {}}, {{a}, {b}}, {{a}, {a}}, {{nil}, {a}}} {
			if _, err := Batch(rows); err == nil {
				t.Fatalf("accepted incompatible rows %v", rows)
			}
		}
		layers, err := Batch([][]Cache{{a}})
		if err != nil || len(layers) != 1 || layers[0] != a {
			t.Fatalf("single row changed its original owner: %v, %v", layers, err)
		}
	})
}
