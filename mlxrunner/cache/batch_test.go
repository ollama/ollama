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
	if len(g) == 0 || len(g) != len(w) {
		return fmt.Errorf("value count = %d, want %d", len(g), len(w))
	}
	for i := range w {
		if delta := math.Abs(float64(g[i] - w[i])); math.IsNaN(delta) || delta > 1e-5 {
			return fmt.Errorf("element %d = %g, want %g", i, g[i], w[i])
		}
	}
	return nil
}

func TestCompareRowArraysRejectsInvalidComparisons(t *testing.T) {
	for _, tt := range []struct {
		name    string
		got     []float32
		want    []float32
		wantErr bool
	}{
		{name: "valid", got: []float32{1, 2}, want: []float32{1, 2}},
		{name: "actual nan", got: []float32{float32(math.NaN())}, want: []float32{0}, wantErr: true},
		{name: "expected nan", got: []float32{0}, want: []float32{float32(math.NaN())}, wantErr: true},
		{name: "matching infinity", got: []float32{float32(math.Inf(1))}, want: []float32{float32(math.Inf(1))}, wantErr: true},
		{name: "empty", wantErr: true},
		{name: "shape mismatch", got: []float32{1}, want: []float32{1, 2}, wantErr: true},
	} {
		mlxtest.RunSubtest(t, tt.name, func(t *mlxtest.T) {
			array := func(values []float32) *mlx.Array {
				if len(values) == 0 {
					return mlx.Zeros(mlx.DTypeFloat32, 0)
				}
				return mlx.FromValues(values, len(values))
			}
			err := compareRowArrays(array(tt.got), array(tt.want))
			if (err != nil) != tt.wantErr {
				t.Fatalf("comparison error = %v, want error %t", err, tt.wantErr)
			}
		})
	}
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

func TestBatchPrefillAttention(t *testing.T) {
	for _, tt := range []struct {
		name     string
		rotating bool
		seeded   bool
	}{
		{name: "full/empty"},
		{name: "full/populated", seeded: true},
		{name: "rotating/empty", rotating: true},
		{name: "rotating/populated", rotating: true, seeded: true},
	} {
		mlxtest.RunSubtest(t, tt.name, func(t *mlxtest.T) {
			var owners, refs [2]Cache
			for i := range owners {
				if tt.rotating {
					owners[i], refs[i] = NewRotatingKVCache(4), NewRotatingKVCache(4)
				} else {
					owners[i], refs[i] = NewKVCache(), NewKVCache()
				}
				defer owners[i].Free()
				defer refs[i].Free()
				if tt.seeded {
					seed := mlx.AddScalar(mlx.Zeros(mlx.DTypeFloat32, 1, 1, i+2, 2), float32(i+1))
					for _, c := range []Cache{owners[i], refs[i]} {
						c.(Attention).Update(newKVBatch(0, i+2), seed, seed)
					}
				}
			}
			for _, n := range []int{3, 1, 5} {
				b := &batch.Batch{InputIDs: mlx.Zeros(mlx.DTypeInt32, 2, n), SeqQueryLens: []int32{int32(n), int32(n)}}
				var keys, wants []*mlx.Array
				for i := range owners {
					pos := owners[i].Offset()
					b.SeqOffsets = append(b.SeqOffsets, int32(pos))
					k := mlx.AddScalar(mlx.Arange(0, float64(n*2), 1, mlx.DTypeFloat32).Reshape(1, 1, n, 2), float32(i+1))
					keys = append(keys, k)
					rb := newKVBatch(pos, n)
					h := refs[i].(Attention).Update(rb, k, k)
					wants = append(wants, nn.ScaledDotProductAttention(rb, k, .5, nn.WithKVHistory(h), nn.WithMask(nn.CausalMask())))
				}
				layers, err := Batch([][]Cache{{owners[0]}, {owners[1]}})
				if err != nil {
					t.Fatal(err)
				}
				k := mlx.Concatenate(keys, 0)
				h := layers[0].(Attention).Update(b, k, k)
				got := nn.ScaledDotProductAttention(b, k, .5, nn.WithKVHistory(h), nn.WithMask(nn.CausalMask()))
				if err := compareRowArrays(got, mlx.Concatenate(wants, 0)); err != nil {
					t.Fatalf("n=%d: %v", n, err)
				}
				for i := range owners {
					if owners[i].Offset() != refs[i].Offset() {
						t.Fatal("cache position mismatch")
					}
				}
			}
		})
	}
}

func TestBatchRecurrentIndependentSplits(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		a, b := NewRecurrentCache(1, 1, 1, 1, 1), NewRecurrentCache(1, 1, 1, 1, 1)
		defer a.Free()
		defer b.Free()
		a.Get(newKVBatch(0, 1), mlx.DTypeFloat32)
		b.Get(newKVBatch(0, 1), mlx.DTypeFloat32)
		a.PrepareSnapshots([]int{1, 4})
		b.PrepareSnapshots([]int{2, 3, 4})
		layers, err := Batch([][]Cache{{a}, {b}})
		if err != nil {
			t.Fatal(err)
		}
		c := layers[0].(Recurrent)
		if got := c.SnapshotSplits(4); !slices.Equal(got, []int{1, 2, 3}) {
			t.Fatalf("splits %v", got)
		}
		var conv, delta []*mlx.Array
		for _, boundary := range []float32{1, 2, 3, 4} {
			conv = append(conv, mlx.FromValues([]float32{boundary, 10 + boundary}, 2, 1, 1))
			delta = append(delta, mlx.FromValues([]float32{100 + boundary, 200 + boundary}, 2, 1, 1, 1))
		}
		c.Put(&batch.Batch{InputIDs: mlx.Zeros(mlx.DTypeInt32, 2, 4), SeqOffsets: []int32{0, 0}, SeqQueryLens: []int32{4, 4}}, conv, delta)
		mlx.Eval(c.State()...)
		for row, owner := range []*RecurrentCache{a, b} {
			boundaries := [][]int{{1, 4}, {2, 3, 4}}[row]
			snaps := owner.TakeSnapshots()
			if len(snaps) != len(boundaries) {
				t.Fatalf("row %d snapshots %d", row, len(snaps))
			}
			for i, snap := range snaps {
				if !owner.Restore(snap, boundaries[i]) {
					t.Fatal("restore failed")
				}
				want := float32(100*(row+1) + boundaries[i])
				if got := owner.State()[1].Floats()[0]; got != want {
					t.Fatalf("row %d boundary %d: %g, want %g", row, boundaries[i], got, want)
				}
				snap.Close()
			}
		}
	})
}

// A snapshot of one request must not retain the allocations for its batch peers.
func TestBatchAttentionSnapshotOwnsRow(t *testing.T) {
	for _, kind := range []string{"full", "rotating"} {
		for _, scheduled := range []bool{false, true} {
			mlxtest.RunSubtest(t, fmt.Sprintf("%s/scheduled=%t", kind, scheduled), func(t *mlxtest.T) {
				const rows, heads, tokens, dims, keep = 8, 4, 256, 64, 3
				const rowElements = heads * tokens * dims
				const rowBytes = 2 * rowElements * 4
				newCache := func() Cache {
					if kind == "rotating" {
						return NewRotatingKVCache(512)
					}
					return NewKVCache()
				}
				mlx.DefaultStream().Synchronize()
				baseline := settledActiveMemory()
				var snapshot Snapshot
				defer func() {
					if snapshot != nil {
						snapshot.Close()
					}
				}()
				mlx.Scoped(func() {
					owners := make([][]Cache, rows)
					b := &batch.Batch{
						InputIDs:   mlx.Zeros(mlx.DTypeInt32, rows, tokens),
						SeqOffsets: make([]int32, rows), SeqQueryLens: make([]int32, rows),
					}
					data := make([]float32, rows*rowElements)
					for i := range rows {
						owners[i] = []Cache{newCache()}
						defer owners[i][0].Free()
						b.SeqQueryLens[i] = tokens
						for j := range rowElements {
							data[i*rowElements+j] = float32(i + 1)
						}
					}
					if scheduled {
						owners[keep][0].PrepareSnapshots([]int{tokens})
					}
					layers, err := Batch(owners)
					if err != nil {
						t.Fatal(err)
					}
					keys := mlx.FromValues(data, rows, heads, tokens, dims)
					values := mlx.Neg(keys)
					layers[0].(Attention).Update(b, keys, values)
					mlx.Eval(layers[0].State()...)
					if scheduled {
						snapshot = owners[keep][0].TakeSnapshots()[0]
					} else {
						snapshot = owners[keep][0].Snapshot(0)
					}
				})
				mlx.DefaultStream().Synchronize()
				if got := snapshot.Size(); got != rowBytes {
					t.Fatalf("snapshot accounts for %d bytes, want one row (%d)", got, rowBytes)
				}
				// Allow allocator rounding, but not another row's allocation.
				if retained := settledActiveMemory() - baseline; retained > rowBytes+64*1024 {
					t.Fatalf("one %d-byte snapshot retains %d bytes after freeing all batch owners", rowBytes, retained)
				}
				restored := newCache()
				defer restored.Free()
				if !restored.Restore(snapshot, tokens) {
					t.Fatal("could not restore the surviving row")
				}
				for i, state := range restored.State() {
					mlx.Eval(state)
					want := float32(keep + 1)
					if i == 1 {
						want = -want
					}
					matches := state.Equal(mlx.FromValue(want)).AsType(mlx.DTypeInt32).Reshape(-1)
					if got := mlx.Sum(matches, 0, false).Int(); got != int32(state.Size()) {
						t.Fatalf("restored state %d has %d matching elements, want %d", i, got, state.Size())
					}
				}
			})
		}
	}
}
