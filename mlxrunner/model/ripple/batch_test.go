package ripple

import (
	"testing"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlx/mlxtest"
	"github.com/ollama/ollama/mlxrunner/batch"
	"github.com/ollama/ollama/mlxrunner/cache"
)

func TestIndependentCacheRows(t *testing.T) {
	m, reference := dflashTargetFixture(t)
	mlxtest.Run(t, func(t *mlxtest.T) {
		if err := m.LoadWeights(fixtureWeights(reference, "bfloat16")); err != nil {
			t.Fatal(err)
		}
		m.SetAuxHiddenLayers([]int{0, 2})
		owners, single := make([][]cache.Cache, 3), make([][]cache.Cache, 3)
		for i := range owners {
			owners[i], single[i] = m.NewCaches(), m.NewCaches()
			for _, c := range append(append([]cache.Cache(nil), owners[i]...), single[i]...) {
				defer c.Free()
			}
		}
		run := func(rows []int, n int, caches [][]cache.Cache) (*mlx.Array, *mlx.Array) {
			b := &batch.Batch{}
			ids := make([]int32, len(rows)*n)
			bundles := make([][]cache.Cache, len(rows))
			for i, row := range rows {
				offset := caches[row][0].Offset()
				b.SeqOffsets = append(b.SeqOffsets, int32(offset))
				b.SeqQueryLens = append(b.SeqQueryLens, int32(n))
				for j := range n {
					ids[i*n+j] = int32((row*3+offset+j)%8 + 1)
				}
				bundles[i] = caches[row]
			}
			b.InputIDs = mlx.FromValues(ids, len(rows), n)
			layers, err := cache.Batch(bundles)
			if err != nil {
				t.Fatal(err)
			}
			hidden, aux := m.Forward(b, layers)
			mlx.Eval(hidden, aux)
			return hidden, aux
		}
		// Different prefixes cross the fixture's three-token sliding window.
		for row, n := range []int{1, 4, 6} {
			run([]int{row}, n, owners)
			run([]int{row}, n, single)
		}
		for _, step := range []struct {
			rows []int
			n    int
		}{
			{[]int{0, 1, 2}, 1}, {[]int{2, 0}, 2}, {[]int{1, 2}, 1}, {[]int{0}, 1}, {[]int{2, 1, 0}, 1},
		} {
			got, aux := run(step.rows, step.n, owners)
			var want, wantAux []*mlx.Array
			for _, row := range step.rows {
				h, a := run([]int{row}, step.n, single)
				want, wantAux = append(want, h), append(wantAux, a)
			}
			// BF16 projections may round differently with a different row count.
			checkTensor(t, "batched hidden", got, referenceTensor{Data: mlx.Concatenate(want, 0).AsType(mlx.DTypeFloat32).Floats(), Shape: got.Dims()}, 3e-3, 3e-3)
			checkTensor(t, "batched draft taps", aux, referenceTensor{Data: mlx.Concatenate(wantAux, 0).AsType(mlx.DTypeFloat32).Floats(), Shape: aux.Dims()}, 3e-3, 3e-3)
			for _, row := range step.rows {
				for i, c := range owners[row] {
					if c.Offset() != single[row][i].Offset() {
						t.Fatalf("row %d cache %d offset %d, want %d", row, i, c.Offset(), single[row][i].Offset())
					}
				}
			}
		}
	})
}
