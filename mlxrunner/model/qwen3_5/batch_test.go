package qwen3_5

import (
	"math"
	"testing"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlx/mlxtest"
	"github.com/ollama/ollama/mlxrunner/batch"
	"github.com/ollama/ollama/mlxrunner/cache"
	"github.com/ollama/ollama/mlxrunner/nn"
)

func TestGatedDeltaIndependentCacheRows(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		cfg := gdnTestConfig()
		cfg.HiddenSize, cfg.LinearConvKernelDim, cfg.RMSNormEps = 8, 3, 1e-6
		const packed, value = 32, 16
		g := &GatedDeltaNet{
			InProjQKVZ: nn.NewLinear(patternArray(packed+value, 8), nil),
			InProjBA:   nn.NewLinear(patternArray(8, 8), nil),
			OutProj:    nn.NewLinear(patternArray(8, value), nil),
			Conv1D:     nn.NewConv1d(patternArray(packed, 3).ExpandDims(2), nil, 1, 0, 1, packed),
			NormWeight: mlx.AddScalar(mlx.Zeros(mlx.DTypeBFloat16, 4), 1),
			DtBias:     mlx.Zeros(mlx.DTypeFloat32, 4),
			AExp:       mlx.AddScalar(mlx.Zeros(mlx.DTypeFloat32, 4), .2),
		}
		owners, refs := make([]cache.Cache, 3), make([]cache.Cache, 3)
		for i := range owners {
			owners[i] = cache.NewRecurrentCache(2, packed, 4, 4, 4)
			refs[i] = cache.NewRecurrentCache(2, packed, 4, 4, 4)
			defer owners[i].Free()
			defer refs[i].Free()
		}
		input := func(row, pos int) *mlx.Array {
			values := make([]float32, 8)
			for j := range values {
				values[j] = float32((row*7+pos*3+j*11)%23-11) * .03
			}
			return mlx.FromValues(values, 1, 1, 8).AsType(mlx.DTypeBFloat16)
		}
		run := func(rows []int, caches []cache.Cache) *mlx.Array {
			b := &batch.Batch{InputIDs: mlx.Zeros(mlx.DTypeInt32, len(rows), 1)}
			var xs []*mlx.Array
			bundles := make([][]cache.Cache, len(rows))
			for i, row := range rows {
				b.SeqOffsets = append(b.SeqOffsets, int32(caches[row].Offset()))
				b.SeqQueryLens = append(b.SeqQueryLens, 1)
				xs = append(xs, input(row, caches[row].Offset()))
				bundles[i] = []cache.Cache{caches[row]}
			}
			layers, err := cache.Batch(bundles)
			if err != nil {
				t.Fatal(err)
			}
			return g.Forward(mlx.Concatenate(xs, 0), b, layers[0], int32(len(rows)), 1, cfg)
		}
		// Different prefix lengths followed by changing active row sets.
		for row, n := range []int{1, 4, 2} {
			for range n {
				mlx.Eval(run([]int{row}, owners), run([]int{row}, refs))
			}
		}
		for _, rows := range [][]int{{0, 1}, {1, 0}, {2, 1}, {0}, {2, 0}} {
			got := run(rows, owners).AsType(mlx.DTypeFloat32)
			var single []*mlx.Array
			for _, row := range rows {
				single = append(single, run([]int{row}, refs))
			}
			want := mlx.Concatenate(single, 0).AsType(mlx.DTypeFloat32)
			mlx.Eval(got, want)
			gv, wv := got.Floats(), want.Floats()
			for j := range wv {
				if math.IsNaN(float64(gv[j])) || math.Abs(float64(gv[j]-wv[j])) > .004 {
					t.Fatalf("rows %v element %d = %g, want %g", rows, j, gv[j], wv[j])
				}
			}
			for _, row := range rows {
				if owners[row].Offset() != refs[row].Offset() {
					t.Fatalf("row %d lost recurrent state", row)
				}
				for j, state := range owners[row].State() {
					a, b := state.AsType(mlx.DTypeFloat32), refs[row].State()[j].AsType(mlx.DTypeFloat32)
					mlx.Eval(a, b)
					av, bv := a.Floats(), b.Floats()
					for k := range bv {
						if math.IsNaN(float64(av[k])) || math.Abs(float64(av[k]-bv[k])) > .004 {
							t.Fatalf("row %d state %d element %d = %g, want %g", row, j, k, av[k], bv[k])
						}
					}
				}
			}
		}
	})
}
