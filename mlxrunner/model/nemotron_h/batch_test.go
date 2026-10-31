package nemotron_h

import (
	"math"
	"testing"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlx/mlxtest"
	"github.com/ollama/ollama/mlxrunner/batch"
	"github.com/ollama/ollama/mlxrunner/cache"
	"github.com/ollama/ollama/mlxrunner/nn"
)

func TestMambaIndependentCacheRows(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		cfg := &Config{HiddenSize: 8, MambaNumHeads: 2, MambaHeadDim: 4, SSMStateSize: 2, NGroups: 1, ConvKernel: 3, LayerNormEpsilon: 1e-6}
		weight := func(rows, cols int) *mlx.Array {
			values := make([]float32, rows*cols)
			for i := range values {
				values[i] = float32((i*7)%19-9) * .03
			}
			return mlx.FromValues(values, rows, cols).AsType(mlx.DTypeBFloat16)
		}
		m := &Mamba2{InProj: nn.NewLinear(weight(22, 8), nil), OutProj: nn.NewLinear(weight(8, 8), nil), Conv1D: nn.NewConv1d(weight(12, 3).AsType(mlx.DTypeFloat32).ExpandDims(2), nil, 1, 0, 1, 12), A: mlx.FromValues([]float32{-.3, -.4}, 2), D: mlx.FromValues([]float32{1, 1}, 2), DtBias: mlx.Zeros(mlx.DTypeFloat32, 2), NormWeight: mlx.AddScalar(mlx.Zeros(mlx.DTypeBFloat16, 8), 1)}
		owners, refs := make([]cache.Cache, 2), make([]cache.Cache, 2)
		for i := range owners {
			owners[i] = cache.NewRecurrentCache(2, 12, 2, 4, 2)
			refs[i] = cache.NewRecurrentCache(2, 12, 2, 4, 2)
			defer owners[i].Free()
			defer refs[i].Free()
		}
		run := func(rows []int, caches []cache.Cache) *mlx.Array {
			b := &batch.Batch{InputIDs: mlx.Zeros(mlx.DTypeInt32, len(rows), 1)}
			bundles := make([][]cache.Cache, len(rows))
			var values []float32
			for i, row := range rows {
				pos := caches[row].Offset()
				b.SeqOffsets = append(b.SeqOffsets, int32(pos))
				b.SeqQueryLens = append(b.SeqQueryLens, 1)
				bundles[i] = []cache.Cache{caches[row]}
				for j := range 8 {
					values = append(values, float32((row*13+pos*7+j*3)%23-11)*.05)
				}
			}
			layers, err := cache.Batch(bundles)
			if err != nil {
				t.Fatal(err)
			}
			return m.Forward(mlx.FromValues(values, len(rows), 1, 8).AsType(mlx.DTypeBFloat16), b, layers[0], int32(len(rows)), 1, cfg)
		}
		for row, n := range []int{1, 5} {
			for range n {
				mlx.Eval(run([]int{row}, owners), run([]int{row}, refs))
			}
		}
		for _, rows := range [][]int{{0, 1}, {1, 0}, {1}, {0, 1}} {
			got := run(rows, owners).AsType(mlx.DTypeFloat32)
			var expected []*mlx.Array
			for _, row := range rows {
				expected = append(expected, run([]int{row}, refs))
			}
			want := mlx.Concatenate(expected, 0).AsType(mlx.DTypeFloat32)
			actuals, expectations := []*mlx.Array{got}, []*mlx.Array{want}
			for _, row := range rows {
				actuals = append(actuals, owners[row].State()...)
				expectations = append(expectations, refs[row].State()...)
			}
			for i, a := range actuals {
				a = a.AsType(mlx.DTypeFloat32)
				e := expectations[i].AsType(mlx.DTypeFloat32)
				mlx.Eval(a, e)
				av, ev := a.Floats(), e.Floats()
				for j, v := range ev {
					if d := math.Abs(float64(av[j] - v)); math.IsNaN(d) || d > .004 {
						t.Fatalf("rows %v array %d element %d got %g want %g", rows, i, j, av[j], v)
					}
				}
			}
		}
	})
}
