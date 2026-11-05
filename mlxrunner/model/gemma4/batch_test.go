package gemma4

import (
	"math"
	"testing"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlx/mlxtest"
	"github.com/ollama/ollama/mlxrunner/batch"
	"github.com/ollama/ollama/mlxrunner/cache"
	"github.com/ollama/ollama/mlxrunner/nn"
)

func TestAttentionIndependentCacheRows(t *testing.T) {
	for _, sliding := range []bool{false, true} {
		name := "full-k-equals-v"
		if sliding {
			name = "rotating"
		}
		mlxtest.RunSubtest(t, name, func(t *mlxtest.T) {
			cfg := tinyMoEConfig()
			cfg.FullRopeDims, cfg.SlidingRopeDims = 8, 8
			cfg.FullRopeBase, cfg.SlidingRopeBase = 10000, 10000
			cfg.SlidingWindow = 4
			weight := func(rows, cols int) *mlx.Array {
				values := make([]float32, rows*cols)
				for i := range values {
					values[i] = float32((i*13)%31-15) * .02
				}
				return mlx.FromValues(values, rows, cols).AsType(mlx.DTypeBFloat16)
			}
			attn := &Attention{
				QProj:       nn.NewLinear(weight(16, 16), nil),
				KProj:       nn.NewLinear(weight(8, 16), nil),
				OProj:       nn.NewLinear(weight(16, 16), nil),
				QNormScaled: mlx.AddScalar(mlx.Zeros(mlx.DTypeBFloat16, 8), 1),
				KNormScaled: mlx.AddScalar(mlx.Zeros(mlx.DTypeBFloat16, 8), 1),
			}
			if sliding {
				attn.VProj = nn.NewLinear(weight(8, 16), nil)
			}
			newCache := func() cache.Cache {
				if sliding {
					return cache.NewRotatingKVCache(4)
				}
				return cache.NewKVCache()
			}
			owners, refs := []cache.Cache{newCache(), newCache()}, []cache.Cache{newCache(), newCache()}
			for i := range owners {
				defer owners[i].Free()
				defer refs[i].Free()
			}
			run := func(rows []int, caches []cache.Cache) (*mlx.Array, *mlx.Array) {
				b := &batch.Batch{InputIDs: mlx.Zeros(mlx.DTypeInt32, len(rows), 1)}
				bundles := make([][]cache.Cache, len(rows))
				var values []float32
				for i, row := range rows {
					pos := caches[row].Offset()
					b.SeqOffsets = append(b.SeqOffsets, int32(pos))
					b.SeqQueryLens = append(b.SeqQueryLens, 1)
					bundles[i] = []cache.Cache{caches[row]}
					for j := range 16 {
						values = append(values, float32((row*11+pos*3+j*7)%29-14)*.04)
					}
				}
				layers, err := cache.Batch(bundles)
				if err != nil {
					t.Fatal(err)
				}
				x := mlx.FromValues(values, len(rows), 1, 16).AsType(mlx.DTypeBFloat16)
				positions := mlx.FromValues(b.SeqOffsets, len(rows))
				out, donor := attn.Forward(x, b, layers[0], positions, int32(len(rows)), 1, sliding, cfg, nil, nn.CausalMask())
				shared, _ := attn.Forward(mlx.MulScalar(x, .7), b, nil, positions, int32(len(rows)), 1, sliding, cfg, donor, nn.CausalMask())
				return out, shared
			}
			for row, n := range []int{1, 7} {
				for range n {
					a, b := run([]int{row}, owners)
					c, d := run([]int{row}, refs)
					mlx.Eval(a, b, c, d)
				}
			}
			for _, rows := range [][]int{{0, 1}, {1, 0}, {0}, {1, 0}} {
				got, shared := run(rows, owners)
				var want, wantShared []*mlx.Array
				for _, row := range rows {
					a, b := run([]int{row}, refs)
					want, wantShared = append(want, a), append(wantShared, b)
					if owners[row].Offset() != refs[row].Offset() {
						t.Fatalf("shared history advanced row %d twice", row)
					}
				}
				for i, actual := range []*mlx.Array{got, shared} {
					actual = actual.AsType(mlx.DTypeFloat32)
					expected := mlx.Concatenate([][]*mlx.Array{want, wantShared}[i], 0).AsType(mlx.DTypeFloat32)
					mlx.Eval(actual, expected)
					a, b := actual.Floats(), expected.Floats()
					for j := range b {
						// B=1 and B>1 projections select different matrix
						// kernels; compare with BF16 arithmetic tolerance.
						tolerance := .02 * math.Max(math.Abs(float64(b[j])), 1)
						if math.IsNaN(float64(a[j])) || math.Abs(float64(a[j]-b[j])) > tolerance {
							t.Fatalf("rows %v donor=%v element %d = %g, want %g", rows, i == 1, j, a[j], b[j])
						}
					}
				}
			}
		})
	}
}
