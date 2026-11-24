package inkling2

import (
	"fmt"
	"math/rand/v2"
	"testing"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlx/mlxtest"
)

func TestMetalLoopControllerMatchesGraph(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		if !mlx.MetalIsAvailable() {
			t.Skip("requires the Metal controller kernel")
		}
		mlx.Scoped(func() {
			const width, rank = 2304, 64
			values := func(x, y int, scale float32, dtype mlx.DType) *mlx.Array {
				a := mlx.Arange(0, float64(x*y), 1, mlx.DTypeFloat32)
				return mlx.MulScalar(mlx.Sin(mlx.MulScalar(a, scale)), .07).AsType(dtype).Reshape(x, y)
			}
			down := values(4*rank, 2*width, .018, mlx.DTypeBFloat16).Reshape(4, rank, 2*width)
			gates := values(4*rank, 2*width, .021, mlx.DTypeFloat32).Reshape(4, rank, 2*width)
			controllers := newLoopControllers(down, gates)
			// Short forwards (up to 8 tokens) scale the FP32 down products by the
			// inverse RMS instead of rounding the normalized inputs to BF16 first,
			// so they differ from the graph by a few BF16 roundings of the
			// compressed values. Longer forwards round like the graph.
			for _, sh := range [][2]int{{1, 1}, {1, 4}, {2, 8}, {3, 8}, {4, 8}, {1, 257}} {
				mlx.Scoped(func() {
					b, n := sh[0], sh[1]
					x := values(b*n, width, .0111, mlx.DTypeFloat32).Reshape(b, n, width)
					m := values(b*n*4, width, .0379, mlx.DTypeFloat32).Reshape(b, n, 4, width)
					for index := range 4 {
						gate := controllers.gate(index)
						got, want := gate.apply(x, m, index), gate.applyGraph(x, selectMessage(m, index))
						diff := mlx.Sub(got, want).Abs().Reshape(-1).MaxAxis(0, false).Float()
						scale := want.Abs().Reshape(-1).MaxAxis(0, false).Float()
						tolerance := scale / 128
						if n <= shortTargetRows {
							tolerance = scale / 64
						}
						if !(diff <= tolerance) {
							t.Fatalf("B=%d T=%d controller %d: largest difference %g for values up to %g", b, n, index, diff, scale)
						}
					}
				})
			}
		})
	})
}

// applyGraph, the controller path off Metal, gives a row the bits it gets in a
// forward of its own, so decode and verification agree on each position.
func TestLoopControllerGraphRowsMatchAlone(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		if !mlx.MetalIsAvailable() && !mlx.CUDAIsAvailable() {
			t.Skip("CPU products are not row-stable across batch shapes")
		}
		mlx.Scoped(func() {
			const width, rank, requests = 2304, 64, 2
			rng := rand.New(rand.NewPCG(61, 67))
			controllers := newLoopControllers(draftTestValues(rng, 0.02, 4, rank, 2*width), draftTestValues(rng, 0.05, 4, rank, 2*width).AsType(mlx.DTypeFloat32))
			gate := controllers.gate(0)
			x := draftTestValues(rng, 1, requests, shortTargetRows, width).AsType(mlx.DTypeFloat32)
			m := draftTestValues(rng, 1, requests, shortTargetRows, width).AsType(mlx.DTypeFloat32)
			joined := gate.applyGraph(x, m)
			for i := range requests {
				for j := range shortTargetRows {
					row := func(a *mlx.Array) *mlx.Array { return a.Slice(mlx.Slice(i, i+1), mlx.Slice(j, j+1), mlx.Slice()) }
					requireIdentical(t, fmt.Sprintf("request %d row %d", i, j), gate.applyGraph(row(x), row(m)), row(joined))
				}
			}
		})
	})
}
