package inkling2

import (
	"testing"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlx/mlxtest"
	"github.com/ollama/ollama/mlxrunner/nn"
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
			gate := messageGate{
				Down:       nn.NewLinear(values(rank, 2*width, .018, mlx.DTypeBFloat16), nil),
				GainForget: nn.NewLinear(values(2*width, rank, .021, mlx.DTypeFloat32), nil),
			}
			for _, sh := range [][2]int{{1, 4}, {2, 8}, {4, 8}, {1, 257}} {
				mlx.Scoped(func() {
					b, n := sh[0], sh[1]
					x := values(b*n, width, .0111, mlx.DTypeFloat32).Reshape(b, n, width)
					m := values(b*n, width, .0379, mlx.DTypeFloat32).Reshape(b, n, width)
					// The kernel computes the gain and forget products in FP32 and can
					// round the compressed BF16 projection differently.
					got, want := gate.apply(x, m), gate.applyGraph(x, m)
					diff := mlx.Sub(got, want).Abs().Reshape(-1).MaxAxis(0, false).Float()
					scale := want.Abs().Reshape(-1).MaxAxis(0, false).Float()
					if diff > scale/128 {
						t.Fatalf("B=%d T=%d: largest difference %g for values up to %g", b, n, diff, scale)
					}
				})
			}
		})
	})
}
