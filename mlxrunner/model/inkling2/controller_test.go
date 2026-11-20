package inkling2

import (
	"testing"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlx/mlxtest"
	"github.com/ollama/ollama/mlxrunner/nn"
)

func TestMetalLoopControllerMatchesSeparateProjections(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		if !mlx.SupportsPackedController() {
			t.Skip("requires qualified Metal controller")
		}
		mlx.Scoped(func() {
			const width, rank = 2304, 64
			values := func(x, y int, scale float32, dtype mlx.DType) *mlx.Array {
				a := mlx.Arange(0, float64(x*y), 1, mlx.DTypeFloat32)
				return mlx.MulScalar(mlx.Sin(mlx.MulScalar(a, scale)), .07).AsType(dtype).Reshape(x, y)
			}
			down := nn.NewLinear(values(rank, 2*width, .018, mlx.DTypeBFloat16), nil)
			gain := nn.NewLinear(values(width, rank, .021, mlx.DTypeFloat32), nil)
			forget := nn.NewLinear(values(width, rank, -.029, mlx.DTypeFloat32), nil)
			ordinary := messageGate{Down: down, Gain: gain, Forget: forget}
			packed := messageGate{Down: down, GainForget: nn.NewLinear(mlx.Concatenate([]*mlx.Array{gain.Weight, forget.Weight}, 0), nil)}
			for _, sh := range [][2]int{{1, 4}, {2, 8}, {4, 8}, {1, 257}} {
				mlx.Scoped(func() {
					b, n := sh[0], sh[1]
					x := values(b*n, width, .0111, mlx.DTypeFloat32).Reshape(b, n, width)
					m := values(b*n, width, .0379, mlx.DTypeFloat32).Reshape(b, n, width)
					want := ordinary.apply(x, m)
					got := packed.apply(x, m)
					equal := got.View(mlx.DTypeUint32).Equal(want.View(mlx.DTypeUint32)).AsType(mlx.DTypeInt32).Reshape(-1).SumAxis(0, false).Int()
					t.Logf("B=%d T=%d exact=%d/%d", b, n, equal, got.Size())
					if int(equal) != got.Size() {
						t.Fatalf("B=%d T=%d: %d differing FP32 values", b, n, got.Size()-int(equal))
					}
				})
			}
		})
	})
}
