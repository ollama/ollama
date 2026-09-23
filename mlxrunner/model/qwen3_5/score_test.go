package qwen3_5

import (
	"math"
	"testing"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlx/mlxtest"
	"github.com/ollama/ollama/mlxrunner/nn"
)

func TestUnembedCandidates(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		mlx.Scoped(func() {
			values := make([]float32, 7*8)
			for i := range values {
				values[i] = float32(math.Sin(float64(i)))
			}
			weight := mlx.FromValues(values, 7, 8).AsType(mlx.DTypeBFloat16)
			hidden := mlx.FromValues([]float32{0.7, 1.3, -2.9, 3.1, 0.06, 1.01, -5.03, 2.17}, 1, 1, 8).AsType(mlx.DTypeBFloat16)
			ids := mlx.FromValues([]int32{6, 2, 0}, 3)
			bias := mlx.FromValues([]float32{0.1, 1.2, 2.3, 3.4, 4.5, 5.6, 6.7}, 7).AsType(mlx.DTypeBFloat16)
			for _, b := range []*mlx.Array{nil, bias} {
				m := &Model{LMHead: nn.NewLinear(weight, b)}
				got := m.UnembedCandidates(hidden, ids)
				full := hidden.AsType(mlx.DTypeFloat32).Matmul(weight.AsType(mlx.DTypeFloat32).Transpose(1, 0)).Reshape(-1)
				if b != nil {
					full = full.Add(b.AsType(mlx.DTypeFloat32))
				}
				want := full.TakeAxis(ids, 0)
				mlx.Eval(got, want)
				if got.DType() != mlx.DTypeFloat32 || got.Dim(0) != 3 {
					t.Fatal("candidate logits must retain FP32 precision")
				}
				actual, expected := got.Floats(), want.Floats()
				for i, v := range actual {
					if math.Abs(float64(v-expected[i])) > 1e-6 {
						t.Fatalf("candidate %d: selected=%g full=%g", i, v, expected[i])
					}
				}
			}
		})
	})
}
