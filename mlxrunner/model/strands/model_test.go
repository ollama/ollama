package strands

import (
	"math"
	"testing"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlx/mlxtest"
	"github.com/ollama/ollama/mlxrunner/nn"
)

func TestPointerHead(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		mlx.Scoped(func() {
			var m *Model
			mlx.ScopedArrays(func() []*mlx.Array {
				identity := mlx.FromValues([]float32{1, 0, 0, 1}, 2, 2)
				m = &Model{
					HeadNorm: &nn.LayerNorm{Weight: mlx.FromValues([]float32{1, 1}, 2), Bias: mlx.FromValues([]float32{0, 0}, 2), Eps: 1e-5},
					Query:    nn.NewLinear(identity, nil), Key: nn.NewLinear(identity, nil),
					config: config{PointerDim: 2, Temperature: 1, TemperatureByKind: map[string]float32{"choice": 2}},
				}
				return mlx.Collect(m)
			})
			// Two independent rows have opposite queries and the same ordered options.
			query := mlx.FromValues([]float32{1, -1, -1, 1}, 2, 1, 2).AsType(mlx.DTypeBFloat16)
			options := mlx.FromValues([]float32{1, -1, -1, 1, 1, -1, -1, 1}, 2, 2, 2).AsType(mlx.DTypeBFloat16)
			for _, kind := range []int{0, 1, 2} {
				got := m.pointer(query, options, kind)
				mlx.Eval(got)
				if got.DType() != mlx.DTypeFloat32 {
					t.Fatalf("pointer output dtype = %v", got.DType())
				}
				magnitude := math.Sqrt(2) / (1 + 1e-5)
				if kind == 1 {
					magnitude /= 2
				}
				for i, want := range []float64{magnitude, -magnitude, -magnitude, magnitude} {
					if v := float64(got.Floats()[i]); math.IsNaN(v) || math.Abs(v-want) > 2e-3 {
						t.Fatalf("kind %d output %d = %g, want %g", kind, i, v, want)
					}
				}
			}
			row := ScoreRow{Tokens: []int32{0, 0, 0}, Pointers: []int32{1, 0}, Type: 0}
			hidden := mlx.FromValues([]float32{1, -1, -1, 1, 1, -1, 99, -99}, 1, 4, 2)
			got := m.FinishScore(row, hidden)
			magnitude := math.Sqrt(2) / (1 + 1e-5)
			if math.Abs(float64(got[0])+magnitude) > 2e-3 || math.Abs(float64(got[1])-magnitude) > 2e-3 {
				t.Fatalf("wrong pointer order or query position with padded tail: %v", got)
			}
		})
	})
}
