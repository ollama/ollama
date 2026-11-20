package inkling2

import (
	"math"
	"slices"
	"testing"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlx/mlxtest"
)

func TestResidualMixPreservesIndependentQueries(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		mlx.Scoped(func() {
			sources := []*mlx.Array{
				mlx.FromValues([]float32{3, 0, 0, 5}, 1, 2, 2),
				mlx.FromValues([]float32{0, 4, 7, 0}, 1, 2, 2),
			}
			queries := mlx.FromValues([]float32{1, 0, 0, 1}, 2, 2)
			got := mixResiduals(sources, queries, true)
			if !slices.Equal(got.Dims(), []int{1, 2, 2, 2}) {
				t.Fatalf("residual mixture shape = %v, want [1 2 2 2]", got.Dims())
			}
			high := math.Sqrt2 / (1 + math.Exp(-1))
			low := math.Sqrt2 - high
			want := []float64{high, low, low, high, high, low, low, high}
			values := got.Floats()
			if len(values) != len(want) {
				t.Fatalf("residual mixture produced %d values, want %d", len(values), len(want))
			}
			for i, value := range values {
				if !(math.Abs(float64(value)-want[i]) <= 1e-5) {
					t.Errorf("residual mixture element %d = %g, want %g", i, value, want[i])
				}
			}
		})
	})
}
