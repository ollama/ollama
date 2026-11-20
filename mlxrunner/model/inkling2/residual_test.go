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

func TestResidualMixUnnormalizedValues(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		mlx.Scoped(func() {
			const width = 2304
			zero := mlx.Zeros(mlx.DTypeBFloat16, 1, 2, width)
			sources := []*mlx.Array{mlx.AddScalar(zero, 2), mlx.AddScalar(zero, 8)}
			// Equal scores average the original values in this configuration.
			got := mixResiduals(sources, mlx.Zeros(mlx.DTypeFloat32, width), false)
			for i, value := range got.Floats() {
				if value != 5 {
					t.Fatalf("unnormalized residual[%d] = %g, want 5", i, value)
				}
			}
		})
	})
}
