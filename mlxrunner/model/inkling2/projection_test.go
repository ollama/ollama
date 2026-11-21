package inkling2

import (
	"math/rand/v2"
	"slices"
	"testing"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlx/mlxtest"
	"github.com/ollama/ollama/mlxrunner/nn"
)

func TestTargetProjectionUsesConsistentRows(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		mlx.Scoped(func() {
			const hidden, output = 2304, 128
			rng := rand.New(rand.NewPCG(448, 1536))
			values := func(n int) []float32 {
				data := make([]float32, n)
				for i := range data {
					data[i] = rng.Float32() - 0.5
				}
				return data
			}
			input := mlx.FromValues(values(2*shortTargetRows*hidden), 2*shortTargetRows, hidden).AsType(mlx.DTypeBFloat16)
			weight := mlx.FromValues(values(output*hidden), output, hidden).AsType(mlx.DTypeBFloat16)
			bias := mlx.FromValues(values(output), output).AsType(mlx.DTypeBFloat16)
			linear := nn.NewLinear(weight, bias)

			var want []float32
			for i := range 2 {
				row := input.Slice(mlx.Slice(i*shortTargetRows, (i+1)*shortTargetRows), mlx.Slice())
				want = append(want, linear.Forward(row).AsType(mlx.DTypeFloat32).Flatten(0, -1).Floats()...)
			}
			batched := projectTarget(linear, input.Reshape(2, shortTargetRows, hidden))
			if got := batched.AsType(mlx.DTypeFloat32).Flatten(0, -1).Floats(); !slices.Equal(got, want) {
				t.Fatal("joining two verification blocks changed their projection")
			}
			for tokens := 1; tokens <= shortTargetRows; tokens++ {
				ids := input.Slice(mlx.Slice(0, tokens), mlx.Slice()).Reshape(1, tokens, hidden)
				got := projectTarget(linear, ids).AsType(mlx.DTypeFloat32).Flatten(0, -1).Floats()
				if !slices.Equal(got, want[:tokens*output]) {
					t.Errorf("projection at width %d differs from width %d", tokens, shortTargetRows)
				}
			}

			prefill := input.Reshape(1, 2*shortTargetRows, hidden)
			got := projectTarget(linear, prefill).AsType(mlx.DTypeFloat32).Flatten(0, -1).Floats()
			ordinary := linear.Forward(prefill).AsType(mlx.DTypeFloat32).Flatten(0, -1).Floats()
			if !slices.Equal(got, ordinary) {
				t.Error("larger prefill changed from the ordinary projection path")
			}
		})
	})
}
