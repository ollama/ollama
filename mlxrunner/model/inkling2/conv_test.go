package inkling2

import (
	"math"
	"math/rand/v2"
	"slices"
	"testing"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlx/mlxtest"
	"github.com/ollama/ollama/mlxrunner/batch"
	"github.com/ollama/ollama/mlxrunner/cache"
	"github.com/ollama/ollama/mlxrunner/nn"
)

func TestDynamicShortConvolutionRetainsHistoryAcrossDecode(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		mlx.Scoped(func() {
			convolution := shortConv{
				Weight:  mlx.FromValues([]float32{1, 2}, 2, 1),
				Dynamic: nn.NewLinear(mlx.FromValues([]float32{0, 1}, 2, 1), nil),
				heads:   1,
			}
			state := cache.NewRecurrentCache(1, 1, 1, 1, 1)
			defer state.Free()
			input := mlx.FromValues([]float32{1, 2, 3}, 1, 3, 1)
			prefill := &batch.Batch{InputIDs: mlx.Zeros(mlx.DTypeInt32, 1, 3), SeqQueryLens: []int32{3}}
			full := convolution.forward(input, input, prefill, nil).Floats()
			for index, want := range []float64{2, 8, 16} {
				chunk := &batch.Batch{InputIDs: mlx.Zeros(mlx.DTypeInt32, 1, 1), SeqOffsets: []int32{int32(index)}, SeqQueryLens: []int32{1}}
				token := input.Slice(mlx.Slice(), mlx.Slice(index, index+1), mlx.Slice())
				got := convolution.forward(token, token, chunk, state).Floats()[0]
				if math.Abs(float64(got)-want/math.Sqrt(5)) > 1e-6 || got != full[index] {
					t.Errorf("token %d: decoded %g, prefill %g, want %g", index, got, full[index], want/math.Sqrt(5))
				}
			}
		})
	})
}

func TestMetalConvTapsMatchEager(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		if !mlx.MetalIsAvailable() {
			t.Skip("requires Metal")
		}
		mlx.EnableCompile()
		random := rand.New(rand.NewPCG(71, 219))
		makeValues := func(shape ...int) *mlx.Array {
			count := 1
			for _, size := range shape {
				count *= size
			}
			values := make([]float32, count)
			for i := range values {
				values[i] = float32(random.NormFloat64())
			}
			return mlx.FromValues(values, shape...).AsType(mlx.DTypeBFloat16)
		}
		const rows, channels, taps, heads = 2, 128, 4, 8
		for _, length := range []int{1, 8, 17, 64} {
			mlx.Scoped(func() {
				x := makeValues(rows, length, channels)
				previous := makeValues(rows, taps-1, channels)
				window := mlx.Concatenate([]*mlx.Array{previous, x}, 1)
				for _, dynamic := range []bool{false, true} {
					var inputs []*mlx.Array
					if dynamic {
						weight := makeValues(taps, channels)
						shift := makeValues(rows, length, heads, taps).AsType(mlx.DTypeFloat32)
						inputs = []*mlx.Array{window, x, weight, shift}
					} else {
						inputs = []*mlx.Array{window, x, makeValues(channels, taps).Transpose(1, 0)}
					}
					eager := convTaps(inputs...)[0].AsType(mlx.DTypeFloat32).Floats()
					compiled := compiledConvTaps(inputs...)[0].AsType(mlx.DTypeFloat32).Floats()
					if !slices.Equal(compiled, eager) {
						t.Errorf("length %d dynamic %t: compiled convolution changed the BF16 result", length, dynamic)
					}
				}
			})
		}
	})
}
