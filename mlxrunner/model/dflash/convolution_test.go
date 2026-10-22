package dflash

import (
	"fmt"
	"math"
	"slices"
	"testing"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlx/mlxtest"
)

func TestConvolutionRestartsAtBlockBoundary(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		x := mlx.FromValues([]float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, 1, 3, 4)
		base := mlx.FromValues([]float32{1, 1, 1, 1, 2, 2, 2, 2}, 2, 4)
		dynamic := mlx.AddScalar(mlx.Zeros(mlx.DTypeFloat32, 1, 3, 2, 2), 1)
		out := convolve(x, dynamic, base)
		mlx.Eval(out)
		want := []float32{2, 4, 6, 8, 13, 18, 23, 28, 33, 38, 43, 48}
		if got := out.Floats(); !slices.Equal(got, want) {
			t.Fatalf("convolution = %v, want %v", got, want)
		}
		// A new one-token proposal must not read the preceding block's tail.
		next := convolve(x.Slice(mlx.Slice(), mlx.Slice(0, 1), mlx.Slice()), dynamic.Slice(mlx.Slice(), mlx.Slice(0, 1), mlx.Slice(), mlx.Slice()), base)
		mlx.Eval(next)
		if got := next.Floats(); !slices.Equal(got, want[:4]) {
			t.Fatalf("next block = %v, want %v", got, want[:4])
		}
	})
}

func TestConvolutionCompiledMatchesEager(t *testing.T) {
	for _, dtype := range []mlx.DType{mlx.DTypeFloat32, mlx.DTypeBFloat16} {
		for _, length := range []int{1, 8} {
			mlxtest.RunSubtest(t, fmt.Sprintf("%v/%d", dtype, length), func(t *mlxtest.T) {
				x := mlx.DivScalar(mlx.Arange(0, float64(length*32), 1, mlx.DTypeFloat32), 32).Reshape(1, length, 32).AsType(dtype)
				kernel := mlx.DivScalar(mlx.Arange(0, float64(length*3*2), 1, mlx.DTypeFloat32), 64).Reshape(1, length, 3, 2).AsType(dtype)
				base := mlx.DivScalar(mlx.Arange(0, 3*32, 1, mlx.DTypeFloat32), 128).Reshape(3, 32).AsType(dtype)
				got, want := convolve(x, kernel, base), convolveEager(x, kernel, base)
				if got.DType() != dtype {
					t.Fatalf("output dtype = %v, want %v", got.DType(), dtype)
				}
				got, want = got.AsType(mlx.DTypeFloat32), want.AsType(mlx.DTypeFloat32)
				mlx.Eval(got, want)
				actual, expected := got.Floats(), want.Floats()
				for i, v := range actual {
					if math.Abs(float64(v-expected[i])) > 1e-6 || math.IsNaN(float64(v)) {
						t.Fatalf("element %d = %g, want %g", i, v, expected[i])
					}
				}
			})
		}
	}
}
