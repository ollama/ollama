package hy4_flash

import (
	"math"
	"slices"
	"testing"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlx/mlxtest"
	"github.com/ollama/ollama/mlxrunner/nn"
)

func TestOutputHeadSingleAndBatched(t *testing.T) {
	t.Run("promote", func(t *testing.T) { testOutputHead(t, false) })
	t.Run("keep single-token projection", func(t *testing.T) { testOutputHead(t, true) })
}

func testOutputHead(t *testing.T, keepSingle bool) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		const vocab, width = 11, 256
		w := mlx.Sin(mlx.MulScalar(mlx.Arange(0, vocab*width, 1, mlx.DTypeFloat32), .17)).AsType(mlx.DTypeBFloat16).Reshape(vocab, width)
		ref := nn.NewLinear(w.AsType(mlx.DTypeFloat32), nil)
		head := loadOutputHead(w)
		mixed, _ := head.(*outputHead)
		if (mlx.MetalIsAvailable() || mlx.CUDAIsAvailable()) && mixed == nil {
			t.Fatal("the GPU backend should use the BF16 head")
		}
		if keepSingle && mixed != nil {
			mixed.Batched = nn.NewLinear(mixed.Weight.Clone(), nil)
		}
		input := func(rows, tokens int) *mlx.Array {
			positions := mlx.Arange(0, float64(rows*tokens*width), 1, mlx.DTypeFloat32)
			return mlx.Cos(mlx.MulScalar(positions, .23)).Reshape(rows, tokens, width)
		}
		check := func(got, want *mlx.Array) {
			t.Helper()
			if got.DType() != mlx.DTypeFloat32 || !slices.Equal(got.Dims(), want.Dims()) {
				t.Fatalf("got %v %v, want %v %v", got.Dims(), got.DType(), want.Dims(), want.DType())
			}
			wantValues := want.Floats()
			for i, v := range got.Floats() {
				if delta := math.Abs(float64(v - wantValues[i])); delta > 1e-5 || math.IsNaN(delta) {
					t.Fatalf("logit %d differs by %g", i, delta)
				}
			}
		}

		// Keep a lazy single-row result alive while a batch replaces the
		// head's held weight. The result must retain its original inputs.
		pending := mlx.ScopedArrays(func() []*mlx.Array {
			x := input(1, 1)
			return []*mlx.Array{head.Forward(x), ref.Forward(x)}
		})
		if mixed != nil && mixed.Weight.DType() != mlx.DTypeBFloat16 {
			t.Fatal("single-row projection unexpectedly promoted its weights")
		}
		if keepSingle && mixed != nil && mixed.Batched.Weight.DType() != mlx.DTypeBFloat16 {
			t.Fatal("single-row projection unexpectedly promoted its batch weights")
		}
		for _, shape := range [][2]int{{1, 4}, {2, 1}, {1, 1}} {
			mlx.Scoped(func() {
				x := input(shape[0], shape[1])
				check(head.Forward(x), ref.Forward(x))
				if mixed == nil {
					return
				}
				if keepSingle {
					if mixed.Weight.DType() != mlx.DTypeBFloat16 || mixed.Batched.Weight.DType() != mlx.DTypeFloat32 {
						t.Fatalf("shape %v: single weight dtype %v, batched weight dtype %v", shape, mixed.Weight.DType(), mixed.Batched.Weight.DType())
					}
				} else if mixed.Weight.DType() != mlx.DTypeFloat32 || w.DType() != mlx.DTypeBFloat16 {
					t.Fatalf("shape %v: head weight dtype %v, source weight dtype %v", shape, mixed.Weight.DType(), w.DType())
				}
			})
		}
		check(pending[0], pending[1])
	})
}

func TestOutputHeadReleasesBF16WhenSingletonIsUnsupported(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		w := mlx.FromValues([]float32{1, 2, 3}, 1, 3).AsType(mlx.DTypeBFloat16)
		head, ok := loadOutputHead(w).(*outputHead)
		if !ok {
			t.Skip("requires the GPU output head")
		}
		head.Batched = nn.NewLinear(head.Weight.Clone(), nil)
		// Width three is outside the native kernel's tiled shape.
		x := mlx.FromValues([]float32{1, 1, 1}, 1, 1, 3)
		if got := head.Forward(x).Floats(); !slices.Equal(got, []float32{6}) {
			t.Fatalf("output = %v, want [6]", got)
		}
		if head.Weight.DType() != mlx.DTypeFloat32 || head.Batched.Weight.DType() != mlx.DTypeFloat32 {
			t.Fatalf("single weight dtype %v, batched weight dtype %v", head.Weight.DType(), head.Batched.Weight.DType())
		}
	})
}
