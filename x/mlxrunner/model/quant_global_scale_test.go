package model

import (
	"math"
	"testing"

	"github.com/ollama/ollama/x/internal/mlxtest"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
)

func TestPrepareGatherQMMGlobalScale(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		if got := PrepareGatherQMMGlobalScale(nil, 4); got != nil {
			t.Fatal("nil global scale did not pass through as nil")
		}

		scalar := PrepareGatherQMMGlobalScale(mlx.FromValues([]float32{2}, 1), 4)
		mlx.Eval(scalar)
		if dims := scalar.Dims(); len(dims) != 1 || dims[0] != 4 {
			t.Fatalf("singleton scale dims = %v, want [4]", dims)
		}
		want := float32(2 * mlx.Nvfp4MaxProduct)
		for i, got := range scalar.Floats() {
			if got != want {
				t.Fatalf("broadcast scale[%d] = %v, want %v", i, got, want)
			}
		}

		checkpointScales := []float32{0.5, 1, 2, 4}
		perExpert := PrepareGatherQMMGlobalScale(mlx.FromValues(checkpointScales, 4), 4)
		mlx.Eval(perExpert)
		for i, got := range perExpert.Floats() {
			if want := checkpointScales[i] * float32(mlx.Nvfp4MaxProduct); got != want {
				t.Fatalf("per-expert scale[%d] = %v, want %v", i, got, want)
			}
		}

		// A scale count that is neither a checkpoint-wide scalar nor one per
		// expert is malformed and fails the load instead of being ignored.
		func() {
			defer func() {
				if recover() == nil {
					t.Fatal("mismatched scale count did not fail")
				}
			}()
			PrepareGatherQMMGlobalScale(mlx.FromValues([]float32{1, 2}, 2), 4)
		}()
	})
}

func TestSameGlobalScales(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		perExpert := PrepareGatherQMMGlobalScale(mlx.FromValues([]float32{0.5, 1, 2, 4}, 4), 4)
		same := PrepareGatherQMMGlobalScale(mlx.FromValues([]float32{0.5, 1, 2, 4}, 4), 4)
		reordered := PrepareGatherQMMGlobalScale(mlx.FromValues([]float32{4, 2, 1, 0.5}, 4), 4)
		broadcast := PrepareGatherQMMGlobalScale(mlx.FromValues([]float32{2}, 1), 4)
		shorter := PrepareGatherQMMGlobalScale(mlx.FromValues([]float32{2}, 1), 2)

		for _, tt := range []struct {
			name string
			a, b *mlx.Array
			want bool
		}{
			{"both nil", nil, nil, true},
			{"only one nil", perExpert, nil, false},
			{"identical array", perExpert, perExpert, true},
			{"equal values", perExpert, same, true},
			{"same values reordered", perExpert, reordered, false},
			{"broadcast scalar matches itself", broadcast, broadcast, true},
			{"different expert counts", broadcast, shorter, false},
		} {
			if got := SameGlobalScales(tt.a, tt.b); got != tt.want {
				t.Fatalf("%s: SameGlobalScales() = %v, want %v", tt.name, got, tt.want)
			}
		}
	})
}

// TestGatherQMMGlobalScaleMatchesDequantized checks the whole scale chain —
// Prepare's conversion to amax units plus the GatherQMM wrapper — against the
// dequantized weights gathered densely. Metal runs the native kernel path;
// CUDA runs the wrapper's output scaling. ModelOpt exports a checkpoint-wide
// scalar, so both it and the per-expert form are covered.
func TestGatherQMMGlobalScaleMatchesDequantized(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		if !mlx.MetalIsAvailable() && !mlx.CUDAIsAvailable() {
			t.Skip("gather_qmm requires a GPU backend")
		}

		const experts, rows, cols, groupSize = 4, 32, 64, 16
		packed := make([]uint32, experts*rows*cols/8)
		for i := range packed {
			for j := range 8 {
				packed[i] |= uint32((i*8+j)%16) << (4 * j)
			}
		}
		scales := make([]uint8, experts*rows*cols/groupSize)
		for i := range scales {
			scales[i] = uint8(((i % 4) + 6) << 3)
		}
		weights := mlx.FromValues(packed, experts, rows, cols/8)
		blockScales := mlx.FromValues(scales, experts, rows, cols/groupSize)

		xValues := make([]float32, cols)
		for i := range xValues {
			xValues[i] = float32(i%7-3) / 8
		}
		x := mlx.FromValues(xValues, 1, cols).AsType(mlx.DTypeBFloat16)
		indices := mlx.FromValues([]int32{0, 1, 2, 3}, 1, experts)

		for _, checkpointScale := range [][]float32{
			{0.5, 1, 2, 4}, // one scale per expert
			{2},            // checkpoint-wide scalar
		} {
			checkpoint := mlx.FromValues(checkpointScale, len(checkpointScale))
			kernelScale := PrepareGatherQMMGlobalScale(checkpoint, experts)

			got := mlx.GatherQMM(
				x, weights, blockScales, nil, nil, indices,
				true, groupSize, 4, "nvfp4", kernelScale, false,
			).AsType(mlx.DTypeFloat32)
			dense := mlx.Dequantize(
				weights, blockScales, nil, groupSize, 4, "nvfp4", checkpoint,
			).AsType(mlx.DTypeFloat32)
			want := mlx.GatherMM(
				x.AsType(mlx.DTypeFloat32), mlx.Transpose(dense, 0, 2, 1), nil, indices, false,
			)
			mlx.Eval(got, want)

			gotValues, wantValues := got.Floats(), want.Floats()
			if len(gotValues) != len(wantValues) {
				t.Fatalf("result length = %d, want %d", len(gotValues), len(wantValues))
			}
			for i := range gotValues {
				if math.IsNaN(float64(gotValues[i])) || math.IsInf(float64(gotValues[i]), 0) {
					t.Fatalf("result[%d] = %v, want finite", i, gotValues[i])
				}
				delta := math.Abs(float64(gotValues[i] - wantValues[i]))
				tolerance := 0.01 * math.Max(math.Abs(float64(wantValues[i])), 1)
				if delta > tolerance {
					t.Fatalf("result[%d] = %v, want %v (delta %v > %v)", i, gotValues[i], wantValues[i], delta, tolerance)
				}
			}
		}
	})
}
