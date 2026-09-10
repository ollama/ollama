package model

import (
	"math"
	"testing"

	"github.com/ollama/ollama/x/internal/mlxtest"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
)

func TestPrepareGatherQMMGlobalScale(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		if !mlx.MetalIsAvailable() {
			t.Skip("gather_qmm global scale requires Metal")
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
		checkpointScale := mlx.FromValues([]float32{0.5, 1, 2, 4}, experts)
		kernelScale, ok := PrepareGatherQMMGlobalScale(checkpointScale, "nvfp4", experts)
		if !ok {
			t.Fatal("PrepareGatherQMMGlobalScale reported unsupported on Metal")
		}
		scalarScale, ok := PrepareGatherQMMGlobalScale(mlx.FromValues([]float32{2}, 1), "nvfp4", experts)
		if !ok {
			t.Fatal("PrepareGatherQMMGlobalScale rejected a singleton checkpoint scale")
		}
		mlx.Eval(scalarScale)
		if dims := scalarScale.Dims(); len(dims) != 1 || dims[0] != experts {
			t.Fatalf("singleton scale dims = %v, want [%d]", dims, experts)
		}
		scalarValues := scalarScale.Floats()
		for i, got := range scalarValues {
			if want := float32(2 * nvfp4MaxProduct); got != want {
				t.Fatalf("singleton scale = %v; value[%d] = %v, want %v", scalarValues, i, got, want)
			}
		}
		if _, ok := PrepareGatherQMMGlobalScale(mlx.FromValues([]float32{1, 2}, 2), "nvfp4", experts); ok {
			t.Fatal("PrepareGatherQMMGlobalScale accepted a scale count that does not match the expert bank")
		}

		xValues := make([]float32, cols)
		for i := range xValues {
			xValues[i] = float32(i%7-3) / 8
		}
		x := mlx.FromValues(xValues, 1, cols).AsType(mlx.DTypeBFloat16)
		indices := mlx.FromValues([]int32{0, 1, 2, 3}, 1, experts)

		got := mlx.GatherQMMWithGlobalScale(
			x, weights, blockScales, nil, nil, indices,
			true, groupSize, 4, "nvfp4", kernelScale, false,
		).AsType(mlx.DTypeFloat32)
		dense := mlx.Dequantize(
			weights, blockScales, nil, groupSize, 4, "nvfp4", checkpointScale,
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
			delta := math.Abs(float64(gotValues[i] - wantValues[i]))
			tolerance := 0.01 * math.Max(math.Abs(float64(wantValues[i])), 1)
			if delta > tolerance {
				t.Fatalf("result[%d] = %v, want %v (delta %v > %v)", i, gotValues[i], wantValues[i], delta, tolerance)
			}
		}
	})
}
