package mlx

import (
	"math"
	"testing"
)

func TestAddmmBroadcastBiasMatchesMatmulAdd(t *testing.T) {
	skipIfNoMLX(t)

	// Zero input isolates the broadcast-bias term. The old MLX addmm path
	// matched matmul-only here, proving the bias was dropped for batched input.
	const (
		seqLen = 4
		inDim  = 8
		outDim = 8
	)

	weightVals := make([]float32, outDim*inDim)
	for i := range weightVals {
		weightVals[i] = float32((i % 5) + 1)
	}
	biasVals := make([]float32, outDim)
	for i := range biasVals {
		biasVals[i] = float32(i%7 - 3)
	}
	inputVals := make([]float32, seqLen*inDim)

	weight := FromValues(weightVals, outDim, inDim).AsType(DTypeBFloat16)
	bias := FromValues(biasVals, outDim).AsType(DTypeBFloat16)
	wT := weight.Transpose(1, 0)

	// Keep all cases on this locked goroutine because MLX streams are
	// thread-local; t.Run would execute subtests on different goroutines.
	for _, tc := range []struct {
		name  string
		input *Array
	}{
		{
			name:  "3d-batched",
			input: FromValues(inputVals, 1, seqLen, inDim).AsType(DTypeBFloat16),
		},
		{
			name:  "2d-batched",
			input: FromValues(inputVals, seqLen, inDim).AsType(DTypeBFloat16),
		},
		{
			name:  "3d-single",
			input: FromValues(inputVals[(seqLen-1)*inDim:], 1, 1, inDim).AsType(DTypeBFloat16),
		},
		{
			name:  "2d-single",
			input: FromValues(inputVals[(seqLen-1)*inDim:], 1, inDim).AsType(DTypeBFloat16),
		},
	} {
		addmm := bias.Addmm(tc.input, wT, 1.0, 1.0).AsType(DTypeFloat32)
		matmulAdd := tc.input.Matmul(wT).Add(bias).AsType(DTypeFloat32)
		matmulOnly := tc.input.Matmul(wT).AsType(DTypeFloat32)
		Eval(addmm, matmulAdd, matmulOnly)

		got := addmm.Floats()
		want := matmulAdd.Floats()
		if len(got) != len(want) {
			t.Fatalf("%s output length = %d, want %d", tc.name, len(got), len(want))
		}

		addDiff, addIndex := maxAbsDiff(got, want)
		onlyDiff, onlyIndex := maxAbsDiff(got, matmulOnly.Floats())
		t.Logf("%s max diff addmm vs matmul+add: %.6f at %d", tc.name, addDiff, addIndex)
		t.Logf("%s max diff addmm vs matmul-only: %.6f at %d", tc.name, onlyDiff, onlyIndex)
		if addDiff > 1e-6 {
			t.Fatalf("%s addmm output[%d] = %.6f, want %.6f (max diff %.6f)", tc.name, addIndex, got[addIndex], want[addIndex], addDiff)
		}
	}
}

func maxAbsDiff(a, b []float32) (float64, int) {
	var maxDiff float64
	var maxIndex int
	for i := range a {
		if diff := math.Abs(float64(a[i] - b[i])); diff > maxDiff {
			maxDiff = diff
			maxIndex = i
		}
	}
	return maxDiff, maxIndex
}
