package mlx

import (
	"fmt"
	"testing"
)

func TestDepthwiseConvSiLUMatchesGraph(t *testing.T) {
	skipIfNoMLX(t)

	// Collected, not reported inside the callback: a t.Fatal there calls
	// runtime.Goexit, which the MLX worker cannot recover, so the job's result
	// is never delivered and the test hangs until the binary timeout.
	var mismatches []string
	withMLXThread(t, func() {
		mismatches = depthwiseConvSiLUMismatches()
	})
	for _, m := range mismatches {
		t.Error(m)
	}
}

func depthwiseConvSiLUMismatches() []string {
	var mismatches []string
	for _, dtype := range []DType{DTypeBFloat16, DTypeFloat32} {
		for _, withBias := range []bool{false, true} {
			for _, shape := range []struct{ B, T, C, K int }{
				{1, 1, 64, 4},
				{1, 4, 64, 4},
				{1, 11, 96, 4},
				{1, 64, 64, 4},
				{1, 333, 64, 4},
				{3, 7, 64, 4},
				{2, 5, 32, 2},
			} {
				name := fmt.Sprintf("%v_bias%v_b%d_t%d_c%d_k%d", dtype, withBias, shape.B, shape.T, shape.C, shape.K)
				x := patternArray(dtype, []int{shape.B, shape.T + shape.K - 1, shape.C}, 0.02, 0.004, 41, 263)
				w := patternArray(dtype, []int{shape.C, shape.K}, 0.1, 0.01, 7, 53)
				var bias *Array
				if withBias {
					bias = patternArray(dtype, []int{shape.C}, -0.3, 0.02, 11, 37)
				}

				ref := SiLU(Conv1d(x, Reshape(w, int32(shape.C), int32(shape.K), 1), bias, 1, 0, 1, int32(shape.C)))
				y := DepthwiseConvSiLU(x, w, bias, shape.T)
				if err := requireExact("y", y, ref); err != nil {
					mismatches = append(mismatches, fmt.Sprintf("%s: %v", name, err))
				}
			}
		}
	}
	return mismatches
}
