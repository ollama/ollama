package mlx

import (
	"fmt"
	"math"
	"testing"

	"github.com/ollama/ollama/mlx/mlxthread/mlxthreadtest"
)

func TestGatherWeightedSum(t *testing.T) {
	withMLXThread(t, func(t *mlxthreadtest.T) {
		for _, dtype := range []DType{DTypeFloat32, DTypeBFloat16, DTypeFloat16} {
			for _, shape := range [][3]int{{1, 3, 17}, {8, 22, 64}, {32, 22, 33}} {
				for _, gather := range []bool{false, true} {
					Scoped(func() {
						N, K, D := shape[0], shape[1], shape[2]
						data := make([]float32, N*K*D)
						for i := range data {
							data[i] = float32(math.Sin(float64(i) * 0.17))
						}
						// Exercise the kernel's contiguous-input conversion too.
						values := Transpose(FromValues(data, D, N*K), 1, 0).AsType(dtype)
						weights := make([]float32, N*K)
						permutation := make([]int32, N*K)
						for i := range weights {
							weights[i] = float32(0.2 + 0.1*math.Cos(float64(i)*0.23))
							permutation[i] = int32(N*K - 1 - i)
						}
						scores := FromValues(weights, N, K)
						var indices *Array
						if gather {
							indices = FromValues(permutation, N, K)
						}
						got := GatherWeightedSum(values, scores, indices)
						if MetalIsAvailable() && got.name != "WEIGHTED_SUM" {
							t.Fatal("Metal weighted-sum kernel was skipped")
						}
						if got.DType() != dtype || got.Dim(0) != N || got.Dim(1) != D {
							t.Fatalf("unexpected output dtype/shape: %v %v", got.DType(), got.Dims())
						}
						fp32 := Contiguous(values.AsType(DTypeFloat32), false)
						Eval(fp32)
						v := fp32.Floats()
						want := make([]float32, N*D)
						for n := range N {
							for d := range D {
								var sum float64
								for k := range K {
									slot := n*K + k
									row := slot
									if gather {
										row = int(permutation[slot])
									}
									sum += float64(v[row*D+d]) * float64(weights[slot])
								}
								want[n*D+d] = float32(sum)
							}
						}
						expected := FromValues(want, N, D).AsType(dtype).AsType(DTypeFloat32)
						actual := got.AsType(DTypeFloat32)
						graph := gatherWeightedSumGraph(values, scores, indices).AsType(DTypeFloat32)
						Eval(actual, expected, graph)
						label := fmt.Sprintf("dtype=%v shape=%v gather=%v", dtype, shape, gather)
						gotValues := actual.Floats()
						for name, reference := range map[string]*Array{"oracle": expected, "fallback": graph} {
							for i, want := range reference.Floats() {
								tolerance := 2e-6 * math.Max(1, math.Abs(float64(want)))
								// Permit one final output ULP when FP32 accumulation
								// straddles a low-precision rounding boundary.
								if dtype != DTypeFloat32 {
									bits := 7
									if dtype == DTypeFloat16 {
										bits = 10
									}
									tolerance = math.Max(2e-6, math.Ldexp(1, math.Ilogb(math.Abs(float64(want)))-bits))
								}
								got := gotValues[i]
								if math.IsNaN(float64(got)) || math.Abs(float64(got-want)) > tolerance {
									t.Errorf("%s %s[%d]=%g, want %g (tolerance %g)", label, name, i, got, want, tolerance)
									break
								}
							}
						}
					})
				}
			}
		}
	})
}

func TestGatherWeightedSumPreservesFP32Scores(t *testing.T) {
	withMLXThread(t, func(t *mlxthreadtest.T) {
		values := FromValues([]float32{1024, -1024}, 2, 1).AsType(DTypeBFloat16)
		scores := FromValues([]float32{0.5001, 0.5}, 1, 2)
		got := GatherWeightedSum(values, scores, nil).AsType(DTypeFloat32)
		want := FromValues([]float32{(float32(0.5001) - 0.5) * 1024}, 1, 1).AsType(DTypeBFloat16).AsType(DTypeFloat32)
		reportMamba2Failures(t, appendArrayCloseError(nil, "FP32 scores", got, want, 0))
	})
}
