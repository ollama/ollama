package flint

import (
	"math"
	"testing"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlx/mlxtest"
)

func eagerFlintSwiGLU(gate, up *mlx.Array) *mlx.Array {
	dtype := gate.DType()
	return mlx.Mul(mlx.SiLU(gate.AsType(mlx.DTypeFloat32)), up.AsType(mlx.DTypeFloat32)).AsType(dtype)
}

func eagerFlintExpertWeightedSum(selected, weights *mlx.Array) *mlx.Array {
	dtype := selected.DType()
	weighted := mlx.Mul(selected.AsType(mlx.DTypeFloat32), mlx.ExpandDims(weights, -1))
	return mlx.Sum(weighted, 1, false).AsType(dtype)
}

func eagerFlintExpertWeightedSumAdd(selected, weights, shared, residual *mlx.Array) *mlx.Array {
	routed := eagerFlintExpertWeightedSum(selected, weights)
	return mlx.Add(residual, mlx.Add(routed, shared).AsType(shared.DType()))
}

func eagerFlintRouter(logits *mlx.Array, topK int32) (weights, indices *mlx.Array) {
	probabilities := mlx.Sigmoid(logits.AsType(mlx.DTypeFloat32))
	indices = mlx.Argpartition(mlx.Neg(probabilities), int(topK)-1, -1)
	indices = mlx.SliceStartStop(indices, []int32{0, 0}, []int32{int32(logits.Dim(0)), topK})
	weights = mlx.TakeAlongAxis(probabilities, indices, -1)
	return mlx.Div(weights, mlx.Sum(weights, -1, true)), indices
}

func TestFlintRouterCompiledMatchesEager(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		mlx.EnableCompile()
		for _, topK := range []int32{1, 2, 6} {
			logits := mlx.FromValues([]float32{
				-2, 0.3, 1.1, -0.7, 2.4, 0.1, 0.8, -1.3,
				0.2, 1.4, -0.5, 2.1, -1.7, 0.9, -0.2, 1.7,
			}, 2, 8).AsType(mlx.DTypeBFloat16)
			got := flintRouter(topK)(logits)
			wantWeights, wantIndices := eagerFlintRouter(logits, topK)
			mlx.Eval(got[0], got[1], wantWeights, wantIndices)
			gotIndices := got[1].AsType(mlx.DTypeInt32).Ints()
			wantIndicesValues := wantIndices.AsType(mlx.DTypeInt32).Ints()
			for i, index := range gotIndices {
				if index != wantIndicesValues[i] {
					t.Fatalf("topK=%d index[%d]=%d, want %d", topK, i, index, wantIndicesValues[i])
				}
			}
			wantWeightsValues := wantWeights.Floats()
			for i, weight := range got[0].Floats() {
				if delta := math.Abs(float64(weight - wantWeightsValues[i])); delta > 1e-6 {
					t.Fatalf("topK=%d weight[%d]=%g, want %g", topK, i, weight, wantWeightsValues[i])
				}
			}
		}
	})
}

func TestFlintCompiledPrecisionMatchesEager(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		mlx.EnableCompile()
		bf16 := func(values []float32, dims ...int) *mlx.Array {
			return mlx.FromValues(values, dims...).AsType(mlx.DTypeBFloat16)
		}
		x := bf16([]float32{-0.2, 0.1, 0.4, 0.8}, 1, 4)
		up := bf16([]float32{0.8, 1.1, 1.4, 0.9}, 1, 4)
		selected := bf16([]float32{0.19, 0.33, 0.61, 0.83, 1.11, 1.37}, 1, 3, 2)
		scores := mlx.FromValues([]float32{0.17, 0.29, 0.54}, 1, 3)
		for _, tc := range []struct {
			name string
			got  *mlx.Array
			want *mlx.Array
		}{
			{"SwiGLU", flintSwiGLU(x, up), eagerFlintSwiGLU(x, up)},
			{"expert sum", flintExpertWeightedSum(selected, scores), eagerFlintExpertWeightedSum(selected, scores)},
		} {
			got := mlx.Contiguous(tc.got.AsType(mlx.DTypeFloat32), false)
			want := mlx.Contiguous(tc.want.AsType(mlx.DTypeFloat32), false)
			mlx.Eval(got, want)
			if got.Size() != want.Size() {
				t.Fatalf("%s output size %d, want %d", tc.name, got.Size(), want.Size())
			}
			gotValues, wantValues := got.Floats(), want.Floats()
			for i, value := range gotValues {
				if delta := math.Abs(float64(value - wantValues[i])); delta > 1e-6 {
					t.Fatalf("%s output[%d] = %g, want %g", tc.name, i, value, wantValues[i])
				}
			}
		}
	})
}

func BenchmarkFlintSwiGLUEager(b *testing.B) {
	benchmarkFlintPrecision(b, "SwiGLU", false)
}

func BenchmarkFlintSwiGLUCompiled(b *testing.B) {
	benchmarkFlintPrecision(b, "SwiGLU", true)
}

func BenchmarkFlintExpertWeightedSumEager(b *testing.B) {
	benchmarkFlintPrecision(b, "expert sum", false)
}

func BenchmarkFlintExpertWeightedSumCompiled(b *testing.B) {
	benchmarkFlintPrecision(b, "expert sum", true)
}

func BenchmarkFlintExpertWeightedSumAddEager(b *testing.B) {
	benchmarkFlintPrecision(b, "expert sum add", false)
}

func BenchmarkFlintExpertWeightedSumAddCompiled(b *testing.B) {
	benchmarkFlintPrecision(b, "expert sum add", true)
}

func BenchmarkFlintRouterEager(b *testing.B) {
	benchmarkFlintPrecision(b, "router", false)
}

func BenchmarkFlintRouterCompiled(b *testing.B) {
	benchmarkFlintPrecision(b, "router", true)
}

func benchmarkFlintPrecision(b *testing.B, operation string, compiled bool) {
	worker := mlxtest.Worker(b)
	if err := worker.Do(b.Context(), func() error {
		mlx.EnableCompile()
		defer mlx.ClearCache()
		x := mlx.AddScalar(mlx.Zeros(mlx.DTypeBFloat16, 1, 4096), 1)
		selected := mlx.AddScalar(mlx.Zeros(mlx.DTypeBFloat16, 1, 8, 4096), 1)
		scores := mlx.AddScalar(mlx.Zeros(mlx.DTypeFloat32, 1, 8), 0.125)
		logits := mlx.FromValues([]float32{0.1, -0.4, 0.9, 1.1, -0.2, 0.5, -0.8, 0.3}, 1, 8)
		mlx.Eval(x, selected, scores, logits)

		var compute func() *mlx.Array
		switch operation {
		case "SwiGLU":
			if compiled {
				compute = func() *mlx.Array { return flintSwiGLU(x, x) }
			} else {
				compute = func() *mlx.Array { return eagerFlintSwiGLU(x, x) }
			}
		case "expert sum":
			if compiled {
				compute = func() *mlx.Array { return flintExpertWeightedSum(selected, scores) }
			} else {
				compute = func() *mlx.Array { return eagerFlintExpertWeightedSum(selected, scores) }
			}
		case "expert sum add":
			if compiled {
				compute = func() *mlx.Array { return flintExpertWeightedSumAdd(selected, scores, x, x)[0] }
			} else {
				compute = func() *mlx.Array { return eagerFlintExpertWeightedSumAdd(selected, scores, x, x) }
			}
		case "router":
			if compiled {
				compute = func() *mlx.Array { return flintRouter(6)(logits)[0] }
			} else {
				compute = func() *mlx.Array {
					weights, _ := eagerFlintRouter(logits, 6)
					return weights
				}
			}
		default:
			b.Fatalf("unknown Flint operation %q", operation)
		}
		mlx.Scoped(func() { mlx.Eval(compute()) })
		b.ResetTimer()
		for range b.N {
			mlx.Scoped(func() { mlx.Eval(compute()) })
		}
		return nil
	}); err != nil {
		b.Fatal(err)
	}
}

func TestBF16OperationsRoundAfterFloat32Math(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		bf16 := func(values []float32, dims ...int) *mlx.Array {
			return mlx.FromValues(values, dims...).AsType(mlx.DTypeBFloat16)
		}
		check := func(name string, array *mlx.Array, want []float32) {
			got := mlx.Contiguous(array.AsType(mlx.DTypeFloat32), false)
			mlx.Eval(got)
			values := got.Floats()
			if len(values) != len(want) {
				t.Fatalf("%s length = %d, want %d", name, len(values), len(want))
			}
			for i := range values {
				if values[i] != want[i] {
					t.Errorf("%s[%d] = %g, want %g", name, i, values[i], want[i])
				}
			}
		}

		check("SwiGLU", flintSwiGLU(
			bf16([]float32{0.2, 0.5, 1.2, 3.4}, 1, 4),
			bf16([]float32{0.8, 1.1, 1.4, 0.9}, 1, 4)),
			[]float32{0.08837890625, 0.34375, 1.296875, 2.96875})
		check("expert sum", flintExpertWeightedSum(
			bf16([]float32{0.19, 0.33, 0.61, 0.83, 1.11, 1.37}, 1, 3, 2),
			mlx.FromValues([]float32{0.17, 0.29, 0.54}, 1, 3)),
			[]float32{0.80859375, 1.03125})
	})
}
