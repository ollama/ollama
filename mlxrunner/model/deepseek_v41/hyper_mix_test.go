package deepseek_v41

import (
	"fmt"
	"math"
	"testing"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlx/mlxtest"
)

func TestConnectionMix(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		const copies, eps = 4, 1e-6
		logitValues := make([]float32, (copies+2)*copies)
		baseValues := make([]float32, len(logitValues))
		for i := range logitValues {
			logitValues[i] = float32(i%7-3) * .37
			baseValues[i] = float32(i%5-2) * .13
		}
		scales := []float32{.7, 1.2, .9}
		logits := mlx.FromValues(logitValues, 1, 1, len(logitValues))
		base := mlx.FromValues(baseValues, len(baseValues))
		scale := mlx.FromValues(scales, 3)
		pre, post, matrix := mlx.ConnectionMix(logits, mlx.Zeros(mlx.DTypeFloat32, 1, 1, 1), scale, base, copies, 20, 1, eps)
		streams := mlx.FromValues([]float32{1, 2, 3, 4, 5, 6, 7, 8}, 1, 1, copies, 2)
		collapsed := collapseConnection(streams, pre)
		injected := injectConnection(mlx.FromValues([]float32{.5, -.75}, 1, 1, 2), streams, post, matrix)
		mlx.Eval(pre, post, matrix, collapsed, injected)

		wantPre, wantPost, wantMatrix := scalarConnectionMix(logitValues, scales, baseValues, copies, 20, eps)
		compareFloats(t, "pre", pre.Floats(), wantPre, 2e-5)
		compareFloats(t, "post", post.Floats(), wantPost, 2e-5)
		compareFloats(t, "matrix", matrix.Floats(), wantMatrix, 2e-5)

		wantCollapsed := make([]float32, 2)
		wantInjected := make([]float32, copies*2)
		streamValues := []float32{1, 2, 3, 4, 5, 6, 7, 8}
		branch := []float32{.5, -.75}
		for i := range copies {
			for channel := range 2 {
				wantCollapsed[channel] += wantPre[i] * streamValues[i*2+channel]
				wantInjected[i*2+channel] = wantPost[i] * branch[channel]
				for j := range copies {
					wantInjected[i*2+channel] += wantMatrix[j*copies+i] * streamValues[j*2+channel]
				}
			}
		}
		compareFloats(t, "collapse", collapsed.Floats(), wantCollapsed, 2e-5)
		compareFloats(t, "inject", injected.Floats(), wantInjected, 2e-5)
	})
}

func scalarConnectionMix(logits, scales, base []float32, copies, iterations int, eps float32) ([]float32, []float32, []float32) {
	pre, post := make([]float32, copies), make([]float32, copies)
	matrix := make([]float32, copies*copies)
	sigmoid := func(x float32) float32 { return 1 / (1 + float32(math.Exp(float64(-x)))) }
	for i := range copies {
		pre[i] = sigmoid(logits[i]*scales[0]+base[i]) + eps
		post[i] = 2 * sigmoid(logits[copies+i]*scales[1]+base[copies+i])
		maxLogit := float32(math.Inf(-1))
		for j := range copies {
			index := 2*copies + i*copies + j
			matrix[i*copies+j] = logits[index]*scales[2] + base[index]
			maxLogit = max(maxLogit, matrix[i*copies+j])
		}
		var sum float32
		for j := range copies {
			index := i*copies + j
			matrix[index] = float32(math.Exp(float64(matrix[index] - maxLogit)))
			sum += matrix[index]
		}
		for j := range copies {
			matrix[i*copies+j] = matrix[i*copies+j]/sum + eps
		}
	}
	normalizeColumns := func() {
		for j := range copies {
			var sum float32
			for i := range copies {
				sum += matrix[i*copies+j]
			}
			for i := range copies {
				matrix[i*copies+j] /= sum + eps
			}
		}
	}
	normalizeRows := func() {
		for i := range copies {
			var sum float32
			for j := range copies {
				sum += matrix[i*copies+j]
			}
			for j := range copies {
				matrix[i*copies+j] /= sum + eps
			}
		}
	}
	normalizeColumns()
	for range iterations - 1 {
		normalizeRows()
		normalizeColumns()
	}
	return pre, post, matrix
}

func compareFloats(t interface {
	Helper()
	Fatal(...any)
}, name string, got, want []float32, tolerance float64,
) {
	t.Helper()
	if err := compareFloatValues(name, got, want, tolerance); err != nil {
		t.Fatal(err)
	}
}

func compareFloatValues(name string, got, want []float32, tolerance float64) error {
	if math.IsNaN(tolerance) || math.IsInf(tolerance, 0) || tolerance < 0 {
		return fmt.Errorf("%s tolerance %g is invalid", name, tolerance)
	}
	if len(got) != len(want) {
		return fmt.Errorf("%s length = %d, want %d", name, len(got), len(want))
	}
	if len(got) == 0 {
		return fmt.Errorf("%s has no values to compare", name)
	}
	for i := range got {
		if math.IsNaN(float64(got[i])) || math.IsInf(float64(got[i]), 0) ||
			math.IsNaN(float64(want[i])) || math.IsInf(float64(want[i]), 0) ||
			math.Abs(float64(got[i]-want[i])) > tolerance {
			return fmt.Errorf("%s[%d] = %g, want %g", name, i, got[i], want[i])
		}
	}
	return nil
}

func TestCompareFloatValuesRejectsNonfinite(t *testing.T) {
	for _, tc := range []struct {
		name      string
		got, want []float32
		tolerance float64
	}{
		{"NaN result", []float32{float32(math.NaN())}, []float32{0}, 1e-5},
		{"infinite result", []float32{float32(math.Inf(1))}, []float32{0}, 1e-5},
		{"NaN expectation", []float32{0}, []float32{float32(math.NaN())}, 1e-5},
		{"NaN tolerance", []float32{0}, []float32{0}, math.NaN()},
		{"empty result", nil, nil, 1e-5},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if err := compareFloatValues("values", tc.got, tc.want, tc.tolerance); err == nil {
				t.Fatal("comparison accepted a nonfinite value")
			}
		})
	}
}
