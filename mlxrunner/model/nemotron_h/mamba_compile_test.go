package nemotron_h

import (
	"fmt"
	"math"
	"testing"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlx/mlxtest"
)

func mambaGateEager(y, gate *mlx.Array) *mlx.Array {
	return mlx.Mul(y, mlx.SiLU(gate.AsType(y.DType())))
}

func mambaNormScaleEager(y, weight *mlx.Array) *mlx.Array {
	return mlx.Mul(y.AsType(weight.DType()), weight)
}

func mambaCompileInputs(tokens int, norm bool) (*mlx.Array, *mlx.Array) {
	const width = 4096
	values := make([]float32, tokens*width)
	for i := range values {
		values[i] = float32(math.Sin(float64(i) * 0.17))
	}
	y := mlx.FromValues(values, 1, tokens, width)
	if norm {
		weight := mlx.FromValues(values[:width], 1, 1, 8, width/8)
		return mlx.Reshape(y, 1, int32(tokens), 8, width/8), weight
	}
	// The recurrent scan returns FP32; the projection supplies BF16 gates.
	return y, mlx.FromValues(values, 1, tokens, width).AsType(mlx.DTypeBFloat16)
}

func TestMambaCompiledMatchesEager(t *testing.T) {
	for _, op := range []struct {
		name     string
		norm     bool
		eager    func(*mlx.Array, *mlx.Array) *mlx.Array
		compiled func(*mlx.Array, *mlx.Array) *mlx.Array
	}{
		{"gate", false, mambaGateEager, mambaGate},
		{"norm_scale", true, mambaNormScaleEager, mambaNormScale},
	} {
		for _, tokens := range []int{1, 128} {
			mlxtest.RunSubtest(t, fmt.Sprintf("%s/%d", op.name, tokens), func(t *mlxtest.T) {
				mlx.EnableCompile()
				mlx.Scoped(func() {
					y, arg := mambaCompileInputs(tokens, op.norm)
					got, want := op.compiled(y, arg), op.eager(y, arg)
					if got.DType() != want.DType() {
						t.Fatalf("dtype = %v, want %v", got.DType(), want.DType())
					}
					mlx.Eval(got, want)
					assertAllClose(t, op.name, got.Floats(), want.Floats(), 1e-6)
				})
			})
		}
	}
}
