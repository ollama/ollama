package mlx

import (
	"math"
	"testing"

	"github.com/ollama/ollama/mlx/mlxthread/mlxthreadtest"
)

func TestShortMatmul(t *testing.T) {
	for name, weightType := range map[string]DType{"bf16": DTypeBFloat16, "f32": DTypeFloat32} {
		t.Run(name, func(t *testing.T) {
			withMLXThread(t, func(t *mlxthreadtest.T) {
				if !MetalIsAvailable() {
					t.Skip("requires Metal")
				}
				const rows, width, columns = 8, 520, 100
				w := patternArray(weightType, []int{columns, width}, 0, 0.01, 7, 97)
				x := patternArray(DTypeBFloat16, []int{rows, width}, 0.1, 0.02, 11, 89)
				all := ShortMatmul(x, w)
				if all == nil {
					t.Fatal("kernel did not run")
				}
				got := all.AsType(DTypeFloat32).Floats()
				for m := range rows {
					// A one-row product is a matrix-vector product with an FP32 reduction.
					row := x.Slice(Slice(m, m+1), Slice())
					alone := ShortMatmul(row, w).AsType(DTypeFloat32).Floats()
					want := row.AsType(DTypeFloat32).Matmul(w.AsType(DTypeFloat32).Transpose(1, 0)).Floats()
					for n := range columns {
						if alone[n] != got[m*columns+n] {
							t.Fatalf("row %d column %d depends on the other rows: %g vs %g", m, n, alone[n], got[m*columns+n])
						}
						if math.Abs(float64(alone[n]-want[n])) > 1e-2*math.Abs(float64(want[n]))+1e-4 {
							t.Fatalf("row %d column %d = %g, want %g", m, n, alone[n], want[n])
						}
					}
				}
			})
		})
	}
}
