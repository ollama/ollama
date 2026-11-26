package mlx

import (
	"math"
	"testing"

	"github.com/ollama/ollama/mlx/mlxthread/mlxthreadtest"
)

func TestRMSNormRoundBF16(t *testing.T) {
	withMLXThread(t, func(t *mlxthreadtest.T) {
		if !MetalIsAvailable() {
			t.Skip("requires Metal")
		}
		x := patternArray(DTypeBFloat16, []int{3, 2304}, 0.2, 0.03, 13, 211)
		w := patternArray(DTypeFloat32, []int{2304}, 1, 0.01, 5, 37)
		got := RMSNormRoundBF16(x, w, 1e-6).AsType(DTypeFloat32).Floats()
		want := RMSNormFn(x.AsType(DTypeFloat32), w, 1e-6).AsType(DTypeBFloat16).AsType(DTypeFloat32).Floats()
		pair := RMSNormPairRoundBF16(x, x.AsType(DTypeFloat32), 1e-6).AsType(DTypeFloat32).Floats()
		plain := RMSNormFn(x.AsType(DTypeFloat32), nil, 1e-6).AsType(DTypeBFloat16).AsType(DTypeFloat32).Floats()
		for i := range want {
			// The FP32 sum of squares can round the normalizer differently.
			if math.Abs(float64(got[i]-want[i])) > math.Abs(float64(want[i]))/128 {
				t.Fatalf("weighted element %d = %g, want %g", i, got[i], want[i])
			}
		}
		for row := range 3 {
			for i := range 2304 {
				want := plain[row*2304+i]
				for source := range 2 {
					if v := pair[(row*2+source)*2304+i]; math.Abs(float64(v-want)) > math.Abs(float64(want))/128 {
						t.Fatalf("row %d source %d element %d = %g, want %g", row, source, i, v, want)
					}
				}
			}
		}
	})
}
