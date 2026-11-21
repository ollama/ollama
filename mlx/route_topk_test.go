package mlx

import (
	"math"
	"testing"

	"github.com/ollama/ollama/mlx/mlxthread/mlxthreadtest"
)

func TestRouteSigmoidTopK(t *testing.T) {
	withMLXThread(t, func(t *mlxthreadtest.T) {
		if !MetalIsAvailable() {
			t.Skip("requires Metal")
		}
		const rows, experts, topK, scale = 3, 128, 8, 2.5
		logits := patternArray(DTypeFloat32, []int{rows, experts}, 0, 0.05, 37, 101)
		bias := patternArray(DTypeFloat32, []int{experts}, 0, 0.001, 11, 29)
		indices, weights := RouteSigmoidTopK(logits, bias, topK, scale)
		if indices == nil {
			t.Fatal("kernel did not run")
		}
		l, b := logits.Floats(), bias.Floats()
		ids, ws := indices.AsType(DTypeInt32).Ints(), weights.Floats()
		sigmoid := func(x float32) float64 { return 1 / (1 + math.Exp(-float64(x))) }
		for r := range rows {
			// Highest biased score first, ties to the lower index.
			taken := map[int]bool{}
			var total float64
			for k := range topK {
				best := -1
				for e := range experts {
					s := sigmoid(l[r*experts+e]) + float64(b[e])
					if !taken[e] && (best < 0 || s > sigmoid(l[r*experts+best])+float64(b[best])) {
						best = e
					}
				}
				taken[best] = true
				if int(ids[r*topK+k]) != best {
					t.Fatalf("row %d choice %d = %d, want %d", r, k, ids[r*topK+k], best)
				}
				total += sigmoid(l[r*experts+best])
			}
			for k := range topK {
				want := sigmoid(l[r*experts+int(ids[r*topK+k])]) / total * scale
				if math.Abs(float64(ws[r*topK+k])-want) > 1e-5 {
					t.Fatalf("row %d weight %d = %g, want %g", r, k, ws[r*topK+k], want)
				}
			}
		}
	})
}
