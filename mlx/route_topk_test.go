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
		logits := patternArray(DTypeFloat32, []int{3, 128}, 0, 0.05, 37, 101)
		bias := patternArray(DTypeFloat32, []int{128}, 0, 0.001, 11, 29)
		checkRouteSigmoidTopK(t, logits, bias, 8, 2.5)
		// Seven logit levels over 256 experts tie many scores.
		checkRouteSigmoidTopK(t, patternArray(DTypeFloat32, []int{5, 256}, 0, 0.5, 3, 7), nil, 6, 1)
		// One thread per expert: a count that is not a multiple of 32.
		checkRouteSigmoidTopK(t, patternArray(DTypeFloat32, []int{2, 72}, 0, 0.05, 13, 61), patternArray(DTypeFloat32, []int{72}, 0, 0.001, 5, 17), 4, 1)

		// NaN logits still choose distinct, valid experts.
		values := make([]float32, 2*128)
		for i := range values {
			values[i] = float32(math.NaN())
		}
		values[128+5] = 1
		indices, _ := RouteSigmoidTopK(FromValues(values, 2, 128), nil, 8, 1)
		if indices == nil {
			t.Fatal("kernel did not run")
		}
		ids := indices.AsType(DTypeInt32).Ints()
		for r := range 2 {
			seen := map[int32]bool{}
			for _, id := range ids[r*8 : (r+1)*8] {
				if id < 0 || id >= 128 || seen[id] {
					t.Fatalf("row %d chose %v from NaN logits", r, ids[r*8:(r+1)*8])
				}
				seen[id] = true
			}
		}
		if ids[8] != 5 {
			t.Errorf("row 1 first choice %d, want the one finite logit 5", ids[8])
		}

		// Gates that all underflow still share the weight. Row 0 ties every
		// expert at -100; row 1 spreads the chosen logits from -100 to -107.
		low := make([]float32, 2*16)
		for i := range 16 {
			low[i] = -100
			low[16+i] = -100 - float32(i)
		}
		_, weights := RouteSigmoidTopK(FromValues(low, 2, 16), nil, 8, 2)
		if weights == nil {
			t.Fatal("kernel did not run")
		}
		got := weights.Floats()
		var total float64
		for k := range 8 {
			total += math.Exp(-float64(k))
		}
		for k := range 8 {
			if got[k] != 0.25 {
				t.Errorf("tied underflow weight %d = %v, want 0.25", k, got[k])
			}
			want := 2 * math.Exp(-float64(k)) / total
			if math.Abs(float64(got[8+k])-want) > 1e-6*want {
				t.Errorf("spread underflow weight %d = %v, want %v", k, got[8+k], want)
			}
		}

		// Only one chosen gate underflows: sigmoid(-87) stays normal while
		// sigmoid(-89) does not, and the weights must still split about 88/12.
		_, mixed := RouteSigmoidTopK(FromValues([]float32{-87, -89, -120, -120}, 1, 4), nil, 2, 1)
		if mixed == nil {
			t.Fatal("kernel did not run")
		}
		split := mixed.Floats()
		for k, want := range []float64{1 / (1 + math.Exp(-2)), 1 / (1 + math.Exp(2))} {
			if math.Abs(float64(split[k])-want) > 1e-6 {
				t.Errorf("mixed underflow weight %d = %v, want %v", k, split[k], want)
			}
		}
	})
}

func checkRouteSigmoidTopK(t *mlxthreadtest.T, logits, bias *Array, topK int, scale float64) {
	t.Helper()
	rows, experts := logits.Dim(0), logits.Dim(1)
	indices, weights := RouteSigmoidTopK(logits, bias, topK, float32(scale))
	if indices == nil {
		t.Fatal("kernel did not run")
	}
	l := logits.Floats()
	b := make([]float32, experts)
	if bias != nil {
		b = bias.Floats()
	}
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
}
