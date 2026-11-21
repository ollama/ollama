package mlx

import (
	"fmt"
	"math"
	"testing"

	"github.com/ollama/ollama/mlx/mlxthread/mlxthreadtest"
)

func TestConnectionMixMatchesGraph(t *testing.T) {
	withMLXThread(t, func(t *mlxthreadtest.T) {
		const normEps, mixEps = 1e-6, 1e-6
		for _, shape := range [][4]int{{1, 1, 4, 20}, {1, 7, 4, 20}, {2, 300, 4, 20}, {1, 3, 2, 1}, {1, 5, 8, 3}} {
			Scoped(func() {
				b, l, copies, iterations := shape[0], shape[1], shape[2], shape[3]
				width := (copies + 2) * copies
				projected := make([]float32, b*l*width)
				for i := range projected {
					projected[i] = float32(3 * math.Sin(float64(i)*0.37))
				}
				variance := make([]float32, b*l)
				for i := range variance {
					variance[i] = float32(0.5 + 0.25*math.Cos(float64(i)*0.11))
				}
				base := make([]float32, width)
				for i := range base {
					base[i] = float32(i%5-2) * .13
				}
				p := FromValues(projected, b, l, width)
				v := FromValues(variance, b, l, 1)
				s := FromValues([]float32{.7, 1.2, .9}, 3)
				o := FromValues(base, width)

				pre, post, residual := ConnectionMix(p, v, s, o, copies, iterations, normEps, mixEps)
				if (MetalIsAvailable() || CUDAIsAvailable()) && residual.name != "CONNECTION_RESIDUAL" {
					t.Fatal("connection-mix kernel was skipped")
				}
				wantPre, wantPost, wantResidual := connectionMixGraph(p, v, s, o, FromValues([]float32{normEps, mixEps}, 2), copies, iterations)
				Eval(pre, post, residual, wantPre, wantPost, wantResidual)
				for _, c := range []struct {
					name      string
					got, want *Array
				}{{"pre", pre, wantPre}, {"post", post, wantPost}, {"residual", residual, wantResidual}} {
					if fmt.Sprint(c.got.Dims()) != fmt.Sprint(c.want.Dims()) {
						t.Fatalf("%v %s shape %v, want %v", shape, c.name, c.got.Dims(), c.want.Dims())
					}
					got, want := c.got.Floats(), c.want.Floats()
					exact := copies == 4 && iterations == 20
					for i := range got {
						if (exact && math.Float32bits(got[i]) != math.Float32bits(want[i])) ||
							(!exact && (math.IsNaN(float64(got[i])) || math.Abs(float64(got[i]-want[i])) > 2e-6+2e-6*math.Abs(float64(want[i])))) {
							t.Fatalf("%v %s[%d] = %g, want %g", shape, c.name, i, got[i], want[i])
						}
					}
				}
			})
		}
	})
}
