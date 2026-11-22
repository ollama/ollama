package mlx

import (
	"fmt"
	"math"
	"testing"

	"github.com/ollama/ollama/mlx/mlxthread/mlxthreadtest"
)

// TestNVFP4SwiGLUQuantizeWidths covers each per-thread value count: from M3
// on, blocks that span 8, 4 and 2 lanes for a few routes, and one lane for
// many routes or wide rows.
func TestNVFP4SwiGLUQuantizeWidths(t *testing.T) {
	const stride, maxProduct = 2, 1536
	// A cap of zero leaves the gate or up projection uncapped, as the graph's softcap does.
	softcap := func(x *Array, c float32) *Array {
		if c <= 0 {
			return x
		}
		return MulScalar(DivScalar(x, c).Tanh(), c)
	}
	for _, tc := range []struct {
		routes, width  int
		gateCap, upCap float32
	}{{6, 1152, 4, 25}, {6, 2304, 4, 25}, {6, 3072, 4, 25}, {6, 4864, 4, 25}, {136, 1152, 4, 25}, {6, 2304, 0, 0}} {
		gateCap, upCap := tc.gateCap, tc.upCap
		t.Run(fmt.Sprintf("routes%d/width%d/caps%g,%g", tc.routes, tc.width, gateCap, upCap), func(t *testing.T) {
			withMLXThread(t, func(t *mlxthreadtest.T) {
				if !MetalIsAvailable() {
					t.Skip("requires Metal")
				}
				routes, width := tc.routes, tc.width
				inner := patternArray(DTypeFloat32, []int{routes, 2 * width}, 0, 1.0/19, 397, 15011)
				xmax := patternArray(DTypeFloat32, []int{routes / stride}, 4, 0.5, 3, 13)
				maxima := patternArray(DTypeFloat32, []int{16}, 9, 0.5, 5, 13)
				ids := make([]uint32, routes)
				for i := range ids {
					ids[i] = uint32(i*7) % 16
				}
				idArray := FromValues(ids, routes)
				scores := patternArray(DTypeFloat32, []int{routes}, 1.2, 0.1, 7, 23)
				denominator := FromValue(float32(2688))
				values, amax := NVFP4SwiGLUQuantize(inner, xmax, maxima, denominator, idArray, scores, stride,
					gateCap, upCap, maxProduct, DTypeFloat32)
				if values == nil {
					t.Fatal("kernel did not run")
				}

				scale := Div(Mul(Take(xmax, FloorDivideScalar(Arange(0, float64(routes), 1, DTypeInt32), stride), 0), Take(maxima, idArray, 0)),
					MulScalar(denominator, maxProduct)).Reshape(routes, 1)
				pairs := Mul(inner, scale).AsType(DTypeBFloat16).AsType(DTypeFloat32).Reshape(routes, width, 2)
				gate := pairs.Slice(Slice(), Slice(), Slice(0, 1)).Reshape(routes, width)
				up := pairs.Slice(Slice(), Slice(), Slice(1, 2)).Reshape(routes, width)
				activated := Mul(Mul(softcap(gate, gateCap), gate.Sigmoid()), softcap(up, upCap))
				activated = Mul(activated, scores.Reshape(routes, 1)).AsType(DTypeBFloat16)
				wantValues, wantMax := QuantizeNVFP4Activation4Over6(activated, maxProduct)

				got, want := values.Floats(), wantValues.AsType(DTypeFloat32).Floats()
				same := 0
				for i := range got {
					if got[i] == want[i] {
						same++
					}
				}
				// Transcendental functions can move a value across a BF16
				// rounding boundary, which can change its block.
				if same < len(got)*999/1000 {
					t.Errorf("%d of %d quantized values match the graph", same, len(got))
				}
				wantMaxima := wantMax.AsType(DTypeFloat32).Floats()
				for i, m := range amax.Floats() {
					if w := wantMaxima[i]; math.Abs(float64(m-w)) > math.Abs(float64(w))/128 {
						t.Errorf("route %d maximum %g, want %g", i, m, w)
					}
				}
			})
		})
	}
}
