package inkling2

import (
	"fmt"
	"math"
	"slices"
	"testing"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlx/mlxtest"
)

type gateUpInputs struct {
	inner, xmax, wmax, denominator, ids, scores *mlx.Array
}

func gateUpInputsFor(rows, topK int) gateUpInputs {
	inner := make([]float32, rows*topK*2304)
	for i := range inner {
		inner[i] = float32((i*397)%15011-7505) / 19
	}
	ids := make([]uint32, rows*topK)
	scores := make([]float32, len(ids))
	for i := range ids {
		ids[i] = uint32((i * 17) % 128)
		scores[i] = float32((i*31)%123+1) / 31
	}
	xmax := make([]float32, rows)
	wmax := make([]float32, 128)
	for i := range xmax {
		xmax[i] = float32(i + 1)
	}
	for i := range wmax {
		wmax[i] = float32(i + 1)
	}
	return gateUpInputs{
		inner:       mlx.FromValues(inner, rows, topK, 1, 2304),
		xmax:        mlx.FromValues(xmax, rows, 1, 1, 1),
		wmax:        mlx.FromValues(wmax, 128),
		denominator: mlx.FromValue(float32(2688)),
		ids:         mlx.FromValues(ids, rows, topK),
		scores:      mlx.FromValues(scores, rows, topK),
	}
}

func (in gateUpInputs) eager() *mlx.Array {
	outer := mlx.Div(mlx.Mul(in.xmax, mlx.Take(in.wmax, in.ids, 0).ExpandDims(-1).ExpandDims(-1)), mlx.MulScalar(in.denominator, 1536))
	y := mlx.Mul(in.inner, outer).AsType(mlx.DTypeBFloat16).AsType(mlx.DTypeFloat32)
	gate := y.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(), mlx.Slice(0, 2304, 2))
	up := y.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(), mlx.Slice(1, 2304, 2))
	gateFactor := mlx.MulScalar(mlx.DivScalar(gate, 4).Tanh(), 4)
	upFactor := mlx.MulScalar(mlx.DivScalar(up, 25).Tanh(), 25)
	return mlx.Mul(mlx.Mul(mlx.Mul(gateFactor, gate.Sigmoid()), upFactor), in.scores.ExpandDims(-1).ExpandDims(-1)).AsType(mlx.DTypeBFloat16)
}

func TestNVFP4SwiGLUQuantize(t *testing.T) {
	for _, tc := range []struct{ rows, topK int }{{2, 8}, {16, 1}} {
		t.Run(fmt.Sprintf("rows%d/top%d", tc.rows, tc.topK), func(t *testing.T) {
			mlxtest.Run(t, func(t *mlxtest.T) {
				mlx.Scoped(func() {
					if !mlx.MetalIsAvailable() {
						t.Skip("Metal epilogue required")
					}
					in := gateUpInputsFor(tc.rows, tc.topK)
					wantValues, wantMax := mlx.QuantizeNVFP4Activation4Over6(in.eager(), nvfp4ActivationDenominator)
					inner := in.inner.Reshape(tc.rows*tc.topK, 2304)
					values, maxima := mlx.NVFP4SwiGLUQuantize(inner, in.xmax, in.wmax, in.denominator, in.ids.Reshape(-1),
						in.scores.Reshape(-1), tc.topK, 4, 25, nvfp4ActivationDenominator, mlx.DTypeFloat32)
					if values == nil {
						t.Fatal("kernel did not run")
					}
					got, want := values.Floats(), wantValues.Floats()
					same := 0
					for i := range got {
						if got[i] == want[i] {
							same++
						}
					}
					// The fused kernel's transcendental functions can move a value
					// across a BF16 rounding boundary, which can change its block.
					if same < len(got)*999/1000 {
						t.Errorf("%d of %d quantized values match the eager graph", same, len(got))
					}
					for i, m := range maxima.Floats() {
						w := wantMax.Floats()[i]
						if math.Abs(float64(m-w)) > math.Abs(float64(w))/128 {
							t.Errorf("route %d maximum %g, want %g", i, m, w)
						}
					}
				})
			})
		})
	}
}

func TestGateUpFloatBank(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		mlx.Scoped(func() {
			weight := mlx.Arange(0, 2*32*64, 1, mlx.DTypeFloat32).Reshape(2, 32, 64)
			e := routedExperts{GateUp: expertBank{Weight: weight}, topK: 1, gateCap: 2, upCap: 7}
			flat := mlx.AddScalar(mlx.Zeros(mlx.DTypeFloat32, 2, 64), 1)
			ids := mlx.FromValues([]uint32{1, 0}, 2, 1)
			scores := mlx.FromValues([]float32{0.25, 0.75}, 2, 1)
			got := e.gateUpInput(flat, ids, scores)
			pair := mlx.GatherMM(flat.Reshape(2, 1, 1, 64), weight.Transpose(0, 2, 1), nil, ids, false)
			gate, up := splitGateUp(pair)
			want := weightedSwiGLU(gate, up, scores, 2, 7, mlx.DTypeFloat32)
			if !slices.Equal(got.Floats(), want.Floats()) {
				t.Error("float bank changed on the existing GatherMM route")
			}
		})
	})
}
