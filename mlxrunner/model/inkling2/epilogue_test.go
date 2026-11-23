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
	if rows >= 2 {
		// Two captured routes put the final weighted FP32 product exactly
		// halfway between adjacent BF16 values on Metal.
		inner[0], inner[1] = 1809456, -2859888
		inner[topK*2304], inner[topK*2304+1] = 888816, 1665824
		xmax[0], xmax[1] = 1.640625, 1.7734375
		wmax[ids[0]], wmax[ids[topK]] = 0.14305396, 0.11680993
		scores[0], scores[topK] = 0.53709847, 0.4751412
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

func (in gateUpInputs) compiled() *mlx.Array {
	gate, up := splitGateUp(in.inner)
	return compiledGateUpEpilogue(gate, up, in.xmax, in.wmax, in.denominator, in.ids, in.scores)[0]
}

func TestNVFP4GateUpEpilogue(t *testing.T) {
	for _, tc := range []struct{ rows, topK int }{{2, 8}, {16, 1}} {
		t.Run(fmt.Sprintf("rows%d/top%d", tc.rows, tc.topK), func(t *testing.T) {
			mlxtest.Run(t, func(t *mlxtest.T) {
				mlx.Scoped(func() {
					if !mlx.MetalIsAvailable() {
						t.Skip("Metal epilogue required")
					}
					mlx.EnableCompile()
					in := gateUpInputsFor(tc.rows, tc.topK)
					want := in.eager().AsType(mlx.DTypeFloat32).Floats()
					got := in.compiled().AsType(mlx.DTypeFloat32).Floats()
					if !slices.Equal(got, want) {
						t.Error("compiled gate/up result differs from the eager graph")
					}
					for i, bits := range []uint32{0xbb9a0000, 0x3a6e0000} {
						if actual := math.Float32bits(got[i*tc.topK*1152]); actual != bits {
							t.Errorf("route %d rounded to %08x, want %08x", i, actual, bits)
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
			got := e.gateUpInput(flat, ids, nil, scores, false)
			pair := mlx.GatherMM(flat.Reshape(2, 1, 1, 64), weight.Transpose(0, 2, 1), nil, ids, false)
			gate, up := splitGateUp(pair)
			want := weightedSwiGLU(gate, up, scores, 2, 7, mlx.DTypeFloat32)
			if !slices.Equal(got.Floats(), want.Floats()) {
				t.Error("float bank changed on the existing GatherMM route")
			}
		})
	})
}

func BenchmarkInklingGateUpEpilogue(b *testing.B) {
	worker := mlxtest.Worker(b)
	var metal bool
	if err := worker.Do(b.Context(), func() error { metal = mlx.MetalIsAvailable(); return nil }); err != nil {
		b.Fatal(err)
	}
	if !metal {
		b.Skip("Metal epilogue required")
	}
	for _, tc := range []struct{ rows, topK int }{{1, 8}, {8, 8}, {512, 1}} {
		for _, mode := range []string{"eager", "compiled"} {
			b.Run(fmt.Sprintf("rows%d/top%d/%s", tc.rows, tc.topK, mode), func(b *testing.B) {
				err := worker.Do(b.Context(), func() error {
					mlx.Scoped(func() {
						mlx.EnableCompile()
						in := gateUpInputsFor(tc.rows, tc.topK)
						mlx.Eval(in.inner, in.xmax, in.wmax, in.denominator, in.ids, in.scores)
						op := in.eager
						if mode == "compiled" {
							op = in.compiled
						}
						eval := func() { mlx.Eval(op()) }
						mlx.Scoped(eval)
						b.ResetTimer()
						for range b.N {
							mlx.Scoped(eval)
						}
						b.StopTimer()
					})
					return nil
				})
				if err != nil {
					b.Fatal(err)
				}
			})
		}
	}
}
