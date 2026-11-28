package mlx_test

import (
	"math"
	"math/rand"
	"testing"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlx/mlxtest"
)

func olalaRandom(rng *rand.Rand, std float64, shape ...int) *mlx.Array {
	n := 1
	for _, d := range shape {
		n *= d
	}
	v := make([]float32, n)
	for i := range v {
		v[i] = float32(std * rng.NormFloat64())
	}
	return mlx.FromValues(v, shape...)
}

func olalaMaxRelErr(t *mlxtest.T, got, want *mlx.Array) float64 {
	mlx.Eval(got, want)
	g, w := got.Floats(), want.Floats()
	if len(g) != len(w) {
		t.Fatalf("got %d values, want %d", len(g), len(w))
	}
	var maxDiff, maxRef float64
	for i := range w {
		maxDiff = math.Max(maxDiff, math.Abs(float64(g[i]-w[i])))
		maxRef = math.Max(maxRef, math.Abs(float64(w[i])))
	}
	return maxDiff / maxRef
}

// olalaGeodesicRef is the graph form of the geodesic residual update.
func olalaGeodesicRef(x, g *mlx.Array, scale, bias float32, layer int) *mlx.Array {
	s := func(v float32) *mlx.Array { return mlx.FromValue(v) }
	xn2 := mlx.Maximum(mlx.Sum(mlx.Mul(x, x), -1, true), s(1e-12))
	proj := mlx.Div(mlx.Sum(mlx.Mul(x, g), -1, true), xn2)
	grad := mlx.Sub(g, mlx.Mul(proj, x))
	tn := mlx.Maximum(mlx.Sum(mlx.Mul(grad, grad), -1, true).Sqrt(), s(1e-8))
	r := mlx.Maximum(xn2.Sqrt(), s(1e-6))
	clamp := s(math.Pi / 4)
	theta := mlx.Minimum(mlx.Div(tn, r), clamp)
	theta = mlx.Minimum(mlx.DivScalar(mlx.AddScalar(mlx.MulScalar(theta, scale), bias), float32(layer+1)), clamp)
	return mlx.Add(mlx.Mul(x, mlx.Cos(theta)), mlx.Mul(mlx.Mul(grad, mlx.Div(r, tn)), mlx.Sin(theta)))
}

func TestOlalaGEMVFormats(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		if !mlx.OlalaKernelsAvailable() {
			t.Skip("Metal is not available")
		}
		rng := rand.New(rand.NewSource(1))
		const K, rows = 512, 200
		for _, tc := range []struct {
			name string
			bits int
			dt   mlx.DType
		}{
			{"f32", 0, mlx.DTypeFloat32},
			{"bf16", 0, mlx.DTypeBFloat16},
			{"q8", 8, mlx.DTypeBFloat16},
			{"q4", 4, mlx.DTypeBFloat16},
		} {
			for _, T := range []int{1, 3} {
				w := olalaRandom(rng, 0.05, rows, K).AsType(tc.dt)
				x := olalaRandom(rng, 1, T, K)
				var ow *mlx.OlalaWeight
				var dense *mlx.Array
				if tc.bits > 0 {
					q, s, b := mlx.Quantize(w, 64, tc.bits, "affine")
					ow = mlx.NewOlalaWeight(q, s, b, tc.bits, 64)
					// f32 scales: the kernel dequantizes in f32, MLX would round to bf16
					dense = mlx.Dequantize(q, s.AsType(mlx.DTypeFloat32), b.AsType(mlx.DTypeFloat32), 64, tc.bits, "affine", nil)
				} else {
					ow = mlx.NewOlalaWeight(w, nil, nil, 0, 0)
					dense = w
				}
				want := mlx.Matmul(x, mlx.Transpose(dense.AsType(mlx.DTypeFloat32), 1, 0))
				got := mlx.OlalaGEMV([]*mlx.Array{x}, []*mlx.OlalaWeight{ow}, T, K, mlx.OlalaGEMVOptions{})[0]
				if e := olalaMaxRelErr(t, got, want); e > 1e-4 {
					t.Fatalf("%s T=%d: max relative error %g", tc.name, T, e)
				}
			}
		}
	})
}

func TestOlalaGEMVPrologues(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		if !mlx.OlalaKernelsAvailable() {
			t.Skip("Metal is not available")
		}
		rng := rand.New(rand.NewSource(2))
		const K, T, layer = 256, 1, 3
		const scale, bias = float32(1.1), float32(0.05)
		wa := olalaRandom(rng, 0.05, 96, K)
		wb := olalaRandom(rng, 0.05, 64, K).AsType(mlx.DTypeBFloat16)
		ws := []*mlx.OlalaWeight{mlx.NewOlalaWeight(wa, nil, nil, 0, 0), mlx.NewOlalaWeight(wb, nil, nil, 0, 0)}
		x := olalaRandom(rng, 1, T, K)
		g := olalaRandom(rng, 0.3, T, K)
		geo := mlx.FromValues([]float32{scale, bias, layer + 1}, 3)
		dense := func(xs *mlx.Array) []*mlx.Array {
			return []*mlx.Array{
				mlx.Matmul(xs, mlx.Transpose(wa, 1, 0)),
				mlx.Matmul(xs, mlx.Transpose(wb.AsType(mlx.DTypeFloat32), 1, 0)),
			}
		}

		// norm: x / |x|
		xn := mlx.Div(x, mlx.Sum(mlx.Mul(x, x), -1, true).Sqrt())
		out := mlx.OlalaGEMV([]*mlx.Array{x}, ws, T, K, mlx.OlalaGEMVOptions{Prologue: mlx.OlalaNorm})
		for i, want := range append(dense(xn), xn) {
			if e := olalaMaxRelErr(t, out[i], want); e > 1e-4 {
				t.Fatalf("norm output %d: max relative error %g", i, e)
			}
		}

		// geo: geodesic(x, g) computed in the prologue
		xg := olalaGeodesicRef(x, g, scale, bias, layer)
		out = mlx.OlalaGEMV([]*mlx.Array{x, g, geo}, ws, T, K, mlx.OlalaGEMVOptions{Prologue: mlx.OlalaGeo})
		for i, want := range append(dense(xg), xg) {
			if e := olalaMaxRelErr(t, out[i], want); e > 1e-4 {
				t.Fatalf("geo output %d: max relative error %g", i, e)
			}
		}

		// stats from a GEMV, consumed by the geos prologue of the next one
		wsq := mlx.NewOlalaWeight(olalaRandom(rng, 0.05, K, 128), nil, nil, 0, 0)
		y := olalaRandom(rng, 1, T, 128)
		r := mlx.OlalaGEMV([]*mlx.Array{y}, []*mlx.OlalaWeight{wsq}, T, 128, mlx.OlalaGEMVOptions{StatsX: x})
		upd, gst := r[0], r[1]
		out = mlx.OlalaGEMV([]*mlx.Array{x, upd, geo, gst}, ws, T, K, mlx.OlalaGEMVOptions{Prologue: mlx.OlalaGeos})
		xu := olalaGeodesicRef(x, upd, scale, bias, layer)
		for i, want := range append(dense(xu), xu) {
			if e := olalaMaxRelErr(t, out[i], want); e > 1e-3 {
				t.Fatalf("geos output %d: max relative error %g", i, e)
			}
		}
		if e := olalaMaxRelErr(t, mlx.OlalaGeodesic(x, upd, geo, T, K), xu); e > 1e-4 {
			t.Fatalf("standalone geodesic: max relative error %g", e)
		}
	})
}
