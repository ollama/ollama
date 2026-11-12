package mlx

import (
	"math"
	"testing"

	"github.com/ollama/ollama/mlx/mlxthread/mlxthreadtest"
)

func TestRotateTailMatchesGraph(t *testing.T) {
	withMLXThread(t, func(t *mlxthreadtest.T) {
		for _, dtype := range []DType{DTypeFloat32, DTypeBFloat16} {
			for _, shape := range [][5]int{{1, 1, 8, 64, 16}, {2, 5, 3, 48, 24}, {1, 7, 1, 32, 32}, {1, 1100, 64, 16, 8}} {
				for _, inverse := range []bool{false, true} {
					Scoped(func() {
						b, l, h, w, rotary := shape[0], shape[1], shape[2], shape[3], shape[4]
						data := make([]float32, b*l*h*w)
						for i := range data {
							data[i] = float32(math.Sin(float64(i)*0.31) * 2)
						}
						x := FromValues(data, b, l, h, w).AsType(dtype)
						positions := make([]int32, b*l)
						for i := range positions {
							positions[i] = int32(i*997 + 13)
						}
						frequencies := make([]float32, rotary/2)
						for i := range frequencies {
							frequencies[i] = float32(math.Pow(10000, -float64(2*i)/float64(rotary)))
						}
						p := FromValues(positions, b, l)
						got := RotateTail(x, p, frequencies, inverse)
						if (MetalIsAvailable() || CUDAIsAvailable()) && got.name != "ROTATE_TAIL" {
							t.Fatal("rotate-tail kernel was skipped")
						}
						want := rotateTailGraph(x, p, FromValues(frequencies, len(frequencies)), inverse)
						g, wv := got.AsType(DTypeFloat32), want.AsType(DTypeFloat32)
						Eval(g, wv)
						if got.DType() != dtype {
							t.Fatalf("dtype %v, want %v", got.DType(), dtype)
						}
						gs, ws := g.Floats(), wv.Floats()
						for i := range gs {
							if math.Float32bits(gs[i]) != math.Float32bits(ws[i]) {
								t.Fatalf("%v %v inverse=%v [%d] = %g, want %g", dtype, shape, inverse, i, gs[i], ws[i])
							}
						}
					})
				}
			}
		}
	})
}

func TestRotateTailInverseRestores(t *testing.T) {
	withMLXThread(t, func(t *mlxthreadtest.T) {
		data := make([]float32, 2*3*4*16)
		for i := range data {
			data[i] = float32(math.Cos(float64(i) * 0.7))
		}
		x := FromValues(data, 2, 3, 4, 16)
		p := FromValues([]int32{5, 6, 7, 900, 901, 902}, 2, 3)
		frequencies := []float32{1, 0.1, 0.01}
		back := RotateTail(RotateTail(x, p, frequencies, false), p, frequencies, true)
		Eval(back)
		for i, v := range back.Floats() {
			if math.Abs(float64(v-data[i])) > 1e-5 {
				t.Fatalf("[%d] = %g, want %g", i, v, data[i])
			}
		}
	})
}
