package mlx

import (
	"math"
	"testing"

	"github.com/ollama/ollama/mlx/mlxthread/mlxthreadtest"
)

func TestGELUCompiledMatchesEager(t *testing.T) {
	values := []float32{-6, -2, -0.5, 0, 0.5, 2, 6}
	tests := []struct {
		name      string
		dtype     DType
		tolerance float32
	}{
		{name: "float32", dtype: DTypeFloat32, tolerance: 1e-6},
		{name: "bfloat16", dtype: DTypeBFloat16, tolerance: 1e-2},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			withMLXThread(t, func(t *mlxthreadtest.T) {
				EnableCompile()
				input := FromValues(values, len(values)).AsType(tt.dtype)

				want := gelu(input)
				got := GELU(input)
				wantF32 := want.AsType(DTypeFloat32)
				gotF32 := got.AsType(DTypeFloat32)
				Eval(wantF32, gotF32)

				wantValues := wantF32.Floats()
				gotValues := gotF32.Floats()
				for i := range wantValues {
					if delta := float32(math.Abs(float64(gotValues[i] - wantValues[i]))); delta > tt.tolerance {
						t.Fatalf("%s GELU[%d] = %v, want %v (delta %v)", tt.name, i, gotValues[i], wantValues[i], delta)
					}
				}
			})
		})
	}
}

func BenchmarkGELUEager(b *testing.B) {
	benchmarkGELU(b, gelu)
}

func BenchmarkGELUCompiled(b *testing.B) {
	benchmarkGELU(b, GELU)
}

func benchmarkGELU(b *testing.B, fn func(*Array) *Array) {
	thread := mlxTestThread(b)
	if err := thread.Do(b.Context(), func() error {
		EnableCompile()
		input := AddScalar(Zeros(DTypeBFloat16, 1, 4096, 8192), 1)
		Eval(input)
		defer ClearCache()

		Scoped(func() { Eval(fn(input)) })

		b.ResetTimer()
		for range b.N {
			Scoped(func() { Eval(fn(input)) })
		}
		return nil
	}); err != nil {
		b.Fatal(err)
	}
}

func TestSwiGLUScaledMatchesSeparateScaling(t *testing.T) {
	for _, tt := range []struct {
		name               string
		gateScale, upScale []float32
	}{
		{name: "no scales"},
		{name: "scalar scales", gateScale: []float32{0.75}, upScale: []float32{0.75}},
		{name: "gate scale only", gateScale: []float32{0.75}},
		{name: "up scale only", upScale: []float32{0.75}},
		{name: "per-output scales", gateScale: []float32{0.5, 0.75, 1.25, 1.5}, upScale: []float32{1.5, 1.25, 0.75, 0.5}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			withMLXThread(t, func(t *mlxthreadtest.T) {
				EnableCompile()
				gate := FromValues([]float32{-3.25, -1.5, -0.25, 0.5, 1.75, 3, 4.5, 6}, 2, 4).AsType(DTypeBFloat16)
				up := FromValues([]float32{2.5, -2, 1.25, -0.75, 0.125, 1.5, -3.5, 5}, 2, 4).AsType(DTypeBFloat16)
				storedScale := func(factors []float32) *Array {
					if factors == nil {
						return nil
					}
					values := make([]float32, len(factors))
					for i := range factors {
						values[i] = factors[i] * float32(Nvfp4MaxProduct)
					}
					if len(values) == 1 {
						return FromValue(values[0])
					}
					return FromValues(values, len(values))
				}
				gateScale, upScale := storedScale(tt.gateScale), storedScale(tt.upScale)

				wantGate, wantUp := gate, up
				if gateScale != nil {
					wantGate = scaleAndCast(wantGate, gateScale)
				}
				if upScale != nil {
					wantUp = scaleAndCast(wantUp, upScale)
				}
				want := SwiGLU(wantGate, wantUp)
				got := SwiGLUScaled(gate, gateScale, up, upScale)
				wantF32, gotF32 := want.AsType(DTypeFloat32), got.AsType(DTypeFloat32)
				Eval(wantF32, gotF32)

				wantValues, gotValues := wantF32.Floats(), gotF32.Floats()
				for i := range wantValues {
					if gotValues[i] != wantValues[i] {
						t.Fatalf("SwiGLUScaled()[%d] = %v, want %v", i, gotValues[i], wantValues[i])
					}
				}
			})
		})
	}
}

func BenchmarkSwiGLUSeparateScaling(b *testing.B) {
	benchmarkSwiGLUScaling(b, func(gate, gateScale, up, upScale *Array) *Array {
		return SwiGLU(scaleAndCast(gate, gateScale), scaleAndCast(up, upScale))
	})
}

func BenchmarkSwiGLUFusedScaling(b *testing.B) {
	benchmarkSwiGLUScaling(b, SwiGLUScaled)
}

func benchmarkSwiGLUScaling(b *testing.B, fn func(gate, gateScale, up, upScale *Array) *Array) {
	thread := mlxTestThread(b)
	if err := thread.Do(b.Context(), func() error {
		EnableCompile()
		// qwen3.8:27b-nvfp4's dense MLP output at a 2048-token prompt.
		gate := AddScalar(Zeros(DTypeBFloat16, 1, 2048, 17408), 0.5)
		up := AddScalar(Zeros(DTypeBFloat16, 1, 2048, 17408), 1.5)
		gateScale := FromValue(float32(Nvfp4MaxProduct) * 0.75)
		upScale := FromValue(float32(Nvfp4MaxProduct) * 1.25)
		Eval(gate, up, gateScale, upScale)
		defer ClearCache()

		Scoped(func() { Eval(fn(gate, gateScale, up, upScale)) })

		b.ResetTimer()
		for range b.N {
			Scoped(func() { Eval(fn(gate, gateScale, up, upScale)) })
		}
		return nil
	}); err != nil {
		b.Fatal(err)
	}
}

func TestReLUSquared(t *testing.T) {
	var got []float32
	withMLXThread(t, func(t *mlxthreadtest.T) {
		x := FromValues([]float32{-2, -0, 0.5, 2}, 4)

		y := ReLUSquared(x)
		Eval(y)
		got = append(got, y.Floats()...)
	})

	want := []float32{0, 0, 0.25, 4}
	if len(got) != len(want) {
		t.Fatalf("got %d values, want %d", len(got), len(want))
	}
	for i, v := range got {
		if v != want[i] {
			t.Errorf("got[%d]=%v want %v", i, v, want[i])
		}
	}
}
