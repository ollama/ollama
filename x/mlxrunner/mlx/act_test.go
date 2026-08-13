package mlx

import (
	"math"
	"testing"

	"github.com/ollama/ollama/x/internal/mlxthread"
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
			withMLXThread(t, func() {
				EnableCompile()
				input := FromValues(values, len(values)).AsType(tt.dtype)
				Pin(input)

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
				Unpin(input)
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
	thread, err := mlxthread.Start("mlx-gelu-benchmark", func() error {
		if err := CheckInit(); err != nil {
			return err
		}
		if GPUIsAvailable() {
			SetDefaultDeviceGPU()
		}
		EnableCompile()
		return nil
	})
	if err != nil {
		b.Skipf("MLX not available: %v", err)
	}
	defer func() {
		if err := thread.Stop(b.Context(), func() {
			Sweep()
			ClearCache()
			resetDefaultStreamCache()
		}); err != nil {
			b.Fatal(err)
		}
	}()

	if err := thread.Do(b.Context(), func() error {
		input := AddScalar(Zeros(DTypeBFloat16, 1, 4096, 8192), 1)
		Eval(input)
		Pin(input)
		defer Unpin(input)

		warmup := fn(input)
		Eval(warmup)
		Sweep()

		b.ResetTimer()
		for range b.N {
			output := fn(input)
			Eval(output)
			Sweep()
		}
		return nil
	}); err != nil {
		b.Fatal(err)
	}
}

func TestReLUSquared(t *testing.T) {
	var got []float32
	withMLXThread(t, func() {
		x := FromValues([]float32{-2, -0, 0.5, 2}, 4)
		Pin(x)
		defer Unpin(x)

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
