package mlx

import (
	"fmt"
	"math"
	"slices"
	"testing"

	"github.com/ollama/ollama/mlx/mlxthread/mlxthreadtest"
)

// requireGPUKernel skips where the kernel path and the reference would both
// be the same graph ops.
func requireGPUKernel(t *mlxthreadtest.T) {
	t.Helper()
	if !MetalIsAvailable() && !CUDAIsAvailable() {
		t.Skip("no GPU custom-kernel backend available")
	}
}

// patternArray builds a deterministic value lattice for kernel parity tests.
func patternArray(dtype DType, shape []int, bias, scale float32, stride, modulus int) *Array {
	size := 1
	for _, dim := range shape {
		size *= dim
	}
	values := make([]float32, size)
	center := modulus / 2
	for i := range values {
		values[i] = bias + float32((i*stride)%modulus-center)*scale
	}
	return FromValues(values, shape...).AsType(dtype)
}

// Reduction order differs from the graph path, so bf16 y can land one ULP (6.1e-5) away.
const gatedDeltaGraphTol = 2e-4

// requireClose compares two arrays within tol after widening to float32.
// NaN never compares close, so a kernel that produces NaN fails.
func requireClose(label string, got, want *Array, tol float64) error {
	got32, want32 := got.AsType(DTypeFloat32), want.AsType(DTypeFloat32)
	Eval(got32, want32)
	gotValues, wantValues := got32.Floats(), want32.Floats()
	if len(gotValues) != len(wantValues) {
		return fmt.Errorf("%s length = %d, want %d", label, len(gotValues), len(wantValues))
	}
	for i := range wantValues {
		if diff := math.Abs(float64(gotValues[i] - wantValues[i])); !(diff <= tol) {
			return fmt.Errorf("%s[%d] = %v, want %v (differs by %g, tolerance %g)",
				label, i, gotValues[i], wantValues[i], diff, tol)
		}
	}
	return nil
}

// requireRelative compares arrays of the same shape within rel times max|want|,
// reading views in logical order. NaN in either array fails.
func requireRelative(label string, got, want *Array, rel float64) error {
	if !slices.Equal(got.Dims(), want.Dims()) {
		return fmt.Errorf("%s shape = %v, want %v", label, got.Dims(), want.Dims())
	}
	got, want = Contiguous(got.AsType(DTypeFloat32), false), Contiguous(want.AsType(DTypeFloat32), false)
	scale := 0.0
	for _, v := range want.Floats() {
		scale = max(scale, math.Abs(float64(v))) // max propagates NaN
	}
	return requireClose(label, got, want, rel*scale)
}

// requireExact compares two arrays bit-for-bit after widening to float32.
func requireExact(label string, got, want *Array) error {
	got32, want32 := got.AsType(DTypeFloat32), want.AsType(DTypeFloat32)
	Eval(got32, want32)
	gotValues, wantValues := got32.Floats(), want32.Floats()
	if len(gotValues) != len(wantValues) {
		return fmt.Errorf("%s length = %d, want %d", label, len(gotValues), len(wantValues))
	}
	for i := range wantValues {
		if math.Float32bits(gotValues[i]) != math.Float32bits(wantValues[i]) {
			return fmt.Errorf("%s[%d] = %v, want %v", label, i, gotValues[i], wantValues[i])
		}
	}
	return nil
}

func TestRequireRelative(t *testing.T) {
	withMLXThread(t, func(t *mlxthreadtest.T) {
		nan := float32(math.NaN())
		want := FromValues([]float32{1, -2, 3, 4}, 2, 2)
		for _, tc := range []struct {
			name string
			got  *Array
		}{
			{"equal", want},
			{"transposed_view", FromValues([]float32{1, 3, -2, 4}, 2, 2).Transpose(1, 0)},
			{"within", FromValues([]float32{1, -2, 3, 4.03}, 2, 2)},
		} {
			if err := requireRelative(tc.name, tc.got, want, 0.01); err != nil {
				t.Errorf("rejected: %v", err)
			}
		}
		for _, tc := range []struct {
			name      string
			got, want *Array
		}{
			{"nan_got", FromValues([]float32{1, nan, 3, 4}, 2, 2), want},
			{"nan_want", want, FromValues([]float32{1, nan, 3, 4}, 2, 2)},
			{"broadcast", FromValues([]float32{1, -2}, 2), want},
			{"shape", FromValues([]float32{1, -2, 3, 4}, 4), want},
			{"beyond", FromValues([]float32{1, -2, 3, 4.05}, 2, 2), want},
		} {
			if requireRelative(tc.name, tc.got, tc.want, 0.01) == nil {
				t.Errorf("%s: accepted", tc.name)
			}
		}
	})
}
