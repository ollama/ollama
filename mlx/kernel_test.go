package mlx

import (
	"fmt"
	"math"

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
func requireClose(label string, got, want *Array, tol float64) error {
	got32, want32 := got.AsType(DTypeFloat32), want.AsType(DTypeFloat32)
	Eval(got32, want32)
	gotValues, wantValues := got32.Floats(), want32.Floats()
	if len(gotValues) != len(wantValues) {
		return fmt.Errorf("%s length = %d, want %d", label, len(gotValues), len(wantValues))
	}
	for i := range wantValues {
		if diff := math.Abs(float64(gotValues[i] - wantValues[i])); diff > tol {
			return fmt.Errorf("%s[%d] = %v, want %v (differs by %g, tolerance %g)",
				label, i, gotValues[i], wantValues[i], diff, tol)
		}
	}
	return nil
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
