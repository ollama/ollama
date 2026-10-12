package mlx

import (
	"fmt"
	"math"
)

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
