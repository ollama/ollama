//go:build mlx

package mlx

// #include "generated.h"
import "C"
import "math"

func GELUApprox(t *Array) *Array {
	return t.Multiply(
		FromValue[float32](0.5),
	).Multiply(
		t.Add(
			t.Power(FromValue[float32](3.0)).Multiply(FromValue[float32](0.044715)),
		).Multiply(
			FromValue(float32(math.Sqrt(2 / math.Pi))),
		).Tanh().Add(FromValue[float32](1.0)),
	).AsType(t.DType())
}

func SILU(t *Array) *Array {
	return t.Multiply(t.Sigmoid()).AsType(t.DType())
}
