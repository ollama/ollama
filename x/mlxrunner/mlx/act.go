package mlx

// #include "generated.h"
import "C"
import "math"

var geluCoeff = float32(math.Sqrt(2 / math.Pi))

// GELUApprox matches mlx.nn.gelu_approx:
//
//	0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
func GELUApprox(x *Array) *Array {
	// Use dtype-matched scalars to avoid implicit upcasts on bf16 inputs.
	half := scalarWithDtype(0.5, x)
	defer C.mlx_array_free(half)
	coeff := scalarWithDtype(geluCoeff, x)
	defer C.mlx_array_free(coeff)
	c := scalarWithDtype(0.044715, x)
	defer C.mlx_array_free(c)

	// x^3 via x*x*x (avoids general Power which is slower)
	x3 := New("GELU_X3")
	C.mlx_multiply(&x3.ctx, x.ctx, x.ctx, DefaultStream().ctx)
	tmp := New("GELU_X3b")
	C.mlx_multiply(&tmp.ctx, x3.ctx, x.ctx, DefaultStream().ctx)
	x3 = tmp

	// 0.044715 * x^3
	cx3 := New("GELU_CX3")
	C.mlx_multiply(&cx3.ctx, c, x3.ctx, DefaultStream().ctx)

	// x + 0.044715 * x^3
	inner := New("GELU_INNER")
	C.mlx_add(&inner.ctx, x.ctx, cx3.ctx, DefaultStream().ctx)

	// sqrt(2/pi) * (x + 0.044715 * x^3)
	scaled := New("GELU_SCALED")
	C.mlx_multiply(&scaled.ctx, coeff, inner.ctx, DefaultStream().ctx)

	// tanh(...)
	th := New("GELU_TANH")
	C.mlx_tanh(&th.ctx, scaled.ctx, DefaultStream().ctx)

	// 1 + tanh(...)
	one := scalarWithDtype(1.0, x)
	defer C.mlx_array_free(one)
	onePlusTanh := New("GELU_1PT")
	C.mlx_add(&onePlusTanh.ctx, one, th.ctx, DefaultStream().ctx)

	// 0.5 * x
	halfX := New("GELU_HALFX")
	C.mlx_multiply(&halfX.ctx, half, x.ctx, DefaultStream().ctx)

	// 0.5 * x * (1 + tanh(...))
	out := New("GELU_APPROX")
	C.mlx_multiply(&out.ctx, halfX.ctx, onePlusTanh.ctx, DefaultStream().ctx)
	return out
}

func SILU(t *Array) *Array {
	return t.Multiply(t.Sigmoid()).AsType(t.DType())
}
