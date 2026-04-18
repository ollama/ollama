package mlx

import "math"

var geluCoeff = float32(math.Sqrt(2 / math.Pi))

// GELUApprox returns 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
// as a fused kernel.
var GELUApprox = Compile1(
	"GELUApprox",
	func(x *Array) *Array {
		// Dtype-matched scalars avoid implicit upcasts on bf16 inputs.
		dt := x.DType()
		half := FromValue[float32](0.5).AsType(dt)
		coeff := FromValue(geluCoeff).AsType(dt)
		c := FromValue[float32](0.044715).AsType(dt)
		one := FromValue[float32](1.0).AsType(dt)

		// x^3 via x*x*x (avoids general Power which is slower).
		x3 := x.Multiply(x).Multiply(x)
		inner := x.Add(c.Multiply(x3))
		tanh := coeff.Multiply(inner).Tanh()
		return half.Multiply(x).Multiply(one.Add(tanh))
	},
	Shapeless(),
)

// SiLU returns a * sigmoid(a) as a fused kernel.
var SiLU = Compile1(
	"SiLU",
	func(a *Array) *Array {
		return a.Multiply(a.Sigmoid())
	},
	Shapeless(),
)

// SwiGLU returns silu(gate) * up as a fused kernel.
var SwiGLU = Compile2(
	"SwiGLU",
	func(gate, up *Array) *Array {
		return SiLU(gate).Multiply(up)
	},
	Shapeless(),
)

// GeGLU returns gelu_approx(gate) * up as a fused kernel. Matches mlx_lm's
// geglu, used by Gemma-family MLP and MoE paths.
var GeGLU = Compile2(
	"GeGLU",
	func(gate, up *Array) *Array {
		return GELUApprox(gate).Multiply(up)
	},
	Shapeless(),
)

// LogitSoftcap returns tanh(x / cap) * cap as a fused kernel. Matches
// mlx_lm's logit_softcap. cap must have the same dtype as x.
var LogitSoftcap = Compile2(
	"LogitSoftcap",
	func(x, cap *Array) *Array {
		return x.Divide(cap).Tanh().Multiply(cap)
	},
	Shapeless(),
)
