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

// SoftplusF32 returns softplus(x) computed in float32 precision and cast back
// to x's original dtype, as a fused kernel. Matches the laguna attention
// output-gate formula: softplus(cast_f32(x)).cast(orig_dtype).
var SoftplusF32 = Compile1(
	"SoftplusF32",
	func(x *Array) *Array {
		dt := x.DType()
		zero := FromValue[float32](0)
		return Logaddexp(x.AsType(DTypeFloat32), zero).AsType(dt)
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

// sigmoidRouterFused traces the DeepSeek-V2 / GLM-MoE aux-loss-free router
// head. Two outputs are returned so the pre-bias sigmoid (used to gather
// per-expert scores after top-k) and the post-bias negation (used as the
// argpartition key for top-k) share a single kernel.
var sigmoidRouterFused = Compile(
	"SigmoidRouter",
	func(in ...*Array) []*Array {
		gates, bias := in[0], in[1]
		orig := gates.Sigmoid()
		neg := orig.Add(bias).Negative()
		return []*Array{orig, neg}
	},
	Shapeless(),
)

// SigmoidRouter returns (sigmoid(gates), -(sigmoid(gates)+bias)) as a fused
// kernel — the DeepSeek-V2 / GLM-MoE aux-loss-free router head.
func SigmoidRouter(gates, bias *Array) (origScores, negScores *Array) {
	out := sigmoidRouterFused(gates, bias)
	return out[0], out[1]
}
