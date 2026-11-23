package inkling2

import (
	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlxrunner/nn"
)

// Inkling's SGLang reference sets the outer scale to the row maximum magnitude
// divided by 1536. Each block uses the 4- or 6-range scale with the smaller L1 error.
const nvfp4ActivationDenominator = 1536

type expertBank struct {
	Weight      *mlx.Array
	BlockScales *mlx.Array
	Maxima      *mlx.Array
	Denominator *mlx.Array
}

// inputRows gathers the gate/up activations after quantization, so each
// token's activation is quantized once even when several experts use it.
func (e expertBank) project(x, indices, inputRows *mlx.Array, grouped bool) *mlx.Array {
	if e.BlockScales == nil {
		return mlx.GatherMM(x, e.Weight.Transpose(0, 2, 1), nil, indices, false)
	}
	if e.Maxima == nil {
		values := mxfp8ExpertInput(x)
		result := mlx.GatherQMM(values, e.Weight, e.BlockScales, nil, nil, indices, true, 32, 8, "mxfp8", nil, false)
		return result.AsType(x.DType())
	}
	result, activationMax := e.nvfp4Inner(x, indices, inputRows, grouped)
	outerScale := nvfp4OuterScale(activationMax, e.Maxima, e.Denominator, indices)
	return mlx.Mul(result, outerScale).AsType(x.DType())
}

func (e expertBank) nvfp4Inner(x, indices, inputRows *mlx.Array, grouped bool) (*mlx.Array, *mlx.Array) {
	values, activationMax := mlx.QuantizeNVFP4Activation4Over6(x, nvfp4ActivationDenominator)
	if inputRows != nil {
		values = mlx.Take(values, inputRows, 0)
		activationMax = mlx.Take(activationMax, inputRows, 0)
	}
	if !grouped && inputRows == nil && indices.Dim(0) == 4 {
		if result := mlx.GatherNVFP4QMV(values, e.Weight, e.BlockScales, indices); result != nil {
			return result, activationMax
		}
	}
	if grouped && inputRows != nil && indices.Dim(0) >= 512 && mlx.PrefersCompactNVFP4QMM() {
		// Each reconstructed E2M1 value times its E4M3 block scale fits exactly in BF16.
		if result := mlx.GatherNVFP4QMM(values.AsType(mlx.DTypeBFloat16), e.Weight, e.BlockScales, indices); result != nil {
			return result, activationMax
		}
	}
	result := mlx.GatherQMM(values, e.Weight, e.BlockScales, nil, nil, indices, true, 16, 4, "nvfp4", nil, grouped)
	return result, activationMax
}

func nvfp4OuterScale(activationMax, expertMaxima, denominator, indices *mlx.Array) *mlx.Array {
	weightMax := mlx.Take(expertMaxima, indices, 0).ExpandDims(-1).ExpandDims(-1)
	return mlx.Div(mlx.Mul(activationMax, weightMax), mlx.MulScalar(denominator, nvfp4ActivationDenominator))
}

func mxfp8ExpertInput(x *mlx.Array) *mlx.Array {
	shape := x.Dims()
	width := shape[len(shape)-1]
	blocks := x.AsType(mlx.DTypeFloat32).Reshape(-1, width/32, 32)
	rawScale := mlx.MulScalar(blocks.Abs().MaxAxis(-1, true), 1.0/448)
	// MX scales are powers of two. Round the exponent up when the
	// FP32 significand has a fractional part.
	bits := rawScale.View(mlx.DTypeInt32)
	exponent := bits.RightShift(mlx.FromValue(23))
	fraction := bits.BitwiseAnd(mlx.FromValue((1 << 23) - 1))
	exponent = exponent.Add(fraction.Greater(mlx.FromValue(0)).AsType(mlx.DTypeInt32))
	exponent = mlx.Minimum(exponent, mlx.FromValue(254))
	minimum := mlx.FromValue(float32(0x1p-127))
	exponent = mlx.Where(rawScale.Greater(minimum), exponent, mlx.FromValue(0))
	encoded := exponent.Multiply(mlx.FromValue(1 << 23)).View(mlx.DTypeFloat32)
	scale := mlx.Maximum(encoded, minimum)
	inverse := mlx.Where(exponent.Equal(mlx.FromValue(0)), mlx.FromValue(float32(0x1p127)), mlx.Div(mlx.FromValue(float32(1)), scale))
	scaled := mlx.Clip(mlx.Mul(blocks, inverse), mlx.FromValue(float32(-448)), mlx.FromValue(float32(448)))
	return mlx.Mul(mlx.FromFP8(mlx.ToFP8(scaled), mlx.DTypeFloat32), scale).Reshape(shape...)
}

func routeExperts(logits, bias *mlx.Array, topK int, scale float32) (indices, weights *mlx.Array) {
	selection := logits.Sigmoid()
	if bias != nil {
		selection = selection.Add(bias)
	}
	indices = mlx.Argpartition(mlx.Neg(selection), topK-1, -1).Slice(mlx.Slice(), mlx.Slice(0, topK))
	chosen := logits.TakeAlongAxis(indices, -1)
	logWeights := mlx.Neg(mlx.Softplus(mlx.Neg(chosen)))
	return indices, mlx.MulScalar(mlx.SoftmaxAxis(logWeights, -1, true), scale)
}

type routedExperts struct {
	Router     *mlx.Array
	Bias       *mlx.Array
	GateUp     expertBank
	Down       expertBank
	topK       int
	routeScale float32
	gateCap    float32
	upCap      float32
}

func (e routedExperts) forward(x *mlx.Array) *mlx.Array {
	batch, tokens, width := x.Dim(0), x.Dim(1), x.Dim(2)
	flat := x.Reshape(batch*tokens, width)
	router := nn.Linear{Weight: e.Router.AsType(mlx.DTypeFloat32)}
	routerInput := flat.AsType(mlx.DTypeFloat32)
	if tokens <= shortTargetRows {
		routerInput = routerInput.Reshape(batch, tokens, width)
	}
	logits := projectTarget(&router, routerInput).Reshape(batch*tokens, e.Router.Dim(0))
	indices, weights := routeExperts(logits, e.Bias, e.topK, e.routeScale)
	// The Metal grouped gather reuses each expert when there are at least
	// four routes per expert on average. Keep short target verification rows
	// on the width-stable path even if multiple requests share a batch.
	grouped := e.GateUp.Maxima != nil && e.Down.Maxima != nil &&
		tokens >= 64 && batch*tokens*e.topK/e.Router.Dim(0) >= 4 && mlx.MetalIsAvailable()
	return e.combine(flat, indices, weights, grouped).Reshape(batch, tokens, width)
}

// combine evaluates the selected experts and restores their routing order
// before accumulating the result for each token.
func (e routedExperts) combine(flat, indices, weights *mlx.Array, grouped bool) *mlx.Array {
	rows, width := flat.Dim(0), flat.Dim(1)
	var inputRows, inverse *mlx.Array
	if grouped {
		flatIndices := indices.Reshape(-1)
		order := mlx.Argsort(flatIndices, 0)
		inverse = mlx.Argsort(order, 0)
		inputRows = mlx.FloorDivideScalar(order, int32(e.topK))
		indices = mlx.Take(flatIndices, order, 0).Reshape(-1, 1)
		weights = mlx.Take(weights.Reshape(-1), order, 0).Reshape(-1, 1)
	}
	weighted := e.gateUpInput(flat, indices, inputRows, weights, grouped)
	down := e.Down.project(weighted, indices, nil, grouped).Squeeze(2).AsType(mlx.DTypeFloat32)
	if grouped {
		down = mlx.Take(down.Reshape(-1, width), inverse, 0).Reshape(rows, e.topK, width)
	}
	return mlx.Sum(down, 1, false).AsType(flat.DType())
}

// Fuse the checkpoint's NVFP4 outer scale, projection rounding and SwiGLU
// pointwise operations after the Metal gather.
func (e routedExperts) gateUpInput(flat, indices, inputRows, scores *mlx.Array, grouped bool) *mlx.Array {
	x := flat.ExpandDims(1).ExpandDims(1)
	if mlx.MetalIsAvailable() && flat.DType() == mlx.DTypeBFloat16 && flat.Dim(1) == 2304 &&
		e.GateUp.BlockScales != nil && e.GateUp.Maxima != nil && e.GateUp.Weight.Dim(0) == 128 && e.GateUp.Weight.Dim(1) == 2304 &&
		e.topK == 8 && e.gateCap == 4 && e.upCap == 25 {
		inner, xmax := e.GateUp.nvfp4Inner(x, indices, inputRows, grouped)
		gate, up := splitGateUp(inner)
		return compiledGateUpEpilogue(gate, up, xmax, e.GateUp.Maxima, e.GateUp.Denominator, indices, scores)[0]
	}
	paired := e.GateUp.project(x, indices, inputRows, grouped).AsType(mlx.DTypeFloat32)
	gate, up := splitGateUp(paired)
	return weightedSwiGLU(gate, up, scores, e.gateCap, e.upCap, flat.DType())
}

func splitGateUp(paired *mlx.Array) (gate, up *mlx.Array) {
	end := paired.Dim(-1)
	return paired.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(), mlx.Slice(0, end, 2)),
		paired.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(), mlx.Slice(1, end, 2))
}

// Routing weights apply before the down projection and its quantizer.
func weightedSwiGLU(gate, up, scores *mlx.Array, gateCap, upCap float32, dtype mlx.DType) *mlx.Array {
	activated := mlx.Mul(mlx.Mul(softcap(gate, gateCap), gate.Sigmoid()), softcap(up, upCap))
	return mlx.Mul(activated, scores.ExpandDims(-1).ExpandDims(-1)).AsType(dtype)
}

var compiledGateUpEpilogue = mlx.Compile("InklingGateUpEpilogue", func(inputs ...*mlx.Array) []*mlx.Array {
	gate, up := inputs[0], inputs[1]
	xmax, wmax, denom, ids, scores := inputs[2], inputs[3], inputs[4], inputs[5], inputs[6]
	outer := nvfp4OuterScale(xmax, wmax, denom, ids)
	// Each projection rounds to BF16 before either nonlinearity.
	gate = mlx.Mul(gate, outer).AsType(mlx.DTypeBFloat16).AsType(mlx.DTypeFloat32)
	up = mlx.Mul(up, outer).AsType(mlx.DTypeBFloat16).AsType(mlx.DTypeFloat32)
	return []*mlx.Array{weightedSwiGLU(gate, up, scores, 4, 25, mlx.DTypeBFloat16)}
}, mlx.Shapeless())

func softcap(x *mlx.Array, cap float32) *mlx.Array {
	if cap <= 0 {
		return x
	}
	return mlx.MulScalar(mlx.DivScalar(x, cap).Tanh(), cap)
}
