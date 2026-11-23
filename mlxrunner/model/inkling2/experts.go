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

// project multiplies x by the selected experts without the Metal epilogues.
func (e expertBank) project(x, indices *mlx.Array) *mlx.Array {
	if e.BlockScales == nil {
		return mlx.GatherMM(x, e.Weight.Transpose(0, 2, 1), nil, indices, false)
	}
	if e.Maxima == nil {
		return e.gather(mxfp8ExpertInput(x), indices, false).AsType(x.DType())
	}
	values, activationMax := mlx.QuantizeNVFP4Activation4Over6(x, nvfp4ActivationDenominator)
	outerScale := nvfp4OuterScale(activationMax, e.Maxima, e.Denominator, indices)
	return mlx.Mul(e.gather(values, indices, false), outerScale).AsType(x.DType())
}

// gather multiplies quantized activations by the selected experts.
func (e expertBank) gather(values, indices *mlx.Array, sorted bool) *mlx.Array {
	mode, group, bits := "mxfp8", 32, 8
	if e.Maxima != nil {
		mode, group, bits = "nvfp4", 16, 4
	}
	// Rows of a short forward often share experts; read each one once.
	if !sorted {
		if out := mlx.ShortGatherQMM(values, e.Weight, e.BlockScales, indices, mode); out != nil {
			return out
		}
	}
	return mlx.GatherQMM(values, e.Weight, e.BlockScales, nil, nil, indices, true, group, bits, mode, nil, sorted)
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
	if indices, weights := mlx.RouteSigmoidTopK(logits, bias, topK, scale); indices != nil {
		return indices, weights
	}
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
	indices, weights := routeExperts(e.routerLogits(x), e.Bias, e.topK, e.routeScale)
	// The Metal grouped gather reuses each expert when there are at least
	// four routes per expert on average. Keep short target verification rows
	// on the width-stable path even if multiple requests share a batch.
	grouped := e.GateUp.BlockScales != nil && e.Down.BlockScales != nil &&
		tokens >= 64 && batch*tokens*e.topK/e.Router.Dim(0) >= 4 && mlx.MetalIsAvailable()
	return e.combine(flat, indices, weights, grouped).Reshape(batch, tokens, width)
}

// routerLogits returns FP32 router scores, [batch*tokens, experts]. Short
// products read BF16 rows directly and accumulate in FP32. Without them, short
// blocks stay 3-D so projectTarget pads decode and draft verification to one
// matrix shape.
func (e routedExperts) routerLogits(x *mlx.Array) *mlx.Array {
	batch, tokens, width := x.Dim(0), x.Dim(1), x.Dim(2)
	router := nn.Linear{Weight: e.Router}
	input := x
	if tokens > shortTargetRows {
		input = x.Reshape(batch*tokens, width).AsType(mlx.DTypeFloat32)
	} else if !mlx.MetalIsAvailable() {
		input = x.AsType(mlx.DTypeFloat32)
	}
	return projectTarget(&router, input).Reshape(batch*tokens, e.Router.Dim(0))
}

// combine evaluates the selected experts and sums them for each token.
// Grouped routes are sorted by expert for the Metal path.
func (e routedExperts) combine(flat, indices, weights *mlx.Array, grouped bool) *mlx.Array {
	if out := e.combineQuantized(flat, indices, weights, grouped); out != nil {
		return out
	}
	weighted := e.gateUpInput(flat, indices, weights)
	down := e.Down.project(weighted, indices).Squeeze(2).AsType(mlx.DTypeFloat32)
	return mlx.Sum(down, 1, false).AsType(flat.DType())
}

// combineQuantized runs NVFP4 or MXFP8 experts with fused Metal epilogues:
// the gate/up scaling, SwiGLU and down quantization, then the down scaling
// and the sum over routes. It returns nil where those kernels cannot run.
func (e routedExperts) combineQuantized(flat, indices, weights *mlx.Array, grouped bool) *mlx.Array {
	if !mlx.MetalIsAvailable() || e.GateUp.BlockScales == nil || e.Down.BlockScales == nil ||
		(e.GateUp.Maxima == nil) != (e.Down.Maxima == nil) || flat.DType() != mlx.DTypeBFloat16 {
		return nil
	}
	nvfp4 := e.GateUp.Maxima != nil
	rows, width := flat.Dim(0), flat.Dim(1)
	routes := rows * e.topK
	// Grouped products use BF16 tiles. Quantized activations are exact in BF16.
	dtype := mlx.DTypeFloat32
	if grouped {
		dtype = mlx.DTypeBFloat16
	}
	x := flat.ExpandDims(1).ExpandDims(1)
	var values, xmax *mlx.Array
	if nvfp4 {
		values, xmax = mlx.QuantizeNVFP4Activation4Over6(x, nvfp4ActivationDenominator)
		values = values.AsType(dtype)
	} else if values = mlx.MXFP8Quantize(x, dtype); values == nil {
		return nil
	}
	xmaxStride := e.topK
	var inverse *mlx.Array
	if grouped {
		flatIndices := indices.Reshape(-1)
		order := mlx.Argsort(flatIndices, 0)
		inverse = mlx.Argsort(order, 0)
		inputRows := mlx.FloorDivideScalar(order, int32(e.topK))
		indices = mlx.Take(flatIndices, order, 0).Reshape(-1, 1)
		weights = mlx.Take(weights.Reshape(-1), order, 0)
		values = mlx.Take(values, inputRows, 0)
		if nvfp4 {
			xmax = mlx.Take(xmax, inputRows, 0)
		}
		xmaxStride = 1
	}
	var down, downMax *mlx.Array
	if nvfp4 && !grouped {
		// The short gate/up gather finishes the SwiGLU in its epilogue, so
		// only the down quantizer remains.
		if activated := mlx.ShortGatherNVFP4SwiGLU(values, e.GateUp.Weight, e.GateUp.BlockScales, indices, xmax,
			e.GateUp.Maxima, e.GateUp.Denominator, weights, xmaxStride, e.gateCap, e.upCap,
			nvfp4ActivationDenominator); activated != nil {
			down, downMax = mlx.QuantizeNVFP4Activation4Over6(activated, nvfp4ActivationDenominator)
		}
	}
	if down == nil {
		inner := e.GateUp.gather(values, indices, grouped).Reshape(routes, -1)
		if nvfp4 {
			down, downMax = mlx.NVFP4SwiGLUQuantize(inner, xmax, e.GateUp.Maxima, e.GateUp.Denominator,
				indices, weights, xmaxStride, e.gateCap, e.upCap, nvfp4ActivationDenominator, dtype)
		} else {
			down = mlx.MXFP8SwiGLUQuantize(inner, weights, e.gateCap, e.upCap, dtype)
		}
		if down == nil {
			return nil
		}
	}
	if grouped {
		down = down.Reshape(routes, 1, 1, -1)
	} else {
		down = down.Reshape(rows, e.topK, 1, -1)
	}
	inner := e.Down.gather(down, indices, grouped).Reshape(routes, width)
	return mlx.ExpertSum(inner, downMax, e.Down.Maxima, e.Down.Denominator, indices, inverse, e.topK, nvfp4ActivationDenominator)
}

func (e routedExperts) gateUpInput(flat, indices, scores *mlx.Array) *mlx.Array {
	paired := e.GateUp.project(flat.ExpandDims(1).ExpandDims(1), indices).AsType(mlx.DTypeFloat32)
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

func softcap(x *mlx.Array, cap float32) *mlx.Array {
	if cap <= 0 {
		return x
	}
	return mlx.MulScalar(mlx.DivScalar(x, cap).Tanh(), cap)
}
