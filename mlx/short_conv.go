package mlx

import (
	"math"
	"slices"
)

// ShortConvolution applies a causal depthwise convolution with a residual to
// x [rows, tokens, channels]. state [rows, taps-1, channels] holds the inputs
// before x. weight [taps, channels] lists taps oldest first, or by lag for a
// dynamic convolution, which adds per-token taps [rows, tokens, heads*taps].
// Each output is the FP32 sum of the tap products and the input, times scale,
// in x's dtype. next holds the final taps-1 inputs.
// It returns nils when the Metal kernel cannot run.
func ShortConvolution(state, x, weight, dynamic *Array, heads int, scale float32) (y, next *Array) {
	return shortConvolutionLaunch(state, x, weight, dynamic, nil, heads, scale, 0)
}

// ShortConvolutionAdd is ShortConvolution with fixed taps whose output, rounded
// to x's dtype, is added in FP32 to residual [rows, tokens, channels]. The sum
// rounds to dtype.
func ShortConvolutionAdd(state, x, weight, residual *Array, scale float32, dtype DType) (sum, next *Array) {
	if residual == nil || x == nil || !slices.Equal(residual.Dims(), x.Dims()) ||
		(residual.DType() != DTypeBFloat16 && residual.DType() != DTypeFloat32) ||
		(dtype != DTypeBFloat16 && dtype != DTypeFloat32) {
		return nil, nil
	}
	return shortConvolutionLaunch(state, x, weight, nil, residual, 1, scale, dtype)
}

func shortConvolutionLaunch(state, x, weight, dynamic, residual *Array, heads int, scale float32, dtype DType) (y, next *Array) {
	if !MetalIsAvailable() || state == nil || x == nil || weight == nil || x.NumDims() != 3 ||
		state.NumDims() != 3 || weight.NumDims() != 2 || state.DType() != x.DType() {
		return nil, nil
	}
	rows, tokens, channels := x.Dim(0), x.Dim(1), x.Dim(2)
	taps := weight.Dim(0)
	tail := taps - 1
	if tail < 1 || weight.Dim(1) != channels || state.Dim(0) != rows || state.Dim(1) != tail ||
		state.Dim(2) != channels || uint64(x.Size()) > math.MaxInt32 {
		return nil, nil
	}
	isDynamic := 0
	if dynamic != nil {
		// Both variants index dynamic as [rows, tokens, heads*taps].
		if heads < 1 || channels%heads != 0 || dynamic.NumDims() != 3 ||
			dynamic.Dim(0) != rows || dynamic.Dim(1) != tokens || dynamic.Dim(2) != heads*taps {
			return nil, nil
		}
		isDynamic = 1
	} else {
		// Fixed taps bind weight in the dynamic slot, which is never read.
		dynamic, heads = weight, 1
	}
	hasResidual, out := 0, x.DType()
	if residual != nil {
		hasResidual, out = 1, dtype
	} else {
		// Without a residual, x fills its slot and is not read there.
		residual = x
	}
	result, ok := shortConvolutionConfig(rows, tokens).applyMetal(gpuLaunch{
		inputs: []*Array{state, x, weight, dynamic, residual},
		dtypes: []gpuDTypeArg{{"T", x.DType()}, {"OutT", out}},
		ints: []gpuIntArg{
			{"Channels", channels},
			{"Taps", taps},
			{"Heads", heads},
			{"Dynamic", isDynamic},
			{"ScaleBits", int(math.Float32bits(scale))},
			{"HasResidual", hasResidual},
		},
		outputs: []gpuOutputSpec{
			{"short_conv", []int32{int32(rows), int32(tokens), int32(channels)}, out},
			{"short_conv_state", []int32{int32(rows), int32(tail), int32(channels)}, x.DType()},
		},
		grid:        [3]int{channels, max(tokens, tail), rows},
		threadGroup: [3]int{min(channels, 256), 1, 1},
	})
	if !ok {
		return nil, nil
	}
	return result[0], result[1]
}

var (
	shortConvolution        = newShortConvolution(false)
	shortConvolutionStrided = newShortConvolution(true)
)

// shortConvolutionConfig returns the kernel variant for rows*tokens positions.
func shortConvolutionConfig(rows, tokens int) *gpuKernel {
	if positions := rows * tokens; positions > 1 && positions <= 16 {
		// Decode positions may be column views of a wider product.
		return shortConvolutionStrided
	}
	// One position of a view is contiguous, and the strided bindings measured
	// slower at n1. Prefill reads copies: strided indexing cost 0.8% of 8K
	// prefill on M5 Max.
	return shortConvolution
}

// newShortConvolution returns the kernel for row-contiguous inputs, or one
// that reads every input in place through its strides.
func newShortConvolution(strided bool) *gpuKernel {
	// Index macros rather than lambdas: Metal before version 4 (macOS 14 and
	// 15) rejects lambdas.
	index := `
#define STATE_AT(i, j, k) ((int64_t(i) * Tail + int64_t(j)) * Channels + int64_t(k))
#define X_AT(i, j, k) ((int64_t(i) * tokens + int64_t(j)) * Channels + int64_t(k))
#define WEIGHT_AT(i, j) (int64_t(i) * Channels + int64_t(j))
#define DYNAMIC_AT(i, j, k) ((int64_t(i) * tokens + int64_t(j)) * (Heads * Taps) + int64_t(k))
#define RESIDUAL_AT(i, j, k) X_AT(i, j, k)
`
	name := "short_convolution"
	if strided {
		name += "_strided"
		index = `
#define STATE_AT(i, j, k) (int64_t(i) * state_strides[0] + int64_t(j) * state_strides[1] + int64_t(k) * state_strides[2])
#define X_AT(i, j, k) (int64_t(i) * x_strides[0] + int64_t(j) * x_strides[1] + int64_t(k) * x_strides[2])
#define WEIGHT_AT(i, j) (int64_t(i) * weight_strides[0] + int64_t(j) * weight_strides[1])
#define DYNAMIC_AT(i, j, k) (int64_t(i) * dynamic_strides[0] + int64_t(j) * dynamic_strides[1] + int64_t(k) * dynamic_strides[2])
#define RESIDUAL_AT(i, j, k) (int64_t(i) * residual_strides[0] + int64_t(j) * residual_strides[1] + int64_t(k) * residual_strides[2])
`
	}
	return &gpuKernel{
		name:    name,
		inputs:  []string{"state", "x", "weight", "dynamic", "residual"},
		outputs: []string{"y", "next"},
		strided: strided,
		metal: gpuSource{source: `
#pragma clang fp contract(off)
constexpr int Tail = Taps - 1;
const int c = thread_position_in_grid.x, t = thread_position_in_grid.y, r = thread_position_in_grid.z;
const int tokens = x_shape[1];
if (c >= Channels) return;
` + index + `
#define WINDOW(j) ((j) < Tail ? float(state[STATE_AT(r, (j), c)]) : float(x[X_AT(r, (j) - Tail, c)]))
if (t < Tail) next[(size_t(r) * Tail + t) * Channels + c] = T(WINDOW(tokens + t));
if (t >= tokens) return;
const int head = c / (Channels / Heads);
float sum = 0.0f;
for (int i = 0; i < Taps; ++i) {
    float coef = float(weight[WEIGHT_AT(i, c)]);
    float value;
    if (Dynamic) {
        coef = coef + float(dynamic[DYNAMIC_AT(r, t, head * Taps + i)]);
        value = WINDOW(t + Tail - i);
    } else {
        value = WINDOW(t + i);
    }
    float product = value * coef;
    sum = i == 0 ? product : sum + product;
}
sum = sum + float(x[X_AT(r, t, c)]);
T rounded = T(sum * as_type<float>(ScaleBits));
const size_t out = (size_t(r) * tokens + t) * Channels + c;
y[out] = HasResidual ? OutT(float(residual[RESIDUAL_AT(r, t, c)]) + float(rounded)) : OutT(rounded);
`},
	}
}
