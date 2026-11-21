package mlx

import "math"

// ShortConvolution applies a causal depthwise convolution with a residual to
// x [rows, tokens, channels]. state [rows, taps-1, channels] holds the inputs
// before x. weight [taps, channels] lists taps oldest first, or by lag for a
// dynamic convolution, which adds per-token taps [rows, tokens, heads*taps].
// Each output is the FP32 sum of the tap products and the input, times scale,
// in x's dtype. next holds the final taps-1 inputs.
// It returns nils when the Metal kernel cannot run.
func ShortConvolution(state, x, weight, dynamic *Array, heads int, scale float32) (y, next *Array) {
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
		if heads < 1 || channels%heads != 0 || dynamic.Size() != rows*tokens*heads*taps {
			return nil, nil
		}
		isDynamic = 1
	} else {
		dynamic, heads = weight, 1
	}
	out, ok := shortConvolution.applyMetal(gpuLaunch{
		inputs: []*Array{state, x, weight, dynamic},
		dtypes: []gpuDTypeArg{{"T", x.DType()}},
		ints: []gpuIntArg{{"Channels", channels}, {"Taps", taps}, {"Heads", heads},
			{"Dynamic", isDynamic}, {"ScaleBits", int(math.Float32bits(scale))}},
		outputs: []gpuOutputSpec{
			{"short_conv", []int32{int32(rows), int32(tokens), int32(channels)}, x.DType()},
			{"short_conv_state", []int32{int32(rows), int32(tail), int32(channels)}, x.DType()},
		},
		grid:        [3]int{channels, max(tokens, tail), rows},
		threadGroup: [3]int{min(channels, 256), 1, 1},
	})
	if !ok {
		return nil, nil
	}
	return out[0], out[1]
}

var shortConvolution = &gpuKernel{
	name:    "short_convolution",
	inputs:  []string{"state", "x", "weight", "dynamic"},
	outputs: []string{"y", "next"},
	metal: gpuSource{source: `
#pragma clang fp contract(off)
constexpr int Tail = Taps - 1;
const int c = thread_position_in_grid.x, t = thread_position_in_grid.y, r = thread_position_in_grid.z;
const int tokens = x_shape[1];
if (c >= Channels) return;
auto window = [&](int j) -> float {
    return j < Tail ? float(state[(size_t(r) * Tail + j) * Channels + c])
                    : float(x[(size_t(r) * tokens + j - Tail) * Channels + c]);
};
if (t < Tail) next[(size_t(r) * Tail + t) * Channels + c] = T(window(tokens + t));
if (t >= tokens) return;
const int head = c / (Channels / Heads);
float sum = 0.0f;
for (int i = 0; i < Taps; ++i) {
    float coef = float(weight[i * Channels + c]);
    float value;
    if (Dynamic) {
        coef = coef + float(dynamic[((size_t(r) * tokens + t) * Heads + head) * Taps + i]);
        value = window(t + Tail - i);
    } else {
        value = window(t + i);
    }
    float product = value * coef;
    sum = i == 0 ? product : sum + product;
}
sum = sum + float(x[(size_t(r) * tokens + t) * Channels + c]);
y[(size_t(r) * tokens + t) * Channels + c] = T(sum * as_type<float>(ScaleBits));
`},
}
